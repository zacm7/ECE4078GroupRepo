import os
import sys
import argparse
import time
import math
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import pygame
from rrt_planner import RRTPlanner, make_obstacles_from_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
WEEK0506_DIR = os.path.join(REPO_ROOT, "Week05-06")

# Import Week05-06 operate.py (GUI + EKF + detector)
sys.path.insert(0, WEEK0506_DIR)
try:
    import operate as operate_mod  # type: ignore
    from operate import Operate    # type: ignore
except Exception:
    import importlib.util
    _op_file = os.path.join(WEEK0506_DIR, "operate.py")
    _spec = importlib.util.spec_from_file_location("operate", _op_file)
    operate_mod = importlib.util.module_from_spec(_spec)  # type: ignore
    assert _spec and _spec.loader
    _spec.loader.exec_module(operate_mod)  # type: ignore
    Operate = operate_mod.Operate  # type: ignore

# Helpers and planner (left imported; A* unused now but not harmful)
from map_utils import read_true_map_robust, load_search_list, print_target_fruits_pos
from astar_planning import plan_waypoints


def normalize_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def angle_diff(a: float, b: float) -> float:
    return normalize_angle(a - b)


class AutoOperateDynamic(Operate):
    """RRT to fixed targets with online obstacle discovery + robust hand-off logic."""

    def __init__(self, args, search_list: List[str], targets_xy: List[List[float]],
                 aruco_obstacles_xy: List[List[float]], grid_res: float,
                 robot_radius: float, safety_margin: float):
        super().__init__(args)

        # Always run detector continuously
        self.command['inference'] = True

        # Planning model
        self.search_list = [s.lower() for s in search_list]
        self.remaining_targets: List[List[float]] = [list(t) for t in targets_xy]
        self.remaining_labels: List[str] = [s.lower() for s in search_list]
        self.known_obstacles: List[List[float]] = [list(o) for o in aruco_obstacles_xy]
        self.discovered_obstacles: List[List[float]] = []

        # Params
        self.grid_res = grid_res
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin

        # Controller state
        self.waypoints: List[List[float]] = []
        self.current_goal: List[float] | None = None
        self.reached_time: float | None = None
        self.active = True

        # Tolerances/commands
        self.waypoint_tol = 0.04          # 4 cm to accept intermediate waypoints
        self.final_target_tol = 0.20      # 20 cm to accept final target (per your requirement)
        self.angle_tol = math.radians(8.0)
        self.turn_cmd = 1
        self.fwd_cmd = 1

        # Marker acquisition state
        self._scan_start = None
        self._scan_dir = 1
        self._creep_until = None
        self._planned_once = False

        # Detection handling
        self.last_obstacle_add_time = 0.0
        self.add_cooldown = 0.5  # seconds
        self.min_obs_separation = 0.15  # m

        # Visual alignment (kept; currently not used in the hold logic)
        self.aligning = False
        self.align_start_time = 0.0
        self.align_timeout = 5.0  # seconds
        self.center_tol_px = 15.0
        self.close_width_px = 110.0
        self._align_scan_dir = 1

        # Cache intrinsics
        self.K = getattr(self.ekf.robot, 'camera_matrix', None)
        self.cx = float(self.K[0, 2]) if self.K is not None else 160.0
        self.fx = float(self.K[0, 0]) if self.K is not None else 320.0

        # Fixed world bounds for drawing / planning (fits your map)
        self.fixed_bounds = ((-1.4136, 1.356), (-1.3272, 1.3368))

    # ============= Navigation primitives =============
    def get_pose(self) -> Tuple[float, float, float]:
        if hasattr(self, "ekf") and self.ekf is not None:
            robot = getattr(self.ekf, "robot", None)
            if robot is not None and hasattr(robot, "state") and robot.state.shape[0] >= 3:
                x = float(robot.state[0, 0])
                y = float(robot.state[1, 0])
                th = float(robot.state[2, 0])
                return x, y, th
        print("ERROR UNKNOWN POSITIONS")
        return 0.0, 0.0, 0.0

    def pick_next_goal(self):
        """Pop next waypoint into current_goal."""
        if not self.waypoints:
            self.current_goal = None
            return
        self.current_goal = self.waypoints.pop(0)
        self.reached_time = None
        self.notification = f'Navigating to: [{self.current_goal[0]:.2f}, {self.current_goal[1]:.2f}]'

    def auto_nav_step(self):
        # Require SLAM
        if not self.ekf_on:
            self.command['motion'] = [0, 0]
            self.notification = 'Press ENTER to start SLAM'
            return

        # Marker acquisition gate (scan/creep) until >=2 tags visible
        tag_count = len(getattr(self.ekf, 'taglist', []))
        now = time.time()
        if tag_count < 2:
            if self._scan_start is None:
                self._scan_start = now
                self._scan_dir = 1
                self._creep_until = None
            if self._creep_until and now < self._creep_until:
                self.command['motion'] = [self.fwd_cmd, 0]
                self.notification = 'Looking for markers: creeping forward'
                return
            elapsed = now - self._scan_start
            if elapsed > 6.0:
                self._creep_until = now + 1.0
                self._scan_start = now
                self.command['motion'] = [self.fwd_cmd, 0]
                self.notification = 'Looking for markers: creeping forward'
                return
            self._scan_dir = 1 if int(elapsed // 2) % 2 == 0 else -1
            self.command['motion'] = [0, self._scan_dir * self.turn_cmd]
            self.notification = 'Looking for markers: scanning'
            return
        else:
            self._scan_start = None
            self._creep_until = None

        # Ensure we have a plan initially or after finishing a target
        if (not self._planned_once and self.active) or (self.active and not self.waypoints and self.remaining_targets):
            self.replan(initial=not self._planned_once)
            self._planned_once = True

        # Aim rule for this tick: prefer a waypoint; if none, aim at the current target (so the hold can fire)
        if self.current_goal is None and self.active:
            self.pick_next_goal()

        aiming_at_final = False
        if self.current_goal is not None:
            gx, gy = self.current_goal
        elif self.remaining_targets:
            gx, gy = self.remaining_targets[0]
            aiming_at_final = True
        else:
            self.command['motion'] = [0, 0]
            self.notification = 'All targets completed'
            return

        # Pose
        x, y, th = self.get_pose()
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        dheading = angle_diff(bearing, th)

        # --- ALWAYS check proximity to the actual target FIRST (20 cm) ---
        in_target_bubble = False
        if self.remaining_targets:
            tx, ty = self.remaining_targets[0]
            dist_to_target = math.hypot(tx - x, ty - y)
            in_target_bubble = (dist_to_target <= self.final_target_tol)

        if in_target_bubble:
            # Hold at target (2 s), then advance → replan → pick next waypoint
            self.command['motion'] = [0, 0]
            if self.reached_time is None:
                self.reached_time = time.time()
                self.notification = f'Reached target [{tx:.2f}, {ty:.2f}] (≤20cm). Holding...'
                return
            if time.time() - self.reached_time >= 2.0:
                self.notification = f'Completed target [{tx:.2f}, {ty:.2f}]'
                self._advance_target()
                self.reached_time = None
                if not self.remaining_targets:
                    self.current_goal = None
                    self.command['motion'] = [0, 0]
                    self.notification = 'All targets completed'
                    return
                self.replan(initial=False)
                self.current_goal = None
                self.pick_next_goal()
                return
            return  # still holding

        # --- Not inside target bubble: waypoint tracking (4 cm tol) ---
        if self.current_goal is not None and dist <= self.waypoint_tol:
            # Interior waypoint → advance to next waypoint
            self.pick_next_goal()
            return

        # --- Turn-then-drive controller ---
        if abs(dheading) > self.angle_tol:
            self.command['motion'] = [0, self.turn_cmd if dheading > 0 else -self.turn_cmd]
        else:
            self.command['motion'] = [self.fwd_cmd, 0]

    def _rrt_bounds_and_obstacles(self):
        """
        Build:
          - bounds: ((xmin,xmax),(ymin,ymax))
          - obstacles: [(x,y,r), ...] using centers from known + discovered obstacles
        """
        pts = []
        pts += [list(t) for t in self.remaining_targets]
        pts += [list(o) for o in self.known_obstacles]
        pts += [list(o) for o in self.discovered_obstacles]
        if not pts:
            pts = [[0.0, 0.0]]

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        margin = max(0.20, 5.0 * self.grid_res)
        inferred_bounds = ((min(xs) - margin, max(xs) + margin),
                           (min(ys) - margin, max(ys) + margin))

        # Build circular obstacles. Radii are conservative here; RRT goal tolerance handles final proximity.
        default_r = 0.10  # typical ArUco/fruit safety radius
        obstacles = []
        for (ox, oy) in (self.known_obstacles + self.discovered_obstacles):
            obstacles.append((float(ox), float(oy), default_r))

        # Prefer fixed bounds (your arena), but fall back to inferred if needed
        bounds = self.fixed_bounds if self.fixed_bounds else inferred_bounds
        return bounds, obstacles

    # ============= Planning =============
    def replan(self, initial: bool = False):
        """Plan waypoints from current pose to the NEXT target using RRT (goal tol = 20cm)."""
        if not self.active:
            return
        if not self.remaining_targets:
            self.waypoints = []
            self.current_goal = None
            self.active = False
            self.notification = 'All targets completed'
            return

        # Pose and goal
        x, y, _ = self.get_pose()
        start_xy = [float(x), float(y)]
        goal_xy  = [float(self.remaining_targets[0][0]), float(self.remaining_targets[0][1])]

        # Build RRT world
        bounds, obstacles = self._rrt_bounds_and_obstacles()

        # IMPORTANT: targets may also be obstacles; since we only need to be within 20cm of the target,
        # we ignore obstacles whose centers lie within that 20cm radius around the goal.
        gx, gy = goal_xy
        eps = 1e-6
        goal_clear_radius = self.final_target_tol + eps
        obstacles = [(ox, oy, r) for (ox, oy, r) in obstacles
                     if math.hypot(ox - gx, oy - gy) > goal_clear_radius]

        # RRT parameters
        step_size = max(0.02, 3.0 * self.grid_res)
        goal_sample_rate = 0.20
        max_iters = 2000
        goal_tol = self.final_target_tol

        try:
            rrt = RRTPlanner(bounds, obstacles, step_size=0.1,
                             goal_sample_rate=goal_sample_rate, max_iters=max_iters)
            path = rrt.plan(start_xy, goal_xy, goal_tol)
            if path is None or len(path) < 2:
                self.waypoints = []
                self.current_goal = None
                self.notification = 'RRT failed to find a path'
                return

            # Use the path as waypoints (skip the start element)
            self.waypoints = [[float(px), float(py)] for (px, py) in path[1:]]
            self.current_goal = None  # pick first on next control step
            self.notification = f'RRT planned {len(self.waypoints)} waypoints' + (' (initial)' if initial else '')

        except Exception as e:
            self.waypoints = []
            self.current_goal = None
            self.notification = f'RRT planning error: {e}'

    def _advance_target(self):
        if not self.remaining_targets:
            return
        self.remaining_targets.pop(0)
        if hasattr(self, 'remaining_labels') and self.remaining_labels:
            self.remaining_labels.pop(0)
        self.aligning = False
        self.reached_time = None
        if not self.remaining_targets:
            self.waypoints = []
            self.current_goal = None
            self.active = False
            self.notification = 'All targets completed'

    def _is_close_to_current_target(self, goal_xy: List[float]) -> bool:
        """Treat waypoint as 'at target' if within final-target tolerance (20cm)."""
        if not self.remaining_targets:
            return False
        tx, ty = self.remaining_targets[0]
        return math.hypot(goal_xy[0] - tx, goal_xy[1] - ty) <= self.final_target_tol

    # ============= Perception integration =============
    def periodic_perception_update(self):
        """Add unknown obstacles from detector; replan if something new is seen."""
        bboxes = getattr(self, 'detector_output', None)
        if not isinstance(bboxes, (list, tuple)) or len(bboxes) == 0:
            return
        now = time.time()
        new_added = False

        # Current pose and intrinsics
        x, y, th = self.get_pose()
        cx, fx = self.cx, self.fx

        # Copy remaining targets to avoid blocking them as obstacles
        known_targets = self.remaining_targets[:]

        for det in bboxes:
            try:
                label: str = str(det[0]).lower()
                xywh = np.asarray(det[1]).astype(float)
                conf = float(det[2])
            except Exception:
                continue
            if conf < 0.4:
                continue

            if label in self.search_list:
                continue  # don't treat targets as obstacles

            # Project detection center to rough world coordinates
            u = float(xywh[0])
            w_px = float(xywh[2])
            alpha = math.atan((u - cx) / fx)
            bearing = th + alpha
            W_assumed = 0.10
            if w_px <= 1.0:
                d = 0.5
            else:
                d = max(0.35, min(1.10, (fx * W_assumed) / w_px))
            ox = x + d * math.cos(bearing)
            oy = y + d * math.sin(bearing)

            # Ignore if too close to a known target position
            if any(math.hypot(ox - tx, oy - ty) <= 0.30 for tx, ty in known_targets):
                continue

            # Ignore duplicates amongst known and discovered obstacles
            all_obs = self.known_obstacles + self.discovered_obstacles
            if any(math.hypot(ox - qx, oy - qy) <= self.min_obs_separation for qx, qy in all_obs):
                continue

            # Rate limit
            if now - self.last_obstacle_add_time < self.add_cooldown:
                continue

            self.discovered_obstacles.append([ox, oy])
            self.last_obstacle_add_time = now
            new_added = True

        if new_added and self.ekf_on and self.active:
            self.replan(initial=False)

    # ============= Alignment helpers (kept, optional) =============
    def _current_target_label(self) -> str:
        try:
            return str(self.remaining_labels[0]).lower()
        except Exception:
            return ""

    def _find_detection_for_label(self, label: str):
        bboxes = getattr(self, 'detector_output', None)
        if not isinstance(bboxes, (list, tuple)) or len(bboxes) == 0:
            return None
        best = None
        best_conf = -1.0
        for det in bboxes:
            try:
                det_label = str(det[0]).lower()
                xywh = np.asarray(det[1]).astype(float)
                conf = float(det[2])
            except Exception:
                continue
            if det_label != label:
                continue
            if conf > best_conf:
                best_conf = conf
                best = (float(xywh[0]), float(xywh[2]), conf)
        return best  # (u, w_px, conf) or None

    def _align_to_target_step(self) -> bool:
        label = self._current_target_label()
        if not label:
            self.command['motion'] = [0, 0]
            return True
        found = self._find_detection_for_label(label)
        cx = self.cx
        if found is None:
            now = time.time()
            if int(now - self.align_start_time) % 2 == 0:
                self._align_scan_dir = 1
            else:
                self._align_scan_dir = -1
            self.command['motion'] = [0, self._align_scan_dir * self.turn_cmd]
            return False

        u, w_px, conf = found
        dx = u - cx
        if abs(dx) > self.center_tol_px:
            turn = self.turn_cmd if dx > 0 else -self.turn_cmd
            self.command['motion'] = [0, turn]
            return False
        if w_px < self.close_width_px:
            self.command['motion'] = [self.fwd_cmd, 0]
            return False
        self.command['motion'] = [0, 0]
        return True

    # ============= Drawing =============
    def draw(self, canvas):
        """Visualization: robot, obstacles, targets, waypoints, planned path."""
        canvas.fill((30, 30, 30))
        width, height = canvas.get_size()

        world_bounds = self.fixed_bounds
        wxmin, wxmax = world_bounds[0]
        wymin, wymax = world_bounds[1]

        def world_to_screen(x, y):
            sx = int((x - wxmin) / (wxmax - wxmin) * width)
            sy = int(height - (y - wymin) / (wymax - wymin) * height)
            return sx, sy

        # Obstacles
        for ox, oy in self.known_obstacles + self.discovered_obstacles:
            sx, sy = world_to_screen(ox, oy)
            pygame.draw.circle(canvas, (200, 50, 50), (sx, sy), 12)

        # Targets
        for i, t in enumerate(self.remaining_targets):
            sx, sy = world_to_screen(t[0], t[1])
            pygame.draw.circle(canvas, (50, 200, 50), (sx, sy), 14)
            pygame.draw.circle(canvas, (255, 255, 255), (sx, sy), 14, 2)
            if hasattr(self, 'remaining_labels') and i < len(self.remaining_labels):
                font = pygame.font.SysFont('Arial', 18)
                label = self.remaining_labels[i]
                txt = font.render(label, True, (255, 255, 255))
                canvas.blit(txt, (sx + 10, sy - 10))

        # Waypoints
        for wp in self.waypoints:
            sx, sy = world_to_screen(wp[0], wp[1])
            pygame.draw.circle(canvas, (255, 255, 0), (sx, sy), 7)

        # Planned path
        if self.waypoints:
            pts = [self.get_pose()[:2]] + self.waypoints
            for i in range(len(pts) - 1):
                sx1, sy1 = world_to_screen(pts[i][0], pts[i][1])
                sx2, sy2 = world_to_screen(pts[i + 1][0], pts[i + 1][1])
                pygame.draw.line(canvas, (0, 255, 255), (sx1, sy1), (sx2, sy2), 6)

        # Robot + heading
        x, y, th = self.get_pose()
        sx, sy = world_to_screen(x, y)
        pygame.draw.circle(canvas, (0, 255, 0), (sx, sy), 16)
        dx = int(24 * math.cos(th))
        dy = int(-24 * math.sin(th))
        pygame.draw.line(canvas, (255, 0, 0), (sx, sy), (sx + dx, sy + dy), 5)

        # Legend
        font = pygame.font.SysFont('Arial', 16)
        legend = [
            ('Robot', (0, 255, 0)),
            ('Heading', (255, 0, 0)),
            ('Obstacle', (200, 50, 50)),
            ('Target', (50, 200, 50)),
            ('Waypoint', (255, 255, 0)),
            ('Path', (0, 255, 255)),
        ]
        for i, (label, color) in enumerate(legend):
            pygame.draw.rect(canvas, color, (20, 40 + 22 * i, 18, 18))
            txt = font.render(label, True, (255, 255, 255))
            canvas.blit(txt, (45, 40 + 22 * i))

        # Notification
        font = pygame.font.SysFont('Arial', 24)
        txt = font.render(self.notification if hasattr(self, 'notification') else '', True, (255, 255, 255))
        canvas.blit(txt, (20, 20))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Level 3: RRT with online obstacle discovery + robust target hand-off")
    parser.add_argument("--ip", type=str, default="192.168.50.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default=os.path.join(REPO_ROOT, "Week02-04", "calibration", "param") + os.sep)
    parser.add_argument("--yolo_model", default=os.path.join(SCRIPT_DIR, "YOLO", "model", "bestv5.pt"))
    parser.add_argument("--map", type=str, default=os.path.join(SCRIPT_DIR, "M3_prac_map_part.txt"))
    parser.add_argument("--list", type=str, default=os.path.join(SCRIPT_DIR, "M3_prac_shopping_list.txt"))
    parser.add_argument("--grid_res", type=float, default=0.02)
    parser.add_argument("--robot_radius", type=float, default=0.12)
    parser.add_argument("--safety_margin", type=float, default=0.15)
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--save_data", action='store_true')
    args, _ = parser.parse_known_args()

    # Provide globals to Week05-06/operate.py expectations
    op_args = SimpleNamespace(
        ip=args.ip, port=args.port, calib_dir=args.calib_dir,
        yolo_model=args.yolo_model, play_data=args.play_data, save_data=args.save_data,
    )
    operate_mod.args = op_args

    # Fonts/icons for Operate GUI
    pygame.font.init()
    TITLE_FONT = pygame.font.Font(os.path.join(WEEK0506_DIR, 'pics', '8-BitMadness.ttf'), 35)
    TEXT_FONT = pygame.font.Font(os.path.join(WEEK0506_DIR, 'pics', '8-BitMadness.ttf'), 40)
    operate_mod.TITLE_FONT = TITLE_FONT
    operate_mod.TEXT_FONT = TEXT_FONT

    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 - Auto Fruit Search (L3)')
    pygame.display.set_icon(pygame.image.load(os.path.join(WEEK0506_DIR, 'pics', '8bit', 'pibot5.png')))
    canvas.fill((0, 0, 0))

    # ---- Use RRT obstacles + fixed goals instead of A* partial map ----
    from rrt_planner import RRTPlanner, make_obstacles_from_file

    # 1) Load circular obstacles (centers + radius) from your map file
    rrt_obstacles = make_obstacles_from_file(os.path.join(SCRIPT_DIR, "M3_prac_map_full.txt"),
                                             radius=0.05)   # [(x,y,r), ...]

    # 2) Fixed goal list (your sequence)
    goals = [[-0.8928, -0.4608], [0.5088, 0.8640], [-0.9888, 0.9792], [1.0560, -1.0080], [-1.0272, -1.0272]]

    # 3) Convert into the formats AutoOperateDynamic expects
    targets_xy: List[List[float]] = [list(g) for g in goals]
    search_list = [f"goal{i+1}" for i in range(len(goals))]
    aruco_obstacles_xy: List[List[float]] = [[float(x), float(y)] for (x, y, r) in rrt_obstacles]

    # Ensure Week05-06 relative asset paths in operate.py resolve
    try:
        os.chdir(WEEK0506_DIR)
    except Exception:
        pass

    # Create operator (planning will call RRT in replan())
    operate = AutoOperateDynamic(op_args, search_list, targets_xy, aruco_obstacles_xy,
                                 grid_res=args.grid_res,
                                 robot_radius=args.robot_radius,
                                 safety_margin=args.safety_margin)

    # Main loop
    running = True
    while running:
        operate.update_keyboard()
        operate.take_pic()
        operate.auto_nav_step()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        operate.detect_target()              # populates detector_output
        operate.periodic_perception_update() # add dynamic obstacles + replan if needed
        operate.draw(canvas)
        pygame.display.update()
