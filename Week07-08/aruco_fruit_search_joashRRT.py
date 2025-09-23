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

# Helpers and planner
from map_utils import read_true_map_robust, load_search_list, print_target_fruits_pos
from astar_planning import plan_waypoints


def normalize_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def angle_diff(a: float, b: float) -> float:
    return normalize_angle(a - b)


class AutoOperateDynamic(Operate):
    """Level 3: Partial map planning + online obstacle discovery and replanning.

    - Known: 10 ArUcos (obstacles) + target fruits (positions from partial map)
    - Unknown: non-target fruit positions (treated as obstacles once detected)
    - Strategy: Follow A* waypoints to targets in order; continuously run YOLO,
                project detections to world, add obstacles if new, and replan.
    """

    def __init__(self, args, search_list: List[str], targets_xy: List[List[float]],
                 aruco_obstacles_xy: List[List[float]], grid_res: float,
                 robot_radius: float, safety_margin: float):
        super().__init__(args)

        # Always run detector continuously
        self.command['inference'] = True

        # Planning model
        self.search_list = [s.lower() for s in search_list]
        self.remaining_targets: List[List[float]] = [list(t) for t in targets_xy]
        # Keep labels aligned with remaining_targets for alignment logic
        self.remaining_labels: List[str] = [s.lower() for s in search_list]
        self.known_obstacles: List[List[float]] = [list(o) for o in aruco_obstacles_xy]
        self.discovered_obstacles: List[List[float]] = []

        # A* params
        self.grid_res = grid_res
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin

        # Controller params (mirror Level 2)
        self.waypoints: List[List[float]] = []
        self.current_goal: List[float] | None = None
        self.reached_time: float | None = None
        self.active = True
        self.dist_tol = 0.1
        self.angle_tol = math.radians(8.0)
        self.turn_cmd = 1
        self.fwd_cmd = 1

        # Marker acquisition (scan/creep) state
        self._scan_start = None
        self._scan_dir = 1
        self._creep_until = None
        self._planned_once = False

        # Detection handling
        self.last_obstacle_add_time = 0.0
        self.add_cooldown = 0.5  # seconds
        self.min_obs_separation = 0.15  # m

        # Visual alignment near target (uses detector_output)
        self.aligning = False
        self.align_start_time = 0.0
        self.align_timeout = 5.0  # seconds
        self.center_tol_px = 15.0  # acceptable horizontal pixel error
        self.close_width_px = 110.0  # consider close enough when bbox is wide
        self._align_scan_dir = 1

        # Cache intrinsics
        self.K = getattr(self.ekf.robot, 'camera_matrix', None)
        self.cx = float(self.K[0, 2]) if self.K is not None else 160.0
        self.fx = float(self.K[0, 0]) if self.K is not None else 320.0

    # ============= Navigation primitives =============
    def get_pose(self) -> Tuple[float, float, float]:
        if hasattr(self, "ekf") and self.ekf is not None:
            robot = getattr(self.ekf, "robot", None)
            if robot is not None and hasattr(robot, "state") and robot.state.shape[0] >= 3:
                x = float(robot.state[0, 0])
                y = float(robot.state[1, 0])
                th = float(robot.state[2, 0])
                #print(f"Robot pose: x={x}, y={y}, th={th}")
                return x, y, th
        # Keep Level 2 behavior for consistency
        print("ERROR UNKNOWN POSITIONS")
        return 0.0, 0.0, 0.0

    def pick_next_goal(self): #uses self.waypoints and sets it as current_goal
        if not self.waypoints:
            self.current_goal = None
            return
        self.current_goal = self.waypoints.pop(0)
        self.reached_time = None
        self.notification = f'Navigating to: [{self.current_goal[0]:.2f}, {self.current_goal[1]:.2f}]'

    def auto_nav_step(self): #main loop for navigation
        # Require SLAM
        if not self.ekf_on:
            self.command['motion'] = [0, 0]
            self.notification = 'Press ENTER to start SLAM'
            return

        # Marker acquisition gate: scan and occasional creep until >=2 tags visible (same as L2)
        tag_count = len(getattr(self.ekf, 'taglist', []))
        now = time.time()
        if tag_count < 0:
            if self._scan_start is None:
                self._scan_start = now
                self._scan_dir = 1
                self._creep_until = None
            # Creep interval active
            if self._creep_until and now < self._creep_until:
                self.command['motion'] = [self.fwd_cmd, 0]
                self.notification = 'Looking for markers: creeping forward'
                return
            # After 6s of scanning, creep forward for 1s
            elapsed = now - self._scan_start
            if elapsed > 6.0:
                self._creep_until = now + 1.0
                self._scan_start = now
                self.command['motion'] = [self.fwd_cmd, 0]
                self.notification = 'Looking for markers: creeping forward'
                return
            # Alternate scan direction every ~2s
            self._scan_dir = 1 if int(elapsed // 2) % 2 == 0 else -1
            self.command['motion'] = [0, self._scan_dir * self.turn_cmd]
            self.notification = 'Looking for markers: scanning'
            return
        else:
            # Reset scanning state when we have enough tags
            self._scan_start = None
            self._creep_until = None

        # Ensure we have a plan from current pose to remaining targets
        if (not self._planned_once and self.active) or (self.active and not self.waypoints):
            self.replan(initial=not self._planned_once)
            self._planned_once = True

        # Get/set goal
        if self.current_goal is None and self.active:
            self.pick_next_goal()
        if not self.current_goal:
            self.command['motion'] = [0, 0]
            return

        # Control to goal (same policy as L2)
        x, y, th = self.get_pose()
        gx, gy = [float(self.remaining_targets[0][0]), float(self.remaining_targets[0][1])]#change
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        dheading = angle_diff(bearing, th)

        # Arrival handling: if this goal is close to the next target, align visually then hold; else skip hold
        if dist <= self.dist_tol:
            print(gx,gy)
            print("DISTANCE",dist)
            # if self._is_close_to_current_target([gx, gy]):
            #     # Start alignment state on first arrival
            #     if not self.aligning:
            #         self.aligning = True
            #         self.align_start_time = time.time()
            #         self._align_scan_dir = 1

            #     # Run alignment until centered/close or timeout
            #     if self.aligning and (time.time() - self.align_start_time) <= self.align_timeout:
            #         if self._align_to_target_step():
            #             self.aligning = False
            #         else:
            #             self.notification = 'Aligning to target...'
            #             return
            #     else:
            #         # Timeout or done
            #         self.aligning = False

            # Once aligned (or timed out), perform brief hold
            if self.reached_time is None:
                self.reached_time = time.time()
                self.notification = f'Reached target [{gx:.2f}, {gy:.2f}]. Holding...'
                self.command['motion'] = [0, 0]
                #self.pick_next_goal()
            if time.time() - self.reached_time >= 2.0:
                self.notification = f'Completed target [{gx:.2f}, {gy:.2f}]'
                self._advance_target()
                self.replan(initial=False)
                self.pick_next_goal()
            return

        # else:   
        #     self.pick_next_goal()
        #     return

        # Turn-then-drive
        if abs(dheading) > self.angle_tol:
            self.command['motion'] = [0, self.turn_cmd if dheading > 0 else -self.turn_cmd]
        else:
            self.command['motion'] = [self.fwd_cmd, 0]
    def _rrt_bounds_and_obstacles(self):
        """
        Build:
          - bounds: ((xmin,xmax),(ymin,ymax)) expanded around goals/obstacles
          - obstacles: [(x,y,r), ...] using centers from known + discovered obstacles
        Radius:
          - If you loaded file obstacles elsewhere with their own radius, pass those
            centers into known_obstacles and we’ll reattach a default radius here.
        """
        # Points to infer bounds from
        pts = []
        pts += [list(t) for t in self.remaining_targets]
        pts += [list(o) for o in self.known_obstacles]
        pts += [list(o) for o in self.discovered_obstacles]
        if not pts:
            pts = [[0.0, 0.0]]

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        margin = max(0.20, 5.0 * self.grid_res)  # small padding
        bounds = ((min(xs) - margin, max(xs) + margin),
                  (min(ys) - margin, max(ys) + margin))

        # Build circular obstacles for RRT: use robot_radius + safety_margin
        default_r = 0.
        obstacles = []
        for (ox, oy) in (self.known_obstacles + self.discovered_obstacles):
            obstacles.append((float(ox), float(oy), default_r))

        return bounds, obstacles

    # ============= Planning =============
    def replan(self, initial: bool = False):
        """Plan waypoints from current pose to the NEXT target using RRT."""
        if not self.active:
            return
        if not self.remaining_targets:
            self.waypoints = []
            self.current_goal = None
            self.active = False
            self.notification = 'All targets completed'
            return

        # Current pose and next goal
        x, y, _ = self.get_pose()
        start_xy = [float(x), float(y)]
        goal_xy  = [float(self.remaining_targets[0][0]), float(self.remaining_targets[0][1])]
        # Build RRT world
        bounds, obstacles = self._rrt_bounds_and_obstacles()
        bounds = ((-1.4136, 1.356), (-1.3272, 1.3368))
        obstacles = [(-0.3936, 0.2208, 0.1), (-0.0288, 0.9503999999999999, 0.1), (-0.6816, 1.0368, 0.1), (0.624, -0.8064, 0.1), (0.9119999999999999, 0.4896, 0.1), (0.0384, 0.288, 0.1), (0.7584, -0.288, 0.1), (0.0192, -0.9792, 0.1), (-0.0096, -0.5664, 0.1), (-1.1136, -0.2592, 0.1), (-1.0272, -1.0272, 0.1), (1.056, -1.008, 0.1), (-0.144, 0.6624, 0.1), (-0.6624, -0.9503999999999999, 0.1), (0.8256, 0.2496, 0.1), (0.33599999999999997, -0.6719999999999999, 0.1), (0.5088, 0.864, 0.1), (-0.8927999999999999, -0.4608, 0.1), (-0.5952, 0.7584, 0.1), (-0.9887999999999999, 0.9792, 0.1)]

     
        #print("Bounds:",bounds)
        #print("obstacles",obstacles)
        # RRT parameters (tuned to your defaults)
        step_size = max(0.02, 3.0 * self.grid_res)   # explore ~ few grid cells per step
        goal_sample_rate = 0.20                       # bias to goal
        max_iters = 10000
        goal_tol = max(self.dist_tol, 0.20)          # match your demo tolerance

        try:
            #print("START:", start_xy,"GOAL:", goal_xy)
            rrt = RRTPlanner(bounds, obstacles,step_size=0.1, goal_sample_rate=0.2, max_iters=2000)
            path = rrt.plan(start_xy, goal_xy, 0.2)
            #print("Obstacles:", obstacles)
            if path is None or len(path) < 2:
                self.waypoints = []
                self.current_goal = None
                self.notification = 'RRT failed to find a path'
                return

            # Use the path as waypoints (skip the start element)
            self.waypoints = [[float(px), float(py)] for (px, py) in path[1:]]
            self.current_goal = None  # pick first on next control step
            if initial:
                self.notification = f'RRT planned {len(self.waypoints)} waypoints (initial)'
            else:
                self.notification = f'RRT replanned {len(self.waypoints)} waypoints'

        except Exception as e:
            self.waypoints = []
            self.current_goal = None
            self.notification = f'RRT planning error: {e}'

    def _advance_target(self):
        if not self.remaining_targets:
            return
        # Remove the front target as completed
        self.remaining_targets.pop(0)
        if hasattr(self, 'remaining_labels') and self.remaining_labels:
            self.remaining_labels.pop(0)
        # Reset alignment state when target advances
        self.aligning = False
        self.reached_time = None
        if not self.remaining_targets:
            self.waypoints = []
            self.current_goal = None
            self.active = False
            self.notification = 'All targets completed'

    def _is_close_to_current_target(self, goal_xy: List[float]) -> bool:
        if not self.remaining_targets:
            return False
        tx, ty = self.remaining_targets[0]
        return math.hypot(goal_xy[0] - tx, goal_xy[1] - ty) <= max(0.12, self.grid_res * 3)

    # ============= Perception integration =============
    def periodic_perception_update(self):
        """Process detector outputs to add unknown obstacles, and replan if new obstacles observed."""
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
                # It's a target class; don't treat as obstacle
                continue

            # Project detection center to a rough world point
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

    # ============= Alignment helpers =============
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
        """Return True when aligned/close; else command motion for alignment and return False."""
        label = self._current_target_label()
        if not label:
            self.command['motion'] = [0, 0]
            return True

        found = self._find_detection_for_label(label)
        cx = self.cx
        if found is None:
            # Slow scan left/right while near target
            now = time.time()
            # toggle direction every ~1s to avoid spinning
            if int(now - self.align_start_time) % 2 == 0:
                self._align_scan_dir = 1
            else:
                self._align_scan_dir = -1
            self.command['motion'] = [0, self._align_scan_dir * self.turn_cmd]
            return False

        u, w_px, conf = found
        dx = u - cx
        # Step 1: center horizontally
        if abs(dx) > self.center_tol_px:
            turn = self.turn_cmd if dx > 0 else -self.turn_cmd
            self.command['motion'] = [0, turn]
            return False

        # Step 2: close-in a little if still far (bbox not wide enough)
        if w_px < self.close_width_px:
            self.command['motion'] = [self.fwd_cmd, 0]
            return False

        # Centered and close enough
        self.command['motion'] = [0, 0]
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Level 3: Partial-map A* with online obstacle discovery + GUI/SLAM")
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
    goals = [[-0.8927999999999999, -0.4608], [0.5088, 0.864], [-0.9887999999999999, 0.9792], [1.056, -1.008], [-1.0272, -1.0272]]


    # 3) Convert into the formats AutoOperateDynamic expects
    #    - remaining_targets: list of [x,y]
    targets_xy: List[List[float]] = [list(g) for g in goals]
    #    - search_list: labels just for UI/messages (not used for detection here)
    search_list = [f"goal{i+1}" for i in range(len(goals))]
    #    - known_obstacles: centers only (radii will be re-attached inside RRT builder)
    aruco_obstacles_xy: List[List[float]] = [[float(x), float(y)] for (x, y, r) in rrt_obstacles]

    # Ensure Week05-06 relative asset paths in operate.py resolve
    try:
        os.chdir(WEEK0506_DIR)
    except Exception:
        pass

    # Create operator (same GUI/SLAM, but planning will call RRT in replan())
    operate = AutoOperateDynamic(op_args, search_list, targets_xy, aruco_obstacles_xy,
                                 grid_res=args.grid_res,
                                 robot_radius=args.robot_radius,
                                 safety_margin=args.safety_margin)
    #print(goals)
    #exit()
    # Main loop unchanged
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
        #exit()
