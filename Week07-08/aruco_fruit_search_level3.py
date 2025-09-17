import os
import sys
import argparse
import time
import math
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import pygame


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
        self.known_obstacles: List[List[float]] = [list(o) for o in aruco_obstacles_xy]
        self.discovered_obstacles: List[List[float]] = []

        # A* params
        self.grid_res = grid_res
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin

        # Controller params
        self.waypoints: List[List[float]] = []
        self.current_goal: List[float] | None = None
        self.reached_time: float | None = None
        self.active = True
        self.dist_tol = 0.25
        self.angle_tol = math.radians(8.0)
        self.turn_cmd = 1
        self.fwd_cmd = 1

        # Detection handling
        self.last_obstacle_add_time = 0.0
        self.add_cooldown = 0.5  # seconds
        self.min_obs_separation = 0.15  # m

        # Cache intrinsics
        self.K = getattr(self.ekf.robot, 'camera_matrix', None)
        self.cx = float(self.K[0, 2]) if self.K is not None else 160.0
        self.fx = float(self.K[0, 0]) if self.K is not None else 320.0

        # Initial plan from origin; will replan once SLAM stabilizes too
        self.replan(initial=True)

    # ============= Navigation primitives =============
    def get_pose(self) -> Tuple[float, float, float]:
        mu = getattr(self.ekf, "mu", None)
        if mu is not None and len(mu) >= 3:
            return float(mu[0]), float(mu[1]), float(mu[2])
        return 0.0, 0.0, 0.0

    def pick_next_goal(self):
        if not self.waypoints:
            self.current_goal = None
            return
        self.current_goal = self.waypoints.pop(0)
        self.reached_time = None
        self.notification = f'Navigating to: [{self.current_goal[0]:.2f}, {self.current_goal[1]:.2f}]'

    def auto_nav_step(self):
        # Require SLAM running
        if not self.ekf_on:
            self.command['motion'] = [0, 0]
            self.notification = 'Press ENTER to start SLAM'
            return
        # Initialize goal if needed
        if self.current_goal is None and self.active:
            self.pick_next_goal()
        if not self.current_goal:
            self.command['motion'] = [0, 0]
            return

        x, y, th = self.get_pose()
        gx, gy = self.current_goal
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        dheading = angle_diff(bearing, th)

        # Arrival (hold 2s)
        if dist <= self.dist_tol:
            if self.reached_time is None:
                self.reached_time = time.time()
                self.notification = f'Reached [{gx:.2f}, {gy:.2f}]. Holding...'
            self.command['motion'] = [0, 0]
            if time.time() - self.reached_time >= 2.0:
                self.notification = f'Completed [{gx:.2f}, {gy:.2f}]'
                # If this goal corresponds exactly to the next target, and we're close, we can pop targets
                self._maybe_advance_target([gx, gy])
                self.pick_next_goal()
            return

        # Turn-then-drive
        if abs(dheading) > self.angle_tol:
            self.command['motion'] = [0, self.turn_cmd if dheading > 0 else -self.turn_cmd]
        else:
            self.command['motion'] = [self.fwd_cmd, 0]

    # ============= Planning =============
    def replan(self, initial: bool = False):
        """Plan waypoints from current pose to remaining targets, avoiding known+discovered obstacles."""
        if not self.active:
            return
        x, y, _ = self.get_pose()
        robot_xy = [x, y]
        obstacles_xy = self.known_obstacles + self.discovered_obstacles
        if not self.remaining_targets:
            self.waypoints = []
            self.current_goal = None
            self.active = False
            self.notification = 'All targets completed'
            return
        try:
            new_waypoints = plan_waypoints(robot_xy, self.remaining_targets, obstacles_xy,
                                           grid_res=self.grid_res,
                                           robot_radius=self.robot_radius,
                                           safety_margin=self.safety_margin)
            self.waypoints = new_waypoints
            self.current_goal = None  # will pick first on next control step
            if initial:
                self.notification = f'Planned {len(self.waypoints)} waypoints via A* (initial)'
            else:
                self.notification = f'Replanned path with {len(self.waypoints)} waypoints'
        except Exception as e:
            self.notification = f'Planning failed: {e}'

    def _maybe_advance_target(self, reached_xy: List[float]):
        if not self.remaining_targets:
            return
        tx, ty = self.remaining_targets[0]
        if math.hypot(reached_xy[0] - tx, reached_xy[1] - ty) <= self.dist_tol:
            # Pop reached target
            self.remaining_targets.pop(0)
            # Trigger replan for subsequent targets
            if self.remaining_targets:
                self.replan(initial=False)
            else:
                self.waypoints = []
                self.current_goal = None
                self.active = False
                self.notification = 'All targets completed'

    # ============= Perception integration =============
    def periodic_perception_update(self):
        """Process detector outputs to add unknown obstacles, and replan if new obstacles observed."""
        # YOLO outputs saved on operate.detector_output by Operate.detect_target()
        bboxes = getattr(self, 'detector_output', None)
        if not isinstance(bboxes, (list, tuple)) or len(bboxes) == 0:
            return
        now = time.time()
        new_added = False

        # Current pose and intrinsics
        x, y, th = self.get_pose()
        cx, fx = self.cx, self.fx

        # Known targets (to avoid misclassifying them as obstacles)
        known_targets = self.remaining_targets[:]  # ignore already completed

        for det in bboxes:
            try:
                label: str = str(det[0]).lower()
                xywh = np.asarray(det[1]).astype(float)
                conf = float(det[2])
            except Exception:
                continue
            if conf < 0.4:
                continue

            # If label is a target type, we ignore it as an obstacle (its position already known from partial map)
            # However, duplicates of a target type could exist as obstacles (rare in provided setting). If desired,
            # enable the below check to consider far-from-known-target duplicates as obstacles.
            if label in self.search_list:
                # Skip classifying as obstacle near any known target of that type (<= 0.3 m)
                # We don't keep per-type positions here; partial map has exactly one per target type.
                continue

            # Project detection to a world point using a naive pinhole model
            u = float(xywh[0])  # bbox center x in pixels
            w_px = float(xywh[2])
            # Horizontal bearing offset from camera optical axis
            alpha = math.atan((u - cx) / fx)
            bearing = th + alpha
            # Distance heuristic: d ≈ fx * W / w_px (assume avg fruit width W ~ 0.10m); clamp
            W_assumed = 0.10
            if w_px <= 1.0:
                d = 0.5
            else:
                d = max(0.35, min(1.10, (fx * W_assumed) / w_px))
            ox = x + d * math.cos(bearing)
            oy = y + d * math.sin(bearing)

            # If close to a known target, ignore
            too_close_to_target = any(math.hypot(ox - tx, oy - ty) <= 0.30 for tx, ty in known_targets)
            if too_close_to_target:
                continue

            # If duplicate obstacle, ignore
            all_obs = self.known_obstacles + self.discovered_obstacles
            is_duplicate = any(math.hypot(ox - qx, oy - qy) <= self.min_obs_separation for qx, qy in all_obs)
            if is_duplicate:
                continue

            # Cooldown to avoid spamming
            if now - self.last_obstacle_add_time < self.add_cooldown:
                continue

            # Add obstacle and mark for replanning
            self.discovered_obstacles.append([ox, oy])
            self.last_obstacle_add_time = now
            new_added = True

        if new_added and self.ekf_on:
            self.replan(initial=False)


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
    parser.add_argument("--safety_margin", type=float, default=0.05)
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

    # Load partial map + shopping list, print targets
    fruit_list, fruit_pos, aruco_pos = read_true_map_robust(args.map)
    search_list = load_search_list(args.list)
    print_target_fruits_pos(search_list, fruit_list, fruit_pos)

    # Build targets (in order) from partial map; obstacles initially only ArUcos
    # For partial map, fruit_list/fruit_pos already contain exactly the targets listed in search_list (one each)
    targets_xy: List[List[float]] = []
    for ft in search_list:
        # find first match of type in fruit_list
        found = False
        for i, name in enumerate(fruit_list):
            if name == ft:
                targets_xy.append([float(fruit_pos[i, 0]), float(fruit_pos[i, 1])])
                found = True
                break
        if not found:
            raise ValueError(f"Target '{ft}' not found in partial map")
    aruco_obstacles_xy: List[List[float]] = [[float(aruco_pos[k, 0]), float(aruco_pos[k, 1])] for k in range(aruco_pos.shape[0])]

    # Ensure Week05-06 relative asset paths in operate.py resolve
    try:
        os.chdir(WEEK0506_DIR)
    except Exception:
        pass

    operate = AutoOperateDynamic(op_args, search_list, targets_xy, aruco_obstacles_xy,
                                 grid_res=args.grid_res,
                                 robot_radius=args.robot_radius,
                                 safety_margin=args.safety_margin)

    running = True
    while running:
        operate.update_keyboard()
        operate.take_pic()
        operate.auto_nav_step()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        operate.detect_target()  # populates operate.detector_output
        operate.periodic_perception_update()  # add obstacles + replan if needed
        operate.draw(canvas)
        pygame.display.update()
