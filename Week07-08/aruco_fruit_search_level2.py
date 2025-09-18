import os
import sys
import argparse
import time
import math
from types import SimpleNamespace

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

# Helpers (same folder)
from map_utils import (
    read_true_map_robust,
    load_search_list,
    print_target_fruits_pos,
    build_targets_and_obstacles,
)

# Planner (try astar_planning.py; fallback to planning.py name if used)
try:
    from astar_planning import plan_waypoints  # type: ignore
except Exception:
    from planning import plan_waypoints  # type: ignore


def normalize_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class AutoOperatePlan(Operate):
    """A* planned waypoint follower on top of Operate GUI/SLAM."""

    def __init__(self, args, waypoints, fruit_indices):
        super().__init__(args)
        self.waypoints = waypoints[:]  # planned path (list[[x,y], ...])
        self.fruit_indices = fruit_indices[:]  # indices of fruit target waypoints
        self.current_goal = None
        self.reached_time = None
        self.active = True if self.waypoints else False
        self.dist_tol = 0.25
        self.angle_tol = math.radians(10.0)
        self.turn_cmd = 1
        self.fwd_cmd = 1
        # Marker acquisition (scan/creep) state
        self._scan_start = None
        self._scan_dir = 1
        self._creep_until = None
        self.waypoint_counter = 0  # track which waypoint we're at

    def get_pose(self):
        if hasattr(self, "ekf") and self.ekf is not None:
            mu = getattr(self.ekf, "robot", None)
            if mu is not None and hasattr(mu, "state") and mu.state.shape[0] >= 3:
                x = float(mu.state[0, 0])
                y = float(mu.state[1, 0])
                th = float(mu.state[2, 0])
                #print("POSE:", x, y, th)
                return x, y, th
        print("ERROR UNKNOWN POSITIONS")
        return 0.0, 0.0, 0.0

    def pick_next_goal(self):
        if not self.waypoints:
            self.current_goal = None
            self.active = False
            self.notification = 'All targets completed'
            return
        self.current_goal = self.waypoints.pop(0)
        self.reached_time = None
        self.notification = f'Navigating to: [{self.current_goal[0]:.2f}, {self.current_goal[1]:.2f}]'
        self.waypoint_counter += 1

    def auto_nav_step(self):
        # Require SLAM
        if not self.ekf_on:
            self.command['motion'] = [0, 0]
            self.notification = 'Press ENTER to start SLAM'
            return

        # Marker acquisition gate: scan and occasional creep until >=2 tags visible
        tag_count = len(getattr(self.ekf, 'taglist', []))
        now = time.time()
        #print("TAG COUNT:",tag_count)
        if tag_count < 2: #checks if tag count more than 3
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

        # Get/set goal
        if self.current_goal is None and self.active:
            self.pick_next_goal()
        if not self.current_goal:
            self.command['motion'] = [0, 0]
            return

        # Control to goal
        x, y, th = self.get_pose()
        #print(x,y,th)
        gx, gy = self.current_goal
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        dheading = normalize_angle(bearing - th)

        # Arrival hold
        if dist <= self.dist_tol:
            is_fruit = (self.waypoint_counter - 1) in self.fruit_indices
            if is_fruit:
                if self.reached_time is None:
                    self.reached_time = time.time()
                    self.notification = f'Reached [{gx:.2f}, {gy:.2f}] (fruit). Holding...'
                self.command['motion'] = [0, 0]
                if time.time() - self.reached_time >= 2.0:
                    self.notification = f'Completed [{gx:.2f}, {gy:.2f}] (fruit)'
                    self.pick_next_goal()
                return
            else:
                self.pick_next_goal()
                return

        # Turn-then-drive
        if abs(dheading) > self.angle_tol:
            #print("TURNING STUCK HERE",abs(dheading),self.angle_tol)
            self.command['motion'] = [0, self.turn_cmd if dheading > 0 else -self.turn_cmd]
        else:
            self.command['motion'] = [self.fwd_cmd, 0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Level 2: Known-map A* planning + GUI/SLAM")
    parser.add_argument("--ip", type=str, default="192.168.50.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default=os.path.join(REPO_ROOT, "Week02-04", "calibration", "param") + os.sep)
    parser.add_argument("--yolo_model", default=os.path.join(SCRIPT_DIR, "YOLO", "model", "bestv5.pt"))
    parser.add_argument("--map", type=str, default=os.path.join(SCRIPT_DIR, "M3_prac_map_full.txt"))
    parser.add_argument("--list", type=str, default=os.path.join(SCRIPT_DIR, "M3_prac_shopping_list.txt"))
    parser.add_argument("--grid_res", type=float, default=0.02)
    parser.add_argument("--robot_radius", type=float, default=0.12)
    parser.add_argument("--safety_margin", type=float, default=0.1)
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
    pygame.display.set_caption('ECE4078 - Auto Fruit Search (L2)')
    pygame.display.set_icon(pygame.image.load(os.path.join(WEEK0506_DIR, 'pics', '8bit', 'pibot5.png')))
    canvas.fill((0, 0, 0))

    # Load map + shopping list, print targets
    fruit_list, fruit_pos, aruco_pos = read_true_map_robust(args.map)
    search_list = load_search_list(args.list)
    print_target_fruits_pos(search_list, fruit_list, fruit_pos)

    # Build targets and obstacles, plan A* waypoints
    targets_xy, obstacles_xy = build_targets_and_obstacles(fruit_list, fruit_pos, aruco_pos, search_list)

    # Initial pose for planning: origin (SLAM will localize during run)
    start_xy = [0.0, 0.0]
    waypoints = plan_waypoints(start_xy, targets_xy, obstacles_xy,
                               grid_res=args.grid_res,
                               robot_radius=args.robot_radius,
                               safety_margin=args.safety_margin)
    # Identify fruit target indices (assuming targets_xy are fruit targets in order)
    fruit_indices = list(range(len(targets_xy)))
    print(f"Planned {len(waypoints)} waypoints via A*")

    # Ensure Week05-06 relative asset paths in operate.py resolve
    try:
        os.chdir(WEEK0506_DIR)
    except Exception:
        pass

    operate = AutoOperatePlan(op_args, waypoints, fruit_indices)

    running = True
    while running:
        operate.update_keyboard()
        operate.take_pic()
        operate.auto_nav_step()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        operate.detect_target()
        operate.draw(canvas)
        pygame.display.update()