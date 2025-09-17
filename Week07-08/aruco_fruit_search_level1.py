import os
import sys
import time
import math
import argparse
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
    # Fallback: explicit load from file so runtime still works even if Pylance complains
    import importlib.util
    _op_file = os.path.join(WEEK0506_DIR, "operate.py")
    _spec = importlib.util.spec_from_file_location("operate", _op_file)
    operate_mod = importlib.util.module_from_spec(_spec)  # type: ignore
    assert _spec and _spec.loader
    _spec.loader.exec_module(operate_mod)  # type: ignore
    Operate = operate_mod.Operate  # type: ignore

# Helpers to load map + shopping list (module in same folder)
from map_utils import read_true_map_robust, load_search_list, print_target_fruits_pos


def parse_waypoints(waypoint_str: str):
    if not waypoint_str:
        return []
    waypoints = []
    for pair in waypoint_str.split(";"):
        if not pair.strip():
            continue
        x_str, y_str = pair.split(",")
        waypoints.append([float(x_str), float(y_str)])
    return waypoints


def normalize_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class AutoOperate(Operate):
    """Waypoint follower built on top of Operate GUI/SLAM."""

    def __init__(self, args, waypoints):
        super().__init__(args)
        self.waypoints = waypoints[:]  # [[x,y], ...]
        self.current_goal = None
        self.reached_time = None
        self.active = True if self.waypoints else False
        self.dist_tol = 0.25
        self.angle_tol = math.radians(8.0)
        self.turn_cmd = 1
        self.fwd_cmd = 1

    def get_pose(self):
        mu = getattr(self.ekf, "mu", None)
        if mu is not None and len(mu) >= 3:
            return float(mu[0]), float(mu[1]), float(mu[2])
        return 0.0, 0.0, 0.0

    def pick_next_goal(self):
        if not self.waypoints:
            self.current_goal = None
            self.active = False
            self.notification = 'All waypoints completed'
            return
        self.current_goal = self.waypoints.pop(0)
        self.reached_time = None
        self.notification = f'Navigating to: [{self.current_goal[0]:.2f}, {self.current_goal[1]:.2f}]'

    def auto_nav_step(self):
        if not self.ekf_on:
            self.command['motion'] = [0, 0]
            self.notification = 'Press ENTER to start SLAM'
            return
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
        dheading = normalize_angle(bearing - th)

        if dist <= self.dist_tol:
            if self.reached_time is None:
                self.reached_time = time.time()
                self.notification = f'Reached [{gx:.2f}, {gy:.2f}]. Holding...'
            self.command['motion'] = [0, 0]
            if time.time() - self.reached_time >= 2.0:
                self.notification = f'Completed [{gx:.2f}, {gy:.2f}]'
                self.pick_next_goal()
            return

        if abs(dheading) > self.angle_tol:
            self.command['motion'] = [0, self.turn_cmd if dheading > 0 else -self.turn_cmd]
        else:
            self.command['motion'] = [self.fwd_cmd, 0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Level 1: Waypoint navigation with GUI/SLAM")
    parser.add_argument("--ip", type=str, default="192.168.50.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default=os.path.join(REPO_ROOT, "Week02-04", "calibration", "param") + os.sep)
    # Default to Week07-08 YOLO model path (absolute), matching operate.py expectation but pointing to your folder
    parser.add_argument("--yolo_model", default=os.path.join(SCRIPT_DIR, "YOLO", "model", "bestv5.pt"))
    parser.add_argument("--waypoints", type=str, default="", help='Format: "x1,y1;x2,y2;..." (meters)')
    parser.add_argument("--map", type=str, default=os.path.join(SCRIPT_DIR, "M3_prac_map_full.txt"))
    parser.add_argument("--list", type=str, default=os.path.join(SCRIPT_DIR, "M3_prac_shopping_list.txt"))
    parser.add_argument("--use_list_targets", action='store_true', help="Use map+list targets as waypoints (no planning)")
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--save_data", action='store_true')
    args, _ = parser.parse_known_args()

    # Resolve YOLO model path locally; auto-disable if not found
    def _resolve_model_path(m: str) -> str:
        # Prefer absolute path if the file exists in known locations; otherwise return as-is (operate.py behavior)
        candidates = [m,
                      os.path.abspath(m),
                      os.path.join(SCRIPT_DIR, m) if not os.path.isabs(m) else m,
                      os.path.join(REPO_ROOT, m) if not os.path.isabs(m) else m,
                      os.path.join(WEEK0506_DIR, m) if not os.path.isabs(m) else m,
                      os.path.join(SCRIPT_DIR, "YOLO", "model", os.path.basename(m))]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)
        print(f"[INFO] YOLO model not found locally ('{m}'). Proceeding with given path; Ultralytics may attempt a download.")
        return m

    yolo_model_path = _resolve_model_path(args.yolo_model)

    # Provide globals to Week05-06/operate.py expectations
    op_args = SimpleNamespace(
        ip=args.ip, port=args.port, calib_dir=args.calib_dir,
        yolo_model=yolo_model_path, play_data=args.play_data, save_data=args.save_data,
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
    pygame.display.set_caption('ECE4078 - Auto Fruit Search (L1)')
    pygame.display.set_icon(pygame.image.load(os.path.join(WEEK0506_DIR, 'pics', '8bit', 'pibot5.png')))
    canvas.fill((0, 0, 0))

    # Waypoints source: manual string or map+list targets
    if args.use_list_targets and os.path.exists(args.map) and os.path.exists(args.list):
        fruit_list, fruit_pos, aruco_pos = read_true_map_robust(args.map)
        search_list = load_search_list(args.list)
        print_target_fruits_pos(search_list, fruit_list, fruit_pos)
        waypoints = []
        used = set()
        for ft in search_list:
            for i, name in enumerate(fruit_list):
                if i in used:
                    continue
                if name == ft:
                    waypoints.append([float(fruit_pos[i, 0]), float(fruit_pos[i, 1])])
                    used.add(i)
                    break
    else:
        waypoints = parse_waypoints(args.waypoints)

    # Ensure Week05-06 relative asset paths in operate.py resolve
    try:
        os.chdir(WEEK0506_DIR)
    except Exception:
        pass

    operate = AutoOperate(op_args, waypoints)

    running = True
    while running:
        operate.update_keyboard()       # ENTER toggles SLAM
        operate.take_pic()
        operate.auto_nav_step()         # set motion command
        drive_meas = operate.control()  # send to robot
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        operate.detect_target()
        operate.draw(canvas)
        pygame.display.update()
