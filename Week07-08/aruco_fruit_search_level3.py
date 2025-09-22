#!/usr/bin/env python3
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
        # remaining_targets holds world positions of targets in order
        self.remaining_targets: List[List[float]] = [list(t) for t in targets_xy]
        # remaining_labels keeps the same order but stores the class label for each target
        self.remaining_labels: List[str] = [s.lower() for s in search_list]

        self.known_obstacles: List[List[float]] = [list(o) for o in aruco_obstacles_xy]
        self.discovered_obstacles: List[List[float]] = []

        # A* params
        self.grid_res = grid_res
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin

        # Controller params (mirror Level 2)
        self.waypoints = []  # type: List[List[float]]
        self.current_goal = None  # type: List[float] | None
        self.reached_time = None
        self.active = True
        # waypoint arrival tolerance (meters)
        self.dist_tol = 0.10
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
        # React faster to new obstacles
        self.add_cooldown = 0.2  # seconds
        self.min_obs_separation = 0.15  # m

        # Cache intrinsics (for projection of bbox -> world)
        self.K = getattr(self.ekf.robot, 'camera_matrix', None)
        self.cx = float(self.K[0, 2]) if self.K is not None else 160.0
        self.fx = float(self.K[0, 0]) if self.K is not None else 320.0

        # Target arrival + backoff to avoid bumping the fruit when turning
        self.target_hold_radius = 0.25  # meters (marking requirement)
        self.backoff_active = False
        self.backoff_start_time = 0.0
        self.backoff_duration = 1  # seconds to reverse
        self.backoff_speed = 0.6    # reverse speed (same scale as fwd_cmd)
        self._advance_after_backoff = False

        # Startup 360° scan: pause every 30° for 0.5s to acquire markers/fruits
        self.startup_scan_enabled = True
        self._startup_scan_done = False
        self._scan_step_rad = math.radians(30.0)
        self._scan_pause = 0.5  # seconds per stop
        self._scan_total_steps = int(round(2 * math.pi / self._scan_step_rad))
        self._scan_steps_done = 0
        self._scan_target_heading = None
        self._scan_pause_until = None

    # ============= Navigation primitives =============
    def get_pose(self) -> Tuple[float, float, float]:
        if hasattr(self, "ekf") and self.ekf is not None:
            robot = getattr(self.ekf, "robot", None)
            if robot is not None and hasattr(robot, "state") and robot.state.shape[0] >= 3:
                x = float(robot.state[0, 0])
                y = float(robot.state[1, 0])
                th = float(robot.state[2, 0])
                return x, y, th
        return 0.0, 0.0, 0.0

    def pick_next_goal(self):
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

        # Startup scan routine (run once): rotate 360° in steps and pause to stabilize detections
        if self.startup_scan_enabled and not self._startup_scan_done:
            self._do_startup_scan()
            return

        # Handle post-hold backoff
        if getattr(self, 'backoff_active', False):
            now = time.time()
            if now - self.backoff_start_time < self.backoff_duration:
                self.command['motion'] = [-self.backoff_speed, 0]
                self.notification = 'Backing off from target'
                return
            # Backoff complete -> stop and advance
            self.backoff_active = False
            self.command['motion'] = [0, 0]
            if self._advance_after_backoff:
                self._advance_after_backoff = False
                self._advance_target()
                self.replan(initial=False)
                self.pick_next_goal()
            return

        # Marker acquisition gate: scan and occasional creep until >=2 tags visible
        tag_count = len(getattr(self.ekf, 'taglist', []))
        now = time.time()
        if tag_count < 2:
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

        # Control to goal
        x, y, th = self.get_pose()
        # Local safety: if too close to any obstacle, stop and replan
        if self.active:
            all_obs = self.known_obstacles + self.discovered_obstacles
            if all_obs:
                dmin = min(math.hypot(x - ox, y - oy) for (ox, oy) in all_obs)
                # stop if within (robot_radius + small buffer)
                if dmin < (self.robot_radius + 0.05):
                    self.command['motion'] = [0, 0]
                    self.notification = 'Too close to obstacle — stopping and replanning'
                    self.replan(initial=False)
                    return

        gx, gy = self.current_goal
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        dheading = angle_diff(bearing, th)

        # If we are within hold radius of the true target, stop/hold even if goal is slightly offset
        if self.remaining_targets:
            tx, ty = self.remaining_targets[0]
            dist_to_target = math.hypot(tx - x, ty - y)
            if dist_to_target <= self.target_hold_radius:
                if self.reached_time is None:
                    self.reached_time = time.time()
                    self.notification = f'Reached target [{tx:.2f}, {ty:.2f}] (within {self.target_hold_radius:.2f}m). Holding...'
                self.command['motion'] = [0, 0]
                if time.time() - self.reached_time >= 2.0:
                    # Start a brief reverse before advancing to the next target
                    self.backoff_active = True
                    self.backoff_start_time = time.time()
                    self._advance_after_backoff = True
                    self.notification = f'Backing off from target before next target'
                    self.command['motion'] = [-self.backoff_speed, 0]
                return

        # Arrival handling: if this goal is close to the next target, hold; else skip hold
        if dist <= self.dist_tol:
            if self._is_close_to_current_target([gx, gy]):
                if self.reached_time is None:
                    self.reached_time = time.time()
                    self.notification = f'Reached target [{gx:.2f}, {gy:.2f}]. Holding...'
                self.command['motion'] = [0, 0]
                if time.time() - self.reached_time >= 2.0:
                    # Start a brief reverse before advancing to the next target
                    self.backoff_active = True
                    self.backoff_start_time = time.time()
                    self._advance_after_backoff = True
                    self.notification = f'Backing off from [{gx:.2f}, {gy:.2f}] before next target'
                    self.command['motion'] = [-self.backoff_speed, 0]
                return
            else:
                self.pick_next_goal()
                return

        # Turn-then-drive
        if abs(dheading) > self.angle_tol:
            # Rotate in place to reduce heading error before moving
            self.command['motion'] = [0, self.turn_cmd if dheading > 0 else -self.turn_cmd]
        else:
            self.command['motion'] = [self.fwd_cmd, 0]

    def _do_startup_scan(self):
        """Rotate 360° in 30° steps, pausing at each step to stabilize detections and SLAM.
        Runs once at startup if enabled. Uses ekf robot heading as feedback."""
        # If we already completed the scan, return
        if self._startup_scan_done:
            return
        # If we've done all steps, finish
        if self._scan_steps_done >= self._scan_total_steps:
            self._startup_scan_done = True
            self.notification = 'Startup scan complete'
            self.command['motion'] = [0, 0]
            return

        # Read current heading
        _, _, th = self.get_pose()
        now = time.time()

        # If we are in a pause window, hold still
        if self._scan_pause_until is not None and now < self._scan_pause_until:
            # Hold perfectly still during pause
            self.command['motion'] = [0, 0]
            self.notification = 'Startup scan: pausing'
            return
        else:
            # End pause
            self._scan_pause_until = None

        # If no target heading yet, set the next one relative to current
        if self._scan_target_heading is None:
            self._scan_target_heading = normalize_angle(th + self._scan_step_rad)

        # Compute smallest signed angle from th to target
        dtheta = angle_diff(self._scan_target_heading, th)
        ang_tol = math.radians(3.0)

        if abs(dtheta) > ang_tol:
            # Turn towards target heading
            self.command['motion'] = [0, self.turn_cmd if dtheta > 0 else -self.turn_cmd]
            self.notification = 'Startup scan: rotating'
        else:
            # Reached this step heading — start pause and advance to next step
            self.command['motion'] = [0, 0]
            self._scan_steps_done += 1
            self._scan_target_heading = None
            self._scan_pause_until = now + self._scan_pause
            self.notification = f'Startup scan: step {self._scan_steps_done}/{self._scan_total_steps}'

    # ============= Planning =============
    def replan(self, initial: bool = False):
        """Plan waypoints from current pose to remaining targets, avoiding known+discovered obstacles."""
        if not self.active:
            return
        if not self.remaining_targets:
            self.waypoints = []
            self.current_goal = None
            self.active = False
            self.notification = 'All targets completed'
            return
        x, y, _ = self.get_pose()
        robot_xy = [x, y]
        obstacles_xy = self.known_obstacles + self.discovered_obstacles
        try:
            new_waypoints = plan_waypoints(robot_xy, self.remaining_targets, obstacles_xy,
                                           grid_res=self.grid_res,
                                           robot_radius=self.robot_radius,
                                           safety_margin=self.safety_margin,
                                           bounds_margin=0.25)
            self.waypoints = new_waypoints
            self.current_goal = None  # will pick first on next control step
            if initial:
                self.notification = f'Planned {len(self.waypoints)} waypoints via A* (initial)'
            else:
                self.notification = f'Replanned path with {len(self.waypoints)} waypoints'
        except Exception as e:
            # Fallback: relax safety margin and try once more
            try:
                relaxed_margin = max(0.05, self.safety_margin - 0.08)
                new_waypoints = plan_waypoints(robot_xy, self.remaining_targets, obstacles_xy,
                                               grid_res=self.grid_res,
                                               robot_radius=self.robot_radius,
                                               safety_margin=relaxed_margin,
                                               bounds_margin=0.35)
                self.waypoints = new_waypoints
                self.current_goal = None
                self.notification = f'Replanned with relaxed margin ({relaxed_margin:.2f}); {len(self.waypoints)} waypoints'
            except Exception as e2:
                # Last resort: enlarge bounds further and relax safety again
                try:
                    more_relaxed = max(0.03, relaxed_margin - 0.04)
                    new_waypoints = plan_waypoints(robot_xy, self.remaining_targets, obstacles_xy,
                                                   grid_res=self.grid_res,
                                                   robot_radius=self.robot_radius,
                                                   safety_margin=more_relaxed,
                                                   bounds_margin=0.45)
                    self.waypoints = new_waypoints
                    self.current_goal = None
                    self.notification = f'Replanned with enlarged bounds + margin ({more_relaxed:.2f}); {len(self.waypoints)} waypoints'
                except Exception as e3:
                    self.notification = f'Planning failed: {e3}'

    def _advance_target(self):
        if not self.remaining_targets:
            return
        # Remove the front target as completed
        self.remaining_targets.pop(0)
        if hasattr(self, 'remaining_labels') and self.remaining_labels:
            self.remaining_labels.pop(0)
        # Reset timing state when target advances
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
        # Defer dynamic obstacle integration until after startup scan and sufficient markers
        if self.startup_scan_enabled and not self._startup_scan_done:
            return
        if len(getattr(self.ekf, 'taglist', [])) < 2:
            return
        bboxes = getattr(self, 'detector_output', None)
        if not isinstance(bboxes, (list, tuple)) or len(bboxes) == 0:
            return
        now = time.time()
        new_added = False

        # Current pose and intrinsics
        x, y, th = self.get_pose()
        cx, fx = self.cx, self.fx

        # Copy remaining targets to avoid blocking them as obstacles where appropriate
        # Note: we will only *ignore* detections that correspond to the current target (index 0).
        known_targets = self.remaining_targets[:]

        # tolerance for matching detection to the current target (meters)
        target_match_tol = 0.30

        for det in bboxes:
            try:
                label: str = str(det[0]).lower()
                xywh = np.asarray(det[1]).astype(float)
                conf = float(det[2])
            except Exception:
                continue
            if conf < 0.4:
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

            # ===== NEW BEHAVIOR: only ignore detection if it corresponds to the CURRENT target =====
            if label in self.search_list:
                # Is there a current target?
                if len(self.remaining_targets) > 0 and len(self.remaining_labels) > 0:
                    current_label = str(self.remaining_labels[0]).lower()
                    if label == current_label:
                        # If the detection is close enough to the known current-target position,
                        # it's likely the same target — ignore it as obstacle.
                        tx, ty = self.remaining_targets[0]
                        if math.hypot(ox - tx, oy - ty) <= target_match_tol:
                            # This detection corresponds to current target -> ignore as obstacle
                            continue
                        # else: detection of same label but not near current target -> treat as obstacle
                    else:
                        # detection is a shopping-list class but NOT the current target -> treat as obstacle
                        pass
                else:
                    # no reliable remaining_targets/labels -> conservatively treat detected shopping-list
                    # classes as obstacles (safe fallback)
                    pass

            # If close to a known target (other than the special-case above), ignore to avoid false positives
            # (But note: we intentionally let non-current shopping-list detections get treated as obstacles.
            #  This check prevents marking an obstacle if it's *very* close to a known target centre.)
            if any(math.hypot(ox - tx, oy - ty) <= 0.10 for tx, ty in known_targets):
                # Very close to a known target centre -> skip adding (avoids tiny projection noise)
                continue

            # Ignore duplicates amongst known and discovered obstacles
            all_obs = self.known_obstacles + self.discovered_obstacles
            if any(math.hypot(ox - qx, oy - qy) <= self.min_obs_separation for qx, qy in all_obs):
                continue

            # Rate limit
            if now - self.last_obstacle_add_time < self.add_cooldown:
                continue

            # Add obstacle and mark for replanning
            self.discovered_obstacles.append([ox, oy])
            self.last_obstacle_add_time = now
            new_added = True

        if new_added and self.ekf_on and self.active:
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
    try:
        pygame.display.set_icon(pygame.image.load(os.path.join(WEEK0506_DIR, 'pics', '8bit', 'pibot5.png')))
    except Exception:
        pass
    canvas.fill((0, 0, 0))

    # Load partial map + shopping list, print targets
    fruit_list, fruit_pos, aruco_pos = read_true_map_robust(args.map)
    search_list = load_search_list(args.list)
    print_target_fruits_pos(search_list, fruit_list, fruit_pos)

    # Build targets (in order) from partial map; obstacles initially only ArUcos
    targets_xy: List[List[float]] = []
    for ft in search_list:
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
        # Detect obstacles and replan BEFORE commanding motion for this cycle
        operate.detect_target()  # populates operate.detector_output
        operate.periodic_perception_update()  # add obstacles + replan if needed
        operate.auto_nav_step()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        operate.draw(canvas)
        pygame.display.update()
