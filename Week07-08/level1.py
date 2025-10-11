"""Level 1 autonomous navigation using full-mapped ground truth.

This entry point mirrors the structure of `wednesday.py` but assumes that
all fruits and ArUco markers are already known (e.g. after running
Week05-06 `operate.py` and converting `slam.txt`/`targets.txt` into
`level1_ground_truth.txt`). The robot plans once from the full map and
does not add new obstacles during runtime.
"""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace
from typing import List

import math
import time

import numpy as np
import pygame

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
WEEK0506_DIR = os.path.join(REPO_ROOT, "Week05-06")

# Reuse the advanced autonomous controller from wednesday.py
import wednesday as wednesday_mod  # type: ignore
from wednesday import AutoOperateDynamic, angle_diff  # type: ignore

from map_utils import (
    read_true_map_robust,
    load_search_list,
    print_target_fruits_pos,
    build_targets_and_obstacles,
)


class AutoOperateLevel1(AutoOperateDynamic):
    """Ground-truth navigation: skips dynamic obstacle discovery."""

    def __init__(
        self,
        args: SimpleNamespace,
        search_list: List[str],
        targets_xy: List[List[float]],
        known_obstacles_xy: List[List[float]],
        grid_res: float,
        robot_radius: float,
        safety_margin: float,
        map_fruit_labels: List[str],
        map_fruit_xy: List[List[float]],
        full_ground_truth: dict | None = None,
    ):
        super().__init__(
            args,
            search_list,
            targets_xy,
            known_obstacles_xy,
            grid_res=grid_res,
            robot_radius=robot_radius,
            safety_margin=safety_margin,
            merge_threshold=0.50,
            obs_max_range=0.40,
            map_fruit_labels=map_fruit_labels,
            map_fruit_xy=map_fruit_xy,
        )
        self.full_ground_truth = full_ground_truth or {}

        # Emergency stop configuration (shared with testing patrol controller)
        self.emergency_enabled = True
        self.emergency_bbox_width_thresh_px = 150.0
        self.emergency_bbox_height_thresh_px = 150.0
        self.emergency_center_tolerance_px = 120.0
        self.emergency_dist_m = 0.22
        self.emergency_hold_time = 1.2
        self.emergency_reverse_time = 0.5
        self.emergency_cooldown = 1.0
        self._emergency_mode: str | None = None
        self._emergency_until = 0.0
        self._emergency_replan_triggered = False
        self._emergency_cooldown_until = 0.0

    def periodic_perception_update(self):
        # For Level 1 the full environment is already known, so skip
        # adding obstacles from detector outputs.
        return

    def _get_current_aruco_obstacles(self) -> List[tuple[float, float]]:
        """Return current EKF landmark positions as obstacle coordinates."""
        obs: List[tuple[float, float]] = []
        try:
            ekf = getattr(self, "ekf", None)
            if ekf is None:
                return obs
            markers = getattr(ekf, "markers", None)
            if isinstance(markers, np.ndarray) and markers.ndim == 2 and markers.shape[0] >= 2:
                for i in range(markers.shape[1]):
                    obs.append((float(markers[0, i]), float(markers[1, i])))
        except Exception:
            pass
        return obs

    def auto_nav_step(self):
        # Require SLAM to be running
        if not self.ekf_on:
            self.command['motion'] = [0, 0]
            self.notification = 'Press ENTER to start SLAM'
            return

        # --- High covariance stabilize spin (preempts other actions) ---
        try:
            now_cov = time.time()
            if self._cov_spin_until is not None and now_cov < self._cov_spin_until:
                if self._cov_spin_start is None:
                    self._cov_spin_start = now_cov
                period = float(self.cov_pulse_spin_time + self.cov_pulse_stop_time)
                phase = (now_cov - self._cov_spin_start) % period
                if phase < self.cov_pulse_spin_time:
                    self.command['motion'] = [0, self._cov_spin_dir * self.turn_cmd]
                    self.notification = 'High covariance: stabilizing spin'
                else:
                    self.command['motion'] = [0, 0]
                    self.notification = 'High covariance: stabilizing spin (pulse stop)'
                return
            if self._cov_spin_until is not None and now_cov >= self._cov_spin_until:
                self._cov_spin_until = None
                self._cov_spin_start = None
                self._cov_cooldown_until = now_cov + self.cov_spin_cooldown
            if now_cov < self._cov_cooldown_until:
                pass
            else:
                P = getattr(self.ekf, 'P', None)
                if isinstance(P, np.ndarray) and P.shape[0] >= 2 and P.shape[1] >= 2:
                    pxx = float(P[0, 0])
                    if pxx > float(self.cov_pos_thresh):
                        self._cov_spin_dir = -self._cov_spin_dir
                        self._cov_spin_until = now_cov + float(self.cov_spin_duration)
                        self._cov_spin_start = now_cov
                        self.command['motion'] = [0, self._cov_spin_dir * self.turn_cmd]
                        self.notification = 'High covariance: stabilizing spin'
                        return
        except Exception:
            pass

        # Periodic calibration trigger (non-blocking start)
        try:
            if (time.time() - self.last_calib_time) >= self.calib_interval and not self._calib_mode:
                self.start_calib_scan()
        except Exception:
            pass

        # If currently in calibration mode, do calib step and return
        if self._calib_mode:
            self._perform_calib_scan_step()
            return

        # --- Emergency stop check (overrides regular motion) ---
        try:
            now = time.time()
            if self._emergency_mode == 'reverse' and self._emergency_until <= now:
                self._emergency_mode = 'hold'
                self._emergency_until = now + float(self.emergency_hold_time)
                if not self._emergency_replan_triggered and self.active:
                    try:
                        self.replan(initial=False)
                    except Exception:
                        pass
                    self._emergency_replan_triggered = True
            if self._emergency_mode == 'hold' and self._emergency_until <= now:
                self._emergency_mode = None
                self._emergency_cooldown_until = now + float(self.emergency_cooldown)
                self._emergency_until = 0.0
                self._emergency_replan_triggered = False

            if self._emergency_mode in ('reverse', 'hold') and self._emergency_until > now:
                remaining = self._emergency_until - now
                if self._emergency_mode == 'reverse':
                    self.command['motion'] = [-self.fwd_cmd, 0]
                    self.notification = f'Emergency: reversing ({remaining:.1f}s)'
                else:
                    self.command['motion'] = [0, 0]
                    self.notification = f'Emergency: holding ({remaining:.1f}s)'
                return

            if self.emergency_enabled:
                if now >= self._emergency_cooldown_until:
                    try:
                        rx, ry, _ = self.get_pose()
                        for mx, my in self._get_current_aruco_obstacles():
                            if math.hypot(float(mx) - rx, float(my) - ry) <= self.emergency_dist_m:
                                rev = float(self.emergency_reverse_time)
                                self._emergency_mode = 'reverse'
                                self._emergency_until = now + rev
                                self.command['motion'] = [-self.fwd_cmd, 0]
                                self.notification = 'Emergency: marker too close'
                                self._emergency_replan_triggered = False
                                return
                    except Exception:
                        pass

                bboxes = getattr(self, 'detector_output', None)
                if isinstance(bboxes, (list, tuple)):
                    cx = float(self.cx)
                    fx = float(self.fx)
                    for det in bboxes:
                        try:
                            label = str(det[0]).lower()
                            xywh = np.asarray(det[1]).astype(float)
                            conf = float(det[2])
                        except Exception:
                            continue
                        if conf < 0.6:
                            continue

                        u = float(xywh[0])
                        w_px = float(xywh[2])
                        h_px = float(xywh[3])
                        if abs(u - cx) > self.emergency_center_tolerance_px:
                            continue
                        if now < self._emergency_cooldown_until:
                            continue

                        if (w_px >= self.emergency_bbox_width_thresh_px) or (h_px >= self.emergency_bbox_height_thresh_px):
                            W_assumed = 0.10
                            d_w = (fx * W_assumed) / max(1.0, w_px)
                            d_h = (fx * W_assumed) / max(1.0, h_px)
                            d_est = min(d_w, d_h)
                            if d_est <= self.emergency_dist_m:
                                rev = float(self.emergency_reverse_time)
                                self._emergency_mode = 'reverse'
                                self._emergency_until = now + rev
                                self.command['motion'] = [-self.fwd_cmd, 0]
                                self.notification = 'Emergency: reversing'
                                self._emergency_replan_triggered = False
                                return
        except Exception:
            pass

        # Marker acquisition gate: require at least 2 tags before navigating
        tag_count = len(getattr(self.ekf, 'taglist', []))
        now = time.time()

        if (now - self._last_pose_log) >= 0.2:
            self._log_pose(now)
            self._last_pose_log = now
        self._flush_log(force=False)
        if tag_count < 2:
            if self._scan_start is None:
                self._scan_start = now
                self._scan_dir = 1
            elapsed = now - self._scan_start
            self._scan_dir = 1 if int(elapsed // 2) % 2 == 0 else -1
            self.command['motion'] = [0, self._scan_dir * self.turn_cmd]
            self.notification = 'Looking for markers: scanning'
            return
        else:
            self._scan_start = None
            self._creep_until = None

        if self._reverse_until is not None:
            if now < self._reverse_until:
                self.command['motion'] = [-self.fwd_cmd, 0]
                self.notification = 'Reversing from target'
                return
            else:
                self._reverse_until = None
                if self._pending_complete_after_reverse:
                    self._pending_complete_after_reverse = False
                    self._advance_target()
                    self.replan(initial=False)
                    self.pick_next_goal()

        if (not self._planned_once and self.active) or (self.active and not self.waypoints):
            self.replan(initial=not self._planned_once)
            self._planned_once = True

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

        if dist <= self.dist_tol:
            if self._is_close_to_current_target([gx, gy]):
                if self.reached_time is None:
                    self.reached_time = time.time()
                    self.notification = f'Reached target [{gx:.2f}, {gy:.2f}]. Holding...'
                self.command['motion'] = [0, 0]
                if time.time() - self.reached_time >= self.hold_duration:
                    self._reverse_until = time.time() + self.reverse_duration
                    self._pending_complete_after_reverse = True
                    self.command['motion'] = [-self.fwd_cmd, 0]
                    self.notification = 'Reversing from target'
                return
            else:
                self.pick_next_goal()
                return

        turning = abs(dheading) > self.angle_tol
        if turning:
            now = time.time()
            if self._nav_last_mode != 'turn':
                self._nav_last_mode = 'turn'
                self._nav_turn_pulse_start = now
            if self._nav_turn_pulse_start is None:
                self._nav_turn_pulse_start = now
            t_period = float(self.nav_turn_pulse_spin_time + self.nav_turn_pulse_stop_time)
            t_phase = (now - self._nav_turn_pulse_start) % t_period
            if t_phase < self.nav_turn_pulse_spin_time:
                self.command['motion'] = [0, self.turn_cmd if dheading > 0 else -self.turn_cmd]
            else:
                self.command['motion'] = [0, 0]
        else:
            now = time.time()
            if self._nav_last_mode != 'drive':
                self._nav_last_mode = 'drive'
                self._nav_drive_pulse_start = now
            if self._nav_drive_pulse_start is None:
                self._nav_drive_pulse_start = now
            d_period = float(self.nav_drive_pulse_period)
            d_phase = (now - self._nav_drive_pulse_start) % d_period
            if d_phase < (self.nav_drive_pulse_period - self.nav_drive_pulse_stop_time):
                self.command['motion'] = [self.fwd_cmd, 0]
            else:
                self.command['motion'] = [0, 0]


def parse_args() -> argparse.Namespace:
    default_map = os.path.join(SCRIPT_DIR, "level1_ground_truth.txt")
    default_list = os.path.join(SCRIPT_DIR, "shopping_list.txt")
    default_yolo = os.path.join(SCRIPT_DIR, "YOLO", "model", "bestv5.pt")
    default_calib = os.path.join(REPO_ROOT, "Week02-04", "calibration", "param") + os.sep

    parser = argparse.ArgumentParser(
        description="Level 1 autonomous navigation with full ground-truth map."
    )
    parser.add_argument("--ip", type=str, default="192.168.50.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default=default_calib)
    parser.add_argument("--yolo_model", type=str, default=default_yolo)
    parser.add_argument("--map", type=str, default=default_map,
                        help="Ground-truth JSON generated from SLAM + targets.")
    parser.add_argument("--list", type=str, default=default_list,
                        help="Shopping list of target fruits (one per line).")
    parser.add_argument("--grid_res", type=float, default=0.02)
    parser.add_argument("--robot_radius", type=float, default=0.14)
    parser.add_argument("--safety_margin", type=float, default=0.18)
    parser.add_argument("--dist_tol", type=float, default=0.15,
                        help="Distance tolerance when declaring waypoint reached (m).")
    parser.add_argument("--angle_tol_deg", type=float, default=10.0,
                        help="Heading tolerance before engaging forward motion (deg).")
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--save_data", action='store_true')
    return parser.parse_args()


def prepare_operate(args: argparse.Namespace) -> SimpleNamespace:
    op_args = SimpleNamespace(
        ip=args.ip,
        port=args.port,
        calib_dir=args.calib_dir,
        yolo_model=args.yolo_model,
        play_data=args.play_data,
        save_data=args.save_data,
    )
    # Provide globals expected by Week05-06/operate.py
    wednesday_mod.operate_mod.args = op_args  # type: ignore[attr-defined]

    # Fonts/icons for GUI
    pygame.font.init()
    try:
        title_font = pygame.font.Font(
            os.path.join(WEEK0506_DIR, "pics", "8-BitMadness.ttf"), 35
        )
        text_font = pygame.font.Font(
            os.path.join(WEEK0506_DIR, "pics", "8-BitMadness.ttf"), 40
        )
    except Exception:
        title_font = pygame.font.SysFont(None, 28)
        text_font = pygame.font.SysFont(None, 32)
    wednesday_mod.operate_mod.TITLE_FONT = title_font  # type: ignore[attr-defined]
    wednesday_mod.operate_mod.TEXT_FONT = text_font  # type: ignore[attr-defined]

    return op_args


def main() -> None:
    args = parse_args()
    op_args = prepare_operate(args)

    pygame.init()
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption("ECE4078 - Level 1 Ground Truth Autonomy")
    try:
        pygame.display.set_icon(
            pygame.image.load(os.path.join(WEEK0506_DIR, "pics", "8bit", "pibot5.png"))
        )
    except Exception:
        pass
    canvas.fill((0, 0, 0))

    with open(args.map, "r", encoding="ascii") as fd:
        ground_truth_dict = json.load(fd)

    fruit_list, fruit_pos, aruco_pos = read_true_map_robust(args.map)
    search_list = load_search_list(args.list)
    print_target_fruits_pos(search_list, fruit_list, fruit_pos)

    targets_xy, obstacles_xy = build_targets_and_obstacles(
        fruit_list,
        fruit_pos,
        aruco_pos,
        search_list,
    )

    # Ensure Week05-06 relative asset paths resolve for operate.py
    prev_cwd = os.getcwd()
    os.chdir(WEEK0506_DIR)

    operate = AutoOperateLevel1(
        op_args,
        search_list=search_list,
        targets_xy=targets_xy,
        known_obstacles_xy=obstacles_xy,
        grid_res=args.grid_res,
        robot_radius=args.robot_radius,
        safety_margin=args.safety_margin,
        map_fruit_labels=list(fruit_list),
        map_fruit_xy=fruit_pos.tolist() if fruit_pos.size else [],
        full_ground_truth=ground_truth_dict,
    )
    operate.dist_tol = args.dist_tol
    operate.angle_tol = np.radians(args.angle_tol_deg)

    try:
        if hasattr(operate, "ekf") and operate.ekf is not None:
            operate.ekf.fixed_aruco_pos = np.array(aruco_pos, dtype=float)
            operate.ekf.lock_aruco = True
    except Exception:
        pass

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
        running = not getattr(operate, "quit", False)

    pygame.quit()
    os.chdir(prev_cwd)


if __name__ == "__main__":
    main()
