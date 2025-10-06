import os
import sys
import argparse
import time
import math
import json
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
try:
    # Prefer direct import if Week05-06 is on sys.path (it is added above)
    from TargetPoseEst import estimate_pose  # type: ignore
except Exception:
    estimate_pose = None  # will fallback to heuristic projection
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
                 robot_radius: float, safety_margin: float, merge_threshold: float = 0.50,
                 obs_max_range: float = 0.30,
                 map_fruit_labels: List[str] | None = None,
                 map_fruit_xy: List[List[float]] | None = None):
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
        # discovered obstacles stored as dicts for safety/alignment
        # each item: {'x': float, 'y': float, 'label': str, 'count': int}
        self.discovered_obstacles: List[dict] = []

        # Lock EKF landmarks to fixed ArUco positions from the partial map
        try:
            if hasattr(self, 'ekf') and self.ekf is not None:
                # Ensure a (10,2) float array if possible
                ap = np.array(self.known_obstacles, dtype=float)
                if ap.ndim == 2 and ap.shape[1] == 2:
                    self.ekf.fixed_aruco_pos = ap
                    self.ekf.lock_aruco = True
        except Exception:
            pass

        # Map fruit ground truth (labels and positions) to suppress obstacle adds near known fruits
        self.map_fruit_labels: List[str] = [str(s).lower() for s in (map_fruit_labels or [])]
        self.map_fruit_xy: List[List[float]] = [list(p) for p in (map_fruit_xy or [])]
        # Consider detection as known fruit if within this distance of a mapped fruit of same label
        self.map_match_tol = 0.35

        # A* params
        self.grid_res = grid_res
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin

        # Controller params (mirror Level 2)
        self.waypoints: List[List[float]] = []
        self.current_goal: List[float] | None = None
        self.reached_time: float | None = None
        self.active = True
        # waypoint arrival tolerance (meters)
        self.dist_tol = 0.075
        self.angle_tol = math.radians(8.0)
        # Make autonomous motions slower
        self.turn_cmd = 1
        self.fwd_cmd = 1

        # Marker acquisition (scan/creep) state
        self._scan_start = None
        self._scan_dir = 1
        self._creep_until = None
        self._planned_once = False

        # Arrival reverse behavior
        self.hold_duration = 3.0
        self.reverse_duration = 0.5
        self._reverse_until = None
        self._pending_complete_after_reverse = False

        # Detection handling
        self.last_obstacle_add_time = 0.0
        self.add_cooldown = 0.5  # seconds
        self.min_obs_separation = 0.15  # m
        # Merge detections of the same obstacle label within this radius (m)
        self.merge_threshold = float(merge_threshold)
        # Larger merge threshold for non-target fruits to cluster more aggressively
        self.merge_threshold_non_target = float(self.merge_threshold) + 0.20
        # Only consider detections as obstacles if within this distance (m) from the robot
        self.obs_max_range = float(obs_max_range)
        # Arena virtual walls (2.4x2.4m centered at origin) with 10cm keep-out
        self.arena_half = 1.20
        self.wall_clearance = 0.10

        # Cache intrinsics (for projection of bbox -> world)
        self.K = getattr(self.ekf.robot, 'camera_matrix', None)
        self.cx = float(self.K[0, 2]) if self.K is not None else 160.0
        self.fx = float(self.K[0, 0]) if self.K is not None else 320.0

        # Covariance-based stabilize spin parameters
        self.cov_pos_thresh = 0.09        # trigger threshold on P[0,0]
        self.cov_spin_duration = 9.0      # seconds to spin when triggered (increased from 6s)
        self.cov_spin_cooldown = 4.0      # seconds to wait before checking again
        self._cov_spin_until = None       # type: ignore[assignment]
        self._cov_cooldown_until = 0.0
        self._cov_spin_dir = 1            # alternate spin direction each trigger
        # Pulsed spin timing: spin for 0.4s, stop for 0.2s, repeat
        self.cov_pulse_spin_time = 0.4
        self.cov_pulse_stop_time = 0.2
        self._cov_spin_start = None       # type: ignore[assignment]

        # Navigation pulse timing (normal turn/drive):
        # - Turning: spin 0.4s, stop 0.2s (same as covariance spin)
        # - Driving forward: period 1.0s with 0.2s stop per second
        self.nav_turn_pulse_spin_time = 0.4
        self.nav_turn_pulse_stop_time = 0.2
        self.nav_drive_pulse_period = 0.55
        self.nav_drive_pulse_stop_time = 0.2
        self._nav_turn_pulse_start = None  # type: ignore[assignment]
        self._nav_drive_pulse_start = None  # type: ignore[assignment]
        self._nav_last_mode = None  # 'turn' | 'drive' | None

        # Small font used for overlay labels (smaller text)
        # pygame.font.init() is called in __main__ before the operate instance is created,
        # so SysFont is available here.
        try:
            self.label_font = pygame.font.SysFont(None, 14)
        except Exception:
            self.label_font = None

        # Calibration (ArUco) periodic scan params
        self.calib_interval =10.0        # seconds between calibration scans (set to 20s)
        self.last_calib_time = time.time()
        self._calib_mode = False         # when True, robot is rotating to look for aruco
        self._calib_scan_start = None
        self._calib_scan_dir = 1
        self.calib_rotate_speed = 0.6    # angular speed used while scanning (tunable)
        self.calib_timeout = 6.0         # seconds to give up if no aruco seen

        # --- Lightweight logging for post-run visualization ---
        self._log = {
            'meta': {
                'search_list': list(self.search_list),
                'targets_xy': [list(t) for t in targets_xy],
                'aruco_obstacles_xy': [list(o) for o in aruco_obstacles_xy],
                'grid_res': self.grid_res,
                'robot_radius': self.robot_radius,
                'safety_margin': self.safety_margin,
                'merge_threshold': self.merge_threshold,
                'obs_max_range': self.obs_max_range,
                'calib_interval': self.calib_interval,
            },
            'poses': [],      # each: [t, x, y, th]
            'plans': [],      # each: {t, waypoints: [[x,y], ...]}
            'obstacles': []   # each: {t, x, y, label, method}
        }
        self._last_pose_log = 0.0
        self._last_flush = 0.0
        # logs go under Week07-08/lab_output regardless of cwd
        week0708_dir = os.path.join(REPO_ROOT, 'Week07-08')
        log_dir = os.path.join(week0708_dir, 'lab_output')
        self._log_path = os.path.join(log_dir, 'auto_nav_log.json')
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass

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

    # --- Live overlay on SLAM panel ---
    def draw(self, canvas):
        # Call base draw first to get standard panels
        super().draw(canvas)

        # Compute where SLAM surface was blitted in base draw
        # Base uses: ekf_view at (2*h_pad + 320, v_pad) with res (320, 480+v_pad)
        v_pad = 40
        h_pad = 20
        slam_origin = (2 * h_pad + 320, v_pad)
        slam_res = (320, 480 + v_pad)

        # We need a surface reference to draw onto; re-generate ekf view here to overlay
        ekf_view = self.ekf.draw_slam_state(res=(320, 480 + v_pad), not_pause=self.ekf_on)

        # Helper: convert world (relative to robot) to image coords used by ekf_view
        def to_im(xy):
            # replicate EKF.to_im_coor behavior with m2pixel=100
            m2pixel = 100
            w, h = (320, 480 + v_pad)
            x, y = xy
            x_im = int(-x * m2pixel + w / 2.0)
            y_im = int(y * m2pixel + h / 2.0)
            return (x_im, y_im)

        # Get robot pose to shift world coords relative to robot
        rx, ry, rth = self.get_pose()

        # Draw planned waypoints to CURRENT target only (blue)
        if self.waypoints and len(self.waypoints) >= 1 and self.remaining_targets:
            current_target = self.remaining_targets[0]
            tx, ty = float(current_target[0]), float(current_target[1])
            stop_tol = max(0.12, self.grid_res * 3)

            pts = []
            pts.append(to_im((0.0, 0.0)))  # robot origin in EKF view
            partial_pts = []
            reached_segment = False
            for wp in self.waypoints:
                wx, wy = float(wp[0]), float(wp[1])
                partial_pts.append((wx, wy))
                # stop when a waypoint reaches vicinity of the current target
                if math.hypot(wx - tx, wy - ty) <= stop_tol:
                    reached_segment = True
                    break

            # If we didn't find a waypoint near the target, draw all we have
            if not partial_pts:
                partial_pts = []
            # Append transformed points
            for wx, wy in partial_pts:
                pts.append(to_im((wx - rx, wy - ry)))

            # Draw if we have at least a segment
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    pygame.draw.line(ekf_view, (40, 90, 220), pts[i], pts[i + 1], 2)
                for p in pts[1:]:
                    pygame.draw.circle(ekf_view, (40, 90, 220), p, 3)

        # Draw known + discovered obstacles (red X)
        for ox, oy in self.known_obstacles:
            px, py = to_im((float(ox) - rx, float(oy) - ry))
            pygame.draw.line(ekf_view, (220, 50, 50), (px - 4, py - 4), (px + 4, py + 4), 2)
            pygame.draw.line(ekf_view, (220, 50, 50), (px - 4, py + 4), (px + 4, py - 4), 2)

        for d in self.discovered_obstacles:
            try:
                ox, oy = float(d['x']), float(d['y'])
            except Exception:
                continue
            px, py = to_im((ox - rx, oy - ry))
            pygame.draw.line(ekf_view, (220, 50, 50), (px - 4, py - 4), (px + 4, py + 4), 2)
            pygame.draw.line(ekf_view, (220, 50, 50), (px - 4, py + 4), (px + 4, py - 4), 2)
            # draw label using the small label font (smaller text)
            try:
                lbl = str(d.get('label', ''))[:8]
                if lbl and self.label_font is not None:
                    lbl_surf = self.label_font.render(lbl, True, (240, 240, 240))
                    ekf_view.blit(lbl_surf, (px + 6, py - 6))
            except Exception:
                pass

        # Draw virtual wall boundary (inner rectangle), if configured
        try:
            inner = max(0.0, float(self.arena_half) - float(self.wall_clearance))
        except Exception:
            inner = 0.0
        if inner > 0.0:
            # Define rectangle corners in world frame, then transform relative to robot pose
            rect_world = [
                (-inner, -inner),
                ( inner, -inner),
                ( inner,  inner),
                (-inner,  inner),
            ]
            rect_pts = [to_im((wx - rx, wy - ry)) for (wx, wy) in rect_world]
            # Close the loop by appending first point at end
            rect_pts.append(rect_pts[0])
            for i in range(len(rect_pts) - 1):
                pygame.draw.line(ekf_view, (120, 180, 120), rect_pts[i], rect_pts[i + 1], 2)

        # --- Calibration / scan status overlay on ekf_view ---
        try:
            if self._calib_mode:
                status_msg = "CALIBRATION SCAN: scanning for ArUco..."
            else:
                # show seconds until next calibration scan
                seconds_left = max(0, int(self.calib_interval - (time.time() - self.last_calib_time)))
                status_msg = f"Next calib in: {seconds_left}s"
            if self.label_font is not None:
                status_surf = self.label_font.render(status_msg, True, (255, 200, 0))
                # place at top-left of ekf_view with small padding
                ekf_view.blit(status_surf, (6, 6))
        except Exception:
            pass

        # Blit the augmented SLAM view back to the main canvas
        canvas.blit(ekf_view, slam_origin)
        return canvas

    # --- Logging helpers ---
    def _log_pose(self, now: float | None = None):
        try:
            t = time.time() if now is None else now
            x, y, th = self.get_pose()
            self._log['poses'].append([t, float(x), float(y), float(th)])
            # Also print the covariance of the robot's position (top-left 2x2 of EKF covariance)
            try:
                P = getattr(self.ekf, 'P', None)
                if isinstance(P, np.ndarray) and P.shape[0] >= 2 and P.shape[1] >= 2:
                    Pxy = P[0:2, 0:2]
                    print("Robot position covariance (P[0:2,0:2]):\n", Pxy)
            except Exception:
                pass
        except Exception:
            pass

    def _log_plan(self):
        try:
            self._log['plans'].append({
                't': time.time(),
                'waypoints': [list(wp) for wp in (self.waypoints or [])]
            })
        except Exception:
            pass

    def _log_obstacle(self, x: float, y: float, label: str, method: str):
        try:
            self._log['obstacles'].append({
                't': time.time(), 'x': float(x), 'y': float(y),
                'label': str(label), 'method': str(method)
            })
        except Exception:
            pass

    def _flush_log(self, force: bool = False):
        try:
            now = time.time()
            if not force and (now - self._last_flush) < 2.0:
                return
            with open(self._log_path, 'w') as f:
                json.dump(self._log, f, indent=2)
            self._last_flush = now
        except Exception:
            pass

    # --- Calibration helpers (new) ---
    def start_calib_scan(self):
        """Begin an in-place rotation scan looking for the first ArUco to calibrate against.
        This pauses regular navigation control (no replanning) until calib finishes or times out.
        """
        if self._calib_mode:
            return
        self._calib_mode = True
        self._calib_scan_start = time.time()
        self._calib_scan_dir = 1
        self.notification = "Starting periodic ArUco calibration scan..."
        # Do not alter waypoints/current_goal — we will resume them after scan.

    def _perform_calib_scan_step(self):
        """Called by auto_nav_step while in calibration mode. Rotates and checks for ArUco tags."""
        now = time.time()
        # Timeout check
        if (now - (self._calib_scan_start or now)) > self.calib_timeout:
            self._calib_mode = False
            self.last_calib_time = now
            self.notification = "Calibration scan timed out — resuming navigation"
            # stop motion immediately
            self.command['motion'] = [0, 0]
            return

        # Alternate rotation direction every ~2s for a sweeping scan
        elapsed = now - (self._calib_scan_start or now)
        self._calib_scan_dir = 1 if int(elapsed // 2) % 2 == 0 else -1
        # Set rotation command (rotate in place)
        self.command['motion'] = [0, self._calib_scan_dir * self.calib_rotate_speed]

        # Check for any ArUco tags visible in EKF taglist
        taglist = getattr(self.ekf, 'taglist', []) or []
        if len(taglist) > 0:
            # take the first tag seen
            tag = taglist[0]
            self._on_aruco_seen_during_calib(tag)
            self._calib_mode = False
            self.last_calib_time = time.time()
            # stop rotation
            self.command['motion'] = [0, 0]
            self.notification = "ArUco seen — calibration finished, resuming path"
            return
        # else keep rotating until timeout

    def _on_aruco_seen_during_calib(self, tag):
        """Handle a detected ArUco during a calibration scan.

        This helper will attempt to use any EKF-facing method to incorporate the observation
        (if your EKF/operate exposes something). If such a method doesn't exist it will log
        the event and rely on normal EKF tag processing (most EKFs already ingest ArUco).
        """
        # Attempt to extract useful info from the tag structure
        try:
            # tag may be a dict with 'id'/'x'/'y' etc, or a tuple/list.
            tag_id = None
            tag_info = None
            if isinstance(tag, dict):
                tag_id = tag.get('id', None)
                tag_info = tag
            elif isinstance(tag, (list, tuple)):
                # common EKF representations: (id, x, y, theta) or (id, pose_dict)
                if len(tag) >= 1:
                    tag_id = tag[0]
                    tag_info = tag
            # Try to call EKF helper if available (best-effort)
            ekf = getattr(self, 'ekf', None)
            if ekf is not None:
                # Look for a method name that might exist in your EKF implementation
                for fname in ('apply_aruco_fix', 'incorporate_aruco', 'register_aruco_observation', 'correct_pose_with_aruco'):
                    if hasattr(ekf, fname):
                        try:
                            getattr(ekf, fname)(tag)  # type: ignore[misc]
                            self.notification = f'Calibrated using ArUco (via ekf.{fname})'
                            return
                        except Exception:
                            pass
            # Fallback: just log the detection and leave EKF to handle it normally
            self._log_obstacle(getattr(tag_info, 'x', 0.0) if isinstance(tag_info, object) else 0.0,
                               getattr(tag_info, 'y', 0.0) if isinstance(tag_info, object) else 0.0,
                               label=f'aruco_{tag_id}', method='calib-detect')
            self._flush_log(force=False)
        except Exception:
            # safe fallback: do nothing special
            pass

    def auto_nav_step(self):
        # Require SLAM
        if not self.ekf_on:
            self.command['motion'] = [0, 0]
            self.notification = 'Press ENTER to start SLAM'
            return

        # --- High covariance stabilize spin (preempts other actions) ---
        try:
            now_cov = time.time()
            # If currently spinning due to high covariance, keep spinning until timeout
            if self._cov_spin_until is not None and now_cov < self._cov_spin_until:
                # Pulsed behavior: rotate for cov_pulse_spin_time then stop for cov_pulse_stop_time
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
            # If a spin just finished, start cooldown timer and resume
            if self._cov_spin_until is not None and now_cov >= self._cov_spin_until:
                self._cov_spin_until = None
                self._cov_spin_start = None
                self._cov_cooldown_until = now_cov + self.cov_spin_cooldown
            # If in cooldown, skip covariance checks
            if now_cov < self._cov_cooldown_until:
                pass  # proceed with normal behavior
            else:
                # Check EKF position covariance P[0,0]
                P = getattr(self.ekf, 'P', None)
                if isinstance(P, np.ndarray) and P.shape[0] >= 2 and P.shape[1] >= 2:
                    pxx = float(P[0, 0])
                    if pxx > float(self.cov_pos_thresh):
                        # trigger spin
                        self._cov_spin_dir = -self._cov_spin_dir  # alternate direction
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
                # Start an in-place rotation and scan; do not change path/waypoints
                self.start_calib_scan()
        except Exception:
            pass

        # If currently in calibration mode, do calib step and return (do not replan or move along path)
        if self._calib_mode:
            self._perform_calib_scan_step()
            return

        # Marker acquisition gate: scan and occasional creep until >=2 tags visible
        tag_count = len(getattr(self.ekf, 'taglist', []))
        now = time.time()

        # Log pose at ~5 Hz and flush log periodically
        if (now - self._last_pose_log) >= 0.2:
            self._log_pose(now)
            self._last_pose_log = now
        self._flush_log(force=False)
        if tag_count < 2:
            if self._scan_start is None:
                self._scan_start = now
                self._scan_dir = 1
            # Rotate only (no creeping forward)
            elapsed = now - self._scan_start
            # Alternate scan direction every ~2s
            self._scan_dir = 1 if int(elapsed // 2) % 2 == 0 else -1
            self.command['motion'] = [0, self._scan_dir * self.turn_cmd]
            self.notification = 'Looking for markers: scanning'
            return
        else:
            # Reset scanning state when we have enough tags
            self._scan_start = None
            self._creep_until = None

        # If finishing a reverse segment after target, finalize completion
        now = time.time()
        if self._reverse_until is not None:
            if now < self._reverse_until:
                # Continue reversing
                self.command['motion'] = [-self.fwd_cmd, 0]
                self.notification = 'Reversing from target'
                return
            else:
                # Reverse finished
                self._reverse_until = None
                if self._pending_complete_after_reverse:
                    self._pending_complete_after_reverse = False
                    # Target considered completed -> advance and replan
                    self._advance_target()
                    self.replan(initial=False)
                    self.pick_next_goal()
                    # Fall through to regular control after completion

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
        gx, gy = self.current_goal
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        dheading = angle_diff(bearing, th)

        # Arrival handling: if this goal is close to the next target, hold; else skip hold
        if dist <= self.dist_tol:
            if self._is_close_to_current_target([gx, gy]):
                if self.reached_time is None:
                    self.reached_time = time.time()
                    self.notification = f'Reached target [{gx:.2f}, {gy:.2f}]. Holding...'
                self.command['motion'] = [0, 0]
                # After holding, perform a brief reverse before completing
                if time.time() - self.reached_time >= self.hold_duration:
                    self._reverse_until = time.time() + self.reverse_duration
                    self._pending_complete_after_reverse = True
                    self.command['motion'] = [-self.fwd_cmd, 0]
                    self.notification = 'Reversing from target'
                return
            else:
                self.pick_next_goal()
                return

        # Turn-then-drive (with pulsed motion)
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
                # rotate in place
                self.command['motion'] = [0, self.turn_cmd if dheading > 0 else -self.turn_cmd]
            else:
                # pulse stop
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
            # drive for (period - stop_time) then hold for stop_time
            if d_phase < (self.nav_drive_pulse_period - self.nav_drive_pulse_stop_time):
                self.command['motion'] = [self.fwd_cmd, 0]
            else:
                self.command['motion'] = [0, 0]

    # ============= Planning =============
    def replan(self, initial: bool = False):
        """Plan waypoints from current pose to remaining targets, avoiding known+discovered obstacles."""
        # If we are calibrating, do not replan (we want to return to the same path)
        if self._calib_mode:
            self.notification = "Calibration active — skipping replan"
            return

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
        # convert discovered dicts to [x,y] for planner
        obstacles_xy = list(self.known_obstacles) + [[float(d['x']), float(d['y'])] for d in self.discovered_obstacles]
        # Add virtual wall obstacles along the inner boundary at ±(arena_half - wall_clearance)
        inner = max(0.0, float(self.arena_half) - float(self.wall_clearance))
        if inner > 0.0:
            step = max(0.02, min(0.10, self.grid_res))
            xs = np.arange(-inner, inner + step, step)
            ys = np.arange(-inner, inner + step, step)
            for xv in xs:
                obstacles_xy.append([float(xv), float(inner)])
                obstacles_xy.append([float(xv), float(-inner)])
            for yv in ys:
                obstacles_xy.append([float(inner), float(yv)])
                obstacles_xy.append([float(-inner), float(yv)])
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
            # Log the new plan
            self._log_plan()
            self._flush_log(force=False)
        except Exception as e:
            self.notification = f'Planning failed: {e}'

    def _advance_target(self):
        if not self.remaining_targets:
            return
        # Remove the front target as completed
        self.remaining_targets.pop(0)
        if hasattr(self, 'remaining_labels') and self.remaining_labels:
            self.remaining_labels.pop(0)
        # Reset timing state when target advances
        self.reached_time = None
        # Ensure the next current target is not blocked by a previously added obstacle
        if self.remaining_targets:
            ntx, nty = self.remaining_targets[0]
            # prune discovered obstacles (and metadata) that overlap the new current target location
            keep_obs: List[dict] = []
            for d in self.discovered_obstacles:
                ox, oy = float(d['x']), float(d['y'])
                if math.hypot(ox - ntx, oy - nty) > max(0.12, self.grid_res * 2):
                    keep_obs.append(d)
            self.discovered_obstacles = keep_obs
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

        # Keep a copy of remaining targets; current target is index 0 (if exists)
        known_targets = self.remaining_targets[:]

        # tolerance for matching detection to a remaining target (meters)
        target_match_tol = 0.35

        for det in bboxes:
            try:
                label: str = str(det[0]).lower()
                xywh = np.asarray(det[1]).astype(float)
                conf = float(det[2])
            except Exception:
                continue
            if conf < 0.8:
                continue

            # Project detection to a world point using TargetPoseEst if available; otherwise fallback to heuristic
            ox, oy = None, None
            used_tpe = False
            try:
                if estimate_pose is not None and self.K is not None:
                    # TargetPoseEst expects obj_info as [label, [x,y,w,h]] and robot_pose as [x,y,theta]
                    obj_info = [label, [float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])]]
                    pose_dict = estimate_pose(self.K, obj_info, [x, y, th])  # type: ignore[arg-type]
                    if pose_dict and 'x' in pose_dict and 'y' in pose_dict:
                        ox = float(pose_dict['x'])
                        oy = float(pose_dict['y'])
                        used_tpe = True
            except Exception:
                # Fallback below
                pass

            if ox is None or oy is None:
                # Fallback rough projection (bearing from u, depth from box width)
                u = float(xywh[0])
                w_px = float(xywh[2])
                alpha = math.atan((u - cx) / max(1e-6, fx))
                bearing = th + alpha
                W_assumed = 0.10
                if w_px <= 1.0:
                    d = 0.5
                else:
                    d = max(0.35, min(1.10, (fx * W_assumed) / w_px))
                ox = x + d * math.cos(bearing)
                oy = y + d * math.sin(bearing)

            # Range gate: only accept detections within obs_max_range from the robot
            if math.hypot(float(ox) - x, float(oy) - y) > self.obs_max_range:
                continue

            # Ignore detections that correspond to the CURRENT target only
            if self.remaining_targets and self.remaining_labels:
                current_label = str(self.remaining_labels[0]).lower()
                tx0, ty0 = self.remaining_targets[0]
                if label == current_label and math.hypot(ox - tx0, oy - ty0) <= target_match_tol:
                    # This is likely the current target; don't add as obstacle
                    continue

            # Avoid false positives extremely close to CURRENT target centre only
            if self.remaining_targets:
                tx0, ty0 = self.remaining_targets[0]
                if math.hypot(ox - tx0, oy - ty0) <= 0.10:
                    continue

            # Skip ArUco markers by label (we don't update known markers)
            if label.startswith('aruco') or label.startswith('aruco_'):
                continue

            # Suppress adding/merging if detection is near a fruit already present in the partial map
            if self.map_fruit_labels and self.map_fruit_xy:
                skip_flag = False
                for (mf_label, (mx, my)) in zip(self.map_fruit_labels, self.map_fruit_xy):
                    if mf_label == label and math.hypot(ox - float(mx), oy - float(my)) <= self.map_match_tol:
                        # Treat as known fruit from map; do not add as obstacle
                        skip_flag = True
                        break
                if skip_flag:
                    continue

            # --- Merge logic using dict-based discovered_obstacles ---
            merged = False
            # Select merge threshold: larger for non-target fruits
            current_label_merge = None
            if self.remaining_targets and self.remaining_labels:
                try:
                    current_label_merge = str(self.remaining_labels[0]).lower()
                except Exception:
                    current_label_merge = None
            use_merge_thr = self.merge_threshold_non_target if (current_label_merge is None or label != current_label_merge) else self.merge_threshold

            for d in self.discovered_obstacles:
                # only merge identical labels
                if d.get('label') == label:
                    px, py = float(d['x']), float(d['y'])
                    if math.hypot(ox - px, oy - py) <= use_merge_thr:
                        # incremental mean update for cluster centre
                        cnt = int(d.get('count', 1))
                        new_x = (px * cnt + ox) / (cnt + 1)
                        new_y = (py * cnt + oy) / (cnt + 1)
                        moved = math.hypot(new_x - px, new_y - py)
                        d['x'] = new_x
                        d['y'] = new_y
                        d['count'] = cnt + 1
                        # log merge (use your existing log helper)
                        self._log_obstacle(new_x, new_y, label=label, method=('merge-tpe' if used_tpe else 'merge-heuristic'))
                        self._flush_log(force=False)
                        if moved > 1e-3:
                            new_added = True
                        merged = True
                        break

            if merged:
                # merged into an existing cluster; skip the rest of add process
                continue

            # Ignore duplicates amongst known and discovered obstacles (any label) using a wider gate
            all_obs = []
            all_obs.extend(self.known_obstacles)
            all_obs.extend([[d['x'], d['y']] for d in self.discovered_obstacles])
            if any(math.hypot(ox - qx, oy - qy) <= self.min_obs_separation for qx, qy in all_obs):
                continue

            # Rate limit
            if now - self.last_obstacle_add_time < self.add_cooldown:
                continue

            # Add obstacle and mark for replanning
            self.discovered_obstacles.append({'x': float(ox), 'y': float(oy), 'label': label, 'count': 1})
            self._log_obstacle(ox, oy, label=label, method=('tpe' if used_tpe else 'heuristic'))
            self._flush_log(force=False)
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
    parser.add_argument("--robot_radius", type=float, default=0.16)
    parser.add_argument("--safety_margin", type=float, default=0.3)
    # default merge threshold increased to 0.50 (50 cm)
    parser.add_argument("--merge_threshold", type=float, default=0.80)
    # only count/add obstacles when seen within this distance (meters)
    parser.add_argument("--obs_max_range", type=float, default=0.48)
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
                                 safety_margin=args.safety_margin,
                                 merge_threshold=args.merge_threshold,
                                 obs_max_range=args.obs_max_range,
                                 map_fruit_labels=list(fruit_list),
                                 map_fruit_xy=[[float(fruit_pos[i, 0]), float(fruit_pos[i, 1])] for i in range(fruit_pos.shape[0])])

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
