# New version with obstacle detection will replace original after verification.
"""Autonomous patrol over four fixed points using pulsed navigation + A* + online
obstacle discovery (fruits) adapted from `pulsing.py`.

Differences vs `pulsing.py`:
 - No partial map / shopping list. Targets are fixed world coords:
       (1,0) -> (0,1) -> (-1,0) -> (0,-1)
 - ArUco markers are ONLY used for SLAM localisation. (Optional: treat them
   as obstacles if desired.)
 - Any YOLO detection (except aruco*) within range is treated as an obstacle.
 - After visiting all four patrol points the robot stops.

Assumptions:
 - Robot world frame origin (0,0) is close to starting pose when SLAM begins.
 - arena size 2.4m square (virtual inner walls used for planning safety).

You can tweak CLI parameters (grid_res, safety margins, etc.) if needed.
"""

from __future__ import annotations

import os
import sys
import time
import math
import argparse
import json
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import pygame

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
WEEK0506_DIR = os.path.join(REPO_ROOT, "Week05-06")

# Import Week05-06 operate (GUI + SLAM + detector)
sys.path.insert(0, WEEK0506_DIR)
try:
    import operate as operate_mod  # type: ignore
    from operate import Operate    # type: ignore
except Exception:  # pragma: no cover - fallback dynamic import
    import importlib.util
    _op_file = os.path.join(WEEK0506_DIR, "operate.py")
    _spec = importlib.util.spec_from_file_location("operate", _op_file)
    operate_mod = importlib.util.module_from_spec(_spec)  # type: ignore
    assert _spec and _spec.loader
    _spec.loader.exec_module(operate_mod)  # type: ignore
    Operate = operate_mod.Operate  # type: ignore

from astar_planning import plan_waypoints


def normalize_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def angle_diff(a: float, b: float) -> float:
    return normalize_angle(a - b)


class AutoPatrol(Operate):
    """Autonomous patrol controller (derives from pulsing.AutoOperateDynamic subset).

    Patrol behaviour:
      1. Wait until SLAM started (ENTER) and at least 2 ArUcos seen (scanning spin).
      2. Plan A* path through remaining patrol points with dynamic obstacle list.
      3. Pulsed turn / drive motion control to follow waypoint list.
      4. On arrival at a patrol point: hold -> short reverse -> proceed to next.
      5. Object detections become obstacles (within range) triggering replans.
      6. Optional: incorporate currently estimated ArUco marker positions as obstacles
         (using --aruco_as_obstacles flag).
    """

    def __init__(self, args, patrol_targets: List[List[float]],
                 grid_res: float, robot_radius: float, safety_margin: float,
                 merge_threshold: float = 0.5, obs_max_range: float = 0.40,
                 aruco_as_obstacles: bool = False,
                 enable_cov_spin: bool = True,
                 enable_periodic_calib: bool = True,
                 calib_interval: float = 10.0):
        super().__init__(args)

        # Continuous detector
        self.command['inference'] = True

        # Planning / target state
        self.remaining_targets: List[List[float]] = [list(t) for t in patrol_targets]
        self.waypoints: List[List[float]] = []
        self.current_goal: List[float] | None = None
        self.active = True if self.remaining_targets else False

        # Obstacles
        self.known_obstacles: List[List[float]] = []  # (optionally updated with ArUcos)
        self.discovered_obstacles: List[dict] = []     # dicts with x,y,label,count
        self.aruco_as_obstacles = aruco_as_obstacles

        # A* parameters
        self.grid_res = grid_res
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin

        # Motion control params (pulsed similar to pulsing.py)
        self.dist_tol = 0.08
        self.angle_tol = math.radians(8.0)
        self.turn_cmd = 1
        self.fwd_cmd = 1

        # Arrival / reverse sequence
        self.reached_time: float | None = None
        self.hold_duration = 2.0
        self.reverse_duration = 0.4
        self._reverse_until: float | None = None
        self._pending_complete_after_reverse = False

        # Scanning for initial markers
        self._scan_start: float | None = None
        self._scan_dir = 1

        # Pulsed nav state
        self.nav_turn_pulse_spin_time = 0.4
        self.nav_turn_pulse_stop_time = 0.2
        self.nav_drive_pulse_period = 0.55
        self.nav_drive_pulse_stop_time = 0.2
        self._nav_turn_pulse_start: float | None = None
        self._nav_drive_pulse_start: float | None = None
        self._nav_last_mode: str | None = None

        # Obstacle detection / merging
        self.last_obstacle_add_time = 0.0
        self.add_cooldown = 0.5
        self.min_obs_separation = 0.15
        self.merge_threshold = float(merge_threshold)
        self.merge_threshold_non_target = self.merge_threshold + 0.2
        self.obs_max_range = float(obs_max_range)

        # Virtual walls (arena 2.4m; keep-out 0.10m)
        self.arena_half = 1.20
        self.wall_clearance = 0.10

        # Camera intrinsics for projection heuristic
        self.K = getattr(self.ekf.robot, 'camera_matrix', None)
        self.cx = float(self.K[0, 2]) if self.K is not None else 160.0
        self.fx = float(self.K[0, 0]) if self.K is not None else 320.0

        # Logging (pose, plans, obstacles) similar to pulsing
        self._log = {
            'meta': {
                'patrol_targets': [list(t) for t in patrol_targets],
                'grid_res': self.grid_res,
                'robot_radius': self.robot_radius,
                'safety_margin': self.safety_margin,
                'merge_threshold': self.merge_threshold,
                'obs_max_range': self.obs_max_range,
                'enable_cov_spin': enable_cov_spin,
                'enable_periodic_calib': enable_periodic_calib,
                'calib_interval': calib_interval,
            },
            'poses': [],   # [t,x,y,th]
            'plans': [],   # {t, waypoints:[...]}
            'obstacles': []  # {t,x,y,label,method}
        }
        self._last_pose_log = 0.0
        self._last_flush = 0.0
        week0708_dir = os.path.join(REPO_ROOT, 'Week07-08')
        log_dir = os.path.join(week0708_dir, 'lab_output')
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass
        self._log_path = os.path.join(log_dir, 'patrol_log.json')

        # High covariance spin stabilization parameters (optional)
        self.enable_cov_spin = enable_cov_spin
        self.cov_pos_thresh = 0.14  # trigger threshold on P[0,0]
        self.cov_spin_duration = 9.0
        self.cov_spin_cooldown = 3.0
        self._cov_spin_until: float | None = None
        self._cov_cooldown_until = 0.0
        self._cov_spin_dir = 1
        self.cov_pulse_spin_time = 0.4
        self.cov_pulse_stop_time = 0.2
        self._cov_spin_start: float | None = None

        # Periodic calibration scan (optional)
        self.enable_periodic_calib = enable_periodic_calib
        self.calib_interval = float(calib_interval)
        self.last_calib_time = time.time()
        self._calib_mode = False
        self._calib_scan_start: float | None = None
        self._calib_scan_dir = 1
        self.calib_rotate_speed = 0.6
        self.calib_timeout = 6.0

        # Fonts for labels
        try:
            self.label_font = pygame.font.SysFont(None, 14)
        except Exception:  # pragma: no cover
            self.label_font = None

    # ---------------- Utility ----------------
    def get_pose(self) -> Tuple[float, float, float]:
        if hasattr(self, 'ekf') and self.ekf is not None:
            robot = getattr(self.ekf, 'robot', None)
            if robot is not None and hasattr(robot, 'state') and robot.state.shape[0] >= 3:
                return float(robot.state[0, 0]), float(robot.state[1, 0]), float(robot.state[2, 0])
        return 0.0, 0.0, 0.0

    def pick_next_goal(self):
        if not self.waypoints:
            self.current_goal = None
            return
        self.current_goal = self.waypoints.pop(0)
        self.reached_time = None
        self.notification = f'Going to: [{self.current_goal[0]:.2f},{self.current_goal[1]:.2f}]'
        self._log_plan()

    # ---------------- Drawing augmentations ----------------
    def draw(self, canvas):  # override to overlay waypoints + obstacles
        canvas = super().draw(canvas)
        v_pad = 40
        h_pad = 20
        slam_origin = (2 * h_pad + 320, v_pad)
        ekf_view = self.ekf.draw_slam_state(res=(320, 480 + v_pad), not_pause=self.ekf_on)

        def to_im(xy):
            m2pixel = 100
            w, h = (320, 480 + v_pad)
            x, y = xy
            return int(-x * m2pixel + w / 2.0), int(y * m2pixel + h / 2.0)

        rx, ry, _ = self.get_pose()
        # Draw current remaining waypoint segment (blue)
        if self.waypoints:
            pts = [to_im((0, 0))]
            for wx, wy in self.waypoints[:30]:  # limit for render performance
                pts.append(to_im((wx - rx, wy - ry)))
            for i in range(len(pts) - 1):
                pygame.draw.line(ekf_view, (50, 120, 230), pts[i], pts[i + 1], 2)
            for p in pts[1:]:
                pygame.draw.circle(ekf_view, (50, 120, 230), p, 3)

        # Draw obstacles (red X)
        for d in self.discovered_obstacles:
            try:
                ox, oy = float(d['x']), float(d['y'])
            except Exception:
                continue
            px, py = to_im((ox - rx, oy - ry))
            pygame.draw.line(ekf_view, (220, 50, 50), (px - 4, py - 4), (px + 4, py + 4), 2)
            pygame.draw.line(ekf_view, (220, 50, 50), (px - 4, py + 4), (px + 4, py - 4), 2)

        # Virtual walls
        inner = max(0.0, float(self.arena_half) - float(self.wall_clearance))
        if inner > 0:
            rect = [(-inner, -inner), (inner, -inner), (inner, inner), (-inner, inner)]
            rect_pts = [to_im((wx - rx, wy - ry)) for (wx, wy) in rect]
            rect_pts.append(rect_pts[0])
            for i in range(len(rect_pts) - 1):
                pygame.draw.line(ekf_view, (120, 180, 120), rect_pts[i], rect_pts[i + 1], 2)

        canvas.blit(ekf_view, slam_origin)
        return canvas

    # ---------------- Logging helpers ----------------
    def _log_pose(self, now: float | None = None):
        try:
            t = time.time() if now is None else now
            x, y, th = self.get_pose()
            self._log['poses'].append([t, float(x), float(y), float(th)])
        except Exception:
            pass

    def _log_plan(self):
        try:
            self._log['plans'].append({'t': time.time(), 'waypoints': [list(wp) for wp in (self.waypoints or [])]})
        except Exception:
            pass

    def _log_obstacle(self, x: float, y: float, label: str, method: str):
        try:
            self._log['obstacles'].append({'t': time.time(), 'x': float(x), 'y': float(y), 'label': str(label), 'method': method})
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

    # ---------------- Calibration helpers ----------------
    def start_calib_scan(self):
        if not self.enable_periodic_calib:
            return
        if self._calib_mode:
            return
        self._calib_mode = True
        self._calib_scan_start = time.time()
        self._calib_scan_dir = 1
        self.notification = 'Starting periodic ArUco calibration scan...'

    def _perform_calib_scan_step(self):
        now = time.time()
        if (now - (self._calib_scan_start or now)) > self.calib_timeout:
            self._calib_mode = False
            self.last_calib_time = now
            self.command['motion'] = [0, 0]
            self.notification = 'Calibration scan timeout'
            return
        elapsed = now - (self._calib_scan_start or now)
        self._calib_scan_dir = 1 if int(elapsed // 2) % 2 == 0 else -1
        self.command['motion'] = [0, self._calib_scan_dir * self.calib_rotate_speed]
        taglist = getattr(self.ekf, 'taglist', []) or []
        if len(taglist) > 0:
            self._calib_mode = False
            self.last_calib_time = now
            self.command['motion'] = [0, 0]
            self.notification = 'Calibration complete'

    # ---------------- Planning ----------------
    def replan(self):
        if not self.active or not self.remaining_targets:
            self.waypoints = []
            self.current_goal = None
            return

        # Optionally keep an updated copy of ArUco marker positions as static obstacles
        if self.aruco_as_obstacles and hasattr(self.ekf, 'markers'):
            try:
                self.known_obstacles = [[float(x), float(y)] for x, y in self.ekf.markers.T]
            except Exception:
                pass

        x, y, _ = self.get_pose()
        robot_xy = [x, y]
        obstacles_xy = list(self.known_obstacles) + [[float(d['x']), float(d['y'])] for d in self.discovered_obstacles]

        # Virtual wall sampling
        inner = max(0.0, float(self.arena_half) - float(self.wall_clearance))
        if inner > 0:
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
            new_wps = plan_waypoints(robot_xy, self.remaining_targets, obstacles_xy,
                                      grid_res=self.grid_res,
                                      robot_radius=self.robot_radius,
                                      safety_margin=self.safety_margin)
            self.waypoints = new_wps
            self.current_goal = None
            self.notification = f'Planned {len(self.waypoints)} wps'
        except Exception as e:  # pragma: no cover
            self.notification = f'Plan fail: {e}'
        else:
            self._log_plan()
            self._flush_log(force=False)

    def _advance_target(self):
        if not self.remaining_targets:
            return
        self.remaining_targets.pop(0)
        self.reached_time = None
        if not self.remaining_targets:
            self.active = False
            self.notification = 'Patrol complete'
            self.waypoints = []
            self.current_goal = None

    def _is_close_to_current_target(self, goal_xy: List[float]) -> bool:
        if not self.remaining_targets:
            return False
        tx, ty = self.remaining_targets[0]
        return math.hypot(goal_xy[0] - tx, goal_xy[1] - ty) <= max(0.12, self.grid_res * 3)

    # ---------------- Perception -> dynamic obstacles ----------------
    def periodic_perception_update(self):
        bboxes = getattr(self, 'detector_output', None)
        if not isinstance(bboxes, (list, tuple)) or len(bboxes) == 0:
            return
        now = time.time()
        new_added = False
        x, y, th = self.get_pose()
        cx, fx = self.cx, self.fx

        for det in bboxes:
            try:
                label = str(det[0]).lower()
                xywh = np.asarray(det[1]).astype(float)
                conf = float(det[2])
            except Exception:
                continue
            if conf < 0.8:
                continue
            if label.startswith('aruco'):
                continue  # ignore markers as obstacles

            # Simple projection (center u + width -> bearing & depth heuristic)
            u = float(xywh[0])
            w_px = float(xywh[2])
            alpha = math.atan((u - cx) / max(1e-6, fx))
            bearing = th + alpha
            W_assumed = 0.10
            if w_px <= 1.0:
                d_est = 0.5
            else:
                d_est = max(0.35, min(1.10, (fx * W_assumed) / w_px))
            ox = x + d_est * math.cos(bearing)
            oy = y + d_est * math.sin(bearing)

            if math.hypot(ox - x, oy - y) > self.obs_max_range:
                continue

            # Merge with existing same-label obstacle clusters
            merged = False
            use_merge_thr = self.merge_threshold_non_target
            for d in self.discovered_obstacles:
                if d.get('label') == label:
                    px, py = float(d['x']), float(d['y'])
                    if math.hypot(ox - px, oy - py) <= use_merge_thr:
                        cnt = int(d.get('count', 1))
                        d['x'] = (px * cnt + ox) / (cnt + 1)
                        d['y'] = (py * cnt + oy) / (cnt + 1)
                        d['count'] = cnt + 1
                        merged = True
                        break
            if merged:
                continue

            # Duplicate proximity check (any label)
            all_obs = []
            all_obs.extend(self.known_obstacles)
            all_obs.extend([[d['x'], d['y']] for d in self.discovered_obstacles])
            if any(math.hypot(ox - qx, oy - qy) <= self.min_obs_separation for qx, qy in all_obs):
                continue
            if now - self.last_obstacle_add_time < self.add_cooldown:
                continue

            self.discovered_obstacles.append({'x': float(ox), 'y': float(oy), 'label': label, 'count': 1})
            self.last_obstacle_add_time = now
            new_added = True

        if new_added and self.ekf_on and self.active:
            self.replan()
            # Log each new obstacle
            try:
                self._log_obstacle(ox, oy, label=label, method='heuristic')  # last ox,oy,label in loop scope
            except Exception:
                pass
            self._flush_log(force=False)

    # ---------------- Main autonomous step ----------------
    def auto_nav_step(self):
        if not self.ekf_on:
            self.command['motion'] = [0, 0]
            self.notification = 'Press ENTER to start SLAM'
            return

        # High covariance stabilize spin (preempts other actions)
        if self.enable_cov_spin:
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
                        self.notification = 'High covariance: pulse stop'
                    return
                if self._cov_spin_until is not None and now_cov >= self._cov_spin_until:
                    self._cov_spin_until = None
                    self._cov_spin_start = None
                    self._cov_cooldown_until = now_cov + self.cov_spin_cooldown
                if now_cov >= self._cov_cooldown_until:
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

        # Periodic calibration trigger
        if self.enable_periodic_calib:
            try:
                if (time.time() - self.last_calib_time) >= self.calib_interval and not self._calib_mode:
                    self.start_calib_scan()
            except Exception:
                pass
        if getattr(self, '_calib_mode', False):
            self._perform_calib_scan_step()
            return

    # Scan for initial ArUcos (need at least 2) before starting patrol path
        tag_count = len(getattr(self.ekf, 'taglist', []))
        now = time.time()
        if tag_count < 2:
            if self._scan_start is None:
                self._scan_start = now
            elapsed = now - (self._scan_start or now)
            self._scan_dir = 1 if int(elapsed // 2) % 2 == 0 else -1
            self.command['motion'] = [0, self._scan_dir * self.turn_cmd]
            self.notification = 'Scanning for markers'
            return
        else:
            self._scan_start = None

        # Log pose at ~5Hz and flush
        if (now - self._last_pose_log) >= 0.2:
            self._log_pose(now)
            self._last_pose_log = now
            self._flush_log(force=False)

        # Reverse handling
        if self._reverse_until is not None:
            if now < self._reverse_until:
                self.command['motion'] = [-self.fwd_cmd, 0]
                self.notification = 'Reversing'
                return
            else:
                self._reverse_until = None
                if self._pending_complete_after_reverse:
                    self._pending_complete_after_reverse = False
                    self._advance_target()
                    if self.active:
                        self.replan()
                        self.pick_next_goal()

        # Ensure we have a plan
        if self.active and (not self.waypoints):
            self.replan()
        if self.current_goal is None and self.waypoints:
            self.pick_next_goal()
        if not self.current_goal:
            self.command['motion'] = [0, 0]
            return

        # Control to current waypoint
        x, y, th = self.get_pose()
        gx, gy = self.current_goal
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        dheading = angle_diff(bearing, th)

        # Arrival processing
        if dist <= self.dist_tol:
            if self._is_close_to_current_target([gx, gy]):
                if self.reached_time is None:
                    self.reached_time = time.time()
                    self.notification = f'At patrol point [{gx:.2f},{gy:.2f}]'
                self.command['motion'] = [0, 0]
                if time.time() - self.reached_time >= self.hold_duration:
                    self._reverse_until = time.time() + self.reverse_duration
                    self._pending_complete_after_reverse = True
                    self.command['motion'] = [-self.fwd_cmd, 0]
                    self.notification = 'Reversing'
                return
            else:
                # Intermediate waypoint
                self.pick_next_goal()
                return

        # Turn then drive (pulsed)
        turning = abs(dheading) > self.angle_tol
        if turning:
            if self._nav_last_mode != 'turn':
                self._nav_last_mode = 'turn'
                self._nav_turn_pulse_start = time.time()
            t_period = float(self.nav_turn_pulse_spin_time + self.nav_turn_pulse_stop_time)
            phase = (time.time() - (self._nav_turn_pulse_start or 0.0)) % t_period
            if phase < self.nav_turn_pulse_spin_time:
                self.command['motion'] = [0, self.turn_cmd if dheading > 0 else -self.turn_cmd]
            else:
                self.command['motion'] = [0, 0]
        else:
            if self._nav_last_mode != 'drive':
                self._nav_last_mode = 'drive'
                self._nav_drive_pulse_start = time.time()
            d_period = float(self.nav_drive_pulse_period)
            d_phase = (time.time() - (self._nav_drive_pulse_start or 0.0)) % d_period
            if d_phase < (self.nav_drive_pulse_period - self.nav_drive_pulse_stop_time):
                self.command['motion'] = [self.fwd_cmd, 0]
            else:
                self.command['motion'] = [0, 0]


def build_patrol_targets() -> List[List[float]]:
    return [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Autonomous 4-point patrol (no map) with dynamic obstacle avoidance")
    parser.add_argument("--ip", type=str, default="192.168.50.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default=os.path.join(REPO_ROOT, "Week02-04", "calibration", "param") + os.sep)
    parser.add_argument("--yolo_model", default=os.path.join(SCRIPT_DIR, "YOLO", "model", "bestv5.pt"))
    parser.add_argument("--grid_res", type=float, default=0.02)
    parser.add_argument("--robot_radius", type=float, default=0.10)
    parser.add_argument("--safety_margin", type=float, default=0.10)
    parser.add_argument("--merge_threshold", type=float, default=0.50)
    parser.add_argument("--obs_max_range", type=float, default=0.45)
    parser.add_argument("--aruco_as_obstacles", action='store_true', help="Treat currently estimated ArUco positions as obstacles")
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--save_data", action='store_true')
    args, _ = parser.parse_known_args()

    # Provide globals expected by Week05-06/operate
    op_args = SimpleNamespace(
        ip=args.ip, port=args.port, calib_dir=args.calib_dir,
        yolo_model=args.yolo_model, play_data=args.play_data, save_data=args.save_data,
    )
    operate_mod.args = op_args

    # Fonts/icons for GUI
    pygame.font.init()
    TITLE_FONT = pygame.font.Font(os.path.join(WEEK0506_DIR, 'pics', '8-BitMadness.ttf'), 35)
    TEXT_FONT = pygame.font.Font(os.path.join(WEEK0506_DIR, 'pics', '8-BitMadness.ttf'), 40)
    operate_mod.TITLE_FONT = TITLE_FONT
    operate_mod.TEXT_FONT = TEXT_FONT

    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 - 4 Point Patrol')
    try:
        pygame.display.set_icon(pygame.image.load(os.path.join(WEEK0506_DIR, 'pics', '8bit', 'pibot5.png')))
    except Exception:
        pass
    canvas.fill((0, 0, 0))

    # Ensure asset relative paths work
    try:
        os.chdir(WEEK0506_DIR)
    except Exception:
        pass

    patrol_targets = build_patrol_targets()
    operate = AutoPatrol(op_args, patrol_targets,
                         grid_res=args.grid_res,
                         robot_radius=args.robot_radius,
                         safety_margin=args.safety_margin,
                         merge_threshold=args.merge_threshold,
                         obs_max_range=args.obs_max_range,
                         aruco_as_obstacles=args.aruco_as_obstacles)

    running = True
    while running:
        operate.update_keyboard()
        operate.take_pic()
        operate.auto_nav_step()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        operate.detect_target()  # populate detector_output
        operate.periodic_perception_update()
        operate.draw(canvas)
        pygame.display.update()
