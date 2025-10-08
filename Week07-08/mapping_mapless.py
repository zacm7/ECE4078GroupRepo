"""Mapless autonomous exploration + target collection.

This script is a stripped / adapted variant of `mapping_aruco.py` that supports
running *without* prior knowledge of fruit target positions or ArUco marker
locations. It implements:

1. Coverage exploration (simple lawnmower pattern inside a virtual arena).
2. Online detection of target fruits (YOLO detections). When a target fruit from
   the shopping list is repeatedly observed from similar positions (confirmation
   count), its estimated world position is promoted to an actionable target.
3. Dynamic A* planning to that confirmed target while continuing to merge /
   track any additional detected but not-yet-confirmed targets.
4. After reaching a target, the robot resumes exploration to locate remaining
   fruits until the list is exhausted.

Assumptions / Simplifications vs mapping_aruco.py:
-------------------------------------------------
- No initial ArUco obstacle set. (You can extend to add ArUco tag positions as
  obstacles once seen by querying the EKF tag list.)
- ArUco markers are *not* explicitly used for obstacles here; (future work:
  upon detection, append their estimated global poses to obstacle list).
- Fruit labels ONLY in the shopping list are treated as goals (never obstacles).
- Any other detections (unknown labels) within range become obstacles, using
  merging logic similar to the original script.
- Coverage pattern is static lawnmower; you can randomize starting direction
  or implement frontier exploration later.

CLI Example (mapless):
  python mapping_mapless.py --list Week07-08/M3_prac_shopping_list.txt \
        --yolo_model Week07-08/YOLO/model/bestv5.pt

You still need calibration parameters and network args for the underlying
`Operate` class. The `--mapless` flag is implicit (no map file needed).
"""

from __future__ import annotations

import os
import sys
import math
import time
import json
import argparse
from types import SimpleNamespace
from typing import List, Dict, Tuple, Optional

import numpy as np
import pygame

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
WEEK0506_DIR = os.path.join(REPO_ROOT, "Week05-06")

# Import Week05-06 operate.py (GUI + SLAM + detector)
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

# Shared utilities
try:
    from astar_planning import plan_waypoints  # type: ignore
except Exception:
    raise

try:
    from TargetPoseEst import estimate_pose  # type: ignore
except Exception:
    estimate_pose = None


def normalize_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def angle_diff(a: float, b: float) -> float:
    return normalize_angle(a - b)


def load_search_list(path: str) -> List[str]:
    out: List[str] = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(line.lower())
    return out


def generate_lawnmower(inner_half: float, lane_spacing: float, margin: float) -> List[List[float]]:
    """Generate a simple lawnmower (boustrephodon) set of waypoints covering a square region.

    inner_half: half-length of navigable square region (meters)
    lane_spacing: distance between adjacent sweep lanes (meters)
    margin: extra inset from boundary for safety
    """
    usable = max(0.0, inner_half - margin)
    if usable <= 0.0:
        return []
    lanes_y = np.arange(-usable, usable + 1e-6, lane_spacing)
    # Ensure last lane is inside limit
    lanes: List[List[float]] = []
    direction = 1  # 1 -> left->right, -1 -> right->left in x
    xs_full = np.linspace(-usable, usable, max(2, int((2 * usable) / lane_spacing) + 1))
    for y in lanes_y:
        xs = xs_full if direction == 1 else xs_full[::-1]
        for x in xs:
            lanes.append([float(x), float(y)])
        direction *= -1
    return lanes


class AutoOperateMapless(Operate):
    """Autonomous controller for *mapless* operation.

    States:
      - Exploration (coverage path following)
      - Approach (active goal = confirmed target fruit position)

    Fruit acquisition pipeline:
      detection -> (projection world) -> cluster per label -> confirmation (>=N) -> promote to target
    """

    def __init__(self, args, search_list: List[str], grid_res: float, robot_radius: float, safety_margin: float,
                 lane_spacing: float = 0.30, confirm_count: int = 2,
                 merge_thr_target: float = 0.35, merge_thr_obstacle: float = 0.30,
                 obs_max_range: float = 0.60):
        super().__init__(args)

        self.command['inference'] = True

        # Planning parameters
        self.grid_res = grid_res
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin

        # Arena / virtual wall (assume 2.4 x 2.4 m as before)
        self.arena_half = 1.20
        self.wall_clearance = 0.10

        # Search / target bookkeeping
        self.search_order: List[str] = [s.lower() for s in search_list]
        self.remaining_labels: List[str] = list(self.search_order)
        self.confirm_count_required = int(confirm_count)

        # Clusters of potential fruits (per label). Each entry:
        # {'label': str, 'x': float, 'y': float, 'count': int, 'confirmed': bool}
        self.fruit_clusters: List[Dict[str, float | str | int | bool]] = []

        # Obstacles discovered (non-target labels)
        self.discovered_obstacles: List[Dict[str, float | str | int]] = []

        self.merge_thr_target = merge_thr_target
        self.merge_thr_obstacle = merge_thr_obstacle
        self.obs_max_range = obs_max_range
        self.min_obs_separation = 0.15
        self.add_cooldown = 0.5
        self.last_obstacle_add_time = 0.0

        # Active plan / waypoints
        self.waypoints: List[List[float]] = []
        self.current_goal: Optional[List[float]] = None
        self.dist_tol = 0.075
        self.angle_tol = math.radians(8)
        self.turn_cmd = 1
        self.fwd_cmd = 1
        self.mode = 'explore'  # or 'approach'

        # Coverage path (pre-generated)
        inner = max(0.0, self.arena_half - self.wall_clearance)
        self.coverage_waypoints = generate_lawnmower(inner, lane_spacing, margin=0.08)
        self.coverage_index = 0

        # Logging
        self._log = {
            'poses': [],
            'fruit_clusters': [],
            'plans': [],
            'events': [],
            'point_debug': []  # diagnostic logs for step 2 navigation
        }
        week0708_dir = os.path.join(REPO_ROOT, 'Week07-08')
        log_dir = os.path.join(week0708_dir, 'lab_output')
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, 'mapless_log.json')
        self._last_pose_log = 0.0
        self._last_flush = 0.0

        # Cache intrinsics for projection
        self.K = getattr(self.ekf.robot, 'camera_matrix', None) if hasattr(self, 'ekf') else None
        self.cx = float(self.K[0, 2]) if self.K is not None else 160.0
        self.fx = float(self.K[0, 0]) if self.K is not None else 320.0

        # ---- Initial ArUco marker spin step (Step 1) ----
        # Behavior: before any fruit approach, if an ArUco marker becomes visible AND is
        # estimated within self.marker_detect_radius (default 0.35m), stop exploration,
        # rotate in place for self.marker_spin_duration seconds to improve localization.
        self.initial_marker_done = False
        self.marker_detect_radius = 0.35
        # Spin duration increased so we can complete (approx) a full 360 in one direction
        self.marker_spin_duration = 10.0
        self._marker_spin_until = None  # type: Optional[float]
        self._marker_spin_dir = 1
        # Marker approach planning parameters
        self.marker_goal = None  # type: Optional[List[float]]
        self.marker_arrival_tol = 0.20
        self.marker_replan_delta = 0.05  # replan if tag pose shifts more than this
        self.marker_mode = True  # active until initial_marker_done set True
        # Pause window after each marker replan so robot briefly stops before following new path
        self._marker_replan_pause_until = 0.0
        # Debug: track last printed marker position to avoid spamming terminal
        self._last_marker_print = None  # (x,y) tuple
        # Track last plan time to avoid excessive replans
        self.marker_last_plan_time = 0.0

        # ---- Step 2 point goal (after initial marker localization) ----
        # Fixed target point in world frame we drive to once initial marker spin completes.
        # Uses dynamic A* with ALL detected fruits (clusters, confirmed or not) and markers as obstacles.
        self.point_goal_xy: List[float] = [0.0, 1.0]
        self.point_goal_done: bool = False
        self.point_last_plan_time: float = 0.0
        self.point_replan_interval: float = 0.5  # seconds between plans unless obstacle set changed
        self._last_point_obstacle_sig: Optional[int] = None  # signature hash to detect changes

        # Stage machine: 'marker' -> 'point' -> 'explore'
        self.stage = 'marker'
        # Progress tracking for point goal (avoid spinning forever)
        self.point_last_dist: Optional[float] = None
        self.point_last_progress_time = 0.0
        self.point_progress_timeout = 3.0  # seconds without distance improvement triggers creep
        self.point_progress_min_delta = 0.02  # min improvement considered progress
        # Adaptive turn sign diagnostics (Step 2)
        self._last_heading_error_point = None  # type: Optional[float]
        self._was_turning_point = False  # type: bool
        self._invert_turn_sign_point = False  # type: bool

    # ----------------- Marker / point helpers -----------------
    def _get_marker_positions(self) -> List[Tuple[float, float]]:
        """Return current estimated marker positions from EKF (if available)."""
        out: List[Tuple[float, float]] = []
        try:
            taglist = getattr(self.ekf, 'taglist', []) or []
            # First try to extract directly
            for tag in taglist:
                try:
                    if isinstance(tag, dict) and 'x' in tag and 'y' in tag:
                        out.append((float(tag['x']), float(tag['y'])))
                    elif isinstance(tag, (list, tuple)) and len(tag) >= 3 and isinstance(tag[1], (int, float)) and isinstance(tag[2], (int, float)):
                        out.append((float(tag[1]), float(tag[2])))
                except Exception:
                    continue
            # If we only had IDs and found none, try markers array
            if not out and all(isinstance(t, (int, np.integer)) for t in taglist):
                markers_arr = getattr(self.ekf, 'markers', None)
                if isinstance(markers_arr, np.ndarray) and markers_arr.ndim == 2 and markers_arr.shape[0] >= 2:
                    n_cols = markers_arr.shape[1]
                    for idx in range(n_cols):
                        try:
                            tx = float(markers_arr[0, idx])
                            ty = float(markers_arr[1, idx])
                            out.append((tx, ty))
                        except Exception:
                            pass
        except Exception:
            pass
        return out

    def _plan_point_goal(self):
        """Plan path to the fixed point goal using ALL detected fruits + markers as obstacles."""
        x, y, _ = self.get_pose()
        robot_xy = [x, y]
        goal_xy = list(self.point_goal_xy)
        obstacles_xy: List[List[float]] = []
        # Add all fruit clusters (treat fruits as obstacles for this specific point navigation)
        for c in self.fruit_clusters:
            try:
                obstacles_xy.append([float(c['x']), float(c['y'])])
            except Exception:
                continue
        # Add discovered non-target obstacles
        for o in self.discovered_obstacles:
            try:
                obstacles_xy.append([float(o['x']), float(o['y'])])
            except Exception:
                continue
        # Add markers as obstacles
        for mx, my in self._get_marker_positions():
            obstacles_xy.append([mx, my])
        # Virtual wall discretization identical to plan_to_goal
        inner = max(0.0, self.arena_half - self.wall_clearance)
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
            wps = plan_waypoints(robot_xy, [goal_xy], obstacles_xy,
                                  grid_res=self.grid_res,
                                  robot_radius=self.robot_radius,
                                  safety_margin=self.safety_margin)
            self.waypoints = wps
            self.current_goal = None
            self.point_last_plan_time = time.time()
            # Build a simple signature for obstacle set to detect changes
            sig = 0
            for ox, oy in obstacles_xy:
                sig ^= hash((round(ox, 2), round(oy, 2)))
            self._last_point_obstacle_sig = sig
            self.notification = 'Planned path to point (0,1)'
            # Log the point-goal plan for visibility (distinguish via tag)
            try:
                self._log['plans'].append({'t': self.point_last_plan_time, 'waypoints': wps, 'type': 'point_goal'})
            except Exception:
                pass
        except Exception as e:
            self.notification = f'Point planning failed: {e}'

    def _step_point_goal(self):
        """Execute navigation to the fixed point goal; call each cycle until done."""
        if self.point_goal_done:
            return
        rx, ry, rth = self.get_pose()
        gx, gy = self.point_goal_xy
        dist_to_goal = math.hypot(gx - rx, gy - ry)
        # Arrival check
        if dist_to_goal <= self.dist_tol:
            self.point_goal_done = True
            self.waypoints = []
            self.current_goal = None
            self.notification = 'Reached point (0,1)'
            self.stage = 'explore'
            return
        # Decide if we need to (re)plan
        need_plan = False
        now_t = time.time()
        # Obstacle signature current
        obs_sig = 0
        try:
            for mx, my in self._get_marker_positions():
                obs_sig ^= hash((round(mx, 2), round(my, 2)))
            for c in self.fruit_clusters:
                obs_sig ^= hash((round(float(c['x']), 2), round(float(c['y']), 2)))
            for o in self.discovered_obstacles:
                obs_sig ^= hash((round(float(o['x']), 2), round(float(o['y']), 2)))
        except Exception:
            pass
        if self._last_point_obstacle_sig is None or obs_sig != self._last_point_obstacle_sig:
            need_plan = True
        if not self.waypoints or (self.current_goal is None and not self.waypoints):
            need_plan = True
        if (now_t - self.point_last_plan_time) >= self.point_replan_interval:
            need_plan = True
        if need_plan:
            self._plan_point_goal()
        # Ensure current goal set
        if self.current_goal is None:
            if self.waypoints:
                self._pick_next_waypoint()
                self.current_goal_changed_flag = True
            else:
                self.command['motion'] = [0, 0]
                return
        if self.current_goal is None:
            self.command['motion'] = [0, 0]
            return
        # Waypoint control with progress watchdog
        cxg, cyg = self.current_goal
        dx, dy = cxg - rx, cyg - ry
        dist_wp = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        dheading = angle_diff(bearing, rth)

        # Waypoint reached check
        if dist_wp <= self.dist_tol:
            if self.waypoints:
                self._pick_next_waypoint()
                self.current_goal_changed_flag = True
            return

        # Progress tracking (waypoint-centric)
        now_t2 = time.time()
        if not hasattr(self, 'point_last_wp_dist') or getattr(self, 'current_goal_changed_flag', False):
            self.point_last_wp_dist = dist_wp  # type: ignore[attr-defined]
            self.point_last_progress_time = now_t2
            self.current_goal_changed_flag = False
        else:
            prev_wp_dist = getattr(self, 'point_last_wp_dist', dist_wp)
            if (prev_wp_dist - dist_wp) >= self.point_progress_min_delta:
                self.point_last_progress_time = now_t2
                self.point_last_wp_dist = dist_wp  # type: ignore[attr-defined]
            elif dist_wp < prev_wp_dist:
                self.point_last_wp_dist = dist_wp  # type: ignore[attr-defined]
        no_progress = (now_t2 - self.point_last_progress_time) > self.point_progress_timeout
        creep_heading_limit = max(self.angle_tol * 2.0, 0.6)
        force_creep = no_progress and abs(dheading) < creep_heading_limit

        # If moving significantly away from waypoint, trigger replan and cancel creep
        growing_away = False
        if hasattr(self, 'point_last_wp_dist'):
            prev = getattr(self, 'point_last_wp_dist')  # type: ignore[attr-defined]
            if dist_wp - prev > 0.10:
                growing_away = True
                force_creep = False
                self._plan_point_goal()
                if self.current_goal is None and self.waypoints:
                    self._pick_next_waypoint()
                    self.current_goal_changed_flag = True
                if self.current_goal is not None:
                    cxg, cyg = self.current_goal
                    dx, dy = cxg - rx, cyg - ry
                    dist_wp = math.hypot(dx, dy)
                    bearing = math.atan2(dy, dx)
                    dheading = angle_diff(bearing, rth)
                    self.point_last_wp_dist = dist_wp  # type: ignore[attr-defined]
                    self.point_last_progress_time = now_t2

        # Control
        if abs(dheading) > self.angle_tol and not force_creep:
            # Adaptive turn sign: if previous turn step increased error, flip sign mapping
            if self._was_turning_point and self._last_heading_error_point is not None:
                if abs(dheading) > abs(self._last_heading_error_point) + 1e-3:
                    # Diverging -> invert mapping assumption
                    self._invert_turn_sign_point = not self._invert_turn_sign_point
            # Base sign from error
            base_sign = 1 if dheading > 0 else -1
            if self._invert_turn_sign_point:
                base_sign *= -1
            self.command['motion'] = [0, base_sign * self.turn_cmd]
            self.notification = 'Point goal: turning in place'
            self._was_turning_point = True
        else:
            fwd_speed = self.fwd_cmd
            if force_creep and abs(dheading) > self.angle_tol:
                fwd_speed = 0.4 * self.fwd_cmd
            self.command['motion'] = [fwd_speed, 0]
            self.notification = 'Navigating to point (0,1)'
            self._was_turning_point = False
        self._last_heading_error_point = dheading

        # Periodic debug logging
        now_dbg = time.time()
        if not hasattr(self, '_last_point_debug'):
            self._last_point_debug = 0.0  # type: ignore[attr-defined]
        if (now_dbg - getattr(self, '_last_point_debug')) >= 0.75:  # type: ignore[attr-defined]
            dbg = {
                't': now_dbg,
                'robot': [rx, ry, rth],
                'goal': [gx, gy],
                'current_wp': [cxg, cyg],
                'dist_to_goal': round(dist_to_goal, 3),
                'dist_to_wp': round(dist_wp, 3),
                'bearing_wp': round(bearing, 3),
                'heading': round(rth, 3),
                'dheading': round(dheading, 3),
                'force_creep': force_creep,
                'no_progress': no_progress,
                'growing_away': growing_away,
                'cmd': list(self.command['motion'])
            }
            try:
                self._log['point_debug'].append(dbg)
            except Exception:
                pass
            self._last_point_debug = now_dbg  # type: ignore[attr-defined]

    # ----------------- Utility / logging -----------------
    def get_pose(self) -> Tuple[float, float, float]:
        if hasattr(self, 'ekf') and self.ekf is not None:
            robot = getattr(self.ekf, 'robot', None)
            if robot is not None and hasattr(robot, 'state') and robot.state.shape[0] >= 3:
                return float(robot.state[0, 0]), float(robot.state[1, 0]), float(robot.state[2, 0])
        return 0.0, 0.0, 0.0

    def _flush_log(self, force: bool = False):
        now = time.time()
        if not force and (now - self._last_flush) < 2.0:
            return
        try:
            with open(self._log_path, 'w') as f:
                json.dump(self._log, f, indent=2)
            self._last_flush = now
        except Exception:
            pass

    def _log_pose(self, now: Optional[float] = None):
        now = time.time() if now is None else now
        x, y, th = self.get_pose()
        self._log['poses'].append([now, x, y, th])

    # ----------------- Coverage Path Handling -----------------
    def _next_coverage_waypoint(self):
        if not self.coverage_waypoints:
            return None
        if self.coverage_index >= len(self.coverage_waypoints):
            # restart coverage (could shuffle for variation)
            self.coverage_index = 0
        wp = self.coverage_waypoints[self.coverage_index]
        self.coverage_index += 1
        return wp

    # ----------------- Planning -----------------
    def plan_to_goal(self, goal_xy: List[float]):
        x, y, _ = self.get_pose()
        robot_xy = [x, y]
        obstacles_xy: List[List[float]] = []
        # Add dynamic obstacles
        obstacles_xy.extend([[float(o['x']), float(o['y'])] for o in self.discovered_obstacles])
        # Add fruit clusters that are *not* the goal (avoid colliding with fruits) (optional)
        for c in self.fruit_clusters:
            if bool(c.get('confirmed', False)) and [c['x'], c['y']] != goal_xy:
                obstacles_xy.append([float(c['x']), float(c['y'])])
        # Virtual wall discretization
        inner = max(0.0, self.arena_half - self.wall_clearance)
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
            wps = plan_waypoints(robot_xy, [goal_xy], obstacles_xy,
                                  grid_res=self.grid_res,
                                  robot_radius=self.robot_radius,
                                  safety_margin=self.safety_margin)
            self.waypoints = wps
            self.current_goal = None
            self._log['plans'].append({'t': time.time(), 'waypoints': wps})
            self.notification = f'Planned path to {goal_xy}'
        except Exception as e:
            self.notification = f'Planning failed: {e}'

    def _pick_next_waypoint(self):
        if not self.waypoints:
            self.current_goal = None
            return
        self.current_goal = self.waypoints.pop(0)
        self.notification = f'Heading to {self.current_goal}'

    # ----------------- Detection Processing -----------------
    def perception_update(self):
        bboxes = getattr(self, 'detector_output', None)
        if not isinstance(bboxes, (list, tuple)) or len(bboxes) == 0:
            return
        now = time.time()
        x, y, th = self.get_pose()
        cx, fx = self.cx, self.fx
        new_goal_promoted = False
        any_new_obstacle = False

        for det in bboxes:
            try:
                label = str(det[0]).lower()
                xywh = np.asarray(det[1]).astype(float)
                conf = float(det[2])
            except Exception:
                continue
            if conf < 0.8:
                continue

            # Project detection -> world
            wx, wy = None, None
            used_tpe = False
            if estimate_pose is not None and self.K is not None:
                try:
                    obj_info = [label, [float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])]]
                    pose_dict = estimate_pose(self.K, obj_info, [x, y, th])  # type: ignore[arg-type]
                    if pose_dict and 'x' in pose_dict and 'y' in pose_dict:
                        wx = float(pose_dict['x'])
                        wy = float(pose_dict['y'])
                        used_tpe = True
                except Exception:
                    pass
            if wx is None or wy is None:
                u = float(xywh[0])
                w_px = float(xywh[2])
                alpha = math.atan((u - cx) / max(1e-6, fx))
                bearing = th + alpha
                W_assumed = 0.10
                depth = 0.6 if w_px <= 1.0 else max(0.35, min(1.20, (fx * W_assumed) / w_px))
                wx = x + depth * math.cos(bearing)
                wy = y + depth * math.sin(bearing)

            # Range gate
            if math.hypot(wx - x, wy - y) > self.obs_max_range:
                continue

            # If detection label is in remaining target list -> update / merge cluster for that label
            if label in self.remaining_labels:
                # Merge into fruit_clusters
                merged = False
                for c in self.fruit_clusters:
                    if c['label'] == label:
                        d = math.hypot(wx - float(c['x']), wy - float(c['y']))
                        if d <= self.merge_thr_target:
                            cnt = int(c['count'])
                            new_x = (float(c['x']) * cnt + wx) / (cnt + 1)
                            new_y = (float(c['y']) * cnt + wy) / (cnt + 1)
                            c['x'] = new_x
                            c['y'] = new_y
                            c['count'] = cnt + 1
                            if cnt + 1 >= self.confirm_count_required and not c['confirmed']:
                                c['confirmed'] = True
                                # If this label is the *front* of the ordered remaining_labels, promote to active goal
                                if self.remaining_labels and label == self.remaining_labels[0]:
                                    self.mode = 'approach'
                                    self.plan_to_goal([new_x, new_y])
                                    new_goal_promoted = True
                            merged = True
                            break
                if not merged:
                    cdict: Dict[str, float | str | int | bool] = {'label': label, 'x': wx, 'y': wy, 'count': 1, 'confirmed': False}
                    self.fruit_clusters.append(cdict)
                continue  # do not treat targets as obstacles

            # Non-target label -> potential obstacle
            # Duplicate gate
            all_obs = [[float(o['x']), float(o['y'])] for o in self.discovered_obstacles]
            if any(math.hypot(wx - ox, wy - oy) <= self.min_obs_separation for ox, oy in all_obs):
                continue
            if now - self.last_obstacle_add_time < self.add_cooldown:
                continue
            # Merge logic (same label cluster) for obstacles
            merged_obs = False
            for o in self.discovered_obstacles:
                if o.get('label') == label:
                    d = math.hypot(wx - float(o['x']), wy - float(o['y']))
                    if d <= self.merge_thr_obstacle:
                        cnt = int(o.get('count', 1))
                        o['x'] = (float(o['x']) * cnt + wx) / (cnt + 1)
                        o['y'] = (float(o['y']) * cnt + wy) / (cnt + 1)
                        o['count'] = cnt + 1
                        merged_obs = True
                        break
            if merged_obs:
                continue
            self.discovered_obstacles.append({'label': label, 'x': wx, 'y': wy, 'count': 1})
            self.last_obstacle_add_time = now
            any_new_obstacle = True

        if any_new_obstacle and self.mode == 'approach' and self.waypoints:
            # Replan to current confirmed target if obstacles changed
            if self.current_goal is not None:
                goal = list(self.current_goal)
            else:
                # If no current_goal but have waypoints, final waypoint is target
                goal = self.waypoints[-1]
            self.plan_to_goal(goal)
        if new_goal_promoted:
            self.notification = 'Confirmed target; approaching'

    # ----------------- Control Loop -----------------
    def auto_nav_step(self):
        if not self.ekf_on:
            self.command['motion'] = [0, 0]
            self.notification = 'Press ENTER to start SLAM'
            return

        # Log pose
        now = time.time()
        if now - self._last_pose_log >= 0.25:
            self._log_pose(now)
            self._flush_log(False)
            self._last_pose_log = now

        # All targets done?
        if not self.remaining_labels:
            self.command['motion'] = [0, 0]
            self.notification = 'All targets collected'
            return

        # ---- Step 1 (A*): plan to first seen ArUco marker (dynamic replanning) then spin ----
        if self.stage == 'marker' and not self.initial_marker_done:
            rx, ry, rth = self.get_pose()

            # If currently spinning at marker, continue until done
            if self._marker_spin_until is not None:
                if time.time() < self._marker_spin_until:
                    # Single-direction spin (no alternating) for stable 360 capture
                    self.command['motion'] = [0, self._marker_spin_dir * self.turn_cmd]
                    self.notification = 'Marker spin (localizing)'
                    return
                # Spin finished
                self._marker_spin_until = None
                self.initial_marker_done = True
                self.stage = 'point'
                # Reset point-stage progress metrics
                self.point_last_dist = None
                self.point_last_progress_time = time.time()
                if hasattr(self, 'point_last_wp_dist'):
                    delattr(self, 'point_last_wp_dist')
                self.current_goal_changed_flag = False
                self.marker_mode = False
                self.waypoints = []
                self.current_goal = None
                # Fall through to exploration this cycle
            if not self.initial_marker_done:
                taglist = getattr(self.ekf, 'taglist', []) or []
                # Build a unified list of (id, x, y) candidates
                marker_candidates: List[Tuple[int, float, float]] = []
                if taglist:
                    # Case 1: tag objects already carry pose
                    for tag in taglist:
                        try:
                            if isinstance(tag, dict) and 'x' in tag and 'y' in tag:
                                tid = int(tag.get('id', tag.get('tag', -1))) if any(k in tag for k in ('id','tag')) else -1
                                marker_candidates.append((tid, float(tag['x']), float(tag['y'])))
                            elif isinstance(tag, (list, tuple)):
                                # common formats: (id, x, y, ...) OR (id, {dict})
                                if len(tag) >= 3 and isinstance(tag[1], (int, float)) and isinstance(tag[2], (int, float)):
                                    marker_candidates.append((int(tag[0]), float(tag[1]), float(tag[2])))
                                elif len(tag) >= 2 and isinstance(tag[1], dict) and 'x' in tag[1] and 'y' in tag[1]:
                                    dct = tag[1]
                                    marker_candidates.append((int(tag[0]), float(dct['x']), float(dct['y'])))
                        except Exception:
                            continue
                    # Case 2: taglist appears to just be IDs; pull from ekf.markers matrix
                    if not marker_candidates and all(isinstance(t, (int, np.integer)) for t in taglist):
                        markers_arr = getattr(self.ekf, 'markers', None)
                        try:
                            if isinstance(markers_arr, np.ndarray) and markers_arr.ndim == 2 and markers_arr.shape[0] >= 2:
                                # markers_arr expected shape (2, N) where columns align with taglist order
                                n_cols = markers_arr.shape[1]
                                for idx, tid in enumerate(taglist):
                                    if idx < n_cols:
                                        tx = float(markers_arr[0, idx])
                                        ty = float(markers_arr[1, idx])
                                        marker_candidates.append((int(tid), tx, ty))
                        except Exception:
                            pass

                best_tag_pos: Optional[Tuple[float, float]] = None
                best_dist = 1e9
                for (_tid, tx, ty) in marker_candidates:
                    d = math.hypot(tx - rx, ty - ry)
                    if d < best_dist:
                        best_dist = d
                        best_tag_pos = (tx, ty)

                # Debug one-off print if we have raw tag IDs but no positions
                if taglist and best_tag_pos is None and self._last_marker_print is None:
                    print('[MARKER DEBUG] taglist detected but no positions could be parsed; check ekf.markers structure')
                    self._last_marker_print = (0.0, 0.0)

                # No marker yet -> slow rotate to search (avoid drifting)
                if best_tag_pos is None:
                    phase = int((time.time() * 1000) // 1800) % 2
                    spin_dir = 1 if phase == 0 else -1
                    self.command['motion'] = [0, spin_dir * self.turn_cmd]
                    self.notification = 'Searching for first marker'
                    return

                # Decide whether to (re)plan: on first sight, significant movement, or empty plan
                need_replan = False
                now_t = time.time()
                if self.marker_goal is None:
                    need_replan = True
                else:
                    shift = math.hypot(best_tag_pos[0] - self.marker_goal[0], best_tag_pos[1] - self.marker_goal[1])
                    if shift >= self.marker_replan_delta:
                        need_replan = True
                if not self.waypoints or (self.current_goal is None and not self.waypoints):
                    need_replan = True
                # Throttle replans to at most 5 Hz
                if need_replan and (now_t - self.marker_last_plan_time) < 0.2:
                    need_replan = False

                if need_replan:
                    self.marker_goal = [best_tag_pos[0], best_tag_pos[1]]
                    try:
                        self.plan_to_goal(self.marker_goal)
                        self.marker_last_plan_time = now_t
                        self._marker_replan_pause_until = now_t + 0.05
                        if self._last_marker_print is None or math.hypot(best_tag_pos[0]-self._last_marker_print[0], best_tag_pos[1]-self._last_marker_print[1]) > 0.015:
                            print(f"[MARKER] Planned to ArUco @ ({self.marker_goal[0]:.3f}, {self.marker_goal[1]:.3f})")
                            self._last_marker_print = best_tag_pos
                    except Exception as e:
                        self.notification = f'Marker planning failed: {e}'

                # After planning ensure we have a current waypoint
                if self.current_goal is None:
                    self._pick_next_waypoint()

                # Pause briefly after replan
                if time.time() < self._marker_replan_pause_until:
                    self.command['motion'] = [0, 0]
                    self.notification = 'Pause after marker replan'
                    return

                # Dynamic arrival check uses latest best_tag_pos (not stale marker_goal)
                dist_to_marker_now = math.hypot(best_tag_pos[0] - rx, best_tag_pos[1] - ry)
                if dist_to_marker_now <= self.marker_arrival_tol:
                    self._marker_spin_dir = 1  # choose direction (set -1 for opposite)
                    self._marker_spin_until = time.time() + self.marker_spin_duration
                    self.command['motion'] = [0, self._marker_spin_dir * self.turn_cmd]
                    self.notification = 'Arrived at marker: spinning (one direction)'
                    return

                # Standard waypoint following (reuse below logic but early-return here)
                if self.current_goal is not None:
                    gx, gy = self.current_goal
                    dx, dy = gx - rx, gy - ry
                    dist_wp = math.hypot(dx, dy)
                    bearing = math.atan2(dy, dx)
                    dheading = angle_diff(bearing, rth)
                    if dist_wp <= self.dist_tol:
                        if self.waypoints:
                            self._pick_next_waypoint()
                        return
                    if abs(dheading) > self.angle_tol:
                        self.command['motion'] = [0, self.turn_cmd if dheading > 0 else -self.turn_cmd]
                    else:
                        self.command['motion'] = [self.fwd_cmd, 0]
                    self.notification = 'Navigating to first marker'
                    return
            # If initial_marker_done was set above, we fall through to exploration
        # ---- Step 2: Navigate to fixed point (0,1) with dynamic A* (once, before exploration) ----
        if self.stage in ('point',) and self.initial_marker_done and not self.point_goal_done:
            self._step_point_goal()
            # If we're still working on point goal, stop further logic this cycle
            if not self.point_goal_done:
                return
            # After completion we allow exploration / fruit acquisition to proceed
            if self.point_goal_done:
                self.stage = 'explore'

    # If in exploration mode and we have a confirmed *front* target not yet planned -> plan
        if self.mode == 'explore':
            if self.remaining_labels:
                front = self.remaining_labels[0]
                for c in self.fruit_clusters:
                    if c['label'] == front and c['confirmed']:
                        self.mode = 'approach'
                        self.plan_to_goal([float(c['x']), float(c['y'])])
                        break
        # Exploration: ensure we have coverage waypoint(s)
        if self.mode == 'explore':
            if not self.waypoints and self.current_goal is None:
                # Refill waypoints with next coverage point (batch a few ahead)
                batch: List[List[float]] = []
                for _ in range(6):
                    nxt = self._next_coverage_waypoint()
                    if nxt is None:
                        break
                    batch.append(nxt)
                self.waypoints = batch
                self.current_goal = None
            if self.current_goal is None:
                self._pick_next_waypoint()

        # Approach mode: ensure a goal
        if self.mode == 'approach':
            if self.current_goal is None:
                self._pick_next_waypoint()
            if self.current_goal is None:
                # no plan (should not happen) -> revert to explore
                self.mode = 'explore'
                return

        # Motion control identical for both modes
        if self.current_goal is None:
            self.command['motion'] = [0, 0]
            return
        x, y, th = self.get_pose()
        gx, gy = self.current_goal
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        dheading = angle_diff(bearing, th)

        if dist <= self.dist_tol:
            # Waypoint reached
            if self.waypoints:
                self._pick_next_waypoint()
                return
            # Final waypoint reached
            if self.mode == 'approach':
                # Mark front label completed
                if self.remaining_labels:
                    lbl = self.remaining_labels[0]
                    self.remaining_labels.pop(0)
                    self.notification = f'Collected {lbl}'
                self.mode = 'explore'
                self.waypoints = []
                self.current_goal = None
            else:
                self.current_goal = None
            return

        # Turn-then-drive (simple)
        if abs(dheading) > self.angle_tol:
            self.command['motion'] = [0, self.turn_cmd if dheading > 0 else -self.turn_cmd]
        else:
            self.command['motion'] = [self.fwd_cmd, 0]

    # ----------------- Drawing Overlays -----------------
    def draw(self, canvas):  # type: ignore[override]
        super().draw(canvas)
        # Minimal overlay: show fruit clusters & current coverage waypoint on SLAM panel
        v_pad = 40
        h_pad = 20
        slam_origin = (2 * h_pad + 320, v_pad)
        ekf_view = self.ekf.draw_slam_state(res=(320, 480 + v_pad), not_pause=self.ekf_on)

        def to_im(xr: float, yr: float):
            m2pixel = 100
            w, h = (320, 480 + v_pad)
            return int(-xr * m2pixel + w / 2.0), int(yr * m2pixel + h / 2.0)

        rx, ry, _ = self.get_pose()

        # Draw fruit clusters
        for c in self.fruit_clusters:
            px, py = to_im(float(c['x']) - rx, float(c['y']) - ry)
            color = (50, 200, 50) if c['confirmed'] else (180, 160, 40)
            pygame.draw.circle(ekf_view, color, (px, py), 4)
            try:
                label = f"{c['label']}:{c['count']}"[:10]
                font = pygame.font.SysFont(None, 14)
                ekf_view.blit(font.render(label, True, (240, 240, 240)), (px + 4, py))
            except Exception:
                pass

        # Draw dynamic obstacles
        for o in self.discovered_obstacles:
            px, py = to_im(float(o['x']) - rx, float(o['y']) - ry)
            pygame.draw.line(ekf_view, (220, 50, 50), (px - 4, py - 4), (px + 4, py + 4), 2)
            pygame.draw.line(ekf_view, (220, 50, 50), (px - 4, py + 4), (px + 4, py - 4), 2)

        # Draw current planned path (blue)
        if self.current_goal or self.waypoints:
            pts = [[rx, ry]]
            if self.current_goal:
                pts.append(list(self.current_goal))
            pts.extend(self.waypoints)
            for i in range(len(pts) - 1):
                x0, y0 = pts[i]
                x1, y1 = pts[i + 1]
                pygame.draw.line(ekf_view, (60, 90, 240), to_im(x0 - rx, y0 - ry), to_im(x1 - rx, y1 - ry), 2)

        canvas.blit(ekf_view, slam_origin)
        return canvas


def main():
    parser = argparse.ArgumentParser("Mapless exploration + target acquisition")
    parser.add_argument('--ip', type=str, default='192.168.50.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--calib_dir', type=str, default=os.path.join(REPO_ROOT, 'Week02-04', 'calibration', 'param') + os.sep)
    parser.add_argument('--yolo_model', type=str, default=os.path.join(SCRIPT_DIR, 'YOLO', 'model', 'bestv5.pt'))
    parser.add_argument('--list', type=str, default=os.path.join(SCRIPT_DIR, "shopping_list.txt"))
    parser.add_argument('--grid_res', type=float, default=0.03)
    parser.add_argument('--robot_radius', type=float, default=0.10)
    parser.add_argument('--safety_margin', type=float, default=0.10)
    parser.add_argument('--lane_spacing', type=float, default=0.30)
    parser.add_argument('--confirm_count', type=int, default=2)
    parser.add_argument('--play_data', action='store_true')
    parser.add_argument('--save_data', action='store_true')
    args, _ = parser.parse_known_args()

    search_list = load_search_list(args.list)

    op_args = SimpleNamespace(
        ip=args.ip, port=args.port, calib_dir=args.calib_dir,
        yolo_model=args.yolo_model, play_data=args.play_data, save_data=args.save_data,
    )
    operate_mod.args = op_args

    pygame.font.init()
    try:
        TITLE_FONT = pygame.font.Font(os.path.join(WEEK0506_DIR, 'pics', '8-BitMadness.ttf'), 35)
        TEXT_FONT = pygame.font.Font(os.path.join(WEEK0506_DIR, 'pics', '8-BitMadness.ttf'), 40)
        operate_mod.TITLE_FONT = TITLE_FONT
        operate_mod.TEXT_FONT = TEXT_FONT
    except Exception:
        pass

    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 - Mapless Fruit Search')
    try:
        pygame.display.set_icon(pygame.image.load(os.path.join(WEEK0506_DIR, 'pics', '8bit', 'pibot5.png')))
    except Exception:
        pass
    canvas.fill((0, 0, 0))

    # Ensure relative assets resolve
    try:
        os.chdir(WEEK0506_DIR)
    except Exception:
        pass

    operate = AutoOperateMapless(op_args, search_list,
                                  grid_res=args.grid_res,
                                  robot_radius=args.robot_radius,
                                  safety_margin=args.safety_margin,
                                  lane_spacing=args.lane_spacing,
                                  confirm_count=args.confirm_count)

    running = True
    clock = pygame.time.Clock()
    while running:
        operate.update_keyboard()
        operate.take_pic()
        operate.auto_nav_step()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        operate.detect_target()
        operate.perception_update()
        operate.draw(canvas)
        pygame.display.update()
        clock.tick(20)  # limit to ~20 FPS for stability


if __name__ == '__main__':
    main()
