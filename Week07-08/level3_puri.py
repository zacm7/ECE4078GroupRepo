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
        self.discovered_obstacles: List[List[float]] = []
        # Keep parallel metadata for discovered obstacles
        self.discovered_labels: List[str] = []
        self.discovered_counts: List[int] = []
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
        self.dist_tol = 0.10
        self.angle_tol = math.radians(8.0)
        self.turn_cmd = 1
        self.fwd_cmd = 1

        # Marker acquisition (scan/creep) state
        self._scan_start = None
        self._scan_dir = 1
        self._creep_until = None
        self._planned_once = False

        # Arrival reverse behavior
        self.hold_duration = 2.5
        self.reverse_duration = 0.75
        self._reverse_until = None
        self._pending_complete_after_reverse = False

        # Detection handling
        self.last_obstacle_add_time = 0.0
        self.add_cooldown = 0.5  # seconds
        self.min_obs_separation = 0.15  # m
        # Merge detections of the same obstacle label within this radius (m)
        self.merge_threshold = float(merge_threshold)

        # Cache intrinsics (for projection of bbox -> world)
        self.K = getattr(self.ekf.robot, 'camera_matrix', None)
        self.cx = float(self.K[0, 2]) if self.K is not None else 160.0
        self.fx = float(self.K[0, 0]) if self.K is not None else 320.0

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
        all_obs = []
        all_obs.extend(self.known_obstacles)
        all_obs.extend(self.discovered_obstacles)
        for ox, oy in all_obs:
            px, py = to_im((float(ox) - rx, float(oy) - ry))
            pygame.draw.line(ekf_view, (220, 50, 50), (px - 4, py - 4), (px + 4, py + 4), 2)
            pygame.draw.line(ekf_view, (220, 50, 50), (px - 4, py + 4), (px + 4, py - 4), 2)

        # Blit the augmented SLAM view back to the main canvas
        canvas.blit(ekf_view, slam_origin)
        return canvas

    # --- Logging helpers ---
    def _log_pose(self, now: float | None = None):
        try:
            t = time.time() if now is None else now
            x, y, th = self.get_pose()
            self._log['poses'].append([t, float(x), float(y), float(th)])
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

    def auto_nav_step(self):
        # Require SLAM
        if not self.ekf_on:
            self.command['motion'] = [0, 0]
            self.notification = 'Press ENTER to start SLAM'
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

        # Turn-then-drive
        if abs(dheading) > self.angle_tol:
            # Rotate in place to reduce heading error before moving
            self.command['motion'] = [0, self.turn_cmd if dheading > 0 else -self.turn_cmd]
        else:
            self.command['motion'] = [self.fwd_cmd, 0]

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
        obstacles_xy = list(self.known_obstacles) + list(self.discovered_obstacles)
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
            keep_obs: List[List[float]] = []
            keep_labels: List[str] = []
            keep_counts: List[int] = []
            for i, (ox, oy) in enumerate(self.discovered_obstacles):
                if math.hypot(ox - ntx, oy - nty) > max(0.12, self.grid_res * 2):
                    keep_obs.append([ox, oy])
                    # Maintain metadata alignment if present
                    if i < len(self.discovered_labels):
                        keep_labels.append(self.discovered_labels[i])
                    if i < len(self.discovered_counts):
                        keep_counts.append(self.discovered_counts[i])
            self.discovered_obstacles = keep_obs
            self.discovered_labels = keep_labels
            self.discovered_counts = keep_counts
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
                for (mf_label, (mx, my)) in zip(self.map_fruit_labels, self.map_fruit_xy):
                    if mf_label == label and math.hypot(ox - float(mx), oy - float(my)) <= self.map_match_tol:
                        # Treat as known fruit from map; do not add as obstacle
                        ox = oy = None  # invalidate to skip further processing
                        break
                if ox is None or oy is None:
                    continue

            # First: try to merge with an existing discovered obstacle of the SAME label within merge_threshold
            merged = False
            merge_idx = -1
            for i, (px, py) in enumerate(self.discovered_obstacles):
                if i < len(self.discovered_labels) and self.discovered_labels[i] == label:
                    if math.hypot(ox - px, oy - py) <= self.merge_threshold:
                        # Incremental mean update for the cluster centre
                        cnt = self.discovered_counts[i] if i < len(self.discovered_counts) else 1
                        new_x = (px * cnt + ox) / (cnt + 1)
                        new_y = (py * cnt + oy) / (cnt + 1)
                        moved = math.hypot(new_x - px, new_y - py)
                        self.discovered_obstacles[i] = [new_x, new_y]
                        if i < len(self.discovered_counts):
                            self.discovered_counts[i] = cnt + 1
                        else:
                            # ensure alignment if counts list was shorter
                            self.discovered_counts.append(cnt + 1)
                        # Log merge and optionally trigger replan if position changed
                        self._log_obstacle(new_x, new_y, label=label, method=('merge-tpe' if used_tpe else 'merge-heuristic'))
                        self._flush_log(force=False)
                        if moved > 1e-3:
                            new_added = True
                        merged = True
                        merge_idx = i
                        break

            if merged:
                # merged into an existing cluster; skip adding/duplicate checks
                continue

            # Ignore duplicates amongst known and discovered obstacles (any label) using a wider gate
            all_obs = self.known_obstacles + self.discovered_obstacles
            if any(math.hypot(ox - qx, oy - qy) <= self.min_obs_separation for qx, qy in all_obs):
                continue

            # Rate limit
            if now - self.last_obstacle_add_time < self.add_cooldown:
                continue

            # Add obstacle and mark for replanning
            self.discovered_obstacles.append([ox, oy])
            self.discovered_labels.append(label)
            self.discovered_counts.append(1)
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
    parser.add_argument("--robot_radius", type=float, default=0.12)
    parser.add_argument("--safety_margin", type=float, default=0.15)
    parser.add_argument("--merge_threshold", type=float, default=0.30)
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
