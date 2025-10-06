import os
import sys
import argparse
import time
import math
from types import SimpleNamespace
from typing import List, Tuple, Optional

import numpy as np
import pygame

# Paths
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

# Import helpers and planner
from map_utils import read_true_map_robust, load_search_list, print_target_fruits_pos
from astar_planning import plan_waypoints

# Reuse advanced dynamic behaviours from wednesday.py if available
# (perception update, drawing, pulsed motion, logging etc.)
try:
    from wednesday import AutoOperateDynamic as BaseAuto  # type: ignore
except Exception:
    BaseAuto = Operate  # fallback to raw Operate if import fails


def normalize_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def angle_diff(a: float, b: float) -> float:
    return normalize_angle(a - b)


class AutoOperateDiamond(BaseAuto):
    """Level 3 strategy: mapping-first diamond tour, then target navigation.

    Phase 1 (Mapping):
    - Visit scan points roughly at a diamond: (1,0) -> (0,1) -> (-1,0) -> (0,-1)
    - At each scan point, perform a pulsed 360 spin for full-scene observations
    - Continuously run SLAM + detection; avoid obstacles via A* waypoints
    - If a scan point is occupied (aruco/fruit/obstacle), adjust to a nearby clear point

    Phase 2 (Navigation):
    - After completing the scan tour, switch to normal A* navigation towards the
      target fruits in the shopping list (same behaviour as AutoOperateDynamic)
    """

    def __init__(self, args,
                 search_list: List[str],
                 targets_xy: List[List[float]],
                 aruco_obstacles_xy: List[List[float]],
                 grid_res: float,
                 robot_radius: float,
                 safety_margin: float,
                 merge_threshold: float = 0.80,
                 obs_max_range: float = 0.48,
                 map_fruit_labels: Optional[List[str]] = None,
                 map_fruit_xy: Optional[List[List[float]]] = None):
        super().__init__(args, search_list, targets_xy, aruco_obstacles_xy,
                         grid_res, robot_radius, safety_margin,
                         merge_threshold=merge_threshold,
                         obs_max_range=obs_max_range,
                         map_fruit_labels=map_fruit_labels,
                         map_fruit_xy=map_fruit_xy)

        # Phase management
        self.phase: str = 'MAP'  # 'MAP' -> 'NAV'

        # Diamond scan points (meters)
        self.base_scan_points: List[List[float]] = [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
        # Adjusted scan points (avoid occupied spots)
        self.scan_points: List[List[float]] = [self._adjust_scan_point(pt) for pt in self.base_scan_points]
        self.scan_index: int = 0

        # Replace remaining_targets to follow scan points first
        self.nav_targets_xy: List[List[float]] = [list(t) for t in targets_xy]  # keep for NAV phase
        self.remaining_targets = [list(p) for p in self.scan_points]
        self.remaining_labels = [f'scan{i+1}' for i in range(len(self.remaining_targets))]

        # Pulse 360 scan settings at each scan point
        self.point_scan_duration = 8.0  # seconds to approximate 360 deg with pulsed spin
        self._point_scan_active: bool = False
        self._point_scan_start: float = 0.0
        self._point_scan_dir: int = 1

        # Disable periodic ArUco calibration scans during mapping to avoid conflicts
        try:
            self.calib_interval = 10e6
        except Exception:
            pass

        # Slightly looser arrival tolerance for scan points
        self.dist_tol = max(self.dist_tol, 0.10)

        # Avoid planning right up against the virtual wall
        self.wall_clearance = max(self.wall_clearance, 0.10)

        # Log phase
        try:
            self._log['meta']['strategy'] = 'diamond_scan_then_nav'
        except Exception:
            pass

    # ----------------- Utility helpers -----------------
    def get_pose(self) -> Tuple[float, float, float]:
        # Use parent's helper if available
        try:
            return super().get_pose()  # type: ignore[misc]
        except Exception:
            if hasattr(self, "ekf") and self.ekf is not None:
                robot = getattr(self.ekf, "robot", None)
                if robot is not None and hasattr(robot, "state") and robot.state.shape[0] >= 3:
                    x = float(robot.state[0, 0])
                    y = float(robot.state[1, 0])
                    th = float(robot.state[2, 0])
                    return x, y, th
            return 0.0, 0.0, 0.0

    def _is_point_blocked(self, pt: List[float], tol: float = 0.25) -> bool:
        px, py = float(pt[0]), float(pt[1])
        # Check known arucos
        for ox, oy in self.known_obstacles:
            if math.hypot(px - float(ox), py - float(oy)) <= tol:
                return True
        # Check discovered obstacles
        for d in getattr(self, 'discovered_obstacles', []) or []:
            try:
                ox, oy = float(d['x']), float(d['y'])
                if math.hypot(px - ox, py - oy) <= tol:
                    return True
            except Exception:
                pass
        # Check mapped fruits (from partial map)
        if getattr(self, 'map_fruit_xy', None):
            for q in self.map_fruit_xy:
                try:
                    qx, qy = float(q[0]), float(q[1])
                    if math.hypot(px - qx, py - qy) <= tol:
                        return True
                except Exception:
                    pass
        return False

    def _adjust_scan_point(self, pt: List[float]) -> List[float]:
        # Keep inside arena inner boundary
        try:
            inner = max(0.0, float(self.arena_half) - float(self.wall_clearance))
        except Exception:
            inner = 1.10
        px, py = float(pt[0]), float(pt[1])
        px = max(-inner + 0.05, min(inner - 0.05, px))
        py = max(-inner + 0.05, min(inner - 0.05, py))
        cand = [px, py]
        if not self._is_point_blocked(cand, tol=0.22):
            return cand
        # Try ring of offsets around the point
        radii = [0.20, 0.30, 0.40]
        angles = [i * (math.pi / 6) for i in range(12)]  # every 30 degrees
        for r in radii:
            for a in angles:
                nx = px + r * math.cos(a)
                ny = py + r * math.sin(a)
                if abs(nx) > inner - 0.05 or abs(ny) > inner - 0.05:
                    continue
                if not self._is_point_blocked([nx, ny], tol=0.20):
                    return [nx, ny]
        # Fall back to original if no clear candidate found
        return [px, py]

    def _phase_done(self) -> bool:
        return self.phase == 'NAV' and not self.remaining_targets

    # ----------------- Planning -----------------
    def replan_to_current_list(self, initial: bool = False):
        # Wrapper to plan waypoints using current remaining_targets + obstacles
        if not self.active:
            return
        if not self.remaining_targets:
            self.waypoints = []
            self.current_goal = None
            self.active = False
            self.notification = 'All targets completed'
            return
        x, y, _ = self.get_pose()
        obstacles_xy = list(self.known_obstacles) + [[float(d['x']), float(d['y'])] for d in self.discovered_obstacles]
        # Add virtual wall as obstacles
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
            new_waypoints = plan_waypoints([x, y], self.remaining_targets, obstacles_xy,
                                           grid_res=self.grid_res,
                                           robot_radius=self.robot_radius,
                                           safety_margin=self.safety_margin)
            self.waypoints = new_waypoints
            self.current_goal = None
            self.notification = f"{'Initial' if initial else 'Re'}planned path with {len(self.waypoints)} waypoints"
            try:
                self._log['plans'].append({'t': time.time(), 'waypoints': [list(wp) for wp in (self.waypoints or [])]})
            except Exception:
                pass
        except Exception as e:
            self.notification = f'Planning failed: {e}'

    # ----------------- Control -----------------
    def _start_point_scan(self):
        self._point_scan_active = True
        self._point_scan_start = time.time()
        self._point_scan_dir *= -1  # alternate direction
        self.notification = 'Scan: pulsed 360 at waypoint'

    def _do_point_scan_step(self) -> bool:
        # Returns True if scan still ongoing, False if finished
        now = time.time()
        elapsed = now - self._point_scan_start
        if elapsed >= self.point_scan_duration:
            self._point_scan_active = False
            self.command['motion'] = [0, 0]
            return False
        # Pulsed turn using same turn timings as nav turning pulses
        t_period = float(self.nav_turn_pulse_spin_time + self.nav_turn_pulse_stop_time)
        phase = (now - self._point_scan_start) % t_period
        if phase < self.nav_turn_pulse_spin_time:
            self.command['motion'] = [0, self._point_scan_dir * self.turn_cmd]
        else:
            self.command['motion'] = [0, 0]
        return True

    def _advance_scan_point(self):
        # Finish this scan point and go to next or switch phase
        if self.scan_index + 1 < len(self.scan_points):
            self.scan_index += 1
            # Update targets list to the new tail (remaining scan points)
            self.remaining_targets = [list(p) for p in self.scan_points[self.scan_index:]]
            self.remaining_labels = [f'scan{i+1}' for i in range(self.scan_index, len(self.scan_points))]
            self.replan_to_current_list(initial=True)
        else:
            # Mapping finished -> switch to NAV phase
            self.phase = 'NAV'
            self.remaining_targets = [list(t) for t in self.nav_targets_xy]
            # keep labels aligned to search_list order
            # self.remaining_labels already holds search_list from base
            self.replan_to_current_list(initial=True)
            self.notification = 'Diamond scan complete — switching to target navigation'

    def auto_nav_step(self):
        # Ensure SLAM
        if not self.ekf_on:
            self.command['motion'] = [0, 0]
            self.notification = 'Press ENTER to start SLAM'
            return

        now = time.time()

        # High covariance stabilize spin (reuse base behaviour if available)
        try:
            # Inline copy of the relevant logic from base to avoid relying on internal fields
            if self._cov_spin_until is not None and now < self._cov_spin_until:
                if self._cov_spin_start is None:
                    self._cov_spin_start = now
                period = float(self.cov_pulse_spin_time + self.cov_pulse_stop_time)
                phase = (now - self._cov_spin_start) % period
                if phase < self.cov_pulse_spin_time:
                    self.command['motion'] = [0, self._cov_spin_dir * self.turn_cmd]
                    self.notification = 'High covariance: stabilizing spin'
                else:
                    self.command['motion'] = [0, 0]
                    self.notification = 'High covariance: stabilizing spin (pulse stop)'
                return
            if self._cov_spin_until is not None and now >= self._cov_spin_until:
                self._cov_spin_until = None
                self._cov_spin_start = None
                self._cov_cooldown_until = now + self.cov_spin_cooldown
            if now >= getattr(self, '_cov_cooldown_until', 0.0):
                P = getattr(self.ekf, 'P', None)
                if isinstance(P, np.ndarray) and P.shape[0] >= 2 and P.shape[1] >= 2:
                    pxx = float(P[0, 0])
                    if pxx > float(self.cov_pos_thresh):
                        self._cov_spin_dir = -self._cov_spin_dir
                        self._cov_spin_until = now + float(self.cov_spin_duration)
                        self._cov_spin_start = now
                        self.command['motion'] = [0, self._cov_spin_dir * self.turn_cmd]
                        self.notification = 'High covariance: stabilizing spin'
                        return
        except Exception:
            pass

        # Log poses and flush occasionally (if base supports logging)
        try:
            if (now - getattr(self, '_last_pose_log', 0.0)) >= 0.2:
                if hasattr(self, '_log_pose'):
                    self._log_pose(now)  # type: ignore[misc]
                self._last_pose_log = now
            if hasattr(self, '_flush_log'):
                self._flush_log(force=False)  # type: ignore[misc]
        except Exception:
            pass

        # If currently performing point scan, continue rotating
        if self._point_scan_active:
            if self._do_point_scan_step():
                return
            # Scan just finished -> reverse briefly like base, then advance
            self._reverse_until = time.time() + self.reverse_duration
            self._pending_complete_after_reverse = True

        # Handle reverse after scan/target (same as base behaviour)
        if self._reverse_until is not None:
            if time.time() < self._reverse_until:
                self.command['motion'] = [-self.fwd_cmd, 0]
                self.notification = 'Reversing from scan/target'
                return
            else:
                self._reverse_until = None
                if self._pending_complete_after_reverse:
                    self._pending_complete_after_reverse = False
                    if self.phase == 'MAP':
                        self._advance_scan_point()
                    else:
                        # In NAV phase, reuse base progression
                        self._advance_target()
                        self.replan_to_current_list(initial=False)

        # Ensure plan
        if not getattr(self, '_planned_once', False) or (self.active and not self.waypoints):
            self.replan_to_current_list(initial=not getattr(self, '_planned_once', False))
            self._planned_once = True

        # Pick next goal waypoint if needed
        if self.current_goal is None and self.active:
            self.pick_next_goal()
        if not self.current_goal:
            self.command['motion'] = [0, 0]
            return

        # Controller towards current waypoint
        x, y, th = self.get_pose()
        gx, gy = self.current_goal
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        dheading = angle_diff(bearing, th)

        # Arrival handling
        if dist <= self.dist_tol:
            # If we're close to the final target for this segment, trigger scan (MAP) or hold (NAV)
            close_to_segment_target = False
            if self.remaining_targets:
                tx, ty = self.remaining_targets[0]
                close_to_segment_target = (math.hypot(gx - tx, gy - ty) <= max(0.12, self.grid_res * 3))

            if close_to_segment_target:
                if self.phase == 'MAP':
                    # Begin pulsed 360 scan at this point
                    if not self._point_scan_active:
                        self._start_point_scan()
                    # Immediate step this tick
                    self._do_point_scan_step()
                    return
                else:
                    # NAV phase: base behaviour — hold then reverse handled by reverse block above
                    if self.reached_time is None:
                        self.reached_time = time.time()
                        self.notification = f'Reached target [{gx:.2f}, {gy:.2f}]. Holding...'
                    self.command['motion'] = [0, 0]
                    if time.time() - self.reached_time >= self.hold_duration:
                        self._reverse_until = time.time() + self.reverse_duration
                        self._pending_complete_after_reverse = True
                        self.command['motion'] = [-self.fwd_cmd, 0]
                    return
            else:
                self.pick_next_goal()
                return

        # Turn-then-drive with pulsing
        turning = abs(dheading) > self.angle_tol
        if turning:
            if getattr(self, '_nav_last_mode', None) != 'turn':
                self._nav_last_mode = 'turn'
                self._nav_turn_pulse_start = time.time()
            if self._nav_turn_pulse_start is None:
                self._nav_turn_pulse_start = time.time()
            t_period = float(self.nav_turn_pulse_spin_time + self.nav_turn_pulse_stop_time)
            t_phase = (time.time() - self._nav_turn_pulse_start) % t_period
            if t_phase < self.nav_turn_pulse_spin_time:
                self.command['motion'] = [0, self.turn_cmd if dheading > 0 else -self.turn_cmd]
            else:
                self.command['motion'] = [0, 0]
        else:
            if getattr(self, '_nav_last_mode', None) != 'drive':
                self._nav_last_mode = 'drive'
                self._nav_drive_pulse_start = time.time()
            if self._nav_drive_pulse_start is None:
                self._nav_drive_pulse_start = time.time()
            d_period = float(self.nav_drive_pulse_period)
            d_phase = (time.time() - self._nav_drive_pulse_start) % d_period
            if d_phase < (self.nav_drive_pulse_period - self.nav_drive_pulse_stop_time):
                self.command['motion'] = [self.fwd_cmd, 0]
            else:
                self.command['motion'] = [0, 0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Level 3: Diamond mapping tour then navigation")
    parser.add_argument("--ip", type=str, default="192.168.50.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default=os.path.join(REPO_ROOT, "Week02-04", "calibration", "param") + os.sep)
    parser.add_argument("--yolo_model", default=os.path.join(SCRIPT_DIR, "YOLO", "model", "bestv5.pt"))
    parser.add_argument("--map", type=str, default=os.path.join(SCRIPT_DIR, "M3_prac_map_part.txt"))
    parser.add_argument("--list", type=str, default=os.path.join(SCRIPT_DIR, "M3_prac_shopping_list.txt"))
    parser.add_argument("--grid_res", type=float, default=0.02)
    parser.add_argument("--robot_radius", type=float, default=0.16)
    parser.add_argument("--safety_margin", type=float, default=0.3)
    parser.add_argument("--merge_threshold", type=float, default=0.80)
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
    pygame.display.set_caption('ECE4078 - L3 Diamond Explore -> Nav')
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

    operate = AutoOperateDiamond(op_args, search_list, targets_xy, aruco_obstacles_xy,
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
        # SLAM + control loop
        operate.auto_nav_step()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        # Data + detection
        operate.record_data()
        operate.save_image()
        operate.detect_target()  # detector_output populated
        # Perception update to add obstacles + replan if needed (from BaseAuto)
        if hasattr(operate, 'periodic_perception_update'):
            operate.periodic_perception_update()  # type: ignore[misc]
        # Draw
        operate.draw(canvas)
        pygame.display.update()
