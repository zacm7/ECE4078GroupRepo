import os
import sys
import argparse
import time
import math
import json
from types import SimpleNamespace
from typing import List, Tuple, Dict

import numpy as np
import pygame


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Import local operate.py (now located in the same Week07-08 directory)
try:
    import operate as operate_mod  # type: ignore
    from operate import Operate    # type: ignore
except Exception:
    # Fallback manual loader (should rarely be needed now that file is local)
    import importlib.util
    _op_file = os.path.join(SCRIPT_DIR, "operate.py")
    _spec = importlib.util.spec_from_file_location("operate", _op_file)
    operate_mod = importlib.util.module_from_spec(_spec)  # type: ignore
    assert _spec and _spec.loader
    _spec.loader.exec_module(operate_mod)  # type: ignore
    Operate = operate_mod.Operate  # type: ignore

# Helpers and planner (map utilities removed – fully dynamic now)
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
        # Autonomous patrol with dynamic obstacle discovery (no prior map).

    def __init__(self, args,
                 grid_res: float,
                 robot_radius: float,
                 safety_margin: float,
                 merge_threshold: float = 0.55,
                 obs_max_range: float = 0.30,
                 patrol_points: List[Tuple[float, float]] | None = None):
        super().__init__(args)

        # Always run detector continuously
        self.command['inference'] = True

        # Ensure map outputs (slam.txt, images.txt) go to local lab_output directory
        try:
            out_dir = os.path.join(SCRIPT_DIR, 'lab_output')
            os.makedirs(out_dir, exist_ok=True)
            if hasattr(operate_mod, 'dh') and hasattr(operate_mod.dh, 'OutputWriter'):
                self.output = operate_mod.dh.OutputWriter(out_dir)
        except Exception:
            pass

        # --- Patrol model (replaces partial map targets) ---
        # Default sequence; robot assumed to start near (0,0).
        default_patrol = [(-0.67, 0.67), (-0.67, -0.67), (0.67, -0.67), (0.67, 0.67)]
        pts: List[Tuple[float, float]] = []
        if patrol_points:
            pts = [(float(a), float(b)) for (a, b) in patrol_points]
        if not pts:
            pts = default_patrol
        # Ensure exactly 4 points (pad/trim)
        while len(pts) < 4:
            pts.append(default_patrol[len(pts) % len(default_patrol)])
        if len(pts) > 4:
            pts = pts[:4]
        # Optional: reorder via CLI argument --quadrants (digits 1-4 like 2341)
        try:
            q = str(getattr(args, 'quadrants', '') or '')
            digits = [ch for ch in q if ch in '1234']
            if len(digits) == 4 and len(set(digits)) == 4:
                order = [int(d) - 1 for d in digits]
                if all(0 <= idx < 4 for idx in order):
                    pts = [pts[idx] for idx in order]
        except Exception:
            # Ignore invalid quadrants input
            pass
        self.patrol_points: List[Tuple[float, float]] = pts
        self.patrol_index: int = 0

        # Emulate previous target interface: single active target in remaining_targets
        self.search_list = [f'patrol{i}' for i in range(len(self.patrol_points))]
        self.remaining_targets: List[List[float]] = [list(self.patrol_points[self.patrol_index])]
        self.remaining_labels: List[str] = [self.search_list[self.patrol_index]]

        # Static known obstacles list retired; we derive obstacles dynamically
        self.known_obstacles: List[List[float]] = []
        # discovered obstacles stored as dicts for safety/alignment
        # each item: {'x': float, 'y': float, 'label': str, 'count': int}
        self.discovered_obstacles: List[dict] = []
        # No map fruit ground truth suppression
        self.map_fruit_labels = []  # type: ignore[assignment]
        self.map_fruit_xy = []      # type: ignore[assignment]
        self.map_match_tol = 0.0

        # A* params
        self.grid_res = grid_res
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
    # Keep a default copy of safety margin for temporary overrides during replanning
        self._safety_margin_default = float(safety_margin)
        self._safety_margin_saved = None  # type: ignore[assignment]

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
        # Track marker count to trigger replans when new ArUcos are added
        self._last_marker_count = 0
        # Initial startup spin (environment survey) configuration
        self._initial_spin_duration = 14.0  # seconds
        # Start this spin when SLAM starts (ekf_on becomes True), not at program start
        self._initial_spin_start = None
        self._initial_spin_done = False

        # Arrival spin behavior (replaces previous hold+reverse): spin 8s then advance
        self.arrival_spin_duration = 24.0
        self._arrival_spin_start = None
        # Pulsed arrival spin (spin/pause cadence similar to other pulsed motions)
        self.arrival_pulse_spin_time = 0.4   # seconds spinning
        self.arrival_pulse_stop_time = 0.35   # seconds stopped
        self._arrival_spin_pulse_start = None

        # Detection handling
        self.last_obstacle_add_time = 0.0
        self.add_cooldown = 0.7  # seconds
        self.min_obs_separation = 0.05  # m
    # If a different fruit label is observed within this radius of an existing fruit,
    # treat it as the same physical object and update the existing fruit instead of
    # creating a new one (helps collapse mislabels that are colocated).
        self.cross_label_merge_radius = 0.05  # m (5 cm)
        # Merge detections of the same obstacle label within this radius (m)
        self.merge_threshold = float(merge_threshold)
        # Larger merge threshold for non-target fruits to cluster more aggressively
        self.merge_threshold_non_target = float(self.merge_threshold) + 0.20
        # Only consider detections as obstacles if within this distance (m) from the robot
        self.obs_max_range = float(obs_max_range)
    # Manage temporary range override during waypoint spin
        self._obs_range_override_active = False
        # Fruit obstacle dynamic update tuning
        self.fruit_update_alpha = 0.5          # smoothing factor for position updates
        self.fruit_replan_move_thr = 0.015     # trigger replan if cluster moved more than this (m)
        self.fruit_stale_time = 1800.0           # prune if not seen for this many seconds
        # Kalman-style update parameters for fruit (approximate consistency with EKF landmark refinement)
        self.fruit_Q = np.diag([1e-5, 1e-5])   # process noise (very small, assume static fruit)
        self.fruit_R = np.diag([0.01, 0.01])   # measurement noise (tunable)
        # Arena virtual walls (2.4x2.4m centered at origin) with 10cm keep-out
        self.arena_half = 1.267
        self.wall_clearance = 0.10

        # Cache intrinsics (for projection of bbox -> world)
        self.K = getattr(self.ekf.robot, 'camera_matrix', None)
        self.cx = float(self.K[0, 2]) if self.K is not None else 160.0
        self.fx = float(self.K[0, 0]) if self.K is not None else 320.0

        # Covariance-based stabilize spin parameters
        self.cov_pos_thresh = 0.14      # trigger threshold on P[0,0]
        self.cov_spin_duration = 9.0      # seconds to spin when triggered (increased from 6s)
        # Grace window after arrival spin to allow fruit detections to be added before we decide to advance patrol
        self.post_spin_grace = 0.6  # seconds to wait after spin for fruit queue to populate
        self._post_spin_grace_until = 0.0
        self.cov_spin_cooldown = 3.0      # seconds to wait before checking again
        self._cov_spin_until = None       # type: ignore[assignment]
        self._cov_cooldown_until = 0.0
        self._cov_spin_dir = 1            # alternate spin direction each trigger
        # Pulsed spin timing: spin for 0.4s, stop for 0.2s, repeat
        self.cov_pulse_spin_time = 0.4
        self.cov_pulse_stop_time = 0.2
        self._cov_spin_start = None       # type: ignore[assignment]
        # Do not perform covariance spin while holding at a fruit; defer until after hold
        self._cov_spin_deferred = False

    # Post-hold reverse behavior
        self.fruit_post_hold_reverse_time = 0.7  # seconds to reverse after holding at a fruit
        self._fruit_post_reverse_until = 0.0

        # Navigation pulse timing (normal turn/drive):
        # - Turning: spin 0.4s, stop 0.2s (same as covariance spin)
        # - Driving forward: now also pulsed explicitly (move then brief stop)
        self.nav_turn_pulse_spin_time = 0.2
        self.nav_turn_pulse_stop_time = 0.4
        # Drive pulse uses the same cadence as turning (spin/stop)
        # move for nav_turn_pulse_spin_time then stop for nav_turn_pulse_stop_time
        self.nav_drive_pulse_move_time = 0.2
        self.nav_drive_pulse_stop_time = 0.4
        self._nav_turn_pulse_start = None  # type: ignore[assignment]
        self._nav_drive_pulse_start = None  # type: ignore[assignment]
        self._nav_last_mode = None  # 'turn' | 'drive' | None
        # Shared pulse anchor for turn/drive so phase doesn't reset when switching
        self._nav_pulse_start = None

        # Small font used for overlay labels (smaller text)
        # pygame.font.init() is called in __main__ before the operate instance is created,
        # so SysFont is available here.
        try:
            self.label_font = pygame.font.SysFont(None, 14)
        except Exception:
            self.label_font = None
        # HUD font for on-screen timers/info
        try:
            self.hud_font = pygame.font.SysFont(None, 18)
        except Exception:
            self.hud_font = None

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
                'mode': 'patrol',
                'patrol_points': [list(p) for p in self.patrol_points],
                'grid_res': self.grid_res,
                'robot_radius': self.robot_radius,
                'safety_margin': self.safety_margin,
                'merge_threshold': self.merge_threshold,
                'obs_max_range': self.obs_max_range,
                'calib_interval': self.calib_interval,
            },
            'poses': [],
            'plans': [],
            'obstacles': []
        }
        self._last_pose_log = 0.0
        self._last_flush = 0.0
        # logs go under local lab_output regardless of cwd
        log_dir = os.path.join(SCRIPT_DIR, 'lab_output')
        self._log_path = os.path.join(log_dir, 'auto_nav_log.json')
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass
        # Fruit locations persistence file (for later retrieval / navigation)
        self._fruit_loc_path = os.path.join(log_dir, 'targets.txt')  # now JSON format
        # Track how many of each base fruit label we've enumerated (for suffix _0, _1, ...)
        self._fruit_label_counts = {}
        # Shopping list (fruits to explicitly visit when discovered between patrol points)
        self.shopping_list = self._load_shopping_list()
    # Feature toggle: visit fruits between patrol waypoints after arrival spin
    # Set to False to completely disable go-to-fruit-and-hold behavior
        self.enable_fruit_visits = True
        # Opportunistic detours: while driving to a patrol point, if a shopping-list fruit is seen,
        # detour to it, hold, then resume original patrol.
        self.enable_opportunistic_detours = True
        # Fruit visit parameters
        self.fruit_visit_radius = 0.09  # meters (15cm; approach within this distance counts as visited)
        self.fruit_hold_duration = 5.0  # seconds to hold at fruit
        # Timeout for reaching/holding at an active fruit target (seconds)
        # 6.7 minutes = 402 seconds
        self.fruit_target_timeout = 402.0
        # Alignment phase: before driving to a fruit, center it and look for a short dwell
        self.fruit_align_duration = 8.0  # seconds to keep fruit centered before starting nav
        self._fruit_align_start = None
        self._fruit_align_pulse_start = None
        self._fruit_align_spin_dir = 1
        # Max time to spend trying to center a target fruit before skipping (seconds)
        self.fruit_align_max_time = 45.0
        self._fruit_align_enter_time = None
        # Track last-centered time per base label (updated by perception)
        self._last_centered_label_time = {}
        # Observe-only alignment for non-target fruits (center and dwell without navigating)
        self.enable_observe_align = True
        self.observe_align_duration = float(self.fruit_align_duration)
        self._observe_align_start = None
        self._observe_align_pulse_start = None
        self._observe_align_spin_dir = 1
    # Add a hard cap for observe-only alignment as well (same as fruit_align_max_time)
        self.observe_align_max_time = float(getattr(self, 'fruit_align_max_time', 45.0))
        self._observe_align_enter_time = None
        self._observe_label_base = None
        self._observe_label_display = None
        self._observe_cooldown_until = 0.0
        # Fruit visit state
        self._fruit_visit_queue = []
        self._fruit_queue_last_sig = None  # track last queue signature to announce changes
        self._fruit_hold_start = None
        # Deadline by which we must reach and begin holding at the active fruit; 0.0 means inactive
        self._fruit_target_deadline = 0.0
        self._visited_fruits_cycle = set()  # enumerated labels visited in current patrol leg
    # Persistent record of visited fruit base labels (e.g., 'potato') for the entire run
        self._visited_fruit_bases = set()
        self._pending_patrol_advance = False  # set when we've deferred advancing to next patrol point until fruit visits done
        self._mode = 'patrol'  # 'patrol' | 'arrival_spin' | 'fruit_nav' | 'fruit_hold'
    # Track the actual fruit target position for proximity checks (avoid using intermediate waypoints)
        self._fruit_target_xy = None
    # Track the base label of the current fruit target (for dynamic updates)
        self._fruit_target_label_base = None
        self._fruit_target_last_update = 0.0
        self._fruit_target_replan_cooldown = 0.2  # seconds
        # Save patrol state when taking a detour so we can resume afterwards
        self._saved_patrol_state = None  # type: ignore[assignment]
    # Track whether we're on an opportunistic detour vs queued fruit visit
        self._detour_active = False
        # Emergency stop (camera-based proximity) parameters
        self.emergency_enabled = True
        self.emergency_bbox_width_thresh_px = 150.0  # px width indicating very close object
        self.emergency_bbox_height_thresh_px = 150.0 # px height indicating very close object
        self.emergency_center_tolerance_px = 120.0   # px from center in x to accept as frontal
        self.emergency_dist_m = 0.20                 # estimated distance cutoff (m)
        self.emergency_hold_time = 1.2               # seconds to stop before resuming
    # New: perform an initial stationary stop before reversing
        self.emergency_initial_stop_time = 5.0       # seconds to stop initially before reverse
        self._emergency_until = 0.0
        self._emergency_replan_triggered = False
        # New: reverse first, then hold + replan, with a brief cooldown to avoid loops
        self.emergency_reverse_time = 0.5            # seconds to back up when triggered
        self.emergency_cooldown = 1.0                # seconds after hold to ignore retriggers
        self._emergency_mode = None                  # None | 'prestop' | 'reverse' | 'hold'
        self._emergency_cooldown_until = 0.0
    # Track which detection triggered the emergency so we can announce it on hold
        self._emergency_trigger_label = None  # type: ignore[assignment]

        # Periodic replan settings
        # In addition to event-driven replans, refresh the plan at a fixed cadence
        self._periodic_replan_interval = 30  # seconds
        self._last_periodic_replan = time.time()
    # Plan-failure watchdog: if planning fails continuously for this long, trigger a cov spin
        self._plan_fail_start = None  # type: ignore[assignment]
        self._plan_fail_threshold = 10.0  # seconds
        # Safety margin probe when planning is stuck: temporarily reduce margin and try
        self._safety_probe_active = False
        self._safety_probe_until = 0.0
        # Duration to keep reduced safety margin active (seconds)
        self.safety_probe_duration = 9.0
        # Waypoint-based probe control (new)
        self._safety_probe_waypoints_left = 0
        self._wp_counted = False
    # Timed multi-stage probe control
    # stage: 0=inactive, 1=0.045, 2=0.020, 3=0.000
        self._safety_probe_stage = 0
        self._safety_probe_stage_start = 0.0
        self._safety_probe_stage_timeout = 10.0
        # Stepwise safety probe candidates and index (filled on demand after 20s of failures)
        self._safety_probe_candidates = []
        self._safety_probe_index = -1
        # Per-candidate time budget (seconds) and start time
        self.safety_probe_candidate_max_secs = 6.0
        self._safety_probe_candidate_start = 0.0

    def _save_fruit_locations(self):
        """Persist fruit obstacle locations as enumerated JSON mapping.

        Output example:
        {
            "orange_0": {"x": -3.09, "y": -3.17},
            "pumpkin_0": {"x": 0.94, "y": 0.87}
        }
        Only non-aruco labels are included. Enumeration is based on discovery order.
        """
        try:
            fruit_map: Dict[str, Dict[str, float]] = {}
            # We'll reconstruct enumeration each save to stay consistent with current obstacles
            per_label_counter: Dict[str, int] = {}
            for d in self.discovered_obstacles:
                raw_label = str(d.get('label', ''))
                if raw_label.startswith('aruco'):
                    continue
                base = raw_label
                # If raw_label already has _<n> suffix from prior enumeration, strip it for stable ordering
                if '_' in raw_label:
                    parts = raw_label.rsplit('_', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        base = parts[0]
                idx = per_label_counter.get(base, 0)
                per_label_counter[base] = idx + 1
                enum_label = f"{base}_{idx}"
                fruit_map[enum_label] = {"x": float(d.get('x', 0.0)), "y": float(d.get('y', 0.0))}
            with open(self._fruit_loc_path, 'w') as f:
                json.dump(fruit_map, f, indent=4)
        except Exception:
            pass

    def _load_shopping_list(self) -> set[str]:
        """Load shopping list (base fruit labels) from shopping_list.txt (lowercased).
        Each non-empty line is treated as a label (before any underscore enumeration).
        """
        sl: set[str] = set()
        try:
            path = os.path.join(SCRIPT_DIR, 'shopping_list.txt')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    for line in f:
                        s = line.strip().lower()
                        if not s:
                            continue
                        # ignore comments
                        if s.startswith('#'):
                            continue
                        # strip enumeration suffix if present in file
                        if '_' in s and s.rsplit('_', 1)[1].isdigit():
                            s = s.rsplit('_', 1)[0]
                        sl.add(s)
        except Exception:
            pass
        return sl

    def _base_label(self, lbl: str) -> str:
        """Return base part of enumerated label (pear_0 -> pear)."""
        if '_' in lbl:
            parts = lbl.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                return parts[0]
        return lbl

    def _prepare_fruit_visit_queue(self):
        """Build queue of fruits (enumerated labels) to visit after a patrol waypoint spin.

        Only includes fruits whose base label is in the shopping list and not already
        visited during the current patrol leg. Queue ordered by distance from current pose.
        """
        self._fruit_visit_queue = []
        try:
            if not self.shopping_list:
                # Announce cleared queue if it was previously non-empty
                try:
                    if getattr(self, '_fruit_queue_last_sig', None) not in (None, ()):  # type: ignore[comparison-overlap]
                        self._fruit_queue_last_sig = ()
                        self._announce("Fruit queue cleared")
                except Exception:
                    pass
                return
            x, y, _ = self.get_pose()
            candidates: List[tuple[float, dict]] = []
            for d in self.discovered_obstacles:
                try:
                    lbl = str(d.get('label', '')).lower()
                    if lbl.startswith('aruco'):
                        continue
                    if lbl in self._visited_fruits_cycle:
                        continue
                    base = self._base_label(lbl)
                    if base not in self.shopping_list:
                        continue
                    # Skip if visited already in this run
                    if base in getattr(self, '_visited_fruit_bases', set()):
                        continue
                    fx = float(d.get('x', 0.0))
                    fy = float(d.get('y', 0.0))
                    dist = math.hypot(fx - x, fy - y)
                    candidates.append((dist, d))
                except Exception:
                    continue
            if not candidates:
                # Announce cleared queue if it was previously non-empty
                try:
                    if getattr(self, '_fruit_queue_last_sig', None) not in (None, ()):  # type: ignore[comparison-overlap]
                        self._fruit_queue_last_sig = ()
                        self._announce("Fruit queue cleared")
                except Exception:
                    pass
                return
            candidates.sort(key=lambda t: t[0])
            self._fruit_visit_queue = [d for _, d in candidates]
            # Build a change signature based on base labels (order-insensitive) and announce
            try:
                sig = tuple(sorted([self._base_label(str(d.get('label', '')).lower()) for d in self._fruit_visit_queue]))
                if sig != getattr(self, '_fruit_queue_last_sig', None):
                    self._fruit_queue_last_sig = sig
                    parts = []
                    for d in self._fruit_visit_queue:
                        try:
                            lbl = str(d.get('label', ''))
                            fx = float(d.get('x', 0.0)); fy = float(d.get('y', 0.0))
                            dist = math.hypot(fx - x, fy - y)
                            parts.append(f"{lbl}@{dist:.2f}m")
                        except Exception:
                            continue
                    summary = ", ".join(parts)
                    self._announce(f"Fruit queue updated ({len(self._fruit_visit_queue)}): {summary}")
            except Exception:
                pass
        except Exception:
            pass

    def _start_next_fruit_target(self) -> bool:
        """Begin navigation to next fruit in queue. Returns True if started."""
        if not self._fruit_visit_queue:
            return False
        d = self._fruit_visit_queue[0]
        try:
            fx = float(d.get('x', 0.0))
            fy = float(d.get('y', 0.0))
        except Exception:
            return False
        # Replace remaining_targets with fruit point (do not advance patrol yet)
        self.remaining_targets = [[fx, fy]]
        lbl = str(d.get('label', 'fruit'))
        self.remaining_labels = [lbl]
        # Track the true fruit target position for proximity gating of the hold timer
        self._fruit_target_xy = (fx, fy)
        self._fruit_target_label_base = self._base_label(lbl)
        self._fruit_target_last_update = time.time()
        self.waypoints = []
        self.current_goal = None
        # Enter alignment phase first: center the fruit and dwell before navigating
        self._fruit_align_start = None
        self._fruit_align_pulse_start = None
        self._fruit_align_enter_time = time.time()
        self._mode = 'fruit_align'
        self.notification = f"Aligning to fruit: {self.remaining_labels[0]} (center + {self.fruit_align_duration:.0f}s)"
        # Start deadline timer for reaching this fruit
        try:
            self._fruit_target_deadline = time.time() + float(self.fruit_target_timeout)
        except Exception:
            self._fruit_target_deadline = 0.0
        try:
            self._announce(f"Visiting fruit: {self.remaining_labels[0]} at ({fx:.2f},{fy:.2f})")
        except Exception:
            pass
        return True

    def _start_detour_to_fruit(self, fruit_dict: dict) -> bool:
        """Start an opportunistic detour to a given fruit (dict with x,y,label). Saves patrol state."""
        try:
            fx = float(fruit_dict.get('x', 0.0))
            fy = float(fruit_dict.get('y', 0.0))
            flabel = str(fruit_dict.get('label', 'fruit'))
        except Exception:
            return False
        # Save current patrol target/waypoints to resume later
        try:
            self._saved_patrol_state = {
                'remaining_targets': [rt[:] for rt in (self.remaining_targets or [])],
                'remaining_labels': list(self.remaining_labels or []),
                'waypoints': [wp[:] for wp in (self.waypoints or [])],
                'current_goal': list(self.current_goal) if self.current_goal is not None else None,
                'mode': self._mode,
                'pending_patrol_advance': self._pending_patrol_advance,
            }
        except Exception:
            self._saved_patrol_state = None

        # Replace path with fruit point
        self.remaining_targets = [[fx, fy]]
        self.remaining_labels = [flabel]
        self._fruit_target_xy = (fx, fy)
        # Track base label and last update to allow dynamic re-targeting mid-approach
        try:
            self._fruit_target_label_base = flabel.rsplit('_', 1)[0] if ('_' in flabel and flabel.rsplit('_',1)[1].isdigit()) else flabel
        except Exception:
            self._fruit_target_label_base = flabel
        self._fruit_target_last_update = time.time()
        self.waypoints = []
        self.current_goal = None
        # Enter alignment phase first: center the fruit and dwell before navigating
        self._fruit_align_start = None
        self._fruit_align_pulse_start = None
        self._fruit_align_enter_time = time.time()
        self._mode = 'fruit_align'
        self._detour_active = True
        # Start deadline timer for reaching this fruit (detour)
        try:
            self._fruit_target_deadline = time.time() + float(self.fruit_target_timeout)
        except Exception:
            self._fruit_target_deadline = 0.0
        # Cancel any pending arrival spin from patrol so detour doesn't spin at fruit
        self._arrival_spin_start = None
        self._arrival_spin_pulse_start = None
        self.notification = f"Heading to {flabel} (detour)"
        try:
            self._announce(f"Detouring to fruit: {flabel} at ({fx:.2f},{fy:.2f})")
        except Exception:
            pass
        return True

    def get_fruit_locations(self) -> List[Tuple[str, float, float]]:
        """Return list of (label, x, y) for currently known fruit obstacles."""
        out: List[Tuple[str, float, float]] = []
        for d in self.discovered_obstacles:
            try:
                label = str(d.get('label', ''))
                if label.startswith('aruco'):
                    continue
                out.append((label, float(d.get('x', 0.0)), float(d.get('y', 0.0))))
            except Exception:
                continue
        return out

    def _update_active_fruit_target(self):
        """If navigating to a fruit, update the target to the latest estimate and replan if it moved."""
        try:
            if self._mode not in ('fruit_nav', 'fruit_hold'):
                return
            if self._fruit_target_label_base is None:
                return
            # Find the closest obstacle whose base label matches our target (in case of multiple)
            best = None
            best_dist = 1e9
            curx, cury = (self._fruit_target_xy or (0.0, 0.0))
            for d in self.discovered_obstacles:
                try:
                    lbl = str(d.get('label', ''))
                    base = self._base_label(lbl)
                    if base != self._fruit_target_label_base:
                        continue
                    fx = float(d.get('x', 0.0)); fy = float(d.get('y', 0.0))
                    dd = math.hypot(fx - curx, fy - cury)
                    if dd < best_dist:
                        best_dist = dd
                        best = d
                except Exception:
                    continue
            if best is None:
                return
            fx = float(best.get('x', 0.0)); fy = float(best.get('y', 0.0))
            # If moved significantly vs current target, update and replan with cooldown
            curx, cury = (self._fruit_target_xy or (fx, fy))
            moved = math.hypot(fx - curx, fy - cury)
            now = time.time()
            if moved > max(0.015, self.grid_res * 0.75) and (now - self._fruit_target_last_update) >= self._fruit_target_replan_cooldown:
                self._fruit_target_xy = (fx, fy)
                self._fruit_target_last_update = now
                self.remaining_targets = [[fx, fy]]
                self.waypoints = []
                self.current_goal = None
                try:
                    self.replan(initial=False)
                except Exception:
                    pass
                self.pick_next_goal()
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
        # Reset probe waypoint count flag when starting a new goal
        self._wp_counted = False
        self.reached_time = None
        self.notification = f'Navigating to: [{self.current_goal[0]:.2f}, {self.current_goal[1]:.2f}]'

    # New: count waypoint arrivals during reduced safety probe
    def _on_waypoint_reached(self):
        try:
            if self._safety_probe_active and not self._wp_counted:
                self._wp_counted = True
                self._safety_probe_waypoints_left = max(0, int(self._safety_probe_waypoints_left) - 1)
                if self._safety_probe_waypoints_left <= 0:
                    # Restore margin and replan (end of current probe stage)
                    restore_to = float(self._safety_margin_default if getattr(self, '_safety_margin_default', None) is not None else self.safety_margin)
                    self.safety_margin = restore_to
                    try:
                        print(f"SAfety Margin this: {float(self.safety_margin):.3f}", flush=True)
                    except Exception:
                        pass
                    # Clear probe state so future failures can start over
                    self._safety_probe_active = False
                    self._safety_probe_stage = 0
                    self._safety_probe_stage_start = 0.0
                    self.notification = 'Probe complete (5 waypoints) — restoring safety margin'
                    try:
                        if self.active:
                            self.replan(initial=False)
                            self.pick_next_goal()
                    except Exception:
                        pass
        except Exception:
            pass

    # --- Terminal announcement helper ---
    def _announce(self, msg: str) -> None:
        try:
            print(msg, flush=True)
        except Exception:
            pass

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

        # Draw dynamic ArUco obstacles (red X) from current EKF landmarks
        for ox, oy in self._get_current_aruco_obstacles():
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
        # --- HUD: show per-fruit timeout until skip (top-left corner) ---
        try:
            dl = float(getattr(self, '_fruit_target_deadline', 0.0) or 0.0)
            if dl > 0.0 and self._mode in ('fruit_align', 'fruit_nav'):
                remaining = max(0.0, dl - time.time())
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                lbl = None
                try:
                    if self.remaining_labels:
                        lbl = str(self.remaining_labels[0])
                except Exception:
                    lbl = None
                text = f"{(lbl or 'fruit')} skip in {mins:02d}:{secs:02d}"
                font = getattr(self, 'hud_font', None) or getattr(self, 'label_font', None)
                if font is not None:
                    # Optional background for readability
                    try:
                        txt_surf = font.render(text, True, (255, 255, 255))
                        # Draw a dark rectangle behind text
                        pad = 6
                        rect = pygame.Rect(8, 8, txt_surf.get_width() + pad, txt_surf.get_height() + pad)
                        pygame.draw.rect(canvas, (0, 0, 0), rect)
                        canvas.blit(txt_surf, (rect.x + pad // 2, rect.y + pad // 2))
                    except Exception:
                        pass
        except Exception:
            pass
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
                    #print("Robot position covariance (P[0:2,0:2]):\n", Pxy)
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

        # If SLAM just started, kick off the initial spin once
        if getattr(self, '_initial_spin_start', None) is None and not getattr(self, '_initial_spin_done', False):
            self._initial_spin_start = time.time()

        # Immediate SLAM map save handling (mirror operate.py 's' key behavior)
        # We do this here so the user gets instant feedback in autonomous mode
        try:
            if self.command.get('output', False):
                if hasattr(self, 'output') and self.output is not None:
                    self.output.write_map(self.ekf)
                    self.notification = 'Map saved'
                # reset the flag so base record_data doesn't re-save unnecessarily
                self.command['output'] = False
        except Exception:
            pass

        # --- Initial startup spin (precedes all other behaviors) ---
        if getattr(self, '_initial_spin_start', None) is not None:
            elapsed_init = time.time() - self._initial_spin_start
            if elapsed_init < self._initial_spin_duration:
                # Pulsed rotation to gather landmarks / detections
                period = float(self.nav_turn_pulse_spin_time + self.nav_turn_pulse_stop_time)
                phase = (time.time() - self._initial_spin_start) % period
                if phase < self.nav_turn_pulse_spin_time:
                    self.command['motion'] = [0, self.turn_cmd]
                else:
                    self.command['motion'] = [0, 0]
                remaining = self._initial_spin_duration - elapsed_init
                self.notification = f'Initial spin: {remaining:.1f}s left'
                return
            else:
                # Finish initial spin
                self._initial_spin_start = None
                self._initial_spin_done = True

        # --- Short reverse after finishing a fruit hold (preempts other actions) ---
        try:
            now_rev = time.time()
            if self._fruit_post_reverse_until and now_rev < float(self._fruit_post_reverse_until):
                remaining = float(self._fruit_post_reverse_until) - now_rev
                self.command['motion'] = [-self.fwd_cmd, 0]
                self.notification = f'Post-hold reverse ({remaining:.1f}s)'
                return
            elif self._fruit_post_reverse_until and now_rev >= float(self._fruit_post_reverse_until):
                # Reverse phase finished; clear and, if covariance is high, trigger stabilizing spin now
                self._fruit_post_reverse_until = 0.0
                try:
                    P = getattr(self.ekf, 'P', None)
                    if isinstance(P, np.ndarray) and P.shape[0] >= 2 and P.shape[1] >= 2 and self._mode != 'fruit_hold' and self._arrival_spin_start is None:
                        pxx = float(P[0, 0])
                        if pxx > float(self.cov_pos_thresh):
                            now_cov = time.time()
                            self._cov_spin_dir = -self._cov_spin_dir
                            self._cov_spin_until = now_cov + float(self.cov_spin_duration)
                            self._cov_spin_start = now_cov
                            self.notification = 'High covariance after reverse: stabilizing spin'
                            # Do not return here; allow the normal cov block below to handle pulsing/return
                except Exception:
                    pass
        except Exception:
            pass

        # --- High covariance stabilize spin (preempts other actions) ---
        try:
            now_cov = time.time()
            # Do not start or run covariance spin during fruit hold, fruit alignment, any emergency phase,
            # arrival (waypoint) spin, or safety-probe window
            if (self._mode not in ('fruit_hold', 'fruit_align', 'observe_align')) and getattr(self, '_emergency_mode', None) is None and self._arrival_spin_start is None and not self._safety_probe_active:
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
                    # Spin finished — refresh plan now that pose is stabilized
                    try:
                        if (not self._calib_mode) and self.active and (self._emergency_mode is None) and (self._mode != 'fruit_hold'):
                            self.replan(initial=False)
                            self.pick_next_goal()
                    except Exception:
                        pass
                # If in cooldown, skip covariance checks
                if now_cov < self._cov_cooldown_until:
                    pass  # proceed with normal behavior
                else:
                    # Check EKF position covariance P[0,0]
                    P = getattr(self.ekf, 'P', None)
                    if isinstance(P, np.ndarray) and P.shape[0] >= 2 and P.shape[1] >= 2 and self._arrival_spin_start is None:
                        pxx = float(P[0, 0])
                        if pxx > float(self.cov_pos_thresh):
                            # trigger spin
                            self._cov_spin_dir = -self._cov_spin_dir  # alternate direction
                            self._cov_spin_until = now_cov + float(self.cov_spin_duration)
                            self._cov_spin_start = now_cov
                            self.command['motion'] = [0, self._cov_spin_dir * self.turn_cmd]
                            self.notification = 'High covariance: stabilizing spin'
                            return
            else:
                # While holding at fruit OR in emergency OR arrival spin OR safety-probe: cancel any ongoing covariance spin and defer
                if self._cov_spin_until is not None:
                    self._cov_spin_until = None
                    self._cov_spin_start = None
                self._cov_spin_deferred = True
        except Exception:
            pass

        # Observe-only alignment state: center a newly-seen non-target fruit, dwell, then resume patrol
        if self.enable_observe_align and self._mode == 'observe_align':
            try:
                now_obs = time.time()
                # Initialize enter-time once per observe session
                if getattr(self, '_observe_align_enter_time', None) is None:
                    self._observe_align_enter_time = now_obs
                else:
                    # Enforce max cap for observe align
                    try:
                        if (now_obs - float(self._observe_align_enter_time or now_obs)) >= float(self.observe_align_max_time):
                            # Timeout: end observe and resume patrol
                            disp = self._observe_label_display or 'fruit'
                            try:
                                self._announce(f"Observation timed out (45s): {disp}")
                            except Exception:
                                pass
                            self._observe_align_pulse_start = None
                            self._observe_align_start = None
                            self._observe_align_enter_time = None
                            self._observe_label_base = None
                            self._observe_label_display = None
                            self._mode = 'patrol'
                            self._observe_cooldown_until = now_obs + 8.0
                            self.notification = "Observation timed out — resuming patrol"
                            return
                    except Exception:
                        pass
                base = self._observe_label_base
                centered_recent = False
                if base is not None:
                    try:
                        t_last = float(self._last_centered_label_time.get(base, 0.0))
                        centered_recent = (now_obs - t_last) <= 0.5
                    except Exception:
                        centered_recent = False

                if not centered_recent:
                    # Not centered yet — keep pulse-spinning in place
                    self._observe_align_start = None  # reset dwell timer
                    if self._observe_align_pulse_start is None:
                        self._observe_align_pulse_start = now_obs
                    a_period = float(self.arrival_pulse_spin_time + self.arrival_pulse_stop_time)
                    a_phase = (now_obs - self._observe_align_pulse_start) % a_period
                    if a_phase < self.arrival_pulse_spin_time:
                        self.command['motion'] = [0, self._observe_align_spin_dir * self.turn_cmd]
                    else:
                        self.command['motion'] = [0, 0]
                    disp = self._observe_label_display or 'fruit'
                    self.notification = f"Centering {disp} (observe)…"
                    return

                # Fruit centered: dwell (look at fruit) for observe_align_duration
                if self._observe_align_start is None:
                    self._observe_align_start = now_obs
                    # flip spin direction next time we need to re-center
                    self._observe_align_spin_dir *= -1
                self.command['motion'] = [0, 0]
                remaining = max(0.0, float(self.observe_align_duration) - (now_obs - float(self._observe_align_start or now_obs)))
                disp = self._observe_label_display or 'fruit'
                self.notification = f"Observing {disp} {remaining:.1f}s"
                if (now_obs - float(self._observe_align_start or now_obs)) >= float(self.observe_align_duration):
                    # Observation complete — resume patrol path (no replan required here)
                    self._observe_align_pulse_start = None
                    self._observe_align_start = None
                    self._observe_align_enter_time = None
                    self._observe_label_base = None
                    self._observe_label_display = None
                    self._mode = 'patrol'
                    self._observe_cooldown_until = now_obs + 8.0  # small cooldown before next observe
                    self.notification = "Observation complete — resuming patrol"
                    return
                return
            except Exception:
                pass

        # Periodic calibration trigger (non-blocking start)
        try:
            if (time.time() - self.last_calib_time) >= self.calib_interval and not self._calib_mode:
                # Only start calibration when idle (no active goal) or during arrival spin,
                # to avoid blocking forward motion right after initial spin.
                if (self._arrival_spin_start is not None) or (self.current_goal is None and self._mode == 'patrol'):
                    self.start_calib_scan()
        except Exception:
            pass

        # If currently in calibration mode, do calib step and return (do not replan or move along path)
        if self._calib_mode:
            self._perform_calib_scan_step()
            return

        # If we had temporarily reduced the safety margin to probe a path, restore after window
        try:
            # Waypoint-based restore: if no waypoints left, ensure restored
            if self._safety_probe_active and int(self._safety_probe_waypoints_left) <= 0:
                restore_to = float(self._safety_margin_default)
                self.safety_margin = restore_to
                self._safety_probe_active = False
                self.notification = 'Restored safety margin; replanning'
                try:
                    if self.active:
                        self.replan(initial=False)
                        self.pick_next_goal()
                except Exception:
                    pass
        except Exception:
            pass

        # --- Emergency stop check (overrides scanning and regular motion) ---
        try:
            now = time.time()
            # Emergency state transitions
            if self._emergency_mode == 'prestop' and self._emergency_until <= now:
                # After initial stop, begin reverse phase
                self._emergency_mode = 'reverse'
                self._emergency_until = now + float(self.emergency_reverse_time)
            if self._emergency_mode == 'reverse' and self._emergency_until <= now:
                # Switch to hold phase
                self._emergency_mode = 'hold'
                self._emergency_until = now + float(self.emergency_hold_time)
                # Announce a collection for the object that triggered the emergency
                try:
                    lbl = self._emergency_trigger_label
                    if isinstance(lbl, str) and lbl:
                        self._announce(f"{lbl} collected")
                    # Clear after announcing to avoid repeats on subsequent emergencies
                    self._emergency_trigger_label = None
                except Exception:
                    pass
                # Trigger a replan once when we start holding
                if not self._emergency_replan_triggered and self.active:
                    try:
                        self.replan(initial=False)
                    except Exception:
                        pass
                    self._emergency_replan_triggered = True
            if self._emergency_mode == 'hold' and self._emergency_until <= now:
                # Finish emergency entirely and start cooldown
                self._emergency_mode = None
                self._emergency_cooldown_until = now + float(self.emergency_cooldown)
                self._emergency_until = 0.0
                self._emergency_replan_triggered = False
                # After emergency finishes, if covariance is high, trigger stabilizing spin now
                try:
                    P = getattr(self.ekf, 'P', None)
                    if isinstance(P, np.ndarray) and P.shape[0] >= 2 and P.shape[1] >= 2:
                        pxx = float(P[0, 0])
                        if pxx > float(self.cov_pos_thresh):
                            now_cov2 = time.time()
                            self._cov_spin_dir = -self._cov_spin_dir
                            self._cov_spin_until = now_cov2 + float(self.cov_spin_duration)
                            self._cov_spin_start = now_cov2
                            self.notification = 'High covariance after emergency: stabilizing spin'
                except Exception:
                    pass

            # Active emergency phase handling
            if self._emergency_mode in ('prestop', 'reverse', 'hold') and self._emergency_until > now:
                remaining = self._emergency_until - now
                if self._emergency_mode == 'prestop':
                    # Stop all motion during initial stop window
                    self.command['motion'] = [0, 0]
                    self.notification = f'Emergency: stopping ({remaining:.1f}s)'
                elif self._emergency_mode == 'reverse':
                    # Back up during reverse window
                    self.command['motion'] = [-self.fwd_cmd, 0]
                    self.notification = f'Emergency: reversing ({remaining:.1f}s)'
                else:
                    # Hold still during hold window
                    self.command['motion'] = [0, 0]
                    self.notification = f'Emergency: holding ({remaining:.1f}s)'
                return
            if self.emergency_enabled:
                bboxes = getattr(self, 'detector_output', None)
                if isinstance(bboxes, (list, tuple)):
                    cx = float(self.cx)
                    fx = float(self.fx)
                    for det in bboxes:
                        try:
                            label = str(det[0]).lower()
                            if label.startswith('aruco'):
                                continue
                            xywh = np.asarray(det[1]).astype(float)
                            conf = float(det[2])
                            if conf < 0.6:
                                continue
                            u = float(xywh[0])
                            w_px = float(xywh[2])
                            h_px = float(xywh[3])
                            # Only consider objects roughly in front (near image center)
                            if abs(u - cx) > self.emergency_center_tolerance_px:
                                continue
                            # While approaching a fruit target, ignore emergency stop for that target fruit label
                            # so we can intentionally get close and then perform the timed hold.
                            if self._mode in ('fruit_nav', 'fruit_hold'):
                                try:
                                    target_base = getattr(self, '_fruit_target_label_base', None)
                                    det_base = self._base_label(label)
                                    if target_base is not None and det_base == target_base:
                                        # Skip emergency for the active target fruit only
                                        continue
                                except Exception:
                                    pass
                            # Skip triggers during cooldown window
                            if now < self._emergency_cooldown_until:
                                continue
                            # Trigger if either dimension is large; estimate distance conservatively
                            if (w_px >= self.emergency_bbox_width_thresh_px) or (h_px >= self.emergency_bbox_height_thresh_px):
                                W_assumed = 0.10
                                # Distance from width and height; use smaller (closer) estimate
                                d_w = (fx * W_assumed) / max(1.0, w_px)
                                d_h = (fx * W_assumed) / max(1.0, h_px)
                                d_est = min(d_w, d_h)
                                if d_est <= self.emergency_dist_m:
                                    # Start with an initial stop, then reverse, then hold
                                    stop_t = float(self.emergency_initial_stop_time)
                                    self._emergency_mode = 'prestop'
                                    self._emergency_until = now + stop_t
                                    # Record the base label (preserve original casing) to announce on hold
                                    try:
                                        raw_lbl = str(det[0])
                                        self._emergency_trigger_label = self._base_label(raw_lbl)
                                    except Exception:
                                        try:
                                            self._emergency_trigger_label = str(det[0])
                                        except Exception:
                                            self._emergency_trigger_label = None
                                    # Immediately stop motion
                                    self.command['motion'] = [0, 0]
                                    self.notification = 'Emergency: stopping'
                                    self._emergency_replan_triggered = False
                                    return
                        except Exception:
                            continue
        except Exception:
            pass

        # Marker acquisition gate: scan and occasional creep until >=2 tags visible
        tag_count = len(getattr(self.ekf, 'taglist', []))
        now = time.time()

        # Log pose at ~5 Hz and flush log periodically
        if (now - self._last_pose_log) >= 0.2:
            self._log_pose(now)
            self._last_pose_log = now
        self._flush_log(force=False)
        if tag_count < 2 and self._mode not in ('fruit_nav', 'fruit_hold', 'fruit_align'):
            if not self._planned_once:
                # Kick off a best-effort initial plan so we can start moving even with few tags
                if self.active:
                    try:
                        self.replan(initial=True)
                        self._planned_once = True if (self.waypoints and len(self.waypoints) > 0) else self._planned_once
                        if self._planned_once:
                            self.notification = 'Initial plan created (low tags)'
                    except Exception:
                        pass
                if not self._planned_once:
                    # Still no plan: rotate to search for markers a bit longer
                    if self._scan_start is None:
                        self._scan_start = now
                        self._scan_dir = 1
                    elapsed = now - self._scan_start
                    self._scan_dir = 1 if int(elapsed // 2) % 2 == 0 else -1
                    self.command['motion'] = [0, self._scan_dir * self.turn_cmd]
                    self.notification = 'Looking for markers: scanning'
                    return
            # We have an initial plan (or already had one); don't block navigation due to low tag count
            self._scan_start = None
            self._creep_until = None
        else:
            # Reset scanning state when we have enough tags
            self._scan_start = None
            self._creep_until = None

        # If number of EKF landmarks increased (new ArUco(s) localised), force a replan
        try:
            current_marker_count = 0
            mk = getattr(self.ekf, 'markers', None)
            if isinstance(mk, np.ndarray) and mk.ndim == 2:
                current_marker_count = mk.shape[1]
            if current_marker_count > self._last_marker_count:
                # Only replan if we already had an initial plan (avoid double initial plan)
                if self._planned_once and self.active:
                    self.replan(initial=False)
                self._last_marker_count = current_marker_count
        except Exception:
            pass

        # Periodic replan: update the path every fixed interval to track subtle changes
        try:
            now = time.time()
            if (now - self._last_periodic_replan) >= float(self._periodic_replan_interval):
                # Avoid while calibrating, in emergency phases, or during fruit hold
                if (not self._calib_mode) and self.active and self._planned_once \
                   and self._emergency_mode is None and self._mode != 'fruit_hold':
                    self.replan(initial=False)
                self._last_periodic_replan = now
        except Exception:
            pass

        # Opportunistic detours: while patrolling, only detour to fruits that are already mapped
        if self.enable_opportunistic_detours and self._mode == 'patrol':
            try:
                if self.shopping_list:
                    x, y, _ = self.get_pose()
                    best = None
                    best_dist = 1e9
                    for d in self.discovered_obstacles:
                        try:
                            lbl = str(d.get('label', '')).lower()
                            if lbl.startswith('aruco'):
                                continue
                            base = self._base_label(lbl)
                            if base not in self.shopping_list:
                                continue
                            if base in getattr(self, '_visited_fruit_bases', set()):
                                continue
                            fx = float(d.get('x', 0.0)); fy = float(d.get('y', 0.0))
                            rng = math.hypot(fx - x, fy - y)
                            if rng > self.obs_max_range:
                                continue
                            if rng < best_dist:
                                best_dist = rng
                                best = {'x': fx, 'y': fy, 'label': lbl}
                        except Exception:
                            continue
                    if best is not None:
                        if self._start_detour_to_fruit(best):
                            return
            except Exception:
                pass

        # Fruit alignment state: pulse-spin to center the target fruit, then dwell before navigation
        if (self.enable_fruit_visits or self.enable_opportunistic_detours) and self._mode == 'fruit_align':
            try:
                now_align = time.time()
                # If taking too long to center, skip this fruit and move on
                try:
                    enter_t = float(self._fruit_align_enter_time or now_align)
                    if (now_align - enter_t) >= float(self.fruit_align_max_time):
                        lbl_timeout = None
                        try:
                            if self.remaining_labels:
                                lbl_timeout = self.remaining_labels[0]
                        except Exception:
                            pass
                        self._announce(f"Centering timed out (45s): skipping {lbl_timeout or 'fruit'}")
                        # Remove from front of queue if it matches
                        try:
                            if self._fruit_visit_queue and lbl_timeout:
                                if str(self._fruit_visit_queue[0].get('label','')).lower() == str(lbl_timeout).lower():
                                    self._fruit_visit_queue.pop(0)
                        except Exception:
                            pass
                        # Clear alignment state
                        self._fruit_align_pulse_start = None
                        self._fruit_align_start = None
                        self._fruit_align_enter_time = None
                        # If detouring, restore patrol state; else try next fruit or resume patrol
                        if self._detour_active and self._saved_patrol_state is not None:
                            try:
                                st = self._saved_patrol_state
                                self.remaining_targets = [list(t) for t in (st.get('remaining_targets') or [])] or self.remaining_targets
                                self.remaining_labels = list(st.get('remaining_labels') or self.remaining_labels)
                                self.waypoints = []
                                self.current_goal = None
                                self._mode = 'patrol'
                                self._pending_patrol_advance = bool(st.get('pending_patrol_advance', False))
                                self._detour_active = False
                                self.replan(initial=False)
                                self.pick_next_goal()
                            except Exception:
                                self._mode = 'patrol'
                            finally:
                                self._saved_patrol_state = None
                        else:
                            if self._start_next_fruit_target():
                                return
                            else:
                                self._mode = 'patrol'
                                # If patrol was pending advance, keep that behavior; otherwise just replan
                                try:
                                    self.replan(initial=False)
                                    self.pick_next_goal()
                                except Exception:
                                    pass
                        return
                except Exception:
                    pass
                base = self._fruit_target_label_base
                centered_recent = False
                if base is not None:
                    try:
                        t_last = float(self._last_centered_label_time.get(base, 0.0))
                        centered_recent = (now_align - t_last) <= 0.5
                    except Exception:
                        centered_recent = False

                if not centered_recent:
                    # Not centered yet — keep pulse-spinning in place
                    self._fruit_align_start = None  # reset dwell timer
                    if self._fruit_align_pulse_start is None:
                        self._fruit_align_pulse_start = now_align
                    a_period = float(self.arrival_pulse_spin_time + self.arrival_pulse_stop_time)
                    a_phase = (now_align - self._fruit_align_pulse_start) % a_period
                    if a_phase < self.arrival_pulse_spin_time:
                        self.command['motion'] = [0, self._fruit_align_spin_dir * self.turn_cmd]
                    else:
                        self.command['motion'] = [0, 0]
                    self.notification = f"Centering {self.remaining_labels[0]}…"
                    return

                # Fruit centered: dwell (look at fruit) for fruit_align_duration
                if self._fruit_align_start is None:
                    self._fruit_align_start = now_align
                    # flip spin direction next time we need to re-center
                    self._fruit_align_spin_dir *= -1
                self.command['motion'] = [0, 0]
                remaining = max(0.0, float(self.fruit_align_duration) - (now_align - float(self._fruit_align_start or now_align)))
                self.notification = f"Holding fruit centered {remaining:.1f}s"
                if (now_align - float(self._fruit_align_start or now_align)) >= float(self.fruit_align_duration):
                    # Alignment complete — plan and start navigating to fruit
                    try:
                        self.replan(initial=False)
                    except Exception:
                        pass
                    self.pick_next_goal()
                    self._mode = 'fruit_nav'
                    self._fruit_align_enter_time = None
                    self._fruit_align_pulse_start = None
                    self.notification = f"Heading to fruit: {self.remaining_labels[0]}"
                    return
                # Keep holding still during dwell; do not fall through to other controllers this cycle
                return
            except Exception:
                pass

        # Fruit visit state handling (executed before arrival spin completion when active)
        # If we are navigating specifically to a fruit (opportunistic detours or queued visits)
        if (self.enable_fruit_visits or self.enable_opportunistic_detours) and self._mode == 'fruit_nav':
            try:
                # Enforce per-fruit target deadline: if expired before reaching hold, abandon this fruit
                try:
                    if float(self._fruit_target_deadline or 0.0) > 0.0 and time.time() >= float(self._fruit_target_deadline):
                        # Timeout: skip this fruit and move on
                        lbl_timeout = None
                        try:
                            if self.remaining_labels:
                                lbl_timeout = self.remaining_labels[0]
                        except Exception:
                            pass
                        self._announce(f"Fruit target timed out: {lbl_timeout or 'fruit'} — skipping")
                        # Remove from front of queue if it matches
                        try:
                            if self._fruit_visit_queue:
                                if lbl_timeout and str(self._fruit_visit_queue[0].get('label','')).lower() == str(lbl_timeout).lower():
                                    self._fruit_visit_queue.pop(0)
                        except Exception:
                            pass
                        # If we were detouring, restore patrol; otherwise, try next fruit
                        if self._detour_active and self._saved_patrol_state is not None:
                            try:
                                st = self._saved_patrol_state
                                self.remaining_targets = [list(t) for t in (st.get('remaining_targets') or [])] or self.remaining_targets
                                self.remaining_labels = list(st.get('remaining_labels') or self.remaining_labels)
                                self.waypoints = []
                                self.current_goal = None
                                self._mode = 'patrol'
                                self._pending_patrol_advance = bool(st.get('pending_patrol_advance', False))
                                self._detour_active = False
                                self.replan(initial=False)
                                self.pick_next_goal()
                            except Exception:
                                self._mode = 'patrol'
                            finally:
                                self._saved_patrol_state = None
                        else:
                            # Try next fruit in the queue or resume patrol if none
                            if self._start_next_fruit_target():
                                return
                            else:
                                self._mode = 'patrol'
                                if self._pending_patrol_advance:
                                    self._pending_patrol_advance = False
                                    self._advance_target()
                                    self.replan(initial=False)
                                    self.pick_next_goal()
                                return
                except Exception:
                    pass
                # Use the actual fruit target position (not intermediate waypoints)
                if self._fruit_target_xy is not None:
                    fx, fy = self._fruit_target_xy
                elif self.remaining_targets:
                    fx, fy = self.remaining_targets[0]
                else:
                    fx, fy = self.current_goal
                rx, ry, _ = self.get_pose()
                if math.hypot(fx - rx, fy - ry) <= self.fruit_visit_radius:
                    # Reached fruit: enter hold (we still perform a timed hold on arrival)
                    if self._fruit_hold_start is None:
                        self._fruit_hold_start = time.time()
                        self.notification = f"At fruit {self.remaining_labels[0]} holding"
                        self.command['motion'] = [0, 0]
                        self._mode = 'fruit_hold'
                        # Clear the deadline now that we've reached and started holding at the fruit
                        self._fruit_target_deadline = 0.0
                        return
            except Exception:
                pass
        if (self.enable_fruit_visits or self.enable_opportunistic_detours) and self._mode == 'fruit_hold':
            # Stay still for hold duration, then mark fruit visited and move to next fruit or resume patrol
            if self._fruit_hold_start is not None:
                held = time.time() - self._fruit_hold_start
                remaining_hold = max(0.0, self.fruit_hold_duration - held)
                self.notification = f"Holding at fruit ({remaining_hold:.1f}s)"
                self.command['motion'] = [0, 0]
                if held >= self.fruit_hold_duration:
                    # Mark visited
                    try:
                        if self.remaining_labels:
                            lbl0 = self.remaining_labels[0].lower()
                            self._visited_fruits_cycle.add(lbl0)
                            # Also remember base label globally to avoid revisiting in future detours
                            self._visited_fruit_bases.add(self._base_label(lbl0))
                    except Exception:
                        pass
                    # Announce collection of the fruit in the terminal
                    try:
                        if self.remaining_labels:
                            self._announce(f"{self.remaining_labels[0]} collected")
                    except Exception:
                        pass
                    self._fruit_hold_start = None
                    # Clear stored fruit target once visit completes
                    self._fruit_target_xy = None
                    self._fruit_target_label_base = None
                    # Start a brief reverse phase before resuming navigation
                    try:
                        self._fruit_post_reverse_until = time.time() + float(self.fruit_post_hold_reverse_time)
                    except Exception:
                        self._fruit_post_reverse_until = 0.0
                    # Remove from queue front if matches
                    if self._fruit_visit_queue:
                        try:
                            front_lbl = str(self._fruit_visit_queue[0].get('label','')).lower()
                            if self.remaining_labels and front_lbl == self.remaining_labels[0].lower():
                                self._fruit_visit_queue.pop(0)
                        except Exception:
                            pass
                    # Decide next action
                    if self._fruit_visit_queue and self.enable_fruit_visits:
                        # Start next fruit
                        if self._start_next_fruit_target():
                            return
                    else:
                        # Resume patrol after detour or visit
                        # If we had saved patrol state for detour, restore target and plan
                        if self._saved_patrol_state is not None:
                            try:
                                st = self._saved_patrol_state
                                self.remaining_targets = [list(t) for t in (st.get('remaining_targets') or [])] or self.remaining_targets
                                self.remaining_labels = list(st.get('remaining_labels') or self.remaining_labels)
                                self.waypoints = []  # force replan to current pose
                                self.current_goal = None
                                self._mode = 'patrol'
                                self._pending_patrol_advance = bool(st.get('pending_patrol_advance', False))
                                # Clear detour flag and resume same patrol target
                                self._detour_active = False
                                try:
                                    if self.remaining_targets:
                                        tx, ty = self.remaining_targets[0]
                                        self._announce(f"Resuming patrol to ({tx:.2f},{ty:.2f})")
                                except Exception:
                                    pass
                                self.replan(initial=False)
                                self.pick_next_goal()
                            except Exception:
                                self._mode = 'patrol'
                            finally:
                                self._saved_patrol_state = None
                        else:
                            self._mode = 'patrol'
                            # Only advance if we were in queued fruit visits flow
                            if self._pending_patrol_advance and not self._detour_active:
                                self._pending_patrol_advance = False
                                self._advance_target()
                                self.replan(initial=False)
                                self.pick_next_goal()
                    return
                # If hold time not yet reached, keep holding and do not fall through to normal control
                return

        # Arrival spin completion check (patrol mode only)
        now = time.time()
        if self._mode == 'patrol' and self._arrival_spin_start is not None:
            elapsed_arrival = now - self._arrival_spin_start
            if elapsed_arrival < self.arrival_spin_duration:
                # Increase detection range while spinning at waypoint
                if not self._obs_range_override_active:
                    self.obs_max_range = 0.75
                    self._obs_range_override_active = True
                # Initialize pulse timer
                if self._arrival_spin_pulse_start is None:
                    self._arrival_spin_pulse_start = self._arrival_spin_start
                # Compute phase in arrival spin pulse
                a_period = float(self.arrival_pulse_spin_time + self.arrival_pulse_stop_time)
                a_phase = (now - self._arrival_spin_pulse_start) % a_period
                if a_phase < self.arrival_pulse_spin_time:
                    self.command['motion'] = [0, self.turn_cmd]
                else:
                    self.command['motion'] = [0, 0]
                remaining = self.arrival_spin_duration - elapsed_arrival
                self.notification = f'Spinning at waypoint (pulsed) ({remaining:.1f}s left)'
                return
            else:
                # Spin finished — revert detection range
                if self._obs_range_override_active:
                    self.obs_max_range = 0.48
                    self._obs_range_override_active = False
                # Spin finished -> start a short grace window to allow fruit queue to populate
                self._arrival_spin_start = None
                self._arrival_spin_pulse_start = None
                try:
                    self._post_spin_grace_until = time.time() + float(self.post_spin_grace)
                except Exception:
                    self._post_spin_grace_until = 0.0
                if self.enable_fruit_visits:
                    # Prepare fruit queue (based on current discoveries & shopping list)
                    self._prepare_fruit_visit_queue()
                    if self._fruit_visit_queue:
                        self._mode = 'fruit_nav'
                        self._pending_patrol_advance = True
                        if self._start_next_fruit_target():
                            return
                # If disabled or no fruits yet, wait through grace period before advancing patrol
                remaining = max(0.0, float(self._post_spin_grace_until or 0.0) - time.time())
                self.notification = f'Post-spin grace ({remaining:.1f}s)'
                self.command['motion'] = [0, 0]
                return

        # Handle grace period after arrival spin: wait briefly to allow fruit queue, then decide
        try:
            if self._post_spin_grace_until and time.time() < float(self._post_spin_grace_until):
                remaining = float(self._post_spin_grace_until) - time.time()
                self.notification = f'Post-spin grace ({remaining:.1f}s)'
                self.command['motion'] = [0, 0]
                # While waiting, keep trying to switch to fruit_nav if queue appears
                if self.enable_fruit_visits:
                    self._prepare_fruit_visit_queue()
                    if self._fruit_visit_queue:
                        self._post_spin_grace_until = 0.0
                        self._mode = 'fruit_nav'
                        self._pending_patrol_advance = True
                        if self._start_next_fruit_target():
                            return
                return
            elif self._post_spin_grace_until:
                # Grace expired: clear and advance patrol if still no fruit
                self._post_spin_grace_until = 0.0
                if (not self.enable_fruit_visits) or (not self._fruit_visit_queue):
                    self._advance_target()
                    self.replan(initial=False)
                    self.pick_next_goal()
        except Exception:
            pass

        # Ensure we have a plan from current pose to remaining targets
        if (not self._planned_once and self.active) or (self.active and not self.waypoints):
            self.replan(initial=not self._planned_once)
            self._planned_once = True

        # Get/set goal
        # If navigating to a fruit, keep target synced with latest estimate
        self._update_active_fruit_target()
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

        # Arrival handling: upon reaching waypoint
        # Only start the arrival spin for patrol waypoints. For fruit_nav, keep advancing waypoints
        # (fruit hold will trigger based on distance to the fruit target above).
        if dist <= self.dist_tol:
            if self._mode == 'patrol' and self._is_close_to_current_target([gx, gy]):
                # If a shopping-list fruit is right here, hold at fruit instead of spinning
                try:
                    rx, ry, _ = self.get_pose()
                    near_fruit = None
                    for d in self.discovered_obstacles:
                        try:
                            lbl = str(d.get('label', ''))
                            base = self._base_label(lbl)
                            if base not in self.shopping_list:
                                continue
                            # Skip if this base fruit was already visited in this run
                            if base in getattr(self, '_visited_fruit_bases', set()):
                                continue
                            fx = float(d.get('x', 0.0)); fy = float(d.get('y', 0.0))
                            if math.hypot(fx - rx, fy - ry) <= self.fruit_visit_radius:
                                near_fruit = lbl
                                break
                        except Exception:
                            continue
                    if near_fruit is not None:
                        # Enter fruit hold here
                        if self._fruit_hold_start is None:
                            self._fruit_hold_start = time.time()
                            self._mode = 'fruit_hold'
                            self.notification = f"At fruit {near_fruit} holding"
                            self.command['motion'] = [0, 0]
                            return
                except Exception:
                    pass
                if self._arrival_spin_start is None:
                    self._arrival_spin_start = time.time()
                    self.notification = f'Reached [{gx:.2f},{gy:.2f}] starting spin'
                # Count this waypoint for probe progress
                try:
                    self._on_waypoint_reached()
                except Exception:
                    pass
                # spin logic handled at top of loop; ensure immediate feedback
                self.command['motion'] = [0, self.turn_cmd]
                return
            else:
                # Count this waypoint for probe progress before advancing
                try:
                    self._on_waypoint_reached()
                except Exception:
                    pass
                self.pick_next_goal()
                return

        # Turn-then-drive (with pulsed motion)
        turning = abs(dheading) > self.angle_tol
        if turning:
            now = time.time()
            if self._nav_last_mode != 'turn':
                self._nav_last_mode = 'turn'
            if self._nav_pulse_start is None:
                self._nav_pulse_start = now
            t_period = float(self.nav_turn_pulse_spin_time + self.nav_turn_pulse_stop_time)
            t_phase = (now - self._nav_pulse_start) % t_period
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
            if self._nav_pulse_start is None:
                self._nav_pulse_start = now
            d_period = float(self.nav_drive_pulse_move_time + self.nav_drive_pulse_stop_time)
            d_phase = (now - self._nav_pulse_start) % d_period
            # Drive pulse: move then brief stop (similar style to turn pulsing)
            if d_phase < self.nav_drive_pulse_move_time:
                self.command['motion'] = [self.fwd_cmd, 0]
            else:
                self.command['motion'] = [0, 0]

    # ============= Planning =============
    def _nudge_waypoints_from_obstacles(self,
                                        waypoints: List[List[float]],
                                        obstacles_xy: List[List[float]],
                                        min_clearance: float = 0.10) -> List[List[float]]:
        """Return a copy of waypoints nudged away from the nearest obstacle
        so each is at least min_clearance away. Small, single-step adjustment.
        """
        try:
            if not waypoints or not obstacles_xy:
                return waypoints
            adjusted: List[List[float]] = []
            for wp in waypoints:
                try:
                    wx, wy = float(wp[0]), float(wp[1])
                except Exception:
                    adjusted.append(wp)
                    continue
                nearest_d = 1e9
                n_ox, n_oy = 0.0, 0.0
                for ob in obstacles_xy:
                    try:
                        ox, oy = float(ob[0]), float(ob[1])
                    except Exception:
                        continue
                    dx = wx - ox
                    dy = wy - oy
                    d = math.hypot(dx, dy)
                    if d < nearest_d:
                        nearest_d = d
                        n_ox, n_oy = ox, oy
                if nearest_d < min_clearance and nearest_d > 1e-6:
                    # push outward from the nearest obstacle by the shortfall + small buffer
                    dx = wx - n_ox
                    dy = wy - n_oy
                    nx = dx / nearest_d
                    ny = dy / nearest_d
                    push = (min_clearance + 0.02) - nearest_d
                    wx += nx * max(0.0, push)
                    wy += ny * max(0.0, push)
                adjusted.append([wx, wy])
            return adjusted
        except Exception:
            return waypoints

    def replan(self, initial: bool = False):
        """Plan waypoints from current pose to patrol target, avoiding dynamic obstacles."""
        if self._calib_mode:
            self.notification = "Calibration active — skipping replan"
            return
        if not self.active:
            return
        if not self.remaining_targets:
            self.remaining_targets = [list(self.patrol_points[self.patrol_index])]
            self.remaining_labels = [self.search_list[self.patrol_index]]
        x, y, _ = self.get_pose()
        robot_xy = [x, y]
        obstacles_xy: List[List[float]] = []
        # ArUco markers: apply a small fixed keep-out radius of 0.04m
        for (ox, oy) in self._get_current_aruco_obstacles():
            obstacles_xy.append([float(ox), float(oy), 0.04])
        # Fruit and other discovered obstacles: use default planner clearance (no explicit radius)
        obstacles_xy.extend([[float(d['x']), float(d['y'])] for d in self.discovered_obstacles])
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
            # Post-process: nudge waypoints away from obstacles if too close
            try:
                # For nudging and clamping, use just positions
                ob_xy_positions: List[List[float]] = []
                for ob in obstacles_xy:
                    if isinstance(ob, (list, tuple)) and len(ob) >= 2:
                        ob_xy_positions.append([float(ob[0]), float(ob[1])])
                new_waypoints = self._nudge_waypoints_from_obstacles(new_waypoints, ob_xy_positions, min_clearance=0.10)
                # Clamp within arena inner boundary if configured (tiny inset to avoid exact wall)
                inner = max(0.0, float(self.arena_half) - float(self.wall_clearance))
                if inner > 0.0 and new_waypoints:
                    clamped: List[List[float]] = []
                    inset = 0.01
                    for w in new_waypoints:
                        try:
                            wx, wy = float(w[0]), float(w[1])
                            wx = max(-inner + inset, min(inner - inset, wx))
                            wy = max(-inner + inset, min(inner - inset, wy))
                            clamped.append([wx, wy])
                        except Exception:
                            clamped.append(w)
                    new_waypoints = clamped
            except Exception:
                pass
            self.waypoints = new_waypoints
            self.current_goal = None
            msg_prefix = 'Planned' if initial else 'Replanned'
            self.notification = f'{msg_prefix} {len(self.waypoints)} waypoints (patrol)'
            self._log_plan()
            self._flush_log(force=False)
            # Reset the plan-failure watchdog on success
            self._plan_fail_start = None
        except Exception as e:
            self.notification = f'Planning failed: {e}'
            # Start or update the plan-failure watchdog
            now_fail = time.time()
            if self._plan_fail_start is None:
                self._plan_fail_start = now_fail
                self._plan_fail_target_snapshot = (list(self.remaining_targets[0]) if self.remaining_targets else None)
            else:
                # If target changed since we started failing, reset the timer — unless a safety probe is active.
                # While probing reduced safety margins, we want the 20s window to continue accruing to allow
                # the probe sequence to run without being interrupted by timer resets.
                current_target_snapshot = (list(self.remaining_targets[0]) if self.remaining_targets else None)
                if current_target_snapshot != getattr(self, '_plan_fail_target_snapshot', None):
                    # Always update the snapshot to avoid repeated mismatches, but only reset the timer
                    # if we are not actively probing safety margins.
                    self._plan_fail_target_snapshot = current_target_snapshot
                    if not getattr(self, '_safety_probe_active', False):
                        self._plan_fail_start = now_fail

            # If we've been failing to plan for long enough, either stabilize or skip to next target
            fail_elapsed = now_fail - float(self._plan_fail_start or now_fail)
            # First threshold: covariance spin assist (kept at existing threshold)
            if fail_elapsed >= float(self._plan_fail_threshold):
                # If a safety probe is in progress, skip starting a covariance spin.
                if getattr(self, '_safety_probe_active', False):
                    pass
                elif (self._cov_spin_until is None) and (not self._calib_mode) and (self._emergency_mode is None) and (self._mode != 'fruit_hold') and (self._arrival_spin_start is None):
                    self._cov_spin_dir = -self._cov_spin_dir  # alternate direction
                    self._cov_spin_until = now_fail + float(self.cov_spin_duration)
                    self._cov_spin_start = now_fail
                    self.notification = 'Planning failed for 10s: starting stabilizing spin'
                    # Do not reset plan fail timer yet; allow it to continue accruing toward skip
            # Second threshold: after 20s, begin timed multi-stage safety probe
            if fail_elapsed >= 20.0:
                try:
                    now_ts = now_fail
                    if not self._safety_probe_active:
                        # Stage 1: 0.045
                        self._safety_margin_saved = getattr(self, '_safety_margin_saved', None) or self.safety_margin
                        self.safety_margin = 0.045
                        try:
                            print(f"SAfety Margin this: {float(self.safety_margin):.3f}", flush=True)
                        except Exception:
                            pass
                        self._safety_probe_active = True
                        self._safety_probe_stage = 1
                        self._safety_probe_stage_start = now_ts
                        self._safety_probe_waypoints_left = 5
                        # Cancel any ongoing covariance spin and suppress while probing
                        if self._cov_spin_until is not None:
                            self._cov_spin_until = None
                            self._cov_spin_start = None
                        try:
                            self._cov_cooldown_until = time.time() + max(2.0, self.cov_spin_cooldown)
                        except Exception:
                            pass
                        self.notification = f"Planning failed >20s: using safety margin {self.safety_margin:.3f} (stage 1)"
                        if self.active:
                            self.replan(initial=False)
                            self.pick_next_goal()
                            return
                    else:
                        # Already probing — check stage timeouts and escalate if still failing
                        stage = int(getattr(self, '_safety_probe_stage', 0))
                        started = float(getattr(self, '_safety_probe_stage_start', now_ts))
                        timeout = float(getattr(self, '_safety_probe_stage_timeout', 10.0))
                        if stage == 1 and (now_ts - started) >= timeout:
                            # Escalate to Stage 2: 0.020
                            self.safety_margin = 0.020
                            try:
                                print(f"SAfety Margin this: {float(self.safety_margin):.3f}", flush=True)
                            except Exception:
                                pass
                            self._safety_probe_stage = 2
                            self._safety_probe_stage_start = now_ts
                            self._safety_probe_waypoints_left = 5
                            # Cancel any ongoing covariance spin
                            if self._cov_spin_until is not None:
                                self._cov_spin_until = None
                                self._cov_spin_start = None
                            try:
                                self._cov_cooldown_until = time.time() + max(2.0, self.cov_spin_cooldown)
                            except Exception:
                                pass
                            self.notification = f"Escalating safety probe: margin {self.safety_margin:.3f} (stage 2)"
                            if self.active:
                                self.replan(initial=False)
                                self.pick_next_goal()
                                return
                        elif stage == 2 and (now_ts - started) >= timeout:
                            # Escalate to Stage 3: 0.000
                            self.safety_margin = 0.000
                            try:
                                print(f"SAfety Margin this: {float(self.safety_margin):.3f}", flush=True)
                            except Exception:
                                pass
                            self._safety_probe_stage = 3
                            self._safety_probe_stage_start = now_ts
                            self._safety_probe_waypoints_left = 5
                            if self._cov_spin_until is not None:
                                self._cov_spin_until = None
                                self._cov_spin_start = None
                            try:
                                self._cov_cooldown_until = time.time() + max(2.0, self.cov_spin_cooldown)
                            except Exception:
                                pass
                            self.notification = f"Escalating safety probe: margin {self.safety_margin:.3f} (stage 3)"
                            if self.active:
                                self.replan(initial=False)
                                self.pick_next_goal()
                                return
                        # Stage 3 has no further escalation; restoration occurs on waypoint arrivals
                except Exception:
                    pass

    def _advance_target(self):
        if not hasattr(self, 'patrol_points') or len(self.patrol_points) == 0:
            return
        self.patrol_index = (self.patrol_index + 1) % len(self.patrol_points)
        self.remaining_targets = [list(self.patrol_points[self.patrol_index])]
        self.remaining_labels = [self.search_list[self.patrol_index]]
        self.reached_time = None
        ntx, nty = self.remaining_targets[0]
        keep_obs: List[dict] = []
        for d in self.discovered_obstacles:
            ox, oy = float(d['x']), float(d['y'])
            if math.hypot(ox - ntx, oy - nty) > max(0.12, self.grid_res * 2):
                keep_obs.append(d)
        self.discovered_obstacles = keep_obs
        self.notification = f'Patrol advancing to ({ntx:.2f},{nty:.2f})'
        try:
            self._announce(f"Going to waypoint {self.remaining_labels[0]} at ({ntx:.2f},{nty:.2f})")
        except Exception:
            pass

    def _is_close_to_current_target(self, goal_xy: List[float]) -> bool:
        if not self.remaining_targets:
            return False
        tx, ty = self.remaining_targets[0]
        return math.hypot(goal_xy[0] - tx, goal_xy[1] - ty) <= max(0.12, self.grid_res * 3)

    # --- Dynamic ArUco obstacle helper (missing earlier) ---
    def _get_current_aruco_obstacles(self) -> List[Tuple[float, float]]:
        """Return current EKF landmark (AruCo) positions as world-frame obstacle coordinates.

        Landmarks are stored in ekf.markers as a 2xN array (x; y). We simply copy them.
        Returns empty list if EKF or markers not available yet.
        """
        obs: List[Tuple[float, float]] = []
        try:
            ekf = getattr(self, 'ekf', None)
            if ekf is None:
                return obs
            mk = getattr(ekf, 'markers', None)
            if isinstance(mk, np.ndarray) and mk.ndim == 2 and mk.shape[0] >= 2:
                for i in range(mk.shape[1]):
                    obs.append((float(mk[0, i]), float(mk[1, i])))
        except Exception:
            pass
        return obs

    # ============= Perception integration =============
    def periodic_perception_update(self):
        """Process detector outputs to add unknown obstacles, and replan if new obstacles observed."""
        # Do not update fruit locations during the initial spin window
        if getattr(self, '_initial_spin_start', None) is not None and not getattr(self, '_initial_spin_done', False):
            return
        bboxes = getattr(self, 'detector_output', None)
        if not isinstance(bboxes, (list, tuple)) or len(bboxes) == 0:
            return
        now = time.time()
        new_added = False
        fruit_changed = False
        center_found = False  # Track if any detection is in the center third this iteration

        # Prune stale fruit obstacles (not updated recently)
        # if self.discovered_obstacles:
        #     keep: List[dict] = []
        #     pruned = False
        #     for d in self.discovered_obstacles:
        #         last_seen = float(d.get('last_seen', d.get('t', now)))
        #         if (now - last_seen) <= self.fruit_stale_time:
        #             keep.append(d)
        #         else:
        #             pruned = True
        #     if pruned:
        #         self.discovered_obstacles = keep
        #         new_added = True
        #         fruit_changed = True

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
            if conf < 0.8:
                continue

            # Only update fruit locations for detections in the center third of the image
            try:
                u = float(xywh[0])
                # Estimate image width: prefer 2*cx when intrinsics available, else assume 320px
                img_w = float(self.cx) * 2.0 if getattr(self, 'cx', None) is not None else 320.0
                center_tol = img_w / 6.0  # half-width of the center third
                if abs(u - float(self.cx)) > center_tol:
                    # Skip updates for off-center detections to reduce jittery position changes
                    continue
                else:
                    # Mark that a centered fruit is present this loop and set a transient message
                    if not label.startswith('aruco'):
                        center_found = True
                        self.notification = "Fruit is in center"
                        # Record last-centered time for this fruit base label (for alignment phase)
                        try:
                            base = self._base_label(label)
                            self._last_centered_label_time[base] = now
                        except Exception:
                            pass
            except Exception:
                # If anything goes wrong computing the check, fall back to allowing the update
                pass

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

            # No partial map suppression in patrol mode

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
                # only merge identical base labels (strip enumeration suffix if present)
                existing_label = str(d.get('label', ''))
                base_existing = existing_label.rsplit('_', 1)[0] if ('_' in existing_label and existing_label.rsplit('_',1)[1].isdigit()) else existing_label
                base_new = label.rsplit('_', 1)[0] if ('_' in label and label.rsplit('_',1)[1].isdigit()) else label
                if base_existing == base_new:
                    px, py = float(d['x']), float(d['y'])
                    if math.hypot(ox - px, oy - py) <= use_merge_thr:
                        # Kalman-style update (static model): x' = x, P' = P + Q; z = measurement
                        try:
                            # Initialize covariance if absent
                            if 'P' not in d or not isinstance(d['P'], np.ndarray):
                                d['P'] = np.diag([0.04, 0.04])  # initial 20cm std dev squared
                            P_prev = d['P']
                            if not isinstance(P_prev, np.ndarray) or P_prev.shape != (2, 2):
                                P_prev = np.diag([0.04, 0.04])
                            Q = self.fruit_Q
                            R = self.fruit_R
                            # Predict
                            P_pred = P_prev + Q
                            z = np.array([ox, oy])
                            x_prev_vec = np.array([px, py])
                            S = P_pred + R
                            K = P_pred @ np.linalg.inv(S)
                            innovation = z - x_prev_vec
                            x_new_vec = x_prev_vec + K @ innovation
                            P_new = (np.eye(2) - K) @ P_pred
                            new_x = float(x_new_vec[0])
                            new_y = float(x_new_vec[1])
                            moved = math.hypot(new_x - px, new_y - py)
                            d['x'] = new_x
                            d['y'] = new_y
                            d['P'] = P_new
                            d['count'] = int(d.get('count', 1)) + 1
                            d['last_seen'] = now
                            self._log_obstacle(new_x, new_y, label=label, method=('kf-tpe' if used_tpe else 'kf-heuristic'))
                            self._flush_log(force=False)
                            if moved > self.fruit_replan_move_thr:
                                new_added = True
                            fruit_changed = True
                        except Exception:
                            # Fallback to simple smoothing
                            alpha = float(self.fruit_update_alpha)
                            new_x = (1 - alpha) * px + alpha * ox
                            new_y = (1 - alpha) * py + alpha * oy
                            moved = math.hypot(new_x - px, new_y - py)
                            d['x'] = new_x
                            d['y'] = new_y
                            d['count'] = int(d.get('count', 1)) + 1
                            d['last_seen'] = now
                            self._log_obstacle(new_x, new_y, label=label, method=('fallback-update'))
                            self._flush_log(force=False)
                            if moved > self.fruit_replan_move_thr:
                                new_added = True
                            fruit_changed = True
                        merged = True
                        break

            if merged:
                # merged into an existing cluster; skip the rest of add process
                continue

            # --- Special cross-label merge rules (10 cm) ---
            # If capsicum and tomato are within 10 cm, merge as capsicum.
            # If lemon and orange are within 10 cm, merge as orange.
            try:
                preferred_map = {
                    ('capsicum', 'tomato'): 'capsicum',
                    ('tomato', 'capsicum'): 'capsicum',
                    ('lemon', 'orange'): 'orange',
                    ('orange', 'lemon'): 'orange',
                }
                base_new = label.rsplit('_', 1)[0] if ('_' in label and label.rsplit('_',1)[1].isdigit()) else label
                did_special_merge = False
                for d in self.discovered_obstacles:
                    existing_label = str(d.get('label', ''))
                    base_existing = existing_label.rsplit('_', 1)[0] if ('_' in existing_label and existing_label.rsplit('_',1)[1].isdigit()) else existing_label
                    pref = preferred_map.get((base_existing, base_new))
                    if not pref:
                        continue
                    px, py = float(d.get('x', 0.0)), float(d.get('y', 0.0))
                    dist_xy = math.hypot(ox - px, oy - py)
                    if dist_xy <= 0.10:
                        # Update the obstacle position using KF-like update when possible
                        moved = 0.0
                        try:
                            if 'P' not in d or not isinstance(d['P'], np.ndarray) or d['P'].shape != (2, 2):
                                d['P'] = np.diag([0.04, 0.04])
                            P_prev = d['P']
                            Q = self.fruit_Q
                            R = self.fruit_R
                            P_pred = P_prev + Q
                            z = np.array([ox, oy])
                            x_prev_vec = np.array([px, py])
                            S = P_pred + R
                            K = P_pred @ np.linalg.inv(S)
                            innovation = z - x_prev_vec
                            x_new_vec = x_prev_vec + K @ innovation
                            P_new = (np.eye(2) - K) @ P_pred
                            new_x = float(x_new_vec[0])
                            new_y = float(x_new_vec[1])
                            moved = math.hypot(new_x - px, new_y - py)
                            d['x'] = new_x
                            d['y'] = new_y
                            d['P'] = P_new
                            d['count'] = int(d.get('count', 1)) + 1
                            d['last_seen'] = now
                            self._log_obstacle(new_x, new_y, label=pref, method='special-merge-kf')
                            self._flush_log(force=False)
                        except Exception:
                            # Fallback smoothing update
                            alpha = float(self.fruit_update_alpha)
                            new_x = (1 - alpha) * px + alpha * ox
                            new_y = (1 - alpha) * py + alpha * oy
                            moved = math.hypot(new_x - px, new_y - py)
                            d['x'] = new_x
                            d['y'] = new_y
                            d['count'] = int(d.get('count', 1)) + 1
                            d['last_seen'] = now
                            self._log_obstacle(new_x, new_y, label=pref, method='special-merge')
                            self._flush_log(force=False)

                        # Relabel obstacle to the preferred base if needed
                        if base_existing != pref:
                            try:
                                idx_pref = int(self._fruit_label_counts.get(pref, 0))
                                enum_pref = f"{pref}_{idx_pref}"
                                self._fruit_label_counts[pref] = idx_pref + 1
                                d['label'] = enum_pref
                            except Exception:
                                d['label'] = pref
                        if moved > self.fruit_replan_move_thr:
                            fruit_changed = True
                        did_special_merge = True
                        break
                if did_special_merge:
                    # We handled this detection via special merge; skip normal add/merge
                    continue
            except Exception:
                pass

            # --- Cross-label merge: if a different fruit is within a small radius, update the first one ---
            # Prevents duplicate obstacles when two different labels are reported for the same physical fruit.
            try:
                nearest_idx = -1
                nearest_dist = 1e9
                for idx, d in enumerate(self.discovered_obstacles):
                    existing_label = str(d.get('label', ''))
                    base_existing = existing_label.rsplit('_', 1)[0] if ('_' in existing_label and existing_label.rsplit('_',1)[1].isdigit()) else existing_label
                    base_new = label.rsplit('_', 1)[0] if ('_' in label and label.rsplit('_',1)[1].isdigit()) else label
                    if base_existing == base_new:
                        # Skip same-label here; handled above
                        continue
                    px, py = float(d.get('x', 0.0)), float(d.get('y', 0.0))
                    dist_xy = math.hypot(ox - px, oy - py)
                    if dist_xy <= float(self.cross_label_merge_radius) and dist_xy < nearest_dist:
                        nearest_dist = dist_xy
                        nearest_idx = idx
                if nearest_idx >= 0:
                    d = self.discovered_obstacles[nearest_idx]
                    px, py = float(d.get('x', 0.0)), float(d.get('y', 0.0))
                    moved = 0.0
                    try:
                        # Use Kalman-style update if covariance present (initialize if missing)
                        if 'P' not in d or not isinstance(d['P'], np.ndarray) or d['P'].shape != (2, 2):
                            d['P'] = np.diag([0.04, 0.04])
                        P_prev = d['P']
                        Q = self.fruit_Q
                        R = self.fruit_R
                        P_pred = P_prev + Q
                        z = np.array([ox, oy])
                        x_prev_vec = np.array([px, py])
                        S = P_pred + R
                        K = P_pred @ np.linalg.inv(S)
                        innovation = z - x_prev_vec
                        x_new_vec = x_prev_vec + K @ innovation
                        P_new = (np.eye(2) - K) @ P_pred
                        new_x = float(x_new_vec[0])
                        new_y = float(x_new_vec[1])
                        moved = math.hypot(new_x - px, new_y - py)
                        d['x'] = new_x
                        d['y'] = new_y
                        d['P'] = P_new
                        d['count'] = int(d.get('count', 1)) + 1
                        d['last_seen'] = now
                        self._log_obstacle(new_x, new_y, label=str(d.get('label','')), method='cross-merge-kf')
                        self._flush_log(force=False)
                    except Exception:
                        # Fallback smoothing update
                        alpha = float(self.fruit_update_alpha)
                        new_x = (1 - alpha) * px + alpha * ox
                        new_y = (1 - alpha) * py + alpha * oy
                        moved = math.hypot(new_x - px, new_y - py)
                        d['x'] = new_x
                        d['y'] = new_y
                        d['count'] = int(d.get('count', 1)) + 1
                        d['last_seen'] = now
                        self._log_obstacle(new_x, new_y, label=str(d.get('label','')), method='cross-merge')
                        self._flush_log(force=False)
                    if moved > self.fruit_replan_move_thr:
                        new_added = True
                    fruit_changed = True
                    # Do not create a new obstacle for this detection
                    continue
            except Exception:
                # If anything goes wrong, fall back to normal flow
                pass

            # Ignore duplicates amongst known and discovered obstacles (any label) using a wider gate
            all_obs = []
            # known_obstacles unused; dynamic markers handled separately
            all_obs.extend([[d['x'], d['y']] for d in self.discovered_obstacles])
            if any(math.hypot(ox - qx, oy - qy) <= self.min_obs_separation for qx, qy in all_obs):
                continue

            # Rate limit
            if now - self.last_obstacle_add_time < self.add_cooldown:
                continue

            # Enumerate label
            base_label = label.rsplit('_',1)[0] if ('_' in label and label.rsplit('_',1)[1].isdigit()) else label
            idx = int(self._fruit_label_counts.get(base_label, 0))
            enum_label = f"{base_label}_{idx}"
            self._fruit_label_counts[base_label] = idx + 1
            # Add obstacle and mark for replanning
            self.discovered_obstacles.append({'x': float(ox), 'y': float(oy), 'label': enum_label, 'count': 1, 'last_seen': now, 'P': np.diag([0.04, 0.04])})
            self._log_obstacle(ox, oy, label=enum_label, method=('tpe' if used_tpe else 'heuristic'))
            self._flush_log(force=False)
            self.last_obstacle_add_time = now
            new_added = True
            fruit_changed = True

            # If configured, briefly align/observe newly added non-target fruit to refine its position
            try:
                if self.enable_observe_align and self._mode == 'patrol' and not self._calib_mode and getattr(self, '_emergency_mode', None) is None:
                    if time.time() >= float(self._observe_cooldown_until or 0.0):
                        self._observe_label_base = base_label
                        self._observe_label_display = enum_label
                        self._observe_align_start = None
                        self._observe_align_pulse_start = None
                        # Alternate spin direction between observations to reduce bias
                        self._observe_align_spin_dir *= -1
                        self._mode = 'observe_align'
                        self.notification = f"Observing {enum_label}: centering before resuming"
            except Exception:
                pass

        if fruit_changed:
            self._save_fruit_locations()
        if new_added and self.ekf_on and self.active:
            self.replan(initial=False)
        # If no centered detection this iteration, do not overwrite notification; it will revert
        # naturally to whatever auto_nav_step set earlier in the loop.


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Autonomous Patrol with Dynamic Obstacles + GUI/SLAM")
    parser.add_argument("--ip", type=str, default="192.168.50.1")
    parser.add_argument("--port", type=int, default=8080)
    # Use local Week07-08 calibration parameters by default now that code is consolidated
    parser.add_argument("--calib_dir", type=str, default=os.path.join(SCRIPT_DIR, "calibration", "param") + os.sep)
    parser.add_argument("--yolo_model", default=os.path.join(SCRIPT_DIR, "YOLO", "model", "bestv5.pt"))
    # Legacy map/list args retained for compatibility but unused
    parser.add_argument("--map", type=str, default="")
    parser.add_argument("--list", type=str, default="")
    parser.add_argument("--grid_res", type=float, default=0.02)
    parser.add_argument("--robot_radius", type=float, default=0.14)
    parser.add_argument("--safety_margin", type=float, default=0.099)
    # Merge threshold (main option). You can also use --merge_thresh alias below
    parser.add_argument("--merge_threshold", type=float, default=0.35,
                        help="Merge radius (meters) for clustering detections of the same fruit label")
    # only count/add obstacles when seen within this distance (meters)
    parser.add_argument("--obs_max_range", type=float, default=0.48)
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--quadrants", type=str, default="1234",
                        help="Reorder default patrol points using a 4-digit code with digits 1-4 (e.g., 2341)")
    args, _ = parser.parse_known_args()

    # Provide globals to operate.py expectations (previously Week05-06, now local)
    op_args = SimpleNamespace(
        ip=args.ip, port=args.port, calib_dir=args.calib_dir,
        yolo_model=args.yolo_model, play_data=args.play_data, save_data=args.save_data,
        quadrants=args.quadrants,
    )
    operate_mod.args = op_args

    # Fonts/icons for Operate GUI
    pygame.font.init()
    TITLE_FONT = pygame.font.Font(os.path.join(SCRIPT_DIR, 'pics', '8-BitMadness.ttf'), 35)
    TEXT_FONT = pygame.font.Font(os.path.join(SCRIPT_DIR, 'pics', '8-BitMadness.ttf'), 40)
    operate_mod.TITLE_FONT = TITLE_FONT
    operate_mod.TEXT_FONT = TEXT_FONT

    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 - Auto Fruit Search (L3)')
    try:
        pygame.display.set_icon(pygame.image.load(os.path.join(SCRIPT_DIR, 'pics', '8bit', 'pibot5.png')))
    except Exception:
        pass
    canvas.fill((0, 0, 0))

    # Parse patrol points from optional environment/CLI input.
    # Preferred numeric form: --waypoints x1 y1 x2 y2 x3 y3 x4 y4
    # Back-compat string form: --patrol_points "x1,y1;x2,y2;x3,y3;x4,y4"
    def _parse_patrol(s: str) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        if not s:
            return pts
        for part in s.split(';'):
            part = part.strip()
            if not part:
                continue
            try:
                a, b = part.split(',')
                pts.append((float(a), float(b)))
            except Exception:
                continue
        return pts
    # New: numeric waypoints (8 floats)
    parser.add_argument("--waypoints", nargs=8, type=float, metavar=(
        "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"),
        help="Four waypoints as 8 numbers: x1 y1 x2 y2 x3 y3 x4 y4")
    # Back-compat: string waypoints
    parser.add_argument("--patrol_points", type=str, default="",
                        help="Four waypoints in 'x1,y1;x2,y2;x3,y3;x4,y4' format")
    # Alias for merge threshold if you prefer shorter name; maps to same dest
    parser.add_argument("--merge_thresh", type=float, dest="merge_threshold",
                        help="Alias for --merge_threshold")
    # Re-parse to include new argument (since we added after initial parse)
    args = parser.parse_args()
    # Build patrol points from preferred --waypoints, otherwise --patrol_points
    patrol_pts: List[Tuple[float, float]] = []
    wp_list = getattr(args, 'waypoints', None)
    if isinstance(wp_list, list) and len(wp_list) == 8:
        try:
            patrol_pts = [(float(wp_list[i]), float(wp_list[i+1])) for i in range(0, 8, 2)]
        except Exception:
            patrol_pts = []
    if not patrol_pts:
        patrol_pts = _parse_patrol(getattr(args, 'patrol_points', ''))

    # Ensure relative asset paths in operate.py resolve from current directory (now local)
    try:
        os.chdir(SCRIPT_DIR)
    except Exception:
        pass

    operate = AutoOperateDynamic(op_args,
                                 grid_res=args.grid_res,
                                 robot_radius=args.robot_radius,
                                 safety_margin=args.safety_margin,
                                 merge_threshold=args.merge_threshold,
                                 obs_max_range=args.obs_max_range,
                                 patrol_points=patrol_pts)

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
