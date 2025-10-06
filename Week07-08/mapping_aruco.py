"""Autonomous ArUco + Fruit Obstacle Mapping (Reactive Pulsed Explorer)

Adds fruit/obstacle detection & reactive avoidance to the earlier minimal
mapper. Still omits global A* path planning; instead, uses a forward-cone
blocking heuristic to pulse-turn around obstacles (markers + detected fruits).

Key Behaviours:
  - Pulsed scan (in-place rotation) then pulsed forward creep.
  - On first marker: approach within 0.2 m, perform a full 360° pulsed spin.
  - Continuous YOLO detection; fruits projected (TargetPoseEst if available
    else heuristic) -> merged into discovered obstacle list.
  - Markers added to known obstacles list on first sighting.
  - Reactive avoidance: if an obstacle lies within a forward arc (±30° inside
    0.35 m) during drive phase, switch to pulsed turning to clear path.
  - Logging: poses, markers, and obstacles (with method used) to JSON plus
    simple text dump of marker map.

Outputs (JSON): Week07-08/lab_output/aruco_mapping_log.json
  meta: params
  poses: [t,x,y,theta]
  arucos: {id,x,y,t}
  obstacles: {t,x,y,label,method}

This file supersedes prior simpler mapping_aruco.py by adding fruit obstacle
integration while keeping GUI look-and-feel and SLAM toggle semantics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pygame  # type: ignore
import numpy as np  # type: ignore

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
WEEK0506_DIR = os.path.join(REPO_ROOT, "Week05-06")

# Ensure Week05-06 is importable (for operate.py)
sys.path.insert(0, WEEK0506_DIR)
try:
    from operate import Operate  # type: ignore
except Exception:
    op_file = os.path.join(WEEK0506_DIR, "operate.py")
    spec = importlib.util.spec_from_file_location("operate", op_file)
    operate_mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(operate_mod)  # type: ignore
    Operate = operate_mod.Operate  # type: ignore
else:  # Provide operate_mod reference for font injection later
    import operate as operate_mod  # type: ignore

# Optional TargetPoseEst import for fruit projection (if available)
try:  # pragma: no cover
    from TargetPoseEst import estimate_pose  # type: ignore
except Exception:  # pragma: no cover
    estimate_pose = None  # type: ignore


def normalize_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def angle_diff(a: float, b: float) -> float:
    return normalize_angle(a - b)


class AutoArUcoMapper(Operate):
    def __init__(self, args, merge_distance: float = 0.10, export_interval: float = 2.0, drive_duration: float = 2.5, scan_toggle_period: float | None = None):
        super().__init__(args)
        self.command['inference'] = True

        # Motion pulse parameters
        self.turn_spin_time = 0.4
        self.turn_stop_time = 0.2
        self.drive_period = 0.8
        self.drive_stop_time = 0.2
        self.drive_linear_speed = 1
        self.turn_speed_cmd = 1

        self.scan_duration = 6.0
        # Allow longer forward exploration via parameter
        self.drive_duration = max(0.5, float(drive_duration))  # safety lower bound
        self._phase = 'scan'
        self._phase_start = time.time()
        self._turn_dir = 1
        # New: period (seconds) before reversing scan direction; if >= scan_duration => no flip within a scan
        self.scan_toggle_period = float(scan_toggle_period) if scan_toggle_period is not None else self.scan_duration

        # First-marker acquisition & spin phases
        self._first_marker_id: int | None = None
        self._first_marker_target: Tuple[float, float] | None = None
        self._marker_acquired = False
        self.acquire_dist_tol = 0.20
        self.heading_tol = math.radians(8.0)
        self._spin_start_heading: float | None = None
        self._spin_last_heading: float | None = None
        self._spin_accum_angle = 0.0
        self._spin_dir = 1

        # Logging
        self._log: Dict[str, Any] = {
            'meta': {
                'merge_distance': merge_distance,
                'scan_duration': self.scan_duration,
                'drive_duration': self.drive_duration,
                'turn_spin_time': self.turn_spin_time,
                'turn_stop_time': self.turn_stop_time,
                'drive_period': self.drive_period,
                'drive_stop_time': self.drive_stop_time,
            },
            'poses': [],
            'arucos': []
        }
        week_dir = os.path.join(REPO_ROOT, 'Week07-08')
        out_dir = os.path.join(week_dir, 'lab_output')
        os.makedirs(out_dir, exist_ok=True)
        self._log_path = os.path.join(out_dir, 'aruco_mapping_log.json')
        self._last_pose_log = 0.0
        self._last_flush = 0.0
        self._flush_interval = 2.0
        self._map_txt_path = os.path.join(out_dir, 'aruco_map.txt')
        self._export_interval = float(export_interval)
        self._last_export = 0.0

        # ArUco registry
        self._aruco_index: Dict[int, Dict[str, float]] = {}
        self._merge_distance = float(merge_distance)

        # Obstacles / Fruit Detection
        self.discovered_obstacles: List[dict] = []
        self.known_obstacles: List[List[float]] = []
        self.last_obstacle_add_time = 0.0
        self.add_cooldown = 0.5
        self.min_obs_separation = 0.15
        self.merge_threshold = 0.40
        self.obs_max_range = 0.60
        try:
            self.K = getattr(self.ekf.robot, 'camera_matrix', None)  # type: ignore[attr-defined]
        except Exception:
            self.K = None
        if self.K is not None:
            try:
                self.cx = float(self.K[0, 2]); self.fx = float(self.K[0, 0])
            except Exception:
                self.cx, self.fx = 160.0, 320.0
        else:
            self.cx, self.fx = 160.0, 320.0
        self.avoid_forward_dist = 0.35
        self.avoid_half_angle = math.radians(30.0)
        self.avoid_turn_dir = 1
        # Extend meta for new params (removed scan_rotations)
        self._log['meta'].update({'obs_max_range': self.obs_max_range, 'merge_threshold': self.merge_threshold, 'drive_duration': self.drive_duration, 'scan_toggle_period': self.scan_toggle_period})
        self._log['obstacles'] = []  # type: ignore[index]
        # Additional OutputWriter targeting Week07-08/lab_output for slam.txt saving
        try:
            self._mapper_output = operate_mod.dh.OutputWriter(os.path.join(REPO_ROOT, 'Week07-08', 'lab_output'))  # type: ignore[attr-defined]
        except Exception:
            self._mapper_output = None

        try:
            self.small_font = pygame.font.SysFont(None, 16)
        except Exception:
            self.small_font = None

    # ---------------- Pose & Logging ----------------
    def get_pose(self) -> Tuple[float, float, float]:
        try:
            if hasattr(self, 'ekf') and self.ekf is not None:
                robot = getattr(self.ekf, 'robot', None)
                if robot is not None and hasattr(robot, 'state'):
                    st = robot.state
                    if st is not None:
                        return float(st[0]), float(st[1]), normalize_angle(float(st[2]))
        except Exception:
            pass
        return 0.0, 0.0, 0.0

    def _log_pose(self, now: float | None = None):
        try:
            t = time.time() if now is None else now
            x, y, th = self.get_pose()
            self._log['poses'].append([t, x, y, th])
        except Exception:
            pass

    def _flush_log(self, force: bool = False):
        now = time.time()
        if not force and (now - self._last_flush) < self._flush_interval:
            return
        try:
            with open(self._log_path, 'w') as f:
                json.dump(self._log, f, indent=2)
            self._last_flush = now
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

    # ---------------- ArUco Processing ----------------
    def _update_arucos(self):
        taglist = getattr(self.ekf, 'taglist', []) or []
        changed = False
        for tag in taglist:
            tag_id = None; tx = ty = None
            try:
                if isinstance(tag, dict):
                    tag_id = int(tag.get('id')) if 'id' in tag else None
                    tx = float(tag.get('x')) if 'x' in tag else None
                    ty = float(tag.get('y')) if 'y' in tag else None
                elif isinstance(tag, (list, tuple)) and len(tag) >= 3:
                    tag_id = int(tag[0]); tx = float(tag[1]); ty = float(tag[2])
                else:
                    tag_id = int(getattr(tag, 'id'))
                    tx = float(getattr(tag, 'x'))
                    ty = float(getattr(tag, 'y'))
            except Exception:
                continue
            if tag_id is None or tx is None or ty is None:
                continue
            prior = self._aruco_index.get(tag_id)
            if prior is None:
                self._aruco_index[tag_id] = {'x': tx, 'y': ty}
                self._log['arucos'].append({'id': tag_id, 'x': tx, 'y': ty, 't': time.time()})
                self.notification = f'Mapped ArUco {tag_id}'
                self.known_obstacles.append([tx, ty])
                changed = True
            else:
                dx = tx - prior['x']; dy = ty - prior['y']
                if math.hypot(dx, dy) > self._merge_distance:
                    prior['x'] = tx; prior['y'] = ty
                    self._log['arucos'].append({'id': tag_id, 'x': tx, 'y': ty, 't': time.time(), 'rev': True})
                    changed = True
        if changed:
            self._flush_log(force=True)
            self._write_arucos_txt(force=True)
            if (self._first_marker_id is None) and (not self._marker_acquired) and self._aruco_index:
                mid = sorted(self._aruco_index.keys())[0]
                pos = self._aruco_index[mid]
                self._first_marker_id = mid
                self._first_marker_target = (pos['x'], pos['y'])
                self._transition('acquire_marker')
                self.notification = f'Approaching ArUco {mid}'

    def _write_arucos_txt(self, force: bool = False):
        now = time.time()
        if not force and (now - self._last_export) < self._export_interval:
            return
        try:
            with open(self._map_txt_path, 'w') as f:
                for mid, pos in sorted(self._aruco_index.items(), key=lambda kv: kv[0]):
                    f.write(f"{mid} {pos['x']:.4f} {pos['y']:.4f}\n")
            self._last_export = now
        except Exception:
            pass

    # ---------------- Motion Phases ----------------
    def _scan_step(self):
        # Time-based scan with extended one-direction rotation controlled by scan_toggle_period
        now = time.time()
        phase_elapsed = now - self._phase_start
        if phase_elapsed >= self.scan_duration:
            self._transition('drive')
            return
        # Determine direction: flip only when crossing multiples of scan_toggle_period
        if self.scan_toggle_period > 0:
            block_index = int(phase_elapsed // self.scan_toggle_period)
            self._turn_dir = 1 if (block_index % 2 == 0) else -1
        period = self.turn_spin_time + self.turn_stop_time
        t_phase = (now - self._phase_start) % period
        if t_phase < self.turn_spin_time:
            self.command['motion'] = [0, self._turn_dir * self.turn_speed_cmd]
        else:
            self.command['motion'] = [0, 0]
        self.notification = f'Scanning for ArUcos ({phase_elapsed:4.1f}s / {self.scan_duration}s)'

    def _drive_step(self):
        now = time.time(); phase_elapsed = now - self._phase_start
        if phase_elapsed >= self.drive_duration:
            self._transition('scan'); return
        rx, ry, rth = self.get_pose()
        turn_away = self._forward_blocked(rx, ry, rth)
        if turn_away is not None:
            period = self.turn_spin_time + self.turn_stop_time
            t_phase = (now - self._phase_start) % period
            if t_phase < self.turn_spin_time:
                self.command['motion'] = [0, turn_away * self.turn_speed_cmd]
            else:
                self.command['motion'] = [0, 0]
            self.notification = 'Avoiding obstacle ahead'
            return
        d_phase = (now - self._phase_start) % self.drive_period
        if d_phase < (self.drive_period - self.drive_stop_time):
            self.command['motion'] = [self.drive_linear_speed, 0]
        else:
            self.command['motion'] = [0, 0]
        self.notification = f'Forward explore ({phase_elapsed:4.1f}s / {self.drive_duration}s)'

    def _transition(self, new_phase: str):
        self._phase = new_phase; self._phase_start = time.time()
        if new_phase == 'scan':
            self.notification = 'Switching to SCAN'
        elif new_phase == 'drive': self.notification = 'Switching to DRIVE'
        elif new_phase == 'acquire_marker': self.notification = 'Acquire first marker'
        elif new_phase == 'spin_marker':
            _, _, th = self.get_pose()
            self._spin_start_heading = th
            self._spin_last_heading = th
            self._spin_accum_angle = 0.0
            self.notification = '360 scan at marker'

    def auto_step(self):
        if not self.ekf_on:
            self.command['motion'] = [0, 0]; self.notification = 'Press ENTER to start SLAM'; return
        now = time.time()
        if (now - self._last_pose_log) >= 0.2:
            self._log_pose(now); self._last_pose_log = now; self._flush_log(False)
        self._update_arucos()
        if self._phase == 'scan': self._scan_step()
        elif self._phase == 'drive': self._drive_step()
        elif self._phase == 'acquire_marker': self._acquire_marker_step()
        elif self._phase == 'spin_marker': self._spin_marker_step()
        else: self._transition('scan')

    # -------- First Marker Acquisition Phase --------
    def _acquire_marker_step(self):
        if self._first_marker_target is None or self._first_marker_id is None:
            self._transition('scan'); return
        if self._first_marker_id in self._aruco_index:
            pos = self._aruco_index[self._first_marker_id]; self._first_marker_target = (pos['x'], pos['y'])
        rx, ry, rth = self.get_pose(); tx, ty = self._first_marker_target
        dx, dy = tx - rx, ty - ry; dist = math.hypot(dx, dy)
        if dist <= self.acquire_dist_tol:
            self.notification = f'Reached ArUco {self._first_marker_id} (~{dist:.2f}m)'
            self._marker_acquired = True; self._transition('spin_marker'); return
        bearing = math.atan2(dy, dx); dheading = angle_diff(bearing, rth)
        now = time.time()
        if abs(dheading) > self.heading_tol:
            period = self.turn_spin_time + self.turn_stop_time
            t_phase = (now - self._phase_start) % period
            turn_dir = 1 if dheading > 0 else -1
            if t_phase < self.turn_spin_time: self.command['motion'] = [0, turn_dir * self.turn_speed_cmd]
            else: self.command['motion'] = [0, 0]
            self.notification = f'Aligning to ArUco {self._first_marker_id} (dist {dist:.2f}m)'
        else:
            d_phase = (now - self._phase_start) % self.drive_period
            if d_phase < (self.drive_period - self.drive_stop_time): self.command['motion'] = [self.drive_linear_speed, 0]
            else: self.command['motion'] = [0, 0]
            self.notification = f'Approaching ArUco {self._first_marker_id} (dist {dist:.2f}m)'

    # -------- 360 Spin Phase --------
    def _spin_marker_step(self):
        if self._spin_start_heading is None: self._transition('scan'); return
        _, _, th = self.get_pose()
        if self._spin_last_heading is not None:
            dth = angle_diff(th, self._spin_last_heading); self._spin_accum_angle += abs(dth)
        self._spin_last_heading = th
        now = time.time(); period = self.turn_spin_time + self.turn_stop_time
        t_phase = (now - self._phase_start) % period
        if t_phase < self.turn_spin_time: self.command['motion'] = [0, self._spin_dir * self.turn_speed_cmd]
        else: self.command['motion'] = [0, 0]
        if self._spin_accum_angle >= 2 * math.pi:
            self.notification = '360 scan complete'; self._transition('scan')
        else:
            remaining = max(0.0, 2 * math.pi - self._spin_accum_angle)
            self.notification = f'360 scan ({self._spin_accum_angle:.2f} rad, rem {remaining:.2f})'

    # ---------------- Forward obstacle blocking ----------------
    def _forward_blocked(self, rx: float, ry: float, rth: float) -> int | None:
        all_obs: List[Tuple[float, float]] = []
        all_obs.extend(self.known_obstacles)
        all_obs.extend([[float(d['x']), float(d['y'])] for d in self.discovered_obstacles])
        for ox, oy in all_obs:
            dx, dy = ox - rx, oy - ry; dist = math.hypot(dx, dy)
            if dist <= 1e-6 or dist > self.avoid_forward_dist: continue
            bearing = math.atan2(dy, dx); ang = angle_diff(bearing, rth)
            if abs(ang) <= self.avoid_half_angle: return -1 if ang > 0 else 1
        return None

    # ---------------- Fruit obstacle update ----------------
    def periodic_perception_update(self):
        bboxes = getattr(self, 'detector_output', None)
        if not isinstance(bboxes, (list, tuple)) or len(bboxes) == 0: return
        now = time.time(); x, y, th = self.get_pose(); cx, fx = self.cx, self.fx
        for det in bboxes:
            try:
                label: str = str(det[0]).lower(); xywh = np.asarray(det[1]).astype(float); conf = float(det[2])
            except Exception:
                continue
            if conf < 0.8 or label.startswith('aruco'): continue
            ox = oy = None; used_tpe = False
            if estimate_pose is not None and self.K is not None:
                try:
                    obj_info = [label, [float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])]]
                    pose_dict = estimate_pose(self.K, obj_info, [x, y, th])  # type: ignore[arg-type]
                    if pose_dict and 'x' in pose_dict and 'y' in pose_dict:
                        ox = float(pose_dict['x']); oy = float(pose_dict['y']); used_tpe = True
                except Exception:
                    pass
            if ox is None or oy is None:
                try:
                    u = float(xywh[0]); w_px = float(xywh[2])
                    alpha = math.atan((u - cx) / max(1e-6, fx))
                    bearing = th + alpha
                    W_assumed = 0.10
                    d = 0.5 if w_px <= 1.0 else max(0.30, min(1.0, (fx * W_assumed) / w_px))
                    ox = x + d * math.cos(bearing); oy = y + d * math.sin(bearing)
                except Exception:
                    continue
            if math.hypot(ox - x, oy - y) > self.obs_max_range: continue
            merged = False
            for dct in self.discovered_obstacles:
                if dct.get('label') == label:
                    px, py = float(dct['x']), float(dct['y'])
                    if math.hypot(ox - px, oy - py) <= self.merge_threshold:
                        cnt = int(dct.get('count', 1))
                        nx = (px * cnt + ox) / (cnt + 1); ny = (py * cnt + oy) / (cnt + 1)
                        dct['x'] = nx; dct['y'] = ny; dct['count'] = cnt + 1
                        self._log_obstacle(nx, ny, label, 'merge-tpe' if used_tpe else 'merge-heuristic')
                        merged = True; break
            if merged: continue
            all_obs = []; all_obs.extend(self.known_obstacles); all_obs.extend([[d['x'], d['y']] for d in self.discovered_obstacles])
            if any(math.hypot(ox - qx, oy - qy) <= self.min_obs_separation for qx, qy in all_obs): continue
            if (now - self.last_obstacle_add_time) < self.add_cooldown: continue
            self.discovered_obstacles.append({'x': float(ox), 'y': float(oy), 'label': label, 'count': 1})
            self._log_obstacle(ox, oy, label, 'tpe' if used_tpe else 'heuristic')
            self.last_obstacle_add_time = now; self._flush_log(False)

    # ---------------- Map saving override ----------------
    def record_data(self):
        """Save SLAM map to Week07-08/lab_output/slam.txt when 's' pressed.
        Uses a dedicated OutputWriter (self._mapper_output) so the file appears
        where users expect for this script, independent of working directory.
        """
        if self.command.get('output'):
            if getattr(self, '_mapper_output', None) is not None:
                try:
                    self._mapper_output.write_map(self.ekf)
                    self.notification = 'Map is saved (Week07-08/lab_output/slam.txt)'
                except Exception:
                    self.notification = 'Failed to save map'
            # prevent base class duplicate handling
            self.command['output'] = False
        # still allow any additional base-side behaviors if needed (but flag cleared)
        try:
            super().record_data()
        except Exception:
            pass

    # ---------------- Keyboard (restrict manual) ----------------
    def update_keyboard(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
                    continue
                if event.key == pygame.K_SPACE:
                    self.command['motion'] = [0, 0]; self.notification = 'Emergency stop (space)'
                elif event.key == pygame.K_i:
                    self.command['save_image'] = True
                elif event.key == pygame.K_s:
                    self.command['output'] = True
                elif event.key == pygame.K_r:
                    if getattr(self, 'double_reset_comfirm', None) is None:
                        self.double_reset_comfirm = 0
                    if self.double_reset_comfirm == 0:
                        self.notification = 'Press again to confirm CLEAR MAP'; self.double_reset_comfirm += 1
                    elif self.double_reset_comfirm == 1:
                        self.notification = 'SLAM Map is cleared'; self.double_reset_comfirm = 0
                        try: self.ekf.reset()
                        except Exception: pass
                elif event.key == pygame.K_RETURN:
                    n_observed_markers = len(self.ekf.taglist)
                    if n_observed_markers == 0:
                        if not self.ekf_on:
                            self.notification = 'SLAM is running'; self.ekf_on = True
                        else:
                            self.notification = '> 2 landmarks is required for pausing'
                    elif n_observed_markers < 3:
                        self.notification = '> 2 landmarks is required for pausing'
                    else:
                        if not self.ekf_on: self.request_recover_robot = True
                        self.ekf_on = not self.ekf_on
                        self.notification = 'SLAM is running' if self.ekf_on else 'SLAM is paused'
                elif event.key == pygame.K_ESCAPE:
                    self.quit = True
            elif event.type == pygame.QUIT: self.quit = True
        if self.quit:
            pygame.quit(); sys.exit()

    # ---------------- UI Overlay ----------------
    def draw(self, canvas):
        super().draw(canvas)
        try:
            if self.small_font is not None:
                text = f'ArUcos: {len(self._aruco_index)} Obst: {len(self.discovered_obstacles)} Phase: {self._phase}'
                surf = self.small_font.render(text, True, (255, 255, 0))
                canvas.blit(surf, (10, 630 - 20))
        except Exception:
            pass
        return canvas


def main():
    parser = argparse.ArgumentParser("Autonomous ArUco mapping (pulsed scan/drive + obstacles)")
    parser.add_argument('--ip', type=str, default='192.168.50.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--calib_dir', type=str, default=os.path.join(REPO_ROOT, 'Week02-04', 'calibration', 'param') + os.sep)
    parser.add_argument('--yolo_model', default=os.path.join(SCRIPT_DIR, 'YOLO', 'model', 'bestv5.pt'))
    parser.add_argument('--merge_distance', type=float, default=0.10)
    parser.add_argument('--export_interval', type=float, default=2.0)
    parser.add_argument('--drive_duration', type=float, default=4.0, help='Seconds of forward exploration per drive phase (default 4.0)')
    parser.add_argument('--scan_toggle_period', type=float, default=None, help='Seconds before reversing scan turn direction (default = scan_duration: one direction per scan)')
    parser.add_argument('--play_data', action='store_true')
    parser.add_argument('--save_data', action='store_true')
    args, _ = parser.parse_known_args()

    op_args = SimpleNamespace(ip=args.ip, port=args.port, calib_dir=args.calib_dir,
                              yolo_model=args.yolo_model, play_data=args.play_data, save_data=args.save_data)
    operate_mod.args = op_args  # type: ignore

    pygame.font.init()
    try:
        TITLE_FONT = pygame.font.Font(os.path.join(WEEK0506_DIR, 'pics', '8-BitMadness.ttf'), 35)
        TEXT_FONT = pygame.font.Font(os.path.join(WEEK0506_DIR, 'pics', '8-BitMadness.ttf'), 40)
        operate_mod.TITLE_FONT = TITLE_FONT  # type: ignore
        operate_mod.TEXT_FONT = TEXT_FONT    # type: ignore
    except Exception:
        pass

    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 - Auto Mapping (L3)')
    try:
        pygame.display.set_icon(pygame.image.load(os.path.join(WEEK0506_DIR, 'pics', '8bit', 'pibot5.png')))
    except Exception:
        pass
    canvas.fill((0, 0, 0))

    try: os.chdir(WEEK0506_DIR)
    except Exception: pass

    mapper = AutoArUcoMapper(op_args, merge_distance=args.merge_distance, export_interval=args.export_interval, drive_duration=args.drive_duration, scan_toggle_period=args.scan_toggle_period)

    running = True; clock = pygame.time.Clock()
    while running:
        mapper.update_keyboard()
        mapper.take_pic()
        mapper.auto_step()
        drive_meas = mapper.control()
        mapper.update_slam(drive_meas)
        mapper.record_data()
        mapper.save_image()
        mapper.detect_target()
        mapper.periodic_perception_update()
        mapper.draw(canvas)
        pygame.display.update()
        clock.tick(30)
        if mapper.quit: running = False

    mapper._flush_log(force=True)


if __name__ == '__main__':
    main()
