# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import importlib

# import SLAM components (uncomment and wire with your M1 code)
BASE_DIR = os.path.dirname(__file__)
try:
    sys.path.insert(0, os.path.join(BASE_DIR, "slam"))
    from slam.ekf import EKF  #
    from slam.robot import Robot  #
    import slam.aruco_detector as aruco  
except Exception:
    EKF = None
    Robot = None
    aruco = None

# import utility functions
sys.path.insert(0, os.path.join(BASE_DIR, "util"))
from util.pibot import PenguinPi
import util.measure as measure

# import Week05-06 YOLO detector and pose estimator if available
W56_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Week05-06"))
try:
    import importlib
    sys.path.insert(0, W56_DIR)
    YOLODetector = importlib.import_module('YOLO.detector').Detector
    yolo_estimate_pose = importlib.import_module('TargetPoseEst').estimate_pose
except Exception:
    YOLODetector = None
    yolo_estimate_pose = None

# Lazy globals for detector and intrinsics
YOLO_MODEL = None  # type: ignore
CAMERA_MATRIX = None  # type: ignore

# SLAM/GUI globals
ekf = None
aruco_det_inst = None
LAST_CAMERA = None
LAST_YOLO_VIS = None
LAST_ARUCO = None

# --- A* Path Planning Implementation ---
import heapq

def astar(grid, start, goal):
    """Basic A* algorithm for grid navigation."""
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    neighbors = [(0,1),(1,0),(-1,0),(0,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data.append(start)
            return data[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + 1
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]):
                if grid[neighbor[0]][neighbor[1]] == 1:
                    continue
            else:
                continue
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
            if tentative_g_score < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return []

def pos_to_grid(x, y, grid_size, arena_origin):
    """Convert real-world coordinates to grid indices."""
    gx = int((x - arena_origin[0]) / grid_size)
    gy = int((y - arena_origin[1]) / grid_size)
    return gx, gy

def grid_to_pos(gx, gy, grid_size, arena_origin):
    """Convert grid indices to real-world coordinates."""
    x = gx * grid_size + arena_origin[0]
    y = gy * grid_size + arena_origin[1]
    return x, y

def build_grid(fruits_list, fruits_true_pos, aruco_true_pos, search_list, grid_size, arena_origin, grid_width, grid_height):
    """Build occupancy grid: 1 for obstacles, 0 for free."""
    grid = np.zeros((grid_width, grid_height), dtype=int)
    # Mark ArUcos as obstacles
    for pos in aruco_true_pos:
        gx, gy = pos_to_grid(pos[0], pos[1], grid_size, arena_origin)
        if 0 <= gx < grid_width and 0 <= gy < grid_height:
            grid[gx][gy] = 1
    # Mark non-target fruits as obstacles
    for i, fruit in enumerate(fruits_list):
        if fruit not in search_list:
            gx, gy = pos_to_grid(fruits_true_pos[i][0], fruits_true_pos[i][1], grid_size, arena_origin)
            if 0 <= gx < grid_width and 0 <= gy < grid_height:
                grid[gx][gy] = 1
    return grid

def inflate_grid(grid, radius_cells):
    if radius_cells <= 0:
        return grid
    gx, gy = grid.shape
    inflated = grid.copy()
    obstacle_cells = [(i, j) for i in range(gx) for j in range(gy) if grid[i, j] == 1]
    for i, j in obstacle_cells:
        for di in range(-radius_cells, radius_cells + 1):
            for dj in range(-radius_cells, radius_cells + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < gx and 0 <= nj < gy:
                    if di * di + dj * dj <= radius_cells * radius_cells:
                        inflated[ni, nj] = 1
    return inflated

def choose_next_target(strategy, remaining_targets, fruits_list, fruits_true_pos, robot_pose):
    """Choose next target index from remaining based on strategy: 'order' or 'nearest'."""
    if strategy == 'order':
        return remaining_targets[0]
    # nearest by Euclidean distance
    rx, ry = robot_pose[0], robot_pose[1]
    best_idx = remaining_targets[0]
    best_d = float('inf')
    for idx in remaining_targets:
        tx, ty = fruits_true_pos[idx]
        d = (tx - rx)**2 + (ty - ry)**2
        if d < best_d:
            best_d = d
            best_idx = idx
    return best_idx

def update_dynamic_obstacles_from_detector(grid, grid_size, arena_origin, grid_width, grid_height):
    """Deprecated signature: use update_dynamic_obstacles_from_detector(grid,..., robot_pose) returning (grid, new_targets)."""
    return grid  # keep for backward compatibility

def _ensure_yolo_loaded():
    """Initialise YOLO detector and camera intrinsics once."""
    global YOLO_MODEL, CAMERA_MATRIX
    if YOLO_MODEL is None and YOLODetector is not None:
        # try model under Week05-06/YOLO/model/bestv5.pt
        model_path = os.path.join(W56_DIR, 'YOLO', 'model', 'bestv5.pt')
        if not os.path.isfile(model_path):
            # also try local Week07-08/YOLO/model if user copied here
            alt_path = os.path.join(BASE_DIR, 'YOLO', 'model', 'bestv5.pt')
            model_path = alt_path if os.path.isfile(alt_path) else model_path
        try:
            YOLO_MODEL = YOLODetector(model_path)
        except Exception as e:
            print(f"[YOLO] Failed to load model at {model_path}: {e}")
            YOLO_MODEL = None
    if CAMERA_MATRIX is None:
        # Look for intrinsics in Week05-06 first, then Week07-08
        cand = [
            os.path.join(W56_DIR, 'calibration', 'param', 'intrinsic.txt'),
            os.path.join(BASE_DIR, 'calibration', 'param', 'intrinsic.txt'),
        ]
        for p in cand:
            if os.path.isfile(p):
                try:
                    CAMERA_MATRIX = np.loadtxt(p, delimiter=',')
                    break
                except Exception as e:
                    print(f"[YOLO] Failed to load intrinsics from {p}: {e}")
        if CAMERA_MATRIX is None:
            print("[YOLO] Camera intrinsics not found; target pose estimation will be disabled.")

def update_dynamic_obstacles_from_detector_live(grid, grid_size, arena_origin, grid_width, grid_height, robot_pose):
    """Run YOLO on the latest frame, add obstacles to the occupancy grid and return any newly detected target positions.

    Returns: (updated_grid, new_targets_dict)
    new_targets_dict maps label -> (x, y) for labels in the M3 target set.
    """
    _ensure_yolo_loaded()
    new_targets = {}
    # If detector not available, just return
    if YOLO_MODEL is None or yolo_estimate_pose is None:
        return grid, new_targets

    # Acquire image from robot camera
    try:
        img = ppi.get_image()
    except Exception as e:
        print(f"[YOLO] Failed to get image: {e}")
        return grid, new_targets
    if img is None or img.size == 0:
        return grid, new_targets

    # Convert RGB->BGR for OpenCV/Ultralytics
    if img.ndim == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img
    # Save camera frame for GUI
    global LAST_CAMERA
    LAST_CAMERA = img_bgr.copy()

    try:
        bboxes, yolo_img = YOLO_MODEL.detect_single_image(img_bgr)
    except Exception as e:
        print(f"[YOLO] Detection failed: {e}")
        return grid, new_targets
    # Store YOLO overlay for GUI (ensure BGR)
    try:
        global LAST_YOLO_VIS
        LAST_YOLO_VIS = yolo_img.copy()
    except Exception:
        pass

    # ArUco overlay (optional)
    try:
        global LAST_ARUCO
        if aruco is not None:
            # If we have EKF robot with intrinsics, use that
            if ekf is not None:
                ad = aruco.aruco_detector(ekf.robot)
            else:
                # Build from intrinsics if available
                K = None; D = None
                for base in [W56_DIR, BASE_DIR]:
                    kpath = os.path.join(base, 'calibration', 'param', 'intrinsic.txt')
                    dpath = os.path.join(base, 'calibration', 'param', 'distCoeffs.txt')
                    if os.path.isfile(kpath) and K is None:
                        try: K = np.loadtxt(kpath, delimiter=',')
                        except Exception: K = None
                    if os.path.isfile(dpath) and D is None:
                        try: D = np.loadtxt(dpath, delimiter=',')
                        except Exception: D = None
                if K is not None and D is not None and Robot is not None:
                    dummy = Robot(0.15, 0.01, K, D)
                    ad = aruco.aruco_detector(dummy)
                else:
                    ad = None
            if ad is not None:
                _, marked = ad.detect_marker_positions(img_bgr)
                LAST_ARUCO = marked.copy()
    except Exception:
        pass

    target_labels = {"garlic", "lemon", "pear", "tomato", "pumpkin"}
    max_valid_dist = 2.0  # metres (relaxed)

    for det in bboxes:
        label = str(det[0]).lower()
        box_xywh = det[1]
        # Estimate world pose if intrinsics available
        est = None
        if CAMERA_MATRIX is not None:
            try:
                est = yolo_estimate_pose(CAMERA_MATRIX, [label, box_xywh], robot_pose)
            except Exception as e:
                # Fallback: skip if estimation fails
                est = None
        if est is None:
            continue
        wx, wy = float(est['x']), float(est['y'])
        # Validate distance from robot
        try:
            dist = float(np.hypot(wx - float(robot_pose[0]), wy - float(robot_pose[1])))
        except Exception:
            dist = 0.0
        if dist > max_valid_dist:
            continue

        if label in target_labels:
            # record or refine position
            if label not in new_targets:
                new_targets[label] = (wx, wy)
            else:
                # simple averaging if multiple boxes
                px, py = new_targets[label]
                new_targets[label] = ((px + wx) / 2.0, (py + wy) / 2.0)
        else:
            # Treat non-target fruits as obstacles
            gx, gy = pos_to_grid(wx, wy, grid_size, arena_origin)
            if 0 <= gx < grid_width and 0 <= gy < grid_height:
                grid[gx][gy] = 1

    return grid, new_targets

def detect_targets_and_obstacles_sim(gt_map, robot_pose, detection_radius=0.6):
    """Simulation-only: reveal targets within a radius from robot_pose using ground-truth map dict.
    Returns: list of (label, x, y, is_target) and list of (x,y) obstacles (non-target fruits).
    """
    if gt_map is None:
        return [], []
    rx, ry = robot_pose[0], robot_pose[1]
    detections = []
    obstacles = []
    targets = {"garlic", "lemon", "pear", "tomato", "pumpkin"}
    for key, val in gt_map.items():
        if key.startswith("aruco"):
            continue
        x, y = float(val["x"]), float(val["y"])
        if np.hypot(x - rx, y - ry) <= detection_radius:
            label = key.rsplit("_", 1)[0]
            if label in targets:
                detections.append((label, x, y, True))
            else:
                obstacles.append((x, y))
    return detections, obstacles

def generate_lawnmower_waypoints(bounds, spacing=0.4):
    """Generate world-coordinate coverage waypoints in a lawnmower pattern.
    bounds = (xmin, xmax, ymin, ymax)
    """
    xmin, xmax, ymin, ymax = bounds
    ys = np.arange(ymin, ymax + 1e-6, spacing)
    xs = np.arange(xmin, xmax + 1e-6, spacing)
    waypoints = []
    reverse = False
    for y in ys:
        if not reverse:
            for x in xs:
                waypoints.append((float(x), float(y)))
        else:
            for x in xs[::-1]:
                waypoints.append((float(x), float(y)))
        reverse = not reverse
    return waypoints

def plan_and_follow_path(grid, start_pose, goal_xy, grid_size, arena_origin, simulate):
    rgx, rgy = pos_to_grid(start_pose[0], start_pose[1], grid_size, arena_origin)
    tgx, tgy = pos_to_grid(goal_xy[0], goal_xy[1], grid_size, arena_origin)
    path = astar(grid, (rgx, rgy), (tgx, tgy))
    if not path:
        print("No path found.")
        return start_pose

    # Compress path: keep only turning points and last cell to reduce stop-turn-go
    def _compress(pl):
        if len(pl) <= 2:
            return pl
        out = [pl[0]]
        prev_dx = pl[1][0] - pl[0][0]
        prev_dy = pl[1][1] - pl[0][1]
        for i in range(2, len(pl)):
            dx = pl[i][0] - pl[i-1][0]
            dy = pl[i][1] - pl[i-1][1]
            if dx != prev_dx or dy != prev_dy:
                out.append(pl[i-1])
            prev_dx, prev_dy = dx, dy
        out.append(pl[-1])
        # Optionally subsample further: every k-th waypoint
        k = max(1, int(0.2 / max(1e-6, grid_size)))  # ~20cm steps
        if k > 1 and len(out) > 2:
            tmp = [out[0]]
            for i in range(1, len(out)-1):
                if i % k == 0:
                    tmp.append(out[i])
            tmp.append(out[-1])
            out = tmp
        return out

    path = _compress(path)

    for gx, gy in path:
        x, y = grid_to_pos(gx, gy, grid_size, arena_origin)
        drive_to_point([x, y], start_pose, simulate=simulate)
        if simulate:
            # advance synthetic pose towards waypoint
            theta = np.arctan2(y - start_pose[1], x - start_pose[0])
            start_pose = [x, y, theta]
        else:
            # Try to get SLAM pose; if unavailable or unchanged, assume we reached the waypoint
            pose = get_robot_pose()
            try:
                if pose is None:
                    raise ValueError
                px, py = float(pose[0]), float(pose[1])
                # if pose didn't change meaningfully, fall back
                if np.hypot(px - start_pose[0], py - start_pose[1]) < 1e-3 and all(abs(float(p)) < 1e-6 for p in pose):
                    raise ValueError
                start_pose = [px, py, float(pose[2])]
            except Exception:
                theta = np.arctan2(y - start_pose[1], x - start_pose[0])
                start_pose = [x, y, theta]
    return start_pose

def draw_viz(grid, grid_size, arena_origin, robot_pose, goal_xy=None, path=None, known_targets=None, win_name='M3-Nav'):
    """Render a simple top-down grid view with OpenCV."""
    h, w = grid.shape[1], grid.shape[0]
    scale = 12  # pixels per cell
    img = np.zeros((h*scale, w*scale, 3), dtype=np.uint8)
    # draw free/obstacles
    for gx in range(w):
        for gy in range(h):
            color = (40, 40, 40) if grid[gx, gy] == 1 else (200, 200, 200)
            cv2.rectangle(img, (gx*scale, gy*scale), (gx*scale+scale-1, gy*scale+scale-1), color, -1)
    # path
    if path:
        for gx, gy in path:
            cv2.rectangle(img, (gx*scale, gy*scale), (gx*scale+scale-1, gy*scale+scale-1), (0, 255, 255), -1)
    # known targets
    if known_targets:
        for lbl, (tx, ty) in known_targets.items():
            tgx, tgy = pos_to_grid(tx, ty, grid_size, arena_origin)
            if 0 <= tgx < w and 0 <= tgy < h:
                cv2.circle(img, (tgx*scale+scale//2, tgy*scale+scale//2), scale//2, (0, 128, 255), 2)
                cv2.putText(img, lbl, (tgx*scale+2, tgy*scale+scale-2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (10,10,10), 1)
    # goal
    if goal_xy is not None:
        gx, gy = pos_to_grid(goal_xy[0], goal_xy[1], grid_size, arena_origin)
        if 0 <= gx < w and 0 <= gy < h:
            cv2.circle(img, (gx*scale+scale//2, gy*scale+scale//2), scale//2, (0, 0, 255), 2)
    # robot
    rgx, rgy = pos_to_grid(robot_pose[0], robot_pose[1], grid_size, arena_origin)
    if 0 <= rgx < w and 0 <= rgy < h:
        cv2.circle(img, (rgx*scale+scale//2, rgy*scale+scale//2), scale//2, (0, 255, 0), -1)
    cv2.imshow(win_name, img)
    cv2.waitKey(1)

def draw_camera_gui(ppi, known_targets=None, show_aruco=True, show_yolo=True, win_name='Camera'):
    """Show live camera with YOLO boxes and ArUco marker overlays (if configured)."""
    try:
        frame = ppi.get_image()
    except Exception as e:
        print(f"[CAM] Failed to get image: {e}")
        return
    if frame is None or frame.size == 0:
        return
    # Convert to BGR for drawing
    if frame.ndim == 3:
        img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = frame

    overlay = img_bgr.copy()

    # YOLO boxes
    if show_yolo and YOLO_MODEL is not None:
        try:
            bboxes, _ = YOLO_MODEL.detect_single_image(img_bgr)
            for bbox in bboxes:
                label = str(bbox[0]).lower()
                xywh = np.asarray(bbox[1]).ravel()
                x, y, w, h = map(float, xywh)
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)
                cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,255), 2)
                cv2.putText(overlay, label, (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        except Exception as e:
            pass

    # ArUco markers
    if show_aruco and aruco is not None and Robot is not None:
        try:
            # Build a minimal Robot to pass intrinsics if available
            # Intrinsics
            K = None
            D = None
            for base in [W56_DIR, BASE_DIR]:
                kpath = os.path.join(base, 'calibration', 'param', 'intrinsic.txt')
                dpath = os.path.join(base, 'calibration', 'param', 'distCoeffs.txt')
                if os.path.isfile(kpath):
                    try:
                        K = np.loadtxt(kpath, delimiter=',')
                    except Exception:
                        K = None
                if os.path.isfile(dpath):
                    try:
                        D = np.loadtxt(dpath, delimiter=',')
                    except Exception:
                        D = None
            if K is not None and D is not None:
                # Minimal robot just to carry K,D
                dummy = Robot(wheels_width=0.15, wheels_scale=0.01, camera_matrix=K, camera_dist=D)
                ad = aruco.aruco_detector(dummy)
                # Use grayscale for detection
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                _, marked = ad.detect_marker_positions(gray)
                # The function returns grayscale overlay; re-draw markers on color frame by re-detecting
                corners, ids, _ = cv2.aruco.detectMarkers(gray, ad.aruco_dict, parameters=ad.aruco_params)
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
        except Exception as e:
            pass

    # Known targets list overlay
    if known_targets:
        y0 = 20
        for lbl in sorted(known_targets.keys()):
            cv2.putText(overlay, f"known: {lbl}", (5, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1)
            y0 += 18

    cv2.imshow(win_name, overlay)
    cv2.waitKey(1)

# --- Pygame GUI similar to operate.py ---
class SimpleGUI:
    def __init__(self, title='ECE4078 M3 Autonomy'):
        import pygame
        pygame.font.init()
        self.pygame = pygame
        self.TITLE_FONT = pygame.font.Font(os.path.join(BASE_DIR, 'pics/8-BitMadness.ttf'), 28) if os.path.isfile(os.path.join(BASE_DIR, 'pics/8-BitMadness.ttf')) else pygame.font.SysFont('Arial', 24)
        self.TEXT_FONT = pygame.font.SysFont('Consolas', 18)
        self.width, self.height = 700, 660
        self.canvas = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(title)

    def draw_pygame_window(self, canvas, cv2_img, position, size=None):
        if cv2_img is None:
            return
        img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        if size is not None:
            img = cv2.resize(img, size)
        img = np.rot90(img)
        surf = self.pygame.surfarray.make_surface(img)
        surf = self.pygame.transform.flip(surf, True, False)
        canvas.blit(surf, position)

    def draw(self, ekf=None):
        # Handle quit
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.pygame.quit()
                raise SystemExit

        self.canvas.fill((0,0,0))
        h_pad, v_pad = 20, 40
        # Camera with ArUco overlay if available
        cam = LAST_ARUCO if LAST_ARUCO is not None else LAST_CAMERA
        self.draw_pygame_window(self.canvas, cam, (h_pad, v_pad), size=(320,240))
        # Detector window
        self.draw_pygame_window(self.canvas, LAST_YOLO_VIS if LAST_YOLO_VIS is not None else cam, (h_pad, 240 + 2*v_pad), size=(320,240))
        # EKF view if available
        if ekf is not None:
            try:
                ekf_view = ekf.draw_slam_state(res=(320, 480 + v_pad), not_pause=True)
                self.canvas.blit(ekf_view, (2*h_pad + 320, v_pad))
            except Exception:
                pass
        # Titles
        self.canvas.blit(self.TITLE_FONT.render('PiBot Cam', False, (200,200,200)), (h_pad, v_pad-25))
        self.canvas.blit(self.TITLE_FONT.render('Detector', False, (200,200,200)), (h_pad, 240+2*v_pad-25))
        self.canvas.blit(self.TITLE_FONT.render('SLAM', False, (200,200,200)), (2*h_pad + 320, v_pad-25))
        self.pygame.display.update()

def spin_360(robot_pose, simulate=False):
    """Turn the robot on the spot by ~360 degrees once at the start."""
    wheel_vel = 30  # turning tick/s
    if simulate:
        print("[SIM] 360-degree spin scan")
        # simulate a short delay and update heading
        time.sleep(1.0)
        robot_pose = [robot_pose[0], robot_pose[1], (robot_pose[2] + 2*np.pi) % (2*np.pi)]
        return robot_pose
    # Read baseline from common locations (meters)
    def _read_scalar_any(paths, default_val):
        for path in paths:
            try:
                arr = np.loadtxt(path, delimiter=',')
                if np.ndim(arr) == 0:
                    return float(arr)
                flat = np.ravel(arr)
                if flat.size > 0:
                    return float(flat[0])
            except Exception:
                continue
        print(f"[CALIB] Using default {default_val} (files not found: {paths})")
        return float(default_val)
    baseline_val = _read_scalar_any([
        os.path.join(BASE_DIR, 'calibration', 'param', 'baseline.txt'),
        os.path.join(os.path.dirname(BASE_DIR), 'Week05-06', 'calibration', 'param', 'baseline.txt'),
    ], default_val=0.15)
    # Map radians to turn time using same placeholder scaling as drive_to_point
    turn_ticks_per_rad = max(1e-3, float(baseline_val))
    turn_time = (2*np.pi) * turn_ticks_per_rad / wheel_vel
    print(f"Spin in place 360 deg for {turn_time:.2f}s")
    try:
        ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
        ppi.set_velocity([0, 0])
    except Exception as e:
        print(f"[MOTION] Spin command failed: {e}")
    # Update heading assumption
    robot_pose = [robot_pose[0], robot_pose[1], (robot_pose[2] + 2*np.pi) % (2*np.pi)]
    return robot_pose

def spin_scan(grid, robot_pose, grid_size, arena_origin, grid_width, grid_height, steps=12, seg_time=0.35, turning_tick=25, simulate=False):
    """Rotate on the spot in small segments and run detector each segment to discover nearby targets.
    Returns: (grid, found_targets_dict, robot_pose)
    """
    found_all = {}
    if simulate:
        # Just perform a simulated spin without detection
        return grid, found_all, spin_360(robot_pose, simulate=True)
    for i in range(steps):
        try:
            ppi.set_velocity([0, 1], turning_tick=turning_tick, time=seg_time)
            ppi.set_velocity([0, 0])
        except Exception as e:
            print(f"[MOTION] Spin segment failed: {e}")
        # Update heading assumption evenly
        robot_pose = [robot_pose[0], robot_pose[1], (robot_pose[2] + 2*np.pi/steps) % (2*np.pi)]
        # Run detector and collect targets
        try:
            grid, found = update_dynamic_obstacles_from_detector_live(
                grid, grid_size, arena_origin, grid_width, grid_height, robot_pose
            )
            for lbl, (x, y) in found.items():
                found_all[lbl] = (x, y)
        except Exception as e:
            print(f"[SCAN] Detection during spin failed: {e}")
        # brief pause for UI
        time.sleep(0.05)
    return grid, found_all, robot_pose


def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 5 target fruits&vegs to search for

    @param fname: filename of the map
    @return:
        1) list of targets, e.g. ['lemon', 'tomato', 'garlic']
        2) locations of the targets, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5]) - 1
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(len(fruit_list)): # there are 5 targets amongst 10 objects
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# note that this function requires your camera and wheel calibration parameters from M2, and the "util" folder from M1
# fully automatic navigation:
# try developing a path-finding algorithm that produces the waypoints automatically
def drive_to_point(waypoint, robot_pose, simulate=False):
    # In simulation, don't touch hardware or load calibration files
    if simulate:
        print(f"[SIM] Drive to {waypoint} from pose {robot_pose}")
        print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))
        return

    # imports camera / wheel calibration parameters (robust, multi-location)
    def _read_scalar_any(paths, default_val):
        for path in paths:
            try:
                arr = np.loadtxt(path, delimiter=',')
                if np.ndim(arr) == 0:
                    return float(arr)
                flat = np.ravel(arr)
                if flat.size > 0:
                    return float(flat[0])
            except Exception:
                continue
        print(f"[CALIB] Using default {default_val} (files not found: {paths})")
        return float(default_val)

    scale_val = _read_scalar_any([
        os.path.join(BASE_DIR, 'calibration', 'param', 'scale.txt'),
        os.path.join(os.path.dirname(BASE_DIR), 'Week05-06', 'calibration', 'param', 'scale.txt'),
    ], default_val=50.0)      # ticks per meter
    baseline_val = _read_scalar_any([
        os.path.join(BASE_DIR, 'calibration', 'param', 'baseline.txt'),
        os.path.join(os.path.dirname(BASE_DIR), 'Week05-06', 'calibration', 'param', 'baseline.txt'),
    ], default_val=0.15)   # metres (wheelbase)

    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point

    # Compute relative goal
    dx = waypoint[0] - robot_pose[0]
    dy = waypoint[1] - robot_pose[1]
    goal_theta = np.arctan2(dy, dx)
    dtheta = goal_theta - robot_pose[2]
    # Normalize angle to [-pi, pi]
    dtheta = (dtheta + np.pi) % (2*np.pi) - np.pi
    dist = np.hypot(dx, dy)

    # Very rough time estimates using calibration scale (ticks per meter)
    ticks_per_meter = float(scale_val)
    # very rough mapping of radians to wheel ticks based on baseline
    turn_ticks_per_rad = max(1e-3, float(baseline_val))  # placeholder scaling
    wheel_vel = 30  # tick/s
    turn_time = abs(dtheta) * turn_ticks_per_rad / wheel_vel
    drive_time = (dist * ticks_per_meter) / wheel_vel

    print(f"Turn {np.degrees(dtheta):.1f} deg for {turn_time:.2f}s; Drive {dist:.2f} m for {drive_time:.2f}s")
    ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
    ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    ####################################################

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))


def get_robot_pose():
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # update the robot pose [x,y,theta]
    robot_pose = [0.0,0.0,0.0] # replace with your calculation
    ####################################################

    return robot_pose

# main loop

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='Week07-08/M3_prac_map_full.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--strategy", choices=['order','nearest'], default='order', help='Target selection strategy for known/partial levels')
    parser.add_argument("--level", choices=['known','partial','minimal'], default='known', help='Run mode')
    parser.add_argument("--simulate", action='store_true', help='Do not move robot; print planned actions only')
    parser.add_argument("--viz", action='store_true', help='Show OpenCV grid visualisation')
    parser.add_argument("--gui", action='store_true', help='Show Pygame GUI (camera, detector, SLAM)')
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)

    # Optional: init EKF (for GUI and future get_robot_pose)
    if args.gui and EKF is not None and Robot is not None:
        try:
            # Load calibration from Week05-06 by default
            fileK = os.path.join(W56_DIR, 'calibration', 'param', 'intrinsic.txt')
            fileD = os.path.join(W56_DIR, 'calibration', 'param', 'distCoeffs.txt')
            fileS = os.path.join(W56_DIR, 'calibration', 'param', 'scale.txt')
            fileB = os.path.join(W56_DIR, 'calibration', 'param', 'baseline.txt')
            K = np.loadtxt(fileK, delimiter=',')
            D = np.loadtxt(fileD, delimiter=',')
            S = np.loadtxt(fileS, delimiter=',')
            B = np.loadtxt(fileB, delimiter=',')
            ekf = EKF(Robot(B, S, K, D))
        except Exception as e:
            print(f"[EKF] Could not initialise: {e}")
            ekf = None
    gui = SimpleGUI() if args.gui else None

    # read in the true map (if minimal, this may only have ArUcos)
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    # read shopping list with fallback
    try:
        search_list = read_search_list()
    except FileNotFoundError:
        # fallback to practice shopping list
        with open('Week07-08/M3_prac_shopping_list.txt','r') as fd:
            search_list = [l.strip() for l in fd.readlines() if l.strip()]
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    # --- Arena/grid setup ---
    # Arena is 2.4m x 2.4m centred at (0,0) -> coordinates in [-1.2, 1.2]
    grid_size = 0.1  # meters per grid cell
    half_size = 1.2
    arena_origin = [-half_size, -half_size]
    grid_width = int((2 * half_size) / grid_size)  # 24
    grid_height = int((2 * half_size) / grid_size) # 24
    if args.level == 'known':
        grid = build_grid(fruits_list, fruits_true_pos, aruco_true_pos, search_list, grid_size, arena_origin, grid_width, grid_height)
    else:
        # minimal/partial: only ArUcos are obstacles initially
        grid = np.zeros((grid_width, grid_height), dtype=int)
        for pos in aruco_true_pos:
            gx, gy = pos_to_grid(pos[0], pos[1], grid_size, arena_origin)
            if 0 <= gx < grid_width and 0 <= gy < grid_height:
                grid[gx][gy] = 1
    # Inflate for robot radius (~0.12 m -> ~1 cell); increase if clipping obstacles
    grid = inflate_grid(grid, radius_cells=1)

    # Get robot start pose (replace with SLAM pose)
    robot_pose = get_robot_pose()
    # Perform a 360-degree scan at start
    robot_pose = spin_360(robot_pose, simulate=args.simulate)

    if args.level == 'known':
        # Map fruits in search_list to indices in fruits_list
        target_indices = []
        for target in search_list:
            for i, fruit in enumerate(fruits_list):
                if fruit == target:
                    target_indices.append(i)
                    break
        remaining = target_indices.copy()
        while remaining:
            next_idx = choose_next_target(args.strategy, remaining, fruits_list, fruits_true_pos, robot_pose)
            goal_xy = (float(fruits_true_pos[next_idx][0]), float(fruits_true_pos[next_idx][1]))
            # Preview path
            if args.viz:
                rgx, rgy = pos_to_grid(robot_pose[0], robot_pose[1], grid_size, arena_origin)
                tgx, tgy = pos_to_grid(goal_xy[0], goal_xy[1], grid_size, arena_origin)
                path = astar(grid, (rgx, rgy), (tgx, tgy))
                draw_viz(grid, grid_size, arena_origin, robot_pose, goal_xy, path, None)
                draw_camera_gui(ppi, known_targets=None)
            robot_pose = plan_and_follow_path(grid, robot_pose, goal_xy, grid_size, arena_origin, args.simulate)
            if gui:
                try:
                    gui.draw(ekf)
                except SystemExit:
                    pass
            ppi.set_velocity([0, 0])
            print(f"Arrived at {fruits_list[next_idx]}, waiting 2 seconds...")
            time.sleep(2)
            remaining.remove(next_idx)
        print("Finished all targets.")
    else:
        # Minimal/partial: explore and collect
        # Load ground-truth for simulation-only detections if available
        gt_map = None
        if args.simulate:
            try:
                with open('Week07-08/M3_prac_map_full.txt','r') as fd:
                    gt_map = json.load(fd)
            except Exception:
                gt_map = None

        coverage = generate_lawnmower_waypoints(bounds=(-half_size, half_size, -half_size, half_size), spacing=0.4)
        coverage_idx = 0
        known_targets = {}  # label -> (x,y)
        visited_labels = set()
        target_labels = {"garlic","lemon","pear","tomato","pumpkin"}

        while len(visited_labels) < len(target_labels):
            # Perception update
            if args.simulate:
                dets, obs = detect_targets_and_obstacles_sim(gt_map, robot_pose)
                # add obstacles
                for ox, oy in obs:
                    gx, gy = pos_to_grid(ox, oy, grid_size, arena_origin)
                    if 0 <= gx < grid_width and 0 <= gy < grid_height:
                        grid[gx][gy] = 1
                # add targets
                for label, x, y, _ in dets:
                    if label in target_labels and label not in visited_labels:
                        known_targets[label] = (x, y)
                # re-inflate after updates
                grid = inflate_grid(grid, radius_cells=1)
            else:
                # Run live detector to update grid and known targets
                grid, found_targets = update_dynamic_obstacles_from_detector_live(
                    grid, grid_size, arena_origin, grid_width, grid_height, robot_pose
                )
                for lbl, (tx, ty) in found_targets.items():
                    if lbl in target_labels and lbl not in visited_labels:
                        known_targets[lbl] = (tx, ty)
                # re-inflate after updates
                grid = inflate_grid(grid, radius_cells=1)

            # Decide goal: nearest known unvisited target else next coverage waypoint
            goal_xy = None
            if any(lbl not in visited_labels for lbl in known_targets.keys()):
                # choose nearest
                best_d = float('inf')
                for lbl, (tx, ty) in known_targets.items():
                    if lbl in visited_labels:
                        continue
                    d = (tx - robot_pose[0])**2 + (ty - robot_pose[1])**2
                    if d < best_d:
                        best_d = d
                        goal_xy = (tx, ty)
                        goal_label = lbl
            else:
                # exploration goal
                if coverage_idx >= len(coverage):
                    print("Exploration complete, no targets found. Stopping.")
                    break
                goal_xy = coverage[coverage_idx]
                goal_label = None
                coverage_idx += 1

            print(f"Heading to {'target ' + goal_label if goal_label else 'explore'} at {goal_xy}")
            if args.viz:
                # draw preview
                rgx, rgy = pos_to_grid(robot_pose[0], robot_pose[1], grid_size, arena_origin)
                gx, gy = pos_to_grid(goal_xy[0], goal_xy[1], grid_size, arena_origin)
                path = astar(grid, (rgx, rgy), (gx, gy))
                draw_viz(grid, grid_size, arena_origin, robot_pose, goal_xy, path, known_targets)
                draw_camera_gui(ppi, known_targets=known_targets)
            robot_pose = plan_and_follow_path(grid, robot_pose, goal_xy, grid_size, arena_origin, args.simulate)
            if gui:
                try:
                    gui.draw(ekf)
                except SystemExit:
                    pass

            # Check arrival and handle target dwell
            if goal_label is not None:
                if np.hypot(robot_pose[0]-goal_xy[0], robot_pose[1]-goal_xy[1]) <= 0.25:
                    ppi.set_velocity([0, 0])
                    print(f"Arrived at {goal_label}, waiting 2 seconds...")
                    time.sleep(2)
                    visited_labels.add(goal_label)

        print("Level 4 finished: targets visited:", visited_labels)
        if args.viz:
            cv2.destroyAllWindows()
