# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import math

# --- Constants (tunable) -------------------------------------------------
DIST_TOL = 0.10          # waypoint positional tolerance (m)
HEADING_TOL = 0.12       # heading tolerance (rad)
TARGET_RADIUS = 0.25     # marking requirement radius (m)
DWELL_SEC = 2.0          # required stop time at target (s)
BOUNDARY_MARGIN = 0.15   # extra margin around map extents (m)
# -------------------------------------------------------------------------

# import SLAM components
# (moved from inside get_robot_pose to top-level so import errors surface early)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'slam'))
from slam.ekf import EKF
from slam.robot import Robot
from slam.aruco_detector import aruco_detector

# import utility functions
sys.path.insert(0, "util")
from util.pibot import PenguinPi
import pygame  # for optional GUI

# Optional YOLO import deferred until requested
try:
    from YOLO.detector import Detector as YOLODetector
except Exception:
    YOLODetector = None

# Runtime globals (initialised in init_runtime)
_runtime = {
    'ekf': None,
    'aruco': None,
    'detector': None,
    'last_frame': None,
    'gui': False,
    'surface': None,
    'fonts': {}
}

def init_runtime(args, aruco_true_pos, fruits_true_pos):
    """Initialise EKF, ArUco detector and optional YOLO + GUI (mirrors operate.py style)"""
    if _runtime['ekf'] is not None:
        return
    # Calibration
    calib_dir = 'calibration/param/'
    try:
        K = np.loadtxt(calib_dir+'intrinsic.txt', delimiter=',')
        dist = np.loadtxt(calib_dir+'distCoeffs.txt', delimiter=',')
        scale = float(np.loadtxt(calib_dir+'scale.txt', delimiter=','))
        baseline = float(np.loadtxt(calib_dir+'baseline.txt', delimiter=','))
    except Exception:
        K = np.array([[1,0,0],[0,1,0],[0,0,1]], float)
        dist = np.zeros(5)
        scale = 0.0
        baseline = 0.1
    robot = Robot(baseline, scale, K, dist)
    ekf = EKF(robot)
    ar = aruco_detector(robot, marker_length=0.07)
    det = None
    if args.yolo_model and YOLODetector is not None:
        try:
            det = YOLODetector(args.yolo_model)
        except Exception:
            print('[init_runtime] Failed to load YOLO model.')
    _runtime['ekf'] = ekf
    _runtime['aruco'] = ar
    _runtime['detector'] = det
    # GUI setup
    if args.gui:
        pygame.font.init()
        _runtime['gui'] = True
        _runtime['surface'] = pygame.display.set_mode((960, 360))
        pygame.display.set_caption('Auto Fruit Search')
        _runtime['fonts']['small'] = pygame.font.SysFont('consolas',16)
        _runtime['fonts']['large'] = pygame.font.SysFont('consolas',20)
        _runtime['fruits'] = fruits_true_pos
        _runtime['aruco_pts'] = aruco_true_pos

def _gui_draw(pose):
    if not _runtime['gui']:
        return
    surf = _runtime['surface']
    frame = _runtime['last_frame']
    ekf = _runtime['ekf']
    det_out = _runtime.get('last_det_vis')
    if frame is None:
        return
    # Convert cv2 (RGB) to pygame surface
    def cv_to_surface(img):
        import numpy as _np
        if img is None: return None
        img = _np.rot90(img)  # rotate for pygame orientation
        pg_surf = pygame.surfarray.make_surface(img)
        pg_surf = pygame.transform.flip(pg_surf, True, False)
        return pg_surf
    cam_view = cv_to_surface(frame)
    if det_out is None:
        det_out = frame
    det_view = cv_to_surface(det_out)
    surf.fill((10,10,10))
    if cam_view: surf.blit(cam_view,(0,0))
    if det_view: surf.blit(det_view,(320,0))
    # Simple SLAM map panel
    map_panel = pygame.Surface((320,360))
    map_panel.fill((30,30,30))
    # draw markers and fruits
    fruits = _runtime.get('fruits', np.empty((0,2)))
    ar_pts = _runtime.get('aruco_pts', np.empty((0,2)))
    allx = np.concatenate([fruits[:,0], ar_pts[:,0]]) if len(fruits)>0 else ar_pts[:,0]
    ally = np.concatenate([fruits[:,1], ar_pts[:,1]]) if len(fruits)>0 else ar_pts[:,1]
    if allx.size==0:
        allx = np.array([-1,1]); ally=np.array([-1,1])
    minx,maxx = allx.min()-0.2, allx.max()+0.2
    miny,maxy = ally.min()-0.2, ally.max()+0.2
    def world_to_px(x,y):
        px = int((x - minx)/(maxx-minx+1e-6)*319)
        py = int((1 - (y - miny)/(maxy-miny+1e-6))*359)
        return px,py
    for p in ar_pts:
        px,py = world_to_px(p[0],p[1]); pygame.draw.circle(map_panel,(255,255,0),(px,py),4)
    for p in fruits:
        px,py = world_to_px(p[0],p[1]); pygame.draw.circle(map_panel,(255,0,0),(px,py),4)
    if pose is not None:
        rx,ry,th = pose
        px,py = world_to_px(rx,ry)
        pygame.draw.circle(map_panel,(255,255,255),(px,py),6)
    surf.blit(map_panel,(640,0))
    pygame.display.update()
import util.measure as measure


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
    with open('M3_prac_shopping_list.txt', 'r') as fd:
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
def _wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def confirm_arrival(goal, radius=TARGET_RADIUS, dwell=DWELL_SEC):
    """Ensure robot stays within radius for dwell seconds.
    Returns (arrived_bool, final_pose, avg_error).
    """
    start_pose = get_robot_pose()
    dist = np.hypot(goal[0]-start_pose[0], goal[1]-start_pose[1])
    if dist > radius:
        return False, start_pose, dist
    t0 = time.time()
    samples = []
    while time.time() - t0 < dwell:
        pose = get_robot_pose()
        d = np.hypot(goal[0]-pose[0], goal[1]-pose[1])
        samples.append(d)
        if d > radius:  # drifted out
            return False, pose, float(np.mean(samples))
        time.sleep(0.05)
    return True, get_robot_pose(), float(np.mean(samples))


def drive_to_point(waypoint, robot_pose):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    try:
        scale = np.loadtxt(fileS, delimiter=',')
    except Exception:
        scale = 0.0
    fileB = "calibration/param/baseline.txt"
    try:
        baseline = np.loadtxt(fileB, delimiter=',')
    except Exception:
        baseline = 0.1

    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point
    # (Closed-loop implementation added below while preserving the TODO block.)

    # Controller parameters (using global constants where applicable)
    heading_tol = HEADING_TOL
    dist_tol = DIST_TOL
    max_time = 25.0           # s safety timeout
    turn_speed = 25           # tick (turning)
    fwd_speed = 35            # tick (forward)
    turn_pulse = 0.12         # s per turning pulse
    drive_pulse = 0.15        # s per forward pulse

    def current_pose_and_errors():
        pose_now = get_robot_pose()
        dx = waypoint[0] - pose_now[0]
        dy = waypoint[1] - pose_now[1]
        dist = float(np.hypot(dx, dy))
        target_heading = np.arctan2(dy, dx)
        hdg_err = _wrap_angle(target_heading - pose_now[2])
        return pose_now, dist, hdg_err

    start_pose = robot_pose
    pose_now, dist, hdg_err = current_pose_and_errors()
    print(f"[drive_to_point] target={waypoint} start={start_pose} d0={dist:.2f} hdg_err={hdg_err:.2f}")

    reached = False
    phase = 'turn'
    start_time = time.time()
    while True:
        if time.time() - start_time > max_time:
            print("[drive_to_point] Timeout; abort.")
            break
        pose_now, dist, hdg_err = current_pose_and_errors()
        if dist <= dist_tol:
            ppi.set_velocity([0,0])
            reached = True
            break
        if phase == 'turn':
            if abs(hdg_err) <= heading_tol:
                phase = 'drive'
                continue
            direction = 1 if hdg_err > 0 else -1
            ppi.set_velocity([0, direction], turning_tick=turn_speed, time=turn_pulse)
        else:  # drive phase
            if abs(hdg_err) > heading_tol * 1.8:
                phase = 'turn'
                continue
            ppi.set_velocity([1, 0], tick=fwd_speed, time=drive_pulse)

    final_pose = get_robot_pose()
    residual = float(np.hypot(waypoint[0]-final_pose[0], waypoint[1]-final_pose[1]))
    print(f"[drive_to_point] done target={waypoint} final={final_pose} residual={residual:.2f} reached={reached}")
    return final_pose, reached, residual


def navigate_waypoints(waypoints, dwell_time=DWELL_SEC):
    for wp in waypoints:
        pose = get_robot_pose()
        final_pose, reached, residual = drive_to_point(wp, pose)
        if reached:
            ok, dwell_pose, avg_err = confirm_arrival(wp, TARGET_RADIUS, dwell_time)
            print(f"[navigate] dwell ok={ok} avg_err={avg_err:.2f} pose={dwell_pose}")
        else:
            print("[navigate] waypoint not reached within tolerance; skipping dwell")


def get_robot_pose():
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # update the robot pose [x,y,theta]
    # --- EKF-based pose estimation scaffold (non-destructive; keeps TODO comment) ---
    # Lazy initialisation of SLAM components the first time this function is invoked.
    # Falls back to last known pose (or origin) if any step fails, so existing flow doesn't break.
    global _ekf_ctx
    try:
        _ekf_ctx
    except NameError:
        _ekf_ctx = {}

    # Prefer runtime ekf if GUI/detector mode initialised
    if _runtime['ekf'] is not None:
        ekf = _runtime['ekf']
        detector = _runtime['aruco']
        # Pull frame
        try:
            frame = ppi.get_image()
            _runtime['last_frame'] = frame
        except Exception:
            frame = None
        # Encoders prediction
        try:
            wl, wr = ppi.get_wheels()
        except Exception:
            wl, wr = 0.0,0.0
        dt = 0.15
        try:
            drive_meas = measure.Drive(left_speed=wl, right_speed=wr, dt=dt)
            ekf.predict(drive_meas)
        except Exception:
            pass
        # Vision update
        meas = []
        if frame is not None:
            try:
                meas, _ = detector.detect_marker_positions(frame)
                ekf.add_landmarks(meas)
                ekf.update(meas)
            except Exception:
                pass
        # Optional YOLO inference for display
        if _runtime['detector'] is not None and frame is not None:
            try:
                yolo_in = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                _, vis = _runtime['detector'].detect_single_image(yolo_in)
                _runtime['last_det_vis'] = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            except Exception:
                _runtime['last_det_vis'] = None
        try:
            state = ekf.robot.state.reshape(-1)
            pose = [float(state[0]), float(state[1]), float(state[2])]
        except Exception:
            pose = [0.0,0.0,0.0]
        _gui_draw(pose)
        return pose

    if 'ekf' not in _ekf_ctx:  # legacy path (non GUI runtime)
        try:
            # Wheel & camera calibration
            try:
                wheels_scale = float(np.loadtxt('calibration/param/scale.txt', delimiter=','))
            except Exception:
                wheels_scale = 0.0
            try:
                wheels_width = float(np.loadtxt('calibration/param/baseline.txt', delimiter=','))
            except Exception:
                wheels_width = 0.1
            try:
                K = np.loadtxt('calibration/param/intrinsic.txt', delimiter=',')
            except Exception:
                K = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=float)
            try:
                dist = np.loadtxt('calibration/param/distCoeffs.txt', delimiter=',')
            except Exception:
                dist = np.zeros(5)
            robot = Robot(wheels_width=wheels_width, wheels_scale=wheels_scale, camera_matrix=K, camera_dist=dist)
            ekf = EKF(robot)
            detector = aruco_detector(robot)
            _ekf_ctx['ekf'] = ekf
            _ekf_ctx['detector'] = detector
            _ekf_ctx['last_time'] = time.time()
            _ekf_ctx['last_pose'] = [0.0,0.0,0.0]
        except Exception:
            return [0.0,0.0,0.0]  # initialise failure → origin pose

    ekf = _ekf_ctx['ekf']
    detector = _ekf_ctx['detector']
    last_time = _ekf_ctx['last_time']

    # 1. Wheel encoder based prediction
    try:
        wl, wr = ppi.get_wheels()  # ticks/s
    except Exception:
        wl, wr = 0.0, 0.0
    now = time.time()
    dt = max(1e-3, now - last_time)
    _ekf_ctx['last_time'] = now
    try:
        # reuse already imported measure module
        drive_meas = measure.Drive(left_speed=wl, right_speed=wr, dt=dt)
        ekf.predict(drive_meas)
    except Exception:
        pass

    # 2. Vision-based landmark (ArUco) update
    try:
        frame = ppi.get_image()
        measurements, _ = detector.detect_marker_positions(frame)
        # Add any new landmarks first
        ekf.add_landmarks(measurements)
        # Update only with those already in the map (conservative)
        existing = [m for m in measurements if m.tag in ekf.taglist]
        ekf.update(existing)
    except Exception:
        pass

    # 3. Extract pose
    try:
        state_vec = ekf.robot.state.reshape(-1)
        robot_pose = [float(state_vec[0]), float(state_vec[1]), float(state_vec[2])]
        _ekf_ctx['last_pose'] = robot_pose
    except Exception:
        robot_pose = _ekf_ctx.get('last_pose', [0.0,0.0,0.0])

    # replace with your calculation (above scaffold already provides SLAM-based pose)
    ####################################################

    return robot_pose

# --- Planning helpers for Level 2 ----------------------------------------

def build_costmap(fruits_true_pos, aruco_true_pos, resolution=0.1, inflation=0.20):
    """Build a simple occupancy grid.
    Obstacles: non-target fruits (will be filtered later) + ArUco blocks.
    Returns dict with grid, origin, resolution.
    """
    pts = np.vstack([fruits_true_pos, aruco_true_pos]) if len(fruits_true_pos)>0 else aruco_true_pos
    min_x, max_x = float(np.min(pts[:,0]) - BOUNDARY_MARGIN), float(np.max(pts[:,0]) + BOUNDARY_MARGIN)
    min_y, max_y = float(np.min(pts[:,1]) - BOUNDARY_MARGIN), float(np.max(pts[:,1]) + BOUNDARY_MARGIN)
    width = int(math.ceil((max_x - min_x)/resolution)) + 1
    height = int(math.ceil((max_y - min_y)/resolution)) + 1
    grid = np.zeros((height, width), dtype=np.uint8)  # 0 free, 1 obstacle
    def to_idx(x,y):
        ix = int(round((x - min_x)/resolution))
        iy = int(round((y - min_y)/resolution))
        return iy, ix
    # mark obstacles (aruco positions as pillars)
    for p in aruco_true_pos:
        iy, ix = to_idx(p[0], p[1])
        if 0 <= iy < height and 0 <= ix < width:
            grid[iy, ix] = 1
    # inflate obstacles
    rad = int(math.ceil(inflation/resolution))
    if rad>0:
        inflated = grid.copy()
        occ_indices = np.argwhere(grid==1)
        for (iy, ix) in occ_indices:
            y0 = max(0, iy-rad); y1 = min(height, iy+rad+1)
            x0 = max(0, ix-rad); x1 = min(width, ix+rad+1)
            inflated[y0:y1, x0:x1] = 1
        grid = inflated
    return {
        'grid': grid,
        'origin': (min_x, min_y),
        'resolution': resolution,
        'width': width,
        'height': height,
        'to_idx': to_idx
    }


def astar_search(costmap, start_xy, goal_xy):
    grid = costmap['grid']; res = costmap['resolution']; ox, oy = costmap['origin']
    def xy_to_node(pt):
        x,y = pt
        ix = int(round((x - ox)/res))
        iy = int(round((y - oy)/res))
        return (ix, iy)
    def node_to_xy(node):
        ix, iy = node
        return (ix*res + ox, iy*res + oy)
    start = xy_to_node(start_xy)
    goal = xy_to_node(goal_xy)
    if not (0 <= start[0] < costmap['width'] and 0 <= start[1] < costmap['height']):
        return []
    if not (0 <= goal[0] < costmap['width'] and 0 <= goal[1] < costmap['height']):
        return []
    # Use (iy,ix) indexing for grid access
    def is_free(n):
        ix, iy = n
        return 0 <= ix < costmap['width'] and 0 <= iy < costmap['height'] and grid[iy, ix] == 0
    import heapq
    openq = []
    g = {start:0.0}
    came = {}
    def h(n): return math.hypot(n[0]-goal[0], n[1]-goal[1])
    heapq.heappush(openq, (h(start), start))
    moves = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    while openq:
        _, current = heapq.heappop(openq)
        if current == goal:
            # reconstruct
            path = [current]
            while current in came:
                current = came[current]
                path.append(current)
            path.reverse()
            return [node_to_xy(n) for n in path]
        for dx,dy in moves:
            nxt = (current[0]+dx, current[1]+dy)
            if not is_free(nxt):
                continue
            tentative = g[current] + math.hypot(dx,dy)
            if tentative < g.get(nxt, 1e18):
                g[nxt] = tentative
                came[nxt] = current
                heapq.heappush(openq, (tentative + h(nxt), nxt))
    return []  # no path


def simplify_path(path, min_dist=0.15):
    if not path: return []
    simplified = [path[0]]
    for p in path[1:]:
        if math.hypot(p[0]-simplified[-1][0], p[1]-simplified[-1][1]) >= min_dist:
            simplified.append(p)
    if simplified[-1] != path[-1]:
        simplified.append(path[-1])
    return simplified


# --- Dynamic obstacle & replan support (Level 3 scaffold) -----------------
class DynamicWorld:
    """Tracks known obstacles and triggers replans when new obstacles added."""
    def __init__(self, base_costmap, inflation=0.25):
        self.base = base_costmap
        self.inflation = inflation
        self.dynamic_obstacles = []  # list of (x,y)

    def add_obstacle(self, pos):
        # Avoid duplicates (within small radius)
        for ox,oy in self.dynamic_obstacles:
            if math.hypot(ox-pos[0], oy-pos[1]) < 0.08:
                return False
        self.dynamic_obstacles.append(tuple(pos))
        return True

    def rebuild_costmap(self):
        # Copy grid then stamp new inflated obstacles
        cm = self.base
        grid = cm['grid'].copy()
        res = cm['resolution']
        ox0, oy0 = cm['origin']
        h,w = grid.shape
        rad = int(math.ceil(self.inflation/res))
        for (x,y) in self.dynamic_obstacles:
            ix = int(round((x - ox0)/res))
            iy = int(round((y - oy0)/res))
            if 0 <= ix < w and 0 <= iy < h:
                y0 = max(0, iy-rad); y1 = min(h, iy+rad+1)
                x0 = max(0, ix-rad); x1 = min(w, ix+rad+1)
                grid[y0:y1, x0:x1] = 1
        new_cm = dict(cm)
        new_cm['grid'] = grid
        return new_cm

# Mock fruit/obstacle detector placeholder -------------------------------
def detect_new_obstacles_frame():
    """Placeholder for Level 3: should run fruit detector and return list of (x,y) new obstacle fruit positions not on shopping list.
    Currently returns empty list.
    """
    return []

# --- Level 4 exploration helpers -----------------------------------------

def generate_exploration_pattern(costmap, lane_spacing=0.40):
    """Generate a simple lawnmower (boustrophedon) coverage path over free cells.
    Returns list of (x,y) waypoints in world coordinates.
    """
    grid = costmap['grid']; res = costmap['resolution']; ox, oy = costmap['origin']
    h, w = grid.shape
    waypoints = []
    # Determine lane step in cells
    lane_step = max(1, int(round(lane_spacing / res)))
    # Sweep across x while moving along y lanes
    for row_block in range(0, h, lane_step):
        y_cell = min(row_block, h-1)
        y = y_cell*res + oy
        # collect free cells in this row
        row_cells = [(x, y_cell) for x in range(w) if grid[y_cell, x] == 0]
        if not row_cells:
            continue
        if (row_block // lane_step) % 2 == 0:
            seq = row_cells
        else:
            seq = list(reversed(row_cells))
        # convert to world coords, sub-sample so we don't overpopulate
        last = None
        for (x_cell, y_cell2) in seq:
            wx = x_cell*res + ox
            wy = y_cell2*res + oy
            if last is None or math.hypot(wx-last[0], wy-last[1]) >= 0.18:
                waypoints.append([wx, wy])
                last = [wx, wy]
    return waypoints

# Placeholder target detection (would integrate fruit classifier + triangulation)
def detect_new_targets_frame():
    """Return list of newly detected target fruit positions (x,y) not yet visited.
    Currently empty placeholder."""
    return []

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map_full.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--level", type=int, default=1, choices=[1,2,3,4], help="Navigation level (1=waypoints,2=known map,3=partial,4=minimal)")
    parser.add_argument("--auto", action='store_true', help="Auto-run without interactive waypoint input (levels>=2 use planner; level1 can take hardcoded demo list)")
    parser.add_argument("--log", type=str, default=None, help="Optional log file to append pose + events")
    parser.add_argument("--gui", action='store_true', help="Show combined camera/detector/SLAM GUI (pygame)")
    parser.add_argument("--yolo_model", type=str, default='', help="Path to YOLO model to enable detector pane")
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)

    # simple logger helper
    def log(msg):
        stamp = time.strftime('%H:%M:%S')
        line = f"[{stamp}] {msg}"
        print(line)
        if args.log:
            try:
                with open(args.log,'a') as lf: lf.write(line+"\n")
            except Exception:
                pass

    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    # Initialise richer runtime (after map loaded so we know extents for GUI) if requested
    if args.gui or args.yolo_model:
        init_runtime(args, aruco_true_pos, fruits_true_pos)

    # Compute arena bounds (min/max) for basic waypoint validation
    try:
        all_pts = np.vstack([fruits_true_pos, aruco_true_pos])
        arena_min_x = float(np.min(all_pts[:,0]) - BOUNDARY_MARGIN)
        arena_max_x = float(np.max(all_pts[:,0]) + BOUNDARY_MARGIN)
        arena_min_y = float(np.min(all_pts[:,1]) - BOUNDARY_MARGIN)
        arena_max_y = float(np.max(all_pts[:,1]) + BOUNDARY_MARGIN)
    except Exception:
        arena_min_x = arena_min_y = -10.0
        arena_max_x = arena_max_y = 10.0

    def in_bounds(x,y):
        return arena_min_x <= x <= arena_max_x and arena_min_y <= y <= arena_max_y

    def run_level1():
        log("Running Level 1 (manual/ waypoint navigation)")
        if args.auto:
            # demo: drive through first two fruit positions (can adjust)
            auto_wps = [fruits_true_pos[i].tolist() for i in range(min(2, len(fruits_true_pos)))]
            navigate_waypoints(auto_wps)
            return
        while True:
            x = input("X coordinate of the waypoint: ")
            try: x = float(x)
            except ValueError: log("Please enter a number."); continue
            y = input("Y coordinate of the waypoint: ")
            try: y = float(y)
            except ValueError: log("Please enter a number."); continue
            if not in_bounds(x,y):
                log("Waypoint outside arena bounds; rejected.")
                continue
            robot_pose = get_robot_pose()
            waypoint = [x,y]
            final_pose, reached, residual = drive_to_point(waypoint,robot_pose)
            if reached:
                ok, dwell_pose, avg_err = confirm_arrival(waypoint)
                log(f"Arrive check ok={ok} residual={residual:.2f} avg_dwell_err={avg_err:.2f}")
            else:
                log(f"Did not reach waypoint residual={residual:.2f}")
            ppi.set_velocity([0, 0])
            uInput = input("Add a new waypoint? [Y/N]")
            if uInput.upper() == 'N':
                break

    def run_level2():
        log("Running Level 2 (known map path planning)")
        # Build costmap (inflation protects path from marker collisions)
        resolution = 0.10
        inflation = 0.25
        costmap = build_costmap(fruits_true_pos, aruco_true_pos, resolution=resolution, inflation=inflation)
        # Plan path visiting fruits in search_list order (targets known in full map)
        # Extract target waypoints by matching names in order
        targets = []
        for name in search_list:
            for idx, fname in enumerate(fruits_list):
                if fname == name:
                    targets.append(fruits_true_pos[idx].tolist())
                    break
        # Start pose from SLAM now
        start_pose = get_robot_pose()
        current_xy = [start_pose[0], start_pose[1]]
        full_plan = []
        for goal in targets:
            segment = astar_search(costmap, current_xy, goal)
            if not segment:
                log(f"No path to target {goal}; aborting remaining.")
                break
            segment = simplify_path(segment, min_dist=0.18)
            # Append segment excluding its first node if overlapping with previous end
            if full_plan and segment and np.allclose(full_plan[-1], segment[0]):
                full_plan.extend(segment[1:])
            else:
                full_plan.extend(segment)
            current_xy = goal
        if not full_plan:
            log("Planner produced empty plan; falling back to manual level1.")
            run_level1()
            return
        log(f"Planned {len(full_plan)} waypoints")
        navigate_waypoints(full_plan)

    def run_level3():
        log("Running Level 3 (partial map with obstacle discovery)")
        # Initial costmap from partial map (provided map expected to contain only targets + markers).
        resolution = 0.10
        inflation = 0.25
        base_costmap = build_costmap(fruits_true_pos, aruco_true_pos, resolution=resolution, inflation=inflation)
        world = DynamicWorld(base_costmap, inflation=inflation)

        # Build ordered target list from search_list (they are known in partial map)
        targets = []
        for name in search_list:
            for idx, fname in enumerate(fruits_list):
                if fname == name:
                    targets.append(fruits_true_pos[idx].tolist())
                    break
        if not targets:
            log("No targets parsed; aborting to Level 1")
            run_level1(); return

        # Plan + execute sequentially with replan if path blocked by new obstacle
        start_pose = get_robot_pose()
        current_xy = [start_pose[0], start_pose[1]]
        for goal in targets:
            while True:
                cm = world.rebuild_costmap()
                path = astar_search(cm, current_xy, goal)
                if not path:
                    log(f"No path to target {goal}; stopping Level 3.")
                    return
                path = simplify_path(path, min_dist=0.18)
                log(f"Path to target {goal} with {len(path)} waypoints")
                # Execute path incrementally, checking for new obstacles each step
                aborted = False
                for wp in path[1:]:  # skip current position assumed at path[0]
                    final_pose, reached, residual = drive_to_point(wp, get_robot_pose())
                    current_xy = [final_pose[0], final_pose[1]]
                    # Scan for new obstacles periodically (placeholder returns none now)
                    new_obs = detect_new_obstacles_frame()
                    added_any = False
                    for ob in new_obs:
                        if world.add_obstacle(ob):
                            added_any = True
                    if added_any:
                        log("New obstacle(s) detected -> replanning")
                        aborted = True
                        break
                if aborted:
                    continue  # recompute new plan
                # Arrived at goal ensure dwell
                ok, dwell_pose, avg_err = confirm_arrival(goal)
                log(f"Reached target {goal} dwell_ok={ok} avg_err={avg_err:.2f}")
                break  # proceed to next goal

    def run_level4():
        log("Running Level 4 (minimal map exploration + target collection)")
        # Minimal map only has markers -> no fruit positions known. We'll explore, detect targets, then navigate to detected ones.
        resolution = 0.10
        inflation = 0.25
        base_costmap = build_costmap(np.empty((0,2)), aruco_true_pos, resolution=resolution, inflation=inflation)
        world = DynamicWorld(base_costmap, inflation=inflation)

        # Coverage pattern
        exploration_path = generate_exploration_pattern(base_costmap, lane_spacing=0.40)
        log(f"Exploration pattern waypoints: {len(exploration_path)}")

        discovered_targets = []  # (x,y) positions
        visited_targets = []

        def is_new_target(pt):
            for a in discovered_targets:
                if math.hypot(a[0]-pt[0], a[1]-pt[1]) < 0.12: return False
            return True

        # Main exploration loop
        idx = 0
        while idx < len(exploration_path):
            wp = exploration_path[idx]
            final_pose, reached, residual = drive_to_point(wp, get_robot_pose())
            # Periodically "detect" targets
            new_targets = detect_new_targets_frame()  # placeholder returns [] now
            for t in new_targets:
                if is_new_target(t):
                    discovered_targets.append(t)
                    log(f"Discovered target at {t}")
            # If we have undiscovered targets to visit (in any order), go visit them before continuing coverage
            pending = [t for t in discovered_targets if t not in visited_targets]
            while pending:
                # Plan path to nearest pending target
                cm = world.rebuild_costmap()
                # pick nearest current_xy
                current_pose = get_robot_pose()
                current_xy = [current_pose[0], current_pose[1]]
                pending.sort(key=lambda p: math.hypot(p[0]-current_xy[0], p[1]-current_xy[1]))
                goal = pending[0]
                path = astar_search(cm, current_xy, goal)
                if not path:
                    log(f"Cannot reach discovered target {goal}; marking visited (unreachable).")
                    visited_targets.append(goal)
                    pending = [t for t in discovered_targets if t not in visited_targets]
                    continue
                path = simplify_path(path, min_dist=0.18)
                log(f"Visiting discovered target {goal} with {len(path)} segment waypoints")
                for step in path[1:]:
                    drive_to_point(step, get_robot_pose())
                ok, dwell_pose, avg_err = confirm_arrival(goal)
                log(f"Target {goal} dwell_ok={ok} avg_err={avg_err:.2f}")
                visited_targets.append(goal)
                pending = [t for t in discovered_targets if t not in visited_targets]
            idx += 1
        log("Exploration complete. Visited targets: {}".format(len(visited_targets)))

    # Dispatcher ------------------------------------------------------------
    {1: run_level1, 2: run_level2, 3: run_level3, 4: run_level4}[args.level]()
