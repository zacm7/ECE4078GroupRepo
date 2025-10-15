# import math
# import heapq
# import numpy as np


# def infer_bounds(points, margin=0.25):
#     pts = np.array(points, dtype=float)
#     minx, miny = np.min(pts, axis=0) - margin
#     maxx, maxy = np.max(pts, axis=0) + margin
#     return float(minx), float(miny), float(maxx), float(maxy)


# def world_to_grid(x, y, bounds, res):
#     minx, miny, _, _ = bounds
#     gx = int(round((x - minx) / res))
#     gy = int(round((y - miny) / res))
#     return gx, gy


# def grid_to_world(ix, iy, bounds, res):
#     minx, miny, _, _ = bounds
#     x = minx + ix * res
#     y = miny + iy * res
#     return float(x), float(y)


# def build_occupancy(obstacles_xy, bounds, res, inflation_radius):
#     minx, miny, maxx, maxy = bounds
#     w = int(math.ceil((maxx - minx) / res)) + 1
#     h = int(math.ceil((maxy - miny) / res)) + 1
#     grid = np.zeros((w, h), dtype=np.uint8)
#     r_cells = int(math.ceil(inflation_radius / res))
#     for (ox, oy) in obstacles_xy:
#         cx, cy = world_to_grid(ox, oy, bounds, res)
#         x0, x1 = max(0, cx - r_cells), min(w - 1, cx + r_cells)
#         y0, y1 = max(0, cy - r_cells), min(h - 1, cy + r_cells)
#         for ix in range(x0, x1 + 1):
#             for iy in range(y0, y1 + 1):
#                 if (ix - cx) ** 2 + (iy - cy) ** 2 <= r_cells ** 2:
#                     grid[ix, iy] = 1
#     return grid


# def nearest_free(grid, start):
#     from collections import deque
#     w, h = grid.shape
#     sx, sy = start
#     if 0 <= sx < w and 0 <= sy < h and grid[sx, sy] == 0:
#         return start
#     q = deque([start])
#     seen = {start}
#     moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
#     while q:
#         x, y = q.popleft()
#         for dx, dy in moves:
#             nx, ny = x + dx, y + dy
#             if (nx, ny) in seen:
#                 continue
#             if 0 <= nx < w and 0 <= ny < h:
#                 if grid[nx, ny] == 0:
#                     return (nx, ny)
#                 seen.add((nx, ny))
#                 q.append((nx, ny))
#     return None


# def a_star(grid, start, goal):
#     w, h = grid.shape
#     sx, sy = start
#     gx, gy = goal
#     if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
#         return None
#     if grid[gx, gy]:
#         ng = nearest_free(grid, (gx, gy))
#         if ng is None:
#             return None
#         gx, gy = ng
#     if grid[sx, sy]:
#         ns = nearest_free(grid, (sx, sy))
#         if ns is None:
#             return None
#         sx, sy = ns
#     moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
#              (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
#              (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
#     hcost = lambda x, y: math.hypot(x - gx, y - gy)
#     openq = []
#     heapq.heappush(openq, (hcost(sx, sy), 0.0, (sx, sy), None))
#     came = {}
#     gscore = {(sx, sy): 0.0}
#     visited = set()
#     while openq:
#         f, g, (x, y), parent = heapq.heappop(openq)
#         if (x, y) in visited:
#             continue
#         visited.add((x, y))
#         came[(x, y)] = parent
#         if (x, y) == (gx, gy):
#             path = []
#             cur = (x, y)
#             while cur is not None:
#                 path.append(cur)
#                 cur = came[cur]
#             path.reverse()
#             return path
#         for dx, dy, c in moves:
#             nx, ny = x + dx, y + dy
#             if not (0 <= nx < w and 0 <= ny < h):
#                 continue
#             if grid[nx, ny]:
#                 continue
#             ng = g + c
#             if ng < gscore.get((nx, ny), 1e18):
#                 gscore[(nx, ny)] = ng
#                 nf = ng + hcost(nx, ny)
#                 heapq.heappush(openq, (nf, ng, (nx, ny), (x, y)))
#     return None


# def sparsify_path(world_pts, angle_eps_deg=5.0):
#     if len(world_pts) <= 2:
#         return world_pts
#     keep = [world_pts[0]]
#     for i in range(1, len(world_pts) - 1):
#         x0, y0 = keep[-1]
#         x1, y1 = world_pts[i]
#         x2, y2 = world_pts[i + 1]
#         v1 = np.array([x1 - x0, y1 - y0])
#         v2 = np.array([x2 - x1, y2 - y1])
#         n1 = np.linalg.norm(v1)
#         n2 = np.linalg.norm(v2)
#         if n1 < 1e-6 or n2 < 1e-6:
#             continue
#         a1 = math.atan2(v1[1], v1[0])
#         a2 = math.atan2(v2[1], v2[0])
#         d = abs(((a2 - a1 + math.pi) % (2 * math.pi)) - math.pi)
#         if math.degrees(d) > angle_eps_deg:
#             keep.append([x1, y1])
#     keep.append(world_pts[-1])
#     return keep


# def plan_waypoints(robot_xy, targets_xy, obstacles_xy, grid_res=0.02, robot_radius=0.12, safety_margin=0.05):
#     all_pts = obstacles_xy + targets_xy + [robot_xy]
#     bounds = infer_bounds(all_pts, margin=0.25)
#     grid = build_occupancy(obstacles_xy, bounds, grid_res, robot_radius + safety_margin)

#     waypoints = []
#     cur_xy = robot_xy
#     for tgt in targets_xy:
#         s = world_to_grid(cur_xy[0], cur_xy[1], bounds, grid_res)
#         g = world_to_grid(tgt[0], tgt[1], bounds, grid_res)
#         path = a_star(grid, s, g)
#         if path is None:
#             raise RuntimeError(f"No path found from {cur_xy} to {tgt}")
#         world_path = [list(grid_to_world(ix, iy, bounds, grid_res)) for (ix, iy) in path]
#         sparse = sparsify_path(world_path, angle_eps_deg=5.0)
#         waypoints.extend(sparse[1:] if waypoints else sparse)
#         cur_xy = tgt
#     return waypoints
# astar_planning.py
import math
import heapq
import numpy as np


def infer_bounds(points, margin=0.25):
    pts = np.array(points, dtype=float)
    minx, miny = np.min(pts, axis=0) - margin
    maxx, maxy = np.max(pts, axis=0) + margin
    return float(minx), float(miny), float(maxx), float(maxy)


def world_to_grid(x, y, bounds, res):
    minx, miny, _, _ = bounds
    gx = int(round((x - minx) / res))
    gy = int(round((y - miny) / res))
    return gx, gy


def grid_to_world(ix, iy, bounds, res):
    minx, miny, _, _ = bounds
    x = minx + ix * res
    y = miny + iy * res
    return float(x), float(y)


def build_occupancy(obstacles_xy, bounds, res, inflation_radius):
    """Legacy occupancy builder: uses a single inflation radius for all obstacles.

    Kept for backward compatibility; prefer build_occupancy_mixed for per-obstacle radii.
    """
    minx, miny, maxx, maxy = bounds
    w = int(math.ceil((maxx - minx) / res)) + 1
    h = int(math.ceil((maxy - miny) / res)) + 1
    grid = np.zeros((w, h), dtype=np.uint8)
    r_cells = int(math.ceil(inflation_radius / res))
    for (ox, oy) in obstacles_xy:
        cx, cy = world_to_grid(ox, oy, bounds, res)
        x0, x1 = max(0, cx - r_cells), min(w - 1, cx + r_cells)
        y0, y1 = max(0, cy - r_cells), min(h - 1, cy + r_cells)
        for ix in range(x0, x1 + 1):
            for iy in range(y0, y1 + 1):
                if (ix - cx) ** 2 + (iy - cy) ** 2 <= r_cells ** 2:
                    grid[ix, iy] = 1
    return grid


def build_occupancy_mixed(obstacles_xy, bounds, res, default_radius):
    """Build occupancy with per-obstacle inflation radii.

    obstacles_xy entries can be either [x, y] (uses default_radius) or [x, y, r] (uses r).
    """
    minx, miny, maxx, maxy = bounds
    w = int(math.ceil((maxx - minx) / res)) + 1
    h = int(math.ceil((maxy - miny) / res)) + 1
    grid = np.zeros((w, h), dtype=np.uint8)
    for ob in obstacles_xy:
        if isinstance(ob, (list, tuple)) and len(ob) >= 2:
            ox, oy = float(ob[0]), float(ob[1])
            r = float(ob[2]) if (len(ob) >= 3) else float(default_radius)
        else:
            # skip malformed
            continue
        cx, cy = world_to_grid(ox, oy, bounds, res)
        r_cells = int(math.ceil(r / res))
        x0, x1 = max(0, cx - r_cells), min(w - 1, cx + r_cells)
        y0, y1 = max(0, cy - r_cells), min(h - 1, cy + r_cells)
        for ix in range(x0, x1 + 1):
            for iy in range(y0, y1 + 1):
                if (ix - cx) ** 2 + (iy - cy) ** 2 <= r_cells ** 2:
                    grid[ix, iy] = 1
    return grid


def nearest_free(grid, start):
    from collections import deque
    w, h = grid.shape
    sx, sy = start
    if 0 <= sx < w and 0 <= sy < h and grid[sx, sy] == 0:
        return start
    q = deque([start])
    seen = {start}
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while q:
        x, y = q.popleft()
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if (nx, ny) in seen:
                continue
            if 0 <= nx < w and 0 <= ny < h:
                if grid[nx, ny] == 0:
                    return (nx, ny)
                seen.add((nx, ny))
                q.append((nx, ny))
    return None


def a_star(grid, start, goal):
    w, h = grid.shape
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return None
    if grid[gx, gy]:
        ng = nearest_free(grid, (gx, gy))
        if ng is None:
            return None
        gx, gy = ng
    if grid[sx, sy]:
        ns = nearest_free(grid, (sx, sy))
        if ns is None:
            return None
        sx, sy = ns
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
             (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
    hcost = lambda x, y: math.hypot(x - gx, y - gy)
    openq = []
    heapq.heappush(openq, (hcost(sx, sy), 0.0, (sx, sy), None))
    came = {}
    gscore = {(sx, sy): 0.0}
    visited = set()
    while openq:
        f, g, (x, y), parent = heapq.heappop(openq)
        if (x, y) in visited:
            continue
        visited.add((x, y))
        came[(x, y)] = parent
        if (x, y) == (gx, gy):
            path = []
            cur = (x, y)
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            path.reverse()
            return path
        for dx, dy, c in moves:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if grid[nx, ny]:
                continue
            ng = g + c
            if ng < gscore.get((nx, ny), 1e18):
                gscore[(nx, ny)] = ng
                nf = ng + hcost(nx, ny)
                heapq.heappush(openq, (nf, ng, (nx, ny), (x, y)))
    return None


def sparsify_path(world_pts, angle_eps_deg=5.0):
    if len(world_pts) <= 2:
        return world_pts
    keep = [world_pts[0]]
    for i in range(1, len(world_pts) - 1):
        x0, y0 = keep[-1]
        x1, y1 = world_pts[i]
        x2, y2 = world_pts[i + 1]
        v1 = np.array([x1 - x0, y1 - y0])
        v2 = np.array([x2 - x1, y2 - y1])
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        a1 = math.atan2(v1[1], v1[0])
        a2 = math.atan2(v2[1], v2[0])
        d = abs(((a2 - a1 + math.pi) % (2 * math.pi)) - math.pi)
        if math.degrees(d) > angle_eps_deg:
            keep.append([x1, y1])
    keep.append(world_pts[-1])
    return keep


def _seg_point_min_dist(ax, ay, bx, by, px, py):
    """Minimum distance from point P to segment AB."""
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx * vx + vy * vy
    if vv <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
    cx, cy = ax + t * vx, ay + t * vy
    return math.hypot(px - cx, py - cy)


def sparsify_path_collision_aware(world_pts, obstacles_xy, clearance_radius, angle_eps_deg=5.0):
    """Greedy sparsification that never creates a segment violating obstacle clearance.

    - Starts with first point and tries to connect directly to farther points.
    - If the direct segment from last kept point to candidate j is clear, keep extending.
    - Otherwise, keep j-1 and continue.
    - Also optionally prunes near-collinear points via angle threshold when safe.
    """
    if len(world_pts) <= 2:
        return world_pts

    # First, do a light pruning for near-collinear segments to reduce noise
    prelim = sparsify_path(world_pts, angle_eps_deg=angle_eps_deg)
    if len(prelim) <= 2:
        return prelim

    kept = [prelim[0]]
    i = 0
    while i < len(prelim) - 1:
        # Try to skip as far as possible while keeping clearance
        j = i + 1
        best_j = j
        while j < len(prelim):
            ax, ay = kept[-1]
            bx, by = prelim[j]
            # Check segment clearance against all obstacles
            clear = True
            for ob in obstacles_xy:
                if isinstance(ob, (list, tuple)) and len(ob) >= 2:
                    ox, oy = float(ob[0]), float(ob[1])
                    r = float(ob[2]) if (len(ob) >= 3) else float(clearance_radius)
                else:
                    continue
                if _seg_point_min_dist(ax, ay, bx, by, ox, oy) < r:
                    clear = False
                    break
            if clear:
                best_j = j
                j += 1
            else:
                break
        kept.append(prelim[best_j])
        i = best_j
    return kept

def _upsample_segment(segment_pts, max_step=0.03):
    """Linearly upsample a short final segment so robot approaches smoothly.
    Inserts intermediate points if consecutive points are farther than max_step.
    """
    if len(segment_pts) <= 1:
        return segment_pts
    out = [segment_pts[0]]
    for a, b in zip(segment_pts[:-1], segment_pts[1:]):
        ax, ay = a
        bx, by = b
        d = math.hypot(bx - ax, by - ay)
        if d <= max_step:
            out.append([bx, by])
        else:
            n = int(math.ceil(d / max_step))
            for k in range(1, n + 1):
                t = k / float(n)
                out.append([ax + (bx - ax) * t, ay + (by - ay) * t])
    return out


def plan_waypoints(robot_xy, targets_xy, obstacles_xy, grid_res=0.02, robot_radius=0.12, safety_margin=0.05):
    """Plan sequential A* paths to each target (on discrete grid), but ensure final waypoint equals true target.
    Also upsample the last small segment before the target for smoother approach.
    """
    all_pts = obstacles_xy + targets_xy + [robot_xy]
    bounds = infer_bounds(all_pts, margin=0.25)
    default_clearance = float(robot_radius + safety_margin)
    # Support per-obstacle radii (elements may be [x,y] or [x,y,r])
    grid = build_occupancy_mixed(obstacles_xy, bounds, grid_res, default_clearance)

    waypoints = []
    cur_xy = robot_xy
    for tgt in targets_xy:
        s = world_to_grid(cur_xy[0], cur_xy[1], bounds, grid_res)
        g = world_to_grid(tgt[0], tgt[1], bounds, grid_res)
        path = a_star(grid, s, g)
        if path is None:
            raise RuntimeError(f"No path found from {cur_xy} to {tgt}")
        world_path = [list(grid_to_world(ix, iy, bounds, grid_res)) for (ix, iy) in path]

        # Sparsify path but ensure segments maintain clearance from obstacles
        sparse = sparsify_path_collision_aware(world_path, obstacles_xy, default_clearance, angle_eps_deg=5.0)

        # Ensure last point is exactly the true target coordinates (avoid grid center quantization)
        if len(sparse) >= 1:
            sparse[-1] = [float(tgt[0]), float(tgt[1])]

        # Upsample the final few meters to make a smooth approach (e.g. max step ~ 3cm)
        # Take last upsample_len points (or whole sparse if small)
        upsample_len = min(len(sparse), 6)
        head = sparse[:-upsample_len] if len(sparse) > upsample_len else []
        tail = sparse[-upsample_len:] if len(sparse) > 0 else []
        tail_up = _upsample_segment(tail, max_step=min(0.03, grid_res))  # clamp step by grid_res

        new_segment = head + tail_up

        # Extend waypoints (avoid duplicating first point)
        if waypoints:
            waypoints.extend(new_segment[1:])
        else:
            waypoints.extend(new_segment)
        cur_xy = tgt
    return waypoints
