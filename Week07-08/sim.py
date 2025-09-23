# sim_rrt_drive.py
# Simulate 2D navigation with your RRTPlanner and a simple turn-then-drive controller.
# - Plans a path to each goal sequentially with RRT
# - Follows waypoints with limited angular/linear speeds
# - Animates on a Matplotlib plot (no hardware)

import math
import time
import json
import random
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

# ---- Bring in your RRT implementation ----------------------------------------
# If rrt_planner.py sits next to this file, the import below will work.
# Otherwise, paste your RRTPlanner/make_obstacles_from_file code here.
from rrt_planner import RRTPlanner, make_obstacles_from_file


# ----------------------------
# Geometry helpers
# ----------------------------
def angle_wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ----------------------------
# Simple point-robot controller
# ----------------------------
class PointRobotSim:
    """
    Very simple unicycle-ish controller:
      - Turn-in-place until heading error < angle_tol
      - Then drive forward toward the next waypoint
    Motion is simulated at fixed dt with max speeds.
    """

    def __init__(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        th0: float = 0.0,
        v_lin: float = 0.25,           # m/s
        v_ang: float = math.radians(90),  # rad/s
        angle_tol: float = math.radians(8),
        pos_tol: float = 0.05,         # m, waypoint acceptance
        dt: float = 0.02               # s
    ):
        self.x = x0
        self.y = y0
        self.th = th0
        self.v_lin = v_lin
        self.v_ang = v_ang
        self.angle_tol = angle_tol
        self.pos_tol = pos_tol
        self.dt = dt

        self.traj_x = [x0]
        self.traj_y = [y0]

    def step_to(self, waypoint: Tuple[float, float]):
        """
        Advance one control step toward a single waypoint.
        Returns True when the waypoint is reached (within pos_tol).
        """
        dx = waypoint[0] - self.x
        dy = waypoint[1] - self.y
        dist = math.hypot(dx, dy)
        if dist <= self.pos_tol:
            return True

        bearing = math.atan2(dy, dx)
        dheading = angle_wrap(bearing - self.th)

        # Turn-then-drive behavior
        if abs(dheading) > self.angle_tol:
            # rotate in place toward the waypoint
            dth = np.clip(dheading, -self.v_ang * self.dt, self.v_ang * self.dt)
            self.th = angle_wrap(self.th + dth)
        else:
            # drive forward
            step = min(self.v_lin * self.dt, dist)
            self.x += step * math.cos(self.th)
            self.y += step * math.sin(self.th)

        self.traj_x.append(self.x)
        self.traj_y.append(self.y)
        return False

    def pose(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.th


# ----------------------------
# Bounds utility
# ----------------------------
def infer_bounds(obstacles: List[Tuple[float, float, float]], goals: List[List[float]], pad: float = 0.2):
    xs = [g[0] for g in goals] + [o[0] for o in obstacles]
    ys = [g[1] for g in goals] + [o[1] for o in obstacles]
    if not xs:  # fallback
        return ((-1, 1), (-1, 1))
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad
    return (xmin, xmax), (ymin, ymax)


# ----------------------------
# RRT planning for a full route (multi-goal)
# ----------------------------
def plan_full_route_rrt(
    start_xy: Tuple[float, float],
    goals: List[List[float]],
    obstacles: List[Tuple[float, float, float]],
    step_size: float = 0.1,
    goal_sample_rate: float = 0.2,
    max_iters: int = 4000,
    goal_tol: float = 0.2,
):
    """
    Plan an RRT path to each goal in sequence; concatenate all legs.
    Returns: concatenated list of waypoints (including start and each leg's nodes).
    """
    # Infer bounds from obstacles + goals
    bounds = infer_bounds(obstacles, goals, pad=0.3)
    #print("BOUNDS:", bounds)
    #print("Obstacles",obstacles)

    full_path: List[Tuple[float, float]] = [tuple(start_xy)]
    current = tuple(start_xy)
    #print(current)
    

    for g in goals:
       # print(current,tuple(g),goal_tol)
        print("OBSTACLE",obstacles)
        planner = RRTPlanner(bounds, obstacles, step_size=step_size,
                             goal_sample_rate=goal_sample_rate, max_iters=max_iters)
        path = planner.plan(current, tuple(g), goal_tol=goal_tol)
        #print(path)
        if not path or len(path) < 2:
            print(f"[RRT] No path found to goal {g}. Aborting.")
            return None
        # Avoid duplicating the current point
        if path[0] == full_path[-1]:
            full_path.extend(path[1:])
        else:
            full_path.extend(path)
        current = tuple(g)

    return full_path


# ----------------------------
# Visualization / animation
# ----------------------------
def simulate_and_animate(
    start: Tuple[float, float],
    goals: List[List[float]],
    obstacles: List[Tuple[float, float, float]],
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)

    # Plan
    path = plan_full_route_rrt(
        start_xy=start,
        goals=goals,
        obstacles=obstacles,
        step_size=0.1,
        goal_sample_rate=0.2,
        max_iters=2000,
        goal_tol=0.2,
    )

    # Plot setup
    (xmin, xmax), (ymin, ymax) = infer_bounds(obstacles, goals, pad=0.3)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.grid(True, alpha=0.25)
    ax.set_title("2D RRT Navigation (Simulated)")

    # Obstacles
    for (ox, oy, r) in obstacles:
        ax.add_patch(patches.Circle((ox, oy), r, color="red", alpha=0.4, lw=0))

    # Start & goals
    ax.plot(start[0], start[1], "go", label="Start")
    for i, g in enumerate(goals):
        ax.plot(g[0], g[1], "bo")
        ax.add_patch(patches.Circle((g[0], g[1]), 0.2, ec="blue", fc="none", ls="--", alpha=0.6))
        ax.text(g[0] + 0.02, g[1] + 0.02, f"G{i+1}", color="blue")

    # Planned path (waypoints)
    if path:
        px, py = zip(*path)
        line_path, = ax.plot(px, py, "-", lw=2, label="RRT path")
    else:
        line_path, = ax.plot([], [], "-", lw=2, label="RRT path (none)")

    # Robot marker + trail
    robot_dot, = ax.plot([], [], "ko", ms=6, label="Robot")
    trail_line, = ax.plot([], [], "k--", lw=1, alpha=0.7, label="Trail")

    ax.legend(loc="upper right")

    # Simulate controller following waypoints
    sim = PointRobotSim(x0=start[0], y0=start[1], th0=0.0,
                        v_lin=0.25, v_ang=math.pi,  # fast-ish turn
                        angle_tol=math.radians(8), pos_tol=0.03, dt=0.02)

    if not path or len(path) < 2:
        print("No path to simulate.")
        plt.show()
        return

    wp_index = 1  # start from first waypoint after start
    trail_x, trail_y = [start[0]], [start[1]]

    # Real-time animation loop (press Ctrl+C to stop)
    try:
        while True:
            goal_wp = path[wp_index]
            done = sim.step_to(goal_wp)

            robot_dot.set_data([sim.x], [sim.y])
            trail_x.append(sim.x)
            trail_y.append(sim.y)
            trail_line.set_data(trail_x, trail_y)

            plt.pause(sim.dt)

            if done:
                wp_index += 1
                if wp_index >= len(path):
                    # Reached final goal
                    break
    except KeyboardInterrupt:
        pass

    print("Simulation finished.")
    plt.show()


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # Obstacles: either load from the same JSON map you use in your project,
    # or define a small manual set for a quick test.
    # Option A) Load from file (your existing helper):
    #   The file is expected to be like "M3_prac_map_full.txt" containing objects with x,y.
    #   Each is wrapped in a circle of given radius.
    try:
        obstacles = make_obstacles_from_file("./M3_prac_map_full.txt", radius=0.1)
    except Exception:
        # Option B) Fallback demo obstacles if the file isn't present
        print("using demo obstacles")
        obstacles = [
            (-0.6,  0.5, 0.08),
            (-0.2,  0.1, 0.08),
            ( 0.25, 0.2, 0.08),
            (-0.4, -0.4, 0.08),
            ( 0.10, -0.5, 0.08),
        ]
        print("[INFO] Using demo obstacles (file not found).")
    #print(obstacles)
    # Start & goals (your sequence)
    start = (0.0, 0.0)
    goals = [[-0.8927999999999999, -0.4608], [0.5088, 0.864], [-0.9887999999999999, 0.9792], [1.056, -1.008], [-1.0272, -1.0272]]

    simulate_and_animate(start, goals, obstacles, seed=123)
