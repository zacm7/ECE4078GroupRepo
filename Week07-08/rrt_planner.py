import math
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class RRTPlanner:
    def __init__(self, bounds, obstacles, step_size=0.02, goal_sample_rate=0.2, max_iters=10000):
        self.xmin, self.xmax = bounds[0]
        self.ymin, self.ymax = bounds[1]
        self.obstacles = obstacles  # list of (x, y, radius)
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iters = max_iters

    class Node:
        def __init__(self, x, y, parent=None):
            self.x = x
            self.y = y
            self.parent = parent

    def plan(self, start, goal, goal_tol=0.2):
        nodes = [self.Node(start[0], start[1])]
        for i in range(self.max_iters):
            # Random sample (biased toward goal sometimes)
            if random.random() < self.goal_sample_rate:
                rnd = self.Node(goal[0], goal[1])
            else:
                rnd = self.Node(random.uniform(self.xmin, self.xmax),
                                random.uniform(self.ymin, self.ymax))

            # Nearest node
            nearest = min(nodes, key=lambda n: (n.x - rnd.x)**2 + (n.y - rnd.y)**2)

            # Step toward rnd
            theta = math.atan2(rnd.y - nearest.y, rnd.x - nearest.x)
            new_x = nearest.x + self.step_size * math.cos(theta)
            new_y = nearest.y + self.step_size * math.sin(theta)

            if not self._collision(new_x, new_y):
                new_node = self.Node(new_x, new_y, nearest)
                nodes.append(new_node)

                # Stop when within tolerance
                if math.hypot(new_x - goal[0], new_y - goal[1]) <= goal_tol:
                    path = []
                    node = new_node
                    while node:
                        path.append([node.x, node.y])
                        node = node.parent
                    return path[::-1]  # start → stop point
        return None

    def _collision(self, x, y):
        for ox, oy, r in self.obstacles:
            if math.hypot(x - ox, y - oy) <= r:
                return True
        return False


def make_obstacles_from_file(filename, radius=0.5):
    with open(filename, "r") as f:
        data = json.load(f)
    obstacles = [(values["x"], values["y"], radius) for values in data.values()]
    return obstacles


if __name__ == "__main__":
    bounds = ((-2, 5), (-2, 5))
    obstacles = make_obstacles_from_file("./M3_prac_map_full.txt", radius=0.05)

    start = [0.0, 0.0]
    goals = [[-0.9, 0.6], [-0.9, -0.1], [0.0, -0.4], [0.3, 0.4], [-0.3, 0.4]]

    rrt = RRTPlanner(bounds, obstacles, step_size=0.1, goal_sample_rate=0.2, max_iters=2000)
    current_start = start

    for idx, g in enumerate(goals):
        path = rrt.plan(current_start, g, goal_tol=0.2)

        fig, ax = plt.subplots()
        # Plot obstacles
        for ox, oy, r in obstacles:
            circle = plt.Circle((ox, oy), r, color='red', alpha=0.5)
            ax.add_patch(circle)

        # Plot start and goal
        ax.plot(current_start[0], current_start[1], 'go', label='Start')
        ax.plot(g[0], g[1], 'bo', label='Goal')
        # Draw goal tolerance circle
        goal_circle = plt.Circle((g[0], g[1]), 0.2, color='blue', alpha=0.2, linestyle='--')
        ax.add_patch(goal_circle)

        # Plot path
        if path:
            px, py = zip(*path)
            ax.plot(px, py, '-g', linewidth=2, label=f'Path to Goal {idx+1}')
            current_start = path[-1]  # stop point within 0.2m
        else:
            print(f"No path found to Goal {idx+1}: {g}")

        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_aspect('equal')
        ax.legend()
        plt.title(f'RRT Path to Goal {idx+1} (within 0.2m)')
        plt.show()
