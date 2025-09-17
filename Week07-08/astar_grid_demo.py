import numpy as np
import heapq
import json

# Grid parameters
GRID_SIZE = 20  # 20x20 grid for more resolution
GRID_X_MIN = -1.5
GRID_X_MAX = 1.5
GRID_Y_MIN = -1.5
GRID_Y_MAX = 1.5

def pos_to_grid(x, y):
    """Convert real-world coordinates to grid indices."""
    gx = int((x - GRID_X_MIN) / ((GRID_X_MAX-GRID_X_MIN)/GRID_SIZE))
    gy = int((y - GRID_Y_MIN) / ((GRID_Y_MAX-GRID_Y_MIN)/GRID_SIZE))
    return gx, gy

def print_grid(grid, path=None, goal=None):
    for y in range(GRID_SIZE):
        row = ''
        for x in range(GRID_SIZE):
            if (x, y) == (GRID_SIZE//2, GRID_SIZE//2):
                row += 'S '  # Start
            elif goal and (x, y) == goal:
                row += 'G '  # Goal
            elif path and (x, y) in path:
                row += '* '
            elif grid[x][y] == 1:
                row += '# '
            else:
                row += '. '
        print(row)
    print()

def astar(grid, start, goal):
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
            if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
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

def main():
    # Load map file
    with open('M3_prac_map_full.txt', 'r') as f:
        map_data = json.load(f)

    # Create empty grid
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    obstacles = []
    goal_positions = {}

    # Define goal fruit names
    goal_fruits = ['garlic_0', 'lemon_0', 'pear_0', 'tomato_0', 'pumpkin_0']

    # Mark obstacles and store goal positions
    for key, val in map_data.items():
        x, y = val['x'], val['y']
        gx, gy = pos_to_grid(x, y)
        if key.startswith('aruco'):
            obstacles.append((gx, gy))
        elif key in goal_fruits:
            goal_positions[key] = (gx, gy)
        else:
            obstacles.append((gx, gy))
    for ox, oy in obstacles:
        if 0 <= ox < GRID_SIZE and 0 <= oy < GRID_SIZE:
            grid[ox][oy] = 1

    # Start at center
    start = (GRID_SIZE//2, GRID_SIZE//2)
    remaining_goals = goal_positions.copy()
    current_pos = start
    visit_order = []

    print('Grid legend: S=Start, G=Goal, #=Obstacle, *=Path, .=Free')
    while remaining_goals:
        # Find nearest goal
        nearest_goal = None
        nearest_dist = None
        for fruit, pos in remaining_goals.items():
            dist = abs(current_pos[0] - pos[0]) + abs(current_pos[1] - pos[1])
            if nearest_goal is None or dist < nearest_dist:
                nearest_goal = fruit
                nearest_dist = dist
        goal = remaining_goals[nearest_goal]
        path = astar(grid, current_pos, goal)
        print(f'Next goal: {nearest_goal}')
        print_grid(grid, path=path, goal=goal)
        print(f'Path from {current_pos} to {goal}:')
        print(path)
        visit_order.append(nearest_goal)
        # Move to this goal
        current_pos = goal
        del remaining_goals[nearest_goal]
    print('Visit order:', visit_order)

if __name__ == '__main__':
    main()