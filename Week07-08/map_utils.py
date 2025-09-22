import os
import json
import numpy as np


def load_search_list(list_path: str):
    """Load shopping list (one fruit type per line)."""
    items = []
    with open(list_path, 'r') as fd:
        for line in fd:
            s = line.strip()
            if s:
                items.append(s)
    return items


def read_true_map_robust(map_path: str):
    """
    Parse GT map JSON and return:
      - fruit_list: list[str] like ['lemon','tomato',...]
      - fruit_true_pos: Nx2 np.array
      - aruco_true_pos: 10x2 np.array, index 0->aruco1_*, 9->aruco10_*
    Keys are like 'lemon_0', 'aruco3_0'.
    """
    with open(map_path, 'r') as fd:
        gt = json.load(fd)

    fruit_list = []
    fruit_true_pos = []
    aruco_true_pos = np.empty((10, 2), dtype=float)

    for key, v in gt.items():
        # Use full precision from the map (no rounding)
        x = float(v['x'])
        y = float(v['y'])
        if key.startswith('aruco'):
            # aruco10_* -> index 9
            if key.startswith('aruco10'):
                aruco_true_pos[9, 0] = x
                aruco_true_pos[9, 1] = y
            else:
                # aruco1_* ... aruco9_*
                idx = int(key[5]) - 1
                aruco_true_pos[idx, 0] = x
                aruco_true_pos[idx, 1] = y
        else:
            # fruit keys like 'lemon_0' -> 'lemon'
            fruit_type = key.rsplit('_', 1)[0]
            fruit_list.append(fruit_type)
            fruit_true_pos.append([x, y])

    fruit_true_pos = np.array(fruit_true_pos, dtype=float) if fruit_true_pos else np.zeros((0, 2), dtype=float)
    return fruit_list, fruit_true_pos, aruco_true_pos


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print 5 targets in order with coordinates."""
    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(len(fruit_list)):
            if fruit == fruit_list[i]:
                print(f"{n_fruit}) {fruit} at [{fruit_true_pos[i][0]}, {fruit_true_pos[i][1]}]")
        n_fruit += 1


def build_targets_and_obstacles(fruit_list, fruit_true_pos, aruco_true_pos, search_list):
    """Build target and obstacle coordinate lists from map + shopping list.

    Targets in order = first instance of each type in search_list.
    Obstacles = all fruits not in search_list + all ArUco markers.
    """
    targets = []
    used_idx = set()
    for ft in search_list:
        found = False
        for i, name in enumerate(fruit_list):
            if i in used_idx:
                continue
            if name == ft:
                targets.append([float(fruit_true_pos[i, 0]), float(fruit_true_pos[i, 1])])
                used_idx.add(i)
                found = True
                break
        if not found:
            raise ValueError(f"Target '{ft}' not found in map")

    obstacles = []
    for i, name in enumerate(fruit_list):
        if name not in search_list:
            obstacles.append([float(fruit_true_pos[i, 0]), float(fruit_true_pos[i, 1])])
    for k in range(aruco_true_pos.shape[0]):
        obstacles.append([float(aruco_true_pos[k, 0]), float(aruco_true_pos[k, 1])])
    return targets, obstacles
