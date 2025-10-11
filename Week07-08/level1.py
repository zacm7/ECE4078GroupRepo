"""Level 1 autonomous navigation using full-mapped ground truth.

This entry point mirrors the structure of `wednesday.py` but assumes that
all fruits and ArUco markers are already known (e.g. after running
Week05-06 `operate.py` and converting `slam.txt`/`targets.txt` into
`level1_ground_truth.txt`). The robot plans once from the full map and
does not add new obstacles during runtime.
"""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace
from typing import List

import numpy as np
import pygame

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
WEEK0506_DIR = os.path.join(REPO_ROOT, "Week05-06")

# Reuse the advanced autonomous controller from wednesday.py
import wednesday as wednesday_mod  # type: ignore
from wednesday import AutoOperateDynamic  # type: ignore

from map_utils import (
    read_true_map_robust,
    load_search_list,
    print_target_fruits_pos,
    build_targets_and_obstacles,
)


class AutoOperateLevel1(AutoOperateDynamic):
    """Ground-truth navigation: skips dynamic obstacle discovery."""

    def __init__(
        self,
        args: SimpleNamespace,
        search_list: List[str],
        targets_xy: List[List[float]],
        known_obstacles_xy: List[List[float]],
        grid_res: float,
        robot_radius: float,
        safety_margin: float,
        map_fruit_labels: List[str],
        map_fruit_xy: List[List[float]],
        full_ground_truth: dict | None = None,
    ):
        super().__init__(
            args,
            search_list,
            targets_xy,
            known_obstacles_xy,
            grid_res=grid_res,
            robot_radius=robot_radius,
            safety_margin=safety_margin,
            merge_threshold=0.50,
            obs_max_range=0.40,
            map_fruit_labels=map_fruit_labels,
            map_fruit_xy=map_fruit_xy,
        )
        self.full_ground_truth = full_ground_truth or {}

    def periodic_perception_update(self):
        # For Level 1 the full environment is already known, so skip
        # adding obstacles from detector outputs.
        return


def parse_args() -> argparse.Namespace:
    default_map = os.path.join(SCRIPT_DIR, "level1_ground_truth.txt")
    default_list = os.path.join(SCRIPT_DIR, "shopping_list.txt")
    default_yolo = os.path.join(SCRIPT_DIR, "YOLO", "model", "bestv5.pt")
    default_calib = os.path.join(REPO_ROOT, "Week02-04", "calibration", "param") + os.sep

    parser = argparse.ArgumentParser(
        description="Level 1 autonomous navigation with full ground-truth map."
    )
    parser.add_argument("--ip", type=str, default="192.168.50.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default=default_calib)
    parser.add_argument("--yolo_model", type=str, default=default_yolo)
    parser.add_argument("--map", type=str, default=default_map,
                        help="Ground-truth JSON generated from SLAM + targets.")
    parser.add_argument("--list", type=str, default=default_list,
                        help="Shopping list of target fruits (one per line).")
    parser.add_argument("--grid_res", type=float, default=0.02)
    parser.add_argument("--robot_radius", type=float, default=0.14)
    parser.add_argument("--safety_margin", type=float, default=0.18)
    parser.add_argument("--dist_tol", type=float, default=0.15,
                        help="Distance tolerance when declaring waypoint reached (m).")
    parser.add_argument("--angle_tol_deg", type=float, default=10.0,
                        help="Heading tolerance before engaging forward motion (deg).")
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--save_data", action='store_true')
    return parser.parse_args()


def prepare_operate(args: argparse.Namespace) -> SimpleNamespace:
    op_args = SimpleNamespace(
        ip=args.ip,
        port=args.port,
        calib_dir=args.calib_dir,
        yolo_model=args.yolo_model,
        play_data=args.play_data,
        save_data=args.save_data,
    )
    # Provide globals expected by Week05-06/operate.py
    wednesday_mod.operate_mod.args = op_args  # type: ignore[attr-defined]

    # Fonts/icons for GUI
    pygame.font.init()
    try:
        title_font = pygame.font.Font(
            os.path.join(WEEK0506_DIR, "pics", "8-BitMadness.ttf"), 35
        )
        text_font = pygame.font.Font(
            os.path.join(WEEK0506_DIR, "pics", "8-BitMadness.ttf"), 40
        )
    except Exception:
        title_font = pygame.font.SysFont(None, 28)
        text_font = pygame.font.SysFont(None, 32)
    wednesday_mod.operate_mod.TITLE_FONT = title_font  # type: ignore[attr-defined]
    wednesday_mod.operate_mod.TEXT_FONT = text_font  # type: ignore[attr-defined]

    return op_args


def main() -> None:
    args = parse_args()
    op_args = prepare_operate(args)

    pygame.init()
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption("ECE4078 - Level 1 Ground Truth Autonomy")
    try:
        pygame.display.set_icon(
            pygame.image.load(os.path.join(WEEK0506_DIR, "pics", "8bit", "pibot5.png"))
        )
    except Exception:
        pass
    canvas.fill((0, 0, 0))

    with open(args.map, "r", encoding="ascii") as fd:
        ground_truth_dict = json.load(fd)

    fruit_list, fruit_pos, aruco_pos = read_true_map_robust(args.map)
    search_list = load_search_list(args.list)
    print_target_fruits_pos(search_list, fruit_list, fruit_pos)

    targets_xy, obstacles_xy = build_targets_and_obstacles(
        fruit_list,
        fruit_pos,
        aruco_pos,
        search_list,
    )

    # Ensure Week05-06 relative asset paths resolve for operate.py
    prev_cwd = os.getcwd()
    os.chdir(WEEK0506_DIR)

    operate = AutoOperateLevel1(
        op_args,
        search_list=search_list,
        targets_xy=targets_xy,
        known_obstacles_xy=obstacles_xy,
        grid_res=args.grid_res,
        robot_radius=args.robot_radius,
        safety_margin=args.safety_margin,
        map_fruit_labels=list(fruit_list),
        map_fruit_xy=fruit_pos.tolist() if fruit_pos.size else [],
        full_ground_truth=ground_truth_dict,
    )
    operate.dist_tol = args.dist_tol
    operate.angle_tol = np.radians(args.angle_tol_deg)

    try:
        if hasattr(operate, "ekf") and operate.ekf is not None:
            operate.ekf.fixed_aruco_pos = np.array(aruco_pos, dtype=float)
            operate.ekf.lock_aruco = True
    except Exception:
        pass

    running = True
    while running:
        operate.update_keyboard()
        operate.take_pic()
        operate.auto_nav_step()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        operate.detect_target()
        operate.draw(canvas)
        pygame.display.update()
        running = not getattr(operate, "quit", False)

    pygame.quit()
    os.chdir(prev_cwd)


if __name__ == "__main__":
    main()
