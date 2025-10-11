"""Final demo Level 1 entry point: manual mapping + autonomous navigation."""

from __future__ import annotations

import argparse
import math
import os
from typing import List

import numpy as np
import pygame

from final_demo_common import (
    REPO_ROOT,
    ensure_operate_cwd,
    load_operate_module,
    configure_operate_module,
    prepare_display,
)

from map_utils import (
    build_targets_and_obstacles,
    load_search_list,
    print_target_fruits_pos,
    read_true_map_robust,
)
from astar_planning import plan_waypoints
from aruco_fruit_search_level2 import AutoOperatePlan


def parse_args() -> argparse.Namespace:
    week0204_calib = os.path.join(REPO_ROOT, "Week02-04", "calibration", "param") + os.sep
    week0708_dir = os.path.join(REPO_ROOT, "Week07-08")
    week0910_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Run Level 1: manual mapping with autonomous navigation"
    )
    parser.add_argument("--ip", type=str, default="192.168.50.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default=week0204_calib)
    parser.add_argument("--yolo_model", type=str,
                        default=os.path.join(week0708_dir, "YOLO", "model", "bestv5.pt"))
    parser.add_argument("--map", type=str,
                        default=os.path.join(week0910_dir, "ground_truth.txt"),
                        help="Ground-truth map JSON containing fruit and marker poses")
    parser.add_argument("--list", type=str,
                        default=os.path.join(week0910_dir, "shopping_list.txt"))
    parser.add_argument("--grid_res", type=float, default=0.02)
    parser.add_argument("--robot_radius", type=float, default=0.12)
    parser.add_argument("--safety_margin", type=float, default=0.10)
    parser.add_argument("--dist_tol", type=float, default=0.25,
                        help="Distance tolerance when declaring waypoint reached (m)")
    parser.add_argument("--angle_tol_deg", type=float, default=10.0,
                        help="Heading tolerance before engaging forward motion (deg)")
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--save_data", action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    operate_mod = load_operate_module()
    op_args = configure_operate_module(
        operate_mod,
        ip=args.ip,
        port=args.port,
        calib_dir=args.calib_dir,
        yolo_model=args.yolo_model,
        play_data=args.play_data,
        save_data=args.save_data,
    )

    pygame.init()
    canvas = prepare_display(caption='ECE4078 Final Demo - Level 1')

    fruit_list, fruit_pos, aruco_pos = read_true_map_robust(args.map)
    search_list = load_search_list(args.list)
    print_target_fruits_pos(search_list, fruit_list, fruit_pos)

    targets_xy, obstacles_xy = build_targets_and_obstacles(
        fruit_list, fruit_pos, aruco_pos, search_list
    )
    start_xy = [0.0, 0.0]
    waypoints: List[List[float]] = plan_waypoints(
        start_xy,
        targets_xy,
        obstacles_xy,
        grid_res=args.grid_res,
        robot_radius=args.robot_radius,
        safety_margin=args.safety_margin,
    )
    fruit_indices = list(range(len(targets_xy)))
    print(f"Planned {len(waypoints)} waypoints from manual map")

    ensure_operate_cwd()
    operate = AutoOperatePlan(op_args, waypoints, fruit_indices)
    operate.dist_tol = args.dist_tol
    operate.angle_tol = math.radians(args.angle_tol_deg)

    try:
        if hasattr(operate, 'ekf') and operate.ekf is not None:
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

        running = not getattr(operate, 'quit', False)

    pygame.quit()


if __name__ == "__main__":
    main()
