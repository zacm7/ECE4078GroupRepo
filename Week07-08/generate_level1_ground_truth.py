"""Generate a Level 1 ground-truth map from Week05-06 SLAM outputs.

This script reads the `slam.txt` and `targets.txt` files produced by
Week05-06's `operate.py` run (saved under `lab_output/`) and merges
their contents into a single JSON map compatible with the M3 practice
maps (e.g. `M3_prac_map_full.txt`). The resulting file contains both
ArUco marker poses and fruit locations.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any


DEFAULT_WEEK0506_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Week05-06")
)
DEFAULT_SLAM_FILE = os.path.join(DEFAULT_WEEK0506_DIR, "lab_output", "slam.txt")
DEFAULT_TARGETS_FILE = os.path.join(DEFAULT_WEEK0506_DIR, "lab_output", "targets.txt")
DEFAULT_OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__), "level1_ground_truth.txt"
)


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="ascii") as f:
        text = f.read().strip()
        if not text:
            raise ValueError(f"File is empty: {path}")
        return json.loads(text)


def build_ground_truth(
    slam_json: Dict[str, Any],
    targets_json: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    ground_truth: Dict[str, Dict[str, float]] = {}

    taglist = slam_json.get("taglist", [])
    map_coords = slam_json.get("map", [])

    if not isinstance(taglist, list) or len(map_coords) != 2:
        raise ValueError("Unexpected SLAM file structure: missing tag list or map arrays")

    x_coords, y_coords = map_coords
    if len(x_coords) != len(taglist) or len(y_coords) != len(taglist):
        raise ValueError("SLAM map coordinate length mismatch with tag list")

    for idx, tag_id in enumerate(taglist):
        aruco_key = f"aruco{tag_id}_0"
        ground_truth[aruco_key] = {
            "y": float(y_coords[idx]),
            "x": float(x_coords[idx]),
        }

    for label, pose in targets_json.items():
        if not isinstance(pose, dict):
            continue
        try:
            x_val = float(pose["x"])
            y_val = float(pose["y"])
        except (KeyError, TypeError, ValueError):
            continue
        ground_truth[label] = {"y": y_val, "x": x_val}

    return ground_truth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge SLAM and target detections into a Level 1 ground-truth file."
    )
    parser.add_argument("--slam", type=str, default=DEFAULT_SLAM_FILE,
                        help="Path to Week05-06 lab_output/slam.txt")
    parser.add_argument("--targets", type=str, default=DEFAULT_TARGETS_FILE,
                        help="Path to Week05-06 lab_output/targets.txt")
    parser.add_argument("--out", type=str, default=DEFAULT_OUTPUT_FILE,
                        help="Destination Level 1 ground-truth JSON file")
    parser.add_argument("--indent", type=int, default=None,
                        help="Optional JSON indent for readability")
    parser.add_argument("--sort-keys", action="store_true",
                        help="Sort top-level keys alphabetically in the output JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    slam_json = load_json(args.slam)
    targets_json = load_json(args.targets)

    ground_truth = build_ground_truth(slam_json, targets_json)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="ascii") as f:
        json.dump(ground_truth, f, indent=args.indent, sort_keys=args.sort_keys)
        if args.indent is None:
            f.write("\n")

    print(f"Wrote {len(ground_truth)} entries to {args.out}")


if __name__ == "__main__":
    main()
