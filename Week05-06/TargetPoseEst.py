import numpy as np
import json
import os
import ast
import cv2
from YOLO.detector import Detector
j
# list of target fruits and vegs types
TARGET_TYPES = ['orange', 'lemon', 'lime', 'tomato', 'capsicum', 'potato', 'pumpkin', 'garlic']


def estimate_pose(camera_matrix, obj_info, robot_pose):
    """
    Estimate the pose of a target object using bounding box size and robot pose.
    """
    focal_length = camera_matrix[0][0]

    # True object dimensions [w, d, h]
    target_dimensions_dict = {
        'orange': [0.084, 0.085, 0.077],
        'lemon': [0.074, 0.047, 0.05],
        'lime': [0.065, 0.06, 0.05],
        'tomato': [0.074, 0.074, 0.064],
        'capsicum': [0.079, 0.08, 0.09],
        'potato': [0.093, 0.073, 0.054],
        'pumpkin': [0.085, 0.082, 0.075],
        'garlic': [0.063, 0.06, 0.075]
    }

    # Parse detection info
    target_class = obj_info[0]
    x, y, w, h = obj_info[1]

    true_height = target_dimensions_dict[target_class][2]

    # Center of bounding box (in pixels)
    bbox_center_x = x + w / 2

    # Distance estimation
    distance = (true_height * focal_length) / h

    # Angle estimation
    image_width = 320
    x_shift = bbox_center_x - image_width / 2
    theta = np.arctan2(x_shift, focal_length)

    # Relative pose (robot frame)
    distance_obj = distance / np.cos(theta)
    x_relative = distance_obj * np.cos(theta)
    y_relative = distance_obj * np.sin(theta)

    # Transform into world frame
    robot_x, robot_y, robot_theta = robot_pose
    delta_x_world = x_relative * np.cos(robot_theta) - y_relative * np.sin(robot_theta)
    delta_y_world = x_relative * np.sin(robot_theta) + y_relative * np.cos(robot_theta)

    target_pose = {'x': robot_x + delta_x_world,
                   'y': robot_y + delta_y_world}

    return target_pose


def merge_estimations(target_pose_dict, dist_thresh=0.2):
    """
    Merge estimations of the same target using a distance threshold.
    """
    merged = {}

    for key, pose in target_pose_dict.items():
        obj_class = key.split("_")[0]

        if obj_class not in merged:
            merged[obj_class] = [pose]
        else:
            found_merge = False
            for existing_pose in merged[obj_class]:
                dist = np.sqrt((pose['x'] - existing_pose['x'])**2 +
                               (pose['y'] - existing_pose['y'])**2)
                if dist < dist_thresh:
                    # average positions
                    existing_pose['x'] = (existing_pose['x'] + pose['x']) / 2
                    existing_pose['y'] = (existing_pose['y'] + pose['y']) / 2
                    found_merge = True
                    break
            if not found_merge:
                merged[obj_class].append(pose)

    # Flatten results: keep up to 3 per class
    target_est = {}
    for obj_class, poses in merged.items():
        for i, pose in enumerate(poses[:3]):
            target_est[f"{obj_class}_{i}"] = pose

    return target_est


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Read camera matrix
    fileK = f'{script_dir}/calibration/param/intrinsic.txt'
    camera_matrix = np.loadtxt(fileK, delimiter=',')

    # Init YOLO model
    model_path = f'{script_dir}/YOLO/model/best.pt'
    yolo = Detector(model_path)

    # Load robot poses
    image_poses = {}
    with open(f'{script_dir}/lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']

    # Estimate poses
    target_pose_dict = {}
    detected_type_list = []
    for image_path in image_poses.keys():
        input_image = cv2.imread(image_path)
        bounding_boxes, bbox_img = yolo.detect_single_image(input_image)
        robot_pose = image_poses[image_path]

        for detection in bounding_boxes:
            occurrence = detected_type_list.count(detection[0])
            target_pose_dict[f'{detection[0]}_{occurrence}'] = estimate_pose(camera_matrix, detection, robot_pose)
            detected_type_list.append(detection[0])

    # Merge
    target_est = merge_estimations(target_pose_dict)

    # Save
    with open(f'{script_dir}/lab_output/targets.txt', 'w') as fo:
        json.dump(target_est, fo, indent=4)

    print("Target estimations saved!")
