import json
import matplotlib.pyplot as plt

def plot_slam_vs_ground_truth(slam_file, true_map_file, targets_file):
    """
    Plots SLAM predicted ArUco positions, ground truth ArUco positions,
    predicted fruit positions (targets.txt), and ground truth fruit positions (true_map.txt).
    Draws lines from predicted -> ground truth. Unmatched predictions are shown with crosses.
    """

    # Load files
    with open(slam_file, 'r') as f:
        slam_data = json.load(f)

    with open(true_map_file, 'r') as f:
        true_map_data = json.load(f)

    with open(targets_file, 'r') as f:
        targets_data = json.load(f)

    # Extract SLAM predicted ArUco positions
    taglist = slam_data['taglist']
    pred_map = slam_data['map']
    pred_positions = {
        f"aruco{tag}_0": (pred_map[0][idx], pred_map[1][idx])
        for idx, tag in enumerate(taglist)
    }

    # Split true map into ground truth ArUcos and fruits
    gt_positions = {k: (v['x'], v['y']) for k, v in true_map_data.items() if k.startswith("aruco")}
    gt_fruit_positions = {k: (v['x'], v['y']) for k, v in true_map_data.items() if not k.startswith("aruco")}

    # Extract predicted fruit positions (from targets.txt)
    pred_fruit_positions = {k: (v['x'], v['y']) for k, v in targets_data.items() if not k.startswith("aruco")}

    # Plotting
    plt.figure(figsize=(9, 9))

    # --- Ground truth ArUco markers ---
    if gt_positions:
        gt_x, gt_y = zip(*gt_positions.values())
        plt.scatter(gt_x, gt_y, c='g', label='Ground Truth (ArUco)', marker='o')
        for label, (x, y) in gt_positions.items():
            plt.text(x + 0.02, y + 0.02, label, fontsize=8, color='g')

    # --- SLAM predicted ArUco markers ---
    for label, (x, y) in pred_positions.items():
        if label in gt_positions:
            # draw line to GT
            gx, gy = gt_positions[label]
            plt.plot([x, gx], [y, gy], 'r--', linewidth=1)
            plt.scatter(x, y, c='r', marker='x')
        else:
            # unmatched prediction -> cross
            plt.scatter(x, y, c='red', marker='X', s=100)
        plt.text(x + 0.02, y + 0.02, label, fontsize=8, color='r')

    # --- Ground truth fruits ---
    if gt_fruit_positions:
        gt_fx, gt_fy = zip(*gt_fruit_positions.values())
        plt.scatter(gt_fx, gt_fy, c='orange', label='Ground Truth Fruits', marker='^')
        for label, (x, y) in gt_fruit_positions.items():
            plt.text(x + 0.02, y + 0.02, label, fontsize=8, color='orange')

    # --- Predicted fruits ---
    for label, (x, y) in pred_fruit_positions.items():
        if label in gt_fruit_positions:
            # draw line to GT
            gx, gy = gt_fruit_positions[label]
            plt.plot([x, gx], [y, gy], 'b--', linewidth=1)
            plt.scatter(x, y, c='b', marker='s')
        else:
            # unmatched prediction -> cross
            plt.scatter(x, y, c='blue', marker='X', s=100)
        plt.text(x + 0.02, y + 0.02, label, fontsize=8, color='b')

    # Formatting
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('SLAM Predicted vs Ground Truth (ArUco + Fruits)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Example call
plot_slam_vs_ground_truth('lab_output/slam.txt','true_map.txt','lab_output/targets.txt')
