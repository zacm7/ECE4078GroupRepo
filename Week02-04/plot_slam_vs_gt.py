import json
import matplotlib.pyplot as plt

def plot_slam_vs_ground_truth(slam_file, gt_file):
    """
    Plots SLAM predicted block positions against ground truth positions.

    Parameters:
        slam_file (str): Path to the slam.txt file (predicted positions).
        gt_file (str): Path to the test_output.txt file (ground truth positions).
    """
    # Load files
    with open(slam_file, 'r') as f:
        slam_data = json.load(f)

    with open(gt_file, 'r') as f:
        gt_data = json.load(f)

    # Extract predicted positions
    taglist = slam_data['taglist']
    pred_map = slam_data['map']
    pred_positions = {
        f"aruco{tag}_0": (pred_map[0][idx], pred_map[1][idx])
        for idx, tag in enumerate(taglist)
    }

    # Extract ground truth positions
    gt_positions = {k: (v['x'], v['y']) for k, v in gt_data.items() if k.startswith("aruco")}

    # Plotting
    plt.figure(figsize=(8, 8))

    # Ground truth
    gt_x, gt_y = zip(*gt_positions.values())
    plt.scatter(gt_x, gt_y, c='g', label='Ground Truth', marker='o')
    for label, (x, y) in gt_positions.items():
        plt.text(x + 0.02, y + 0.02, label, fontsize=8, color='g')

    # Predictions
    pred_x, pred_y = zip(*pred_positions.values())
    plt.scatter(pred_x, pred_y, c='r', label='Predicted', marker='x')
    for label, (x, y) in pred_positions.items():
        plt.text(x + 0.02, y + 0.02, label, fontsize=8, color='r')

    # Formatting
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('SLAM Predicted Map vs Ground Truth Map')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

plot_slam_vs_ground_truth('lab_output/slam.txt', 'TrueMap.txt')