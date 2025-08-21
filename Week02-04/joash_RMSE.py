import numpy as np
import json

def compute_rmse(pred_file, truth_file):
    # Load predicted and ground truth positions
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)
    with open(truth_file, 'r') as f:
        truth_data = json.load(f)

    # Extract positions
    pred_taglist = pred_data['taglist']
    pred_map = pred_data['map']
    pred_positions = {tag: (pred_map[0][i], pred_map[1][i]) for i, tag in enumerate(pred_taglist)}
    truth_positions = {int(k.replace('aruco','').replace('_0','')): (v['x'], v['y']) for k, v in truth_data.items() if k.startswith('aruco')}

    # Match tags
    matched_tags = [tag for tag in pred_taglist if tag in truth_positions]
    if not matched_tags:
        print('No matching tags found.')
        return None
    pred_pts = np.array([pred_positions[tag] for tag in matched_tags]).T
    truth_pts = np.array([truth_positions[tag] for tag in matched_tags]).T

    # Compute per-point MSE and total MSE
    residual = pred_pts - truth_pts  # shape (2, N)
    mse_per_point = np.sum(residual**2, axis=0)  # sum of squared error for each point
    for i, tag in enumerate(matched_tags):
        print(f'Tag {tag}: MSE = {mse_per_point[i]:.6f}')
    total_mse = np.sum(mse_per_point)
    print(f'Total MSE (sum of all points): {total_mse:.6f}')
    rmse = np.sqrt(np.mean(mse_per_point))
    print(f'RMSE (root mean of all points): {rmse:.6f}')
    return mse_per_point, total_mse, rmse

# Example usage:
compute_rmse('lab_output/slam.txt', 'TrueMap.txt')
