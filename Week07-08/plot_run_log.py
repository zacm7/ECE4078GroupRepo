import json
import os
import sys
import matplotlib.pyplot as plt

# Usage: python plot_run_log.py [log_path]
# Defaults to Week05-06/lab_output/auto_nav_log.json when run from repo root or Week07-08.

def load_log(log_path):
    with open(log_path, 'r') as f:
        return json.load(f)


def plot_log(data):
    poses = data.get('poses', [])
    plans = data.get('plans', [])
    obstacles = data.get('obstacles', [])
    meta = data.get('meta', {})

    xs = [p[1] for p in poses]
    ys = [p[2] for p in poses]

    plt.figure(figsize=(7, 7))
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.title('Auto-Nav Run: Trajectory, Plans, and Discovered Obstacles')

    # Robot trajectory
    if xs and ys:
        plt.plot(xs, ys, 'k-', linewidth=1.5, label='Robot trajectory')

    # Planned waypoints (show a few snapshots)
    for i, pl in enumerate(plans):
        wps = pl.get('waypoints', [])
        if len(wps) >= 2:
            wpx = [w[0] for w in wps]
            wpy = [w[1] for w in wps]
            alpha = max(0.2, min(1.0, 0.2 + 0.8 * (i / max(1, len(plans)-1))))
            plt.plot(wpx, wpy, color=(0.2, 0.4, 1.0, alpha), linewidth=1, label='Planned path' if i == 0 else None)
            plt.scatter(wpx, wpy, s=8, color=(0.2, 0.4, 1.0, alpha))

    # Discovered obstacles
    if obstacles:
        ox = [o['x'] for o in obstacles]
        oy = [o['y'] for o in obstacles]
        plt.scatter(ox, oy, c='r', marker='x', s=40, label='Discovered obstacles')
        # Optionally annotate few
        for o in obstacles[:10]:
            plt.annotate(o.get('label','?'), (o['x'], o['y']), textcoords='offset points', xytext=(4,4), fontsize=8, color='r')

    # Static ArUco obstacles (from meta)
    aruco = meta.get('aruco_obstacles_xy', [])
    if aruco:
        ax = [a[0] for a in aruco]
        ay = [a[1] for a in aruco]
        plt.scatter(ax, ay, c='g', marker='s', s=25, label='ArUco obstacles (map)')

    # Targets (from meta)
    targets = meta.get('targets_xy', [])
    if targets:
        tx = [t[0] for t in targets]
        ty = [t[1] for t in targets]
        plt.scatter(tx, ty, c='orange', marker='o', s=35, label='Target order (map)')
        for idx, (txi, tyi) in enumerate(targets, start=1):
            plt.annotate(str(idx), (txi, tyi), textcoords='offset points', xytext=(4,-8), fontsize=8, color='orange')

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Figure out a reasonable default path
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        # Default where Level 3 writes logs (cwd switched to Week05-06 at runtime)
        # Try both relative to current working dir and repo layout
        candidates = [
            os.path.join('Week07-08', 'lab_output', 'auto_nav_log.json'),
            os.path.join('lab_output', 'auto_nav_log.json'),
            os.path.join('Week05-06', 'lab_output', 'auto_nav_log.json'),
        ]
        log_path = None
        for c in candidates:
            if os.path.exists(c):
                log_path = c
                break
        if log_path is None:
            print('Could not find auto_nav_log.json. Provide a path:')
            print('  python Week07-08/plot_run_log.py path/to/auto_nav_log.json')
            sys.exit(1)

    data = load_log(log_path)
    plot_log(data)
