import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def world_to_pixel(x, y, origin_x, origin_y, resolution):
    """
    Convert world metres → image row/col indices.
    Matches GridComp.OccupancyGridMapping.world_to_map().
    """
    col = (x + origin_x) / resolution
    row = (y + origin_y) / resolution
    return row, col


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map_name",
        default="occ_grid.npy",
        help="Grid filename inside saved_map/ (default: occ_grid.npy)",
    )
    parser.add_argument(
        "--pose_name",
        default="pose_trace.npy",
        help="Pose trace filename inside saved_map/ (default: pose_trace.npy)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.10,
        help="Grid resolution in metres per cell (default 0.10)",
    )
    parser.add_argument(
        "--origin_x",
        type=float,
        default=30.0,
        help="Map origin_x_wc used when logging (default 0.0)",
    )
    parser.add_argument(
        "--origin_y",
        type=float,
        default=30.0,
        help="Map origin_y_wc used when logging (default 30.0)",
    )
    args = parser.parse_args()

    pkg_dir = Path(__file__).resolve().parent
    map_path = pkg_dir / "saved_map" / args.map_name
    pose_path = pkg_dir / "saved_map" / args.pose_name

    if not map_path.exists():
        print(f"[viewer] grid file not found: {map_path}")
        return
    prob_grid = np.load(map_path)
    print(f"[viewer] Loaded grid {prob_grid.shape} from {map_path}")

    if pose_path.exists():
        poses = np.load(pose_path)  # shape (N,3)
        print(f"[viewer] Loaded {poses.shape[0]} poses from {pose_path}")
    else:
        poses = None
        print(f"[viewer] Pose file not found: {pose_path}  (plotting grid only)")

    # ----------------------------------------------------------------- figure
    plt.figure("Occupancy grid + path")
    plt.imshow(
        prob_grid,
        cmap="gray",
        vmin=0,
        vmax=100,
        origin="upper",
        interpolation="none",
    )
    plt.title("Saved occupancy grid (+ path)")
    plt.axis("off")
    cbar = plt.colorbar(label="P(occupied) × 100")

    # overlay path ------------------------------------------------------------
    if poses is not None:
        rows, cols = world_to_pixel(
            poses[:, 0],
            poses[:, 1],
            origin_x=args.origin_x,
            origin_y=args.origin_y,
            resolution=args.resolution,
        )
        plt.plot(cols, rows, c="white", lw=1.0, label="trajectory")
        plt.scatter(cols[0], rows[0], c="lime", s=40, label="start")
        plt.scatter(cols[-1], rows[-1], c="red", s=40, label="end")
        plt.legend(loc="lower right")

    plt.show()


if __name__ == "__main__":
    main()
