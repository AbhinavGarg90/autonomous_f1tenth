import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map_name",
        default="occ_grid.npy",
        help="Filename inside saved_maps/ to load (default: occ_grid.npy)",
    )
    args = parser.parse_args()

    pkg_dir = Path(__file__).resolve().parent
    map_path = pkg_dir / "saved_maps" / args.map_name

    if not map_path.exists():
        print(f"[viewer] File not found: {map_path}")
        return

    prob_grid = np.load(map_path)  
    print(f"[viewer] Loaded grid {prob_grid.shape} from {map_path}")

    plt.figure("Saved occupancy grid")
    plt.imshow(prob_grid, cmap="viridis", vmin=0, vmax=100, origin="upper",interpolation="none",)
    plt.show()


if __name__ == "__main__":
    main()