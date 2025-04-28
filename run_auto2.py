import argparse
import csv
import os
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import rospy
from live_plotter import LivePlotter

from ICP import ICPLocalizer
from GridComp import OccupancyGridMapping
from lidar_polled import get_lidar_data
# from waypoint_planner import plan_path_to_goal_region

def save_waypoints_csv(waypoints, csv_path):
    """Save (x, y, θ) list → CSV (*θ in degrees*) for vicon_planner."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        for x, y, theta in waypoints:
            writer.writerow([x, y, np.degrees(theta)])
    rospy.loginfo(f"[run_auto] Wrote {len(waypoints)} way‑points → {csv_path}")


def prob_to_occ(prob_grid):
    """Convert 0-100 int8 probability grid → planner occupancy grid."""
    occ = prob_grid.astype(np.int16)
    occ[occ == 50] = -1  # treat unknown as occupied
    return occ

def visualise_grid(prob_grid, title="Occupancy grid"):
    """Display a probability grid (0-100 int8) with Matplotlib."""
    plt.figure(title)
    plt.imshow(prob_grid, cmap="viridis", vmin=0, vmax=100, origin="upper")
    plt.title(title)
    plt.axis("off")
    plt.show()


def main():
    # for CLI args ---------------------------------------------------------------
    # to skip mapping and plan immediately from the current map or to save the waypoints or to skip planning 
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_mapping", action="store_true",
                        help="Use last saved map; skip SLAM")
    parser.add_argument("--map_only", action="store_true",
                        help="Run SLAM, save + display map, **no planning**")
    parser.add_argument("--show_saved_map", action="store_true",
                        help="Display saved map then exit (no SLAM, no planning)")
    parser.add_argument("--csv_name", default="xyhead_demo_pp.csv",
                        help="Way-point CSV inside waypoints/ (default)")
    parser.add_argument("--map_name", default="occ_grid.npy",
                        help="Filename inside saved_maps/ for grid serialisation")
    args, _ = parser.parse_known_args()

    # Directories -------------------------------------------------------------
    PKG_DIR  = Path(__file__).resolve().parent
    MAP_DIR  = PKG_DIR / "saved_maps"; MAP_DIR.mkdir(exist_ok=True)
    CSV_DIR  = PKG_DIR / "waypoints";  CSV_DIR.mkdir(exist_ok=True)
    MAP_PATH = MAP_DIR / args.map_name

    # ROS init ----------------------------------------------------------------
    rospy.init_node("icp_runner", anonymous=True)
    sim = True  # flip to False on real car
    lidar_topic = "/car_1/scan" if sim else "scan"

    # ── SLAM components -------------------------------------------------------
    occ_node = OccupancyGridMapping(origin_x_wc=0)
    lidar_data, raw_data = get_lidar_data(lidar_topic)
    
    icp = ICPLocalizer()
    icp.initialize(lidar_data)

    # Ground‑truth tracker ----------------------------------------------------
    if sim:
        from model_pos import CarPoseTracker
        gt_tracker = CarPoseTracker()
    # else:
    #     from vicon_bridge import Vicon
    #     gt_tracker = Vicon()


    # ─────────────────── Either build a map or load one ─────────────────────
    if args.skip_mapping:
        if not MAP_PATH.exists():
            rospy.logerr(f"[run_auto] --skip_mapping requested but {MAP_PATH} not found")
            sys.exit(1)
        rospy.loginfo(f"[run_auto] loading grid from {MAP_PATH}")
        prob_map = np.load(MAP_PATH)
    
    else:
        # ── SLAM initialisation ---------------------------------------------
        lidar_data, raw_data = get_lidar_data(lidar_topic)
        icp.initialize(lidar_data)

        # Live plot -----------------------------------------------------------
        h, w = occ_node.log_odds.shape
        fig, ax = plt.subplots()
        im = ax.imshow(np.zeros((h, w), dtype=np.int8), cmap="viridis",
                       vmin=0, vmax=100, interpolation="none")
        robot_dot = ax.scatter([], [], marker="^", s=60, c="red")
        ax.set_xticks([]); ax.set_yticks([])

        rospy.loginfo("[run_auto] Mapping - drive the vehicle, Ctrl-C when done …")
        try:
            rate = rospy.Rate(30)
            while not rospy.is_shutdown():
                scan, raw = get_lidar_data(lidar_topic)
                est_pose = icp.update(scan)
                act_pose = gt_tracker.get_pose() 
                used_pose = act_pose
                occ_node.update_map(used_pose[0], used_pose[1], used_pose[2], raw_data)

                prob = occ_node.get_probability_map()
                im.set_data(prob)
                r, c = occ_node.world_to_map(used_pose[0], used_pose[1])
                robot_dot.set_offsets([[c, r]])
                plt.pause(0.001)

        except KeyboardInterrupt:
            rospy.loginfo("[run_auto] Mapping ended by user → saving grid …")

        prob_map = occ_node.get_probability_map()
        np.save(MAP_PATH, prob_map)
        rospy.loginfo(f"[run_auto] grid serialised → {MAP_PATH}")

        if args.map_only:
            rospy.loginfo("[run_auto] --map_only flag set ⇒ exiting before planning")
            return

    # ─────────────────────── Planning stage ────────────────────────────────
    # resolution = occ_node.resolution
    # origin_x = -occ_node.origin_x_wc
    # origin_y = -occ_node.origin_y_wc

    # start_pose = (origin_x, origin_y, 0.0)  

    # visualise_grid(prob_map, title="Final map after SLAM, before planning")

    # occ_map = prob_to_occ(prob_map)   # prob_map from line 154
    # rospy.loginfo("[run_auto] Calling Hybrid A★ …")
    # waypoints = plan_path_to_goal_region(start_pose, occ_map, resolution,
    #                                      origin_x, origin_y)
    # if waypoints is None:
    #     rospy.logerr("[run_auto] Planner failed — exiting")
    #     sys.exit(1)

    # csv_path = CSV_DIR / args.csv_name
    # save_waypoints_csv(waypoints, csv_path)
    # rospy.loginfo("[run_auto] Done — launch vicon_planner.py to follow the path")


if __name__ == "__main__":
    main()