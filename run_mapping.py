#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import rospy

from ICP import ICPLocalizer
from GridComp import OccupancyGridMapping
from lidar_polled import get_lidar_data
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from vicon_bridge import Vicon

# Global storage for incoming messages
latest_pose = None
latest_lidar = None

def pose_callback(msg):
    global latest_pose
    print(latest_pose)
    latest_pose = (
        msg.pose.position.x,
        msg.pose.position.y,
        2 * np.arctan2(msg.pose.orientation.z, msg.pose.orientation.w)  # extract yaw
    )

# def lidar_callback(msg):
#     global latest_lidar
#     data = np.array(msg.data, dtype=np.float32)
#     # print(data.size)
#     # if data.size % 2 == 0:
#     latest_lidar = data #data.reshape(-1, 2)
#     print(latest_lidar)

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
                        help="Filename inside saved_map/ for grid serialisation")
    args, _ = parser.parse_known_args()

    PKG_DIR  = Path(__file__).resolve().parent
    MAP_DIR  = PKG_DIR / "saved_map"; MAP_DIR.mkdir(exist_ok=True)
    CSV_DIR  = PKG_DIR / "waypoints";  CSV_DIR.mkdir(exist_ok=True)
    MAP_PATH = MAP_DIR / args.map_name

    global latest_pose

    rospy.init_node('run_mapping_node')

    rospy.Subscriber('/icp_estimated_pose', PoseStamped, pose_callback)
    # rospy.Subscriber('/raw_lidar_points', Float32MultiArray, lidar_callback)

    occupancy_node = OccupancyGridMapping() 
    height_gc, width_gc = occupancy_node.log_odds.shape
    grid = np.zeros((height_gc, width_gc), dtype=np.int8)

# rate = rospy.Rate(10)
    poses_list= []
    # gt_tracker = Vicon()
    # gt_origin = gt_tracker.get_pose()
    # print(latest_pose, latest_lidar)
    lidar_data, raw_data = get_lidar_data("/scan")
    # print(raw_data)
    hz_const = 10
    iterct = 0
    prev_time = time.time()
    try:
        rospy.loginfo("[run_auto] Mapping - drive the vehicle, Ctrl-C when done …")
        while not rospy.is_shutdown():
            if iterct % hz_const == 0:
                print(f'running at {hz_const/ (time.time() - prev_time)}')
                prev_time = time.time()
            lidar_data, raw_data = get_lidar_data("/scan")
            latest_pose = 1
            if latest_pose is not None and raw_data is not None:
                x, y, theta = latest_pose
                '''
                act_pose = gt_tracker.get_pose()
                act_pose[0] = act_pose[0] - gt_origin[0]
                act_pose[1] = act_pose[1] - gt_origin[1]
                act_pose[2] = act_pose[2] - gt_origin[2]
                x, y, theta = act_pose
                '''

                occupancy_node.update_map(x, y, theta, raw_data)
                poses_list.append([x, y, theta])
                # map = occupancy_node.get_probability_map()
                # im.set_data(map)

                row, col = occupancy_node.world_to_map(x, y)
                iterct += 1;
    except KeyboardInterrupt:
        print("wxltpg")
    rospy.loginfo("[run_auto] Mapping ended by user → saving grid …")
    print("saving map")
    prob_map = occupancy_node.get_probability_map()
    np.save(MAP_PATH, prob_map)
    poses_arr = np.asarray(poses_list, dtype=np.float32)
    poses_path = MAP_DIR / "pose_trace.npy"
    np.save(poses_path, poses_arr)
    rospy.loginfo(f"[run_auto] grid serialised → {MAP_PATH}")

    if args.map_only:
        rospy.loginfo("[run_auto] --map_only flag set ⇒ exiting before planning")
        return

if __name__ == '__main__':
    main()
