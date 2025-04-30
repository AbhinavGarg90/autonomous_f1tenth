#!/usr/bin/env python3

import rospy
import numpy as np
from GridComp import OccupancyGridMapping
import matplotlib.pyplot as plt

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray

# Global storage for incoming messages
latest_pose = None
latest_lidar = None

def pose_callback(msg):
    global latest_pose
    latest_pose = (
        msg.pose.position.x,
        msg.pose.position.y,
        2 * np.arctan2(msg.pose.orientation.z, msg.pose.orientation.w)  # extract yaw
    )

def lidar_callback(msg):
    global latest_lidar
    data = np.array(msg.data, dtype=np.float32)
    if data.size % 2 == 0:
        latest_lidar = data.reshape(-1, 2)

def main():
    global latest_pose, latest_lidar

    rospy.init_node('run_mapping_node')

    rospy.Subscriber('/icp_estimated_pose', PoseStamped, pose_callback)
    rospy.Subscriber('/raw_lidar_points', Float32MultiArray, lidar_callback)

    occupancy_node = OccupancyGridMapping(origin_x_wc=0) 
    height_gc, width_gc = occupancy_node.log_odds.shape
    grid = np.zeros((height_gc, width_gc), dtype=np.int8)

    # Setup matplotlib plot
    fig, ax = plt.subplots()
    im = ax.imshow(grid, cmap='viridis', vmin=0, vmax=100, interpolation='none')

    robot_dot = ax.scatter([], [], marker='^', s=60, c='red')

    ax.set_xticks(np.arange(-0.5, width_gc, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height_gc, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.1)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_xticks([])
    ax.set_yticks([])

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if latest_pose is not None and latest_lidar is not None:
            x, y, theta = latest_pose
            occupancy_node.update_map(x, y, theta, latest_lidar)
            map = occupancy_node.get_probability_map()
            im.set_data(map)

            row, col = occupancy_node.world_to_map(x, y)
            robot_dot.set_offsets([[col, row]])

            plt.pause(0.01)

        rate.sleep()

if __name__ == '__main__':
    main()
