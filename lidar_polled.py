#!/usr/bin/env python

import rospy
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan

def get_lidar_data():
        msg = rospy.wait_for_message('/car_1/scan', LaserScan)

        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)

        # Filter valid ranges
        valid = (ranges > msg.range_min) & (ranges < msg.range_max)
        ranges = ranges[valid]
        angles = angles[valid]

        # Convert polar to Cartesian
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        return x,y


def main():
    rospy.init_node('lidar_realtime_plot')
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(6, 6))

    # Initial empty plot
    scatter = ax.scatter([], [], s=2, c='blue')
    ax.set_title("LiDAR Live Plot")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.grid(True)

    while not rospy.is_shutdown():
        x, y = get_lidar_data()
        scatter.set_offsets(np.c_[x, y])
        ax.set_xlim(np.min(x)-1, np.max(x)+1)
        ax.set_ylim(np.min(y)-1, np.max(y)+1)
        fig.canvas.draw()
        fig.canvas.flush_events()

if __name__ == '__main__':
    main()

