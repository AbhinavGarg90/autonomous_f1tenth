#!/usr/bin/env python

import rospy
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan

def scan_callback(msg):
    angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
    ranges = np.array(msg.ranges)
    print(len(ranges))
    print(len(msg.intensities))

    # Filter valid ranges
    valid = (ranges > msg.range_min) & (ranges < msg.range_max)
    ranges = ranges[valid]
    angles = angles[valid]

    # Convert polar to Cartesian
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)

    # Plot
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, s=2, c='blue')
    plt.title("LiDAR Snapshot")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

    rospy.signal_shutdown("Scan plotted, exiting.")

def main():
    rospy.init_node('lidar_snapshot_plot')
    rospy.Subscriber('/car_1/scan', LaserScan, scan_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
