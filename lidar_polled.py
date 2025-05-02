#!/usr/bin/env python

import rospy
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan

def get_lidar_data(topic):
        msg = rospy.wait_for_message(topic, LaserScan)

        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)

        # Filter valid ranges
        valid = (ranges > msg.range_min) & (ranges < msg.range_max)
        ranges = ranges[valid]
        angles = angles[valid]
        
        min_angle = -110 * np.pi / 180
        max_angle = 110 * np.pi / 180
        valid = (angles > min_angle) & (angles < max_angle)
        ranges = ranges[valid]
        angles = angles[valid]
        # Convert polar to Cartesian
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        return np.array(list(zip(x, y))), (angles, ranges)


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
    sim = False
    lidar_topic = '/car_1/scan' if sim else 'scan'

    while not rospy.is_shutdown():
        ldata, msg = get_lidar_data(lidar_topic)
        x = [xi for xi, _ in ldata]
        y = [yi for _, yi in ldata]
        scatter.set_offsets(np.c_[x, y])
        ax.set_xlim(np.min(x)-1, np.max(x)+1)
        ax.set_ylim(np.min(y)-1, np.max(y)+1)
        fig.canvas.draw()
        fig.canvas.flush_events()

if __name__ == '__main__':
    main()

