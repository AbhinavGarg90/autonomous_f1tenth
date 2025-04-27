import rospy
from nav_msgs.msg import Odometry
from queue import Queue
import time
import numpy as np

class VESCMotorIntegrator:
    def __init__(self):
        self.vel_queue = Queue()
        self.last_time = None
        self.sub = rospy.Subscriber('/vesc/odom', Odometry, self.odom_callback)

    def odom_callback(self, msg):
        # Get x velocity from odometry
        vx = msg.twist.twist.linear.x

        # Get current time
        current_time = time.time()

        if self.last_time is not None:
            dt = current_time - self.last_time
            self.vel_queue.put((vx, dt))

        self.last_time = current_time

    def integrate_pose(self, prev_pose):
        """
        prev_pose: tuple (x, y, theta)
        returns: updated_pose (x_new, y_new, theta)
        """
        x = prev_pose[0]
        y = prev_pose[1]
        theta = prev_pose[2]

        while not self.vel_queue.empty():
            vx, dt = self.vel_queue.get()
            # Integrate based on current heading
            x += vx * dt * np.cos(theta)
            y += vx * dt * np.sin(theta)

        return [x, y, theta]

