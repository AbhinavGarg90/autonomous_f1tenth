import rospy
from pymavlink import mavutil
import sys, os
import numpy as np
from std_msgs.msg import Float64MultiArray

# Conversion to -pi to pi
def pi_2_pi(angle):
    if angle > np.pi:
        return angle - 2.0 * np.pi
    if angle < -np.pi:
        return angle + 2.0 * np.pi
    return angle

class Vicon:
    def __init__(self):
        self.data = Float64MultiArray()
        self.data.data = [0, ] * (9 + 2 + 4 + 1)

        self.data_path = Float64MultiArray()
        self.data_path.data = [0, ] * 4

    def get_pose(self):
        master = mavutil.mavlink_connection('udpin:0.0.0.0:10086')

        msg = master.recv_match(blocking=True)

        if not msg:
            return None

        if msg.get_type() == 'LOCAL_POSITION_NED_COV':
            self.data.data[0] = msg.x / 1000.
            self.data.data[1] = msg.y / 1000.
            self.data.data[2] = msg.z / 1000.
            self.data.data[3] = msg.vx / 1000.
            self.data.data[4] = msg.vy / 1000.
            self.data.data[5] = msg.vz / 1000.
            self.data.data[6] = msg.ax / 1000.
            self.data.data[7] = msg.ay / 1000.
            self.data.data[8] = msg.az / 1000.

            # use msg.covaricane to store the yaw and yaw_rate, and q
            offset = 100.
            self.data.data[9] = msg.covariance[0] - offset   # yaw
            self.data.data[10] = msg.covariance[1] - offset  # yaw_rate

            self.data.data[11] = msg.covariance[2] - offset
            self.data.data[12] = msg.covariance[3] - offset
            self.data.data[13] = msg.covariance[4] - offset
            self.data.data[14] = msg.covariance[5] - offset

            now = rospy.get_rostime()
            now = now.to_sec()
            self.data.data[-1] = now

            # print("X, Y, Yaw:", round(data.data[0],3), round(data.data[1],3), round(data.data[9],3))

            x_global = round(self.data.data[0], 3)
            y_global = round(self.data.data[1], 3)
            yaw_global = round(np.degrees(self.data.data[9]))

            print("X_global, Y_global, Yaw_global", x_global, y_global, yaw_global)

            # -------------------------------- For Global Origin Conversion --------------------------------
            # 0.545
            theta = np.pi/2 + 0.545  # body frame rotate w.r.t global frame
            x_p   = 0.43 # body origin move w.r.t. global frame along x
            y_p   = 0.54   # body origin move w.r.t. global frame along y

            # ----------------------------------------------------------------------------------------------

            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            x_new     = x_global * cos_theta + y_global * sin_theta - (x_p * cos_theta + y_p * sin_theta)
            y_new     = -x_global * sin_theta + y_global * cos_theta + (x_p * sin_theta - y_p * cos_theta)

            yaw_new = pi_2_pi(round(self.data.data[9]-3.686, 3))
            self.data_path.data[0] = x_new
            self.data_path.data[1] = y_new
            self.data_path.data[2] = yaw_new
            self.data_path.data[3] = round(np.degrees(yaw_new))
            print("X_new, Y_new, Yaw_new_deg:", x_new, y_new, np.degrees(yaw_new))
            print("\n")
            return x_new, y_new, yaw_new
        return None
