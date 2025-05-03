#!/usr/bin/env python3

#================================================================
# File name: gem_gnss_pp_tracker_pid.py                                                                  
# Description: gnss waypoints tracker using pid and pure pursuit                                                                
# Author: Hang Cui
# Email: hangcui3@illinois.edu                                                                     
# Date created: 08/02/2021                                                                
# Date last modified: 08/15/2022                                                          
# Version: 1.0                                                                   
# Usage: rosrun gem_gnss gem_gnss_pp_tracker.py                                                                      
# Python version: 3.8                                                             
#================================================================

from __future__ import print_function

# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal

# ROS Headers
import rospy

# GEM Sensor Headers
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Bool, Float32, Float64, Float64MultiArray
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


class PurePursuit(object):
    
    def __init__(self):
        
        # 0.5 - 0.1 - 0.41

        self.look_ahead = 0.3 # 4
        self.wheelbase  = 0.325 # meters
        self.offset     = 0.15 # meters        
        
        self.ctrl_pub  = rospy.Publisher("/vesc/low_level/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=1)
        self.rate = rospy.Rate(50)
        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.header.frame_id = "f1tenth_control"
        self.drive_msg.drive.speed     = 1.3 # m/s, reference speed

        self.vicon_sub = rospy.Subscriber('/icp_estimated_pose', PoseStamped,  self.pose_callback )
        self.x   = 0
        self.y   = 0
        self.yaw = 0
        
        # read waypoints into the system 
        self.goal = 0            
        self.read_waypoints() 
        
    def pose_callback(self,msg):    
        self.x = msg.pose.position.x,
        self.y = msg.pose.position.y,
        self.yaw = 2 * np.arctan2(msg.pose.orientation.z, msg.pose.orientation.w)  # extract yaw
        # print(self.x, self.y, self.yaw)
            

    def read_waypoints(self):
        # read recorded GPS lat, lon, heading
        dirname  = "waypoints_in_csv"
        filename = os.path.join(dirname, "waypoints_world.csv")  #jay is sure the slash is needed 
        print(filename)
        with open(filename) as f:
            path_points = [tuple(line) for line in csv.reader(f)]
            print(path_points)
        # x towards East and y towards North
        self.path_points_x_record   = [float(point[0]) for point in path_points] # x
        self.path_points_y_record   = [float(point[1]) for point in path_points] # y
        self.path_points_yaw_record = [float(point[2]) for point in path_points] # yaw
        self.wp_size                = len(self.path_points_x_record)
        self.dist_arr               = np.zeros(self.wp_size)

    def get_f1tenth_state(self):

        # heading to yaw (degrees to radians)
        # heading is calculated from two GNSS antennas
        curr_yaw = np.radians(self.yaw)

        # reference point is located at the center of rear axle
        curr_x = self.x - self.offset * np.cos(curr_yaw)
        curr_y = self.y - self.offset * np.sin(curr_yaw)
        print("Curr x: ", curr_x, " Curr y: ", curr_y)

        return np.round(curr_x, 3), np.round(curr_y, 3), np.round(curr_yaw, 4)

    # find the angle bewtween two vectors    
    def find_angle(self, v1, v2):
        cosang = np.dot(v1, v2)
        sinang = la.norm(np.cross(v1, v2))
        # [-pi, pi]
        return np.arctan2(sinang, cosang)

    # computes the Euclidean distance between two 2D points
    def dist(self, p1, p2):
        return np.round(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 3)

    def start_pp(self):
        
        while not rospy.is_shutdown():

            self.path_points_x = np.array(self.path_points_x_record)
            self.path_points_y = np.array(self.path_points_y_record)

            curr_x, curr_y, curr_yaw = self.get_f1tenth_state()

            # finding the distance of each way point from the current position
            for i in range(len(self.path_points_x)):
                self.dist_arr[i] = self.dist((self.path_points_x[i], self.path_points_y[i]), (curr_x, curr_y))

            # finding those points which are less than the look ahead distance (will be behind and ahead of the vehicle)
            goal_arr = np.where((self.dist_arr < self.look_ahead + 0.05) & (self.dist_arr > self.look_ahead - 0.05))[0]

            # finding the goal point which is the last in the set of points less than the lookahead distance
            for idx in goal_arr:
                v1 = [float(self.path_points_x[idx])-curr_x , float(self.path_points_y[idx])-curr_y]
                v1 = np.concatenate(v1)
                v2 = [np.cos(curr_yaw), np.sin(curr_yaw)]
                temp_angle = self.find_angle(v1,v2)
                # find correct look-ahead point by using heading information
                if abs(temp_angle) < np.pi/2:
                    self.goal = idx
                    break

            # finding the distance between the goal point and the vehicle
            # true look-ahead distance between a waypoint and current position
            L = self.dist_arr[self.goal]

            # find the curvature and the angle 
            alpha = np.radians(self.path_points_yaw_record[self.goal]) - curr_yaw

            # ----------------- tuning this part as needed -----------------
            k       = 0.2
            angle_i = math.atan((k * 2 * self.wheelbase * math.sin(alpha)) / L) 
            angle   = angle_i*2
            # ----------------- tuning this part as needed -----------------

            f_delta = np.round(np.clip(angle, -0.3, 0.3), 3)

            f_delta_deg = np.round(np.degrees(f_delta))

            print("TRYING TO REACH: ", self.path_points_x[self.goal], self.path_points_y[self.goal])
            # print("Current index: " + str(self.goal))
            ct_error = np.round(np.sin(alpha) * L, 3)
            # print("Crosstrack Error: " + str(ct_error))
            # print("Front steering angle: " + str(f_delta_deg) + " degrees")
            # print("\n")

            self.drive_msg.header.stamp = rospy.get_rostime()
            self.drive_msg.drive.steering_angle = f_delta
            self.ctrl_pub.publish(self.drive_msg)
        
            self.rate.sleep()


def pure_pursuit():

    rospy.init_node('vicon_pp_node', anonymous=True)
    pp = PurePursuit()

    try:
        pp.start_pp()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    pure_pursuit()