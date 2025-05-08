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

class VehicleController():

    def __init__(self):
        # Publisher to publish the control input to the vehicle model
        self.controlPub = rospy.Publisher("/ackermann_cmd", AckermannDrive, queue_size = 1)
        self.prev_vel = 0
        self.L = 0.325 # Wheelbase, can be get from gem_control.py

        self.ctrl_pub  = rospy.Publisher("/vesc/low_level/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=1)
        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.header.frame_id = "f1tenth_control"
        self.drive_msg.drive.speed     = 1.0 # m/s, reference speed

        self.vicon_sub = rospy.Subscriber('/icp_estimated_pose', PoseStamped,  self.pose_callback )
        self.x   = 0
        self.y   = 0
        self.yaw = 0
        self.offset     = 0.15 # meters        

        self.goal_idx = 1 # 0 is 0,0,0

        self.read_waypoints() 

    def pose_callback(self,msg):    
        self.x = msg.pose.position.x,
        self.y = msg.pose.position.y,
        self.yaw = 2 * np.arctan2(msg.pose.orientation.z, msg.pose.orientation.w)  # extract yaw

    def get_f1tenth_state(self):

        # heading to yaw (degrees to radians)
        # heading is calculated from two GNSS antennas
        curr_yaw = self.yaw

        # reference point is located at the center of rear axle
        curr_x = self.x - self.offset * np.cos(curr_yaw)
        curr_y = self.y - self.offset * np.sin(curr_yaw)
        print("Curr x: ", curr_x, " Curr y: ", curr_y, "Curr yaw: ", curr_yaw)

        return np.round(curr_x, 3), np.round(curr_y, 3), np.round(self.yaw, 4)

    def read_waypoints(self):
        # read recorded GPS lat, lon, heading
        dirname  = "waypoints_in_csv"
        filename = os.path.join(dirname, "waypoints_world.csv")  #jay is sure the slash is needed 
        print(filename)
        with open(filename) as f:
            path_points = [tuple(line) for line in csv.reader(f)]
            path_points = [(float(x), float(y), float(theta)) for x, y, theta in path_points]
            print(path_points)
        # x towards East and y towards North
        # self.path_points_x_record   = [float(point[0]) for point in path_points] # x
        # self.path_points_y_record   = [float(point[1]) for point in path_points] # y
        # self.path_points_yaw_record = [float(point[2]) for point in path_points] # yaw
        self.waypoints = path_points
        print(path_points)

    # Task 3: Lateral Controller (Pure Pursuit)
    def pure_pursuit_lateral_controller(self, curr_x, curr_y, curr_yaw, future_unreached_waypoints ,curr_vel):

        def compute_pure_pursuit_steering(curr_x, curr_y, curr_yaw, target_x, target_y, wheelbase):
            dx = target_x - curr_x
            dy = target_y - curr_y

            ld = math.sqrt(dx**2 + dy**2)
            if ld < 1e-6:
                return 0
            
            heading = math.atan2(dy, dx)

            alpha = heading - curr_yaw
            alpha = (alpha + math.pi) % (2*math.pi) - math.pi

            return math.atan2(2 * wheelbase * math.sin(alpha), ld)

        ####################### TODO: Your TASK 3 code starts Here #######################
        #tx, ty = target_point
        #target_steering = compute_pure_pursuit_steering(curr_x, curr_y, curr_yaw, tx, ty, self.L)

        ####################### TODO: Your TASK 3 code starts Here #######################

        min_ld = 3.0           # minimum look-ahead distance
        max_ld = 15.0          # maximum look-ahead distance
        k_ld = 0.5              # scale factor for speed

        Ld = k_ld * curr_vel
        Ld = 1.0 # TODO: CHECK

        if len(future_unreached_waypoints) == 0:
            return 0.0

        if len(future_unreached_waypoints) == 1:
            target_x, target_y = future_unreached_waypoints[0]
            return compute_pure_pursuit_steering(curr_x, curr_y, curr_yaw, target_x, target_y, self.L)
        
        accumulated_dist = 0.0
        prev_wx, prev_wy = curr_x, curr_y

        lookahead_x = None
        lookahead_y = None

        for wp in future_unreached_waypoints:
            wx, wy, theta = wp
            seg_dist = math.sqrt((wx - prev_wx)**2 + (wy - prev_wy)**2)
            if accumulated_dist + seg_dist >= Ld:
                remain = Ld - accumulated_dist
                ratio = remain / seg_dist
                lookahead_x = prev_wx + ratio * (wx - prev_wx)
                lookahead_y = prev_wy + ratio * (wy - prev_wy)
                break
            else:
                accumulated_dist += seg_dist
                prev_wx, prev_wy = wx, wy

        # If we did not exceed Ld, just use the final waypoint
        if lookahead_x is None or lookahead_y is None:
            lookahead_x, lookahead_y = future_unreached_waypoints[-1]

        # 3) Compute steering
        target_steering = compute_pure_pursuit_steering(
            curr_x, curr_y, curr_yaw, lookahead_x, lookahead_y, self.L
        )

        return target_steering


    def euclidian_dist(self, curr_pose, other_pose):
        return np.sqrt((curr_pose[0] - other_pose[0])**2 + (curr_pose[1] - other_pose[1])**2)

    def get_closest_waypoint(self, curr_pose, waypoints_list):
        goal_idx = self.goal_idx
        limit = 0.1
        if self.euclidian_dist(curr_pose, waypoints_list[goal_idx]) < limit:
            self.goal_idx = goal_idx + 1
        return waypoints_list[goal_idx:]
        

    def execute(self):
        # Compute the control input to the vehicle according to the
        # current and reference pose of the vehicle
        # Input:
        #   currentPose: ModelState, the current state of the vehicle
        #   target_point: [target_x, target_y]
        #   future_unreached_waypoints: a list of future waypoints[[target_x, target_y]]
        # Output: None

        curr_pose = self.get_f1tenth_state()
        curr_x, curr_y, curr_yaw = curr_pose
        curr_vel = 1.0 # TODO: might fuck things up

        target_velocity = 1.0
        future_unreached_waypoints = self.get_closest_waypoint(curr_pose, self.waypoints)
        target_steering = self.pure_pursuit_lateral_controller(curr_x, curr_y, curr_yaw, future_unreached_waypoints, curr_vel)
        #Pack computed velocity and steering angle into Ackermann command
        self.drive_msg.header.stamp = rospy.get_rostime()
        self.drive_msg.drive.steering_angle = target_steering
        self.ctrl_pub.publish(self.drive_msg)
        print(self.drive_msg)
        return

if __name__ == '__main__':
    rospy.init_node('vicon_pp_node', anonymous=True)
    controller = VehicleController()
    while not rospy.is_shutdown():
        controller.execute()
