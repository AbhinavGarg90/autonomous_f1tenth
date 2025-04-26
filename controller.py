#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------
# ROS Controller Node for Hybrid A* Path Following (using Odometry)
# Author: Rahul's Team + ChatGPT Modifications
# Description: This node subscribes to the robot's odometry (pose + velocity)
#              and a planned path (in the map frame). It transforms the robot's
#              pose into the map frame using TF. It calculates steering commands
#              using Pure Pursuit and speed commands using a PID controller based
#              on measured velocity. It publishes AckermannDriveStamped commands
#              to the VESC Ackermann Mux input topic.
# Requirements: rospy, numpy, math, tf2_ros, tf2_geometry_msgs,
#               ackermann_msgs, nav_msgs, geometry_msgs
# Assumes:
#   - A node (e.g., VESC driver) publishes nav_msgs/Odometry on /vesc/odom (or configured topic).
#   - A localization system (like robot_localization or direct TF publish from ICP) provides
#     a TF transform from the 'odom' frame to the 'map' frame.
#   - A planner node publishes nav_msgs/Path on /planned_path in the 'map' frame.
#   - The VESC Ackermann Mux listens on /vesc/ackermann_cmd_mux/input/navigation.
# -------------------------

import rospy
import numpy as np
import math
import tf2_ros # For coordinate transformations
import tf2_geometry_msgs # For transforming ROS geometry messages
from ackermann_msgs.msg import AckermannDriveStamped # Output command type
from nav_msgs.msg import Path, Odometry # Path input, Odometry input
from geometry_msgs.msg import PoseStamped, Point, TransformStamped # Needed for TF transforms

# Try to import tf quaternion utilities, provide fallback
try:
    from tf.transformations import euler_from_quaternion
    tf_quat_available = True
except ImportError:
    rospy.logwarn("tf.transformations not found. Using basic quaternion math for yaw extraction (less robust).")
    tf_quat_available = False

# ==============================================================
# Utility Functions
# ==============================================================

def normalize_angle(angle):
    """Normalizes an angle to the range [-pi, pi] radians."""
    return math.atan2(math.sin(angle), math.cos(angle))

def get_yaw_from_quaternion_msg(quaternion):
    """Extracts yaw angle from geometry_msgs/Quaternion."""
    if tf_quat_available:
        try:
             euler = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
             return euler[2] # Yaw is the third element
        except Exception as e:
            rospy.logerr_throttle(5.0, f"Error converting quaternion to yaw using tf: {e}")
            return 0.0
    else:
        # Basic conversion if tf is not available
        q = quaternion
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        # rospy.logwarn_throttle(10.0,"Using basic yaw calculation due to missing tf library.")
        return yaw

def distance_between_points(p1, p2):
    """Calculates Euclidean distance between two points (Point messages or tuples)."""
    # Handles both Point messages and tuples/lists gracefully
    x1 = p1.x if hasattr(p1, 'x') else p1[0]
    y1 = p1.y if hasattr(p1, 'y') else p1[1]
    x2 = p2.x if hasattr(p2, 'x') else p2[0]
    y2 = p2.y if hasattr(p2, 'y') else p2[1]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# ==============================================================
# Controller Node Class
# ==============================================================

class ControllerNode:
    """
    ROS Node implementing path following control using Odometry input.
    """
    def __init__(self):
        """Initialize the controller node, parameters, subscribers, publishers, and TF listener."""
        rospy.init_node('path_following_controller')
        rospy.loginfo("Path Following Controller Node Initializing...")

        # --- Parameters ---
        # --- Vehicle ---
        self.wheelbase = rospy.get_param("~wheelbase", 0.325) # meters - F1TENTH typical. MUST MATCH REALITY/PLANNER!
        self.max_steering_angle = rospy.get_param("~max_steering", 0.4189) # Max physical steer angle (rad) - F1TENTH typical
        self.max_speed = rospy.get_param("~max_speed", 4.0) # Max desired speed (m/s) - Adjust for safety!
        self.min_speed = rospy.get_param("~min_speed", 0.2) # Min speed when moving (m/s) to avoid stalling PID

        # --- Pure Pursuit (Lateral Control) ---
        self.lookahead_base = rospy.get_param("~lookahead_base", 0.4) # Base lookahead distance (m) - TUNE
        self.lookahead_factor = rospy.get_param("~lookahead_factor", 0.15) # Lookahead proportional to speed (s) Ld = base + factor*speed - TUNE
        self.min_lookahead = rospy.get_param("~min_lookahead", 0.5) # Minimum lookahead distance (m) - TUNE
        # Note: Pure Pursuit gain 'k' is implicitly handled by the formula using Ld

        # --- PID (Longitudinal Control) ---
        self.target_speed = rospy.get_param("~target_speed", 2.0) # Desired constant speed (m/s) - TUNE THIS
        self.Kp = rospy.get_param("~Kp", 1.2) # Proportional gain - TUNE THIS FIRST
        self.Ki = rospy.get_param("~Ki", 0.1) # Integral gain - TUNE THIS LAST, START SMALL
        self.Kd = rospy.get_param("~Kd", 0.3) # Derivative gain - TUNE THIS SECOND
        self.integral_max = rospy.get_param("~integral_max", 1.0) # Max value for integral term anti-windup
        self.integral_min = rospy.get_param("~integral_min", -1.0)# Min value for integral term anti-windup

        # --- General ---
        self.controller_rate = rospy.get_param("~controller_rate", 50.0) # Hz - Frequency of control loop
        self.goal_reached_tolerance = rospy.get_param("~goal_reached_tolerance", 0.3) # meters - How close to final waypoint to stop

        # --- Coordinate Frames ---
        self.map_frame = rospy.get_param("~map_frame", "map") # Target frame for calculations
        # self.odom_frame = rospy.get_param("~odom_frame", "odom") # Frame ID expected from Odometry msg header
        self.robot_base_frame = rospy.get_param("~robot_base_frame", "base_link") # Robot's base frame (child_frame_id in Odometry)

        # --- Topic Names ---
        odom_topic = rospy.get_param("~odom_topic", "/vesc/odom") # Topic with nav_msgs/Odometry
        path_topic = rospy.get_param("~path_topic", "/planned_path") # Topic with nav_msgs/Path
        drive_topic = rospy.get_param("~drive_topic", "/vesc/ackermann_cmd_mux/input/navigation") # AckermannDriveStamped output

        # --- State Variables ---
        self.current_odom = None # Stores the latest Odometry message
        self.current_path = None # Stores the latest Path message
        self.current_path_segment_index = 0 # Index to track progress along the path

        # PID state variables
        self.pid_integral = 0.0
        self.pid_last_error = 0.0
        self.pid_last_time = None

        # --- ROS Communication & TF ---
        # Publisher for AckermannDriveStamped commands
        self.drive_pub = rospy.Publisher(drive_topic, AckermannDriveStamped, queue_size=1)

        # Subscribers
        rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber(path_topic, Path, self.path_callback, queue_size=1)

        # TF Listener Setup
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0)) # Buffer stores transforms for 10 seconds
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.loginfo("TF Listener initialized.")

        # Timer for the main control loop
        self.control_timer = rospy.Timer(rospy.Duration(1.0 / self.controller_rate), self.control_loop_callback)

        rospy.loginfo("Controller Node Initialized.")
        rospy.loginfo(f" TargetSpeed: {self.target_speed:.2f}, MaxSpeed: {self.max_speed:.2f}")
        rospy.loginfo(f" Kp: {self.Kp:.2f}, Ki: {self.Ki:.2f}, Kd: {self.Kd:.2f}")
        rospy.loginfo(f" Lookahead Base: {self.lookahead_base:.2f}, Factor: {self.lookahead_factor:.2f}, Min: {self.min_lookahead:.2f}")
        rospy.loginfo(f" Subscribing Odometry: {odom_topic}, Path: {path_topic}")
        rospy.loginfo(f" Publishing Drive to: {drive_topic}")
        rospy.loginfo(f" Operating in '{self.map_frame}' frame, expecting robot base frame '{self.robot_base_frame}'.")


    def odom_callback(self, msg):
        """Stores the latest odometry message."""
        # Store the entire message, contains pose and twist (velocity)
        self.current_odom = msg
        # We no longer need to estimate velocity here!


    def path_callback(self, msg):
        """Stores the latest path and resets path tracking state."""
        if not msg.poses: # Check if the received path is empty
             rospy.logwarn("Received empty path. Stopping and clearing current path.")
             self.current_path = None
             self.stop_vehicle()
             return

        # Check path frame ID (optional but recommended)
        if msg.header.frame_id != self.map_frame:
             rospy.logwarn_throttle(10.0, f"Received path in frame '{msg.header.frame_id}' but controller operates in '{self.map_frame}'. Assuming TF is correct.")
             # Ideally, planner should publish in map_frame or controller should transform path

        rospy.loginfo(f"Controller received new path with {len(msg.poses)} waypoints in frame '{msg.header.frame_id}'.")
        self.current_path = msg
        self.current_path_segment_index = 0 # Reset to start tracking from the beginning
        # Reset PID state when a new path is received
        self.pid_integral = 0.0
        self.pid_last_error = 0.0
        self.pid_last_time = rospy.Time.now() # Reset PID time


    def get_robot_pose_in_map_frame(self):
        """
        Looks up the TF transform from the odometry frame to the map frame
        and applies it to the current odometry pose.
        Returns:
            geometry_msgs/PoseStamped: The robot's pose in the map frame, or None if transform fails.
        """
        if self.current_odom is None:
            return None

        source_frame = self.current_odom.header.frame_id # Usually 'odom'
        target_frame = self.map_frame # Usually 'map'
        source_time = self.current_odom.header.stamp

        try:
            # Wait for transform (briefly) and get the TransformStamped
            # Use source_time to get the transform at the time the odom was captured
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, source_time, rospy.Duration(0.1))

            # Create a PoseStamped message from the Odometry pose to transform it
            pose_source = PoseStamped()
            pose_source.header = self.current_odom.header # Use odom header (frame_id, stamp)
            pose_source.pose = self.current_odom.pose.pose # The pose part of Odometry

            # Transform the pose using tf2_geometry_msgs
            pose_target = tf2_geometry_msgs.do_transform_pose(pose_source, transform)

            # rospy.logdebug(f"TF Success: Transformed pose from {source_frame} to {target_frame}")
            return pose_target # This is a PoseStamped

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException, tf2_ros.TransformException) as e:
            rospy.logwarn_throttle(2.0, f"TF Exception: Could not transform from {source_frame} to {target_frame}: {e}")
            return None


    def find_lookahead_point(self, current_pose_map):
        """
        Finds the lookahead point on the current path in the map frame.
        Args:
            current_pose_map (geometry_msgs/Pose): Robot's pose in the map frame.
        Returns:
            geometry_msgs/Point: The lookahead point in the map frame, or None.
        """
        if not self.current_path or not self.current_path.poses:
            return None

        current_x = current_pose_map.position.x
        current_y = current_pose_map.position.y
        current_velocity = self.current_odom.twist.twist.linear.x # Use velocity from odom

        # Calculate dynamic lookahead distance based on current velocity
        lookahead_distance = self.lookahead_base + self.lookahead_factor * abs(current_velocity)
        lookahead_distance = max(self.min_lookahead, lookahead_distance)
        # rospy.logdebug(f"Calculated Lookahead Distance (Ld): {lookahead_distance:.2f} m (v={current_velocity:.2f})")

        # Start searching from the current segment index to avoid going backwards
        search_start_index = self.current_path_segment_index

        # Iterate through path segments starting from the current index
        for i in range(search_start_index, len(self.current_path.poses) - 1):
            p1 = self.current_path.poses[i].pose.position     # Start point of segment
            p2 = self.current_path.poses[i+1].pose.position   # End point of segment

            # Check simple distance first: If the end of the segment is further than Ld,
            # the target might be on this segment.
            dist_to_p2 = distance_between_points(current_pose_map.position, p2)

            if dist_to_p2 >= lookahead_distance:
                # --- Precise Intersection Calculation (Circle-Line Segment) ---
                d_segment = distance_between_points(p1, p2)
                if d_segment < 1e-6: continue # Skip zero-length segments

                vec_rx_p1_x = p1.x - current_x # Vector from robot to p1 (x)
                vec_rx_p1_y = p1.y - current_y # Vector from robot to p1 (y)
                vec_p1_p2_x = p2.x - p1.x      # Vector along segment (x)
                vec_p1_p2_y = p2.y - p1.y      # Vector along segment (y)

                # Solve quadratic equation: a*t^2 + b*t + c = 0 for intersection fraction 't'
                a = vec_p1_p2_x**2 + vec_p1_p2_y**2 # = d_segment^2
                b = 2 * (vec_rx_p1_x * vec_p1_p2_x + vec_rx_p1_y * vec_p1_p2_y)
                c = vec_rx_p1_x**2 + vec_rx_p1_y**2 - lookahead_distance**2
                discriminant = b**2 - 4*a*c

                if discriminant >= 0: # Real intersection exists
                    sqrt_discriminant = math.sqrt(discriminant)
                    # Find the two potential intersection parameters along the infinite line
                    t1 = (-b + sqrt_discriminant) / (2*a)
                    t2 = (-b - sqrt_discriminant) / (2*a)

                    # Select the valid parameter 't' that falls WITHIN the segment [0, 1]
                    # Prefer the one further along the path if both are valid (t1 usually)
                    valid_t_found = False
                    if 0 <= t1 <= 1:
                        t = t1
                        valid_t_found = True
                    elif 0 <= t2 <= 1:
                        t = t2
                        valid_t_found = True

                    if valid_t_found:
                        # Calculate the intersection point using the valid 't'
                        target_x = p1.x + t * vec_p1_p2_x
                        target_y = p1.y + t * vec_p1_p2_y
                        target_point = Point(x=target_x, y=target_y)
                        # Update the index so we start searching from here next time
                        self.current_path_segment_index = i
                        # rospy.logdebug(f"Lookahead point found on segment {i} at t={t:.2f}")
                        return target_point

            # If we haven't returned yet, the intersection wasn't on this segment,
            # or the end point p2 was closer than Ld. Continue to the next segment.

        # --- If Loop Finishes ---
        # This means no intersection point was found further along the path,
        # or we are very close to the end. Target the last point of the path.
        if len(self.current_path.poses) > 0:
             target_point = self.current_path.poses[-1].pose.position
             # Keep the index on the last segment
             self.current_path_segment_index = max(0, len(self.current_path.poses) - 2)
             # rospy.logdebug("Lookahead point set to last waypoint.")
             return target_point

        # Should not be reached if path is valid, but return None as fallback
        return None


    def pure_pursuit_control(self, current_pose_map, target_point):
        """
        Calculates the steering angle using Pure Pursuit algorithm.
        Args:
            current_pose_map (geometry_msgs/Pose): Robot's pose in the map frame.
            target_point (geometry_msgs/Point): Lookahead point in the map frame.
        Returns:
            float: Required steering angle in radians.
        """
        if target_point is None:
            rospy.logwarn_throttle(1.0, "Pure Pursuit: No target point received.")
            return 0.0 # Command straight steering

        current_x = current_pose_map.position.x
        current_y = current_pose_map.position.y
        current_yaw = get_yaw_from_quaternion_msg(current_pose_map.orientation)

        # Calculate vector to target point in world frame
        dx = target_point.x - current_x
        dy = target_point.y - current_y

        # Calculate alpha: angle between robot's heading vector and lookahead vector
        angle_to_target = math.atan2(dy, dx)
        alpha = normalize_angle(angle_to_target - current_yaw)

        # Calculate the actual distance to the lookahead point found
        lookahead_distance = distance_between_points(current_pose_map.position, target_point)
        # Prevent division by zero or instability if lookahead is very small
        lookahead_distance = max(0.01, lookahead_distance) # Use a small minimum value

        # Calculate steering angle using the standard Pure Pursuit formula
        # Use atan2 for numerical stability, equivalent to atan(term)
        steering_rad = math.atan2(2.0 * self.wheelbase * math.sin(alpha), lookahead_distance)

        # Clamp steering angle to physical limits of the robot
        steering_clamped = np.clip(steering_rad, -self.max_steering_angle, self.max_steering_angle)

        # rospy.logdebug(f"Pure Pursuit: Ld={lookahead_distance:.2f}, Alpha={math.degrees(alpha):.1f}deg, Steer={math.degrees(steering_clamped):.1f}deg")
        return steering_clamped


    def pid_control(self, current_velocity, current_time):
        """
        Calculates target speed using PID controller based on measured velocity.
        Args:
            current_velocity (float): Measured linear velocity (m/s) from Odometry.
            current_time (rospy.Time): Timestamp for dt calculation.
        Returns:
            float: Target speed command (m/s), clamped within limits.
        """
        # Handle first run
        if self.pid_last_time is None:
            self.pid_last_time = current_time
            return self.min_speed # Start slow

        dt = (current_time - self.pid_last_time).to_sec()
        # Prevent division by zero or large spikes if dt is too small or zero
        if dt <= 1e-4:
            # If dt is negligible, return the previous target speed or a safe speed
            # to avoid large derivative spikes. Let's just reuse the last command implicitly
            # by not updating pid_last_time, or return clamped target.
            # Returning clamped target is safer if velocity might change rapidly.
            return np.clip(self.target_speed, self.min_speed, self.max_speed)

        # --- Calculate PID Terms ---
        # Error: Difference between desired speed and current speed
        error = self.target_speed - current_velocity

        # Proportional Term: Directly proportional to the current error
        p_term = self.Kp * error

        # Integral Term: Accumulates error over time to eliminate steady-state error
        self.pid_integral += error * dt
        # Anti-windup: Clamp the integral term to prevent excessive accumulation
        self.pid_integral = np.clip(self.pid_integral, self.integral_min, self.integral_max)
        i_term = self.Ki * self.pid_integral

        # Derivative Term: Responds to the rate of change of the error (predictive)
        # Calculate derivative based on change in error, not change in measurement,
        # to avoid "derivative kick" when the target speed changes.
        error_diff = error - self.pid_last_error
        d_term = self.Kd * (error_diff / dt)

        # --- Update State for Next Iteration ---
        self.pid_last_error = error
        self.pid_last_time = current_time

        # --- Calculate Final Output ---
        # Sum the terms to get the raw PID output
        pid_output = p_term + i_term + d_term

        # Clamp the output speed to the vehicle's operational limits
        final_speed = np.clip(pid_output, self.min_speed, self.max_speed)

        # --- Logging (Optional - can be verbose) ---
        # rospy.logdebug(f"PID: Target={self.target_speed:.2f}, Current={current_velocity:.2f}, Error={error:.2f}, P={p_term:.2f}, I={i_term:.2f}, D={d_term:.2f}, Out={final_speed:.2f}")

        return final_speed


    def control_loop_callback(self, event=None): # Add event=None for direct calls
        """
        Main control loop executed periodically by the ROS Timer.
        Orchestrates getting state, calculating controls, and publishing commands.
        """
        # --- 1. Check for valid inputs ---
        if self.current_odom is None:
            rospy.logwarn_throttle(5.0, "Controller waiting for initial odometry message...")
            # Optional: Could publish stop command here just in case
            # self.stop_vehicle()
            return
        if self.current_path is None or not self.current_path.poses:
            # No path received yet, or path is empty. Wait or stop.
            # If already moving, good practice to send stop.
            # rospy.logwarn_throttle(5.0, "Controller waiting for a valid path...")
            self.stop_vehicle()
            return

        # --- 2. Get Robot Pose in Map Frame ---
        # Transform the current odometry pose into the map frame for calculations
        current_pose_map_stamped = self.get_robot_pose_in_map_frame()
        if current_pose_map_stamped is None:
            rospy.logwarn_throttle(2.0, "Controller could not get robot pose in map frame. Skipping control cycle.")
            # Publish stop command if TF fails persistently?
            self.stop_vehicle()
            return
        # Use the pose part for calculations
        current_pose_map = current_pose_map_stamped.pose
        current_time = current_pose_map_stamped.header.stamp # Use timestamp from TF'd pose

        # --- 3. Find Lookahead Point ---
        # Find the target point on the path based on the robot's position in the map frame
        target_point = self.find_lookahead_point(current_pose_map)

        # --- 4. Check if Goal Reached or Target Lost ---
        # Check distance to the *final* waypoint in the path
        dist_to_final_wp = distance_between_points(current_pose_map.position, self.current_path.poses[-1].pose.position)

        if dist_to_final_wp < self.goal_reached_tolerance:
              rospy.loginfo("Controller: Goal region reached (close to final waypoint). Stopping.")
              self.stop_vehicle()
              self.current_path = None # Clear path to stop further control actions
              return

        if target_point is None:
             # This might happen if the path is very short or calculation fails
             rospy.logwarn_throttle(2.0,"Controller: Path available but failed to find valid lookahead point. Stopping.")
             self.stop_vehicle()
             self.current_path = None # Clear path as we can't follow it
             return

        # --- 5. Calculate Control Commands ---
        # --- Lateral Control ---
        # Calculate steering angle using Pure Pursuit based on map frame pose and target
        target_steering = self.pure_pursuit_control(current_pose_map, target_point)

        # --- Longitudinal Control ---
        # Get current velocity directly from Odometry message twist component
        current_velocity = self.current_odom.twist.twist.linear.x
        # Calculate target speed using PID controller
        target_speed = self.pid_control(current_velocity, current_time)

        # --- 6. Publish Command ---
        self.publish_drive_command(target_speed, target_steering, current_time)


    def publish_drive_command(self, speed, steering_angle, timestamp):
        """Creates and publishes an AckermannDriveStamped message."""
        drive_msg = AckermannDriveStamped()
        # Header: Use the timestamp from the state message used for calculation
        drive_msg.header.stamp = timestamp
        # Frame ID: Often 'base_link' or the odom frame, depending on what the VESC driver expects.
        # Check VESC driver documentation. Let's assume odom frame is safer.
        drive_msg.header.frame_id = self.current_odom.header.frame_id if self.current_odom else self.robot_base_frame

        # Drive command: Populate speed and steering angle
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        # Set other fields to zero if not controlled
        drive_msg.drive.steering_angle_velocity = 0.0
        drive_msg.drive.acceleration = 0.0
        drive_msg.drive.jerk = 0.0

        self.drive_pub.publish(drive_msg)


    def stop_vehicle(self):
        """Publishes a command with zero speed to stop the vehicle."""
        # Use current ROS time for the timestamp, as we don't have a specific state time
        self.publish_drive_command(speed=0.0, steering_angle=0.0, timestamp=rospy.Time.now())
        # rospy.logdebug("Stop command published.")


# ==============================================================
# Main Execution / Node Entry Point
# ==============================================================
if __name__ == '__main__':
    try:
        # Create an instance of the ControllerNode class. This runs __init__().
        controller_node = ControllerNode()
        # rospy.spin() keeps the node running and processing callbacks and timers
        # until the node is shut down (e.g., Ctrl+C).
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Controller node shutting down.")
    except Exception as e:
        # Log any other unexpected exceptions
        rospy.logfatal(f"Unhandled exception in Controller Node: {e}")
        import traceback
        traceback.print_exc()