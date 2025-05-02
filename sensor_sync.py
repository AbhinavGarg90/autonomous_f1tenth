import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Header
from message_filters import ApproximateTimeSynchronizer
from sensor_msgs.msg import LaserScan
import time

class SyncedSensorProcessor:
    def __init__(self):
        self.latest_velocity = None
        self.latest_steering = None
        self.latest_lidar_msg = None  # assuming this is a Header or something with .stamp
        self.time_stamp = time.time()

        # Subscribers
        self.steering_sub = rospy.Subscriber('/vesc/commands/servo/position', Float64, self.turning_callback)

    def turning_callback(self, steering_msg):
        # Store values internally so other functions can use them
        lidar_msg = rospy.wait_for_message('/scan', LaserScan)
        odom_msg = rospy.wait_for_message('/vesc/odom', Odometry)

        self.latest_velocity = odom_msg.twist.twist.linear.x
        self.latest_steering = steering_msg
        self.latest_lidar_msg = lidar_msg

    def get_sensor_data(self):
        dt = time.time() - self.time_stamp
        self.time_stamp = time.time()
        return (self.latest_velocity, self.latest_steering, self.latest_lidar_msg, dt)
