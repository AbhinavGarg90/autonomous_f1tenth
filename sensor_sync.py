import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from sensor_msgs.msg import LaserScan
from message_filters import Subscriber, ApproximateTimeSynchronizer
import time

class SyncedSensorProcessor:
    def __init__(self):
        self.latest_velocity = None
        self.latest_steering = None
        self.latest_lidar_msg = None
        self.time_stamp = time.time()

        # Create synchronized subscribers
        self.lidar_sub = Subscriber('/scan', LaserScan)
        self.odom_sub = Subscriber('/vesc/odom', Odometry)
        self.steering_sub = Subscriber('/vesc/commands/servo/position', Float64)

        self.sync = ApproximateTimeSynchronizer(
            [self.lidar_sub, self.odom_sub, self.steering_sub],
            queue_size=10,
            slop=0.05,
            allow_headerless=True
        )
        self.sync.registerCallback(self.synced_callback)

    def synced_callback(self, lidar_msg, odom_msg, steering_msg):
        self.latest_velocity = odom_msg.twist.twist.linear.x
        self.latest_steering = steering_msg
        self.latest_lidar_msg = lidar_msg
        self.time_stamp = time.time()

    def get_sensor_data(self):
        dt = time.time() - self.time_stamp
        self.time_stamp = time.time()
        return (self.latest_velocity, self.latest_steering, self.latest_lidar_msg, dt)
