import rospy
from std_msgs.msg import Float64
from queue import Queue
import math

#default angle was 0.53 radians
class SteeringIntegrator:
    def __init__(self, v=2.0, L=0.5, dt=0.05):
        """
        v: forward velocity (m/s)
        L: wheelbase (m)
        dt: time step (s)
        """
        self.v = v
        self.L = L
        self.dt = dt
        self.theta = 0.0  # internal heading (radians)
        self.sub = rospy.Subscriber('/vesc/commands/servo/position', Float64, self.servo_callback)

    def servo_callback(self, msg):
        delta = msg.data - 0.53  # steering angle in radians + standardize
        d_theta = (self.v / self.L) * math.tan(delta) * self.dt
        self.theta += d_theta

    def get_heading(self):
        return self.theta

