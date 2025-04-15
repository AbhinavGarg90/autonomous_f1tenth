import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Pose
import math

def quaternion_to_yaw(q: Quaternion) -> float:
    # Yaw calculation from quaternion
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class CarPoseTracker:
    def __init__(self, model_name='car_1'):
        self.model_name = model_name

    def get_pose(self):
        msg = rospy.wait_for_message('/gazebo/model_states', ModelStates)
        try:
            idx = msg.name.index(self.model_name)
            position = msg.pose[idx].position
            orientation = msg.pose[idx].orientation
            return position.x, position.y, quaternion_to_yaw(orientation)
        except ValueError:
            rospy.logwarn(f"Model '{self.model_name}' not found in /gazebo/model_states")
            return None

if __name__ == '__main__':
    rospy.init_node('car_pose_tracker')
    tracker = CarPoseTracker('car_1')
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        pose = tracker.get_pose()
        if pose:
            pass
            # print(f"car_1 position: x={pose.position.x:.2f}, y={pose.position.y:.2f}")
        rate.sleep()

