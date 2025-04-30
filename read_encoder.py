import rospy
from sensor_msgs.msg import Imu

def polled_listener():
    rospy.init_node('accel_imu_info_polled', anonymous=True)
    rate = rospy.Rate(10)  # 10 Hz polling rate

    while not rospy.is_shutdown():
        try:
            msg = rospy.wait_for_message('/D435I/accel/sample', Imu, timeout=1.0)
            gyro = rospy.wait_for_message('/D435I/gyro/sample', Imu, timeout=1.0)
            rospy.loginfo(msg)
            rospy.loginfo(gyro)
        except rospy.ROSException:
            rospy.logwarn("Timeout waiting for /D435I/accel/imu_info")
            rate.sleep()

if __name__ == '__main__':
    polled_listener()
