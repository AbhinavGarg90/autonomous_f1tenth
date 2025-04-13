#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
import sys
import termios
import tty

def get_key():
    """Function to get a key press without Enter."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def motor_control(steering_angle: float, speed: float):
    # Publishers for steering and velocity control
    pub_left_steering = rospy.Publisher('/car_1/left_steering_hinge_position_controller/command', Float64, queue_size=1)
    pub_right_steering = rospy.Publisher('/car_1/right_steering_hinge_position_controller/command', Float64, queue_size=1)
    pub_left_velocity = rospy.Publisher('/car_1/left_front_wheel_velocity_controller/command', Float64, queue_size=1)  # Same for both wheels
    pub_right_velocity = rospy.Publisher('/car_1/right_front_wheel_velocity_controller/command', Float64, queue_size=1)

    steering_angle = max(-0.5, min(0.5, steering_angle))  # Steering angle limits
    speed = max(-1.0, min(1.0, speed))  # Speed limits

    # Publish to steering and velocity controllers
    pub_left_steering.publish(steering_angle)
    pub_right_steering.publish(steering_angle)
    pub_left_velocity.publish(speed)
    pub_right_velocity.publish(speed)
    
    print("Speed: %.2f m/s | Steering angle: %.2f radians" % (speed, steering_angle))


def main():
    rospy.init_node('wasd_drive_control')

    steering_angle = 0.0  # steering angle in radians
    speed = 0.0  # speed in m/s

    print("Use W/S to move forward/backward, A/D to steer left/right. Press Q to quit.")

    while not rospy.is_shutdown():
        key = get_key()

        if key.lower() == 'w':
            speed += 0.1  # Move forward
        elif key.lower() == 's':
            speed -= 0.1  # Move backward
        elif key.lower() == 'a':
            steering_angle += 0.1  # Steer left
        elif key.lower() == 'd':
            steering_angle -= 0.1  # Steer right
        elif key.lower() == 'q':
            break  # Exit the loop

        steering_angle = max(-0.5, min(0.5, steering_angle))  # Steering angle limits
        speed = max(-1.0, min(1.0, speed))  # Speed limits
        motor_control(steering_angle, speed)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

