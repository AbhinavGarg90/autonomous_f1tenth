from lidar_polled import get_lidar_data
from ICP import ICPLocalizer
from live_plotter import LivePlotter
import rospy
import numpy as np

sim = True
rospy.init_node("icp_runner")
lidar_topic = '/car_1/scan' if sim else 'scan'

lidar_data = get_lidar_data(lidar_topic)

if sim:
    from model_pos import CarPoseTracker
    gtpose_tracker = CarPoseTracker()
else:
    from vicon_bridge import Vicon
    gtpose_tracker = Vicon()

gt_pose = gtpose_tracker.get_pose()
icp = ICPLocalizer()
icp.initialize(lidar_data)

plotter = LivePlotter(gt_pose)

rate = rospy.Rate(10)
prev_lidar_data = get_lidar_data(lidar_topic)
while not rospy.is_shutdown():
    lidar_data = get_lidar_data(lidar_topic)
    est_pose = icp.update(lidar_data)

    gt_pose = gtpose_tracker.get_pose()
    plotter.update(est_pose, gt_pose)

    rate.sleep()

