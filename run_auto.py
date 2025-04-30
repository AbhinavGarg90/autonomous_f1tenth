from lidar_polled import get_lidar_data
from ICP import ICPLocalizer
from live_plotter import LivePlotter
import rospy
import numpy as np
import time
from GridComp import OccupancyGridMapping
import matplotlib.pyplot as plt
from odom import VESCMotorIntegrator
import sys
import subprocess
import numpy as np
import pickle
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray

pose_pub = rospy.Publisher('/icp_estimated_pose', PoseStamped, queue_size=10)
lidar_pub = rospy.Publisher('/raw_lidar_points', Float32MultiArray, queue_size=10)

occupancy_node = OccupancyGridMapping(origin_x_wc=0) 
height_gc, width_gc = occupancy_node.log_odds.shape
grid = np.zeros((height_gc, width_gc), dtype=np.int8) # setting appropriate grid size for imshow
# Setup plot
fig, ax = plt.subplots()
im = ax.imshow(grid, cmap='viridis', vmin=0, vmax=100, interpolation='none')

robot_dot = ax.scatter([],[],marker='^',s=60,c = 'red' )
robot_dot_gt = ax.scatter([],[],marker='^',s=60,c = 'green' )

# Add grid lines between cells
ax.set_xticks(np.arange(-0.5, 200, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 200, 1), minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=0.1)
ax.tick_params(which='minor', bottom=False, left=False)
ax.set_xticks([])
ax.set_yticks([])

sim = True
rospy.init_node("icp_runner")
lidar_topic = '/car_1/scan' if sim else 'scan'

lidar_data, raw_data = get_lidar_data(lidar_topic)

using_gt = False
if sim:
    from model_pos import CarPoseTracker
    gtpose_tracker = CarPoseTracker()
    using_gt = True
# else:
#     from vicon_bridge import Vicon
#     gtpose_tracker = Vicon()

# gt_pose_orig = gtpose_tracker.get_pose()
icp = ICPLocalizer()
icp.initialize(lidar_data)

# plotter = LivePlotter(gt_pose)
# Suppose you read the LaserScan from your own subscription or from icp
# Init imshow plot
prev_lidar_data, raw_data = get_lidar_data(lidar_topic)
est_pose = [0, 0, 0]

proc = subprocess.Popen(
    ["python3", "run_mapping.py"],
    stdin=subprocess.PIPE
)

while not rospy.is_shutdown():
    lidar_data, raw_data = get_lidar_data(lidar_topic)
    est_pose[2] = icp.update(lidar_data)[2]
    est_pose = vesc.integrate_pose(est_pose)
    # if using_gt:
    #     act_pose = gtpose_tracker.get_pose()
    pose_msg = PoseStamped()
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.header.frame_id = "map"
    pose_msg.pose.position.x = est_pose[0]
    pose_msg.pose.position.y = est_pose[1]
    pose_msg.pose.position.z = 0.0
    pose_msg.pose.orientation.z = np.sin(est_pose[2]/2.0)
    pose_msg.pose.orientation.w = np.cos(est_pose[2]/2.0)
    pose_pub.publish(pose_msg)

    flat_lidar = raw_data.flatten()
    lidar_msg = Float32MultiArray()
    lidar_msg.data = flat_lidar.tolist()
    lidar_pub.publish(lidar_msg)
