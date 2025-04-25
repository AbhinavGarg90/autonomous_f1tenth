from lidar_polled import get_lidar_data
from ICP import ICPLocalizer
from live_plotter import LivePlotter
import rospy
import numpy as np
import time
from GridComp import OccupancyGridMapping
import matplotlib.pyplot as plt


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

if sim:
    from model_pos import CarPoseTracker
    gtpose_tracker = CarPoseTracker()
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
while not rospy.is_shutdown():
    lidar_data, raw_data = get_lidar_data(lidar_topic)
    est_pose = icp.update(lidar_data)
    act_pose = gtpose_tracker.get_pose()
    used_pose = act_pose
    occupancy_node.update_map(used_pose[0], used_pose[1], used_pose[2], raw_data)
    # occupancy_node.update_map(act_pose[0] - gt_pose_orig[0],
    #                           act_pose[1] - gt_pose_orig[1],
    #                           act_pose[2] - gt_pose_orig[2],
    #                           raw_data)
    map = occupancy_node.get_probability_map()
    im.set_data(map)
    plt.pause(0.01)

    map_pose = est_pose
    row, col = occupancy_node.world_to_map(est_pose[0], est_pose[1])
    row_gt, col_gt = occupancy_node.world_to_map(act_pose[0], act_pose[1])
    robot_dot.set_offsets([[col,row]])
    robot_dot_gt.set_offsets([[col_gt,row_gt]])
    # plt.pause(0.01)  # Allow time to render
