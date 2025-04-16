from lidar_polled import get_lidar_data
from ICP import ICPLocalizer
from live_plotter import LivePlotter
import rospy
import numpy as np
import time
from GridComp import OccupancyGridMapping
import matplotlib.pyplot as plt

# Setup plot
fig, ax = plt.subplots()
grid = np.random.randint(0, 101, size=(100, 100))
im = ax.imshow(grid, cmap='viridis', vmin=0, vmax=100, interpolation='none')

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
else:
    from vicon_bridge import Vicon
    gtpose_tracker = Vicon()

gt_pose_orig = gtpose_tracker.get_pose()
icp = ICPLocalizer()
icp.initialize(lidar_data)

# plotter = LivePlotter(gt_pose)

occupancy_node = OccupancyGridMapping()
# Suppose you read the LaserScan from your own subscription or from icp
# Init imshow plot
prev_lidar_data, raw_data = get_lidar_data(lidar_topic)
while not rospy.is_shutdown():
    lidar_data, raw_data = get_lidar_data(lidar_topic)
    est_pose = icp.update(lidar_data)
    act_pose = gtpose_tracker.get_pose()
    # occupancy_node.update_map(est_pose[0], est_pose[1], est_pose[2], raw_data)
    occupancy_node.update_map(act_pose[0] - gt_pose_orig[0],
                              act_pose[1] - gt_pose_orig[1],
                              act_pose[2] - gt_pose_orig[2],
                              raw_data)
    map = occupancy_node.get_probability_map()
    im.set_data(map)
    plt.pause(0.01)  # Allow time to render
