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

# plotter = LivePlotter(gt_pose)
# Suppose you read the LaserScan from your own subscription or from icp
# Init imshow plot

time.sleep(1)

while 1:
    size_bytes = sys.stdin.buffer.read(4)
    if not size_bytes:
        break  # No more data

    size = int.from_bytes(size_bytes, 'big')

    # Read the actual data
    data = sys.stdin.buffer.read(size)

    if not data:
        break

    est_pose, raw_data = pickle.loads(data)
    used_pose = est_pose

    occupancy_node.update_map(used_pose[0], used_pose[1], used_pose[2], raw_data)
    map = occupancy_node.get_probability_map()
    im.set_data(map)
    plt.pause(0.01)

    map_pose = est_pose
    row, col = occupancy_node.world_to_map(est_pose[0], est_pose[1])
    robot_dot.set_offsets([[col,row]])
