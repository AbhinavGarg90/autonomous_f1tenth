#!/usr/bin/env python
"""
Test file for the Occupancy Grid Mapping module.
This test simulates a robot moving straight through a corridor,
where the robot’s LIDAR detects side walls.
It does not depend on ROS—only the occupancy grid logic is exercised.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from Grid import OccupancyGridMapping  

# **MODIFIED:** Define a FakeLaserScan class to simulate ROS LaserScan messages.
class FakeLaserScan:
    def __init__(self, angle_min, angle_increment, ranges):
        self.angle_min = angle_min
        self.angle_increment = angle_increment
        self.ranges = ranges

# **MODIFIED:** This function simulates a LIDAR scan for a corridor.
# The corridor is assumed to have walls at y = +5 (left) and y = -5 (right).
# The simulated scan uses a field-of-view from -90° to +90° in the robot frame.
def simulate_scan():
    num_beams = 19
    angle_min = -math.pi/2   # -90 degrees
    angle_increment = math.pi / 18  # 10° increments, total FOV = 180°
    ranges = []
    for i in range(num_beams):
        angle = angle_min + i * angle_increment
        # For beams pointing to the side, compute the range to the wall.
        # For beams nearly horizontal (|sin(angle)| is very small), use a maximum range.
        if abs(math.sin(angle)) < 0.001:
            r = 20.0
        else:
            # When angle > 0, left wall is at y=5; when angle < 0, right wall is at y=-5.
            # Distance = wall_distance / |sin(angle)|.
            r_hit = 5.0 / abs(math.sin(angle))
            r = r_hit if r_hit < 20.0 else 20.0
        ranges.append(r)
    return FakeLaserScan(angle_min, angle_increment, ranges)

def main():
    # Set up an occupancy grid for testing.
    # For these tests we use a smaller grid with known origin.
    width = 200             # number of columns (cells)
    height = 200            # number of rows (cells)
    resolution = 0.1        # each cell is 0.1 x 0.1 meters
    origin_x = -10.0        # world x-coordinate corresponding to grid[0,0]
    origin_y = -10.0        # world y-coordinate corresponding to grid[0,0]
    
    # Create an instance of the occupancy grid mapping (the module from your final code).
    ogm = OccupancyGridMapping(width, height, resolution, origin_x, origin_y)

    # Set the initial robot pose. In this test, the robot starts at (0, 0) and is facing right (theta = 0).
    robot_pose = (0.0, 0.0, 0.0)

    # **MODIFIED:** Simulate a series of scans while the robot moves straight.
    # In this simulation the robot advances by 0.2 m in the x-direction after each scan.
    num_scans = 5
    for _ in range(num_scans):
        fake_scan = simulate_scan()
        ogm.update(robot_pose, fake_scan)
        # Update the robot pose to simulate forward motion.
        robot_pose = (robot_pose[0] + 0.2, robot_pose[1], robot_pose[2])

    # Retrieve the probability map from the occupancy grid.
    prob_map = ogm.get_probability_map()

    # Visualize the occupancy grid probability map.
    plt.imshow(prob_map, cmap='gray', origin='lower')
    plt.title("Occupancy Grid Probability Map")
    plt.xlabel("Grid Column")
    plt.ylabel("Grid Row")
    cbar = plt.colorbar()
    cbar.set_label("Occupancy Probability (0-100)")
    plt.show()

    # As a basic test, print the probability values for cells expected to represent the walls.
    # With origin (-10, -10) and resolution 0.1:
    # - The left wall at y = +5 corresponds approximately to row index: (5 - (-10)) / 0.1 = 150.
    #   We test the cell at (x=0, y=5) → world coordinate (0, 5).
    # - The right wall at y = -5 corresponds approximately to row index: (-5 - (-10)) / 0.1 = 50.
    #   We test the cell at (x=0, y=-5) → world coordinate (0, -5).
    test_cell_left = ogm.world_to_map(0.0, 5.0)    # returns (row, col)
    test_cell_right = ogm.world_to_map(0.0, -5.0)
    print("Probability at left wall cell (0,5):", prob_map[test_cell_left[0], test_cell_left[1]])
    print("Probability at right wall cell (0,-5):", prob_map[test_cell_right[0], test_cell_right[1]])

if __name__ == '__main__':
    main()
