import numpy as np
import math

# --- OccupancyGridMapping class (the core logic) ---
class OccupancyGridMapping:
    def __init__(self, width, height, resolution, origin_x, origin_y):
        """
        Initialize the occupancy grid mapping module.
        width, height: number of grid cells.
        origin_x, origin_y: World coordinates corresponding to grid[0,0].
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y

        # Initialize the grid in log odds (0 means 50% probability).
        self.log_odds = np.zeros((height, width), dtype=np.float32)

        # Log odds update parameters:
        self.l_occ = math.log(0.7 / 0.3)  # occupied update ≈ 0.847
        self.l_free = math.log(0.3 / 0.7) # free update ≈ -0.847
        self.l_min = -5.0 
        self.l_max =  5.0

    def world_to_map(self, x, y):
        """ Convert world coordinates (x, y) into grid indices (i, j). """
        j = int((x - self.origin_x) / self.resolution)
        i = int((y - self.origin_y) / self.resolution)
        return i, j

    def bresenham2D(self, x0, y0, x1, y1):
        """ Bresenham's line algorithm: returns list of (row, col) cells along a line. """
        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x1))
        y1 = int(round(y1))
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        cells = []
        while True:
            cells.append((y0, x0))  # (row, col)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return cells

    def update(self, robot_pose, scan):
        """
        Update the occupancy grid using the robot's current pose (x, y, theta) and a scan.
        'scan' is expected to have: angle_min, angle_increment, and ranges.
        """
        rx, ry, rtheta = robot_pose

        angle_min = scan.angle_min
        angle_increment = scan.angle_increment
        ranges = np.array(scan.ranges)
        num_beams = len(ranges)
        angles = angle_min + np.arange(num_beams) * angle_increment

        # Iterate through each beam in the scan.
        for i in range(num_beams):
            r = ranges[i]
            if np.isinf(r) or np.isnan(r):
                continue

            # Convert the beam angle from sensor to world frame.
            beam_angle = rtheta + angles[i]
            x_end = rx + r * math.cos(beam_angle)
            y_end = ry + r * math.sin(beam_angle)

            # Convert positions to grid indices.
            robot_i, robot_j = self.world_to_map(rx, ry)
            end_i, end_j = self.world_to_map(x_end, y_end)

            # Get the cells along the line using Bresenham.
            cells = self.bresenham2D(robot_j, robot_i, end_j, end_i)

            # Update free space (all cells before the last one).
            for (cell_i, cell_j) in cells[:-1]:
                if 0 <= cell_i < self.height and 0 <= cell_j < self.width:
                    self.log_odds[cell_i, cell_j] += self.l_free
                    self.log_odds[cell_i, cell_j] = max(self.l_min, self.log_odds[cell_i, cell_j])

            # Update the endpoint cell as occupied.
            if cells:
                cell_i, cell_j = cells[-1]
                if 0 <= cell_i < self.height and 0 <= cell_j < self.width:
                    self.log_odds[cell_i, cell_j] += self.l_occ
                    self.log_odds[cell_i, cell_j] = min(self.l_max, self.log_odds[cell_i, cell_j])

    def get_probability_map(self):
        """ Convert the log odds grid to a probability map scaled to [0, 100]. """
        prob = 1 - 1 / (1 + np.exp(self.log_odds))
        binary = (prob > 0.5).astype(np.int8)
        return binary

# --- Dummy classes simulating minimal ROS messages ---

class DummyLaserScan:
    """ Mimics a ROS LaserScan message with only the attributes we need. """
    def __init__(self, angle_min, angle_increment, ranges):
        self.angle_min = angle_min
        self.angle_increment = angle_increment
        self.ranges = ranges

# For our purposes the robot pose is simply a tuple (x, y, theta).

# --- Simulation/Test Function ---
def simulate_lidar_and_robot_poses():
    # Create a small occupancy grid (100x100 cells, 0.1 m resolution).
    width, height = 100, 100
    resolution = 0.1
    origin_x, origin_y = -5.0, -5.0  # sets world coordinate for grid[0,0]
    ogm = OccupancyGridMapping(width, height, resolution, origin_x, origin_y)
    
    # ---- First Simulation: Robot at (0, 0, 0) ----
    # Robot is at the origin, facing positive x-axis.
    robot_pose1 = (0.0, 0.0, 0.0)
    
    # Simulate a LaserScan with 7 beams spanning an arc from -0.3 to +0.3 radians.
    angle_min = -0.3          # radians
    angle_increment = 0.1       # radians
    ranges1 = [5.0, 5.0, 4.0, 4.0, 4.5, 5.0, 5.0]
    scan1 = DummyLaserScan(angle_min, angle_increment, ranges1)
    
    print("Simulating first robot pose (0, 0, 0) with scan1...")
    ogm.update(robot_pose1, scan1)
    
    # ---- Second Simulation: Robot moves to (0.5, 0.0) with a slight rotation (0.1 rad) ----
    robot_pose2 = (0.5, 0.0, 0.1)
    # In this simulation, the distances are slightly reduced, simulating that obstacles are closer.
    ranges2 = [4.5, 4.5, 3.5, 3.5, 4.0, 4.5, 4.5]
    scan2 = DummyLaserScan(angle_min, angle_increment, ranges2)
    
    print("Simulating second robot pose (0.5, 0.0, 0.1) with scan2...")
    ogm.update(robot_pose2, scan2)
    
    # Get the final occupancy grid probability map.
    prob_map = ogm.get_probability_map()
    
    # Print a subset (center 20x20 region) of the grid.
    center_i = height // 2
    center_j = width // 2
    subset = prob_map[center_i-10:center_i+10, center_j-10:center_j+10]
    
    print("\nFinal occupancy grid probability map (center 20x20 region):")
    print(subset)

# Run the simulation test.
if __name__ == '__main__':
    simulate_lidar_and_robot_poses()
