import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Int8MultiArray, MultiArrayDimension


# All indices are either world coordinates (_wc) or grid grid coordiantes (_gc)
class OccupancyGridMapping:
    def __init__(self, width_wc=60, height_wc=60, resolution=0.1, origin_x_wc=30, origin_y_wc=30):

        self.width_gc = int(width_wc / resolution)
        self.height_gc = int(height_wc / resolution)
        self.resolution = resolution # world to grid

        self.origin_x_gc = int(origin_x_wc / resolution) 
        self.origin_y_gc = int(origin_y_wc / resolution)

        self.origin_x_wc = origin_x_wc
        self.origin_y_wc = origin_y_wc

        self.x_min_wc = -origin_x_wc
        self.x_max_wc =  width_wc - origin_x_wc
        self.y_min_wc = -origin_y_wc
        self.y_max_wc =  height_wc - origin_y_wc

        # Initialize the grid in log odds (0 means 50% probability)
        self.log_odds = np.zeros((self.height_gc, self.width_gc), dtype=np.float32)

        # Log odds update parameters:
        self.l_occ = math.log(0.7 / 0.3)   # update for an occupied cell = p/1-p 0.847
        self.l_free = math.log(0.3 / 0.7)  # update for a free cell = p/1-p -0.847
        self.l_min = -5.0 
        self.l_max =  5.0  

    def world_to_map(self, x_wc, y_wc):
        # row = y,  col = x
        row = int((y_wc + self.origin_y_wc) / self.resolution)
        col = int((x_wc + self.origin_x_wc) / self.resolution)
        # assert row >=0 and col >=0 
        return row, col


    def bresenham2D(self, r0, c0, r1, c1):
        """
        Bresenham's Line Algorithm in terms of row/col indices.
        Returns a list of (row, col) cells along the line from (r0, c0) to (r1, c1).
        """
        r0 = int(round(r0))
        c0 = int(round(c0))
        r1 = int(round(r1))
        c1 = int(round(c1))

        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1

        err = dr - dc
        r = r0
        c = c0

        cells = []

        while True:
            cells.append((r, c))        #  appends last cell too 
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dc:       # move in row (vertical)
                err -= dc
                r += sr
            if e2 < dr:        # move in column (horizontal)
                err += dr
                c += sc        # if both moves happen → diagonal

        return cells

    def update(self, robot_pose, scan):
        """ Update the occupancy grid using the robot's current pose and a LIDAR scan. """
        rx, ry, rtheta = robot_pose

        # starting angle
        angle_min = scan.angle_min
        # Retrieve the angular difference between consecutive laser beams in the scan
        angle_increment = scan.angle_increment
        # Convert the list of range measurements from the LaserScan into a NumPy array for easier manipulation
        ranges = np.array(scan.ranges)
        # Determine the number of laser beams in the scan data
        num_beams = len(ranges)
        # Compute an array of angles corresponding to each laser beam in the scan,
        angles = angle_min + np.arange(num_beams) * angle_increment


        for i in range(num_beams):

            r = ranges[i]

            if np.isnan(r) or r < scan.range_min:  # INVALID : discard beam for now
                continue

            if r > scan.range_max or np.isinf(r):
                r   = scan.range_max               # cast to max range
                hit = False                       
            else:
                r   = r                           # valid hit
                hit = True

            # Convert the beam angle from the sensor frame to the world frame by adding the robot's orientation
            beam_angle = rtheta + angles[i]

            # Compute end point of the beam in world coordinates
            x_end = rx + r * math.cos(beam_angle)
            y_end = ry + r * math.sin(beam_angle)

            r_row, r_col = self.world_to_map(rx, ry)        # robot
            e_row, e_col = self.world_to_map(x_end, y_end)  # end of laser beam
    
            # Get cells along the beam using Bresenham's algorithm
            cells = self.bresenham2D(r_row, r_col, e_row, e_col) 
            # Update all cells along the beam as free (except the final cell)
            # for row, col in (cells[:-1] if hit else cells):
            #     if 0 <= row < self.height_gc and 0 <= col < self.width_gc:
            #         self.log_odds[row, col] = max(self.l_min, self.log_odds[row, col] + self.l_free)

            # # Update the hit cell (last cell) as occupied
            # if hit:
            #     row, col = cells[-1]
            #     if 0 <= row < self.height_gc and 0 <= col < self.width_gc:
            #         self.log_odds[row, col] = min(self.l_max, self.log_odds[row, col] + self.l_occ)

            last_valid_idx = -1          # keep track of the last cell that is inside

            for k, (row, col) in enumerate(cells):
                inside = (0 <= row < self.height_gc) and (0 <= col < self.width_gc)
                if not inside:
                    break                # stop processing laser as it exits the map

                # mark free space for every cell except the last if hit
                if not (hit and k == len(cells) - 1):
                    self.log_odds[row, col] = max(self.l_min,
                                                self.log_odds[row, col] + self.l_free)
                last_valid_idx = k

            if hit and last_valid_idx == len(cells) - 1:
                row, col = cells[last_valid_idx]
                self.log_odds[row, col] = min(self.l_max, self.log_odds[row, col] + self.l_occ)

    def get_probability_map(self):
        """ Convert the log odds grid to a probability map scaled to [0, 100] (integer values)."""
        prob = 1 - 1 / (1 + np.exp(self.log_odds))
        prob_scaled = (prob * 100).astype(np.int8)  # Scale to [0, 100]
        binary = (prob > 0.5).astype(np.int8)
        return prob_scaled
    
    def update_map(self, x, y, theta, scan_msg):
        self.robot_pose = (x, y, theta)
        self.update(self.robot_pose, scan_msg)


def main():
    rospy.init_node('occupancy_grid_mapping_node', anonymous=True)
    ogm_node = OccupancyGridMapping()
    rospy.loginfo("Occupancy Grid Mapping Node Started")
    rospy.spin()

if __name__ == '__main__':
    main()
    

    
