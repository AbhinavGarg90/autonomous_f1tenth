import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Int8MultiArray, MultiArrayDimension
from geometry_msgs.msg import Pose, Point, Quaternion
from nav_msgs.msg import OccupancyGrid, MapMetaData


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

        # --- NEW: set up map publisher ---
        self.map_pub = rospy.Publisher('map', OccupancyGrid, queue_size=1)
        self.map_msg = OccupancyGrid()
        self.map_msg.header.frame_id = "map"
        # Fill in the meta‐data
        self.map_msg.info = MapMetaData()
        self.map_msg.info.resolution = self.resolution
        self.map_msg.info.width = self.width_gc
        self.map_msg.info.height = self.height_gc

        # Define the origin pose of the map in the "map" frame
        self.map_msg.info.origin = Pose(
            position=Point(x=-self.origin_x_wc,  # map origin (world coords)
                           y=-self.origin_y_wc,
                           z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        )

    def world_to_map(self, x_wc, y_wc):
        # row = y,  col = x
        row = int((y_wc + self.origin_y_wc) / self.resolution)
        col = int((x_wc + self.origin_x_wc) / self.resolution)
        
        # clamp to [0 .. size-1]
        row = max(0, min(self.height_gc - 1, row))
        col = max(0, min(self.width_gc  - 1, col))

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
        angles = scan[0]
        ranges = scan[1]
        num_beams = len(angles)
        rx, ry, rtheta = robot_pose
        assert len(angles) == len(ranges)

        for i in range(num_beams):

            r = ranges[i]

            # Convert the beam angle from the sensor frame to the world frame by adding the robot's orientation
            beam_angle = rtheta + angles[i]
            hit = True

            # Compute end point of the beam in world coordinates
            x_end = rx + r * math.cos(beam_angle)
            y_end = ry + r * math.sin(beam_angle)

            r_row, r_col = self.world_to_map(rx, ry)        # robot
            e_row, e_col = self.world_to_map(x_end, y_end)  # end of laser beam
    
            # Get cells along the beam using Bresenham's algorithm
            cells = self.bresenham2D(r_row, r_col, e_row, e_col) 

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
    
    # def update_map(self, x, y, theta, scan_msg):
    #     self.robot_pose = (x, y, theta)
    #     self.update(self.robot_pose, scan_msg)
    def update_map(self, x, y, theta, scan_msg):
        """Call this from your subscriber callback to update & immediately publish."""
        self.robot_pose = (x, y, theta)
        self.update(self.robot_pose, scan_msg)


def main():
    rospy.init_node('occupancy_grid_mapping_node', anonymous=True)
    ogm_node = OccupancyGridMapping()
    rospy.loginfo("Occupancy Grid Mapping Node Started")
    rospy.spin()

if __name__ == '__main__':
    main()
    

    
