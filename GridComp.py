import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Int8MultiArray, MultiArrayDimension


class OccupancyGridMapping:
    def __init__(self, width, height, resolution, origin_x, origin_y):

        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y

        # Initialize the grid in log odds (0 means 50% probability)
        self.log_odds = np.zeros((height, width), dtype=np.float32)

        # Log odds update parameters:
        self.l_occ = math.log(0.7 / 0.3)   # update for an occupied cell = p/1-p 0.847
        self.l_free = math.log(0.3 / 0.7)  # update for a free cell = p/1-p -0.847
        self.l_min = -5.0 
        self.l_max =  5.0  

        # Initialize the occupancy grid publisher
        # This will publish the occupancy grid as an Int8MultiArray message.
        self.array_pub = rospy.Publisher('/occupancy_grid', Int8MultiArray, queue_size=1)

        # Grid parameters 
        self.resolution = rospy.get_param("~resolution", 0.1)  # each cell is 0.1 m
        self.width = rospy.get_param("~grid_width", 200)
        self.height = rospy.get_param("~grid_height", 200)
        # Move origin so that (0,0) of the world is near the center
        self.origin_x = rospy.get_param("~origin_x", -10.0)
        self.origin_y = rospy.get_param("~origin_y", -10.0)
        
        self.robot_pose = None  

        # Subscribers
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        # Create a publisher for the OccupancyGrid message used by RViz (nav_msgs/OccupancyGrid).
        self.map_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=1)
        # Create another publisher for the Int8MultiArray message for alternative visualization.
        self.array_pub = rospy.Publisher('/occupancy_grid', Int8MultiArray, queue_size=1)

        # Set up the OccupancyGrid message meta data
        self.map_msg = OccupancyGrid()
        self.map_msg.header.frame_id = "map"
        self.map_msg.info = MapMetaData()
        self.map_msg.info.resolution = self.resolution
        self.map_msg.info.width = self.width
        self.map_msg.info.height = self.height
        # The origin of the map is defined as a Pose (we only set x and y here)
        self.map_msg.info.origin.position.x = self.origin_x
        self.map_msg.info.origin.position.y = self.origin_y

    def world_to_map(self, x, y): # double check the column and row order
        """ Convert world coordinates (x, y) into grid indices (i, j). """
        j = int((x - self.origin_x) / self.resolution)
        i = int((y - self.origin_y) / self.resolution)
        return i, j

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
            cells.append((r, c))
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

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
            # can also limit to max of the lidar
            if np.isinf(r) or np.isnan(r): 
                continue

            # Convert the beam angle from the sensor frame to the world frame by adding the robot's orientation
            beam_angle = rtheta + angles[i]

            # Compute end point of the beam in world coordinates
            x_end = rx + r * math.cos(beam_angle)
            y_end = ry + r * math.sin(beam_angle)

            # Convert robot and beam endpoint to grid indices
            robot_i, robot_j = self.world_to_map(rx, ry)
            end_i, end_j = self.world_to_map(x_end, y_end)

            # Get cells along the beam using Bresenham's algorithm
            cells = self.bresenham2D(robot_j, robot_i, end_j, end_i)

            # Update all cells along the beam as free (except the final cell)
            for (cell_i, cell_j) in cells[:-1]:
                if 0 <= cell_i < self.height and 0 <= cell_j < self.width:
                    # This decreases the log odds, moving the cell's occupancy probability closer to 0 (free)
                    self.log_odds[cell_i, cell_j] += self.l_free
                    self.log_odds[cell_i, cell_j] = max(self.l_min, self.log_odds[cell_i, cell_j])

            # Update the hit cell (last cell) as occupied
            if cells:
                cell_i, cell_j = cells[-1]
                if 0 <= cell_i < self.height and 0 <= cell_j < self.width:
                    # This increases the log odds, pushing the occupancy probability closer to 1 (occupied)
                    self.log_odds[cell_i, cell_j] += self.l_occ
                    self.log_odds[cell_i, cell_j] = min(self.l_max, self.log_odds[cell_i, cell_j])

    def get_probability_map(self):
        """ Convert the log odds grid to a probability map scaled to [0, 100] (integer values)."""
        prob = 1 - 1 / (1 + np.exp(self.log_odds))
        prob_scaled = (prob * 100).astype(np.int8)  # Scale to [0, 100]
        binary = (prob > 0.5).astype(np.int8)
        return prob_scaled
    
    def update_map(self, x, y, theta, scan_msg):
        self.robot_pose = (x, y, theta)
        self.ogm.update(self.robot_pose, scan_msg)
        self.publish_map(scan_msg.header.stamp)

    def publish_map(self, stamp):
        """ Publish the occupancy grid as a nav_msgs/OccupancyGrid message."""
        self.map_msg.header.stamp = stamp
        # Get the probability map and flatten to a list (row-major order)
        prob_map = self.ogm.get_probability_map()
        self.map_msg.data = prob_map.flatten().tolist()
        self.map_pub.publish(self.map_msg)

        # Create an Int8MultiArray message object to hold the 2D occupancy grid map.
        array_msg = Int8MultiArray()

        # Define the first dimension (rows) for the 2D layout.
        dim1 = MultiArrayDimension()
        dim1.label = "rows"                     
        dim1.size = prob_map.shape[0]         
        # The stride is the total number of elements in the map, which is rows*cols (i.e., jump to the next "row" block).
        dim1.stride = prob_map.shape[0] * prob_map.shape[1]  

        # Define the second dimension (cols) for the 2D layout.
        dim2 = MultiArrayDimension()
        dim2.label = "cols"                    
        dim2.size = prob_map.shape[1]
        # The stride here is the number of elements per row (i.e., the number of columns).         
        dim2.stride = prob_map.shape[1]        

        # Set the layout of the multi-array by assigning the dimensions in order (first rows, then columns).
        array_msg.layout.dim = [dim1, dim2]

        # Since our data begins at the start of the flattened array, we set it to 0.
        array_msg.layout.data_offset = 0

        # Convert the 2D probability map (a NumPy array) into a 1D list using row-major order.
        array_msg.data = prob_map.flatten().tolist()

        # Publish the complete Int8MultiArray message which now contains the 2D occupancy grid in a flattened format.
        self.array_pub.publish(array_msg)

def main():
    rospy.init_node('occupancy_grid_mapping_node', anonymous=True)
    ogm_node = OccupancyGridMapping()
    rospy.loginfo("Occupancy Grid Mapping Node Started")
    rospy.spin()

if __name__ == '__main__':
    main()
    

    
