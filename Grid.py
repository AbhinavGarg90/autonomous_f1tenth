# This ROS node implements an occupancy grid mapping module.
# It uses log‐odds updates and ray‐tracing (Bresenham) to update a large grid
# based on LaserScan data and the robot's pose. The pose is assumed to be provided
# by another module (e.g. an ICP-SLAM or EKF-SLAM node).


import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Int8MultiArray, MultiArrayDimension

class OccupancyGridMapping:
    def __init__(self, width, height, resolution, origin_x, origin_y):
        """ 
        Initializing the occupancy grid mapping module. width, height: 5000 x 5000 cells.
        origin_x, origin_y: World coordinates corresponding to grid[0,0]
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y

        # Initialize the grid in log odds (0 means 50% probability)
        self.log_odds = np.zeros((height, width), dtype=np.float32)

        # Log odds update parameters:
        self.l_occ = math.log(0.7 / 0.3)   # update for an occupied cell = p/1-p 0.847
        self.l_free = math.log(0.3 / 0.7)  # update for a free cell
        self.l_min = -5.0 
        self.l_max =  5.0  

    def world_to_map(self, x, y): # double check the column and row order
        """ Convert world coordinates (x, y) into grid indices (i, j). """
        j = int((x - self.origin_x) / self.resolution)
        i = int((y - self.origin_y) / self.resolution)
        return i, j

    def bresenham2D(self, x0, y0, x1, y1):
        """ Bresenham's Line Algorithm: Compute the grid cells along a line from (x0, y0) to (x1, y1)."""
        # Initialize starting point
        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x1))
        y1 = int(round(y1))
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        # to determine if we move forwards or backwards in x and y
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        cells = []
        while True:
            cells.append((y0, x0))  # (row, col)
            if x0 == x1 and y0 == y1:
                break
            # The doubled error (e2) helps decide if you need to take an extra step sideways or up/down 
            # to stay as close as possible to the ideal straight line between two points
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return cells

    def update(self, robot_pose, scan):
        """ Update the occupancy grid using the robot's current pose and a LIDAR scan. """
        rx, ry, rtheta = robot_pose

        # Retrieve the minimum angle of the scan (the starting angle of the scan arc)
        angle_min = scan.angle_min
        # Retrieve the angular difference between consecutive laser beams in the scan
        angle_increment = scan.angle_increment
        # Convert the list of range measurements from the LaserScan into a NumPy array for easier manipulation
        ranges = np.array(scan.ranges)
        # Determine the number of laser beams in the scan data
        num_beams = len(ranges)
        # Compute an array of angles corresponding to each laser beam in the scan,
        # starting at angle_min and increasing by angle_increment for each beam.
        angles = angle_min + np.arange(num_beams) * angle_increment

        # Iterate through each laser beam in the scan
        for i in range(num_beams):
            # Get the range measurement for the current beam
            r = ranges[i]
            # Skip invalid measurements
            if np.isinf(r) or np.isnan(r): # can also limit to max of the lidar
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
                    self.log_odds[cell_i, cell_j] += self.l_free
                    # Clamp the value
                    self.log_odds[cell_i, cell_j] = max(self.l_min, self.log_odds[cell_i, cell_j])

            # Update the hit cell (last cell) as occupied
            if cells:
                cell_i, cell_j = cells[-1]
                if 0 <= cell_i < self.height and 0 <= cell_j < self.width:
                    self.log_odds[cell_i, cell_j] += self.l_occ
                    self.log_odds[cell_i, cell_j] = min(self.l_max, self.log_odds[cell_i, cell_j])

    def get_probability_map(self):
        """ Convert the log odds grid to a probability map scaled to [0, 100] (integer values)."""
        prob = 1 - 1 / (1 + np.exp(self.log_odds))
        prob_scaled = (prob * 100).astype(np.int8)
        return prob_scaled

# logic behind log odds:
# log odds = log(p/(1-p))
# where p is the probability of occupancy
# p = 0.5 -> log odds = 0
# p = 0.7 -> log odds = log(0.7/(1-0.7)) = log(2.333) = 0.847
# p = 0.3 -> log odds = log(0.3/(1-0.3)) = log(0.428) = -0.847
# The log odds are clamped to a range of [-5, 5] to avoid overflow and underflow issues.
# The log odds are updated based on the LIDAR scan data and the robot's pose.
# The occupancy grid is represented as a 2D numpy array of log odds values.
# The occupancy grid is updated using the Bresenham's line algorithm to determine the cells
# When a laser beam passes through a cell we update that cell by adding a value self.l_free (which is negative).
# When the laser beam hits an obstacle we update that cell by adding a value self.l_occ (which is positive).
# The occupancy grid is then converted to a probability map using the logistic function.
# The probability map is scaled to [0, 100] for the OccupancyGrid message.

class OccupancyGridMappingNode:
    def __init__(self):
        self.resolution = rospy.get_param("~resolution", 0.1)  # meters 
        self.width = rospy.get_param("~grid_width", 5000)
        self.height = rospy.get_param("~grid_height", 5000)
        # need to figure out the origin_x and origin_y
        # currently set so that the robot is at (0,0) in the center of the map
        self.origin_x = rospy.get_param("~origin_x", -125.0)
        self.origin_y = rospy.get_param("~origin_y", -125.0)

        # Create the occupancy grid mapping object.
        self.ogm = OccupancyGridMapping(self.width, self.height, self.resolution, self.origin_x, self.origin_y)

        self.robot_pose = None  # Will be updated from the /robot_pose topic - from RISHI

        # Subscribers
        rospy.Subscriber('/robot_pose', PoseStamped, self.pose_callback)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        # Publisher for the occupancy grid (nav_msgs/OccupancyGrid)
        self.map_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=1)

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

    def pose_callback(self, msg):
        """
        Update the current robot pose from a PoseStamped message. - RISHI needs to publish this
        """
        x = msg.pose.position.x
        y = msg.pose.position.y
        theta = self.quaternion_to_yaw(msg.pose.orientation)
        self.robot_pose = (x, y, theta)

    def quaternion_to_yaw(self, q):
        """
        Convert a quaternion into a yaw angle.
        """
        # Using the standard conversion formula
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, scan_msg):
        """
        Process an incoming LaserScan message.
        Update the occupancy grid based on the latest robot pose and LIDAR data.
        """
        if self.robot_pose is None:
            return  # We need a valid pose to update the grid

        # Update the occupancy grid using the current scan and pose
        self.ogm.update(self.robot_pose, scan_msg)
        # Publish the updated map
        self.publish_map(scan_msg.header.stamp)

    def publish_map(self, stamp):
        """
        Publish the occupancy grid as a nav_msgs/OccupancyGrid message.
        """
        self.map_msg.header.stamp = stamp
        # Get the probability map and flatten to a list (row-major order)
        prob_map = self.ogm.get_probability_map()
        self.map_msg.data = prob_map.flatten().tolist()
        self.map_pub.publish(self.map_msg)

        # Create an Int8MultiArray message object to hold the 2D occupancy grid map.
        array_msg = Int8MultiArray()

        # Define the first dimension (rows) for the 2D layout.
        dim1 = MultiArrayDimension()
        dim1.label = "rows"                     # Label the dimension as "rows" for clarity.
        dim1.size = prob_map.shape[0]           # The size of this dimension is the number of rows in the probability map.
        # The stride is the total number of elements in the map, which is rows*cols (i.e., jump to the next "row" block).
        dim1.stride = prob_map.shape[0] * prob_map.shape[1]  

        # Define the second dimension (cols) for the 2D layout.
        dim2 = MultiArrayDimension()
        dim2.label = "cols"                     # Label the dimension as "cols" for clarity.
        dim2.size = prob_map.shape[1]           # The size of this dimension is the number of columns in the probability map.
        dim2.stride = prob_map.shape[1]         # The stride here is the number of elements per row (i.e., the number of columns).

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
    ogm_node = OccupancyGridMappingNode()
    rospy.loginfo("Occupancy Grid Mapping Node Started")
    rospy.spin()

if __name__ == '__main__':
    main()


# def int8multiarray_to_2d(array_msg):
#     # Extract the number of rows 
#     rows = array_msg.layout.dim[0].size
#     # Extract the number of columns
#     cols = array_msg.layout.dim[1].size
#     # Convert into a NumPy array.
#     flat_data = np.array(array_msg.data, dtype=np.int8)
#     # Reshape the flat array into a 2D array
#     grid = flat_data.reshape((rows, cols))
#     return grid