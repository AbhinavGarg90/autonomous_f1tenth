#!/usr/bin/env python
# This is a ROS node script written in Python to perform occupancy grid mapping using log-odds updates.
# It subscribes to ICP-updated robot pose data (PoseStamped) on '/icp_pose' and raw LaserScan data on '/scan',
# then uses a ray tracing algorithm (Bresenham's) to update a grid map, and finally publishes the map for visualization.

import rospy                      # Import ROS Python API for node creation and communication.
import numpy as np                # Import NumPy for efficient numerical operations.
import math                       # Import math module for mathematical functions.
from sensor_msgs.msg import LaserScan  # Import LaserScan message definition.
from geometry_msgs.msg import PoseStamped, Quaternion  # Import PoseStamped and Quaternion for pose representation.
from nav_msgs.msg import OccupancyGrid, MapMetaData    # Import OccupancyGrid and MapMetaData for map publishing.
from std_msgs.msg import Int8MultiArray, MultiArrayDimension   # Import Int8MultiArray for publishing grid data in a multi-array format.


# ---------------------------------------------------------------------
# OccupancyGridMapping class: Handles the grid creation, update using log-odds, 
# and ray tracing (Bresenham's algorithm) to mark free/occupied cells.
# ---------------------------------------------------------------------
class OccupancyGridMapping:
    # The constructor sets up the grid dimensions and initializes key parameters.
    def __init__(self, width, height, resolution, origin_x, origin_y):
        # Store the total width (number of columns) of the grid.
        self.width = width
        # Store the total height (number of rows) of the grid.
        self.height = height
        # Resolution: size of each grid cell in meters.
        self.resolution = resolution
        # World coordinate for the grid origin x (grid cell [0,0] corresponds to this x-coordinate).
        self.origin_x = origin_x
        # World coordinate for the grid origin y (grid cell [0,0] corresponds to this y-coordinate).
        self.origin_y = origin_y

        # Initialize a 2D NumPy array (grid) for the log-odds values of each cell.
        # All cells are initially 0 which corresponds to 50% probability (uncertainty).
        self.log_odds = np.zeros((height, width), dtype=np.float32)

        # Positive update for an occupied cell when a laser beam ends in it.
        self.l_occ = math.log(0.7 / 0.3)  # This equals approximately 0.847. The higher the value, the greater the confidence.
        # Negative update for a free cell when a laser beam passes through it.
        self.l_free = math.log(0.3 / 0.7)  # Approximately -0.847, which will decrease the likelihood of occupancy.
        # Define a minimum limit for log-odds to prevent over-certainty.
        self.l_min = -5.0
        # Define a maximum limit for log-odds to cap occupancy certainty.
        self.l_max = 5.0

        # Create a ROS publisher to publish the grid in the Int8MultiArray format on the '/occupancy_grid' topic.
        self.array_pub = rospy.Publisher('/occupancy_grid', Int8MultiArray, queue_size=1)

    # This helper function converts real-world (x,y) coordinates to indices in the grid map.
    def world_to_map(self, x, y):
        # Compute the column (j) index by subtracting the origin and dividing by the resolution.
        j = int((x - self.origin_x) / self.resolution)
        # Compute the row (i) index similarly for the y coordinate.
        i = int((y - self.origin_y) / self.resolution)
        return i, j   # Return as (row, column) indices.

    # Bresenham's line algorithm computes the grid cells intersected by a line from (x0, y0) to (x1, y1).
    def bresenham2D(self, x0, y0, x1, y1):
        # Round the start and end coordinates to the nearest integer (grid indices).
        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x1))
        y1 = int(round(y1))
        # Compute absolute differences in x and y coordinates.
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        # Determine step directions (1 for positive direction, -1 for negative).
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        # Initialize the error term for the algorithm.
        err = dx - dy

        # Initialize an empty list to hold the cells along the computed line.
        cells = []
        # Continue until the end of the line is reached.
        while True:
            cells.append((y0, x0))  # Append the current cell as (row, column).
            # Break out of the loop if we've reached the target cell.
            if x0 == x1 and y0 == y1:
                break
            # Compute doubled error.
            e2 = 2 * err
            # If error is sufficiently large in one direction, step in x.
            if e2 > -dy:
                err -= dy
                x0 += sx
            # Likewise, if error is sufficiently large in the other direction, step in y.
            if e2 < dx:
                err += dx
                y0 += sy
        return cells  # Return the list of grid cells along the line.

    # Update the log-odds occupancy grid using the robot's current pose and the LaserScan.
    def update(self, robot_pose, scan):
        # Unpack robot pose: (x, y, theta) - position and orientation.
        rx, ry, rtheta = robot_pose

        # Retrieve the starting angle of the scan from the message.
        angle_min = scan.angle_min
        # Retrieve the angular resolution (increment) of the scan.
        angle_increment = scan.angle_increment
        # Convert the LaserScan ranges into a NumPy array for vectorized operations.
        ranges = np.array(scan.ranges)
        # Determine the total number of laser beams in the scan.
        num_beams = len(ranges)
        # Compute the corresponding angle for each beam.
        angles = angle_min + np.arange(num_beams) * angle_increment

        # Process each beam in the LaserScan.
        for i in range(num_beams):
            # Get the range (distance) for the current laser beam.
            r = ranges[i]
            # Skip beams with invalid values (infinite or NaN).
            if np.isinf(r) or np.isnan(r):
                continue

            # Calculate the beam angle in the world frame by adding the robot's orientation.
            beam_angle = rtheta + angles[i]
            # Compute the world coordinates (x_end, y_end) where the beam ends.
            x_end = rx + r * math.cos(beam_angle)
            y_end = ry + r * math.sin(beam_angle)

            # Convert the robot's starting position to grid indices.
            robot_i, robot_j = self.world_to_map(rx, ry)
            # Convert the beam's endpoint to grid indices.
            end_i, end_j = self.world_to_map(x_end, y_end)

            # Use Bresenham's algorithm to determine which cells are intersected by the beam.
            cells = self.bresenham2D(robot_j, robot_i, end_j, end_i)

            # For all cells along the beam (except the last one), mark them as free space.
            for (cell_i, cell_j) in cells[:-1]:
                # Only update if the cell is within grid boundaries.
                if 0 <= cell_i < self.height and 0 <= cell_j < self.width:
                    # Update log-odds: add the negative value for free space.
                    self.log_odds[cell_i, cell_j] += self.l_free
                    # Clamp the value so it does not go below the defined minimum.
                    self.log_odds[cell_i, cell_j] = max(self.l_min, self.log_odds[cell_i, cell_j])

            # For the final cell (where the beam ended), mark it as occupied.
            if cells:
                cell_i, cell_j = cells[-1]
                # Update only if the cell lies within valid bounds.
                if 0 <= cell_i < self.height and 0 <= cell_j < self.width:
                    # Update the log-odds to reflect occupancy.
                    self.log_odds[cell_i, cell_j] += self.l_occ
                    # Clamp the updated log-odds to the defined maximum.
                    self.log_odds[cell_i, cell_j] = min(self.l_max, self.log_odds[cell_i, cell_j])

    # This function converts the log-odds values in the grid into a probability map scaled from 0 to 100.
    def get_probability_map(self):
        # Convert log-odds to probabilities using the logistic function.
        # The formula used is: p = 1 - 1 / (1 + exp(log_odds))
        prob = 1 - 1 / (1 + np.exp(self.log_odds))
        # Multiply probabilities by 100 and cast to 8-bit integers.
        prob_scaled = (prob * 100).astype(np.int8)
        return prob_scaled  # Return the processed probability map.


# ---------------------------------------------------------------------------------
# OccupancyGridMappingNode class: Integrates the mapping module with ROS by subscribing
# to the ICP-corrected pose and LaserScan data, then publishing the updated maps.
# ---------------------------------------------------------------------------------
class OccupancyGridMappingNode:
    def __init__(self):
        # Retrieve grid resolution from ROS parameters or use default value 0.1 meters.
        self.resolution = rospy.get_param("~resolution", 0.1)
        # Retrieve grid width (number of cells along x) from parameters or default to 5000.
        self.width = rospy.get_param("~grid_width", 5000)
        # Retrieve grid height (number of cells along y) from parameters or default to 5000.
        self.height = rospy.get_param("~grid_height", 5000)
        # Retrieve the world x-coordinate corresponding to the grid's [0,0] cell.
        self.origin_x = rospy.get_param("~origin_x", -125.0)
        # Retrieve the world y-coordinate corresponding to the grid's [0,0] cell.
        self.origin_y = rospy.get_param("~origin_y", -125.0)

        # Create an instance of the occupancy grid mapping module with the defined parameters.
        self.ogm = OccupancyGridMapping(self.width, self.height, self.resolution,
                                        self.origin_x, self.origin_y)
        # Initialize a variable to hold the most recent robot pose from the ICP module.
        self.robot_pose = None

        # Subscribe to the ICP-corrected pose published on '/icp_pose'.
        rospy.Subscriber('/icp_pose', PoseStamped, self.pose_callback)
        # Subscribe to the LaserScan data on '/scan'.
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        # Create a publisher for the OccupancyGrid message used by RViz (nav_msgs/OccupancyGrid).
        self.map_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=1)
        # Create another publisher for the Int8MultiArray message for alternative visualization.
        self.array_pub = rospy.Publisher('/occupancy_grid', Int8MultiArray, queue_size=1)

        # Set up an OccupancyGrid message with metadata so RViz knows how to interpret the map.
        self.map_msg = OccupancyGrid()
        # Set the coordinate frame for the map (must match the frame in RViz, e.g., "map").
        self.map_msg.header.frame_id = "map"
        # Set up the map metadata including resolution, width, height, and origin.
        self.map_msg.info = MapMetaData()
        self.map_msg.info.resolution = self.resolution
        self.map_msg.info.width = self.width
        self.map_msg.info.height = self.height
        self.map_msg.info.origin.position.x = self.origin_x
        self.map_msg.info.origin.position.y = self.origin_y

    # Callback function executed whenever a new ICP pose is published.
    def pose_callback(self, msg):
        # Extract the x-coordinate from the pose message.
        x = msg.pose.position.x
        # Extract the y-coordinate from the pose message.
        y = msg.pose.position.y
        # Convert the quaternion orientation into a yaw angle.
        theta = self.quaternion_to_yaw(msg.pose.orientation)
        # Update the stored robot pose as a tuple (x, y, theta).
        self.robot_pose = (x, y, theta)
        # Log receipt of the ICP pose with details.
        rospy.loginfo("Received ICP Pose: x: {:.3f}, y: {:.3f}, theta: {:.3f}".format(x, y, theta))

    # Helper function to convert a quaternion into a yaw (rotation about the z-axis).
    def quaternion_to_yaw(self, q):
        # Compute intermediate values for conversion.
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        # Calculate and return the yaw angle in radians using the arctan2 function.
        return math.atan2(siny_cosp, cosy_cosp)

    # Callback function executed when a new LaserScan message is received.
    def scan_callback(self, scan_msg):
        # Check if the robot pose is available (from ICP); if not, skip grid update.
        if self.robot_pose is None:
            return
        # Update the occupancy grid using the latest pose and scan data.
        self.ogm.update(self.robot_pose, scan_msg)
        # Publish the updated map using the timestamp from the LaserScan message.
        self.publish_map(scan_msg.header.stamp)

    # Function to publish the updated occupancy grid in two different message formats.
    def publish_map(self, stamp):
        # Set the header timestamp for the OccupancyGrid message.
        self.map_msg.header.stamp = stamp
        # Retrieve the probability map (0-100 occupancy values) from the mapping module.
        prob_map = self.ogm.get_probability_map()
        # Flatten the 2D probability map to a 1D list in row-major order for publishing.
        self.map_msg.data = prob_map.flatten().tolist()
        # Publish the OccupancyGrid message to the '/map' topic.
        self.map_pub.publish(self.map_msg)

        # Create a new Int8MultiArray message to publish the grid in an alternative format.
        array_msg = Int8MultiArray()
        # Create and configure the first dimension for rows.
        dim1 = MultiArrayDimension()
        dim1.label = "rows"                               # Label for clarity.
        dim1.size = prob_map.shape[0]                     # Number of rows.
        dim1.stride = prob_map.shape[0] * prob_map.shape[1]  # Total number of cells.
        # Create and configure the second dimension for columns.
        dim2 = MultiArrayDimension()
        dim2.label = "cols"                               # Label for clarity.
        dim2.size = prob_map.shape[1]                     # Number of columns.
        dim2.stride = prob_map.shape[1]                   # Number of cells per row.
        # Set the layout dimensions of the multi-array.
        array_msg.layout.dim = [dim1, dim2]
        # Indicate that the data starts at index 0.
        array_msg.layout.data_offset = 0
        # Flatten the probability map (2D array) into a 1D list to be stored in the message.
        array_msg.data = prob_map.flatten().tolist()
        # Publish the multi-array message to the '/occupancy_grid' topic.
        self.array_pub.publish(array_msg)


# Main function: Initializes the ROS node and starts the mapping process.
def main():
    # Initialize this script as a ROS node named 'occupancy_grid_mapping_node'.
    rospy.init_node('occupancy_grid_mapping_node', anonymous=True)
    # Create an instance of our mapping node.
    ogm_node = OccupancyGridMappingNode()
    # Log startup information.
    rospy.loginfo("Occupancy Grid Mapping Node with ICP integration started")
    # Keep the node running until it is shut down.
    rospy.spin()


# If this script is executed (instead of imported as a module), run the main function.
if __name__ == '__main__':
    main()
