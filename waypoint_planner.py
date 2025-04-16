
import rospy
import numpy as np
import math
import heapq  # Library for implementing the priority queue (min-heap) used in A*
from nav_msgs.msg import OccupancyGrid, Path  # ROS message types for map and path
from geometry_msgs.msg import PoseStamped, Quaternion  # ROS message type for robot pose and goal pose
from tf.transformations import euler_from_quaternion, quaternion_from_euler  # ROS library utilities for converting between Euler angles (like yaw) and Quaternions

# ----------------------- Utility Functions -----------------------

def world_to_map(x, y, origin_x, origin_y, resolution):
    """
    Converts world coordinates (in meters, e.g., from localization) into
    discrete grid cell indices (row, col) based on the map's metadata.
    Required for checking cells in the occupancy grid.
    """
    # Calculate column index (j) based on x coordinate
    col = int((x - origin_x) / resolution)
    # Calculate row index (i) based on y coordinate
    row = int((y - origin_y) / resolution)
    return row, col

def normalize_angle(angle):
    """
    Normalizes an angle to the range [-pi, pi] radians.
    This ensures angles are compared consistently, avoiding issues with wrap-around
    (e.g., 3.15 radians vs -3.13 radians being recognized as close).
    Uses atan2(sin(angle), cos(angle)) which robustly handles all quadrants.
    """
    return math.atan2(math.sin(angle), math.cos(angle))

# ----------------------- Hybrid A* Path Planner Class -----------------------

class HybridAStarPlanner:
    """
    Implements the core Hybrid A* search algorithm logic.
    It handles state expansion, collision checking, cost calculation, and path reconstruction.
    """
    def __init__(self, wheelbase, step_size, max_steering_angle, num_steering_angles,
                 obstacle_threshold=65, robot_length=0.4, robot_width=0.3,
                 goal_xy_tolerance=0.2, goal_theta_tolerance=0.1,
                 num_angle_bins=72, heuristic_weight=1.0):
        """
        Initializes the Hybrid A* planner with robot parameters and planning settings.
        Args:
            wheelbase (float): Distance between front and rear axles (meters). Used in kinematic model.
            step_size (float): Simulation distance for each motion primitive (meters).
            max_steering_angle (float): Maximum steering angle (radians).
            num_steering_angles (int): Number of discrete steering actions to generate between -max and +max.
            obstacle_threshold (int): Occupancy grid value above which a cell is considered an obstacle (0-100).
            robot_length (float): Length of the robot (meters) for collision checking.
            robot_width (float): Width of the robot (meters) for collision checking.
            goal_xy_tolerance (float): Position tolerance (meters) for reaching the goal.
            goal_theta_tolerance (float): Orientation tolerance (radians) for reaching the goal.
            num_angle_bins (int): How many bins to divide 360 degrees into for discrete state checking.
            heuristic_weight (float): Multiplier for the heuristic cost (higher values favor reaching goal faster, potentially sacrificing optimality).
        """
        # --- Store Robot Kinematic & Dimension Parameters ---
        self.WHEELBASE = wheelbase           # Distance between axles
        self.STEP_SIZE = step_size           # Distance covered in one simulation step
        self.MAX_STEERING = max_steering_angle # Max steering angle (radians)
        self.NUM_STEERING_ANGLES = num_steering_angles # How many steering options (e.g., 5 = Left, SlightLeft, Straight, SlightRight, Right)
        self.ROBOT_LENGTH = robot_length     # Robot dimension along its driving direction
        self.ROBOT_WIDTH = robot_width       # Robot dimension perpendicular to driving direction

        # --- Store Planning & Grid Parameters ---
        self.OBSTACLE_THRESHOLD = obstacle_threshold # Grid cells with value >= this are obstacles
        self.GOAL_XY_TOL = goal_xy_tolerance     # Acceptable distance error to goal position
        self.GOAL_THETA_TOL = goal_theta_tolerance # Acceptable angle error to goal orientation
        self.NUM_ANGLE_BINS = num_angle_bins     # Discretization level for orientation
        self.ANGLE_BIN_SIZE = 2.0 * math.pi / num_angle_bins # Size of each angle bin in radians
        self.HEURISTIC_WEIGHT = heuristic_weight # Weight applied to the heuristic calculation

        # --- Precompute Steering Angles ---
        # Generate a list of discrete steering angles from -MAX_STEERING to +MAX_STEERING
        if self.NUM_STEERING_ANGLES > 1:
            self.steering_angles = np.linspace(-self.MAX_STEERING, self.MAX_STEERING, self.NUM_STEERING_ANGLES)
        else:
            self.steering_angles = [0.0] # If only 1 angle allowed, it must be straight

        # --- Precompute Robot Footprint Points (relative to robot center [0,0]) ---
        # Used for collision checking. Define corners relative to the center.
        half_L = self.ROBOT_LENGTH / 2.0
        half_W = self.ROBOT_WIDTH / 2.0
        # List of (x, y) offsets from the center representing the corners
        self.footprint_rel = [
            (half_L, half_W), (half_L, -half_W),
            (-half_L, -half_W), (-half_L, half_W)
        ]
        # Note: For more accuracy, especially with large step sizes or high resolutions,
        # you might add more points along the edges of the footprint here.

        # --- Map Data Storage (Initialized later by set_map) ---
        self.map_info = None        # Will store MapMetaData object
        self.map_width = 0          # Map width in cells
        self.map_height = 0         # Map height in cells
        self.map_resolution = 0     # Map resolution in meters/cell
        self.map_origin_x = 0       # World X coordinate of grid cell (0,0)
        self.map_origin_y = 0       # World Y coordinate of grid cell (0,0)
        self.map_data = None        # Will store 2D numpy array of occupancy values

    def set_map(self, occupancy_grid_msg):
        """
        Callback function to receive and store the occupancy grid map data from ROS.
        This needs to be called before planning can start.
        """
        rospy.loginfo("Hybrid A* Planner: Received new map.")
        # Store the metadata (resolution, width, height, origin)
        self.map_info = occupancy_grid_msg.info
        self.map_width = self.map_info.width
        self.map_height = self.map_info.height
        self.map_resolution = self.map_info.resolution
        self.map_origin_x = self.map_info.origin.position.x
        self.map_origin_y = self.map_info.origin.position.y
        # Convert the 1D list of map data into a 2D numpy array (row-major order)
        self.map_data = np.array(occupancy_grid_msg.data).reshape((self.map_height, self.map_width))

    def _discretize_state(self, x, y, theta):
        """
        Converts a continuous state (x, y, theta) into a discrete tuple
        (grid_row, grid_col, angle_bin_index).
        This is used as the key in dictionaries (cost_so_far, came_from) to
        efficiently track visited states and costs.
        """
        # Convert world x, y to grid row, col
        row, col = world_to_map(x, y, self.map_origin_x, self.map_origin_y, self.map_resolution)
        # Normalize the angle and calculate which bin it falls into
        # Adding pi shifts range to [0, 2pi] before dividing, ensuring positive bin index
        theta_bin = int((normalize_angle(theta) + math.pi) / self.ANGLE_BIN_SIZE) % self.NUM_ANGLE_BINS
        return row, col, theta_bin

    def _simulate_motion(self, x, y, theta, steering_angle):
        """
        Simulates the robot's movement for one STEP_SIZE distance using the
        kinematic bicycle model with a given steering angle.
        Returns the resulting continuous state (next_x, next_y, next_theta).
        """
        # Ensure steering angle is within limits (optional sanity check)
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING, self.MAX_STEERING)

        # Handle straight motion (or very small steering angle to avoid division by zero)
        if abs(steering_angle) < 1e-6:
            next_x = x + self.STEP_SIZE * math.cos(theta)
            next_y = y + self.STEP_SIZE * math.sin(theta)
            next_theta = theta # Orientation doesn't change
        else:
            # Apply bicycle model formulas for turning motion
            turn_radius = self.WHEELBASE / math.tan(steering_angle) # Calculate turning radius
            # Calculate the angle turned during this step (arc length / radius)
            beta = self.STEP_SIZE / turn_radius
            # Calculate the final orientation after turning
            next_theta = normalize_angle(theta + beta)
            # Calculate the change in x and y based on the circular path segment
            # This form avoids calculating the center of the circle explicitly
            next_x = x + turn_radius * (math.sin(next_theta) - math.sin(theta))
            next_y = y - turn_radius * (math.cos(next_theta) - math.cos(theta))

        return next_x, next_y, next_theta

    def _is_collision_free(self, x, y, theta):
        """
        **MODIFIED FOR BETTER COLLISION CHECKING**
        Checks if the robot's footprint at the continuous state (x, y, theta)
        collides with any obstacles in the occupancy grid.
        Uses a bounding box approach around the rotated footprint.
        Returns True if collision-free, False otherwise.
        """
        if self.map_data is None: return False # Cannot check without map

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # --- 1. Calculate World Coordinates of Footprint Corners ---
        corners_world = []
        for corner_x_rel, corner_y_rel in self.footprint_rel:
            # Rotate and translate the relative corner points to world frame
            world_x = x + (corner_x_rel * cos_t - corner_y_rel * sin_t)
            world_y = y + (corner_x_rel * sin_t + corner_y_rel * cos_t)
            corners_world.append((world_x, world_y))

        # --- 2. Calculate Grid Cell Indices of Footprint Corners ---
        corners_grid = []
        for wx, wy in corners_world:
            row, col = world_to_map(wx, wy, self.map_origin_x, self.map_origin_y, self.map_resolution)
            corners_grid.append((row, col))

        # --- 3. Get Bounding Box of Footprint in Grid Coordinates ---
        # Find the minimum and maximum row and column indices covered by the corners
        min_r = min(r for r, c in corners_grid)
        max_r = max(r for r, c in corners_grid)
        min_c = min(c for r, c in corners_grid)
        max_c = max(c for r, c in corners_grid)

        # --- 4. Check All Cells Within the Bounding Box ---
        # Clip the bounding box to ensure it stays within map boundaries
        min_r_clipped = max(0, min_r)
        max_r_clipped = min(self.map_height - 1, max_r)
        min_c_clipped = max(0, min_c)
        max_c_clipped = min(self.map_width - 1, max_c)

        # Iterate through every grid cell within the clipped bounding box
        for r in range(min_r_clipped, max_r_clipped + 1):
            for c in range(min_c_clipped, max_c_clipped + 1):
                # Check if the cell is outside the actual map bounds (redundant due to clipping, but safe)
                # if not (0 <= r < self.map_height and 0 <= c < self.map_width):
                #     return False # Part of footprint check area is outside map -> collision

                # Check if the occupancy value indicates an obstacle (or unknown space treated as obstacle)
                if self.map_data[r, c] >= self.OBSTACLE_THRESHOLD or self.map_data[r,c] == -1:
                    return False # Collision detected

        # If no collisions were found in any cell within the bounding box, the state is collision-free
        return True
        # Note: This bounding box check is safer than just checking corners, but can be
        # conservative (might report collision even if only the corners of the bbox hit
        # obstacles, but the actual robot footprint doesn't). For perfect accuracy,
        # polygon rasterization (e.g., using skimage.draw.polygon) is needed.

    def _heuristic(self, x, y, goal_x, goal_y):
        """
        Calculates the heuristic cost (estimated cost from current state to goal).
        Uses simple Euclidean distance multiplied by a weight.
        This guides the A* search towards the goal position.
        """
        # np.hypot(dx, dy) is equivalent to sqrt(dx^2 + dy^2)
        return self.HEURISTIC_WEIGHT * np.hypot(goal_x - x, goal_y - y)

    def reconstruct_path(self, came_from, current_discrete, current_state):
        """
        Rebuilds the path by backtracking from the goal state using the came_from dictionary.
        Args:
            came_from (dict): Maps {child_discrete_state: (parent_discrete_state, parent_continuous_state)}
            current_discrete: The final discrete state that met goal criteria.
            current_state: The final continuous state corresponding to current_discrete.
        Returns:
            list: A list of continuous (x, y, theta) state tuples representing the path from start to goal.
        """
        path = [current_state] # Start the path list with the goal state
        # Trace back predecessors until the start state is reached
        while current_discrete in came_from:
            # Get the parent's discrete state and continuous state from the dictionary
            parent_discrete, parent_state = came_from[current_discrete]
            # Add the parent's continuous state to the path
            path.append(parent_state)
            # Move to the parent for the next iteration
            current_discrete = parent_discrete
            current_state = parent_state # Keep track of continuous state for potential start check

        # The path is built goal-to-start, so reverse it before returning
        return path[::-1]

    def plan_path(self, start, goal):
        """
        Main Hybrid A* path planning function. Executes the search.
        Args:
            start (tuple): Continuous start state (x, y, theta).
            goal (tuple): Continuous goal state (x, y, theta).
        Returns:
            list or None: A list of continuous (x, y, theta) waypoints if a path is found, otherwise None.
        """
        # --- Initialization ---
        if self.map_data is None:
            rospy.logerr("Hybrid A* Planner: Map data not set!")
            return None

        # Priority queue (min-heap). Stores tuples: (f_score, continuous_state)
        # f_score = g_score + h_score (estimated total cost)
        open_set = []
        start_discrete = self._discretize_state(*start) # Discrete version of start state
        start_heuristic = self._heuristic(start[0], start[1], goal[0], goal[1])
        # Push the start node onto the heap. g_score is 0, f_score = 0 + heuristic.
        heapq.heappush(open_set, (start_heuristic, start))

        # came_from dictionary: Stores the path predecessors.
        # Key: child_discrete_state, Value: (parent_discrete_state, parent_continuous_state)
        came_from = {}

        # cost_so_far dictionary (g_score): Stores the actual cost (path length) from start
        # to a given discrete state. Key: discrete_state, Value: cost
        cost_so_far = {start_discrete: 0.0}

        # Store goal theta separately for quick access in the loop
        goal_theta = goal[2]

        rospy.loginfo("Hybrid A*: Planning started...")
        start_time = rospy.Time.now()

        # --- Main A* Search Loop ---
        while open_set: # While there are nodes to explore
            # Pop the node with the lowest f_score from the priority queue
            current_f_score, current_state = heapq.heappop(open_set)
            # Get discrete version for dictionary lookups
            current_discrete = self._discretize_state(*current_state)

            # Optimization: If the cost to reach this state (current_g) is already higher
            # than a previously found cost, skip it (this handles stale entries in the heap).
            current_g_score = cost_so_far.get(current_discrete, float('inf'))
            # Check if the f_score from heap matches calculated f_score based on current g_score
            # Add small epsilon for float comparison safety
            expected_f_score = current_g_score + self._heuristic(current_state[0], current_state[1], goal[0], goal[1])
            if current_f_score > expected_f_score + 1e-5:
                continue # Stale node, a better path was found already

            # --- Goal Check ---
            # Check if the current state is within the position and orientation tolerances of the goal.
            position_error = np.hypot(current_state[0] - goal[0], current_state[1] - goal[1])
            orientation_error = abs(normalize_angle(current_state[2] - goal_theta))

            if (position_error < self.GOAL_XY_TOL and orientation_error < self.GOAL_THETA_TOL):
                # Goal reached! Reconstruct and return the path.
                rospy.loginfo(f"Hybrid A*: Goal reached! Planning time: {(rospy.Time.now() - start_time).to_sec():.3f}s")
                return self.reconstruct_path(came_from, current_discrete, current_state)

            # --- Expand Neighbors ---
            # Iterate through all possible steering actions
            for steering in self.steering_angles:
                # Simulate motion for one step with the current steering angle
                next_state = self._simulate_motion(*current_state, steering)

                # Check if the resulting state is valid (collision-free)
                # *** This now uses the improved collision check ***
                if not self._is_collision_free(*next_state):
                    continue # Skip this steering action, it leads to collision

                # Calculate the cost (g_score) to reach this next_state
                # Cost is the cost to reach the current state plus the cost of this step (step_size)
                new_cost = current_g_score + self.STEP_SIZE

                # Get the discrete version of the next state for dictionary checks
                next_discrete = self._discretize_state(*next_state)

                # Check if this path to the neighbor is better than any previous path found
                if new_cost < cost_so_far.get(next_discrete, float('inf')):
                    # Update the cost to reach this neighbor
                    cost_so_far[next_discrete] = new_cost
                    # Calculate the priority (f_score) for the neighbor node
                    priority = new_cost + self._heuristic(next_state[0], next_state[1], goal[0], goal[1])
                    # Add the neighbor to the open set (priority queue)
                    heapq.heappush(open_set, (priority, next_state))
                    # Record that we reached next_discrete from current_state
                    came_from[next_discrete] = (current_discrete, current_state)

        # If the open set becomes empty and we haven't returned a path, it means no path was found
        rospy.logwarn(f"Hybrid A*: Failed to find path after {(rospy.Time.now() - start_time).to_sec():.3f}s.")
        return None

# ----------------------- ROS Node Wrapper Class -----------------------

class HybridAStarNode:
    """
    Manages the ROS interactions for the Hybrid A* planner.
    Handles subscriptions, publishing, and triggering the planner logic.
    """
    def __init__(self):
        """
        Initializes the ROS node, sets up the planner, subscribers, and publisher.
        """
        rospy.init_node('hybrid_astar_node')
        rospy.loginfo("Hybrid A* Node Initializing...")

        # --- Create the Planner Instance ---
        # TODO: Consider using ROS parameters for these values instead of hardcoding
        #       e.g., wheelbase = rospy.get_param("~wheelbase", 0.3)
        self.planner = HybridAStarPlanner(
            wheelbase=0.3,          # Example value, adjust for your robot
            step_size=0.1,          # Example value, adjust for speed/accuracy tradeoff
            max_steering_angle=0.6, # Example value (~35 deg), adjust for your robot
            num_steering_angles=5,  # Example value (L, SL, S, SR, R)
            robot_length=0.4,       # Example value, measure your robot
            robot_width=0.3,        # Example value, measure your robot
            heuristic_weight=1.5    # Example value, tune for performance
            # Other parameters use defaults defined in HybridAStarPlanner.__init__
        )

        # --- Internal State Variable ---
        # Stores the latest PoseStamped message received from the localization node
        self.current_pose = None
        self.map_received = False # Flag to ensure map is ready before planning

        # --- ROS Subscribers ---
        # Subscribe to the OccupancyGrid map topic
        map_topic = "/map" # Default map topic
        rospy.Subscriber(map_topic, OccupancyGrid, self.map_callback)
        # Subscribe to the robot's pose estimate (published by ICP/localization node)
        pose_topic = "/robot_pose" # Assumed topic name for localization output
        rospy.Subscriber(pose_topic, PoseStamped, self.pose_callback)
        # Subscribe to the goal topic (typically published by RViz's "2D Nav Goal" tool)
        goal_topic = "/move_base_simple/goal" # Standard topic for RViz goals
        rospy.Subscriber(goal_topic, PoseStamped, self.goal_callback)

        # --- ROS Publisher ---
        # Publisher for the calculated path
        path_topic = "/planned_path" # Topic to publish the resulting path on
        self.path_pub = rospy.Publisher(path_topic, Path, queue_size=1) # queue_size=1: only keep latest path

        rospy.loginfo(f"Hybrid A* Node Ready. Subscribed to map:'{map_topic}', pose:'{pose_topic}', goal:'{goal_topic}'. Publishing path to:'{path_topic}'")

    def map_callback(self, msg):
        """ROS Callback: Stores the received map data in the planner instance."""
        if self.planner:
            self.planner.set_map(msg)
            self.map_received = True

    def pose_callback(self, msg):
        """ROS Callback: Stores the latest robot pose estimate from localization."""
        # --- Coordinate Frame Check ---
        # COMMENT: It's crucial that the pose received here is in the same
        #          coordinate frame as the map. Ideally, check msg.header.frame_id
        #          against self.planner.map_info.header.frame_id.
        # HOW TO CHECK:
        # 1. Use `rostopic info /robot_pose` and `rostopic info /map` to see the message types.
        # 2. Use `rostopic echo /robot_pose/header/frame_id` to see the frame name being published for the pose.
        # 3. Use `rostopic echo /map/header/frame_id` (or /map/info/header/frame_id depending on structure) to see the map's frame name.
        # 4. Visualize in RViz: Add TF display and Map display. Add a Pose display subscribed to /robot_pose. See if the pose appears correctly located on the map.
        # IF FRAMES DON'T MATCH: You *must* use the TF library (`import tf`) to look up the transform
        #                       between the pose's frame and the map's frame and apply it to the pose
        #                       data before using it. This is beyond the scope of this basic check comment.
        # Basic Check (Warning Only):
        if self.map_received and self.planner.map_info and msg.header.frame_id != self.planner.map_info.header.frame_id:
             rospy.logwarn_throttle(10, f"Pose frame '{msg.header.frame_id}' may not match map frame '{self.planner.map_info.header.frame_id}'. Verify frames or use TF!")

        self.current_pose = msg # Store the latest pose

    def goal_callback(self, goal_msg):
        """
        ROS Callback: Triggered when a new goal is received (e.g., from RViz).
        Initiates the path planning process using the latest map, pose, and the received goal.
        """
        rospy.loginfo("Hybrid A*: Received new goal request.")

        # --- Pre-planning Checks ---
        if not self.map_received or self.planner.map_data is None:
            rospy.logwarn("Hybrid A*: Cannot plan - map not yet received or processed.")
            return
        if self.current_pose is None:
            rospy.logwarn("Hybrid A*: Cannot plan - current robot pose unknown.")
            return
        if self.planner is None:
             rospy.logerr("Hybrid A*: Planner object not initialized!")
             return

        # --- Coordinate Frame Check for Goal ---
        # COMMENT: Similar to the pose check, ensure the goal is in the map frame.
        # HOW TO CHECK: Use `rostopic echo /move_base_simple/goal/header/frame_id`. It should match the map frame. RViz usually sends goals in the frame selected in its 'Fixed Frame' setting.
        # IF FRAMES DON'T MATCH: Use TF library to transform the goal pose into the map frame before planning.
        map_frame = self.planner.map_info.header.frame_id
        if goal_msg.header.frame_id != map_frame:
             rospy.logerr(f"Goal frame '{goal_msg.header.frame_id}' does not match map frame '{map_frame}'. Cannot plan. Use TF or ensure goal is published in map frame.")
             return
        # Basic check for pose frame consistency (already warned in pose_callback)
        if self.current_pose.header.frame_id != map_frame:
             rospy.logerr(f"Current pose frame '{self.current_pose.header.frame_id}' does not match map frame '{map_frame}'. Cannot plan.")
             return


        # --- Extract Start State from Current Pose ---
        start_pos = self.current_pose.pose.position
        start_q = self.current_pose.pose.orientation
        # Convert quaternion orientation to yaw angle (theta)
        start_theta = euler_from_quaternion([start_q.x, start_q.y, start_q.z, start_q.w])[2]
        # Create the start state tuple (x, y, theta)
        start = (start_pos.x, start_pos.y, start_theta)

        # --- Extract Goal State from Goal Message ---
        goal_pos = goal_msg.pose.position
        goal_q = goal_msg.pose.orientation
        # Convert quaternion orientation to yaw angle (theta)
        goal_theta = euler_from_quaternion([goal_q.x, goal_q.y, goal_q.z, goal_q.w])[2]
        # Create the goal state tuple (x, y, theta)
        goal = (goal_pos.x, goal_pos.y, goal_theta)

        rospy.loginfo(f"Hybrid A*: Planning from Start:({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f}) to Goal:({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f})")

        # --- Execute the Planner ---
        # This calls the main planning logic in the HybridAStarPlanner class
        path_points = self.planner.plan_path(start, goal) # Returns list of (x,y,theta) tuples or None

        # --- Prepare and Publish Path Message ---
        path_msg = Path()
        # Set the header: frame_id must match the coordinate system the path points are in (map frame)
        path_msg.header.frame_id = map_frame # Use the verified map frame
        path_msg.header.stamp = rospy.Time.now() # Timestamp the path generation time

        if path_points: # If the planner returned a valid path
            # Iterate through the (x, y, theta) waypoints
            for x, y, theta in path_points:
                # Create a PoseStamped message for each waypoint
                pose = PoseStamped()
                # Set the header for the individual pose (optional, but good practice)
                pose.header.stamp = path_msg.header.stamp
                pose.header.frame_id = path_msg.header.frame_id
                # Set the position
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0 # Assume 2D navigation
                # Convert the yaw angle (theta) back to a quaternion for the pose orientation
                q = quaternion_from_euler(0, 0, theta) # Roll=0, Pitch=0, Yaw=theta
                pose.pose.orientation = Quaternion(*q) # Unpack the quaternion elements
                # Add the completed PoseStamped message to the Path message's list of poses
                path_msg.poses.append(pose)

            # Publish the complete Path message
            self.path_pub.publish(path_msg)
            rospy.loginfo("Hybrid A*: Path published with %d waypoints." % len(path_msg.poses))
        else:
            # If planner returned None (no path found)
            rospy.logwarn("Hybrid A*: Failed to find a path. Publishing empty path.")
            # Publish an empty path message - this can clear previous paths in RViz
            self.path_pub.publish(path_msg)

# ----------------------- Main Execution Block -----------------------

if __name__ == '__main__':
    try:
        # Create an instance of the ROS node class. This calls __init__().
        node = HybridAStarNode()
        # Enter the ROS spin loop. This keeps the node alive, processing callbacks
        # (for map, pose, goal) until the node is shut down (e.g., Ctrl+C).
        rospy.spin()
    except rospy.ROSInterruptException:
        # Catch the exception raised when ROS wants to shut down
        rospy.loginfo("Hybrid A* Node shutting down.")
        pass # Exit cleanly
    except Exception as e:
        # Catch any other unexpected errors during initialization or execution
        rospy.logfatal(f"Unhandled exception in Hybrid A* Node: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging

