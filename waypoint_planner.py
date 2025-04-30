

import numpy as np
import math
import heapq  
import time   



# ----------------------- Hardcoded Goal Region Definition -----------------------
# Define the target rectangular area in WORLD coordinates (meters).
# The planner succeeds if it finds a valid state (x, y, theta) where
# the (x, y) position falls within these boundaries.
# IMPORTANT: Ensure this region is collision-free on the map you provide!
GOAL_X_MIN = 4.5  # Minimum X coordinate of the goal region
GOAL_X_MAX = 5.0  # Maximum X coordinate of the goal region
GOAL_Y_MIN = -0.5 # Minimum Y coordinate of the goal region
GOAL_Y_MAX = 0.5  # Maximum Y coordinate of the goal region

GOAL_TARGET_THETA = None # MODIFIED: Set to None to ignore final orientation

# --- Heuristic Calculation ---
# Calculate the center of the goal region. The heuristic function (estimated
# cost to goal) will aim towards this central point.
GOAL_CENTER_X = (GOAL_X_MIN + GOAL_X_MAX) / 2.0
GOAL_CENTER_Y = (GOAL_Y_MIN + GOAL_Y_MAX) / 2.0



def world_to_map(x, y, origin_x, origin_y, resolution):
    """
    Converts continuous world coordinates (meters) into discrete grid cell indices (row, col).
    Args:
        x, y (float): World coordinates.
        origin_x, origin_y (float): World coordinates of the bottom-left corner (cell [0,0]) of the map.
        resolution (float): Size of one grid cell in meters.
    Returns:
        tuple: (row, col) integer indices corresponding to the world coordinates.
    """
    # Calculate column index based on X coordinate relative to origin
    col = int((x - origin_x) / resolution)
    # Calculate row index based on Y coordinate relative to origin
    row = int((y - origin_y) / resolution)
    return row, col

def normalize_angle(angle):
    """
    Normalizes an angle (in radians) to the range [-pi, pi].
    Uses math.atan2(sin(a), cos(a)) for numerical stability and correctness across quadrants.
    """
    return math.atan2(math.sin(angle), math.cos(angle))


class HybridAStarPlanner:
    """
    Implements the core Hybrid A* search algorithm logic.

    Hybrid A* extends the standard A* algorithm by searching in a continuous state
    space (x, y, theta) and using kinematically feasible motion primitives (simulated
    car movements) as edges between states, instead of simple grid movements. It uses
    a discrete grid map for collision checking and often for discretizing states
    for efficient storage in visited sets.
    """
    def __init__(self, wheelbase, step_size, max_steering_angle, num_steering_angles,
                 obstacle_threshold=65, robot_length=0.5, robot_width=0.3,
                 goal_xy_tolerance=0.1, # Tolerance used if comparing against a specific point (less relevant for region goal)
                 goal_theta_tolerance=0.1, # Tolerance used ONLY if GOAL_TARGET_THETA is set
                 num_angle_bins=72, heuristic_weight=1.0):
        """
        Initializes the planner with robot kinematic parameters, physical dimensions,
        and algorithm tuning parameters.
        """
        # --- Robot Kinematic Parameters (for motion simulation) ---
        self.WHEELBASE = wheelbase           # Distance between front and rear axles (m)
        self.STEP_SIZE = step_size           # Simulation distance per motion primitive (m)
        self.MAX_STEERING = max_steering_angle # Maximum steering angle (radians)
        self.NUM_STEERING_ANGLES = num_steering_angles # Number of discrete steering actions to try

        # --- Robot Dimension Parameters (for collision checking) ---
        self.ROBOT_LENGTH = robot_length     # Robot length (m) along driving direction
        self.ROBOT_WIDTH = robot_width       # Robot width (m) perpendicular to driving direction

        # --- Planning & Grid Parameters ---
        self.OBSTACLE_THRESHOLD = obstacle_threshold # Occupancy grid value >= this is an obstacle
        self.GOAL_XY_TOL = goal_xy_tolerance     # Positional tolerance (less critical for region goal)
        self.GOAL_THETA_TOL = goal_theta_tolerance # Angular tolerance (used if GOAL_TARGET_THETA is defined)
        self.NUM_ANGLE_BINS = num_angle_bins     # Discretization level for orientation in visited set
        self.ANGLE_BIN_SIZE = 2.0 * math.pi / num_angle_bins # Size of each angle bin (radians)
        self.HEURISTIC_WEIGHT = heuristic_weight # Multiplier for the heuristic cost (Weighted A*)

        # --- Precompute Discrete Steering Actions ---
        # Creates a list of steering angles from -max to +max to simulate
        if self.NUM_STEERING_ANGLES > 1:
            self.steering_angles = np.linspace(-self.MAX_STEERING, self.MAX_STEERING, self.NUM_STEERING_ANGLES)
        else:
            self.steering_angles = [0.0] # Only allow driving straight

        # --- Precompute Robot Footprint Points (relative to robot center at [0,0]) ---
        # Used for collision checking. Defines corners relative to the center (origin).
        half_L = self.ROBOT_LENGTH / 2.0
        half_W = self.ROBOT_WIDTH / 2.0
        self.footprint_rel = [(half_L, half_W), (half_L, -half_W), (-half_L, -half_W), (-half_L, half_W)]

        # --- Map Data Storage (Initialized via set_map) ---
        self.map_data = None        # 2D NumPy array of occupancy values
        self.map_resolution = 0.0   # Meters per cell
        self.map_origin_x = 0.0     # World X coordinate of cell [0,0]'s corner
        self.map_origin_y = 0.0     # World Y coordinate of cell [0,0]'s corner
        self.map_height = 0         # Map height in cells
        self.map_width = 0          # Map width in cells
        print("Hybrid A* Planner Initialized (Standalone). Call set_map() before planning.")

    def set_map(self, map_data_array, resolution, origin_x, origin_y):
        """Stores the provided map data and metadata required for planning."""
        # Input validation
        if not isinstance(map_data_array, np.ndarray) or map_data_array.ndim != 2:
            raise ValueError("map_data_array must be a 2D NumPy array.")
        if resolution <= 0:
             raise ValueError("Resolution must be positive.")

        self.map_data = map_data_array
        self.map_resolution = resolution
        self.map_origin_x = origin_x
        self.map_origin_y = origin_y
        self.map_height, self.map_width = map_data_array.shape # Get dimensions from array
        print(f"Map Set: Dimensions={self.map_width}x{self.map_height}, Resolution={self.map_resolution:.3f}, Origin=({self.map_origin_x:.2f}, {self.map_origin_y:.2f})")

    def _discretize_state(self, x, y, theta):
        """
        Converts a continuous state (x, y, theta) into a discrete tuple representation
        (grid_row, grid_col, angle_bin_index). This discrete state is used as a key
        in dictionaries (cost_so_far, came_from) to efficiently track which approximate
        states have been visited and their associated costs/parents. This avoids issues
        with floating-point comparisons and significantly reduces redundant exploration.
        """
        row, col = world_to_map(x, y, self.map_origin_x, self.map_origin_y, self.map_resolution)
        # Normalize theta, shift to [0, 2pi), divide by bin size, take integer part, modulo N
        theta_bin = int((normalize_angle(theta) + math.pi) / self.ANGLE_BIN_SIZE) % self.NUM_ANGLE_BINS
        return row, col, theta_bin

    def _simulate_motion(self, x, y, theta, steering_angle):
        """
        Simulates the robot's movement for one 'STEP_SIZE' distance using the
        kinematic bicycle model for a given 'steering_angle'. This generates a potential
        next continuous state (a motion primitive).
        Returns:
            tuple: (next_x, next_y, next_theta) representing the continuous state after the step.
        """
        # Ensure steering angle is within physical limits
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING, self.MAX_STEERING)

        # Handle straight motion (or near-zero steering to avoid numerical issues)
        if abs(steering_angle) < 1e-6:
            next_x = x + self.STEP_SIZE * math.cos(theta)
            next_y = y + self.STEP_SIZE * math.sin(theta)
            next_theta = theta # Orientation does not change
        else:
            # Apply bicycle model formulas for turning motion
            turn_radius = self.WHEELBASE / math.tan(steering_angle) # Calculate turning radius
            # Calculate the angle turned during this step (arc length / radius)
            beta = self.STEP_SIZE / turn_radius
            # Calculate the final orientation after turning
            next_theta = normalize_angle(theta + beta)
            # Calculate the change in x and y based on the circular path segment
            # (Alternative formulations exist, e.g., calculating circle center first)
            next_x = x + turn_radius * (math.sin(next_theta) - math.sin(theta))
            next_y = y - turn_radius * (math.cos(next_theta) - math.cos(theta))

        return next_x, next_y, next_theta

    def _is_collision_free(self, x, y, theta):
        """
        Checks if the robot's footprint at the continuous state (x, y, theta)
        collides with any obstacles in the occupancy grid. This is a critical step.
        Uses a bounding box approach for efficiency (safer than corners, less accurate
        than polygon rasterization).
        Returns:
            bool: True if the state is collision-free, False otherwise.
        """
        if self.map_data is None: return False # Cannot check without a map

        # --- 1. Calculate World Coordinates of Footprint Corners ---
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        corners_world = []
        for corner_x_rel, corner_y_rel in self.footprint_rel:
            # Rotate and translate the relative corner points to world frame
            world_x = x + (corner_x_rel * cos_t - corner_y_rel * sin_t)
            world_y = y + (corner_x_rel * sin_t + corner_y_rel * cos_t)
            corners_world.append((world_x, world_y))

        # --- 2. Calculate Grid Cell Indices of Footprint Corners ---
        corners_grid = []
        try:
            for wx, wy in corners_world:
                 row, col = world_to_map(wx, wy, self.map_origin_x, self.map_origin_y, self.map_resolution)
                 corners_grid.append((row, col))
        except ValueError: # Handle potential errors from world_to_map if resolution is bad
            print("Error in world_to_map during collision check.")
            return False

        # --- 3. Simple Out-of-Bounds Check (Any Corner Outside Map) ---
        # If any corner maps to a cell outside the grid dimensions, it's invalid.
        for r, c in corners_grid:
             if not (0 <= r < self.map_height and 0 <= c < self.map_width):
                  return False # Collision (out of bounds)

        # --- 4. Bounding Box Check ---
        # Find the min/max row/col indices covered by the corners.
        min_r = min(r for r, c in corners_grid)
        max_r = max(r for r, c in corners_grid)
        min_c = min(c for r, c in corners_grid)
        max_c = max(c for r, c in corners_grid)

        # Clip bounds just in case (should be redundant after corner check, but safe)
        min_r_clipped = max(0, min_r)
        max_r_clipped = min(self.map_height - 1, max_r)
        min_c_clipped = max(0, min_c)
        max_c_clipped = min(self.map_width - 1, max_c)

        # --- 5. Check Occupancy of Cells within Bounding Box ---
        # Iterate through every grid cell within the clipped bounding box.
        # This is an approximation - it checks more cells than the exact footprint might cover.
        for r in range(min_r_clipped, max_r_clipped + 1):
            for c in range(min_c_clipped, max_c_clipped + 1):
                # Check the occupancy value from the map data
                # Treat unknown (-1) as occupied for safety.
                if self.map_data[r, c] >= self.OBSTACLE_THRESHOLD or self.map_data[r, c] == -1:
                    return False # Collision detected within the bounding box

        # If no collisions were found after checking all cells in the bbox, consider state safe.
        return True

    def _heuristic(self, x, y):
        """
        Calculates the heuristic cost (h_score): an estimate of the cost remaining
        from the current state (x, y) to the goal. It guides the search.
        Here, we use simple Euclidean distance to the *center* of the goal region,
        multiplied by a weighting factor. This heuristic ignores obstacles and kinematics,
        making it admissible (it doesn't overestimate the true cost) if weight is 1.0.
        """
        dx = GOAL_CENTER_X - x
        dy = GOAL_CENTER_Y - y
        # np.hypot calculates sqrt(dx^2 + dy^2)
        return self.HEURISTIC_WEIGHT * np.hypot(dx, dy)

    def reconstruct_path(self, came_from, current_discrete, current_state):
        """
        Rebuilds the path by backtracking from the goal state using the 'came_from'
        dictionary, which stores the parent of each visited state.
        Args:
            came_from (dict): Maps {child_discrete_state: (parent_discrete_state, parent_continuous_state)}
            current_discrete: The final discrete state that met goal criteria.
            current_state: The final continuous state corresponding to current_discrete.
        Returns:
            list: A list of continuous (x, y, theta) state tuples from start to goal.
        """
        path = [current_state] # Start the list with the final state
        # Follow the chain of parents back to the start
        while current_discrete in came_from:
            # Retrieve the parent's discrete and continuous states
            parent_discrete, parent_state = came_from[current_discrete]
            # Prepend the parent state to the path list
            path.append(parent_state)
            # Update current states for the next iteration
            current_discrete = parent_discrete
            # current_state = parent_state # Not strictly needed inside loop anymore

        return path[::-1] # Reverse the list to get start -> goal order

    # --- Main Planning Function ---
    def find_path_internal(self, start):
        """
        Internal implementation of the Hybrid A* search algorithm.
        Args:
            start (tuple): Continuous start state (x, y, theta).
        Returns:
            list or None: List of waypoints [(x, y, theta), ...] or None if no path found.
        """
        # --- 1. Pre-computation & Sanity Checks ---
        if self.map_data is None:
            print("Error: Map data not set. Call set_map() before planning.")
            return None
        if not isinstance(start, tuple) or len(start) != 3:
            raise ValueError("Start state must be a tuple (x, y, theta)")
        if not self._is_collision_free(*start):
             print(f"Error: Start state {start} is in collision!")
             return None

        # --- 2. Initialization ---
        # --- Priority Queue (Open Set) ---
        # Stores nodes to be explored, prioritized by f_score.
        # Uses heapq module (min-heap). Lower f_score = higher priority.
        # Elements are tuples: (f_score, continuous_state_tuple)
        open_set = []
        start_discrete = self._discretize_state(*start)
        start_g_score = 0.0 # Cost from start to start is zero
        start_h_score = self._heuristic(start[0], start[1]) # Estimated cost from start to goal
        start_f_score = start_g_score + start_h_score # A* formula: f = g + h
        heapq.heappush(open_set, (start_f_score, start))

        # --- came_from dictionary ---
        # Stores the path predecessors for reconstructing the path later.
        # Key: discrete_state tuple (row, col, bin) of the child node
        # Value: tuple (parent_discrete_state, parent_continuous_state)
        came_from = {}

        # --- cost_so_far dictionary (g_score) ---
        # Stores the minimum cost found so far to reach a specific discrete state from the start.
        # Key: discrete_state tuple (row, col, bin)
        # Value: float (cost, typically path length or time)
        cost_so_far = {start_discrete: start_g_score}

        print("Hybrid A* Planning started...")
        start_time = time.time()
        nodes_expanded = 0 # Counter for search statistics

        # --- 3. Main A* Search Loop ---
        # Continue as long as there are nodes in the open set to explore
        while open_set:
            nodes_expanded += 1

            # --- 3a. Select Best Node ---
            # Pop the node with the lowest f_score from the priority queue (heap).
            # This node is the most promising one to explore next according to f = g + h.
            current_f_score, current_state = heapq.heappop(open_set)

            # --- 3b. Check for Stale Node (Optimization) ---
            # It's possible to have multiple entries for the same discrete state in the heap
            # if we found a shorter path to it later. This check ensures we only process
            # a discrete state using the lowest cost found so far for it.
            current_discrete = self._discretize_state(*current_state)
            current_g_score = cost_so_far.get(current_discrete, float('inf'))
            # Recalculate f_score based on the *actual* current g_score stored.
            # If the f_score popped from heap is significantly larger, it means the popped
            # node corresponds to an older, longer path to this state. Skip it.
            expected_f_score = current_g_score + self._heuristic(current_state[0], current_state[1])
            if current_f_score > expected_f_score + 1e-5: # Add small tolerance for float errors
                continue # This node is stale, ignore it.

            # --- 3c. Goal Check ---
            # Check if the current continuous state's (x, y) position is inside the target region.
            current_x, current_y, current_theta = current_state
            is_in_goal_xy = (GOAL_X_MIN <= current_x <= GOAL_X_MAX and
                             GOAL_Y_MIN <= current_y <= GOAL_Y_MAX)

            # MODIFIED: Only check XY position. Orientation is ignored as GOAL_TARGET_THETA is None.
            if is_in_goal_xy:
                 # Optional: Check if GOAL_TARGET_THETA is set and check orientation
                 # goal_orientation_met = True
                 # if GOAL_TARGET_THETA is not None:
                 #     orientation_error = abs(normalize_angle(current_theta - GOAL_TARGET_THETA))
                 #     goal_orientation_met = (orientation_error < self.GOAL_THETA_TOL)
                 # if goal_orientation_met: # Only succeed if orientation is also met (if required)
                 #      ...

                 # Succeed if XY position is within the defined goal region bounds.
                 print(f"Hybrid A*: Goal region reached! Planning time: {time.time() - start_time:.3f}s ({nodes_expanded} nodes)")
                 # Reconstruct the path from the 'came_from' map and return it.
                 return self.reconstruct_path(came_from, current_discrete, current_state)


            # --- 3d. Expand Neighbors (Generate Successors) ---
            # If not at the goal, explore possible next states by applying motion primitives.
            for steering in self.steering_angles:
                # --- i. Simulate Motion ---
                # Apply the kinematic model for the current steering angle.
                next_state = self._simulate_motion(*current_state, steering)

                # --- ii. Check Collision ---
                # Ensure the simulated next state is physically valid (not colliding).
                if not self._is_collision_free(*next_state):
                    continue # Skip this steering action, path is blocked.

                # --- iii. Calculate Cost (g_score) ---
                # The cost to reach the neighbor ('next_state') is the cost to reach the
                # 'current_state' plus the cost of the motion primitive (step size).
                new_cost = current_g_score + self.STEP_SIZE

                # --- iv. Check if Path is Better ---
                # Get the discrete representation of the neighbor state.
                next_discrete = self._discretize_state(*next_state)
                # Compare the new cost to the best cost found *so far* to reach this 'next_discrete' state.
                if new_cost < cost_so_far.get(next_discrete, float('inf')):
                    # This is a better path! Update records.

                    # Update the minimum cost found to reach 'next_discrete'.
                    cost_so_far[next_discrete] = new_cost

                    # Calculate the f_score (priority) for this neighbor node: f = new_g + h
                    heuristic_cost = self._heuristic(next_state[0], next_state[1])
                    priority = new_cost + heuristic_cost

                    # Add the neighbor node to the priority queue to be explored later.
                    heapq.heappush(open_set, (priority, next_state))

                    # Record that we reached 'next_discrete' from 'current_state'.
                    # Store the parent's discrete and continuous state for path reconstruction.
                    came_from[next_discrete] = (current_discrete, current_state)

        # --- 4. No Path Found ---
        # If the open set becomes empty, it means we explored all reachable states
        # without finding a path to the goal region.
        print(f"Warning: Hybrid A* failed to find path after {time.time() - start_time:.3f}s ({nodes_expanded} nodes).")
        return None # Return None to indicate failure

# ==============================================================================
# Main Planning Function Interface (Use this function from other scripts)
# ==============================================================================

# --- Create a default planner instance ---
# This can be reused across multiple calls if the parameters don't change.
# Consider making parameters configurable via function arguments if needed.
default_planner = HybridAStarPlanner(
    wheelbase=0.3, step_size=0.1, max_steering_angle=0.6, num_steering_angles=5,
    robot_length=0.4, robot_width=0.3, obstacle_threshold=65, heuristic_weight=1.5
)

def plan_path_to_goal_region(start_pose, occupancy_map, resolution, origin_x, origin_y, planner=default_planner):
    """
    High-level function to plan a path using the Hybrid A* planner towards
    the hardcoded GOAL region. This function acts as the main interface.

    Args:
        start_pose (tuple): Current robot pose (x, y, theta) in meters and radians (world frame).
        occupancy_map (numpy.ndarray): 2D numpy array representing the grid map (0-100 or -1).
        resolution (float): Map resolution in meters per cell.
        origin_x (float): World X coordinate of the map's bottom-left corner (cell [0,0]).
        origin_y (float): World Y coordinate of the map's bottom-left corner (cell [0,0]).
        planner (HybridAStarPlanner, optional): An existing planner instance. Defaults to 'default_planner'.

    Returns:
        list or None: A list of waypoint tuples [(x, y, theta), ...] if a path is found,
                      otherwise None. This list is the direct output for the controller.
    """
    print("\n--- New Planning Request ---")
    if planner is None:
        print("Error: Planner instance is not provided.")
        return None

    # 1. Update the planner with the latest map information
    #    This needs to be done for every planning request if the map changes.
    try:
        planner.set_map(occupancy_map, resolution, origin_x, origin_y)
    except ValueError as e:
        print(f"Error setting map: {e}")
        return None

    # 2. Prepare start state (ensure theta is normalized for internal consistency)
    start_x, start_y, start_theta = start_pose
    normalized_start = (start_x, start_y, normalize_angle(start_theta))
    print(f"Planning from normalized start: ({normalized_start[0]:.2f}, {normalized_start[1]:.2f}, {normalized_start[2]:.2f})")
    print(f"Targeting Goal Region: X=[{GOAL_X_MIN:.2f},{GOAL_X_MAX:.2f}], Y=[{GOAL_Y_MIN:.2f},{GOAL_Y_MAX:.2f}]")
    # MODIFIED: Removed the print statement about goal orientation as it's ignored
    # if GOAL_TARGET_THETA is not None: ...

    # 3. Call the internal planning method which performs the search
    waypoints = planner.find_path_internal(normalized_start)

    # 4. Return the resulting waypoints list (or None)
    if waypoints:
        print(f"Planning successful. Returning {len(waypoints)} waypoints.")
    else:
        print("Planning failed.")
    print("--- Planning Request End ---")
    return waypoints

