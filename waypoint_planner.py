import numpy as np
import math
import heapq
import time

# ==============================================================================
# Utility Functions 
# ==============================================================================

def world_to_map(x_world, y_world, resolution):
    """Converts world coordinates (map's fixed frame, origin at [0,0]) to map indices."""
    if resolution <= 0:
        raise ValueError("Map resolution must be positive")
    col = int(x_world / resolution)
    row = int(y_world / resolution)
    return row, col

def map_to_world(row, col, resolution):
    """Converts map indices to world coordinates (center of cell, map's fixed frame)."""
    if resolution <= 0:
        raise ValueError("Map resolution must be positive")
    x_world = (col ) * resolution
    y_world = (row) * resolution
    return x_world, y_world

def normalize_angle(angle):
    """Normalizes angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))

# ==============================================================================
# Hybrid A* Planner Class 
# ==============================================================================

class HybridAStarPlanner:
    """Implements Hybrid A* logic, operating internally in a fixed world frame."""
    def __init__(self, wheelbase, step_size, max_steering_angle, num_steering_angles,
                 obstacle_threshold=65, robot_length=0.4, robot_width=0.3,
                 num_angle_bins=72, heuristic_weight=1.0):

        # --- Robot & Planning Parameters ---
        self.WHEELBASE = wheelbase
        self.STEP_SIZE = step_size # Meters
        self.MAX_STEERING = max_steering_angle
        self.NUM_STEERING_ANGLES = num_steering_angles
        self.OBSTACLE_THRESHOLD = obstacle_threshold
        self.ROBOT_LENGTH = robot_length # Meters
        self.ROBOT_WIDTH = robot_width   # Meters
        self.NUM_ANGLE_BINS = num_angle_bins
        self.ANGLE_BIN_SIZE = 2.0 * math.pi / self.NUM_ANGLE_BINS
        self.HEURISTIC_WEIGHT = heuristic_weight

        # --- Precompute Steering Actions ---
        if self.NUM_STEERING_ANGLES > 1:
            self.steering_angles = np.linspace(-self.MAX_STEERING, self.MAX_STEERING, self.NUM_STEERING_ANGLES)
        else:
            self.steering_angles = [0.0]

        # --- Precompute Robot Footprint (relative to robot center [0,0] in meters) ---
        half_L = self.ROBOT_LENGTH / 2.0
        half_W = self.ROBOT_WIDTH / 2.0
        self.footprint_rel = [(half_L, half_W), (half_L, -half_W), (-half_L, -half_W), (-half_L, half_W)]

        # --- Map Data Storage (Set via set_map) ---
        self.map_data = None
        self.map_resolution = 0.1 # Default, overridden by set_map
        self.map_height = 0
        self.map_width = 0

        # --- Goal Info (Set within find_path_internal) ---
        self.goal_center_x_world = 0.0
        self.goal_center_y_world = 0.0
        self.goal_min_row = 0
        self.goal_max_row = 0
        self.goal_min_col = 0
        self.goal_max_col = 0

        print("Hybrid A* Planner Initialized (Simplified Interface).")

    def set_map(self, map_data_array, resolution):
        """Stores the map data and resolution."""
        if not isinstance(map_data_array, np.ndarray) or map_data_array.ndim != 2:
            raise ValueError("map_data_array must be a 2D NumPy array.")
        if resolution <= 0:
             raise ValueError("Resolution must be positive.")

        self.map_data = map_data_array
        self.map_resolution = resolution
        self.map_height, self.map_width = map_data_array.shape
        print(f"Map Set: Dimensions={self.map_width}x{self.map_height}, Resolution={self.map_resolution:.3f}")

    def _discretize_state(self, x_world, y_world, theta_world):
        """Converts world state to a discrete tuple for visited sets."""
        row, col = world_to_map(x_world, y_world, self.map_resolution)
        theta_bin = int((normalize_angle(theta_world) + math.pi) / self.ANGLE_BIN_SIZE) % self.NUM_ANGLE_BINS
        return row, col, theta_bin

    def _simulate_motion(self, x_world, y_world, theta_world, steering_angle):
        """Simulates motion in the fixed world frame (meters)."""
        # --- Kinematic simulation code remains IDENTICAL ---
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING, self.MAX_STEERING)
        if abs(steering_angle) < 1e-6:
            next_x_world = x_world + self.STEP_SIZE * math.cos(theta_world)
            next_y_world = y_world + self.STEP_SIZE * math.sin(theta_world)
            next_theta_world = theta_world
        else:
            turn_radius = self.WHEELBASE / math.tan(steering_angle)
            beta = self.STEP_SIZE / turn_radius
            next_theta_world = normalize_angle(theta_world + beta)
            next_x_world = x_world + turn_radius * (math.sin(next_theta_world) - math.sin(theta_world))
            next_y_world = y_world - turn_radius * (math.cos(next_theta_world) - math.cos(theta_world))
        return next_x_world, next_y_world, next_theta_world

    def _is_collision_free(self, x_world, y_world, theta_world):
        """Checks collision using world pose and map grid."""
        if self.map_data is None: return False

        # --- 1. Calculate World Coords of Footprint Corners ---
        cos_t, sin_t = math.cos(theta_world), math.sin(theta_world)
        corners_world = []
        for corner_x_rel, corner_y_rel in self.footprint_rel: # Use footprint in meters
            wx = x_world + (corner_x_rel * cos_t - corner_y_rel * sin_t)
            wy = y_world + (corner_x_rel * sin_t + corner_y_rel * cos_t)
            corners_world.append((wx, wy))

        # --- 2. Calculate Grid Cell Indices of Footprint Corners ---
        corners_grid = []
        try:
            for wx, wy in corners_world:
                 row, col = world_to_map(wx, wy, self.map_resolution)
                 corners_grid.append((row, col))
        except ValueError: return False # Should not happen if resolution is set

        # --- 3. Out-of-Bounds Check ---
        for r, c in corners_grid:
             if not (0 <= r < self.map_height and 0 <= c < self.map_width): return False

        # --- 4. Bounding Box & Occupancy Check ---
        min_r = min(r for r, c in corners_grid)
        max_r = max(r for r, c in corners_grid)
        min_c = min(c for r, c in corners_grid)
        max_c = max(c for r, c in corners_grid)
        min_r_clipped = max(0, min_r)
        max_r_clipped = min(self.map_height - 1, max_r)
        min_c_clipped = max(0, min_c)
        max_c_clipped = min(self.map_width - 1, max_c)

        try:
            subgrid = self.map_data[min_r_clipped : max_r_clipped + 1, min_c_clipped : max_c_clipped + 1]
            if np.any((subgrid >= self.OBSTACLE_THRESHOLD) | (subgrid == -1)): return False
        except IndexError: return False # Treat index errors as collision

        return True # Collision free

    def _heuristic(self, x_world, y_world):
        """Calculates heuristic cost to goal center in world frame."""
        dx = self.goal_center_x_world - x_world
        dy = self.goal_center_y_world - y_world
        return self.HEURISTIC_WEIGHT * np.hypot(dx, dy)

    def reconstruct_path(self, came_from, current_discrete, current_state_world):
        """Rebuilds path, returns list of states in WORLD frame."""
        path_world = [current_state_world]
        while current_discrete in came_from:
            parent_discrete, parent_state_world = came_from[current_discrete]
            path_world.append(parent_state_world)
            current_discrete = parent_discrete
        return path_world[::-1]

    def find_path_internal(self, start_state_world, goal_region_map):
        """Internal planner, operates in fixed world frame."""
        # --- 1. Setup & Validation ---
        if self.map_data is None:
            print("Error: Map data not set. Call set_map() first.")
            return None

        self.goal_min_row, self.goal_max_row, self.goal_min_col, self.goal_max_col = goal_region_map
        # Basic validation of goal region map coordinates
        if not (0 <= self.goal_min_row <= self.goal_max_row < self.map_height and
                0 <= self.goal_min_col <= self.goal_max_col < self.map_width):
             print(f"Error: Goal region map coordinates {goal_region_map} invalid/outside map bounds.")
             return None # Fail if goal is fundamentally invalid

        # Calculate goal center in WORLD coords for heuristic
        goal_center_row = (self.goal_min_row + self.goal_max_row) / 2.0
        goal_center_col = (self.goal_min_col + self.goal_max_col) / 2.0
        self.goal_center_x_world, self.goal_center_y_world = map_to_world(
            goal_center_row, goal_center_col, self.map_resolution
        )
        print(f"Goal Region (Map Coords): Row=[{self.goal_min_row}-{self.goal_max_row}], Col=[{self.goal_min_col}-{self.goal_max_col}]")
        print(f"Goal Center (World Coords): ({self.goal_center_x_world:.2f}, {self.goal_center_y_world:.2f})")


        # Check start collision
        if not self._is_collision_free(*start_state_world):
             print(f"Error: Start state world({start_state_world[0]:.2f},{start_state_world[1]:.2f}) is in collision!")
             return None

        # --- 2. Initialization ---
        open_set = []
        start_discrete = self._discretize_state(*start_state_world)
        start_g_score = 0.0
        start_h_score = self._heuristic(start_state_world[0], start_state_world[1])
        start_f_score = start_g_score + start_h_score
        heapq.heappush(open_set, (start_f_score, start_state_world))

        came_from = {}
        cost_so_far = {start_discrete: start_g_score}

        print("Hybrid A* Planning started (Fixed World Frame)...")
        start_time = time.time()
        nodes_expanded = 0

        # --- 3. Main A* Search Loop ---
        while open_set:
            nodes_expanded += 1
            current_f_score, current_state_world = heapq.heappop(open_set)
            current_discrete = self._discretize_state(*current_state_world)
            current_g_score = cost_so_far.get(current_discrete, float('inf'))

            # Stale node check
            expected_f_score = current_g_score + self._heuristic(current_state_world[0], current_state_world[1])
            if current_f_score > expected_f_score + 1e-5: continue

            # --- Goal Check (convert current world pose to map cell) ---
            current_row, current_col = world_to_map(current_state_world[0], current_state_world[1], self.map_resolution)
            is_in_goal_map_region = (self.goal_min_row <= current_row <= self.goal_max_row and
                                     self.goal_min_col <= current_col <= self.goal_max_col)

            if is_in_goal_map_region:
                 print(f"Hybrid A*: Goal MAP region reached at cell ({current_row}, {current_col})! "
                       f"Planning time: {time.time() - start_time:.3f}s ({nodes_expanded} nodes)")
                 # Path reconstruction returns WORLD coordinates
                 return self.reconstruct_path(came_from, current_discrete, current_state_world)

            # --- Expand Neighbors ---
            for steering in self.steering_angles:
                next_state_world = self._simulate_motion(*current_state_world, steering)
                if not self._is_collision_free(*next_state_world): continue

                new_cost = current_g_score + self.STEP_SIZE
                next_discrete = self._discretize_state(*next_state_world)

                if new_cost < cost_so_far.get(next_discrete, float('inf')):
                    cost_so_far[next_discrete] = new_cost
                    heuristic_cost = self._heuristic(next_state_world[0], next_state_world[1])
                    priority = new_cost + heuristic_cost
                    heapq.heappush(open_set, (priority, next_state_world))
                    came_from[next_discrete] = (current_discrete, current_state_world)

        # --- 4. No Path Found ---
        print(f"Warning: Hybrid A* failed to find path after {time.time() - start_time:.3f}s ({nodes_expanded} nodes).")
        return None # Return None on failure

# ==============================================================================
# Main Planning Function Interface (Simplified)
# ==============================================================================

# --- Default planner instance ---
default_planner = HybridAStarPlanner(
    wheelbase=0.3, step_size=0.15, max_steering_angle=0.6, num_steering_angles=20,
    robot_length=0.4, robot_width=0.3, obstacle_threshold=65, heuristic_weight=1.5
)

# --- Simplified High-level function ---
def plan_path_to_goal_region(
        start_pose_map,      # DIRECTLY: (row, col, theta_rad)
        goal_region_map,     # DIRECTLY: (min_r, max_r, min_c, max_c)
        occupancy_map,       # Map array
        resolution,          # Meters per cell
        planner=default_planner):
    """
    Plans path using Hybrid A*. Inputs/Outputs are in MAP coordinates.

    Args:
        start_pose_map (tuple): Start pose in MAP coordinates (row, col, theta_radians).
        goal_region_map (tuple): Goal region in MAP coordinates (min_r, max_r, min_c, max_c).
        occupancy_map (numpy.ndarray): 2D numpy grid map.
        resolution (float): Map resolution (meters per cell).
        planner (HybridAStarPlanner, optional): Existing planner instance.

    Returns:
        list or None: Waypoints [(row, col, theta_rad), ...] in MAP coordinates, or None.
    """
    print("\n--- New Planning Request (Simplified Interface) ---")
    if planner is None: print("Error: Planner instance is not provided."); return None

    # 1. Set map data in planner
    try:
        planner.set_map(occupancy_map, resolution)
    except ValueError as e: print(f"Error setting map: {e}"); return None

    # 2. Convert MAP start pose to internal WORLD pose
    try:
        start_row, start_col, start_theta = start_pose_map
        # Validate start map coords are within map bounds
        if not (0 <= start_row < planner.map_height and 0 <= start_col < planner.map_width):
             raise ValueError(f"Start map coordinates ({start_row}, {start_col}) are outside map dimensions.")

        start_x_world, start_y_world = map_to_world(start_row, start_col, resolution)
        start_theta_normalized = normalize_angle(start_theta)
        start_pose_world = (start_x_world, start_y_world, start_theta_normalized)
        print(f"Planning from MAP start: (row={start_row}, col={start_col}, theta={start_theta:.2f})")
        print(f"Internal WORLD start: ({start_pose_world[0]:.2f}m, {start_pose_world[1]:.2f}m, {start_pose_world[2]:.2f}rad)")
    except Exception as e:
        print(f"Error converting start pose from map to world: {e}"); return None

    # 3. Call internal planner (needs world start, map goal)
    waypoints_world = planner.find_path_internal(start_pose_world, goal_region_map)

    # 4. Convert WORLD waypoints back to MAP coordinates for output
    if waypoints_world:
        print(f"Planning successful. Converting {len(waypoints_world)} world waypoints to map coordinates.")
        waypoints_map = []
        try:
            for wp_x, wp_y, wp_theta in waypoints_world:
                wp_row, wp_col = world_to_map(wp_x, wp_y, resolution)
                # Clamp to bounds just in case path goes near edge
                wp_row = max(0, min(planner.map_height - 1, wp_row))
                wp_col = max(0, min(planner.map_width - 1, wp_col))
                waypoints_map.append((wp_row, wp_col, wp_theta)) # Use clamped map coords

            if not waypoints_map: print("Warning: Conversion resulted in empty path."); return None
            print(f"Returning {len(waypoints_map)} waypoints in MAP coordinates.")
            print("--- Planning Request End ---")
            return waypoints_map
        except Exception as e:
            print(f"Error converting world waypoints to map: {e}"); return None
    else:
        print("Planning failed (internal planner returned None).")
        print("--- Planning Request End ---")
        return None