#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------
# Standalone Hybrid A* Planner Module (Robot-Centric Frame)
# Author: Rahul's Team + ChatGPT Modifications
# Description: Plans a path given map, start (map coords), and goal (map coords).
#              Operates internally relative to the start pose (0,0,theta).
#              Outputs waypoints in absolute MAP coordinates (row, col, theta).
# -------------------------

import numpy as np
import math
import heapq
import time

# ==============================================================================
# Utility Functions (Standalone - No ROS)
# ==============================================================================

# No map_to_world needed externally, as internal frame is robot-centric.
# world_to_map is handled internally by the class using start reference.

def normalize_angle(angle):
    """Normalizes angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))

# ==============================================================================
# Hybrid A* Planner Class (Robot-Centric Frame)
# ==============================================================================

class HybridAStarPlanner:
    """
    Implements Hybrid A* logic using a robot-centric internal frame.
    Input/Output uses absolute map coordinates.
    """
    def __init__(self, wheelbase, step_size, max_steering_angle, num_steering_angles,
                 obstacle_threshold=65, robot_length=0.5, robot_width= 1.2,
                 num_angle_bins=72, heuristic_weight=1.0):

        # --- Robot & Planning Parameters ---
        self.WHEELBASE = wheelbase
        self.STEP_SIZE = step_size # Meters (relative displacement)
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

        # --- Map Data Storage (Set via setup) ---
        self.map_data = None
        self.map_resolution = 0.1 # Default, overridden
        self.map_height = 0
        self.map_width = 0
        self.start_row_map = 0 # Absolute row index of start
        self.start_col_map = 0 # Absolute col index of start

        # --- Goal Info (Set within find_path_internal) ---
        # Goal position relative to start (in meters) for heuristic
        self.goal_center_x_rel = 0.0
        self.goal_center_y_rel = 0.0
        # Goal region in absolute map coordinates
        self.goal_min_row = 0
        self.goal_max_row = 0
        self.goal_min_col = 0
        self.goal_max_col = 0

        print("Hybrid A* Planner Initialized (Robot-Centric).")

    # --- NEW: Internal coordinate conversion helper ---
    def _robot_frame_to_map_cell(self, x_robot, y_robot):
        """
        Converts robot-centric coordinates (meters relative to start)
        into absolute map cell indices (row, col).
        Uses the stored start_row_map, start_col_map, and map_resolution.
        """
        # Calculate change in grid cells from start
        delta_col = x_robot / self.map_resolution
        delta_row = y_robot / self.map_resolution

        # Add delta to start cell to get current absolute cell
        # Use round() or floor() before int() for potentially better centering
        current_col = int(round(delta_col + self.start_col_map))
        current_row = int(round(delta_row + self.start_row_map))

        return current_row, current_col

    # --- NEW: Setup function ---
    def setup(self, occupancy_map, resolution, start_pose_map):
        """Stores map, resolution, and start pose map coordinates."""
        if not isinstance(occupancy_map, np.ndarray) or occupancy_map.ndim != 2:
            raise ValueError("occupancy_map must be a 2D NumPy array.")
        if resolution <= 0:
             raise ValueError("Resolution must be positive.")
        if not isinstance(start_pose_map, tuple) or len(start_pose_map) != 3:
             raise ValueError("start_pose_map must be tuple (row, col, theta)")

        self.map_data = occupancy_map
        self.map_resolution = resolution
        self.map_height, self.map_width = occupancy_map.shape
        self.start_row_map, self.start_col_map, _ = start_pose_map

        # Validate start map coords
        if not (0 <= self.start_row_map < self.map_height and 0 <= self.start_col_map < self.map_width):
             raise ValueError(f"Start map coords ({self.start_row_map}, {self.start_col_map}) are outside map bounds.")

        print(f"Planner Setup: Map={self.map_width}x{self.map_height}, Res={self.map_resolution:.3f}, "
              f"StartMap=({self.start_row_map}, {self.start_col_map})")


    def _discretize_state(self, x_robot, y_robot, theta_robot):
        """Converts robot-centric state to a discrete tuple for visited sets."""
        # Use the helper to get absolute map cell for discretization key
        row, col = self._robot_frame_to_map_cell(x_robot, y_robot)
        # Angle bin is relative to robot's frame (which is the planning frame)
        theta_bin = int((normalize_angle(theta_robot) + math.pi) / self.ANGLE_BIN_SIZE) % self.NUM_ANGLE_BINS
        # Key uses absolute row/col but relative theta binning
        return row, col, theta_bin


    def _simulate_motion(self, x_robot, y_robot, theta_robot, steering_angle):
        """Simulates motion in the robot-centric frame (meters relative to start)."""
        # --- Kinematic simulation code operates on relative coords ---
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING, self.MAX_STEERING)
        if abs(steering_angle) < 1e-6:
            # Change in x/y relative to current relative pose
            next_x_robot = x_robot + self.STEP_SIZE * math.cos(theta_robot)
            next_y_robot = y_robot + self.STEP_SIZE * math.sin(theta_robot)
            next_theta_robot = theta_robot
        else:
            turn_radius = self.WHEELBASE / math.tan(steering_angle)
            beta = self.STEP_SIZE / turn_radius
            next_theta_robot = normalize_angle(theta_robot + beta)
            # These give the *new* relative position from start
            next_x_robot = x_robot + turn_radius * (math.sin(next_theta_robot) - math.sin(theta_robot))
            next_y_robot = y_robot - turn_radius * (math.cos(next_theta_robot) - math.cos(theta_robot))
        return next_x_robot, next_y_robot, next_theta_robot


    def _is_collision_free(self, x_robot, y_robot, theta_robot):
        """Checks collision using robot-centric pose and map grid."""
        if self.map_data is None: return False

        # --- 1. Calculate Footprint Corners relative to robot's CURRENT pose (in robot-centric frame) ---
        cos_t, sin_t = math.cos(theta_robot), math.sin(theta_robot)
        corners_relative_to_current_pose = []
        for corner_x_rel_center, corner_y_rel_center in self.footprint_rel: # footprint_rel is in meters
            # Coords of corner relative to robot center (in robot-centric frame)
            cx = x_robot + (corner_x_rel_center * cos_t - corner_y_rel_center * sin_t)
            cy = y_robot + (corner_x_rel_center * sin_t + corner_y_rel_center * cos_t)
            corners_relative_to_current_pose.append((cx, cy))

        # --- 2. Convert each corner's relative coords to ABSOLUTE map cell ---
        corners_grid = []
        try:
            for cx_robot, cy_robot in corners_relative_to_current_pose:
                 # Convert robot frame coord (relative to start) to absolute map cell
                 row, col = self._robot_frame_to_map_cell(cx_robot, cy_robot)
                 corners_grid.append((row, col))
        except ValueError: return False # Should not happen if resolution is set

        # --- 3. Out-of-Bounds Check (using absolute map cells) ---
        for r, c in corners_grid:
             if not (0 <= r < self.map_height and 0 <= c < self.map_width): return False

        # --- 4. Bounding Box & Occupancy Check (using absolute map cells) ---
        min_r = min(r for r, c in corners_grid)
        max_r = max(r for r, c in corners_grid)
        min_c = min(c for r, c in corners_grid)
        max_c = max(c for r, c in corners_grid)
        min_r_clipped = max(0, min_r)
        max_r_clipped = min(self.map_height - 1, max_r)
        min_c_clipped = max(0, min_c)
        max_c_clipped = min(self.map_width - 1, max_c)

        try:
            # Check the corresponding subgrid on the map_data
            subgrid = self.map_data[min_r_clipped : max_r_clipped + 1, min_c_clipped : max_c_clipped + 1]
            if np.any((subgrid >= self.OBSTACLE_THRESHOLD) | (subgrid == -1)): return False
        except IndexError: return False

        return True # Collision free


    def _heuristic(self, x_robot, y_robot):
        """Calculates heuristic cost using poses relative to start."""
        # Use the pre-calculated goal center relative to start
        dx = self.goal_center_x_rel - x_robot
        dy = self.goal_center_y_rel - y_robot
        return self.HEURISTIC_WEIGHT * np.hypot(dx, dy)


    def reconstruct_path(self, came_from, current_discrete, current_state_robot):
        """
        Rebuilds path. came_from stores parent states in ROBOT-CENTRIC frame.
        Returns list of states in ROBOT-CENTRIC frame [(x_robot, y_robot, theta_robot), ...].
        """
        path_robot = [current_state_robot]
        while current_discrete in came_from:
            # came_from stores parent state in robot-centric coordinates
            parent_discrete, parent_state_robot = came_from[current_discrete]
            path_robot.append(parent_state_robot)
            current_discrete = parent_discrete
        return path_robot[::-1]


    def find_path_internal(self, start_theta, goal_region_map):
        """
        Internal planner. Operates relative to start pose (0,0,start_theta).
        Needs absolute goal region map coords to know when to stop and for heuristic calculation.
        """
        # --- 1. Setup & Validation ---
        if self.map_data is None: print("Error: Map not set via setup()."); return None

        self.goal_min_row, self.goal_max_row, self.goal_min_col, self.goal_max_col = goal_region_map
        if not (0 <= self.goal_min_row <= self.goal_max_row < self.map_height and
                0 <= self.goal_min_col <= self.goal_max_col < self.map_width):
             print(f"Error: Goal region map coordinates invalid/outside map."); return None

        # Calculate goal center RELATIVE TO START in meters for heuristic
        goal_center_row = (self.goal_min_row + self.goal_max_row) / 2.0
        goal_center_col = (self.goal_min_col + self.goal_max_col) / 2.0
        # Calculate relative displacement in cells
        delta_col_goal = goal_center_col - self.start_col_map
        delta_row_goal = goal_center_row - self.start_row_map
        # Convert cell displacement to meter displacement (relative coords)
        self.goal_center_x_rel = delta_col_goal * self.map_resolution
        self.goal_center_y_rel = delta_row_goal * self.map_resolution

        print(f"Goal Region (Map Coords): Row=[{self.goal_min_row}-{self.goal_max_row}], Col=[{self.goal_min_col}-{self.goal_max_col}]")
        print(f"Goal Center (Relative Coords): ({self.goal_center_x_rel:.2f}m, {self.goal_center_y_rel:.2f}m)")


        # Define start state in ROBOT-CENTRIC frame (always 0,0 + initial theta)
        start_state_robot = (0.0, 0.0, normalize_angle(start_theta))

        # Check start collision (at relative coords 0,0)
        if not self._is_collision_free(*start_state_robot):
             print(f"Error: Start state map({self.start_row_map},{self.start_col_map}) is in collision!"); return None

        # --- 2. Initialization (using ROBOT-CENTRIC states) ---
        open_set = []
        # Discrete key uses absolute map cell derived from (0,0) + start_cell
        start_discrete = self._discretize_state(*start_state_robot)
        start_g_score = 0.0
        # Heuristic is calculated from robot frame (0,0) to relative goal center
        start_h_score = self._heuristic(start_state_robot[0], start_state_robot[1])
        start_f_score = start_g_score + start_h_score
        heapq.heappush(open_set, (start_f_score, start_state_robot)) # Heap stores robot-centric states

        came_from = {} # Keys: discrete states (abs_row, abs_col, theta_bin)
                       # Values: (parent_discrete, parent_robot_state)
        cost_so_far = {start_discrete: start_g_score} # Keys are discrete states

        print("Hybrid A* Planning started (Robot-Centric Frame)...")
        start_time = time.time()
        nodes_expanded = 0

        # --- 3. Main A* Search Loop (operates on ROBOT-CENTRIC states) ---
        while open_set:
            nodes_expanded += 1
            # Pop the state with the lowest f-score (state is relative to start)
            current_f_score, current_state_robot = heapq.heappop(open_set)
            # Get the discrete key for this state (based on absolute map cell)
            current_discrete = self._discretize_state(*current_state_robot)
            current_g_score = cost_so_far.get(current_discrete, float('inf'))

            # Stale node check
            expected_f_score = current_g_score + self._heuristic(current_state_robot[0], current_state_robot[1])
            if current_f_score > expected_f_score + 1e-5: continue

            # --- Goal Check ---
            # Convert current ROBOT-CENTRIC pose to ABSOLUTE map cell indices
            current_row_abs, current_col_abs = self._robot_frame_to_map_cell(
                current_state_robot[0], current_state_robot[1]
            )
            # Check if these ABSOLUTE map indices are inside the goal MAP region
            is_in_goal_map_region = (self.goal_min_row <= current_row_abs <= self.goal_max_row and
                                     self.goal_min_col <= current_col_abs <= self.goal_max_col)

            if is_in_goal_map_region:
                 print(f"Goal MAP region reached at cell ({current_row_abs}, {current_col_abs})!")
                 # Reconstruct path - returns waypoints in ROBOT-CENTRIC coordinates
                 return self.reconstruct_path(came_from, current_discrete, current_state_robot)

            # --- Expand Neighbors (calculate next state relative to start)---
            for steering in self.steering_angles:
                # Simulate motion - result is next pose relative to start
                next_state_robot = self._simulate_motion(*current_state_robot, steering)
                # Check collision using the robot-centric pose (converts to map cells internally)
                if not self._is_collision_free(*next_state_robot): continue

                # Calculate cost
                new_cost = current_g_score + self.STEP_SIZE
                # Get discrete key for the next state (based on absolute map cell)
                next_discrete = self._discretize_state(*next_state_robot)

                # If this path to next_discrete is better...
                if new_cost < cost_so_far.get(next_discrete, float('inf')):
                    cost_so_far[next_discrete] = new_cost
                    # Heuristic calculated from the robot-centric state to relative goal
                    heuristic_cost = self._heuristic(next_state_robot[0], next_state_robot[1])
                    priority = new_cost + heuristic_cost
                    # Push ROBOT-CENTRIC state onto heap
                    heapq.heappush(open_set, (priority, next_state_robot))
                    # Record parent (store parent's ROBOT-CENTRIC state)
                    came_from[next_discrete] = (current_discrete, current_state_robot)

        # --- 4. No Path Found ---
        print(f"Warning: Hybrid A* failed to find path after {time.time() - start_time:.3f}s ({nodes_expanded} nodes).")
        return None

# ==============================================================================
# Main Planning Function Interface (Simplified)
# ==============================================================================

default_planner = HybridAStarPlanner(
    wheelbase=0.3, step_size=0.2, max_steering_angle=0.35, num_steering_angles=20,
    robot_length=0.5, robot_width=0.45, obstacle_threshold=65, heuristic_weight=1.5
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
    Uses a robot-centric frame internally.

    Args:
        start_pose_map (tuple): Start pose in MAP coordinates (row, col, theta_radians).
        goal_region_map (tuple): Goal region in MAP coordinates (min_r, max_r, min_c, max_c).
        occupancy_map (numpy.ndarray): 2D numpy grid map.
        resolution (float): Map resolution (meters per cell).
        planner (HybridAStarPlanner, optional): Existing planner instance.

    Returns:
        list or None: Waypoints [(row, col, theta_rad), ...] in MAP coordinates, or None.
    """
    print("\n--- New Planning Request (Robot-Centric Interface) ---")
    if planner is None: print("Error: Planner instance not provided."); return None

    # 1. Setup planner with map, resolution, and start pose (map coords)
    try:
        planner.setup(occupancy_map, resolution, start_pose_map)
    except ValueError as e: print(f"Error during setup: {e}"); return None

    # 2. Extract start theta for internal planner
    start_theta = start_pose_map[2] # Planner starts at (0,0) internally with this theta

    # 3. Call internal planner (needs start_theta, map goal)
    #    This function returns waypoints in ROBOT-CENTRIC coordinates (relative to start)
    waypoints_robot_frame = planner.find_path_internal(start_theta, goal_region_map)

    # 4. Convert resulting ROBOT-CENTRIC waypoints back to absolute MAP coordinates
    if waypoints_robot_frame:
        print(f"Planning successful. Converting {len(waypoints_robot_frame)} robot-centric waypoints to map coordinates.")
        waypoints_map = []
        start_row = planner.start_row_map
        start_col = planner.start_col_map
        res = planner.map_resolution
        map_h = planner.map_height
        map_w = planner.map_width
        try:
            for x_robot, y_robot, theta_robot in waypoints_robot_frame:
                # Convert robot frame pose (relative displacement) to absolute map cell
                wp_row_abs, wp_col_abs = planner._robot_frame_to_map_cell(x_robot, y_robot)

                # Clamp to bounds to ensure valid indices
                wp_row_abs = max(0, min(map_h - 1, wp_row_abs))
                wp_col_abs = max(0, min(map_w - 1, wp_col_abs))

                # Append absolute map coordinates (row, col, theta)
                waypoints_map.append((wp_row_abs, wp_col_abs, theta_robot))

            if not waypoints_map: print("Warning: Conversion resulted in empty path."); return None
            print(f"Returning {len(waypoints_map)} waypoints in MAP coordinates.")
            print("--- Planning Request End ---")
            return waypoints_map
        except Exception as e:
            print(f"Error converting robot-centric waypoints to map: {e}"); return None
    else:
        print("Planning failed (internal planner returned None).")
        print("--- Planning Request End ---")
        return None