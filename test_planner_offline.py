

import numpy as np
import math
import time # For pausing in simulation

# --- Attempt to import the planner module ---
try:
    import waypoint_planner as planner_module
    from waypoint_planner import GOAL_X_MIN, GOAL_X_MAX, GOAL_Y_MIN, GOAL_Y_MAX
except ImportError:
    print("Error: Could not import hybrid_astar_planner.")
    print("Please ensure 'waypoint_planner.py' is in the same directory or Python path.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    exit()

# --- Attempt to import matplotlib ---
matplotlib_available = False
plt = None
Rectangle = None
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    matplotlib_available = True
    print("Matplotlib found. Visualization will be enabled.")
except ImportError:
    print("Warning: Matplotlib not found. Skipping visualization.")
    print("         To visualize, install it: pip install matplotlib")

# ==============================================================
# Main Test Execution Block
# ==============================================================
if __name__ == '__main__':

    print("\n--- Starting Offline Hybrid A* Test ---")

    # ----------------------------------------------------------
    # 1. Define Mock Map Data and Metadata
    # ----------------------------------------------------------
    print("1. Creating Mock Map...")
    resolution = 0.1
    origin_x = -5.0
    origin_y = -5.0
    map_height_cells = 100
    map_width_cells = 100
    mock_map_data = np.zeros((map_height_cells, map_width_cells), dtype=np.int8)
    obstacle_val = 100
    mock_map_data[20:80, 70:73] = obstacle_val # Wall 1
    mock_map_data[40:43, 20:70] = obstacle_val # Wall 2
    mock_map_data[65:75, 35:45] = obstacle_val # Box

    print(f"   Map Size: {map_width_cells}x{map_height_cells} cells, Resolution: {resolution} m/cell")
    print(f"   Origin: ({origin_x:.2f}, {origin_y:.2f}) m")
    print(f"   Goal Region: X=[{GOAL_X_MIN:.2f},{GOAL_X_MAX:.2f}], Y=[{GOAL_Y_MIN:.2f},{GOAL_Y_MAX:.2f}]")

    # ----------------------------------------------------------
    # 2. Define Start Pose
    # ----------------------------------------------------------
    start_pose = (0.0, 0.0, 0.0)
    print(f"2. Start Pose: ({start_pose[0]:.2f}m, {start_pose[1]:.2f}m, {start_pose[2]:.2f}rad)")

    # ----------------------------------------------------------
    # 3. Call the Planner
    # ----------------------------------------------------------
    print("3. Calling planner...")
    waypoints = planner_module.plan_path_to_goal_region(
        start_pose, mock_map_data, resolution, origin_x, origin_y
    )

    # ----------------------------------------------------------
    # 4. Process Results & Visualize
    # ----------------------------------------------------------
    print("4. Processing results...")

    final_plot_title = "Hybrid A* Offline Test" # Default title

    # --- Setup Plotting Objects (only if matplotlib is available) ---
    fig, ax = None, None
    if matplotlib_available:
        try:
            fig, ax = plt.subplots(figsize=(9, 9))
            world_extent = [origin_x, origin_x + map_width_cells * resolution,
                            origin_y, origin_y + map_height_cells * resolution]

            # Plot Map, Goal Region, Start Pose (always shown if plotting is possible)
            ax.imshow(mock_map_data, cmap='Greys', origin='lower', extent=world_extent, vmin=0, vmax=100, interpolation='none')
            goal_width = GOAL_X_MAX - GOAL_X_MIN
            goal_height = GOAL_Y_MAX - GOAL_Y_MIN
            goal_rect = Rectangle((GOAL_X_MIN, GOAL_Y_MIN), goal_width, goal_height,
                                  linewidth=1, edgecolor='r', facecolor='red', alpha=0.3, label='Goal Region')
            ax.add_patch(goal_rect)
            ax.plot(start_pose[0], start_pose[1], 'go', markersize=10, label='Start')
            ax.arrow(start_pose[0], start_pose[1],
                     0.5 * math.cos(start_pose[2]), 0.5 * math.sin(start_pose[2]),
                     head_width=0.2, length_includes_head=True, color='g', lw=1)

        except Exception as e:
            print(f"\nError setting up basic plot: {e}")
            matplotlib_available = False # Disable further plotting if setup fails
            fig, ax = None, None # Ensure they are None


    # --- Process Path and Add to Plot/Print ---
    if waypoints:
        print(f"   SUCCESS: Path found with {len(waypoints)} waypoints.")
        final_plot_title = f"Hybrid A* Test - Path Found ({len(waypoints)} pts)"

        if matplotlib_available and ax is not None:
            # Plot the full path
            path_x = [p[0] for p in waypoints]
            path_y = [p[1] for p in waypoints]
            ax.plot(path_x, path_y, 'b--', linewidth=0.5, label='Full Path')

            # --- Simplified Simulation ---
            try:
                print("   Attempting path simulation visualization...")
                plt.show(block=False) # Show plot window now
                plt.pause(0.5)

                sim_robot_marker, = ax.plot([], [], 'mo', markersize=8, label='Simulated Pos')
                sim_robot_arrow = None

                for i, wp in enumerate(waypoints):
                    if not plt.fignum_exists(fig.number): break # Stop if user closes window

                    sim_x, sim_y, sim_theta = wp
                    sim_robot_marker.set_data([sim_x], [sim_y])

                    arrow_len = 0.4
                    dx = arrow_len * math.cos(sim_theta)
                    dy = arrow_len * math.sin(sim_theta)
                    if sim_robot_arrow is not None: sim_robot_arrow.remove()
                    sim_robot_arrow = ax.arrow(sim_x, sim_y, dx, dy, head_width=0.15, length_includes_head=True, color='m', lw=1)

                    ax.set_title(f"Hybrid A* Simulation - Step {i+1}/{len(waypoints)}")
                    fig.canvas.draw_idle()
                    plt.pause(0.05)

                final_plot_title = f"Hybrid A* Simulation Complete ({len(waypoints)} steps)"
                print("   Simulation finished.")

            except Exception as e:
                print(f"\nError during path simulation visualization: {e}")
                # Continue without simulation if it fails

        else: # Matplotlib not available or plot setup failed, print to console
             print("\n   Path Waypoints (x, y, theta):")
             for i, p in enumerate(waypoints):
                  print(f"     {i}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

    else: # No path found
        print("   FAILURE: No path found by the planner.")
        final_plot_title = "Hybrid A* Test - PLANNING FAILED"


    # --- Finalize and Show Plot (if possible) ---
    if matplotlib_available and ax is not None:
        print("   Displaying final plot. Close plot window to exit.")
        ax.set_xlabel("X coordinate (meters)")
        ax.set_ylabel("Y coordinate (meters)")
        ax.set_title(final_plot_title)
        ax.legend()
        ax.grid(True)
        ax.axis('equal')
        plt.show(block=True) # Keep open until closed


    print("\n--- Test Script Finished ---")