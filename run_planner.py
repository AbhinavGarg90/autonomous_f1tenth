import numpy as np
import matplotlib.pyplot as plt
from waypoint_planner import plan_path_to_goal_region
import os
import csv


occupancy_map = np.load('saved_map/occ_grid.npy')
map_height, map_width = occupancy_map.shape

# 2. Define map resolution
resolution = 0.1  # meters per cell
print(f"Map metadata: Resolution={resolution}")

# 3. Define START pose DIRECTLY in MAP coordinates (row, col, theta_rad)
#    Example: Start near center if map is 600x600
start_row = 300
start_col = 300
start_theta = 0.0 # Radians
start_pose_map = (start_row, start_col, start_theta)
print(f"Start pose (Map Coords): (row={start_row}, col={start_col}, theta={start_theta:.2f})")



# 4. Define GOAL region DIRECTLY in MAP coordinates (min_row, max_row, min_col, max_col)

goal_min_row = 450 
goal_max_row = 465 
goal_min_col = 430 
goal_max_col = 445 
goal_region_map = (goal_min_row, goal_max_row, goal_min_col, goal_max_col)
print(f"Goal region (Map Coords): Rows=[{goal_min_row}-{goal_max_row}], Cols=[{goal_min_col}-{goal_max_col}]")


waypoints_map = plan_path_to_goal_region(
    start_pose_map=start_pose_map,     # Pass start in map coords
    goal_region_map=goal_region_map,   # Pass goal in map coords
    occupancy_map=occupancy_map,
    resolution=resolution
    # No need to pass origin if assuming (0,0)
    # planner=default_planner # Optional: Pass a specific planner instance
)


# === Convert waypoints from MAP coordinates to WORLD coordinates and save to CSV ===
if waypoints_map:
    waypoints_world = []
    for row, col, theta in waypoints_map:
        x_world = (col - start_col) * resolution
        y_world = (row - start_row) * resolution
        waypoints_world.append((x_world, y_world, theta))


    # Save to CSV
    output_dir = "waypoints_in_csv"
    output_file = os.path.join(output_dir, "waypoints_world.csv")
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        for x, y, theta in waypoints_world:
            writer.writerow([round(x, 4), round(y, 4), round(theta, 6)])

    print(f"\nWaypoints successfully saved to: {output_file}")

else:
    print("\nSkipping world coordinate conversion and CSV export: no waypoints.")

# 6. Plotting (using map coordinates directly)
plt.figure(figsize=(12, 12))
plt.imshow(occupancy_map, cmap='gray_r', origin='lower',
           extent=[0, map_width, 0, map_height]) # Axes match indices
plt.colorbar(label='Occupancy Value')

# Plot start position (using map coordinates)
plt.plot(start_col, start_row, 'go', markersize=10, label='Start')

# Plot goal region (using map coordinates)
goal_rect_x = [goal_min_col, goal_max_col, goal_max_col, goal_min_col, goal_min_col]
goal_rect_y = [goal_min_row, goal_min_row, goal_max_row, goal_max_row, goal_min_row]
plt.plot(goal_rect_x, goal_rect_y, 'b-', linewidth=2, label='Goal Region')

# Plot waypoints if available (already in map coordinates)
if waypoints_map:
    wp_col = [col for _, col, _ in waypoints_map] # Column index for x-axis
    wp_row = [row for row, _, _ in waypoints_map] # Row index for y-axis
    print(f"\nPlotting {len(waypoints_map)} waypoints...")
    plt.plot(wp_col, wp_row, 'r.-', markersize=4, linewidth=1.5, label='Waypoints')
    plt.legend()
else:
    print("\nNo valid path found by the planner.")
    plt.legend() # Still show legend for start/goal

plt.title('Occupancy Grid and Hybrid A* Waypoints (Map Coordinates)')
plt.xlabel('Map Column Index')
plt.ylabel('Map Row Index')
plt.axis('equal')
plt.grid(True)
plt.gca().invert_yaxis() # Often maps have row 0 at the top, imshow(origin='lower') puts 0 at bottom. Invert Y axis to match common image convention if needed. Comment out if origin='lower' is desired display.
plt.show()
