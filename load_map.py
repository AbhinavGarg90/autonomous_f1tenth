import numpy as np
import matplotlib.pyplot as plt

# Load the numpy array from the file
occ_grid = np.load('saved_map/final_demo1.npy')

# Verify the shape
print("Shape of loaded array:", occ_grid.shape)

# Extract the desired subregion (250 to 350)


# Plot the occupancy grid subregion with a larger figure size
plt.figure(figsize=(12, 12))
plt.imshow(occ_grid, cmap='gray', origin='lower', vmin=0, vmax=100)
plt.colorbar(label='Occupancy Probability (0-100)')
plt.title('Occupancy Grid Visualization')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(False)
plt.show()
