import numpy as np
import matplotlib.pyplot as plt

# Load pose data from CSV (assumes no header)
pose_array = np.loadtxt("poses.csv", delimiter=",")

def plot_pose_pairs(pose_array):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    for entry in pose_array:
        curr_x, curr_y, curr_theta = entry[0], entry[1], entry[2]
        tgt_x, tgt_y, tgt_theta = entry[3], entry[4], entry[5]
        
        # Plot current pose arrow (blue)
        ax.quiver(curr_x, curr_y,
                  np.cos(curr_theta), np.sin(curr_theta),
                  angles='xy', scale_units='xy', scale=1, color='blue')

        # Plot target pose arrow (green)
        ax.quiver(tgt_x, tgt_y,
                  np.cos(tgt_theta), np.sin(tgt_theta),
                  angles='xy', scale_units='xy', scale=1, color='green')

        # Draw correction vector (red dashed)
        ax.annotate("",
                    xy=(curr_x, curr_y), xytext=(tgt_x, tgt_y),
                    arrowprops=dict(arrowstyle="->", color='red', linestyle='dashed'))

    ax.set_title("Current vs Target Poses")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    plt.legend(["Current Pose", "Target Pose", "Correction"], loc="upper right")
    plt.show()

plot_pose_pairs(pose_array)
