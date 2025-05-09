import numpy as np
import matplotlib.pyplot as plt
import re

def parse_log_file(filename):
    pose_data = []
    with open(filename, "r") as file:
        lines = file.readlines()

    for i in range(0, len(lines), 3):  # Each data point spans 3 lines
        if i + 1 >= len(lines):
            break
        
        curr_match = re.findall(r"-?\d+\.?\d*", lines[i])
        tgt_match = re.findall(r"-?\d+\.?\d*", lines[i + 1])

        if len(curr_match) >= 3 and len(tgt_match) >= 3:
            curr_x, curr_y = float(curr_match[0]), float(curr_match[1])
            tgt_x, tgt_y = float(tgt_match[0]), float(tgt_match[1])
            pose_data.append([curr_x, curr_y, tgt_x, tgt_y])

    return np.array(pose_data)

def plot_correction_vectors(pose_array, trim_last_n=0):
    if trim_last_n > 0:
        pose_array = pose_array[:-trim_last_n]

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    for entry in pose_array:
        curr_x, curr_y, tgt_x, tgt_y = entry

        # Plot dots
        ax.plot(curr_x, curr_y, 'bo')  # Blue = current
        ax.plot(tgt_x, tgt_y, 'go')    # Green = target

        # Arrow from current to target
        ax.annotate("",
                    xy=(tgt_x, tgt_y), xytext=(curr_x, curr_y),
                    arrowprops=dict(arrowstyle="->", color='red'))

    ax.set_title("Current → Target Pose Arrows")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    plt.legend(["Current", "Target"], loc="upper right")
    plt.show()

# ==== USAGE ====
pose_array = parse_log_file("good_log.txt")
plot_correction_vectors(pose_array, trim_last_n=21)  # Change to any number
