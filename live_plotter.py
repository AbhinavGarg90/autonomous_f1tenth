import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class LivePlotter:
    def __init__(self, origin_gt_pose, xlim=(-20, 20), ylim=(-20, 20)):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal")
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.grid(True)
        self.est_line, = self.ax.plot([], [], 'r-', label='Estimated Trajectory')
        self.gt_line, = self.ax.plot([], [], 'g-', label='Ground Truth Trajectory')
        self.ax.legend()

        self.est_xs, self.est_ys = [], []
        self.gt_xs, self.gt_ys = [], []

        self.est_arrow = None
        self.gt_arrow = None
        self.arrow_length = 1.0
        self.origin_gt_pose = origin_gt_pose

    def update(self, est_pose, gt_pose):
        est_x, est_y, est_theta = est_pose
        gt_x, gt_y, gt_theta = gt_pose

        # Adjust gt to origin
        gt_x -= self.origin_gt_pose[0]
        gt_y -= self.origin_gt_pose[1]
        gt_theta -= self.origin_gt_pose[2]

        # Update trajectory
        self.est_xs.append(est_x)
        self.est_ys.append(est_y)
        self.gt_xs.append(gt_x)
        self.gt_ys.append(gt_y)

        self.est_line.set_data(self.est_xs, self.est_ys)
        self.gt_line.set_data(self.gt_xs, self.gt_ys)

        # Remove old arrows
        if self.est_arrow:
            self.est_arrow.remove()
        if self.gt_arrow:
            self.gt_arrow.remove()

        # Add new arrows
        self.est_arrow = patches.FancyArrow(
            est_x, est_y,
            self.arrow_length * np.cos(est_theta),
            self.arrow_length * np.sin(est_theta),
            width=0.3, color='red'
        )
        self.gt_arrow = patches.FancyArrow(
            gt_x, gt_y,
            self.arrow_length * np.cos(gt_theta),
            self.arrow_length * np.sin(gt_theta),
            width=0.3, color='green'
        )
        self.ax.add_patch(self.est_arrow)
        self.ax.add_patch(self.gt_arrow)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

