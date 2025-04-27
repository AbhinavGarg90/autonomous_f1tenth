import numpy as np
from scipy.spatial import KDTree

class ICPLocalizer:
    def __init__(self, vehicle_length=5, time_step=0.1, num_points=1000):
        self.vehicle_length = vehicle_length
        self.time_step = time_step
        self.num_points = num_points
        self.old_scan = None
        self.position = (0.0, 0.0)
        self.theta = 0.0
        self.x_history = [0.0]
        self.y_history = [0.0]

    def initialize(self, initial_scan):
        self.old_scan = initial_scan
    
    def update(self, new_scan):
        rotation, translation, _ = self.icp(self.old_scan, new_scan)
        translation = translation * 2.0
        delta_theta = np.arctan2(rotation[1, 0], rotation[0, 0])

        R_global = np.array([
            [np.cos(self.theta), -np.sin(self.theta)],
            [np.sin(self.theta),  np.cos(self.theta)]
        ])

        self.theta -= delta_theta
        self.position -= R_global @ translation
        # self.theta    += delta_theta
        # self.position += R_global @ translation

        self.x_history.append(self.position[0])
        self.y_history.append(self.position[1])

        self.old_scan = new_scan
        return self.position[0], self.position[1], self.theta

    def icp(self, old_scan, new_scan, max_iterations=100, tolerance=1e-6):
        src = np.array(old_scan, copy=True)
        dst = np.array(new_scan, copy=True)
        old_error = 0.0

        for _ in range(max_iterations):
            tree = KDTree(dst)
            distances, indices = tree.query(src)
            R, t = self.point_to_point_transform(src, dst[indices])
            src = (R @ src.T).T + t
            mean_error = np.mean(distances)
            if abs(old_error - mean_error) < tolerance:
                break
            old_error = mean_error

        return self.point_to_point_transform(old_scan, src) + (src,)

    def point_to_point_transform(self, old_scan, new_scan):
        assert old_scan.shape == new_scan.shape
        old_cent = np.mean(old_scan, axis=0)
        new_cent = np.mean(new_scan, axis=0)
        rel_old = old_scan - old_cent
        rel_new = new_scan - new_cent
        H = rel_old.T @ rel_new
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = new_cent - R @ old_cent
        return R, t
