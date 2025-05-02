import numpy as np


class PoseIntegrator():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.theta = 0

    def update_position(self, dx, dy):
        self.x += dx
        self.y += dy

    def update_theta(self, dtheta_icp, dtheta_steering):
        self.theta += dtheta_icp
    
    def get_pose(self):
        return (self.x, self.y, self.theta)

def get_dxdy(velocity, theta, dt):
    """
    prev_pose: tuple (x, y, theta)
    returns: updated_pose (x_new, y_new, theta)
    """
    dx = velocity * dt * np.cos(theta)
    dy = velocity * dt * np.sin(theta)

    return dx, dy

