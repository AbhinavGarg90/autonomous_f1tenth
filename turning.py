import math

#default angle was 0.53 radians
L = 0.325
turning_0 = 0.53
def turning_dtheta(steering, velocity, dt):
    delta = steering.data - turning_0 # steering angle in radians + standardize
    d_theta = (velocity / L) * math.tan(delta) * dt
    print(delta, d_delta)
    return d_theta
