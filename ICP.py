import numpy as np
from scipy.spatial import KDTree

pose = {
    "x": 0.0,
    "y": 0.0,
    "theta": 0.0  # in radians
}

old_vel = 0.0
old_angle = 0.0

time_step = 0.1
vehicle_length = 5
num_points = 1000 #points returned from scan
point_granularity = 270/num_points #distance in degrees between each measurement
controls = []

# def get_scan():
#     # print("scan")
#     global pose

#     angles_deg = np.linspace(-135, 135, num_points) #270 degree FOV from our lidar
#     angles_rad = np.deg2rad(angles_deg)
#     radius = 10
#     noise = np.random.normal(0, 0.1, num_points)  #points shouldn't vary by this much - test how to sim walls and shi
#     distances = radius# + noise
#     x = distances * np.cos(angles_rad)
#     y = distances * np.sin(angles_rad)
#     scan = np.stack((x, y), axis=-1) #final points in cartesian coords

#     dtheta = np.deg2rad(1.0)  #rotating one degree every measurement
#     dx = 0.05  #0.05 meters every measurement? - figure out units
#     pose["theta"] += dtheta #bookkeeping
#     pose["x"] += dx * np.cos(pose["theta"])
#     pose["y"] += dx * np.sin(pose["theta"])

#     c, s = np.cos(pose["theta"]), np.sin(pose["theta"]) 
#     rotation = np.array([[c, -s], [s, c]]) #rotation matrix
#     scan = (rotation @ scan.T).T #apply that bih
#     scan += np.array([pose["x"], pose["y"]])  # translatation matrix - move it over

#     print("it is: ", dtheta, dx)

#     return scan

def get_scan():
    global pose

    angles_deg = np.linspace(-135, 135, num_points)
    angles_rad = np.deg2rad(angles_deg)
    scan = np.zeros((num_points, 2))

    # Create a square room: walls at x=10, x=-10, y=10, y=-10
    for i, angle in enumerate(angles_rad):
        # Cast a ray
        for dist in np.linspace(0.1, 20, 500):
            # Ray endpoint in robot frame
            x = dist * np.cos(angle)
            y = dist * np.sin(angle)

            # World coords of that point
            wx = pose["x"] + x * np.cos(pose["theta"]) - y * np.sin(pose["theta"])
            wy = pose["y"] + x * np.sin(pose["theta"]) + y * np.cos(pose["theta"])

            # Check if it hits a wall
            if abs(wx) >= 10 or abs(wy) >= 10:
                # noise = np.random.normal(0, 0.05)
                noise = 0
                scan[i] = [x + noise, y + noise]
                break

    # Pose update
    dtheta = np.deg2rad(1.0)
    dx = 0.05
    print("it is: ", dtheta, dx)
    pose["theta"] += dtheta
    pose["x"] += dx * np.cos(pose["theta"])
    pose["y"] += dx * np.sin(pose["theta"])

    return scan, [dx/time_step, np.deg2rad(0.5)]


def publish_transform(transform):
    # print(transform[0], transform[1]) #broadcasts R|T
    rotation = transform[0]
    translation = transform[1]
    angle = np.arctan2(rotation[1, 0], rotation[0, 0])
    displacement = np.sqrt(translation[0] ** 2 + translation[1] ** 2)
    print("we think its: ", 0-angle, displacement)

#control_inputs should be array containing current velocity and current angle
#or it could be current velocity and turning angle and we could do math?
def pre_processing(control_inputs, time_step):
    curr_vel = control_inputs[0]
    curr_turn = np.rad2deg(control_inputs[1])
    angular_velocity = (curr_vel/vehicle_length) * np.tan(curr_turn)
    delta_angle = angular_velocity * time_step #how much we've change how we're facing
    #now we have how many angles we've turned
    delta_angle = delta_angle * 0.9 #don't wanna overestimate
    cos_a = np.cos(delta_angle)
    sin_a = np.sin(delta_angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return rotation_matrix

    #do the math to figure how many points this shifts us over


#given two arrays (whose points are alr matched one to one) find how to transform the input to the output
def point_to_point_transform(old_scan, new_scan):
    #2d point cloud info
    assert old_scan.shape == new_scan.shape

    old_cent = np.mean(old_scan, axis=0)
    new_cent = np.mean(new_scan, axis=0)
    relative_old = old_scan - old_cent
    relative_new = new_scan - new_cent
    H = relative_old.T @ relative_new
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = new_cent - R @ old_cent

    return R, t

def icp(old_scan, new_scan, max_iterations=100, tolerance=1e-6):
    src = np.array(old_scan, copy=True)
    dst = np.array(new_scan, copy=True)

    old_error = 0.0

    for i in range(max_iterations):
        tree = KDTree(dst) #set up destination as KD-Tree
        distances, indices = tree.query(src)
        R, t = point_to_point_transform(src, dst[indices]) #look up the KD-Tree to see each point in src's closest neighbor 
        src = (R @ src.T).T + t #updates src to be closer to dst
        mean_error = np.mean(distances)
        if abs(old_error - mean_error) < tolerance: #if our new estimate is good enough just stop
            print("Took this many iterations: ", i)
            break
        old_error = mean_error

    R_final, t_final = point_to_point_transform(old_scan, src) #get transform between our best estimate and initial
    return R_final, t_final, src

def run_ICP():
    old_scan, _ = get_scan() #init 2 cycle delay
    new_scan, controls = get_scan()
    while(1):
        #closest new is the best we could match the data --> figure out if we can accumulate error and increase uncertainity
        rot_matrix = pre_processing(controls, time_step)
        # predicted_scan = (rot_matrix @ old_scan.T).T
        predicted_scan = old_scan
        rotation, translation, closest_new = icp(predicted_scan, new_scan) #figure out tolerance param
        # rotation = rotation @ rot_matrix
        publish_transform([rotation, translation])
        old_scan = new_scan #closest_new  --> will help with scan continuity but idk we want this (we dont want it)
        new_scan, controls = get_scan()

if __name__ == "__main__":
    run_ICP()