import numpy as np
import math
import matplotlib.pyplot as plt

class LaserScanSim(object):
    """
    A minimal class to mimic sensor_msgs/LaserScan fields needed by occupancy grid code.
    We'll fill angle_min, angle_increment, ranges, etc.
    """
    def __init__(self, angle_min, angle_max, angle_increment, ranges):
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.angle_increment = angle_increment
        self.ranges = ranges

        self.range_min = 0.0
        self.range_max = 10.0  # or something large


class OccupancyGridMapping:
    def __init__(self, width, height, resolution, origin_x, origin_y):
        self.width = width 
        self.height = height
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y

        # Initialize the grid in log odds (0 means 50% probability)
        self.log_odds = np.zeros((height, width), dtype=np.float32)

        # Log odds update parameters:
        self.l_occ = math.log(0.7 / 0.3)   # ~ +0.847
        self.l_free = math.log(0.3 / 0.7)  # ~ -0.847
        self.l_min = -5.0 
        self.l_max =  5.0  

    def world_to_map(self, x, y):
        """ Convert world coordinates (x, y) into grid indices (row_i, col_j). """
        col_j = int((x - self.origin_x) / self.resolution)
        row_i = int((y - self.origin_y) / self.resolution)
        return row_i, col_j

    def bresenham2D(self, r0, c0, r1, c1):
        r0 = int(round(r0))
        c0 = int(round(c0))
        r1 = int(round(r1))
        c1 = int(round(c1))

        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc

        r, c = r0, c0
        cells = []

        while True:
            cells.append((r, c))
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

        return cells

    def update(self, robot_pose, scan):
        rx, ry, rtheta = robot_pose

        angle_min = scan.angle_min
        angle_increment = scan.angle_increment
        ranges = np.array(scan.ranges)
        num_beams = len(ranges)
        angles = angle_min + np.arange(num_beams) * angle_increment

        for i in range(num_beams):
            r = ranges[i]
            if np.isinf(r) or np.isnan(r):
                continue

            beam_angle = rtheta + angles[i]
            x_end = rx + r * math.cos(beam_angle)
            y_end = ry + r * math.sin(beam_angle)

            start_r, start_c = self.world_to_map(rx, ry)
            end_r, end_c = self.world_to_map(x_end, y_end)

            cells = self.bresenham2D(start_r, start_c, end_r, end_c)

            # Free along the beam
            for (fr, fc) in cells[:-1]:
                if 0 <= fr < self.height and 0 <= fc < self.width:
                    self.log_odds[fr, fc] += self.l_free
                    if self.log_odds[fr, fc] < self.l_min:
                        self.log_odds[fr, fc] = self.l_min

            # Occupied at the last cell
            if len(cells) > 0:
                hit_r, hit_c = cells[-1]
                if 0 <= hit_r < self.height and 0 <= hit_c < self.width:
                    self.log_odds[hit_r, hit_c] += self.l_occ
                    if self.log_odds[hit_r, hit_c] > self.l_max:
                        self.log_odds[hit_r, hit_c] = self.l_max

    def get_probability_map(self):
        prob = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))
        return (prob * 100).astype(np.int8)


def simulate_lidar_scan(rx, ry, rtheta, angle_min, angle_max, num_beams):
    """
    Create a synthetic LaserScanSim object for a simple corridor:
      - corridor walls at y=-2 and y=+2
      - corridor extends from x=0 to x=10
    The robot is at (rx, ry) facing rtheta (radians).
    We'll cast rays from angle_min..angle_max and find intersection with walls.
    """

    # For a 270 deg FOV: angle_min = -135 deg, angle_max= +135 deg
    # That is a range of 270 deg = 4.71239 rad. We use num_beams=1081 as requested.
    angles = np.linspace(angle_min, angle_max, num_beams, endpoint=True)

    ranges = []
    for ang in angles:
        # We'll transform each ray into world coordinates:
        # The ray has direction rtheta + ang in the world.
        ray_angle = rtheta + ang
        # We parametric check intersection with corridor walls.
        dist = find_intersection_with_corridor(rx, ry, ray_angle)
        ranges.append(dist)

    scan = LaserScanSim(angle_min=angle_min,
                        angle_max=angle_max,
                        angle_increment=(angle_max - angle_min)/(num_beams-1),
                        ranges=ranges)
    return scan


def find_intersection_with_corridor(rx, ry, ray_angle):
    """
    The corridor has:
      - left wall at y=+2
      - right wall at y=-2
      - corridor extends in x from 0 to 10
    We find the distance from (rx, ry) to the nearest intersection along `ray_angle`.
    If it doesn't hit anything (e.g. if behind the corridor?), we can cap at 15.0 or so.
    """
    # We'll convert the direction into dx, dy
    dx = math.cos(ray_angle)
    dy = math.sin(ray_angle)
    # param t such that point = (rx + t*dx, ry + t*dy)

    # We'll find intersections with y=2, y=-2, x=0, x=10 and take the minimum positive t.
    # 1) Intersection with y= 2 => 2 = ry + t*dy => t = (2-ry)/dy if dy != 0
    # 2) Intersection with y=-2 => -2= ry+ t*dy => t= (-2-ry)/dy
    # 3) Intersection with x= 0 => 0 = rx + t*dx => t= (-rx)/dx
    # 4) Intersection with x=10 => 10= rx+ t*dx => t= (10-rx)/dx
    # Then we pick the smallest positive t that is within the corridor region
    possible_ts = []

    # Intersection with top wall y=2
    if abs(dy) > 1e-9:
        t_top = (2.0 - ry)/dy
        if t_top > 0:
            # check if the x is within [0,10] at this t
            x_hit = rx + t_top*dx
            if 0 <= x_hit <= 10:
                possible_ts.append(t_top)

    # Intersection with bottom wall y=-2
    if abs(dy) > 1e-9:
        t_bot = (-2.0 - ry)/dy
        if t_bot > 0:
            x_hit = rx + t_bot*dx
            if 0 <= x_hit <= 10:
                possible_ts.append(t_bot)

    # Intersection with left boundary x=0
    if abs(dx) > 1e-9:
        t_left = (0.0 - rx)/dx
        if t_left > 0:
            y_hit = ry + t_left*dy
            if -2 <= y_hit <= 2:
                possible_ts.append(t_left)

    # Intersection with right boundary x=10
    if abs(dx) > 1e-9:
        t_right = (10.0 - rx)/dx
        if t_right > 0:
            y_hit = ry + t_right*dy
            if -2 <= y_hit <= 2:
                possible_ts.append(t_right)

    if len(possible_ts) == 0:
        # means no intersection in front => let's limit range
        return 15.0  # or something large

    return min(possible_ts)


# ----------------------------
# 3) Main test routine
# ----------------------------

def main():
    # 3a) Create occupancy grid instance
    # Let's use a 200 x 200 grid at 0.1 m resolution => covers 20m x 20m.
    # Put the origin so that x=-5,y=-5 => cell(0,0) => that we have 10m in +x and 5m in -x etc.
    width = 200
    height = 200
    resolution = 0.1
    origin_x = -5.0
    origin_y = -5.0

    ogm = OccupancyGridMapping(width, height, resolution, origin_x, origin_y)

    # 3b) We'll simulate a robot that starts at (x=1,y=0,theta=0) and moves forward in +x
    # corridor is from x=0 to x=10, y in [-2,2].
    rx = 1.0
    ry = 0.0
    rtheta = 0.0  # facing +x

    # For each step, we move e.g. 0.1 m in +x
    num_steps = 50
    step_size = 0.1

    # LIDAR parameters: 1081 beams, 270 deg => -135 deg to +135 deg
    # Convert degrees to radians
    angle_min_deg = -135.0
    angle_max_deg =  135.0
    angle_min = math.radians(angle_min_deg)
    angle_max = math.radians(angle_max_deg)
    num_beams = 1081

    for step in range(num_steps):
        # Generate a synthetic scan
        scan_msg = simulate_lidar_scan(rx, ry, rtheta, angle_min, angle_max, num_beams)

        # Update occupancy grid
        ogm.update((rx, ry, rtheta), scan_msg)

        # Move the robot forward in +x
        rx += step_size

        # If we exit corridor or reach x=9.5, let's stop
        if rx > 9.5:
            break

    # 3c) After finishing steps, let's visualize the final occupancy grid
    prob_map = ogm.get_probability_map()

    plt.figure(figsize=(6,6))
    # Because our row=0 is at top in typical array indexing, we'll flip vertically or use origin='lower'
    # We'll do origin='lower' so that row=0 is at bottom. 
    # (You might see an inverted y-axis if you prefer normal matrix indexing. It's just a debug.)
    plt.imshow(prob_map, origin='lower', cmap='gray', vmin=0, vmax=100)
    plt.title("Simulated Corridor Occupancy Grid")
    plt.colorbar(label='Occupancy [0..100]')
    plt.xlabel("Grid X (cells)")
    plt.ylabel("Grid Y (cells)")

    plt.show()

if __name__ == "__main__":
    main()
