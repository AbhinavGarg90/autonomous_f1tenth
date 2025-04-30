from lidar_polled import get_lidar_data
from ICP import ICPLocalizer
import rospy
import numpy as np
import time
from GridComp import OccupancyGridMapping
import matplotlib.pyplot as plt
import threading

# Shared data and locks
pose_data = {"pose": None, "raw": None}
pose_lock = threading.Lock()
shutdown_event = threading.Event()

# --- Occupancy Grid Thread ---
def occupancy_map_updater(occupancy_node, im):
    while not shutdown_event.is_set():
        with pose_lock:
            pose = pose_data["pose"]
            raw = pose_data["raw"]

        if pose is not None and raw is not None:
            occupancy_node.update_map(pose[0], pose[1], pose[2], raw)
            map = occupancy_node.get_probability_map()
            im.set_data(map)
            plt.pause(0.01)

        time.sleep(0.01)

# --- Main ---
def main():
    sim = False
    rospy.init_node("icp_runner")
    lidar_topic = '/car_1/scan' if sim else 'scan'

    lidar_data, raw_data = get_lidar_data(lidar_topic)

    icp = ICPLocalizer()
    icp.initialize(lidar_data)

    occupancy_node = OccupancyGridMapping()

    # Set up matplotlib once
    fig, ax = plt.subplots()
    grid = np.random.randint(0, 101, size=(100, 100))
    im = ax.imshow(grid, cmap='viridis', vmin=0, vmax=100, interpolation='none')
    ax.set_xticks(np.arange(-0.5, 200, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 200, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.1)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Start visualization thread
    viz_thread = threading.Thread(target=occupancy_map_updater, args=(occupancy_node, im))
    viz_thread.start()

    try:
        while not rospy.is_shutdown():
            lidar_data, raw_data = get_lidar_data(lidar_topic)
            est_pose = icp.update(lidar_data)

            with pose_lock:
                pose_data["pose"] = est_pose
                pose_data["raw"] = raw_data

            time.sleep(0.01)

    finally:
        shutdown_event.set()
        viz_thread.join()

if __name__ == "__main__":
    main()
