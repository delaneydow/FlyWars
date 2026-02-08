# control_interface.py
from planner import plan_targets
import numpy as np
from laser_interface import LaserInterface

laser = LaserInterface()
LASER_ORIGIN = np.array([512, 384])  # example


def control_step(tracks, track_states, frame_idx):
    plan = plan_targets(tracks, track_states, LASER_ORIGIN, frame_idx)

    if not plan:
        return

    # fire only the first planned shot per frame
    cmd = plan[0]
    laser.fire(cmd["aim"])
