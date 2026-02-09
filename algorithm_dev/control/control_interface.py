# control_interface.py
from planner import plan_targets
import numpy as np
from laser_interface import LaserInterface
from mirror_planner import MirrorPlanner

laser = LaserInterface()
mirror = MirrorPlanner(fx,fy, uv_mean, uv_std, uv_bounds, spot_radius_px) #TODO fix inputs 
LASER_ORIGIN = np.array([512, 384])  # example


def control_step(tracks, track_states, frame_idx):

    if not MirrorPlanner.is_reachable(x,y):
        return # skip shot because of clamp aiming

    # plan targets
    plan = plan_targets(tracks, track_states, LASER_ORIGIN, frame_idx)

    if not plan:
        return #do nothing for frame
    
    
    # attach mirror commands
    for cmd in plan: 
        x, y = cmd["aim"]
        u, v = mirror.find_uv_for_xy(x, y) 
        cmd["u"] = u
        cmd["v"] = v

    # fire laser on highest-priority ranked target

    if uncertainty > threshold: #TODO establish these params
        return 

    # fire only the first planned shot per frame
    cmd = plan[0]
    laser.fire(cmd["aim"]) #TODO : can also fire using mirror UV mayve
