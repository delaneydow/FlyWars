# control_interface.py
from planner import plan_targets
import numpy as np
from laser_interface import LaserInterface
from mirror_planner import MirrorPlanner
from object_scoring import SPOT_RADIUS_PX_SAFE

# classes
laser = LaserInterface()
mirror = MirrorPlanner(map_file="mirror_map1.npz", spot_radius_px=SPOT_RADIUS_PX_SAFE)  

# constants
LASER_ORIGIN = np.array([512, 384])  # example #TODO figure out if this is correct or not (measure in reality)


def control_step(tracks, track_states, frame_idx):

    # plan targets
    plan = plan_targets(tracks, track_states, LASER_ORIGIN, frame_idx) #TODO i think this is actually mirror, pass where mirror moved to last time it fired? 

    if not plan:
        return #do nothing for frame
    
    # attach mirror commands
    for cmd in plan: 
        x, y = cmd["aim"]
        # mirror handles clipping internally 
        u, v = mirror.find_uv_for_xy(x, y) 
        cmd["u"] = u
        cmd["v"] = v

    # fire laser on highest-priority ranked target

    #if uncertainty > threshold: #TODO establish these params
    #    return 

    # fire only the first planned shot per frame
    cmd = plan[0] #TODO see if this is right/need this
    MirrorPlanner.send_uv(cmd["u"], cmd["v"]) 
    laser.fire(cmd["aim"]) #TODO : can also fire using mirror UV mayve
