# control_interface.py
from planner import plan_targets, LASER_COOLDOWN_FRAMES
import numpy as np
import time
from laser_interface import LaserInterface
from mirror_planner import MirrorPlanner
from object_scoring import SPOT_RADIUS_PX_SAFE, PREDICT_HORIZON, predict_position

# classes
laser = LaserInterface()
mirror = MirrorPlanner(map_file="mirror_map1.npz", spot_radius_px=SPOT_RADIUS_PX_SAFE) 
mirror_settle_time = 0.25 # seconds, given rating of settling time + how long to switch directions (avg.) 

# constants
beam_position = np.array([512, 384])  # TODO figure out 0,0 origin, initialize once 

def control_step(tracks, track_states, frame_idx):
    global beam_position
    laser_fire = False

    for t in tracks: 

        fire_delay = LASER_COOLDOWN_FRAMES
        pred_xy, k_eff = predict_position(t, k=PREDICT_HORIZON + fire_delay)
        t.cached_prediction = pred_xy
        t.cached_k = k_eff

    # plan targets
    plan = plan_targets(tracks, track_states, beam_position, frame_idx) #TODO i think this is actually mirror, pass where mirror moved to last time it fired? 

    if not plan:
        return #do nothing for frame
    
    # attach mirror commands
    for cmd in plan: 
        x, y = cmd["aim"]
        # mirror handles clipping internally 
        u, v = mirror.find_uv_for_xy(x, y) 
        cmd["u"] = u
        cmd["v"] = v

    # fire laser on highest-priority ranked target, first planned shot per frame
    cmd = plan[0] #highest priority target #TODO see if this is right/need this
    mirror.send_uv(cmd["u"], cmd["v"]) 
    time.sleep(mirror_settle_time) #allow for settling time before firing laser #TODO add in uncertainty?? 
    laser.fire()
    laser_fire = True # change flag

    beam_position = cmd["aim"] #stores last position of mirror

    return {
        "fired": laser_fire, #whether flag is switched or not 
        "frame": frame_idx,
        "track_id": cmd["track_id"],
        "score": cmd["score"],
        "aim_x": float(cmd["aim"][0]),
        "aim_y": float(cmd["aim"][1]),
    }

