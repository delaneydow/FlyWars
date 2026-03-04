#control_interface.py
from algorithm_dev.control.planner import plan_targets, LASER_COOLDOWN_FRAMES
import numpy as np
import time
from algorithm_dev.control.laser_interface import LaserInterface
from algorithm_dev.control.mirror_planner import MirrorPlanner
from algorithm_dev.control.object_scoring import SPOT_RADIUS_PX_SAFE, PREDICT_HORIZON, predict_position

mirror_settle_time = 0.025 # 25ms, given rating of settling time + how long to switch directions (avg.) 

# constants
beam_position = np.array([512, 384])  # TODO figure out 0,0 origin, initialize once 

DEBUG_CNTRL = False  # set True only when debugging
DEBUG_SCORE = False 

def control_step(tracks, track_states, frame_idx, laser, mirror):
    global beam_position

    if DEBUG_CNTRL:
        print(f"[CONTROL] {len(tracks)} tracks, {len(track_states)} states")

    if not tracks:
        if DEBUG_CNTRL:
            print("[CONTROL] no tracks, skipping")
        return None

    for t in tracks: 

        pred_xy, k_eff = predict_position(t, k=PREDICT_HORIZON + LASER_COOLDOWN_FRAMES)
        t.cached_prediction = pred_xy
        t.cached_k = k_eff
        if DEBUG_SCORE:
            print(f"  [SCORE DEBUG] track {t.id} pred={pred_xy} k_eff={k_eff}")

    # plan targets
    plan = plan_targets(tracks, track_states, beam_position, frame_idx) #TODO i think this is actually mirror, pass where mirror moved to last time it fired? 

    if not plan:
        if DEBUG_CNTRL:
            print("[CONTROL] planner returned empty, skipping")
        return None #do nothing for frame
   

    # fire laser on highest-priority ranked target, first planned shot per frame
    cmd = plan[0] #highest priority target 
    u, v = mirror.find_uv_for_xy(*cmd["aim"]) #compute u & v

    mirror.send_uv(u, v) # use local u, v
    time.sleep(mirror_settle_time) #allow for settling time before firing laser #TODO add in uncertainty?? 
    if DEBUG_CNTRL:
        print(f"[FIRE] track={cmd['track_id']} aim={cmd['aim']} uv=({cmd['u']:.3f},{cmd['v']:.3f})")
    laser.fire()
    if DEBUG_CNTRL:
        print(f"[FIRE] laser fired")
  

    beam_position = cmd["aim"] #stores last position of mirror

    return {
        "fired": True,
        "frame": frame_idx,
        "track_id": cmd["track_id"],
        "score": cmd["score"],
        "aim_x": float(cmd["aim"][0]),
        "aim_y": float(cmd["aim"][1]),
    }

