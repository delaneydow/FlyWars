""" 
GOAL: path planning & scheduling 

Prioritization by: sort by score, respect laser cooldown period, predict aim point at fire time

"""

#planner.py


import numpy as np
from algorithm_dev.control.object_scoring import(
        score_track, MAX_COV_THRESHOLD, SPOT_RADIUS_PX_SAFE, FRAME_DT)
from algorithm_dev.vision.state_defs import *


# === DECLARE CONSTANTS ===
LASER_COOLDOWN_FRAMES = int(0.25/ FRAME_DT) # minimum firing time / FPS 

SCORE_DEBUG = False

# === DEFINE FUNCTIONS ===

def plan_targets(tracks, track_states, beam_position, frame_idx): 
    """ 
    Generate planned shots, including redundancy within spot radius 
    """

    scored = []
    for t in tracks: 
        state = track_states.get(t.id, STATE_UNKNOWN)
        # debug - TODO remove once firing works
        pred = getattr(t, "cached_prediction", None)
        cov = t.kf.errorCovPost
        uncertainty = cov[0,0] + cov[1,1]
        if SCORE_DEBUG:
            print(f" [SCORE DEBUG] track {t.id} state={state} pred={pred} uncertainty={uncertainty:.2f}")
        score = score_track(t, state, beam_position)
        if SCORE_DEBUG:
            print(f" [SCORE DEBUG] track {t.id} score={score:.4f}")
        if score > 0: 
            scored.append({
                "score": score,
                "track": t,
                "state": state
            })
    if not scored: 
        return []

    # --- sort: hovering first, then cruising, then by score ---
    state_priority = {STATE_HOVERING: 2,
                      STATE_CRUISING: 1,
                      STATE_ACCELERATING: 0,
                      }
    scored.sort(key=lambda x: (state_priority.get(x["state"], 0), x["score"]),reverse=True)

    plan = []
    fire_time = frame_idx

    for item in scored: #TODO are vars correct here not that track is updated
        track = item["track"]
        aim = getattr(track, "cached_prediction", None)

        if aim is None:
            continue #skip track if no prediction cached yet

        # add redundant points if high uncertainty 
        cov_trace = np.trace(track.kf.errorCovPost)

        redundancy = 3 if cov_trace > MAX_COV_THRESHOLD * 0.8 else 1 #change from 3 points

        for _ in range(redundancy): 
            # small random jitter within spot radius
            jitter = np.random.uniform(-SPOT_RADIUS_PX_SAFE/2, SPOT_RADIUS_PX_SAFE/2, size=2)
            plan.append({ # ensure redundancy actually happens
                "track_id": track.id,
                "aim": aim + jitter, 
                "fire_frame": fire_time,
                "score": item["score"],
                "state": item["state"]
            })
        fire_time += LASER_COOLDOWN_FRAMES * redundancy

    return plan


def attach_mirror_commands(plan, mirror_planner): 
    # post-processing stage
    enriched = []
    
    for cmd in plan: 
        x, y = cmd["aim"]
        u, v = mirror_planner.find_uv_for_xy(x,y)

        enriched.append({
            **cmd, 
            "u": float(u), 
            "v": float(v)
        })
    return enriched
