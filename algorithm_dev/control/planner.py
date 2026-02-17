""" 
GOAL: path planning & scheduling 

Prioritization by: sort by score, respect laser cooldown period, predict aim point at fire time

"""

#planner.py

from object_scoring import score_track, predict_position, PREDICT_HORIZON, MAX_COV_THRESHOLD, SPOT_RADIUS_PX_SAFE, FRAME_DT
import numpy as np


# === DECLARE CONSTANTS ===
LASER_COOLDOWN_FRAMES = int(0.25/ FRAME_DT) # minimum firing time / FPS 

# === DEFINE FUNCTIONS ===

def plan_targets(tracks, track_states, laser_origin, frame_idx): 
    """ 
    Generate planned shots, including redundancy within spot radius 
    """

    scored = []
    for t in tracks: 
        state = track_states.get(t.id, "unknown")
        score = score_track(t, state, laser_origin)
        if score > 0: 
            scored.append((score, t, state))

    # --- sort: hovering first, then cruising, then by score ---
    state_priority = {"hovering": 2, "cruising": 1, "accelerating":0}
    scored.sort(key=lambda x: (state_priority.get(x[2],0), x[0]), reverse=True)

    plan = []
    fire_time = frame_idx

    for _, track, _ in scored: 
        aim = predict_position(track, k=PREDICT_HORIZON)

        # add redundant points if high uncertainty 
        cov_trace = np.trace(track.kf.errorCovPost)
        redundancy = 1

        if cov_trace > MAX_COV_THRESHOLD * 0.5:
            redundancy = 3 # 3 points for uncertain track

        for r in range(redundancy): 
            # small random jitter within spot radius
            jitter = np.random.uniform(-SPOT_RADIUS_PX_SAFE/2, SPOT_RADIUS_PX_SAFE/2, size=2)
            plan.append({ # ensure redundancy actually happens
                "track_id": track.id,
                "aim": aim + jitter, 
                "fire_frame": fire_time
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
