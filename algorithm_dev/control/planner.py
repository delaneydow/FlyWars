""" 
GOAL: path planning & scheduling 

Prioritization by: sort by score, respect laser cooldown period, predict aim point at fire time

"""

#planner.py

from object_scoring import score_track, predict_position
from mirror_planner import MirrorPlanner

# === DECLARE CONSTANTS ===
LASER_COOLDOWN_FRAMES = 2

# === DEFINE FUNCTIONS ===

def plan_targets(tracks, track_states, laser_origin, frame_idx): 
    """ 
    Returns ordered list of fire commands 
    """

    scored = []
    for t in tracks: 
        state = track_states.get(t.id, "unknown")
        score = score_track(t, state, laser_origin)
        if score > 0: 
            scored.append((score, t))

    scored.sort(reverse=True, key=lambda x: x[0])

    plan = []
    fire_time = frame_idx

    for _, track in scored: 
        aim = predict_position(track)
        plan.append({
            "track_id": track.id,
            "aim": aim, 
            "fire_frame": fire_time
        })
        fire_time += LASER_COOLDOWN_FRAMES
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
