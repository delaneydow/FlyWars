""" 
GOAL: path planning & scheduling 

Prioritization by: sort by score, respect laser cooldown period, predict aim point at fire time

"""

#planner.py

from object_scoring import score_track, predict_position

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
