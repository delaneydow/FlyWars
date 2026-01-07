#state.py

#GOAL: BEHAVIOR CLASSIFICATION

import numpy as np

# define functions
def classify_state(track): 
    vx = track.kf.statePost[2,0]
    vy = track.kf.statePost[3,0]
    speed = np.hypot(vx,vy)

    # TODO MAY NEED TO ADJUST THRESHOLDS IF NECESSARY
    if speed < 0.5:
        return "hovering"
    elif speed < 3: 
        return "cruising"
    else: 
        return "accelerating" 