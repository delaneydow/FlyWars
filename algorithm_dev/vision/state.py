#state.py

#GOAL: BEHAVIOR CLASSIFICATION

import numpy as np

# define functions
def classify_state(track): 
   
    speed = track.speed() # gets speed from centroids

    # TODO MAY NEED TO ADJUST THRESHOLDS IF NECESSARY
    if speed < 0.5:
        return "hovering"
    elif speed < 3: 
        return "cruising"
    else: 
        return "accelerating" 