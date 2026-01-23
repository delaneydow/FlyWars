#state.py

#GOAL: BEHAVIOR CLASSIFICATION

import numpy as np

# define functions
def classify_state(track): 
   
    speed = track.speed() # gets speed from centroids

    # TODO MAY NEED TO ADJUST THRESHOLDS IF NECESSARY (velocity given in pixels / second)
    if speed < 10:
        return "hovering"
    elif speed < 60: 
        return "cruising"
    else: 
        return "accelerating" 