#state.py

#GOAL: BEHAVIOR CLASSIFICATION

import numpy as np

from algorithm_dev.vision.state_defs import *


# define functions
def classify_state(track): 
   
    speed = track.speed() # gets speed from centroids

    # TODO MAY NEED TO ADJUST THRESHOLDS IF NECESSARY (velocity given in pixels / second)
    if speed < 5: 
        return STATE_HOVERING
    elif speed < 60: 
        return STATE_CRUISING
    else: 
        return STATE_ACCELERATING 
