#Algorithm outline and testing
""" GOAL: KALMAN, ASSOCIATION, TRACK LIFECYCLE"""

# === PART 1: DETECT & LOCALIZE FLIES PER FRAME ===
""" goal is to record latency of frame substraction, ensure that program can keep track of multiple flies in each frame """

#imports
import numpy as np

# tunable constants

ENGAGE_RADIUS = 120       # px from laser center
MIN_SPEED = 2.0           # px/frame
PREDICT_HORIZON = 5       # frames
UNCERTAINTY_PENALTY = 0.5 

# function definitions

def predict_position(track, k=PREDICT_HORIZON): 
    """
    Predict future (x,y) after k frames using given Kalman velocity
    """

    x,y = track.kf.statePost[0,0], track.kf.statePost[1,0]
    vx, vy = track.kf.statePost[2,0], track.kf.statePost[3,0]
    return np.array([x + vx * k, y + vy * k])


def score_track(track, state, laser_origin): 
    """ 
    Returns priority score, higher = more urgent
    """

    # account for position + velocity
    vx, vy = track.kf.statePost[2:, 0]
    speed=np.hypot(vx, vy)

    # if object is not moving fast enough, don't prioritize 
    if speed < MIN_SPEED: #TODO figure out what best way to do this may be
        return 0.0

    # predict position
    prediction = predict_position(track)
    distance = np.linalg.norm(prediction - laser_origin)

    # if distance is outside of engage radius i.e. not within range 
    if dist > ENGAGE_RADIUS: 
        return 0.0


    # motion state weighting
    state_weight = {
        "flying": 1.0, 
        "hovering": 0.5, 
        "stationary": 0.1,
    }.get(state, 0.3)

    # uncertainty (trace of covariance)
    cov = track.kf.errorCovPost
    uncertainty = cov[0,0] + cov[1,1]

    score = (state_weight * speed / (1.0 + dist * 0.05) * np.exp(-UNCERTAINTY_PENALTY * uncertainty))
    
    return float(score)