#Algorithm outline and testing
""" GOAL: KALMAN, ASSOCIATION, TRACK LIFECYCLE"""

# === PART 1: DETECT & LOCALIZE FLIES PER FRAME ===
""" goal is to record latency of frame substraction, ensure that program can keep track of multiple flies in each frame """

#imports
import numpy as np
from laser_interface import MIN_FIRE_TIME

# tunable constants

ENGAGE_RADIUS = 120       # px from laser center
MIN_SPEED = 2.0           # px/frame
MAX_COV_THRESHOLD = 15 #TODO see if i need to tune this value
SPOT_RADIUS_MM = 1.7 # laser spot radius in mm #TODO get actual estimate 
# empirical estimates
MM_PER_PX = 0.533 # mm per pixel
SPOT_RADIUS_PX = SPOT_RADIUS_MM / MM_PER_PX
CALIBRATION_ERROR_FACTOR = 1.05 # 5% error margin
SPOT_RADIUS_PX_SAFE = SPOT_RADIUS_MM * CALIBRATION_ERROR_FACTOR

# latency detection (in ms)
camera_latency = 1# exposure + sensor readout + gigE transfer + driver buffering
detection_latency = 1 #measured to average ~24.6 ms in testing
planning_latency = 1 # TODO need to measure
mirror_latency = 1 # step response + settling time + position feedback loop 
laser_trigger_latency = 1 #modulation delay + thermal dwell constraint

# predict horizon tied to latency 
FRAME_DT = 1/120.0 # seconds per frame
SYSTEM_LATENCY = 0.075 + MIN_FIRE_TIME# (listed in seconds) #TODO tweak this value!!! 
PREDICT_HORIZON =  int(SYSTEM_LATENCY/ FRAME_DT) #8 #int(SYSTEM_LATENCY/FRAME_DT) # num. of frames TODO change/refine eventually
UNCERTAINTY_PENALTY = 0.5 

# function definitions

def predict_position(track, k=PREDICT_HORIZON): 
    """
    Predict future (x,y) after k frames using given Kalman velocity
    """
    
    x,y = track.kf.statePost[0,0], track.kf.statePost[1,0]
    vx, vy = track.kf.statePost[2,0], track.kf.statePost[3,0]
    speed = np.hypot(vx, vy)

    # Adaptive horizon scaling -- longer horizon for faster objects 
    adaptive_k = k * min(1.5, max(0.5, speed / 5.0))

    # acceleration estimate
        # store previous velocity in track object
        # fall back to 0 if prev_vx / prev_v not present
    ax = (getattr(track, "prev_vx", vx) - vx) / FRAME_DT
    ay = (getattr(track, "prev_vy", vy) - vy) / FRAME_DT

    # update prev_vx, prev_vy for next frame
    track.prev_vx = vx
    track.prev_vy = vy

    # linear * acceleration
    x_pred = x + vx * adaptive_k + 0.5 * ax * adaptive_k**2
    y_pred = y + vy * adaptive_k + 0.5 * ay * adaptive_k**2

    return np.array([x_pred, y_pred])

def classify_motion(speed):
    if speed < 1:
        return "hovering"
    elif speed < 5:
        return "cruising"
    else:
        return "accelerating"


def score_track(track, state, beam_position): 
    """ 
    Returns priority score, higher = more urgent
    """

    # account for position + velocity
    vx, vy = track.kf.statePost[2:, 0]
    speed=np.hypot(vx, vy)
    
    # uncertainty (trace of covariance)
    cov = track.kf.errorCovPost
    uncertainty = cov[0,0] + cov[1,1]

    #filter out high-uncertaint tracks (skip scoring)
    if uncertainty > MAX_COV_THRESHOLD: 
        return 0.0

    # predict position #TODO I DON'T THINK THIS IS RIGHT
    prediction = predict_position(track)
    distance = np.linalg.norm(prediction - beam_position)

    # treat "within spot" as fully engaged
    if distance <= SPOT_RADIUS_PX_SAFE:
        distance_weight = 1.0 # no penalty
    else: 
        distance_weight = SPOT_RADIUS_PX_SAFE / distance # decays smoothly 

    # if distance is outside of engage radius i.e. not within range 
    if distance > ENGAGE_RADIUS: 
        return 0.0

    # continous weighting
    state_weight = np.clip(speed / 10.0, 0.2, 1.0)
    # motion state weighting
    #state_weight = {
     #   "flying": 1.0, 
      #  "hovering": 0.5, 
       # "accelerating": 0.1,
    #}.get(state, 0.3)

    score = (state_weight * speed / (1.0 + distance * 0.05) * np.exp(-UNCERTAINTY_PENALTY * uncertainty))
    score = score * distance_weight
    
    return float(score)

