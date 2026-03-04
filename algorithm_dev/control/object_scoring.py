#Algorithm outline and testing
""" GOAL: KALMAN, ASSOCIATION, TRACK LIFECYCLE"""

# === PART 1: DETECT & LOCALIZE FLIES PER FRAME ===
""" goal is to record latency of frame substraction, ensure that program can keep track of multiple flies in each frame """

#imports
import numpy as np
from algorithm_dev.vision.state_defs import *
# tunable constants

ENGAGE_RADIUS = 120       # px from laser center
MIN_SPEED = 2.0           # px/frame
MAX_COV_THRESHOLD = 100 #TODO see if i need to tune this value
SPOT_RADIUS_MM = 1.7 # laser spot radius in mm #TODO get actual estimate 
# empirical estimates
MM_PER_PX = 0.533 # mm per pixel
SPOT_RADIUS_PX = SPOT_RADIUS_MM / MM_PER_PX
CALIBRATION_ERROR_FACTOR = 1.05 # 5% error margin
SPOT_RADIUS_PX_SAFE = SPOT_RADIUS_MM * CALIBRATION_ERROR_FACTOR
MIN_FIRE_TIME = 0.25 #seconds

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

DEBUG_SCORING = False  # set True only when debugging

if DEBUG_SCORING:
    print(f"  [SCORE DEBUG] ...")

# function definitions

def predict_position(track, k=PREDICT_HORIZON): 
    """
    Predict future (x,y) after k frames using given Kalman velocity
    """
    
    x = track.kf.statePost[0,0]
    y = track.kf.statePost[1,0]
    vx = track.kf.statePost[2,0]
    vy = track.kf.statePost[3,0]
    speed = np.hypot(vx, vy) #px/second

    # velocity damping for slow / hovering targets only 
    if speed < 10:
        vx *= 0.5
        vy *= 0.5

    # adaptive horizon scaling 
    if speed < 10: #nearly stationary
        adaptive_k = 1.0 #nearly current position
    elif speed < 50.0: #slow crawl 
        adaptive_k = k * 0.5
    elif speed < 150.0: #moderate flight
        adaptive_k = k * 0.8
    else: 
        adaptive_k = k * 1.0 #fast flight 
    #adaptive_k = k * min(1.5, max(0.5, speed / 5.0)) #TODO figure out how to balance this 

    # acceleration estimate
        # store previous velocity in track object
        # fall back to 0 if prev_vx / prev_v not present
        # eqn: a = vt - vt-1 / dt
    ax = (vx - getattr(track, "prev_vx", vx)) / FRAME_DT
    ay = (vy - getattr(track, "prev_vy", vy)) / FRAME_DT

    # update prev_vx, prev_vy for next frame
    track.prev_vx = vx
    track.prev_vy = vy

    # linear * acceleration
    x_pred = x + vx * adaptive_k + 0.5 * ax * adaptive_k**2
    y_pred = y + vy * adaptive_k + 0.5 * ay * adaptive_k**2

    return np.array([x_pred, y_pred]), int(adaptive_k)


def score_track(track, state, beam_position): 
    """ 
    Returns priority score, higher = better target
    """

    #TODO look into distance scoring + improve further
    # priority = interceptability x stability x engagement payoff 

    # === predict position ===
    prediction = getattr(track, "cached_prediction", None)

    if prediction is None: 
        return 0.0
    
    prediction = np.asarray(prediction)

    # === uncertainty filter === 
    
    cov = track.kf.errorCovPost
    uncertainty = cov[0,0] + cov[1,1]
    if uncertainty > MAX_COV_THRESHOLD:
        return 0.0

    #filter on settled tracks (>5 frames)
    #if track.last_seen > 5 and uncertainty > MAX_COV_THRESHOLD: 
     #       return 0.0 
    
    stability = 1.0/ (1.0+float(uncertainty)) #np.exp(-UNCERTAINTY_PENALTY * uncertainty)

    # === mirror travel cost === 
    mirror_delta = float(np.linalg.norm(prediction - np.asarray(beam_position)))
    mirror_cost = 1.0 / (1.0+ mirror_delta) #TODO figure out if this is necessary 

    # === motion state weighting (hovering first) === 
    vx = float(track.kf.statePost[2,0])
    vy = float(track.kf.statePost[3,0]) #first time x, second term y
    speed = float(np.hypot(vx, vy)) 
    #speed = track.speed()

    state_weight = {
        STATE_HOVERING: 1.3, 
        STATE_CRUISING: 0.9, 
        STATE_ACCELERATING: 0.5,
    }.get(state, 0.5) 

    # speed penality 
    speed_penalty = 1.0 / (1.0 + speed)
  
    # === final score ==

    score = (
        mirror_cost * state_weight * stability * speed_penalty # * commitment * cluster_bonus
    )
    if DEBUG_SCORING: 
        print(f"  [SCORE DEBUG] mirror_cost={mirror_cost:.4f} stability={stability:.4f} state_weight={state_weight} speed_penalty={speed_penalty:.4f}")
    
    return float(score)

