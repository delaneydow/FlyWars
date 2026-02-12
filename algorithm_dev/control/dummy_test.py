import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from planner import plan_targets
from object_scoring import predict_position, classify_motion, PREDICT_HORIZON, MAX_COV_THRESHOLD, SPOT_RADIUS_PX_SAFE
from control_interface import control_step
from algorithm_dev.vision.tracking import process_video

# === latency settings ===
FRAME_DT = 1/120.0
TOTAL_LATENCY = 0.075 # seconds (initial estimate)
PREDICT_FRAMES = int(TOTAL_LATENCY/FRAME_DT)
HORIZON_RANGE = range(4,13) # frames to sweep 

# capture tracking output
tracks, latency_log, det_counts, track_counts = process_video()

# wrap control_step timing and log it
control_latency_log, prediction_errors, prediction_speeds, prediction_inside_spot = [], [], [], []

horizon_results = {k: [] for k in HORIZON_RANGE}

for frame_idx, track_objs in enumerate(tracks_per_frame): # tracks per frame is a list of Track lists per frame
    start=time.perf_counter()

    # build track_states dict 

    track_states = {t.id: classify_motion(np.hypot(t.kf.statePost[2,0], t.kf.statePost[3,0])) for t in track_objs}

    # run control
    control_step(track_objs, track_states, frame_idx)

    t_control = (time.perf_counter() - start) * 1000  # ms
    control_latency_log.append({
        "frame": frame_idx,
        "t_control_ms": t_control
    })

    # data logging 
    df_control = pd.DataFrame(control_latency_log)
    df_control.to_csv("control_latency_log.csv", index=False)
    print(f"[INFO] Saved {len(df_control)} frames control latency log")

