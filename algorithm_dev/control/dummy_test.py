import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from planner import plan_targets
from object_scoring import predict_position, classify_motion, PREDICT_HORIZON, MAX_COV_THRESHOLD
from control_interface import control_step

# === latency settings ===
FRAME_DT = 1/120.0
TOTAL_LATENCY = 0.075 # seconds (initial estimate)
PREDICT_FRAMES = int(TOTAL_LATENCY/FRAME_DT)
HORIZON_RANGE = range(4,13) # frames to sweep 

# === Dummy track adapter ===
class DummyKF: 
    def __init__(self, x, y, vx, vy, cov_xx, cov_yy): 
        self.statePost = np.array([[float(x)], [float(y)], [float(vx)], [float(vy)]])
        self.errorCovPost = np.diag([cov_xx, cov_yy, 1, 1])


class DummyTrack: 
    def __init__(self, row): 
        self.id = int(row.track_id)
        self.kf = DummyKF(
            row.x, row.y, 
            row.vx, row.vy,
            row.cov_xx, row.cov_yy
        )

# === LOAD CSV ===

df = pd.read_csv(
    r"C:\Users\dmdow\Documents\GitHub\FlyWars\algorithm_dev\run_005_4_tracks.csv"
)
#df = pd.read_csv(csv_path)
print("Starting replay")

# Pre-index CSV for quick future lookup
frame_lookup = {
    f: g.set_index("track_id")
    for f, g in df.groupby("frame")
}

prediction_errors = []
prediction_speeds = []

horizon_results = {k: [] for k in HORIZON_RANGE}

for frame_idx, frame_df in df.groupby("frame"):

    start = time.perf_counter()

    tracks = [DummyTrack(r) for _, r in frame_df.iterrows()]
    track_states = {
    int(r.track_id): classify_motion(
        np.hypot(r.vx, r.vy)
    )
    for _, r in frame_df.iterrows()
}
    # ---- Existing control pipeline ----
    control_step(tracks, track_states, frame_idx)

    # ---- Prediction error analysis ----
    future_frame = frame_idx + PREDICT_HORIZON

    if future_frame in frame_lookup:

        future_df = frame_lookup[future_frame]

        for t in tracks:
            if t.id not in future_df.index:
                continue
            cov_trace = np.trace(t.kf.errorCovPost)
            if cov_trace > MAX_COV_THRESHOLD:
                continue # skip track

            #pred_xy = predict_position(t, k=PREDICT_HORIZON)

            actual_row = future_df.loc[t.id]
            actual_xy = np.array([actual_row.x, actual_row.y])

            vx, vy = t.kf.statePost[2:, 0]
            speed = np.hypot(vx, vy)
             
            for k in HORIZON_RANGE: 
                pred_xy = predict_position(t, k=k)
                err1 = np.linalg.norm(pred_xy - actual_xy)
                horizon_results[k].append(err1)

            #print("pred_xy:", pred_xy, type(pred_xy))
            #print("actual_xy:", actual_xy, type(actual_xy))

            # collect stats for k = PREDICT_HORIZON
            pred_xy = predict_position(t, k=PREDICT_HORIZON)
            err = np.linalg.norm(pred_xy - actual_xy)
            prediction_errors.append(err)
            prediction_speeds.append(speed)
            for k in HORIZON_RANGE:
                errors = np.array(horizon_results[k])
                print(f"Horizon {k}: mean={errors.mean():.2f}, median={np.median(errors):.2f}, max={errors.max():.2f}")


    elapsed = (time.perf_counter() - start) * 1000
    print(f"Frame {frame_idx}: total pipeline {elapsed:.2f} ms")


# error plotting 
plt.figure()
# plot extremes separately to better visualize distribution
plt.scatter([s for s,e in zip(prediction_speeds,prediction_errors) if e<200],
            [e for e in prediction_errors if e<200], alpha=0.3)
plt.xlabel("Speed (px/frame)")
plt.ylabel("Prediction error (px)")
plt.title("Prediction error vs speed")
plt.grid(True)
plt.show() 

# plot mean error per horizon from sweep
means = [np.mean(horizon_results[k]) for k in HORIZON_RANGE]
plt.figure()
plt.plot(list(HORIZON_RANGE), means, marker='o')
plt.xlabel("Prediction Horizon (frames)")
plt.ylabel("Mean Prediction Error (px)")
plt.title("Mean Error vs Horizon")
plt.show()

# summary stats
if prediction_errors:
    print("\nPrediction Error Summary:")
    print("Mean error:", np.mean(prediction_errors))
    print("Median error:", np.median(prediction_errors))
    print("Max error:", np.max(prediction_errors))
