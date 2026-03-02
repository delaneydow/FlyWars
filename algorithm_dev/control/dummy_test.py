import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from planner import plan_targets
from object_scoring import predict_position, classify_motion, PREDICT_HORIZON, MAX_COV_THRESHOLD, SPOT_RADIUS_PX_SAFE
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

        self.cached_prediction = None
        self.cached_k = None

# === LOAD CSV ===

df = pd.read_csv(
    r"C:\Users\dmdow\Documents\GitHub\FlyWars\algorithm_dev\run_006_1_tracks.csv"
)
#df = pd.read_csv(csv_path)
print("Starting replay")

# Pre-index CSV for quick future lookup
frame_lookup = {
    f: g.set_index("track_id")
    for f, g in df.groupby("frame")
}

prediction_errors, prediction_speeds, prediction_inside_spot = [], [], []

horizon_results = {k: [] for k in HORIZON_RANGE}

for frame_idx, frame_df in df.groupby("frame"):

    start = time.perf_counter()

    tracks = [DummyTrack(r) for _, r in frame_df.iterrows()]
    track_states = {
    int(r.track_id): classify_motion( #TODO should we just pull "state" from here instead or not really rely
        np.hypot(r.vx, r.vy)
    )
    for _, r in frame_df.iterrows()
}
    # ---- Existing control pipeline ----
    control_step(tracks, track_states, frame_idx)

    # ---- Prediction error analysis ----
    future_frame = frame_idx + int(round(PREDICT_HORIZON)) 

    for t in tracks: 
            pred_xy, k_eff = predict_position(t, k=PREDICT_HORIZON)
            

            future_frame = frame_idx + int(round(k_eff))

            if future_frame not in frame_lookup:
                continue

            future_df = frame_lookup[future_frame]

            if t.id not in future_df.index: 
                continue

            actual_row = future_df.loc[t.id]
            actual_xy = np.array([actual_row.x, actual_row.y])

            err = np.linalg.norm(pred_xy - actual_xy)
            vx, vy = t.kf.statePost[2:, 0]
            speed = np.hypot(vx, vy)
            prediction_speeds.append(speed)

             
            for k in HORIZON_RANGE: 

                future_frame = frame_idx + int(round(k))

                if future_frame not in frame_lookup:
                    continue
                if t.id not in frame_lookup[future_frame].index:
                    continue

                actual_xy = frame_lookup[future_frame].loc[t.id][["x", "y"]].values
                err1 = np.linalg.norm(pred_xy - actual_xy)
                horizon_results[k].append(err1)


            inside_spot = err <=SPOT_RADIUS_PX_SAFE #log whether actual was inside predicted spot
            prediction_errors.append(err)
            prediction_inside_spot.append(inside_spot)
    
    

    inside_rate=np.mean(prediction_inside_spot)    

    elapsed = (time.perf_counter() - start) * 1000
    print(f"Frame {frame_idx}: total pipeline {elapsed:.2f} ms")

# horizon summary
for k in HORIZON_RANGE:
                errors = np.array(horizon_results[k])

                if len(errors) == 0:
                    print(f"Horizon {k}: no samples")
                    continue

                print(
                    f"Horizon {k}: "
                    f"mean={errors.mean():.2f}, "
                    f"median={np.median(errors):.2f}, "
                    f"max={errors.max():.2f}"
                )


# error plotting 
plt.figure()
# plot extremes separately to better visualize distribution based on paired data (not independent)
pairs = [
    (s, e)
    for s, e in zip(prediction_speeds, prediction_errors)
    if e < 200
]

if pairs:
    speeds, errors = zip(*pairs)

    plt.figure()
    plt.scatter(speeds, errors, alpha=0.3)
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

print(f"Fraction of predictions hitting within spot radius: {inside_rate*100:.1f}%")
