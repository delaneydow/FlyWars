import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from planner import plan_targets
from object_scoring import predict_position, classify_motion, PREDICT_HORIZON, MAX_COV_THRESHOLD, SPOT_RADIUS_PX_SAFE
from control_interface import control_step
from cooldown import *

# import Tracking pipeline
from algorithm_dev.vision.track import Track #import Track class
from algorithm_dev.vision.tracking import process_video

# === latency settings ===
FRAME_DT = 1/120.0
TOTAL_LATENCY = 0.075 # seconds (initial estimate)
PREDICT_FRAMES = int(TOTAL_LATENCY/FRAME_DT)
HORIZON_RANGE = range(4,13) # frames to sweep 



# main control/ pipeline
def main(): 
    # capture tracking output
    print("[INFO] Starting tracking stage...")
    tracks, track_states, tracks_per_frame, track_states_per_frame = process_video()


    print("[INFO] Starting control stage...")
    pipeline_log = []

    for frame_idx, (track_objs, states) in enumerate(zip(tracks_per_frame, track_states_per_frame)): # tracks per frame is a list of Track lists per frame

        # build track_states dict 
        #track_states = {t.id: classify_motion(np.hypot(t.kf.statePost[2,0], t.kf.statePost[3,0])) for t in tracks}

        # run control
        start = time.perf_counter()
        control_step(track_objs, states, frame_idx)
        t_cntrl = time.perf_counter()-start  *1000 #ms
        t_combined =(time.perf_counter() - start) * 1000  # ms

        #thermal monitoring
        cpu_temp = get_cpu_temp()
        cooldown = adaptive_cooldown(cpu_temp)

        # TODO figure out if I should send cooldown to planning or not? 
        if cooldown > 0: 
            time.sleep(cooldown)

        total_pipeline = (time.perf_counter - start) * 1000 #ms
        fps = 1.0 / total_pipeline if total_pipeline > 0 else 0 


        pipeline_log.append({
            "frame": frame_idx,
            "t_control_ms": t_cntrl, 
            "t_combined": t_combined, 
            "total_pipeline": total_pipeline,
            "fps": fps, 
            "cpu_temp_C": cpu_temp,
            "cooldown_s": cooldown
        })

        print(
            f"[Frame {frame_idx}] "
            f"control={t_cntrl:.2f}ms | "
            f"pipeline={total_pipeline:.2f}ms | "
            f"temp={cpu_temp:.1f}C"
        )

    # data logging 
    df_control = pd.DataFrame(pipeline_log)
    df_control.to_csv("pipeline_log.csv", index=False)
    print(f"[INFO] Saved {len(df_control)} frames pipeline latency log")

if __name__=="__main__": 
    main()
