#tracking.py

#this is stripped back version of tracking.ipynb 

# === PART 1: DETECT & LOCALIZE FLIES PER FRAME ===
""" goal is to record latency of frame substraction, ensure that program can keep track of multiple flies in each frame """

#imports
import cv2
import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from collections import deque
from camera_interface import Camera
from track import Track
from state import classify_state
from tracking_helper import *


#constants
VIDEO_PATH = "LUCID_TRI032S-M_251700592__20251114112116359_video3.avi" #path to AVI file
FRAME_CAP = 100 #only saves a select group of frames for quick validation
REALTIME = True # disable visualization for real-time mode
MAX_FRAMES = 1000 # save csv after 1000 frames for all experimental trials

# === MAIN PROCESSING LOOP ===

def process_video(): 
    print(f"beginning...\n") 

    
    # instantiate empty arrays for tracking and latency calculations
    next_track_id = 0 
    #explicit multi-target capacity metrics
    tracks, log_frames, track_log, latency_log, track_debug, latencies, det_counts, track_counts= [], [], [], [], [], [], [], []
    tracks_per_frame = []
    # kalman state factoring   
    track_states = {}
    frame_idx = 0
    
    # instance of camera class 
    camera = Camera()
    # grab first frame to initialize prev_gray
    frame = camera.get_frame()
    prev_gray, roi_offset = preprocess_and_crop(frame)
    last_time = None
    
    #begin loop, occurs over duration of video
    try: 

        while frame_idx < MAX_FRAMES: # stop stream after max frames, exit gracefuly 

            frame = camera.get_frame() # start stream
            start_frame = time.perf_counter() #records detection speed

            now = time.time()
            # compute dt dynamically from timestamps
            if last_time is None: 
                dt = 1/60.0 # fallback first time
            else: 
                dt = now - last_time

            last_time = now
    
             # Preprocess & crop
            start = time.perf_counter()
            # don't need to crop frame anymore since FOV is about size from video (24 inches x 24 inch board)
            curr_gray, _ = preprocess_and_crop(frame)
            t_preprocess = time.perf_counter() - start
    
            # Moving detection
            start = time.perf_counter()
            detections, thresh = detect_moving_objects_fast(prev_gray, curr_gray)
            t_move = time.perf_counter() - start
    
            # tracking only (no merging) 
            start = time.perf_counter()
            tracks, next_track_id = associate_detections_to_tracking_fast(
                detections, tracks, next_track_id, dt
            )
            if frame_idx % 3 == 0: # run less often (i.e. less expensive)
                tracks = deduplicate_tracks(tracks) # one track per fly, protect stationary leak
            t_tracking = time.perf_counter() - start

            now = time.time()
            #kalman calculations (tracking and state update)
            for t in tracks: 
                x, y, vx, vy = t.kf.statePost[:,0] #consolidate state calls
                cov = t.kf.errorCovPost
                #vx = t.kf.statePost[2, 0]
                #vy = t.kf.statePost[3, 0]
                speed = t.speed()
                
                print(
                    f"Track {t.id}: "
                    f"vx={vx:.2f}, vy={vy:.2f}, speed={t.speed():.2f}"
                
                 )
                print(
                    f"Track {t.id} raw state:",
                    t.kf.statePost.ravel()
                )
                #if frame_idx %3 == 0: # only track every 3 frames to gauge velocity better
                track_states[t.id] = classify_state(t) # just look every frame, trivial cost

                track_log.append({
                    "frame": frame_idx,
                    "time": now, 
                    "track_id": t.id,
                    "x": float(x),
                    "y": float(y),
                    "vx": float(vx), 
                    "vy": float(vy),
                    "speed": float(speed),
                    "state": track_states.get(t.id, "unknown"),
                    "cov_xx": float(cov[0,0]),
                    "cov_yy": float(cov[1,1]),
                })

                track_debug.append({
                "frame_idx": frame_idx,
                "track_id": t.id,
                "detected": len(detections) is not None,
                "centroid_x": float(t.centroids[-1],[0]),
                "centroid_y": float(t.centroids[-1][1]),
                "kf_vx": float(t.kf.statePost[2,0]),
                "kf_vy": float(t.kf.statePost[3,0]),
                "speed": float(t.speed()),
                "missed": t.missed(),
                #pred_x, pred_y, measurement_dx, measurement_dy
                "dt": dt
            })
                
                 # latency testing
            latency_log.append({
                "frame": frame_idx,
                "t_preprocess_ms": t_preprocess*1000,
                "t_detection_ms": t_move*1000,
                "t_tracking_ms": t_tracking*1000,
                "t_total_ms": t_total_ms,
                "n detections": len(detections),
                "n tracks": len(tracks)
            })
        
    
            # Metrics
            total_frame = (time.perf_counter() - start_frame) #in seconds
            t_total_ms = total_frame * 1000
            print(f"Frame {frame_idx}: preprocess={t_preprocess*1000:.1f}ms, "
                  f"move={t_move*1000:.1f}ms, "
                  f"tracking={t_tracking*1000:.1f}ms,"
                  f"total={t_total_ms:.1f}ms")
        

            latencies.append(t_total_ms)
            det_counts.append(len(detections))
            track_counts.append(len(tracks))
            tracks_per_frame.append(list(tracks))

             # log results
            log_frames.append({
                "frame": frame_idx,
                "time": time.time(),
                "detections": len(detections),
                "tracks": len(tracks),
                "latency_ms": total_frame,
                "max_speed": max(
                    (t.speed() for t in tracks),default=0)
            })

           

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                print("[INFO] ESC pressed — stopping")
                break

            prev_gray = curr_gray
            frame_idx += 1
    finally: 
       camera.close()

       # === SAVE OFF DATA LOGGING ===

       # data frame index
       df = pd.DataFrame(log_frames)
       df.to_csv("run_005_6.csv", index=False)
       print(f"[INFO] Saved {len(df)} frames to run_005_6.csv") 

        # tracking log
       df_tracks=pd.DataFrame(track_log) # save individual tracking information
       df_tracks.to_csv("run_005_6_tracks.csv", index=False)
       print(f"[INFO] Saved {len(df_tracks)} track states")

        # tracking debug
       df_debug = pd.DataFrame(track_debug)
       df_debug.to_csv("run_005_08_debug.csv", index=False)
       print(f"[INFO] Saved {len(df_debug)} frames") 

        # latency logging
       df_latency = pd.DataFrame(latency_log)
       df_latency.to_csv("pipeline_latency_log.csv", index=False)
       print(f"[INFO] Saved {len(df_latency)} frames latency log")

    
    return tracks, latency_log, det_counts, track_counts, tracks_per_frame
    
