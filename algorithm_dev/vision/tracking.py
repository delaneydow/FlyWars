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
import cProfile
import pstats
from camera_interface import Camera
from track import Track
from state import classify_state
from tracking_helper import *


#constants
VIDEO_PATH = "LUCID_TRI032S-M_251700592__20251114112116359_video3.avi" #path to AVI file
FRAME_CAP = 100 #only saves a select group of frames for quick validation
REALTIME = True # disable visualization for real-time mode
MAX_FRAMES = 1000 # save csv after 1000 frames for all experimental trials

#visualization frames
DRAW_EVERY = 3

# === MAIN PROCESSING LOOP ===

def process_video(): 
    print(f"beginning...\n") 

    
    # instantiate empty arrays for tracking and latency calculations
    next_track_id = 0 
    #explicit multi-target capacity metrics
    tracks, log_frames, track_log, latencies, det_counts, track_counts, frames, thresh_frames = [], [], [], [], [], [], [], []
    # kalman state factoring   
    track_states = {}
    frame_idx = 0
    
    # instance of camera class 
    camera = Camera()
    # grab first frame to initialize prev_gray
    frame = camera.get_frame()
    prev_gray, roi_offset = preprocess_and_crop(frame)
    
    
    #begin loop, occurs over duration of video
    try: 

        while frame_idx < MAX_FRAMES: # stop stream after max frames, exit gracefuly 

            frame = camera.get_frame() # start stream
            start_frame = time.perf_counter() #records detection speed
    
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
                detections, tracks, next_track_id
            )
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
        
            # Visualization - only do for videos
            if not REALTIME: 
      
                start = time.perf_counter()
                visual = cv2.cvtColor(curr_gray, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(visual, (0,0), (visual.shape[1]-1, visual.shape[0]-1), (255,255,0), 1)
                for t in tracks:
                    if len(t.centroids) >= 2: 
                        cv2.line(visual, t.centroids[-2], t.centroids[-1], (0, 255,0), 1)
                    #for i in range(1, len(t.centroids)):
                        # repeat n times
                        #cv2.line(visual, t.centroids[i-1], t.centroids[i], (0,255,0), 1)
                    # do NOT repeat n times (moved outside loop)
                    cv2.circle(visual, t.last_position, 3, (0,0,255), -1)
                    cv2.putText(visual, f"ID {t.id}", (t.last_position[0]+5, t.last_position[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
                    # ring buffer 
                    if frame_idx % 10 == 0:
                        frames.append(visual.copy())
                        if len(frames) > FRAME_CAP:
                            frames.pop(0)
                t_vis = time.perf_counter() - start
    
            # Metrics
            total_frame = (time.perf_counter() - start_frame) * 1000.0
            print(f"Frame {frame_idx}: preprocess={t_preprocess*1000:.1f}ms, "
                  f"move={t_move*1000:.1f}ms, "
                  f"tracking={t_tracking*1000:.1f}ms,"
                  f"total={total_frame:.1f}ms")

            latencies.append(total_frame)
            det_counts.append(len(detections))
            track_counts.append(len(tracks))

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

       # save data frame
       df = pd.DataFrame(log_frames)
       df.to_csv("run_005_4.csv", index=False)

       print(f"[INFO] Saved {len(df)} frames to run_005_4.csv") 

       df_tracks=pd.DataFrame(track_log) # save individual tracking information
       df_tracks.to_csv("run_005_4_tracks.csv", index=False)

       print(f"[INFO] Saved {len(df_tracks)} track states")
    
    return latencies, det_counts, track_counts, frames, thresh_frames
        
latencies, det_counts, track_counts, frames, thresh_frames = process_video()
