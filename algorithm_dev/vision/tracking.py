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
from collections import deque
from algorithm_dev.vision.camera_interface import Camera
from algorithm_dev.vision.track import Track
from algorithm_dev.vision.state import classify_state
from algorithm_dev.vision.tracking_helper import *


#constants
MAX_FRAMES = 1000 # save csv after 1000 frames for all experimental trials

# === MAIN PROCESSING LOOP ===

def process_video(): 
    print(f"beginning...\n") 

    
    # instantiate empty arrays for tracking and latency calculations
    next_track_id = 0 
    #explicit multi-target capacity metrics
    tracks, latencies, det_counts, track_counts= [], [], [], []
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

        while True: # stop stream after max frames, exit gracefuly 

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
                detections, tracks, next_track_id, 
            )
            if frame_idx % 3 == 0: # run less often (i.e. less expensive)
                tracks = deduplicate_tracks(tracks) # one track per fly, protect stationary leak
            t_tracking = time.perf_counter() - start

            now = time.time()
            #kalman calculations (tracking and state update)
            for t in tracks: #TODO see if we should move this to be before association
                t.predict(dt)
                x, y, vx, vy = t.kf.statePost[:,0] #consolidate state calls
                cov = t.kf.errorCovPost
                #vx = t.kf.statePost[2, 0]
                #vy = t.kf.statePost[3, 0]
                speed = t.speed()
                
                #if frame_idx %3 == 0: # only track every 3 frames to gauge velocity better
                track_states[t.id] = classify_state(t) # just look every frame, trivial cost

                total_frame = (time.perf_counter() - start_frame) #in seconds
                t_total_ms = total_frame * 1000

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                print("[INFO] ESC pressed — stopping")
                break

            yield {
                "frame": frame_idx,
                "tracks": tracks, 
                "states": track_states,
                "vision_latency_ms": t_total_ms,
                "detections": len(detections)
            }

            prev_gray = curr_gray
            frame_idx += 1
    finally: 
       camera.close()

  

      
    
