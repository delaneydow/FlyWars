#tracking.py

#this is stripped back version of tracking.ipynb 

# === PART 1: DETECT & LOCALIZE FLIES PER FRAME ===
""" goal is to record latency of frame substraction, ensure that program can keep track of multiple flies in each frame """

#imports
import cv2
import time
from algorithm_dev.vision.state import classify_state
from algorithm_dev.vision.tracking_helper import *


#constants
#MAX_FRAMES = 1000 # save csv after 1000 frames for all experimental trials

# === MAIN PROCESSING LOOP ===

def process_video(camera, display=False): 
    print(f"beginning...\n") 

    
    # instantiate empty arrays for tracking and latency calculations
    next_track_id = 0 
    #explicit multi-target capacity metrics
    tracks = []
    # kalman state factoring   
    track_states = {}
    frame_idx = 0
    
    # instance of camera class (now control/main should handle this)
    #camera = Camera()
    # grab first frame to initialize prev_gray
    frame = None
    while frame is None:
        frame = camera.get_frame()

    prev_gray, _ = preprocess_and_crop(frame)
    last_time = None
    
    #begin loop, occurs over duration of video
    try: 

        while True: # stop stream after max frames, exit gracefuly 

            frame = camera.get_frame() # start stream
            # tolerate dropped frames
            if frame is None: 
                #print("[VISION] None frame")
                continue

            vision_start = time.perf_counter() #records detection speed

            now = time.time()
            # compute dt dynamically from timestamps
            if last_time is None: 
                dt = 1/60.0 # fallback first time
            else: 
                dt = now - last_time

            last_time = now
    
             # Preprocess & crop
            # don't need to crop frame anymore since FOV is about size from video (24 inches x 24 inch board)
            try: 
                curr_gray, _ = preprocess_and_crop(frame)
            except Exception as e: 
                #print(f"[VISION preprocess failed: {e}")
                import traceback; traceback.print_exc()
                continue
    
            # Moving detection
            try:

                detections, _ = detect_moving_objects_fast(prev_gray, curr_gray)
            except Exception as e: 
                #print(f"[VISION] detection failed: {e}")
                import traceback; traceback.print_exc()
                continue

    
            # tracking only (no merging) 
            try:
                tracks, next_track_id = associate_detections_to_tracking_fast(
                        detections, tracks, next_track_id,
                        )
                active_ids = {t.id for t in tracks}
                track_states = {k: v for k, v in track_states.items() if k in active_ids}
                if frame_idx % 3 == 0: # run less often (i.e. less expensive)
                    tracks = deduplicate_tracks(tracks) # one track per fly, protect stationary leak
            except Exception as e:
                #print(f"[VISION] tracking failed: {e}")
                import traceback; traceback.print_exc()
                continue

            now = time.time()
            #kalman calculations (tracking and state update)
            try:
                for t in tracks: #TODO see if we should move this to be before association
                    t.predict(dt)
                    track_states[t.id] = classify_state(t) # look every frame, trivial cost increase
            except Exception as e: 
                #print(f"[VISION] kalman failed: {e}")
                import traceback; traceback.print_exc()
                continue

            # compute total frame latency once
            vision_latency_ms = (time.perf_counter() - vision_start) * 1000

            if display: 
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    print("[INFO] ESC pressed — stopping")
                    break
            try:

                yield {
                    "frame": frame_idx,
                    "tracks": tracks, 
                    "states": track_states,
                    "vision_latency_ms": vision_latency_ms,
                    "detections": len(detections),
                    "timestamp": time.perf_counter()
                }
            except GeneratorExit:
                print("[VISION] GeneratorExit -- caller closed the generator")
                return
            except Exception as e: 
                print(f"[VISION] exception throwin INTO generator after yield: {e}")
                import traceback; traceback.print_exc()
                # don't break, keep looping

            prev_gray = curr_gray
            frame_idx += 1
    except Exception as e: 
        print(f"[VISION] outer loop crashed: {e}")
        import traceback; traceback.print_exc()
    finally: 
       print("[VISION] shutdown")

  

      
    
