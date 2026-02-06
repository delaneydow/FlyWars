#tracking.py

#this is stripped back version of tracking.ipynb 

# === PART 1: DETECT & LOCALIZE FLIES PER FRAME ===
""" goal is to record latency of frame substraction, ensure that program can keep track of multiple flies in each frame """

#imports
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from collections import deque
import cProfile
import pstats
from camera_interface import Camera
from state import classify_state

#constants
VIDEO_PATH = "LUCID_TRI032S-M_251700592__20251114112116359_video3.avi" #path to AVI file
DOWNSCALE = 1.0 #<1.0 to downscale frames for speed
THRESH_VAL = 25 # frame subtraction threshold
MIN_AREA = 5 # minimum blob area (pixels)
MAX_AREA = 300 # max blob area (pixels)
MAX_MISSED = 5 # allows tracks to survive 5 frames without a detection
MAX_TRACK_DIST = 50 # max distance for track association (pixels)
HISTORY = 50 # frame to keep for plotting
MAX_TRACKS = 150 # TODO tune this
FRAME_CAP = 100 #only saves a select group of frames for quick validation
REALTIME = True # disable visualization for real-time mode
# ROI Configuration
# define ROI as fractions of frame dimensions
ROI_X_MIN = 0.5
ROI_Y_MIN = 0.0

# Precomputed morphology kernels (create once)
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# Precompute a small circle mask for radius 6
CIRCLE_RADIUS = 20
circle_mask = np.zeros((2*CIRCLE_RADIUS+1, 2*CIRCLE_RADIUS+1), dtype=np.uint8)
cv2.circle(circle_mask, (CIRCLE_RADIUS, CIRCLE_RADIUS), CIRCLE_RADIUS, 255, -1)

#visualization frames
DRAW_EVERY = 3

#kalman constants
FPS = 60.0 # or read from video metadata
DT = 1.0 / FPS


# === TRACKING CLASS ===
class Track: 
    def __init__(self, track_id, centroid): 
        self.id = track_id
        self.centroids = deque(maxlen=HISTORY)
        self.centroids.append(centroid)
        #self.last_seen = 0 # frame idx when last seen
        self.missed = 0 # num of consecutive frames w/o detection

        # ADD KALMAN STATE
        # using initial vector [x, y, vx, vy]T
        self.kf = cv2.KalmanFilter(4,2)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, DT, 0],
            [0, 1, 0, DT], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]], dtype=np.float32)

        self.kf.measurementMatrix=np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) *1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

        self.kf.statePre = np.array([[centroid[0]],
                                     [centroid[1]],
                                     [0],
                                     [0]], dtype=np.float32)

    def predict(self): 
        pred = self.kf.predict()
        return pred[0,0], pred[1,0]

    def update(self, detection=None):
        if detection is not None: 
            measured = np.array([[np.float32(detection[0])],
                                 [np.float32(detection[1])]])
            self.kf.correct(measured)
            
            x = int(self.kf.statePost[0,0])
            y = int(self.kf.statePost[1,0])
            self.centroids.append((x,y))
            self.missed = 0
        else:
            self.centroids.append(self.centroids[-1])
            self.missed += 1

    @property
    def last_position(self): 
        return self.centroids[-1] # returns last position stored in centroid

# === HELPER FUNCTIONS ===
def preprocess_and_crop(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape 

    # Define ROI fractions (example: right half)
    x0 = int(ROI_X_MIN * w)
    y0 = int(ROI_Y_MIN * h)
    x1 = w
    y1 = h

    roi = gray[y0:y1, x0:x1]

    # force consistent minimum size
    if roi.shape[0] < 8 or roi.shape[1] < 8:
        raise ValueError(f"ROI too small: {roi.shape} from frame {gray.shape}")

    return roi, (x0, y0)


def detect_moving_objects_fast(prev_gray, curr_gray):
    diff = cv2.absdiff(curr_gray, prev_gray)
    diff = cv2.GaussianBlur(diff, (5,5), 0)
    _, thresh = cv2.threshold(diff, THRESH_VAL, 255, cv2.THRESH_BINARY)
    # reuse kernel3, do NOT create a new one
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel3)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    detections = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if MIN_AREA <= area <= MAX_AREA:
            cx, cy = centroids[i]
            detections.append((int(cx), int(cy)))
    return detections, thresh


def associate_detections_to_tracking_fast(detections, tracks, next_id):
    
    # USING EARLY-EXIT GATED REPLACEMENT
    if not detections: 
        for t in tracks: 
            t.update(None)
        return [t for t in tracks if t.missed <= MAX_MISSED], next_id

    used = [False] * len(detections)
    max_dist_sq = MAX_TRACK_DIST * MAX_TRACK_DIST

    # match existing tracks using greedy algorithm 
    for t in tracks: 
        px, py =t.predict()
        best_idx = -1
        best_dist_sq = max_dist_sq

        #early-exit dist gating
        for i, (dx, dy) in enumerate(detections): 
            if used[i]: 
                continue
                
            ddx = dx-px 
            ddy = dy-py 
            dist_sq = ddx * ddx + ddy * ddy

            # early accept if very close 
            if dist_sq < best_dist_sq: 
                best_dist_sq = dist_sq
                best_idx = i 
                # optional and TODO TEST hard gate
                if dist_sq < 35: # ~5px
                    break
        if best_idx !=-1: 
            t.update(detections[best_idx])
            used[best_idx]=True
        else:
            t.update(None)

    # generate new tracks ONLY for unmatched detections
    for i, d in enumerate (detections): 
        #if not used[i]: 
        if len(tracks) < MAX_TRACKS: #caps # of tracks to stabilize runtime
            tracks.append(Track(next_id, d))
            next_id += 1

    # remove and prune old tracks
    tracks = [t for t in tracks if t.missed <=MAX_MISSED]

    return tracks, next_id


def deduplicate_tracks(tracks, radius=5): 
    # distance gating in association or deduplication only once every N frames
    # prevent same location collapse
    if len(tracks) <=1: 
        return tracks

    keep = []
    for t in tracks: 
        if not any(np.linalg.norm(
            np.array(t.last_position) - np.array(k.last_position)
        ) < radius for k in keep): 
            keep.append(t)
    return keep

# === MAIN PROCESSING LOOP ===

def process_video(): 
    print(f"beginning...\n") 
    # if reading from video file -- not applicable for live tracking
    #capture = cv2.VideoCapture(VIDEO_PATH)
   #assert capture.isOpened(), "Could not open video file"
    
    #ret, frame = capture.read()
    #if not ret: 
    #    raise RuntimeError("Failed to read first frame")

    #prev_gray, roi_offset = preprocess_and_crop(frame)
    
    # instantiate empty arrays for tracking and latency calculations
    next_track_id = 0 
    #explicit multi-target capacity metrics
    tracks, log, latencies, det_counts, track_counts, frames, thresh_frames = [], [], [], [], [], [], []
    # kalman state factoring   
    track_states = {}
    frame_idx = 0
    print("Checkpoint1")
    
    # instance of camera class 
    camera = Camera()
    print("Checkpoint2")
    # grab first frame to initialize prev_gray
    frame = camera.get_frame()
    print("Checkpoint3")
    prev_gray, roi_offset = preprocess_and_crop(frame)
    print("Checkpoint4")
    
    
    #begin loop, occurs over duration of video
    try: 
        print("Checkpoint5")

        while True: 

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

            #kalman calculations (tracking and state update)
            for t in tracks: 
                if frame_idx %5 == 0: # only track every 5 frames to gauge velocity better
                    track_states[t.id] = classify_state(t)
        
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
                  f"vis={t_vis*1000:.1f}ms, total={total_frame:.1f}ms")

            latencies.append(total_frame)
            det_counts.append(len(detections))
            track_counts.append(len(tracks))

             # log results
            log.append({
                "frame": frame_idx,
                "time": time.time(),
                "detections": len(detections),
                "tracks": len(tracks),
                "latency_ms": total_frame,
                "max_speed": max(
                    (np.hypot(t.kf.statePost[2,0], t.kf.statePost[3,0]) for t in tracks),
                    default=0
                )
            })

            prev_gray = curr_gray
            frame_idx += 1
    except KeyboardInterrupt: 
       print("\nStopping stream...")
    finally: 
       camera.close()

       df = pd.DataFrame(log)
       df.to_csv("run_001_control.csv", index=False)

       print(f"[INFO] Saved {len(df)} frames to run_001_control.csv")

       # save data frame
       df = pd.DataFrame(log)
       df.to_csv("run_001_control.csv", index=False)
    
    return latencies, det_counts, track_counts, frames, thresh_frames
        
latencies, det_counts, track_counts, frames, thresh_frames = process_video()

