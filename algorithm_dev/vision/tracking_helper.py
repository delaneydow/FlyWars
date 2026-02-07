import cv2
from track import Track
import numpy as np
from track import Track

# constants to access 
THRESH_VAL = 12 # frame subtraction threshold
MIN_AREA = 5 # minimum blob area (pixels)
MAX_AREA = 3500 # max blob area (pixels)
MAX_MISSED = 5 # allows tracks to survive 5 frames without a detection
MAX_TRACK_DIST = 50 # max distance for track association (pixels), need to be a bit because the objects are falling / flying
MAX_TRACKS = 20 # TODO tune this

#kalman constants
FPS = 60.0 # or read from video metadata
DT = 1.0 / FPS

# ROI Configuration
# define ROI as fractions of frame dimensions
ROI_X_MIN = 0.5
ROI_Y_MIN = 0.0

# Precomputed morphology kernels (create once)
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# === HELPER FUNCTIONS ===
def preprocess_and_crop(frame):
    """ accepts either Mono8 image (H,W) or color image (H, W, 3) 
        Returns grayscale ROI """ 
    if frame.ndim == 2:
        gray = frame

    # Handle BGR images
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    else:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    h, w = gray.shape

    # Define ROI fractions
    x0 = int(ROI_X_MIN * w)
    y0 = int(ROI_Y_MIN * h)
    x1 = w
    y1 = h

    roi = gray[y0:y1, x0:x1]

    if roi.shape[0] < 8 or roi.shape[1] < 8:
        raise ValueError(f"ROI too small: {roi.shape} from frame {gray.shape}")

    return roi, (x0, y0)


def detect_moving_objects_fast(prev_gray, curr_gray):
    diff = cv2.absdiff(curr_gray, prev_gray)
    diff = cv2.GaussianBlur(diff, (5,5), 0) 
    _, thresh = cv2.threshold(diff, THRESH_VAL, 255, cv2.THRESH_BINARY)
    
    
    # reuse kernel3, do NOT create a new one
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel3) #temp disable, keep closed only
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel5) #closing thresh

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
    # dynamic gating based on velocity
    #max_dist_sq = MAX_TRACK_DIST * MAX_TRACK_DIST
    vx, vy = t.fk.statePost[2:,0]
    speed = np.hypot(vx, vy)

    adaptive_dist=max(MAX_TRACK_DIST, speed * DT *2)
    max_dist_sq = adaptive_dist* adaptive_dist

    predicted = {t: t.predict() for t in tracks} # predict once per frame
    # match existing tracks using greedy algorithm 
    for t in tracks: 
        px, py =predicted[t]
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
            if dist_sq < best_dist_sq and dist_sq < MAX_TRACK_DIST**2: # only allow detections near predicted position 
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
        if not used[i] and len(tracks) < MAX_TRACKS: #caps # of tracks to stabilize runtime
            tracks.append(Track(next_id, d))
            next_id += 1

    # remove and prune old tracks
    tracks = [t for t in tracks if t.missed <=MAX_MISSED]

    return tracks, next_id


def deduplicate_tracks(tracks, radius=15, vel_thresh=50): #TODO FIX THIS TO IMPROVE DEDUPLICATION, CURRENTLY BASED ON LAST POSITION ONLY
    # distance gating in association or deduplication only once every N frames
    # prevent same location collapse
    if len(tracks) <=1: 
        return tracks

    grid = {} # use spatial grid hasing / grid binning, duplicates only occur when tracks are close
    keep = []

    # prefer tracks with fewer misses
    tracks = sorted(tracks, key=lambda t: t.missed)

    for t in tracks: 
        x, y = t.last_position
        key = (x // radius, y // radius)

        vx1, vy1 = t.kf.statePost[2:, 0]

        duplicate = False

        # check in neighbordin bins
        for dx in (-1, 0, 1): 
            for dy in (-1, 0, 1): 

                neighbor_key = (key[0] + dx, key[1] + dy)
                if neighbor_key not in grid:
                    continue

                k = grid[neighbor_key]

                # get spatial distance
                dist = np.linalg.norm(
                    np.array(t.last_position) - np.array(k.last_position)
                )

                if dist > radius: 
                    continue

                # velocity similarity check
                vx2, vy2 = k.kf.statePost[2:,0]
                vel_diff = np.hypot(vx1 - vx2, vy1 - vy2)

                if vel_diff < vel_thresh: 
                    duplicate = True
                    break
                
                if duplicate: 
                    break

            if not duplicate: 
                keep.append(t)
                grid[key] = t
        #if not any(np.linalg.norm(
         #   np.array(t.last_position) - np.array(k.last_position)
        #) < radius for k in keep): 
         #   keep.append(t)
    return keep
