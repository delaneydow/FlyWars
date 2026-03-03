import cv2
from algorithm_dev.vision.track import Track
import numpy as np
from scipy.optimize import linear_sum_assignment


# constants to access 
THRESH_VAL = 12 # frame subtraction threshold
MIN_AREA = 5 # minimum blob area (pixels)
MAX_AREA = 3500 # max blob area (pixels)
MAX_MISSED = 5 # allows tracks to survive 5 frames without a detection
MAX_TRACK_DIST = 50 # max distance for track association (pixels), need to be a bit because the objects are falling / flying
MAX_TRACKS = 20 # TODO tune this

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


def associate_detections_to_tracking_fast(detections, tracks, next_id): #TODO take out dt 

    # CASE 1: NO DETECTIONS
    if not detections: 
        for t in tracks: 
            t.update(None)
        return [t for t in tracks if t.missed <= MAX_MISSED], next_id

    used = [False] * len(detections)
    track_matches = {}

    # CASE 2: find best detection per track via fast gated nearest neighbor
    
    # fast gated nearest neighbor,
    for ti, t in enumerate(tracks):
        px, py = t.last_position
        best_i = -1
        best_dist = MAX_TRACK_DIST*MAX_TRACK_DIST

        for di, d in enumerate(detections):
            if used[di]:
                continue

            dx = px - d[0]
            dy = py - d[1]
            dist = dx*dx + dy*dy

            if dist < best_dist:
                best_dist = dist
                best_i = di

        if best_i >=0:
            track_matches[ti] = best_i
            used[best_i] = True
        else:
            t.update(None)

    # CASE 3: create new tracks
    for di, d in enumerate(detections): 
        if not used[di] and len(tracks) < MAX_TRACKS:
            tracks.append(Track(next_id, d))
            next_id += 1

       
    # CASE 4: remove and prune old tracks
    for ti, di in track_matches.items():
        tracks[ti].update(detections[di])
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
        
        duplicate = False

        # check in neighbording bins
        for dx in (-1, 0, 1): 
            for dy in (-1, 0, 1): 

                neighbor_key = (key[0] + dx, key[1] + dy)
                if neighbor_key not in grid:
                    continue

                k = grid[neighbor_key]

                # get spatial distance
                dxp = t.last_position[0] - k.last_position[0]
                dyp = t.last_position[1] - k.last_position[1]
                dist_sq = dxp*dxp + dyp*dyp

                if dist_sq > radius**2: 
                    continue

                # velocity similarity check
                vx1, vy1 = t.kf.statePost[2:, 0]
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
