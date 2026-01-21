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
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel5) #closing thresh

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    detections = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
        if areas:
            print("Largest area:", max(areas), "num blobs:", len(areas))
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


def deduplicate_tracks(tracks, radius=15): #TODO FIX THIS TO IMPROVE DEDUPLICATION, CURRENTLY BASED ON LAST POSITION ONLY
    # distance gating in association or deduplication only once every N frames
    # prevent same location collapse
    if len(tracks) <=1: 
        return tracks

    grid = {} # use spatial grid hasing / grid binning, duplicates only occur when tracks are close
    keep = []

    # prefer older tracks
    tracks = sorted(tracks, key=lambda t: t.last_seen, reverse=True)

    for t in tracks: 
        x, y = t.last_position
        key = (x // radius, y // radius)

        duplicate = False

        for dx in (-1, 0, 1): 
            for dy in (-1, 0, 1): 
                if (key[0] + dx, key[1] + dy) in grid: 
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
