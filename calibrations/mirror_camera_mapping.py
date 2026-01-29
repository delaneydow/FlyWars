# #mirror_camera_mapping.py 

""" This will extract laser spot centroid from images 
and fit the mirror --> image mapping """ 


import cv2
import numpy as np 
import json
from pathlib import Path

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline 


def validate_resolution(img, meta): 
    if img is None: 
        raise ValueError("validate_resolution received None image")

    h, w = img.shape
    expected_w, expected_h = meta["camera"]["resolution"]

    if w != expected_w or h != expected_h:
        raise ValueError(
            f"Resolution mismatch: image = {w}x{h}, "
            f"json={expected_w}x{expected_h}"
            )

def load_calibration(json_path, min_area, max_area):
    json_path = Path(json_path).resolve()
    base_dir = json_path.parent
    ref_path = base_dir / "0.00X_0.00Y_adjusted.png"
    prev_img = None

    print("Loading calibration from: ", json_path)

    f = open(str(json_path), "r", encoding="utf-8-sig")
    data = json.load(f)
    f.close()

    mirror_uv, beam_xy = [], []
    # sort samples by mirror motion
    samples = data["samples"]
    samples = sorted(samples, key=lambda s: (s["u"], s["v"]))

    for s in samples: 
        img_path = base_dir / s["image"]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        validate_resolution(img, data) # enforce consistency at load time
        if img is None: 
            raise FileNotFoundError(s["image"])
        #if prev_img is None: 
         #   prev_img = img
          #  continue

        try: 
            x, y = extract_laser_centroid(img, min_area, max_area)
        except RuntimeError as e: 
            print(f"Skipping frame {s['image']}: {e}")
            continue

        debug_centroid_overlay(img, x, y) # visual centroid overlay check

        mirror_uv.append([s["u"], s["v"]]) #shape: (N,2)
        beam_xy.append([x,y]) # shape (N, 2)

    return (
        np.asarray(mirror_uv, dtype=np.float32), 
        np.asarray(beam_xy, dtype=np.float32),
        data # data = metadata for later use
        )

def estimate_spot_area(images): 
    areas = []
    EDGE_MARGIN = 5
    PERCENTILES = [99.7, 99.5, 99.2, 99.0]

    for gray in images: 
        h, w = gray.shape
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        spot = None

        # percentile ladder
        for p in PERCENTILES: 
            thresh_val = np.percentile(blur, p)
            _, candidate = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)

            candidate = cv2.morphologyEx(
                candidate, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            candidate = cv2.morphologyEx(
                candidate, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

            num, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)

            # pick best laser-like blob
            best = None

            for i in range(1, num): 
                area = stats[i, cv2.CC_STAT_AREA]
                cx, cy, w0, h0, _ = stats[i]

                # reject tiny noise
                if area < 100: 
                    continue

                # reject FOV edge
                if cx < EDGE_MARGIN or cx > w-EDGE_MARGIN or cy < EDGE_MARGIN or cy > h-EDGE_MARGIN: 
                    continue

                best=area
                break

            if best is not None:
                areas.append(best)
                break # stop percetile ladder

    return np.asarray(areas)

def load_all_calibration_images(json_path):
    json_path = Path(json_path).resolve()
    base_dir = json_path.parent

    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    images = []
    for s in data["samples"]:
        img_path = base_dir / s["image"]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(img_path)

        validate_resolution(img, data)
        images.append(img)

    return images, data

def extract_laser_centroid(gray, min_area, max_area):
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    PERCENTILES = [99.7, 99.5, 99.2, 99.0]
    EDGE_MARGIN = 20

    for p in PERCENTILES:
        thresh_val = np.percentile(blur, p)
        _, spot = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)

        spot = cv2.morphologyEx(
            spot, cv2.MORPH_OPEN, np.ones((3,3), np.uint8)
        )
        spot = cv2.morphologyEx(
            spot, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8)
        )

        num, labels, stats, _ = cv2.connectedComponentsWithStats(spot)

        best = None
        best_score = -1

        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, w0, h0, _ = stats[i]

            # --- area sanity ---
            if area < min_area or area > max_area:
                continue

            # --- edge rejection ---
            if x < EDGE_MARGIN or y < EDGE_MARGIN or \
               x+w0 > w-EDGE_MARGIN or y+h0 > h-EDGE_MARGIN:
                continue

            # --- shape sanity (laser ≈ round-ish) ---
            aspect = w0 / max(h0, 1)
            if aspect < 0.5 or aspect > 2.0:
                continue

            mask = (labels == i)
            intensity = gray[mask].astype(np.float32)

            # bright AND compact
            score = intensity.mean() * area

            if score > best_score:
                best = mask
                best_score = score

        if best is not None:
            ys, xs = np.where(best)
            weights = gray[ys, xs].astype(np.float32)

            cx = np.sum(xs * weights) / np.sum(weights)
            cy = np.sum(ys * weights) / np.sum(weights)

            return float(cx), float(cy)

    raise RuntimeError("No valid laser spot found")




def extract_laser_centroid_diff(gray, reference):
    # Absolute difference
    diff = cv2.absdiff(gray, reference)

    # Normalize
    #diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # suppress background 
    diff = cv2.GaussianBlur(diff, (5,5), 0)

    # Threshold changed pixels
    thresh = np.percentile(diff, 99.5)
    _, mask = cv2.threshold(diff, 0, 255,
                             cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    if np.count_nonzero(mask) < 30:
        raise RuntimeError("No moving laser spot detected")

    # Distance transform
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, _, _, (cx, cy) = cv2.minMaxLoc(dist)

    return float(cx), float(cy)



def debug_centroid(gray): 
   x, y = extract_laser_centroid(gray, min_area, max_area)
   vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
   cv2.circle(vis, (int(x), int(y)), 8, (0,0,255), 2)

   return vis

def fit_poly(mirror_uv, beam_xy, degree = 3): 
    """ fit mirror --> image mapping using third order polynomial to start
    due to beamsplitter, mirror, perspective, inversion don't necessarily want 
    to use an affine filter 

    input: mirror uv coordinates, beam xy coordinates, degree of polynomial
    output: models 
    """ 
    models = []

    for dim in range(2): 
        model = Pipeline([
            ("poly", PolynomialFeatures(degree)), 
            ("ridge", Ridge(alpha=1e-3))
            ])
        model.fit(mirror_uv, beam_xy[:, dim])
        models.append(model)

    return models 

def show_full_image(title, img, max_size=900):
    h, w = img.shape[:2]
    scale = min(max_size / w, max_size / h, 1.0)
    vis = cv2.resize(img, (int(w*scale), int(h*scale)))
    cv2.imshow(title, vis)
    cv2.waitKey(0)

def debug_centroid_overlay(img, cx, cy): 
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawMarker(vis, (int(cx), int(cy)), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    show_full_image("Centroid debug", vis)
    cv2.waitKey(0)


def mirror_to_image(u,v): 
    """ Prediction module """ 
    uv = np.array([[u, v]])
    return (
        model_x.predict(uv)[0], 
        model_y.predict(uv)[0]
     )

