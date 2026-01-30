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
from sklearn.linear_model import RANSACRegressor

UV_DEADBAND = 0.03   # mirror units (~3–5% of range is typical)
CENTER_EXPAND = 0.15  # fraction of image to allow around center


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
        try: 
            x, y, roi = extract_laser_centroid(
                img, min_area, max_area,
               mirror_uv=(s["u"], s["v"]))
            prev_xy = (x, y)
        except RuntimeError as e: 
            print(f"Skipping frame {s['image']}: {e}")
            continue

        debug_centroid_overlay(img, x, y, roi=roi) # visual centroid overlay check

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

def extract_laser_centroid(gray, min_area, max_area, mirror_uv=None):
    h, w = gray.shape
    expected = None
    roi = None
    if mirror_uv is not None: 
        u, v = mirror_uv
        expected = np.array(
            mirror_uv_to_image_xy(u, v, w, h), 
            dtype=np.float32) 
        roi = mirror_uv_to_roi(u, v, w, h)

    debug_roi_overlay(gray, roi, expected, u, v) # call before thresholding 

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

            bbox_area = w0 * h0
            print("bbox:", x, y, w0, h0)
            fill_ratio = area / max(bbox_area, 1)
            if fill_ratio < 0.25: 
                continue

            mask = (labels == i)
            ys, xs = np.where(mask) #suspected x and y vals

            intensity = gray[mask].astype(np.float32)
            peak = intensity.mean() # look at max or average 

            # --- compute centroid for this particular candidate
            cx = np.sum(xs*intensity) / np.sum(intensity)
            cy = np.sum(ys * intensity) / np.sum(intensity)
                # centroid raw
            print("centroid raw:", cx, cy)

            # --- enforce roi / quadrant matching for candidate centroid ---
            if roi is not None: 
                rx0, ry0, rx1, ry1 = roi
                if not (rx0 <= cx <= rx1 and ry0 <= cy <= ry1): 
                    continue

            # --- compare to spatial prior (soft limit) ---
            if expected is not None: 
                dist = np.hypot(cx - expected[0], cy - expected[1])
                spatial_weight = np.exp(-dist / 200.0)
            else: 
                spatial_weight = 1.0

            # final score computation (including spatial weight)
            score = peak * area * spatial_weight
            # temp debugging --> score output
            print(f"p={p} area={area:.0f} peak={peak:.1f} dist={dist:.1f} score={score:.1e}")
            print(f"---------------------------------------------------------")

            if score > best_score:
                best = mask
                best_score = score

        if best is not None:
            ys, xs = np.where(best)
            weights = gray[ys, xs].astype(np.float32)

            cx = np.sum(xs * weights) / np.sum(weights)
            cy = np.sum(ys * weights) / np.sum(weights)

            return float(cx), float(cy), roi

    raise RuntimeError("No valid laser spot found")


def debug_centroid(gray): 
   x, y = extract_laser_centroid(gray, min_area, max_area)
   vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
   cv2.circle(vis, (int(x), int(y)), 8, (0,0,255), 2)

   return vis

def show_full_image(title, img, max_size=900):
    h, w = img.shape[:2]
    scale = min(max_size / w, max_size / h, 1.0)
    vis = cv2.resize(img, (int(w*scale), int(h*scale)))
    cv2.imshow(title, vis)
    cv2.waitKey(0)

def show_debug_window(name, img, scale=0.4):
    h, w = img.shape[:2]
    resized = cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA
    )
    cv2.imshow(name, resized)

def debug_centroid_overlay(img, cx, cy, roi=None): 
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # visualize ROI to see supervision
    if roi is not None: 
        x0, y0, x1, y1 = roi
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255,0,0), 2)
    cv2.drawMarker(vis, (int(cx), int(cy)), (0,0,255), 
                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    show_full_image("Centroid debug", vis)
    cv2.waitKey(0)

def debug_roi_overlay(img, roi, expected, u, v):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # draw center cross
    h, w = img.shape
    cv2.drawMarker(vis, (w//2, h//2), (255,255,0),
                   cv2.MARKER_CROSS, 40, 2)

    # draw ROI
    if roi is not None:
        x0, y0, x1, y1 = roi
        cv2.rectangle(vis, (x0,y0), (x1,y1), (0,255,0), 3)

    # draw expected point
    if expected is not None:
        ex, ey = map(int, expected)
        cv2.circle(vis, (ex,ey), 6, (0,0,255), -1)

    # annotate mirror coords
    cv2.putText(
        vis, f"u={u:+.3f}, v={v:+.3f} | deadband={UV_DEADBAND}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (255,255,255), 2
    )

    show_debug_window("ROI debug", vis)
    cv2.waitKey(0)



def mirror_uv_to_image_xy(u, v, w, h):
    """
    Mirror → image mapping per spec:
    - -u → right
    - +u → left
    - -v → bottom
    - +v → top
    """
    x = (0.5 - 0.5 * u) * w
    y = (0.5 - 0.5 * v) * h
    return x, y

def mirror_uv_to_roi(u, v, w, h):
    cx = w * 0.5
    cy = h * 0.5

    # edge case : near (0,0)
    if abs(u) < UV_DEADBAND and abs(v) < UV_DEADBAND:
        dx = w * CENTER_EXPAND
        dy = h * CENTER_EXPAND
        return(
            int(cx-dx), int(cy-dy), 
            int(cx + dx), int(cy+dy)
        )
    # quadrant logic with deadband expansion
    # X: +u → left, -u → right
    if u > UV_DEADBAND: # not zero / exact
        x_min, x_max = 0, cx
    elif u < -UV_DEADBAND:
        x_min, x_max = cx, w
    else: 
        # near x=0 --> allow both sides
        x_min, x_max = cx-w*0.25, cx+w*0.25

    # Y: +v → top, -v → bottom
    if v > UV_DEADBAND:
        y_min, y_max = 0, cy
    elif v < -UV_DEADBAND:
        y_min, y_max = cy, h
    else: 
        # near y =0 --> allow both sides 
        y_min, y_max = cy - h*0.25, cy+h*0.25

    return (
        int(max(0, x_min)), int(max(0,y_min)),
        int(min(w, x_max)), int(min(h, y_max))
    )


def mirror_to_image(u,v): 
    """ Prediction module """ 
    uv = np.array([[u, v]])
    return (
        model_x.predict(uv)[0], 
        model_y.predict(uv)[0]
     )

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

