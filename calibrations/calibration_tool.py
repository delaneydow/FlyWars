"""
calibration_tool.py
====================
Interactive calibration pipeline for raw laser images.

WORKFLOW:
  1. Point it at a folder of .raw (or .png) images
  2. It shows each image so you can click the laser centroid
  3. Saves/updates mirror_coordinates.json with centroids
  4. Builds mirror_map.npz ready for MirrorPlanner

USAGE:
  # Label centroids interactively and build map:
  python calibration_tool.py --images ./my_raw_images --json mirror_coordinates.json --build-map

  # Re-build map from existing JSON only (skip labeling):
  python calibration_tool.py --json mirror_coordinates.json --build-map --skip-label

  # Label only, no map build:
  python calibration_tool.py --images ./my_raw_images --json mirror_coordinates.json

  # After building a map, review and re-label high-error images:
  python calibration_tool.py --images ./my_raw_images --json mirror_coordinates.json \
      --review-errors --map-out mirror_map.npz

  # Review with a custom error threshold (default: 2.0 × median error):
  python calibration_tool.py --images ./my_raw_images --json mirror_coordinates.json \
      --review-errors --error-threshold 3.0 --map-out mirror_map.npz

  # Review then rebuild the map with the corrected labels:
  python calibration_tool.py --images ./my_raw_images --json mirror_coordinates.json \
      --review-errors --build-map --map-out mirror_map.npz

CONTROLS (during labeling):
  Left-click          : set centroid (green cross appears)
  Right-click / R     : reset centroid for this image
  S / Enter / Space   : confirm and save centroid, move to next image
  B                   : go back one image
  Q / Escape          : quit (saves progress so far)
  Scroll wheel        : zoom in/out around cursor
  Middle-click + drag : pan when zoomed
  Z                   : zoom to fit
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import Rbf
from scipy.spatial import ConvexHull


# ---------------------------------------------------------------------------
# RAW image loader
# ---------------------------------------------------------------------------

def load_raw_image(path: Path, width: int = None, height: int = None) -> np.ndarray:
    """
    Load a .raw file as a grayscale uint8 image.
    Tries several common raw formats automatically.
    Pass width/height to override auto-detection.
    """
    raw_bytes = path.read_bytes()
    n = len(raw_bytes)

    # --- try to auto-detect resolution if not given ---
    if width is None or height is None:
        # common sensor resolutions to try (w, h)
        candidates = [
            (2048, 1536), (1920, 1080), (1280, 960),
            (1280, 720),  (640, 480),   (2592, 1944),
        ]
        for w, h in candidates:
            if n == w * h:           # 8-bit
                width, height = w, h
                dtype = np.uint8
                break
            elif n == w * h * 2:     # 16-bit little-endian
                width, height = w, h
                dtype = np.uint16
                break
        else:
            # fallback: assume square-ish 8-bit
            side = int(math.isqrt(n))
            width, height = side, side
            dtype = np.uint8
    else:
        dtype = np.uint8 if len(raw_bytes) == width * height else np.uint16

    arr = np.frombuffer(raw_bytes, dtype=dtype).reshape((height, width))

    # normalise to uint8 for display
    if arr.dtype != np.uint8:
        arr = (arr.astype(np.float32) / arr.max() * 255).astype(np.uint8)

    return arr


def load_image(path: Path, raw_width=None, raw_height=None) -> np.ndarray:
    """Load .png/.jpg/.tiff directly; .raw via raw loader."""
    suffix = path.suffix.lower()
    if suffix == ".raw":
        return load_raw_image(path, raw_width, raw_height)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


# ---------------------------------------------------------------------------
# Filename → (u, v) parser  e.g. "0.05X_-0.10Y.raw"
# ---------------------------------------------------------------------------

def parse_uv_from_filename(name: str):
    """Return (u, v) floats from filenames like '0.05X_-0.10Y.raw'."""
    m = re.search(r"([+-]?\d+(?:\.\d+)?)X_([+-]?\d+(?:\.\d+)?)Y", name, re.IGNORECASE)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


# ---------------------------------------------------------------------------
# Per-sample error analysis
# ---------------------------------------------------------------------------

def compute_per_sample_errors(json_path: Path) -> list[dict]:
    """
    Fit a TPS model to all labelled samples and return per-sample reprojection
    errors, sorted worst-first.

    Each returned dict contains:
        image       – filename
        u, v        – mirror commands
        centroid    – labelled [cx, cy]
        predicted   – TPS-predicted [cx, cy]
        error_px    – Euclidean reprojection error in pixels
    """
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    samples = [s for s in data["samples"] if "centroid" in s]
    if len(samples) < 4:
        raise ValueError(f"Need at least 4 labelled samples, found {len(samples)}")

    uv = np.array([[s["u"], s["v"]] for s in samples], dtype=np.float64)
    xy = np.array([s["centroid"]     for s in samples], dtype=np.float64)

    # deduplicate (keep first occurrence so we can map back to samples)
    seen = {}
    dedup_idx = []
    for i, key in enumerate(map(tuple, uv)):
        if key not in seen:
            seen[key] = i
            dedup_idx.append(i)
    uv_d = uv[dedup_idx]
    xy_d = xy[dedup_idx]
    samples_d = [samples[i] for i in dedup_idx]

    uv_mean = uv_d.mean(axis=0)
    uv_std  = uv_d.std(axis=0)
    uvn = (uv_d - uv_mean) / uv_std
    u_n, v_n = uvn[:, 0], uvn[:, 1]

    fx = Rbf(u_n, v_n, xy_d[:, 0], function="thin_plate", smooth=1e-2)
    fy = Rbf(u_n, v_n, xy_d[:, 1], function="thin_plate", smooth=1e-2)

    pred_x = fx(u_n, v_n)
    pred_y = fy(u_n, v_n)

    results = []
    for i, s in enumerate(samples_d):
        err = float(np.hypot(pred_x[i] - xy_d[i, 0], pred_y[i] - xy_d[i, 1]))
        results.append({
            "image":     s["image"],
            "u":         s["u"],
            "v":         s["v"],
            "centroid":  s["centroid"],
            "predicted": [round(float(pred_x[i]), 1), round(float(pred_y[i]), 1)],
            "error_px":  round(err, 2),
        })

    results.sort(key=lambda r: r["error_px"], reverse=True)
    return results


def print_error_report(errors: list[dict], threshold_px: float, images_dir: Path | None = None):
    """
    Pretty-print a per-sample error table and flag outliers.

    Args:
        errors:        Output of compute_per_sample_errors(), sorted worst-first.
        threshold_px:  Errors >= this value are flagged as outliers.
        images_dir:    If given, warn about flagged images that can't be found on disk.
    """
    all_errors = [r["error_px"] for r in errors]
    median_err = float(np.median(all_errors))
    mean_err   = float(np.mean(all_errors))
    max_err    = float(np.max(all_errors))

    outliers = [r for r in errors if r["error_px"] >= threshold_px]

    print("\n" + "=" * 72)
    print("  PER-SAMPLE REPROJECTION ERROR REPORT")
    print("=" * 72)
    print(f"  Samples:        {len(errors)}")
    print(f"  Mean error:     {mean_err:.2f} px")
    print(f"  Median error:   {median_err:.2f} px")
    print(f"  Max error:      {max_err:.2f} px")
    print(f"  Outlier cutoff: {threshold_px:.2f} px  ({len(outliers)} flagged)")
    print("-" * 72)
    print(f"  {'#':>3}  {'Image':<38}  {'u':>6} {'v':>6}  {'Error':>8}  Flag")
    print("-" * 72)

    for rank, r in enumerate(errors, 1):
        flag = "  ◄ HIGH ERROR" if r["error_px"] >= threshold_px else ""
        print(f"  {rank:>3}  {r['image']:<38}  {r['u']:>+6.3f} {r['v']:>+6.3f}"
              f"  {r['error_px']:>7.2f}px{flag}")

    print("=" * 72)

    if outliers:
        print(f"\n  {len(outliers)} image(s) with error ≥ {threshold_px:.2f}px:")
        for r in outliers:
            on_disk = ""
            if images_dir is not None:
                img_path = images_dir / r["image"]
                if not img_path.exists():
                    on_disk = "  [NOT FOUND ON DISK]"
            print(f"    • {r['image']}  (error={r['error_px']:.2f}px,  "
                  f"labelled={r['centroid']},  predicted={r['predicted']}){on_disk}")
    else:
        print("\n  No outliers found — all errors are below the threshold.")

    print()


# ---------------------------------------------------------------------------
# Interactive centroid labeler
# ---------------------------------------------------------------------------

class CentroidLabeler:
    WINDOW = "Calibration Labeler  [click=centroid | S=save | B=back | Q=quit | scroll=zoom]"

    def __init__(self, images_dir: Path, json_path: Path,
                 raw_width=None, raw_height=None,
                 filter_names: list[str] | None = None):
        """
        Args:
            filter_names: If given, only present images whose filename is in
                          this list (used for error-review mode).
        """
        self.images_dir = images_dir
        self.json_path = json_path
        self.raw_width = raw_width
        self.raw_height = raw_height

        self.data = self._load_json()

        all_files = sorted(
            [p for p in images_dir.iterdir()
             if p.suffix.lower() in (".raw", ".png", ".jpg", ".tiff", ".tif")],
            key=lambda p: p.name
        )

        if filter_names is not None:
            # Preserve the order supplied by filter_names (worst-first from error report)
            name_to_path = {p.name: p for p in all_files}
            self.image_files = [
                name_to_path[n] for n in filter_names if n in name_to_path
            ]
            missing = [n for n in filter_names if n not in name_to_path]
            if missing:
                print(f"[WARN] {len(missing)} flagged image(s) not found in {images_dir}:")
                for m in missing:
                    print(f"       • {m}")
        else:
            self.image_files = all_files

        if not self.image_files:
            print(f"[WARN] No images to label in {images_dir}")

        # view state
        self._zoom = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._drag_start = None
        self._pan_start = None
        self._centroid = None   # (cx, cy) in image coords
        self._img = None        # current full-res grayscale
        self._idx = 0

    # ------------------------------------------------------------------
    def _load_json(self) -> dict:
        if self.json_path.exists():
            with open(self.json_path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        # minimal scaffold
        return {
            "camera": {"resolution": [2048, 1536], "pixel_size_um": None, "distance_mm": 235},
            "beam": {},
            "laser": {"spot_radius_px": 35, "spot_radius_px_std": 5},
            "mirror": {
                "mechanical_range": {"u": [-0.50, 0.50], "v": [-0.50, 0.50]},
                "optical_range":    {"u": [-0.16, 0.15], "v": [-0.23, 0.20]}
            },
            "samples": []
        }

    def _save_json(self):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)
        print(f"  → Saved {self.json_path}")

    def _get_sample(self, img_name: str):
        for s in self.data["samples"]:
            if Path(s["image"]).name == img_name:
                return s
        return None

    def _upsert_sample(self, img_path: Path, u: float, v: float, cx: float, cy: float):
        name = img_path.name
        for s in self.data["samples"]:
            if Path(s["image"]).name == name:
                s["centroid"] = [round(cx, 1), round(cy, 1)]
                return
        self.data["samples"].append({
            "image": name,
            "u": u,
            "v": v,
            "centroid": [round(cx, 1), round(cy, 1)]
        })

    # ------------------------------------------------------------------
    # View helpers
    # ------------------------------------------------------------------

    def _to_display(self, img: np.ndarray, error_info: str | None = None) -> tuple:
        """Apply pan/zoom and render overlay."""
        h, w = img.shape
        # crop region in image coords
        vw = int(w / self._zoom)
        vh = int(h / self._zoom)
        x0 = int(np.clip(self._pan_x, 0, w - vw))
        y0 = int(np.clip(self._pan_y, 0, h - vh))
        x1, y1 = x0 + vw, y0 + vh

        crop = img[y0:y1, x0:x1]
        disp_w, disp_h = min(w, 1400), min(h, 1050)
        vis = cv2.resize(crop, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        # draw centroid if set
        if self._centroid is not None:
            cx, cy = self._centroid
            # map image coords → display coords
            sx = int((cx - x0) / vw * disp_w)
            sy = int((cy - y0) / vh * disp_h)
            cv2.drawMarker(vis, (sx, sy), (0, 255, 0),
                           cv2.MARKER_CROSS, 30, 2)
            cv2.circle(vis, (sx, sy), 15, (0, 255, 0), 1)
            label = f"({cx:.1f}, {cy:.1f})"
            cv2.putText(vis, label, (sx + 18, sy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        # draw TPS-predicted centroid if available (orange cross)
        if self._predicted is not None:
            px_, py_ = self._predicted
            sx = int((px_ - x0) / vw * disp_w)
            sy = int((py_ - y0) / vh * disp_h)
            cv2.drawMarker(vis, (sx, sy), (0, 165, 255),
                           cv2.MARKER_TILTED_CROSS, 24, 2)
            cv2.putText(vis, f"TPS ({px_:.1f}, {py_:.1f})",
                        (sx + 18, sy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 165, 255), 1)

        # draw line between predicted and labelled
        if self._centroid is not None and self._predicted is not None:
            cx, cy   = self._centroid
            px_, py_ = self._predicted
            s1 = (int((cx  - x0) / vw * disp_w), int((cy  - y0) / vh * disp_h))
            s2 = (int((px_ - x0) / vw * disp_w), int((py_ - y0) / vh * disp_h))
            cv2.line(vis, s1, s2, (0, 100, 255), 1, cv2.LINE_AA)

        # error badge
        if error_info:
            cv2.putText(vis, error_info, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)

        return vis, (x0, y0, vw, vh, disp_w, disp_h)

    def _display_to_image(self, dx, dy, view_params):
        x0, y0, vw, vh, dw, dh = view_params
        ix = x0 + dx / dw * vw
        iy = y0 + dy / dh * vh
        return ix, iy

    def _reset_view(self):
        self._zoom = 1.0
        self._pan_x = 0
        self._pan_y = 0

    # ------------------------------------------------------------------
    # Mouse callback
    # ------------------------------------------------------------------

    def _make_mouse_cb(self, view_ref: list):
        """view_ref[0] holds the latest view_params tuple."""
        def cb(event, x, y, flags, param):
            vp = view_ref[0]
            if vp is None:
                return

            if event == cv2.EVENT_LBUTTONDOWN:
                ix, iy = self._display_to_image(x, y, vp)
                self._centroid = (round(ix, 1), round(iy, 1))

            elif event == cv2.EVENT_RBUTTONDOWN:
                self._centroid = None

            elif event == cv2.EVENT_MBUTTONDOWN:
                self._drag_start = (x, y)
                self._pan_start  = (self._pan_x, self._pan_y)

            elif event == cv2.EVENT_MOUSEMOVE and self._drag_start:
                dx = x - self._drag_start[0]
                dy = y - self._drag_start[1]
                x0, y0, vw, vh, dw, dh = vp
                self._pan_x = int(self._pan_start[0] - dx / dw * vw)
                self._pan_y = int(self._pan_start[1] - dy / dh * vh)

            elif event == cv2.EVENT_MBUTTONUP:
                self._drag_start = None

            elif event == cv2.EVENT_MOUSEWHEEL:
                h, w = self._img.shape
                # zoom toward cursor
                ix, iy = self._display_to_image(x, y, vp)
                factor = 1.15 if flags > 0 else (1 / 1.15)
                self._zoom = np.clip(self._zoom * factor, 1.0, 20.0)
                vw_new = w / self._zoom
                vh_new = h / self._zoom
                self._pan_x = int(ix - x / vp[4] * vw_new)
                self._pan_y = int(iy - y / vp[5] * vh_new)

        return cb

    # ------------------------------------------------------------------
    # Main labeling loop
    # ------------------------------------------------------------------

    def run(self, error_lookup: dict[str, dict] | None = None):
        """
        Args:
            error_lookup: Optional dict mapping image filename → error record
                          (from compute_per_sample_errors). When provided, the
                          labeler shows the TPS-predicted centroid as an orange
                          cross and displays the error in px as an overlay.
        """
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 1400, 1050)

        view_ref = [None]
        cb = self._make_mouse_cb(view_ref)
        cv2.setMouseCallback(self.WINDOW, cb)

        i = 0
        files = self.image_files
        n = len(files)

        mode = "REVIEW (high-error)" if error_lookup else "LABEL"
        print(f"\nFound {n} images to {mode} in {self.images_dir}")
        print("Controls: click=centroid | S/Enter=save+next | B=back | R=reset | Z=zoom-fit | Q=quit\n")

        while 0 <= i < n:
            img_path = files[i]
            u, v = parse_uv_from_filename(img_path.name)
            if u is None:
                print(f"  [SKIP] Can't parse (u,v) from: {img_path.name}")
                i += 1
                continue

            # load image
            try:
                self._img = load_image(img_path, self.raw_width, self.raw_height)
            except Exception as e:
                print(f"  [ERROR] {img_path.name}: {e}")
                i += 1
                continue

            # pre-fill centroid from existing JSON
            existing = self._get_sample(img_path.name)
            if existing and "centroid" in existing:
                self._centroid = tuple(existing["centroid"])
            else:
                self._centroid = None

            # error overlay data
            err_record = (error_lookup or {}).get(img_path.name)
            self._predicted = tuple(err_record["predicted"]) if err_record else None
            error_info = (f"TPS error: {err_record['error_px']:.2f}px  "
                          f"(green=labelled  orange=TPS predicted)"
                          if err_record else None)

            self._reset_view()
            status = "existing" if existing else "new"
            err_str = f"  error={err_record['error_px']:.2f}px" if err_record else ""
            print(f"[{i+1}/{n}] {img_path.name}  u={u:+.3f} v={v:+.3f}  [{status}]{err_str}")

            while True:
                vis, vp = self._to_display(self._img, error_info=error_info)
                view_ref[0] = vp

                # HUD
                h_disp, w_disp = vis.shape[:2]
                hud = f"{img_path.name}  [{i+1}/{n}]  u={u:+.3f} v={v:+.3f}  zoom={self._zoom:.1f}x"
                cv2.putText(vis, hud, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 50), 1)
                tip = "S/Enter=confirm  B=back  R=reset  Z=fit  Q=quit"
                cv2.putText(vis, tip, (10, h_disp - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
                if self._centroid is None:
                    cv2.putText(vis, "NO CENTROID SET", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)

                cv2.imshow(self.WINDOW, vis)
                key = cv2.waitKey(30) & 0xFF

                if key in (ord('s'), ord('S'), 13, 32):   # S / Enter / Space
                    if self._centroid is None:
                        print("  [!] No centroid set — click first or press R to skip")
                        continue
                    cx, cy = self._centroid
                    self._upsert_sample(img_path, u, v, cx, cy)
                    self._save_json()
                    print(f"  ✓  centroid=({cx:.1f}, {cy:.1f})")
                    i += 1
                    break

                elif key in (ord('b'), ord('B')):
                    i = max(0, i - 1)
                    break

                elif key in (ord('r'), ord('R')):
                    self._centroid = None

                elif key in (ord('z'), ord('Z')):
                    self._reset_view()

                elif key in (ord('q'), ord('Q'), 27):      # Q / Escape
                    print("\nLabeling stopped. Progress saved.")
                    cv2.destroyAllWindows()
                    return

        cv2.destroyAllWindows()
        print(f"\nAll {n} images processed.")


# ---------------------------------------------------------------------------
# map builder  →  mirror_map.npz
# ---------------------------------------------------------------------------

def _shrink_hull(hull_pts: np.ndarray, margin: float) -> np.ndarray:
    """
    Return a convex hull shrunk inward by `margin` pixels on every side.

    Strategy: compute the centroid of the hull, move each vertex toward
    the centroid by `margin` pixels along the inward normal direction.
    This is a simple uniform erosion that works well for convex shapes.
    """
    centroid = hull_pts.mean(axis=0)
    shrunk = []
    for pt in hull_pts:
        direction = centroid - pt
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            shrunk.append(pt)
        else:
            shrunk.append(pt + direction / dist * margin)
    return np.array(shrunk, dtype=np.float32)


def build_mirror_map(json_path: Path, out_path: Path, grid_res: int = 200):
    """
    Reads mirror_coordinates.json and writes mirror_map.npz containing:
      uv, xy          – raw calibration samples
      uv_mean/std     – normalisation constants
      x_map, y_map    – flattened dense grid (image coords) — SAFE points only
      u_map, v_map    – corresponding mirror commands         — SAFE points only
      bounds          – (u_min, u_max, v_min, v_max) of SAFE grid
      hull            – convex hull vertices of SAFE reachable (x,y)
      hull_raw        – convex hull before spot-radius erosion (for display)
      spot_radius_px  – the margin used, saved for reference

    Safety margin: every grid point whose beam footprint (centroid ± spot_radius_px)
    would extend outside the raw reachable hull is excluded.  The saved hull,
    x_map/y_map, u_map/v_map, and UV bounds all reflect this safe region only,
    so MirrorPlanner.is_reachable() and find_uv_for_xy() never return a command
    that clips the beam at the boundary.
    """
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    samples = [s for s in data["samples"] if "centroid" in s]
    if len(samples) < 4:
        raise ValueError(f"Need at least 4 labelled samples, found {len(samples)}")

    spot_r = float(data.get("laser", {}).get("spot_radius_px", 35))
    print(f"  Spot radius = {spot_r:.1f}px  (from laser.spot_radius_px in JSON)")

    uv = np.array([[s["u"], s["v"]] for s in samples], dtype=np.float32)
    xy = np.array([s["centroid"]     for s in samples], dtype=np.float32)

    # deduplicate
    _, idx = np.unique(uv, axis=0, return_index=True)
    uv = uv[idx];  xy = xy[idx]
    print(f"  Building map from {len(uv)} unique calibration points")

    # normalise
    uv_mean = uv.mean(axis=0)
    uv_std  = uv.std(axis=0)
    uvn = (uv - uv_mean) / uv_std
    u_n, v_n = uvn[:, 0], uvn[:, 1]
    x_c, y_c = xy[:, 0], xy[:, 1]

    # TPS fit  uv → xy
    fx = Rbf(u_n, v_n, x_c, function="thin_plate", smooth=1e-2)
    fy = Rbf(u_n, v_n, y_c, function="thin_plate", smooth=1e-2)

    # RMS sanity
    pred = np.column_stack([fx(u_n, v_n), fy(u_n, v_n)])
    rms = np.sqrt(np.mean(np.sum((pred - xy) ** 2, axis=1)))
    print(f"  TPS RMS reprojection error: {rms:.2f} px")

    from matplotlib.path import Path as MplPath

    # ── calibration-sample hull (image coords) ────────────────────────────
    # This is the ONLY authoritative boundary — derived from actual measured
    # sample positions, not from TPS-extrapolated grid values.
    # TPS extrapolation outside this region is unreliable and excluded.
    sample_hull_idx = ConvexHull(xy).vertices
    sample_hull_xy  = xy[sample_hull_idx].astype(np.float32)
    print(f"  Calibration sample hull: {len(sample_hull_idx)} vertices")

    # ── safe hull = sample hull eroded inward by spot_radius_px ──────────
    hull_safe = _shrink_hull(sample_hull_xy, spot_r)
    try:
        safe_hull_idx = ConvexHull(hull_safe).vertices
        hull_safe = hull_safe[safe_hull_idx]
    except Exception:
        pass   # degenerate — keep as-is

    safe_path  = MplPath(hull_safe)
    cal_path   = MplPath(sample_hull_xy)

    # ── dense grid clamped to measured UV range ───────────────────────────
    # Grid spans only measured UV so TPS stays in the interpolation regime.
    u_min, u_max = uv[:, 0].min(), uv[:, 0].max()
    v_min, v_max = uv[:, 1].min(), uv[:, 1].max()

    u_grid = np.linspace(u_min, u_max, grid_res)
    v_grid = np.linspace(v_min, v_max, grid_res)
    UU, VV = np.meshgrid(u_grid, v_grid)
    uu_flat = UU.ravel();  vv_flat = VV.ravel()

    uvn_grid = (np.column_stack([uu_flat, vv_flat]) - uv_mean) / uv_std
    xx_flat = fx(uvn_grid[:, 0], uvn_grid[:, 1])
    yy_flat = fy(uvn_grid[:, 0], uvn_grid[:, 1])

    pts_all = np.column_stack([xx_flat, yy_flat])

    # ── filter 1: inside calibration sample hull (no extrapolation) ──────
    in_cal_mask  = cal_path.contains_points(pts_all)

    # ── filter 2: erode by spot radius for safety margin ─────────────────
    in_safe_mask = safe_path.contains_points(pts_all)

    inside_mask = in_cal_mask & in_safe_mask
    n_total  = len(xx_flat)
    n_in_cal = int(in_cal_mask.sum())
    n_safe   = int(inside_mask.sum())
    print(f"  Grid points inside calibration hull:   {n_in_cal} / {n_total}")
    print(f"  Grid points after spot-radius erosion: {n_safe} / {n_total}  "
          f"(margin={spot_r:.1f}px)")

    if n_safe < 10:
        raise ValueError(
            f"Only {n_safe} safe grid points after {spot_r:.0f}px erosion. "
            "Check spot_radius_px in your JSON or add more calibration samples "
            "closer to the edges.")

    xx_safe = xx_flat[inside_mask].astype(np.float32)
    yy_safe = yy_flat[inside_mask].astype(np.float32)
    uu_safe = uu_flat[inside_mask].astype(np.float32)
    vv_safe = vv_flat[inside_mask].astype(np.float32)

    # ── safe UV bounds (tightest box around safe grid points) ────────────
    u_min_safe = float(uu_safe.min());  u_max_safe = float(uu_safe.max())
    v_min_safe = float(vv_safe.min());  v_max_safe = float(vv_safe.max())
    bounds_safe = np.array([u_min_safe, u_max_safe,
                             v_min_safe, v_max_safe], dtype=np.float32)

    print(f"  Safe UV bounds:  u=[{u_min_safe:.3f}, {u_max_safe:.3f}]  "
          f"v=[{v_min_safe:.3f}, {v_max_safe:.3f}]")
    print(f"  (raw UV bounds:  u=[{u_min:.3f}, {u_max:.3f}]  "
          f"v=[{v_min:.3f}, {v_max:.3f}])")

    np.savez(out_path,
             uv=uv, xy=xy,
             uv_mean=uv_mean, uv_std=uv_std,
             x_map=xx_safe,
             y_map=yy_safe,
             u_map=uu_safe,
             v_map=vv_safe,
             bounds=bounds_safe,
             hull=hull_safe.astype(np.float32),
             hull_raw=sample_hull_xy,
             spot_radius_px=np.float32(spot_r))

    print(f"  → Saved {out_path}")
    print(f"  Grid points in map: {n_safe}")


def diagnose_uv_monotonicity(json_path: Path, u_axis_tolerance: float = 0.005):
    """
    Check for physically implausible samples: cases where sorting by u (or v)
    produces non-monotonic pixel centroid motion, which strongly suggests a
    mislabeled image or a filename/capture mismatch.

    Prints warnings for any violations found.
    """
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    samples = [s for s in data["samples"] if "centroid" in s]

    print("\n" + "=" * 72)
    print("  MONOTONICITY / CONSISTENCY DIAGNOSTIC")
    print("=" * 72)

    # ── 1. Check u-slices: for fixed v, does x decrease as u increases? ──
    # Group samples by v (within tolerance)
    from itertools import groupby

    def round_v(s): return round(s["v"] / u_axis_tolerance) * u_axis_tolerance

    by_v = {}
    for s in samples:
        key = round(s["v"] / u_axis_tolerance) * u_axis_tolerance
        by_v.setdefault(key, []).append(s)

    violations_found = 0
    for v_key, group in sorted(by_v.items()):
        if len(group) < 2:
            continue
        group_sorted = sorted(group, key=lambda s: s["u"])
        print(f"\n  v ≈ {v_key:+.3f}  ({len(group)} samples, sorted by u):")
        prev = None
        for s in group_sorted:
            cx, cy = s["centroid"]
            flag = ""
            if prev is not None:
                du = s["u"] - prev["u"]
                dcx = cx - prev["centroid"][0]
                # u more positive → beam moves left → cx should decrease
                # (flip sign if your mirror convention is opposite)
                if du > 0 and dcx > 10:
                    flag = "  ◄◄ SUSPICIOUS: u↑ but cx↑ too"
                    violations_found += 1
                elif du > 0 and dcx < -200:
                    flag = "  ◄◄ SUSPICIOUS: implausibly large jump"
                    violations_found += 1
                elif du < 0 and dcx < -10:
                    flag = "  ◄◄ SUSPICIOUS: u↓ but cx↓ too"
                    violations_found += 1
            print(f"    u={s['u']:+.4f}  cx={cx:7.1f}  cy={cy:7.1f}  {s['image']}{flag}")
            prev = s

    # ── 2. Duplicate centroid check ───────────────────────────────────────
    print(f"\n  Duplicate / near-duplicate centroid check (within 5px):")
    dup_found = False
    for i, a in enumerate(samples):
        for j, b in enumerate(samples):
            if j <= i:
                continue
            dist = math.hypot(a["centroid"][0] - b["centroid"][0],
                              a["centroid"][1] - b["centroid"][1])
            if dist < 5.0 and (a["u"] != b["u"] or a["v"] != b["v"]):
                print(f"    ◄◄ NEAR-DUPLICATE centroids ({dist:.1f}px apart):")
                print(f"       {a['image']}  u={a['u']:+.4f} v={a['v']:+.4f}  → {a['centroid']}")
                print(f"       {b['image']}  u={b['u']:+.4f} v={b['v']:+.4f}  → {b['centroid']}")
                dup_found = True
    if not dup_found:
        print("    No near-duplicate centroids found.")

    # ── 3. Neighbor distance outliers ─────────────────────────────────────
    print(f"\n  Neighbor-distance outlier check:")
    print(f"  (for each sample, expected px-per-unit-u/v from global linear fit)")
    uv_arr = np.array([[s["u"], s["v"]] for s in samples])
    xy_arr = np.array([s["centroid"]     for s in samples])
    # simple least-squares affine fit: [cx, cy] = A @ [u, v, 1]
    A_mat = np.column_stack([uv_arr, np.ones(len(samples))])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, xy_arr, rcond=None)
    pred_linear = A_mat @ coeffs
    linear_errs = np.linalg.norm(xy_arr - pred_linear, axis=1)
    median_lin  = float(np.median(linear_errs))
    print(f"  Linear-fit median residual: {median_lin:.1f}px")
    for s, err in sorted(zip(samples, linear_errs), key=lambda x: -x[1]):
        flag = "  ◄◄ HIGH" if err > 3 * median_lin else ""
        if err > 2 * median_lin or flag:
            print(f"    {s['image']:<42}  linear_err={err:6.1f}px{flag}")

    print("\n" + "=" * 72)
    if violations_found == 0:
        print("  No monotonicity violations found.")
    else:
        print(f"  {violations_found} monotonicity violation(s) found — review flagged images above.")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Laser calibration tool: label centroids and build mirror map")
    ap.add_argument("--images",      type=Path, default=None,
                    help="Directory containing .raw (or .png) calibration images")
    ap.add_argument("--json",        type=Path, default=Path("mirror_coordinates.json"),
                    help="Path to mirror_coordinates.json (created/updated in-place)")
    ap.add_argument("--map-out",     type=Path, default=Path("mirror_map.npz"),
                    help="Output path for mirror_map.npz  (default: mirror_map.npz)")
    ap.add_argument("--build-map",   action="store_true",
                    help="Build mirror_map.npz after labeling")
    ap.add_argument("--skip-label",  action="store_true",
                    help="Skip interactive labeling (only build map from existing JSON)")
    ap.add_argument("--raw-width",   type=int, default=None,
                    help="Width of .raw images in pixels (auto-detected if omitted)")
    ap.add_argument("--raw-height",  type=int, default=None,
                    help="Height of .raw images in pixels (auto-detected if omitted)")
    ap.add_argument("--grid-res",    type=int, default=200,
                    help="Dense grid resolution for inverse map (default 200)")

    # ── error review ──────────────────────────────────────────────────────
    ap.add_argument("--review-errors", action="store_true",
                    help=(
                        "Compute per-sample TPS reprojection errors, print a ranked "
                        "report, then open the interactive labeler for any image whose "
                        "error exceeds --error-threshold.  Requires --images and --json."
                    ))
    ap.add_argument("--error-threshold", type=float, default=None,
                    metavar="MULTIPLIER",
                    help=(
                        "Flag samples whose error >= MULTIPLIER × median error. "
                        "Pass a raw pixel value with --error-threshold-px instead. "
                        "Default: 2.0 × median."
                    ))
    ap.add_argument("--error-threshold-px", type=float, default=None,
                    metavar="PIXELS",
                    help=(
                        "Flag samples whose error >= this absolute pixel value. "
                        "Overrides --error-threshold if both are given."
                    ))
    ap.add_argument("--report-only", action="store_true",
                    help=(
                        "With --review-errors: print the error report but do NOT "
                        "open the interactive labeler (useful for a quick audit)."
                    ))

    args = ap.parse_args()

    # ── validate ──────────────────────────────────────────────────────────
    if args.review_errors and not args.json.exists():
        ap.error(f"--review-errors requires an existing JSON file: {args.json}")

    # ── error review mode ─────────────────────────────────────────────────
    if args.review_errors:
        print(f"\nComputing per-sample reprojection errors from {args.json} …")
        errors = compute_per_sample_errors(args.json)

        # determine threshold in pixels
        all_errs = [r["error_px"] for r in errors]
        median_err = float(np.median(all_errs))

        if args.error_threshold_px is not None:
            threshold_px = args.error_threshold_px
        elif args.error_threshold is not None:
            threshold_px = args.error_threshold * median_err
        else:
            threshold_px = 2.0 * median_err   # sensible default

        print_error_report(errors, threshold_px=threshold_px,
                           images_dir=args.images)

        if not args.report_only:
            outlier_names = [r["image"] for r in errors if r["error_px"] >= threshold_px]
            if not outlier_names:
                print("No outliers to review — nothing to re-label.")
            else:
                if args.images is None:
                    ap.error("--images is required to open the interactive labeler")
                if not args.images.is_dir():
                    ap.error(f"--images path is not a directory: {args.images}")

                print(f"Opening labeler for {len(outlier_names)} high-error image(s) …\n")
                error_lookup = {r["image"]: r for r in errors}
                labeler = CentroidLabeler(
                    args.images, args.json,
                    raw_width=args.raw_width,
                    raw_height=args.raw_height,
                    filter_names=outlier_names,
                )
                labeler.run(error_lookup=error_lookup)

        # fall through to --build-map if requested
        if args.build_map:
            print(f"\nRebuilding mirror map from {args.json} …")
            build_mirror_map(args.json, args.map_out, grid_res=args.grid_res)
            print("Done.")
        return

    # ── normal label mode ─────────────────────────────────────────────────
    if not args.skip_label:
        if args.images is None:
            ap.error("--images is required unless --skip-label is set")
        if not args.images.is_dir():
            ap.error(f"--images path is not a directory: {args.images}")
        labeler = CentroidLabeler(
            args.images, args.json,
            raw_width=args.raw_width, raw_height=args.raw_height
        )
        labeler.run()

    # --- build map ---
    if args.build_map:
        if not args.json.exists():
            print(f"[ERROR] JSON not found: {args.json}")
            sys.exit(1)
        print(f"\nBuilding mirror map from {args.json} ...")
        build_mirror_map(args.json, args.map_out, grid_res=args.grid_res)
        print("Done.")


if __name__ == "__main__":
    main()