"""
field_calibration.py
=====================
Autonomous field recalibration for laser/mirror systems.

Fires the laser at a known 9-point grid, images the spot via the Arena SDK
camera, extracts the centroid of each spot, then compares measured centroids
to the existing mirror_map.npz.  If drift is detected it applies a global
affine correction (translation + scale + rotation) to the map in-place and
writes a new corrected .npz file.

USAGE
-----
  # Full auto: fire, image, compare, correct, save
  python field_calibration.py

  # Dry-run: compute correction but don't overwrite map
  python field_calibration.py --dry-run

  # Use a custom grid (space-separated "u,v" pairs)
  python field_calibration.py --grid "-0.15,0.15 0,0.15 0.15,0.15 -0.15,0 0,0 0.15,0 -0.15,-0.15 0,-0.15 0.15,-0.15"

  # Point at a different map file
  python field_calibration.py --map mirror_map.npz --out mirror_map_corrected.npz

  # Skip firing (use previously saved images in --capture-dir)
  python field_calibration.py --skip-capture --capture-dir ./calib_frames

  # Save captured frames for inspection / archival
  python field_calibration.py --capture-dir ./calib_frames

  # Verbose mode: print per-point residuals and show overlay images
  python field_calibration.py --verbose

DEPENDENCIES
------------
  pip install numpy scipy opencv-python pyserial
  Arena SDK Python bindings (arena_api) must be on PYTHONPATH.

HARDWARE CONNECTIONS
--------------------
  Camera  : Lucid Vision via Arena SDK (GigE or USB3)
  Mirror  : MRE-3 via USB serial  (mirror_planner.py convention)
  Laser   : Radxa X4 MicroPython MCU via USB serial
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Default 9-point calibration grid  (u, v) #TODO expand grid, figure out how many points strike balance
# ---------------------------------------------------------------------------
DEFAULT_GRID: list[tuple[float, float]] = [
    (-0.10,  0.10), ( 0.00,  0.10), ( 0.10,  0.10),
    (-0.10,  0.00), ( 0.00,  0.00), ( 0.10,  0.00),
    (-0.10, -0.10), ( 0.00, -0.10), ( 0.10, -0.10),
]

# ---------------------------------------------------------------------------
# Camera interface  (Arena SDK)
# ---------------------------------------------------------------------------

class ArenaCamera:
    """
    Thin wrapper around the Lucid Vision Arena SDK for single-frame capture.
    Exposure and gain are set at construction time; call .grab() to acquire.

    If arena_api is not importable the class falls back to a synthetic test
    mode that generates a Gaussian spot at a plausible pixel location so that
    the rest of the pipeline can be validated without hardware.
    """

    def __init__(self, exposure_us: float = 5000.0, gain_db: float = 0.0):
        self._synthetic = False
        try:
            from arena_api.system import system as _arena_system
            from arena_api.__future__.save import Writer  # noqa: F401 – probe import

            devices = _arena_system.create_device()
            if not devices:
                raise RuntimeError("No Arena devices found")
            self._dev = devices[0]
            nodemap = self._dev.nodemap

            nodemap["AcquisitionMode"].value      = "SingleFrame"
            nodemap["PixelFormat"].value          = "Mono8"
            nodemap["ExposureAuto"].value         = "Off"
            nodemap["ExposureTime"].value         = float(exposure_us)
            nodemap["GainAuto"].value             = "Off"
            nodemap["Gain"].value                 = float(gain_db)

            self._dev.start_stream(1)
            print(f"[Camera] Arena device opened  "
                  f"(exposure={exposure_us}µs, gain={gain_db}dB)")
        except Exception as exc:
            print(f"[Camera] Arena SDK unavailable ({exc}) — running in SYNTHETIC mode")
            self._synthetic = True
            self._synthetic_uv: tuple[float, float] | None = None  # set by hint()

    def hint(self, u: float, v: float) -> None:
        """Tell synthetic mode where the mirror is pointing (for fake spot)."""
        if self._synthetic:
            self._synthetic_uv = (u, v)

    def grab(self) -> np.ndarray:
        """Return a Mono8 numpy array (H × W, uint8)."""
        if self._synthetic:
            return self._make_synthetic_frame()

        from arena_api.system import system as _arena_system  # noqa: F401
        buf = self._dev.get_buffer()
        arr = np.frombuffer(buf.data, dtype=np.uint8).reshape(
            buf.height, buf.width
        ).copy()
        self._dev.requeue_buffer(buf)
        return arr

    def _make_synthetic_frame(self) -> np.ndarray:
        """Generate a 2048×1536 frame with a Gaussian blob at a fake location."""
        H, W = 1536, 2048
        img = np.zeros((H, W), dtype=np.uint8)
        u, v = self._synthetic_uv if self._synthetic_uv else (0.0, 0.0)
        # Simple linear mapping for synthetic mode: u → x, v → y (inverted)
        cx = int(W / 2 + u * 800)
        cy = int(H / 2 - v * 600)
        # draw Gaussian spot (radius ~30px)
        yy, xx = np.ogrid[:H, :W]
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 25**2))
        img = (blob * 230).astype(np.uint8)
        # add small noise
        noise = np.random.randint(0, 8, (H, W), dtype=np.uint8)
        return cv2.add(img, noise)

    def close(self) -> None:
        if not self._synthetic:
            try:
                self._dev.stop_stream()
                from arena_api.system import system as _arena_system
                _arena_system.destroy_device()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Laser interface (re-implemented inline to avoid external import dependency)
# ---------------------------------------------------------------------------

class LaserInterface:
    """
    Communicates with the MicroPython laser MCU over USB serial.
    Sends 'FIRE\\n' and waits for the MCU acknowledgment.
    """

    def __init__(
        self,
        port: str = "/dev/serial/by-id/usb-MicroPython_Board_in_FS_mode_e6641cb2cf799623-if00",
        baud: int = 115200,
        settle_s: float = 8.0,
    ):
        self._synthetic = False
        try:
            import serial as _serial
            self.ser = _serial.Serial(port, baud, timeout=1)
            time.sleep(settle_s)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            print(f"[Laser]  MCU connected on {port}")
        except Exception as exc:
            print(f"[Laser]  Serial unavailable ({exc}) — SYNTHETIC mode")
            self._synthetic = True
            self.ser = None

    def fire(self, pulse_s: float = 0.26) -> None:
        if self._synthetic:
            time.sleep(0.05)
            return
        self.ser.write(b"FIRE\n")
        self.ser.flush()
        time.sleep(pulse_s)
        try:
            resp = self.ser.readline().decode().strip()
            if resp:
                print(f"         MCU: {resp}")
        except Exception:
            pass

    def off(self) -> None:
        if self._synthetic or self.ser is None:
            return
        try:
            self.ser.write(b"OFF\n")
            self.ser.flush()
        except Exception:
            pass

    def close(self) -> None:
        self.off()
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Mirror interface (minimal — only what calibration needs)
# ---------------------------------------------------------------------------

class MirrorInterface:
    """
    Sends XY commands to the MRE-3 galvo mirror via USB serial.
    Mirrors mirror_planner.MirrorPlanner.send_uv() settle logic.
    """

    _SETTLE_DEG_SMALL = 0.1
    _SETTLE_MS_SMALL  = 2.0
    _SETTLE_DEG_LARGE = 10.0
    _SETTLE_MS_LARGE  = 8.0
    _DEG_PER_UNIT     = 25.0
    _SETTLE_FLOOR_MS  = 0.5

    def __init__(
        self,
        port: str = "/dev/serial/by-id/usb-Optotune_Virtual_ComPort_3578335B3233-if00",
        baud: int = 115200,
    ):
        self._current = (0.0, 0.0)
        self._synthetic = False
        try:
            import serial as _serial
            self.ser = _serial.Serial(port, baud, timeout=0.01)
            print(f"[Mirror] MRE-3 connected on {port}")
        except Exception as exc:
            print(f"[Mirror] Serial unavailable ({exc}) — SYNTHETIC mode")
            self._synthetic = True
            self.ser = None

    def _settle_s(self, u: float, v: float) -> float:
        slope = (self._SETTLE_MS_LARGE - self._SETTLE_MS_SMALL) / (
            self._SETTLE_DEG_LARGE - self._SETTLE_DEG_SMALL
        )
        du = u - self._current[0]
        dv = v - self._current[1]
        deg = float(np.hypot(du, dv)) * self._DEG_PER_UNIT
        deg = np.clip(deg, self._SETTLE_DEG_SMALL, self._SETTLE_DEG_LARGE)
        ms = self._SETTLE_MS_SMALL + slope * (deg - self._SETTLE_DEG_SMALL)
        return max(ms, self._SETTLE_FLOOR_MS) / 1000.0

    def send(self, u: float, v: float) -> None:
        settle = self._settle_s(u, v)
        if not self._synthetic and self.ser is not None:
            self.ser.write(f"XY={u:.3f};{v:.3f}\r\n".encode())
            if self.ser.in_waiting:
                self.ser.read(self.ser.in_waiting)
        self._current = (u, v)
        time.sleep(settle)

    def center(self) -> None:
        self.send(0.0, 0.0)

    def close(self) -> None:
        self.center()
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Centroid extraction
# ---------------------------------------------------------------------------

def extract_centroid(
    img: np.ndarray,
    threshold_pct: float = 0.4,
    min_area_px: int = 10,
    max_area_px: int = 50_000,
) -> tuple[float, float] | None:
    """
    Find the laser spot centroid in a Mono8 image.

    Strategy
    --------
    1. Threshold at (threshold_pct × max_pixel_value) to isolate the bright spot.
    2. Find the single largest connected component above min_area_px.
    3. Return its intensity-weighted centroid (image moments).

    Returns (cx, cy) in pixel coordinates, or None if no spot found.
    """
    if img is None or img.size == 0:
        return None

    # normalise dynamic range then threshold
    peak = int(img.max())
    if peak < 10:
        return None  # image is effectively dark

    thresh_val = int(peak * threshold_pct)
    _, binary = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)

    # connected components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    best_label = -1
    best_area  = 0
    for lbl in range(1, n_labels):  # skip background (0)
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if min_area_px <= area <= max_area_px and area > best_area:
            best_area  = area
            best_label = lbl

    if best_label < 0:
        return None

    # intensity-weighted moments within the component mask
    mask = (labels == best_label).astype(np.uint8)
    region = img.astype(np.float32) * mask
    M = cv2.moments(region)
    if M["m00"] < 1e-6:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return round(cx, 1), round(cy, 1)


# ---------------------------------------------------------------------------
# Map lookup: expected centroid from mirror_map for a given (u, v)
# ---------------------------------------------------------------------------

def expected_centroid_from_map(
    u: float,
    v: float,
    u_map: np.ndarray,
    v_map: np.ndarray,
    x_map: np.ndarray,
    y_map: np.ndarray,
) -> tuple[float, float] | None:
    """Find the pre-calibrated centroid for (u, v) via nearest-neighbour lookup."""
    uv_arr = np.column_stack([u_map, v_map])
    tree   = cKDTree(uv_arr)
    _, idx = tree.query([u, v])
    return float(x_map[idx]), float(y_map[idx])


# ---------------------------------------------------------------------------
# Affine correction  (least-squares fit)
# ---------------------------------------------------------------------------

def fit_affine_correction(
    measured: np.ndarray,  # (N, 2)  measured centroids
    expected: np.ndarray,  # (N, 2)  expected centroids from map
) -> np.ndarray:
    """
    Fit a 2-D affine transform  T  such that  T @ [x, y, 1]ᵀ ≈ [x_m, y_m]ᵀ.

    In other words: map_corrected = T(map_expected) → measured_world.

    Returns a (2, 3) matrix  [[a, b, tx], [c, d, ty]].
    """
    N = len(measured)
    if N < 3:
        raise ValueError(f"Need ≥ 3 point pairs for affine fit, got {N}")

    # build system: for each point  [x_e, y_e, 1] @ A = [x_m, y_m]
    A = np.column_stack([expected, np.ones(N)])   # (N, 3)
    B = measured                                   # (N, 2)

    # least-squares solution
    result, _, _, _ = np.linalg.lstsq(A, B, rcond=None)  # (3, 2)
    T = result.T  # (2, 3):  row 0 = [a, b, tx],  row 1 = [c, d, ty]
    return T


def apply_affine_to_map(
    x_map: np.ndarray,
    y_map: np.ndarray,
    T: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a (2, 3) affine transform to every point in the map grid.

    Returns corrected (x_map_new, y_map_new).
    """
    pts = np.column_stack([x_map, y_map, np.ones(len(x_map))])  # (N, 3)
    corrected = pts @ T.T                                         # (N, 2)
    return corrected[:, 0].astype(np.float32), corrected[:, 1].astype(np.float32)


def apply_affine_to_hull(hull: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply affine T to convex hull vertices."""
    n = len(hull)
    pts = np.column_stack([hull, np.ones(n)])
    return (pts @ T.T).astype(np.float32)


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def print_calibration_report(
    grid: list[tuple[float, float]],
    measured: list[tuple[float, float] | None],
    expected: list[tuple[float, float] | None],
    T: np.ndarray | None,
    residuals_before: list[float],
    residuals_after: list[float] | None,
):
    print("\n" + "=" * 76)
    print("  FIELD CALIBRATION REPORT")
    print("=" * 76)
    print(f"  {'#':>2}  {'u':>7} {'v':>7}   "
          f"{'Meas cx':>8} {'Meas cy':>8}   "
          f"{'Exp cx':>8} {'Exp cy':>8}   "
          f"{'Δ before':>9}  {'Δ after':>9}")
    print("-" * 76)

    valid = []
    for i, (uv, meas, exp, db, da) in enumerate(
        zip(grid, measured, expected, residuals_before,
            residuals_after if residuals_after else [None] * len(grid))
    ):
        u, v = uv
        if meas is None or exp is None:
            print(f"  {i+1:>2}  {u:>+7.3f} {v:>+7.3f}   {'MISSED':>8}")
            continue
        after_str = f"{da:>9.2f}" if da is not None else "       —"
        print(f"  {i+1:>2}  {u:>+7.3f} {v:>+7.3f}   "
              f"{meas[0]:>8.1f} {meas[1]:>8.1f}   "
              f"{exp[0]:>8.1f} {exp[1]:>8.1f}   "
              f"{db:>9.2f}  {after_str}")
        valid.append(db)

    print("=" * 76)

    if valid:
        print(f"\n  Before correction:  mean={np.mean(valid):.2f}px  "
              f"max={np.max(valid):.2f}px  RMS={np.sqrt(np.mean(np.array(valid)**2)):.2f}px")

    if residuals_after:
        after_vals = [a for a in residuals_after if a is not None]
        if after_vals:
            print(f"  After  correction:  mean={np.mean(after_vals):.2f}px  "
                  f"max={np.max(after_vals):.2f}px  RMS={np.sqrt(np.mean(np.array(after_vals)**2)):.2f}px")

    if T is not None:
        print(f"\n  Affine matrix T (2×3):")
        print(f"    [{T[0,0]:+.6f}  {T[0,1]:+.6f}  {T[0,2]:+.2f}]")
        print(f"    [{T[1,0]:+.6f}  {T[1,1]:+.6f}  {T[1,2]:+.2f}]")
        tx, ty = T[0, 2], T[1, 2]
        print(f"\n  Translation:  Δx={tx:+.1f}px  Δy={ty:+.1f}px")
        scale_x = np.hypot(T[0, 0], T[1, 0])
        scale_y = np.hypot(T[0, 1], T[1, 1])
        print(f"  Scale:        sx={scale_x:.4f}  sy={scale_y:.4f}")
    print()


def save_debug_image(
    img: np.ndarray,
    measured: tuple[float, float] | None,
    expected: tuple[float, float] | None,
    out_path: Path,
):
    """Save annotated debug image for a single calibration point."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if expected is not None:
        ex, ey = int(expected[0]), int(expected[1])
        cv2.drawMarker(vis, (ex, ey), (0, 165, 255), cv2.MARKER_TILTED_CROSS, 30, 2)
        cv2.putText(vis, "expected", (ex + 16, ey - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    if measured is not None:
        mx, my = int(measured[0]), int(measured[1])
        cv2.drawMarker(vis, (mx, my), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
        cv2.putText(vis, "measured", (mx + 16, my + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if measured is not None and expected is not None:
        cv2.line(vis,
                 (int(expected[0]), int(expected[1])),
                 (int(measured[0]), int(measured[1])),
                 (0, 100, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)


# ---------------------------------------------------------------------------
# Core calibration routine
# ---------------------------------------------------------------------------

def run_calibration(
    grid: list[tuple[float, float]],
    map_file: Path,
    out_file: Path,
    capture_dir: Path | None,
    skip_capture: bool,
    dry_run: bool,
    verbose: bool,
    laser_port: str,
    mirror_port: str,
    exposure_us: float,
    gain_db: float,
    laser_pulse_s: float,
    settle_extra_s: float,
) -> bool:
    """
    Main calibration pipeline.

    Returns True if map was successfully corrected (or dry-run completed),
    False if too few points could be measured.
    """

    # ── load mirror map ────────────────────────────────────────────────────
    print(f"\n[Map]    Loading {map_file} …")
    data = np.load(map_file)
    x_map   = data["x_map"].copy()
    y_map   = data["y_map"].copy()
    u_map   = data["u_map"].copy()
    v_map   = data["v_map"].copy()
    uv_orig = data["uv"].copy()
    xy_orig = data["xy"].copy()
    hull    = data["hull"].copy()

    # optional fields (may not be in older maps)
    hull_raw       = data["hull_raw"].copy()        if "hull_raw"       in data else hull.copy()
    spot_radius_px = float(data["spot_radius_px"])  if "spot_radius_px" in data else 35.0
    uv_mean        = data["uv_mean"].copy()
    uv_std         = data["uv_std"].copy()
    bounds         = data["bounds"].copy()

    # ── optional capture directory ─────────────────────────────────────────
    if capture_dir is not None:
        capture_dir.mkdir(parents=True, exist_ok=True)

    # ── initialise hardware (or synthetics) ───────────────────────────────
    laser  = LaserInterface(port=laser_port)
    mirror = MirrorInterface(port=mirror_port)
    camera = ArenaCamera(exposure_us=exposure_us, gain_db=gain_db)

    measured_list: list[tuple[float, float] | None] = []
    expected_list: list[tuple[float, float] | None] = []
    frames: dict[int, np.ndarray] = {}

    print(f"\n[Calib]  Starting {len(grid)}-point grid capture …\n")

    try:
        for i, (u, v) in enumerate(grid):
            tag = f"u{u:+.3f}_v{v:+.3f}".replace(".", "p")
            print(f"  [{i+1:>2}/{len(grid)}]  u={u:+.4f}  v={v:+.4f} ", end="", flush=True)

            # ── load from disk if skip_capture ─────────────────────────────
            if skip_capture and capture_dir is not None:
                img_path = capture_dir / f"{tag}.png"
                if img_path.exists():
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"[WARN] Could not read {img_path} — skipping")
                        measured_list.append(None)
                        expected_list.append(None)
                        continue
                    print(f"(loaded from disk) ", end="", flush=True)
                else:
                    print(f"[WARN] {img_path} not found — skipping")
                    measured_list.append(None)
                    expected_list.append(None)
                    continue
            else:
                # ── move mirror ────────────────────────────────────────────
                mirror.send(u, v)
                if settle_extra_s > 0:
                    time.sleep(settle_extra_s)

                # ── fire laser + capture ───────────────────────────────────
                camera.hint(u, v)
                laser.fire(pulse_s=laser_pulse_s)

                img = camera.grab()

                # ── optionally save frame ──────────────────────────────────
                if capture_dir is not None:
                    cv2.imwrite(str(capture_dir / f"{tag}.png"), img)

            frames[i] = img

            # ── centroid extraction ────────────────────────────────────────
            centroid = extract_centroid(img)
            if centroid is None:
                print("→ NO SPOT DETECTED")
                measured_list.append(None)
                expected_list.append(None)
                continue

            # ── expected centroid from current map ─────────────────────────
            exp = expected_centroid_from_map(u, v, u_map, v_map, x_map, y_map)
            err_px = float(np.hypot(centroid[0] - exp[0], centroid[1] - exp[1]))

            print(f"→ meas=({centroid[0]:.1f}, {centroid[1]:.1f})  "
                  f"exp=({exp[0]:.1f}, {exp[1]:.1f})  "
                  f"Δ={err_px:.1f}px")

            measured_list.append(centroid)
            expected_list.append(exp)

    finally:
        laser.off()
        mirror.center()
        laser.close()
        mirror.close()
        camera.close()

    # ── collect valid pairs ────────────────────────────────────────────────
    valid_idx = [i for i, (m, e) in enumerate(zip(measured_list, expected_list))
                 if m is not None and e is not None]

    n_valid = len(valid_idx)
    print(f"\n[Calib]  {n_valid}/{len(grid)} points successfully measured.")

    if n_valid < 3:
        print(f"[ERROR]  Need ≥ 3 valid points for affine fit — only got {n_valid}.")
        print("         Check laser power, camera exposure, and mirror connectivity.")
        return False

    meas_pts = np.array([measured_list[i] for i in valid_idx], dtype=np.float64)
    exp_pts  = np.array([expected_list[i]  for i in valid_idx], dtype=np.float64)

    # ── residuals before correction ────────────────────────────────────────
    residuals_before = [
        float(np.hypot(measured_list[i][0] - expected_list[i][0],
                        measured_list[i][1] - expected_list[i][1]))
        if measured_list[i] is not None and expected_list[i] is not None else 0.0
        for i in range(len(grid))
    ]

    # ── fit affine correction ──────────────────────────────────────────────
    T = fit_affine_correction(meas_pts, exp_pts)

    # ── apply to map grid ──────────────────────────────────────────────────
    x_map_new, y_map_new = apply_affine_to_map(x_map, y_map, T)
    hull_new             = apply_affine_to_hull(hull, T)
    hull_raw_new         = apply_affine_to_hull(hull_raw, T)

    # also update the raw calibration sample centroids stored in the map
    xy_new = np.column_stack(
        apply_affine_to_map(xy_orig[:, 0], xy_orig[:, 1], T)
    ).astype(np.float32)

    # ── residuals after correction ─────────────────────────────────────────
    # Re-query the corrected map for each measured point
    new_tree = cKDTree(np.column_stack([u_map, v_map]))
    residuals_after: list[float | None] = []
    for i in range(len(grid)):
        if measured_list[i] is None or expected_list[i] is None:
            residuals_after.append(None)
            continue
        u_i, v_i   = grid[i]
        _, idx      = new_tree.query([u_i, v_i])
        new_exp     = (float(x_map_new[idx]), float(y_map_new[idx]))
        res_after   = float(np.hypot(measured_list[i][0] - new_exp[0],
                                     measured_list[i][1] - new_exp[1]))
        residuals_after.append(res_after)

    # ── report ─────────────────────────────────────────────────────────────
    print_calibration_report(
        grid, measured_list, expected_list, T,
        residuals_before, residuals_after
    )

    # ── debug images ───────────────────────────────────────────────────────
    if verbose and capture_dir is not None:
        for i in valid_idx:
            tag = f"u{grid[i][0]:+.3f}_v{grid[i][1]:+.3f}".replace(".", "p")
            save_debug_image(
                frames[i],
                measured_list[i],
                expected_list[i],
                capture_dir / f"{tag}_annotated.png"
            )
        print(f"[Debug]  Annotated images saved to {capture_dir}")

    # ── save corrected map ─────────────────────────────────────────────────
    if dry_run:
        print("[Dry-run] Map NOT saved — remove --dry-run to apply correction.")
        return True

    # Back up original
    if out_file == map_file:
        backup = map_file.with_suffix(".bak.npz")
        shutil.copy2(map_file, backup)
        print(f"[Map]    Original backed up to {backup}")

    np.savez(
        out_file,
        uv=uv_orig,
        xy=xy_new,
        uv_mean=uv_mean,
        uv_std=uv_std,
        x_map=x_map_new,
        y_map=y_map_new,
        u_map=u_map,
        v_map=v_map,
        bounds=bounds,
        hull=hull_new.astype(np.float32),
        hull_raw=hull_raw_new.astype(np.float32),
        spot_radius_px=np.float32(spot_radius_px),
    )
    print(f"[Map]    Corrected map saved → {out_file}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_grid(grid_str: str) -> list[tuple[float, float]]:
    pts = []
    for token in grid_str.strip().split():
        parts = token.split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"Grid point must be 'u,v' — got: {token!r}")
        pts.append((float(parts[0]), float(parts[1])))
    return pts


def main():
    ap = argparse.ArgumentParser(
        description="Autonomous field calibration: fire-image-compare-correct mirror map",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── I/O ───────────────────────────────────────────────────────────────
    ap.add_argument("--map",     type=Path, default=Path("mirror_map.npz"),
                    help="Input mirror_map.npz  (default: mirror_map.npz)")
    ap.add_argument("--out",     type=Path, default=None,
                    help="Output path for corrected map  (default: overwrite --map)")

    # ── grid ──────────────────────────────────────────────────────────────
    ap.add_argument("--grid", type=str, default=None,
                    help='Space-separated "u,v" pairs e.g. "-0.1,0.1 0,0 0.1,-0.1"')

    # ── hardware ports ────────────────────────────────────────────────────
    ap.add_argument("--laser-port",  type=str,
                    default="/dev/serial/by-id/usb-MicroPython_Board_in_FS_mode_e6641cb2cf799623-if00",
                    help="Laser MCU serial port")
    ap.add_argument("--mirror-port", type=str,
                    default="/dev/serial/by-id/usb-Optotune_Virtual_ComPort_3578335B3233-if00",
                    help="MRE-3 mirror serial port")

    # ── camera ────────────────────────────────────────────────────────────
    ap.add_argument("--exposure",    type=float, default=5000.0,
                    help="Camera exposure in µs  (default: 5000)")
    ap.add_argument("--gain",        type=float, default=0.0,
                    help="Camera gain in dB  (default: 0)")

    # ── timing ────────────────────────────────────────────────────────────
    ap.add_argument("--pulse",       type=float, default=0.26,
                    help="Laser pulse duration in seconds  (default: 0.26)")
    ap.add_argument("--settle",      type=float, default=0.0,
                    help="Extra settle delay after mirror move in seconds  (default: 0)")

    # ── workflow flags ────────────────────────────────────────────────────
    ap.add_argument("--skip-capture",  action="store_true",
                    help="Skip hardware capture — load frames from --capture-dir")
    ap.add_argument("--capture-dir",   type=Path, default=None,
                    help="Directory to save/load raw capture frames as PNGs")
    ap.add_argument("--dry-run",       action="store_true",
                    help="Compute correction but do not save corrected map")
    ap.add_argument("--verbose",       action="store_true",
                    help="Print per-point details and save annotated debug images")

    args = ap.parse_args()

    # ── resolve defaults ──────────────────────────────────────────────────
    out_file = args.out if args.out is not None else args.map
    grid     = parse_grid(args.grid) if args.grid else DEFAULT_GRID

    if not args.map.exists():
        ap.error(f"Map file not found: {args.map}")

    if args.skip_capture and args.capture_dir is None:
        ap.error("--skip-capture requires --capture-dir")

    print("=" * 60)
    print("  FIELD CALIBRATION")
    print("=" * 60)
    print(f"  Map in:       {args.map}")
    print(f"  Map out:      {out_file}")
    print(f"  Grid points:  {len(grid)}")
    print(f"  Dry run:      {args.dry_run}")
    print(f"  Capture dir:  {args.capture_dir or '(not saving)'}")
    print("=" * 60)

    success = run_calibration(
        grid          = grid,
        map_file      = args.map,
        out_file      = out_file,
        capture_dir   = args.capture_dir,
        skip_capture  = args.skip_capture,
        dry_run       = args.dry_run,
        verbose       = args.verbose,
        laser_port    = args.laser_port,
        mirror_port   = args.mirror_port,
        exposure_us   = args.exposure,
        gain_db       = args.gain,
        laser_pulse_s = args.pulse,
        settle_extra_s= args.settle,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()