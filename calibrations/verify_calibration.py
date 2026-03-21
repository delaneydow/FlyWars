"""
verify_calibration.py
======================
Visual verification tool for mirror_map.npz and mirror_coordinates.json.

FEATURES:
  1. Reachable FOV map  — dense grid of reachable (x,y) image coords, coloured
     by mirror command distance from center (precision contours), with spot-size
     circles showing the true laser footprint at each grid node.
  2. JSON sample scatter — all labelled calibration points overlaid on the FOV
     map; arrows show expected vs actual centroid to flag miscalibrated samples.
  3. Scale calibration  — load a ruler image, click two known-distance points,
     and the tool computes px/mm, spot diameter in mm, and updates the JSON.
  4. Residual heatmap   — TPS reprojection error across the mirror command space,
     so you can see where the mapping is least accurate.
  5. Per-sample inspector — click any scatter point to zoom in on the raw image
     and see the labelled centroid overlaid.

USAGE:
  python verify_calibration.py \
      --map mirror_map.npz \
      --json mirror_coordinates.json \
      --images ./calibration_images

  # With a ruler image for scale calibration:
  python verify_calibration.py \
      --map mirror_map.npz \
      --json mirror_coordinates.json \
      --images ./calibration_images \
      --ruler ruler_image.raw \
      --ruler-distance-mm 10.0

CONTROLS (FOV / scatter window):
  Click scatter point   : open per-sample inspector for that image
  R key                 : open ruler scale calibration window
  H key                 : toggle residual heatmap overlay
  S key                 : save annotated FOV figure to PNG
  Q / Escape            : quit
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # works headlessly too if TkAgg absent → Agg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import Rbf
from scipy.spatial import ConvexHull


# ── helpers ──────────────────────────────────────────────────────────────────

def load_raw(path: Path, width=None, height=None) -> np.ndarray:
    raw = path.read_bytes()
    n = len(raw)
    candidates = [
        (2048, 1536), (1920, 1080), (1280, 960),
        (1280, 720),  (640, 480),   (2592, 1944),
    ]
    dtype = np.uint8
    if width is None or height is None:
        for w, h in candidates:
            if n == w * h:
                width, height, dtype = w, h, np.uint8; break
            elif n == w * h * 2:
                width, height, dtype = w, h, np.uint16; break
        else:
            side = int(math.isqrt(n))
            width = height = side
    arr = np.frombuffer(raw, dtype=dtype).reshape((height, width))
    if arr.dtype != np.uint8:
        arr = (arr.astype(np.float32) / arr.max() * 255).astype(np.uint8)
    return arr


def load_image(path: Path, raw_w=None, raw_h=None) -> np.ndarray:
    if path.suffix.lower() == ".raw":
        return load_raw(path, raw_w, raw_h)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def find_image(name: str, images_dir: Path, raw_w=None, raw_h=None):
    """Try exact name, then swap extension to .raw / .png."""
    p = images_dir / name
    if p.exists():
        return load_image(p, raw_w, raw_h)
    stem = Path(name).stem
    for ext in (".raw", ".png", ".jpg", ".tiff"):
        p2 = images_dir / (stem + ext)
        if p2.exists():
            return load_image(p2, raw_w, raw_h)
    return None


# ── main verifier ─────────────────────────────────────────────────────────────

class CalibrationVerifier:

    def __init__(self, map_path, json_path, images_dir=None,
                 ruler_path=None, ruler_dist_mm=None, raw_w=None, raw_h=None):
        self.map_path      = Path(map_path)
        self.json_path     = Path(json_path)
        self.images_dir    = Path(images_dir) if images_dir else None
        self.ruler_path    = Path(ruler_path) if ruler_path else None
        self.ruler_dist_mm = ruler_dist_mm
        self.raw_w, self.raw_h = raw_w, raw_h

        self._load_map()
        self._load_json()
        self._build_tps()

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_map(self):
        d = np.load(self.map_path)
        self.uv       = d["uv"]
        self.xy       = d["xy"]
        self.uv_mean  = d["uv_mean"]
        self.uv_std   = d["uv_std"]
        self.x_map    = d["x_map"]
        self.y_map    = d["y_map"]
        self.u_map    = d["u_map"]
        self.v_map    = d["v_map"]
        self.bounds   = d["bounds"]          # u_min,u_max,v_min,v_max
        self.hull_pts = d["hull"]
        print(f"[map]  {len(self.uv)} calibration samples, "
              f"{len(self.x_map)} grid points")

    def _load_json(self):
        with open(self.json_path, "r", encoding="utf-8-sig") as f:
            self.jdata = json.load(f)
        self.samples = [s for s in self.jdata["samples"] if "centroid" in s]
        self.cam_res = self.jdata["camera"]["resolution"]   # [w, h]
        self.spot_r  = self.jdata["laser"].get("spot_radius_px", 35)
        self.px_per_mm = None   # filled by scale calibration
        print(f"[json] {len(self.samples)} labelled samples, "
              f"cam {self.cam_res[0]}×{self.cam_res[1]}, "
              f"spot_r={self.spot_r}px")

    def _build_tps(self):
        uvn = (self.uv - self.uv_mean) / self.uv_std
        u_n, v_n = uvn[:, 0], uvn[:, 1]
        x_c = self.xy[:, 0];  y_c = self.xy[:, 1]
        self.fx = Rbf(u_n, v_n, x_c, function="thin_plate", smooth=1e-2)
        self.fy = Rbf(u_n, v_n, y_c, function="thin_plate", smooth=1e-2)
        pred = np.column_stack([self.fx(u_n, v_n), self.fy(u_n, v_n)])
        self.residuals = np.sqrt(np.sum((pred - self.xy) ** 2, axis=1))
        self.rms = float(np.sqrt(np.mean(self.residuals ** 2)))
        print(f"[tps]  RMS reprojection error = {self.rms:.2f} px")

    # ── FOV + scatter plot ────────────────────────────────────────────────────

    def plot_fov(self):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor("#0f0f12")
        for ax in axes:
            ax.set_facecolor("#0f0f12")
            for sp in ax.spines.values():
                sp.set_color("#444")
            ax.tick_params(colors="#aaa")
            ax.xaxis.label.set_color("#aaa")
            ax.yaxis.label.set_color("#aaa")
            ax.title.set_color("#eee")

        self._draw_fov_panel(axes[0])
        self._draw_scatter_panel(axes[1])

        fig.suptitle("Mirror calibration verification", color="#eee",
                     fontsize=13, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # key bindings
        def on_key(event):
            if event.key in ("q", "escape"):
                plt.close("all")
            elif event.key == "s":
                out = self.json_path.parent / "calibration_verification.png"
                fig.savefig(str(out), dpi=150, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
                print(f"Saved → {out}")
            elif event.key == "r":
                plt.close(fig)
                self.run_ruler_calibration()
                self.plot_fov()
            elif event.key == "h":
                plt.close(fig)
                self.plot_residual_heatmap()

        fig.canvas.mpl_connect("key_press_event", on_key)

        # click on scatter to open inspector
        def on_click(event):
            if event.inaxes != axes[1] or event.button != 1:
                return
            xs = np.array([s["centroid"][0] for s in self.samples])
            ys = np.array([s["centroid"][1] for s in self.samples])
            dists = np.hypot(xs - event.xdata, ys - event.ydata)
            i = int(np.argmin(dists))
            if dists[i] < 60:
                self._open_inspector(self.samples[i])

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.text(0.5, 0.01,
                 "H=heatmap  R=ruler calibration  S=save  Q=quit  "
                 "click scatter point=inspect image",
                 ha="center", color="#666", fontsize=9)
        plt.show()

    def _draw_fov_panel(self, ax):
        """Left panel: reachable FOV with precision contours and spot circles."""
        cw, ch = self.cam_res
        ax.set_xlim(0, cw);  ax.set_ylim(ch, 0)
        ax.set_aspect("equal")
        ax.set_title("Reachable FOV — spot-size adjusted", fontsize=11)
        ax.set_xlabel("Camera X (px)");  ax.set_ylabel("Camera Y (px)")

        # grid background: distance from mirror center as colour
        dist_from_center = np.hypot(self.u_map, self.v_map)
        norm = Normalize(vmin=0, vmax=dist_from_center.max())
        cmap = plt.cm.plasma

        # draw spot circles (true laser footprint at each grid node)
        r = self.spot_r
        for xi, yi, d in zip(self.x_map, self.y_map, dist_from_center):
            c = cmap(norm(d))
            circ = mpatches.Circle((xi, yi), r, color=c, alpha=0.18,
                                   linewidth=0)
            ax.add_patch(circ)

        # scatter: grid centroids coloured by distance
        sc = ax.scatter(self.x_map, self.y_map,
                        c=dist_from_center, cmap="plasma",
                        s=4, alpha=0.5, linewidths=0)

        # contour lines at 0.05 UV intervals
        from scipy.interpolate import griddata
        xi = np.linspace(0, cw, 300)
        yi = np.linspace(0, ch, 300)
        XI, YI = np.meshgrid(xi, yi)
        ZI = griddata(
            np.column_stack([self.x_map, self.y_map]),
            dist_from_center,
            (XI, YI), method="linear"
        )
        cs = ax.contour(XI, YI, ZI,
                        levels=np.arange(0.05, dist_from_center.max(), 0.05),
                        colors="#ffffff", linewidths=0.4, alpha=0.3)
        ax.clabel(cs, fmt="%.2f", fontsize=7, colors="#ccc")

        # convex hull outline
        hull_closed = np.vstack([self.hull_pts, self.hull_pts[0]])
        ax.plot(hull_closed[:, 0], hull_closed[:, 1],
                color="#00e5ff", lw=1.2, ls="--", label="reachable boundary")

        # calibration sample positions
        xs = np.array([s["centroid"][0] for s in self.samples])
        ys = np.array([s["centroid"][1] for s in self.samples])
        ax.scatter(xs, ys, s=35, c="#ffffff", zorder=5,
                   edgecolors="#333", linewidths=0.5, label="cal samples")

        plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     label="UV distance from center",
                     fraction=0.03, pad=0.02)
        ax.legend(fontsize=8, facecolor="#1a1a20", edgecolor="#444",
                  labelcolor="#ccc")

        # scale bar if px/mm known
        if self.px_per_mm:
            bar_mm = 10
            bar_px = bar_mm * self.px_per_mm
            ax.annotate("", xy=(40 + bar_px, ch - 40), xytext=(40, ch - 40),
                        arrowprops=dict(arrowstyle="<->", color="#0f0", lw=1.2))
            ax.text(40 + bar_px / 2, ch - 55,
                    f"{bar_mm} mm", ha="center", color="#0f0", fontsize=8)

    def _draw_scatter_panel(self, ax):
        """Right panel: JSON samples, residual arrows, anomaly flags."""
        cw, ch = self.cam_res
        ax.set_xlim(0, cw);  ax.set_ylim(ch, 0)
        ax.set_aspect("equal")
        ax.set_title("Sample residuals  (arrow = TPS error)", fontsize=11)
        ax.set_xlabel("Camera X (px)");  ax.set_ylabel("Camera Y (px)")
        ax.set_facecolor("#0f0f12")

        # hull boundary
        hull_closed = np.vstack([self.hull_pts, self.hull_pts[0]])
        ax.fill(hull_closed[:, 0], hull_closed[:, 1],
                color="#1a2030", zorder=0)
        ax.plot(hull_closed[:, 0], hull_closed[:, 1],
                color="#00e5ff", lw=1, ls="--", alpha=0.5)

        uvn = (self.uv - self.uv_mean) / self.uv_std
        pred_x = self.fx(uvn[:, 0], uvn[:, 1])
        pred_y = self.fy(uvn[:, 0], uvn[:, 1])

        ERR_THRESH = self.spot_r * 0.5   # flag if error > half spot radius

        for i, s in enumerate(self.samples):
            cx, cy = s["centroid"]
            u, v = s["u"], s["v"]

            # find matching row in self.uv
            diffs = np.hypot(self.uv[:, 0] - u, self.uv[:, 1] - v)
            j = int(np.argmin(diffs))
            px, py = pred_x[j], pred_y[j]
            err = math.hypot(px - cx, py - cy)

            color = "#ff4040" if err > ERR_THRESH else "#40ff80"
            ax.scatter(cx, cy, s=30, c=color, zorder=4,
                       edgecolors="#000", linewidths=0.4)

            # residual arrow (scaled 5× for visibility)
            scale = 5.0
            dx, dy = (px - cx) * scale, (py - cy) * scale
            if abs(dx) + abs(dy) > 1:
                ax.annotate("",
                    xy=(cx + dx, cy + dy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->",
                                    color="#ffaa00", lw=0.8, alpha=0.7),
                    zorder=5)

            # label high-error points
            if err > ERR_THRESH:
                ax.text(cx + 8, cy - 8,
                        f"{s['image'][:12]}\n{err:.1f}px",
                        color="#ff6060", fontsize=6, zorder=6)

        # legend
        ok_patch  = mpatches.Patch(color="#40ff80", label=f"error ≤ {ERR_THRESH:.0f}px")
        bad_patch = mpatches.Patch(color="#ff4040", label=f"error > {ERR_THRESH:.0f}px")
        ax.legend(handles=[ok_patch, bad_patch], fontsize=8,
                  facecolor="#1a1a20", edgecolor="#444", labelcolor="#ccc")
        ax.text(10, 30, f"RMS = {self.rms:.2f} px", color="#ffdd88",
                fontsize=9, zorder=7)
        ax.text(10, 55,
                f"spot_r = {self.spot_r}px"
                + (f"  ({self.spot_r/self.px_per_mm:.1f}mm)"
                   if self.px_per_mm else ""),
                color="#88ddff", fontsize=9, zorder=7)

    # ── residual heatmap ──────────────────────────────────────────────────────

    def plot_residual_heatmap(self):
        from scipy.interpolate import griddata

        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor("#0f0f12")
        ax.set_facecolor("#0f0f12")
        ax.title.set_color("#eee")
        ax.set_title("TPS residual heatmap (mirror UV space)", fontsize=11)
        ax.set_xlabel("U", color="#aaa")
        ax.set_ylabel("V", color="#aaa")
        ax.tick_params(colors="#aaa")

        u_min, u_max, v_min, v_max = self.bounds
        ui = np.linspace(u_min, u_max, 80)
        vi = np.linspace(v_min, v_max, 80)
        UI, VI = np.meshgrid(ui, vi)

        ZI = griddata(self.uv, self.residuals, (UI, VI), method="cubic")
        im = ax.contourf(UI, VI, ZI, levels=20, cmap="inferno")
        plt.colorbar(im, ax=ax, label="reprojection error (px)",
                     fraction=0.04)

        ax.scatter(self.uv[:, 0], self.uv[:, 1],
                   c=self.residuals, cmap="inferno",
                   s=50, edgecolors="#fff", linewidths=0.5, zorder=5)
        for i, s in enumerate(self.samples):
            ax.text(self.uv[i, 0] + 0.003, self.uv[i, 1] + 0.003,
                    f"{self.residuals[i]:.1f}", color="#fff",
                    fontsize=6, zorder=6)

        def on_key(event):
            if event.key in ("q", "escape"):
                plt.close("all")
        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.text(0.5, 0.01, "Q=quit", ha="center", color="#666", fontsize=9)
        plt.tight_layout()
        plt.show()

    # ── per-sample inspector ──────────────────────────────────────────────────

    def _open_inspector(self, sample):
        if self.images_dir is None:
            print("[inspector] --images not provided, skipping")
            return
        img = find_image(sample["image"], self.images_dir,
                         self.raw_w, self.raw_h)
        if img is None:
            print(f"[inspector] image not found: {sample['image']}")
            return

        cx, cy = sample["centroid"]
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # draw spot circle
        cv2.circle(vis, (int(cx), int(cy)), self.spot_r,
                   (0, 255, 80), 2)
        # draw centroid cross
        cv2.drawMarker(vis, (int(cx), int(cy)), (0, 255, 80),
                       cv2.MARKER_CROSS, 30, 2)
        # label
        cv2.putText(vis,
                    f"u={sample['u']:+.3f} v={sample['v']:+.3f}  "
                    f"({cx:.1f},{cy:.1f})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 80), 2)

        # spot size info
        if self.px_per_mm:
            diam_mm = 2 * self.spot_r / self.px_per_mm
            cv2.putText(vis,
                        f"spot diameter = {diam_mm:.2f} mm",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (80, 200, 255), 2)

        h, w = vis.shape[:2]
        scale = min(1400 / w, 1050 / h, 1.0)
        small = cv2.resize(vis, (int(w * scale), int(h * scale)))
        win = f"Inspector: {sample['image']}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, small)
        print(f"[inspector] {sample['image']}  "
              f"centroid=({cx:.1f},{cy:.1f})  "
              f"u={sample['u']:+.3f} v={sample['v']:+.3f}")
        cv2.waitKey(0)
        cv2.destroyWindow(win)

    # ── ruler scale calibration ───────────────────────────────────────────────

    def run_ruler_calibration(self):
            """
            Zoomable ruler calibration window.
    
            Controls
            --------
            Scroll wheel          : zoom in / out around cursor
            Middle-click + drag   : pan when zoomed in
            Left-click            : place point 1, then point 2
            Right-click / U       : undo last placed point
            Z                     : zoom to fit (reset view)
            +  /  -               : zoom in / out by fixed step
            Q / Escape            : finish (saves result if 2 points placed)
    
            A 160×160 px magnifier inset follows the cursor so you can see
            individual pixels before committing a click.  The inset is
            rendered at 4× the current zoom so fine ruler graduations are
            always legible.
            """
            if self.ruler_path is None or self.ruler_dist_mm is None:
                print("[ruler] pass --ruler and --ruler-distance-mm to use this")
                return
    
            img = find_image(self.ruler_path.name,
                            self.ruler_path.parent,
                            self.raw_w, self.raw_h)
            if img is None:
                img = load_image(self.ruler_path, self.raw_w, self.raw_h)
    
            img_bgr  = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            ih, iw   = img_bgr.shape[:2]
    
            # ── display / viewport state ──────────────────────────────────────
            DISP_W, DISP_H = min(iw, 1400), min(ih, 1050)
            state = {
                "zoom":    1.0,
                "pan_x":  0,       # top-left corner of viewport in image coords
                "pan_y":  0,
                "pts":    [],      # list of (ix, iy) image-coord clicks
                "mx":     0,       # current mouse x in display coords
                "my":     0,       # current mouse y in display coords
                "drag":   False,
                "drag_sx": 0, "drag_sy": 0,   # display coords at drag start
                "drag_px": 0, "drag_py": 0,   # pan_x/y at drag start
                "result": None,    # filled when 2nd point placed
            }
    
            WIN = "Ruler calibration  [scroll=zoom | mid-drag=pan | click=point | U=undo | Z=fit | Q=done]"
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN, DISP_W, DISP_H)
    
            # ── coordinate helpers ────────────────────────────────────────────
    
            def viewport():
                """Return (x0,y0,vw,vh) crop rectangle in image coords."""
                z = state["zoom"]
                vw = int(iw / z);  vh = int(ih / z)
                x0 = int(np.clip(state["pan_x"], 0, max(0, iw - vw)))
                y0 = int(np.clip(state["pan_y"], 0, max(0, ih - vh)))
                return x0, y0, vw, vh
    
            def disp_to_img(dx, dy):
                x0, y0, vw, vh = viewport()
                ix = x0 + dx / DISP_W * vw
                iy = y0 + dy / DISP_H * vh
                return ix, iy
    
            def img_to_disp(ix, iy):
                x0, y0, vw, vh = viewport()
                dx = (ix - x0) / vw * DISP_W
                dy = (iy - y0) / vh * DISP_H
                return int(dx), int(dy)
    
            def zoom_toward(cx_d, cy_d, factor):
                ix, iy = disp_to_img(cx_d, cy_d)
                state["zoom"] = float(np.clip(state["zoom"] * factor, 1.0, 40.0))
                x0, y0, vw, vh = viewport()
                state["pan_x"] = int(ix - cx_d / DISP_W * vw)
                state["pan_y"] = int(iy - cy_d / DISP_H * vh)
    
            def reset_view():
                state["zoom"]  = 1.0
                state["pan_x"] = 0
                state["pan_y"] = 0
    
            # ── render ────────────────────────────────────────────────────────
    
            def render():
                x0, y0, vw, vh = viewport()
                crop = img_bgr[y0:y0+vh, x0:x0+vw]
                vis  = cv2.resize(crop, (DISP_W, DISP_H),
                                interpolation=cv2.INTER_LINEAR)
    
                # ── placed points ──
                colors = [(0, 255, 255), (255, 120, 0)]
                for k, (px, py) in enumerate(state["pts"]):
                    dx, dy = img_to_disp(px, py)
                    cv2.drawMarker(vis, (dx, dy), colors[k],
                                cv2.MARKER_CROSS, 28, 2)
                    cv2.circle(vis, (dx, dy), 10, colors[k], 1)
                    cv2.putText(vis, f"P{k+1} ({px:.1f},{py:.1f})",
                                (dx + 14, dy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[k], 1)
    
                # ── line between points ──
                if len(state["pts"]) == 2:
                    d0 = img_to_disp(*state["pts"][0])
                    d1 = img_to_disp(*state["pts"][1])
                    cv2.line(vis, d0, d1, (0, 200, 255), 1)
                    mx_d = (d0[0] + d1[0]) // 2
                    my_d = (d0[1] + d1[1]) // 2
                    if state["result"]:
                        cv2.putText(vis,
                                    f"{state['result']['px_dist']:.1f}px = "
                                    f"{self.ruler_dist_mm}mm → "
                                    f"{state['result']['px_per_mm']:.3f}px/mm",
                                    (mx_d - 120, my_d - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (0, 255, 80), 1)
    
                # ── HUD ──
                n_placed = len(state["pts"])
                if n_placed == 0:
                    msg = f"Click point 1 of 2  ({self.ruler_dist_mm}mm apart)"
                elif n_placed == 1:
                    msg = "Click point 2 of 2  |  U=undo"
                else:
                    msg = "2 points placed  |  U=undo  Q=save & close"
                cv2.putText(vis, msg, (10, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 50), 1)
                cv2.putText(vis,
                            f"zoom {state['zoom']:.1f}x   "
                            f"img ({int(disp_to_img(state['mx'],state['my'])[0])},"
                            f"{int(disp_to_img(state['mx'],state['my'])[1])})",
                            (10, DISP_H - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1)
    
                # ── magnifier inset (bottom-right) ──
                MAG_SIZE   = 160    # display pixels
                MAG_RADIUS = 40     # half-width in image pixels to sample
                mx, my = state["mx"], state["my"]
                ix, iy = disp_to_img(mx, my)
                ix, iy = int(np.clip(ix, MAG_RADIUS, iw - MAG_RADIUS)), \
                        int(np.clip(iy, MAG_RADIUS, ih - MAG_RADIUS))
                patch = img_bgr[iy-MAG_RADIUS:iy+MAG_RADIUS,
                                ix-MAG_RADIUS:ix+MAG_RADIUS]
                if patch.size > 0:
                    mag = cv2.resize(patch, (MAG_SIZE, MAG_SIZE),
                                    interpolation=cv2.INTER_NEAREST)
                    # crosshair on magnifier
                    c = MAG_SIZE // 2
                    cv2.line(mag, (c, 0), (c, MAG_SIZE), (0, 255, 255), 1)
                    cv2.line(mag, (0, c), (MAG_SIZE, c), (0, 255, 255), 1)
                    cv2.rectangle(mag, (0, 0), (MAG_SIZE-1, MAG_SIZE-1),
                                (80, 80, 80), 1)
                    # paste into bottom-right corner with small border
                    bx = DISP_W - MAG_SIZE - 8
                    by = DISP_H - MAG_SIZE - 8
                    vis[by:by+MAG_SIZE, bx:bx+MAG_SIZE] = mag
                    cv2.putText(vis, f"4× inset",
                                (bx, by - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100,100,100), 1)
    
                return vis
    
            # ── mouse callback ────────────────────────────────────────────────
    
            def mouse_cb(event, x, y, flags, param):
                state["mx"], state["my"] = x, y
    
                if event == cv2.EVENT_LBUTTONDOWN:
                    if len(state["pts"]) < 2:
                        ix, iy = disp_to_img(x, y)
                        state["pts"].append((round(ix, 2), round(iy, 2)))
                        if len(state["pts"]) == 2:
                            px_dist = math.hypot(
                                state["pts"][1][0] - state["pts"][0][0],
                                state["pts"][1][1] - state["pts"][0][1])
                            ppm = px_dist / self.ruler_dist_mm
                            state["result"] = {"px_dist": px_dist,
                                            "px_per_mm": ppm}
    
                elif event == cv2.EVENT_RBUTTONDOWN:
                    if state["pts"]:
                        state["pts"].pop()
                        state["result"] = None
    
                elif event == cv2.EVENT_MBUTTONDOWN:
                    state["drag"]   = True
                    state["drag_sx"], state["drag_sy"] = x, y
                    state["drag_px"] = state["pan_x"]
                    state["drag_py"] = state["pan_y"]
    
                elif event == cv2.EVENT_MOUSEMOVE:
                    if state["drag"]:
                        x0, y0, vw, vh = viewport()
                        dx = x - state["drag_sx"]
                        dy = y - state["drag_sy"]
                        state["pan_x"] = state["drag_px"] - int(dx / DISP_W * vw)
                        state["pan_y"] = state["drag_py"] - int(dy / DISP_H * vh)
    
                elif event == cv2.EVENT_MBUTTONUP:
                    state["drag"] = False
    
                elif event == cv2.EVENT_MOUSEWHEEL:
                    factor = 1.18 if flags > 0 else (1 / 1.18)
                    zoom_toward(x, y, factor)
    
            cv2.setMouseCallback(WIN, mouse_cb)
    
            # ── event loop ────────────────────────────────────────────────────
    
            print(f"\n[ruler] Click two points {self.ruler_dist_mm}mm apart.")
            print("  Scroll=zoom  Mid-drag=pan  U=undo  Z=fit  Q=save & close\n")
    
            while True:
                cv2.imshow(WIN, render())
                key = cv2.waitKey(20) & 0xFF
    
                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key in (ord("u"), ord("U")):
                    if state["pts"]:
                        state["pts"].pop()
                        state["result"] = None
                elif key in (ord("z"), ord("Z")):
                    reset_view()
                elif key in (ord("+"), ord("=")):
                    zoom_toward(DISP_W // 2, DISP_H // 2, 1.3)
                elif key in (ord("-"), ord("_")):
                    zoom_toward(DISP_W // 2, DISP_H // 2, 1 / 1.3)
    
            cv2.destroyWindow(WIN)
    
            # ── commit result ─────────────────────────────────────────────────
    
            if state["result"]:
                r = state["result"]
                self.px_per_mm = r["px_per_mm"]
                spot_mm = 2 * self.spot_r / self.px_per_mm
                print(f"[ruler] {r['px_dist']:.2f}px = {self.ruler_dist_mm}mm  "
                    f"→ {self.px_per_mm:.4f}px/mm")
                print(f"[ruler] spot diameter = {spot_mm:.4f}mm  "
                    f"(radius={self.spot_r}px)")
                self.jdata["camera"]["pixel_size_um"] = round(
                    1000 / self.px_per_mm, 4)
                with open(self.json_path, "w", encoding="utf-8") as f:
                    json.dump(self.jdata, f, indent=4)
                print(f"[ruler] pixel_size_um = "
                    f"{self.jdata['camera']['pixel_size_um']}μm  "
                    f"saved → {self.json_path}")
            else:
                print("[ruler] fewer than 2 points placed — nothing saved")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Verify mirror_map.npz and mirror_coordinates.json")
    ap.add_argument("--map",  type=Path, required=True,
                    help="Path to mirror_map.npz")
    ap.add_argument("--json", type=Path, required=True,
                    help="Path to mirror_coordinates.json")
    ap.add_argument("--images", type=Path, default=None,
                    help="Directory with calibration images (for inspector)")
    ap.add_argument("--ruler", type=Path, default=None,
                    help="Path to a ruler image for scale calibration")
    ap.add_argument("--ruler-distance-mm", type=float, default=None,
                    dest="ruler_dist",
                    help="Known distance between the two ruler clicks (mm)")
    ap.add_argument("--raw-width",  type=int, default=None)
    ap.add_argument("--raw-height", type=int, default=None)
    ap.add_argument("--heatmap", action="store_true",
                    help="Open residual heatmap instead of FOV plot")
    args = ap.parse_args()

    v = CalibrationVerifier(
        map_path=args.map,
        json_path=args.json,
        images_dir=args.images,
        ruler_path=args.ruler,
        ruler_dist_mm=args.ruler_dist,
        raw_w=args.raw_width,
        raw_h=args.raw_height,
    )

    # run ruler first if provided so px_per_mm is populated before plotting
    if args.ruler and args.ruler_dist:
        v.run_ruler_calibration()

    if args.heatmap:
        v.plot_residual_heatmap()
    else:
        v.plot_fov()


if __name__ == "__main__":
    main()