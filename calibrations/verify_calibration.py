"""
verify_calibration_publication.py
===================================
Publication-ready calibration verification figure generator.

Produces a clean, high-quality two-panel figure suitable for thesis submission:
  - Left: Reachable FOV with calibration samples and boundary
  - Right: Sample residuals with error distribution

USAGE:
  python verify_calibration_publication.py \
      --map mirror_map.npz \
      --json mirror_coordinates.json \
      --output calibration_figure.png \
      --dpi 300

  # With custom figure size (width x height in inches):
  python verify_calibration_publication.py \
      --map mirror_map.npz \
      --json mirror_coordinates.json \
      --output calibration_figure.pdf \
      --figsize 12 5 \
      --dpi 300
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for batch generation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import Rbf
from scipy.spatial import ConvexHull


class PublicationVerifier:
    """Generate publication-ready calibration verification figures."""

    def __init__(self, map_path: Path, json_path: Path):
        self.map_path = Path(map_path)
        self.json_path = Path(json_path)
        self._load_data()
        self._build_tps()

    def _load_data(self):
        """Load map and JSON data."""
        # Load .npz map
        d = np.load(self.map_path)
        self.uv = d["uv"]
        self.xy = d["xy"]
        self.uv_mean = d["uv_mean"]
        self.uv_std = d["uv_std"]
        self.x_map = d["x_map"]
        self.y_map = d["y_map"]
        self.u_map = d["u_map"]
        self.v_map = d["v_map"]
        self.bounds = d["bounds"]
        self.hull_pts = d["hull"]
        self.hull_raw = d.get("hull_raw", self.hull_pts)
        self.spot_radius = float(d.get("spot_radius_px", 35))

        print(f"[map]  {len(self.uv)} calibration samples, "
              f"{len(self.x_map)} grid points")

        # Load JSON
        with open(self.json_path, "r", encoding="utf-8-sig") as f:
            self.jdata = json.load(f)
        self.samples = [s for s in self.jdata["samples"] if "centroid" in s]
        self.cam_res = self.jdata["camera"]["resolution"]
        px_size_um = self.jdata["camera"].get("pixel_size_um")
        self.px_per_mm = 1000.0 / px_size_um if px_size_um else None

        print(f"[json] {len(self.samples)} labelled samples, "
              f"camera {self.cam_res[0]}×{self.cam_res[1]}")
        if self.px_per_mm:
            print(f"       {self.px_per_mm:.3f} px/mm")

    def _build_tps(self):
        """Build TPS interpolator and compute residuals."""
        uvn = (self.uv - self.uv_mean) / self.uv_std
        u_n, v_n = uvn[:, 0], uvn[:, 1]
        x_c = self.xy[:, 0]
        y_c = self.xy[:, 1]

        self.fx = Rbf(u_n, v_n, x_c, function="thin_plate", smooth=1e-2)
        self.fy = Rbf(u_n, v_n, y_c, function="thin_plate", smooth=1e-2)

        # Compute residuals
        pred = np.column_stack([self.fx(u_n, v_n), self.fy(u_n, v_n)])
        self.residuals = np.sqrt(np.sum((pred - self.xy) ** 2, axis=1))
        self.rms = float(np.sqrt(np.mean(self.residuals ** 2)))

        print(f"[tps]  RMS reprojection error = {self.rms:.2f} px")

    def _set_style(self, dpi):
        plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'axes.linewidth': 1.5,  # thicker axes
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })

    def _finalize_axes(self, ax):
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        ax.xaxis.label.set_fontweight('bold')
        ax.yaxis.label.set_fontweight('bold')

    def generate_figure(self, output_prefix: Path, figsize=(12, 9), dpi=300):
        """
        Generate publication-ready two-panel figure.

        Parameters
        ----------
        output_path : Path
            Output file path (.png, .pdf, .svg supported)
        figsize : tuple
            Figure size in inches (width, height)
        dpi : int
            Resolution for raster formats
        """
        # Set publication style
        """plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.dpi': dpi,
            'savefig.dpi': dpi,
            'savefig.bbox': 'tight',
            'axes.linewidth': 1.0,
            'grid.linewidth': 0.5,
            'lines.linewidth': 1.5,
            'patch.linewidth': 1.0,
        }) """

        self._set_style(dpi)

        """fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.patch.set_facecolor('white')

        for ax in axes:
            ax.set_facecolor('white')
            for spine in ax.spines.values():
                spine.set_color('black')
                spine.set_linewidth(1.0)

        # Generate panels
        self._draw_fov_panel(axes[0])
        self._draw_residual_panel(axes[1])

        # Overall title
        fig.suptitle('Mirror Calibration Verification',
                     fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save figure
        output_path = Path(output_path)
        fig.savefig(str(output_path), dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"\n[output] Saved figure → {output_path}")
        print(f"         Size: {figsize[0]}×{figsize[1]} in, {dpi} DPI")

        plt.close(fig) """

        # Figure A: FOV
        fig1, ax1 = plt.subplots(figsize=figsize)
        self._draw_fov_panel(ax1)
        self._finalize_axes(ax1)
        fig1.savefig(f"{output_prefix}_fov.png", dpi=dpi, bbox_inches='tight')
        plt.close(fig1)

        # Figure B: Residuals
        fig2, ax2 = plt.subplots(figsize=figsize)
        self._draw_residual_panel(ax2)
        self._finalize_axes(ax2)
        fig2.savefig(f"{output_prefix}_residuals.png", dpi=dpi, bbox_inches='tight')
        plt.close(fig2)

        # Figure C: RMS histogram
        fig3, ax3 = plt.subplots(figsize=figsize)
        self._draw_rms_histogram(ax3)
        self._finalize_axes(ax3)
        fig3.savefig(f"{output_prefix}_rms.png", dpi=dpi, bbox_inches='tight')
        plt.close(fig3)

        print(f"[output] Saved figures with prefix: {output_prefix}")

    def _draw_fov_panel(self, ax):
        """
        Left panel: Reachable field of view with calibration samples.
        
        Shows:
        - Safe reachable region (after spot-radius erosion)
        - Calibration sample locations
        - Spot-size footprint indicators
        - Coordinate system and scale bar
        """
        """cw, ch = self.cam_res
        ax.set_xlim(0, cw)
        ax.set_ylim(ch, 0)
        ax.set_aspect('equal')
        ax.set_title('(a) Reachable Field of View', loc='left', fontweight='bold')
        ax.set_xlabel('Camera X (pixels)')
        ax.set_ylabel('Camera Y (pixels)')

        # Fill reachable region with light gray
        hull_closed = np.vstack([self.hull_pts, self.hull_pts[0]])
        ax.fill(hull_closed[:, 0], hull_closed[:, 1],
                color='#f0f0f0', zorder=1, label='Reachable region')

        # Draw grid of reachable points (small dots)
        ax.scatter(self.x_map, self.y_map,
                   s=8, c='#d0d0d0', alpha=0.4, linewidths=0,
                   zorder=2, label='Safe grid points')

        # Draw boundary
        ax.plot(hull_closed[:, 0], hull_closed[:, 1],
                'k-', linewidth=1.5, zorder=3, label='Safe boundary')

        # Draw raw hull (before erosion) as dashed line
        if self.hull_raw is not None:
            hull_raw_closed = np.vstack([self.hull_raw, self.hull_raw[0]])
            ax.plot(hull_raw_closed[:, 0], hull_raw_closed[:, 1],
                    'k--', linewidth=1.0, alpha=0.5, zorder=3,
                    label='Pre-erosion boundary')

        # Plot calibration sample centroids
        xs = np.array([s["centroid"][0] for s in self.samples])
        ys = np.array([s["centroid"][1] for s in self.samples])
        ax.scatter(xs, ys, s=60, c='#2166ac', marker='o',
                   edgecolors='black', linewidths=0.8, zorder=5,
                   label=f'Calibration samples (n={len(self.samples)})')

        # Draw representative spot-size circles at a subset of grid points
        # (drawing all would clutter the figure)
        n_circles = min(50, len(self.x_map))
        indices = np.linspace(0, len(self.x_map) - 1, n_circles, dtype=int)
        for idx in indices:
            circle = mpatches.Circle(
                (self.x_map[idx], self.y_map[idx]),
                self.spot_radius,
                color='#b2182b', fill=False, linewidth=0.5,
                alpha=0.3, zorder=2
            )
            ax.add_patch(circle)

        # Add one labeled spot circle as legend entry
        legend_circle = mpatches.Circle(
            (xs[0], ys[0]), self.spot_radius,
            color='#b2182b', fill=False, linewidth=1.0,
            label=f'Laser spot (r={self.spot_radius:.0f} px)'
        )
        ax.add_patch(legend_circle)

        # Add scale bar if pixel size is known
        if self.px_per_mm:
            bar_mm = 10 if cw > 1500 else 5
            bar_px = bar_mm * self.px_per_mm
            bar_y = ch - 80
            bar_x = 80

            # Draw scale bar
            ax.plot([bar_x, bar_x + bar_px], [bar_y, bar_y],
                    'k-', linewidth=2.5, zorder=10)
            ax.plot([bar_x, bar_x], [bar_y - 10, bar_y + 10],
                    'k-', linewidth=2.5, zorder=10)
            ax.plot([bar_x + bar_px, bar_x + bar_px], [bar_y - 10, bar_y + 10],
                    'k-', linewidth=2.5, zorder=10)
            ax.text(bar_x + bar_px / 2, bar_y - 25,
                    f'{bar_mm} mm', ha='center', va='top',
                    fontsize=9, fontweight='bold', zorder=10)

        # Legend
        ax.legend(loc='upper right', frameon=True, fancybox=False,
                  edgecolor='black', facecolor='white', framealpha=0.9)

        # Grid
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, color='gray')"""
        cw, ch = self.cam_res
        ax.set_xlim(0, cw)
        ax.set_ylim(ch, 0)
        ax.set_aspect('equal')

        ax.set_title('Reachable Field of View', fontweight='bold')
        ax.set_xlabel('Camera X (pixels)')
        ax.set_ylabel('Camera Y (pixels)')

        hull_closed = np.vstack([self.hull_pts, self.hull_pts[0]])

        # Reachable region
        ax.fill(hull_closed[:, 0], hull_closed[:, 1],
                color='#f0f0f0', zorder=1,
                label='Reachable FOV (post-erosion)')

        # Grid points
        ax.scatter(self.x_map, self.y_map,
                s=8, c='#bdbdbd', alpha=0.5,
                label='Valid steering grid', zorder=2)

        # Boundary
        ax.plot(hull_closed[:, 0], hull_closed[:, 1],
                'k-', linewidth=1.5, label='FOV boundary', zorder=3)

        # Calibration samples
        xs = np.array([s["centroid"][0] for s in self.samples])
        ys = np.array([s["centroid"][1] for s in self.samples])

        ax.scatter(xs, ys, s=60, c='#2166ac',
                edgecolors='black', linewidths=0.8,
                label=f'Calibration samples (n={len(xs)})',
                zorder=5)

        # Move legend OUTSIDE
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                frameon=True)

        ax.grid(True, linestyle=':', alpha=0.3)

    def _draw_rms_histogram(self, ax):
        ax.hist(self.residuals, bins=20,
                edgecolor='black', alpha=0.7)

        ax.axvline(self.rms, linestyle='--',
                linewidth=2, label=f'RMS = {self.rms:.2f}px')

        ax.set_title('Reprojection Error Distribution', fontweight='bold')
        ax.set_xlabel('Error (pixels)')
        ax.set_ylabel('Count')

        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.3)

    def _draw_residual_panel(self, ax):
        """
        Right panel: Sample residuals and error statistics.
        
        Shows:
        - Each calibration sample colored by reprojection error
        - Error vectors (scaled for visibility)
        - Error histogram inset
        - Summary statistics
        """
        """cw, ch = self.cam_res
        ax.set_xlim(0, cw)
        ax.set_ylim(ch, 0)
        ax.set_aspect('equal')
        ax.set_title('(b) Calibration Residuals', loc='left', fontweight='bold')
        ax.set_xlabel('Camera X (pixels)')
        ax.set_ylabel('Camera Y (pixels)')

        # Fill reachable region
        hull_closed = np.vstack([self.hull_pts, self.hull_pts[0]])
        ax.fill(hull_closed[:, 0], hull_closed[:, 1],
                color='#f5f5f5', zorder=1)
        ax.plot(hull_closed[:, 0], hull_closed[:, 1],
                'k-', linewidth=1.0, alpha=0.5, zorder=2)

        # Compute predicted positions for all samples
        uvn = (self.uv - self.uv_mean) / self.uv_std
        pred_x = self.fx(uvn[:, 0], uvn[:, 1])
        pred_y = self.fy(uvn[:, 0], uvn[:, 1])

        # Color map: white (low error) to red (high error)
        max_err = self.residuals.max()
        norm = Normalize(vmin=0, vmax=max_err)
        cmap = plt.cm.YlOrRd

        # Draw error vectors (scaled 3× for visibility)
        arrow_scale = 3.0
        for i, (res, s) in enumerate(zip(self.residuals, self.samples)):
            cx, cy = s["centroid"]
            px, py = pred_x[i], pred_y[i]

            # Error vector
            dx = (px - cx) * arrow_scale
            dy = (py - cy) * arrow_scale

            if abs(dx) > 0.5 or abs(dy) > 0.5:
                ax.annotate('',
                    xy=(cx + dx, cy + dy),
                    xytext=(cx, cy),
                    arrowprops=dict(
                        arrowstyle='->', color='black',
                        linewidth=0.8, alpha=0.4,
                        shrinkA=0, shrinkB=0
                    ),
                    zorder=3
                )

        # Plot samples colored by error
        xs = self.xy[:, 0]
        ys = self.xy[:, 1]
        scatter = ax.scatter(xs, ys, c=self.residuals,
                           cmap=cmap, norm=norm,
                           s=80, edgecolors='black',
                           linewidths=0.8, zorder=4)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Reprojection Error (pixels)', fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        # Add statistics text box
        stats_text = (
            f'RMS error: {self.rms:.2f} px\n'
            f'Mean error: {self.residuals.mean():.2f} px\n'
            f'Median error: {np.median(self.residuals):.2f} px\n'
            f'Max error: {max_err:.2f} px\n'
            f'Samples: {len(self.samples)}'
        )
        
        if self.px_per_mm:
            stats_text += f'\n({self.rms / self.px_per_mm:.3f} mm RMS)'

        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white',
                         edgecolor='black', linewidth=1.0, alpha=0.9),
                fontsize=9, family='monospace')

        # Inset histogram of errors
        ax_inset = ax.inset_axes([0.62, 0.05, 0.35, 0.25])
        ax_inset.hist(self.residuals, bins=15, color='#2166ac',
                     edgecolor='black', linewidth=0.5, alpha=0.7)
        ax_inset.axvline(self.rms, color='red', linestyle='--',
                        linewidth=1.5, label=f'RMS={self.rms:.2f}px')
        ax_inset.set_xlabel('Error (px)', fontsize=8)
        ax_inset.set_ylabel('Count', fontsize=8)
        ax_inset.tick_params(labelsize=7)
        ax_inset.legend(fontsize=7, frameon=False)
        ax_inset.set_facecolor('white')
        for spine in ax_inset.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.8)

        # Note about arrow scaling
        ax.text(0.98, 0.02,
                f'Error vectors scaled {arrow_scale:.0f}× for visibility',
                transform=ax.transAxes,
                ha='right', va='bottom',
                fontsize=8, style='italic', color='#666666')

        # Grid
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, color='gray') """
        cw, ch = self.cam_res
        ax.set_xlim(0, cw)
        ax.set_ylim(ch, 0)
        ax.set_aspect('equal')

        ax.set_title('Calibration Residuals', fontweight='bold')
        ax.set_xlabel('Camera X (pixels)')
        ax.set_ylabel('Camera Y (pixels)')

        hull_closed = np.vstack([self.hull_pts, self.hull_pts[0]])

        ax.fill(hull_closed[:, 0], hull_closed[:, 1],
                color='#f5f5f5', zorder=1)

        ax.plot(hull_closed[:, 0], hull_closed[:, 1],
                'k-', linewidth=1.0, alpha=0.5)

        # Predictions
        uvn = (self.uv - self.uv_mean) / self.uv_std
        pred_x = self.fx(uvn[:, 0], uvn[:, 1])
        pred_y = self.fy(uvn[:, 0], uvn[:, 1])

        norm = Normalize(vmin=0, vmax=self.residuals.max())
        cmap = plt.cm.YlOrRd

        scatter = ax.scatter(self.xy[:, 0], self.xy[:, 1],
                            c=self.residuals,
                            cmap=cmap, norm=norm,
                            s=80,
                            edgecolors='black',
                            linewidths=0.8)

        # Colorbar (clean placement)
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Reprojection Error (pixels)', fontweight='bold')

        # Stats box only
        stats_text = (
            f'RMS: {self.rms:.2f}px\n'
            f'Mean: {self.residuals.mean():.2f}px\n'
            f'Max: {self.residuals.max():.2f}px'
        )

        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                va='top',
                bbox=dict(facecolor='white', edgecolor='black'))

        ax.grid(True, linestyle=':', alpha=0.3)


def main():
    ap = argparse.ArgumentParser(
        description='Generate publication-ready calibration verification figure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # High-resolution PNG for thesis:
  python verify_calibration_publication.py --map mirror_map.npz \\
      --json mirror_coordinates.json --output calibration.png --dpi 300

  # Vector PDF for LaTeX:
  python verify_calibration_publication.py --map mirror_map.npz \\
      --json mirror_coordinates.json --output calibration.pdf

  # Custom figure size (wide format):
  python verify_calibration_publication.py --map mirror_map.npz \\
      --json mirror_coordinates.json --output calibration.png \\
      --figsize 14 5 --dpi 300
        """
    )

    ap.add_argument('--map', type=Path, required=True,
                    help='Path to mirror_map.npz')
    ap.add_argument('--json', type=Path, required=True,
                    help='Path to mirror_coordinates.json')
    ap.add_argument('--output', type=Path, required=True,
                    help='Output figure path (.png, .pdf, or .svg)')
    ap.add_argument('--figsize', type=float, nargs=2, default=[12, 5],
                    metavar=('WIDTH', 'HEIGHT'),
                    help='Figure size in inches (default: 12 5)')
    ap.add_argument('--dpi', type=int, default=300,
                    help='Resolution in DPI for raster formats (default: 300)')

    args = ap.parse_args()

    # Validate inputs
    if not args.map.exists():
        ap.error(f'Map file not found: {args.map}')
    if not args.json.exists():
        ap.error(f'JSON file not found: {args.json}')

    # Generate figure
    verifier = PublicationVerifier(args.map, args.json)
    verifier.generate_figure(
       output_prefix=args.output.with_suffix(''),
        figsize=tuple(args.figsize),
        dpi=args.dpi
    )

    print('\nDone. Figure ready for publication.')


if __name__ == '__main__':
    main()