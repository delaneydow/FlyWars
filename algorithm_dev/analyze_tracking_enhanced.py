#analyze_tracking_enhanced.py

"""
Enhanced Tracking Algorithm Analysis — Adaptive vs Fixed Horizon Comparison
============================================================================
Adds comparison of adaptive horizon against fixed horizons to demonstrate
superiority of the adaptive approach across different speed regimes.
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import binomtest


# ─────────────────────────────────────────────
# USER CONFIG
# ─────────────────────────────────────────────
CSV_DIR = Path("./algorithm_dev")
OUT_DIR = Path("./figures")
DPI = 300

CATEGORIES = {
    "multi_slow":   "Multi-Object\nSlow",
    "single_slow":  "Single-Object\nSlow",
    "multi_fast":   "Multi-Object\nFast",
    "single_fast":  "Single-Object\nFast",
    "multi_mix":    "Multi-Object\nMixed",
    "control":       "Control"
}

# ── Algorithm constants ──────────────────────────────────────────────────────
FRAME_DT        = 1 / 120.0
MIN_FIRE_TIME   = 0.25
SYSTEM_LATENCY  = 0.075 + MIN_FIRE_TIME
PREDICT_HORIZON = int(SYSTEM_LATENCY / FRAME_DT)   # ≈ 39 frames

SPOT_RADIUS_MM          = 1.7
MM_PER_PX               = 0.533
CALIBRATION_ERROR_FACTOR = 1.05
SPOT_RADIUS_PX_SAFE     = SPOT_RADIUS_MM * CALIBRATION_ERROR_FACTOR

MAX_COV_THRESHOLD = 50
ARENA_DIAG        = np.hypot(1600, 1200)
LASER_COOLDOWN_FRAMES = int(0.25 / FRAME_DT)

STATE_HOVERING     = "hovering"
STATE_CRUISING     = "cruising"
STATE_ACCELERATING = "accelerating"

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
PALETTE = {
    "control":"#8E8E8E",
    "multi_slow":"#0072B2",
    "single_slow":"#009E73",
    "multi_fast":"#D55E00",
    "single_fast":"#CC79A7",
    "multi_mix":"#E69F00"
}

STATE_COLORS = {
    STATE_HOVERING:     "#4FC3F7",
    STATE_CRUISING:     "#81C784",
    STATE_ACCELERATING: "#FF8A65",
}


def apply_poster_style():
    mpl.rcParams.update({
        "figure.facecolor":"white",
        "axes.facecolor":"white",
        "font.family":"sans-serif",
        "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
        "font.size":14,
        "axes.labelsize":16,
        "axes.titlesize":18,
        "xtick.labelsize":13,
        "ytick.labelsize":13,
        "axes.spines.top":False,
        "axes.spines.right":False,
        "axes.linewidth":1.2,
        "grid.alpha":0.25,
        "grid.linestyle":"--",
        "legend.frameon":False
    })


# ─────────────────────────────────────────────
# SIMULATION HELPERS
# ─────────────────────────────────────────────

def adaptive_k(speed, k=PREDICT_HORIZON):
    """Mirror predict_position() adaptive scaling."""
    if speed < 10:
        return 1.0
    elif speed < 50:
        return k * 0.5
    elif speed < 150:
        return k * 0.8
    else:
        return k * 1.0


def predict_xy(row, prev_row=None):
    """
    Predict future position using ADAPTIVE horizon.
    Returns (x_pred, y_pred, k_eff).
    """
    vx, vy   = row["vx"], row["vy"]
    speed    = row["speed"]
    k_eff    = adaptive_k(speed)

    # velocity damping for near-stationary
    if speed < 10:
        vx *= 0.5
        vy *= 0.5

    # acceleration estimate
    if prev_row is not None:
        ax = (vx - prev_row["vx"]) / FRAME_DT
        ay = (vy - prev_row["vy"]) / FRAME_DT
    else:
        ax = ay = 0.0

    x_pred = row["x"] + vx * k_eff + 0.5 * ax * k_eff ** 2
    y_pred = row["y"] + vy * k_eff + 0.5 * ay * k_eff ** 2

    return x_pred, y_pred, k_eff


def predict_xy_fixed(row, k_fixed, prev_row=None):
    """
    Predict position using a FIXED horizon k_fixed (no adaptation).
    Returns (x_pred, y_pred).
    """
    vx, vy = row["vx"], row["vy"]
    speed = row["speed"]
    
    # Apply same velocity damping as adaptive
    if speed < 10:
        vx *= 0.5
        vy *= 0.5
    
    # acceleration estimate
    if prev_row is not None:
        ax = (vx - prev_row["vx"]) / FRAME_DT
        ay = (vy - prev_row["vy"]) / FRAME_DT
    else:
        ax = ay = 0.0
    
    x_pred = row["x"] + vx * k_fixed + 0.5 * ax * k_fixed ** 2
    y_pred = row["y"] + vy * k_fixed + 0.5 * ay * k_fixed ** 2
    
    return x_pred, y_pred


def score_row(row, beam_x=800.0, beam_y=600.0):
    """
    Replicate score_track() on a single CSV row.
    """
    uncertainty = row["cov_xx"] + row["cov_yy"]
    age = row.get("age", 3)
    rel_max_cov = MAX_COV_THRESHOLD * (1.0 + age * 0.5)

    if uncertainty > rel_max_cov:
        return 0.0

    stability = 1.0 / (1.0 + float(uncertainty))

    pred_x, pred_y, _ = predict_xy(row)
    mirror_delta = np.hypot(pred_x - beam_x, pred_y - beam_y)
    mirror_norm  = 1.0 - (mirror_delta / ARENA_DIAG)
    mirror_cost  = max(0.1, mirror_norm)

    state_weight = {
        STATE_HOVERING:     1.3,
        STATE_CRUISING:     0.9,
        STATE_ACCELERATING: 0.5,
    }.get(row["state"], 0.5)

    speed_penalty = 1.0 / (1.0 + row["speed"])

    return float(mirror_cost * state_weight * stability * speed_penalty)


def simulate_hits(df_cat, spot_radius=SPOT_RADIUS_PX_SAFE):
    """
    Simulate tracking with ADAPTIVE horizon.
    """
    records = []
    last_fire_frame = -LASER_COOLDOWN_FRAMES

    frames = sorted(df_cat["frame"].unique())

    for frame in frames:
        sub = df_cat[df_cat["frame"] == frame].copy()

        if frame - last_fire_frame < LASER_COOLDOWN_FRAMES:
            continue

        scored = []
        for _, row in sub.iterrows():
            s = score_row(row)
            if s > 0:
                scored.append((s, row))

        if not scored:
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_row = scored[0]

        x_pred, y_pred, k_eff = predict_xy(best_row)
        k_eff_int = max(1, int(round(k_eff)))

        future_frame = frame + k_eff_int
        future = df_cat[
            (df_cat["frame"] == future_frame) &
            (df_cat["track_id"] == best_row["track_id"])
        ]

        if future.empty:
            records.append({
                "frame": frame,
                "track_id": best_row["track_id"],
                "pred_error_px": np.nan,
                "hit": False,
                "score": best_score,
                "k_eff": k_eff,
                "speed": best_row["speed"],
                "state": best_row["state"],
            })
            last_fire_frame = frame
            continue

        actual = future.iloc[0]
        err = np.hypot(x_pred - actual["x"], y_pred - actual["y"])
        hit = err <= spot_radius

        records.append({
            "frame": frame,
            "track_id": best_row["track_id"],
            "pred_error_px": err,
            "hit": hit,
            "score": best_score,
            "k_eff": k_eff,
            "speed": best_row["speed"],
            "state": best_row["state"],
        })
        last_fire_frame = frame

    return pd.DataFrame(records)


def simulate_fixed_horizon(dfs, k_fixed):
    """
    Re-simulate all scenarios using a FIXED horizon k_fixed.
    Use SAME scoring + firing logic as adaptive so comparison is fair
    Returns dict of category -> DataFrame with pred_error_px.
    """
    results_fixed = {}
    
    for cat, df_cat in dfs.items():
        records = []
        last_fire_frame = -LASER_COOLDOWN_FRAMES
        
        frames = sorted(df_cat["frame"].unique())
        
        for frame in frames:
            if frame - last_fire_frame < LASER_COOLDOWN_FRAMES:
                continue

            sub = df_cat[df_cat["frame"] == frame].copy()

            scored = []

            for _, row in sub.iterrows():

                s = score_row(row)

                if s > 0:
                    scored.append((s,row))

            if not scored:
                continue

            scored.sort(key=lambda x: x[0], reverse=True)

            best_score, best_row = scored[0]

            # predict using fixed horizon
            x_pred, y_pred = predict_xy_fixed(best_row,k_fixed)

            future_frame = frame + int(round(k_fixed))

            future = df_cat[
                (df_cat["frame"] == future_frame) &
                (df_cat["track_id"] == best_row["track_id"])
            ]

            if future.empty:

                records.append({
                    "frame":frame,
                    "track_id":best_row["track_id"],
                    "pred_error_px":np.nan,
                    "score":best_score,
                    "speed":best_row["speed"],
                    "state":best_row["state"]
                })

                last_fire_frame = frame
                continue

            actual = future.iloc[0]

            err = np.hypot(
                x_pred - actual["x"],
                y_pred - actual["y"]
            )

            records.append({
                "frame":frame,
                "track_id":best_row["track_id"],
                "pred_error_px":err,
                "score":best_score,
                "speed":best_row["speed"],
                "state":best_row["state"]
            })

            last_fire_frame = frame

        results_fixed[cat] = pd.DataFrame(records)

    return results_fixed


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_csvs():
    dfs = {}
    for key in CATEGORIES:
        p = CSV_DIR / f"{key}.csv"
        if not p.exists():
            print(f"  [WARN] {p} not found — skipping")
            continue

        try:
            df = pd.read_csv(p, sep=None, engine="python")
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=[
                "frame","time","track_id","x","y",
                "vx","vy","speed","state","cov_xx","cov_yy"
            ])

        df.columns = [c.strip().lower() for c in df.columns]

        if df.empty:
            dfs[key] = df
            print(f"  Loaded {key}: EMPTY (control case)")
            continue

        for col in ["frame", "x", "y", "vx", "vy", "speed", "cov_xx", "cov_yy"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["state"] = df["state"].str.strip().str.lower()
        dfs[key] = df
        print(f"  Loaded {key}: {len(df)} rows, "
              f"{df['frame'].nunique()} frames, "
              f"{df['track_id'].nunique()} tracks")
    return dfs


# ─────────────────────────────────────────────
# NEW: ADAPTIVE VS FIXED COMPARISON PLOT
# ─────────────────────────────────────────────

def plot_adaptive_vs_fixed_comparison(dfs, results_adaptive):
    """
    Compare adaptive horizon against multiple fixed horizons.
    Shows that adaptive performs better across different speed regimes.
    """
    apply_poster_style()
    
    # Test multiple fixed horizon values
    fixed_horizons = [1, 5, 10, 20, PREDICT_HORIZON]
    
    #fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig1, ax1 = plt.subplots(figsize=(7,5))
    fig2, ax2 = plt.subplots(figsize=(7,5))
    
    # ====== LEFT PANEL: Overall error comparison ======
    #ax1 = axes[0]
    
    # Collect median errors for each approach
    adaptive_errors = []
    fixed_errors = {k: [] for k in fixed_horizons}
    
    # Get adaptive errors
    for cat, df in results_adaptive.items():
        errors = df["pred_error_px"].dropna()
        if len(errors) > 0:
            adaptive_errors.extend(errors.values)
    
    # Simulate fixed horizons
    for k_fixed in fixed_horizons:
        print(f"  Simulating fixed horizon k={k_fixed}...")
        results_fixed = simulate_fixed_horizon(dfs, k_fixed)
        
        for cat, df in results_fixed.items():
            errors = df["pred_error_px"].dropna()
            if len(errors) > 0:
                fixed_errors[k_fixed].extend(errors.values)
    
    # Box plot comparison
    data_to_plot = [adaptive_errors] + [fixed_errors[k] for k in fixed_horizons]
    labels = ["Adaptive"] +[
                            "k=1",
                            "k=5",
                            "k=10",
                            "k=20",
                            "k=39\n(system latency)"]
 #[f"k={k}" for k in fixed_horizons]
    colors = ["#E69F00",  # Adaptive (orange)

    "#56B4E9",  # k=1  light blue
    "#009E73",  # k=5  green
    "#0072B2",  # k=10 blue
    "#CC79A7",  # k=20 purple
    "#D55E00" ]  # k=39 red (important)
    
    bp = ax1.boxplot(data_to_plot, tick_labels=labels, patch_artist=True,
                     widths=0.6, showfliers=False)
    
    for box in bp['boxes']:
        box.set_linewidth(1.5)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add median values as text
    medians = [np.median(d) for d in data_to_plot]
    for i, (med, label) in enumerate(zip(medians, labels)):
        #ax1.text(i + 1, med, f'{med:.1f}', 
               # ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(i+1,med*1.25,   # vertical offset works well on log scale
                    f'{med:.1f}px',ha='center',fontsize=10,fontweight='bold'
        )
        
    
    ax1.axhline(SPOT_RADIUS_PX_SAFE, linestyle="--", color="red", 
               linewidth=1.5, alpha=0.7, label="Laser radius")
    #ax1.text(0.6,SPOT_RADIUS_PX_SAFE*1.1,"Hit threshold",color="red",fontsize=10)
    laser_line = Line2D([0],[0],color="red",linestyle="--",
                        linewidth=1.5,label="Laser radius (hit threshold)")
    ax1.set_ylabel("Prediction error (pixels)")
    ax1.set_yscale("log")
    ax1.set_title("Adaptive vs. Fixed Horizon: Overall Performance")
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(handles=[laser_line],loc='upper left')
    
    # ====== RIGHT PANEL: Error by speed regime ======
    #ax2 = axes[1]
    
    # Combine all data with speed info
    all_adaptive = pd.concat([df.assign(category=cat) for cat, df in results_adaptive.items()])
    all_adaptive = all_adaptive.dropna(subset=["speed", "pred_error_px"])
    
    # Define speed regimes
    speed_bins = [(0, 10, "Hovering"), (10, 50, "Cruising"), 
                  (50, 150, "Accelerating")]
    
    regime_labels = [label for _, _, label in speed_bins]
    x_pos = np.arange(len(speed_bins))
    width = 0.2
    fixed_colors = {
        5:"#56B4E9",
        20:"#0072B2",
        PREDICT_HORIZON:"#003F5C"
    }
    
    # Adaptive errors by regime
    adaptive_regime_errors = []
    for low, high, _ in speed_bins:
        regime_data = all_adaptive[(all_adaptive["speed"] >= low) & 
                                   (all_adaptive["speed"] < high)]
        median_err = regime_data["pred_error_px"].median() if len(regime_data) > 0 else np.nan
        adaptive_regime_errors.append(median_err)
    
    # Plot adaptive
    ax2.bar( x_pos - width*1.5, adaptive_regime_errors, width,
                label="Adaptive",color="#E69F00", alpha = 0.85)
    
    # Plot selected fixed horizons
    selected_fixed = [5, 20, PREDICT_HORIZON]
    for j, k_fixed in enumerate(selected_fixed):
        results_fixed = simulate_fixed_horizon(dfs, k_fixed)
        all_fixed = pd.concat([df.assign(category=cat) for cat, df in results_fixed.items()])
        all_fixed = all_fixed.dropna(subset=["speed", "pred_error_px"])
        
        fixed_regime_errors = []
        for low, high, _ in speed_bins:
            regime_data = all_fixed[(all_fixed["speed"] >= low) & 
                                   (all_fixed["speed"] < high)]
            median_err = regime_data["pred_error_px"].median() if len(regime_data) > 0 else np.nan
            fixed_regime_errors.append(median_err)
        
        offset = (-0.5 + j + 0.5)*width #(j+1 - len(selected_fixed)/2)*width
        ax2.bar(x_pos + offset, fixed_regime_errors, width, 
               label=f"k={k_fixed}", color=fixed_colors[k_fixed], alpha = 0.75)#alpha=min(0.6 + j*0.1, 1.0))
    
    ax2.axhline(SPOT_RADIUS_PX_SAFE, linestyle="--", color="red", 
                linewidth=1.5, alpha=0.7, label="Laser radius")
    ax2.set_xticks(x_pos + width * 1.5)
    ax2.set_xticklabels(regime_labels)
    ax2.set_ylabel("Median prediction error (pixels)")
    ax2.set_yscale("log")
    ax2.set_ylim(0.3, 2000)
    ax2.set_title("Performance by Speed Regime")
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(axis='y', alpha=0.3)

    #count = len(regime_data)
    #ax2.text(x_pos[i],0.5,f"n={count}",ha='center', fontsize=9)

    # Add sample counts per regime (adaptive data)
    for idx, (low, high, _) in enumerate(speed_bins):

        regime_data = all_adaptive[
            (all_adaptive["speed"] >= low) &
            (all_adaptive["speed"] < high)
        ]

        count = len(regime_data)

        if count > 0:
            ax2.text(
                x_pos[idx],
                0.6,
                f"n={count}",
                ha='center',
                fontsize=9
            )

    
    fig1.tight_layout()
    fig1.savefig(OUT_DIR/"poster_adaptive_vs_fixed_overall.pdf", bbox_inches="tight")
    fig1.savefig(OUT_DIR/"poster_adaptive_vs_fixed_overall.svg", bbox_inches="tight")
    #fig.savefig(OUT_DIR/"poster_adaptive_vs_fixed.png", dpi=DPI, bbox_inches="tight")

    fig2.tight_layout()
    fig2.savefig(OUT_DIR/"poster_speed_regime_comparison.pdf", bbox_inches="tight")
    fig2.savefig(OUT_DIR/"poster_speed_regime_comparison.svg", bbox_inches="tight")

    plt.close()
    
    print(f"\n✓ Saved: poster_adaptive_vs_fixed.pdf/svg/png")
    
    # Print summary statistics
    print("\n=== Adaptive vs Fixed Horizon Summary ===")
    #print(f"Adaptive median error: {np.median(adaptive_errors):.2f} px")
    #for k_fixed in fixed_horizons:
     #   print(f"Fixed k={k_fixed} median error: {np.median(fixed_errors[k_fixed]):.2f} px")
    
    #improvement = (np.median(fixed_errors[PREDICT_HORIZON]) - np.median(adaptive_errors)) / np.median(fixed_errors[PREDICT_HORIZON]) * 100
    #print(f"\nAdaptive improvement over max fixed horizon: {improvement:.1f}%")

    adaptive_median = np.median(adaptive_errors)

    adaptive_hit_rate = np.mean(
        np.array(adaptive_errors) <= SPOT_RADIUS_PX_SAFE
    )

    print(f"Adaptive median error: {adaptive_median:.2f} px")
    print(f"Adaptive within laser radius: {adaptive_hit_rate*100:.1f}%")

    for k_fixed in fixed_horizons:

        fixed_med = np.median(fixed_errors[k_fixed])

        fixed_hit_rate = np.mean(
            np.array(fixed_errors[k_fixed]) <= SPOT_RADIUS_PX_SAFE
        )

        print(f"Fixed k={k_fixed} median error: {fixed_med:.2f} px")
        print(f"Fixed k={k_fixed} within radius: {fixed_hit_rate*100:.1f}%")

    improvement = (
        (np.median(fixed_errors[PREDICT_HORIZON])
        - adaptive_median)
        / np.median(fixed_errors[PREDICT_HORIZON])
        * 100
    )

    print(f"\nAdaptive improvement over max fixed horizon: {improvement:.1f}%")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Loading CSVs ===")
    dfs = load_csvs()

    if not dfs:
        print("\n[ERROR] No CSV files found.")
        return

    print("\n=== Running adaptive simulations ===")
    results = {}
    for cat, df in dfs.items():
        print(f"  Simulating {cat} ...", end=" ")
        sim = simulate_hits(df)
        results[cat] = sim
        n = len(sim)
        hits = sim["hit"].sum() if n > 0 else 0
        print(f"{n} shots simulated, {hits} hits "
              f"({hits/n*100:.1f}% hit rate)" if n > 0 else "0 shots")

    print("\n=== Generating adaptive vs fixed comparison ===")
    plot_adaptive_vs_fixed_comparison(dfs, results)
    
    print(f"\n✓ All figures saved to: {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()