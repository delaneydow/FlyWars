#analyze_tracking.py

"""
Tracking Algorithm — Conference Poster Analysis Script
=======================================================
Metrics:
  1. Prediction Error      — predicted vs. actual position per frame
  2. Adaptive Horizon      — k_eff vs. measured speed
  3. Hit Rate / Miss Rate  — per scenario category + summary

Usage
-----
1. Drop your CSVs into the same folder as this script (or set CSV_DIR below).
   Name them exactly:
       multi_slow.csv
       single_slow.csv
       multi_fast.csv
       single_fast.csv
       multi_mix.csv

2. Run:
       python analyze_tracking.py

3. Figures are saved as high-res PNGs in ./figures/

CSV format expected (tab-separated, header row):
    frame  time  track_id  x  y  vx  vy  speed  state  cov_xx  cov_yy
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
CSV_DIR = Path("./algorithm_dev")          # folder containing the CSVs
OUT_DIR = Path("./figures")  # where PNGs are saved
DPI = 300                    # publication quality

CATEGORIES = {
    "multi_slow":   "Multi-Object\nSlow",
    "single_slow":  "Single-Object\nSlow",
    "multi_fast":   "Multi-Object\nFast",
    "single_fast":  "Single-Object\nFast",
    "multi_mix":    "Multi-Object\nMixed",
    "control":       "Control"
}

# ── Algorithm constants (mirror object_scoring.py) ──────────────────────────
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

BG      = "#0D1117"
PANEL   = "#161B22"
TEXT    = "#E6EDF3"
GRID    = "#30363D"
ACCENT  = "#58A6FF"

def apply_dark_style():
    mpl.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    PANEL,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TEXT,
        "axes.titlecolor":   TEXT,
        "xtick.color":       TEXT,
        "ytick.color":       TEXT,
        "text.color":        TEXT,
        "grid.color":        GRID,
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "legend.facecolor":  PANEL,
        "legend.edgecolor":  GRID,
        "legend.labelcolor": TEXT,
        "font.family":       "monospace",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

# ─────────────────────────────────────────────
# SIMULATION HELPERS  (re-implement algorithm logic)
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
    Predict future position for one row using the algorithm's formula.
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


def score_row(row, beam_x=800.0, beam_y=600.0):
    """
    Replicate score_track() on a single CSV row.
    Returns score (float), or 0 if filtered out.
    """
    uncertainty = row["cov_xx"] + row["cov_yy"]
    age = row.get("age", 3)          # age not in CSV; assume visible ≥ 3
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
    Walk through frames in order.  For each frame where a target is scored > 0
    (and cooldown has elapsed), predict its future position and compare to
    actual position k_eff frames later.  Record hit / miss.

    Returns a DataFrame with columns:
        frame, track_id, pred_error_px, hit, score, k_eff, speed, state
    """
    records = []
    last_fire_frame = -LASER_COOLDOWN_FRAMES  # start ready to fire

    frames = sorted(df_cat["frame"].unique())

    for frame in frames:
        sub = df_cat[df_cat["frame"] == frame].copy()

        # cooldown guard
        if frame - last_fire_frame < LASER_COOLDOWN_FRAMES:
            continue

        # score every track in this frame
        scored = []
        for _, row in sub.iterrows():
            s = score_row(row)
            if s > 0:
                scored.append((s, row))

        if not scored:
            continue

        # pick highest-scoring target
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_row = scored[0]

        x_pred, y_pred, k_eff = predict_xy(best_row)
        k_eff_int = max(1, int(round(k_eff)))

        # look up actual position k_eff frames later
        future_frame = frame + k_eff_int
        future = df_cat[
            (df_cat["frame"] == future_frame) &
            (df_cat["track_id"] == best_row["track_id"])
        ]

        if future.empty:
            # track lost — count as miss
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
        #df = pd.read_csv(p, sep=None, engine="python")   # auto-detect sep
        #df.columns = [c.strip().lower() for c in df.columns]

        try:
            df = pd.read_csv(p, sep=None, engine="python")
        except pd.errors.EmptyDataError:
            # create empty dataframe with expected columns
            df = pd.DataFrame(columns=[
                "frame","time","track_id","x","y",
                "vx","vy","speed","state","cov_xx","cov_yy"
            ])

        df.columns = [c.strip().lower() for c in df.columns]

        if df.empty:
            dfs[key] = df
            print(f"  Loaded {key}: EMPTY (control case)")
            continue
        # ensure numeric
        for col in ["frame", "x", "y", "vx", "vy", "speed", "cov_xx", "cov_yy"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["state"] = df["state"].str.strip().str.lower()
        dfs[key] = df
        print(f"  Loaded {key}: {len(df)} rows, "
              f"{df['frame'].nunique()} frames, "
              f"{df['track_id'].nunique()} tracks")
    return dfs

# export simulation results 
def export_results(results, all_sim):
    """Export all computed metrics for external plotting."""
    
    export_dir = OUT_DIR / "data"
    export_dir.mkdir(exist_ok=True)

    summary_rows = []

    for cat, df in results.items():

        out_file = export_dir / f"{cat}_simulation_results.csv"
        df.to_csv(out_file, index=False)

        total = len(df)
        hits = df["hit"].sum() if total > 0 else 0

        pred_err = df["pred_error_px"].dropna()

        summary_rows.append({
            "category":cat,
            "shots":total,
            "hits":hits,
            "hit_rate":hits/total if total>0 else 0,
            "median_error":pred_err.median() if len(pred_err)>0 else np.nan,
            "mean_error":pred_err.mean() if len(pred_err)>0 else np.nan,
            "std_error":pred_err.std() if len(pred_err)>0 else np.nan,
            "mean_k_eff":df["k_eff"].mean() if total>0 else np.nan,
            "mean_score":df["score"].mean() if total>0 else np.nan
        })

    summary = pd.DataFrame(summary_rows)

    summary_file = export_dir / "summary_metrics.csv"
    summary.to_csv(summary_file,index=False)

    all_file = export_dir / "all_simulations.csv"
    all_sim.to_csv(all_file,index=False)

    print("\n=== Data exported ===")
    print(f"  {summary_file}")
    print(f"  {all_file}")

    #stats analysis

def hit_rate_ci(hits, total, confidence=0.95):
        """Binomial confidence interval."""
        if total == 0:
            return (0,0)

        res = binomtest(int(hits), int(total))

        ci = res.proportion_ci(confidence_level=confidence)

        return float(ci.low), float(ci.high)

def speed_regime(speed):

    if speed < 10:
        return "stationary"
    elif speed < 50:
        return "slow"
    elif speed < 150:
        return "moderate"
    else:
        return "fast"
    
def export_advanced_metrics(results, all_sim):

    export_dir = OUT_DIR / "data"
    export_dir.mkdir(exist_ok=True)

    rows = []

    for cat,df in results.items():

        total = len(df)
        hits = df["hit"].sum() if total>0 else 0

        hit_rate = hits/total if total>0 else 0

        ci_low,ci_high = hit_rate_ci(hits,total)

        errors = df["pred_error_px"].dropna()

        q1 = errors.quantile(0.25) if len(errors)>0 else np.nan
        q2 = errors.quantile(0.50) if len(errors)>0 else np.nan
        q3 = errors.quantile(0.75) if len(errors)>0 else np.nan

        # state hit rates
        state_rates = {}

        for st in [STATE_HOVERING,STATE_CRUISING,STATE_ACCELERATING]:

            sub = df[df["state"]==st]

            if len(sub)==0:
                state_rates[st]=np.nan
            else:
                state_rates[st]=sub["hit"].mean()

        # speed regime stats
        df["speed_regime"] = df["speed"].apply(speed_regime)

        regime_rates={}

        for reg in ["stationary","slow","moderate","fast"]:

            sub=df[df["speed_regime"]==reg]

            if len(sub)==0:
                regime_rates[reg]=np.nan
            else:
                regime_rates[reg]=sub["hit"].mean()

        # efficiency metrics

        mean_k = df["k_eff"].mean() if total>0 else np.nan

        mean_score=df["score"].mean() if total>0 else np.nan

        tracking_efficiency = hits / total if total>0 else 0

        rows.append({

            "category":cat,

            "shots":total,

            "hits":hits,

            "hit_rate":hit_rate,

            "hit_rate_ci_low":ci_low,

            "hit_rate_ci_high":ci_high,

            "median_error":q2,

            "error_q1":q1,

            "error_q3":q3,

            "mean_error":errors.mean() if len(errors)>0 else np.nan,

            "std_error":errors.std() if len(errors)>0 else np.nan,

            "hover_hit_rate":state_rates[STATE_HOVERING],

            "cruise_hit_rate":state_rates[STATE_CRUISING],

            "accel_hit_rate":state_rates[STATE_ACCELERATING],

            "stationary_hit_rate":regime_rates["stationary"],

            "slow_hit_rate":regime_rates["slow"],

            "moderate_hit_rate":regime_rates["moderate"],

            "fast_hit_rate":regime_rates["fast"],

            "mean_k_eff":mean_k,

            "mean_priority_score":mean_score,

            "tracking_efficiency":tracking_efficiency
        })

    advanced = pd.DataFrame(rows)

    advanced_file = export_dir / "advanced_metrics.csv"

    advanced.to_csv(advanced_file,index=False)

    print(f"  {advanced_file}")



# ─────────────────────────────────────────────
# FIGURE 1 — Prediction Error  (per-category + summary)
# ─────────────────────────────────────────────

def fig_prediction_error(results):
    apply_dark_style()
    cats = list(results.keys())
    n = len(cats)

    fig = plt.figure(figsize=(7 * n // 2 + 2, 10), facecolor=BG)
    fig.suptitle("Prediction Error: Predicted vs. Actual Position",
                 fontsize=16, fontweight="bold", color=TEXT, y=0.98)

    gs = gridspec.GridSpec(2, n, figure=fig,
                           hspace=0.45, wspace=0.35,
                           top=0.92, bottom=0.08)

    # ── top row: per-category box / violin ──────────────────────────────────
    for i, cat in enumerate(cats):
        ax = fig.add_subplot(gs[0, i])
        data = results[cat]["pred_error_px"].dropna()

        parts = ax.violinplot(data, positions=[0], widths=0.7,
                              showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(PALETTE[cat])
            pc.set_alpha(0.75)
        parts["cmedians"].set_color(TEXT)
        parts["cmedians"].set_linewidth(2)

        # scatter jitter
        jitter = np.random.uniform(-0.12, 0.12, size=len(data))
        ax.scatter(jitter, data, s=6, alpha=0.4,
                   color=PALETTE[cat], zorder=3)

        ax.axhline(SPOT_RADIUS_PX_SAFE, color="#FF6B6B",
                   linestyle="--", linewidth=1.2, label=f"Spot radius\n({SPOT_RADIUS_PX_SAFE:.1f} px)")
        ax.set_title(CATEGORIES[cat], fontsize=10, pad=6)
        ax.set_ylabel("Error (px)" if i == 0 else "")
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.4)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

        median = float(np.median(data)) if len(data) else 0
        ax.text(0.5, 0.02, f"med={median:.1f}px",
                transform=ax.transAxes, ha="center", fontsize=8,
                color=TEXT, alpha=0.8)

    # ── bottom row: summary CDF ──────────────────────────────────────────────
    ax_sum = fig.add_subplot(gs[1, :])
    for cat in cats:
        data = results[cat]["pred_error_px"].dropna().sort_values()
        if data.empty:
            continue
        cdf = np.arange(1, len(data) + 1) / len(data)
        ax_sum.plot(data, cdf, color=PALETTE[cat],
                    linewidth=2, label=CATEGORIES[cat].replace("\n", " "))

    ax_sum.axvline(SPOT_RADIUS_PX_SAFE, color="#FF6B6B",
                   linestyle="--", linewidth=1.5, label="Spot radius (safe)")
    ax_sum.set_xlabel("Prediction Error (px)")
    ax_sum.set_ylabel("Cumulative Fraction")
    ax_sum.set_title("CDF of Prediction Error — All Categories", fontsize=11)
    ax_sum.legend(fontsize=8, ncol=3)
    ax_sum.grid(alpha=0.4)
    ax_sum.set_xlim(left=0)
    ax_sum.set_ylim(0, 1.02)

    out = OUT_DIR / "fig1_prediction_error.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────
# FIGURE 2 — Adaptive Horizon  k_eff vs Speed
# ─────────────────────────────────────────────

def fig_adaptive_horizon(results):
    apply_dark_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig.suptitle("Adaptive Prediction Horizon: k_eff vs. Target Speed",
                 fontsize=15, fontweight="bold", color=TEXT, y=1.01)

    # ── left: scatter all categories ────────────────────────────────────────
    ax = axes[0]
    for cat in results:
        df = results[cat]
        ax.scatter(df["speed"], df["k_eff"],
                   s=12, alpha=0.35,
                   color=PALETTE[cat],
                   label=CATEGORIES[cat].replace("\n", " "),
                   zorder=3)

    # overlay theoretical curve
    speeds = np.linspace(0, 350, 500)
    k_theory = np.array([adaptive_k(s) for s in speeds])
    ax.plot(speeds, k_theory, color=ACCENT,
            linewidth=2.5, zorder=5, label="Algorithm curve")

    # speed regime bands
    for lo, hi, lbl in [(0, 10, "Stationary"), (10, 50, "Slow"),
                        (50, 150, "Moderate"), (150, 350, "Fast")]:
        ax.axvspan(lo, hi, alpha=0.06, color=ACCENT)
        ax.text((lo + hi) / 2, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 40,
                lbl, ha="center", fontsize=7, color=TEXT, alpha=0.5)

    ax.set_xlabel("Speed (px/frame)")
    ax.set_ylabel("Effective Horizon k_eff (frames)")
    ax.set_title("k_eff vs. Speed (all categories)", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.4)

    # ── right: mean k_eff per state per category ─────────────────────────────
    ax2 = axes[1]
    states = [STATE_HOVERING, STATE_CRUISING, STATE_ACCELERATING]
    state_labels = ["Hovering", "Cruising", "Accelerating"]
    x_pos = np.arange(len(states))
    width = 0.15
    cat_list = list(results.keys())

    for j, cat in enumerate(cat_list):
        df = results[cat]
        means = []
        for st in states:
            sub = df[df["state"] == st]["k_eff"]
            means.append(sub.mean() if not sub.empty else 0)
        offset = (j - len(cat_list) / 2) * width + width / 2
        bars = ax2.bar(x_pos + offset, means, width * 0.9,
                       color=PALETTE[cat], alpha=0.85,
                       label=CATEGORIES[cat].replace("\n", " "))

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(state_labels)
    ax2.set_ylabel("Mean k_eff (frames)")
    ax2.set_title("Mean Horizon by Motion State & Category", fontsize=11)
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(axis="y", alpha=0.4)

    fig.tight_layout()
    out = OUT_DIR / "fig2_adaptive_horizon.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────
# FIGURE 3 — Hit / Miss Rate
# ─────────────────────────────────────────────

def fig_hit_rate(results):
    apply_dark_style()

    cats     = list(results.keys())
    hit_rates  = []
    shot_counts = []

    for cat in cats:
        df = results[cat]
        total = len(df)
        hits  = df["hit"].sum() if "hit" in df.columns and total > 0 else 0
        hit_rates.append(hits / total if total > 0 else 0)
        shot_counts.append(total)

    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    fig.suptitle("Hit Confirmation Rate by Scenario Category",
                 fontsize=16, fontweight="bold", color=TEXT, y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.55, wspace=0.4,
                           top=0.91, bottom=0.08)

    # ── top-left: grouped bar (hit vs miss) ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    x   = np.arange(len(cats))
    w   = 0.35

    hit_bars  = ax1.bar(x - w/2, [r * 100 for r in hit_rates],
                        w, color=ACCENT, alpha=0.85, label="Hit %")
    miss_bars = ax1.bar(x + w/2, [(1 - r) * 100 for r in hit_rates],
                        w, color="#FF6B6B", alpha=0.85, label="Miss %")

    for bar, r in zip(hit_bars, hit_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 f"{r*100:.1f}%", ha="center", va="bottom",
                 fontsize=9, color=TEXT, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels([CATEGORIES[c].replace("\n", " ") for c in cats],
                        fontsize=9)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_ylim(0, 115)
    ax1.set_title("Hit vs. Miss Rate per Category", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.4)

    # ── top-right: donut (overall) ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    total_shots = sum(shot_counts)
    total_hits  = sum(int(r * n) for r, n in zip(hit_rates, shot_counts))
    total_misses = total_shots - total_hits

    wedge_sizes = [total_hits, total_misses]
    wedge_cols  = [ACCENT, "#FF6B6B"]
    wedge_labels = [f"Hits\n{total_hits}", f"Misses\n{total_misses}"]

    wedges, texts = ax2.pie(
        wedge_sizes, colors=wedge_cols,
        startangle=90, counterclock=False,
        wedgeprops=dict(width=0.52, edgecolor=BG, linewidth=2)
    )
    ax2.text(0, 0,
             f"{total_hits/total_shots*100:.1f}%\nhit rate",
             ha="center", va="center",
             fontsize=13, fontweight="bold", color=TEXT)
    ax2.legend(wedges, wedge_labels, loc="lower center",
               fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax2.set_title("Overall Hit Rate", fontsize=11)

    # ── bottom row: per-category breakdown by state ──────────────────────────
    for i, cat in enumerate(cats[:3]):     # first 3 cats in bottom row
        ax = fig.add_subplot(gs[1, i])
        df = results[cat]
        states = [STATE_HOVERING, STATE_CRUISING, STATE_ACCELERATING]
        s_labels = ["Hover", "Cruise", "Accel"]
        s_hits  = []
        s_total = []
        for st in states:
            sub = df[df["state"] == st]
            s_total.append(len(sub))
            s_hits.append(sub["hit"].sum() if not sub.empty else 0)

        rates = [h/t*100 if t > 0 else 0 for h, t in zip(s_hits, s_total)]
        bar_cols = [STATE_COLORS.get(s, ACCENT) for s in states]
        bars = ax.bar(s_labels, rates, color=bar_cols, alpha=0.85)
        for bar, r, t in zip(bars, rates, s_total):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{r:.0f}%\n(n={t})",
                    ha="center", va="bottom", fontsize=7.5, color=TEXT)
        ax.set_ylim(0, 120)
        ax.set_title(CATEGORIES[cat], fontsize=10)
        ax.set_ylabel("Hit Rate (%)" if i == 0 else "")
        ax.grid(axis="y", alpha=0.4)
        ax.axhline(50, color=GRID, linestyle=":", linewidth=1)

    # remaining cats: show as small text summary in remaining slots (if any)
    remaining = cats[3:]
    for i, cat in enumerate(remaining):
        ax = fig.add_subplot(gs[1, i + 3] if i + 3 < 3 else gs[1, 2])
        # handled above; skip

    out = OUT_DIR / "fig3_hit_rate.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────
# FIGURE 4 — Summary Dashboard  (poster-ready)
# ─────────────────────────────────────────────

def fig_summary_dashboard(results, all_sim):
    """Single-page summary figure suitable for the poster itself."""
    apply_dark_style()

    fig = plt.figure(figsize=(20, 11), facecolor=BG)
    fig.suptitle("Adaptive Predictive Tracking & Prioritization — Performance Summary",
                 fontsize=18, fontweight="bold", color=TEXT, y=0.99)

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.50, wspace=0.40,
                           top=0.92, bottom=0.07,
                           left=0.06, right=0.97)

    cats = list(results.keys())

    # ── [0,0] Prediction error box per category ──────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    medians  = []
    means    = []
    cat_lbls = []
    for cat in cats:
        data = results[cat]["pred_error_px"].dropna()
        medians.append(np.median(data) if len(data) else 0)
        means.append(np.mean(data) if len(data) else 0)
        cat_lbls.append(CATEGORIES[cat].replace("\n", " "))

    x = np.arange(len(cats))
    ax0.bar(x, medians, color=[PALETTE[c] for c in cats], alpha=0.85)
    ax0.scatter(x, means, marker="D", color=TEXT,
                s=40, zorder=5, label="Mean")
    ax0.axhline(SPOT_RADIUS_PX_SAFE, color="#FF6B6B",
                linestyle="--", linewidth=1.5, label="Spot radius")
    ax0.set_xticks(x)
    ax0.set_xticklabels(cat_lbls, fontsize=7, rotation=20, ha="right")
    ax0.set_ylabel("Error (px)")
    ax0.set_title("Median Prediction Error", fontsize=10)
    ax0.legend(fontsize=7)
    ax0.grid(axis="y", alpha=0.4)

    # ── [0,1] CDF ────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    for cat in cats:
        data = results[cat]["pred_error_px"].dropna().sort_values()
        if data.empty:
            continue
        cdf = np.arange(1, len(data) + 1) / len(data)
        ax1.plot(data, cdf, color=PALETTE[cat], linewidth=2,
                 label=CATEGORIES[cat].replace("\n", " "))
    ax1.axvline(SPOT_RADIUS_PX_SAFE, color="#FF6B6B",
                linestyle="--", linewidth=1.2)
    ax1.set_xlabel("Prediction Error (px)", fontsize=9)
    ax1.set_ylabel("CDF", fontsize=9)
    ax1.set_title("Prediction Error CDF", fontsize=10)
    ax1.legend(fontsize=6.5, ncol=1)
    ax1.grid(alpha=0.4)
    ax1.set_xlim(left=0)

    # ── [0,2] k_eff vs speed scatter (all data) ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    for cat in cats:
        df = results[cat]
        ax2.scatter(df["speed"], df["k_eff"],
                    s=8, alpha=0.3, color=PALETTE[cat])

    speeds = np.linspace(0, 350, 500)
    ax2.plot(speeds, [adaptive_k(s) for s in speeds],
             color=ACCENT, linewidth=2.5, label="Algorithm")
    ax2.set_xlabel("Speed (px/frame)", fontsize=9)
    ax2.set_ylabel("k_eff (frames)", fontsize=9)
    ax2.set_title("Adaptive Horizon vs. Speed", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.4)

    # ── [0,3] Hit rate donut + bar ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 3])
    hit_rates = []
    for cat in cats:
        df = results[cat]
        total = len(df)
        hits  = df["hit"].sum() if total > 0 else 0
        hit_rates.append(hits / total if total > 0 else 0)

    ax3.barh([CATEGORIES[c].replace("\n", " ") for c in cats],
             [r * 100 for r in hit_rates],
             color=[PALETTE[c] for c in cats], alpha=0.85)
    ax3.axvline(50, color=GRID, linestyle=":", linewidth=1)
    ax3.set_xlim(0, 105)
    ax3.set_xlabel("Hit Rate (%)", fontsize=9)
    ax3.set_title("Hit Rate by Category", fontsize=10)
    for i, (r, cat) in enumerate(zip(hit_rates, cats)):
        ax3.text(r * 100 + 1, i, f"{r*100:.1f}%",
                 va="center", fontsize=8, color=TEXT)
    ax3.grid(axis="x", alpha=0.4)

    # ── bottom row: per-state breakdown across categories ────────────────────
    states    = [STATE_HOVERING, STATE_CRUISING, STATE_ACCELERATING]
    s_labels  = ["Hovering", "Cruising", "Accelerating"]

    # [1,0] Mean prediction error by state
    ax4 = fig.add_subplot(gs[1, 0])
    for cat in cats:
        df = results[cat]
        means_s = []
        for st in states:
            sub = df[df["state"] == st]["pred_error_px"].dropna()
            means_s.append(sub.mean() if not sub.empty else 0)
        ax4.plot(s_labels, means_s, marker="o", linewidth=2,
                 color=PALETTE[cat],
                 label=CATEGORIES[cat].replace("\n", " "))
    ax4.axhline(SPOT_RADIUS_PX_SAFE, color="#FF6B6B",
                linestyle="--", linewidth=1.2)
    ax4.set_ylabel("Mean Error (px)", fontsize=9)
    ax4.set_title("Pred. Error by Motion State", fontsize=10)
    ax4.legend(fontsize=6.5)
    ax4.grid(alpha=0.4)

    # [1,1] mean k_eff by state
    ax5 = fig.add_subplot(gs[1, 1])
    for cat in cats:
        df = results[cat]
        means_k = []
        for st in states:
            sub = df[df["state"] == st]["k_eff"]
            means_k.append(sub.mean() if not sub.empty else 0)
        ax5.plot(s_labels, means_k, marker="s", linewidth=2,
                 color=PALETTE[cat])
    ax5.set_ylabel("Mean k_eff (frames)", fontsize=9)
    ax5.set_title("Adaptive Horizon by Motion State", fontsize=10)
    ax5.grid(alpha=0.4)

    # [1,2] hit rate by state across categories (grouped bar)
    ax6 = fig.add_subplot(gs[1, 2])
    x = np.arange(len(states))
    w = 0.15
    for j, cat in enumerate(cats):
        df = results[cat]
        hr_s = []
        for st in states:
            sub = df[df["state"] == st]
            hr_s.append(sub["hit"].mean() * 100 if not sub.empty else 0)
        offset = (j - len(cats) / 2) * w + w / 2
        ax6.bar(x + offset, hr_s, w * 0.9,
                color=PALETTE[cat], alpha=0.85)
    ax6.set_xticks(x)
    ax6.set_xticklabels(s_labels)
    ax6.set_ylabel("Hit Rate (%)", fontsize=9)
    ax6.set_title("Hit Rate by Motion State", fontsize=10)
    ax6.grid(axis="y", alpha=0.4)
    ax6.set_ylim(0, 115)

    # [1,3] score distribution violin
    ax7 = fig.add_subplot(gs[1, 3])
    data_list = [results[cat]["score"].dropna().values for cat in cats]
    data_list = [d for d in data_list if len(d) > 0]
    if data_list:
        parts = ax7.violinplot(data_list,
                               positions=range(len(data_list)),
                               showmedians=True, showextrema=False)
        for pc, cat in zip(parts["bodies"], cats):
            pc.set_facecolor(PALETTE[cat])
            pc.set_alpha(0.75)
        parts["cmedians"].set_color(TEXT)
    ax7.set_xticks(range(len(cats)))
    ax7.set_xticklabels([CATEGORIES[c].replace("\n", " ") for c in cats],
                        fontsize=7, rotation=20, ha="right")
    ax7.set_ylabel("Priority Score", fontsize=9)
    ax7.set_title("Score Distribution", fontsize=10)
    ax7.grid(axis="y", alpha=0.4)

    # legend strip at bottom
    legend_elements = [Line2D([0], [0], color=PALETTE[cat], linewidth=3,
                               label=CATEGORIES[cat].replace("\n", " "))
                       for cat in cats]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=len(cats), fontsize=9,
               frameon=True, facecolor=PANEL, edgecolor=GRID,
               bbox_to_anchor=(0.5, 0.0))

    out = OUT_DIR / "fig4_summary_dashboard.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved {out}")


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


def plot_prediction_error_poster(results):

    apply_poster_style()

    fig,ax = plt.subplots(figsize=(5,4))

    cats=list(results.keys())

    medians=[]
    q1=[]
    q3=[]
    labels=[]

    for cat in cats:

        data=results[cat]["pred_error_px"].dropna()

        medians.append(data.median() if len(data)>0 else 0)

        q1.append(data.quantile(.25) if len(data)>0 else 0)

        q3.append(data.quantile(.75) if len(data)>0 else 0)

        labels.append(CATEGORIES[cat].replace("\n"," "))


    x=np.arange(len(cats))

    ax.bar(
        x,
        medians,
        color=[PALETTE[c] for c in cats],
        width=.65
    )

    ax.errorbar(

        x,

        medians,

        yerr=[

            np.array(medians)-np.array(q1),

            np.array(q3)-np.array(medians)

        ],

        fmt='none',

        color='black',

        capsize=5

    )

    ax.axhline(

        SPOT_RADIUS_PX_SAFE,

        linestyle="--",

        color="red",

        label="Laser radius"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(

        labels,

        rotation=25,

        ha='right'
    )

    ax.set_ylabel("Prediction error (pixels)")

    ax.set_title("Median Prediction Error by Scenario")

    ax.grid(axis='y')

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUT_DIR/"poster_prediction_error.pdf",
        bbox_inches="tight"
    )

    fig.savefig(
        OUT_DIR/"poster_prediction_error.svg",
        bbox_inches="tight"
    )

    plt.close()


def plot_hit_rate_poster(results):

    apply_poster_style()

    fig,ax=plt.subplots(figsize=(5,4))

    cats=list(results.keys())

    rates=[]
    shots=[]
    labels=[]

    for cat in cats:

        df=results[cat]

        total=len(df)

        hits=df["hit"].sum() if total>0 else 0

        rates.append(hits/total*100 if total>0 else 0)

        shots.append(total)

        labels.append(CATEGORIES[cat].replace("\n"," "))


    x=np.arange(len(cats))

    bars=ax.bar(

        x,

        rates,

        color=[PALETTE[c] for c in cats],

        width=.65
    )

    for bar,r,n in zip(bars,rates,shots):

        ax.text(

            bar.get_x()+bar.get_width()/2,

            bar.get_height()+1,

            f"{r:.1f}%\n(n={n})",

            ha='center',

            fontsize=11
        )


    ax.set_ylabel("Hit rate (%)")

    ax.set_xticks(x)

    ax.set_xticklabels(

        labels,

        rotation=25,

        ha='right'
    )

    ax.set_ylim(0,105)

    ax.set_title("Hit Rate by Scenario")

    ax.grid(axis='y')

    fig.tight_layout()

    fig.savefig(
        OUT_DIR/"poster_hit_rate.pdf",
        bbox_inches="tight"
    )

    fig.savefig(
        OUT_DIR/"poster_hit_rate.svg",
        bbox_inches="tight"
    )

    plt.close()

def plot_horizon_vs_speed_poster(results):

    apply_poster_style()

    fig,ax=plt.subplots(figsize=(5,4))

    all_df=pd.concat(results.values())

    # separate hits and misses
    hits=all_df[all_df["hit"]==True]
    miss=all_df[all_df["hit"]==False]

    # misses first (background)
    ax.scatter(

        miss["speed"],

        miss["k_eff"],

        s=28,

        color="#D55E00",

        alpha=.6,

        label="Miss"
    )

    # hits on top
    ax.scatter(

        hits["speed"],

        hits["k_eff"],

        s=28,

        color="#009E73",

        alpha=.7,

        label="Hit"
    )

    # thin model curve (reference only)
    speeds=np.linspace(0,350,400)

    ax.plot(

        speeds,

        [adaptive_k(s) for s in speeds],

        color="black",

        linewidth=1.5,

        linestyle="--",

        alpha=.7,

        label="Adaptive model"
    )

    # speed regime regions with legend labels
    ax.axvspan(0,10,alpha=.07,color="gray",label="Stationary")

    ax.axvspan(10,50,alpha=.05,color="green",label="Slow")

    ax.axvspan(50,150,alpha=.05,color="gold",label="Moderate")

    ax.axvspan(150,350,alpha=.05,color="red",label="Fast")

    ax.set_xlabel("Target speed (pixels/frame)")

    ax.set_ylabel("Prediction horizon (frames)")

    ax.set_title("Adaptive Prediction Horizon vs Speed")

    ax.grid(alpha=.3)

    # clean legend (no duplicates)
    handles,labels=ax.get_legend_handles_labels()

    by_label=dict(zip(labels,handles))

    ax.legend(

        by_label.values(),

        by_label.keys(),

        fontsize=9,

        loc="upper left"
    )

    fig.savefig(

        OUT_DIR/"poster_adaptive_horizon.pdf",

        bbox_inches="tight"
    )

    fig.savefig(

        OUT_DIR/"poster_adaptive_horizon.svg",

        bbox_inches="tight"
    )

    plt.close()

def plot_adaptive_design_validation(results):

    apply_poster_style()

    fig,ax=plt.subplots(figsize=(5,4))

    for label,df in results.items():

        df=df.copy()

        # remove invalid rows
        print(label)
        print(df.columns)
        df=df.dropna(subset=["speed","pred_error_px"])

        if len(df)==0:
            continue

        smin=df["speed"].min()
        smax=df["speed"].max()

        # handle degenerate case
        if smax == smin:

            ax.scatter(

                df["speed"],

                df["pred_error_px"],

                s=25,

                alpha=.6,

                label=f"{label} (constant speed)"
            )

            continue

        # create bins safely
        bins=np.linspace(smin,smax,15)

        df["speed_bin"]=pd.cut(

            df["speed"],

            bins=bins,

            duplicates="drop"
        )

        med=df.groupby(
            "speed_bin",
            observed=True
        )["pred_error_px"].median()

        centers=[b.mid for b in med.index]

        ax.plot(

            centers,

            med,

            linewidth=2.5,

            marker="o",

            markersize=5,

            label=label
        )

    ax.set_xlabel(
        "Target speed (pixels/frame)"
    )

    ax.set_ylabel(
        "Median prediction error (pixels)"
    )

    ax.set_title(
        "Adaptive Prediction Horizon Improves Accuracy Across Target Speeds"
    )

    ax.grid(alpha=.3)

    ax.legend()

    fig.tight_layout()

    fig.savefig(

        OUT_DIR/"poster_adaptive_design.pdf",

        bbox_inches="tight"
    )

    plt.close()




# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=== Loading CSVs ===")
    dfs = load_csvs()

    if not dfs:
        print("\n[ERROR] No CSV files found. "
              f"Place them in: {CSV_DIR.resolve()}\n"
              "Expected names: multi_slow.csv, single_slow.csv, "
              "multi_fast.csv, single_fast.csv, multi_mix.csv")
        return

    print("\n=== Running simulations ===")
    results = {}
    for cat, df in dfs.items():
        print(f"  Simulating {cat} ...", end=" ")
        sim = simulate_hits(df)
        results[cat] = sim
        n = len(sim)
        hits = sim["hit"].sum() if n > 0 else 0
        print(f"{n} shots simulated, {hits} hits "
              f"({hits/n*100:.1f}% hit rate)" if n > 0 else "0 shots")

    # combine all simulations for dashboard
    all_sim = pd.concat(
        [df.assign(category=cat) for cat, df in results.items()],
        ignore_index=True
    )

    print("\n=== Generating figures ===")
    #fig_prediction_error(results)
    #fig_adaptive_horizon(results)
    #fig_hit_rate(results)
    #fig_summary_dashboard(results, all_sim)
    print("Generating poster plots...")

    #plot_prediction_error_poster(results)

    #plot_hit_rate_poster(results)

    #plot_horizon_vs_speed_poster(results)
    plot_adaptive_design_validation(results)

    print("\n=== Exporting data ===")
    #export_results(results, all_sim)

    print("\n=== Advanced metrics ===")
    #export_advanced_metrics(results,all_sim)


    """ print(f"\n✓ All figures saved to: {OUT_DIR.resolve()}/")
    print("  fig1_prediction_error.png")
    print("  fig2_adaptive_horizon.png")
    print("  fig3_hit_rate.png")
    print("  fig4_summary_dashboard.png  ← main poster panel") """


if __name__ == "__main__":
    main()