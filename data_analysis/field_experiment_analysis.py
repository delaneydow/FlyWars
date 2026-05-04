# ==========================================
# FlyWars Statistical Analysis Pipeline
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf

# ------------------------------------------
# 1. PLOT STYLING (publication quality)
# ------------------------------------------
sns.set_theme(style="whitegrid", context="talk")

plt.rcParams.update({
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "legend.frameon": True
})

# ------------------------------------------
# 2. LOAD DATA
# ------------------------------------------
# Expected columns:
# trial_id, condition, time_min, trap_count, fallen_count

df = pd.read_csv("data_analysis/field_experiments.csv")

# Convert time → hours
df["time_hr"] = df["time_min"] / 60

# ------------------------------------------
# 3. COMPUTE RATES
# ------------------------------------------
df["RAI"] = df["trap_count"] / df["time_hr"]
df["LER"] = df["fallen_count"] / df["time_hr"]
df["combined_rate"] = (df["trap_count"] + df["fallen_count"]) / df["time_hr"]

# Use combined rate as primary metric
df["rate"] = df["combined_rate"]

# ------------------------------------------
# 4. WITHIN-TRIAL MATCHING (KEY STEP)
# ------------------------------------------
baseline = df[df["condition"] == "baseline"]
intervention = df[df["condition"] == "intervention"]

merged = pd.merge(
    baseline,
    intervention,
    on=["trial_id", "time_hr"],
    suffixes=("_base", "_int")
)

# Difference at each matched timepoint
merged["diff"] = merged["rate_int"] - merged["rate_base"]

# ------------------------------------------
# 5. TRIAL-LEVEL EFFECT (PRIMARY ANALYSIS)
# ------------------------------------------
trial_effects = merged.groupby("trial_id")["diff"].mean()

# Normality test
stat, p_normal = stats.shapiro(trial_effects)

if p_normal > 0.05:
    test_name = "One-sample t-test"
    stat, p_value = stats.ttest_1samp(trial_effects, 0)
else:
    test_name = "Wilcoxon signed-rank"
    stat, p_value = stats.wilcoxon(trial_effects)

# Effect size
cohen_d = trial_effects.mean() / trial_effects.std()

# Bootstrap CI
boot_means = []
for _ in range(5000):
    sample = np.random.choice(trial_effects, size=len(trial_effects), replace=True)
    boot_means.append(np.mean(sample))

ci_lower = np.percentile(boot_means, 2.5)
ci_upper = np.percentile(boot_means, 97.5)

print("\n==========================================")
print("WITHIN-TRIAL EFFECT ANALYSIS")
print("==========================================")
print(f"Test used: {test_name}")
print(f"Mean effect (Intervention - Baseline): {trial_effects.mean():.3f}")
print(f"P-value: {p_value:.5f}")
print(f"Cohen's d: {cohen_d:.3f}")
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
print("==========================================\n")

# ------------------------------------------
# 6. MIXED EFFECTS MODEL (STRONGEST ANALYSIS)
# ------------------------------------------
model = smf.mixedlm(
    "rate ~ condition * time_hr",
    df,
    groups=df["trial_id"]
).fit()

print(model.summary())

print("\n==========================================")
print("WITHIN-TRIAL TIMEPOINT ANALYSIS")
print("==========================================")

trial_results = []

for trial in merged["trial_id"].unique():
    sub = merged[merged["trial_id"] == trial]
    
    diffs = sub["diff"].values
    
    # Wilcoxon test (paired across timepoints)
    if len(diffs) >= 3:
        try:
            stat, p = stats.wilcoxon(diffs)
        except:
            p = np.nan
    else:
        p = np.nan
    
    # Sign test (very robust)
    n_pos = np.sum(diffs > 0)
    n_total = len(diffs)
    
    # Binomial sign test
    p_sign = stats.binomtest(n_pos, n_total, 0.5, alternative='greater')
    
    print(f"\nTrial {trial}:")
    print(f"  Mean diff: {np.mean(diffs):.2f}")
    print(f"  Positive timepoints: {n_pos}/{n_total}")
    print(f"  Wilcoxon p: {p:.4f}")
    print(f"  Sign test p: {p_sign}")
    
    trial_results.append({
        "trial": trial,
        "mean_diff": np.mean(diffs),
        "wilcoxon_p": p,
        "sign_p": p_sign
    })

trial_results = pd.DataFrame(trial_results)

# ------------------------------------------
# 7. FIGURE 1 — TIME SERIES (Mean ± SEM)
# ------------------------------------------
plt.figure(figsize=(10,6))

sns.lineplot(
    data=df,
    x="time_hr",
    y="rate",
    hue="condition",
    estimator="mean",
    errorbar="se",
    linewidth=3
)

plt.xlabel("Time (hours)")
plt.ylabel("Rate (flies/hour)")
plt.title("Fly Capture and Elimination Rates Over Time")
plt.legend(title="Condition")

plt.tight_layout()
plt.savefig("figure_time_series.png", dpi=300)
plt.show()

# ------------------------------------------
# 8. FIGURE 2 — WITHIN-TRIAL EFFECT TRAJECTORIES
# ------------------------------------------
plt.figure(figsize=(10,6))

for trial in merged["trial_id"].unique():
    sub = merged[merged["trial_id"] == trial]
    plt.plot(sub["time_hr"], sub["diff"], marker='o', alpha=0.6)

plt.axhline(0, linestyle="--", linewidth=2)

plt.xlabel("Time (hours)")
plt.ylabel("Intervention - Baseline (flies/hour)")
plt.title("Within-Trial Effect Over Time")

plt.tight_layout()
plt.savefig("figure_trial_effects.png", dpi=300)
plt.show()

# ------------------------------------------
# 9. FIGURE 3 — PAIRED TRIAL SUMMARY
# ------------------------------------------
plt.figure(figsize=(6,6))

baseline_means = merged.groupby("trial_id")["rate_base"].mean()
intervention_means = merged.groupby("trial_id")["rate_int"].mean()

for i in range(len(baseline_means)):
    plt.plot(
        [0,1],
        [baseline_means.iloc[i], intervention_means.iloc[i]],
        marker='o'
    )

plt.xticks([0,1], ["Baseline", "Intervention"])
plt.ylabel("Rate (flies/hour)")
plt.title("Paired Trial Comparison")

plt.tight_layout()
plt.savefig("figure_paired_trials.png", dpi=300)
plt.show()

plt.figure(figsize=(10,6))

sns.stripplot(
    data=merged,
    x="trial_id",
    y="diff",
    jitter=True,
    size=8
)

sns.boxplot(
    data=merged,
    x="trial_id",
    y="diff",
    showcaps=False,
    boxprops={'facecolor':'none'},
    showfliers=False,
    whiskerprops={'linewidth':0}
)

plt.axhline(0, linestyle="--", linewidth=2)

plt.xlabel("Trial ID")
plt.ylabel("Intervention - Baseline (flies/hour)")
plt.title("Within-Trial Timepoint Differences")

plt.tight_layout()
plt.savefig("figure_timepoint_differences.png", dpi=300)
plt.show()


# ------------------------------------------
# 10. OPTIONAL: SAVE RESULTS
# ------------------------------------------
trial_effects.to_csv("trial_effects.csv")

print("Analysis complete. Figures and results saved.")
