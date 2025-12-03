"""
raw_baseline.py

Baseline CONSORT performance analysis using raw actual scores (not percentages).
FILTERED TO: Articles WITH full text available (PMC full text only)

Analyzes:
1. Overall CONSORT performance by version (raw scores)
2. Item-level improvements: 250-word vs Original (raw scores)
3. Item-level improvements: 300-word vs Original (raw scores)
4. Relationship between total score and readability metrics

Outputs:
- CSV files with raw baseline scores and improvements
- Bar charts and visualizations
- Scatter plots for score vs readability analysis
"""

import os
import json
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Configuration ---------------- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

ORIGINAL_PATH = os.path.join(PROJECT_ROOT, "data", "scores", "original")
ORIGINAL_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "original")  # For full text checking
REWRITTEN_250_PATH = os.path.join(PROJECT_ROOT, "data", "scores", "rewritten", "250")
REWRITTEN_300_PATH = os.path.join(PROJECT_ROOT, "data", "scores", "rewritten", "300")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Protocol v3.0 rubric (14 items, max 25 points)
RUBRIC_MAX = {
    "Title": 1,
    "Trial_Design": 2,
    "Participants": 2,
    "Intervention": 2,
    "Objective": 1,
    "Method_Outcome": 2,
    "Randomisation_Allocation": 2,
    "Blinding": 3,
    "Number_Randomised": 2,
    "Number_Analysed": 2,
    "Result_Outcome": 3,
    "Harms": 1,
    "Trial_Registration": 1,
    "Funding": 1,
}
ITEM_ORDER = list(RUBRIC_MAX.keys())
TOTAL_MAX_SCORE = sum(RUBRIC_MAX.values())  # 25


# ---------------- Utility Functions ---------------- #

def get_full_text_status(pmcid):
    """Check if article has full text available"""
    filepath = os.path.join(ORIGINAL_DATA_PATH, f"{pmcid}.json")
    if not os.path.exists(filepath):
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    data_source = data.get("data_source", "")
    full_text = data.get("full_text", "")

    # Has full text if NOT pubmed_fallback AND has substantial full text
    has_full_text = (data_source != "pubmed_fallback" and
                     bool(full_text and len(full_text.strip()) > 100))

    return has_full_text


def _safe_float(x, default=0.0):
    """Convert to float safely"""
    try:
        return float(x)
    except Exception:
        return default


def load_scores_with_readability(folder_path, version_label):
    """Load CONSORT scores AND readability metrics from folder"""
    records = []

    for fn in os.listdir(folder_path):
        if not fn.endswith(".json"):
            continue

        with open(os.path.join(folder_path, fn), "r", encoding="utf-8") as f:
            data = json.load(f)

        # Clean PMCID from filename
        pmcid = fn.replace(".json", "").replace("_scores", "").replace("_250", "").replace("_300", "")

        # Extract CONSORT scores
        cons = data.get("consort_scores", {})
        if not isinstance(cons, dict) or "items" not in cons:
            continue

        record = {
            "pmcid": pmcid,
            "version": version_label,
            "total_score": _safe_float(cons.get("total_score", 0)),
        }

        # Extract per-item scores
        items = cons.get("items", {})
        for item_name in ITEM_ORDER:
            score = items.get(item_name, {}).get("score", 0)
            record[item_name] = _safe_float(score, 0.0)

        # Extract readability metrics
        readability = data.get("readability", {})
        record["flesch_kincaid"] = _safe_float(readability.get("flesch_kincaid"), np.nan)
        record["reading_ease"] = _safe_float(readability.get("reading_ease"), np.nan)
        record["word_count"] = _safe_float(readability.get("word_count"), np.nan)

        # Add full text status
        record["has_full_text"] = get_full_text_status(pmcid)

        records.append(record)

    return pd.DataFrame(records) if records else pd.DataFrame()


def calculate_baseline_stats(df, version_label):
    """Calculate baseline statistics using raw scores"""
    stats = {
        "version": version_label,
        "n": len(df),
        "total_mean": df["total_score"].mean(),
        "total_sd": df["total_score"].std(),
        "total_median": df["total_score"].median(),
        "total_min": df["total_score"].min(),
        "total_max": df["total_score"].max(),
    }

    # Per-item means
    for item in ITEM_ORDER:
        if item in df.columns:
            stats[f"{item}_mean"] = df[item].mean()
            stats[f"{item}_sd"] = df[item].std()

    return stats


def create_paired_dataset(df_orig, df_rewritten, suffix_orig, suffix_rewrite):
    """Create paired dataset matching by PMCID"""
    common = set(df_orig["pmcid"]).intersection(set(df_rewritten["pmcid"]))

    if not common:
        return pd.DataFrame()

    df_orig_matched = df_orig[df_orig["pmcid"].isin(common)].sort_values("pmcid").reset_index(drop=True)
    df_rewrite_matched = df_rewritten[df_rewritten["pmcid"].isin(common)].sort_values("pmcid").reset_index(drop=True)

    # Rename columns
    df_orig_renamed = df_orig_matched.add_suffix(f"_{suffix_orig}")
    df_rewrite_renamed = df_rewrite_matched.add_suffix(f"_{suffix_rewrite}")

    # Merge
    paired = pd.concat([df_orig_renamed, df_rewrite_renamed], axis=1)
    paired["pmcid"] = df_orig_matched["pmcid"]

    return paired


def calculate_item_improvements_raw(paired_df, suffix_orig, suffix_rewrite):
    """Calculate item-level improvements using raw scores"""
    improvements = []

    for item in ITEM_ORDER:
        col_orig = f"{item}_{suffix_orig}"
        col_rewrite = f"{item}_{suffix_rewrite}"

        if col_orig not in paired_df.columns or col_rewrite not in paired_df.columns:
            continue

        orig_scores = paired_df[col_orig]
        rewrite_scores = paired_df[col_rewrite]

        # Calculate differences
        diff = rewrite_scores - orig_scores

        # Statistics
        improvements.append({
            "item": item,
            "item_max": RUBRIC_MAX[item],
            "n_pairs": len(diff),
            "orig_mean": orig_scores.mean(),
            "orig_sd": orig_scores.std(),
            "rewrite_mean": rewrite_scores.mean(),
            "rewrite_sd": rewrite_scores.std(),
            "mean_diff": diff.mean(),
            "sd_diff": diff.std(),
            "median_diff": diff.median(),
            "n_improved": (diff > 0).sum(),
            "n_declined": (diff < 0).sum(),
            "n_unchanged": (diff == 0).sum(),
        })

    return pd.DataFrame(improvements)


# ---------------- Load Data ---------------- #

print("=" * 80)
print("RAW BASELINE ANALYSIS - Protocol v3.0")
print("FILTERED TO: Articles WITH full text available (PMC full text only)")
print("=" * 80)

print("\n📥 Loading scores with readability metrics...")
df_orig = load_scores_with_readability(ORIGINAL_PATH, "Original")
df_250 = load_scores_with_readability(REWRITTEN_250_PATH, "250-word")
df_300 = load_scores_with_readability(REWRITTEN_300_PATH, "300-word")

print(f"   Original abstracts: {len(df_orig)}")
print(f"   250-word rewrites: {len(df_250)}")
print(f"   300-word rewrites: {len(df_300)}")

# Filter to only WITH full text
df_orig = df_orig.dropna(subset=["has_full_text"])
df_250 = df_250.dropna(subset=["has_full_text"])
df_300 = df_300.dropna(subset=["has_full_text"])

print(f"\nAfter removing records with unknown full text status:")
print(f"   Original: {len(df_orig)}")
print(f"   250-word: {len(df_250)}")
print(f"   300-word: {len(df_300)}")

# Filter to WITH full text only
df_orig = df_orig[df_orig["has_full_text"] == True]
df_250 = df_250[df_250["has_full_text"] == True]
df_300 = df_300[df_300["has_full_text"] == True]

print(f"\n✅ Filtered to WITH full text only:")
print(f"   Original: {len(df_orig)}")
print(f"   250-word: {len(df_250)}")
print(f"   300-word: {len(df_300)}")

# ---------------- PART 1: Overall Baseline Performance ---------------- #

print("\n" + "=" * 80)
print("PART 1: OVERALL BASELINE CONSORT PERFORMANCE (RAW SCORES)")
print("=" * 80)

baseline_stats = []
for df, label in [(df_orig, "Original"), (df_250, "250-word"), (df_300, "300-word")]:
    if not df.empty:
        stats = calculate_baseline_stats(df, label)
        baseline_stats.append(stats)

df_baseline = pd.DataFrame(baseline_stats)

# Save CSV
csv_baseline = os.path.join(OUTPUT_DIR, "baseline_raw_scores.csv")
df_baseline.to_csv(csv_baseline, index=False)
print(f"\n📄 Saved: {csv_baseline}")

# Print summary
print("\nOverall CONSORT Performance (Raw Scores):")
print(f"{'Version':<15} {'N':<6} {'Mean':<8} {'SD':<8} {'Median':<8} {'Min':<6} {'Max':<6}")
print("-" * 65)
for _, row in df_baseline.iterrows():
    print(f"{row['version']:<15} {row['n']:<6.0f} {row['total_mean']:<8.2f} {row['total_sd']:<8.2f} "
          f"{row['total_median']:<8.2f} {row['total_min']:<6.1f} {row['total_max']:<6.1f}")

# Visualization 1: Overall Performance Bar Chart
if baseline_stats:
    fig, ax = plt.subplots(figsize=(10, 6))

    versions = [s["version"] for s in baseline_stats]
    means = [s["total_mean"] for s in baseline_stats]
    sds = [s["total_sd"] for s in baseline_stats]

    colors = ["steelblue", "orange", "green"]
    bars = ax.bar(versions, means, yerr=sds, capsize=5, color=colors, alpha=0.7, edgecolor="black")

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{mean:.2f}',
                ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel("Total CONSORT Score (Raw)", fontsize=12, fontweight="bold")
    ax.set_title(f"Overall CONSORT Performance by Version (WITH Full Text Only)\n(Max Score = {TOTAL_MAX_SCORE})",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, TOTAL_MAX_SCORE + 2)
    ax.axhline(TOTAL_MAX_SCORE, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Maximum")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_baseline = os.path.join(OUTPUT_DIR, "baseline_overall_performance.png")
    plt.savefig(plot_baseline, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {plot_baseline}")
    plt.close()

# ---------------- PART 2: 250-word Item-Level Improvements (Raw Scores) ---------------- #

print("\n" + "=" * 80)
print("PART 2: ITEM-LEVEL IMPROVEMENTS - 250-WORD VS ORIGINAL (RAW SCORES)")
print("=" * 80)

paired_250 = create_paired_dataset(df_orig, df_250, "orig", "250")

if not paired_250.empty:
    improvements_250_raw = calculate_item_improvements_raw(paired_250, "orig", "250")

    # Save CSV
    csv_250_raw = os.path.join(OUTPUT_DIR, "item_improvements_250_raw_scores.csv")
    improvements_250_raw.to_csv(csv_250_raw, index=False)
    print(f"\n📄 Saved: {csv_250_raw}")

    # Print summary
    print("\n250-word Item-Level Changes (Raw Score Differences):")
    print(f"{'Item':<25} {'Orig Mean':<11} {'250w Mean':<11} {'Mean Δ':<10} {'# Improved':<11}")
    print("-" * 75)
    for _, row in improvements_250_raw.iterrows():
        arrow = "↑" if row["mean_diff"] > 0 else "↓" if row["mean_diff"] < 0 else "→"
        print(f"{row['item']:<25} {row['orig_mean']:<11.2f} {row['rewrite_mean']:<11.2f} "
              f"{arrow} {row['mean_diff']:>+7.2f}  {row['n_improved']:<11.0f}")

    # Visualization 2: 250-word Item Changes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Sort by mean difference
    improvements_250_sorted = improvements_250_raw.sort_values("mean_diff")

    items = improvements_250_sorted["item"].tolist()
    diffs = improvements_250_sorted["mean_diff"].tolist()
    colors = ["green" if d > 0 else "red" if d < 0 else "gray" for d in diffs]

    bars = ax.barh(items, diffs, color=colors, alpha=0.7, edgecolor="black")

    # Add value labels
    for bar, diff in zip(bars, diffs):
        width = bar.get_width()
        label_x_pos = width + 0.05 if width > 0 else width - 0.05
        ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2.,
                f'{diff:+.2f}',
                ha='left' if width > 0 else 'right', va='center', fontweight='bold')

    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("Mean Change in Raw Score (250-word - Original)", fontsize=11, fontweight="bold")
    ax.set_title("Item-Level Improvements: 250-word vs Original (WITH Full Text Only)\n(Raw Score Changes)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plot_250_raw = os.path.join(OUTPUT_DIR, "item_improvements_250_raw_scores.png")
    plt.savefig(plot_250_raw, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {plot_250_raw}")
    plt.close()
else:
    print("\n⚠️ No paired data available for 250-word comparison")

# ---------------- PART 3: 300-word Item-Level Improvements (Raw Scores) ---------------- #

print("\n" + "=" * 80)
print("PART 3: ITEM-LEVEL IMPROVEMENTS - 300-WORD VS ORIGINAL (RAW SCORES)")
print("=" * 80)

paired_300 = create_paired_dataset(df_orig, df_300, "orig", "300")

if not paired_300.empty:
    improvements_300_raw = calculate_item_improvements_raw(paired_300, "orig", "300")

    # Save CSV
    csv_300_raw = os.path.join(OUTPUT_DIR, "item_improvements_300_raw_scores.csv")
    improvements_300_raw.to_csv(csv_300_raw, index=False)
    print(f"\n📄 Saved: {csv_300_raw}")

    # Print summary
    print("\n300-word Item-Level Changes (Raw Score Differences):")
    print(f"{'Item':<25} {'Orig Mean':<11} {'300w Mean':<11} {'Mean Δ':<10} {'# Improved':<11}")
    print("-" * 75)
    for _, row in improvements_300_raw.iterrows():
        arrow = "↑" if row["mean_diff"] > 0 else "↓" if row["mean_diff"] < 0 else "→"
        print(f"{row['item']:<25} {row['orig_mean']:<11.2f} {row['rewrite_mean']:<11.2f} "
              f"{arrow} {row['mean_diff']:>+7.2f}  {row['n_improved']:<11.0f}")

    # Visualization 3: 300-word Item Changes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Sort by mean difference
    improvements_300_sorted = improvements_300_raw.sort_values("mean_diff")

    items = improvements_300_sorted["item"].tolist()
    diffs = improvements_300_sorted["mean_diff"].tolist()
    colors = ["green" if d > 0 else "red" if d < 0 else "gray" for d in diffs]

    bars = ax.barh(items, diffs, color=colors, alpha=0.7, edgecolor="black")

    # Add value labels
    for bar, diff in zip(bars, diffs):
        width = bar.get_width()
        label_x_pos = width + 0.05 if width > 0 else width - 0.05
        ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2.,
                f'{diff:+.2f}',
                ha='left' if width > 0 else 'right', va='center', fontweight='bold')

    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("Mean Change in Raw Score (300-word - Original)", fontsize=11, fontweight="bold")
    ax.set_title("Item-Level Improvements: 300-word vs Original (WITH Full Text Only)\n(Raw Score Changes)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plot_300_raw = os.path.join(OUTPUT_DIR, "item_improvements_300_raw_scores.png")
    plt.savefig(plot_300_raw, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {plot_300_raw}")
    plt.close()
else:
    print("\n⚠️ No paired data available for 300-word comparison")

# ---------------- PART 4: Total Score vs Readability Analysis ---------------- #

print("\n" + "=" * 80)
print("PART 4: TOTAL SCORE VS READABILITY ANALYSIS")
print("=" * 80)

# Combine all data for correlation analysis
all_data = pd.concat([df_orig, df_250, df_300], ignore_index=True)

# Remove rows with missing readability data
all_data_clean = all_data.dropna(subset=["total_score", "flesch_kincaid", "reading_ease", "word_count"])

print(f"\nAnalyzing {len(all_data_clean)} abstracts with complete data (WITH full text only)...")

# Calculate correlations
if len(all_data_clean) > 0:
    corr_fk = all_data_clean[["total_score", "flesch_kincaid"]].corr().iloc[0, 1]
    corr_ease = all_data_clean[["total_score", "reading_ease"]].corr().iloc[0, 1]
    corr_wc = all_data_clean[["total_score", "word_count"]].corr().iloc[0, 1]

    print(f"\nCorrelations with Total CONSORT Score:")
    print(f"  Flesch-Kincaid Grade Level: r = {corr_fk:.3f}")
    print(f"  Reading Ease: r = {corr_ease:.3f}")
    print(f"  Word Count: r = {corr_wc:.3f}")

    # Save correlation data
    correlation_data = []
    for version in ["Original", "250-word", "300-word"]:
        subset = all_data_clean[all_data_clean["version"] == version]
        if len(subset) > 1:
            correlation_data.append({
                "version": version,
                "n": len(subset),
                "corr_total_vs_flesch_kincaid": subset[["total_score", "flesch_kincaid"]].corr().iloc[0, 1],
                "corr_total_vs_reading_ease": subset[["total_score", "reading_ease"]].corr().iloc[0, 1],
                "corr_total_vs_word_count": subset[["total_score", "word_count"]].corr().iloc[0, 1],
                "mean_total_score": subset["total_score"].mean(),
                "mean_flesch_kincaid": subset["flesch_kincaid"].mean(),
                "mean_reading_ease": subset["reading_ease"].mean(),
                "mean_word_count": subset["word_count"].mean(),
            })

    df_correlations = pd.DataFrame(correlation_data)
    csv_correlations = os.path.join(OUTPUT_DIR, "score_vs_readability_correlations.csv")
    df_correlations.to_csv(csv_correlations, index=False)
    print(f"\n📄 Saved: {csv_correlations}")

    # Visualization 4a: Total Score vs Flesch-Kincaid
    fig, ax = plt.subplots(figsize=(10, 7))

    colors_map = {"Original": "steelblue", "250-word": "orange", "300-word": "green"}

    for version in ["Original", "250-word", "300-word"]:
        subset = all_data_clean[all_data_clean["version"] == version]
        if not subset.empty:
            ax.scatter(subset["flesch_kincaid"], subset["total_score"],
                       alpha=0.6, s=60, color=colors_map[version], label=version, edgecolors="black")

    # Add regression line for all data
    if len(all_data_clean) > 1:
        z = np.polyfit(all_data_clean["flesch_kincaid"], all_data_clean["total_score"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(all_data_clean["flesch_kincaid"].min(),
                             all_data_clean["flesch_kincaid"].max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7,
                label=f"Trend line (r={corr_fk:.3f})")

    ax.set_xlabel("Flesch-Kincaid Grade Level", fontsize=12, fontweight="bold")
    ax.set_ylabel("Total CONSORT Score (Raw)", fontsize=12, fontweight="bold")
    ax.set_title("CONSORT Score vs Readability Grade Level (WITH Full Text Only)\n(Does complexity affect completeness?)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_score_fk = os.path.join(OUTPUT_DIR, "score_vs_flesch_kincaid.png")
    plt.savefig(plot_score_fk, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {plot_score_fk}")
    plt.close()

    # Visualization 4b: Total Score vs Reading Ease
    fig, ax = plt.subplots(figsize=(10, 7))

    for version in ["Original", "250-word", "300-word"]:
        subset = all_data_clean[all_data_clean["version"] == version]
        if not subset.empty:
            ax.scatter(subset["reading_ease"], subset["total_score"],
                       alpha=0.6, s=60, color=colors_map[version], label=version, edgecolors="black")

    # Add regression line
    if len(all_data_clean) > 1:
        z = np.polyfit(all_data_clean["reading_ease"], all_data_clean["total_score"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(all_data_clean["reading_ease"].min(),
                             all_data_clean["reading_ease"].max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7,
                label=f"Trend line (r={corr_ease:.3f})")

    ax.set_xlabel("Reading Ease Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Total CONSORT Score (Raw)", fontsize=12, fontweight="bold")
    ax.set_title("CONSORT Score vs Reading Ease (WITH Full Text Only)\n(Does readability affect completeness?)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_score_ease = os.path.join(OUTPUT_DIR, "score_vs_reading_ease.png")
    plt.savefig(plot_score_ease, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {plot_score_ease}")
    plt.close()

    # Visualization 4c: Total Score vs Word Count
    fig, ax = plt.subplots(figsize=(10, 7))

    for version in ["Original", "250-word", "300-word"]:
        subset = all_data_clean[all_data_clean["version"] == version]
        if not subset.empty:
            ax.scatter(subset["word_count"], subset["total_score"],
                       alpha=0.6, s=60, color=colors_map[version], label=version, edgecolors="black")

    # Add regression line
    if len(all_data_clean) > 1:
        z = np.polyfit(all_data_clean["word_count"], all_data_clean["total_score"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(all_data_clean["word_count"].min(),
                             all_data_clean["word_count"].max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7,
                label=f"Trend line (r={corr_wc:.3f})")

    ax.set_xlabel("Word Count", fontsize=12, fontweight="bold")
    ax.set_ylabel("Total CONSORT Score (Raw)", fontsize=12, fontweight="bold")
    ax.set_title("CONSORT Score vs Word Count (WITH Full Text Only)\n(Does length improve completeness?)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_score_wc = os.path.join(OUTPUT_DIR, "score_vs_word_count.png")
    plt.savefig(plot_score_wc, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {plot_score_wc}")
    plt.close()

    # Visualization 4d: Multi-panel comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: vs Flesch-Kincaid
    for version in ["Original", "250-word", "300-word"]:
        subset = all_data_clean[all_data_clean["version"] == version]
        if not subset.empty:
            axes[0].scatter(subset["flesch_kincaid"], subset["total_score"],
                            alpha=0.6, s=40, color=colors_map[version], label=version)
    axes[0].set_xlabel("Flesch-Kincaid Grade", fontweight="bold")
    axes[0].set_ylabel("Total CONSORT Score", fontweight="bold")
    axes[0].set_title(f"Grade Level\n(r={corr_fk:.3f})", fontweight="bold")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(alpha=0.3)

    # Panel 2: vs Reading Ease
    for version in ["Original", "250-word", "300-word"]:
        subset = all_data_clean[all_data_clean["version"] == version]
        if not subset.empty:
            axes[1].scatter(subset["reading_ease"], subset["total_score"],
                            alpha=0.6, s=40, color=colors_map[version], label=version)
    axes[1].set_xlabel("Reading Ease", fontweight="bold")
    axes[1].set_ylabel("Total CONSORT Score", fontweight="bold")
    axes[1].set_title(f"Reading Ease\n(r={corr_ease:.3f})", fontweight="bold")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(alpha=0.3)

    # Panel 3: vs Word Count
    for version in ["Original", "250-word", "300-word"]:
        subset = all_data_clean[all_data_clean["version"] == version]
        if not subset.empty:
            axes[2].scatter(subset["word_count"], subset["total_score"],
                            alpha=0.6, s=40, color=colors_map[version], label=version)
    axes[2].set_xlabel("Word Count", fontweight="bold")
    axes[2].set_ylabel("Total CONSORT Score", fontweight="bold")
    axes[2].set_title(f"Word Count\n(r={corr_wc:.3f})", fontweight="bold")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle("CONSORT Score vs Readability Metrics (WITH Full Text Only)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    plot_multi = os.path.join(OUTPUT_DIR, "score_vs_readability_multipanel.png")
    plt.savefig(plot_multi, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {plot_multi}")
    plt.close()

# ---------------- Final Summary ---------------- #

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - WITH FULL TEXT ONLY")
print("=" * 80)

print("\n📊 Key Findings:")

if baseline_stats:
    print("\n1. Overall Performance (Raw Scores out of 25):")
    for stats in baseline_stats:
        print(f"   {stats['version']}: {stats['total_mean']:.2f} ± {stats['total_sd']:.2f}")

if not paired_250.empty and not improvements_250_raw.empty:
    total_improvement_250 = improvements_250_raw["mean_diff"].sum()
    items_improved_250 = (improvements_250_raw["mean_diff"] > 0).sum()
    print(f"\n2. 250-word Rewriting:")
    print(f"   Total improvement: +{total_improvement_250:.2f} points")
    print(f"   Items improved: {items_improved_250}/{len(improvements_250_raw)}")

if not paired_300.empty and not improvements_300_raw.empty:
    total_improvement_300 = improvements_300_raw["mean_diff"].sum()
    items_improved_300 = (improvements_300_raw["mean_diff"] > 0).sum()
    print(f"\n3. 300-word Rewriting:")
    print(f"   Total improvement: +{total_improvement_300:.2f} points")
    print(f"   Items improved: {items_improved_300}/{len(improvements_300_raw)}")

if len(all_data_clean) > 0:
    print(f"\n4. Readability Relationships:")
    print(f"   Score vs Grade Level: r = {corr_fk:.3f}")
    print(f"   Score vs Reading Ease: r = {corr_ease:.3f}")
    print(f"   Score vs Word Count: r = {corr_wc:.3f}")

    if corr_wc > 0.3:
        print("\n   ✅ Positive correlation: Longer abstracts tend to have higher CONSORT scores")
    elif corr_wc < -0.3:
        print("\n   ⚠️ Negative correlation: Longer abstracts tend to have lower CONSORT scores")
    else:
        print("\n   → Weak correlation: Length has minimal impact on CONSORT completeness")

print("\n" + "=" * 80)
print("Generated outputs:")
print(f"  • {csv_baseline}")
print(f"  • {plot_baseline}")

if 'csv_250_raw' in locals():
    print(f"  • {csv_250_raw}")
    print(f"  • {plot_250_raw}")

if 'csv_300_raw' in locals():
    print(f"  • {csv_300_raw}")
    print(f"  • {plot_300_raw}")

if 'csv_correlations' in locals():
    print(f"  • {csv_correlations}")
    print(f"  • {plot_score_fk}")
    print(f"  • {plot_score_ease}")
    print(f"  • {plot_score_wc}")
    print(f"  • {plot_multi}")

print("\n✅ All analyses complete!")