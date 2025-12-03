"""
consort_evaluation.py

Statistical evaluation of CONSORT performance comparing original abstracts
to 250-word and 300-word rewritten versions.
FILTERED TO: Articles WITH full text available (PMC full text only)

Implements paired t-tests, confidence intervals, and effect sizes as specified
in Protocol v3.0 (Results section).
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

def load_one_score(path):
    """Load a single score JSON file and extract CONSORT scores"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pmcid = data.get("pmcid", os.path.basename(path).replace("_scores.json", ""))
    cons = data.get("consort_scores", {})

    # Check if new schema (v3.0 with items dict)
    if isinstance(cons, dict) and "items" in cons and "total_score" in cons:
        row = {
            "pmcid": pmcid,
            "total_score": _safe_float(cons.get("total_score", 0)),
            "max_score": _safe_float(cons.get("max_score", 25)),
        }

        # Extract per-item scores
        items = cons.get("items", {})
        for item_name in ITEM_ORDER:
            score = items.get(item_name, {}).get("score", 0)
            row[item_name] = _safe_float(score, 0.0)

        # Add full text status
        row["has_full_text"] = get_full_text_status(pmcid)

        return row
    else:
        return None

def load_folder(folder_path):
    """Load all score files from a folder"""
    records = []
    for fn in os.listdir(folder_path):
        if fn.endswith(".json"):
            record = load_one_score(os.path.join(folder_path, fn))
            if record:
                records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Add normalized total (% of max)
    df["normalized_total"] = (df["total_score"] / df["max_score"]) * 100.0
    return df

def cohen_d(x, y):
    """Calculate Cohen's d effect size for paired samples"""
    diff = x - y
    return diff.mean() / diff.std()

def paired_ttest_with_ci(orig, rewritten, label):
    """
    Perform paired t-test and return statistics dict.
    Returns mean difference, 95% CI, t-statistic, p-value, Cohen's d
    """
    diff = rewritten - orig
    n = len(diff)

    if n < 2:
        return {
            "comparison": label,
            "n_pairs": n,
            "mean_diff": np.nan,
            "median_diff": np.nan,
            "sd_diff": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "cohens_d": np.nan,
        }

    mean_diff = diff.mean()
    median_diff = diff.median()
    sd_diff = diff.std()
    se_diff = sd_diff / np.sqrt(n)

    # 95% CI using t-distribution
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(rewritten, orig)

    # Effect size
    d = cohen_d(rewritten, orig)

    return {
        "comparison": label,
        "n_pairs": n,
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "sd_diff": sd_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "t_stat": t_stat,
        "p_value": p_value,
        "cohens_d": d,
    }

# ---------------- Load Data ---------------- #

print("=" * 80)
print("CONSORT EVALUATION - Protocol v3.0")
print("FILTERED TO: Articles WITH full text available (PMC full text only)")
print("=" * 80)

print("\n📥 Loading CONSORT scores...")
df_orig = load_folder(ORIGINAL_PATH)
df_250 = load_folder(REWRITTEN_250_PATH)
df_300 = load_folder(REWRITTEN_300_PATH)

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

# ---------------- Create Paired Datasets ---------------- #

def create_paired_data(df_a, df_b, suffix_a, suffix_b):
    """Create paired dataset by matching PMCIDs"""
    # Find common PMCIDs
    common = set(df_a["pmcid"]).intersection(set(df_b["pmcid"]))

    if not common:
        return pd.DataFrame()

    # Filter to common PMCIDs and sort
    df_a_matched = df_a[df_a["pmcid"].isin(common)].sort_values("pmcid").reset_index(drop=True)
    df_b_matched = df_b[df_b["pmcid"].isin(common)].sort_values("pmcid").reset_index(drop=True)

    # Rename columns
    df_a_renamed = df_a_matched.add_suffix(f"_{suffix_a}")
    df_b_renamed = df_b_matched.add_suffix(f"_{suffix_b}")

    # Merge
    paired = pd.concat([df_a_renamed, df_b_renamed], axis=1)
    paired["pmcid"] = df_a_matched["pmcid"]

    return paired

print("\n🔗 Creating paired datasets...")
paired_250 = create_paired_data(df_orig, df_250, "orig", "250")
paired_300 = create_paired_data(df_orig, df_300, "orig", "300")

print(f"   Original ↔ 250: {len(paired_250)} matched pairs")
print(f"   Original ↔ 300: {len(paired_300)} matched pairs")

# ---------------- Analysis 1: Total Score Comparison ---------------- #

print("\n" + "=" * 80)
print("ANALYSIS 1: TOTAL SCORE COMPARISON")
print("=" * 80)

results_total = []

# 250-word vs Original
if not paired_250.empty:
    stats_250 = paired_ttest_with_ci(
        paired_250["total_score_orig"],
        paired_250["total_score_250"],
        "250 vs Original"
    )
    results_total.append(stats_250)

    print(f"\n250-word vs Original (n={stats_250['n_pairs']}):")
    print(f"  Mean Δ: {stats_250['mean_diff']:.2f} points (Median: {stats_250['median_diff']:.1f})")
    print(f"  95% CI: [{stats_250['ci_lower']:.2f}, {stats_250['ci_upper']:.2f}]")
    print(f"  t = {stats_250['t_stat']:.2f}, p = {stats_250['p_value']:.3f}")
    print(f"  Cohen's d = {stats_250['cohens_d']:.2f}")

    # Normalized (percentage points)
    stats_250_norm = paired_ttest_with_ci(
        paired_250["normalized_total_orig"],
        paired_250["normalized_total_250"],
        "250 vs Original (normalized %)"
    )
    print(f"  Normalized Δ: {stats_250_norm['mean_diff']:.2f} pp")
    print(f"  95% CI: [{stats_250_norm['ci_lower']:.2f}, {stats_250_norm['ci_upper']:.2f}] pp")

# 300-word vs Original
if not paired_300.empty:
    stats_300 = paired_ttest_with_ci(
        paired_300["total_score_orig"],
        paired_300["total_score_300"],
        "300 vs Original"
    )
    results_total.append(stats_300)

    print(f"\n300-word vs Original (n={stats_300['n_pairs']}):")
    print(f"  Mean Δ: {stats_300['mean_diff']:.2f} points (Median: {stats_300['median_diff']:.1f})")
    print(f"  95% CI: [{stats_300['ci_lower']:.2f}, {stats_300['ci_upper']:.2f}]")
    print(f"  t = {stats_300['t_stat']:.2f}, p = {stats_300['p_value']:.3f}")
    print(f"  Cohen's d = {stats_300['cohens_d']:.2f}")

    # Normalized (percentage points)
    stats_300_norm = paired_ttest_with_ci(
        paired_300["normalized_total_orig"],
        paired_300["normalized_total_300"],
        "300 vs Original (normalized %)"
    )
    print(f"  Normalized Δ: {stats_300_norm['mean_diff']:.2f} pp")
    print(f"  95% CI: [{stats_300_norm['ci_lower']:.2f}, {stats_300_norm['ci_upper']:.2f}] pp")

# Save total score comparison
df_total = pd.DataFrame(results_total)
csv_total = os.path.join(OUTPUT_DIR, "total_score_comparison.csv")
df_total.to_csv(csv_total, index=False)
print(f"\n📄 Saved: {csv_total}")

# ---------------- Analysis 2: Item-Level Performance ---------------- #

print("\n" + "=" * 80)
print("ANALYSIS 2: ITEM-LEVEL PERFORMANCE")
print("=" * 80)

def analyze_item_level(paired_df, suffix_orig, suffix_rewrite, comparison_label):
    """Analyze per-item changes"""
    results = []

    for item in ITEM_ORDER:
        col_orig = f"{item}_{suffix_orig}"
        col_rewrite = f"{item}_{suffix_rewrite}"

        if col_orig not in paired_df.columns or col_rewrite not in paired_df.columns:
            continue

        # Convert to % of item max
        orig_pct = (paired_df[col_orig] / RUBRIC_MAX[item]) * 100.0
        rewrite_pct = (paired_df[col_rewrite] / RUBRIC_MAX[item]) * 100.0

        # Paired t-test
        stats_dict = paired_ttest_with_ci(orig_pct, rewrite_pct, item)
        stats_dict["item"] = item
        stats_dict["item_max"] = RUBRIC_MAX[item]
        stats_dict["comparison"] = comparison_label
        stats_dict["mean_orig_pct"] = orig_pct.mean()
        stats_dict["mean_rewrite_pct"] = rewrite_pct.mean()

        results.append(stats_dict)

    return pd.DataFrame(results)

# 250-word item analysis
if not paired_250.empty:
    print("\n250-word vs Original (item-level):")
    items_250 = analyze_item_level(paired_250, "orig", "250", "250 vs Original")
    print(items_250[["item", "mean_diff", "ci_lower", "ci_upper", "p_value"]].to_string(index=False))

    csv_items_250 = os.path.join(OUTPUT_DIR, "item_level_improvements_250.csv")
    items_250.to_csv(csv_items_250, index=False)
    print(f"\n📄 Saved: {csv_items_250}")

# 300-word item analysis
if not paired_300.empty:
    print("\n300-word vs Original (item-level):")
    items_300 = analyze_item_level(paired_300, "orig", "300", "300 vs Original")
    print(items_300[["item", "mean_diff", "ci_lower", "ci_upper", "p_value"]].to_string(index=False))

    csv_items_300 = os.path.join(OUTPUT_DIR, "item_level_improvements_300.csv")
    items_300.to_csv(csv_items_300, index=False)
    print(f"\n📄 Saved: {csv_items_300}")

# ---------------- Analysis 3: Summary Statistics ---------------- #

print("\n" + "=" * 80)
print("ANALYSIS 3: SUMMARY STATISTICS")
print("=" * 80)

summary_rows = []

# Overall means
for df, label in [(df_orig, "Original"), (df_250, "250-word"), (df_300, "300-word")]:
    if not df.empty:
        summary_rows.append({
            "version": label,
            "n": len(df),
            "mean_total": df["total_score"].mean(),
            "sd_total": df["total_score"].std(),
            "mean_normalized_pct": df["normalized_total"].mean(),
            "sd_normalized_pct": df["normalized_total"].std(),
        })

df_summary = pd.DataFrame(summary_rows)
print("\nOverall Means by Version (WITH full text only):")
print(df_summary.to_string(index=False))

csv_summary = os.path.join(OUTPUT_DIR, "summary_statistics.csv")
df_summary.to_csv(csv_summary, index=False)
print(f"\n📄 Saved: {csv_summary}")

# ---------------- Visualization 1: Total Score Changes ---------------- #

print("\n📊 Generating visualizations...")

if results_total:
    fig, ax = plt.subplots(figsize=(8, 6))

    comparisons = [r["comparison"] for r in results_total]
    mean_diffs = [r["mean_diff"] for r in results_total]
    ci_lowers = [r["ci_lower"] for r in results_total]
    ci_uppers = [r["ci_upper"] for r in results_total]

    # Error bars
    errors = [[m - l for m, l in zip(mean_diffs, ci_lowers)],
              [u - m for m, u in zip(mean_diffs, ci_uppers)]]

    ax.barh(comparisons, mean_diffs, xerr=errors, capsize=5, color="steelblue", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Mean Difference in Total Score (points)")
    ax.set_title("Change in CONSORT Total Score (WITH Full Text Only)\n(Rewritten vs Original, with 95% CI)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    plot_total = os.path.join(OUTPUT_DIR, "total_score_comparison.png")
    plt.savefig(plot_total, dpi=300)
    print(f"✅ Saved: {plot_total}")
    plt.close()

# ---------------- Visualization 2: Item-Level Changes ---------------- #

def plot_item_changes(items_df, comparison_label, output_filename):
    """Generate item-level bar chart with error bars"""
    if items_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    # Sort by mean difference
    items_sorted = items_df.sort_values("mean_diff")

    items_list = items_sorted["item"].tolist()
    mean_diffs = items_sorted["mean_diff"].tolist()
    ci_lowers = items_sorted["ci_lower"].tolist()
    ci_uppers = items_sorted["ci_upper"].tolist()

    errors = [[m - l for m, l in zip(mean_diffs, ci_lowers)],
              [u - m for m, u in zip(mean_diffs, ci_uppers)]]

    # Color by positive/negative
    colors = ["green" if d > 0 else "red" for d in mean_diffs]

    ax.barh(items_list, mean_diffs, xerr=errors, capsize=4, color=colors, alpha=0.6)
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Mean Difference (% of item max)")
    ax.set_title(f"Item-Level Changes: {comparison_label} (WITH Full Text Only)\n(with 95% CI)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(output_path, dpi=300)
    print(f"✅ Saved: {output_path}")
    plt.close()

    return output_path

# Generate plot for 250-word comparison
if not paired_250.empty and 'items_250' in locals():
    plot_items_250 = plot_item_changes(items_250, "250-word vs Original", "item_level_changes_250.png")

# Generate plot for 300-word comparison
if not paired_300.empty and 'items_300' in locals():
    plot_items_300 = plot_item_changes(items_300, "300-word vs Original", "item_level_changes_300.png")

# ---------------- Visualization 3: Heatmap of Mean Scores by Version ---------------- #

if not df_orig.empty:
    # Calculate mean % of max for each item by version
    heatmap_data = []

    for item in ITEM_ORDER:
        row = {"Item": item}
        for df, label in [(df_orig, "Original"), (df_250, "250w"), (df_300, "300w")]:
            if not df.empty and item in df.columns:
                pct = (df[item] / RUBRIC_MAX[item]) * 100.0
                row[label] = pct.mean()
            else:
                row[label] = np.nan
        heatmap_data.append(row)

    df_heatmap = pd.DataFrame(heatmap_data).set_index("Item")

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(df_heatmap, annot=True, fmt=".1f", cmap="RdYlGn",
                vmin=0, vmax=100, cbar_kws={"label": "% of Item Max"},
                linewidths=0.5, ax=ax)
    ax.set_title("CONSORT Item Performance Heatmap (WITH Full Text Only)\n(Mean % of Item Max by Version)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()

    plot_heatmap = os.path.join(OUTPUT_DIR, "consort_heatmap_by_version.png")
    plt.savefig(plot_heatmap, dpi=300)
    print(f"✅ Saved: {plot_heatmap}")
    plt.close()

# ---------------- Final Summary ---------------- #

print("\n" + "=" * 80)
print("EVALUATION COMPLETE - WITH FULL TEXT ONLY")
print("=" * 80)
print("\nGenerated outputs:")
print(f"  • {csv_total}")
print(f"  • {csv_summary}")
if 'csv_items_250' in locals():
    print(f"  • {csv_items_250}")
if 'csv_items_300' in locals():
    print(f"  • {csv_items_300}")
if 'plot_total' in locals():
    print(f"  • {plot_total}")
if 'plot_items_250' in locals():
    print(f"  • {plot_items_250}")
if 'plot_items_300' in locals():
    print(f"  • {plot_items_300}")
if 'plot_heatmap' in locals():
    print(f"  • {plot_heatmap}")

print("\n✅ All analyses complete!")