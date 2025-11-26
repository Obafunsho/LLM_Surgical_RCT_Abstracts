"""
average_per_item_consort_evaluation.py

Per-item CONSORT performance analysis comparing original abstracts
to 250-word and 300-word rewritten versions.

Analyzes:
- Average scores per CONSORT item by version
- Percentage of maximum possible score per item
- Items with best/worst compliance
- Improvements/declines after rewriting
- Visual comparisons across all items

Outputs:
- CSV with average per-item scores
- CSV with improvement metrics
- Bar charts comparing performance
- Heatmap visualization
- Grouped bar chart for direct comparison
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Configuration ---------------- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

ORIGINAL_PATH = os.path.join(PROJECT_ROOT, "data", "scores", "original")
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
    if isinstance(cons, dict) and "items" in cons:
        row = {"pmcid": pmcid}

        # Extract per-item scores
        items = cons.get("items", {})
        for item_name in ITEM_ORDER:
            score = items.get(item_name, {}).get("score", 0)
            row[item_name] = _safe_float(score, 0.0)

        return row
    else:
        return None


def load_folder(folder_path, version_label):
    """Load all score files from a folder"""
    records = []
    for fn in os.listdir(folder_path):
        if fn.endswith(".json"):
            record = load_one_score(os.path.join(folder_path, fn))
            if record:
                record["version"] = version_label
                records.append(record)

    return pd.DataFrame(records) if records else pd.DataFrame()


def calculate_item_statistics(df, version_label):
    """Calculate statistics for each CONSORT item"""
    stats = []

    for item in ITEM_ORDER:
        if item not in df.columns:
            continue

        item_max = RUBRIC_MAX[item]
        scores = df[item].dropna()

        if len(scores) == 0:
            continue

        # Calculate percentage of maximum possible score
        pct_scores = (scores / item_max) * 100.0

        stats.append({
            "item": item,
            "version": version_label,
            "item_max": item_max,
            "n": len(scores),
            "mean_score": scores.mean(),
            "sd_score": scores.std(),
            "median_score": scores.median(),
            "min_score": scores.min(),
            "max_score": scores.max(),
            "mean_pct_of_max": pct_scores.mean(),
            "sd_pct_of_max": pct_scores.std(),
            "median_pct_of_max": pct_scores.median(),
            # Count how many achieved max score
            "n_at_max": (scores == item_max).sum(),
            "pct_at_max": ((scores == item_max).sum() / len(scores)) * 100.0,
            # Count how many scored zero
            "n_at_zero": (scores == 0).sum(),
            "pct_at_zero": ((scores == 0).sum() / len(scores)) * 100.0,
        })

    return pd.DataFrame(stats)


def calculate_improvements(df_orig, df_250, df_300):
    """Calculate improvement metrics comparing rewritten to original"""
    improvements = []

    for item in ITEM_ORDER:
        if item not in df_orig.columns:
            continue

        item_max = RUBRIC_MAX[item]

        # Original stats
        orig_mean = df_orig[item].mean()
        orig_pct = (orig_mean / item_max) * 100.0

        # 250-word stats
        if not df_250.empty and item in df_250.columns:
            w250_mean = df_250[item].mean()
            w250_pct = (w250_mean / item_max) * 100.0
            improvement_250_abs = w250_mean - orig_mean
            improvement_250_pct = w250_pct - orig_pct
        else:
            w250_mean = np.nan
            w250_pct = np.nan
            improvement_250_abs = np.nan
            improvement_250_pct = np.nan

        # 300-word stats
        if not df_300.empty and item in df_300.columns:
            w300_mean = df_300[item].mean()
            w300_pct = (w300_mean / item_max) * 100.0
            improvement_300_abs = w300_mean - orig_mean
            improvement_300_pct = w300_pct - orig_pct
        else:
            w300_mean = np.nan
            w300_pct = np.nan
            improvement_300_abs = np.nan
            improvement_300_pct = np.nan

        improvements.append({
            "item": item,
            "item_max": item_max,
            "orig_mean_score": orig_mean,
            "orig_mean_pct": orig_pct,
            "250w_mean_score": w250_mean,
            "250w_mean_pct": w250_pct,
            "improvement_250_abs": improvement_250_abs,
            "improvement_250_pct": improvement_250_pct,
            "300w_mean_score": w300_mean,
            "300w_mean_pct": w300_pct,
            "improvement_300_abs": improvement_300_abs,
            "improvement_300_pct": improvement_300_pct,
        })

    return pd.DataFrame(improvements)


# ---------------- Load Data ---------------- #

print("=" * 80)
print("PER-ITEM CONSORT ANALYSIS - Protocol v3.0")
print("=" * 80)

print("\n📥 Loading CONSORT scores...")
df_orig = load_folder(ORIGINAL_PATH, "Original")
df_250 = load_folder(REWRITTEN_250_PATH, "250-word")
df_300 = load_folder(REWRITTEN_300_PATH, "300-word")

print(f"   Original abstracts: {len(df_orig)}")
print(f"   250-word rewrites: {len(df_250)}")
print(f"   300-word rewrites: {len(df_300)}")

# ---------------- Analysis 1: Item-Level Statistics by Version ---------------- #

print("\n" + "=" * 80)
print("ANALYSIS 1: ITEM-LEVEL STATISTICS BY VERSION")
print("=" * 80)

stats_orig = calculate_item_statistics(df_orig, "Original")
stats_250 = calculate_item_statistics(df_250, "250-word")
stats_300 = calculate_item_statistics(df_300, "300-word")

# Combine all statistics
all_stats = pd.concat([stats_orig, stats_250, stats_300], ignore_index=True)

# Save to CSV
csv_item_stats = os.path.join(OUTPUT_DIR, "average_per_item_consort_scores.csv")
all_stats.to_csv(csv_item_stats, index=False)
print(f"\n📄 Saved: {csv_item_stats}")

# Print summary table
print("\nAverage % of Maximum Score by Item:")
summary_table = all_stats.pivot(index="item", columns="version", values="mean_pct_of_max")
summary_table = summary_table.reindex(ITEM_ORDER)
print(summary_table.round(1).to_string())

# ---------------- Analysis 2: Improvement Metrics ---------------- #

print("\n" + "=" * 80)
print("ANALYSIS 2: IMPROVEMENT METRICS")
print("=" * 80)

improvements = calculate_improvements(df_orig, df_250, df_300)

# Save to CSV
csv_improvements = os.path.join(OUTPUT_DIR, "per_item_improvements.csv")
improvements.to_csv(csv_improvements, index=False)
print(f"\n📄 Saved: {csv_improvements}")

# Print improvements
print("\nImprovement Summary (percentage points):")
print("\nItem                      | Orig %  | 250w Δ  | 300w Δ")
print("-" * 60)
for _, row in improvements.iterrows():
    print(
        f"{row['item']:<25} | {row['orig_mean_pct']:>6.1f} | {row['improvement_250_pct']:>+6.1f} | {row['improvement_300_pct']:>+6.1f}")

# ---------------- Analysis 3: Identify Best/Worst Performing Items ---------------- #

print("\n" + "=" * 80)
print("ANALYSIS 3: BEST AND WORST PERFORMING ITEMS")
print("=" * 80)

# Best performing items (original)
best_items_orig = stats_orig.nlargest(5, "mean_pct_of_max")[["item", "mean_pct_of_max", "pct_at_max"]]
print("\nBest Performing Items (Original):")
print(best_items_orig.to_string(index=False))

# Worst performing items (original)
worst_items_orig = stats_orig.nsmallest(5, "mean_pct_of_max")[["item", "mean_pct_of_max", "pct_at_zero"]]
print("\nWorst Performing Items (Original):")
print(worst_items_orig.to_string(index=False))

# Most improved items (250-word)
if not improvements.empty:
    most_improved_250 = improvements.nlargest(5, "improvement_250_pct")[
        ["item", "improvement_250_pct", "250w_mean_pct"]]
    print("\nMost Improved Items (250-word version):")
    print(most_improved_250.to_string(index=False))

    # Items that declined (if any)
    declined_250 = improvements[improvements["improvement_250_pct"] < 0].sort_values("improvement_250_pct")[
        ["item", "improvement_250_pct", "250w_mean_pct"]]
    if not declined_250.empty:
        print("\nItems with Declined Performance (250-word version):")
        print(declined_250.to_string(index=False))

# ---------------- Visualization 1: Grouped Bar Chart (Side-by-Side Comparison) ---------------- #

print("\n📊 Generating visualizations...")

if not all_stats.empty:
    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for grouped bar chart
    pivot_data = all_stats.pivot(index="item", columns="version", values="mean_pct_of_max")
    pivot_data = pivot_data.reindex(ITEM_ORDER)

    # Create grouped bars
    x = np.arange(len(ITEM_ORDER))
    width = 0.25

    versions = ["Original", "250-word", "300-word"]
    colors = ["steelblue", "orange", "green"]

    for i, (version, color) in enumerate(zip(versions, colors)):
        if version in pivot_data.columns:
            offset = width * (i - 1)
            ax.bar(x + offset, pivot_data[version], width, label=version, color=color, alpha=0.8)

    ax.set_xlabel("CONSORT Item", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average % of Maximum Score", fontsize=12, fontweight="bold")
    ax.set_title("Average CONSORT Item Performance by Version", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ITEM_ORDER, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plot_grouped = os.path.join(OUTPUT_DIR, "per_item_comparison_grouped.png")
    plt.savefig(plot_grouped, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {plot_grouped}")
    plt.close()

# ---------------- Visualization 2: Heatmap of Performance ---------------- #

if not all_stats.empty:
    fig, ax = plt.subplots(figsize=(10, 10))

    # Prepare data for heatmap
    heatmap_data = all_stats.pivot(index="item", columns="version", values="mean_pct_of_max")
    heatmap_data = heatmap_data.reindex(ITEM_ORDER)
    heatmap_data = heatmap_data[["Original", "250-word", "300-word"]]

    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="RdYlGn",
                vmin=0, vmax=100, cbar_kws={"label": "% of Item Max"},
                linewidths=0.5, ax=ax, cbar=True)

    ax.set_title("CONSORT Item Performance Heatmap\n(Average % of Maximum Score)",
                 fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Version", fontsize=12, fontweight="bold")
    ax.set_ylabel("CONSORT Item", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plot_heatmap = os.path.join(OUTPUT_DIR, "per_item_heatmap.png")
    plt.savefig(plot_heatmap, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {plot_heatmap}")
    plt.close()

# ---------------- Visualization 3: Improvement Bar Chart ---------------- #

if not improvements.empty:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Sort by 250-word improvement
    improvements_sorted_250 = improvements.sort_values("improvement_250_pct")

    # 250-word improvements
    colors_250 = ["green" if x > 0 else "red" for x in improvements_sorted_250["improvement_250_pct"]]
    ax1.barh(improvements_sorted_250["item"], improvements_sorted_250["improvement_250_pct"],
             color=colors_250, alpha=0.7)
    ax1.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    ax1.set_xlabel("Improvement (percentage points)", fontsize=11)
    ax1.set_title("Change in Performance: 250-word vs Original", fontsize=12, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    # Sort by 300-word improvement
    improvements_sorted_300 = improvements.sort_values("improvement_300_pct")

    # 300-word improvements
    colors_300 = ["green" if x > 0 else "red" for x in improvements_sorted_300["improvement_300_pct"]]
    ax2.barh(improvements_sorted_300["item"], improvements_sorted_300["improvement_300_pct"],
             color=colors_300, alpha=0.7)
    ax2.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    ax2.set_xlabel("Improvement (percentage points)", fontsize=11)
    ax2.set_title("Change in Performance: 300-word vs Original", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plot_improvements = os.path.join(OUTPUT_DIR, "per_item_improvements_bars.png")
    plt.savefig(plot_improvements, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {plot_improvements}")
    plt.close()

# ---------------- Visualization 4: Stacked Bar Chart (Zero vs Partial vs Max) ---------------- #

if not all_stats.empty:
    fig, ax = plt.subplots(figsize=(14, 8))

    # Calculate proportions for each item and version
    stacked_data = []
    for item in ITEM_ORDER:
        for version in ["Original", "250-word", "300-word"]:
            subset = all_stats[(all_stats["item"] == item) & (all_stats["version"] == version)]
            if not subset.empty:
                row = subset.iloc[0]
                pct_zero = row["pct_at_zero"]
                pct_max = row["pct_at_max"]
                pct_partial = 100.0 - pct_zero - pct_max

                stacked_data.append({
                    "item": item,
                    "version": version,
                    "pct_zero": pct_zero,
                    "pct_partial": pct_partial,
                    "pct_max": pct_max,
                })

    df_stacked = pd.DataFrame(stacked_data)

    # Create stacked bars
    x = np.arange(len(ITEM_ORDER))
    width = 0.25

    for i, version in enumerate(["Original", "250-word", "300-word"]):
        subset = df_stacked[df_stacked["version"] == version].set_index("item").reindex(ITEM_ORDER)
        offset = width * (i - 1)

        # Stack: Zero (bottom), Partial (middle), Max (top)
        ax.bar(x + offset, subset["pct_zero"], width, label=f"{version} (Zero)" if i == 0 else "",
               color="red", alpha=0.6)
        ax.bar(x + offset, subset["pct_partial"], width, bottom=subset["pct_zero"],
               label=f"{version} (Partial)" if i == 0 else "", color="yellow", alpha=0.6)
        ax.bar(x + offset, subset["pct_max"], width,
               bottom=subset["pct_zero"] + subset["pct_partial"],
               label=f"{version} (Max)" if i == 0 else "", color="green", alpha=0.6)

    ax.set_xlabel("CONSORT Item", fontsize=12, fontweight="bold")
    ax.set_ylabel("Percentage of Abstracts", fontsize=12, fontweight="bold")
    ax.set_title("Distribution of Scores: Zero / Partial / Maximum", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ITEM_ORDER, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_stacked = os.path.join(OUTPUT_DIR, "per_item_score_distribution.png")
    plt.savefig(plot_stacked, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {plot_stacked}")
    plt.close()

# ---------------- Final Summary ---------------- #

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\n📋 Key Findings:")

# Most consistently reported
if not stats_orig.empty:
    best_item = stats_orig.loc[stats_orig["mean_pct_of_max"].idxmax()]
    print(f"\n✅ Most consistently reported item (Original):")
    print(f"   {best_item['item']}: {best_item['mean_pct_of_max']:.1f}% average")
    print(f"   ({best_item['pct_at_max']:.1f}% achieved maximum score)")

# Most under-reported
if not stats_orig.empty:
    worst_item = stats_orig.loc[stats_orig["mean_pct_of_max"].idxmin()]
    print(f"\n❌ Most under-reported item (Original):")
    print(f"   {worst_item['item']}: {worst_item['mean_pct_of_max']:.1f}% average")
    print(f"   ({worst_item['pct_at_zero']:.1f}% scored zero)")

# Biggest improvement
if not improvements.empty:
    best_improvement_250 = improvements.loc[improvements["improvement_250_pct"].idxmax()]
    print(f"\n📈 Biggest improvement (250-word):")
    print(f"   {best_improvement_250['item']}: +{best_improvement_250['improvement_250_pct']:.1f} pp")
    print(f"   (from {best_improvement_250['orig_mean_pct']:.1f}% to {best_improvement_250['250w_mean_pct']:.1f}%)")

print("\n" + "=" * 80)
print("Generated outputs:")
print(f"  • {csv_item_stats}")
print(f"  • {csv_improvements}")
print(f"  • {plot_grouped}")
print(f"  • {plot_heatmap}")
print(f"  • {plot_improvements}")
print(f"  • {plot_stacked}")
print("\n✅ All analyses complete!")