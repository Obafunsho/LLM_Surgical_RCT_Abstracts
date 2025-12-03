"""
flesch_kincaid_evaluation.py

Statistical evaluation of readability metrics comparing original abstracts
to 250-word and 300-word rewritten versions.
FILTERED TO: Articles WITH full text available (PMC full text only)

Implements paired t-tests, confidence intervals, word count compliance checks,
and visualizations as specified in Protocol v3.0 (Results section).
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

# Word count targets and tolerance (±10% as per protocol)
TARGET_250 = 250
TARGET_300 = 300
TOLERANCE = 0.10


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


def load_readability(folder_path, version_label):
    """Load readability metrics from all JSON files in a folder"""
    records = []
    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
            data = json.load(f)

        # Clean PMCID from filename
        pmcid = filename.replace(".json", "").replace("_scores", "").replace("_250", "").replace("_300", "")

        r = data.get("readability", {})
        if r:
            records.append({
                "pmcid": pmcid,
                "version": version_label,
                "flesch_kincaid": pd.to_numeric(r.get("flesch_kincaid"), errors="coerce"),
                "reading_ease": pd.to_numeric(r.get("reading_ease"), errors="coerce"),
                "word_count": pd.to_numeric(r.get("word_count"), errors="coerce"),
                "has_full_text": get_full_text_status(pmcid),
            })

    return pd.DataFrame(records)


def cohen_d(x, y):
    """Calculate Cohen's d effect size for paired samples"""
    diff = x - y
    return diff.mean() / diff.std()


def paired_ttest_with_ci(orig, rewritten, metric_name):
    """
    Perform paired t-test and return statistics dict.
    Returns mean difference, 95% CI, t-statistic, p-value, Cohen's d
    """
    diff = rewritten - orig
    n = len(diff)

    if n < 2:
        return {
            "metric": metric_name,
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
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(rewritten, orig)

    # Effect size
    d = cohen_d(rewritten, orig)

    return {
        "metric": metric_name,
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

    # Calculate deltas
    for col in ["flesch_kincaid", "reading_ease", "word_count"]:
        paired[f"delta_{col}"] = paired[f"{col}_{suffix_b}"] - paired[f"{col}_{suffix_a}"]

    return paired


def check_word_count_compliance(df, target, tolerance=0.10):
    """Check compliance with target word count (within ±tolerance)"""
    if df.empty or "word_count" not in df.columns:
        return {}

    lower_bound = target * (1 - tolerance)
    upper_bound = target * (1 + tolerance)

    compliant = df["word_count"].between(lower_bound, upper_bound)

    return {
        "target": target,
        "tolerance_pct": tolerance * 100,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "n_total": len(df),
        "n_compliant": compliant.sum(),
        "pct_compliant": (compliant.sum() / len(df)) * 100 if len(df) > 0 else 0,
        "mean_wc": df["word_count"].mean(),
        "sd_wc": df["word_count"].std(),
        "median_wc": df["word_count"].median(),
        "min_wc": df["word_count"].min(),
        "max_wc": df["word_count"].max(),
    }


# ---------------- Load Data ---------------- #

print("=" * 80)
print("READABILITY EVALUATION - Protocol v3.0")
print("FILTERED TO: Articles WITH full text available (PMC full text only)")
print("=" * 80)

print("\n📥 Loading readability metrics...")
df_orig = load_readability(ORIGINAL_PATH, "original")
df_250 = load_readability(REWRITTEN_250_PATH, "250")
df_300 = load_readability(REWRITTEN_300_PATH, "300")

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

print("\n🔗 Creating paired datasets...")
paired_250 = create_paired_data(df_orig, df_250, "orig", "250")
paired_300 = create_paired_data(df_orig, df_300, "orig", "300")

print(f"   Original ↔ 250: {len(paired_250)} matched pairs")
print(f"   Original ↔ 300: {len(paired_300)} matched pairs")

# Export paired data
if not paired_250.empty:
    csv_paired_250 = os.path.join(OUTPUT_DIR, "readability_paired_differences_250.csv")
    paired_250.to_csv(csv_paired_250, index=False)
    print(f"   📄 Saved: {csv_paired_250}")

if not paired_300.empty:
    csv_paired_300 = os.path.join(OUTPUT_DIR, "readability_paired_differences_300.csv")
    paired_300.to_csv(csv_paired_300, index=False)
    print(f"   📄 Saved: {csv_paired_300}")

# ---------------- Analysis 1: Readability Metrics Comparison ---------------- #

print("\n" + "=" * 80)
print("ANALYSIS 1: READABILITY METRICS COMPARISON")
print("=" * 80)

readability_results = []

# 250-word comparison
if not paired_250.empty:
    print("\n250-word vs Original:")

    # Flesch-Kincaid Grade Level
    fk_stats = paired_ttest_with_ci(
        paired_250["flesch_kincaid_orig"],
        paired_250["flesch_kincaid_250"],
        "Flesch-Kincaid Grade"
    )
    fk_stats["comparison"] = "250 vs Original"
    readability_results.append(fk_stats)

    print(f"  Flesch-Kincaid Grade Level:")
    print(f"    Mean Δ: {fk_stats['mean_diff']:+.2f} (Median: {fk_stats['median_diff']:+.2f})")
    print(f"    95% CI: [{fk_stats['ci_lower']:.2f}, {fk_stats['ci_upper']:.2f}]")
    print(f"    t = {fk_stats['t_stat']:.2f}, p = {fk_stats['p_value']:.3f}")
    print(f"    Cohen's d = {fk_stats['cohens_d']:.2f}")

    # Reading Ease
    ease_stats = paired_ttest_with_ci(
        paired_250["reading_ease_orig"],
        paired_250["reading_ease_250"],
        "Reading Ease"
    )
    ease_stats["comparison"] = "250 vs Original"
    readability_results.append(ease_stats)

    print(f"  Reading Ease:")
    print(f"    Mean Δ: {ease_stats['mean_diff']:+.2f} (Median: {ease_stats['median_diff']:+.2f})")
    print(f"    95% CI: [{ease_stats['ci_lower']:.2f}, {ease_stats['ci_upper']:.2f}]")
    print(f"    t = {ease_stats['t_stat']:.2f}, p = {ease_stats['p_value']:.3f}")
    print(f"    Cohen's d = {ease_stats['cohens_d']:.2f}")

    # Word Count
    wc_stats = paired_ttest_with_ci(
        paired_250["word_count_orig"],
        paired_250["word_count_250"],
        "Word Count"
    )
    wc_stats["comparison"] = "250 vs Original"
    readability_results.append(wc_stats)

    print(f"  Word Count:")
    print(f"    Mean Δ: {wc_stats['mean_diff']:+.1f} words")

# 300-word comparison
if not paired_300.empty:
    print("\n300-word vs Original:")

    # Flesch-Kincaid Grade Level
    fk_stats = paired_ttest_with_ci(
        paired_300["flesch_kincaid_orig"],
        paired_300["flesch_kincaid_300"],
        "Flesch-Kincaid Grade"
    )
    fk_stats["comparison"] = "300 vs Original"
    readability_results.append(fk_stats)

    print(f"  Flesch-Kincaid Grade Level:")
    print(f"    Mean Δ: {fk_stats['mean_diff']:+.2f} (Median: {fk_stats['median_diff']:+.2f})")
    print(f"    95% CI: [{fk_stats['ci_lower']:.2f}, {fk_stats['ci_upper']:.2f}]")
    print(f"    t = {fk_stats['t_stat']:.2f}, p = {fk_stats['p_value']:.3f}")
    print(f"    Cohen's d = {fk_stats['cohens_d']:.2f}")

    # Reading Ease
    ease_stats = paired_ttest_with_ci(
        paired_300["reading_ease_orig"],
        paired_300["reading_ease_300"],
        "Reading Ease"
    )
    ease_stats["comparison"] = "300 vs Original"
    readability_results.append(ease_stats)

    print(f"  Reading Ease:")
    print(f"    Mean Δ: {ease_stats['mean_diff']:+.2f} (Median: {ease_stats['median_diff']:+.2f})")
    print(f"    95% CI: [{ease_stats['ci_lower']:.2f}, {ease_stats['ci_upper']:.2f}]")
    print(f"    t = {ease_stats['t_stat']:.2f}, p = {ease_stats['p_value']:.3f}")
    print(f"    Cohen's d = {ease_stats['cohens_d']:.2f}")

    # Word Count
    wc_stats = paired_ttest_with_ci(
        paired_300["word_count_orig"],
        paired_300["word_count_300"],
        "Word Count"
    )
    wc_stats["comparison"] = "300 vs Original"
    readability_results.append(wc_stats)

    print(f"  Word Count:")
    print(f"    Mean Δ: {wc_stats['mean_diff']:+.1f} words")

# Export readability comparison
df_readability = pd.DataFrame(readability_results)
csv_readability = os.path.join(OUTPUT_DIR, "readability_comparison.csv")
df_readability.to_csv(csv_readability, index=False)
print(f"\n📄 Saved: {csv_readability}")

# ---------------- Analysis 2: Grade Level Specific Analysis ---------------- #

print("\n" + "=" * 80)
print("ANALYSIS 2: GRADE LEVEL COMPARISON")
print("=" * 80)

gradelevel_results = []

for paired_df, label in [(paired_250, "250 vs Original"), (paired_300, "300 vs Original")]:
    if paired_df.empty:
        continue

    suffix = "250" if "250" in label else "300"

    stats_dict = paired_ttest_with_ci(
        paired_df["flesch_kincaid_orig"],
        paired_df[f"flesch_kincaid_{suffix}"],
        "Flesch-Kincaid Grade Level"
    )

    # Add mean values for context
    stats_dict["comparison"] = label
    stats_dict["mean_orig"] = paired_df["flesch_kincaid_orig"].mean()
    stats_dict["mean_rewritten"] = paired_df[f"flesch_kincaid_{suffix}"].mean()

    gradelevel_results.append(stats_dict)

df_gradelevel = pd.DataFrame(gradelevel_results)
csv_gradelevel = os.path.join(OUTPUT_DIR, "gradelevel_comparison.csv")
df_gradelevel.to_csv(csv_gradelevel, index=False)
print(f"📄 Saved: {csv_gradelevel}")

# ---------------- Analysis 3: Word Count Compliance ---------------- #

print("\n" + "=" * 80)
print("ANALYSIS 3: WORD COUNT COMPLIANCE")
print("=" * 80)

compliance_results = []

# Original (for reference - no target)
orig_compliance = {
    "version": "Original",
    "target": "N/A",
    "n_total": len(df_orig),
    "mean_wc": df_orig["word_count"].mean() if not df_orig.empty else np.nan,
    "sd_wc": df_orig["word_count"].std() if not df_orig.empty else np.nan,
    "median_wc": df_orig["word_count"].median() if not df_orig.empty else np.nan,
}
compliance_results.append(orig_compliance)

print(f"\nOriginal abstracts (WITH full text only):")
print(f"  Mean: {orig_compliance['mean_wc']:.1f} words (SD: {orig_compliance['sd_wc']:.1f})")
print(f"  Median: {orig_compliance['median_wc']:.1f} words")

# 250-word compliance
if not df_250.empty:
    comp_250 = check_word_count_compliance(df_250, TARGET_250, TOLERANCE)
    comp_250["version"] = "250-word"
    compliance_results.append(comp_250)

    print(f"\n250-word rewrites (target: {TARGET_250} ±{TOLERANCE * 100:.0f}%):")
    print(f"  Mean: {comp_250['mean_wc']:.1f} words (SD: {comp_250['sd_wc']:.1f})")
    print(f"  Median: {comp_250['median_wc']:.1f} words")
    print(f"  Range: [{comp_250['min_wc']:.0f}, {comp_250['max_wc']:.0f}]")
    print(f"  Compliant: {comp_250['n_compliant']}/{comp_250['n_total']} ({comp_250['pct_compliant']:.1f}%)")
    print(f"  Target bounds: [{comp_250['lower_bound']:.0f}, {comp_250['upper_bound']:.0f}] words")

# 300-word compliance
if not df_300.empty:
    comp_300 = check_word_count_compliance(df_300, TARGET_300, TOLERANCE)
    comp_300["version"] = "300-word"
    compliance_results.append(comp_300)

    print(f"\n300-word rewrites (target: {TARGET_300} ±{TOLERANCE * 100:.0f}%):")
    print(f"  Mean: {comp_300['mean_wc']:.1f} words (SD: {comp_300['sd_wc']:.1f})")
    print(f"  Median: {comp_300['median_wc']:.1f} words")
    print(f"  Range: [{comp_300['min_wc']:.0f}, {comp_300['max_wc']:.0f}]")
    print(f"  Compliant: {comp_300['n_compliant']}/{comp_300['n_total']} ({comp_300['pct_compliant']:.1f}%)")
    print(f"  Target bounds: [{comp_300['lower_bound']:.0f}, {comp_300['upper_bound']:.0f}] words")

# Export compliance data
df_compliance = pd.DataFrame(compliance_results)
csv_compliance = os.path.join(OUTPUT_DIR, "wordcount_compliance.csv")
df_compliance.to_csv(csv_compliance, index=False)
print(f"\n📄 Saved: {csv_compliance}")

# ---------------- Analysis 4: Summary Statistics ---------------- #

print("\n" + "=" * 80)
print("ANALYSIS 4: SUMMARY STATISTICS")
print("=" * 80)

all_df = pd.concat([df_orig, df_250, df_300], ignore_index=True)
summary_stats = all_df.groupby("version")[["flesch_kincaid", "reading_ease", "word_count"]].agg(
    ["mean", "std", "median"])

print("\nOverall Means by Version (WITH full text only):")
print(summary_stats.round(2))

csv_summary = os.path.join(OUTPUT_DIR, "summary_statistics.csv")
summary_stats.to_csv(csv_summary)
print(f"\n📄 Saved: {csv_summary}")

# ---------------- Visualization 1: Readability Metrics Bar Chart ---------------- #

print("\n📊 Generating visualizations...")

if not all_df.empty:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ["flesch_kincaid", "reading_ease", "word_count"]
    titles = ["Flesch-Kincaid Grade Level", "Reading Ease", "Word Count"]

    for ax, metric, title in zip(axes, metrics, titles):
        group_means = all_df.groupby("version")[metric].mean().reindex(["original", "250", "300"])

        ax.bar(range(len(group_means)), group_means.values, color=["steelblue", "orange", "green"], alpha=0.7)
        ax.set_xticks(range(len(group_means)))
        ax.set_xticklabels(["Original", "250w", "300w"])
        ax.set_title(title)
        ax.set_ylabel("Score" if metric != "word_count" else "Words")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Readability Metrics by Version (WITH Full Text Only)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_bar = os.path.join(OUTPUT_DIR, "readability_comparison_bar.png")
    plt.savefig(plot_bar, dpi=300)
    print(f"✅ Saved: {plot_bar}")
    plt.close()

# ---------------- Visualization 2: Change Distributions (Box Plots) ---------------- #

if not paired_250.empty or not paired_300.empty:
    deltas_list = []

    if not paired_250.empty:
        for _, row in paired_250.iterrows():
            deltas_list.append(
                {"comparison": "250 vs Orig", "metric": "Grade Level", "delta": row["delta_flesch_kincaid"]})
            deltas_list.append(
                {"comparison": "250 vs Orig", "metric": "Reading Ease", "delta": row["delta_reading_ease"]})

    if not paired_300.empty:
        for _, row in paired_300.iterrows():
            deltas_list.append(
                {"comparison": "300 vs Orig", "metric": "Grade Level", "delta": row["delta_flesch_kincaid"]})
            deltas_list.append(
                {"comparison": "300 vs Orig", "metric": "Reading Ease", "delta": row["delta_reading_ease"]})

    df_deltas = pd.DataFrame(deltas_list)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Grade Level changes
    grade_data = df_deltas[df_deltas["metric"] == "Grade Level"]
    if not grade_data.empty:
        axes[0].axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
        grade_data.boxplot(column="delta", by="comparison", ax=axes[0])
        axes[0].set_title("Change in Flesch-Kincaid Grade Level")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Δ Grade Level")
        axes[0].get_figure().suptitle("")

    # Reading Ease changes
    ease_data = df_deltas[df_deltas["metric"] == "Reading Ease"]
    if not ease_data.empty:
        axes[1].axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ease_data.boxplot(column="delta", by="comparison", ax=axes[1])
        axes[1].set_title("Change in Reading Ease")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Δ Reading Ease")
        axes[1].get_figure().suptitle("")

    fig.suptitle("Readability Changes (WITH Full Text Only)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_box = os.path.join(OUTPUT_DIR, "readability_changes_boxplot.png")
    plt.savefig(plot_box, dpi=300)
    print(f"✅ Saved: {plot_box}")
    plt.close()

# ---------------- Visualization 3: Scatter Plot (Reading Ease vs Grade Level) ---------------- #

if not all_df.empty:
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"original": "steelblue", "250": "orange", "300": "green"}

    for version, color in colors.items():
        subset = all_df[all_df["version"] == version]
        if not subset.empty:
            ax.scatter(subset["flesch_kincaid"], subset["reading_ease"],
                       alpha=0.5, s=50, color=color, label=version.capitalize())

    ax.set_xlabel("Flesch-Kincaid Grade Level")
    ax.set_ylabel("Reading Ease")
    ax.set_title("Correlation: Reading Ease vs. Grade Level (WITH Full Text Only)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    plot_scatter = os.path.join(OUTPUT_DIR, "readability_scatter.png")
    plt.savefig(plot_scatter, dpi=300)
    print(f"✅ Saved: {plot_scatter}")
    plt.close()

# ---------------- Final Summary ---------------- #

print("\n" + "=" * 80)
print("EVALUATION COMPLETE - WITH FULL TEXT ONLY")
print("=" * 80)
print("\nGenerated outputs:")
print(f"  • {csv_readability}")
print(f"  • {csv_gradelevel}")
print(f"  • {csv_compliance}")
print(f"  • {csv_summary}")
if 'csv_paired_250' in locals():
    print(f"  • {csv_paired_250}")
if 'csv_paired_300' in locals():
    print(f"  • {csv_paired_300}")
if 'plot_bar' in locals():
    print(f"  • {plot_bar}")
if 'plot_box' in locals():
    print(f"  • {plot_box}")
if 'plot_scatter' in locals():
    print(f"  • {plot_scatter}")

print("\n✅ All analyses complete!")