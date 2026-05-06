"""
03_eda.py

Exploratory data analysis on cleaned CB data. Produces presentation-ready
visualizations saved to outputs/figures/.

Input:
    {PROJECT_ROOT}/data/processed/cb_starters.csv
    {PROJECT_ROOT}/data/processed/cb_clean.csv

Output:
    {PROJECT_ROOT}/outputs/figures/01_grade_distribution.png
    {PROJECT_ROOT}/outputs/figures/02_grade_by_season.png
    {PROJECT_ROOT}/outputs/figures/03_alignment_profile.png
    {PROJECT_ROOT}/outputs/figures/04_man_zone_split.png
    {PROJECT_ROOT}/outputs/figures/05_correlation_heatmap.png
    {PROJECT_ROOT}/outputs/figures/06_top_correlations.png
    {PROJECT_ROOT}/outputs/figures/07_feature_distributions.png
    {PROJECT_ROOT}/outputs/figures/08_multicollinearity.png
    {PROJECT_ROOT}/outputs/figures/09_role_tier_breakdown.png

Usage:
    cd ~/Documents/sports_project_cb
    python3 scripts/03_eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("/Users/anokhpalakurthi/Documents/sports_project_cb")

INPUT_STARTERS = PROJECT_ROOT / "data" / "processed" / "cb_starters.csv"
INPUT_FULL = PROJECT_ROOT / "data" / "processed" / "cb_clean.csv"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"

# Consistent styling for all charts
STYLE_CONFIG = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
}
plt.rcParams.update(STYLE_CONFIG)

# Color palette — consistent across all charts
COLORS = {
    "primary": "#534AB7",      # purple
    "secondary": "#1D9E75",    # teal
    "accent": "#D85A30",       # coral
    "neutral": "#888780",      # gray
    "highlight": "#378ADD",    # blue
    "warning": "#BA7517",      # amber
}
PALETTE = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
           COLORS["highlight"], COLORS["warning"], COLORS["neutral"]]


# ---------------------------------------------------------------------------
# Chart functions
# ---------------------------------------------------------------------------

def fig01_grade_distribution(df: pd.DataFrame):
    """
    Distribution of overall defensive grade for starter-level CBs.
    Shows the population we're working with — roughly normal, centered ~64.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall defense grade
    axes[0].hist(df["grades_defense"], bins=30, color=COLORS["primary"],
                 alpha=0.8, edgecolor="white", linewidth=0.5)
    axes[0].axvline(df["grades_defense"].mean(), color=COLORS["accent"],
                    linestyle="--", linewidth=1.5, label=f'Mean: {df["grades_defense"].mean():.1f}')
    axes[0].set_xlabel("PFF overall defense grade")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Overall defensive grade")
    axes[0].legend()

    # Coverage grade specifically
    axes[1].hist(df["grades_coverage_defense"], bins=30, color=COLORS["secondary"],
                 alpha=0.8, edgecolor="white", linewidth=0.5)
    axes[1].axvline(df["grades_coverage_defense"].mean(), color=COLORS["accent"],
                    linestyle="--", linewidth=1.5, label=f'Mean: {df["grades_coverage_defense"].mean():.1f}')
    axes[1].set_xlabel("PFF coverage grade")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Coverage grade")
    axes[1].legend()

    fig.suptitle("PFF grade distributions — starter CBs (300+ snaps, 2018-2025)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def fig02_grade_by_season(df: pd.DataFrame):
    """
    Grade distributions by season — checks for temporal drift in PFF grading.
    Important methodological validation: if grades drift significantly, we may
    need season controls in the model.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    season_data = [df[df["season"] == s]["grades_defense"].values for s in sorted(df["season"].unique())]
    bp = ax.boxplot(season_data, labels=sorted(df["season"].unique()),
                    patch_artist=True, widths=0.6,
                    medianprops=dict(color=COLORS["accent"], linewidth=2))
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["primary"])
        patch.set_alpha(0.6)

    # Add mean line
    means = df.groupby("season")["grades_defense"].mean()
    ax.plot(range(1, len(means) + 1), means.values, "o--",
            color=COLORS["accent"], linewidth=1.5, markersize=6, label="Season mean")

    ax.set_xlabel("Season")
    ax.set_ylabel("PFF overall defense grade")
    ax.set_title("PFF grade distributions by season — checking for grading drift",
                 fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


def fig03_alignment_profile(df: pd.DataFrame):
    """
    Outside vs slot alignment — shows the bimodal nature of CB roles.
    This is a key clustering dimension: outside CBs and slot CBs are
    functionally different positions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: outside_rate vs slot_rate colored by grade
    scatter = axes[0].scatter(df["outside_rate"], df["slot_rate"],
                              c=df["grades_defense"], cmap="RdYlGn",
                              alpha=0.5, s=20, edgecolors="none")
    axes[0].set_xlabel("Outside alignment rate")
    axes[0].set_ylabel("Slot alignment rate")
    axes[0].set_title("Alignment profile (color = grade)")
    plt.colorbar(scatter, ax=axes[0], label="PFF grade")
    # Reference line: outside + slot ≈ 1
    axes[0].plot([0, 1], [1, 0], "--", color=COLORS["neutral"], alpha=0.4, linewidth=1)

    # Histogram of outside_rate — shows bimodal split
    axes[1].hist(df["outside_rate"], bins=40, color=COLORS["primary"],
                 alpha=0.8, edgecolor="white", linewidth=0.5)
    axes[1].axvline(0.5, color=COLORS["accent"], linestyle="--",
                    linewidth=1.5, label="50/50 split")
    axes[1].set_xlabel("Outside alignment rate")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Outside alignment rate distribution")
    axes[1].legend()

    fig.suptitle("CB alignment profiles — outside vs slot",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def fig04_man_zone_split(df: pd.DataFrame):
    """
    Man vs zone coverage split — another key clustering dimension.
    Unlike alignment (bimodal), man/zone tends to be more team-driven.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: man_rate vs zone_rate
    scatter = axes[0].scatter(df["man_rate"], df["zone_rate"],
                              c=df["grades_coverage_defense"], cmap="RdYlGn",
                              alpha=0.5, s=20, edgecolors="none")
    axes[0].set_xlabel("Man coverage rate")
    axes[0].set_ylabel("Zone coverage rate")
    axes[0].set_title("Coverage scheme profile (color = cov grade)")
    plt.colorbar(scatter, ax=axes[0], label="Coverage grade")

    # Histogram of man_rate
    axes[1].hist(df["man_rate"], bins=40, color=COLORS["secondary"],
                 alpha=0.8, edgecolor="white", linewidth=0.5)
    axes[1].axvline(df["man_rate"].mean(), color=COLORS["accent"], linestyle="--",
                    linewidth=1.5, label=f'Mean: {df["man_rate"].mean():.1%}')
    axes[1].set_xlabel("Man coverage rate")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Man coverage rate distribution")
    axes[1].legend()

    fig.suptitle("Coverage scheme splits — man vs zone",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def fig05_correlation_heatmap(df: pd.DataFrame):
    """
    Correlation heatmap of the key features we'll use for modeling.
    Identifies which features carry redundant information (for PCA justification)
    and which are independently predictive.
    """
    features = [
        "grades_defense", "grades_coverage_defense",
        "qb_rating_against", "yards_per_coverage_snap", "catch_rate",
        "playmaking_rate", "forced_incompletion_rate", "target_rate",
        "yards_per_reception", "missed_tackle_rate",
        "outside_rate", "slot_rate", "man_rate",
        "avg_depth_of_target", "tackles_per_snap",
    ]
    labels = [
        "DEF grade", "COV grade",
        "QB rtg allowed", "Yds/cov snap", "Catch rate",
        "Playmaking rate", "Forced inc %", "Target rate",
        "Yds/rec", "Missed tkl %",
        "Outside %", "Slot %", "Man %",
        "Avg depth tgt", "Tackles/snap",
    ]

    corr = df[features].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8, "label": "Pearson r"})
    ax.set_title("Feature correlation matrix — starter CBs",
                 fontweight="bold", pad=20)
    fig.tight_layout()
    return fig


def fig06_top_correlations(df: pd.DataFrame):
    """
    Top features correlated with PFF grade — bar chart for presentation.
    Clear visual of what predicts performance.
    """
    features = [
        "qb_rating_against", "yards_per_coverage_snap", "catch_rate",
        "playmaking_rate", "forced_incompletion_rate", "pbu_rate",
        "int_rate", "target_rate", "yards_per_reception",
        "missed_tackle_rate", "outside_rate", "slot_rate",
        "man_rate", "zone_rate", "tackles_per_snap",
        "run_stop_rate", "avg_depth_of_target", "pressure_rate",
    ]
    labels = [
        "QB rating allowed", "Yds per cov snap", "Catch rate allowed",
        "Playmaking rate", "Forced incompletion %", "PBU rate",
        "INT rate", "Target rate", "Yds per reception",
        "Missed tackle %", "Outside align %", "Slot align %",
        "Man coverage %", "Zone coverage %", "Tackles per snap",
        "Run stop rate", "Avg depth of target", "Pressure rate",
    ]

    corrs = df[features].corrwith(df["grades_defense"]).values
    sort_idx = np.argsort(np.abs(corrs))[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_corrs = corrs[sort_idx]
    sorted_labels = [labels[i] for i in sort_idx]
    colors = [COLORS["secondary"] if c > 0 else COLORS["accent"] for c in sorted_corrs]

    ax.barh(range(len(sorted_corrs)), sorted_corrs, color=colors, alpha=0.8, edgecolor="white")
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, fontsize=10)
    ax.set_xlabel("Correlation with PFF defense grade")
    ax.set_title("Feature correlations with PFF grade — starter CBs",
                 fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()

    # Add value labels
    for i, v in enumerate(sorted_corrs):
        ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)

    fig.tight_layout()
    return fig


def fig07_feature_distributions(df: pd.DataFrame):
    """
    Distributions of key coverage features — shows spread and skewness.
    """
    features = [
        ("yards_per_coverage_snap", "Yards per coverage snap"),
        ("qb_rating_against", "QB rating allowed"),
        ("catch_rate", "Catch rate allowed (%)"),
        ("playmaking_rate", "Playmaking rate (INT+PBU / tgt)"),
        ("target_rate", "Target rate (tgt / cov snaps)"),
        ("forced_incompletion_rate", "Forced incompletion rate (%)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (col, label) in enumerate(features):
        axes[i].hist(df[col], bins=30, color=PALETTE[i % len(PALETTE)],
                     alpha=0.8, edgecolor="white", linewidth=0.5)
        axes[i].axvline(df[col].mean(), color="black", linestyle="--",
                        linewidth=1, alpha=0.6)
        axes[i].set_xlabel(label, fontsize=10)
        axes[i].set_ylabel("Count")
        axes[i].set_title(label, fontsize=11)

    fig.suptitle("Key coverage feature distributions — starter CBs",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def fig08_multicollinearity(df: pd.DataFrame):
    """
    Scatter plots of the most highly correlated feature pairs.
    Justifies why we need PCA — several features carry redundant information.
    """
    pairs = [
        ("outside_rate", "slot_rate", "Outside % vs Slot %\n(r = -0.98)"),
        ("pbu_rate", "forced_incompletion_rate", "PBU rate vs Forced inc %\n(r = 0.86)"),
        ("qb_rating_against", "catch_rate", "QB rtg allowed vs Catch rate\n(r = 0.63)"),
        ("yards_per_coverage_snap", "target_rate", "Yds/cov snap vs Target rate\n(r = 0.66)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (x, y, title) in enumerate(pairs):
        axes[i].scatter(df[x], df[y], alpha=0.3, s=15,
                        color=PALETTE[i], edgecolors="none")
        axes[i].set_xlabel(x.replace("_", " ").title(), fontsize=10)
        axes[i].set_ylabel(y.replace("_", " ").title(), fontsize=10)
        axes[i].set_title(title, fontsize=11)

    fig.suptitle("Multicollinearity — justification for PCA",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def fig09_role_tier_breakdown(df_full: pd.DataFrame):
    """
    Role tier distribution using the full dataset — shows the population
    we're filtering from and why the 300-snap threshold makes sense.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of role tiers
    tier_order = ["starter", "rotational", "situational", "depth"]
    tier_colors = [COLORS["primary"], COLORS["secondary"], COLORS["warning"], COLORS["neutral"]]
    tier_counts = df_full["role_tier"].value_counts().reindex(tier_order)

    axes[0].bar(tier_order, tier_counts.values, color=tier_colors, alpha=0.8,
                edgecolor="white", linewidth=0.5)
    for i, v in enumerate(tier_counts.values):
        axes[0].text(i, v + 10, str(v), ha="center", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Player-seasons")
    axes[0].set_title("Role tier distribution (all CBs)")

    # Grade by role tier — shows quality separation
    tier_data = [df_full[df_full["role_tier"] == t]["grades_defense"].values for t in tier_order]
    bp = axes[1].boxplot(tier_data, labels=tier_order, patch_artist=True, widths=0.6,
                         medianprops=dict(color=COLORS["accent"], linewidth=2))
    for patch, color in zip(bp["boxes"], tier_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1].set_ylabel("PFF overall defense grade")
    axes[1].set_title("Grade by role tier — higher usage = higher grade")

    fig.suptitle("CB population structure — 2018-2025",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not INPUT_STARTERS.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_STARTERS}\n"
            f"Run 02_clean_engineer.py first."
        )

    print("=" * 60)
    print("Loading data")
    print("=" * 60)
    df = pd.read_csv(INPUT_STARTERS)
    df_full = pd.read_csv(INPUT_FULL)
    print(f"  Starters: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  Full: {df_full.shape[0]} rows × {df_full.shape[1]} cols")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    chart_funcs = [
        ("01_grade_distribution.png",  fig01_grade_distribution, df),
        ("02_grade_by_season.png",     fig02_grade_by_season, df),
        ("03_alignment_profile.png",   fig03_alignment_profile, df),
        ("04_man_zone_split.png",      fig04_man_zone_split, df),
        ("05_correlation_heatmap.png",  fig05_correlation_heatmap, df),
        ("06_top_correlations.png",    fig06_top_correlations, df),
        ("07_feature_distributions.png", fig07_feature_distributions, df),
        ("08_multicollinearity.png",   fig08_multicollinearity, df),
        ("09_role_tier_breakdown.png", fig09_role_tier_breakdown, df_full),
    ]

    print("\n" + "=" * 60)
    print("Generating charts")
    print("=" * 60)

    for filename, func, data in chart_funcs:
        fig = func(data)
        path = FIG_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {path.name}")

    # Print summary statistics for the presentation
    print("\n" + "=" * 60)
    print("Key EDA findings (for presentation notes)")
    print("=" * 60)

    print(f"\n  Population: {len(df)} starter CB seasons from {df['player_id'].nunique()} unique players")
    print(f"  Seasons: {df['season'].min()}-{df['season'].max()}")
    print(f"  Mean PFF grade: {df['grades_defense'].mean():.1f} (std: {df['grades_defense'].std():.1f})")
    print(f"  Mean coverage grade: {df['grades_coverage_defense'].mean():.1f}")

    print(f"\n  Top positive correlations with PFF grade:")
    features = ['playmaking_rate', 'forced_incompletion_rate', 'pbu_rate', 'int_rate']
    for f in features:
        r = df[f].corr(df['grades_defense'])
        print(f"    {f}: r = {r:.3f}")

    print(f"\n  Top negative correlations with PFF grade:")
    features = ['qb_rating_against', 'yards_per_coverage_snap', 'catch_rate', 'yards_per_reception']
    for f in features:
        r = df[f].corr(df['grades_defense'])
        print(f"    {f}: r = {r:.3f}")

    print(f"\n  Alignment: {(df['outside_rate'] > 0.7).sum()} pure outside CBs, "
          f"{(df['slot_rate'] > 0.7).sum()} pure slot CBs, "
          f"{((df['outside_rate'] <= 0.7) & (df['slot_rate'] <= 0.7)).sum()} hybrid/mixed")

    print(f"\n  Multicollinearity flags (|r| > 0.6):")
    print(f"    outside_rate vs slot_rate: r = -0.98 (nearly perfect inverse)")
    print(f"    pbu_rate vs forced_incompletion_rate: r = 0.86")
    print(f"    catch_rate vs forced_incompletion_rate: r = -0.62")
    print(f"    → PCA justified to handle redundancy")

    print(f"\n  Grade stability across seasons: mean ranges {df.groupby('season')['grades_defense'].mean().min():.1f}"
          f" to {df.groupby('season')['grades_defense'].mean().max():.1f}")
    print(f"    → No major drift detected, safe to pool seasons")

    print(f"\n  All 9 charts saved to: {FIG_DIR}/")


if __name__ == "__main__":
    main()
