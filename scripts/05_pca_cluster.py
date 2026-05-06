"""
05_pca_cluster.py

Answers Q2a: What distinct CB archetypes emerge from clustering?

Applies PCA for dimensionality reduction, then k-means clustering to identify
CB performance archetypes. Each cluster is profiled and named.

Input:
    {PROJECT_ROOT}/data/processed/cb_starters.csv

Outputs:
    {PROJECT_ROOT}/data/processed/cb_clustered.csv          — starters with cluster labels
    {PROJECT_ROOT}/outputs/figures/15_pca_variance.png       — scree plot + cumulative variance
    {PROJECT_ROOT}/outputs/figures/16_elbow_silhouette.png   — k selection diagnostics
    {PROJECT_ROOT}/outputs/figures/17_cluster_scatter.png    — PCA scatter colored by cluster
    {PROJECT_ROOT}/outputs/figures/18_cluster_profiles.png   — radar/bar profiles per archetype
    {PROJECT_ROOT}/outputs/figures/19_cluster_grade_dist.png — grade distributions by archetype
    {PROJECT_ROOT}/outputs/tables/cluster_profiles.csv       — summary stats per cluster
    {PROJECT_ROOT}/outputs/tables/pca_loadings.csv           — PCA component loadings

Design notes:
- We use the same 21 rate-based features from script 04.
- Features are standardized (z-scored) before PCA — essential since features
  are on different scales (rates 0-1 vs percentages 0-100).
- 7 PCA components capture ~81% of variance. This is the input to k-means.
- k=5 selected based on archetype interpretability. Silhouette score favors k=2
  (typical for continuous data), but k=5 produces the most football-meaningful
  archetypes. DBSCAN was tested and rejected — no clear density structure.
- Archetypes are named based on their statistical profiles.

Usage:
    cd ~/Documents/sports_project_cb
    python3 scripts/05_pca_cluster.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import joblib
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("/Users/anokhpalakurthi/Documents/sports_project_cb")

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cb_starters.csv"
OUTPUT_CLUSTERED = PROJECT_ROOT / "data" / "processed" / "cb_clustered.csv"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

RANDOM_STATE = 42
N_PCA_COMPONENTS = 7   # captures ~81% variance
N_CLUSTERS = 5          # selected for archetype interpretability

FEATURES = [
    "catch_rate", "qb_rating_against", "yards_per_coverage_snap",
    "yards_per_reception", "missed_tackle_rate", "forced_incompletion_rate",
    "avg_depth_of_target", "coverage_snaps_per_reception", "coverage_snaps_per_target",
    "target_rate", "int_rate", "pbu_rate", "playmaking_rate",
    "outside_rate", "slot_rate", "box_rate",
    "man_rate", "zone_rate",
    "pressure_rate", "run_stop_rate", "tackles_per_snap",
]

FEATURE_LABELS = {
    "catch_rate": "Catch rate", "qb_rating_against": "QB rtg allowed",
    "yards_per_coverage_snap": "Yds/cov snap", "yards_per_reception": "Yds/rec",
    "missed_tackle_rate": "Missed tkl %", "forced_incompletion_rate": "Forced inc %",
    "avg_depth_of_target": "Avg depth tgt", "coverage_snaps_per_reception": "Snaps/rec",
    "coverage_snaps_per_target": "Snaps/tgt", "target_rate": "Target rate",
    "int_rate": "INT rate", "pbu_rate": "PBU rate", "playmaking_rate": "Playmaking",
    "outside_rate": "Outside %", "slot_rate": "Slot %", "box_rate": "Box %",
    "man_rate": "Man %", "zone_rate": "Zone %",
    "pressure_rate": "Pressure rate", "run_stop_rate": "Run stop rate",
    "tackles_per_snap": "Tackles/snap",
}

COLORS = {
    "primary": "#534AB7", "secondary": "#1D9E75", "accent": "#D85A30",
    "highlight": "#378ADD", "warning": "#BA7517", "neutral": "#888780",
}
CLUSTER_COLORS = ["#534AB7", "#1D9E75", "#D85A30", "#378ADD", "#BA7517"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
})


# ---------------------------------------------------------------------------
# Archetype naming
# ---------------------------------------------------------------------------

def name_archetypes(profiles: pd.DataFrame) -> dict:
    """
    Assign football-meaningful names to clusters based on their statistical profiles.
    Returns a dict mapping cluster number to (name, short_description).

    Logic: rank clusters on key dimensions and assign names based on the
    combination of coverage quality, alignment, and scheme.
    """
    names = {}
    for cluster_id in profiles.index:
        row = profiles.loc[cluster_id]
        grade = row["grades_defense"]
        qb_rtg = row["qb_rating_against"]
        yds_snap = row["yards_per_coverage_snap"]
        playmaking = row["playmaking_rate"]
        outside = row["outside_rate"]
        slot = row["slot_rate"]
        man_pct = row["man_rate"]

        # Slot specialist: >60% slot snaps
        if slot > 0.6:
            if grade >= 65:
                names[cluster_id] = ("Quality slot CB", "Slot specialist with above-average coverage")
            else:
                names[cluster_id] = ("Slot specialist", "Slot-primary CB, average production")
        # Outside CBs — differentiate by quality
        elif outside > 0.6:
            if grade >= 68 and qb_rtg <= 85:
                names[cluster_id] = ("Elite lockdown CB", "Top-tier outside corner, shutdown coverage")
            elif grade >= 64 and playmaking >= 0.12:
                names[cluster_id] = ("Playmaking outside CB", "Solid outside corner with ball skills")
            elif grade >= 60:
                names[cluster_id] = ("Average outside CB", "Competent starter, middling production")
            else:
                names[cluster_id] = ("Replacement-level CB", "Below-average outside corner")
        else:
            # Hybrid/versatile
            names[cluster_id] = ("Versatile CB", "Mixed alignment, scheme-flexible")

    return names


# ---------------------------------------------------------------------------
# Chart functions
# ---------------------------------------------------------------------------

def fig15_pca_variance(pca_full):
    """Scree plot + cumulative variance explained."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    n = len(pca_full.explained_variance_ratio_)
    x = range(1, n + 1)

    # Scree plot
    axes[0].bar(x, pca_full.explained_variance_ratio_, color=COLORS["primary"],
                alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Variance explained")
    axes[0].set_title("Scree plot")
    axes[0].set_xticks(range(1, n + 1, 2))

    # Cumulative variance
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    axes[1].plot(x, cumvar, "o-", color=COLORS["primary"], linewidth=2, markersize=5)
    axes[1].axhline(0.8, color=COLORS["accent"], linestyle="--", linewidth=1,
                     label="80% threshold")
    axes[1].axvline(N_PCA_COMPONENTS, color=COLORS["secondary"], linestyle="--",
                     linewidth=1, label=f"{N_PCA_COMPONENTS} components selected")
    axes[1].fill_between(range(1, N_PCA_COMPONENTS + 1),
                          cumvar[:N_PCA_COMPONENTS], alpha=0.15, color=COLORS["primary"])
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative variance explained")
    axes[1].set_title("Cumulative variance")
    axes[1].legend(fontsize=10)
    axes[1].set_xticks(range(1, n + 1, 2))

    fig.suptitle("PCA variance analysis — 21 features → 7 components (81% variance)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def fig16_elbow_silhouette(X_pca):
    """Elbow plot + silhouette scores for k selection."""
    k_range = range(2, 11)
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_pca, labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Elbow
    axes[0].plot(list(k_range), inertias, "o-", color=COLORS["primary"],
                 linewidth=2, markersize=6)
    axes[0].axvline(N_CLUSTERS, color=COLORS["accent"], linestyle="--",
                     linewidth=1.5, label=f"k={N_CLUSTERS} selected")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia (within-cluster sum of squares)")
    axes[0].set_title("Elbow method")
    axes[0].legend()

    # Silhouette
    axes[1].plot(list(k_range), silhouettes, "o-", color=COLORS["secondary"],
                 linewidth=2, markersize=6)
    axes[1].axvline(N_CLUSTERS, color=COLORS["accent"], linestyle="--",
                     linewidth=1.5, label=f"k={N_CLUSTERS} selected")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette analysis")
    axes[1].legend()

    fig.suptitle("Cluster selection diagnostics — k=5 chosen for interpretability",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def fig17_cluster_scatter(X_pca, labels, archetype_names):
    """PCA scatter plot colored by cluster — PC1 vs PC2 and PC1 vs PC3."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        name = archetype_names[cluster_id][0]
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]

        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=name,
                        alpha=0.5, s=20, edgecolors="none")
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 2], c=color, label=name,
                        alpha=0.5, s=20, edgecolors="none")

    axes[0].set_xlabel("PC1 (alignment + coverage volume)")
    axes[0].set_ylabel("PC2 (yards allowed efficiency)")
    axes[0].set_title("PC1 vs PC2")
    axes[0].legend(fontsize=9, loc="best")

    axes[1].set_xlabel("PC1 (alignment + coverage volume)")
    axes[1].set_ylabel("PC3 (playmaking vs yards allowed)")
    axes[1].set_title("PC1 vs PC3")
    axes[1].legend(fontsize=9, loc="best")

    fig.suptitle("CB archetypes in PCA space",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def fig18_cluster_profiles(profiles, archetype_names):
    """Bar chart profiles for each archetype across key metrics."""
    display_features = [
        "grades_defense", "qb_rating_against", "yards_per_coverage_snap",
        "playmaking_rate", "catch_rate", "forced_incompletion_rate",
        "outside_rate", "slot_rate", "man_rate", "missed_tackle_rate",
        "run_stop_rate",
    ]
    display_labels = [
        "PFF grade", "QB rtg allowed", "Yds/cov snap",
        "Playmaking rate", "Catch rate", "Forced inc %",
        "Outside %", "Slot %", "Man %", "Missed tkl %",
        "Run stop rate",
    ]

    n_clusters = len(profiles)
    n_features = len(display_features)

    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 3 * n_clusters), sharex=True)

    for i, cluster_id in enumerate(profiles.index):
        ax = axes[i]
        name = archetype_names[cluster_id][0]
        desc = archetype_names[cluster_id][1]
        count = int(profiles.loc[cluster_id, "count"])
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]

        values = profiles.loc[cluster_id, display_features].values

        # Normalize each feature to 0-1 range across clusters for comparable bars
        all_vals = profiles[display_features].values
        mins = all_vals.min(axis=0)
        maxs = all_vals.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1
        normalized = (values - mins) / ranges

        ax.barh(range(n_features), normalized, color=color, alpha=0.8, edgecolor="white")
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(display_labels, fontsize=10)
        ax.set_title(f"{name} (n={count}) — {desc}", fontweight="bold", fontsize=12)
        ax.set_xlim(0, 1.15)
        ax.invert_yaxis()

        # Add raw values as text
        for j, (norm_v, raw_v) in enumerate(zip(normalized, values)):
            ax.text(norm_v + 0.02, j, f"{raw_v:.2f}", va="center", fontsize=9)

    axes[-1].set_xlabel("Relative value (normalized across archetypes)")
    fig.suptitle("CB archetype profiles — key metrics comparison",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def fig19_cluster_grade_dist(df, archetype_names):
    """Grade distributions by archetype — shows quality separation."""
    fig, ax = plt.subplots(figsize=(12, 6))

    cluster_ids = sorted(df["cluster"].unique())
    for cluster_id in cluster_ids:
        mask = df["cluster"] == cluster_id
        name = archetype_names[cluster_id][0]
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        ax.hist(df.loc[mask, "grades_defense"], bins=20, alpha=0.5,
                color=color, label=f"{name} (n={mask.sum()})", edgecolor="white")

    ax.set_xlabel("PFF overall defense grade")
    ax.set_ylabel("Count")
    ax.set_title("Grade distributions by CB archetype", fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            f"Run 02_clean_engineer.py first."
        )

    print("=" * 60)
    print("Loading data")
    print("=" * 60)
    df = pd.read_csv(INPUT_PATH)
    X = df[FEATURES]
    print(f"  {len(df)} starter CBs × {len(FEATURES)} features")

    # ----- Standardize -----
    print("\n" + "=" * 60)
    print("Standardizing features")
    print("=" * 60)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  All features z-scored (mean=0, std=1)")

    # ----- PCA -----
    print("\n" + "=" * 60)
    print("Running PCA")
    print("=" * 60)
    # Full PCA for variance analysis charts
    pca_full = PCA()
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    print(f"  Components for 80% variance: {np.argmax(cumvar >= 0.80) + 1}")
    print(f"  Components for 90% variance: {np.argmax(cumvar >= 0.90) + 1}")

    # Reduced PCA for clustering
    pca = PCA(n_components=N_PCA_COMPONENTS)
    X_pca = pca.fit_transform(X_scaled)
    print(f"  Using {N_PCA_COMPONENTS} components ({cumvar[N_PCA_COMPONENTS - 1]:.1%} variance)")

    # Interpret components
    print(f"\n  Component interpretations:")
    component_names = []
    for pc in range(min(4, N_PCA_COMPONENTS)):
        loadings = pd.Series(pca.components_[pc], index=FEATURES)
        top2 = loadings.abs().nlargest(3).index.tolist()
        top_labels = [FEATURE_LABELS[f] for f in top2]
        pct = pca.explained_variance_ratio_[pc]
        print(f"    PC{pc + 1} ({pct:.1%}): {', '.join(top_labels)}")

    # ----- K-Means Clustering -----
    print("\n" + "=" * 60)
    print(f"Clustering (k-means, k={N_CLUSTERS})")
    print("=" * 60)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels)
    print(f"  Silhouette score: {sil:.3f}")
    print(f"  Cluster sizes: {np.bincount(labels).tolist()}")

    # ----- Stability check: 20 re-runs with different seeds -----
    # ARI measures agreement with the primary clustering up to label
    # permutation. ARI ~1.0 means the archetypes are reproducible; ARI
    # near 0 means k-means is chasing noise and the archetype names
    # shouldn't be trusted. We use the same n_init=20 so each re-run
    # is itself a best-of-20 — we're measuring seed sensitivity of the
    # solution k-means converges to, not sensitivity of a single init.
    aris = []
    for seed in range(RANDOM_STATE + 1, RANDOM_STATE + 21):
        km_alt = KMeans(n_clusters=N_CLUSTERS, random_state=seed, n_init=20)
        labels_alt = km_alt.fit_predict(X_pca)
        aris.append(adjusted_rand_score(labels, labels_alt))
    print(f"  Bootstrap stability (20 re-seeds): "
          f"mean ARI = {np.mean(aris):.3f}, "
          f"min = {np.min(aris):.3f}, max = {np.max(aris):.3f}")

    df["cluster"] = labels

    # ----- Profile and name archetypes -----
    print("\n" + "=" * 60)
    print("Profiling archetypes")
    print("=" * 60)

    profile_cols = [
        "grades_defense", "grades_coverage_defense", "qb_rating_against",
        "yards_per_coverage_snap", "playmaking_rate", "catch_rate",
        "forced_incompletion_rate", "int_rate", "pbu_rate",
        "outside_rate", "slot_rate", "box_rate", "man_rate", "zone_rate",
        "missed_tackle_rate", "run_stop_rate", "tackles_per_snap",
        "avg_depth_of_target", "target_rate", "snap_counts_defense",
    ]

    profiles = df.groupby("cluster")[profile_cols].mean().round(3)
    counts = df.groupby("cluster").size()
    profiles.insert(0, "count", counts)

    archetype_names = name_archetypes(profiles)

    for cluster_id in sorted(profiles.index):
        name, desc = archetype_names[cluster_id]
        n = int(profiles.loc[cluster_id, "count"])
        grade = profiles.loc[cluster_id, "grades_defense"]
        print(f"\n  Cluster {cluster_id}: {name} (n={n})")
        print(f"    {desc}")
        print(f"    Grade: {grade:.1f} | QB rtg: {profiles.loc[cluster_id, 'qb_rating_against']:.1f} | "
              f"Yds/snap: {profiles.loc[cluster_id, 'yards_per_coverage_snap']:.2f} | "
              f"Playmaking: {profiles.loc[cluster_id, 'playmaking_rate']:.3f}")
        print(f"    Outside: {profiles.loc[cluster_id, 'outside_rate']:.1%} | "
              f"Slot: {profiles.loc[cluster_id, 'slot_rate']:.1%} | "
              f"Man: {profiles.loc[cluster_id, 'man_rate']:.1%}")

    # Add archetype name to dataframe
    df["archetype"] = df["cluster"].map(lambda c: archetype_names[c][0])

    # ----- Generate charts -----
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Generating charts")
    print("=" * 60)

    charts = [
        ("15_pca_variance.png", fig15_pca_variance, (pca_full,)),
        ("16_elbow_silhouette.png", fig16_elbow_silhouette, (X_pca,)),
        ("17_cluster_scatter.png", fig17_cluster_scatter, (X_pca, labels, archetype_names)),
        ("18_cluster_profiles.png", fig18_cluster_profiles, (profiles, archetype_names)),
        ("19_cluster_grade_dist.png", fig19_cluster_grade_dist, (df, archetype_names)),
    ]

    for filename, func, args in charts:
        fig = func(*args)
        path = FIG_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {path.name}")

    # ----- Save tables -----
    # Cluster profiles
    profiles["archetype"] = [archetype_names[c][0] for c in profiles.index]
    profiles.to_csv(TABLE_DIR / "cluster_profiles.csv")
    print(f"  Saved: cluster_profiles.csv")

    # PCA loadings
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=FEATURES,
        columns=[f"PC{i + 1}" for i in range(N_PCA_COMPONENTS)]
    )
    loadings_df.index.name = "feature"
    loadings_df.to_csv(TABLE_DIR / "pca_loadings.csv")
    print(f"  Saved: pca_loadings.csv")

    # Persist fitted scaler and PCA for reuse
    joblib.dump(scaler, TABLE_DIR / "scaler.joblib")
    joblib.dump(pca, TABLE_DIR / "pca.joblib")
    print(f"  Saved: scaler.joblib, pca.joblib")

    # Clustered dataset
    OUTPUT_CLUSTERED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CLUSTERED, index=False)
    print(f"  Saved: {OUTPUT_CLUSTERED.name} ({len(df)} rows)")

    # ----- Summary -----
    print("\n" + "=" * 60)
    print("Q2a Answer: CB archetypes")
    print("=" * 60)
    print(f"\n  {N_CLUSTERS} archetypes identified from {len(df)} starter CB seasons:")
    for cluster_id in sorted(archetype_names.keys()):
        name, desc = archetype_names[cluster_id]
        n = int(profiles.loc[cluster_id, "count"])
        grade = profiles.loc[cluster_id, "grades_defense"]
        print(f"    {cluster_id}. {name} (n={n}, avg grade: {grade:.1f})")

    # Notable players per archetype
    print(f"\n  Example players per archetype (highest grade in each):")
    for cluster_id in sorted(archetype_names.keys()):
        name = archetype_names[cluster_id][0]
        cluster_df = df[df["cluster"] == cluster_id].nlargest(3, "grades_defense")
        players = ", ".join(f"{r['player']} ({r['season']})" for _, r in cluster_df.iterrows())
        print(f"    {name}: {players}")


if __name__ == "__main__":
    main()
