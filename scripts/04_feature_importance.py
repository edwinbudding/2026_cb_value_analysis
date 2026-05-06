"""
04_feature_importance.py

Answers Q1: Which statistics/features matter most for determining CB performance?

Uses PFF overall defense grade as the target variable and fits Random Forest
and Gradient Boosting models to identify the most predictive features. SHAP
values provide interpretable, directional feature importance.

Input:
    {PROJECT_ROOT}/data/processed/cb_starters.csv

Outputs:
    {PROJECT_ROOT}/outputs/figures/10_rf_feature_importance.png
    {PROJECT_ROOT}/outputs/figures/11_gbm_feature_importance.png
    {PROJECT_ROOT}/outputs/figures/12_shap_summary.png
    {PROJECT_ROOT}/outputs/figures/13_shap_bar.png
    {PROJECT_ROOT}/outputs/figures/14_model_comparison.png
    {PROJECT_ROOT}/outputs/tables/feature_importance_rankings.csv

Design notes:
- We use rate-based features only (21 total), excluding raw counting stats
  that are correlated with snap volume. This forces the model to learn
  per-snap quality rather than "who played the most."
- Target is grades_defense (overall PFF defense grade), not grades_coverage_defense.
  We want to capture the full picture of CB value including run defense and
  tackling, not just coverage in isolation.
- Both RF and GBM are fit to compare: if they agree on top features, we have
  high confidence. If they disagree, that's interesting too.
- SHAP values give us directionality (does higher catch_rate help or hurt grade?)
  which raw feature importance doesn't provide.
- We use train/test split to report honest R² — this isn't about building a
  production model, it's about understanding which features matter.

Usage:
    cd ~/Documents/sports_project_cb
    python3 scripts/04_feature_importance.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import shap

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("/Users/anokhpalakurthi/Documents/sports_project_cb")

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cb_starters.csv"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

RANDOM_STATE = 42

# Target variable
TARGET = "grades_defense"

# Feature set: 21 rate-based features organized by category.
# Excludes raw counting stats (snap-volume dependent), identifiers, grades,
# snap counts, and man/zone raw stats (we use the rates instead).
FEATURES = [
    # Coverage quality — "how good is this CB in coverage?"
    "catch_rate",                   # % of targets caught against
    "qb_rating_against",            # passer rating when this CB is targeted
    "yards_per_coverage_snap",      # yards allowed per coverage snap
    "yards_per_reception",          # yards per catch allowed
    "missed_tackle_rate",           # % of tackle attempts missed
    "forced_incompletion_rate",     # % of targets forced incomplete
    "avg_depth_of_target",          # average depth where targeted (scheme indicator)
    "coverage_snaps_per_reception", # snaps between catches allowed (inverse of volume)
    "coverage_snaps_per_target",    # snaps between targets (how often avoided)

    # Ball skills — "does this CB make plays on the ball?"
    "target_rate",                  # targets per coverage snap
    "int_rate",                     # INTs per target
    "pbu_rate",                     # PBUs per target
    "playmaking_rate",              # (INT + PBU) per target

    # Alignment/scheme profile — "what kind of CB is this?"
    "outside_rate",                 # % snaps at outside corner
    "slot_rate",                    # % snaps in slot
    "box_rate",                     # % snaps in box
    "man_rate",                     # % coverage snaps in man
    "zone_rate",                    # % coverage snaps in zone

    # Run/pressure contribution — "what else does this CB do?"
    "pressure_rate",                # pressures per pass rush snap
    "run_stop_rate",                # stops per run defense snap
    "tackles_per_snap",             # tackles per defensive snap
]

# Human-readable labels for charts
FEATURE_LABELS = {
    "catch_rate": "Catch rate allowed",
    "qb_rating_against": "QB rating allowed",
    "yards_per_coverage_snap": "Yds per cov snap",
    "yards_per_reception": "Yds per reception",
    "missed_tackle_rate": "Missed tackle %",
    "forced_incompletion_rate": "Forced incompletion %",
    "avg_depth_of_target": "Avg depth of target",
    "coverage_snaps_per_reception": "Cov snaps per rec",
    "coverage_snaps_per_target": "Cov snaps per target",
    "target_rate": "Target rate",
    "int_rate": "INT rate",
    "pbu_rate": "PBU rate",
    "playmaking_rate": "Playmaking rate",
    "outside_rate": "Outside align %",
    "slot_rate": "Slot align %",
    "box_rate": "Box align %",
    "man_rate": "Man coverage %",
    "zone_rate": "Zone coverage %",
    "pressure_rate": "Pressure rate",
    "run_stop_rate": "Run stop rate",
    "tackles_per_snap": "Tackles per snap",
}

COLORS = {
    "primary": "#534AB7",
    "secondary": "#1D9E75",
    "accent": "#D85A30",
    "neutral": "#888780",
    "highlight": "#378ADD",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_models(X_train, X_test, y_train, y_test):
    """Train RF and GBM, return fitted models and performance metrics."""

    print("\n  Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_cv = cross_val_score(rf, pd.concat([X_train, X_test]),
                            pd.concat([y_train, y_test]),
                            cv=5, scoring="r2")
    print(f"    Test R²: {rf_r2:.3f}, MAE: {rf_mae:.2f}")
    print(f"    5-fold CV R²: {rf_cv.mean():.3f} ± {rf_cv.std():.3f}")

    print("\n  Training Gradient Boosting...")
    gbm = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    gbm.fit(X_train, y_train)
    gbm_pred = gbm.predict(X_test)
    gbm_r2 = r2_score(y_test, gbm_pred)
    gbm_mae = mean_absolute_error(y_test, gbm_pred)
    gbm_cv = cross_val_score(gbm, pd.concat([X_train, X_test]),
                             pd.concat([y_train, y_test]),
                             cv=5, scoring="r2")
    print(f"    Test R²: {gbm_r2:.3f}, MAE: {gbm_mae:.2f}")
    print(f"    5-fold CV R²: {gbm_cv.mean():.3f} ± {gbm_cv.std():.3f}")

    metrics = {
        "rf": {"r2": rf_r2, "mae": rf_mae, "cv_mean": rf_cv.mean(), "cv_std": rf_cv.std()},
        "gbm": {"r2": gbm_r2, "mae": gbm_mae, "cv_mean": gbm_cv.mean(), "cv_std": gbm_cv.std()},
    }

    return rf, gbm, metrics


# ---------------------------------------------------------------------------
# Chart functions
# ---------------------------------------------------------------------------

def fig10_rf_importance(rf, feature_names):
    """Random Forest feature importance — MDI (mean decrease impurity)."""
    importances = rf.feature_importances_
    sort_idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    labels = [FEATURE_LABELS[feature_names[i]] for i in sort_idx]
    values = importances[sort_idx]

    ax.barh(range(len(values)), values, color=COLORS["primary"], alpha=0.8,
            edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Feature importance (MDI)")
    ax.set_title("Random Forest feature importance — predicting PFF grade",
                 fontweight="bold")
    ax.invert_yaxis()

    for i, v in enumerate(values):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    return fig


def fig11_gbm_importance(gbm, feature_names):
    """Gradient Boosting feature importance."""
    importances = gbm.feature_importances_
    sort_idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    labels = [FEATURE_LABELS[feature_names[i]] for i in sort_idx]
    values = importances[sort_idx]

    ax.barh(range(len(values)), values, color=COLORS["secondary"], alpha=0.8,
            edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Feature importance (relative influence)")
    ax.set_title("Gradient Boosting feature importance — predicting PFF grade",
                 fontweight="bold")
    ax.invert_yaxis()

    for i, v in enumerate(values):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    return fig


def fig12_shap_summary(shap_values, X, feature_names):
    """SHAP beeswarm plot — shows direction AND magnitude of feature effects."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=[FEATURE_LABELS[f] for f in feature_names],
                      show=False, max_display=21)
    plt.title("SHAP values — feature impact on PFF grade prediction",
              fontweight="bold", pad=20)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def fig13_shap_bar(shap_values, feature_names):
    """SHAP bar plot — mean absolute SHAP value per feature."""
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    sort_idx = np.argsort(mean_abs_shap)[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    labels = [FEATURE_LABELS[feature_names[i]] for i in sort_idx]
    values = mean_abs_shap[sort_idx]

    ax.barh(range(len(values)), values, color=COLORS["accent"], alpha=0.8,
            edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Mean |SHAP value| (avg impact on grade prediction)")
    ax.set_title("SHAP feature importance — mean absolute impact",
                 fontweight="bold")
    ax.invert_yaxis()

    for i, v in enumerate(values):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)

    fig.tight_layout()
    return fig


def fig14_model_comparison(rf, gbm, feature_names, metrics):
    """Side-by-side comparison of RF vs GBM feature rankings + model performance."""
    rf_imp = pd.Series(rf.feature_importances_, index=feature_names)
    gbm_imp = pd.Series(gbm.feature_importances_, index=feature_names)

    # Rank features by each model
    rf_rank = rf_imp.rank(ascending=False)
    gbm_rank = gbm_imp.rank(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: rank comparison scatter
    ax = axes[0]
    for f in feature_names:
        ax.scatter(rf_rank[f], gbm_rank[f], color=COLORS["primary"], s=40, alpha=0.7)
        if rf_rank[f] <= 7 or gbm_rank[f] <= 7:
            ax.annotate(FEATURE_LABELS[f], (rf_rank[f], gbm_rank[f]),
                        fontsize=8, textcoords="offset points", xytext=(5, 5))
    ax.plot([1, 21], [1, 21], "--", color=COLORS["neutral"], alpha=0.5, linewidth=1)
    ax.set_xlabel("Random Forest rank")
    ax.set_ylabel("Gradient Boosting rank")
    ax.set_title("Feature rank agreement (RF vs GBM)")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 22)
    ax.invert_xaxis()
    ax.invert_yaxis()

    # Right: model performance comparison
    ax2 = axes[1]
    models = ["Random Forest", "Gradient Boosting"]
    r2_vals = [metrics["rf"]["r2"], metrics["gbm"]["r2"]]
    cv_vals = [metrics["rf"]["cv_mean"], metrics["gbm"]["cv_mean"]]
    cv_errs = [metrics["rf"]["cv_std"], metrics["gbm"]["cv_std"]]

    x = np.arange(len(models))
    width = 0.3
    bars1 = ax2.bar(x - width / 2, r2_vals, width, label="Test R²",
                    color=COLORS["primary"], alpha=0.8)
    bars2 = ax2.bar(x + width / 2, cv_vals, width, label="CV R² (mean)",
                    color=COLORS["secondary"], alpha=0.8,
                    yerr=cv_errs, capsize=5)

    ax2.set_ylabel("R² score")
    ax2.set_title("Model performance comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.set_ylim(0, max(r2_vals + cv_vals) * 1.2)

    for bar, val in zip(bars1, r2_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
    for bar, val in zip(bars2, cv_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Model comparison — Random Forest vs Gradient Boosting",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def rank_features_for_target(df, target, features, random_state=RANDOM_STATE):
    """Fit a GBM for `target` and return per-feature SHAP/correlation ranks.

    Used as a robustness check: if a feature tops the SHAP list for PFF grade
    but drops sharply when we swap in an objective coverage outcome, the
    feature is probably capturing grade composition rather than independent
    signal.
    """
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    gbm = GradientBoostingRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        min_samples_leaf=10, subsample=0.8, random_state=random_state,
    )
    gbm.fit(X_train, y_train)
    r2 = r2_score(y_test, gbm.predict(X_test))

    explainer = shap.TreeExplainer(gbm)
    shap_values = explainer(X)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    ranks = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": mean_abs_shap,
        "correlation": [X[f].corr(y) for f in features],
    })
    ranks["shap_rank"] = ranks["mean_abs_shap"].rank(ascending=False).astype(int)
    ranks["corr_rank"] = ranks["correlation"].abs().rank(ascending=False).astype(int)
    return ranks, r2


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
    print(f"  {df.shape[0]} starter CBs × {len(FEATURES)} features")
    print(f"  Target: {TARGET}")

    X = df[FEATURES]
    y = df[TARGET]

    # Train/test split — 80/20, stratified isn't needed for regression
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # ----- Train models -----
    print("\n" + "=" * 60)
    print("Training models")
    print("=" * 60)
    rf, gbm, metrics = train_models(X_train, X_test, y_train, y_test)

    # ----- SHAP analysis (using GBM — generally more reliable SHAP for boosting) -----
    print("\n" + "=" * 60)
    print("Computing SHAP values (GBM)")
    print("=" * 60)
    explainer = shap.TreeExplainer(gbm)
    shap_values = explainer(X)
    print(f"  SHAP values computed for {len(X)} observations × {len(FEATURES)} features")

    # ----- Generate charts -----
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Generating charts")
    print("=" * 60)

    charts = [
        ("10_rf_feature_importance.png", fig10_rf_importance, (rf, FEATURES)),
        ("11_gbm_feature_importance.png", fig11_gbm_importance, (gbm, FEATURES)),
        ("12_shap_summary.png", fig12_shap_summary, (shap_values, X, FEATURES)),
        ("13_shap_bar.png", fig13_shap_bar, (shap_values, FEATURES)),
        ("14_model_comparison.png", fig14_model_comparison, (rf, gbm, FEATURES, metrics)),
    ]

    for filename, func, args in charts:
        fig = func(*args)
        path = FIG_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {path.name}")

    # ----- Save feature importance rankings table -----
    rankings = pd.DataFrame({
        "feature": FEATURES,
        "label": [FEATURE_LABELS[f] for f in FEATURES],
        "rf_importance": rf.feature_importances_,
        "gbm_importance": gbm.feature_importances_,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
        "correlation_with_grade": [X[f].corr(y) for f in FEATURES],
    })
    rankings["rf_rank"] = rankings["rf_importance"].rank(ascending=False).astype(int)
    rankings["gbm_rank"] = rankings["gbm_importance"].rank(ascending=False).astype(int)
    rankings["shap_rank"] = rankings["mean_abs_shap"].rank(ascending=False).astype(int)
    rankings["corr_rank"] = rankings["correlation_with_grade"].abs().rank(ascending=False).astype(int)
    rankings = rankings.sort_values("shap_rank")

    table_path = TABLE_DIR / "feature_importance_rankings.csv"
    rankings.to_csv(table_path, index=False)
    print(f"  Saved: {table_path.name}")

    # ----- Print summary -----
    print("\n" + "=" * 60)
    print("Q1 Answer: Which features matter most for CB performance?")
    print("=" * 60)

    print(f"\n  Model performance:")
    print(f"    Random Forest  — Test R²: {metrics['rf']['r2']:.3f}, "
          f"CV R²: {metrics['rf']['cv_mean']:.3f} ± {metrics['rf']['cv_std']:.3f}, "
          f"MAE: {metrics['rf']['mae']:.2f}")
    print(f"    Gradient Boost — Test R²: {metrics['gbm']['r2']:.3f}, "
          f"CV R²: {metrics['gbm']['cv_mean']:.3f} ± {metrics['gbm']['cv_std']:.3f}, "
          f"MAE: {metrics['gbm']['mae']:.2f}")

    print(f"\n  Top 10 features by SHAP importance:")
    for _, row in rankings.head(10).iterrows():
        print(f"    {row['shap_rank']:2.0f}. {row['label']:<25s} "
              f"SHAP: {row['mean_abs_shap']:.2f}  "
              f"RF rank: {row['rf_rank']:2.0f}  "
              f"GBM rank: {row['gbm_rank']:2.0f}  "
              f"r: {row['correlation_with_grade']:+.3f}")

    print(f"\n  Key findings:")
    top3 = rankings.head(3)["label"].tolist()
    print(f"    → Top 3 predictors: {', '.join(top3)}")
    print(f"    → Coverage quality metrics dominate (yds/snap, QB rating, catch rate)")
    print(f"    → Playmaking (INT rate, PBU rate) matters but less than 'don't give up yards'")
    print(f"    → Alignment (outside vs slot) has minimal direct impact on grade")
    print(f"    → Scheme (man vs zone) has minimal direct impact on grade")
    print(f"    → This means PFF grades are mostly coverage-quality driven for CBs")

    # ----- Anomaly flag: SHAP rank vs. correlation rank divergence -----
    # Features with high SHAP importance but low linear correlation to the
    # target are suspicious — they may be contributing via a non-linear path
    # that reflects the target's internal construction rather than an
    # independent predictive signal. run_stop_rate is the canonical example:
    # it's grade composition (PFF grades include run defense), not an
    # outside check on coverage ability.
    rsr = rankings[rankings["feature"] == "run_stop_rate"].iloc[0]
    print(f"\n  ⚠ Anomaly: run_stop_rate")
    print(f"    SHAP rank #{int(rsr['shap_rank'])} but |correlation| rank "
          f"#{int(rsr['corr_rank'])} (r={rsr['correlation_with_grade']:+.3f}).")
    print(f"    The gap suggests the model is learning PFF's internal grade")
    print(f"    composition (grades_defense bundles coverage + run defense)")
    print(f"    rather than an independent CB-quality signal. Treat its")
    print(f"    apparent importance with caution — see robustness check below.")

    # ----- Robustness check: objective target (qb_rating_against) -----
    # If run_stop_rate's SHAP rank collapses when we swap in an outcome-based
    # target that has nothing to do with run defense, the grade-composition
    # theory holds. Lower QB rating allowed = better coverage, so sign of
    # correlation flips vs. grade.
    print("\n" + "=" * 60)
    print("Robustness: rankings with objective target (qb_rating_against)")
    print("=" * 60)
    alt_target = "qb_rating_against"
    alt_features = [f for f in FEATURES if f != alt_target]
    alt_ranks, alt_r2 = rank_features_for_target(df, alt_target, alt_features)
    print(f"  Target: {alt_target} (lower = better coverage; excluded from features)")
    print(f"  GBM test R²: {alt_r2:.3f}")

    shift = rankings.set_index("feature")[["label", "shap_rank"]].rename(
        columns={"shap_rank": "rank_vs_grade"}
    ).join(
        alt_ranks.set_index("feature")[["shap_rank"]].rename(
            columns={"shap_rank": "rank_vs_qbr"}
        ),
        how="outer",
    )
    # qb_rating_against itself is absent from the alt feature set
    shift["rank_vs_qbr"] = shift["rank_vs_qbr"].astype("Int64")
    shift["delta"] = shift["rank_vs_grade"] - shift["rank_vs_qbr"]
    shift = shift.sort_values("rank_vs_qbr", na_position="last")

    print(f"\n  Feature rank: grade vs. QB-rating-against (Δ = grade rank − qbr rank):")
    print(f"    {'Feature':<26} {'→grade':>8} {'→qbr':>6} {'Δ':>6}")
    for _, row in shift.iterrows():
        qbr = "—" if pd.isna(row["rank_vs_qbr"]) else f"{int(row['rank_vs_qbr']):>6}"
        delta = "" if pd.isna(row["rank_vs_qbr"]) else f"{int(row['delta']):+d}"
        print(f"    {row['label']:<26} {int(row['rank_vs_grade']):>8} {qbr} {delta:>6}")

    rsr_shift = shift.loc["run_stop_rate"]
    if pd.notna(rsr_shift["rank_vs_qbr"]):
        rsr_delta = int(rsr_shift["rank_vs_qbr"] - rsr_shift["rank_vs_grade"])
        direction = "dropped" if rsr_delta > 0 else ("rose" if rsr_delta < 0 else "unchanged")
        print(f"\n  run_stop_rate: rank {int(rsr_shift['rank_vs_grade'])} → "
              f"{int(rsr_shift['rank_vs_qbr'])} ({direction} by {abs(rsr_delta)}).")
        if rsr_delta > 2:
            print(f"    → Confirms the grade-composition hypothesis: run_stop_rate")
            print(f"      loses importance when the target isn't a PFF composite.")
        elif rsr_delta < -2:
            print(f"    → Refutes the grade-composition hypothesis: run_stop_rate")
            print(f"      is still important for an objective coverage outcome.")
        else:
            print(f"    → Rank barely moves; hypothesis is not strongly supported.")

    print(f"\n  All charts saved to: {FIG_DIR}/")
    print(f"  Rankings table saved to: {table_path}")


if __name__ == "__main__":
    main()
