"""
02_clean_engineer.py

Takes the merged CB dataset from 01_ingest_merge.py and produces a clean,
analysis-ready file with engineered features.

Input:
    {PROJECT_ROOT}/data/processed/cb_merged_raw.csv

Outputs:
    {PROJECT_ROOT}/data/processed/cb_clean.csv         — full cleaned dataset (all CBs)
    {PROJECT_ROOT}/data/processed/cb_starters.csv      — starter-level CBs only (300+ snaps)

Cleaning decisions (all documented and justified below):
    1. Drop columns that are irrelevant for CB analysis
    2. Apply snap count threshold for the starter subset
    3. Handle missing data per column category
    4. Engineer rate features for per-snap normalization
    5. Add role tier labels based on snap counts

Usage:
    cd ~/Documents/sports_project_cb
    python3 scripts/02_clean_engineer.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("/Users/anokhpalakurthi/Documents/sports_project_cb")

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cb_merged_raw.csv"
OUTPUT_FULL = PROJECT_ROOT / "data" / "processed" / "cb_clean.csv"
OUTPUT_STARTERS = PROJECT_ROOT / "data" / "processed" / "cb_starters.csv"

# Snap threshold for the "starters" subset used in clustering.
# 300 defensive snaps captures ~53% of CB player-seasons (964 obs) — roughly the
# population of weekly starters and high-usage rotational guys. This filters out
# late-season callups, special teamers, and injured players with partial seasons.
STARTER_SNAP_THRESHOLD = 300

# Columns to drop: DL-specific alignment snaps that are always zero or near-zero
# for CBs. These exist because the export includes all defensive positions.
COLS_TO_DROP = [
    "snap_counts_dl_a_gap",     # always 0 for CBs
    "snap_counts_dl_b_gap",     # always 0 for CBs
    "snap_counts_dl_over_t",    # always 0 for CBs
    "safeties",                 # nearly always 0 for CBs
    "fumble_recovery_touchdowns",  # extremely rare, not a skill indicator
    "interception_touchdowns",  # rare, more about return ability than coverage
    "coverage_percent",         # redundant with snap_counts_coverage / snap_counts_pass_play
    "base_snap_counts_coverage",  # redundant with snap_counts_coverage from other file
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rate-based features from counting stats. Rate stats normalize for
    playing time and are more useful for clustering than raw counts, which are
    heavily correlated with snap volume.

    All rate features use safe division (returns NaN when denominator is 0)
    so we don't introduce inf values for players with 0 targets/snaps.
    """

    def safe_divide(numerator, denominator):
        """Divide where denominator > 0, NaN otherwise."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denominator > 0, numerator / denominator, np.nan)

    # Build all new features as a dict, then concat once to avoid fragmentation
    new = {}

    # --- Coverage rate stats (per-target or per-snap) ---

    # Target rate: how often is this CB targeted relative to coverage snaps?
    # Higher = more targeted, which could mean a high-volume outside CB or a weak link.
    new["target_rate"] = safe_divide(df["targets"].values, df["snap_counts_coverage"].values)

    # Interception rate: INTs per target faced. Ball-hawking ability.
    new["int_rate"] = safe_divide(df["interceptions"].values, df["targets"].values)

    # Pass breakup rate: PBUs per target. Playmaking on the ball.
    new["pbu_rate"] = safe_divide(df["pass_break_ups"].values, df["targets"].values)

    # Playmaking rate: (INTs + PBUs) per target. Combined ball skills.
    new["playmaking_rate"] = safe_divide(
        (df["interceptions"] + df["pass_break_ups"]).values, df["targets"].values
    )

    # --- Pressure/blitz rate ---

    # Pressure rate: pressures per pass rush snap. Identifies blitz-heavy nickel CBs.
    new["pressure_rate"] = safe_divide(
        df["total_pressures"].values, df["snap_counts_pass_rush"].values
    )

    # --- Alignment rates (positional profile) ---

    # Outside/slot/box rates tell us where this CB lines up, which is critical
    # for archetype clustering (outside CB vs slot CB vs hybrid).
    new["outside_rate"] = safe_divide(
        df["snap_counts_corner"].values, df["snap_counts_defense"].values
    )
    new["slot_rate"] = safe_divide(
        df["snap_counts_slot"].values, df["snap_counts_defense"].values
    )
    new["box_rate"] = safe_divide(
        df["snap_counts_box"].values, df["snap_counts_defense"].values
    )

    # --- Man/zone profile ---

    # man_coverage_pct already exists as man_snap_counts_coverage_percent, but
    # let's create a clean version that's a 0-1 ratio instead of 0-100 percent.
    new["man_rate"] = safe_divide(
        df["man_snap_counts_coverage"].values, df["snap_counts_coverage"].values
    )
    new["zone_rate"] = safe_divide(
        df["zone_snap_counts_coverage"].values, df["snap_counts_coverage"].values
    )

    # --- Run defense contribution ---

    # Stops per run defense snap. How effective in run support.
    new["run_stop_rate"] = safe_divide(df["stops"].values, df["snap_counts_run_defense"].values)

    # Tackle efficiency: missed tackle rate already exists, but let's also
    # create tackles per snap for volume context.
    new["tackles_per_snap"] = safe_divide(
        df["tackles"].values, df["snap_counts_defense"].values
    )

    # Concat all new columns at once (avoids pandas fragmentation warning)
    new_df = pd.DataFrame(new, index=df.index)
    return pd.concat([df, new_df], axis=1)


# ---------------------------------------------------------------------------
# Missing data handling
# ---------------------------------------------------------------------------

def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values with column-specific strategies.

    Philosophy: NaN in this dataset is almost always *informative* — it means
    the player didn't have enough of a specific activity to generate a stat
    (e.g. 0 targets = no catch_rate, 0 pass rush snaps = no pass rush grade).
    We handle each case based on what the missingness means.
    """

    # --- grades_pass_rush_defense (39% missing) ---
    # Missing = CB didn't rush the passer enough for PFF to assign a grade.
    # This is meaningful: a pure coverage CB has no pass rush grade by design.
    # Strategy: fill with league-average CB pass rush grade (for CBs who do have one).
    # This is conservative — it says "if you didn't rush, assume you're average at it."
    # Alternative would be to drop the column entirely, but it helps identify
    # blitz-heavy nickel CBs, which is a real archetype.
    cb_prg_mean = df["grades_pass_rush_defense"].mean()
    df["grades_pass_rush_defense"] = df["grades_pass_rush_defense"].fillna(cb_prg_mean)
    print(f"  Pass rush grade: filled {df['grades_pass_rush_defense'].isna().sum()} remaining NaN "
          f"with CB mean ({cb_prg_mean:.1f})")

    # --- grades_coverage_defense, grades_run_defense, grades_tackle ---
    # Small number missing (~1-6%), likely very low-snap players.
    # Fill with position mean for same reason as above.
    for col in ["grades_coverage_defense", "grades_run_defense", "grades_tackle"]:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            col_mean = df[col].mean()
            df[col] = df[col].fillna(col_mean)
            print(f"  {col}: filled {n_missing} NaN with mean ({col_mean:.1f})")

    # --- Coverage rate stats (catch_rate, yards_per_reception, etc.) ---
    # Missing when a CB has 0 targets or 0 receptions. This is meaningful:
    # a CB with 0 targets allowed has *great* coverage, not missing coverage.
    # Strategy: fill rate stats with 0 for the "good" direction.
    # catch_rate=0 means nobody caught on you, yards_per_reception=0 means no receptions.
    rate_fill_zero = [
        "catch_rate", "yards_per_reception", "yards_per_coverage_snap",
        "yards_after_catch", "missed_tackle_rate",
    ]
    for col in rate_fill_zero:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            df[col] = df[col].fillna(0)
            print(f"  {col}: filled {n_missing} NaN with 0 (no activity)")

    # --- Coverage snaps per target/reception ---
    # Missing when 0 targets. High coverage_snaps_per_target = good (rarely targeted).
    # Fill with a high value (max observed) since 0 targets means "never targeted."
    for col in ["coverage_snaps_per_target", "coverage_snaps_per_reception"]:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            fill_val = df[col].max()
            df[col] = df[col].fillna(fill_val)
            print(f"  {col}: filled {n_missing} NaN with max ({fill_val:.1f}) — 0 targets")

    # --- forced_incompletion_rate, avg_depth_of_target ---
    # Missing when 0 targets. Fill with 0 (no targets to force incomplete on).
    for col in ["forced_incompletion_rate", "avg_depth_of_target"]:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            df[col] = df[col].fillna(0)
            print(f"  {col}: filled {n_missing} NaN with 0 (no targets)")

    # --- longest (longest play allowed) ---
    # Missing when 0 receptions allowed. Fill with 0.
    n_missing = df["longest"].isna().sum()
    if n_missing > 0:
        df["longest"] = df["longest"].fillna(0)
        print(f"  longest: filled {n_missing} NaN with 0 (no receptions)")

    # --- Coverage counting stats from coverage_summary ---
    # dropped_ints, forced_incompletes, snap_counts_pass_play are missing for the
    # ~30 CBs who had 0 coverage snaps and didn't appear in coverage_summary.
    # Fill with 0 (no coverage activity).
    cov_count_cols = ["dropped_ints", "forced_incompletes", "snap_counts_pass_play"]
    for col in cov_count_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            df[col] = df[col].fillna(0)
            print(f"  {col}: filled {n_missing} NaN with 0 (no coverage snaps)")

    # --- Man/zone specific stats ---
    # These are missing when a player had 0 snaps in that coverage type.
    # Man stats missing = played only zone. Zone stats missing = played only man.
    # Fill counting stats with 0, rate stats with 0 or NaN depending on use.
    man_zone_count_cols = [
        "man_assists", "man_dropped_ints", "man_forced_incompletes",
        "man_interceptions", "man_missed_tackles", "man_pass_break_ups",
        "man_receptions", "man_snap_counts_coverage", "man_snap_counts_pass_play",
        "man_stops", "man_tackles", "man_targets", "man_touchdowns",
        "man_yards", "man_yards_after_catch",
        "zone_assists", "zone_dropped_ints", "zone_forced_incompletes",
        "zone_interceptions", "zone_missed_tackles", "zone_pass_break_ups",
        "zone_receptions", "zone_snap_counts_coverage", "zone_snap_counts_pass_play",
        "zone_stops", "zone_tackles", "zone_targets", "zone_touchdowns",
        "zone_yards", "zone_yards_after_catch",
    ]
    man_zone_fill_count = 0
    for col in man_zone_count_cols:
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                df[col] = df[col].fillna(0)
                man_zone_fill_count += n
    print(f"  Man/zone counting stats: filled {man_zone_fill_count} total NaN with 0")

    # Man/zone rate stats: fill with 0 when no snaps in that scheme type
    man_zone_rate_cols = [
        "man_catch_rate", "man_avg_depth_of_target", "man_forced_incompletion_rate",
        "man_missed_tackle_rate", "man_qb_rating_against", "man_yards_per_coverage_snap",
        "man_yards_per_reception", "man_coverage_snaps_per_target",
        "man_coverage_snaps_per_reception", "man_coverage_percent",
        "man_snap_counts_coverage_percent", "man_longest",
        "man_grades_coverage_defense",
        "zone_catch_rate", "zone_avg_depth_of_target", "zone_forced_incompletion_rate",
        "zone_missed_tackle_rate", "zone_qb_rating_against", "zone_yards_per_coverage_snap",
        "zone_yards_per_reception", "zone_coverage_snaps_per_target",
        "zone_coverage_snaps_per_reception", "zone_coverage_percent",
        "zone_snap_counts_coverage_percent", "zone_longest",
        "zone_grades_coverage_defense",
    ]
    man_zone_rate_count = 0
    for col in man_zone_rate_cols:
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                df[col] = df[col].fillna(0)
                man_zone_rate_count += n
    print(f"  Man/zone rate stats: filled {man_zone_rate_count} total NaN with 0")

    # --- Engineered features ---
    # Our safe_divide already returns NaN for 0 denominators. For the starter
    # subset these will be minimal, but for the full dataset some low-snap guys
    # will have NaN engineered features. We fill with 0 (no activity).
    engineered_cols = [
        "target_rate", "int_rate", "pbu_rate", "playmaking_rate",
        "pressure_rate", "outside_rate", "slot_rate", "box_rate",
        "man_rate", "zone_rate", "run_stop_rate", "tackles_per_snap",
    ]
    eng_fill_count = 0
    for col in engineered_cols:
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                df[col] = df[col].fillna(0)
                eng_fill_count += n
    print(f"  Engineered rate features: filled {eng_fill_count} total NaN with 0")

    return df


# ---------------------------------------------------------------------------
# Role tier labeling
# ---------------------------------------------------------------------------

def add_role_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a role tier based on defensive snap count. This is NOT used as a
    clustering feature — it's a descriptive label for post-hoc analysis.

    Tiers:
        starter:      500+ snaps (full-time starter, ~16 game seasons)
        rotational:   300-499 snaps (significant contributor, platoon role)
        situational:  100-299 snaps (sub-packages, matchup specific)
        depth:        <100 snaps (backup, injured, late-season callup)
    """
    conditions = [
        df["snap_counts_defense"] >= 500,
        df["snap_counts_defense"] >= 300,
        df["snap_counts_defense"] >= 100,
    ]
    choices = ["starter", "rotational", "situational", "depth"]
    df["role_tier"] = np.select(conditions, choices[:3], default=choices[3])
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            f"Run 01_ingest_merge.py first."
        )

    print("=" * 60)
    print("Loading merged CB data")
    print("=" * 60)
    df = pd.read_csv(INPUT_PATH)
    print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

    # ----- Step 1: Drop irrelevant columns -----
    print("\n" + "=" * 60)
    print("Step 1: Dropping irrelevant columns")
    print("=" * 60)
    existing_drops = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=existing_drops)
    print(f"  Dropped {len(existing_drops)} columns: {existing_drops}")
    print(f"  Remaining: {df.shape[1]} cols")

    # ----- Step 2: Engineer features -----
    print("\n" + "=" * 60)
    print("Step 2: Engineering rate features")
    print("=" * 60)
    df = engineer_features(df)
    new_features = [
        "target_rate", "int_rate", "pbu_rate", "playmaking_rate",
        "pressure_rate", "outside_rate", "slot_rate", "box_rate",
        "man_rate", "zone_rate", "run_stop_rate", "tackles_per_snap",
    ]
    print(f"  Created {len(new_features)} new features:")
    for f in new_features:
        non_null = df[f].notna().sum()
        print(f"    {f}: {non_null}/{len(df)} non-null, mean={df[f].mean():.4f}")

    # ----- Step 3: Handle missing data -----
    print("\n" + "=" * 60)
    print("Step 3: Handling missing data")
    print("=" * 60)
    df = handle_missing_data(df)

    # Final missing data check
    remaining_missing = df.isnull().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    if len(remaining_missing) > 0:
        print(f"\n  WARNING: {len(remaining_missing)} columns still have missing values:")
        print(remaining_missing.to_string())
    else:
        print(f"\n  All missing values handled — 0 NaN remaining")

    # ----- Step 4: Add role tier labels -----
    print("\n" + "=" * 60)
    print("Step 4: Adding role tier labels")
    print("=" * 60)
    df = add_role_tier(df)
    print("  Role tier distribution:")
    print(df["role_tier"].value_counts().to_string())

    # ----- Step 5: Save full dataset -----
    print("\n" + "=" * 60)
    print("Step 5: Saving outputs")
    print("=" * 60)
    OUTPUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FULL, index=False)
    print(f"  Full dataset: {df.shape[0]} rows × {df.shape[1]} cols -> {OUTPUT_FULL}")

    # ----- Step 6: Create starter subset -----
    starters = df[df["snap_counts_defense"] >= STARTER_SNAP_THRESHOLD].copy()
    starters.to_csv(OUTPUT_STARTERS, index=False)
    print(f"  Starter subset ({STARTER_SNAP_THRESHOLD}+ snaps): "
          f"{starters.shape[0]} rows × {starters.shape[1]} cols -> {OUTPUT_STARTERS}")

    # ----- Summary -----
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Input:    {INPUT_PATH.name} ({pd.read_csv(INPUT_PATH).shape[1]} cols)")
    print(f"  Output 1: {OUTPUT_FULL.name} ({df.shape[0]} rows, {df.shape[1]} cols)")
    print(f"  Output 2: {OUTPUT_STARTERS.name} ({starters.shape[0]} rows, {starters.shape[1]} cols)")
    print(f"  Columns dropped: {len(existing_drops)}")
    print(f"  Features added: {len(new_features)}")
    print(f"  Starter threshold: {STARTER_SNAP_THRESHOLD} snaps")
    print(f"\n  Starters per season:")
    print(starters.groupby("season").size().to_string())


if __name__ == "__main__":
    main()
