"""
01_ingest_merge.py

Ingests PFF defensive CSVs across 2018-2025 seasons, filters to cornerbacks,
and merges the three relevant file types (defense_summary, defense_coverage_summary,
defense_coverage_scheme) into a single player-season dataset.

Inputs:
    {PROJECT_ROOT}/data/raw/PFF Archives/{YEAR}/Defense/defense_summary.csv
    {PROJECT_ROOT}/data/raw/PFF Archives/{YEAR}/Defense/defense_coverage_summary.csv
    {PROJECT_ROOT}/data/raw/PFF Archives/{YEAR}/Defense/defense_coverage_scheme.csv

Output:
    {PROJECT_ROOT}/data/processed/cb_merged_raw.csv — one row per CB per season

Design notes:
- defense_summary is the spine; we left-join coverage files onto it so we never drop
  a player who appears in defense_summary but is missing from coverage files (e.g.
  a CB who played 0 coverage snaps, rare but possible).
- defense_summary and defense_coverage_summary share 31 columns (identifiers, grades,
  and overlapping stats). We keep the defense_summary versions and only pull the 9
  unique coverage columns (coverage rate stats) from defense_coverage_summary.
- defense_coverage_scheme has man_*/zone_* prefixed columns that don't conflict,
  so we pull everything except identifier columns.
- Column schemas are consistent across all 8 years (verified in EDA), so a single
  load+concat pattern works without per-year special casing.

Usage:
    cd ~/Documents/sports_project_cb
    python3 scripts/01_ingest_merge.py
"""

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Project root — update this one line if you move the project.
# All other scripts in the pipeline (02-07) should use this same root.
PROJECT_ROOT = Path("/Users/anokhpalakurthi/Documents/sports_project_cb")

BASE_DIR = PROJECT_ROOT / "data" / "raw" / "PFF Archives"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cb_merged_raw.csv"

SEASONS = range(2018, 2026)  # 2018-2025 inclusive
POSITION_FILTER = "CB"

# Columns unique to defense_coverage_summary (not present in defense_summary).
# These are the 9 coverage-specific rate stats we want to pull in.
COVERAGE_UNIQUE_COLS = [
    "avg_depth_of_target",
    "coverage_percent",
    "coverage_snaps_per_reception",
    "coverage_snaps_per_target",
    "dropped_ints",
    "forced_incompletes",
    "forced_incompletion_rate",
    "snap_counts_pass_play",
    "yards_per_coverage_snap",
]

# Join keys — player_id + season uniquely identifies a player-season row.
# We verified zero duplicates on this key in the EDA.
JOIN_KEYS = ["player_id", "season"]

# Identifier columns in defense_coverage_scheme that duplicate defense_summary.
# We drop these from the scheme file before merging to avoid _x/_y suffix mess.
SCHEME_DROP_COLS = [
    "player", "position", "team_name", "player_game_count",
    "franchise_id", "penalties", "declined_penalties",
]


# ---------------------------------------------------------------------------
# Load functions
# ---------------------------------------------------------------------------

def load_file_across_seasons(filename: str) -> pd.DataFrame:
    """Load a given CSV filename across all seasons and concatenate with season tag."""
    frames = []
    for season in SEASONS:
        path = BASE_DIR / str(season) / "Defense" / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Expected file not found: {path}\n"
                f"Check that PROJECT_ROOT is set correctly and PFF Archives is in data/raw/"
            )
        df = pd.read_csv(path)
        df["season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def filter_to_cbs(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to cornerback position only."""
    return df[df["position"] == POSITION_FILTER].copy()


# ---------------------------------------------------------------------------
# Main merge
# ---------------------------------------------------------------------------

def main():
    # Verify the project structure exists before we start loading
    if not BASE_DIR.exists():
        raise FileNotFoundError(
            f"PFF Archives folder not found at: {BASE_DIR}\n"
            f"Make sure you've copied 'PFF Archives' into {PROJECT_ROOT}/data/raw/"
        )

    print("=" * 60)
    print("Loading defense_summary (spine)")
    print("=" * 60)
    defense_summary = load_file_across_seasons("defense_summary.csv")
    cb_spine = filter_to_cbs(defense_summary)
    print(f"  Loaded {len(defense_summary)} total rows, {len(cb_spine)} CB rows")

    print("\n" + "=" * 60)
    print("Loading defense_coverage_summary (rate stats)")
    print("=" * 60)
    coverage_summary = load_file_across_seasons("defense_coverage_summary.csv")
    cb_coverage = filter_to_cbs(coverage_summary)
    # Keep only join keys + unique coverage columns to avoid collision with spine
    cb_coverage = cb_coverage[JOIN_KEYS + COVERAGE_UNIQUE_COLS]
    print(f"  Loaded {len(coverage_summary)} total rows, {len(cb_coverage)} CB rows")
    print(f"  Keeping {len(COVERAGE_UNIQUE_COLS)} unique coverage columns")

    print("\n" + "=" * 60)
    print("Loading defense_coverage_scheme (man/zone splits)")
    print("=" * 60)
    coverage_scheme = load_file_across_seasons("defense_coverage_scheme.csv")
    cb_scheme = filter_to_cbs(coverage_scheme)
    # Drop the identifier columns that duplicate what's already in the spine
    drop_existing = [c for c in SCHEME_DROP_COLS if c in cb_scheme.columns]
    cb_scheme = cb_scheme.drop(columns=drop_existing)
    print(f"  Loaded {len(coverage_scheme)} total rows, {len(cb_scheme)} CB rows")
    print(f"  Dropped {len(drop_existing)} duplicate ID cols, keeping {len(cb_scheme.columns) - 2} new cols")

    # ---------------------------------------------------------------------
    # Merge: left-join coverage files onto the defense_summary spine
    # ---------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Merging files on (player_id, season)")
    print("=" * 60)

    merged = cb_spine.merge(cb_coverage, on=JOIN_KEYS, how="left")
    print(f"  After coverage_summary merge: {len(merged)} rows, {len(merged.columns)} cols")

    merged = merged.merge(cb_scheme, on=JOIN_KEYS, how="left")
    print(f"  After coverage_scheme merge:  {len(merged)} rows, {len(merged.columns)} cols")

    # Sanity checks: row count must match spine (we're left-joining)
    assert len(merged) == len(cb_spine), (
        f"Row count changed during merge: {len(cb_spine)} -> {len(merged)}"
    )
    # Duplicate check on join keys
    dupes = merged.duplicated(subset=JOIN_KEYS).sum()
    assert dupes == 0, f"Found {dupes} duplicate player-seasons after merge"

    # Coverage tracking: how many spine rows got coverage data?
    cov_match = merged["yards_per_coverage_snap"].notna().sum()
    scheme_match = merged["base_snap_counts_coverage"].notna().sum()
    print(f"\n  {cov_match}/{len(merged)} ({cov_match/len(merged)*100:.1f}%) CBs matched with coverage_summary")
    print(f"  {scheme_match}/{len(merged)} ({scheme_match/len(merged)*100:.1f}%) CBs matched with coverage_scheme")

    # ---------------------------------------------------------------------
    # Column reorganization: put identifiers first for readability
    # ---------------------------------------------------------------------
    id_cols = ["player", "player_id", "season", "team_name", "position",
               "player_game_count", "franchise_id"]
    id_cols = [c for c in id_cols if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in id_cols]
    merged = merged[id_cols + other_cols]

    # ---------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print("\n" + "=" * 60)
    print(f"Saved to {OUTPUT_PATH}")
    print(f"Final shape: {merged.shape[0]} rows × {merged.shape[1]} cols")
    print("=" * 60)

    # Summary breakdown by season
    print("\nRows per season:")
    print(merged.groupby("season").size().to_string())


if __name__ == "__main__":
    main()