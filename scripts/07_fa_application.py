"""
07_fa_application.py

Answers Q3: For the 2026 free agent CB class, what does each player's cluster
membership suggest about their fair market value?

Identifies 2026 CB free agents, assigns them to archetypes based on their most
recent performance data, and estimates fair contract value based on historical
contract distributions within each archetype.

Inputs:
    {PROJECT_ROOT}/data/processed/cb_with_contracts.csv
    {PROJECT_ROOT}/data/raw/cb_contracts_otc.csv
    {PROJECT_ROOT}/data/raw/salary_cap_history.csv

Outputs:
    {PROJECT_ROOT}/outputs/figures/24_fa_archetype_breakdown.png
    {PROJECT_ROOT}/outputs/figures/25_fa_value_estimates.png
    {PROJECT_ROOT}/outputs/figures/26_fa_key_targets.png
    {PROJECT_ROOT}/outputs/tables/fa_2026_valuations.csv

Design notes:
- A 2026 FA is defined as a player whose contract end_year = 2025 (their
  last season under contract is 2025, so they hit free agency in 2026).
- Each FA is assigned to an archetype based on their most recent season
  in our clustered dataset.
- Fair value is estimated using the historical cap-unit distribution for
  that archetype: we report the median, 25th/75th percentile, and mean.
- We project the 2026 salary cap using the historical growth rate to
  convert cap units back to dollar estimates.

Usage:
    cd ~/Documents/sports_project_cb
    python3 scripts/07_fa_application.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("/Users/anokhpalakurthi/Documents/sports_project_cb")

INPUT_CLUSTERED = PROJECT_ROOT / "data" / "processed" / "cb_with_contracts.csv"
INPUT_CONTRACTS = PROJECT_ROOT / "data" / "raw" / "cb_contracts_otc.csv"
INPUT_CAP = PROJECT_ROOT / "data" / "raw" / "salary_cap_history.csv"

FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

# Reuse the name map from script 06
NAME_MAP = {
    "Antonio Hamilton Sr.": "Antonio Hamilton", "Beanie Bishop Jr.": "Beanie Bishop",
    "Byron Murphy Jr.": "Byron Murphy", "Carlton Davis III": "Carlton Davis",
    "Casey Hayward Jr.": "Casey Hayward", "David Long Jr.": "David Long",
    "Desmond King II": "Desmond King", "Greg Stroman Jr.": "Greg Stroman",
    "Jarvis Brownlee Jr.": "Jarvis Brownlee", "Keith Taylor Jr.": "Keith Taylor",
    "Kenny Moore II": "Kenny Moore", "Martin Emerson Jr.": "Martin Emerson",
    "Pat Surtain II": "Patrick Surtain II", "Sidney Jones IV": "Sidney Jones",
    "Tramaine Brock Sr.": "Tramaine Brock", "Troy Pride Jr.": "Troy Pride, Jr.",
    "Vernon Hargreaves III": "Vernon Hargreaves", "DJ Turner II": "D.J. Turner",
    "CJ Henderson": "C.J. Henderson", "Cor'Dale Flott": "Cordale Flott",
    "Dax Hill": "Daxton Hill", "DeAndre Baker": "Deandre Baker",
    "Dont'e Deayon": "Donte Deayon", "JuJu Brents": "Julius Brents",
    "Coby Bryant": "Cobee Bryant", "Sauce Gardner": "Ahmad Gardner",
    "C.J. Gardner-Johnson": "Chauncey Gardner-Johnson, Jr.",
    "Ugo Amadi": "Ugochukwu Amadi", "Billy Bowman Jr.": "Billy Bowman",
    "Tre Hawkins III": "Tre Hawkins", "Samuel Womack III": "Sam Womack",
}
REVERSE_MAP = {v: k for k, v in NAME_MAP.items()}

CLUSTER_COLORS = ["#534AB7", "#1D9E75", "#D85A30", "#378ADD", "#BA7517"]
ARCHETYPE_COLOR_MAP = {
    "Elite lockdown CB": "#1D9E75",
    "Playmaking outside CB": "#D85A30",
    "Average outside CB": "#534AB7",
    "Slot specialist": "#BA7517",
    "Replacement-level CB": "#378ADD",
}

COLORS = {
    "primary": "#534AB7", "secondary": "#1D9E75", "accent": "#D85A30",
    "highlight": "#378ADD", "warning": "#BA7517", "neutral": "#888780",
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
})


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def project_2026_cap(cap_history: pd.DataFrame) -> float:
    """
    Project the 2026 salary cap using compound annual growth rate (CAGR).
    Excludes 2020-2021 COVID dip from the growth calculation for a cleaner
    trend line, then applies the CAGR to the 2025 cap.
    """
    # Use 2018-2019 and 2022-2025 for growth rate (skip COVID dip)
    non_covid = cap_history[~cap_history["season"].isin([2020, 2021])].sort_values("season")
    first_year = non_covid.iloc[0]
    last_year = non_covid.iloc[-1]
    n_years = last_year["season"] - first_year["season"]
    cagr = (last_year["salary_cap"] / first_year["salary_cap"]) ** (1 / n_years) - 1

    cap_2025 = cap_history[cap_history["season"] == 2025]["salary_cap"].values[0]
    cap_2026 = cap_2025 * (1 + cagr)

    print(f"  CAGR (ex-COVID): {cagr:.1%}")
    print(f"  2025 cap: ${cap_2025:,.0f}")
    print(f"  2026 projected cap: ${cap_2026:,.0f}")

    return cap_2026


def identify_fa_class(contracts: pd.DataFrame) -> pd.DataFrame:
    """Identify 2026 free agents (contracts ending in 2025)."""
    fa = contracts[contracts["end_year"] == 2025].copy()
    # Take most recent contract per player
    fa = fa.sort_values("signing_year", ascending=False).drop_duplicates("player", keep="first")
    fa["pff_name"] = fa["player"].map(lambda x: REVERSE_MAP.get(x, x))
    return fa


def assign_archetypes(fa: pd.DataFrame, clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each FA to an archetype based on their most recent season
    in our clustered dataset.

    Note: this is a lookup-based assignment — we read the archetype label
    already attached to the player's most recent row in cb_clustered.csv,
    rather than re-scoring their features through the scaler/PCA/k-means
    pipeline from script 05. A player who has no row in the 2018–2025
    clustered dataset (e.g., a 2024 rookie with no PFF-clustered season,
    or a player whose only seasons were outside the starter threshold)
    is silently skipped here and will not appear in the valued output.
    To score such a player, load scaler.joblib and pca.joblib from
    outputs/tables/ and fit the k-means centroids to their features.
    """
    fa_profiles = []

    for _, row in fa.iterrows():
        pff_name = row["pff_name"]
        player_data = clustered[clustered["player"] == pff_name].sort_values("season", ascending=False)

        if len(player_data) > 0:
            latest = player_data.iloc[0]
            # True if the player's last 2 clustered seasons share an archetype.
            # Single-season players default to True (no evidence of drift).
            if len(player_data) >= 2:
                stable = bool(latest["archetype"] == player_data.iloc[1]["archetype"])
            else:
                stable = True
            fa_profiles.append({
                "player": pff_name,
                "otc_name": row["player"],
                "team": row["team"],
                "old_apy": row["apy"],
                "archetype": latest["archetype"],
                "archetype_stable": stable,
                "cluster": latest["cluster"],
                "latest_season": int(latest["season"]),
                "latest_grade": latest["grades_defense"],
                "latest_cov_grade": latest["grades_coverage_defense"],
                "qb_rating_against": latest["qb_rating_against"],
                "yards_per_cov_snap": latest["yards_per_coverage_snap"],
                "playmaking_rate": latest["playmaking_rate"],
                "outside_rate": latest["outside_rate"],
                "slot_rate": latest["slot_rate"],
            })

    return pd.DataFrame(fa_profiles)


def estimate_fair_value(fa_df: pd.DataFrame, clustered: pd.DataFrame, cap_2026: float) -> pd.DataFrame:
    """
    Estimate fair contract value for each FA based on their archetype's
    historical cap-unit distribution.

    For each archetype, we compute the historical distribution of cap units
    among players with similar grades, then convert to 2026 dollars.

    Two estimates are produced:
    1. "Fair value" — archetype median (includes rookie deals, represents
       the efficient price based on historical production).
    2. "Market-adjusted" — uses only non-rookie contracts (>1.5% cap) within
       the archetype+grade band, reflecting what FAs actually sign for.
       This captures the positional premium, age scarcity, and bidding
       dynamics that the fair value estimate misses.
    """
    fa_df = fa_df.copy()
    clustered_valid = clustered.dropna(subset=["cap_units"])
    # Non-rookie subset: contracts above 1.5% of cap (filters out rookie deals
    # and vet minimums that would never apply to a free agent signing)
    non_rookie = clustered_valid[clustered_valid["cap_units"] > 1.5]

    estimates = []
    for _, row in fa_df.iterrows():
        archetype = row["archetype"]
        grade = row["latest_grade"]

        # --- Fair value estimate (all contracts, archetype + grade band) ---
        arch_data = clustered_valid[clustered_valid["archetype"] == archetype]
        grade_band = arch_data[
            (arch_data["grades_defense"] >= grade - 8) &
            (arch_data["grades_defense"] <= grade + 8)
        ]
        # used_grade_band = True means the valuation is a "tight" estimate
        # from similar-grade comps; False means we fell back to the full
        # archetype because the ±8 band was too thin (< 15 comps).
        used_grade_band = len(grade_band) >= 15
        ref_data = grade_band if used_grade_band else arch_data

        cap_pcts = ref_data["cap_units"]

        # --- Market-adjusted estimate (non-rookie contracts only) ---
        arch_nr = non_rookie[non_rookie["archetype"] == archetype]
        grade_band_nr = arch_nr[
            (arch_nr["grades_defense"] >= grade - 8) &
            (arch_nr["grades_defense"] <= grade + 8)
        ]
        used_grade_band_mkt = len(grade_band_nr) >= 10
        mkt_ref = grade_band_nr if used_grade_band_mkt else arch_nr

        mkt_pcts = mkt_ref["cap_units"]

        estimates.append({
            # Fair value (archetype median, includes rookies)
            "est_cap_units_p25": cap_pcts.quantile(0.25),
            "est_cap_units_median": cap_pcts.median(),
            "est_cap_units_p75": cap_pcts.quantile(0.75),
            "est_cap_units_mean": cap_pcts.mean(),
            "est_apy_low": cap_pcts.quantile(0.25) / 100 * cap_2026,
            "est_apy_mid": cap_pcts.median() / 100 * cap_2026,
            "est_apy_high": cap_pcts.quantile(0.75) / 100 * cap_2026,
            "comparable_n": len(ref_data),
            "used_grade_band": used_grade_band,
            # Market-adjusted (non-rookie contracts, what FAs actually sign)
            "mkt_cap_units_p25": mkt_pcts.quantile(0.25),
            "mkt_cap_units_median": mkt_pcts.median(),
            "mkt_cap_units_p75": mkt_pcts.quantile(0.75),
            "mkt_apy_low": mkt_pcts.quantile(0.25) / 100 * cap_2026,
            "mkt_apy_mid": mkt_pcts.median() / 100 * cap_2026,
            "mkt_apy_high": mkt_pcts.quantile(0.75) / 100 * cap_2026,
            "mkt_comparable_n": len(mkt_ref),
            "used_grade_band_mkt": used_grade_band_mkt,
            # Premium gap
            "market_premium_pct": (mkt_pcts.median() - cap_pcts.median()),
        })

    estimates_df = pd.DataFrame(estimates)
    return pd.concat([fa_df.reset_index(drop=True), estimates_df], axis=1)


# ---------------------------------------------------------------------------
# Chart functions
# ---------------------------------------------------------------------------

def fig24_fa_archetype_breakdown(fa_valued):
    """Pie/bar chart of 2026 FA class by archetype."""
    counts = fa_valued["archetype"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    colors = [ARCHETYPE_COLOR_MAP.get(a, COLORS["neutral"]) for a in counts.index]
    axes[0].barh(range(len(counts)), counts.values, color=colors, alpha=0.8, edgecolor="white")
    axes[0].set_yticks(range(len(counts)))
    axes[0].set_yticklabels(counts.index, fontsize=11)
    axes[0].set_xlabel("Number of free agents")
    axes[0].set_title("2026 FA class by archetype")
    axes[0].invert_yaxis()
    for i, v in enumerate(counts.values):
        axes[0].text(v + 0.3, i, str(v), va="center", fontsize=11, fontweight="bold")

    # Mean grade by archetype for FAs
    mean_grades = fa_valued.groupby("archetype")["latest_grade"].mean().reindex(counts.index)
    colors2 = [ARCHETYPE_COLOR_MAP.get(a, COLORS["neutral"]) for a in mean_grades.index]
    axes[1].barh(range(len(mean_grades)), mean_grades.values, color=colors2, alpha=0.8, edgecolor="white")
    axes[1].set_yticks(range(len(mean_grades)))
    axes[1].set_yticklabels(mean_grades.index, fontsize=11)
    axes[1].set_xlabel("Mean PFF grade (most recent season)")
    axes[1].set_title("2026 FA class quality by archetype")
    axes[1].invert_yaxis()
    for i, v in enumerate(mean_grades.values):
        axes[1].text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=11, fontweight="bold")

    fig.suptitle("2026 CB free agent class composition",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def fig25_fa_value_estimates(fa_valued, cap_2026):
    """Top FA value estimates with confidence ranges."""
    # Show top 20 FAs by estimated mid APY
    top = fa_valued.nlargest(20, "est_apy_mid").copy()
    top = top.sort_values("est_apy_mid", ascending=True)  # for horizontal bar

    fig, ax = plt.subplots(figsize=(12, 9))

    colors = [ARCHETYPE_COLOR_MAP.get(a, COLORS["neutral"]) for a in top["archetype"]]

    # Error bars showing 25th-75th percentile range
    y_pos = range(len(top))
    ax.barh(y_pos, top["est_apy_mid"] / 1e6, color=colors, alpha=0.8, edgecolor="white", height=0.6)

    # Add range lines
    for i, (_, row) in enumerate(top.iterrows()):
        ax.plot([row["est_apy_low"] / 1e6, row["est_apy_high"] / 1e6], [i, i],
                color="black", linewidth=1.5, alpha=0.6)
        ax.plot([row["est_apy_low"] / 1e6], [i], "|", color="black", markersize=8, alpha=0.6)
        ax.plot([row["est_apy_high"] / 1e6], [i], "|", color="black", markersize=8, alpha=0.6)

    labels = [f"{row['player']} ({row['archetype'][:15]})" for _, row in top.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Estimated APY ($M, 2026 dollars)")
    ax.set_title("2026 CB free agent value estimates — top 20 by projected APY",
                 fontweight="bold")

    # Add value labels
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(row["est_apy_mid"] / 1e6 + 0.2, i,
                f"${row['est_apy_mid'] / 1e6:.1f}M",
                va="center", fontsize=9)

    fig.tight_layout()
    return fig


def fig26_key_targets(fa_valued, cap_2026):
    """
    Highlight the best value targets: FAs whose grade is above their
    archetype's average but whose old APY suggests they're undervalued.
    """
    fa = fa_valued.copy()
    fa["grade_vs_archetype"] = fa.apply(
        lambda r: r["latest_grade"] - fa[fa["archetype"] == r["archetype"]]["latest_grade"].mean(),
        axis=1
    )

    # Scatter: grade vs estimated APY
    fig, ax = plt.subplots(figsize=(12, 7))

    for archetype in sorted(fa["archetype"].unique()):
        arch_mask = fa["archetype"] == archetype
        color = ARCHETYPE_COLOR_MAP.get(archetype, COLORS["neutral"])
        # Stable archetype: filled dot. Unstable: hollow ring — visual warning
        # that the comp set for this player's valuation may not fit cleanly.
        stable = arch_mask & fa["archetype_stable"]
        unstable = arch_mask & ~fa["archetype_stable"]
        ax.scatter(fa.loc[stable, "latest_grade"], fa.loc[stable, "est_apy_mid"] / 1e6,
                   c=color, label=archetype, alpha=0.6, s=40, edgecolors="none")
        ax.scatter(fa.loc[unstable, "latest_grade"], fa.loc[unstable, "est_apy_mid"] / 1e6,
                   facecolors="none", edgecolors=color, linewidths=1.5, alpha=0.85, s=55)

    # Label notable FAs; append an asterisk when the archetype is unstable
    notable = fa.nlargest(12, "latest_grade")
    for _, row in notable.iterrows():
        label = row["player"] + ("*" if not row["archetype_stable"] else "")
        ax.annotate(label, (row["latest_grade"], row["est_apy_mid"] / 1e6),
                    fontsize=8, textcoords="offset points", xytext=(5, 5), alpha=0.8)

    # Instability marker in the legend (one shared entry across archetypes)
    ax.scatter([], [], facecolors="none", edgecolors=COLORS["neutral"], linewidths=1.5,
               s=55, label="Unstable archetype (last 2 seasons differ)")

    ax.set_xlabel("PFF grade (most recent season)")
    ax.set_ylabel("Estimated APY ($M, 2026)")
    ax.set_title("2026 FA class — grade vs. projected contract value",
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for path, name in [(INPUT_CLUSTERED, "clustered"), (INPUT_CONTRACTS, "contracts"),
                        (INPUT_CAP, "cap history")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    print("=" * 60)
    print("Loading data")
    print("=" * 60)
    clustered = pd.read_csv(INPUT_CLUSTERED)
    contracts_raw = pd.read_csv(INPUT_CONTRACTS, header=None,
                                 names=["player", "team", "signing_year", "years", "total_value", "apy"])
    contracts_raw["total_value"] = contracts_raw["total_value"].str.replace("$", "").str.replace(",", "").astype(float)
    contracts_raw["apy"] = contracts_raw["apy"].str.replace("$", "").str.replace(",", "").astype(float)
    contracts_raw = contracts_raw.dropna(subset=["years"])
    contracts_raw["end_year"] = (contracts_raw["signing_year"] + contracts_raw["years"] - 1).astype(int)

    cap_history = pd.read_csv(INPUT_CAP, header=None, names=["season", "salary_cap"])
    cap_history["salary_cap"] = cap_history["salary_cap"].str.replace(",", "").astype(int)

    print(f"  Clustered: {len(clustered)} player-seasons")
    print(f"  Contracts: {len(contracts_raw)} rows")

    # ----- Project 2026 cap -----
    print("\n" + "=" * 60)
    print("Projecting 2026 salary cap")
    print("=" * 60)
    # OTC projection for the 2026 salary cap
    cap_2026 = 301_200_000

    # ----- Identify 2026 FA class -----
    print("\n" + "=" * 60)
    print("Identifying 2026 FA class")
    print("=" * 60)
    fa_class = identify_fa_class(contracts_raw)
    print(f"  Total 2026 CB free agents: {len(fa_class)}")

    # ----- Assign archetypes -----
    print("\n" + "=" * 60)
    print("Assigning archetypes to FAs")
    print("=" * 60)
    fa_profiled = assign_archetypes(fa_class, clustered)
    print(f"  FAs matched to our data: {len(fa_profiled)}")
    print(f"  By archetype:")
    print(fa_profiled["archetype"].value_counts().to_string())

    # ----- Estimate fair value -----
    print("\n" + "=" * 60)
    print("Estimating fair contract values")
    print("=" * 60)
    fa_valued = estimate_fair_value(fa_profiled, clustered, cap_2026)
    print(f"  Valuations computed for {len(fa_valued)} FAs")

    # ----- Generate charts -----
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Generating charts")
    print("=" * 60)

    charts = [
        ("24_fa_archetype_breakdown.png", fig24_fa_archetype_breakdown, (fa_valued,)),
        ("25_fa_value_estimates.png", fig25_fa_value_estimates, (fa_valued, cap_2026)),
        ("26_fa_key_targets.png", fig26_key_targets, (fa_valued, cap_2026)),
    ]

    for filename, func, args in charts:
        fig = func(*args)
        path = FIG_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {path.name}")

    # ----- Save valuation table -----
    output_cols = [
        "player", "team", "archetype", "archetype_stable",
        "latest_season", "latest_grade",
        "latest_cov_grade", "qb_rating_against", "yards_per_cov_snap",
        "playmaking_rate", "outside_rate", "slot_rate",
        "old_apy",
        "est_cap_units_median", "est_apy_low", "est_apy_mid", "est_apy_high",
        "comparable_n", "used_grade_band",
        "mkt_cap_units_median", "mkt_apy_low", "mkt_apy_mid", "mkt_apy_high",
        "mkt_comparable_n", "used_grade_band_mkt", "market_premium_pct",
    ]
    fa_output = fa_valued[output_cols].sort_values("mkt_apy_mid", ascending=False)
    fa_output.to_csv(TABLE_DIR / "fa_2026_valuations.csv", index=False)
    print(f"  Saved: fa_2026_valuations.csv")

    # ----- Print key findings -----
    print("\n" + "=" * 60)
    print("Q3 Answer: 2026 FA class valuations")
    print("=" * 60)

    print(f"\n  Projected 2026 salary cap: ${cap_2026:,.0f}")
    print(f"  FAs valued: {len(fa_valued)}")

    print(f"\n  Top 10 projected contracts (market-adjusted):")
    for _, row in fa_valued.nlargest(10, "mkt_apy_mid").iterrows():
        print(f"    {row['player']:25s} | {row['archetype']:25s} | "
              f"Grade: {row['latest_grade']:.1f} | "
              f"Market APY: ${row['mkt_apy_mid'] / 1e6:.1f}M "
              f"(${row['mkt_apy_low'] / 1e6:.1f}M - ${row['mkt_apy_high'] / 1e6:.1f}M) | "
              f"Fair: ${row['est_apy_mid'] / 1e6:.1f}M | "
              f"Premium: +{row['market_premium_pct']:.1f}%")

    print(f"\n  Best value targets (high grade, below-archetype-median cost):")
    fa_valued_sorted = fa_valued.sort_values("latest_grade", ascending=False)
    value_targets = fa_valued_sorted[
        fa_valued_sorted["latest_grade"] > fa_valued_sorted["archetype"].map(
            fa_valued_sorted.groupby("archetype")["latest_grade"].mean()
        )
    ].head(10)
    for _, row in value_targets.iterrows():
        print(f"    {row['player']:25s} | {row['archetype']:25s} | "
              f"Grade: {row['latest_grade']:.1f} | "
              f"Market: ${row['mkt_apy_mid'] / 1e6:.1f}M | Fair: ${row['est_apy_mid'] / 1e6:.1f}M")

    print(f"\n  Avoid list (high old APY, low grade):")
    overpaid = fa_valued[
        (fa_valued["old_apy"] > 3_000_000) &
        (fa_valued["latest_grade"] < 58)
    ].sort_values("old_apy", ascending=False)
    for _, row in overpaid.iterrows():
        print(f"    {row['player']:25s} | Grade: {row['latest_grade']:.1f} | "
              f"Old APY: ${row['old_apy'] / 1e6:.1f}M | "
              f"Fair value: ${row['est_apy_mid'] / 1e6:.1f}M")

    print(f"\n  All charts saved to: {FIG_DIR}/")
    print(f"  Valuation table saved to: {TABLE_DIR}/fa_2026_valuations.csv")


if __name__ == "__main__":
    main()