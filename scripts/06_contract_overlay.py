"""
06_contract_overlay.py

Answers Q2b: How does the market price each CB archetype?

Merges Over The Cap contract data onto the clustered CB dataset, calculates
cap units (APY as % of salary cap), and analyzes contract value by archetype.

Inputs:
    {PROJECT_ROOT}/data/processed/cb_clustered.csv
    {PROJECT_ROOT}/data/raw/salary_cap_history.csv     (from OTC — Sheet1)
    {PROJECT_ROOT}/data/raw/cb_contracts_otc.csv        (from OTC — Sheet2)

Outputs:
    {PROJECT_ROOT}/data/processed/cb_with_contracts.csv
    {PROJECT_ROOT}/outputs/figures/20_cap_units_by_archetype.png
    {PROJECT_ROOT}/outputs/figures/21_archetype_value_scatter.png
    {PROJECT_ROOT}/outputs/figures/22_overpaid_underpaid.png
    {PROJECT_ROOT}/outputs/figures/23_contract_tier_heatmap.png
    {PROJECT_ROOT}/outputs/tables/archetype_contract_summary.csv

Design notes:
- Name matching: 86% exact match between PFF and OTC names. Remaining 14%
  handled via manual mapping for suffix/abbreviation differences (Jr., III,
  C.J. vs CJ, nicknames like Sauce → Ahmad). ~7 players are cross-position
  (safeties PFF tagged as CB) and won't have CB contracts — these get NaN.
- Contract-to-season: for each player-season, we find the contract active
  during that season (signing_year <= season <= end_year). When multiple
  contracts overlap, we take the most recently signed one.
- Cap units: APY / salary_cap for that season. This normalizes across years
  so a $10M/yr deal in 2018 (5.6% of cap) is comparable to a $16M/yr deal
  in 2025 (5.7% of cap).

Usage:
    cd ~/Documents/sports_project_cb
    python3 scripts/06_contract_overlay.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("/Users/anokhpalakurthi/Documents/sports_project_cb")

INPUT_CLUSTERED = PROJECT_ROOT / "data" / "processed" / "cb_clustered.csv"
INPUT_CAP = PROJECT_ROOT / "data" / "raw" / "salary_cap_history.csv"
INPUT_CONTRACTS = PROJECT_ROOT / "data" / "raw" / "cb_contracts_otc.csv"

OUTPUT_MERGED = PROJECT_ROOT / "data" / "processed" / "cb_with_contracts.csv"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

# Manual name mapping: PFF name → OTC name
# Built by comparing the 48 PFF names that don't exact-match OTC and finding
# the correct OTC equivalent via fuzzy matching + manual verification.
NAME_MAP = {
    # Suffix differences
    "Antonio Hamilton Sr.": "Antonio Hamilton",
    "Beanie Bishop Jr.": "Beanie Bishop",
    "Byron Murphy Jr.": "Byron Murphy",
    "Carlton Davis III": "Carlton Davis",
    "Casey Hayward Jr.": "Casey Hayward",
    "David Long Jr.": "David Long",
    "Desmond King II": "Desmond King",
    "Greg Stroman Jr.": "Greg Stroman",
    "Jarvis Brownlee Jr.": "Jarvis Brownlee",
    "Keith Taylor Jr.": "Keith Taylor",
    "Kenny Moore II": "Kenny Moore",
    "Martin Emerson Jr.": "Martin Emerson",
    "Pat Surtain II": "Patrick Surtain II",
    "Sidney Jones IV": "Sidney Jones",
    "Tramaine Brock Sr.": "Tramaine Brock",
    "Troy Pride Jr.": "Troy Pride, Jr.",
    "Vernon Hargreaves III": "Vernon Hargreaves",
    "DJ Turner II": "D.J. Turner",
    "Tre Hawkins III": "Tre Hawkins",
    "Billy Bowman Jr.": "Billy Bowman",
    "Samuel Womack III": "Sam Womack",
    # Abbreviation / spelling differences
    "CJ Henderson": "C.J. Henderson",
    "Cor'Dale Flott": "Cordale Flott",
    "Dax Hill": "Daxton Hill",
    "DeAndre Baker": "Deandre Baker",
    "Dont'e Deayon": "Donte Deayon",
    "JuJu Brents": "Julius Brents",
    "Coby Bryant": "Cobee Bryant",
    # Nickname / full name
    "Sauce Gardner": "Ahmad Gardner",
    "C.J. Gardner-Johnson": "Chauncey Gardner-Johnson, Jr.",
    "Ugo Amadi": "Ugochukwu Amadi",
}

# Players who won't match because they're cross-position (safeties/LBs that
# PFF sometimes lists as CB) or genuinely missing from OTC's CB data.
# These will have NaN contract data — that's expected and documented.
EXPECTED_NO_MATCH = {
    "Budda Baker", "Minkah Fitzpatrick", "Brian Branch", "Isaiah Simmons",
    "Chamarri Conner", "Damontae Kazee", "Christian Izien", "Eric Murray",
    "Lamarcus Joyner", "Mike Jackson", "Nazeeh Johnson", "Quentin Lake",
    "Rafael Bush", "Tykee Smith", "Upton Stout", "Will Harris", "Will Parks",
    "Samuel Womack III",
}

CLUSTER_COLORS = ["#534AB7", "#1D9E75", "#D85A30", "#378ADD", "#BA7517"]

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
# Data loading and merging
# ---------------------------------------------------------------------------

def load_and_clean_contracts(path: Path) -> pd.DataFrame:
    """Load OTC contract data and clean formatting."""
    df = pd.read_csv(path, header=None,
                     names=["player", "team", "signing_year", "years", "total_value", "apy"])
    df["total_value"] = df["total_value"].str.replace("$", "").str.replace(",", "").astype(float)
    df["apy"] = df["apy"].str.replace("$", "").str.replace(",", "").astype(float)
    # Drop rows with missing years (rare data entry issues in OTC)
    df = df.dropna(subset=["years"])
    df["end_year"] = (df["signing_year"] + df["years"] - 1).astype(int)
    return df


def load_cap_history(path: Path) -> pd.DataFrame:
    """Load salary cap history and clean formatting."""
    df = pd.read_csv(path, header=None, names=["season", "salary_cap"])
    df["salary_cap"] = df["salary_cap"].str.replace(",", "").astype(int)
    return df


def match_contracts_to_seasons(clustered: pd.DataFrame, contracts: pd.DataFrame) -> pd.DataFrame:
    """
    For each player-season in the clustered data, find the active contract.

    A contract is active during a season if: signing_year <= season <= end_year.
    When multiple contracts overlap, take the most recently signed one
    (the player's current deal at that point).
    """
    clustered = clustered.copy().reset_index(drop=True)
    clustered["otc_name"] = clustered["player"].map(lambda x: NAME_MAP.get(x, x))
    clustered.index.name = "_row_id"

    # Inner-merge every player-season against every contract on the player's
    # OTC name, filter to contracts whose window spans the season, then pick
    # the most recently signed contract per player-season.
    merged = clustered[["otc_name", "season"]].reset_index().merge(
        contracts[["player", "signing_year", "end_year", "apy", "years", "total_value"]],
        left_on="otc_name", right_on="player", how="inner",
    )
    active = merged[
        (merged["signing_year"] <= merged["season"]) &
        (merged["end_year"] >= merged["season"])
    ]
    best_idx = active.groupby("_row_id")["signing_year"].idxmax()
    best = active.loc[best_idx].set_index("_row_id")

    # Index-aligned assignment: rows without a matching contract stay NaN.
    clustered["apy"] = best["apy"]
    clustered["contract_years"] = best["years"]
    clustered["total_value"] = best["total_value"]
    clustered.index.name = None

    matched_mask = clustered["apy"].notna()
    matched_count = int(matched_mask.sum())
    unmatched_players = set(
        clustered.loc[~matched_mask & ~clustered["player"].isin(EXPECTED_NO_MATCH), "player"]
    )

    print(f"  Matched: {matched_count}/{len(clustered)} player-seasons "
          f"({matched_count / len(clustered) * 100:.1f}%)")
    if unmatched_players:
        print(f"  Unexpected unmatched ({len(unmatched_players)}): {sorted(unmatched_players)}")

    return clustered


def add_cap_units(df: pd.DataFrame, cap_history: pd.DataFrame) -> pd.DataFrame:
    """Calculate APY as percentage of salary cap (cap units)."""
    df = df.merge(cap_history, on="season", how="left")
    df["cap_units"] = df["apy"] / df["salary_cap"] * 100  # as percentage
    return df


# ---------------------------------------------------------------------------
# Chart functions
# ---------------------------------------------------------------------------

def fig20_cap_units_by_archetype(df):
    """Box plot of cap units by archetype — the core value question."""
    df_valid = df.dropna(subset=["cap_units"])

    # Order archetypes by median cap units
    archetype_order = (df_valid.groupby("archetype")["cap_units"]
                       .median().sort_values(ascending=False).index.tolist())

    fig, ax = plt.subplots(figsize=(12, 6))

    data = [df_valid[df_valid["archetype"] == a]["cap_units"].values for a in archetype_order]
    bp = ax.boxplot(data, labels=archetype_order, patch_artist=True, widths=0.6,
                    medianprops=dict(color="white", linewidth=2))

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(CLUSTER_COLORS[i % len(CLUSTER_COLORS)])
        patch.set_alpha(0.8)

    ax.set_ylabel("Cap units (APY as % of salary cap)")
    ax.set_xlabel("CB archetype")
    ax.set_title("Contract value by CB archetype — who gets paid?", fontweight="bold")

    # Add median labels
    for i, med in enumerate(df_valid.groupby("archetype")["cap_units"].median().reindex(archetype_order)):
        ax.text(i + 1, med + 0.15, f"{med:.2f}%", ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    return fig


def fig21_value_scatter(df):
    """Scatter: PFF grade vs cap units, colored by archetype."""
    df_valid = df.dropna(subset=["cap_units"])

    fig, ax = plt.subplots(figsize=(12, 7))

    archetypes = sorted(df_valid["archetype"].unique())
    for i, archetype in enumerate(archetypes):
        mask = df_valid["archetype"] == archetype
        ax.scatter(df_valid.loc[mask, "grades_defense"],
                   df_valid.loc[mask, "cap_units"],
                   c=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                   label=archetype, alpha=0.5, s=25, edgecolors="none")

    ax.set_xlabel("PFF overall defense grade")
    ax.set_ylabel("Cap units (APY as % of salary cap)")
    ax.set_title("Grade vs. contract value by archetype — finding the inefficiencies",
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")

    # Add quadrant labels
    grade_med = df_valid["grades_defense"].median()
    cap_med = df_valid["cap_units"].median()
    ax.axhline(cap_med, color=COLORS["neutral"], linestyle="--", alpha=0.4)
    ax.axvline(grade_med, color=COLORS["neutral"], linestyle="--", alpha=0.4)
    ax.text(85, cap_med + 0.5, "Overpaid", fontsize=9, color=COLORS["neutral"], ha="right")
    ax.text(85, cap_med - 0.5, "Value", fontsize=9, color=COLORS["neutral"], ha="right")
    ax.text(40, cap_med + 0.5, "Bad contracts", fontsize=9, color=COLORS["neutral"])
    ax.text(40, cap_med - 0.5, "Cheap + bad", fontsize=9, color=COLORS["neutral"])

    fig.tight_layout()
    return fig


def fig22_overpaid_underpaid(df):
    """Bar chart: mean cap units vs mean grade per archetype — efficiency view."""
    df_valid = df.dropna(subset=["cap_units"])
    # FA-market subset: strip out rookie deals and vet minimums so the
    # efficiency ratio reflects prices an actual free agent would command.
    df_fa_market = df_valid[df_valid["cap_units"] > 1.5]

    summary = df_valid.groupby("archetype").agg(
        mean_grade=("grades_defense", "mean"),
        mean_cap_units=("cap_units", "mean"),
        count=("player", "size"),
    ).sort_values("mean_grade", ascending=False)

    summary["efficiency_all"] = summary["mean_grade"] / summary["mean_cap_units"]

    fa_stats = df_fa_market.groupby("archetype").agg(
        fa_mean_grade=("grades_defense", "mean"),
        fa_mean_cap_units=("cap_units", "mean"),
    )
    summary = summary.join(fa_stats)
    summary["efficiency_fa_market"] = summary["fa_mean_grade"] / summary["fa_mean_cap_units"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Left: mean grade by archetype
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(summary))]
    axes[0].barh(range(len(summary)), summary["mean_grade"], color=colors, alpha=0.8,
                 edgecolor="white")
    axes[0].set_yticks(range(len(summary)))
    axes[0].set_yticklabels(summary.index, fontsize=10)
    axes[0].set_xlabel("Mean PFF grade")
    axes[0].set_title("Performance by archetype")
    axes[0].invert_yaxis()
    for i, v in enumerate(summary["mean_grade"]):
        axes[0].text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=10)

    # Middle: mean cap units by archetype
    axes[1].barh(range(len(summary)), summary["mean_cap_units"], color=colors, alpha=0.8,
                 edgecolor="white")
    axes[1].set_yticks(range(len(summary)))
    axes[1].set_yticklabels(summary.index, fontsize=10)
    axes[1].set_xlabel("Mean cap units (%)")
    axes[1].set_title("Cost by archetype")
    axes[1].invert_yaxis()
    for i, v in enumerate(summary["mean_cap_units"]):
        axes[1].text(v + 0.05, i, f"{v:.2f}%", va="center", fontsize=10)

    # Right: efficiency comparison (all contracts vs FA-market contracts only)
    y = np.arange(len(summary))
    h = 0.4
    axes[2].barh(y - h / 2, summary["efficiency_all"], height=h, color=colors,
                 alpha=0.9, edgecolor="white", label="efficiency_all")
    axes[2].barh(y + h / 2, summary["efficiency_fa_market"], height=h, color=colors,
                 alpha=0.45, edgecolor="white", label="efficiency_fa_market (cap units > 1.5%)")
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(summary.index, fontsize=10)
    axes[2].set_xlabel("Grade points per cap unit")
    axes[2].set_title("Efficiency: all contracts vs FA market")
    axes[2].invert_yaxis()
    axes[2].legend(fontsize=9, loc="lower right")
    for i, (e_all, e_fa) in enumerate(zip(summary["efficiency_all"], summary["efficiency_fa_market"])):
        axes[2].text(e_all + 0.5, i - h / 2, f"{e_all:.1f}", va="center", fontsize=9)
        if pd.notna(e_fa):
            axes[2].text(e_fa + 0.5, i + h / 2, f"{e_fa:.1f}", va="center", fontsize=9)

    fig.suptitle("Performance vs. cost — which archetypes are efficient?",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def fig23_contract_tier_heatmap(df):
    """Heatmap: archetype vs contract tier (rookie, mid, premium, elite)."""
    df_valid = df.dropna(subset=["cap_units"]).copy()

    # Define contract tiers
    df_valid["contract_tier"] = pd.cut(
        df_valid["cap_units"],
        bins=[0, 1, 3, 6, 100],
        labels=["Rookie/min (<1%)", "Mid-range (1-3%)", "Premium (3-6%)", "Elite (6%+)"]
    )

    # Cross-tab
    ct = pd.crosstab(df_valid["archetype"], df_valid["contract_tier"], normalize="index") * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(ct, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "% of archetype"}, linewidths=0.5)
    ax.set_title("Contract tier distribution by archetype (% of player-seasons)",
                 fontweight="bold", pad=15)
    ax.set_ylabel("CB archetype")
    ax.set_xlabel("Contract tier")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for path, name in [(INPUT_CLUSTERED, "clustered"), (INPUT_CAP, "cap history"),
                        (INPUT_CONTRACTS, "contracts")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    print("=" * 60)
    print("Loading data")
    print("=" * 60)
    clustered = pd.read_csv(INPUT_CLUSTERED)
    contracts = load_and_clean_contracts(INPUT_CONTRACTS)
    cap_history = load_cap_history(INPUT_CAP)
    print(f"  Clustered: {len(clustered)} player-seasons")
    print(f"  Contracts: {len(contracts)} rows")
    print(f"  Cap history: {len(cap_history)} seasons")

    # ----- Match contracts to player-seasons -----
    print("\n" + "=" * 60)
    print("Matching contracts to player-seasons")
    print("=" * 60)
    df = match_contracts_to_seasons(clustered, contracts)

    # ----- Add cap units -----
    print("\n" + "=" * 60)
    print("Calculating cap units")
    print("=" * 60)
    df = add_cap_units(df, cap_history)
    valid = df["cap_units"].notna()
    print(f"  {valid.sum()} player-seasons with contract data ({valid.sum() / len(df) * 100:.1f}%)")
    print(f"  Cap units range: {df['cap_units'].min():.3f}% - {df['cap_units'].max():.3f}%")
    print(f"  Cap units mean: {df['cap_units'].mean():.3f}%, median: {df['cap_units'].median():.3f}%")

    # ----- Generate charts -----
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Generating charts")
    print("=" * 60)

    charts = [
        ("20_cap_units_by_archetype.png", fig20_cap_units_by_archetype, (df,)),
        ("21_archetype_value_scatter.png", fig21_value_scatter, (df,)),
        ("22_overpaid_underpaid.png", fig22_overpaid_underpaid, (df,)),
        ("23_contract_tier_heatmap.png", fig23_contract_tier_heatmap, (df,)),
    ]

    for filename, func, args in charts:
        fig = func(*args)
        path = FIG_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {path.name}")

    # ----- Save outputs -----
    df.to_csv(OUTPUT_MERGED, index=False)
    print(f"  Saved: {OUTPUT_MERGED.name}")

    # Archetype contract summary table
    df_valid = df.dropna(subset=["cap_units"])
    # FA-market subset excludes rookie deals / vet mins so efficiency reflects
    # prices a free agent would actually see.
    df_fa_market = df_valid[df_valid["cap_units"] > 1.5]

    summary = df_valid.groupby("archetype").agg(
        count=("player", "size"),
        mean_grade=("grades_defense", "mean"),
        mean_apy=("apy", "mean"),
        median_apy=("apy", "median"),
        mean_cap_units=("cap_units", "mean"),
        median_cap_units=("cap_units", "median"),
        min_cap_units=("cap_units", "min"),
        max_cap_units=("cap_units", "max"),
    )
    summary["efficiency_all"] = summary["mean_grade"] / summary["mean_cap_units"]

    fa_stats = df_fa_market.groupby("archetype").agg(
        fa_market_count=("player", "size"),
        fa_market_mean_grade=("grades_defense", "mean"),
        fa_market_mean_cap_units=("cap_units", "mean"),
    )
    summary = summary.join(fa_stats)
    summary["efficiency_fa_market"] = (
        summary["fa_market_mean_grade"] / summary["fa_market_mean_cap_units"]
    )
    summary = summary.round(3)
    summary.to_csv(TABLE_DIR / "archetype_contract_summary.csv")
    print(f"  Saved: archetype_contract_summary.csv")

    # ----- Print summary -----
    print("\n" + "=" * 60)
    print("Q2b Answer: How does the market price each archetype?")
    print("=" * 60)

    for archetype in summary.sort_values("mean_cap_units", ascending=False).index:
        row = summary.loc[archetype]
        print(f"\n  {archetype}:")
        print(f"    Mean grade: {row['mean_grade']:.1f} | "
              f"Mean APY: ${row['mean_apy']:,.0f} | "
              f"Mean cap units: {row['mean_cap_units']:.2f}%")
        print(f"    Median APY: ${row['median_apy']:,.0f} | "
              f"Cap unit range: {row['min_cap_units']:.2f}% - {row['max_cap_units']:.2f}%")

    # Efficiency ranking (all contracts vs FA market only)
    print(f"\n  Efficiency ranking (grade per cap unit):")
    print(f"    {'Archetype':<30} {'efficiency_all':>16} {'efficiency_fa_market':>22}")
    for archetype in summary.sort_values("efficiency_fa_market", ascending=False).index:
        row = summary.loc[archetype]
        fa_str = f"{row['efficiency_fa_market']:.1f}" if pd.notna(row["efficiency_fa_market"]) else "n/a"
        print(f"    {archetype:<30} {row['efficiency_all']:>16.1f} {fa_str:>22}")

    print(f"\n  All charts saved to: {FIG_DIR}/")


if __name__ == "__main__":
    main()
