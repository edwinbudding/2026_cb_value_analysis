"""
09_fa_2027_extension.py

Extends the FA valuation framework to the 2027 free agent class.
Uses the same archetype-based valuation methodology as script 07,
with OTC's projected 2027 salary cap of $327M.

Input:
    {PROJECT_ROOT}/data/processed/cb_with_contracts.csv
    {PROJECT_ROOT}/data/raw/fa_2027_otc.csv

Outputs:
    {PROJECT_ROOT}/outputs/figures/32_fa_2027_archetype_breakdown.png
    {PROJECT_ROOT}/outputs/figures/33_fa_2027_value_estimates.png
    {PROJECT_ROOT}/outputs/figures/34_fa_2027_key_targets.png
    {PROJECT_ROOT}/outputs/figures/35_fa_2026_vs_2027_comparison.png
    {PROJECT_ROOT}/outputs/tables/fa_2027_valuations.csv

Usage:
    cd ~/Documents/sports_project_cb
    python3 scripts/09_fa_2027_extension.py
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
INPUT_FA_2027 = PROJECT_ROOT / "data" / "raw" / "fa_2027_otc.csv"
INPUT_FA_2026_VALS = PROJECT_ROOT / "outputs" / "tables" / "fa_2026_valuations.csv"

FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

# OTC projected cap
CAP_2027 = 327_000_000

# PFF ↔ OTC name mapping (OTC name → PFF name for the 2027 list)
OTC_TO_PFF = {
    "Michael Jackson Sr.": "Mike Jackson",
    "Daxton Hill": "Dax Hill",
    "Kenny Moore": "Kenny Moore II",
    "Julius Brents": "JuJu Brents",
    "D.J. Turner": "DJ Turner II",
    "Cobee Bryant": "Coby Bryant",
    "Cordale Flott": "Cor'Dale Flott",
    "C.J. Henderson": "CJ Henderson",
    "Tre Hawkins": "Tre Hawkins III",
    "Chauncey Gardner-Johnson, Jr.": "C.J. Gardner-Johnson",
    "Ahmad Gardner": "Sauce Gardner",
    "Ugochukwu Amadi": "Ugo Amadi",
    "Patrick Surtain II": "Pat Surtain II",
    "Billy Bowman": "Billy Bowman Jr.",
}

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

def assign_archetypes(fa: pd.DataFrame, clustered: pd.DataFrame) -> pd.DataFrame:
    """Assign each 2027 FA to an archetype based on most recent season.

    Note: this is a lookup-based assignment — we read the archetype label
    already attached to the player's most recent row in cb_clustered.csv,
    rather than re-scoring their features through the scaler/PCA/k-means
    pipeline from script 05. A 2027 FA who has no row in the 2018–2025
    clustered dataset (e.g., a player who wasn't a starter in any of
    those seasons, or one who only logged PFF snaps outside the coverage
    threshold) is silently skipped here and will not appear in the valued
    output. To score such a player, load scaler.joblib and pca.joblib
    from outputs/tables/ and project their features onto the k-means
    centroids.
    """
    fa = fa.copy()
    fa["pff_name"] = fa["player"].map(lambda x: OTC_TO_PFF.get(x, x))

    profiles = []
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
            profiles.append({
                "player": pff_name,
                "otc_name": row["player"],
                "team": row["team"],
                "fa_type": row["fa_type"],
                "age": row["age"],
                "current_apy": row["current_apy"],
                "archetype": latest["archetype"],
                "archetype_stable": stable,
                "latest_season": int(latest["season"]),
                "latest_grade": latest["grades_defense"],
                "latest_cov_grade": latest["grades_coverage_defense"],
                "qb_rating_against": latest["qb_rating_against"],
                "yards_per_cov_snap": latest["yards_per_coverage_snap"],
                "playmaking_rate": latest["playmaking_rate"],
                "outside_rate": latest["outside_rate"],
                "slot_rate": latest["slot_rate"],
            })

    return pd.DataFrame(profiles)


def estimate_fair_value(fa_df: pd.DataFrame, clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate fair contract value using archetype + grade band historical comps.

    Two estimates:
    1. "Fair value" — archetype median (includes rookie deals, the efficient price).
    2. "Market-adjusted" — non-rookie contracts only (>1.5% cap), reflecting
       what free agents actually sign for including positional premium and scarcity.
    """
    fa_df = fa_df.copy()
    clustered_valid = clustered.dropna(subset=["cap_units"])
    non_rookie = clustered_valid[clustered_valid["cap_units"] > 1.5]

    estimates = []
    for _, row in fa_df.iterrows():
        archetype = row["archetype"]
        grade = row["latest_grade"]

        # --- Fair value estimate (all contracts) ---
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
            # Fair value
            "est_cap_units_p25": cap_pcts.quantile(0.25),
            "est_cap_units_median": cap_pcts.median(),
            "est_cap_units_p75": cap_pcts.quantile(0.75),
            "est_cap_units_mean": cap_pcts.mean(),
            "est_apy_low": cap_pcts.quantile(0.25) / 100 * CAP_2027,
            "est_apy_mid": cap_pcts.median() / 100 * CAP_2027,
            "est_apy_high": cap_pcts.quantile(0.75) / 100 * CAP_2027,
            "comparable_n": len(ref_data),
            "used_grade_band": used_grade_band,
            # Market-adjusted
            "mkt_cap_units_p25": mkt_pcts.quantile(0.25),
            "mkt_cap_units_median": mkt_pcts.median(),
            "mkt_cap_units_p75": mkt_pcts.quantile(0.75),
            "mkt_apy_low": mkt_pcts.quantile(0.25) / 100 * CAP_2027,
            "mkt_apy_mid": mkt_pcts.median() / 100 * CAP_2027,
            "mkt_apy_high": mkt_pcts.quantile(0.75) / 100 * CAP_2027,
            "mkt_comparable_n": len(mkt_ref),
            "used_grade_band_mkt": used_grade_band_mkt,
            # Premium gap
            "market_premium_pct": (mkt_pcts.median() - cap_pcts.median()),
        })

    return pd.concat([fa_df.reset_index(drop=True), pd.DataFrame(estimates)], axis=1)


# ---------------------------------------------------------------------------
# Chart functions
# ---------------------------------------------------------------------------

def fig32_archetype_breakdown(fa_valued):
    """2027 FA class by archetype — count and quality."""
    counts = fa_valued["archetype"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = [ARCHETYPE_COLOR_MAP.get(a, COLORS["neutral"]) for a in counts.index]
    axes[0].barh(range(len(counts)), counts.values, color=colors, alpha=0.8, edgecolor="white")
    axes[0].set_yticks(range(len(counts)))
    axes[0].set_yticklabels(counts.index, fontsize=11)
    axes[0].set_xlabel("Number of free agents")
    axes[0].set_title("2027 FA class by archetype")
    axes[0].invert_yaxis()
    for i, v in enumerate(counts.values):
        axes[0].text(v + 0.3, i, str(v), va="center", fontsize=11, fontweight="bold")

    mean_grades = fa_valued.groupby("archetype")["latest_grade"].mean().reindex(counts.index)
    colors2 = [ARCHETYPE_COLOR_MAP.get(a, COLORS["neutral"]) for a in mean_grades.index]
    axes[1].barh(range(len(mean_grades)), mean_grades.values, color=colors2, alpha=0.8, edgecolor="white")
    axes[1].set_yticks(range(len(mean_grades)))
    axes[1].set_yticklabels(mean_grades.index, fontsize=11)
    axes[1].set_xlabel("Mean PFF grade (most recent season)")
    axes[1].set_title("2027 FA class quality by archetype")
    axes[1].invert_yaxis()
    for i, v in enumerate(mean_grades.values):
        axes[1].text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=11, fontweight="bold")

    fig.suptitle("2027 CB free agent class composition",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def fig33_value_estimates(fa_valued):
    """Top 2027 FA value estimates with confidence ranges."""
    top = fa_valued.nlargest(20, "est_apy_mid").sort_values("est_apy_mid", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 9))

    colors = [ARCHETYPE_COLOR_MAP.get(a, COLORS["neutral"]) for a in top["archetype"]]
    y_pos = range(len(top))

    ax.barh(y_pos, top["est_apy_mid"] / 1e6, color=colors, alpha=0.8, edgecolor="white", height=0.6)

    for i, (_, row) in enumerate(top.iterrows()):
        ax.plot([row["est_apy_low"] / 1e6, row["est_apy_high"] / 1e6], [i, i],
                color="black", linewidth=1.5, alpha=0.6)
        ax.plot([row["est_apy_low"] / 1e6], [i], "|", color="black", markersize=8, alpha=0.6)
        ax.plot([row["est_apy_high"] / 1e6], [i], "|", color="black", markersize=8, alpha=0.6)

    labels = [f"{row['player']} ({row['archetype'][:15]})" for _, row in top.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Estimated APY ($M, 2027 dollars)")
    ax.set_title("2027 CB free agent value estimates — top 20 by projected APY",
                 fontweight="bold")

    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(row["est_apy_mid"] / 1e6 + 0.2, i, f"${row['est_apy_mid'] / 1e6:.1f}M",
                va="center", fontsize=9)

    fig.tight_layout()
    return fig


def fig34_key_targets(fa_valued):
    """Grade vs estimated APY scatter for 2027 FAs."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for archetype in sorted(fa_valued["archetype"].unique()):
        arch_mask = fa_valued["archetype"] == archetype
        color = ARCHETYPE_COLOR_MAP.get(archetype, COLORS["neutral"])
        # Stable archetype: filled dot. Unstable: hollow ring — visual warning
        # that the comp set for this player's valuation may not fit cleanly.
        stable = arch_mask & fa_valued["archetype_stable"]
        unstable = arch_mask & ~fa_valued["archetype_stable"]
        ax.scatter(fa_valued.loc[stable, "latest_grade"],
                   fa_valued.loc[stable, "est_apy_mid"] / 1e6,
                   c=color, label=archetype, alpha=0.6, s=40, edgecolors="none")
        ax.scatter(fa_valued.loc[unstable, "latest_grade"],
                   fa_valued.loc[unstable, "est_apy_mid"] / 1e6,
                   facecolors="none", edgecolors=color, linewidths=1.5, alpha=0.85, s=55)

    notable = fa_valued.nlargest(12, "latest_grade")
    for _, row in notable.iterrows():
        label = row["player"] + ("*" if not row["archetype_stable"] else "")
        ax.annotate(label, (row["latest_grade"], row["est_apy_mid"] / 1e6),
                    fontsize=8, textcoords="offset points", xytext=(5, 5), alpha=0.8)

    ax.scatter([], [], facecolors="none", edgecolors=COLORS["neutral"], linewidths=1.5,
               s=55, label="Unstable archetype (last 2 seasons differ)")

    ax.set_xlabel("PFF grade (most recent season)")
    ax.set_ylabel("Estimated APY ($M, 2027)")
    ax.set_title("2027 FA class — grade vs. projected contract value", fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    return fig


def fig35_class_comparison(fa_2027_valued, fa_2026_path):
    """Side-by-side comparison of 2026 vs 2027 FA classes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 2027 class
    counts_27 = fa_2027_valued["archetype"].value_counts()
    colors_27 = [ARCHETYPE_COLOR_MAP.get(a, COLORS["neutral"]) for a in counts_27.index]
    axes[1].barh(range(len(counts_27)), counts_27.values, color=colors_27, alpha=0.8, edgecolor="white")
    axes[1].set_yticks(range(len(counts_27)))
    axes[1].set_yticklabels(counts_27.index, fontsize=10)
    axes[1].set_xlabel("Number of free agents")
    axes[1].set_title("2027 FA class")
    axes[1].invert_yaxis()
    for i, v in enumerate(counts_27.values):
        axes[1].text(v + 0.2, i, str(v), va="center", fontsize=10, fontweight="bold")

    # 2026 class (if available)
    if fa_2026_path.exists():
        fa_2026 = pd.read_csv(fa_2026_path)
        counts_26 = fa_2026["archetype"].value_counts()
        colors_26 = [ARCHETYPE_COLOR_MAP.get(a, COLORS["neutral"]) for a in counts_26.index]
        axes[0].barh(range(len(counts_26)), counts_26.values, color=colors_26, alpha=0.8, edgecolor="white")
        axes[0].set_yticks(range(len(counts_26)))
        axes[0].set_yticklabels(counts_26.index, fontsize=10)
        axes[0].set_xlabel("Number of free agents")
        axes[0].set_title("2026 FA class")
        axes[0].invert_yaxis()
        for i, v in enumerate(counts_26.values):
            axes[0].text(v + 0.2, i, str(v), va="center", fontsize=10, fontweight="bold")
    else:
        axes[0].text(0.5, 0.5, "2026 FA data\nnot available", ha="center", va="center",
                     transform=axes[0].transAxes, fontsize=14)

    fig.suptitle("FA class comparison — 2026 vs 2027",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not INPUT_CLUSTERED.exists():
        raise FileNotFoundError(f"Clustered data not found: {INPUT_CLUSTERED}")
    if not INPUT_FA_2027.exists():
        raise FileNotFoundError(f"2027 FA data not found: {INPUT_FA_2027}")

    print("=" * 60)
    print("Loading data")
    print("=" * 60)
    clustered = pd.read_csv(INPUT_CLUSTERED)
    fa_2027_raw = pd.read_csv(INPUT_FA_2027)
    print(f"  Clustered: {len(clustered)} player-seasons")
    print(f"  2027 FA candidates: {len(fa_2027_raw)}")
    print(f"  Projected 2027 cap: ${CAP_2027:,.0f}")

    # ----- Assign archetypes -----
    print("\n" + "=" * 60)
    print("Assigning archetypes")
    print("=" * 60)
    fa_profiled = assign_archetypes(fa_2027_raw, clustered)
    print(f"  Matched: {len(fa_profiled)} / {len(fa_2027_raw)}")
    print(f"  By archetype:")
    print(fa_profiled["archetype"].value_counts().to_string())

    # ----- Estimate fair value -----
    print("\n" + "=" * 60)
    print("Estimating fair contract values")
    print("=" * 60)
    fa_valued = estimate_fair_value(fa_profiled, clustered)
    print(f"  Valuations computed for {len(fa_valued)} FAs")

    # ----- Generate charts -----
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Generating charts")
    print("=" * 60)

    charts = [
        ("32_fa_2027_archetype_breakdown.png", fig32_archetype_breakdown, (fa_valued,)),
        ("33_fa_2027_value_estimates.png", fig33_value_estimates, (fa_valued,)),
        ("34_fa_2027_key_targets.png", fig34_key_targets, (fa_valued,)),
        ("35_fa_2026_vs_2027_comparison.png", fig35_class_comparison, (fa_valued, INPUT_FA_2026_VALS)),
    ]

    for filename, func, args in charts:
        fig = func(*args)
        path = FIG_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {path.name}")

    # ----- Save valuation table -----
    output_cols = [
        "player", "team", "fa_type", "age", "archetype", "archetype_stable",
        "latest_season",
        "latest_grade", "latest_cov_grade", "qb_rating_against", "yards_per_cov_snap",
        "playmaking_rate", "outside_rate", "slot_rate",
        "current_apy",
        "est_cap_units_median", "est_apy_low", "est_apy_mid", "est_apy_high",
        "comparable_n", "used_grade_band",
        "mkt_cap_units_median", "mkt_apy_low", "mkt_apy_mid", "mkt_apy_high",
        "mkt_comparable_n", "used_grade_band_mkt", "market_premium_pct",
    ]
    fa_output = fa_valued[output_cols].sort_values("mkt_apy_mid", ascending=False)
    fa_output.to_csv(TABLE_DIR / "fa_2027_valuations.csv", index=False)
    print(f"  Saved: fa_2027_valuations.csv")

    # ----- Print findings -----
    print("\n" + "=" * 60)
    print("2027 FA class analysis")
    print("=" * 60)

    print(f"\n  Top 10 projected contracts (market-adjusted):")
    for _, row in fa_valued.nlargest(10, "mkt_apy_mid").iterrows():
        print(f"    {row['player']:25s} | {row['archetype']:25s} | "
              f"Grade: {row['latest_grade']:.1f} | Age: {row['age']} | "
              f"Market APY: ${row['mkt_apy_mid'] / 1e6:.1f}M "
              f"(${row['mkt_apy_low'] / 1e6:.1f}M - ${row['mkt_apy_high'] / 1e6:.1f}M) | "
              f"Fair: ${row['est_apy_mid'] / 1e6:.1f}M | "
              f"Premium: +{row['market_premium_pct']:.1f}%")

    print(f"\n  Best value targets (high grade, efficient archetype):")
    value = fa_valued.sort_values("latest_grade", ascending=False)
    value_targets = value[
        value["latest_grade"] > value["archetype"].map(
            value.groupby("archetype")["latest_grade"].mean()
        )
    ].head(10)
    for _, row in value_targets.iterrows():
        print(f"    {row['player']:25s} | {row['archetype']:25s} | "
              f"Grade: {row['latest_grade']:.1f} | Age: {row['age']} | "
              f"Market: ${row['mkt_apy_mid'] / 1e6:.1f}M | Fair: ${row['est_apy_mid'] / 1e6:.1f}M")

    print(f"\n  Christian Gonzalez spotlight:")
    cg = fa_valued[fa_valued["player"] == "Christian Gonzalez"]
    if len(cg) > 0:
        r = cg.iloc[0]
        print(f"    Archetype: {r['archetype']}")
        print(f"    Grade: {r['latest_grade']:.1f} (coverage: {r['latest_cov_grade']:.1f})")
        print(f"    QB rating allowed: {r['qb_rating_against']:.1f}")
        print(f"    Current APY: ${r['current_apy']:,.0f} ({r['current_apy']/CAP_2027*100:.2f}% of 2027 cap)")
        print(f"    Fair value: ${r['est_apy_mid']/1e6:.1f}M | "
              f"Market-adjusted: ${r['mkt_apy_mid']/1e6:.1f}M "
              f"(${r['mkt_apy_low']/1e6:.1f}M - ${r['mkt_apy_high']/1e6:.1f}M)")
        print(f"    Market premium: +{r['market_premium_pct']:.1f}% of cap over fair value")
        print(f"    At age {r['age']}, he's the youngest premium FA in this class")

    print(f"\n  Avoid list (declining production, high current APY):")
    overpaid = fa_valued[
        (fa_valued["current_apy"] > 3_000_000) &
        (fa_valued["latest_grade"] < 60)
    ].sort_values("current_apy", ascending=False)
    for _, row in overpaid.iterrows():
        print(f"    {row['player']:25s} | Grade: {row['latest_grade']:.1f} | Age: {row['age']} | "
              f"Current: ${row['current_apy']/1e6:.1f}M | Fair: ${row['est_apy_mid']/1e6:.1f}M")

    print(f"\n  All charts saved to: {FIG_DIR}/")
    print(f"  Valuation table saved to: {TABLE_DIR}/fa_2027_valuations.csv")


if __name__ == "__main__":
    main()