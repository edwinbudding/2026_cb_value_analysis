"""
08_historical_cases.py

Deep-dive into historical case studies that bring the archetypes and contract
analysis to life. Uses cap units with raw dollar amounts in parentheses
for cross-year comparability.

Input:
    {PROJECT_ROOT}/data/processed/cb_with_contracts.csv

Outputs:
    {PROJECT_ROOT}/outputs/figures/27_best_worst_contracts.png
    {PROJECT_ROOT}/outputs/figures/28_archetype_transitions.png
    {PROJECT_ROOT}/outputs/figures/29_rookie_deal_steals.png
    {PROJECT_ROOT}/outputs/figures/30_post_extension_busts.png
    {PROJECT_ROOT}/outputs/figures/31_case_study_trajectories.png
    {PROJECT_ROOT}/outputs/tables/historical_cases.csv

Design notes:
- All contract values are shown as cap units (% of salary cap) with raw
  dollar amounts in parentheses. This makes 2018 and 2025 contracts directly
  comparable: Josh Norman at 7.97% in 2019 ($15M) is a worse deal than
  L'Jarius Sneed at 7.75% in 2024 ($19.8M) because Norman got less grade
  per cap point.
- Case studies selected to illustrate each archetype's risk/reward profile
  and common front office mistakes.

Usage:
    cd ~/Documents/sports_project_cb
    python3 scripts/08_historical_cases.py
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

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cb_with_contracts.csv"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

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


def fmt_cap(row):
    """Format cap units with dollar amount: '7.97% ($15.0M)'"""
    if pd.isna(row["cap_units"]) or pd.isna(row["apy"]):
        return "N/A"
    return f"{row['cap_units']:.2f}% (${row['apy'] / 1e6:.1f}M)"


# ---------------------------------------------------------------------------
# Chart functions
# ---------------------------------------------------------------------------

def fig27_best_worst_contracts(df):
    """
    Side-by-side: best value contracts vs worst value contracts.
    The contrast tells the entire story of CB market efficiency.
    """
    valid = df.dropna(subset=["cap_units"])

    # Best: high grade, low cap
    best = valid[(valid["grades_defense"] >= 70) & (valid["cap_units"] < 1.5)]
    best = best.sort_values("grades_defense", ascending=False).head(10)

    # Worst: low grade, high cap
    worst = valid[(valid["grades_defense"] < 55) & (valid["cap_units"] > 4)]
    worst = worst.sort_values("cap_units", ascending=False).head(10)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Best value
    ax = axes[0]
    colors = [ARCHETYPE_COLOR_MAP.get(a, COLORS["neutral"]) for a in best["archetype"]]
    labels = [f"{r['player']} ({int(r['season'])})" for _, r in best.iterrows()]
    y_pos = range(len(best))

    ax.barh(y_pos, best["grades_defense"].values, color=colors, alpha=0.8, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("PFF grade")
    ax.set_title("Best value — elite play, bargain price", fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(65, 95)

    for i, (_, r) in enumerate(best.iterrows()):
        ax.text(r["grades_defense"] + 0.3, i, fmt_cap(r), va="center", fontsize=8)

    # Worst value
    ax2 = axes[1]
    colors2 = [ARCHETYPE_COLOR_MAP.get(a, COLORS["neutral"]) for a in worst["archetype"]]
    labels2 = [f"{r['player']} ({int(r['season'])})" for _, r in worst.iterrows()]
    y_pos2 = range(len(worst))

    ax2.barh(y_pos2, worst["cap_units"].values, color=colors2, alpha=0.8, edgecolor="white", height=0.6)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(labels2, fontsize=9)
    ax2.set_xlabel("Cap units (% of salary cap)")
    ax2.set_title("Worst value — premium price, poor production", fontweight="bold")
    ax2.invert_yaxis()

    for i, (_, r) in enumerate(worst.iterrows()):
        ax2.text(r["cap_units"] + 0.1, i,
                 f"Grade: {r['grades_defense']:.0f} (${r['apy'] / 1e6:.1f}M)",
                 va="center", fontsize=8)

    fig.suptitle("The CB contract spectrum — best steals vs worst overpays",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def fig28_archetype_transitions(df):
    """
    Track notable players across seasons showing archetype changes.
    Demonstrates that CB performance (and archetype) is volatile.
    """
    valid = df.dropna(subset=["cap_units"])

    # Select players with interesting trajectories
    case_players = [
        "Jaire Alexander",     # Replacement → Playmaking → Elite → Elite
        "Jaylon Johnson",      # Playmaking → Elite lockdown → bust
        "DaRon Bland",         # Slot → Playmaking (breakout) → Replacement
        "Xavien Howard",       # Elite → Playmaking (paid) → decline
        "Devon Witherspoon",   # Average outside → Slot specialist (elite)
        "Pat Surtain II",      # Playmaking → Elite → Average → Elite
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, player_name in enumerate(case_players):
        ax = axes[idx]
        player_data = valid[valid["player"] == player_name].sort_values("season")

        if len(player_data) == 0:
            # Try without contract data
            player_data = df[df["player"] == player_name].sort_values("season")

        if len(player_data) == 0:
            ax.text(0.5, 0.5, f"{player_name}\n(not in data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            continue

        seasons = player_data["season"].values
        grades = player_data["grades_defense"].values
        archetypes = player_data["archetype"].values

        # Plot grade line
        ax.plot(seasons, grades, "o-", color=COLORS["neutral"], linewidth=2, markersize=8, zorder=5)

        # Color each point by archetype
        for s, g, a in zip(seasons, grades, archetypes):
            color = ARCHETYPE_COLOR_MAP.get(a, COLORS["neutral"])
            ax.scatter(s, g, c=color, s=80, zorder=10, edgecolors="white", linewidth=1)

        # Add archetype labels
        for s, g, a in zip(seasons, grades, archetypes):
            short_name = a.replace(" outside CB", "").replace(" CB", "").replace("-level", "")
            ax.annotate(short_name, (s, g), textcoords="offset points",
                       xytext=(0, 12), fontsize=7, ha="center", alpha=0.8)

        # Add cap info where available
        if "cap_units" in player_data.columns:
            for _, r in player_data.iterrows():
                if pd.notna(r.get("cap_units")):
                    ax.annotate(f"{r['cap_units']:.1f}%", (r["season"], r["grades_defense"]),
                               textcoords="offset points", xytext=(0, -15),
                               fontsize=7, ha="center", color=COLORS["neutral"])

        ax.set_title(player_name, fontweight="bold", fontsize=12)
        ax.set_ylabel("PFF grade")
        ax.set_ylim(30, 95)
        ax.set_xticks(range(2018, 2026))
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Archetype volatility — CB performance fluctuates year to year",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def fig29_rookie_steals(df):
    """Rookie contract steals — elite production at minimum cost."""
    valid = df.dropna(subset=["cap_units"])
    steals = valid[(valid["cap_units"] < 1.0) & (valid["grades_defense"] >= 75)]
    steals = steals.sort_values("grades_defense", ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = [ARCHETYPE_COLOR_MAP.get(a, COLORS["neutral"]) for a in steals["archetype"]]
    labels = [f"{r['player']} ({int(r['season'])})" for _, r in steals.iterrows()]

    ax.scatter(steals["cap_units"], steals["grades_defense"], c=colors, s=100,
               alpha=0.8, edgecolors="white", linewidth=1, zorder=5)

    for _, r in steals.iterrows():
        ax.annotate(f"{r['player']} ({int(r['season'])})",
                    (r["cap_units"], r["grades_defense"]),
                    textcoords="offset points", xytext=(8, 0), fontsize=9)

    ax.set_xlabel("Cap units (% of salary cap)")
    ax.set_ylabel("PFF grade")
    ax.set_title("Rookie deal steals — elite CB play at <1% of cap", fontweight="bold")
    ax.set_xlim(0, 1.1)
    ax.set_ylim(74, 92)

    # Add context box
    ax.text(0.95, 0.05, "All players on rookie/minimum contracts\n"
            "producing at elite or near-elite levels.\n"
            "This is the CB value sweet spot.",
            transform=ax.transAxes, fontsize=10, va="bottom", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor=COLORS["neutral"]))

    fig.tight_layout()
    return fig


def fig30_post_extension_busts(df):
    """Players who got paid and then declined — the extension trap."""
    valid = df.dropna(subset=["cap_units"])
    busts = valid[(valid["cap_units"] >= 5) & (valid["grades_defense"] < 60)]
    busts = busts.sort_values("cap_units", ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = [ARCHETYPE_COLOR_MAP.get(a, COLORS["neutral"]) for a in busts["archetype"]]

    ax.scatter(busts["grades_defense"], busts["cap_units"], c=colors, s=100,
               alpha=0.8, edgecolors="white", linewidth=1, zorder=5)

    for _, r in busts.iterrows():
        ax.annotate(f"{r['player']} ({int(r['season'])})\n{fmt_cap(r)}",
                    (r["grades_defense"], r["cap_units"]),
                    textcoords="offset points", xytext=(8, 0), fontsize=8)

    ax.set_xlabel("PFF grade")
    ax.set_ylabel("Cap units (% of salary cap)")
    ax.set_title("The extension trap — premium money, replacement-level play", fontweight="bold")

    # Danger zone shading
    ax.axvspan(25, 55, alpha=0.05, color="red")
    ax.axhspan(5, 12, alpha=0.05, color="red")

    fig.tight_layout()
    return fig


def fig31_case_study_trajectories(df):
    """
    Deep dive: 3 case studies showing the full contract lifecycle.
    Each panel shows grade trajectory with cap unit annotations.
    """
    valid = df.dropna(subset=["cap_units"])

    cases = {
        "Jalen Ramsey": "The franchise CB — consistent elite play, multiple big deals",
        "Xavien Howard": "Boom-bust playmaker — one great year, paid, then declined",
        "Desmond King II": "The slot value play — elite production at minimum cost",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (player_name, subtitle) in enumerate(cases.items()):
        ax = axes[idx]
        player_data = df[df["player"] == player_name].sort_values("season")

        if len(player_data) == 0:
            ax.text(0.5, 0.5, f"{player_name}\n(not in data)", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        seasons = player_data["season"].values
        grades = player_data["grades_defense"].values
        archetypes = player_data["archetype"].values

        # Grade bars
        for s, g, a in zip(seasons, grades, archetypes):
            color = ARCHETYPE_COLOR_MAP.get(a, COLORS["neutral"])
            ax.bar(s, g, color=color, alpha=0.8, edgecolor="white", width=0.7)

        # Cap unit annotations
        for _, r in player_data.iterrows():
            if pd.notna(r.get("cap_units")) and pd.notna(r.get("apy")):
                label = f"{r['cap_units']:.1f}%\n(${r['apy'] / 1e6:.1f}M)"
            else:
                label = "No data"
            ax.text(r["season"], r["grades_defense"] + 1.5, label,
                    ha="center", fontsize=7, color=COLORS["neutral"])

        ax.set_title(f"{player_name}\n{subtitle}", fontweight="bold", fontsize=11)
        ax.set_ylabel("PFF grade")
        ax.set_ylim(0, 100)
        ax.set_xticks(range(int(min(seasons)), int(max(seasons)) + 1))
        ax.tick_params(axis="x", rotation=45, labelsize=9)

        # Archetype legend for this player
        unique_archs = list(dict.fromkeys(archetypes))
        for i, a in enumerate(unique_archs):
            color = ARCHETYPE_COLOR_MAP.get(a, COLORS["neutral"])
            short = a.replace(" outside CB", " outs.").replace(" CB", "")
            ax.text(0.02, 0.98 - i * 0.06, f"■ {short}", transform=ax.transAxes,
                    fontsize=8, color=color, va="top", fontweight="bold")

    fig.suptitle("Case studies — CB career arcs and contract value",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            f"Run 06_contract_overlay.py first."
        )

    print("=" * 60)
    print("Loading data")
    print("=" * 60)
    df = pd.read_csv(INPUT_PATH)
    valid = df.dropna(subset=["cap_units"])
    print(f"  {len(df)} total player-seasons, {len(valid)} with contract data")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Compute key lists -----
    print("\n" + "=" * 60)
    print("Identifying historical cases")
    print("=" * 60)

    arch_means = valid.groupby("archetype")["grades_defense"].mean()

    # Best value
    best = valid[(valid["grades_defense"] >= 70) & (valid["cap_units"] < 1.5)]
    best = best.sort_values("grades_defense", ascending=False)
    print(f"  Best value contracts (grade 70+, <1.5% cap): {len(best)}")

    # Worst value
    worst = valid[(valid["grades_defense"] < 55) & (valid["cap_units"] > 4)]
    worst = worst.sort_values("cap_units", ascending=False)
    print(f"  Worst value contracts (grade <55, >4% cap): {len(worst)}")

    # Rookie steals
    steals = valid[(valid["cap_units"] < 1.0) & (valid["grades_defense"] >= 75)]
    print(f"  Rookie deal steals (grade 75+, <1% cap): {len(steals)}")

    # Post-extension busts
    busts = valid[(valid["cap_units"] >= 5) & (valid["grades_defense"] < 60)]
    print(f"  Post-extension busts (grade <60, >5% cap): {len(busts)}")

    # Archetype transitions
    multi = valid.groupby("player").filter(lambda x: x["archetype"].nunique() > 1)
    transition_players = multi["player"].nunique()
    print(f"  Players who changed archetypes: {transition_players}")

    # ----- Generate charts -----
    print("\n" + "=" * 60)
    print("Generating charts")
    print("=" * 60)

    charts = [
        ("27_best_worst_contracts.png", fig27_best_worst_contracts, (df,)),
        ("28_archetype_transitions.png", fig28_archetype_transitions, (df,)),
        ("29_rookie_deal_steals.png", fig29_rookie_steals, (df,)),
        ("30_post_extension_busts.png", fig30_post_extension_busts, (df,)),
        ("31_case_study_trajectories.png", fig31_case_study_trajectories, (df,)),
    ]

    for filename, func, args in charts:
        fig = func(*args)
        path = FIG_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {path.name}")

    # ----- Save historical cases table -----
    cases_list = []
    for _, r in best.head(10).iterrows():
        cases_list.append({
            "category": "Best value", "player": r["player"], "season": int(r["season"]),
            "archetype": r["archetype"], "grade": r["grades_defense"],
            "cap_units": r["cap_units"], "apy": r["apy"],
            "cap_label": fmt_cap(r),
        })
    for _, r in worst.head(10).iterrows():
        cases_list.append({
            "category": "Worst value", "player": r["player"], "season": int(r["season"]),
            "archetype": r["archetype"], "grade": r["grades_defense"],
            "cap_units": r["cap_units"], "apy": r["apy"],
            "cap_label": fmt_cap(r),
        })
    for _, r in steals.head(10).iterrows():
        cases_list.append({
            "category": "Rookie steal", "player": r["player"], "season": int(r["season"]),
            "archetype": r["archetype"], "grade": r["grades_defense"],
            "cap_units": r["cap_units"], "apy": r["apy"],
            "cap_label": fmt_cap(r),
        })
    for _, r in busts.head(10).iterrows():
        cases_list.append({
            "category": "Extension bust", "player": r["player"], "season": int(r["season"]),
            "archetype": r["archetype"], "grade": r["grades_defense"],
            "cap_units": r["cap_units"], "apy": r["apy"],
            "cap_label": fmt_cap(r),
        })

    cases_df = pd.DataFrame(cases_list)
    cases_df.to_csv(TABLE_DIR / "historical_cases.csv", index=False)
    print(f"  Saved: historical_cases.csv")

    # ----- Print key narratives -----
    print("\n" + "=" * 60)
    print("Key narratives for presentation")
    print("=" * 60)

    print("\n  1. THE ROOKIE DEAL WINDOW")
    print("     The best CB value in the NFL comes from players on rookie contracts.")
    print("     Examples:")
    for _, r in steals.head(5).iterrows():
        print(f"       {r['player']} ({int(r['season'])}): Grade {r['grades_defense']:.0f} at {fmt_cap(r)}")
    print("     → Teams should draft CBs, not sign them in free agency.")

    print("\n  2. THE EXTENSION TRAP")
    print("     Paying top dollar for CBs based on past performance is high-risk.")
    print("     Examples:")
    for _, r in busts.head(5).iterrows():
        print(f"       {r['player']} ({int(r['season'])}): Grade {r['grades_defense']:.0f} at {fmt_cap(r)}")
    print("     → CB performance is volatile; long-term guaranteed money is dangerous.")

    print("\n  3. ARCHETYPE INSTABILITY")
    print(f"     {transition_players} of {valid['player'].nunique()} multi-season CBs "
          f"({transition_players / valid['player'].nunique() * 100:.0f}%) changed archetypes.")
    print("     Even elite CBs bounce between archetypes year to year:")
    print("       Jaire Alexander: Replacement → Playmaking → Elite → Elite")
    print("       DaRon Bland: Slot → Playmaking (breakout) → Replacement")
    print("       Jaylon Johnson: Playmaking → Elite (90.1 grade) → Elite (58.7)")
    print("     → This is why slot CBs are the best value: their role is more stable.")

    print("\n  4. THE SLOT CB ARBITRAGE")
    print("     Slot specialists consistently produce at 63+ grade for <1.5% of cap.")
    print("     The market systematically undervalues them relative to outside CBs.")
    print("       Desmond King II (2018): Grade 88.5 at 0.38% ($670K)")
    print("       Cooper DeJean (2024): Grade 86.3 at 0.91% ($2.3M)")
    print("       Bryce Callahan (2018): Grade 81.3 at 1.08% ($1.9M)")
    print("     → A $3M/yr slot CB outperforms a $15M/yr outside CB who busts.")

    print(f"\n  All charts saved to: {FIG_DIR}/")
    print(f"  Cases table saved to: {TABLE_DIR}/historical_cases.csv")


if __name__ == "__main__":
    main()
