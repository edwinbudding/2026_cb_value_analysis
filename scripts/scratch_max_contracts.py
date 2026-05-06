import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

PROJECT_ROOT = Path("/Users/anokhpalakurthi/Documents/sports_project_cb")
df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "cb_with_contracts.csv")

# Filter to non-rookie deals (>1.5% of cap, matches your script 06 methodology)
fa_market = df.dropna(subset=["cap_units"]).copy()
fa_market = fa_market[fa_market["cap_units"] > 1.5]
print(f"Non-rookie sample: {len(fa_market)} player-seasons")

# Match deck colors (same archetype color map you've been using)
ARCHETYPE_COLORS = {
    "Elite lockdown CB": "#534AB7",
    "Playmaking outside CB": "#1D9E75",
    "Average outside CB": "#D85A30",
    "Replacement-level CB": "#378ADD",
    "Slot specialist": "#BA7517",
}

# Order by median cap units descending
order = (fa_market.groupby("archetype")["cap_units"]
         .median().sort_values(ascending=False).index.tolist())

fig, ax = plt.subplots(figsize=(12, 6))
data = [fa_market[fa_market["archetype"] == a]["cap_units"].values for a in order]
bp = ax.boxplot(data, tick_labels=order, patch_artist=True, widths=0.6,
                medianprops=dict(color="white", linewidth=2))

for patch, archetype in zip(bp["boxes"], order):
    patch.set_facecolor(ARCHETYPE_COLORS[archetype])
    patch.set_alpha(0.85)

ax.set_ylabel("Cap units (APY as % of salary cap)")
ax.set_xlabel("CB archetype")
ax.set_title("Contract value by CB archetype — FA market only (rookie deals excluded)",
             fontweight="bold")

# Add median labels
medians = fa_market.groupby("archetype")["cap_units"].median().reindex(order)
for i, med in enumerate(medians):
    ax.text(i + 1, med + 0.2, f"{med:.2f}%", ha="center",
            fontsize=10, fontweight="bold")

# Sample size annotations
counts = fa_market.groupby("archetype").size().reindex(order)
for i, n in enumerate(counts):
    ax.text(i + 1, -0.5, f"n={n}", ha="center", fontsize=9, color="#666666")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(PROJECT_ROOT / "outputs" / "figures" / "20b_cap_units_by_archetype_fa_only.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved: 20b_cap_units_by_archetype_fa_only.png")

# Also print the new median values for your reference
print("\nNew medians (non-rookie only):")
for archetype in order:
    med = fa_market[fa_market["archetype"] == archetype]["cap_units"].median()
    print(f"  {archetype}: {med:.2f}%")