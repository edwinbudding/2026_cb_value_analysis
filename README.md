# NFL Cornerback Value Analysis

**A data-driven framework for identifying market mispricing in NFL cornerback contracts, applied to a real Patriots front-office decision: extending Christian Gonzalez.**

Authors: Anokh Palakurthi, Mason Marathias, Alon Wigodsky
Course: Brandeis MSBA — Sports Analytics, Spring 2026

---

## Project Overview

NFL cornerback contracts have grown more than 40% since 2018, but cornerback performance is notoriously volatile year-over-year. This project asks: **are NFL teams paying cornerbacks efficiently?** We answer that question with a four-stage framework:

1. Cluster CBs into performance archetypes using PFF tracking data (2018–2025)
2. Overlay 5,189 contracts from Over The Cap to measure cap efficiency by archetype
3. Identify systematic market mispricing across archetypes and contract tiers
4. Apply the framework to the 2027 free-agent market — specifically, whether the Patriots should extend Christian Gonzalez before he hits free agency

---

## Key Findings

- **The slot CB arbitrage**: Slot specialists deliver 46.1 grade points per cap unit, vs. 23.4 for elite lockdown CBs — a 2x efficiency gap that the market does not correct
- **The replacement-level mispricing**: Backup-level CBs (sub-60 PFF grade) command nearly identical median cap space (3.65%) as competent starters (3.72%) — pure market inefficiency
- **The max-contract trap**: Of 136 player-seasons at 5%+ of cap (2018–2025), only 38% delivered elite play (PFF 70+); 27% were outright busts. The 5–7.5% cap tier has the worst hit rate of any tier (35%)
- **Archetype volatility**: 52% of multi-season CBs change archetypes year-to-year, making long-term guaranteed money at the position high-risk
- **Application — Christian Gonzalez**: Stable Playmaker archetype across both qualifying seasons, with coverage metrics improving year-over-year (QB rating allowed: 70.5 → 57.0). Recommendation: extend now with a frontloaded structure, before market pricing shifts

---

## Methodology

### Data
- **Performance**: PFF premium grading and tracking data, 2018–2025 (964 starter player-seasons after a 300-snap filter)
- **Contracts**: Over The Cap APY, contract length, and signing year for 5,189 CB contracts
- **Cap context**: Salary cap history 2018–2025, with 2026 hardcoded to OTC's $301.2M projection and 2027 to $327M

### Pipeline (9 scripts)
1. `01_ingest_merge.py` — Raw PFF CSV ingestion across three file types and eight seasons
2. `02_clean_engineer.py` — Feature engineering (21 rate-based features), missing-data handling, role tier classification
3. `03_eda.py` — Distributions, correlations, multicollinearity diagnostics
4. `04_feature_importance.py` — Random Forest + Gradient Boosting + SHAP, with a robustness check predicting QB rating allowed (rather than PFF grade) to isolate independent CB-quality signal from PFF's grade composition
5. `05_pca_cluster.py` — PCA (7 components, 81% variance explained), k-means clustering (k=5), bootstrap stability validation (mean ARI = 0.92 across 20 seeds)
6. `06_contract_overlay.py` — Contract matching, cap-unit calculation, archetype efficiency analysis (with both all-contract and FA-market-only views)
7. `07_fa_application.py` — 2026 FA class identification and valuation (fair value + market-adjusted estimates)
8. `08_historical_cases.py` — Best-value contracts, post-extension busts, archetype transitions, rookie-deal steals
9. `09_fa_2027_extension.py` — 2027 FA class with Christian Gonzalez case study

### Modeling Decisions Worth Noting
- **Cap units instead of dollars** to normalize across years of cap inflation
- **Median rather than mean** for archetype contract analysis to handle right-skew from mega-contracts
- **Player-season as unit of analysis** for the max-contract hit rate study (each year is its own cap allocation decision)
- **Non-rookie filter (>1.5% cap)** for FA market analysis to avoid rookie-contract distortion
- **±8 PFF grade band** for FA value comp lookups, with full-archetype fallback when fewer than 15 in-band comps exist

---

## Repository Structure

```
sports_project_cb/
├── data/
│   ├── raw/                    # PFF CSVs and OTC contract data
│   └── processed/              # Cleaned and clustered datasets
├── scripts/                    # Numbered analysis pipeline (01-09)
├── outputs/
│   ├── figures/                # 35+ charts produced by the pipeline
│   └── tables/                 # Cluster profiles, FA valuations, historical cases
├── requirements.txt
└── README.md
```

---

## Reproducing the Analysis

```bash
# Install dependencies
pip install -r requirements.txt --break-system-packages

# Run the pipeline in order
python3 scripts/01_ingest_merge.py
python3 scripts/02_clean_engineer.py
python3 scripts/03_eda.py
python3 scripts/04_feature_importance.py
python3 scripts/05_pca_cluster.py
python3 scripts/06_contract_overlay.py
python3 scripts/07_fa_application.py
python3 scripts/08_historical_cases.py
python3 scripts/09_fa_2027_extension.py
```

Each script prints summary statistics and writes figures and tables to `outputs/`. The pipeline runs end-to-end in approximately 60 seconds on a modern laptop.

---

## Limitations

This analysis is honest about its boundaries:

- **PFF grades evaluate role fulfillment, not team impact.** A CB excelling in his coverage assignments may still play on a defense that struggles overall
- **Non-coverage responsibilities are underrepresented.** Run support, communication, and leadership matter for real CB value but are not directly measured in our framework
- **Cluster assignments use only the most recent season** — a player whose archetype changed year-to-year may be misassigned
- **Contract projections are archetype-median based** and tend to compress estimates toward the middle. The model is likely conservative for ascending young CBs
- **Each team's cap context differs.** A Super Bowl-window team has different risk tolerance than a rebuilding one; the framework provides inputs to that decision, not a one-size-fits-all answer

---

## Tech Stack

- Python 3.12, pandas, numpy
- scikit-learn (StandardScaler, PCA, KMeans, RandomForestRegressor, GradientBoostingRegressor)
- shap (feature importance)
- matplotlib (visualization)
- joblib (model persistence)

---

## Contact

Anokh Palakurthi — [bignokh.com](https://bignokh.com)

NOTE: THE DATA FOLDER IS TOO BIG (AND PROPRIETARY) FOR ME TO UPLOAD DIRECTLY; PLEASE CONTACT ME DIRECTLY FOR DATA USED IN THE PROJECT
