"""T18 Step 5 — AI gradient across three groups per period.

Plot AI-mention (strict + broad) prevalence for SWE / SWE-adjacent / Control at
each period. Also compute gap_SWE_minus_ctrl per period to see whether SWE is
pulling ahead or control is catching up.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
ART = ROOT / "exploration" / "artifacts" / "T18"
TAB = ROOT / "exploration" / "tables" / "T18"
FIG = ROOT / "exploration" / "figures" / "T18"
FIG.mkdir(parents=True, exist_ok=True)

FEAT = pd.read_parquet(ART / "T18_posting_features.parquet")
PERIODS = ["2024-01", "2024-04", "2026-03", "2026-04"]
PERIOD_LABELS = {
    "2024-01": "Asaniczka\n2024-01",
    "2024-04": "Arshkon\n2024-04",
    "2026-03": "Scraped\n2026-03",
    "2026-04": "Scraped\n2026-04",
}

METRICS = {
    "ai_strict_binary": "AI-mention (strict)",
    "ai_broad_binary": "AI-mention (broad)",
    "has_ai_tech": "Has any AI-era tech",
    "tech_count": "Mean tech count",
    "org_scope_count": "Mean org-scope count",
    "requirement_breadth": "Mean requirement breadth",
}

# Aggregate
agg = (
    FEAT.groupby(["group", "period"], observed=False)[list(METRICS.keys())]
    .mean()
    .reset_index()
)
agg.to_csv(TAB / "T18_ai_gradient_table.csv", index=False)

# Compute gaps
gap_records = []
for period in PERIODS:
    swe = agg[(agg["group"] == "SWE") & (agg["period"] == period)]
    adj = agg[(agg["group"] == "adjacent") & (agg["period"] == period)]
    ctrl = agg[(agg["group"] == "control") & (agg["period"] == period)]
    if swe.empty or ctrl.empty or adj.empty:
        continue
    for m in METRICS:
        gap_records.append({
            "period": period,
            "metric": m,
            "SWE": swe[m].values[0],
            "adjacent": adj[m].values[0],
            "control": ctrl[m].values[0],
            "SWE_minus_control": swe[m].values[0] - ctrl[m].values[0],
            "adj_minus_control": adj[m].values[0] - ctrl[m].values[0],
            "SWE_minus_adj": swe[m].values[0] - adj[m].values[0],
        })
gaps = pd.DataFrame(gap_records)
gaps.to_csv(TAB / "T18_ai_gradient_gaps.csv", index=False)
print("=== AI gradient gaps ===")
print(gaps.to_string())

# Plot
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
colors = {"SWE": "#d62728", "adjacent": "#ff7f0e", "control": "#1f77b4"}
for i, (m, title) in enumerate(METRICS.items()):
    ax = axes[i]
    for group in ["SWE", "adjacent", "control"]:
        sub = agg[agg["group"] == group].set_index("period").reindex(PERIODS)
        ax.plot(
            range(len(PERIODS)),
            sub[m].values,
            marker="o",
            color=colors[group],
            label=group,
            linewidth=2,
        )
    ax.set_xticks(range(len(PERIODS)))
    ax.set_xticklabels([PERIOD_LABELS[p] for p in PERIODS], fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)

plt.suptitle("T18 — Three-group trends (SWE / adjacent / control)", fontsize=12)
plt.tight_layout()
plt.savefig(FIG / "T18_ai_gradient.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Wrote {FIG / 'T18_ai_gradient.png'}")

# Parallel trends plot just for AI strict
fig, ax = plt.subplots(figsize=(8, 5))
for group in ["SWE", "adjacent", "control"]:
    sub = agg[agg["group"] == group].set_index("period").reindex(PERIODS)
    ax.plot(
        range(len(PERIODS)),
        sub["ai_strict_binary"].values,
        marker="o",
        color=colors[group],
        label=group,
        linewidth=2,
    )
ax.set_xticks(range(len(PERIODS)))
ax.set_xticklabels([PERIOD_LABELS[p] for p in PERIODS])
ax.set_ylabel("AI-mention (strict) prevalence")
ax.set_title("AI-mention gradient across three occupation groups")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "T18_parallel_trends_ai_strict.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Wrote {FIG / 'T18_parallel_trends_ai_strict.png'}")
