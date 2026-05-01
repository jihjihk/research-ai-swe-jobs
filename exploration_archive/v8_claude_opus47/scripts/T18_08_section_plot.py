"""T18 Step 8 — Section anatomy plot.

Plots requirements-section share by group-period. Builds on the T18 step 6
output.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
TAB = ROOT / "exploration" / "tables" / "T18"
FIG = ROOT / "exploration" / "figures" / "T18"

sec = pd.read_csv(TAB / "T18_section_anatomy_by_group_period.csv")
PERIODS = ["2024-01", "2024-04", "2026-03", "2026-04"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = {"SWE": "#d62728", "adjacent": "#ff7f0e", "control": "#1f77b4"}

# Requirements share
ax = axes[0]
for group in ["SWE", "adjacent", "control"]:
    sub = sec[sec["group"] == group].set_index("period").reindex(PERIODS)
    ax.plot(range(len(PERIODS)), sub["share_requirements"].values,
            marker="o", color=colors[group], label=group, linewidth=2)
ax.set_xticks(range(len(PERIODS)))
ax.set_xticklabels(PERIODS, fontsize=9, rotation=15)
ax.set_title("Requirements-section share")
ax.set_ylabel("Share of description characters")
ax.grid(True, alpha=0.3)
ax.legend()

# Responsibilities share
ax = axes[1]
for group in ["SWE", "adjacent", "control"]:
    sub = sec[sec["group"] == group].set_index("period").reindex(PERIODS)
    ax.plot(range(len(PERIODS)), sub["share_responsibilities"].values,
            marker="o", color=colors[group], label=group, linewidth=2)
ax.set_xticks(range(len(PERIODS)))
ax.set_xticklabels(PERIODS, fontsize=9, rotation=15)
ax.set_title("Responsibilities-section share")
ax.set_ylabel("Share of description characters")
ax.grid(True, alpha=0.3)
ax.legend()

plt.suptitle("T18 — Section anatomy across three groups", fontsize=12)
plt.tight_layout()
plt.savefig(FIG / "T18_section_anatomy.png", dpi=150, bbox_inches="tight")
plt.close()
print("Wrote", FIG / "T18_section_anatomy.png")
