#!/usr/bin/env python3
"""T28 step 3: Figures for the report."""
from __future__ import annotations

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

OUT_F = "exploration/figures/T28"
os.makedirs(OUT_F, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150})

# ---------------------------------------------------------------------------
# Figure 1: Archetype share shift (domain only)
# ---------------------------------------------------------------------------
dom = pd.read_csv("exploration/tables/T28/archetype_share_by_period_domain.csv")
dom = dom.sort_values("delta_pp", ascending=True)
fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#d62728" if d < 0 else "#2ca02c" for d in dom["delta_pp"]]
ax.barh(dom["archetype_name"].str[:45], dom["delta_pp"], color=colors)
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("Δ share 2024→2026 (percentage points, within domain)")
ax.set_title("T28 Fig 1: Archetype share shift (aggregators included, artifacts excluded)")
plt.tight_layout()
plt.savefig(f"{OUT_F}/fig1_archetype_share_shift.png")
plt.close()

# ---------------------------------------------------------------------------
# Figure 2: Per-archetype entry rate (YOE<=2 share) by period
# ---------------------------------------------------------------------------
contrib = pd.read_csv("exploration/tables/T28/entry_share_per_archetype_contrib.csv")
contrib["short_name"] = contrib["archetype_name"].str[:35]
contrib = contrib.sort_values("rate_2026", ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
x = np.arange(len(contrib))
width = 0.38
ax.barh(x - width / 2, contrib["rate_2024"] * 100, width, label="2024", color="#1f77b4")
ax.barh(x + width / 2, contrib["rate_2026"] * 100, width, label="2026", color="#ff7f0e")
ax.set_yticks(x)
ax.set_yticklabels(contrib["short_name"])
ax.set_xlabel("Entry-share under combined_augmented (%)")
ax.set_title("T28 Fig 2: Entry share per archetype (combined augmented)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_F}/fig2_entry_rate_per_archetype.png")
plt.close()

# ---------------------------------------------------------------------------
# Figure 3: Within-Between decomposition bar
# ---------------------------------------------------------------------------
decomp = pd.read_csv("exploration/tables/T28/entry_share_decomposition.csv")
fig, ax = plt.subplots(figsize=(10, 6))
labels = decomp["label"]
within = decomp["within_pp"]
between = decomp["between_pp"]
interaction = decomp["interaction_pp"]
x = np.arange(len(labels))
ax.bar(x, within, label="Within-domain", color="#1f77b4")
ax.bar(x, between, bottom=within, label="Between-domain", color="#ff7f0e")
ax.bar(x, interaction, bottom=within + between, label="Interaction", color="#2ca02c")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha="right")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("Δ entry-share (percentage points)")
ax.set_title("T28 Fig 3: Within/Between decomposition by seniority operationalization")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_F}/fig3_decomposition_bar.png")
plt.close()

# ---------------------------------------------------------------------------
# Figure 4: Credential stack depth 7+ rise by archetype
# ---------------------------------------------------------------------------
cred = pd.read_csv("exploration/tables/T28/credential_stack_7_by_archetype.csv")
cred["short"] = cred["archetype_name"].str[:35]
cred = cred.sort_values("pct_2026", ascending=True)
fig, ax = plt.subplots(figsize=(10, 8))
x = np.arange(len(cred))
ax.barh(x - 0.18, cred["pct_2024"], 0.36, label="2024", color="#1f77b4")
ax.barh(x + 0.18, cred["pct_2026"], 0.36, label="2026", color="#ff7f0e")
ax.set_yticks(x)
ax.set_yticklabels(cred["short"])
ax.set_xlabel("Share of postings with credential stack depth ≥7 (%)")
ax.set_title("T28 Fig 4: Credential stacking (stack≥7) by archetype")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_F}/fig4_credential_stack_by_archetype.png")
plt.close()

# ---------------------------------------------------------------------------
# Figure 5: Senior mentoring growth by archetype
# ---------------------------------------------------------------------------
sen = pd.read_csv("exploration/tables/T28/senior_mentor_vs_pmgr_by_archetype.csv")
sen["short"] = sen["archetype_name"].str[:35]
sen = sen.sort_values("mentor_delta_pp", ascending=True)
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(sen["short"], sen["mentor_delta_pp"], color="#2ca02c")
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("Δ mentoring language 2024→2026 (pp, senior tier only, YOE≥5)")
ax.set_title("T28 Fig 5: Senior-tier mentoring growth by archetype")
plt.tight_layout()
plt.savefig(f"{OUT_F}/fig5_senior_mentoring_by_archetype.png")
plt.close()

print("All figures saved.")
