"""Supplementary T21 figures — cross-seniority mentor rise + domain deltas."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/jihgaboot/gabor/job-research")
TBL = ROOT / "exploration/tables/T21"
FIG = ROOT / "exploration/figures/T21"
FIG.mkdir(parents=True, exist_ok=True)

# --- Cross-seniority mentor bar ---
cross = pd.read_csv(TBL / "T21_cross_seniority_mentor.csv")
cross["period"] = cross["period"].astype(str)
all_sub = cross[cross["subset"] == "all"]
pvt = all_sub.pivot(
    index="seniority", columns="period", values="mentor_binary_share"
).reindex(["entry", "associate", "mid-senior", "director"])

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(pvt))
w = 0.35
ax.bar(x - w / 2, pvt["2024"].values, w, label="2024", color="#3B82F6")
ax.bar(x + w / 2, pvt["2026"].values, w, label="2026", color="#F97316")
for i, b in enumerate(pvt.index):
    ax.text(i - w / 2, pvt.loc[b, "2024"] + 0.005, f"{pvt.loc[b, '2024']:.3f}", ha="center", fontsize=9)
    ax.text(i + w / 2, pvt.loc[b, "2026"] + 0.005, f"{pvt.loc[b, '2026']:.3f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(pvt.index)
ax.set_ylabel("Mentor-binary share (per posting)")
ax.set_title("T21: Mentor rate by seniority × period (V1-refined pattern)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "T21_cross_seniority_mentor.png", dpi=150)
plt.close()
print(f"wrote {FIG / 'T21_cross_seniority_mentor.png'}")

# --- Domain delta plot ---
dom = pd.read_csv(TBL / "T21_domain_deltas.csv")
fig, ax = plt.subplots(figsize=(12, 6))
metrics = [
    ("delta_mgmt_binary", "Δ mgmt"),
    ("delta_orch_strict_binary", "Δ orch"),
    ("delta_strat_broad_binary", "Δ strat-broad"),
    ("delta_ai_mention", "Δ AI"),
    ("delta_mentor_binary", "Δ mentor"),
]
archs = dom["archetype"].tolist()
x = np.arange(len(archs))
bw = 0.15
for i, (col, lbl) in enumerate(metrics):
    ax.bar(x + (i - 2) * bw, dom[col].values, bw, label=lbl)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels([a[:15] for a in archs], rotation=20, ha="right")
ax.set_ylabel("Δ binary share (2026 − 2024)")
ax.set_title("T21: Senior density shifts by archetype")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "T21_domain_deltas.png", dpi=150)
plt.close()
print(f"wrote {FIG / 'T21_domain_deltas.png'}")
