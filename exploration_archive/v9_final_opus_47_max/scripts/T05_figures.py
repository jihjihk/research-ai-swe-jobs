"""Figures for T05."""
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TBL = Path("/home/jihgaboot/gabor/job-research/exploration/tables/T05")
FIG = Path("/home/jihgaboot/gabor/job-research/exploration/figures/T05")
FIG.mkdir(parents=True, exist_ok=True)

# Seniority bar
sen = pd.read_csv(TBL / "seniority_shares.csv")
sen = sen.set_index("seniority_final")
order = ["entry", "associate", "mid-senior", "director"]
sen = sen.reindex(order)
ax = sen.plot.bar(figsize=(8, 5), width=0.75)
ax.set_ylabel("share of labeled SWE postings")
ax.set_title("seniority_final distribution by source (LinkedIn, SWE, non-aggregator)\nexcluding 'unknown'")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(FIG / "seniority_final_bar.png", dpi=120)
plt.close()
print(f"wrote {FIG / 'seniority_final_bar.png'}")


# State shares top 15
state = pd.read_csv(TBL / "state_shares.csv")
state["max_share"] = state[["kaggle_arshkon", "kaggle_asaniczka", "scraped"]].max(axis=1)
state = state.sort_values("max_share", ascending=False).head(15)
state = state.set_index("state_normalized")[["kaggle_arshkon", "kaggle_asaniczka", "scraped"]]
ax = state.plot.bar(figsize=(11, 5), width=0.75)
ax.set_ylabel("share of SWE postings")
ax.set_title("State distribution by source (top 15 states) — LinkedIn SWE, non-aggregator")
plt.tight_layout()
plt.savefig(FIG / "state_shares_top15.png", dpi=120)
plt.close()
print(f"wrote {FIG / 'state_shares_top15.png'}")


# Title-level stability chart: mid-senior share per title per source
native = pd.read_csv(TBL / "shared_titles_seniority_native.csv")
native = native[native["source"].isin(["kaggle_arshkon", "scraped"])]
tot = native.groupby(["title_normalized", "source"])["n"].sum().reset_index().rename(
    columns={"n": "total"}
)
m = native.merge(tot, on=["title_normalized", "source"])
m["share"] = m["n"] / m["total"]
# Plot mid-senior share
mid = m[m["seniority_native"] == "mid-senior"].pivot_table(
    index="title_normalized", columns="source", values="share"
).fillna(0)
entry = m[m["seniority_native"] == "entry"].pivot_table(
    index="title_normalized", columns="source", values="share"
).fillna(0)
mid = mid.sort_values("kaggle_arshkon")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(mid["kaggle_arshkon"], mid["scraped"], s=40)
axes[0].plot([0, 1], [0, 1], "r--", alpha=0.4)
for i, t in enumerate(mid.index):
    axes[0].annotate(t[:20], (mid["kaggle_arshkon"].iloc[i], mid["scraped"].iloc[i]),
                     fontsize=6, alpha=0.7)
axes[0].set_xlabel("arshkon (2024) mid-senior share of native labels")
axes[0].set_ylabel("scraped (2026) mid-senior share of native labels")
axes[0].set_title("Native mid-senior share drift per title\n(y > x ⇒ platform relabeling toward mid-senior)")
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)

axes[1].scatter(entry["kaggle_arshkon"], entry["scraped"], s=40, color="orange")
axes[1].plot([0, 1], [0, 1], "r--", alpha=0.4)
for i, t in enumerate(entry.index):
    axes[1].annotate(t[:20], (entry["kaggle_arshkon"].iloc[i], entry["scraped"].iloc[i]),
                     fontsize=6, alpha=0.7)
axes[1].set_xlabel("arshkon (2024) entry share of native labels")
axes[1].set_ylabel("scraped (2026) entry share of native labels")
axes[1].set_title("Native entry share drift per title")
axes[1].set_xlim(0, 0.35)
axes[1].set_ylim(0, 0.35)
plt.tight_layout()
plt.savefig(FIG / "title_native_seniority_drift.png", dpi=120)
plt.close()
print(f"wrote {FIG / 'title_native_seniority_drift.png'}")

# YOE drift per title
yoe = pd.read_csv(TBL / "shared_titles_yoe.csv")
pm = yoe.pivot_table(index="title_normalized", columns="source", values="mean_llm").dropna()
plt.figure(figsize=(8, 6))
plt.scatter(pm["kaggle_arshkon"], pm["scraped"], s=40)
for t, row in pm.iterrows():
    plt.annotate(t[:20], (row["kaggle_arshkon"], row["scraped"]), fontsize=7, alpha=0.7)
lo = min(pm["kaggle_arshkon"].min(), pm["scraped"].min()) - 0.5
hi = max(pm["kaggle_arshkon"].max(), pm["scraped"].max()) + 0.5
plt.plot([lo, hi], [lo, hi], "r--", alpha=0.4)
plt.xlim(lo, hi)
plt.ylim(lo, hi)
plt.xlabel("arshkon (2024) mean yoe_min_years_llm")
plt.ylabel("scraped (2026) mean yoe_min_years_llm")
plt.title("Per-title YOE drift (top 20 shared titles)")
plt.tight_layout()
plt.savefig(FIG / "title_yoe_drift.png", dpi=120)
plt.close()
print(f"wrote {FIG / 'title_yoe_drift.png'}")
