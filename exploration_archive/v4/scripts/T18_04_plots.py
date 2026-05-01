"""T18 figures — parallel trends, AI gradient, boundary similarity."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "exploration/figures/T18"
os.makedirs(OUT_DIR, exist_ok=True)
TAB = "exploration/tables/T18"

# ---- 1. AI rate by group and period ----
grad = pd.read_csv(os.path.join(TAB, "ai_gradient_by_period.csv"))
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
colors = {"SWE": "#d62728", "SWE_adjacent": "#ff7f0e", "control": "#1f77b4"}
periods_ord = ["2024-01", "2024-04", "2026-03", "2026-04"]
for occ, sub in grad.groupby("occ"):
    sub2 = sub.set_index("period").reindex(periods_ord).reset_index()
    ax[0].plot(sub2.period, sub2.ai_rate * 100, "-o", label=occ, color=colors[occ], linewidth=2)
    ax[1].plot(sub2.period, sub2.agentic_rate * 100, "-o", label=occ, color=colors[occ], linewidth=2)
ax[0].set_title("AI keyword prevalence (any)")
ax[0].set_ylabel("% of postings")
ax[0].legend()
ax[0].grid(alpha=0.3)
ax[1].set_title("'agentic' token (~95% precision)")
ax[1].set_ylabel("% of postings")
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ai_gradient.png"), dpi=140, bbox_inches="tight")
plt.close()

# ---- 2. DiD bar chart ----
did = pd.read_csv(os.path.join(TAB, "did_baseline.csv"))
metrics = ["entry_share_best", "yoe_le2_share", "ai_rate", "agentic_rate", "scope_rate"]
did_sub = did[did.metric.isin(metrics)].copy()
x = np.arange(len(did_sub))
w = 0.25
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w, did_sub.SWE_delta * 100, w, label="SWE", color=colors["SWE"])
ax.bar(x, did_sub.SWE_adjacent_delta * 100, w, label="SWE_adjacent", color=colors["SWE_adjacent"])
ax.bar(x + w, did_sub.control_delta * 100, w, label="control", color=colors["control"])
ax.set_xticks(x)
ax.set_xticklabels(did_sub.metric, rotation=30, ha="right")
ax.set_ylabel("2024→2026 Δ (pp)")
ax.set_title("Cross-occupation delta by metric (baseline)")
ax.axhline(0, color="black", linewidth=0.5)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "did_bars.png"), dpi=140, bbox_inches="tight")
plt.close()

# ---- 3. Description length (raw vs LLM-text-controlled) ----
tr = pd.read_csv(os.path.join(TAB, "trends_baseline.csv"))
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for occ, sub in tr.groupby("occ_group"):
    sub = sub.sort_values("period2")
    ax[0].plot(sub.period2, sub.median_len, "-o", label=occ, color=colors[occ], linewidth=2, markersize=9)
    ax[1].plot(sub.period2, sub.median_len_llm_text, "-o", label=occ, color=colors[occ], linewidth=2, markersize=9)
ax[0].set_title("Median description length (raw)")
ax[0].set_ylabel("characters")
ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].set_title("Median description length (LLM-cleaned text only)")
ax[1].set_ylabel("characters")
ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "length_by_group.png"), dpi=140, bbox_inches="tight")
plt.close()

# ---- 4. Boundary similarity (SWE↔adjacent vs SWE↔control over periods) ----
bs = pd.read_csv(os.path.join(TAB, "boundary_similarity.csv"))
fig, ax = plt.subplots(figsize=(7, 5))
for pair, sub in bs.groupby("pair"):
    sub = sub.sort_values("period")
    ax.plot(sub.period, sub.cosine, "-o", label=pair, linewidth=2, markersize=10)
ax.set_ylabel("TF-IDF centroid cosine similarity")
ax.set_title("Cross-occupation textual similarity by period")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "boundary_similarity.png"), dpi=140, bbox_inches="tight")
plt.close()

print("Figures saved to", OUT_DIR)
