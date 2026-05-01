#!/usr/bin/env python3
"""
T09 step 5: Visualization — 2D UMAP and PCA embeddings colored by
(a) best-method clusters, (b) period, (c) seniority. Also a matplotlib
bar chart of top terms per BERTopic topic and an archetype-by-period heatmap.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from umap import UMAP
from sklearn.decomposition import PCA

OUT_FIG = "exploration/figures/T09"
os.makedirs(OUT_FIG, exist_ok=True)

# Load
sample = pd.read_parquet("exploration/tables/T09/sample.parquet")
bt = pd.read_parquet("exploration/tables/T09/bertopic_topics.parquet")
terms_bt = pd.read_csv("exploration/tables/T09/bertopic_topic_terms.csv")
df = sample.merge(bt, on="uid").reset_index(drop=True)

emb_index = pd.read_parquet("exploration/artifacts/shared/swe_embedding_index.parquet")
emb_all = np.load("exploration/artifacts/shared/swe_embeddings.npy")
uid_to_row = {u: i for i, u in enumerate(emb_index["uid"].tolist())}
idx = np.array([uid_to_row[u] for u in df["uid"]], dtype=np.int64)
emb = emb_all[idx]

# ---------------------------------------------------------------------------
# 2D projections
# ---------------------------------------------------------------------------
print("UMAP 2D...")
umap2 = UMAP(
    n_neighbors=30, n_components=2, min_dist=0.1, metric="cosine", random_state=42
).fit_transform(emb)
print("PCA 2D...")
pca2 = PCA(n_components=2, random_state=42).fit_transform(emb)


def plot_scatter(coords, labels, title, legend_title, fname, cmap_name="tab20",
                 max_classes_in_legend=15):
    fig, ax = plt.subplots(figsize=(7, 6))
    unique = pd.Series(labels).dropna().unique().tolist()
    # sort, with noise/-1 last
    def sort_key(x):
        try:
            v = int(x)
            return (1 if v == -1 else 0, 0, v)
        except Exception:
            return (0, 1, str(x))
    unique = sorted(unique, key=sort_key)
    cmap = cm.get_cmap(cmap_name, max(len(unique), 3))
    for i, u in enumerate(unique):
        mask = labels == u
        if u == -1 or str(u) == "-1":
            color = (0.7, 0.7, 0.7, 0.35)
            label = "noise"
        else:
            color = cmap(i)
            label = str(u)
        ax.scatter(coords[mask, 0], coords[mask, 1], s=3, c=[color], label=label, alpha=0.55, linewidths=0)
    if len(unique) <= max_classes_in_legend:
        ax.legend(
            title=legend_title,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7,
            markerscale=2,
            frameon=False,
        )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {fname}")


cluster_labels = df["bertopic_topic_mts30"].values
period_labels = df["period_bucket"].values
sen_labels = df["seniority_best_available"].fillna("unknown").values

plot_scatter(
    umap2, cluster_labels, "UMAP 2D — BERTopic clusters", "topic",
    f"{OUT_FIG}/umap_clusters.png", "tab20", max_classes_in_legend=25,
)
plot_scatter(
    umap2, period_labels, "UMAP 2D — Period", "period",
    f"{OUT_FIG}/umap_period.png", "tab10",
)
plot_scatter(
    umap2, sen_labels, "UMAP 2D — Seniority (combined best-available)", "seniority",
    f"{OUT_FIG}/umap_seniority.png", "tab10",
)
plot_scatter(
    pca2, cluster_labels, "PCA 2D — BERTopic clusters", "topic",
    f"{OUT_FIG}/pca_clusters.png", "tab20", max_classes_in_legend=25,
)
plot_scatter(
    pca2, period_labels, "PCA 2D — Period", "period",
    f"{OUT_FIG}/pca_period.png", "tab10",
)
plot_scatter(
    pca2, sen_labels, "PCA 2D — Seniority (combined best-available)", "seniority",
    f"{OUT_FIG}/pca_seniority.png", "tab10",
)

# ---------------------------------------------------------------------------
# BERTopic top-terms bar chart (replacement for visualize_barchart)
# ---------------------------------------------------------------------------
top_topics = (
    pd.read_csv("exploration/tables/T09/bertopic_topic_info.csv")
    .query("Topic != -1")
    .sort_values("Count", ascending=False)
    .head(16)
)
fig, axes = plt.subplots(4, 4, figsize=(14, 12))
for ax, (_, row) in zip(axes.ravel(), top_topics.iterrows()):
    tid = int(row.Topic)
    sub = (
        terms_bt[terms_bt.topic == tid]
        .sort_values("rank")
        .head(8)[["term", "weight"]]
        .iloc[::-1]
    )
    ax.barh(sub["term"], sub["weight"], color="steelblue")
    ax.set_title(f"T{tid:02d} (n={int(row.Count)})", fontsize=9)
    ax.tick_params(labelsize=7)
plt.tight_layout()
plt.savefig(f"{OUT_FIG}/bertopic_topterms_grid.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  wrote {OUT_FIG}/bertopic_topterms_grid.png")

# ---------------------------------------------------------------------------
# Archetype-by-period heatmap (share of each period)
# ---------------------------------------------------------------------------
temp = pd.read_csv("exploration/tables/T09/archetype_temporal_dynamics.csv")
# put noise at the end and sort positive topics by 2024 share descending
noise_row = temp[temp["archetype"] == -1]
pos = temp[temp["archetype"] != -1].sort_values("delta_2024_to_2026", ascending=False)
ordered = pd.concat([pos, noise_row], ignore_index=True)
mat = ordered[["2024", "2026-03", "2026-04"]].values
labels = [
    f"T{int(a):02d}  {n[:40]}" if a != -1 else "noise/outliers"
    for a, n in zip(ordered["archetype"], ordered["archetype_name"].fillna(""))
]
fig, ax = plt.subplots(figsize=(7, 8))
im = ax.imshow(mat, aspect="auto", cmap="viridis")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=7)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["2024", "2026-03", "2026-04"])
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(
            j,
            i,
            f"{mat[i, j]*100:.1f}%",
            ha="center",
            va="center",
            color="white" if mat[i, j] < mat.max() * 0.6 else "black",
            fontsize=6,
        )
fig.colorbar(im, ax=ax, label="within-period share")
ax.set_title("Archetype share by period (BERTopic, mts=30)")
plt.tight_layout()
plt.savefig(f"{OUT_FIG}/archetype_period_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  wrote {OUT_FIG}/archetype_period_heatmap.png")

# ---------------------------------------------------------------------------
# Sensitivity comparison plot — baseline vs no_aggregator vs llm_text_only
# ---------------------------------------------------------------------------
sens = pd.read_csv("exploration/tables/T09/archetype_sensitivities.csv")
# Aggregate over periods: total share per variant
agg = sens.groupby(["variant", "archetype"])["n"].sum().reset_index()
tot = sens.groupby("variant")["n"].sum().reset_index().rename(columns={"n": "tot"})
agg = agg.merge(tot, on="variant")
agg["share"] = agg["n"] / agg["tot"]
wide = agg.pivot(index="archetype", columns="variant", values="share").fillna(0)
wide = wide.sort_values("baseline", ascending=True)
fig, ax = plt.subplots(figsize=(8, 8))
y = np.arange(len(wide))
ax.barh(y - 0.25, wide["baseline"], 0.25, label="baseline", color="steelblue")
ax.barh(y, wide["no_aggregator"], 0.25, label="no aggregator", color="darkorange")
ax.barh(y + 0.25, wide["llm_text_only"], 0.25, label="llm text only", color="seagreen")
ax.set_yticks(y)
ax.set_yticklabels([f"T{int(a):02d}" if a != -1 else "noise" for a in wide.index], fontsize=7)
ax.set_xlabel("overall share")
ax.set_title("Archetype share — sensitivity to aggregators and text source")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_FIG}/archetype_sensitivities.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  wrote {OUT_FIG}/archetype_sensitivities.png")

print("\nDONE visualization stage")
