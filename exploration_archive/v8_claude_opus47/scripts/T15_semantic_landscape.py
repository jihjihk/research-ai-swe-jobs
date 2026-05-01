"""T15 — Semantic similarity landscape & convergence analysis (Agent I, Wave 2).

Steps:
  1. Stratified sample (up to 2,000 per period × seniority_3level) from the embedding index
  2. Build TF-IDF on cleaned text, reduce via TruncatedSVD to 100 components
  3. Compute centroid similarity matrices (embedding + TF-IDF) across period × seniority
  4. Trimmed centroid convergence (drop 10% most distant), with within-2024 calibration
  5. Within-group dispersion (mean pairwise cosine)
  6. UMAP, PCA, t-SNE visualizations
  7. Nearest-neighbor analysis (2026 junior → 2024 neighbors, both reps)
  8. Robustness table: each finding under embedding vs TF-IDF
  9. Outlier identification (top 50 postings most unlike their seniority peers)

Outputs: exploration/tables/T15/, exploration/figures/T15/.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

BASE = Path("/home/jihgaboot/gabor/job-research")
SHARED = BASE / "exploration" / "artifacts" / "shared"
FIG = BASE / "exploration" / "figures" / "T15"
TAB = BASE / "exploration" / "tables" / "T15"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

# ---------- Load ----------
print("Loading artifacts...")
emb = np.load(SHARED / "swe_embeddings.npy")
emb_idx = pd.read_parquet(SHARED / "swe_embedding_index.parquet")
text = pd.read_parquet(
    SHARED / "swe_cleaned_text.parquet",
    columns=[
        "uid",
        "description_cleaned",
        "source",
        "period",
        "seniority_final",
        "seniority_3level",
        "text_source",
        "is_aggregator",
        "yoe_extracted",
    ],
)
# Only text_source='llm' rows have embeddings
text = text[text.text_source == "llm"].copy()
print(f"emb: {emb.shape}, text(llm): {len(text):,}")

# Join embeddings with metadata
df = emb_idx.merge(text, on="uid", how="inner")
# ensure alignment
df = df.sort_values("row_idx").reset_index(drop=True)
assert (df.row_idx.values == np.arange(len(df))).all()
print(f"Aligned embedding/meta: {len(df):,}")

# period bucket
df["period_bucket"] = df.period.map(
    {"2024-04": "2024", "2024-01": "2024", "2026-03": "2026", "2026-04": "2026"}
)
df["J2"] = df.seniority_final.isin(["entry", "associate"])
df["J3"] = (df.yoe_extracted <= 2).fillna(False)
df["S1"] = df.seniority_final.isin(["mid-senior", "director"])
df["S4"] = (df.yoe_extracted >= 5).fillna(False)

# group key: period × seniority_3level  (+ arshkon/asaniczka sub-split for within-2024 calibration)
df["group"] = df["period_bucket"] + "_" + df["seniority_3level"]
# source-specific 2024 groups for within-2024 calibration
df["group_srcaware"] = np.where(
    df["period_bucket"] == "2024",
    df["source"].map({"kaggle_arshkon": "2024ar", "kaggle_asaniczka": "2024as"}) + "_" + df["seniority_3level"],
    df["group"],
)

# Report source × group
print("\nGroup counts (period x seniority_3level):")
print(df.group.value_counts())
print("\nSource-aware group counts:")
print(df.group_srcaware.value_counts())

# ---------- Step 1: Stratified sample ----------
print("\n[Step 1] Stratified sampling (up to 2,000 per period × seniority_3level)...")
MAX_PER_GROUP = 2000
sample_uids = []
for g, sub in df.groupby("group"):
    n = min(MAX_PER_GROUP, len(sub))
    picked = sub.sample(n=n, random_state=42).row_idx.tolist()
    sample_uids.extend(picked)
sample_uids.sort()
sample_mask = df.row_idx.isin(set(sample_uids)).values
samp = df.loc[sample_mask].reset_index(drop=True)
print(f"Sample size: {len(samp):,}")
print(samp.group.value_counts())

# Recover embedding rows for sample
sample_row_idx = samp.row_idx.values
X_emb = emb[sample_row_idx]  # N × 384 embedding subset
# L2-normalize for cosine-as-dot
X_emb_n = normalize(X_emb, axis=1)
print(f"Embedding matrix (sample): {X_emb_n.shape}")

# ---------- Step 1b: TF-IDF + SVD ----------
print("\n[Step 1b] Building TF-IDF + TruncatedSVD...")
texts = samp["description_cleaned"].tolist()
vec = TfidfVectorizer(
    max_features=20000,
    min_df=5,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True,
)
tfidf = vec.fit_transform(texts)
print(f"TF-IDF matrix: {tfidf.shape}")
svd = TruncatedSVD(n_components=100, random_state=42)
X_tfidf = svd.fit_transform(tfidf)
# normalize for cosine
X_tfidf_n = normalize(X_tfidf, axis=1)
print(f"TF-IDF SVD: {X_tfidf_n.shape}  explained_var_sum: {svd.explained_variance_ratio_.sum():.3f}")

# ---------- Step 2: Centroid similarity ----------
print("\n[Step 2] Centroid similarity matrices...")

def group_centroids(X_normed, group_labels, min_n=20):
    groups = sorted(set(group_labels))
    cents = {}
    for g in groups:
        mask = np.array([g_ == g for g_ in group_labels])
        if mask.sum() < min_n:
            continue
        cents[g] = normalize(X_normed[mask].mean(axis=0).reshape(1, -1), axis=1).ravel()
    return cents


def centroid_sim(cents):
    keys = list(cents.keys())
    M = np.array([cents[k] for k in keys])
    sim = M @ M.T
    return keys, sim


sample_groups = samp.group.tolist()
cents_emb = group_centroids(X_emb_n, sample_groups)
cents_tfidf = group_centroids(X_tfidf_n, sample_groups)
keys_e, S_e = centroid_sim(cents_emb)
keys_t, S_t = centroid_sim(cents_tfidf)
pd.DataFrame(S_e, index=keys_e, columns=keys_e).to_csv(TAB / "centroid_sim_embedding.csv")
pd.DataFrame(S_t, index=keys_t, columns=keys_t).to_csv(TAB / "centroid_sim_tfidf.csv")

print("Embedding centroid similarity:")
print(pd.DataFrame(S_e, index=keys_e, columns=keys_e).round(3).to_string())
print("\nTF-IDF SVD centroid similarity:")
print(pd.DataFrame(S_t, index=keys_t, columns=keys_t).round(3).to_string())

# Source-aware groups (include 2024ar/2024as split to enable within-2024 calibration)
cents_emb_src = group_centroids(X_emb_n, samp.group_srcaware.tolist())
cents_tfidf_src = group_centroids(X_tfidf_n, samp.group_srcaware.tolist())
keys_es, S_es = centroid_sim(cents_emb_src)
keys_ts, S_ts = centroid_sim(cents_tfidf_src)
pd.DataFrame(S_es, index=keys_es, columns=keys_es).to_csv(TAB / "centroid_sim_embedding_srcaware.csv")
pd.DataFrame(S_ts, index=keys_ts, columns=keys_ts).to_csv(TAB / "centroid_sim_tfidf_srcaware.csv")

# ---------- Step 3: Convergence — trimmed centroids + within-2024 calibration ----------
print("\n[Step 3] Convergence analysis (trimmed centroids)...")


def trimmed_centroid(X, fraction=0.10):
    """Remove the 10% most distant rows from the raw centroid, recompute."""
    c = X.mean(axis=0)
    # cosine distance = 1 - cosine sim; already normalized
    sims = X @ c
    thresh = np.quantile(sims, fraction)
    keep = sims > thresh
    if keep.sum() < 3:
        keep = np.ones_like(sims, dtype=bool)
    return normalize(X[keep].mean(axis=0).reshape(1, -1), axis=1).ravel(), int(keep.sum())


def trimmed_group_centroid(X_normed, group_labels, min_n=20):
    groups = sorted(set(group_labels))
    cents = {}
    ns = {}
    for g in groups:
        mask = np.array([g_ == g for g_ in group_labels])
        if mask.sum() < min_n:
            continue
        c, n_kept = trimmed_centroid(X_normed[mask])
        cents[g] = c
        ns[g] = n_kept
    return cents, ns


t_emb, ne = trimmed_group_centroid(X_emb_n, samp.group.tolist())
t_tfidf, nt = trimmed_group_centroid(X_tfidf_n, samp.group.tolist())
t_emb_src, _ = trimmed_group_centroid(X_emb_n, samp.group_srcaware.tolist())
t_tfidf_src, _ = trimmed_group_centroid(X_tfidf_n, samp.group_srcaware.tolist())


def pairwise_cos(cents, a, b):
    if a not in cents or b not in cents:
        return np.nan
    return float(cents[a] @ cents[b])


conv_rows = []
# Within-period senior/junior similarity (trimmed)
for period in ["2024", "2026"]:
    for (a, b, name) in [
        ("junior", "senior", "junior_vs_senior"),
        ("junior", "mid", "junior_vs_mid"),
        ("mid", "senior", "mid_vs_senior"),
    ]:
        ga = f"{period}_{a}"
        gb = f"{period}_{b}"
        conv_rows.append(
            {
                "representation": "embedding",
                "period": period,
                "pair": name,
                "group_a": ga,
                "group_b": gb,
                "sim_trimmed": pairwise_cos(t_emb, ga, gb),
                "sim_raw": pairwise_cos(cents_emb, ga, gb),
            }
        )
        conv_rows.append(
            {
                "representation": "tfidf_svd",
                "period": period,
                "pair": name,
                "group_a": ga,
                "group_b": gb,
                "sim_trimmed": pairwise_cos(t_tfidf, ga, gb),
                "sim_raw": pairwise_cos(cents_tfidf, ga, gb),
            }
        )
conv = pd.DataFrame(conv_rows)
conv.to_csv(TAB / "convergence_trimmed.csv", index=False)
print(conv.to_string(index=False))

# Within-2024 calibration: arshkon vs asaniczka on same seniority
cal_rows = []
for sen in ["junior", "mid", "senior", "unknown"]:
    ar = f"2024ar_{sen}"
    asn = f"2024as_{sen}"
    cal_rows.append(
        {
            "seniority": sen,
            "representation": "embedding",
            "sim_arshkon_vs_asaniczka": pairwise_cos(t_emb_src, ar, asn),
        }
    )
    cal_rows.append(
        {
            "seniority": sen,
            "representation": "tfidf_svd",
            "sim_arshkon_vs_asaniczka": pairwise_cos(t_tfidf_src, ar, asn),
        }
    )
cal = pd.DataFrame(cal_rows)
cal.to_csv(TAB / "within2024_calibration.csv", index=False)
print("\nWithin-2024 calibration (arshkon vs asaniczka, same seniority, trimmed):")
print(cal.to_string(index=False))

# ---------- Step 4: Within-group dispersion ----------
print("\n[Step 4] Within-group dispersion (mean pairwise cosine)...")


def dispersion(X_normed, group_labels):
    rows = []
    for g in sorted(set(group_labels)):
        mask = np.array([gx == g for gx in group_labels])
        if mask.sum() < 20:
            continue
        Xg = X_normed[mask]
        # centroid sim: 1 - 1/n sum_i (x_i . centroid)  → inverse = dispersion
        c = normalize(Xg.mean(axis=0).reshape(1, -1), axis=1).ravel()
        sims_c = Xg @ c
        # pairwise mean (sample if >500)
        n = len(Xg)
        if n > 500:
            idx = RNG.choice(n, 500, replace=False)
            Xs = Xg[idx]
        else:
            Xs = Xg
        pair = Xs @ Xs.T
        np.fill_diagonal(pair, np.nan)
        mean_pair = float(np.nanmean(pair))
        rows.append(
            {
                "group": g,
                "n": int(n),
                "mean_centroid_sim": float(sims_c.mean()),
                "mean_pairwise_cos": mean_pair,
                "std_centroid_sim": float(sims_c.std()),
            }
        )
    return pd.DataFrame(rows)


disp_emb = dispersion(X_emb_n, samp.group.tolist()).assign(representation="embedding")
disp_tfidf = dispersion(X_tfidf_n, samp.group.tolist()).assign(representation="tfidf_svd")
disp = pd.concat([disp_emb, disp_tfidf], ignore_index=True)
disp.to_csv(TAB / "dispersion.csv", index=False)
print("\nDispersion (embedding):")
print(disp_emb.to_string(index=False))
print("\nDispersion (tfidf_svd):")
print(disp_tfidf.to_string(index=False))

# ---------- Step 5: Visualizations ----------
print("\n[Step 5] Visualizations (UMAP, PCA, t-SNE)...")
# Subsample for UMAP if too large: we have ~12k which is fine for UMAP
import umap

print("  UMAP on embedding...")
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=25, min_dist=0.3, metric="cosine")
umap_emb = reducer.fit_transform(X_emb_n)
print("  UMAP on TF-IDF SVD...")
reducer_t = umap.UMAP(n_components=2, random_state=42, n_neighbors=25, min_dist=0.3, metric="cosine")
umap_tfidf = reducer_t.fit_transform(X_tfidf_n)

print("  PCA on embedding...")
pca_emb = PCA(n_components=2, random_state=42).fit_transform(X_emb_n)
print("  PCA on TF-IDF SVD...")
pca_tfidf = PCA(n_components=2, random_state=42).fit_transform(X_tfidf_n)

print("  t-SNE on embedding (subsample 4000)...")
sub_idx = RNG.choice(len(samp), min(4000, len(samp)), replace=False)
tsne_emb = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, metric="cosine").fit_transform(X_emb_n[sub_idx])
print("  t-SNE on TF-IDF SVD (subsample 4000)...")
tsne_tfidf = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, metric="cosine").fit_transform(X_tfidf_n[sub_idx])

# Save embeddings 2D
viz_df = samp.copy()
viz_df["umap_emb_x"] = umap_emb[:, 0]
viz_df["umap_emb_y"] = umap_emb[:, 1]
viz_df["umap_tfidf_x"] = umap_tfidf[:, 0]
viz_df["umap_tfidf_y"] = umap_tfidf[:, 1]
viz_df["pca_emb_x"] = pca_emb[:, 0]
viz_df["pca_emb_y"] = pca_emb[:, 1]
viz_df["pca_tfidf_x"] = pca_tfidf[:, 0]
viz_df["pca_tfidf_y"] = pca_tfidf[:, 1]
viz_df.to_parquet(TAB / "sample_with_2d.parquet", index=False)

# UMAP figure
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
color_map = {
    "2024_junior": "#2E86AB",
    "2024_mid": "#A23B72",
    "2024_senior": "#F18F01",
    "2024_unknown": "#6C757D",
    "2026_junior": "#5BC0EB",
    "2026_mid": "#E94F64",
    "2026_senior": "#FFA500",
    "2026_unknown": "#AAAAAA",
}

def scatter_groups(ax, x, y, df_mask, groups, title):
    for g in groups:
        mask = df_mask == g
        if mask.sum() == 0:
            continue
        c = color_map.get(g, "gray")
        ax.scatter(x[mask], y[mask], s=3, alpha=0.3, c=c, label=f"{g} (n={mask.sum()})")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("d1")
    ax.set_ylabel("d2")
    ax.legend(fontsize=7, markerscale=2, loc="best")


groups_order = ["2024_junior", "2024_mid", "2024_senior", "2024_unknown", "2026_junior", "2026_mid", "2026_senior", "2026_unknown"]
scatter_groups(axes[0, 0], umap_emb[:, 0], umap_emb[:, 1], samp.group.values, groups_order, "UMAP (embedding)")
scatter_groups(axes[0, 1], umap_tfidf[:, 0], umap_tfidf[:, 1], samp.group.values, groups_order, "UMAP (TF-IDF SVD)")
scatter_groups(axes[1, 0], pca_emb[:, 0], pca_emb[:, 1], samp.group.values, groups_order, "PCA (embedding)")
scatter_groups(axes[1, 1], pca_tfidf[:, 0], pca_tfidf[:, 1], samp.group.values, groups_order, "PCA (TF-IDF SVD)")

# Add centroid movement arrows in UMAP embedding panel
cent_2d = {}
for g in groups_order:
    mask = samp.group.values == g
    if mask.sum() < 20:
        continue
    cent_2d[g] = (umap_emb[mask, 0].mean(), umap_emb[mask, 1].mean())
for sen in ["junior", "mid", "senior"]:
    a = f"2024_{sen}"
    b = f"2026_{sen}"
    if a in cent_2d and b in cent_2d:
        axes[0, 0].annotate(
            "",
            xy=cent_2d[b],
            xytext=cent_2d[a],
            arrowprops=dict(arrowstyle="->", color=color_map.get(b), lw=2, alpha=0.7),
        )
        axes[0, 0].scatter(*cent_2d[a], s=120, marker="o", color=color_map[a], edgecolors="black", linewidths=1.5, zorder=5)
        axes[0, 0].scatter(*cent_2d[b], s=200, marker="*", color=color_map[b], edgecolors="black", linewidths=1.5, zorder=5)

plt.suptitle(f"Semantic landscape — period × seniority_3level (n={len(samp):,})", fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "umap_pca_landscape.png", dpi=150, bbox_inches="tight")
plt.close()

# t-SNE separately
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
sub_mask = samp.group.values[sub_idx]
scatter_groups(axes[0], tsne_emb[:, 0], tsne_emb[:, 1], sub_mask, groups_order, "t-SNE (embedding)")
scatter_groups(axes[1], tsne_tfidf[:, 0], tsne_tfidf[:, 1], sub_mask, groups_order, "t-SNE (TF-IDF SVD)")
plt.suptitle(f"t-SNE — period × seniority_3level (subsample n={len(sub_idx):,})", fontsize=12)
plt.tight_layout()
plt.savefig(FIG / "tsne_landscape.png", dpi=150, bbox_inches="tight")
plt.close()

# Density-contour UMAP for publication figure
fig, ax = plt.subplots(figsize=(12, 10))
from scipy.stats import gaussian_kde

for g in groups_order:
    mask = samp.group.values == g
    if mask.sum() < 50:
        continue
    c = color_map.get(g, "gray")
    x = umap_emb[mask, 0]
    y = umap_emb[mask, 1]
    # scatter lightly
    ax.scatter(x, y, s=2, alpha=0.1, c=c)
    # density contour
    try:
        kde = gaussian_kde(np.vstack([x, y]))
        xmin, xmax = umap_emb[:, 0].min(), umap_emb[:, 0].max()
        ymin, ymax = umap_emb[:, 1].min(), umap_emb[:, 1].max()
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.contour(xx, yy, zz, levels=3, colors=c, alpha=0.6, linewidths=1.2)
    except Exception:
        pass

for sen in ["junior", "mid", "senior"]:
    a = f"2024_{sen}"
    b = f"2026_{sen}"
    if a in cent_2d and b in cent_2d:
        ax.annotate(
            "",
            xy=cent_2d[b],
            xytext=cent_2d[a],
            arrowprops=dict(arrowstyle="->", color="black", lw=2, alpha=0.9),
        )
        ax.scatter(*cent_2d[a], s=160, marker="o", color=color_map[a], edgecolors="black", linewidths=2, zorder=5, label=f"{a} centroid")
        ax.scatter(*cent_2d[b], s=280, marker="*", color=color_map[b], edgecolors="black", linewidths=2, zorder=5, label=f"{b} centroid")

ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.set_title(f"SWE semantic landscape, UMAP on MiniLM embeddings (n={len(samp):,})\nDensity contours per period × seniority_3level; arrows trace centroid movement 2024→2026")
# custom legend
from matplotlib.patches import Patch

handles = [Patch(color=color_map[g], label=g) for g in groups_order if (samp.group.values == g).sum() >= 50]
ax.legend(handles=handles, loc="best", fontsize=9)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(FIG / "umap_density_contours.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------- Step 6: Nearest-neighbor analysis ----------
print("\n[Step 6] Nearest-neighbor analysis (2026 junior → 2024 neighbors)...")


def knn_analysis(X_normed, query_mask, ref_mask, ref_labels, k=5):
    Q = X_normed[query_mask]
    R = X_normed[ref_mask]
    ref_lab = np.asarray(ref_labels)[ref_mask]
    # cosine: Q @ R.T
    # chunked to avoid memory blowup if R very large
    sims = Q @ R.T  # (q × r)
    idx = np.argpartition(-sims, kth=min(k - 1, sims.shape[1] - 1), axis=1)[:, :k]
    # label distribution
    labels = ref_lab[idx]
    # per-query label array
    return labels


query_mask = (samp.period_bucket == "2026") & (samp.J2 == True)
ref_mask = samp.period_bucket == "2024"
ref_labels = samp.seniority_3level.values
print(f"  J2 2026 query: {query_mask.sum()}  2024 ref: {ref_mask.sum()}")

if query_mask.sum() > 0 and ref_mask.sum() > 0:
    labels_emb = knn_analysis(X_emb_n, query_mask.values, ref_mask.values, ref_labels, k=5)
    labels_tfidf = knn_analysis(X_tfidf_n, query_mask.values, ref_mask.values, ref_labels, k=5)
    # label distribution
    from collections import Counter

    flat_emb = Counter(labels_emb.flatten().tolist())
    flat_tfidf = Counter(labels_tfidf.flatten().tolist())
    total_emb = sum(flat_emb.values())
    total_tfidf = sum(flat_tfidf.values())
    # 2024 ref base rates (for excess)
    ref_label_dist = Counter(ref_labels[ref_mask.values].tolist())
    total_ref = sum(ref_label_dist.values())
    knn_rows = []
    for lab in ["junior", "mid", "senior", "unknown"]:
        base_rate = ref_label_dist.get(lab, 0) / total_ref if total_ref else 0
        knn_rows.append(
            {
                "label": lab,
                "base_rate_2024": base_rate,
                "emb_nn_rate": flat_emb.get(lab, 0) / total_emb if total_emb else 0,
                "tfidf_nn_rate": flat_tfidf.get(lab, 0) / total_tfidf if total_tfidf else 0,
            }
        )
    knn_df = pd.DataFrame(knn_rows)
    knn_df["emb_excess_over_base"] = knn_df.emb_nn_rate - knn_df.base_rate_2024
    knn_df["tfidf_excess_over_base"] = knn_df.tfidf_nn_rate - knn_df.base_rate_2024
    knn_df.to_csv(TAB / "knn_2026j2_to_2024.csv", index=False)
    print(knn_df.to_string(index=False))
else:
    print("  insufficient data")

# Also run knn for 2026 senior → 2024 neighbors (for comparison)
query_mask_s = (samp.period_bucket == "2026") & (samp.S1 == True)
if query_mask_s.sum() > 0 and ref_mask.sum() > 0:
    labels_emb_s = knn_analysis(X_emb_n, query_mask_s.values, ref_mask.values, ref_labels, k=5)
    flat_emb_s = Counter(labels_emb_s.flatten().tolist())
    total_emb_s = sum(flat_emb_s.values())
    knn_s_rows = []
    for lab in ["junior", "mid", "senior", "unknown"]:
        base_rate = ref_label_dist.get(lab, 0) / total_ref if total_ref else 0
        knn_s_rows.append(
            {
                "label": lab,
                "base_rate_2024": base_rate,
                "emb_nn_rate": flat_emb_s.get(lab, 0) / total_emb_s if total_emb_s else 0,
            }
        )
    pd.DataFrame(knn_s_rows).to_csv(TAB / "knn_2026s1_to_2024.csv", index=False)

# ---------- Step 7: Robustness table ----------
print("\n[Step 7] Robustness table (embedding vs TF-IDF)...")

# Extract key findings
robust_rows = []

# 1. Junior-senior within-2026 similarity
e26_js = pairwise_cos(t_emb, "2026_junior", "2026_senior")
t26_js = pairwise_cos(t_tfidf, "2026_junior", "2026_senior")
e24_js = pairwise_cos(t_emb, "2024_junior", "2024_senior")
t24_js = pairwise_cos(t_tfidf, "2024_junior", "2024_senior")
robust_rows.append(
    {
        "finding": "junior-senior centroid similarity 2024 vs 2026",
        "emb_2024": e24_js,
        "emb_2026": e26_js,
        "emb_direction": "converging" if e26_js > e24_js else "diverging",
        "tfidf_2024": t24_js,
        "tfidf_2026": t26_js,
        "tfidf_direction": "converging" if t26_js > t24_js else "diverging",
        "agrees": ("converging" if e26_js > e24_js else "diverging") == ("converging" if t26_js > t24_js else "diverging"),
    }
)

# 2. Within-2024 junior vs asaniczka same seniority
e_within = cal[(cal.representation == "embedding") & (cal.seniority == "junior")].sim_arshkon_vs_asaniczka.iloc[0] if not cal.empty else np.nan
t_within = cal[(cal.representation == "tfidf_svd") & (cal.seniority == "junior")].sim_arshkon_vs_asaniczka.iloc[0] if not cal.empty else np.nan
e_mid_within = cal[(cal.representation == "embedding") & (cal.seniority == "mid")].sim_arshkon_vs_asaniczka.iloc[0] if not cal.empty else np.nan
t_mid_within = cal[(cal.representation == "tfidf_svd") & (cal.seniority == "mid")].sim_arshkon_vs_asaniczka.iloc[0] if not cal.empty else np.nan

# 3. Dispersion 2024 vs 2026 (senior as reference)
disp_senior_e_2024 = disp_emb[disp_emb.group == "2024_senior"].mean_pairwise_cos.iloc[0] if not disp_emb[disp_emb.group == "2024_senior"].empty else np.nan
disp_senior_e_2026 = disp_emb[disp_emb.group == "2026_senior"].mean_pairwise_cos.iloc[0] if not disp_emb[disp_emb.group == "2026_senior"].empty else np.nan
disp_senior_t_2024 = disp_tfidf[disp_tfidf.group == "2024_senior"].mean_pairwise_cos.iloc[0] if not disp_tfidf[disp_tfidf.group == "2024_senior"].empty else np.nan
disp_senior_t_2026 = disp_tfidf[disp_tfidf.group == "2026_senior"].mean_pairwise_cos.iloc[0] if not disp_tfidf[disp_tfidf.group == "2026_senior"].empty else np.nan
robust_rows.append(
    {
        "finding": "senior group dispersion 2024 vs 2026 (higher pairwise cos = more homogeneous)",
        "emb_2024": disp_senior_e_2024,
        "emb_2026": disp_senior_e_2026,
        "emb_direction": "more_homogeneous" if disp_senior_e_2026 > disp_senior_e_2024 else "more_diverse",
        "tfidf_2024": disp_senior_t_2024,
        "tfidf_2026": disp_senior_t_2026,
        "tfidf_direction": "more_homogeneous" if disp_senior_t_2026 > disp_senior_t_2024 else "more_diverse",
        "agrees": ("more_homogeneous" if disp_senior_e_2026 > disp_senior_e_2024 else "more_diverse") == ("more_homogeneous" if disp_senior_t_2026 > disp_senior_t_2024 else "more_diverse"),
    }
)

disp_junior_e_2024 = disp_emb[disp_emb.group == "2024_junior"].mean_pairwise_cos.iloc[0] if not disp_emb[disp_emb.group == "2024_junior"].empty else np.nan
disp_junior_e_2026 = disp_emb[disp_emb.group == "2026_junior"].mean_pairwise_cos.iloc[0] if not disp_emb[disp_emb.group == "2026_junior"].empty else np.nan
disp_junior_t_2024 = disp_tfidf[disp_tfidf.group == "2024_junior"].mean_pairwise_cos.iloc[0] if not disp_tfidf[disp_tfidf.group == "2024_junior"].empty else np.nan
disp_junior_t_2026 = disp_tfidf[disp_tfidf.group == "2026_junior"].mean_pairwise_cos.iloc[0] if not disp_tfidf[disp_tfidf.group == "2026_junior"].empty else np.nan
robust_rows.append(
    {
        "finding": "junior group dispersion 2024 vs 2026",
        "emb_2024": disp_junior_e_2024,
        "emb_2026": disp_junior_e_2026,
        "emb_direction": "more_homogeneous" if disp_junior_e_2026 > disp_junior_e_2024 else "more_diverse",
        "tfidf_2024": disp_junior_t_2024,
        "tfidf_2026": disp_junior_t_2026,
        "tfidf_direction": "more_homogeneous" if disp_junior_t_2026 > disp_junior_t_2024 else "more_diverse",
        "agrees": ("more_homogeneous" if disp_junior_e_2026 > disp_junior_e_2024 else "more_diverse") == ("more_homogeneous" if disp_junior_t_2026 > disp_junior_t_2024 else "more_diverse"),
    }
)

# 4. Within-2024 calibration check on junior (arshkon vs asaniczka): is it larger than arshkon-vs-scraped on same seniority?
# For arshkon vs scraped we need those specific trimmed centroids
ar_j = t_emb_src.get("2024ar_junior")
scrap_j = t_emb.get("2026_junior")
ar_s = t_emb_src.get("2024ar_senior")
scrap_s = t_emb.get("2026_senior")
as_j = t_emb_src.get("2024as_junior")
as_s = t_emb_src.get("2024as_senior")

ar_scraped_sim_j = float(ar_j @ scrap_j) if (ar_j is not None and scrap_j is not None) else np.nan
ar_as_sim_j = float(ar_j @ as_j) if (ar_j is not None and as_j is not None) else np.nan
ar_scraped_sim_s = float(ar_s @ scrap_s) if (ar_s is not None and scrap_s is not None) else np.nan
ar_as_sim_s = float(ar_s @ as_s) if (ar_s is not None and as_s is not None) else np.nan
print(f"\nWithin-2024 calibration (embedding):")
print(f"  arshkon→asaniczka junior similarity: {ar_as_sim_j:.3f}")
print(f"  arshkon→scraped junior similarity:   {ar_scraped_sim_j:.3f}")
print(f"  arshkon→asaniczka senior similarity: {ar_as_sim_s:.3f}")
print(f"  arshkon→scraped senior similarity:   {ar_scraped_sim_s:.3f}")

pd.DataFrame(robust_rows).to_csv(TAB / "robustness_table.csv", index=False)
print("\nRobustness table:")
print(pd.DataFrame(robust_rows).to_string(index=False))

# ---------- Step 8: Outlier identification ----------
print("\n[Step 8] Outlier identification...")
# Per seniority_3level, find top 50 most distant from group centroid
outliers = []
for g in sorted(set(samp.group.tolist())):
    mask = samp.group.values == g
    if mask.sum() < 100:
        continue
    Xg = X_emb_n[mask]
    c = normalize(Xg.mean(axis=0).reshape(1, -1), axis=1).ravel()
    sims = Xg @ c
    # 50 most distant
    order = np.argsort(sims)[:50]
    sub = samp.loc[mask].iloc[order]
    sub = sub.assign(distance=1 - sims[order], group=g)
    # Add preview of first 120 chars of description
    sub = sub.assign(text_preview=sub.description_cleaned.str[:150])
    outliers.append(sub[["uid", "group", "seniority_final", "source", "period", "distance", "text_preview"]])
outliers_df = pd.concat(outliers, ignore_index=True)
outliers_df.to_csv(TAB / "outliers.csv", index=False)
print(f"  {len(outliers_df)} outliers saved (top 50 per group × 8 groups)")

# Heatmap of centroid similarity (embedding, full + src-aware)
fig, axes = plt.subplots(1, 2, figsize=(22, 9))
# Reorder keys: 2024 first, then 2026
def sort_key(k):
    prefix_order = {"2024": 0, "2024ar": 0, "2024as": 1, "2026": 2}
    for p in ["2024ar", "2024as", "2024", "2026"]:
        if k.startswith(p):
            base = prefix_order[p]
            return (base, k)
    return (9, k)

idx_e = sorted(keys_e, key=sort_key)
S_e_reindex = pd.DataFrame(S_e, index=keys_e, columns=keys_e).loc[idx_e, idx_e].values
im1 = axes[0].imshow(S_e_reindex, aspect="auto", cmap="coolwarm", vmin=0.5, vmax=1.0)
axes[0].set_xticks(range(len(idx_e)))
axes[0].set_xticklabels(idx_e, rotation=45, ha="right", fontsize=8)
axes[0].set_yticks(range(len(idx_e)))
axes[0].set_yticklabels(idx_e, fontsize=8)
for i in range(len(idx_e)):
    for j in range(len(idx_e)):
        axes[0].text(j, i, f"{S_e_reindex[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")
axes[0].set_title("Embedding centroid similarity (period × seniority_3level)")
plt.colorbar(im1, ax=axes[0], fraction=0.03)

idx_t = sorted(keys_t, key=sort_key)
S_t_reindex = pd.DataFrame(S_t, index=keys_t, columns=keys_t).loc[idx_t, idx_t].values
im2 = axes[1].imshow(S_t_reindex, aspect="auto", cmap="coolwarm", vmin=0.5, vmax=1.0)
axes[1].set_xticks(range(len(idx_t)))
axes[1].set_xticklabels(idx_t, rotation=45, ha="right", fontsize=8)
axes[1].set_yticks(range(len(idx_t)))
axes[1].set_yticklabels(idx_t, fontsize=8)
for i in range(len(idx_t)):
    for j in range(len(idx_t)):
        axes[1].text(j, i, f"{S_t_reindex[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")
axes[1].set_title("TF-IDF SVD centroid similarity (period × seniority_3level)")
plt.colorbar(im2, ax=axes[1], fraction=0.03)
plt.tight_layout()
plt.savefig(FIG / "centroid_similarity_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# Summary JSON
summary = {
    "n_sample": int(len(samp)),
    "n_embedding_dim": int(X_emb_n.shape[1]),
    "n_tfidf_dim": int(X_tfidf_n.shape[1]),
    "tfidf_explained_var": float(svd.explained_variance_ratio_.sum()),
    "group_counts": samp.group.value_counts().to_dict(),
    "centroid_sim_2024_junior_senior_embedding": float(e24_js) if not np.isnan(e24_js) else None,
    "centroid_sim_2024_junior_senior_tfidf": float(t24_js) if not np.isnan(t24_js) else None,
    "centroid_sim_2026_junior_senior_embedding": float(e26_js) if not np.isnan(e26_js) else None,
    "centroid_sim_2026_junior_senior_tfidf": float(t26_js) if not np.isnan(t26_js) else None,
    "within2024_arshkon_asaniczka_junior_embedding": float(ar_as_sim_j) if not np.isnan(ar_as_sim_j) else None,
    "within2024_arshkon_asaniczka_senior_embedding": float(ar_as_sim_s) if not np.isnan(ar_as_sim_s) else None,
    "arshkon_vs_scraped_junior_embedding": float(ar_scraped_sim_j) if not np.isnan(ar_scraped_sim_j) else None,
    "arshkon_vs_scraped_senior_embedding": float(ar_scraped_sim_s) if not np.isnan(ar_scraped_sim_s) else None,
}
(TAB / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("\nDone. Outputs:")
for p in sorted(TAB.glob("*")):
    print("  ", p.relative_to(BASE))
for p in sorted(FIG.glob("*")):
    print("  ", p.relative_to(BASE))
