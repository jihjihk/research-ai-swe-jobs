"""T15: Semantic similarity landscape & convergence analysis.

Uses shared MiniLM-L6-v2 embeddings (L2-normalized, 34,099 rows).
Filters metadata to text_source='llm' rows before sampling.

Outputs under exploration/figures/T15/ and exploration/tables/T15/.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

import umap

REPO = Path("/home/jihgaboot/gabor/job-research")
SHARED = REPO / "exploration" / "artifacts" / "shared"
TBL = REPO / "exploration" / "tables" / "T15"
FIG = REPO / "exploration" / "figures" / "T15"
TBL.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
CACHE = REPO / "exploration" / "artifacts" / "t15_tfidf_svd.npz"


def load_data():
    print("[load] embeddings + metadata")
    idx = pq.read_table(SHARED / "swe_embedding_index.parquet").to_pandas()
    emb = np.load(SHARED / "swe_embeddings.npy")
    meta = pq.read_table(
        SHARED / "swe_cleaned_text.parquet",
        columns=[
            "uid",
            "text_source",
            "period",
            "seniority_final",
            "seniority_3level",
            "is_aggregator",
            "company_name_canonical",
            "source",
            "swe_classification_tier",
            "yoe_extracted",
        ],
    ).to_pandas()
    idx = idx.merge(meta, on="uid", how="left")
    idx = idx[idx["text_source"] == "llm"].reset_index(drop=True)
    # Must keep embeddings aligned — use row_idx to pull
    emb_aligned = emb[idx["row_idx"].values]
    idx["period_bucket"] = np.where(idx["period"].str.startswith("2024"), "2024", "2026")
    return idx, emb_aligned


def stratified_sample(idx: pd.DataFrame, emb: np.ndarray, per_group: int = 2000):
    print("[sample] stratified by period x seniority_3level")
    groups = []
    rows_keep = []
    for per in ["2024", "2026"]:
        for sen in ["junior", "mid", "senior", "unknown"]:
            sub = idx[(idx["period_bucket"] == per) & (idx["seniority_3level"] == sen)]
            n = len(sub)
            if n == 0:
                continue
            take = min(per_group, n)
            pick = sub.sample(n=take, random_state=42).index.tolist()
            rows_keep.extend(pick)
            groups.append(
                {
                    "period": per,
                    "seniority_3level": sen,
                    "n_total": n,
                    "n_sampled": take,
                }
            )
    sampled = idx.loc[rows_keep].reset_index(drop=True)
    sampled["emb_row"] = rows_keep
    sampled_emb = emb[rows_keep]
    pd.DataFrame(groups).to_csv(TBL / "sample_composition.csv", index=False)
    print(f"[sample] total sampled = {len(sampled)}")
    return sampled, sampled_emb


def load_tfidf_for_sampled(sampled: pd.DataFrame) -> np.ndarray:
    """Build TF-IDF SVD for sampled uids. Cache on disk."""
    if CACHE.exists():
        data = np.load(CACHE, allow_pickle=True)
        cached_uids = data["uids"]
        if len(cached_uids) == len(sampled) and np.array_equal(
            cached_uids, sampled["uid"].values
        ):
            print("[tfidf] cache hit")
            return data["svd"]

    print("[tfidf] building from swe_cleaned_text + DuckDB for description_core_llm")
    # Use cleaned tokens from raw artifact
    uids = sampled["uid"].tolist()
    # Pull description_core_llm from unified.parquet
    conn = duckdb.connect()
    query = """
    SELECT uid, description_core_llm
    FROM read_parquet(?)
    WHERE uid IN (SELECT unnest(?))
    """
    texts = conn.execute(query, [str(REPO / "data" / "unified.parquet"), uids]).df()
    conn.close()
    texts = sampled[["uid"]].merge(texts, on="uid", how="left")
    texts["description_core_llm"] = texts["description_core_llm"].fillna("")
    corpus = texts["description_core_llm"].tolist()

    print("[tfidf] fitting")
    vec = TfidfVectorizer(
        max_features=20000,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2),
        stop_words="english",
    )
    tfidf = vec.fit_transform(corpus)
    print(f"[tfidf] shape={tfidf.shape}")
    svd = TruncatedSVD(n_components=100, random_state=42)
    svd_mat = svd.fit_transform(tfidf)
    # L2 normalize for cosine
    norms = np.linalg.norm(svd_mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    svd_mat = svd_mat / norms
    svd_mat = svd_mat.astype(np.float32)
    np.savez(CACHE, svd=svd_mat, uids=sampled["uid"].values)
    return svd_mat


def trimmed_centroid(X: np.ndarray, trim: float = 0.1) -> np.ndarray:
    """Compute centroid after dropping `trim` fraction most-distant points from initial mean."""
    if len(X) == 0:
        return np.zeros(X.shape[1] if X.ndim == 2 else 0)
    mu = X.mean(axis=0)
    mu /= max(np.linalg.norm(mu), 1e-9)
    sims = X @ mu
    k = int(len(X) * (1 - trim))
    keep = np.argsort(sims)[-k:]
    mu2 = X[keep].mean(axis=0)
    mu2 /= max(np.linalg.norm(mu2), 1e-9)
    return mu2


def structural_map(sampled: pd.DataFrame, emb: np.ndarray, tfidf_svd: np.ndarray):
    print("[struct] centroid similarity matrix (period x seniority_3level)")
    sampled["group"] = sampled["period_bucket"] + "|" + sampled["seniority_3level"]
    groups = sorted(sampled["group"].unique())

    def _centroid_matrix(X: np.ndarray) -> pd.DataFrame:
        mat = np.zeros((len(groups), X.shape[1]), dtype=np.float32)
        for i, g in enumerate(groups):
            rows = sampled[sampled["group"] == g].index.values
            mat[i] = trimmed_centroid(X[rows], trim=0.1)
        sim = cosine_similarity(mat)
        return pd.DataFrame(sim, index=groups, columns=groups)

    emb_sim = _centroid_matrix(emb)
    tfidf_sim = _centroid_matrix(tfidf_svd)
    emb_sim.to_csv(TBL / "centroid_similarity_embedding.csv")
    tfidf_sim.to_csv(TBL / "centroid_similarity_tfidf.csv")

    # heatmap figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(emb_sim, annot=True, fmt=".3f", cmap="viridis", ax=axes[0], cbar_kws={"label": "cosine"})
    axes[0].set_title("Embedding (trimmed) centroid similarity")
    sns.heatmap(tfidf_sim, annot=True, fmt=".3f", cmap="viridis", ax=axes[1], cbar_kws={"label": "cosine"})
    axes[1].set_title("TF-IDF SVD (trimmed) centroid similarity")
    plt.tight_layout()
    plt.savefig(FIG / "centroid_similarity_heatmap.png", dpi=150)
    plt.close()

    # dominant variation diagnostic: average cross-group similarity split by shared dimension
    def _avg_split(sim_df):
        within_period = []
        across_period = []
        within_sen = []
        across_sen = []
        for a in groups:
            for b in groups:
                if a == b:
                    continue
                pa, sa = a.split("|")
                pb, sb = b.split("|")
                v = sim_df.loc[a, b]
                if pa == pb:
                    within_period.append(v)
                else:
                    across_period.append(v)
                if sa == sb:
                    within_sen.append(v)
                else:
                    across_sen.append(v)
        return {
            "within_period_mean": float(np.mean(within_period)),
            "across_period_mean": float(np.mean(across_period)),
            "within_seniority_mean": float(np.mean(within_sen)),
            "across_seniority_mean": float(np.mean(across_sen)),
        }

    split_emb = _avg_split(emb_sim)
    split_tfidf = _avg_split(tfidf_sim)
    with (TBL / "dominant_variation_diagnostic.txt").open("w") as fh:
        fh.write("Embedding similarities (higher = more similar):\n")
        for k, v in split_emb.items():
            fh.write(f"  {k}: {v:.4f}\n")
        fh.write(f"  period gap (within - across) = {split_emb['within_period_mean'] - split_emb['across_period_mean']:.4f}\n")
        fh.write(f"  seniority gap (within - across) = {split_emb['within_seniority_mean'] - split_emb['across_seniority_mean']:.4f}\n")
        fh.write("\nTF-IDF SVD similarities:\n")
        for k, v in split_tfidf.items():
            fh.write(f"  {k}: {v:.4f}\n")
        fh.write(f"  period gap (within - across) = {split_tfidf['within_period_mean'] - split_tfidf['across_period_mean']:.4f}\n")
        fh.write(f"  seniority gap (within - across) = {split_tfidf['within_seniority_mean'] - split_tfidf['across_seniority_mean']:.4f}\n")

    return emb_sim, tfidf_sim, split_emb, split_tfidf


def convergence_analysis(sampled: pd.DataFrame, emb: np.ndarray, tfidf_svd: np.ndarray):
    """Convergence between seniority levels within each period, calibrated by within-2024."""
    print("[conv] convergence with within-2024 calibration")

    def _sim_between(g1: str, g2: str, X: np.ndarray) -> float:
        r1 = sampled[sampled["group"] == g1].index.values
        r2 = sampled[sampled["group"] == g2].index.values
        if len(r1) == 0 or len(r2) == 0:
            return float("nan")
        c1 = trimmed_centroid(X[r1])
        c2 = trimmed_centroid(X[r2])
        return float(np.dot(c1, c2))

    # Primary: junior-senior similarity, each period
    sampled["group"] = sampled["period_bucket"] + "|" + sampled["seniority_3level"]

    emb_js_2024 = _sim_between("2024|junior", "2024|senior", emb)
    emb_js_2026 = _sim_between("2026|junior", "2026|senior", emb)
    tfidf_js_2024 = _sim_between("2024|junior", "2024|senior", tfidf_svd)
    tfidf_js_2026 = _sim_between("2026|junior", "2026|senior", tfidf_svd)

    # Within-2024 calibration: arshkon vs asaniczka SWE (pooled on seniority)
    sampled["src_group"] = sampled["source"] + "|" + sampled["seniority_3level"]

    def _src_sim(src_a, src_b, sen, X):
        ra = sampled[(sampled["source"] == src_a) & (sampled["seniority_3level"] == sen)].index.values
        rb = sampled[(sampled["source"] == src_b) & (sampled["seniority_3level"] == sen)].index.values
        if len(ra) < 20 or len(rb) < 20:
            return float("nan")
        ca = trimmed_centroid(X[ra])
        cb = trimmed_centroid(X[rb])
        return float(np.dot(ca, cb))

    # Within-2024 arshkon vs asaniczka, on junior-senior distance
    # Compute the junior-senior gap within arshkon vs the junior-senior gap within asaniczka,
    # and compute the shift between arshkon and asaniczka on the same gap.
    def _js_gap_within_source(source, X):
        rj = sampled[(sampled["source"] == source) & (sampled["seniority_3level"] == "junior")].index.values
        rs = sampled[(sampled["source"] == source) & (sampled["seniority_3level"] == "senior")].index.values
        if len(rj) < 20 or len(rs) < 20:
            return float("nan")
        cj = trimmed_centroid(X[rj])
        cs = trimmed_centroid(X[rs])
        return float(np.dot(cj, cs))

    emb_js_arshkon = _js_gap_within_source("kaggle_arshkon", emb)
    emb_js_asaniczka = _js_gap_within_source("kaggle_asaniczka", emb)
    emb_js_scraped = _js_gap_within_source("scraped", emb)

    tfidf_js_arshkon = _js_gap_within_source("kaggle_arshkon", tfidf_svd)
    tfidf_js_asaniczka = _js_gap_within_source("kaggle_asaniczka", tfidf_svd)
    tfidf_js_scraped = _js_gap_within_source("scraped", tfidf_svd)

    # within-2024 baseline noise: |arshkon js_gap - asaniczka js_gap|
    # cross-period signal: |2024 (pooled) js_gap - scraped js_gap|
    within_24_emb_noise = abs(emb_js_arshkon - emb_js_asaniczka) if not np.isnan(emb_js_arshkon) else float("nan")
    cross_emb_signal = abs(emb_js_2024 - emb_js_2026)

    within_24_tfidf_noise = abs(tfidf_js_arshkon - tfidf_js_asaniczka) if not np.isnan(tfidf_js_arshkon) else float("nan")
    cross_tfidf_signal = abs(tfidf_js_2024 - tfidf_js_2026)

    rows = [
        {
            "metric": "junior_senior_centroid_similarity_embedding",
            "2024_pooled": emb_js_2024,
            "2026_scraped": emb_js_2026,
            "cross_period_shift": emb_js_2026 - emb_js_2024,
            "arshkon_js_sim": emb_js_arshkon,
            "asaniczka_js_sim": emb_js_asaniczka,
            "scraped_js_sim": emb_js_scraped,
            "within_2024_shift": within_24_emb_noise,
            "cross_abs": cross_emb_signal,
            "snr": (cross_emb_signal / within_24_emb_noise) if within_24_emb_noise > 1e-9 else float("nan"),
        },
        {
            "metric": "junior_senior_centroid_similarity_tfidf",
            "2024_pooled": tfidf_js_2024,
            "2026_scraped": tfidf_js_2026,
            "cross_period_shift": tfidf_js_2026 - tfidf_js_2024,
            "arshkon_js_sim": tfidf_js_arshkon,
            "asaniczka_js_sim": tfidf_js_asaniczka,
            "scraped_js_sim": tfidf_js_scraped,
            "within_2024_shift": within_24_tfidf_noise,
            "cross_abs": cross_tfidf_signal,
            "snr": (cross_tfidf_signal / within_24_tfidf_noise) if within_24_tfidf_noise > 1e-9 else float("nan"),
        },
    ]
    conv_df = pd.DataFrame(rows)
    conv_df.to_csv(TBL / "convergence_with_calibration.csv", index=False)

    # Verdict
    survives_emb = (cross_emb_signal >= 2 * within_24_emb_noise) if not np.isnan(within_24_emb_noise) else False
    survives_tfidf = (cross_tfidf_signal >= 2 * within_24_tfidf_noise) if not np.isnan(within_24_tfidf_noise) else False
    # Direction: convergence means sim(junior, senior) RISES
    converging_emb = emb_js_2026 > emb_js_2024
    converging_tfidf = tfidf_js_2026 > tfidf_js_2024

    with (TBL / "convergence_verdict.txt").open("w") as fh:
        fh.write("=== Embedding (MiniLM-L6-v2) ===\n")
        fh.write(f"  junior↔senior cosine 2024 pooled: {emb_js_2024:.4f}\n")
        fh.write(f"  junior↔senior cosine 2026 scraped: {emb_js_2026:.4f}\n")
        fh.write(f"  shift: {emb_js_2026 - emb_js_2024:+.4f} ({'↑ converging' if converging_emb else '↓ diverging'})\n")
        fh.write(f"  within-2024 baseline |arshkon js - asaniczka js| = {within_24_emb_noise:.4f}\n")
        fh.write(f"  cross abs shift = {cross_emb_signal:.4f}\n")
        fh.write(f"  SNR (cross / within-2024) = {conv_df.iloc[0]['snr']:.2f}\n")
        fh.write(f"  Convergence signal survives calibration (SNR >= 2)? {'YES' if survives_emb else 'NO'}\n")
        fh.write("\n=== TF-IDF SVD ===\n")
        fh.write(f"  junior↔senior cosine 2024 pooled: {tfidf_js_2024:.4f}\n")
        fh.write(f"  junior↔senior cosine 2026 scraped: {tfidf_js_2026:.4f}\n")
        fh.write(f"  shift: {tfidf_js_2026 - tfidf_js_2024:+.4f} ({'↑ converging' if converging_tfidf else '↓ diverging'})\n")
        fh.write(f"  within-2024 baseline |arshkon js - asaniczka js| = {within_24_tfidf_noise:.4f}\n")
        fh.write(f"  cross abs shift = {cross_tfidf_signal:.4f}\n")
        fh.write(f"  SNR (cross / within-2024) = {conv_df.iloc[1]['snr']:.2f}\n")
        fh.write(f"  Convergence signal survives calibration (SNR >= 2)? {'YES' if survives_tfidf else 'NO'}\n")
    return conv_df, survives_emb


def within_group_dispersion(sampled: pd.DataFrame, emb: np.ndarray):
    print("[disp] within-group dispersion")
    rows = []
    for per in ["2024", "2026"]:
        for sen in ["junior", "mid", "senior", "unknown"]:
            sub = sampled[(sampled["period_bucket"] == per) & (sampled["seniority_3level"] == sen)]
            if len(sub) < 30:
                continue
            X = emb[sub.index.values]
            # subsample for speed
            if len(X) > 1000:
                pick = RNG.choice(len(X), 1000, replace=False)
                Xs = X[pick]
            else:
                Xs = X
            sim = cosine_similarity(Xs)
            # mean off-diagonal
            iu = np.triu_indices_from(sim, k=1)
            rows.append(
                {
                    "period": per,
                    "seniority_3level": sen,
                    "n": len(sub),
                    "mean_pairwise_cosine": float(sim[iu].mean()),
                    "median_pairwise_cosine": float(np.median(sim[iu])),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TBL / "within_group_dispersion.csv", index=False)
    # bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=out, x="seniority_3level", y="mean_pairwise_cosine", hue="period", ax=ax)
    ax.set_title("Within-group mean pairwise cosine (higher = more homogeneous)")
    plt.tight_layout()
    plt.savefig(FIG / "within_group_dispersion.png", dpi=150)
    plt.close()
    return out


def visualize_umap_pca_tsne(sampled: pd.DataFrame, emb: np.ndarray):
    print("[vis] UMAP / PCA / t-SNE")
    # subsample to speed
    if len(sampled) > 6000:
        pick = RNG.choice(len(sampled), 6000, replace=False)
    else:
        pick = np.arange(len(sampled))
    X = emb[pick]
    meta = sampled.iloc[pick].copy().reset_index(drop=True)
    meta["group"] = meta["period_bucket"] + " / " + meta["seniority_3level"]

    # UMAP
    print("[vis] UMAP")
    reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine", n_neighbors=30, min_dist=0.3)
    umap_xy = reducer.fit_transform(X)
    # PCA
    print("[vis] PCA")
    pca_xy = PCA(n_components=2, random_state=42).fit_transform(X)
    # t-SNE (subsample further)
    print("[vis] t-SNE")
    tsne_n = min(3000, len(X))
    tsne_idx = RNG.choice(len(X), tsne_n, replace=False)
    tsne_xy = TSNE(n_components=2, random_state=42, metric="cosine", init="pca", learning_rate="auto").fit_transform(X[tsne_idx])

    # Save embeddings coords
    meta["umap_x"] = umap_xy[:, 0]
    meta["umap_y"] = umap_xy[:, 1]
    meta["pca_x"] = pca_xy[:, 0]
    meta["pca_y"] = pca_xy[:, 1]
    meta.to_csv(TBL / "umap_pca_coords.csv", index=False)

    # Main figure — UMAP with density per group + arrows
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    palette = {
        "2024 / junior": "#1f77b4",
        "2024 / senior": "#aec7e8",
        "2024 / unknown": "#c5c5c5",
        "2024 / mid": "#6baed6",
        "2026 / junior": "#d62728",
        "2026 / senior": "#ff9896",
        "2026 / unknown": "#888888",
        "2026 / mid": "#e7969c",
    }

    # UMAP panel: scatter + density contours for junior/senior 2024 vs 2026
    ax = axes[0]
    for g, color in palette.items():
        sub = meta[meta["group"] == g]
        if len(sub) == 0:
            continue
        ax.scatter(sub["umap_x"], sub["umap_y"], s=3, alpha=0.25, c=color, label=g)
    # density contours for the primary 4 groups
    for g in ["2024 / junior", "2024 / senior", "2026 / junior", "2026 / senior"]:
        sub = meta[meta["group"] == g]
        if len(sub) < 50:
            continue
        try:
            sns.kdeplot(
                x=sub["umap_x"], y=sub["umap_y"], ax=ax, levels=5, color=palette[g],
                linewidths=1.5, alpha=0.7,
            )
        except Exception:
            pass
    # movement arrows for junior and senior centroids 2024 -> 2026
    for sen in ["junior", "senior"]:
        c_2024 = meta[(meta["period_bucket"] == "2024") & (meta["seniority_3level"] == sen)][["umap_x", "umap_y"]].mean()
        c_2026 = meta[(meta["period_bucket"] == "2026") & (meta["seniority_3level"] == sen)][["umap_x", "umap_y"]].mean()
        ax.annotate(
            "",
            xy=(c_2026["umap_x"], c_2026["umap_y"]),
            xytext=(c_2024["umap_x"], c_2024["umap_y"]),
            arrowprops=dict(arrowstyle="->", color="black", lw=2),
        )
        ax.scatter([c_2024["umap_x"]], [c_2024["umap_y"]], s=250, marker="o", c="white", edgecolors="black", linewidths=2, zorder=10)
        ax.scatter([c_2026["umap_x"]], [c_2026["umap_y"]], s=250, marker="X", c="black", edgecolors="white", linewidths=2, zorder=10)
    ax.set_title("UMAP (cosine, k=30): period × seniority")
    ax.legend(loc="best", fontsize=7, markerscale=3, framealpha=0.8)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    # PCA panel
    ax = axes[1]
    for g, color in palette.items():
        sub = meta[meta["group"] == g]
        ax.scatter(sub["pca_x"], sub["pca_y"], s=3, alpha=0.3, c=color, label=g)
    ax.set_title("PCA (2D)")
    ax.set_xlabel("PC-1")
    ax.set_ylabel("PC-2")

    # t-SNE panel
    ax = axes[2]
    meta_tsne = meta.iloc[tsne_idx].copy().reset_index(drop=True)
    meta_tsne["tsne_x"] = tsne_xy[:, 0]
    meta_tsne["tsne_y"] = tsne_xy[:, 1]
    for g, color in palette.items():
        sub = meta_tsne[meta_tsne["group"] == g]
        ax.scatter(sub["tsne_x"], sub["tsne_y"], s=3, alpha=0.3, c=color, label=g)
    ax.set_title("t-SNE")
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")

    plt.tight_layout()
    plt.savefig(FIG / "umap_pca_tsne.png", dpi=150)
    plt.close()

    # Standalone UMAP figure (publication quality focus)
    fig, ax = plt.subplots(figsize=(10, 8))
    for g in ["2024 / junior", "2024 / senior", "2026 / junior", "2026 / senior"]:
        sub = meta[meta["group"] == g]
        if len(sub) == 0:
            continue
        ax.scatter(sub["umap_x"], sub["umap_y"], s=4, alpha=0.3, c=palette[g], label=f"{g} (n={len(sub)})")
    for g in ["2024 / junior", "2024 / senior", "2026 / junior", "2026 / senior"]:
        sub = meta[meta["group"] == g]
        if len(sub) < 50:
            continue
        try:
            sns.kdeplot(
                x=sub["umap_x"], y=sub["umap_y"], ax=ax, levels=5, color=palette[g], linewidths=1.5, alpha=0.8,
            )
        except Exception:
            pass
    for sen in ["junior", "senior"]:
        c_2024 = meta[(meta["period_bucket"] == "2024") & (meta["seniority_3level"] == sen)][["umap_x", "umap_y"]].mean()
        c_2026 = meta[(meta["period_bucket"] == "2026") & (meta["seniority_3level"] == sen)][["umap_x", "umap_y"]].mean()
        ax.annotate(
            "",
            xy=(c_2026["umap_x"], c_2026["umap_y"]),
            xytext=(c_2024["umap_x"], c_2024["umap_y"]),
            arrowprops=dict(arrowstyle="->", color="black", lw=2),
        )
    ax.set_title("Semantic landscape UMAP — SWE postings 2024 vs 2026")
    ax.legend(loc="best", fontsize=9, markerscale=3, framealpha=0.9)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(FIG / "umap_density_contours.png", dpi=150)
    plt.close()
    return meta


def nearest_neighbor_analysis(sampled: pd.DataFrame, emb: np.ndarray, tfidf_svd: np.ndarray):
    print("[nn] 2026 entry -> 5 nearest 2024")
    mask_2026_entry = (sampled["period_bucket"] == "2026") & (sampled["seniority_3level"] == "junior")
    mask_2024 = sampled["period_bucket"] == "2024"
    q_idx = np.where(mask_2026_entry)[0]
    d_idx = np.where(mask_2024)[0]
    if len(q_idx) == 0 or len(d_idx) == 0:
        return None

    def _nn(X, q, d, k=5):
        Q = X[q]
        D = X[d]
        sims = Q @ D.T
        topk = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        return topk

    def _summary(topk, d_idx, label):
        sens = []
        for row in topk:
            sens.append(sampled.iloc[d_idx[row]]["seniority_3level"].values)
        flat = np.concatenate(sens)
        vc = pd.Series(flat).value_counts(normalize=True)
        # base rate in 2024 sample
        base = sampled.iloc[d_idx]["seniority_3level"].value_counts(normalize=True)
        rows = []
        for sen in ["junior", "mid", "senior", "unknown"]:
            rows.append(
                {
                    "representation": label,
                    "neighbor_seniority": sen,
                    "nn_share": float(vc.get(sen, 0.0)),
                    "base_rate_2024": float(base.get(sen, 0.0)),
                    "excess_vs_base_pp": (float(vc.get(sen, 0.0)) - float(base.get(sen, 0.0))) * 100,
                }
            )
        return rows

    topk_emb = _nn(emb, q_idx, d_idx)
    topk_tfidf = _nn(tfidf_svd, q_idx, d_idx)

    rows = _summary(topk_emb, d_idx, "embedding") + _summary(topk_tfidf, d_idx, "tfidf")
    out = pd.DataFrame(rows)
    out.to_csv(TBL / "nearest_neighbor_2026entry_to_2024.csv", index=False)
    return out


def outlier_analysis(sampled: pd.DataFrame, emb: np.ndarray):
    print("[out] outliers per group")
    rows = []
    for per in ["2024", "2026"]:
        for sen in ["junior", "senior"]:
            sub = sampled[(sampled["period_bucket"] == per) & (sampled["seniority_3level"] == sen)]
            if len(sub) < 30:
                continue
            X = emb[sub.index.values]
            c = trimmed_centroid(X)
            sims = X @ c
            # bottom 20 (most distant)
            order = np.argsort(sims)[:20]
            for rank, idx in enumerate(order):
                r = sub.iloc[idx]
                rows.append(
                    {
                        "period": per,
                        "seniority_3level": sen,
                        "rank_outlier": rank + 1,
                        "sim_to_centroid": float(sims[idx]),
                        "uid": r["uid"],
                        "company_name_canonical": r["company_name_canonical"],
                    }
                )
    pd.DataFrame(rows).to_csv(TBL / "outliers_per_group.csv", index=False)


def sensitivities(sampled: pd.DataFrame, emb: np.ndarray):
    print("[sens] aggregator, swe tier, source restriction")
    sampled["group"] = sampled["period_bucket"] + "|" + sampled["seniority_3level"]

    def _js_sim(sub_mask, X):
        sub = sampled[sub_mask]
        rj = sub[sub["seniority_3level"] == "junior"].index.values
        rs = sub[sub["seniority_3level"] == "senior"].index.values
        if len(rj) < 30 or len(rs) < 30:
            return float("nan")
        cj = trimmed_centroid(X[rj])
        cs = trimmed_centroid(X[rs])
        return float(np.dot(cj, cs))

    # Per spec
    rows = []
    # (a) aggregator exclusion
    for per in ["2024", "2026"]:
        all_s = _js_sim((sampled["period_bucket"] == per), emb)
        excl = _js_sim((sampled["period_bucket"] == per) & ~sampled["is_aggregator"].fillna(False), emb)
        rows.append({"sensitivity": "aggregator_exclusion", "period": per, "all": all_s, "excl_aggregator": excl})
    # (c) YOE-based seniority proxy: junior = yoe<=2, senior = yoe>=5
    s2 = sampled.copy()
    s2["yoe_sen"] = np.where(s2["yoe_extracted"].le(2), "junior", np.where(s2["yoe_extracted"].ge(5), "senior", "mid"))
    for per in ["2024", "2026"]:
        sub = s2[s2["period_bucket"] == per]
        rj = sub[sub["yoe_sen"] == "junior"].index.values
        rs = sub[sub["yoe_sen"] == "senior"].index.values
        if len(rj) < 20 or len(rs) < 20:
            continue
        cj = trimmed_centroid(emb[rj])
        cs = trimmed_centroid(emb[rs])
        rows.append({"sensitivity": "yoe_proxy_seniority", "period": per, "all": float(np.dot(cj, cs)), "excl_aggregator": float("nan")})
    # (e) arshkon-only baseline
    for per in ["2024"]:
        mask = (sampled["period_bucket"] == per) & (sampled["source"] == "kaggle_arshkon")
        rows.append({"sensitivity": "arshkon_only_2024", "period": per, "all": _js_sim(mask, emb), "excl_aggregator": float("nan")})
    # (g) swe tier: exclude title_lookup_llm
    for per in ["2024", "2026"]:
        mask = (sampled["period_bucket"] == per) & (sampled["swe_classification_tier"] != "title_lookup_llm")
        rows.append({"sensitivity": "swe_tier_no_title_lookup", "period": per, "all": _js_sim(mask, emb), "excl_aggregator": float("nan")})

    out = pd.DataFrame(rows)
    out.to_csv(TBL / "sensitivity_summary.csv", index=False)
    return out


def robustness_table(conv_df, split_emb, split_tfidf):
    print("[robust] representation robustness")
    rows = [
        {
            "finding": "junior↔senior centroid similarity rises 2024→2026",
            "embedding_direction": "↑" if conv_df.iloc[0]["2026_scraped"] > conv_df.iloc[0]["2024_pooled"] else "↓",
            "embedding_magnitude": conv_df.iloc[0]["cross_period_shift"],
            "tfidf_direction": "↑" if conv_df.iloc[1]["2026_scraped"] > conv_df.iloc[1]["2024_pooled"] else "↓",
            "tfidf_magnitude": conv_df.iloc[1]["cross_period_shift"],
        },
        {
            "finding": "period gap larger than seniority gap",
            "embedding_direction": "YES" if (split_emb["within_period_mean"] - split_emb["across_period_mean"]) > (split_emb["within_seniority_mean"] - split_emb["across_seniority_mean"]) else "NO",
            "embedding_magnitude": split_emb["within_period_mean"] - split_emb["across_period_mean"],
            "tfidf_direction": "YES" if (split_tfidf["within_period_mean"] - split_tfidf["across_period_mean"]) > (split_tfidf["within_seniority_mean"] - split_tfidf["across_seniority_mean"]) else "NO",
            "tfidf_magnitude": split_tfidf["within_period_mean"] - split_tfidf["across_period_mean"],
        },
    ]
    pd.DataFrame(rows).to_csv(TBL / "robustness_table.csv", index=False)


def main():
    idx, emb = load_data()
    sampled, sampled_emb = stratified_sample(idx, emb, per_group=2000)

    # Build TF-IDF for sampled uids
    tfidf_svd = load_tfidf_for_sampled(sampled)

    emb_sim, tfidf_sim, split_emb, split_tfidf = structural_map(sampled, sampled_emb, tfidf_svd)
    conv_df, _ = convergence_analysis(sampled, sampled_emb, tfidf_svd)
    robustness_table(conv_df, split_emb, split_tfidf)
    within_group_dispersion(sampled, sampled_emb)
    visualize_umap_pca_tsne(sampled, sampled_emb)
    nearest_neighbor_analysis(sampled, sampled_emb, tfidf_svd)
    outlier_analysis(sampled, sampled_emb)
    sensitivities(sampled, sampled_emb)
    print("[done] T15 complete")


if __name__ == "__main__":
    main()
