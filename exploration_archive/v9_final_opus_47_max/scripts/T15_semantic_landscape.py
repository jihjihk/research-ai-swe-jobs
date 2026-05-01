"""T15 — Semantic similarity landscape & convergence analysis.

Outputs:
  exploration/tables/T15/sample_meta.csv
  exploration/tables/T15/centroid_similarity_embedding.csv         (group x group)
  exploration/tables/T15/centroid_similarity_tfidf.csv             (group x group)
  exploration/tables/T15/convergence_within_period.csv             (seniority-level convergence)
  exploration/tables/T15/convergence_calibration.csv               (within-2024 vs cross-period)
  exploration/tables/T15/within_group_dispersion.csv
  exploration/tables/T15/nearest_neighbor_2026_entry.csv            (yield counts)
  exploration/tables/T15/nearest_neighbor_excess_over_base_rate.csv
  exploration/tables/T15/robustness_table.csv
  exploration/tables/T15/outliers_distance_to_own_group.csv
  exploration/figures/T15_umap_period_seniority.png
  exploration/figures/T15_pca_period_seniority.png
  exploration/figures/T15_tsne_period_seniority.png
  exploration/figures/T15_similarity_heatmap_embedding.png
  exploration/figures/T15_similarity_heatmap_tfidf.png
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/home/jihgaboot/gabor/job-research")
SHARED = ROOT / "exploration/artifacts/shared"
OUT_TBL = ROOT / "exploration/tables/T15"
OUT_FIG = ROOT / "exploration/figures"
OUT_TBL.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

EMB = SHARED / "swe_embeddings.npy"
EMB_IDX = SHARED / "swe_embedding_index.parquet"
TEXT = SHARED / "swe_cleaned_text.parquet"

RNG = np.random.RandomState(0)
SAMPLE_PER_CELL = 2000
MAX_TOTAL = 12000

# ---------------------------------------------------------------------------
# Load + sample
# ---------------------------------------------------------------------------

def load_frames():
    print("[load] embeddings + index + text...", flush=True)
    emb = np.load(EMB).astype(np.float32)
    idx = pq.read_table(EMB_IDX).to_pandas()
    idx["row_idx"] = idx["row_idx"].astype(int)

    text_cols = ["uid", "description_cleaned", "text_source", "source", "period",
                 "seniority_final", "seniority_3level", "yoe_min_years_llm",
                 "llm_classification_coverage", "swe_classification_tier",
                 "is_aggregator", "company_name_canonical"]
    txt = pq.read_table(TEXT, columns=text_cols).to_pandas()

    # period2 collapse
    def period2(p):
        if isinstance(p, str) and p.startswith("2024"):
            return "2024"
        if isinstance(p, str) and p.startswith("2026"):
            return "2026"
        return None
    txt["period2"] = txt["period"].apply(period2)
    # Only embed rows use llm text
    joined = idx.merge(txt, on="uid", how="inner")
    joined = joined[joined["text_source"] == "llm"].reset_index(drop=True)
    return emb, joined


def stratified_sample(df, per_cell=SAMPLE_PER_CELL, max_total=MAX_TOTAL):
    """Sample up to per_cell per (period2, seniority_3level), capped at max_total total."""
    # Only keep groups with >= 50 eligible rows; include 'unknown' as a group
    keys = ["period2", "seniority_3level"]
    allowed = df[df["period2"].isin(["2024", "2026"]) &
                 df["seniority_3level"].isin(["junior", "mid", "senior", "unknown"])].copy()

    # Apply company cap at 20 per (period2, company) to reduce prolific-firm dominance
    allowed = allowed.copy()
    allowed["_rn"] = allowed.groupby(["period2", "company_name_canonical"]).cumcount()
    allowed = allowed[allowed["_rn"] < 20].drop(columns=["_rn"])

    parts = []
    for (p, s), g in allowed.groupby(keys):
        n = min(len(g), per_cell)
        parts.append(g.sample(n=n, random_state=0))
    sample = pd.concat(parts, ignore_index=True)
    if len(sample) > max_total:
        sample = sample.sample(n=max_total, random_state=0).reset_index(drop=True)
    return sample


# ---------------------------------------------------------------------------
# TF-IDF / SVD
# ---------------------------------------------------------------------------

def tfidf_svd(texts, n_components=100, max_features=20000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.7,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    Xs = svd.fit_transform(X)
    # L2-normalize so cosine == dot product
    Xs = normalize(Xs)
    explained = float(svd.explained_variance_ratio_.sum())
    return Xs, explained, vec, svd


# ---------------------------------------------------------------------------
# Centroid + similarity + dispersion
# ---------------------------------------------------------------------------

def trimmed_centroid(X, keep=0.9):
    """Centroid with bottom-10-pct-closest (i.e. top-10-pct-furthest) removed.

    Procedure: compute centroid naively, then remove the 10% most distant
    points from the centroid, then re-average.
    """
    from sklearn.preprocessing import normalize
    if len(X) == 0:
        return None
    c = X.mean(axis=0)
    # normalize centroid for cosine distances
    cn = c / (np.linalg.norm(c) + 1e-12)
    Xn = normalize(X)
    sims = Xn @ cn
    n_keep = max(1, int(np.floor(len(X) * keep)))
    keep_idx = np.argsort(-sims)[:n_keep]
    c2 = X[keep_idx].mean(axis=0)
    return c2


def pairwise_cosine(A, B):
    """Cosine similarity matrix between row-groups A and B."""
    from sklearn.preprocessing import normalize
    An = normalize(A.reshape(1, -1) if A.ndim == 1 else A)
    Bn = normalize(B.reshape(1, -1) if B.ndim == 1 else B)
    return An @ Bn.T


def centroid_matrix(sample, X, group_keys, trim=True):
    """Return centroid similarity matrix over group_keys rows of sample."""
    groups = sorted(set([tuple(row) for row in sample[group_keys].values.tolist()]))
    centroids = []
    labels = []
    for g in groups:
        mask = np.logical_and.reduce([sample[k] == v for k, v in zip(group_keys, g)])
        Xg = X[mask.values] if hasattr(mask, "values") else X[mask]
        if len(Xg) < 5:
            continue
        c = trimmed_centroid(Xg) if trim else Xg.mean(axis=0)
        centroids.append(c)
        labels.append("|".join(map(str, g)))
    cmat = np.vstack(centroids)
    sim = pairwise_cosine(cmat, cmat)
    df = pd.DataFrame(sim, index=labels, columns=labels)
    return df, labels, cmat


def within_group_dispersion(sample, X, group_keys):
    rows = []
    groups = sorted(set([tuple(r) for r in sample[group_keys].values.tolist()]))
    from sklearn.preprocessing import normalize
    for g in groups:
        mask = np.logical_and.reduce([sample[k] == v for k, v in zip(group_keys, g)])
        Xg = X[mask.values] if hasattr(mask, "values") else X[mask]
        if len(Xg) < 20:
            continue
        # Subsample to 600 for O(N^2) avg cosine
        n_sub = min(600, len(Xg))
        sel = np.random.RandomState(0).choice(len(Xg), n_sub, replace=False)
        Xs = normalize(Xg[sel])
        sims = Xs @ Xs.T
        # exclude self-sims (diag)
        np.fill_diagonal(sims, np.nan)
        avg = float(np.nanmean(sims))
        med = float(np.nanmedian(sims))
        rows.append({"group": "|".join(map(str, g)), "n_group": int(len(Xg)),
                     "avg_pairwise_cosine": avg, "median_pairwise_cosine": med})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Nearest neighbor analysis
# ---------------------------------------------------------------------------

def nearest_neighbor_2026_entry(sample, X, X_tfidf=None, k=5):
    """For each 2026 junior posting, find the k nearest 2024 neighbors.

    Compute seniority distribution of neighbors, then the excess-over-base rate.
    """
    import sklearn.neighbors as nn
    sample2024 = sample[sample.period2 == "2024"].reset_index(drop=True)
    sample2026_j = sample[(sample.period2 == "2026") & (sample.seniority_3level == "junior")].reset_index(drop=True)
    if len(sample2026_j) == 0 or len(sample2024) == 0:
        return None, None, None
    # Using normalized cosine (dot)
    from sklearn.preprocessing import normalize
    idx2024 = sample[sample.period2 == "2024"].index.values
    idx2026j = sample[(sample.period2 == "2026") & (sample.seniority_3level == "junior")].index.values

    def compute_dist(Xmat):
        A = normalize(Xmat[idx2024])
        B = normalize(Xmat[idx2026j])
        sim = B @ A.T  # shape (n_2026_j, n_2024)
        nn_idx = np.argsort(-sim, axis=1)[:, :k]
        return nn_idx

    # Baseline seniority distribution in 2024 sample
    base = sample2024["seniority_3level"].value_counts(normalize=True).to_dict()

    def summarize(nn_idx, label):
        rows = []
        for i, neigh in enumerate(nn_idx):
            neigh_sen = sample2024.iloc[neigh]["seniority_3level"].tolist()
            rows.append({"row": i, "neighbors_seniority": ";".join(neigh_sen)})
        df_rows = pd.DataFrame(rows)
        # Fraction of 2026-junior whose top-k contains each seniority
        flat = pd.Series([s for r in df_rows.neighbors_seniority for s in r.split(";")])
        shares = flat.value_counts(normalize=True).to_dict()
        out_rows = []
        for sen in ["junior", "mid", "senior", "unknown"]:
            rate = shares.get(sen, 0.0)
            base_rate = base.get(sen, 0.0)
            out_rows.append({
                "representation": label,
                "neighbor_seniority": sen,
                "neighbor_share_of_top_k": rate,
                "base_rate_2024_sample": base_rate,
                "excess_over_base": rate - base_rate,
                "ratio": rate / base_rate if base_rate > 0 else np.nan,
            })
        return pd.DataFrame(out_rows)

    nn_emb = compute_dist(X)
    summary_emb = summarize(nn_emb, "embedding")
    if X_tfidf is not None:
        nn_tfidf = compute_dist(X_tfidf)
        summary_tfidf = summarize(nn_tfidf, "tfidf")
    else:
        summary_tfidf = None

    return summary_emb, summary_tfidf, len(sample2026_j)


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------

def identify_outliers(sample, X, group_keys, k=25):
    """Points most unlike their own group (by centroid distance)."""
    from sklearn.preprocessing import normalize
    out_rows = []
    groups = sorted(set([tuple(r) for r in sample[group_keys].values.tolist()]))
    for g in groups:
        mask = np.logical_and.reduce([sample[k2] == v for k2, v in zip(group_keys, g)])
        sub = sample[mask].reset_index(drop=True)
        Xg = X[mask.values] if hasattr(mask, "values") else X[mask]
        if len(Xg) < 50:
            continue
        c = trimmed_centroid(Xg)
        Xn = normalize(Xg)
        cn = c / (np.linalg.norm(c) + 1e-12)
        sims = Xn @ cn
        # bottom-k least similar
        bot = np.argsort(sims)[:k]
        for i in bot:
            out_rows.append({
                "group": "|".join(map(str, g)),
                "uid": sub.iloc[i]["uid"],
                "company": sub.iloc[i].get("company_name_canonical"),
                "source": sub.iloc[i]["source"],
                "cosine_to_group_centroid": float(sims[i]),
                "seniority_final": sub.iloc[i].get("seniority_final"),
                "text_snippet": (sub.iloc[i].get("description_cleaned") or "")[:240],
            })
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# UMAP + PCA + t-SNE
# ---------------------------------------------------------------------------

def compute_umap(X, n_neighbors=30, min_dist=0.15, metric="cosine"):
    import umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        metric=metric, random_state=0)
    return reducer.fit_transform(X)


def compute_pca(X, n=2):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    Xn = normalize(X)
    pca = PCA(n_components=n, random_state=0)
    return pca.fit_transform(Xn), pca.explained_variance_ratio_


def compute_tsne(X, perplexity=30):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import normalize
    Xn = normalize(X)
    # Initialize with PCA for speed on 384D -> 50D first
    from sklearn.decomposition import PCA
    if Xn.shape[1] > 50:
        Xn = PCA(n_components=50, random_state=0).fit_transform(Xn)
    t = TSNE(n_components=2, perplexity=perplexity, init="pca",
             learning_rate="auto", random_state=0)
    return t.fit_transform(Xn)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_2d(coords, sample, out_path, title, method="UMAP"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # color by (period, seniority)
    palette = {
        ("2024", "junior"): "#1f77b4",
        ("2024", "mid"): "#17becf",
        ("2024", "senior"): "#2ca02c",
        ("2024", "unknown"): "#bdbdbd",
        ("2026", "junior"): "#ff7f0e",
        ("2026", "mid"): "#e377c2",
        ("2026", "senior"): "#d62728",
        ("2026", "unknown"): "#7f7f7f",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    # Left: all
    for period, color in zip(["2024", "2026"], ["#1f77b4", "#d62728"]):
        mask = (sample.period2 == period).values
        axes[0].scatter(coords[mask, 0], coords[mask, 1], s=5,
                        alpha=0.25, c=color, label=period)
    axes[0].set_title(f"{method}: by period")
    axes[0].legend(markerscale=3)

    # Right: (period x seniority) faceted colors
    for (p, s), color in palette.items():
        mask = ((sample.period2 == p) & (sample.seniority_3level == s)).values
        if not mask.any():
            continue
        axes[1].scatter(coords[mask, 0], coords[mask, 1], s=5,
                        alpha=0.3, c=color, label=f"{p}-{s}")
    axes[1].set_title(f"{method}: by period × seniority")
    axes[1].legend(markerscale=3, fontsize=8)

    # Centroid arrows per seniority (2024 -> 2026)
    for sen in ["junior", "senior"]:
        c24 = coords[((sample.period2 == "2024") & (sample.seniority_3level == sen)).values].mean(axis=0) \
            if ((sample.period2 == "2024") & (sample.seniority_3level == sen)).any() else None
        c26 = coords[((sample.period2 == "2026") & (sample.seniority_3level == sen)).values].mean(axis=0) \
            if ((sample.period2 == "2026") & (sample.seniority_3level == sen)).any() else None
        if c24 is not None and c26 is not None:
            axes[1].annotate("", xy=(c26[0], c26[1]), xytext=(c24[0], c24[1]),
                             arrowprops=dict(arrowstyle="->", color="black", lw=2))
            axes[1].text(c24[0], c24[1], f"{sen}'24", fontsize=9, fontweight="bold")
            axes[1].text(c26[0], c26[1], f"{sen}'26", fontsize=9, fontweight="bold")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_similarity_heatmap(sim_df, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_df.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(sim_df.columns)))
    ax.set_xticklabels(sim_df.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(sim_df.index)))
    ax.set_yticklabels(sim_df.index, fontsize=8)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.04)
    # Annotate cells with values
    for i in range(sim_df.shape[0]):
        for j in range(sim_df.shape[1]):
            v = sim_df.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(v) < 0.6 else "white")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


# ---------------------------------------------------------------------------
# Convergence / calibration
# ---------------------------------------------------------------------------

def convergence_analysis(sample, X_emb, X_tfidf, label="embedding"):
    """Seniority-level centroid similarity within each period; within-2024 calibration."""
    from sklearn.preprocessing import normalize
    rows = []
    # Use 3-level seniority junior/senior (skip mid/unknown for the core test)
    for period in ["2024", "2026"]:
        mask_j = (sample.period2 == period) & (sample.seniority_3level == "junior")
        mask_s = (sample.period2 == period) & (sample.seniority_3level == "senior")
        if mask_j.sum() < 20 or mask_s.sum() < 20:
            continue
        for Xmat, repr_label in [(X_emb, "embedding"), (X_tfidf, "tfidf")]:
            cj = trimmed_centroid(Xmat[mask_j.values])
            cs = trimmed_centroid(Xmat[mask_s.values])
            sim = float(pairwise_cosine(cj, cs)[0, 0])
            rows.append({"period": period, "pair": "junior-vs-senior",
                         "representation": repr_label,
                         "cosine_similarity": sim,
                         "n_j": int(mask_j.sum()),
                         "n_s": int(mask_s.sum())})
    # within-2024 calibration: arshkon vs asaniczka same-seniority
    for sen in ["junior", "senior", "unknown"]:
        mask_a = (sample.period2 == "2024") & (sample.seniority_3level == sen) & (sample.source == "kaggle_arshkon")
        mask_b = (sample.period2 == "2024") & (sample.seniority_3level == sen) & (sample.source == "kaggle_asaniczka")
        if mask_a.sum() < 15 or mask_b.sum() < 15:
            continue
        for Xmat, repr_label in [(X_emb, "embedding"), (X_tfidf, "tfidf")]:
            ca = trimmed_centroid(Xmat[mask_a.values])
            cb = trimmed_centroid(Xmat[mask_b.values])
            sim = float(pairwise_cosine(ca, cb)[0, 0])
            rows.append({"period": "within-2024", "pair": f"arshkon-vs-asaniczka-{sen}",
                         "representation": repr_label,
                         "cosine_similarity": sim,
                         "n_j": int(mask_a.sum()),
                         "n_s": int(mask_b.sum())})
    # cross-period same-seniority (arshkon2024 vs scraped2026; asaniczka2024 vs scraped2026)
    for sen in ["junior", "senior"]:
        for src24 in ["kaggle_arshkon", "kaggle_asaniczka"]:
            mask_a = (sample.period2 == "2024") & (sample.seniority_3level == sen) & (sample.source == src24)
            mask_b = (sample.period2 == "2026") & (sample.seniority_3level == sen)
            if mask_a.sum() < 15 or mask_b.sum() < 15:
                continue
            for Xmat, repr_label in [(X_emb, "embedding"), (X_tfidf, "tfidf")]:
                ca = trimmed_centroid(Xmat[mask_a.values])
                cb = trimmed_centroid(Xmat[mask_b.values])
                sim = float(pairwise_cosine(ca, cb)[0, 0])
                rows.append({"period": "cross-period", "pair": f"{src24}-vs-scraped-{sen}",
                             "representation": repr_label,
                             "cosine_similarity": sim,
                             "n_j": int(mask_a.sum()),
                             "n_s": int(mask_b.sum())})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Robustness table
# ---------------------------------------------------------------------------

def robustness_table(conv_df):
    """Does each finding hold under both embedding and tfidf?"""
    rows = []
    for (period, pair), g in conv_df.groupby(["period", "pair"]):
        emb_val = g[g.representation == "embedding"]["cosine_similarity"].values
        tfidf_val = g[g.representation == "tfidf"]["cosine_similarity"].values
        rows.append({
            "period": period, "pair": pair,
            "embedding": float(emb_val[0]) if len(emb_val) else np.nan,
            "tfidf": float(tfidf_val[0]) if len(tfidf_val) else np.nan,
        })
    df = pd.DataFrame(rows)
    df["both_high"] = (df["embedding"] > 0.5) & (df["tfidf"] > 0.5)
    df["both_low"] = (df["embedding"] < 0.5) & (df["tfidf"] < 0.5)
    df["diverges"] = ((df["embedding"] > 0.6) & (df["tfidf"] < 0.4)) | \
                      ((df["embedding"] < 0.4) & (df["tfidf"] > 0.6))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    emb, joined = load_frames()
    print(f"  joined n={len(joined):,}")

    print("[sample] stratified sample (cap 20/company)...", flush=True)
    sample = stratified_sample(joined, SAMPLE_PER_CELL, MAX_TOTAL)
    print(f"  sample n={len(sample):,}")
    # write sample meta
    sample_meta = sample.groupby(["period2", "seniority_3level"]).agg(
        n=("uid", "count"),
        n_companies=("company_name_canonical", "nunique"),
    ).reset_index()
    # Add text source distribution
    ts = sample.groupby(["period2", "seniority_3level", "text_source"]).size().reset_index(name="n")
    sample_meta.to_csv(OUT_TBL / "sample_meta.csv", index=False)
    ts.to_csv(OUT_TBL / "sample_meta_text_source.csv", index=False)

    # Subset embeddings to sample order
    # Need row_idx per sample row to pluck from emb
    X_emb = emb[sample["row_idx"].values]
    print(f"  X_emb shape: {X_emb.shape}")

    # TF-IDF SVD
    print("[tfidf] vectorize + SVD(100)...", flush=True)
    X_tfidf, explained, _, _ = tfidf_svd(sample["description_cleaned"].fillna("").tolist())
    print(f"  explained variance sum: {explained:.3f}")
    with open(OUT_TBL / "tfidf_meta.json", "w") as f:
        json.dump({"explained_variance_ratio_sum": explained,
                   "n_components": 100}, f, indent=2)

    # Centroid similarity matrices
    print("[centroids] embedding + tfidf...", flush=True)
    sim_emb, labels, _ = centroid_matrix(sample, X_emb,
                                          ["period2", "seniority_3level"], trim=True)
    sim_tfidf, _, _ = centroid_matrix(sample, X_tfidf,
                                       ["period2", "seniority_3level"], trim=True)
    sim_emb.to_csv(OUT_TBL / "centroid_similarity_embedding.csv")
    sim_tfidf.to_csv(OUT_TBL / "centroid_similarity_tfidf.csv")

    # Source-level matrix for calibration inside 2024
    sim_emb_src, _, _ = centroid_matrix(sample, X_emb,
                                         ["period2", "seniority_3level", "source"], trim=True)
    sim_tfidf_src, _, _ = centroid_matrix(sample, X_tfidf,
                                           ["period2", "seniority_3level", "source"], trim=True)
    sim_emb_src.to_csv(OUT_TBL / "centroid_similarity_embedding_source.csv")
    sim_tfidf_src.to_csv(OUT_TBL / "centroid_similarity_tfidf_source.csv")

    # Convergence + calibration
    print("[convergence] + calibration...", flush=True)
    conv = convergence_analysis(sample, X_emb, X_tfidf)
    conv.to_csv(OUT_TBL / "convergence_within_period.csv", index=False)
    robust = robustness_table(conv)
    robust.to_csv(OUT_TBL / "robustness_table.csv", index=False)

    # Within-group dispersion
    print("[dispersion] within-group pairwise cosines...", flush=True)
    disp_emb = within_group_dispersion(sample, X_emb, ["period2", "seniority_3level"])
    disp_emb["representation"] = "embedding"
    disp_tfidf = within_group_dispersion(sample, X_tfidf, ["period2", "seniority_3level"])
    disp_tfidf["representation"] = "tfidf"
    pd.concat([disp_emb, disp_tfidf], ignore_index=True).to_csv(
        OUT_TBL / "within_group_dispersion.csv", index=False)

    # Nearest neighbor
    print("[nn] 2026 junior -> 2024 neighbors...", flush=True)
    nn_emb, nn_tfidf, n_q = nearest_neighbor_2026_entry(sample, X_emb, X_tfidf, k=5)
    if nn_emb is not None:
        nn_emb.to_csv(OUT_TBL / "nearest_neighbor_2026_entry_embedding.csv", index=False)
    if nn_tfidf is not None:
        nn_tfidf.to_csv(OUT_TBL / "nearest_neighbor_2026_entry_tfidf.csv", index=False)
    if nn_emb is not None:
        combined = pd.concat([nn_emb, nn_tfidf], ignore_index=True) if nn_tfidf is not None else nn_emb
        combined.to_csv(OUT_TBL / "nearest_neighbor_excess_over_base_rate.csv", index=False)
        with open(OUT_TBL / "nearest_neighbor_meta.json", "w") as f:
            json.dump({"n_query_2026_junior": int(n_q), "k": 5}, f, indent=2)

    # Outliers
    print("[outliers]...", flush=True)
    outliers = identify_outliers(sample, X_emb,
                                  ["period2", "seniority_3level"], k=15)
    outliers.to_csv(OUT_TBL / "outliers_distance_to_own_group.csv", index=False)

    # Figures
    print("[figures] UMAP + PCA + t-SNE + heatmaps...", flush=True)
    coords_umap = compute_umap(X_emb)
    plot_2d(coords_umap, sample, OUT_FIG / "T15_umap_period_seniority.png",
            title="T15 — UMAP of SWE posting embeddings (cap-20/company sample)",
            method="UMAP")
    # also UMAP of TF-IDF
    coords_umap_t = compute_umap(X_tfidf)
    plot_2d(coords_umap_t, sample, OUT_FIG / "T15_umap_tfidf_period_seniority.png",
            title="T15 — UMAP of SWE TF-IDF-SVD(100)", method="UMAP (TF-IDF)")

    coords_pca, ev = compute_pca(X_emb, n=2)
    plot_2d(coords_pca, sample, OUT_FIG / "T15_pca_period_seniority.png",
            title=f"T15 — PCA (EV={ev[0]:.2f}/{ev[1]:.2f})",
            method="PCA")
    coords_tsne = compute_tsne(X_emb)
    plot_2d(coords_tsne, sample, OUT_FIG / "T15_tsne_period_seniority.png",
            title="T15 — t-SNE (perplexity 30)",
            method="t-SNE")

    plot_similarity_heatmap(sim_emb, OUT_FIG / "T15_similarity_heatmap_embedding.png",
                             "T15 — Centroid similarity (embedding)")
    plot_similarity_heatmap(sim_tfidf, OUT_FIG / "T15_similarity_heatmap_tfidf.png",
                             "T15 — Centroid similarity (TF-IDF-SVD100)")

    print(f"[done] {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
