"""
Composite B v3 — sub-cluster probe on the AI-coded cohort.

Tests whether the AI-coded portion of the Data + AI/ML mega-cluster
(family_split_k30 == "0_AI", ~5k postings) is internally structured into
distinct role archetypes — the way web development split into frontend /
backend / fullstack / distributed systems — or whether "AI engineer" is
still one undifferentiated role in 2026.

Four validity tests, from the plan in
`eda/research_memos/composite_B_v3_findings.md` §"Inside the AI cohort":

  1. Density separation — HDBSCAN sweep at min_cluster_size ∈ {10,20,40,80}
     on a 5D UMAP of the AI-subset embeddings alone.
  2. Silhouette score + inter/intra centroid-distance ratio (higher-dim cosine).
  3. Seed stability — ARI across UMAP random seeds {42, 1337, 2026}.
  4. Human-interpretable characterization — per-cluster c-TF-IDF terms and
     regex-themed growth decomposition at the fine (mcs=10) granularity.

Inputs (reuses the v3-LLM cache, no re-embedding):
  eda/artifacts/_composite_B_v3llm_cache/embeddings.npy
  eda/artifacts/_composite_B_v3llm_cache/uids.npy
  eda/artifacts/_composite_B_v3llm_cache/sample.parquet
  eda/artifacts/composite_B_v3llm_labels.parquet

Outputs:
  eda/tables/composite_B_v3llm_ai_subcluster_sweep.csv
  eda/tables/composite_B_v3llm_ai_subcluster_metrics.csv
  eda/tables/composite_B_v3llm_ai_subcluster_clusters.csv
  eda/tables/composite_B_v3llm_ai_theme_growth.csv
  eda/artifacts/composite_B_v3llm_ai_subcluster_summary.json

Run:
  ./.venv/bin/python eda/scripts/composite_B_v3_ai_subcluster_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Avoid eda/scripts/profile.py shadowing stdlib `profile`.
_THIS_DIR = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _THIS_DIR and p != ""]

import json
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES = PROJECT_ROOT / "eda" / "tables"
ARTIFACTS = PROJECT_ROOT / "eda" / "artifacts"
CACHE = ARTIFACTS / "_composite_B_v3llm_cache"

RANDOM_STATE = 42
STABILITY_SEEDS = (42, 1337, 2026)
MCS_SWEEP = (10, 20, 40, 80)
PRIMARY_MCS = 20  # the setting with the most stable cluster structure

# Regex themes for the fine-grained (mcs=10) growth decomposition. These
# assign each micro-cluster to the first matching theme based on its top
# c-TF-IDF terms. Order matters: LLM/agents is tested first because
# many 2026 postings use agent vocabulary in passing.
THEME_HINTS: list[tuple[str, re.Pattern]] = [
    ("llm_agents_infra", re.compile(
        r"\b(llm|rag|retrieval|vector|embedding|prompt|agent|agents|langchain|"
        r"agentic|bedrock|foundry|copilot studio|orchestration)\b", re.I)),
    ("ml_ops_deployment", re.compile(
        r"\b(mlops|sagemaker|inference|pipelines?|ci/cd|cicd|serving|triton|"
        r"model deployment|production)\b", re.I)),
    ("applied_ml_ranking_forecast", re.compile(
        r"\b(recommendation|ranking|ads|fraud|forecast|anomaly|risk|churn|"
        r"pricing|personalization|supply chain)\b", re.I)),
    ("data_platform_analytics", re.compile(
        r"\b(dbt|snowflake|databricks|fabric|spark|warehouse|etl|feature store|"
        r"data engineer|data pipeline|analytics engineer|bi)\b", re.I)),
    ("classical_data_science", re.compile(
        r"\b(statistical|statistics|data science|data scientist|econometrics|"
        r"analytics|business insights|marketing analytics)\b", re.I)),
    ("voice_chat_product", re.compile(
        r"\b(voice|conversational|chatbot|speech|assistant|copilot|"
        r"customer experience)\b", re.I)),
    ("foundation_research", re.compile(
        r"\b(research|publications|phd|pretrain|foundation model|novel|"
        r"state.of.the.art|paper|architecture research)\b", re.I)),
]


def load_ai_subset():
    emb_full = np.load(CACHE / "embeddings.npy")
    uids_full = np.load(CACHE / "uids.npy", allow_pickle=True)
    labels = pq.read_table(ARTIFACTS / "composite_B_v3llm_labels.parquet").to_pandas()
    sample = pq.read_table(CACHE / "sample.parquet").to_pandas()

    uid_to_idx = {u: i for i, u in enumerate(uids_full)}
    sub = labels[labels["family_split_k30"] == "0_AI"].reset_index(drop=True).copy()
    sub["emb_idx"] = sub["uid"].map(uid_to_idx)
    assert sub["emb_idx"].notna().all(), "uid join broken"

    emb_sub = emb_full[sub["emb_idx"].to_numpy()]
    text = sub.merge(
        sample[["uid", "title", "description_core_llm", "company_name_canonical"]],
        on="uid", how="left",
    )
    print(f"[load] AI subset n={len(sub)} "
          f"(2024={int((sub['period_bucket']=='2024').sum())}, "
          f"2026={int((sub['period_bucket']=='2026').sum())})")
    return sub, emb_sub, text


def umap_5d(emb: np.ndarray, seed: int) -> np.ndarray:
    from umap import UMAP
    reducer = UMAP(n_components=5, n_neighbors=15, min_dist=0.0,
                   metric="cosine", random_state=seed)
    return reducer.fit_transform(emb)


def hdbscan_fit(coords: np.ndarray, mcs: int) -> np.ndarray:
    import hdbscan
    return hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean").fit(coords).labels_


# ---------------------------------------------------------------------------
# Test 1 — density separation sweep
# ---------------------------------------------------------------------------

def density_sweep(coords: np.ndarray) -> pd.DataFrame:
    rows = []
    for mcs in MCS_SWEEP:
        lab = hdbscan_fit(coords, mcs)
        n_clusters = int(lab.max() + 1) if lab.max() >= 0 else 0
        noise = float((lab == -1).mean())
        sizes = pd.Series(lab[lab >= 0]).value_counts().sort_values(ascending=False)
        rows.append({
            "min_cluster_size": mcs,
            "n_clusters": n_clusters,
            "noise_rate": round(noise, 4),
            "largest_cluster": int(sizes.iloc[0]) if len(sizes) else 0,
            "second_cluster": int(sizes.iloc[1]) if len(sizes) > 1 else 0,
            "third_cluster": int(sizes.iloc[2]) if len(sizes) > 2 else 0,
        })
        print(f"[sweep] mcs={mcs}: n_clusters={n_clusters} noise={noise:.3f} "
              f"top sizes={sizes.head(5).tolist()}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 2 — silhouette + inter/intra ratio
# ---------------------------------------------------------------------------

def separation_metrics(coords_5d: np.ndarray, emb: np.ndarray,
                       labels: np.ndarray) -> dict:
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_distances

    out = {"silhouette_5d": None, "inter_intra_ratio_384d": None,
           "mean_inter_centroid_cosine": None, "mean_intra_nn_cosine": None}

    mask = labels >= 0
    unique = np.unique(labels[mask])
    if mask.sum() < 10 or len(unique) < 2:
        print("[metrics] insufficient clusters for silhouette/ratio")
        return out

    out["silhouette_5d"] = float(silhouette_score(coords_5d[mask], labels[mask]))

    emb_n = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
    centroids = []
    intra_nn = []
    rng = np.random.default_rng(RANDOM_STATE)
    for c in unique:
        idx = np.where(labels == c)[0]
        if len(idx) < 3:
            continue
        pts = emb_n[idx]
        cen = pts.mean(axis=0)
        centroids.append(cen / np.clip(np.linalg.norm(cen), 1e-9, None))
        # Mean nearest-neighbour cosine distance within the cluster,
        # sub-sampled for clusters larger than 400 points.
        if len(idx) > 400:
            pts = pts[rng.choice(len(idx), 400, replace=False)]
        d = cosine_distances(pts, pts)
        np.fill_diagonal(d, np.inf)
        intra_nn.append(float(d.min(axis=1).mean()))

    cen_mat = np.stack(centroids)
    inter = cosine_distances(cen_mat, cen_mat)
    np.fill_diagonal(inter, np.nan)
    out["mean_inter_centroid_cosine"] = float(np.nanmean(inter))
    out["mean_intra_nn_cosine"] = float(np.mean(intra_nn))
    out["inter_intra_ratio_384d"] = (
        out["mean_inter_centroid_cosine"] / out["mean_intra_nn_cosine"]
        if out["mean_intra_nn_cosine"] > 0 else float("nan")
    )
    print(f"[metrics] silhouette(5D)={out['silhouette_5d']:.3f} "
          f"ratio(384D)={out['inter_intra_ratio_384d']:.3f}")
    return out


# ---------------------------------------------------------------------------
# Test 3 — stability across seeds
# ---------------------------------------------------------------------------

def stability_across_seeds(emb: np.ndarray, mcs: int) -> dict:
    from sklearn.metrics import adjusted_rand_score
    seed_labels = {}
    for s in STABILITY_SEEDS:
        coords = umap_5d(emb, s)
        seed_labels[s] = hdbscan_fit(coords, mcs)
        print(f"[stab] seed={s} n_clusters={int(seed_labels[s].max()+1) if seed_labels[s].max()>=0 else 0}")
    aris = {}
    pairs = [(42, 1337), (42, 2026), (1337, 2026)]
    for a, b in pairs:
        aris[f"ari_{a}_vs_{b}"] = float(adjusted_rand_score(seed_labels[a], seed_labels[b]))
    aris["ari_mean"] = float(np.mean(list(aris.values())))
    print(f"[stab] {aris}")
    return aris


# ---------------------------------------------------------------------------
# Test 4a — per-cluster characterization (at primary mcs)
# ---------------------------------------------------------------------------

def characterize_clusters(text_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    from sklearn.feature_extraction.text import TfidfVectorizer

    cluster_ids = sorted([c for c in np.unique(labels) if c >= 0])
    if not cluster_ids:
        return pd.DataFrame()

    docs = [" ".join(text_df.loc[labels == c, "description_core_llm"]
                     .fillna("").astype(str).tolist())
            for c in cluster_ids]
    vec = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), min_df=1, max_df=1.0,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-/+]{1,}\b",
    )
    X = vec.fit_transform(docs)
    terms = np.array(vec.get_feature_names_out())

    rows = []
    for i, c in enumerate(cluster_ids):
        row = X[i].toarray().ravel()
        top_terms = terms[row.argsort()[::-1][:15]].tolist()
        cdf = text_df[labels == c]
        n_2024 = int((cdf["period_bucket"] == "2024").sum())
        n_2026 = int((cdf["period_bucket"] == "2026").sum())
        top_companies = (cdf["company_name_canonical"].dropna()
                         .value_counts().head(8).to_dict())
        rows.append({
            "cluster_id": int(c),
            "n_total": int(len(cdf)),
            "n_2024": n_2024,
            "n_2026": n_2026,
            "growth_ratio": round(n_2026 / n_2024, 2) if n_2024 > 0 else None,
            "top_terms": "; ".join(top_terms),
            "top_companies": "; ".join(f"{k} ({v})" for k, v in top_companies.items()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 4b — fine-grained theme growth (mcs=10 + regex themes)
# ---------------------------------------------------------------------------

def theme_growth(coords_5d: np.ndarray, text_df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.feature_extraction.text import TfidfVectorizer
    labels_fine = hdbscan_fit(coords_5d, 10)
    cluster_ids = sorted([c for c in np.unique(labels_fine) if c >= 0])

    if not cluster_ids:
        return pd.DataFrame()

    docs = [" ".join(text_df.loc[labels_fine == c, "description_core_llm"]
                     .fillna("").astype(str).tolist())
            for c in cluster_ids]
    vec = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), min_df=1, max_df=1.0,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-/+]{1,}\b",
    )
    X = vec.fit_transform(docs)
    terms = np.array(vec.get_feature_names_out())

    cluster_top = {}
    for i, c in enumerate(cluster_ids):
        row = X[i].toarray().ravel()
        cluster_top[c] = " ".join(terms[row.argsort()[::-1][:25]].tolist())

    def theme_of(cid: int) -> str:
        text = cluster_top[cid]
        for name, rx in THEME_HINTS:
            if rx.search(text):
                return name
        return "unlabeled"

    themed = pd.DataFrame({
        "period_bucket": text_df["period_bucket"].values,
        "theme": [theme_of(c) if c >= 0 else "noise" for c in labels_fine],
    })
    pivot = (themed.groupby(["theme", "period_bucket"]).size()
             .unstack(fill_value=0))
    for col in ("2024", "2026"):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["total"] = pivot["2024"] + pivot["2026"]
    pivot["growth_ratio"] = pivot.apply(
        lambda r: round(r["2026"] / r["2024"], 2) if r["2024"] > 0 else float("inf"),
        axis=1,
    )
    return pivot.sort_values("total", ascending=False).reset_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    sub, emb, text_df = load_ai_subset()

    print("\n[umap] fitting 5D UMAP (seed=42)...")
    coords = umap_5d(emb, RANDOM_STATE)

    # Test 1 — density sweep
    sweep = density_sweep(coords)
    sweep.to_csv(TABLES / "composite_B_v3llm_ai_subcluster_sweep.csv", index=False)

    # Primary clustering at chosen mcs
    primary_labels = hdbscan_fit(coords, PRIMARY_MCS)

    # Test 2 — separation metrics
    metrics = separation_metrics(coords, emb, primary_labels)

    # Test 3 — seed stability
    aris = stability_across_seeds(emb, PRIMARY_MCS)

    metrics_row = {**metrics, **aris, "primary_min_cluster_size": PRIMARY_MCS}
    pd.DataFrame([metrics_row]).to_csv(
        TABLES / "composite_B_v3llm_ai_subcluster_metrics.csv", index=False
    )

    # Test 4a — characterize the clusters found at primary mcs
    clusters = characterize_clusters(text_df, primary_labels)
    clusters.to_csv(
        TABLES / "composite_B_v3llm_ai_subcluster_clusters.csv", index=False
    )
    print("\n[char] primary-mcs clusters:")
    print(clusters[["cluster_id", "n_total", "n_2024", "n_2026",
                    "growth_ratio"]].to_string(index=False))

    # Test 4b — theme-level growth at fine granularity
    theme = theme_growth(coords, text_df)
    theme.to_csv(TABLES / "composite_B_v3llm_ai_theme_growth.csv", index=False)
    print("\n[theme] fine-grained growth by regex theme:")
    print(theme.to_string(index=False))

    # Summary
    summary = {
        "n_ai_subset": int(len(sub)),
        "n_2024": int((sub["period_bucket"] == "2024").sum()),
        "n_2026": int((sub["period_bucket"] == "2026").sum()),
        "hdbscan_sweep": sweep.to_dict(orient="records"),
        "primary_min_cluster_size": PRIMARY_MCS,
        "n_primary_clusters": int(len(clusters)),
        "silhouette_5d": metrics["silhouette_5d"],
        "inter_intra_ratio_384d": metrics["inter_intra_ratio_384d"],
        "stability_ari_mean": aris["ari_mean"],
        "stability_ari_pairs": {k: v for k, v in aris.items() if k != "ari_mean"},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(ARTIFACTS / "composite_B_v3llm_ai_subcluster_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[done] total runtime: {time.time() - t0:.1f}s")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("hdbscan_sweep",)}, indent=2, default=str))


if __name__ == "__main__":
    main()
