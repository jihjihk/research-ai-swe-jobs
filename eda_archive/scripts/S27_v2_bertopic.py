"""
S27 v2 — BERTopic rerun on unified_core SWE postings.

Executes T09 spec (docs/task-reference-exploration.md:534-581) on SWE postings
in data/unified_core.parquet for composite article B (role landscape).

Outputs:
  eda/artifacts/composite_B_archetype_labels.parquet  -- (uid, archetype_id, archetype_name, prob)
  eda/tables/S27_v2_*.csv                             -- characterization tables
  eda/figures/S27_v2_*.png                            -- UMAP visualizations

Run:
  ./.venv/bin/python eda/scripts/S27_v2_bertopic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Remove the script directory from sys.path BEFORE other imports — otherwise
# eda/scripts/profile.py shadows the stdlib `profile` used by torch._dynamo.
_THIS_DIR = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _THIS_DIR and p != ""]

import json
import os
import re
import time

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE = str(PROJECT_ROOT / "data" / "unified_core.parquet")
TABLES = PROJECT_ROOT / "eda" / "tables"
FIGURES = PROJECT_ROOT / "eda" / "figures"
ARTIFACTS = PROJECT_ROOT / "eda" / "artifacts"
CACHE = PROJECT_ROOT / "eda" / "artifacts" / "_s27v2_cache"
for p in (TABLES, FIGURES, ARTIFACTS, CACHE):
    p.mkdir(parents=True, exist_ok=True)

EMBED_CACHE = CACHE / "embeddings.npy"
UIDS_CACHE = CACHE / "uids.npy"
SAMPLE_CACHE = CACHE / "sample.parquet"

# Canonical AI-vocab regex from eda/scripts/S27_role_landscape.py
AI_VOCAB_PATTERN = (
    r"(?i)\b(llm|gpt|chatgpt|claude|copilot|openai|anthropic|gemini|bard|mistral|"
    r"llama|large\ language\ model|generative\ ai|genai|gen\ ai|foundation\ model|"
    r"transformer\ model|ai\ agent|agentic|ai\-powered|ai\ tooling|ai\-assisted|"
    r"rag|retrieval\ augmented|vector\ database|vector\ store|embedding\ model|"
    r"prompt\ engineering|prompt\ engineer|ml\ ops|mlops|llmops|cursor\ ide|"
    r"windsurf\ ide|github\ copilot)\b"
)

CAP_PER_COMPANY = 30
MIN_DESC_LEN = 200
EMBED_BATCH = 256
RANDOM_STATE = 42
STABILITY_SEEDS = (42, 1337, 2026)


# ---------------------------------------------------------------------------
# STEP 1 — Sample
# ---------------------------------------------------------------------------

def build_sample(con) -> pd.DataFrame:
    if SAMPLE_CACHE.exists():
        print(f"Loading cached sample from {SAMPLE_CACHE}")
        return pd.read_parquet(SAMPLE_CACHE)

    print("Building SWE sample (company-capped at 30 per company per period)...")
    sql = f"""
      WITH base AS (
        SELECT uid, period,
               CASE WHEN period LIKE '2024%' THEN '2024' ELSE '2026' END AS period_bucket,
               title, description, description_core_llm,
               company_name_canonical, is_aggregator,
               seniority_3level, seniority_final, yoe_min_years_llm,
               metro_area,
               COALESCE(description_core_llm, description) AS text_for_embed
        FROM '{CORE}'
        WHERE is_swe = true
          AND is_english = true
          AND date_flag = 'ok'
          AND llm_extraction_coverage = 'labeled'
          AND description_core_llm IS NOT NULL
          AND length(description_core_llm) >= {MIN_DESC_LEN}
      ),
      ranked AS (
        SELECT *,
               ROW_NUMBER() OVER (
                 PARTITION BY COALESCE(company_name_canonical, 'NA'), period_bucket
                 ORDER BY HASH(uid)
               ) AS rn
        FROM base
      )
      SELECT *
      FROM ranked
      WHERE rn <= {CAP_PER_COMPANY}
    """
    df = con.execute(sql).df()
    df = df.reset_index(drop=True)
    print(f"Sample: {len(df)} rows | per period: {df['period_bucket'].value_counts().to_dict()}")
    # Per methodology protocol, AI-vocab matching uses description_core_llm
    # (COALESCE to raw on the rare unlabeled rows).
    _txt = df["description_core_llm"].fillna(df["description"]).fillna("")
    df["ai_match"] = _txt.str.contains(AI_VOCAB_PATTERN, regex=True, na=False)
    df.to_parquet(SAMPLE_CACHE, index=False)
    return df


# ---------------------------------------------------------------------------
# STEP 2 — Embed
# ---------------------------------------------------------------------------

def embed_sample(df: pd.DataFrame) -> np.ndarray:
    if EMBED_CACHE.exists() and UIDS_CACHE.exists():
        print(f"Loading cached embeddings from {EMBED_CACHE}")
        cached_uids = np.load(UIDS_CACHE, allow_pickle=True)
        if len(cached_uids) == len(df) and (cached_uids == df["uid"].values).all():
            return np.load(EMBED_CACHE)
        print("Cache uid mismatch, recomputing...")

    from sentence_transformers import SentenceTransformer
    print("Loading sentence-transformers all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = df["text_for_embed"].fillna("").tolist()
    print(f"Embedding {len(texts)} docs in batches of {EMBED_BATCH}...")
    t0 = time.time()
    emb = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"Embedding shape {emb.shape}, took {time.time() - t0:.1f}s")
    np.save(EMBED_CACHE, emb)
    np.save(UIDS_CACHE, df["uid"].values)
    return emb


# ---------------------------------------------------------------------------
# STEP 3 — BERTopic fit (primary) + NMF (comparison)
# ---------------------------------------------------------------------------

def fit_bertopic(df: pd.DataFrame, emb: np.ndarray, seed: int, min_topic_size: int = 35):
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
        low_memory=True,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=10,
        max_df=0.4,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-\+/]+\b",
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=True,
    )
    texts = df["text_for_embed"].fillna("").tolist()
    print(f"Fitting BERTopic (seed={seed}, min_topic_size={min_topic_size})...")
    t0 = time.time()
    topics, probs = topic_model.fit_transform(texts, embeddings=emb)
    print(f"  BERTopic fit took {time.time() - t0:.1f}s | n_topics = {len(set(topics))}")
    return topic_model, np.array(topics), probs


def fit_nmf(df: pd.DataFrame, k: int = 20):
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer

    print(f"Fitting NMF (k={k}) on TF-IDF...")
    texts = df["text_for_embed"].fillna("").tolist()
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.4, min_df=20,
        max_features=30000,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-\+/]+\b",
    )
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())
    t0 = time.time()
    nmf = NMF(n_components=k, random_state=RANDOM_STATE, init="nndsvd", max_iter=400)
    W = nmf.fit_transform(X)
    print(f"  NMF fit took {time.time() - t0:.1f}s")
    assignments = np.argmax(W, axis=1)
    top_terms = []
    H = nmf.components_
    for i in range(k):
        top_idx = np.argsort(H[i])[::-1][:15]
        top_terms.append(terms[top_idx].tolist())
    return assignments, top_terms


# ---------------------------------------------------------------------------
# Characterization
# ---------------------------------------------------------------------------

def label_from_topwords(words):
    """Generate a compact readable name from top words."""
    return "/".join(w.replace(" ", "_") for w in words[:3] if w)


def characterize(df: pd.DataFrame, topic_model, topics: np.ndarray) -> pd.DataFrame:
    topic_info = topic_model.get_topic_info()
    rows = []
    for tid in sorted(set(topics)):
        mask = topics == tid
        sub = df.loc[mask]
        top = topic_model.get_topic(tid)
        top_words = [w for w, _ in top] if top else []
        name = "noise/outlier" if tid == -1 else label_from_topwords(top_words)
        n = len(sub)
        n_2024 = int((sub["period_bucket"] == "2024").sum())
        n_2026 = int((sub["period_bucket"] == "2026").sum())
        ai_rate = float(sub["ai_match"].mean()) if n else float("nan")
        # seniority breakdown
        sen = sub["seniority_3level"].value_counts(normalize=True).to_dict()
        rows.append({
            "topic_id": int(tid),
            "name": name,
            "n": int(n),
            "n_2024": n_2024,
            "n_2026": n_2026,
            "growth_ratio": (n_2026 + 1) / (n_2024 + 1),
            "ai_rate": ai_rate,
            "median_yoe": float(sub["yoe_min_years_llm"].median()) if sub["yoe_min_years_llm"].notna().any() else float("nan"),
            "junior_share": float(sen.get("junior", 0.0)),
            "mid_share": float(sen.get("mid", 0.0)),
            "senior_share": float(sen.get("senior", 0.0)),
            "top_words": ", ".join(top_words[:10]),
            "top_firms": ", ".join(sub["company_name_canonical"].value_counts().head(5).index.tolist()),
            "top_metros": ", ".join(
                str(m) for m in sub["metro_area"].value_counts().head(3).index.tolist()
            ),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def exemplars(df: pd.DataFrame, topics: np.ndarray, n_per: int = 5) -> pd.DataFrame:
    out = []
    for tid in sorted(set(topics)):
        if tid == -1:
            continue
        mask = topics == tid
        sub = df.loc[mask].head(n_per)
        for _, r in sub.iterrows():
            desc = (r.get("description_core_llm") or r.get("description") or "")[:400]
            out.append({
                "topic_id": int(tid),
                "uid": r["uid"],
                "title": r["title"],
                "company": r["company_name_canonical"],
                "period": r["period_bucket"],
                "seniority": r["seniority_3level"],
                "desc_snippet": desc.replace("\n", " "),
            })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

def stability_ari(df: pd.DataFrame, emb: np.ndarray, seeds=STABILITY_SEEDS) -> pd.DataFrame:
    from sklearn.metrics import adjusted_rand_score
    runs = {}
    for s in seeds:
        _, topics, _ = fit_bertopic(df, emb, seed=s)
        runs[s] = topics
    rows = []
    seed_list = list(seeds)
    for i, a in enumerate(seed_list):
        for b in seed_list[i + 1:]:
            ari = adjusted_rand_score(runs[a], runs[b])
            rows.append({"seed_a": a, "seed_b": b, "ari": ari})
    return pd.DataFrame(rows), runs


# ---------------------------------------------------------------------------
# v9 comparison
# ---------------------------------------------------------------------------

def compare_to_v9(df: pd.DataFrame, topics: np.ndarray) -> dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    v9 = pd.read_parquet(PROJECT_ROOT / "exploration-archive/v9_final_opus_47/artifacts/shared/swe_archetype_labels.parquet")
    df_ = df[["uid"]].copy()
    df_["our_topic"] = topics
    joined = df_.merge(v9, on="uid", how="inner")
    if len(joined) == 0:
        return {"n_overlap": 0}
    ari = adjusted_rand_score(joined["archetype"], joined["our_topic"])
    nmi = normalized_mutual_info_score(joined["archetype"], joined["our_topic"])
    # Crosstab saved
    ct = pd.crosstab(joined["archetype_name"], joined["our_topic"])
    return {"n_overlap": len(joined), "ari": ari, "nmi": nmi, "crosstab": ct}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def umap_2d_plot(emb: np.ndarray, seed: int = RANDOM_STATE) -> np.ndarray:
    cache = CACHE / "umap2d.npy"
    if cache.exists():
        return np.load(cache)
    from umap import UMAP
    print("Fitting 2D UMAP for visualization...")
    t0 = time.time()
    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                   metric="cosine", random_state=seed, low_memory=True)
    xy = reducer.fit_transform(emb)
    print(f"  UMAP 2D took {time.time() - t0:.1f}s")
    np.save(cache, xy)
    return xy


def plot_umap(xy: np.ndarray, labels: np.ndarray, title: str, out_path: Path,
              discrete: bool = True):
    fig, ax = plt.subplots(figsize=(9, 7))
    if discrete:
        unique = sorted(set(labels))
        cmap = plt.colormaps.get_cmap("tab20")
        for i, lab in enumerate(unique):
            m = labels == lab
            ax.scatter(xy[m, 0], xy[m, 1], s=2, alpha=0.5, c=[cmap(i % 20)],
                       label=str(lab) if len(unique) <= 25 else None)
        if len(unique) <= 25:
            ax.legend(markerscale=3, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    else:
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=2, alpha=0.5, c=labels, cmap="viridis")
        plt.colorbar(sc, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    con = duckdb.connect()
    df = build_sample(con)
    emb = embed_sample(df)

    # Primary BERTopic run (seed = RANDOM_STATE)
    topic_model, topics, probs = fit_bertopic(df, emb, seed=RANDOM_STATE)

    char = characterize(df, topic_model, topics)
    char.to_csv(TABLES / "S27_v2_bertopic_topics.csv", index=False)
    print("Wrote S27_v2_bertopic_topics.csv")
    print(char.head(20).to_string(index=False))

    ex = exemplars(df, topics, n_per=5)
    ex.to_csv(TABLES / "S27_v2_bertopic_exemplars.csv", index=False)

    # Stability runs (include primary seed + 2 more)
    print("\n=== Stability runs ===")
    stab_df, runs = stability_ari(df, emb, seeds=STABILITY_SEEDS)
    stab_df.to_csv(TABLES / "S27_v2_stability_ari.csv", index=False)
    print(stab_df)

    # Compare to v9 T09
    print("\n=== Compare to v9 T09 ===")
    cmp = compare_to_v9(df, topics)
    print(f"overlap uids: {cmp['n_overlap']}, ARI = {cmp.get('ari'):.3f}, NMI = {cmp.get('nmi'):.3f}")
    if "crosstab" in cmp:
        cmp["crosstab"].to_csv(TABLES / "S27_v2_v9_crosstab.csv")
    pd.DataFrame([{"n_overlap": cmp["n_overlap"],
                   "ari_vs_v9": cmp.get("ari", float("nan")),
                   "nmi_vs_v9": cmp.get("nmi", float("nan"))}]).to_csv(
        TABLES / "S27_v2_v9_comparison.csv", index=False
    )

    # NMF comparison
    print("\n=== NMF (k=20) ===")
    nmf_labels, nmf_terms = fit_nmf(df, k=20)
    nmf_rows = []
    for i, words in enumerate(nmf_terms):
        mask = nmf_labels == i
        sub = df.loc[mask]
        n_2024 = int((sub["period_bucket"] == "2024").sum())
        n_2026 = int((sub["period_bucket"] == "2026").sum())
        nmf_rows.append({
            "nmf_topic": i,
            "n": int(mask.sum()),
            "n_2024": n_2024,
            "n_2026": n_2026,
            "growth_ratio": (n_2026 + 1) / (n_2024 + 1),
            "ai_rate": float(sub["ai_match"].mean()) if len(sub) else float("nan"),
            "top_terms": ", ".join(words[:10]),
        })
    nmf_df = pd.DataFrame(nmf_rows).sort_values("n", ascending=False)
    nmf_df.to_csv(TABLES / "S27_v2_nmf_topics.csv", index=False)
    print(nmf_df.head(20).to_string(index=False))

    # BERTopic vs NMF alignment (ARI)
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari_bt_nmf = adjusted_rand_score(topics, nmf_labels)
    nmi_bt_nmf = normalized_mutual_info_score(topics, nmf_labels)
    pd.DataFrame([{"ari_bertopic_vs_nmf": ari_bt_nmf,
                   "nmi_bertopic_vs_nmf": nmi_bt_nmf}]).to_csv(
        TABLES / "S27_v2_method_alignment.csv", index=False
    )
    print(f"BERTopic vs NMF: ARI = {ari_bt_nmf:.3f}, NMI = {nmi_bt_nmf:.3f}")

    # Save archetype labels artifact
    labels_df = df[["uid"]].copy()
    labels_df["archetype_id"] = topics
    # Probability per doc — when calculate_probabilities=False, BERTopic returns
    # membership strength from HDBSCAN via probabilities_ if available
    try:
        prob_vec = topic_model.hdbscan_model.probabilities_
    except Exception:
        prob_vec = np.ones(len(topics))
    labels_df["prob"] = prob_vec
    name_map = {r["topic_id"]: r["name"] for _, r in char.iterrows()}
    labels_df["archetype_name"] = labels_df["archetype_id"].map(name_map)
    labels_df.to_parquet(ARTIFACTS / "composite_B_archetype_labels.parquet", index=False)
    print(f"Wrote {ARTIFACTS / 'composite_B_archetype_labels.parquet'}")

    # Plots: UMAP 2D colored by archetype / period / seniority
    xy = umap_2d_plot(emb)
    plot_umap(xy, topics, "S27 v2 — UMAP 2D, colored by BERTopic archetype",
              FIGURES / "S27_v2_umap_archetype.png", discrete=True)
    plot_umap(xy, df["period_bucket"].values, "S27 v2 — UMAP 2D, colored by period",
              FIGURES / "S27_v2_umap_period.png", discrete=True)
    sen_map = {"junior": 0, "mid": 1, "senior": 2}
    sen_vals = df["seniority_3level"].map(sen_map).fillna(-1).astype(int).values
    plot_umap(xy, sen_vals, "S27 v2 — UMAP 2D, colored by seniority (0=junior,1=mid,2=senior,-1=unknown)",
              FIGURES / "S27_v2_umap_seniority.png", discrete=True)

    # Top growers / shrinkers bar chart
    top_growers = char[char["topic_id"] != -1].sort_values("growth_ratio", ascending=False).head(10)
    top_shrinkers = char[char["topic_id"] != -1].sort_values("growth_ratio", ascending=True).head(8)
    mm = pd.concat([top_growers, top_shrinkers])
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2ca02c" if g > 1 else "#d62728" for g in mm["growth_ratio"]]
    y_labels = [f"{r.topic_id}: {r['name'][:40]}" for _, r in mm.iterrows()]
    ax.barh(y_labels, mm["growth_ratio"], color=colors)
    ax.axvline(1.0, color="black", ls="--", linewidth=0.7)
    ax.set_xlabel("Growth ratio 2026/2024")
    ax.set_title("S27 v2 — BERTopic archetype growth/decline")
    fig.tight_layout()
    fig.savefig(FIGURES / "S27_v2_bertopic_movers.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Summary dump
    summary = {
        "n_sample": int(len(df)),
        "n_topics": int(char["topic_id"].nunique()),
        "n_noise": int((topics == -1).sum()),
        "noise_pct": float((topics == -1).mean() * 100),
        "stability_ari": stab_df["ari"].tolist(),
        "stability_ari_mean": float(stab_df["ari"].mean()),
        "ari_vs_v9": cmp.get("ari"),
        "nmi_vs_v9": cmp.get("nmi"),
        "n_overlap_v9": cmp.get("n_overlap"),
        "ari_bertopic_vs_nmf": ari_bt_nmf,
    }
    (ARTIFACTS / "composite_B_bertopic_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
