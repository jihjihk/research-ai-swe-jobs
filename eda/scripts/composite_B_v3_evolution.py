"""
Composite B v3-LLM — Role-landscape evolution, 2024 → 2026.

Rigorous re-run of v3 using the direct LLM classifier verdict
(`swe_classification_llm`) as the role filter, instead of the composite
rule-based flag (`is_swe OR is_swe_adjacent`). The composite flag pulled in
~10k postings the LLM classified as NOT_SWE via embedding-adjacent and regex
tiers (most visibly, ~400 building architects). Filtering on the LLM column
directly removes that contamination.

Three-axis change measurement, unchanged from v3:
  1. Compositional delta (share shifts per archetype family)
  2. Emergence index (share of 2026 postings unlike any 2024 posting)
  3. Within-archetype drift (c-TF-IDF term turnover inside stable families)

See eda/research_memos/composite_B_v3_plan.md for design rationale.

Run:
  ./.venv/bin/python eda/scripts/composite_B_v3_evolution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Remove script dir from sys.path BEFORE other imports to avoid
# eda/scripts/profile.py shadowing stdlib `profile`.
_THIS_DIR = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _THIS_DIR and p != ""]

import json
import time

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE = str(PROJECT_ROOT / "data" / "unified_core.parquet")
TABLES = PROJECT_ROOT / "eda" / "tables"
FIGURES = PROJECT_ROOT / "eda" / "figures"
ARTIFACTS = PROJECT_ROOT / "eda" / "artifacts"
CACHE = ARTIFACTS / "_composite_B_v3llm_cache"
for p in (TABLES, FIGURES, ARTIFACTS, CACHE):
    p.mkdir(parents=True, exist_ok=True)

SAMPLE_CACHE = CACHE / "sample.parquet"
EMBED_CACHE = CACHE / "embeddings.npy"
UIDS_CACHE = CACHE / "uids.npy"
UMAP2D_CACHE = CACHE / "umap2d.npy"
TOPICS_CACHE = CACHE / "topics_primary.npy"
PROBS_CACHE = CACHE / "probs_primary.npy"

# Canonical AI-vocab regex (matches S27_v2)
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
MIN_TOPIC_SIZE = 35
TARGET_FAMILIES = [12, 20, 30]   # sweep; primary = 20


# ---------------------------------------------------------------------------
# STEP 1 — Sample
# ---------------------------------------------------------------------------

def build_sample(con) -> pd.DataFrame:
    if SAMPLE_CACHE.exists():
        print(f"[sample] cache hit: {SAMPLE_CACHE}")
        return pd.read_parquet(SAMPLE_CACHE)

    print("[sample] building SWE + SWE-adjacent pooled sample (LLM filter)...")
    sql = f"""
      WITH base AS (
        SELECT uid, period,
               CASE WHEN period LIKE '2024%' THEN '2024' ELSE '2026' END AS period_bucket,
               CASE WHEN swe_classification_llm = 'SWE' THEN 'SWE'
                    WHEN swe_classification_llm = 'SWE_ADJACENT' THEN 'SWE-adjacent'
                    ELSE 'other' END AS role_group,
               title, description, description_core_llm,
               company_name_canonical, is_aggregator,
               seniority_3level, seniority_final, yoe_min_years_llm,
               metro_area,
               COALESCE(description_core_llm, description) AS text_for_embed
        FROM '{CORE}'
        WHERE swe_classification_llm IN ('SWE', 'SWE_ADJACENT')
          AND llm_classification_coverage = 'labeled'
          AND llm_extraction_coverage = 'labeled'
          AND is_english = true
          AND date_flag = 'ok'
          AND description_core_llm IS NOT NULL
          AND length(description_core_llm) >= {MIN_DESC_LEN}
      ),
      ranked AS (
        SELECT *,
               ROW_NUMBER() OVER (
                 PARTITION BY COALESCE(company_name_canonical, 'NA'),
                              period_bucket, role_group
                 ORDER BY HASH(uid)
               ) AS rn
        FROM base
      )
      SELECT * FROM ranked WHERE rn <= {CAP_PER_COMPANY}
    """
    df = con.execute(sql).df().reset_index(drop=True)
    print(f"[sample] n={len(df):,}")
    print(
        "[sample] breakdown:\n"
        + df.groupby(["role_group", "period_bucket"]).size()
          .rename("n").reset_index().to_string(index=False)
    )
    _txt = df["description_core_llm"].fillna(df["description"]).fillna("")
    df["ai_match"] = _txt.str.contains(AI_VOCAB_PATTERN, regex=True, na=False)
    df.to_parquet(SAMPLE_CACHE, index=False)
    return df


# ---------------------------------------------------------------------------
# STEP 2 — Embed
# ---------------------------------------------------------------------------

def embed_sample(df: pd.DataFrame) -> np.ndarray:
    if EMBED_CACHE.exists() and UIDS_CACHE.exists():
        cached_uids = np.load(UIDS_CACHE, allow_pickle=True)
        if len(cached_uids) == len(df) and (cached_uids == df["uid"].values).all():
            print(f"[embed] cache hit: {EMBED_CACHE}")
            return np.load(EMBED_CACHE)
        print("[embed] cache uid mismatch, recomputing...")

    from sentence_transformers import SentenceTransformer
    print("[embed] loading all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = df["text_for_embed"].fillna("").tolist()
    print(f"[embed] encoding {len(texts):,} docs (batch={EMBED_BATCH})...")
    t0 = time.time()
    emb = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"[embed] shape={emb.shape}, took {time.time() - t0:.1f}s")
    np.save(EMBED_CACHE, emb)
    np.save(UIDS_CACHE, df["uid"].values)
    return emb


# ---------------------------------------------------------------------------
# STEP 3 — Joint BERTopic fit
# ---------------------------------------------------------------------------

def build_topic_model(seed: int, min_topic_size: int = MIN_TOPIC_SIZE):
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0,
        metric="cosine", random_state=seed, low_memory=True,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size, min_samples=10,
        metric="euclidean", cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        ngram_range=(1, 2), stop_words="english",
        min_df=10, max_df=0.4,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-\+/]+\b",
    )
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=False,
    )


def fit_joint_primary(df: pd.DataFrame, emb: np.ndarray):
    if TOPICS_CACHE.exists() and PROBS_CACHE.exists():
        print(f"[joint] cache hit: {TOPICS_CACHE}")
        # Need to refit to get the model object for reduce_topics etc.
        # So cache stores topics only as a shortcut; always refit model.

    print(f"[joint] fitting BERTopic (seed={RANDOM_STATE})...")
    topic_model = build_topic_model(RANDOM_STATE)
    texts = df["text_for_embed"].fillna("").tolist()
    t0 = time.time()
    topics, probs = topic_model.fit_transform(texts, embeddings=emb)
    topics = np.array(topics)
    n_topics = len(set(topics)) - (1 if -1 in set(topics) else 0)
    noise_pct = (topics == -1).mean() * 100
    print(
        f"[joint] fit took {time.time() - t0:.1f}s | "
        f"n_topics={n_topics} | noise={noise_pct:.1f}%"
    )
    np.save(TOPICS_CACHE, topics)
    np.save(PROBS_CACHE, np.array(probs) if probs is not None else np.ones(len(topics)))
    return topic_model, topics


# ---------------------------------------------------------------------------
# STEP 4 — Stability across seeds
# ---------------------------------------------------------------------------

def stability_runs(df: pd.DataFrame, emb: np.ndarray):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    cache = CACHE / "stability_raw.parquet"
    if cache.exists():
        print(f"[stab] cache hit: {cache}")
        return pd.read_parquet(cache)

    texts = df["text_for_embed"].fillna("").tolist()
    all_topics = {}
    for seed in STABILITY_SEEDS:
        print(f"[stab] fitting seed={seed}...")
        tm = build_topic_model(seed)
        t0 = time.time()
        topics, _ = tm.fit_transform(texts, embeddings=emb)
        print(f"[stab]   took {time.time() - t0:.1f}s, n_topics={len(set(topics))}")
        all_topics[seed] = np.array(topics)

    rows = []
    seeds = list(all_topics.keys())
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a, b = all_topics[seeds[i]], all_topics[seeds[j]]
            rows.append({
                "seed_a": seeds[i], "seed_b": seeds[j],
                "ari": adjusted_rand_score(a, b),
                "nmi": normalized_mutual_info_score(a, b),
            })
    out = pd.DataFrame(rows)
    out.to_parquet(cache, index=False)
    print(out.to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# STEP 5 — Reduce to archetype families (12/20/30 sweep)
# ---------------------------------------------------------------------------

def reduce_and_characterize(topic_model, df: pd.DataFrame, emb: np.ndarray, target_k: int):
    """Reduce joint topics to target_k families, characterize each family."""
    import copy
    from sklearn.feature_extraction.text import CountVectorizer
    texts = df["text_for_embed"].fillna("").tolist()

    # BERTopic's reduce_topics mutates the model; work on a copy so we can sweep.
    tm = copy.deepcopy(topic_model)
    # Swap to a permissive vectorizer before reduce_topics. The original
    # (min_df=10, max_df=0.4) operates across topic-aggregated pseudo-docs during
    # c-TF-IDF recomputation; with only ~target_k classes those thresholds become
    # unsatisfiable. This vectorizer sees each class document (concatenation of
    # all postings in a topic) so raw term frequency is already high.
    tm.vectorizer_model = CountVectorizer(
        ngram_range=(1, 2), stop_words="english",
        min_df=1, max_df=1.0,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-\+/]+\b",
    )
    tm.reduce_topics(texts, nr_topics=target_k)
    new_topics = np.array(tm.topics_)

    info = tm.get_topic_info()
    rows = []
    for _, r in info.iterrows():
        tid = int(r["Topic"])
        if tid == -1:
            continue
        mask = new_topics == tid
        sub = df.loc[mask]
        n = int(mask.sum())
        n_2024 = int((sub["period_bucket"] == "2024").sum())
        n_2026 = int((sub["period_bucket"] == "2026").sum())
        n_swe = int((sub["role_group"] == "SWE").sum())
        n_adj = int((sub["role_group"] == "SWE-adjacent").sum())
        words = tm.get_topic(tid)
        top_terms = ", ".join(w for w, _ in words[:15]) if words else ""
        auto_name = "/".join(w.replace(" ", "_") for w, _ in (words[:3] if words else []))
        rows.append({
            "family_id": tid, "n": n,
            "n_2024": n_2024, "n_2026": n_2026,
            "growth_ratio": (n_2026 + 1) / (n_2024 + 1),
            "n_swe": n_swe, "n_swe_adjacent": n_adj,
            "share_swe": n_swe / max(n, 1),
            "ai_rate": float(sub["ai_match"].mean()) if n else float("nan"),
            "ai_rate_2024": float(sub.loc[sub["period_bucket"] == "2024", "ai_match"].mean())
                if n_2024 else float("nan"),
            "ai_rate_2026": float(sub.loc[sub["period_bucket"] == "2026", "ai_match"].mean())
                if n_2026 else float("nan"),
            "top_terms": top_terms,
            "auto_name": auto_name,
        })
    char = pd.DataFrame(rows).sort_values("n", ascending=False)
    return tm, new_topics, char


# ---------------------------------------------------------------------------
# STEP 6 — Per-period fits + alignment
# ---------------------------------------------------------------------------

def fit_period_and_align(df: pd.DataFrame, emb: np.ndarray,
                          joint_families: np.ndarray, joint_char: pd.DataFrame):
    """Fit BERTopic on 2024 and 2026 subsets separately; measure how well each
    joint-fit family maps to a period-specific topic (centroid cosine)."""
    results = {}
    texts = df["text_for_embed"].fillna("").fillna("").values
    for period in ("2024", "2026"):
        mask = (df["period_bucket"] == period).values
        sub_texts = texts[mask].tolist()
        sub_emb = emb[mask]
        print(f"[period] fitting BERTopic on {period}-only (n={mask.sum():,})...")
        tm = build_topic_model(RANDOM_STATE)
        t0 = time.time()
        topics_p, _ = tm.fit_transform(sub_texts, embeddings=sub_emb)
        topics_p = np.array(topics_p)
        print(f"[period]   took {time.time() - t0:.1f}s, n_topics={len(set(topics_p))}")

        # Centroids for each period-specific topic
        period_centroids = {}
        for t in set(topics_p):
            if t == -1:
                continue
            period_centroids[t] = sub_emb[topics_p == t].mean(axis=0)

        # Centroids for each joint family, restricted to this period's rows
        family_centroids = {}
        joint_sub = joint_families[mask]
        for fid in sorted(set(joint_families)):
            if fid == -1:
                continue
            m2 = joint_sub == fid
            if m2.sum() >= 10:
                family_centroids[fid] = sub_emb[m2].mean(axis=0)

        # Best-match cosine for each family
        rows = []
        for fid, fc in family_centroids.items():
            fc_n = fc / (np.linalg.norm(fc) + 1e-12)
            best_sim, best_t = -1.0, -1
            for pt, pc in period_centroids.items():
                pc_n = pc / (np.linalg.norm(pc) + 1e-12)
                sim = float(np.dot(fc_n, pc_n))
                if sim > best_sim:
                    best_sim, best_t = sim, pt
            rows.append({
                "family_id": fid, "period": period,
                "best_period_topic": best_t, "best_cosine": best_sim,
                "n_in_period": int((joint_sub == fid).sum()),
            })
        results[period] = pd.DataFrame(rows)
    align = pd.concat(results.values(), ignore_index=True)
    return align


# ---------------------------------------------------------------------------
# STEP 7 — Emergence index
# ---------------------------------------------------------------------------

def emergence_index(df: pd.DataFrame, emb: np.ndarray, joint_families: np.ndarray):
    """For each 2026 posting: cosine distance to its nearest 2024 neighbor.
    Threshold = 95th percentile of 2024→2024 nearest-neighbor distances (excluding self).
    Emergence rate per family = share of its 2026 postings above threshold."""
    from sklearn.neighbors import NearestNeighbors

    idx_2024 = np.where(df["period_bucket"].values == "2024")[0]
    idx_2026 = np.where(df["period_bucket"].values == "2026")[0]
    emb_2024 = emb[idx_2024]
    emb_2026 = emb[idx_2026]

    # Calibrate: 2024→2024 nearest neighbor (k=2 because self is nearest)
    print(f"[emerge] calibrating on 2024→2024 (n={len(idx_2024):,})...")
    nn_cal = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute",
                               n_jobs=-1).fit(emb_2024)
    d_cal, _ = nn_cal.kneighbors(emb_2024)
    d_cal = d_cal[:, 1]  # drop self
    threshold = float(np.percentile(d_cal, 95))
    print(f"[emerge] 95th-pct 2024→2024 nearest-neighbor distance = {threshold:.4f}")

    # Score: 2026 → nearest 2024
    print(f"[emerge] scoring 2026→2024 (n={len(idx_2026):,})...")
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute",
                           n_jobs=-1).fit(emb_2024)
    d_2026, _ = nn.kneighbors(emb_2026)
    d_2026 = d_2026[:, 0]

    emergent_mask = d_2026 > threshold
    print(f"[emerge] global 2026 emergent share: {emergent_mask.mean() * 100:.1f}%")

    # Per-family emergence + also per (family × role_group)
    rg_2026 = df["role_group"].values[idx_2026]
    fam_2026 = joint_families[idx_2026]
    rows = []
    for fid in sorted(set(joint_families)):
        if fid == -1:
            continue
        mask = fam_2026 == fid
        if mask.sum() == 0:
            continue
        rows.append({
            "family_id": fid,
            "n_2026": int(mask.sum()),
            "emergence_share": float(emergent_mask[mask].mean()),
            "median_distance_to_2024": float(np.median(d_2026[mask])),
            "p95_distance_to_2024": float(np.percentile(d_2026[mask], 95)),
            "emergence_share_swe": float(
                emergent_mask[mask & (rg_2026 == "SWE")].mean()
            ) if (mask & (rg_2026 == "SWE")).sum() else float("nan"),
            "emergence_share_adj": float(
                emergent_mask[mask & (rg_2026 == "SWE-adjacent")].mean()
            ) if (mask & (rg_2026 == "SWE-adjacent")).sum() else float("nan"),
        })
    emerge_df = pd.DataFrame(rows).sort_values("emergence_share", ascending=False)

    # Also save per-doc distances for figure work later
    per_doc = pd.DataFrame({
        "uid": df["uid"].values[idx_2026],
        "family_id": fam_2026,
        "role_group": rg_2026,
        "distance_to_nearest_2024": d_2026,
        "is_emergent": emergent_mask,
    })
    return emerge_df, per_doc, threshold


# ---------------------------------------------------------------------------
# STEP 8 — Within-archetype drift (c-TF-IDF term turnover)
# ---------------------------------------------------------------------------

def within_family_drift(df: pd.DataFrame, joint_families: np.ndarray,
                         char: pd.DataFrame, top_k: int = 8):
    """For the top-k families by volume, compute c-TF-IDF top terms on 2024 and
    2026 subsets separately, and report term entries/exits in the top 20."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import normalize

    top_families = char.sort_values("n", ascending=False).head(top_k)["family_id"].tolist()
    rows = []
    for fid in top_families:
        mask = joint_families == fid
        sub = df.loc[mask].copy()
        if (sub["period_bucket"] == "2024").sum() < 30 or \
           (sub["period_bucket"] == "2026").sum() < 30:
            continue

        def top_terms(texts):
            vec = CountVectorizer(
                ngram_range=(1, 2), stop_words="english",
                min_df=5, max_df=0.5,
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-\+/]+\b",
            )
            X = vec.fit_transform(texts)
            # c-TF-IDF: treat the corpus as one document per class
            tf = np.asarray(X.sum(axis=0)).flatten()
            # "class-based" normalization: log(1 + tf) * idf-ish
            terms = np.array(vec.get_feature_names_out())
            # rank by raw frequency within-class (c-TF-IDF requires multi-class;
            # here we just compare frequency ranks 2024 vs 2026 which is cleaner)
            order = np.argsort(tf)[::-1]
            return terms[order][:30].tolist()

        texts_2024 = sub.loc[sub["period_bucket"] == "2024", "text_for_embed"].fillna("").tolist()
        texts_2026 = sub.loc[sub["period_bucket"] == "2026", "text_for_embed"].fillna("").tolist()
        t24 = set(top_terms(texts_2024)[:20])
        t26 = set(top_terms(texts_2026)[:20])
        entered = t26 - t24
        exited = t24 - t26
        stable = t24 & t26
        rows.append({
            "family_id": fid,
            "n_2024": int((sub["period_bucket"] == "2024").sum()),
            "n_2026": int((sub["period_bucket"] == "2026").sum()),
            "entered_top20": ", ".join(sorted(entered)),
            "exited_top20": ", ".join(sorted(exited)),
            "stable_top20": ", ".join(sorted(stable)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# STEP 8b — Family-level stability ARI (refit seeds, reduce to target_k,
#            compute pairwise ARI on the reduced family labels)
# ---------------------------------------------------------------------------

def family_level_stability_ari(df: pd.DataFrame, emb: np.ndarray,
                                target_k: int) -> pd.DataFrame:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.feature_extraction.text import CountVectorizer
    cache = CACHE / f"family_stability_k{target_k}.parquet"
    if cache.exists():
        print(f"[fam_stab] cache hit: {cache}")
        return pd.read_parquet(cache)

    texts = df["text_for_embed"].fillna("").tolist()
    reduced = {}
    permissive_vec = CountVectorizer(
        ngram_range=(1, 2), stop_words="english",
        min_df=1, max_df=1.0,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-\+/]+\b",
    )
    for seed in STABILITY_SEEDS:
        print(f"[fam_stab] fitting + reducing seed={seed} to k={target_k}...")
        tm = build_topic_model(seed)
        t0 = time.time()
        tm.fit_transform(texts, embeddings=emb)
        tm.vectorizer_model = permissive_vec
        tm.reduce_topics(texts, nr_topics=target_k)
        reduced[seed] = np.array(tm.topics_)
        print(f"[fam_stab]   seed={seed} took {time.time() - t0:.1f}s")

    rows = []
    seeds = list(reduced.keys())
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a, b = reduced[seeds[i]], reduced[seeds[j]]
            rows.append({
                "target_k": target_k,
                "seed_a": seeds[i], "seed_b": seeds[j],
                "ari": adjusted_rand_score(a, b),
                "nmi": normalized_mutual_info_score(a, b),
            })
    out = pd.DataFrame(rows)
    out.to_parquet(cache, index=False)
    print(out.to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# STEP 8c — Split mega-cluster (Family 0) by AI-vocab regex match
# ---------------------------------------------------------------------------

def split_family_zero_by_ai(df: pd.DataFrame, families: np.ndarray) -> np.ndarray:
    """Return a string array: "0_AI" / "0_nonAI" for family==0, else str(family_id).

    The data-and-AI mega-cluster (Family 0 at k=30) blends pure-AI engineering
    content with data-science / ML vocabulary that embeddings can't separate.
    The existing AI_VOCAB_PATTERN regex gives a sharp, documentation-consistent
    binary split that we report alongside the joint-fit family id.
    """
    split = np.where(
        families == 0,
        np.where(df["ai_match"].values, "0_AI", "0_nonAI"),
        families.astype(str),
    )
    return split


def characterize_split(df: pd.DataFrame, split: np.ndarray) -> pd.DataFrame:
    """Recompute characterization at the split-family granularity."""
    rows = []
    for key in sorted(set(split)):
        if key == "-1":
            continue
        mask = split == key
        sub = df.loc[mask]
        n = int(mask.sum())
        n_2024 = int((sub["period_bucket"] == "2024").sum())
        n_2026 = int((sub["period_bucket"] == "2026").sum())
        n_swe = int((sub["role_group"] == "SWE").sum())
        n_adj = int((sub["role_group"] == "SWE-adjacent").sum())
        rows.append({
            "family_split": key,
            "n": n, "n_2024": n_2024, "n_2026": n_2026,
            "growth_ratio": (n_2026 + 1) / (n_2024 + 1),
            "n_swe": n_swe, "n_swe_adjacent": n_adj,
            "share_swe": n_swe / max(n, 1),
            "ai_rate_2024": float(sub.loc[sub["period_bucket"] == "2024", "ai_match"].mean())
                if n_2024 else float("nan"),
            "ai_rate_2026": float(sub.loc[sub["period_bucket"] == "2026", "ai_match"].mean())
                if n_2026 else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


# ---------------------------------------------------------------------------
# STEP 9 — 2D UMAP for shared map
# ---------------------------------------------------------------------------

def umap_2d(emb: np.ndarray) -> np.ndarray:
    if UMAP2D_CACHE.exists():
        print(f"[umap2d] cache hit: {UMAP2D_CACHE}")
        return np.load(UMAP2D_CACHE)
    from umap import UMAP
    print("[umap2d] fitting 2D UMAP for visualization...")
    t0 = time.time()
    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    metric="cosine", random_state=RANDOM_STATE, low_memory=True)
    xy = reducer.fit_transform(emb)
    print(f"[umap2d] took {time.time() - t0:.1f}s")
    np.save(UMAP2D_CACHE, xy)
    return xy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_overall = time.time()
    con = duckdb.connect()

    # 1-2. Sample + embed
    df = build_sample(con)
    emb = embed_sample(df)

    # 3. Primary joint fit
    topic_model, topics_primary = fit_joint_primary(df, emb)

    # 4. Stability (3 seeds)
    stab = stability_runs(df, emb)
    stab.to_csv(TABLES / "composite_B_v3llm_stability.csv", index=False)

    # 5. Reduce sweep (12, 20, 30); primary = 20
    family_results = {}
    for k in TARGET_FAMILIES:
        print(f"\n[reduce] target k={k}...")
        tm_k, fam_k, char_k = reduce_and_characterize(topic_model, df, emb, k)
        char_k.to_csv(TABLES / f"composite_B_v3llm_families_k{k}.csv", index=False)
        np.save(CACHE / f"families_k{k}.npy", fam_k)
        family_results[k] = (tm_k, fam_k, char_k)
        print(f"[reduce]   n_families={char_k.shape[0]}")

    # Primary analysis runs at k=30 for richer contraction-side resolution.
    # k=20 is retained in the sweep artifacts for comparison.
    PRIMARY_K = 30
    tm_p, joint_families, joint_char = family_results[PRIMARY_K]

    # 6. Per-period alignment (at primary k)
    align = fit_period_and_align(df, emb, joint_families, joint_char)
    align.to_csv(TABLES / f"composite_B_v3llm_alignment_k{PRIMARY_K}.csv", index=False)
    print("\n[align] alignment summary:")
    print(align.groupby("period")["best_cosine"].describe().to_string())

    # 7. Emergence index
    emerge_df, per_doc, threshold = emergence_index(df, emb, joint_families)
    emerge_df.to_csv(TABLES / f"composite_B_v3llm_emergence_k{PRIMARY_K}.csv", index=False)
    per_doc.to_parquet(ARTIFACTS / f"composite_B_v3llm_per_doc_emergence_k{PRIMARY_K}.parquet",
                        index=False)
    print("\n[emerge] top-5 by emergence share:")
    merged = emerge_df.merge(joint_char[["family_id", "n", "auto_name"]], on="family_id")
    print(merged.sort_values("emergence_share", ascending=False).head(10).to_string(index=False))

    # 8. Within-family drift
    drift = within_family_drift(df, joint_families, joint_char, top_k=10)
    drift.to_csv(TABLES / f"composite_B_v3llm_drift_k{PRIMARY_K}.csv", index=False)

    # 8b. Family-level stability ARI (primary k)
    fam_stab = family_level_stability_ari(df, emb, PRIMARY_K)
    fam_stab.to_csv(TABLES / f"composite_B_v3llm_family_stability_k{PRIMARY_K}.csv",
                     index=False)

    # 8c. Split mega-cluster (Family 0) by AI vocabulary
    family_split = split_family_zero_by_ai(df, joint_families)
    split_char = characterize_split(df, family_split)
    split_char.to_csv(TABLES / f"composite_B_v3llm_families_k{PRIMARY_K}_ai_split.csv",
                       index=False)
    print("\n[split] Family 0 AI-split:")
    print(split_char[split_char["family_split"].str.startswith("0")].to_string(index=False))

    # 9. 2D UMAP map
    xy = umap_2d(emb)

    # Final labels artifact: uid, role_group, period, primary topic,
    # family ids at k=20 and k=30, AI-split label, UMAP 2D coords.
    labels = pd.DataFrame({
        "uid": df["uid"].values,
        "role_group": df["role_group"].values,
        "period_bucket": df["period_bucket"].values,
        "ai_match": df["ai_match"].values,
        "primary_topic": topics_primary,
        "family_id_k20": family_results[20][1],
        "family_id_k30": family_results[30][1],
        "family_split_k30": family_split,
        "umap_x": xy[:, 0], "umap_y": xy[:, 1],
    })
    labels.to_parquet(ARTIFACTS / "composite_B_v3llm_labels.parquet", index=False)

    # Summary JSON
    summary = {
        "n_sample": int(len(df)),
        "role_period_breakdown": df.groupby(["role_group", "period_bucket"])
                                   .size().to_dict(),
        "n_primary_topics": int(len(set(topics_primary)) - (1 if -1 in set(topics_primary) else 0)),
        "noise_pct_primary": float((topics_primary == -1).mean() * 100),
        "topic_level_ari_mean": float(stab["ari"].mean()),
        "topic_level_ari_min": float(stab["ari"].min()),
        f"family_level_ari_mean_k{PRIMARY_K}": float(fam_stab["ari"].mean()),
        f"family_level_ari_min_k{PRIMARY_K}": float(fam_stab["ari"].min()),
        "emergence_threshold_cosine": float(threshold),
        "global_emergence_share_2026": float(per_doc["is_emergent"].mean()),
        "alignment_mean_cosine_2024": float(
            align[align["period"] == "2024"]["best_cosine"].mean()),
        "alignment_mean_cosine_2026": float(
            align[align["period"] == "2026"]["best_cosine"].mean()),
        "n_families_primary_k": int(joint_char.shape[0]),
        "primary_k": PRIMARY_K,
        "target_k_swept": TARGET_FAMILIES,
    }
    # Convert tuple keys to strings for JSON
    summary["role_period_breakdown"] = {
        f"{k[0]}|{k[1]}": int(v) for k, v in summary["role_period_breakdown"].items()
    }
    with open(ARTIFACTS / "composite_B_v3llm_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"[done] total runtime: {(time.time() - t_overall) / 60:.1f} min")
    print("=" * 60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
