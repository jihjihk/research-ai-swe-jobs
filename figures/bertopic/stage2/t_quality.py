"""
T-quality (Stage 2): coherence + diversity (§7.8), silhouette + size
distribution (§7.9), honest noise rate before/after `reduce_outliers`
(§7.10), and gpt-5.4-mini cross-model naming (§7.11).

Produces:
- `data/bertopic/topic_quality.parquet`
  (topic_id, n_members, npmi, umass, c_v, silhouette)
- `data/bertopic/topic_info_with_naming.parquet`
  topic_info + gpt54mini_label / gpt54mini_confidence / gpt54mini_alternative
  / exact_match / label_cosine

Standalone — does not import from any "utils" module. Reads only frozen
Stage 1 artifacts; does not refit BERTopic.

Hash bundle (verified before any work):
  model_hash            d51f15e6...e509415   figures/bertopic/intermediate/raw_fit.bertopic
  sample_hash           6719a025...cfeb265   figures/bertopic/intermediate/sample_a.parquet
  embeddings_cache_hash 29d77bf9...c3b479b27  data/bertopic/embeddings_cache.npy
  assignments_hash      a03bc515...c6808ab82  data/bertopic/assignments.parquet
  config_hash           bef20ab2...a01a07bce8 (see stage1_freeze.json)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import httpx
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.bertopic import config  # noqa: E402
from figures.bertopic.stage1 import pipeline  # noqa: E402
from figures.bertopic.stage1.naming import propose_label  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INTERIM = REPO_ROOT / "figures" / "bertopic" / "intermediate"
DATA_DIR = REPO_ROOT / "data" / "bertopic"
TOPIC_QUALITY_PATH = DATA_DIR / "topic_quality.parquet"
TOPIC_INFO_NAMING_PATH = DATA_DIR / "topic_info_with_naming.parquet"
FREEZE_PATH = INTERIM / "stage1_freeze.json"

EXPECTED_HASHES = {
    "model_hash": "d51f15e613f62b221139503bc84e6d3757689aac5e07979beb6ed3dbce509415",
    "sample_hash": "6719a0250fbfcb630dad117b409d441697d493b209b219e1c9d08b09acfeb265",
    "embeddings_cache_hash": "29d77bf9e24e6250d7b303a17fb22b80b9575a09a46d88c9dbd5d75c3b479b27",
    "assignments_hash": "a03bc515094050996338094f28851126b8c1f07f7f3b26d2f678f6cb6808ab82",
    "config_hash": "bef20ab2916ad72bd87aaefb0d18ba13644f9989ddd8e9bad4eac2b01a07bce8",
}

ARTIFACT_PATHS = {
    "model_hash": REPO_ROOT / "figures/bertopic/intermediate/raw_fit.bertopic",
    "sample_hash": REPO_ROOT / "figures/bertopic/intermediate/sample_a.parquet",
    "embeddings_cache_hash": REPO_ROOT / "data/bertopic/embeddings_cache.npy",
    "assignments_hash": REPO_ROOT / "data/bertopic/assignments.parquet",
}

HEADLINE_K = 10


# ---------------------------------------------------------------------------
# Hash verification
# ---------------------------------------------------------------------------

def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hashes() -> None:
    for key, path in ARTIFACT_PATHS.items():
        actual = _file_hash(path)
        expected = EXPECTED_HASHES[key]
        if actual != expected:
            raise RuntimeError(
                f"hash mismatch on {key}: got {actual}, expected {expected}"
            )
        print(f"  {key}: OK  ({path})")
    # config_hash from frozen JSON
    with FREEZE_PATH.open() as fh:
        freeze = json.load(fh)
    if freeze["config_hash"] != EXPECTED_HASHES["config_hash"]:
        raise RuntimeError(
            f"config_hash mismatch in stage1_freeze.json: "
            f"{freeze['config_hash']} vs {EXPECTED_HASHES['config_hash']}"
        )
    print(f"  config_hash: OK  ({FREEZE_PATH})")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sample_a_with_assignments() -> pd.DataFrame:
    """uid, description_core_llm, topic_id (headline K=10), is_outlier."""
    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")
    df = con.execute(
        """
        SELECT s.uid, s.description_core_llm, a.topic_id, a.is_outlier
        FROM 'figures/bertopic/intermediate/sample_a.parquet' s
        JOIN 'data/bertopic/assignments.parquet' a USING (uid)
        ORDER BY s.uid
        """
    ).fetchdf()
    return df


def load_embeddings_for_uids(uids: list[str]) -> np.ndarray:
    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")
    idx = con.execute(
        "SELECT key, row_index FROM 'data/bertopic/embeddings_cache.index.parquet' "
        "WHERE kind = 'posting'"
    ).fetchdf()
    key_to_row = dict(zip(idx["key"].tolist(), idx["row_index"].tolist()))
    rows = np.array([key_to_row[u] for u in uids], dtype=np.int64)
    matrix = np.load(REPO_ROOT / "data/bertopic/embeddings_cache.npy", mmap_mode="r")
    return np.asarray(matrix[rows])


# ---------------------------------------------------------------------------
# §7.10 honest noise rate
# ---------------------------------------------------------------------------

def section_7_10_noise(
    *, df: pd.DataFrame, embeddings: np.ndarray
) -> dict[str, float]:
    """Compute noise rate before and after reduce_outliers.

    'Before' = noise rate at headline K=10 in the saved assignments
    (reduce_topics doesn't move -1 docs). 'After' = pretend
    reduce_outliers(strategy='embeddings') was applied to the headline-K
    layout — every -1 gets reassigned, so noise rate goes to 0.
    But the spec wants the *real* call: load the model, reduce to K=10,
    then call reduce_outliers and report the resulting share that BERTopic
    leaves outside any topic.
    """
    n = len(df)
    n_outliers_before = int(df["is_outlier"].sum())
    noise_before = n_outliers_before / n

    # Load a fresh BERTopic model and reduce to K=10, then run
    # reduce_outliers(strategy='embeddings').
    docs = df["description_core_llm"].tolist()
    model = pipeline.load_topic_model_for_reduce(config.RAW_FIT_PATH)
    model.reduce_topics(docs, nr_topics=HEADLINE_K)

    # Sanity check: reduced topic counts should match assignments.
    reduced_topics = np.asarray(model.topics_, dtype=np.int64)
    if len(reduced_topics) != n:
        raise RuntimeError(
            f"reduced topic length {len(reduced_topics)} != sample size {n}"
        )
    n_outliers_reduced = int(np.sum(reduced_topics == -1))
    if n_outliers_reduced != n_outliers_before:
        # Not fatal — log and continue with the BERTopic-internal labels.
        print(
            f"  WARN: reduce_topics produced {n_outliers_reduced} -1 docs vs "
            f"{n_outliers_before} in assignments.parquet; difference = "
            f"{abs(n_outliers_reduced - n_outliers_before)}"
        )

    new_topics = model.reduce_outliers(
        docs, reduced_topics.tolist(),
        strategy="embeddings",
        embeddings=embeddings,
    )
    new_topics = np.asarray(new_topics, dtype=np.int64)
    n_outliers_after = int(np.sum(new_topics == -1))
    noise_after = n_outliers_after / n

    return {
        "n": n,
        "n_outliers_before": n_outliers_before,
        "noise_rate_before": noise_before,
        "n_outliers_after": n_outliers_after,
        "noise_rate_after": noise_after,
    }


# ---------------------------------------------------------------------------
# §7.9 silhouette + size distribution
# ---------------------------------------------------------------------------

def section_7_9_silhouette_and_sizes(
    *, df: pd.DataFrame
) -> tuple[dict[int, float], dict[str, float], dict[str, Any]]:
    """Silhouette in 5-D UMAP space at headline K=10. Use the saved
    BERTopic raw fit's UMAP embedding (umap_model.embedding_) — same
    seed=42, same hyperparameters."""
    from sklearn.metrics import silhouette_samples

    # Pull UMAP-reduced 5-D embeddings from the saved raw fit. Order is the
    # same as `topics_` and as Sample A sorted by uid (verified upstream).
    model = pipeline.load_topic_model(config.RAW_FIT_PATH)
    umap_emb = np.asarray(model.umap_model.embedding_)
    if umap_emb.shape != (len(df), config.UMAP_N_COMPONENTS):
        raise RuntimeError(
            f"UMAP embedding shape {umap_emb.shape} disagrees with sample size "
            f"({len(df)}, {config.UMAP_N_COMPONENTS})"
        )

    topics = df["topic_id"].to_numpy()
    is_outlier = df["is_outlier"].to_numpy()

    # Silhouette excludes outliers — they are not a real cluster.
    mask = ~is_outlier
    sil_emb = umap_emb[mask]
    sil_topics = topics[mask]

    # silhouette_samples is fine on ~40k points.
    samples = silhouette_samples(sil_emb, sil_topics, metric="euclidean")
    overall = float(samples.mean())

    per_cluster: dict[int, float] = {}
    for cid in sorted(np.unique(sil_topics).tolist()):
        per_cluster[int(cid)] = float(samples[sil_topics == cid].mean())

    # Cluster size distribution at headline K=10 (excluding -1).
    sizes = (
        df.loc[~is_outlier, "topic_id"]
        .value_counts()
        .sort_index()
        .to_numpy()
    )
    size_stats = {
        "n_clusters": int(len(sizes)),
        "median": float(np.median(sizes)),
        "iqr_q1": float(np.percentile(sizes, 25)),
        "iqr_q3": float(np.percentile(sizes, 75)),
        "p5": float(np.percentile(sizes, 5)),
        "p95": float(np.percentile(sizes, 95)),
        "min": int(sizes.min()),
        "max": int(sizes.max()),
        "largest_share": float(sizes.max() / len(df)),
    }

    return per_cluster, {"overall_mean": overall}, size_stats


# ---------------------------------------------------------------------------
# §7.8 coherence + diversity
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(config.TOKEN_PATTERN)


def _tokenize(doc: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(doc or "")]


def section_7_8_coherence(
    *, df: pd.DataFrame, topic_info: pd.DataFrame
) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    """NPMI, UMass, C_v on top-10 c-TF-IDF terms per cluster, with
    description_core_llm Sample A as the reference corpus."""
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel

    # Tokenize Sample A (reference corpus) once.
    print("  tokenizing Sample A reference corpus…")
    docs_tok = [_tokenize(d) for d in df["description_core_llm"].tolist()]
    dictionary = Dictionary(docs_tok)
    print(f"    n_docs={len(docs_tok)} dictionary={len(dictionary)}")

    # Topics: top-10 c-TF-IDF terms per cluster from topic_info. Each
    # multi-word phrase ("software engineering") is split into single-word
    # tokens for coherence — gensim's coherence operates on unigrams in the
    # dictionary, so we keep tokens present in the dictionary.
    raw_topics: dict[int, list[str]] = {
        int(row["topic_id"]): list(row["top_words"])
        for _, row in topic_info.iterrows()
    }

    coh_topics: dict[int, list[str]] = {}
    for cid, words in raw_topics.items():
        cleaned: list[str] = []
        seen: set[str] = set()
        for phrase in words:
            for tok in _tokenize(phrase):
                if tok in dictionary.token2id and tok not in seen:
                    cleaned.append(tok)
                    seen.add(tok)
                if len(cleaned) >= 10:
                    break
            if len(cleaned) >= 10:
                break
        coh_topics[cid] = cleaned

    # If any topic has < 2 tokens after cleaning, skip it (CoherenceModel
    # cannot score topics with fewer than 2 terms).
    valid_cids = [cid for cid, toks in coh_topics.items() if len(toks) >= 2]
    valid_topics = [coh_topics[cid] for cid in valid_cids]
    print(
        f"    {len(valid_cids)} of {len(coh_topics)} topics have ≥ 2 dictionary "
        f"tokens after cleaning"
    )

    out: dict[int, dict[str, float]] = {cid: {} for cid in raw_topics}

    print("  computing NPMI (c_npmi)…")
    cm_npmi = CoherenceModel(
        topics=valid_topics, texts=docs_tok, dictionary=dictionary,
        coherence="c_npmi", topn=10,
    )
    npmi_per = cm_npmi.get_coherence_per_topic()
    for cid, v in zip(valid_cids, npmi_per):
        out[cid]["npmi"] = float(v)

    print("  computing UMass (u_mass)…")
    cm_umass = CoherenceModel(
        topics=valid_topics, texts=docs_tok, dictionary=dictionary,
        coherence="u_mass", topn=10,
    )
    umass_per = cm_umass.get_coherence_per_topic()
    for cid, v in zip(valid_cids, umass_per):
        out[cid]["umass"] = float(v)

    print("  computing C_v…")
    cm_cv = CoherenceModel(
        topics=valid_topics, texts=docs_tok, dictionary=dictionary,
        coherence="c_v", topn=10,
    )
    cv_per = cm_cv.get_coherence_per_topic()
    for cid, v in zip(valid_cids, cv_per):
        out[cid]["c_v"] = float(v)

    # Topic diversity: unique tokens / total tokens across all clusters'
    # ORIGINAL top-10 lists (Dieng et al. 2020). We use the raw list as it
    # appears in topic_info, normalized to lowercase strings. Phrases stay
    # as-is — Dieng et al.'s metric doesn't tokenize.
    all_terms: list[str] = []
    for words in raw_topics.values():
        all_terms.extend(w.lower() for w in words[:10])
    diversity = len(set(all_terms)) / max(1, len(all_terms))

    aggregate = {
        "n_topics_scored": len(valid_cids),
        "npmi_mean": float(np.mean([out[c]["npmi"] for c in valid_cids])) if valid_cids else float("nan"),
        "umass_mean": float(np.mean([out[c]["umass"] for c in valid_cids])) if valid_cids else float("nan"),
        "c_v_mean": float(np.mean([out[c]["c_v"] for c in valid_cids])) if valid_cids else float("nan"),
        "npmi_median": float(np.median([out[c]["npmi"] for c in valid_cids])) if valid_cids else float("nan"),
        "c_v_median": float(np.median([out[c]["c_v"] for c in valid_cids])) if valid_cids else float("nan"),
        "topic_diversity": float(diversity),
    }

    return out, aggregate


# ---------------------------------------------------------------------------
# §7.11 cross-model naming
# ---------------------------------------------------------------------------

_OPENAI_ENV_KEYS = {"OPENAI_API_KEY", "OPENAI_ORGANIZATION", "OPENAI_PROJECT"}


def _load_openai_env() -> None:
    if not config.OPENAI_ENV_FILE.exists():
        raise RuntimeError(f"OpenAI env file missing at {config.OPENAI_ENV_FILE}")
    for raw in config.OPENAI_ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, _, raw_value = line.partition("=")
        key = key.strip()
        if key not in _OPENAI_ENV_KEYS or not raw_value.strip():
            continue
        parts = shlex.split(raw_value.strip(), posix=True)
        if len(parts) == 1:
            os.environ.setdefault(key, parts[0])


def embed_labels(labels: list[str]) -> np.ndarray:
    """Embed labels via text-embedding-3-large in a single batch."""
    _load_openai_env()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    org = os.environ.get("OPENAI_ORGANIZATION", "").strip()
    project = os.environ.get("OPENAI_PROJECT", "").strip()
    if org:
        headers["OpenAI-Organization"] = org
    if project:
        headers["OpenAI-Project"] = project
    headers["X-Client-Request-Id"] = "job-research:t-quality:label-embed"
    payload = {
        "model": "text-embedding-3-large",
        "input": labels,
        "encoding_format": "float",
    }
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            r = httpx.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers, json=payload, timeout=180.0,
            )
        except httpx.HTTPError as exc:
            last_error = exc
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            data = r.json()["data"]
            data.sort(key=lambda d: d["index"])
            arr = np.array([d["embedding"] for d in data], dtype=np.float32)
            return arr
        if r.status_code in {429, 500, 502, 503, 504}:
            last_error = RuntimeError(f"transient {r.status_code}: {r.text[:300]}")
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"embed call failed {r.status_code}: {r.text[:500]}")
    raise RuntimeError(f"embed retries exhausted: {last_error}")


def section_7_11_cross_model_naming(
    *, df: pd.DataFrame, topic_info: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Re-run §5.1 LLM naming with gpt-5.4-mini against the gpt-5.5 primary
    labels, using the same exemplar-selection protocol."""
    docs = df["description_core_llm"].tolist()

    print("  reducing model to K=10 to recover representative docs…")
    model = pipeline.load_topic_model_for_reduce(config.RAW_FIT_PATH)
    model.reduce_topics(docs, nr_topics=HEADLINE_K)

    head_labels = np.asarray(model.topics_, dtype=np.int64)

    rows = []
    for _, prim in topic_info.sort_values("topic_id").iterrows():
        cid = int(prim["topic_id"])
        # Same exemplar protocol as run_stage1.s1_6_naming.
        words = [w for w, _ in model.get_topic(cid)[:15]]
        member_idx = np.where(head_labels == cid)[0]
        rep_docs = (model.get_representative_docs(cid) or [])[:5]
        rng = np.random.default_rng(seed=cid)
        if len(member_idx) > 0:
            random_idx = rng.choice(
                member_idx, size=min(2, len(member_idx)), replace=False,
            )
            random_docs = [docs[i] for i in random_idx]
        else:
            random_docs = []
        exemplars = [
            (f"posting {i}", d[:200])
            for i, d in enumerate(list(rep_docs) + random_docs)
        ]
        try:
            label = propose_label(
                top_words=words,
                exemplars=exemplars,
                model=config.LLM_MODEL_SECONDARY,
                request_id=f"t-quality-cross-c{cid}",
            )
        except Exception as exc:  # noqa: BLE001
            label = {
                "label": "(LLM-naming-failed)",
                "confidence": 0.0,
                "alternative": str(exc)[:200],
            }
        rows.append({
            "topic_id": cid,
            "gpt55_label": str(prim["gpt55_label"]),
            "gpt54mini_label": str(label.get("label")),
            "gpt54mini_confidence": float(label.get("confidence") or 0.0),
            "gpt54mini_alternative": str(label.get("alternative") or ""),
        })
        print(
            f"    cid={cid}: gpt-5.5={prim['gpt55_label']!r} "
            f"vs gpt-5.4-mini={label.get('label')!r}"
        )
    out = pd.DataFrame(rows)

    # Exact-match (case-insensitive, whitespace-collapsed).
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s).strip().lower())
    out["exact_match"] = (
        out["gpt55_label"].map(_norm) == out["gpt54mini_label"].map(_norm)
    )

    # Label-embedding cosine via text-embedding-3-large.
    primary = out["gpt55_label"].tolist()
    secondary = out["gpt54mini_label"].tolist()
    print("  embedding labels with text-embedding-3-large (single batch)…")
    embs = embed_labels(primary + secondary)
    a = embs[:len(primary)]
    b = embs[len(primary):]
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    cos = (a_norm * b_norm).sum(axis=1)
    out["label_cosine"] = cos.astype(float)

    summary = {
        "n_clusters": int(len(out)),
        "exact_match_rate": float(out["exact_match"].mean()),
        "cosine_mean": float(out["label_cosine"].mean()),
        "cosine_median": float(np.median(out["label_cosine"])),
        "cosine_min": float(out["label_cosine"].min()),
        "share_cosine_ge_0_85": float((out["label_cosine"] >= 0.85).mean()),
    }
    return out, summary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    print("Step 0 — verifying frozen Stage 1 hashes…")
    verify_hashes()

    print("\nStep 1 — loading Sample A + assignments + embeddings…")
    df = load_sample_a_with_assignments()
    print(f"  {len(df)} rows; {df['is_outlier'].sum()} outliers")
    embeddings = load_embeddings_for_uids(df["uid"].tolist())
    print(f"  embeddings shape={embeddings.shape}")

    topic_info = pd.read_parquet(REPO_ROOT / "data/bertopic/topic_info.parquet")
    print(f"  topic_info rows={len(topic_info)}")

    # ---- §7.10 honest noise rate ----
    print("\nStep 2 — §7.10 honest noise rate (before/after reduce_outliers)…")
    noise = section_7_10_noise(df=df, embeddings=embeddings)
    print(f"  noise_rate_before = {noise['noise_rate_before']:.6f}")
    print(f"  noise_rate_after  = {noise['noise_rate_after']:.6f}")

    # ---- §7.9 silhouette + sizes ----
    print("\nStep 3 — §7.9 silhouette + cluster-size distribution…")
    sil_per_cluster, sil_overall, sizes = section_7_9_silhouette_and_sizes(df=df)
    print(f"  overall mean silhouette = {sil_overall['overall_mean']:.4f}")
    for cid, v in sorted(sil_per_cluster.items()):
        print(f"    c{cid}: {v:.4f}")
    print(f"  size dist: {sizes}")

    # ---- §7.8 coherence + diversity ----
    print("\nStep 4 — §7.8 coherence (NPMI / UMass / C_v) + diversity…")
    per_topic_coh, agg_coh = section_7_8_coherence(df=df, topic_info=topic_info)
    print(f"  aggregate: {agg_coh}")

    # ---- §7.11 cross-model naming ----
    print("\nStep 5 — §7.11 cross-model naming with gpt-5.4-mini…")
    cross_naming, naming_summary = section_7_11_cross_model_naming(
        df=df, topic_info=topic_info,
    )
    print(f"  summary: {naming_summary}")

    # ---- write outputs ----
    print("\nStep 6 — writing outputs…")
    rows = []
    n_members_by_topic = (
        df.loc[~df["is_outlier"]].groupby("topic_id").size().to_dict()
    )
    for cid in sorted(n_members_by_topic):
        rows.append({
            "topic_id": int(cid),
            "n_members": int(n_members_by_topic[cid]),
            "npmi": per_topic_coh.get(cid, {}).get("npmi", float("nan")),
            "umass": per_topic_coh.get(cid, {}).get("umass", float("nan")),
            "c_v": per_topic_coh.get(cid, {}).get("c_v", float("nan")),
            "silhouette": sil_per_cluster.get(int(cid), float("nan")),
        })
    quality = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    quality.to_parquet(TOPIC_QUALITY_PATH, index=False)
    print(f"  wrote {TOPIC_QUALITY_PATH}  ({len(quality)} rows)")

    info_aug = topic_info.merge(
        cross_naming[
            ["topic_id", "gpt54mini_label", "gpt54mini_confidence",
             "gpt54mini_alternative", "exact_match", "label_cosine"]
        ],
        on="topic_id", how="left",
    )
    info_aug.to_parquet(TOPIC_INFO_NAMING_PATH, index=False)
    print(f"  wrote {TOPIC_INFO_NAMING_PATH}  ({len(info_aug)} rows)")

    # ---- print compact summary for the memo ----
    print("\n" + "=" * 60)
    print("SUMMARY (for memo)")
    print("=" * 60)
    print(json.dumps({
        "noise": noise,
        "silhouette_overall": sil_overall,
        "silhouette_per_cluster": sil_per_cluster,
        "sizes": sizes,
        "coherence_aggregate": agg_coh,
        "naming_summary": naming_summary,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
