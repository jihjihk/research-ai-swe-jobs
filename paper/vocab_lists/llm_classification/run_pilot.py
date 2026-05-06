"""Pilot driver — stratified 25-posting sample × 3 models × 3 reps = 225 calls.

Outputs:
  - pilot_sample.parquet : the 25 postings (frozen, with regex topic labels)
  - pilot_results.jsonl  : one row per call

Usage:
    ./.venv/bin/python paper/vocab_lists/llm_classification/run_pilot.py
        [--models gpt-5.4-nano,gpt-5.4-mini,gpt-5.4]
        [--reps 3]
        [--max-workers 6]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
PARQUET = REPO / "data/unified_core.parquet"
VOCAB = REPO / "paper/vocab_lists/vocab_lists.json"
OUT_DIR = REPO / "paper/vocab_lists/llm_classification"
SAMPLE_PATH = OUT_DIR / "pilot_sample.parquet"
RESULTS_PATH = OUT_DIR / "pilot_results.jsonl"

STRATA = [
    ("kaggle_arshkon", "2024-04"),
    ("kaggle_asaniczka", "2024-01"),
    ("scraped", "2026-04"),
]

DEFAULT_MODELS = ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4"]
DEFAULT_REPS = 3
DEFAULT_WORKERS = 6
SEED = 20260506
WORD_LIKE = re.compile(r"^[A-Za-z0-9 \-']+$")

# Persistent-fluff concepts the regex review flagged across rounds — used to
# pick stratum-D postings.
HIGH_NOISE_TARGETS = [
    ("performance", "depth_claim_language"),
    ("mentorship", "Guidance and influence"),
]

# inline classifier (imports from local module)
sys.path.insert(0, str(Path(__file__).parent))
import classifier  # noqa: E402

# ============================================================
# Regex labeling — mirrors run_calibration.py's matching logic
# ============================================================

def compile_pattern(kw: str) -> re.Pattern:
    kw_lc = kw.lower()
    if WORD_LIKE.match(kw_lc):
        return re.compile(rf"\b{re.escape(kw_lc)}\b", re.IGNORECASE)
    starts_word = kw_lc[0].isalnum()
    ends_word = kw_lc[-1].isalnum()
    pre = r"(?<![A-Za-z0-9])" if starts_word else r""
    post = r"(?![A-Za-z0-9])" if ends_word else r""
    return re.compile(pre + re.escape(kw_lc) + post, re.IGNORECASE)


def build_topic_concept_patterns(vocab: dict):
    """For each (topic, concept), build (keyword_lower, compiled_pattern) pairs.
    Keeping the lowercase keyword alongside the pattern lets us do a fast
    substring prefilter (`kw in text_lc`) before running the expensive regex.
    """
    out: dict[str, dict[str, list[tuple[str, re.Pattern]]]] = {}
    for slug, t in vocab["topics"].items():
        out[slug] = {}
        for c in t["core_concepts"]:
            out[slug][c["name"]] = [(k.lower(), compile_pattern(k)) for k in c["keywords"]]
    return out


def label_posting(text: str, patterns: dict) -> tuple[set[str], set[tuple[str, str]]]:
    """Return (set of topic slugs hit, set of (topic, concept) pairs hit).
    Fast path: substring `in` check before regex (mirrors run_calibration.py).
    """
    text_lc = text.lower() if text else ""
    if not text_lc:
        return set(), set()
    topics_hit: set[str] = set()
    concepts_hit: set[tuple[str, str]] = set()
    for slug, concepts in patterns.items():
        for cname, kw_pats in concepts.items():
            hit = False
            for kw_lc, pat in kw_pats:
                if kw_lc not in text_lc:
                    continue
                if pat.search(text_lc):
                    hit = True
                    break
            if hit:
                topics_hit.add(slug)
                concepts_hit.add((slug, cname))
    return topics_hit, concepts_hit


# ============================================================
# Sample selection
# ============================================================

def load_swe_postings() -> pd.DataFrame:
    print(f"Loading SWE postings from {PARQUET.name} for {len(STRATA)} strata...", flush=True)
    con = duckdb.connect()
    frames = []
    for src, period in STRATA:
        q = f"""
        SELECT uid, source, period, title, description_core_llm, description, company_name_effective
        FROM read_parquet('{PARQUET}')
        WHERE source = ? AND period = ? AND is_swe = TRUE
              AND description_core_llm IS NOT NULL
              AND length(description_core_llm) >= 80
        """
        frames.append(con.execute(q, [src, period]).fetchdf())
    df = pd.concat(frames, ignore_index=True)
    print(f"  total in-frame SWE postings: {len(df):,}", flush=True)
    return df


def label_all(df: pd.DataFrame, patterns: dict) -> pd.DataFrame:
    n = len(df)
    print(f"Computing regex topic+concept labels for {n:,} postings...", flush=True)
    topic_labels = []
    concept_pairs = []
    t0 = time.time()
    for i, text in enumerate(df["description_core_llm"].fillna("").tolist()):
        topics, concepts = label_posting(text, patterns)
        topic_labels.append(sorted(topics))
        concept_pairs.append([f"{t}/{c}" for t, c in sorted(concepts)])
        if (i + 1) % 5000 == 0 or i + 1 == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / rate if rate else 0
            print(f"  {i+1:>6,}/{n:,}  ({rate:.0f}/s, ETA {eta:.0f}s)", flush=True)
    df = df.copy()
    df["regex_topics"] = topic_labels
    df["regex_concepts"] = concept_pairs
    df["n_regex_topics"] = df["regex_topics"].map(len)
    return df


def stratified_pick(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    """Pick 25 postings:
      - 8 single-topic (one per topic)
      - 5 multi-topic (3+ topics)
      - 5 zero-topic
      - 5 high-noise
      - 2 random
    """
    picked: list[int] = []  # row indices
    used: set = set()

    def reserve(idx: int):
        used.add(idx)
        picked.append(idx)

    # 8 single-topic — one posting per topic where ONLY that topic fires
    print("Stratum A: 8 single-topic ...", flush=True)
    for slug in classifier.LABELS:
        candidates = df.index[
            (df["n_regex_topics"] == 1)
            & df["regex_topics"].map(lambda labs, s=slug: labs == [s])
            & ~df.index.to_series().isin(used)
        ].tolist()
        if not candidates:
            print(f"  WARNING: no single-topic posting found for {slug}", flush=True)
            continue
        idx = rng.choice(candidates)
        reserve(idx)
        print(f"  {slug}: uid={df.loc[idx,'uid']} title={df.loc[idx,'title'][:60]!r}", flush=True)

    # 5 multi-topic — 3+ topics
    print("Stratum B: 5 multi-topic (3+) ...", flush=True)
    candidates = df.index[(df["n_regex_topics"] >= 3) & ~df.index.to_series().isin(used)].tolist()
    rng.shuffle(candidates)
    for idx in candidates[:5]:
        reserve(idx)
        print(f"  uid={df.loc[idx,'uid']} topics={df.loc[idx,'regex_topics']}", flush=True)

    # 5 zero-topic
    print("Stratum C: 5 zero-topic ...", flush=True)
    candidates = df.index[(df["n_regex_topics"] == 0) & ~df.index.to_series().isin(used)].tolist()
    rng.shuffle(candidates)
    for idx in candidates[:5]:
        reserve(idx)
        print(f"  uid={df.loc[idx,'uid']} title={df.loc[idx,'title'][:60]!r}", flush=True)

    # 5 high-noise — postings hitting persistent-fluff concepts
    print("Stratum D: 5 high-noise ...", flush=True)
    target_keys = [f"{t}/{c}" for t, c in HIGH_NOISE_TARGETS]
    has_noise = df["regex_concepts"].map(lambda cs: any(k in cs for k in target_keys))
    candidates = df.index[has_noise & ~df.index.to_series().isin(used)].tolist()
    rng.shuffle(candidates)
    for idx in candidates[:5]:
        reserve(idx)
        print(f"  uid={df.loc[idx,'uid']} hits={[k for k in df.loc[idx,'regex_concepts'] if k in target_keys]}",
              flush=True)

    # 2 random
    print("Stratum E: 2 random ...", flush=True)
    candidates = df.index[~df.index.to_series().isin(used)].tolist()
    rng.shuffle(candidates)
    for idx in candidates[:2]:
        reserve(idx)
        print(f"  uid={df.loc[idx,'uid']} title={df.loc[idx,'title'][:60]!r}", flush=True)

    sample = df.loc[picked].reset_index(drop=True)
    sample["stratum"] = (
        ["A_single"] * 8
        + ["B_multi"] * 5
        + ["C_zero"] * 5
        + ["D_noise"] * 5
        + ["E_random"] * 2
    )[: len(sample)]
    return sample


# ============================================================
# Pilot execution
# ============================================================

def run_pilot_calls(sample: pd.DataFrame, models: list[str], reps: int, workers: int) -> int:
    """Execute (posting × model × rep) calls in a thread pool, append to JSONL."""
    plan = []
    for _, row in sample.iterrows():
        for m in models:
            for r in range(reps):
                plan.append((row, m, r))
    print(f"\n{len(plan)} calls planned ({len(sample)} postings × {len(models)} models × {reps} reps).", flush=True)

    # truncate any prior results — fresh pilot
    if RESULTS_PATH.exists():
        RESULTS_PATH.unlink()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    def worker(row_model_rep):
        row, model, rep = row_model_rep
        result = classifier.classify(
            title=row["title"] or "",
            description=row["description_core_llm"] or "",
            model=model,
            seed=None,  # Responses API does not honour seed
        )
        rec = {
            "uid": row["uid"],
            "stratum": row["stratum"],
            "model": model,
            "rep": rep,
            "regex_topics": row["regex_topics"],
            "labels": result.labels,
            "parsed_ok": result.parsed_ok,
            "latency_s": result.latency_s,
            "request_id": result.request_id,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cached_tokens": result.cached_tokens,
            "error": result.error,
        }
        return rec

    completed = 0
    t0 = time.time()
    fh = RESULTS_PATH.open("a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, item) for item in plan]
            for fut in as_completed(futures):
                rec = fut.result()
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                completed += 1
                if completed % 25 == 0 or completed == len(plan):
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"  {completed:>3}/{len(plan)}  {rate:.1f} calls/s", flush=True)
    finally:
        fh.close()
    return completed


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=",".join(DEFAULT_MODELS),
                   help="Comma-separated model IDs.")
    p.add_argument("--reps", type=int, default=DEFAULT_REPS)
    p.add_argument("--max-workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--reuse-sample", action="store_true",
                   help="Skip sample selection if pilot_sample.parquet exists.")
    args = p.parse_args()

    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Sample ----
    if args.reuse_sample and SAMPLE_PATH.exists():
        sample = pd.read_parquet(SAMPLE_PATH)
        # Re-cast list columns from numpy arrays back to lists if needed
        for col in ("regex_topics", "regex_concepts"):
            if col in sample.columns:
                sample[col] = sample[col].map(lambda v: list(v) if hasattr(v, "tolist") else list(v))
        print(f"Reusing existing sample: {SAMPLE_PATH} ({len(sample)} postings)", flush=True)
    else:
        vocab = json.loads(VOCAB.read_text())
        patterns = build_topic_concept_patterns(vocab)
        df = load_swe_postings()
        df = label_all(df, patterns)
        print(f"\nRegex topic-count distribution:")
        for n, c in df["n_regex_topics"].value_counts().sort_index().items():
            print(f"  {n} topics: {c:,}")
        print()
        sample = stratified_pick(df, rng)
        # write parquet
        pq.write_table(pa.Table.from_pandas(sample), SAMPLE_PATH)
        print(f"\nWrote {SAMPLE_PATH} ({len(sample)} postings)", flush=True)

    # ---- 2. Calls ----
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    n = run_pilot_calls(sample, models, args.reps, args.max_workers)
    print(f"\nWrote {n} call results to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
