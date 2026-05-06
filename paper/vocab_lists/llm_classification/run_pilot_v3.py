"""V3 pilot driver — 3 variants × 3 models × 3 reps × 31 postings = 837 calls.

Variants: skill / role_family / combined  (defined in classifier_v3.VARIANT_PROMPTS)
Models: gpt-5.4-nano / gpt-5.4-mini / gpt-5.4
Reps: 3

Output: pilot_v3_results.jsonl (one row per call)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DIR = REPO / "paper/vocab_lists/llm_classification"
SAMPLE = DIR / "pilot_sample_v3.parquet"
RESULTS = DIR / "pilot_v3_results.jsonl"

DEFAULT_VARIANTS = ("skill", "role_family", "combined")
DEFAULT_MODELS = ("gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4")
DEFAULT_REPS = 3
DEFAULT_WORKERS = 6

sys.path.insert(0, str(Path(__file__).parent))
from classifier_v3 import classify_v3  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--reps", type=int, default=DEFAULT_REPS)
    p.add_argument("--max-workers", type=int, default=DEFAULT_WORKERS)
    args = p.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    sample = pd.read_parquet(SAMPLE)
    n_post = len(sample)
    n_calls = n_post * len(variants) * len(models) * args.reps
    print(f"V3 pilot: {n_post} postings × {len(variants)} variants × {len(models)} models × "
          f"{args.reps} reps = {n_calls} calls", flush=True)

    plan = []
    for _, row in sample.iterrows():
        for variant in variants:
            for model in models:
                for rep in range(args.reps):
                    plan.append((row, variant, model, rep))

    if RESULTS.exists():
        RESULTS.unlink()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    def worker(item):
        row, variant, model, rep = item
        result = classify_v3(
            variant=variant,
            title=row["title"] or "",
            description=row["description_core_llm"] or "",
            model=model,
        )
        return {
            "uid": row["uid"],
            "stratum_v3": row["stratum_v3"],
            "title": row["title"],
            "role_family_heuristic": row["role_family_heuristic"],
            "regex_skill_topics": list(row["regex_skill_topics"]) if row["regex_skill_topics"] is not None else [],
            "variant": variant,
            "model": model,
            "rep": rep,
            "skill_themes": result.skill_themes,
            "role_families": result.role_families,
            "parsed_ok": result.parsed_ok,
            "latency_s": result.latency_s,
            "request_id": result.request_id,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cached_tokens": result.cached_tokens,
            "error": result.error,
        }

    completed = 0
    t0 = time.time()
    fh = RESULTS.open("a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = [pool.submit(worker, item) for item in plan]
            for fut in as_completed(futures):
                rec = fut.result()
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                completed += 1
                if completed % 50 == 0 or completed == len(plan):
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed else 0
                    eta = (len(plan) - completed) / rate if rate else 0
                    print(f"  {completed:>4}/{len(plan)}  {rate:.1f} calls/s  ETA {eta:.0f}s",
                          flush=True)
    finally:
        fh.close()

    elapsed = time.time() - t0
    print(f"\nWrote {completed} results to {RESULTS} in {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
