#!/usr/bin/env python3
"""
LLM Spot-Check Validation — lightweight batch version.

Validates pipeline quality using claude -p with batched prompts (10 postings per call).
Runs on the V2 unified.parquet after pipeline rebuild completes.

Usage:
    python preprocessing/scripts/llm_spot_check.py

Produces:
    preprocessing/logs/llm_validation_report.md
    preprocessing/intermediate/llm_validation_responses.json
"""

import gc
import json
import logging
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "llm_spot_check.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def sample_postings(parquet_path: str, n: int, filter_col: str = None,
                    filter_val=None, stratify_col: str = None,
                    columns: list = None, seed: int = 42) -> pd.DataFrame:
    """Memory-safe sampling from parquet."""
    pf = pq.ParquetFile(parquet_path)
    chunks = []
    for batch in pf.iter_batches(batch_size=200_000, columns=columns):
        df = batch.to_pandas()
        if filter_col and filter_val is not None:
            if isinstance(filter_val, bool):
                df = df[df[filter_col] == filter_val]
            else:
                df = df[df[filter_col].isin(filter_val) if isinstance(filter_val, list) else df[filter_col] == filter_val]
        chunks.append(df)
        del batch
    full = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()

    if stratify_col and stratify_col in full.columns:
        groups = full.groupby(stratify_col)
        per_group = max(1, n // len(groups))
        sampled = groups.apply(lambda x: x.sample(min(per_group, len(x)), random_state=seed))
        return sampled.reset_index(drop=True).head(n)
    else:
        return full.sample(min(n, len(full)), random_state=seed)


def call_claude(prompt: str, retries: int = 2) -> str:
    """Call claude -p and return the result text."""
    for attempt in range(retries + 1):
        try:
            result = subprocess.run(
                ['claude', '-p', prompt, '--output-format', 'json'],
                capture_output=True, text=True, timeout=90,
            )
            if result.returncode == 0:
                outer = json.loads(result.stdout.strip())
                return outer.get('result', '')
            else:
                log.warning(f"  claude returned {result.returncode}: {result.stderr[:200]}")
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            log.warning(f"  Attempt {attempt+1} failed: {e}")
        time.sleep(2)
    return ""


def parse_json_from_response(text: str) -> list[dict]:
    """Extract JSON array from LLM response (may be wrapped in markdown fences)."""
    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass
    # Extract from markdown code block
    match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding array directly
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return []


# ---------------------------------------------------------------------------
# Validation tasks
# ---------------------------------------------------------------------------

def validate_swe_classification(parquet_path: str) -> dict:
    """Validate SWE classification on 50 postings."""
    log.info("=== SWE Classification Validation (50 postings) ===")

    cols = ['title', 'company_name', 'description_core', 'is_swe', 'is_swe_adjacent',
            'swe_classification_tier', 'swe_confidence', 'source']

    # Sample 20 SWE, 15 SWE-adjacent, 15 non-SWE with engineering titles
    swe = sample_postings(parquet_path, 20, 'is_swe', True, 'source', cols)
    adj = sample_postings(parquet_path, 15, 'is_swe_adjacent', True, 'source', cols)

    # Non-SWE with "engineer" in title
    pf = pq.ParquetFile(parquet_path)
    non_swe_chunks = []
    for batch in pf.iter_batches(batch_size=200_000, columns=cols):
        df = batch.to_pandas()
        mask = (~df['is_swe']) & (~df['is_swe_adjacent']) & df['title'].str.contains(r'(?i)engineer', na=False)
        non_swe_chunks.append(df[mask])
        del batch
    non_swe = pd.concat(non_swe_chunks).sample(min(15, len(pd.concat(non_swe_chunks))), random_state=42)
    del non_swe_chunks; gc.collect()

    sample = pd.concat([swe, adj, non_swe], ignore_index=True)

    # Batch into groups of 10
    results = []
    for i in range(0, len(sample), 10):
        batch = sample.iloc[i:i+10]
        items = []
        for j, (_, row) in enumerate(batch.iterrows()):
            desc = str(row.get('description_core', ''))[:300]
            items.append(f'{j+1}. Title: "{row["title"]}" | Company: {row.get("company_name", "?")} | Desc: {desc[:200]}')

        prompt = f"""Classify each job as SWE, SWE_ADJACENT, or NOT_SWE.
SWE = builds/maintains software. SWE_ADJACENT = works with software but doesn't primarily code. NOT_SWE = unrelated.

{chr(10).join(items)}

Respond as JSON array: [{{"num":1,"classification":"SWE"|"SWE_ADJACENT"|"NOT_SWE","reason":"..."}}]"""

        response = call_claude(prompt)
        batch_results = parse_json_from_response(response)
        results.extend(batch_results)
        log.info(f"  Batch {i//10+1}: {len(batch_results)} classified")
        time.sleep(1)

    # Compute agreement
    agreements = 0
    total = min(len(results), len(sample))
    for idx in range(total):
        row = sample.iloc[idx]
        llm = results[idx].get('classification', '') if idx < len(results) else ''
        pipeline = 'SWE' if row['is_swe'] else ('SWE_ADJACENT' if row.get('is_swe_adjacent') else 'NOT_SWE')
        if llm == pipeline:
            agreements += 1

    rate = agreements / max(total, 1)
    log.info(f"  Agreement rate: {agreements}/{total} = {rate:.1%}")
    return {"task": "swe_classification", "n": total, "agreements": agreements, "rate": rate, "details": results}


def validate_seniority(parquet_path: str) -> dict:
    """Validate seniority classification on 50 SWE postings."""
    log.info("=== Seniority Validation (50 SWE postings) ===")

    cols = ['title', 'description_core', 'seniority_final', 'seniority_imputed',
            'seniority_native', 'seniority_source', 'is_swe', 'source']

    sample = sample_postings(parquet_path, 50, 'is_swe', True, 'seniority_imputed', cols)

    results = []
    for i in range(0, len(sample), 10):
        batch = sample.iloc[i:i+10]
        items = []
        for j, (_, row) in enumerate(batch.iterrows()):
            desc = str(row.get('description_core', ''))[:400]
            items.append(f'{j+1}. Title: "{row["title"]}" | Desc: {desc[:300]}')

        prompt = f"""What seniority level is each role? Answer: entry level / associate / mid-senior level / director / unknown

{chr(10).join(items)}

Respond as JSON array: [{{"num":1,"seniority":"...","signals":"one sentence"}}]"""

        response = call_claude(prompt)
        batch_results = parse_json_from_response(response)
        results.extend(batch_results)
        log.info(f"  Batch {i//10+1}: {len(batch_results)} classified")
        time.sleep(1)

    agreements = 0
    total = min(len(results), len(sample))
    for idx in range(total):
        row = sample.iloc[idx]
        llm = results[idx].get('seniority', '') if idx < len(results) else ''
        pipeline_sen = row.get('seniority_final', row.get('seniority_imputed', ''))
        if llm == pipeline_sen:
            agreements += 1

    rate = agreements / max(total, 1)
    log.info(f"  Agreement rate: {agreements}/{total} = {rate:.1%}")
    return {"task": "seniority", "n": total, "agreements": agreements, "rate": rate, "details": results}


def validate_ghost_jobs(parquet_path: str) -> dict:
    """Validate ghost job detection on 30 entry-level SWE postings."""
    log.info("=== Ghost Job Validation (30 entry-level SWE) ===")

    cols = ['title', 'description_core', 'seniority_final', 'ghost_job_risk',
            'is_swe', 'source']

    pf = pq.ParquetFile(parquet_path)
    chunks = []
    for batch in pf.iter_batches(batch_size=200_000, columns=cols):
        df = batch.to_pandas()
        mask = df['is_swe'] & (df['seniority_final'] == 'entry level')
        chunks.append(df[mask])
        del batch
    entry = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()

    high = entry[entry['ghost_job_risk'] == 'high'].sample(min(15, len(entry[entry['ghost_job_risk'] == 'high'])), random_state=42)
    low = entry[entry['ghost_job_risk'] == 'low'].sample(min(15, len(entry[entry['ghost_job_risk'] == 'low'])), random_state=42)
    sample = pd.concat([high, low], ignore_index=True)

    results = []
    for i in range(0, len(sample), 10):
        batch = sample.iloc[i:i+10]
        items = []
        for j, (_, row) in enumerate(batch.iterrows()):
            desc = str(row.get('description_core', ''))[:400]
            items.append(f'{j+1}. Title: "{row["title"]}" (labeled entry-level) | Desc: {desc[:300]}')

        prompt = f"""For each entry-level role, are the requirements realistic? Answer: realistic / inflated / ghost_likely

{chr(10).join(items)}

Respond as JSON array: [{{"num":1,"assessment":"realistic"|"inflated"|"ghost_likely","reason":"..."}}]"""

        response = call_claude(prompt)
        batch_results = parse_json_from_response(response)
        results.extend(batch_results)
        log.info(f"  Batch {i//10+1}: {len(batch_results)} assessed")
        time.sleep(1)

    return {"task": "ghost_jobs", "n": len(sample), "details": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    log.info("=" * 60)
    log.info("LLM SPOT-CHECK VALIDATION")
    log.info("=" * 60)

    parquet_path = str(DATA_DIR / "unified.parquet")

    all_results = {}

    # 1. SWE classification
    all_results["swe"] = validate_swe_classification(parquet_path)

    # 2. Seniority
    all_results["seniority"] = validate_seniority(parquet_path)

    # 3. Ghost jobs
    all_results["ghost_jobs"] = validate_ghost_jobs(parquet_path)

    # Save raw results
    output = INTERMEDIATE_DIR / "llm_validation_responses.json"
    with open(output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nSaved responses to {output}")

    # Generate report
    report = generate_report(all_results)
    report_path = LOG_DIR / "llm_validation_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    log.info(f"Saved report to {report_path}")


def generate_report(results: dict) -> str:
    lines = [
        "# LLM Spot-Check Validation Report",
        f"",
        f"Date: 2026-03-19",
        f"Method: claude -p batch review (10 postings per call)",
        f"",
        "## Summary",
        "",
    ]

    # SWE classification
    swe = results.get("swe", {})
    lines.append(f"### SWE Classification")
    lines.append(f"- Sample size: {swe.get('n', 0)}")
    lines.append(f"- Agreement rate: {swe.get('rate', 0):.1%}")
    lines.append(f"- {swe.get('agreements', 0)} of {swe.get('n', 0)} pipeline labels matched LLM assessment")
    lines.append("")

    # Seniority
    sen = results.get("seniority", {})
    lines.append(f"### Seniority Classification")
    lines.append(f"- Sample size: {sen.get('n', 0)}")
    lines.append(f"- Agreement rate: {sen.get('rate', 0):.1%}")
    lines.append(f"- {sen.get('agreements', 0)} of {sen.get('n', 0)} pipeline labels matched LLM assessment")
    lines.append("")

    # Ghost jobs
    ghost = results.get("ghost_jobs", {})
    details = ghost.get("details", [])
    if details:
        realistic = sum(1 for d in details if d.get('assessment') == 'realistic')
        inflated = sum(1 for d in details if d.get('assessment') == 'inflated')
        ghost_likely = sum(1 for d in details if d.get('assessment') == 'ghost_likely')
        lines.append(f"### Ghost Job Assessment")
        lines.append(f"- Sample size: {ghost.get('n', 0)} entry-level SWE postings")
        lines.append(f"- Realistic requirements: {realistic}")
        lines.append(f"- Inflated requirements: {inflated}")
        lines.append(f"- Likely ghost jobs: {ghost_likely}")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    run()
