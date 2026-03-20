#!/usr/bin/env python3
"""
Stage Final: Produce unified.parquet, quality_report.json, preprocessing_log.txt

Reads stage8_final.parquet in chunks (200K rows) to stay within 31GB RAM.
"""

import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/home/jihgaboot/gabor/job-research")
INTERMEDIATE = BASE / "preprocessing" / "intermediate"
DATA = BASE / "data"
INPUT_FILE = INTERMEDIATE / "stage8_final.parquet"
OUTPUT_PARQUET = DATA / "unified.parquet"
OUTPUT_REPORT = DATA / "quality_report.json"
OUTPUT_LOG = DATA / "preprocessing_log.txt"

CHUNK_SIZE = 200_000

# ---------------------------------------------------------------------------
# Known funnel values from earlier stage logs
# ---------------------------------------------------------------------------
KAGGLE_ARSHKON_RAW = 123_849
KAGGLE_ASANICZKA_RAW = 1_348_454
SCRAPED_RAW = 100_739  # after URL dedup within scraped
TOTAL_RAW = KAGGLE_ARSHKON_RAW + KAGGLE_ASANICZKA_RAW + SCRAPED_RAW  # 1,573,042
AGGREGATOR_FLAGGED = 43_121
DEDUP_REMOVED = 369_225
AFTER_DEDUP = 1_203_817


def main():
    t0 = time.time()
    print("=" * 60)
    print("FINAL OUTPUT: unified.parquet + quality_report + log")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Copy stage8_final.parquet -> data/unified.parquet
    # ------------------------------------------------------------------
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"\n[1/3] Copying {INPUT_FILE} -> {OUTPUT_PARQUET} ...")
    shutil.copy2(str(INPUT_FILE), str(OUTPUT_PARQUET))
    print(f"  Done. Size: {OUTPUT_PARQUET.stat().st_size / (1024**3):.2f} GB")

    # ------------------------------------------------------------------
    # 2. Build quality report by streaming through parquet in chunks
    # ------------------------------------------------------------------
    print(f"\n[2/3] Building quality report (chunk size={CHUNK_SIZE:,}) ...")

    pf = pq.ParquetFile(str(INPUT_FILE))
    total_rows = pf.metadata.num_rows
    num_cols = pf.metadata.num_columns
    print(f"  Total rows: {total_rows:,}, columns: {num_cols}")

    # Accumulators
    total_swe = 0
    total_control = 0
    swe_by_source = Counter()
    seniority_dist_swe = Counter()  # seniority_imputed for SWE rows
    swe_regex = 0
    swe_embedding = 0
    swe_unresolved = 0
    seniority_imputed_title = 0
    seniority_imputed_description = 0
    seniority_unknown = 0
    ghost_high = 0
    ghost_medium = 0
    desc_empty = 0
    non_english = 0
    date_out_of_range = 0
    aggregator_total = 0
    swe_from_aggregators = 0

    # For missing data — track counts for key fields
    key_fields = [
        "title", "description", "company_name", "location",
        "seniority_native", "min_salary", "company_industry",
        "company_size", "date_posted", "skills_raw",
    ]
    missing_counts = {f: 0 for f in key_fields}
    empty_counts = {f: 0 for f in key_fields}

    # Boilerplate stats accumulators (SWE only)
    swe_desc_lengths = []
    swe_core_lengths = []

    rows_processed = 0

    # Columns we actually need — read only these to save memory
    needed_cols = [
        "is_swe", "is_control", "source", "seniority_imputed",
        "swe_classification_tier", "seniority_source",
        "ghost_job_risk", "description_quality_flag", "lang_detected",
        "date_flag", "is_aggregator",
        "description_length", "core_length",
    ] + key_fields

    # Deduplicate (some overlap)
    needed_cols = list(dict.fromkeys(needed_cols))

    for batch in pf.iter_batches(batch_size=CHUNK_SIZE, columns=needed_cols):
        tbl = batch.to_pydict()
        n = len(tbl["is_swe"])
        rows_processed += n

        is_swe = tbl["is_swe"]
        is_control = tbl["is_control"]
        source = tbl["source"]
        seniority_imputed = tbl["seniority_imputed"]
        swe_tier = tbl["swe_classification_tier"]
        seniority_src = tbl["seniority_source"]
        ghost_risk = tbl["ghost_job_risk"]
        desc_quality = tbl["description_quality_flag"]
        lang = tbl["lang_detected"]
        date_f = tbl["date_flag"]
        is_agg = tbl["is_aggregator"]
        desc_len = tbl["description_length"]
        core_len = tbl["core_length"]

        for i in range(n):
            # SWE / control counts
            if is_swe[i]:
                total_swe += 1
                swe_by_source[source[i]] += 1
                sen = seniority_imputed[i] if seniority_imputed[i] else "unknown"
                seniority_dist_swe[sen] += 1

                # Boilerplate stats (SWE only) — collect lengths
                if desc_len[i] is not None:
                    swe_desc_lengths.append(desc_len[i])
                if core_len[i] is not None:
                    swe_core_lengths.append(core_len[i])

            if is_control[i]:
                total_control += 1

            # Classification tier counts
            tier = swe_tier[i]
            if tier == "regex":
                swe_regex += 1
            elif tier == "embedding":
                swe_embedding += 1
            elif tier is None or tier == "":
                swe_unresolved += 1
            else:
                swe_unresolved += 1

            # Seniority source counts
            ss = seniority_src[i]
            if ss == "imputed_title":
                seniority_imputed_title += 1
            elif ss == "imputed_description":
                seniority_imputed_description += 1
            else:
                seniority_unknown += 1

            # Quality flags
            gr = ghost_risk[i]
            if gr == "high":
                ghost_high += 1
            elif gr == "medium":
                ghost_medium += 1

            dq = desc_quality[i]
            if dq == "empty" or dq == "too_short":
                desc_empty += 1

            la = lang[i]
            if la == "non_en":
                non_english += 1

            df = date_f[i]
            if df is not None and df != "ok" and df != "":
                date_out_of_range += 1

            # Aggregator
            if is_agg[i]:
                aggregator_total += 1
                if is_swe[i]:
                    swe_from_aggregators += 1

            # Missing data for key fields
            for field in key_fields:
                val = tbl[field][i]
                if val is None:
                    missing_counts[field] += 1
                elif isinstance(val, str) and val.strip() == "":
                    empty_counts[field] += 1
                elif isinstance(val, float) and (val != val):  # NaN check
                    missing_counts[field] += 1

        print(f"  Processed {rows_processed:,}/{total_rows:,} rows")

    # Compute medians for boilerplate stats
    median_full_swe = float(np.median(swe_desc_lengths)) if swe_desc_lengths else 0.0
    median_core_swe = float(np.median(swe_core_lengths)) if swe_core_lengths else 0.0

    # Build missing data section
    missing_data = {}
    for field in key_fields:
        missing_data[field] = {
            "missing_pct": round(missing_counts[field] / total_rows * 100, 2),
            "empty_pct": round(empty_counts[field] / total_rows * 100, 2),
        }

    # Build the report
    report = {
        "pipeline_version": "1.0",
        "run_date": "2026-03-19",
        "total_rows": total_rows,
        "total_columns": num_cols,
        "funnel": {
            "kaggle_arshkon_raw": KAGGLE_ARSHKON_RAW,
            "kaggle_asaniczka_raw": KAGGLE_ASANICZKA_RAW,
            "scraped_raw": SCRAPED_RAW,
            "after_dedup": total_rows,
            "final_swe": total_swe,
            "final_control": total_control,
        },
        "swe_by_source": dict(sorted(swe_by_source.items(), key=lambda x: -x[1])),
        "seniority_distribution_swe": dict(
            sorted(seniority_dist_swe.items(), key=lambda x: -x[1])
        ),
        "classification_rates": {
            "swe_regex": swe_regex,
            "swe_embedding": swe_embedding,
            "swe_unresolved": swe_unresolved,
            "seniority_imputed_title": seniority_imputed_title,
            "seniority_imputed_description": seniority_imputed_description,
            "seniority_unknown": seniority_unknown,
        },
        "missing_data": missing_data,
        "aggregator_stats": {
            "total_flagged": aggregator_total,
            "swe_from_aggregators": swe_from_aggregators,
        },
        "boilerplate_stats": {
            "median_full_length_swe": round(median_full_swe, 1),
            "median_core_length_swe": round(median_core_swe, 1),
        },
        "quality_flags": {
            "ghost_job_high": ghost_high,
            "ghost_job_medium": ghost_medium,
            "description_empty": desc_empty,
            "non_english": non_english,
            "date_out_of_range": date_out_of_range,
        },
    }

    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Quality report written to {OUTPUT_REPORT}")

    # ------------------------------------------------------------------
    # 3. Generate preprocessing_log.txt
    # ------------------------------------------------------------------
    print(f"\n[3/3] Writing preprocessing log ...")

    swe_src_lines = ""
    for src, cnt in sorted(swe_by_source.items(), key=lambda x: -x[1]):
        swe_src_lines += f"  {src:<35s} {cnt:>8,}\n"

    sen_lines = ""
    for level, cnt in sorted(seniority_dist_swe.items(), key=lambda x: -x[1]):
        pct = cnt / total_swe * 100 if total_swe else 0
        sen_lines += f"  {level:<30s} {cnt:>8,}  ({pct:.1f}%)\n"

    missing_lines = ""
    for field in key_fields:
        m = missing_data[field]
        missing_lines += (
            f"  {field:<25s}  missing={m['missing_pct']:.1f}%  empty={m['empty_pct']:.1f}%\n"
        )

    dedup_pct = DEDUP_REMOVED / TOTAL_RAW * 100

    log_text = f"""\
PREPROCESSING PIPELINE LOG
==========================
Pipeline version: 1.0
Run date: 2026-03-19
Input sources: 3 (kaggle_arshkon, kaggle_asaniczka, scraped)

DATA FUNNEL
-----------
Stage 1 - Ingest:          {TOTAL_RAW:>10,} rows ({KAGGLE_ARSHKON_RAW:,} arshkon + {KAGGLE_ASANICZKA_RAW:,} asaniczka + {SCRAPED_RAW:,} scraped)
Stage 2 - Aggregators:     {TOTAL_RAW:>10,} rows ({AGGREGATOR_FLAGGED:,} flagged as aggregator, {AGGREGATOR_FLAGGED/TOTAL_RAW*100:.1f}%)
Stage 3 - Boilerplate:     {TOTAL_RAW:>10,} rows (description_core created)
Stage 4 - Dedup:           {AFTER_DEDUP:>10,} rows ({DEDUP_REMOVED:,} duplicates removed, {dedup_pct:.1f}%)
Stage 5 - Classification:  {AFTER_DEDUP:>10,} rows ({total_swe:,} SWE, 3-tier regex+embedding)
Stage 6a - Company names:  {AFTER_DEDUP:>10,} rows (normalized company names)
Stage 6b-6e - Normalize:   {AFTER_DEDUP:>10,} rows (location, salary, remote, dates)
Stage 7 - Temporal:        {AFTER_DEDUP:>10,} rows (period, scrape_week aligned)
Stage 8 - Quality flags:   {AFTER_DEDUP:>10,} rows ({num_cols} columns, quality flags added)

FINAL DATASET
-------------
Total rows:           {total_rows:>10,}
Total columns:        {num_cols:>10}
SWE postings:         {total_swe:>10,}
Control postings:     {total_control:>10,}
Other postings:       {total_rows - total_swe - total_control:>10,}

SWE BY SOURCE
-------------
{swe_src_lines}
SENIORITY DISTRIBUTION (SWE only)
----------------------------------
{sen_lines}
CLASSIFICATION RATES
--------------------
  SWE regex:                    {swe_regex:>10,}
  SWE embedding:                {swe_embedding:>10,}
  SWE unresolved:               {swe_unresolved:>10,}
  Seniority from title:         {seniority_imputed_title:>10,}
  Seniority from description:   {seniority_imputed_description:>10,}
  Seniority unknown:            {seniority_unknown:>10,}

MISSING DATA
------------
{missing_lines}
AGGREGATOR STATS
----------------
  Total aggregator postings:    {aggregator_total:>10,}
  SWE from aggregators:         {swe_from_aggregators:>10,}

BOILERPLATE STATS (SWE only)
-----------------------------
  Median full description length: {median_full_swe:,.0f} chars
  Median core description length: {median_core_swe:,.0f} chars

QUALITY FLAGS
-------------
  Ghost job risk HIGH:          {ghost_high:>10,}
  Ghost job risk MEDIUM:        {ghost_medium:>10,}
  Description empty/too short:  {desc_empty:>10,}
  Non-English:                  {non_english:>10,}
  Date out of range:            {date_out_of_range:>10,}

OUTPUT FILES
------------
  data/unified.parquet          {OUTPUT_PARQUET.stat().st_size / (1024**3):.2f} GB
  data/quality_report.json      {OUTPUT_REPORT.stat().st_size / 1024:.1f} KB
  data/preprocessing_log.txt    (this file)
"""

    with open(OUTPUT_LOG, "w") as f:
        f.write(log_text)
    print(f"  Preprocessing log written to {OUTPUT_LOG}")

    elapsed = time.time() - t0
    print(f"\nAll outputs generated in {elapsed:.1f}s")
    print(f"  {OUTPUT_PARQUET}")
    print(f"  {OUTPUT_REPORT}")
    print(f"  {OUTPUT_LOG}")


if __name__ == "__main__":
    main()
