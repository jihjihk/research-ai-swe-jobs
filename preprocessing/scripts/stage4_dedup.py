#!/usr/bin/env python3
"""
Stage 4: Deduplication (memory-safe, two-pass)

Pass 1 — Load only dedup key columns (~0.3 GB) + description hashes.
          Compute which rows to keep via exact dedup, near-dedup, and
          multi-location flagging.
Pass 2 — Stream full data in chunks, filter to kept rows, write output.

This stays well under 31 GB RAM.

Input:  intermediate/stage3_boilerplate.parquet  (1.57M rows, ~6.4 GB)
Output: intermediate/stage4_dedup.parquet
"""

import gc
import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).parent.parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "stage4_dedup.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

CHUNK_SIZE = 200_000
FUZZY_TITLE_THRESHOLD = 85  # token_set_ratio threshold for near-dup titles
MAX_COMPANY_GROUP_FOR_FUZZY = 5000  # skip fuzzy for mega-groups (aggregators)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_text(text: str) -> str:
    """MD5 hash of text for fast equality check."""
    if not isinstance(text, str) or not text:
        return ""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def _count_filled(row: pd.Series, cols: list[str]) -> int:
    """Count how many of the given columns have non-null, non-empty values."""
    count = 0
    for c in cols:
        v = row.get(c)
        if v is not None and v != "" and not (isinstance(v, float) and np.isnan(v)):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Pass 1: Build the keep/drop index + description hashes
# ---------------------------------------------------------------------------

def pass1_build_index(input_path: Path) -> tuple[set, set, dict]:
    """
    Returns:
        keep_indices : set of integer row indices to keep
        multi_loc_indices : set of row indices flagged as multi-location
        source_funnels : dict[source -> {raw, after_exact, after_near}]
    """
    log.info("--- Pass 1: Loading key columns ---")
    t0 = time.time()

    # Columns needed for dedup decisions (lightweight)
    key_cols = [
        "job_id", "source", "title_normalized", "company_name_normalized",
        "location", "description_length", "description_core",
        "min_salary", "max_salary", "skills_raw", "seniority_native",
    ]

    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows

    # Read in chunks, compute description hash per chunk to avoid holding
    # full description_core strings in memory
    frames = []
    offset = 0
    for batch in pf.iter_batches(batch_size=CHUNK_SIZE, columns=key_cols):
        chunk = batch.to_pandas()
        chunk.index = range(offset, offset + len(chunk))

        # Compute hash of description_core, then drop the heavy column
        chunk["desc_hash"] = chunk["description_core"].apply(_hash_text)
        chunk.drop(columns=["description_core"], inplace=True)

        frames.append(chunk)
        offset += len(chunk)
        log.info(f"  Pass 1 read: {offset:,}/{total_rows:,}")

        del batch
        gc.collect()

    df = pd.concat(frames, ignore_index=False)
    del frames
    gc.collect()

    mem_gb = df.memory_usage(deep=True).sum() / 1e9
    log.info(f"  Key columns loaded: {len(df):,} rows, {mem_gb:.2f} GB")

    # Track source funnel
    sources = df["source"].unique().tolist()
    funnel = {src: {"raw": 0, "after_exact": 0, "after_near": 0} for src in sources}
    for src in sources:
        funnel[src]["raw"] = int((df["source"] == src).sum())

    # -------------------------------------------------------------------
    # 4a. Exact dedup: job_id
    # -------------------------------------------------------------------
    log.info("--- 4a. Exact dedup on job_id ---")
    before = len(df)
    dup_jobid = df.duplicated(subset=["job_id"], keep="first")
    n_jobid_dup = dup_jobid.sum()
    log.info(f"  job_id duplicates found: {n_jobid_dup:,}")
    drop_jobid = set(df.index[dup_jobid])

    # -------------------------------------------------------------------
    # 4a. Exact dedup: (title_normalized, company_name_normalized, location)
    # -------------------------------------------------------------------
    log.info("--- 4a. Exact dedup on (title, company, location) ---")
    # Work on the non-job_id-dup rows
    df_work = df.drop(index=list(drop_jobid))
    dup_exact = df_work.duplicated(
        subset=["title_normalized", "company_name_normalized", "location"],
        keep="first",
    )
    n_exact_dup = dup_exact.sum()
    log.info(f"  (title, company, location) duplicates found: {n_exact_dup:,}")
    drop_exact = set(df_work.index[dup_exact])
    del df_work, dup_exact
    gc.collect()

    all_drops_exact = drop_jobid | drop_exact
    remaining_indices = set(df.index) - all_drops_exact

    # Log after-exact counts
    for src in sources:
        mask = (df["source"] == src) & df.index.isin(remaining_indices)
        funnel[src]["after_exact"] = int(mask.sum())

    log.info(f"  After exact dedup: {len(remaining_indices):,} rows "
             f"(removed {len(all_drops_exact):,})")

    # -------------------------------------------------------------------
    # 4b. Near-duplicate detection (within company+location groups)
    # -------------------------------------------------------------------
    log.info("--- 4b. Near-duplicate detection ---")
    df_remaining = df.loc[sorted(remaining_indices)].copy()

    # Vectorized completeness score for tie-breaking
    completeness_cols = ["min_salary", "max_salary", "skills_raw", "seniority_native"]
    _comp = pd.DataFrame(index=df_remaining.index)
    for c in completeness_cols:
        col = df_remaining[c]
        if col.dtype == object:
            _comp[c] = col.notna() & (col != "")
        else:
            _comp[c] = col.notna()
    df_remaining["_completeness"] = _comp.sum(axis=1).values
    del _comp

    # Group by (company, location) — we only drop same-location near-dups,
    # so this is far more efficient than grouping by company alone
    cl_groups = df_remaining.groupby(
        ["company_name_normalized", "location"], sort=False
    )
    n_groups = cl_groups.ngroups
    log.info(f"  (company, location) groups: {n_groups:,}")

    near_dup_drops = set()
    skipped_large = 0
    processed_groups = 0
    fuzzy_comparisons = 0
    fuzzy_matches = 0

    for (company, loc), group in cl_groups:
        processed_groups += 1
        if processed_groups % 50000 == 0:
            log.info(f"  Near-dedup progress: {processed_groups:,}/{n_groups:,} groups "
                     f"({len(near_dup_drops):,} drops so far)")

        if len(group) < 2:
            continue

        if len(group) > MAX_COMPANY_GROUP_FOR_FUZZY:
            skipped_large += 1
            continue

        titles = group["title_normalized"].values
        indices = group.index.values
        completeness = group["_completeness"].values
        desc_lengths = group["description_length"].values

        n = len(group)
        already_dropped = set()

        for i in range(n):
            if indices[i] in already_dropped:
                continue
            for j in range(i + 1, n):
                if indices[j] in already_dropped:
                    continue

                fuzzy_comparisons += 1
                score = fuzz.token_set_ratio(titles[i], titles[j])

                if score >= FUZZY_TITLE_THRESHOLD:
                    fuzzy_matches += 1
                    # Keep the row with more complete fields; break ties by description length
                    if (completeness[j] > completeness[i]) or \
                       (completeness[j] == completeness[i] and
                        desc_lengths[j] > desc_lengths[i]):
                        already_dropped.add(indices[i])
                        break  # i is dropped, move to next i
                    else:
                        already_dropped.add(indices[j])

        near_dup_drops.update(already_dropped)

    log.info(f"  Fuzzy comparisons: {fuzzy_comparisons:,}")
    log.info(f"  Fuzzy matches (>= {FUZZY_TITLE_THRESHOLD}): {fuzzy_matches:,}")
    log.info(f"  Near-dup drops: {len(near_dup_drops):,}")
    log.info(f"  Skipped {skipped_large} (company,location) groups with >{MAX_COMPANY_GROUP_FOR_FUZZY} postings")

    all_drops = all_drops_exact | near_dup_drops
    keep_indices = set(df.index) - all_drops

    for src in sources:
        mask = (df["source"] == src) & df.index.isin(keep_indices)
        funnel[src]["after_near"] = int(mask.sum())

    log.info(f"  After near-dedup: {len(keep_indices):,} rows "
             f"(removed {len(near_dup_drops):,} near-dups)")

    # -------------------------------------------------------------------
    # 4c. Multi-location flagging
    # -------------------------------------------------------------------
    log.info("--- 4c. Multi-location flagging ---")
    df_kept = df.loc[sorted(keep_indices)]

    # Multi-location: same (title_normalized, company_name_normalized, desc_hash)
    # appearing at 2+ distinct locations
    ml_key = df_kept.groupby(
        ["title_normalized", "company_name_normalized", "desc_hash"]
    )["location"].transform("nunique")
    multi_loc_mask = ml_key >= 2
    multi_loc_indices = set(df_kept.index[multi_loc_mask])

    log.info(f"  Multi-location postings flagged: {len(multi_loc_indices):,}")

    del df, df_remaining, df_kept
    gc.collect()

    elapsed = time.time() - t0
    log.info(f"  Pass 1 complete in {elapsed:.1f}s")

    return keep_indices, multi_loc_indices, funnel


# ---------------------------------------------------------------------------
# Pass 2: Stream full data, filter to kept rows, write output
# ---------------------------------------------------------------------------

def pass2_write_output(
    input_path: Path,
    output_path: Path,
    keep_indices: set,
    multi_loc_indices: set,
):
    log.info("--- Pass 2: Writing deduplicated output ---")
    t0 = time.time()

    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows
    writer = None
    written = 0
    offset = 0

    for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
        batch_len = batch.num_rows
        chunk = batch.to_pandas()
        chunk.index = range(offset, offset + batch_len)

        # Filter to kept rows
        keep_mask = chunk.index.isin(keep_indices)
        chunk = chunk.loc[keep_mask].copy()

        offset += batch_len

        if len(chunk) == 0:
            del chunk, batch
            gc.collect()
            continue

        # Add dedup columns
        chunk["is_multi_location"] = chunk.index.isin(multi_loc_indices)

        # Force consistent dtypes for new columns
        chunk["is_multi_location"] = chunk["is_multi_location"].astype(bool)

        # Write
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)

        written += len(chunk)
        log.info(f"  Pass 2: written {written:,} rows (read {offset:,}/{total_rows:,})")

        del chunk, table, batch
        gc.collect()

    if writer is not None:
        writer.close()

    elapsed = time.time() - t0
    log.info(f"  Pass 2 complete: {written:,} rows written in {elapsed:.1f}s")
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_stage4():
    log.info("=" * 60)
    log.info("STAGE 4: Deduplication (two-pass, memory-safe)")
    log.info("=" * 60)

    input_path = INTERMEDIATE_DIR / "stage3_boilerplate.parquet"
    output_path = INTERMEDIATE_DIR / "stage4_dedup.parquet"

    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows
    log.info(f"Input: {total_rows:,} rows")

    # Pass 1: determine which rows to keep
    keep_indices, multi_loc_indices, funnel = pass1_build_index(input_path)

    # Pass 2: write filtered output
    written = pass2_write_output(input_path, output_path, keep_indices, multi_loc_indices)

    # -------------------------------------------------------------------
    # 4d. Dedup funnel report
    # -------------------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("DEDUP FUNNEL")
    log.info("=" * 60)
    log.info(f"  {'Source':<30} {'Raw':>10} {'Exact':>10} {'Near':>10} {'Pct kept':>10}")
    log.info(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    total_raw = 0
    total_exact = 0
    total_near = 0

    for src in sorted(funnel.keys()):
        f = funnel[src]
        pct = f["after_near"] / f["raw"] * 100 if f["raw"] > 0 else 0
        log.info(f"  {src:<30} {f['raw']:>10,} {f['after_exact']:>10,} {f['after_near']:>10,} {pct:>9.1f}%")
        total_raw += f["raw"]
        total_exact += f["after_exact"]
        total_near += f["after_near"]

    pct_total = total_near / total_raw * 100 if total_raw > 0 else 0
    log.info(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    log.info(f"  {'TOTAL':<30} {total_raw:>10,} {total_exact:>10,} {total_near:>10,} {pct_total:>9.1f}%")

    log.info(f"\n  Multi-location flagged: {len(multi_loc_indices):,}")
    log.info(f"  Output: {output_path}")
    log.info(f"  Rows written: {written:,}")


if __name__ == "__main__":
    run_stage4()
