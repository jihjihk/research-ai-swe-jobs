#!/usr/bin/env python3
"""
Stage 4: Company canonicalization + deduplication (memory-safe, two-pass)

Prep   — Build a canonical company-name lookup from Stage 2's
         `company_name_effective` field and persist it as an audit artifact.
Pass 1 — Load only dedup key columns (~0.3 GB) + description hashes.
          Compute which rows to keep via exact dedup, near-dedup, and
          multi-location flagging.
Pass 2 — Stream full data in chunks, filter to kept rows, write output.

This stays well under 31 GB RAM.

Input:  intermediate/stage3_boilerplate.parquet  (1.57M rows, ~6.4 GB)
Output: intermediate/stage4_dedup.parquet
        intermediate/stage4_company_name_lookup.parquet
"""

import gc
import hashlib
import logging
import re
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from company_name_canonicalization import build_company_name_lookup
from io_utils import cleanup_temp_file, prepare_temp_output, promote_temp_file, promote_null_schema

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
LOOKUP_OUTPUT_PATH = INTERMEDIATE_DIR / "stage4_company_name_lookup.parquet"
COMPANY_SUFFIXES = {
    "inc",
    "incorporated",
    "llc",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "limited",
    "plc",
    "gmbh",
    "ag",
    "sa",
    "lp",
    "llp",
    "pte",
    "holdings",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_text(text: str) -> str:
    """MD5 hash of lightly normalized text for fast equality check."""
    if not isinstance(text, str) or not text:
        return ""
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    if not normalized:
        return ""
    return hashlib.md5(normalized.encode("utf-8", errors="replace")).hexdigest()


def _normalize_entity_text(value: str) -> str:
    """Lowercase text with punctuation collapsed for deterministic keys."""
    if not isinstance(value, str):
        return ""
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_company_for_dedup(value: str) -> str:
    """Normalize company names and drop common legal suffixes."""
    tokens = _normalize_entity_text(value).split()
    while tokens and tokens[-1] in COMPANY_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _blank_mask(series: pd.Series) -> pd.Series:
    if len(series) == 0:
        return pd.Series(dtype=bool)
    return series.isna() | series.astype(str).str.strip().eq("")


def _resolve_company_effective(chunk: pd.DataFrame, company_source_col: str) -> pd.Series:
    if company_source_col in chunk.columns:
        effective = chunk[company_source_col].copy()
    else:
        effective = chunk["company_name"].copy()

    if "company_name" in chunk.columns and company_source_col != "company_name":
        effective = effective.where(~_blank_mask(effective), chunk["company_name"])

    return effective


def _normalize_location_for_dedup(value: str) -> str:
    """Normalize location strings for exact-location matching."""
    if not isinstance(value, str):
        return ""
    text = re.sub(r",\s*US$", "", value, flags=re.IGNORECASE)
    return _normalize_entity_text(text)


def _normalize_title_for_dedup(title: str) -> str:
    """Normalize titles for dedup without deleting level or discipline words."""
    if not isinstance(title, str):
        return ""
    value = title.lower().strip()
    value = re.sub(r"\s*[-–—]\s*(remote|hybrid|onsite|on-site)\s*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[^a-z0-9+#/& ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _titles_are_near_duplicates(a: str, b: str) -> bool:
    """Precision-first fuzzy match for titles within the same company/location."""
    if not a or not b:
        return False

    token_set = fuzz.token_set_ratio(a, b)
    if token_set < FUZZY_TITLE_THRESHOLD:
        return False

    # Block pairs whose only difference is a meaningful title token
    # like mechanical/electrical or senior/junior.
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    only_a = a_tokens - b_tokens
    only_b = b_tokens - a_tokens
    if len(only_a) == 1 and len(only_b) == 1:
        diff_a = next(iter(only_a))
        diff_b = next(iter(only_b))
        if len(diff_a) >= 3 and len(diff_b) >= 3 and fuzz.ratio(diff_a, diff_b) < 75:
            return False

    return fuzz.ratio(a, b) >= 75

# ---------------------------------------------------------------------------
# Pass 1: Build the keep/drop index + description hashes
# ---------------------------------------------------------------------------

def build_stage4_company_lookup(
    input_path: Path,
    lookup_output_path: Path,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str], str]:
    schema_names = pq.ParquetFile(input_path).schema.names
    company_source_col = "company_name_effective" if "company_name_effective" in schema_names else "company_name"

    log.info("--- Stage 4 prep: company canonicalization lookup ---")
    log.info("  Source field for canonicalization: %s", company_source_col)

    table = pq.read_table(str(input_path), columns=[company_source_col])
    names = table.column(company_source_col).to_pandas()
    lookup_df = build_company_name_lookup(names, source_column=company_source_col)
    lookup_df = lookup_df.rename(columns={company_source_col: "company_name_effective"})

    changed = int(
        (
            lookup_df["company_name_effective"]
            != lookup_df["company_name_canonical"]
        ).sum()
    )
    method_counts = lookup_df["company_name_canonical_method"].value_counts().to_dict()
    log.info("  Lookup rows: %s", f"{len(lookup_df):,}")
    log.info("  Canonical label differs from effective label: %s", f"{changed:,}")
    for method, count in sorted(method_counts.items()):
        log.info("    %-16s %10s", method + ":", f"{count:,}")

    lookup_df.to_parquet(lookup_output_path, index=False)
    log.info("  Wrote lookup artifact: %s", lookup_output_path)

    canonical_map = dict(
        zip(lookup_df["company_name_effective"], lookup_df["company_name_canonical"])
    )
    method_map = dict(
        zip(
            lookup_df["company_name_effective"],
            lookup_df["company_name_canonical_method"],
        )
    )
    return lookup_df, canonical_map, method_map, company_source_col


def pass1_build_index(
    input_path: Path,
    company_source_col: str,
    canonical_map: dict[str, str],
) -> tuple[set, set, dict]:
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
        "job_id", "source", "title", company_source_col, "company_name",
        "location", "description_length", "description_core",
        "skills_raw", "seniority_native",
    ]
    key_cols = list(dict.fromkeys(key_cols))

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
        effective_company = _resolve_company_effective(chunk, company_source_col)
        chunk["company_name_canonical"] = effective_company.map(canonical_map)
        missing_canonical = chunk["company_name_canonical"].isna()
        chunk.loc[missing_canonical, "company_name_canonical"] = effective_company.loc[missing_canonical]
        chunk["desc_hash"] = chunk["description_core"].apply(_hash_text)
        chunk["title_key"] = chunk["title"].apply(_normalize_title_for_dedup)
        chunk["company_key"] = chunk["company_name_canonical"].apply(_normalize_company_for_dedup)
        chunk["location_key"] = chunk["location"].apply(_normalize_location_for_dedup)
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
    dup_jobid = df.duplicated(subset=["job_id"], keep="first")
    n_jobid_dup = dup_jobid.sum()
    log.info(f"  job_id duplicates found: {n_jobid_dup:,}")
    drop_jobid = set(df.index[dup_jobid])

    # -------------------------------------------------------------------
    # 4a. Exact dedup: canonical opening key with matching description hash
    # -------------------------------------------------------------------
    log.info("--- 4a. Exact dedup on (company, title, location, desc_hash) ---")
    # Work on the non-job_id-dup rows
    df_work = df.drop(index=list(drop_jobid))
    df_work_with_desc = df_work[df_work["desc_hash"] != ""]
    dup_exact = df_work_with_desc.duplicated(
        subset=["company_key", "title_key", "location_key", "desc_hash"],
        keep="first",
    )
    n_exact_dup = dup_exact.sum()
    log.info(f"  exact opening duplicates found: {n_exact_dup:,}")
    log.info(
        "  rows skipped for exact opening dedup due to missing description: %s",
        f"{len(df_work) - len(df_work_with_desc):,}",
    )
    drop_exact = set(df_work_with_desc.index[dup_exact])
    del df_work, df_work_with_desc, dup_exact
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
    # 4b. Near-duplicate detection (within company+location+description groups)
    # -------------------------------------------------------------------
    log.info("--- 4b. Near-duplicate detection ---")
    df_remaining = df.loc[sorted(remaining_indices)].copy()

    # Vectorized completeness score for tie-breaking
    completeness_cols = ["skills_raw", "seniority_native"]
    _comp = pd.DataFrame(index=df_remaining.index)
    for c in completeness_cols:
        col = df_remaining[c]
        if col.dtype == object:
            _comp[c] = col.notna() & (col != "")
        else:
            _comp[c] = col.notna()
    df_remaining["_completeness"] = _comp.sum(axis=1).values
    del _comp

    # Fuzzy matching only runs when description evidence agrees.
    df_fuzzy = df_remaining[df_remaining["desc_hash"] != ""]
    cl_groups = df_fuzzy.groupby(["company_key", "location_key", "desc_hash"], sort=False)
    n_groups = cl_groups.ngroups
    log.info(f"  (company, location, desc_hash) fuzzy groups: {n_groups:,}")

    near_dup_drops = set()
    skipped_large = 0
    processed_groups = 0
    fuzzy_comparisons = 0
    fuzzy_matches = 0

    for _, group in cl_groups:
        processed_groups += 1
        if processed_groups % 50000 == 0:
            log.info(f"  Near-dedup progress: {processed_groups:,}/{n_groups:,} groups "
                     f"({len(near_dup_drops):,} drops so far)")

        if len(group) < 2:
            continue

        if len(group) > MAX_COMPANY_GROUP_FOR_FUZZY:
            skipped_large += 1
            continue

        titles = group["title_key"].values
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
                if _titles_are_near_duplicates(titles[i], titles[j]):
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
    log.info(
        f"  Skipped {skipped_large} (company,location,desc_hash) groups "
        f"with >{MAX_COMPANY_GROUP_FOR_FUZZY} postings"
    )

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

    # Multi-location: same canonical opening at 2+ distinct normalized locations.
    ml_key = df_kept.groupby(
        ["company_key", "title_key", "desc_hash"]
    )["location_key"].transform("nunique")
    multi_loc_mask = (df_kept["desc_hash"] != "") & (ml_key >= 2)
    multi_loc_indices = set(df_kept.index[multi_loc_mask])

    log.info(f"  Multi-location postings flagged: {len(multi_loc_indices):,}")

    del df, df_remaining, df_fuzzy, df_kept
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
    company_source_col: str,
    canonical_map: dict[str, str],
    method_map: dict[str, str],
):
    log.info("--- Pass 2: Writing deduplicated output ---")
    t0 = time.time()

    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows

    # Build stable output schema (null→string promotion + new columns)
    output_schema = promote_null_schema(pf.schema_arrow, extra_fields=[
        pa.field("company_name_canonical", pa.string()),
        pa.field("company_name_canonical_method", pa.string()),
        pa.field("is_multi_location", pa.bool_()),
    ])

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
        effective_company = _resolve_company_effective(chunk, company_source_col)
        chunk["company_name_effective"] = effective_company
        chunk["company_name_canonical"] = effective_company.map(canonical_map)
        missing_canonical = chunk["company_name_canonical"].isna()
        chunk.loc[missing_canonical, "company_name_canonical"] = effective_company.loc[missing_canonical]
        chunk["company_name_canonical_method"] = effective_company.map(method_map)
        missing_method = chunk["company_name_canonical_method"].isna()
        chunk.loc[missing_method, "company_name_canonical_method"] = "passthrough"
        chunk["is_multi_location"] = chunk.index.isin(multi_loc_indices)

        # Force consistent dtypes for new columns
        chunk["company_name_effective"] = chunk["company_name_effective"].astype("string")
        chunk["company_name_canonical"] = chunk["company_name_canonical"].astype("string")
        chunk["company_name_canonical_method"] = chunk["company_name_canonical_method"].astype("string")
        chunk["is_multi_location"] = chunk["is_multi_location"].astype(bool)

        # Write (cast to unified schema)
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        table = table.cast(output_schema)
        if writer is None:
            writer = pq.ParquetWriter(output_path, output_schema)
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
    log.info("STAGE 4: Company canonicalization + deduplication")
    log.info("=" * 60)

    input_path = INTERMEDIATE_DIR / "stage3_boilerplate.parquet"
    output_path = INTERMEDIATE_DIR / "stage4_dedup.parquet"
    tmp_output_path = prepare_temp_output(output_path)
    tmp_lookup_output_path = prepare_temp_output(LOOKUP_OUTPUT_PATH)

    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows
    log.info(f"Input: {total_rows:,} rows")

    try:
        _, canonical_map, method_map, company_source_col = build_stage4_company_lookup(
            input_path,
            tmp_lookup_output_path,
        )

        # Pass 1: determine which rows to keep
        keep_indices, multi_loc_indices, funnel = pass1_build_index(
            input_path,
            company_source_col,
            canonical_map,
        )

        # Pass 2: write filtered output
        written = pass2_write_output(
            input_path,
            tmp_output_path,
            keep_indices,
            multi_loc_indices,
            company_source_col,
            canonical_map,
            method_map,
        )
    except Exception:
        cleanup_temp_file(tmp_lookup_output_path)
        cleanup_temp_file(tmp_output_path)
        raise

    promote_temp_file(tmp_lookup_output_path, LOOKUP_OUTPUT_PATH)
    promote_temp_file(tmp_output_path, output_path)

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
