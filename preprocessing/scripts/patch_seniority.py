#!/usr/bin/env python3
"""
Post-processing patch: Fix seniority normalization and create seniority_final.

Issues fixed:
1. asaniczka uses "mid senior" vs "mid-senior level" — normalize
2. Create seniority_final: prefer native label, fall back to imputed
3. Recompute seniority_3level from seniority_final

Memory-safe: chunked pyarrow processing.
"""

import gc
import re
import logging
import time
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "patch_seniority.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

SENIORITY_NORM = {
    "mid senior": "mid-senior level",
    "mid senior level": "mid-senior level",
    "mid-senior level": "mid-senior level",
    "senior": "mid-senior level",
    "entry level": "entry level",
    "entry_level": "entry level",
    "junior": "entry level",
    "associate": "associate",
    "director": "director",
    "executive": "executive",
    "internship": "internship",
    "not applicable": "",
}


def to_3level(s: str) -> str:
    if s in ("entry level", "internship"):
        return "junior"
    if s == "associate":
        return "mid"
    if s in ("mid-senior level", "director", "executive"):
        return "senior"
    return "unknown"


def run():
    input_path = DATA_DIR / "unified.parquet"
    output_path = DATA_DIR / "unified.parquet.tmp"

    log.info("Patching seniority normalization...")

    pf = pq.ParquetFile(input_path)
    writer = None
    processed = 0
    t0 = time.time()

    # Accumulators for SWE seniority stats
    seniority_counts = {}

    for batch in pf.iter_batches(batch_size=200_000):
        chunk = batch.to_pandas()

        # 1. Normalize seniority_native
        chunk["seniority_native"] = chunk["seniority_native"].apply(
            lambda x: SENIORITY_NORM.get(str(x).strip().lower(), str(x).strip().lower()) if pd.notna(x) and str(x).strip() else ""
        )

        # 2. Create seniority_final: native if filled, else imputed
        chunk["seniority_final"] = chunk["seniority_native"].where(
            chunk["seniority_native"].str.len() > 0,
            chunk["seniority_imputed"]
        )

        # 3. Recompute seniority_3level from final
        chunk["seniority_3level"] = chunk["seniority_final"].apply(to_3level)

        # 4. Update seniority_source to reflect the final choice
        chunk["seniority_source"] = np.where(
            chunk["seniority_native"].str.len() > 0,
            "native_label",
            chunk["seniority_source"]
        )

        # Accumulate stats
        swe_mask = chunk["is_swe"]
        if swe_mask.any():
            for level, count in chunk.loc[swe_mask, "seniority_final"].value_counts().items():
                seniority_counts[level] = seniority_counts.get(level, 0) + count

        # Write
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)

        processed += len(chunk)
        log.info(f"  {processed:,}/{pf.metadata.num_rows:,}")

        del chunk, table, batch
        gc.collect()

    writer.close()

    # Replace original
    import shutil
    shutil.move(str(output_path), str(input_path))

    log.info(f"\nDone in {time.time()-t0:.0f}s")
    log.info("\nSWE seniority_final distribution:")
    total_swe = sum(seniority_counts.values())
    for level in sorted(seniority_counts, key=seniority_counts.get, reverse=True):
        count = seniority_counts[level]
        log.info(f"  {level:<25} {count:>6,} ({count/total_swe:.1%})")

    unknown = seniority_counts.get("unknown", 0)
    log.info(f"\nUnknown rate: {unknown:,} / {total_swe:,} = {unknown/total_swe:.1%}")


if __name__ == "__main__":
    run()
