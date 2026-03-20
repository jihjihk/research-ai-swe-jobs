#!/usr/bin/env python3
"""
Stage 3: Boilerplate Removal (memory-safe, chunked)

Processes parquet in chunks of ~200K rows to stay under 31GB RAM.

Strips non-job-content sections from descriptions:
  - EEO/diversity statements
  - Company "About Us" / benefits / application sections

Produces description_core (role + responsibilities + requirements + nice-to-haves).

Input:  intermediate/stage2_aggregators.parquet
Output: intermediate/stage3_boilerplate.parquet
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
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "stage3_boilerplate.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

CHUNK_SIZE = 200_000  # rows per chunk — keeps peak memory ~4-5 GB

# ---------------------------------------------------------------------------
# Compiled patterns (module-level, allocated once)
# ---------------------------------------------------------------------------
_HEADER_RE = re.compile(
    r'(?im)^[\s#*]*('
    r'about\s+(?:us|the\s+company|our\s+company|the\s+role|the\s+position|the\s+job|the\s+opportunity|the\s+team)'
    r'|who\s+we\s+are|our\s+(?:company|mission|story|values)|company\s+(?:overview|description)'
    r'|(?:key\s+)?responsibilities|what\s+you.?ll\s+do|your\s+role|the\s+role|duties|job\s+dut\w*|job\s+description|role\s+overview|position\s+overview'
    r'|(?:minimum\s+|basic\s+)?requirements?|qualifications?|what\s+you.?ll\s+need|what\s+we.?re\s+looking\s+for|must\s+have|minimum|skills?\s+(?:and\s+)?(?:requirements?|qualifications?)|required\s+skills?'
    r'|nice\s+to\s+have|preferred|bonus|plus|desired|additional|strongly\s+preferred'
    r'|benefits?|perks|what\s+we\s+offer|compensation|we\s+offer|salary|total\s+rewards?|our\s+benefits|why\s+(?:join\s+)?us'
    r'|equal\s+(?:employment\s+)?opportunity|eeo|diversity|we\s+are\s+(?:an?\s+)?equal|non-?discrimination|ada\s+statement|accommodation'
    r'|how\s+to\s+apply|to\s+apply|application|apply\s+now|next\s+steps?'
    r')[\s:]*$'
)

_KEEP_TYPES = frozenset({'about_role', 'responsibilities', 'requirements', 'nice_to_have', 'unknown'})

_EEO_RE = re.compile(
    r'(?i)(?:'
    r'equal\s+(?:employment\s+)?opportunity\s+employer'
    r'|we\s+are\s+(?:an?\s+)?equal\s+opportunity'
    r'|does\s+not\s+discriminate\s+(?:on\s+the\s+basis|based\s+on)'
    r'|regardless\s+of\s+race,?\s*color,?\s*religion'
    r'|all\s+qualified\s+applicants\s+will\s+receive\s+consideration'
    r'|m/f/disability/vet'
    r'|e-?verify\s+(?:employer|participant)'
    r')'
)


def _classify(header: str) -> str:
    h = header.lower().strip()
    if re.match(r'about\s+the\s+(?:role|position|job|opportunity|team)', h):
        return 'about_role'
    if re.match(r'(?:key\s+)?responsibilities|what\s+you.?ll\s+do|your\s+role|the\s+role|dut|job\s+desc|role\s+over|position\s+over', h):
        return 'responsibilities'
    if re.match(r'(?:minimum|basic)?\.?s*requirements?|qualifications?|what\s+you.?ll\s+need|what\s+we.?re|must\s+have|minimum|skills?|required', h):
        return 'requirements'
    if re.match(r'nice|preferred|bonus|plus|desired|additional|strongly', h):
        return 'nice_to_have'
    if re.match(r'benefits?|perks|what\s+we\s+offer|compensation|we\s+offer|salary|total|our\s+benefits|why', h):
        return 'benefits'
    if re.match(r'equal|eeo|diversity|we\s+are|non-?disc|ada|accommodation', h):
        return 'eeo'
    if re.match(r'how\s+to|to\s+apply|application|apply|next\s+step', h):
        return 'application'
    if re.match(r'about\s+(?:us|the\s+company|our)|who\s+we|our\s+(?:company|mission|story|values)|company', h):
        return 'about_company'
    return 'unknown'


_TAIL_NOISE = re.compile(
    r'(?im)(?:'
    r'show\s+more\s*[\n\r]*show\s+less'
    r'|show\s+more$'
    r'|show\s+less$'
    r')\s*$'
)

_URL_LINE = re.compile(r'^\s*https?://\S+\s*$', re.MULTILINE)

# Staffing agency boilerplate at the START of descriptions
_STAFFING_INTRO = re.compile(
    r'(?is)^(?:'
    r'(?:this\s+(?:role|position)\s+is\s+for\s+a\s+client\s+of\s+|who\s+is\s+)'
    r'.*?(?:\.com/?\s*\n|our\s+client\s*[\n:])'  # ends at URL or "Our Client"
    r')',
    re.DOTALL
)

def _strip_common_noise(text: str) -> str:
    """Remove common noise patterns that don't need section headers."""
    # LinkedIn "Show more / Show less"
    text = _TAIL_NOISE.sub('', text)
    # Standalone URL lines
    text = _URL_LINE.sub('', text)
    # Staffing agency intro paragraphs
    text = _STAFFING_INTRO.sub('', text)
    return text.strip()


def _is_boilerplate_paragraph(para: str) -> bool:
    """Check if a paragraph is boilerplate (EEO, benefits, application instructions)."""
    if _EEO_RE.search(para):
        return True
    # Benefits-style paragraphs (list of perks without job content)
    if re.search(r'(?i)(?:401\s*\(?k\)?|dental|vision|pto|paid\s+time\s+off|tuition\s+reimbursement)', para):
        if not re.search(r'(?i)(?:develop|engineer|design|implement|build|architect|code|software)', para):
            return True
    return False


def extract_core(text: str) -> str:
    """Extract core job content, stripping boilerplate sections and EEO."""
    if not isinstance(text, str) or len(text) < 50:
        return text or ''

    # Phase 0: strip common noise (Show more/less, URLs, staffing intros)
    text = _strip_common_noise(text)

    lines = text.split('\n')
    sections = []
    cur_type = 'unknown'
    cur_lines = []

    for line in lines:
        m = _HEADER_RE.match(line.strip())
        if m:
            if cur_lines:
                sections.append((cur_type, '\n'.join(cur_lines)))
            cur_type = _classify(m.group(1))
            cur_lines = []
        else:
            cur_lines.append(line)

    if cur_lines:
        sections.append((cur_type, '\n'.join(cur_lines)))

    types_found = {s[0] for s in sections}

    # No headers detected — strip boilerplate paragraphs individually
    if types_found == {'unknown'}:
        paras = re.split(r'\n\s*\n', text)
        return '\n\n'.join(p for p in paras if not _is_boilerplate_paragraph(p)).strip()

    # Keep relevant sections, strip boilerplate from each
    parts = []
    for stype, content in sections:
        if stype in _KEEP_TYPES:
            paras = re.split(r'\n\s*\n', content)
            cleaned = '\n\n'.join(p for p in paras if not _is_boilerplate_paragraph(p)).strip()
            if cleaned:
                parts.append(cleaned)

    result = '\n\n'.join(parts)

    # Fallback: if we stripped too aggressively
    if len(result) < 100 and len(text) > 200:
        paras = re.split(r'\n\s*\n', text)
        return '\n\n'.join(p for p in paras if not _EEO_RE.search(p)).strip()

    return result


def process_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
    """Process a single chunk: add description_core, core_length, boilerplate_flag."""
    chunk_df["description_core"] = chunk_df["description"].apply(extract_core)
    chunk_df["core_length"] = chunk_df["description_core"].str.len().astype("float64")

    removal_pct = 1 - (chunk_df["core_length"] / chunk_df["description_length"].clip(lower=1))
    chunk_df["boilerplate_flag"] = "ok"
    chunk_df.loc[(removal_pct > 0.80) & (chunk_df["description_length"] > 200), "boilerplate_flag"] = "over_removed"
    chunk_df.loc[(removal_pct < 0.10) & (chunk_df["description_length"] > 500), "boilerplate_flag"] = "under_removed"
    chunk_df.loc[chunk_df["core_length"] < 50, "boilerplate_flag"] = "empty_core"

    return chunk_df


# ---------------------------------------------------------------------------
# Main — chunked processing
# ---------------------------------------------------------------------------
def run_stage3():
    log.info("=" * 60)
    log.info("STAGE 3: Boilerplate Removal (chunked, memory-safe)")
    log.info("=" * 60)

    input_path = INTERMEDIATE_DIR / "stage2_aggregators.parquet"
    output_path = INTERMEDIATE_DIR / "stage3_boilerplate.parquet"

    # Load company name lookup (small — 3.4 MB)
    lookup_path = INTERMEDIATE_DIR / "company_name_lookup.parquet"
    lookup = None
    if lookup_path.exists():
        lookup = pd.read_parquet(lookup_path)
        log.info(f"  Company name lookup loaded: {len(lookup):,} rows")

    # Get total row count without loading the file
    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows
    log.info(f"  Input: {total_rows:,} rows, {pf.metadata.num_row_groups} row groups")

    # Process in chunks, writing to parquet incrementally
    writer = None
    processed = 0
    t0 = time.time()

    # Stats accumulators
    stats = {src: {"full": [], "core": []} for src in
             ["kaggle_arshkon_2024", "kaggle_asaniczka_2024", "scraped_linkedin_2026", "scraped_indeed_2026"]}
    flag_counts = {"ok": 0, "over_removed": 0, "under_removed": 0, "empty_core": 0}

    for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
        chunk = batch.to_pandas()

        # Join company name lookup
        if lookup is not None:
            chunk = chunk.merge(lookup, on="company_name", how="left")
            chunk["company_name_normalized"] = chunk["company_name_normalized"].fillna(chunk["company_name"])
        else:
            chunk["company_name_normalized"] = chunk["company_name"]

        # Process
        chunk = process_chunk(chunk)

        # Accumulate stats (SWE only)
        for src in stats:
            swe_mask = (chunk["source"] == src) & (chunk["is_swe"])
            if swe_mask.any():
                stats[src]["full"].extend(chunk.loc[swe_mask, "description_length"].tolist())
                stats[src]["core"].extend(chunk.loc[swe_mask, "core_length"].tolist())

        for flag in ["ok", "over_removed", "under_removed", "empty_core"]:
            flag_counts[flag] += (chunk["boilerplate_flag"] == flag).sum()

        # Write chunk to parquet
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)

        processed += len(chunk)
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0
        log.info(f"  Chunk done: {processed:,}/{total_rows:,} ({processed/total_rows:.0%}) — {rate:.0f} rows/s")

        # Free memory
        del chunk, table, batch
        gc.collect()

    if writer is not None:
        writer.close()

    # --- Report stats ---
    log.info("\n--- BOILERPLATE REMOVAL SUMMARY ---")
    log.info(f"  {'Source':<30} {'Full':>8} {'Core':>8} {'Removed':>8}")
    for src, vals in stats.items():
        if vals["full"]:
            full_med = float(np.median(vals["full"]))
            core_med = float(np.median(vals["core"]))
            log.info(f"  {src:<30} {full_med:>8.0f} {core_med:>8.0f} {full_med-core_med:>8.0f}")

    log.info(f"\n  Boilerplate flags:")
    for flag, count in sorted(flag_counts.items()):
        log.info(f"    {flag}: {count:,} ({count/total_rows:.1%})")

    log.info(f"\n  Total time: {time.time()-t0:.1f}s")
    log.info(f"  Saved to {output_path}")


if __name__ == "__main__":
    run_stage3()
