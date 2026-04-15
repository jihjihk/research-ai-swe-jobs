#!/usr/bin/env python3
"""
Stage 2: Aggregator / Staffing Company Handling

Identifies and flags aggregator/staffing company postings.
Attempts to extract real employer from description and derives the
`company_name_effective` field used for Stage 4 canonicalization and dedup.

Input:  intermediate/stage1_unified.parquet
Output: intermediate/stage2_aggregators.parquet
        (adds is_aggregator, real_employer, company_name_effective columns)
"""

import gc
import re
import logging
from pathlib import Path
from collections import Counter

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from io_utils import prepare_temp_output, promote_temp_file, cleanup_temp_file, promote_null_schema

PROJECT_ROOT = Path(__file__).parent.parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "stage2_aggregators.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Aggregator/Staffing company list
# ---------------------------------------------------------------------------
AGGREGATORS = {
    # Job boards / aggregators
    "lensa", "jobs via dice", "dice", "jobot", "cybercoders",
    "hired", "ziprecruiter", "talentally", "ladders",
    "efinancialcareers", "ventureloop", "builtin", "jobs for humanity",
    "jobs via efinancialcareers", "clearancejobs", "clickjobs.io",
    # Staffing companies
    "actalent", "randstad", "robert half", "teksystems",
    "kforce", "insight global", "motion recruitment", "harnham",
    "apex systems", "modis", "experis", "manpowergroup",
    "kelly services", "adecco", "hays", "michael page",
    "collabera", "infosys bpm", "wipro", "tata consultancy",
    "tata consultancy services", "tata consultancy services (tcs)",
    "cognizant", "capgemini", "revature", "smoothstack",
    "recruiting from scratch", "aston carter", "solomon page", "turing",
    "creative circle", "applicantz", "aditi consulting",
    "ask consulting", "inspyr solutions", "net2source inc.",
    "net2source inc", "net2source",
    # Data annotation / crowdwork (suspected non-traditional SWE)
    "dataannotation", "data annotation", "appen", "scale ai",
    "remotasks", "clickworker",
}

# Patterns for fuzzy matching aggregator names
AGGREGATOR_PATTERNS = [
    re.compile(r'(?i)\blensa\b'),
    re.compile(r'(?i)\bjobs?\s+via\b'),
    re.compile(r'(?i)\bjobs?\s*via\s*dice\b'),
    re.compile(r'(?i)\bjobs?\s*via\s*efinancialcareers\b'),
    re.compile(r'(?i)\b(dice|dice\.com)\b'),
    re.compile(r'(?i)\bjobot\b'),
    re.compile(r'(?i)\bcybercoders\b'),
    re.compile(r'(?i)\bclickjobs(?:\.io)?\b'),
    re.compile(r'(?i)\bclearancejobs\b'),
    re.compile(r'(?i)\bziprecruiter\b'),
    re.compile(r'(?i)\bjobs?\s+for\s+humanity\b'),
    re.compile(r'(?i)\bactalent\b'),
    re.compile(r'(?i)\baston\s*carter\b'),
    re.compile(r'(?i)\brandstad\b'),
    re.compile(r'(?i)\brobert\s*half\b'),
    re.compile(r'(?i)\bteksystems\b'),
    re.compile(r'(?i)\bkforce\b'),
    re.compile(r'(?i)\binsight\s*global\b'),
    re.compile(r'(?i)\bapex\s*systems\b'),
    re.compile(r'(?i)\bsolomon\s+page\b'),
    re.compile(r'(?i)\bcreative\s+circle\b'),
    re.compile(r'(?i)\brecruiting\s+from\s+scratch\b'),
    re.compile(r'(?i)\bdata\s*annotation\b'),
    re.compile(r'(?i)\brevature\b'),
    re.compile(r'(?i)\bsmoothstack\b'),
    re.compile(r'(?i)\btalentally\b'),
    re.compile(r'(?i)\bharnham\b'),
    re.compile(r'(?i)\bmotion\s*recruitment\b'),
    re.compile(r'(?i)\baditi\s+consulting\b'),
    re.compile(r'(?i)\bask\s+consulting\b'),
    re.compile(r'(?i)\binspyr\s+solutions\b'),
    re.compile(r'(?i)\bnet2source(?:\s+inc\.?)?\b'),
    re.compile(r'(?i)\btata\s+consultancy(?:\s+services)?(?:\s+\(tcs\))?\b'),
    re.compile(r'(?i)\bapplicantz\b'),
    re.compile(r"\bturing\b(?!\s+(pharmaceutical|medical))", re.IGNORECASE),
]


def is_aggregator(company_name: str) -> bool:
    """Check if a company name matches the aggregator list."""
    if not isinstance(company_name, str):
        return False
    name_lower = company_name.strip().lower()
    # Exact match
    if name_lower in AGGREGATORS:
        return True
    # Pattern match
    for pat in AGGREGATOR_PATTERNS:
        if pat.search(company_name):
            return True
    return False


# Real employer extraction patterns
REAL_EMPLOYER_PATTERNS = [
    # "Our client, [Company Name], is..."
    re.compile(r'(?i)our\s+client[,:]?\s*([A-Z][A-Za-z0-9\s&.,-]+?)(?:,|\s+is|\s+has|\s+seeks|\s+needs|\.)'),
    # "on behalf of [Company Name]"
    re.compile(r'(?i)on\s+behalf\s+of\s+([A-Z][A-Za-z0-9\s&.,-]+?)(?:,|\s+is|\s+we|\.)'),
    # "partnered with [Company Name]"
    re.compile(r'(?i)partner(?:ed|ing)\s+with\s+([A-Z][A-Za-z0-9\s&.,-]+?)(?:,|\s+to|\s+for|\.)'),
    # "working with [Company Name]"
    re.compile(r'(?i)working\s+with\s+([A-Z][A-Za-z0-9\s&.,-]+?)\s+(?:to\s+find|who\s+is|looking)'),
    # Lensa: "Lensa is the leading... This job is also available..." then company name
    re.compile(r'(?i)(?:job|position|role)\s+(?:is\s+)?(?:with|at|for)\s+([A-Z][A-Za-z0-9\s&.,-]+?)(?:\.|,|\s+in\b|\s+located)'),
]

BAD_EMPLOYER_TERMS = re.compile(
    r'(?i)\b('
    r'client|clients|company|companies|organization|team|role|position|opportunity|'
    r'seeking|looking|hiring|available|access|extension|contract|'
    r'engineer|developer|manager|director|architect|scientist|analyst|'
    r'specialist|consultant|administrator|technician|recruiter|designer|'
    r'located|location|remote|hybrid|onsite|site|week|weeks|sector|industry|'
    r'ownership|detail\s+oriented|direct\s+client|integrators'
    r')\b'
)

ORG_SUFFIX_HINT = re.compile(
    r'(?i)\b('
    r'inc|llc|ltd|corp|corporation|company|co|group|systems|solutions|'
    r'technologies|technology|labs|lab|health|partners|holdings|bank|'
    r'university|international|networks|software|services'
    r')\.?\b'
)

LEADING_FRAGMENT = re.compile(r'(?i)^(in|at|for|with|to|on|of|from)\b')


def _clean_employer_candidate(candidate: str) -> str:
    cleaned = candidate.strip(" \t\n\r-.,:;()[]{}\"'")
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned


def _looks_like_organization_name(candidate: str) -> bool:
    if LEADING_FRAGMENT.match(candidate):
        return False
    tokens = candidate.split()
    if len(tokens) == 0 or len(tokens) > 6:
        return False
    uppercase_tokens = sum(1 for token in tokens if any(ch.isupper() for ch in token))
    if ORG_SUFFIX_HINT.search(candidate):
        return True
    if len(tokens) == 1:
        return uppercase_tokens == 1 and tokens[0].isalpha()
    return uppercase_tokens >= 2


def extract_real_employer(description: str, company_name: str) -> str | None:
    """Try to extract the real employer from an aggregator posting description."""
    if not isinstance(description, str) or len(description) < 50:
        return None

    for pattern in REAL_EMPLOYER_PATTERNS:
        match = pattern.search(description[:2000])  # Only check first 2000 chars
        if match:
            employer = _clean_employer_candidate(match.group(1))
            # Precision-first validation: reject obvious fragments and role phrases.
            if (
                2 < len(employer) < 80
                and any(ch.isupper() for ch in employer)
                and _looks_like_organization_name(employer)
                and not BAD_EMPLOYER_TERMS.search(employer)
                and not is_aggregator(employer)
                and employer.lower() != company_name.lower()
            ):
                return employer
    return None


# ---------------------------------------------------------------------------
# Main (chunked, memory-safe)
# ---------------------------------------------------------------------------
CHUNK_SIZE = 50_000


def _process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Apply aggregator flagging, real employer extraction, and effective name to a chunk."""
    chunk["is_aggregator"] = chunk["company_name"].apply(is_aggregator)

    chunk["real_employer"] = None
    agg_mask = chunk["is_aggregator"]
    if agg_mask.any():
        chunk.loc[agg_mask, "real_employer"] = chunk.loc[agg_mask].apply(
            lambda r: extract_real_employer(r["description"], r["company_name"]),
            axis=1,
        )

    chunk["company_name_effective"] = chunk["real_employer"]
    effective_missing = chunk["company_name_effective"].isna() | chunk["company_name_effective"].astype(str).str.strip().eq("")
    chunk.loc[effective_missing, "company_name_effective"] = chunk.loc[effective_missing, "company_name"]

    return chunk


def run_stage2():
    log.info("=" * 60)
    log.info("STAGE 2: Aggregator / Staffing Company Handling (chunked, memory-safe)")
    log.info("=" * 60)

    input_path = INTERMEDIATE_DIR / "stage1_unified.parquet"
    output_path = INTERMEDIATE_DIR / "stage2_aggregators.parquet"
    tmp_output_path = prepare_temp_output(output_path)

    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows
    log.info(f"  Input: {total_rows:,} rows, {pf.metadata.num_row_groups} row groups")

    # Build output schema: input schema + new columns, with null types promoted to string
    output_schema = promote_null_schema(pf.schema_arrow, extra_fields=[
        pa.field("is_aggregator", pa.bool_()),
        pa.field("real_employer", pa.string()),
        pa.field("company_name_effective", pa.string()),
    ])

    writer = None
    processed = 0

    # Lightweight accumulators for summary stats (no full-data copies)
    total_agg = 0
    total_extracted = 0
    agg_by_source: dict[str, tuple[int, int]] = {}   # source -> (agg_count, total)
    top_agg_companies: Counter = Counter()
    sample_extractions: list[tuple[str, str, str]] = []  # (company, real_employer, title)
    da_count = 0

    try:
        for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
            chunk = batch.to_pandas()
            chunk = _process_chunk(chunk)

            # Accumulate stats
            agg_mask = chunk["is_aggregator"]
            chunk_agg = int(agg_mask.sum())
            total_agg += chunk_agg
            total_extracted += int(chunk.loc[agg_mask, "real_employer"].notna().sum())

            for src in chunk["source"].dropna().unique():
                src_mask = chunk["source"] == src
                prev_agg, prev_total = agg_by_source.get(src, (0, 0))
                agg_by_source[src] = (
                    prev_agg + int((src_mask & agg_mask).sum()),
                    prev_total + int(src_mask.sum()),
                )

            if chunk_agg > 0:
                top_agg_companies.update(
                    chunk.loc[agg_mask, "company_name"].dropna().tolist()
                )

            if len(sample_extractions) < 20:
                extracted_rows = chunk.loc[
                    agg_mask & chunk["real_employer"].notna(),
                    ["company_name", "real_employer", "title"],
                ]
                for _, row in extracted_rows.iterrows():
                    if len(sample_extractions) >= 20:
                        break
                    sample_extractions.append(
                        (row["company_name"], row["real_employer"], row["title"])
                    )

            da_count += int(
                chunk["company_name"].str.contains(r'(?i)data\s*annotation', na=False).sum()
            )

            # Write chunk, casting to unified output schema
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            table = table.cast(output_schema)
            if writer is None:
                writer = pq.ParquetWriter(tmp_output_path, output_schema)
            writer.write_table(table)

            processed += len(chunk)
            del chunk, table, batch
            gc.collect()
            log.info(f"  Processed {processed:,} / {total_rows:,} rows")

        if writer is not None:
            writer.close()
            writer = None

        promote_temp_file(tmp_output_path, output_path)
        log.info(f"\nSaved to {output_path}")

    except Exception:
        if writer is not None:
            writer.close()
        cleanup_temp_file(tmp_output_path)
        raise

    # --- Summary logging ---
    log.info(f"  Aggregator postings: {total_agg:,} ({total_agg/max(total_rows,1):.1%})")
    for src, (a, t) in sorted(agg_by_source.items()):
        log.info(f"    {src}: {a:,} / {t:,} ({a/max(t,1):.1%})")

    log.info("\n  Top aggregator companies:")
    for company, count in top_agg_companies.most_common(20):
        log.info(f"    {company}: {count:,}")

    log.info(f"\n  Real employer extracted: {total_extracted:,} / {total_agg:,} ({total_extracted/max(total_agg,1):.1%})")
    log.info(f"  company_name_effective uses extracted real employer for {total_extracted:,} rows")

    log.info("\n  Sample real employer extractions:")
    for company, employer, title in sample_extractions:
        log.info(f"    {company} -> {employer} ({title[:50]})")

    log.info(f"\n  DataAnnotation postings flagged: {da_count:,}")

    return total_rows, total_agg


if __name__ == "__main__":
    total, agg = run_stage2()
    print(f"\nDone: {total:,} rows, {agg:,} aggregators flagged")
