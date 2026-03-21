#!/usr/bin/env python3
"""
Stages 6b-6d, 7, 8: Location normalization, date validation,
language detection, temporal alignment, and quality flags.

All operations are per-row with no cross-row dependencies, so chunked
streaming works perfectly.

Input:  intermediate/stage5_classification.parquet  (~1.2M rows)
Output: intermediate/stage8_final.parquet

Memory-safe: iter_batches(batch_size=200_000), incremental ParquetWriter,
gc.collect() after each chunk.  All new numeric columns forced to float64.
"""

import gc
import hashlib
import logging
import re
import string
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from langdetect import DetectorFactory, LangDetectException, detect

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"

INPUT_PATH = INTERMEDIATE_DIR / "stage5_classification.parquet"
OUTPUT_PATH = INTERMEDIATE_DIR / "stage8_final.parquet"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "stage678.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

CHUNK_SIZE = 200_000

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# US state abbreviations (for location normalization)
US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
}

# Full state name -> abbreviation mapping
STATE_NAME_TO_ABBR = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC",
}

# Date ranges per (source, platform) for stage 6d
DATE_RANGES = {
    ("kaggle_arshkon", "linkedin"):   (pd.Timestamp("2024-04-01"), pd.Timestamp("2024-05-01")),
    ("kaggle_asaniczka", "linkedin"): (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")),
    ("scraped", "linkedin"):          (pd.Timestamp("2026-03-20"), pd.Timestamp("2026-12-31")),
    ("scraped", "indeed"):            (pd.Timestamp("2026-03-20"), pd.Timestamp("2026-12-31")),
}

# Period mapping for historical sources; scraped is derived from scrape_date
SOURCE_PERIOD = {
    "kaggle_asaniczka": "2024-01",
    "kaggle_arshkon":   "2024-04",
}

# Remote indicators in location strings
REMOTE_RE = re.compile(r"\b(remote|anywhere|work\s*from\s*home|wfh)\b", re.IGNORECASE)

# Pattern: "City, ST" where ST is a 2-letter US state abbreviation
CITY_STATE_RE = re.compile(r"^(.+?),\s*([A-Z]{2})$")

# Pattern: "City, State Name, Country" or "City, State Name"
CITY_STATENAME_RE = re.compile(r"^(.+?),\s*(.+?)(?:,\s*(.+))?$")

# Ghost job: experience-years patterns in description
YEARS_EXP_RE = re.compile(
    r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp\b)",
    re.IGNORECASE,
)

DetectorFactory.seed = 0
ASCII_PRINTABLE = set(string.printable)


# ---------------------------------------------------------------------------
# Stage 6b: Location normalization
# ---------------------------------------------------------------------------
def normalize_location(loc: str, is_remote_existing: bool):
    """Parse location string into (city, state_normalized, country, is_remote_updated).

    Returns a dict with the extracted fields.
    """
    result = {
        "city_extracted": None,
        "state_normalized": None,
        "country_extracted": None,
        "is_remote_location": False,  # True if location string indicates remote
    }

    if pd.isna(loc) or not isinstance(loc, str) or loc.strip() == "":
        return result

    loc = loc.strip()

    # Check for remote indicators in the location string
    if REMOTE_RE.search(loc):
        result["is_remote_location"] = True

    # Try "City, ST" pattern (most common US format)
    m = CITY_STATE_RE.match(loc)
    if m:
        city_part, state_part = m.group(1).strip(), m.group(2).strip()
        if state_part in US_STATES:
            result["city_extracted"] = city_part
            result["state_normalized"] = state_part
            result["country_extracted"] = "US"
            return result

    # Try "City, StateName/Region, Country" or "City, StateName"
    m = CITY_STATENAME_RE.match(loc)
    if m:
        city_part = m.group(1).strip()
        middle_part = m.group(2).strip()
        country_part = m.group(3).strip() if m.group(3) else None

        # Check if middle part is a US state name
        state_abbr = STATE_NAME_TO_ABBR.get(middle_part.lower())
        if state_abbr:
            result["city_extracted"] = city_part
            result["state_normalized"] = state_abbr
            result["country_extracted"] = "US"
            return result

        # Check if middle part is a 2-letter state abbreviation (with country after)
        if len(middle_part) == 2 and middle_part.upper() in US_STATES:
            result["city_extracted"] = city_part
            result["state_normalized"] = middle_part.upper()
            result["country_extracted"] = country_part if country_part else "US"
            return result

        # Non-US pattern: "City, Region, Country"
        result["city_extracted"] = city_part
        if country_part:
            result["country_extracted"] = country_part
        else:
            # "State, Country" pattern like "New Jersey, United States"
            if middle_part == "United States":
                result["country_extracted"] = "US"
                # city_part might be a state name
                state_abbr = STATE_NAME_TO_ABBR.get(city_part.lower())
                if state_abbr:
                    result["state_normalized"] = state_abbr
                    result["city_extracted"] = None
            else:
                result["country_extracted"] = middle_part

        return result

    # Single value (no comma): "United States", "Metro Area", country name, etc.
    if loc == "United States":
        result["country_extracted"] = "US"
    elif loc.lower() in STATE_NAME_TO_ABBR:
        result["state_normalized"] = STATE_NAME_TO_ABBR[loc.lower()]
        result["country_extracted"] = "US"
    else:
        # Could be a metro area, country name, etc. - leave as-is
        result["city_extracted"] = loc

    return result


# ---------------------------------------------------------------------------
# Stage 6c: Date validation
# ---------------------------------------------------------------------------
def validate_dates(row_source, row_platform, scrape_date_str, date_posted_str):
    """Return a date_flag string. 'ok' if valid, else describes the issue."""
    flags = []

    date_range = DATE_RANGES.get((row_source, row_platform))
    if date_range is None:
        return "unknown_source"

    lo, hi = date_range

    for col_name, date_str in [("scrape_date", scrape_date_str),
                                ("date_posted", date_posted_str)]:
        if pd.isna(date_str) or not isinstance(date_str, str) or date_str.strip() == "":
            # Missing dates are not flagged (common for date_posted)
            continue
        try:
            dt = pd.Timestamp(date_str)
            if dt < lo or dt > hi:
                flags.append(f"{col_name}_out_of_range")
        except (ValueError, pd.errors.OutOfBoundsDatetime):
            flags.append(f"{col_name}_invalid")

    return "|".join(flags) if flags else "ok"


# ---------------------------------------------------------------------------
# Stage 6d: Language detection
# ---------------------------------------------------------------------------
def detect_language(text):
    """Return True/False for English/non-English, or pd.NA if unknown."""
    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
        return pd.NA

    text = text.strip()
    if len(text) == 0:
        return pd.NA

    # Fast pre-check: extremely low ASCII share is almost never English.
    ascii_count = sum(1 for c in text if c in ASCII_PRINTABLE)
    ratio = ascii_count / len(text)
    if ratio < 0.60:
        return False

    try:
        return detect(text[:5000]) == "en"
    except LangDetectException:
        return pd.NA


# ---------------------------------------------------------------------------
# Stage 8: Ghost job detection
# ---------------------------------------------------------------------------
def detect_ghost_job(seniority_3level: str, yoe_extracted, yoe_contradiction: bool) -> str:
    """Detect ghost jobs: entry-level title with high experience requirement.

    Returns 'high', 'medium', or 'low'.
    """
    if seniority_3level != "junior":
        return "low"

    if pd.isna(yoe_extracted):
        return "low"

    if yoe_extracted >= 5:
        return "high"
    if yoe_contradiction or yoe_extracted >= 3:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Stage 8: Description quality
# ---------------------------------------------------------------------------
def assess_description_quality(description_core) -> str:
    """Return quality flag for description_core."""
    if pd.isna(description_core) or not isinstance(description_core, str):
        return "empty"

    text = description_core.strip()
    if len(text) == 0:
        return "empty"
    if len(text) < 50:
        return "too_short"

    return "ok"


def build_description_hash(text):
    """Stable sha256 hash of the full description used for LLM caching."""
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def derive_period(row_source, scrape_date):
    """Map rows into analysis periods."""
    if row_source == "scraped":
        ts = pd.to_datetime(scrape_date, errors="coerce")
        if pd.isna(ts):
            return "unknown"
        return ts.strftime("%Y-%m")

    return SOURCE_PERIOD.get(row_source, "unknown")


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
def process_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all stage 6b-6d, 7, 8 transformations to a chunk."""

    # ---- Stage 6b: Location normalization ----
    loc_results = df.apply(
        lambda row: normalize_location(row["location"], row["is_remote"]),
        axis=1,
        result_type="expand",
    )
    df["city_extracted"] = loc_results["city_extracted"]
    df["state_normalized"] = loc_results["state_normalized"]
    df["country_extracted"] = loc_results["country_extracted"]
    # Update is_remote: True if already True OR location string indicates remote
    df["is_remote"] = df["is_remote"].fillna(False) | loc_results["is_remote_location"].fillna(False)

    # ---- Stage 6c: Date validation ----
    df["date_flag"] = df.apply(
        lambda row: validate_dates(
            row["source"], row["source_platform"], row["scrape_date"], row["date_posted"]
        ),
        axis=1,
    )

    # ---- Stage 6d: Language detection ----
    df["is_english"] = df["description_core"].apply(detect_language).astype("boolean")
    df["description_hash"] = df["description"].apply(build_description_hash)

    # ---- Stage 7: Temporal alignment ----
    df["period"] = df.apply(lambda row: derive_period(row["source"], row["scrape_date"]), axis=1)
    df["posting_age_days"] = (
        pd.to_datetime(df["scrape_date"], errors="coerce")
        - pd.to_datetime(df["date_posted"], errors="coerce")
    ).dt.days.astype("float64")

    # scrape_week: ISO week from scrape_date
    scrape_ts = pd.to_datetime(df["scrape_date"], errors="coerce")
    df["scrape_week"] = scrape_ts.dt.isocalendar().week.astype("float64")

    # ---- Stage 8: Quality flags ----
    df["ghost_job_risk"] = df.apply(
        lambda row: detect_ghost_job(
            row["seniority_3level"], row["yoe_extracted"], bool(row["yoe_seniority_contradiction"])
        ),
        axis=1,
    )

    df["description_quality_flag"] = df["description_core"].apply(
        assess_description_quality
    )

    # Data provenance
    df["preprocessing_version"] = "3.0"
    df["dedup_method"] = "stage4_title_company_deduplicated"
    df["boilerplate_removed"] = True

    return df


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("Stages 6b-6d, 7, 8: Normalize, temporal align, quality flags")
    log.info("=" * 70)
    log.info(f"Input:  {INPUT_PATH}")
    log.info(f"Output: {OUTPUT_PATH}")

    pf = pq.ParquetFile(INPUT_PATH)
    total_rows = pf.metadata.num_rows
    log.info(f"Input rows: {total_rows:,}")

    writer = None
    rows_written = 0
    chunk_num = 0

    # Counters for summary stats
    stats = {
        "remote_updated": 0,
        "state_extracted": 0,
        "date_flagged": 0,
        "non_english": 0,
        "ghost_high": 0,
        "ghost_medium": 0,
        "desc_empty": 0,
        "desc_too_short": 0,
    }

    for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
        chunk_num += 1
        t_chunk = time.time()

        # Convert to pandas
        table = pa.Table.from_batches([batch])
        df = table.to_pandas()
        n = len(df)

        log.info(f"Chunk {chunk_num}: {n:,} rows")

        # Track is_remote before update for counting
        remote_before = df["is_remote"].sum()

        # Apply all transformations
        df = process_chunk(df)

        # Collect stats
        stats["remote_updated"] += int(df["is_remote"].sum() - remote_before)
        stats["state_extracted"] += int(df["state_normalized"].notna().sum())
        stats["date_flagged"] += int((df["date_flag"] != "ok").sum())
        stats["non_english"] += int((df["is_english"] == False).sum())
        stats["ghost_high"] += int((df["ghost_job_risk"] == "high").sum())
        stats["ghost_medium"] += int((df["ghost_job_risk"] == "medium").sum())
        stats["desc_empty"] += int((df["description_quality_flag"] == "empty").sum())
        stats["desc_too_short"] += int(
            (df["description_quality_flag"] == "too_short").sum()
        )

        # Convert back to Arrow and write
        out_table = pa.Table.from_pandas(df, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(OUTPUT_PATH, out_table.schema)

        writer.write_table(out_table)
        rows_written += n

        elapsed = time.time() - t_chunk
        log.info(
            f"  Chunk {chunk_num} done in {elapsed:.1f}s "
            f"({rows_written:,}/{total_rows:,} rows written)"
        )

        # Memory cleanup
        del df, table, out_table, batch
        gc.collect()

    if writer is not None:
        writer.close()

    elapsed_total = time.time() - t0
    log.info("=" * 70)
    log.info(f"COMPLETE: {rows_written:,} rows written in {elapsed_total:.1f}s")
    log.info("=" * 70)

    # Summary stats
    log.info("--- Summary ---")
    log.info(f"  Remote postings updated from location string: {stats['remote_updated']:,}")
    log.info(f"  US state extracted: {stats['state_extracted']:,}")
    log.info(f"  Date flags (non-ok): {stats['date_flagged']:,}")
    log.info(f"  Non-English descriptions: {stats['non_english']:,}")
    log.info(f"  Ghost job risk HIGH: {stats['ghost_high']:,}")
    log.info(f"  Ghost job risk MEDIUM: {stats['ghost_medium']:,}")
    log.info(f"  Description empty: {stats['desc_empty']:,}")
    log.info(f"  Description too short: {stats['desc_too_short']:,}")

    # Verify output
    log.info("--- Output verification ---")
    out_pf = pq.ParquetFile(OUTPUT_PATH)
    log.info(f"  Output rows: {out_pf.metadata.num_rows:,}")
    log.info(f"  Output columns: {out_pf.metadata.num_columns}")
    schema = out_pf.schema_arrow
    log.info("  Column list:")
    for i in range(len(schema)):
        log.info(f"    {schema.field(i).name}: {schema.field(i).type}")


if __name__ == "__main__":
    main()
