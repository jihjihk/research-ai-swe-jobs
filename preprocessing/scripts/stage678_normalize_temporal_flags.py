#!/usr/bin/env python3
"""
Stages 6-8: field normalization, temporal alignment, and quality flags.

Stage 6 owns row-level location parsing, remote inference, and metro enrichment.
Stage 7 owns temporal derivations (`period`, `posting_age_days`, `scrape_week`).
Stage 8 owns quality flags and utility fields (`date_flag`, `is_english`,
`description_hash`, `ghost_job_risk`, `description_quality_flag`) plus
provenance metadata.

All operations are per-row with no cross-row dependencies, so chunked
streaming works perfectly.

Input:  intermediate/stage5_classification.parquet  (~1.2M rows)
Output: intermediate/stage8_final.parquet

Memory-safe: iter_batches(batch_size=200_000), incremental ParquetWriter,
gc.collect() after each chunk.  All new numeric columns forced to float64.
"""

import gc
import hashlib
import json
import logging
import re
import string
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from langdetect import DetectorFactory, LangDetectException, detect

from io_utils import cleanup_temp_file, prepare_temp_output, promote_temp_file, promote_null_schema

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"
REFERENCE_DIR = PROJECT_ROOT / "preprocessing" / "reference"

INPUT_PATH = INTERMEDIATE_DIR / "stage5_classification.parquet"
OUTPUT_PATH = INTERMEDIATE_DIR / "stage8_final.parquet"
METRO_ALIAS_PATH = REFERENCE_DIR / "metro_aliases.json"
# Optional offline city/state reference built by:
#   ./.venv/bin/python preprocessing/scripts/build_metro_city_state_reference.py
# Stage 6 uses it when present and falls back to scraped search-metro evidence
# plus manual aliases when absent.
METRO_CITY_STATE_REFERENCE_PATH = REFERENCE_DIR / "metro_city_state_lookup.parquet"

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

# Date ranges per (source, platform) for quality flagging.
# Current-format scraped files should not have a hard-coded upper bound.
DATE_RANGES = {
    ("kaggle_arshkon", "linkedin"):   (pd.Timestamp("2024-04-01"), pd.Timestamp("2024-05-01")),
    ("kaggle_asaniczka", "linkedin"): (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")),
    ("scraped", "linkedin"):          (pd.Timestamp("2026-03-20"), None),
    ("scraped", "indeed"):            (pd.Timestamp("2026-03-20"), None),
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

NON_METRO_LABEL_RE = re.compile(
    r"\b(remote|anywhere|united states|usa|us|hybrid|on[- ]?site|onsite)\b",
    re.IGNORECASE,
)

DetectorFactory.seed = 0
ASCII_PRINTABLE = set(string.printable)


def normalize_key(value: str | None) -> str:
    """Normalize a free-text label for deterministic lookup keys."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_metro_aliases() -> dict[str, str]:
    """Load manual metro alias mappings keyed by normalized alias string."""
    if not METRO_ALIAS_PATH.exists():
        return {}

    raw = json.loads(METRO_ALIAS_PATH.read_text())
    aliases: dict[str, str] = {}
    for canonical, alias_list in raw.items():
        canonical_clean = canonical.strip()
        if not canonical_clean:
            continue
        aliases[normalize_key(canonical_clean)] = canonical_clean
        for alias in alias_list:
            alias_key = normalize_key(alias)
            if alias_key:
                aliases[alias_key] = canonical_clean
    return aliases


def canonicalize_metro_label(
    label: str | None,
    metro_aliases: dict[str, str],
    *,
    allow_passthrough: bool = False,
) -> str | None:
    """Return the canonical study-metro label for a free-text metro-like string."""
    key = normalize_key(label)
    if not key:
        return None
    if NON_METRO_LABEL_RE.search(key):
        return None
    canonical = metro_aliases.get(key)
    if canonical is not None:
        return canonical
    if allow_passthrough and label and str(label).strip():
        return str(label).strip()
    return None


def metro_lookup_key(city: str | None, state: str | None) -> str | None:
    """Return the deterministic city/state key used for metro lookup."""
    city_key = normalize_key(city)
    state_key = normalize_key(state)
    if not city_key or not state_key:
        return None
    return f"{city_key}|{state_key.upper()}"


def build_scraped_city_state_metro_lookup(
    input_path: Path,
    metro_aliases: dict[str, str],
) -> tuple[dict[str, tuple[str, str]], dict[str, int]]:
    """Build a city/state -> metro lookup from controlled scraped metro metadata."""
    pf = pq.ParquetFile(input_path)
    metro_counts: dict[str, Counter] = defaultdict(Counter)
    stats = {
        "scraped_rows_seen": 0,
        "scraped_rows_with_search_metro": 0,
        "scraped_rows_with_city_state": 0,
        "lookup_keys": 0,
        "lookup_high": 0,
        "lookup_medium": 0,
    }

    for batch in pf.iter_batches(
        batch_size=CHUNK_SIZE,
        columns=["source", "location", "search_metro_name"],
    ):
        chunk = batch.to_pandas()
        chunk = chunk.loc[chunk["source"] == "scraped", ["location", "search_metro_name"]]
        stats["scraped_rows_seen"] += len(chunk)
        if len(chunk) == 0:
            continue

        for row in chunk.itertuples(index=False):
            canonical_metro = canonicalize_metro_label(
                row.search_metro_name,
                metro_aliases,
                allow_passthrough=True,
            )
            if canonical_metro is None:
                continue

            stats["scraped_rows_with_search_metro"] += 1
            loc = normalize_location(row.location)
            if loc["country_extracted"] not in {"US", None}:
                continue

            key = metro_lookup_key(loc["city_extracted"], loc["state_normalized"])
            if key is None:
                continue

            stats["scraped_rows_with_city_state"] += 1
            metro_counts[key][canonical_metro] += 1

    lookup: dict[str, tuple[str, str]] = {}
    for key, counts in metro_counts.items():
        total = sum(counts.values())
        if total < 2:
            continue
        metro, count = counts.most_common(1)[0]
        share = count / total
        if share >= 0.95 and total >= 3:
            lookup[key] = (metro, "high")
            stats["lookup_high"] += 1
        elif share >= 0.75:
            lookup[key] = (metro, "medium")
            stats["lookup_medium"] += 1

    stats["lookup_keys"] = len(lookup)
    return lookup, stats


def load_metro_city_state_reference_lookup(
    reference_path: Path,
) -> tuple[dict[str, tuple[str, str]], dict[str, int]]:
    """Load an offline city/state -> metro lookup built from cached geocoding."""
    if not reference_path.exists():
        return {}, {"reference_rows": 0, "reference_keys": 0}

    table = pq.read_table(
        reference_path,
        columns=["metro_lookup_key", "metro_area", "match_confidence", "cbsa_code"],
    )
    df = table.to_pandas()
    df = df.loc[
        df["metro_lookup_key"].notna()
        & df["metro_area"].notna()
        & df["cbsa_code"].notna()
    ].drop_duplicates(subset=["metro_lookup_key"], keep="first")

    lookup = {
        row.metro_lookup_key: (row.metro_area, row.match_confidence)
        for row in df.itertuples(index=False)
    }
    return lookup, {"reference_rows": len(df), "reference_keys": len(lookup)}


# ---------------------------------------------------------------------------
# Stage 6b: Location normalization
# ---------------------------------------------------------------------------
def normalize_location(loc: str):
    """Parse location string into city/state/country fields plus remote inference.

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
        if not REMOTE_RE.fullmatch(city_part):
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
        if not REMOTE_RE.fullmatch(loc):
            result["city_extracted"] = loc

    return result


# ---------------------------------------------------------------------------
# Stage 6c: Metro enrichment
# ---------------------------------------------------------------------------
def infer_metro(
    row_source,
    row_location,
    row_search_metro_name,
    row_is_remote,
    row_is_remote_inferred,
    city_extracted,
    state_normalized,
    metro_aliases: dict[str, str],
    city_state_lookup: dict[str, tuple[str, str]],
    city_state_reference_lookup: dict[str, tuple[str, str]],
):
    """Infer a study-frame metro label without changing row cardinality."""
    remote_flag = bool(row_is_remote) or bool(row_is_remote_inferred)

    if row_source == "scraped" and not remote_flag:
        search_metro = canonicalize_metro_label(
            row_search_metro_name,
            metro_aliases,
            allow_passthrough=True,
        )
        if search_metro is not None:
            return {
                "metro_area": search_metro,
                "metro_source": "search_metro",
                "metro_confidence": "high",
            }

    manual_metro = canonicalize_metro_label(row_location, metro_aliases)
    if manual_metro is not None and not remote_flag:
        return {
            "metro_area": manual_metro,
            "metro_source": "manual_alias",
            "metro_confidence": "high",
        }

    lookup_key = metro_lookup_key(city_extracted, state_normalized)
    if lookup_key is not None and not remote_flag:
        lookup_result = city_state_lookup.get(lookup_key)
        if lookup_result is not None:
            metro_area, confidence = lookup_result
            return {
                "metro_area": metro_area,
                "metro_source": "city_state_lookup",
                "metro_confidence": confidence,
            }

        reference_result = city_state_reference_lookup.get(lookup_key)
        if reference_result is not None:
            metro_area, confidence = reference_result
            return {
                "metro_area": metro_area,
                "metro_source": "city_state_reference",
                "metro_confidence": confidence,
            }

    return {
        "metro_area": None,
        "metro_source": "unresolved",
        "metro_confidence": "low",
    }


# ---------------------------------------------------------------------------
# Stage 8a: Date validation
# ---------------------------------------------------------------------------
MIN_PLAUSIBLE_DATE = pd.Timestamp("2020-01-01")


def validate_dates(row_source, row_platform, scrape_date_str, date_posted_str):
    """Return a date_flag string.

    Stage 8 date checks are a lightweight sanity screen, not an age heuristic.
    A date is flagged only if it cannot be parsed or is implausibly early.
    """
    del row_source, row_platform

    flags = []
    for col_name, date_str in [("scrape_date", scrape_date_str), ("date_posted", date_posted_str)]:
        if pd.isna(date_str) or not isinstance(date_str, str) or date_str.strip() == "":
            # Missing dates are not flagged (common for date_posted).
            continue
        try:
            dt = pd.Timestamp(date_str)
        except (ValueError, pd.errors.OutOfBoundsDatetime):
            flags.append(f"{col_name}_invalid")
            continue

        if dt < MIN_PLAUSIBLE_DATE:
            flags.append(f"{col_name}_out_of_range")

    return "|".join(flags) if flags else "ok"


# ---------------------------------------------------------------------------
# Stage 8b: Language detection
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
def is_junior_like_seniority(seniority_final: str) -> bool:
    """Return True only for the junior-like 5-level seniority bucket."""
    return str(seniority_final).strip().lower() == "entry"


def detect_ghost_job(seniority_final: str, yoe_extracted, yoe_contradiction: bool) -> str:
    """Detect ghost jobs: entry-level title with high experience requirement.

    Uses the canonical 5-level seniority label rather than the coarse 3-level
    bucket. Returns 'high', 'medium', or 'low'.
    """
    if not is_junior_like_seniority(seniority_final):
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
def assess_description_quality(description) -> str:
    """Return quality flag for the raw description."""
    if pd.isna(description) or not isinstance(description, str):
        return "empty"

    text = description.strip()
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
def process_chunk(
    df: pd.DataFrame,
    metro_aliases: dict[str, str],
    city_state_lookup: dict[str, tuple[str, str]],
    city_state_reference_lookup: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Apply all Stage 6-8 row-preserving transformations to a chunk."""

    # ---- Stage 6b: Location normalization ----
    loc_results = df.apply(
        lambda row: normalize_location(row["location"]),
        axis=1,
        result_type="expand",
    )
    df["city_extracted"] = loc_results["city_extracted"]
    df["state_normalized"] = loc_results["state_normalized"]
    df["country_extracted"] = loc_results["country_extracted"]
    # Preserve Stage 1's normalized source flag and store location-text inference separately.
    df["is_remote_inferred"] = loc_results["is_remote_location"].fillna(False).astype(bool)
    metro_results = df.apply(
        lambda row: infer_metro(
            row["source"],
            row["location"],
            row.get("search_metro_name"),
            row["is_remote"],
            row["is_remote_inferred"],
            row["city_extracted"],
            row["state_normalized"],
            metro_aliases,
            city_state_lookup,
            city_state_reference_lookup,
        ),
        axis=1,
        result_type="expand",
    )
    df["metro_area"] = metro_results["metro_area"]
    df["metro_source"] = metro_results["metro_source"]
    df["metro_confidence"] = metro_results["metro_confidence"]

    # ---- Stage 7: Temporal alignment ----
    df["period"] = df.apply(lambda row: derive_period(row["source"], row["scrape_date"]), axis=1)
    df["posting_age_days"] = (
        pd.to_datetime(df["scrape_date"], errors="coerce")
        - pd.to_datetime(df["date_posted"], errors="coerce")
    ).dt.days.astype("float64")

    # scrape_week: ISO week from scrape_date
    scrape_ts = pd.to_datetime(df["scrape_date"], errors="coerce")
    df["scrape_week"] = scrape_ts.dt.isocalendar().week.astype("float64")

    # ---- Stage 8: Quality flags and utility fields ----
    df["date_flag"] = df.apply(
        lambda row: validate_dates(
            row["source"], row["source_platform"], row["scrape_date"], row["date_posted"]
        ),
        axis=1,
    )
    df["is_english"] = df["description"].apply(detect_language).astype("boolean")
    df["description_hash"] = df["description"].apply(build_description_hash)
    df["ghost_job_risk"] = df.apply(
        lambda row: detect_ghost_job(
            row["seniority_final"], row["yoe_extracted"], bool(row["yoe_seniority_contradiction"])
        ),
        axis=1,
    )

    df["description_quality_flag"] = df["description"].apply(
        assess_description_quality
    )

    # Data provenance
    df["preprocessing_version"] = "3.0"
    df["dedup_method"] = "stage4_title_company_deduplicated"

    return df


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("Stages 6-8: field normalization, temporal alignment, quality flags")
    log.info("=" * 70)
    log.info(f"Input:  {INPUT_PATH}")
    log.info(f"Output: {OUTPUT_PATH}")

    pf = pq.ParquetFile(INPUT_PATH)
    total_rows = pf.metadata.num_rows
    log.info(f"Input rows: {total_rows:,}")

    metro_aliases = load_metro_aliases()
    city_state_lookup, metro_lookup_stats = build_scraped_city_state_metro_lookup(
        INPUT_PATH,
        metro_aliases,
    )
    city_state_reference_lookup, reference_lookup_stats = load_metro_city_state_reference_lookup(
        METRO_CITY_STATE_REFERENCE_PATH,
    )
    log.info("--- Stage 6 metro reference build ---")
    log.info(f"  Manual metro aliases loaded: {len(metro_aliases):,}")
    log.info(f"  Scraped rows seen: {metro_lookup_stats['scraped_rows_seen']:,}")
    log.info(
        "  Scraped rows with search metro and parsed city/state: "
        f"{metro_lookup_stats['scraped_rows_with_city_state']:,}"
    )
    log.info(f"  City/state metro lookup keys: {metro_lookup_stats['lookup_keys']:,}")
    log.info(f"    High-confidence keys: {metro_lookup_stats['lookup_high']:,}")
    log.info(f"    Medium-confidence keys: {metro_lookup_stats['lookup_medium']:,}")
    log.info(
        f"  Cached city/state reference keys: {reference_lookup_stats['reference_keys']:,}"
    )

    tmp_output_path = prepare_temp_output(OUTPUT_PATH)

    writer = None
    rows_written = 0
    chunk_num = 0

    # Counters for summary stats
    stats = {
        "remote_inferred": 0,
        "state_extracted": 0,
        "metro_resolved": 0,
        "metro_search": 0,
        "metro_lookup": 0,
        "metro_manual": 0,
        "metro_reference": 0,
        "date_flagged": 0,
        "non_english": 0,
        "ghost_high": 0,
        "ghost_medium": 0,
        "desc_empty": 0,
        "desc_too_short": 0,
    }

    try:
        for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
            chunk_num += 1
            t_chunk = time.time()

            # Convert to pandas
            table = pa.Table.from_batches([batch])
            df = table.to_pandas()
            n = len(df)

            log.info(f"Chunk {chunk_num}: {n:,} rows")

            # Apply all transformations
            df = process_chunk(
                df,
                metro_aliases,
                city_state_lookup,
                city_state_reference_lookup,
            )

            # Collect stats
            stats["remote_inferred"] += int(df["is_remote_inferred"].sum())
            stats["state_extracted"] += int(df["state_normalized"].notna().sum())
            stats["metro_resolved"] += int(df["metro_area"].notna().sum())
            stats["metro_search"] += int((df["metro_source"] == "search_metro").sum())
            stats["metro_lookup"] += int((df["metro_source"] == "city_state_lookup").sum())
            stats["metro_manual"] += int((df["metro_source"] == "manual_alias").sum())
            stats["metro_reference"] += int(
                (df["metro_source"] == "city_state_reference").sum()
            )
            stats["date_flagged"] += int((df["date_flag"] != "ok").sum())
            stats["non_english"] += int((df["is_english"] == False).sum())
            stats["ghost_high"] += int((df["ghost_job_risk"] == "high").sum())
            stats["ghost_medium"] += int((df["ghost_job_risk"] == "medium").sum())
            stats["desc_empty"] += int((df["description_quality_flag"] == "empty").sum())
            stats["desc_too_short"] += int(
                (df["description_quality_flag"] == "too_short").sum()
            )

            # Convert back to Arrow and write (cast to unified schema)
            out_table = pa.Table.from_pandas(df, preserve_index=False)

            if writer is None:
                output_schema = promote_null_schema(out_table.schema)
                writer = pq.ParquetWriter(tmp_output_path, output_schema)

            out_table = out_table.cast(output_schema)
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
    except Exception:
        if writer is not None:
            writer.close()
        cleanup_temp_file(tmp_output_path)
        raise

    if writer is not None:
        writer.close()

    elapsed_total = time.time() - t0
    log.info("=" * 70)
    log.info(f"COMPLETE: {rows_written:,} rows written in {elapsed_total:.1f}s")
    log.info("=" * 70)

    # Summary stats
    log.info("--- Summary ---")
    log.info(f"  Remote postings inferred from location string: {stats['remote_inferred']:,}")
    log.info(f"  US state extracted: {stats['state_extracted']:,}")
    log.info(f"  Metro area resolved: {stats['metro_resolved']:,}")
    log.info(f"    from search metro: {stats['metro_search']:,}")
    log.info(f"    from city/state lookup: {stats['metro_lookup']:,}")
    log.info(f"    from manual alias: {stats['metro_manual']:,}")
    log.info(f"    from cached city/state reference: {stats['metro_reference']:,}")
    log.info(f"  Date flags (non-ok): {stats['date_flagged']:,}")
    log.info(f"  Non-English descriptions: {stats['non_english']:,}")
    log.info(f"  Ghost job risk HIGH: {stats['ghost_high']:,}")
    log.info(f"  Ghost job risk MEDIUM: {stats['ghost_medium']:,}")
    log.info(f"  Description empty: {stats['desc_empty']:,}")
    log.info(f"  Description too short: {stats['desc_too_short']:,}")

    # Verify output
    log.info("--- Output verification ---")
    out_pf = pq.ParquetFile(tmp_output_path)
    log.info(f"  Output rows: {out_pf.metadata.num_rows:,}")
    log.info(f"  Output columns: {out_pf.metadata.num_columns}")
    schema = out_pf.schema_arrow
    log.info("  Column list:")
    for i in range(len(schema)):
        log.info(f"    {schema.field(i).name}: {schema.field(i).type}")

    promote_temp_file(tmp_output_path, OUTPUT_PATH)
    log.info(f"Promoted temp output to final path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
