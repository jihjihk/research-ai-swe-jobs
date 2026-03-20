#!/usr/bin/env python3
"""
Harmonize scraped and Kaggle LinkedIn job postings into a unified schema
for cross-period analysis.

Usage:
    python harmonize.py                          # Harmonize all available data
    python harmonize.py --output data/unified.parquet

Produces a single file with consistent column names, types, and text format
that can be fed directly into the analysis notebook.
"""

import argparse
import hashlib
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

BASE_DIR = Path(__file__).parent.parent  # project root (one level up from scraper/)
DATA_DIR = BASE_DIR / "data"
SCRAPED_DIR = DATA_DIR / "scraped"

# ---------------------------------------------------------------------------
# Unified schema — these are the columns the analysis notebook expects
# ---------------------------------------------------------------------------
UNIFIED_COLUMNS = [
    "job_id",           # unique identifier
    "source",           # "scraped_linkedin_2026" | "scraped_indeed_2026" | "apify_2026"
    "scrape_date",      # date the data was collected (YYYY-MM-DD)
    "date_posted",      # date the job was posted (YYYY-MM-DD, may be null)
    "title",            # raw job title
    "company_name",     # company name
    "location",         # location string
    "seniority",        # normalized: entry level | associate | mid-senior level | director | executive | internship
    "work_type",        # full-time | part-time | contract | internship
    "is_remote",        # bool
    "description",      # plain text (markdown stripped)
    "description_raw",  # original description (markdown or HTML preserved)
    "skills_raw",       # raw skills field (if available)
    "min_salary",       # annual salary (float, normalized to annual)
    "max_salary",       # annual salary (float, normalized to annual)
    "salary_currency",  # USD etc.
    "company_industry", # industry (if available)
    "company_size",     # employee count (float)
    "job_url",          # link to posting
    "job_group",        # "swe" | "control" | "adjacent" | "other"
    "is_swe",           # bool: job_group == "swe"
    "is_control",       # bool: job_group == "control"
]

# ---------------------------------------------------------------------------
# Classification patterns — must stay aligned with scrape_linkedin_swe.py
# ---------------------------------------------------------------------------

# SWE: matches the scraper's SWE_PATTERN (broader than original, includes AI roles)
SWE_PATTERN = re.compile(
    r'(?i)\b(software\s*(engineer|developer|dev(elopment)?)|swe|full[- ]?stack|front[- ]?end|'
    r'back[- ]?end|web\s*developer|mobile\s*developer|devops|platform\s*engineer|'
    r'data\s*engineer|ml\s*engineer|machine\s*learning\s*engineer|site\s*reliability|'
    r'ai\s*engineer|ai[/ ]ml\s*engineer|llm\s*engineer|agent\s*engineer|'
    r'applied\s*ai\s*engineer|prompt\s*engineer|infrastructure\s*engineer|'
    r'founding\s*engineer|member\s*of\s*technical\s*staff|product\s*engineer|'
    r'cloud\s*engineer)\b'
)

# Control: non-AI-exposed occupations for DiD (matches scraper's control tier)
CONTROL_PATTERN = re.compile(
    r'(?i)\b(civil\s*engineer|mechanical\s*engineer|electrical\s*engineer|'
    r'chemical\s*engineer|registered\s*nurse|accountant|'
    r'financial\s*analyst|marketing\s*manager|'
    r'human\s*resources|sales\s*representative|nurse\s*practitioner|nursing)\b'
)

# Adjacent: AI-exposed tech roles that are NOT SWE (comparison group, not control)
ADJACENT_PATTERN = re.compile(
    r'(?i)\b(data\s*scientist|data\s*analyst|product\s*manager|'
    r'ux\s*designer|qa\s*engineer|quality\s*assurance|security\s*engineer|'
    r'solutions\s*engineer|technical\s*program\s*manager|scrum\s*master|'
    r'applied\s*scientist)\b'
)


def classify_title(title) -> str:
    """Classify a job title into swe/control/adjacent/other."""
    if not isinstance(title, str):
        return "other"
    if SWE_PATTERN.search(title):
        return "swe"
    if CONTROL_PATTERN.search(title):
        return "control"
    if ADJACENT_PATTERN.search(title):
        return "adjacent"
    return "other"


def strip_markdown(text: str) -> str:
    """Remove markdown formatting to get plain text."""
    if not isinstance(text, str):
        return ""
    # Remove markdown bold/italic
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    # Remove markdown links [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove markdown headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove markdown escape chars
    text = re.sub(r'\\([^\s])', r'\1', text)
    # Remove bullet markers
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def strip_html(text: str) -> str:
    """Remove HTML tags to get plain text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def normalize_seniority(val) -> str:
    """Map various seniority strings to a canonical set."""
    if not isinstance(val, str):
        return ""
    val = val.strip().lower()
    mapping = {
        "entry level": "entry level",
        "entry_level": "entry level",
        "junior": "entry level",
        "associate": "associate",
        "mid-senior level": "mid-senior level",
        "mid senior level": "mid-senior level",
        "senior": "mid-senior level",
        "director": "director",
        "executive": "executive",
        "internship": "internship",
        "not applicable": "",
    }
    return mapping.get(val, val)


def normalize_salary_to_annual(amount, pay_period) -> float:
    """Convert salary to annual equivalent."""
    if pd.isna(amount) or amount is None:
        return float("nan")
    amount = float(amount)
    if not isinstance(pay_period, str):
        return amount
    period = pay_period.strip().upper()
    multipliers = {
        "HOURLY": 2080,
        "DAILY": 260,
        "WEEKLY": 52,
        "BIWEEKLY": 26,
        "MONTHLY": 12,
        "YEARLY": 1,
        "ANNUAL": 1,
    }
    return amount * multipliers.get(period, 1)


def canonicalize_job_url(raw_url) -> str:
    if not isinstance(raw_url, str):
        return ""
    raw_url = raw_url.strip()
    if not raw_url:
        return ""
    try:
        parts = urlsplit(raw_url)
    except ValueError:
        return raw_url
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, "", ""))


def normalize_text_fragment(value, *, limit: int | None = None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    if limit is not None:
        text = text[:limit]
    return text


def make_snapshot_dedup_key(row) -> str:
    scrape_date = normalize_text_fragment(row.get("scrape_date"))
    job_url = canonicalize_job_url(row.get("job_url"))
    if job_url:
        return f"{scrape_date}|{job_url}"

    job_id = normalize_text_fragment(row.get("job_id"))
    if job_id:
        return f"{scrape_date}|{job_id}"

    parts = [
        normalize_text_fragment(row.get("source")),
        normalize_text_fragment(row.get("title")),
        normalize_text_fragment(row.get("company_name")),
        normalize_text_fragment(row.get("location")),
        normalize_text_fragment(row.get("date_posted")),
        normalize_text_fragment(row.get("description_raw"), limit=500),
    ]
    digest = hashlib.md5("|".join(parts).encode()).hexdigest()
    return f"{scrape_date}|{digest}"


def make_posting_dedup_key(row) -> str:
    job_url = canonicalize_job_url(row.get("job_url"))
    if job_url:
        return job_url

    job_id = normalize_text_fragment(row.get("job_id"))
    if job_id:
        return f"{normalize_text_fragment(row.get('source'))}|{job_id}"

    parts = [
        normalize_text_fragment(row.get("source")),
        normalize_text_fragment(row.get("title")),
        normalize_text_fragment(row.get("company_name")),
        normalize_text_fragment(row.get("location")),
        normalize_text_fragment(row.get("date_posted")),
        normalize_text_fragment(row.get("description_raw"), limit=500),
    ]
    digest = hashlib.md5("|".join(parts).encode()).hexdigest()
    return digest




def _harmonize_one_scraped_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Transform a single raw scraped DataFrame into the unified schema."""
    out = pd.DataFrame()
    out["job_id"] = df["id"].astype(str)
    if "site" in df.columns:
        out["source"] = df["site"].apply(lambda s: f"scraped_{s}_2026" if pd.notna(s) else "scraped_2026")
    else:
        out["source"] = "scraped_2026"
    out["scrape_date"] = df.get("scrape_date", datetime.now().strftime("%Y-%m-%d"))
    out["date_posted"] = pd.to_datetime(df.get("date_posted"), errors="coerce").dt.strftime("%Y-%m-%d")
    out["title"] = df["title"]
    out["company_name"] = df.get("company", df.get("company_name"))
    out["location"] = df["location"]
    out["seniority"] = df.get("job_level", df.get("seniority", "")).apply(normalize_seniority)
    out["work_type"] = df.get("job_type")
    out["is_remote"] = df.get("is_remote", False).fillna(False).astype(bool)
    out["description"] = df["description"].apply(strip_markdown)
    out["description_raw"] = df["description"]
    out["skills_raw"] = df.get("skills")
    out["min_salary"] = pd.to_numeric(df.get("min_amount"), errors="coerce")
    out["max_salary"] = pd.to_numeric(df.get("max_amount"), errors="coerce")
    out["salary_currency"] = df.get("currency")
    out["company_industry"] = df.get("company_industry")
    out["company_size"] = pd.to_numeric(df.get("company_num_employees"), errors="coerce")
    out["job_url"] = df.get("job_url")
    out["job_group"] = df["title"].apply(classify_title)
    out["is_swe"] = out["job_group"] == "swe"
    out["is_control"] = out["job_group"] == "control"
    return out


def harmonize_scraped(
    scraped_dir: str,
    output_path: Path | None = None,
    observations_output_path: Path | None = None,
) -> pd.DataFrame | None:
    """Load and harmonize all daily scraped CSVs, one file at a time to limit memory.

    `output_path` is the canonical postings table: one row per unique posting globally.
    `observations_output_path` is the daily panel: one row per posting per scrape_date.

    If either output path is given, writes parquet directly via pyarrow and returns None
    (streaming mode — avoids holding all data in memory). Otherwise collects and returns
    the canonical postings table (legacy mode for small datasets).
    """
    scraped_path = Path(scraped_dir)
    # Use explicit prefix pattern to avoid *_swe_jobs.csv also matching *_non_swe_jobs.csv
    swe_files = [f for f in sorted(scraped_path.glob("*_swe_jobs.csv")) if "_non_swe_jobs.csv" not in f.name]
    non_swe_files = sorted(scraped_path.glob("*_non_swe_jobs.csv"))
    csv_files = swe_files + non_swe_files

    if not csv_files:
        print(f"  No scraped CSVs found in {scraped_dir}")
        if output_path or observations_output_path:
            return None
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    print(f"Loading {len(csv_files)} scraped CSV(s) from {scraped_dir}...")

    import gc
    import pyarrow as pa
    import pyarrow.parquet as pq

    seen_snapshot_keys: set[str] = set()
    seen_posting_keys: set[str] = set()
    total_raw = 0
    total_observations = 0
    total_postings = 0
    postings_writer = None
    observations_writer = None

    try:
        for f in csv_files:
            try:
                raw = pd.read_csv(f, low_memory=False)
            except Exception as e:
                print(f"  WARN: Could not read {f.name}: {e}")
                continue

            total_raw += len(raw)
            harmonized = _harmonize_one_scraped_csv(raw)
            del raw
            gc.collect()

            # Preserve one observation per job per scrape date.
            snapshot_keys = harmonized.apply(make_snapshot_dedup_key, axis=1)
            observation_mask = ~snapshot_keys.isin(seen_snapshot_keys)
            observations = harmonized[observation_mask].copy()
            seen_snapshot_keys.update(snapshot_keys[observation_mask].tolist())
            observation_count = len(observations)
            total_observations += observation_count

            posting_count = 0
            postings = pd.DataFrame(columns=UNIFIED_COLUMNS)
            if observation_count > 0:
                posting_keys = observations.apply(make_posting_dedup_key, axis=1)
                posting_mask = ~posting_keys.isin(seen_posting_keys)
                postings = observations[posting_mask].copy()
                seen_posting_keys.update(posting_keys[posting_mask].tolist())
                posting_count = len(postings)
                total_postings += posting_count

            print(
                f"  {f.name}: +{observation_count:,} observations, "
                f"+{posting_count:,} canonical postings"
            )

            if observations_output_path and observation_count > 0:
                table = pa.Table.from_pandas(observations, preserve_index=False)
                if observations_writer is None:
                    observations_writer = pq.ParquetWriter(
                        str(observations_output_path), table.schema
                    )
                observations_writer.write_table(table)
                del table

            if output_path and posting_count > 0:
                table = pa.Table.from_pandas(postings, preserve_index=False)
                if postings_writer is None:
                    postings_writer = pq.ParquetWriter(str(output_path), table.schema)
                postings_writer.write_table(table)
                del table
            elif not output_path and not observations_output_path:
                # Legacy mode — should only be used for small datasets
                if not hasattr(harmonize_scraped, "_parts"):
                    harmonize_scraped._parts = []
                harmonize_scraped._parts.append(postings)

            del postings
            del observations
            del harmonized
            gc.collect()
    finally:
        if postings_writer:
            postings_writer.close()
        if observations_writer:
            observations_writer.close()

    print(
        f"  Raw rows: {total_raw:,} -> {total_observations:,} observations "
        f"-> {total_postings:,} canonical postings"
    )

    if output_path:
        print(f"  Streamed {total_postings:,} canonical rows to {output_path}")
    if observations_output_path:
        print(
            f"  Streamed {total_observations:,} observation rows "
            f"to {observations_output_path}"
        )
    if output_path or observations_output_path:
        return None

    # Legacy mode
    parts = getattr(harmonize_scraped, "_parts", [])
    harmonize_scraped._parts = []
    if not parts:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    print(f"  Harmonized: {len(out):,} rows")
    return out


def harmonize_apify(path: str) -> pd.DataFrame:
    """Load and harmonize an Apify LinkedIn export CSV."""
    print(f"Loading Apify data from {path}...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Raw rows: {len(df):,}")

    out = pd.DataFrame()
    out["job_id"] = df.get("id", df.index).astype(str)
    out["source"] = "apify_2026"
    out["scrape_date"] = df.get("postedAt", pd.NaT)
    out["date_posted"] = pd.to_datetime(df.get("postedAt"), errors="coerce").dt.strftime("%Y-%m-%d")
    out["title"] = df["title"]
    out["company_name"] = df.get("companyName", df.get("company_name"))
    out["location"] = df["location"]
    out["seniority"] = df.get("seniorityLevel", "").apply(normalize_seniority)
    out["work_type"] = df.get("employmentType")
    out["is_remote"] = df["location"].str.contains(r'(?i)remote', na=False)
    # Apify has HTML descriptions
    out["description"] = df.get("descriptionText", df.get("descriptionHtml", "").apply(strip_html))
    out["description_raw"] = df.get("descriptionHtml", df.get("descriptionText"))
    out["skills_raw"] = None
    out["min_salary"] = None
    out["max_salary"] = None
    out["salary_currency"] = None
    out["company_industry"] = df.get("industries")
    out["company_size"] = pd.to_numeric(df.get("companyEmployeesCount"), errors="coerce")
    out["job_url"] = df.get("link")
    out["job_group"] = df["title"].apply(classify_title)
    out["is_swe"] = out["job_group"] == "swe"
    out["is_control"] = out["job_group"] == "control"

    print(f"  Harmonized: {len(out):,} rows")
    return out


def run(args):
    output = Path(args.output)
    observations_output = Path(args.observations_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    observations_output.parent.mkdir(parents=True, exist_ok=True)

    if not SCRAPED_DIR.exists():
        print("No data found to harmonize.")
        return

    # Stream directly to parquet to avoid OOM on small instances
    harmonize_scraped(
        str(SCRAPED_DIR),
        output_path=output,
        observations_output_path=observations_output,
    )

    # Read back canonical output for quality report.
    if output.exists():
        unified = pd.read_parquet(output)
        print(f"Saved to {output}")
        if observations_output.exists():
            print(f"Saved daily observations to {observations_output}")
        _print_quality_report(unified)


def _print_quality_report(unified: pd.DataFrame):
    """Print data quality summary."""
    print("\n" + "=" * 60)
    print("QUALITY REPORT")
    print("=" * 60)

    print("\n--- JOB GROUP BREAKDOWN ---")
    for group, count in unified["job_group"].value_counts().items():
        print(f"  {group:15s} {count:>7,} ({count/len(unified):6.1%})")

    for src in unified["source"].unique():
        sub = unified[unified["source"] == src]
        swe = sub[sub["is_swe"]]
        ctrl = sub[sub["is_control"]]
        adj = sub[sub["job_group"] == "adjacent"]
        other = sub[sub["job_group"] == "other"]
        print(f"\n--- {src} (n={len(sub):,}, SWE={len(swe):,}, control={len(ctrl):,}, adjacent={len(adj):,}, other={len(other):,}) ---")
        for col in ["title", "description", "seniority", "date_posted",
                     "company_name", "location", "min_salary", "is_remote",
                     "company_industry", "company_size", "skills_raw"]:
            if col == "is_remote":
                rate = sub[col].sum() / len(sub)
                print(f"  {col:25s} {rate:6.1%} are remote")
            else:
                rate = sub[col].notna().mean() if col in sub.columns else 0
                non_empty = (sub[col].astype(str).str.strip().str.len() > 0).mean() if col in sub.columns else 0
                print(f"  {col:25s} notna={rate:6.1%}  non-empty={non_empty:6.1%}")

        if len(swe) > 0:
            print(f"  SWE seniority dist:")
            for level, count in swe["seniority"].value_counts().head(6).items():
                print(f"    {level or '(empty)':25s} {count:5d} ({count/len(swe):.1%})")

        if len(ctrl) > 0:
            print(f"  Control title dist (top 8):")
            for title, count in ctrl["title"].value_counts().head(8).items():
                print(f"    {title[:40]:40s} {count:5d}")

    swe_all = unified[unified["is_swe"]]
    if swe_all["source"].nunique() > 1:
        print(f"\n--- CROSS-PERIOD SWE COMPARISON ---")
        print(f"  Description word count (median):")
        for src in swe_all["source"].unique():
            sub = swe_all[swe_all["source"] == src]
            wc = sub["description"].str.split().str.len().median()
            print(f"    {src:20s} {wc:.0f} words")


def main():
    parser = argparse.ArgumentParser(description="Harmonize LinkedIn job posting datasets")
    parser.add_argument("--output", default="data/unified.parquet",
                        help="Canonical postings output path (default: data/unified.parquet)")
    parser.add_argument(
        "--observations-output",
        default="data/unified_observations.parquet",
        help="Daily observations output path (default: data/unified_observations.parquet)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
