#!/usr/bin/env python3
"""
Stage 1: Ingest & Schema Unification

Loads raw data from three sources:
  1. Kaggle arshkon (124K LinkedIn postings, April 2024)
  2. Kaggle asaniczka (1.3M LinkedIn postings, 2024)
  3. Scraped data (14 days of March 2026, LinkedIn + Indeed)

Produces a unified DataFrame with expanded schema including:
  - Expanded SWE_PATTERN (canonical, shared with scraper)
  - Kaggle companion file joins (industry, company size)
  - New computed columns (source_platform, description_length, etc.)

Output: intermediate/stage1_unified.parquet
"""

import re
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KAGGLE_DIR = DATA_DIR / "kaggle-linkedin-jobs-2023-2024"
ASANICZKA_DIR = DATA_DIR / "kaggle-asaniczka-1.3m"
SCRAPED_DIR = DATA_DIR / "scraped"
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"

INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "stage1_ingest.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical SWE pattern (expanded, matches scraper)
# ---------------------------------------------------------------------------
SWE_INCLUDE = re.compile(
    r'(?i)\b('
    r'software\s*(engineer|developer|dev|development\s*engineer)|'
    r'swe\b|full[- ]?stack|front[- ]?end\s*(engineer|developer)|'
    r'back[- ]?end\s*(engineer|developer)|'
    r'web\s*(developer|engineer)|mobile\s*(developer|engineer)|'
    r'devops\s*(engineer)?|platform\s*engineer|'
    r'data\s*engineer|ml\s*engineer|machine\s*learning\s*engineer|'
    r'site\s*reliability\s*(engineer)?|'
    r'ai\s*engineer|llm\s*engineer|agent\s*engineer|'
    r'applied\s*(ai|ml)\s*engineer|prompt\s*engineer|'
    r'infrastructure\s*engineer|cloud\s*engineer|'
    r'founding\s*engineer|member\s*of\s*technical\s*staff|'
    r'product\s*engineer|systems?\s*engineer'
    r')\b'
)

SWE_EXCLUDE = re.compile(
    r'(?i)\b('
    r'sales\s*engineer|support\s*engineer|field\s*(service|engineer)|'
    r'customer\s*(success|support)\s*engineer|'
    r'solutions?\s*(architect|engineer)|'
    r'systems?\s*administrator|'
    r'civil\s*engineer|mechanical\s*engineer|electrical\s*engineer|'
    r'chemical\s*engineer|industrial\s*engineer|'
    r'audio\s*engineer|recording\s*engineer|sound\s*engineer|'
    r'network\s*engineer|hardware\s*engineer'
    r')\b'
)

CONTROL_PATTERN = re.compile(
    r'(?i)\b('
    r'civil\s*engineer|mechanical\s*engineer|'
    r'nurse|registered\s*nurse|nursing|'
    r'electrical\s*engineer|chemical\s*engineer|'
    r'accountant|accounting|'
    r'financial\s*analyst'
    r')\b'
)

# ---------------------------------------------------------------------------
# Text cleaning utilities
# ---------------------------------------------------------------------------
def strip_markdown(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\\([^\s])', r'\1', text)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def strip_html(text: str) -> str:
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
    if pd.isna(amount) or amount is None:
        return float("nan")
    amount = float(amount)
    if not isinstance(pay_period, str):
        return amount
    period = pay_period.strip().upper()
    multipliers = {
        "HOURLY": 2080, "DAILY": 260, "WEEKLY": 52,
        "BIWEEKLY": 26, "MONTHLY": 12, "YEARLY": 1, "ANNUAL": 1,
    }
    return amount * multipliers.get(period, 1)


def classify_swe(title: str) -> tuple[bool, bool]:
    """Returns (is_swe, is_control)."""
    if not isinstance(title, str):
        return False, False
    is_swe = bool(SWE_INCLUDE.search(title)) and not bool(SWE_EXCLUDE.search(title))
    is_control = bool(CONTROL_PATTERN.search(title))
    return is_swe, is_control


def normalize_title(title: str) -> str:
    """Lowercase, strip level indicators for dedup."""
    if not isinstance(title, str):
        return ""
    t = title.lower().strip()
    # Strip common level prefixes/suffixes
    t = re.sub(r'\b(senior|sr\.?|junior|jr\.?|lead|staff|principal|distinguished)\b', '', t)
    t = re.sub(r'\b(i{1,3}|iv|v)\b', '', t)  # Roman numerals
    t = re.sub(r'\b[1-5]\b', '', t)  # Numeric levels
    t = re.sub(r'\s*[-–—]\s*(remote|hybrid|onsite|on-site)\s*$', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


# ---------------------------------------------------------------------------
# Source 1: Kaggle arshkon (April 2024)
# ---------------------------------------------------------------------------
def load_kaggle_arshkon() -> pd.DataFrame:
    """Load and harmonize the arshkon Kaggle dataset with companion file joins."""
    path = KAGGLE_DIR / "postings.csv"
    if not path.exists():
        log.warning(f"Kaggle arshkon not found at {path}")
        return pd.DataFrame()

    log.info(f"Loading Kaggle arshkon from {path}...")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"  Raw rows: {len(df):,}")

    # --- Join companion files ---
    # Industry: job_industries.csv -> mappings/industries.csv
    industry_df = _load_kaggle_industries()
    if industry_df is not None:
        df = df.merge(industry_df, on="job_id", how="left")
        coverage = df["industry_name"].notna().mean()
        log.info(f"  Industry join: {coverage:.1%} coverage")
    else:
        df["industry_name"] = None

    # Company info: companies.csv
    company_df = _load_kaggle_companies()
    if company_df is not None:
        df = df.merge(company_df, on="company_id", how="left")
        size_coverage = df["company_size_cat"].notna().mean()
        log.info(f"  Company join: {size_coverage:.1%} size coverage")
    else:
        df["company_size_cat"] = None
        df["employee_count"] = None

    # --- Build unified schema ---
    out = pd.DataFrame()
    out["job_id"] = df["job_id"].astype(str)
    out["source"] = "kaggle_arshkon_2024"
    out["source_platform"] = "linkedin"
    out["scrape_date"] = pd.to_datetime(df["listed_time"], unit="ms", errors="coerce").dt.strftime("%Y-%m-%d")
    out["date_posted"] = pd.to_datetime(df["original_listed_time"], unit="ms", errors="coerce").dt.strftime("%Y-%m-%d")
    out["title"] = df["title"]
    out["company_name"] = df["company_name"]
    out["location"] = df["location"]
    out["seniority_native"] = df["formatted_experience_level"].apply(normalize_seniority)
    out["work_type"] = df["formatted_work_type"]
    out["is_remote"] = df["remote_allowed"].fillna(0).astype(bool)
    out["description"] = df["description"]  # Kaggle is already plain text
    out["description_raw"] = df["description"]
    out["skills_raw"] = df["skills_desc"]

    # Salary
    out["min_salary"] = df.apply(
        lambda r: normalize_salary_to_annual(r.get("min_salary"), r.get("pay_period")), axis=1
    )
    out["max_salary"] = df.apply(
        lambda r: normalize_salary_to_annual(r.get("max_salary"), r.get("pay_period")), axis=1
    )
    out["salary_currency"] = df.get("currency")
    out["salary_source"] = np.where(out["min_salary"].notna() | out["max_salary"].notna(), "employer", "missing")

    # Joined fields
    out["company_industry"] = df.get("industry_name")
    out["company_size"] = pd.to_numeric(df.get("employee_count"), errors="coerce")
    out["company_size_category"] = df.get("company_size_cat")
    out["job_url"] = df.get("job_posting_url")
    out["company_id_kaggle"] = df.get("company_id")

    # Classification
    swe_control = out["title"].apply(classify_swe)
    out["is_swe"] = swe_control.apply(lambda x: x[0])
    out["is_control"] = swe_control.apply(lambda x: x[1])

    # Computed columns
    out["title_normalized"] = out["title"].apply(normalize_title)
    out["description_length"] = out["description"].str.len()
    out["posting_age_days"] = (
        pd.to_datetime(out["scrape_date"]) - pd.to_datetime(out["date_posted"])
    ).dt.days

    log.info(f"  Harmonized: {len(out):,} rows, SWE={out['is_swe'].sum():,}, Control={out['is_control'].sum():,}")
    return out


def _load_kaggle_industries() -> pd.DataFrame | None:
    """Join job_industries + mappings/industries to get industry names."""
    ji_path = KAGGLE_DIR / "jobs" / "job_industries.csv"
    mi_path = KAGGLE_DIR / "mappings" / "industries.csv"
    if not ji_path.exists() or not mi_path.exists():
        log.warning("Kaggle industry files not found")
        return None

    ji = pd.read_csv(ji_path)
    mi = pd.read_csv(mi_path)
    # job_industries has (job_id, industry_id) — take first industry per job
    ji = ji.drop_duplicates(subset=["job_id"], keep="first")
    ji = ji.merge(mi, on="industry_id", how="left")
    return ji[["job_id", "industry_name"]]


def _load_kaggle_companies() -> pd.DataFrame | None:
    """Load company metadata for joins."""
    comp_path = KAGGLE_DIR / "companies" / "companies.csv"
    if not comp_path.exists():
        log.warning("Kaggle companies file not found")
        return None

    comp = pd.read_csv(comp_path, low_memory=False)
    # employee_counts has more granular data
    emp_path = KAGGLE_DIR / "companies" / "employee_counts.csv"
    if emp_path.exists():
        emp = pd.read_csv(emp_path)
        # Take the latest count per company
        emp = emp.sort_values("follower_count", ascending=False).drop_duplicates(
            subset=["company_id"], keep="first"
        )
        comp = comp.merge(emp[["company_id", "employee_count"]], on="company_id", how="left")
    else:
        comp["employee_count"] = None

    return comp[["company_id", "company_size", "employee_count"]].rename(
        columns={"company_size": "company_size_cat"}
    )


# ---------------------------------------------------------------------------
# Source 2: Kaggle asaniczka (1.3M, 2024)
# ---------------------------------------------------------------------------
def load_kaggle_asaniczka() -> pd.DataFrame:
    """Load and harmonize the asaniczka 1.3M LinkedIn dataset.

    Key findings from evaluation:
    - 1,348,454 rows, ALL from January 2024 (single month snapshot)
    - job_summary.csv has real descriptions (median ~2K chars)
    - Join key between postings and summaries is job_link
    - Seniority field is job_level (e.g., "Mid senior", "Entry level")
    - job_type is work arrangement (Onsite/Remote/Hybrid), not employment type
    """
    path = ASANICZKA_DIR / "linkedin_job_postings.csv"
    if not path.exists():
        log.warning(f"Kaggle asaniczka not found at {path}")
        return pd.DataFrame()

    log.info(f"Loading Kaggle asaniczka from {path}...")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"  Raw rows: {len(df):,}")

    # Date range (all January 2024)
    df["first_seen_dt"] = pd.to_datetime(df["first_seen"], errors="coerce")
    log.info(f"  Date range: {df['first_seen_dt'].min()} to {df['first_seen_dt'].max()}")
    monthly = df["first_seen_dt"].dt.to_period("M").value_counts().sort_index()
    for month, count in monthly.items():
        log.info(f"    {month}: {count:,}")

    # Load job_summary for descriptions — 4.8GB, use chunked approach
    summary_path = ASANICZKA_DIR / "job_summary.csv"
    if summary_path.exists():
        log.info(f"  Loading job summaries (chunked)...")
        summary_chunks = []
        for chunk in pd.read_csv(summary_path, chunksize=200000, low_memory=False):
            # Only keep job_link and job_summary columns
            summary_chunks.append(chunk[["job_link", "job_summary"]])
        summaries = pd.concat(summary_chunks, ignore_index=True)
        summaries = summaries.drop_duplicates(subset=["job_link"], keep="first")
        log.info(f"  Summaries loaded: {len(summaries):,} rows")
        summary_lengths = summaries["job_summary"].astype(str).str.len()
        log.info(f"  Summary text length: median={summary_lengths.median():.0f}, "
                 f"mean={summary_lengths.mean():.0f}, max={summary_lengths.max():.0f}")

        df = df.merge(summaries, on="job_link", how="left")
        desc_coverage = df["job_summary"].notna().mean()
        log.info(f"  Description join coverage: {desc_coverage:.1%}")
    else:
        df["job_summary"] = None
        log.info("  No job_summary.csv found")

    # Load augmented skills
    skills_path = ASANICZKA_DIR / "job_skills.csv"
    if skills_path.exists():
        log.info(f"  Loading job skills...")
        skills = pd.read_csv(skills_path, low_memory=False)
        # job_skills column is already comma-separated per row
        skills = skills.drop_duplicates(subset=["job_link"], keep="first")
        skills = skills.rename(columns={"job_skills": "skills_structured"})
        df = df.merge(skills[["job_link", "skills_structured"]], on="job_link", how="left")
        skill_coverage = df["skills_structured"].notna().mean()
        log.info(f"  Skills coverage: {skill_coverage:.1%}")
    else:
        df["skills_structured"] = None

    # --- Build unified schema ---
    out = pd.DataFrame()
    # Extract numeric job ID from URL
    out["job_id"] = df["job_link"].astype(str).apply(
        lambda x: "asa_" + x.rstrip("/").split("/")[-1].split("-")[-1]
        if "/" in str(x) else "asa_" + str(hash(x))
    )
    out["source"] = "kaggle_asaniczka_2024"
    out["source_platform"] = "linkedin"
    out["scrape_date"] = df["first_seen_dt"].dt.strftime("%Y-%m-%d")
    out["date_posted"] = df["first_seen_dt"].dt.strftime("%Y-%m-%d")
    out["title"] = df["job_title"]
    out["company_name"] = df["company"]
    out["location"] = df["job_location"]

    # Normalize seniority — asaniczka uses "Mid senior", "Entry level", etc.
    out["seniority_native"] = df["job_level"].apply(normalize_seniority)

    # job_type in asaniczka is work arrangement (Onsite/Remote/Hybrid)
    out["work_type"] = None  # No employment type info
    out["is_remote"] = df["job_type"].str.lower().str.contains("remote", na=False)

    out["description"] = df["job_summary"].fillna("")
    out["description_raw"] = df["job_summary"].fillna("")
    out["skills_raw"] = df.get("skills_structured")
    out["min_salary"] = np.nan
    out["max_salary"] = np.nan
    out["salary_currency"] = None
    out["salary_source"] = "missing"
    out["company_industry"] = None
    out["company_size"] = np.nan
    out["company_size_category"] = None
    out["job_url"] = df["job_link"]
    out["company_id_kaggle"] = None

    # Classification
    swe_control = out["title"].apply(classify_swe)
    out["is_swe"] = swe_control.apply(lambda x: x[0])
    out["is_control"] = swe_control.apply(lambda x: x[1])

    # Computed columns
    out["title_normalized"] = out["title"].apply(normalize_title)
    out["description_length"] = out["description"].str.len()
    out["posting_age_days"] = np.nan  # Single snapshot

    log.info(f"  Harmonized: {len(out):,} rows, SWE={out['is_swe'].sum():,}, Control={out['is_control'].sum():,}")
    return out


# ---------------------------------------------------------------------------
# Source 3: Scraped data (March 2026)
# ---------------------------------------------------------------------------
def load_scraped() -> pd.DataFrame:
    """Load and harmonize all daily scraped CSVs."""
    if not SCRAPED_DIR.exists():
        log.warning(f"Scraped directory not found: {SCRAPED_DIR}")
        return pd.DataFrame()

    csv_files = sorted(SCRAPED_DIR.glob("*_swe_jobs.csv")) + sorted(SCRAPED_DIR.glob("*_non_swe_jobs.csv"))
    if not csv_files:
        log.warning("No scraped CSVs found")
        return pd.DataFrame()

    log.info(f"Loading {len(csv_files)} scraped CSV(s)...")
    dfs = []
    for f in csv_files:
        try:
            chunk = pd.read_csv(f, low_memory=False)
            # Tag which file type this came from
            chunk["_file_type"] = "swe" if "_swe_jobs" in f.name else "non_swe"
            chunk["_file_date"] = f.name.split("_")[0]  # e.g., "2026-03-05"
            dfs.append(chunk)
        except Exception as e:
            log.warning(f"  Could not read {f.name}: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    raw_count = len(df)

    # Dedup across daily files by job_url
    df = df.drop_duplicates(subset=["job_url"], keep="first")
    log.info(f"  Raw: {raw_count:,} -> {len(df):,} after URL dedup")

    # --- Build unified schema ---
    out = pd.DataFrame()
    out["job_id"] = df["id"].astype(str)
    out["source"] = df["site"].apply(
        lambda s: f"scraped_{s}_2026" if pd.notna(s) else "scraped_2026"
    )
    out["source_platform"] = df["site"].str.lower().fillna("unknown")
    out["scrape_date"] = df.get("scrape_date", df["_file_date"])
    out["date_posted"] = pd.to_datetime(df.get("date_posted"), errors="coerce").dt.strftime("%Y-%m-%d")
    out["title"] = df["title"]
    out["company_name"] = df.get("company", df.get("company_name"))
    out["location"] = df["location"]
    out["seniority_native"] = df.get("job_level", pd.Series(dtype=str)).apply(normalize_seniority)
    out["work_type"] = df.get("job_type")
    out["is_remote"] = df.get("is_remote", False)
    # Handle mixed types in is_remote
    out["is_remote"] = out["is_remote"].apply(
        lambda x: bool(x) if not pd.isna(x) else False
    )

    # Description: strip markdown for scraped data
    out["description"] = df["description"].apply(strip_markdown)
    out["description_raw"] = df["description"]
    out["skills_raw"] = df.get("skills")

    # Salary
    out["min_salary"] = pd.to_numeric(df.get("min_amount"), errors="coerce")
    out["max_salary"] = pd.to_numeric(df.get("max_amount"), errors="coerce")
    out["salary_currency"] = df.get("currency")
    out["salary_source"] = df.get("salary_source", "missing").fillna("missing")

    out["company_industry"] = df.get("company_industry")
    out["company_size"] = pd.to_numeric(df.get("company_num_employees"), errors="coerce")
    out["company_size_category"] = None
    out["job_url"] = df.get("job_url")
    out["company_id_kaggle"] = None

    # Classification
    swe_control = out["title"].apply(classify_swe)
    out["is_swe"] = swe_control.apply(lambda x: x[0])
    out["is_control"] = swe_control.apply(lambda x: x[1])

    # Computed columns
    out["title_normalized"] = out["title"].apply(normalize_title)
    out["description_length"] = out["description"].str.len()
    out["posting_age_days"] = (
        pd.to_datetime(out["scrape_date"]) - pd.to_datetime(out["date_posted"])
    ).dt.days

    # Log platform breakdown
    platform_counts = out["source_platform"].value_counts()
    for platform, count in platform_counts.items():
        log.info(f"  Platform {platform}: {count:,}")

    log.info(f"  Harmonized: {len(out):,} rows, SWE={out['is_swe'].sum():,}, Control={out['is_control'].sum():,}")
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_stage1() -> pd.DataFrame:
    """Execute Stage 1: load all sources and produce unified DataFrame."""
    log.info("=" * 60)
    log.info("STAGE 1: Ingest & Schema Unification")
    log.info("=" * 60)

    parts = []

    # Source 1: Kaggle arshkon
    arshkon = load_kaggle_arshkon()
    if len(arshkon) > 0:
        parts.append(arshkon)

    # Source 2: Kaggle asaniczka
    asaniczka = load_kaggle_asaniczka()
    if len(asaniczka) > 0:
        parts.append(asaniczka)

    # Source 3: Scraped
    scraped = load_scraped()
    if len(scraped) > 0:
        parts.append(scraped)

    if not parts:
        log.error("No data loaded!")
        return pd.DataFrame()

    unified = pd.concat(parts, ignore_index=True)
    log.info(f"\nCombined: {len(unified):,} rows from {len(parts)} sources")

    # --- Summary statistics ---
    log.info("\n--- Source breakdown ---")
    for src, count in unified["source"].value_counts().items():
        swe_count = unified.loc[unified["source"] == src, "is_swe"].sum()
        ctrl_count = unified.loc[unified["source"] == src, "is_control"].sum()
        log.info(f"  {src}: {count:,} total, {swe_count:,} SWE, {ctrl_count:,} control")

    log.info("\n--- Platform breakdown ---")
    for plat, count in unified["source_platform"].value_counts().items():
        log.info(f"  {plat}: {count:,}")

    log.info("\n--- SWE classification ---")
    log.info(f"  Total SWE: {unified['is_swe'].sum():,}")
    log.info(f"  Total Control: {unified['is_control'].sum():,}")
    log.info(f"  Total Other: {(~unified['is_swe'] & ~unified['is_control']).sum():,}")

    log.info("\n--- Missing data rates ---")
    for col in ["title", "description", "company_name", "location",
                 "seniority_native", "min_salary", "company_industry", "company_size"]:
        if col in unified.columns:
            missing = unified[col].isna().mean()
            empty = (unified[col].astype(str).str.strip().str.len() == 0).mean()
            log.info(f"  {col:25s} missing={missing:.1%}  empty={empty:.1%}")

    log.info("\n--- Description length by source ---")
    for src in unified["source"].unique():
        sub = unified[unified["source"] == src]
        swe_sub = sub[sub["is_swe"]]
        if len(swe_sub) > 0:
            med = swe_sub["description_length"].median()
            log.info(f"  {src} SWE: median={med:.0f} chars (n={len(swe_sub):,})")

    # Save
    output_path = INTERMEDIATE_DIR / "stage1_unified.parquet"
    unified.to_parquet(output_path, index=False)
    log.info(f"\nSaved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    return unified


if __name__ == "__main__":
    df = run_stage1()
    print(f"\nFinal: {len(df):,} rows, {df['is_swe'].sum():,} SWE")
