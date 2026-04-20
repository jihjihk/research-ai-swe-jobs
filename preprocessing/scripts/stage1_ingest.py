#!/usr/bin/env python3
"""
Stage 1: Ingest and schema unification for the v3 pipeline.

This stage:
  - ingests the approved sources with source-specific schema handling
  - loads all available scraped CSVs in the current 41-column format and skips legacy incompatible files
  - preserves daily scraped observations separately from the canonical corpus
  - writes a downstream-compatible parquet with the v3 canonical fields

Outputs:
  - preprocessing/intermediate/stage1_unified.parquet
  - preprocessing/intermediate/stage1_observations.parquet
  - preprocessing/logs/stage1_ingest_summary.json
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

import os
import resource
import pyarrow as pa
import pyarrow.parquet as pq

from io_utils import (
    cleanup_temp_file,
    prepare_temp_output,
    promote_null_schema,
    promote_temp_file,
    write_parquet_atomic,
    write_text_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KAGGLE_DIR = DATA_DIR / "kaggle-linkedin-jobs-2023-2024"
ASANICZKA_DIR = DATA_DIR / "kaggle-asaniczka-1.3m"
SCRAPED_DIR = DATA_DIR / "scraped"
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"

INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "stage1_ingest.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

ASANICZKA_CHUNK_SIZE = 200_000
PARQUET_COMPRESSION = "zstd"
SCRAPED_EXPECTED_COLUMNS = 41


SENIORITY_MAP = {
    "entry level": "entry",
    "entry_level": "entry",
    "entry": "entry",
    "junior": "entry",
    "associate": "associate",
    "mid senior": "mid-senior",
    "mid-senior level": "mid-senior",
    "mid senior level": "mid-senior",
    "mid-senior": "mid-senior",
    "director": "director",
    "executive": "executive",
    "internship": "intern",
    "intern": "intern",
    "not applicable": None,
}

OUTPUT_COLUMNS = [
    "uid",
    "job_id",
    "source",
    "source_platform",
    "site",
    "title",
    "title_normalized",
    "company_name",
    "company_name_normalized",
    "location",
    "date_posted",
    "scrape_date",
    "description",
    "description_raw",
    "description_length",
    "seniority_native",
    "company_industry",
    "company_size",
    "company_size_raw",
    "company_size_category",
    "search_query",
    "query_tier",
    "search_metro_id",
    "search_metro_name",
    "search_metro_region",
    "search_location",
    "work_type",
    "is_remote",
    "job_url",
    "skills_raw",
    "asaniczka_skills",
    "company_id_kaggle",
    "posting_age_days",
]


def normalize_text(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).replace("\ufeff", "").replace("\u200e", "").replace("\u200f", "")
    text = text.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
    text = text.replace("\\-", "-").replace("\\*", "*").replace("\\#", "#")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text or None


def preserve_raw_text(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value)
    return text if text != "" else None


def normalize_title(title: str | None) -> str:
    if not isinstance(title, str):
        return ""
    value = title.lower().strip()
    value = re.sub(r"\b(senior|sr\.?|junior|jr\.?|lead|staff|principal|distinguished)\b", "", value)
    value = re.sub(r"\b(i{1,3}|iv|v)\b", "", value)
    value = re.sub(r"\b[1-5]\b", "", value)
    value = re.sub(r"\s*[-–—]\s*(remote|hybrid|onsite|on-site)\s*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def map_seniority(value) -> str | None:
    if pd.isna(value):
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    return SENIORITY_MAP.get(key)


def normalize_date_series(series: pd.Series, unit: str | None = None) -> pd.Series:
    if unit is None:
        dt = pd.to_datetime(series, errors="coerce")
    else:
        dt = pd.to_datetime(series, unit=unit, errors="coerce")
    return dt.dt.strftime("%Y-%m-%d")


def parse_company_size(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return np.nan
    nums = [int(part.replace(",", "")) for part in re.findall(r"\d[\d,]*", text)]
    if not nums:
        return np.nan
    if len(nums) == 1 or "+" in text:
        return float(nums[0])
    return float(nums[0] + nums[1]) / 2.0


def normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def finalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["title"] = out["title"].apply(normalize_text)
    out["company_name"] = out["company_name"].apply(normalize_text)
    out["location"] = out["location"].apply(normalize_text)
    out["description"] = out["description"].apply(normalize_text)
    out["description_raw"] = out["description_raw"].apply(preserve_raw_text)
    out["search_query"] = out["search_query"].apply(normalize_text)
    out["query_tier"] = out["query_tier"].apply(normalize_text)
    out["search_metro_id"] = out["search_metro_id"].apply(normalize_text)
    out["search_metro_name"] = out["search_metro_name"].apply(normalize_text)
    out["search_metro_region"] = out["search_metro_region"].apply(normalize_text)
    out["search_location"] = out["search_location"].apply(normalize_text)
    out["work_type"] = out["work_type"].apply(normalize_text)
    out["job_url"] = out["job_url"].apply(normalize_text)
    out["skills_raw"] = out["skills_raw"].apply(normalize_text)
    out["asaniczka_skills"] = out["asaniczka_skills"].apply(normalize_text)
    out["company_industry"] = out["company_industry"].apply(normalize_text)
    out["company_size_raw"] = out["company_size_raw"].apply(normalize_text)
    out["company_size_category"] = out["company_size_category"].apply(normalize_text)
    out["seniority_native"] = out["seniority_native"].apply(normalize_text)
    out["source_platform"] = out["source_platform"].astype("string")
    out["site"] = out["site"].astype("string")
    out["source"] = out["source"].astype("string")
    out["uid"] = out["uid"].astype("string")
    out["job_id"] = out["uid"]
    out["title_normalized"] = out["title"].apply(normalize_title)
    out["company_name_normalized"] = out["company_name"]
    out["description_length"] = out["description"].fillna("").str.len().astype("Int64")
    out["is_remote"] = out["is_remote"].apply(normalize_bool)
    out["company_size"] = pd.to_numeric(out["company_size"], errors="coerce")
    out["scrape_date"] = normalize_date_series(out["scrape_date"])
    out["date_posted"] = normalize_date_series(out["date_posted"])

    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    return out[OUTPUT_COLUMNS]


def load_arshkon_companions() -> tuple[pd.DataFrame, pd.DataFrame]:
    job_industries = pd.read_csv(KAGGLE_DIR / "jobs" / "job_industries.csv")
    industries = pd.read_csv(KAGGLE_DIR / "mappings" / "industries.csv")
    job_industries = (
        job_industries.drop_duplicates(subset=["job_id"], keep="first")
        .merge(industries, on="industry_id", how="left")
    )

    companies = pd.read_csv(KAGGLE_DIR / "companies" / "companies.csv", low_memory=False)
    employee_counts = pd.read_csv(KAGGLE_DIR / "companies" / "employee_counts.csv")
    employee_counts = (
        employee_counts.sort_values(["time_recorded", "follower_count"], ascending=[False, False])
        .drop_duplicates(subset=["company_id"], keep="first")
    )
    companies = companies.merge(
        employee_counts[["company_id", "employee_count"]],
        on="company_id",
        how="left",
    )
    return job_industries[["job_id", "industry_name"]], companies[["company_id", "company_size", "employee_count"]]


def load_kaggle_arshkon(summary: dict) -> pd.DataFrame:
    log.info("Loading Kaggle arshkon...")
    usecols = [
        "job_id",
        "company_name",
        "title",
        "description",
        "location",
        "company_id",
        "formatted_work_type",
        "remote_allowed",
        "job_posting_url",
        "formatted_experience_level",
        "skills_desc",
        "listed_time",
    ]
    df = pd.read_csv(KAGGLE_DIR / "postings.csv", usecols=usecols, low_memory=False)
    summary["arshkon_raw"] = int(len(df))

    industry_df, company_df = load_arshkon_companions()
    df = df.merge(industry_df, on="job_id", how="left")
    df = df.merge(company_df, on="company_id", how="left")

    summary["arshkon_industry_join_rate"] = round(df["industry_name"].notna().mean(), 4)
    summary["arshkon_employee_count_join_rate"] = round(df["employee_count"].notna().mean(), 4)

    out = pd.DataFrame(
        {
            "uid": "arshkon_" + df["job_id"].astype("Int64").astype(str),
            "source": "kaggle_arshkon",
            "source_platform": "linkedin",
            "site": "linkedin",
            "title": df["title"],
            "company_name": df["company_name"],
            "location": df["location"],
            "date_posted": pd.to_datetime(df["listed_time"], unit="ms", errors="coerce"),
            "scrape_date": pd.Series([pd.NaT] * len(df)),
            "description": df["description"],
            "description_raw": df["description"],
            "seniority_native": df["formatted_experience_level"].apply(map_seniority),
            "company_industry": df["industry_name"],
            "company_size": df["employee_count"],
            "company_size_raw": df["employee_count"],
            "company_size_category": df["company_size"],
            "search_query": pd.NA,
            "query_tier": pd.NA,
            "search_metro_id": pd.NA,
            "search_metro_name": pd.NA,
            "search_metro_region": pd.NA,
            "search_location": pd.NA,
            "work_type": df["formatted_work_type"],
            "is_remote": df["remote_allowed"],
            "job_url": df["job_posting_url"],
            "skills_raw": df["skills_desc"],
            "asaniczka_skills": pd.NA,
            "company_id_kaggle": df["company_id"],
        }
    )
    raw_seniority = df["formatted_experience_level"]
    out = finalize_frame(out)
    summary["arshkon_description_missing_rate"] = round(out["description"].isna().mean(), 4)
    summary["arshkon_seniority_unmapped_rate"] = round(
        (raw_seniority.notna() & out["seniority_native"].isna()).mean(), 4
    )
    log.info(
        "  arshkon rows loaded: %s, median description length: %s",
        f"{len(out):,}",
        f"{int(out['description_length'].median()):,}" if len(out) else "0",
    )
    return out


def load_kaggle_asaniczka(summary: dict) -> pd.DataFrame:
    log.info("Loading Kaggle asaniczka in chunks...")
    postings_path = ASANICZKA_DIR / "linkedin_job_postings.csv"
    usecols = [
        "job_link",
        "job_title",
        "company",
        "job_location",
        "first_seen",
        "search_city",
        "search_country",
        "search_position",
        "job_level",
        "job_type",
    ]

    matched_chunks: list[pd.DataFrame] = []
    job_links: set[str] = set()
    total_rows = 0
    us_rows = 0

    for chunk in pd.read_csv(postings_path, usecols=usecols, chunksize=ASANICZKA_CHUNK_SIZE, low_memory=False):
        total_rows += len(chunk)
        us_mask = chunk["search_country"].eq("United States")
        us_rows += int(us_mask.sum())
        chunk = chunk.loc[us_mask].copy()
        if not chunk.empty:
            matched_chunks.append(chunk)
            job_links.update(chunk["job_link"].astype(str).tolist())

    summary["asaniczka_raw"] = int(total_rows)
    summary["asaniczka_us_filtered"] = int(us_rows)

    if not matched_chunks:
        summary["asaniczka_description_join_rate"] = 0.0
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    postings = pd.concat(matched_chunks, ignore_index=True)

    summary_frames: list[pd.DataFrame] = []
    summary_path = ASANICZKA_DIR / "job_summary.csv"
    for chunk in pd.read_csv(summary_path, usecols=["job_link", "job_summary"], chunksize=ASANICZKA_CHUNK_SIZE, low_memory=False):
        kept = chunk.loc[chunk["job_link"].isin(job_links)]
        if not kept.empty:
            summary_frames.append(kept)
    if summary_frames:
        summaries = pd.concat(summary_frames, ignore_index=True).drop_duplicates(subset=["job_link"], keep="first")
    else:
        summaries = pd.DataFrame(columns=["job_link", "job_summary"])

    skill_frames: list[pd.DataFrame] = []
    skills_path = ASANICZKA_DIR / "job_skills.csv"
    for chunk in pd.read_csv(skills_path, usecols=["job_link", "job_skills"], chunksize=ASANICZKA_CHUNK_SIZE, low_memory=False):
        kept = chunk.loc[chunk["job_link"].isin(job_links)]
        if not kept.empty:
            skill_frames.append(kept)
    if skill_frames:
        skills = pd.concat(skill_frames, ignore_index=True).drop_duplicates(subset=["job_link"], keep="first")
    else:
        skills = pd.DataFrame(columns=["job_link", "job_skills"])

    postings = postings.merge(summaries, on="job_link", how="left")
    postings = postings.merge(skills, on="job_link", how="left")

    summary["asaniczka_description_join_rate"] = round(postings["job_summary"].notna().mean(), 4)
    summary["asaniczka_skills_join_rate"] = round(postings["job_skills"].notna().mean(), 4)

    def hash_uid(job_link: str) -> str:
        digest = hashlib.sha256(job_link.encode("utf-8")).hexdigest()[:16]
        return f"asaniczka_{digest}"

    out = pd.DataFrame(
        {
            "uid": postings["job_link"].astype(str).apply(hash_uid),
            "source": "kaggle_asaniczka",
            "source_platform": "linkedin",
            "site": "linkedin",
            "title": postings["job_title"],
            "company_name": postings["company"],
            "location": postings["job_location"],
            "date_posted": pd.to_datetime(postings["first_seen"], errors="coerce"),
            "scrape_date": pd.Series([pd.NaT] * len(postings)),
            "description": postings["job_summary"],
            "description_raw": postings["job_summary"],
            "seniority_native": postings["job_level"].apply(map_seniority),
            "company_industry": pd.NA,
            "company_size": np.nan,
            "company_size_raw": pd.NA,
            "company_size_category": pd.NA,
            "search_query": postings["search_position"],
            "query_tier": pd.NA,
            "search_metro_id": pd.NA,
            "search_metro_name": postings["search_city"],
            "search_metro_region": pd.NA,
            "search_location": pd.NA,
            "work_type": pd.NA,
            "is_remote": postings["job_type"].astype("string").str.lower().str.contains("remote", na=False),
            "job_url": postings["job_link"],
            "skills_raw": postings["job_skills"],
            "asaniczka_skills": postings["job_skills"],
            "company_id_kaggle": pd.NA,
        }
    )
    raw_seniority = postings["job_level"]
    out = finalize_frame(out)
    summary["asaniczka_description_missing_rate"] = round(out["description"].isna().mean(), 4)
    summary["asaniczka_seniority_unmapped_rate"] = round(
        (raw_seniority.notna() & out["seniority_native"].isna()).mean(), 4
    )
    log.info(
        "  asaniczka US rows loaded: %s, description join rate: %.1f%%",
        f"{len(out):,}",
        100 * summary["asaniczka_description_join_rate"],
    )
    return out


def list_scraped_files() -> list[Path]:
    files = []
    for path in sorted(SCRAPED_DIR.glob("*_jobs.csv")):
        if "yc" in path.name.lower():
            continue
        match = re.match(r"(\d{4}-\d{2}-\d{2})_(swe|non_swe)_jobs\.csv$", path.name)
        if not match:
            continue
        files.append(path)
    return files


def load_scraped(summary: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Loading scraped observations...")
    files = list_scraped_files()
    if not files:
        summary["scraped_files_loaded"] = []
        summary["scraped_files_skipped"] = []
        summary["scraped_observation_rows"] = 0
        summary["scraped_unique_ids"] = 0
        return pd.DataFrame(columns=OUTPUT_COLUMNS), pd.DataFrame(columns=OUTPUT_COLUMNS)

    frames = []
    loaded_files: list[str] = []
    skipped_files: list[dict[str, object]] = []
    for path in files:
        match = re.match(r"(\d{4}-\d{2}-\d{2})_", path.name)
        file_date = match.group(1)
        chunk = pd.read_csv(path, low_memory=False)
        if len(chunk.columns) != SCRAPED_EXPECTED_COLUMNS:
            skipped_files.append(
                {
                    "file": path.name,
                    "reason": "unexpected_column_count",
                    "column_count": int(len(chunk.columns)),
                }
            )
            log.warning(
                "Skipping scraped file %s: %s columns found, expected %s",
                path.name,
                len(chunk.columns),
                SCRAPED_EXPECTED_COLUMNS,
            )
            continue
        chunk["scrape_date"] = file_date
        frames.append(chunk)
        loaded_files.append(path.name)

    summary["scraped_files_loaded"] = loaded_files
    summary["scraped_files_skipped"] = skipped_files

    if not frames:
        summary["scraped_observation_rows"] = 0
        summary["scraped_unique_ids"] = 0
        return pd.DataFrame(columns=OUTPUT_COLUMNS), pd.DataFrame(columns=OUTPUT_COLUMNS)

    raw = pd.concat(frames, ignore_index=True)
    summary["scraped_observation_rows"] = int(len(raw))
    summary["scraped_sites"] = {str(k): int(v) for k, v in raw["site"].value_counts(dropna=False).items()}
    summary["scraped_query_tiers"] = {
        str(k): int(v) for k, v in raw["query_tier"].fillna("<NA>").value_counts(dropna=False).items()
    }

    out = pd.DataFrame(
        {
            "uid": raw["site"].astype(str).str.lower() + "_" + raw["id"].astype(str),
            "source": "scraped",
            "source_platform": raw["site"].astype("string").str.lower(),
            "site": raw["site"].astype("string").str.lower(),
            "title": raw["title"],
            "company_name": raw["company"],
            "location": raw["location"],
            "date_posted": pd.to_datetime(raw["date_posted"], errors="coerce"),
            "scrape_date": pd.to_datetime(raw["scrape_date"], errors="coerce"),
            "description": raw["description"],
            "description_raw": raw["description"],
            "seniority_native": raw["job_level"].apply(map_seniority),
            "company_industry": raw["company_industry"],
            "company_size": raw["company_num_employees"].apply(parse_company_size),
            "company_size_raw": raw["company_num_employees"],
            "company_size_category": pd.NA,
            "search_query": raw["search_query"],
            "query_tier": raw["query_tier"],
            "search_metro_id": raw["search_metro_id"],
            "search_metro_name": raw["search_metro_name"],
            "search_metro_region": raw["search_metro_region"],
            "search_location": raw["search_location"],
            "work_type": raw["job_type"],
            "is_remote": raw["is_remote"],
            "job_url": raw["job_url"],
            "skills_raw": raw["skills"],
            "asaniczka_skills": pd.NA,
            "company_id_kaggle": pd.NA,
        }
    )
    raw_seniority = raw["job_level"]
    observations = finalize_frame(out)
    summary["scraped_description_missing_rate"] = round(observations["description"].isna().mean(), 4)
    summary["scraped_seniority_unmapped_rate"] = round(
        (raw_seniority.notna() & observations["seniority_native"].isna()).mean(), 4
    )
    summary["scraped_unique_ids"] = int(observations["uid"].nunique())

    canonical = (
        observations.sort_values(["uid", "scrape_date"], kind="stable")
        .drop_duplicates(subset=["uid"], keep="first")
        .reset_index(drop=True)
    )

    log.info(
        "  scraped observations: %s, unique postings: %s",
        f"{len(observations):,}",
        f"{len(canonical):,}",
    )
    return canonical, observations


NULL_RATE_COLUMNS = [
    "title",
    "description",
    "description_raw",
    "company_name",
    "location",
    "date_posted",
    "seniority_native",
    "company_industry",
    "company_size",
    "search_query",
]


def build_null_rate_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for source, subset in df.groupby("source", dropna=False):
        out[str(source)] = {
            col: round(float(subset[col].isna().mean()), 4) for col in NULL_RATE_COLUMNS
        }
    return out


def log_null_rates(df: pd.DataFrame, label: str) -> None:
    log.info("Null rates for %s by source:", label)
    for source, rates in build_null_rate_summary(df).items():
        log.info("  source=%s", source)
        for col in NULL_RATE_COLUMNS:
            log.info("    %-20s %.1f%%", col, 100 * rates[col])


def _mem_gb() -> tuple[float, float, float]:
    """(current_rss_gb, peak_rss_gb, system_available_gb)."""
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak_kb / (1024 * 1024)
    try:
        with open(f"/proc/{os.getpid()}/statm") as fh:
            # statm: size resident shared text lib data dt (in pages)
            resident_pages = int(fh.read().split()[1])
        page_size = resource.getpagesize()
        current_gb = (resident_pages * page_size) / (1024 ** 3)
    except Exception:
        current_gb = -1.0
    try:
        with open("/proc/meminfo") as fh:
            meminfo = {
                line.split(":")[0]: int(line.split()[1])
                for line in fh
                if line.split()[0].rstrip(":") in ("MemAvailable", "MemFree")
            }
        avail_gb = meminfo.get("MemAvailable", 0) / (1024 * 1024)
    except Exception:
        avail_gb = -1.0
    return current_gb, peak_gb, avail_gb


def _log_mem(tag: str) -> None:
    cur, peak, avail = _mem_gb()
    log.info(
        "[mem] %-50s rss=%5.2fGB peak=%5.2fGB sys_avail=%5.2fGB",
        tag,
        cur,
        peak,
        avail,
    )


class _StreamingParquet:
    """Append DataFrames to a single parquet file without ever concatenating
    them in memory. The first part defines the schema; subsequent parts are
    aligned to it (missing columns filled with nulls, extras dropped).
    """

    def __init__(self, final_path: Path, compression: str | None) -> None:
        self.final_path = final_path
        self.tmp_path = prepare_temp_output(final_path)
        self.compression = compression
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema | None = None
        self.rows = 0

    def append(self, df: pd.DataFrame, chunk_size: int = 200_000) -> None:
        if df is None or len(df) == 0:
            return
        total = len(df)
        tag = self.final_path.name
        _log_mem(f"{tag} append start rows={total:,}")
        for start in range(0, total, chunk_size):
            sub = df.iloc[start : start + chunk_size]
            table = pa.Table.from_pandas(sub, preserve_index=False)
            if self.writer is None:
                self.schema = promote_null_schema(table.schema)
                self.writer = pq.ParquetWriter(
                    self.tmp_path, self.schema, compression=self.compression
                )
            else:
                # Add missing columns as null arrays, drop extras, reorder.
                have = set(table.schema.names)
                need = list(self.schema.names)
                for col in need:
                    if col not in have:
                        field = self.schema.field(col)
                        null_arr = pa.nulls(len(table), type=field.type)
                        table = table.append_column(field, null_arr)
                table = table.select(need)
            self.writer.write_table(table.cast(self.schema))
            self.rows += len(sub)
            del table
            del sub
            gc.collect()
            if ((start // chunk_size) % 5) == 0:
                _log_mem(f"{tag} wrote {min(start + chunk_size, total):,}/{total:,}")

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            promote_temp_file(self.tmp_path, self.final_path)

    def abort(self) -> None:
        if self.writer is not None:
            try:
                self.writer.close()
            finally:
                self.writer = None
        cleanup_temp_file(self.tmp_path)


def _per_source_stats(df: pd.DataFrame) -> dict[str, dict]:
    """Return stats keyed by source value. Each source entry contains row
    counts, platform breakdown, null rates, and required-field null counts.
    """
    out: dict[str, dict] = {}
    if len(df) == 0:
        return out
    for source, subset in df.groupby("source", dropna=False):
        key = str(source)
        out[key] = {
            "rows": int(len(subset)),
            "platform_counts": {
                str(k): int(v)
                for k, v in subset["source_platform"].value_counts(dropna=False).items()
            },
            "null_rates": {
                col: round(float(subset[col].isna().mean()), 4)
                for col in NULL_RATE_COLUMNS
            },
            "required_nulls": {
                col: int(subset[col].isna().sum())
                for col in ["uid", "source", "title"]
            },
        }
    return out


def _merge_stats(acc: dict[str, dict], new: dict[str, dict]) -> None:
    """Merge per-source stats dicts. Null rates are remixed by row-weighted
    averaging; all other integer counts sum.
    """
    for source, s in new.items():
        if source not in acc:
            acc[source] = {
                "rows": 0,
                "platform_counts": {},
                "null_rates": {col: 0.0 for col in NULL_RATE_COLUMNS},
                "null_counts": {col: 0 for col in NULL_RATE_COLUMNS},
                "required_nulls": {col: 0 for col in ["uid", "source", "title"]},
            }
        bucket = acc[source]
        bucket["rows"] += s["rows"]
        for plat, n in s["platform_counts"].items():
            bucket["platform_counts"][plat] = bucket["platform_counts"].get(plat, 0) + n
        for col, rate in s["null_rates"].items():
            bucket["null_counts"][col] += int(round(rate * s["rows"]))
        for col, n in s["required_nulls"].items():
            bucket["required_nulls"][col] = bucket["required_nulls"].get(col, 0) + n


def _finalize_stats(acc: dict[str, dict]) -> dict[str, dict]:
    """Convert accumulated null counts back to rates."""
    out: dict[str, dict] = {}
    for source, bucket in acc.items():
        n = max(bucket["rows"], 1)
        out[source] = {
            "rows": bucket["rows"],
            "platform_counts": bucket["platform_counts"],
            "null_rates": {
                col: round(bucket["null_counts"][col] / n, 4)
                for col in NULL_RATE_COLUMNS
            },
            "required_nulls": bucket["required_nulls"],
        }
    return out


def run_stage1() -> tuple[int, int]:
    log.info("=" * 60)
    log.info("STAGE 1: INGEST AND SCHEMA UNIFICATION")
    log.info("=" * 60)

    summary: dict[str, object] = {
        "scraped_file_policy": "load all YYYY-MM-DD_{swe,non_swe}_jobs.csv files; skip YC and non-41-column legacy files"
    }

    unified_path = INTERMEDIATE_DIR / "stage1_unified.parquet"
    observations_path = INTERMEDIATE_DIR / "stage1_observations.parquet"
    summary_path = LOG_DIR / "stage1_ingest_summary.json"

    unified_writer = _StreamingParquet(unified_path, compression=PARQUET_COMPRESSION)
    obs_writer = _StreamingParquet(observations_path, compression=PARQUET_COMPRESSION)

    canonical_stats: dict[str, dict] = {}
    observation_stats: dict[str, dict] = {}

    def _process(df: pd.DataFrame, *, target: str) -> None:
        if df is None or len(df) == 0:
            return
        stats = _per_source_stats(df)
        if target == "unified":
            _merge_stats(canonical_stats, stats)
            unified_writer.append(df)
        else:
            _merge_stats(observation_stats, stats)
            obs_writer.append(df)

    try:
        _log_mem("before load_kaggle_arshkon")
        arshkon = load_kaggle_arshkon(summary)
        _log_mem(f"after load_kaggle_arshkon rows={len(arshkon):,}")
        _process(arshkon, target="unified")
        _process(arshkon, target="observations")
        del arshkon
        gc.collect()
        _log_mem("after del arshkon")

        _log_mem("before load_kaggle_asaniczka")
        asaniczka = load_kaggle_asaniczka(summary)
        _log_mem(f"after load_kaggle_asaniczka rows={len(asaniczka):,}")
        _process(asaniczka, target="unified")
        _process(asaniczka, target="observations")
        del asaniczka
        gc.collect()
        _log_mem("after del asaniczka")

        _log_mem("before load_scraped")
        scraped_canonical, scraped_observations = load_scraped(summary)
        _log_mem(
            f"after load_scraped canonical={len(scraped_canonical):,} "
            f"observations={len(scraped_observations):,}"
        )
        _process(scraped_canonical, target="unified")
        del scraped_canonical
        gc.collect()
        _log_mem("after del scraped_canonical")
        _process(scraped_observations, target="observations")
        del scraped_observations
        gc.collect()
        _log_mem("after del scraped_observations")

        unified_writer.close()
        log.info("Saved %s", unified_path)
        _log_mem("after close unified writer")
        obs_writer.close()
        log.info("Saved %s", observations_path)
        _log_mem("after close obs writer")
    except Exception:
        unified_writer.abort()
        obs_writer.abort()
        raise

    canonical_final = _finalize_stats(canonical_stats)
    observation_final = _finalize_stats(observation_stats)

    canonical_rows = unified_writer.rows
    observations_rows = obs_writer.rows

    summary["stage1_unified_rows"] = canonical_rows
    summary["stage1_observation_rows"] = observations_rows
    summary["source_breakdown_unified"] = {
        src: info["rows"] for src, info in canonical_final.items()
    }
    platform_breakdown: dict[str, int] = {}
    for info in canonical_final.values():
        for plat, n in info["platform_counts"].items():
            platform_breakdown[plat] = platform_breakdown.get(plat, 0) + n
    summary["source_platform_breakdown_unified"] = platform_breakdown
    summary["canonical_null_rates_by_source"] = {
        src: info["null_rates"] for src, info in canonical_final.items()
    }
    summary["observation_null_rates_by_source"] = {
        src: info["null_rates"] for src, info in observation_final.items()
    }
    summary["required_field_nulls"] = {
        col: sum(info["required_nulls"].get(col, 0) for info in canonical_final.values())
        for col in ["uid", "source", "title"]
    }

    log.info("Unified rows: %s", f"{canonical_rows:,}")
    for src, n in summary["source_breakdown_unified"].items():
        log.info("  %s: %s", src, f"{n:,}")
    for plat, n in platform_breakdown.items():
        log.info("  platform=%s: %s", plat, f"{n:,}")

    log.info("Null rates for canonical output by source:")
    for src, info in canonical_final.items():
        log.info("  source=%s", src)
        for col in NULL_RATE_COLUMNS:
            log.info("    %-20s %.1f%%", col, 100 * info["null_rates"][col])

    write_text_atomic(json.dumps(summary, indent=2), summary_path)
    log.info("Saved %s", summary_path)

    return canonical_rows, observations_rows


if __name__ == "__main__":
    unified_rows, observations_rows = run_stage1()
    print(
        f"stage1_unified={unified_rows:,} rows | "
        f"stage1_observations={observations_rows:,} rows"
    )
