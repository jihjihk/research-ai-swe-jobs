#!/usr/bin/env python3
"""T07 external benchmarks and power analysis.

All local row-level work is pushed into DuckDB with a 4GB memory cap. Python only
materializes small grouped results, the canonical T30 panel, and external
benchmark tables.
"""

from __future__ import annotations

import csv
import http.client
import json
import math
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import NormalDist

import duckdb


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "exploration" / "tables" / "T07"
FIGURE_DIR = ROOT / "exploration" / "figures" / "T07"
RAW_DIR = TABLE_DIR / "benchmark_raw"
PANEL_PATH = ROOT / "exploration" / "artifacts" / "shared" / "seniority_definition_panel.csv"
UNIFIED_PATH = ROOT / "data" / "unified.parquet"

DEFAULT_FILTER = """
source_platform = 'linkedin'
AND is_english = true
AND date_flag = 'ok'
AND is_swe = true
"""

STATE_ABBR_BY_FIPS = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}

SOCS_FOR_BENCHMARK = {
    "15-1252": "151252",
    # The task requested 15-1256, but current OEWS metadata has no 151256 row.
    # 15-1253 is the current detailed QA/tester companion to software developers.
    "15-1253_substitute_for_requested_15-1256": "151253",
}


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def duckdb_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=1")
    return con


def sql_quote_path(path: Path) -> str:
    return str(path).replace("'", "''")


def copy_query(con: duckdb.DuckDBPyConnection, query: str, out_path: Path) -> None:
    out = sql_quote_path(out_path)
    con.execute(f"COPY ({query}) TO '{out}' (HEADER, DELIMITER ',')")


def fetch_dict_rows(con: duckdb.DuckDBPyConnection, query: str) -> list[dict[str, object]]:
    result = con.execute(query)
    cols = [d[0] for d in result.description]
    return [dict(zip(cols, row)) for row in result.fetchall()]


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_int(value: object) -> int:
    if value in (None, ""):
        return 0
    return int(round(float(value)))


def mde_values(n1: int, n2: int, alpha: float = 0.05, power: float = 0.80) -> tuple[float | None, float | None]:
    if n1 <= 1 or n2 <= 1:
        return None, None
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    z = z_alpha + z_power
    se_unit = math.sqrt(1 / n1 + 1 / n2)
    mde_continuous = z * se_unit
    # Conservative worst-case binary MDE at p=0.5, where p(1-p) is maximized.
    mde_binary = z * math.sqrt(0.25 * (1 / n1 + 1 / n2))
    return mde_binary, mde_continuous


def verdict(n1: int, n2: int, mde_binary: float | None, mde_continuous: float | None) -> str:
    if mde_binary is None or mde_continuous is None or min(n1, n2) < 50:
        return "underpowered"
    if mde_binary <= 0.05 and mde_continuous <= 0.10:
        return "well-powered"
    if mde_binary <= 0.08 and mde_continuous <= 0.20:
        return "adequate"
    if mde_binary <= 0.14 and mde_continuous <= 0.35:
        return "thin"
    return "underpowered"


def make_group_sizes_and_feasibility(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    panel_rows = read_csv_dicts(PANEL_PATH)
    wanted_defs = {"J1", "J2", "J3", "J4", "S1", "S2", "S3", "S4"}
    panel_rows = [r for r in panel_rows if r["definition"] in wanted_defs]

    all_swe_rows = fetch_dict_rows(
        con,
        f"""
        WITH base AS (
          SELECT
            CASE
              WHEN source = 'kaggle_arshkon' THEN 'arshkon'
              WHEN source = 'kaggle_asaniczka' THEN 'asaniczka'
              WHEN source = 'scraped' THEN 'scraped_2026'
            END AS source,
            CASE
              WHEN source IN ('kaggle_arshkon', 'kaggle_asaniczka') THEN '2024'
              WHEN source = 'scraped' THEN '2026'
            END AS period
          FROM read_parquet('{sql_quote_path(UNIFIED_PATH)}')
          WHERE {DEFAULT_FILTER}
        ),
        source_counts AS (
          SELECT 'all_swe' AS definition, 'all' AS side, period, source,
                 count(*) AS n_of_all, count(*) AS n_of_known,
                 1.0 AS share_of_all, 1.0 AS share_of_known
          FROM base
          GROUP BY period, source
        ),
        pooled AS (
          SELECT 'all_swe' AS definition, 'all' AS side, '2024' AS period, 'pooled_2024' AS source,
                 count(*) AS n_of_all, count(*) AS n_of_known,
                 1.0 AS share_of_all, 1.0 AS share_of_known
          FROM base
          WHERE period = '2024'
        )
        SELECT * FROM source_counts
        UNION ALL
        SELECT * FROM pooled
        ORDER BY definition, source
        """,
    )

    group_size_rows: list[dict[str, object]] = []
    for r in panel_rows:
        group_size_rows.append(
            {
                "definition": r["definition"],
                "side": r["side"],
                "period": r["period"],
                "source": r["source"],
                "n_of_all": as_int(r["n_of_all"]),
                "n_of_known": as_int(r["n_of_known"]),
                "share_of_all": r["share_of_all"],
                "share_of_known": r["share_of_known"],
            }
        )
    group_size_rows.extend(all_swe_rows)
    write_csv(
        TABLE_DIR / "seniority_group_sizes.csv",
        group_size_rows,
        ["definition", "side", "period", "source", "n_of_all", "n_of_known", "share_of_all", "share_of_known"],
    )

    n_by_def_source: dict[tuple[str, str], int] = {}
    for r in group_size_rows:
        n_by_def_source[(str(r["definition"]), str(r["source"]))] = as_int(r["n_of_all"])

    feasibility_rows: list[dict[str, object]] = []

    def add_row(analysis_type: str, comparison: str, seniority_def: str, source1: str, source2: str) -> None:
        n1 = n_by_def_source[(seniority_def, source1)]
        n2 = n_by_def_source[(seniority_def, source2)]
        mde_binary, mde_continuous = mde_values(n1, n2)
        feasibility_rows.append(
            {
                "analysis_type": analysis_type,
                "comparison": comparison,
                "seniority_def": "N/A" if seniority_def == "all_swe" else seniority_def,
                "n_group1": n1,
                "n_group2": n2,
                "MDE_binary": "" if mde_binary is None else round(mde_binary, 6),
                "MDE_continuous": "" if mde_continuous is None else round(mde_continuous, 6),
                "verdict": verdict(n1, n2, mde_binary, mde_continuous),
            }
        )

    for definition in ["J1", "J2", "J3", "J4"]:
        add_row("junior", "arshkon_vs_scraped", definition, "arshkon", "scraped_2026")
        add_row("junior", "pooled_2024_vs_scraped", definition, "pooled_2024", "scraped_2026")

    for definition in ["S1", "S2", "S3", "S4"]:
        add_row("senior", "arshkon_senior_vs_scraped_senior", definition, "arshkon", "scraped_2026")
        add_row("senior", "pooled_2024_senior_vs_scraped_senior", definition, "pooled_2024", "scraped_2026")

    add_row("all_swe", "arshkon_vs_scraped", "all_swe", "arshkon", "scraped_2026")
    add_row("all_swe", "pooled_2024_vs_scraped", "all_swe", "pooled_2024", "scraped_2026")

    write_csv(
        TABLE_DIR / "feasibility_summary.csv",
        feasibility_rows,
        ["analysis_type", "comparison", "seniority_def", "n_group1", "n_group2", "MDE_binary", "MDE_continuous", "verdict"],
    )

    verdict_counts: dict[str, int] = {}
    for row in feasibility_rows:
        verdict_counts[str(row["verdict"])] = verdict_counts.get(str(row["verdict"]), 0) + 1

    return {
        "group_sizes": group_size_rows,
        "feasibility": feasibility_rows,
        "verdict_counts": verdict_counts,
    }


def make_metro_tables(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    copy_query(
        con,
        f"""
        WITH base AS (
          SELECT
            CASE WHEN source IN ('kaggle_arshkon', 'kaggle_asaniczka') THEN '2024'
                 WHEN source = 'scraped' THEN '2026'
            END AS period_group,
            metro_area
          FROM read_parquet('{sql_quote_path(UNIFIED_PATH)}')
          WHERE {DEFAULT_FILTER}
            AND metro_area IS NOT NULL
        )
        SELECT
          metro_area,
          sum(CASE WHEN period_group = '2024' THEN 1 ELSE 0 END) AS n_2024,
          sum(CASE WHEN period_group = '2026' THEN 1 ELSE 0 END) AS n_2026,
          (sum(CASE WHEN period_group = '2024' THEN 1 ELSE 0 END) >= 50
            AND sum(CASE WHEN period_group = '2026' THEN 1 ELSE 0 END) >= 50) AS qualifies_ge50_both_periods,
          (sum(CASE WHEN period_group = '2024' THEN 1 ELSE 0 END) >= 100
            AND sum(CASE WHEN period_group = '2026' THEN 1 ELSE 0 END) >= 100) AS qualifies_ge100_both_periods
        FROM base
        GROUP BY metro_area
        ORDER BY qualifies_ge100_both_periods DESC, n_2026 DESC, n_2024 DESC, metro_area
        """,
        TABLE_DIR / "metro_counts_by_period.csv",
    )

    copy_query(
        con,
        f"""
        WITH base AS (
          SELECT
            CASE WHEN source IN ('kaggle_arshkon', 'kaggle_asaniczka') THEN '2024'
                 WHEN source = 'scraped' THEN '2026'
            END AS period_group,
            metro_area,
            is_multi_location
          FROM read_parquet('{sql_quote_path(UNIFIED_PATH)}')
          WHERE {DEFAULT_FILTER}
        )
        SELECT
          period_group,
          count(*) AS n_swe_rows,
          sum(CASE WHEN metro_area IS NOT NULL THEN 1 ELSE 0 END) AS n_with_metro,
          sum(CASE WHEN metro_area IS NULL THEN 1 ELSE 0 END) AS n_excluded_null_metro,
          sum(CASE WHEN coalesce(is_multi_location, false) THEN 1 ELSE 0 END) AS n_multi_location,
          sum(CASE WHEN metro_area IS NULL AND NOT coalesce(is_multi_location, false) THEN 1 ELSE 0 END) AS n_other_null_metro
        FROM base
        GROUP BY period_group
        ORDER BY period_group
        """,
        TABLE_DIR / "metro_exclusions.csv",
    )

    metro_rows = read_csv_dicts(TABLE_DIR / "metro_counts_by_period.csv")
    summary_rows: list[dict[str, object]] = []
    for threshold in [50, 100]:
        qualified = [
            r
            for r in metro_rows
            if as_int(r["n_2024"]) >= threshold and as_int(r["n_2026"]) >= threshold
        ]
        summary_rows.append(
            {
                "threshold": f">={threshold}_both_periods",
                "n_qualified_metros": len(qualified),
                "qualified_metros": "; ".join(r["metro_area"] for r in qualified),
            }
        )
    write_csv(TABLE_DIR / "metro_feasibility_summary.csv", summary_rows, ["threshold", "n_qualified_metros", "qualified_metros"])

    return {"metro_summary": summary_rows, "metro_exclusions": read_csv_dicts(TABLE_DIR / "metro_exclusions.csv")}


def make_company_tables(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    copy_query(
        con,
        f"""
        WITH company_counts AS (
          SELECT
            company_name_canonical,
            sum(CASE WHEN source = 'kaggle_arshkon' THEN 1 ELSE 0 END) AS n_arshkon,
            sum(CASE WHEN source = 'scraped' THEN 1 ELSE 0 END) AS n_scraped
          FROM read_parquet('{sql_quote_path(UNIFIED_PATH)}')
          WHERE {DEFAULT_FILTER}
            AND source IN ('kaggle_arshkon', 'scraped')
            AND company_name_canonical IS NOT NULL
            AND trim(company_name_canonical) <> ''
          GROUP BY company_name_canonical
        )
        SELECT
          company_name_canonical,
          n_arshkon,
          n_scraped,
          n_arshkon + n_scraped AS n_total
        FROM company_counts
        WHERE n_arshkon >= 3 AND n_scraped >= 3
        ORDER BY n_total DESC, n_scraped DESC, n_arshkon DESC, company_name_canonical
        """,
        TABLE_DIR / "company_overlap_ge3.csv",
    )

    summary_rows = fetch_dict_rows(
        con,
        f"""
        WITH company_counts AS (
          SELECT
            company_name_canonical,
            sum(CASE WHEN source = 'kaggle_arshkon' THEN 1 ELSE 0 END) AS n_arshkon,
            sum(CASE WHEN source = 'scraped' THEN 1 ELSE 0 END) AS n_scraped
          FROM read_parquet('{sql_quote_path(UNIFIED_PATH)}')
          WHERE {DEFAULT_FILTER}
            AND source IN ('kaggle_arshkon', 'scraped')
            AND company_name_canonical IS NOT NULL
            AND trim(company_name_canonical) <> ''
          GROUP BY company_name_canonical
        )
        SELECT
          count(*) AS companies_ge3_both,
          sum(n_arshkon) AS covered_arshkon_rows,
          sum(n_scraped) AS covered_scraped_rows,
          avg(n_arshkon) AS mean_arshkon_rows,
          avg(n_scraped) AS mean_scraped_rows
        FROM company_counts
        WHERE n_arshkon >= 3 AND n_scraped >= 3
        """,
    )
    write_csv(
        TABLE_DIR / "company_overlap_summary.csv",
        summary_rows,
        ["companies_ge3_both", "covered_arshkon_rows", "covered_scraped_rows", "mean_arshkon_rows", "mean_scraped_rows"],
    )
    return {"company_overlap_summary": summary_rows[0] if summary_rows else {}}


def make_local_distribution_tables(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    copy_query(
        con,
        f"""
        SELECT
          source,
          min(date_posted) AS min_date_posted,
          max(date_posted) AS max_date_posted,
          count(*) AS n_default_linkedin_swe
        FROM read_parquet('{sql_quote_path(UNIFIED_PATH)}')
        WHERE {DEFAULT_FILTER}
        GROUP BY source
        ORDER BY source
        """,
        TABLE_DIR / "source_date_ranges.csv",
    )

    copy_query(
        con,
        f"""
        SELECT
          state_normalized AS state,
          count(*) AS n_all_linkedin_swe,
          sum(CASE WHEN source = 'scraped' THEN 1 ELSE 0 END) AS n_scraped_2026,
          sum(CASE WHEN source = 'kaggle_arshkon' THEN 1 ELSE 0 END) AS n_arshkon,
          sum(CASE WHEN source = 'kaggle_asaniczka' THEN 1 ELSE 0 END) AS n_asaniczka
        FROM read_parquet('{sql_quote_path(UNIFIED_PATH)}')
        WHERE {DEFAULT_FILTER}
          AND state_normalized IS NOT NULL
        GROUP BY state_normalized
        ORDER BY n_all_linkedin_swe DESC, state_normalized
        """,
        TABLE_DIR / "state_counts_our_sample.csv",
    )

    copy_query(
        con,
        f"""
        WITH known AS (
          SELECT
            source,
            CASE WHEN source = 'scraped' THEN 'scraped_2026'
                 WHEN source = 'kaggle_arshkon' THEN 'arshkon_2024'
                 ELSE source
            END AS source_group,
            trim(company_industry) AS company_industry
          FROM read_parquet('{sql_quote_path(UNIFIED_PATH)}')
          WHERE {DEFAULT_FILTER}
            AND source IN ('kaggle_arshkon', 'scraped')
            AND company_industry IS NOT NULL
            AND trim(company_industry) <> ''
        ),
        counts AS (
          SELECT source_group, company_industry, count(*) AS n
          FROM known
          GROUP BY source_group, company_industry
        ),
        denoms AS (
          SELECT source_group, sum(n) AS denom
          FROM counts
          GROUP BY source_group
        )
        SELECT
          counts.source_group,
          counts.company_industry,
          counts.n,
          round(counts.n * 1.0 / denoms.denom, 6) AS share_of_known_industry_rows
        FROM counts
        JOIN denoms USING (source_group)
        QUALIFY row_number() OVER (PARTITION BY counts.source_group ORDER BY counts.n DESC, counts.company_industry) <= 25
        ORDER BY source_group, n DESC, company_industry
        """,
        TABLE_DIR / "our_industry_distribution_top25.csv",
    )

    return {
        "source_date_ranges": read_csv_dicts(TABLE_DIR / "source_date_ranges.csv"),
        "state_counts": read_csv_dicts(TABLE_DIR / "state_counts_our_sample.csv"),
        "our_industry_top": read_csv_dicts(TABLE_DIR / "our_industry_distribution_top25.csv"),
    }


def get_url_text(url: str, timeout: int = 30) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "job-research-t07/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def load_oews_state_metadata() -> tuple[list[dict[str, str]], dict[str, str], dict[str, str]]:
    area_text = get_url_text("https://downloadt.bls.gov/pub/time.series/oe/oe.area")
    occupation_text = get_url_text("https://downloadt.bls.gov/pub/time.series/oe/oe.occupation")
    industry_text = get_url_text("https://downloadt.bls.gov/pub/time.series/oe/oe.industry")

    (RAW_DIR / "oews_occupation_metadata.tsv").write_text(occupation_text)
    (RAW_DIR / "oews_industry_metadata.tsv").write_text(industry_text)

    area_rows = list(csv.DictReader(area_text.splitlines(), delimiter="\t"))
    state_rows = []
    for row in area_rows:
        if row["areatype_code"] == "S" and row["state_code"] in STATE_ABBR_BY_FIPS:
            state_rows.append(
                {
                    "state_fips": row["state_code"],
                    "state": STATE_ABBR_BY_FIPS[row["state_code"]],
                    "area_code": row["area_code"],
                    "state_name": row["area_name"],
                }
            )

    occupation_rows = list(csv.DictReader(occupation_text.splitlines(), delimiter="\t"))
    occupations = {r["occupation_code"]: r["occupation_name"] for r in occupation_rows}

    industry_rows = list(csv.DictReader(industry_text.splitlines(), delimiter="\t"))
    industries = {r["industry_code"]: r["industry_name"] for r in industry_rows}
    write_csv(TABLE_DIR / "oews_state_metadata.csv", state_rows, ["state_fips", "state", "area_code", "state_name"])
    return state_rows, occupations, industries


def bls_api_request(series_ids: list[str], start_year: str = "2024", end_year: str = "2024") -> dict[str, float | None]:
    payload = json.dumps({"seriesid": series_ids, "startyear": start_year, "endyear": end_year}).encode("utf-8")
    request = urllib.request.Request(
        "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "job-research-t07/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    values: dict[str, float | None] = {sid: None for sid in series_ids}
    if parsed.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS API request failed: {parsed}")
    for series in parsed.get("Results", {}).get("series", []):
        sid = series["seriesID"]
        data = series.get("data", [])
        if data:
            raw_value = data[0].get("value")
            try:
                values[sid] = float(str(raw_value).replace(",", ""))
            except (TypeError, ValueError):
                values[sid] = None
    return values


def batched_bls_values(series_ids: list[str], batch_size: int = 25) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for start in range(0, len(series_ids), batch_size):
        batch = series_ids[start : start + batch_size]
        values.update(bls_api_request(batch))
        time.sleep(0.2)
    return values


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    mean_x = sum(x_vals) / len(x_vals)
    mean_y = sum(y_vals) / len(y_vals)
    ss_x = sum((x - mean_x) ** 2 for x in x_vals)
    ss_y = sum((y - mean_y) ** 2 for y in y_vals)
    if ss_x == 0 or ss_y == 0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    return cov / math.sqrt(ss_x * ss_y)


def make_oews_benchmarks() -> dict[str, object]:
    status: dict[str, object] = {
        "source_urls": [
            "https://downloadt.bls.gov/pub/time.series/oe/",
            "https://api.bls.gov/publicAPI/v2/timeseries/data/",
            "https://www.bls.gov/oes/tables.htm",
        ]
    }
    try:
        state_rows, occupations, industries = load_oews_state_metadata()
        status["requested_15_1256_in_oews_metadata"] = "151256" in occupations
        status["substitute_used_for_15_1256"] = "15-1253 Software Quality Assurance Analysts and Testers"

        cached_corr = TABLE_DIR / "state_oews_correlations.csv"
        cached_state = TABLE_DIR / "oews_state_employment.csv"
        if cached_corr.exists() and cached_state.exists():
            status["state_benchmark"] = "succeeded_cached"
            status["state_correlations"] = read_csv_dicts(cached_corr)
            status["industry_benchmark"] = "not_retrieved_after_unkeyed_bls_api_threshold"
            status["industry_limitation"] = (
                "State-level OEWS succeeded, but the later industry-specific API retrieval hit the "
                "BLS unkeyed daily request threshold. The bulk current OEWS file is 316MB, so it was "
                "not downloaded for this task."
            )
            return status

        series_lookup: dict[str, dict[str, str]] = {}
        for state in state_rows:
            for soc_label, occupation_code in SOCS_FOR_BENCHMARK.items():
                sid = f"OEUS{state['area_code']}000000{occupation_code}01"
                series_lookup[sid] = {
                    "state": state["state"],
                    "state_name": state["state_name"],
                    "soc": soc_label,
                    "occupation_code": occupation_code,
                    "series_id": sid,
                }

        values = batched_bls_values(list(series_lookup))
        oews_rows = []
        for sid, meta in series_lookup.items():
            row = dict(meta)
            row["employment"] = "" if values.get(sid) is None else int(values[sid] or 0)
            oews_rows.append(row)
        write_csv(
            TABLE_DIR / "oews_state_employment.csv",
            oews_rows,
            ["state", "state_name", "soc", "occupation_code", "series_id", "employment"],
        )

        our_state_rows = read_csv_dicts(TABLE_DIR / "state_counts_our_sample.csv")
        our_by_state = {
            r["state"]: {
                "n_all_linkedin_swe": as_int(r["n_all_linkedin_swe"]),
                "n_scraped_2026": as_int(r["n_scraped_2026"]),
                "n_arshkon": as_int(r["n_arshkon"]),
                "n_asaniczka": as_int(r["n_asaniczka"]),
            }
            for r in our_state_rows
        }

        bls_by_state: dict[str, dict[str, float]] = {}
        for row in oews_rows:
            if row["employment"] == "":
                continue
            bls_by_state.setdefault(str(row["state"]), {})[str(row["soc"])] = float(row["employment"])

        joined_rows: list[dict[str, object]] = []
        for state in sorted(STATE_ABBR_BY_FIPS.values()):
            dev = bls_by_state.get(state, {}).get("15-1252", 0.0)
            qa = bls_by_state.get(state, {}).get("15-1253_substitute_for_requested_15-1256", 0.0)
            our = our_by_state.get(state, {})
            joined_rows.append(
                {
                    "state": state,
                    "oews_15_1252_employment": int(dev),
                    "oews_15_1253_substitute_employment": int(qa),
                    "oews_combined_employment": int(dev + qa),
                    "our_all_linkedin_swe": our.get("n_all_linkedin_swe", 0),
                    "our_scraped_2026_swe": our.get("n_scraped_2026", 0),
                    "our_arshkon_swe": our.get("n_arshkon", 0),
                    "our_asaniczka_swe": our.get("n_asaniczka", 0),
                }
            )
        write_csv(
            TABLE_DIR / "state_oews_joined_counts.csv",
            joined_rows,
            [
                "state",
                "oews_15_1252_employment",
                "oews_15_1253_substitute_employment",
                "oews_combined_employment",
                "our_all_linkedin_swe",
                "our_scraped_2026_swe",
                "our_arshkon_swe",
                "our_asaniczka_swe",
            ],
        )

        correlation_rows: list[dict[str, object]] = []
        for oews_col in ["oews_15_1252_employment", "oews_combined_employment"]:
            for our_col in ["our_all_linkedin_swe", "our_scraped_2026_swe", "our_arshkon_swe"]:
                xs = [float(r[oews_col]) for r in joined_rows]
                ys = [float(r[our_col]) for r in joined_rows]
                r_val = pearson(xs, ys)
                correlation_rows.append(
                    {
                        "oews_measure": oews_col,
                        "our_measure": our_col,
                        "n_states": len(joined_rows),
                        "pearson_r": "" if r_val is None else round(r_val, 4),
                    }
                )
        write_csv(
            TABLE_DIR / "state_oews_correlations.csv",
            correlation_rows,
            ["oews_measure", "our_measure", "n_states", "pearson_r"],
        )
        status["state_benchmark"] = "succeeded"
        status["state_correlations"] = correlation_rows

        # Lightweight industry context: level-4 national industry-specific OEWS.
        industry_series: dict[str, dict[str, str]] = {}
        industry_meta_rows = list(csv.DictReader((RAW_DIR / "oews_industry_metadata.tsv").read_text().splitlines(), delimiter="\t"))
        for industry in industry_meta_rows:
            if (
                industry["industry_code"].isdigit()
                and industry["industry_code"] != "000000"
                and industry["display_level"] == "4"
                and industry["selectable"] == "T"
            ):
                for soc_label, occupation_code in SOCS_FOR_BENCHMARK.items():
                    sid = f"OEUN0000000{industry['industry_code']}{occupation_code}01"
                    industry_series[sid] = {
                        "industry_code": industry["industry_code"],
                        "industry_name": industries.get(industry["industry_code"], industry["industry_code"]),
                        "soc": soc_label,
                        "occupation_code": occupation_code,
                        "series_id": sid,
                    }
        industry_values = batched_bls_values(list(industry_series), batch_size=25)
        industry_totals: dict[str, dict[str, object]] = {}
        for sid, meta in industry_series.items():
            value = industry_values.get(sid)
            if value is None:
                continue
            key = str(meta["industry_code"])
            industry_totals.setdefault(
                key,
                {"industry_code": key, "industry_name": meta["industry_name"], "oews_swe_employment": 0.0},
            )
            industry_totals[key]["oews_swe_employment"] = float(industry_totals[key]["oews_swe_employment"]) + value
        total_emp = sum(float(r["oews_swe_employment"]) for r in industry_totals.values())
        industry_rows = []
        for row in industry_totals.values():
            emp = float(row["oews_swe_employment"])
            industry_rows.append(
                {
                    "industry_code": row["industry_code"],
                    "industry_name": row["industry_name"],
                    "oews_swe_employment": int(emp),
                    "share_of_retrieved_level4_oews_swe": round(emp / total_emp, 6) if total_emp else "",
                }
            )
        industry_rows.sort(key=lambda r: (-as_int(r["oews_swe_employment"]), str(r["industry_code"])))
        write_csv(
            TABLE_DIR / "oews_industry_distribution_top25.csv",
            industry_rows[:25],
            ["industry_code", "industry_name", "oews_swe_employment", "share_of_retrieved_level4_oews_swe"],
        )
        status["industry_benchmark"] = "succeeded_level4_context_not_directly_mapped"
        status["industry_rows_retrieved"] = len(industry_rows)
    except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        cached_corr = TABLE_DIR / "state_oews_correlations.csv"
        if cached_corr.exists():
            status["state_benchmark"] = "succeeded_state_csv_written_before_later_failure"
            status["state_correlations"] = read_csv_dicts(cached_corr)
        else:
            status["state_benchmark"] = "failed"
        status["industry_benchmark"] = "failed"
        status["error"] = str(exc)

    return status


def make_fred_context() -> dict[str, object]:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=JTU5100JOL,JTU5100HIL,JTU5100LDL"
    status: dict[str, object] = {"source_url": url}
    try:
        try:
            completed = subprocess.run(
                ["curl", "-fsSL", "--max-time", "60", url],
                check=True,
                capture_output=True,
                text=True,
            )
            csv_text = completed.stdout
        except subprocess.CalledProcessError:
            csv_text = get_url_text(url, timeout=60)
        (RAW_DIR / "fred_jolts_information.csv").write_text(csv_text)
        rows = list(csv.DictReader(csv_text.splitlines()))
        for row in rows:
            for col in ["JTU5100JOL", "JTU5100HIL", "JTU5100LDL"]:
                if row.get(col) == ".":
                    row[col] = ""

        wanted_dates = {"2024-01-01", "2024-04-01", "2026-03-01", "2026-04-01"}
        latest = next((r for r in reversed(rows) if r.get("JTU5100JOL")), None)
        context_rows = []
        for row in rows:
            if row["observation_date"] in wanted_dates:
                context_rows.append(row)
        if latest and all(r["observation_date"] != latest["observation_date"] for r in context_rows):
            context_rows.append(latest)
        context_rows.sort(key=lambda r: r["observation_date"])
        write_csv(
            TABLE_DIR / "fred_jolts_information_context.csv",
            context_rows,
            ["observation_date", "JTU5100JOL", "JTU5100HIL", "JTU5100LDL"],
        )
        status["fred_benchmark"] = "succeeded"
        status["latest_observation"] = latest
        status["context_rows"] = context_rows
    except (urllib.error.URLError, TimeoutError, subprocess.CalledProcessError) as exc:
        status["fred_benchmark"] = "failed"
        status["error"] = str(exc)
    return status


def main() -> None:
    ensure_dirs()
    con = duckdb_connection()

    summary: dict[str, object] = {}
    summary["feasibility"] = make_group_sizes_and_feasibility(con)
    summary["metro"] = make_metro_tables(con)
    summary["company"] = make_company_tables(con)
    summary["local_distributions"] = make_local_distribution_tables(con)
    summary["oews"] = make_oews_benchmarks()
    summary["fred"] = make_fred_context()

    (TABLE_DIR / "T07_summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
