#!/usr/bin/env python
from __future__ import annotations

import math
import re
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from scipy.stats import pearsonr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
REPORT_DIR = ROOT / "exploration" / "reports"
TABLE_DIR = ROOT / "exploration" / "tables" / "T07"
FIG_DIR = ROOT / "exploration" / "figures" / "T07"
SCRIPT_DIR = ROOT / "exploration" / "scripts"

DEFAULT_FILTER = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'"
SWE_FILTER = f"{DEFAULT_FILTER} AND is_swe"

BLS_STATE_INDEX_URL = "https://www.bls.gov/oes/2024/may/oessrcst.htm"
FRED_JOLTS_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=JTU5100JOL"

STATE_ABBR_TO_NAME = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}

STATE_NAME_TO_ABBR = {v: k for k, v in STATE_ABBR_TO_NAME.items()}


def ensure_dirs() -> None:
    for path in [REPORT_DIR, TABLE_DIR, FIG_DIR, SCRIPT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def qdf(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def qone(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchone()[0]


def save_csv(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    return path


def save_fig(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def clean_label(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def parse_int_cell(value: str) -> int:
    matches = re.findall(r"[0-9][0-9,]*", str(value))
    if not matches:
        raise ValueError(f"Could not parse integer from {value!r}")
    return int(matches[-1].replace(",", ""))


def cohens_d_from_summary(n1: int, n2: int, sd1: float, sd2: float) -> float:
    if min(n1, n2) < 2 or sd1 <= 0 or sd2 <= 0:
        return float("nan")
    pooled = ((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2)
    if pooled <= 0:
        return float("nan")
    return math.sqrt(pooled)


def power_binary_h(n1: int, n2: int, alpha: float = 0.05, power: float = 0.8) -> float:
    from statsmodels.stats.power import NormalIndPower

    return float(
        NormalIndPower().solve_power(
            nobs1=n1,
            ratio=n2 / n1,
            alpha=alpha,
            power=power,
            alternative="two-sided",
        )
    )


def power_continuous_d(n1: int, n2: int, alpha: float = 0.05, power: float = 0.8) -> float:
    from statsmodels.stats.power import TTestIndPower

    return float(
        TTestIndPower().solve_power(
            nobs1=n1,
            ratio=n2 / n1,
            alpha=alpha,
            power=power,
            alternative="two-sided",
        )
    )


def verdict_from_mde(n1: int, n2: int, mde_binary: float, mde_continuous: float) -> str:
    min_n = min(n1, n2)
    worst = max(mde_binary, mde_continuous)
    if min_n >= 1000 and worst <= 0.20:
        return "well-powered"
    if min_n >= 100 and worst <= 0.35:
        return "marginal"
    return "underpowered"


def query_current_counts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return qdf(
        con,
        f"""
        SELECT
          COUNT(*) AS total_rows,
          COUNT_IF(source = 'kaggle_arshkon') AS kaggle_arshkon,
          COUNT_IF(source = 'kaggle_asaniczka') AS kaggle_asaniczka,
          COUNT_IF(source = 'scraped') AS scraped,
          COUNT_IF(source_platform = 'linkedin') AS linkedin_rows,
          COUNT_IF(source_platform = 'indeed') AS indeed_rows,
          COUNT_IF(is_swe) AS swe_rows
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {DEFAULT_FILTER}
        """,
    )


def query_group_sizes(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows = []
    source_labels = [
        ("kaggle_arshkon", "arshkon"),
        ("kaggle_asaniczka", "asaniczka"),
        ("scraped", "scraped"),
    ]
    for source, label in source_labels:
        all_swe = qone(
            con,
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {SWE_FILTER} AND source = '{source}'
            """,
        )
        entry = qone(
            con,
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {SWE_FILTER} AND source = '{source}' AND seniority_final = 'entry'
            """,
        )
        mid_senior = qone(
            con,
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {SWE_FILTER} AND source = '{source}' AND seniority_final = 'mid-senior'
            """,
        )
        senior = qone(
            con,
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {SWE_FILTER} AND source = '{source}' AND seniority_3level = 'senior'
            """,
        )
        rows.extend(
            [
                {"comparison": f"{label} all SWE", "source": source, "group": "all_swe", "n": int(all_swe)},
                {"comparison": f"{label} entry", "source": source, "group": "entry", "n": int(entry)},
                {
                    "comparison": f"{label} mid-senior",
                    "source": source,
                    "group": "mid-senior",
                    "n": int(mid_senior),
                },
                {"comparison": f"{label} senior", "source": source, "group": "senior", "n": int(senior)},
            ]
        )

    pooled_2024 = qone(
        con,
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {SWE_FILTER} AND source IN ('kaggle_arshkon', 'kaggle_asaniczka')
        """,
    )
    pooled_entry = qone(
        con,
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {SWE_FILTER} AND source IN ('kaggle_arshkon', 'kaggle_asaniczka') AND seniority_final = 'entry'
        """,
    )
    pooled_mid = qone(
        con,
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {SWE_FILTER} AND source IN ('kaggle_arshkon', 'kaggle_asaniczka') AND seniority_final = 'mid-senior'
        """,
    )
    pooled_senior = qone(
        con,
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {SWE_FILTER} AND source IN ('kaggle_arshkon', 'kaggle_asaniczka') AND seniority_3level = 'senior'
        """,
    )
    rows.extend(
        [
            {"comparison": "pooled 2024 all SWE", "source": "pooled_2024", "group": "all_swe", "n": int(pooled_2024)},
            {"comparison": "pooled 2024 entry", "source": "pooled_2024", "group": "entry", "n": int(pooled_entry)},
            {
                "comparison": "pooled 2024 mid-senior",
                "source": "pooled_2024",
                "group": "mid-senior",
                "n": int(pooled_mid),
            },
            {"comparison": "pooled 2024 senior", "source": "pooled_2024", "group": "senior", "n": int(pooled_senior)},
        ]
    )

    return pd.DataFrame(rows)


def build_feasibility_table(group_sizes: pd.DataFrame) -> pd.DataFrame:
    lookup = {
        row["comparison"]: int(row["n"])
        for _, row in group_sizes.iterrows()
    }
    comps = [
        ("entry arshkon vs scraped", "arshkon entry", "scraped entry"),
        ("mid-senior arshkon vs scraped", "arshkon mid-senior", "scraped mid-senior"),
        ("senior arshkon vs scraped", "arshkon senior", "scraped senior"),
        ("all SWE arshkon vs scraped", "arshkon all SWE", "scraped all SWE"),
        ("pooled 2024 vs scraped", "pooled 2024 all SWE", "scraped all SWE"),
    ]
    rows = []
    for label, left, right in comps:
        n1 = lookup[left]
        n2 = lookup[right]
        mde_binary = power_binary_h(n1, n2)
        mde_cont = power_continuous_d(n1, n2)
        verdict = verdict_from_mde(n1, n2, mde_binary, mde_cont)
        rows.append(
            {
                "analysis_type": label,
                "comparison": f"{left} vs {right}",
                "n_group1": n1,
                "n_group2": n2,
                "MDE_binary": mde_binary,
                "MDE_continuous": mde_cont,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def query_metro_feasibility(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    excluded = qone(
        con,
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {SWE_FILTER} AND is_multi_location = true
        """,
    )

    period_counts = qdf(
        con,
        f"""
        SELECT
          period,
          metro_area,
          COUNT(*) AS n
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {SWE_FILTER}
          AND is_multi_location = false
          AND metro_area IS NOT NULL
          AND metro_area <> ''
        GROUP BY 1,2
        ORDER BY 1,3 DESC,2
        """,
    )

    summary = (
        period_counts.assign(
            ge50=lambda df: df["n"] >= 50,
            ge100=lambda df: df["n"] >= 100,
        )
        .groupby("period", as_index=False)
        .agg(
            metros_any=("metro_area", "nunique"),
            metros_ge50=("ge50", "sum"),
            metros_ge100=("ge100", "sum"),
        )
    )
    summary.insert(0, "excluded_multi_location_swe_rows", int(excluded))

    overlap_source = period_counts[period_counts["period"].isin(["2024-04", "2026-03", "2026-04"])].copy()
    overlap_source["bucket"] = overlap_source["period"].replace({"2026-03": "2026_combined", "2026-04": "2026_combined"})
    overlap_source = (
        overlap_source.groupby(["bucket", "metro_area"], as_index=False)["n"].sum().rename(columns={"bucket": "period"})
    )
    arsh = period_counts[period_counts["period"] == "2024-04"][["metro_area", "n"]].rename(columns={"n": "n_2024_04"})
    scrape = (
        overlap_source[overlap_source["period"] == "2026_combined"][["metro_area", "n"]]
        .rename(columns={"n": "n_2026_combined"})
    )
    overlap = arsh.merge(scrape, on="metro_area", how="inner")
    overlap["ge50_both"] = (overlap["n_2024_04"] >= 50) & (overlap["n_2026_combined"] >= 50)
    overlap["ge100_both"] = (overlap["n_2024_04"] >= 100) & (overlap["n_2026_combined"] >= 100)
    overlap = overlap.sort_values(["ge100_both", "ge50_both", "n_2024_04", "n_2026_combined"], ascending=False)

    overlap_summary = pd.DataFrame(
        [
            {
                "comparison": "arshkon 2024-04 vs scraped 2026-03+04",
                "metros_ge50_both": int(overlap["ge50_both"].sum()),
                "metros_ge100_both": int(overlap["ge100_both"].sum()),
            }
        ]
    )
    return summary, overlap_summary, overlap


def query_company_overlap(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    counts = qdf(
        con,
        f"""
        WITH company_counts AS (
          SELECT
            source,
            company_name_canonical,
            COUNT(*) AS n
          FROM read_parquet('{DATA.as_posix()}')
          WHERE {SWE_FILTER}
            AND company_name_canonical IS NOT NULL
            AND company_name_canonical <> ''
          GROUP BY 1,2
        )
        SELECT
          'arshkon vs scraped' AS comparison,
          COUNT(*) FILTER (
            WHERE a.n >= 3 AND s.n >= 3
          ) AS companies_ge3_both,
          COUNT(*) FILTER (
            WHERE a.n >= 5 AND s.n >= 5
          ) AS companies_ge5_both
        FROM company_counts a
        JOIN company_counts s
          ON a.company_name_canonical = s.company_name_canonical
         AND a.source = 'kaggle_arshkon'
         AND s.source = 'scraped'

        UNION ALL

        SELECT
          'pooled 2024 vs scraped' AS comparison,
          COUNT(*) FILTER (
            WHERE p.n >= 3 AND s.n >= 3
          ) AS companies_ge3_both,
          COUNT(*) FILTER (
            WHERE p.n >= 5 AND s.n >= 5
          ) AS companies_ge5_both
        FROM (
          SELECT company_name_canonical, COUNT(*) AS n
          FROM read_parquet('{DATA.as_posix()}')
          WHERE {SWE_FILTER}
            AND source IN ('kaggle_arshkon', 'kaggle_asaniczka')
            AND company_name_canonical IS NOT NULL
            AND company_name_canonical <> ''
          GROUP BY 1
        ) p
        JOIN (
          SELECT company_name_canonical, COUNT(*) AS n
          FROM read_parquet('{DATA.as_posix()}')
          WHERE {SWE_FILTER}
            AND source = 'scraped'
            AND company_name_canonical IS NOT NULL
            AND company_name_canonical <> ''
          GROUP BY 1
        ) s
          ON p.company_name_canonical = s.company_name_canonical
        """,
    )
    return counts


def query_state_counts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return qdf(
        con,
        f"""
        SELECT
          source,
          period,
          state_normalized AS state_abbr,
          COUNT(*) AS swe_postings
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {SWE_FILTER}
          AND state_normalized IS NOT NULL
          AND state_normalized <> ''
        GROUP BY 1,2,3
        ORDER BY 1,2,4 DESC,3
        """,
    )


def query_industry_profile(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return qdf(
        con,
        f"""
        SELECT
          source,
          company_industry,
          COUNT(*) AS n
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {SWE_FILTER}
          AND source IN ('kaggle_arshkon', 'scraped')
          AND company_industry IS NOT NULL
          AND company_industry <> ''
        GROUP BY 1,2
        ORDER BY 1,3 DESC,2
        """,
    )


def fetch_bls_state_benchmark() -> pd.DataFrame:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page(viewport={"width": 1400, "height": 2000})
        page.goto(BLS_STATE_INDEX_URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2000)
        links = page.eval_on_selector_all(
            "a",
            """
            els => els
              .map(a => ({text: a.textContent.trim(), href: a.href}))
              .filter(x => x.href.includes('data.bls.gov/oes/#/area/') && x.text)
            """,
        )
        state_links = []
        for item in links:
            name = clean_label(item["text"])
            if name in STATE_NAME_TO_ABBR:
                state_links.append(item)
        state_links = sorted(state_links, key=lambda x: STATE_NAME_TO_ABBR[clean_label(x["text"])])

        rows = []
        for item in state_links:
            state_name = clean_label(item["text"])
            abbr = STATE_NAME_TO_ABBR[state_name]
            href = item["href"]
            area_code_match = re.search(r"/area/(\d+)", href)
            area_code = area_code_match.group(1) if area_code_match else None
            page.goto(href, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(2000)
            page.wait_for_function("document.querySelectorAll('table').length >= 4", timeout=60000)
            data = page.evaluate(
                """
                () => {
                  const tables = document.querySelectorAll('table');
                  const rows = [];
                  for (const table of tables) {
                    for (const tr of table.querySelectorAll('tr')) {
                      const cells = Array.from(tr.querySelectorAll('th,td')).map(el => el.innerText.trim());
                      if (cells.length >= 2) {
                        rows.push(cells);
                      }
                    }
                  }
                  return rows;
                }
                """
            )
            dev = None
            for cells in data:
                occ = cells[0]
                emp = cells[1]
                if "Software Developers" in occ and "15-1252" in occ:
                    dev = emp
            if dev is None:
                raise RuntimeError(f"Failed to parse BLS state rows for {state_name}: {dev=}")
            rows.append(
                {
                    "state_abbr": abbr,
                    "state_name": state_name,
                    "area_code": area_code,
                    "software_developers_151252": parse_int_cell(dev),
                }
            )
        browser.close()
    return pd.DataFrame(rows)


def fetch_fred_jolts() -> pd.DataFrame:
    resp = requests.get(FRED_JOLTS_URL, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(pd.io.common.StringIO(resp.text))
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df["JTU5100JOL"] = pd.to_numeric(df["JTU5100JOL"], errors="coerce")
    return df


def make_state_figure(state_bls: pd.DataFrame, our_state_counts: pd.DataFrame, output: Path) -> pd.DataFrame:
    periods = [
        ("2024-04", "kaggle_arshkon"),
        ("2026-03", "scraped"),
        ("2026-04", "scraped"),
    ]
    period_labels = {
        "2024-04": "arshkon 2024-04",
        "2026-03": "scraped 2026-03",
        "2026-04": "scraped 2026-04",
    }

    our_rows = []
    for period, source in periods:
        subset = our_state_counts[(our_state_counts["period"] == period) & (our_state_counts["source"] == source)].copy()
        total = subset["swe_postings"].sum()
        subset["state_name"] = subset["state_abbr"].map(STATE_ABBR_TO_NAME)
        subset["share"] = subset["swe_postings"] / total
        merged = subset.merge(
            state_bls[["state_abbr", "software_developers_151252"]],
            on="state_abbr",
            how="inner",
        )
        merged["bls_share"] = merged["software_developers_151252"] / state_bls["software_developers_151252"].sum()
        r_raw = pearsonr(merged["swe_postings"], merged["software_developers_151252"])[0]
        r_share = pearsonr(merged["share"], merged["bls_share"])[0]
        our_rows.append(
            {
                "period": period,
                "source": source,
                "pearson_r_raw": r_raw,
                "pearson_r_share": r_share,
                "n_states": len(merged),
            }
        )

    summary = pd.DataFrame(our_rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharex=True, sharey=True)
    for ax, (period, source) in zip(axes, periods):
        subset = our_state_counts[(our_state_counts["period"] == period) & (our_state_counts["source"] == source)].copy()
        total = subset["swe_postings"].sum()
        subset["share"] = subset["swe_postings"] / total
        merged = subset.merge(state_bls[["state_abbr", "software_developers_151252"]], on="state_abbr", how="inner")
        merged["bls_share"] = merged["software_developers_151252"] / state_bls["software_developers_151252"].sum()
        r_share = pearsonr(merged["share"], merged["bls_share"])[0]
        ax.scatter(merged["bls_share"], merged["share"], s=28, alpha=0.8)
        lim = max(merged["bls_share"].max(), merged["share"].max()) * 1.05
        ax.plot([0, lim], [0, lim], color="0.4", lw=1, ls="--")
        ax.set_title(f"{period_labels[period]}\nr={r_share:.2f}")
        ax.set_xlabel("BLS share")
        ax.set_ylabel("Our SWE share")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    fig.suptitle("State geographic alignment vs BLS software benchmark")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return summary


def make_jolts_figure(jolts: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.plot(jolts["observation_date"], jolts["JTU5100JOL"], color="#1f77b4", lw=2)
    ax.axvspan(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31"), color="#7f8c8d", alpha=0.12, label="asaniczka 2024-01")
    ax.axvspan(pd.Timestamp("2024-04-01"), pd.Timestamp("2024-04-30"), color="#f39c12", alpha=0.12, label="arshkon 2024-04")
    ax.axvspan(pd.Timestamp("2026-03-01"), pd.Timestamp("2026-04-30"), color="#27ae60", alpha=0.12, label="scraped 2026-03-04")
    ax.set_title("FRED JOLTS information-sector job openings")
    ax.set_ylabel("Openings")
    ax.set_xlabel("Month")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    con = duckdb.connect()

    current_counts = query_current_counts(con)
    save_csv(current_counts, "T07_current_counts.csv")

    group_sizes = query_group_sizes(con)
    save_csv(group_sizes, "T07_group_sizes.csv")

    feasibility = build_feasibility_table(group_sizes)
    feasibility["MDE_binary"] = feasibility["MDE_binary"].round(3)
    feasibility["MDE_continuous"] = feasibility["MDE_continuous"].round(3)
    save_csv(feasibility, "T07_feasibility_table.csv")

    metro_summary, metro_overlap_summary, metro_overlap = query_metro_feasibility(con)
    save_csv(metro_summary, "T07_metro_feasibility.csv")
    save_csv(metro_overlap_summary, "T07_metro_overlap_summary.csv")
    save_csv(metro_overlap, "T07_metro_overlap_detail.csv")

    company_overlap = query_company_overlap(con)
    save_csv(company_overlap, "T07_company_overlap.csv")

    state_counts = query_state_counts(con)
    save_csv(state_counts, "T07_state_counts.csv")

    industry_profile = query_industry_profile(con)
    save_csv(industry_profile, "T07_our_industry_profile.csv")

    bls_state = fetch_bls_state_benchmark()
    save_csv(bls_state, "T07_bls_state_benchmark.csv")

    state_summary = make_state_figure(
        bls_state,
        state_counts,
        FIG_DIR / "T07_state_alignment.png",
    )
    save_csv(state_summary, "T07_state_alignment_summary.csv")

    jolts = fetch_fred_jolts()
    jolts.to_csv(TABLE_DIR / "T07_jolts_information_sector.csv", index=False)
    make_jolts_figure(jolts, FIG_DIR / "T07_jolts_information_sector.png")

    # Small supplementary table: top industries in our LinkedIn SWE sample.
    our_industry = (
        industry_profile.groupby("company_industry", as_index=False)["n"].sum()
        .sort_values("n", ascending=False)
        .assign(share=lambda df: df["n"] / df["n"].sum())
        .head(20)
    )
    save_csv(our_industry, "T07_our_top_industries.csv")

    print("T07 complete")
    print(current_counts.to_string(index=False))
    print(feasibility.to_string(index=False))
    print(state_summary.to_string(index=False))
    print(company_overlap.to_string(index=False))


if __name__ == "__main__":
    main()
