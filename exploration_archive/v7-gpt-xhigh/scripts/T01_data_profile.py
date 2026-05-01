#!/usr/bin/env python
"""T01 data profile and column coverage for the SWE labor-market exploration.

The script queries Parquet through DuckDB and only materializes aggregated
results. It intentionally does not load the full unified dataset into pandas.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "unified.parquet"
TABLE_DIR = ROOT / "exploration" / "tables" / "T01"
FIGURE_DIR = ROOT / "exploration" / "figures" / "T01"
REPORT_PATH = ROOT / "exploration" / "reports" / "T01.md"

DEFAULT_WHERE = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'"
SOURCE_ORDER = ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def pct(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{100 * float(value):.{digits}f}%"


def intfmt(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{int(value):,}"


def md_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if columns is not None:
        df = df[columns].copy()
    if max_rows is not None:
        df = df.head(max_rows).copy()
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in rows))
        for idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body])


def save_csv(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    return path


def row_counts(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = con.execute(
        f"""
        SELECT
          source,
          source_platform,
          count(*) AS total_rows,
          sum(CASE WHEN is_swe THEN 1 ELSE 0 END) AS swe_rows,
          sum(CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) AS swe_adjacent_rows,
          sum(CASE WHEN is_control THEN 1 ELSE 0 END) AS control_rows,
          sum(CASE WHEN NOT coalesce(is_swe, false)
                    AND NOT coalesce(is_swe_adjacent, false)
                    AND NOT coalesce(is_control, false)
                   THEN 1 ELSE 0 END) AS other_rows,
          sum(CASE WHEN is_english THEN 1 ELSE 0 END) AS english_rows,
          sum(CASE WHEN date_flag = 'ok' THEN 1 ELSE 0 END) AS date_ok_rows
        FROM read_parquet('{DATASET.as_posix()}')
        GROUP BY source, source_platform
        ORDER BY source, source_platform
        """
    ).fetchdf()

    default = con.execute(
        f"""
        SELECT
          source,
          source_platform,
          count(*) AS total_rows,
          sum(CASE WHEN is_swe THEN 1 ELSE 0 END) AS swe_rows,
          sum(CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) AS swe_adjacent_rows,
          sum(CASE WHEN is_control THEN 1 ELSE 0 END) AS control_rows,
          sum(CASE WHEN NOT coalesce(is_swe, false)
                    AND NOT coalesce(is_swe_adjacent, false)
                    AND NOT coalesce(is_control, false)
                   THEN 1 ELSE 0 END) AS other_rows
        FROM read_parquet('{DATASET.as_posix()}')
        WHERE {DEFAULT_WHERE}
        GROUP BY source, source_platform
        ORDER BY source, source_platform
        """
    ).fetchdf()

    ranges = con.execute(
        f"""
        SELECT
          source,
          source_platform,
          count(*) AS total_rows,
          min(date_posted) AS min_date_posted,
          max(date_posted) AS max_date_posted,
          min(scrape_date) AS min_scrape_date,
          max(scrape_date) AS max_scrape_date,
          count(date_posted) AS date_posted_non_null,
          count(scrape_date) AS scrape_date_non_null
        FROM read_parquet('{DATASET.as_posix()}')
        WHERE is_english = true AND date_flag = 'ok'
        GROUP BY source, source_platform
        ORDER BY source, source_platform
        """
    ).fetchdf()

    return raw, default, ranges


def column_coverage(con: duckdb.DuckDBPyConnection, columns: list[tuple[str, str]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    subset_expr = (
        "CASE "
        "WHEN is_swe THEN 'is_swe=true' "
        "WHEN is_swe IS NULL THEN 'is_swe=null' "
        "ELSE 'is_swe=false' END"
    )

    for idx, (name, dtype) in enumerate(columns, start=1):
        col = qident(name)
        print(f"[T01] coverage {idx:03d}/{len(columns):03d}: {name}", flush=True)
        string_non_empty = (
            f"sum(CASE WHEN {col} IS NOT NULL AND trim(CAST({col} AS VARCHAR)) <> '' THEN 1 ELSE 0 END)"
            if "VARCHAR" in dtype.upper()
            else "NULL"
        )
        sql = f"""
        SELECT
          source,
          'all' AS is_swe_subset,
          count(*) AS total_rows,
          sum(CASE WHEN {col} IS NOT NULL THEN 1 ELSE 0 END) AS non_null_count,
          {string_non_empty} AS non_empty_count,
          count(DISTINCT {col}) AS distinct_count
        FROM read_parquet('{DATASET.as_posix()}')
        WHERE {DEFAULT_WHERE}
        GROUP BY source
        UNION ALL
        SELECT
          source,
          {subset_expr} AS is_swe_subset,
          count(*) AS total_rows,
          sum(CASE WHEN {col} IS NOT NULL THEN 1 ELSE 0 END) AS non_null_count,
          {string_non_empty} AS non_empty_count,
          count(DISTINCT {col}) AS distinct_count
        FROM read_parquet('{DATASET.as_posix()}')
        WHERE {DEFAULT_WHERE}
        GROUP BY source, {subset_expr}
        """
        df = con.execute(sql).fetchdf()
        df.insert(0, "column_name", name)
        df.insert(1, "column_type", dtype)
        frames.append(df)

    coverage = pd.concat(frames, ignore_index=True)
    coverage["non_null_rate"] = coverage["non_null_count"] / coverage["total_rows"]
    coverage["null_rate"] = 1 - coverage["non_null_rate"]
    coverage["non_empty_rate"] = coverage["non_empty_count"] / coverage["total_rows"]
    coverage["source"] = pd.Categorical(coverage["source"], SOURCE_ORDER, ordered=True)
    coverage = coverage.sort_values(["column_name", "source", "is_swe_subset"]).reset_index(drop=True)
    return coverage


def write_heatmap(coverage: pd.DataFrame, column_order: list[str]) -> Path:
    all_rows = coverage[coverage["is_swe_subset"] == "all"].copy()
    matrix = all_rows.pivot(index="column_name", columns="source", values="non_null_rate")
    matrix = matrix.reindex(index=column_order, columns=SOURCE_ORDER)

    height = max(12, 0.2 * len(matrix))
    fig, ax = plt.subplots(figsize=(7.5, height))
    sns.heatmap(
        matrix,
        ax=ax,
        vmin=0,
        vmax=1,
        cmap="viridis",
        cbar_kws={"label": "Non-null rate"},
        linewidths=0.1,
        linecolor="#eeeeee",
    )
    ax.set_title("T01 column coverage by LinkedIn source\n(default filters: LinkedIn, English, date_flag='ok')")
    ax.set_xlabel("Source")
    ax.set_ylabel("Column")
    ax.tick_params(axis="y", labelsize=6)
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    path = FIGURE_DIR / "T01_coverage_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def llm_extraction_distribution(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(
        f"""
        WITH counts AS (
          SELECT
            source,
            coalesce(llm_extraction_coverage, 'NULL') AS llm_extraction_coverage,
            count(*) AS rows
          FROM read_parquet('{DATASET.as_posix()}')
          WHERE {DEFAULT_WHERE} AND is_swe = true
          GROUP BY source, coalesce(llm_extraction_coverage, 'NULL')
        ),
        den AS (
          SELECT source, sum(rows) AS denominator
          FROM counts
          GROUP BY source
        )
        SELECT
          counts.source,
          counts.llm_extraction_coverage,
          counts.rows,
          den.denominator,
          counts.rows::DOUBLE / den.denominator AS share
        FROM counts
        JOIN den USING (source)
        ORDER BY counts.source, rows DESC
        """
    ).fetchdf()
    df["source"] = pd.Categorical(df["source"], SOURCE_ORDER, ordered=True)
    return df.sort_values(["source", "rows"], ascending=[True, False]).reset_index(drop=True)


def key_column_coverage(coverage: pd.DataFrame) -> pd.DataFrame:
    key_columns = [
        "description",
        "description_core_llm",
        "llm_extraction_coverage",
        "seniority_native",
        "seniority_final",
        "seniority_final_source",
        "yoe_extracted",
        "company_name_effective",
        "company_name_canonical",
        "company_industry",
        "company_size",
        "date_posted",
        "scrape_date",
        "posting_age_days",
        "metro_area",
        "is_remote",
        "is_remote_inferred",
        "search_query",
        "query_tier",
        "search_metro_name",
        "skills_raw",
        "asaniczka_skills",
        "swe_classification_llm",
        "ghost_assessment_llm",
        "yoe_min_years_llm",
    ]
    all_rows = coverage[
        (coverage["is_swe_subset"] == "all") & (coverage["column_name"].isin(key_columns))
    ].copy()
    all_rows = all_rows.sort_values(["column_name", "source"])
    return all_rows[
        [
            "column_name",
            "source",
            "total_rows",
            "non_null_count",
            "non_null_rate",
            "distinct_count",
        ]
    ]


def write_report(
    raw_counts: pd.DataFrame,
    default_counts: pd.DataFrame,
    ranges: pd.DataFrame,
    coverage: pd.DataFrame,
    high_null: pd.DataFrame,
    llm_dist: pd.DataFrame,
    heatmap_path: Path,
) -> None:
    raw_fmt = raw_counts.copy()
    default_fmt = default_counts.copy()
    ranges_fmt = ranges.copy()
    for df in [raw_fmt, default_fmt, ranges_fmt]:
        for col in df.columns:
            if col.endswith("_rows") or col == "total_rows":
                df[col] = df[col].map(intfmt)

    high_null_summary = (
        high_null.groupby("source", observed=False)
        .agg(columns_over_50pct_null=("column_name", "nunique"))
        .reset_index()
    )
    high_null_summary["columns_over_50pct_null"] = high_null_summary[
        "columns_over_50pct_null"
    ].map(intfmt)

    key_examples = high_null[
        high_null["column_name"].isin(
            [
                "description_core_llm",
                "seniority_native",
                "company_industry",
                "company_size",
                "scrape_date",
                "posting_age_days",
                "metro_area",
                "search_query",
                "asaniczka_skills",
                "swe_classification_llm",
                "ghost_assessment_llm",
            ]
        )
    ].copy()
    key_examples["null_rate"] = key_examples["null_rate"].map(pct)
    key_examples["non_null_rate"] = key_examples["non_null_rate"].map(pct)
    key_examples["total_rows"] = key_examples["total_rows"].map(intfmt)
    key_examples["non_null_count"] = key_examples["non_null_count"].map(intfmt)
    key_examples = key_examples[
        ["column_name", "source", "total_rows", "non_null_count", "non_null_rate", "null_rate"]
    ].sort_values(["column_name", "source"])

    llm_fmt = llm_dist.copy()
    llm_fmt["rows"] = llm_fmt["rows"].map(intfmt)
    llm_fmt["denominator"] = llm_fmt["denominator"].map(intfmt)
    llm_fmt["share"] = llm_fmt["share"].map(pct)

    swe_default_total = int(default_counts["swe_rows"].sum())
    total_default = int(default_counts["total_rows"].sum())
    total_raw = int(raw_counts["total_rows"].sum())
    linked_raw = int(raw_counts.loc[raw_counts["source_platform"] == "linkedin", "total_rows"].sum())

    lines = [
        "# T01. Data Profile & Column Coverage",
        "",
        "## Headline Finding",
        "",
        (
            "The current `data/unified.parquet` artifact contains "
            f"{total_raw:,} rows total and {linked_raw:,} LinkedIn rows. After the default "
            "`source_platform = 'linkedin'`, `is_english = true`, `date_flag = 'ok'` filters, "
            f"the cross-source LinkedIn analysis frame has {total_default:,} rows, including "
            f"{swe_default_total:,} SWE rows. The binding cross-period constraints are not row volume; "
            "they are source-specific field availability, especially LLM-cleaned text coverage, "
            "company metadata gaps, asaniczka's native seniority gap, and geography/search-field semantics."
        ),
        "",
        "## Methodology",
        "",
        (
            "All counts and coverage metrics were queried from `data/unified.parquet` with DuckDB via "
            "`./.venv/bin/python`. Column coverage uses the default LinkedIn/English/date-ok analysis "
            "frame unless explicitly labeled as raw artifact counts. Non-null rate is `count(column) / "
            "count(*)`; distinct count is exact `COUNT(DISTINCT column)` within each source and `is_swe` "
            "subset. The coverage CSV also includes `is_swe=true` and `is_swe=false` rows."
        ),
        "",
        "## Authoritative Row Counts",
        "",
        "Raw artifact counts, no default filters:",
        "",
        md_table(raw_fmt),
        "",
        "Default-filtered LinkedIn analysis frame:",
        "",
        md_table(default_fmt),
        "",
        "Date/scrape ranges after `is_english = true` and `date_flag = 'ok'`:",
        "",
        md_table(ranges_fmt),
        "",
        "## Column Coverage Artifacts",
        "",
        f"- Coverage CSV: `exploration/tables/T01/column_coverage_by_source_is_swe.csv`",
        f"- High-null flag CSV: `exploration/tables/T01/high_null_columns_over_50pct.csv`",
        f"- Key-column coverage CSV: `exploration/tables/T01/key_column_coverage.csv`",
        f"- Coverage heatmap: `exploration/figures/T01/{heatmap_path.name}`",
        "",
        "Columns with >50% nullness in at least one cross-period LinkedIn source:",
        "",
        md_table(high_null_summary),
        "",
        "Selected high-null examples relevant to Wave 2-3 analyses:",
        "",
        md_table(key_examples, max_rows=40),
        "",
        "## LLM Cleaned-Text Coverage",
        "",
        (
            "Subset: default-filtered LinkedIn SWE rows. Denominator is source-specific SWE rows under "
            "the default filters. `description_core_llm` should only be used where "
            "`llm_extraction_coverage = 'labeled'`."
        ),
        "",
        md_table(llm_fmt),
        "",
        "## Different Semantics Across Sources",
        "",
        "- `seniority_native`: arshkon and scraped include entry/intern labels; asaniczka has no native entry-level labels and only contributes `associate`/`mid-senior` for the T02 comparison.",
        "- `company_industry`: arshkon has companion-data industry labels and scraped LinkedIn has platform industry labels; asaniczka and scraped Indeed have no usable industry coverage. Treat arshkon-vs-scraped industry comparisons as instrument-dependent.",
        "- `company_size`: arshkon uses Kaggle companion company fields, Indeed has parsed size coverage, and asaniczka/scraped LinkedIn are effectively absent. It is not a cross-period LinkedIn control.",
        "- `date_posted`: Kaggle sources are historical snapshot dates; scraped LinkedIn `date_posted` is sparse by design, while `scrape_date` defines the current observation window. Do not interpret date fields as a continuous unified event-time panel.",
        "- `search_query` and `search_metro_*`: for scraped rows they are scraper search metadata; for asaniczka, `search_query` comes from `search_position`; for arshkon it is absent. Use `metro_area` for posting geography, not search metadata.",
        "- `is_remote`: 2024 source remote flags are effectively an artifact. Use remote fields only as source-specific descriptors unless a downstream task explicitly validates them.",
        "- `skills_raw` / `asaniczka_skills`: arshkon/scraped skills fields and asaniczka structured skills come from different source instruments; compare them only as validation or source-specific artifacts, not as one pooled variable.",
        "- LLM columns (`description_core_llm`, `swe_classification_llm`, `ghost_assessment_llm`, `yoe_min_years_llm`) are coverage-managed sample outputs, not complete source fields. Always state coverage and denominator.",
        "",
        "## Constraint Map For Wave 2-3",
        "",
        md_table(
            pd.DataFrame(
                [
                    {
                        "category": "Text",
                        "binding_constraint": "Cleaned text is LLM-only; scraped LinkedIn SWE labeled coverage is materially lower than historical sources.",
                        "severity": "High",
                    },
                    {
                        "category": "Seniority",
                        "binding_constraint": "Asaniczka has no native entry labels; `seniority_final` entry cells are small and partly LLM-derived.",
                        "severity": "High",
                    },
                    {
                        "category": "Geography",
                        "binding_constraint": "`metro_area` is the analysis field; search-metro fields exist only for scraped/search sources and multi-location rows can be unresolved.",
                        "severity": "Moderate",
                    },
                    {
                        "category": "Company",
                        "binding_constraint": "Company names/canonical names are strong, but industry and size are not consistently available across sources.",
                        "severity": "Moderate-High",
                    },
                    {
                        "category": "Requirements",
                        "binding_constraint": "YOE is broadly available but extracted from text; ghost/LLM requirement fields are coverage-limited and must be denominator-reported.",
                        "severity": "Moderate",
                    },
                ]
            )
        ),
        "",
        "## Surprises & Unexpected Patterns",
        "",
        "- Arshkon has near-complete LLM extraction coverage for SWE rows, while scraped LinkedIn has many `not_selected` SWE rows. This makes unbalanced cleaned-text comparisons risky even when raw descriptions are complete.",
        "- The raw artifact includes a large non-SWE/other reservoir in every source; Wave 2 tasks should not assume the file is already only SWE/adjacent/control.",
        "- `seniority_final` is complete by construction, but entry-level effective counts are much smaller than overall SWE counts, so the seniority problem is power and label construction rather than missingness.",
        "- Some fields with tempting face validity, especially remote status, company size, and search metadata, are source-instrument fields rather than comparable labor-market measures.",
        "",
        "## Action Items For Downstream Agents",
        "",
        "- Use `seniority_final` for primary seniority, but validate junior claims with the T30 panel and YOE proxies.",
        "- For text-sensitive work, restrict to `llm_extraction_coverage = 'labeled'` and report the source-specific denominator; do not silently backfill raw text.",
        "- For company controls, rely on `company_name_effective` / `company_name_canonical`; avoid pooled industry or size controls unless source-specific.",
        "- For geography, use `metro_area` and report unresolved/multi-location exclusions.",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    columns = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{DATASET.as_posix()}')").fetchall()
    columns = [(row[0], row[1]) for row in columns]

    raw_counts, default_counts, ranges = row_counts(con)
    save_csv(raw_counts, "row_counts_raw_by_source_platform.csv")
    save_csv(default_counts, "row_counts_default_linkedin_by_source.csv")
    save_csv(ranges, "date_ranges_by_source_platform_quality_filtered.csv")

    coverage = column_coverage(con, columns)
    save_csv(coverage, "column_coverage_by_source_is_swe.csv")

    coverage_all = coverage[coverage["is_swe_subset"] == "all"].copy()
    high_null = coverage_all[coverage_all["null_rate"] > 0.5].copy()
    high_null = high_null.sort_values(["source", "null_rate", "column_name"], ascending=[True, False, True])
    save_csv(high_null, "high_null_columns_over_50pct.csv")

    key_cov = key_column_coverage(coverage)
    save_csv(key_cov, "key_column_coverage.csv")

    heatmap_path = write_heatmap(coverage, [name for name, _ in columns])

    llm_dist = llm_extraction_distribution(con)
    save_csv(llm_dist, "llm_extraction_coverage_swe_rows.csv")

    write_report(raw_counts, default_counts, ranges, coverage, high_null, llm_dist, heatmap_path)
    print(f"[T01] wrote {REPORT_PATH.relative_to(ROOT)}")
    print(f"[T01] wrote {heatmap_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
