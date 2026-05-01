"""
T01 — Data profile and column coverage.

Produces:
- exploration/tables/T01/row_counts.csv               (by source x platform x SWE tier)
- exploration/tables/T01/row_counts_aggregator.csv    (aggregator-stratified)
- exploration/tables/T01/column_coverage.csv          (non-null rate + distinct count per col x source x is_swe)
- exploration/tables/T01/column_coverage_flagged.csv  (>50% null for any source used in cross-period)
- exploration/tables/T01/llm_extraction_coverage.csv  (description_core_llm coverage by source, SWE)
- exploration/tables/T01/semantic_differences.csv     (columns with different semantics by source)
- exploration/figures/T01/coverage_heatmap_all.png    (columns x sources, all rows)
- exploration/figures/T01/coverage_heatmap_swe.png    (columns x sources, SWE subset)
- exploration/figures/T01/coverage_heatmap_swe_linkedin.png (SWE + LinkedIn subset)

Memory-safe: uses DuckDB aggregations, never loads full parquet into pandas.
"""
from __future__ import annotations

from pathlib import Path
import duckdb
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = str(ROOT / "data" / "unified.parquet")
TABLES = ROOT / "exploration" / "tables" / "T01"
FIGS = ROOT / "exploration" / "figures" / "T01"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

SOURCES = ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]
# Columns to use in heatmap — all columns (we'll trim VARCHAR/bool comprehensively).
# Some columns are expected to be source-specific (e.g., arshkon-only).


def con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


def get_columns() -> list[tuple[str, str]]:
    c = con()
    df = c.sql(f"DESCRIBE SELECT * FROM '{DATA}'").df()
    return list(zip(df["column_name"].tolist(), df["column_type"].tolist()))


# ---------------- 1. Row counts ---------------- #

def row_counts() -> None:
    c = con()
    # Row counts by source, source_platform, SWE/adjacent/control.
    rows = c.sql(
        f"""
        SELECT
            source,
            source_platform,
            count(*) AS n_total,
            sum(CASE WHEN is_swe THEN 1 ELSE 0 END) AS n_swe,
            sum(CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) AS n_adjacent,
            sum(CASE WHEN is_control THEN 1 ELSE 0 END) AS n_control,
            sum(CASE WHEN NOT is_swe AND NOT is_swe_adjacent AND NOT is_control THEN 1 ELSE 0 END) AS n_other,
            sum(CASE WHEN source_platform='linkedin' AND is_english AND date_flag='ok' THEN 1 ELSE 0 END) AS n_default_filters,
            sum(CASE WHEN source_platform='linkedin' AND is_english AND date_flag='ok' AND is_swe THEN 1 ELSE 0 END) AS n_swe_default_filters,
            sum(CASE WHEN is_aggregator THEN 1 ELSE 0 END) AS n_aggregator,
            sum(CASE WHEN is_aggregator AND is_swe THEN 1 ELSE 0 END) AS n_swe_aggregator
        FROM '{DATA}'
        GROUP BY source, source_platform
        ORDER BY source, source_platform
        """
    ).df()
    rows.to_csv(TABLES / "row_counts.csv", index=False)
    print("Row counts by source x platform:")
    print(rows.to_string())

    # Aggregator-stratified for each source / SWE tier
    agg_rows = c.sql(
        f"""
        SELECT
            source,
            source_platform,
            is_aggregator,
            count(*) AS n_total,
            sum(CASE WHEN is_swe THEN 1 ELSE 0 END) AS n_swe,
            sum(CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) AS n_adjacent,
            sum(CASE WHEN is_control THEN 1 ELSE 0 END) AS n_control
        FROM '{DATA}'
        GROUP BY source, source_platform, is_aggregator
        ORDER BY source, source_platform, is_aggregator
        """
    ).df()
    agg_rows.to_csv(TABLES / "row_counts_aggregator.csv", index=False)

    # Period info
    periods = c.sql(
        f"""
        SELECT source, source_platform, period, count(*) AS n,
               min(date_posted) AS min_date, max(date_posted) AS max_date,
               min(scrape_date) AS min_scrape, max(scrape_date) AS max_scrape
        FROM '{DATA}'
        WHERE source_platform = 'linkedin'
        GROUP BY source, source_platform, period
        ORDER BY source, period
        """
    ).df()
    periods.to_csv(TABLES / "period_ranges.csv", index=False)


# ---------------- 2. Column coverage ---------------- #

def column_coverage(cols: list[tuple[str, str]]) -> pd.DataFrame:
    """Compute non-null rate and distinct count by (source, is_swe_subset, column)."""
    c = con()

    # Subsets: (label, filter_clause)
    subsets: dict[str, str] = {
        "all": "TRUE",
        "swe": "is_swe",
        "swe_linkedin_default": "is_swe AND source_platform='linkedin' AND is_english AND date_flag='ok'",
    }

    records: list[dict] = []

    for col, coltype in cols:
        # Skip id-level columns in the null-check (they're all non-null by construction)
        for source in SOURCES:
            for subset_label, subset_filter in subsets.items():
                where = f"source = '{source}' AND {subset_filter}"

                is_varchar_like = coltype.upper().startswith("VARCHAR")
                # Consider empty strings as null for VARCHAR columns for coverage purposes
                null_expr = f"{col} IS NULL" + (f" OR {col} = ''" if is_varchar_like else "")

                distinct_expr = f"count(DISTINCT {col})"

                q = f"""
                SELECT
                    count(*) AS n_rows,
                    sum(CASE WHEN {null_expr} THEN 1 ELSE 0 END) AS n_null,
                    {distinct_expr} AS n_distinct
                FROM '{DATA}'
                WHERE {where}
                """
                try:
                    row = c.sql(q).fetchone()
                except Exception as e:
                    row = (0, 0, 0)
                    print(f"[warn] column {col}: {e}")
                n_rows, n_null, n_distinct = row
                non_null_rate = 1.0 - (n_null / n_rows) if n_rows else 0.0
                records.append(
                    {
                        "column": col,
                        "column_type": coltype,
                        "source": source,
                        "subset": subset_label,
                        "n_rows": n_rows,
                        "n_null": n_null,
                        "non_null_rate": round(non_null_rate, 6),
                        "n_distinct": n_distinct,
                    }
                )

    df = pd.DataFrame(records)
    df.to_csv(TABLES / "column_coverage.csv", index=False)
    return df


def flag_coverage_gaps(cov: pd.DataFrame) -> None:
    """Flag columns with >50% null for any source used in cross-period comparisons."""
    # Focus on the SWE LinkedIn default-filter subset (the real working sample).
    sub = cov[cov["subset"] == "swe_linkedin_default"].copy()
    sub = sub.pivot(index="column", columns="source", values="non_null_rate").fillna(0.0)

    # For each column, flag if any source has <50% non-null rate (i.e., >50% null).
    flagged = sub[(sub < 0.5).any(axis=1)].copy()
    flagged["min_non_null_rate"] = flagged.min(axis=1)
    flagged.sort_values("min_non_null_rate", inplace=True)
    flagged.to_csv(TABLES / "column_coverage_flagged.csv")
    print(f"\nFlagged {len(flagged)} columns with <50% non-null in at least one source (SWE LinkedIn default subset)")


def coverage_heatmap(cov: pd.DataFrame, subset_label: str, filename: str, title_suffix: str) -> None:
    """Heatmap: columns (rows) vs sources (cols), colored by non-null rate."""
    sub = cov[cov["subset"] == subset_label].copy()
    # Pivot
    mat = sub.pivot(index="column", columns="source", values="non_null_rate").fillna(0.0)
    # Order columns: arshkon, asaniczka, scraped
    mat = mat[[s for s in SOURCES if s in mat.columns]]
    # Order rows by total non-null rate (descending), so filled columns sit at the top.
    mat["__avg"] = mat.mean(axis=1)
    mat = mat.sort_values("__avg", ascending=False)
    mat.drop(columns="__avg", inplace=True)

    n_rows, n_cols = mat.shape
    fig_h = max(6, n_rows * 0.14)
    fig, ax = plt.subplots(figsize=(4.5, fig_h))
    im = ax.imshow(mat.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(mat.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(mat.index, fontsize=6)
    ax.set_title(f"Column non-null rate by source\n({title_suffix})", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("non-null rate", fontsize=8)
    # Light gridlines between rows
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.tight_layout()
    plt.savefig(FIGS / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------- 3. LLM extraction coverage ---------------- #

def llm_extraction_coverage() -> None:
    c = con()
    out = c.sql(
        f"""
        SELECT
            source,
            source_platform,
            llm_extraction_coverage,
            count(*) AS n,
            sum(CASE WHEN is_swe THEN 1 ELSE 0 END) AS n_swe
        FROM '{DATA}'
        WHERE source_platform = 'linkedin' AND is_english AND date_flag='ok'
        GROUP BY source, source_platform, llm_extraction_coverage
        ORDER BY source, llm_extraction_coverage
        """
    ).df()
    out.to_csv(TABLES / "llm_extraction_coverage.csv", index=False)
    print("\nllm_extraction_coverage by source (LinkedIn / English / date ok):")
    print(out.to_string())

    out2 = c.sql(
        f"""
        SELECT
            source,
            source_platform,
            llm_classification_coverage,
            count(*) AS n,
            sum(CASE WHEN is_swe THEN 1 ELSE 0 END) AS n_swe
        FROM '{DATA}'
        WHERE source_platform = 'linkedin' AND is_english AND date_flag='ok'
        GROUP BY source, source_platform, llm_classification_coverage
        ORDER BY source, llm_classification_coverage
        """
    ).df()
    out2.to_csv(TABLES / "llm_classification_coverage.csv", index=False)
    print("\nllm_classification_coverage by source:")
    print(out2.to_string())

    # SWE-specific LLM coverage (descriptive)
    swe_ext = c.sql(
        f"""
        SELECT
            source,
            source_platform,
            llm_extraction_coverage,
            count(*) AS n
        FROM '{DATA}'
        WHERE source_platform = 'linkedin' AND is_english AND date_flag='ok' AND is_swe
        GROUP BY source, source_platform, llm_extraction_coverage
        ORDER BY source, llm_extraction_coverage
        """
    ).df()
    swe_ext.to_csv(TABLES / "llm_extraction_coverage_swe.csv", index=False)
    print("\nllm_extraction_coverage SWE-only:")
    print(swe_ext.to_string())

    swe_clf = c.sql(
        f"""
        SELECT
            source,
            source_platform,
            llm_classification_coverage,
            count(*) AS n
        FROM '{DATA}'
        WHERE source_platform = 'linkedin' AND is_english AND date_flag='ok' AND is_swe
        GROUP BY source, source_platform, llm_classification_coverage
        ORDER BY source, llm_classification_coverage
        """
    ).df()
    swe_clf.to_csv(TABLES / "llm_classification_coverage_swe.csv", index=False)


# ---------------- 4. Semantic differences across sources ---------------- #

def semantic_differences() -> None:
    """Look at value-frequency differences that hint at different semantics across sources."""
    c = con()
    # company_industry: check multi-label (compound) presence
    ci = c.sql(
        f"""
        SELECT
            source,
            count(*) AS n_total,
            sum(CASE WHEN company_industry IS NULL OR company_industry='' THEN 0 ELSE 1 END) AS n_with,
            count(DISTINCT company_industry) AS n_distinct,
            sum(CASE WHEN company_industry LIKE '%,%' OR company_industry LIKE '% and %' THEN 1 ELSE 0 END) AS n_compound
        FROM '{DATA}'
        WHERE source_platform='linkedin'
        GROUP BY source ORDER BY source
        """
    ).df()
    ci.to_csv(TABLES / "semantic_company_industry.csv", index=False)
    print("\ncompany_industry semantic check:")
    print(ci.to_string())

    # seniority_native distribution
    sn = c.sql(
        f"""
        SELECT source, seniority_native, count(*) AS n
        FROM '{DATA}'
        WHERE source_platform='linkedin'
        GROUP BY source, seniority_native
        ORDER BY source, n DESC
        """
    ).df()
    sn.to_csv(TABLES / "semantic_seniority_native.csv", index=False)
    print("\nseniority_native distribution by source (LinkedIn all):")
    print(sn.to_string())

    # work_type — different semantics
    wt = c.sql(
        f"""
        SELECT source, work_type, count(*) AS n
        FROM '{DATA}'
        WHERE source_platform='linkedin'
        GROUP BY source, work_type
        ORDER BY source, n DESC
        """
    ).df()
    wt.to_csv(TABLES / "semantic_work_type.csv", index=False)

    # skills_raw: check whether it's pipe-delimited, comma-delimited, JSON, etc.
    sk = c.sql(
        f"""
        SELECT source,
               count(*) AS n_total,
               sum(CASE WHEN skills_raw IS NULL OR skills_raw='' THEN 0 ELSE 1 END) AS n_with,
               sum(CASE WHEN skills_raw LIKE '[%]' THEN 1 ELSE 0 END) AS n_json_like,
               sum(CASE WHEN skills_raw LIKE '%,%' THEN 1 ELSE 0 END) AS n_comma,
               sum(CASE WHEN skills_raw LIKE '%|%' THEN 1 ELSE 0 END) AS n_pipe
        FROM '{DATA}'
        WHERE source_platform='linkedin'
        GROUP BY source ORDER BY source
        """
    ).df()
    sk.to_csv(TABLES / "semantic_skills_raw.csv", index=False)
    print("\nskills_raw format check:")
    print(sk.to_string())

    # date_posted semantics — check populated share (scraped linkedin is only 2.8% populated by source)
    dp = c.sql(
        f"""
        SELECT source, source_platform,
               count(*) AS n_total,
               sum(CASE WHEN date_posted IS NULL OR date_posted='' THEN 0 ELSE 1 END) AS n_with,
               sum(CASE WHEN scrape_date IS NULL OR scrape_date='' THEN 0 ELSE 1 END) AS n_with_scrape
        FROM '{DATA}'
        GROUP BY source, source_platform ORDER BY source, source_platform
        """
    ).df()
    dp.to_csv(TABLES / "semantic_date_posted.csv", index=False)
    print("\ndate_posted presence by source/platform:")
    print(dp.to_string())

    # is_remote: known data artifact (0% in 2024 sources)
    rem = c.sql(
        f"""
        SELECT source, source_platform,
               count(*) AS n_total,
               sum(CASE WHEN is_remote THEN 1 ELSE 0 END) AS n_remote,
               sum(CASE WHEN is_remote_inferred THEN 1 ELSE 0 END) AS n_remote_inferred
        FROM '{DATA}'
        GROUP BY source, source_platform ORDER BY source, source_platform
        """
    ).df()
    rem.to_csv(TABLES / "semantic_is_remote.csv", index=False)
    print("\nis_remote / is_remote_inferred by source:")
    print(rem.to_string())


# ---------------- 5. Aggregator sensitivity ---------------- #

def aggregator_sensitivity(cols: list[tuple[str, str]]) -> None:
    """Compute column coverage separately for is_aggregator=true/false."""
    c = con()
    # Focus on the SWE LinkedIn default-filter set.
    # For memory, only compute for key analysis-relevant columns rather than all 96.
    key_cols = [
        "description", "description_core_llm", "title", "seniority_native",
        "seniority_final", "yoe_min_years_llm", "yoe_extracted",
        "company_industry", "company_size", "company_name_canonical",
        "metro_area", "state_normalized", "is_remote_inferred",
        "date_posted", "period", "swe_classification_llm", "ghost_assessment_llm",
        "skills_raw", "asaniczka_skills", "search_query", "search_metro_name",
    ]
    records = []
    for agg_flag in [True, False]:
        for source in SOURCES:
            for col in key_cols:
                where = (
                    f"source='{source}' AND is_swe AND source_platform='linkedin'"
                    f" AND is_english AND date_flag='ok' AND is_aggregator={agg_flag}"
                )
                null_expr = f"{col} IS NULL OR {col} = ''" if col not in {"company_size", "yoe_min_years_llm", "yoe_extracted", "is_remote_inferred"} else f"{col} IS NULL"
                q = f"""
                SELECT count(*) AS n, sum(CASE WHEN {null_expr} THEN 1 ELSE 0 END) AS n_null,
                       count(DISTINCT {col}) AS n_distinct
                FROM '{DATA}' WHERE {where}
                """
                try:
                    n, nn, nd = c.sql(q).fetchone()
                except Exception:
                    n, nn, nd = (0, 0, 0)
                records.append({
                    "column": col,
                    "source": source,
                    "is_aggregator": agg_flag,
                    "n_rows": n,
                    "n_null": nn,
                    "non_null_rate": round(1 - nn / n, 4) if n else 0.0,
                    "n_distinct": nd,
                })
    df = pd.DataFrame(records)
    df.to_csv(TABLES / "aggregator_sensitivity.csv", index=False)
    print(f"\nAggregator sensitivity written ({len(df)} rows)")


# ---------------- Main ---------------- #

def main() -> None:
    cols = get_columns()
    print(f"Discovered {len(cols)} columns")

    row_counts()
    cov = column_coverage(cols)
    flag_coverage_gaps(cov)

    # Heatmaps — all rows, SWE subset, SWE LinkedIn default subset
    coverage_heatmap(cov, "all", "coverage_heatmap_all.png", "all rows")
    coverage_heatmap(cov, "swe", "coverage_heatmap_swe.png", "is_swe only")
    coverage_heatmap(cov, "swe_linkedin_default", "coverage_heatmap_swe_linkedin.png",
                     "is_swe & linkedin & english & date_ok")

    llm_extraction_coverage()
    semantic_differences()
    aggregator_sensitivity(cols)

    print("\nT01 artifacts written.")


if __name__ == "__main__":
    main()
