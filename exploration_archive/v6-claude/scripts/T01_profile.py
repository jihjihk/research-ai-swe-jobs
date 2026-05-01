"""T01: Data profile and column coverage.

Computes authoritative row counts by source x period x classification, column
non-null coverage heatmap, llm_extraction/classification coverage for SWE rows,
and key constraint mapping. All queries run in DuckDB; no full-parquet pandas loads.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "unified.parquet"
OUT_TABLES = REPO / "exploration" / "tables" / "T01"
OUT_FIGS = REPO / "exploration" / "figures" / "T01"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

DEFAULT_FILTER = "source_platform = 'linkedin' AND is_english AND date_flag = 'ok'"


def q(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def main() -> None:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW u AS SELECT * FROM '{DATA.as_posix()}'")

    # --- 1. Authoritative row counts -----------------------------------------
    total = q(con, "SELECT count(*) AS n FROM u")["n"].iloc[0]
    print(f"Total rows in unified.parquet: {total:,}")

    by_source_platform = q(
        con,
        """
        SELECT source, source_platform, count(*) AS n
        FROM u
        GROUP BY 1,2
        ORDER BY 1,2
        """,
    )
    by_source_platform.to_csv(OUT_TABLES / "rows_by_source_platform.csv", index=False)
    print("\nRows by source x source_platform:\n", by_source_platform)

    by_source_period = q(
        con,
        """
        SELECT source, source_platform, period, count(*) AS n
        FROM u
        WHERE source_platform IN ('linkedin','indeed')
        GROUP BY 1,2,3
        ORDER BY 1,2,3
        """,
    )
    by_source_period.to_csv(OUT_TABLES / "rows_by_source_period.csv", index=False)
    print("\nRows by source x platform x period:\n", by_source_period)

    # Authoritative: source x period x is_swe / is_swe_adjacent / is_control,
    # LinkedIn only after default filter (primary analysis frame).
    counts_primary = q(
        con,
        f"""
        SELECT source, period,
               sum(CASE WHEN is_swe THEN 1 ELSE 0 END) AS n_swe,
               sum(CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) AS n_swe_adjacent,
               sum(CASE WHEN is_control THEN 1 ELSE 0 END) AS n_control,
               sum(CASE WHEN NOT is_swe AND NOT is_swe_adjacent AND NOT is_control THEN 1 ELSE 0 END) AS n_other,
               count(*) AS n_total
        FROM u
        WHERE {DEFAULT_FILTER}
        GROUP BY 1,2
        ORDER BY 1,2
        """,
    )
    counts_primary.to_csv(OUT_TABLES / "counts_linkedin_primary.csv", index=False)
    print("\nLinkedIn primary counts (default filter):\n", counts_primary)

    # Unfiltered (no default filter) counts -- so downstream can see filter cost
    counts_unfiltered = q(
        con,
        """
        SELECT source, source_platform, period,
               sum(CASE WHEN is_swe THEN 1 ELSE 0 END) AS n_swe,
               sum(CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) AS n_swe_adjacent,
               sum(CASE WHEN is_control THEN 1 ELSE 0 END) AS n_control,
               count(*) AS n_total
        FROM u
        GROUP BY 1,2,3
        ORDER BY 1,2,3
        """,
    )
    counts_unfiltered.to_csv(OUT_TABLES / "counts_unfiltered.csv", index=False)

    # Indeed separately
    indeed_counts = q(
        con,
        """
        SELECT source, period,
               sum(CASE WHEN is_swe THEN 1 ELSE 0 END) AS n_swe,
               sum(CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) AS n_swe_adjacent,
               sum(CASE WHEN is_control THEN 1 ELSE 0 END) AS n_control,
               count(*) AS n_total
        FROM u
        WHERE source_platform = 'indeed'
        GROUP BY 1,2
        ORDER BY 1,2
        """,
    )
    indeed_counts.to_csv(OUT_TABLES / "counts_indeed.csv", index=False)
    print("\nIndeed counts (all, no language/date filter):\n", indeed_counts)

    # Default-filter cost vs unfiltered
    filter_cost = q(
        con,
        """
        SELECT source, source_platform,
               count(*) AS n_raw,
               sum(CASE WHEN is_english THEN 1 ELSE 0 END) AS n_english,
               sum(CASE WHEN date_flag='ok' THEN 1 ELSE 0 END) AS n_date_ok,
               sum(CASE WHEN source_platform='linkedin' AND is_english AND date_flag='ok' THEN 1 ELSE 0 END) AS n_after_default
        FROM u
        GROUP BY 1,2
        ORDER BY 1,2
        """,
    )
    filter_cost.to_csv(OUT_TABLES / "default_filter_cost.csv", index=False)
    print("\nDefault filter cost:\n", filter_cost)

    # --- 2. LLM coverage distributions for SWE rows --------------------------
    llm_ext_cov = q(
        con,
        f"""
        SELECT source, llm_extraction_coverage, count(*) AS n
        FROM u
        WHERE {DEFAULT_FILTER} AND is_swe
        GROUP BY 1,2
        ORDER BY 1,2
        """,
    )
    llm_ext_cov.to_csv(OUT_TABLES / "llm_extraction_coverage_swe.csv", index=False)
    print("\nllm_extraction_coverage for SWE (LinkedIn default filter):\n", llm_ext_cov)

    llm_cls_cov = q(
        con,
        f"""
        SELECT source, llm_classification_coverage, count(*) AS n
        FROM u
        WHERE {DEFAULT_FILTER} AND is_swe
        GROUP BY 1,2
        ORDER BY 1,2
        """,
    )
    llm_cls_cov.to_csv(OUT_TABLES / "llm_classification_coverage_swe.csv", index=False)
    print("\nllm_classification_coverage for SWE:\n", llm_cls_cov)

    # Same for Indeed SWE
    indeed_llm = q(
        con,
        """
        SELECT llm_extraction_coverage, llm_classification_coverage, count(*) AS n
        FROM u
        WHERE source_platform='indeed' AND is_swe
        GROUP BY 1,2
        ORDER BY 3 DESC
        """,
    )
    indeed_llm.to_csv(OUT_TABLES / "llm_coverage_indeed_swe.csv", index=False)

    # --- 3. Column coverage heatmap (all columns x sources) -----------------
    cols = [r[0] for r in con.execute("DESCRIBE u").fetchall()]

    # Build a single query computing non-null rates for each source on LinkedIn filtered rows
    sources = ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]

    # Non-null rate overall (linkedin default filter) by source
    def nonnull_rates(
        extra_filter: str,
        label: str,
    ) -> pd.DataFrame:
        rows = []
        # Get row counts per source under this filter
        base_counts = q(
            con,
            f"""
            SELECT source, count(*) AS n
            FROM u
            WHERE {DEFAULT_FILTER} {extra_filter}
            GROUP BY 1
            """,
        ).set_index("source")["n"].to_dict()
        for col in cols:
            # Build one SQL with all 3 sources
            parts = []
            for s in sources:
                parts.append(
                    f"sum(CASE WHEN source='{s}' AND {col} IS NOT NULL "
                    f"AND CAST({col} AS VARCHAR) <> '' THEN 1 ELSE 0 END) AS nn_{s}"
                )
            sql = f"""
                SELECT {', '.join(parts)}
                FROM u
                WHERE {DEFAULT_FILTER} {extra_filter}
            """
            r = con.execute(sql).fetchone()
            row = {"column": col, "scope": label}
            for i, s in enumerate(sources):
                n = base_counts.get(s, 0)
                nn = r[i] or 0
                row[f"{s}_nonnull"] = nn
                row[f"{s}_n"] = n
                row[f"{s}_rate"] = nn / n if n else np.nan
            # distinct counts
            for s in sources:
                try:
                    d = con.execute(
                        f"SELECT count(DISTINCT {col}) FROM u WHERE {DEFAULT_FILTER} {extra_filter} AND source='{s}'"
                    ).fetchone()[0]
                except Exception:
                    d = np.nan
                row[f"{s}_distinct"] = d
            rows.append(row)
        return pd.DataFrame(rows)

    print("\nComputing column coverage (all LinkedIn rows)...")
    cov_all = nonnull_rates("", "linkedin_all")
    cov_all.to_csv(OUT_TABLES / "column_coverage_linkedin_all.csv", index=False)

    print("Computing column coverage (SWE only)...")
    cov_swe = nonnull_rates(" AND is_swe", "linkedin_swe")
    cov_swe.to_csv(OUT_TABLES / "column_coverage_linkedin_swe.csv", index=False)

    # Indeed column coverage separately
    print("Computing column coverage for Indeed (scraped)...")
    indeed_rows = []
    base = con.execute(
        "SELECT count(*) FROM u WHERE source_platform='indeed'"
    ).fetchone()[0]
    for col in cols:
        nn = con.execute(
            f"SELECT count(*) FROM u WHERE source_platform='indeed' AND {col} IS NOT NULL AND CAST({col} AS VARCHAR) <> ''"
        ).fetchone()[0]
        indeed_rows.append({"column": col, "indeed_n": base, "indeed_nonnull": nn, "indeed_rate": (nn / base) if base else np.nan})
    indeed_cov = pd.DataFrame(indeed_rows)
    indeed_cov.to_csv(OUT_TABLES / "column_coverage_indeed.csv", index=False)

    # --- 4. Heatmap figure ---------------------------------------------------
    heat = cov_all[["column"] + [f"{s}_rate" for s in sources]].copy()
    heat = heat.merge(indeed_cov[["column", "indeed_rate"]], on="column", how="left")
    heat.columns = ["column", "arshkon", "asaniczka", "scraped_linkedin", "scraped_indeed"]
    heat.to_csv(OUT_TABLES / "coverage_heatmap.csv", index=False)

    # Plot
    mat = heat.set_index("column")[["arshkon", "asaniczka", "scraped_linkedin", "scraped_indeed"]]
    fig, ax = plt.subplots(figsize=(8, max(12, 0.22 * len(mat))))
    im = ax.imshow(mat.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=30, ha="right", fontsize=9)
    ax.set_title("Column non-null rate by source (default LinkedIn filter; Indeed raw)")
    plt.colorbar(im, ax=ax, label="non-null rate")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "coverage_heatmap_all_cols.png", dpi=150)
    plt.close()

    # Focused heatmap on SWE rows, key analytical columns only
    focus_cols = [
        "title_normalized",
        "description",
        "description_core_llm",
        "seniority_native",
        "seniority_final",
        "seniority_final_source",
        "yoe_extracted",
        "yoe_min_years_llm",
        "metro_area",
        "city_extracted",
        "state_normalized",
        "is_remote_inferred",
        "company_name_canonical",
        "company_industry",
        "company_size",
        "skills_raw",
        "asaniczka_skills",
        "swe_classification_llm",
        "ghost_assessment_llm",
        "ghost_job_risk",
        "date_posted",
        "scrape_date",
        "period",
    ]
    heat_swe = cov_swe[["column"] + [f"{s}_rate" for s in sources]].copy()
    heat_swe = heat_swe.merge(indeed_cov[["column", "indeed_rate"]], on="column", how="left")
    heat_swe.columns = ["column", "arshkon", "asaniczka", "scraped_linkedin", "scraped_indeed"]
    heat_swe_focus = heat_swe[heat_swe["column"].isin(focus_cols)].set_index("column").reindex(focus_cols)
    heat_swe_focus.to_csv(OUT_TABLES / "coverage_heatmap_swe_focus.csv")
    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(heat_swe_focus.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(heat_swe_focus.index)))
    ax.set_yticklabels(heat_swe_focus.index, fontsize=9)
    ax.set_xticks(range(heat_swe_focus.shape[1]))
    ax.set_xticklabels(heat_swe_focus.columns, rotation=20, ha="right", fontsize=9)
    for i in range(heat_swe_focus.shape[0]):
        for j in range(heat_swe_focus.shape[1]):
            v = heat_swe_focus.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=7)
    ax.set_title("Coverage heatmap: analytical columns x sources (SWE only)")
    plt.colorbar(im, ax=ax, label="non-null rate")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "coverage_heatmap_swe_focus.png", dpi=150)
    plt.close()

    # --- 5. Columns >50% null per source (flag list) ------------------------
    flagged = []
    for _, row in cov_all.iterrows():
        col = row["column"]
        for s in sources:
            rate = row[f"{s}_rate"]
            if pd.notna(rate) and rate < 0.5:
                flagged.append({"column": col, "source": s, "nonnull_rate": rate, "n_nonnull": row[f"{s}_nonnull"], "n_total": row[f"{s}_n"]})
    flagged_df = pd.DataFrame(flagged)
    flagged_df.sort_values(["source", "nonnull_rate"]).to_csv(
        OUT_TABLES / "columns_over_50pct_null.csv", index=False
    )
    print(f"\nColumns >50% null across any source: {len(flagged_df)}")

    # --- 6. Semantic differences spot checks --------------------------------
    # company_industry: distinct counts and top values per source
    ind_stats = q(
        con,
        f"""
        SELECT source,
               count(DISTINCT company_industry) AS distinct_industries,
               avg(CASE WHEN company_industry LIKE '%,%' OR company_industry LIKE '% and %' THEN 1 ELSE 0 END) AS compound_rate,
               sum(CASE WHEN company_industry IS NOT NULL THEN 1 ELSE 0 END) AS n_nonnull,
               count(*) AS n
        FROM u
        WHERE {DEFAULT_FILTER}
        GROUP BY 1
        """,
    )
    ind_stats.to_csv(OUT_TABLES / "company_industry_semantics.csv", index=False)
    print("\ncompany_industry semantics by source:\n", ind_stats)

    top_industries = q(
        con,
        f"""
        SELECT source, company_industry, count(*) AS n
        FROM u
        WHERE {DEFAULT_FILTER} AND company_industry IS NOT NULL
        GROUP BY 1,2
        QUALIFY row_number() OVER (PARTITION BY source ORDER BY n DESC) <= 10
        ORDER BY source, n DESC
        """,
    )
    top_industries.to_csv(OUT_TABLES / "top_industries_by_source.csv", index=False)

    # period distinct values by source
    period_vals = q(
        con,
        f"""
        SELECT source, period, count(*) AS n
        FROM u
        WHERE {DEFAULT_FILTER}
        GROUP BY 1,2
        ORDER BY 1,2
        """,
    )
    period_vals.to_csv(OUT_TABLES / "period_values_by_source.csv", index=False)

    # date ranges per source
    date_ranges = q(
        con,
        """
        SELECT source, source_platform,
               min(date_posted) AS dp_min, max(date_posted) AS dp_max,
               min(scrape_date) AS sd_min, max(scrape_date) AS sd_max
        FROM u
        GROUP BY 1,2
        ORDER BY 1,2
        """,
    )
    date_ranges.to_csv(OUT_TABLES / "date_ranges.csv", index=False)
    print("\nDate ranges:\n", date_ranges)

    # --- 7. Save a summary JSON for the report ------------------------------
    summary = {
        "total_rows": int(total),
        "by_source_platform": by_source_platform.to_dict(orient="records"),
        "linkedin_primary_counts": counts_primary.to_dict(orient="records"),
        "indeed_counts": indeed_counts.to_dict(orient="records"),
        "llm_extraction_coverage_swe": llm_ext_cov.to_dict(orient="records"),
        "llm_classification_coverage_swe": llm_cls_cov.to_dict(orient="records"),
        "columns_over_50pct_null_count": int(len(flagged_df)),
        "date_ranges": date_ranges.to_dict(orient="records"),
    }
    with open(OUT_TABLES / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nT01 done. Outputs written to", OUT_TABLES)


if __name__ == "__main__":
    main()
