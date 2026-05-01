"""T01 — Data profile and column coverage.

Computes non-null rate and distinct-value count for every column in
`data/unified.parquet`, sliced by source (kaggle_arshkon, kaggle_asaniczka,
scraped) and by is_swe subset. Outputs a CSV of the full coverage matrix and
renders a heatmap (columns x sources, colored by non-null rate) restricted to
the default filter (source_platform='linkedin' AND is_english AND date_flag='ok').

Runs entirely through DuckDB — the parquet is never materialized into pandas.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path("/home/jihgaboot/gabor/job-research")
PARQUET = ROOT / "data" / "unified.parquet"
TABLE_DIR = ROOT / "exploration" / "tables" / "T01"
FIG_DIR = ROOT / "exploration" / "figures" / "T01"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]

# Default filter language reused across the task
DEFAULT_FILTER = (
    "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'"
)


def get_columns(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    rows = con.execute(f"DESCRIBE SELECT * FROM '{PARQUET}'").fetchall()
    return [(r[0], r[1]) for r in rows]


def compute_coverage(
    con: duckdb.DuckDBPyConnection,
    columns: list[tuple[str, str]],
    extra_filter: str | None = None,
    label: str = "",
) -> pd.DataFrame:
    """For each (source, column), compute row count, non-null count, non-null
    rate, distinct count. Builds ONE big query per source for efficiency."""

    filter_sql = f" AND {extra_filter}" if extra_filter else ""
    all_rows: list[dict] = []

    for src in SOURCES:
        where = f"source = '{src}' AND {DEFAULT_FILTER}{filter_sql}"
        # First: total row count
        total = con.execute(
            f"SELECT count(*) FROM '{PARQUET}' WHERE {where}"
        ).fetchone()[0]

        # Build all metric expressions in a single query. For distinct counts
        # we use COUNT(DISTINCT ...) which DuckDB handles efficiently.
        metric_parts: list[str] = []
        for name, ctype in columns:
            ident = f'"{name}"'
            if ctype == "VARCHAR":
                # Treat empty string as null for coverage purposes, consistent
                # with the project convention (description, etc.).
                nn_expr = (
                    f"SUM(CASE WHEN {ident} IS NOT NULL AND {ident} <> '' "
                    f"THEN 1 ELSE 0 END)"
                )
            else:
                nn_expr = (
                    f"SUM(CASE WHEN {ident} IS NOT NULL THEN 1 ELSE 0 END)"
                )
            metric_parts.append(f"{nn_expr} AS nn__{name}")
            metric_parts.append(f"COUNT(DISTINCT {ident}) AS nd__{name}")

        sql = (
            f"SELECT {', '.join(metric_parts)} FROM '{PARQUET}' WHERE {where}"
        )
        row = con.execute(sql).fetchone()
        metric_names = [
            desc[0] for desc in con.execute(sql).description
        ]
        # Second execute above re-runs; redo without description usage (one call)
        # Simpler: we already have row; derive column names from metric_parts order.
        metric_names = [p.split(" AS ")[-1] for p in metric_parts]
        metric_map = dict(zip(metric_names, row))

        for name, _ in columns:
            nn = metric_map[f"nn__{name}"] or 0
            nd = metric_map[f"nd__{name}"] or 0
            rate = (nn / total) if total else 0.0
            all_rows.append(
                {
                    "subset": label,
                    "source": src,
                    "column": name,
                    "rows": total,
                    "non_null": nn,
                    "non_null_rate": rate,
                    "distinct": nd,
                }
            )

    return pd.DataFrame(all_rows)


def pivot_rate(df: pd.DataFrame, columns_order: list[str]) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="column", columns="source", values="non_null_rate", aggfunc="first"
    )
    pivot = pivot.reindex(columns_order)
    pivot = pivot[SOURCES]
    return pivot


def render_heatmap(pivot: pd.DataFrame, path: Path, title: str) -> None:
    # Order rows: first by average coverage asc so low-coverage columns stand out.
    order = pivot.mean(axis=1).sort_values(ascending=True).index
    p = pivot.loc[order]

    height = max(18, 0.26 * len(p))
    fig, ax = plt.subplots(figsize=(8.5, height))
    sns.heatmap(
        p,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "non-null rate"},
        linewidths=0.25,
        linecolor="white",
        annot_kws={"size": 8},
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("source", fontsize=10)
    ax.set_ylabel("column", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=10)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def description_core_coverage(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    # llm_extraction_coverage distribution for SWE rows by source
    sql = f"""
    SELECT source, period, llm_extraction_coverage, count(*) n
    FROM '{PARQUET}'
    WHERE is_swe AND {DEFAULT_FILTER}
    GROUP BY source, period, llm_extraction_coverage
    ORDER BY source, period, llm_extraction_coverage
    """
    return con.execute(sql).df()


def llm_classification_coverage(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = f"""
    SELECT source, period, llm_classification_coverage, count(*) n
    FROM '{PARQUET}'
    WHERE is_swe AND {DEFAULT_FILTER}
    GROUP BY source, period, llm_classification_coverage
    ORDER BY source, period, llm_classification_coverage
    """
    return con.execute(sql).df()


def semantic_diff_company_industry(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = f"""
    SELECT source,
           count(*) FILTER (WHERE company_industry IS NOT NULL AND company_industry <> '') nn,
           count(DISTINCT company_industry) n_distinct,
           count(*) FILTER (WHERE company_industry LIKE '%,%' OR company_industry LIKE '% and %') n_compound,
           count(*) total
    FROM '{PARQUET}'
    WHERE {DEFAULT_FILTER}
    GROUP BY source
    ORDER BY source
    """
    return con.execute(sql).df()


def main() -> None:
    print(f"[T01] opening {PARQUET}", file=sys.stderr)
    con = duckdb.connect()
    # Cap memory usage
    con.execute("PRAGMA memory_limit='16GB'")
    con.execute("PRAGMA threads=4")

    columns = get_columns(con)
    col_names = [c[0] for c in columns]
    print(f"[T01] {len(columns)} columns discovered", file=sys.stderr)

    # All-rows coverage (passing default filter)
    print("[T01] computing coverage (all rows)", file=sys.stderr)
    cov_all = compute_coverage(con, columns, label="all")
    # SWE-only coverage
    print("[T01] computing coverage (is_swe)", file=sys.stderr)
    cov_swe = compute_coverage(con, columns, extra_filter="is_swe", label="swe")

    coverage = pd.concat([cov_all, cov_swe], ignore_index=True)
    coverage.to_csv(TABLE_DIR / "coverage.csv", index=False)
    print(f"[T01] wrote {TABLE_DIR/'coverage.csv'}", file=sys.stderr)

    # Heatmap — ALL rows — drives the primary figure
    pivot_all = pivot_rate(cov_all, col_names)
    pivot_swe = pivot_rate(cov_swe, col_names)

    pivot_all.to_csv(TABLE_DIR / "coverage_heatmap_all.csv")
    pivot_swe.to_csv(TABLE_DIR / "coverage_heatmap_swe.csv")

    render_heatmap(
        pivot_all,
        FIG_DIR / "coverage_heatmap.png",
        "Column non-null coverage by source — all rows (default filter)",
    )
    render_heatmap(
        pivot_swe,
        FIG_DIR / "coverage_heatmap_swe.png",
        "Column non-null coverage by source — is_swe subset (default filter)",
    )
    print(f"[T01] wrote heatmaps to {FIG_DIR}", file=sys.stderr)

    # Flag >50% null columns for any source
    mask = (pivot_all < 0.50)
    flagged = []
    for col in pivot_all.index:
        sources_low = [s for s in SOURCES if mask.at[col, s]]
        if sources_low:
            flagged.append(
                {
                    "column": col,
                    "arshkon_rate": pivot_all.at[col, "kaggle_arshkon"],
                    "asaniczka_rate": pivot_all.at[col, "kaggle_asaniczka"],
                    "scraped_rate": pivot_all.at[col, "scraped"],
                    "sources_below_50pct": ",".join(sources_low),
                }
            )
    flagged_df = pd.DataFrame(flagged).sort_values("column")
    flagged_df.to_csv(TABLE_DIR / "flagged_gt50pct_null.csv", index=False)
    print(
        f"[T01] flagged {len(flagged_df)} columns with >50% null in >=1 source",
        file=sys.stderr,
    )

    # description_core_llm coverage
    dcl = description_core_coverage(con)
    dcl.to_csv(TABLE_DIR / "description_core_llm_coverage.csv", index=False)

    llc = llm_classification_coverage(con)
    llc.to_csv(TABLE_DIR / "llm_classification_coverage.csv", index=False)

    # Semantic difference: company_industry
    semdiff = semantic_diff_company_industry(con)
    semdiff.to_csv(TABLE_DIR / "semantic_diff_company_industry.csv", index=False)

    # Print compact summary to stdout for log / report drafting
    print("\n=== ALL-ROWS COVERAGE PIVOT ===")
    print(pivot_all.round(3).to_string())
    print("\n=== SWE-SUBSET COVERAGE PIVOT ===")
    print(pivot_swe.round(3).to_string())
    print("\n=== COLUMNS <50% IN ANY SOURCE (all rows) ===")
    print(flagged_df.to_string(index=False))
    print("\n=== description_core_llm COVERAGE (is_swe, default filter) ===")
    print(dcl.to_string())
    print("\n=== llm_classification_coverage (is_swe, default filter) ===")
    print(llc.to_string())
    print("\n=== company_industry semantic differences ===")
    print(semdiff.to_string())


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
