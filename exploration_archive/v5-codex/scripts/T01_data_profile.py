from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
FIG_DIR = ROOT / "exploration" / "figures" / "T01"
TAB_DIR = ROOT / "exploration" / "tables" / "T01"

PRIMARY_FILTER = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'"
SOURCES = ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]


def dq(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{DATA.as_posix()}')").fetchdf()
    columns = schema["column_name"].tolist()

    row_counts_primary = con.execute(
        f"""
        SELECT source,
               COUNT(*) AS total_n,
               SUM(CASE WHEN is_swe THEN 1 ELSE 0 END) AS swe_n,
               SUM(CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) AS swe_adjacent_n,
               SUM(CASE WHEN is_control THEN 1 ELSE 0 END) AS control_n,
               SUM(CASE WHEN NOT is_swe AND NOT is_swe_adjacent AND NOT is_control THEN 1 ELSE 0 END) AS unclassified_n
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {PRIMARY_FILTER}
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    row_counts_primary.to_csv(TAB_DIR / "T01_row_counts_primary.csv", index=False)

    row_counts_full = con.execute(
        f"""
        SELECT source_platform,
               COUNT(*) AS total_n,
               SUM(CASE WHEN is_swe THEN 1 ELSE 0 END) AS swe_n,
               SUM(CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) AS swe_adjacent_n,
               SUM(CASE WHEN is_control THEN 1 ELSE 0 END) AS control_n,
               SUM(CASE WHEN NOT is_swe AND NOT is_swe_adjacent AND NOT is_control THEN 1 ELSE 0 END) AS unclassified_n
        FROM read_parquet('{DATA.as_posix()}')
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    row_counts_full.to_csv(TAB_DIR / "T01_row_counts_full.csv", index=False)

    coverage_rows = []
    for source in SOURCES:
        for subset_name, subset_filter in [
            ("all", ""),
            ("swe", "AND is_swe = true"),
        ]:
            where_clause = f"WHERE {PRIMARY_FILTER} AND source = '{source}' {subset_filter}"
            select_exprs = ["COUNT(*) AS total_n"]
            alias_map = []
            for idx, col in enumerate(columns):
                nn_alias = f"nn__{idx}"
                dc_alias = f"dc__{idx}"
                quoted = dq(col)
                select_exprs.append(f"COUNT(*) FILTER (WHERE {quoted} IS NOT NULL) AS {nn_alias}")
                select_exprs.append(f"COUNT(DISTINCT {quoted}) AS {dc_alias}")
                alias_map.append((col, nn_alias, dc_alias))

            query = f"""
            SELECT {', '.join(select_exprs)}
            FROM read_parquet('{DATA.as_posix()}')
            {where_clause}
            """
            row = con.execute(query).fetchone()
            total_n = row[0]
            for i, (col, nn_alias, dc_alias) in enumerate(alias_map, start=1):
                non_null_n = row[2 * i - 1]
                distinct_n = row[2 * i]
                coverage_rows.append(
                    {
                        "source": source,
                        "subset": subset_name,
                        "column_name": col,
                        "total_n": total_n,
                        "non_null_n": non_null_n,
                        "non_null_rate": None if total_n == 0 else non_null_n / total_n,
                        "null_rate": None if total_n == 0 else 1 - (non_null_n / total_n),
                        "distinct_count": distinct_n,
                    }
                )

    coverage = pd.DataFrame(coverage_rows)
    coverage = coverage[
        ["source", "subset", "column_name", "total_n", "non_null_n", "non_null_rate", "null_rate", "distinct_count"]
    ]
    coverage.to_csv(TAB_DIR / "T01_column_coverage_long.csv", index=False)

    sparse = coverage[(coverage["subset"] == "all") & (coverage["null_rate"] > 0.5)].copy()
    sparse.sort_values(["source", "null_rate", "column_name"], ascending=[True, False, True], inplace=True)
    sparse.to_csv(TAB_DIR / "T01_sparse_columns_gt50_null_primary_all.csv", index=False)

    llm_cov = con.execute(
        f"""
        SELECT source,
               llm_extraction_coverage,
               COUNT(*) AS n,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY source), 1) AS pct_of_source
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {PRIMARY_FILTER} AND is_swe = true
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchdf()
    llm_cov.to_csv(TAB_DIR / "T01_llm_extraction_coverage_swe.csv", index=False)

    semantic_notes = con.execute(
        f"""
        SELECT source,
               COUNT(*) FILTER (
                   WHERE company_industry IS NOT NULL
                     AND (company_industry LIKE '%,%' OR company_industry LIKE '%;%' OR company_industry LIKE '%|%' OR company_industry LIKE '%/%')
               ) AS company_industry_compound_like_n,
               COUNT(*) FILTER (WHERE company_industry IS NOT NULL) AS company_industry_non_null_n,
               COUNT(*) FILTER (WHERE company_size_category IS NOT NULL) AS company_size_category_non_null_n,
               COUNT(*) FILTER (WHERE company_id_kaggle IS NOT NULL) AS company_id_kaggle_non_null_n
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {PRIMARY_FILTER}
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    semantic_notes["company_industry_compound_like_rate"] = semantic_notes["company_industry_compound_like_n"] / semantic_notes[
        "company_industry_non_null_n"
    ].replace({0: pd.NA})
    semantic_notes.to_csv(TAB_DIR / "T01_source_semantics_notes.csv", index=False)

    # Figure: two-panel heatmap for all rows and SWE rows.
    sns.set_theme(style="white")
    heatmaps = {}
    for subset in ["all", "swe"]:
        frame = coverage[coverage["subset"] == subset].pivot(index="column_name", columns="source", values="non_null_rate")
        frame = frame.reindex(columns=SOURCES)
        heatmaps[subset] = frame

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, max(14, 0.22 * len(columns))), sharey=True)
    cmap = sns.color_palette("viridis", as_cmap=True)
    for ax, subset in zip(axes, ["all", "swe"]):
        sns.heatmap(
            heatmaps[subset],
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=1,
            cbar=subset == "swe",
            cbar_kws={"label": "Non-null rate"} if subset == "swe" else None,
            linewidths=0.1,
            linecolor="white",
        )
        ax.set_title("All rows" if subset == "all" else "SWE rows", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=6)
        ax.tick_params(axis="x", labelsize=8, rotation=0)

    fig.suptitle("T01 coverage heatmap: non-null rate by source", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(FIG_DIR / "T01_coverage_heatmap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
