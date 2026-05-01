"""
Phase A corpus profile for `data/unified.parquet`.

Produces:
  eda/tables/A_corpus_by_period.csv
  eda/tables/A_fill_rates.csv
  eda/tables/A_llm_coverage.csv
  eda/tables/A_swe_share_by_period.csv
  eda/figures/A_corpus_overview.png

Run:
  ./.venv/bin/python eda/scripts/profile.py
"""

from __future__ import annotations

import random
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np

random.seed(0)
np.random.seed(0)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_PATH = PROJECT_ROOT / "data" / "unified.parquet"
TABLES_DIR = PROJECT_ROOT / "eda" / "tables"
FIGURES_DIR = PROJECT_ROOT / "eda" / "figures"


def q_corpus_by_period(con: duckdb.DuckDBPyConnection):
    return con.execute(f"""
      SELECT source, source_platform, period,
             COALESCE(seniority_final, 'null') AS seniority_final,
             COUNT(*) AS n
      FROM '{UNIFIED_PATH}'
      GROUP BY 1,2,3,4
      ORDER BY 1,2,3,4
    """).df()


def q_fill_rates(con: duckdb.DuckDBPyConnection):
    cols = [
        "description", "seniority_native", "seniority_final",
        "date_posted", "company_industry", "company_size",
        "yoe_min_years_llm", "description_core_llm",
        "metro_area", "is_remote_inferred", "is_aggregator",
    ]
    nn_exprs = ", ".join(
        f"SUM(CASE WHEN {c} IS NOT NULL THEN 1 ELSE 0 END) AS n_{c}"
        for c in cols
    )
    df = con.execute(f"""
      SELECT source, source_platform,
             COUNT(*) AS n_rows,
             {nn_exprs}
      FROM '{UNIFIED_PATH}'
      GROUP BY 1,2
      ORDER BY 1,2
    """).df()
    # Normalize to fill-rate columns
    for c in cols:
        df[f"fill_{c}"] = df[f"n_{c}"] / df["n_rows"]
    keep = ["source", "source_platform", "n_rows"] + [f"fill_{c}" for c in cols]
    return df[keep]


def q_llm_coverage(con: duckdb.DuckDBPyConnection):
    return con.execute(f"""
      SELECT source, source_platform, period,
             CASE
               WHEN is_swe THEN 'swe'
               WHEN is_swe_adjacent THEN 'swe_adjacent'
               WHEN is_control THEN 'control'
               ELSE 'other'
             END AS analysis_group,
             llm_classification_coverage,
             llm_extraction_coverage,
             COUNT(*) AS n
      FROM '{UNIFIED_PATH}'
      GROUP BY 1,2,3,4,5,6
      ORDER BY 1,2,3,4,5,6
    """).df()


def q_swe_share_by_period(con: duckdb.DuckDBPyConnection):
    return con.execute(f"""
      SELECT source, source_platform, period,
             SUM(CASE WHEN is_swe THEN 1 ELSE 0 END) AS swe,
             SUM(CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) AS swe_adjacent,
             SUM(CASE WHEN is_control THEN 1 ELSE 0 END) AS control,
             COUNT(*) AS total
      FROM '{UNIFIED_PATH}'
      WHERE is_english = true AND date_flag = 'ok'
      GROUP BY 1,2,3
      ORDER BY 1,2,3
    """).df()


def build_figure(corpus_df, swe_df, llm_df, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel 1: Rows by period x source (stacked bar)
    pivot = (corpus_df
             .groupby(["period", "source", "source_platform"])["n"].sum()
             .reset_index())
    pivot["src_key"] = pivot["source"] + " / " + pivot["source_platform"]
    wide = pivot.pivot_table(index="period", columns="src_key",
                             values="n", aggfunc="sum", fill_value=0)
    wide.plot(kind="bar", stacked=True, ax=axes[0, 0], colormap="tab10")
    axes[0, 0].set_title(f"Postings by period × source  (n={int(wide.values.sum()):,})")
    axes[0, 0].set_ylabel("rows")
    axes[0, 0].legend(loc="upper left", fontsize=7)
    axes[0, 0].tick_params(axis="x", rotation=30)

    # Panel 2: Seniority mix by period (SWE only)
    sen = corpus_df.copy()
    sen = sen[sen["seniority_final"].isin(
        ["entry", "associate", "mid-senior", "director", "unknown", "null"])]
    # Filter to SWE LinkedIn for cleanest view — but corpus_df has no is_swe. Use swe_df as a merge.
    # Simpler: show seniority mix over ALL rows, then restrict to LinkedIn English ok in panel 2
    sen_wide = sen.pivot_table(
        index="period", columns="seniority_final",
        values="n", aggfunc="sum", fill_value=0
    )
    order = ["entry", "associate", "mid-senior", "director", "unknown", "null"]
    sen_wide = sen_wide.reindex(columns=[c for c in order if c in sen_wide.columns])
    sen_pct = sen_wide.div(sen_wide.sum(axis=1), axis=0) * 100
    sen_pct.plot(kind="bar", stacked=True, ax=axes[0, 1], colormap="viridis")
    axes[0, 1].set_title(f"Seniority mix by period (all rows, n={int(sen_wide.values.sum()):,})")
    axes[0, 1].set_ylabel("% of rows")
    axes[0, 1].legend(loc="upper left", fontsize=7)
    axes[0, 1].tick_params(axis="x", rotation=30)

    # Panel 3: LLM classification coverage rate by source x period
    lc = llm_df.copy()
    lc["labeled"] = (lc["llm_classification_coverage"] == "labeled").astype(int) * lc["n"]
    agg = lc.groupby(["source", "source_platform", "period"]).agg(
        labeled=("labeled", "sum"), total=("n", "sum")
    ).reset_index()
    agg["rate"] = agg["labeled"] / agg["total"]
    agg["key"] = agg["source"] + "/" + agg["source_platform"] + "\n" + agg["period"]
    axes[1, 0].bar(agg["key"], agg["rate"], color="steelblue")
    axes[1, 0].set_title("LLM classification coverage rate (labeled / total)")
    axes[1, 0].set_ylabel("labeled share")
    axes[1, 0].tick_params(axis="x", rotation=75, labelsize=7)
    axes[1, 0].set_ylim(0, 1)

    # Panel 4: SWE / adjacent / control shares by period (default filter applied)
    sw = swe_df.copy()
    sw["key"] = sw["source"] + "/" + sw["source_platform"] + "\n" + sw["period"]
    sw["swe_share"] = sw["swe"] / sw["total"]
    sw["adj_share"] = sw["swe_adjacent"] / sw["total"]
    sw["ctl_share"] = sw["control"] / sw["total"]
    idx = np.arange(len(sw))
    width = 0.25
    axes[1, 1].bar(idx - width, sw["swe_share"], width, label="SWE", color="#d62728")
    axes[1, 1].bar(idx, sw["adj_share"], width, label="SWE-adjacent", color="#ff7f0e")
    axes[1, 1].bar(idx + width, sw["ctl_share"], width, label="control", color="#2ca02c")
    axes[1, 1].set_xticks(idx)
    axes[1, 1].set_xticklabels(sw["key"], rotation=75, fontsize=7)
    axes[1, 1].set_title(
        f"SWE / adjacent / control share by period  (en=true, date_flag=ok; n_total={int(sw['total'].sum()):,})"
    )
    axes[1, 1].set_ylabel("share of rows")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Corpus profile — data/unified.parquet (Phase A)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    print("Running Q1: corpus by period × source × seniority")
    corpus_df = q_corpus_by_period(con)
    corpus_df.to_csv(TABLES_DIR / "A_corpus_by_period.csv", index=False)
    print(f"  rows: {len(corpus_df)}, total n: {corpus_df['n'].sum():,}")

    print("Running Q2: fill rates by source")
    fill_df = q_fill_rates(con)
    fill_df.to_csv(TABLES_DIR / "A_fill_rates.csv", index=False)
    print(f"  rows: {len(fill_df)}")

    print("Running Q3: LLM coverage")
    llm_df = q_llm_coverage(con)
    llm_df.to_csv(TABLES_DIR / "A_llm_coverage.csv", index=False)
    print(f"  rows: {len(llm_df)}")

    print("Running Q4: SWE/adj/control share by period")
    swe_df = q_swe_share_by_period(con)
    swe_df.to_csv(TABLES_DIR / "A_swe_share_by_period.csv", index=False)
    print(f"  rows: {len(swe_df)}")

    print("Building overview figure")
    build_figure(corpus_df, swe_df, llm_df, FIGURES_DIR / "A_corpus_overview.png")
    print(f"  figure: {FIGURES_DIR / 'A_corpus_overview.png'}")

    # Console summary for sanity
    total = int(corpus_df["n"].sum())
    print(f"\nCorpus total rows: {total:,}")
    print("Expected from schema doc (pre-run): 1,452,875")
    assert total == 1_452_875, f"Row invariant broke: got {total:,}"
    print("Row invariant OK.")


if __name__ == "__main__":
    main()
