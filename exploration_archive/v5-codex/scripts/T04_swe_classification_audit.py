from __future__ import annotations

import math
import re
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
REPORT_DIR = ROOT / "exploration" / "reports"
TABLE_DIR = ROOT / "exploration" / "tables" / "T04"
FIG_DIR = ROOT / "exploration" / "figures" / "T04"


def ensure_dirs() -> None:
    for d in [REPORT_DIR, TABLE_DIR, FIG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def q(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def save_csv(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    return path


def save_fig(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ensure_dirs()
    con = duckdb.connect()

    # Regex hygiene: test the title filters before using them in SQL.
    engineer_pat = re.compile(r"(engineer|developer|software)", re.I)
    assert engineer_pat.search("Software Engineer")
    assert engineer_pat.search("Frontend Developer")
    assert not engineer_pat.search("Accountant")

    frame = "source_platform='linkedin' AND is_english=true AND date_flag='ok'"

    counts = q(
        con,
        f"""
        SELECT
          count(*) AS total_rows,
          count_if(is_swe) AS swe_rows,
          count_if(is_swe_adjacent) AS swe_adjacent_rows,
          count_if(is_control) AS control_rows,
          count_if(is_swe AND swe_classification_tier = 'regex') AS swe_regex_rows,
          count_if(is_swe AND swe_classification_tier = 'embedding_high') AS swe_embedding_high_rows,
          count_if(is_swe AND swe_classification_tier = 'title_lookup_llm') AS swe_title_lookup_llm_rows,
          count_if(is_swe AND swe_classification_tier = 'unresolved') AS swe_unresolved_rows
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame}
        """
    )
    save_csv(counts, "T04_counts_overview.csv")

    tier_breakdown = q(
        con,
        f"""
        SELECT source, period, swe_classification_tier, count(*) AS n
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame} AND is_swe
        GROUP BY 1,2,3
        ORDER BY 1,2,3
        """
    )
    tier_breakdown["share"] = tier_breakdown["n"] / tier_breakdown.groupby(["source", "period"])["n"].transform("sum")
    save_csv(tier_breakdown, "T04_swe_tier_breakdown_by_source_period.csv")

    dual_flags = q(
        con,
        f"""
        SELECT
          sum(CASE WHEN CAST(is_swe AS INTEGER) + CAST(is_swe_adjacent AS INTEGER) + CAST(is_control AS INTEGER) > 1 THEN 1 ELSE 0 END) AS dual_flag_violations,
          sum(CASE WHEN NOT is_swe AND NOT is_swe_adjacent AND NOT is_control THEN 1 ELSE 0 END) AS unclassified_rows,
          count(*) AS total_rows
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame}
        """
    )
    save_csv(dual_flags, "T04_dual_flag_check.csv")

    adjacent_profile = q(
        con,
        f"""
        SELECT title_normalized, count(*) AS n
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame} AND is_swe_adjacent
        GROUP BY 1
        ORDER BY n DESC, title_normalized
        LIMIT 25
        """
    )
    save_csv(adjacent_profile, "T04_adjacent_top_titles.csv")

    control_profile = q(
        con,
        f"""
        SELECT title_normalized, count(*) AS n
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame} AND is_control
        GROUP BY 1
        ORDER BY n DESC, title_normalized
        LIMIT 25
        """
    )
    save_csv(control_profile, "T04_control_top_titles.csv")

    boundary_titles = [
        ("data engineer", r"data engineer"),
        ("data scientist", r"data scientist"),
        ("ml engineer", r"ml engineer|machine learning engineer"),
        ("devops/sre", r"devops|site reliability|\\bsre\\b"),
        ("qa/test engineer", r"qa|quality assurance|test engineer|test automation"),
        ("security engineer", r"security engineer|cybersecurity|cyber security"),
        ("systems architect", r"architect"),
        ("project engineer", r"project engineer"),
    ]
    boundary_rows = []
    for label, pat in boundary_titles:
        df = q(
            con,
            f"""
            SELECT period,
                   count_if(is_swe) AS swe,
                   count_if(is_swe_adjacent) AS adjacent,
                   count_if(is_control) AS control,
                   count(*) AS total
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {frame} AND regexp_matches(lower(title_normalized), '{pat}')
            GROUP BY 1
            ORDER BY 1
            """
        )
        df.insert(0, "boundary_title", label)
        boundary_rows.append(df)
    boundary_matrix = pd.concat(boundary_rows, ignore_index=True)
    save_csv(boundary_matrix, "T04_boundary_title_matrix.csv")

    # Samples for manual review.
    swe_sample = q(
        con,
        f"""
        WITH pool AS (
          SELECT uid, source, period, title, company_name, swe_confidence, swe_classification_tier,
                 left(coalesce(description, ''), 220) AS desc_excerpt,
                 row_number() OVER (PARTITION BY period ORDER BY random()) AS rn
          FROM read_parquet('{DATA.as_posix()}')
          WHERE {frame} AND is_swe
            AND (swe_confidence BETWEEN 0.3 AND 0.7 OR swe_classification_tier = 'title_lookup_llm')
        )
        SELECT *,
               CASE
                 WHEN regexp_matches(lower(title), '(data scientist|data engineer|ml engineer|machine learning engineer|devops|sre|qa|quality assurance|test engineer|security engineer|architect|automation engineer|firmware engineer)')
                   THEN 'boundary_family'
                 ELSE 'other_llm_routed'
               END AS review_family
        FROM pool
        WHERE rn <= CASE period
          WHEN '2024-01' THEN 12
          WHEN '2024-04' THEN 13
          WHEN '2026-03' THEN 13
          ELSE 12
        END
        ORDER BY period, uid
        """
    )
    save_csv(swe_sample, "T04_borderline_swe_sample_50.csv")

    non_swe_sample = q(
        con,
        f"""
        WITH pool AS (
          SELECT uid, source, period, title, company_name, swe_confidence, swe_classification_tier,
                 is_swe_adjacent, is_control,
                 left(coalesce(description, ''), 220) AS desc_excerpt,
                 row_number() OVER (PARTITION BY period ORDER BY random()) AS rn
          FROM read_parquet('{DATA.as_posix()}')
          WHERE {frame} AND NOT is_swe
            AND regexp_matches(lower(title_normalized), '(engineer|developer|software)')
        )
        SELECT *,
               CASE
                 WHEN regexp_matches(lower(title), '(data scientist|data engineer|ml engineer|machine learning engineer|devops|sre|qa|quality assurance|test engineer|security engineer|architect|automation engineer|firmware engineer|project engineer)')
                   THEN 'boundary_family'
                 WHEN regexp_matches(lower(title), '(mechanical|electrical|civil|nurse|accountant|analyst|technician)')
                   THEN 'clear_non_swe'
                 ELSE 'mixed'
               END AS review_family
        FROM pool
        WHERE rn <= CASE period
          WHEN '2024-01' THEN 12
          WHEN '2024-04' THEN 13
          WHEN '2026-03' THEN 13
          ELSE 12
        END
        ORDER BY period, uid
        """
    )
    save_csv(non_swe_sample, "T04_borderline_non_swe_sample_50.csv")

    # Figure 1: SWE tier breakdown shares by period.
    tier_plot = tier_breakdown.copy()
    tier_plot["group"] = tier_plot["source"] + " " + tier_plot["period"]
    order = ["kaggle_asaniczka 2024-01", "kaggle_arshkon 2024-04", "scraped 2026-03", "scraped 2026-04"]
    tier_plot["group"] = pd.Categorical(tier_plot["group"], categories=order, ordered=True)
    pivot = tier_plot.pivot_table(index="group", columns="swe_classification_tier", values="share", fill_value=0, observed=False).reindex(order)
    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = pd.Series([0.0] * len(pivot), index=pivot.index)
    palette = {
        "regex": "#4C78A8",
        "embedding_high": "#54A24B",
        "title_lookup_llm": "#F58518",
        "unresolved": "#E45756",
    }
    for col in ["regex", "embedding_high", "title_lookup_llm", "unresolved"]:
        if col in pivot.columns:
            ax.bar(pivot.index.astype(str), pivot[col], bottom=bottom, color=palette[col], label=col)
            bottom += pivot[col]
    ax.set_ylabel("Share of SWE rows")
    ax.set_title("SWE classification tier composition by source and period")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    save_fig(fig, "T04_swe_tier_composition.png")

    # Figure 2: top adjacent/control titles.
    adj = adjacent_profile.head(10).assign(group="adjacent")
    ctrl = control_profile.head(10).assign(group="control")
    combo = pd.concat([adj, ctrl], ignore_index=True)
    combo["label"] = combo["group"] + ": " + combo["title_normalized"]
    fig, ax = plt.subplots(figsize=(11, 7))
    subset = combo.sort_values(["group", "n"], ascending=[True, False])
    sns.barplot(
        data=subset,
        y="label",
        x="n",
        hue="group",
        dodge=False,
        ax=ax,
        palette={"adjacent": "#4C78A8", "control": "#E45756"},
    )
    ax.set_title("Most common adjacent and control titles")
    ax.set_xlabel("Rows")
    ax.set_ylabel("")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, "T04_adjacent_control_top_titles.png")


if __name__ == "__main__":
    main()
