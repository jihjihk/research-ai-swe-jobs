#!/usr/bin/env python3
"""T17 geographic market structure diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is present in the project venv today
    stats = None


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
SHARED = ROOT / "exploration" / "artifacts" / "shared"
TECH = SHARED / "swe_tech_matrix.parquet"
TAXONOMY = SHARED / "tech_taxonomy.csv"
ARCHETYPES = SHARED / "swe_archetype_labels.parquet"
COMPLEXITY = ROOT / "exploration" / "tables" / "T11" / "posting_complexity_features.parquet"
TABLE_DIR = ROOT / "exploration" / "tables" / "T17"
FIG_DIR = ROOT / "exploration" / "figures" / "T17"
SUMMARY_PATH = TABLE_DIR / "summary.json"

BROAD_AI_COLS = [
    "agents",
    "anthropic_api",
    "chatgpt",
    "claude",
    "claude_api",
    "codex",
    "copilot",
    "cursor",
    "evals",
    "fine_tuning",
    "gemini",
    "generative_ai",
    "llm",
    "mcp",
    "openai_api",
    "prompt_engineering",
    "rag",
    "langchain",
    "llamaindex",
    "hugging_face",
    "chroma",
    "pinecone",
    "weaviate",
    "vector_databases",
    "machine_learning",
    "deep_learning",
    "nlp",
    "computer_vision",
    "mlops",
    "pytorch",
    "tensorflow",
]

TECH_HUB_METROS = {
    "San Francisco Bay Area",
    "Seattle Metro",
    "New York City Metro",
    "Boston Metro",
    "Austin Metro",
    "Raleigh-Durham-Chapel Hill",
    "San Diego Metro",
    "Los Angeles Metro",
}

CHANGE_METRICS = [
    "j1_entry_share",
    "j2_entry_associate_share",
    "j3_low_yoe_share",
    "j4_low_yoe_share",
    "broad_ai_share",
    "ai_tool_strict_share",
    "org_scope_count_mean_llm",
    "requirement_breadth_mean_llm",
    "median_raw_description_length",
    "median_cleaned_length_llm",
    "tech_count_mean",
    "platform_tech_count_mean",
    "tech_diversity_cols",
]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "exploration" / "artifacts" / "tmp_duckdb").mkdir(parents=True, exist_ok=True)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=1")
    con.execute("PRAGMA temp_directory='exploration/artifacts/tmp_duckdb'")
    return con


def sum_expr(cols: Iterable[str], prefix: str = "tm") -> str:
    parts = [f"COALESCE({prefix}.{c}::INT, 0)" for c in cols]
    return " + ".join(parts) if parts else "0"


def any_expr(cols: Iterable[str], prefix: str = "tm") -> str:
    return f"(({sum_expr(cols, prefix)}) > 0)"


def tech_diversity_expr(cols: Iterable[str], prefix: str = "tm") -> str:
    parts = [f"MAX(COALESCE({prefix}.{c}::INT, 0))" for c in cols]
    return " + ".join(parts) if parts else "0"


def load_taxonomy() -> tuple[list[str], list[str], list[str], list[str]]:
    tax = pd.read_csv(TAXONOMY)
    tech_cols = tax["column"].tolist()
    broad_ai = [c for c in BROAD_AI_COLS if c in tech_cols]
    ai_tool_strict = tax.loc[
        tax["category"].eq("ai_tool") & ~tax["column"].isin(["mcp"]), "column"
    ].tolist()
    platform_cols = tax.loc[
        tax["category"].isin(["cloud", "devops", "ops", "architecture", "tooling"]),
        "column",
    ].tolist()
    return tech_cols, broad_ai, ai_tool_strict, platform_cols


def create_row_view(con: duckdb.DuckDBPyConnection) -> None:
    tech_cols, broad_ai_cols, ai_tool_cols, platform_cols = load_taxonomy()
    if not COMPLEXITY.exists():
        raise FileNotFoundError(f"Missing T11 feature artifact: {COMPLEXITY}")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW row_features AS
        SELECT
            u.uid,
            u.source,
            CASE
                WHEN u.source = 'kaggle_arshkon' THEN 'arshkon'
                WHEN u.source = 'kaggle_asaniczka' THEN 'asaniczka'
                WHEN u.source = 'scraped' THEN 'scraped_2026'
                ELSE u.source
            END AS source_group,
            CASE
                WHEN u.source IN ('kaggle_arshkon', 'kaggle_asaniczka') THEN '2024'
                WHEN u.source = 'scraped' THEN '2026'
                ELSE u.period
            END AS period_group,
            u.period,
            u.metro_area,
            u.state_normalized,
            COALESCE(u.is_multi_location, false) AS is_multi_location,
            COALESCE(u.is_remote_inferred, false) AS is_remote_inferred,
            NULLIF(u.company_name_canonical, '') AS company_name_canonical,
            COALESCE(u.is_aggregator, false) AS is_aggregator,
            u.seniority_final,
            u.seniority_3level,
            u.yoe_extracted,
            u.description_length,
            u.swe_classification_tier,
            {sum_expr(tech_cols)}::DOUBLE AS tech_count,
            {sum_expr(platform_cols)}::DOUBLE AS platform_tech_count,
            {any_expr(broad_ai_cols)} AS broad_ai_any,
            {any_expr(ai_tool_cols)} AS ai_tool_strict_any,
            cf.text_source,
            cf.char_len AS cleaned_char_len,
            cf.org_scope_count,
            cf.requirement_breadth,
            ar.archetype_name
        FROM read_parquet('{DATA.as_posix()}') AS u
        JOIN read_parquet('{TECH.as_posix()}') AS tm USING (uid)
        LEFT JOIN read_parquet('{COMPLEXITY.as_posix()}') AS cf USING (uid)
        LEFT JOIN read_parquet('{ARCHETYPES.as_posix()}') AS ar USING (uid)
        WHERE u.source_platform = 'linkedin'
          AND u.is_english = true
          AND u.date_flag = 'ok'
          AND u.is_swe = true
        """
    )


def exclusion_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    out = con.execute(
        """
        SELECT
            source_group,
            period_group,
            COUNT(*)::INTEGER AS eligible_n,
            SUM(is_multi_location::INT)::INTEGER AS multi_location_excluded_n,
            SUM(CASE WHEN NOT is_multi_location AND metro_area IS NULL THEN 1 ELSE 0 END)::INTEGER AS unresolved_non_multi_location_n,
            SUM(CASE WHEN NOT is_multi_location AND metro_area IS NOT NULL THEN 1 ELSE 0 END)::INTEGER AS metro_rollup_candidate_n
        FROM row_features
        GROUP BY 1, 2
        ORDER BY 1
        """
    ).fetchdf()
    out["multi_location_excluded_share"] = out["multi_location_excluded_n"] / out["eligible_n"]
    out.to_csv(TABLE_DIR / "metro_exclusions_multi_location.csv", index=False)
    return out


def metro_metrics(con: duckdb.DuckDBPyConnection, where_extra: str = "TRUE", cap_company: int | None = None) -> pd.DataFrame:
    tech_cols, _, _, _ = load_taxonomy()
    if cap_company is None:
        rows_cte = f"""
            SELECT *
            FROM row_features
            WHERE metro_area IS NOT NULL
              AND NOT is_multi_location
              AND {where_extra}
        """
    else:
        rows_cte = f"""
            SELECT *
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY metro_area, period_group, COALESCE(company_name_canonical, uid)
                        ORDER BY uid
                    ) AS company_row_number
                FROM row_features
                WHERE metro_area IS NOT NULL
                  AND NOT is_multi_location
                  AND {where_extra}
            )
            WHERE company_row_number <= {int(cap_company)}
        """
    query = f"""
        WITH eligible AS ({rows_cte})
        SELECT
            e.metro_area,
            e.period_group,
            COUNT(*)::INTEGER AS postings,
            COUNT(DISTINCT e.company_name_canonical)::INTEGER AS companies,
            SUM(e.is_aggregator::INT)::INTEGER AS aggregator_n,
            SUM(CASE WHEN e.yoe_extracted IS NOT NULL THEN 1 ELSE 0 END)::INTEGER AS yoe_known_n,
            SUM(CASE WHEN e.text_source = 'llm' THEN 1 ELSE 0 END)::INTEGER AS llm_text_n,
            SUM(CASE WHEN e.archetype_name IS NOT NULL THEN 1 ELSE 0 END)::INTEGER AS archetype_labeled_n,
            AVG(CASE WHEN e.seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) AS j1_entry_share,
            AVG(CASE WHEN e.seniority_final IN ('entry', 'associate') THEN 1.0 ELSE 0.0 END) AS j2_entry_associate_share,
            SUM(CASE WHEN e.yoe_extracted <= 2 THEN 1 ELSE 0 END)::DOUBLE
              / NULLIF(SUM(CASE WHEN e.yoe_extracted IS NOT NULL THEN 1 ELSE 0 END), 0) AS j3_low_yoe_share,
            SUM(CASE WHEN e.yoe_extracted <= 3 THEN 1 ELSE 0 END)::DOUBLE
              / NULLIF(SUM(CASE WHEN e.yoe_extracted IS NOT NULL THEN 1 ELSE 0 END), 0) AS j4_low_yoe_share,
            AVG(e.broad_ai_any::INT)::DOUBLE AS broad_ai_share,
            AVG(e.ai_tool_strict_any::INT)::DOUBLE AS ai_tool_strict_share,
            AVG(e.is_remote_inferred::INT)::DOUBLE AS remote_inferred_share,
            quantile_cont(e.description_length, 0.5) AS median_raw_description_length,
            quantile_cont(CASE WHEN e.text_source = 'llm' THEN e.cleaned_char_len END, 0.5) AS median_cleaned_length_llm,
            AVG(e.tech_count)::DOUBLE AS tech_count_mean,
            AVG(e.platform_tech_count)::DOUBLE AS platform_tech_count_mean,
            ({tech_diversity_expr(tech_cols)})::INTEGER AS tech_diversity_cols,
            AVG(CASE WHEN e.text_source = 'llm' THEN e.org_scope_count END)::DOUBLE AS org_scope_count_mean_llm,
            AVG(CASE WHEN e.text_source = 'llm' THEN e.requirement_breadth END)::DOUBLE AS requirement_breadth_mean_llm
        FROM eligible e
        JOIN read_parquet('{TECH.as_posix()}') tm USING (uid)
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    return con.execute(query).fetchdf()


def eligible_metros(metrics: pd.DataFrame, min_per_period: int = 50) -> list[str]:
    wide = metrics.pivot(index="metro_area", columns="period_group", values="postings")
    wide = wide.dropna(subset=["2024", "2026"], how="any")
    return sorted(wide[(wide["2024"] >= min_per_period) & (wide["2026"] >= min_per_period)].index)


def metric_changes(metrics: pd.DataFrame, metros: list[str], spec: str) -> pd.DataFrame:
    work = metrics[metrics["metro_area"].isin(metros)].copy()
    rows: list[dict[str, object]] = []
    for metro, sub in work.groupby("metro_area", dropna=False):
        wide = sub.set_index("period_group")
        if "2024" not in wide.index or "2026" not in wide.index:
            continue
        row: dict[str, object] = {
            "spec": spec,
            "metro_area": metro,
            "postings_2024": int(wide.loc["2024", "postings"]),
            "postings_2026": int(wide.loc["2026", "postings"]),
            "companies_2024": int(wide.loc["2024", "companies"]),
            "companies_2026": int(wide.loc["2026", "companies"]),
            "llm_text_n_2024": int(wide.loc["2024", "llm_text_n"]),
            "llm_text_n_2026": int(wide.loc["2026", "llm_text_n"]),
            "archetype_labeled_n_2024": int(wide.loc["2024", "archetype_labeled_n"]),
            "archetype_labeled_n_2026": int(wide.loc["2026", "archetype_labeled_n"]),
            "is_tech_hub_heuristic": metro in TECH_HUB_METROS,
        }
        for metric in CHANGE_METRICS:
            row[f"{metric}_2024"] = wide.loc["2024", metric]
            row[f"{metric}_2026"] = wide.loc["2026", metric]
            row[f"delta_{metric}"] = wide.loc["2026", metric] - wide.loc["2024", metric]
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("delta_broad_ai_share", ascending=False)
    return out


def correlations(changes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ai_metrics = ["delta_broad_ai_share", "delta_ai_tool_strict_share"]
    compare = [
        "delta_j1_entry_share",
        "delta_j3_low_yoe_share",
        "delta_requirement_breadth_mean_llm",
        "delta_tech_count_mean",
        "delta_platform_tech_count_mean",
        "postings_2024",
    ]
    for ai in ai_metrics:
        for other in compare:
            sub = changes[[ai, other]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 5:
                continue
            pearson = sub[ai].corr(sub[other], method="pearson")
            spearman = sub[ai].corr(sub[other], method="spearman")
            pearson_p = np.nan
            spearman_p = np.nan
            if stats is not None:
                pearson_p = float(stats.pearsonr(sub[ai], sub[other]).pvalue)
                spearman_p = float(stats.spearmanr(sub[ai], sub[other]).pvalue)
            rows.append(
                {
                    "ai_metric": ai,
                    "comparison_metric": other,
                    "n_metros": len(sub),
                    "pearson_r": pearson,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman,
                    "spearman_p": spearman_p,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "metro_change_correlations.csv", index=False)
    return out


def concentration_test(changes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in ["delta_broad_ai_share", "delta_ai_tool_strict_share", "delta_requirement_breadth_mean_llm"]:
        for label, sub in [
            ("tech_hub_heuristic", changes[changes["is_tech_hub_heuristic"]]),
            ("other_metros", changes[~changes["is_tech_hub_heuristic"]]),
        ]:
            rows.append(
                {
                    "metric": metric,
                    "metro_group": label,
                    "metros": len(sub),
                    "mean_delta": sub[metric].mean(),
                    "median_delta": sub[metric].median(),
                    "min_delta": sub[metric].min(),
                    "max_delta": sub[metric].max(),
                    "positive_delta_share": sub[metric].gt(0).mean(),
                }
            )
        top_quartile = changes.nlargest(max(1, len(changes) // 4), "postings_2024")
        bottom_rest = changes.drop(top_quartile.index)
        for label, sub in [("top_2024_volume_quartile", top_quartile), ("lower_2024_volume_metros", bottom_rest)]:
            rows.append(
                {
                    "metric": metric,
                    "metro_group": label,
                    "metros": len(sub),
                    "mean_delta": sub[metric].mean(),
                    "median_delta": sub[metric].median(),
                    "min_delta": sub[metric].min(),
                    "max_delta": sub[metric].max(),
                    "positive_delta_share": sub[metric].gt(0).mean(),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "metro_ai_concentration_test.csv", index=False)
    return out


def sensitivity_summary(primary: pd.DataFrame, noagg: pd.DataFrame, cap20: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, sens in [("aggregator_excluded", noagg), ("company_cap20", cap20)]:
        merged = primary.merge(sens, on="metro_area", suffixes=("_primary", f"_{label}"))
        for metric in [
            "delta_j1_entry_share",
            "delta_j3_low_yoe_share",
            "delta_broad_ai_share",
            "delta_ai_tool_strict_share",
            "delta_requirement_breadth_mean_llm",
            "delta_tech_count_mean",
        ]:
            p = merged[f"{metric}_primary"]
            s = merged[f"{metric}_{label}"]
            rel = (s - p).abs() / p.abs().replace(0, np.nan)
            flip = np.sign(s) != np.sign(p)
            material = rel.gt(0.30) | flip
            rows.append(
                {
                    "sensitivity": label,
                    "metric": metric,
                    "metros_compared": int(len(merged)),
                    "primary_mean_delta": float(p.mean()),
                    "sensitivity_mean_delta": float(s.mean()),
                    "mean_abs_relative_change": float(rel.replace([np.inf, -np.inf], np.nan).mean()),
                    "direction_flip_metros": int(flip.sum()),
                    "material_metros": int(material.sum()),
                    "material_metro_share": float(material.mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "metro_sensitivity_summary.csv", index=False)
    return out


def remote_tables(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_metro = con.execute(
        """
        SELECT
            metro_area,
            COUNT(*)::INTEGER AS scraped_2026_postings,
            AVG(is_remote_inferred::INT)::DOUBLE AS remote_inferred_share,
            SUM(is_remote_inferred::INT)::INTEGER AS remote_inferred_n
        FROM row_features
        WHERE source_group = 'scraped_2026'
          AND metro_area IS NOT NULL
          AND NOT is_multi_location
        GROUP BY 1
        HAVING COUNT(*) >= 50
        ORDER BY remote_inferred_share DESC, scraped_2026_postings DESC
        """
    ).fetchdf()
    by_metro.to_csv(TABLE_DIR / "remote_2026_by_metro.csv", index=False)
    summary = con.execute(
        """
        SELECT
            COUNT(*)::INTEGER AS scraped_2026_linkedin_swe_n,
            SUM(CASE WHEN metro_area IS NOT NULL AND NOT is_multi_location THEN 1 ELSE 0 END)::INTEGER AS metro_rollup_n,
            SUM(is_multi_location::INT)::INTEGER AS multi_location_n,
            AVG(is_remote_inferred::INT)::DOUBLE AS remote_inferred_share_all_scraped,
            AVG(CASE WHEN metro_area IS NOT NULL AND NOT is_multi_location THEN is_remote_inferred::INT END)::DOUBLE AS remote_inferred_share_metro_rollup
        FROM row_features
        WHERE source_group = 'scraped_2026'
        """
    ).fetchdf()
    summary.to_csv(TABLE_DIR / "remote_2026_summary.csv", index=False)
    return by_metro, summary


def archetype_tables(con: duckdb.DuckDBPyConnection, metros: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not metros:
        empty = pd.DataFrame()
        empty.to_csv(TABLE_DIR / "metro_archetype_distribution.csv", index=False)
        empty.to_csv(TABLE_DIR / "metro_ai_domain_adjustment.csv", index=False)
        return empty, empty
    metro_values = ",".join("'" + m.replace("'", "''") + "'" for m in metros)
    distribution = con.execute(
        f"""
        WITH labeled AS (
            SELECT *
            FROM row_features
            WHERE metro_area IN ({metro_values})
              AND NOT is_multi_location
              AND archetype_name IS NOT NULL
              AND text_source = 'llm'
        ),
        counts AS (
            SELECT metro_area, period_group, archetype_name, COUNT(*)::INTEGER AS labeled_n
            FROM labeled
            GROUP BY 1, 2, 3
        )
        SELECT
            *,
            labeled_n::DOUBLE / SUM(labeled_n) OVER (PARTITION BY metro_area, period_group) AS share_of_metro_period
        FROM counts
        ORDER BY metro_area, period_group, labeled_n DESC
        """
    ).fetchdf()
    distribution.to_csv(TABLE_DIR / "metro_archetype_distribution.csv", index=False)

    labeled_rates = con.execute(
        f"""
        WITH labeled AS (
            SELECT *
            FROM row_features
            WHERE metro_area IN ({metro_values})
              AND NOT is_multi_location
              AND archetype_name IS NOT NULL
              AND text_source = 'llm'
        ),
        national AS (
            SELECT period_group, archetype_name, AVG(broad_ai_any::INT)::DOUBLE AS national_ai_rate
            FROM labeled
            GROUP BY 1, 2
        ),
        metro_period AS (
            SELECT
                metro_area,
                period_group,
                COUNT(*)::INTEGER AS labeled_n,
                AVG(broad_ai_any::INT)::DOUBLE AS actual_ai_rate_labeled
            FROM labeled
            GROUP BY 1, 2
        ),
        comp AS (
            SELECT
                d.metro_area,
                d.period_group,
                SUM(d.share_of_metro_period * n.national_ai_rate)::DOUBLE AS predicted_ai_rate_from_archetype_mix
            FROM (
                SELECT
                    metro_area,
                    period_group,
                    archetype_name,
                    COUNT(*)::DOUBLE / SUM(COUNT(*)) OVER (PARTITION BY metro_area, period_group) AS share_of_metro_period
                FROM labeled
                GROUP BY 1, 2, 3
            ) d
            JOIN national n USING (period_group, archetype_name)
            GROUP BY 1, 2
        )
        SELECT
            m.metro_area,
            m.period_group,
            m.labeled_n,
            m.actual_ai_rate_labeled,
            c.predicted_ai_rate_from_archetype_mix,
            m.actual_ai_rate_labeled - c.predicted_ai_rate_from_archetype_mix AS residual_ai_rate_after_archetype_mix
        FROM metro_period m
        JOIN comp c USING (metro_area, period_group)
        ORDER BY metro_area, period_group
        """
    ).fetchdf()

    rows: list[dict[str, object]] = []
    for metro, sub in labeled_rates.groupby("metro_area"):
        wide = sub.set_index("period_group")
        if "2024" in wide.index and "2026" in wide.index:
            rows.append(
                {
                    "metro_area": metro,
                    "labeled_n_2024": int(wide.loc["2024", "labeled_n"]),
                    "labeled_n_2026": int(wide.loc["2026", "labeled_n"]),
                    "actual_ai_delta_labeled": wide.loc["2026", "actual_ai_rate_labeled"]
                    - wide.loc["2024", "actual_ai_rate_labeled"],
                    "predicted_ai_delta_from_archetype_mix": wide.loc[
                        "2026", "predicted_ai_rate_from_archetype_mix"
                    ]
                    - wide.loc["2024", "predicted_ai_rate_from_archetype_mix"],
                    "residual_ai_delta_after_archetype_mix": wide.loc[
                        "2026", "residual_ai_rate_after_archetype_mix"
                    ]
                    - wide.loc["2024", "residual_ai_rate_after_archetype_mix"],
                }
            )
    adjustment = pd.DataFrame(rows).sort_values("actual_ai_delta_labeled", ascending=False)
    adjustment.to_csv(TABLE_DIR / "metro_ai_domain_adjustment.csv", index=False)
    return distribution, adjustment


def seniority_panel(changes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for definition, metric in [
        ("J1", "delta_j1_entry_share"),
        ("J2", "delta_j2_entry_associate_share"),
        ("J3", "delta_j3_low_yoe_share"),
        ("J4", "delta_j4_low_yoe_share"),
    ]:
        if metric not in changes:
            continue
        vals = changes[metric].dropna()
        rows.append(
            {
                "definition": definition,
                "metros": int(len(vals)),
                "mean_delta": vals.mean(),
                "median_delta": vals.median(),
                "positive_metros": int(vals.gt(0).sum()),
                "negative_metros": int(vals.lt(0).sum()),
                "agreement_verdict": "mostly up"
                if vals.gt(0).mean() >= 0.75
                else "mostly down"
                if vals.lt(0).mean() >= 0.75
                else "mixed",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "seniority_panel_metro_changes.csv", index=False)
    return out


def write_plots(changes: pd.DataFrame, remote: pd.DataFrame) -> None:
    if not changes.empty:
        heat_metrics = [
            "delta_j1_entry_share",
            "delta_j3_low_yoe_share",
            "delta_broad_ai_share",
            "delta_ai_tool_strict_share",
            "delta_requirement_breadth_mean_llm",
            "delta_tech_count_mean",
        ]
        plot = changes.sort_values("delta_broad_ai_share", ascending=False).set_index("metro_area")[heat_metrics]
        z = (plot - plot.mean()) / plot.std(ddof=0).replace(0, np.nan)
        z = z.fillna(0)
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(z.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
        ax.set_yticks(range(len(z.index)))
        ax.set_yticklabels(z.index, fontsize=7)
        ax.set_xticks(range(len(heat_metrics)))
        ax.set_xticklabels([m.replace("delta_", "").replace("_", "\n") for m in heat_metrics], fontsize=7)
        ax.set_title("T17 metro delta heatmap, z-scored across metros")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "metro_metric_delta_heatmap.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        colors = np.where(changes["is_tech_hub_heuristic"], "#1f77b4", "#7f7f7f")
        ax.scatter(changes["delta_ai_tool_strict_share"], changes["delta_j3_low_yoe_share"], c=colors)
        for _, row in changes.nlargest(5, "delta_ai_tool_strict_share").iterrows():
            ax.annotate(row["metro_area"].replace(" Metro", ""), (row["delta_ai_tool_strict_share"], row["delta_j3_low_yoe_share"]), fontsize=7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("AI-tool-specific prevalence change")
        ax.set_ylabel("Low-YOE share change (YOE <= 2)")
        ax.set_title("T17 AI-tool change vs low-YOE change by metro")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "ai_tool_delta_vs_low_yoe_delta.png", dpi=150)
        plt.close(fig)

    if not remote.empty:
        plot = remote.head(20).sort_values("remote_inferred_share", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(plot["metro_area"], plot["remote_inferred_share"])
        ax.set_xlabel("Remote-inferred share, scraped 2026")
        ax.set_title("T17 remote share by metro, scraped 2026 only")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "remote_share_2026_by_metro.png", dpi=150)
        plt.close(fig)


def main() -> None:
    ensure_dirs()
    con = connect()
    create_row_view(con)
    exclusions = exclusion_table(con)

    primary_metrics = metro_metrics(con)
    primary_metrics.to_csv(TABLE_DIR / "metro_period_metrics_primary.csv", index=False)
    metros = eligible_metros(primary_metrics, min_per_period=50)
    changes = metric_changes(primary_metrics, metros, "primary_pooled_2024")
    changes.to_csv(TABLE_DIR / "metro_metric_changes.csv", index=False)

    noagg_metrics = metro_metrics(con, where_extra="NOT is_aggregator")
    noagg_metrics.to_csv(TABLE_DIR / "metro_period_metrics_no_aggregators.csv", index=False)
    noagg_changes = metric_changes(noagg_metrics, metros, "aggregator_excluded")
    noagg_changes.to_csv(TABLE_DIR / "metro_metric_changes_no_aggregators.csv", index=False)

    cap_metrics = metro_metrics(con, cap_company=20)
    cap_metrics.to_csv(TABLE_DIR / "metro_period_metrics_company_cap20.csv", index=False)
    cap_changes = metric_changes(cap_metrics, metros, "company_cap20")
    cap_changes.to_csv(TABLE_DIR / "metro_metric_changes_company_cap20.csv", index=False)

    arsh_metrics = metro_metrics(con, where_extra="source_group IN ('arshkon', 'scraped_2026')")
    arsh_metrics.to_csv(TABLE_DIR / "metro_period_metrics_arshkon_only.csv", index=False)
    arsh_metros = eligible_metros(arsh_metrics, min_per_period=50)
    arsh_changes = metric_changes(arsh_metrics, arsh_metros, "arshkon_only_2024")
    arsh_changes.to_csv(TABLE_DIR / "metro_metric_changes_arshkon_only.csv", index=False)

    corr = correlations(changes)
    concentration = concentration_test(changes)
    sensitivity = sensitivity_summary(changes, noagg_changes, cap_changes)
    remote, remote_summary = remote_tables(con)
    archetype_dist, archetype_adjust = archetype_tables(con, metros)
    seniority = seniority_panel(changes)
    write_plots(changes, remote)

    summary = {
        "eligible_metros_ge50_each_period": metros,
        "eligible_metro_count": len(metros),
        "arshkon_only_eligible_metro_count": len(arsh_metros),
        "multi_location_exclusions": exclusions.to_dict(orient="records"),
        "headline_ai_delta": {
            "metros_positive_broad_ai": int(changes["delta_broad_ai_share"].gt(0).sum()),
            "metros_total": int(len(changes)),
            "mean_broad_ai_delta": float(changes["delta_broad_ai_share"].mean()),
            "min_broad_ai_delta": float(changes["delta_broad_ai_share"].min()),
            "max_broad_ai_delta": float(changes["delta_broad_ai_share"].max()),
            "mean_ai_tool_delta": float(changes["delta_ai_tool_strict_share"].mean()),
        },
        "seniority_panel": seniority.to_dict(orient="records"),
        "remote_summary": remote_summary.to_dict(orient="records"),
        "sensitivity_summary": sensitivity.to_dict(orient="records"),
        "correlations": corr.to_dict(orient="records"),
        "archetype_adjustment_rows": len(archetype_adjust),
        "notes": [
            "Metro rollups exclude is_multi_location=true and do not expand rows.",
            "Remote analysis is 2026 scraped-only because 2024 remote flags are artifact zero.",
            "AI broad includes MCP; AI-tool-specific excludes MCP.",
            "Requirement breadth and org-scope use LLM text rows only.",
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote T17 tables to {TABLE_DIR}")
    print(f"Wrote T17 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
