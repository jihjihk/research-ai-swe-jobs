from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
STAGE8 = ROOT / "preprocessing" / "intermediate" / "stage8_final.parquet"

T01_DIR = ROOT / "exploration" / "tables" / "T01"
T03_DIR = ROOT / "exploration" / "tables" / "T03"
FIG_T01_DIR = ROOT / "exploration" / "figures" / "T01"
FIG_T03_DIR = ROOT / "exploration" / "figures" / "T03"
REPORT_T01 = ROOT / "exploration" / "reports" / "T01.md"
REPORT_T03 = ROOT / "exploration" / "reports" / "T03.md"


def ensure_dirs() -> None:
    for path in [
        T01_DIR,
        T03_DIR,
        FIG_T01_DIR,
        FIG_T03_DIR,
        ROOT / "exploration" / "reports",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def get_columns(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{STAGE8.as_posix()}')"
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def short_value(value: object, max_len: int = 80) -> str:
    if value is None:
        return "NULL"
    text = str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


def build_group_filter(mode: str) -> str:
    if mode == "T01":
        return "WHERE source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'"
    if mode == "T03":
        return ""
    raise ValueError(mode)


def build_base_cte(mode: str, column: str) -> str:
    group_filter = build_group_filter(mode)
    if mode == "T01":
        return f"""
        WITH base AS (
            SELECT source, 'all'::VARCHAR AS subset, {column} AS v
            FROM read_parquet('{STAGE8.as_posix()}')
            {group_filter}
            UNION ALL
            SELECT source, 'swe'::VARCHAR AS subset, {column} AS v
            FROM read_parquet('{STAGE8.as_posix()}')
            {group_filter} AND is_swe = true
        )
        """
    if mode == "T03":
        return f"""
        WITH base AS (
            SELECT source, source_platform, 'all'::VARCHAR AS subset, {column} AS v
            FROM read_parquet('{STAGE8.as_posix()}')
            UNION ALL
            SELECT source, source_platform, 'swe'::VARCHAR AS subset, {column} AS v
            FROM read_parquet('{STAGE8.as_posix()}')
            WHERE is_swe = true
        )
        """
    raise ValueError(mode)


def metrics_query(mode: str, column: str) -> str:
    base = build_base_cte(mode, column)
    if mode == "T01":
        return base + """
        SELECT
            source,
            subset,
            count(*) AS n_rows,
            sum(CASE WHEN v IS NOT NULL THEN 1 ELSE 0 END) AS non_null_n,
            count(DISTINCT v) AS distinct_n
        FROM base
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    return base + """
        SELECT
            source,
            source_platform,
            subset,
            count(*) AS n_rows,
            sum(CASE WHEN v IS NOT NULL THEN 1 ELSE 0 END) AS non_null_n,
            count(DISTINCT v) AS distinct_n
        FROM base
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
        """


def top_values_query(mode: str, column: str) -> str:
    base = build_base_cte(mode, column)
    if mode == "T01":
        return base + """
        , counts AS (
            SELECT source, subset, CAST(v AS VARCHAR) AS value, count(*) AS cnt
            FROM base
            WHERE v IS NOT NULL
            GROUP BY 1, 2, 3
        ),
        ranked AS (
            SELECT
                source,
                subset,
                value,
                cnt,
                row_number() OVER (PARTITION BY source, subset ORDER BY cnt DESC, value) AS rn
            FROM counts
        )
        SELECT source, subset, value, cnt, rn
        FROM ranked
        WHERE rn <= 5
        ORDER BY 1, 2, rn
        """
    return base + """
    , counts AS (
        SELECT source, source_platform, subset, CAST(v AS VARCHAR) AS value, count(*) AS cnt
        FROM base
        WHERE v IS NOT NULL
        GROUP BY 1, 2, 3, 4
    ),
    ranked AS (
        SELECT
            source,
            source_platform,
            subset,
            value,
            cnt,
            row_number() OVER (PARTITION BY source, source_platform, subset ORDER BY cnt DESC, value) AS rn
        FROM counts
    )
    SELECT source, source_platform, subset, value, cnt, rn
    FROM ranked
    WHERE rn <= 5
    ORDER BY 1, 2, 3, rn
    """


def run_column_audit(
    con: duckdb.DuckDBPyConnection, mode: str, columns: Iterable[tuple[str, str]]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, (column, dtype) in enumerate(columns, start=1):
        mdf = con.execute(metrics_query(mode, column)).fetchdf()
        tdf = con.execute(top_values_query(mode, column)).fetchdf()

        if mode == "T01":
            key_cols = ["source", "subset"]
        else:
            key_cols = ["source", "source_platform", "subset"]

        grouped: dict[tuple[object, ...], list[str]] = {}
        for _, r in tdf.iterrows():
            key = tuple(r[c] for c in key_cols)
            grouped.setdefault(key, []).append(f"{short_value(r['value'])} ({int(r['cnt'])})")

        for _, r in mdf.iterrows():
            key = tuple(r[c] for c in key_cols)
            top_values = grouped.get(key, [])
            rows.append(
                {
                    "column": column,
                    "dtype": dtype,
                    **{k: r[k] for k in key_cols},
                    "n_rows": int(r["n_rows"]),
                    "non_null_n": int(r["non_null_n"]),
                    "non_null_rate": float(r["non_null_n"]) / float(r["n_rows"]) if int(r["n_rows"]) else 0.0,
                    "null_rate": 1.0 - (float(r["non_null_n"]) / float(r["n_rows"]) if int(r["n_rows"]) else 0.0),
                    "distinct_n": int(r["distinct_n"]),
                    "top_5_values": json.dumps(top_values, ensure_ascii=True),
                }
            )
        print(f"[{mode}] completed {idx}/{len(list(columns)) if hasattr(columns, '__len__') else '?'}: {column}")
    return pd.DataFrame(rows)


def make_heatmap(t01_df: pd.DataFrame) -> None:
    # Use the all-rows subset for the coverage map and keep SWE in a second panel.
    order = (
        t01_df[t01_df["subset"] == "all"]
        .groupby("column")["non_null_rate"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    source_order = ["kaggle_asaniczka", "kaggle_arshkon", "scraped"]
    panel_all = (
        t01_df[t01_df["subset"] == "all"]
        .pivot(index="column", columns="source", values="non_null_rate")
        .reindex(order)
        .reindex(columns=source_order)
    )
    panel_swe = (
        t01_df[t01_df["subset"] == "swe"]
        .pivot(index="column", columns="source", values="non_null_rate")
        .reindex(order)
        .reindex(columns=source_order)
    )

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(14, max(16, len(order) * 0.18)), constrained_layout=True)
    for ax, panel, title in zip(axes, [panel_all, panel_swe], ["All rows", "SWE subset"]):
        sns.heatmap(
            panel,
            ax=ax,
            cmap="viridis",
            vmin=0,
            vmax=1,
            cbar=ax is axes[1],
            cbar_kws={"label": "Non-null rate"} if ax is axes[1] else None,
        )
        ax.set_title(title)
        ax.set_xlabel("Source")
        ax.set_ylabel("Column")
        ax.tick_params(axis="y", labelsize=6)
        ax.tick_params(axis="x", labelrotation=25)
    fig.suptitle("Stage 8 coverage by source", fontsize=14)
    fig.savefig(FIG_T01_DIR / "coverage_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_t03_heatmap(t03_df: pd.DataFrame) -> None:
    # Focus the missingness heatmap on the LinkedIn subset because that is the
    # cross-period comparison frame, but keep the raw table covering all platform rows.
    linkedin = t03_df[(t03_df["source_platform"] == "linkedin") & (t03_df["subset"] == "all")]
    order = linkedin.groupby("column")["non_null_rate"].mean().sort_values(ascending=False).index.tolist()
    source_order = ["kaggle_asaniczka", "kaggle_arshkon", "scraped"]
    panel = (
        linkedin.pivot(index="column", columns="source", values="non_null_rate")
        .reindex(order)
        .reindex(columns=source_order)
    )

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(10, max(16, len(order) * 0.18)), constrained_layout=True)
    sns.heatmap(panel, ax=ax, cmap="magma", vmin=0, vmax=1, cbar_kws={"label": "Non-null rate"})
    ax.set_title("LinkedIn-only missingness by source")
    ax.set_xlabel("Source")
    ax.set_ylabel("Column")
    ax.tick_params(axis="y", labelsize=6)
    ax.tick_params(axis="x", labelrotation=25)
    fig.savefig(FIG_T03_DIR / "linkedin_missingness_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize_top_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["column", "subset", "non_null_rate"], ascending=[True, True, False]).copy()


def field_summary_tables(df: pd.DataFrame, mode: str) -> None:
    if mode == "T01":
        group_cols = ["source", "subset"]
        wide = df[df["subset"] == "all"].pivot(index="column", columns="source", values="non_null_rate")
        flag = (
            df[df["subset"] == "all"]
            .groupby("column")["null_rate"]
            .max()
            .reset_index()
            .query("null_rate > 0.5")
            .sort_values("null_rate", ascending=False)
        )
        flag.to_csv(T01_DIR / "columns_over_50pct_null_any_source.csv", index=False)
    else:
        group_cols = ["source", "source_platform", "subset"]
        wide = df[(df["source_platform"] == "linkedin") & (df["subset"] == "all")].pivot(
            index="column", columns="source", values="non_null_rate"
        )
        flag = (
            df[df["subset"] == "all"]
            .groupby(["column", "source", "source_platform"])["null_rate"]
            .max()
            .reset_index()
            .query("null_rate > 0.5")
            .sort_values("null_rate", ascending=False)
        )
        flag.to_csv(T03_DIR / "columns_over_50pct_null_any_source_platform.csv", index=False)


def write_rq_tables(t01_df: pd.DataFrame) -> None:
    def coverage_for(cols: list[str]) -> pd.DataFrame:
        sub = t01_df[(t01_df["subset"] == "all") & (t01_df["column"].isin(cols))]
        agg = (
            sub.groupby("column")
            .agg(
                min_non_null_rate=("non_null_rate", "min"),
                avg_non_null_rate=("non_null_rate", "mean"),
                max_null_rate=("null_rate", "max"),
            )
            .reset_index()
        )
        return agg

    rq_rows = [
        {
            "rq": "RQ1",
            "analysis_need": "employer-side restructuring",
            "recommended_fields": ", ".join(
                [
                    "period",
                    "source",
                    "company_name_effective",
                    "company_name_canonical",
                    "seniority_final",
                    "seniority_final_source",
                    "is_swe",
                    "title_normalized",
                    "description_core",
                    "company_industry",
                    "company_size",
                    "metro_area",
                    "is_remote",
                ]
            ),
            "coverage_note": "Company, seniority, and geography fields are mostly usable; description_core remains rule-based only.",
        },
        {
            "rq": "RQ2",
            "analysis_need": "task and requirement migration",
            "recommended_fields": ", ".join(
                [
                    "title_normalized",
                    "description_core",
                    "description",
                    "skills_raw",
                    "asaniczka_skills",
                    "yoe_extracted",
                    "seniority_final",
                    "is_swe",
                    "is_swe_adjacent",
                    "is_control",
                ]
            ),
            "coverage_note": "Text fields are high coverage; requirement-level fields are strongest where description text is present and non-empty.",
        },
        {
            "rq": "RQ3",
            "analysis_need": "employer-requirement / worker-usage divergence",
            "recommended_fields": ", ".join(
                [
                    "description",
                    "description_core",
                    "skills_raw",
                    "yoe_extracted",
                    "ghost_job_risk",
                    "date_posted",
                    "period",
                    "source",
                    "is_swe",
                ]
            ),
            "coverage_note": "No LLM augmentation columns are present in stage 8, so divergence analysis still depends on rule-based proxies.",
        },
        {
            "rq": "RQ4",
            "analysis_need": "qualitative mechanism inference",
            "recommended_fields": "N/A",
            "coverage_note": "RQ4 depends on interview artifacts rather than the job-posting parquet.",
        },
    ]
    pd.DataFrame(rq_rows).to_csv(T01_DIR / "usable_columns_by_rq.csv", index=False)


def title_overlap(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    q = f"""
    WITH base AS (
        SELECT source, seniority_native, title_normalized
        FROM read_parquet('{STAGE8.as_posix()}')
        WHERE source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'
    ),
    asan AS (
        SELECT DISTINCT title_normalized
        FROM base
        WHERE source = 'kaggle_asaniczka' AND seniority_native = 'associate' AND title_normalized IS NOT NULL
    ),
    arsh AS (
        SELECT DISTINCT title_normalized
        FROM base
        WHERE source = 'kaggle_arshkon' AND seniority_native = 'entry' AND title_normalized IS NOT NULL
    )
    SELECT title_normalized
    FROM asan
    INNER JOIN arsh USING (title_normalized)
    ORDER BY title_normalized
    """
    overlap = con.execute(q).fetchdf()

    top_shared = con.execute(
        f"""
        WITH base AS (
            SELECT source, seniority_native, title_normalized
            FROM read_parquet('{STAGE8.as_posix()}')
            WHERE source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'
        )
        SELECT title_normalized,
               sum(CASE WHEN source = 'kaggle_asaniczka' AND seniority_native = 'associate' THEN 1 ELSE 0 END) AS asan_associate_n,
               sum(CASE WHEN source = 'kaggle_arshkon' AND seniority_native = 'entry' THEN 1 ELSE 0 END) AS arsh_entry_n
        FROM base
        WHERE title_normalized IN (
            SELECT title_normalized
            FROM base
            WHERE source = 'kaggle_asaniczka' AND seniority_native = 'associate' AND title_normalized IS NOT NULL
            INTERSECT
            SELECT title_normalized
            FROM base
            WHERE source = 'kaggle_arshkon' AND seniority_native = 'entry' AND title_normalized IS NOT NULL
        )
        GROUP BY 1
        ORDER BY (asan_associate_n + arsh_entry_n) DESC, title_normalized
        LIMIT 20
        """
    ).fetchdf()

    summary = pd.DataFrame(
        [
            {
                "asaniczka_associate_unique_titles": int(
                    con.execute(
                        f"""
                        SELECT count(DISTINCT title_normalized)
                        FROM read_parquet('{STAGE8.as_posix()}')
                        WHERE source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'
                          AND source = 'kaggle_asaniczka' AND seniority_native = 'associate' AND title_normalized IS NOT NULL
                        """
                    ).fetchone()[0]
                ),
                "arshkon_entry_unique_titles": int(
                    con.execute(
                        f"""
                        SELECT count(DISTINCT title_normalized)
                        FROM read_parquet('{STAGE8.as_posix()}')
                        WHERE source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'
                          AND source = 'kaggle_arshkon' AND seniority_native = 'entry' AND title_normalized IS NOT NULL
                        """
                    ).fetchone()[0]
                ),
                "shared_titles": int(len(overlap)),
                "jaccard_unique_titles": float(len(overlap))
                / float(
                    con.execute(
                        f"""
                        WITH asan AS (
                            SELECT DISTINCT title_normalized
                            FROM read_parquet('{STAGE8.as_posix()}')
                            WHERE source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'
                              AND source = 'kaggle_asaniczka' AND seniority_native = 'associate' AND title_normalized IS NOT NULL
                        ),
                        arsh AS (
                            SELECT DISTINCT title_normalized
                            FROM read_parquet('{STAGE8.as_posix()}')
                            WHERE source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'
                              AND source = 'kaggle_arshkon' AND seniority_native = 'entry' AND title_normalized IS NOT NULL
                        )
                        SELECT count(*) FROM (SELECT title_normalized FROM asan UNION SELECT title_normalized FROM arsh)
                        """
                    ).fetchone()[0]
                )
                if len(overlap)
                else 0.0,
            }
        ]
    )
    summary.to_csv(T03_DIR / "asaniczka_associate_vs_arshkon_entry_title_overlap.csv", index=False)
    top_shared.to_csv(T03_DIR / "asaniczka_associate_vs_arshkon_entry_shared_titles.csv", index=False)
    return summary, top_shared, overlap


def seniority_support(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    q = f"""
    SELECT source, source_platform, seniority_native, count(*) AS n
    FROM read_parquet('{STAGE8.as_posix()}')
    GROUP BY 1, 2, 3
    ORDER BY 1, 2, n DESC NULLS LAST, seniority_native
    """
    df = con.execute(q).fetchdf()
    df.to_csv(T03_DIR / "seniority_native_by_source_platform.csv", index=False)
    return df


def effective_sample_sizes(t01_df: pd.DataFrame) -> pd.DataFrame:
    # Non-null row counts after default filtering, by source and subset.
    df = t01_df[["column", "source", "subset", "n_rows", "non_null_n", "non_null_rate"]].copy()
    df.to_csv(T01_DIR / "effective_sample_sizes_by_field_source_subset.csv", index=False)
    return df


def write_reports(
    t01_df: pd.DataFrame,
    t03_df: pd.DataFrame,
    t03_summary: pd.DataFrame,
    t03_top_shared: pd.DataFrame,
    t03_overlap: pd.DataFrame,
    seniority_df: pd.DataFrame,
    cols: list[tuple[str, str]],
) -> None:
    # T01 summary
    low_coverage = (
        t01_df[t01_df["subset"] == "all"]
        .groupby("column")["null_rate"]
        .max()
        .reset_index()
        .query("null_rate > 0.5")
        .sort_values("null_rate", ascending=False)
    )
    desc_cov = t01_df[t01_df["column"].isin(["description_core", "description"])]
    desc_llm_exists = any(c == "description_core_llm" for c, _ in cols)
    desc_llm_cov = "absent"
    if desc_llm_exists:
        desc_llm_cov = str(t01_df[t01_df["column"] == "description_core_llm"]["non_null_rate"].max())

    t01_lines = [
        "# T01: Column coverage audit",
        "## Finding",
        "Stage 8 has no `description_core_llm`, so the LLM boilerplate-removal path is not yet available in this artifact; the rule-based `description_core` is the only stripped-text field. Coverage is strong for core identity, title, company, and seniority fields in the LinkedIn-only analysis frame, but several geography and skill-adjacent fields are sparse enough to require source-aware handling.",
        "## Implication for analysis",
        "RQ1 and RQ2 can proceed with `company_name_effective`, `company_name_canonical`, `title_normalized`, `seniority_final`, and the core description fields. RQ3 remains proxy-based until LLM augmentation lands, and RQ4 is outside the parquet. Cross-period work should avoid columns flagged below 50% null in at least one source.",
        "## Data quality note",
        f"`description_core_llm` is {'present' if desc_llm_exists else 'absent'} in stage 8. The default LinkedIn-only audit shows {len(low_coverage)} columns with >50% null in at least one source; those fields are concentrated in search metadata, some geography fields, and sparse company attributes.",
        "## Action items",
        "Use the coverage heatmap and the low-coverage CSV as the gate for downstream field selection. Treat `description_core` as temporary until Stage 11 produces `description_core_llm`.",
        "",
        "### Output files",
        f"- `{(FIG_T01_DIR / 'coverage_heatmap.png').as_posix()}`",
        f"- `{(T01_DIR / 'coverage_metrics_by_source_and_subset.csv').as_posix()}`",
        f"- `{(T01_DIR / 'columns_over_50pct_null_any_source.csv').as_posix()}`",
        f"- `{(T01_DIR / 'usable_columns_by_rq.csv').as_posix()}`",
        f"- `{(T01_DIR / 'effective_sample_sizes_by_field_source_subset.csv').as_posix()}`",
    ]
    REPORT_T01.write_text("\n".join(t01_lines) + "\n")

    # T03 summary
    arsh = seniority_df[(seniority_df["source"] == "kaggle_arshkon") & (seniority_df["source_platform"] == "linkedin") & seniority_df["seniority_native"].notna()]
    asan = seniority_df[(seniority_df["source"] == "kaggle_asaniczka") & (seniority_df["source_platform"] == "linkedin") & seniority_df["seniority_native"].notna()]
    scr_link = seniority_df[(seniority_df["source"] == "scraped") & (seniority_df["source_platform"] == "linkedin") & seniority_df["seniority_native"].notna()]
    scr_indeed = seniority_df[(seniority_df["source"] == "scraped") & (seniority_df["source_platform"] == "indeed") & seniority_df["seniority_native"].notna()]

    t03_lines = [
        "# T03: Missing data audit",
        "## Finding",
        "Missingness is highly source- and platform-specific. LinkedIn Kaggle sources are the cleanest cross-period comparison frame, while scraped Indeed rows are structurally different and should be excluded from any analysis that depends on LinkedIn-style metadata. Asaniczka provides only `associate` and `mid-senior` native seniority labels; arshkon and scraped LinkedIn also provide entry, director, intern, and executive labels.",
        "## Implication for analysis",
        "Cross-period analysis should stay on the LinkedIn-only frame and use `seniority_final` with explicit source stratification. Any comparison of title distributions between asaniczka associate and arshkon entry needs to be treated as only partial overlap, not a direct one-to-one mapping.",
        "## Data quality note",
        f"The LinkedIn-only title overlap check found {int(t03_summary['shared_titles'].iloc[0])} exact shared `title_normalized` values between asaniczka associate and arshkon entry. That overlap is non-zero, but the title spaces are not identical; the exact shared-title table should be read as a conservative overlap measure. Scraped Indeed rows have a separate platform profile and should not be mixed into LinkedIn cross-period comparisons.",
        "## Action items",
        "Use the source-platform missingness table before choosing any field for RQ1-RQ3. For seniority work, keep a separate note that asaniczka cannot support an entry-level baseline from native labels alone.",
        "",
        "### Seniority support by source",
        f"- arshkon LinkedIn native labels: {', '.join(sorted([x for x in arsh['seniority_native'].dropna().unique().tolist()]))}",
        f"- asaniczka LinkedIn native labels: {', '.join(sorted([x for x in asan['seniority_native'].dropna().unique().tolist()]))}",
        f"- scraped LinkedIn native labels: {', '.join(sorted([x for x in scr_link['seniority_native'].dropna().unique().tolist()]))}",
        f"- scraped Indeed native labels: {', '.join(sorted([x for x in scr_indeed['seniority_native'].dropna().unique().tolist()]) if len(scr_indeed) else [])}",
        "",
        "### Output files",
        f"- `{(FIG_T03_DIR / 'linkedin_missingness_heatmap.png').as_posix()}`",
        f"- `{(T03_DIR / 'missingness_metrics_by_source_platform_and_subset.csv').as_posix()}`",
        f"- `{(T03_DIR / 'seniority_native_by_source_platform.csv').as_posix()}`",
        f"- `{(T03_DIR / 'asaniczka_associate_vs_arshkon_entry_title_overlap.csv').as_posix()}`",
        f"- `{(T03_DIR / 'asaniczka_associate_vs_arshkon_entry_shared_titles.csv').as_posix()}`",
    ]
    REPORT_T03.write_text("\n".join(t03_lines) + "\n")


def main() -> None:
    ensure_dirs()
    con = duckdb.connect()
    cols = get_columns(con)

    t01_df = run_column_audit(con, "T01", cols)
    t01_df.to_csv(T01_DIR / "coverage_metrics_by_source_and_subset.csv", index=False)
    make_heatmap(t01_df)
    field_summary_tables(t01_df, "T01")
    write_rq_tables(t01_df)
    effective_sample_sizes(t01_df)

    t03_df = run_column_audit(con, "T03", cols)
    t03_df.to_csv(T03_DIR / "missingness_metrics_by_source_platform_and_subset.csv", index=False)
    make_t03_heatmap(t03_df)
    field_summary_tables(t03_df, "T03")
    seniority_df = seniority_support(con)
    t03_summary, t03_top_shared, t03_overlap = title_overlap(con)

    # Keep a compact field summary for the report authoring stage.
    t03_df[["column", "source", "source_platform", "subset", "non_null_rate"]].to_csv(
        T03_DIR / "missingness_compact.csv", index=False
    )

    write_reports(t01_df, t03_df, t03_summary, t03_top_shared, t03_overlap, seniority_df, cols)


if __name__ == "__main__":
    main()
