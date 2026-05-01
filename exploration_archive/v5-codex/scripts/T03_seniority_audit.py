from __future__ import annotations

import math
import os
import re
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
REPORT_DIR = ROOT / "exploration" / "reports"
TABLE_DIR = ROOT / "exploration" / "tables" / "T03"
FIG_DIR = ROOT / "exploration" / "figures" / "T03"


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


def add_share_columns(df: pd.DataFrame, group_cols: list[str], count_col: str = "n") -> pd.DataFrame:
    out = df.copy()
    totals = out.groupby(group_cols, dropna=False)[count_col].transform("sum")
    out["share"] = out[count_col] / totals
    return out


def canon_native(values: pd.Series) -> pd.Series:
    return values.replace({"intern": "entry", "executive": "director"})


def main() -> None:
    ensure_dirs()
    con = duckdb.connect()

    # Regex hygiene: test the weak-marker pattern before using it in SQL.
    weak_pat = re.compile(r"(?:\b(?:i|ii|iii|iv|v)\b|\bl[1-5]\b|\blevel\s*[1-5]\b|\bgrade\s*[1-5]\b)", re.I)
    assert weak_pat.search("Software Engineer L3")
    assert weak_pat.search("Senior Software Engineer II")
    assert weak_pat.search("Level 4 Systems Engineer")
    assert not weak_pat.search("Software Engineer")
    assert not weak_pat.search("Mechanical Engineer")

    frame = "source_platform='linkedin' AND is_english=true AND date_flag='ok'"

    # Current row counts.
    counts = q(
        con,
        f"""
        SELECT
          count(*) AS total_rows,
          count_if(source = 'kaggle_arshkon') AS arshkon_rows,
          count_if(source = 'kaggle_asaniczka') AS asaniczka_rows,
          count_if(source = 'scraped') AS scraped_rows,
          count_if(is_swe) AS swe_rows,
          count_if(is_swe_adjacent) AS swe_adjacent_rows,
          count_if(is_control) AS control_rows,
          count_if(source_platform = 'linkedin') AS linkedin_rows,
          count_if(source_platform = 'indeed') AS indeed_rows
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame}
        """
    )
    save_csv(counts, "T03_counts_overview.csv")

    swe_source_period = q(
        con,
        f"""
        SELECT source, period, count(*) AS n
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame} AND is_swe
        GROUP BY 1,2
        ORDER BY 1,2
        """
    )
    save_csv(swe_source_period, "T03_swe_counts_by_source_period.csv")

    final_source = q(
        con,
        f"""
        SELECT source, period, seniority_final_source, count(*) AS n
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame} AND is_swe
        GROUP BY 1,2,3
        ORDER BY 1,2,3
        """
    )
    final_source = add_share_columns(final_source, ["source", "period"])
    save_csv(final_source, "T03_seniority_final_source_by_source_period.csv")

    final_dist = q(
        con,
        f"""
        SELECT source, period, seniority_final, count(*) AS n
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame} AND is_swe
        GROUP BY 1,2,3
        ORDER BY 1,2,3
        """
    )
    save_csv(final_dist, "T03_seniority_final_distribution_by_source_period.csv")

    native_profile = q(
        con,
        f"""
        SELECT source, period,
               count(*) AS n,
               count_if(seniority_native IS NOT NULL) AS native_nonnull,
               count_if(seniority_native = 'entry') AS native_entry,
               count_if(seniority_native = 'associate') AS native_associate,
               count_if(seniority_native = 'mid-senior') AS native_mid_senior,
               count_if(seniority_native = 'director') AS native_director,
               count_if(seniority_native = 'intern') AS native_intern,
               count_if(seniority_native = 'executive') AS native_executive
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame} AND is_swe
        GROUP BY 1,2
        ORDER BY 1,2
        """
    )
    save_csv(native_profile, "T03_native_profile_by_source_period.csv")

    native_entry_yoe = q(
        con,
        f"""
        SELECT source,
               count(*) AS n,
               avg(yoe_extracted) AS mean_yoe,
               median(yoe_extracted) AS median_yoe,
               quantile_cont(yoe_extracted, 0.25) AS p25_yoe,
               quantile_cont(yoe_extracted, 0.75) AS p75_yoe,
               avg(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) AS share_le2,
               avg(CASE WHEN yoe_extracted <= 3 THEN 1.0 ELSE 0.0 END) AS share_le3
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame} AND is_swe AND source IN ('kaggle_arshkon', 'scraped')
          AND source_platform = 'linkedin'
          AND seniority_native = 'entry'
          AND yoe_extracted IS NOT NULL
        GROUP BY 1
        ORDER BY 1
        """
    )
    save_csv(native_entry_yoe, "T03_native_entry_yoe_profile.csv")

    final_entry_yoe = q(
        con,
        f"""
        SELECT source, period,
               count(*) AS n,
               avg(yoe_extracted) AS mean_yoe,
               median(yoe_extracted) AS median_yoe,
               avg(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) AS share_le2,
               avg(CASE WHEN yoe_extracted <= 3 THEN 1.0 ELSE 0.0 END) AS share_le3
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame} AND is_swe AND seniority_final = 'entry' AND yoe_extracted IS NOT NULL
        GROUP BY 1,2
        ORDER BY 1,2
        """
    )
    save_csv(final_entry_yoe, "T03_final_entry_yoe_profile.csv")

    unknown_yoe = q(
        con,
        f"""
        SELECT source, period,
               count(*) AS n,
               avg(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) AS share_le2,
               avg(CASE WHEN yoe_extracted <= 3 THEN 1.0 ELSE 0.0 END) AS share_le3,
               avg(yoe_extracted) AS mean_yoe,
               median(yoe_extracted) AS median_yoe
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame} AND is_swe AND seniority_final = 'unknown' AND yoe_extracted IS NOT NULL
        GROUP BY 1,2
        ORDER BY 1,2
        """
    )
    save_csv(unknown_yoe, "T03_unknown_yoe_profile.csv")

    junior_shares = q(
        con,
        f"""
        SELECT 'seniority_final' AS metric, period,
               avg(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) AS share,
               count(*) AS n
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame} AND is_swe
        GROUP BY period
        UNION ALL
        SELECT 'seniority_native_arshkon' AS metric, '2024-04' AS period,
               avg(CASE WHEN seniority_native = 'entry' THEN 1.0 ELSE 0.0 END) AS share,
               count(*) AS n
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame} AND is_swe AND source = 'kaggle_arshkon'
        UNION ALL
        SELECT 'yoe_le_2' AS metric, period,
               avg(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) AS share,
               count(*) AS n
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame} AND is_swe AND yoe_extracted IS NOT NULL
        GROUP BY period
        ORDER BY metric, period
        """
    )
    save_csv(junior_shares, "T03_junior_share_comparison.csv")

    junior_sensitivity = q(
        con,
        f"""
        SELECT 'seniority_final_no_agg' AS metric, period,
               avg(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) AS share,
               count(*) AS n
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame} AND is_swe AND NOT is_aggregator
        GROUP BY period
        UNION ALL
        SELECT 'yoe_le_2_no_agg' AS metric, period,
               avg(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) AS share,
               count(*) AS n
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame} AND is_swe AND NOT is_aggregator AND yoe_extracted IS NOT NULL
        GROUP BY period
        ORDER BY metric, period
        """
    )
    save_csv(junior_sensitivity, "T03_junior_share_sensitivity_no_aggregators.csv")

    # Confusion matrices: LinkedIn-only, raw and canonicalized native labels.
    arshkon = q(
        con,
        f"""
        SELECT seniority_native, seniority_final
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame} AND is_swe AND source = 'kaggle_arshkon' AND seniority_native IS NOT NULL
        """
    )
    scraped = q(
        con,
        f"""
        SELECT seniority_native, seniority_final
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame} AND is_swe AND source = 'scraped' AND seniority_native IS NOT NULL
        """
    )

    def confusion_df(df: pd.DataFrame, canonicalize: bool) -> pd.DataFrame:
        native = df["seniority_native"].replace({"intern": "entry", "executive": "director"}) if canonicalize else df["seniority_native"]
        mat = pd.crosstab(native, df["seniority_final"])
        mat = mat.reindex(index=["entry", "associate", "mid-senior", "director", "intern", "executive"], fill_value=0)
        if canonicalize:
            mat = mat.drop(index=["intern", "executive"])
        mat = mat.reindex(columns=["entry", "associate", "mid-senior", "director", "unknown"], fill_value=0)
        return mat

    ar_raw = confusion_df(arshkon, False)
    ar_can = confusion_df(arshkon, True)
    sc_raw = confusion_df(scraped, False)
    sc_can = confusion_df(scraped, True)

    ar_raw.to_csv(TABLE_DIR / "T03_confusion_arshkon_raw.csv")
    ar_can.to_csv(TABLE_DIR / "T03_confusion_arshkon_canonical.csv")
    sc_raw.to_csv(TABLE_DIR / "T03_confusion_scraped_raw.csv")
    sc_can.to_csv(TABLE_DIR / "T03_confusion_scraped_canonical.csv")

    metrics = pd.DataFrame(
        [
            {
                "subset": "arshkon_raw",
                "n": len(arshkon),
                "kappa": cohen_kappa_score(arshkon["seniority_native"], arshkon["seniority_final"]),
                "accuracy": (arshkon["seniority_native"] == arshkon["seniority_final"]).mean(),
            },
            {
                "subset": "arshkon_canonical",
                "n": len(arshkon),
                "kappa": cohen_kappa_score(arshkon["seniority_native"].replace({"intern": "entry", "executive": "director"}), arshkon["seniority_final"]),
                "accuracy": (
                    arshkon["seniority_native"].replace({"intern": "entry", "executive": "director"}) == arshkon["seniority_final"]
                ).mean(),
            },
            {
                "subset": "scraped_raw",
                "n": len(scraped),
                "kappa": cohen_kappa_score(scraped["seniority_native"], scraped["seniority_final"]),
                "accuracy": (scraped["seniority_native"] == scraped["seniority_final"]).mean(),
            },
            {
                "subset": "scraped_canonical",
                "n": len(scraped),
                "kappa": cohen_kappa_score(scraped["seniority_native"].replace({"intern": "entry", "executive": "director"}), scraped["seniority_final"]),
                "accuracy": (
                    scraped["seniority_native"].replace({"intern": "entry", "executive": "director"}) == scraped["seniority_final"]
                ).mean(),
            },
        ]
    )
    save_csv(metrics, "T03_native_final_agreement_metrics.csv")

    # Sample for llm-routed titles. Only 16 rows in the corpus match the weak level-code detector;
    # the remaining rows are a broader LLM-routed spot-check so the requested 100-row review is still available.
    llm_weak = q(
        con,
        f"""
        SELECT uid, source, period, title, company_name, seniority_final, seniority_native,
               yoe_extracted, seniority_final_source, 'weak_level_code' AS sample_reason
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame} AND is_swe AND seniority_final_source = 'llm'
          AND regexp_matches(lower(title_normalized), '(\\b(i|ii|iii|iv|v)\\b|\\bl[1-5]\\b|\\blevel\\s*[1-5]\\b|\\bgrade\\s*[1-5]\\b)')
        ORDER BY source, period, uid
        """
    )
    llm_pool = q(
        con,
        f"""
        SELECT uid, source, period, title, company_name, seniority_final, seniority_native,
               yoe_extracted, seniority_final_source
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {frame} AND is_swe AND seniority_final_source = 'llm'
        """
    )
    weak_ids = set(llm_weak["uid"].tolist())
    remaining = llm_pool[~llm_pool["uid"].isin(weak_ids)].copy()
    remaining["period_rank"] = remaining["period"].map({"2024-01": 0, "2024-04": 1, "2026-03": 2, "2026-04": 3})
    remaining = remaining.sort_values(["period_rank", "uid"]).drop(columns=["period_rank"])
    # deterministic sample of 84 rows balanced across periods
    samples = [llm_weak]
    needed = 100 - len(llm_weak)
    if needed > 0 and len(remaining) > 0:
        per_period = math.ceil(needed / remaining["period"].nunique())
        taken = []
        for period in ["2024-01", "2024-04", "2026-03", "2026-04"]:
            block = remaining[remaining["period"] == period]
            if block.empty:
                continue
            taken.append(block.head(per_period))
        extra = pd.concat(taken, ignore_index=True).head(needed)
        extra = extra.assign(sample_reason="broader_llm_routed_spot_check")
        samples.append(extra)
    llm_sample = pd.concat(samples, ignore_index=True)
    llm_sample = llm_sample.assign(
        title_has_level_code=llm_sample["title"].str.lower().str.contains(weak_pat.pattern, regex=True),
    )
    save_csv(llm_sample, "T03_llm_routed_sample_100.csv")

    # Figure 1: final-source composition by source/period.
    plot_df = final_source.copy()
    plot_df["group"] = plot_df["source"] + " " + plot_df["period"]
    order = ["kaggle_asaniczka 2024-01", "kaggle_arshkon 2024-04", "scraped 2026-03", "scraped 2026-04"]
    plot_df["group"] = pd.Categorical(plot_df["group"], categories=order, ordered=True)
    pivot = plot_df.pivot_table(index="group", columns="seniority_final_source", values="share", fill_value=0, observed=False).reindex(order)
    colors = {
        "title_keyword": "#4C78A8",
        "title_manager": "#F58518",
        "llm": "#54A24B",
        "unknown": "#E45756",
    }
    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = pd.Series([0.0] * len(pivot), index=pivot.index)
    for col in ["title_keyword", "title_manager", "llm", "unknown"]:
        if col in pivot.columns:
            ax.bar(pivot.index.astype(str), pivot[col], bottom=bottom, color=colors[col], label=col)
            bottom += pivot[col]
    ax.set_ylabel("Share of SWE rows")
    ax.set_title("Seniority source composition by source and period")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    save_fig(fig, "T03_seniority_source_composition.png")

    # Figure 2: junior-share comparison.
    js = junior_shares.copy()
    js["period"] = pd.Categorical(js["period"], categories=["2024-01", "2024-04", "2026-03", "2026-04"], ordered=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    palette = {"seniority_final": "#4C78A8", "yoe_le_2": "#54A24B", "seniority_native_arshkon": "#E45756"}
    for metric, grp in js.groupby("metric"):
        ax.plot(grp["period"], grp["share"], marker="o", label=metric, color=palette.get(metric, None))
    ax.set_ylabel("Entry / low-YOE share")
    ax.set_title("Junior-share comparison across seniority operationalizations")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(js["share"]) * 1.2)
    fig.tight_layout()
    save_fig(fig, "T03_junior_share_comparison.png")

    # Figure 3: native vs final confusion matrices.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, title in [
        (axes[0], ar_can, "Arshkon native vs final"),
        (axes[1], sc_can, "Scraped LinkedIn native vs final"),
    ]:
        row_pct = mat.div(mat.sum(axis=1).replace(0, math.nan), axis=0).fillna(0)
        sns.heatmap(
            row_pct,
            ax=ax,
            annot=True,
            fmt=".0%",
            cmap="Blues",
            cbar=False,
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(title)
        ax.set_xlabel("seniority_final")
        ax.set_ylabel("seniority_native")
    fig.tight_layout()
    save_fig(fig, "T03_native_vs_final_heatmaps.png")

    # Save a compact markdown summary for inspection convenience; the final report will be written separately.
    summary = {
        "total_rows": int(counts["total_rows"].iloc[0]),
        "linkedin_rows": int(counts["linkedin_rows"].iloc[0]),
        "swe_rows": int(counts["swe_rows"].iloc[0]),
        "arshkon_swe": int(swe_source_period.loc[swe_source_period["source"] == "kaggle_arshkon", "n"].sum()),
        "asaniczka_swe": int(swe_source_period.loc[swe_source_period["source"] == "kaggle_asaniczka", "n"].sum()),
        "scraped_swe": int(swe_source_period.loc[swe_source_period["source"] == "scraped", "n"].sum()),
        "weak_llm_marker_rows": int(len(llm_weak)),
        "llm_sample_rows": int(len(llm_sample)),
    }
    (TABLE_DIR / "T03_summary.json").write_text(pd.Series(summary).to_json(indent=2))


if __name__ == "__main__":
    main()
