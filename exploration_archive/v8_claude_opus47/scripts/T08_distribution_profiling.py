"""T08 — Distribution profiling and anomaly detection.

Primary outputs (SWE, LinkedIn-only, default filter):
  - Univariate distributions of description_length, yoe_extracted,
    seniority_final, seniority_3level, is_aggregator, metro_area, industry.
  - Arshkon native-label YOE diagnostic.
  - T30 panel junior/senior share table (2 baselines x 4 junior / 4 senior vars).
  - Ranked-change table of all binary/share metrics and continuous metrics
    with Cohen's d, cross-period absolute effect, and SNR (from
    calibration_table.csv where available, or computed inline).
  - Domain-proxy (backend/frontend/ml/...) x period decomposition of J2.
  - Company-size-quartile stratification (arshkon) of J2 share, AI prevalence
    and tech count.
  - Specialist-exclusion + aggregator-exclusion sensitivity on headline
    metrics.

Runs entirely through DuckDB + pyarrow; never loads the parquet into pandas.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
PARQUET = ROOT / "data" / "unified.parquet"
SHARED = ROOT / "exploration" / "artifacts" / "shared"
TABLE_DIR = ROOT / "exploration" / "tables" / "T08"
FIG_DIR = ROOT / "exploration" / "figures" / "T08"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_FILTER = (
    "source_platform='linkedin' AND is_english=true AND date_flag='ok' "
    "AND is_swe=true"
)

PERIOD_LABELS = {
    "kaggle_arshkon": "arshkon (2024-04)",
    "kaggle_asaniczka": "asaniczka (2024-01)",
    "scraped_2026": "scraped (2026-03/04)",
    "pooled_2024": "pooled 2024",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def period_case(col: str = "source") -> str:
    """SQL CASE to collapse source+period into arshkon / asaniczka / scraped_2026."""
    return (
        "CASE WHEN source='kaggle_arshkon' THEN 'arshkon' "
        "WHEN source='kaggle_asaniczka' THEN 'asaniczka' "
        "WHEN source='scraped' THEN 'scraped_2026' END"
    )


def cohen_d(mean_a: float, mean_b: float, var_a: float, var_b: float,
            n_a: int, n_b: int) -> float:
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled == 0 or np.isnan(pooled):
        return float("nan")
    return (mean_b - mean_a) / pooled


def share_diff_snr(effect_cross: float, effect_within: float) -> float:
    """SNR = |cross| / |within|, unsigned."""
    if abs(effect_within) < 1e-9:
        return float("inf")
    return abs(effect_cross) / abs(effect_within)


def save_table(df: pd.DataFrame, name: str) -> Path:
    p = TABLE_DIR / name
    df.to_csv(p, index=False)
    print(f"  wrote {p.relative_to(ROOT)}  ({len(df)} rows)")
    return p


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4; PRAGMA memory_limit='20GB';")
    # Register specialist employer list for reuse as SQL CTE.
    specialists_path = SHARED / "entry_specialist_employers.csv"
    specialists = pd.read_csv(specialists_path, usecols=["company"])
    # company names in specialists list are company_name_canonical or close
    con.register("specialists", specialists)
    return con


# ---------------------------------------------------------------------------
# Step 0: load specialists and panel
# ---------------------------------------------------------------------------


def load_specialists() -> set[str]:
    df = pd.read_csv(SHARED / "entry_specialist_employers.csv", usecols=["company"])
    return set(df["company"].dropna().astype(str).tolist())


def load_calibration() -> pd.DataFrame:
    return pd.read_csv(SHARED / "calibration_table.csv")


# ---------------------------------------------------------------------------
# Step 1: Univariate profiles
# ---------------------------------------------------------------------------


def univariate_profile(con: duckdb.DuckDBPyConnection) -> None:
    print("[STEP 1] Univariate profiling")

    # Overall counts per (source_bucket, seniority_final)
    q = f"""
    SELECT {period_case()} AS bucket, seniority_final,
           COUNT(*) AS n
    FROM '{PARQUET}'
    WHERE {DEFAULT_FILTER}
    GROUP BY ALL ORDER BY bucket, seniority_final
    """
    df = con.execute(q).fetchdf()
    save_table(df, "univariate_seniority_counts.csv")

    # Description length distribution summary
    q = f"""
    SELECT {period_case()} AS bucket, seniority_final,
           COUNT(*) AS n,
           AVG(description_length) AS mean_len,
           APPROX_QUANTILE(description_length, 0.25) AS q25,
           APPROX_QUANTILE(description_length, 0.5) AS median,
           APPROX_QUANTILE(description_length, 0.75) AS q75,
           AVG(description_length*description_length) - AVG(description_length)*AVG(description_length) AS var_len
    FROM '{PARQUET}'
    WHERE {DEFAULT_FILTER}
    GROUP BY ALL ORDER BY bucket, seniority_final
    """
    df = con.execute(q).fetchdf()
    save_table(df, "desc_length_by_bucket_seniority.csv")

    # YOE extracted distribution summary (where extracted is non-null)
    q = f"""
    SELECT {period_case()} AS bucket, seniority_final,
           COUNT(*) AS n_known,
           AVG(yoe_extracted) AS mean_yoe,
           APPROX_QUANTILE(yoe_extracted, 0.25) AS q25,
           APPROX_QUANTILE(yoe_extracted, 0.5) AS median,
           APPROX_QUANTILE(yoe_extracted, 0.75) AS q75,
           AVG(CASE WHEN yoe_extracted<=2 THEN 1.0 ELSE 0.0 END) AS share_le2,
           AVG(CASE WHEN yoe_extracted>=5 THEN 1.0 ELSE 0.0 END) AS share_ge5
    FROM '{PARQUET}'
    WHERE {DEFAULT_FILTER} AND yoe_extracted IS NOT NULL
    GROUP BY ALL ORDER BY bucket, seniority_final
    """
    df = con.execute(q).fetchdf()
    save_table(df, "yoe_extracted_by_bucket_seniority.csv")

    # Metro top-15 share by bucket
    q = f"""
    WITH counts AS (
      SELECT {period_case()} AS bucket, metro_area,
             COUNT(*) AS n
      FROM '{PARQUET}'
      WHERE {DEFAULT_FILTER} AND metro_area IS NOT NULL
      GROUP BY ALL
    ), totals AS (
      SELECT bucket, SUM(n) AS total FROM counts GROUP BY bucket
    )
    SELECT c.bucket, c.metro_area, c.n, c.n*1.0/t.total AS share
    FROM counts c JOIN totals t USING (bucket)
    """
    df = con.execute(q).fetchdf()
    # Keep top 15 by within-bucket share
    top = (df.sort_values(["bucket", "share"], ascending=[True, False])
             .groupby("bucket", group_keys=False).head(15))
    save_table(top, "metro_top15_by_bucket.csv")

    # Industry top-15 (arshkon + scraped only, asaniczka has none)
    q = f"""
    WITH counts AS (
      SELECT {period_case()} AS bucket, company_industry,
             COUNT(*) AS n
      FROM '{PARQUET}'
      WHERE {DEFAULT_FILTER} AND company_industry IS NOT NULL
      GROUP BY ALL
    ), totals AS (
      SELECT bucket, SUM(n) AS total FROM counts GROUP BY bucket
    )
    SELECT c.bucket, c.company_industry, c.n, c.n*1.0/t.total AS share
    FROM counts c JOIN totals t USING (bucket)
    """
    df = con.execute(q).fetchdf()
    top = (df.sort_values(["bucket", "share"], ascending=[True, False])
             .groupby("bucket", group_keys=False).head(15))
    save_table(top, "industry_top15_by_bucket.csv")

    # Aggregator share
    q = f"""
    SELECT {period_case()} AS bucket,
           AVG(CASE WHEN is_aggregator THEN 1.0 ELSE 0.0 END) AS agg_share,
           COUNT(*) AS n
    FROM '{PARQUET}'
    WHERE {DEFAULT_FILTER}
    GROUP BY ALL ORDER BY bucket
    """
    df = con.execute(q).fetchdf()
    save_table(df, "aggregator_share_by_bucket.csv")

    # seniority_final_source breakdown (anomaly: are bucket differences
    # driven by rule vs LLM source of label)?
    q = f"""
    SELECT {period_case()} AS bucket, seniority_final_source, seniority_final,
           COUNT(*) AS n
    FROM '{PARQUET}'
    WHERE {DEFAULT_FILTER}
    GROUP BY ALL ORDER BY bucket, seniority_final_source, seniority_final
    """
    df = con.execute(q).fetchdf()
    save_table(df, "seniority_source_breakdown.csv")


# ---------------------------------------------------------------------------
# Step 2: Side-by-side histograms
# ---------------------------------------------------------------------------


def plot_distributions(con: duckdb.DuckDBPyConnection) -> None:
    print("[STEP 2] Producing distribution plots")

    # Figure 1: description_length histogram (log) by bucket and seniority (J2 vs S1)
    q = f"""
    SELECT {period_case()} AS bucket,
           CASE WHEN seniority_final IN ('entry','associate') THEN 'J2 junior'
                WHEN seniority_final IN ('mid-senior','director') THEN 'S1 senior'
                ELSE 'other/unknown' END AS slice,
           description_length
    FROM '{PARQUET}'
    WHERE {DEFAULT_FILTER} AND description_length IS NOT NULL
      AND description_length BETWEEN 1 AND 30000
    """
    df = con.execute(q).fetchdf()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    buckets = ["arshkon", "asaniczka", "scraped_2026"]
    for ax, b in zip(axes, buckets):
        sub = df[df["bucket"] == b]
        for slice_ in ["J2 junior", "S1 senior"]:
            vals = sub.loc[sub["slice"] == slice_, "description_length"].values
            if len(vals) == 0:
                continue
            ax.hist(np.log10(vals + 1), bins=40, alpha=0.5, label=f"{slice_} (n={len(vals)})")
        ax.set_title(b)
        ax.set_xlabel("log10(description_length + 1)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("count")
    fig.suptitle("description_length by bucket × seniority slice (J2/S1)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_desc_length_by_bucket_slice.png", dpi=150)
    plt.close(fig)

    # Figure 2: YOE extracted distribution by bucket
    q = f"""
    SELECT {period_case()} AS bucket, yoe_extracted
    FROM '{PARQUET}'
    WHERE {DEFAULT_FILTER} AND yoe_extracted IS NOT NULL
      AND yoe_extracted BETWEEN 0 AND 20
    """
    df = con.execute(q).fetchdf()
    fig, ax = plt.subplots(figsize=(7, 4))
    for b in buckets:
        vals = df.loc[df["bucket"] == b, "yoe_extracted"].values
        if len(vals) == 0:
            continue
        bins = np.arange(0, 21, 1)
        hist, _ = np.histogram(vals, bins=bins)
        centers = (bins[:-1] + bins[1:]) / 2
        ax.plot(centers, hist / hist.sum(), marker="o", label=f"{b} (n={len(vals):,})")
    ax.set_xlabel("yoe_extracted (years)")
    ax.set_ylabel("share of YOE-known rows")
    ax.set_title("YOE distribution by bucket (SWE LinkedIn, YOE-known)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_yoe_distribution.png", dpi=150)
    plt.close(fig)

    # Figure 3: seniority_final composition bar chart (stacked)
    q = f"""
    SELECT {period_case()} AS bucket, seniority_final,
           COUNT(*) AS n
    FROM '{PARQUET}'
    WHERE {DEFAULT_FILTER}
    GROUP BY ALL
    """
    df = con.execute(q).fetchdf()
    piv = df.pivot(index="bucket", columns="seniority_final", values="n").fillna(0)
    order = ["entry", "associate", "mid-senior", "director", "unknown"]
    piv = piv[[c for c in order if c in piv.columns]]
    piv_pct = piv.div(piv.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    piv_pct.loc[buckets].plot(kind="barh", stacked=True, ax=ax,
                              colormap="viridis")
    ax.set_xlabel("share of SWE rows")
    ax.set_ylabel("")
    ax.set_title("seniority_final composition by bucket")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_seniority_composition.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 3: Arshkon native-entry YOE diagnostic
# ---------------------------------------------------------------------------


def arshkon_native_yoe_diagnostic(con: duckdb.DuckDBPyConnection) -> dict:
    print("[STEP 3] Arshkon native-entry YOE diagnostic")
    q = f"""
    SELECT seniority_native, COUNT(*) AS n_all,
           SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS n_known,
           AVG(yoe_extracted) AS yoe_mean,
           APPROX_QUANTILE(yoe_extracted, 0.5) AS yoe_median,
           AVG(CASE WHEN yoe_extracted>=5 THEN 1.0 ELSE 0.0 END) AS share_ge5,
           AVG(CASE WHEN yoe_extracted<=2 THEN 1.0 ELSE 0.0 END) AS share_le2
    FROM '{PARQUET}'
    WHERE {DEFAULT_FILTER} AND source='kaggle_arshkon'
      AND seniority_native IS NOT NULL
    GROUP BY seniority_native ORDER BY seniority_native
    """
    df = con.execute(q).fetchdf()
    save_table(df, "arshkon_native_yoe_diagnostic.csv")

    entry_row = df[df["seniority_native"] == "entry"].iloc[0]
    return {
        "n_entry_native": int(entry_row["n_all"]),
        "yoe_mean": float(entry_row["yoe_mean"]),
        "yoe_median": float(entry_row["yoe_median"]) if pd.notna(entry_row["yoe_median"]) else None,
        "share_ge5": float(entry_row["share_ge5"]),
        "share_le2": float(entry_row["share_le2"]),
    }


# ---------------------------------------------------------------------------
# Step 4: Within-2024 calibration table extension
# ---------------------------------------------------------------------------


def extended_calibration(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Extends the shared calibration_table.csv with a few additional metrics
    stratified by S1 (mid-senior) for the calibration claim.

    Computes on S1 slice (mid-senior + director) LINKEDIN SWE only:
      - mean description_length
      - share with `ai` token via broad regex on raw description (case-insensitive)
      - share with org_scope keywords (broad match as in calibration table)
      - yoe_extracted mean
    """
    print("[STEP 4] Extended within-2024 calibration (S1 slice)")
    q = f"""
    WITH base AS (
      SELECT {period_case()} AS bucket,
             description,
             description_length,
             yoe_extracted,
             (CASE WHEN seniority_final IN ('mid-senior','director') THEN 1 ELSE 0 END) AS is_s1,
             regexp_matches(lower(description),
               '\\b(ai|artificial intelligence|machine learning|ml|llm|gpt|chatgpt|copilot)\\b') AS has_ai_broad,
             regexp_matches(lower(description),
               '\\b(cross[- ]functional|stakeholder|roadmap|strategy|platform team|org[- ]wide|company[- ]wide)\\b') AS has_org_scope
      FROM '{PARQUET}'
      WHERE {DEFAULT_FILTER}
    )
    SELECT bucket,
           COUNT(*) AS n_s1,
           AVG(description_length) AS desc_len_mean,
           AVG(CASE WHEN has_ai_broad THEN 1.0 ELSE 0.0 END) AS ai_broad_share,
           AVG(CASE WHEN has_org_scope THEN 1.0 ELSE 0.0 END) AS org_scope_share,
           AVG(yoe_extracted) AS yoe_mean
    FROM base WHERE is_s1=1
    GROUP BY bucket ORDER BY bucket
    """
    df = con.execute(q).fetchdf()

    rows = []
    # Compute within-2024 and cross-period for each metric
    ark = df[df["bucket"] == "arshkon"].iloc[0]
    asa = df[df["bucket"] == "asaniczka"].iloc[0]
    scr = df[df["bucket"] == "scraped_2026"].iloc[0]
    pool = {
        "desc_len_mean": (ark["n_s1"] * ark["desc_len_mean"] + asa["n_s1"] * asa["desc_len_mean"])
                         / (ark["n_s1"] + asa["n_s1"]),
        "ai_broad_share": (ark["n_s1"] * ark["ai_broad_share"] + asa["n_s1"] * asa["ai_broad_share"])
                          / (ark["n_s1"] + asa["n_s1"]),
        "org_scope_share": (ark["n_s1"] * ark["org_scope_share"] + asa["n_s1"] * asa["org_scope_share"])
                           / (ark["n_s1"] + asa["n_s1"]),
        "yoe_mean": (ark["n_s1"] * ark["yoe_mean"] + asa["n_s1"] * asa["yoe_mean"])
                    / (ark["n_s1"] + asa["n_s1"]),
    }
    for metric in ["desc_len_mean", "ai_broad_share", "org_scope_share", "yoe_mean"]:
        within = ark[metric] - asa[metric]
        cross_ark = scr[metric] - ark[metric]
        cross_pool = scr[metric] - pool[metric]
        snr_ark = abs(cross_ark) / abs(within) if abs(within) > 1e-9 else float("inf")
        snr_pool = abs(cross_pool) / abs(within) if abs(within) > 1e-9 else float("inf")
        rows.append({
            "metric": f"s1_{metric}",
            "arshkon": ark[metric],
            "asaniczka": asa[metric],
            "scraped_2026": scr[metric],
            "pooled_2024": pool[metric],
            "within_2024_effect": within,
            "cross_period_arshkon": cross_ark,
            "cross_period_pooled": cross_pool,
            "snr_arshkon": snr_ark,
            "snr_pooled": snr_pool,
        })
    out = pd.DataFrame(rows)
    save_table(out, "s1_calibration_extension.csv")
    return out


# ---------------------------------------------------------------------------
# Step 5: Junior share under T30 panel (J1..J4) by baseline
# ---------------------------------------------------------------------------


def junior_share_panel(con: duckdb.DuckDBPyConnection) -> None:
    """Uses the T30 panel definitions J1–J4:
      J1: seniority_final = 'entry' (strict)
      J2: seniority_final IN ('entry','associate')  [PRIMARY]
      J3: yoe_extracted <= 2 (of YOE-known)  [PRIMARY ROBUSTNESS]
      J4: yoe_extracted <= 4 (of YOE-known)
    Senior:
      S1: seniority_final IN ('mid-senior','director')  [PRIMARY]
      S2: seniority_final = 'director'
      S3: all rows (share = mid-senior+director / all)
      S4: yoe_extracted >= 5 (of YOE-known)  [PRIMARY ROBUSTNESS]
    """
    print("[STEP 5] Junior/senior share by T30 panel × baseline")
    q = f"""
    SELECT {period_case()} AS bucket,
           COUNT(*) AS n_all,
           SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS n_yoe,
           AVG(CASE WHEN seniority_final='entry' THEN 1.0 ELSE 0.0 END) AS j1_share_all,
           AVG(CASE WHEN seniority_final IN ('entry','associate') THEN 1.0 ELSE 0.0 END) AS j2_share_all,
           AVG(CASE WHEN seniority_final IN ('mid-senior','director') THEN 1.0 ELSE 0.0 END) AS s1_share_all,
           AVG(CASE WHEN seniority_final = 'director' THEN 1.0 ELSE 0.0 END) AS s2_share_all,
           AVG(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted<=2 THEN 1.0 ELSE 0.0 END)
             / NULLIF(AVG(CASE WHEN yoe_extracted IS NOT NULL THEN 1.0 ELSE 0.0 END),0) AS j3_share_known,
           AVG(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted<=4 THEN 1.0 ELSE 0.0 END)
             / NULLIF(AVG(CASE WHEN yoe_extracted IS NOT NULL THEN 1.0 ELSE 0.0 END),0) AS j4_share_known,
           AVG(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted>=5 THEN 1.0 ELSE 0.0 END)
             / NULLIF(AVG(CASE WHEN yoe_extracted IS NOT NULL THEN 1.0 ELSE 0.0 END),0) AS s4_share_known
    FROM '{PARQUET}'
    WHERE {DEFAULT_FILTER}
    GROUP BY bucket ORDER BY bucket
    """
    df = con.execute(q).fetchdf()

    # Produce the 2x4 table: (arshkon-only, pooled-2024) x (J1,J2,J3,J4)
    # And 2x4 for senior.
    ark = df[df["bucket"] == "arshkon"].iloc[0]
    asa = df[df["bucket"] == "asaniczka"].iloc[0]
    scr = df[df["bucket"] == "scraped_2026"].iloc[0]
    # pooled counts
    pool_n = ark["n_all"] + asa["n_all"]
    pool_n_yoe = ark["n_yoe"] + asa["n_yoe"]
    pool_j1 = (ark["n_all"] * ark["j1_share_all"] + asa["n_all"] * asa["j1_share_all"]) / pool_n
    pool_j2 = (ark["n_all"] * ark["j2_share_all"] + asa["n_all"] * asa["j2_share_all"]) / pool_n
    pool_s1 = (ark["n_all"] * ark["s1_share_all"] + asa["n_all"] * asa["s1_share_all"]) / pool_n
    pool_s2 = (ark["n_all"] * ark["s2_share_all"] + asa["n_all"] * asa["s2_share_all"]) / pool_n
    pool_j3 = (ark["n_yoe"] * ark["j3_share_known"] + asa["n_yoe"] * asa["j3_share_known"]) / pool_n_yoe
    pool_j4 = (ark["n_yoe"] * ark["j4_share_known"] + asa["n_yoe"] * asa["j4_share_known"]) / pool_n_yoe
    pool_s4 = (ark["n_yoe"] * ark["s4_share_known"] + asa["n_yoe"] * asa["s4_share_known"]) / pool_n_yoe

    rows = []
    for defn, ark_v, pool_v, scr_v in [
        ("J1", ark["j1_share_all"], pool_j1, scr["j1_share_all"]),
        ("J2", ark["j2_share_all"], pool_j2, scr["j2_share_all"]),
        ("J3", ark["j3_share_known"], pool_j3, scr["j3_share_known"]),
        ("J4", ark["j4_share_known"], pool_j4, scr["j4_share_known"]),
        ("S1", ark["s1_share_all"], pool_s1, scr["s1_share_all"]),
        ("S2", ark["s2_share_all"], pool_s2, scr["s2_share_all"]),
        ("S4", ark["s4_share_known"], pool_s4, scr["s4_share_known"]),
    ]:
        rows.append({
            "definition": defn,
            "baseline_arshkon": ark_v,
            "baseline_pooled_2024": pool_v,
            "scraped_2026": scr_v,
            "effect_arshkon": scr_v - ark_v,
            "effect_pooled": scr_v - pool_v,
            "direction_arshkon": "up" if scr_v > ark_v else "down",
            "direction_pooled": "up" if scr_v > pool_v else "down",
            "agree": (scr_v > ark_v) == (scr_v > pool_v),
        })
    out = pd.DataFrame(rows)
    save_table(out, "junior_senior_panel_by_baseline.csv")

    # Plot Fig 4: 2x4 junior/senior panel
    junior_defs = ["J1", "J2", "J3", "J4"]
    senior_defs = ["S1", "S2", "S4"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, defs, title in [
        (axes[0], junior_defs, "Junior share: scraped vs 2 baselines"),
        (axes[1], senior_defs, "Senior share: scraped vs 2 baselines"),
    ]:
        sub = out[out["definition"].isin(defs)].set_index("definition").loc[defs]
        x = np.arange(len(sub))
        w = 0.25
        ax.bar(x - w, sub["baseline_arshkon"], width=w, label="baseline arshkon")
        ax.bar(x, sub["baseline_pooled_2024"], width=w, label="baseline pooled 2024")
        ax.bar(x + w, sub["scraped_2026"], width=w, label="scraped 2026")
        ax.set_xticks(x)
        ax.set_xticklabels(defs)
        ax.set_ylabel("share")
        ax.set_title(title)
        ax.legend(fontsize=8)
        # annotate disagreements
        for xi, d in enumerate(defs):
            row = out[out["definition"] == d].iloc[0]
            if not row["agree"]:
                ax.text(xi, max(row["baseline_arshkon"], row["baseline_pooled_2024"],
                                row["scraped_2026"]) * 1.02,
                        "direction split", ha="center", fontsize=7, color="darkred")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_junior_senior_by_baseline.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 6: Ranked-change table
# ---------------------------------------------------------------------------


def ranked_changes(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Rank metrics by |cross-period effect| * SNR (where SNR computable).

    Pulls from shared calibration_table.csv (21 metrics), the junior/senior
    panel, and a small number of additional binary indicators computed
    inline: is_aggregator, is_remote_inferred, ghost_job_risk!=low,
    yoe_known_share (already in calibration), aggregator_share (already in),
    `c_+` AI-tool presence (from shared tech matrix).
    """
    print("[STEP 6] Building ranked-change table")
    cal = load_calibration()

    # Additional inline binary metrics at the full SWE level
    q = f"""
    SELECT {period_case()} AS bucket,
           COUNT(*) AS n,
           AVG(CASE WHEN is_remote_inferred THEN 1.0 ELSE 0.0 END) AS remote_share,
           AVG(CASE WHEN ghost_job_risk <> 'low' THEN 1.0 ELSE 0.0 END) AS ghost_nonlow_share,
           AVG(CASE WHEN is_multi_location THEN 1.0 ELSE 0.0 END) AS multi_loc_share,
           AVG(CASE WHEN metro_area IS NOT NULL THEN 1.0 ELSE 0.0 END) AS metro_assigned_share,
           AVG(CASE WHEN seniority_final='unknown' THEN 1.0 ELSE 0.0 END) AS unknown_senior_share,
           AVG(CASE WHEN seniority_3level='senior' THEN 1.0 ELSE 0.0 END) AS senior_3level_share,
           AVG(description_length) AS desc_len_mean,
           AVG(description_length*description_length) - AVG(description_length)*AVG(description_length) AS desc_len_var
    FROM '{PARQUET}'
    WHERE {DEFAULT_FILTER}
    GROUP BY bucket ORDER BY bucket
    """
    df = con.execute(q).fetchdf()
    ark = df[df["bucket"] == "arshkon"].iloc[0]
    asa = df[df["bucket"] == "asaniczka"].iloc[0]
    scr = df[df["bucket"] == "scraped_2026"].iloc[0]

    rows = []
    for m in ["remote_share", "ghost_nonlow_share", "multi_loc_share",
              "metro_assigned_share", "unknown_senior_share",
              "senior_3level_share"]:
        within = ark[m] - asa[m]
        cross_ark = scr[m] - ark[m]
        pool = (ark["n"] * ark[m] + asa["n"] * asa[m]) / (ark["n"] + asa["n"])
        cross_pool = scr[m] - pool
        snr_ark = abs(cross_ark) / abs(within) if abs(within) > 1e-9 else float("inf")
        snr_pool = abs(cross_pool) / abs(within) if abs(within) > 1e-9 else float("inf")
        rows.append({
            "metric": m,
            "arshkon": ark[m],
            "asaniczka": asa[m],
            "scraped": scr[m],
            "pooled_2024": pool,
            "within_2024_effect": within,
            "cross_period_effect_arshkon": cross_ark,
            "cross_period_effect_pooled": cross_pool,
            "snr_arshkon": snr_ark,
            "snr_pooled": snr_pool,
            "kind": "binary_share",
        })

    # Cohen's d on description_length
    pooled_var = ((ark["n"] - 1) * ark["desc_len_var"] + (asa["n"] - 1) * asa["desc_len_var"]) \
                 / (ark["n"] + asa["n"] - 2)
    pool_mean = (ark["n"] * ark["desc_len_mean"] + asa["n"] * asa["desc_len_mean"]) / (ark["n"] + asa["n"])
    pool_n = ark["n"] + asa["n"]
    d_cross_ark = (scr["desc_len_mean"] - ark["desc_len_mean"]) / np.sqrt(
        ((ark["n"] - 1) * ark["desc_len_var"] + (scr["n"] - 1) * scr["desc_len_var"])
        / (ark["n"] + scr["n"] - 2))
    d_cross_pool = (scr["desc_len_mean"] - pool_mean) / np.sqrt(
        ((pool_n - 1) * pooled_var + (scr["n"] - 1) * scr["desc_len_var"])
        / (pool_n + scr["n"] - 2))
    d_within = (ark["desc_len_mean"] - asa["desc_len_mean"]) / np.sqrt(pooled_var)
    rows.append({
        "metric": "desc_length_cohen_d",
        "arshkon": ark["desc_len_mean"],
        "asaniczka": asa["desc_len_mean"],
        "scraped": scr["desc_len_mean"],
        "pooled_2024": pool_mean,
        "within_2024_effect": d_within,
        "cross_period_effect_arshkon": d_cross_ark,
        "cross_period_effect_pooled": d_cross_pool,
        "snr_arshkon": abs(d_cross_ark) / abs(d_within) if abs(d_within) > 1e-9 else float("inf"),
        "snr_pooled": abs(d_cross_pool) / abs(d_within) if abs(d_within) > 1e-9 else float("inf"),
        "kind": "cohens_d",
    })

    # Add calibration_table rows in the same schema
    for _, r in cal.iterrows():
        rows.append({
            "metric": r["metric"],
            "arshkon": r["arshkon_value"],
            "asaniczka": r["asaniczka_value"],
            "scraped": r["scraped_value"],
            "pooled_2024": r["pooled_2024_value"],
            "within_2024_effect": r["within_2024_effect"],
            "cross_period_effect_arshkon": r["cross_period_effect_arshkon"],
            "cross_period_effect_pooled": r["cross_period_effect_pooled"],
            "snr_arshkon": r["snr_arshkon"],
            "snr_pooled": r["snr_pooled"],
            "kind": "binary_share" if "share" in str(r["metric"]) or "binary" in str(r["metric"])
                    else ("density" if "density" in str(r["metric"])
                          else ("count" if "count" in str(r["metric"])
                                else "length_mean" if "length" in str(r["metric"])
                                else "misc")),
        })
    out = pd.DataFrame(rows)

    # Rank by |pooled cross-period effect| * SNR(pooled), desc
    # For metrics whose "cross_period_effect" is on a share scale,
    # the |effect| is in pp; for cohen_d it's already effect-size.
    out["abs_cross_pooled"] = out["cross_period_effect_pooled"].abs()
    out["abs_cross_arshkon"] = out["cross_period_effect_arshkon"].abs()
    out["rank_score_pooled"] = out["abs_cross_pooled"] * out["snr_pooled"].clip(upper=200)
    out["rank_score_arshkon"] = out["abs_cross_arshkon"] * out["snr_arshkon"].clip(upper=200)

    out = out.sort_values("rank_score_pooled", ascending=False).reset_index(drop=True)
    save_table(out, "ranked_change_table.csv")

    # Figure 5: top 10 by rank_score_pooled
    top = out.head(12).copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    top_plot = top.iloc[::-1]
    ypos = np.arange(len(top_plot))
    ax.barh(ypos, top_plot["snr_pooled"].clip(upper=50), color="tab:blue", alpha=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{m}\n(cross=Δ {v:+.3g})"
                        for m, v in zip(top_plot["metric"], top_plot["cross_period_effect_pooled"])],
                       fontsize=8)
    ax.set_xlabel("SNR (pooled baseline, clipped to 50)")
    ax.set_title("Top-12 metrics by |cross-period effect| × SNR (pooled baseline)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_top_metrics_by_snr.png", dpi=150)
    plt.close(fig)

    return out


# ---------------------------------------------------------------------------
# Step 7: Domain × seniority decomposition (J2 share × domain-proxy)
# ---------------------------------------------------------------------------


DOMAIN_BUCKETS = {
    "ml_ai": r"\b(machine learning|ml engineer|mlops|ai engineer|data scientist|applied scientist|research scientist|deep learning|nlp|cv|computer vision|llm)\b",
    "data": r"\b(data engineer|analytics engineer|data analyst|data platform|data pipeline|etl|bigquery|warehouse|analyst)\b",
    "backend": r"\b(backend|back[- ]end|api|distributed systems|platform engineer|infrastructure)\b",
    "frontend": r"\b(frontend|front[- ]end|ui engineer|javascript|react|web engineer)\b",
    "fullstack": r"\b(full[- ]?stack)\b",
    "mobile": r"\b(ios|android|mobile)\b",
    "devops": r"\b(devops|sre|site reliability|platform reliability|cloud engineer)\b",
    "security": r"\b(security engineer|appsec|application security|cybersecurity|infosec)\b",
    "qa": r"\b(qa|quality assurance|sdet|test engineer|automation tester)\b",
}


def domain_decomposition(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    print("[STEP 7] Domain × seniority decomposition")

    case_parts = []
    for name, pattern in DOMAIN_BUCKETS.items():
        case_parts.append(
            f"WHEN regexp_matches(lower(title_normalized), '{pattern}') THEN '{name}'"
        )
    domain_case = "CASE " + " ".join(case_parts) + " ELSE 'other' END"

    q = f"""
    WITH base AS (
      SELECT {period_case()} AS bucket,
             {domain_case} AS domain,
             (CASE WHEN seniority_final IN ('entry','associate') THEN 1 ELSE 0 END) AS is_j2,
             (CASE WHEN seniority_final IN ('mid-senior','director') THEN 1 ELSE 0 END) AS is_s1
      FROM '{PARQUET}'
      WHERE {DEFAULT_FILTER}
    )
    SELECT bucket, domain, COUNT(*) AS n,
           AVG(is_j2) AS j2_share, AVG(is_s1) AS s1_share
    FROM base GROUP BY ALL ORDER BY bucket, domain
    """
    df = con.execute(q).fetchdf()
    save_table(df, "domain_seniority_decomposition.csv")

    # Between-domain vs within-domain decomposition of J2 share change
    # (arshkon → scraped). Use decomposition: total change = within-effect
    # (weighted domain share change at fixed weight) + between-effect (share-weight change).
    ark_w = df[df["bucket"] == "arshkon"].set_index("domain")
    scr_w = df[df["bucket"] == "scraped_2026"].set_index("domain")
    all_domains = sorted(set(ark_w.index) | set(scr_w.index))

    ark_n = ark_w["n"].sum()
    scr_n = scr_w["n"].sum()

    decomp_rows = []
    within_tot = 0.0
    between_tot = 0.0
    for d in all_domains:
        w_ark = (ark_w.loc[d, "n"] / ark_n) if d in ark_w.index else 0.0
        w_scr = (scr_w.loc[d, "n"] / scr_n) if d in scr_w.index else 0.0
        j_ark = ark_w.loc[d, "j2_share"] if d in ark_w.index else 0.0
        j_scr = scr_w.loc[d, "j2_share"] if d in scr_w.index else 0.0
        # Fix weights at (w_ark + w_scr) / 2 for symmetric decomposition
        avg_w = (w_ark + w_scr) / 2
        avg_j = (j_ark + j_scr) / 2
        within = avg_w * (j_scr - j_ark)
        between = (w_scr - w_ark) * avg_j
        within_tot += within
        between_tot += between
        decomp_rows.append({
            "domain": d,
            "w_arshkon": w_ark, "w_scraped": w_scr,
            "j2_arshkon": j_ark, "j2_scraped": j_scr,
            "within_contrib": within,
            "between_contrib": between,
        })
    decomp = pd.DataFrame(decomp_rows)
    total_change = (scr_w["j2_share"] * scr_w["n"] / scr_n).sum() - (
        ark_w["j2_share"] * ark_w["n"] / ark_n).sum()
    decomp = pd.concat([decomp, pd.DataFrame([{
        "domain": "TOTAL",
        "w_arshkon": 1.0, "w_scraped": 1.0,
        "j2_arshkon": (ark_w["j2_share"] * ark_w["n"] / ark_n).sum(),
        "j2_scraped": (scr_w["j2_share"] * scr_w["n"] / scr_n).sum(),
        "within_contrib": within_tot,
        "between_contrib": between_tot,
    }])], ignore_index=True)
    save_table(decomp, "j2_domain_decomposition_arshkon_to_scraped.csv")
    print(f"    J2 arshkon→scraped: total Δ={total_change:+.4f}, "
          f"within={within_tot:+.4f}, between={between_tot:+.4f}, "
          f"between-share of total = "
          f"{between_tot/total_change*100 if abs(total_change)>1e-9 else float('nan'):.1f}%")
    return decomp


# ---------------------------------------------------------------------------
# Step 8: Company-size stratification (arshkon-only)
# ---------------------------------------------------------------------------


def company_size_stratification(con: duckdb.DuckDBPyConnection) -> None:
    print("[STEP 8] Company-size stratification (arshkon-only)")
    # Load tech matrix for AI prevalence + tech count
    # Join on uid to classify by company size quartile
    tech_path = SHARED / "swe_tech_matrix.parquet"
    text_path = SHARED / "swe_cleaned_text.parquet"

    q = f"""
    WITH arsh AS (
      SELECT uid, company_size,
             CASE WHEN seniority_final IN ('entry','associate') THEN 1 ELSE 0 END AS is_j2
      FROM '{PARQUET}'
      WHERE {DEFAULT_FILTER} AND source='kaggle_arshkon'
        AND company_size IS NOT NULL
    ), quartiles AS (
      SELECT uid, is_j2, company_size,
             NTILE(4) OVER (ORDER BY company_size) AS size_q
      FROM arsh
    ), tech AS (
      SELECT * FROM '{tech_path}'
    ), text AS (
      SELECT uid, description_cleaned, text_source FROM '{text_path}'
    )
    SELECT q.size_q,
           COUNT(*) AS n,
           MIN(q.company_size) AS size_min,
           MAX(q.company_size) AS size_max,
           AVG(q.is_j2) AS j2_share,
           AVG(CASE WHEN text.description_cleaned IS NOT NULL AND
                         (regexp_matches(text.description_cleaned, '\\bai\\b') OR
                          regexp_matches(text.description_cleaned, '\\bartificial intelligence\\b') OR
                          regexp_matches(text.description_cleaned, '\\bllm\\b'))
                    THEN 1.0 ELSE 0.0 END) AS ai_share,
           AVG((CAST(tech.copilot AS INTEGER) + CAST(tech.cursor AS INTEGER)
               + CAST(tech.claude AS INTEGER) + CAST(tech.openai AS INTEGER)
               + CAST(tech.anthropic AS INTEGER) + CAST(tech.chatgpt AS INTEGER))) AS ai_tool_count,
           AVG((CAST(tech.python AS INTEGER) + CAST(tech.java AS INTEGER)
               + CAST(tech.javascript AS INTEGER) + CAST(tech.typescript AS INTEGER)
               + CAST(tech.go AS INTEGER) + CAST(tech.rust AS INTEGER)
               + CAST(tech.cpp AS INTEGER) + CAST(tech.csharp AS INTEGER)
               + CAST(tech.ruby AS INTEGER) + CAST(tech.sql AS INTEGER)
               + CAST(tech.aws AS INTEGER) + CAST(tech.azure AS INTEGER)
               + CAST(tech.gcp AS INTEGER) + CAST(tech.kubernetes AS INTEGER)
               + CAST(tech.docker AS INTEGER) + CAST(tech.react AS INTEGER))) AS core_tech_count_16
    FROM quartiles q
    LEFT JOIN tech ON q.uid = tech.uid
    LEFT JOIN text ON q.uid = text.uid
    GROUP BY q.size_q ORDER BY q.size_q
    """
    df = con.execute(q).fetchdf()
    save_table(df, "arshkon_company_size_quartiles.csv")


# ---------------------------------------------------------------------------
# Step 9: Sensitivity analyses — specialist exclusion + aggregator exclusion
# ---------------------------------------------------------------------------


def sensitivity_specialist_and_aggregator(con: duckdb.DuckDBPyConnection) -> None:
    print("[STEP 9] Sensitivity — specialist exclusion × aggregator exclusion")
    specialists = list(load_specialists())
    specialists_sql = "(" + ",".join(f"'{s.replace(chr(39), chr(39)*2)}'" for s in specialists) + ")"

    # Run J2, S1, ai_mention, org_scope, ghost, desc_length under 4 sens configs
    q = f"""
    WITH base AS (
      SELECT {period_case()} AS bucket,
             company_name_canonical IN {specialists_sql} AS is_specialist,
             is_aggregator,
             CASE WHEN seniority_final IN ('entry','associate') THEN 1 ELSE 0 END AS is_j2,
             CASE WHEN seniority_final IN ('mid-senior','director') THEN 1 ELSE 0 END AS is_s1,
             description_length,
             regexp_matches(lower(description),
               '\\b(ai|artificial intelligence|machine learning|llm|gpt|copilot)\\b') AS has_ai
      FROM '{PARQUET}'
      WHERE {DEFAULT_FILTER}
    )
    SELECT bucket,
           CASE WHEN is_specialist THEN 'spec' ELSE 'nonspec' END AS sp,
           CASE WHEN is_aggregator THEN 'agg' ELSE 'nonagg' END AS ag,
           COUNT(*) AS n,
           AVG(is_j2) AS j2, AVG(is_s1) AS s1,
           AVG(description_length) AS len_mean,
           AVG(CASE WHEN has_ai THEN 1.0 ELSE 0.0 END) AS ai_share
    FROM base GROUP BY ALL ORDER BY bucket, sp, ag
    """
    df = con.execute(q).fetchdf()
    save_table(df, "sensitivity_spec_x_agg.csv")

    # Pivot to show: base, exclude-spec, exclude-agg, exclude-both
    rows = []
    for bucket in ["arshkon", "asaniczka", "scraped_2026"]:
        sub = df[df["bucket"] == bucket]
        for cfg, mask_rows in [
            ("all_rows", sub),
            ("excl_spec", sub[sub["sp"] == "nonspec"]),
            ("excl_agg", sub[sub["ag"] == "nonagg"]),
            ("excl_both", sub[(sub["sp"] == "nonspec") & (sub["ag"] == "nonagg")]),
        ]:
            n = mask_rows["n"].sum()
            if n == 0:
                continue
            rows.append({
                "bucket": bucket,
                "config": cfg,
                "n": int(n),
                "j2": (mask_rows["j2"] * mask_rows["n"]).sum() / n,
                "s1": (mask_rows["s1"] * mask_rows["n"]).sum() / n,
                "len_mean": (mask_rows["len_mean"] * mask_rows["n"]).sum() / n,
                "ai_share": (mask_rows["ai_share"] * mask_rows["n"]).sum() / n,
            })
    pivot = pd.DataFrame(rows)
    save_table(pivot, "sensitivity_spec_x_agg_pivot.csv")

    # Check scraped cross-period effects under each config
    print("\n    AI-share cross-period (arshkon→scraped) sensitivity:")
    for cfg in ["all_rows", "excl_spec", "excl_agg", "excl_both"]:
        ark = pivot[(pivot["bucket"] == "arshkon") & (pivot["config"] == cfg)].iloc[0]
        scr = pivot[(pivot["bucket"] == "scraped_2026") & (pivot["config"] == cfg)].iloc[0]
        print(f"      {cfg}: arsh={ark['ai_share']:.4f} "
              f"→ scr={scr['ai_share']:.4f} (Δ={scr['ai_share']-ark['ai_share']:+.4f})")
    print("    J2 cross-period (arshkon→scraped) sensitivity:")
    for cfg in ["all_rows", "excl_spec", "excl_agg", "excl_both"]:
        ark = pivot[(pivot["bucket"] == "arshkon") & (pivot["config"] == cfg)].iloc[0]
        scr = pivot[(pivot["bucket"] == "scraped_2026") & (pivot["config"] == cfg)].iloc[0]
        print(f"      {cfg}: arsh={ark['j2']:.4f} "
              f"→ scr={scr['j2']:.4f} (Δ={scr['j2']-ark['j2']:+.4f})")


# ---------------------------------------------------------------------------
# Step 10: Anomaly scan
# ---------------------------------------------------------------------------


def anomaly_scan(con: duckdb.DuckDBPyConnection) -> None:
    print("[STEP 10] Anomaly scan")
    # Bimodality probe on description_length by bucket
    q = f"""
    SELECT {period_case()} AS bucket,
           FLOOR(description_length / 500) * 500 AS bin,
           COUNT(*) AS n
    FROM '{PARQUET}'
    WHERE {DEFAULT_FILTER} AND description_length BETWEEN 1 AND 30000
    GROUP BY ALL ORDER BY bucket, bin
    """
    df = con.execute(q).fetchdf()
    save_table(df, "anomaly_desc_length_bins.csv")

    # Yoe spike at 5 specifically (very common "5+ years" phrasing)
    q = f"""
    SELECT {period_case()} AS bucket, yoe_extracted, COUNT(*) AS n
    FROM '{PARQUET}'
    WHERE {DEFAULT_FILTER} AND yoe_extracted IS NOT NULL
    GROUP BY ALL ORDER BY bucket, yoe_extracted
    """
    df = con.execute(q).fetchdf()
    save_table(df, "anomaly_yoe_spikes.csv")

    # ghost_job_risk distribution (only entry rows can be elevated by design)
    q = f"""
    SELECT {period_case()} AS bucket, ghost_job_risk, seniority_final, COUNT(*) AS n
    FROM '{PARQUET}'
    WHERE {DEFAULT_FILTER}
    GROUP BY ALL ORDER BY bucket, ghost_job_risk, seniority_final
    """
    df = con.execute(q).fetchdf()
    save_table(df, "anomaly_ghost_by_bucket_senior.csv")

    # seniority_native anomaly on asaniczka: only mid-senior / associate
    q = f"""
    SELECT {period_case()} AS bucket, seniority_native, COUNT(*) AS n
    FROM '{PARQUET}'
    WHERE {DEFAULT_FILTER}
    GROUP BY ALL ORDER BY bucket, seniority_native
    """
    df = con.execute(q).fetchdf()
    save_table(df, "anomaly_seniority_native_by_bucket.csv")

    # LLM coverage divergence by bucket x seniority
    q = f"""
    SELECT {period_case()} AS bucket, seniority_final,
           llm_extraction_coverage, llm_classification_coverage,
           COUNT(*) AS n
    FROM '{PARQUET}'
    WHERE {DEFAULT_FILTER}
    GROUP BY ALL ORDER BY bucket, seniority_final, llm_extraction_coverage,
                          llm_classification_coverage
    """
    df = con.execute(q).fetchdf()
    save_table(df, "anomaly_llm_coverage_by_bucket_senior.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Inline asserts for domain regex
    assert re.search(DOMAIN_BUCKETS["ml_ai"], "senior ml engineer"), "ml_ai bucket match"
    assert re.search(DOMAIN_BUCKETS["backend"], "backend developer"), "backend bucket match"
    assert re.search(DOMAIN_BUCKETS["security"], "application security engineer"), "security bucket match"
    assert not re.search(DOMAIN_BUCKETS["security"], "software security review"), "security bucket no-false-positive"

    con = connect()
    univariate_profile(con)
    plot_distributions(con)
    native_yoe = arshkon_native_yoe_diagnostic(con)
    print(f"    arshkon native entry: n={native_yoe['n_entry_native']}, "
          f"YOE mean={native_yoe['yoe_mean']:.2f}, median={native_yoe['yoe_median']}, "
          f"share≥5={native_yoe['share_ge5']:.3f}, share≤2={native_yoe['share_le2']:.3f}")
    extended_calibration(con)
    junior_share_panel(con)
    ranked_changes(con)
    domain_decomposition(con)
    company_size_stratification(con)
    sensitivity_specialist_and_aggregator(con)
    anomaly_scan(con)

    # Also persist the native_yoe diagnostic as a small summary
    with open(TABLE_DIR / "arshkon_native_yoe_summary.json", "w") as f:
        json.dump(native_yoe, f, indent=2)


if __name__ == "__main__":
    main()
