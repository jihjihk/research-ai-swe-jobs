"""
S27 — Role landscape composite (Applied-AI, FDE, Emerging clusters, Legacy substitution).

Methodology exploration for composite article B. Outputs to eda/tables/S27_*.csv
and eda/figures/S27_*.png.

Run:
  ./.venv/bin/python eda/scripts/S27_role_landscape.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE = str(PROJECT_ROOT / "data" / "unified_core.parquet")
TABLES = PROJECT_ROOT / "eda" / "tables"
FIGURES = PROJECT_ROOT / "eda" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# Canonical AI-vocab regex from eda/scripts/scans.py
AI_VOCAB_PATTERN = (
    r"(?i)\b(llm|gpt|chatgpt|claude|copilot|openai|anthropic|gemini|bard|mistral|"
    r"llama|large\ language\ model|generative\ ai|genai|gen\ ai|foundation\ model|"
    r"transformer\ model|ai\ agent|agentic|ai\-powered|ai\ tooling|ai\-assisted|"
    r"rag|retrieval\ augmented|vector\ database|vector\ store|embedding\ model|"
    r"prompt\ engineering|prompt\ engineer|ml\ ops|mlops|llmops|cursor\ ide|"
    r"windsurf\ ide|github\ copilot)\b"
)

# Title-regex definitions of Applied-AI archetype
APPLIED_AI_TITLE_REGEX = (
    r"(?i)\b(applied\s+ai|applied\s+ml|ai\s+engineer|ml\s+engineer|llm\s+engineer|"
    r"machine\s+learning\s+engineer|mlops\s+engineer|genai\s+engineer|"
    r"generative\s+ai\s+engineer|foundation\s+model\s+engineer|"
    r"agent(?:ic)?\s+engineer|ai/ml\s+engineer)\b"
)

FDE_TITLE_REGEX = r"(?i)forward[\s\-]?deployed"
FDE_DESC_REGEX = (
    r"(?i)forward[\s\-]?deployed|customer[\s\-]?facing\s+engineer|"
    r"deployment\s+(?:strategist|engineer)\s+(?:on|with)?\s*(?:customer|client)"
)

CORE_FILTER = "is_english = true AND date_flag = 'ok'"


def period_bucket(p):
    return "2024" if p.startswith("2024") else "2026"


# ---------------------------------------------------------------------------
# THREAD 1 — Applied-AI archetype: T34-cluster vs title-regex
# ---------------------------------------------------------------------------

def thread1_applied_ai(con):
    print("\n=== THREAD 1: Applied-AI archetype ===")
    # Method (a) — v9 T34 cluster definition (project-rate from labels)
    # Read T34 cluster temporal counts directly
    t34 = pd.read_csv(PROJECT_ROOT / "exploration-archive/v9_final_opus_47/tables/T34/cluster_precondition_check.csv")
    t34_cluster0 = t34[t34.cluster_id == 0].iloc[0]
    t34_method = {
        "method": "v9_T34_cluster0",
        "n_2024": int(t34_cluster0["n_2024"]),
        "n_2026": int(t34_cluster0["n_2026"]),
        "growth_ratio": float(t34_cluster0["growth_ratio"]),
        "share_2026": float(t34_cluster0["share_2026"]),
        "median_yoe": 6.0,  # from v9 story
        "director_share_pct": 1.92,  # from v9 story
        "n_firms": 1163,
    }

    # Method (b) — title-regex on unified_core
    sql = f"""
      SELECT period_bucket, COUNT(*) AS n,
             SUM(CASE WHEN regexp_matches(LOWER(COALESCE(title, '')), '{APPLIED_AI_TITLE_REGEX}') THEN 1 ELSE 0 END) AS n_applied_ai,
             SUM(CASE WHEN seniority_final = 'director' THEN 1 ELSE 0 END) AS n_director_total
      FROM (
        SELECT *,
               CASE WHEN period LIKE '2024%' THEN '2024' ELSE '2026' END AS period_bucket
        FROM '{CORE}'
        WHERE {CORE_FILTER} AND is_swe
      )
      GROUP BY 1
      ORDER BY 1
    """
    base = con.execute(sql).df()

    sql_aa = f"""
      WITH aa AS (
        SELECT *,
               CASE WHEN period LIKE '2024%' THEN '2024' ELSE '2026' END AS period_bucket
        FROM '{CORE}'
        WHERE {CORE_FILTER} AND is_swe
          AND regexp_matches(LOWER(COALESCE(title, '')), '{APPLIED_AI_TITLE_REGEX}')
      )
      SELECT period_bucket,
             COUNT(*) AS n,
             SUM(CASE WHEN seniority_final = 'director' THEN 1 ELSE 0 END) AS n_director,
             AVG(yoe_min_years_llm) AS mean_yoe,
             MEDIAN(yoe_min_years_llm) AS median_yoe,
             COUNT(DISTINCT company_name_canonical) AS n_firms
      FROM aa
      GROUP BY 1
      ORDER BY 1
    """
    title_method_df = con.execute(sql_aa).df()
    print("Title-regex Applied-AI by period:")
    print(title_method_df)

    title_method_df.to_csv(TABLES / "S27_thread1_title_regex_applied_ai.csv", index=False)
    base.to_csv(TABLES / "S27_thread1_swe_base.csv", index=False)

    # Compute growth + director share for title method
    if len(title_method_df) == 2:
        n_2024 = int(title_method_df.loc[title_method_df.period_bucket == "2024", "n"].iloc[0])
        n_2026 = int(title_method_df.loc[title_method_df.period_bucket == "2026", "n"].iloc[0])
        med_yoe_2026 = float(title_method_df.loc[title_method_df.period_bucket == "2026", "median_yoe"].iloc[0])
        n_dir_2026 = int(title_method_df.loc[title_method_df.period_bucket == "2026", "n_director"].iloc[0])
        n_firms_2026 = int(title_method_df.loc[title_method_df.period_bucket == "2026", "n_firms"].iloc[0])
        title_method = {
            "method": "title_regex",
            "n_2024": n_2024,
            "n_2026": n_2026,
            "growth_ratio": (n_2026 / max(n_2024, 1)),
            "share_2026": n_2026 / int(base.loc[base.period_bucket == "2026", "n"].iloc[0]),
            "median_yoe": med_yoe_2026,
            "director_share_pct": 100 * n_dir_2026 / max(n_2026, 1),
            "n_firms": n_firms_2026,
        }
    else:
        title_method = {"method": "title_regex", "error": "missing period bucket"}

    # Compare benchmarks
    senior_mask = "seniority_3level = 'senior'"
    sql_benchmark = f"""
      WITH base AS (
        SELECT *,
               CASE WHEN period LIKE '2024%' THEN '2024' ELSE '2026' END AS period_bucket
        FROM '{CORE}'
        WHERE {CORE_FILTER} AND is_swe AND period LIKE '2026%' AND {senior_mask}
      )
      SELECT
        AVG(yoe_min_years_llm) AS mean_yoe_overall_senior,
        MEDIAN(yoe_min_years_llm) AS median_yoe_overall_senior,
        AVG(CASE WHEN seniority_final = 'director' THEN 1.0 ELSE 0.0 END) AS director_share_overall_senior
      FROM base
    """
    benchmark = con.execute(sql_benchmark).df().iloc[0].to_dict()
    print("Benchmark senior 2026:", benchmark)

    comparison = pd.DataFrame([t34_method, title_method])
    comparison["benchmark_median_yoe_senior_2026"] = benchmark["median_yoe_overall_senior"]
    comparison["benchmark_director_share_senior_2026_pct"] = 100 * benchmark["director_share_overall_senior"]
    comparison.to_csv(TABLES / "S27_thread1_method_comparison.csv", index=False)
    print("Wrote S27_thread1_method_comparison.csv")

    # Figure: side-by-side method comparison
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    methods = ["v9 T34 cluster0", "Title regex"]

    growth = [t34_method["growth_ratio"], title_method["growth_ratio"]]
    axes[0].bar(methods, growth, color=["#1f77b4", "#ff7f0e"])
    axes[0].set_ylabel("Growth ratio (2026 / 2024)")
    axes[0].set_title("Volume growth")
    for i, v in enumerate(growth):
        axes[0].text(i, v + 0.3, f"{v:.1f}×", ha="center", fontsize=11)

    yoe = [t34_method["median_yoe"], title_method["median_yoe"]]
    axes[1].bar(methods, yoe, color=["#1f77b4", "#ff7f0e"])
    axes[1].axhline(benchmark["median_yoe_overall_senior"], color="black", ls="--",
                    label=f"senior median = {benchmark['median_yoe_overall_senior']:.1f}")
    axes[1].set_ylabel("Median YOE")
    axes[1].set_title("Experience floor")
    axes[1].legend()

    dirshare = [t34_method["director_share_pct"], title_method["director_share_pct"]]
    axes[2].bar(methods, dirshare, color=["#1f77b4", "#ff7f0e"])
    axes[2].axhline(100 * benchmark["director_share_overall_senior"], color="black", ls="--",
                    label=f"senior dir share = {100 * benchmark['director_share_overall_senior']:.2f}%")
    axes[2].set_ylabel("Director share (%)")
    axes[2].set_title("Director density")
    axes[2].legend()

    fig.suptitle("S27 Thread 1 — Applied-AI: T34-cluster vs title-regex method comparison")
    fig.tight_layout()
    fig.savefig(FIGURES / "S27_thread1_applied_ai_methods.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    return comparison


# ---------------------------------------------------------------------------
# THREAD 2 — FDE: title-only vs title+description
# ---------------------------------------------------------------------------

def thread2_fde(con):
    print("\n=== THREAD 2: Forward-Deployed Engineer ===")
    # Method (a) — title only
    sql_title = f"""
      WITH base AS (
        SELECT *,
               CASE WHEN period LIKE '2024%' THEN '2024' ELSE '2026' END AS period_bucket
        FROM '{CORE}'
        WHERE {CORE_FILTER} AND is_swe
      )
      SELECT period_bucket,
             COUNT(*) AS n_swe,
             SUM(CASE WHEN regexp_matches(COALESCE(title, ''), '{FDE_TITLE_REGEX}') THEN 1 ELSE 0 END) AS n_fde_title
      FROM base
      GROUP BY 1
      ORDER BY 1
    """
    title_df = con.execute(sql_title).df()
    title_df["fde_share"] = title_df["n_fde_title"] / title_df["n_swe"]
    print("Title-only FDE:")
    print(title_df)

    # Method (b) — title or description
    sql_combined = f"""
      WITH base AS (
        SELECT *,
               CASE WHEN period LIKE '2024%' THEN '2024' ELSE '2026' END AS period_bucket
        FROM '{CORE}'
        WHERE {CORE_FILTER} AND is_swe
      )
      SELECT period_bucket,
             COUNT(*) AS n_swe,
             SUM(CASE WHEN regexp_matches(COALESCE(title, ''), '{FDE_TITLE_REGEX}') THEN 1 ELSE 0 END) AS n_fde_title,
             SUM(CASE WHEN regexp_matches(COALESCE(title, ''), '{FDE_TITLE_REGEX}')
                       OR regexp_matches(COALESCE(description, ''), '{FDE_DESC_REGEX}')
                  THEN 1 ELSE 0 END) AS n_fde_combined
      FROM base
      GROUP BY 1
      ORDER BY 1
    """
    combined_df = con.execute(sql_combined).df()
    combined_df["fde_share"] = combined_df["n_fde_title"] / combined_df["n_swe"]
    combined_df["fde_combined_share"] = combined_df["n_fde_combined"] / combined_df["n_swe"]
    print("Title+desc FDE:")
    print(combined_df)
    combined_df.to_csv(TABLES / "S27_thread2_fde_method_comparison.csv", index=False)

    # 2026 firm count + AI density (verify story claims)
    sql_firms = f"""
      WITH fde AS (
        SELECT *
        FROM '{CORE}'
        WHERE {CORE_FILTER} AND is_swe
          AND period LIKE '2026%'
          AND regexp_matches(COALESCE(title, ''), '{FDE_TITLE_REGEX}')
      )
      SELECT COUNT(*) AS n_postings,
             COUNT(DISTINCT company_name_canonical) AS n_firms,
             AVG(CASE WHEN regexp_matches(description, '{AI_VOCAB_PATTERN}') THEN 1.0 ELSE 0.0 END) AS ai_rate,
             AVG(yoe_min_years_llm) AS mean_yoe,
             MEDIAN(yoe_min_years_llm) AS median_yoe
      FROM fde
    """
    fde_2026 = con.execute(sql_firms).df().iloc[0].to_dict()
    print("FDE 2026 stats:", fde_2026)

    sql_general = f"""
      SELECT COUNT(*) AS n,
             AVG(CASE WHEN regexp_matches(description, '{AI_VOCAB_PATTERN}') THEN 1.0 ELSE 0.0 END) AS ai_rate,
             AVG(yoe_min_years_llm) AS mean_yoe,
             MEDIAN(yoe_min_years_llm) AS median_yoe
      FROM '{CORE}'
      WHERE {CORE_FILTER} AND is_swe AND period LIKE '2026%'
    """
    general_2026 = con.execute(sql_general).df().iloc[0].to_dict()
    print("General SWE 2026 stats:", general_2026)

    summary = pd.DataFrame([
        {"cohort": "FDE_2026", **fde_2026},
        {"cohort": "general_SWE_2026", **general_2026},
    ])
    summary["ai_density_ratio_vs_general"] = summary["ai_rate"] / general_2026["ai_rate"]
    summary.to_csv(TABLES / "S27_thread2_fde_density_check.csv", index=False)

    # Top firms among FDE
    sql_top_firms = f"""
      SELECT company_name_canonical, COUNT(*) AS n
      FROM '{CORE}'
      WHERE {CORE_FILTER} AND is_swe AND period LIKE '2026%'
        AND regexp_matches(COALESCE(title, ''), '{FDE_TITLE_REGEX}')
      GROUP BY 1 ORDER BY n DESC, company_name_canonical ASC
    """
    top_firms = con.execute(sql_top_firms).df()
    top_firms.to_csv(TABLES / "S27_thread2_fde_firms_2026.csv", index=False)
    print(f"Distinct FDE firms (2026): {len(top_firms)}")
    print("Top 10:", top_firms.head(10).to_string(index=False))

    # Figure: title vs title+desc comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    periods = combined_df["period_bucket"].tolist()
    x = np.arange(len(periods))
    width = 0.35
    axes[0].bar(x - width/2, combined_df["n_fde_title"], width, label="title only", color="#1f77b4")
    axes[0].bar(x + width/2, combined_df["n_fde_combined"], width, label="title or desc", color="#ff7f0e")
    axes[0].set_xticks(x); axes[0].set_xticklabels(periods)
    axes[0].set_ylabel("posting count")
    axes[0].set_title("FDE postings — count by method")
    for i, p in enumerate(periods):
        axes[0].text(i - width/2, combined_df["n_fde_title"].iloc[i] + 1, f"{int(combined_df['n_fde_title'].iloc[i])}",
                     ha="center", fontsize=9)
        axes[0].text(i + width/2, combined_df["n_fde_combined"].iloc[i] + 1, f"{int(combined_df['n_fde_combined'].iloc[i])}",
                     ha="center", fontsize=9)
    axes[0].legend()

    axes[1].bar(x - width/2, combined_df["fde_share"] * 1e4, width, label="title only", color="#1f77b4")
    axes[1].bar(x + width/2, combined_df["fde_combined_share"] * 1e4, width, label="title or desc", color="#ff7f0e")
    axes[1].set_xticks(x); axes[1].set_xticklabels(periods)
    axes[1].set_ylabel("share (per 10,000 postings)")
    axes[1].set_title("FDE share of SWE")
    axes[1].legend()

    fig.suptitle("S27 Thread 2 — Forward-Deployed Engineer: title vs title+desc")
    fig.tight_layout()
    fig.savefig(FIGURES / "S27_thread2_fde_methods.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    return combined_df, summary, top_firms


# ---------------------------------------------------------------------------
# THREAD 3 — Emerging clusters: ingest v9 T09 labels + TF-IDF emergent terms
# ---------------------------------------------------------------------------

def thread3_clusters_ingest(con):
    print("\n=== THREAD 3: Emerging clusters — ingest v9 T09 labels ===")
    labels_path = PROJECT_ROOT / "exploration-archive/v9_final_opus_47/artifacts/shared/swe_archetype_labels.parquet"
    labels = pd.read_parquet(labels_path)
    print(f"Loaded {len(labels)} labels, {labels['archetype_name'].nunique()} archetype names")

    # Join labels onto unified_core via uid
    con.register("labels", labels)
    sql = f"""
      WITH joined AS (
        SELECT u.uid, u.period,
               CASE WHEN u.period LIKE '2024%' THEN '2024' ELSE '2026' END AS period_bucket,
               u.title, u.is_swe, u.seniority_3level, u.yoe_min_years_llm,
               u.company_name_canonical, u.is_aggregator,
               regexp_matches(u.description, '{AI_VOCAB_PATTERN}') AS ai_match,
               l.archetype, l.archetype_name
        FROM '{CORE}' u
        INNER JOIN labels l USING (uid)
        WHERE u.is_english = true AND u.date_flag = 'ok' AND u.is_swe
      )
      SELECT * FROM joined
    """
    j = con.execute(sql).df()
    print(f"Joined {len(j)} SWE rows")
    print("Period bucket distribution:", j["period_bucket"].value_counts().to_dict())

    # Per archetype × period growth
    g = (j.groupby(["archetype_name", "period_bucket"])
           .size().unstack(fill_value=0))
    g["total"] = g.sum(axis=1)
    g["share_2024"] = g.get("2024", 0) / max(int(g.get("2024", 0).sum()), 1)
    g["share_2026"] = g.get("2026", 0) / max(int(g.get("2026", 0).sum()), 1)
    g["share_change_pp"] = (g["share_2026"] - g["share_2024"]) * 100
    g["growth_ratio"] = (g.get("2026", 0) + 1) / (g.get("2024", 0) + 1)
    g = g.sort_values("share_change_pp", ascending=False)
    g.to_csv(TABLES / "S27_thread3_v9_archetype_growth.csv")
    print("\nTop 12 by share change (pp):")
    print(g.head(12)[["2024", "2026", "share_change_pp", "growth_ratio"]])
    print("\nBottom 12:")
    print(g.tail(12)[["2024", "2026", "share_change_pp", "growth_ratio"]])

    # AI mention by archetype
    ai_by_arch = (j.groupby("archetype_name")
                    .agg(n=("uid", "count"), ai_rate=("ai_match", "mean"))
                    .sort_values("ai_rate", ascending=False))
    ai_by_arch.to_csv(TABLES / "S27_thread3_archetype_ai_density.csv")
    print("\nAI rate by archetype (top 8):")
    print(ai_by_arch.head(8))

    # YOE by archetype
    yoe_by_arch = (j.groupby("archetype_name")
                     .agg(n=("uid", "count"),
                          mean_yoe=("yoe_min_years_llm", "mean"),
                          median_yoe=("yoe_min_years_llm", "median")))
    yoe_by_arch.to_csv(TABLES / "S27_thread3_archetype_yoe.csv")

    # Figure: top growers vs shrinkers
    movers = pd.concat([g.head(8), g.tail(5)])
    movers = movers[movers.index != "noise/outlier"]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in movers["share_change_pp"]]
    ax.barh(movers.index, movers["share_change_pp"], color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Share change 2026 − 2024 (percentage points within v9 8k sample)")
    ax.set_title("S27 Thread 3a — v9 T09 archetypes: who grew, who shrank")
    fig.tight_layout()
    fig.savefig(FIGURES / "S27_thread3a_v9_archetype_movers.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    return g, ai_by_arch


def thread3_tfidf_emerging(con):
    print("\n=== THREAD 3b: TF-IDF emergent terms ===")
    # Lightweight TF-IDF on bucketed periods.
    # We sample to control memory: 12k per period from SWE rows.
    sql = f"""
      SELECT
             CASE WHEN period LIKE '2024%' THEN '2024' ELSE '2026' END AS period_bucket,
             title, description
      FROM '{CORE}'
      WHERE {CORE_FILTER} AND is_swe AND description IS NOT NULL AND length(description) > 200
        AND COALESCE(is_aggregator, false) = false
      USING SAMPLE 24000 ROWS
    """
    df = con.execute(sql).df()
    # Re-balance to ~equal per period
    n_per = df["period_bucket"].value_counts().min()
    balanced = pd.concat([df[df.period_bucket == p].sample(n=int(min(n_per, 12000)), random_state=0)
                          for p in df["period_bucket"].unique()])
    print(f"Balanced sample: {balanced['period_bucket'].value_counts().to_dict()}")

    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(
        ngram_range=(1, 3),
        max_df=0.4, min_df=20,
        max_features=20000,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
    )
    texts = balanced["description"].tolist()
    X = vec.fit_transform(texts)
    print(f"TF-IDF matrix: {X.shape}")
    terms = np.array(vec.get_feature_names_out())

    # Compute period-mean TF-IDF
    period = balanced["period_bucket"].values
    mask_2024 = period == "2024"
    mask_2026 = period == "2026"
    mean_2024 = np.asarray(X[mask_2024].mean(axis=0)).ravel()
    mean_2026 = np.asarray(X[mask_2026].mean(axis=0)).ravel()

    # Document-level appearance rate per period (binary >0)
    rate_2024 = np.asarray((X[mask_2024] > 0).mean(axis=0)).ravel()
    rate_2026 = np.asarray((X[mask_2026] > 0).mean(axis=0)).ravel()

    delta = rate_2026 - rate_2024
    df_terms = pd.DataFrame({
        "term": terms,
        "rate_2024": rate_2024,
        "rate_2026": rate_2026,
        "delta_pp": delta * 100,
        "tfidf_2024": mean_2024,
        "tfidf_2026": mean_2026,
    })
    # Apply emergent-term criteria: rate_2026 > 1% AND rate_2024 < 0.1%
    emergent = df_terms[(df_terms["rate_2026"] > 0.01) & (df_terms["rate_2024"] < 0.001)] \
                 .sort_values("rate_2026", ascending=False)
    print(f"Emergent terms (>1% in 2026, <0.1% in 2024): {len(emergent)}")
    print(emergent.head(40).to_string(index=False))

    # Disappearing terms: rate_2024 > 1% AND rate_2026 < 0.5% (relative collapse)
    disappearing = df_terms[(df_terms["rate_2024"] > 0.01) & (df_terms["rate_2026"] < 0.005)] \
                     .sort_values("rate_2024", ascending=False)
    print(f"\nDisappearing terms (>1% in 2024, <0.5% in 2026): {len(disappearing)}")
    print(disappearing.head(20).to_string(index=False))

    df_terms.to_csv(TABLES / "S27_thread3b_tfidf_all_terms.csv", index=False)
    emergent.to_csv(TABLES / "S27_thread3b_tfidf_emergent.csv", index=False)
    disappearing.to_csv(TABLES / "S27_thread3b_tfidf_disappearing.csv", index=False)

    # Figure: top emergent and disappearing terms
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    top_em = emergent.head(15)
    axes[0].barh(top_em["term"][::-1], top_em["rate_2026"][::-1] * 100, color="#2ca02c")
    axes[0].set_xlabel("2026 doc-rate (%)")
    axes[0].set_title("S27 Thread 3b — Top 15 emergent terms\n(>1% 2026, <0.1% 2024)")

    top_dis = disappearing.head(15)
    axes[1].barh(top_dis["term"][::-1], top_dis["rate_2024"][::-1] * 100, color="#d62728")
    axes[1].set_xlabel("2024 doc-rate (%)")
    axes[1].set_title("Top 15 disappearing terms\n(>1% 2024, <0.5% 2026)")

    fig.tight_layout()
    fig.savefig(FIGURES / "S27_thread3b_tfidf_emergent.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    return df_terms, emergent, disappearing


# ---------------------------------------------------------------------------
# THREAD 4 — Legacy substitution: T36 nearest-neighbor vs YoY title-frequency-delta
# ---------------------------------------------------------------------------

def thread4_legacy(con):
    print("\n=== THREAD 4: Legacy substitution ===")
    # Method (a) — ingest T36 substitution_table_top1
    t36 = pd.read_csv(PROJECT_ROOT / "exploration-archive/v9_final_opus_47/tables/T36/substitution_table_top1.csv")
    t36_ai = pd.read_csv(PROJECT_ROOT / "exploration-archive/v9_final_opus_47/tables/T36/ai_vocab_comparison.csv")
    print("T36 substitution table:")
    print(t36)
    print("\nT36 neighbor AI rates:")
    print(t36_ai)
    mean_neighbor_ai = t36_ai["neighbor_ai_strict_rate"].mean()
    print(f"Mean neighbor AI rate (T36): {mean_neighbor_ai:.4f}")

    # Verify market average AI-strict on unified_core 2026
    sql = f"""
      SELECT AVG(CASE WHEN regexp_matches(description, '{AI_VOCAB_PATTERN}') THEN 1.0 ELSE 0.0 END) AS ai_rate,
             COUNT(*) AS n
      FROM '{CORE}'
      WHERE {CORE_FILTER} AND is_swe AND period LIKE '2026%'
    """
    market = con.execute(sql).df().iloc[0].to_dict()
    print(f"Market AI rate 2026 (unified_core): {market}")

    # Method (b) — YoY title-frequency-delta (no embedding)
    sql_titles = f"""
      WITH base AS (
        SELECT LOWER(TRIM(title)) AS title_norm,
               CASE WHEN period LIKE '2024%' THEN '2024' ELSE '2026' END AS bucket,
               regexp_matches(description, '{AI_VOCAB_PATTERN}') AS ai
        FROM '{CORE}'
        WHERE {CORE_FILTER} AND is_swe
          AND title IS NOT NULL
          AND COALESCE(is_aggregator, false) = false
      )
      SELECT title_norm,
             SUM(CASE WHEN bucket='2024' THEN 1 ELSE 0 END) AS n_2024,
             SUM(CASE WHEN bucket='2026' THEN 1 ELSE 0 END) AS n_2026,
             SUM(CASE WHEN bucket='2024' AND ai THEN 1 ELSE 0 END) AS n_ai_2024,
             SUM(CASE WHEN bucket='2026' AND ai THEN 1 ELSE 0 END) AS n_ai_2026
      FROM base
      GROUP BY title_norm
      HAVING SUM(CASE WHEN bucket='2024' THEN 1 ELSE 0 END) >= 5
          OR SUM(CASE WHEN bucket='2026' THEN 1 ELSE 0 END) >= 5
    """
    tit = con.execute(sql_titles).df()
    n_2024_tot = int(tit["n_2024"].sum())
    n_2026_tot = int(tit["n_2026"].sum())
    tit["share_2024"] = tit["n_2024"] / n_2024_tot
    tit["share_2026"] = tit["n_2026"] / n_2026_tot
    tit["share_delta_pp"] = (tit["share_2026"] - tit["share_2024"]) * 100

    # "Disappearing" titles: present in 2024 with n>=5 but n_2026 < n_2024 / 4
    disappearing_titles = tit[(tit["n_2024"] >= 5) & (tit["n_2026"] < tit["n_2024"] / 4)] \
                            .sort_values("share_delta_pp", ascending=True)
    print(f"\nDisappearing titles (n_2024 >= 5, n_2026 < n_2024 / 4): {len(disappearing_titles)}")
    print(disappearing_titles.head(20).to_string(index=False))
    disappearing_titles.to_csv(TABLES / "S27_thread4_disappearing_titles_freqdelta.csv", index=False)

    # Emerging titles: n_2024 < 5 AND n_2026 >= 30
    emerging_titles = tit[(tit["n_2024"] < 5) & (tit["n_2026"] >= 30)] \
                        .sort_values("n_2026", ascending=False)
    emerging_titles.to_csv(TABLES / "S27_thread4_emerging_titles_freqdelta.csv", index=False)
    print(f"\nEmerging titles (n_2024 < 5, n_2026 >= 30): {len(emerging_titles)}")
    print(emerging_titles.head(15).to_string(index=False))

    # Verify the "3.6% vs 14.4%" claim:
    # Identify the v9 disappearing titles, find their 2026 freq, compute AI rate among 2026 occurrences
    v9_disappearing = ["java architect", "drupal developer", "scala developer",
                       "java application developer", "database developer", "devops architect"]
    sql_check = f"""
      SELECT LOWER(TRIM(title)) AS title_norm,
             COUNT(*) AS n_2026,
             AVG(CASE WHEN regexp_matches(description, '{AI_VOCAB_PATTERN}') THEN 1.0 ELSE 0.0 END) AS ai_rate
      FROM '{CORE}'
      WHERE {CORE_FILTER} AND is_swe AND period LIKE '2026%'
        AND LOWER(TRIM(title)) IN ({','.join("'" + t + "'" for t in v9_disappearing)})
      GROUP BY 1 ORDER BY n_2026 DESC
    """
    check_2026 = con.execute(sql_check).df()
    print("\nv9 disappearing titles — 2026 prevalence + AI rate (unified_core):")
    print(check_2026)

    # Now compute average AI rate on the 2026 NEIGHBOR titles from T36
    neighbor_titles = t36["top_2026_neighbor"].tolist()
    sql_neigh = f"""
      SELECT LOWER(TRIM(title)) AS title_norm,
             COUNT(*) AS n_2026,
             AVG(CASE WHEN regexp_matches(description, '{AI_VOCAB_PATTERN}') THEN 1.0 ELSE 0.0 END) AS ai_rate
      FROM '{CORE}'
      WHERE {CORE_FILTER} AND is_swe AND period LIKE '2026%'
        AND LOWER(TRIM(title)) IN ({','.join("'" + t.lower() + "'" for t in neighbor_titles)})
      GROUP BY 1 ORDER BY n_2026 DESC
    """
    neigh_2026 = con.execute(sql_neigh).df()
    weighted_ai = (neigh_2026["ai_rate"] * neigh_2026["n_2026"]).sum() / neigh_2026["n_2026"].sum() if neigh_2026["n_2026"].sum() > 0 else float("nan")
    print(f"\nT36 neighbor titles in unified_core 2026 (weighted ai rate = {weighted_ai:.4f}):")
    print(neigh_2026)

    summary = pd.DataFrame([
        {"metric": "T36 ai-vocab comparison mean (canonical, AI-strict, exploration archive)",
         "value": float(mean_neighbor_ai)},
        {"metric": "T36 ai-vocab comparison mean as percentage", "value": float(mean_neighbor_ai * 100)},
        {"metric": "Neighbor titles weighted AI-strict rate on unified_core 2026 (re-derived)",
         "value": float(weighted_ai)},
        {"metric": "Market AI-strict rate on unified_core 2026 SWE",
         "value": float(market["ai_rate"])},
        {"metric": "Market AI-strict rate as percentage",
         "value": float(market["ai_rate"] * 100)},
    ])
    summary.to_csv(TABLES / "S27_thread4_legacy_substitution_check.csv", index=False)

    # Figure: legacy → modern, AI rate side-by-side
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_x = ["T36 disappearing\nsource AI rate",
              "T36 neighbor\nAI rate (canonical)",
              "Neighbor AI rate\n(unified_core 2026)",
              "Market AI rate\n(unified_core 2026)"]
    bars_y = [
        float(t36_ai["source_ai_strict_rate"].mean() * 100),
        float(mean_neighbor_ai * 100),
        float(weighted_ai * 100) if not np.isnan(weighted_ai) else 0,
        float(market["ai_rate"] * 100),
    ]
    colors = ["#888", "#ff7f0e", "#1f77b4", "#2ca02c"]
    ax.bar(bars_x, bars_y, color=colors)
    for i, v in enumerate(bars_y):
        ax.text(i, v + 0.3, f"{v:.2f}%", ha="center", fontsize=10)
    ax.set_ylabel("AI-strict mention rate (%)")
    ax.set_title("S27 Thread 4 — Legacy titles map to modern-stack neighbors that under-mention AI")
    fig.tight_layout()
    fig.savefig(FIGURES / "S27_thread4_legacy_substitution.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    return summary, disappearing_titles, emerging_titles


def main():
    con = duckdb.connect()
    out = {}
    out["thread1"] = thread1_applied_ai(con)
    out["thread2"] = thread2_fde(con)
    out["thread3a"] = thread3_clusters_ingest(con)
    out["thread3b"] = thread3_tfidf_emerging(con)
    out["thread4"] = thread4_legacy(con)
    print("\nAll threads complete.")


if __name__ == "__main__":
    main()
