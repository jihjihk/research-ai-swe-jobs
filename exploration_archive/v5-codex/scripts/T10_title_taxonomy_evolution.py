#!/usr/bin/env python
from __future__ import annotations

import math
import re
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]
UNIFIED = ROOT / "data" / "unified.parquet"
CLEANED = ROOT / "exploration" / "artifacts" / "shared" / "swe_cleaned_text.parquet"

REPORT_DIR = ROOT / "exploration" / "reports"
TABLE_DIR = ROOT / "exploration" / "tables" / "T10"
FIG_DIR = ROOT / "exploration" / "figures" / "T10"

LINKEDIN_FILTER = "source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe"


def ensure_dirs() -> None:
    for path in [REPORT_DIR, TABLE_DIR, FIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def qdf(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def save_csv(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    return path


def save_fig(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


TITLE_AI_TOOL_PAT = re.compile(
    r"\b(?:ai|llm|agent|copilot|cursor|claude|mcp|chatgpt|gemini|openai|anthropic)\b|"
    r"machine learning|generative ai|gen ai|mlops|prompt engineering|fine[- ]tuning",
    re.I,
)
TITLE_AI_DOMAIN_PAT = re.compile(r"\bdata\b|\bml\b|\bai\b", re.I)
TITLE_SENIORITY_PAT = re.compile(r"\b(?:senior|lead|principal|staff|junior|associate)\b", re.I)


def assert_regex_hygiene() -> None:
    assert TITLE_AI_TOOL_PAT.search("AI engineer")
    assert TITLE_AI_TOOL_PAT.search("machine learning engineer")
    assert TITLE_AI_TOOL_PAT.search("gen ai engineer")
    assert TITLE_AI_DOMAIN_PAT.search("data engineer")
    assert TITLE_SENIORITY_PAT.search("senior software engineer")
    assert TITLE_SENIORITY_PAT.search("associate product engineer")
    assert not TITLE_SENIORITY_PAT.search("software engineer")
    assert not TITLE_AI_TOOL_PAT.search("software engineer")


def classify_theme(title: str) -> str:
    t = title.lower()
    if re.search(r"\b(ai|llm|machine learning|mlops|gen ai|generative ai|agent|prompt|copilot|cursor|claude|openai|anthropic)\b", t):
        return "AI/ML"
    if re.search(r"\b(data engineer|data platform|data scientist|data analyst|analytics|business intelligence|bi engineer)\b", t):
        return "Data/Analytics"
    if re.search(r"\b(platform engineer|platform developer|cloud platform)\b", t):
        return "Platform"
    if re.search(r"\b(site reliability|sre|reliability engineer|reliability)\b", t):
        return "Reliability/SRE"
    if re.search(r"\b(embedded|firmware|systems engineer|system engineer|hardware)\b", t):
        return "Embedded/Systems"
    if re.search(r"\b(frontend|front end|web|ui engineer|ui/ux)\b", t):
        return "Frontend/Web"
    if re.search(r"\b(backend|back end|api engineer)\b", t):
        return "Backend/API"
    if re.search(r"\b(full stack|fullstack)\b", t):
        return "Full Stack"
    if re.search(r"\b(android|ios|mobile)\b", t):
        return "Mobile"
    if re.search(r"\b(security|cybersecurity|cyber security)\b", t):
        return "Security"
    if re.search(r"\b(qa|quality assurance|test engineer|test automation)\b", t):
        return "QA/Testing"
    if re.search(r"\b(devops|dev sec ops|devsecops)\b", t):
        return "DevOps"
    if re.search(r"\b(software engineer|software developer|developer|engineer)\b", t):
        return "General SWE"
    return "Other"


def classify_title_ai(title: str) -> str:
    t = title.lower()
    if re.search(r"\b(ai|llm|agent|copilot|cursor|claude|mcp|chatgpt|gemini|openai|anthropic)\b", t) or re.search(
        r"machine learning|generative ai|gen ai|mlops|prompt engineering|fine[- ]tuning", t
    ):
        return "ai_tool_or_domain"
    if re.search(r"\bdata\b|\bml\b", t):
        return "ai_domain"
    return "other"


def class_title_marker(title: str) -> str:
    t = title.lower()
    for marker, pat in [
        ("senior", r"\bsenior\b"),
        ("lead", r"\blead\b"),
        ("principal", r"\bprincipal\b"),
        ("staff", r"\bstaff\b"),
        ("junior", r"\bjunior\b"),
        ("associate", r"\bassociate\b"),
        ("manager", r"\bmanager\b"),
    ]:
        if re.search(pat, t):
            return marker
    return "none"


def text_similarity(doc_a: str, doc_b: str) -> float:
    if not doc_a or not doc_b:
        return float("nan")
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=4000)
    X = vectorizer.fit_transform([doc_a, doc_b])
    return float(cosine_similarity(X[0], X[1])[0, 0])


def describe_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = df.groupby(group_cols, dropna=False).agg(
        n=("n", "sum"),
        unique_titles=("title_normalized", "nunique"),
        top_title_n=("n", "max"),
    ).reset_index()
    out["unique_titles_per_1000"] = out["unique_titles"] * 1000 / out["n"]
    out["top_title_share"] = out["top_title_n"] / out["n"]
    return out


def title_stats(counts: pd.Series) -> tuple[float, float, float, float]:
    x = counts.astype(float).to_numpy()
    total = x.sum()
    if total <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    p = x / total
    hhi = float(np.square(p).sum())
    safe_p = p[p > 0]
    entropy = float(-(safe_p * np.log(safe_p)).sum())
    eff = float(np.exp(entropy))
    top_share = float(x.max() / total)
    return hhi, eff, top_share, total


def main() -> None:
    ensure_dirs()
    assert_regex_hygiene()
    con = duckdb.connect()

    # Title counts by source, period, and aggregation state.
    title_counts = qdf(
        con,
        f"""
        WITH base AS (
          SELECT source, period, title_normalized, is_aggregator, count(*) AS n
          FROM read_parquet('{UNIFIED.as_posix()}')
          WHERE {LINKEDIN_FILTER}
          GROUP BY 1,2,3,4
        )
        SELECT
          source,
          period,
          title_normalized,
          sum(n) AS n,
          sum(CASE WHEN NOT is_aggregator THEN n ELSE 0 END) AS n_noagg
        FROM base
        GROUP BY 1,2,3
        ORDER BY source, period, n DESC, title_normalized
        """
    )
    title_counts["theme"] = title_counts["title_normalized"].map(classify_theme)
    title_counts["ai_class"] = title_counts["title_normalized"].map(classify_title_ai)
    title_counts["seniority_marker"] = title_counts["title_normalized"].map(class_title_marker)
    save_csv(title_counts, "T10_title_counts_by_source_period.csv")

    # Source-level concentration summaries.
    source_title_counts = (
        title_counts.groupby(["source", "title_normalized"], dropna=False)
        .agg(n=("n", "sum"), n_noagg=("n_noagg", "sum"), theme=("theme", "first"), ai_class=("ai_class", "first"), seniority_marker=("seniority_marker", "first"))
        .reset_index()
    )
    source_summary_rows = []
    for source, g in source_title_counts.groupby("source", dropna=False):
        hhi, eff, top_share, total = title_stats(g["n"])
        hhi_noagg, eff_noagg, top_share_noagg, total_noagg = title_stats(g["n_noagg"])
        source_summary_rows.append(
            {
                "source": source,
                "n": total,
                "unique_titles": g["title_normalized"].nunique(),
                "unique_titles_noagg": int((g["n_noagg"] > 0).sum()),
                "top_title_n": float(g["n"].max()),
                "top_title_n_noagg": float(g["n_noagg"].max()),
                "hhi": hhi,
                "hhi_noagg": hhi_noagg,
                "effective_titles": eff,
                "effective_titles_noagg": eff_noagg,
                "top_title_share": top_share,
                "top_title_share_noagg": top_share_noagg,
                "unique_titles_per_1000": g["title_normalized"].nunique() * 1000 / total,
                "unique_titles_per_1000_noagg": int((g["n_noagg"] > 0).sum()) * 1000 / total,
            }
        )
    source_summary = pd.DataFrame(source_summary_rows)
    source_summary["unique_titles_per_1000"] = source_summary["unique_titles"] * 1000 / source_summary["n"]
    source_summary["unique_titles_per_1000_noagg"] = source_summary["unique_titles_noagg"] * 1000 / source_summary["n"]
    save_csv(source_summary, "T10_title_concentration_by_source.csv")

    source_period_summary = (
        title_counts.groupby(["source", "period"], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "n": g["n"].sum(),
                    "unique_titles": g["title_normalized"].nunique(),
                    "unique_titles_noagg": g.loc[g["n_noagg"] > 0, "title_normalized"].nunique(),
                    "top_title_n": g["n"].max(),
                    "top_title_n_noagg": g["n_noagg"].max(),
                    "hhi": float(np.square(g["n"] / g["n"].sum()).sum()),
                    "hhi_noagg": float(np.square(g["n_noagg"] / g["n_noagg"].sum()).sum()),
                }
            )
        )
        .reset_index()
    )
    source_period_summary["unique_titles_per_1000"] = source_period_summary["unique_titles"] * 1000 / source_period_summary["n"]
    source_period_summary["unique_titles_per_1000_noagg"] = source_period_summary["unique_titles_noagg"] * 1000 / source_period_summary["n"]
    source_period_summary["top_title_share"] = source_period_summary["top_title_n"] / source_period_summary["n"]
    source_period_summary["top_title_share_noagg"] = source_period_summary["top_title_n_noagg"] / source_period_summary["n"]
    save_csv(source_period_summary, "T10_title_concentration_by_source_period.csv")

    # Title overlap and calibration.
    source_sets = {
        source: set(source_title_counts.loc[source_title_counts.source == source, "title_normalized"])
        for source in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]
    }

    def jaccard(a: set[str], b: set[str]) -> float:
        return len(a & b) / len(a | b) if a or b else float("nan")

    overlap_rows = []
    for a, b in [("kaggle_arshkon", "kaggle_asaniczka"), ("kaggle_arshkon", "scraped"), ("kaggle_asaniczka", "scraped")]:
        overlap_rows.append(
            {
                "source_a": a,
                "source_b": b,
                "unique_a": len(source_sets[a]),
                "unique_b": len(source_sets[b]),
                "overlap": len(source_sets[a] & source_sets[b]),
                "union": len(source_sets[a] | source_sets[b]),
                "jaccard": jaccard(source_sets[a], source_sets[b]),
                "a_only": len(source_sets[a] - source_sets[b]),
                "b_only": len(source_sets[b] - source_sets[a]),
            }
        )
    overlap_df = pd.DataFrame(overlap_rows)
    save_csv(overlap_df, "T10_title_overlap_summary.csv")

    # New / disappeared titles for the arshkon -> scraped comparison.
    arshkon = source_title_counts[source_title_counts.source == "kaggle_arshkon"].copy()
    scraped = source_title_counts[source_title_counts.source == "scraped"].copy()
    title_union = (
        arshkon[["title_normalized", "n"]]
        .merge(scraped[["title_normalized", "n"]], on="title_normalized", how="outer", suffixes=("_arshkon", "_scraped"))
        .fillna(0)
    )
    title_union["theme"] = title_union["title_normalized"].map(classify_theme)
    title_union["ai_class"] = title_union["title_normalized"].map(classify_title_ai)
    title_union["marker"] = title_union["title_normalized"].map(class_title_marker)
    title_union["change_class"] = np.select(
        [
            (title_union.n_arshkon == 0) & (title_union.n_scraped > 0),
            (title_union.n_arshkon > 0) & (title_union.n_scraped == 0),
        ],
        ["new_2026", "disappeared_2026"],
        default="shared",
    )
    new_titles = title_union[title_union.change_class == "new_2026"].sort_values(["n_scraped", "title_normalized"], ascending=[False, True])
    disappeared_titles = title_union[title_union.change_class == "disappeared_2026"].sort_values(["n_arshkon", "title_normalized"], ascending=[False, True])
    shared_titles = title_union[title_union.change_class == "shared"].sort_values(["n_arshkon", "n_scraped", "title_normalized"], ascending=[False, False, True])
    save_csv(new_titles, "T10_new_titles_2026_vs_2024.csv")
    save_csv(disappeared_titles, "T10_disappeared_titles_2024_vs_2026.csv")
    save_csv(shared_titles, "T10_shared_titles.csv")

    # New title theme summaries.
    new_theme_summary = (
        new_titles.groupby("theme", dropna=False)
        .agg(unique_titles=("title_normalized", "nunique"), postings=("n_scraped", "sum"))
        .reset_index()
        .sort_values(["postings", "unique_titles"], ascending=[False, False])
    )
    new_theme_summary["postings_share"] = new_theme_summary["postings"] / new_titles["n_scraped"].sum()
    save_csv(new_theme_summary, "T10_new_title_theme_summary.csv")

    disappeared_theme_summary = (
        disappeared_titles.groupby("theme", dropna=False)
        .agg(unique_titles=("title_normalized", "nunique"), postings=("n_arshkon", "sum"))
        .reset_index()
        .sort_values(["postings", "unique_titles"], ascending=[False, False])
    )
    disappeared_theme_summary["postings_share"] = disappeared_theme_summary["postings"] / disappeared_titles["n_arshkon"].sum()
    save_csv(disappeared_theme_summary, "T10_disappeared_title_theme_summary.csv")

    # AI-related title shares and seniority-marker shares.
    marker_rows = []
    for source in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]:
        for period in sorted(title_counts.loc[title_counts.source == source, "period"].unique()):
            sub = title_counts[(title_counts.source == source) & (title_counts.period == period)].copy()
            n = sub["n"].sum()
            marker_rows.append(
                {
                    "source": source,
                    "period": period,
                    "n": int(n),
                    "ai_hybrid_postings": int(sub.loc[sub.ai_class != "other", "n"].sum()),
                    "ai_domain_postings": int(sub.loc[sub.ai_class == "ai_domain", "n"].sum()),
                    "ai_tool_or_domain_postings": int(sub.loc[sub.ai_class == "ai_tool_or_domain", "n"].sum()),
                    "seniority_marked_postings": int(sub.loc[sub.seniority_marker != "none", "n"].sum()),
                    "seniority_marked_titles": int((sub.seniority_marker != "none").sum()),
                }
            )
    marker_df = pd.DataFrame(marker_rows)
    marker_df["ai_hybrid_share"] = marker_df["ai_hybrid_postings"] / marker_df["n"]
    marker_df["ai_domain_share"] = marker_df["ai_domain_postings"] / marker_df["n"]
    marker_df["seniority_marker_share"] = marker_df["seniority_marked_postings"] / marker_df["n"]
    save_csv(marker_df, "T10_ai_and_seniority_marker_shares.csv")

    source_marker_rows = []
    for source, g in source_title_counts.groupby("source", dropna=False):
        total = g["n"].sum()
        source_marker_rows.append(
            {
                "source": source,
                "n": total,
                "ai_hybrid_postings": float(g.loc[g.ai_class != "other", "n"].sum()),
                "ai_domain_postings": float(g.loc[g.ai_class == "ai_domain", "n"].sum()),
                "ai_tool_or_domain_postings": float(g.loc[g.ai_class == "ai_tool_or_domain", "n"].sum()),
                "seniority_marked_postings": float(g.loc[g.seniority_marker != "none", "n"].sum()),
                "seniority_marked_titles": int((g.seniority_marker != "none").sum()),
            }
        )
    source_marker_df = pd.DataFrame(source_marker_rows)
    source_marker_df["ai_hybrid_share"] = source_marker_df["ai_hybrid_postings"] / source_marker_df["n"]
    source_marker_df["ai_domain_share"] = source_marker_df["ai_domain_postings"] / source_marker_df["n"]
    source_marker_df["seniority_marker_share"] = source_marker_df["seniority_marked_postings"] / source_marker_df["n"]
    save_csv(source_marker_df, "T10_ai_and_seniority_marker_shares_by_source.csv")

    # Top titles shared across arshkon and scraped.
    shared_arshkon_scraped = title_union[(title_union.n_arshkon > 0) & (title_union.n_scraped > 0)].copy()
    top_shared = shared_arshkon_scraped.sort_values(["n_arshkon", "n_scraped", "title_normalized"], ascending=[False, False, True]).head(10)
    save_csv(top_shared, "T10_top_shared_titles.csv")

    # Title seniority shifts for the top shared titles.
    top_title_list = top_shared["title_normalized"].tolist()
    top_title_detail = qdf(
        con,
        f"""
        SELECT u.title_normalized,
               u.source,
               u.period,
               count(*) AS n,
               avg(CASE WHEN u.seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) AS entry_share_final,
               avg(CASE WHEN u.yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) AS yoe_le2_share,
               avg(u.yoe_extracted) AS mean_yoe,
               median(u.yoe_extracted) AS median_yoe
        FROM read_parquet('{UNIFIED.as_posix()}') u
        WHERE {LINKEDIN_FILTER}
          AND u.source IN ('kaggle_arshkon', 'kaggle_asaniczka', 'scraped')
          AND u.title_normalized IN ({','.join(sql_quote(t) for t in top_title_list)})
        GROUP BY 1,2,3
        ORDER BY title_normalized, source
        """
    )
    save_csv(top_title_detail, "T10_common_title_seniority_shift.csv")

    # Title-to-content alignment on the cleaned-text subset.
    top_title_text_rows = qdf(
        con,
        f"""
        SELECT u.title_normalized, u.source, c.period, c.text_source, c.description_cleaned
        FROM read_parquet('{UNIFIED.as_posix()}') u
        JOIN read_parquet('{CLEANED.as_posix()}') c USING (uid)
        WHERE {LINKEDIN_FILTER}
          AND u.source IN ('kaggle_arshkon', 'kaggle_asaniczka', 'scraped')
          AND c.text_source = 'llm'
          AND u.title_normalized IN ({','.join(sql_quote(t) for t in top_title_list)})
        """
    )
    similarity_rows = []
    calibration_rows = []
    for title in top_title_list:
        sub = top_title_text_rows[top_title_text_rows.title_normalized == title]
        docs = {}
        counts = {}
        for source in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]:
            texts = sub.loc[sub.source == source, "description_cleaned"].dropna().astype(str).tolist()
            docs[source] = " ".join(texts)
            counts[source] = len(texts)
        pair_specs = [
            ("arshkon_vs_scraped", "kaggle_arshkon", "scraped"),
            ("arshkon_vs_asaniczka", "kaggle_arshkon", "kaggle_asaniczka"),
        ]
        for pair_name, a, b in pair_specs:
            sim = text_similarity(docs[a], docs[b])
            similarity_rows.append(
                {
                    "title_normalized": title,
                    "pair": pair_name,
                    "source_a": a,
                    "source_b": b,
                    "similarity": sim,
                    "n_a": counts[a],
                    "n_b": counts[b],
                    "shared_llm_rows": min(counts[a], counts[b]),
                }
            )
    similarity_df = pd.DataFrame(similarity_rows)
    save_csv(similarity_df, "T10_title_content_similarity.csv")

    # A compact calibration table for title-space metrics.
    calib = []
    for metric in ["unique_titles_per_1000", "top_title_share", "ai_hybrid_share", "seniority_marker_share"]:
        if metric == "unique_titles_per_1000":
            values = source_summary.set_index("source")[metric]
            calib.append(
                {
                    "metric": metric,
                    "arshkon": float(values["kaggle_arshkon"]),
                    "asaniczka": float(values["kaggle_asaniczka"]),
                    "scraped": float(values["scraped"]),
                    "within_2024_diff": float(values["kaggle_arshkon"] - values["kaggle_asaniczka"]),
                    "cross_period_diff": float(values["scraped"] - values["kaggle_arshkon"]),
                    "signal_to_noise": abs(values["scraped"] - values["kaggle_arshkon"]) / max(abs(values["kaggle_arshkon"] - values["kaggle_asaniczka"]), 1e-9),
                }
            )
        elif metric == "top_title_share":
            values = source_summary.set_index("source")[metric]
            calib.append(
                {
                    "metric": metric,
                    "arshkon": float(values["kaggle_arshkon"]),
                    "asaniczka": float(values["kaggle_asaniczka"]),
                    "scraped": float(values["scraped"]),
                    "within_2024_diff": float(values["kaggle_arshkon"] - values["kaggle_asaniczka"]),
                    "cross_period_diff": float(values["scraped"] - values["kaggle_arshkon"]),
                    "signal_to_noise": abs(values["scraped"] - values["kaggle_arshkon"]) / max(abs(values["kaggle_arshkon"] - values["kaggle_asaniczka"]), 1e-9),
                }
            )
        else:
            vals = source_marker_df.set_index("source")[metric]
            calib.append(
                {
                    "metric": metric,
                    "arshkon": float(vals["kaggle_arshkon"]),
                    "asaniczka": float(vals["kaggle_asaniczka"]),
                    "scraped": float(vals["scraped"]),
                    "within_2024_diff": float(vals["kaggle_arshkon"] - vals["kaggle_asaniczka"]),
                    "cross_period_diff": float(vals["scraped"] - vals["kaggle_arshkon"]),
                    "signal_to_noise": abs(vals["scraped"] - vals["kaggle_arshkon"]) / max(abs(vals["kaggle_arshkon"] - vals["kaggle_asaniczka"]), 1e-9),
                }
            )
    calib_df = pd.DataFrame(calib)
    save_csv(calib_df, "T10_title_calibration_metrics.csv")

    # Figures.
    sns.set_theme(style="whitegrid")

    # Figure 1: title concentration by source.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_df = source_summary.copy()
    plot_df["source_label"] = plot_df["source"].map(
        {
            "kaggle_arshkon": "arshkon",
            "kaggle_asaniczka": "asaniczka",
            "scraped": "scraped",
        }
    )
    sns.barplot(data=plot_df, x="source_label", y="unique_titles_per_1000", ax=axes[0], color="#4C78A8")
    axes[0].set_title("Unique titles per 1,000 SWE postings")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Unique titles / 1,000")
    sns.barplot(data=plot_df, x="source_label", y="top_title_share", ax=axes[1], color="#F58518")
    axes[1].set_title("Top title share")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Share of SWE postings")
    save_fig(fig, "T10_title_concentration.png")

    # Figure 2: AI-hybrid title shares and seniority-marker shares.
    fig, ax = plt.subplots(figsize=(10, 4))
    marker_plot = marker_df.copy()
    marker_plot["source_label"] = marker_plot["source"].map(
        {"kaggle_arshkon": "arshkon", "kaggle_asaniczka": "asaniczka", "scraped": "scraped"}
    )
    marker_long = marker_plot.melt(
        id_vars=["source_label", "period"],
        value_vars=["ai_hybrid_share", "seniority_marker_share"],
        var_name="metric",
        value_name="share",
    )
    marker_long["metric"] = marker_long["metric"].map(
        {"ai_hybrid_share": "AI-related title share", "seniority_marker_share": "Explicit seniority marker share"}
    )
    marker_long["group"] = marker_long["source_label"] + " " + marker_long["period"]
    order = ["asaniczka 2024-01", "arshkon 2024-04", "scraped 2026-03", "scraped 2026-04"]
    marker_long["group"] = pd.Categorical(marker_long["group"], categories=order, ordered=True)
    sns.barplot(data=marker_long, x="group", y="share", hue="metric", ax=ax)
    ax.set_title("Title-level AI and explicit seniority markers")
    ax.set_xlabel("")
    ax.set_ylabel("Share of SWE postings")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False)
    save_fig(fig, "T10_ai_and_marker_shares.png")

    # Figure 3: top shared title content similarity.
    sim_plot = similarity_df.copy()
    sim_plot["source_pair"] = sim_plot["pair"].map(
        {"arshkon_vs_scraped": "arshkon vs scraped", "arshkon_vs_asaniczka": "arshkon vs asaniczka"}
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=sim_plot,
        y="title_normalized",
        x="similarity",
        hue="source_pair",
        ax=ax,
        order=top_title_list,
    )
    ax.set_title("TF-IDF cosine similarity of aggregate descriptions for shared titles")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("")
    ax.legend(frameon=False)
    save_fig(fig, "T10_title_content_similarity.png")

    # Figure 4: new 2026 title themes.
    fig, ax = plt.subplots(figsize=(10, 5))
    theme_plot = new_theme_summary.copy().sort_values("postings", ascending=True)
    sns.barplot(data=theme_plot, y="theme", x="postings", ax=ax, color="#54A24B")
    ax.set_title("Themes among titles new in scraped 2026")
    ax.set_xlabel("Postings")
    ax.set_ylabel("")
    save_fig(fig, "T10_new_title_themes.png")


if __name__ == "__main__":
    main()
