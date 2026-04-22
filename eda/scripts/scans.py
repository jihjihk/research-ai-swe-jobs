"""
Phase B hypothesis-driven scans for `data/unified.parquet`.

Pre-committed metric, pre-committed filter, one figure, one CSV per scan.
Default filter applied throughout (unless overridden per scan):
    source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'

Scans:
  S1  H1  AI-vocab prevalence by period × seniority_3level (SWE)
  S2  H1  AI-requirement intensity (LLM column) + ghost_assessment_llm by period
  S3  H2  New-AI-title emergence share by period
  S4  H2  Title n-gram top-delta 2024 vs 2026 (SWE)
  S5  H3  Remote-posting share by period × top metros (SWE)
  S6  H3  Aggregator share by period + top-10 aggregator composition
  S7  H4  Industry dispersion of SWE (Herfindahl + top-10 industries)
  S8  H4  Non-tech industry SWE share by period
  S9  H5  Junior-vs-senior trajectory (3 panels)
  S10 H6  Big Tech vs rest — volume share + AI-vocab rate
  S11 H7  SWE vs control — volume, remote, aggregator, AI-vocab (2026 only)
  Sv  --  v8 replication: seniority mix + mean description_length by period

Run:
  ./.venv/bin/python eda/scripts/scans.py
"""

from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

random.seed(0)
np.random.seed(0)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_PATH = PROJECT_ROOT / "data" / "unified.parquet"
TABLES_DIR = PROJECT_ROOT / "eda" / "tables"
FIGURES_DIR = PROJECT_ROOT / "eda" / "figures"

# ---------------------------------------------------------------------------
# Pre-committed vocabulary lists (DO NOT tweak post-hoc based on results)
# ---------------------------------------------------------------------------

AI_VOCAB_PHRASES = [
    # Models and tools
    "llm", "gpt", "chatgpt", "claude", "copilot", "openai", "anthropic",
    "gemini", "bard", "mistral", "llama",
    # Concepts
    "large language model", "generative ai", "genai", "gen ai",
    "foundation model", "transformer model",
    # Agents/tooling
    "ai agent", "agentic", "ai-powered", "ai tooling", "ai-assisted",
    # RAG / retrieval
    "rag", "retrieval augmented", "vector database", "vector store",
    "embedding model",
    # Roles/skills (phrase-level, not title-level)
    "prompt engineering", "prompt engineer", "ml ops", "mlops", "llmops",
    # Tools
    "cursor ide", "windsurf ide", "github copilot",
]

# Pattern: word-boundary case-insensitive alternation
AI_VOCAB_PATTERN = (
    r"(?i)\b(" + "|".join(re.escape(p) for p in AI_VOCAB_PHRASES) + r")\b"
)

NEW_AI_TITLE_PATTERNS = [
    "ai engineer", "ml engineer", "llm engineer", "agent engineer",
    "applied ai", "applied ml", "prompt engineer", "ai/ml",
    "machine learning engineer", "mlops engineer",
    "ai research", "applied scientist", "genai engineer",
    "foundation model", "founding ai",
]
NEW_AI_TITLE_PATTERN = (
    r"(?i)(" + "|".join(re.escape(p) for p in NEW_AI_TITLE_PATTERNS) + r")"
)

BIG_TECH_CANONICAL = {
    # Canonical-name exact matches (lowercased). `company_name_canonical` is Title-cased
    # in the parquet (e.g. "Google", "Amazon Web Services (AWS)") so we lowercase both
    # sides before comparing. Variants for the same parent company are enumerated.
    "alphabet", "google",
    "meta", "facebook",
    "amazon", "amazon web services", "amazon web services (aws)", "aws",
    "amazon lab126", "amazon music", "prime video & amazon mgm studios",
    "apple",
    "microsoft", "microsoft ai",
    "oracle",
    "netflix",
    "block", "square",
    "uber",
    "airbnb",
    "salesforce",
    "anthropic",
    "openai",
    "nvidia",
    "tesla",
    "adobe",
    "ibm",
    "linkedin",
}

TECH_INDUSTRIES = {
    "Computer Software",
    "Information Technology and Services",
    "Information Technology & Services",
    "Internet",
    "Computer Games",
    "Computer Hardware",
    "Computer Networking",
    "Computer & Network Security",
    "Semiconductors",
    "Telecommunications",
    "Software Development",
    "IT Services and IT Consulting",
    "Technology, Information and Internet",
}

# Default filter — applied in every scan unless overridden
DEFAULT_FILTER = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'"

# ---------------------------------------------------------------------------
# Substrate (text column) for AI-vocab matching
# ---------------------------------------------------------------------------
# `description_core_llm` is the LLM-stripped, boilerplate-removed substrate
# (median ~2,423 chars vs raw ~4,280; coverage 99.2% on unified_core).
# Per the methodology protocol (eda/research_memos/methodology_protocol.md)
# this is the primary substrate for AI-vocab regex matching. Length-comparison
# scans (Sv) intentionally read both raw and core side-by-side and should NOT
# use text_col(); they hardcode their column names.
SUBSTRATE = "description_core_llm"


def text_col(table_alias: str | None = None) -> str:
    """Return the AI-vocab matching substrate column, optionally aliased.

    STRICT-CORE: returns the bare `description_core_llm` column with no
    fallback. Callers MUST pair this with `text_filter()` in their WHERE
    clause to avoid scanning NULL rows. Per the methodology protocol
    (eda/research_memos/methodology_protocol.md), this is the primary
    substrate for AI-vocab regex matching; ~0.8% of `unified_core.parquet`
    rows lack the column and are filtered out.
    """
    prefix = f"{table_alias}." if table_alias else ""
    return f"{prefix}{SUBSTRATE}"


def text_filter(table_alias: str | None = None) -> str:
    """Return the SQL fragment '<prefix>description_core_llm IS NOT NULL'.

    Use in WHERE clauses paired with text_col() to enforce strict-core
    semantics — scans only rows where the LLM-cleaned substrate exists.
    """
    prefix = f"{table_alias}." if table_alias else ""
    return f"{prefix}{SUBSTRATE} IS NOT NULL"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fig_caption(ax, text):
    ax.text(0.01, -0.18, text, transform=ax.transAxes, fontsize=8,
            color="#555", ha="left", va="top", wrap=True)


def save_fig(fig, name):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def period_order(df, col="period"):
    return sorted(df[col].unique())


# ---------------------------------------------------------------------------
# S1 — H1 — AI-vocab prevalence by period × seniority_3level (SWE)
# ---------------------------------------------------------------------------

def scan_s1(con):
    sql = f"""
      SELECT period, seniority_3level,
             COUNT(*) AS n,
             SUM(CASE WHEN regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}') THEN 1 ELSE 0 END) AS n_ai
      FROM '{UNIFIED_PATH}'
      WHERE {DEFAULT_FILTER} AND is_swe = true AND {text_filter()}
      GROUP BY 1,2
      ORDER BY 1,2
    """
    df = con.execute(sql).df()
    df["ai_rate"] = df["n_ai"] / df["n"]
    df.to_csv(TABLES_DIR / "S1_ai_vocab_by_period_seniority.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    levels = ["junior", "mid", "senior", "unknown"]
    colors = {"junior": "#d62728", "mid": "#ff7f0e", "senior": "#2ca02c", "unknown": "#7f7f7f"}
    periods = period_order(df)
    x = np.arange(len(periods))
    width = 0.2
    for i, lvl in enumerate(levels):
        sub = df[df["seniority_3level"] == lvl].set_index("period").reindex(periods)
        ax.bar(x + (i - 1.5) * width, sub["ai_rate"].fillna(0), width,
               label=lvl, color=colors.get(lvl, "#888"))
    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=25)
    ax.set_ylabel("share of SWE postings with ≥1 AI-vocab mention")
    ax.set_title("S1 (H1) — AI-vocab prevalence by period × seniority (SWE, LinkedIn, EN, date_ok)")
    ax.legend(title="seniority_3level")
    fig_caption(ax, f"n = {int(df['n'].sum()):,} SWE postings; filter: {DEFAULT_FILTER} AND is_swe")
    save_fig(fig, "S1_ai_vocab_by_period_seniority")
    return df


# ---------------------------------------------------------------------------
# S2 — H1 — AI-requirement intensity via LLM column, + ghost rate
# ---------------------------------------------------------------------------

def scan_s2(con):
    # Among labeled rows, compute AI-vocab rate on description_core_llm (cleaner),
    # and ghost-assessment rate. Use description_core_llm which is LLM-cleaned.
    sql = f"""
      SELECT period,
             COUNT(*) AS n_labeled,
             SUM(CASE WHEN regexp_matches(description_core_llm, '{AI_VOCAB_PATTERN}') THEN 1 ELSE 0 END) AS n_ai_labeled,
             SUM(CASE WHEN ghost_assessment_llm = 'inflated' THEN 1 ELSE 0 END) AS n_inflated,
             SUM(CASE WHEN ghost_assessment_llm = 'ghost_likely' THEN 1 ELSE 0 END) AS n_ghost,
             SUM(CASE WHEN swe_classification_llm = 'SWE' THEN 1 ELSE 0 END) AS n_swe_llm
      FROM '{UNIFIED_PATH}'
      WHERE {DEFAULT_FILTER}
        AND is_swe = true
        AND llm_extraction_coverage = 'labeled'
        AND llm_classification_coverage = 'labeled'
      GROUP BY 1
      ORDER BY 1
    """
    df = con.execute(sql).df()
    df["ai_rate_labeled"] = df["n_ai_labeled"] / df["n_labeled"]
    df["inflated_rate"] = df["n_inflated"] / df["n_labeled"]
    df["ghost_rate"] = df["n_ghost"] / df["n_labeled"]
    df.to_csv(TABLES_DIR / "S2_ai_requirement_intensity.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    width = 0.28
    ax.bar(x - width, df["ai_rate_labeled"], width, label="AI-vocab rate (cleaned text)", color="#1f77b4")
    ax.bar(x, df["inflated_rate"], width, label="ghost_assessment='inflated'", color="#ff7f0e")
    ax.bar(x + width, df["ghost_rate"], width, label="ghost_assessment='ghost_likely'", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(df["period"], rotation=25)
    ax.set_ylabel("share of labeled SWE rows")
    ax.set_title("S2 (H1) — AI-requirement intensity + ghost rates (LLM labeled only)")
    ax.legend()
    fig_caption(ax, f"n_labeled = {int(df['n_labeled'].sum()):,}; filter: {DEFAULT_FILTER} AND is_swe AND llm_*_coverage='labeled'")
    save_fig(fig, "S2_ai_requirement_intensity")
    return df


# ---------------------------------------------------------------------------
# S3 — H2 — New-AI-title emergence share
# ---------------------------------------------------------------------------

def scan_s3(con):
    sql = f"""
      SELECT period,
             COUNT(*) AS n_swe,
             SUM(CASE WHEN regexp_matches(title_normalized, '{NEW_AI_TITLE_PATTERN}') THEN 1 ELSE 0 END) AS n_new_ai
      FROM '{UNIFIED_PATH}'
      WHERE {DEFAULT_FILTER} AND is_swe = true
      GROUP BY 1
      ORDER BY 1
    """
    df = con.execute(sql).df()
    df["new_ai_share"] = df["n_new_ai"] / df["n_swe"]
    df.to_csv(TABLES_DIR / "S3_new_ai_title_share.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["period"], df["new_ai_share"], color="#9467bd")
    for i, row in df.iterrows():
        ax.text(i, row["new_ai_share"] + 0.003,
                f"{row['new_ai_share']*100:.1f}%\n(n={int(row['n_new_ai']):,})",
                ha="center", fontsize=8)
    ax.set_ylabel("share of SWE postings")
    ax.set_title("S3 (H2) — Share of SWE titles matching new-AI-title patterns")
    ax.tick_params(axis="x", rotation=25)
    fig_caption(ax, f"Patterns: {', '.join(NEW_AI_TITLE_PATTERNS[:5])}…; n = {int(df['n_swe'].sum()):,}")
    save_fig(fig, "S3_new_ai_title_share")
    return df


# ---------------------------------------------------------------------------
# S4 — H2 — Title n-gram top-delta 2024 vs 2026
# ---------------------------------------------------------------------------

def scan_s4(con):
    # Pull SWE LinkedIn titles, bucket by year (2024 vs 2026 = Kaggle vs scraped),
    # compute top unigrams/bigrams per bucket, then compute share delta.
    sql = f"""
      SELECT CASE
               WHEN source LIKE 'kaggle%' THEN '2024'
               WHEN source = 'scraped' THEN '2026'
             END AS period_bucket,
             title_normalized
      FROM '{UNIFIED_PATH}'
      WHERE {DEFAULT_FILTER} AND is_swe = true
        AND title_normalized IS NOT NULL
    """
    df = con.execute(sql).df()

    def tokens(t):
        return re.findall(r"[a-z0-9/+#.-]+", (t or "").lower())

    def ngrams(tokens, n):
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    STOP = {"&", "-", "/", "at", "of", "for", "in", "the", "and", "or", "a", "an", "to"}

    def extract(row_titles):
        uni = Counter()
        bi = Counter()
        n_titles = 0
        for t in row_titles:
            toks = [x for x in tokens(t) if x not in STOP and len(x) > 1]
            uni.update(toks)
            bi.update(ngrams(toks, 2))
            n_titles += 1
        return uni, bi, n_titles

    results = []
    by_bucket = {b: df[df["period_bucket"] == b]["title_normalized"].tolist()
                 for b in ["2024", "2026"]}
    counts = {b: extract(v) for b, v in by_bucket.items()}

    for bucket, (uni, bi, n) in counts.items():
        for tok, c in uni.most_common(200):
            results.append({"ngram": tok, "kind": "uni", "bucket": bucket,
                            "count": c, "share": c / n})
        for tok, c in bi.most_common(200):
            results.append({"ngram": tok, "kind": "bi", "bucket": bucket,
                            "count": c, "share": c / n})
    all_df = pd.DataFrame(results)
    wide = all_df.pivot_table(index=["ngram", "kind"], columns="bucket",
                              values="share", fill_value=0).reset_index()
    wide["delta_2026_minus_2024"] = wide["2026"] - wide["2024"]
    wide = wide.sort_values("delta_2026_minus_2024", ascending=False)

    # Keep top-30 positive and top-30 negative deltas for the table
    top_pos = wide.head(30).copy()
    top_neg = wide.tail(30).copy()
    out = pd.concat([top_pos, top_neg]).reset_index(drop=True)
    out.to_csv(TABLES_DIR / "S4_title_ngram_delta.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    top_pos_plot = top_pos.head(20).iloc[::-1]  # reverse for horizontal plot
    axes[0].barh(top_pos_plot["ngram"] + " (" + top_pos_plot["kind"] + ")",
                 top_pos_plot["delta_2026_minus_2024"] * 100, color="#2ca02c")
    axes[0].set_xlabel("Δ share percentage points (2026 − 2024)")
    axes[0].set_title("S4 (H2) — Rising title n-grams 2024→2026")
    top_neg_plot = top_neg.tail(20)
    axes[1].barh(top_neg_plot["ngram"] + " (" + top_neg_plot["kind"] + ")",
                 top_neg_plot["delta_2026_minus_2024"] * 100, color="#d62728")
    axes[1].set_xlabel("Δ share percentage points (2026 − 2024)")
    axes[1].set_title("Falling title n-grams")
    fig.suptitle(f"SWE LinkedIn titles, 2024 (Kaggle) vs 2026 (scraped); n_2024 = {counts['2024'][2]:,}, n_2026 = {counts['2026'][2]:,}")
    fig.tight_layout()
    save_fig(fig, "S4_title_ngram_delta")
    return out


# ---------------------------------------------------------------------------
# S5 — H3 — Remote-posting share by period × top-10 metros
# ---------------------------------------------------------------------------

def scan_s5(con):
    sql = f"""
      WITH swe AS (
        SELECT period, metro_area,
               SUM(CASE WHEN is_remote THEN 1 ELSE 0 END) AS n_remote,
               COUNT(*) AS n
        FROM '{UNIFIED_PATH}'
        WHERE {DEFAULT_FILTER} AND is_swe = true AND metro_area IS NOT NULL
        GROUP BY 1,2
      ),
      top_metros AS (
        SELECT metro_area FROM swe
        GROUP BY 1 ORDER BY SUM(n) DESC LIMIT 10
      )
      SELECT s.period, s.metro_area,
             s.n, s.n_remote,
             s.n_remote::DOUBLE / NULLIF(s.n, 0) AS remote_rate
      FROM swe s INNER JOIN top_metros t USING (metro_area)
      ORDER BY 1, 2
    """
    df = con.execute(sql).df()
    df.to_csv(TABLES_DIR / "S5_remote_share.csv", index=False)

    # Also compute overall remote rate per period
    overall = df.groupby("period").agg(
        n=("n", "sum"), n_remote=("n_remote", "sum")
    ).reset_index()
    overall["remote_rate"] = overall["n_remote"] / overall["n"]

    fig, ax = plt.subplots(figsize=(11, 6))
    pivot = df.pivot_table(index="metro_area", columns="period",
                           values="remote_rate", fill_value=np.nan)
    pivot = pivot.sort_values(by=pivot.columns[-1], ascending=False)
    pivot.plot(kind="bar", ax=ax, colormap="tab10", width=0.8)
    # overlay overall remote rate line
    ax2 = ax.twinx()
    ax2.plot(range(len(pivot)), [overall["remote_rate"].iloc[-1]] * len(pivot),
             "k--", alpha=0.4, label=f"All-metro avg ({overall['period'].iloc[-1]})")
    ax.set_ylabel("remote_inferred share")
    ax.set_title("S5 (H3) — Remote SWE posting share by metro × period (top-10 metros by SWE volume)")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.legend(title="period", fontsize=8)
    fig_caption(ax, f"n = {int(df['n'].sum()):,} SWE postings; all-period overall remote_rate printed in table")
    save_fig(fig, "S5_remote_share")
    return df


# ---------------------------------------------------------------------------
# S6 — H3 — Aggregator share + top-10 aggregator composition
# ---------------------------------------------------------------------------

def scan_s6(con):
    sql = f"""
      SELECT period,
             COUNT(*) AS n,
             SUM(CASE WHEN is_aggregator THEN 1 ELSE 0 END) AS n_agg
      FROM '{UNIFIED_PATH}'
      WHERE {DEFAULT_FILTER} AND is_swe = true
      GROUP BY 1 ORDER BY 1
    """
    df = con.execute(sql).df()
    df["agg_share"] = df["n_agg"] / df["n"]

    # Top aggregator names by period — use company_name_canonical as aggregator identity
    # (schema does not expose aggregator_name; is_aggregator flag + canonical name is the proxy)
    top_agg = con.execute(f"""
      SELECT period, company_name_canonical AS aggregator_name, COUNT(*) AS n
      FROM '{UNIFIED_PATH}'
      WHERE {DEFAULT_FILTER} AND is_swe = true AND is_aggregator = true
        AND company_name_canonical IS NOT NULL
      GROUP BY 1,2
      ORDER BY 1, n DESC
    """).df()

    df.to_csv(TABLES_DIR / "S6_aggregator_share.csv", index=False)
    top_agg.to_csv(TABLES_DIR / "S6_aggregator_top_names.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(df["period"], df["agg_share"], color="#8c564b")
    for i, row in df.iterrows():
        axes[0].text(i, row["agg_share"] + 0.003,
                     f"{row['agg_share']*100:.1f}%", ha="center", fontsize=9)
    axes[0].set_ylabel("aggregator share of SWE postings")
    axes[0].set_title("S6 (H3) — Aggregator share by period")
    axes[0].tick_params(axis="x", rotation=25)

    # Top 5 aggregator names stacked bar per period
    top5_per_period = top_agg.groupby("period").head(5)
    pivot = top5_per_period.pivot_table(index="aggregator_name", columns="period",
                                        values="n", fill_value=0)
    pivot.plot(kind="bar", ax=axes[1], colormap="Paired")
    axes[1].set_ylabel("posting count")
    axes[1].set_title("Top-5 aggregators per period")
    axes[1].tick_params(axis="x", rotation=30, labelsize=8)
    fig_caption(axes[0], f"n = {int(df['n'].sum()):,} SWE postings; filter: {DEFAULT_FILTER} AND is_swe")
    fig.tight_layout()
    save_fig(fig, "S6_aggregator_share")
    return df


# ---------------------------------------------------------------------------
# S7 — H4 — Industry dispersion (HHI + top-10 industries)
# ---------------------------------------------------------------------------

def scan_s7(con):
    # Industry only populated for arshkon + scraped-LinkedIn (per schema).
    sql = f"""
      SELECT period, company_industry, COUNT(*) AS n
      FROM '{UNIFIED_PATH}'
      WHERE {DEFAULT_FILTER} AND is_swe = true
        AND company_industry IS NOT NULL
      GROUP BY 1,2
      ORDER BY 1, n DESC
    """
    df = con.execute(sql).df()
    totals = df.groupby("period")["n"].sum().rename("total").reset_index()
    df = df.merge(totals, on="period")
    df["share"] = df["n"] / df["total"]

    # HHI per period
    hhi = df.assign(s2=lambda d: d["share"] ** 2).groupby("period")["s2"].sum().rename("hhi").reset_index()
    hhi.to_csv(TABLES_DIR / "S7_industry_hhi.csv", index=False)

    # Top 15 industries per period
    top = df.groupby("period").head(15).reset_index(drop=True)
    top.to_csv(TABLES_DIR / "S7_top_industries.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].bar(hhi["period"], hhi["hhi"], color="#17becf")
    for i, row in hhi.iterrows():
        axes[0].text(i, row["hhi"] + 0.003, f"{row['hhi']:.3f}", ha="center", fontsize=9)
    axes[0].set_ylabel("Herfindahl (sum of squared industry shares)")
    axes[0].set_title("S7 (H4) — Industry HHI of SWE postings by period")
    axes[0].tick_params(axis="x", rotation=25)

    # Horizontal bar: top-10 by max period share
    by_industry = top.pivot_table(index="company_industry", columns="period",
                                  values="share", fill_value=0)
    by_industry["max"] = by_industry.max(axis=1)
    by_industry = by_industry.sort_values("max", ascending=True).drop(columns="max").tail(12)
    by_industry.plot(kind="barh", ax=axes[1], colormap="tab10", width=0.8)
    axes[1].set_xlabel("share of SWE postings with labeled industry")
    axes[1].set_title("Top industries by max-period share")
    axes[1].legend(title="period", fontsize=8)
    fig_caption(axes[0], f"n = {int(df['n'].sum()):,} SWE rows with industry label; arshkon+scraped-LinkedIn only")
    fig.tight_layout()
    save_fig(fig, "S7_industry_dispersion")
    return df


# ---------------------------------------------------------------------------
# S8 — H4 — Non-tech industry SWE share
# ---------------------------------------------------------------------------

def scan_s8(con):
    tech_list = ", ".join(f"'{t.replace(chr(39), chr(39) + chr(39))}'" for t in TECH_INDUSTRIES)
    sql = f"""
      SELECT period,
             COUNT(*) AS n,
             SUM(CASE WHEN company_industry IN ({tech_list}) THEN 1 ELSE 0 END) AS n_tech,
             SUM(CASE WHEN company_industry IS NOT NULL
                       AND company_industry NOT IN ({tech_list}) THEN 1 ELSE 0 END) AS n_nontech,
             SUM(CASE WHEN company_industry IS NULL THEN 1 ELSE 0 END) AS n_null
      FROM '{UNIFIED_PATH}'
      WHERE {DEFAULT_FILTER} AND is_swe = true
      GROUP BY 1 ORDER BY 1
    """
    df = con.execute(sql).df()
    df["nontech_share_of_labeled"] = df["n_nontech"] / (df["n_tech"] + df["n_nontech"]).replace(0, np.nan)
    df.to_csv(TABLES_DIR / "S8_nontech_industry_share.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["period"], df["nontech_share_of_labeled"], color="#e377c2")
    for i, row in df.iterrows():
        val = row["nontech_share_of_labeled"]
        ax.text(i, (val or 0) + 0.005,
                f"{(val or 0)*100:.1f}%\n(n_labeled={int(row['n_tech']+row['n_nontech']):,})",
                ha="center", fontsize=8)
    ax.set_ylabel("non-tech industry share of labeled SWE postings")
    ax.set_title("S8 (H4) — Non-tech-industry share among SWE postings with labeled industry")
    ax.tick_params(axis="x", rotation=25)
    fig_caption(ax, f"Tech-industry exclusion set of {len(TECH_INDUSTRIES)} labels; arshkon+scraped-LinkedIn only (where industry is populated)")
    save_fig(fig, "S8_nontech_industry_share")
    return df


# ---------------------------------------------------------------------------
# S9 — H5 — Junior-vs-senior trajectory (3 panels)
# ---------------------------------------------------------------------------

def scan_s9(con):
    # Panel a: seniority mix by period among SWE
    a = con.execute(f"""
      SELECT period, seniority_3level, COUNT(*) AS n
      FROM '{UNIFIED_PATH}'
      WHERE {DEFAULT_FILTER} AND is_swe = true
      GROUP BY 1,2 ORDER BY 1,2
    """).df()
    a_totals = a.groupby("period")["n"].sum().rename("total").reset_index()
    a = a.merge(a_totals, on="period")
    a["share"] = a["n"] / a["total"]

    # Panel b: AI-vocab rate by period × seniority (reuse S1 shape)
    b = con.execute(f"""
      SELECT period, seniority_3level,
             COUNT(*) AS n,
             SUM(CASE WHEN regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}') THEN 1 ELSE 0 END) AS n_ai
      FROM '{UNIFIED_PATH}'
      WHERE {DEFAULT_FILTER} AND is_swe = true AND {text_filter()}
      GROUP BY 1,2 ORDER BY 1,2
    """).df()
    b["ai_rate"] = b["n_ai"] / b["n"]

    # Panel c: mean yoe_min_years_llm by period × seniority (labeled only)
    c = con.execute(f"""
      SELECT period, seniority_3level,
             COUNT(*) AS n_labeled,
             AVG(yoe_min_years_llm) AS mean_yoe,
             STDDEV(yoe_min_years_llm) AS sd_yoe
      FROM '{UNIFIED_PATH}'
      WHERE {DEFAULT_FILTER} AND is_swe = true
        AND llm_classification_coverage = 'labeled'
        AND yoe_min_years_llm IS NOT NULL
      GROUP BY 1,2 ORDER BY 1,2
    """).df()

    a.to_csv(TABLES_DIR / "S9_senior_mix.csv", index=False)
    b.to_csv(TABLES_DIR / "S9_ai_rate_by_seniority.csv", index=False)
    c.to_csv(TABLES_DIR / "S9_yoe_by_seniority.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    levels = ["junior", "mid", "senior", "unknown"]
    colors = {"junior": "#d62728", "mid": "#ff7f0e", "senior": "#2ca02c", "unknown": "#7f7f7f"}

    # (a) seniority mix
    periods = period_order(a)
    x = np.arange(len(periods))
    bottom = np.zeros(len(periods))
    for lvl in levels:
        sub = a[a["seniority_3level"] == lvl].set_index("period").reindex(periods)["share"].fillna(0).values
        axes[0].bar(x, sub, bottom=bottom, color=colors.get(lvl, "#888"), label=lvl)
        bottom += sub
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(periods, rotation=25)
    axes[0].set_ylabel("seniority share (SWE)")
    axes[0].set_title("(a) Seniority mix by period")
    axes[0].legend(title="seniority_3level", fontsize=8)

    # (b) AI rate by seniority
    for lvl in levels:
        sub = b[b["seniority_3level"] == lvl].set_index("period").reindex(periods)
        axes[1].plot(periods, sub["ai_rate"], "-o", color=colors.get(lvl, "#888"), label=lvl)
    axes[1].set_ylabel("AI-vocab rate")
    axes[1].set_title("(b) AI-vocab rate by seniority × period")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(title="seniority_3level", fontsize=8)

    # (c) YOE mean by seniority (labeled only)
    for lvl in levels:
        sub = c[c["seniority_3level"] == lvl].set_index("period").reindex(periods)
        axes[2].plot(periods, sub["mean_yoe"], "-o", color=colors.get(lvl, "#888"), label=lvl)
    axes[2].set_ylabel("mean yoe_min_years_llm (labeled)")
    axes[2].set_title("(c) Mean LLM-YOE by seniority × period")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].legend(title="seniority_3level", fontsize=8)

    fig.suptitle("S9 (H5) — Junior vs senior SWE trajectory")
    fig_caption(axes[0], f"n_SWE = {int(a['n'].sum()):,}; filter: {DEFAULT_FILTER} AND is_swe; (c) restricted to llm_classification_coverage='labeled'")
    fig.tight_layout()
    save_fig(fig, "S9_junior_senior_trajectory")
    return a, b, c


# ---------------------------------------------------------------------------
# S10 — H6 — Big Tech vs rest
# ---------------------------------------------------------------------------

def scan_s10(con):
    bt_list = ", ".join(f"'{b}'" for b in BIG_TECH_CANONICAL)
    sql = f"""
      SELECT period,
             CASE WHEN LOWER(company_name_canonical) IN ({bt_list}) THEN 'big_tech' ELSE 'rest' END AS tier,
             COUNT(*) AS n,
             SUM(CASE WHEN regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}') THEN 1 ELSE 0 END) AS n_ai
      FROM '{UNIFIED_PATH}'
      WHERE {DEFAULT_FILTER} AND is_swe = true AND {text_filter()}
      GROUP BY 1,2 ORDER BY 1,2
    """
    df = con.execute(sql).df()
    period_totals = df.groupby("period")["n"].sum().rename("period_total").reset_index()
    df = df.merge(period_totals, on="period")
    df["volume_share"] = df["n"] / df["period_total"]
    df["ai_rate"] = df["n_ai"] / df["n"]
    df.to_csv(TABLES_DIR / "S10_bigtech_vs_rest.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bt = df[df["tier"] == "big_tech"]
    rest = df[df["tier"] == "rest"]
    periods = period_order(df)
    x = np.arange(len(periods))
    width = 0.35

    bt_share = bt.set_index("period").reindex(periods)["volume_share"].fillna(0)
    axes[0].bar(x, bt_share, color="#1f77b4")
    for i, v in enumerate(bt_share):
        axes[0].text(i, v + 0.002, f"{v*100:.2f}%", ha="center", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(periods, rotation=25)
    axes[0].set_ylabel("Big Tech share of SWE postings")
    axes[0].set_title("(a) Big Tech volume share by period")

    bt_ai = bt.set_index("period").reindex(periods)["ai_rate"].fillna(0)
    rest_ai = rest.set_index("period").reindex(periods)["ai_rate"].fillna(0)
    axes[1].bar(x - width/2, bt_ai, width, label="Big Tech", color="#1f77b4")
    axes[1].bar(x + width/2, rest_ai, width, label="rest", color="#ff7f0e")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(periods, rotation=25)
    axes[1].set_ylabel("AI-vocab rate")
    axes[1].set_title("(b) AI-vocab rate — Big Tech vs rest")
    axes[1].legend()

    fig.suptitle(f"S10 (H6) — Big Tech vs rest (match on company_name_canonical; {len(BIG_TECH_CANONICAL)} canonical names)")
    fig_caption(axes[0], f"n_total = {int(df['n'].sum()):,}; filter: {DEFAULT_FILTER} AND is_swe")
    fig.tight_layout()
    save_fig(fig, "S10_bigtech_vs_rest")
    return df


# ---------------------------------------------------------------------------
# S11 — H7 — SWE vs control divergence
# ---------------------------------------------------------------------------

def scan_s11(con):
    sql = f"""
      SELECT period,
             CASE WHEN is_swe THEN 'swe'
                  WHEN is_control THEN 'control'
                  WHEN is_swe_adjacent THEN 'adjacent'
                  ELSE 'other' END AS group_label,
             COUNT(*) AS n,
             SUM(CASE WHEN is_remote THEN 1 ELSE 0 END) AS n_remote,
             SUM(CASE WHEN is_aggregator THEN 1 ELSE 0 END) AS n_agg,
             SUM(CASE WHEN regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}') THEN 1 ELSE 0 END) AS n_ai
      FROM '{UNIFIED_PATH}'
      WHERE {DEFAULT_FILTER} AND {text_filter()}
      GROUP BY 1,2 ORDER BY 1,2
    """
    df = con.execute(sql).df()
    df["remote_rate"] = df["n_remote"] / df["n"]
    df["agg_rate"] = df["n_agg"] / df["n"]
    df["ai_rate"] = df["n_ai"] / df["n"]
    df.to_csv(TABLES_DIR / "S11_swe_vs_control.csv", index=False)

    # Filter to swe and control (main interest), show 3 metrics side by side
    sub = df[df["group_label"].isin(["swe", "control"])].copy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [("remote_rate", "is_remote"), ("agg_rate", "is_aggregator"), ("ai_rate", "AI-vocab")]
    for ax, (col, label) in zip(axes, metrics):
        pivot = sub.pivot_table(index="period", columns="group_label", values=col, fill_value=0)
        pivot.plot(kind="bar", ax=ax, color=["#d62728", "#1f77b4"])
        ax.set_title(f"{label}")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(title="group")

    fig.suptitle("S11 (H7) — SWE vs control divergence — remote, aggregator, AI-vocab rates")
    n_swe = int(sub[sub['group_label']=='swe']['n'].sum())
    n_ctrl = int(sub[sub['group_label']=='control']['n'].sum())
    fig_caption(axes[0], f"n_swe = {n_swe:,}, n_control = {n_ctrl:,}; control exists only in scraped 2026 periods")
    fig.tight_layout()
    save_fig(fig, "S11_swe_vs_control")
    return df


# ---------------------------------------------------------------------------
# Sv — v8 replication (seniority mix + mean description_length)
# ---------------------------------------------------------------------------

def scan_sv(con):
    a = con.execute(f"""
      SELECT period, seniority_3level, COUNT(*) AS n
      FROM '{UNIFIED_PATH}'
      WHERE {DEFAULT_FILTER} AND is_swe = true
      GROUP BY 1,2 ORDER BY 1,2
    """).df()

    b = con.execute(f"""
      SELECT period,
             COUNT(*) AS n,
             AVG(description_length) AS mean_desc_len,
             AVG(LENGTH(description_core_llm)) FILTER (WHERE llm_extraction_coverage='labeled') AS mean_core_len,
             AVG(LENGTH(description)) AS mean_desc_len_runtime
      FROM '{UNIFIED_PATH}'
      WHERE {DEFAULT_FILTER} AND is_swe = true
      GROUP BY 1 ORDER BY 1
    """).df()

    a.to_csv(TABLES_DIR / "Sv_seniority_mix_swe.csv", index=False)
    b.to_csv(TABLES_DIR / "Sv_description_length.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    periods = period_order(a)
    levels = ["junior", "mid", "senior", "unknown"]
    colors = {"junior": "#d62728", "mid": "#ff7f0e", "senior": "#2ca02c", "unknown": "#7f7f7f"}

    totals = a.groupby("period")["n"].sum().rename("total").reset_index()
    a = a.merge(totals, on="period")
    a["share"] = a["n"] / a["total"]
    x = np.arange(len(periods))
    bottom = np.zeros(len(periods))
    for lvl in levels:
        sub = a[a["seniority_3level"] == lvl].set_index("period").reindex(periods)["share"].fillna(0).values
        axes[0].bar(x, sub, bottom=bottom, color=colors.get(lvl, "#888"), label=lvl)
        bottom += sub
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(periods, rotation=25)
    axes[0].set_title("Seniority mix by period (SWE) — v8 inverted-scope check")
    axes[0].legend(title="seniority_3level", fontsize=8)
    axes[0].set_ylabel("share")

    axes[1].plot(b["period"], b["mean_desc_len"], "-o", label="raw description", color="#1f77b4")
    axes[1].plot(b["period"], b["mean_core_len"], "-s", label="description_core_llm (labeled)", color="#ff7f0e")
    axes[1].set_ylabel("mean length (chars)")
    axes[1].set_title("Mean description length by period — v8 said −19% requirements shrink")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend()

    fig.suptitle("Sv — v8 replication on unified.parquet")
    fig_caption(axes[0], f"n_SWE = {int(a['n'].sum()):,}; filter: {DEFAULT_FILTER} AND is_swe")
    fig.tight_layout()
    save_fig(fig, "Sv_v8_replication")
    return a, b


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    scans = [
        ("S1  H1 AI-vocab prevalence", scan_s1),
        ("S2  H1 AI-requirement intensity (LLM)", scan_s2),
        ("S3  H2 New-AI-title share", scan_s3),
        ("S4  H2 Title n-gram delta", scan_s4),
        ("S5  H3 Remote share by metro", scan_s5),
        ("S6  H3 Aggregator share", scan_s6),
        ("S7  H4 Industry HHI + top industries", scan_s7),
        ("S8  H4 Non-tech industry share", scan_s8),
        ("S9  H5 Junior-vs-senior trajectory", scan_s9),
        ("S10 H6 Big Tech vs rest", scan_s10),
        ("S11 H7 SWE vs control", scan_s11),
        ("Sv  v8 replication", scan_sv),
    ]

    for name, fn in scans:
        print(f"Running {name} ...")
        try:
            _ = fn(con)
            print("  OK")
        except Exception as e:
            print(f"  FAILED: {e}")
            raise


if __name__ == "__main__":
    main()
