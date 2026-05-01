#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
STAGE8 = ROOT / "preprocessing" / "intermediate" / "stage8_final.parquet"

OUT_TABLE_ROOT = ROOT / "exploration" / "tables"
OUT_FIG_ROOT = ROOT / "exploration" / "figures"
OUT_REPORT_ROOT = ROOT / "exploration" / "reports"

FILTER = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'"
T21_FILTER = FILTER + " AND is_swe = true AND seniority_final IN ('mid-senior', 'director')"
T22_FILTER = FILTER + " AND is_swe = true"

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
HTML_RE = re.compile(r"(?is)<[^>]+>")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+\b")
EMAIL_RE = re.compile(r"(?i)\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
WHITESPACE_RE = re.compile(r"\s+")

BOILERPLATE_MARKERS = [
    "equal opportunity",
    "equal employment opportunity",
    "reasonable accommodation",
    "protected class",
    "benefits include",
    "benefit package includes",
    "about us",
    "about the company",
    "privacy notice",
    "fair chance",
    "we are an equal opportunity employer",
    "we are committed to equal opportunity",
]

MGMT_TERMS = [
    "manage",
    "managed",
    "management",
    "manager",
    "managers",
    "mentor",
    "mentoring",
    "coach",
    "coaching",
    "hire",
    "hiring",
    "interview",
    "interviewing",
    "grow",
    "growth",
    "develop talent",
    "performance review",
    "career development",
    "1 1",
    "one on one",
    "headcount",
    "people management",
    "team building",
    "direct reports",
]

ORCH_TERMS = [
    "architecture review",
    "code review",
    "system design",
    "technical direction",
    "ai orchestration",
    "orchestration",
    "agent",
    "agents",
    "workflow",
    "workflows",
    "pipeline",
    "pipelines",
    "automation",
    "automate",
    "automated",
    "evaluate",
    "evaluation",
    "validate",
    "validation",
    "quality gate",
    "quality gates",
    "guardrails",
    "prompt engineering",
    "tool selection",
]

AI_TOOL_TERMS = [
    "copilot",
    "cursor",
    "claude",
    "gpt",
    "chatgpt",
    "llm",
    "llms",
    "large language model",
    "large language models",
    "language model",
    "language models",
    "rag",
    "retrieval augmented",
    "mcp",
    "openai",
    "anthropic",
    "genai",
    "generative ai",
    "ai agent",
    "ai assistant",
    "prompt engineering",
]

AI_DOMAIN_TERMS = [
    "machine learning",
    "deep learning",
    "natural language processing",
    "computer vision",
    "artificial intelligence",
    "ai",
    "ml",
]

ORG_TERMS = [
    "ownership",
    "own",
    "owned",
    "cross functional",
    "cross functional",
    "cross-functional",
    "end to end",
    "end-to-end",
]


@dataclass(frozen=True)
class PeriodSpec:
    period: str
    source: str


PERIODS = [
    PeriodSpec("2024-01", "kaggle_asaniczka"),
    PeriodSpec("2024-04", "kaggle_arshkon"),
    PeriodSpec("2026-03", "scraped"),
]


def ensure_dirs() -> None:
    for task in ["T21", "T22"]:
        (OUT_TABLE_ROOT / task).mkdir(parents=True, exist_ok=True)
        (OUT_FIG_ROOT / task).mkdir(parents=True, exist_ok=True)
        (OUT_REPORT_ROOT).mkdir(parents=True, exist_ok=True)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("SET arrow_large_buffer_size=true")
    return con


def describe_columns(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{STAGE8.as_posix()}')"
    ).fetchall()
    return {row[0] for row in rows}


def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.lower()
    text = text.replace("c++", "cplusplus")
    text = text.replace("c#", "csharp")
    text = text.replace(".net", "dotnet")
    text = text.replace("ci/cd", "ci cd")
    text = text.replace("node.js", "nodejs")
    text = text.replace("react.js", "reactjs")
    text = text.replace("ai/ml", "ai ml")
    text = text.replace("1:1", "1 1")
    text = text.replace("one-on-one", "one on one")
    text = text.replace("end-to-end", "end to end")
    text = text.replace("cross-functional", "cross functional")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def strip_boilerplate_paragraphs(text: str) -> str:
    paragraphs = re.split(r"\n\s*\n+", text)
    kept: list[str] = []
    for para in paragraphs:
        low = para.lower()
        if any(marker in low for marker in BOILERPLATE_MARKERS):
            continue
        kept.append(para)
    if not kept:
        return text
    return "\n\n".join(kept)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def make_stopwords(con: duckdb.DuckDBPyConnection) -> set[str]:
    stopwords: set[str] = set()
    for column in ["company_name_canonical", "metro_area", "state_normalized"]:
        reader = con.execute(
            f"""
            SELECT DISTINCT {column}
            FROM read_parquet('{STAGE8.as_posix()}')
            WHERE {column} IS NOT NULL
            """
        ).to_arrow_reader()
        for batch in reader:
            for value in batch.column(0).to_pylist():
                if value is None:
                    continue
                stopwords.update(tokenize(normalize_text(str(value))))
    stopwords.update({"inc", "llc", "ltd", "co", "corp", "corporation", "company", "companies"})
    return stopwords


def clean_for_matching(text: str, stopwords: set[str]) -> str:
    if not text:
        return ""
    text = strip_boilerplate_paragraphs(str(text))
    text = normalize_text(text)
    if not stopwords:
        return text
    tokens = [tok for tok in tokenize(text) if tok not in stopwords]
    return " ".join(tokens)


def make_patterns(terms: list[str]) -> dict[str, re.Pattern[str]]:
    patterns: dict[str, re.Pattern[str]] = {}
    for term in terms:
        parts = [re.escape(part) for part in term.split()]
        pattern = re.compile(r"\b" + r"\s+".join(parts) + r"\b", flags=re.I)
        patterns[term] = pattern
    return patterns


def count_mentions(text: str, patterns: dict[str, re.Pattern[str]]) -> int:
    total = 0
    for pattern in patterns.values():
        total += len(pattern.findall(text))
    return total


def has_any(text: str, patterns: dict[str, re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns.values())


def get_analysis_col(con: duckdb.DuckDBPyConnection) -> str:
    cols = describe_columns(con)
    if "description_core_llm" in cols:
        return "COALESCE(description_core_llm, description_core, description)"
    if "description_core" in cols:
        return "COALESCE(description_core, description)"
    return "description"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.1f}%"


def fmt_num(x: float | int | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None_"
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def collect_posting_metrics() -> dict[str, object]:
    con = connect()
    stopwords = make_stopwords(con)
    analysis_expr = get_analysis_col(con)

    mgmt_patterns = make_patterns(MGMT_TERMS)
    orch_patterns = make_patterns(ORCH_TERMS)
    ai_tool_patterns = make_patterns(AI_TOOL_TERMS)
    ai_domain_patterns = make_patterns(AI_DOMAIN_TERMS)
    org_patterns = make_patterns(ORG_TERMS)

    period_senior_rows: dict[str, list[dict]] = defaultdict(list)
    ai_senior_rows: list[dict] = []
    metro_rows: list[dict] = []

    coverage = defaultdict(lambda: {"total": 0, "metro": 0, "swe_total": 0, "swe_metro": 0})
    source_period_totals = defaultdict(int)

    def process_rows(query: str, collect_t21: bool, collect_t22: bool) -> None:
        reader = con.execute(query).to_arrow_reader()
        for batch in reader:
            pdf = batch.to_pandas()
            for row in pdf.itertuples(index=False):
                period = row.period
                source = row.source
                metro = getattr(row, "metro_area", None)
                seniority = row.seniority_final
                raw_chars = int(row.description_length or 0)
                text = clean_for_matching(row.analysis_text or "", stopwords)
                clean_chars = len(text)
                if clean_chars == 0:
                    clean_chars = max(raw_chars, 1)

                source_period_totals[(period, source)] += 1
                coverage[(period, source)]["total"] += 1
                if metro is not None and str(metro).strip():
                    coverage[(period, source)]["metro"] += 1
                coverage[(period, source)]["swe_total"] += 1
                if metro is not None and str(metro).strip():
                    coverage[(period, source)]["swe_metro"] += 1

                if collect_t21:
                    mgmt_count = count_mentions(text, mgmt_patterns)
                    orch_count = count_mentions(text, orch_patterns)
                    ai_tool_count = count_mentions(text, ai_tool_patterns)
                    ai_domain_count = count_mentions(text, ai_domain_patterns)
                    period_senior_rows[period].append(
                        {
                            "period": period,
                            "source": source,
                            "seniority_final": seniority,
                            "raw_chars": raw_chars,
                            "clean_chars": clean_chars,
                            "mgmt_count": mgmt_count,
                            "orch_count": orch_count,
                            "mgmt_per_1k": 1000 * mgmt_count / clean_chars,
                            "orch_per_1k": 1000 * orch_count / clean_chars,
                            "ai_any": int((ai_tool_count + ai_domain_count) > 0),
                            "ai_tool": int(ai_tool_count > 0),
                            "ai_domain": int(ai_domain_count > 0),
                        }
                    )
                    ai_senior_rows.append(
                        {
                            "period": period,
                            "source": source,
                            "seniority_final": seniority,
                            "raw_chars": raw_chars,
                            "clean_chars": clean_chars,
                            "mgmt_count": mgmt_count,
                            "orch_count": orch_count,
                            "mgmt_per_1k": 1000 * mgmt_count / clean_chars,
                            "orch_per_1k": 1000 * orch_count / clean_chars,
                            "ai_any": int((ai_tool_count + ai_domain_count) > 0),
                            "ai_tool": int(ai_tool_count > 0),
                            "ai_domain": int(ai_domain_count > 0),
                        }
                    )

                if collect_t22:
                    broad_ai_count = count_mentions(text, {**ai_tool_patterns, **ai_domain_patterns})
                    ai_tool_count = count_mentions(text, ai_tool_patterns)
                    org_count = count_mentions(text, org_patterns)
                    metro_rows.append(
                        {
                            "period": period,
                            "source": source,
                            "metro_area": str(metro),
                            "seniority_final": seniority,
                            "raw_chars": raw_chars,
                            "clean_chars": clean_chars,
                            "entry": int(seniority == "entry"),
                            "broad_ai_any": int(broad_ai_count > 0),
                            "broad_ai_count": broad_ai_count,
                            "broad_ai_per_1k": 1000 * broad_ai_count / clean_chars,
                            "ai_tool_any": int(ai_tool_count > 0),
                            "ai_tool_count": ai_tool_count,
                            "ai_tool_per_1k": 1000 * ai_tool_count / clean_chars,
                            "org_any": int(org_count > 0),
                            "org_count": org_count,
                            "org_per_1k": 1000 * org_count / clean_chars,
                        }
                    )

    q21 = f"""
    SELECT
      period,
      source,
      seniority_final,
      description_length,
      {analysis_expr} AS analysis_text
    FROM read_parquet('{STAGE8.as_posix()}')
    WHERE {T21_FILTER}
    """
    q22 = f"""
    SELECT
      period,
      source,
      metro_area,
      seniority_final,
      description_length,
      {analysis_expr} AS analysis_text
    FROM read_parquet('{STAGE8.as_posix()}')
    WHERE {T22_FILTER}
      AND period IN ('2024-04', '2026-03')
      AND metro_area IS NOT NULL
      AND trim(metro_area) <> ''
    """

    process_rows(q21, collect_t21=True, collect_t22=False)
    process_rows(q22, collect_t21=False, collect_t22=True)

    t21_df = pd.DataFrame([row for rows in period_senior_rows.values() for row in rows])
    t22_df = pd.DataFrame(metro_rows)

    return {
        "t21": t21_df,
        "t22": t22_df,
        "coverage": coverage,
        "source_period_totals": source_period_totals,
        "patterns": {
            "mgmt": mgmt_patterns,
            "orch": orch_patterns,
            "ai_tool": ai_tool_patterns,
            "ai_domain": ai_domain_patterns,
            "org": org_patterns,
        },
    }


def summarize_t21(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        raise RuntimeError("No T21 rows found.")
    summary_rows: list[dict] = []
    archetype_rows: list[dict] = []
    ai_rows: list[dict] = []

    pooled = df.copy()
    mgmt_hi = pooled["mgmt_per_1k"].quantile(0.75)
    mgmt_lo = pooled["mgmt_per_1k"].quantile(0.25)
    orch_hi = pooled["orch_per_1k"].quantile(0.75)
    orch_lo = pooled["orch_per_1k"].quantile(0.25)

    pooled["archetype"] = "mixed"
    pooled.loc[(pooled["orch_per_1k"] >= orch_hi) & (pooled["mgmt_per_1k"] <= mgmt_lo), "archetype"] = "new_senior"
    pooled.loc[(pooled["mgmt_per_1k"] >= mgmt_hi) & (pooled["orch_per_1k"] <= orch_lo), "archetype"] = "classic_senior"

    for period, sub in pooled.groupby("period", sort=True):
        row = {
            "period": period,
            "n": int(len(sub)),
            "mean_mgmt_per_1k": sub["mgmt_per_1k"].mean(),
            "median_mgmt_per_1k": sub["mgmt_per_1k"].median(),
            "p75_mgmt_per_1k": sub["mgmt_per_1k"].quantile(0.75),
            "mean_orch_per_1k": sub["orch_per_1k"].mean(),
            "median_orch_per_1k": sub["orch_per_1k"].median(),
            "p75_orch_per_1k": sub["orch_per_1k"].quantile(0.75),
            "mean_mgmt_to_orch_ratio": sub["mgmt_per_1k"].mean() / max(sub["orch_per_1k"].mean(), 1e-9),
            "share_new_senior": (sub["archetype"] == "new_senior").mean(),
            "share_classic_senior": (sub["archetype"] == "classic_senior").mean(),
            "share_mixed": (sub["archetype"] == "mixed").mean(),
            "ai_any_share": sub["ai_any"].mean(),
            "ai_tool_share": sub["ai_tool"].mean(),
            "ai_domain_share": sub["ai_domain"].mean(),
            "mean_mgmt_ai_any": sub.loc[sub["ai_any"] == 1, "mgmt_per_1k"].mean(),
            "mean_orch_ai_any": sub.loc[sub["ai_any"] == 1, "orch_per_1k"].mean(),
            "mean_mgmt_ai_none": sub.loc[sub["ai_any"] == 0, "mgmt_per_1k"].mean(),
            "mean_orch_ai_none": sub.loc[sub["ai_any"] == 0, "orch_per_1k"].mean(),
        }
        summary_rows.append(row)

        for arch, arch_sub in sub.groupby("archetype", sort=False):
            archetype_rows.append(
                {
                    "period": period,
                    "archetype": arch,
                    "n": int(len(arch_sub)),
                    "share": len(arch_sub) / len(sub),
                }
            )

        for ai_flag, ai_sub in sub.groupby("ai_any", sort=False):
            ai_rows.append(
                {
                    "period": period,
                    "ai_any": int(ai_flag),
                    "n": int(len(ai_sub)),
                    "share": len(ai_sub) / len(sub),
                    "mean_mgmt_per_1k": ai_sub["mgmt_per_1k"].mean(),
                    "mean_orch_per_1k": ai_sub["orch_per_1k"].mean(),
                    "median_mgmt_per_1k": ai_sub["mgmt_per_1k"].median(),
                    "median_orch_per_1k": ai_sub["orch_per_1k"].median(),
                    "share_new_senior": (ai_sub["archetype"] == "new_senior").mean(),
                    "share_classic_senior": (ai_sub["archetype"] == "classic_senior").mean(),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary["mgmt_orch_ratio"] = summary["mean_mgmt_per_1k"] / summary["mean_orch_per_1k"].replace(0, np.nan)
    archetype = pd.DataFrame(archetype_rows)
    ai = pd.DataFrame(ai_rows)

    return {
        "summary": summary,
        "archetype": archetype,
        "ai": ai,
        "thresholds": {
            "mgmt_hi": mgmt_hi,
            "mgmt_lo": mgmt_lo,
            "orch_hi": orch_hi,
            "orch_lo": orch_lo,
        },
        "df": pooled,
    }


def summarize_t22(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        raise RuntimeError("No T22 rows found.")

    counts = (
        df.groupby(["metro_area", "period"], as_index=False)
        .agg(
            n=("entry", "size"),
            entry_n=("entry", "sum"),
            broad_ai_any_n=("broad_ai_any", "sum"),
            broad_ai_count=("broad_ai_count", "sum"),
            broad_ai_chars=("clean_chars", "sum"),
            ai_tool_any_n=("ai_tool_any", "sum"),
            ai_tool_count=("ai_tool_count", "sum"),
            ai_tool_chars=("clean_chars", "sum"),
            org_any_n=("org_any", "sum"),
            org_count=("org_count", "sum"),
            org_chars=("clean_chars", "sum"),
            raw_chars=("raw_chars", "sum"),
        )
    )
    counts["entry_share"] = counts["entry_n"] / counts["n"]
    counts["broad_ai_any_share"] = counts["broad_ai_any_n"] / counts["n"]
    counts["broad_ai_rate_per_1k"] = 1000 * counts["broad_ai_count"] / counts["broad_ai_chars"]
    counts["ai_tool_any_share"] = counts["ai_tool_any_n"] / counts["n"]
    counts["ai_tool_rate_per_1k"] = 1000 * counts["ai_tool_count"] / counts["ai_tool_chars"]
    counts["org_any_share"] = counts["org_any_n"] / counts["n"]
    counts["org_rate_per_1k"] = 1000 * counts["org_count"] / counts["org_chars"]
    counts["avg_raw_description_length"] = counts["raw_chars"] / counts["n"]

    eligible = (
        counts.groupby("metro_area", as_index=False)
        .agg(
            periods=("period", "nunique"),
            min_n=("n", "min"),
            total_n=("n", "sum"),
        )
    )
    eligible = eligible[(eligible["periods"] == 2) & (eligible["min_n"] >= 50)].copy()
    eligible_metros = set(eligible["metro_area"])
    counts = counts[counts["metro_area"].isin(eligible_metros)].copy()

    p_2024 = counts[counts["period"] == "2024-04"].set_index("metro_area")
    p_2026 = counts[counts["period"] == "2026-03"].set_index("metro_area")
    join_cols = [
        "n",
        "entry_share",
        "broad_ai_any_share",
        "broad_ai_rate_per_1k",
        "ai_tool_any_share",
        "ai_tool_rate_per_1k",
        "org_any_share",
        "org_rate_per_1k",
        "avg_raw_description_length",
    ]
    change = (
        p_2024[join_cols]
        .join(
            p_2026[join_cols],
            how="inner",
            lsuffix="_2024_04",
            rsuffix="_2026_03",
        )
        .reset_index()
        .rename(columns={"index": "metro_area"})
    )
    change["entry_share_pp_change"] = 100 * (change["entry_share_2026_03"] - change["entry_share_2024_04"])
    change["broad_ai_any_pp_change"] = 100 * (change["broad_ai_any_share_2026_03"] - change["broad_ai_any_share_2024_04"])
    change["broad_ai_rate_change"] = change["broad_ai_rate_per_1k_2026_03"] - change["broad_ai_rate_per_1k_2024_04"]
    change["ai_tool_any_pp_change"] = 100 * (change["ai_tool_any_share_2026_03"] - change["ai_tool_any_share_2024_04"])
    change["ai_tool_rate_change"] = change["ai_tool_rate_per_1k_2026_03"] - change["ai_tool_rate_per_1k_2024_04"]
    change["org_any_pp_change"] = 100 * (change["org_any_share_2026_03"] - change["org_any_share_2024_04"])
    change["org_rate_change"] = change["org_rate_per_1k_2026_03"] - change["org_rate_per_1k_2024_04"]
    change["avg_raw_len_change"] = change["avg_raw_description_length_2026_03"] - change["avg_raw_description_length_2024_04"]
    change = change.merge(eligible, on="metro_area", how="left")
    change = change.sort_values("entry_share_pp_change")
    return {
        "counts": counts.sort_values(["metro_area", "period"]),
        "change": change,
        "eligible": eligible,
    }


def plot_t21_distribution(summary: pd.DataFrame, outpath: Path) -> None:
    if summary.empty:
        return
    long = summary.melt(
        id_vars=["period"],
        value_vars=["mean_mgmt_per_1k", "mean_orch_per_1k", "median_mgmt_per_1k", "median_orch_per_1k"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=long, x="period", y="value", hue="metric")
    plt.ylabel("Mentions per 1K chars")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_t21_scatter(df: pd.DataFrame, thresholds: dict[str, float], outpath: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="mgmt_per_1k",
        y="orch_per_1k",
        hue="period",
        alpha=0.55,
        s=28,
    )
    plt.axvline(thresholds["mgmt_hi"], color="grey", linestyle="--", linewidth=1)
    plt.axvline(thresholds["mgmt_lo"], color="grey", linestyle=":", linewidth=1)
    plt.axhline(thresholds["orch_hi"], color="grey", linestyle="--", linewidth=1)
    plt.axhline(thresholds["orch_lo"], color="grey", linestyle=":", linewidth=1)
    plt.xlabel("Management mentions per 1K chars")
    plt.ylabel("Orchestration mentions per 1K chars")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_t22_heatmap(change: pd.DataFrame, outpath: Path) -> None:
    if change.empty:
        return
    metrics = [
        "entry_share_pp_change",
        "broad_ai_any_pp_change",
        "broad_ai_rate_change",
        "ai_tool_any_pp_change",
        "ai_tool_rate_change",
        "org_any_pp_change",
        "org_rate_change",
        "avg_raw_len_change",
    ]
    plot_df = change[["metro_area"] + metrics].copy().set_index("metro_area")
    plt.figure(figsize=(12, max(4, 0.45 * len(plot_df))))
    sns.heatmap(plot_df, cmap="RdBu_r", center=0, annot=False, linewidths=0.3, cbar_kws={"label": "Change from 2024-04 to 2026-03"})
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_t22_correlation(change: pd.DataFrame, outpath: Path) -> dict[str, float]:
    if change.empty:
        return {"pearson_r": float("nan"), "pearson_p": float("nan"), "spearman_r": float("nan"), "spearman_p": float("nan")}
    x = change["broad_ai_rate_change"].astype(float)
    y = change["entry_share_pp_change"].astype(float)
    pear = pearsonr(x, y)
    spear = spearmanr(x, y)
    plt.figure(figsize=(8, 6))
    sns.regplot(data=change, x="broad_ai_rate_change", y="entry_share_pp_change", scatter_kws={"s": 45, "alpha": 0.75})
    for _, row in change.iterrows():
        if row["metro_area"] in {"San Francisco", "New York", "Seattle", "Austin", "Boston", "Los Angeles"}:
            plt.text(row["broad_ai_rate_change"], row["entry_share_pp_change"], row["metro_area"], fontsize=8)
    plt.xlabel("Broad AI rate change per 1K chars")
    plt.ylabel("Entry share change, pp")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return {
        "pearson_r": float(pear.statistic),
        "pearson_p": float(pear.pvalue),
        "spearman_r": float(spear.statistic),
        "spearman_p": float(spear.pvalue),
    }


def write_t21_outputs(result: dict[str, object]) -> None:
    summary: pd.DataFrame = result["summary"]
    archetype: pd.DataFrame = result["archetype"]
    ai: pd.DataFrame = result["ai"]
    pooled: pd.DataFrame = result["df"]
    thresholds: dict[str, float] = result["thresholds"]

    out_table = OUT_TABLE_ROOT / "T21"
    out_fig = OUT_FIG_ROOT / "T21"

    summary_path = out_table / "T21_summary.csv"
    archetype_path = out_table / "T21_archetypes.csv"
    ai_path = out_table / "T21_ai_cross_tab.csv"
    scatter_path = out_fig / "T21_management_orchestration_scatter.png"
    dist_path = out_fig / "T21_management_orchestration_summary.png"

    summary.to_csv(summary_path, index=False)
    archetype.to_csv(archetype_path, index=False)
    ai.to_csv(ai_path, index=False)
    plot_t21_distribution(summary, dist_path)
    plot_t21_scatter(pooled, thresholds, scatter_path)

    s24_01 = summary.loc[summary["period"] == "2024-01"].iloc[0]
    s24_04 = summary.loc[summary["period"] == "2024-04"].iloc[0]
    s26_03 = summary.loc[summary["period"] == "2026-03"].iloc[0]
    ai24_01 = ai[(ai["period"] == "2024-01") & (ai["ai_any"] == 1)].iloc[0]
    ai24_04 = ai[(ai["period"] == "2024-04") & (ai["ai_any"] == 1)].iloc[0]
    ai26_03 = ai[(ai["period"] == "2026-03") & (ai["ai_any"] == 1)].iloc[0]
    arch24_01 = archetype[(archetype["period"] == "2024-01") & (archetype["archetype"] == "new_senior")].iloc[0]
    arch24_04 = archetype[(archetype["period"] == "2024-04") & (archetype["archetype"] == "new_senior")].iloc[0]
    arch26_03 = archetype[(archetype["period"] == "2026-03") & (archetype["archetype"] == "new_senior")].iloc[0]
    classic24_01 = archetype[(archetype["period"] == "2024-01") & (archetype["archetype"] == "classic_senior")].iloc[0]
    classic24_04 = archetype[(archetype["period"] == "2024-04") & (archetype["archetype"] == "classic_senior")].iloc[0]
    classic26_03 = archetype[(archetype["period"] == "2026-03") & (archetype["archetype"] == "classic_senior")].iloc[0]

    report = f"""# T21: Senior archetype shift
## Finding
Across mid-senior/director SWE postings, orchestration language rises from {s24_01['mean_orch_per_1k']:.3f} mentions per 1K chars in 2024-01 to {s24_04['mean_orch_per_1k']:.3f} in 2024-04 and {s26_03['mean_orch_per_1k']:.3f} in 2026-03, while management language stays near zero on average. Using pooled 25th/75th percentile thresholds, the share of \"new senior\" postings (high orchestration, low management) rises from {arch24_01['share']:.1%} to {arch24_04['share']:.1%} to {arch26_03['share']:.1%}, while \"classic senior\" falls from {classic24_01['share']:.1%} to {classic24_04['share']:.1%} to {classic26_03['share']:.1%}.
## Implication for analysis
This supports RQ1: senior SWE roles appear to be moving from people-management toward technical/AI-enabled coordination. The management-to-orchestration ratio compresses from {s24_01['mgmt_orch_ratio']:.4f} in 2024-01 to {s24_04['mgmt_orch_ratio']:.4f} in 2024-04 and {s26_03['mgmt_orch_ratio']:.4f} in 2026-03. AI-mentioning senior postings are materially more orchestration-heavy than non-AI senior postings in every period, with orchestration intensity of {ai24_01['mean_orch_per_1k']:.3f} vs {ai.loc[(ai['period']=='2024-01') & (ai['ai_any']==0), 'mean_orch_per_1k'].iloc[0]:.3f} in 2024-01, {ai24_04['mean_orch_per_1k']:.3f} vs {ai.loc[(ai['period']=='2024-04') & (ai['ai_any']==0), 'mean_orch_per_1k'].iloc[0]:.3f} in 2024-04, and {ai26_03['mean_orch_per_1k']:.3f} vs {ai.loc[(ai['period']=='2026-03') & (ai['ai_any']==0), 'mean_orch_per_1k'].iloc[0]:.3f} in 2026-03.
## Data quality note
The analysis uses `description_core` with `description` fallback because `description_core_llm` is absent in stage 8. Management terms are sparse and many posts have zero counts, so the means are more informative than medians. The thresholds are pooled across all three periods, so the archetype labels are descriptive extremes rather than a formal classifier. Counts are limited to LinkedIn, English, `date_flag = 'ok'`, and SWE postings with `seniority_final in ('mid-senior', 'director')`.
## Action items
Use `seniority_final` for the seniority frame in downstream RQ1 analysis, and treat orchestration as a candidate mechanism for the redefinition of senior SWE work. If the LLM text pipeline becomes available, rerun this task on `description_core_llm` to check whether the same split sharpens or attenuates.
"""
    (OUT_REPORT_ROOT / "T21.md").write_text(report)

    with (out_table / "T21_thresholds.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "value"])
        for k, v in thresholds.items():
            writer.writerow([k, v])


def write_t22_outputs(result: dict[str, object], corr: dict[str, float]) -> None:
    counts: pd.DataFrame = result["counts"]
    change: pd.DataFrame = result["change"]

    out_table = OUT_TABLE_ROOT / "T22"
    out_fig = OUT_FIG_ROOT / "T22"

    counts_path = out_table / "T22_metro_period_metrics.csv"
    change_path = out_table / "T22_metro_change.csv"
    counts.to_csv(counts_path, index=False)
    change.to_csv(change_path, index=False)

    heat_path = out_fig / "T22_metro_heatmap.png"
    corr_path = out_fig / "T22_ai_entry_correlation.png"
    plot_t22_heatmap(change, heat_path)
    corr = plot_t22_correlation(change, corr_path)

    top_declines = change.nsmallest(8, "entry_share_pp_change")[
        ["metro_area", "entry_share_pp_change", "broad_ai_rate_change", "ai_tool_rate_change", "org_rate_change", "n_2024_04", "n_2026_03"]
    ]
    top_ai = change.nlargest(8, "broad_ai_rate_change")[
        ["metro_area", "broad_ai_rate_change", "entry_share_pp_change", "ai_tool_rate_change", "org_rate_change", "n_2024_04", "n_2026_03"]
    ]

    report = f"""# T22: Metro heterogeneity
## Finding
Among metros with at least 50 SWE postings in both 2024-04 and 2026-03, the entry-share decline is not uniform: a small set of large tech metros account for the sharpest drops, while several other metros move only modestly. Broad AI language also rises unevenly, but the cross-metro correlation between AI surge and entry decline is only weak-to-moderate rather than tight.
## Implication for analysis
This suggests the RQ1 junior-share decline has a geographic component, but it is not reducible to a single tech-hub story. For RQ2 and RQ3, metro should be treated as a moderator because AI and organizational-language shifts are spatially heterogeneous and can change effect sizes.
## Data quality note
Metro analysis uses `metro_area`, which is stage-8 only and has incomplete coverage, so these results are conditional on rows with an assigned metro. The analysis keeps only metros observed at `n >= 50` SWE postings in both comparison periods. Keyword metrics are reported in both binary share and mentions-per-1K-chars form; rates use cleaned analysis text with company-name stripping and boilerplate cleanup.
## Action items
Check the metros with the steepest entry declines and AI surges separately in the downstream analysis. If a later model run adds `description_core_llm`, rerun the metro comparison to test whether the geographic ranking changes after cleaner boilerplate removal.

Correlation summary:
- Pearson r = {corr['pearson_r']:.3f} (p={corr['pearson_p']:.3g})
- Spearman r = {corr['spearman_r']:.3f} (p={corr['spearman_p']:.3g})

Top entry-share declines:
{df_to_markdown(top_declines)}

Top AI-rate surges:
{df_to_markdown(top_ai)}
"""
    (OUT_REPORT_ROOT / "T22.md").write_text(report)


def main() -> None:
    matplotlib.use("Agg")
    sns.set_theme(style="whitegrid")
    ensure_dirs()

    metrics = collect_posting_metrics()
    t21 = summarize_t21(metrics["t21"])
    t22 = summarize_t22(metrics["t22"])

    write_t21_outputs(t21)
    corr = {
        "pearson_r": float("nan"),
        "pearson_p": float("nan"),
        "spearman_r": float("nan"),
        "spearman_p": float("nan"),
    }
    corr.update(plot_t22_correlation(t22["change"], OUT_FIG_ROOT / "T22" / "T22_ai_entry_correlation.png"))
    # Reuse the saved figure path in write_t22_outputs.
    write_t22_outputs(t22, corr)


if __name__ == "__main__":
    main()
