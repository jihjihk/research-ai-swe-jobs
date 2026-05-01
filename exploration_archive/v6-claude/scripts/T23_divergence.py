"""
T23 — Employer-requirement vs worker-usage divergence.

Steps:
1. AI requirement rate in SWE postings by period and seniority, using:
   - Narrow (T05-style): LIKE / regex on "ai" / "artificial intelligence"
   - Broad (T14 24-term union): aggregate OR across AI tech columns
   - AI-as-tool: copilot / cursor_tool / claude_tool / chatgpt / prompt_engineering
   - AI-as-domain: machine_learning / deep_learning / nlp / computer_vision /
     tensorflow / pytorch / scikit_learn
   - AI-generic: llm / gpt / langchain / rag / agents_framework / openai_api /
     claude_api / mcp / embedding / transformer / vector_db / huggingface / fine_tuning
2. External benchmarks: Stack Overflow 2024/2025, Anthropic labor market,
   GitHub Copilot adoption. Hard-coded from WebFetch.
3. Divergence computation: posting AI rate vs worker usage rate.
4. Temporal divergence.
5. Divergence by specificity (per-tool).
6. Benchmark sensitivity: assume worker usage at 50/65/75/85.
7. Ghost cross-check using T22's ai_ghostiness_section.csv.
8. Divergence chart + per-tool chart + temporal chart.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "exploration" / "artifacts" / "shared"
FIG_DIR = ROOT / "exploration" / "figures" / "T23"
TBL_DIR = ROOT / "exploration" / "tables" / "T23"
T22_TBL_DIR = ROOT / "exploration" / "tables" / "T22"
REPORT_PATH = ROOT / "exploration" / "reports" / "T23.md"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

CLEANED = str(SHARED / "swe_cleaned_text.parquet")
TECH = str(SHARED / "swe_tech_matrix.parquet")
UNIFIED = str(ROOT / "data" / "unified.parquet")

# Tech column groups
AI_TOOL_COLS = ["copilot", "cursor_tool", "claude_tool", "chatgpt", "prompt_engineering",
                "gemini_tool", "codex_tool"]
AI_DOMAIN_COLS = ["machine_learning", "deep_learning", "nlp", "computer_vision",
                  "tensorflow", "pytorch", "scikit_learn"]
AI_GENERIC_COLS = ["llm", "gpt", "agents_framework", "rag", "langchain", "langgraph",
                   "huggingface", "openai_api", "claude_api", "fine_tuning", "mcp",
                   "transformer_arch", "embedding", "vector_db"]
AI_ALL_COLS = AI_TOOL_COLS + AI_DOMAIN_COLS + AI_GENERIC_COLS

# Narrow AI pattern (T05-style)
NARROW_AI_RE = re.compile(r"\b(ai|a\.i\.|artificial intelligence)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# External benchmarks (hard-coded from WebFetch / WebSearch, dated 2026-04-15)
# ---------------------------------------------------------------------------
BENCHMARKS = {
    "stackoverflow_2024_pro_current_use": {
        "value": 0.632,
        "metric": "% of professional developers currently using AI tools",
        "period": "2024",
        "source": "Stack Overflow Developer Survey 2024",
        "url": "https://survey.stackoverflow.co/2024/ai",
        "notes": "63.2% currently using, 13.5% plan to soon.",
    },
    "stackoverflow_2025_pro_daily": {
        "value": 0.506,
        "metric": "% of professional developers using AI tools DAILY",
        "period": "2025",
        "source": "Stack Overflow Developer Survey 2025",
        "url": "https://survey.stackoverflow.co/2025/ai",
        "notes": "Daily 50.6%, weekly 17.4%, monthly 12.8% → combined any-use 80.8%.",
    },
    "stackoverflow_2025_pro_any_use": {
        "value": 0.808,
        "metric": "% of professional developers using AI tools daily+weekly+monthly",
        "period": "2025",
        "source": "Stack Overflow Developer Survey 2025",
        "url": "https://survey.stackoverflow.co/2025/ai",
        "notes": "Sum of daily 50.6% + weekly 17.4% + monthly 12.8% = 80.8%.",
    },
    "stackoverflow_2025_planning": {
        "value": 0.84,
        "metric": "% using OR planning to use AI tools (all respondents)",
        "period": "2025",
        "source": "Stack Overflow Developer Survey 2025 press release",
        "url": "https://stackoverflow.blog/2025/12/29/",
        "notes": "All respondents (including students), not pro-only.",
    },
    "anthropic_programmer_exposure": {
        "value": 0.75,
        "metric": "Observed AI task coverage for Computer Programmer occupation",
        "period": "~2024-2025",
        "source": "Anthropic Labor Market Impact research",
        "url": "https://www.anthropic.com/research/labor-market-impacts",
        "notes": "Share of Computer Programmer tasks with observed LLM usage per Anthropic's economic index.",
    },
    "copilot_accenture_daily": {
        "value": 0.67,
        "metric": "% of Accenture developers using Copilot at least 5 days/week",
        "period": "2024-2025",
        "source": "Accenture enterprise Copilot study (via GitHub blog)",
        "url": "https://github.blog/",
        "notes": "67% of deployed developers use Copilot at least 5 days/week.",
    },
    "copilot_surveys_2024_2025": {
        "value": 0.33,
        "metric": "Rough range of regular/occasional Copilot usage in external surveys",
        "period": "2024-2025",
        "source": "Multiple 2024-2025 developer surveys (range midpoint)",
        "url": "—",
        "notes": "Range 26-40%; midpoint 33%.",
    },
    "github_copilot_adopters_total": {
        "value": 20_000_000,
        "metric": "Total GitHub Copilot users (count, not share)",
        "period": "July 2025",
        "source": "GitHub Copilot official",
        "url": "https://github.blog/",
        "notes": "~20M total users, 4.7M paid subscribers Jan 2026.",
    },
    "stackoverflow_2024_copilot_among_users": {
        "value": 0.679,
        "metric": "Copilot usage share AMONG AI-using professional developers",
        "period": "2025",
        "source": "Stack Overflow Developer Survey 2025",
        "url": "https://survey.stackoverflow.co/2025/ai",
        "notes": "Of developers using AI agents, 67.9% use GitHub Copilot.",
    },
    "stackoverflow_2025_chatgpt_among_users": {
        "value": 0.817,
        "metric": "ChatGPT usage share AMONG AI-using professional developers",
        "period": "2025",
        "source": "Stack Overflow Developer Survey 2025",
        "url": "https://survey.stackoverflow.co/2025/ai",
        "notes": "Of AI-using devs, 81.7% use ChatGPT.",
    },
    "stackoverflow_2025_claude_among_users": {
        "value": 0.408,
        "metric": "Claude Code usage share AMONG AI-using professional developers",
        "period": "2025",
        "source": "Stack Overflow Developer Survey 2025",
        "url": "https://survey.stackoverflow.co/2025/ai",
        "notes": "Of AI-using devs, 40.8% use Claude Code. Pro-dev rate (45%) higher than learners (30%).",
    },
}


# ---------------------------------------------------------------------------
# Step 1 — AI requirement rate in SWE postings
# ---------------------------------------------------------------------------
def compute_posting_rates():
    print("[step1] loading data for posting-side AI rates ...", flush=True)
    con = duckdb.connect()
    # Join tech matrix + metadata
    con.execute(f"CREATE VIEW tech AS SELECT * FROM '{TECH}'")
    con.execute(f"CREATE VIEW meta AS SELECT * FROM '{CLEANED}'")
    con.execute(f"CREATE VIEW u AS SELECT uid, description_core_llm FROM '{UNIFIED}'")

    # Build a single frame with tech_matrix booleans + metadata
    tool_sql = " OR ".join(f"tech.{c}" for c in AI_TOOL_COLS)
    domain_sql = " OR ".join(f"tech.{c}" for c in AI_DOMAIN_COLS)
    generic_sql = " OR ".join(f"tech.{c}" for c in AI_GENERIC_COLS)
    all_sql = " OR ".join(f"tech.{c}" for c in AI_ALL_COLS)

    df = con.execute(f"""
        SELECT
            meta.uid, meta.source, meta.period, meta.seniority_final,
            meta.is_aggregator, meta.yoe_extracted, meta.text_source,
            ({tool_sql}) AS any_ai_tool,
            ({domain_sql}) AS any_ai_domain,
            ({generic_sql}) AS any_ai_generic,
            ({all_sql}) AS any_ai_broad,
            tech.copilot, tech.cursor_tool, tech.claude_tool, tech.chatgpt,
            tech.llm, tech.gpt, tech.langchain, tech.rag, tech.agents_framework,
            tech.machine_learning, tech.deep_learning, tech.nlp, tech.computer_vision,
            tech.pytorch, tech.tensorflow, tech.scikit_learn,
            tech.openai_api, tech.claude_api, tech.mcp, tech.embedding,
            tech.transformer_arch, tech.vector_db, tech.huggingface, tech.fine_tuning,
            tech.prompt_engineering, tech.gemini_tool, tech.codex_tool,
            u.description_core_llm
        FROM meta
        INNER JOIN tech ON tech.uid = meta.uid
        LEFT JOIN u ON u.uid = meta.uid
    """).fetch_df()
    con.close()
    print(f"[step1] loaded {len(df):,} rows", flush=True)

    # Narrow AI via regex on description_core_llm (T05-style).
    # NOTE: narrow_ai_hit is only meaningful on rows with description_core_llm
    # available; we also record `has_llm_text` so rate calculations can filter.
    def narrow_hit(text):
        if text is None:
            return None
        return bool(NARROW_AI_RE.search(text))

    df["narrow_ai_hit"] = df["description_core_llm"].apply(narrow_hit)
    df["has_llm_text"] = df["description_core_llm"].notna()

    df["period_year"] = df["period"].str.slice(0, 4)
    return df


def posting_rates_table(df: pd.DataFrame):
    # Aggregate by period. Narrow AI rate is only computed on the subset
    # with description_core_llm available (same denominator as T05).
    rates = []
    for period_year in ("2024", "2026"):
        sub = df[df["period_year"] == period_year]
        n = len(sub)
        sub_llm = sub[sub["has_llm_text"]]
        rates.append({
            "period": period_year,
            "n": n,
            "n_llm_text": len(sub_llm),
            "narrow_ai_rate": sub_llm["narrow_ai_hit"].mean(),
            "broad_ai_rate_full": sub["any_ai_broad"].mean(),
            "broad_ai_rate_llm_subset": sub_llm["any_ai_broad"].mean(),
            "broad_ai_rate": sub["any_ai_broad"].mean(),
            "ai_tool_rate": sub["any_ai_tool"].mean(),
            "ai_domain_rate": sub["any_ai_domain"].mean(),
            "ai_generic_rate": sub["any_ai_generic"].mean(),
            "copilot_rate": sub["copilot"].mean(),
            "cursor_rate": sub["cursor_tool"].mean(),
            "claude_rate": sub["claude_tool"].mean(),
            "chatgpt_rate": sub["chatgpt"].mean(),
            "llm_rate": sub["llm"].mean(),
            "ml_rate": sub["machine_learning"].mean(),
            "rag_rate": sub["rag"].mean(),
            "langchain_rate": sub["langchain"].mean(),
            "agents_rate": sub["agents_framework"].mean(),
        })
    df_out = pd.DataFrame(rates)
    df_out.to_csv(TBL_DIR / "posting_ai_rates_by_period.csv", index=False)
    return df_out


def posting_rates_by_seniority(df: pd.DataFrame):
    df2 = df[df["seniority_final"].isin(("entry", "mid-senior"))]
    rates = []
    for period_year in ("2024", "2026"):
        for sen in ("entry", "mid-senior"):
            sub = df2[(df2["period_year"] == period_year) & (df2["seniority_final"] == sen)]
            if len(sub) == 0:
                continue
            rates.append({
                "period": period_year,
                "seniority": sen,
                "n": len(sub),
                "narrow_ai_rate": sub["narrow_ai_hit"].mean(),
                "broad_ai_rate": sub["any_ai_broad"].mean(),
                "ai_tool_rate": sub["any_ai_tool"].mean(),
                "ai_domain_rate": sub["any_ai_domain"].mean(),
                "ai_generic_rate": sub["any_ai_generic"].mean(),
            })
    df_out = pd.DataFrame(rates)
    df_out.to_csv(TBL_DIR / "posting_ai_rates_by_period_seniority.csv", index=False)
    return df_out


def posting_rates_aggregator(df: pd.DataFrame):
    rates = []
    for period_year in ("2024", "2026"):
        for agg in (False, True):
            sub = df[(df["period_year"] == period_year) & (df["is_aggregator"] == agg)]
            rates.append({
                "period": period_year,
                "is_aggregator": agg,
                "n": len(sub),
                "narrow_ai_rate": sub["narrow_ai_hit"].mean(),
                "broad_ai_rate": sub["any_ai_broad"].mean(),
                "ai_tool_rate": sub["any_ai_tool"].mean(),
            })
    df_out = pd.DataFrame(rates)
    df_out.to_csv(TBL_DIR / "posting_ai_rates_by_aggregator.csv", index=False)
    return df_out


# ---------------------------------------------------------------------------
# Step 3 — Divergence computation
# ---------------------------------------------------------------------------
def divergence_table(rates: pd.DataFrame):
    """
    Produce a divergence table comparing posting rates against each benchmark.

    Primary comparison point: 2026 posting rate vs 2025 Stack Overflow "any use"
    (most recent benchmark at 80.8%).
    """
    r2024 = rates[rates["period"] == "2024"].iloc[0]
    r2026 = rates[rates["period"] == "2026"].iloc[0]

    rows = []
    for name, info in BENCHMARKS.items():
        if info["value"] < 0 or info["value"] > 2:
            # skip non-rate benchmarks (total counts etc)
            continue
        worker_rate = info["value"]
        for posting_name, posting_rate in (
            ("narrow_2024", r2024["narrow_ai_rate"]),
            ("narrow_2026", r2026["narrow_ai_rate"]),
            ("broad_2024", r2024["broad_ai_rate"]),
            ("broad_2026", r2026["broad_ai_rate"]),
            ("ai_tool_2024", r2024["ai_tool_rate"]),
            ("ai_tool_2026", r2026["ai_tool_rate"]),
        ):
            gap_pp = (posting_rate - worker_rate) * 100
            ratio = posting_rate / worker_rate if worker_rate > 0 else float("inf")
            rows.append({
                "benchmark": name,
                "worker_rate": worker_rate,
                "benchmark_period": info["period"],
                "posting_metric": posting_name,
                "posting_rate": posting_rate,
                "gap_pp": gap_pp,
                "ratio": ratio,
                "direction": "employer_above" if gap_pp > 0 else "worker_above",
            })
    out = pd.DataFrame(rows)
    out.to_csv(TBL_DIR / "divergence_full.csv", index=False)

    # Highlight headline: 2026 broad posting rate vs 2025 Stack Overflow pro any-use
    broad_2026 = r2026["broad_ai_rate"]
    narrow_2026 = r2026["narrow_ai_rate"]
    ai_tool_2026 = r2026["ai_tool_rate"]
    so_2025_any = BENCHMARKS["stackoverflow_2025_pro_any_use"]["value"]
    so_2024_pro = BENCHMARKS["stackoverflow_2024_pro_current_use"]["value"]
    headline = [
        {
            "metric": "narrow_ai_posting",
            "posting_rate_2026": narrow_2026,
            "worker_rate_so2025_any": so_2025_any,
            "gap_pp": (narrow_2026 - so_2025_any) * 100,
            "ratio": narrow_2026 / so_2025_any,
        },
        {
            "metric": "broad_ai_posting",
            "posting_rate_2026": broad_2026,
            "worker_rate_so2025_any": so_2025_any,
            "gap_pp": (broad_2026 - so_2025_any) * 100,
            "ratio": broad_2026 / so_2025_any,
        },
        {
            "metric": "ai_tool_posting",
            "posting_rate_2026": ai_tool_2026,
            "worker_rate_so2025_any": so_2025_any,
            "gap_pp": (ai_tool_2026 - so_2025_any) * 100,
            "ratio": ai_tool_2026 / so_2025_any,
        },
    ]
    pd.DataFrame(headline).to_csv(TBL_DIR / "divergence_headline.csv", index=False)
    return out, headline


# ---------------------------------------------------------------------------
# Step 4 — Temporal divergence: growth rates
# ---------------------------------------------------------------------------
def temporal_divergence(rates: pd.DataFrame):
    r2024 = rates[rates["period"] == "2024"].iloc[0]
    r2026 = rates[rates["period"] == "2026"].iloc[0]
    rows = [
        {"metric": "narrow_ai_posting", "r_2024": r2024["narrow_ai_rate"], "r_2026": r2026["narrow_ai_rate"]},
        {"metric": "broad_ai_posting", "r_2024": r2024["broad_ai_rate"], "r_2026": r2026["broad_ai_rate"]},
        {"metric": "ai_tool_posting", "r_2024": r2024["ai_tool_rate"], "r_2026": r2026["ai_tool_rate"]},
        {"metric": "ai_domain_posting", "r_2024": r2024["ai_domain_rate"], "r_2026": r2026["ai_domain_rate"]},
        {"metric": "ai_generic_posting", "r_2024": r2024["ai_generic_rate"], "r_2026": r2026["ai_generic_rate"]},
        {
            "metric": "worker_any_use (StackOverflow 2024 pro → 2025 pro any)",
            "r_2024": BENCHMARKS["stackoverflow_2024_pro_current_use"]["value"],
            "r_2026": BENCHMARKS["stackoverflow_2025_pro_any_use"]["value"],
        },
        {
            "metric": "worker_daily_use (StackOverflow 2025 daily only)",
            "r_2024": None,
            "r_2026": BENCHMARKS["stackoverflow_2025_pro_daily"]["value"],
        },
    ]
    out = pd.DataFrame(rows)
    out["delta_pp"] = (out["r_2026"] - out["r_2024"].astype(float)) * 100
    out["ratio_2026_over_2024"] = out["r_2026"] / out["r_2024"].astype(float)
    out.to_csv(TBL_DIR / "temporal_divergence.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Step 5 — Divergence by specificity (per-tool)
# ---------------------------------------------------------------------------
def per_tool_divergence(rates: pd.DataFrame):
    r2026 = rates[rates["period"] == "2026"].iloc[0]

    # Worker-side benchmarks for specific tools are via SO 2025 AMONG-USERS share
    # Multiply by the any-use rate 80.8% to get unconditional worker usage.
    any_use = BENCHMARKS["stackoverflow_2025_pro_any_use"]["value"]
    worker_copilot = BENCHMARKS["stackoverflow_2024_copilot_among_users"]["value"] * any_use
    worker_chatgpt = BENCHMARKS["stackoverflow_2025_chatgpt_among_users"]["value"] * any_use
    worker_claude = BENCHMARKS["stackoverflow_2025_claude_among_users"]["value"] * any_use

    tools = [
        ("copilot", r2026["copilot_rate"], worker_copilot,
         "GitHub Copilot (67.9% among AI-users × 80.8% any-use = 54.9%)"),
        ("chatgpt", r2026["chatgpt_rate"], worker_chatgpt,
         "ChatGPT (81.7% among AI-users × 80.8% any-use = 66.0%)"),
        ("claude_tool", r2026["claude_rate"], worker_claude,
         "Claude Code (40.8% among AI-users × 80.8% any-use = 33.0%)"),
        ("cursor_tool", r2026["cursor_rate"], None, "Cursor — no reliable benchmark"),
    ]
    rows = []
    for name, post, worker, note in tools:
        if worker is None:
            rows.append({"tool": name, "posting_rate_2026": post, "worker_rate": None,
                         "gap_pp": None, "ratio": None, "note": note})
            continue
        rows.append({
            "tool": name,
            "posting_rate_2026": post,
            "worker_rate": worker,
            "gap_pp": (post - worker) * 100,
            "ratio": post / worker,
            "note": note,
        })
    out = pd.DataFrame(rows)
    out.to_csv(TBL_DIR / "per_tool_divergence.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Step 6 — Benchmark sensitivity
# ---------------------------------------------------------------------------
def benchmark_sensitivity(rates: pd.DataFrame):
    r2026 = rates[rates["period"] == "2026"].iloc[0]
    rows = []
    for assumed in (0.50, 0.65, 0.75, 0.808, 0.85):
        for metric_name, posting_rate in (
            ("narrow_ai", r2026["narrow_ai_rate"]),
            ("broad_ai", r2026["broad_ai_rate"]),
            ("ai_tool", r2026["ai_tool_rate"]),
        ):
            rows.append({
                "metric": metric_name,
                "assumed_worker_rate": assumed,
                "posting_rate_2026": posting_rate,
                "gap_pp": (posting_rate - assumed) * 100,
                "ratio": posting_rate / assumed,
                "direction": "employer_above" if posting_rate > assumed else "worker_above",
            })
    out = pd.DataFrame(rows)
    out.to_csv(TBL_DIR / "benchmark_sensitivity.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Step 7 — Ghost cross-check (using T22 section-level AI ghostiness)
# ---------------------------------------------------------------------------
def ghost_cross_check(rates: pd.DataFrame):
    sec_csv = T22_TBL_DIR / "ai_ghostiness_section.csv"
    if not sec_csv.exists():
        print("[step7] T22 section file not found — skipping ghost cross-check")
        return None
    sec = pd.read_csv(sec_csv)
    # For 2026 row, compute AI requirements-section share
    row_2026 = sec[sec["period"] == 2026].iloc[0]
    row_2024 = sec[sec["period"] == 2024].iloc[0]

    r2026 = rates[rates["period"] == "2026"].iloc[0]
    r2024 = rates[rates["period"] == "2024"].iloc[0]

    # Adjusted posting "hard requirement" rate = description-level AI rate ×
    # share of AI mentions appearing in the requirements section
    broad_2026 = r2026["broad_ai_rate"]
    broad_2024 = r2024["broad_ai_rate"]
    ai_req_share_2026 = row_2026["ai_requirements_share"]
    ai_req_share_2024 = row_2024["ai_requirements_share"]

    rows = [
        {
            "metric": "broad_posting_all_sections",
            "r_2024": broad_2024,
            "r_2026": broad_2026,
        },
        {
            "metric": "broad_posting_requirements_only (section-adjusted)",
            "r_2024": broad_2024 * ai_req_share_2024,
            "r_2026": broad_2026 * ai_req_share_2026,
        },
        {
            "metric": "ai_req_share_in_requirements",
            "r_2024": ai_req_share_2024,
            "r_2026": ai_req_share_2026,
        },
        {
            "metric": "ai_req_share_in_preferred",
            "r_2024": row_2024["ai_preferred_share"],
            "r_2026": row_2026["ai_preferred_share"],
        },
        {
            "metric": "ai_req_share_in_responsibilities",
            "r_2024": row_2024["ai_responsibilities_share"],
            "r_2026": row_2026["ai_responsibilities_share"],
        },
    ]
    out = pd.DataFrame(rows)
    out["delta_pp"] = (out["r_2026"] - out["r_2024"]) * 100
    out.to_csv(TBL_DIR / "ghost_cross_check.csv", index=False)

    print("[step7] Ghost cross-check:")
    print(out.to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(rates, sensitivity, temporal, per_tool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (1) Divergence chart: posting rate vs worker benchmark, multiple assumptions
    try:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        pivot = sensitivity.pivot_table(
            index="assumed_worker_rate", columns="metric", values="gap_pp"
        )
        pivot.plot(kind="line", marker="o", ax=ax)
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_ylabel("Employer–worker gap (pp)")
        ax.set_xlabel("Assumed worker AI usage rate")
        ax.set_title("Divergence: SWE posting AI rate − worker usage rate")
        ax.legend(title="posting metric", loc="best")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "divergence_sensitivity.png")
        plt.close(fig)
    except Exception as e:
        print(f"[fig] divergence_sensitivity failed: {e}")

    # (2) Per-tool divergence bars
    try:
        pt = per_tool.dropna(subset=["gap_pp"])
        if len(pt) > 0:
            fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
            x = np.arange(len(pt))
            ax.bar(x, pt["posting_rate_2026"] * 100, width=0.35, label="Posting rate 2026", color="#fd8d3c")
            ax.bar(x + 0.4, pt["worker_rate"] * 100, width=0.35, label="Worker rate (est.)", color="#6baed6")
            ax.set_xticks(x + 0.2)
            ax.set_xticklabels(pt["tool"])
            ax.set_ylabel("Rate (%)")
            ax.set_title("Per-tool divergence: posting requirement rate vs worker usage")
            ax.legend()
            fig.tight_layout()
            fig.savefig(FIG_DIR / "per_tool_divergence.png")
            plt.close(fig)
    except Exception as e:
        print(f"[fig] per_tool failed: {e}")

    # (3) Temporal chart
    try:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
        labels = []
        r24 = []
        r26 = []
        for _, row in temporal.iterrows():
            if pd.isna(row["r_2024"]) or pd.isna(row["r_2026"]):
                continue
            labels.append(row["metric"].split("(")[0].strip())
            r24.append(row["r_2024"] * 100)
            r26.append(row["r_2026"] * 100)
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, r24, w, label="2024", color="#6baed6")
        ax.bar(x + w / 2, r26, w, label="2026", color="#fd8d3c")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Rate (%)")
        ax.set_title("Temporal divergence: posting AI rate vs worker AI usage")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "temporal_divergence.png")
        plt.close(fig)
    except Exception as e:
        print(f"[fig] temporal failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = compute_posting_rates()
    rates = posting_rates_table(df)
    rates_sen = posting_rates_by_seniority(df)
    rates_agg = posting_rates_aggregator(df)

    print("\nPOSTING RATES BY PERIOD:")
    print(rates.to_string(index=False))
    print("\nPOSTING RATES BY PERIOD × SENIORITY:")
    print(rates_sen.to_string(index=False))
    print("\nPOSTING RATES BY AGGREGATOR:")
    print(rates_agg.to_string(index=False))

    full_div, headline = divergence_table(rates)
    print("\nDIVERGENCE HEADLINE:")
    for row in headline:
        print(row)

    temporal = temporal_divergence(rates)
    print("\nTEMPORAL DIVERGENCE:")
    print(temporal.to_string(index=False))

    per_tool = per_tool_divergence(rates)
    print("\nPER-TOOL DIVERGENCE:")
    print(per_tool.to_string(index=False))

    sensitivity = benchmark_sensitivity(rates)
    print("\nBENCHMARK SENSITIVITY:")
    print(sensitivity.to_string(index=False))

    ghost = ghost_cross_check(rates)

    make_figures(rates, sensitivity, temporal, per_tool)

    # Save benchmarks as JSON
    with open(TBL_DIR / "benchmarks.json", "w") as f:
        json.dump(BENCHMARKS, f, indent=2)
    print("[done] T23 script complete")


if __name__ == "__main__":
    main()
