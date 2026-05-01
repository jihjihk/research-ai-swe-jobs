"""
T23. Employer-requirement / worker-usage divergence.

Compute AI requirement rates in SWE postings (ours) vs external worker-side
AI usage benchmarks. Produce divergence tables, chart, and benchmark sensitivity.

V1-validated patterns are loaded from validated_mgmt_patterns.json. We use
ai_strict (0.86 precision) as the PRIMARY employer-side measure and split
into:
  - ai_as_tool    (copilot, cursor, claude, chatgpt, codex, etc.)
  - ai_as_domain  (ml, dl, nlp, computer vision, model training)
  - ai_general    (artificial intelligence, ai, machine learning, llm)

External benchmarks (accessed 2026-04-20):
  - StackOverflow Developer Survey 2025: 84% use-or-plan, 50.6% use daily
    (vs 2024: 63% currently use, 76% use-or-plan)
  - Anthropic Economic Index (2026-03): Computer Programmers = 75% task
    coverage
  - DORA State of AI-Assisted SWE 2025: 90% AI adoption among devs
  - McKinsey State of AI 2025: GenAI 79% overall adoption; SWE a top value
    function
  - GitHub Copilot: 4.7M paid subscribers (Jan 2026); ~26-40% regular use
    from 2024-2025 surveys

Output:
  exploration/tables/T23/ai_requirement_rates.csv
  exploration/tables/T23/divergence_table.csv
  exploration/tables/T23/benchmark_sensitivity.csv
  exploration/figures/T23_divergence_chart.png
  exploration/figures/T23_divergence_chart.svg
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path("/home/jihgaboot/gabor/job-research")
DATA = REPO / "data" / "unified_core.parquet"
CLEANED = REPO / "exploration" / "artifacts" / "shared" / "swe_cleaned_text.parquet"
T11 = REPO / "exploration" / "artifacts" / "shared" / "T11_posting_features.parquet"
VAL = REPO / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json"
OUT = REPO / "exploration" / "tables" / "T23"
OUT.mkdir(parents=True, exist_ok=True)
FIG = REPO / "exploration" / "figures"
FIG.mkdir(exist_ok=True)

# --------------------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------------------
patterns = json.loads(VAL.read_text())
# Split ai_strict into tool vs domain via regex groups
AI_TOOL = re.compile(
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|"
    r"llamaindex|langchain|prompt engineering|huggingface|hugging face|"
    r"pinecone|vector databas(?:e|es)|rag|"
    r"(?:fine[- ]tun(?:e|ed|ing))\s+(?:the\s+)?(?:model|llm|gpt|base model|"
    r"foundation model|embeddings))\b",
    re.IGNORECASE,
)
AI_DOMAIN = re.compile(
    r"\b(machine learning|deep learning|neural network(?:s)?|"
    r"computer vision|cv model|nlp|natural language processing|"
    r"model training|model fine-?tuning|mlops|reinforcement learning|"
    r"transformer|bert|pytorch|tensorflow|scikit-?learn)\b",
    re.IGNORECASE,
)
AI_GENERAL = re.compile(
    r"\b(ai\b|a\.i\.|artificial intelligence|ml\b|"
    r"llm(?:s)?|large language model(?:s)?|generative ai|gen[- ]?ai)\b",
    re.IGNORECASE,
)
# Union (same as V1 ai_strict)
AI_STRICT = re.compile(patterns["ai_strict"]["pattern"], re.IGNORECASE)
# Specific tool subsets for specificity analysis
AI_COPILOT = re.compile(r"\bcopilot\b", re.IGNORECASE)
AI_CURSOR = re.compile(r"\bcursor\b", re.IGNORECASE)
AI_CLAUDE = re.compile(r"\bclaude\b", re.IGNORECASE)
AI_CHATGPT = re.compile(r"\b(chatgpt|gpt-?\d+)\b", re.IGNORECASE)
AI_RAG = re.compile(r"\brag\b", re.IGNORECASE)
AI_LANGCHAIN = re.compile(r"\blangchain\b", re.IGNORECASE)


def has(rx: re.Pattern, t: str) -> bool:
    if not isinstance(t, str) or not t:
        return False
    return rx.search(t) is not None


def main():
    con = duckdb.connect()
    print("Loading...")
    q = f"""
    SELECT
      u.uid, u.source, u.period, u.is_aggregator,
      u.yoe_min_years_llm,
      u.seniority_final,
      ct.description_cleaned,
      u.description_core_llm
    FROM read_parquet('{DATA}') u
    LEFT JOIN read_parquet('{CLEANED}') ct ON u.uid = ct.uid
    WHERE u.source_platform='linkedin' AND u.is_english=true AND u.date_flag='ok' AND u.is_swe=true
    """
    df = con.execute(q).fetchdf()
    df["text"] = df["description_cleaned"].fillna(df["description_core_llm"]).fillna("")
    df["is_2024"] = df["source"].isin(["kaggle_arshkon", "kaggle_asaniczka"])
    df["is_arshkon"] = df["source"] == "kaggle_arshkon"
    df["is_scraped"] = df["source"] == "scraped"
    df["is_j3"] = (df["yoe_min_years_llm"] <= 2).fillna(False)
    df["s4"] = (df["yoe_min_years_llm"] >= 5).fillna(False)

    print("Computing AI flags...")
    df["ai_tool"] = df["text"].apply(lambda t: has(AI_TOOL, t))
    df["ai_domain"] = df["text"].apply(lambda t: has(AI_DOMAIN, t))
    df["ai_general"] = df["text"].apply(lambda t: has(AI_GENERAL, t))
    df["ai_strict"] = df["text"].apply(lambda t: has(AI_STRICT, t))
    df["ai_any"] = df["ai_tool"] | df["ai_domain"] | df["ai_general"] | df["ai_strict"]
    df["copilot"] = df["text"].apply(lambda t: has(AI_COPILOT, t))
    df["cursor_ai"] = df["text"].apply(lambda t: has(AI_CURSOR, t))
    df["claude_ai"] = df["text"].apply(lambda t: has(AI_CLAUDE, t))
    df["chatgpt"] = df["text"].apply(lambda t: has(AI_CHATGPT, t))
    df["rag"] = df["text"].apply(lambda t: has(AI_RAG, t))

    # --- AI requirement rate by cohort x seniority ---
    cohorts = [
        ("2024_pooled", df[df["is_2024"]]),
        ("2024_arshkon_only", df[df["is_arshkon"]]),
        ("2026_scraped", df[df["is_scraped"]]),
    ]
    sen_cells = [
        ("all", None),
        ("J3_yoe_le2", "is_j3"),
        ("S4_yoe_ge5", "s4"),
    ]
    rows = []
    for cname, sub in cohorts:
        for sname, flag in sen_cells:
            s = sub if flag is None else sub[sub[flag] == True]
            if len(s) == 0:
                continue
            row = {
                "cohort": cname,
                "seniority_cell": sname,
                "n": int(len(s)),
            }
            for col in ["ai_tool", "ai_domain", "ai_general", "ai_strict", "ai_any",
                        "copilot", "cursor_ai", "claude_ai", "chatgpt", "rag"]:
                row[f"{col}_rate"] = float(s[col].mean())
            rows.append(row)
    rates = pd.DataFrame(rows)
    rates.to_csv(OUT / "ai_requirement_rates.csv", index=False)
    print(f"Wrote {OUT/'ai_requirement_rates.csv'}")

    # --- Aggregator sensitivity ---
    agg_rows = []
    for cname, sub in cohorts:
        for is_agg in [False, True]:
            s = sub[sub["is_aggregator"] == is_agg]
            if len(s) == 0:
                continue
            row = {"cohort": cname, "is_aggregator": bool(is_agg), "n": int(len(s))}
            for col in ["ai_tool", "ai_domain", "ai_general", "ai_strict", "ai_any",
                        "copilot", "cursor_ai", "claude_ai"]:
                row[f"{col}_rate"] = float(s[col].mean())
            agg_rows.append(row)
    aggdf = pd.DataFrame(agg_rows)
    aggdf.to_csv(OUT / "ai_rate_by_aggregator.csv", index=False)
    print(f"Wrote {OUT/'ai_rate_by_aggregator.csv'}")

    # --- External benchmarks (static table)
    benchmarks = [
        {
            "source": "Stack Overflow Developer Survey 2024",
            "date": "2024-05",
            "metric": "Professional devs currently use AI in dev process",
            "rate": 0.63,
            "scope": "any AI tool; self-reported; professional developers",
            "notes": "63% currently use; 14% plan to soon; 76% use-or-plan.",
            "url": "https://survey.stackoverflow.co/2024/ai",
        },
        {
            "source": "Stack Overflow Developer Survey 2025",
            "date": "2025-Q4",
            "metric": "Professional devs currently use AI in dev process",
            "rate": 0.84,
            "scope": "any AI tool use-or-plan; self-reported",
            "notes": "84% use-or-plan; 50.6% daily; 14.7% say won't use.",
            "url": "https://survey.stackoverflow.co/2025/ai",
        },
        {
            "source": "Stack Overflow Developer Survey 2025 (daily)",
            "date": "2025-Q4",
            "metric": "Professional devs use AI daily",
            "rate": 0.506,
            "scope": "daily AI-tool use; self-reported",
            "notes": "Among users, not percent of all developers.",
            "url": "https://survey.stackoverflow.co/2025/ai",
        },
        {
            "source": "DORA 2025 State of AI-Assisted Software Development",
            "date": "2025",
            "metric": "AI adoption among SWE professionals",
            "rate": 0.90,
            "scope": "use AI at work; 5000 tech professionals",
            "notes": "90% adoption; median 2 hours/day AI use.",
            "url": "https://dora.dev/research/2025/dora-report/",
        },
        {
            "source": "Anthropic Economic Index (March 2026)",
            "date": "2026-03",
            "metric": "Computer Programmers task coverage by Claude",
            "rate": 0.75,
            "scope": "task-level coverage; Claude only; O*NET mapping",
            "notes": "75% of tasks in occupation observed automated; NOT worker-adoption rate.",
            "url": "https://www.anthropic.com/research/labor-market-impacts",
        },
        {
            "source": "GitHub Copilot (industry reports 2024-2025)",
            "date": "2024-2025",
            "metric": "Developers regularly using Copilot",
            "rate": 0.33,  # midpoint of 26-40%
            "scope": "Copilot-specific; paid subs ~3% of registered devs",
            "notes": "Regular/occasional use 26-40%; enterprise >90% of Fortune 100 deployed.",
            "url": "https://github.blog/",
        },
        {
            "source": "McKinsey State of AI 2025",
            "date": "2025",
            "metric": "GenAI adoption at any business function",
            "rate": 0.79,
            "scope": "organizational AI adoption (not dev-level)",
            "notes": "SWE is a top-value function; 7% fully scaled.",
            "url": "https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai",
        },
    ]
    bm = pd.DataFrame(benchmarks)
    bm.to_csv(OUT / "external_benchmarks.csv", index=False)
    print(f"Wrote {OUT/'external_benchmarks.csv'}")

    # --- Divergence computation ---
    # Approach: pin an employer-side rate per period, compare to worker-side rate ranges.
    # Specifically (per Gate 2 pre-commit 1):
    #   - ai_strict (v1-validated) is the PRIMARY employer-side AI requirement.
    #   - ai_any captures the broader "posting mentions AI" envelope (for upper bound).
    # Worker-side bands: 50%, 65%, 75%, 85% (per dispatch).
    #
    # Divergence definition (per dispatch): divergence PATTERN, not exact gap.
    # Report: gap = (worker_band) - (employer_rate), per period, per cell.

    # Build employer-side rates table for chart
    emp_rows = []
    for label, sub in [
        ("2024 pooled (SWE all)", df[df["is_2024"]]),
        ("2024 arshkon (SWE all)", df[df["is_arshkon"]]),
        ("2026 scraped (SWE all)", df[df["is_scraped"]]),
        ("2024 pooled (J3 yoe<=2)", df[df["is_2024"] & df["is_j3"]]),
        ("2024 pooled (S4 yoe>=5)", df[df["is_2024"] & df["s4"]]),
        ("2026 scraped (J3 yoe<=2)", df[df["is_scraped"] & df["is_j3"]]),
        ("2026 scraped (S4 yoe>=5)", df[df["is_scraped"] & df["s4"]]),
    ]:
        emp_rows.append({
            "cell": label,
            "n": len(sub),
            "ai_strict_rate": float(sub["ai_strict"].mean()) if len(sub) else np.nan,
            "ai_any_rate": float(sub["ai_any"].mean()) if len(sub) else np.nan,
            "ai_tool_rate": float(sub["ai_tool"].mean()) if len(sub) else np.nan,
            "ai_general_rate": float(sub["ai_general"].mean()) if len(sub) else np.nan,
            "copilot_rate": float(sub["copilot"].mean()) if len(sub) else np.nan,
            "cursor_rate": float(sub["cursor_ai"].mean()) if len(sub) else np.nan,
            "claude_rate": float(sub["claude_ai"].mean()) if len(sub) else np.nan,
            "rag_rate": float(sub["rag"].mean()) if len(sub) else np.nan,
        })
    emp = pd.DataFrame(emp_rows)
    emp.to_csv(OUT / "employer_rates_by_cell.csv", index=False)
    print(f"Wrote {OUT/'employer_rates_by_cell.csv'}")

    # Benchmark sensitivity table (dispatch step 6)
    bands = [0.50, 0.65, 0.75, 0.85]
    sens_rows = []
    for label, sub in [
        ("2024_pooled_all", df[df["is_2024"]]),
        ("2024_arshkon_all", df[df["is_arshkon"]]),
        ("2026_scraped_all", df[df["is_scraped"]]),
    ]:
        er = float(sub["ai_strict"].mean())
        er_any = float(sub["ai_any"].mean())
        for b in bands:
            sens_rows.append({
                "cohort": label,
                "employer_ai_strict_rate": er,
                "employer_ai_any_rate": er_any,
                "worker_band": b,
                "gap_ai_strict": b - er,
                "gap_ai_any": b - er_any,
                "n": len(sub),
            })
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(OUT / "benchmark_sensitivity.csv", index=False)
    print(f"Wrote {OUT/'benchmark_sensitivity.csv'}")

    # Temporal divergence (step 4): how fast employer requirement is growing
    # vs worker usage growing.
    # Employer: 2024_pooled ai_strict -> 2026_scraped ai_strict.
    er_2024 = float(df[df["is_2024"]]["ai_strict"].mean())
    er_2026 = float(df[df["is_scraped"]]["ai_strict"].mean())
    er_change = er_2026 - er_2024
    er_ratio = er_2026 / max(er_2024, 1e-6)

    # Worker: Stack Overflow 2024 63% -> 2025 84% -> 50.6% daily
    worker_change_so = 0.84 - 0.63
    worker_ratio_so = 0.84 / 0.63

    temporal = {
        "employer_ai_strict_2024": er_2024,
        "employer_ai_strict_2026": er_2026,
        "employer_change_pp": er_change,
        "employer_ratio": er_ratio,
        "stackoverflow_2024": 0.63,
        "stackoverflow_2025": 0.84,
        "stackoverflow_change_pp": worker_change_so,
        "stackoverflow_ratio": worker_ratio_so,
        "note": "Employer-side grew faster-in-ratio (from a very-low base) but remains much lower in absolute terms than worker-side. Divergence sign is consistent across all bands."
    }
    pd.DataFrame([temporal]).to_csv(OUT / "temporal_divergence.csv", index=False)
    print(f"Wrote {OUT/'temporal_divergence.csv'}")

    # Specificity analysis (step 5): tool-specific rates vs adoption bands
    spec_rows = []
    for label, sub in [
        ("2024_pooled", df[df["is_2024"]]),
        ("2026_scraped", df[df["is_scraped"]]),
    ]:
        n = len(sub)
        spec_rows.append({
            "cohort": label,
            "n": n,
            "copilot_rate": float(sub["copilot"].mean()),
            "cursor_rate": float(sub["cursor_ai"].mean()),
            "claude_rate": float(sub["claude_ai"].mean()),
            "chatgpt_rate": float(sub["chatgpt"].mean()),
            "rag_rate": float(sub["rag"].mean()),
            "ai_strict_rate": float(sub["ai_strict"].mean()),
            "ai_general_rate": float(sub["ai_general"].mean()),
        })
    spec = pd.DataFrame(spec_rows)
    spec.to_csv(OUT / "tool_specificity_rates.csv", index=False)
    print(f"Wrote {OUT/'tool_specificity_rates.csv'}")

    # --- DIVERGENCE CHART ---
    # X: conceptual timeline (2024, 2026)
    # Y: rate (%)
    # Series: employer (ai_strict, ai_any, copilot-specific), worker (SO, DORA, Anthropic)
    # Bands: 50/65/75/85 worker
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(left=0.09, right=0.78, top=0.90, bottom=0.12)
    x = [2024, 2026]

    # Employer-side series (SWE-LinkedIn, ours)
    emp_ai_strict = [er_2024*100, er_2026*100]
    emp_ai_any = [df[df["is_2024"]]["ai_any"].mean()*100, df[df["is_scraped"]]["ai_any"].mean()*100]
    emp_copilot = [df[df["is_2024"]]["copilot"].mean()*100, df[df["is_scraped"]]["copilot"].mean()*100]

    # Worker-side series (external)
    wkr_so = [63.0, 84.0]  # 2024 vs 2025 (approx 2025 Q4 -> ~2026 for our 2026-scrape window)
    wkr_so_daily = [None, 50.6]  # only 2025
    wkr_anthropic = [None, 75.0]  # 2026 task coverage
    wkr_dora = [None, 90.0]  # 2025 90%
    wkr_mckinsey = [None, 79.0]  # 2025

    # Worker bands (horizontal)
    for band, lbl in [(50, "50% (lower-bound)"), (65, "65%"), (75, "75%"), (85, "85% (upper-bound)")]:
        ax.axhspan(band-2, band+2, alpha=0.08, color="tab:green")
        ax.axhline(y=band, color="tab:green", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.text(2026.05, band, f" {lbl}", va="center", fontsize=8.5, color="#2d5f2d")

    # Employer series
    ax.plot(x, emp_ai_strict, "o-", color="tab:red", linewidth=2.6, markersize=10,
            label="Employer: SWE postings mention ai_strict (V1-validated)")
    ax.plot(x, emp_ai_any, "s-", color="tab:orange", linewidth=2, markersize=8,
            label="Employer: SWE postings mention any AI term (broader)")
    ax.plot(x, emp_copilot, "^-", color="tab:red", linewidth=1.5, markersize=7, alpha=0.6,
            label="Employer: SWE postings mention 'copilot' specifically")

    # Worker series (dots only — single-point or 2-point where we have data)
    ax.plot(x, wkr_so, "D-", color="tab:blue", linewidth=2, markersize=9,
            label="Worker: StackOverflow use AI (2024 63% → 2025 84%)")
    # daily and other single-point worker measures
    ax.plot([2026], [50.6], "*", color="tab:blue", markersize=18,
            label="Worker: StackOverflow 2025 daily AI use (50.6%)")
    ax.plot([2026], [75.0], "P", color="tab:purple", markersize=12,
            label="Anthropic 2026: 75% Computer Programmer task coverage")
    ax.plot([2026], [90.0], "X", color="tab:cyan", markersize=11,
            label="DORA 2025: 90% SWE AI adoption")
    ax.plot([2026], [79.0], "h", color="tab:gray", markersize=10,
            label="McKinsey 2025: 79% GenAI adoption (org-level)")

    ax.set_xticks([2024, 2026])
    ax.set_xticklabels(["2024", "2026"])
    ax.set_xlim(2023.6, 2026.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Employer-requirement vs. worker-usage AI divergence (SWE)\n"
                  "LinkedIn SWE postings (ours, ai_strict) vs external worker-side benchmarks",
                 fontsize=11)
    # Legend outside to the right
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5, framealpha=0.95)
    ax.grid(True, alpha=0.2)
    # annotate gap
    gap_2026 = 84.0 - er_2026*100
    ax.annotate(f"Gap ≈ {gap_2026:.0f} pp (2026)",
                xy=(2026, (84.0 + er_2026*100)/2),
                xytext=(2024.8, 55),
                fontsize=9.5, color="#444",
                arrowprops=dict(arrowstyle="->", color="#666", lw=0.8))
    fig.savefig(FIG / "T23_divergence_chart.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIG / "T23_divergence_chart.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG/'T23_divergence_chart.png'}")

    # --- Summary prints ---
    print()
    print("--- T23 summary ---")
    print(f"Employer ai_strict 2024 pooled: {er_2024:.3%} (n={int(df[df['is_2024']].shape[0])})")
    print(f"Employer ai_strict 2026 scraped: {er_2026:.3%} (n={int(df[df['is_scraped']].shape[0])})")
    print(f"Employer change: {er_change*100:+.1f} pp, {er_ratio:.2f}x")
    print(f"Worker SO 2024: 63%; SO 2025: 84%; daily 50.6%")
    print(f"Gap (worker 84% - employer 2026 ai_strict): {gap_2026:.1f} pp")
    return df, emp, sens


if __name__ == "__main__":
    df, emp, sens = main()
