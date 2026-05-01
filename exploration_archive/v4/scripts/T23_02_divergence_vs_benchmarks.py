"""T23 step 2 — Employer-requirement vs worker-usage divergence.

External benchmarks (retrieved 2026-04-10 via WebFetch; see report for URLs):

  StackOverflow Developer Survey
    - 2024: 62-63% of professional developers currently use AI tools
            (44% → 62% year-over-year among all respondents)
    - 2025: 80.8% of professional developers actively use AI tools
            (daily 50.6% + weekly 17.4% + monthly 12.8%)
            - AI agents (daily): 14.9% of professional developers
            - AI agents (weekly): 9.2%

  Anthropic "Labor Market Impacts" (2024)
    - Computer programmer observed exposure: 75%
    - Computer & Math broad: 33% observed, 94% theoretical

  GitHub Blog Survey (late 2023/early 2024)
    - 92% of US-based developers use AI coding tools "in and outside of work"
    - (NB: self-selected and any-use definition; likely over-estimate vs
       strict "use in production work")

Sensitivity scenarios for "true" developer AI usage:
  - conservative_2024 = 50%, conservative_2026 = 65%
  - central_2024     = 62%, central_2026     = 80%
  - high_2024        = 75%, high_2026        = 90%

Agentic/AI-agent usage benchmark (daily or weekly):
  - 2024 -> ~0%  (not meaningfully adopted)
  - 2026 -> ~25% (combined StackOverflow daily 14.9 + weekly 9.2 ≈ 24%)

For each benchmark we compute divergence = (requirement_rate) - (usage_rate).

Outputs:
  tables/T23/divergence_scenarios.csv
  tables/T23/divergence_agentic.csv
  tables/T23/divergence_temporal_growth.csv
  figures/T23/divergence_chart.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
FEATS = ROOT / "exploration" / "artifacts" / "T22" / "ghost_indicators_per_posting.parquet"
TAB = ROOT / "exploration" / "tables" / "T23"
FIG = ROOT / "exploration" / "figures" / "T23"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_parquet(FEATS)

    # Exclude aggregators for the primary analysis
    direct = df[df["is_aggregator"] == False].copy()

    # ANY AI rate
    ai_cols = ["ai_tool", "ai_domain", "ai_general", "agentic", "ai_agent_phrase", "rag_phrase"]
    for d in (df, direct):
        d["any_ai"] = (d[ai_cols].sum(axis=1) > 0).astype(int)

    req_rates = {
        "all_postings": df.groupby("period2").agg(
            any_ai_rate=("any_ai", "mean"),
            ai_tool_rate=("ai_tool", "mean"),
            ai_domain_rate=("ai_domain", "mean"),
            agentic_rate=("agentic", "mean"),
            ai_agent_phrase_rate=("ai_agent_phrase", "mean"),
            rag_phrase_rate=("rag_phrase", "mean"),
            n=("uid", "size"),
        ).reset_index(),
        "direct_only": direct.groupby("period2").agg(
            any_ai_rate=("any_ai", "mean"),
            ai_tool_rate=("ai_tool", "mean"),
            ai_domain_rate=("ai_domain", "mean"),
            agentic_rate=("agentic", "mean"),
            ai_agent_phrase_rate=("ai_agent_phrase", "mean"),
            rag_phrase_rate=("rag_phrase", "mean"),
            n=("uid", "size"),
        ).reset_index(),
    }

    benchmarks = {
        "conservative":    {"2024": 0.50, "2026": 0.65},
        "central":         {"2024": 0.62, "2026": 0.80},
        "high":            {"2024": 0.75, "2026": 0.90},
        "github_blog":     {"2024": 0.92, "2026": 0.92},  # likely over
    }

    rows = []
    for scope, rr in req_rates.items():
        for bench, b in benchmarks.items():
            for _, r in rr.iterrows():
                p = r["period2"]
                usage = b[p]
                req = r["any_ai_rate"]
                rows.append({
                    "scope": scope,
                    "benchmark": bench,
                    "period": p,
                    "usage_rate": usage,
                    "req_any_ai_rate": req,
                    "divergence_req_minus_usage": req - usage,
                })
    div_df = pd.DataFrame(rows)
    div_df.to_csv(TAB / "divergence_scenarios.csv", index=False)

    # --- Agentic-specific divergence ---
    # SO 2025 daily+weekly AI agent usage ≈ 24%; assume 0% in 2024 as agents not mainstream
    agent_usage = {"2024": 0.00, "2026": 0.24}
    agent_rows = []
    for scope, rr in req_rates.items():
        for _, r in rr.iterrows():
            p = r["period2"]
            agent_rows.append({
                "scope": scope,
                "period": p,
                "req_agentic_rate": r["agentic_rate"],
                "req_ai_agent_phrase_rate": r["ai_agent_phrase_rate"],
                "usage_rate": agent_usage[p],
                "divergence_req_minus_usage": r["agentic_rate"] - agent_usage[p],
            })
    agent_df = pd.DataFrame(agent_rows)
    agent_df.to_csv(TAB / "divergence_agentic.csv", index=False)

    # --- Temporal growth rates ---
    # Employer side:
    def growth(series):
        a = series.iloc[0]
        b = series.iloc[1]
        if a == 0:
            return float("inf") if b > 0 else 0.0
        return (b - a) / a

    temp_rows = []
    for scope, rr in req_rates.items():
        rr = rr.sort_values("period2")
        temp_rows.append({
            "scope": scope,
            "metric": "any_ai_rate",
            "y2024": rr["any_ai_rate"].iloc[0],
            "y2026": rr["any_ai_rate"].iloc[1],
            "abs_change_pp": (rr["any_ai_rate"].iloc[1] - rr["any_ai_rate"].iloc[0]) * 100,
            "pct_change": growth(rr["any_ai_rate"]),
        })
        for metric in ("ai_tool_rate", "ai_domain_rate", "agentic_rate", "rag_phrase_rate"):
            temp_rows.append({
                "scope": scope,
                "metric": metric,
                "y2024": rr[metric].iloc[0],
                "y2026": rr[metric].iloc[1],
                "abs_change_pp": (rr[metric].iloc[1] - rr[metric].iloc[0]) * 100,
                "pct_change": growth(rr[metric]),
            })
    # Usage side
    so_usage = {"2024": 0.62, "2026": 0.80}
    temp_rows.append({
        "scope": "benchmark",
        "metric": "SO_any_AI_usage",
        "y2024": so_usage["2024"],
        "y2026": so_usage["2026"],
        "abs_change_pp": (so_usage["2026"] - so_usage["2024"]) * 100,
        "pct_change": (so_usage["2026"] - so_usage["2024"]) / so_usage["2024"],
    })
    temp_rows.append({
        "scope": "benchmark",
        "metric": "SO_agent_usage",
        "y2024": 0.0,
        "y2026": 0.24,
        "abs_change_pp": 24.0,
        "pct_change": float("inf"),
    })
    temp_df = pd.DataFrame(temp_rows)
    temp_df.to_csv(TAB / "divergence_temporal_growth.csv", index=False)

    # --- Divergence chart ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    rr = req_rates["direct_only"].sort_values("period2")
    x = np.arange(2)
    width = 0.15
    labels = ["2024", "2026"]

    ax = axes[0]
    # req rate
    ax.bar(x - width*2, rr["any_ai_rate"], width, label="Any-AI requirement (direct SWE postings)", color="#d62728")
    ax.bar(x - width, rr["ai_tool_rate"], width, label="AI-tool requirement", color="#ff7f0e")
    ax.bar(x, rr["ai_domain_rate"], width, label="AI-domain requirement", color="#2ca02c")
    # usage benchmarks (central scenario)
    usage_central = [0.62, 0.80]
    usage_low = [0.50, 0.65]
    usage_high = [0.75, 0.90]
    ax.plot(x + width, usage_central, marker="o", linestyle="-", color="#1f77b4", label="SO dev AI usage (central)")
    ax.fill_between(x + width, usage_low, usage_high, alpha=0.2, color="#1f77b4", label="SO range (low-high)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate")
    ax.set_title("Employer AI requirements vs developer AI usage")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, 1.0)

    ax = axes[1]
    ax.bar(x - width/2, rr["agentic_rate"], width, label="'agentic' in postings", color="#9467bd")
    ax.bar(x + width/2, rr["ai_agent_phrase_rate"], width, label="AI-agent phrase", color="#8c564b")
    ax.plot(x, [0.0, 0.24], marker="o", color="#1f77b4", label="Worker agent usage (SO daily+weekly)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Agentic AI requirements vs usage")
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG / "divergence_chart.png", dpi=120)
    plt.close()

    print("Saved T23 tables and figures.")


if __name__ == "__main__":
    main()
