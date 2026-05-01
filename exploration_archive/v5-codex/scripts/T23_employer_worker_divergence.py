from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from T22_T23_common import (
    AI_DOMAIN_RE,
    AI_GENERAL_RE,
    AI_TOOL_RE,
    REPORT_DIR,
    ensure_dirs,
    load_core_text_frame,
    load_full_text_frame,
    load_tech_counts,
    qdf,
)


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "exploration" / "tables" / "T23"
FIG_DIR = ROOT / "exploration" / "figures" / "T23"


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


def build_feature_frame(con: duckdb.DuckDBPyConnection, text_source: str = "llm") -> pd.DataFrame:
    frame = load_core_text_frame(con, text_source=text_source)
    tech = load_tech_counts(con, frame["uid"].tolist())
    frame = frame.merge(tech, on="uid", how="left")
    frame["core_text"] = frame["core_text"].fillna("")
    frame["any_ai_tool"] = frame["core_text"].map(lambda s: bool(AI_TOOL_RE.search(s or "")))
    frame["any_ai_domain"] = frame["core_text"].map(lambda s: bool(AI_DOMAIN_RE.search(s or "")))
    frame["any_ai_general"] = frame["core_text"].map(lambda s: bool(AI_GENERAL_RE.search(s or "")))
    frame["any_ai"] = frame[["any_ai_tool", "any_ai_domain", "any_ai_general"]].any(axis=1)
    return frame


def benchmark_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "benchmark": "StackOverflow 2024 professional devs: AI-assisted tech at work",
                "benchmark_rate": 0.324,
                "source": "https://survey.stackoverflow.co/2024/professional-developers/",
                "note": "Official survey page; access to AI-assisted tech at work.",
            },
            {
                "benchmark": "StackOverflow 2024 professional devs: AI-powered search",
                "benchmark_rate": 0.15,
                "source": "https://survey.stackoverflow.co/2024/professional-developers/",
                "note": "Official survey page; technical-question search behavior.",
            },
            {
                "benchmark": "StackOverflow 2025: use or plan to use AI tools",
                "benchmark_rate": 0.84,
                "source": "https://survey.stackoverflow.co/2025/",
                "note": "Official survey page; broad usage/planning benchmark.",
            },
            {
                "benchmark": "StackOverflow 2025: professional devs using AI tools daily",
                "benchmark_rate": 0.51,
                "source": "https://survey.stackoverflow.co/2025/",
                "note": "Official survey page; daily usage benchmark.",
            },
            {
                "benchmark": "GitHub US developer survey 2024: AI coding tools at work",
                "benchmark_rate": 0.99,
                "source": "https://github.blog/wp-content/uploads/2024/08/2024-Developer-Survey-United-States.pdf",
                "note": "Large-company US sample. Use as an upper-bound benchmark only.",
            },
        ]
    )


def ai_rates_by_group(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, period, seniority_final, is_aggregator), g in frame.groupby(["source", "period", "seniority_final", "is_aggregator"], dropna=False):
        rows.append(
            {
                "source": source,
                "period": period,
                "seniority_final": seniority_final,
                "is_aggregator": bool(is_aggregator),
                "n": len(g),
                "ai_tool_rate": float(g["any_ai_tool"].mean()),
                "ai_domain_rate": float(g["any_ai_domain"].mean()),
                "ai_general_rate": float(g["any_ai_general"].mean()),
                "any_ai_rate": float(g["any_ai"].mean()),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_table(rates: pd.DataFrame, benchmarks: pd.DataFrame) -> pd.DataFrame:
    latest = rates[(rates["source"] == "scraped") & (rates["period"] == "2026-04") & (~rates["is_aggregator"])].copy()
    latest["key"] = 1
    benchmarks = benchmarks.copy()
    benchmarks["key"] = 1
    out = latest.merge(benchmarks, on="key").drop(columns=["key"])
    out["divergence_pp"] = 100.0 * (out["ai_tool_rate"] - out["benchmark_rate"])
    out["divergence_pct_of_benchmark"] = out["divergence_pp"] / (100.0 * out["benchmark_rate"])
    out["above_benchmark"] = out["divergence_pp"] > 0
    return out.sort_values(["seniority_final", "benchmark_rate"])


def main() -> None:
    ensure_dirs(TABLE_DIR, FIG_DIR, REPORT_DIR)
    con = duckdb.connect()

    primary = build_feature_frame(con, text_source="llm")
    all_core = build_feature_frame(con, text_source=None)
    raw_full = load_full_text_frame(con, text_source="raw")
    raw_full = raw_full.merge(load_tech_counts(con, raw_full["uid"].tolist()), on="uid", how="left")
    raw_full["core_text"] = raw_full["core_text"].fillna("")
    raw_full["any_ai_tool"] = raw_full["core_text"].map(lambda s: bool(AI_TOOL_RE.search(s or "")))
    raw_full["any_ai_domain"] = raw_full["core_text"].map(lambda s: bool(AI_DOMAIN_RE.search(s or "")))
    raw_full["any_ai_general"] = raw_full["core_text"].map(lambda s: bool(AI_GENERAL_RE.search(s or "")))
    raw_full["any_ai"] = raw_full[["any_ai_tool", "any_ai_domain", "any_ai_general"]].any(axis=1)

    rates = ai_rates_by_group(primary)
    save_csv(rates.assign(text_source="llm_core"), "T23_ai_requirement_rates_primary_llm.csv")
    save_csv(ai_rates_by_group(raw_full).assign(text_source="raw_full"), "T23_ai_requirement_rates_raw_full.csv")

    benchmarks = benchmark_table()
    save_csv(benchmarks, "T23_benchmark_sources.csv")
    sens = sensitivity_table(rates, benchmarks)
    save_csv(sens, "T23_benchmark_sensitivity.csv")

    # Compare direct employers versus aggregators.
    agg = (
        primary.groupby(["is_aggregator"], dropna=False)
        .agg(
            n=("uid", "size"),
            ai_tool_rate=("any_ai_tool", "mean"),
            ai_domain_rate=("any_ai_domain", "mean"),
            ai_general_rate=("any_ai_general", "mean"),
            any_ai_rate=("any_ai", "mean"),
        )
        .reset_index()
    )
    save_csv(agg, "T23_aggregator_sensitivity.csv")

    # Seniority operationalization check using the YOE proxy.
    yoe = primary.copy()
    yoe["yoe_proxy_junior"] = yoe["yoe_extracted"].fillna(999) <= 2
    yoe_rates = (
        yoe.groupby(["period", "yoe_proxy_junior"], dropna=False)
        .agg(
            n=("uid", "size"),
            ai_tool_rate=("any_ai_tool", "mean"),
            ai_domain_rate=("any_ai_domain", "mean"),
            ai_general_rate=("any_ai_general", "mean"),
            any_ai_rate=("any_ai", "mean"),
        )
        .reset_index()
    )
    save_csv(yoe_rates, "T23_yoe_proxy_ai_rates.csv")

    # Raw all-core sensitivity with same structure for easy comparison in the memo.
    raw_vs_llm = (
        pd.concat(
            [
                primary.assign(text_source="llm_core"),
                raw_full.assign(text_source="raw_full"),
            ],
            ignore_index=True,
        )
        .groupby(["text_source"], dropna=False)
        .agg(
            n=("uid", "size"),
            ai_tool_rate=("any_ai_tool", "mean"),
            ai_domain_rate=("any_ai_domain", "mean"),
            ai_general_rate=("any_ai_general", "mean"),
            any_ai_rate=("any_ai", "mean"),
        )
        .reset_index()
    )
    save_csv(raw_vs_llm, "T23_text_source_sensitivity.csv")

    # Divergence chart.
    chart_df = (
        rates[(rates["is_aggregator"] == False)]
        .groupby(["period"], dropna=False)
        .agg(
            ai_tool_rate=("ai_tool_rate", "mean"),
            any_ai_rate=("any_ai_rate", "mean"),
        )
        .reset_index()
    )
    chart_df = chart_df.sort_values("period")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(chart_df))
    ax.plot(x, chart_df["ai_tool_rate"], marker="o", linewidth=2, label="Posting AI-tool rate")
    ax.plot(x, chart_df["any_ai_rate"], marker="o", linewidth=2, label="Posting any-AI rate")
    for _, row in benchmarks.iterrows():
        ax.axhline(row["benchmark_rate"], linestyle="--", linewidth=1, alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(chart_df["period"].tolist())
    ax.set_ylabel("Share of postings")
    ax.set_title("Posting-side AI requirements versus worker-usage benchmarks")
    ax.legend(frameon=False)
    save_fig(fig, "T23_ai_requirement_vs_benchmarks.png")

    # Helper table for the memo.
    save_csv(
        chart_df.assign(
            benchmark_50pct=0.50,
            benchmark_65pct=0.65,
            benchmark_75pct=0.75,
            benchmark_85pct=0.85,
        ),
        "T23_period_summary_for_memo.csv",
    )


if __name__ == "__main__":
    main()
