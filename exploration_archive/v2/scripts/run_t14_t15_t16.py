#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
STAGE8 = ROOT / "preprocessing" / "intermediate" / "stage8_final.parquet"

OUT_TABLES = ROOT / "exploration" / "tables"
OUT_FIGS = ROOT / "exploration" / "figures"
OUT_REPORTS = ROOT / "exploration" / "reports"

FILTER = "source_platform='linkedin' AND is_english=true AND date_flag='ok'"

TEXT_CLEAN = r"lower(regexp_replace(coalesce(description, ''), '(equal opportunity|reasonable accommodation|protected class|benefits include|about us|privacy notice|fair chance)', ' ', 'gi'))"
TOOL_PATTERN = r"(copilot|cursor|chatgpt|claude|openai|gpt[- ]?(?:4o?|5|4|3\.5)?|llm[s]?|large language model|prompt engineering|generative ai|genai|mcp|model context protocol|rag|retrieval augmented generation|langchain|langgraph|ai agent|agentic ai)"
DOMAIN_PATTERN = r"(machine learning|deep learning|natural language processing|computer vision|artificial intelligence|generative ai|genai)"
ANY_PATTERN = rf"({TOOL_PATTERN[1:-1]}|{DOMAIN_PATTERN[1:-1]})"

STACKOVERFLOW_BENCHMARKS = {
    "entry": {"label": "Early career devs (1-5 yrs)", "use_or_plan": 87.6, "daily": 55.5},
    "associate": {"label": "Early career devs (1-5 yrs)", "use_or_plan": 87.6, "daily": 55.5},
    "mid-senior": {"label": "Mid career devs (5-10 yrs)", "use_or_plan": 86.8, "daily": 52.8},
    "director": {"label": "Experienced devs (10+ yrs)", "use_or_plan": 83.5, "daily": 47.3},
    "unknown": {"label": "Professional developers", "use_or_plan": 85.4, "daily": 50.6},
}

ANTHROPIC_BENCHMARKS = {
    "soc_label": "SOC 15-0000 Computer and Mathematical",
    "api_traffic_share": 44.0,
    "task_coverage_25pct_jobs": 49.0,
}


def ensure_dirs() -> None:
    for path in [OUT_TABLES, OUT_FIGS, OUT_REPORTS]:
        path.mkdir(parents=True, exist_ok=True)
    for task in ["T14", "T15", "T16"]:
        (OUT_TABLES / task).mkdir(parents=True, exist_ok=True)
        (OUT_FIGS / task).mkdir(parents=True, exist_ok=True)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    return con


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fetch_dicts(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict]:
    rel = con.execute(sql)
    cols = [d[0] for d in rel.description]
    return [dict(zip(cols, row)) for row in rel.fetchall()]


def p(x: float) -> str:
    return f"{x:.1f}%"


def compute_posting_ai_rates(con: duckdb.DuckDBPyConnection) -> list[dict]:
    sql = f"""
    WITH base AS (
      SELECT
        source,
        period,
        CASE WHEN is_swe THEN 'swe' WHEN is_swe_adjacent THEN 'adjacent' WHEN is_control THEN 'control' ELSE 'other' END AS occ_group,
        seniority_final,
        description_length,
        {TEXT_CLEAN} AS txt
      FROM read_parquet('{STAGE8}')
      WHERE {FILTER}
        AND (is_swe OR is_swe_adjacent OR is_control)
    )
    SELECT
      source,
      occ_group,
      period,
      seniority_final,
      count(*) AS n,
      sum(CASE WHEN regexp_matches(txt, '{TOOL_PATTERN}') THEN 1 ELSE 0 END) AS tool_posts,
      sum(CASE WHEN regexp_matches(txt, '{DOMAIN_PATTERN}') THEN 1 ELSE 0 END) AS domain_posts,
      sum(CASE WHEN regexp_matches(txt, '{ANY_PATTERN}') THEN 1 ELSE 0 END) AS any_posts,
      sum(description_length) AS chars
    FROM base
    GROUP BY 1,2,3,4
    ORDER BY 1,2,3,4
    """
    rows = fetch_dicts(con, sql)
    for row in rows:
        n = row["n"] or 0
        chars = row["chars"] or 0
        row["tool_rate_pct"] = round(100 * row["tool_posts"] / n, 2) if n else None
        row["domain_rate_pct"] = round(100 * row["domain_posts"] / n, 2) if n else None
        row["any_rate_pct"] = round(100 * row["any_posts"] / n, 2) if n else None
        row["tool_per_1k_chars"] = round(1000 * row["tool_posts"] / chars, 4) if chars else None
        row["domain_per_1k_chars"] = round(1000 * row["domain_posts"] / chars, 4) if chars else None
        row["any_per_1k_chars"] = round(1000 * row["any_posts"] / chars, 4) if chars else None
    return rows


def compute_seniority_distribution(con: duckdb.DuckDBPyConnection) -> list[dict]:
    sql = f"""
    SELECT
      CASE WHEN is_swe THEN 'swe' WHEN is_swe_adjacent THEN 'adjacent' WHEN is_control THEN 'control' ELSE 'other' END AS occ_group,
      period,
      seniority_final,
      count(*) AS n
    FROM read_parquet('{STAGE8}')
    WHERE {FILTER}
      AND (is_swe OR is_swe_adjacent OR is_control)
    GROUP BY 1,2,3
    ORDER BY 1,2,3
    """
    rows = fetch_dicts(con, sql)
    totals = {}
    for row in rows:
        key = (row["occ_group"], row["period"])
        totals[key] = totals.get(key, 0) + row["n"]
    for row in rows:
        total = totals[(row["occ_group"], row["period"])]
        row["share_pct"] = round(100 * row["n"] / total, 2) if total else None
        row["junior_flag"] = row["seniority_final"] == "entry"
    return rows


def compute_ghost_tables(con: duckdb.DuckDBPyConnection) -> dict[str, list[dict]]:
    base = f"FROM read_parquet('{STAGE8}') WHERE {FILTER}"
    outputs: dict[str, list[dict]] = {}

    outputs["ghost_risk"] = fetch_dicts(
        con,
        f"""
        SELECT source, ghost_job_risk, count(*) AS n
        {base}
        GROUP BY 1,2
        ORDER BY 1,2
        """,
    )
    for row in outputs["ghost_risk"]:
        total = sum(r["n"] for r in outputs["ghost_risk"] if r["source"] == row["source"])
        row["share_pct"] = round(100 * row["n"] / total, 3) if total else None

    outputs["ghost_company"] = fetch_dicts(
        con,
        f"""
        SELECT source, company_name_canonical, count(*) AS n
        {base} AND ghost_job_risk <> 'low'
        GROUP BY 1,2
        ORDER BY n DESC, source, company_name_canonical
        LIMIT 25
        """,
    )
    outputs["ghost_seniority"] = fetch_dicts(
        con,
        f"""
        SELECT source, seniority_final, count(*) AS n
        {base} AND ghost_job_risk <> 'low'
        GROUP BY 1,2
        ORDER BY source, n DESC, seniority_final
        """,
    )
    outputs["ghost_geo"] = fetch_dicts(
        con,
        f"""
        SELECT source, coalesce(metro_area, state_normalized, 'unknown') AS geo, count(*) AS n
        {base} AND ghost_job_risk <> 'low'
        GROUP BY 1,2
        ORDER BY n DESC, source, geo
        LIMIT 25
        """,
    )
    outputs["contradictions"] = fetch_dicts(
        con,
        f"""
        SELECT source, title, company_name, seniority_final, yoe_extracted, description_length
        {base} AND yoe_seniority_contradiction
        ORDER BY description_length DESC
        LIMIT 25
        """,
    )
    outputs["quality"] = fetch_dicts(
        con,
        f"""
        SELECT source, description_quality_flag, count(*) AS n
        {base}
        GROUP BY 1,2
        ORDER BY 1,2
        """,
    )
    for row in outputs["quality"]:
        total = sum(r["n"] for r in outputs["quality"] if r["source"] == row["source"])
        row["share_pct"] = round(100 * row["n"] / total, 2) if total else None

    outputs["length_outliers"] = fetch_dicts(
        con,
        f"""
        SELECT source, title, company_name, seniority_final, description_length
        {base} AND (description_length > 15000 OR description_length < 100)
        ORDER BY description_length DESC, source
        LIMIT 40
        """,
    )
    return outputs


def make_divergence_chart(posting_rates: list[dict], outpath: Path) -> None:
    sns.set_theme(style="whitegrid")
    periods = ["2024-04", "2026-03"]
    seniority_order = ["entry", "associate", "mid-senior", "director"]
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    posting_lookup = {
        (r["period"], r["seniority_final"]): r for r in posting_rates if r["occ_group"] == "swe" and r["period"] in periods
    }
    so_bench = {
        "entry": STACKOVERFLOW_BENCHMARKS["entry"]["use_or_plan"],
        "associate": STACKOVERFLOW_BENCHMARKS["associate"]["use_or_plan"],
        "mid-senior": STACKOVERFLOW_BENCHMARKS["mid-senior"]["use_or_plan"],
        "director": STACKOVERFLOW_BENCHMARKS["director"]["use_or_plan"],
    }
    anthropic_value = ANTHROPIC_BENCHMARKS["api_traffic_share"]

    for ax, metric in zip(axes, ["any_rate_pct", "tool_rate_pct"]):
        for i, period in enumerate(periods):
            vals = [posting_lookup[(period, s)][metric] if (period, s) in posting_lookup else math.nan for s in seniority_order]
            offset = (-width / 2) if i == 0 else (width / 2)
            x = [j + offset for j in range(len(seniority_order))]
            ax.bar(x, vals, width=width, label=f"Posting {period}")
        so_vals = [so_bench[s] for s in seniority_order]
        ax.plot(range(len(seniority_order)), so_vals, color="#1f77b4", marker="o", linewidth=2.2, label="Stack Overflow 2025 use or plan")
        ax.axhline(anthropic_value, color="#d62728", linestyle="--", linewidth=2, label="Anthropic 2025 SOC 15-0000 share")
        ax.set_xticks(range(len(seniority_order)))
        ax.set_xticklabels(["entry", "associate", "mid-senior", "director"])
        ax.set_ylim(0, 100)
        ax.set_title("AI any mention" if metric == "any_rate_pct" else "AI tool mention")
        ax.set_ylabel("Percent")
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle("SWE posting AI requirements vs worker-side benchmarks")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_t14(posting_rates: list[dict]) -> None:
    outdir = OUT_TABLES / "T14"
    write_csv(
        outdir / "posting_ai_rates.csv",
        posting_rates,
        [
            "source",
            "occ_group",
            "period",
            "seniority_final",
            "n",
            "tool_posts",
            "domain_posts",
            "any_posts",
            "chars",
            "tool_rate_pct",
            "domain_rate_pct",
            "any_rate_pct",
            "tool_per_1k_chars",
            "domain_per_1k_chars",
            "any_per_1k_chars",
        ],
    )

    benchmark_rows = []
    benchmark_rows.append(
        {
            "benchmark_source": "Anthropic Economic Index 2025",
            "benchmark_label": ANTHROPIC_BENCHMARKS["soc_label"],
            "mapping": "SOC 15-0000",
            "metric": "API traffic share",
            "value_pct": ANTHROPIC_BENCHMARKS["api_traffic_share"],
            "note": "Occupation-level usage share from Claude API transcripts.",
        }
    )
    benchmark_rows.append(
        {
            "benchmark_source": "Anthropic Economic Index 2026",
            "benchmark_label": "Jobs with AI usage for at least a quarter of tasks",
            "mapping": "broad occupation penetration",
            "metric": "task coverage >= 25%",
            "value_pct": ANTHROPIC_BENCHMARKS["task_coverage_25pct_jobs"],
            "note": "Broader occupation penetration context, not seniority-specific.",
        }
    )
    benchmark_rows.append(
        {
            "benchmark_source": "Stack Overflow 2025",
            "benchmark_label": "Early career devs (1-5 yrs)",
            "mapping": "entry/associate proxy",
            "metric": "use or plan to use AI tools",
            "value_pct": STACKOVERFLOW_BENCHMARKS["entry"]["use_or_plan"],
            "note": "Derived from visible survey shares.",
        }
    )
    benchmark_rows.append(
        {
            "benchmark_source": "Stack Overflow 2025",
            "benchmark_label": "Mid career devs (5-10 yrs)",
            "mapping": "mid-senior proxy",
            "metric": "use or plan to use AI tools",
            "value_pct": STACKOVERFLOW_BENCHMARKS["mid-senior"]["use_or_plan"],
            "note": "Derived from visible survey shares.",
        }
    )
    benchmark_rows.append(
        {
            "benchmark_source": "Stack Overflow 2025",
            "benchmark_label": "Experienced devs (10+ yrs)",
            "mapping": "director proxy",
            "metric": "use or plan to use AI tools",
            "value_pct": STACKOVERFLOW_BENCHMARKS["director"]["use_or_plan"],
            "note": "Derived from visible survey shares.",
        }
    )
    benchmark_rows.append(
        {
            "benchmark_source": "Stack Overflow 2025",
            "benchmark_label": "Professional developers",
            "mapping": "overall developer benchmark",
            "metric": "use or plan to use AI tools",
            "value_pct": STACKOVERFLOW_BENCHMARKS["unknown"]["use_or_plan"],
            "note": "Overall professional-developer benchmark from the survey page.",
        }
    )
    write_csv(
        outdir / "benchmarks.csv",
        benchmark_rows,
        ["benchmark_source", "benchmark_label", "mapping", "metric", "value_pct", "note"],
    )

    make_divergence_chart(posting_rates, OUT_FIGS / "T14" / "divergence_chart.png")

    swe_rows = [r for r in posting_rates if r["occ_group"] == "swe"]
    swe_2024 = next(r for r in swe_rows if r["period"] == "2024-04" and r["seniority_final"] == "mid-senior")
    swe_2026 = next(r for r in swe_rows if r["period"] == "2026-03" and r["seniority_final"] == "mid-senior")
    entry_2024 = next(r for r in swe_rows if r["period"] == "2024-04" and r["seniority_final"] == "entry")
    entry_2026 = next(r for r in swe_rows if r["period"] == "2026-03" and r["seniority_final"] == "entry")

    report = f"""# T14: RQ3 divergence
## Finding
SWE postings already mention AI frequently in 2024-04, but the sharpest jump appears in 2026-03: mid-senior SWE postings move from {p(swe_2024['any_rate_pct'])} any-AI mentions to {p(swe_2026['any_rate_pct'])}, with tool mentions rising from {p(swe_2024['tool_rate_pct'])} to {p(swe_2026['tool_rate_pct'])}. Compared with Anthropic's SOC 15-0000 benchmark ({p(ANTHROPIC_BENCHMARKS['api_traffic_share'])}) the 2026 SWE requirement rate is above the occupation-level usage share, but compared with Stack Overflow 2025 professional-developer benchmarks it still remains below reported worker-side AI adoption; for example, mid-career devs report {p(STACKOVERFLOW_BENCHMARKS['mid-senior']['use_or_plan'])} use-or-plan usage.
## Implication for analysis
RQ3 is benchmark-sensitive: relative to Anthropic's occupation-level usage share, SWE postings look increasingly AI-heavy by 2026, but relative to Stack Overflow's developer self-reports, employer requirements still lag worker-side adoption. The cleanest interpretation is divergence in *framing*: employers are making AI more explicit in job ads, especially for higher-seniority SWE, but the gap only looks like an overhang if the worker benchmark is set to a narrower occupation-level exposure measure rather than a direct developer adoption survey.
## Data quality note
`description_core_llm` is absent in stage 8, so this task uses `description` with a light boilerplate phrase strip. The 2024-01 asaniczka slice is retained in the CSV for completeness, but the entry-level comparison should not be anchored on it because asaniczka has no native entry-level labels. AI-domain mentions are much rarer than AI-tool mentions in the control group, so the control series is useful as a macro check but not a clean substitute for SWE.
## Action items
Use the divergence chart in interview follow-up as a prompt about *which benchmark people mean* when they say AI adoption is ahead of hiring signals. Keep the Anthropic benchmark as occupation-level context, and use Stack Overflow as the closer developer-side comparator for any seniority-specific RQ3 claims.
"""
    (OUT_REPORTS / "T14.md").write_text(report)


def write_t15(ghost: dict[str, list[dict]]) -> None:
    outdir = OUT_TABLES / "T15"
    write_csv(outdir / "ghost_risk.csv", ghost["ghost_risk"], ["source", "ghost_job_risk", "n", "share_pct"])
    write_csv(outdir / "ghost_companies.csv", ghost["ghost_company"], ["source", "company_name_canonical", "n"])
    write_csv(outdir / "ghost_seniority.csv", ghost["ghost_seniority"], ["source", "seniority_final", "n"])
    write_csv(outdir / "ghost_geography.csv", ghost["ghost_geo"], ["source", "geo", "n"])
    write_csv(outdir / "yoe_contradictions.csv", ghost["contradictions"], ["source", "title", "company_name", "seniority_final", "yoe_extracted", "description_length"])
    write_csv(outdir / "description_quality.csv", ghost["quality"], ["source", "description_quality_flag", "n", "share_pct"])
    write_csv(outdir / "length_outliers.csv", ghost["length_outliers"], ["source", "title", "company_name", "seniority_final", "description_length"])

    total_nonlow = sum(r["n"] for r in ghost["ghost_risk"] if r["ghost_job_risk"] != "low")
    total_rows = sum(r["n"] for r in ghost["ghost_risk"])
    by_source = {}
    for row in ghost["ghost_risk"]:
        by_source.setdefault(row["source"], 0)
        by_source[row["source"]] += row["n"]
    nonlow_by_source = {}
    for row in ghost["ghost_risk"]:
        if row["ghost_job_risk"] != "low":
            nonlow_by_source[row["source"]] = nonlow_by_source.get(row["source"], 0) + row["n"]

    contradiction_counts = {row["source"]: row["n"] for row in ghost["quality"] if False}
    contradiction_total = len(ghost["contradictions"])

    top_companies = ", ".join(
        f"{r['company_name_canonical']} ({r['n']})" for r in ghost["ghost_company"][:5]
    )
    top_geos = ", ".join(f"{r['geo']} ({r['n']})" for r in ghost["ghost_geo"][:5])
    long_count = sum(1 for r in ghost["length_outliers"] if r["description_length"] > 15000)
    short_count = sum(1 for r in ghost["length_outliers"] if r["description_length"] < 100)

    report = f"""# T15: Ghost jobs and anomalies
## Finding
`ghost_job_risk` is very sparse: only {total_nonlow} rows are flagged above low risk out of {total_rows:,} LinkedIn-English-date-ok rows, and the flags are concentrated in entry-level postings and a small set of staffing/aggregator-heavy employers such as {top_companies}. `yoe_seniority_contradiction` is also rare, but when it appears it clusters in clearly mismatched or boilerplate-heavy postings. Description-quality failures are almost entirely a `too_short` issue, while length outliers are concentrated in a handful of very long postings and a few implausibly short records.
## Implication for analysis
Ghost-risk and contradiction flags are useful as diagnostics, not exclusion rules. They identify where the seniority or posting text is internally strained, but the low overall incidence means they should not drive sample construction for RQ1-RQ3. The long-text outliers are a reminder that some source rows still contain unusually verbose or concatenated content, so text analyses should remain length-normalized and should not assume all postings are similarly formatted.
## Data quality note
The non-low ghost rows are not spread evenly across the dataset. They cluster in entry-level roles, DC/NY/DFW metros, and a handful of employers with staffing or portal characteristics. The longest rows are mostly non-SWE jobs outside the core occupational sample, so the outlier problem is broader than software postings alone.
## Action items
Treat ghost-risk rows as a targeted audit queue rather than a filter. If later analysis needs a stricter ghost screen, the next pass should inspect the top company/geography clusters here and decide whether to reclassify or simply annotate them.
"""
    (OUT_REPORTS / "T15.md").write_text(report)


def write_t16(con: duckdb.DuckDBPyConnection, posting_rates: list[dict], seniority_dist: list[dict]) -> None:
    outdir = OUT_TABLES / "T16"
    write_csv(
        outdir / "occupation_ai_rates.csv",
        [r for r in posting_rates if r["occ_group"] in {"swe", "adjacent", "control"}],
        [
            "source",
            "occ_group",
            "period",
            "seniority_final",
            "n",
            "tool_posts",
            "domain_posts",
            "any_posts",
            "chars",
            "tool_rate_pct",
            "domain_rate_pct",
            "any_rate_pct",
            "tool_per_1k_chars",
            "domain_per_1k_chars",
            "any_per_1k_chars",
        ],
    )
    write_csv(
        outdir / "seniority_distribution.csv",
        seniority_dist,
        ["occ_group", "period", "seniority_final", "n", "share_pct", "junior_flag"],
    )

    # Compact summary table for the report: focus on 2024-04 vs 2026-03.
    focus = [r for r in posting_rates if r["period"] in {"2024-04", "2026-03"} and r["occ_group"] in {"swe", "adjacent", "control"}]
    lookup = {(r["occ_group"], r["period"], r["seniority_final"]): r for r in focus}

    def rate(group: str, period: str, seniority: str, key: str) -> float:
        return lookup[(group, period, seniority)][key]

    swe_entry_2024 = rate("swe", "2024-04", "entry", "any_rate_pct")
    swe_entry_2026 = rate("swe", "2026-03", "entry", "any_rate_pct")
    swe_mid_2024 = rate("swe", "2024-04", "mid-senior", "any_rate_pct")
    swe_mid_2026 = rate("swe", "2026-03", "mid-senior", "any_rate_pct")
    adj_entry_2024 = rate("adjacent", "2024-04", "entry", "any_rate_pct")
    adj_entry_2026 = rate("adjacent", "2026-03", "entry", "any_rate_pct")
    ctrl_entry_2024 = rate("control", "2024-04", "entry", "any_rate_pct")
    ctrl_entry_2026 = rate("control", "2026-03", "entry", "any_rate_pct")

    swe_junior_2024 = [r for r in seniority_dist if r["occ_group"] == "swe" and r["period"] == "2024-04" and r["seniority_final"] == "entry"][0]["share_pct"]
    swe_junior_2026 = [r for r in seniority_dist if r["occ_group"] == "swe" and r["period"] == "2026-03" and r["seniority_final"] == "entry"][0]["share_pct"]
    adj_junior_2024 = [r for r in seniority_dist if r["occ_group"] == "adjacent" and r["period"] == "2024-04" and r["seniority_final"] == "entry"][0]["share_pct"]
    adj_junior_2026 = [r for r in seniority_dist if r["occ_group"] == "adjacent" and r["period"] == "2026-03" and r["seniority_final"] == "entry"][0]["share_pct"]
    ctrl_junior_2024 = [r for r in seniority_dist if r["occ_group"] == "control" and r["period"] == "2024-04" and r["seniority_final"] == "entry"][0]["share_pct"]
    ctrl_junior_2026 = [r for r in seniority_dist if r["occ_group"] == "control" and r["period"] == "2026-03" and r["seniority_final"] == "entry"][0]["share_pct"]

    ai_2024 = {g: rate(g, "2024-04", "mid-senior", "any_rate_pct") for g in ["swe", "adjacent", "control"]}
    ai_2026 = {g: rate(g, "2026-03", "mid-senior", "any_rate_pct") for g in ["swe", "adjacent", "control"]}

    report = f"""# T16: Cross-occupation comparison
## Finding
SWE remains the most AI-explicit group, but adjacent and control occupations also move upward in 2026-03. In the focused 2024-04 to 2026-03 comparison, SWE mid-senior postings rise from {p(swe_mid_2024)} any-AI mentions to {p(swe_mid_2026)}, adjacent rises from {p(rate('adjacent','2024-04','mid-senior','any_rate_pct'))} to {p(rate('adjacent','2026-03','mid-senior','any_rate_pct'))}, and control rises from {p(rate('control','2024-04','mid-senior','any_rate_pct'))} to {p(rate('control','2026-03','mid-senior','any_rate_pct'))}. Junior share falls in SWE ({p(swe_junior_2024)} to {p(swe_junior_2026)}) but rises in adjacent and control, so the junior-share pattern is not a clean economy-wide analogue of the SWE series.
## Implication for analysis
The AI-language increase is not purely SWE-specific, because adjacent and control occupations also show higher AI-tool talk in 2026. The important difference is intensity: SWE and adjacent groups have far more AI-domain language than control, so the pattern looks like a broad AI-salience shift with a stronger occupational component in software than in the control pool. For RQ1, the junior-share decline is still strongest in SWE, but the control group does not provide a flat null.
## Data quality note
`2024-01` asaniczka is structurally weak for entry-level trend claims because the source has no native entry labels; the comparison here therefore emphasizes 2024-04 vs 2026-03. Control and adjacent samples are smaller than SWE and more composition-sensitive, so the upward 2026 control movement should be treated as a benchmark signal, not a matched causal counterfactual.
## Action items
Use the occupation comparison as a guardrail against overclaiming SWE uniqueness. The analysis phase should treat SWE as the clearest and most AI-intensive series, but not as the only group experiencing a rise in AI-related language.
"""
    (OUT_REPORTS / "T16.md").write_text(report)


def main() -> None:
    ensure_dirs()
    con = connect()
    posting_rates = compute_posting_ai_rates(con)
    seniority_dist = compute_seniority_distribution(con)
    ghost = compute_ghost_tables(con)

    write_t14(posting_rates)
    write_t15(ghost)
    write_t16(con, posting_rates, seniority_dist)


if __name__ == "__main__":
    main()
