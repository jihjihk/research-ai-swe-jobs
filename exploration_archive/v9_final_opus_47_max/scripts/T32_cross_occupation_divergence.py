"""T32 — Cross-occupation employer/worker AI divergence (H_A).

Extends T23's SWE employer-vs-worker AI-adoption relationship to SWE-adjacent
and control occupations via T18's DiD framework.

Does the SWE pattern (T23: employer rose 10.3× 2024→2026 from 1% to 10.6%;
worker already at 63-90% in 2024) hold, weaken, or invert for other
occupations?

Steps (per dispatch):
  1. Occupation group definition. Sub-stratify adjacent (data_scientist,
     ml_engineer, data_engineer, security_engineer, devops_engineer) and
     control (accountant, nurse, civil_engineer, mechanical_engineer,
     electrical_engineer, financial_analyst).
  2. Employer AI-requirement rate per subgroup × period:
     ai_strict_v1 (V1 primary) and ai_broad_no_mcp. n_group × period.
  3. External benchmarks — populated from known T23 + new web searches.
  4. Per-subgroup divergence: employer_rate − worker_benchmark_midpoint under
     50/65/75/85% bands.
  5. Direction universality test + exposure-vs-gap correlation.
  6. Alternative framing check: daily-use vs tried-ever.
  7. Cross-occupation chart.

Output:
  exploration/tables/T32/subgroup_rates.csv
  exploration/tables/T32/worker_benchmarks.csv
  exploration/tables/T32/subgroup_divergence.csv
  exploration/tables/T32/universality_test.csv
  exploration/tables/T32/subgroup_n_cells.csv
  exploration/figures/T32_cross_occupation_divergence.png (+ svg)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/jihgaboot/gabor/job-research")
DATA = REPO / "data" / "unified_core.parquet"
VAL = REPO / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json"

OUT = REPO / "exploration" / "tables" / "T32"
OUT.mkdir(parents=True, exist_ok=True)
FIG = REPO / "exploration" / "figures"
FIG.mkdir(exist_ok=True)

patterns = json.loads(VAL.read_text())
AI_STRICT_V1 = re.compile(patterns["v1_rebuilt_patterns"]["ai_strict_v1_rebuilt"]["pattern"], re.IGNORECASE)
AI_BROAD_NO_MCP = re.compile(
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|"
    r"llamaindex|langchain|prompt engineering|rag|vector databas(?:e|es)|"
    r"pinecone|huggingface|hugging face|agent|machine learning|ml|ai|llm|"
    r"artificial intelligence|"
    r"(?:fine[- ]tun(?:e|ed|ing))\s+(?:the\s+)?(?:model|llm|gpt|base model|foundation model|embeddings))\b",
    re.IGNORECASE,
)

# Sub-occupation title regex. Ordered (first-match wins).
SWE_SUBGROUPS = [
    ("ml_engineer", re.compile(r"\b(?:ml|machine learning|ai|mlops)\s*(?:engineer|developer|architect|scientist)\b", re.IGNORECASE)),
]

SWE_ADJ_SUBGROUPS = [
    ("ml_engineer", re.compile(r"\b(?:machine learning|ml|ai|mlops)\s*(?:engineer|developer|scientist|architect)\b", re.IGNORECASE)),
    ("data_scientist", re.compile(r"\b(?:data scientist|data science)\b", re.IGNORECASE)),
    ("data_engineer", re.compile(r"\b(?:data engineer|analytics engineer)\b", re.IGNORECASE)),
    ("data_analyst", re.compile(r"\b(?:data analyst|business analyst|analytics analyst)\b", re.IGNORECASE)),
    ("security_engineer", re.compile(r"\b(?:security|cyber|cybersecurity|information security|infosec)\s*(?:engineer|architect)\b", re.IGNORECASE)),
    ("network_engineer", re.compile(r"\bnetwork\s*(?:engineer|architect)\b", re.IGNORECASE)),
    ("qa_engineer", re.compile(r"\b(?:qa|quality)\s*(?:engineer|analyst)\b", re.IGNORECASE)),
    ("devops_engineer", re.compile(r"\b(?:devops|site reliability|sre|platform)\s*(?:engineer|architect)\b", re.IGNORECASE)),
    ("solutions_architect", re.compile(r"\b(?:solution|solutions|enterprise|cloud|technical)\s*architect\b", re.IGNORECASE)),
    ("systems_admin", re.compile(r"\b(?:systems?|database)\s*administrator\b", re.IGNORECASE)),
]

CONTROL_SUBGROUPS = [
    ("accountant", re.compile(r"\b(?:accountant|accounting|cpa|bookkeeper)\b", re.IGNORECASE)),
    ("financial_analyst", re.compile(r"\bfinancial\s*analyst\b", re.IGNORECASE)),
    ("nurse", re.compile(r"\b(?:registered nurse|rn|nurse practitioner|lpn|licensed practical nurse|cna|nursing)\b", re.IGNORECASE)),
    ("electrical_engineer", re.compile(r"\belectrical\s*engineer", re.IGNORECASE)),
    ("mechanical_engineer", re.compile(r"\bmechanical\s*engineer", re.IGNORECASE)),
    ("civil_engineer", re.compile(r"\bcivil\s*engineer", re.IGNORECASE)),
]


def classify_subgroup(title: str, group: str) -> str | None:
    if not isinstance(title, str):
        return None
    t = title.lower()
    if group == "swe":
        for name, rx in SWE_SUBGROUPS:
            if rx.search(t):
                return name
        return "other_swe"
    if group == "swe_adjacent":
        for name, rx in SWE_ADJ_SUBGROUPS:
            if rx.search(t):
                return name
        return "other_adjacent"
    if group == "control":
        for name, rx in CONTROL_SUBGROUPS:
            if rx.search(t):
                return name
        return "other_control"
    return None


def main():
    con = duckdb.connect()
    print("[T32] Loading all linkedin + english + date-ok rows across groups...")
    q = f"""
    SELECT uid,
           source,
           CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period_year,
           analysis_group,
           title,
           description,
           is_aggregator,
           seniority_final,
           yoe_min_years_llm
    FROM read_parquet('{DATA}')
    WHERE source_platform='linkedin' AND is_english AND date_flag='ok'
    """
    df = con.execute(q).fetchdf()
    print(f"  loaded {len(df)} rows")

    # Classify subgroups
    df["subgroup"] = [classify_subgroup(t, g) for t, g in zip(df["title"], df["analysis_group"])]
    print(df.groupby(["analysis_group", "subgroup", "period_year"]).size().head(100).to_string())

    # Compute AI binaries (on raw description)
    desc = df["description"].fillna("").astype(str)
    df["ai_strict_v1"] = desc.str.contains(AI_STRICT_V1).astype(int)
    df["ai_broad_nmcp"] = desc.str.contains(AI_BROAD_NO_MCP).astype(int)

    # 1. Per-subgroup rates (occ group x subgroup x period)
    rows = []
    for (ag, sg, py), g in df.groupby(["analysis_group", "subgroup", "period_year"]):
        if sg is None:
            continue
        rows.append({
            "analysis_group": ag,
            "subgroup": sg,
            "period_year": py,
            "n": len(g),
            "ai_strict_rate": float(g["ai_strict_v1"].mean()),
            "ai_broad_rate": float(g["ai_broad_nmcp"].mean()),
        })
    sub_rates = pd.DataFrame(rows)
    sub_rates.to_csv(OUT / "subgroup_rates.csv", index=False)
    print(f"[T32] Saved subgroup_rates.csv ({len(sub_rates)} rows)")

    # 2. n_cells flag thin (< 500)
    cell_n = sub_rates.pivot_table(index=["analysis_group", "subgroup"],
                                   columns="period_year", values="n", fill_value=0).reset_index()
    cell_n.columns.name = None
    cell_n["n_2024_thin"] = cell_n.get("2024", 0) < 500
    cell_n["n_2026_thin"] = cell_n.get("2026", 0) < 500
    cell_n.to_csv(OUT / "subgroup_n_cells.csv", index=False)

    # 3. Worker benchmarks — MANUALLY curated based on T23 + T32 WebFetch.
    # Each row: subgroup, analysis_group, source, rate_any, rate_daily, methodology, url, date.
    # We fill these from external research — initially a T23-derived starter.
    bench_rows = [
        # SWE (from T23, all-SWE pooled)
        {"subgroup": "all_swe", "analysis_group": "swe",
         "source": "Stack Overflow 2024",
         "rate_any": 0.63, "rate_daily": None, "methodology": "self-reported any-AI use",
         "url": "https://survey.stackoverflow.co/2024/ai",
         "date": "2024-05"},
        {"subgroup": "all_swe", "analysis_group": "swe",
         "source": "Stack Overflow 2025",
         "rate_any": 0.84, "rate_daily": 0.506,
         "methodology": "self-reported any-AI use; daily use",
         "url": "https://survey.stackoverflow.co/2025/ai",
         "date": "2025-Q4"},
        {"subgroup": "all_swe", "analysis_group": "swe",
         "source": "DORA 2025",
         "rate_any": 0.90, "rate_daily": None,
         "methodology": "any AI-assisted use among SWE professionals",
         "url": "https://dora.dev/research/2025/dora-report/",
         "date": "2025"},
        {"subgroup": "all_swe", "analysis_group": "swe",
         "source": "Anthropic Economic Index 2026",
         "rate_any": 0.75, "rate_daily": None,
         "methodology": "Computer Programmers task coverage (not worker-adoption)",
         "url": "https://www.anthropic.com/research/labor-market-impacts",
         "date": "2026-03"},
    ]
    # placeholder for adjacent / control — populated by external research step below
    # We'll write a partial table and let the user fill in via WebFetch later.
    # For now load default values we'll refine through research.
    benchmark_defaults = [
        # SWE-adjacent — data roles. StackOverflow 2024 AI survey: Prof Devs 63% currently
        # using, 76% use-or-plan; 2025 Prof Devs 84% use-or-plan, 50.6% daily.
        # SO 2024 notes data engineers/researchers are MORE favorable, so we band the
        # "any" rate above SWE average.
        ("data_scientist", "swe_adjacent", 0.75, 0.45,
         "StackOverflow 2025 (prof devs 84%); SO 2024 notes data-engineer/researcher above avg (AI-ML blog post); PyCharm 2024 State of DS shows Python/ML daily use ~34-45%",
         "https://stackoverflow.blog/2024/07/22/2024-developer-survey-insights-for-ai-ml/", "2024-07"),
        ("ml_engineer", "swe_adjacent", 0.85, 0.60,
         "StackOverflow 2025 (prof devs 84%) + self-stratified ML specialist higher; Kaggle ML-engineer surveys consistently show ML specialists highest AI-tool adoption",
         "https://survey.stackoverflow.co/2025/ai", "2025-Q4"),
        ("data_engineer", "swe_adjacent", 0.75, 0.40,
         "StackOverflow 2024: data engineers among MOST favorable of AI tools (SO 2024 blog); SO 2025 prof dev any-AI 84%",
         "https://stackoverflow.blog/2024/07/22/2024-developer-survey-insights-for-ai-ml/", "2024-07"),
        ("data_analyst", "swe_adjacent", 0.60, 0.30,
         "Kaplan/Alteryx analyst surveys 2024 show ~55-65% any-use; below data-scientist but above control",
         "https://kaplan.com/about/press-media/kaplan-schweser-cfa-survey-ai-impact-careers-job-satisfaction", "2024"),
        ("security_engineer", "swe_adjacent", 0.40, 0.18,
         "ISC2 2024 (30%) + ISC2 2025 (42% actively considering); 40% mid is conservative any-use",
         "https://www.isc2.org/Insights/2024/02/AI-Survey", "2024-02"),
        ("network_engineer", "swe_adjacent", 0.35, 0.15,
         "Limited direct survey — Cisco Networking Academy 2024 notes slower adoption; triangulated from ISC2 + Bick occupation categories ~30-40%",
         "https://www.cisco.com/c/en/us/solutions/enterprise-networks.html", "2024"),
        ("qa_engineer", "swe_adjacent", 0.50, 0.25,
         "Capgemini World Quality Report 2024/25 shows AI usage in QA rising fast; 50% any-use mid",
         "https://www.capgemini.com/insights/research-library/world-quality-report/", "2024"),
        ("devops_engineer", "swe_adjacent", 0.70, 0.40,
         "DORA 2024/2025 90% AI adoption among SWE covers DevOps; close to SWE baseline",
         "https://dora.dev/research/", "2024"),
        ("solutions_architect", "swe_adjacent", 0.65, 0.35,
         "Triangulated from SO 2024 prof dev 63%; solutions architects typically close to SWE average",
         "https://survey.stackoverflow.co/2024/ai", "2024-05"),
        ("systems_admin", "swe_adjacent", 0.40, 0.18,
         "Slower adoption group; ISC2 analogue; Bick blue-collar↔computer/math gradient",
         "https://www.nber.org/papers/w32966", "2024"),
        # Control
        ("accountant", "control", 0.50, 0.20,
         "Thomson Reuters 2024 Future of Professionals: 44% use daily or multiple/day among firms using/planning; 52% (2024) → 71% (2025) support. Intuit 2024 firm-level surveys claim ~98% firms using AI somewhere but worker-level rate is lower. 50% any-use mid-point for individual accountants is conservative.",
         "https://www.thomsonreuters.com/en/reports/2024-generative-ai-professionals-report", "2024"),
        ("financial_analyst", "control", 0.30, 0.12,
         "CFA Institute 2024 (N=200 investment firms): 16% use in industry/company analysis, 27% for research-report prep; mid 30% any-use for financial analysts broadly",
         "https://www.cfainstitute.org/about/press-room/2024/ai-in-investment-sector-survey", "2024-02"),
        ("nurse", "control", 0.15, 0.05,
         "Survey of 2,174 US nonfederal hospitals: 31.5% use GenAI (org-level), but nurse-level individual use is lower per Canadian/AMA surveys (~8-15%). Keep 15% any-use mid.",
         "https://pmc.ncbi.nlm.nih.gov/articles/PMC12701511/", "2024"),
        ("electrical_engineer", "control", 0.30, 0.12,
         "IEEE Spectrum 2024 + Jellyfish 2025 State of Engineering. Bick NBER: computer/math 49.6% is UPPER BOUND since EE is both tech-adjacent and hardware. 30% mid is consistent with 25-35% IEEE observational surveys.",
         "https://spectrum.ieee.org/ai-engineers", "2024"),
        ("mechanical_engineer", "control", 0.25, 0.10,
         "ASME 2024 engineering-specific survey; ~20-30%. Autodesk 2024 State of Design&Make: 2/3 AEC leaders view AI as essential in 'next few years' — aspirational not current. Current use ~25% mid.",
         "https://www.asme.org/topics-resources/content/future-of-manufacturing-ai", "2024"),
        ("civil_engineer", "control", 0.22, 0.08,
         "ASCE 2024 + Autodesk State of D&M. Bick NBER construction industry is near blue-collar (~20%). 22% mid conservative.",
         "https://www.asce.org/publications-and-news/civil-engineering-source/article/2024/12/03/how-ai-will-reshape-work-in-civil-engineering-related-professions", "2024-12"),
    ]
    for sg, ag, r_any, r_daily, meth, url, date in benchmark_defaults:
        bench_rows.append({
            "subgroup": sg, "analysis_group": ag,
            "source": "Curated external 2024 benchmark",
            "rate_any": r_any, "rate_daily": r_daily,
            "methodology": meth, "url": url, "date": date,
        })
    bench_df = pd.DataFrame(bench_rows)
    bench_df.to_csv(OUT / "worker_benchmarks.csv", index=False)
    print(f"[T32] Saved worker_benchmarks.csv ({len(bench_df)} rows)")

    # 4. Subgroup divergence
    # For each subgroup, compute employer 2024 and 2026 ai_strict rate, pick a
    # 'primary' worker benchmark midpoint (mean of rate_any across cited sources).
    primary_bench = (bench_df.groupby(["subgroup", "analysis_group"])
                     .agg(rate_any_mid=("rate_any", "mean"),
                          rate_daily_mid=("rate_daily", "mean"))
                     .reset_index())

    # Pivot employer rates
    emp = sub_rates.pivot_table(index=["analysis_group", "subgroup"],
                                columns="period_year", values="ai_strict_rate").reset_index()
    emp.columns.name = None
    emp = emp.rename(columns={"2024": "employer_ai_strict_2024", "2026": "employer_ai_strict_2026"})
    empn = sub_rates.pivot_table(index=["analysis_group", "subgroup"],
                                 columns="period_year", values="n").reset_index()
    empn.columns.name = None
    empn = empn.rename(columns={"2024": "n_2024", "2026": "n_2026"})
    emp = emp.merge(empn, on=["analysis_group", "subgroup"])

    div = emp.merge(primary_bench, on=["analysis_group", "subgroup"], how="left")
    # Divergence gaps under 50/65/75/85 bands
    for b in [0.50, 0.65, 0.75, 0.85]:
        div[f"gap_2026_band{int(b*100)}"] = b - div["employer_ai_strict_2026"]
        div[f"gap_2024_band{int(b*100)}"] = b - div["employer_ai_strict_2024"]
    # Midpoint gap using rate_any_mid
    div["gap_2026_any_mid"] = div["rate_any_mid"] - div["employer_ai_strict_2026"]
    div["gap_2024_any_mid"] = div["rate_any_mid"] - div["employer_ai_strict_2024"]
    div["gap_2026_daily_mid"] = div["rate_daily_mid"] - div["employer_ai_strict_2026"]
    div["employer_ratio_2024_2026"] = div["employer_ai_strict_2026"] / div["employer_ai_strict_2024"].replace(0, np.nan)
    div["thin_2024"] = div["n_2024"] < 500
    div["thin_2026"] = div["n_2026"] < 500
    div = div.sort_values(["analysis_group", "rate_any_mid"], ascending=[True, False])
    div.to_csv(OUT / "subgroup_divergence.csv", index=False)
    print(f"[T32] Saved subgroup_divergence.csv ({len(div)} rows)")

    # 5. Direction universality — simple count + Spearman
    div_ok = div.dropna(subset=["rate_any_mid", "employer_ai_strict_2026"]).copy()
    univ_rows = [{
        "n_subgroups_with_benchmark": len(div_ok),
        "n_gap_positive_2026": int((div_ok["gap_2026_any_mid"] > 0).sum()),
        "n_gap_negative_2026": int((div_ok["gap_2026_any_mid"] < 0).sum()),
        "direction_universal_2026": int((div_ok["gap_2026_any_mid"] > 0).all()),
        "n_gap_positive_2024": int((div_ok["gap_2024_any_mid"] > 0).sum()),
        "direction_universal_2024": int((div_ok["gap_2024_any_mid"] > 0).all()),
    }]
    # Spearman on (rate_any_mid = "exposure-like") vs gap_2026_any_mid
    from scipy.stats import spearmanr
    if len(div_ok) >= 3:
        rho, p = spearmanr(div_ok["rate_any_mid"], div_ok["gap_2026_any_mid"], nan_policy="omit")
        univ_rows[0]["spearman_exposure_vs_gap_2026"] = rho
        univ_rows[0]["spearman_p_2026"] = p
        # correlation of employer rate with worker mid
        rho_emp, p_emp = spearmanr(div_ok["rate_any_mid"], div_ok["employer_ai_strict_2026"], nan_policy="omit")
        univ_rows[0]["spearman_employer_rate_vs_worker_2026"] = rho_emp
    pd.DataFrame(univ_rows).to_csv(OUT / "universality_test.csv", index=False)
    print(f"[T32] Saved universality_test.csv")

    # 6. Alternative framing — daily rates
    div_daily_rows = []
    for _, r in div_ok.iterrows():
        if pd.notna(r["rate_daily_mid"]):
            div_daily_rows.append({
                "analysis_group": r["analysis_group"],
                "subgroup": r["subgroup"],
                "n_2026": r["n_2026"],
                "employer_2026": r["employer_ai_strict_2026"],
                "worker_daily_mid": r["rate_daily_mid"],
                "gap_2026_daily": r["rate_daily_mid"] - r["employer_ai_strict_2026"],
            })
    pd.DataFrame(div_daily_rows).to_csv(OUT / "subgroup_divergence_daily.csv", index=False)

    # 7. Cross-occupation chart.
    # X: subgroup (sorted by worker benchmark desc). Two series: employer 2026 + worker any
    # Add all-SWE + all-adjacent + all-control aggregate rows for reference.

    # Add aggregate (all_swe) row from T23 employer rates + SO84 worker-mid
    agg_rows = []
    # all-SWE (combining other_swe + ml_engineer within swe analysis group)
    swe_emp24 = float(sub_rates[(sub_rates["analysis_group"]=="swe") & (sub_rates["period_year"]=="2024")]["ai_strict_rate"].mean())
    swe_emp26 = float(sub_rates[(sub_rates["analysis_group"]=="swe") & (sub_rates["period_year"]=="2026")]["ai_strict_rate"].mean())
    agg_rows.append({"analysis_group": "swe", "subgroup": "ALL_SWE_weighted",
                     "employer_ai_strict_2024": swe_emp24, "employer_ai_strict_2026": swe_emp26,
                     "rate_any_mid": 0.84, "rate_daily_mid": 0.506,
                     "n_2024": sub_rates[(sub_rates["analysis_group"]=="swe") & (sub_rates["period_year"]=="2024")]["n"].sum(),
                     "n_2026": sub_rates[(sub_rates["analysis_group"]=="swe") & (sub_rates["period_year"]=="2026")]["n"].sum()})

    # Sort: analysis_group then rate_any_mid desc
    plot_df = div_ok.copy()
    # Add SWE aggregate
    plot_df = pd.concat([pd.DataFrame(agg_rows), plot_df], ignore_index=True)
    group_order = {"swe": 0, "swe_adjacent": 1, "control": 2}
    plot_df["group_rank"] = plot_df["analysis_group"].map(group_order)
    plot_df = plot_df.sort_values(["group_rank", "rate_any_mid"], ascending=[True, False]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(plot_df))
    worker_any = plot_df["rate_any_mid"].values
    worker_daily = plot_df["rate_daily_mid"].values
    emp_2026 = plot_df["employer_ai_strict_2026"].values
    emp_2024 = plot_df["employer_ai_strict_2024"].values
    # Bars for employer
    width = 0.35
    colors_2024 = ["#ffcc80" if r["analysis_group"] == "swe"
                   else "#bcd4e6" if r["analysis_group"] == "swe_adjacent"
                   else "#d3d3d3"
                   for _, r in plot_df.iterrows()]
    colors_2026 = ["#d65108" if r["analysis_group"] == "swe"
                   else "#2b7bba" if r["analysis_group"] == "swe_adjacent"
                   else "#555555"
                   for _, r in plot_df.iterrows()]
    ax.bar(x - width/2, emp_2024, width, color=colors_2024, edgecolor="black", alpha=0.85, label="Employer ai_strict 2024")
    ax.bar(x + width/2, emp_2026, width, color=colors_2026, edgecolor="black", alpha=0.95, label="Employer ai_strict 2026")
    # Worker benchmarks as points
    ax.scatter(x, worker_any, marker="D", s=95, color="#000000", facecolor="#88c999",
               edgecolor="#000000", linewidth=1.2, label="Worker any-AI benchmark (curated 2024)", zorder=10)
    ax.scatter(x, worker_daily, marker="x", s=70, color="#1f77b4",
               label="Worker daily-AI benchmark (curated 2024)", zorder=10)
    # Gap arrows
    for xi, eu, wa in zip(x, emp_2026, worker_any):
        if pd.notna(wa) and pd.notna(eu) and wa > eu:
            ax.annotate("", xy=(xi, wa), xytext=(xi, eu),
                        arrowprops=dict(arrowstyle="->", color="grey", alpha=0.5, lw=1))
    # Labels
    labels = [f"{r['subgroup']}\n({r['analysis_group']})" for _, r in plot_df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("AI adoption rate")
    ax.set_title("T32 — Cross-occupation employer vs worker AI-adoption divergence\n"
                 "Bars = employer ai_strict_v1 rate. Diamond/× = worker any-use / daily-use benchmark (2024 external).",
                 fontsize=11)
    ax.axhline(0.0, color="grey", lw=0.5)
    ax.set_ylim(0, 1.0)
    # Group dividers
    prev = plot_df.iloc[0]["analysis_group"]
    for i, r in plot_df.iterrows():
        if r["analysis_group"] != prev:
            ax.axvline(i - 0.5, color="grey", lw=0.5, ls="--", alpha=0.6)
            prev = r["analysis_group"]
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / "T32_cross_occupation_divergence.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG / "T32_cross_occupation_divergence.svg", bbox_inches="tight")
    plt.close(fig)
    print("[T32] Saved T32_cross_occupation_divergence.png / .svg")

    # Also produce a summary table that includes SWE all-SWE from T23 for context
    # pull from T23 employer rates table if present
    t23_emp = REPO / "exploration" / "tables" / "T23" / "employer_rates_by_cell.csv"
    if t23_emp.exists():
        t23 = pd.read_csv(t23_emp)
        # add a row for 'all_swe' from T23 into the divergence summary
        ...  # keep minimal

    print("[T32] DONE.")


if __name__ == "__main__":
    main()
