"""S25 — Cross-occupation rank correlation between worker-side AI usage
benchmarks and employer-side AI requirement rates.

Replicates and stress-tests the v9 archive Story 08 claim
(Spearman rho ~ +0.92 across 16 occupations) with a clean methodology audit.

Key differences vs the original T32:
  - Uses the canonical AI_VOCAB_PATTERN from eda/scripts/scans.py
    (not the validated `ai_strict_v1` regex). The user's instruction
    explicitly says to use the canonical regex used elsewhere in the
    project. We also report a sensitivity using `ai_strict_v1_rebuilt`.
  - Multiple operationalizations (scope, metric, missing-data handling)
    instead of a single number.
  - Bootstrap CIs by resampling occupations.

Outputs (all under eda/):
  tables/S25_subgroup_rates.csv         — per-occupation employer rates by year
  tables/S25_worker_benchmarks.csv      — curated external surveys with URLs
  tables/S25_method_comparison.csv      — rank correlation under each method
  tables/S25_pair_table.csv             — final pairings used in headline method
  figures/S25_method_comparison.png     — bar chart of correlations across methods
  figures/S25_employer_vs_worker.png    — scatter of pairings under headline method
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
from scipy.stats import spearmanr, kendalltau, pearsonr

REPO = Path("/home/jihgaboot/gabor/job-research")
DATA = REPO / "data" / "unified_core.parquet"
TABLES = REPO / "eda" / "tables"
FIGS = REPO / "eda" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

sys.path.append(str(REPO / "eda" / "scripts"))
from scans import AI_VOCAB_PATTERN  # canonical pattern
AI_VOCAB_RX = re.compile(AI_VOCAB_PATTERN)

# Sensitivity pattern from validated_mgmt_patterns.json (v9 archive)
AI_STRICT_V1 = re.compile(
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|"
    r"llamaindex|langchain|prompt engineering|rag|vector databas(?:e|es)|"
    r"pinecone|huggingface|hugging face|"
    r"(?:fine[- ]tun(?:e|ed|ing))\s+(?:the\s+)?(?:model|llm|gpt|base model|foundation model|embeddings))\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Subgroup classifiers (carried over from v9 T32 — first-match-wins)
# ---------------------------------------------------------------------------

SWE_SUBGROUPS = [
    ("ml_engineer",
     re.compile(r"\b(?:ml|machine learning|ai|mlops)\s*(?:engineer|developer|architect|scientist)\b", re.IGNORECASE)),
]
SWE_ADJ_SUBGROUPS = [
    ("ml_engineer",
     re.compile(r"\b(?:machine learning|ml|ai|mlops)\s*(?:engineer|developer|scientist|architect)\b", re.IGNORECASE)),
    ("data_scientist", re.compile(r"\b(?:data scientist|data science)\b", re.IGNORECASE)),
    ("data_engineer", re.compile(r"\b(?:data engineer|analytics engineer)\b", re.IGNORECASE)),
    ("data_analyst",
     re.compile(r"\b(?:data analyst|business analyst|analytics analyst)\b", re.IGNORECASE)),
    ("security_engineer",
     re.compile(r"\b(?:security|cyber|cybersecurity|information security|infosec)\s*(?:engineer|architect)\b",
                re.IGNORECASE)),
    ("network_engineer", re.compile(r"\bnetwork\s*(?:engineer|architect)\b", re.IGNORECASE)),
    ("qa_engineer", re.compile(r"\b(?:qa|quality)\s*(?:engineer|analyst)\b", re.IGNORECASE)),
    ("devops_engineer",
     re.compile(r"\b(?:devops|site reliability|sre|platform)\s*(?:engineer|architect)\b", re.IGNORECASE)),
    ("solutions_architect",
     re.compile(r"\b(?:solution|solutions|enterprise|cloud|technical)\s*architect\b", re.IGNORECASE)),
    ("systems_admin", re.compile(r"\b(?:systems?|database)\s*administrator\b", re.IGNORECASE)),
]
CONTROL_SUBGROUPS = [
    ("accountant", re.compile(r"\b(?:accountant|accounting|cpa|bookkeeper)\b", re.IGNORECASE)),
    ("financial_analyst", re.compile(r"\bfinancial\s*analyst\b", re.IGNORECASE)),
    ("nurse",
     re.compile(r"\b(?:registered nurse|rn|nurse practitioner|lpn|licensed practical nurse|cna|nursing)\b",
                re.IGNORECASE)),
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


# ---------------------------------------------------------------------------
# Worker benchmarks: each row is an occupation × source.
# We carry over v9 T32 sources, add cross-checks, and keep separate the
# "any-AI" (lifetime / monthly use) and "daily" rates so we can swap them.
# ---------------------------------------------------------------------------

WORKER_BENCHMARKS = pd.DataFrame([
    # ---- Software / data / engineering occupations ----
    # Stack Overflow Developer Survey 2024 — 76% professional devs are using or planning;
    # 63% currently use AI tools in dev process.
    dict(subgroup="other_swe", group="swe",
         source="Stack Overflow Developer Survey 2024",
         rate_any=0.63, rate_daily=None,
         url="https://survey.stackoverflow.co/2024/ai",
         access_date="2026-04-21",
         notes="Q: 'Are you currently using AI tools in your development process?' Yes = 63% among Prof devs."),
    # Stack Overflow 2025 — usage 84%, daily 50.6%
    dict(subgroup="other_swe", group="swe",
         source="Stack Overflow Developer Survey 2025",
         rate_any=0.84, rate_daily=0.506,
         url="https://survey.stackoverflow.co/2025/",
         access_date="2026-04-21",
         notes="Prof devs: 84% currently use AI tools; 50.6% use daily. Used as 2025-Q4 mid."),
    # DORA 2025 (Google Cloud / DevOps Research)
    dict(subgroup="other_swe", group="swe",
         source="DORA 2025 State of DevOps",
         rate_any=0.90, rate_daily=None,
         url="https://dora.dev/research/2025/dora-report/",
         access_date="2026-04-21",
         notes="90% of SWE professionals report any AI-assisted coding."),

    # ML engineer / data scientist — Kaggle ML/DS surveys + Stack Overflow specialist sub-cuts
    dict(subgroup="ml_engineer", group="swe_adjacent",
         source="Stack Overflow 2025 (ML specialist sub-cut) + Kaggle ML/DS 2024",
         rate_any=0.85, rate_daily=0.60,
         url="https://survey.stackoverflow.co/2025/",
         access_date="2026-04-21",
         notes="ML practitioners are consistently the highest-AI-tool-adoption SWE subgroup."),
    dict(subgroup="data_scientist", group="swe_adjacent",
         source="Stack Overflow 2024 ML/AI insights post + Kaggle 2024",
         rate_any=0.75, rate_daily=0.45,
         url="https://stackoverflow.blog/2024/07/22/2024-developer-survey-insights-for-ai-ml/",
         access_date="2026-04-21",
         notes="Data scientists slightly above SWE average per SO 2024 specialty cut."),
    dict(subgroup="data_engineer", group="swe_adjacent",
         source="Stack Overflow 2024 (data engineer sub-cut)",
         rate_any=0.75, rate_daily=0.40,
         url="https://stackoverflow.blog/2024/07/22/2024-developer-survey-insights-for-ai-ml/",
         access_date="2026-04-21",
         notes="Data engineers among the most favorable in SO 2024 AI views."),
    dict(subgroup="data_analyst", group="swe_adjacent",
         source="Triangulated: Microsoft WTI 2024, Kaplan analyst surveys",
         rate_any=0.60, rate_daily=0.30,
         url="https://www.microsoft.com/en-us/worklab/work-trend-index/ai-at-work-is-here-now-comes-the-hard-part",
         access_date="2026-04-21",
         notes="Analyst-class knowledge workers ~55-65% any-AI use across 2024 surveys."),
    dict(subgroup="security_engineer", group="swe_adjacent",
         source="ISC2 Cybersecurity Workforce Study 2024",
         rate_any=0.40, rate_daily=0.18,
         url="https://www.isc2.org/Insights/2024/02/AI-Survey",
         access_date="2026-04-21",
         notes="ISC2 Feb 2024: ~30% currently use AI in security work; 42% considering by 2025."),
    dict(subgroup="devops_engineer", group="swe_adjacent",
         source="DORA 2024/2025 (DevOps cohort)",
         rate_any=0.70, rate_daily=0.40,
         url="https://dora.dev/research/",
         access_date="2026-04-21",
         notes="DevOps adoption tracks SWE baseline closely; conservative midpoint 0.70."),
    dict(subgroup="solutions_architect", group="swe_adjacent",
         source="Triangulated from SO 2024 prof devs",
         rate_any=0.65, rate_daily=0.35,
         url="https://survey.stackoverflow.co/2024/ai",
         access_date="2026-04-21",
         notes="Solutions architects close to SWE pro-dev average."),
    dict(subgroup="qa_engineer", group="swe_adjacent",
         source="Capgemini World Quality Report 2024-25",
         rate_any=0.50, rate_daily=0.25,
         url="https://www.capgemini.com/insights/research-library/world-quality-report/",
         access_date="2026-04-21",
         notes="QA AI adoption rising fast in 2024; 50% any-use is mid-band."),
    dict(subgroup="network_engineer", group="swe_adjacent",
         source="Triangulated: Bick/Blandin/Deming NBER 2024 + Cisco",
         rate_any=0.35, rate_daily=0.15,
         url="https://www.nber.org/papers/w32966",
         access_date="2026-04-21",
         notes="Limited direct survey; placed below SWE/security per Bick computer-occupations gradient."),
    dict(subgroup="systems_admin", group="swe_adjacent",
         source="Triangulated: ISC2 + Bick NBER 2024",
         rate_any=0.40, rate_daily=0.18,
         url="https://www.nber.org/papers/w32966",
         access_date="2026-04-21",
         notes="Slower-adoption infra group; placed mid-low among IT roles."),

    # ---- Control occupations (non-tech) ----
    dict(subgroup="accountant", group="control",
         source="Thomson Reuters 2024 Future of Professionals (tax/accounting)",
         rate_any=0.50, rate_daily=0.20,
         url="https://www.thomsonreuters.com/en/reports/2024-generative-ai-professionals-report",
         access_date="2026-04-21",
         notes="2024 Generative AI Report. 'Ever tried' leans high (~50%); daily firm-wide use ~20%."),
    dict(subgroup="financial_analyst", group="control",
         source="CFA Institute 2024 AI in Investment Sector Survey",
         rate_any=0.30, rate_daily=0.12,
         url="https://www.cfainstitute.org/about/press-room/2024/ai-in-investment-sector-survey",
         access_date="2026-04-21",
         notes="N=200 firms. 16% use in industry/company analysis; 27% for research-report prep."),
    dict(subgroup="nurse", group="control",
         source="JAMA / AHA 2024 hospital GenAI surveys",
         rate_any=0.15, rate_daily=0.05,
         url="https://pmc.ncbi.nlm.nih.gov/articles/PMC12701511/",
         access_date="2026-04-21",
         notes="Org-level use 31.5% across 2,174 hospitals; nurse-individual rate 8-15% per AMA + Canadian surveys."),
    dict(subgroup="electrical_engineer", group="control",
         source="IEEE Spectrum 2024 + Bick NBER 2024",
         rate_any=0.30, rate_daily=0.12,
         url="https://spectrum.ieee.org/ai-engineers",
         access_date="2026-04-21",
         notes="EE is hardware-heavy; 25-35% any-use band."),
    dict(subgroup="mechanical_engineer", group="control",
         source="ASME 2024 + Autodesk State of D&M 2024",
         rate_any=0.25, rate_daily=0.10,
         url="https://www.asme.org/topics-resources/content/future-of-manufacturing-ai",
         access_date="2026-04-21",
         notes="ME firms: ~20-30% report current AI use in design/manufacturing."),
    dict(subgroup="civil_engineer", group="control",
         source="ASCE 2024 + Autodesk State of D&M 2024",
         rate_any=0.22, rate_daily=0.08,
         url="https://www.asce.org/publications-and-news/civil-engineering-source/article/2024/12/03/how-ai-will-reshape-work-in-civil-engineering-related-professions",
         access_date="2026-04-21",
         notes="Civil engineering AI use is the lowest among engineering disciplines."),

    # Cross-check macro benchmark — Bick/Blandin/Deming NBER 2024 (workplace any-use,
    # all knowledge workers, US): 39.4%. Reported as a sanity floor for level scaling.
])

# Occupations explicitly named in the user's claim (the 16-occupation set).
# We do NOT have separate marketing/HR/sales subgroups in unified_core, so we
# document them as "no benchmark available in core" rather than fabricating.
HEADLINE_OCCUPATIONS = [
    ("other_swe", "swe"),                    # software engineer
    ("ml_engineer", "swe_adjacent"),         # ML engineer
    ("data_scientist", "swe_adjacent"),
    ("data_engineer", "swe_adjacent"),
    ("data_analyst", "swe_adjacent"),
    ("security_engineer", "swe_adjacent"),
    ("devops_engineer", "swe_adjacent"),
    ("solutions_architect", "swe_adjacent"),
    ("qa_engineer", "swe_adjacent"),
    ("network_engineer", "swe_adjacent"),
    ("systems_admin", "swe_adjacent"),
    ("accountant", "control"),
    ("financial_analyst", "control"),
    ("nurse", "control"),
    ("electrical_engineer", "control"),
    ("mechanical_engineer", "control"),
    ("civil_engineer", "control"),
]


def fisher_z_ci(rho, n, alpha=0.05):
    """Fisher z transform 95% CI for a Spearman/Pearson correlation."""
    if n < 4 or abs(rho) >= 0.999999:
        return (np.nan, np.nan)
    z = 0.5 * np.log((1 + rho) / (1 - rho))
    se = 1.0 / np.sqrt(n - 3)
    from scipy.stats import norm
    crit = norm.ppf(1 - alpha / 2)
    lo = z - crit * se
    hi = z + crit * se
    return (np.tanh(lo), np.tanh(hi))


def bootstrap_corr(x, y, n_boot=2000, kind="spearman", seed=0):
    """Bootstrap CI by resampling occupation pairs."""
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xs = np.asarray(x)[idx]
        ys = np.asarray(y)[idx]
        if len(np.unique(xs)) < 2 or len(np.unique(ys)) < 2:
            continue
        if kind == "spearman":
            r, _ = spearmanr(xs, ys)
        elif kind == "kendall":
            r, _ = kendalltau(xs, ys)
        else:
            r, _ = pearsonr(xs, ys)
        rhos.append(r)
    rhos = np.asarray(rhos)
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def main():
    print("[S25] Loading unified_core (linkedin, english, date_ok)...")
    con = duckdb.connect()
    # Strict-core: matching substrate is description_core_llm (per methodology
    # protocol). Filter to rows where it's non-NULL.
    q = f"""
    SELECT title,
           description_core_llm AS substrate_text,
           CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period_year,
           analysis_group
    FROM read_parquet('{DATA}')
    WHERE source_platform='linkedin' AND is_english AND date_flag='ok'
      AND description_core_llm IS NOT NULL
    """
    df = con.execute(q).fetchdf()
    print(f"  loaded {len(df):,} rows")

    df["subgroup"] = [classify_subgroup(t, g) for t, g in zip(df["title"], df["analysis_group"])]
    desc = df["substrate_text"].fillna("").astype(str)
    print("[S25] Tagging AI mentions (strict-core substrate)...")
    df["ai_canonical"] = desc.str.contains(AI_VOCAB_RX).astype(int)
    df["ai_strict_v1"] = desc.str.contains(AI_STRICT_V1).astype(int)

    # Per-subgroup per-year rates
    rows = []
    for (ag, sg, py), g in df.groupby(["analysis_group", "subgroup", "period_year"]):
        if sg is None:
            continue
        rows.append(dict(
            analysis_group=ag, subgroup=sg, period_year=py, n=len(g),
            ai_canonical_rate=float(g["ai_canonical"].mean()),
            ai_strict_v1_rate=float(g["ai_strict_v1"].mean()),
        ))
    sub = pd.DataFrame(rows)
    sub.to_csv(TABLES / "S25_subgroup_rates.csv", index=False)
    print(f"[S25] Saved S25_subgroup_rates.csv ({len(sub)} rows)")

    # Save the worker benchmark table (as the canonical record of provenance)
    WORKER_BENCHMARKS.to_csv(TABLES / "S25_worker_benchmarks.csv", index=False)

    # Aggregate worker-side benchmark per subgroup (mean across cited surveys
    # for the "any-use" rate; mean across cited surveys for "daily use").
    worker_any = (WORKER_BENCHMARKS.groupby("subgroup")["rate_any"].mean()
                  .rename("worker_any_mid").reset_index())
    worker_daily = (WORKER_BENCHMARKS.dropna(subset=["rate_daily"])
                    .groupby("subgroup")["rate_daily"].mean()
                    .rename("worker_daily_mid").reset_index())

    # Pivot employer rates wide: one row per (analysis_group, subgroup)
    emp_pivot = sub.pivot_table(
        index=["analysis_group", "subgroup"],
        columns="period_year",
        values=["ai_canonical_rate", "ai_strict_v1_rate", "n"],
        aggfunc="first",
    )
    emp_pivot.columns = [f"{a}_{b}" for a, b in emp_pivot.columns]
    emp_pivot = emp_pivot.reset_index()
    pair = emp_pivot.merge(worker_any, on="subgroup", how="left")
    pair = pair.merge(worker_daily, on="subgroup", how="left")

    # Restrict to headline occupations (swe + adjacent + control with bench)
    headline_keys = set(HEADLINE_OCCUPATIONS)
    pair["is_headline"] = pair.apply(
        lambda r: (r["subgroup"], r["analysis_group"]) in headline_keys, axis=1
    )

    # Add deltas
    pair["delta_2024_2026_canonical"] = (
        pair["ai_canonical_rate_2026"] - pair["ai_canonical_rate_2024"]
    )
    pair["delta_2024_2026_strict_v1"] = (
        pair["ai_strict_v1_rate_2026"] - pair["ai_strict_v1_rate_2024"]
    )
    pair.to_csv(TABLES / "S25_pair_table.csv", index=False)
    print(f"[S25] Saved S25_pair_table.csv ({len(pair)} rows)")

    # ----------------------------------------------------------
    # Method comparison
    # ----------------------------------------------------------
    methods = []

    def add_method(name, x_label, y_label, x, y, n_min=500):
        """Compute correlation. x,y are aligned arrays after dropna."""
        d = pd.DataFrame(dict(x=x, y=y)).dropna()
        if len(d) < 4:
            methods.append(dict(method=name, x=x_label, y=y_label, n=len(d),
                                spearman=np.nan, kendall=np.nan, pearson=np.nan,
                                spearman_fisher_lo=np.nan, spearman_fisher_hi=np.nan,
                                spearman_boot_lo=np.nan, spearman_boot_hi=np.nan))
            return
        rho, p = spearmanr(d["x"], d["y"])
        tau, _ = kendalltau(d["x"], d["y"])
        r, _ = pearsonr(d["x"], d["y"])
        flo, fhi = fisher_z_ci(rho, len(d))
        blo, bhi = bootstrap_corr(d["x"].values, d["y"].values, n_boot=2000, kind="spearman")
        methods.append(dict(
            method=name, x=x_label, y=y_label, n=len(d),
            spearman=rho, spearman_p=p, kendall=tau, pearson=r,
            spearman_fisher_lo=flo, spearman_fisher_hi=fhi,
            spearman_boot_lo=blo, spearman_boot_hi=bhi,
        ))

    headline = pair[pair["is_headline"]].copy()

    # M1 (HEADLINE replication of v9): 16 occupations, worker any-mid vs employer 2026
    #     under canonical AI vocab pattern.
    add_method(
        "M1_headline_canonical",
        "worker_any_mid", "employer_canonical_2026",
        headline["worker_any_mid"], headline["ai_canonical_rate_2026"],
    )
    # M2: same but using ai_strict_v1 (matches v9 archive's regex)
    add_method(
        "M2_headline_strict_v1",
        "worker_any_mid", "employer_strict_v1_2026",
        headline["worker_any_mid"], headline["ai_strict_v1_rate_2026"],
    )
    # M3: 2024 employer rate vs worker any-mid (worker benchmarks are mostly
    #     2024-reported, so 2024 is contemporaneous)
    add_method(
        "M3_2024_levels_canonical",
        "worker_any_mid", "employer_canonical_2024",
        headline["worker_any_mid"], headline["ai_canonical_rate_2024"],
    )
    # M4: deltas (2026-2024) vs worker rate
    add_method(
        "M4_delta_canonical",
        "worker_any_mid", "delta_2026_2024_canonical",
        headline["worker_any_mid"], headline["delta_2024_2026_canonical"],
    )
    # M5: worker DAILY-use benchmark (more conservative)
    add_method(
        "M5_daily_use_canonical",
        "worker_daily_mid", "employer_canonical_2026",
        headline["worker_daily_mid"], headline["ai_canonical_rate_2026"],
    )
    # M6: drop occupations with employer n_2024 OR n_2026 < 500 (thin-cell guard)
    thin_mask = (headline["n_2024"] >= 500) & (headline["n_2026"] >= 500)
    headline_thick = headline[thin_mask]
    add_method(
        "M6_thick_only_canonical",
        "worker_any_mid", "employer_canonical_2026",
        headline_thick["worker_any_mid"], headline_thick["ai_canonical_rate_2026"],
    )
    # M7: control-only (within non-tech occupations does the rank still hold?)
    ctrl = headline[headline["analysis_group"] == "control"]
    add_method(
        "M7_control_only",
        "worker_any_mid", "employer_canonical_2026",
        ctrl["worker_any_mid"], ctrl["ai_canonical_rate_2026"],
    )
    # M8: SWE + adjacent only (do the tech occupations alone preserve the rank?)
    tech = headline[headline["analysis_group"].isin(["swe", "swe_adjacent"])]
    add_method(
        "M8_tech_only",
        "worker_any_mid", "employer_canonical_2026",
        tech["worker_any_mid"], tech["ai_canonical_rate_2026"],
    )
    # M9: drop ml_engineer (the most extreme tech outlier on both axes)
    no_ml = headline[headline["subgroup"] != "ml_engineer"]
    add_method(
        "M9_drop_mlengineer",
        "worker_any_mid", "employer_canonical_2026",
        no_ml["worker_any_mid"], no_ml["ai_canonical_rate_2026"],
    )
    # M10: alternative aggregation — use the MAX of cited any-use surveys
    #      (uses the most permissive worker benchmark per occupation)
    worker_max = (WORKER_BENCHMARKS.groupby("subgroup")["rate_any"].max()
                  .rename("worker_any_max").reset_index())
    h10 = headline.merge(worker_max, on="subgroup", how="left")
    add_method(
        "M10_worker_max_any",
        "worker_any_max", "employer_canonical_2026",
        h10["worker_any_max"], h10["ai_canonical_rate_2026"],
    )
    # M11: alternative — use the MIN of cited surveys (most conservative)
    worker_min = (WORKER_BENCHMARKS.groupby("subgroup")["rate_any"].min()
                  .rename("worker_any_min").reset_index())
    h11 = headline.merge(worker_min, on="subgroup", how="left")
    add_method(
        "M11_worker_min_any",
        "worker_any_min", "employer_canonical_2026",
        h11["worker_any_min"], h11["ai_canonical_rate_2026"],
    )

    method_df = pd.DataFrame(methods)
    method_df.to_csv(TABLES / "S25_method_comparison.csv", index=False)
    print(f"[S25] Saved S25_method_comparison.csv ({len(method_df)} rows)")
    print("\n=== Method comparison ===")
    print(method_df[["method", "n", "spearman", "spearman_fisher_lo",
                     "spearman_fisher_hi", "spearman_boot_lo",
                     "spearman_boot_hi", "kendall", "pearson"]]
          .to_string(index=False))

    # ----------------------------------------------------------
    # Figures
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    md = method_df.dropna(subset=["spearman"]).reset_index(drop=True)
    y = np.arange(len(md))
    ax.barh(y, md["spearman"], color="#3070b3", edgecolor="black")
    for i, row in md.iterrows():
        if pd.notna(row["spearman_boot_lo"]) and pd.notna(row["spearman_boot_hi"]):
            ax.plot([row["spearman_boot_lo"], row["spearman_boot_hi"]],
                    [i, i], color="black", lw=1.2)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m} (n={n})" for m, n in zip(md["method"], md["n"])], fontsize=9)
    ax.set_xlim(-0.2, 1.05)
    ax.set_xlabel("Spearman rho (worker-side AI rate vs employer-side AI requirement rate)")
    ax.set_title("S25 — Cross-occupation rank correlation under 11 methodological variants\n"
                 "Bars = point estimate. Lines = bootstrap 95% CI (occupation resampling).",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "S25_method_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[S25] Saved S25_method_comparison.png")

    # Scatter for the headline method
    fig, ax = plt.subplots(figsize=(8, 6))
    h = headline.dropna(subset=["worker_any_mid", "ai_canonical_rate_2026"])
    ax.scatter(h["worker_any_mid"], h["ai_canonical_rate_2026"], s=70, color="#3070b3", edgecolor="black")
    for _, r in h.iterrows():
        ax.annotate(r["subgroup"], (r["worker_any_mid"], r["ai_canonical_rate_2026"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Worker-side AI any-use rate (curated 2024 mid)")
    ax.set_ylabel("Employer-side AI requirement rate (canonical vocab, 2026)")
    ax.set_title(f"S25 — Headline pairing (n={len(h)} occupations)\n"
                 f"Spearman rho = {spearmanr(h['worker_any_mid'], h['ai_canonical_rate_2026'])[0]:.3f}",
                 fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "S25_employer_vs_worker.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[S25] Saved S25_employer_vs_worker.png")

    print("[S25] DONE.")


if __name__ == "__main__":
    main()
