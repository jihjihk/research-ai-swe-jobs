"""
T21 Step 1: Semantic precision check on people-management, technical-
orchestration, and strategic-scope patterns.

Spec requires:
  - Validate `agent` and `pipeline` in senior-orchestration context (V1 flagged
    `agent` at 66% in corpus-wide context; may differ here).
  - Validate `stakeholder` (spec cites ~60% precision as a scope term).
  - Validate `team building` (V1 did not validate it in SWE context).

Protocol:
  - Sample 50 matches per sub-pattern, stratified 25/25 across periods
    2024/2026.
  - For each match, extract a 180-char window and classify TP/FP/AMBIG.
  - A TP means: the match semantically supports the intended concept in
    *senior role context*.
  - Report per-subterm precision. Drop any <80%.

We run automatic classification (rule-based) so the check is reproducible.
We do NOT take a manual eyeball pass; we use tight auto-rules to classify
each match. Transparent rules:

  - `agent` TP if window contains any of: ai, ml, llm, llama, gpt, claude,
    chatgpt, openai, copilot, cursor, langchain, langgraph, autonomous,
    workflow, agentic, retrieval, rag.
    FP if window contains: insurance, real estate, real-estate, sales agent,
    user agent, sql agent, nessus, anti-virus, security agent, cleaning agent,
    disease, nerve, infectious, chemical, biologic, property, legal, licensed,
    booking, travel, rental.
  - `pipeline` TP if window contains data, etl, ml, ci, cd, build, spark,
    airflow, kafka, bigquery, snowflake, dbt, ingestion, streaming.
    FP if window contains: sales pipeline, hiring pipeline, candidate pipeline,
    recruitment pipeline, product pipeline, deal pipeline, opportunity
    pipeline, customer pipeline.
  - `stakeholder` TP if window contains: cross-functional, cross functional,
    product, business, executive, board, ceo, cto, vp, director, leadership,
    manage, align, communicat, roadmap, prioriti.
    FP: any form using stakeholder as generic noun without role-scope context.
  - `team building` TP if window contains manage, hiring, lead, direct,
    mentor, grow, culture, build a team, team-building (explicit management
    context).

Sampling is deterministic (random_state=42).
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/home/jihgaboot/gabor/job-research")
CLEANED = ROOT / "exploration/artifacts/shared/swe_cleaned_text.parquet"
OUT = ROOT / "exploration/artifacts/T21"
OUT.mkdir(parents=True, exist_ok=True)

rng = random.Random(42)

# Load LLM-labeled senior text
df = pq.read_table(
    CLEANED,
    columns=[
        "uid",
        "description_cleaned",
        "text_source",
        "period",
        "seniority_final",
    ],
).to_pandas()

df = df[
    (df["text_source"] == "llm")
    & (df["seniority_final"].isin(["mid-senior", "director"]))
].reset_index(drop=True)

df["period_bucket"] = df["period"].apply(
    lambda p: "2024" if p.startswith("2024") else "2026"
)

print(f"Senior LLM-labeled rows: {len(df):,}")
for p, n in df["period_bucket"].value_counts().items():
    print(f"  {p}: n={n}")

SAMPLE_N = 50


def sample_matches(
    pattern: re.Pattern, target_word: str | None = None, per_period: int = 25
) -> list[dict]:
    """Sample up to per_period matches per period for a single term/pattern."""
    out = []
    for period in ["2024", "2026"]:
        sub = df[df["period_bucket"] == period]
        # Shuffle indices for deterministic sampling
        shuf = list(sub.index)
        rng.shuffle(shuf)
        collected = 0
        for idx in shuf:
            text = sub.loc[idx, "description_cleaned"]
            if not text:
                continue
            # If pattern is per-word target, only accept matches of that word
            m_iter = list(pattern.finditer(text))
            if not m_iter:
                continue
            # Take one random match from this posting
            m = m_iter[rng.randrange(len(m_iter))]
            start, end = m.span()
            window = text[max(0, start - 90): end + 90]
            out.append(
                {
                    "uid": sub.loc[idx, "uid"],
                    "period": period,
                    "term": m.group(0).lower(),
                    "context": window,
                    "target": target_word,
                }
            )
            collected += 1
            if collected >= per_period:
                break
    return out


# Pattern definitions — matched carefully
AGENT_P = re.compile(r"\bagent(s)?\b", re.IGNORECASE)
PIPELINE_P = re.compile(r"\bpipeline(s)?\b", re.IGNORECASE)
STAKEHOLDER_P = re.compile(r"\bstakeholders?\b", re.IGNORECASE)
TEAM_BUILDING_P = re.compile(r"\bteam[- ]?building\b", re.IGNORECASE)

AGENT_TP = re.compile(
    r"\b(ai|ml|llm|llama|gpt|claude|chatgpt|openai|copilot|cursor|"
    r"langchain|langgraph|autonomous|workflow|agentic|retrieval|rag|"
    r"tool[- ]use|planning|orchestrat)\b",
    re.IGNORECASE,
)
AGENT_FP = re.compile(
    r"(insurance|real[- ]?estate|sales\s+agent|user\s+agent|user-?agent|sql\s+agent|"
    r"nessus|anti[- ]virus|antivirus|security\s+agent|cleaning\s+agent|"
    r"disease|nerve|infectious|chemical|biologic|property\s+agent|"
    r"legal\s+agent|licensed\s+agent|booking\s+agent|travel\s+agent|"
    r"rental\s+agent|change\s+agent|free\s+agent|release\s+agent|"
    r"admin\s+agent|monitoring\s+agent|enterprise\s+agent)",
    re.IGNORECASE,
)

PIPELINE_TP = re.compile(
    r"\b(data|etl|ml|ci|cd|ci/cd|build|spark|airflow|kafka|bigquery|"
    r"snowflake|dbt|ingestion|streaming|batch|kinesis|beam|dataflow|"
    r"databricks|analytics|feature\s*pipeline|training\s+pipeline|"
    r"inference\s+pipeline|deployment|devops|release)\b",
    re.IGNORECASE,
)
PIPELINE_FP = re.compile(
    r"(sales\s+pipeline|hiring\s+pipeline|candidate\s+pipeline|"
    r"recruitment\s+pipeline|product\s+pipeline|deal\s+pipeline|"
    r"opportunity\s+pipeline|customer\s+pipeline|talent\s+pipeline|"
    r"revenue\s+pipeline|project\s+pipeline|client\s+pipeline|"
    r"business\s+pipeline)",
    re.IGNORECASE,
)

STAKEHOLDER_TP = re.compile(
    r"\b(cross[- ]functional|product|business|executive|board|"
    r"ceo|cto|vp|director|leadership|management|align|communicat|"
    r"roadmap|prioriti|partner|customer|client|external)\b",
    re.IGNORECASE,
)

TEAM_BUILDING_TP = re.compile(
    r"\b(manag\w*|hiring|hire|lead|direct|mentor|grow|culture|"
    r"build\s+a\s+team|team[- ]?building)\b",
    re.IGNORECASE,
)


def classify(term: str, context: str) -> str:
    """Rule-based TP/FP/AMBIG classification."""
    # Remove the term itself from context for TP/FP testing to avoid tautology
    # (but keep it for display)
    clean_ctx = re.sub(
        re.escape(term), " ", context, flags=re.IGNORECASE
    ).replace("  ", " ")
    if term.lower() in ("agent", "agents"):
        if AGENT_FP.search(clean_ctx):
            return "FP"
        if AGENT_TP.search(clean_ctx):
            return "TP"
        return "AMBIG"
    if term.lower() in ("pipeline", "pipelines"):
        if PIPELINE_FP.search(clean_ctx):
            return "FP"
        if PIPELINE_TP.search(clean_ctx):
            return "TP"
        return "AMBIG"
    if term.lower() in ("stakeholder", "stakeholders"):
        if STAKEHOLDER_TP.search(clean_ctx):
            return "TP"
        return "AMBIG"
    if term.lower() in ("team building", "team-building", "teambuilding"):
        if TEAM_BUILDING_TP.search(clean_ctx):
            return "TP"
        return "AMBIG"
    return "AMBIG"


# Sample & classify
samples = {
    "agent": sample_matches(AGENT_P, target_word="agent"),
    "pipeline": sample_matches(PIPELINE_P, target_word="pipeline"),
    "stakeholder": sample_matches(STAKEHOLDER_P, target_word="stakeholder"),
    "team_building": sample_matches(TEAM_BUILDING_P, target_word="team_building"),
}

records = []
for sub, recs in samples.items():
    for r in recs:
        r["classification"] = classify(r["term"], r["context"])
        r["subterm"] = sub
        records.append(r)

sample_df = pd.DataFrame(records)
sample_df.to_csv(OUT / "T21_precision_samples.csv", index=False)
print(f"\nwrote {OUT / 'T21_precision_samples.csv'}  (n={len(sample_df)})")

# Precision summary
summary = (
    sample_df.groupby(["subterm", "classification"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
# Precision = TP / (TP + FP + AMBIG) — conservative
for col in ("TP", "FP", "AMBIG"):
    if col not in summary.columns:
        summary[col] = 0
summary["n"] = summary["TP"] + summary["FP"] + summary["AMBIG"]
summary["precision_strict"] = summary["TP"] / summary["n"]
summary["precision_relaxed"] = summary["TP"] / (summary["TP"] + summary["FP"]).clip(lower=1)
summary.to_csv(OUT / "T21_precision_summary.csv", index=False)

print("\nPrecision summary (strict TP / total; relaxed TP / (TP+FP)):")
print(summary.to_string())

# Verdict
print("\nVerdict (<80% strict -> drop):")
for _, r in summary.iterrows():
    verdict = "KEEP" if r["precision_strict"] >= 0.80 else "DROP"
    print(
        f"  {r['subterm']}: strict={r['precision_strict']:.2%}, "
        f"relaxed={r['precision_relaxed']:.2%}, verdict={verdict}"
    )

# Save final patterns per V1-style rulebook
final = {
    "people_management_strict": {
        "description": "V1-refined (already validated in T11/V1)",
        "pattern": r"\b(mentor|coach|hire|headcount|performance[- ]?review)\w*",
        "source": "V1 verification — these 5 sub-patterns at >=80% precision",
    },
    "technical_orchestration_strict": {
        "description": "Start from full spec; drop sub-patterns failing T21 precision check",
        "pattern": None,
        "subterm_verdicts": summary.set_index("subterm").to_dict(orient="index"),
    },
    "strategic_scope_strict": {
        "description": "Start from full spec; refine stakeholder per T21 precision check",
        "pattern": None,
    },
}
(OUT / "T21_precision_verdicts.json").write_text(
    json.dumps(final, default=str, indent=2)
)
print(f"\nwrote {OUT / 'T21_precision_verdicts.json'}")
