"""
T21 Step 2: Per-posting density of people-management, technical-orchestration,
strategic-scope profiles for senior postings (mid-senior + director).

Patterns AFTER T21 precision check:
  - People management (strict): mentor|coach|hire|headcount|performance review
  - Technical orchestration (strict): architecture review|code review|system design|
    technical direction|ai orchestration|workflow|pipeline|automation|evaluate|
    validate|quality gate|guardrails|prompt engineering|tool selection
    (DROPPED `agent` per T21 precision check — 44% strict precision)
  - Strategic scope (strict): business impact|revenue|product strategy|roadmap|
    prioritization|resource allocation|budgeting|cross-functional alignment
    (DROPPED `stakeholder` from strict set per T21 precision check)
  - Strategic scope (broad): strict + stakeholder

Also compute them for entry and mid-senior for cross-seniority management
comparison (T21 step 8).

Inputs:
  - exploration/artifacts/shared/swe_cleaned_text.parquet
Outputs:
  - exploration/artifacts/T21/T21_densities.parquet
  - exploration/tables/T21/T21_densities_summary.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/home/jihgaboot/gabor/job-research")
CLEANED = ROOT / "exploration/artifacts/shared/swe_cleaned_text.parquet"
OUT_ART = ROOT / "exploration/artifacts/T21"
OUT_TBL = ROOT / "exploration/tables/T21"
OUT_TBL.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
MGMT_PATTERN = re.compile(
    r"\b(mentor|coach|hire|headcount|performance[- ]?review)\w*",
    re.IGNORECASE,
)

ORCH_STRICT_PATTERN = re.compile(
    r"\b("
    r"architecture review|code review|system design|technical direction|"
    r"ai orchestration|workflow|pipeline|automation|evaluate|validate|"
    r"quality gate|guardrails|prompt engineering|tool selection"
    r")\w*",
    re.IGNORECASE,
)

# Broad orchestration adds back `agent` for sensitivity
ORCH_BROAD_PATTERN = re.compile(
    r"\b("
    r"architecture review|code review|system design|technical direction|"
    r"ai orchestration|agent|workflow|pipeline|automation|evaluate|validate|"
    r"quality gate|guardrails|prompt engineering|tool selection"
    r")\w*",
    re.IGNORECASE,
)

STRAT_STRICT_PATTERN = re.compile(
    r"\b("
    r"business impact|revenue|product strategy|roadmap|prioritization|"
    r"resource allocation|budgeting|cross[- ]functional alignment"
    r")\w*",
    re.IGNORECASE,
)

STRAT_BROAD_PATTERN = re.compile(
    r"\b("
    r"stakeholder|business impact|revenue|product strategy|roadmap|"
    r"prioritization|resource allocation|budgeting|cross[- ]functional alignment"
    r")\w*",
    re.IGNORECASE,
)


def count(text: str, pat: re.Pattern) -> int:
    if not text:
        return 0
    return len(pat.findall(text))


# TDD asserts
def run_tests() -> None:
    assert count("mentor junior engineers", MGMT_PATTERN) == 1
    assert count("mentor hire coach", MGMT_PATTERN) == 3
    assert count("not relevant", MGMT_PATTERN) == 0

    assert count("review the architecture", ORCH_STRICT_PATTERN) == 0  # doesn't match "architecture review"
    assert count("architecture review weekly", ORCH_STRICT_PATTERN) == 1
    assert count("build ci/cd pipeline", ORCH_STRICT_PATTERN) == 1
    assert count("build ci/cd pipelines", ORCH_STRICT_PATTERN) == 1  # stem
    assert count("evaluate candidates", ORCH_STRICT_PATTERN) == 1

    assert count("stakeholder alignment", STRAT_STRICT_PATTERN) == 0
    assert count("stakeholder alignment", STRAT_BROAD_PATTERN) == 1
    assert count("drive product strategy", STRAT_STRICT_PATTERN) == 1
    assert count("own the roadmap", STRAT_STRICT_PATTERN) == 1


run_tests()

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pq.read_table(
    CLEANED,
    columns=[
        "uid",
        "description_cleaned",
        "text_source",
        "source",
        "period",
        "seniority_final",
        "seniority_3level",
        "seniority_final_source",
        "is_aggregator",
        "company_name_canonical",
        "swe_classification_tier",
    ],
).to_pandas()

# Restrict to LLM-labeled, known-seniority (for fair cross-seniority comparison)
df = df[(df["text_source"] == "llm") & (df["seniority_final"] != "unknown")].reset_index(drop=True)
print(f"Loaded rows: {len(df):,}")

# Period bucket
df["period_bucket"] = df["period"].apply(
    lambda p: "2024" if p.startswith("2024") else "2026"
)

# Compute counts and densities
print("Computing profile densities ...")
df["desc_len"] = df["description_cleaned"].str.len().fillna(1).astype("int64")
df["mgmt_count"] = df["description_cleaned"].apply(lambda t: count(t, MGMT_PATTERN))
df["orch_strict_count"] = df["description_cleaned"].apply(lambda t: count(t, ORCH_STRICT_PATTERN))
df["orch_broad_count"] = df["description_cleaned"].apply(lambda t: count(t, ORCH_BROAD_PATTERN))
df["strat_strict_count"] = df["description_cleaned"].apply(lambda t: count(t, STRAT_STRICT_PATTERN))
df["strat_broad_count"] = df["description_cleaned"].apply(lambda t: count(t, STRAT_BROAD_PATTERN))

df["mgmt_density"] = df["mgmt_count"] / df["desc_len"].clip(lower=1) * 1000
df["orch_strict_density"] = df["orch_strict_count"] / df["desc_len"].clip(lower=1) * 1000
df["orch_broad_density"] = df["orch_broad_count"] / df["desc_len"].clip(lower=1) * 1000
df["strat_strict_density"] = df["strat_strict_count"] / df["desc_len"].clip(lower=1) * 1000
df["strat_broad_density"] = df["strat_broad_count"] / df["desc_len"].clip(lower=1) * 1000

df["mgmt_binary"] = (df["mgmt_count"] > 0).astype("int64")
df["orch_strict_binary"] = (df["orch_strict_count"] > 0).astype("int64")
df["orch_broad_binary"] = (df["orch_broad_count"] > 0).astype("int64")
df["strat_strict_binary"] = (df["strat_strict_count"] > 0).astype("int64")
df["strat_broad_binary"] = (df["strat_broad_count"] > 0).astype("int64")

# Save full density table
out_cols = [
    "uid",
    "source",
    "period",
    "period_bucket",
    "seniority_final",
    "seniority_3level",
    "seniority_final_source",
    "is_aggregator",
    "company_name_canonical",
    "swe_classification_tier",
    "desc_len",
    "mgmt_count",
    "mgmt_density",
    "mgmt_binary",
    "orch_strict_count",
    "orch_strict_density",
    "orch_strict_binary",
    "orch_broad_count",
    "orch_broad_density",
    "orch_broad_binary",
    "strat_strict_count",
    "strat_strict_density",
    "strat_strict_binary",
    "strat_broad_count",
    "strat_broad_density",
    "strat_broad_binary",
]
pq.write_table(pa.Table.from_pandas(df[out_cols], preserve_index=False), OUT_ART / "T21_densities.parquet")
print(f"wrote {OUT_ART / 'T21_densities.parquet'}  ({len(df):,} rows)")

# Per seniority × period summary
def summary_for(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    g = frame.groupby(["period_bucket", "seniority_final"])
    agg = g.agg(
        n=("uid", "count"),
        desc_len_mean=("desc_len", "mean"),
        mgmt_binary_share=("mgmt_binary", "mean"),
        mgmt_density_mean=("mgmt_density", "mean"),
        orch_strict_binary_share=("orch_strict_binary", "mean"),
        orch_strict_density_mean=("orch_strict_density", "mean"),
        orch_broad_binary_share=("orch_broad_binary", "mean"),
        orch_broad_density_mean=("orch_broad_density", "mean"),
        strat_strict_binary_share=("strat_strict_binary", "mean"),
        strat_strict_density_mean=("strat_strict_density", "mean"),
        strat_broad_binary_share=("strat_broad_binary", "mean"),
        strat_broad_density_mean=("strat_broad_density", "mean"),
    ).reset_index()
    agg["subset"] = label
    return agg


s_all = summary_for(df, "all")
s_noagg = summary_for(df[~df["is_aggregator"]], "no_aggregator")
s_arsh_only = summary_for(df[df["source"] == "kaggle_arshkon"], "arshkon_only_2024")
# Arshkon-only gives 2024 baseline from that single source; 2026 is scraped-only already.

summary_all = pd.concat([s_all, s_noagg, s_arsh_only], ignore_index=True)
summary_all.to_csv(OUT_TBL / "T21_densities_summary.csv", index=False)
print(f"wrote {OUT_TBL / 'T21_densities_summary.csv'}")

# Also compute arshkon-only 2024 vs scraped-only 2026 (matching Wave-3 spec)
senior_arsh_2024 = df[(df["source"] == "kaggle_arshkon") & (df["period_bucket"] == "2024") & (df["seniority_final"].isin(["mid-senior", "director"]))]
senior_scraped_2026 = df[(df["source"] == "scraped") & (df["period_bucket"] == "2026") & (df["seniority_final"].isin(["mid-senior", "director"]))]
print(f"\nSenior arshkon 2024: n={len(senior_arsh_2024)}")
print(f"Senior scraped 2026: n={len(senior_scraped_2026)}")

# Print summary
print("\nAll subset summary (by period x seniority):")
print(s_all.to_string())
