"""
T20 Feature Builder.

Builds a per-posting feature vector for the T20 seniority-boundary analysis.

Features (per T20 spec):
  - yoe_numeric (impute median=3 where null)
  - tech_count (from tech matrix or T11 features)
  - ai_mention (binary, V1-refined strict pattern)
  - org_scope_density (per 1K chars cleaned text)
  - management_density (per 1K chars, V1-refined strict pattern)
  - description_length_cleaned
  - education_level (ordinal 0=none, 1=BS, 2=MS, 3=PhD)

Scope: SWE, LinkedIn-only, is_english=true, date_flag='ok', is_swe=true, text_source='llm'.

Inputs:
  - exploration/artifacts/shared/swe_cleaned_text.parquet
  - exploration/artifacts/shared/swe_tech_matrix.parquet
  - exploration/artifacts/T11/T11_posting_features.parquet

Outputs:
  - exploration/artifacts/T20/T20_features.parquet
  - exploration/artifacts/T20/T20_feature_summary.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/home/jihgaboot/gabor/job-research")
SHARED = ROOT / "exploration/artifacts/shared"
T11 = ROOT / "exploration/artifacts/T11"
OUT = ROOT / "exploration/artifacts/T20"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. V1-refined patterns.
# ---------------------------------------------------------------------------
# AI-mention STRICT (V1-refined): dropped mcp
AI_STRICT_PATTERN = re.compile(
    r"\b("
    r"copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|"
    r"llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|"
    r"vector database|pinecone|huggingface|hugging face"
    r")\b",
    re.IGNORECASE,
)

# Management STRICT (V1-refined): mentor|coach|hire|headcount|performance review
MGMT_STRICT_PATTERN = re.compile(
    r"\b(mentor|coach|hire|headcount|performance[- ]?review)\w*",
    re.IGNORECASE,
)

# Org scope language (same definition used in T11, shared across tasks)
ORG_SCOPE_PATTERN = re.compile(
    r"\b("
    r"ownership|end[- ]to[- ]end|cross[- ]functional|cross functional|"
    r"stakeholders?|strategic|roadmap|scope|initiative|strategy|"
    r"vision|mission|organization|orchestrat\w*|company[- ]wide|org[- ]wide|"
    r"enterprise[- ]wide"
    r")\b",
    re.IGNORECASE,
)

# Education regexes. Each searches for a credential mention; we take the max
# level found in the text as the ordinal feature.
EDU_PHD_PATTERN = re.compile(
    r"\b(ph\.?d\.?|doctor(?:ate|al)?|doctoral\s+degree)\b",
    re.IGNORECASE,
)
EDU_MS_PATTERN = re.compile(
    r"\b("
    r"m\.?s\.?c?\.?|master['\u2019]?s?|masters(?:\s+degree)?|"
    r"mba|ms\s+in\s+\w+|master\s+of\b"
    r")\b",
    re.IGNORECASE,
)
EDU_BS_PATTERN = re.compile(
    r"\b("
    r"b\.?s\.?c?\.?|b\.?a\.?|bachelor['\u2019]?s?|bachelors(?:\s+degree)?|"
    r"bs\s+in\s+\w+|ba\s+in\s+\w+|bachelor\s+of\b|undergraduate\s+degree"
    r")\b",
    re.IGNORECASE,
)


def count_matches(text: str, pattern: re.Pattern) -> int:
    if not text:
        return 0
    return len(pattern.findall(text))


def classify_education(text: str) -> int:
    """Return 0=none, 1=BS, 2=MS, 3=PhD (max mention found)."""
    if not text:
        return 0
    if EDU_PHD_PATTERN.search(text):
        return 3
    if EDU_MS_PATTERN.search(text):
        return 2
    if EDU_BS_PATTERN.search(text):
        return 1
    return 0


# ---------------------------------------------------------------------------
# 2. TDD asserts.
# ---------------------------------------------------------------------------
def run_tests() -> None:
    # AI strict
    assert AI_STRICT_PATTERN.search("we use copilot") is not None
    assert AI_STRICT_PATTERN.search("fine-tuning RAG pipelines") is not None
    assert AI_STRICT_PATTERN.search("insurance agent") is None
    assert AI_STRICT_PATTERN.search("user agent") is None
    assert AI_STRICT_PATTERN.search("mcp certified") is None  # dropped

    # Mgmt strict
    assert MGMT_STRICT_PATTERN.search("mentor junior engineers") is not None
    assert MGMT_STRICT_PATTERN.search("performance review") is not None
    assert MGMT_STRICT_PATTERN.search("performance-review") is not None
    assert MGMT_STRICT_PATTERN.search("we coach our leads") is not None
    # Avoid false positives on 'management' / 'manager' (not in strict set)
    assert MGMT_STRICT_PATTERN.search("product manager") is None
    assert MGMT_STRICT_PATTERN.search("project management") is None
    assert count_matches("mentor and coach and hire new mentors", MGMT_STRICT_PATTERN) == 4

    # Scope
    assert ORG_SCOPE_PATTERN.search("work end-to-end") is not None
    assert ORG_SCOPE_PATTERN.search("own roadmap") is not None
    assert ORG_SCOPE_PATTERN.search("nothing here") is None

    # Education
    assert classify_education("BS in Computer Science") == 1
    assert classify_education("Bachelor's degree") == 1
    assert classify_education("MS in AI or BS in CS") == 2
    assert classify_education("PhD preferred") == 3
    assert classify_education("Ph.D. or equivalent experience") == 3
    assert classify_education("No degree required") == 0


run_tests()


# ---------------------------------------------------------------------------
# 3. Load data.
# ---------------------------------------------------------------------------
print("Loading swe_cleaned_text.parquet ...")
t_cols = [
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
    "yoe_extracted",
    "swe_classification_tier",
]
cleaned = pq.read_table(SHARED / "swe_cleaned_text.parquet", columns=t_cols).to_pandas()
print(f"  cleaned rows: {len(cleaned):,}")

# Filter to LLM-labeled (T20 step 1 requires density metrics)
cleaned = cleaned[cleaned["text_source"] == "llm"].reset_index(drop=True)
print(f"  LLM-labeled rows: {len(cleaned):,}")

# Filter to non-unknown seniority
cleaned = cleaned[cleaned["seniority_final"] != "unknown"].reset_index(drop=True)
print(f"  known-seniority rows: {len(cleaned):,}")

# Load T11 features to reuse tech_count
print("Loading T11_posting_features.parquet ...")
t11_cols = ["uid", "tech_count", "org_scope_count", "desc_len_chars"]
t11 = pq.read_table(T11 / "T11_posting_features.parquet", columns=t11_cols).to_pandas()
print(f"  T11 rows: {len(t11):,}")

# Load tech matrix (backup if T11 missing for a row, but T11 covers llm=34102)
print("Loading swe_tech_matrix.parquet ...")
tech = pq.read_table(SHARED / "swe_tech_matrix.parquet").to_pandas()
tech_cols = [c for c in tech.columns if c != "uid"]
tech["tech_count_matrix"] = tech[tech_cols].sum(axis=1).astype("int64")
tech_min = tech[["uid", "tech_count_matrix"]]

# Merge
print("Merging ...")
feat = cleaned.merge(t11, on="uid", how="left")
feat = feat.merge(tech_min, on="uid", how="left")

# Fill tech_count from matrix if T11 missing
feat["tech_count"] = feat["tech_count"].fillna(feat["tech_count_matrix"]).astype("int64")

print("Computing per-row features ...")
# Description length (prefer T11's desc_len_chars; fall back to len(cleaned))
feat["description_length_cleaned"] = feat["desc_len_chars"].fillna(
    feat["description_cleaned"].str.len()
).astype("int64")

# AI mention binary
feat["ai_mention"] = feat["description_cleaned"].apply(
    lambda t: 1 if (t and AI_STRICT_PATTERN.search(t)) else 0
).astype("int64")

# Management count (V1 strict)
feat["management_count_strict"] = feat["description_cleaned"].apply(
    lambda t: count_matches(t, MGMT_STRICT_PATTERN)
).astype("int64")

# Management density per 1K chars
feat["management_density"] = (
    feat["management_count_strict"] / feat["description_length_cleaned"].clip(lower=1) * 1000.0
).astype("float64")

# Org scope (take T11's count if present; else re-derive)
feat["org_scope_count_fill"] = feat["org_scope_count"].fillna(
    feat["description_cleaned"].apply(lambda t: count_matches(t, ORG_SCOPE_PATTERN))
).astype("int64")
feat["org_scope_density"] = (
    feat["org_scope_count_fill"] / feat["description_length_cleaned"].clip(lower=1) * 1000.0
).astype("float64")

# Education level
feat["education_level"] = feat["description_cleaned"].apply(classify_education).astype("int64")

# YOE: impute median (spec says median 3; we use 3)
feat["yoe_numeric"] = feat["yoe_extracted"].fillna(3.0).astype("float64")
feat["yoe_imputed"] = feat["yoe_extracted"].isna()

# ---------------------------------------------------------------------------
# 4. Period bucket.
# ---------------------------------------------------------------------------
# 2024-01 and 2024-04 → 2024; 2026-03 and 2026-04 → 2026.
feat["period_bucket"] = feat["period"].apply(
    lambda p: "2024" if p.startswith("2024") else ("2026" if p.startswith("2026") else "other")
)

# Source bucket (arshkon only vs pooled)
feat["is_arshkon"] = (feat["source"] == "kaggle_arshkon").astype("int64")
feat["is_scraped"] = (feat["source"] == "scraped").astype("int64")

# Final feature columns
feature_cols = [
    "uid",
    "source",
    "period",
    "period_bucket",
    "seniority_final",
    "seniority_3level",
    "is_aggregator",
    "company_name_canonical",
    "swe_classification_tier",
    "yoe_numeric",
    "yoe_imputed",
    "tech_count",
    "ai_mention",
    "org_scope_density",
    "management_density",
    "description_length_cleaned",
    "education_level",
    "management_count_strict",
    "org_scope_count_fill",
]
feat_out = feat[feature_cols].copy()

# Rename org_scope_count_fill -> org_scope_count for clarity
feat_out = feat_out.rename(columns={"org_scope_count_fill": "org_scope_count"})

# ---------------------------------------------------------------------------
# 5. Write outputs.
# ---------------------------------------------------------------------------
out_parquet = OUT / "T20_features.parquet"
pq.write_table(pa.Table.from_pandas(feat_out, preserve_index=False), out_parquet)
print(f"wrote {out_parquet}  ({len(feat_out):,} rows, {len(feat_out.columns)} cols)")

# Summary
summary = feat_out.groupby(["period_bucket", "seniority_final"]).agg(
    n=("uid", "count"),
    yoe_mean=("yoe_numeric", "mean"),
    yoe_imputed_share=("yoe_imputed", "mean"),
    tech_count_mean=("tech_count", "mean"),
    ai_mention_share=("ai_mention", "mean"),
    org_scope_density_mean=("org_scope_density", "mean"),
    management_density_mean=("management_density", "mean"),
    description_length_mean=("description_length_cleaned", "mean"),
    education_level_mean=("education_level", "mean"),
).reset_index()
summary.to_csv(OUT / "T20_feature_summary.csv", index=False)
print(f"wrote {OUT / 'T20_feature_summary.csv'}")
print(summary.to_string())
