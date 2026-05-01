"""T29 — LLM-authored description detection.

Steps:
1. Define LLM-authorship signals (signature vocab, em-dash density, T13 readability).
2. Compute authorship score per posting.
3. Distribution by period (2024 vs 2026), variance check.
4. Distribution by company.
5. Correlate authorship with Wave 2 findings (length, tech density, AI rate, credential).
6. Re-test headline Wave 2 findings on low-LLM-subset.
7. Verdict.

Critical confound handling:
- Text source: reports under (a) raw `description` text only (all rows) and
  (b) LLM-cleaned text only (labeled rows). The raw-description track is the
  primary authorship measurement — LLM cleaning confounds any authorship signal
  on the cleaned text. Headline Wave 2 re-tests use raw description for authorship
  scoring, then evaluate the Wave 2 metric on the same subset.
- Aggregator sensitivity separately.

Usage:
    ./.venv/bin/python exploration/scripts/T29_authorship_detection.py
"""
from __future__ import annotations

import os
import re
import sys
import json
import numpy as np
import pandas as pd
import duckdb
import pyarrow.parquet as pq

ROOT = "/home/jihgaboot/gabor/job-research"
UNI = f"{ROOT}/data/unified.parquet"
SHARED = f"{ROOT}/exploration/artifacts/shared"
OUT = f"{ROOT}/exploration/tables/T29"
os.makedirs(OUT, exist_ok=True)

SCOPE_SQL = (
    "is_swe AND source_platform='linkedin' AND is_english AND date_flag='ok'"
)

# --------------------------------------------------------------------------
# Signature vocabulary — LLM stylistic tells (2024 research)
# --------------------------------------------------------------------------
# Organized into buckets. "Very strong" LLM tells (delve, tapestry, realm, embark)
# count double. We build one unified density per 1K chars.
SIG_STRONG = [
    r"\bdelve\b", r"\bdelving\b", r"\btapestry\b",
    r"\bin the realm of\b", r"\bembark on\b", r"\bembark upon\b",
    r"\bit['’]s worth noting\b", r"\bat the forefront\b",
    r"\bpivotal\b", r"\bharness\w*\b",
    r"\bcomprehensive\b", r"\bseamless\w*\b", r"\bholistic\b",
    r"\bfurthermore\b", r"\bmoreover\b", r"\bnotably\b",
    r"\brobust\w*\b", r"\bleverage\w*\b", r"\bunleash\w*\b",
    r"\bcutting[- ]edge\b", r"\bstate[- ]of[- ]the[- ]art\b",
    r"\bdynamic\w*\b", r"\bvibrant\b", r"\bstreamlin\w*\b",
    r"\balign with\b", r"\baligned with\b", r"\bin alignment with\b",
    r"\bnavigat\w*\b",
    r"\border of the day\b", r"\bthrive\w*\b", r"\bdive deep\b",
    r"\bmeticulous\w*\b",
    r"\bfoster\w*\b",
]
# Compile — case-insensitive
SIG_PATTERN = re.compile("|".join(SIG_STRONG), re.IGNORECASE)

# Em-dashes: unicode em-dash or double-hyphen acting as em-dash
EM_DASH_PATTERN = re.compile(r"—|(?<![\w-])--(?![\w-])")
# Bullet markers
BULLET_PATTERN = re.compile(r"(?m)^[ \t]*(?:[-*•◦●⁃‣]|\d+[\.\)])\s+")

AI_STRICT_PATTERN = re.compile(
    r"\b(?:copilot|cursor|github copilot|claude|chatgpt|gpt-?4|gpt-?5|openai|anthropic|"
    r"gemini|llama|mistral|rag|retrieval[- ]augmented|langchain|langgraph|llamaindex|"
    r"mcp|multimodal|multi[- ]agent|genai|generative ai|ai[- ]powered|embeddings?|vector "
    r"database|pinecone|weaviate|chromadb)\b",
    re.IGNORECASE
)


def compute_signals(text: str) -> dict:
    """Per-posting signature-vocab density, em-dash density, bullet counts, length.

    Densities per 1K chars, so longer postings aren't over-penalized.
    """
    if not isinstance(text, str) or len(text) == 0:
        return {"sig_vocab_count": 0, "sig_vocab_density": 0.0,
                "em_dash_count": 0, "em_dash_density": 0.0,
                "bullet_count": 0, "bullet_density": 0.0,
                "ai_strict_count": 0, "char_length": 0}
    L = max(len(text), 1)
    per_1k = 1000.0 / L
    sig_count = len(SIG_PATTERN.findall(text))
    em_count = len(EM_DASH_PATTERN.findall(text))
    bullet_count = len(BULLET_PATTERN.findall(text))
    ai_count = len(AI_STRICT_PATTERN.findall(text))
    return {
        "sig_vocab_count": sig_count,
        "sig_vocab_density": sig_count * per_1k,
        "em_dash_count": em_count,
        "em_dash_density": em_count * per_1k,
        "bullet_count": bullet_count,
        "bullet_density": bullet_count * per_1k,
        "ai_strict_count": ai_count,
        "char_length": len(text),
    }


# --------------------------------------------------------------------------
# Quick semantic sanity: apply to a few known 2024 and 2026 samples
# --------------------------------------------------------------------------
print("=" * 72)
print("Step 0: Signal pattern sanity check on small sample")
print("=" * 72)

con = duckdb.connect()
sample = con.execute(f"""
    SELECT uid, source, period,
           description, description_core_llm, llm_extraction_coverage
    FROM read_parquet('{UNI}')
    WHERE {SCOPE_SQL}
    LIMIT 30
""").fetchdf()

# Manually look at densities for a few before bulk run
for i in range(min(5, len(sample))):
    row = sample.iloc[i]
    r = compute_signals(row["description"] or "")
    print(f"uid={row['uid']} src={row['source']} sig_density={r['sig_vocab_density']:.2f} "
          f"em_dash_density={r['em_dash_density']:.2f} len={r['char_length']}")
print()

# --------------------------------------------------------------------------
# Step 1-3: Compute authorship features on full corpus (raw description)
# --------------------------------------------------------------------------
print("=" * 72)
print("Step 1: Compute authorship signals on full SWE LinkedIn corpus")
print("=" * 72)

# Pull raw + cleaned text for all scope rows
corp = con.execute(f"""
    SELECT uid, source, period, company_name_canonical, is_aggregator,
           seniority_final, seniority_3level, yoe_min_years_llm,
           llm_extraction_coverage, description, description_core_llm
    FROM read_parquet('{UNI}')
    WHERE {SCOPE_SQL}
""").fetchdf()
print(f"Rows: {len(corp):,}")

corp["period_year"] = np.where(corp["source"] == "scraped", "2026", "2024")

# Apply on raw description (this is the primary authorship text)
print("Computing raw-description signals...")
raw_sig = corp["description"].apply(lambda x: pd.Series(compute_signals(x)))
raw_sig = raw_sig.add_prefix("raw_")

# Cleaned text (LLM-extracted) — only where labeled
print("Computing LLM-cleaned signals...")
def _cond_signals(row):
    if row["llm_extraction_coverage"] == "labeled" and isinstance(row["description_core_llm"], str):
        return pd.Series(compute_signals(row["description_core_llm"]))
    return pd.Series({"sig_vocab_count": np.nan, "sig_vocab_density": np.nan,
                      "em_dash_count": np.nan, "em_dash_density": np.nan,
                      "bullet_count": np.nan, "bullet_density": np.nan,
                      "ai_strict_count": np.nan, "char_length": np.nan})
llm_sig = corp.apply(_cond_signals, axis=1)
llm_sig = llm_sig.add_prefix("llm_")

corp = pd.concat([corp, raw_sig, llm_sig], axis=1)

# Merge T13 readability features
t13 = pq.read_table(f"{SHARED}/T13_readability_metrics.parquet").to_pandas()[
    ["uid", "avg_sentence_length", "sentence_length_sd", "type_token_ratio",
     "gunning_fog", "flesch_reading_ease", "imperative_density",
     "inclusive_density", "passive_density"]
]
corp = corp.merge(t13, on="uid", how="left")

# --------------------------------------------------------------------------
# Build composite authorship score
# --------------------------------------------------------------------------
# Normalized z-scores across the pooled corpus (so 0 ~= median 2024+2026).
# Higher score => more LLM-like. Features:
#   +raw_sig_vocab_density   (LLM signature words)
#   +raw_em_dash_density     (LLM em-dashes)
#   +avg_sentence_length     (LLM longer sentences)
#   -sentence_length_sd      (LLM more uniform -> lower SD is LLM-like)
#   -type_token_ratio        (LLM more repetitive -> lower TTR is LLM-like)
# We use raw- only for authorship (LLM-cleaned text has a pipeline confound;
# we compare cleaned-only numbers as a sensitivity later).

def zscore(s: pd.Series) -> pd.Series:
    v = s.astype(float)
    m = v.mean()
    sd = v.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return v * 0
    return (v - m) / sd

corp["z_sig_vocab"] = zscore(corp["raw_sig_vocab_density"])
corp["z_em_dash"] = zscore(corp["raw_em_dash_density"])
corp["z_sentence_length"] = zscore(corp["avg_sentence_length"])
# Higher length_sd => more human-like, so invert
corp["z_neg_sd"] = -zscore(corp["sentence_length_sd"])
# Lower TTR => more LLM-like (repetitive), so invert
corp["z_neg_ttr"] = -zscore(corp["type_token_ratio"])

# Composite
corp["authorship_score"] = (
    corp["z_sig_vocab"].fillna(0) +
    corp["z_em_dash"].fillna(0) +
    corp["z_sentence_length"].fillna(0) +
    corp["z_neg_sd"].fillna(0) +
    corp["z_neg_ttr"].fillna(0)
) / 5.0

# Also keep per-feature composite for diagnostics
author_cols = ["uid", "source", "period_year", "company_name_canonical",
               "is_aggregator", "yoe_min_years_llm", "seniority_final",
               "llm_extraction_coverage",
               "raw_sig_vocab_count", "raw_sig_vocab_density",
               "raw_em_dash_count", "raw_em_dash_density",
               "raw_bullet_count", "raw_bullet_density",
               "raw_char_length",
               "llm_sig_vocab_density", "llm_em_dash_density",
               "avg_sentence_length", "sentence_length_sd", "type_token_ratio",
               "gunning_fog", "flesch_reading_ease", "imperative_density",
               "inclusive_density", "passive_density",
               "z_sig_vocab", "z_em_dash", "z_sentence_length", "z_neg_sd", "z_neg_ttr",
               "authorship_score"]
corp[author_cols].to_csv(f"{OUT}/authorship_scores.csv", index=False)
print(f"Saved authorship_scores.csv")

# --------------------------------------------------------------------------
# Step 4: Distribution by period
# --------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Step 4: Authorship distribution by period")
print("=" * 72)

# Print comparison: 2024 vs 2026 for each feature
summary_rows = []
for feature in ["raw_sig_vocab_density", "raw_em_dash_density", "raw_bullet_density",
                "avg_sentence_length", "sentence_length_sd", "type_token_ratio",
                "authorship_score"]:
    for period_year, g in corp.groupby("period_year"):
        v = g[feature].dropna()
        summary_rows.append({
            "feature": feature,
            "period_year": period_year,
            "n": len(v),
            "mean": v.mean(),
            "median": v.median(),
            "sd": v.std(),
            "p25": v.quantile(0.25),
            "p75": v.quantile(0.75),
        })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{OUT}/step4_distribution_by_period.csv", index=False)
print(summary_df.round(3).to_string(index=False))

# Period contrast: means
print("\n=== Period contrast (2026 - 2024) ===")
contrast_rows = []
for feature in ["raw_sig_vocab_density", "raw_em_dash_density", "raw_bullet_density",
                "avg_sentence_length", "sentence_length_sd", "type_token_ratio",
                "authorship_score"]:
    a = corp[corp["period_year"] == "2024"][feature].dropna()
    b = corp[corp["period_year"] == "2026"][feature].dropna()
    if len(a) == 0 or len(b) == 0:
        continue
    pooled_sd = np.sqrt((a.var() * len(a) + b.var() * len(b)) / (len(a) + len(b)))
    d_cohen = (b.mean() - a.mean()) / pooled_sd if pooled_sd > 0 else np.nan
    contrast_rows.append({
        "feature": feature,
        "mean_2024": a.mean(),
        "mean_2026": b.mean(),
        "delta": b.mean() - a.mean(),
        "cohen_d": d_cohen,
        "sd_2024": a.std(),
        "sd_2026": b.std(),
    })
contrast_df = pd.DataFrame(contrast_rows)
contrast_df.to_csv(f"{OUT}/step4_period_contrast.csv", index=False)
print(contrast_df.round(3).to_string(index=False))

# Sensitivity: text source — within scraped rows, compare labeled vs not_selected
print("\n=== Scraped text-source split ===")
scraped = corp[corp["source"] == "scraped"].copy()
scraped["text_source"] = np.where(scraped["llm_extraction_coverage"] == "labeled", "llm_labeled", "raw_fallback")
scr_src = scraped.groupby("text_source")[["raw_sig_vocab_density", "raw_em_dash_density",
                                            "avg_sentence_length", "sentence_length_sd",
                                            "authorship_score"]].agg(["mean", "median", "count"])
print(scr_src.round(3).to_string())
scr_src.to_csv(f"{OUT}/step4_scraped_textsrc_split.csv")

# --------------------------------------------------------------------------
# Step 5: Distribution by company
# --------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Step 5: Authorship by company")
print("=" * 72)

comp = corp.groupby(["company_name_canonical", "period_year"]).agg(
    n=("uid", "count"),
    mean_score=("authorship_score", "mean"),
    mean_sig=("raw_sig_vocab_density", "mean"),
    mean_em_dash=("raw_em_dash_density", "mean"),
    mean_sent_len=("avg_sentence_length", "mean"),
    mean_ttr=("type_token_ratio", "mean"),
).reset_index()

# Most LLM-like and least LLM-like companies in 2026 with >= 20 postings
large = comp[(comp["period_year"] == "2026") & (comp["n"] >= 20)].copy()
large_sorted = large.sort_values("mean_score", ascending=False)
print("\n=== Top 20 'most LLM-like' companies (2026, n>=20) ===")
print(large_sorted.head(20)[["company_name_canonical", "n", "mean_score",
                                "mean_sig", "mean_em_dash", "mean_sent_len"]].round(3).to_string(index=False))
print("\n=== Bottom 20 'least LLM-like' companies (2026, n>=20) ===")
print(large_sorted.tail(20)[["company_name_canonical", "n", "mean_score",
                                "mean_sig", "mean_em_dash", "mean_sent_len"]].round(3).to_string(index=False))
large_sorted.to_csv(f"{OUT}/step5_company_authorship_2026.csv", index=False)

# Same for 2024 with >= 10 postings
large_2024 = comp[(comp["period_year"] == "2024") & (comp["n"] >= 10)].copy()
large_2024_sorted = large_2024.sort_values("mean_score", ascending=False)
print("\n=== Top 20 'most LLM-like' 2024 (n>=10) — LLM-style in 2024 already? ===")
print(large_2024_sorted.head(20)[["company_name_canonical", "n", "mean_score",
                                      "mean_sig", "mean_em_dash", "mean_sent_len"]].round(3).to_string(index=False))
large_2024_sorted.to_csv(f"{OUT}/step5_company_authorship_2024.csv", index=False)

# Company change: companies present in both periods
comp_pivot = comp.pivot(index="company_name_canonical", columns="period_year",
                         values=["mean_score", "n"])
comp_pivot.columns = [f"{c[0]}_{c[1]}" for c in comp_pivot.columns]
comp_pivot = comp_pivot.dropna(subset=["mean_score_2024", "mean_score_2026"])
comp_pivot = comp_pivot[(comp_pivot["n_2024"] >= 5) & (comp_pivot["n_2026"] >= 5)]
comp_pivot["score_delta"] = comp_pivot["mean_score_2026"] - comp_pivot["mean_score_2024"]
comp_pivot = comp_pivot.sort_values("score_delta", ascending=False)
print(f"\nCompanies with n>=5 in BOTH periods: {len(comp_pivot)}")
print("\n=== Top 20 companies with biggest authorship-score increase 2024->2026 ===")
print(comp_pivot.head(20).round(3).to_string())
print("\n=== Bottom 20 companies with biggest authorship-score decrease 2024->2026 ===")
print(comp_pivot.tail(20).round(3).to_string())
comp_pivot.to_csv(f"{OUT}/step5_company_authorship_change.csv")

# --------------------------------------------------------------------------
# Step 6: Correlate authorship with Wave 2 findings
# --------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Step 6: Correlation with Wave 2 findings")
print("=" * 72)

# Merge T11 features
t11 = pq.read_table(f"{SHARED}/T11_posting_features.parquet").to_pandas()[
    ["uid", "tech_count", "requirement_breadth_resid", "credential_stack_depth",
     "scope_density", "tech_density", "mgmt_broad_density", "ai_binary"]
]
w = corp.merge(t11, on="uid", how="inner")

# Within-period correlations
corr_rows = []
for period_year, g in w.groupby("period_year"):
    for feat in ["raw_char_length", "tech_count", "requirement_breadth_resid",
                 "credential_stack_depth", "scope_density", "tech_density", "mgmt_broad_density"]:
        pair = g[["authorship_score", feat]].dropna()
        if len(pair) < 100:
            continue
        r_pearson = pair.corr().iloc[0, 1]
        r_spearman = pair.rank().corr().iloc[0, 1]
        corr_rows.append({
            "period_year": period_year,
            "metric": feat,
            "n": len(pair),
            "pearson_r": r_pearson,
            "spearman_r": r_spearman,
        })
    # AI rate contrast: within period, compare authorship-low vs -high quartiles
    g["author_quartile"] = pd.qcut(g["authorship_score"].fillna(g["authorship_score"].median()),
                                     q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
    ai_by_q = g.groupby("author_quartile")["ai_binary"].mean()
    for q_name, rate in ai_by_q.items():
        corr_rows.append({
            "period_year": period_year,
            "metric": f"ai_rate_by_{q_name}",
            "n": int((g["author_quartile"] == q_name).sum()),
            "pearson_r": np.nan,
            "spearman_r": np.nan,
            "rate": rate,
        })
corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(f"{OUT}/step6_correlations.csv", index=False)
print(corr_df.round(3).to_string(index=False))

# --------------------------------------------------------------------------
# Step 7: Unifying-mechanism test — re-test Wave 2 findings on low-LLM subset
# --------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Step 7: Low-LLM-subset headline re-test (unifying-mechanism)")
print("=" * 72)

# Define "low-LLM" as bottom-quartile authorship score (within each period separately)
corp_with_score = corp.dropna(subset=["authorship_score"]).copy()
corp_with_score["author_quartile"] = corp_with_score.groupby("period_year")["authorship_score"].transform(
    lambda x: pd.qcut(x, q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
)
corp_with_score["is_low_llm"] = corp_with_score["author_quartile"] == "Q1_low"

print("Low-LLM subset size:")
print(corp_with_score.groupby(["period_year", "is_low_llm"]).size())

# Merge Wave 2 features
work = corp_with_score.merge(t11, on="uid", how="inner")
# merge T13 section fields for requirements share
t13_full = pq.read_table(f"{SHARED}/T13_readability_metrics.parquet").to_pandas()[
    ["uid", "sec_requirements_share", "raw_length"]]
work = work.merge(t13_full, on="uid", how="inner")

# Define J3/S4
work["is_J3"] = (work["yoe_min_years_llm"] <= 2) & work["yoe_min_years_llm"].notna()
work["is_S4"] = (work["yoe_min_years_llm"] >= 5) & work["yoe_min_years_llm"].notna()

# For each of the 5 headline findings, compute on (all) vs (low-LLM only) vs (high-LLM only)
headline_rows = []
for subset_name, mask in [("ALL", pd.Series(True, index=work.index)),
                           ("LOW-LLM-Q1", work["author_quartile"] == "Q1_low"),
                           ("HIGH-LLM-Q4", work["author_quartile"] == "Q4_high")]:
    sub = work[mask]
    d24 = sub[sub["period_year"] == "2024"]
    d26 = sub[sub["period_year"] == "2026"]
    if len(d24) < 50 or len(d26) < 50:
        continue

    # Length growth
    headline_rows.append({
        "subset": subset_name, "metric": "raw_length_mean",
        "val_2024": d24["raw_length"].mean(), "val_2026": d26["raw_length"].mean(),
        "delta": d26["raw_length"].mean() - d24["raw_length"].mean(),
        "n_24": len(d24), "n_26": len(d26),
    })
    # AI-strict prevalence (binary from T11)
    headline_rows.append({
        "subset": subset_name, "metric": "ai_binary_rate",
        "val_2024": d24["ai_binary"].astype(float).mean(),
        "val_2026": d26["ai_binary"].astype(float).mean(),
        "delta": (d26["ai_binary"].astype(float).mean() - d24["ai_binary"].astype(float).mean()),
        "n_24": len(d24), "n_26": len(d26),
    })
    # Scope density
    headline_rows.append({
        "subset": subset_name, "metric": "scope_density_mean",
        "val_2024": d24["scope_density"].mean(), "val_2026": d26["scope_density"].mean(),
        "delta": d26["scope_density"].mean() - d24["scope_density"].mean(),
        "n_24": len(d24), "n_26": len(d26),
    })
    # Requirements share change (T13)
    headline_rows.append({
        "subset": subset_name, "metric": "requirements_share",
        "val_2024": d24["sec_requirements_share"].mean(),
        "val_2026": d26["sec_requirements_share"].mean(),
        "delta": d26["sec_requirements_share"].mean() - d24["sec_requirements_share"].mean(),
        "n_24": len(d24), "n_26": len(d26),
    })
    # Credential stack J3
    for tier_name, tmask in [("J3", sub["is_J3"]), ("S4", sub["is_S4"])]:
        t24 = d24[d24["is_J3"] if tier_name == "J3" else d24["is_S4"]]
        t26 = d26[d26["is_J3"] if tier_name == "J3" else d26["is_S4"]]
        if len(t24) < 20 or len(t26) < 20:
            continue
        # share with credential_stack_depth >= 5
        r24 = (t24["credential_stack_depth"] >= 5).mean()
        r26 = (t26["credential_stack_depth"] >= 5).mean()
        headline_rows.append({
            "subset": subset_name, "metric": f"credential_stack>=5_{tier_name}",
            "val_2024": r24, "val_2026": r26, "delta": r26 - r24,
            "n_24": len(t24), "n_26": len(t26),
        })
    # Tech count
    headline_rows.append({
        "subset": subset_name, "metric": "tech_count_mean",
        "val_2024": d24["tech_count"].mean(), "val_2026": d26["tech_count"].mean(),
        "delta": d26["tech_count"].mean() - d24["tech_count"].mean(),
        "n_24": len(d24), "n_26": len(d26),
    })
    # Requirement breadth resid
    headline_rows.append({
        "subset": subset_name, "metric": "requirement_breadth_resid_mean",
        "val_2024": d24["requirement_breadth_resid"].mean(),
        "val_2026": d26["requirement_breadth_resid"].mean(),
        "delta": d26["requirement_breadth_resid"].mean() - d24["requirement_breadth_resid"].mean(),
        "n_24": len(d24), "n_26": len(d26),
    })

head_df = pd.DataFrame(headline_rows)
head_df.to_csv(f"{OUT}/step7_low_llm_subset_retest.csv", index=False)
print("\n=== Headline Wave 2 findings on ALL vs LOW-LLM-Q1 vs HIGH-LLM-Q4 ===")
print(head_df.round(3).to_string(index=False))

# Compute shrinkage ratio: LOW-LLM delta / ALL delta
print("\n=== Shrinkage ratio: delta_LOW-LLM / delta_ALL (lower -> more tool-mediated) ===")
piv = head_df.pivot_table(index="metric", columns="subset", values="delta")
if "LOW-LLM-Q1" in piv.columns and "ALL" in piv.columns:
    piv["low_over_all"] = piv["LOW-LLM-Q1"] / piv["ALL"]
    if "HIGH-LLM-Q4" in piv.columns:
        piv["high_over_all"] = piv["HIGH-LLM-Q4"] / piv["ALL"]
    print(piv.round(3).to_string())
    piv.to_csv(f"{OUT}/step7_shrinkage_ratio.csv")

# Sensitivity: same thing using LLM-cleaned text authorship score
print("\n\n=== Sensitivity: LLM-cleaned-text authorship score (labeled rows only) ===")
llm_text_mask = corp["llm_extraction_coverage"] == "labeled"
llm_text_corp = corp[llm_text_mask].copy()
# Build llm-based score
llm_text_corp["llm_z_sig"] = zscore(llm_text_corp["llm_sig_vocab_density"])
llm_text_corp["llm_z_em"] = zscore(llm_text_corp["llm_em_dash_density"])
llm_text_corp["llm_z_sentlen"] = zscore(llm_text_corp["avg_sentence_length"])
llm_text_corp["llm_z_neg_sd"] = -zscore(llm_text_corp["sentence_length_sd"])
llm_text_corp["llm_z_neg_ttr"] = -zscore(llm_text_corp["type_token_ratio"])
llm_text_corp["authorship_score_llm"] = (
    llm_text_corp["llm_z_sig"].fillna(0) +
    llm_text_corp["llm_z_em"].fillna(0) +
    llm_text_corp["llm_z_sentlen"].fillna(0) +
    llm_text_corp["llm_z_neg_sd"].fillna(0) +
    llm_text_corp["llm_z_neg_ttr"].fillna(0)
) / 5.0

for period_year, g in llm_text_corp.groupby("period_year"):
    v = g["authorship_score_llm"].dropna()
    print(f"  LLM-cleaned authorship 2026={period_year} mean={v.mean():.3f} median={v.median():.3f} n={len(v)}")

llm_period_contrast = []
for feature in ["llm_sig_vocab_density", "llm_em_dash_density",
                "avg_sentence_length", "sentence_length_sd", "type_token_ratio",
                "authorship_score_llm"]:
    a = llm_text_corp[llm_text_corp["period_year"] == "2024"][feature].dropna()
    b = llm_text_corp[llm_text_corp["period_year"] == "2026"][feature].dropna()
    llm_period_contrast.append({
        "feature": feature, "mean_2024": a.mean(), "mean_2026": b.mean(),
        "delta": b.mean() - a.mean(),
        "n_2024": len(a), "n_2026": len(b),
    })
llm_contrast_df = pd.DataFrame(llm_period_contrast)
print("\n=== LLM-cleaned text period contrast ===")
print(llm_contrast_df.round(3).to_string(index=False))
llm_contrast_df.to_csv(f"{OUT}/step7_llm_text_period_contrast.csv", index=False)

# --------------------------------------------------------------------------
# Step 8: Semantic validation on 30-row sample for signature vocab
# --------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Step 8: Signature vocab density — 30-row sample inspection")
print("=" * 72)

high_score = corp.nlargest(15, "authorship_score")[["uid", "source", "period_year",
                                                       "company_name_canonical",
                                                       "authorship_score",
                                                       "raw_sig_vocab_count",
                                                       "raw_sig_vocab_density",
                                                       "raw_em_dash_density",
                                                       "avg_sentence_length",
                                                       "sentence_length_sd"]]
low_score = corp.nsmallest(15, "authorship_score")[["uid", "source", "period_year",
                                                      "company_name_canonical",
                                                      "authorship_score",
                                                      "raw_sig_vocab_count",
                                                      "raw_sig_vocab_density",
                                                      "raw_em_dash_density",
                                                      "avg_sentence_length",
                                                      "sentence_length_sd"]]
print("\n=== Highest-score 15 ===")
print(high_score.round(3).to_string(index=False))
print("\n=== Lowest-score 15 ===")
print(low_score.round(3).to_string(index=False))
high_score.to_csv(f"{OUT}/step8_top_examples.csv", index=False)
low_score.to_csv(f"{OUT}/step8_bottom_examples.csv", index=False)

# Compare by source
print("\n=== Authorship score by source x period (cross-check) ===")
src_dist = corp.groupby(["source", "period_year"]).agg(
    n=("uid", "count"),
    mean_sig=("raw_sig_vocab_density", "mean"),
    mean_em=("raw_em_dash_density", "mean"),
    mean_score=("authorship_score", "mean"),
).round(3)
print(src_dist)
src_dist.to_csv(f"{OUT}/step8_source_period_distribution.csv")

print("\n\nDone. Outputs in", OUT)
