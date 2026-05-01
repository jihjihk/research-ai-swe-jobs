#!/usr/bin/env python3
"""
T09 step 1: Build the balanced sample for archetype discovery.

Target: up to 8,000 SWE LinkedIn postings with balanced period representation
across three periods (2024, 2026-03, 2026-04). Within each period, stratify by
the combined best-available seniority column. For 2024, prefer arshkon over
asaniczka (arshkon has entry-level labels). Prefer rows with text_source='llm'.

Output: exploration/tables/T09/sample_uids.parquet (uid + metadata)
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import duckdb

OUT_DIR = "exploration/tables/T09"
os.makedirs(OUT_DIR, exist_ok=True)

RNG = np.random.default_rng(42)
TARGET_PER_PERIOD = 2700
MAX_TOTAL = 8000

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")

BASE_SQL = """
WITH base AS (
  SELECT
    u.uid,
    u.period,
    u.source,
    u.is_aggregator,
    u.swe_classification_tier,
    u.yoe_extracted,
    u.llm_classification_coverage,
    u.seniority_llm,
    u.seniority_final,
    u.seniority_native,
    c.text_source,
    c.description_cleaned,
    c.company_name_canonical,
    c.metro_area,
    CASE
      WHEN u.llm_classification_coverage = 'labeled' THEN u.seniority_llm
      WHEN u.llm_classification_coverage = 'rule_sufficient' THEN u.seniority_final
      ELSE NULL
    END AS seniority_best_available
  FROM 'data/unified.parquet' u
  JOIN 'exploration/artifacts/shared/swe_cleaned_text.parquet' c USING (uid)
  WHERE u.source_platform = 'linkedin'
    AND u.is_english = true
    AND u.date_flag = 'ok'
    AND u.is_swe = true
)
SELECT * FROM base
"""

pool = con.execute(BASE_SQL).fetch_df()
print(f"Full SWE LinkedIn pool (joined with cleaned text): {len(pool):,}")

# Derive an analytical period bucket: 2024 combines 2024-01 and 2024-04.
pool["period_bucket"] = pool["period"].map(
    {"2024-01": "2024", "2024-04": "2024", "2026-03": "2026-03", "2026-04": "2026-04"}
)

# Preference scoring: llm text > rule; arshkon > asaniczka for 2024; labeled with seniority > None.
pool["has_llm_text"] = (pool["text_source"] == "llm").astype(int)
pool["is_arshkon"] = (pool["source"] == "kaggle_arshkon").astype(int)
pool["has_known_seniority"] = pool["seniority_best_available"].notna() & (
    pool["seniority_best_available"] != "unknown"
)

# Stratified sampling within each period bucket by seniority_best_available.
# We treat None/unknown as one stratum ("unknown") and include proportionally.
pool["stratum"] = pool["seniority_best_available"].fillna("unknown")
pool.loc[pool["stratum"] == "unknown", "stratum"] = "unknown"


def sample_period(df_period: pd.DataFrame, target: int, rng: np.random.Generator) -> pd.DataFrame:
    """Proportional stratified sampling, preferring llm text and arshkon source.

    Within each stratum, we sort by preference (llm text first, arshkon first)
    and take the top-k random subset.
    """
    if len(df_period) == 0:
        return df_period
    target = min(target, len(df_period))
    # Compute per-stratum allocation proportional to the natural distribution
    # but floor each known seniority stratum to receive at least its rows.
    stratum_counts = df_period["stratum"].value_counts()
    # proportional allocation
    raw = (stratum_counts / stratum_counts.sum()) * target
    alloc = np.floor(raw).astype(int)
    remainder = target - int(alloc.sum())
    # distribute remainder based on fractional parts
    frac = (raw - alloc).sort_values(ascending=False)
    for s in frac.index[:remainder]:
        alloc[s] += 1
    # Ensure small known-seniority strata get represented (min 5 if available)
    for s, cnt in stratum_counts.items():
        if s == "unknown":
            continue
        if alloc[s] < min(5, cnt):
            delta = min(5, cnt) - alloc[s]
            alloc[s] += delta
    # reduce unknown to fit
    overshoot = int(alloc.sum()) - target
    if overshoot > 0:
        # take from the largest stratum (usually unknown or mid-senior)
        order = alloc.sort_values(ascending=False).index.tolist()
        for s in order:
            take = min(overshoot, alloc[s] - 1)
            alloc[s] -= take
            overshoot -= take
            if overshoot <= 0:
                break

    parts = []
    for stratum, k in alloc.items():
        if k <= 0:
            continue
        sub = df_period[df_period["stratum"] == stratum].copy()
        # preference: llm text first, arshkon first (for 2024), then random
        sub["_rand"] = rng.random(len(sub))
        sub = sub.sort_values(
            by=["has_llm_text", "is_arshkon", "_rand"], ascending=[False, False, True]
        )
        parts.append(sub.head(k).drop(columns=["_rand"]))
    return pd.concat(parts, ignore_index=True)


# Sample each period bucket
samples = []
period_summary_rows = []
for period_bucket in ["2024", "2026-03", "2026-04"]:
    df_p = pool[pool["period_bucket"] == period_bucket]
    # For 2024: prefer arshkon first — we hard-prefer arshkon by splitting pool
    if period_bucket == "2024":
        arshkon = df_p[df_p["source"] == "kaggle_arshkon"]
        asaniczka = df_p[df_p["source"] == "kaggle_asaniczka"]
        # Take up to TARGET_PER_PERIOD from arshkon first; fill remainder from asaniczka.
        ark_take = min(len(arshkon), TARGET_PER_PERIOD)
        ark_sample = sample_period(arshkon, ark_take, RNG)
        remaining = TARGET_PER_PERIOD - len(ark_sample)
        if remaining > 0:
            asan_sample = sample_period(asaniczka, remaining, RNG)
            sample_p = pd.concat([ark_sample, asan_sample], ignore_index=True)
        else:
            sample_p = ark_sample
    else:
        sample_p = sample_period(df_p, TARGET_PER_PERIOD, RNG)

    samples.append(sample_p)
    period_summary_rows.append(
        {
            "period_bucket": period_bucket,
            "pool_size": int(len(df_p)),
            "sampled": int(len(sample_p)),
        }
    )
    print(f"  {period_bucket}: pool={len(df_p):,}  sampled={len(sample_p):,}")

sample = pd.concat(samples, ignore_index=True)
# Enforce max cap
if len(sample) > MAX_TOTAL:
    sample = sample.sample(n=MAX_TOTAL, random_state=42).reset_index(drop=True)
print(f"\nTotal sample: {len(sample):,}")

# Save sample index
keep_cols = [
    "uid",
    "period",
    "period_bucket",
    "source",
    "is_aggregator",
    "swe_classification_tier",
    "yoe_extracted",
    "llm_classification_coverage",
    "seniority_llm",
    "seniority_final",
    "seniority_native",
    "seniority_best_available",
    "text_source",
    "company_name_canonical",
    "metro_area",
    "description_cleaned",
]
sample[keep_cols].to_parquet(f"{OUT_DIR}/sample.parquet", index=False)
print(f"Wrote {OUT_DIR}/sample.parquet")

# Composition tables
comp = (
    sample.groupby(["period_bucket", "source"])
    .size()
    .rename("n")
    .reset_index()
)
print("\nSample composition by period_bucket x source:")
print(comp.to_string(index=False))
comp.to_csv(f"{OUT_DIR}/sample_composition_source.csv", index=False)

comp_ts = (
    sample.groupby(["period_bucket", "text_source"]).size().rename("n").reset_index()
)
print("\nSample composition by period_bucket x text_source:")
print(comp_ts.to_string(index=False))
comp_ts.to_csv(f"{OUT_DIR}/sample_composition_textsource.csv", index=False)

comp_sen = (
    sample.assign(sen=sample["seniority_best_available"].fillna("missing"))
    .groupby(["period_bucket", "sen"]).size().rename("n").reset_index()
)
print("\nSample composition by period_bucket x seniority_best_available:")
print(comp_sen.to_string(index=False))
comp_sen.to_csv(f"{OUT_DIR}/sample_composition_seniority.csv", index=False)

pd.DataFrame(period_summary_rows).to_csv(f"{OUT_DIR}/period_pool_sizes.csv", index=False)

# Sanity: check for nulls in description
n_null = sample["description_cleaned"].isna().sum()
print(f"\nNull description_cleaned: {n_null}")
n_empty = (sample["description_cleaned"].fillna("").str.len() < 50).sum()
print(f"Cleaned text <50 chars: {n_empty}")
