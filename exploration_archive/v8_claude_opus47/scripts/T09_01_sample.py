"""T09 Step 1: Build the 8k-row stratified sample.

Strategy (per task spec):
- ~2,700 per period-bucket across [2024-04, 2024-01, 2026-03/04].
- Within each period, stratify by seniority_final using T30 primary slices
  (J2 = entry|associate, mid-senior, S1/director).
- Prefer text_source='llm' rows (mandatory here, since downstream topic modeling
  uses LLM-cleaned text).
- For the 2024 buckets: 2024-04 is arshkon-only and 2024-01 is asaniczka-only,
  so the "prefer arshkon" instruction simply means we don't downweight 2024-04.

Reads: exploration/artifacts/shared/swe_cleaned_text.parquet
Writes: exploration/artifacts/T09/sample_index.parquet (uid + columns needed downstream)
"""

import numpy as np
import pandas as pd
import duckdb

SEED = 20260417
TARGET_PER_PERIOD = 2_700
TARGET_TOTAL = 8_000

# Strata within each period: (label, list_of_seniority_final)
STRATA = [
    ("j2_entry_associate", ["entry", "associate"]),
    ("mid_senior", ["mid-senior"]),
    ("director", ["director"]),
    ("unknown", ["unknown"]),
]
# Proportions used to carve each period budget.
# Rationale: spec says "stratify by J2 junior, mid-senior, S1/director".
# Mid-senior dominates naturally; we upweight J2 and director to ensure
# cluster characterization has >=50 rows of each cell per period.
STRATA_SHARE = {
    "j2_entry_associate": 0.25,
    "mid_senior": 0.45,
    "director": 0.10,
    "unknown": 0.20,
}

# Period buckets and source priority
PERIODS = [
    ("2024-04", ["kaggle_arshkon"]),       # arshkon-only for 2024-04
    ("2024-01", ["kaggle_asaniczka"]),     # asaniczka-only for 2024-01
    ("2026-03", ["scraped"]),
    ("2026-04", ["scraped"]),
]
# 2024-04 + 2024-01 + 2026-03/04 -> 4 period codes but 3 semantic buckets
# Spec wants 2,700 per semantic bucket:
PERIOD_BUDGET = {
    "2024-04": 2_700,
    "2024-01": 2_700,
    "2026-03": 1_350,
    "2026-04": 1_250,   # slight under to keep total near 8k
}


def sample_period(df_period: pd.DataFrame, budget: int, rng: np.random.Generator) -> pd.DataFrame:
    """Stratified sample within a single period."""
    picks = []
    remainder_budget = budget
    strata_order = [s for s, _ in STRATA]
    # First pass: stratum quota
    for stratum_name in strata_order:
        seniority_levels = dict(STRATA)[stratum_name]
        stratum_df = df_period[df_period["seniority_final"].isin(seniority_levels)]
        target = int(round(budget * STRATA_SHARE[stratum_name]))
        n = min(target, len(stratum_df))
        if n > 0:
            idx = rng.choice(len(stratum_df), size=n, replace=False)
            picks.append(stratum_df.iloc[idx])
            remainder_budget -= n

    picked = pd.concat(picks) if picks else pd.DataFrame(columns=df_period.columns)

    # Second pass: fill remaining budget from anywhere we haven't picked yet.
    if remainder_budget > 0:
        already = set(picked["uid"])
        pool = df_period[~df_period["uid"].isin(already)]
        n = min(remainder_budget, len(pool))
        if n > 0:
            idx = rng.choice(len(pool), size=n, replace=False)
            picked = pd.concat([picked, pool.iloc[idx]])
    return picked


def main():
    rng = np.random.default_rng(SEED)
    con = duckdb.connect()

    # Load all LLM rows with columns we need
    df = con.execute("""
        SELECT uid, description_cleaned, text_source, source, period,
               seniority_final, seniority_3level, seniority_final_source,
               is_aggregator, company_name_canonical, yoe_extracted,
               swe_classification_tier,
               LENGTH(description_cleaned) AS description_cleaned_length
        FROM 'exploration/artifacts/shared/swe_cleaned_text.parquet'
        WHERE text_source = 'llm'
    """).fetchdf()

    print(f"LLM-labeled SWE LinkedIn rows: {len(df):,}")

    # Restrict to source constraints per period
    allowed = []
    for period, sources in PERIODS:
        sub = df[(df.period == period) & (df.source.isin(sources))].copy()
        allowed.append(sub)
    df = pd.concat(allowed, ignore_index=True)
    print(f"After source-priority filter: {len(df):,}")

    # Remove rows with near-empty text (avoid embedding noise)
    df = df[df.description_cleaned.str.len() >= 200].reset_index(drop=True)
    print(f"After min-length 200 chars filter: {len(df):,}")

    # Sample per period
    chunks = []
    for period, _ in PERIODS:
        sub = df[df.period == period]
        budget = PERIOD_BUDGET[period]
        sampled = sample_period(sub, budget, rng)
        sampled["_bucket_period"] = period
        print(f"  {period} budget={budget}, got={len(sampled)}")
        chunks.append(sampled)
    sample = pd.concat(chunks, ignore_index=True)
    print(f"Total sample: {len(sample):,}")

    # Final composition checks
    print("\n-- Sample composition --")
    print(sample.groupby(["period", "source"]).size().rename("n").to_string())
    print()
    print("text_source:", sample.text_source.value_counts().to_dict())
    print("seniority_final:")
    print(sample.seniority_final.value_counts().to_string())
    print("is_aggregator:", sample.is_aggregator.value_counts().to_dict())
    print("swe_classification_tier:", sample.swe_classification_tier.value_counts().to_dict())

    # Save
    out = "exploration/artifacts/T09/sample_index.parquet"
    sample.to_parquet(out, index=False)
    print(f"\nWrote {out}")
    return sample


if __name__ == "__main__":
    main()
