"""
T19 — rate of change estimation + within-snapshot stability checks.

Snapshots: asaniczka (2024-01, Jan 12-17, 6 days), arshkon (2024-04, Apr 5-20),
scraped 2026-03 (Mar 20-27, 8d) + 2026-04 (up to 30d).

Key metrics (SWE sample, linkedin/en/ok):
 - Entry share: seniority_native, seniority_final, combined best-available, YOE<=2 proxy
 - AI keyword prevalence (agentic + any-AI family)
 - Median description length
 - Median tech count (using T18 safe tech list)
 - Org scope language density

For each metric we compute:
 - Value at each snapshot
 - Within-2024 annualized rate: (arshkon - asaniczka) / (3/12 year) = 4 * (arshkon - asaniczka)
 - Cross-period annualized rate: (scraped_2026 - arshkon) / (23/12) = (12/23)*(scraped - arshkon)
 - Acceleration ratio
"""
import os
import duckdb
import pandas as pd
import numpy as np

PARQUET = "data/unified.parquet"
OUT_DIR = "exploration/tables/T19"
os.makedirs(OUT_DIR, exist_ok=True)

con = duckdb.connect()

BASE = """source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe"""

AI_CASE = """CASE WHEN regexp_matches(lower(coalesce(description_core_llm, description_core, description)),
   'agentic|multi[- ]agent|\\bai[- ]agent|\\bllm\\b|\\brag\\b|generative ai|\\bcopilot\\b|ai[- ]powered|\\bclaude\\b|large language model') THEN 1 ELSE 0 END"""
AGENTIC_CASE = """CASE WHEN regexp_matches(lower(coalesce(description_core_llm, description_core, description)),
   'agentic') THEN 1 ELSE 0 END"""

SENIORITY_BEST = """
CASE
  WHEN llm_classification_coverage = 'labeled'         THEN seniority_llm
  WHEN llm_classification_coverage = 'rule_sufficient' THEN seniority_final
  ELSE NULL
END
"""

SCOPE_CASE = """CASE WHEN regexp_matches(lower(coalesce(description_core_llm, description_core, description)),
   'cross[- ]functional|stakeholders?|\\broadmap\\b|org[- ]wide|company[- ]wide|strategic') THEN 1 ELSE 0 END"""

# tech count using same safe list as T18
TECHS = [
    "python", "java", "javascript|\\bjs\\b", "typescript", "\\bgolang\\b", "\\brust\\b",
    "\\bc\\+\\+|\\bcpp\\b", "\\bc#", "\\breact\\b", "kubernetes|\\bk8s\\b", "\\bdocker\\b",
    "\\baws\\b", "\\bsql\\b", "tensorflow", "pytorch",
]
tech_expr = " + ".join(
    f"(CASE WHEN regexp_matches(lower(coalesce(description_core_llm, description_core, description)), '{p}') THEN 1 ELSE 0 END)"
    for p in TECHS
)

# Snapshot definitions (include asaniczka even under native-only where excluded explicitly in downstream)
SNAPSHOTS = [
    ("asaniczka_2024_01", "source='kaggle_asaniczka'"),
    ("arshkon_2024_04", "source='kaggle_arshkon'"),
    ("scraped_2026_03", "source='scraped' AND period='2026-03'"),
    ("scraped_2026_04", "source='scraped' AND period='2026-04'"),
]


def snapshot_metrics(extra_filter=""):
    rows = []
    for name, cond in SNAPSHOTS:
        q = f"""
        WITH b AS (
          SELECT
            {SENIORITY_BEST} AS sen_best,
            seniority_native, seniority_final, yoe_extracted,
            LENGTH(coalesce(description_core_llm, description_core, description)) AS desc_len,
            {AI_CASE} AS ai_any,
            {AGENTIC_CASE} AS agentic_flag,
            {SCOPE_CASE} AS scope_flag,
            {tech_expr} AS tech_count
          FROM '{PARQUET}'
          WHERE {BASE} AND {cond} {extra_filter}
        )
        SELECT
          '{name}' AS snapshot,
          COUNT(*) AS n,
          AVG(CASE WHEN seniority_native = 'entry' THEN 1.0 WHEN seniority_native IS NULL THEN NULL ELSE 0.0 END) AS entry_native,
          AVG(CASE WHEN seniority_final = 'entry' THEN 1.0 WHEN seniority_final IS NULL THEN NULL ELSE 0.0 END) AS entry_final,
          AVG(CASE WHEN sen_best = 'entry' THEN 1.0 WHEN sen_best IS NULL THEN NULL ELSE 0.0 END) AS entry_best,
          AVG(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted <= 2 THEN 1.0 WHEN yoe_extracted IS NULL THEN NULL ELSE 0.0 END) AS yoe_le2,
          AVG(ai_any::DOUBLE) AS ai_rate,
          AVG(agentic_flag::DOUBLE) AS agentic_rate,
          median(desc_len) AS median_len,
          median(tech_count) AS median_tech,
          AVG(tech_count::DOUBLE) AS mean_tech,
          AVG(scope_flag::DOUBLE) AS scope_rate
        FROM b
        """
        rows.append(con.execute(q).df())
    return pd.concat(rows, ignore_index=True)


print("[T19.1] Snapshot metrics (baseline SWE)...")
snap = snapshot_metrics()
snap.to_csv(os.path.join(OUT_DIR, "snapshot_metrics.csv"), index=False)
print(snap.to_string())


# Rate-of-change table
def rate_table(snap):
    metrics = ["entry_native", "entry_final", "entry_best", "yoe_le2", "ai_rate",
               "agentic_rate", "median_len", "median_tech", "mean_tech", "scope_rate"]
    rows = []
    # Use scraped_2026_03 as the primary 2026 anchor (like arshkon which spans Apr 2024)
    asa = snap[snap.snapshot == "asaniczka_2024_01"].iloc[0]
    ars = snap[snap.snapshot == "arshkon_2024_04"].iloc[0]
    sc3 = snap[snap.snapshot == "scraped_2026_03"].iloc[0]
    sc4 = snap[snap.snapshot == "scraped_2026_04"].iloc[0]
    # Pooled 2026 average weighted by n (excluding nulls)
    for m in metrics:
        a, r, s3, s4 = asa[m], ars[m], sc3[m], sc4[m]
        # pooled 2026 weighted by n
        n3, n4 = sc3["n"], sc4["n"]
        if pd.isna(s3) and pd.isna(s4):
            pooled = np.nan
        elif pd.isna(s3):
            pooled = s4
        elif pd.isna(s4):
            pooled = s3
        else:
            pooled = (s3 * n3 + s4 * n4) / (n3 + n4)
        # Within-2024 annualized: (ars - asa) / (3/12) = 4*(ars-asa)
        within24 = 4 * (r - a) if pd.notna(r) and pd.notna(a) else np.nan
        # Cross-period annualized: (pooled_2026 - arshkon) / (23/12)
        cross = (12 / 23) * (pooled - r) if pd.notna(pooled) and pd.notna(r) else np.nan
        # Acceleration ratio
        if pd.notna(within24) and pd.notna(cross) and within24 not in (0, 0.0):
            accel = cross / within24
        else:
            accel = np.nan
        rows.append({"metric": m, "asaniczka_2024_01": a, "arshkon_2024_04": r,
                     "scraped_2026_03": s3, "scraped_2026_04": s4, "scraped_2026_pooled": pooled,
                     "within_2024_annualized": within24,
                     "cross_period_annualized": cross,
                     "acceleration_ratio": accel})
    return pd.DataFrame(rows)


rates = rate_table(snap)
rates.to_csv(os.path.join(OUT_DIR, "rates_annualized.csv"), index=False)
print("\n[T19.1] Rate-of-change table:")
print(rates.to_string())

# Within-arshkon stability check
print("\n[T19.2] Within-arshkon stability (by scrape_week)...")
q_arsh = f"""
SELECT scrape_week, COUNT(*) AS n,
  AVG(CASE WHEN seniority_native='entry' THEN 1.0 ELSE 0.0 END) as entry_native,
  AVG({AI_CASE}::DOUBLE) as ai_rate,
  median(LENGTH(coalesce(description_core_llm, description_core, description))) as median_len,
  AVG({tech_expr}::DOUBLE) as mean_tech
FROM '{PARQUET}'
WHERE {BASE} AND source='kaggle_arshkon'
GROUP BY scrape_week ORDER BY scrape_week
"""
arsh_weeks = con.execute(q_arsh).df()
print(arsh_weeks.to_string())
arsh_weeks.to_csv(os.path.join(OUT_DIR, "arshkon_within_stability.csv"), index=False)

# scrape_date-level breakdown for scraped (March stability)
print("\n[T19.3] Scraped daily breakdown (March 2026)...")
q_daily = f"""
SELECT scrape_date, COUNT(*) AS n,
  AVG({AI_CASE}::DOUBLE) as ai_rate,
  AVG({AGENTIC_CASE}::DOUBLE) as agentic_rate,
  AVG(CASE WHEN seniority_native='entry' THEN 1.0 WHEN seniority_native IS NULL THEN NULL ELSE 0.0 END) as entry_native_rate,
  AVG(CASE WHEN {SENIORITY_BEST} = 'entry' THEN 1.0 WHEN {SENIORITY_BEST} IS NULL THEN NULL ELSE 0.0 END) as entry_best_rate,
  AVG(CASE WHEN yoe_extracted <= 2 THEN 1.0 WHEN yoe_extracted IS NULL THEN NULL ELSE 0.0 END) as yoe_le2_rate,
  median(LENGTH(coalesce(description_core_llm, description_core, description))) as median_len,
  AVG(CASE WHEN is_aggregator THEN 1.0 ELSE 0.0 END) as agg_rate
FROM '{PARQUET}'
WHERE {BASE} AND source='scraped' AND period='2026-03'
GROUP BY scrape_date ORDER BY scrape_date
"""
sc_daily = con.execute(q_daily).df()
print(sc_daily.to_string())
sc_daily.to_csv(os.path.join(OUT_DIR, "scraped_march_daily.csv"), index=False)

# April daily
print("\n[T19.4] Scraped daily breakdown (April 2026)...")
q_daily_apr = q_daily.replace("period='2026-03'", "period='2026-04'")
sc_daily_apr = con.execute(q_daily_apr).df()
print(sc_daily_apr.to_string())
sc_daily_apr.to_csv(os.path.join(OUT_DIR, "scraped_april_daily.csv"), index=False)

# 2026-03 vs 2026-04 direct comparison
print("\n[T19.5] 2026-03 vs 2026-04 stability...")
q_cmp = f"""
SELECT period, COUNT(*) AS n,
  AVG({AI_CASE}::DOUBLE) as ai_rate,
  AVG({AGENTIC_CASE}::DOUBLE) as agentic_rate,
  AVG(CASE WHEN {SENIORITY_BEST} = 'entry' THEN 1.0 WHEN {SENIORITY_BEST} IS NULL THEN NULL ELSE 0.0 END) as entry_best,
  AVG(CASE WHEN yoe_extracted <= 2 THEN 1.0 WHEN yoe_extracted IS NULL THEN NULL ELSE 0.0 END) as yoe_le2,
  AVG(CASE WHEN seniority_native='entry' THEN 1.0 WHEN seniority_native IS NULL THEN NULL ELSE 0.0 END) as entry_native,
  median(LENGTH(coalesce(description_core_llm, description_core, description))) as median_len,
  AVG({tech_expr}::DOUBLE) as mean_tech
FROM '{PARQUET}'
WHERE {BASE} AND source='scraped'
GROUP BY period ORDER BY period
"""
cmp26 = con.execute(q_cmp).df()
print(cmp26.to_string())
cmp26.to_csv(os.path.join(OUT_DIR, "march_vs_april_2026.csv"), index=False)

# Posting age analysis
print("\n[T19.6] Posting age coverage (scraped only)...")
q_age = f"""
SELECT period, COUNT(*) AS n,
  SUM(CASE WHEN posting_age_days IS NOT NULL THEN 1 ELSE 0 END) AS n_with_age,
  AVG(posting_age_days) AS mean_age,
  median(posting_age_days) AS median_age,
  quantile_cont(posting_age_days, 0.9) AS p90_age
FROM '{PARQUET}'
WHERE {BASE} AND source='scraped'
GROUP BY period ORDER BY period
"""
age = con.execute(q_age).df()
print(age.to_string())
age.to_csv(os.path.join(OUT_DIR, "posting_age.csv"), index=False)

# Day-of-week analysis for scraped
print("\n[T19.7] Day-of-week analysis for scraped...")
q_dow = f"""
SELECT period, dayofweek(CAST(scrape_date AS DATE)) AS dow, COUNT(*) AS n,
  AVG({AI_CASE}::DOUBLE) as ai_rate,
  AVG(CASE WHEN seniority_native='entry' THEN 1.0 WHEN seniority_native IS NULL THEN NULL ELSE 0.0 END) as entry_native,
  median(LENGTH(coalesce(description_core_llm, description_core, description))) as median_len
FROM '{PARQUET}'
WHERE {BASE} AND source='scraped'
GROUP BY period, dow ORDER BY period, dow
"""
dow = con.execute(q_dow).df()
print(dow.to_string())
dow.to_csv(os.path.join(OUT_DIR, "day_of_week.csv"), index=False)

print("\nDone. Outputs in", OUT_DIR)
