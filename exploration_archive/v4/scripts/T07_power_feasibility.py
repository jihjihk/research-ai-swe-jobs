#!/usr/bin/env python3
"""
T07: Power analysis & feasibility assessment for cross-period comparisons.
Computes minimum detectable effect sizes (MDE) for binary and continuous outcomes
at 80% power, alpha=0.05, for each key comparison group.
"""

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
import csv
import os

OUT_TABLES = "exploration/tables/T07"
OUT_FIGURES = "exploration/figures/T07"
os.makedirs(OUT_TABLES, exist_ok=True)
os.makedirs(OUT_FIGURES, exist_ok=True)

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")

BASE_FILTER = """
    source_platform = 'linkedin'
    AND is_english = true
    AND date_flag = 'ok'
    AND is_swe = true
"""

# =============================================================================
# Part A: Feasibility table
# =============================================================================

# 1. Group sizes
print("=" * 70)
print("Step 1: Actual group sizes")
print("=" * 70)

# --- Overall SWE counts by source ---
overall = con.execute(f"""
    SELECT source, COUNT(*) as n
    FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER}
    GROUP BY source ORDER BY source
""").fetchdf()
print("\nOverall SWE by source:")
print(overall.to_string(index=False))

# --- Seniority counts using seniority_llm (labeled only) ---
seniority_llm = con.execute(f"""
    SELECT source, seniority_llm, COUNT(*) as n
    FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER}
      AND llm_classification_coverage = 'labeled'
      AND seniority_llm IS NOT NULL
    GROUP BY source, seniority_llm ORDER BY source, seniority_llm
""").fetchdf()
print("\nSeniority LLM (labeled only):")
print(seniority_llm.to_string(index=False))

# --- Seniority counts using seniority_final ---
seniority_final = con.execute(f"""
    SELECT source, seniority_final, COUNT(*) as n
    FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER}
    GROUP BY source, seniority_final ORDER BY source, seniority_final
""").fetchdf()
print("\nSeniority final (all SWE):")
print(seniority_final.to_string(index=False))

# --- Seniority counts using seniority_native ---
seniority_native = con.execute(f"""
    SELECT source, seniority_native, COUNT(*) as n
    FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER}
      AND seniority_native IS NOT NULL
    GROUP BY source, seniority_native ORDER BY source, seniority_native
""").fetchdf()
print("\nSeniority native (where available):")
print(seniority_native.to_string(index=False))


# =============================================================================
# 2. Power analysis functions
# =============================================================================

def mde_binary(n1, n2, p_baseline=0.5, alpha=0.05, power=0.80):
    """
    Minimum detectable effect size for a two-sample test of proportions.
    Returns the MDE as an absolute difference in proportions.
    Uses the formula: MDE = (z_alpha/2 + z_beta) * sqrt(p*(1-p)*(1/n1 + 1/n2))
    where p is the baseline proportion.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    se = np.sqrt(p_baseline * (1 - p_baseline) * (1/n1 + 1/n2))
    mde = (z_alpha + z_beta) * se
    return mde


def mde_continuous(n1, n2, sigma=1.0, alpha=0.05, power=0.80):
    """
    Minimum detectable effect size for a two-sample t-test.
    Returns Cohen's d and also the MDE in original units (sigma=1 gives Cohen's d).
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    d = (z_alpha + z_beta) * np.sqrt(1/n1 + 1/n2)
    return d


# =============================================================================
# 3. Build feasibility table
# =============================================================================

# Define the key comparisons
# We need: analysis_type, comparison, n_group1, n_group2

comparisons = []

# Helper: get count for source + seniority level from seniority_final
def get_sf_count(source, level):
    mask = (seniority_final['source'] == source) & (seniority_final['seniority_final'] == level)
    vals = seniority_final.loc[mask, 'n'].values
    return int(vals[0]) if len(vals) > 0 else 0

def get_sllm_count(source, level):
    mask = (seniority_llm['source'] == source) & (seniority_llm['seniority_llm'] == level)
    vals = seniority_llm.loc[mask, 'n'].values
    return int(vals[0]) if len(vals) > 0 else 0

def get_snative_count(source, level):
    mask = (seniority_native['source'] == source) & (seniority_native['seniority_native'] == level)
    vals = seniority_native.loc[mask, 'n'].values
    return int(vals[0]) if len(vals) > 0 else 0

def get_overall_count(source):
    mask = overall['source'] == source
    vals = overall.loc[mask, 'n'].values
    return int(vals[0]) if len(vals) > 0 else 0

# --- All SWE comparisons ---
# arshkon vs scraped
n_arshkon = get_overall_count('kaggle_arshkon')
n_asaniczka = get_overall_count('kaggle_asaniczka')
n_scraped = get_overall_count('scraped')
n_pooled_2024 = n_arshkon + n_asaniczka

comparisons.append(("All SWE", "arshkon vs scraped",
                     n_arshkon, n_scraped, "seniority_final"))
comparisons.append(("All SWE", "pooled_2024 vs scraped",
                     n_pooled_2024, n_scraped, "seniority_final"))
comparisons.append(("All SWE", "asaniczka vs scraped",
                     n_asaniczka, n_scraped, "seniority_final"))

# --- Entry-level comparisons (seniority_final) ---
comparisons.append(("Entry-level (seniority_final)", "arshkon vs scraped",
                     get_sf_count('kaggle_arshkon', 'entry'),
                     get_sf_count('scraped', 'entry'), "seniority_final"))

# --- Entry-level comparisons (seniority_llm, labeled) ---
comparisons.append(("Entry-level (seniority_llm)", "arshkon vs scraped",
                     get_sllm_count('kaggle_arshkon', 'entry'),
                     get_sllm_count('scraped', 'entry'), "seniority_llm"))
comparisons.append(("Entry-level (seniority_llm)", "asaniczka vs scraped",
                     get_sllm_count('kaggle_asaniczka', 'entry'),
                     get_sllm_count('scraped', 'entry'), "seniority_llm"))
comparisons.append(("Entry-level (seniority_llm)", "pooled_2024 vs scraped",
                     get_sllm_count('kaggle_arshkon', 'entry') + get_sllm_count('kaggle_asaniczka', 'entry'),
                     get_sllm_count('scraped', 'entry'), "seniority_llm"))

# --- Entry-level comparisons (seniority_native) ---
comparisons.append(("Entry-level (seniority_native)", "arshkon vs scraped",
                     get_snative_count('kaggle_arshkon', 'entry'),
                     get_snative_count('scraped', 'entry'), "seniority_native"))

# --- Mid-senior comparisons ---
comparisons.append(("Mid-senior (seniority_final)", "arshkon vs scraped",
                     get_sf_count('kaggle_arshkon', 'mid-senior'),
                     get_sf_count('scraped', 'mid-senior'), "seniority_final"))
comparisons.append(("Mid-senior (seniority_final)", "pooled_2024 vs scraped",
                     get_sf_count('kaggle_arshkon', 'mid-senior') + get_sf_count('kaggle_asaniczka', 'mid-senior'),
                     get_sf_count('scraped', 'mid-senior'), "seniority_final"))

comparisons.append(("Mid-senior (seniority_llm)", "arshkon vs scraped",
                     get_sllm_count('kaggle_arshkon', 'mid-senior'),
                     get_sllm_count('scraped', 'mid-senior'), "seniority_llm"))
comparisons.append(("Mid-senior (seniority_llm)", "pooled_2024 vs scraped",
                     get_sllm_count('kaggle_arshkon', 'mid-senior') + get_sllm_count('kaggle_asaniczka', 'mid-senior'),
                     get_sllm_count('scraped', 'mid-senior'), "seniority_llm"))

# --- Associate comparisons ---
comparisons.append(("Associate (seniority_final)", "arshkon vs scraped",
                     get_sf_count('kaggle_arshkon', 'associate'),
                     get_sf_count('scraped', 'associate'), "seniority_final"))

comparisons.append(("Associate (seniority_llm)", "arshkon vs scraped",
                     get_sllm_count('kaggle_arshkon', 'associate'),
                     get_sllm_count('scraped', 'associate'), "seniority_llm"))

# --- Director comparisons ---
comparisons.append(("Director (seniority_final)", "arshkon vs scraped",
                     get_sf_count('kaggle_arshkon', 'director'),
                     get_sf_count('scraped', 'director'), "seniority_final"))

# Build the table
feasibility_rows = []
for analysis_type, comparison, n1, n2, sen_var in comparisons:
    if n1 == 0 or n2 == 0:
        mde_bin = float('inf')
        mde_cont = float('inf')
        verdict = "IMPOSSIBLE (n=0)"
    else:
        mde_bin = mde_binary(n1, n2, p_baseline=0.3)  # typical binary prevalence
        mde_cont = mde_continuous(n1, n2)

        # Verdict logic
        if mde_bin <= 0.03 and mde_cont <= 0.10:
            verdict = "well-powered"
        elif mde_bin <= 0.05 and mde_cont <= 0.20:
            verdict = "adequate"
        elif mde_bin <= 0.10 and mde_cont <= 0.30:
            verdict = "marginal"
        else:
            verdict = "underpowered"

    feasibility_rows.append({
        'analysis_type': analysis_type,
        'comparison': comparison,
        'seniority_variable': sen_var,
        'n_group1': n1,
        'n_group2': n2,
        'MDE_binary_p30': round(mde_bin, 4) if mde_bin != float('inf') else 'inf',
        'MDE_continuous_d': round(mde_cont, 4) if mde_cont != float('inf') else 'inf',
        'verdict': verdict
    })

feas_df = pd.DataFrame(feasibility_rows)
feas_df.to_csv(f"{OUT_TABLES}/feasibility_summary.csv", index=False)
print("\n" + "=" * 70)
print("Feasibility Summary Table")
print("=" * 70)
print(feas_df.to_string(index=False))

# =============================================================================
# 3b. Also compute MDE at different baseline proportions for entry-level share
# =============================================================================
print("\n" + "=" * 70)
print("Entry-level share MDE sensitivity to baseline proportion")
print("=" * 70)
# The key question: can we detect a change in entry-level share?
# Under seniority_final: arshkon entry = 848, arshkon total = 5019 => p = 16.9%
# Under seniority_native: arshkon entry = 769, arshkon total ~3444 (those with native labels)

entry_share_rows = []
configs = [
    ("seniority_final", "arshkon vs scraped", n_arshkon, n_scraped,
     get_sf_count('kaggle_arshkon', 'entry') / n_arshkon),
    ("seniority_final", "pooled_2024 vs scraped", n_pooled_2024, n_scraped,
     (get_sf_count('kaggle_arshkon', 'entry') + get_sf_count('kaggle_asaniczka', 'entry')) / n_pooled_2024),
    ("seniority_llm (labeled)", "arshkon vs scraped", 3133, 3749,
     get_sllm_count('kaggle_arshkon', 'entry') / 3133 if 3133 > 0 else 0),
    ("seniority_llm (labeled)", "pooled_2024 vs scraped", 3133 + 2777, 3749,
     (get_sllm_count('kaggle_arshkon', 'entry') + get_sllm_count('kaggle_asaniczka', 'entry')) / (3133 + 2777)),
    ("seniority_native", "arshkon vs scraped",
     int(seniority_native[seniority_native['source'] == 'kaggle_arshkon']['n'].sum()),
     int(seniority_native[seniority_native['source'] == 'scraped']['n'].sum()),
     get_snative_count('kaggle_arshkon', 'entry') / max(1, int(seniority_native[seniority_native['source'] == 'kaggle_arshkon']['n'].sum()))),
]

for sen_var, comparison, n1, n2, p_base in configs:
    if n1 > 0 and n2 > 0 and p_base > 0:
        mde = mde_binary(n1, n2, p_baseline=p_base)
        entry_share_rows.append({
            'seniority_variable': sen_var,
            'comparison': comparison,
            'n_period1': n1,
            'n_period2': n2,
            'baseline_entry_share': round(p_base, 4),
            'MDE_absolute': round(mde, 4),
            'MDE_relative_pct': round(mde / p_base * 100, 1) if p_base > 0 else 'inf',
        })

entry_share_df = pd.DataFrame(entry_share_rows)
entry_share_df.to_csv(f"{OUT_TABLES}/entry_share_mde.csv", index=False)
print(entry_share_df.to_string(index=False))


# =============================================================================
# 4. Metro-level feasibility
# =============================================================================
print("\n" + "=" * 70)
print("Step 4: Metro-level feasibility")
print("=" * 70)

metro_counts = con.execute(f"""
    SELECT metro_area, source,
           COUNT(*) as n_swe
    FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER}
      AND metro_area IS NOT NULL
    GROUP BY metro_area, source
""").fetchdf()

# Pivot to wide format
metro_pivot = metro_counts.pivot_table(
    index='metro_area', columns='source', values='n_swe', fill_value=0
).reset_index()

# Ensure all columns exist
for col in ['kaggle_arshkon', 'kaggle_asaniczka', 'scraped']:
    if col not in metro_pivot.columns:
        metro_pivot[col] = 0

metro_pivot['pooled_2024'] = metro_pivot['kaggle_arshkon'] + metro_pivot['kaggle_asaniczka']
metro_pivot['arshkon_and_scraped'] = (metro_pivot['kaggle_arshkon'] >= 1).astype(int) & (metro_pivot['scraped'] >= 1).astype(int)

# Count metros at various thresholds
for threshold in [30, 50, 100, 200]:
    # For arshkon vs scraped
    n_metros_arshkon = ((metro_pivot['kaggle_arshkon'] >= threshold) & (metro_pivot['scraped'] >= threshold)).sum()
    # For pooled 2024 vs scraped
    n_metros_pooled = ((metro_pivot['pooled_2024'] >= threshold) & (metro_pivot['scraped'] >= threshold)).sum()
    print(f"  Metros with >= {threshold} SWE per period:")
    print(f"    arshkon & scraped: {n_metros_arshkon}")
    print(f"    pooled_2024 & scraped: {n_metros_pooled}")

# Save the metro feasibility table (26-metro frame)
# Only scraped data covers the 26-metro frame
metro_26 = metro_pivot[metro_pivot['scraped'] > 0].sort_values('scraped', ascending=False)
metro_26.to_csv(f"{OUT_TABLES}/metro_feasibility.csv", index=False)
print(f"\nMetros in scraped frame: {len(metro_26)}")
print("\nTop metros with both arshkon and scraped presence:")
both_present = metro_26[(metro_26['kaggle_arshkon'] > 0) & (metro_26['scraped'] > 0)].copy()
both_present = both_present.sort_values('scraped', ascending=False)
print(both_present[['metro_area', 'kaggle_arshkon', 'kaggle_asaniczka', 'scraped', 'pooled_2024']].head(30).to_string(index=False))


# =============================================================================
# 5. Company overlap panel feasibility
# =============================================================================
print("\n" + "=" * 70)
print("Step 5: Company overlap panel feasibility")
print("=" * 70)

for threshold in [1, 2, 3, 5, 10, 20]:
    r = con.execute(f"""
    WITH arshkon_companies AS (
        SELECT company_name_canonical, COUNT(*) as n
        FROM read_parquet('data/unified.parquet')
        WHERE {BASE_FILTER}
          AND source = 'kaggle_arshkon' AND company_name_canonical IS NOT NULL
        GROUP BY company_name_canonical
        HAVING COUNT(*) >= {threshold}
    ),
    scraped_companies AS (
        SELECT company_name_canonical, COUNT(*) as n
        FROM read_parquet('data/unified.parquet')
        WHERE {BASE_FILTER}
          AND source = 'scraped' AND company_name_canonical IS NOT NULL
        GROUP BY company_name_canonical
        HAVING COUNT(*) >= {threshold}
    )
    SELECT COUNT(*) as n_companies_overlap,
           SUM(a.n) as arshkon_postings,
           SUM(s.n) as scraped_postings
    FROM arshkon_companies a
    JOIN scraped_companies s ON a.company_name_canonical = s.company_name_canonical
    """).fetchone()
    print(f"  Threshold >= {threshold}: {r[0]} companies ({r[1]} arshkon posts, {r[2]} scraped posts)")

# Pooled 2024 vs scraped
print("\n  Pooled 2024 vs scraped:")
for threshold in [1, 2, 3, 5, 10, 20]:
    r = con.execute(f"""
    WITH hist_companies AS (
        SELECT company_name_canonical, COUNT(*) as n
        FROM read_parquet('data/unified.parquet')
        WHERE {BASE_FILTER}
          AND source IN ('kaggle_arshkon', 'kaggle_asaniczka') AND company_name_canonical IS NOT NULL
        GROUP BY company_name_canonical
        HAVING COUNT(*) >= {threshold}
    ),
    scraped_companies AS (
        SELECT company_name_canonical, COUNT(*) as n
        FROM read_parquet('data/unified.parquet')
        WHERE {BASE_FILTER}
          AND source = 'scraped' AND company_name_canonical IS NOT NULL
        GROUP BY company_name_canonical
        HAVING COUNT(*) >= {threshold}
    )
    SELECT COUNT(*) as n_companies_overlap,
           SUM(a.n) as hist_postings,
           SUM(s.n) as scraped_postings
    FROM hist_companies a
    JOIN scraped_companies s ON a.company_name_canonical = s.company_name_canonical
    """).fetchone()
    print(f"    Threshold >= {threshold}: {r[0]} companies ({r[1]} 2024 posts, {r[2]} scraped posts)")


# =============================================================================
# 6. Industry distribution
# =============================================================================
print("\n" + "=" * 70)
print("Step 6: Industry distribution (our data)")
print("=" * 70)

industry = con.execute(f"""
    SELECT source, company_industry, COUNT(*) as n
    FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER}
      AND company_industry IS NOT NULL AND company_industry != ''
    GROUP BY source, company_industry
    ORDER BY source, n DESC
""").fetchdf()

for src in ['kaggle_arshkon', 'scraped']:
    src_data = industry[industry['source'] == src].head(15)
    print(f"\n  Top 15 industries ({src}):")
    print(src_data.to_string(index=False))

# Save industry data
industry.to_csv(f"{OUT_TABLES}/industry_distribution.csv", index=False)

# =============================================================================
# 7. State-level counts for BLS correlation
# =============================================================================
print("\n" + "=" * 70)
print("Step 7: State-level SWE counts for BLS correlation")
print("=" * 70)

state_counts = con.execute(f"""
    SELECT state_normalized, source, COUNT(*) as n_swe
    FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER}
      AND state_normalized IS NOT NULL AND state_normalized != ''
    GROUP BY state_normalized, source
    ORDER BY state_normalized, source
""").fetchdf()

# Pivot
state_pivot = state_counts.pivot_table(
    index='state_normalized', columns='source', values='n_swe', fill_value=0
).reset_index()
for col in ['kaggle_arshkon', 'kaggle_asaniczka', 'scraped']:
    if col not in state_pivot.columns:
        state_pivot[col] = 0
state_pivot['total'] = state_pivot['kaggle_arshkon'] + state_pivot['kaggle_asaniczka'] + state_pivot['scraped']
state_pivot.to_csv(f"{OUT_TABLES}/state_swe_counts.csv", index=False)
print(f"States with SWE postings: {len(state_pivot)}")
print(state_pivot.sort_values('total', ascending=False).head(15).to_string(index=False))


# =============================================================================
# 8. Seniority-stratified power table (comprehensive)
# =============================================================================
print("\n" + "=" * 70)
print("Step 8: Comprehensive seniority-stratified power analysis")
print("=" * 70)

# For each seniority variable, compute: entry share per source, and the MDE
# This uses ALL SWE as the denominator for share calculations

# seniority_final shares
sf_shares = con.execute(f"""
    SELECT source,
           SUM(CASE WHEN seniority_final = 'entry' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as entry_share,
           SUM(CASE WHEN seniority_final = 'associate' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as assoc_share,
           SUM(CASE WHEN seniority_final = 'mid-senior' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as midsen_share,
           SUM(CASE WHEN seniority_final = 'director' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as director_share,
           SUM(CASE WHEN seniority_final = 'unknown' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as unknown_share,
           COUNT(*) as n
    FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER}
    GROUP BY source ORDER BY source
""").fetchdf()
print("\nSeniority_final shares by source:")
print(sf_shares.to_string(index=False))

# seniority_llm shares (labeled only)
sllm_shares = con.execute(f"""
    SELECT source,
           SUM(CASE WHEN seniority_llm = 'entry' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as entry_share,
           SUM(CASE WHEN seniority_llm = 'associate' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as assoc_share,
           SUM(CASE WHEN seniority_llm = 'mid-senior' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as midsen_share,
           SUM(CASE WHEN seniority_llm = 'director' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as director_share,
           SUM(CASE WHEN seniority_llm = 'unknown' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as unknown_share,
           COUNT(*) as n
    FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER}
      AND llm_classification_coverage = 'labeled'
    GROUP BY source ORDER BY source
""").fetchdf()
print("\nSeniority_llm shares by source (labeled only):")
print(sllm_shares.to_string(index=False))

# seniority_native shares (where available)
snative_shares = con.execute(f"""
    SELECT source,
           SUM(CASE WHEN seniority_native = 'entry' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as entry_share,
           SUM(CASE WHEN seniority_native = 'associate' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as assoc_share,
           SUM(CASE WHEN seniority_native = 'mid-senior' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as midsen_share,
           SUM(CASE WHEN seniority_native = 'director' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as director_share,
           COUNT(*) as n
    FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER}
      AND seniority_native IS NOT NULL
    GROUP BY source ORDER BY source
""").fetchdf()
print("\nSeniority_native shares by source (where available):")
print(snative_shares.to_string(index=False))


# =============================================================================
# 9. YOE distribution for continuous outcome power
# =============================================================================
print("\n" + "=" * 70)
print("Step 9: YOE distribution for continuous outcome baseline")
print("=" * 70)

yoe_stats = con.execute(f"""
    SELECT source,
           COUNT(yoe_extracted) as n_with_yoe,
           COUNT(*) as n_total,
           ROUND(COUNT(yoe_extracted) * 100.0 / COUNT(*), 1) as pct_with_yoe,
           ROUND(AVG(yoe_extracted), 2) as mean_yoe,
           ROUND(STDDEV(yoe_extracted), 2) as sd_yoe,
           ROUND(MEDIAN(yoe_extracted), 1) as median_yoe
    FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER}
      AND yoe_extracted IS NOT NULL
    GROUP BY source ORDER BY source
""").fetchdf()
print(yoe_stats.to_string(index=False))

# YOE by seniority level
yoe_by_seniority = con.execute(f"""
    SELECT source, seniority_final,
           COUNT(yoe_extracted) as n_with_yoe,
           ROUND(AVG(yoe_extracted), 2) as mean_yoe,
           ROUND(STDDEV(yoe_extracted), 2) as sd_yoe
    FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER}
      AND yoe_extracted IS NOT NULL
      AND seniority_final != 'unknown'
    GROUP BY source, seniority_final ORDER BY source, seniority_final
""").fetchdf()
print("\nYOE by seniority and source:")
print(yoe_by_seniority.to_string(index=False))


# =============================================================================
# 10. Additional power calculations for specific analyses
# =============================================================================
print("\n" + "=" * 70)
print("Step 10: Power for specific downstream analyses")
print("=" * 70)

specific_analyses = []

# T08: Entry share trend (the central RQ1 metric)
# Need to detect change in entry share from ~17% (arshkon seniority_final) to X
p_entry_arshkon_sf = get_sf_count('kaggle_arshkon', 'entry') / n_arshkon
p_entry_scraped_sf = get_sf_count('scraped', 'entry') / n_scraped
mde_entry = mde_binary(n_arshkon, n_scraped, p_baseline=p_entry_arshkon_sf)
specific_analyses.append({
    'task': 'T08 Entry share (seniority_final)',
    'n1': n_arshkon, 'n2': n_scraped,
    'baseline': round(p_entry_arshkon_sf, 4),
    'observed': round(p_entry_scraped_sf, 4),
    'observed_diff': round(p_entry_scraped_sf - p_entry_arshkon_sf, 4),
    'MDE': round(mde_entry, 4),
    'detectable': abs(p_entry_scraped_sf - p_entry_arshkon_sf) >= mde_entry
})

# T09: Junior scope inflation (YOE within entry-level)
n_entry_arshkon_yoe = con.execute(f"""
    SELECT COUNT(*) FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER} AND source = 'kaggle_arshkon'
      AND seniority_final = 'entry' AND yoe_extracted IS NOT NULL
""").fetchone()[0]
n_entry_scraped_yoe = con.execute(f"""
    SELECT COUNT(*) FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER} AND source = 'scraped'
      AND seniority_final = 'entry' AND yoe_extracted IS NOT NULL
""").fetchone()[0]
if n_entry_arshkon_yoe > 0 and n_entry_scraped_yoe > 0:
    d_entry_yoe = mde_continuous(n_entry_arshkon_yoe, n_entry_scraped_yoe)
    specific_analyses.append({
        'task': 'T09 Entry YOE inflation (seniority_final)',
        'n1': n_entry_arshkon_yoe, 'n2': n_entry_scraped_yoe,
        'baseline': 'N/A', 'observed': 'N/A', 'observed_diff': 'N/A',
        'MDE': round(d_entry_yoe, 4),
        'detectable': 'depends on effect'
    })

# T10: Senior archetype shift (keyword prevalence in mid-senior)
n_midsen_arshkon = get_sf_count('kaggle_arshkon', 'mid-senior')
n_midsen_scraped = get_sf_count('scraped', 'mid-senior')
mde_midsen_kw = mde_binary(n_midsen_arshkon, n_midsen_scraped, p_baseline=0.10)
specific_analyses.append({
    'task': 'T10 Senior keyword (p=0.10, seniority_final)',
    'n1': n_midsen_arshkon, 'n2': n_midsen_scraped,
    'baseline': 0.10, 'observed': 'N/A', 'observed_diff': 'N/A',
    'MDE': round(mde_midsen_kw, 4),
    'detectable': 'depends on effect'
})

# T16: Company panel (threshold >= 3)
# Within-company changes: paired test power
n_panel_companies = 228
# For paired test, MDE = (z_alpha + z_beta) * sigma_diff / sqrt(n)
z_a = stats.norm.ppf(0.975)
z_b = stats.norm.ppf(0.80)
mde_paired = (z_a + z_b) / np.sqrt(n_panel_companies)
specific_analyses.append({
    'task': 'T16 Company panel (paired, n=228)',
    'n1': n_panel_companies, 'n2': n_panel_companies,
    'baseline': 'N/A', 'observed': 'N/A', 'observed_diff': 'N/A',
    'MDE': round(mde_paired, 4),
    'detectable': 'well-powered for medium effects'
})

spec_df = pd.DataFrame(specific_analyses)
spec_df.to_csv(f"{OUT_TABLES}/specific_analysis_power.csv", index=False)
print(spec_df.to_string(index=False))


# =============================================================================
# 11. Additional: text analysis sample sizes (description_core_llm coverage)
# =============================================================================
print("\n" + "=" * 70)
print("Step 11: Text analysis sample sizes (description_core_llm)")
print("=" * 70)

text_coverage = con.execute(f"""
    SELECT source, llm_extraction_coverage, COUNT(*) as n
    FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER}
    GROUP BY source, llm_extraction_coverage
    ORDER BY source, llm_extraction_coverage
""").fetchdf()
print(text_coverage.to_string(index=False))

# SWE-specific
text_coverage_swe = con.execute(f"""
    SELECT source, llm_extraction_coverage, COUNT(*) as n
    FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER}
    GROUP BY source, llm_extraction_coverage
    ORDER BY source, llm_extraction_coverage
""").fetchdf()

# How many SWE have description_core_llm available?
desc_llm = con.execute(f"""
    SELECT source,
           SUM(CASE WHEN description_core_llm IS NOT NULL AND description_core_llm != '' THEN 1 ELSE 0 END) as n_with_llm_text,
           COUNT(*) as n_total,
           ROUND(SUM(CASE WHEN description_core_llm IS NOT NULL AND description_core_llm != '' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct
    FROM read_parquet('data/unified.parquet')
    WHERE {BASE_FILTER}
    GROUP BY source ORDER BY source
""").fetchdf()
print("\nSWE rows with description_core_llm available:")
print(desc_llm.to_string(index=False))

print("\n\nDone. All tables saved to", OUT_TABLES)
