#!/usr/bin/env python3
"""T04: SWE classification audit.

Assesses SWE classification quality: tier breakdown, borderline sampling,
false-positive/negative estimation, dual-flag violations, boundary-case analysis.
"""
import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

BASE = "/home/jihgaboot/gabor/job-research"
FIG_DIR = f"{BASE}/exploration/figures/T04"
TBL_DIR = f"{BASE}/exploration/tables/T04"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

con = duckdb.connect()
PARQUET = f"'{BASE}/data/unified.parquet'"
BASE_FILTER = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'"

# ============================================================
# 1. SWE rows by swe_classification_tier breakdown
# ============================================================
print("=" * 60)
print("STEP 1: SWE classification tier breakdown")
print("=" * 60)

tier_breakdown = con.execute(f"""
    SELECT is_swe, swe_classification_tier, count(*) as n,
      round(count(*) * 100.0 / sum(count(*)) OVER (PARTITION BY is_swe), 2) as pct
    FROM {PARQUET}
    WHERE {BASE_FILTER}
    GROUP BY is_swe, swe_classification_tier
    ORDER BY is_swe DESC, n DESC
""").fetchdf()
print(tier_breakdown.to_string())
tier_breakdown.to_csv(f"{TBL_DIR}/swe_classification_tier_breakdown.csv", index=False)

# By source
tier_by_source = con.execute(f"""
    SELECT source, is_swe, swe_classification_tier, count(*) as n
    FROM {PARQUET}
    WHERE {BASE_FILTER} AND is_swe = true
    GROUP BY source, is_swe, swe_classification_tier
    ORDER BY source, swe_classification_tier
""").fetchdf()
print("\nSWE tier by source:")
print(tier_by_source.to_string())
tier_by_source.to_csv(f"{TBL_DIR}/swe_tier_by_source.csv", index=False)

# Overall flag counts
flag_counts = con.execute(f"""
    SELECT
      sum(CASE WHEN is_swe THEN 1 ELSE 0 END) as swe_count,
      sum(CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) as adjacent_count,
      sum(CASE WHEN is_control THEN 1 ELSE 0 END) as control_count,
      count(*) as total
    FROM {PARQUET}
    WHERE {BASE_FILTER}
""").fetchdf()
print("\nOverall flag counts:")
print(flag_counts.to_string())

# ============================================================
# 2. Sample 50 borderline SWE postings (confidence 0.3-0.7 or title_lookup_llm tier)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Borderline SWE sample (confidence 0.3-0.7 or title_lookup_llm)")
print("=" * 60)

borderline_swe = con.execute(f"""
    SELECT uid, title, LEFT(description, 200) as desc_snippet,
      swe_confidence, swe_classification_tier, source
    FROM {PARQUET}
    WHERE {BASE_FILTER} AND is_swe = true
      AND (swe_confidence BETWEEN 0.3 AND 0.7 OR swe_classification_tier = 'title_lookup_llm')
    ORDER BY random()
    LIMIT 50
""").fetchdf()

print(f"Borderline SWE sample: {len(borderline_swe)} rows")
for i, row in borderline_swe.iterrows():
    print(f"\n  [{i+1}] uid={row['uid']}")
    print(f"      Title: {row['title']}")
    print(f"      Tier: {row['swe_classification_tier']}, Confidence: {row['swe_confidence']:.2f}")
    print(f"      Desc: {row['desc_snippet'][:150]}...")

borderline_swe.to_csv(f"{TBL_DIR}/borderline_swe_sample.csv", index=False)

# Manual quality assessment heuristic
# Count how many have suspicious titles (not clearly SWE)
suspicious_swe_patterns = [
    'data', 'analyst', 'scientist', 'qa ', 'quality assurance',
    'test ', 'tester', 'product manager', 'project manager',
    'business', 'sales', 'support', 'network', 'system admin',
    'security', 'hardware', 'mechanical', 'electrical', 'civil',
    'dba', 'database admin'
]
suspicious_count = 0
for _, row in borderline_swe.iterrows():
    title_lower = str(row['title']).lower()
    if any(p in title_lower for p in suspicious_swe_patterns):
        suspicious_count += 1
print(f"\nSuspicious titles (potential FP) in borderline SWE: {suspicious_count}/{len(borderline_swe)}")

# ============================================================
# 3. Sample 50 borderline non-SWE (titles with engineer/developer/software, is_swe=False)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Borderline non-SWE sample (titles with SWE-like keywords)")
print("=" * 60)

borderline_nonswe = con.execute(f"""
    SELECT uid, title, LEFT(description, 200) as desc_snippet,
      swe_confidence, swe_classification_tier, is_swe_adjacent, source
    FROM {PARQUET}
    WHERE {BASE_FILTER} AND is_swe = false
      AND (lower(title) LIKE '%engineer%' OR lower(title) LIKE '%developer%' OR lower(title) LIKE '%software%')
    ORDER BY random()
    LIMIT 50
""").fetchdf()

print(f"Borderline non-SWE sample: {len(borderline_nonswe)} rows")
for i, row in borderline_nonswe.iterrows():
    print(f"\n  [{i+1}] uid={row['uid']}")
    print(f"      Title: {row['title']}")
    print(f"      Tier: {row['swe_classification_tier']}, Confidence: {row['swe_confidence']:.2f}, Adjacent: {row['is_swe_adjacent']}")
    print(f"      Desc: {row['desc_snippet'][:150]}...")

borderline_nonswe.to_csv(f"{TBL_DIR}/borderline_nonswe_sample.csv", index=False)

# Assess potential false negatives
potential_fn_patterns = [
    'software engineer', 'software developer', 'full stack', 'fullstack',
    'backend', 'frontend', 'web developer', 'mobile developer',
    'application developer', 'platform engineer'
]
fn_count = 0
for _, row in borderline_nonswe.iterrows():
    title_lower = str(row['title']).lower()
    if any(p in title_lower for p in potential_fn_patterns):
        fn_count += 1
print(f"\nPotential false negatives (SWE-like titles classified non-SWE): {fn_count}/{len(borderline_nonswe)}")

# ============================================================
# 4. Profile is_swe_adjacent and is_control rows
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: SWE-adjacent and control row profiles")
print("=" * 60)

# Top adjacent titles
adjacent_titles = con.execute(f"""
    SELECT title_normalized, count(*) as n
    FROM {PARQUET}
    WHERE {BASE_FILTER} AND is_swe_adjacent = true
    GROUP BY title_normalized ORDER BY n DESC LIMIT 30
""").fetchdf()
print("Top 30 SWE-adjacent titles:")
print(adjacent_titles.to_string())
adjacent_titles.to_csv(f"{TBL_DIR}/swe_adjacent_top_titles.csv", index=False)

# Top control titles
control_titles = con.execute(f"""
    SELECT title_normalized, count(*) as n
    FROM {PARQUET}
    WHERE {BASE_FILTER} AND is_control = true
    GROUP BY title_normalized ORDER BY n DESC LIMIT 30
""").fetchdf()
print("\nTop 30 control titles:")
print(control_titles.to_string())
control_titles.to_csv(f"{TBL_DIR}/control_top_titles.csv", index=False)

# Adjacent tier distribution
adjacent_tiers = con.execute(f"""
    SELECT swe_classification_tier, count(*) as n
    FROM {PARQUET}
    WHERE {BASE_FILTER} AND is_swe_adjacent = true
    GROUP BY swe_classification_tier ORDER BY n DESC
""").fetchdf()
print("\nSWE-adjacent tier distribution:")
print(adjacent_tiers.to_string())

# ============================================================
# 5. Estimated false-positive and false-negative rates
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: FP/FN rate estimation")
print("=" * 60)

# Use swe_classification_llm as cross-check where available
fp_fn_check = con.execute(f"""
    SELECT
      is_swe,
      swe_classification_llm,
      count(*) as n
    FROM {PARQUET}
    WHERE {BASE_FILTER}
      AND swe_classification_llm IS NOT NULL
    GROUP BY is_swe, swe_classification_llm
    ORDER BY is_swe, swe_classification_llm
""").fetchdf()
print("Rule-based is_swe vs LLM swe_classification_llm (where LLM available):")
print(fp_fn_check.to_string())
fp_fn_check.to_csv(f"{TBL_DIR}/rule_vs_llm_swe_classification.csv", index=False)

# Compute rates
if len(fp_fn_check) > 0:
    # FP: is_swe=True but LLM says NOT_SWE
    swe_true = fp_fn_check[fp_fn_check['is_swe'] == True]
    swe_true_total = swe_true['n'].sum()
    swe_true_not_swe_llm = swe_true[swe_true['swe_classification_llm'] == 'NOT_SWE']['n'].sum()
    swe_true_adjacent_llm = swe_true[swe_true['swe_classification_llm'] == 'SWE_ADJACENT']['n'].sum()

    # FN: is_swe=False but LLM says SWE
    swe_false = fp_fn_check[fp_fn_check['is_swe'] == False]
    swe_false_total = swe_false['n'].sum()
    swe_false_swe_llm = swe_false[swe_false['swe_classification_llm'] == 'SWE']['n'].sum()

    print(f"\nFP estimate (rule=SWE, LLM=NOT_SWE): {swe_true_not_swe_llm}/{swe_true_total} = {swe_true_not_swe_llm/swe_true_total:.4f}")
    print(f"FP-adjacent (rule=SWE, LLM=SWE_ADJACENT): {swe_true_adjacent_llm}/{swe_true_total} = {swe_true_adjacent_llm/swe_true_total:.4f}")
    print(f"FN estimate (rule=not-SWE, LLM=SWE): {swe_false_swe_llm}/{swe_false_total} = {swe_false_swe_llm/swe_false_total:.4f}")

# By tier
fp_fn_by_tier = con.execute(f"""
    SELECT
      swe_classification_tier,
      is_swe,
      swe_classification_llm,
      count(*) as n
    FROM {PARQUET}
    WHERE {BASE_FILTER}
      AND swe_classification_llm IS NOT NULL
      AND is_swe = true
    GROUP BY swe_classification_tier, is_swe, swe_classification_llm
    ORDER BY swe_classification_tier, swe_classification_llm
""").fetchdf()
print("\nSWE disagreement by classification tier:")
print(fp_fn_by_tier.to_string())
fp_fn_by_tier.to_csv(f"{TBL_DIR}/fp_by_tier.csv", index=False)

# ============================================================
# 6. Verify no dual-flag violations
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Dual-flag violation check")
print("=" * 60)

dual_flag = con.execute(f"""
    SELECT count(*) as violations
    FROM {PARQUET}
    WHERE {BASE_FILTER}
      AND (CAST(is_swe AS INT) + CAST(is_swe_adjacent AS INT) + CAST(is_control AS INT)) > 1
""").fetchdf()
print(f"Dual-flag violations: {dual_flag['violations'].iloc[0]}")

# Also check rows with NONE of the flags
no_flag = con.execute(f"""
    SELECT count(*) as no_flag_count
    FROM {PARQUET}
    WHERE {BASE_FILTER}
      AND NOT is_swe AND NOT is_swe_adjacent AND NOT is_control
""").fetchdf()
print(f"Rows with no flags (not SWE, not adjacent, not control): {no_flag['no_flag_count'].iloc[0]}")

# ============================================================
# 7. Boundary cases: roles that straddle SWE/adjacent
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Boundary-case analysis (SWE/adjacent straddlers)")
print("=" * 60)

# Key boundary roles
boundary_roles = [
    'ml engineer', 'machine learning engineer',
    'data engineer', 'data engineering',
    'devops', 'dev ops', 'site reliability', 'sre',
    'platform engineer',
    'cloud engineer',
    'security engineer', 'cybersecurity engineer',
    'qa engineer', 'quality assurance engineer', 'test engineer', 'sdet',
    'solutions engineer', 'solutions architect',
    'ai engineer', 'artificial intelligence engineer',
]

# Check classification of boundary-role titles across periods
boundary_results = []
for role_pattern in boundary_roles:
    q = con.execute(f"""
        SELECT
          '{role_pattern}' as pattern,
          source,
          is_swe,
          is_swe_adjacent,
          swe_classification_tier,
          count(*) as n
        FROM {PARQUET}
        WHERE {BASE_FILTER}
          AND lower(title) LIKE '%{role_pattern}%'
        GROUP BY source, is_swe, is_swe_adjacent, swe_classification_tier
        ORDER BY source, is_swe DESC
    """).fetchdf()
    if len(q) > 0:
        boundary_results.append(q)

if boundary_results:
    boundary_df = pd.concat(boundary_results, ignore_index=True)
    print(boundary_df.to_string())
    boundary_df.to_csv(f"{TBL_DIR}/boundary_roles_classification.csv", index=False)

# Check if boundary roles get classified differently across sources/periods
print("\n--- Temporal classification consistency for key boundary roles ---")
for role_pattern in ['ml engineer', 'data engineer', 'devops', 'platform engineer', 'cloud engineer', 'ai engineer']:
    q = con.execute(f"""
        SELECT
          source,
          sum(CASE WHEN is_swe THEN 1 ELSE 0 END) as swe_count,
          sum(CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) as adj_count,
          sum(CASE WHEN NOT is_swe AND NOT is_swe_adjacent THEN 1 ELSE 0 END) as other_count,
          count(*) as total,
          round(sum(CASE WHEN is_swe THEN 1 ELSE 0 END) * 100.0 / count(*), 1) as swe_pct
        FROM {PARQUET}
        WHERE {BASE_FILTER}
          AND lower(title) LIKE '%{role_pattern}%'
        GROUP BY source ORDER BY source
    """).fetchdf()
    if len(q) > 0:
        print(f"\n  '{role_pattern}':")
        print(f"    {q.to_string()}")

# ============================================================
# 8. SWE confidence distribution
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: SWE confidence score distribution")
print("=" * 60)

conf_dist = con.execute(f"""
    SELECT
      CASE
        WHEN swe_confidence < 0.3 THEN '0.0-0.3'
        WHEN swe_confidence < 0.5 THEN '0.3-0.5'
        WHEN swe_confidence < 0.7 THEN '0.5-0.7'
        WHEN swe_confidence < 0.85 THEN '0.7-0.85'
        WHEN swe_confidence < 0.95 THEN '0.85-0.95'
        ELSE '0.95-1.0'
      END as confidence_bin,
      is_swe,
      count(*) as n
    FROM {PARQUET}
    WHERE {BASE_FILTER}
    GROUP BY confidence_bin, is_swe
    ORDER BY confidence_bin, is_swe
""").fetchdf()
print(conf_dist.to_string())
conf_dist.to_csv(f"{TBL_DIR}/swe_confidence_distribution.csv", index=False)

# ============================================================
# FIGURES
# ============================================================
print("\n" + "=" * 60)
print("Creating figures...")
print("=" * 60)

# Figure 1: SWE classification tier breakdown
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

swe_tiers = tier_breakdown[tier_breakdown['is_swe'] == True].copy()
if len(swe_tiers) > 0:
    axes[0].barh(swe_tiers['swe_classification_tier'], swe_tiers['n'], color='steelblue')
    axes[0].set_xlabel('Count')
    axes[0].set_title('SWE Classification Tier (is_swe=True)')
    for i, (_, row) in enumerate(swe_tiers.iterrows()):
        axes[0].text(row['n'] + 50, i, f"{row['pct']:.1f}%", va='center', fontsize=9)

nonswe_tiers = tier_breakdown[tier_breakdown['is_swe'] == False].head(6).copy()
if len(nonswe_tiers) > 0:
    axes[1].barh(nonswe_tiers['swe_classification_tier'], nonswe_tiers['n'], color='coral')
    axes[1].set_xlabel('Count')
    axes[1].set_title('SWE Classification Tier (is_swe=False, top 6)')

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/swe_tier_breakdown.png", dpi=150, bbox_inches='tight')
plt.close()

# Figure 2: Boundary role SWE classification rates by source
boundary_summary = []
for role_pattern in ['ml engineer', 'data engineer', 'devops', 'platform engineer', 'cloud engineer', 'qa engineer', 'ai engineer', 'sre']:
    q = con.execute(f"""
        SELECT
          source,
          round(sum(CASE WHEN is_swe THEN 1 ELSE 0 END) * 100.0 / count(*), 1) as swe_pct,
          count(*) as total
        FROM {PARQUET}
        WHERE {BASE_FILTER}
          AND lower(title) LIKE '%{role_pattern}%'
        GROUP BY source ORDER BY source
    """).fetchdf()
    for _, row in q.iterrows():
        boundary_summary.append({
            'role': role_pattern,
            'source': row['source'],
            'swe_pct': row['swe_pct'],
            'total': row['total']
        })

bs_df = pd.DataFrame(boundary_summary)
if len(bs_df) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = bs_df.pivot_table(index='role', columns='source', values='swe_pct', fill_value=0)
    pivot.plot(kind='bar', ax=ax, colormap='Set2')
    ax.set_ylabel('SWE classification rate (%)')
    ax.set_title('Boundary Role SWE Classification Rate by Source')
    ax.set_xlabel('Role pattern')
    ax.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/boundary_role_swe_rates.png", dpi=150, bbox_inches='tight')
    plt.close()

# Figure 3: Confidence distribution
fig, ax = plt.subplots(figsize=(10, 5))
conf_pivot = conf_dist.pivot_table(index='confidence_bin', columns='is_swe', values='n', fill_value=0)
conf_pivot.columns = ['Not SWE', 'SWE']
conf_pivot.plot(kind='bar', ax=ax, color=['coral', 'steelblue'])
ax.set_ylabel('Count')
ax.set_title('SWE Confidence Score Distribution')
ax.set_xlabel('Confidence bin')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/swe_confidence_distribution.png", dpi=150, bbox_inches='tight')
plt.close()

# Figure 4: Rule vs LLM SWE classification (where LLM available)
if len(fp_fn_check) > 0:
    fig, ax = plt.subplots(figsize=(8, 5))
    fp_pivot = fp_fn_check.pivot_table(index='is_swe', columns='swe_classification_llm', values='n', fill_value=0)
    fp_pivot.index = ['Rule: Not SWE', 'Rule: SWE']
    fp_pivot.plot(kind='bar', ax=ax, colormap='Set2')
    ax.set_ylabel('Count')
    ax.set_title('Rule-based is_swe vs LLM swe_classification\n(rows with LLM labels)')
    ax.set_xlabel('')
    ax.legend(title='LLM classification')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/rule_vs_llm_swe.png", dpi=150, bbox_inches='tight')
    plt.close()

print("\nAll T04 outputs saved.")
print(f"  Figures: {FIG_DIR}/")
print(f"  Tables:  {TBL_DIR}/")
