#!/usr/bin/env python3
"""T06: Company concentration & within-company decomposition.

Checks if a few employers dominate and bias findings.
Decomposes aggregate seniority changes into within-company vs between-company components.
"""

import duckdb
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

PARQUET = 'data/unified.parquet'
FIG_DIR = 'exploration/figures/T06'
TAB_DIR = 'exploration/tables/T06'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

BASE_FILTER = """
    source_platform = 'linkedin'
    AND is_english = true
    AND date_flag = 'ok'
    AND is_swe = true
"""

con = duckdb.connect()

# ============================================================
# 1. Concentration metrics: HHI, top-k share, Gini
# ============================================================
print("=== 1. Concentration Metrics ===")

company_counts = con.execute(f"""
    SELECT source, company_name_canonical, COUNT(*) as cnt
    FROM '{PARQUET}'
    WHERE {BASE_FILTER}
      AND company_name_canonical IS NOT NULL
    GROUP BY source, company_name_canonical
    ORDER BY source, cnt DESC
""").fetchdf()

sources = ['kaggle_arshkon', 'kaggle_asaniczka', 'scraped']
source_labels = {'kaggle_arshkon': 'Arshkon (2024-04)', 'kaggle_asaniczka': 'Asaniczka (2024-01)', 'scraped': 'Scraped (2026)'}

def compute_hhi(counts):
    total = counts.sum()
    shares = counts / total
    return (shares ** 2).sum()

def compute_gini(counts):
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    cum = np.cumsum(sorted_counts)
    return (2 * np.sum(np.arange(1, n+1) * sorted_counts) - (n + 1) * cum[-1]) / (n * cum[-1])

def top_k_share(counts_sorted, k):
    total = counts_sorted.sum()
    return counts_sorted[:k].sum() / total * 100

concentration_results = []
for s in sources:
    sub = company_counts[company_counts.source == s].sort_values('cnt', ascending=False)
    counts = sub['cnt'].values
    n_companies = len(counts)
    total_postings = counts.sum()
    hhi = compute_hhi(counts)
    gini = compute_gini(counts)
    top1 = top_k_share(counts, 1)
    top5 = top_k_share(counts, 5)
    top10 = top_k_share(counts, 10)
    top20 = top_k_share(counts, 20)

    print(f"\n{source_labels[s]}:")
    print(f"  Companies: {n_companies:,d}, Postings: {total_postings:,d}")
    print(f"  HHI: {hhi:.6f}")
    print(f"  Gini: {gini:.4f}")
    print(f"  Top-1: {top1:.1f}%, Top-5: {top5:.1f}%, Top-10: {top10:.1f}%, Top-20: {top20:.1f}%")

    concentration_results.append({
        'source': s, 'n_companies': n_companies, 'total_postings': total_postings,
        'hhi': hhi, 'gini': gini,
        'top1_share': top1, 'top5_share': top5, 'top10_share': top10, 'top20_share': top20
    })

pd.DataFrame(concentration_results).to_csv(f'{TAB_DIR}/concentration_metrics.csv', index=False)

# ============================================================
# 2. Companies with >3% SWE postings in any dataset
# ============================================================
print("\n\n=== 2. Companies with >3% Share ===")

dominant_companies = []
for s in sources:
    sub = company_counts[company_counts.source == s]
    total = sub['cnt'].sum()
    sub = sub.copy()
    sub['share'] = sub['cnt'] / total * 100
    dominant = sub[sub['share'] > 3]
    for _, row in dominant.iterrows():
        print(f"  {source_labels[s]:30s} {row['company_name_canonical']:40s} {row['share']:.1f}% ({int(row['cnt'])} postings)")
        dominant_companies.append({
            'source': s, 'company': row['company_name_canonical'],
            'count': int(row['cnt']), 'share_pct': row['share']
        })

pd.DataFrame(dominant_companies).to_csv(f'{TAB_DIR}/dominant_companies_3pct.csv', index=False)

# ============================================================
# 3. Within-company vs aggregate decomposition (CORE OUTPUT)
# ============================================================
print("\n\n=== 3. Within-Company vs Aggregate Decomposition ===")

# Use multiple seniority operationalizations
for sen_var, sen_label, extra_filter in [
    ('seniority_llm', 'seniority_llm (primary)', "AND llm_classification_coverage = 'labeled'"),
    ('seniority_native', 'seniority_native', "AND seniority_native IS NOT NULL"),
    ('seniority_final', 'seniority_final', ''),
]:
    print(f"\n--- {sen_label} ---")

    # Get company x source x seniority counts
    decomp_data = con.execute(f"""
        SELECT source, company_name_canonical as company, {sen_var} as seniority, COUNT(*) as cnt
        FROM '{PARQUET}'
        WHERE {BASE_FILTER}
          AND company_name_canonical IS NOT NULL
          AND {sen_var} IS NOT NULL
          AND {sen_var} != 'unknown'
          {extra_filter}
        GROUP BY source, company_name_canonical, {sen_var}
    """).fetchdf()

    # Identify overlap companies with >= 5 postings (with known seniority) in BOTH arshkon and scraped
    co_ark = decomp_data[decomp_data.source == 'kaggle_arshkon'].groupby('company')['cnt'].sum()
    co_scr = decomp_data[decomp_data.source == 'scraped'].groupby('company')['cnt'].sum()

    overlap_ark = set(co_ark[co_ark >= 5].index)
    overlap_scr = set(co_scr[co_scr >= 5].index)
    overlap_companies = overlap_ark & overlap_scr

    print(f"  Companies with >=5 known-seniority postings in BOTH arshkon and scraped: {len(overlap_companies)}")

    if len(overlap_companies) == 0:
        print("  WARNING: No overlap companies found. Trying >=3 threshold...")
        overlap_ark = set(co_ark[co_ark >= 3].index)
        overlap_scr = set(co_scr[co_scr >= 3].index)
        overlap_companies = overlap_ark & overlap_scr
        print(f"  Companies with >=3 known-seniority postings in BOTH: {len(overlap_companies)}")

    if len(overlap_companies) == 0:
        print("  Skipping decomposition for this variable - no overlap companies")
        continue

    # Compute entry share: overlap companies only
    for period_src, period_label in [('kaggle_arshkon', '2024 (arshkon)'), ('scraped', '2026 (scraped)')]:
        sub = decomp_data[(decomp_data.source == period_src) & (decomp_data.company.isin(overlap_companies))]
        total = sub['cnt'].sum()
        entry = sub[sub.seniority == 'entry']['cnt'].sum()
        entry_share = entry / total * 100 if total > 0 else 0
        print(f"  Overlap companies entry share ({period_label}): {entry}/{total} = {entry_share:.1f}%")

    # Full-sample aggregate entry share
    for period_src, period_label in [('kaggle_arshkon', '2024 (arshkon)'), ('scraped', '2026 (scraped)')]:
        sub = decomp_data[decomp_data.source == period_src]
        total = sub['cnt'].sum()
        entry = sub[sub.seniority == 'entry']['cnt'].sum()
        entry_share = entry / total * 100 if total > 0 else 0
        print(f"  Full-sample entry share ({period_label}): {entry}/{total} = {entry_share:.1f}%")

    # Compute the actual decomposition
    # Within-company: hold company composition constant, compute entry share change
    # Between-company: hold entry shares constant, compute effect of composition change
    overlap_list = list(overlap_companies)

    # For each company, compute entry share in each period
    company_entry_shares = {}
    company_weights = {}
    for co in overlap_list:
        for period_src in ['kaggle_arshkon', 'scraped']:
            sub = decomp_data[(decomp_data.source == period_src) & (decomp_data.company == co)]
            total = sub['cnt'].sum()
            entry = sub[sub.seniority == 'entry']['cnt'].sum()
            e_share = entry / total if total > 0 else 0
            company_entry_shares[(co, period_src)] = e_share
            company_weights[(co, period_src)] = total

    # Aggregate entry share in each period (overlap only)
    for period_src in ['kaggle_arshkon', 'scraped']:
        total_all = sum(company_weights[(co, period_src)] for co in overlap_list)
        agg_entry = sum(company_entry_shares[(co, period_src)] * company_weights[(co, period_src)]
                        for co in overlap_list) / total_all if total_all > 0 else 0
        print(f"  Weighted aggregate entry share (overlap, {period_src}): {agg_entry*100:.1f}%")

    # Shift-share decomposition
    # Total change = within-company effect + between-company effect + interaction
    # within = sum_i w_i0 * (e_i1 - e_i0)  -- holds weights at period 0
    # between = sum_i (w_i1 - w_i0) * e_i0  -- holds entry shares at period 0
    # interaction = sum_i (w_i1 - w_i0) * (e_i1 - e_i0)

    total_t0 = sum(company_weights.get((co, 'kaggle_arshkon'), 0) for co in overlap_list)
    total_t1 = sum(company_weights.get((co, 'scraped'), 0) for co in overlap_list)

    within_effect = 0
    between_effect = 0
    interaction_effect = 0
    for co in overlap_list:
        w0 = company_weights.get((co, 'kaggle_arshkon'), 0) / total_t0 if total_t0 > 0 else 0
        w1 = company_weights.get((co, 'scraped'), 0) / total_t1 if total_t1 > 0 else 0
        e0 = company_entry_shares.get((co, 'kaggle_arshkon'), 0)
        e1 = company_entry_shares.get((co, 'scraped'), 0)

        within_effect += w0 * (e1 - e0)
        between_effect += (w1 - w0) * e0
        interaction_effect += (w1 - w0) * (e1 - e0)

    agg_t0 = sum(company_entry_shares.get((co, 'kaggle_arshkon'), 0) * company_weights.get((co, 'kaggle_arshkon'), 0)
                 for co in overlap_list) / total_t0 if total_t0 > 0 else 0
    agg_t1 = sum(company_entry_shares.get((co, 'scraped'), 0) * company_weights.get((co, 'scraped'), 0)
                 for co in overlap_list) / total_t1 if total_t1 > 0 else 0
    total_change = agg_t1 - agg_t0

    print(f"\n  SHIFT-SHARE DECOMPOSITION ({sen_label}):")
    print(f"    Entry share t0 (arshkon overlap): {agg_t0*100:.2f}%")
    print(f"    Entry share t1 (scraped overlap):  {agg_t1*100:.2f}%")
    print(f"    Total change:     {total_change*100:+.2f} pp")
    print(f"    Within-company:   {within_effect*100:+.2f} pp")
    print(f"    Between-company:  {between_effect*100:+.2f} pp")
    print(f"    Interaction:      {interaction_effect*100:+.2f} pp")
    print(f"    Check: {(within_effect + between_effect + interaction_effect)*100:+.2f} pp (should match total)")

    if abs(total_change) > 0.001:
        print(f"    Within-company share of total change: {within_effect/total_change*100:.0f}%")
        print(f"    Between-company share: {between_effect/total_change*100:.0f}%")
        print(f"    Interaction share: {interaction_effect/total_change*100:.0f}%")

    # Interpretation
    if within_effect > 0 and total_change > 0:
        print("    INTERPRETATION: Within-company entry share increased. Companies that existed in both periods are posting more entry roles.")
    elif within_effect < 0 and total_change < 0:
        print("    INTERPRETATION: Within-company entry share decreased. Existing companies are posting fewer entry roles.")
    elif within_effect > 0 and total_change < 0:
        print("    INTERPRETATION: Within-company effect is positive but between-company composition pulls aggregate down.")
    elif within_effect < 0 and total_change > 0:
        print("    INTERPRETATION: Within-company effect is negative but composition shifts pull aggregate up.")

# ============================================================
# 4. Company-capped sensitivity: entry share after capping at 10 postings/company
# ============================================================
print("\n\n=== 4. Company-Capped Sensitivity (cap=10 postings/company) ===")

for sen_var, sen_label, extra_filter in [
    ('seniority_llm', 'seniority_llm (primary)', "AND llm_classification_coverage = 'labeled'"),
    ('seniority_native', 'seniority_native', "AND seniority_native IS NOT NULL"),
    ('seniority_final', 'seniority_final', ''),
]:
    # Get per-company seniority data with row-level uid for sampling
    capped_data = con.execute(f"""
        WITH ranked AS (
            SELECT source, company_name_canonical as company, {sen_var} as seniority,
                   ROW_NUMBER() OVER (PARTITION BY source, company_name_canonical ORDER BY uid) as rn
            FROM '{PARQUET}'
            WHERE {BASE_FILTER}
              AND company_name_canonical IS NOT NULL
              AND {sen_var} IS NOT NULL
              AND {sen_var} != 'unknown'
              {extra_filter}
        )
        SELECT source, seniority, COUNT(*) as cnt
        FROM ranked
        WHERE rn <= 10
        GROUP BY source, seniority
        ORDER BY source, seniority
    """).fetchdf()

    # Uncapped
    uncapped_data = con.execute(f"""
        SELECT source, {sen_var} as seniority, COUNT(*) as cnt
        FROM '{PARQUET}'
        WHERE {BASE_FILTER}
          AND {sen_var} IS NOT NULL
          AND {sen_var} != 'unknown'
          {extra_filter}
        GROUP BY source, {sen_var}
        ORDER BY source, {sen_var}
    """).fetchdf()

    print(f"\n  {sen_label}:")
    for s in sources:
        # Uncapped
        sub_u = uncapped_data[uncapped_data.source == s]
        total_u = sub_u['cnt'].sum()
        entry_u = sub_u[sub_u.seniority == 'entry']['cnt'].sum()
        pct_u = entry_u / total_u * 100 if total_u > 0 else 0

        # Capped
        sub_c = capped_data[capped_data.source == s]
        total_c = sub_c['cnt'].sum()
        entry_c = sub_c[sub_c.seniority == 'entry']['cnt'].sum()
        pct_c = entry_c / total_c * 100 if total_c > 0 else 0

        print(f"    {source_labels[s]:30s} uncapped: {pct_u:.1f}% ({int(entry_u)}/{int(total_u)})  capped: {pct_c:.1f}% ({int(entry_c)}/{int(total_c)})  delta: {pct_c - pct_u:+.1f} pp")

# ============================================================
# 5. Aggregator analysis
# ============================================================
print("\n\n=== 5. Aggregator Analysis ===")

agg_rates = con.execute(f"""
    SELECT source,
           COUNT(*) as total,
           SUM(CASE WHEN is_aggregator THEN 1 ELSE 0 END) as agg_count,
           ROUND(100.0 * SUM(CASE WHEN is_aggregator THEN 1 ELSE 0 END) / COUNT(*), 1) as agg_pct
    FROM '{PARQUET}'
    WHERE {BASE_FILTER}
    GROUP BY source
""").fetchdf()

print("\nAggregator rates by source:")
print(agg_rates.to_string(index=False))

# Aggregator vs non-aggregator comparison
print("\nAggregator vs non-aggregator profiles:")
agg_profile = con.execute(f"""
    SELECT source, is_aggregator,
           COUNT(*) as n,
           MEDIAN(description_length) as median_desc_len,
           AVG(description_length) as mean_desc_len,
           MEDIAN(yoe_extracted) as median_yoe
    FROM '{PARQUET}'
    WHERE {BASE_FILTER}
    GROUP BY source, is_aggregator
    ORDER BY source, is_aggregator
""").fetchdf()
print(agg_profile.to_string(index=False))

# Seniority distribution for aggregators vs non-aggregators
print("\nSeniority distribution (seniority_final, excl unknown) by aggregator status:")
agg_sen = con.execute(f"""
    SELECT source, is_aggregator, seniority_final, COUNT(*) as cnt
    FROM '{PARQUET}'
    WHERE {BASE_FILTER}
      AND seniority_final != 'unknown'
    GROUP BY source, is_aggregator, seniority_final
    ORDER BY source, is_aggregator, seniority_final
""").fetchdf()

for s in sources:
    print(f"\n  {source_labels[s]}:")
    for is_agg in [False, True]:
        sub = agg_sen[(agg_sen.source == s) & (agg_sen.is_aggregator == is_agg)]
        total = sub['cnt'].sum()
        label = "Aggregator" if is_agg else "Direct"
        for _, row in sub.iterrows():
            pct = row['cnt'] / total * 100 if total > 0 else 0
            print(f"    {label:12s} {row['seniority_final']:15s} {int(row['cnt']):>5d} ({pct:.1f}%)")

# Top aggregator companies
print("\nTop aggregator companies by source:")
agg_companies = con.execute(f"""
    SELECT source, company_name_canonical, COUNT(*) as cnt
    FROM '{PARQUET}'
    WHERE {BASE_FILTER}
      AND is_aggregator = true
    GROUP BY source, company_name_canonical
    ORDER BY source, cnt DESC
""").fetchdf()

for s in sources:
    sub = agg_companies[agg_companies.source == s].head(5)
    print(f"\n  {source_labels[s]}:")
    for _, row in sub.iterrows():
        print(f"    {row['company_name_canonical']:40s} {int(row['cnt']):>5d}")

# ============================================================
# 6. New entrants analysis
# ============================================================
print("\n\n=== 6. New Entrants Analysis ===")

# Companies in scraped not in arshkon
ark_companies = set(company_counts[company_counts.source == 'kaggle_arshkon']['company_name_canonical'])
scr_companies = set(company_counts[company_counts.source == 'scraped']['company_name_canonical'])

new_entrants = scr_companies - ark_companies
returning = scr_companies & ark_companies

print(f"Companies in scraped: {len(scr_companies)}")
print(f"Companies in arshkon: {len(ark_companies)}")
print(f"New entrants (in scraped, not in arshkon): {len(new_entrants)}")
print(f"Returning (in both): {len(returning)}")

# Seniority profile: new entrants vs returning (using seniority_final)
print("\nSeniority profile of new vs returning companies (seniority_final, excl unknown):")
new_sen = con.execute(f"""
    SELECT
        CASE WHEN company_name_canonical IN (
            SELECT DISTINCT company_name_canonical
            FROM '{PARQUET}'
            WHERE source = 'kaggle_arshkon'
              AND {BASE_FILTER}
              AND company_name_canonical IS NOT NULL
        ) THEN 'returning' ELSE 'new_entrant' END as company_type,
        seniority_final,
        COUNT(*) as cnt
    FROM '{PARQUET}'
    WHERE {BASE_FILTER}
      AND source = 'scraped'
      AND company_name_canonical IS NOT NULL
      AND seniority_final != 'unknown'
    GROUP BY company_type, seniority_final
    ORDER BY company_type, seniority_final
""").fetchdf()

for ct in ['new_entrant', 'returning']:
    sub = new_sen[new_sen.company_type == ct]
    total = sub['cnt'].sum()
    print(f"\n  {ct} (total known seniority: {total}):")
    for _, row in sub.iterrows():
        pct = row['cnt'] / total * 100 if total > 0 else 0
        print(f"    {row['seniority_final']:15s} {int(row['cnt']):>5d} ({pct:.1f}%)")

# Also with seniority_llm
print("\nSeniority profile of new vs returning companies (seniority_llm, labeled, excl unknown):")
new_sen_llm = con.execute(f"""
    SELECT
        CASE WHEN company_name_canonical IN (
            SELECT DISTINCT company_name_canonical
            FROM '{PARQUET}'
            WHERE source = 'kaggle_arshkon'
              AND {BASE_FILTER}
              AND company_name_canonical IS NOT NULL
        ) THEN 'returning' ELSE 'new_entrant' END as company_type,
        seniority_llm,
        COUNT(*) as cnt
    FROM '{PARQUET}'
    WHERE {BASE_FILTER}
      AND source = 'scraped'
      AND company_name_canonical IS NOT NULL
      AND seniority_llm IS NOT NULL
      AND seniority_llm != 'unknown'
      AND llm_classification_coverage = 'labeled'
    GROUP BY company_type, seniority_llm
    ORDER BY company_type, seniority_llm
""").fetchdf()

for ct in ['new_entrant', 'returning']:
    sub = new_sen_llm[new_sen_llm.company_type == ct]
    total = sub['cnt'].sum()
    print(f"\n  {ct} (total known seniority: {total}):")
    for _, row in sub.iterrows():
        pct = row['cnt'] / total * 100 if total > 0 else 0
        print(f"    {row['seniority_llm']:15s} {int(row['cnt']):>5d} ({pct:.1f}%)")

# Posting volume of new entrants
new_entrant_counts = company_counts[(company_counts.source == 'scraped') &
                                     (company_counts.company_name_canonical.isin(new_entrants))]
new_total = new_entrant_counts['cnt'].sum()
scr_total = company_counts[company_counts.source == 'scraped']['cnt'].sum()
print(f"\nNew entrant posting volume: {new_total}/{scr_total} = {new_total/scr_total*100:.1f}% of scraped postings")

returning_counts = company_counts[(company_counts.source == 'scraped') &
                                   (company_counts.company_name_canonical.isin(returning))]
ret_total = returning_counts['cnt'].sum()
print(f"Returning company posting volume: {ret_total}/{scr_total} = {ret_total/scr_total*100:.1f}% of scraped postings")

# ============================================================
# Figures
# ============================================================

# Figure 1: Concentration Lorenz curves
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#2196F3', '#FF9800', '#4CAF50']
for s, c in zip(sources, colors):
    sub = company_counts[company_counts.source == s].sort_values('cnt', ascending=True)
    counts = sub['cnt'].values
    cum_share = np.cumsum(counts) / counts.sum()
    company_pct = np.arange(1, len(counts)+1) / len(counts)
    ax.plot(company_pct, cum_share, label=source_labels[s], color=c, linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect equality')
ax.set_xlabel('Cumulative share of companies')
ax.set_ylabel('Cumulative share of postings')
ax.set_title('SWE Posting Concentration (Lorenz Curves)')
ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/lorenz_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: lorenz_curves.png")

# Figure 2: New vs returning company seniority profile
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
seniority_order = ['entry', 'associate', 'mid-senior', 'director']

for ax, (df, title) in zip(axes, [(new_sen, 'seniority_final'), (new_sen_llm, 'seniority_llm')]):
    for i, ct in enumerate(['returning', 'new_entrant']):
        sub = df[df.company_type == ct]
        total = sub['cnt'].sum()
        vals = []
        for sen in seniority_order:
            row = sub[sub.iloc[:, 1] == sen]
            vals.append(row['cnt'].values[0] / total * 100 if len(row) > 0 and total > 0 else 0)
        x = np.arange(len(seniority_order))
        ax.bar(x + i*0.35, vals, 0.35,
               label=ct.replace('_', ' ').title(),
               color=['#2196F3', '#FF9800'][i], alpha=0.8)
    ax.set_xticks(x + 0.175)
    ax.set_xticklabels(seniority_order)
    ax.set_ylabel('Share (%)')
    ax.set_title(f'Scraped 2026: New vs Returning ({title})')
    ax.legend()

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/new_vs_returning_seniority.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: new_vs_returning_seniority.png")

con.close()
print("\n=== T06 Complete ===")
