#!/usr/bin/env python3
"""T06: Company concentration analysis.

Assess whether a few employers dominate SWE postings and could bias findings.
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

BASE = Path('/home/jihgaboot/gabor/job-research')
PARQUET = str(BASE / 'preprocessing/intermediate/stage8_final.parquet')
FIG_DIR = BASE / 'exploration/figures/T06'
TAB_DIR = BASE / 'exploration/tables/T06'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")

BASE_FILTER = """
    source_platform = 'linkedin'
    AND is_english = true
    AND date_flag = 'ok'
    AND is_swe = true
"""

results = {}

# ============================================================
# 1. PER-DATASET: HHI, Top-1/5/10/20 share, Gini
# ============================================================
print("=== 1. Concentration Metrics by Dataset ===")

sources = ['kaggle_arshkon', 'kaggle_asaniczka', 'scraped']
concentration = []

for src in sources:
    company_counts = con.execute(f"""
        SELECT company_name_canonical, COUNT(*) as n
        FROM parquet_scan('{PARQUET}')
        WHERE {BASE_FILTER}
          AND source = '{src}'
          AND company_name_canonical IS NOT NULL
          AND company_name_canonical != ''
        GROUP BY company_name_canonical
        ORDER BY n DESC
    """).fetchdf()

    total = company_counts['n'].sum()
    shares = (company_counts['n'] / total).values
    n_companies = len(shares)

    # HHI
    hhi = (shares ** 2).sum()

    # Top-k shares
    top1 = shares[:1].sum() * 100
    top5 = shares[:5].sum() * 100
    top10 = shares[:10].sum() * 100
    top20 = shares[:20].sum() * 100

    # Gini coefficient
    sorted_shares = np.sort(shares)
    n = len(sorted_shares)
    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_shares) - (n + 1) * np.sum(sorted_shares)) / (n * np.sum(sorted_shares))

    # Effective number of firms (1/HHI)
    eff_n = 1 / hhi if hhi > 0 else 0

    concentration.append({
        'source': src,
        'total_postings': total,
        'n_companies': n_companies,
        'hhi': round(hhi, 6),
        'effective_n_firms': round(eff_n, 1),
        'top1_share': round(top1, 2),
        'top5_share': round(top5, 2),
        'top10_share': round(top10, 2),
        'top20_share': round(top20, 2),
        'gini': round(gini, 4),
        'top1_company': company_counts.iloc[0]['company_name_canonical'],
        'top1_n': int(company_counts.iloc[0]['n'])
    })

    # Save top-50 companies per source
    top50 = company_counts.head(50).copy()
    top50['share_pct'] = (top50['n'] / total * 100).round(2)
    top50['cum_share_pct'] = top50['share_pct'].cumsum().round(2)
    top50.to_csv(TAB_DIR / f'top50_companies_{src.replace("kaggle_", "")}.csv', index=False)

conc_df = pd.DataFrame(concentration)
print(conc_df.to_string(index=False))
conc_df.to_csv(TAB_DIR / 'concentration_metrics.csv', index=False)
results['concentration'] = conc_df

# ============================================================
# 2. COMPANIES WITH >3% SWE POSTINGS
# ============================================================
print("\n=== 2. Companies with >3% SWE Postings ===")

big_companies = []
for src in sources:
    total = con.execute(f"""
        SELECT COUNT(*) FROM parquet_scan('{PARQUET}')
        WHERE {BASE_FILTER} AND source = '{src}'
    """).fetchone()[0]

    r = con.execute(f"""
        SELECT company_name_canonical, COUNT(*) as n
        FROM parquet_scan('{PARQUET}')
        WHERE {BASE_FILTER}
          AND source = '{src}'
          AND company_name_canonical IS NOT NULL
        GROUP BY company_name_canonical
        HAVING COUNT(*) > {total * 0.03}
        ORDER BY n DESC
    """).fetchdf()

    for _, row in r.iterrows():
        big_companies.append({
            'source': src,
            'company': row['company_name_canonical'],
            'n': int(row['n']),
            'share_pct': round(row['n'] / total * 100, 2)
        })

big_df = pd.DataFrame(big_companies)
print(big_df.to_string(index=False))
big_df.to_csv(TAB_DIR / 'companies_above_3pct.csv', index=False)
results['big_companies'] = big_df

# ============================================================
# 3. OVERLAP SET: within-company seniority comparison
# ============================================================
print("\n=== 3. Within-Company Seniority Comparison (Overlap Set) ===")

# Find companies present in both arshkon and scraped
overlap_companies = con.execute(f"""
    WITH arsh AS (
        SELECT DISTINCT company_name_canonical
        FROM parquet_scan('{PARQUET}')
        WHERE {BASE_FILTER} AND source = 'kaggle_arshkon'
          AND company_name_canonical IS NOT NULL
    ),
    scra AS (
        SELECT DISTINCT company_name_canonical
        FROM parquet_scan('{PARQUET}')
        WHERE {BASE_FILTER} AND source = 'scraped'
          AND company_name_canonical IS NOT NULL
    )
    SELECT arsh.company_name_canonical
    FROM arsh INNER JOIN scra ON arsh.company_name_canonical = scra.company_name_canonical
""").fetchdf()

overlap_set = set(overlap_companies['company_name_canonical'])
print(f"Companies in both arshkon and scraped: {len(overlap_set)}")

# Get seniority distributions for overlap companies
overlap_sen = con.execute(f"""
    SELECT source, seniority_final, seniority_3level, COUNT(*) as n
    FROM parquet_scan('{PARQUET}')
    WHERE {BASE_FILTER}
      AND company_name_canonical IN (
          SELECT a.company_name_canonical
          FROM (SELECT DISTINCT company_name_canonical FROM parquet_scan('{PARQUET}')
                WHERE {BASE_FILTER} AND source = 'kaggle_arshkon' AND company_name_canonical IS NOT NULL) a
          INNER JOIN (SELECT DISTINCT company_name_canonical FROM parquet_scan('{PARQUET}')
                      WHERE {BASE_FILTER} AND source = 'scraped' AND company_name_canonical IS NOT NULL) s
          ON a.company_name_canonical = s.company_name_canonical
      )
      AND source IN ('kaggle_arshkon', 'scraped')
      AND seniority_final != 'unknown'
    GROUP BY source, seniority_final, seniority_3level
    ORDER BY source, seniority_final
""").fetchdf()

overlap_pivot = overlap_sen.pivot_table(index='seniority_final', columns='source', values='n', fill_value=0)
overlap_shares = overlap_pivot.div(overlap_pivot.sum(axis=0), axis=1) * 100

print("\nSeniority distribution within overlap companies:")
print(overlap_pivot)
print("\nShares (%):")
print(overlap_shares.round(2))
overlap_shares.round(2).to_csv(TAB_DIR / 'overlap_seniority_shares.csv')

# For top-20 overlap companies, show individual seniority distributions
top_overlap = con.execute(f"""
    WITH arsh_counts AS (
        SELECT company_name_canonical, COUNT(*) as n
        FROM parquet_scan('{PARQUET}')
        WHERE {BASE_FILTER} AND source = 'kaggle_arshkon' AND company_name_canonical IS NOT NULL
        GROUP BY company_name_canonical
    ),
    scra_counts AS (
        SELECT company_name_canonical, COUNT(*) as n
        FROM parquet_scan('{PARQUET}')
        WHERE {BASE_FILTER} AND source = 'scraped' AND company_name_canonical IS NOT NULL
        GROUP BY company_name_canonical
    )
    SELECT a.company_name_canonical,
           a.n as arshkon_n,
           s.n as scraped_n,
           a.n + s.n as total_n
    FROM arsh_counts a
    INNER JOIN scra_counts s ON a.company_name_canonical = s.company_name_canonical
    ORDER BY total_n DESC
    LIMIT 20
""").fetchdf()

print("\nTop-20 overlap companies by total postings:")
print(top_overlap.to_string(index=False))
top_overlap.to_csv(TAB_DIR / 'top_overlap_companies.csv', index=False)

# Per-company seniority comparison for top overlap companies
top_overlap_names = top_overlap['company_name_canonical'].tolist()
names_sql = "','".join([n.replace("'", "''") for n in top_overlap_names])

per_company_sen = con.execute(f"""
    SELECT company_name_canonical, source, seniority_3level, COUNT(*) as n
    FROM parquet_scan('{PARQUET}')
    WHERE {BASE_FILTER}
      AND company_name_canonical IN ('{names_sql}')
      AND source IN ('kaggle_arshkon', 'scraped')
      AND seniority_3level != 'unknown'
    GROUP BY company_name_canonical, source, seniority_3level
    ORDER BY company_name_canonical, source, seniority_3level
""").fetchdf()

# Compute junior share per company per period
per_co_pivot = per_company_sen.pivot_table(
    index=['company_name_canonical', 'source'],
    columns='seniority_3level', values='n', fill_value=0
)
per_co_pivot['total'] = per_co_pivot.sum(axis=1)
for col in ['junior', 'mid', 'senior']:
    if col in per_co_pivot.columns:
        per_co_pivot[f'{col}_pct'] = (per_co_pivot[col] / per_co_pivot['total'] * 100).round(1)

per_co_pivot.to_csv(TAB_DIR / 'per_company_seniority_overlap.csv')
print("\nPer-company seniority (top overlap companies):")
print(per_co_pivot.to_string())

# ============================================================
# 4. COMPANY-CAPPED SENSITIVITY
# ============================================================
print("\n=== 4. Company-Capped Sensitivity (10 postings/company) ===")

# For each source, compute junior share with and without capping
for src in sources:
    # Uncapped
    uncapped = con.execute(f"""
        SELECT seniority_3level, COUNT(*) as n
        FROM parquet_scan('{PARQUET}')
        WHERE {BASE_FILTER} AND source = '{src}' AND seniority_3level != 'unknown'
        GROUP BY seniority_3level
    """).fetchdf()
    total_uncapped = uncapped['n'].sum()
    junior_uncapped = uncapped[uncapped['seniority_3level'] == 'junior']['n'].sum()

    # Capped at 10 per company
    capped = con.execute(f"""
        WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY company_name_canonical ORDER BY uid) as rn
            FROM parquet_scan('{PARQUET}')
            WHERE {BASE_FILTER} AND source = '{src}' AND seniority_3level != 'unknown'
        )
        SELECT seniority_3level, COUNT(*) as n
        FROM ranked
        WHERE rn <= 10
        GROUP BY seniority_3level
    """).fetchdf()
    total_capped = capped['n'].sum()
    junior_capped = capped[capped['seniority_3level'] == 'junior']['n'].sum()

    print(f"  {src}:")
    print(f"    Uncapped: junior={junior_uncapped}/{total_uncapped} ({junior_uncapped/total_uncapped*100:.1f}%)")
    print(f"    Capped@10: junior={junior_capped}/{total_capped} ({junior_capped/total_capped*100:.1f}%)")

# Also do capped at 5 and 20
cap_results = []
for cap in [5, 10, 20, 50]:
    for src in sources:
        capped = con.execute(f"""
            WITH ranked AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY company_name_canonical ORDER BY uid) as rn
                FROM parquet_scan('{PARQUET}')
                WHERE {BASE_FILTER} AND source = '{src}' AND seniority_3level != 'unknown'
            )
            SELECT seniority_3level, COUNT(*) as n
            FROM ranked
            WHERE rn <= {cap}
            GROUP BY seniority_3level
        """).fetchdf()
        total_c = capped['n'].sum()
        for _, row in capped.iterrows():
            cap_results.append({
                'source': src,
                'cap': cap,
                'seniority_3level': row['seniority_3level'],
                'n': int(row['n']),
                'share_pct': round(row['n'] / total_c * 100, 2)
            })

cap_df = pd.DataFrame(cap_results)
cap_pivot = cap_df.pivot_table(index=['source', 'cap'], columns='seniority_3level', values='share_pct')
print("\nJunior share under different caps:")
print(cap_pivot.to_string())
cap_pivot.to_csv(TAB_DIR / 'capped_seniority_shares.csv')
results['capped'] = cap_pivot

# ============================================================
# 5. AGGREGATOR ANALYSIS
# ============================================================
print("\n=== 5. Aggregator Analysis ===")

agg_stats = con.execute(f"""
    SELECT source, is_aggregator,
           COUNT(*) as n,
           AVG(description_length) as mean_desc_len,
           AVG(core_length) as mean_core_len,
           AVG(yoe_extracted) as mean_yoe
    FROM parquet_scan('{PARQUET}')
    WHERE {BASE_FILTER}
    GROUP BY source, is_aggregator
    ORDER BY source, is_aggregator
""").fetchdf()

print("Aggregator vs non-aggregator stats:")
print(agg_stats.to_string(index=False))
agg_stats.to_csv(TAB_DIR / 'aggregator_stats.csv', index=False)

# Aggregator seniority
agg_sen = con.execute(f"""
    SELECT source, is_aggregator, seniority_3level, COUNT(*) as n
    FROM parquet_scan('{PARQUET}')
    WHERE {BASE_FILTER}
      AND seniority_3level != 'unknown'
    GROUP BY source, is_aggregator, seniority_3level
    ORDER BY source, is_aggregator, seniority_3level
""").fetchdf()

agg_sen_pivot = agg_sen.pivot_table(
    index=['source', 'is_aggregator'],
    columns='seniority_3level', values='n', fill_value=0
)
agg_sen_pivot['total'] = agg_sen_pivot.sum(axis=1)
for col in ['junior', 'mid', 'senior']:
    if col in agg_sen_pivot.columns:
        agg_sen_pivot[f'{col}_pct'] = (agg_sen_pivot[col] / agg_sen_pivot['total'] * 100).round(1)

print("\nAggregator seniority distribution:")
print(agg_sen_pivot.to_string())
agg_sen_pivot.to_csv(TAB_DIR / 'aggregator_seniority.csv')
results['agg_seniority'] = agg_sen_pivot

# Aggregator fraction by source
agg_frac = con.execute(f"""
    SELECT source,
           COUNT(*) as total,
           SUM(CASE WHEN is_aggregator THEN 1 ELSE 0 END) as agg_n,
           ROUND(100.0 * SUM(CASE WHEN is_aggregator THEN 1 ELSE 0 END) / COUNT(*), 1) as agg_pct
    FROM parquet_scan('{PARQUET}')
    WHERE {BASE_FILTER}
    GROUP BY source
""").fetchdf()
print("\nAggregator fraction:")
print(agg_frac.to_string(index=False))
results['agg_frac'] = agg_frac

# Top aggregators by source
for src in sources:
    r = con.execute(f"""
        SELECT company_name, COUNT(*) as n
        FROM parquet_scan('{PARQUET}')
        WHERE {BASE_FILTER}
          AND source = '{src}'
          AND is_aggregator = true
        GROUP BY company_name
        ORDER BY n DESC
        LIMIT 10
    """).fetchdf()
    print(f"\nTop aggregators in {src}:")
    print(r.to_string(index=False))

# ============================================================
# 6. NEW ENTRANTS (2026 scraped companies not in 2024 arshkon)
# ============================================================
print("\n=== 6. New Entrants Analysis ===")

new_entrants = con.execute(f"""
    WITH arsh_companies AS (
        SELECT DISTINCT company_name_canonical
        FROM parquet_scan('{PARQUET}')
        WHERE {BASE_FILTER} AND source = 'kaggle_arshkon' AND company_name_canonical IS NOT NULL
    ),
    scraped_companies AS (
        SELECT company_name_canonical, COUNT(*) as n
        FROM parquet_scan('{PARQUET}')
        WHERE {BASE_FILTER} AND source = 'scraped' AND company_name_canonical IS NOT NULL
        GROUP BY company_name_canonical
    )
    SELECT sc.company_name_canonical, sc.n,
           CASE WHEN ac.company_name_canonical IS NULL THEN 'new' ELSE 'returning' END as status
    FROM scraped_companies sc
    LEFT JOIN arsh_companies ac ON sc.company_name_canonical = ac.company_name_canonical
""").fetchdf()

new_count = (new_entrants['status'] == 'new').sum()
returning_count = (new_entrants['status'] == 'returning').sum()
new_postings = new_entrants[new_entrants['status'] == 'new']['n'].sum()
returning_postings = new_entrants[new_entrants['status'] == 'returning']['n'].sum()

print(f"Companies in 2026 scraped: {len(new_entrants)}")
print(f"  New (not in arshkon): {new_count} ({new_count/len(new_entrants)*100:.1f}%)")
print(f"  Returning: {returning_count} ({returning_count/len(new_entrants)*100:.1f}%)")
print(f"  New company postings: {new_postings} ({new_postings/(new_postings+returning_postings)*100:.1f}%)")
print(f"  Returning company postings: {returning_postings} ({returning_postings/(new_postings+returning_postings)*100:.1f}%)")

# Top new entrants
new_top = new_entrants[new_entrants['status'] == 'new'].nlargest(30, 'n')
print("\nTop-30 new entrant companies:")
print(new_top.to_string(index=False))
new_top.to_csv(TAB_DIR / 'new_entrants_top30.csv', index=False)

# New entrants by industry
new_industry = con.execute(f"""
    WITH arsh_companies AS (
        SELECT DISTINCT company_name_canonical
        FROM parquet_scan('{PARQUET}')
        WHERE {BASE_FILTER} AND source = 'kaggle_arshkon' AND company_name_canonical IS NOT NULL
    )
    SELECT company_industry, COUNT(*) as n_postings, COUNT(DISTINCT company_name_canonical) as n_companies
    FROM parquet_scan('{PARQUET}')
    WHERE {BASE_FILTER}
      AND source = 'scraped'
      AND company_name_canonical IS NOT NULL
      AND company_name_canonical NOT IN (SELECT company_name_canonical FROM arsh_companies)
      AND company_industry IS NOT NULL
      AND company_industry != ''
    GROUP BY company_industry
    ORDER BY n_postings DESC
    LIMIT 20
""").fetchdf()
print("\nNew entrant industries:")
print(new_industry.to_string(index=False))
new_industry.to_csv(TAB_DIR / 'new_entrants_industry.csv', index=False)
results['new_industry'] = new_industry

# New entrants seniority distribution
new_sen = con.execute(f"""
    WITH arsh_companies AS (
        SELECT DISTINCT company_name_canonical
        FROM parquet_scan('{PARQUET}')
        WHERE {BASE_FILTER} AND source = 'kaggle_arshkon' AND company_name_canonical IS NOT NULL
    )
    SELECT
        CASE WHEN company_name_canonical NOT IN (SELECT company_name_canonical FROM arsh_companies) THEN 'new' ELSE 'returning' END as status,
        seniority_3level,
        COUNT(*) as n
    FROM parquet_scan('{PARQUET}')
    WHERE {BASE_FILTER}
      AND source = 'scraped'
      AND company_name_canonical IS NOT NULL
      AND seniority_3level != 'unknown'
    GROUP BY status, seniority_3level
""").fetchdf()

new_sen_pivot = new_sen.pivot_table(index='seniority_3level', columns='status', values='n', fill_value=0)
new_sen_shares = new_sen_pivot.div(new_sen_pivot.sum(axis=0), axis=1) * 100
print("\nSeniority by new vs returning companies (2026):")
print(new_sen_shares.round(2))
new_sen_shares.round(2).to_csv(TAB_DIR / 'new_entrants_seniority.csv')

# ============================================================
# FIGURES
# ============================================================

# Figure 1: Lorenz curves
fig, ax = plt.subplots(figsize=(8, 6))
for src in sources:
    company_counts = con.execute(f"""
        SELECT company_name_canonical, COUNT(*) as n
        FROM parquet_scan('{PARQUET}')
        WHERE {BASE_FILTER} AND source = '{src}'
          AND company_name_canonical IS NOT NULL
        GROUP BY company_name_canonical
        ORDER BY n ASC
    """).fetchdf()

    vals = company_counts['n'].values
    cum = np.cumsum(vals) / vals.sum()
    x = np.arange(1, len(cum)+1) / len(cum)
    label = src.replace('kaggle_', '')
    ax.plot(x, cum, label=f"{label} (n={len(vals):,}, Gini={1 - 2*np.trapz(cum, x):.3f})")

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect equality')
ax.set_xlabel('Cumulative share of companies')
ax.set_ylabel('Cumulative share of postings')
ax.set_title('Lorenz Curves: SWE Posting Concentration by Company')
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(FIG_DIR / 'lorenz_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved lorenz_curves.png")

# Figure 2: Capped junior share sensitivity
fig, ax = plt.subplots(figsize=(8, 5))
cap_junior = cap_df[cap_df['seniority_3level'] == 'junior'].copy()
# Also add uncapped
for src in sources:
    uncapped_r = con.execute(f"""
        SELECT seniority_3level, COUNT(*) as n
        FROM parquet_scan('{PARQUET}')
        WHERE {BASE_FILTER} AND source = '{src}' AND seniority_3level != 'unknown'
        GROUP BY seniority_3level
    """).fetchdf()
    total_u = uncapped_r['n'].sum()
    jr_u = uncapped_r[uncapped_r['seniority_3level'] == 'junior']['n'].sum()
    cap_junior = pd.concat([cap_junior, pd.DataFrame([{
        'source': src, 'cap': 9999, 'seniority_3level': 'junior',
        'n': jr_u, 'share_pct': round(jr_u / total_u * 100, 2)
    }])], ignore_index=True)

for src in sources:
    subset = cap_junior[cap_junior['source'] == src].sort_values('cap')
    label = src.replace('kaggle_', '')
    caps = subset['cap'].values
    caps_labels = [str(c) if c < 9999 else 'None' for c in caps]
    ax.plot(range(len(caps)), subset['share_pct'].values, 'o-', label=label)
    ax.set_xticks(range(len(caps)))
    ax.set_xticklabels(caps_labels)

ax.set_xlabel('Company cap (max postings per company)')
ax.set_ylabel('Junior share (%)')
ax.set_title('Junior Share Sensitivity to Company Capping')
ax.legend()

plt.tight_layout()
plt.savefig(FIG_DIR / 'capped_junior_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved capped_junior_sensitivity.png")

# Figure 3: Aggregator comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Aggregator fraction by source
agg_frac_vals = results['agg_frac']
axes[0].bar(range(len(agg_frac_vals)), agg_frac_vals['agg_pct'].values)
axes[0].set_xticks(range(len(agg_frac_vals)))
axes[0].set_xticklabels([s.replace('kaggle_', '') for s in agg_frac_vals['source']])
axes[0].set_ylabel('Aggregator share (%)')
axes[0].set_title('Aggregator Fraction of SWE Postings')

# Seniority by aggregator status (scraped only, as example)
scraped_agg = agg_sen[agg_sen['source'] == 'scraped'].copy()
scraped_agg_piv = scraped_agg.pivot_table(index='seniority_3level', columns='is_aggregator', values='n', fill_value=0)
scraped_agg_piv_share = scraped_agg_piv.div(scraped_agg_piv.sum(axis=0), axis=1) * 100
scraped_agg_piv_share = scraped_agg_piv_share.reindex(['junior', 'mid', 'senior'])
scraped_agg_piv_share.columns = ['Direct', 'Aggregator']
scraped_agg_piv_share.plot(kind='bar', ax=axes[1], rot=0)
axes[1].set_ylabel('Share (%)')
axes[1].set_title('Seniority by Posting Type (Scraped 2026)')

plt.tight_layout()
plt.savefig(FIG_DIR / 'aggregator_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved aggregator_analysis.png")

print("\n=== T06 COMPLETE ===")
