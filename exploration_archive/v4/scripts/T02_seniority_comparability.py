"""T02: Seniority comparability analysis and figures."""
import duckdb
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

con = duckdb.connect()

# Figure 1: YOE distributions comparison
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

groups = [
    ('Arshkon Entry', "source='kaggle_arshkon' AND seniority_native='entry'"),
    ('Arshkon Associate', "source='kaggle_arshkon' AND seniority_native='associate'"),
    ('Asaniczka Associate', "source='kaggle_asaniczka' AND seniority_native='associate'"),
    ('Arshkon Mid-Senior', "source='kaggle_arshkon' AND seniority_native='mid-senior'"),
]

for ax, (name, filt) in zip(axes, groups):
    q = f"""
    SELECT yoe_extracted
    FROM 'data/unified.parquet'
    WHERE {filt} AND is_swe AND source_platform='linkedin' AND is_english AND date_flag='ok'
      AND yoe_extracted IS NOT NULL
    """
    yoe = con.sql(q).df()['yoe_extracted'].values
    if len(yoe) > 0:
        bins = np.arange(0, 16, 1)
        counts, _ = np.histogram(yoe, bins=bins)
        pcts = counts / counts.sum() * 100
        ax.bar(bins[:-1] + 0.5, pcts, width=0.8, alpha=0.7, color='steelblue')
    ax.set_title(name, fontsize=9)
    ax.set_xlabel('YOE')
    ax.set_xlim(0, 15)
    n_total = len(yoe) if len(yoe) > 0 else 0
    med = np.median(yoe) if len(yoe) > 0 else 0
    ax.text(0.95, 0.95, f'n={n_total}\nmed={med:.0f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color='gray')

axes[0].set_ylabel('% of postings with YOE')
fig.suptitle('YOE Distributions by Source x Native Seniority (SWE only)', fontsize=11)
plt.tight_layout()
plt.savefig('exploration/figures/T02/yoe_comparison.png', dpi=150, bbox_inches='tight')
print('Saved YOE comparison figure')

# Figure 2: seniority_final conditional distribution
fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))

cross_groups = [
    ('Arshkon Entry\n(native)', "source='kaggle_arshkon' AND seniority_native='entry'"),
    ('Arshkon Associate\n(native)', "source='kaggle_arshkon' AND seniority_native='associate'"),
    ('Asaniczka Associate\n(native)', "source='kaggle_asaniczka' AND seniority_native='associate'"),
]

colors = {'entry': '#4CAF50', 'associate': '#2196F3', 'mid-senior': '#FF9800', 'director': '#F44336', 'unknown': '#9E9E9E'}

for ax, (name, filt) in zip(axes2, cross_groups):
    q = f"""
    SELECT seniority_final, count(*) n
    FROM 'data/unified.parquet'
    WHERE {filt} AND is_swe AND source_platform='linkedin' AND is_english AND date_flag='ok'
    GROUP BY seniority_final ORDER BY n DESC
    """
    df = con.sql(q).df()
    total = df['n'].sum()
    df['pct'] = df['n'] / total * 100
    
    bars = ax.bar(range(len(df)), df['pct'], color=[colors.get(s, '#9E9E9E') for s in df['seniority_final']])
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['seniority_final'], fontsize=8, rotation=30)
    ax.set_title(name, fontsize=9)
    ax.set_ylabel('% of rows')
    
    for i, (pct, n) in enumerate(zip(df['pct'], df['n'])):
        ax.text(i, pct + 1, f'{pct:.0f}%\n({n})', ha='center', va='bottom', fontsize=7)

fig2.suptitle('seniority_final Distribution Conditional on seniority_native (SWE)', fontsize=11)
plt.tight_layout()
plt.savefig('exploration/figures/T02/seniority_final_conditional.png', dpi=150, bbox_inches='tight')
print('Saved seniority_final conditional figure')

# Save comparability audit table
audit_data = []
for name, filt in [
    ('arshkon_entry', "source='kaggle_arshkon' AND seniority_native='entry'"),
    ('arshkon_associate', "source='kaggle_arshkon' AND seniority_native='associate'"),
    ('asaniczka_associate', "source='kaggle_asaniczka' AND seniority_native='associate'"),
    ('arshkon_mid_senior', "source='kaggle_arshkon' AND seniority_native='mid-senior'"),
]:
    q = f"""
    SELECT 
      count(*) AS n,
      sum(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS has_yoe,
      round(avg(yoe_extracted), 2) AS mean_yoe,
      round(median(yoe_extracted), 1) AS median_yoe,
      round(percentile_cont(0.25) WITHIN GROUP (ORDER BY yoe_extracted), 1) AS p25_yoe,
      round(percentile_cont(0.75) WITHIN GROUP (ORDER BY yoe_extracted), 1) AS p75_yoe,
      round(100.0 * sum(CASE WHEN regexp_matches(title_normalized, 'junior|\\bjr\\b|entry.level|intern|new.grad|graduate') THEN 1 ELSE 0 END) / count(*), 1) AS pct_junior_cue,
      round(100.0 * sum(CASE WHEN regexp_matches(title_normalized, 'senior|\\bsr\\b|lead|principal|staff|architect') THEN 1 ELSE 0 END) / count(*), 1) AS pct_senior_cue,
      round(100.0 * sum(CASE WHEN seniority_final = 'entry' THEN 1 ELSE 0 END) / count(*), 1) AS pct_final_entry,
      round(100.0 * sum(CASE WHEN seniority_final = 'associate' THEN 1 ELSE 0 END) / count(*), 1) AS pct_final_associate,
      round(100.0 * sum(CASE WHEN seniority_final = 'mid-senior' THEN 1 ELSE 0 END) / count(*), 1) AS pct_final_mid
    FROM 'data/unified.parquet'
    WHERE {filt} AND is_swe AND source_platform='linkedin' AND is_english AND date_flag='ok'
    """
    row = con.sql(q).df().iloc[0].to_dict()
    row['group'] = name
    audit_data.append(row)

audit_df = pd.DataFrame(audit_data)
cols = ['group'] + [c for c in audit_df.columns if c != 'group']
audit_df = audit_df[cols]
audit_df.to_csv('exploration/tables/T02/comparability_audit.csv', index=False)
print('Saved audit table')
print(audit_df.to_string())
