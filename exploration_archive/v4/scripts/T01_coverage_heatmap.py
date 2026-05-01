"""T01: Generate coverage heatmap for columns x sources."""
import duckdb
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

con = duckdb.connect()

# Key analysis columns grouped by category
column_groups = {
    'Text': [
        ('description', 'description'),
        ('description_core', 'description_core'),
        ('description_core_llm', 'description_core_llm'),
    ],
    'Seniority': [
        ('seniority_native', 'seniority_native'),
        ('seniority_final (known)', 'seniority_final_known'),
        ('seniority_llm (known)', 'seniority_llm_known'),
        ('seniority_3level (known)', 'seniority_3level_known'),
    ],
    'Company': [
        ('company_name', 'company_name'),
        ('company_industry', 'company_industry'),
        ('company_size', 'company_size'),
        ('company_name_canonical', 'company_name_canonical'),
    ],
    'Geography': [
        ('metro_area', 'metro_area'),
        ('state_normalized', 'state_normalized'),
        ('is_remote', 'is_remote'),
    ],
    'Time': [
        ('date_posted', 'date_posted'),
        ('period', 'period'),
        ('scrape_date', 'scrape_date'),
    ],
    'Requirements': [
        ('yoe_extracted', 'yoe_extracted'),
        ('skills_raw', 'skills_raw'),
    ],
    'Quality/Ghost': [
        ('ghost_job_risk', 'ghost_job_risk'),
        ('ghost_assessment_llm', 'ghost_assessment_llm'),
        ('description_quality_flag', 'desc_quality'),
    ],
    'LLM Coverage': [
        ('llm_extraction_coverage', 'llm_ext_cov'),
        ('llm_classification_coverage', 'llm_cls_cov'),
        ('swe_classification_llm', 'swe_cls_llm'),
    ],
}

# Build query for SWE rows with default filters
cases = []
for cat, cols in column_groups.items():
    for label, col_key in cols:
        real_col = label  # default
        if col_key == 'seniority_final_known':
            cases.append(f"round(100.0 * sum(CASE WHEN seniority_final != 'unknown' THEN 1 ELSE 0 END) / count(*), 1) AS \"{label}\"")
        elif col_key == 'seniority_llm_known':
            cases.append(f"round(100.0 * sum(CASE WHEN seniority_llm IS NOT NULL AND seniority_llm != '' AND seniority_llm != 'unknown' THEN 1 ELSE 0 END) / count(*), 1) AS \"{label}\"")
        elif col_key == 'seniority_3level_known':
            cases.append(f"round(100.0 * sum(CASE WHEN seniority_3level != 'unknown' THEN 1 ELSE 0 END) / count(*), 1) AS \"{label}\"")
        elif col_key in ('is_remote',):
            cases.append(f"round(100.0 * sum(CASE WHEN {label} THEN 1 ELSE 0 END) / count(*), 1) AS \"{label}\"")
        elif col_key in ('yoe_extracted', 'company_size'):
            cases.append(f"round(100.0 * sum(CASE WHEN {label} IS NOT NULL THEN 1 ELSE 0 END) / count(*), 1) AS \"{label}\"")
        else:
            cases.append(f"round(100.0 * sum(CASE WHEN \"{label}\" IS NOT NULL AND \"{label}\" != '' THEN 1 ELSE 0 END) / count(*), 1) AS \"{label}\"")

q = f"""
SELECT source, count(*) AS total, {', '.join(cases)}
FROM 'data/unified.parquet'
WHERE source_platform='linkedin' AND is_english AND date_flag='ok' AND is_swe
GROUP BY source ORDER BY source
"""
df = con.sql(q).df().set_index('source')
total = df['total']
df = df.drop(columns=['total'])

# Build matrix for heatmap
labels = []
for cat, cols in column_groups.items():
    for label, _ in cols:
        labels.append(label)

sources = ['kaggle_arshkon', 'kaggle_asaniczka', 'scraped']
matrix = df.loc[sources, labels].values.astype(float).T

# Plot
fig, ax = plt.subplots(figsize=(8, 12))

# Color scheme: green for high coverage, yellow for partial, red for low
cmap = plt.cm.RdYlGn

im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100)

# Ticks
ax.set_xticks(range(len(sources)))
ax.set_xticklabels(['Arshkon\n(2024-04)', 'Asaniczka\n(2024-01)', 'Scraped\n(2026-03+)'], fontsize=10)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)

# Add category separators
row_idx = 0
for cat, cols in column_groups.items():
    if row_idx > 0:
        ax.axhline(y=row_idx - 0.5, color='black', linewidth=1.5)
    # Add category label
    mid = row_idx + len(cols) / 2 - 0.5
    ax.text(-0.7, mid, cat, ha='right', va='center', fontsize=7, fontweight='bold',
            transform=ax.get_yaxis_transform())
    row_idx += len(cols)

# Annotate cells with values
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        color = 'white' if val < 30 or val > 85 else 'black'
        ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=7, color=color)

# Add sample sizes in column headers
for j, src in enumerate(sources):
    ax.text(j, -1.3, f'n={int(total[src]):,}', ha='center', va='center', fontsize=8, color='gray')

plt.colorbar(im, ax=ax, label='Coverage %', shrink=0.6)
ax.set_title('Column Coverage by Source (SWE rows, LinkedIn/English/date_ok)', fontsize=11, pad=20)
plt.tight_layout()
plt.savefig('exploration/figures/T01/coverage_heatmap.png', dpi=150, bbox_inches='tight')
print('Saved heatmap to exploration/figures/T01/coverage_heatmap.png')

# Also save the data as CSV
df_out = pd.DataFrame(matrix, index=labels, columns=sources)
df_out.to_csv('exploration/tables/T01/coverage_heatmap_data.csv')
print('Saved CSV to exploration/tables/T01/coverage_heatmap_data.csv')
