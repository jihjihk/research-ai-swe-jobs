"""
T01 + T03: Column coverage audit and missing data audit
Produces coverage heatmaps, missing data tables, and CSV outputs.
Uses DuckDB for all queries (no full pandas load).
"""
import duckdb
import pandas as pd
import numpy as np
import os

DATA = 'preprocessing/intermediate/stage8_final.parquet'
OUT_T01 = 'exploration/tables/T01'
OUT_T03 = 'exploration/tables/T03'
FIG_T01 = 'exploration/figures/T01'
FIG_T03 = 'exploration/figures/T03'

os.makedirs(OUT_T01, exist_ok=True)
os.makedirs(OUT_T03, exist_ok=True)
os.makedirs(FIG_T01, exist_ok=True)
os.makedirs(FIG_T03, exist_ok=True)

con = duckdb.connect()

# Default filters
BASE_FILTER = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'"

# Get all column names
cols_df = con.execute(f"DESCRIBE SELECT * FROM '{DATA}'").df()
all_cols = list(cols_df['column_name'])
col_types = dict(zip(cols_df['column_name'], cols_df['column_type']))

print(f"Total columns: {len(all_cols)}")

# ========================================================================
# T01 Step 1: Non-null rate by source (all rows with default filters)
# ========================================================================
print("\n=== T01: Computing non-null rates by source ===")

sources = ['kaggle_arshkon', 'kaggle_asaniczka', 'scraped']

# Build a query that computes non-null rate for every column, by source
# We need to handle different column types appropriately
results_all = []
results_swe = []

for src in sources:
    for subset_label, extra_filter in [('all', ''), ('swe', ' AND is_swe = true')]:
        # Build COUNT expressions for every column
        count_exprs = []
        for c in all_cols:
            if col_types[c] in ('VARCHAR',):
                count_exprs.append(
                    f"SUM(CASE WHEN \"{c}\" IS NOT NULL AND \"{c}\" != '' THEN 1 ELSE 0 END) AS \"{c}_nonnull\""
                )
            else:
                count_exprs.append(
                    f"SUM(CASE WHEN \"{c}\" IS NOT NULL THEN 1 ELSE 0 END) AS \"{c}_nonnull\""
                )

        count_str = ', '.join(count_exprs)
        query = f"""
            SELECT COUNT(*) as total_rows, {count_str}
            FROM '{DATA}'
            WHERE {BASE_FILTER} AND source = '{src}'{extra_filter}
        """
        row = con.execute(query).df().iloc[0]
        total = row['total_rows']

        for c in all_cols:
            nonnull = row[f'{c}_nonnull']
            rate = nonnull / total if total > 0 else 0
            entry = {
                'column': c, 'source': src, 'subset': subset_label,
                'total_rows': int(total), 'nonnull_count': int(nonnull),
                'nonnull_rate': round(rate, 4)
            }
            if subset_label == 'all':
                results_all.append(entry)
            else:
                results_swe.append(entry)

coverage_all = pd.DataFrame(results_all)
coverage_swe = pd.DataFrame(results_swe)

# Save raw coverage tables
coverage_all.to_csv(f'{OUT_T01}/coverage_all_by_source.csv', index=False)
coverage_swe.to_csv(f'{OUT_T01}/coverage_swe_by_source.csv', index=False)
print(f"  Saved coverage tables to {OUT_T01}/")

# ========================================================================
# T01 Step 1b: Distinct counts and top-5 values (for key analysis columns only)
# ========================================================================
print("\n=== T01: Computing distinct counts and top-5 values ===")

# Focus on key analysis columns (not description text)
key_cols = [
    'source', 'source_platform', 'title_normalized', 'company_name',
    'company_name_effective', 'company_name_canonical', 'company_industry',
    'company_size_category', 'location', 'location_normalized',
    'city_extracted', 'state_normalized', 'country_extracted', 'metro_area',
    'seniority_raw', 'seniority_native', 'seniority_imputed', 'seniority_final',
    'seniority_final_source', 'seniority_3level',
    'is_swe', 'is_swe_adjacent', 'is_control', 'swe_classification_tier',
    'is_aggregator', 'is_remote', 'is_remote_inferred',
    'period', 'date_flag', 'is_english', 'work_type',
    'ghost_job_risk', 'description_quality_flag', 'boilerplate_flag',
    'dedup_method', 'query_tier', 'search_metro_name',
    'skills_raw', 'asaniczka_skills'
]

distinct_results = []
for src in sources:
    for c in key_cols:
        try:
            # Distinct count
            dc = con.execute(f"""
                SELECT COUNT(DISTINCT "{c}") as dc
                FROM '{DATA}'
                WHERE {BASE_FILTER} AND source = '{src}' AND "{c}" IS NOT NULL
            """).fetchone()[0]

            # Top 5 values
            top5 = con.execute(f"""
                SELECT "{c}" as val, COUNT(*) as cnt
                FROM '{DATA}'
                WHERE {BASE_FILTER} AND source = '{src}' AND "{c}" IS NOT NULL
                GROUP BY "{c}" ORDER BY cnt DESC LIMIT 5
            """).df()
            top5_str = '; '.join([f"{r['val']}({r['cnt']})" for _, r in top5.iterrows()])

            distinct_results.append({
                'column': c, 'source': src,
                'distinct_count': dc, 'top_5_values': top5_str
            })
        except Exception as e:
            distinct_results.append({
                'column': c, 'source': src,
                'distinct_count': -1, 'top_5_values': f'ERROR: {e}'
            })

distinct_df = pd.DataFrame(distinct_results)
distinct_df.to_csv(f'{OUT_T01}/distinct_counts_top5.csv', index=False)
print(f"  Saved distinct counts to {OUT_T01}/distinct_counts_top5.csv")

# ========================================================================
# T01 Step 2: Coverage heatmap
# ========================================================================
print("\n=== T01: Generating coverage heatmap ===")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Pivot coverage_all for heatmap: columns as rows, sources as columns
pivot_all = coverage_all.pivot(index='column', columns='source', values='nonnull_rate')
# Reorder columns
pivot_all = pivot_all[sources]

# Group columns by category for readability
col_groups = {
    'Identity': ['uid', 'job_id', 'source', 'source_platform', 'site'],
    'Job Content': ['title', 'title_normalized', 'description', 'description_raw',
                    'description_length', 'description_core', 'core_length', 'boilerplate_flag'],
    'Company': ['company_name', 'company_name_normalized', 'is_aggregator', 'real_employer',
                'company_name_effective', 'company_name_canonical', 'company_name_canonical_method',
                'company_industry', 'company_size', 'company_size_raw', 'company_size_category',
                'company_id_kaggle'],
    'Seniority': ['seniority_raw', 'seniority_native', 'seniority_imputed', 'seniority_source',
                  'seniority_confidence', 'seniority_final', 'seniority_final_source',
                  'seniority_final_confidence', 'yoe_extracted', 'yoe_seniority_contradiction',
                  'seniority_cross_check', 'seniority_3level'],
    'Classification': ['is_swe', 'is_swe_adjacent', 'swe_confidence', 'swe_classification_tier',
                       'is_control'],
    'Geography': ['location', 'location_normalized', 'city_extracted', 'state_normalized',
                  'country_extracted', 'is_remote', 'is_remote_inferred', 'metro_area',
                  'metro_source', 'metro_confidence'],
    'Search Meta': ['search_query', 'query_tier', 'search_metro_id', 'search_metro_name',
                    'search_metro_region', 'search_location'],
    'Temporal': ['date_posted', 'scrape_date', 'period', 'posting_age_days', 'scrape_week'],
    'Quality': ['date_flag', 'is_english', 'description_hash', 'ghost_job_risk',
                'description_quality_flag'],
    'Pipeline': ['preprocessing_version', 'dedup_method', 'boilerplate_removed',
                 'is_multi_location', 'work_type', 'job_url', 'skills_raw', 'asaniczka_skills'],
}

# Build ordered list
ordered_cols = []
group_labels = []
group_positions = []
for gname, gcols in col_groups.items():
    start = len(ordered_cols)
    for c in gcols:
        if c in pivot_all.index:
            ordered_cols.append(c)
    if len(ordered_cols) > start:
        group_positions.append((start, len(ordered_cols)-1, gname))

# Reorder
pivot_ordered = pivot_all.loc[ordered_cols]

# Create heatmap - ALL rows
fig, ax = plt.subplots(figsize=(10, 28))
data = pivot_ordered.values
cmap = plt.cm.RdYlGn  # Red=low, green=high
norm = mcolors.Normalize(vmin=0, vmax=1)

im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')
ax.set_xticks(range(len(sources)))
ax.set_xticklabels(['arshkon', 'asaniczka', 'scraped'], fontsize=10, fontweight='bold')
ax.set_yticks(range(len(ordered_cols)))
ax.set_yticklabels(ordered_cols, fontsize=6)
ax.set_xlabel('Source')
ax.set_title('T01: Column Coverage (Non-null Rate)\nAll LinkedIn+English+date_ok rows', fontsize=12, fontweight='bold')

# Add text annotations
for i in range(len(ordered_cols)):
    for j in range(len(sources)):
        val = data[i, j]
        color = 'white' if val < 0.4 or val > 0.8 else 'black'
        ax.text(j, i, f'{val:.0%}', ha='center', va='center', fontsize=5, color=color)

# Add group separators
for start, end, gname in group_positions:
    if start > 0:
        ax.axhline(y=start - 0.5, color='black', linewidth=1.5)
    ax.text(-0.7, (start + end) / 2, gname, ha='right', va='center', fontsize=6,
            fontweight='bold', rotation=0)

plt.colorbar(im, ax=ax, label='Non-null rate', shrink=0.5)
plt.tight_layout()
plt.savefig(f'{FIG_T01}/coverage_heatmap_all.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved heatmap to {FIG_T01}/coverage_heatmap_all.png")

# SWE subset heatmap
pivot_swe = coverage_swe.pivot(index='column', columns='source', values='nonnull_rate')
pivot_swe = pivot_swe[sources]
pivot_swe_ordered = pivot_swe.loc[ordered_cols]

fig, ax = plt.subplots(figsize=(10, 28))
data_swe = pivot_swe_ordered.values
im = ax.imshow(data_swe, cmap=cmap, norm=norm, aspect='auto')
ax.set_xticks(range(len(sources)))
ax.set_xticklabels(['arshkon', 'asaniczka', 'scraped'], fontsize=10, fontweight='bold')
ax.set_yticks(range(len(ordered_cols)))
ax.set_yticklabels(ordered_cols, fontsize=6)
ax.set_xlabel('Source')
ax.set_title('T01: Column Coverage (Non-null Rate)\nSWE Subset (LinkedIn+English+date_ok)', fontsize=12, fontweight='bold')

for i in range(len(ordered_cols)):
    for j in range(len(sources)):
        val = data_swe[i, j]
        color = 'white' if val < 0.4 or val > 0.8 else 'black'
        ax.text(j, i, f'{val:.0%}', ha='center', va='center', fontsize=5, color=color)

for start, end, gname in group_positions:
    if start > 0:
        ax.axhline(y=start - 0.5, color='black', linewidth=1.5)
    ax.text(-0.7, (start + end) / 2, gname, ha='right', va='center', fontsize=6,
            fontweight='bold', rotation=0)

plt.colorbar(im, ax=ax, label='Non-null rate', shrink=0.5)
plt.tight_layout()
plt.savefig(f'{FIG_T01}/coverage_heatmap_swe.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved SWE heatmap to {FIG_T01}/coverage_heatmap_swe.png")

# ========================================================================
# T01 Step 3: Flag columns >50% null for any source
# ========================================================================
print("\n=== T01: Flagging columns >50% null ===")

flagged = coverage_all[coverage_all['nonnull_rate'] < 0.5][['column', 'source', 'nonnull_rate', 'total_rows', 'nonnull_count']]
flagged = flagged.sort_values(['column', 'source'])
flagged.to_csv(f'{OUT_T01}/columns_gt50pct_null.csv', index=False)
print(f"  {len(flagged)} column-source pairs with >50% null")
print(flagged.to_string(index=False))

# ========================================================================
# T01 Step 4: Usable columns per RQ
# ========================================================================
print("\n=== T01: Usable columns per RQ ===")

# Define which columns each RQ needs
rq_columns = {
    'RQ1: Junior share/volume': {
        'critical': ['seniority_final', 'is_swe', 'period', 'source'],
        'important': ['seniority_native', 'seniority_3level', 'seniority_final_source',
                      'title_normalized', 'company_name_effective'],
        'nice_to_have': ['metro_area', 'company_industry', 'is_aggregator', 'yoe_extracted']
    },
    'RQ2: Task/requirement migration': {
        'critical': ['description', 'is_swe', 'period', 'seniority_final'],
        'important': ['description_core', 'skills_raw', 'title_normalized',
                      'is_swe_adjacent'],
        'nice_to_have': ['work_type', 'is_remote_inferred', 'metro_area']
    },
    'RQ3: Employer-requirement divergence': {
        'critical': ['description', 'is_swe', 'period', 'seniority_final'],
        'important': ['description_core', 'title_normalized', 'company_name_effective'],
        'nice_to_have': ['company_industry', 'company_size', 'metro_area']
    },
}

rq_usable = []
for rq, col_dict in rq_columns.items():
    for priority, cols in col_dict.items():
        for c in cols:
            for src in sources:
                row_match = coverage_all[(coverage_all['column'] == c) & (coverage_all['source'] == src)]
                if len(row_match) > 0:
                    rate = row_match.iloc[0]['nonnull_rate']
                    usable = 'Yes' if rate >= 0.5 else 'Marginal' if rate >= 0.1 else 'No'
                else:
                    rate = None
                    usable = 'N/A'
                rq_usable.append({
                    'RQ': rq, 'priority': priority, 'column': c,
                    'source': src, 'nonnull_rate': rate, 'usable': usable
                })

rq_df = pd.DataFrame(rq_usable)
rq_df.to_csv(f'{OUT_T01}/usable_columns_per_rq.csv', index=False)
print(f"  Saved to {OUT_T01}/usable_columns_per_rq.csv")

# Print summary
for rq in rq_columns:
    print(f"\n  {rq}:")
    sub = rq_df[rq_df['RQ'] == rq]
    for priority in ['critical', 'important', 'nice_to_have']:
        sub_p = sub[sub['priority'] == priority]
        no_cols = sub_p[sub_p['usable'] == 'No']['column'].unique()
        marginal_cols = sub_p[sub_p['usable'] == 'Marginal']['column'].unique()
        if len(no_cols) > 0:
            print(f"    {priority} - UNUSABLE for some sources: {list(no_cols)}")
        if len(marginal_cols) > 0:
            print(f"    {priority} - MARGINAL for some sources: {list(marginal_cols)}")

# ========================================================================
# T03 Step 1: Field x source x platform missing data table
# ========================================================================
print("\n\n=== T03: Missing data audit ===")

# Compute for all rows AND swe subset, by source AND platform
t03_results = []

for src in sources:
    for platform in ['linkedin']:  # Focus on linkedin per default filters
        for subset_label, extra_filter in [('all', ''), ('swe', ' AND is_swe = true')]:
            count_exprs = []
            for c in all_cols:
                if col_types[c] in ('VARCHAR',):
                    count_exprs.append(
                        f"SUM(CASE WHEN \"{c}\" IS NOT NULL AND \"{c}\" != '' THEN 1 ELSE 0 END) AS \"{c}_nonnull\""
                    )
                else:
                    count_exprs.append(
                        f"SUM(CASE WHEN \"{c}\" IS NOT NULL THEN 1 ELSE 0 END) AS \"{c}_nonnull\""
                    )
            count_str = ', '.join(count_exprs)
            query = f"""
                SELECT COUNT(*) as total_rows, {count_str}
                FROM '{DATA}'
                WHERE {BASE_FILTER} AND source = '{src}' AND source_platform = '{platform}'{extra_filter}
            """
            row = con.execute(query).df().iloc[0]
            total = row['total_rows']

            for c in all_cols:
                nonnull = row[f'{c}_nonnull']
                rate = nonnull / total if total > 0 else 0
                t03_results.append({
                    'column': c, 'source': src, 'platform': platform, 'subset': subset_label,
                    'total_rows': int(total), 'nonnull_count': int(nonnull),
                    'nonnull_pct': round(rate * 100, 2)
                })

t03_df = pd.DataFrame(t03_results)
t03_df.to_csv(f'{OUT_T03}/missing_data_full.csv', index=False)
print(f"  Saved full missing data table to {OUT_T03}/missing_data_full.csv")

# ========================================================================
# T03 Step 2: Seniority labels per source
# ========================================================================
print("\n=== T03: Seniority labels per source ===")

seniority_by_source = con.execute(f"""
    SELECT source, seniority_native, COUNT(*) as n,
           ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY source), 2) as pct
    FROM '{DATA}'
    WHERE {BASE_FILTER}
    GROUP BY source, seniority_native
    ORDER BY source, n DESC
""").df()
seniority_by_source.to_csv(f'{OUT_T03}/seniority_native_by_source.csv', index=False)
print(seniority_by_source.to_string(index=False))

# Seniority final by source (SWE only)
seniority_final_swe = con.execute(f"""
    SELECT source, seniority_final, COUNT(*) as n,
           ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY source), 2) as pct
    FROM '{DATA}'
    WHERE {BASE_FILTER} AND is_swe = true
    GROUP BY source, seniority_final
    ORDER BY source, n DESC
""").df()
seniority_final_swe.to_csv(f'{OUT_T03}/seniority_final_swe_by_source.csv', index=False)
print("\nSeniority_final (SWE subset):")
print(seniority_final_swe.to_string(index=False))

# ========================================================================
# T03 Step 3: Associate vs Entry-level title overlap
# ========================================================================
print("\n=== T03: Associate/Entry-level title overlap investigation ===")

# Get asaniczka "associate" titles (SWE)
asaniczka_assoc = con.execute(f"""
    SELECT title_normalized, COUNT(*) as n
    FROM '{DATA}'
    WHERE {BASE_FILTER} AND source = 'kaggle_asaniczka'
          AND is_swe = true AND seniority_native = 'associate'
    GROUP BY title_normalized
    ORDER BY n DESC
    LIMIT 30
""").df()
print("Asaniczka 'associate' SWE titles (top 30):")
print(asaniczka_assoc.to_string(index=False))

# Get arshkon "entry" titles (SWE)
arshkon_entry = con.execute(f"""
    SELECT title_normalized, COUNT(*) as n
    FROM '{DATA}'
    WHERE {BASE_FILTER} AND source = 'kaggle_arshkon'
          AND is_swe = true AND seniority_native = 'entry'
    GROUP BY title_normalized
    ORDER BY n DESC
    LIMIT 30
""").df()
print("\nArshkon 'entry' SWE titles (top 30):")
print(arshkon_entry.to_string(index=False))

# Overlap: titles that appear as both "associate" in asaniczka and "entry" in arshkon
overlap = con.execute(f"""
    WITH asaniczka_assoc AS (
        SELECT title_normalized, COUNT(*) as asaniczka_count
        FROM '{DATA}'
        WHERE {BASE_FILTER} AND source = 'kaggle_asaniczka'
              AND is_swe = true AND seniority_native = 'associate'
        GROUP BY title_normalized
    ),
    arshkon_entry AS (
        SELECT title_normalized, COUNT(*) as arshkon_count
        FROM '{DATA}'
        WHERE {BASE_FILTER} AND source = 'kaggle_arshkon'
              AND is_swe = true AND seniority_native = 'entry'
        GROUP BY title_normalized
    )
    SELECT a.title_normalized, a.asaniczka_count, e.arshkon_count
    FROM asaniczka_assoc a
    INNER JOIN arshkon_entry e ON a.title_normalized = e.title_normalized
    ORDER BY a.asaniczka_count + e.arshkon_count DESC
""").df()
print("\nOverlapping titles (associate in asaniczka, entry in arshkon):")
print(overlap.to_string(index=False))
overlap.to_csv(f'{OUT_T03}/associate_entry_overlap.csv', index=False)

# Also check: same title appearing at different seniority levels in arshkon
arshkon_multi_seniority = con.execute(f"""
    SELECT title_normalized, seniority_native, COUNT(*) as n
    FROM '{DATA}'
    WHERE {BASE_FILTER} AND source = 'kaggle_arshkon'
          AND is_swe = true AND seniority_native IS NOT NULL
    GROUP BY title_normalized, seniority_native
    HAVING title_normalized IN (
        SELECT title_normalized FROM '{DATA}'
        WHERE {BASE_FILTER} AND source = 'kaggle_arshkon'
              AND is_swe = true AND seniority_native IS NOT NULL
        GROUP BY title_normalized
        HAVING COUNT(DISTINCT seniority_native) > 1
    )
    ORDER BY title_normalized, n DESC
""").df()
print("\nArshkon titles at multiple seniority levels (SWE):")
print(arshkon_multi_seniority.to_string(index=False))
arshkon_multi_seniority.to_csv(f'{OUT_T03}/arshkon_multi_seniority_titles.csv', index=False)

# ========================================================================
# T03 Step 4: Effective sample sizes after excluding nulls
# ========================================================================
print("\n=== T03: Effective sample sizes per source ===")

# Cross-period analysis fields
cross_period_fields = [
    'seniority_final', 'seniority_native', 'seniority_3level',
    'description', 'description_core', 'title_normalized',
    'company_name_effective', 'company_industry', 'company_size',
    'metro_area', 'state_normalized', 'is_remote_inferred',
    'skills_raw', 'yoe_extracted', 'work_type'
]

eff_samples = []
for src in sources:
    for subset_label, extra_filter in [('all', ''), ('swe', ' AND is_swe = true')]:
        for field in cross_period_fields:
            if col_types.get(field) in ('VARCHAR',):
                null_condition = f"(\"{field}\" IS NOT NULL AND \"{field}\" != '')"
            else:
                null_condition = f"\"{field}\" IS NOT NULL"

            # Also handle seniority_final 'unknown' as effectively null
            if field in ('seniority_final', 'seniority_3level'):
                null_condition += f" AND \"{field}\" != 'unknown'"

            result = con.execute(f"""
                SELECT
                    COUNT(*) as total_rows,
                    SUM(CASE WHEN {null_condition} THEN 1 ELSE 0 END) as usable_rows
                FROM '{DATA}'
                WHERE {BASE_FILTER} AND source = '{src}'{extra_filter}
            """).fetchone()

            total = result[0]
            usable = result[1]
            eff_samples.append({
                'source': src, 'subset': subset_label, 'field': field,
                'total_rows': total, 'usable_rows': usable,
                'usable_pct': round(usable / total * 100, 2) if total > 0 else 0
            })

eff_df = pd.DataFrame(eff_samples)
eff_df.to_csv(f'{OUT_T03}/effective_sample_sizes.csv', index=False)
print(f"  Saved to {OUT_T03}/effective_sample_sizes.csv")

# Print SWE subset summary
print("\nEffective sample sizes (SWE subset, usable rows / total):")
eff_swe = eff_df[eff_df['subset'] == 'swe'].pivot(index='field', columns='source', values='usable_rows')
eff_swe = eff_swe[sources]
eff_swe_pct = eff_df[eff_df['subset'] == 'swe'].pivot(index='field', columns='source', values='usable_pct')
eff_swe_pct = eff_swe_pct[sources]
print(eff_swe.to_string())
print("\nAs percentages:")
print(eff_swe_pct.to_string())

# ========================================================================
# T03: Missing-data heatmap (fields x sources, SWE subset)
# ========================================================================
print("\n=== T03: Missing data heatmap (SWE subset) ===")

# Focus on cross-period analysis fields
fields_for_heatmap = cross_period_fields
heatmap_data = eff_swe_pct.loc[fields_for_heatmap]

fig, ax = plt.subplots(figsize=(8, 8))
data_h = heatmap_data.values
im = ax.imshow(data_h, cmap=plt.cm.RdYlGn, vmin=0, vmax=100, aspect='auto')
ax.set_xticks(range(len(sources)))
ax.set_xticklabels(['arshkon', 'asaniczka', 'scraped'], fontsize=10, fontweight='bold')
ax.set_yticks(range(len(fields_for_heatmap)))
ax.set_yticklabels(fields_for_heatmap, fontsize=8)
ax.set_title('T03: Usable Data Coverage (SWE Subset)\n% non-null/non-unknown rows by source', fontsize=12, fontweight='bold')

for i in range(len(fields_for_heatmap)):
    for j in range(len(sources)):
        val = data_h[i, j]
        color = 'white' if val < 30 or val > 80 else 'black'
        ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=8, color=color)

plt.colorbar(im, ax=ax, label='Usable %', shrink=0.7)
plt.tight_layout()
plt.savefig(f'{FIG_T03}/missing_data_heatmap_swe.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved to {FIG_T03}/missing_data_heatmap_swe.png")

# ========================================================================
# Summary stats for reports
# ========================================================================
print("\n=== Summary statistics for reports ===")

# Row counts with default filters
for src in sources:
    for subset_label, extra_filter in [('all', ''), ('swe', ' AND is_swe = true')]:
        cnt = con.execute(f"""
            SELECT COUNT(*) FROM '{DATA}'
            WHERE {BASE_FILTER} AND source = '{src}'{extra_filter}
        """).fetchone()[0]
        print(f"  {src} {subset_label}: {cnt:,}")

# Entry-level counts
print("\nEntry-level counts (seniority_final = 'entry', SWE):")
for src in sources:
    cnt = con.execute(f"""
        SELECT COUNT(*) FROM '{DATA}'
        WHERE {BASE_FILTER} AND source = '{src}'
              AND is_swe = true AND seniority_final = 'entry'
    """).fetchone()[0]
    print(f"  {src}: {cnt}")

# Entry-level counts by seniority_native
print("\nEntry-level counts (seniority_native = 'entry', SWE):")
for src in sources:
    cnt = con.execute(f"""
        SELECT COUNT(*) FROM '{DATA}'
        WHERE {BASE_FILTER} AND source = '{src}'
              AND is_swe = true AND seniority_native = 'entry'
    """).fetchone()[0]
    print(f"  {src}: {cnt}")

con.close()
print("\n=== Done ===")
