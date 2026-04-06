#!/usr/bin/env python3
"""T10: Title taxonomy evolution — SWE LinkedIn postings, 2024 vs 2026.
V2: Uses raw title for seniority marker analysis (title_normalized strips level markers)."""

import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

BASE = Path("/home/jihgaboot/gabor/job-research")
DATA = BASE / "data/unified.parquet"
FIG_DIR = BASE / "exploration/figures/T10"
TAB_DIR = BASE / "exploration/tables/T10"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

FILTERS = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe = true"
con = duckdb.connect()

print("Loading base data...")
base = con.sql(f"""
    SELECT uid, title_normalized, title, period, source, seniority_final, seniority_3level,
           is_aggregator, company_name_canonical, yoe_extracted,
           description_length, core_length
    FROM '{DATA}'
    WHERE {FILTERS}
""").fetchdf()

# Create lowercase title for marker analysis (from original title, not normalized)
base['title_lower'] = base['title'].str.lower().str.strip()

arshkon = base[base['source'] == 'kaggle_arshkon']
asaniczka = base[base['source'] == 'kaggle_asaniczka']
scraped = base[base['source'] == 'scraped']
p2024_01 = asaniczka
p2024_04 = arshkon
p2026 = scraped

# ============================================================
# 5 (REVISED). Seniority-marker title shares — using raw title
# ============================================================
print("\n=== 5. Seniority-marker title shares (from raw title) ===")

seniority_markers = {
    'junior': r'\bjunior\b',
    'associate': r'\bassociate\b',
    'senior': r'\bsenior\b',
    'staff': r'\bstaff\b',
    'principal': r'\bprincipal\b',
    'lead': r'\blead\b',
    'director': r'\bdirector\b',
    'architect': r'\barchitect\b',
    'founding': r'\bfounding\b',
    'intern': r'\bintern\b',
    'entry': r'\bentry[\s-]?level\b',
    'ii_iii': r'\b(?:ii|iii|iv|level\s*[234])\b',
}

marker_results = []
for pname, pdf in [('2024-01', p2024_01), ('2024-04', p2024_04), ('2026-03', p2026)]:
    n = len(pdf)
    for marker, pattern in seniority_markers.items():
        count = pdf['title_lower'].str.contains(pattern, na=False, flags=re.IGNORECASE).sum()
        marker_results.append({
            'period': pname,
            'marker': marker,
            'count': count,
            'share': count / n,
            'variant': 'all'
        })

    pdf_na = pdf[~pdf['is_aggregator']]
    n_na = len(pdf_na)
    for marker, pattern in seniority_markers.items():
        count = pdf_na['title_lower'].str.contains(pattern, na=False, flags=re.IGNORECASE).sum()
        marker_results.append({
            'period': pname,
            'marker': marker,
            'count': count,
            'share': count / n_na if n_na > 0 else 0,
            'variant': 'no_aggregator'
        })

marker_df = pd.DataFrame(marker_results)
for variant in ['all', 'no_aggregator']:
    print(f"\n{variant}:")
    sub = marker_df[marker_df['variant'] == variant]
    pivot = sub.pivot(index='marker', columns='period', values='share').round(4)
    print(pivot.to_string())
    # Compute change columns
    if '2024-04' in pivot.columns and '2026-03' in pivot.columns:
        pivot['change_04_to_26'] = pivot['2026-03'] - pivot['2024-04']
        pivot['pct_change'] = ((pivot['2026-03'] - pivot['2024-04']) / pivot['2024-04'].replace(0, np.nan) * 100).round(1)
        print("\nWith change:")
        print(pivot.to_string())

marker_df.to_csv(TAB_DIR / "seniority_marker_shares_v2.csv", index=False)

# ============================================================
# Fig 1 (REVISED): Seniority marker share change with raw titles
# ============================================================
print("\n=== Generating revised figure ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Seniority level markers
markers_level = ['junior', 'associate', 'senior', 'staff', 'principal', 'lead', 'director', 'intern']
ax = axes[0]
plot_data = marker_df[(marker_df['variant'] == 'all') & (marker_df['marker'].isin(markers_level))]
for marker in markers_level:
    sub = plot_data[plot_data['marker'] == marker].sort_values('period')
    ax.plot(sub['period'], sub['share'] * 100, 'o-', label=marker, markersize=7)
ax.set_ylabel('Share of SWE postings (%)')
ax.set_xlabel('Period')
ax.set_title('A. Seniority Level Markers in SWE Titles')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)

# Panel B: Same but aggregator-excluded
ax = axes[1]
plot_data = marker_df[(marker_df['variant'] == 'no_aggregator') & (marker_df['marker'].isin(markers_level))]
for marker in markers_level:
    sub = plot_data[plot_data['marker'] == marker].sort_values('period')
    ax.plot(sub['period'], sub['share'] * 100, 'o-', label=marker, markersize=7)
ax.set_ylabel('Share of SWE postings (%)')
ax.set_xlabel('Period')
ax.set_title('B. Same, Aggregator-Excluded')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "seniority_marker_trends.png", dpi=150, bbox_inches='tight')
plt.close()

# Also output a focused table showing the narrative clearly
print("\n=== Key narrative: seniority markers in titles ===")
for variant in ['all', 'no_aggregator']:
    sub = marker_df[marker_df['variant'] == variant]
    pivot = sub.pivot(index='marker', columns='period', values='share')
    if '2024-04' in pivot.columns and '2026-03' in pivot.columns:
        pivot['pp_change'] = (pivot['2026-03'] - pivot['2024-04']) * 100
        print(f"\n{variant} (pp change from 2024-04 to 2026-03):")
        print(pivot[['2024-04', '2026-03', 'pp_change']].sort_values('pp_change', ascending=False).round(3).to_string())

print("\n=== Done ===")
