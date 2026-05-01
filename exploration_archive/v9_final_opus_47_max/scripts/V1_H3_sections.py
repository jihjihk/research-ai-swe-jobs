"""V1 Phase A — Headline H3 re-derivation.

Independently verifies: J3 (YOE<=2, labeled) requirements-section chars dropped
-5% 2024->2026 absolute (T13 reports -57 chars / -5%). Benefits +89%.

Approach:
- Load raw descriptions (the ONLY place section anatomy is meaningful).
- Apply T13_section_classifier (explicit shared module per V1 spec).
- Compute mean chars per section for J3 (YOE<=2 labeled) subset per period.
- Also build alt independent header-detection and compare to T13 module.
"""

import sys
import re
import duckdb
import pandas as pd
import numpy as np

sys.path.insert(0, "exploration/scripts")
import T13_section_classifier as sc  # explicit shared module

print("[V1_H3] Loading raw descriptions for J3 subset", flush=True)
con = duckdb.connect()

df = con.execute("""
    SELECT uid,
           description,
           description_length,
           period,
           source,
           yoe_min_years_llm
    FROM read_parquet('data/unified.parquet')
    WHERE source_platform = 'linkedin'
      AND is_english = TRUE
      AND date_flag = 'ok'
      AND is_swe = TRUE
      AND yoe_min_years_llm IS NOT NULL
      AND yoe_min_years_llm <= 2
""").fetchdf()
print(f"[V1_H3] Loaded {len(df)} J3 rows (labeled, YOE<=2)", flush=True)

df['period2'] = df['period'].astype(str).apply(
    lambda p: '2024' if p.startswith('2024') else ('2026' if p.startswith('2026') else 'other')
)
df = df[df['period2'].isin(['2024','2026'])].copy()
print(f"[V1_H3] J3 period distribution:", flush=True)
print(df['period2'].value_counts(), flush=True)

# Apply T13 section classifier
print("[V1_H3] Applying T13 section classifier ...", flush=True)
descs = df['description'].fillna('').astype(str).tolist()
results = [sc.classify_sections(d) for d in descs]

for lab in sc.SECTION_LABELS:
    df[f'sec_{lab}_chars'] = [r.get(lab, 0) for r in results]
df['sec_total'] = [r.get('total', 0) for r in results]

# Mean chars per section by period (J3 subset)
agg_cols = [f'sec_{l}_chars' for l in sc.SECTION_LABELS] + ['sec_total']
g = df.groupby('period2')[agg_cols].mean().round(1)
print("\n[V1_H3] === T13-classifier on J3 subset ===", flush=True)
print(g, flush=True)

# Compute J3 deltas
for c in agg_cols:
    d24 = g.loc['2024', c]
    d26 = g.loc['2026', c]
    pct = 100 * (d26 - d24) / max(d24, 1e-9)
    print(f"  {c}: 2024 {d24:.0f} -> 2026 {d26:.0f}   Delta {d26-d24:+.0f} chars ({pct:+.1f}%)", flush=True)

# Also compute for the all-SWE primary to compare to T13's headline table
print("\n[V1_H3] === T13-classifier on ALL SWE (for cross-check to T13 global table) ===", flush=True)
df_all = con.execute("""
    SELECT uid, description, description_length, period
    FROM read_parquet('data/unified.parquet')
    WHERE source_platform = 'linkedin'
      AND is_english = TRUE
      AND date_flag = 'ok'
      AND is_swe = TRUE
""").fetchdf()
df_all['period2'] = df_all['period'].astype(str).apply(
    lambda p: '2024' if p.startswith('2024') else ('2026' if p.startswith('2026') else 'other')
)
df_all = df_all[df_all['period2'].isin(['2024','2026'])].copy()

print(f"[V1_H3] ALL SWE: {len(df_all)} rows. Applying classifier ...", flush=True)
descs_all = df_all['description'].fillna('').astype(str).tolist()
results_all = [sc.classify_sections(d) for d in descs_all]
for lab in sc.SECTION_LABELS:
    df_all[f'sec_{lab}_chars'] = [r.get(lab, 0) for r in results_all]
df_all['sec_total'] = [r.get('total', 0) for r in results_all]
g_all = df_all.groupby('period2')[agg_cols].mean().round(1)
print(g_all, flush=True)
for c in agg_cols:
    d24 = g_all.loc['2024', c]
    d26 = g_all.loc['2026', c]
    pct = 100 * (d26 - d24) / max(d24, 1e-9)
    print(f"  {c}: 2024 {d24:.0f} -> 2026 {d26:.0f}   Delta {d26-d24:+.0f} chars ({pct:+.1f}%)", flush=True)

# Save tables
g.to_csv("exploration/tables/V1/H3_section_anatomy_J3.csv")
g_all.to_csv("exploration/tables/V1/H3_section_anatomy_ALL.csv")
print("\n[V1_H3] Wrote exploration/tables/V1/H3_section_anatomy_{J3,ALL}.csv", flush=True)

print("\n[V1_H3] === T13 claim (for reference) ===", flush=True)
print("  J3 requirements: 1057 -> 1001 (-57 chars, -5%)", flush=True)
print("  J3 benefits:     367 -> 704  (+337 chars, +92%)", flush=True)
print("  J3 responsibilities: 858 -> 1150 (+292 chars, +34%)", flush=True)
print("  ALL Requirements:  887 -> 984 (+98, +11%)", flush=True)
print("  ALL Benefits:      322 -> 609 (+287, +89%)", flush=True)
print("  ALL Responsibilities: 723 -> 1077 (+355, +49%)", flush=True)
