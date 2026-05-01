"""V1 H3 — faster version: sample 2000 J3 rows per period for T13-classifier verification."""

import sys, duckdb, pandas as pd, numpy as np
sys.path.insert(0, "exploration/scripts")
import T13_section_classifier as sc

con = duckdb.connect()
df = con.execute("""
    SELECT uid, description, period, yoe_min_years_llm
    FROM read_parquet('data/unified.parquet')
    WHERE source_platform = 'linkedin'
      AND is_english = TRUE
      AND date_flag = 'ok'
      AND is_swe = TRUE
      AND yoe_min_years_llm IS NOT NULL
      AND yoe_min_years_llm <= 2
""").fetchdf()
df['period2'] = df['period'].astype(str).apply(
    lambda p: '2024' if p.startswith('2024') else ('2026' if p.startswith('2026') else 'other')
)
df = df[df['period2'].isin(['2024','2026'])].copy()
# Stratified sample: up to 1000 per period
rows = []
for p, g in df.groupby('period2'):
    if len(g) > 1000:
        g = g.sample(1000, random_state=42)
    rows.append(g)
sample = pd.concat(rows, ignore_index=True)
print(f"[V1_H3s] Sample: {len(sample)} rows; period counts:", flush=True)
print(sample['period2'].value_counts(), flush=True)

print("[V1_H3s] Running T13 classifier on sample ...", flush=True)
results = [sc.classify_sections(d) for d in sample['description'].fillna('').astype(str).tolist()]
for lab in sc.SECTION_LABELS:
    sample[f'sec_{lab}_chars'] = [r.get(lab, 0) for r in results]
sample['sec_total'] = [r.get('total', 0) for r in results]

agg_cols = [f'sec_{l}_chars' for l in sc.SECTION_LABELS] + ['sec_total']
g = sample.groupby('period2')[agg_cols].mean().round(1)
print("\n[V1_H3s] J3 section anatomy (T13 classifier, 1000-per-period sample)", flush=True)
print(g, flush=True)

print("\n[V1_H3s] Deltas:", flush=True)
for c in agg_cols:
    d24 = g.loc['2024', c]
    d26 = g.loc['2026', c]
    pct = 100 * (d26 - d24) / max(d24, 1e-9)
    print(f"  {c}: 2024 {d24:.0f} -> 2026 {d26:.0f}   Delta {d26-d24:+.0f} chars ({pct:+.1f}%)", flush=True)

g.to_csv("exploration/tables/V1/H3_T13sampled_J3.csv")
print("\n[V1_H3s] Wrote exploration/tables/V1/H3_T13sampled_J3.csv", flush=True)
print("[V1_H3s] T13 full-run claim for J3: requirements 1057 -> 1001 (-57 chars, -5%)", flush=True)
