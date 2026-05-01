"""V1 Phase A — Headline H4 re-derivation (Tension A).

Independently computes within-vs-between J3 decomposition under multiple
panel constructions. Wave 2 T08 claim: within-co J3 = +3.5 to +6.4 pp under
pooled / asaniczka / min-3 / min-5 / min-10 panels, vs Gate 1 T06 claim ~0
under arshkon-only min5 (n=125).

Approach:
- Load unified.parquet filtered to SWE LinkedIn English date_flag=ok.
- For each panel construction:
  - Define "returning companies": those with >= min_n postings in BOTH 2024 and 2026.
  - Compute J3 share in 2024 and 2026.
  - Decompose aggregate Delta into within-company and between-company.
- Identify what determines the "defensible primary" estimate.
- Note: T06 panel of n=125 uses all SWE rows (no aggregator exclusion) for
  identifying overlap cos; J3 shares computed on labeled only.
"""

import duckdb
import numpy as np
import pandas as pd

print("[V1_H4] Loading SWE LinkedIn frame", flush=True)
con = duckdb.connect()

df = con.execute("""
    SELECT uid,
           source,
           period,
           company_name_canonical,
           yoe_min_years_llm,
           is_aggregator,
           llm_classification_coverage
    FROM read_parquet('data/unified.parquet')
    WHERE source_platform = 'linkedin'
      AND is_english = TRUE
      AND date_flag = 'ok'
      AND is_swe = TRUE
""").fetchdf()
print(f"[V1_H4] Loaded {len(df)} SWE LinkedIn rows", flush=True)

df['period2'] = df['period'].astype(str).apply(
    lambda p: '2024' if p.startswith('2024') else ('2026' if p.startswith('2026') else 'other')
)
df = df[df['period2'].isin(['2024','2026'])].copy()

# J3 = YOE<=2 among labeled rows
df['labeled'] = df['yoe_min_years_llm'].notna()
df['is_j3'] = df['labeled'] & (df['yoe_min_years_llm'] <= 2)

# Baseline: aggregate J3
j3_2024_pool = df[(df['period2']=='2024') & df['labeled']]['is_j3'].mean()
j3_2026_pool = df[(df['period2']=='2026') & df['labeled']]['is_j3'].mean()
print(f"[V1_H4] Aggregate J3: 2024={j3_2024_pool:.4f}  2026={j3_2026_pool:.4f}  "
      f"Delta={100*(j3_2026_pool-j3_2024_pool):+.2f} pp", flush=True)

def decompose_within_between(all_df, source_filter=None, min_n_per_period=1,
                             exclude_aggregator=False):
    """Overlap-panel within-vs-between decomposition.

    all_df: dataframe with period2, company_name_canonical, labeled, is_j3, source
    source_filter: e.g. 'kaggle_arshkon' to keep only arshkon rows in 2024 baseline,
                   None for pooled (arshkon + asaniczka).
    min_n_per_period: minimum postings per period per company to keep in overlap panel.
    exclude_aggregator: drop is_aggregator=True rows before overlap determination.

    Overlap is determined on ALL SWE rows (not just labeled), matching T06/T08 convention.
    J3 shares use LABELED rows only.
    """
    df2 = all_df.copy()
    if exclude_aggregator:
        df2 = df2[~df2['is_aggregator'].fillna(False)]
    if source_filter is not None:
        keep = ((df2['period2'] == '2024') & (df2['source'] == source_filter)) | \
               (df2['period2'] == '2026')
        df2 = df2[keep]
    # Overlap based on ALL rows (not labeled-filtered)
    cnt = df2.groupby(['company_name_canonical', 'period2']).size().unstack(fill_value=0)
    cnt.columns = [f"n_{c}" for c in cnt.columns]
    if 'n_2024' not in cnt: cnt['n_2024'] = 0
    if 'n_2026' not in cnt: cnt['n_2026'] = 0
    overlap = cnt[(cnt['n_2024'] >= min_n_per_period) & (cnt['n_2026'] >= min_n_per_period)]
    cos = overlap.index.tolist()
    n_cos = len(overlap)
    # Compute J3 shares per cos x period on LABELED rows
    df_labeled = df2[df2['labeled'] & df2['company_name_canonical'].isin(cos)]
    grp = df_labeled.groupby(['company_name_canonical','period2'])
    shares = grp['is_j3'].mean().unstack()
    shares.columns = [f"j3_{c}" for c in shares.columns]
    sizes = grp.size().unstack()
    sizes.columns = [f"n_{c}" for c in sizes.columns]
    merged = shares.join(sizes).dropna(subset=['j3_2024','j3_2026'])
    # Use weights: 2026 volume as weights (within-company delta weighted)
    # Weighted aggregate J3 in each period
    j3_24_agg = (merged['j3_2024'] * merged['n_2024']).sum() / merged['n_2024'].sum()
    j3_26_agg = (merged['j3_2026'] * merged['n_2026']).sum() / merged['n_2026'].sum()
    agg_delta_pp = 100 * (j3_26_agg - j3_24_agg)
    within_pp = 100 * ((merged['j3_2026'] - merged['j3_2024']) * merged['n_2026']).sum() / merged['n_2026'].sum()
    between_pp = agg_delta_pp - within_pp
    # Note: labeled-only n after overlap
    n_cos_with_j3 = len(merged)
    return {
        "n_cos_overlap": n_cos,
        "n_cos_with_j3_in_both_periods": n_cos_with_j3,
        "j3_2024_agg": j3_24_agg,
        "j3_2026_agg": j3_26_agg,
        "agg_delta_pp": agg_delta_pp,
        "within_pp": within_pp,
        "between_pp": between_pp,
    }

panels = [
    ("pooled_min1",   None,              1, False),
    ("pooled_min3",   None,              3, False),
    ("pooled_min5",   None,              5, False),
    ("pooled_min10",  None,             10, False),
    ("arshkon_min5",  "kaggle_arshkon",  5, False),
    ("asaniczka_min5","kaggle_asaniczka",5, False),
    # Also with aggregator exclusion for robustness
    ("arshkon_min5_no_agg", "kaggle_arshkon", 5, True),
    ("pooled_min5_no_agg",  None,              5, True),
]

print("\n[V1_H4] === WITHIN-VS-BETWEEN DECOMPOSITION (T06 / T08 replication) ===", flush=True)
print(f"{'panel':<25} {'n_cos':>6} {'j24':>7} {'j26':>7} {'agg':>8} {'within':>8} {'between':>8}", flush=True)
rows = []
for name, src, min_n, exc_agg in panels:
    r = decompose_within_between(df, source_filter=src, min_n_per_period=min_n,
                                 exclude_aggregator=exc_agg)
    r['panel'] = name
    rows.append(r)
    print(f"{name:<25} {r['n_cos_overlap']:>6} {r['j3_2024_agg']:>7.4f} {r['j3_2026_agg']:>7.4f} "
          f"{r['agg_delta_pp']:>+8.2f} {r['within_pp']:>+8.2f} {r['between_pp']:>+8.2f}", flush=True)

verdict = pd.DataFrame(rows)[['panel','n_cos_overlap','n_cos_with_j3_in_both_periods',
                              'j3_2024_agg','j3_2026_agg',
                              'agg_delta_pp','within_pp','between_pp']]
verdict.to_csv("exploration/tables/V1/H4_within_between.csv", index=False)
print("\n[V1_H4] Wrote exploration/tables/V1/H4_within_between.csv", flush=True)

print("\n[V1_H4] === VERDICT vs T08 claims ===", flush=True)
print("T08 table:", flush=True)
print("  pooled_min1:   +6.17 agg, +4.23 within, +1.94 between", flush=True)
print("  pooled_min3:   +8.17 agg, +5.11 within, +3.05 between", flush=True)
print("  pooled_min5:   +7.90 agg, +5.05 within, +2.85 between", flush=True)
print("  arshkon_min5:  +5.02 agg, -0.03 within, +5.05 between (n=125, T06 claim)", flush=True)
print("  asaniczka_min5:+8.70 agg, +6.43 within, +2.26 between", flush=True)
print("  pooled_min10:  +7.90 agg, +5.20 within, +2.69 between", flush=True)

# Per-company audit for arshkon panel — distribution of within-company deltas
print("\n[V1_H4] === DRILL-DOWN: Arshkon min5 per-company delta distribution ===", flush=True)
r = decompose_within_between(df, source_filter="kaggle_arshkon", min_n_per_period=5)
print(f"Number of overlap companies: {r['n_cos_overlap']}", flush=True)
