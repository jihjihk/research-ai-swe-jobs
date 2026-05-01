"""V1 Phase D — Composite-score correlation audit.

For each composite score in Wave 2, compute:
- Correlation of each COMPONENT with description length.
- Correlation of each COMPONENT with an outcome (e.g., raw breadth with length).
- Verify length-residualized variants behave as expected.
- Validate T11's residualization formula by independent re-fit.
"""

import duckdb
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

print("[V1_D] Loading T11 posting features", flush=True)
con = duckdb.connect()
df = con.execute("""
    SELECT *
    FROM read_parquet('exploration/artifacts/shared/T11_posting_features.parquet')
""").fetchdf()
print(f"[V1_D] Loaded {len(df)} posting features", flush=True)

df['period2'] = df['period'].astype(str).apply(
    lambda p: '2024' if p.startswith('2024') else ('2026' if p.startswith('2026') else 'other')
)
df = df[df['period2'].isin(['2024','2026'])].copy()
df['log_length'] = np.log1p(df['description_cleaned_length'].fillna(0))

# Full-corpus correlations for every composite + component
cols = ['requirement_breadth','credential_stack_depth','tech_count',
        'scope_density','mgmt_strong_density','mgmt_broad_density','ai_binary']

print("\n[V1_D] === COMPOSITE x log_length correlations (full corpus) ===", flush=True)
print(f"{'metric':<30} {'r(pooled)':>10} {'r(2024)':>10} {'r(2026)':>10}", flush=True)
out_rows = []
for c in cols:
    if c not in df.columns:
        continue
    mask = df['log_length'].notna() & df[c].notna()
    r_p = df.loc[mask, c].corr(df.loc[mask, 'log_length'])
    mask_24 = mask & (df['period2']=='2024')
    mask_26 = mask & (df['period2']=='2026')
    r_24 = df.loc[mask_24, c].corr(df.loc[mask_24, 'log_length'])
    r_26 = df.loc[mask_26, c].corr(df.loc[mask_26, 'log_length'])
    print(f"{c:<30} {r_p:>+10.4f} {r_24:>+10.4f} {r_26:>+10.4f}", flush=True)
    out_rows.append({"metric": c, "r_pooled": r_p, "r_2024": r_24, "r_2026": r_26})

pd.DataFrame(out_rows).to_csv("exploration/tables/V1/D_composite_length_corr.csv", index=False)

# Component-level: break requirement_breadth into its parts
# T11 defines: breadth = tech_count + soft_skill + scope + mgmt_broad + ai_binary + (edu>0) + (yoe.notna())
# Verify by examining T11_posting_features columns
print("\n[V1_D] === Component coverage check ===", flush=True)
needed = ['tech_count','scope_density','mgmt_strong_density','ai_binary','education_level','yoe_min_years_llm']
for c in needed:
    if c in df.columns:
        nn = df[c].notna().sum()
        print(f"  {c}: {nn} non-null (of {len(df)})", flush=True)
    else:
        print(f"  {c}: MISSING", flush=True)

# ---- T11 residualization re-validation ----
# re-fit requirement_breadth ~ a + b * log_length on full corpus
print("\n[V1_D] === T11 residualization formula re-fit ===", flush=True)
for metric in ['requirement_breadth', 'credential_stack_depth']:
    mask = df['log_length'].notna() & df[metric].notna()
    X = df.loc[mask, 'log_length'].values.reshape(-1, 1)
    y = df.loc[mask, metric].values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    resid_v1 = y - y_pred
    # Compare to T11's resid column
    t11_resid_col = f"{metric}_resid"
    if t11_resid_col in df.columns:
        t11_resid = df.loc[mask, t11_resid_col].values
        diff = resid_v1 - t11_resid
        max_abs_diff = np.abs(diff).max()
        mean_abs_diff = np.abs(diff).mean()
        print(f"  {metric}: fit b0={reg.intercept_:.4f} b1={reg.coef_[0]:.4f}", flush=True)
        print(f"    Max |V1 - T11 resid|: {max_abs_diff:.5f}   Mean: {mean_abs_diff:.5f}", flush=True)
        print(f"    Residualization {'MATCHES' if max_abs_diff < 0.01 else 'DIFFERS (may use different fit data)'}", flush=True)

# ---- Component-by-component correlation with outcome (breadth) ----
print("\n[V1_D] === Per-component correlation with raw breadth ===", flush=True)
# Not perfectly valid since breadth is composed of components, but informative.
# Focus on whether residualized components behave like their raw.
comp_vars = ['tech_count','scope_density','mgmt_strong_density','ai_binary']
for c in comp_vars:
    if c not in df.columns:
        continue
    mask = df['requirement_breadth'].notna() & df[c].notna()
    r = df.loc[mask, c].corr(df.loc[mask, 'requirement_breadth'])
    print(f"  {c} vs requirement_breadth: r = {r:+.4f}", flush=True)

# ---- Stack depth component check ----
print("\n[V1_D] === Stack depth x length correlation ===", flush=True)
for p in ['2024','2026']:
    dd = df[df['period2']==p].dropna(subset=['log_length','credential_stack_depth'])
    r = dd['credential_stack_depth'].corr(dd['log_length'])
    # T11 claimed r = 0.544 (2024) / 0.286 (2026) on labeled; 0.342 pooled
    print(f"  Period {p}: r = {r:+.4f}   (T11 reported: 2024=0.544, 2026=0.286 pooled 0.342)", flush=True)

# ---- "X attenuates under matching" check ----
# T11 reports: raw breadth +3.05 pooled; resid +1.22. So most of raw was length-driven.
# This is valid — let me verify the numbers.
print("\n[V1_D] === Attenuation check: raw vs resid breadth delta ===", flush=True)
for metric in ['requirement_breadth','credential_stack_depth']:
    raw = df.groupby('period2')[metric].mean()
    resid = df.groupby('period2')[f'{metric}_resid'].mean() if f'{metric}_resid' in df.columns else pd.Series()
    print(f"  {metric}: 2024 raw={raw.get('2024',np.nan):.2f} 2026 raw={raw.get('2026',np.nan):.2f}  "
          f"raw Δ={raw.get('2026',np.nan)-raw.get('2024',np.nan):+.2f}", flush=True)
    if not resid.empty:
        print(f"  {metric}_resid: 2024={resid.get('2024',np.nan):.2f} 2026={resid.get('2026',np.nan):.2f}  "
              f"resid Δ={resid.get('2026',np.nan)-resid.get('2024',np.nan):+.2f}", flush=True)

# ---- T14 tech phi correlation audit ----
# T14 claims: pinecone x weaviate phi 0 -> 0.71; rag x llm phi 0.20 -> 0.51
# Need to check tech_matrix
print("\n[V1_D] === T14 tech phi audit ===", flush=True)
tech = con.execute("""
    SELECT tech.*, u.period, u.source, u.source_platform, u.is_english, u.date_flag, u.is_swe
    FROM read_parquet('exploration/artifacts/shared/swe_tech_matrix.parquet') tech
    LEFT JOIN read_parquet('data/unified.parquet') u
      ON tech.uid = u.uid
    WHERE u.source_platform = 'linkedin'
      AND u.is_english = TRUE
      AND u.date_flag = 'ok'
      AND u.is_swe = TRUE
""").fetchdf()

cols = [c for c in tech.columns if c not in ('uid','period','source','source_platform','is_english','date_flag','is_swe')]
tech['period2'] = tech['period'].astype(str).apply(
    lambda p: '2024' if p.startswith('2024') else ('2026' if p.startswith('2026') else 'other')
)
tech = tech[tech['period2'].isin(['2024','2026'])].copy()

# Check pinecone + weaviate columns exist
pinecone_col = None; weaviate_col = None; rag_col = None; llm_col = None
for c in cols:
    if c.lower() == 'pinecone': pinecone_col = c
    if c.lower() == 'weaviate': weaviate_col = c
    if c.lower() == 'rag': rag_col = c
    if c.lower() == 'llm': llm_col = c
print(f"  pinecone={pinecone_col} weaviate={weaviate_col} rag={rag_col} llm={llm_col}", flush=True)

def phi(x, y):
    """Pearson phi coefficient between two binary series"""
    x = x.astype(int); y = y.astype(int)
    mean_x = x.mean(); mean_y = y.mean()
    num = ((x-mean_x) * (y-mean_y)).mean()
    den = np.sqrt((x-mean_x).var() * (y-mean_y).var())
    return num / den if den > 0 else np.nan

if pinecone_col and weaviate_col:
    for p in ['2024','2026']:
        dd = tech[tech['period2']==p]
        if len(dd) == 0: continue
        phi_val = phi(dd[pinecone_col], dd[weaviate_col])
        print(f"  Period {p}: phi(pinecone, weaviate) = {phi_val:+.4f}  (T14 claim: 2024=0, 2026=0.71)", flush=True)
if rag_col and llm_col:
    for p in ['2024','2026']:
        dd = tech[tech['period2']==p]
        if len(dd) == 0: continue
        phi_val = phi(dd[rag_col], dd[llm_col])
        print(f"  Period {p}: phi(rag, llm) = {phi_val:+.4f}  (T14 claim: 2024=0.20, 2026=0.51)", flush=True)
