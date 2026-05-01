"""V1 Phase A — Headline H6 re-derivation (Tension B/C).

Independently re-computes residualized requirement breadth by period x (J3, S4).
Claim: J3 +1.58, S4 +2.61 residualized units.

Approach:
- Load T11_posting_features.parquet (inputs trusted: raw feature columns).
- Re-run the residualization formula from scratch:
  requirement_breadth ~ b0 + b1 * log(description_cleaned_length).
- Compute J3 (YOE<=2 labeled) and S4 (YOE>=5 labeled) means by period.
- Also audit: correlation of raw requirement_breadth with log(length).
- Check stack_depth x length correlation for residualization validity.
"""

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

print("[V1_H6] Loading T11 posting features", flush=True)
con = duckdb.connect()

df = con.execute("""
    SELECT *
    FROM read_parquet('exploration/artifacts/shared/T11_posting_features.parquet')
""").fetchdf()
print(f"[V1_H6] Loaded {len(df)} posting features", flush=True)
print(f"[V1_H6] Columns: {list(df.columns)}", flush=True)

# Period2: 2024 vs 2026
df['period2'] = df['period'].astype(str).apply(
    lambda p: '2024' if p.startswith('2024') else ('2026' if p.startswith('2026') else 'other')
)
df = df[df['period2'].isin(['2024','2026'])].copy()

# Compute log(length + 1)
df['log_length'] = np.log1p(df['description_cleaned_length'].fillna(0))

# ---- AUDIT 1: Correlation of raw requirement_breadth with log_length ----
print("\n[V1_H6] === COMPOSITE LENGTH-CORRELATION AUDIT ===", flush=True)
cols_to_audit = [
    'requirement_breadth', 'credential_stack_depth',
    'tech_count', 'scope_density', 'mgmt_strong_density', 'mgmt_broad_density',
]
for p in ['2024','2026']:
    dd = df[df['period2']==p].dropna(subset=['log_length'])
    print(f"  Period {p} (n={len(dd)}):", flush=True)
    for c in cols_to_audit:
        if c not in dd.columns:
            continue
        # Pearson r with log_length
        ok = dd[c].notna()
        if ok.sum() < 10:
            continue
        r = dd[ok][c].corr(dd[ok]['log_length'])
        print(f"    corr({c:<30}, log_length) = {r:+.4f}", flush=True)

# ---- Re-run residualization from scratch ----
# T11 claims: fit y = b0 + b1 * log(desc_cleaned_length) on the full SWE LinkedIn frame,
# and residuals are reported.
print("\n[V1_H6] === RE-RUN RESIDUALIZATION FROM SCRATCH ===", flush=True)
# Only rows with valid log_length and requirement_breadth
mask = df['log_length'].notna() & df['requirement_breadth'].notna()
X = df.loc[mask, 'log_length'].values.reshape(-1, 1)
y = df.loc[mask, 'requirement_breadth'].values
reg = LinearRegression().fit(X, y)
b0, b1 = reg.intercept_, reg.coef_[0]
print(f"[V1_H6] requirement_breadth ~ {b0:.4f} + {b1:.4f} * log_length", flush=True)
df.loc[mask, 'breadth_resid_v1'] = y - reg.predict(X)
print(f"[V1_H6] Correlation check: raw breadth vs length r = "
      f"{df.loc[mask,'requirement_breadth'].corr(df.loc[mask,'log_length']):.4f}", flush=True)

# Same for credential_stack_depth
mask_s = df['log_length'].notna() & df['credential_stack_depth'].notna()
X_s = df.loc[mask_s, 'log_length'].values.reshape(-1, 1)
y_s = df.loc[mask_s, 'credential_stack_depth'].values
reg_s = LinearRegression().fit(X_s, y_s)
b0s, b1s = reg_s.intercept_, reg_s.coef_[0]
print(f"[V1_H6] credential_stack_depth ~ {b0s:.4f} + {b1s:.4f} * log_length", flush=True)
df.loc[mask_s, 'stack_resid_v1'] = y_s - reg_s.predict(X_s)

# ---- J3 and S4 means by period ----
print("\n[V1_H6] === J3 / S4 RESIDUALIZED BREADTH BY PERIOD (V1 independent re-derivation) ===", flush=True)
df['labeled'] = df['yoe_min_years_llm'].notna()
df['is_j3'] = df['labeled'] & (df['yoe_min_years_llm'] <= 2)
df['is_s4'] = df['labeled'] & (df['yoe_min_years_llm'] >= 5)

rows = []
for cut_name, mask in [('J3 (YOE<=2)', df['is_j3']),
                       ('S4 (YOE>=5)', df['is_s4'])]:
    for p in ['2024','2026']:
        sub = df[mask & (df['period2']==p)]
        mean_resid_v1 = sub['breadth_resid_v1'].mean()
        mean_resid_t11 = sub.get('requirement_breadth_resid', pd.Series([np.nan]*len(sub))).mean()
        mean_raw = sub['requirement_breadth'].mean()
        mean_len = sub['description_cleaned_length'].mean()
        rows.append({"cut": cut_name, "period": p, "n": len(sub),
                     "breadth_raw": mean_raw,
                     "breadth_resid_v1": mean_resid_v1,
                     "breadth_resid_t11": mean_resid_t11,
                     "desc_length_mean": mean_len})
        print(f"  {cut_name:<13} {p}: n={len(sub)}  raw={mean_raw:.2f}  "
              f"resid_v1={mean_resid_v1:+.2f}  resid_t11={mean_resid_t11:+.2f}", flush=True)

# Compute deltas
print("\n[V1_H6] === DELTAS (2026 - 2024) ===", flush=True)
for cut_name in ['J3 (YOE<=2)', 'S4 (YOE>=5)']:
    r24 = next(r for r in rows if r['cut']==cut_name and r['period']=='2024')
    r26 = next(r for r in rows if r['cut']==cut_name and r['period']=='2026')
    d_raw = r26['breadth_raw'] - r24['breadth_raw']
    d_v1 = r26['breadth_resid_v1'] - r24['breadth_resid_v1']
    d_t11 = r26['breadth_resid_t11'] - r24['breadth_resid_t11']
    print(f"  {cut_name}: raw delta {d_raw:+.2f}  resid_v1 {d_v1:+.2f}  resid_t11 {d_t11:+.2f}", flush=True)

pd.DataFrame(rows).to_csv("exploration/tables/V1/H6_breadth_by_period.csv", index=False)
print("\n[V1_H6] Wrote exploration/tables/V1/H6_breadth_by_period.csv", flush=True)
print("\n[V1_H6] Wave 2 T11 claim: J3 resid delta +1.58, S4 resid delta +2.61. Senior > junior.", flush=True)

# ---- H6 composite-score audit: stack_depth length correlation
print("\n[V1_H6] === STACK DEPTH x LENGTH AUDIT ===", flush=True)
for p in ['2024','2026']:
    dd = df[df['period2']==p].dropna(subset=['log_length','credential_stack_depth'])
    r = dd['credential_stack_depth'].corr(dd['log_length'])
    print(f"  Period {p}: corr(stack_depth, log_length) = {r:+.4f}", flush=True)
