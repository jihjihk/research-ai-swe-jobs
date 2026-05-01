"""V1 Phase A — Headline H2 re-derivation.

Independently computes junior-vs-senior TF-IDF centroid cosine for 2024 and 2026.
Claim: 0.946 (2024) -> 0.863 (2026), Delta = -0.083 (diverging).

Approach:
- Load swe_cleaned_text.parquet + unified.parquet.
- Sample ~2000 per (period x seniority_3level) where available.
- Build TF-IDF (n-gram 1-2, min_df=5, max_df=0.7), truncated SVD 100 dims.
- Compute centroid cosine junior-vs-senior within each period.
"""

import sys
import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

RNG = np.random.RandomState(91919)

print("[V1_H2] Loading cleaned text + seniority labels", flush=True)
con = duckdb.connect()

# Load LLM-cleaned SWE text with period and seniority_3level.
# Wave 2 T15 definition: period2 = 2024 (arshkon+asaniczka) vs 2026 (scraped).
df = con.execute("""
    SELECT c.uid,
           c.description_cleaned,
           c.text_source,
           u.source,
           u.period,
           u.seniority_3level,
           u.company_name_canonical
    FROM read_parquet('exploration/artifacts/shared/swe_cleaned_text.parquet') c
    LEFT JOIN read_parquet('data/unified.parquet') u
      ON c.uid = u.uid
    WHERE c.text_source = 'llm'
      AND u.source_platform = 'linkedin'
      AND u.is_english = TRUE
      AND u.date_flag = 'ok'
      AND u.is_swe = TRUE
""").fetchdf()
print(f"[V1_H2] Loaded {len(df)} LLM-cleaned SWE rows", flush=True)

# Map period to period2 (2024 vs 2026)
df['period2'] = df['period'].astype(str).apply(
    lambda p: '2024' if p.startswith('2024') else ('2026' if p.startswith('2026') else 'other')
)
df = df[df['period2'].isin(['2024','2026'])].copy()
print(f"[V1_H2] Period distribution:", flush=True)
print(df['period2'].value_counts(), flush=True)
print(f"[V1_H2] Seniority distribution:", flush=True)
print(df['seniority_3level'].value_counts(dropna=False), flush=True)

# Stratified sample: up to 2000 per (period2 x seniority).
SAMP_CAP = 2000
# Also cap 20 per canonical company per period x seniority (per T15 methodology)
COMPANY_CAP = 20
rows = []
for (period, sen), g in df.groupby(['period2', 'seniority_3level']):
    if pd.isna(sen) or sen not in ('junior', 'senior'):
        continue
    # company cap
    g2 = g.sample(frac=1.0, random_state=91).groupby('company_name_canonical').head(COMPANY_CAP)
    if len(g2) > SAMP_CAP:
        g2 = g2.sample(n=SAMP_CAP, random_state=91)
    rows.append(g2)
    print(f"  {period} {sen}: available {len(g)}, after cap {len(g2)}", flush=True)
sample = pd.concat(rows, ignore_index=True)
print(f"[V1_H2] Total sample: {len(sample)}", flush=True)

# Build TF-IDF on the full sample
vec = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,2),
    min_df=5,
    max_df=0.7,
    sublinear_tf=True,
    token_pattern=r"(?u)\b[a-z][a-z0-9+\-\.#]+\b",
)
texts = sample['description_cleaned'].fillna('').astype(str).str.lower()
X = vec.fit_transform(texts)
print(f"[V1_H2] TF-IDF shape: {X.shape}", flush=True)

# Truncated SVD to 100 dims
svd = TruncatedSVD(n_components=100, random_state=91)
Xs = svd.fit_transform(X)
# L2 normalize for cosine similarity via dot product
Xs_n = Xs / np.linalg.norm(Xs, axis=1, keepdims=True).clip(1e-12)
print(f"[V1_H2] SVD EV: {svd.explained_variance_ratio_.sum():.4f}", flush=True)

# Compute centroids (trimmed to 90th pct of pairwise-to-centroid cosine).
def trimmed_centroid(vecs):
    """Initial mean; drop bottom 10% distance-to-mean; recompute mean."""
    if len(vecs) < 5:
        return vecs.mean(axis=0)
    c0 = vecs.mean(axis=0)
    c0_n = c0 / (np.linalg.norm(c0) + 1e-12)
    # cos between each row and c0_n
    cos_to_c = vecs @ c0_n
    thresh = np.percentile(cos_to_c, 10)
    keep = cos_to_c >= thresh
    return vecs[keep].mean(axis=0)

results = []
for period in ['2024', '2026']:
    mask_j = (sample['period2'] == period) & (sample['seniority_3level'] == 'junior')
    mask_s = (sample['period2'] == period) & (sample['seniority_3level'] == 'senior')
    n_j = int(mask_j.sum())
    n_s = int(mask_s.sum())
    vj = Xs_n[mask_j.values]
    vs = Xs_n[mask_s.values]
    if n_j < 3 or n_s < 3:
        continue
    cj = trimmed_centroid(vj)
    cs = trimmed_centroid(vs)
    # Cosine on centroids (need to L2 normalize again since trimmed centroid is a mean)
    cj_n = cj / (np.linalg.norm(cj) + 1e-12)
    cs_n = cs / (np.linalg.norm(cs) + 1e-12)
    cos = float(np.dot(cj_n, cs_n))
    results.append({"period": period, "n_junior": n_j, "n_senior": n_s,
                    "tfidf_cos_junior_senior": cos})
    print(f"[V1_H2] {period}: n_j={n_j} n_s={n_s} TF-IDF junior-senior cosine = {cos:.4f}", flush=True)

res = pd.DataFrame(results)
print("\n[V1_H2] === RESULTS ===", flush=True)
print(res.to_string(index=False), flush=True)
delta_v1 = res.loc[res['period']=='2026','tfidf_cos_junior_senior'].iloc[0] - \
           res.loc[res['period']=='2024','tfidf_cos_junior_senior'].iloc[0]
print(f"[V1_H2] Delta V1 = {delta_v1:+.4f}", flush=True)
print(f"[V1_H2] Wave 2 T15 claim: 0.946 (2024) -> 0.863 (2026), delta -0.083", flush=True)

# Verdict
cos_2024_v1 = res.loc[res['period']=='2024','tfidf_cos_junior_senior'].iloc[0]
cos_2026_v1 = res.loc[res['period']=='2026','tfidf_cos_junior_senior'].iloc[0]
# 5% tolerance on absolute magnitude AND delta direction
print(f"\n[V1_H2] V1 verdict:", flush=True)
print(f"  2024 cos: V1 {cos_2024_v1:.4f} vs T15 0.946  (delta {cos_2024_v1 - 0.946:+.4f})", flush=True)
print(f"  2026 cos: V1 {cos_2026_v1:.4f} vs T15 0.863  (delta {cos_2026_v1 - 0.863:+.4f})", flush=True)
print(f"  Direction (negative = diverging): V1 {delta_v1:+.4f} vs T15 -0.083", flush=True)

res.to_csv("exploration/tables/V1/H2_boundary_verification.csv", index=False)
print("\n[V1_H2] Wrote exploration/tables/V1/H2_boundary_verification.csv", flush=True)
