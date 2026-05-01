"""V1 extra - T15 period vs seniority embedding-space dominance check.

Wave 2 claim: period ~180× seniority in embedding space.
Independently verify using shared embeddings.
"""
import duckdb
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

OUT_DIR = Path("/home/jihgaboot/gabor/job-research/exploration/artifacts/V1")

con = duckdb.connect()

# Load embeddings + uid + period + seniority
emb = np.load("/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_embeddings.npy")
idx_df = con.execute("SELECT * FROM '/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_embedding_index.parquet' ORDER BY row_idx").df()
corpus = con.execute("SELECT uid, period, seniority_final FROM '/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_cleaned_text.parquet'").df()

# Join
meta = idx_df.merge(corpus, on="uid")
meta["period_group"] = meta["period"].apply(lambda p: "2024" if str(p).startswith("2024") else "2026")

# Map seniority to J2/S1 bins
def sen_bin(s):
    if s in ("entry", "associate"):
        return "entry"
    if s in ("mid-senior", "director"):
        return "senior"
    return "other"

meta["sen_bin"] = meta["seniority_final"].apply(sen_bin)

# Keep only entry and senior
meta = meta[meta["sen_bin"].isin(["entry", "senior"])].copy()
print(f"Rows: {len(meta)}")
print(meta.groupby(["period_group", "sen_bin"]).size())

# Compute group centroids
centroids = {}
for (pg, sb), sub in meta.groupby(["period_group", "sen_bin"]):
    if len(sub) < 10:
        continue
    row_idx = sub["row_idx"].values
    group_emb = emb[row_idx]
    # Normalize
    norm = group_emb / np.linalg.norm(group_emb, axis=1, keepdims=True)
    centroid = norm.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    centroids[(pg, sb)] = centroid
    print(f"  {pg}-{sb}: n={len(sub)}")

# Compute cross-group cosines
labels = list(centroids.keys())
sim_matrix = np.zeros((len(labels), len(labels)))
for i, a in enumerate(labels):
    for j, b in enumerate(labels):
        sim_matrix[i, j] = float(centroids[a] @ centroids[b])

print("\nGroup centroid pairwise cosines:")
cos_df = pd.DataFrame(sim_matrix, index=[f"{a[0]}-{a[1]}" for a in labels], columns=[f"{a[0]}-{a[1]}" for a in labels])
print(cos_df.round(4).to_string())

# within-period cross-seniority: (2024-entry, 2024-senior), (2026-entry, 2026-senior)
within_period_cross_sen = []
if ("2024", "entry") in centroids and ("2024", "senior") in centroids:
    within_period_cross_sen.append(centroids[("2024", "entry")] @ centroids[("2024", "senior")])
if ("2026", "entry") in centroids and ("2026", "senior") in centroids:
    within_period_cross_sen.append(centroids[("2026", "entry")] @ centroids[("2026", "senior")])

# cross-period same-seniority: (2024-entry, 2026-entry), (2024-senior, 2026-senior)
cross_period_same_sen = []
if ("2024", "entry") in centroids and ("2026", "entry") in centroids:
    cross_period_same_sen.append(centroids[("2024", "entry")] @ centroids[("2026", "entry")])
if ("2024", "senior") in centroids and ("2026", "senior") in centroids:
    cross_period_same_sen.append(centroids[("2024", "senior")] @ centroids[("2026", "senior")])

mean_within_period = float(np.mean(within_period_cross_sen))
mean_cross_period = float(np.mean(cross_period_same_sen))
print(f"\nMean within-period cross-seniority: {mean_within_period:.4f}")
print(f"Mean cross-period same-seniority: {mean_cross_period:.4f}")
# Distance as (1 - cos)
d_within = 1 - mean_within_period
d_cross = 1 - mean_cross_period
print(f"Distance within-period: {d_within:.4f}")
print(f"Distance cross-period: {d_cross:.4f}")
ratio = d_cross / d_within if d_within > 0 else float("inf")
print(f"Period/seniority ratio: {ratio:.1f}x (Wave 2 reported ~180x)")

summary = {
    "mean_within_period_cross_seniority": mean_within_period,
    "mean_cross_period_same_seniority": mean_cross_period,
    "distance_within_period": float(d_within),
    "distance_cross_period": float(d_cross),
    "ratio": float(ratio),
    "within_period_values": [float(x) for x in within_period_cross_sen],
    "cross_period_values": [float(x) for x in cross_period_same_sen],
}
with open(OUT_DIR / "V1_extra_t15_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved to {OUT_DIR / 'V1_extra_t15_summary.json'}")
