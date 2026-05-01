#!/usr/bin/env python3
"""
T28 step 1: Propagate T09 archetype labels to all 63K SWE LinkedIn rows
via nearest-centroid assignment in the sentence-transformer embedding space.

T09 clustered only an 8,000-row sample. T28 needs archetype assignments for
the full SWE LinkedIn corpus so we can decompose entry-share and scope
changes within vs between archetypes at scale.

Approach:
  1. Load all 63,294 SWE LinkedIn embeddings and the 8,000 T09 labels.
  2. Compute centroids for each archetype (excluding noise = -1) from
     labeled rows, L2-normalized.
  3. Assign every row to the archetype with highest cosine similarity.
  4. Flag rows whose max cosine < threshold as 'unclassified' (treat as
     residual/noise). Use per-archetype stability check: hold out 20% of
     labeled rows as validation and report accuracy.
  5. Exclude the known boilerplate artifact archetypes (14 Dice, 20 salary,
     21 Expedia) from downstream analyses — flag them but keep for audit.

Output: exploration/tables/T28/archetype_assignments.parquet
        (uid, archetype, archetype_name, cosine_top, is_artifact, assign_source)
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics.pairwise import cosine_similarity  # not needed but handy

OUT = "exploration/tables/T28"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
emb = np.load("exploration/artifacts/shared/swe_embeddings.npy")
idx = pq.read_table("exploration/artifacts/shared/swe_embedding_index.parquet").to_pandas()
labs = pq.read_table("exploration/artifacts/shared/swe_archetype_labels.parquet").to_pandas()

assert emb.shape[0] == len(idx) == 63294
print(f"Embeddings: {emb.shape}, rows with T09 labels: {len(labs)}")

# Sanity — check L2 norm
norms = np.linalg.norm(emb, axis=1)
print(f"Embedding norm min/max/mean: {norms.min():.4f}/{norms.max():.4f}/{norms.mean():.4f}")
assert np.allclose(norms, 1.0, atol=1e-3), "embeddings should be L2 normalized"

# Merge labels with embedding index positions
idx["pos"] = np.arange(len(idx))
labeled = labs.merge(idx, on="uid", how="inner")
print(f"Labeled rows joined to embeddings: {len(labeled)}")
assert len(labeled) == 8000

# ---------------------------------------------------------------------------
# Identify artifact archetypes from T09 report:
#   T14 Dice reposts, T20 salary boilerplate, T21 Expedia boilerplate
#   (T09 report says: "exclude archetypes 14, 20, 21")
# ---------------------------------------------------------------------------
ARTIFACT_IDS = {14, 20, 21}
archetype_name_map = (
    labeled.drop_duplicates("archetype").set_index("archetype")["archetype_name"].to_dict()
)
# Add name for noise -1
archetype_name_map[-1] = "noise_outliers"
print("Archetype name map:")
for k in sorted(archetype_name_map.keys()):
    print(f"  {k}: {archetype_name_map[k]}")

# ---------------------------------------------------------------------------
# Compute centroids (exclude noise; include artifacts for diagnosis)
# ---------------------------------------------------------------------------
centroids = {}
for arch_id, sub in labeled.groupby("archetype"):
    if arch_id == -1:
        continue
    vecs = emb[sub["pos"].values]
    c = vecs.mean(axis=0)
    c = c / np.linalg.norm(c)
    centroids[arch_id] = c

arch_ids = sorted(centroids.keys())
C = np.stack([centroids[a] for a in arch_ids], axis=0)  # (K, 384)
print(f"Centroids: {C.shape}")

# ---------------------------------------------------------------------------
# Validation: hold-out 20% of labeled non-noise rows, rebuild centroids, test
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
labeled_nn = labeled[labeled["archetype"] != -1].copy()
perm = rng.permutation(len(labeled_nn))
split = int(0.8 * len(labeled_nn))
train_idx = labeled_nn.iloc[perm[:split]]
val_idx = labeled_nn.iloc[perm[split:]]

val_cents = {}
for arch_id, sub in train_idx.groupby("archetype"):
    vecs = emb[sub["pos"].values]
    c = vecs.mean(axis=0)
    c = c / np.linalg.norm(c)
    val_cents[arch_id] = c

val_arch_ids = sorted(val_cents.keys())
Cv = np.stack([val_cents[a] for a in val_arch_ids], axis=0)
val_pos = val_idx["pos"].values
sims = emb[val_pos] @ Cv.T
pred_idx = sims.argmax(axis=1)
pred_arch = np.array(val_arch_ids)[pred_idx]
true_arch = val_idx["archetype"].values
acc = (pred_arch == true_arch).mean()
print(f"\nHeld-out nearest-centroid accuracy: {acc:.3f} (n={len(val_idx)})")

# Per-archetype recall/precision (for rows that were labeled)
val_df = pd.DataFrame({"true": true_arch, "pred": pred_arch})
per_arch = (
    val_df.groupby("true").apply(lambda d: (d["pred"] == d["true"]).mean()).reset_index()
)
per_arch.columns = ["archetype", "recall"]
per_arch["archetype_name"] = per_arch["archetype"].map(archetype_name_map)
per_arch.to_csv(f"{OUT}/centroid_holdout_accuracy.csv", index=False)
print(per_arch.to_string(index=False))

# ---------------------------------------------------------------------------
# Full assignment: all 63,294 rows
# ---------------------------------------------------------------------------
print("\nAssigning full corpus...")
all_sims = emb @ C.T  # (N, K)
top_idx = all_sims.argmax(axis=1)
top_sim = all_sims[np.arange(len(emb)), top_idx]
assigned_arch = np.array(arch_ids)[top_idx]

out = pd.DataFrame({
    "uid": idx["uid"].values,
    "archetype": assigned_arch,
    "cosine_top": top_sim,
})
out["archetype_name"] = out["archetype"].map(archetype_name_map)
out["is_artifact"] = out["archetype"].isin(ARTIFACT_IDS)

# Flag low-confidence assignments
out["is_confident"] = out["cosine_top"] >= 0.35

# For already-labeled rows, restore ground-truth T09 assignment (including -1 noise)
gt = labeled[["uid", "archetype", "archetype_name"]].rename(
    columns={"archetype": "gt_archetype", "archetype_name": "gt_archetype_name"}
)
out = out.merge(gt, on="uid", how="left")
out["assign_source"] = np.where(out["gt_archetype"].notna(), "t09_labeled", "propagated")
# Override with GT
mask = out["gt_archetype"].notna()
out.loc[mask, "archetype"] = out.loc[mask, "gt_archetype"].astype(int)
out.loc[mask, "archetype_name"] = out.loc[mask, "gt_archetype_name"]
out["is_artifact"] = out["archetype"].isin(ARTIFACT_IDS)
out = out.drop(columns=["gt_archetype", "gt_archetype_name"])

print(f"\nAssign source distribution:\n{out['assign_source'].value_counts()}")
print(f"\nArchetype distribution (full corpus):")
print(
    out.groupby(["archetype", "archetype_name"])
    .size()
    .reset_index(name="n")
    .sort_values("n", ascending=False)
    .to_string(index=False)
)

print(f"\nConfident share (cosine>=0.35): {out['is_confident'].mean():.3f}")
print(f"Cosine top distribution:")
print(out["cosine_top"].describe())

out.to_parquet(f"{OUT}/archetype_assignments.parquet", index=False)
print(f"\nSaved: {OUT}/archetype_assignments.parquet")
