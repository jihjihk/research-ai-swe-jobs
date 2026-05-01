"""T28 step 1: project T09 archetype labels (8k) onto full 34k LLM-labeled SWE corpus.

Method: nearest centroid in 384-dim MiniLM embedding space.
- Compute centroid per archetype (mean of embeddings for that archetype's 8k members).
- For each of the 34,102 LLM-labeled rows, assign the archetype whose centroid has
  the highest cosine similarity.
- Save projected labels keyed by uid.

Uses only numpy + pyarrow so fits well under memory budget.
"""
from __future__ import annotations
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

SHARED = Path('/home/jihgaboot/gabor/job-research/exploration/artifacts/shared')
OUT = Path('/home/jihgaboot/gabor/job-research/exploration/artifacts/T28')
OUT.mkdir(parents=True, exist_ok=True)

print('[load] embeddings + index')
idx = pq.read_table(SHARED / 'swe_embedding_index.parquet').to_pandas()
emb = np.load(SHARED / 'swe_embeddings.npy')
print(f'  idx rows {len(idx)} emb shape {emb.shape}')

print('[load] archetype labels (8k)')
labels = pq.read_table(SHARED / 'swe_archetype_labels.parquet').to_pandas()

# merge idx <- labels for lookup
idx_with_lab = idx.merge(labels, on='uid', how='left')
assert len(idx_with_lab) == len(idx)
labeled_mask = idx_with_lab['archetype_name'].notna().values
print(f'  labeled rows {labeled_mask.sum()} / {len(idx)}')

# Compute L2-normalize
norms = np.linalg.norm(emb, axis=1, keepdims=True)
norms[norms == 0] = 1.0
emb_n = emb / norms

# centroids per archetype
arch_names = sorted(labels['archetype_name'].unique().tolist())
print(f'  {len(arch_names)} archetypes')
centroids = np.zeros((len(arch_names), emb.shape[1]), dtype=np.float32)
for i, name in enumerate(arch_names):
    m = (idx_with_lab['archetype_name'] == name).values
    if m.sum() == 0:
        continue
    c = emb[m].mean(axis=0)
    nrm = np.linalg.norm(c)
    if nrm == 0:
        continue
    centroids[i] = c / nrm

# cosine sim all rows to all centroids
print('[compute] cosine similarity 34k x 22')
sims = emb_n @ centroids.T
best = sims.argmax(axis=1)
best_sim = sims[np.arange(len(emb)), best]

out = pd.DataFrame({
    'uid': idx['uid'].values,
    'projected_archetype': [arch_names[i] for i in best],
    'projected_similarity': best_sim.astype(np.float32),
})
# attach original label for audit
out = out.merge(labels[['uid', 'archetype_name']], on='uid', how='left')
out = out.rename(columns={'archetype_name': 'original_archetype'})

# Accuracy on the 8k (how often does nearest centroid recover the assigned archetype?)
sub = out[out['original_archetype'].notna()]
acc = (sub['projected_archetype'] == sub['original_archetype']).mean()
print(f'[sanity] projection accuracy on the 8k originals: {acc:.4f}')

out.to_parquet(OUT / 'projected_archetypes.parquet', index=False)
print(f'  saved {OUT / "projected_archetypes.parquet"}  n={len(out)}')

# distribution summary
dist = out['projected_archetype'].value_counts()
print('\nProjected archetype distribution (34k):')
print(dist)
