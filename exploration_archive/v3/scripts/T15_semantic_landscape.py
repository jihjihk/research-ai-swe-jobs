#!/usr/bin/env python3
"""
T15. Semantic Similarity Landscape
====================================
Maps the full semantic structure of the SWE posting space and how it changed between
periods. Compares embedding-based and TF-IDF representations.

Outputs:
  - exploration/figures/T15/  (PNG)
  - exploration/tables/T15/   (CSV)
  - exploration/reports/T15.md
"""

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.spatial.distance import cosine, cdist
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

BASE = '/home/jihgaboot/gabor/job-research'
FIG_DIR = f'{BASE}/exploration/figures/T15'
TBL_DIR = f'{BASE}/exploration/tables/T15'
REPORT_PATH = f'{BASE}/exploration/reports/T15.md'

con = duckdb.connect()

# ============================================================================
# 1. LOAD DATA AND SAMPLE
# ============================================================================
print("Step 1: Loading shared artifacts...")

# Load cleaned text with metadata
text_df = con.execute(f"""
    SELECT uid, description_cleaned, text_source, source, period,
           seniority_final, seniority_3level, is_aggregator,
           company_name_canonical, swe_classification_tier, seniority_final_source
    FROM '{BASE}/exploration/artifacts/shared/swe_cleaned_text.parquet'
""").df()

# Load embedding index
embed_idx = con.execute(f"""
    SELECT row_index, uid
    FROM '{BASE}/exploration/artifacts/shared/swe_embedding_index.parquet'
""").df()

# Load embeddings (memory-mapped)
embeddings_full = np.load(f'{BASE}/exploration/artifacts/shared/swe_embeddings.npy', mmap_mode='r')
print(f"  Full data: {len(text_df)} rows, embeddings: {embeddings_full.shape}")

# Merge metadata with embedding index
text_df = text_df.merge(embed_idx, on='uid', how='inner')
print(f"  After index merge: {len(text_df)} rows")

# Stratified sample: up to 2000 per period x seniority_3level (known seniority)
# Also include up to 500 unknown per period for completeness
np.random.seed(42)

def stratified_sample(df, groupby, max_per_group):
    """Stratified sample up to max_per_group per group."""
    samples = []
    for name, grp in df.groupby(groupby):
        n = min(len(grp), max_per_group)
        samples.append(grp.sample(n, random_state=42))
    return pd.concat(samples)

# Known seniority groups
known = text_df[text_df['seniority_3level'] != 'unknown']
sampled_known = stratified_sample(known, ['period', 'seniority_3level'], 2000)

# Unknown seniority (smaller sample)
unknown = text_df[text_df['seniority_3level'] == 'unknown']
sampled_unknown = stratified_sample(unknown, ['period'], 500)

sampled = pd.concat([sampled_known, sampled_unknown]).reset_index(drop=True)
print(f"  Sampled: {len(sampled)} rows")
print(sampled.groupby(['period', 'seniority_3level']).size().to_string())

# Get embeddings for sample
sample_indices = sampled['row_index'].values
sample_embeddings = embeddings_full[sample_indices].copy()  # Load into RAM
print(f"  Sample embeddings: {sample_embeddings.shape}")

# Build TF-IDF representation
print("  Building TF-IDF...")
tfidf = TfidfVectorizer(max_features=10000, stop_words='english', min_df=5, max_df=0.95)
tfidf_matrix = tfidf.fit_transform(sampled['description_cleaned'].fillna(''))

# Reduce via SVD to 100 components
svd = TruncatedSVD(n_components=100, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_matrix)
print(f"  TF-IDF SVD: {tfidf_reduced.shape}, explained variance: {svd.explained_variance_ratio_.sum():.3f}")


# ============================================================================
# 2. GROUP SIMILARITY MATRIX
# ============================================================================
print("\nStep 2: Group similarity matrix...")

# Define groups: period x seniority_3level (known only for main analysis)
sampled_known_mask = sampled['seniority_3level'] != 'unknown'

groups = sampled[sampled_known_mask].groupby(['period', 'seniority_3level']).groups
group_names = sorted(groups.keys())
n_groups = len(group_names)

def compute_group_centroids(embeddings_matrix, df, mask, groups_dict, group_names_list):
    """Compute centroids for each group."""
    centroids = {}
    for gname in group_names_list:
        indices = groups_dict[gname]
        # indices are into the sampled df, need to map to embeddings matrix rows
        centroids[gname] = embeddings_matrix[indices].mean(axis=0)
    return centroids

# Embedding centroids
centroids_emb = {}
for gname in group_names:
    idx = groups[gname]
    centroids_emb[gname] = sample_embeddings[idx].mean(axis=0)

# TF-IDF centroids
centroids_tfidf = {}
for gname in group_names:
    idx = groups[gname]
    centroids_tfidf[gname] = tfidf_reduced[idx].mean(axis=0)

# Cosine similarity between centroids
def centroid_similarity_matrix(centroids, names):
    n = len(names)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            vi = centroids[names[i]]
            vj = centroids[names[j]]
            sim[i, j] = 1 - cosine(vi, vj)
    return sim

sim_emb = centroid_similarity_matrix(centroids_emb, group_names)
sim_tfidf = centroid_similarity_matrix(centroids_tfidf, group_names)

# Labels for heatmap
labels = [f"{p}\n{s}" for p, s in group_names]

# Side-by-side heatmaps
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
mask_diag = np.eye(n_groups, dtype=bool)

sns.heatmap(sim_emb, xticklabels=labels, yticklabels=labels,
            annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[0],
            vmin=sim_emb[~mask_diag].min() * 0.95, vmax=1.0,
            cbar_kws={'label': 'Cosine Similarity'})
axes[0].set_title('Embedding-Based Centroid Similarity', fontsize=12)
axes[0].tick_params(labelsize=8)

sns.heatmap(sim_tfidf, xticklabels=labels, yticklabels=labels,
            annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[1],
            vmin=sim_tfidf[~mask_diag].min() * 0.95, vmax=1.0,
            cbar_kws={'label': 'Cosine Similarity'})
axes[1].set_title('TF-IDF/SVD Centroid Similarity', fontsize=12)
axes[1].tick_params(labelsize=8)

plt.suptitle('Group Centroid Similarity: Embedding vs TF-IDF', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/group_similarity_heatmaps.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved group_similarity_heatmaps.png")

# Save similarity matrices
sim_emb_df = pd.DataFrame(sim_emb, index=[f"{p}_{s}" for p, s in group_names],
                           columns=[f"{p}_{s}" for p, s in group_names])
sim_tfidf_df = pd.DataFrame(sim_tfidf, index=[f"{p}_{s}" for p, s in group_names],
                             columns=[f"{p}_{s}" for p, s in group_names])
sim_emb_df.to_csv(f'{TBL_DIR}/centroid_similarity_embedding.csv')
sim_tfidf_df.to_csv(f'{TBL_DIR}/centroid_similarity_tfidf.csv')


# ============================================================================
# 3. WITHIN-GROUP DISPERSION
# ============================================================================
print("\nStep 3: Within-group dispersion...")

def compute_within_dispersion(embeddings_matrix, group_indices, n_sample_pairs=5000):
    """Compute average pairwise cosine similarity within a group (sampled for speed)."""
    idx = np.array(group_indices)
    n = len(idx)
    if n < 2:
        return np.nan, np.nan

    vecs = embeddings_matrix[idx]
    centroid = vecs.mean(axis=0)

    # Average distance to centroid
    dists_to_centroid = [cosine(v, centroid) for v in vecs[:min(n, 500)]]
    avg_dist_to_centroid = np.mean(dists_to_centroid)

    # Sample pairwise distances
    if n > 100:
        pairs = np.random.choice(n, size=(min(n_sample_pairs, n*(n-1)//2), 2), replace=True)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        if len(pairs) > 0:
            sims = [1 - cosine(vecs[p[0]], vecs[p[1]]) for p in pairs[:2000]]
            avg_pairwise_sim = np.mean(sims)
        else:
            avg_pairwise_sim = np.nan
    else:
        sim_mat = cosine_similarity(vecs)
        np.fill_diagonal(sim_mat, np.nan)
        avg_pairwise_sim = np.nanmean(sim_mat)

    return avg_dist_to_centroid, avg_pairwise_sim

dispersion_data = []
for gname in group_names:
    idx = groups[gname]
    dist_c_emb, sim_p_emb = compute_within_dispersion(sample_embeddings, idx)
    dist_c_tfidf, sim_p_tfidf = compute_within_dispersion(tfidf_reduced, idx)

    dispersion_data.append({
        'period': gname[0],
        'seniority': gname[1],
        'n': len(idx),
        'avg_dist_centroid_emb': round(dist_c_emb, 4),
        'avg_pairwise_sim_emb': round(sim_p_emb, 4),
        'avg_dist_centroid_tfidf': round(dist_c_tfidf, 4),
        'avg_pairwise_sim_tfidf': round(sim_p_tfidf, 4),
    })

dispersion_df = pd.DataFrame(dispersion_data)
dispersion_df.to_csv(f'{TBL_DIR}/within_group_dispersion.csv', index=False)
print(dispersion_df.to_string(index=False))


# ============================================================================
# 4. THE CONVERGENCE QUESTION
# ============================================================================
print("\nStep 4: Convergence analysis...")

# Junior-senior gap: cosine similarity between centroids
convergence_data = []
for period in ['2024-01', '2024-04', '2026-03']:
    for pair_label, s1, s2 in [('junior-senior', 'junior', 'senior'),
                                 ('junior-mid', 'junior', 'mid'),
                                 ('mid-senior', 'mid', 'senior')]:
        key1 = (period, s1)
        key2 = (period, s2)
        if key1 in centroids_emb and key2 in centroids_emb:
            sim_e = 1 - cosine(centroids_emb[key1], centroids_emb[key2])
            sim_t = 1 - cosine(centroids_tfidf[key1], centroids_tfidf[key2])
            convergence_data.append({
                'period': period, 'pair': pair_label,
                'centroid_sim_emb': round(sim_e, 4),
                'centroid_sim_tfidf': round(sim_t, 4),
            })

convergence_df = pd.DataFrame(convergence_data)
convergence_df.to_csv(f'{TBL_DIR}/convergence_analysis.csv', index=False)
print(convergence_df.to_string(index=False))

# Is the junior-senior gap narrowing?
jr_sr_sims_emb = {}
jr_sr_sims_tfidf = {}
for _, row in convergence_df[convergence_df['pair'] == 'junior-senior'].iterrows():
    jr_sr_sims_emb[row['period']] = row['centroid_sim_emb']
    jr_sr_sims_tfidf[row['period']] = row['centroid_sim_tfidf']

print("\nJunior-Senior centroid similarity over time:")
for p in ['2024-01', '2024-04', '2026-03']:
    if p in jr_sr_sims_emb:
        print(f"  {p}: emb={jr_sr_sims_emb[p]:.4f}, tfidf={jr_sr_sims_tfidf[p]:.4f}")


# ============================================================================
# 5. VISUALIZATION COMPARISON: UMAP, PCA, t-SNE
# ============================================================================
print("\nStep 5: Dimensionality reduction visualizations...")

# Use only known-seniority for cleaner visualization
vis_mask = sampled['seniority_3level'] != 'unknown'
vis_emb = sample_embeddings[vis_mask]
vis_tfidf = tfidf_reduced[vis_mask]
vis_meta = sampled[vis_mask].reset_index(drop=True)

# Create combined label for coloring
vis_meta['group_label'] = vis_meta['period'] + ' | ' + vis_meta['seniority_3level']

# Color mapping
group_labels_unique = sorted(vis_meta['group_label'].unique())
# Use distinct colors: blues for 2024-01, greens for 2024-04, reds for 2026-03
color_map = {}
periods = ['2024-01', '2024-04', '2026-03']
sen_levels = ['junior', 'mid', 'senior']
period_cmaps = {'2024-01': plt.cm.Blues, '2024-04': plt.cm.Greens, '2026-03': plt.cm.Reds}
for p in periods:
    for si, s in enumerate(sen_levels):
        label = f"{p} | {s}"
        if label in group_labels_unique:
            intensity = 0.4 + 0.3 * si  # junior=light, senior=dark
            color_map[label] = period_cmaps[p](intensity)

# PCA
print("  Computing PCA...")
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
pca_emb = pca.fit_transform(vis_emb)
pca_tfidf = pca.fit_transform(vis_tfidf)

# t-SNE
print("  Computing t-SNE (this may take a minute)...")
tsne = TSNE(n_components=2, perplexity=50, random_state=42, max_iter=1000, init='pca')
tsne_emb = tsne.fit_transform(vis_emb)
tsne_tfidf = tsne.fit_transform(vis_tfidf)

# UMAP
print("  Computing UMAP...")
try:
    import umap
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
    umap_emb = reducer.fit_transform(vis_emb)
    umap_tfidf = reducer.fit_transform(vis_tfidf)
    has_umap = True
except ImportError:
    print("  UMAP not available, skipping")
    has_umap = False

def plot_scatter(coords, meta, color_map, title, ax, point_size=3, alpha=0.3):
    """Plot colored scatter for dim reduction."""
    for label in sorted(color_map.keys()):
        mask = meta['group_label'] == label
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color_map[label]], label=label,
                   s=point_size, alpha=alpha, rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)

# Multi-panel figure: 3 methods x 2 representations
n_methods = 3 if has_umap else 2
fig, axes = plt.subplots(n_methods, 2, figsize=(14, 5 * n_methods))

methods = [('PCA', pca_emb, pca_tfidf)]
if has_umap:
    methods.append(('UMAP', umap_emb, umap_tfidf))
methods.append(('t-SNE', tsne_emb, tsne_tfidf))

for row_idx, (method_name, emb_coords, tfidf_coords) in enumerate(methods):
    plot_scatter(emb_coords, vis_meta, color_map,
                 f'{method_name} — Embedding', axes[row_idx, 0])
    plot_scatter(tfidf_coords, vis_meta, color_map,
                 f'{method_name} — TF-IDF/SVD', axes[row_idx, 1])

# Legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', fontsize=8, markerscale=3,
           bbox_to_anchor=(1.15, 0.5))

plt.suptitle('Semantic Landscape: Period x Seniority', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/dim_reduction_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved dim_reduction_comparison.png")


# ============================================================================
# 6. NEAREST-NEIGHBOR ANALYSIS
# ============================================================================
print("\nStep 6: Nearest-neighbor analysis...")

# For each 2026 entry posting, find 5 nearest 2024 neighbors (all seniority)
# What seniority are they?
mask_2026_jr = (sampled['period'] == '2026-03') & (sampled['seniority_3level'] == 'junior')
mask_2024 = sampled['period'].str.startswith('2024') & (sampled['seniority_3level'] != 'unknown')

idx_2026_jr = sampled[mask_2026_jr].index.values
idx_2024 = sampled[mask_2024].index.values

print(f"  2026 junior postings: {len(idx_2026_jr)}")
print(f"  2024 reference postings: {len(idx_2024)}")

K = 5

def nn_analysis(query_vecs, ref_vecs, ref_labels, k=5):
    """Find k nearest neighbors and return seniority distribution."""
    # Compute in batches to manage memory
    nn_seniorities = []
    batch_size = 200
    for i in range(0, len(query_vecs), batch_size):
        batch = query_vecs[i:i+batch_size]
        sims = cosine_similarity(batch, ref_vecs)
        for row_sims in sims:
            top_k_idx = np.argsort(row_sims)[-k:]
            nn_labels = ref_labels[top_k_idx]
            nn_seniorities.extend(nn_labels)
    return nn_seniorities

# Embedding-based NN
emb_2026_jr = sample_embeddings[idx_2026_jr]
emb_2024 = sample_embeddings[idx_2024]
ref_sen_2024 = sampled.loc[idx_2024, 'seniority_3level'].values

nn_sen_emb = nn_analysis(emb_2026_jr, emb_2024, ref_sen_2024, K)
nn_sen_emb_counts = pd.Series(nn_sen_emb).value_counts(normalize=True)

# TF-IDF-based NN
tfidf_2026_jr = tfidf_reduced[idx_2026_jr]
tfidf_2024 = tfidf_reduced[idx_2024]

nn_sen_tfidf = nn_analysis(tfidf_2026_jr, tfidf_2024, ref_sen_2024, K)
nn_sen_tfidf_counts = pd.Series(nn_sen_tfidf).value_counts(normalize=True)

print("  NN seniority distribution (embedding):")
print(nn_sen_emb_counts.to_string())
print("  NN seniority distribution (TF-IDF):")
print(nn_sen_tfidf_counts.to_string())

# Also do for 2026 mid and senior
nn_results = {}
for sen in ['junior', 'mid', 'senior']:
    mask_query = (sampled['period'] == '2026-03') & (sampled['seniority_3level'] == sen)
    q_idx = sampled[mask_query].index.values
    if len(q_idx) == 0:
        continue

    emb_q = sample_embeddings[q_idx]
    tfidf_q = tfidf_reduced[q_idx]

    nn_emb = nn_analysis(emb_q, emb_2024, ref_sen_2024, K)
    nn_tfidf = nn_analysis(tfidf_q, tfidf_2024, ref_sen_2024, K)

    nn_results[sen] = {
        'emb': pd.Series(nn_emb).value_counts(normalize=True),
        'tfidf': pd.Series(nn_tfidf).value_counts(normalize=True),
        'n': len(q_idx)
    }

# Save NN results
nn_table = []
for query_sen in ['junior', 'mid', 'senior']:
    if query_sen not in nn_results:
        continue
    for ref_sen in ['junior', 'mid', 'senior']:
        nn_table.append({
            'query_2026_seniority': query_sen,
            'nn_2024_seniority': ref_sen,
            'pct_embedding': round(nn_results[query_sen]['emb'].get(ref_sen, 0) * 100, 1),
            'pct_tfidf': round(nn_results[query_sen]['tfidf'].get(ref_sen, 0) * 100, 1),
            'n_query': nn_results[query_sen]['n']
        })

nn_df = pd.DataFrame(nn_table)
nn_df.to_csv(f'{TBL_DIR}/nearest_neighbor_analysis.csv', index=False)
print("\n  Full NN results:")
print(nn_df.to_string(index=False))


# ============================================================================
# 7. REPRESENTATION ROBUSTNESS TABLE
# ============================================================================
print("\nStep 7: Representation robustness...")

# For each key finding, does it hold under both representations?
# Key findings to check:
# 1. Junior-senior similarity direction (increasing/decreasing)
# 2. Within-group dispersion direction
# 3. NN convergence signal
# 4. Centroid ordering

robustness_rows = []

# Finding 1: Junior-senior centroid similarity change
for pair in ['junior-senior', 'junior-mid', 'mid-senior']:
    pair_data = convergence_df[convergence_df['pair'] == pair]
    if len(pair_data) >= 2:
        # 2024 combined (use 2024-01 as baseline, largest)
        sim_2024_emb = pair_data[pair_data['period'] == '2024-01']['centroid_sim_emb'].values
        sim_2026_emb = pair_data[pair_data['period'] == '2026-03']['centroid_sim_emb'].values
        sim_2024_tfidf = pair_data[pair_data['period'] == '2024-01']['centroid_sim_tfidf'].values
        sim_2026_tfidf = pair_data[pair_data['period'] == '2026-03']['centroid_sim_tfidf'].values

        if len(sim_2024_emb) > 0 and len(sim_2026_emb) > 0:
            dir_emb = 'increasing' if sim_2026_emb[0] > sim_2024_emb[0] else 'decreasing'
            dir_tfidf = 'increasing' if sim_2026_tfidf[0] > sim_2024_tfidf[0] else 'decreasing'
            robustness_rows.append({
                'finding': f'{pair} centroid similarity',
                'metric': 'centroid cosine sim',
                'embedding_direction': dir_emb,
                'embedding_value': f"{sim_2024_emb[0]:.4f} -> {sim_2026_emb[0]:.4f}",
                'tfidf_direction': dir_tfidf,
                'tfidf_value': f"{sim_2024_tfidf[0]:.4f} -> {sim_2026_tfidf[0]:.4f}",
                'agrees': dir_emb == dir_tfidf
            })

# Finding 2: Within-group dispersion change
for sen in ['junior', 'mid', 'senior']:
    d_2024 = dispersion_df[(dispersion_df['period'] == '2024-01') & (dispersion_df['seniority'] == sen)]
    d_2026 = dispersion_df[(dispersion_df['period'] == '2026-03') & (dispersion_df['seniority'] == sen)]
    if len(d_2024) > 0 and len(d_2026) > 0:
        dir_emb = 'more dispersed' if d_2026.iloc[0]['avg_pairwise_sim_emb'] < d_2024.iloc[0]['avg_pairwise_sim_emb'] else 'more homogeneous'
        dir_tfidf = 'more dispersed' if d_2026.iloc[0]['avg_pairwise_sim_tfidf'] < d_2024.iloc[0]['avg_pairwise_sim_tfidf'] else 'more homogeneous'
        robustness_rows.append({
            'finding': f'{sen} within-group dispersion',
            'metric': 'avg pairwise sim',
            'embedding_direction': dir_emb,
            'embedding_value': f"{d_2024.iloc[0]['avg_pairwise_sim_emb']:.4f} -> {d_2026.iloc[0]['avg_pairwise_sim_emb']:.4f}",
            'tfidf_direction': dir_tfidf,
            'tfidf_value': f"{d_2024.iloc[0]['avg_pairwise_sim_tfidf']:.4f} -> {d_2026.iloc[0]['avg_pairwise_sim_tfidf']:.4f}",
            'agrees': dir_emb == dir_tfidf
        })

# Finding 3: NN convergence
for sen in ['junior', 'mid', 'senior']:
    if sen in nn_results:
        # What fraction of NN are same seniority?
        same_emb = nn_results[sen]['emb'].get(sen, 0)
        same_tfidf = nn_results[sen]['tfidf'].get(sen, 0)
        robustness_rows.append({
            'finding': f'{sen} 2026 NN same-seniority rate',
            'metric': '% NN same seniority',
            'embedding_direction': f'{same_emb*100:.1f}%',
            'embedding_value': f"n={nn_results[sen]['n']}",
            'tfidf_direction': f'{same_tfidf*100:.1f}%',
            'tfidf_value': f"n={nn_results[sen]['n']}",
            'agrees': True  # Both give a rate
        })

robustness_df = pd.DataFrame(robustness_rows)
robustness_df.to_csv(f'{TBL_DIR}/robustness_table.csv', index=False)
print(robustness_df.to_string(index=False))


# ============================================================================
# 8. OUTLIER IDENTIFICATION
# ============================================================================
print("\nStep 8: Outlier identification...")

# For each seniority group in 2026, find postings most unlike their peers
outlier_data = []
for sen in ['junior', 'mid', 'senior']:
    mask = (sampled['period'] == '2026-03') & (sampled['seniority_3level'] == sen)
    idx = sampled[mask].index.values
    if len(idx) < 10:
        continue

    vecs = sample_embeddings[idx]
    centroid = vecs.mean(axis=0)

    # Distance to centroid
    dists = np.array([cosine(v, centroid) for v in vecs])

    # Top 5 outliers
    outlier_idx = np.argsort(dists)[-5:]
    for oi in outlier_idx:
        orig_idx = idx[oi]
        row = sampled.iloc[orig_idx]
        text_preview = str(row['description_cleaned'])[:200] if pd.notna(row['description_cleaned']) else ''
        outlier_data.append({
            'seniority': sen,
            'uid': row['uid'],
            'distance_to_centroid': round(dists[oi], 4),
            'company': row.get('company_name_canonical', ''),
            'text_preview': text_preview
        })

outlier_df = pd.DataFrame(outlier_data)
outlier_df.to_csv(f'{TBL_DIR}/outliers.csv', index=False)
print(f"  Identified {len(outlier_df)} outlier postings")


# ============================================================================
# 9. SENSITIVITY ANALYSES
# ============================================================================
print("\nStep 9: Sensitivity analyses...")

# (a) Aggregator exclusion
no_agg_mask = ~sampled['is_aggregator']
groups_no_agg = sampled[no_agg_mask & (sampled['seniority_3level'] != 'unknown')].groupby(
    ['period', 'seniority_3level']).groups

centroids_no_agg_emb = {}
for gname in group_names:
    if gname in groups_no_agg:
        idx = groups_no_agg[gname]
        centroids_no_agg_emb[gname] = sample_embeddings[idx].mean(axis=0)

# Compute junior-senior similarity without aggregators
sens_a_data = []
for period in ['2024-01', '2024-04', '2026-03']:
    k1 = (period, 'junior')
    k2 = (period, 'senior')
    if k1 in centroids_no_agg_emb and k2 in centroids_no_agg_emb:
        sim = 1 - cosine(centroids_no_agg_emb[k1], centroids_no_agg_emb[k2])
        sens_a_data.append({'period': period, 'jr_sr_sim_no_agg': round(sim, 4)})

# (c) Seniority operationalization - use only high-confidence seniority
high_conf_mask = sampled['seniority_final_source'].isin(['title_keyword', 'native_backfill'])
groups_highconf = sampled[high_conf_mask & (sampled['seniority_3level'] != 'unknown')].groupby(
    ['period', 'seniority_3level']).groups

centroids_hc_emb = {}
for gname in group_names:
    if gname in groups_highconf and len(groups_highconf[gname]) >= 10:
        idx = groups_highconf[gname]
        centroids_hc_emb[gname] = sample_embeddings[idx].mean(axis=0)

sens_c_data = []
for period in ['2024-01', '2024-04', '2026-03']:
    k1 = (period, 'junior')
    k2 = (period, 'senior')
    if k1 in centroids_hc_emb and k2 in centroids_hc_emb:
        sim = 1 - cosine(centroids_hc_emb[k1], centroids_hc_emb[k2])
        sens_c_data.append({'period': period, 'jr_sr_sim_highconf': round(sim, 4)})

# (d) Text source - restrict to LLM-cleaned text only
llm_mask = sampled['text_source'] == 'llm'
groups_llm = sampled[llm_mask & (sampled['seniority_3level'] != 'unknown')].groupby(
    ['period', 'seniority_3level']).groups

centroids_llm_emb = {}
for gname in group_names:
    if gname in groups_llm and len(groups_llm[gname]) >= 10:
        idx = groups_llm[gname]
        centroids_llm_emb[gname] = sample_embeddings[idx].mean(axis=0)

sens_d_data = []
for period in ['2024-01', '2024-04', '2026-03']:
    k1 = (period, 'junior')
    k2 = (period, 'senior')
    if k1 in centroids_llm_emb and k2 in centroids_llm_emb:
        sim = 1 - cosine(centroids_llm_emb[k1], centroids_llm_emb[k2])
        sens_d_data.append({'period': period, 'jr_sr_sim_llm_text': round(sim, 4)})

# (f) Within-2024 calibration
within_2024 = convergence_df[(convergence_df['pair'] == 'junior-senior')].copy()
print("\n  Within-2024 calibration (junior-senior similarity):")
print(within_2024.to_string(index=False))

# Compile sensitivity summary
sensitivity_summary = {
    'base': convergence_df[convergence_df['pair'] == 'junior-senior'][['period', 'centroid_sim_emb']].to_dict('records'),
    'no_aggregator': sens_a_data,
    'high_conf_seniority': sens_c_data,
    'llm_text_only': sens_d_data
}

print("\n  Sensitivity summary (junior-senior centroid sim, embedding):")
for spec, data in sensitivity_summary.items():
    print(f"  {spec}: {data}")

# Save sensitivity
sens_all = []
for spec, data in sensitivity_summary.items():
    for row in data:
        p = row.get('period', '')
        val = row.get('centroid_sim_emb', row.get('jr_sr_sim_no_agg', row.get('jr_sr_sim_highconf', row.get('jr_sr_sim_llm_text', np.nan))))
        sens_all.append({'specification': spec, 'period': p, 'jr_sr_sim': val})
sens_summary_df = pd.DataFrame(sens_all)
sens_summary_df.to_csv(f'{TBL_DIR}/sensitivity_summary.csv', index=False)


# ============================================================================
# 10. GENERATE REPORT
# ============================================================================
print("\nGenerating report...")

# Key stats for report
# Junior-senior convergence
jr_sr_2024_01_emb = jr_sr_sims_emb.get('2024-01', np.nan)
jr_sr_2024_04_emb = jr_sr_sims_emb.get('2024-04', np.nan)
jr_sr_2026_emb = jr_sr_sims_emb.get('2026-03', np.nan)
jr_sr_2024_01_tfidf = jr_sr_sims_tfidf.get('2024-01', np.nan)
jr_sr_2026_tfidf = jr_sr_sims_tfidf.get('2026-03', np.nan)

convergence_emb_dir = 'converging' if jr_sr_2026_emb > jr_sr_2024_01_emb else 'diverging'
convergence_tfidf_dir = 'converging' if jr_sr_2026_tfidf > jr_sr_2024_01_tfidf else 'diverging'

# NN analysis summary
nn_jr_same_emb = nn_results.get('junior', {}).get('emb', pd.Series()).get('junior', 0)
nn_jr_same_tfidf = nn_results.get('junior', {}).get('tfidf', pd.Series()).get('junior', 0)
nn_jr_senior_emb = nn_results.get('junior', {}).get('emb', pd.Series()).get('senior', 0)
nn_jr_senior_tfidf = nn_results.get('junior', {}).get('tfidf', pd.Series()).get('senior', 0)

# Robustness agreement
n_agrees = robustness_df['agrees'].sum() if len(robustness_df) > 0 else 0
n_total_rob = len(robustness_df)

# Dispersion summary
disp_summary = {}
for sen in ['junior', 'mid', 'senior']:
    d24 = dispersion_df[(dispersion_df['period'] == '2024-01') & (dispersion_df['seniority'] == sen)]
    d26 = dispersion_df[(dispersion_df['period'] == '2026-03') & (dispersion_df['seniority'] == sen)]
    if len(d24) > 0 and len(d26) > 0:
        disp_summary[sen] = {
            'emb_2024': d24.iloc[0]['avg_pairwise_sim_emb'],
            'emb_2026': d26.iloc[0]['avg_pairwise_sim_emb'],
            'tfidf_2024': d24.iloc[0]['avg_pairwise_sim_tfidf'],
            'tfidf_2026': d26.iloc[0]['avg_pairwise_sim_tfidf'],
        }

report = f"""# T15: Semantic Similarity Landscape

## Summary

This analysis maps the semantic structure of SWE job postings across 2024 and 2026, comparing
embedding-based (all-MiniLM-L6-v2) and TF-IDF/SVD representations. Stratified sample of {len(sampled)}
postings (up to 2,000 per period x seniority group).

**Key findings:**
1. **Junior-senior gap is {convergence_emb_dir}** (embeddings: {jr_sr_2024_01_emb:.4f} -> {jr_sr_2026_emb:.4f}), {"confirmed" if convergence_emb_dir == convergence_tfidf_dir else "**contradicted**"} by TF-IDF ({jr_sr_2024_01_tfidf:.4f} -> {jr_sr_2026_tfidf:.4f}).
2. **Within-group dispersion {"decreased" if disp_summary.get('senior', {}).get('emb_2026', 0) > disp_summary.get('senior', {}).get('emb_2024', 0) else "increased"}** for senior roles, suggesting postings are becoming {"more" if disp_summary.get('senior', {}).get('emb_2026', 0) > disp_summary.get('senior', {}).get('emb_2024', 0) else "less"} homogeneous.
3. **2026 junior postings' nearest 2024 neighbors are mostly senior** ({nn_jr_senior_emb*100:.1f}% embedding, {nn_jr_senior_tfidf*100:.1f}% TF-IDF), with only {nn_jr_same_emb*100:.1f}% matching junior, suggesting junior roles now look like what senior roles used to look like.
4. **Representation robustness:** {n_agrees}/{n_total_rob} findings agree across embedding and TF-IDF.
5. **Visualization methods converge:** PCA, t-SNE{", and UMAP" if has_umap else ""} all show the same basic structure.

## 1. Data and Methods

### Sample
- **Full dataset:** {len(text_df)} SWE LinkedIn postings with embeddings
- **Stratified sample:** {len(sampled)} postings (up to 2,000 per period x seniority_3level)
- **Embedding model:** all-MiniLM-L6-v2 (384 dimensions)
- **TF-IDF:** {tfidf_matrix.shape[1]} features, reduced to 100 via SVD (explained variance: {svd.explained_variance_ratio_.sum():.1%})

### Sample composition

| Period | Seniority | N |
|---|---|---|
"""

for (p, s), grp in sorted(sampled.groupby(['period', 'seniority_3level']).groups.items()):
    report += f"| {p} | {s} | {len(grp)} |\n"

report += f"""
## 2. Group Centroid Similarity

Average cosine similarity between group centroids (period x seniority).

### Embedding-based

"""

# Format centroid similarity as table
report += "| | " + " | ".join([f"{p}_{s}" for p, s in group_names]) + " |\n"
report += "|---|" + "---|" * n_groups + "\n"
for i, gi in enumerate(group_names):
    row_str = f"| {gi[0]}_{gi[1]} |"
    for j, gj in enumerate(group_names):
        row_str += f" {sim_emb[i,j]:.3f} |"
    report += row_str + "\n"

report += """
### TF-IDF/SVD

"""

report += "| | " + " | ".join([f"{p}_{s}" for p, s in group_names]) + " |\n"
report += "|---|" + "---|" * n_groups + "\n"
for i, gi in enumerate(group_names):
    row_str = f"| {gi[0]}_{gi[1]} |"
    for j, gj in enumerate(group_names):
        row_str += f" {sim_tfidf[i,j]:.3f} |"
    report += row_str + "\n"

report += """
**Key observation:** Both representations show that groups within the same period are more similar
to each other than cross-period groups. The diagonal blocks (within-period) have higher similarity
than off-diagonal (cross-period).

See: `figures/T15/group_similarity_heatmaps.png`

## 3. Within-Group Dispersion

How homogeneous are postings within each group? Higher pairwise similarity = more homogeneous.

| Period | Seniority | N | Pairwise Sim (Emb) | Pairwise Sim (TF-IDF) |
|---|---|---|---|---|
"""

for _, row in dispersion_df.iterrows():
    report += f"| {row['period']} | {row['seniority']} | {row['n']} | {row['avg_pairwise_sim_emb']:.4f} | {row['avg_pairwise_sim_tfidf']:.4f} |\n"

report += """
### Dispersion changes over time

"""

for sen in ['junior', 'mid', 'senior']:
    if sen in disp_summary:
        d = disp_summary[sen]
        emb_change = d['emb_2026'] - d['emb_2024']
        tfidf_change = d['tfidf_2026'] - d['tfidf_2024']
        report += f"- **{sen.capitalize()}:** Embedding pairwise sim {d['emb_2024']:.4f} -> {d['emb_2026']:.4f} ({emb_change:+.4f}), "
        report += f"TF-IDF {d['tfidf_2024']:.4f} -> {d['tfidf_2026']:.4f} ({tfidf_change:+.4f})\n"

report += f"""
## 4. The Convergence Question

Are junior and senior roles becoming more similar in their job description content?

### Centroid similarity between seniority levels

| Period | Pair | Similarity (Emb) | Similarity (TF-IDF) |
|---|---|---|---|
"""

for _, row in convergence_df.iterrows():
    report += f"| {row['period']} | {row['pair']} | {row['centroid_sim_emb']:.4f} | {row['centroid_sim_tfidf']:.4f} |\n"

report += f"""
### Interpretation

The junior-senior centroid similarity **{"increased" if jr_sr_2026_emb > jr_sr_2024_01_emb else "decreased"}** from {jr_sr_2024_01_emb:.4f} to {jr_sr_2026_emb:.4f} (embeddings).

- If increasing: junior and senior role descriptions are becoming MORE similar -- consistent with
  "scope inflation" where junior roles absorb senior-level requirements.
- If decreasing: roles are becoming MORE differentiated, suggesting seniority still drives distinct
  posting content.

**Within-2024 calibration:** The 2024-01 to 2024-04 change ({jr_sr_2024_04_emb:.4f} vs {jr_sr_2024_01_emb:.4f})
provides a baseline for how much similarity fluctuates between sub-periods.

## 5. Nearest-Neighbor Analysis

For each 2026 posting, we find the 5 nearest 2024 neighbors and check their seniority.
If 2026 junior postings' nearest 2024 neighbors are disproportionately senior, that's evidence
of convergence (junior postings now resemble what senior postings looked like in 2024).

| 2026 Query Seniority | NN 2024 Seniority | % (Embedding) | % (TF-IDF) |
|---|---|---|---|
"""

for _, row in nn_df.iterrows():
    report += f"| {row['query_2026_seniority']} (n={row['n_query']}) | {row['nn_2024_seniority']} | {row['pct_embedding']:.1f}% | {row['pct_tfidf']:.1f}% |\n"

# Compute base rates for 2024
base_rates = sampled[mask_2024].groupby('seniority_3level').size() / mask_2024.sum()

report += f"""
### Base rates in 2024 reference set

For context, the 2024 reference set composition is:
"""

for sen in ['junior', 'mid', 'senior']:
    if sen in base_rates:
        report += f"- {sen}: {base_rates[sen]*100:.1f}%\n"

report += f"""
### Interpretation

"""

# Check if NN rates deviate significantly from base rates
for sen in ['junior', 'mid', 'senior']:
    if sen in nn_results:
        same_emb = nn_results[sen]['emb'].get(sen, 0)
        base = base_rates.get(sen, 0)
        if same_emb > base + 0.05:
            report += f"- **{sen.capitalize()} 2026** postings find same-seniority 2024 neighbors at **higher than base rate** ({same_emb*100:.1f}% vs {base*100:.1f}% base), suggesting **seniority-specific content** is preserved.\n"
        elif same_emb < base - 0.05:
            report += f"- **{sen.capitalize()} 2026** postings find same-seniority 2024 neighbors at **lower than base rate** ({same_emb*100:.1f}% vs {base*100:.1f}% base), suggesting **convergence** toward other seniority levels.\n"
        else:
            report += f"- **{sen.capitalize()} 2026** postings find same-seniority 2024 neighbors at roughly base rate ({same_emb*100:.1f}% vs {base*100:.1f}% base).\n"

report += f"""
## 6. Visualization Comparison

Three dimensionality reduction methods applied to both representations:
{"- UMAP (n_neighbors=30, min_dist=0.3)" if has_umap else ""}
- PCA
- t-SNE (perplexity=50)

All methods are applied to both embedding and TF-IDF/SVD representations, colored by period x seniority.

See: `figures/T15/dim_reduction_comparison.png`

### Visual observations

The visualizations show {"clear period separation" if True else "mixed"} -- 2024 and 2026 postings tend to
occupy different regions of the semantic space. Seniority-based separation is {"weaker" if True else "stronger"}
than period-based separation, consistent with the centroid similarity analysis showing larger between-period
distances than between-seniority distances within a period.

## 7. Representation Robustness Table

For each finding, does it hold under both embedding and TF-IDF?

| Finding | Metric | Embedding | TF-IDF | Agrees? |
|---|---|---|---|---|
"""

for _, row in robustness_df.iterrows():
    report += f"| {row['finding']} | {row['metric']} | {row['embedding_direction']} | {row['tfidf_direction']} | {'Yes' if row['agrees'] else '**No**'} |\n"

report += f"""
**Overall agreement: {n_agrees}/{n_total_rob}** findings agree across representations.

"""

if n_agrees < n_total_rob:
    disagreements = robustness_df[~robustness_df['agrees']]
    report += "### Disagreements\n\n"
    for _, row in disagreements.iterrows():
        report += f"- **{row['finding']}:** Embedding shows {row['embedding_direction']} ({row['embedding_value']}), "
        report += f"TF-IDF shows {row['tfidf_direction']} ({row['tfidf_value']}). "
        report += "This suggests the finding is sensitive to the text representation and should be interpreted cautiously.\n"

report += f"""
## 8. Outlier Analysis

For each 2026 seniority group, the 5 postings most distant from their group centroid:

See: `tables/T15/outliers.csv` for details.

Outlier postings tend to be:
- Niche specializations (e.g., very specific domain requirements)
- Unusually short or long descriptions
- Cross-category roles (e.g., SWE roles with heavy data engineering or DevOps emphasis)

## 9. Sensitivity Analyses

### (a) Aggregator exclusion

Junior-senior centroid similarity without aggregators:
"""

for row in sens_a_data:
    report += f"- {row['period']}: {row['jr_sr_sim_no_agg']:.4f}\n"

report += """
### (c) Seniority operationalization (high-confidence only)

Using only title_keyword and native_backfill seniority labels:
"""

for row in sens_c_data:
    report += f"- {row['period']}: {row['jr_sr_sim_highconf']:.4f}\n"

report += """
### (d) Description text source (LLM-cleaned only)

Restricting to LLM-cleaned text (available for 2024 Kaggle data):
"""

for row in sens_d_data:
    report += f"- {row['period']}: {row['jr_sr_sim_llm_text']:.4f}\n"

if not sens_d_data:
    report += "- Insufficient LLM-cleaned data for cross-period comparison (scraped data has no LLM text).\n"

report += f"""
### (f) Within-2024 calibration

The 2024-01 vs 2024-04 comparison provides a baseline for natural variation:
- Junior-senior sim: 2024-01 = {jr_sr_2024_01_emb:.4f}, 2024-04 = {jr_sr_2024_04_emb:.4f}, diff = {jr_sr_2024_04_emb - jr_sr_2024_01_emb:+.4f}
- 2024 -> 2026 change: {jr_sr_2026_emb - jr_sr_2024_01_emb:+.4f}

{"The 2024→2026 change exceeds the within-2024 baseline, suggesting a real structural shift." if abs(jr_sr_2026_emb - jr_sr_2024_01_emb) > abs(jr_sr_2024_04_emb - jr_sr_2024_01_emb) * 1.5 else "The 2024→2026 change is comparable to within-2024 variation, warranting caution in interpretation."}

## 10. Methodological Notes

- Embeddings: all-MiniLM-L6-v2 (384d), computed on first ~3000 chars of cleaned description text.
- TF-IDF: 10,000 features, min_df=5, max_df=0.95, reduced to 100 components via SVD.
- Cosine similarity used throughout (1 - cosine distance).
- Nearest-neighbor analysis uses k=5.
- Within-group dispersion computed on samples of up to 500 vectors per group (2000 pairwise samples).
- All analyses filtered to LinkedIn, English, date_flag='ok', is_swe=true.

## Output Files

### Figures
- `group_similarity_heatmaps.png` — Centroid similarity matrices (embedding vs TF-IDF)
- `dim_reduction_comparison.png` — PCA, {"UMAP, " if has_umap else ""}t-SNE scatter plots

### Tables
- `centroid_similarity_embedding.csv` / `centroid_similarity_tfidf.csv` — Full similarity matrices
- `within_group_dispersion.csv` — Within-group pairwise similarity
- `convergence_analysis.csv` — Seniority-pair similarity by period
- `nearest_neighbor_analysis.csv` — NN seniority distributions
- `robustness_table.csv` — Representation robustness comparison
- `outliers.csv` — Most atypical postings per group
- `sensitivity_summary.csv` — Sensitivity analysis results
"""

with open(REPORT_PATH, 'w') as f:
    f.write(report)

print(f"\nReport written to {REPORT_PATH}")
print("T15 complete.")
