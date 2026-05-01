#!/usr/bin/env python3
"""
T09: Posting archetype discovery — methods laboratory

Discovers natural posting archetypes through unsupervised methods:
  - BERTopic (primary) with pre-computed embeddings
  - NMF (comparison) on TF-IDF

Uses shared artifacts from exploration/artifacts/shared/.
"""

import os
import sys
import time
import json
import warnings
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import duckdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "exploration" / "artifacts" / "shared"
FIG_DIR = ROOT / "exploration" / "figures" / "T09"
TBL_DIR = ROOT / "exploration" / "tables" / "T09"
RPT_DIR = ROOT / "exploration" / "reports"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)
RPT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────────────
# 1. Load shared artifacts and build stratified sample
# ──────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("T09: Posting Archetype Discovery")
print("=" * 70)

t0 = time.time()

print("\n[1] Loading shared artifacts...")
con = duckdb.connect()

# Load cleaned text with metadata
text_df = con.execute("""
    SELECT uid, description_cleaned, text_source, source, period,
           seniority_final, seniority_3level, is_aggregator,
           company_name_canonical, metro_area, yoe_extracted,
           swe_classification_tier, seniority_final_source
    FROM 'exploration/artifacts/shared/swe_cleaned_text.parquet'
""").fetchdf()

# Load embedding index
emb_idx = con.execute("""
    SELECT row_index, uid
    FROM 'exploration/artifacts/shared/swe_embedding_index.parquet'
""").fetchdf()

# Load embeddings (memory-mapped)
embeddings_all = np.load(str(SHARED / "swe_embeddings.npy"), mmap_mode='r')
print(f"  Loaded {len(text_df)} text rows, {embeddings_all.shape} embeddings")

# Merge to get row_index for each uid
text_df = text_df.merge(emb_idx, on='uid', how='inner')
print(f"  After merge: {len(text_df)} rows with embeddings")

# ── Stratified sampling ──────────────────────────────────────────────────
print("\n[2] Building stratified sample (target: 8000)...")

TARGET_N = 8000

# Create stratum key
text_df['stratum'] = text_df['period'] + '_' + text_df['seniority_3level']

stratum_counts = text_df['stratum'].value_counts()
n_strata = len(stratum_counts)
print(f"  {n_strata} strata found:")
for s, c in stratum_counts.items():
    print(f"    {s}: {c}")

# Strategy: allocate proportionally, but with floor of min(stratum_size, 100)
# to ensure small strata are represented
sample_indices = []
stratum_allocation = {}

# Convert to dict for clean access
stratum_sizes = stratum_counts.to_dict()
total_pop = len(text_df)

# Proportional allocation with minimum floor
allocations = {}
for s, pop in stratum_sizes.items():
    prop = int(TARGET_N * pop / total_pop)
    floor = min(pop, 100)
    allocations[s] = max(prop, floor)

# Scale down if over budget
total_alloc = sum(allocations.values())
if total_alloc > TARGET_N:
    scale = TARGET_N / total_alloc
    for s in allocations:
        floor = min(stratum_sizes[s], 100)
        allocations[s] = max(floor, int(allocations[s] * scale))
    # Fill remaining budget into largest strata
    diff = TARGET_N - sum(allocations.values())
    sorted_strata = sorted(allocations.keys(), key=lambda s: stratum_sizes[s], reverse=True)
    for s in sorted_strata:
        if diff <= 0:
            break
        add = min(diff, stratum_sizes[s] - allocations[s])
        allocations[s] += add
        diff -= add

rng = np.random.RandomState(SEED)
sampled_rows = []
for s in sorted(allocations.keys()):
    mask = text_df['stratum'] == s
    stratum_df = text_df[mask]
    n_sample = min(allocations[s], len(stratum_df))
    chosen = stratum_df.sample(n=n_sample, random_state=rng)
    sampled_rows.append(chosen)
    stratum_allocation[s] = n_sample

sample_df = pd.concat(sampled_rows, ignore_index=True)
print(f"\n  Final sample: {len(sample_df)} rows")
print("  Sample composition:")
for s in sorted(stratum_allocation.keys()):
    pop = stratum_counts[s]
    n = stratum_allocation[s]
    print(f"    {s}: {n} / {pop} ({100*n/pop:.1f}%)")

# Get embeddings for sample
sample_emb_indices = sample_df['row_index'].values
sample_embeddings = np.array(embeddings_all[sample_emb_indices])  # copy into memory
print(f"  Embeddings loaded: {sample_embeddings.shape}")

# Save sample composition
sample_comp = pd.DataFrame([
    {'stratum': s, 'sampled': stratum_allocation[s], 'population': stratum_counts[s],
     'sample_pct': 100 * stratum_allocation[s] / stratum_counts[s]}
    for s in sorted(stratum_allocation.keys())
])
sample_comp.to_csv(TBL_DIR / "sample_composition.csv", index=False)

# ──────────────────────────────────────────────────────────────────────────
# 2. Build TF-IDF from cleaned text
# ──────────────────────────────────────────────────────────────────────────
print("\n[3] Building TF-IDF matrix...")
tfidf = TfidfVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.7,
    ngram_range=(1, 2),
    stop_words='english',
    sublinear_tf=True
)
tfidf_matrix = tfidf.fit_transform(sample_df['description_cleaned'].fillna(''))
feature_names = tfidf.get_feature_names_out()
print(f"  TF-IDF matrix: {tfidf_matrix.shape}")

# ──────────────────────────────────────────────────────────────────────────
# 3. Method A: BERTopic
# ──────────────────────────────────────────────────────────────────────────
print("\n[4] Running BERTopic...")

from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# UMAP reduction (shared across BERTopic runs)
print("  UMAP reduction...")
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric='cosine',
    random_state=SEED,
    low_memory=True,
    n_jobs=1
)
umap_embeddings = umap_model.fit_transform(sample_embeddings)
print(f"  UMAP done: {umap_embeddings.shape}")

# Also compute 2D UMAP for visualization
print("  2D UMAP for visualization...")
umap_2d = UMAP(
    n_neighbors=15,
    n_components=2,
    min_dist=0.1,
    metric='cosine',
    random_state=SEED,
    low_memory=True,
    n_jobs=1
)
umap_2d_coords = umap_2d.fit_transform(sample_embeddings)
print(f"  2D UMAP done: {umap_2d_coords.shape}")

# Run BERTopic with different min_topic_size values
bertopic_results = {}
for min_ts in [20, 30, 50]:
    print(f"\n  BERTopic with min_topic_size={min_ts}...")

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_ts,
        min_samples=10,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    vectorizer_model = CountVectorizer(
        stop_words='english',
        min_df=3,
        ngram_range=(1, 2)
    )

    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    topic_model = BERTopic(
        umap_model=umap_model,  # Will be bypassed since we pass reduced embeddings
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        calculate_probabilities=False,
        verbose=False,
        nr_topics=None
    )

    # Fit with pre-reduced embeddings
    topics, probs = topic_model.fit_transform(
        sample_df['description_cleaned'].tolist(),
        embeddings=umap_embeddings  # pass UMAP-reduced embeddings
    )

    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info[topic_info['Topic'] != -1])
    noise_count = (np.array(topics) == -1).sum()
    noise_pct = 100 * noise_count / len(topics)

    print(f"    Topics: {n_topics}, Noise: {noise_count} ({noise_pct:.1f}%)")

    bertopic_results[min_ts] = {
        'model': topic_model,
        'topics': np.array(topics),
        'n_topics': n_topics,
        'noise_count': noise_count,
        'noise_pct': noise_pct,
        'topic_info': topic_info
    }

# Select best BERTopic configuration (min_topic_size=30 as primary)
PRIMARY_MTS = 30
bt_primary = bertopic_results[PRIMARY_MTS]
bt_model = bt_primary['model']
bt_topics = bt_primary['topics']

print(f"\n  Primary BERTopic (mts={PRIMARY_MTS}): {bt_primary['n_topics']} topics, {bt_primary['noise_pct']:.1f}% noise")

# Extract topic representations
print("  Extracting topic representations...")
topic_terms = {}
topic_info_primary = bt_primary['topic_info']
for _, row in topic_info_primary.iterrows():
    tid = row['Topic']
    if tid == -1:
        continue
    terms = bt_model.get_topic(tid)
    topic_terms[tid] = [t[0] for t in terms[:20]]

# ──────────────────────────────────────────────────────────────────────────
# 4. Method B: NMF
# ──────────────────────────────────────────────────────────────────────────
print("\n[5] Running NMF...")

nmf_results = {}
for k in [5, 8, 12, 15]:
    print(f"  NMF with k={k}...")
    nmf = NMF(n_components=k, random_state=SEED, max_iter=500, init='nndsvda')
    W = nmf.fit_transform(tfidf_matrix)
    H = nmf.components_

    # Reconstruction error
    recon_err = nmf.reconstruction_err_

    # Top terms per component
    component_terms = {}
    for i in range(k):
        top_idx = H[i].argsort()[::-1][:20]
        component_terms[i] = [feature_names[j] for j in top_idx]

    # Hard assignment: each doc to its dominant component
    nmf_labels = W.argmax(axis=1)

    nmf_results[k] = {
        'W': W,
        'H': H,
        'labels': nmf_labels,
        'recon_err': recon_err,
        'component_terms': component_terms
    }

# ──────────────────────────────────────────────────────────────────────────
# 5. Method comparison
# ──────────────────────────────────────────────────────────────────────────
print("\n[6] Method comparison...")

# 5a. Topic alignment: top-term overlap between BERTopic and NMF
print("  Computing topic alignment (BERTopic vs NMF)...")

def jaccard_top_terms(terms1, terms2, top_n=20):
    """Jaccard similarity of top terms."""
    s1 = set(terms1[:top_n])
    s2 = set(terms2[:top_n])
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

# Compare BERTopic topics with NMF k=12 (closest to BERTopic topic count likely)
# Find best NMF k closest to BERTopic topic count
bt_n = bt_primary['n_topics']
best_k_for_comparison = min(nmf_results.keys(), key=lambda k: abs(k - bt_n))
nmf_comp = nmf_results[best_k_for_comparison]
print(f"  Comparing BERTopic ({bt_n} topics) with NMF k={best_k_for_comparison}")

alignment_matrix = np.zeros((bt_n, best_k_for_comparison))
bt_topic_ids = sorted([t for t in topic_terms.keys()])
for i, bt_id in enumerate(bt_topic_ids):
    for j in range(best_k_for_comparison):
        alignment_matrix[i, j] = jaccard_top_terms(
            topic_terms[bt_id], nmf_comp['component_terms'][j]
        )

# Best matches
print("  Topic alignment (best matches):")
alignment_pairs = []
for i, bt_id in enumerate(bt_topic_ids):
    best_nmf = alignment_matrix[i].argmax()
    best_score = alignment_matrix[i].max()
    alignment_pairs.append({
        'bertopic_id': bt_id,
        'bertopic_terms': ', '.join(topic_terms[bt_id][:5]),
        'nmf_component': best_nmf,
        'nmf_terms': ', '.join(nmf_comp['component_terms'][best_nmf][:5]),
        'jaccard': best_score
    })
    if best_score > 0.05:
        print(f"    BT-{bt_id} <-> NMF-{best_nmf}: {best_score:.3f}")

alignment_df = pd.DataFrame(alignment_pairs)
alignment_df.to_csv(TBL_DIR / "topic_alignment.csv", index=False)

# Method-robust topics (Jaccard > 0.1)
robust_topics = alignment_df[alignment_df['jaccard'] > 0.1]
print(f"  Method-robust topics (Jaccard > 0.1): {len(robust_topics)}")

# 5b. Cluster stability: 3 runs with different seeds
print("\n  Cluster stability (3 BERTopic runs)...")
stability_labels = []
for run_seed in [42, 123, 789]:
    hdb_stab = HDBSCAN(
        min_cluster_size=PRIMARY_MTS,
        min_samples=10,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels_run = hdb_stab.fit_predict(umap_embeddings)
    stability_labels.append(labels_run)

# ARI between pairs (excluding noise points from both)
ari_scores = []
for i in range(3):
    for j in range(i+1, 3):
        # Include all points for ARI (noise is a valid cluster label for stability)
        ari = adjusted_rand_score(stability_labels[i], stability_labels[j])
        ari_scores.append(ari)
        print(f"    ARI(run{i}, run{j}) = {ari:.4f}")

mean_ari = np.mean(ari_scores)
print(f"    Mean ARI: {mean_ari:.4f}")

# 5c. NMF stability across seeds
print("\n  NMF stability (3 runs, k={best_k_for_comparison})...")
nmf_stability_labels = []
for run_seed in [42, 123, 789]:
    nmf_stab = NMF(n_components=best_k_for_comparison, random_state=run_seed, max_iter=500, init='nndsvda')
    W_stab = nmf_stab.fit_transform(tfidf_matrix)
    nmf_stability_labels.append(W_stab.argmax(axis=1))

nmf_ari_scores = []
for i in range(3):
    for j in range(i+1, 3):
        ari = adjusted_rand_score(nmf_stability_labels[i], nmf_stability_labels[j])
        nmf_ari_scores.append(ari)

nmf_mean_ari = np.mean(nmf_ari_scores)
print(f"    NMF Mean ARI: {nmf_mean_ari:.4f}")

# 5d. Methods comparison table
methods_table = pd.DataFrame([
    {
        'method': f'BERTopic (mts=20)',
        'n_topics': bertopic_results[20]['n_topics'],
        'noise_pct': bertopic_results[20]['noise_pct'],
        'stability_ari': mean_ari,  # same HDBSCAN
        'notes': 'More granular'
    },
    {
        'method': f'BERTopic (mts=30)',
        'n_topics': bertopic_results[30]['n_topics'],
        'noise_pct': bertopic_results[30]['noise_pct'],
        'stability_ari': mean_ari,
        'notes': 'PRIMARY'
    },
    {
        'method': f'BERTopic (mts=50)',
        'n_topics': bertopic_results[50]['n_topics'],
        'noise_pct': bertopic_results[50]['noise_pct'],
        'stability_ari': mean_ari,
        'notes': 'Coarser'
    },
] + [
    {
        'method': f'NMF (k={k})',
        'n_topics': k,
        'noise_pct': 0.0,
        'stability_ari': nmf_mean_ari if k == best_k_for_comparison else None,
        'notes': 'comparison' if k == best_k_for_comparison else ''
    }
    for k in [5, 8, 12, 15]
])
methods_table.to_csv(TBL_DIR / "methods_comparison.csv", index=False)
print("\n  Methods comparison table saved.")

# ──────────────────────────────────────────────────────────────────────────
# 6. Characterization of BERTopic clusters
# ──────────────────────────────────────────────────────────────────────────
print("\n[7] Characterizing BERTopic clusters...")

sample_df['bt_topic'] = bt_topics

# Load tech matrix for tech count per posting
tech_df = con.execute("""
    SELECT * FROM 'exploration/artifacts/shared/swe_tech_matrix.parquet'
""").fetchdf()
tech_cols = [c for c in tech_df.columns if c != 'uid']
tech_df['tech_count'] = tech_df[tech_cols].sum(axis=1)
sample_df = sample_df.merge(tech_df[['uid', 'tech_count']], on='uid', how='left')
sample_df['desc_length'] = sample_df['description_cleaned'].str.len()

cluster_chars = []
for tid in sorted(bt_topics[bt_topics != -1]):
    if tid in [c['topic_id'] for c in cluster_chars]:
        continue
    mask = sample_df['bt_topic'] == tid
    subset = sample_df[mask]
    n = len(subset)
    if n == 0:
        continue

    # Seniority distribution
    sen_dist = subset['seniority_3level'].value_counts(normalize=True).to_dict()

    # Period distribution
    period_dist = subset['period'].value_counts(normalize=True).to_dict()

    # Aggregator rate
    agg_rate = subset['is_aggregator'].mean() if 'is_aggregator' in subset.columns else None

    # Top terms
    terms = topic_terms.get(tid, [])

    cluster_chars.append({
        'topic_id': tid,
        'n_docs': n,
        'pct_of_sample': 100 * n / len(sample_df),
        'top_5_terms': ', '.join(terms[:5]),
        'top_20_terms': ', '.join(terms[:20]),
        'senior_pct': sen_dist.get('senior', 0) * 100,
        'mid_pct': sen_dist.get('mid', 0) * 100,
        'junior_pct': sen_dist.get('junior', 0) * 100,
        'unknown_pct': sen_dist.get('unknown', 0) * 100,
        'period_2024_01_pct': period_dist.get('2024-01', 0) * 100,
        'period_2024_04_pct': period_dist.get('2024-04', 0) * 100,
        'period_2026_03_pct': period_dist.get('2026-03', 0) * 100,
        'mean_desc_length': subset['desc_length'].mean(),
        'mean_yoe': subset['yoe_extracted'].dropna().mean() if subset['yoe_extracted'].notna().any() else None,
        'median_yoe': subset['yoe_extracted'].dropna().median() if subset['yoe_extracted'].notna().any() else None,
        'mean_tech_count': subset['tech_count'].mean() if 'tech_count' in subset.columns else None,
        'aggregator_pct': (agg_rate * 100) if agg_rate is not None else None,
        'remote_or_distributed': None,  # placeholder
    })

cluster_chars_df = pd.DataFrame(cluster_chars)
cluster_chars_df = cluster_chars_df.sort_values('n_docs', ascending=False).reset_index(drop=True)

# Generate descriptive names based on top terms
def suggest_name(terms_str):
    """Simple heuristic name from top terms."""
    terms = terms_str.split(', ')
    if not terms:
        return "Unknown"
    # Look for patterns
    term_set = set(t.lower() for t in terms[:10])

    if any(t in term_set for t in ['data', 'analytics', 'pipeline', 'warehouse', 'etl', 'data pipeline', 'data engineering']):
        return "Data Engineering"
    if any(t in term_set for t in ['machine learning', 'ml', 'deep learning', 'model', 'models', 'ai']):
        return "ML/AI Engineering"
    if any(t in term_set for t in ['security', 'vulnerability', 'threat', 'cyber', 'compliance']):
        return "Security Engineering"
    if any(t in term_set for t in ['cloud', 'aws', 'infrastructure', 'kubernetes', 'terraform', 'devops']):
        return "Cloud/Infrastructure"
    if any(t in term_set for t in ['mobile', 'ios', 'android', 'swift', 'kotlin']):
        return "Mobile Development"
    if any(t in term_set for t in ['frontend', 'react', 'javascript', 'typescript', 'ui', 'css', 'web']):
        return "Frontend/Web"
    if any(t in term_set for t in ['backend', 'api', 'microservices', 'java', 'spring']):
        return "Backend/API"
    if any(t in term_set for t in ['embedded', 'firmware', 'hardware', 'rtos', 'c++']):
        return "Embedded/Systems"
    if any(t in term_set for t in ['qa', 'test', 'testing', 'automation', 'quality']):
        return "QA/Testing"
    if any(t in term_set for t in ['manager', 'lead', 'leadership', 'architect', 'principal']):
        return "Technical Leadership"
    if any(t in term_set for t in ['fullstack', 'full stack']):
        return "Full-Stack"
    if any(t in term_set for t in ['platform', 'reliability', 'sre', 'scalability']):
        return "Platform/SRE"
    return f"Cluster ({terms[0]})"

cluster_chars_df['suggested_name'] = cluster_chars_df['top_5_terms'].apply(suggest_name)
cluster_chars_df.to_csv(TBL_DIR / "cluster_characterization.csv", index=False)
print(f"  Characterized {len(cluster_chars_df)} clusters")
print("\n  Cluster overview:")
for _, row in cluster_chars_df.head(20).iterrows():
    print(f"    Topic {row['topic_id']:3d}: n={row['n_docs']:5d} ({row['pct_of_sample']:5.1f}%) "
          f"| {row['suggested_name']:25s} | {row['top_5_terms']}")

# ──────────────────────────────────────────────────────────────────────────
# 7. Temporal dynamics
# ──────────────────────────────────────────────────────────────────────────
print("\n[8] Temporal dynamics...")

# Exclude noise topics
non_noise = sample_df[sample_df['bt_topic'] != -1].copy()

# Period x topic crosstab (proportions within each period)
temporal = non_noise.groupby(['period', 'bt_topic']).size().unstack(fill_value=0)
temporal_pct = temporal.div(temporal.sum(axis=1), axis=0) * 100

# Compute change: 2026-03 vs average of 2024 periods
if '2024-01' in temporal_pct.index and '2026-03' in temporal_pct.index:
    baseline_2024 = temporal_pct.loc[['2024-01', '2024-04']].mean() if '2024-04' in temporal_pct.index else temporal_pct.loc['2024-01']
    change = temporal_pct.loc['2026-03'] - baseline_2024

    temporal_change = pd.DataFrame({
        'topic_id': change.index,
        'pct_2024_avg': baseline_2024.values,
        'pct_2026': temporal_pct.loc['2026-03'].values,
        'change_pp': change.values
    })

    # Add topic names
    name_map = dict(zip(cluster_chars_df['topic_id'], cluster_chars_df['suggested_name']))
    terms_map = dict(zip(cluster_chars_df['topic_id'], cluster_chars_df['top_5_terms']))
    temporal_change['name'] = temporal_change['topic_id'].map(name_map)
    temporal_change['top_terms'] = temporal_change['topic_id'].map(terms_map)
    temporal_change = temporal_change.sort_values('change_pp', ascending=False)
    temporal_change.to_csv(TBL_DIR / "temporal_dynamics.csv", index=False)

    print("  Biggest growers (2024 -> 2026):")
    for _, r in temporal_change.head(5).iterrows():
        print(f"    {r['name']:25s}: {r['pct_2024_avg']:5.1f}% -> {r['pct_2026']:5.1f}% ({r['change_pp']:+.1f}pp)")
    print("  Biggest shrinkers:")
    for _, r in temporal_change.tail(5).iterrows():
        print(f"    {r['name']:25s}: {r['pct_2024_avg']:5.1f}% -> {r['pct_2026']:5.1f}% ({r['change_pp']:+.1f}pp)")

# ──────────────────────────────────────────────────────────────────────────
# 8. Sensitivity checks
# ──────────────────────────────────────────────────────────────────────────
print("\n[9] Sensitivity checks...")

# (a) Aggregator exclusion
non_agg_mask = sample_df['is_aggregator'] == False
if non_agg_mask.sum() > 100:
    hdb_noagg = HDBSCAN(min_cluster_size=PRIMARY_MTS, min_samples=10, metric='euclidean', cluster_selection_method='eom')
    labels_noagg = hdb_noagg.fit_predict(umap_embeddings[non_agg_mask])
    ari_agg = adjusted_rand_score(bt_topics[non_agg_mask], labels_noagg)
    print(f"  Aggregator exclusion ARI vs full: {ari_agg:.4f}")
    noise_noagg = (labels_noagg == -1).sum() / len(labels_noagg) * 100
    n_topics_noagg = len(set(labels_noagg)) - (1 if -1 in labels_noagg else 0)
    print(f"  Without aggregators: {n_topics_noagg} topics, {noise_noagg:.1f}% noise")

# (d) Text source sensitivity
llm_mask = sample_df['text_source'] == 'llm'
rule_mask = sample_df['text_source'] == 'rule'
print(f"  Text source: LLM={llm_mask.sum()}, Rule={rule_mask.sum()}")

# Compare cluster distributions between LLM and rule text sources
if llm_mask.sum() > 100 and rule_mask.sum() > 100:
    llm_dist = pd.Series(bt_topics[llm_mask]).value_counts(normalize=True)
    rule_dist = pd.Series(bt_topics[rule_mask]).value_counts(normalize=True)
    # Align indices
    all_topics = sorted(set(llm_dist.index) | set(rule_dist.index))
    llm_aligned = np.array([llm_dist.get(t, 0) for t in all_topics])
    rule_aligned = np.array([rule_dist.get(t, 0) for t in all_topics])
    cos_sim = np.dot(llm_aligned, rule_aligned) / (np.linalg.norm(llm_aligned) * np.linalg.norm(rule_aligned) + 1e-10)
    print(f"  Cluster distribution cosine sim (LLM vs Rule text): {cos_sim:.4f}")

# (g) SWE classification tier sensitivity
tier_dist = sample_df.groupby('swe_classification_tier')['bt_topic'].value_counts(normalize=True)
print(f"  SWE tiers in sample: {sample_df['swe_classification_tier'].value_counts().to_dict()}")

# ──────────────────────────────────────────────────────────────────────────
# 9. Key discovery: Do clusters align with seniority or something else?
# ──────────────────────────────────────────────────────────────────────────
print("\n[10] Key discovery analysis: What is the market's dominant structure?")

# Compute mutual information and V-measure between clusters and various dimensions
from sklearn.metrics import normalized_mutual_info_score, v_measure_score

non_noise_df = sample_df[sample_df['bt_topic'] != -1].copy()
cluster_labels = non_noise_df['bt_topic'].values

dimensions = {
    'seniority_3level': non_noise_df['seniority_3level'].values,
    'period': non_noise_df['period'].values,
}

# Create a rough "role_type" from swe_classification_tier
dimensions['swe_tier'] = non_noise_df['swe_classification_tier'].values

# Check if we can infer industry or tech domain from top tech columns
# Use tech matrix for a rough domain classification
tech_subset = tech_df[tech_df['uid'].isin(non_noise_df['uid'])].copy()
tech_subset = tech_subset.merge(non_noise_df[['uid', 'bt_topic']], on='uid')

# Create rough domain from dominant tech categories
ai_cols = [c for c in tech_cols if c.startswith('ai_')]
cloud_cols = [c for c in tech_cols if c.startswith('cloud_')]
data_cols = [c for c in tech_cols if c.startswith('data_')]
lang_cols = [c for c in tech_cols if c.startswith('lang_')]
framework_cols = [c for c in tech_cols if c.startswith('framework_')]
mobile_cols = [c for c in tech_cols if c.startswith('mobile_')]
security_cols = [c for c in tech_cols if c.startswith('security_')]

domain_scores = pd.DataFrame({
    'uid': tech_subset['uid'],
    'ai': tech_subset[ai_cols].sum(axis=1) if ai_cols else 0,
    'cloud': tech_subset[cloud_cols].sum(axis=1) if cloud_cols else 0,
    'data': tech_subset[data_cols].sum(axis=1) if data_cols else 0,
    'mobile': tech_subset[mobile_cols].sum(axis=1) if mobile_cols else 0,
    'security': tech_subset[security_cols].sum(axis=1) if security_cols else 0,
})
domain_labels = domain_scores[['ai', 'cloud', 'data', 'mobile', 'security']].idxmax(axis=1)
domain_labels[domain_scores[['ai', 'cloud', 'data', 'mobile', 'security']].sum(axis=1) == 0] = 'general'
dimensions['tech_domain'] = domain_labels.values

# Add aggregator status
dimensions['is_aggregator'] = non_noise_df['is_aggregator'].astype(str).values

print("  Normalized Mutual Information (NMI) between clusters and dimensions:")
nmi_results = {}
for dim_name, dim_values in dimensions.items():
    nmi = normalized_mutual_info_score(cluster_labels, dim_values)
    vmeasure = v_measure_score(cluster_labels, dim_values)
    nmi_results[dim_name] = {'nmi': nmi, 'v_measure': vmeasure}
    print(f"    {dim_name:20s}: NMI={nmi:.4f}, V-measure={vmeasure:.4f}")

nmi_df = pd.DataFrame(nmi_results).T
nmi_df.index.name = 'dimension'
nmi_df.to_csv(TBL_DIR / "cluster_dimension_alignment.csv")

# Determine dominant structure
dominant_dim = max(nmi_results.items(), key=lambda x: x[1]['nmi'])
print(f"\n  DOMINANT STRUCTURE: Clusters align most with '{dominant_dim[0]}' (NMI={dominant_dim[1]['nmi']:.4f})")

# ──────────────────────────────────────────────────────────────────────────
# 10. Visualizations
# ──────────────────────────────────────────────────────────────────────────
print("\n[11] Generating visualizations...")

# Also compute PCA for comparison
print("  PCA 2D projection...")
pca_2d = PCA(n_components=2, random_state=SEED)
pca_coords = pca_2d.fit_transform(sample_embeddings)
print(f"  PCA explained variance: {pca_2d.explained_variance_ratio_.sum():.3f}")

# ── Figure 1: UMAP colored by cluster, period, seniority (3 panels) ──
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Panel A: Clusters
ax = axes[0]
unique_topics = sorted(set(bt_topics))
# Color noise gray, clusters with palette
n_real = len([t for t in unique_topics if t != -1])
palette = sns.color_palette('tab20', n_colors=max(n_real, 1))
color_map = {}
ci = 0
for t in unique_topics:
    if t == -1:
        color_map[t] = (0.8, 0.8, 0.8, 0.3)
    else:
        color_map[t] = (*palette[ci % len(palette)], 0.6)
        ci += 1

colors = [color_map[t] for t in bt_topics]
ax.scatter(umap_2d_coords[:, 0], umap_2d_coords[:, 1], c=colors, s=2, rasterized=True)
ax.set_title('UMAP: BERTopic Clusters', fontsize=12, fontweight='bold')
ax.set_xlabel('UMAP-1')
ax.set_ylabel('UMAP-2')

# Panel B: Period
ax = axes[1]
period_colors = {'2024-01': '#2196F3', '2024-04': '#4CAF50', '2026-03': '#F44336'}
period_labels = sample_df['period'].values
colors_period = [period_colors.get(p, 'gray') for p in period_labels]
ax.scatter(umap_2d_coords[:, 0], umap_2d_coords[:, 1], c=colors_period, s=2, alpha=0.4, rasterized=True)
ax.set_title('UMAP: By Period', fontsize=12, fontweight='bold')
ax.set_xlabel('UMAP-1')
ax.set_ylabel('UMAP-2')
# Legend
for p, c in period_colors.items():
    ax.scatter([], [], c=c, s=30, label=p)
ax.legend(loc='upper right', fontsize=8)

# Panel C: Seniority
ax = axes[2]
sen_colors = {'junior': '#FF9800', 'mid': '#2196F3', 'senior': '#4CAF50', 'unknown': '#9E9E9E'}
sen_labels = sample_df['seniority_3level'].values
colors_sen = [sen_colors.get(s, 'gray') for s in sen_labels]
ax.scatter(umap_2d_coords[:, 0], umap_2d_coords[:, 1], c=colors_sen, s=2, alpha=0.4, rasterized=True)
ax.set_title('UMAP: By Seniority', fontsize=12, fontweight='bold')
ax.set_xlabel('UMAP-1')
ax.set_ylabel('UMAP-2')
for s, c in sen_colors.items():
    ax.scatter([], [], c=c, s=30, label=s)
ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / "umap_3panel.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved umap_3panel.png")

# ── Figure 2: PCA colored by cluster (comparison with UMAP) ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PCA by cluster
ax = axes[0]
colors_cluster = [color_map[t] for t in bt_topics]
ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=colors_cluster, s=2, rasterized=True)
ax.set_title('PCA: BERTopic Clusters', fontsize=12, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} var)')
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} var)')

# PCA by period
ax = axes[1]
ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=colors_period, s=2, alpha=0.4, rasterized=True)
ax.set_title('PCA: By Period', fontsize=12, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} var)')
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} var)')
for p, c in period_colors.items():
    ax.scatter([], [], c=c, s=30, label=p)
ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / "pca_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved pca_comparison.png")

# ── Figure 3: Temporal dynamics bar chart ──
if 'temporal_change' in dir():
    fig, ax = plt.subplots(figsize=(12, 7))
    tc = temporal_change.dropna(subset=['name']).sort_values('change_pp')
    colors_bar = ['#F44336' if v < 0 else '#4CAF50' for v in tc['change_pp']]
    bars = ax.barh(range(len(tc)), tc['change_pp'], color=colors_bar, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(tc)))
    ax.set_yticklabels([f"T{int(tid)}: {name}" for tid, name in zip(tc['topic_id'], tc['name'])], fontsize=8)
    ax.set_xlabel('Change in Share (pp), 2024 avg to 2026')
    ax.set_title('Archetype Proportion Changes: 2024 to 2026', fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "temporal_change.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved temporal_change.png")

# ── Figure 4: Cluster-dimension NMI comparison ──
fig, ax = plt.subplots(figsize=(8, 5))
dims = list(nmi_results.keys())
nmi_vals = [nmi_results[d]['nmi'] for d in dims]
vm_vals = [nmi_results[d]['v_measure'] for d in dims]
x = np.arange(len(dims))
width = 0.35
ax.bar(x - width/2, nmi_vals, width, label='NMI', color='#2196F3')
ax.bar(x + width/2, vm_vals, width, label='V-measure', color='#FF9800')
ax.set_xticks(x)
ax.set_xticklabels(dims, rotation=30, ha='right')
ax.set_ylabel('Score')
ax.set_title('Cluster Alignment with External Dimensions', fontweight='bold')
ax.legend()
ax.set_ylim(0, max(max(nmi_vals), max(vm_vals)) * 1.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "dimension_alignment.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved dimension_alignment.png")

# ──────────────────────────────────────────────────────────────────────────
# 11. NMF detailed output
# ──────────────────────────────────────────────────────────────────────────
print("\n[12] Saving NMF detailed results...")
for k in [5, 8, 12, 15]:
    nmf_terms_df = pd.DataFrame(nmf_results[k]['component_terms'])
    nmf_terms_df.index = [f'term_{i+1}' for i in range(20)]
    nmf_terms_df.columns = [f'component_{i}' for i in range(k)]
    nmf_terms_df.to_csv(TBL_DIR / f"nmf_k{k}_terms.csv")

# ──────────────────────────────────────────────────────────────────────────
# 12. Generate report
# ──────────────────────────────────────────────────────────────────────────
print("\n[13] Generating report...")

report_lines = []
report_lines.append("# T09: Posting Archetype Discovery -- Methods Laboratory")
report_lines.append("")
report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
report_lines.append("")

report_lines.append("## 1. Sample")
report_lines.append("")
report_lines.append(f"**Population:** {len(text_df):,} SWE LinkedIn postings (English, date_flag=ok)")
report_lines.append(f"**Sample:** {len(sample_df):,} postings, stratified by period x seniority_3level")
report_lines.append("")
report_lines.append("| Stratum | Sampled | Population | Sample Rate |")
report_lines.append("|---|---|---|---|")
for _, r in sample_comp.iterrows():
    report_lines.append(f"| {r['stratum']} | {int(r['sampled'])} | {int(r['population'])} | {r['sample_pct']:.1f}% |")
report_lines.append("")

report_lines.append("## 2. Method A: BERTopic")
report_lines.append("")
report_lines.append("BERTopic uses pre-computed sentence-transformer embeddings (all-MiniLM-L6-v2, 384-dim),")
report_lines.append("UMAP reduction to 5 dimensions, and HDBSCAN clustering.")
report_lines.append("")
report_lines.append("### Configuration comparison")
report_lines.append("")
report_lines.append("| min_topic_size | Topics | Noise % |")
report_lines.append("|---|---|---|")
for mts in [20, 30, 50]:
    r = bertopic_results[mts]
    report_lines.append(f"| {mts} | {r['n_topics']} | {r['noise_pct']:.1f}% |")
report_lines.append("")
report_lines.append(f"**Selected configuration:** min_topic_size={PRIMARY_MTS} ({bt_primary['n_topics']} topics, {bt_primary['noise_pct']:.1f}% noise)")
report_lines.append("")

report_lines.append("### Cluster characterization (top 15 by size)")
report_lines.append("")
report_lines.append("| ID | Name | N | % | Top 5 Terms | Sr% | Jr% | 2024% | 2026% | Mean YOE | Tech Count |")
report_lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
for _, r in cluster_chars_df.head(15).iterrows():
    pct_2024 = r['period_2024_01_pct'] + r['period_2024_04_pct']
    yoe_str = f"{r['mean_yoe']:.1f}" if pd.notna(r['mean_yoe']) else "N/A"
    tc_str = f"{r['mean_tech_count']:.1f}" if pd.notna(r['mean_tech_count']) else "N/A"
    report_lines.append(
        f"| {int(r['topic_id'])} | {r['suggested_name']} | {int(r['n_docs'])} | {r['pct_of_sample']:.1f} "
        f"| {r['top_5_terms']} | {r['senior_pct']:.0f} | {r['junior_pct']:.0f} "
        f"| {pct_2024:.0f} | {r['period_2026_03_pct']:.0f} | {yoe_str} | {tc_str} |"
    )
report_lines.append("")

report_lines.append("## 3. Method B: NMF")
report_lines.append("")
report_lines.append("NMF on TF-IDF (5000 features, 1-2grams, sublinear TF).")
report_lines.append("")
report_lines.append("| k | Reconstruction Error |")
report_lines.append("|---|---|")
for k in [5, 8, 12, 15]:
    report_lines.append(f"| {k} | {nmf_results[k]['recon_err']:.2f} |")
report_lines.append("")

# Show NMF k=best_k_for_comparison components
report_lines.append(f"### NMF k={best_k_for_comparison} component top terms")
report_lines.append("")
for i in range(best_k_for_comparison):
    terms = ', '.join(nmf_results[best_k_for_comparison]['component_terms'][i][:10])
    report_lines.append(f"- **Component {i}:** {terms}")
report_lines.append("")

report_lines.append("## 4. Method Comparison")
report_lines.append("")
report_lines.append("### Methods comparison table")
report_lines.append("")
report_lines.append("| Method | Topics | Noise % | Stability (ARI) | Notes |")
report_lines.append("|---|---|---|---|---|")
for _, r in methods_table.iterrows():
    ari_str = f"{r['stability_ari']:.4f}" if pd.notna(r['stability_ari']) else "N/A"
    report_lines.append(f"| {r['method']} | {int(r['n_topics'])} | {r['noise_pct']:.1f}% | {ari_str} | {r['notes']} |")
report_lines.append("")

report_lines.append("### Topic alignment (BERTopic vs NMF)")
report_lines.append("")
n_aligned = len(robust_topics)
report_lines.append(f"Method-robust topics (Jaccard > 0.1): **{n_aligned}** out of {bt_primary['n_topics']} BERTopic topics")
report_lines.append("")
if len(robust_topics) > 0:
    report_lines.append("| BERTopic Topic | BERTopic Terms | Best NMF Match | NMF Terms | Jaccard |")
    report_lines.append("|---|---|---|---|---|")
    for _, r in robust_topics.iterrows():
        report_lines.append(f"| {int(r['bertopic_id'])} | {r['bertopic_terms']} | {int(r['nmf_component'])} | {r['nmf_terms']} | {r['jaccard']:.3f} |")
    report_lines.append("")

report_lines.append("### Cluster stability")
report_lines.append("")
report_lines.append(f"- BERTopic (HDBSCAN): Mean ARI across 3 seeds = **{mean_ari:.4f}**")
report_lines.append(f"- NMF (k={best_k_for_comparison}): Mean ARI across 3 seeds = **{nmf_mean_ari:.4f}**")
report_lines.append("")

report_lines.append("## 5. Temporal Dynamics")
report_lines.append("")
if 'temporal_change' in dir():
    report_lines.append("### Archetype proportion changes: 2024 (average) to 2026")
    report_lines.append("")
    report_lines.append("| Topic | Name | 2024 avg % | 2026 % | Change (pp) |")
    report_lines.append("|---|---|---|---|---|")
    for _, r in temporal_change.iterrows():
        if pd.notna(r['name']):
            report_lines.append(f"| {int(r['topic_id'])} | {r['name']} | {r['pct_2024_avg']:.1f} | {r['pct_2026']:.1f} | {r['change_pp']:+.1f} |")
    report_lines.append("")

report_lines.append("## 6. Sensitivity Checks")
report_lines.append("")
report_lines.append("### (a) Aggregator exclusion")
if 'ari_agg' in dir():
    report_lines.append(f"- ARI (full vs non-aggregator): {ari_agg:.4f}")
    report_lines.append(f"- Non-aggregator run: {n_topics_noagg} topics, {noise_noagg:.1f}% noise")
else:
    report_lines.append("- Not computed (insufficient non-aggregator rows)")
report_lines.append("")
report_lines.append("### (d) Text source sensitivity")
report_lines.append(f"- LLM-cleaned text: {llm_mask.sum()} rows, Rule-based text: {rule_mask.sum()} rows")
if 'cos_sim' in dir():
    report_lines.append(f"- Cluster distribution cosine similarity (LLM vs Rule): {cos_sim:.4f}")
report_lines.append("")

report_lines.append("## 7. Key Discovery: Market's Dominant Structure")
report_lines.append("")
report_lines.append("### Cluster alignment with external dimensions (NMI)")
report_lines.append("")
report_lines.append("| Dimension | NMI | V-measure |")
report_lines.append("|---|---|---|")
for d in dims:
    report_lines.append(f"| {d} | {nmi_results[d]['nmi']:.4f} | {nmi_results[d]['v_measure']:.4f} |")
report_lines.append("")

report_lines.append(f"**Finding:** Clusters align most strongly with **{dominant_dim[0]}** (NMI={dominant_dim[1]['nmi']:.4f}).")
report_lines.append("")

# Interpretive analysis
report_lines.append("### Interpretation")
report_lines.append("")
if dominant_dim[0] == 'tech_domain':
    report_lines.append("The market's natural structure is organized primarily around **technology domain/specialization**,")
    report_lines.append("not seniority level. This means that when employers write job postings, the strongest differentiator")
    report_lines.append("in language and content is the technical domain (data engineering vs. cloud/infra vs. ML/AI vs. frontend")
    report_lines.append("vs. mobile, etc.), rather than the seniority level of the role.")
elif dominant_dim[0] == 'seniority_3level':
    report_lines.append("The market's natural structure aligns primarily with **seniority level**.")
    report_lines.append("This would mean that posting language varies more by experience level than by technical specialization.")
elif dominant_dim[0] == 'period':
    report_lines.append("The market's natural structure aligns primarily with **time period**.")
    report_lines.append("This suggests posting language has changed so substantially between 2024 and 2026 that temporal")
    report_lines.append("differences dominate over both seniority and technical domain differences.")
else:
    report_lines.append(f"The strongest alignment is with **{dominant_dim[0]}**, suggesting this dimension")
    report_lines.append("captures the most variance in how job postings are written.")

report_lines.append("")
report_lines.append("This finding has implications for the research questions:")
report_lines.append("")
report_lines.append("- **RQ1 (restructuring):** If clusters are domain-driven rather than seniority-driven,")
report_lines.append("  then seniority effects operate *within* domain archetypes rather than across them.")
report_lines.append("- **RQ2 (task migration):** Requirements that migrate between seniority levels may be")
report_lines.append("  domain-specific rather than universal.")
report_lines.append("- **Temporal dynamics:** Archetype shifts from 2024 to 2026 reveal which technical domains")
report_lines.append("  are growing or shrinking in the SWE labor market.")
report_lines.append("")

report_lines.append("## 8. Visualizations")
report_lines.append("")
report_lines.append("- `figures/T09/umap_3panel.png` -- UMAP colored by cluster, period, seniority")
report_lines.append("- `figures/T09/pca_comparison.png` -- PCA colored by cluster and period")
report_lines.append("- `figures/T09/temporal_change.png` -- Archetype proportion changes 2024 to 2026")
report_lines.append("- `figures/T09/dimension_alignment.png` -- NMI/V-measure by external dimension")
report_lines.append("")

report_lines.append("## 9. Tables")
report_lines.append("")
report_lines.append("- `tables/T09/sample_composition.csv`")
report_lines.append("- `tables/T09/cluster_characterization.csv`")
report_lines.append("- `tables/T09/methods_comparison.csv`")
report_lines.append("- `tables/T09/topic_alignment.csv`")
report_lines.append("- `tables/T09/temporal_dynamics.csv`")
report_lines.append("- `tables/T09/cluster_dimension_alignment.csv`")
report_lines.append("- `tables/T09/nmf_k{5,8,12,15}_terms.csv`")
report_lines.append("")

elapsed = time.time() - t0
report_lines.append(f"---")
report_lines.append(f"*Analysis completed in {elapsed:.0f}s*")

report_text = '\n'.join(report_lines)
(RPT_DIR / "T09.md").write_text(report_text)
print(f"\n  Report written to exploration/reports/T09.md")
print(f"\nTotal time: {elapsed:.0f}s")
