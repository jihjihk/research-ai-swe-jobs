#!/usr/bin/env python3
"""
T12 Step 8: BERTopic cross-validation.
Fit BERTopic on combined corpus with period as class variable.
Identify most period-specific topics and compare with Fightin' Words.
"""

import sys, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

warnings.filterwarnings('ignore')

BASE = Path('/home/jihgaboot/gabor/job-research')
FIG_DIR = BASE / 'exploration/figures/T12'
TAB_DIR = BASE / 'exploration/tables/T12'
SHARED = BASE / 'exploration/artifacts/shared'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 150})

# ──────────────────────────────────────────────────────────────────────
# Load data and pre-computed embeddings
# ──────────────────────────────────────────────────────────────────────
print("Loading data...")
con = duckdb.connect()
df = con.execute(f"""
    SELECT uid, description_cleaned, source, period, seniority_3level
    FROM '{SHARED}/swe_cleaned_text.parquet'
    WHERE length(description_cleaned) > 50
""").fetchdf()

# Load embeddings + index
emb_index = con.execute(f"SELECT * FROM '{SHARED}/swe_embedding_index.parquet'").fetchdf()
embeddings = np.load(SHARED / 'swe_embeddings.npy')

print(f"Data: {len(df)} rows, Embeddings: {embeddings.shape}")

# Align: merge df with embedding index
df_merged = df.merge(emb_index, on='uid', how='inner')
df_merged = df_merged.sort_values('row_index').reset_index(drop=True)
emb_aligned = embeddings[df_merged['row_index'].values]

print(f"Aligned: {len(df_merged)} rows with embeddings")

# Create period labels
# Focus on arshkon vs scraped for primary comparison
mask_primary = df_merged['source'].isin(['kaggle_arshkon', 'scraped'])
df_bt = df_merged[mask_primary].reset_index(drop=True)
emb_bt = emb_aligned[mask_primary.values]

period_labels = df_bt['source'].map({
    'kaggle_arshkon': '2024',
    'scraped': '2026'
}).values

print(f"BERTopic input: {len(df_bt)} docs ({(period_labels=='2024').sum()} 2024, {(period_labels=='2026').sum()} 2026)")

# ──────────────────────────────────────────────────────────────────────
# Sample if too large (BERTopic UMAP is memory-intensive)
# ──────────────────────────────────────────────────────────────────────
MAX_DOCS = 20000
if len(df_bt) > MAX_DOCS:
    print(f"Sampling to {MAX_DOCS} docs (stratified by period)...")
    rng = np.random.RandomState(42)
    idx_2024 = np.where(period_labels == '2024')[0]
    idx_2026 = np.where(period_labels == '2026')[0]
    n_2024 = min(len(idx_2024), MAX_DOCS // 2)
    n_2026 = MAX_DOCS - n_2024
    if n_2026 > len(idx_2026):
        n_2026 = len(idx_2026)
        n_2024 = MAX_DOCS - n_2026
    sample_idx = np.concatenate([
        rng.choice(idx_2024, n_2024, replace=False),
        rng.choice(idx_2026, n_2026, replace=False)
    ])
    sample_idx.sort()
    df_bt = df_bt.iloc[sample_idx].reset_index(drop=True)
    emb_bt = emb_bt[sample_idx]
    period_labels = period_labels[sample_idx]
    print(f"  Sampled: {(period_labels=='2024').sum()} 2024, {(period_labels=='2026').sum()} 2026")

# ──────────────────────────────────────────────────────────────────────
# BERTopic
# ──────────────────────────────────────────────────────────────────────
print("\nFitting BERTopic...")
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

umap_model = UMAP(n_neighbors=15, n_components=10, min_dist=0.0,
                   metric='cosine', random_state=42, low_memory=True)
hdbscan_model = HDBSCAN(min_cluster_size=50, min_samples=10,
                          metric='euclidean', prediction_data=True)
vectorizer = CountVectorizer(min_df=10, max_df=0.95, ngram_range=(1, 2),
                              token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z+#.]{1,}\b')

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    nr_topics='auto',
    top_n_words=15,
    verbose=True
)

docs = df_bt['description_cleaned'].tolist()
topics, probs = topic_model.fit_transform(docs, embeddings=emb_bt)

topic_info = topic_model.get_topic_info()
print(f"\nFound {len(topic_info) - 1} topics (excluding outlier topic -1)")
print(f"Outlier docs: {(np.array(topics) == -1).sum()}")

# ──────────────────────────────────────────────────────────────────────
# Topics per class (period)
# ──────────────────────────────────────────────────────────────────────
print("\nComputing topics per class...")
topics_per_class = topic_model.topics_per_class(docs, classes=period_labels)

# Compute period specificity: for each topic, what fraction is from each period?
df_bt['topic'] = topics
df_bt['period_label'] = period_labels

topic_period = df_bt.groupby(['topic', 'period_label']).size().unstack(fill_value=0)
if '2024' not in topic_period.columns:
    topic_period['2024'] = 0
if '2026' not in topic_period.columns:
    topic_period['2026'] = 0

topic_period['total'] = topic_period['2024'] + topic_period['2026']
topic_period['pct_2026'] = topic_period['2026'] / topic_period['total']
topic_period['pct_2024'] = topic_period['2024'] / topic_period['total']

# Base rates
base_2026 = (period_labels == '2026').sum() / len(period_labels)
base_2024 = 1 - base_2026

topic_period['lift_2026'] = topic_period['pct_2026'] / base_2026
topic_period['lift_2024'] = topic_period['pct_2024'] / base_2024

# Filter out outlier topic and tiny topics
topic_period_filtered = topic_period[(topic_period.index != -1) & (topic_period['total'] >= 30)]
topic_period_filtered = topic_period_filtered.sort_values('lift_2026', ascending=False)

# Get topic labels
topic_labels = {}
for topic_id in topic_period_filtered.index:
    words = topic_model.get_topic(topic_id)
    if words:
        topic_labels[topic_id] = ' | '.join([w for w, _ in words[:5]])
    else:
        topic_labels[topic_id] = f'Topic {topic_id}'

topic_period_filtered['label'] = topic_period_filtered.index.map(topic_labels)

print("\n--- Most 2026-specific topics (highest lift) ---")
for idx, row in topic_period_filtered.head(15).iterrows():
    print(f"  Topic {idx:3d} (n={row['total']:4.0f}, {row['pct_2026']:.0%} 2026, lift={row['lift_2026']:.2f}): {row['label']}")

print("\n--- Most 2024-specific topics (highest 2024 lift) ---")
most_2024 = topic_period_filtered.sort_values('lift_2024', ascending=False).head(15)
for idx, row in most_2024.iterrows():
    print(f"  Topic {idx:3d} (n={row['total']:4.0f}, {row['pct_2024']:.0%} 2024, lift={row['lift_2024']:.2f}): {row['label']}")

# Save topic results
topic_period_filtered.to_csv(TAB_DIR / 'bertopic_period_specificity.csv')
topic_info.to_csv(TAB_DIR / 'bertopic_topic_info.csv', index=False)
topics_per_class.to_csv(TAB_DIR / 'bertopic_topics_per_class.csv', index=False)

# ──────────────────────────────────────────────────────────────────────
# Cross-validation with Fightin' Words
# ──────────────────────────────────────────────────────────────────────
print("\n--- Cross-validation with Fightin' Words ---")
# Load FW primary results
fw = pd.read_csv(TAB_DIR / 'fw_primary_unigrams.csv')
fw_top_2026 = set(fw.head(50)['term'].values)
fw_top_2024 = set(fw.tail(50)['term'].values)

# Check if top FW terms appear in top BERTopic topics
for topic_id in topic_period_filtered.head(10).index:
    words = topic_model.get_topic(topic_id)
    if words:
        topic_terms = set([w for w, _ in words])
        fw_overlap_2026 = topic_terms & fw_top_2026
        fw_overlap_2024 = topic_terms & fw_top_2024
        lift = topic_period_filtered.loc[topic_id, 'lift_2026']
        pct_2026 = topic_period_filtered.loc[topic_id, 'pct_2026']
        if fw_overlap_2026 or fw_overlap_2024:
            print(f"  Topic {topic_id} (lift_2026={lift:.2f}): FW-2026 overlap: {fw_overlap_2026}, FW-2024 overlap: {fw_overlap_2024}")

print("\nDone. BERTopic results saved to tables/T12/.")
