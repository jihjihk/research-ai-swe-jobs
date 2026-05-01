"""T09 Step 1-5: Sample, run BERTopic + NMF, compare methods.

Outputs:
  - exploration/tables/T09/sample_composition.csv
  - exploration/tables/T09/methods_comparison.csv
  - exploration/tables/T09/bertopic_runs.csv  (min_topic_size x n_topics, noise %)
  - exploration/tables/T09/bertopic_stability.csv  (ARI across 3 seeds)
  - exploration/tables/T09/nmf_topics.csv  (k x top terms)
  - exploration/tables/T09/bertopic_topics.csv  (top-20 c-TF-IDF per topic)
  - exploration/tables/T09/method_overlap.csv  (topic alignment)
  - exploration/artifacts/T09_sample.parquet  (sample uids + metadata + cluster assignments)
  - exploration/artifacts/T09_bertopic_model.pkl  (primary model — dill)

This is a BIG script. Run once; downstream steps load the artifact.
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score

ROOT = Path("/home/jihgaboot/gabor/job-research")
ART = ROOT / "exploration/artifacts"
SHARED = ART / "shared"
TABLES = ROOT / "exploration/tables/T09"
TABLES.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)

RNG = 42
MAX_SAMPLE = 8000
PER_PERIOD_TARGET = 2700  # ~8,100 across 3 periods
COMPANY_CAP = 50

# Inline sanity asserts
assert (ROOT / "exploration/artifacts/shared/swe_cleaned_text.parquet").exists()
assert (ROOT / "exploration/artifacts/shared/swe_embeddings.npy").exists()
assert (ROOT / "exploration/artifacts/shared/swe_embedding_index.parquet").exists()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Step 1: Sample
# ---------------------------------------------------------------------------
log("Loading cleaned text + embedding index")
con = duckdb.connect()

# Period grouping: 2024 (prefer arshkon over asaniczka), 2026-03, 2026-04
# But spec says ~2,700 per period, 3 periods. Treat 2024 as one period, 2026-03 as one, 2026-04 as one.
# Prefer arshkon within 2024.
cleaned = con.execute(
    """
    SELECT uid, description_cleaned, text_source, source, period,
           seniority_final, seniority_3level, is_aggregator,
           company_name_canonical, metro_area, yoe_extracted,
           swe_classification_tier, seniority_final_source
    FROM 'exploration/artifacts/shared/swe_cleaned_text.parquet'
    WHERE text_source = 'llm'
      AND description_cleaned IS NOT NULL
      AND length(description_cleaned) > 50
    """
).df()

log(f"Eligible llm-text rows: {len(cleaned):,}")

# Assign period_group: '2024' for 2024-*, '2026-03', '2026-04'
cleaned["period_group"] = cleaned["period"].str.startswith("2024").map(
    {True: "2024", False: ""}
)
mask2026 = ~cleaned["period"].str.startswith("2024")
cleaned.loc[mask2026, "period_group"] = cleaned.loc[mask2026, "period"]

log(f"Period groups: {cleaned['period_group'].value_counts().to_dict()}")


def sample_period(df: pd.DataFrame, period_group: str, n_target: int) -> pd.DataFrame:
    sub = df[df["period_group"] == period_group].copy()
    if period_group == "2024":
        # Prefer arshkon
        ar = sub[sub["source"] == "kaggle_arshkon"]
        asa = sub[sub["source"] == "kaggle_asaniczka"]
        take_ar = min(len(ar), n_target)
        ar = ar.sample(n=take_ar, random_state=RNG) if take_ar < len(ar) else ar
        remain = n_target - len(ar)
        asa = asa.sample(n=min(remain, len(asa)), random_state=RNG) if remain > 0 else asa.iloc[:0]
        out = pd.concat([ar, asa], ignore_index=True)
    else:
        if len(sub) > n_target:
            # Stratify by seniority_3level
            strata = []
            levels = sub["seniority_3level"].fillna("unknown").unique()
            per_level = max(1, n_target // len(levels))
            for lv in levels:
                lvs = sub[sub["seniority_3level"].fillna("unknown") == lv]
                take = min(len(lvs), per_level)
                strata.append(lvs.sample(n=take, random_state=RNG))
            out = pd.concat(strata, ignore_index=True)
            # If under target (because some strata small), fill remaining
            deficit = n_target - len(out)
            if deficit > 0:
                rest = sub[~sub["uid"].isin(set(out["uid"]))]
                fill = rest.sample(n=min(deficit, len(rest)), random_state=RNG)
                out = pd.concat([out, fill], ignore_index=True)
        else:
            out = sub
    return out


samples = []
for pg, n in [("2024", PER_PERIOD_TARGET), ("2026-03", PER_PERIOD_TARGET), ("2026-04", PER_PERIOD_TARGET)]:
    samples.append(sample_period(cleaned, pg, n))
sample = pd.concat(samples, ignore_index=True)
log(f"Pre-cap sample: {len(sample):,}")

# Company capping: cap 50 per company_name_canonical
def cap_companies(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    return (
        df.groupby(df["company_name_canonical"].fillna("_unknown"), group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), cap), random_state=RNG))
        .reset_index(drop=True)
    )


sample = cap_companies(sample, COMPANY_CAP)
if len(sample) > MAX_SAMPLE:
    # Down-sample while preserving period balance
    out = []
    for pg, grp in sample.groupby("period_group"):
        n_keep = MAX_SAMPLE * len(grp) // len(sample)
        out.append(grp.sample(n=n_keep, random_state=RNG))
    sample = pd.concat(out, ignore_index=True)

log(f"Post-cap sample: {len(sample):,}")
log(f"Period distribution: {sample['period_group'].value_counts().to_dict()}")
log(f"Source distribution: {sample['source'].value_counts().to_dict()}")
log(f"Seniority 3level: {sample['seniority_3level'].value_counts().to_dict()}")
log(f"Aggregator share: {sample['is_aggregator'].mean():.3f}")

# Sample composition table
comp = sample.groupby(["source", "period", "seniority_3level", "text_source"]).size().reset_index(name="n")
comp.to_csv(TABLES / "sample_composition.csv", index=False)

# ---------------------------------------------------------------------------
# Step 2: Load shared embeddings aligned to sample
# ---------------------------------------------------------------------------
log("Loading shared embeddings")
emb_index = pd.read_parquet(SHARED / "swe_embedding_index.parquet")
embeddings_all = np.load(SHARED / "swe_embeddings.npy")
log(f"Shared embeddings shape: {embeddings_all.shape}")

uid_to_row = dict(zip(emb_index["uid"], emb_index["row_idx"]))
sample["emb_row"] = sample["uid"].map(uid_to_row)
missing = sample["emb_row"].isna().sum()
log(f"Sample rows without embedding: {missing}")
sample = sample.dropna(subset=["emb_row"]).reset_index(drop=True)
sample["emb_row"] = sample["emb_row"].astype(int)

X = embeddings_all[sample["emb_row"].values]
log(f"Sample embedding matrix: {X.shape}")

# ---------------------------------------------------------------------------
# Step 3: BERTopic (primary)
# ---------------------------------------------------------------------------
log("Running BERTopic runs")
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP

docs = sample["description_cleaned"].tolist()
bertopic_runs = []
models_by_mts = {}

# Load company stoplist for vectorizer stopwords
with open(SHARED / "company_stoplist.txt") as f:
    company_stops = set(line.strip() for line in f if line.strip())
log(f"Company stoplist: {len(company_stops):,}")

from nltk.corpus import stopwords
try:
    eng_stops = set(stopwords.words("english"))
except LookupError:
    import nltk
    nltk.download("stopwords", quiet=True)
    eng_stops = set(stopwords.words("english"))

# Build stopwords: english + most common company tokens only (to avoid 60k stopword list choking sklearn)
# Keep it manageable: cap company stops to ~5000 most frequent tokens — but the file lists all lowercased tokens
# Use all of them by filtering to <=20 char reasonable ones
company_stops_small = {w for w in company_stops if 2 <= len(w) <= 20 and w.isalpha()}
all_stops = list(eng_stops | company_stops_small)
log(f"Combined stopwords: {len(all_stops):,}")

def make_vectorizer():
    # min_df is small because BERTopic's c-TF-IDF fits this vectorizer on a CORPUS of
    # per-topic concatenated documents (one doc per topic), so the effective n is small.
    return CountVectorizer(
        stop_words=all_stops,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        token_pattern=r"[A-Za-z][A-Za-z0-9+#./_-]{1,}",
    )


for mts in [20, 30, 50]:
    log(f"  BERTopic min_topic_size={mts}")
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=RNG
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=mts, metric="euclidean", cluster_selection_method="eom", prediction_data=True
    )
    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=make_vectorizer(),
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = model.fit_transform(docs, embeddings=X)
    n_topics = len([t for t in set(topics) if t != -1])
    noise = (np.array(topics) == -1).mean()
    log(f"    n_topics={n_topics}, noise={noise:.3f}")
    bertopic_runs.append({"min_topic_size": mts, "n_topics": n_topics, "noise_share": noise})
    models_by_mts[mts] = (model, np.array(topics))

pd.DataFrame(bertopic_runs).to_csv(TABLES / "bertopic_runs.csv", index=False)

# Pick primary: min_topic_size=30 as per spec
primary_mts = 30
primary_model, primary_topics = models_by_mts[primary_mts]
_n_primary_topics = len([t for t in set(primary_topics) if t != -1])
log(f"Primary BERTopic model: min_topic_size={primary_mts}, n_topics={_n_primary_topics}, n_nonnoise_docs={(primary_topics != -1).sum()}")

# ---------------------------------------------------------------------------
# Step 3b: BERTopic stability — 3 seeds at mts=30
# ---------------------------------------------------------------------------
log("BERTopic stability (3 seeds)")
stab_labels = [primary_topics]
for seed in [101, 202]:
    log(f"  seed={seed}")
    um = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed)
    hd = HDBSCAN(min_cluster_size=primary_mts, metric="euclidean", cluster_selection_method="eom")
    m = BERTopic(umap_model=um, hdbscan_model=hd, vectorizer_model=make_vectorizer(), calculate_probabilities=False, verbose=False)
    tp, _ = m.fit_transform(docs, embeddings=X)
    stab_labels.append(np.array(tp))

pairs = [(0, 1), (0, 2), (1, 2)]
stab_rows = []
for i, j in pairs:
    mask = (stab_labels[i] != -1) & (stab_labels[j] != -1)
    ari = adjusted_rand_score(stab_labels[i][mask], stab_labels[j][mask])
    stab_rows.append({"seed_pair": f"{i}x{j}", "ari_nonnoise": ari, "n_nonnoise": int(mask.sum())})
    log(f"  ARI pair {i}x{j} (non-noise n={mask.sum()}): {ari:.3f}")
pd.DataFrame(stab_rows).to_csv(TABLES / "bertopic_stability.csv", index=False)

# ---------------------------------------------------------------------------
# Step 4: NMF
# ---------------------------------------------------------------------------
log("Running NMF")
tfidf_vec = TfidfVectorizer(
    stop_words=all_stops,
    min_df=5,
    max_df=0.5,
    ngram_range=(1, 2),
    token_pattern=r"[A-Za-z][A-Za-z0-9+#./_-]{1,}",
    max_features=20000,
)
tfidf = tfidf_vec.fit_transform(docs)
terms = np.array(tfidf_vec.get_feature_names_out())
log(f"TF-IDF shape: {tfidf.shape}")

nmf_results = {}
nmf_topic_rows = []
for k in [5, 8, 12, 15]:
    log(f"  NMF k={k}")
    nmf = NMF(n_components=k, random_state=RNG, init="nndsvd", max_iter=400)
    W = nmf.fit_transform(tfidf)
    H = nmf.components_
    assign = W.argmax(axis=1)
    nmf_results[k] = (W, H, assign)
    for t in range(k):
        top_idx = np.argsort(H[t])[::-1][:20]
        nmf_topic_rows.append(
            {"k": k, "component": t, "top_terms": " | ".join(terms[top_idx])}
        )
pd.DataFrame(nmf_topic_rows).to_csv(TABLES / "nmf_topics.csv", index=False)

# ---------------------------------------------------------------------------
# Step 5: Method comparison
# ---------------------------------------------------------------------------
log("Method comparison")

# BERTopic top-20 per topic (primary)
bt_topics = primary_model.get_topics()
bt_topic_rows = []
bt_top_terms = {}
for tid, trms in bt_topics.items():
    if tid == -1:
        continue
    top20 = [w for w, _ in trms[:20]]
    bt_top_terms[tid] = set(top20)
    bt_topic_rows.append({"topic_id": tid, "top_terms": " | ".join(top20), "n": int((primary_topics == tid).sum())})
pd.DataFrame(bt_topic_rows).sort_values("topic_id").to_csv(TABLES / "bertopic_topics.csv", index=False)

# Pick k=12 NMF for overlap alignment (comparable scale to BERTopic)
k_cmp = 12
_, H_cmp, nmf_assign_cmp = nmf_results[k_cmp]
nmf_top_terms = {}
for t in range(k_cmp):
    top_idx = np.argsort(H_cmp[t])[::-1][:20]
    nmf_top_terms[t] = set(terms[top_idx])

overlap_rows = []
for bt_id, bt_set in bt_top_terms.items():
    best = (None, 0.0)
    for nmf_id, nmf_set in nmf_top_terms.items():
        jac = len(bt_set & nmf_set) / max(1, len(bt_set | nmf_set))
        if jac > best[1]:
            best = (nmf_id, jac)
    overlap_rows.append({"bertopic_id": bt_id, "best_nmf_id": best[0], "jaccard": best[1]})
pd.DataFrame(overlap_rows).sort_values("jaccard", ascending=False).to_csv(
    TABLES / "method_overlap.csv", index=False
)

# Methods comparison table
n_bt_topics = len([t for t in set(primary_topics) if t != -1])
noise_bt = (primary_topics == -1).mean()
methods_rows = [
    {
        "method": "BERTopic (mts=20)",
        "n_topics": len([t for t in set(models_by_mts[20][1]) if t != -1]),
        "noise_share": (models_by_mts[20][1] == -1).mean(),
        "notes": "Finer topics, more noise",
    },
    {
        "method": "BERTopic (mts=30) [PRIMARY]",
        "n_topics": n_bt_topics,
        "noise_share": noise_bt,
        "notes": "Primary archetype model",
    },
    {
        "method": "BERTopic (mts=50)",
        "n_topics": len([t for t in set(models_by_mts[50][1]) if t != -1]),
        "noise_share": (models_by_mts[50][1] == -1).mean(),
        "notes": "Coarser topics, less noise",
    },
    {"method": "NMF k=5", "n_topics": 5, "noise_share": 0.0, "notes": "Broad soft-assigned components"},
    {"method": "NMF k=8", "n_topics": 8, "noise_share": 0.0, "notes": ""},
    {"method": "NMF k=12", "n_topics": 12, "noise_share": 0.0, "notes": "Used for method alignment"},
    {"method": "NMF k=15", "n_topics": 15, "noise_share": 0.0, "notes": ""},
]
pd.DataFrame(methods_rows).to_csv(TABLES / "methods_comparison.csv", index=False)

# ---------------------------------------------------------------------------
# Save sample with assignments
# ---------------------------------------------------------------------------
sample["bertopic_primary"] = primary_topics
sample["nmf12_assign"] = nmf_assign_cmp
sample.drop(columns=["emb_row"]).to_parquet(ART / "T09_sample.parquet", index=False)

# Save the UMAP-reduced 2D coords for plotting later (separate UMAP run)
from umap import UMAP as UMAP2

log("Computing 2D UMAP for plotting")
umap_2d = UMAP2(n_neighbors=15, n_components=2, min_dist=0.1, metric="cosine", random_state=RNG)
X_2d = umap_2d.fit_transform(X)
np.save(ART / "T09_umap2d.npy", X_2d)

# Also save PCA 2D
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=RNG)
X_pca = pca.fit_transform(X)
np.save(ART / "T09_pca2d.npy", X_pca)

# Save primary model embeddings for nearest-neighbor downstream assignment
np.save(ART / "T09_sample_embeddings.npy", X)

log("Step 1-5 complete.")
