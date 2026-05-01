"""T09 Step 3: Run BERTopic with UMAP + HDBSCAN on the 8k sample.

Experiments min_topic_size in {20, 30, 50} and reports topics, outlier %.
Saves topic assignments + top terms for the best-chosen configuration.

Custom vectorizer:
  - English stopwords
  - ngram_range (1, 2)
  - min_df=10, max_df=0.85 (prune doc-frequency extremes)
  - Additional SWE/JD-specific stopwords (we don't want topic labels to be
    dominated by "experience", "team", "work", etc.)

Outputs:
  exploration/artifacts/T09/bertopic_results.parquet  (uid, topic, prob, run_id)
  exploration/tables/T09/bertopic_config_sweep.csv    (per config: n_topics, outlier%, coherence)
  exploration/tables/T09/bertopic_topic_terms.csv     (best config: topic, top-20 terms)
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic

SEED = 20260417
OUTDIR = "exploration/artifacts/T09"
TABLES = "exploration/tables/T09"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

# Additional stopwords (SWE/JD boilerplate we do NOT want driving topic names)
# Keeping most tech tokens; excluding generic JD filler.
EXTRA_STOPWORDS = {
    # generic JD filler
    "experience", "work", "working", "team", "teams", "role", "roles",
    "opportunity", "opportunities", "join", "looking", "position",
    "responsibilities", "responsible", "requirements", "required", "require",
    "qualifications", "skills", "skill", "ability", "ability", "strong",
    "excellent", "good", "great", "proven", "demonstrated", "knowledge",
    "understanding", "effective", "proficient", "proficiency",
    "candidate", "candidates", "applicant", "applicants", "hire", "hiring",
    "company", "companies", "business", "businesses",
    "benefits", "benefit", "compensation", "salary",
    "years", "year", "months", "month",
    "day", "days", "daily", "per", "also", "will", "must", "may", "shall",
    "like", "etc", "including", "include", "includes",
    "ensure", "help", "helping", "across", "within", "well",
    "ability", "able", "support", "provide", "providing", "across",
    "solutions", "solution", "products", "product",
    "hands", "hand", "fast", "paced", "environment",
    "passionate", "passion", "exciting",
    "mission", "vision", "values", "culture",
    "equal", "employer", "opportunity", "disability",
    "communication", "collaborate", "collaboration",
    "understanding", "understand",
    "life", "balance", "paid", "time", "off", "vacation",
    "insurance", "medical", "dental",
    "new", "learn", "learning", "grow", "growth",
    "best", "top", "leading", "world", "class", "leader",
    "use", "using", "used", "make", "making",
    "many", "much", "one", "two", "three",
    "e.g.", "eg", "ie", "i.e.",
    "https", "http", "www", "com",
    "nan", "none",
    "please", "resume", "cover", "letter", "apply", "application",
    "monday", "tuesday", "wednesday", "thursday", "friday",
}


def build_vectorizer(stopword_list, min_df=10, max_df=0.85):
    return CountVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        stop_words=stopword_list,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\+\#\.\-/]{1,}\b",
    )


def run_one(docs, embeddings, uids, min_topic_size, random_state, stopword_list):
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0,
        metric="cosine", random_state=random_state, low_memory=True,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size, metric="euclidean",
        cluster_selection_method="eom", prediction_data=True,
    )
    # BERTopic applies the vectorizer only to c-TF-IDF concatenation (stage 2),
    # which sees one pseudo-document per topic. With few topics (e.g. mts=50
    # might produce <10), relative min_df or high absolute min_df fails.
    # Use min_df=1 (no doc-frequency cutoff), max_df=1.0 to be safe.
    # Rare-term noise is handled post hoc when we pick top-20 terms per topic
    # (BERTopic c-TF-IDF already ranks discriminative terms).
    vectorizer = build_vectorizer(stopword_list, min_df=1, max_df=1.0)
    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        calculate_probabilities=False,
        verbose=False,
        min_topic_size=min_topic_size,
    )
    topics, probs = model.fit_transform(docs, embeddings)
    return model, np.asarray(topics), probs, vectorizer


def topic_coherence_npmi(model, docs, vectorizer, top_n=10):
    """Approximate NPMI coherence: use BERTopic's own token frequencies from its
    c-TF-IDF vocabulary. Returns the mean pairwise NPMI over top_n words per
    topic (excluding -1). Robust and fast to compute."""
    from collections import Counter
    import itertools
    import math

    # Token presence per doc (not counts) to approximate PMI
    analyzer = vectorizer.build_analyzer()
    doc_tokens = [set(analyzer(d)) for d in docs]
    token_df = Counter()
    for toks in doc_tokens:
        for t in toks:
            token_df[t] += 1
    N = len(doc_tokens)

    def pair_df(t1, t2):
        n = 0
        for toks in doc_tokens:
            if t1 in toks and t2 in toks:
                n += 1
        return n

    topics = [t for t in model.get_topics().keys() if t != -1]
    coherences = []
    for tid in topics:
        words = [w for w, _ in model.get_topic(tid)[:top_n]]
        words = [w for w in words if token_df[w] > 0]
        if len(words) < 2:
            continue
        scores = []
        for w1, w2 in itertools.combinations(words, 2):
            p1 = token_df[w1] / N
            p2 = token_df[w2] / N
            p12 = pair_df(w1, w2) / N
            if p12 == 0 or p1 == 0 or p2 == 0:
                continue
            pmi = math.log(p12 / (p1 * p2))
            npmi = pmi / -math.log(p12)
            scores.append(npmi)
        if scores:
            coherences.append(sum(scores) / len(scores))
    return float(np.mean(coherences)) if coherences else float("nan")


def main():
    docs_df = pd.read_parquet(f"{OUTDIR}/sample_docs.parquet")
    embeddings = np.load(f"{OUTDIR}/sample_embeddings.npy")
    docs = docs_df["description_cleaned"].tolist()
    uids = docs_df["uid"].tolist()
    assert len(docs) == len(embeddings)
    print(f"Loaded {len(docs)} docs, embeddings {embeddings.shape}")

    # Build stopword list (shared across runs)
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stopword_list = list(ENGLISH_STOP_WORDS.union(EXTRA_STOPWORDS))

    # Config sweep
    sweep_rows = []
    results = {}
    for min_topic_size in [20, 30, 50]:
        print(f"\n>>> Running min_topic_size={min_topic_size}")
        model, topics, probs, vectorizer = run_one(
            docs, embeddings, uids, min_topic_size, SEED, stopword_list,
        )
        topic_info = model.get_topic_info()
        n_topics = (topic_info["Topic"] != -1).sum()
        outlier_frac = float((topics == -1).mean())
        # Approximate coherence on this model
        try:
            coh = topic_coherence_npmi(model, docs, vectorizer)
        except Exception as e:
            print(f"coherence failed: {e}")
            coh = float("nan")
        print(f"  topics={n_topics}, outlier_frac={outlier_frac:.3f}, coherence={coh:.4f}")
        sweep_rows.append({
            "min_topic_size": min_topic_size,
            "n_topics": int(n_topics),
            "outlier_frac": outlier_frac,
            "coherence_npmi_top10": coh,
        })
        results[min_topic_size] = {
            "model": model,
            "topics": topics,
            "probs": probs,
            "topic_info": topic_info,
        }

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(f"{TABLES}/bertopic_config_sweep.csv", index=False)
    print("\nConfig sweep:")
    print(sweep_df.to_string(index=False))

    # Pick the config with the best balance of interpretable topic count
    # and lowest outlier fraction.
    # Rule: if n_topics < 8 (under-segmenting: one giant topic holds >80% of
    # docs), the config is uninformative regardless of other scores.
    # Among remaining candidates, prefer lower outlier fraction, tiebreak by
    # higher coherence. Use mts=30 as our reference default.
    def score(row):
        if row.n_topics < 8:
            return -1e9  # disqualify
        # Favor balance: fewer outliers is better; more topics beyond 40 is worse
        ntopic_penalty = 0.0
        if row.n_topics > 40: ntopic_penalty = (row.n_topics - 40) * 0.01
        return -row.outlier_frac - ntopic_penalty + 0.1 * row.coherence_npmi_top10
    sweep_df["score"] = sweep_df.apply(score, axis=1)
    sweep_df = sweep_df.sort_values("score", ascending=False)
    best_mts = int(sweep_df.iloc[0]["min_topic_size"])
    print(f"\nPicked best min_topic_size = {best_mts}")

    best = results[best_mts]
    model = best["model"]
    topics = best["topics"]
    probs = best["probs"]

    # Also reduce outliers via c-TF-IDF reassignment so every doc gets a topic.
    # Keep the original topics for the "outlier %" story; store reduced topics
    # separately.
    try:
        topics_reduced = model.reduce_outliers(docs, topics, strategy="c-tf-idf")
        topics_reduced = np.asarray(topics_reduced)
        print(f"After reduce_outliers: outlier frac {(topics_reduced == -1).mean():.3f}")
    except Exception as e:
        print(f"reduce_outliers failed: {e}; using original topics")
        topics_reduced = topics.copy()

    # Save topic info + top terms
    topic_info = model.get_topic_info()
    topic_info.to_csv(f"{TABLES}/bertopic_topic_info.csv", index=False)

    rows = []
    for tid in topic_info["Topic"]:
        words = model.get_topic(tid)
        if not words:
            continue
        for rank, (w, s) in enumerate(words[:20]):
            rows.append({"topic": int(tid), "rank": rank, "term": w, "weight": float(s)})
    pd.DataFrame(rows).to_csv(f"{TABLES}/bertopic_topic_terms.csv", index=False)

    # Save topic assignments
    out = docs_df[["uid"]].copy()
    out["topic"] = topics
    out["topic_reduced"] = topics_reduced
    if probs is not None:
        # probs may be 2D (n_docs x n_topics) or 1D
        if probs.ndim == 1:
            out["prob"] = probs
    out["run_id"] = f"mts={best_mts}_seed={SEED}"
    out.to_parquet(f"{OUTDIR}/bertopic_assignments.parquet", index=False)

    # Save model
    model.save(
        f"{OUTDIR}/bertopic_model", serialization="safetensors",
        save_ctfidf=True, save_embedding_model=False,
    )

    # Also save config info
    with open(f"{OUTDIR}/bertopic_best_config.json", "w") as f:
        json.dump({
            "min_topic_size": best_mts,
            "n_topics": int((topics != -1).sum() > 0 and len(np.unique(topics[topics != -1]))),
            "outlier_frac": float((topics == -1).mean()),
            "seed": SEED,
        }, f, indent=2)
    print(f"Saved model + assignments. Outlier frac = {(topics == -1).mean():.3f}")


if __name__ == "__main__":
    main()
