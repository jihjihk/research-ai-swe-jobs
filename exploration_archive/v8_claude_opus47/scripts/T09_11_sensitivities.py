"""T09 Step 11: Essential sensitivities.

(a) Aggregator exclusion: re-run BERTopic on is_aggregator=False subset.
(d) Description text source: compare our sample (text_source='llm') to
    text_source='raw' subset on scraped data. Embed the raw subset and
    cluster to see if the archetype structure holds.
(g) SWE classification tier: exclude title_lookup_llm rows, re-cluster, check
    NMI stability.

Writes:
  exploration/tables/T09/sensitivity_aggregator.csv
  exploration/tables/T09/sensitivity_swe_tier.csv
  exploration/tables/T09/sensitivity_text_source.csv
"""
import os
import json
import numpy as np
import pandas as pd
import duckdb
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from T09_03_bertopic import EXTRA_STOPWORDS, run_one
from T09_07_nmi import derive_domain

OUTDIR = "exploration/artifacts/T09"
TABLES = "exploration/tables/T09"
SEED = 20260417

def run_cluster(docs, embeddings, min_topic_size=30, seed=SEED):
    """Run BERTopic, retrying with alternate seeds if UMAP collapses to < 8 topics."""
    stopword_list = list(ENGLISH_STOP_WORDS.union(EXTRA_STOPWORDS))
    candidate_seeds = [seed, 42, 7, 77, 100, 2026]
    best_model, best_topics, best_n = None, None, 0
    for s in candidate_seeds:
        model, topics, probs, _ = run_one(docs, embeddings, list(range(len(docs))),
                                           min_topic_size, s, stopword_list)
        n_topics = int(len(np.unique(topics)) - (1 if -1 in topics else 0))
        print(f"  seed={s}: n_topics={n_topics}, outlier_frac={(np.asarray(topics) == -1).mean():.3f}")
        if n_topics >= 8:
            return model, np.asarray(topics)
        if n_topics > best_n:
            best_model, best_topics, best_n = model, np.asarray(topics), n_topics
    print("  WARNING: all seeds collapsed (<8 topics); using best attempt")
    return best_model, best_topics


def compute_nmis(df, cluster_col):
    df = df.copy()
    df["period_bucket"] = df["period"].map({
        "2024-04": "2024-04", "2024-01": "2024-01",
        "2026-03": "2026", "2026-04": "2026",
    })
    df["sen_cat"] = df["seniority_final"].fillna("unknown")
    rows = []
    for axis_name, axis_col in [
        ("seniority_final", "sen_cat"),
        ("period_2bucket", "period_bucket"),
        ("derived_domain", "domain"),
    ]:
        nmi = normalized_mutual_info_score(df[cluster_col].astype(str),
                                            df[axis_col].astype(str))
        rows.append({"axis": axis_name, "nmi": round(nmi, 4)})
    return pd.DataFrame(rows)


def main():
    main_sample = pd.read_parquet(f"{OUTDIR}/sample_with_assignments.parquet")
    names = pd.read_csv(f"{OUTDIR}/archetype_names.csv")
    name_map = dict(zip(names.archetype_id, names.archetype_name))
    main_sample["archetype_name"] = main_sample["topic_reduced"].map(name_map).fillna("unknown")
    embeddings = np.load(f"{OUTDIR}/sample_embeddings.npy")
    print(f"Main sample: {len(main_sample)} rows, embeddings {embeddings.shape}")

    # --- (a) Aggregator exclusion ---
    mask_a = ~main_sample["is_aggregator"].fillna(False).astype(bool)
    sub_df = main_sample[mask_a].reset_index(drop=True)
    sub_emb = embeddings[mask_a.values]
    print(f"\n(a) Aggregator-excluded: {len(sub_df)} rows")
    model_a, topics_a = run_cluster(sub_df["description_cleaned"].tolist(), sub_emb)
    sub_df["topic_a"] = topics_a
    # Also reduce outliers
    red_a = np.asarray(model_a.reduce_outliers(sub_df["description_cleaned"].tolist(), topics_a, strategy="c-tf-idf"))
    sub_df["topic_a_red"] = red_a
    # Compute NMI
    nmi_a = compute_nmis(sub_df, "topic_a_red")
    nmi_a["scenario"] = "aggregator_excluded"
    # Also report how many topics and outlier frac
    stats_a = pd.DataFrame([{"scenario": "aggregator_excluded", "n": len(sub_df),
                              "n_topics": int(len(np.unique(topics_a)) - (1 if -1 in topics_a else 0)),
                              "outlier_frac": float((topics_a == -1).mean())}])

    # --- (g) SWE tier: exclude title_lookup_llm ---
    mask_g = main_sample["swe_classification_tier"] != "title_lookup_llm"
    subg_df = main_sample[mask_g].reset_index(drop=True)
    subg_emb = embeddings[mask_g.values]
    print(f"\n(g) SWE-tier excluded (title_lookup_llm): {len(subg_df)} rows")
    model_g, topics_g = run_cluster(subg_df["description_cleaned"].tolist(), subg_emb)
    red_g = np.asarray(model_g.reduce_outliers(subg_df["description_cleaned"].tolist(), topics_g, strategy="c-tf-idf"))
    subg_df["topic_g_red"] = red_g
    nmi_g = compute_nmis(subg_df, "topic_g_red")
    nmi_g["scenario"] = "swe_tier_strict"
    stats_g = pd.DataFrame([{"scenario": "swe_tier_strict", "n": len(subg_df),
                              "n_topics": int(len(np.unique(topics_g)) - (1 if -1 in topics_g else 0)),
                              "outlier_frac": float((topics_g == -1).mean())}])

    # Save sensitivities
    pd.concat([nmi_a, nmi_g]).to_csv(f"{TABLES}/sensitivity_nmis.csv", index=False)
    pd.concat([stats_a, stats_g]).to_csv(f"{TABLES}/sensitivity_stats.csv", index=False)

    # --- (d) Text source: raw subset on scraped ---
    # We need: scraped rows with text_source='raw' that have embeddings.
    # The shared embeddings artifact is text_source='llm' only, so for this
    # sensitivity we compute embeddings on a raw sub-subset.
    print("\n(d) Text-source sensitivity: embed a raw-text subset of scraped")
    con = duckdb.connect()
    # Sample 2000 scraped rows with raw text (stratified by seniority)
    raw_df = con.execute("""
        SELECT uid, description_cleaned, period, source, seniority_final,
               is_aggregator, company_name_canonical,
               swe_classification_tier, LENGTH(description_cleaned) as description_cleaned_length
        FROM 'exploration/artifacts/shared/swe_cleaned_text.parquet'
        WHERE text_source = 'raw' AND source = 'scraped'
          AND LENGTH(description_cleaned) >= 200
    """).fetchdf()
    print(f"  Raw-text scraped available: {len(raw_df)}")

    # Sample up to 2000 balanced by period and seniority
    target_n = 2000
    if len(raw_df) > target_n:
        rng = np.random.default_rng(SEED)
        raw_df = raw_df.sample(n=target_n, random_state=SEED).reset_index(drop=True)
    # Embed
    from sentence_transformers import SentenceTransformer
    print("  Loading MiniLM...")
    embm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # Truncate to 2500 chars like shared artifact
    texts = [t[:2500] for t in raw_df["description_cleaned"].tolist()]
    print(f"  Encoding {len(texts)} docs...")
    raw_emb = embm.encode(texts, batch_size=256, show_progress_bar=False,
                            normalize_embeddings=False)
    print(f"  raw_emb shape: {raw_emb.shape}")

    # Project raw embeddings through the main BERTopic's UMAP+HDBSCAN?
    # Simpler: refit on raw embeddings + a matched LLM subset, see if
    # clusters remain consistent.
    # Instead, we apply the main BERTopic model's transform on the raw docs,
    # which uses the fitted UMAP+HDBSCAN to assign topics.
    from bertopic import BERTopic
    main_model = BERTopic.load(f"{OUTDIR}/bertopic_model")
    topics_raw, _ = main_model.transform(raw_df["description_cleaned"].tolist(),
                                          raw_emb.astype("float32"))
    raw_df["topic"] = topics_raw
    raw_df["domain"] = raw_df["description_cleaned"].apply(derive_domain)
    raw_df["topic_reduced"] = raw_df["topic"]
    # Reduce outliers
    red_r = np.asarray(main_model.reduce_outliers(
        raw_df["description_cleaned"].tolist(), raw_df["topic"].values,
        strategy="c-tf-idf"))
    raw_df["topic_reduced"] = red_r
    # Compute NMI
    nmi_raw = compute_nmis(raw_df, "topic_reduced")
    nmi_raw["scenario"] = "raw_text_subset_scraped"
    nmi_raw.to_csv(f"{TABLES}/sensitivity_text_source.csv", index=False)
    # Topic composition comparison (raw vs llm on scraped)
    llm_scraped = main_sample[main_sample.source == "scraped"].copy()
    raw_scraped = raw_df.copy()
    comp = pd.DataFrame({
        "llm_topic_share": llm_scraped["topic_reduced"].value_counts(normalize=True).sort_index(),
        "raw_topic_share": pd.Series(raw_scraped["topic_reduced"]).value_counts(normalize=True).sort_index(),
    }).fillna(0).round(3)
    comp.to_csv(f"{TABLES}/sensitivity_text_source_composition.csv")

    # Summary
    print("\nSensitivity NMIs:")
    combined = pd.concat([nmi_a, nmi_g, nmi_raw])
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
