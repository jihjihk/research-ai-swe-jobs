"""T09 Step 15: Build the methods comparison summary table.

Columns: method, n_topics, coherence (NPMI), stability (ARI), outlier %, notes.
"""
import json
import pandas as pd

TABLES = "exploration/tables/T09"
OUTDIR = "exploration/artifacts/T09"


def main():
    sweep = pd.read_csv(f"{TABLES}/bertopic_config_sweep.csv")
    with open(f"{OUTDIR}/bertopic_best_config.json") as f:
        bt_cfg = json.load(f)
    ari = pd.read_csv(f"{TABLES}/bertopic_stability_ari.csv")
    # mean ARI over informative pairs (exclude pairs where either run collapsed)
    # Load expanded stability (if available)
    try:
        expanded = pd.read_csv(f"{TABLES}/bertopic_stability_expanded.csv")
    except FileNotFoundError:
        expanded = None

    # For BERTopic best config: n_topics=22, outlier=27.85%
    best = sweep[sweep.min_topic_size == bt_cfg["min_topic_size"]].iloc[0]

    # NMF config
    with open(f"{OUTDIR}/nmf_best_config.json") as f:
        nmf_cfg = json.load(f)

    # Method ARI
    try:
        method_ari = pd.read_csv(f"{TABLES}/method_ari.csv")
        bt_nmf_ari = method_ari[method_ari.label == "ari_reduced_bt_vs_nmf"].iloc[0]["value"]
    except FileNotFoundError:
        bt_nmf_ari = None

    # Stability: average ARI on non-collapsed pairs (from the expanded run)
    try:
        full = pd.read_csv(f"{TABLES}/bertopic_stability_full.csv")
        mean_ari = full.ari_reduced.mean()
        n_pairs = len(full)
    except FileNotFoundError:
        stable_pairs = ari[(ari.n_topics_a >= 8) & (ari.n_topics_b >= 8)]
        mean_ari = stable_pairs.ari_reduced.mean() if len(stable_pairs) > 0 else None
        n_pairs = len(stable_pairs) if mean_ari is not None else 0

    rows = [
        {
            "method": "BERTopic (UMAP+HDBSCAN, min_topic_size=30)",
            "n_topics": int(best.n_topics),
            "outlier_pct": round(best.outlier_frac * 100, 1),
            "coherence_npmi_top10": round(best.coherence_npmi_top10, 4),
            "stability_ari_cross_seed": round(mean_ari, 3) if mean_ari is not None else None,
            "note": f"Mean ARI={mean_ari:.2f} over {n_pairs} non-collapsed pairs. 3/8 seeds collapsed to <5 topics -- UMAP manifold has two basins. Topology is domain-structured on non-collapsed runs.",
        },
        {
            "method": "BERTopic (mts=20)",
            "n_topics": int(sweep[sweep.min_topic_size == 20].iloc[0].n_topics),
            "outlier_pct": round(sweep[sweep.min_topic_size == 20].iloc[0].outlier_frac * 100, 1),
            "coherence_npmi_top10": round(sweep[sweep.min_topic_size == 20].iloc[0].coherence_npmi_top10, 4),
            "stability_ari_cross_seed": None,
            "note": "More granular; higher outlier share.",
        },
        {
            "method": "BERTopic (mts=50)",
            "n_topics": int(sweep[sweep.min_topic_size == 50].iloc[0].n_topics),
            "outlier_pct": round(sweep[sweep.min_topic_size == 50].iloc[0].outlier_frac * 100, 1),
            "coherence_npmi_top10": round(sweep[sweep.min_topic_size == 50].iloc[0].coherence_npmi_top10, 4),
            "stability_ari_cross_seed": None,
            "note": "Collapsed to 2 topics -- uninformative.",
        },
        {
            "method": f"NMF (TF-IDF, k={nmf_cfg['k']})",
            "n_topics": int(nmf_cfg["k"]),
            "outlier_pct": 0.0,
            "coherence_npmi_top10": None,
            "stability_ari_cross_seed": None,
            "note": (f"Deterministic (given init=nndsvd). ARI vs BERTopic reduced = "
                     f"{bt_nmf_ari:.3f}. Top-term Jaccard median = 0.25. "
                     f"Isolated GenAI/LLM as its own topic; BERTopic merged into "
                     f"broader ai_ml_engineering."),
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(f"{TABLES}/method_comparison_summary.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
