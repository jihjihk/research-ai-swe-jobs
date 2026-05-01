"""V1.4 - Full-corpus NMI re-computation.

T09 computed NMI on 8,000-sample. Re-compute on full 63,701 corpus.
Project T09 archetype labels onto unsampled rows via nearest-centroid (cosine).
Compute NMI(cluster, domain=archetype), NMI(cluster, period), NMI(cluster, seniority).
"""
import duckdb
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score

OUT_DIR = Path("/home/jihgaboot/gabor/job-research/exploration/artifacts/V1")
EMB_PATH = "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_embeddings.npy"
EMB_IDX_PATH = "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_embedding_index.parquet"
ARCH_PATH = "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_archetype_labels.parquet"
CLEAN_PARQ = "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_cleaned_text.parquet"


def main():
    con = duckdb.connect()

    # Load embeddings and UID index
    print("Loading embeddings...")
    X = np.load(EMB_PATH)
    print(f"  embeddings shape: {X.shape}")
    idx_df = con.execute(f"SELECT * FROM '{EMB_IDX_PATH}' ORDER BY row_idx").df()
    print(f"  index rows: {len(idx_df)}")
    uid_to_row = dict(zip(idx_df["uid"], idx_df["row_idx"]))

    # Load archetype labels for the 8k sample
    arch = con.execute(f"SELECT * FROM '{ARCH_PATH}'").df()
    print(f"  archetype labels: {len(arch)}")

    # Load SWE corpus with period, seniority
    corpus = con.execute(f"""
        SELECT uid, period, seniority_final
        FROM '{CLEAN_PARQ}'
    """).df()
    print(f"  corpus: {len(corpus)}")

    # Compute centroids for each archetype on embedding space
    print("Computing centroids...")
    arch_with_emb = arch.merge(idx_df, on="uid")
    print(f"  archetype-with-embedding: {len(arch_with_emb)}")
    centroids = {}
    for arch_id, sub in arch_with_emb.groupby("archetype_id"):
        emb = X[sub["row_idx"].values]
        # Normalize to unit vectors (for cosine)
        emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        centroid = emb_norm.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        centroids[arch_id] = centroid
    print(f"  centroids: {len(centroids)}")

    # Project all 34,102 rows onto nearest centroid
    print("Projecting all rows to nearest centroid...")
    cent_ids = sorted(centroids.keys())
    cent_matrix = np.array([centroids[i] for i in cent_ids])
    # Normalize X
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    sims = X_norm @ cent_matrix.T
    assigned_idx = sims.argmax(axis=1)
    assigned_arch = np.array([cent_ids[i] for i in assigned_idx])

    proj_df = pd.DataFrame({
        "uid": idx_df["uid"].values,
        "row_idx": idx_df["row_idx"].values,
        "archetype_assigned": assigned_arch,
    })
    # Join with period and seniority
    proj_df = proj_df.merge(corpus, on="uid", how="left")
    proj_df["period_group"] = proj_df["period"].apply(lambda p: "2024" if str(p).startswith("2024") else "2026")
    print(f"  projected rows: {len(proj_df)}")

    # Domain comparison uses the ORIGINAL archetype from T09 for the 8k sample (ground truth)
    # Full-corpus re-computation: use assigned archetype as the cluster, and various labels
    # Compare:
    # 1. NMI(archetype_assigned, archetype_ground_truth) - for those in the 8k sample only
    # 2. NMI(archetype_assigned, period)
    # 3. NMI(archetype_assigned, seniority)

    # For the domain comparison, we want NMI(cluster, domain). Since archetype IS the domain,
    # the question is: does the archetype structure predict itself (tautological: should be high)?
    # The T09 finding is: if we cluster the corpus separately and check NMI to domain labels,
    # we get domain-dominated clustering. Here we're using the same labels as clusters and asking
    # how they align with period/seniority. That's fine for the question but we need to be careful.

    # ACTUALLY the T09 question was: given the archetype clusters (obtained from embeddings),
    # how well do THEY predict period / seniority? The answer was period>>seniority in NMI.
    # But in the published T09 number, "domain=0.26" means the clustering separates domains well.
    # T09 compared some clustering (e.g., KMeans) to (domain_label, period, seniority) labels.
    # Without running the clustering, I'll instead:
    # (a) Project archetypes -> full corpus = cluster_label_fullcorpus
    # (b) Compute NMI(cluster_label_fullcorpus, period), NMI(cluster_label_fullcorpus, seniority_final)
    # These are the key numbers. For the domain measure, the assigned archetype IS the domain label
    # (self-tautological). T09's domain=0.26 compared a clustering to a separate domain label obtained
    # from some other signal. I will instead use **title_normalized** or **title_lc** substring-matching
    # as an independent domain proxy, to avoid tautology.

    # Load title info
    con2 = duckdb.connect()
    title_df = con2.execute("""
        SELECT uid, lower(title_normalized) AS title
        FROM '/home/jihgaboot/gabor/job-research/data/unified.parquet'
        WHERE source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe = true
    """).df()

    # Simple title-based domain proxy:
    def title_to_domain(title):
        if not isinstance(title, str):
            return "other"
        t = title.lower()
        if any(k in t for k in ["ml ", "machine learning", " ai ", "data scient", "nlp"]):
            return "ml_ai"
        if any(k in t for k in ["frontend", "front-end", "react", "angular", "vue"]):
            return "frontend"
        if any(k in t for k in ["backend", "back-end", "java"]):
            return "backend"
        if any(k in t for k in ["devops", "sre", "site reliability", "platform engineer", "kubernetes"]):
            return "devops"
        if any(k in t for k in ["mobile", "ios", "android"]):
            return "mobile"
        if any(k in t for k in ["full stack", "full-stack", "fullstack"]):
            return "fullstack"
        if any(k in t for k in ["data engineer"]):
            return "data_eng"
        if any(k in t for k in ["embedded", "firmware"]):
            return "embedded"
        if any(k in t for k in ["security", "cyber"]):
            return "security"
        if any(k in t for k in ["systems engineer"]):
            return "systems"
        return "generic_swe"

    title_df["domain_proxy"] = title_df["title"].apply(title_to_domain)
    proj_df = proj_df.merge(title_df[["uid", "domain_proxy"]], on="uid", how="left")

    # === Compute NMI on full corpus (34,102 embedded rows) ===
    # Only use rows with available labels
    mask = proj_df["seniority_final"].notna() & proj_df["period_group"].notna() & proj_df["domain_proxy"].notna()
    sub = proj_df[mask].copy()
    print(f"\nFull-corpus rows for NMI: {len(sub)}")

    nmi_period = normalized_mutual_info_score(sub["archetype_assigned"], sub["period_group"])
    nmi_seniority = normalized_mutual_info_score(sub["archetype_assigned"], sub["seniority_final"])
    nmi_domain_proxy = normalized_mutual_info_score(sub["archetype_assigned"], sub["domain_proxy"])

    print(f"\n=== Full-corpus NMI (cluster=archetype_assigned, 34,102 rows) ===")
    print(f"  NMI(archetype, period):        {nmi_period:.4f}")
    print(f"  NMI(archetype, seniority):     {nmi_seniority:.4f}")
    print(f"  NMI(archetype, domain_proxy):  {nmi_domain_proxy:.4f}")

    # Also re-compute on the original 8k sample for reference (should match T09)
    orig_mask = sub["uid"].isin(arch["uid"])
    orig_sub = sub[orig_mask]
    orig_sub = orig_sub.merge(arch[["uid", "archetype_id"]], on="uid")
    print(f"\nOriginal 8k sample for NMI: {len(orig_sub)}")
    if len(orig_sub) > 0:
        nmi_o_period = normalized_mutual_info_score(orig_sub["archetype_id"], orig_sub["period_group"])
        nmi_o_seniority = normalized_mutual_info_score(orig_sub["archetype_id"], orig_sub["seniority_final"])
        nmi_o_domain = normalized_mutual_info_score(orig_sub["archetype_id"], orig_sub["domain_proxy"])
        print(f"=== 8k-sample NMI (cluster=T09 archetype ground truth) ===")
        print(f"  NMI(archetype, period):        {nmi_o_period:.4f}  (T09 reported 0.04)")
        print(f"  NMI(archetype, seniority):     {nmi_o_seniority:.4f}  (T09 reported 0.03)")
        print(f"  NMI(archetype, domain_proxy):  {nmi_o_domain:.4f}  (T09 reported domain=0.26, but different domain def)")

    # Summary
    summary = {
        "n_full_corpus": int(len(sub)),
        "n_projected_rows": int(len(proj_df)),
        "full_corpus_nmi": {
            "period": float(nmi_period),
            "seniority": float(nmi_seniority),
            "domain_proxy": float(nmi_domain_proxy),
        },
        "ordering": {
            "domain_gt_period": nmi_domain_proxy > nmi_period,
            "period_gt_seniority": nmi_period > nmi_seniority,
            "domain_gt_seniority": nmi_domain_proxy > nmi_seniority,
            "ratio_domain_to_period": float(nmi_domain_proxy / nmi_period) if nmi_period > 0 else None,
            "ratio_period_to_seniority": float(nmi_period / nmi_seniority) if nmi_seniority > 0 else None,
        },
    }
    if len(orig_sub) > 0:
        summary["sample_8k_nmi"] = {
            "period": float(nmi_o_period),
            "seniority": float(nmi_o_seniority),
            "domain_proxy": float(nmi_o_domain),
        }
    with open(OUT_DIR / "V1_4_nmi_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'V1_4_nmi_summary.json'}")


if __name__ == "__main__":
    main()
