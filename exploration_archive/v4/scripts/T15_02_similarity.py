"""T15 Steps 2-4, 6, 7: Centroid similarity matrix, convergence analysis,
within-group dispersion, nearest-neighbor analysis. Under both embeddings & TF-IDF.

Outputs:
  tables/T15/centroid_similarity_embeddings.csv
  tables/T15/centroid_similarity_tfidf.csv
  tables/T15/convergence_by_period.csv
  tables/T15/convergence_yoe_proxy.csv
  tables/T15/within_group_dispersion.csv
  tables/T15/nearest_neighbor_entry2026.csv
  tables/T15/representation_robustness.csv
  figures/T15/centroid_similarity_heatmap_embeddings.png
  figures/T15/centroid_similarity_heatmap_tfidf.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path("/home/jihgaboot/gabor/job-research")
ART = ROOT / "exploration/artifacts/T15"
TABLES = ROOT / "exploration/tables/T15"
FIGS = ROOT / "exploration/figures/T15"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

TRIM_FRAC = 0.10


def trimmed_centroid(X: np.ndarray, trim: float = TRIM_FRAC) -> np.ndarray:
    """Remove trim% most distant from raw centroid (cosine distance), then recompute."""
    if len(X) == 0:
        return np.zeros(X.shape[1])
    c = X.mean(axis=0)
    c = c / (np.linalg.norm(c) + 1e-12)
    # Assume rows are L2-normalized -> cosine sim = dot
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sims = Xn @ c
    cutoff = np.quantile(sims, trim)
    mask = sims >= cutoff
    c2 = Xn[mask].mean(axis=0)
    return c2 / (np.linalg.norm(c2) + 1e-12)


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return An @ Bn.T


def main():
    idx = pd.read_parquet(ART / "sample_index.parquet")
    emb = np.load(ART / "sample_embeddings.npy")
    tfidf = np.load(ART / "sample_tfidf_svd.npy")
    # Ensure L2 norm
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    tfidf = tfidf / (np.linalg.norm(tfidf, axis=1, keepdims=True) + 1e-12)
    print(f"  sample={len(idx)}  emb={emb.shape}  tfidf={tfidf.shape}")

    idx["group"] = idx["period2"] + "_" + idx["seniority_3level"]
    groups = ["2024_junior", "2024_mid", "2024_senior",
              "2026_junior", "2026_mid", "2026_senior"]

    # Centroids per group (trimmed)
    def build_centroids(X):
        out = {}
        for g in groups:
            rows = idx[idx["group"] == g]["sample_row"].values
            out[g] = trimmed_centroid(X[rows])
        return out

    c_emb = build_centroids(emb)
    c_tf = build_centroids(tfidf)

    # Centroid similarity matrix
    def sim_matrix(c):
        C = np.vstack([c[g] for g in groups])
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        return C @ C.T
    sim_e = sim_matrix(c_emb)
    sim_t = sim_matrix(c_tf)

    pd.DataFrame(sim_e, index=groups, columns=groups).to_csv(TABLES / "centroid_similarity_embeddings.csv")
    pd.DataFrame(sim_t, index=groups, columns=groups).to_csv(TABLES / "centroid_similarity_tfidf.csv")

    for name, mat in [("embeddings", sim_e), ("tfidf", sim_t)]:
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(mat, annot=True, fmt=".3f", cmap="viridis",
                    xticklabels=groups, yticklabels=groups, ax=ax,
                    vmin=mat.min(), vmax=1.0)
        ax.set_title(f"Centroid cosine similarity ({name}, trimmed 10%)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(FIGS / f"centroid_similarity_heatmap_{name}.png", dpi=130)
        plt.close()

    # ---------- Step 3: Convergence analysis ----------
    # Within each period, similarity between seniority levels.
    conv_rows = []
    for rep_name, c in [("embeddings", c_emb), ("tfidf", c_tf)]:
        for period in ["2024", "2026"]:
            pairs = [("junior", "senior"), ("junior", "mid"), ("mid", "senior")]
            for a, b in pairs:
                ga = f"{period}_{a}"; gb = f"{period}_{b}"
                sim = float(c[ga] @ c[gb])
                conv_rows.append({"representation": rep_name, "period": period,
                                  "pair": f"{a}_vs_{b}", "cosine": sim})
    conv_df = pd.DataFrame(conv_rows)
    # Wide form showing 2024 vs 2026 per pair
    wide = conv_df.pivot_table(index=["representation", "pair"], columns="period", values="cosine").reset_index()
    wide["delta_2026_minus_2024"] = wide["2026"] - wide["2024"]
    wide.to_csv(TABLES / "convergence_by_period.csv", index=False)
    print("  convergence by period:")
    print(wide.to_string(index=False))

    # Within-2024 calibration: if we had source-split convergence that's tricky because
    # asaniczka has almost no entry. Instead compute asaniczka->arshkon centroid shift
    # (source variation baseline) vs arshkon->scraped (cross-period).
    def sim_between(Xsel_a, Xsel_b):
        ca = trimmed_centroid(emb[Xsel_a])
        cb = trimmed_centroid(emb[Xsel_b])
        return float(ca @ cb)

    arsh = idx[idx["source"] == "kaggle_arshkon"]["sample_row"].values
    asan = idx[idx["source"] == "kaggle_asaniczka"]["sample_row"].values
    scr = idx[idx["source"] == "scraped"]["sample_row"].values

    calib_rows = []
    for rep_name, X in [("embeddings", emb), ("tfidf", tfidf)]:
        ca_arsh = trimmed_centroid(X[arsh])
        ca_asan = trimmed_centroid(X[asan])
        ca_scr = trimmed_centroid(X[scr])
        calib_rows.append({
            "representation": rep_name,
            "arshkon_vs_asaniczka_cos": float(ca_arsh @ ca_asan),
            "arshkon_vs_scraped_cos": float(ca_arsh @ ca_scr),
            "asaniczka_vs_scraped_cos": float(ca_asan @ ca_scr),
        })
    calib_df = pd.DataFrame(calib_rows)
    calib_df.to_csv(TABLES / "within2024_calibration_centroids.csv", index=False)
    print("  within-2024 calibration centroids:")
    print(calib_df.to_string(index=False))

    # ---------- YOE-based proxy convergence ----------
    # Define: entry = yoe<=2, mid = 3-5, senior = 6+
    y = idx["yoe_extracted"]
    def yoe_group(v):
        if pd.isna(v): return "unknown"
        try: v = float(v)
        except: return "unknown"
        if v <= 2: return "junior"
        if v <= 5: return "mid"
        return "senior"
    idx["yoe_bin"] = y.map(yoe_group)
    idx["group_yoe"] = idx["period2"] + "_" + idx["yoe_bin"]

    yoe_groups = ["2024_junior", "2024_mid", "2024_senior",
                  "2026_junior", "2026_mid", "2026_senior"]
    def build_c_yoe(X):
        out = {}
        for g in yoe_groups:
            rows = idx[idx["group_yoe"] == g]["sample_row"].values
            out[g] = trimmed_centroid(X[rows]) if len(rows) >= 20 else np.zeros(X.shape[1])
        return out
    cy_e = build_c_yoe(emb)
    cy_t = build_c_yoe(tfidf)

    yoe_rows = []
    yoe_count = idx.groupby(["period2", "yoe_bin"]).size().to_dict()
    for rep_name, c in [("embeddings", cy_e), ("tfidf", cy_t)]:
        for period in ["2024", "2026"]:
            for a, b in [("junior", "senior"), ("junior", "mid"), ("mid", "senior")]:
                ga = f"{period}_{a}"; gb = f"{period}_{b}"
                na = yoe_count.get((period, a), 0); nb = yoe_count.get((period, b), 0)
                if na < 20 or nb < 20:
                    continue
                sim = float(c[ga] @ c[gb])
                yoe_rows.append({"representation": rep_name, "period": period,
                                 "pair": f"{a}_vs_{b}", "cosine": sim,
                                 "n_a": na, "n_b": nb})
    yoe_df = pd.DataFrame(yoe_rows)
    yoe_wide = yoe_df.pivot_table(index=["representation", "pair"], columns="period", values="cosine").reset_index()
    if "2024" in yoe_wide.columns and "2026" in yoe_wide.columns:
        yoe_wide["delta_2026_minus_2024"] = yoe_wide["2026"] - yoe_wide["2024"]
    yoe_wide.to_csv(TABLES / "convergence_yoe_proxy.csv", index=False)
    print("  convergence (yoe proxy):")
    print(yoe_wide.to_string(index=False))

    # ---------- Step 4: Within-group dispersion ----------
    disp_rows = []
    for rep_name, X in [("embeddings", emb), ("tfidf", tfidf)]:
        for g in groups:
            rows = idx[idx["group"] == g]["sample_row"].values
            if len(rows) < 2:
                continue
            Xg = X[rows]
            # Sample up to 500 for pairwise cost
            rng = np.random.default_rng(1)
            take = rng.choice(len(Xg), size=min(500, len(Xg)), replace=False)
            Xs = Xg[take]
            S = Xs @ Xs.T
            iu = np.triu_indices(len(Xs), k=1)
            avg_pair = float(S[iu].mean())
            disp_rows.append({"representation": rep_name, "group": g,
                              "n_total": len(rows), "n_sampled": len(Xs),
                              "mean_pairwise_cosine": avg_pair,
                              "mean_centroid_dist": 1 - avg_pair})
    disp_df = pd.DataFrame(disp_rows)
    disp_df.to_csv(TABLES / "within_group_dispersion.csv", index=False)

    # ---------- Step 6: Nearest-neighbor analysis ----------
    # For each 2026 entry (junior) posting, find 5 NN in 2024 embeddings.
    q_idx = idx[(idx["period2"] == "2026") & (idx["seniority_3level"] == "junior")].copy()
    d_idx = idx[idx["period2"] == "2024"].copy()
    Xq = emb[q_idx["sample_row"].values]
    Xd = emb[d_idx["sample_row"].values]
    S = Xq @ Xd.T  # (nq, nd)
    top5 = np.argsort(-S, axis=1)[:, :5]

    d_sen = d_idx["seniority_3level"].values
    d_src = d_idx["source"].values
    base_rate = d_idx["seniority_3level"].value_counts(normalize=True).to_dict()

    counts = {"junior": 0, "mid": 0, "senior": 0, "unknown": 0}
    src_counts = {"kaggle_arshkon": 0, "kaggle_asaniczka": 0}
    for i in range(len(q_idx)):
        for j in top5[i]:
            s = d_sen[j]
            counts[s] = counts.get(s, 0) + 1
            src_counts[d_src[j]] = src_counts.get(d_src[j], 0) + 1
    total = sum(counts.values())
    nn_rows = []
    for s, c in counts.items():
        br = base_rate.get(s, 0)
        obs = c / total if total else 0
        nn_rows.append({"representation": "embeddings", "nn_seniority": s,
                        "count": c, "observed_rate": obs, "base_rate": br,
                        "excess": obs - br})
    pd.DataFrame(nn_rows).to_csv(TABLES / "nearest_neighbor_entry2026_emb.csv", index=False)
    print("  NN entry 2026 -> 2024 (embeddings):")
    print(pd.DataFrame(nn_rows).to_string(index=False))

    # Repeat with TF-IDF
    Xq_t = tfidf[q_idx["sample_row"].values]
    Xd_t = tfidf[d_idx["sample_row"].values]
    S_t = Xq_t @ Xd_t.T
    top5_t = np.argsort(-S_t, axis=1)[:, :5]
    counts_t = {"junior": 0, "mid": 0, "senior": 0, "unknown": 0}
    for i in range(len(q_idx)):
        for j in top5_t[i]:
            counts_t[d_sen[j]] = counts_t.get(d_sen[j], 0) + 1
    total_t = sum(counts_t.values())
    nn_rows_t = []
    for s, c in counts_t.items():
        br = base_rate.get(s, 0)
        obs = c / total_t if total_t else 0
        nn_rows_t.append({"representation": "tfidf", "nn_seniority": s,
                          "count": c, "observed_rate": obs, "base_rate": br,
                          "excess": obs - br})
    pd.DataFrame(nn_rows_t).to_csv(TABLES / "nearest_neighbor_entry2026_tfidf.csv", index=False)
    print("  NN entry 2026 -> 2024 (tfidf):")
    print(pd.DataFrame(nn_rows_t).to_string(index=False))

    # Source distribution of NN
    src_tbl = pd.DataFrame([{"source": k, "count": v, "rate": v/total}
                            for k, v in src_counts.items()])
    src_tbl.to_csv(TABLES / "nearest_neighbor_entry2026_source.csv", index=False)

    # ---------- Representation robustness summary ----------
    rob = []
    # Convergence direction agreement
    for _, r in wide.iterrows():
        rob.append({"finding": f"{r['pair']} similarity 2024->2026",
                    "representation": r["representation"],
                    "2024": r["2024"], "2026": r["2026"],
                    "delta": r.get("delta_2026_minus_2024", np.nan)})
    # NN majority
    top_nn_emb = max(nn_rows, key=lambda r: r["observed_rate"])
    top_nn_tf = max(nn_rows_t, key=lambda r: r["observed_rate"])
    rob.append({"finding": "NN majority seniority (2026 entry -> 2024)",
                "representation": "embeddings",
                "2024": top_nn_emb["nn_seniority"], "2026": "",
                "delta": top_nn_emb["excess"]})
    rob.append({"finding": "NN majority seniority (2026 entry -> 2024)",
                "representation": "tfidf",
                "2024": top_nn_tf["nn_seniority"], "2026": "",
                "delta": top_nn_tf["excess"]})
    pd.DataFrame(rob).to_csv(TABLES / "representation_robustness.csv", index=False)

    print("Done T15 step 02.")


if __name__ == "__main__":
    main()
