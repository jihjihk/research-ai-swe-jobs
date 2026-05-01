"""T15 Step 5: UMAP visualization with density contours and movement arrows.
Also produces PCA and t-SNE comparisons.

Outputs:
  figures/T15/umap_period_seniority.png
  figures/T15/umap_density_contours.png
  figures/T15/pca_period_seniority.png
  figures/T15/tsne_period_seniority.png
  artifacts/T15/umap_coords.npy
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

ROOT = Path("/home/jihgaboot/gabor/job-research")
ART = ROOT / "exploration/artifacts/T15"
FIGS = ROOT / "exploration/figures/T15"
FIGS.mkdir(parents=True, exist_ok=True)


def main():
    idx = pd.read_parquet(ART / "sample_index.parquet")
    emb = np.load(ART / "sample_embeddings.npy")
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    print(f"  sample={len(idx)} emb={emb.shape}")

    # UMAP
    print("  fitting UMAP...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.15, metric="cosine", random_state=42)
    U = reducer.fit_transform(emb)
    np.save(ART / "umap_coords.npy", U)

    idx["ux"] = U[:, 0]
    idx["uy"] = U[:, 1]
    idx["group"] = idx["period2"] + "_" + idx["seniority_3level"]

    colors = {
        "2024_junior": "#1f77b4", "2024_mid": "#6baed6", "2024_senior": "#08306b",
        "2026_junior": "#d73027", "2026_mid": "#f46d43", "2026_senior": "#7f0000",
    }

    # Scatter: period x seniority
    fig, ax = plt.subplots(figsize=(12, 10))
    for g, c in colors.items():
        sub = idx[idx["group"] == g]
        if len(sub) == 0:
            continue
        ax.scatter(sub["ux"], sub["uy"], s=5, alpha=0.4, c=c, label=f"{g} (n={len(sub)})")
    ax.legend(markerscale=2, fontsize=9, loc="best")
    ax.set_title("UMAP of SWE description embeddings — period x seniority_3level\n(cosine metric, sentence-transformer embeddings)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(FIGS / "umap_period_seniority.png", dpi=140)
    plt.close()
    print("  wrote umap_period_seniority.png")

    # Density contours + movement arrows
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharex=True, sharey=True)
    for ax, sen in zip(axes, ["junior", "mid", "senior"]):
        for p, c in [("2024", "steelblue"), ("2026", "tomato")]:
            sub = idx[(idx["period2"] == p) & (idx["seniority_3level"] == sen)]
            if len(sub) < 10:
                continue
            ax.scatter(sub["ux"], sub["uy"], s=4, alpha=0.15, c=c, label=f"{p} (n={len(sub)})")
            try:
                sns.kdeplot(x=sub["ux"], y=sub["uy"], levels=5, ax=ax, color=c,
                            linewidths=1.5, alpha=0.9)
            except Exception:
                pass
        # centroid arrow
        sub24 = idx[(idx["period2"] == "2024") & (idx["seniority_3level"] == sen)]
        sub26 = idx[(idx["period2"] == "2026") & (idx["seniority_3level"] == sen)]
        if len(sub24) > 0 and len(sub26) > 0:
            c24 = (sub24["ux"].mean(), sub24["uy"].mean())
            c26 = (sub26["ux"].mean(), sub26["uy"].mean())
            ax.annotate("", xy=c26, xytext=c24,
                        arrowprops=dict(arrowstyle="->", color="black", lw=2))
            ax.plot(*c24, "o", color="steelblue", markersize=10, markeredgecolor="black")
            ax.plot(*c26, "s", color="tomato", markersize=10, markeredgecolor="black")
        ax.set_title(f"Seniority = {sen}")
        ax.legend(fontsize=9)
        ax.set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")
    plt.suptitle("UMAP density contours by period with centroid-movement arrows", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGS / "umap_density_contours.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("  wrote umap_density_contours.png")

    # PCA (2D)
    print("  PCA...")
    pca = PCA(n_components=2, random_state=42)
    P = pca.fit_transform(emb)
    fig, ax = plt.subplots(figsize=(12, 10))
    for g, c in colors.items():
        sub_idx = idx["group"] == g
        if sub_idx.sum() == 0:
            continue
        ax.scatter(P[sub_idx, 0], P[sub_idx, 1], s=5, alpha=0.4, c=c,
                   label=f"{g} (n={sub_idx.sum()})")
    ax.legend(markerscale=2, fontsize=9)
    ax.set_title(f"PCA of SWE description embeddings\nExplained var: {pca.explained_variance_ratio_.sum():.2%}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(FIGS / "pca_period_seniority.png", dpi=140)
    plt.close()

    # t-SNE (on subsample for speed)
    print("  t-SNE (on 4k subsample)...")
    rng = np.random.default_rng(42)
    sub = rng.choice(len(emb), size=min(4000, len(emb)), replace=False)
    tsne = TSNE(n_components=2, metric="cosine", init="pca", random_state=42,
                perplexity=40, learning_rate="auto")
    T = tsne.fit_transform(emb[sub])
    sub_idx = idx.iloc[sub].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    for g, c in colors.items():
        mask = sub_idx["group"] == g
        if mask.sum() == 0:
            continue
        ax.scatter(T[mask, 0], T[mask, 1], s=6, alpha=0.5, c=c,
                   label=f"{g} (n={mask.sum()})")
    ax.legend(markerscale=2, fontsize=9)
    ax.set_title("t-SNE of SWE description embeddings (4k subsample, cosine)")
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    plt.tight_layout()
    plt.savefig(FIGS / "tsne_period_seniority.png", dpi=140)
    plt.close()

    print("Done T15 step 03.")


if __name__ == "__main__":
    main()
