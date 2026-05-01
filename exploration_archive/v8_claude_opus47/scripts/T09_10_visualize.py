"""T09 Step 10: Visualization of embeddings.

UMAP (2D) + PCA (2D) + t-SNE (2D), colored by:
 (a) archetype (BERTopic reduced)
 (b) period bucket (2024-01 / 2024-04 / 2026)
 (c) seniority (J2 vs mid-senior vs S1 vs unknown)
 (d) derived domain

Writes PNGs into exploration/figures/T09/
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

OUTDIR = "exploration/artifacts/T09"
FIGS = "exploration/figures/T09"
os.makedirs(FIGS, exist_ok=True)
SEED = 20260417


def colorize(labels, cmap="tab20"):
    import matplotlib.cm as cm
    unique = sorted(set(labels))
    cmap_obj = cm.get_cmap(cmap, len(unique))
    m = {u: cmap_obj(i) for i, u in enumerate(unique)}
    return [m[l] for l in labels], m


def scatter_by(ax, xy, labels, cmap="tab20", title="", size=3, alpha=0.55, legend_max=22):
    colors, m = colorize(labels, cmap)
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=size, alpha=alpha, linewidths=0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    # Legend (up to legend_max)
    handles = []
    for lbl, col in m.items():
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="",
                                   color=col, markersize=6, label=str(lbl)))
    ax.legend(handles=handles[:legend_max], loc="center left",
              bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False)


def main():
    df = pd.read_parquet(f"{OUTDIR}/sample_with_assignments.parquet")
    names = pd.read_csv(f"{OUTDIR}/archetype_names.csv")
    name_map = dict(zip(names.archetype_id, names.archetype_name))
    df["archetype_name"] = df["topic_reduced"].map(name_map).fillna("unknown")
    embeddings = np.load(f"{OUTDIR}/sample_embeddings.npy")
    print(f"Loaded {len(df)} rows, emb {embeddings.shape}")

    df["period_bucket"] = df["period"].map({
        "2024-04": "2024-04 arshkon",
        "2024-01": "2024-01 asaniczka",
        "2026-03": "2026 scraped",
        "2026-04": "2026 scraped",
    })
    # Seniority category
    def sen_cat(x):
        if x in ("entry", "associate"):
            return "J2 (entry/associate)"
        if x == "mid-senior":
            return "mid-senior"
        if x == "director":
            return "director"
        return "unknown"
    df["sen_cat"] = df["seniority_final"].apply(sen_cat)

    # --- 2D UMAP ---
    print("UMAP 2D...")
    umap2 = UMAP(n_neighbors=30, n_components=2, min_dist=0.1,
                 metric="cosine", random_state=SEED, low_memory=True)
    xy_umap = umap2.fit_transform(embeddings)
    np.save(f"{OUTDIR}/xy_umap.npy", xy_umap)

    print("PCA 2D...")
    xy_pca = PCA(n_components=2, random_state=SEED).fit_transform(embeddings)
    np.save(f"{OUTDIR}/xy_pca.npy", xy_pca)

    print("t-SNE 2D...")
    xy_tsne = TSNE(n_components=2, perplexity=30, random_state=SEED,
                   metric="cosine", init="pca").fit_transform(embeddings)
    np.save(f"{OUTDIR}/xy_tsne.npy", xy_tsne)

    for projname, xy in [("umap", xy_umap), ("pca", xy_pca), ("tsne", xy_tsne)]:
        # One figure per axis
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        scatter_by(ax, xy, df["archetype_name"].tolist(), cmap="tab20",
                   title=f"Embeddings {projname.upper()} — colored by archetype",
                   size=4, alpha=0.6)
        fig.tight_layout()
        fig.savefig(f"{FIGS}/{projname}_by_archetype.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        scatter_by(ax, xy, df["period_bucket"].tolist(), cmap="Set1",
                   title=f"Embeddings {projname.upper()} — colored by period",
                   size=4, alpha=0.6)
        fig.tight_layout()
        fig.savefig(f"{FIGS}/{projname}_by_period.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        scatter_by(ax, xy, df["sen_cat"].tolist(), cmap="Set2",
                   title=f"Embeddings {projname.upper()} — colored by seniority (J2/mid/S1)",
                   size=4, alpha=0.6)
        fig.tight_layout()
        fig.savefig(f"{FIGS}/{projname}_by_seniority.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        scatter_by(ax, xy, df["domain"].tolist(), cmap="tab20",
                   title=f"Embeddings {projname.upper()} — colored by derived domain",
                   size=4, alpha=0.6)
        fig.tight_layout()
        fig.savefig(f"{FIGS}/{projname}_by_domain.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote figures to {FIGS}/")


if __name__ == "__main__":
    main()
