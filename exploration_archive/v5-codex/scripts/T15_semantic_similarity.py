from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx  # imported for consistency with exploration env; not used directly
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from T14_T15_common import (
    ensure_dir,
    load_cleaned_text,
    load_embeddings,
    load_stoplist,
    l2_normalize,
    trimmed_centroid,
)


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ensure_dir(ROOT / "exploration" / "tables" / "T15")
FIG_DIR = ensure_dir(ROOT / "exploration" / "figures" / "T15")

VIZ_CAP_PER_GROUP = 300
NN_K = 5


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_frame() -> pd.DataFrame:
    text = load_cleaned_text(
        [
            "uid",
            "description_cleaned",
            "text_source",
            "source",
            "period",
            "seniority_final",
            "seniority_3level",
            "is_aggregator",
            "company_name_canonical",
            "yoe_extracted",
        ]
    )
    emb, idx = load_embeddings()
    frame = idx.merge(text, on="uid", how="inner")
    frame["company_key"] = (
        frame["company_name_canonical"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "unknown_company")
    )
    frame["is_yoe_junior"] = frame["yoe_extracted"].fillna(99).le(2)
    return frame


def validate_tokenizer() -> None:
    stop = {"acme", "corp", "ltd"}
    pat = re.compile(r"[a-z0-9][a-z0-9+#.\-/]{1,}")

    def tok(doc: str) -> list[str]:
        return [t for t in pat.findall(doc.lower()) if t not in stop]

    assert "c++" in tok("C++ engineer")
    assert "next.js" in tok("Next.js platform role")
    assert "ci/cd" in tok("CI/CD pipelines")
    assert "acme" not in tok("Acme Corp builds software")
    assert "corp" not in tok("Acme Corp builds software")


def tokenizer_factory(stoplist: set[str]):
    pat = re.compile(r"[a-z0-9][a-z0-9+#.\-/]{1,}")

    def tok(doc: str) -> list[str]:
        return [t for t in pat.findall(doc.lower()) if t not in stoplist]

    return tok


def stratified_sample(frame: pd.DataFrame, cap: int = VIZ_CAP_PER_GROUP) -> pd.DataFrame:
    rows = []
    for _, group in frame.groupby(["period", "seniority_3level"], dropna=False):
        if len(group) <= cap:
            rows.append(group)
        else:
            rows.append(group.sample(n=cap, random_state=42))
    return pd.concat(rows, ignore_index=True)


def compute_centroids(vectors: np.ndarray, labels: pd.Series) -> dict[str, np.ndarray]:
    out = {}
    for label in sorted(labels.unique()):
        mask = labels == label
        out[label] = trimmed_centroid(vectors[mask.to_numpy()], trim_frac=0.10)
    return out


def centroid_matrix(centroids: dict[str, np.ndarray]) -> pd.DataFrame:
    keys = sorted(centroids)
    rows = []
    for a in keys:
        row = {"group": a}
        for b in keys:
            sim = float(
                np.dot(centroids[a], centroids[b])
                / (np.linalg.norm(centroids[a]) * np.linalg.norm(centroids[b]) + 1e-12)
            )
            row[b] = sim
        rows.append(row)
    return pd.DataFrame(rows)


def group_distance_table(frame: pd.DataFrame, vectors: np.ndarray, centroids: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    labels = frame["period"] + "|" + frame["seniority_3level"]
    for g in sorted(labels.unique()):
        mask = labels == g
        X = vectors[mask.to_numpy()]
        c = centroids[g]
        sims = (X @ c) / (np.linalg.norm(X, axis=1) * np.linalg.norm(c) + 1e-12)
        rows.append(
            {
                "group": g,
                "n": int(len(X)),
                "mean_cosine_distance": float((1 - sims).mean()),
                "p90_cosine_distance": float(np.quantile(1 - sims, 0.90)),
            }
        )
    return pd.DataFrame(rows)


def source_convergence(frame: pd.DataFrame, vectors: np.ndarray) -> pd.DataFrame:
    rows = []
    for source in ["kaggle_asaniczka", "kaggle_arshkon", "scraped"]:
        for seniority in ["junior", "mid", "senior", "unknown"]:
            mask = (frame["source"] == source) & (frame["seniority_3level"] == seniority)
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "source": source,
                    "seniority_3level": seniority,
                    "n": int(mask.sum()),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    centroids = {}
    for source in ["kaggle_asaniczka", "kaggle_arshkon", "scraped"]:
        for seniority in ["junior", "mid", "senior", "unknown"]:
            mask = (frame["source"] == source) & (frame["seniority_3level"] == seniority)
            if mask.sum() == 0:
                continue
            centroids[(source, seniority)] = trimmed_centroid(vectors[mask.to_numpy()], trim_frac=0.10)
    rows = []
    for source in ["kaggle_asaniczka", "kaggle_arshkon", "scraped"]:
        if (source, "junior") in centroids and (source, "senior") in centroids:
            rows.append(
                {
                    "source": source,
                    "metric": "junior_senior_similarity",
                    "value": float(
                        np.dot(centroids[(source, "junior")], centroids[(source, "senior")])
                        / (
                            np.linalg.norm(centroids[(source, "junior")])
                            * np.linalg.norm(centroids[(source, "senior")])
                            + 1e-12
                        )
                    ),
                }
            )
        if (source, "junior") in centroids and (source, "mid") in centroids:
            rows.append(
                {
                    "source": source,
                    "metric": "junior_mid_similarity",
                    "value": float(
                        np.dot(centroids[(source, "junior")], centroids[(source, "mid")])
                        / (
                            np.linalg.norm(centroids[(source, "junior")])
                            * np.linalg.norm(centroids[(source, "mid")])
                            + 1e-12
                        )
                    ),
                }
            )
        if (source, "mid") in centroids and (source, "senior") in centroids:
            rows.append(
                {
                    "source": source,
                    "metric": "mid_senior_similarity",
                    "value": float(
                        np.dot(centroids[(source, "mid")], centroids[(source, "senior")])
                        / (
                            np.linalg.norm(centroids[(source, "mid")])
                            * np.linalg.norm(centroids[(source, "senior")])
                            + 1e-12
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def yoe_convergence(frame: pd.DataFrame, vectors: np.ndarray) -> pd.DataFrame:
    rows = []
    for source in ["kaggle_asaniczka", "kaggle_arshkon", "scraped"]:
        mask_j = (frame["source"] == source) & frame["is_yoe_junior"]
        mask_s = (frame["source"] == source) & (~frame["is_yoe_junior"])
        if mask_j.sum() == 0 or mask_s.sum() == 0:
            continue
        cj = trimmed_centroid(vectors[mask_j.to_numpy()], trim_frac=0.10)
        cs = trimmed_centroid(vectors[mask_s.to_numpy()], trim_frac=0.10)
        rows.append(
            {
                "source": source,
                "metric": "yoe_junior_senior_similarity",
                "value": float(
                    np.dot(cj, cs) / (np.linalg.norm(cj) * np.linalg.norm(cs) + 1e-12)
                ),
                "n_junior": int(mask_j.sum()),
                "n_senior": int(mask_s.sum()),
            }
        )
    return pd.DataFrame(rows)


def average_similarity_comparison(centroids: dict[str, np.ndarray]) -> pd.DataFrame:
    periods = ["2024-01", "2024-04", "2026-03", "2026-04"]
    seniorities = ["junior", "mid", "senior", "unknown"]
    same_sen = []
    same_per = []
    for seniority in seniorities:
        keys = [f"{p}|{seniority}" for p in periods if f"{p}|{seniority}" in centroids]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = centroids[keys[i]], centroids[keys[j]]
                same_sen.append(
                    float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                )
    for period in periods:
        keys = [f"{period}|{s}" for s in seniorities if f"{period}|{s}" in centroids]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = centroids[keys[i]], centroids[keys[j]]
                same_per.append(
                    float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                )
    return pd.DataFrame(
        [
            {
                "comparison": "same_seniority_across_periods",
                "mean_similarity": float(np.mean(same_sen)),
                "n_pairs": int(len(same_sen)),
            },
            {
                "comparison": "same_period_across_seniority",
                "mean_similarity": float(np.mean(same_per)),
                "n_pairs": int(len(same_per)),
            },
        ]
    )


def top_neighbors(frame: pd.DataFrame, vectors: np.ndarray, rep_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = frame["source"].isin(["kaggle_arshkon", "kaggle_asaniczka"])
    query_mask = (frame["source"] == "scraped") & (frame["seniority_3level"] == "junior")
    X_train = vectors[train_mask.to_numpy()]
    X_query = vectors[query_mask.to_numpy()]
    train_meta = frame[train_mask].reset_index(drop=True)
    query_meta = frame[query_mask].reset_index(drop=True)
    nn = NearestNeighbors(n_neighbors=NN_K, metric="cosine").fit(X_train)
    dist, idxs = nn.kneighbors(X_query)
    rows = []
    for qi in range(len(query_meta)):
        qrow = query_meta.iloc[qi]
        for rank, (d, j) in enumerate(zip(dist[qi], idxs[qi]), start=1):
            trow = train_meta.iloc[j]
            rows.append(
                {
                    "representation": rep_name,
                    "query_uid": qrow.uid,
                    "query_period": qrow.period,
                    "neighbor_uid": trow.uid,
                    "neighbor_source": trow.source,
                    "neighbor_period": trow.period,
                    "neighbor_seniority_3level": trow.seniority_3level,
                    "rank": rank,
                    "cosine_distance": float(d),
                    "cosine_similarity": 1 - float(d),
                }
            )
    nn_df = pd.DataFrame(rows)
    base = (
        train_meta["seniority_3level"].value_counts(normalize=True)
        .rename_axis("neighbor_seniority_3level")
        .reset_index(name="base_rate")
    )
    obs = (
        nn_df.groupby("neighbor_seniority_3level")
        .size()
        .div(len(nn_df))
        .reset_index(name="neighbor_rate")
    )
    summary = base.merge(obs, on="neighbor_seniority_3level", how="outer").fillna(0.0)
    summary["excess_pp"] = summary["neighbor_rate"] - summary["base_rate"]
    summary["representation"] = rep_name
    return nn_df, summary


def outlier_table(frame: pd.DataFrame, vectors: np.ndarray, centroids: dict[str, np.ndarray], rep_name: str) -> pd.DataFrame:
    rows = []
    labels = frame["period"] + "|" + frame["seniority_3level"]
    for g in sorted(labels.unique()):
        mask = labels == g
        X = vectors[mask.to_numpy()]
        if len(X) == 0:
            continue
        c = centroids[g]
        sims = (X @ c) / (np.linalg.norm(X, axis=1) * np.linalg.norm(c) + 1e-12)
        dist = 1 - sims
        idx = np.argsort(dist)[::-1][:5]
        sub = frame[mask].reset_index(drop=True).iloc[idx]
        for rank, (local_idx, d) in enumerate(zip(idx, dist[idx]), start=1):
            row = sub.iloc[rank - 1]
            rows.append(
                {
                    "representation": rep_name,
                    "group": g,
                    "rank": rank,
                    "uid": row.uid,
                    "source": row.source,
                    "period": row.period,
                    "seniority_3level": row.seniority_3level,
                    "company_name_canonical": row.company_name_canonical,
                    "description_cleaned": row.description_cleaned[:220],
                    "cosine_distance": float(d),
                }
            )
    return pd.DataFrame(rows)


def plot_similarity_heatmaps(embed_mat: pd.DataFrame, tfidf_mat: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), constrained_layout=True)
    for ax, mat, title in zip(
        axes,
        [embed_mat, tfidf_mat],
        ["Embedding centroid similarity", "TF-IDF centroid similarity"],
    ):
        sns.heatmap(
            mat.set_index("group"),
            cmap="rocket_r",
            vmin=0.70,
            vmax=1.0,
            square=False,
            ax=ax,
            cbar_kws={"label": "Cosine similarity"},
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
    return fig


def plot_manifold(viz_frame: pd.DataFrame, embeddings: np.ndarray) -> plt.Figure:
    labels = viz_frame["period"] + " / " + viz_frame["seniority_3level"]
    palette = sns.color_palette("tab20", n_colors=12)
    groups = sorted(labels.unique())
    color_map = {g: palette[i % len(palette)] for i, g in enumerate(groups)}
    y = np.array([color_map[g] for g in labels])

    umap_proj = umap.UMAP(
        n_neighbors=30,
        min_dist=0.12,
        metric="cosine",
        random_state=42,
    ).fit_transform(embeddings)
    pca_proj = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    tsne_proj = TSNE(
        n_components=2,
        perplexity=min(50, max(5, len(viz_frame) // 100)),
        learning_rate="auto",
        init="pca",
        random_state=42,
    ).fit_transform(embeddings)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)
    projections = [("UMAP", umap_proj), ("PCA", pca_proj), ("t-SNE", tsne_proj)]
    for ax, (name, proj) in zip(axes, projections):
        ax.scatter(proj[:, 0], proj[:, 1], c=y, s=12, alpha=0.55, linewidths=0)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        if name == "UMAP":
            centroids = {}
            for g in groups:
                mask = labels == g
                centroids[g] = proj[mask].mean(axis=0)
                if mask.sum() >= 20:
                    sns.kdeplot(
                        x=proj[mask, 0],
                        y=proj[mask, 1],
                        ax=ax,
                        levels=2,
                        color=color_map[g],
                        linewidths=1.0,
                        fill=False,
                        alpha=0.45,
                    )
            # arrows connecting chronological centroids within each seniority
            periods = ["2024-01", "2024-04", "2026-03", "2026-04"]
            for seniority in ["junior", "mid", "senior", "unknown"]:
                seq = [f"{p} / {seniority}" for p in periods if f"{p} / {seniority}" in centroids]
                if len(seq) < 2:
                    continue
                color = color_map[seq[0]]
                for a, b in zip(seq, seq[1:]):
                    ax.annotate(
                        "",
                        xy=centroids[b],
                        xytext=centroids[a],
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.6, alpha=0.9),
                    )
            for g, xy in centroids.items():
                ax.text(xy[0], xy[1], g.replace(" / ", "\n"), fontsize=7, ha="center", va="center")
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", label=g, markerfacecolor=color_map[g], markersize=7)
            for g in groups
        ]
    axes[0].legend(handles=handles, title="Period / seniority", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    return fig


def plot_nn_excess(nn_summary_embed: pd.DataFrame, nn_summary_tfidf: pd.DataFrame) -> plt.Figure:
    combined = pd.concat([nn_summary_embed, nn_summary_tfidf], ignore_index=True)
    order = ["junior", "mid", "senior", "unknown"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=combined,
        x="neighbor_seniority_3level",
        y="excess_pp",
        hue="representation",
        order=order,
        ax=ax,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("2024 neighbor seniority")
    ax.set_ylabel("Neighbor share minus base rate")
    ax.set_title("Nearest-neighbor seniority excess for 2026 entry postings")
    return fig


def main() -> None:
    validate_tokenizer()
    frame = build_frame()
    frame = frame[frame["text_source"] == "llm"].copy()
    emb, idx = load_embeddings()
    vectors = l2_normalize(np.asarray(emb, dtype=np.float32))
    frame = frame.sort_values("row_index").reset_index(drop=True)

    # Coverage / sample table
    coverage = (
        frame.groupby(["source", "period", "seniority_3level"], dropna=False)
        .agg(
            n=("uid", "size"),
            n_yoe_junior=("is_yoe_junior", "sum"),
            n_unknown=("seniority_final", lambda s: (s == "unknown").sum()),
        )
        .reset_index()
    )
    save_csv(coverage, TABLE_DIR / "T15_llm_coverage_by_source_period_seniority.csv")

    # Balanced visual sample
    viz = stratified_sample(frame, cap=VIZ_CAP_PER_GROUP)
    viz_vectors = vectors[viz["row_index"].to_numpy()]

    # Full centroids and similarity matrices under embeddings
    labels = frame["period"] + "|" + frame["seniority_3level"]
    embed_centroids = compute_centroids(vectors, labels)
    embed_matrix = centroid_matrix(embed_centroids)
    save_csv(embed_matrix, TABLE_DIR / "T15_group_similarity_embedding.csv")

    embed_disp = group_distance_table(frame, vectors, embed_centroids)
    save_csv(embed_disp, TABLE_DIR / "T15_group_dispersion_embedding.csv")

    # TF-IDF representation on the same labeled subset
    stoplist = load_stoplist()
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer_factory(stoplist),
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        min_df=5,
        max_features=25000,
        ngram_range=(1, 2),
        norm="l2",
    )
    tfidf = vectorizer.fit_transform(frame["description_cleaned"].fillna(""))
    tfidf_reduced = normalize(TruncatedSVD(n_components=100, random_state=42).fit_transform(tfidf))
    tfidf_centroids = compute_centroids(tfidf_reduced, labels)
    tfidf_matrix = centroid_matrix(tfidf_centroids)
    save_csv(tfidf_matrix, TABLE_DIR / "T15_group_similarity_tfidf.csv")

    tfidf_disp = group_distance_table(frame, tfidf_reduced, tfidf_centroids)
    save_csv(tfidf_disp, TABLE_DIR / "T15_group_dispersion_tfidf.csv")

    # Source-level convergence and calibration
    emb_source = source_convergence(frame, vectors)
    tfidf_source = source_convergence(frame, tfidf_reduced)
    emb_source["representation"] = "embedding"
    tfidf_source["representation"] = "tfidf"
    source_conv = pd.concat([emb_source, tfidf_source], ignore_index=True)
    save_csv(source_conv, TABLE_DIR / "T15_source_convergence.csv")

    yoe_emb = yoe_convergence(frame, vectors)
    yoe_tfidf = yoe_convergence(frame, tfidf_reduced)
    yoe_emb["representation"] = "embedding"
    yoe_tfidf["representation"] = "tfidf"
    save_csv(pd.concat([yoe_emb, yoe_tfidf], ignore_index=True), TABLE_DIR / "T15_yoe_convergence.csv")

    comparison_embed = average_similarity_comparison(embed_centroids)
    comparison_tfidf = average_similarity_comparison(tfidf_centroids)
    comparison_embed["representation"] = "embedding"
    comparison_tfidf["representation"] = "tfidf"
    save_csv(pd.concat([comparison_embed, comparison_tfidf], ignore_index=True), TABLE_DIR / "T15_structure_comparison.csv")

    # Robustness table
    robust = pd.DataFrame(
        [
            {
                "finding": "junior_senior_similarity_source_arshkon",
                "embedding_value": float(
                    source_conv[
                        (source_conv["representation"] == "embedding")
                        & (source_conv["source"] == "kaggle_arshkon")
                        & (source_conv["metric"] == "junior_senior_similarity")
                    ]["value"].iloc[0]
                ),
                "tfidf_value": float(
                    source_conv[
                        (source_conv["representation"] == "tfidf")
                        & (source_conv["source"] == "kaggle_arshkon")
                        & (source_conv["metric"] == "junior_senior_similarity")
                    ]["value"].iloc[0]
                ),
            },
            {
                "finding": "junior_senior_similarity_source_scraped",
                "embedding_value": float(
                    source_conv[
                        (source_conv["representation"] == "embedding")
                        & (source_conv["source"] == "scraped")
                        & (source_conv["metric"] == "junior_senior_similarity")
                    ]["value"].iloc[0]
                ),
                "tfidf_value": float(
                    source_conv[
                        (source_conv["representation"] == "tfidf")
                        & (source_conv["source"] == "scraped")
                        & (source_conv["metric"] == "junior_senior_similarity")
                    ]["value"].iloc[0]
                ),
            },
            {
                "finding": "same_period_vs_same_seniority_gap",
                "embedding_value": float(
                    comparison_embed.loc[
                        comparison_embed["comparison"] == "same_period_across_seniority", "mean_similarity"
                    ].iloc[0]
                    - comparison_embed.loc[
                        comparison_embed["comparison"] == "same_seniority_across_periods", "mean_similarity"
                    ].iloc[0]
                ),
                "tfidf_value": float(
                    comparison_tfidf.loc[
                        comparison_tfidf["comparison"] == "same_period_across_seniority", "mean_similarity"
                    ].iloc[0]
                    - comparison_tfidf.loc[
                        comparison_tfidf["comparison"] == "same_seniority_across_periods", "mean_similarity"
                    ].iloc[0]
                ),
            },
            {
                "finding": "yoe_junior_senior_similarity_arshkon",
                "embedding_value": float(
                    yoe_emb[(yoe_emb["representation"] == "embedding") & (yoe_emb["source"] == "kaggle_arshkon")][
                        "value"
                    ].iloc[0]
                ),
                "tfidf_value": float(
                    yoe_tfidf[(yoe_tfidf["representation"] == "tfidf") & (yoe_tfidf["source"] == "kaggle_arshkon")][
                        "value"
                    ].iloc[0]
                ),
            },
        ]
    )
    save_csv(robust, TABLE_DIR / "T15_representation_robustness.csv")

    # Nearest-neighbor analysis
    nn_embed, nn_summary_embed = top_neighbors(frame, vectors, "embedding")
    save_csv(nn_embed, TABLE_DIR / "T15_nearest_neighbors_embedding.csv")
    save_csv(nn_summary_embed, TABLE_DIR / "T15_nn_summary_embedding.csv")

    nn_tfidf, nn_summary_tfidf = top_neighbors(frame, tfidf_reduced, "tfidf")
    save_csv(nn_tfidf, TABLE_DIR / "T15_nearest_neighbors_tfidf.csv")
    save_csv(nn_summary_tfidf, TABLE_DIR / "T15_nn_summary_tfidf.csv")
    save_csv(
        nn_summary_embed.merge(nn_summary_tfidf, on="neighbor_seniority_3level", how="outer", suffixes=("_embed", "_tfidf")),
        TABLE_DIR / "T15_nn_excess_summary.csv",
    )

    # Outliers
    outliers_embed = outlier_table(frame, vectors, embed_centroids, "embedding")
    outliers_tfidf = outlier_table(frame, tfidf_reduced, tfidf_centroids, "tfidf")
    save_csv(outliers_embed, TABLE_DIR / "T15_outliers_embedding.csv")
    save_csv(outliers_tfidf, TABLE_DIR / "T15_outliers_tfidf.csv")

    # Figures
    fig = plot_similarity_heatmaps(embed_matrix, tfidf_matrix)
    save_fig(fig, FIG_DIR / "T15_similarity_heatmaps.png")

    fig = plot_manifold(viz, viz_vectors)
    save_fig(fig, FIG_DIR / "T15_umap_pca_tsne.png")

    fig = plot_nn_excess(nn_summary_embed, nn_summary_tfidf)
    save_fig(fig, FIG_DIR / "T15_nearest_neighbor_excess.png")


if __name__ == "__main__":
    main()
