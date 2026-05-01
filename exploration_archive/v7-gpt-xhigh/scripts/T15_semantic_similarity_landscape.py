#!/usr/bin/env python3
"""T15 semantic similarity landscape and convergence analysis.

Memory posture:
- uses shared LLM embeddings; does not recompute embeddings
- samples at most 2,000 rows per year x seniority_3level for primary maps
- uses bounded nearest-neighbor pools instead of all-pairs large groups
- DuckDB uses a 4GB memory limit and one thread
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "exploration" / "artifacts" / "shared"
TABLE_DIR = ROOT / "exploration" / "tables" / "T15"
FIG_DIR = ROOT / "exploration" / "figures" / "T15"

UNIFIED = ROOT / "data" / "unified.parquet"
CLEANED = SHARED / "swe_cleaned_text.parquet"
EMBED_INDEX = SHARED / "swe_embedding_index.parquet"
EMBEDDINGS = SHARED / "swe_embeddings.npy"
ARCHETYPES = SHARED / "swe_archetype_labels.parquet"
T30_PANEL = SHARED / "seniority_definition_panel.csv"

RANDOM_SEED = 20260416
PRIMARY_GROUP_CAP = 2000
NN_QUERY_CAP = 500
NN_CANDIDATE_CAP = 8000
COMPANY_CAP = 50
MIN_CHARS = 100
SENIOR_TITLE_RE = re.compile(r"\b(senior|sr\.?|staff|principal|lead|architect|distinguished)\b", re.I)

JUNIOR_DEFS = ["J1", "J2", "J3", "J4"]
SENIOR_DEFS = ["S1", "S2", "S3", "S4"]


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=1")
    return con


def q(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def stable_rank(value: object) -> int:
    return int(hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12], 16)


def year_from_period(period: object) -> str:
    return str(period)[:4]


def add_panel_flags(df: pd.DataFrame) -> pd.DataFrame:
    seniority = df["seniority_final"].fillna("unknown")
    yoe = df["yoe_extracted"]
    title = df["title_normalized"].fillna("")
    df["J1"] = seniority.eq("entry")
    df["J2"] = seniority.isin(["entry", "associate"])
    df["J3"] = yoe.le(2).fillna(False)
    df["J4"] = yoe.le(3).fillna(False)
    df["S1"] = seniority.isin(["mid-senior", "director"])
    df["S2"] = seniority.eq("director")
    df["S3"] = title.map(lambda s: bool(SENIOR_TITLE_RE.search(s)))
    df["S4"] = yoe.ge(5).fillna(False)
    return df


def load_text_source_coverage() -> pd.DataFrame:
    con = connect()
    coverage = con.execute(
        f"""
        SELECT source, period, text_source, count(*) AS rows
        FROM read_parquet('{q(CLEANED)}')
        GROUP BY source, period, text_source
        ORDER BY source, period, text_source
        """
    ).fetchdf()
    con.close()
    return coverage


def load_embedding_metadata(llm_only: bool = True) -> pd.DataFrame:
    con = connect()
    text_filter = "AND c.text_source = 'llm'" if llm_only else ""
    meta = con.execute(
        f"""
        SELECT
          c.uid,
          c.description_cleaned,
          c.text_source,
          c.source,
          c.period,
          c.seniority_final,
          c.seniority_3level,
          c.is_aggregator,
          c.company_name_canonical,
          c.yoe_extracted,
          c.swe_classification_tier,
          c.seniority_final_source,
          length(c.description_cleaned)::INTEGER AS char_len,
          e.embedding_row,
          u.title,
          u.title_normalized,
          a.archetype,
          a.archetype_name
        FROM read_parquet('{q(CLEANED)}') c
        JOIN read_parquet('{q(EMBED_INDEX)}') e USING (uid)
        LEFT JOIN (
          SELECT uid, title, title_normalized
          FROM read_parquet('{q(UNIFIED)}')
        ) u USING (uid)
        LEFT JOIN read_parquet('{q(ARCHETYPES)}') a USING (uid)
        WHERE length(c.description_cleaned) >= {MIN_CHARS}
          {text_filter}
        """
    ).fetchdf()
    con.close()
    meta["year"] = meta["period"].map(year_from_period)
    meta["source_group"] = meta["source"].map(
        {
            "kaggle_arshkon": "arshkon",
            "kaggle_asaniczka": "asaniczka",
            "scraped": "scraped_2026",
        }
    )
    meta["company_name_canonical"] = meta["company_name_canonical"].fillna("unknown_company")
    meta["is_aggregator"] = meta["is_aggregator"].fillna(False).astype(bool)
    meta["archetype_name"] = meta["archetype_name"].fillna("Unlabeled")
    meta["stable_rank"] = meta["uid"].map(stable_rank)
    meta["company_cap_rank"] = (
        meta["stable_rank"].groupby([meta["year"], meta["company_name_canonical"]]).rank(method="first")
    )
    return add_panel_flags(meta)


def primary_sample(meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in meta.groupby(["year", "seniority_3level"], dropna=False):
        rows.append(group.sort_values("stable_rank").head(PRIMARY_GROUP_CAP))
    sample = pd.concat(rows, ignore_index=True)
    sample.sort_values("stable_rank", inplace=True)
    sample.reset_index(drop=True, inplace=True)
    return sample


def load_embeddings(rows: pd.Series | np.ndarray) -> np.ndarray:
    mmap = np.load(EMBEDDINGS, mmap_mode="r")
    arr = np.asarray(mmap[np.asarray(rows, dtype=np.int64)], dtype=np.float32)
    return normalize(arr, norm="l2", copy=False)


def tfidf_svd(texts: pd.Series, n_components: int = 100) -> tuple[np.ndarray, TfidfVectorizer, TruncatedSVD]:
    vectorizer = TfidfVectorizer(
        max_features=20000,
        min_df=3,
        max_df=0.90,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[\w][\w./#+-]*\b",
        dtype=np.float32,
        sublinear_tf=True,
    )
    mat = vectorizer.fit_transform(texts.fillna(""))
    k = max(2, min(n_components, mat.shape[1] - 1, mat.shape[0] - 1))
    svd = TruncatedSVD(n_components=k, random_state=RANDOM_SEED)
    dense = svd.fit_transform(mat).astype(np.float32)
    return normalize(dense, norm="l2", copy=False), vectorizer, svd


def group_centroids(
    meta: pd.DataFrame,
    x: np.ndarray,
    group_col: str,
    min_n: int = 5,
    trim: float = 0.10,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    centroids: dict[str, np.ndarray] = {}
    for label, idx in meta.groupby(group_col).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        if len(idx_arr) < min_n:
            continue
        sub = x[idx_arr]
        initial = normalize(sub.mean(axis=0, keepdims=True))[0]
        sims = sub @ initial
        if len(idx_arr) >= 20 and trim > 0:
            keep_n = max(min_n, int(math.ceil(len(idx_arr) * (1 - trim))))
            keep = np.argsort(sims)[-keep_n:]
            trimmed = sub[keep]
        else:
            trimmed = sub
        centroid = normalize(trimmed.mean(axis=0, keepdims=True))[0]
        centroids[str(label)] = centroid
        rows.append(
            {
                "group": str(label),
                "n": len(idx_arr),
                "n_trimmed": len(trimmed),
                "mean_cosine_to_trimmed_centroid": float((sub @ centroid).mean()),
            }
        )
    return pd.DataFrame(rows), centroids


def centroid_similarity_table(centroids: dict[str, np.ndarray]) -> pd.DataFrame:
    labels = sorted(centroids)
    if not labels:
        return pd.DataFrame()
    mat = np.vstack([centroids[label] for label in labels])
    sims = mat @ mat.T
    rows = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            rows.append({"group_a": a, "group_b": b, "cosine_similarity": float(sims[i, j])})
    return pd.DataFrame(rows)


def corpus_mask(meta: pd.DataFrame, corpus: str) -> pd.Series:
    if corpus == "arshkon":
        return meta["source"].eq("kaggle_arshkon")
    if corpus == "asaniczka":
        return meta["source"].eq("kaggle_asaniczka")
    if corpus == "pooled_2024":
        return meta["year"].eq("2024")
    if corpus == "scraped_2026":
        return meta["source"].eq("scraped")
    raise ValueError(corpus)


def centroid_for_mask(x: np.ndarray, mask: np.ndarray, min_n: int = 5, trim: float = 0.10) -> tuple[np.ndarray | None, int, int]:
    idx = np.flatnonzero(mask)
    if len(idx) < min_n:
        return None, len(idx), 0
    sub = x[idx]
    c0 = normalize(sub.mean(axis=0, keepdims=True))[0]
    sims = sub @ c0
    if len(idx) >= 20 and trim > 0:
        keep_n = max(min_n, int(math.ceil(len(idx) * (1 - trim))))
        keep = np.argsort(sims)[-keep_n:]
        sub2 = sub[keep]
    else:
        sub2 = sub
    c = normalize(sub2.mean(axis=0, keepdims=True))[0]
    return c, len(idx), len(sub2)


def similarity_between_masks(x: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray, min_n: int = 5) -> tuple[float, int, int]:
    ca, n_a, _ = centroid_for_mask(x, mask_a, min_n=min_n)
    cb, n_b, _ = centroid_for_mask(x, mask_b, min_n=min_n)
    if ca is None or cb is None:
        return np.nan, n_a, n_b
    return float(ca @ cb), n_a, n_b


def convergence_seniority(meta: pd.DataFrame, x: np.ndarray, representation: str, spec: str) -> pd.DataFrame:
    rows = []
    work_mask = np.ones(len(meta), dtype=bool)
    if spec == "no_aggregators":
        work_mask &= ~meta["is_aggregator"].to_numpy()
    elif spec == "company_cap50":
        work_mask &= meta["company_cap_rank"].le(COMPANY_CAP).to_numpy()
    elif spec == "recommended_swe_tier":
        work_mask &= meta["swe_classification_tier"].ne("title_lookup_llm").to_numpy()
    elif spec != "primary_all":
        raise ValueError(spec)

    sims = {}
    for corpus in ["arshkon", "asaniczka", "pooled_2024", "scraped_2026"]:
        base = work_mask & corpus_mask(meta, corpus).to_numpy()
        junior = base & meta["seniority_3level"].eq("junior").to_numpy()
        senior = base & meta["seniority_3level"].eq("senior").to_numpy()
        sim, n_j, n_s = similarity_between_masks(x, junior, senior)
        sims[corpus] = sim
        rows.append(
            {
                "spec": spec,
                "representation": representation,
                "corpus": corpus,
                "junior_n": n_j,
                "senior_n": n_s,
                "junior_senior_similarity": sim,
            }
        )
    out = pd.DataFrame(rows)
    cross = sims["scraped_2026"] - sims["arshkon"]
    pooled_cross = sims["scraped_2026"] - sims["pooled_2024"]
    calibration = sims["asaniczka"] - sims["arshkon"]
    out["arshkon_to_scraped_shift"] = cross
    out["pooled_2024_to_scraped_shift"] = pooled_cross
    out["within_2024_asaniczka_minus_arshkon"] = calibration
    out["calibration_verdict"] = (
        "passes_calibration" if pd.notna(cross) and cross > 0 and abs(cross) > abs(calibration) else "reject_convergence"
    )
    return out


def t30_convergence(meta: pd.DataFrame, x: np.ndarray, representation: str) -> pd.DataFrame:
    rows = []
    for jdef in JUNIOR_DEFS:
        for sdef in SENIOR_DEFS:
            sims = {}
            counts = {}
            for corpus in ["arshkon", "asaniczka", "pooled_2024", "scraped_2026"]:
                base = corpus_mask(meta, corpus).to_numpy()
                jm = base & meta[jdef].to_numpy()
                sm = base & meta[sdef].to_numpy()
                sim, n_j, n_s = similarity_between_masks(x, jm, sm)
                sims[corpus] = sim
                counts[corpus] = (n_j, n_s)
            cross = sims["scraped_2026"] - sims["arshkon"]
            calibration = sims["asaniczka"] - sims["arshkon"]
            rows.append(
                {
                    "representation": representation,
                    "junior_definition": jdef,
                    "senior_definition": sdef,
                    "arshkon_similarity": sims["arshkon"],
                    "asaniczka_similarity": sims["asaniczka"],
                    "pooled_2024_similarity": sims["pooled_2024"],
                    "scraped_2026_similarity": sims["scraped_2026"],
                    "arshkon_junior_n": counts["arshkon"][0],
                    "arshkon_senior_n": counts["arshkon"][1],
                    "scraped_junior_n": counts["scraped_2026"][0],
                    "scraped_senior_n": counts["scraped_2026"][1],
                    "arshkon_to_scraped_shift": cross,
                    "pooled_2024_to_scraped_shift": sims["scraped_2026"] - sims["pooled_2024"],
                    "within_2024_asaniczka_minus_arshkon": calibration,
                    "calibration_verdict": (
                        "passes_calibration"
                        if pd.notna(cross) and cross > 0 and abs(cross) > abs(calibration)
                        else "reject_convergence"
                    ),
                    "thin_cell_flag": min(counts["arshkon"] + counts["scraped_2026"]) < 20,
                }
            )
    return pd.DataFrame(rows)


def archetype_convergence(meta: pd.DataFrame, x: np.ndarray, representation: str) -> pd.DataFrame:
    rows = []
    for archetype, group in meta.loc[meta["archetype_name"].ne("Unlabeled")].groupby("archetype_name"):
        idx = group.index.to_numpy()
        local = meta.loc[idx]
        sims = {}
        counts = {}
        for corpus in ["arshkon", "asaniczka", "scraped_2026"]:
            base = corpus_mask(local, corpus).to_numpy()
            junior = base & local["seniority_3level"].eq("junior").to_numpy()
            senior = base & local["seniority_3level"].eq("senior").to_numpy()
            sim, n_j, n_s = similarity_between_masks(x[idx], junior, senior, min_n=10)
            sims[corpus] = sim
            counts[corpus] = (n_j, n_s)
        rows.append(
            {
                "representation": representation,
                "archetype_name": archetype,
                "arshkon_similarity": sims["arshkon"],
                "asaniczka_similarity": sims["asaniczka"],
                "scraped_2026_similarity": sims["scraped_2026"],
                "arshkon_junior_n": counts["arshkon"][0],
                "arshkon_senior_n": counts["arshkon"][1],
                "scraped_junior_n": counts["scraped_2026"][0],
                "scraped_senior_n": counts["scraped_2026"][1],
                "arshkon_to_scraped_shift": sims["scraped_2026"] - sims["arshkon"]
                if pd.notna(sims["scraped_2026"]) and pd.notna(sims["arshkon"])
                else np.nan,
                "within_2024_asaniczka_minus_arshkon": sims["asaniczka"] - sims["arshkon"]
                if pd.notna(sims["asaniczka"]) and pd.notna(sims["arshkon"])
                else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["calibration_verdict"] = np.where(
        out["arshkon_to_scraped_shift"].notna()
        & out["within_2024_asaniczka_minus_arshkon"].notna()
        & (out["arshkon_to_scraped_shift"] > 0)
        & (out["arshkon_to_scraped_shift"].abs() > out["within_2024_asaniczka_minus_arshkon"].abs()),
        "passes_calibration",
        "reject_or_thin",
    )
    return out


def pairwise_dispersion(meta: pd.DataFrame, x: np.ndarray, group_col: str, representation: str) -> pd.DataFrame:
    rows = []
    for label, idx in meta.groupby(group_col).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        n = len(idx_arr)
        if n < 2:
            continue
        sub = x[idx_arr]
        summed = sub.sum(axis=0)
        avg_pairwise = (float(summed @ summed) - n) / (n * (n - 1))
        rows.append(
            {
                "representation": representation,
                "group": label,
                "n": n,
                "avg_pairwise_cosine": avg_pairwise,
                "dispersion_1_minus_avg_cosine": 1 - avg_pairwise,
            }
        )
    return pd.DataFrame(rows)


def variance_explained(meta: pd.DataFrame, x: np.ndarray, representation: str) -> pd.DataFrame:
    rows = []
    global_mean = x.mean(axis=0)
    total_ss = float(((x - global_mean) ** 2).sum())
    factors = ["year", "source_group", "seniority_3level", "archetype_name"]
    for factor in factors:
        within = 0.0
        for _, idx in meta.groupby(factor, dropna=False).groups.items():
            idx_arr = np.asarray(list(idx), dtype=int)
            sub = x[idx_arr]
            c = sub.mean(axis=0)
            within += float(((sub - c) ** 2).sum())
        rows.append(
            {
                "representation": representation,
                "factor": factor,
                "groups": meta[factor].nunique(dropna=False),
                "eta_squared": (total_ss - within) / total_ss if total_ss else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["representation", "eta_squared"], ascending=[True, False])


def nearest_neighbor_analysis(meta_all: pd.DataFrame) -> pd.DataFrame:
    candidate = meta_all.loc[meta_all["year"].eq("2024")].copy()
    candidate = candidate.loc[candidate["company_cap_rank"] <= COMPANY_CAP].sort_values("stable_rank").head(NN_CANDIDATE_CAP)
    rows = []
    cand_embed = load_embeddings(candidate["embedding_row"].to_numpy())
    cand_sen = candidate["seniority_3level"].fillna("unknown").to_numpy()
    base_rates = pd.Series(cand_sen).value_counts(normalize=True).to_dict()

    for jdef in JUNIOR_DEFS:
        query = meta_all.loc[meta_all["source"].eq("scraped") & meta_all[jdef]].sort_values("stable_rank").head(NN_QUERY_CAP)
        if query.empty:
            continue
        q_embed = load_embeddings(query["embedding_row"].to_numpy())
        sim = q_embed @ cand_embed.T
        top_idx = np.argpartition(-sim, kth=min(4, sim.shape[1] - 1), axis=1)[:, :5]
        neigh = cand_sen[top_idx.ravel()]
        counts = pd.Series(neigh).value_counts(normalize=True).to_dict()
        for seniority in ["junior", "mid", "senior", "unknown"]:
            rows.append(
                {
                    "representation": "embedding",
                    "query_definition": jdef,
                    "query_n": len(query),
                    "candidate_n": len(candidate),
                    "neighbor_seniority": seniority,
                    "neighbor_share": counts.get(seniority, 0.0),
                    "candidate_base_share": base_rates.get(seniority, 0.0),
                    "excess_over_base_pp": (counts.get(seniority, 0.0) - base_rates.get(seniority, 0.0)) * 100,
                }
            )

    # TF-IDF nearest neighbors on the same bounded candidate/query pools per definition.
    for jdef in JUNIOR_DEFS:
        query = meta_all.loc[meta_all["source"].eq("scraped") & meta_all[jdef]].sort_values("stable_rank").head(NN_QUERY_CAP)
        if query.empty:
            continue
        both_text = pd.concat([candidate["description_cleaned"], query["description_cleaned"]], ignore_index=True)
        vectorizer = TfidfVectorizer(
            max_features=20000,
            min_df=3,
            max_df=0.90,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[\w][\w./#+-]*\b",
            dtype=np.float32,
            sublinear_tf=True,
        )
        mat = vectorizer.fit_transform(both_text.fillna(""))
        cand_mat = mat[: len(candidate)]
        query_mat = mat[len(candidate) :]
        sim = cosine_similarity(query_mat, cand_mat, dense_output=True)
        top_idx = np.argpartition(-sim, kth=min(4, sim.shape[1] - 1), axis=1)[:, :5]
        neigh = cand_sen[top_idx.ravel()]
        counts = pd.Series(neigh).value_counts(normalize=True).to_dict()
        for seniority in ["junior", "mid", "senior", "unknown"]:
            rows.append(
                {
                    "representation": "tfidf",
                    "query_definition": jdef,
                    "query_n": len(query),
                    "candidate_n": len(candidate),
                    "neighbor_seniority": seniority,
                    "neighbor_share": counts.get(seniority, 0.0),
                    "candidate_base_share": base_rates.get(seniority, 0.0),
                    "excess_over_base_pp": (counts.get(seniority, 0.0) - base_rates.get(seniority, 0.0)) * 100,
                }
            )
    return pd.DataFrame(rows)


def outliers(meta: pd.DataFrame, x: np.ndarray, representation: str) -> pd.DataFrame:
    rows = []
    for label, idx in meta.groupby("year_seniority").groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        if len(idx_arr) < 20:
            continue
        centroid, _n, _nt = centroid_for_mask(x, np.isin(np.arange(len(meta)), idx_arr), min_n=5)
        if centroid is None:
            continue
        sims = x[idx_arr] @ centroid
        order = np.argsort(sims)[: min(20, len(sims))]
        for local_i in order:
            row = meta.iloc[idx_arr[local_i]]
            rows.append(
                {
                    "representation": representation,
                    "year_seniority": label,
                    "uid": row["uid"],
                    "source": row["source"],
                    "title": row["title"],
                    "seniority_final": row["seniority_final"],
                    "seniority_3level": row["seniority_3level"],
                    "archetype_name": row["archetype_name"],
                    "cosine_to_group_centroid": float(sims[local_i]),
                    "snippet": " ".join(str(row["description_cleaned"]).split())[:350],
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out.sort_values(["representation", "year_seniority", "cosine_to_group_centroid"], inplace=True)
    return out


def raw_text_tfidf_sensitivity() -> pd.DataFrame:
    con = connect()
    raw = con.execute(
        f"""
        SELECT
          uid, description, source, period, seniority_final, seniority_3level,
          is_aggregator, company_name_canonical, yoe_extracted, swe_classification_tier,
          title_normalized, length(description)::INTEGER AS char_len
        FROM read_parquet('{q(UNIFIED)}')
        WHERE source_platform = 'linkedin'
          AND is_english = true
          AND date_flag = 'ok'
          AND is_swe = true
          AND description IS NOT NULL
          AND length(description) >= 100
        """
    ).fetchdf()
    con.close()
    raw["year"] = raw["period"].map(year_from_period)
    raw["source_group"] = raw["source"].map(
        {
            "kaggle_arshkon": "arshkon",
            "kaggle_asaniczka": "asaniczka",
            "scraped": "scraped_2026",
        }
    )
    raw["company_name_canonical"] = raw["company_name_canonical"].fillna("unknown_company")
    raw["stable_rank"] = raw["uid"].map(stable_rank)
    raw["company_cap_rank"] = (
        raw["stable_rank"].groupby([raw["year"], raw["company_name_canonical"]]).rank(method="first")
    )
    raw = add_panel_flags(raw)
    rows = []
    for _, group in raw.groupby(["year", "seniority_3level"], dropna=False):
        rows.append(group.sort_values("stable_rank").head(PRIMARY_GROUP_CAP))
    sample = pd.concat(rows, ignore_index=True)
    sample.sort_values("stable_rank", inplace=True)
    sample.reset_index(drop=True, inplace=True)
    x_raw, _vec, _svd = tfidf_svd(sample["description"], n_components=100)
    return convergence_seniority(sample, x_raw, "raw_description_tfidf_svd", "primary_all")


def plot_similarity_heatmaps(emb_sim: pd.DataFrame, tfidf_sim: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, table, title in [
        (axes[0], emb_sim, "Embeddings"),
        (axes[1], tfidf_sim, "TF-IDF/SVD"),
    ]:
        mat = table.pivot(index="group_a", columns="group_b", values="cosine_similarity")
        labels = sorted(mat.index)
        mat = mat.loc[labels, labels]
        im = ax.imshow(mat.to_numpy(), vmin=0, vmax=1, cmap="magma")
        ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=8)
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82)
    cbar.set_label("Trimmed centroid cosine")
    fig.suptitle("Period x seniority centroid similarity")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "centroid_similarity_heatmaps.png", dpi=150)
    plt.close(fig)


def plot_pca(meta: pd.DataFrame, x: np.ndarray) -> None:
    coords = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(x)
    plot_2d_map(meta, coords, FIG_DIR / "pca_structural_map.png", "PCA")


def plot_umap(meta: pd.DataFrame, x: np.ndarray) -> None:
    try:
        import umap
    except Exception as exc:  # pragma: no cover - environment diagnostic
        pd.DataFrame([{"figure": "umap_structural_map.png", "skipped_reason": str(exc)}]).to_csv(
            TABLE_DIR / "umap_skip_reason.csv", index=False
        )
        return
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.15,
        metric="cosine",
        random_state=RANDOM_SEED,
        low_memory=True,
        n_jobs=1,
    )
    coords = reducer.fit_transform(x)
    plot_2d_map(meta, coords, FIG_DIR / "umap_structural_map.png", "UMAP")


def plot_2d_map(meta: pd.DataFrame, coords: np.ndarray, path: Path, method_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    groups = meta["year_seniority"].astype(str)
    colors = plt.cm.tab10(np.linspace(0, 1, groups.nunique()))
    color_map = dict(zip(sorted(groups.unique()), colors, strict=False))
    for group in sorted(groups.unique()):
        idx = groups.eq(group).to_numpy()
        ax.scatter(coords[idx, 0], coords[idx, 1], s=5, alpha=0.28, color=color_map[group], label=group)
    for year, color in [("2024", "#222222"), ("2026", "#777777")]:
        idx = meta["year"].eq(year).to_numpy()
        pts = coords[idx]
        if len(pts) >= 100:
            draw_density_contour(ax, pts, color=color, alpha=0.45)
    for seniority in ["junior", "mid", "senior", "unknown"]:
        c2024 = coords[(meta["year"].eq("2024") & meta["seniority_3level"].eq(seniority)).to_numpy()]
        c2026 = coords[(meta["year"].eq("2026") & meta["seniority_3level"].eq(seniority)).to_numpy()]
        if len(c2024) and len(c2026):
            a = c2024.mean(axis=0)
            b = c2026.mean(axis=0)
            ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#111111"})
            ax.text(b[0], b[1], seniority, fontsize=8, weight="bold")
    ax.set_title(f"{method_name}: period x seniority")
    ax.legend(markerscale=3, fontsize=7, frameon=False, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    arch = meta["archetype_name"].astype(str)
    top_arch = arch.value_counts().head(8).index.tolist()
    arch_plot = arch.where(arch.isin(top_arch), "Other/Unlabeled")
    colors = plt.cm.tab20(np.linspace(0, 1, arch_plot.nunique()))
    color_map = dict(zip(sorted(arch_plot.unique()), colors, strict=False))
    for group in sorted(arch_plot.unique()):
        idx = arch_plot.eq(group).to_numpy()
        ax.scatter(coords[idx, 0], coords[idx, 1], s=5, alpha=0.28, color=color_map[group], label=group)
    ax.set_title(f"{method_name}: archetype/domain")
    ax.legend(markerscale=3, fontsize=7, frameon=False, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_density_contour(ax: plt.Axes, pts: np.ndarray, color: str, alpha: float = 0.4) -> None:
    if len(pts) > 2500:
        rng = np.random.default_rng(RANDOM_SEED)
        pts = pts[rng.choice(len(pts), size=2500, replace=False)]
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    if xmax <= xmin or ymax <= ymin:
        return
    xpad = (xmax - xmin) * 0.05
    ypad = (ymax - ymin) * 0.05
    xx, yy = np.mgrid[xmin - xpad : xmax + xpad : 80j, ymin - ypad : ymax + ypad : 80j]
    try:
        z = gaussian_kde(pts.T)(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    except Exception:
        return
    levels = np.quantile(z[z > 0], [0.70, 0.86, 0.95]) if np.any(z > 0) else []
    if len(levels):
        ax.contour(xx, yy, z, levels=np.unique(levels), colors=[color], linewidths=0.8, alpha=alpha)


def plot_nn(nn: pd.DataFrame) -> None:
    senior = nn.loc[nn["neighbor_seniority"].eq("senior")].copy()
    if senior.empty:
        return
    pivot = senior.pivot(index="query_definition", columns="representation", values="excess_over_base_pp").loc[JUNIOR_DEFS]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(pivot.index))
    width = 0.35
    ax.bar(x - width / 2, pivot.get("embedding", pd.Series(index=pivot.index, dtype=float)), width, label="embedding")
    ax.bar(x + width / 2, pivot.get("tfidf", pd.Series(index=pivot.index, dtype=float)), width, label="TF-IDF")
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(x, labels=pivot.index)
    ax.set_ylabel("Senior-neighbor excess over 2024 base rate (pp)")
    ax.set_title("2026 junior-definition nearest neighbors in 2024")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "nearest_neighbor_excess.png", dpi=150)
    plt.close(fig)


def write_outputs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pd.read_csv(T30_PANEL).to_csv(TABLE_DIR / "t30_panel_loaded_for_reference.csv", index=False)

    coverage = load_text_source_coverage()
    coverage.to_csv(TABLE_DIR / "text_source_coverage.csv", index=False)

    meta_all = load_embedding_metadata(llm_only=True)
    sample = primary_sample(meta_all)
    sample["year_seniority"] = sample["year"] + "_" + sample["seniority_3level"].astype(str)
    meta_all["year_seniority"] = meta_all["year"] + "_" + meta_all["seniority_3level"].astype(str)

    sample_comp = (
        sample.groupby(["source", "period", "year", "seniority_3level", "text_source"], dropna=False)
        .size()
        .reset_index(name="sample_n")
    )
    eligible_comp = (
        meta_all.groupby(["source", "period", "year", "seniority_3level", "text_source"], dropna=False)
        .size()
        .reset_index(name="eligible_n")
    )
    sample_comp = sample_comp.merge(
        eligible_comp,
        on=["source", "period", "year", "seniority_3level", "text_source"],
        how="left",
    )
    sample_comp["sample_share_of_eligible"] = sample_comp["sample_n"] / sample_comp["eligible_n"]
    sample_comp.to_csv(TABLE_DIR / "sample_composition.csv", index=False)
    sample.drop(columns=["description_cleaned"]).to_csv(TABLE_DIR / "sample_index.csv", index=False)

    x_emb = load_embeddings(sample["embedding_row"].to_numpy())
    x_tfidf, _vec, _svd = tfidf_svd(sample["description_cleaned"])

    centroid_rows_emb, centroids_emb = group_centroids(sample, x_emb, "year_seniority")
    centroid_rows_tfidf, centroids_tfidf = group_centroids(sample, x_tfidf, "year_seniority")
    centroid_rows_emb["representation"] = "embedding"
    centroid_rows_tfidf["representation"] = "tfidf_svd"
    pd.concat([centroid_rows_emb, centroid_rows_tfidf], ignore_index=True).to_csv(
        TABLE_DIR / "centroid_group_stats.csv", index=False
    )

    sim_emb = centroid_similarity_table(centroids_emb)
    sim_tfidf = centroid_similarity_table(centroids_tfidf)
    sim_emb["representation"] = "embedding"
    sim_tfidf["representation"] = "tfidf_svd"
    pd.concat([sim_emb, sim_tfidf], ignore_index=True).to_csv(TABLE_DIR / "centroid_similarity_matrix.csv", index=False)

    conv_parts = []
    for spec in ["primary_all", "no_aggregators", "company_cap50", "recommended_swe_tier"]:
        conv_parts.append(convergence_seniority(sample, x_emb, "embedding", spec))
        conv_parts.append(convergence_seniority(sample, x_tfidf, "tfidf_svd", spec))
    convergence = pd.concat(conv_parts, ignore_index=True)
    convergence.to_csv(TABLE_DIR / "convergence_seniority_3level.csv", index=False)
    convergence.to_csv(TABLE_DIR / "sensitivity_summary.csv", index=False)

    t30 = pd.concat(
        [
            t30_convergence(sample, x_emb, "embedding"),
            t30_convergence(sample, x_tfidf, "tfidf_svd"),
        ],
        ignore_index=True,
    )
    t30.to_csv(TABLE_DIR / "t30_convergence_panel.csv", index=False)

    arch = pd.concat(
        [
            archetype_convergence(sample, x_emb, "embedding"),
            archetype_convergence(sample, x_tfidf, "tfidf_svd"),
        ],
        ignore_index=True,
    )
    arch.to_csv(TABLE_DIR / "archetype_convergence.csv", index=False)

    dispersion = pd.concat(
        [
            pairwise_dispersion(sample, x_emb, "year_seniority", "embedding"),
            pairwise_dispersion(sample, x_tfidf, "year_seniority", "tfidf_svd"),
        ],
        ignore_index=True,
    )
    dispersion.to_csv(TABLE_DIR / "within_group_dispersion.csv", index=False)

    variance = pd.concat(
        [
            variance_explained(sample, x_emb, "embedding"),
            variance_explained(sample, x_tfidf, "tfidf_svd"),
        ],
        ignore_index=True,
    )
    variance.to_csv(TABLE_DIR / "variance_explained_by_factor.csv", index=False)

    nn = nearest_neighbor_analysis(meta_all)
    nn.to_csv(TABLE_DIR / "nearest_neighbor_analysis.csv", index=False)

    out = pd.concat(
        [
            outliers(sample, x_emb, "embedding"),
            outliers(sample, x_tfidf, "tfidf_svd"),
        ],
        ignore_index=True,
    )
    out.to_csv(TABLE_DIR / "semantic_outliers_top_by_group.csv", index=False)

    raw_sens = raw_text_tfidf_sensitivity()
    raw_sens.to_csv(TABLE_DIR / "description_text_source_sensitivity_raw_tfidf.csv", index=False)

    robustness_rows = []
    for rep, table in [("embedding", convergence), ("tfidf_svd", convergence)]:
        sub = table.loc[table["representation"].eq(rep) & table["spec"].eq("primary_all")]
        verdict = sub["calibration_verdict"].iloc[0] if len(sub) else "missing"
        shift = sub["arshkon_to_scraped_shift"].iloc[0] if len(sub) else np.nan
        robustness_rows.append(
            {
                "finding": "junior_senior_convergence_seniority_3level",
                "representation": rep,
                "effect": shift,
                "verdict": verdict,
            }
        )
    for rep in ["embedding", "tfidf"]:
        senior_nn = nn.loc[(nn["representation"].eq(rep)) & (nn["query_definition"].eq("J1")) & (nn["neighbor_seniority"].eq("senior"))]
        if not senior_nn.empty:
            robustness_rows.append(
                {
                    "finding": "J1_2026_neighbors_senior_excess",
                    "representation": rep,
                    "effect": senior_nn["excess_over_base_pp"].iloc[0],
                    "verdict": "positive_excess" if senior_nn["excess_over_base_pp"].iloc[0] > 0 else "no_positive_excess",
                }
            )
    variance_top = variance.sort_values(["representation", "eta_squared"], ascending=[True, False]).groupby("representation").head(1)
    for _, row in variance_top.iterrows():
        robustness_rows.append(
            {
                "finding": "dominant_structure_factor",
                "representation": row["representation"],
                "effect": row["eta_squared"],
                "verdict": row["factor"],
            }
        )
    pd.DataFrame(robustness_rows).to_csv(TABLE_DIR / "representation_robustness_summary.csv", index=False)

    plot_similarity_heatmaps(sim_emb, sim_tfidf)
    plot_pca(sample, x_emb)
    plot_umap(sample, x_emb)
    plot_nn(nn)

    print("T15 complete")
    print(f"eligible_llm_rows: {len(meta_all):,}")
    print(f"sample_rows: {len(sample):,}")
    print(f"tfidf_components: {x_tfidf.shape[1]}")


if __name__ == "__main__":
    write_outputs()
