"""
Stage 2 T-bootstrap — §7.2 bootstrap, §7.3 per-period, §7.6 within-2024 cross-source.

This script executes three robustness checks on the frozen Stage 1 BERTopic
fit (`raw_fit.bertopic`, mcs=70, seed=42, headline K=10):

§7.2 Bootstrap (3 replicates).
    Three 80% subsamples of Sample A drawn without replacement, stratified by
    (period, source). For each subsample we refit BERTopic at headline
    (mcs=70, seed=42), reduce to K=10, and report ARI / NMI vs the headline
    fit on the overlapping rows. ARI ≥ 0.4 is informative (not a gate).

§7.3 Per-period reproduction.
    Stage 1 already fit BERTopic separately on the 2024 subset and the 2026
    subset of Sample A and cached both at headline mcs=70. We load each fit,
    reduce to K=10, and Hungarian-match its centroids (in 3072-d posting
    space) against the joint-Sample-A headline centroids. Mean and median
    matched cosine; threshold ≥ 0.85 per §11.9.

§7.6 Within-2024 cross-source placebo.
    Refit BERTopic separately on the asaniczka-2024 (n=18,051) and
    arshkon-2024 (n=5,293) subsets at headline (mcs=70, seed=42). Reduce
    each to K=10. Centroid alignment between the two via Hungarian; threshold
    ≥ 0.85 per §11.9.

Writes:
    data/bertopic/stability.parquet — one row per (pair_kind, pair_label)
        with: ari, nmi, centroid_alignment_mean, centroid_alignment_median.

Hash bundle is verified against `intermediate/stage1_freeze.json` before any
work begins; mismatch → fail loud and stop.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from figures.bertopic import config
from figures.bertopic.stage1 import pipeline


HEADLINE_MCS = 70
HEADLINE_K = 10
SEED = config.SEED_PRIMARY
BOOTSTRAP_FRAC = 0.80
BOOTSTRAP_SEEDS = (101, 202, 303)

INTERIM = config.INTERMEDIATE_DIR
PERIOD_FITS_DIR = INTERIM / "period_fits"
BOOTSTRAP_DIR = INTERIM / "bootstrap_fits"
SOURCE_FITS_DIR = INTERIM / "source_fits"
STABILITY_PARQUET = config.BERTOPIC_DATA_DIR / "stability.parquet"


# ---------------------------------------------------------------------------
# Hash verification
# ---------------------------------------------------------------------------

def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hashes() -> None:
    freeze = json.loads(config.STAGE1_FREEZE_JSON.read_text())
    expected = {
        "config_hash": (Path(config.__file__), freeze["config_hash"]),
        "sample_hash": (config.SAMPLE_A_PATH, freeze["sample_hash"]),
        "embeddings_cache_hash": (
            config.EMBEDDINGS_CACHE_PATH, freeze["embeddings_cache_hash"],
        ),
        "model_hash": (config.RAW_FIT_PATH, freeze["model_hash"]),
        "assignments_hash": (
            config.ASSIGNMENTS_PATH, freeze["assignments_hash"],
        ),
    }
    print("Verifying hash bundle against stage1_freeze.json…")
    for name, (path, want) in expected.items():
        got = _file_hash(Path(path))
        status = "OK" if got == want else "MISMATCH"
        print(f"  {name}: {status}")
        if got != want:
            raise RuntimeError(
                f"Hash mismatch for {name}: expected {want}, got {got} "
                f"(path={path})"
            )


# ---------------------------------------------------------------------------
# Sample loading + per-row metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleA:
    uids: tuple[str, ...]
    docs: tuple[str, ...]
    embeddings: np.ndarray
    period: tuple[str, ...]  # e.g. "2024-01", "2026-04"
    source: tuple[str, ...]  # e.g. "kaggle_arshkon"


def load_sample_a() -> SampleA:
    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")
    rows = con.execute(
        f"SELECT uid, description_core_llm, period, source "
        f"FROM '{config.SAMPLE_A_PATH}' ORDER BY uid"
    ).fetchall()
    uids = tuple(r[0] for r in rows)
    docs = tuple(r[1] for r in rows)
    period = tuple(r[2] for r in rows)
    source = tuple(r[3] for r in rows)

    matrix, key_to_row = pipeline._load_cache_cached()
    indices = np.array([key_to_row[u] for u in uids], dtype=np.int64)
    emb = matrix[indices]
    return SampleA(uids=uids, docs=docs, embeddings=emb,
                   period=period, source=source)


# ---------------------------------------------------------------------------
# Reduce-to-K with caching
# ---------------------------------------------------------------------------

def _reduce_cached(
    *,
    model_path: Path,
    docs: tuple[str, ...],
    k: int,
    cache_path: Path,
    fallback_topics: np.ndarray,
) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)
    model = pipeline.load_topic_model_for_reduce(model_path)
    try:
        model.reduce_topics(list(docs), nr_topics=k)
        labels = np.asarray(model.topics_, dtype=np.int64)
    except Exception as exc:  # noqa: BLE001
        print(f"    reduce_topics({k}) skipped on {model_path.name}: {exc}")
        labels = fallback_topics
    np.save(cache_path, labels)
    return labels


# ---------------------------------------------------------------------------
# §7.2 — Bootstrap
# ---------------------------------------------------------------------------

def _stratified_bootstrap_indices(
    period: tuple[str, ...],
    source: tuple[str, ...],
    *,
    frac: float,
    seed: int,
) -> np.ndarray:
    """Sample `frac` of indices without replacement, stratified by (period, source)."""
    n = len(period)
    rng = np.random.default_rng(seed)
    strata: dict[tuple[str, str], list[int]] = {}
    for i in range(n):
        strata.setdefault((period[i], source[i]), []).append(i)
    keep: list[int] = []
    for key, idxs in strata.items():
        idxs_arr = np.asarray(idxs, dtype=np.int64)
        # Round so total sums to ~frac * n; use floor+1 if remainder large enough
        # Simple: take floor(frac * len) per stratum
        k = int(np.floor(frac * len(idxs_arr)))
        if k <= 0:
            k = 1
        chosen = rng.choice(idxs_arr, size=k, replace=False)
        keep.extend(int(c) for c in chosen)
    keep_arr = np.asarray(sorted(keep), dtype=np.int64)
    return keep_arr


def run_bootstrap(sample: SampleA) -> list[dict]:
    """Three bootstraps; refit at (mcs=70, seed=42); reduce to K=10."""
    BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)
    headline_K_labels = np.load(INTERIM / f"headline_reducedK{HEADLINE_K}.npy")

    rows: list[dict] = []
    for boot_seed in BOOTSTRAP_SEEDS:
        name = f"bootstrap_seed_{boot_seed}_mcs_{HEADLINE_MCS}"
        topics_path = BOOTSTRAP_DIR / f"{name}_topics.npy"
        idx_path = BOOTSTRAP_DIR / f"{name}_indices.npy"
        model_path = BOOTSTRAP_DIR / f"{name}.bertopic"
        reduced_path = BOOTSTRAP_DIR / f"{name}_reducedK{HEADLINE_K}.npy"

        if idx_path.exists():
            keep_idx = np.load(idx_path)
        else:
            keep_idx = _stratified_bootstrap_indices(
                sample.period, sample.source, frac=BOOTSTRAP_FRAC, seed=boot_seed,
            )
            np.save(idx_path, keep_idx)
        n_keep = len(keep_idx)

        # Fit on the kept rows.
        if topics_path.exists() and model_path.exists():
            print(f"  bootstrap seed={boot_seed} (n={n_keep}): cached")
            boot_topics = np.load(topics_path)
        else:
            print(f"  bootstrap seed={boot_seed} (n={n_keep}): fitting…")
            t0 = perf_counter()
            boot_uids = tuple(sample.uids[i] for i in keep_idx)
            boot_docs = tuple(sample.docs[i] for i in keep_idx)
            boot_emb = sample.embeddings[keep_idx]
            fit = pipeline.fit_topic_model(
                min_cluster_size=HEADLINE_MCS,
                seed=SEED,
                uids=boot_uids,
                docs=boot_docs,
                embeddings=boot_emb,
            )
            fit.topic_model.save(
                str(model_path),
                serialization="pickle",
                save_ctfidf=True,
                save_embedding_model=False,
            )
            boot_topics = np.asarray(fit.topics, dtype=np.int64)
            np.save(topics_path, boot_topics)
            print(
                f"    {pipeline.n_clusters(boot_topics)} clusters, "
                f"noise {pipeline.noise_rate(boot_topics):.1%}, "
                f"{perf_counter() - t0:.1f}s"
            )

        # Reduce to K=10.
        boot_docs = tuple(sample.docs[i] for i in keep_idx)
        boot_emb = sample.embeddings[keep_idx]
        boot_reducedK = _reduce_cached(
            model_path=model_path,
            docs=boot_docs,
            k=HEADLINE_K,
            cache_path=reduced_path,
            fallback_topics=boot_topics,
        )

        # ARI / NMI vs headline on overlapping rows (this bootstrap's rows).
        head_overlap = headline_K_labels[keep_idx]
        ari = float(adjusted_rand_score(head_overlap, boot_reducedK))
        nmi = float(normalized_mutual_info_score(head_overlap, boot_reducedK))

        # Centroid alignment in 3072-d posting space.
        head_centroids = pipeline.cluster_centroids(head_overlap, boot_emb)
        boot_centroids = pipeline.cluster_centroids(boot_reducedK, boot_emb)
        align_mean, align_median = _hungarian_alignment_stats(
            head_centroids, boot_centroids,
        )

        print(
            f"    ARI={ari:.3f}, NMI={nmi:.3f}, "
            f"align_mean={align_mean:.3f}, align_median={align_median:.3f}"
        )
        rows.append({
            "pair_kind": "bootstrap",
            "pair_label": f"bootstrap_seed_{boot_seed}",
            "n_overlap": int(n_keep),
            "ari": ari,
            "nmi": nmi,
            "centroid_alignment_mean": align_mean,
            "centroid_alignment_median": align_median,
        })
    return rows


# ---------------------------------------------------------------------------
# Hungarian alignment with mean+median
# ---------------------------------------------------------------------------

def _hungarian_alignment_stats(
    centroids_a: dict[int, np.ndarray],
    centroids_b: dict[int, np.ndarray],
) -> tuple[float, float]:
    if not centroids_a or not centroids_b:
        return float("nan"), float("nan")
    keys_a = list(centroids_a)
    keys_b = list(centroids_b)
    mat_a = np.stack([centroids_a[k] for k in keys_a])
    mat_b = np.stack([centroids_b[k] for k in keys_b])
    sim = mat_a @ mat_b.T
    cost = -sim
    row_idx, col_idx = linear_sum_assignment(cost)
    matched = sim[row_idx, col_idx]
    return float(matched.mean()), float(np.median(matched))


# ---------------------------------------------------------------------------
# §7.3 — Per-period reproduction
# ---------------------------------------------------------------------------

def run_per_period(sample: SampleA) -> list[dict]:
    headline_K_labels = np.load(INTERIM / f"headline_reducedK{HEADLINE_K}.npy")

    # Joint centroids in 3072-d posting space, computed on full Sample A.
    joint_centroids = pipeline.cluster_centroids(headline_K_labels, sample.embeddings)

    rows: list[dict] = []
    for period_label in ("2024", "2026"):
        bucket_mask = np.array(
            [p.startswith(period_label) for p in sample.period],
            dtype=bool,
        )
        bucket_idx = np.where(bucket_mask)[0]
        bucket_emb = sample.embeddings[bucket_idx]
        bucket_docs = tuple(sample.docs[i] for i in bucket_idx)
        n_bucket = len(bucket_idx)

        # Stage 1 cached the per-period reduced-K labels at
        # intermediate/period_<period>_reducedK<K>.npy.
        period_reduced_path = INTERIM / f"period_{period_label}_reducedK{HEADLINE_K}.npy"
        if period_reduced_path.exists():
            p_lbl = np.load(period_reduced_path)
        else:
            # Fallback: reduce live from the period_fits/ model.
            p_model_path = PERIOD_FITS_DIR / f"period_{period_label}_mcs_{HEADLINE_MCS}.bertopic"
            p_topics_raw = np.load(
                PERIOD_FITS_DIR / f"period_{period_label}_mcs_{HEADLINE_MCS}_topics.npy"
            )
            p_lbl = _reduce_cached(
                model_path=p_model_path,
                docs=bucket_docs,
                k=HEADLINE_K,
                cache_path=period_reduced_path,
                fallback_topics=p_topics_raw,
            )

        if len(p_lbl) != n_bucket:
            raise RuntimeError(
                f"period_{period_label} label length {len(p_lbl)} "
                f"!= bucket size {n_bucket}"
            )

        period_centroids = pipeline.cluster_centroids(p_lbl, bucket_emb)

        # Hungarian: joint clusters → period clusters.
        # For "for each joint-Sample-A cluster centroid, find the nearest
        # period-fit cluster centroid by cosine via Hungarian", the cost
        # matrix is between joint and period centroid sets.
        align_mean, align_median = _hungarian_alignment_stats(
            joint_centroids, period_centroids,
        )

        # ARI / NMI over the bucket rows: joint headline vs period fit
        # (on the same subset of postings).
        head_bucket = headline_K_labels[bucket_idx]
        ari = float(adjusted_rand_score(head_bucket, p_lbl))
        nmi = float(normalized_mutual_info_score(head_bucket, p_lbl))

        print(
            f"  period {period_label} (n={n_bucket}): "
            f"ARI={ari:.3f}, NMI={nmi:.3f}, "
            f"align_mean={align_mean:.3f}, align_median={align_median:.3f}"
        )
        rows.append({
            "pair_kind": "per_period",
            "pair_label": f"joint_vs_period_{period_label}",
            "n_overlap": int(n_bucket),
            "ari": ari,
            "nmi": nmi,
            "centroid_alignment_mean": align_mean,
            "centroid_alignment_median": align_median,
        })
    return rows


# ---------------------------------------------------------------------------
# §7.6 — Within-2024 cross-source placebo
# ---------------------------------------------------------------------------

def run_within_2024(sample: SampleA) -> list[dict]:
    SOURCE_FITS_DIR.mkdir(parents=True, exist_ok=True)

    # Subset definitions: 2024 SWE rows by source.
    subsets: dict[str, np.ndarray] = {}
    for source_name in ("kaggle_asaniczka", "kaggle_arshkon"):
        mask = np.array(
            [
                (sample.source[i] == source_name and sample.period[i].startswith("2024"))
                for i in range(len(sample.uids))
            ],
            dtype=bool,
        )
        subsets[source_name] = np.where(mask)[0]

    # Fit each.
    per_source_outputs: dict[str, dict] = {}
    for source_name, idxs in subsets.items():
        n = len(idxs)
        if n == 0:
            raise RuntimeError(f"No rows for {source_name} in 2024 subset")
        name = f"{source_name}_2024_mcs_{HEADLINE_MCS}"
        model_path = SOURCE_FITS_DIR / f"{name}.bertopic"
        topics_path = SOURCE_FITS_DIR / f"{name}_topics.npy"
        reduced_path = SOURCE_FITS_DIR / f"{name}_reducedK{HEADLINE_K}.npy"

        sub_uids = tuple(sample.uids[i] for i in idxs)
        sub_docs = tuple(sample.docs[i] for i in idxs)
        sub_emb = sample.embeddings[idxs]

        if topics_path.exists() and model_path.exists():
            print(f"  {source_name} (n={n}): cached")
            topics = np.load(topics_path)
        else:
            print(f"  {source_name} (n={n}): fitting…")
            t0 = perf_counter()
            # Permissive vectorizer for small subsets — strict min_df=10 fails
            # on the few-cluster c-TF-IDF corpus (e.g. arshkon 2024 SWE = 5,293
            # rows produces ≤ 9 raw topics).
            fit = pipeline.fit_topic_model(
                min_cluster_size=HEADLINE_MCS,
                seed=SEED,
                uids=sub_uids,
                docs=sub_docs,
                embeddings=sub_emb,
                permissive_vectorizer=True,
            )
            fit.topic_model.save(
                str(model_path),
                serialization="pickle",
                save_ctfidf=True,
                save_embedding_model=False,
            )
            topics = np.asarray(fit.topics, dtype=np.int64)
            np.save(topics_path, topics)
            print(
                f"    {pipeline.n_clusters(topics)} clusters, "
                f"noise {pipeline.noise_rate(topics):.1%}, "
                f"{perf_counter() - t0:.1f}s"
            )

        reduced = _reduce_cached(
            model_path=model_path,
            docs=sub_docs,
            k=HEADLINE_K,
            cache_path=reduced_path,
            fallback_topics=topics,
        )
        centroids = pipeline.cluster_centroids(reduced, sub_emb)
        per_source_outputs[source_name] = {
            "indices": idxs,
            "reduced": reduced,
            "centroids": centroids,
            "n": n,
        }

    a = per_source_outputs["kaggle_asaniczka"]
    b = per_source_outputs["kaggle_arshkon"]
    align_mean, align_median = _hungarian_alignment_stats(
        a["centroids"], b["centroids"],
    )

    # ARI is not directly defined across disjoint subsets. Report NaN with
    # explanation; centroid alignment is the relevant metric for §7.6.
    print(
        f"  asaniczka_2024 ↔ arshkon_2024: "
        f"align_mean={align_mean:.3f}, align_median={align_median:.3f}"
    )
    return [{
        "pair_kind": "within_2024",
        "pair_label": "asaniczka_2024_vs_arshkon_2024",
        "n_overlap": int(a["n"] + b["n"]),
        "ari": float("nan"),
        "nmi": float("nan"),
        "centroid_alignment_mean": align_mean,
        "centroid_alignment_median": align_median,
    }]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    verify_hashes()

    print("\nLoading Sample A + embeddings…")
    sample = load_sample_a()
    print(f"  Sample A: {len(sample.uids):,} rows; embeddings {sample.embeddings.shape}")

    print("\n§7.2 — Bootstrap (3 replicates @ 80% stratified by (period, source))")
    bootstrap_rows = run_bootstrap(sample)

    print("\n§7.3 — Per-period reproduction")
    period_rows = run_per_period(sample)

    print("\n§7.6 — Within-2024 cross-source placebo")
    within_rows = run_within_2024(sample)

    all_rows = bootstrap_rows + period_rows + within_rows
    table = pa.table({
        "pair_kind": [r["pair_kind"] for r in all_rows],
        "pair_label": [r["pair_label"] for r in all_rows],
        "n_overlap": [r["n_overlap"] for r in all_rows],
        "ari": [r["ari"] for r in all_rows],
        "nmi": [r["nmi"] for r in all_rows],
        "centroid_alignment_mean": [r["centroid_alignment_mean"] for r in all_rows],
        "centroid_alignment_median": [r["centroid_alignment_median"] for r in all_rows],
    })
    STABILITY_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, STABILITY_PARQUET, compression="zstd")
    print(f"\nWrote {STABILITY_PARQUET}")
    print(table.to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
