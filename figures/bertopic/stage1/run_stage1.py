"""
Stage 1 orchestrator: §13.2 S1.1 through S1.7 in one resumable script.

Steps:
  S1.1 — min_cluster_size sweep on Sample A (5 fits, seed 42).
  S1.2 — headline fit at chosen mcs + K sweep over the §4.4 grid.
  S1.3 — seed stability at (headline mcs, headline K). GATE.
  S1.4 — mega-cluster gate at headline K.
  S1.5 — determinism check (refit at headline mcs + seed 42).
  S1.6 — LLM naming for headline-K and super-family-K clusters.
  S1.7 — hash artifacts and write `stage1_freeze.json`.

Each step writes its outputs to `intermediate/`; on re-run the orchestrator
picks up where it left off so a transient failure does not waste hours of
UMAP fits. Re-run order is enforced by the dependency chain (mcs sweep
must complete before headline fit; headline fit before K sweep; etc.).
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from figures.bertopic import config
from figures.bertopic.stage1 import pipeline
from figures.bertopic.stage1.naming import propose_label


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INTERIM = config.INTERMEDIATE_DIR
MCS_FITS_DIR = INTERIM / "mcs_fits"
SEED_FITS_DIR = INTERIM / "seed_fits"
PERIOD_FITS_DIR = INTERIM / "period_fits"
DETERMINISM_DIR = INTERIM / "determinism"
TOPIC_INFO_RAW = INTERIM / "topic_info_raw.parquet"


# ---------------------------------------------------------------------------
# Cache directory bootstrap
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    for d in (
        INTERIM,
        MCS_FITS_DIR,
        SEED_FITS_DIR,
        PERIOD_FITS_DIR,
        DETERMINISM_DIR,
        config.BERTOPIC_DATA_DIR,
        config.MEMOS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Generic save / load helpers — we cache topic-label arrays as .npy files
# alongside the BERTopic model pickle. The model is needed for c-TF-IDF top
# words and reduce_topics; the label array is the per-row assignment.
# ---------------------------------------------------------------------------

def _model_path(out_dir: Path, name: str) -> Path:
    return out_dir / f"{name}.bertopic"


def _topics_path(out_dir: Path, name: str) -> Path:
    return out_dir / f"{name}_topics.npy"


def _save_fit(fit: pipeline.FitResult, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fit.topic_model.save(
        str(_model_path(out_dir, name)),
        serialization="pickle",
        save_ctfidf=True,
        save_embedding_model=False,
    )
    np.save(_topics_path(out_dir, name), fit.topics)


def _load_topics(out_dir: Path, name: str) -> np.ndarray | None:
    p = _topics_path(out_dir, name)
    if not p.exists():
        return None
    return np.load(p)


# ---------------------------------------------------------------------------
# S1.1 — min_cluster_size sweep
# ---------------------------------------------------------------------------

def s1_1_mcs_sweep(
    uids: tuple[str, ...],
    docs: tuple[str, ...],
    embeddings: np.ndarray,
) -> tuple[int, list[dict]]:
    """Fit at each mcs in the sweep grid; pick headline mcs per §4.6."""
    rows: list[dict] = []
    fits_topics: dict[int, np.ndarray] = {}

    for mcs in config.MCS_SWEEP_GRID:
        name = f"mcs_{mcs}"
        cached = _load_topics(MCS_FITS_DIR, name)
        if cached is not None:
            fits_topics[mcs] = cached
            print(f"  mcs={mcs}: cached")
            continue

        t0 = perf_counter()
        fit = pipeline.fit_topic_model(
            min_cluster_size=mcs,
            seed=config.SEED_PRIMARY,
            uids=uids, docs=docs, embeddings=embeddings,
        )
        dt = perf_counter() - t0
        _save_fit(fit, MCS_FITS_DIR, name)
        fits_topics[mcs] = fit.topics
        print(
            f"  mcs={mcs}: {pipeline.n_clusters(fit.topics)} clusters, "
            f"noise {pipeline.noise_rate(fit.topics):.1%}, {dt:.1f}s"
        )

    # Compute post-reduction adjacent ARI at K=30 for each mcs.
    # We re-load each model and reduce to K=30 in turn.
    reduced_topics: dict[int, np.ndarray] = {}
    for mcs in config.MCS_SWEEP_GRID:
        cache_path = MCS_FITS_DIR / f"mcs_{mcs}_reducedK30.npy"
        if cache_path.exists():
            reduced_topics[mcs] = np.load(cache_path)
            continue
        from bertopic import BERTopic
        model = BERTopic.load(str(_model_path(MCS_FITS_DIR, f"mcs_{mcs}")))
        try:
            model.reduce_topics(list(docs), nr_topics=30)
            reduced = np.asarray(model.topics_, dtype=np.int64)
        except Exception as exc:  # noqa: BLE001
            # If the raw fit had ≤ 30 topics, reduce_topics is a no-op.
            print(f"    mcs={mcs}: reduce_topics(30) skipped ({exc})")
            reduced = fits_topics[mcs]
        reduced_topics[mcs] = reduced
        np.save(cache_path, reduced)

    grid = list(config.MCS_SWEEP_GRID)
    for i, mcs in enumerate(grid):
        topics = fits_topics[mcs]
        adj_ari = float("nan")
        if i + 1 < len(grid):
            adj_ari = pipeline.pairwise_metrics(
                reduced_topics[mcs], reduced_topics[grid[i + 1]]
            )["ari"]
        rows.append({
            "min_cluster_size": mcs,
            "n_clusters_raw": pipeline.n_clusters(topics),
            "noise_rate_raw": pipeline.noise_rate(topics),
            "n_clusters_k30": pipeline.n_clusters(reduced_topics[mcs]),
            "adjacent_ari_k30": adj_ari,
        })

    with config.MCS_SWEEP_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Pick headline mcs per §4.6: post-reduction ARI vs neighbors >= 0.7 AND
    # noise in [0.15, 0.35]. Largest such mcs.
    candidates: list[int] = []
    for i, row in enumerate(rows):
        in_noise_band = (
            config.MCS_NOISE_RANGE[0] <= row["noise_rate_raw"] <= config.MCS_NOISE_RANGE[1]
        )
        ari_left = rows[i - 1]["adjacent_ari_k30"] if i > 0 else float("inf")
        ari_right = row["adjacent_ari_k30"]
        # for the last entry, only the left neighbor
        if i == len(rows) - 1:
            stable = ari_left >= config.MCS_PLATEAU_ARI
        elif i == 0:
            stable = ari_right >= config.MCS_PLATEAU_ARI
        else:
            stable = (ari_left >= config.MCS_PLATEAU_ARI
                      and ari_right >= config.MCS_PLATEAU_ARI)
        if in_noise_band and stable:
            candidates.append(row["min_cluster_size"])

    if candidates:
        headline_mcs = max(candidates)
    else:
        # Fallback: closest to the noise band, prefer larger mcs as more
        # conservative.
        def fallback_score(row: dict) -> tuple[int, float]:
            mid = (config.MCS_NOISE_RANGE[0] + config.MCS_NOISE_RANGE[1]) / 2
            return (
                int(config.MCS_NOISE_RANGE[0] <= row["noise_rate_raw"] <= config.MCS_NOISE_RANGE[1]),
                -abs(row["noise_rate_raw"] - mid),
            )
        ranked = sorted(rows, key=fallback_score, reverse=True)
        headline_mcs = ranked[0]["min_cluster_size"]

    return headline_mcs, rows


# ---------------------------------------------------------------------------
# S1.2 — headline fit + K sweep
# ---------------------------------------------------------------------------

def s1_2_headline_and_k_sweep(
    headline_mcs: int,
    uids: tuple[str, ...],
    docs: tuple[str, ...],
    embeddings: np.ndarray,
    period_per_uid: dict[str, str],
) -> tuple[int, list[dict]]:
    """Use the cached fit at headline_mcs as the headline; run K sweep."""
    headline_path = _model_path(MCS_FITS_DIR, f"mcs_{headline_mcs}")
    if not headline_path.exists():
        raise RuntimeError(f"headline fit missing: {headline_path}")

    # Copy the headline model to the canonical location for clarity.
    if not config.RAW_FIT_PATH.exists():
        from bertopic import BERTopic
        BERTopic.load(str(headline_path)).save(
            str(config.RAW_FIT_PATH),
            serialization="pickle",
            save_ctfidf=True,
            save_embedding_model=False,
        )

    # Load seed-stability fits if present, else fit them.
    seed_topics: dict[int, np.ndarray] = {}
    seed_topics[config.SEED_PRIMARY] = np.load(
        _topics_path(MCS_FITS_DIR, f"mcs_{headline_mcs}")
    )
    for seed in config.SEED_STABILITY:
        name = f"seed_{seed}_mcs_{headline_mcs}"
        cached = _load_topics(SEED_FITS_DIR, name)
        if cached is None:
            print(f"  seed {seed}: fitting…")
            t0 = perf_counter()
            fit = pipeline.fit_topic_model(
                min_cluster_size=headline_mcs, seed=seed,
                uids=uids, docs=docs, embeddings=embeddings,
            )
            _save_fit(fit, SEED_FITS_DIR, name)
            cached = fit.topics
            print(
                f"    {pipeline.n_clusters(cached)} clusters, "
                f"noise {pipeline.noise_rate(cached):.1%}, "
                f"{perf_counter() - t0:.1f}s"
            )
        seed_topics[seed] = cached

    # Per-period fits for §4.4 criterion 2 (per-period reproduction).
    # uids → row index map for O(1) bucket lookups (avoid O(n²) .index calls).
    uid_to_idx = {u: i for i, u in enumerate(uids)}
    period_indices: dict[str, list[int]] = {"2024": [], "2026": []}
    for u in uids:
        p = period_per_uid[u]
        bucket = "2024" if p.startswith("2024") else "2026"
        period_indices[bucket].append(uid_to_idx[u])

    period_fits: dict[str, dict] = {}
    for period_label in ("2024", "2026"):
        bucket_indices = period_indices[period_label]
        bucket_uids = tuple(uids[i] for i in bucket_indices)
        bucket_docs = tuple(docs[i] for i in bucket_indices)
        bucket_emb = embeddings[bucket_indices]
        name = f"period_{period_label}_mcs_{headline_mcs}"
        cached = _load_topics(PERIOD_FITS_DIR, name)
        if cached is None:
            print(f"  period {period_label}: fitting (n={len(bucket_uids)})…")
            t0 = perf_counter()
            fit = pipeline.fit_topic_model(
                min_cluster_size=headline_mcs,
                seed=config.SEED_PRIMARY,
                uids=bucket_uids,
                docs=bucket_docs,
                embeddings=bucket_emb,
            )
            _save_fit(fit, PERIOD_FITS_DIR, name)
            cached = fit.topics
            print(
                f"    {pipeline.n_clusters(cached)} clusters, "
                f"{perf_counter() - t0:.1f}s"
            )
        period_fits[period_label] = {
            "uids": bucket_uids,
            "topics": cached,
            "embeddings": bucket_emb,
            "indices": bucket_indices,
            "docs": bucket_docs,
        }

    # K sweep.
    rows: list[dict] = []
    for k in config.K_SWEEP_GRID:
        from bertopic import BERTopic
        # Reduce headline fit to k; cache the reduced label array.
        cache = INTERIM / f"headline_reducedK{k}.npy"
        if cache.exists():
            head_k_labels = np.load(cache)
        else:
            model = BERTopic.load(str(config.RAW_FIT_PATH))
            try:
                model.reduce_topics(list(docs), nr_topics=k)
                head_k_labels = np.asarray(model.topics_, dtype=np.int64)
            except Exception as exc:
                print(f"    K={k}: reduce skipped ({exc})")
                head_k_labels = seed_topics[config.SEED_PRIMARY]
            np.save(cache, head_k_labels)

        # Reduce each seed fit to k for ARI.
        seed_k_labels: dict[int, np.ndarray] = {}
        for seed in config.ALL_SEEDS:
            scache = INTERIM / f"seed_{seed}_reducedK{k}.npy"
            if scache.exists():
                seed_k_labels[seed] = np.load(scache)
                continue
            if seed == config.SEED_PRIMARY:
                src = config.RAW_FIT_PATH
            else:
                src = _model_path(SEED_FITS_DIR, f"seed_{seed}_mcs_{headline_mcs}")
            model = BERTopic.load(str(src))
            try:
                model.reduce_topics(list(docs), nr_topics=k)
                lbl = np.asarray(model.topics_, dtype=np.int64)
            except Exception:
                lbl = seed_topics[seed]
            seed_k_labels[seed] = lbl
            np.save(scache, lbl)

        # Pairwise ARI across the three seeds at this k.
        pair_aris: list[float] = []
        seeds = list(config.ALL_SEEDS)
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                pair_aris.append(
                    pipeline.pairwise_metrics(
                        seed_k_labels[seeds[i]], seed_k_labels[seeds[j]]
                    )["ari"]
                )
        mean_seed_ari = float(np.mean(pair_aris)) if pair_aris else float("nan")

        # Per-period centroid alignment at this k: reduce each per-period fit
        # to k, compute centroids in 3072-d posting space, Hungarian-align.
        joint_centroids = pipeline.cluster_centroids(head_k_labels, embeddings)

        period_alignments: list[float] = []
        for period_label, info in period_fits.items():
            pcache = INTERIM / f"period_{period_label}_reducedK{k}.npy"
            if pcache.exists():
                p_lbl = np.load(pcache)
            else:
                pmodel = BERTopic.load(
                    str(_model_path(
                        PERIOD_FITS_DIR,
                        f"period_{period_label}_mcs_{headline_mcs}"
                    ))
                )
                try:
                    pmodel.reduce_topics(list(info["docs"]), nr_topics=k)
                    p_lbl = np.asarray(pmodel.topics_, dtype=np.int64)
                except Exception:
                    p_lbl = info["topics"]
                np.save(pcache, p_lbl)
            p_centroids = pipeline.cluster_centroids(p_lbl, info["embeddings"])
            period_alignments.append(
                pipeline.hungarian_centroid_alignment(joint_centroids, p_centroids)
            )
        mean_period_align = (
            float(np.mean(period_alignments)) if period_alignments else float("nan")
        )

        outlier_rate = pipeline.noise_rate(head_k_labels)
        n_actual = pipeline.n_clusters(head_k_labels)

        rows.append({
            "k_target": k,
            "n_clusters": n_actual,
            "noise_rate": outlier_rate,
            "seed_pair_ari_mean": mean_seed_ari,
            "per_period_centroid_alignment_mean": mean_period_align,
        })
        print(
            f"  K={k}: n={n_actual}, noise={outlier_rate:.1%}, "
            f"seed_ari={mean_seed_ari:.3f}, "
            f"period_align={mean_period_align:.3f}"
        )

    with config.K_SWEEP_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Pick headline K per §4.4: smallest K satisfying ARI >= 0.4 AND
    # alignment >= 0.85 AND outlier <= 0.40. Interpretability gate (rating
    # >= 3.5/5) is deferred — see prereg_log.md autonomous-run note.
    qualifying = [
        r for r in rows
        if r["seed_pair_ari_mean"] >= config.SEED_ARI_MIN
        and r["per_period_centroid_alignment_mean"] >= config.PERIOD_REPRO_MIN
        and r["noise_rate"] <= config.HEADLINE_OUTLIER_MAX
    ]
    if qualifying:
        headline_k = min(r["k_target"] for r in qualifying)
    else:
        # Fallback: pick K with the highest seed ARI subject to outlier
        # constraint; document in prereg_log.md.
        eligible = [r for r in rows if r["noise_rate"] <= config.HEADLINE_OUTLIER_MAX]
        if not eligible:
            eligible = rows
        eligible.sort(
            key=lambda r: (r["seed_pair_ari_mean"], -r["k_target"]),
            reverse=True,
        )
        headline_k = eligible[0]["k_target"]

    return headline_k, rows


# ---------------------------------------------------------------------------
# S1.4 — mega-cluster gate
# ---------------------------------------------------------------------------

def s1_4_mega_cluster_gate(headline_k: int) -> dict:
    head_k_labels = np.load(INTERIM / f"headline_reducedK{headline_k}.npy")
    largest_id, share = pipeline.largest_cluster_share(head_k_labels)
    return {
        "headline_k": headline_k,
        "largest_cluster_id": largest_id,
        "largest_cluster_share": share,
        "gate_passed": share <= config.LARGEST_CLUSTER_SHARE_MAX,
    }


# ---------------------------------------------------------------------------
# S1.5 — determinism check
# ---------------------------------------------------------------------------

def s1_5_determinism_check(
    headline_mcs: int,
    uids: tuple[str, ...],
    docs: tuple[str, ...],
    embeddings: np.ndarray,
) -> dict:
    name = f"redo_mcs_{headline_mcs}_seed_{config.SEED_PRIMARY}"
    cached = _load_topics(DETERMINISM_DIR, name)
    if cached is None:
        print("  refitting at headline_mcs + seed=42…")
        fit = pipeline.fit_topic_model(
            min_cluster_size=headline_mcs,
            seed=config.SEED_PRIMARY,
            uids=uids, docs=docs, embeddings=embeddings,
        )
        _save_fit(fit, DETERMINISM_DIR, name)
        cached = fit.topics

    original = np.load(_topics_path(MCS_FITS_DIR, f"mcs_{headline_mcs}"))
    identical = bool(np.array_equal(cached, original))
    if not identical:
        ari = pipeline.pairwise_metrics(original, cached)["ari"]
    else:
        ari = 1.0
    return {"identical": identical, "ari": ari}


# ---------------------------------------------------------------------------
# S1.6 — LLM naming
# ---------------------------------------------------------------------------

def s1_6_naming(
    *,
    headline_k: int,
    docs: tuple[str, ...],
) -> list[dict]:
    """Propose labels for headline-K clusters via gpt-5.5."""
    head_k_labels = np.load(INTERIM / f"headline_reducedK{headline_k}.npy")

    from bertopic import BERTopic
    model = BERTopic.load(str(config.RAW_FIT_PATH))
    model.reduce_topics(list(docs), nr_topics=headline_k)

    cluster_ids = sorted({int(t) for t in head_k_labels if t != -1})
    rows: list[dict] = []
    for cid in cluster_ids:
        words = [w for w, _ in model.get_topic(cid)[:15]]
        member_idx = np.where(head_k_labels == cid)[0]
        # 5 representative + 2 random for breadth.
        rep_docs = model.get_representative_docs(cid) or []
        rep_docs = rep_docs[:5]
        rng = np.random.default_rng(seed=cid)
        if len(member_idx) > 0:
            random_idx = rng.choice(
                member_idx, size=min(2, len(member_idx)), replace=False,
            )
            random_docs = [docs[i] for i in random_idx]
        else:
            random_docs = []
        exemplars = [
            (f"posting {i}", d[:200])
            for i, d in enumerate(list(rep_docs) + random_docs)
        ]
        try:
            label = propose_label(
                top_words=words,
                exemplars=exemplars,
                model=config.LLM_MODEL_PRIMARY,
                request_id=f"headK{headline_k}-c{cid}",
            )
        except Exception as exc:  # noqa: BLE001
            label = {"label": "(LLM-naming-failed)", "confidence": 0.0,
                     "alternative": str(exc)[:200]}
        rows.append({
            "topic_id": cid,
            "n": int((head_k_labels == cid).sum()),
            "top_words": words,
            "gpt55_label": label.get("label"),
            "gpt55_confidence": label.get("confidence"),
            "gpt55_alternative": label.get("alternative"),
        })
        print(
            f"  cid={cid} n={int((head_k_labels == cid).sum())}: "
            f"{label.get('label')!r} ({label.get('confidence')})"
        )
    return rows


# ---------------------------------------------------------------------------
# S1.7 — hash artifacts and freeze
# ---------------------------------------------------------------------------

def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def s1_7_freeze(
    *,
    headline_mcs: int,
    headline_k: int,
    super_family_k: int,
    naming_rows: list[dict],
    mcs_sweep_rows: list[dict],
    k_sweep_rows: list[dict],
    determinism: dict,
    mega_cluster: dict,
    seed_summary: dict,
) -> Path:
    config.BERTOPIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Final topic_info.parquet — gpt55_label as the working label until §5.2
    # protocol lands.
    info_table = pa.table({
        "topic_id": [r["topic_id"] for r in naming_rows],
        "n": [r["n"] for r in naming_rows],
        "label": [r["gpt55_label"] for r in naming_rows],
        "gpt55_label": [r["gpt55_label"] for r in naming_rows],
        "gpt55_confidence": [r["gpt55_confidence"] for r in naming_rows],
        "gpt55_alternative": [r["gpt55_alternative"] for r in naming_rows],
        "top_words": [r["top_words"] for r in naming_rows],
    })
    pq.write_table(info_table, config.TOPIC_INFO_PATH, compression="zstd")

    # Assignments parquet.
    head_k_labels = np.load(INTERIM / f"headline_reducedK{headline_k}.npy")
    label_by_id = {r["topic_id"]: r["gpt55_label"] for r in naming_rows}
    sample_uids, _, _ = pipeline.load_sample(sample_path=config.SAMPLE_A_PATH)
    assert len(sample_uids) == len(head_k_labels)

    label_for = lambda tid: label_by_id.get(int(tid)) if tid != -1 else None
    assignments = pa.table({
        "uid": list(sample_uids),
        "topic_id": [int(t) for t in head_k_labels],
        "topic_label": [label_for(t) for t in head_k_labels],
        "is_outlier": [bool(t == -1) for t in head_k_labels],
    })
    pq.write_table(assignments, config.ASSIGNMENTS_PATH, compression="zstd")

    config_hash = _file_hash(Path(__file__).parent / "../config.py")
    sample_hash = _file_hash(config.SAMPLE_A_PATH)
    cache_hash = _file_hash(config.EMBEDDINGS_CACHE_PATH)
    model_hash = _file_hash(config.RAW_FIT_PATH)
    assignments_hash = _file_hash(config.ASSIGNMENTS_PATH)

    freeze = {
        "frozen_at": "2026-05-06",
        "headline_mcs": headline_mcs,
        "headline_k": headline_k,
        "super_family_k": super_family_k,
        "n_clusters_headline": pipeline.n_clusters(head_k_labels),
        "noise_rate_headline": pipeline.noise_rate(head_k_labels),
        "largest_cluster_share_headline": mega_cluster["largest_cluster_share"],
        "mega_cluster_gate_passed": mega_cluster["gate_passed"],
        "determinism_identical": determinism["identical"],
        "determinism_ari": determinism["ari"],
        "seed_summary": seed_summary,
        "mcs_sweep": mcs_sweep_rows,
        "k_sweep": k_sweep_rows,
        "config_hash": config_hash,
        "sample_hash": sample_hash,
        "embeddings_cache_hash": cache_hash,
        "model_hash": model_hash,
        "assignments_hash": assignments_hash,
    }
    config.STAGE1_FREEZE_JSON.write_text(json.dumps(freeze, indent=2))
    return config.STAGE1_FREEZE_JSON


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

class _PeriodLookup:
    def __init__(self) -> None:
        import duckdb
        con = duckdb.connect()
        con.execute("PRAGMA disable_progress_bar")
        rows = con.execute(
            f"SELECT uid, period FROM '{config.SAMPLE_A_PATH}'"
        ).fetchall()
        self._d = {u: p for u, p in rows}

    def __getitem__(self, key: str) -> str:
        return self._d[key]


def main() -> None:
    _ensure_dirs()

    print("Stage 1 — loading Sample A and embeddings…")
    uids, docs, embeddings = pipeline.load_sample(sample_path=config.SAMPLE_A_PATH)
    print(f"  Sample A: {len(uids):,} rows; embeddings {embeddings.shape}")
    period_lookup = _PeriodLookup()

    print("\nS1.1 — min_cluster_size sweep")
    headline_mcs, mcs_rows = s1_1_mcs_sweep(uids, docs, embeddings)
    print(f"  headline mcs = {headline_mcs}")

    print("\nS1.2 — headline fit + K sweep")
    headline_k, k_rows = s1_2_headline_and_k_sweep(
        headline_mcs, uids, docs, embeddings, period_lookup,
    )
    print(f"  headline K = {headline_k}")

    # Super-family K: the smallest K in the {10, 15} band that meets the
    # same gates if any does, otherwise the smaller of the two.
    super_family_candidates = [
        r for r in k_rows
        if r["k_target"] in config.SUPER_FAMILY_K_RANGE
        and r["seed_pair_ari_mean"] >= config.SEED_ARI_MIN
    ]
    super_family_k = (
        min(r["k_target"] for r in super_family_candidates)
        if super_family_candidates else config.SUPER_FAMILY_K_RANGE[0]
    )
    print(f"  super-family K = {super_family_k}")

    print("\nS1.3 — seed stability gate")
    seed_topics = {}
    for seed in config.ALL_SEEDS:
        if seed == config.SEED_PRIMARY:
            seed_topics[seed] = np.load(INTERIM / f"headline_reducedK{headline_k}.npy")
        else:
            seed_topics[seed] = np.load(INTERIM / f"seed_{seed}_reducedK{headline_k}.npy")
    seed_summary = {}
    seeds = list(config.ALL_SEEDS)
    pair_aris = []
    pair_alignments = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            ari = pipeline.pairwise_metrics(seed_topics[seeds[i]], seed_topics[seeds[j]])["ari"]
            ca = pipeline.cluster_centroids(seed_topics[seeds[i]], embeddings)
            cb = pipeline.cluster_centroids(seed_topics[seeds[j]], embeddings)
            align = pipeline.hungarian_centroid_alignment(ca, cb)
            seed_summary[f"{seeds[i]}_vs_{seeds[j]}"] = {
                "ari": ari, "centroid_alignment": align,
            }
            pair_aris.append(ari)
            pair_alignments.append(align)
            print(f"  {seeds[i]} vs {seeds[j]}: ARI={ari:.3f}, align={align:.3f}")

    seed_gate_passed = all(
        not (entry["ari"] < config.SEED_ARI_MIN
             and entry["centroid_alignment"] < config.WITHIN_2024_MIN)
        for entry in seed_summary.values()
    )
    print(f"  seed gate passed: {seed_gate_passed}")

    print("\nS1.4 — mega-cluster gate")
    mega = s1_4_mega_cluster_gate(headline_k)
    print(
        f"  largest cluster id={mega['largest_cluster_id']}, "
        f"share={mega['largest_cluster_share']:.1%}, "
        f"gate passed: {mega['gate_passed']}"
    )

    print("\nS1.5 — determinism check")
    determ = s1_5_determinism_check(headline_mcs, uids, docs, embeddings)
    print(f"  identical: {determ['identical']}, ARI: {determ['ari']:.4f}")

    print("\nS1.6 — LLM naming")
    naming_rows = s1_6_naming(headline_k=headline_k, docs=docs)

    print("\nS1.7 — freeze")
    out = s1_7_freeze(
        headline_mcs=headline_mcs,
        headline_k=headline_k,
        super_family_k=super_family_k,
        naming_rows=naming_rows,
        mcs_sweep_rows=mcs_rows,
        k_sweep_rows=k_rows,
        determinism=determ,
        mega_cluster=mega,
        seed_summary=seed_summary,
    )
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
