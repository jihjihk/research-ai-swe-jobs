"""T-drift — §6.3 per-cluster centroid drift 2024→2026, control-differenced.

Sub-agent task `T-drift` of the Stage 2 fan-out. Reads frozen Stage 1 inputs
(BERTopic assignments at headline K=10, embeddings cache, Sample A + Sample B),
verifies the §13.4 hash bundle, then:

1. For each headline-K cluster, computes Δ_swe = mean_emb(2026) - mean_emb(2024).
2. Computes the analogous Δ for control postings, where controls are clustered
   into broad occupation buckets via nearest BERTopic centroid (argmax cosine
   in 3072-d, using 2024 SWE centroids as references).
3. Pairs each SWE cluster with the control bucket whose 2024 centroid is most
   similar by cosine; computes Δ_swe_specific = Δ_swe - Δ_control_paired.
4. Builds the five §6.1 axes (PCA on anchor differences, embeddings already
   cached) and projects Δ_swe, Δ_swe_specific, and Δ_control onto each axis.
5. Bootstrap robustness: 5x 80% subset of cluster members per period, recompute
   |Δ_swe|; report the IQR.

Outputs:
- data/bertopic/centroid_drift.parquet
- figures/output/t_drift_panel.{pdf,png}

Standalone: no shared utils; standalone per §13.8.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Repo imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from figures.bertopic import config  # noqa: E402
from figures.style import setup, save, FIGSIZE_DOUBLE  # noqa: E402

OUTPUT_DRIFT_PARQUET = PROJECT_ROOT / "data/bertopic/centroid_drift.parquet"
RNG_SEED = 42
BOOTSTRAP_N = 5
BOOTSTRAP_FRAC = 0.80
MIN_MEMBERS_PER_PERIOD = 20

# Period grouping per §3.1.
PERIODS_2024 = ("2024-01", "2024-04")
PERIODS_2026 = ("2026-03", "2026-04")


# ---------------------------------------------------------------------------
# Hash verification (§13.4)
# ---------------------------------------------------------------------------

EXPECTED_HASHES = {
    "model_hash": "d51f15e613f62b221139503bc84e6d3757689aac5e07979beb6ed3dbce509415",
    "sample_hash": "6719a0250fbfcb630dad117b409d441697d493b209b219e1c9d08b09acfeb265",
    "embeddings_cache_hash": "29d77bf9e24e6250d7b303a17fb22b80b9575a09a46d88c9dbd5d75c3b479b27",
    "assignments_hash": "a03bc515094050996338094f28851126b8c1f07f7f3b26d2f678f6cb6808ab82",
    "config_hash": "bef20ab2916ad72bd87aaefb0d18ba13644f9989ddd8e9bad4eac2b01a07bce8",
}


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hashes() -> None:
    actual = {
        "model_hash": _file_hash(config.RAW_FIT_PATH),
        "sample_hash": _file_hash(config.SAMPLE_A_PATH),
        "embeddings_cache_hash": _file_hash(config.EMBEDDINGS_CACHE_PATH),
        "assignments_hash": _file_hash(config.ASSIGNMENTS_PATH),
        "config_hash": _file_hash(Path(config.__file__)),
    }
    bad = {k: (actual[k], EXPECTED_HASHES[k]) for k in EXPECTED_HASHES if actual[k] != EXPECTED_HASHES[k]}
    if bad:
        raise RuntimeError(f"Stage 1 hash mismatch — REFUSING TO PROCEED. Diffs: {bad}")
    print("[hash] all five Stage 1 hashes match")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_index_and_cache() -> tuple[dict[str, int], np.ndarray]:
    """Returns (uid → row_index for postings, mmapped 3072-d cache, anchor lookup)."""
    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")
    df = con.execute(
        f"SELECT key, kind, row_index FROM '{config.EMBEDDINGS_INDEX_PATH}'"
    ).fetchdf()
    posting_idx = {r.key: int(r.row_index) for r in df[df["kind"] == "posting"].itertuples()}
    anchor_idx = {r.key: int(r.row_index) for r in df[df["kind"] == "anchor"].itertuples()}
    cache = np.load(config.EMBEDDINGS_CACHE_PATH, mmap_mode="r")
    return posting_idx, anchor_idx, cache


def load_sample(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")
    cols = ", ".join(columns)
    return con.execute(f"SELECT {cols} FROM '{path}'").fetchdf()


def load_assignments() -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")
    return con.execute(
        f"SELECT uid, topic_id, topic_label, is_outlier FROM '{config.ASSIGNMENTS_PATH}'"
    ).fetchdf()


# ---------------------------------------------------------------------------
# Cosine utilities
# ---------------------------------------------------------------------------


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return v / n


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A : (m, d), B : (n, d). Returns (m, n) cosine similarity."""
    return normalize(A) @ normalize(B).T


# ---------------------------------------------------------------------------
# Axis construction (§6.1) — local rebuild
# ---------------------------------------------------------------------------


def build_axes(anchor_idx: dict[str, int], cache: np.ndarray) -> dict[str, np.ndarray]:
    """For each axis, form D = {v_pos_i - v_neg_i}, take PC1 as axis vector g.

    Returns name -> (3072,) unit vector.
    """
    axes: dict[str, np.ndarray] = {}
    for axis_name, poles in config.AXIS_ANCHORS.items():
        diffs = []
        for i in range(len(poles["positive"])):
            kp = f"axis::{axis_name}::positive::{i}"
            kn = f"axis::{axis_name}::negative::{i}"
            vp = np.asarray(cache[anchor_idx[kp]], dtype=np.float64)
            vn = np.asarray(cache[anchor_idx[kn]], dtype=np.float64)
            diffs.append(vp - vn)
        D = np.stack(diffs, axis=0)  # (6, 3072)
        # PC1 of D
        D_c = D - D.mean(axis=0, keepdims=True)
        # SVD on (n_samples, n_features); right singular vector of largest sv = PC1
        _, _, Vt = np.linalg.svd(D_c, full_matrices=False)
        g = Vt[0]  # (3072,)
        # Sign convention: align with mean of D so positive pole is positive
        if np.dot(g, D.mean(axis=0)) < 0:
            g = -g
        g = g / np.linalg.norm(g)
        axes[axis_name] = g.astype(np.float32)
    return axes


# ---------------------------------------------------------------------------
# Cluster centroids
# ---------------------------------------------------------------------------


def gather_embeddings(uids: list[str], posting_idx: dict[str, int], cache: np.ndarray) -> np.ndarray:
    rows = np.fromiter((posting_idx[u] for u in uids), dtype=np.int64, count=len(uids))
    return np.asarray(cache[rows], dtype=np.float32)


def compute_swe_centroids(
    sample_a: pd.DataFrame,
    assignments: pd.DataFrame,
    posting_idx: dict[str, int],
    cache: np.ndarray,
) -> dict[int, dict[str, dict]]:
    """For each non-outlier topic_id and each period, return cluster-mean embedding + uid list."""
    df = sample_a.merge(assignments, on="uid", how="inner")
    df = df[~df["is_outlier"]].copy()
    df["period_bucket"] = np.where(df["period"].isin(PERIODS_2024), "2024", "2026")

    out: dict[int, dict[str, dict]] = {}
    for tid, grp in df.groupby("topic_id"):
        per: dict[str, dict] = {}
        for bucket in ("2024", "2026"):
            sub = grp[grp["period_bucket"] == bucket]
            if len(sub) == 0:
                per[bucket] = {"n": 0, "uids": [], "centroid": None}
            else:
                emb = gather_embeddings(sub["uid"].tolist(), posting_idx, cache)
                per[bucket] = {
                    "n": len(sub),
                    "uids": sub["uid"].tolist(),
                    "centroid": emb.mean(axis=0).astype(np.float32),
                }
        out[int(tid)] = per
    return out


def compute_control_centroids(
    sample_b: pd.DataFrame,
    swe_centroids_2024: dict[int, np.ndarray],
    posting_idx: dict[str, int],
    cache: np.ndarray,
) -> dict[int, dict[str, dict]]:
    """Bucket each control posting by its nearest 2024 SWE centroid (argmax cosine).

    Then compute per-bucket per-period centroids. Buckets are keyed by SWE topic_id.
    """
    controls = sample_b[sample_b["is_control"] & ~sample_b["is_swe"]].copy()
    controls["period_bucket"] = np.where(controls["period"].isin(PERIODS_2024), "2024", "2026")

    centroid_ids = sorted(swe_centroids_2024.keys())
    C = np.stack([swe_centroids_2024[t] for t in centroid_ids], axis=0).astype(np.float32)

    # Score in chunks to avoid 50k x 3072 @ 9 x 3072 -> trivial, but be safe.
    uids = controls["uid"].tolist()
    emb = gather_embeddings(uids, posting_idx, cache)
    sims = cosine_matrix(emb, C)  # (n, K)
    nearest = np.argmax(sims, axis=1)
    controls["bucket_id"] = [centroid_ids[i] for i in nearest]

    out: dict[int, dict[str, dict]] = {}
    for bid, grp in controls.groupby("bucket_id"):
        per: dict[str, dict] = {}
        for bucket in ("2024", "2026"):
            sub = grp[grp["period_bucket"] == bucket]
            if len(sub) == 0:
                per[bucket] = {"n": 0, "uids": [], "centroid": None}
            else:
                e = gather_embeddings(sub["uid"].tolist(), posting_idx, cache)
                per[bucket] = {
                    "n": len(sub),
                    "uids": sub["uid"].tolist(),
                    "centroid": e.mean(axis=0).astype(np.float32),
                }
        out[int(bid)] = per
    return out


# ---------------------------------------------------------------------------
# Drift + axis decomposition
# ---------------------------------------------------------------------------


def axis_loadings(delta: np.ndarray, axes: dict[str, np.ndarray]) -> dict[str, float]:
    """Cosine of delta with each unit-axis vector."""
    out = {}
    norm = np.linalg.norm(delta)
    if norm == 0:
        return {k: 0.0 for k in axes}
    for name, g in axes.items():
        out[name] = float(np.dot(delta, g) / (norm * 1.0))
    return out


def bootstrap_drift_iqr(
    uids_2024: list[str],
    uids_2026: list[str],
    posting_idx: dict[str, int],
    cache: np.ndarray,
    rng: np.random.Generator,
) -> float:
    if min(len(uids_2024), len(uids_2026)) < MIN_MEMBERS_PER_PERIOD:
        return float("nan")
    mags = []
    for _ in range(BOOTSTRAP_N):
        s24 = rng.choice(len(uids_2024), size=int(BOOTSTRAP_FRAC * len(uids_2024)), replace=False)
        s26 = rng.choice(len(uids_2026), size=int(BOOTSTRAP_FRAC * len(uids_2026)), replace=False)
        u24 = [uids_2024[i] for i in s24]
        u26 = [uids_2026[i] for i in s26]
        e24 = gather_embeddings(u24, posting_idx, cache).mean(axis=0)
        e26 = gather_embeddings(u26, posting_idx, cache).mean(axis=0)
        mags.append(float(np.linalg.norm(e26 - e24)))
    q75, q25 = np.percentile(mags, [75, 25])
    return float(q75 - q25)


def permutation_drift_null(
    uids_2024: list[str],
    uids_2026: list[str],
    posting_idx: dict[str, int],
    cache: np.ndarray,
    rng: np.random.Generator,
    n_perm: int = 200,
) -> tuple[float, float]:
    """Permute period labels within the cluster; recompute |Δ| under each permutation.

    Returns (null_median_magnitude, p_value) where p = P(|Δ_null| >= |Δ_obs|).
    """
    if min(len(uids_2024), len(uids_2026)) < MIN_MEMBERS_PER_PERIOD:
        return float("nan"), float("nan")
    pool = uids_2024 + uids_2026
    n24 = len(uids_2024)
    n_pool = len(pool)
    # Observed
    e24 = gather_embeddings(uids_2024, posting_idx, cache).mean(axis=0)
    e26 = gather_embeddings(uids_2026, posting_idx, cache).mean(axis=0)
    obs = float(np.linalg.norm(e26 - e24))

    # Pre-fetch all embeddings once for speed
    all_emb = gather_embeddings(pool, posting_idx, cache)
    null_mags = []
    idx_arr = np.arange(n_pool)
    for _ in range(n_perm):
        rng.shuffle(idx_arr)
        a = idx_arr[:n24]
        b = idx_arr[n24:]
        m24 = all_emb[a].mean(axis=0)
        m26 = all_emb[b].mean(axis=0)
        null_mags.append(float(np.linalg.norm(m26 - m24)))
    null_arr = np.asarray(null_mags)
    p = float((np.sum(null_arr >= obs) + 1) / (n_perm + 1))
    return float(np.median(null_arr)), p


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def pair_swe_to_control(
    swe_centroids: dict[int, dict],
    control_centroids: dict[int, dict],
) -> dict[int, int]:
    """Pair each SWE cluster to the most-similar control bucket (centroid cosine in 2024).

    The control-bucket key space is also SWE topic_ids (since we bucketed
    controls by nearest 2024 SWE centroid). So the natural identity pairing is
    a candidate, but we still pick by max cosine — controls in a given bucket
    don't necessarily lie at exactly the SWE centroid for that bucket.
    """
    pairs: dict[int, int] = {}
    swe_ids = []
    swe_vecs = []
    for tid, p in swe_centroids.items():
        if p["2024"]["centroid"] is None:
            continue
        swe_ids.append(tid)
        swe_vecs.append(p["2024"]["centroid"])
    ctl_ids = []
    ctl_vecs = []
    for bid, p in control_centroids.items():
        if p["2024"]["centroid"] is None:
            continue
        ctl_ids.append(bid)
        ctl_vecs.append(p["2024"]["centroid"])
    if not ctl_ids:
        return {}
    Sw = np.stack(swe_vecs, axis=0)
    Cw = np.stack(ctl_vecs, axis=0)
    sims = cosine_matrix(Sw, Cw)
    nearest = np.argmax(sims, axis=1)
    for i, sid in enumerate(swe_ids):
        pairs[int(sid)] = int(ctl_ids[int(nearest[i])])
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    print("[T-drift] start")
    verify_hashes()

    print("[T-drift] loading index + cache")
    posting_idx, anchor_idx, cache = load_index_and_cache()
    print(f"[T-drift] cache shape={cache.shape} dtype={cache.dtype}")

    print("[T-drift] loading samples + assignments")
    sample_a = load_sample(config.SAMPLE_A_PATH, ("uid", "period"))
    sample_b = load_sample(
        config.SAMPLE_B_PATH, ("uid", "period", "is_swe", "is_control")
    )
    assignments = load_assignments()

    print("[T-drift] building axes (§6.1) from anchors")
    axes = build_axes(anchor_idx, cache)
    print(f"[T-drift] axes built: {list(axes.keys())}")

    print("[T-drift] SWE centroids per cluster x period")
    swe = compute_swe_centroids(sample_a, assignments, posting_idx, cache)
    print(f"[T-drift] SWE clusters (non-outlier): {sorted(swe.keys())}")

    swe_2024_centroids = {
        t: p["2024"]["centroid"] for t, p in swe.items() if p["2024"]["centroid"] is not None
    }
    print("[T-drift] control bucketing by nearest 2024 SWE centroid")
    ctl = compute_control_centroids(sample_b, swe_2024_centroids, posting_idx, cache)
    print(f"[T-drift] control buckets populated: {sorted(ctl.keys())}")

    pairs = pair_swe_to_control(swe, ctl)
    print(f"[T-drift] cluster pairings: {pairs}")

    rng = np.random.default_rng(RNG_SEED)

    # Cluster labels
    label_map = (
        assignments[["topic_id", "topic_label"]]
        .drop_duplicates()
        .set_index("topic_id")["topic_label"]
        .to_dict()
    )

    rows = []
    for tid in sorted(swe.keys()):
        p = swe[tid]
        n24, n26 = p["2024"]["n"], p["2026"]["n"]
        if min(n24, n26) < MIN_MEMBERS_PER_PERIOD:
            print(f"[T-drift] cluster {tid}: insufficient (n24={n24}, n26={n26}) — skipping")
            continue
        c24, c26 = p["2024"]["centroid"], p["2026"]["centroid"]
        delta_swe = (c26 - c24).astype(np.float64)
        mag_swe = float(np.linalg.norm(delta_swe))
        load_swe = axis_loadings(delta_swe, axes)

        # Control pair
        pair_id = pairs.get(int(tid))
        delta_ctl = None
        load_ctl = {k: float("nan") for k in axes}
        n_ctl_24 = n_ctl_26 = 0
        mag_ctl = float("nan")
        if pair_id is not None and pair_id in ctl:
            cp = ctl[pair_id]
            n_ctl_24, n_ctl_26 = cp["2024"]["n"], cp["2026"]["n"]
            if (
                cp["2024"]["centroid"] is not None
                and cp["2026"]["centroid"] is not None
                and min(n_ctl_24, n_ctl_26) >= MIN_MEMBERS_PER_PERIOD
            ):
                delta_ctl = (cp["2026"]["centroid"] - cp["2024"]["centroid"]).astype(np.float64)
                mag_ctl = float(np.linalg.norm(delta_ctl))
                load_ctl = axis_loadings(delta_ctl, axes)

        # SWE-specific
        if delta_ctl is not None:
            delta_swe_spec = delta_swe - delta_ctl
        else:
            delta_swe_spec = delta_swe.copy()  # no control to subtract; equals raw drift
        mag_swe_spec = float(np.linalg.norm(delta_swe_spec))
        load_swe_spec = axis_loadings(delta_swe_spec, axes)

        # Per-axis ratio = |swe_axis_shift| / max(|ctl_axis_shift|, eps)
        ratios = {}
        for k in axes:
            num = abs(load_swe[k]) * mag_swe
            den = abs(load_ctl[k]) * mag_ctl if not np.isnan(load_ctl[k]) and not np.isnan(mag_ctl) else float("nan")
            ratios[k] = float(num / den) if den and not np.isnan(den) and den > 0 else float("nan")

        # Bootstrap
        iqr = bootstrap_drift_iqr(
            p["2024"]["uids"], p["2026"]["uids"], posting_idx, cache, rng
        )
        # Permutation null
        null_med, perm_p = permutation_drift_null(
            p["2024"]["uids"], p["2026"]["uids"], posting_idx, cache, rng
        )

        rows.append({
            "cluster_id": int(tid),
            "cluster_label": label_map.get(int(tid)),
            "n_2024": int(n24),
            "n_2026": int(n26),
            "drift_magnitude": mag_swe,
            "drift_swe_specific_magnitude": mag_swe_spec,
            "axis_loadings_dict": json.dumps(load_swe),
            "axis_loadings_swe_specific_dict": json.dumps(load_swe_spec),
            "control_axis_loadings_dict": json.dumps(load_ctl),
            "control_pair_id": int(pair_id) if pair_id is not None else None,
            "n_control_2024": int(n_ctl_24),
            "n_control_2026": int(n_ctl_26),
            "drift_control_magnitude": mag_ctl,
            "axis_drift_ratio_dict": json.dumps(ratios),
            "bootstrap_iqr": float(iqr),
            "permutation_null_median_magnitude": float(null_med),
            "permutation_p": float(perm_p),
        })

    df = pd.DataFrame(rows)
    print(df.to_string())
    OUTPUT_DRIFT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df), OUTPUT_DRIFT_PARQUET, compression="zstd")
    print(f"[T-drift] wrote {OUTPUT_DRIFT_PARQUET}")

    # Summary stats for memo
    summary = {
        "n_clusters_evaluated": int(len(df)),
        "drift_magnitude_min": float(df["drift_magnitude"].min()),
        "drift_magnitude_max": float(df["drift_magnitude"].max()),
        "drift_magnitude_median": float(df["drift_magnitude"].median()),
        "drift_swe_specific_median": float(df["drift_swe_specific_magnitude"].median()),
        "control_drift_median": float(df["drift_control_magnitude"].median()),
        "bootstrap_iqr_median": float(df["bootstrap_iqr"].median()),
    }
    print(json.dumps(summary, indent=2))

    # ---------------- Figure: per-cluster drift bars + axis loadings heatmap ---
    setup()
    import matplotlib.pyplot as plt

    fig, axarr = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE, gridspec_kw={"width_ratios": [1.1, 1.5]})
    df_plot = df.sort_values("drift_magnitude", ascending=True).reset_index(drop=True)
    labels = [f"{r.cluster_id}: {r.cluster_label}" for r in df_plot.itertuples()]

    # Left panel: |Δ| bars with control overlay
    y = np.arange(len(df_plot))
    axarr[0].barh(y, df_plot["drift_magnitude"], height=0.6, color="#4477AA", label=r"$|\Delta_{\mathrm{SWE}}|$")
    axarr[0].barh(y, df_plot["drift_control_magnitude"], height=0.35, color="#EE6677", alpha=0.85, label=r"$|\Delta_{\mathrm{control}}|$")
    axarr[0].set_yticks(y)
    axarr[0].set_yticklabels(labels)
    axarr[0].set_xlabel("Centroid drift magnitude (3072-d L2)")
    axarr[0].legend(loc="lower right", frameon=False)
    axarr[0].set_title("(a) Per-cluster drift |Δ|")

    # Right panel: axis loadings heatmap
    axes_order = list(axes.keys())
    H = np.array([list(json.loads(s).values()) for s in df_plot["axis_loadings_dict"]])
    im = axarr[1].imshow(H, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axarr[1].set_yticks(y)
    axarr[1].set_yticklabels([f"{r.cluster_id}" for r in df_plot.itertuples()])
    axarr[1].set_xticks(np.arange(len(axes_order)))
    axarr[1].set_xticklabels(axes_order, rotation=30, ha="right")
    axarr[1].set_title("(b) Axis loadings (cos Δ_SWE, axis)")
    fig.colorbar(im, ax=axarr[1], shrink=0.7)

    fig.tight_layout()
    save(fig, "t_drift_panel")
    print("[T-drift] figure saved -> figures/output/t_drift_panel.{pdf,png}")
    print(f"[T-drift] elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
