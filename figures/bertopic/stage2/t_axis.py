"""Stage 2 / T-axis — semantic-axis projection over Sample A (§6.1).

Builds the five pre-registered axes from `config.AXIS_ANCHORS`, projects every
Sample A posting onto each axis, then runs the §6.6 robustness suite:
permutation null on the period-mean shift, anchor leave-one-out sensitivity,
held-out anchor validation, and a random-direction permutation null on the
overall axis effect size. The script writes three parquet artifacts plus the
F9-candidate small-multiples figure and is fully standalone — it does not
depend on any other Stage 2 file.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from figures.bertopic import config  # noqa: E402
from figures.style import FIGSIZE_DOUBLE, save, setup  # noqa: E402

PERMUTATION_N = 1000
RNG_SEED = 42
HELDOUT_FRACTION = 0.30  # rounds to 2 of 6 anchors per pole
AXIS_VECTORS_DIR = config.INTERMEDIATE_DIR / "axes"

EXPECTED_HASHES = {
    "model_hash": "d51f15e613f62b221139503bc84e6d3757689aac5e07979beb6ed3dbce509415",
    "sample_hash": "6719a0250fbfcb630dad117b409d441697d493b209b219e1c9d08b09acfeb265",
    "embeddings_cache_hash": "29d77bf9e24e6250d7b303a17fb22b80b9575a09a46d88c9dbd5d75c3b479b27",
    "assignments_hash": "a03bc515094050996338094f28851126b8c1f07f7f3b26d2f678f6cb6808ab82",
    "config_hash": "bef20ab2916ad72bd87aaefb0d18ba13644f9989ddd8e9bad4eac2b01a07bce8",
}

# The frozen BERTopic model artifact is the raw fit at the headline mcs (70).
# data/bertopic/model.bertopic was never materialized as a separate file; the
# Stage 1 freeze recorded the raw_fit.bertopic SHA under that key.
MODEL_ARTIFACT_PATH = config.INTERMEDIATE_DIR / "raw_fit.bertopic"


@dataclass(frozen=True)
class AxisResult:
    axis_id: str
    name: str
    anchor_ids: tuple[str, ...]
    axis_vector_path: str
    leave_one_out_spread: float
    held_out_hit_rate: float
    permutation_null_percentile: float
    period_shift: float
    permutation_null_p: float


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dir_sha256(path: Path) -> str:
    if path.is_file():
        return _file_sha256(path)
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(path)).encode())
            h.update(_file_sha256(f).encode())
    return h.hexdigest()


def verify_hashes() -> None:
    """Recompute Stage 1 artifact hashes and fail loud on any mismatch."""
    actual = {
        "model_hash": _dir_sha256(MODEL_ARTIFACT_PATH),
        "sample_hash": _file_sha256(config.SAMPLE_A_PATH),
        "embeddings_cache_hash": _file_sha256(config.EMBEDDINGS_CACHE_PATH),
        "assignments_hash": _file_sha256(config.ASSIGNMENTS_PATH),
        "config_hash": _file_sha256(Path(config.__file__)),
    }
    mismatches = {k: (actual[k], EXPECTED_HASHES[k]) for k in EXPECTED_HASHES if actual[k] != EXPECTED_HASHES[k]}
    if mismatches:
        for k, (a, e) in mismatches.items():
            print(f"HASH MISMATCH {k}: expected {e}, got {a}", file=sys.stderr)
        raise SystemExit("Stage 1 hash bundle does not verify; aborting.")
    print("Hash bundle verified (5/5).")


def load_anchor_index() -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")
    return con.execute(
        f"SELECT key, row_index FROM '{config.EMBEDDINGS_INDEX_PATH}' WHERE kind='anchor'"
    ).fetchdf()


def load_sample_a_with_assignments() -> pd.DataFrame:
    """Inner-join Sample A with assignments, plus the cached embedding row index.

    Drops outliers (topic_id == -1) since per-cluster summaries operate on the
    nine real clusters; the projection itself is computed on every Sample A row.
    """
    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")
    return con.execute(
        f"""
        SELECT s.uid, s.period, a.topic_id, idx.row_index
        FROM '{config.SAMPLE_A_PATH}' s
        JOIN '{config.ASSIGNMENTS_PATH}' a ON a.uid = s.uid
        JOIN '{config.EMBEDDINGS_INDEX_PATH}' idx ON idx.key = s.uid AND idx.kind='posting'
        """
    ).fetchdf()


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-norm vector")
    return v / n


def _pc1_of_differences(diffs: np.ndarray) -> np.ndarray:
    """Return the first principal component of a (n, d) difference matrix.

    Bolukbasi (2016) treats each pair difference as a noisy estimate of the
    same bias direction, so the SVD is computed on the uncentered difference
    matrix. Centering removes the very signal we want to recover (`PC1` of
    centered differences for our anchor set is empirically near-orthogonal to
    the mean direction). Sign-aligned to the mean difference so the positive
    pole stays positive.
    """
    _, _, vt = np.linalg.svd(diffs, full_matrices=False)
    pc1 = vt[0]
    if np.dot(pc1, diffs.mean(axis=0)) < 0:
        pc1 = -pc1
    return _l2_normalize(pc1)


def build_axis_vector(
    embeddings: np.ndarray,
    anchor_index: pd.DataFrame,
    axis_id: str,
    keep_pos_idx: tuple[int, ...] | None = None,
    keep_neg_idx: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Build one axis vector via PC1 of (positive_i − negative_i) differences."""
    pos_keys = [f"axis::{axis_id}::positive::{i}" for i in (keep_pos_idx or range(6))]
    neg_keys = [f"axis::{axis_id}::negative::{i}" for i in (keep_neg_idx or range(6))]
    pos_rows = anchor_index.set_index("key").loc[pos_keys, "row_index"].to_numpy()
    neg_rows = anchor_index.set_index("key").loc[neg_keys, "row_index"].to_numpy()
    pos_vecs = embeddings[pos_rows]
    neg_vecs = embeddings[neg_rows]
    if pos_vecs.shape[0] != neg_vecs.shape[0]:
        raise ValueError("axis pole sizes differ; PC1-of-differences requires paired anchors")
    diffs = pos_vecs - neg_vecs
    return _pc1_of_differences(diffs)


def project_postings(embeddings: np.ndarray, row_indices: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Cosine of each posting against axis. Embeddings are unit-norm by construction."""
    posts = embeddings[row_indices]  # (n, 3072)
    norms = np.linalg.norm(posts, axis=1)
    return (posts @ axis) / np.clip(norms, 1e-12, None)


def period_mean_shift(projections: np.ndarray, periods: np.ndarray) -> float:
    is_2026 = np.char.startswith(periods.astype(str), "2026")
    is_2024 = np.char.startswith(periods.astype(str), "2024")
    return float(projections[is_2026].mean() - projections[is_2024].mean())


def period_shift_permutation_p(
    projections: np.ndarray, periods: np.ndarray, observed: float, n: int = PERMUTATION_N
) -> float:
    """Two-sided p-value of period-mean shift under label permutation."""
    is_2024 = np.char.startswith(periods.astype(str), "2024")
    is_2026 = np.char.startswith(periods.astype(str), "2026")
    keep = is_2024 | is_2026
    proj = projections[keep]
    labels = is_2026[keep].astype(np.int8)
    rng = np.random.default_rng(RNG_SEED)
    null = np.empty(n, dtype=np.float64)
    for i in range(n):
        rng.shuffle(labels)
        null[i] = proj[labels == 1].mean() - proj[labels == 0].mean()
    return float((np.abs(null) >= abs(observed)).mean())


def random_direction_percentile(
    projections_actual: float,
    embeddings: np.ndarray,
    row_indices: np.ndarray,
    periods: np.ndarray,
    n: int = PERMUTATION_N,
) -> float:
    """Percentile of the actual axis effect size across n random unit directions.

    Effect size = |period_mean_shift|, matched against the same effect computed
    on projections onto a random unit direction in 3072-d.
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    is_2026 = np.char.startswith(periods.astype(str), "2026")
    is_2024 = np.char.startswith(periods.astype(str), "2024")
    posts = embeddings[row_indices].astype(np.float32)
    posts_norm = np.linalg.norm(posts, axis=1, keepdims=True)
    posts_norm = np.clip(posts_norm, 1e-12, None)
    null = np.empty(n, dtype=np.float64)
    target = abs(projections_actual)
    for i in range(n):
        v = rng.standard_normal(embeddings.shape[1]).astype(np.float32)
        v /= np.linalg.norm(v)
        proj = (posts @ v) / posts_norm[:, 0]
        shift = abs(proj[is_2026].mean() - proj[is_2024].mean())
        null[i] = shift
    return float((null < target).mean())


def leave_one_out_spread(
    embeddings: np.ndarray,
    anchor_index: pd.DataFrame,
    axis_id: str,
    full_axis: np.ndarray,
) -> tuple[float, list[float]]:
    """Drop each anchor pair one at a time; report max(1 - |cos(full, refit)|).

    PC1-of-differences (§6.1) is defined over paired (positive_i, negative_i)
    differences, so dropping anchor `i` from one pole only is undefined
    without re-pairing. We follow the spec's intent — sensitivity to *any*
    single anchor — by dropping the entire pair `i`, leaving 5 paired
    differences. This is the standard reading of Bolukbasi-style LOO and
    matches §11.8's sensitivity discipline.

    Returns (max_spread, per-pair differences).
    """
    diffs: list[float] = []
    for i in range(6):
        keep = tuple(j for j in range(6) if j != i)
        refit = build_axis_vector(embeddings, anchor_index, axis_id, keep, keep)
        cos = float(np.dot(full_axis, refit))
        diffs.append(1.0 - abs(cos))
    return max(diffs), diffs


def held_out_validation(
    embeddings: np.ndarray, anchor_index: pd.DataFrame, axis_id: str
) -> float:
    """Hold out 30% of anchor pairs (2 of 6), refit on the remaining 4 pairs,
    and project the held-out anchors.

    Hit rate = fraction of held-out positives that land positive plus fraction
    of held-out negatives that land negative, averaged over multiple random
    splits. Pairs (not individual pole anchors) are held out so the refit's
    PC1-of-differences stays well-defined at every split.
    """
    rng = np.random.default_rng(RNG_SEED + 2)
    key2row = anchor_index.set_index("key")["row_index"]
    n_repeats = 20
    hits = 0
    total = 0
    for _ in range(n_repeats):
        held = tuple(sorted(rng.choice(6, size=2, replace=False).tolist()))
        keep = tuple(i for i in range(6) if i not in held)
        refit = build_axis_vector(embeddings, anchor_index, axis_id, keep, keep)
        for i in held:
            row_pos = key2row[f"axis::{axis_id}::positive::{i}"]
            row_neg = key2row[f"axis::{axis_id}::negative::{i}"]
            proj_pos = float(embeddings[row_pos] @ refit / np.linalg.norm(embeddings[row_pos]))
            proj_neg = float(embeddings[row_neg] @ refit / np.linalg.norm(embeddings[row_neg]))
            hits += int(proj_pos > 0)
            hits += int(proj_neg < 0)
            total += 2
    return hits / max(total, 1)


def cluster_axis_profile(
    df: pd.DataFrame, axis_id: str, projections: np.ndarray
) -> pd.DataFrame:
    """Cluster mean and IQR per axis from `assignments.parquet`'s topic_id."""
    rows = []
    df = df.assign(projection=projections)
    df = df[df["topic_id"] >= 0]  # exclude outliers from per-cluster summaries
    for tid, grp in df.groupby("topic_id"):
        rows.append({
            "cluster_id": int(tid),
            "axis_id": axis_id,
            "mean": float(grp["projection"].mean()),
            "q25": float(grp["projection"].quantile(0.25)),
            "q75": float(grp["projection"].quantile(0.75)),
        })
    return pd.DataFrame(rows)


def render_figure(
    projections_by_axis: dict[str, np.ndarray],
    periods: np.ndarray,
    output_name: str = "fig_axis_projection",
) -> Path:
    """Five small multiples — one per axis — overlaying 2024 vs 2026 distributions."""
    setup()
    n_axes = len(projections_by_axis)
    fig, axes = plt.subplots(1, n_axes, figsize=FIGSIZE_DOUBLE, sharey=True)
    is_2024 = np.char.startswith(periods.astype(str), "2024")
    is_2026 = np.char.startswith(periods.astype(str), "2026")
    for ax, (axis_id, proj) in zip(axes, projections_by_axis.items()):
        bins = np.linspace(-0.5, 0.5, 41)
        ax.hist(proj[is_2024], bins=bins, density=True, alpha=0.55, color="#4477AA", label="2024")
        ax.hist(proj[is_2026], bins=bins, density=True, alpha=0.55, color="#EE6677", label="2026")
        ax.axvline(0, color="#888888", linewidth=0.7, linestyle="--")
        ax.set_title(axis_id.replace("_", " "), fontsize=8)
        ax.set_xlabel("cosine projection")
    axes[0].set_ylabel("density")
    axes[-1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    return save(fig, output_name)


def main() -> None:
    t0 = time.time()
    verify_hashes()
    AXIS_VECTORS_DIR.mkdir(parents=True, exist_ok=True)

    anchor_index = load_anchor_index()
    sample = load_sample_a_with_assignments()
    print(f"Sample A rows joined to assignments: {len(sample):,}")

    embeddings = np.load(config.EMBEDDINGS_CACHE_PATH, mmap_mode="r")
    print(f"Embeddings shape: {embeddings.shape}")

    row_indices = sample["row_index"].to_numpy()
    periods = sample["period"].to_numpy()

    axis_results: list[AxisResult] = []
    projections_long: list[pd.DataFrame] = []
    cluster_profile_long: list[pd.DataFrame] = []
    projections_by_axis: dict[str, np.ndarray] = {}

    for axis_id in config.AXIS_ANCHORS:
        print(f"\n=== {axis_id} ===")
        full_axis = build_axis_vector(embeddings, anchor_index, axis_id)
        axis_path = AXIS_VECTORS_DIR / f"{axis_id}.npy"
        np.save(axis_path, full_axis)

        proj = project_postings(embeddings, row_indices, full_axis)
        projections_by_axis[axis_id] = proj

        shift = period_mean_shift(proj, periods)
        p_perm = period_shift_permutation_p(proj, periods, shift)
        loo_spread, _ = leave_one_out_spread(embeddings, anchor_index, axis_id, full_axis)
        hit_rate = held_out_validation(embeddings, anchor_index, axis_id)
        rand_pct = random_direction_percentile(shift, embeddings, row_indices, periods)

        anchor_ids = tuple(
            f"axis::{axis_id}::{pole}::{i}" for pole in ("positive", "negative") for i in range(6)
        )
        axis_results.append(AxisResult(
            axis_id=axis_id,
            name=axis_id.replace("_", " "),
            anchor_ids=anchor_ids,
            axis_vector_path=str(axis_path.relative_to(PROJECT_ROOT)),
            leave_one_out_spread=loo_spread,
            held_out_hit_rate=hit_rate,
            permutation_null_percentile=rand_pct,
            period_shift=shift,
            permutation_null_p=p_perm,
        ))
        print(
            f"  shift={shift:+.4f}  p_perm={p_perm:.4f}  "
            f"LOO={loo_spread:.4f}  ratio={abs(shift)/max(loo_spread,1e-9):.2f}  "
            f"hit_rate={hit_rate:.2f}  rand_dir_pct={rand_pct:.4f}"
        )

        projections_long.append(pd.DataFrame({
            "uid": sample["uid"].values,
            "axis_id": axis_id,
            "projection": proj,
        }))
        cluster_profile_long.append(cluster_axis_profile(sample, axis_id, proj))

    # ---- Write artifacts ----
    axes_records = [
        {
            "axis_id": r.axis_id,
            "name": r.name,
            "anchor_ids": list(r.anchor_ids),
            "axis_vector_path": r.axis_vector_path,
            "leave_one_out_spread": r.leave_one_out_spread,
            "held_out_hit_rate": r.held_out_hit_rate,
            "permutation_null_percentile": r.permutation_null_percentile,
            "period_shift": r.period_shift,
            "permutation_null_p": r.permutation_null_p,
        }
        for r in axis_results
    ]
    pd.DataFrame(axes_records).to_parquet(config.BERTOPIC_DATA_DIR / "axes.parquet", index=False)
    pd.concat(projections_long, ignore_index=True).to_parquet(
        config.BERTOPIC_DATA_DIR / "axis_projections.parquet", index=False
    )
    pd.concat(cluster_profile_long, ignore_index=True).to_parquet(
        config.BERTOPIC_DATA_DIR / "cluster_axis_profile.parquet", index=False
    )

    fig_path = render_figure(projections_by_axis, periods)
    print(f"\nFigure: {fig_path}")

    summary = {
        "task": "T-axis",
        "n_postings": int(len(sample)),
        "axes": axes_records,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    summary_path = AXIS_VECTORS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nElapsed: {summary['elapsed_seconds']:.1f}s")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
