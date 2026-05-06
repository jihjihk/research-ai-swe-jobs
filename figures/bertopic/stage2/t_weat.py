"""
T-weat — five pre-registered WEAT-style association tests (§6.4).

For each test in `config.WEAT_TESTS`:

1. Resolve target sets X, Y from the Stage 1 cluster catalog and Sample A.
2. Build attribute sets A, B by averaging six anchor embeddings each
   (post-L2-normalize the average).
3. Compute differential cosine s(t, A, B) = mean cos(t, A) − mean cos(t, B)
   per posting; Cohen's d of the X-vs-Y differential.
4. Permutation null: 10,000 random splits of (X ∪ Y) into same-sized
   halves; two-sided p on Cohen's d.
5. Bonferroni-correct across the five tests (alpha' = 0.01 / 5 = 0.002).

Inputs (all hash-frozen via `intermediate/stage1_freeze.json`):
- `data/bertopic/embeddings_cache.npy`           (108514 × 3072 float32)
- `data/bertopic/embeddings_cache.index.parquet` (key, kind, row_index)
- `data/bertopic/assignments.parquet`            (uid, topic_id, is_outlier)
- `data/bertopic/topic_info.parquet`             (topic_id, top_words, ...)
- `figures/bertopic/intermediate/sample_a.parquet`

Output:
- `data/bertopic/weat_results.parquet`

Run from project root:
    .venv/bin/python -m figures.bertopic.stage2.t_weat
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from figures.bertopic import config


# ---------------------------------------------------------------------------
# Resolution rules (per task spec)
# ---------------------------------------------------------------------------

AI_KEYWORDS: tuple[str, ...] = (
    "llm", "ai", "ml engineer", "machine learning", "rag",
    "agent", "vector", "foundation model", "generative",
)
LEGACY_KEYWORDS: tuple[str, ...] = (
    ".net", "cobol", "mainframe", "php", "wordpress",
    "autosar", "servicenow", "plc", "fortran",
)

# seniority_final mapping for the seniority test.
# Source: design.md does not pin senior/junior buckets; we use the obvious
# `seniority_final` enum split. Documented in the memo.
SENIOR_LABELS: tuple[str, ...] = ("mid-senior", "director")
JUNIOR_LABELS: tuple[str, ...] = ("entry", "associate")

PERMUTATION_N = 10_000
PERMUTATION_SEED = 42


# ---------------------------------------------------------------------------
# Cluster-catalog resolution
# ---------------------------------------------------------------------------


def _has_keyword(top_words: list[str], keyword: str) -> bool:
    """Word-boundary match of `keyword` against the joined top_words string."""
    text = " ".join(w.lower() for w in top_words)
    pattern = r"(?<![a-z0-9])" + re.escape(keyword) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def resolve_clusters(topic_info: pd.DataFrame) -> dict[str, list[int]]:
    """Return {ai_clusters, legacy_clusters, non_ai_clusters} as topic_id lists.

    Outlier topic (-1) is excluded from all sets; all sets contain only
    real (n>0, non-outlier) clusters.
    """
    ai_ids: list[int] = []
    legacy_ids: list[int] = []
    real_topic_ids: list[int] = []
    for _, row in topic_info.iterrows():
        tid = int(row["topic_id"])
        if tid < 0:
            continue
        real_topic_ids.append(tid)
        tw = list(row["top_words"])
        if any(_has_keyword(tw, k) for k in AI_KEYWORDS):
            ai_ids.append(tid)
        if any(_has_keyword(tw, k) for k in LEGACY_KEYWORDS):
            legacy_ids.append(tid)
    non_ai_ids = [t for t in real_topic_ids if t not in ai_ids]
    return {
        "ai_clusters": sorted(ai_ids),
        "legacy_clusters": sorted(legacy_ids),
        "non_ai_clusters": sorted(non_ai_ids),
    }


# ---------------------------------------------------------------------------
# Target-set resolution
# ---------------------------------------------------------------------------


def _period_year(period: str) -> str:
    return "2024" if period.startswith("2024-") else ("2026" if period.startswith("2026-") else "?")


def resolve_target_uids(
    target_name: str,
    sample_a: pd.DataFrame,
    cluster_resolution: dict[str, list[int]],
) -> tuple[np.ndarray, str]:
    """Return (uids, human-readable definition string) for one target set.

    Cluster targets exclude HDBSCAN outliers (is_outlier == True) by
    construction since outlier rows have topic_id == -1 which is not in any
    cluster_resolution list.
    """
    if target_name in {"ai_clusters", "legacy_clusters", "non_ai_clusters"}:
        cluster_ids = cluster_resolution[target_name]
        mask = sample_a["topic_id"].isin(cluster_ids) & (~sample_a["is_outlier"])
        uids = sample_a.loc[mask, "uid"].to_numpy()
        defn = f"{target_name} = topic_id ∈ {cluster_ids}"
        return uids, defn
    if target_name == "period_2024":
        mask = sample_a["period"].str.startswith("2024-")
        return sample_a.loc[mask, "uid"].to_numpy(), "period in {2024-01, 2024-04}"
    if target_name == "period_2026":
        mask = sample_a["period"].str.startswith("2026-")
        return sample_a.loc[mask, "uid"].to_numpy(), "period in {2026-03, 2026-04}"
    if target_name == "senior_swe":
        mask = sample_a["seniority_final"].isin(SENIOR_LABELS)
        return sample_a.loc[mask, "uid"].to_numpy(), f"seniority_final ∈ {list(SENIOR_LABELS)}"
    if target_name == "junior_swe":
        mask = sample_a["seniority_final"].isin(JUNIOR_LABELS)
        return sample_a.loc[mask, "uid"].to_numpy(), f"seniority_final ∈ {list(JUNIOR_LABELS)}"
    raise ValueError(f"unknown target name: {target_name}")


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def load_anchor_centroid(
    cache: np.ndarray,
    index_df: pd.DataFrame,
    set_name: str,
) -> np.ndarray:
    """Average the six WEAT anchors for `set_name` and L2-normalize the mean."""
    keys = [f"weat::{set_name}::{i}" for i in range(len(config.WEAT_ATTRIBUTES[set_name]))]
    rows = index_df.loc[index_df["key"].isin(keys), ["key", "row_index"]]
    rows = rows.set_index("key").loc[keys]  # preserve order
    indices = rows["row_index"].to_numpy()
    vecs = cache[indices].astype(np.float32)
    if vecs.shape[0] != len(keys):
        raise RuntimeError(f"missing WEAT anchor rows for set {set_name}: have {vecs.shape[0]} of {len(keys)}")
    mean = vecs.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm == 0.0 or not np.isfinite(norm):
        raise RuntimeError(f"degenerate anchor centroid for {set_name}")
    return (mean / norm).astype(np.float32)


def differential_cosine(
    posting_vecs: np.ndarray,
    centroid_a: np.ndarray,
    centroid_b: np.ndarray,
) -> np.ndarray:
    """For each row of posting_vecs, return cos(row, a) - cos(row, b).

    Posting cache rows are already L2-normalized (norms ≈ 1.0001 from
    Stage 0); centroid_a/b are L2-normalized post-mean. So a single dot
    product equals the cosine.
    """
    # Renormalize posting rows defensively to absorb the ~1e-3 drift in cached norms.
    norms = np.linalg.norm(posting_vecs, axis=1, keepdims=True)
    # avoid div-by-zero
    norms = np.where(norms == 0.0, 1.0, norms)
    unit = posting_vecs / norms
    return unit @ (centroid_a - centroid_b)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d with pooled SD (unbiased, sample std with ddof=1)."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    mx, my = float(x.mean()), float(y.mean())
    sx, sy = float(x.std(ddof=1)), float(y.std(ddof=1))
    pooled = np.sqrt(((nx - 1) * sx ** 2 + (ny - 1) * sy ** 2) / (nx + ny - 2))
    if pooled == 0.0:
        return float("nan")
    return (mx - my) / pooled


def permutation_p(
    diffs_x: np.ndarray,
    diffs_y: np.ndarray,
    observed_d: float,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    """Two-sided permutation p-value on Cohen's d.

    Pool (diffs_x ∪ diffs_y); shuffle; split into same-sized halves; recompute
    d. p = (#permutations with |d_perm| >= |observed_d| + 1) / (n_perm + 1).
    """
    pooled = np.concatenate([diffs_x, diffs_y])
    n_x = len(diffs_x)
    n_total = len(pooled)
    abs_obs = abs(observed_d)
    # Vectorize Cohen's d across permutations.
    # For each perm, draw indices for X (size n_x); rest goes to Y.
    # Compute means and pooled SD analytically.
    sum_total = pooled.sum()
    sumsq_total = (pooled ** 2).sum()

    extreme = 0
    # Process in chunks to keep memory bounded but do most work vectorized.
    chunk = 1000
    for start in range(0, n_perm, chunk):
        size = min(chunk, n_perm - start)
        # For each of `size` permutations, pick random sample of n_x indices.
        # Use np.random for argpartition-style sampling. Cheaper: argsort uniform.
        u = rng.random((size, n_total))
        idx_part = np.argpartition(u, n_x - 1, axis=1)[:, :n_x]
        x_vals = pooled[idx_part]               # (size, n_x)
        sum_x = x_vals.sum(axis=1)
        sumsq_x = (x_vals ** 2).sum(axis=1)
        n_y = n_total - n_x
        sum_y = sum_total - sum_x
        sumsq_y = sumsq_total - sumsq_x

        mean_x = sum_x / n_x
        mean_y = sum_y / n_y
        # sample variance with ddof=1: (sum sq - n*mean^2) / (n - 1)
        var_x = (sumsq_x - n_x * mean_x ** 2) / (n_x - 1)
        var_y = (sumsq_y - n_y * mean_y ** 2) / (n_y - 1)
        # numerical guard
        var_x = np.maximum(var_x, 0.0)
        var_y = np.maximum(var_y, 0.0)
        pooled_sd = np.sqrt(((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_total - 2))
        # divide-by-zero guards
        with np.errstate(divide="ignore", invalid="ignore"):
            d_perm = (mean_x - mean_y) / pooled_sd
        d_perm = np.where(pooled_sd > 0, d_perm, 0.0)
        extreme += int(np.sum(np.abs(d_perm) >= abs_obs))
    # +1 in numerator and denominator → conservative, never zero.
    return (extreme + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class WeatResult:
    test_name: str
    X_definition: str
    Y_definition: str
    A_set: str
    B_set: str
    n_X: int
    n_Y: int
    cohens_d: float
    p_value: float
    p_bonf: float
    mean_diff_X: float
    mean_diff_Y: float


def run() -> pd.DataFrame:
    t0 = time.time()
    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")

    # --- load Stage 1 catalog
    topic_info = con.execute(
        f"SELECT * FROM '{config.TOPIC_INFO_PATH}' ORDER BY topic_id"
    ).df()
    cluster_res = resolve_clusters(topic_info)
    print("[t_weat] cluster resolution:")
    for k, v in cluster_res.items():
        print(f"  {k}: {v}")

    # --- load Sample A (with topic assignments) joined to assignments
    sample_a = con.execute(
        f"""
        SELECT
          sa.uid, sa.period, sa.source, sa.seniority_final,
          a.topic_id, a.is_outlier
        FROM '{config.SAMPLE_A_PATH}' sa
        JOIN '{config.ASSIGNMENTS_PATH}' a USING (uid)
        """
    ).df()
    print(f"[t_weat] sample_a joined assignments: {len(sample_a):,} rows")

    # --- load embedding cache + index
    print("[t_weat] loading embedding cache (mmap)...")
    cache = np.load(config.EMBEDDINGS_CACHE_PATH, mmap_mode="r")
    print(f"[t_weat] cache shape: {cache.shape} dtype={cache.dtype}")
    index_df = con.execute(
        f"SELECT key, kind, row_index FROM '{config.EMBEDDINGS_INDEX_PATH}'"
    ).df()
    print(f"[t_weat] index rows: {len(index_df):,}")

    # uid → row_index table for postings only
    posting_idx = (
        index_df.loc[index_df["kind"] == "posting", ["key", "row_index"]]
        .rename(columns={"key": "uid"})
        .set_index("uid")["row_index"]
    )

    # --- precompute attribute centroids for every WEAT_ATTRIBUTE set
    centroids: dict[str, np.ndarray] = {
        name: load_anchor_centroid(cache, index_df, name)
        for name in config.WEAT_ATTRIBUTES
    }
    print(f"[t_weat] built {len(centroids)} attribute centroids")

    # --- iterate the five pre-registered tests
    rng = np.random.default_rng(PERMUTATION_SEED)
    n_tests = len(config.WEAT_TESTS)
    rows: list[WeatResult] = []
    for test_name, (target_x, target_y, attr_a, attr_b) in config.WEAT_TESTS.items():
        print(f"\n[t_weat] === {test_name} ===")
        uids_x, def_x = resolve_target_uids(target_x, sample_a, cluster_res)
        uids_y, def_y = resolve_target_uids(target_y, sample_a, cluster_res)
        # Postings present in the embedding cache only (sample_a is fully covered, but be safe)
        rows_x = posting_idx.loc[posting_idx.index.intersection(uids_x)].to_numpy()
        rows_y = posting_idx.loc[posting_idx.index.intersection(uids_y)].to_numpy()
        if len(rows_x) == 0 or len(rows_y) == 0:
            raise RuntimeError(f"empty target set for {test_name}: |X|={len(rows_x)}, |Y|={len(rows_y)}")

        # Materialize embedding slices (copies; OK at 58k × 3072 = ~700MB max).
        # Read in chunks of indices to stay friendly.
        def gather(rows_idx: np.ndarray) -> np.ndarray:
            # numpy fancy indexing copies; on a 4 GB cache this is fine.
            order = np.argsort(rows_idx)
            sorted_idx = rows_idx[order]
            buf = np.asarray(cache[sorted_idx], dtype=np.float32)
            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            return buf[inv]

        vecs_x = gather(rows_x)
        vecs_y = gather(rows_y)
        print(f"  |X|={len(vecs_x)}  |Y|={len(vecs_y)}")
        print(f"  X def: {def_x}")
        print(f"  Y def: {def_y}")
        print(f"  A: {attr_a}  B: {attr_b}")

        cent_a = centroids[attr_a]
        cent_b = centroids[attr_b]
        diffs_x = differential_cosine(vecs_x, cent_a, cent_b)
        diffs_y = differential_cosine(vecs_y, cent_a, cent_b)
        d = cohens_d(diffs_x, diffs_y)
        print(f"  mean_diff_X={diffs_x.mean():+.4f}  mean_diff_Y={diffs_y.mean():+.4f}  d={d:+.4f}")

        # 10,000-perm null
        t_perm = time.time()
        p = permutation_p(diffs_x, diffs_y, d, PERMUTATION_N, rng)
        print(f"  permutation p={p:.5f}  (n_perm={PERMUTATION_N}, took {time.time()-t_perm:.1f}s)")

        rows.append(
            WeatResult(
                test_name=test_name,
                X_definition=f"{target_x}: {def_x}",
                Y_definition=f"{target_y}: {def_y}",
                A_set=attr_a,
                B_set=attr_b,
                n_X=len(vecs_x),
                n_Y=len(vecs_y),
                cohens_d=float(d),
                p_value=float(p),
                p_bonf=float(min(p * n_tests, 1.0)),
                mean_diff_X=float(diffs_x.mean()),
                mean_diff_Y=float(diffs_y.mean()),
            )
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    out_path = config.BERTOPIC_DATA_DIR / "weat_results.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n[t_weat] wrote {out_path} ({len(df)} rows)")
    print(df.to_string(index=False))
    print(f"[t_weat] total runtime: {time.time()-t0:.1f}s")
    return df


if __name__ == "__main__":
    run()
