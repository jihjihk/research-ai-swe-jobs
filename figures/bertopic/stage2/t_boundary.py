"""T-boundary — §6.2 cluster-difference vectors and boundary postings.

For pairs of headline-K BERTopic clusters of substantive interest, compute the
centroid difference δ_AB = v_A − v_B in 3072-d posting space, L2-normalize,
project every posting belonging to A or B onto δ_AB, and call boundary
postings those with |projection| < 0.05 (cosine units).

Outputs:
- data/bertopic/boundary_postings.parquet (uid, cluster_pair, projection, period)
- data/bertopic/boundary_summary.parquet (cluster_pair, n_2024, n_2026,
  boundary_frac_2024, boundary_frac_2026, delta, permutation_p)

Gate 2 threshold: boundary-fraction change >= 5pp AND permutation p < 0.05.

Standalone script — no shared utilities. Run from repo root with the project
venv:

    .venv/bin/python figures/bertopic/stage2/t_boundary.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Ensure we can import the project package.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from figures.bertopic import config  # noqa: E402

import duckdb  # noqa: E402


# ---------------------------------------------------------------------------
# Hash verification — sub-agent first action per spec.
# ---------------------------------------------------------------------------

EXPECTED_HASHES = {
    "model_hash": "d51f15e613f62b221139503bc84e6d3757689aac5e07979beb6ed3dbce509415",
    "sample_hash": "6719a0250fbfcb630dad117b409d441697d493b209b219e1c9d08b09acfeb265",
    "embeddings_cache_hash": (
        "29d77bf9e24e6250d7b303a17fb22b80b9575a09a46d88c9dbd5d75c3b479b27"
    ),
    "assignments_hash": (
        "a03bc515094050996338094f28851126b8c1f07f7f3b26d2f678f6cb6808ab82"
    ),
    "config_hash": "bef20ab2916ad72bd87aaefb0d18ba13644f9989ddd8e9bad4eac2b01a07bce8",
}


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_dir():
        for fp in sorted(path.rglob("*")):
            if fp.is_file():
                h.update(fp.relative_to(path).as_posix().encode())
                with open(fp, "rb") as f:
                    while chunk := f.read(1024 * 1024):
                        h.update(chunk)
        return h.hexdigest()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_hashes() -> None:
    paths = {
        "model_hash": config.RAW_FIT_PATH,
        "sample_hash": config.SAMPLE_A_PATH,
        "embeddings_cache_hash": config.EMBEDDINGS_CACHE_PATH,
        "assignments_hash": config.ASSIGNMENTS_PATH,
        "config_hash": Path(config.__file__),
    }
    mismatches = []
    for key, path in paths.items():
        actual = _sha256_path(path)
        expected = EXPECTED_HASHES[key]
        ok = actual == expected
        print(f"  {key}: {'OK' if ok else 'MISMATCH'}  {actual[:16]}... vs {expected[:16]}...")
        if not ok:
            mismatches.append((key, actual, expected))
    if mismatches:
        raise RuntimeError(
            f"Stage 1 hash mismatch — fail loud, do not proceed: {mismatches}"
        )


# ---------------------------------------------------------------------------
# Cluster pair selection (§6.2).
# ---------------------------------------------------------------------------

# Pair list. Labels mapped from topic_info.parquet at headline K = 10.
# AI = topic 0 (AI Software Engineering), Backend/Full-Stack = topic 4
# (Full Stack Developer), Data = topic 2 (Data Engineer), E-commerce =
# topic 6 (has ML/AI vocabulary), Test = topic 1, Salesforce = topic 3,
# Mobile = topic 5, App Sys Analyst = topic 7, ServiceNow = topic 8.
#
# No DevOps or SRE cluster exists at headline K = 10, so we substitute
# the closest platform-developer pair (Salesforce vs ServiceNow) and
# document the absence in the memo.
CLUSTER_PAIRS: list[tuple[str, int, int]] = [
    ("AI_vs_FullStack", 0, 4),  # spec-required: AI vs Backend-flavored
    ("AI_vs_Data", 0, 2),  # spec-required: AI vs Data-Scientist-flavored
    ("AI_vs_Ecommerce", 0, 6),  # AI vs E-commerce (also ML-flavored)
    ("Test_vs_FullStack", 1, 4),  # adjacent SWE roles
    ("Salesforce_vs_ServiceNow", 3, 8),  # platform-developer pair (DevOps proxy)
    ("Data_vs_Ecommerce", 2, 6),  # both data/ML adjacent
    ("FullStack_vs_Mobile", 4, 5),  # client-facing dev
    ("AppAnalyst_vs_ServiceNow", 7, 8),  # both system-of-record platforms
]

BOUNDARY_THRESHOLD = 0.05  # cosine units, per design.md §6.2
N_PERMUTATIONS = 1000
PERMUTATION_SEED = 42


# ---------------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------------

def load_inputs() -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    """Load assignments, sample (for period), and embeddings cache.

    Returns
    -------
    posting_df : DataFrame indexed 0..n-1 with columns
        uid, topic_id, is_outlier, period, period2 ('2024' or '2026'),
        emb_row (row index in embeddings_cache.npy)
    embeddings : ndarray (n_cache, 3072)
    uid_to_emb_row : dict[uid -> embeddings row index]
    """
    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")

    # Sample A is the substrate that BERTopic was fit on.
    sample = con.execute(
        f"""
        SELECT uid, period
        FROM read_parquet('{config.SAMPLE_A_PATH}')
        """
    ).df()
    assignments = con.execute(
        f"""
        SELECT uid, topic_id, is_outlier
        FROM read_parquet('{config.ASSIGNMENTS_PATH}')
        """
    ).df()
    emb_index = con.execute(
        f"""
        SELECT key AS uid, row_index AS emb_row
        FROM read_parquet('{config.EMBEDDINGS_INDEX_PATH}')
        WHERE kind = 'posting'
        """
    ).df()

    df = sample.merge(assignments, on="uid", how="inner").merge(
        emb_index, on="uid", how="inner"
    )
    if len(df) != len(sample):
        raise RuntimeError(
            f"Join lost rows: sample={len(sample)} -> joined={len(df)}"
        )

    df["period2"] = df["period"].str.slice(0, 4)
    df = df.sort_values("emb_row").reset_index(drop=True)

    embeddings = np.load(config.EMBEDDINGS_CACHE_PATH, mmap_mode="r")
    if embeddings.shape[1] != config.EMBEDDING_DIMS:
        raise RuntimeError(
            f"Embedding dim mismatch: {embeddings.shape} vs expected "
            f"({config.EMBEDDING_DIMS},)"
        )

    print(f"  postings (sample A joined): {len(df):,}")
    print(f"  embeddings_cache shape: {embeddings.shape}")
    print(f"  period2 distribution:\n{df['period2'].value_counts().sort_index().to_string()}")
    print(f"  topic distribution (incl. outliers):\n{df['topic_id'].value_counts().sort_index().to_string()}")
    return df, np.asarray(embeddings), None


# ---------------------------------------------------------------------------
# Per-pair analysis.
# ---------------------------------------------------------------------------

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0 or not np.isfinite(n):
        raise RuntimeError(f"Non-finite norm in centroid difference: {n}")
    return v / n


def _normalize_rows(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0):
        raise RuntimeError("Non-finite or zero embedding norms encountered")
    return M / norms


def analyze_pair(
    pair_name: str,
    topic_a: int,
    topic_b: int,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Return per-posting projection rows + per-pair summary dict."""
    sub = df[df["topic_id"].isin([topic_a, topic_b]) & (~df["is_outlier"])]
    if sub.empty:
        raise RuntimeError(f"No postings for pair {pair_name} ({topic_a}, {topic_b})")
    rows_a = sub.loc[sub["topic_id"] == topic_a, "emb_row"].to_numpy()
    rows_b = sub.loc[sub["topic_id"] == topic_b, "emb_row"].to_numpy()
    if len(rows_a) == 0 or len(rows_b) == 0:
        raise RuntimeError(
            f"Empty cluster for pair {pair_name}: |A|={len(rows_a)}, |B|={len(rows_b)}"
        )

    emb_a = embeddings[rows_a]
    emb_b = embeddings[rows_b]
    centroid_a = emb_a.mean(axis=0)
    centroid_b = emb_b.mean(axis=0)
    delta = _l2_normalize(centroid_a - centroid_b)

    # Cosine projection of L2-normalized posting embeddings onto delta.
    pair_rows = sub["emb_row"].to_numpy()
    pair_emb = embeddings[pair_rows]
    pair_emb_norm = _normalize_rows(pair_emb)
    projections = pair_emb_norm @ delta  # (n,) — cosine units

    pair_df = pd.DataFrame(
        {
            "uid": sub["uid"].to_numpy(),
            "topic_id": sub["topic_id"].to_numpy(),
            "cluster_pair": pair_name,
            "projection": projections.astype(np.float32),
            "period": sub["period2"].to_numpy(),
        }
    )

    # Boundary fraction per period.
    is_boundary = np.abs(projections) < BOUNDARY_THRESHOLD
    period = pair_df["period"].to_numpy()
    n_2024 = int((period == "2024").sum())
    n_2026 = int((period == "2026").sum())
    if n_2024 == 0 or n_2026 == 0:
        raise RuntimeError(
            f"Empty period in pair {pair_name}: n_2024={n_2024}, n_2026={n_2026}"
        )
    frac_2024 = float(is_boundary[period == "2024"].mean())
    frac_2026 = float(is_boundary[period == "2026"].mean())
    delta_frac = frac_2026 - frac_2024

    # Permutation null on the change: shuffle period labels 1,000 times.
    obs_abs = abs(delta_frac)
    n_total = len(period)
    boundary_int = is_boundary.astype(np.int32)
    null_deltas = np.empty(N_PERMUTATIONS, dtype=np.float32)
    for i in range(N_PERMUTATIONS):
        perm = rng.permutation(n_total)
        # Permuted period label: first n_2024 -> 2024, rest -> 2026.
        # Equivalently: random assignment of postings to periods preserving
        # marginal counts.
        idx_2024 = perm[:n_2024]
        idx_2026 = perm[n_2024:]
        f24 = boundary_int[idx_2024].mean()
        f26 = boundary_int[idx_2026].mean()
        null_deltas[i] = f26 - f24
    perm_p = float((np.abs(null_deltas) >= obs_abs).mean())

    summary = {
        "cluster_pair": pair_name,
        "topic_a": topic_a,
        "topic_b": topic_b,
        "n_total": n_total,
        "n_2024": n_2024,
        "n_2026": n_2026,
        "boundary_frac_2024": frac_2024,
        "boundary_frac_2026": frac_2026,
        "delta": delta_frac,
        "abs_delta": obs_abs,
        "permutation_p": perm_p,
        "centroid_distance_cos": float(
            1.0 - np.dot(_l2_normalize(centroid_a), _l2_normalize(centroid_b))
        ),
    }
    return pair_df, summary


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    print("=== T-boundary ===")
    print("Step 1: verify Stage 1 hashes…")
    verify_hashes()

    print("Step 2: load inputs…")
    df, embeddings, _ = load_inputs()

    print("Step 3: per-pair analysis…")
    rng = np.random.default_rng(PERMUTATION_SEED)
    all_pair_dfs: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for pair_name, ta, tb in CLUSTER_PAIRS:
        print(f"  - {pair_name} (topics {ta}, {tb})")
        pair_df, summary = analyze_pair(pair_name, ta, tb, df, embeddings, rng)
        print(
            f"      n_2024={summary['n_2024']}, n_2026={summary['n_2026']}, "
            f"frac_2024={summary['boundary_frac_2024']:.4f}, "
            f"frac_2026={summary['boundary_frac_2026']:.4f}, "
            f"delta={summary['delta']:+.4f}, perm_p={summary['permutation_p']:.4f}"
        )
        all_pair_dfs.append(pair_df)
        summaries.append(summary)

    out_postings = pd.concat(all_pair_dfs, axis=0, ignore_index=True)
    out_summary = pd.DataFrame(summaries)

    out_postings_path = config.BERTOPIC_DATA_DIR / "boundary_postings.parquet"
    out_summary_path = config.BERTOPIC_DATA_DIR / "boundary_summary.parquet"

    pq.write_table(pa.Table.from_pandas(out_postings, preserve_index=False), out_postings_path)
    pq.write_table(pa.Table.from_pandas(out_summary, preserve_index=False), out_summary_path)
    print(f"\nWrote {out_postings_path} ({len(out_postings):,} rows)")
    print(f"Wrote {out_summary_path} ({len(out_summary)} rows)")

    # Console summary.
    print("\n=== Summary ===")
    print(out_summary.to_string(index=False))
    print(f"\nElapsed: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
