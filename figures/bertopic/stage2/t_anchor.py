"""T-anchor — §6.5 anchor-neighborhood diffusion.

For each of the five `config.NEIGHBORHOOD_ANCHORS`, compute the cosine
similarity between the anchor description embedding and every Sample A
posting, then for each cosine threshold in
`config.ANCHOR_NEIGHBORHOOD_THRESHOLDS = (0.5, 0.6, 0.7, 0.8)` and each
period (2024, 2026):

  - neighborhood size (count of postings with cos >= threshold)
  - top-5 BERTopic clusters (Stage 1 headline K=10) by share of
    neighborhood members
  - bootstrap IQR of neighborhood size (5 x 80% bootstraps per period)
  - sensitivity to a +/- 0.05 perturbation in cosine threshold

Outputs:
  - data/bertopic/anchor_neighborhoods.parquet
  - figures/output/fig_anchor_neighborhood.pdf

Standalone script — no shared utilities. Read-only on Stage 1 frozen
artifacts.
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from figures.bertopic import config  # noqa: E402
from figures.style import FIGSIZE_DOUBLE, save, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "bertopic" / "embeddings_cache.npy"
INDEX_PATH = PROJECT_ROOT / "data" / "bertopic" / "embeddings_cache.index.parquet"
ASSIGNMENTS_PATH = PROJECT_ROOT / "data" / "bertopic" / "assignments.parquet"
TOPIC_INFO_PATH = PROJECT_ROOT / "data" / "bertopic" / "topic_info.parquet"
SAMPLE_A_PATH = PROJECT_ROOT / "figures" / "bertopic" / "intermediate" / "sample_a.parquet"
STAGE1_FREEZE_PATH = PROJECT_ROOT / "figures" / "bertopic" / "intermediate" / "stage1_freeze.json"

OUT_PARQUET = PROJECT_ROOT / "data" / "bertopic" / "anchor_neighborhoods.parquet"
FIG_NAME = "fig_anchor_neighborhood"

BOOTSTRAP_N = 5
BOOTSTRAP_FRAC = 0.80
PERTURBATION = 0.05

PRIMARY_SEED = config.SEED_PRIMARY  # 42

EXPECTED_HASHES = {
    "model_hash": "d51f15e613f62b221139503bc84e6d3757689aac5e07979beb6ed3dbce509415",
    "sample_hash": "6719a0250fbfcb630dad117b409d441697d493b209b219e1c9d08b09acfeb265",
    "embeddings_cache_hash": "29d77bf9e24e6250d7b303a17fb22b80b9575a09a46d88c9dbd5d75c3b479b27",
    "assignments_hash": "a03bc515094050996338094f28851126b8c1f07f7f3b26d2f678f6cb6808ab82",
    "config_hash": "bef20ab2916ad72bd87aaefb0d18ba13644f9989ddd8e9bad4eac2b01a07bce8",
}

# model.bertopic was renamed to raw_fit.bertopic; freeze hash points to that file.
MODEL_PATH = PROJECT_ROOT / "figures" / "bertopic" / "intermediate" / "raw_fit.bertopic"
CONFIG_PATH = PROJECT_ROOT / "figures" / "bertopic" / "config.py"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hashes() -> None:
    actual = {
        "model_hash": sha256_file(MODEL_PATH),
        "sample_hash": sha256_file(SAMPLE_A_PATH),
        "embeddings_cache_hash": sha256_file(EMBEDDINGS_PATH),
        "assignments_hash": sha256_file(ASSIGNMENTS_PATH),
        "config_hash": sha256_file(CONFIG_PATH),
    }
    mismatches = {k: (EXPECTED_HASHES[k], actual[k]) for k in EXPECTED_HASHES if EXPECTED_HASHES[k] != actual[k]}
    if mismatches:
        raise SystemExit(f"HASH MISMATCH (failing loud): {mismatches}")
    print("[t_anchor] hash bundle verified:")
    for k, v in actual.items():
        print(f"  {k} = {v}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_inputs():
    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")

    print("[t_anchor] loading sample A + assignments + index ...")
    df = con.execute(f"""
        SELECT
            a.uid,
            a.period,
            i.row_index,
            asg.topic_id,
            asg.topic_label,
            asg.is_outlier
        FROM read_parquet('{SAMPLE_A_PATH}') a
        INNER JOIN read_parquet('{INDEX_PATH}') i
            ON a.uid = i.key AND i.kind = 'posting'
        INNER JOIN read_parquet('{ASSIGNMENTS_PATH}') asg
            ON a.uid = asg.uid
        ORDER BY i.row_index
    """).df()
    if len(df) != 57766:
        raise SystemExit(f"sample A row count mismatch: got {len(df)}, expected 57766")
    df["period_yr"] = df["period"].str.slice(0, 4)
    if df["period_yr"].nunique() != 2:
        raise SystemExit(f"unexpected period years: {df['period_yr'].unique()}")
    print(f"  sample A: {len(df):,} rows, periods={dict(df['period_yr'].value_counts())}")

    print("[t_anchor] loading topic_info (cluster labels) ...")
    topic_info = con.execute(
        f"SELECT * FROM read_parquet('{TOPIC_INFO_PATH}')"
    ).df()
    label_col = "label" if "label" in topic_info.columns else "topic_label"
    topic_label_map = {int(r["topic_id"]): str(r[label_col]) for _, r in topic_info.iterrows()}
    print(f"  K = {len(topic_info)} clusters")

    print("[t_anchor] loading anchor index entries ...")
    anchor_idx = con.execute(f"""
        SELECT key, row_index FROM read_parquet('{INDEX_PATH}')
        WHERE kind = 'anchor' AND key LIKE 'neighborhood::%'
        ORDER BY key
    """).df()
    anchor_row = {r.key: int(r.row_index) for r in anchor_idx.itertuples()}
    expected_keys = {f"neighborhood::{role}" for role in config.NEIGHBORHOOD_ANCHORS}
    missing = expected_keys - set(anchor_row)
    if missing:
        raise SystemExit(f"missing anchor keys in cache: {missing}")
    print(f"  anchors: {len(anchor_row)} keys")

    print("[t_anchor] loading embeddings cache (memmap) ...")
    cache = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    if cache.shape[1] != config.EMBEDDING_DIMS:
        raise SystemExit(f"embedding dim mismatch: {cache.shape}")
    if cache.dtype != np.float32:
        raise SystemExit(f"embedding dtype {cache.dtype} != float32")

    # Sanity: anchor norms within tolerance
    for key, row in anchor_row.items():
        n = float(np.linalg.norm(cache[row]))
        lo, hi = config.EMBEDDING_NORM_TOLERANCE
        if not (lo <= n <= hi):
            raise SystemExit(f"anchor {key} norm {n:.6f} out of tolerance {lo, hi}")

    return df, topic_label_map, anchor_row, cache


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


def cosine_to_anchor(cache: np.ndarray, anchor_row: int, posting_rows: np.ndarray) -> np.ndarray:
    """Cosine similarity = dot product on unit-normalized embeddings."""
    anchor_vec = np.asarray(cache[anchor_row], dtype=np.float32)
    # Renormalize anchor in case of float drift; postings already in tolerance.
    n = np.linalg.norm(anchor_vec)
    if n > 0:
        anchor_vec = anchor_vec / n
    # Pull only the rows we need (sorted), then dot.
    # cache is mmap; fancy index loads on demand.
    posting_vecs = np.asarray(cache[posting_rows], dtype=np.float32)
    sims = posting_vecs @ anchor_vec
    return sims


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def top5_clusters(member_topic_ids: np.ndarray, total: int, topic_label_map: dict[int, str]) -> dict:
    if total == 0 or len(member_topic_ids) == 0:
        return {}
    vals, counts = np.unique(member_topic_ids, return_counts=True)
    order = np.argsort(-counts)
    top = order[:5]
    out: dict[str, dict] = {}
    for idx in top:
        tid = int(vals[idx])
        cnt = int(counts[idx])
        label = topic_label_map.get(tid) if tid != -1 else "<outlier>"
        out[str(tid)] = {
            "topic_label": label if label is not None else "<unlabeled>",
            "count": cnt,
            "share": cnt / total,
        }
    return out


def run_analysis(df: pd.DataFrame, topic_label_map: dict[int, str], anchor_row: dict[str, int], cache: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(PRIMARY_SEED)
    rows = []

    posting_rows = df["row_index"].to_numpy()
    period_yrs = df["period_yr"].to_numpy()
    topic_ids = df["topic_id"].to_numpy()

    periods = ["2024", "2026"]
    period_masks = {p: (period_yrs == p) for p in periods}
    n_per_period = {p: int(period_masks[p].sum()) for p in periods}
    print(f"[t_anchor] period sizes: {n_per_period}")

    thresholds = list(config.ANCHOR_NEIGHBORHOOD_THRESHOLDS)
    extended_thresholds = sorted(set(
        thresholds
        + [round(t + PERTURBATION, 4) for t in thresholds]
        + [round(t - PERTURBATION, 4) for t in thresholds]
    ))

    for role in sorted(config.NEIGHBORHOOD_ANCHORS):
        anchor_key = f"neighborhood::{role}"
        a_row = anchor_row[anchor_key]
        print(f"\n[t_anchor] anchor: {role}  (row {a_row})")
        sims = cosine_to_anchor(cache, a_row, posting_rows)
        print(
            f"  sims: min={sims.min():.3f} max={sims.max():.3f} "
            f"mean={sims.mean():.3f} std={sims.std():.3f}"
        )

        # Pre-build period bootstrap index sets (same for all thresholds)
        boot_indices = {}
        for p in periods:
            mask = period_masks[p]
            n_p = int(mask.sum())
            n_boot = int(round(BOOTSTRAP_FRAC * n_p))
            local_rng = np.random.default_rng(PRIMARY_SEED + hash(role + p) % 2**31)
            boots = []
            # Indices into the full df (not period-local), so we can index sims directly.
            full_idx = np.flatnonzero(mask)
            for b in range(BOOTSTRAP_N):
                draw = local_rng.choice(full_idx, size=n_boot, replace=False)
                boots.append(draw)
            boot_indices[p] = boots

        for t in thresholds:
            for p in periods:
                mask = period_masks[p]
                full_idx = np.flatnonzero(mask)
                period_sims = sims[full_idx]
                period_topics = topic_ids[full_idx]
                in_neigh = period_sims >= t
                size = int(in_neigh.sum())
                top5 = top5_clusters(period_topics[in_neigh], size, topic_label_map)
                period_size = int(mask.sum())
                share = size / period_size if period_size else 0.0

                # Bootstrap IQR of neighborhood size on period sub-samples
                boot_sizes = []
                for boot_full_idx in boot_indices[p]:
                    bsims = sims[boot_full_idx]
                    boot_sizes.append(int((bsims >= t).sum()))
                boot_sizes = np.asarray(boot_sizes)
                iqr = float(np.percentile(boot_sizes, 75) - np.percentile(boot_sizes, 25))

                # Threshold sensitivity: +/- 0.05
                t_plus = t + PERTURBATION
                t_minus = t - PERTURBATION
                size_plus = int((period_sims >= t_plus).sum())
                size_minus = int((period_sims >= t_minus).sum())

                rows.append({
                    "anchor_id": role,
                    "anchor_string": config.NEIGHBORHOOD_ANCHORS[role],
                    "period": p,
                    "threshold": float(t),
                    "neighborhood_size": size,
                    "period_total": period_size,
                    "neighborhood_share": share,
                    "top_clusters_dict": json.dumps(top5),
                    "bootstrap_iqr_size": iqr,
                    "bootstrap_sizes": json.dumps(boot_sizes.tolist()),
                    "size_threshold_plus_005": size_plus,
                    "size_threshold_minus_005": size_minus,
                })

        # Quick sanity print per anchor
        for t in thresholds:
            sizes = [r for r in rows if r["anchor_id"] == role and r["threshold"] == t]
            if sizes:
                s24 = next(s for s in sizes if s["period"] == "2024")
                s26 = next(s for s in sizes if s["period"] == "2026")
                d_share = s26["neighborhood_share"] - s24["neighborhood_share"]
                print(
                    f"  t={t:.1f}: 2024 n={s24['neighborhood_size']:>5d} "
                    f"({s24['neighborhood_share']*100:5.2f}%)  "
                    f"2026 n={s26['neighborhood_size']:>5d} "
                    f"({s26['neighborhood_share']*100:5.2f}%)  "
                    f"Δshare={d_share*100:+5.2f}pp"
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Gate-2 monotonicity check
# ---------------------------------------------------------------------------


def gate2_check(results: pd.DataFrame) -> dict:
    """For each anchor: is the 2024->2026 direction the same across all four thresholds?"""
    out = {}
    for role in sorted(config.NEIGHBORHOOD_ANCHORS):
        sub = results[results["anchor_id"] == role]
        deltas_size = []
        deltas_share = []
        for t in config.ANCHOR_NEIGHBORHOOD_THRESHOLDS:
            r24 = sub[(sub["period"] == "2024") & (sub["threshold"] == t)].iloc[0]
            r26 = sub[(sub["period"] == "2026") & (sub["threshold"] == t)].iloc[0]
            deltas_size.append(r26["neighborhood_size"] - r24["neighborhood_size"])
            deltas_share.append(r26["neighborhood_share"] - r24["neighborhood_share"])
        signs_size = {int(np.sign(x)) for x in deltas_size}
        signs_share = {int(np.sign(x)) for x in deltas_share}
        out[role] = {
            "deltas_size": deltas_size,
            "deltas_share": deltas_share,
            "monotonic_size_direction": len(signs_size) == 1 and 0 not in signs_size,
            "monotonic_share_direction": len(signs_share) == 1 and 0 not in signs_share,
            "common_size_sign": signs_size,
            "common_share_sign": signs_share,
        }
    return out


# ---------------------------------------------------------------------------
# Plot F10
# ---------------------------------------------------------------------------


def plot_f10(results: pd.DataFrame) -> None:
    setup()
    anchors = sorted(config.NEIGHBORHOOD_ANCHORS)
    n = len(anchors)
    # 1 row x 5 cols, double-width geometry
    fig, axes = plt.subplots(
        1, n, figsize=(FIGSIZE_DOUBLE[0], FIGSIZE_DOUBLE[1] * 1.05),
        sharex=True,
    )
    if n == 1:
        axes = [axes]

    palette = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]
    thresholds = list(config.ANCHOR_NEIGHBORHOOD_THRESHOLDS)
    periods = ["2024", "2026"]

    for ax, role in zip(axes, anchors):
        sub = results[results["anchor_id"] == role]
        for color, t in zip(palette, thresholds):
            ys = []
            errs = []
            for p in periods:
                row = sub[(sub["period"] == p) & (sub["threshold"] == t)].iloc[0]
                ys.append(row["neighborhood_size"])
                errs.append(row["bootstrap_iqr_size"] / 2.0)
            ax.errorbar(
                periods, ys, yerr=errs, fmt="-o", color=color,
                label=f"τ={t:.1f}", linewidth=1.0, markersize=3, capsize=2,
            )
        ax.set_title(role.replace("_", " "), fontsize=8)
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", alpha=0.4)

    axes[0].set_ylabel("Neighborhood size (log)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(thresholds),
        bbox_to_anchor=(0.5, -0.02), frameon=False,
    )
    fig.suptitle("F10: Anchor-neighborhood size by period and cosine threshold", y=1.02, fontsize=9)
    fig.tight_layout()
    pdf_path = save(fig, FIG_NAME)
    plt.close(fig)
    print(f"[t_anchor] saved {pdf_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    verify_hashes()

    df, topic_label_map, anchor_row, cache = load_inputs()
    results = run_analysis(df, topic_label_map, anchor_row, cache)

    # Persist results
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(results, preserve_index=False)
    pq.write_table(table, OUT_PARQUET)
    print(f"[t_anchor] wrote {OUT_PARQUET} ({len(results)} rows)")

    # Gate 2 monotonicity report
    gate2 = gate2_check(results)
    print("\n[t_anchor] Gate 2 monotonicity:")
    for role, info in gate2.items():
        print(
            f"  {role}: size dirs={info['common_size_sign']} "
            f"share dirs={info['common_share_sign']} "
            f"size-monotonic={info['monotonic_size_direction']} "
            f"share-monotonic={info['monotonic_share_direction']}"
        )
        print(f"    Δsize per τ: {info['deltas_size']}")
        print(f"    Δshare per τ: {[f'{x:+.4f}' for x in info['deltas_share']]}")

    plot_f10(results)

    elapsed = time.time() - t0
    print(f"\n[t_anchor] done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
