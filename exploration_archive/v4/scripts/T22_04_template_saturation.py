"""T22 step 4 — Template saturation per company.

For each company with >=5 SWE postings in a given period, compute the pairwise
cosine similarity of requirement-section text using sentence-transformer
embeddings (reusing the shared embedding index where possible, otherwise
falling back to TF-IDF).

Dedup step: identical description_hash rows are collapsed to ONE row per
hash before similarity is computed. This avoids trivial 1.0 similarity from
the 6-company exact-dup clusters (Affirm/Canonical/Epic/Google/SkillStorm/Uber).

Outputs:
  tables/T22/template_saturation_by_company_period.csv
  tables/T22/template_saturation_top20_companies.csv
  tables/T22/template_saturation_dedup_impact.csv
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/home/jihgaboot/gabor/job-research")
FEATS = ROOT / "exploration" / "artifacts" / "T22" / "ghost_indicators_per_posting.parquet"
EMB_NPY = ROOT / "exploration" / "artifacts" / "shared" / "swe_embeddings.npy"
EMB_IDX = ROOT / "exploration" / "artifacts" / "shared" / "swe_embedding_index.parquet"
TAB = ROOT / "exploration" / "tables" / "T22"

MIN_POSTS = 5


def mean_pairwise_cos(embeddings: np.ndarray) -> float:
    """Mean off-diagonal cosine sim. embeddings is L2-normalized (n,d).

    For n=1 returns nan. For small n computes dense sim; for large n uses
    chunked mean-of-dot-products.
    """
    n = embeddings.shape[0]
    if n < 2:
        return float("nan")
    sim = embeddings @ embeddings.T
    # subtract diag and normalize by n*(n-1)
    total = sim.sum() - np.trace(sim)
    return float(total / (n * (n - 1)))


def main() -> None:
    con = duckdb.connect()
    feats = con.execute(f"SELECT uid, period2, company_name_canonical, is_aggregator, description_hash FROM read_parquet('{FEATS}')").fetchdf()

    emb = np.load(EMB_NPY)
    idx = pq.read_table(EMB_IDX).to_pandas()
    uid_to_row = dict(zip(idx["uid"].tolist(), np.arange(len(idx))))

    feats["emb_row"] = feats["uid"].map(uid_to_row)
    feats = feats[feats["emb_row"].notna()].copy()
    feats["emb_row"] = feats["emb_row"].astype(int)

    # dedup by description_hash: collapse to first uid per (company, period, hash)
    before = len(feats)
    feats_dedup = (
        feats.sort_values("uid")
        .drop_duplicates(subset=["company_name_canonical", "period2", "description_hash"], keep="first")
    )
    after = len(feats_dedup)
    print(f"Dedup removed {before-after:,} exact-hash duplicates (retained {after:,}/{before:,})")

    def compute_saturation(frame: pd.DataFrame) -> pd.DataFrame:
        groups = frame.groupby(["company_name_canonical", "period2", "is_aggregator"])
        rows = []
        for (company, period, is_agg), sub in groups:
            if len(sub) < MIN_POSTS:
                continue
            sub_emb = emb[sub["emb_row"].values]
            sat = mean_pairwise_cos(sub_emb)
            rows.append({
                "company": company,
                "period": period,
                "is_aggregator": is_agg,
                "n_posts": len(sub),
                "mean_pairwise_cos": sat,
            })
        return pd.DataFrame(rows)

    before_df = compute_saturation(feats)
    after_df = compute_saturation(feats_dedup)

    before_df.to_csv(TAB / "template_saturation_before_dedup.csv", index=False)
    after_df.to_csv(TAB / "template_saturation_by_company_period.csv", index=False)

    # Top 20 most template-saturated companies (after dedup)
    top = after_df[after_df["n_posts"] >= 10].sort_values("mean_pairwise_cos", ascending=False).head(30)
    top.to_csv(TAB / "template_saturation_top_companies.csv", index=False)

    # Impact of dedup
    impact = before_df.merge(
        after_df[["company", "period", "mean_pairwise_cos", "n_posts"]],
        on=["company", "period"],
        how="outer",
        suffixes=("_before", "_after"),
    )
    impact["sim_drop"] = impact["mean_pairwise_cos_before"] - impact["mean_pairwise_cos_after"]
    impact = impact.sort_values("sim_drop", ascending=False).head(30)
    impact.to_csv(TAB / "template_saturation_dedup_impact.csv", index=False)

    # Overall stats
    print()
    print("Before dedup:")
    print(f"  companies with >=5 posts: {len(before_df):,}")
    print(f"  mean pairwise cos: {before_df['mean_pairwise_cos'].mean():.3f}")
    print(f"  % with saturation > 0.8: {(before_df['mean_pairwise_cos']>0.8).mean():.3f}")

    print("After dedup:")
    print(f"  companies with >=5 posts: {len(after_df):,}")
    print(f"  mean pairwise cos: {after_df['mean_pairwise_cos'].mean():.3f}")
    print(f"  % with saturation > 0.8: {(after_df['mean_pairwise_cos']>0.8).mean():.3f}")

    # Also compute by period
    print()
    print("By period (after dedup):")
    print(after_df.groupby("period")["mean_pairwise_cos"].agg(["mean", "median", "count"]).round(3))


if __name__ == "__main__":
    main()
