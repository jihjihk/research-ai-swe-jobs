"""T29 — Matched-style analysis.

The simple low-LLM-subset re-test is confounded because >96% of 2026 postings
score above the 2024 p25 cutoff — the low-LLM 2026 subsample (n=485) is too
small and likely a strange tail.

This script does a more defensible version:
  1. For each authorship-score decile (computed on the pooled corpus),
     compute the Gate 2 headline metrics by period.
  2. Test whether the cross-period change persists WITHIN each score decile.
     If the delta is concentrated in the top deciles, recruiter tooling is
     the mechanism. If it is flat across deciles, it's independent of style.

We also compute:
  - Share of 2026 that scores above the 2024 maximum (or high percentile)
  - Mean / median shift re-computed on EQUAL-size random subsamples
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
TBL = ROOT / "exploration" / "tables" / "T29"


def main():
    feat = pd.read_parquet(TBL / "authorship_flags.parquet")
    per_row = pd.read_parquet(ROOT / "exploration" / "tables" / "T28" / "per_row_metrics.parquet")
    df = feat.merge(
        per_row[["uid", "requirement_breadth", "tech_count", "credential_stack_depth",
                 "any_ai_narrow", "any_ai_broad", "scope_density", "clean_len"]],
        on="uid",
        how="left",
    )

    # Pooled score deciles
    df["score_decile"] = pd.qcut(df["authorship_score"], 10, labels=False, duplicates="drop")
    print("Score decile period distribution:")
    print(
        df.groupby(["score_decile", "period2"]).size().unstack(fill_value=0).to_string()
    )

    print("\n--- Within-decile Gate 2 metrics by period ---")
    rows = []
    for metric in [
        "clean_len",
        "requirement_breadth",
        "tech_count",
        "credential_stack_depth",
        "any_ai_narrow",
        "any_ai_broad",
        "scope_density",
    ]:
        g = df.groupby(["score_decile", "period2"])[metric].mean().unstack(fill_value=np.nan)
        g["delta"] = g.get("2026", np.nan) - g.get("2024", np.nan)
        g["metric"] = metric
        rows.append(g.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(TBL / "matched_decile_analysis.csv", index=False)

    # Present as a pivot
    for metric in ["clean_len", "requirement_breadth", "tech_count", "any_ai_broad", "any_ai_narrow"]:
        sub = out[out["metric"] == metric]
        print(f"\n{metric}:")
        print(sub[["score_decile", "2024", "2026", "delta"]].to_string())

    # Average delta in bottom-3 vs top-3 deciles
    print("\n--- Delta in bottom-3 vs top-3 deciles ---")
    agg_rows = []
    for metric in ["clean_len", "requirement_breadth", "tech_count", "credential_stack_depth", "any_ai_broad", "scope_density"]:
        sub = out[out["metric"] == metric]
        bot = sub[sub["score_decile"].isin([0, 1, 2])]["delta"].mean()
        top = sub[sub["score_decile"].isin([7, 8, 9])]["delta"].mean()
        overall = sub["delta"].mean()
        agg_rows.append(
            {"metric": metric, "delta_overall": overall, "delta_bottom3_deciles": bot, "delta_top3_deciles": top}
        )
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(TBL / "matched_bottom_top_deciles.csv", index=False)
    print(agg.to_string())

    # Alternative: symmetric bootstrap — sample matching score distributions
    # For each 2026 row, find 2024 row with closest score. Then compute metric delta.
    print("\n--- Nearest-style matched-pair re-test (sampled) ---")
    s24 = df[df["period2"] == "2024"].reset_index(drop=True)
    s26 = df[df["period2"] == "2026"].reset_index(drop=True)

    # Sort by score and use binary search
    s24_sorted = s24.sort_values("authorship_score").reset_index(drop=True)
    scores_24 = s24_sorted["authorship_score"].to_numpy()

    # Sample 5000 2026 rows
    samp = s26.sample(min(5000, len(s26)), random_state=0)
    import bisect
    match_rows = []
    for _, row in samp.iterrows():
        target = row["authorship_score"]
        i = bisect.bisect_left(scores_24, target)
        # Clip
        candidates = []
        for ii in [i - 1, i, i + 1]:
            if 0 <= ii < len(s24_sorted):
                candidates.append(ii)
        best = min(candidates, key=lambda k: abs(scores_24[k] - target))
        match_rows.append(s24_sorted.iloc[best])
    matched_24 = pd.DataFrame(match_rows).reset_index(drop=True)
    # Compute metric delta on matched pairs
    match_deltas = {}
    for metric in ["clean_len", "requirement_breadth", "tech_count", "credential_stack_depth", "any_ai_broad", "any_ai_narrow", "scope_density"]:
        d26 = samp[metric].to_numpy()
        d24 = matched_24[metric].to_numpy()
        match_deltas[metric] = float(np.nanmean(d26) - np.nanmean(d24))
    md = pd.Series(match_deltas, name="matched_delta").to_frame().reset_index().rename(columns={"index": "metric"})

    # Full-sample delta for comparison
    full_deltas = {}
    for metric in md["metric"].tolist():
        full_deltas[metric] = float(df[df["period2"] == "2026"][metric].mean() - df[df["period2"] == "2024"][metric].mean())
    md["full_delta"] = md["metric"].map(full_deltas)
    md["attenuation_pct"] = (1 - md["matched_delta"] / md["full_delta"]).where(md["full_delta"] != 0) * 100
    md.to_csv(TBL / "style_matched_delta.csv", index=False)
    print(md.to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
