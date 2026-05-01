"""T18 Step 2 — SWE-specificity DiD.

For each metric: compute within-group pre (pooled-2024) vs post (scraped, combined 2026-03+04)
means; SWE change minus Control change = DiD. Bootstrap CIs on DiD.
Flag "SWE-specific" when SWE DiD >= 50% of SWE-only change AND sign agrees.
Output table: metric, pre_SWE, post_SWE, swe_change, pre_CTRL, post_CTRL, ctrl_change,
did, did_ci_lo, did_ci_hi, did_pct_of_swe_change, sensitivity_aggregator (DiD excluding aggregators).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
ART = ROOT / "exploration" / "artifacts" / "T18"
TAB = ROOT / "exploration" / "tables" / "T18"
TAB.mkdir(parents=True, exist_ok=True)

FEAT_PATH = ART / "T18_posting_features.parquet"

# Metrics treated as continuous (means); binaries included as prevalence (0/1 mean).
METRICS = [
    "ai_strict_binary",
    "ai_broad_binary",
    "has_ai_tech",
    "desc_len_chars",
    "org_scope_count",
    "soft_skill_count",
    "mgmt_strict_count",
    "tech_count",
    "ai_tech_count",
    "requirement_breadth",
    "edu_level",
]

RNG = np.random.default_rng(42)


def bootstrap_did(pre_swe, post_swe, pre_ctrl, post_ctrl, n_boot=400, rng=RNG):
    """Bootstrap DiD = (post_SWE - pre_SWE) - (post_CTRL - pre_CTRL)."""
    dids = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        s_pre = rng.choice(pre_swe, size=len(pre_swe), replace=True).mean()
        s_post = rng.choice(post_swe, size=len(post_swe), replace=True).mean()
        c_pre = rng.choice(pre_ctrl, size=len(pre_ctrl), replace=True).mean()
        c_post = rng.choice(post_ctrl, size=len(post_ctrl), replace=True).mean()
        dids[i] = (s_post - s_pre) - (c_post - c_pre)
    return float(np.percentile(dids, 2.5)), float(np.percentile(dids, 97.5))


def compute_block(df, group, metric, exclude_aggregator=False):
    """Return (pre_values, post_values) arrays for group on metric."""
    sub = df[df["group"] == group]
    if exclude_aggregator:
        sub = sub[~sub["is_aggregator"].fillna(False)]
    pre = sub[sub["period"].isin(["2024-01", "2024-04"])][metric].astype(float).to_numpy()
    post = sub[sub["period"].isin(["2026-03", "2026-04"])][metric].astype(float).to_numpy()
    return pre, post


def main():
    feat = pd.read_parquet(FEAT_PATH)
    print("Rows:", len(feat))

    rows = []
    for metric in METRICS:
        for exclude_agg in [False, True]:
            pre_swe, post_swe = compute_block(feat, "SWE", metric, exclude_agg)
            pre_ctrl, post_ctrl = compute_block(feat, "control", metric, exclude_agg)
            pre_adj, post_adj = compute_block(feat, "adjacent", metric, exclude_agg)
            swe_change = post_swe.mean() - pre_swe.mean()
            ctrl_change = post_ctrl.mean() - pre_ctrl.mean()
            adj_change = post_adj.mean() - pre_adj.mean()
            did = swe_change - ctrl_change
            did_adj = adj_change - ctrl_change
            lo, hi = bootstrap_did(pre_swe, post_swe, pre_ctrl, post_ctrl)
            lo_adj, hi_adj = bootstrap_did(pre_adj, post_adj, pre_ctrl, post_ctrl)
            # DiD as share of SWE change
            if abs(swe_change) > 1e-12:
                did_share = did / swe_change
            else:
                did_share = np.nan
            # Flag: "mostly macro" when |DiD| <= 50% of |swe_change|
            if np.isnan(did_share):
                flag = "no_swe_change"
            elif abs(did) <= 0.5 * abs(swe_change):
                flag = "mostly_macro"
            else:
                flag = "swe_specific"
            rows.append({
                "metric": metric,
                "exclude_aggregator": exclude_agg,
                "pre_swe": pre_swe.mean(),
                "post_swe": post_swe.mean(),
                "swe_change": swe_change,
                "pre_ctrl": pre_ctrl.mean(),
                "post_ctrl": post_ctrl.mean(),
                "ctrl_change": ctrl_change,
                "did_swe_vs_ctrl": did,
                "did_swe_ci_lo": lo,
                "did_swe_ci_hi": hi,
                "pre_adj": pre_adj.mean(),
                "post_adj": post_adj.mean(),
                "adj_change": adj_change,
                "did_adj_vs_ctrl": did_adj,
                "did_adj_ci_lo": lo_adj,
                "did_adj_ci_hi": hi_adj,
                "did_share_of_swe_change": did_share,
                "flag": flag,
            })

    out = pd.DataFrame(rows)
    out.to_csv(TAB / "T18_did_table.csv", index=False)
    print(out[out["exclude_aggregator"] == False][[
        "metric", "pre_swe", "post_swe", "swe_change",
        "pre_ctrl", "post_ctrl", "ctrl_change", "did_swe_vs_ctrl", "did_share_of_swe_change", "flag"
    ]].to_string())

    # Also compute a parallel-trends panel by period
    panel = (
        feat.groupby(["group", "period", "source"], observed=False)[METRICS]
        .mean()
        .reset_index()
    )
    panel.to_csv(TAB / "T18_parallel_trends.csv", index=False)
    print("Wrote DiD table and parallel-trends panel.")


if __name__ == "__main__":
    main()
