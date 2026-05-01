"""T18 Step 7 — Sensitivities (a) aggregator and (g) SWE classification tier.

For the headline DiD metrics, rerun with:
  - aggregators excluded
  - SWE sample restricted by tier: top (title_keyword), tier 1+2 only
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
ART = ROOT / "exploration" / "artifacts" / "T18"
TAB = ROOT / "exploration" / "tables" / "T18"


FEAT = pd.read_parquet(ART / "T18_posting_features.parquet")
print("Rows:", len(FEAT))
print("swe_classification_tier values:", FEAT["swe_classification_tier"].value_counts(dropna=False))

METRICS = [
    "ai_strict_binary", "ai_broad_binary", "has_ai_tech",
    "desc_len_chars", "tech_count", "org_scope_count",
    "mgmt_strict_count", "requirement_breadth",
]


def did_for_sample(df):
    """Return DiD table per metric."""
    rec = []
    for m in METRICS:
        pre_swe = df[(df["group"] == "SWE") & df["period"].isin(["2024-01", "2024-04"])][m].astype(float)
        post_swe = df[(df["group"] == "SWE") & df["period"].isin(["2026-03", "2026-04"])][m].astype(float)
        pre_ctrl = df[(df["group"] == "control") & df["period"].isin(["2024-01", "2024-04"])][m].astype(float)
        post_ctrl = df[(df["group"] == "control") & df["period"].isin(["2026-03", "2026-04"])][m].astype(float)
        if len(pre_swe) == 0 or len(pre_ctrl) == 0:
            continue
        swe_delta = post_swe.mean() - pre_swe.mean()
        ctrl_delta = post_ctrl.mean() - pre_ctrl.mean()
        did = swe_delta - ctrl_delta
        rec.append({
            "metric": m,
            "swe_pre": pre_swe.mean(), "swe_post": post_swe.mean(), "swe_delta": swe_delta,
            "ctrl_pre": pre_ctrl.mean(), "ctrl_post": post_ctrl.mean(), "ctrl_delta": ctrl_delta,
            "did": did,
            "did_share": did / swe_delta if swe_delta != 0 else np.nan,
        })
    return pd.DataFrame(rec)


# Baseline
base = did_for_sample(FEAT)
base["variant"] = "baseline"

# Sensitivity (a): exclude aggregators
agg_excl = did_for_sample(FEAT[~FEAT["is_aggregator"].fillna(False)])
agg_excl["variant"] = "no_aggregators"

# Sensitivity (g): restrict SWE to highest-confidence tier
# Look at tier values
tiers = FEAT["swe_classification_tier"].value_counts()
print("Tier distribution:\n", tiers)

# Keep only SWE rows with a high-confidence tier; keep adjacent/control as-is.
# Data's canonical SWE tiers: regex, title_lookup_llm, embedding_high.
# High-confidence = regex + embedding_high (title_lookup_llm has LLM adjudication but is
# applied when regex was unresolved, so it can include edge cases).
high_conf_tiers = ["regex", "embedding_high"]
keep_mask = (FEAT["group"] != "SWE") | FEAT["swe_classification_tier"].isin(high_conf_tiers)
hi_conf = did_for_sample(FEAT[keep_mask])
hi_conf["variant"] = "tier_high_conf"

out = pd.concat([base, agg_excl, hi_conf], ignore_index=True)
out.to_csv(TAB / "T18_sensitivity_did.csv", index=False)
print("\n=== Sensitivity DiD ===")
print(out.to_string())
