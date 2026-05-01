"""T18 Step 3 — SWE-adjacent role trajectories.

Identifies top adjacent titles (Data Scientist, Data Analyst, Quality/Network/Security
Engineer, Systems Admin, ML Engineer, DevOps, SRE) and tracks AI-mention, tech_count,
requirement_breadth within each title across pooled-2024 vs scraped-2026. The question:
are any adjacent titles moving toward SWE?
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
ART = ROOT / "exploration" / "artifacts" / "T18"
TAB = ROOT / "exploration" / "tables" / "T18"

FEAT = pd.read_parquet(ART / "T18_posting_features.parquet")
print("Rows:", len(FEAT))

# Title buckets (regex on title_normalized lowercase)
TITLE_BUCKETS = {
    "data_scientist": re.compile(r"\bdata scientist", re.IGNORECASE),
    "data_analyst": re.compile(r"\bdata analyst", re.IGNORECASE),
    "data_engineer": re.compile(r"\bdata engineer", re.IGNORECASE),
    "ml_engineer": re.compile(
        r"\b(ml|machine learning|ai|artificial intelligence) engineer", re.IGNORECASE
    ),
    "devops_engineer": re.compile(r"\bdevops\b", re.IGNORECASE),
    "sre": re.compile(r"\bsite reliability\b|\bsre\b", re.IGNORECASE),
    "quality_engineer": re.compile(
        r"\b(qa|quality|test(?:ing)?) (engineer|analyst)", re.IGNORECASE
    ),
    "network_engineer": re.compile(r"\bnetwork engineer", re.IGNORECASE),
    "security_engineer": re.compile(r"\bsecurity engineer", re.IGNORECASE),
    "systems_admin": re.compile(r"\bsystem(?:s)? admin", re.IGNORECASE),
    "cloud_engineer": re.compile(r"\bcloud engineer", re.IGNORECASE),
    "platform_engineer": re.compile(r"\bplatform engineer", re.IGNORECASE),
}


def bucket_title(title: str | None):
    if not isinstance(title, str):
        return None
    for name, pat in TITLE_BUCKETS.items():
        if pat.search(title):
            return name
    return None


adj = FEAT[FEAT["group"] == "adjacent"].copy()
adj["bucket"] = adj["title_normalized"].apply(bucket_title)

# Also compute SWE baseline
swe = FEAT[FEAT["group"] == "SWE"].copy()
swe["bucket"] = "SWE_baseline"

COMBO = pd.concat([adj, swe], ignore_index=True)
COMBO["era"] = np.where(
    COMBO["period"].isin(["2024-01", "2024-04"]), "pre_2024", "post_2026"
)

METRICS = [
    "ai_strict_binary",
    "ai_broad_binary",
    "tech_count",
    "ai_tech_count",
    "org_scope_count",
    "requirement_breadth",
    "desc_len_chars",
]

group_agg = (
    COMBO[COMBO["bucket"].notna()]
    .groupby(["bucket", "era"], observed=True)[METRICS]
    .agg(["mean", "count"])
    .reset_index()
)

# Flatten columns
group_agg.columns = [
    "_".join([c for c in col if c]).rstrip("_") for col in group_agg.columns.values
]
group_agg = group_agg.rename(columns={"bucket": "bucket", "era": "era"})

# Pivot pre/post and compute change
pivoted = group_agg.copy()
pivot_records = []
for bucket in pivoted["bucket"].unique():
    pre = pivoted[(pivoted["bucket"] == bucket) & (pivoted["era"] == "pre_2024")]
    post = pivoted[(pivoted["bucket"] == bucket) & (pivoted["era"] == "post_2026")]
    if pre.empty or post.empty:
        continue
    rec = {"bucket": bucket, "n_pre": pre["ai_strict_binary_count"].values[0], "n_post": post["ai_strict_binary_count"].values[0]}
    for m in METRICS:
        rec[f"pre_{m}"] = pre[f"{m}_mean"].values[0]
        rec[f"post_{m}"] = post[f"{m}_mean"].values[0]
        rec[f"delta_{m}"] = rec[f"post_{m}"] - rec[f"pre_{m}"]
    pivot_records.append(rec)

out = pd.DataFrame(pivot_records).sort_values("delta_ai_strict_binary", ascending=False)
out.to_csv(TAB / "T18_adjacent_title_trajectories.csv", index=False)

# Distance-to-SWE: how close each adjacent title is to SWE on key metrics
swe_baseline = out[out["bucket"] == "SWE_baseline"].iloc[0]
closeness_records = []
for _, r in out.iterrows():
    if r["bucket"] == "SWE_baseline":
        continue
    rec = {"bucket": r["bucket"]}
    for m in ["ai_strict_binary", "tech_count", "org_scope_count", "requirement_breadth"]:
        rec[f"pre_gap_{m}"] = swe_baseline[f"pre_{m}"] - r[f"pre_{m}"]
        rec[f"post_gap_{m}"] = swe_baseline[f"post_{m}"] - r[f"post_{m}"]
        rec[f"gap_delta_{m}"] = rec[f"post_gap_{m}"] - rec[f"pre_gap_{m}"]
    closeness_records.append(rec)

closeness = pd.DataFrame(closeness_records).sort_values("post_gap_ai_strict_binary")
closeness.to_csv(TAB / "T18_adjacent_closeness_to_swe.csv", index=False)

print("=== Adjacent title trajectories ===")
print(out[[
    "bucket", "n_pre", "n_post",
    "pre_ai_strict_binary", "post_ai_strict_binary", "delta_ai_strict_binary",
    "pre_tech_count", "post_tech_count", "delta_tech_count",
    "pre_requirement_breadth", "post_requirement_breadth", "delta_requirement_breadth",
]].to_string())

print("\n=== Closeness to SWE (gap = SWE - bucket) ===")
print(closeness.to_string())
