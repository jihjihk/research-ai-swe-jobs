"""T09 Step 10 + sensitivities: full-corpus labeling, sensitivity (a) aggregator
exclusion, sensitivity (g) SWE classification tier.

Outputs:
  - exploration/artifacts/shared/swe_archetype_labels.parquet  — CRITICAL downstream artifact
  - exploration/tables/T09/sensitivity_aggregator_exclusion.csv
  - exploration/tables/T09/sensitivity_swe_tier.csv
  - exploration/tables/T09/archetype_tech_profile.csv
  - exploration/tables/T09/full_corpus_archetype_share_by_period.csv
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path("/home/jihgaboot/gabor/job-research")
ART = ROOT / "exploration/artifacts"
SHARED = ART / "shared"
TABLES = ROOT / "exploration/tables/T09"
TABLES.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Load
sample = pd.read_parquet(ART / "T09_sample_characterized.parquet")
X_sample = np.load(ART / "T09_sample_embeddings.npy")
log(f"Sample: {len(sample):,}")

# ---------------------------------------------------------------------------
# Step 10: Full-corpus nearest-neighbor labeling
# ---------------------------------------------------------------------------
# We have BERTopic cluster assignments for 7,730 sample rows.
# We label the full SWE LinkedIn corpus (63,701) using k-NN over shared embeddings
# for `text_source='llm'` rows (34,099 embedded). Non-embedded rows get archetype = -2 (no_text).

log("Loading shared embeddings + index")
emb_index = pd.read_parquet(SHARED / "swe_embedding_index.parquet")
all_emb = np.load(SHARED / "swe_embeddings.npy")
log(f"Shared embeddings: {all_emb.shape}")

# Use only non-noise training rows to seed NN
train_mask = sample["bertopic_primary"] != -1
X_train = X_sample[train_mask.values]
y_train = sample.loc[train_mask, "bertopic_primary"].values
log(f"Non-noise training rows: {len(X_train):,}")

# Restrict to k=5 NN voting for robustness
nn = NearestNeighbors(n_neighbors=5, metric="cosine", n_jobs=-1)
nn.fit(X_train)

log("KNN over full embedded corpus (34,099 rows)")
_, idx = nn.kneighbors(all_emb)
labels_full = np.zeros(len(all_emb), dtype=int)
for i in range(len(all_emb)):
    votes = y_train[idx[i]]
    labels_full[i] = np.bincount(votes).argmax()

full_emb_labels = pd.DataFrame({"uid": emb_index["uid"], "archetype": labels_full})

# Load full cleaned text (for all 63,701) to get uids for non-embedded rows
log("Loading full cleaned-text uids")
con = duckdb.connect()
all_uids = con.execute(
    "SELECT uid, text_source FROM 'exploration/artifacts/shared/swe_cleaned_text.parquet'"
).df()
log(f"Full SWE uids: {len(all_uids):,}")

# Merge: embedded rows get labels from kNN; raw-text rows get archetype = -2 (no_text)
merged = all_uids.merge(full_emb_labels, on="uid", how="left")
merged["archetype"] = merged["archetype"].fillna(-2).astype(int)

# Load archetype names
names = pd.read_csv(TABLES / "archetype_names.csv")
name_map = dict(zip(names["archetype"], names["archetype_name"]))
name_map[-2] = "No text / raw-only (unlabeled)"
merged["archetype_name"] = merged["archetype"].map(name_map)
log(f"Coverage by archetype assignment status:")
log(f"  labeled: {(merged['archetype'] >= 0).sum():,} / {len(merged):,} = {(merged['archetype'] >= 0).mean():.1%}")
log(f"  no_text: {(merged['archetype'] == -2).sum():,}")

out = merged[["uid", "archetype", "archetype_name"]].copy()
out.to_parquet(SHARED / "swe_archetype_labels.parquet", index=False)
log(f"SAVED: {SHARED / 'swe_archetype_labels.parquet'} rows={len(out):,}")

# ---------------------------------------------------------------------------
# Full corpus archetype share by period (sanity check vs sample)
# ---------------------------------------------------------------------------
log("Full corpus archetype share by period")
full_with_period = con.execute(
    """
    SELECT s.uid, s.period, s.source, s.is_aggregator, s.swe_classification_tier, s.seniority_3level,
           s.yoe_extracted, a.archetype, a.archetype_name
    FROM 'exploration/artifacts/shared/swe_cleaned_text.parquet' s
    JOIN 'exploration/artifacts/shared/swe_archetype_labels.parquet' a USING (uid)
    WHERE s.text_source = 'llm' AND a.archetype >= 0
    """
).df()

# Period group
full_with_period["period_group"] = np.where(
    full_with_period["period"].str.startswith("2024"),
    "2024",
    full_with_period["period"],
)

full_share = (
    full_with_period.groupby(["archetype", "archetype_name", "period_group"])
    .size()
    .reset_index(name="n")
)
full_totals = full_with_period.groupby("period_group").size().to_dict()
full_share["share"] = full_share.apply(lambda r: r["n"] / full_totals[r["period_group"]], axis=1)
pivot = full_share.pivot_table(index=["archetype", "archetype_name"], columns="period_group", values="share").reset_index()
pivot["delta_2024_to_2026avg"] = (pivot.get("2026-03", 0) + pivot.get("2026-04", 0)) / 2 - pivot.get("2024", 0)
pivot = pivot.sort_values("delta_2024_to_2026avg", ascending=False)
pivot.to_csv(TABLES / "full_corpus_archetype_share_by_period.csv", index=False)
print(pivot.to_string(index=False))

# ---------------------------------------------------------------------------
# Full-corpus per-archetype entry share by period (THE headline numbers)
# ---------------------------------------------------------------------------
log("Full-corpus per-archetype entry shares")
entry_rows = []
for arch, grp in full_with_period.groupby(["archetype", "archetype_name"]):
    for pg, pgrp in grp.groupby("period_group"):
        known = pgrp[pgrp["seniority_3level"].isin(["junior", "mid", "senior"])]
        yoe_nn = pgrp[pgrp["yoe_extracted"].notna()]
        entry_rows.append({
            "archetype": arch[0],
            "archetype_name": arch[1],
            "period_group": pg,
            "n_total": len(pgrp),
            "n_known_sen": len(known),
            "junior_share": (known["seniority_3level"] == "junior").sum() / max(1, len(known)),
            "n_yoe": len(yoe_nn),
            "yoe_le2_share": (yoe_nn["yoe_extracted"] <= 2).sum() / max(1, len(yoe_nn)),
        })
entry_full = pd.DataFrame(entry_rows)
entry_full.to_csv(TABLES / "full_corpus_entry_share_by_archetype.csv", index=False)

# ---------------------------------------------------------------------------
# Sensitivity (a): aggregator exclusion
# ---------------------------------------------------------------------------
log("Sensitivity: aggregator exclusion")
agg_rows = []
noagg = full_with_period[~full_with_period["is_aggregator"].fillna(False)]
totals_na = noagg.groupby("period_group").size().to_dict()
for arch, grp in noagg.groupby(["archetype", "archetype_name"]):
    for pg, pgrp in grp.groupby("period_group"):
        agg_rows.append({
            "archetype": arch[0],
            "archetype_name": arch[1],
            "period_group": pg,
            "n": len(pgrp),
            "share_noagg": len(pgrp) / max(1, totals_na.get(pg, 1)),
        })
pd.DataFrame(agg_rows).to_csv(TABLES / "sensitivity_aggregator_exclusion.csv", index=False)

# Compare: share before/after aggregator exclusion
with_agg_share = full_share.set_index(["archetype", "archetype_name", "period_group"])["share"]
compare = pd.DataFrame(agg_rows).set_index(["archetype", "archetype_name", "period_group"])
compare["share_all"] = with_agg_share
compare["delta_agg_exclusion"] = compare["share_noagg"] - compare["share_all"]
compare.reset_index().to_csv(TABLES / "sensitivity_aggregator_compare.csv", index=False)

# ---------------------------------------------------------------------------
# Sensitivity (g): SWE classification tier (exclude title_lookup_llm)
# ---------------------------------------------------------------------------
log("Sensitivity: exclude title_lookup_llm SWE tier")
notier = full_with_period[full_with_period["swe_classification_tier"] != "title_lookup_llm"]
totals_nt = notier.groupby("period_group").size().to_dict()
tier_rows = []
for arch, grp in notier.groupby(["archetype", "archetype_name"]):
    for pg, pgrp in grp.groupby("period_group"):
        tier_rows.append({
            "archetype": arch[0],
            "archetype_name": arch[1],
            "period_group": pg,
            "n": len(pgrp),
            "share_strict_tier": len(pgrp) / max(1, totals_nt.get(pg, 1)),
        })
pd.DataFrame(tier_rows).to_csv(TABLES / "sensitivity_swe_tier.csv", index=False)

# ---------------------------------------------------------------------------
# Archetype tech profile (top-5 technologies per archetype from full corpus)
# ---------------------------------------------------------------------------
log("Archetype tech profile on full corpus")
tech_full = con.execute(
    """
    SELECT t.*, a.archetype, a.archetype_name
    FROM 'exploration/artifacts/shared/swe_tech_matrix.parquet' t
    JOIN 'exploration/artifacts/shared/swe_archetype_labels.parquet' a USING (uid)
    WHERE a.archetype >= 0
    """
).df()

tech_cols = [c for c in tech_full.columns if c not in ("uid", "archetype", "archetype_name")]
tech_prof_rows = []
for arch, grp in tech_full.groupby(["archetype", "archetype_name"]):
    prev = grp[tech_cols].mean().sort_values(ascending=False)
    top10 = prev.head(10)
    tech_prof_rows.append({
        "archetype": arch[0],
        "archetype_name": arch[1],
        "n": len(grp),
        "top10_tech": " | ".join([f"{t}({p:.2f})" for t, p in top10.items()]),
    })
pd.DataFrame(tech_prof_rows).to_csv(TABLES / "archetype_tech_profile.csv", index=False)

log("DONE. Critical artifact saved to exploration/artifacts/shared/swe_archetype_labels.parquet")
