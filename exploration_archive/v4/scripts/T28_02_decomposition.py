#!/usr/bin/env python3
"""
T28 step 2: Domain-stratified scope decomposition.

Merges T11 features, T28 propagated archetype labels, plus a few extra
columns from unified.parquet for aggregator exclusion and the dedup-safe
period split. Produces:

- Archetype × period distribution, AI/ML trend
- Within/between/interaction decomposition of entry-share change under
  two operationalizations: (combined best-available) and (YOE proxy).
- Per-archetype scope-feature deltas (requirement_breadth, tech_count,
  scope_count, ai_mention, credential_stack_depth, mgmts_mentor).
- Per-archetype junior-vs-senior comparison.
- Per-archetype senior-tier mentoring vs people-management shift.
- AI/ML archetype deep dive.
- Entry share under combined column and YOE proxy within AI/ML.

Reports tables as CSV under exploration/tables/T28/.
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import duckdb

OUT_T = "exploration/tables/T28"
os.makedirs(OUT_T, exist_ok=True)

# ---------------------------------------------------------------------------
# Load T11 features (63,294 rows) + T28 archetype assignments
# ---------------------------------------------------------------------------
feat = pd.read_parquet("exploration/tables/T11/T11_features.parquet")
arch = pd.read_parquet(f"{OUT_T}/archetype_assignments.parquet")

df = feat.merge(arch[["uid", "archetype", "archetype_name", "is_artifact"]], on="uid", how="left")
assert len(df) == 63294
print(f"Joined features + archetypes: {len(df):,}")

# ---------------------------------------------------------------------------
# Also need: description_hash (for dedup), yoe_extracted already present,
#            is_aggregator (already in feat)
# ---------------------------------------------------------------------------
con = duckdb.connect()
extra = con.execute("""
    SELECT uid, description_hash
    FROM read_parquet('data/unified.parquet')
    WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true
""").df()
df = df.merge(extra, on="uid", how="left")
print(f"With description_hash: {df['description_hash'].notna().mean():.3f} coverage")

# ---------------------------------------------------------------------------
# Period label: 2024 vs 2026 (using existing year col)
# ---------------------------------------------------------------------------
df["period2"] = np.where(df["year"] == "2024", "2024", "2026")

# Exclude noise archetype (-1) and artifact archetypes (14, 20, 21) for
# domain-level stratified analyses, but keep aggregator-excluded variants.
df["archetype_clean"] = df["archetype"]
df["is_noise"] = df["archetype"] == -1

# Canonical "domain set" for reporting: exclude artifacts and noise
df_dom = df[~df["is_artifact"] & ~df["is_noise"]].copy()

# Also: exclude aggregators for scope-feature analyses (per sensitivity (a))
df_dom_noagg = df_dom[~df_dom["is_aggregator"]].copy()

print(f"\nCorpus: {len(df):,}")
print(f"  domain (no noise, no artifact): {len(df_dom):,}")
print(f"  + no aggregator: {len(df_dom_noagg):,}")

# ---------------------------------------------------------------------------
# Seniority operationalizations
# ---------------------------------------------------------------------------
# combined best-available already in feat as seniority_best_available
# YOE proxy: yoe <= 2 -> entry-proxy
df["yoe_proxy"] = np.where(df["yoe_extracted"].isna(), "unknown",
                           np.where(df["yoe_extracted"] <= 2, "entry_proxy",
                                    np.where(df["yoe_extracted"] <= 4, "mid_proxy", "senior_proxy")))
df_dom["yoe_proxy"] = np.where(df_dom["yoe_extracted"].isna(), "unknown",
                               np.where(df_dom["yoe_extracted"] <= 2, "entry_proxy",
                                        np.where(df_dom["yoe_extracted"] <= 4, "mid_proxy", "senior_proxy")))
df_dom_noagg["yoe_proxy"] = df_dom["yoe_proxy"].loc[df_dom_noagg.index]

# ===========================================================================
# STEP 1: Archetype distribution by period, AI/ML trend confirmation
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 1: Archetype distribution by period")
print("=" * 70)

# Total SWE corpus (with noise + artifacts for honest total share)
total_by_period = df.groupby("period2").size().rename("total")
print(f"Total: {total_by_period.to_dict()}")

# 1a. Share of each archetype within total SWE (all sources pooled)
share_tbl = (
    df.groupby(["period2", "archetype", "archetype_name"])
    .size()
    .reset_index(name="n")
)
share_tbl["share"] = share_tbl["n"] / share_tbl["period2"].map(total_by_period)
pivot_share = share_tbl.pivot_table(
    index=["archetype", "archetype_name"], columns="period2", values="share", fill_value=0.0
)
pivot_share["delta_pp"] = (pivot_share["2026"] - pivot_share["2024"]) * 100
pivot_share = pivot_share.sort_values("delta_pp", ascending=False)
pivot_share.to_csv(f"{OUT_T}/archetype_share_by_period_all.csv")
print("\nArchetype share shift (all sources, incl artifacts and noise):")
print((pivot_share * 100).round(2).to_string())

# 1b. Same but restricted to domain (no artifacts/noise)
total_dom = df_dom.groupby("period2").size().rename("total")
share_dom = (
    df_dom.groupby(["period2", "archetype", "archetype_name"])
    .size()
    .reset_index(name="n")
)
share_dom["share"] = share_dom["n"] / share_dom["period2"].map(total_dom)
pivot_dom = share_dom.pivot_table(
    index=["archetype", "archetype_name"], columns="period2", values="share", fill_value=0.0
)
pivot_dom["delta_pp"] = (pivot_dom["2026"] - pivot_dom["2024"]) * 100
pivot_dom = pivot_dom.sort_values("delta_pp", ascending=False)
pivot_dom.to_csv(f"{OUT_T}/archetype_share_by_period_domain.csv")
print("\nArchetype share shift (domain only):")
print((pivot_dom * 100).round(2).to_string())

# Confirm T09's AI/ML finding: archetype 1 is AI/ML
aiml_row = pivot_dom.loc[(1, pivot_dom.reset_index()[pivot_dom.reset_index()["archetype"] == 1]["archetype_name"].iloc[0])]
print(f"\nAI/ML (archetype 1) domain shift: {aiml_row['2024']*100:.2f}% -> {aiml_row['2026']*100:.2f}% (+{aiml_row['delta_pp']:.2f}pp)")

# ===========================================================================
# STEP 2: Within/Between/Interaction decomposition of entry-share change
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 2: Within-Between-Interaction decomposition of entry-share")
print("=" * 70)

def decompose_entry(df_use, period_col, arch_col, entry_mask_fn, label):
    """
    Classic 3-way decomposition: Delta = Within + Between + Interaction.

      Delta = sum_j (p_j_26 * r_j_26) - sum_j (p_j_24 * r_j_24)
      Within  = sum_j p_j_24 * (r_j_26 - r_j_24)
      Between = sum_j (p_j_26 - p_j_24) * r_j_24
      Interaction = sum_j (p_j_26 - p_j_24) * (r_j_26 - r_j_24)

    where p_j is archetype share within period, r_j is entry-rate
    within archetype×period.
    """
    d = df_use.copy()
    d["is_entry"] = entry_mask_fn(d)
    # per-archetype rate per period
    agg = d.groupby([period_col, arch_col]).agg(
        n=("is_entry", "size"),
        entries=("is_entry", "sum"),
    ).reset_index()
    agg["rate"] = agg["entries"] / agg["n"]
    totals = agg.groupby(period_col)["n"].sum()
    agg["share"] = agg.apply(lambda r: r["n"] / totals[r[period_col]], axis=1)

    wide = agg.pivot_table(index=arch_col, columns=period_col, values=["rate", "share"], fill_value=np.nan)
    # Rates: fill missing archetype×period with 0 rate and 0 share (meaning absent)
    rate_24 = wide["rate"].get("2024", pd.Series(0, index=wide.index)).fillna(0)
    rate_26 = wide["rate"].get("2026", pd.Series(0, index=wide.index)).fillna(0)
    share_24 = wide["share"].get("2024", pd.Series(0, index=wide.index)).fillna(0)
    share_26 = wide["share"].get("2026", pd.Series(0, index=wide.index)).fillna(0)

    aggregate_24 = (share_24 * rate_24).sum()
    aggregate_26 = (share_26 * rate_26).sum()
    delta_total = aggregate_26 - aggregate_24
    within = (share_24 * (rate_26 - rate_24)).sum()
    between = ((share_26 - share_24) * rate_24).sum()
    interaction = ((share_26 - share_24) * (rate_26 - rate_24)).sum()

    summary = {
        "label": label,
        "aggregate_2024_pct": round(aggregate_24 * 100, 3),
        "aggregate_2026_pct": round(aggregate_26 * 100, 3),
        "delta_pp": round(delta_total * 100, 3),
        "within_pp": round(within * 100, 3),
        "between_pp": round(between * 100, 3),
        "interaction_pp": round(interaction * 100, 3),
        "n_2024": int(d[d[period_col] == "2024"].shape[0]),
        "n_2026": int(d[d[period_col] == "2026"].shape[0]),
    }
    # Per-archetype contributions
    per_arch = pd.DataFrame({
        "archetype": wide.index,
        "share_2024": share_24.values,
        "share_2026": share_26.values,
        "rate_2024": rate_24.values,
        "rate_2026": rate_26.values,
    })
    per_arch["within_contrib_pp"] = (share_24 * (rate_26 - rate_24)).values * 100
    per_arch["between_contrib_pp"] = ((share_26 - share_24) * rate_24).values * 100
    per_arch["interaction_contrib_pp"] = ((share_26 - share_24) * (rate_26 - rate_24)).values * 100
    return summary, per_arch


# Operationalization A: combined best-available entry
def mask_combined_entry(d):
    return (d["seniority_best_available"] == "entry")

# Operationalization B: YOE proxy
def mask_yoe_entry(d):
    return (d["yoe_extracted"].notna() & (d["yoe_extracted"] <= 2))

# Operationalization C: augmented best-available (uses seniority_final fallback)
def mask_aug_entry(d):
    return (d["seniority_best_available_aug"] == "entry")

decomps = []
all_per_arch = {}

for label, mask_fn in [
    ("combined_best_available", mask_combined_entry),
    ("combined_augmented", mask_aug_entry),
    ("yoe_proxy_leq2", mask_yoe_entry),
]:
    for subset_label, sub in [("domain", df_dom), ("domain_noagg", df_dom_noagg)]:
        summary, per_arch = decompose_entry(sub, "period2", "archetype", mask_fn, f"{label}__{subset_label}")
        decomps.append(summary)
        all_per_arch[f"{label}__{subset_label}"] = per_arch
        print(f"\n{summary['label']}:")
        print(f"  aggregate: {summary['aggregate_2024_pct']:.2f}% -> {summary['aggregate_2026_pct']:.2f}% (delta={summary['delta_pp']:+.2f}pp)")
        print(f"  within-domain: {summary['within_pp']:+.3f}pp")
        print(f"  between-domain: {summary['between_pp']:+.3f}pp")
        print(f"  interaction: {summary['interaction_pp']:+.3f}pp")

decomp_df = pd.DataFrame(decomps)
decomp_df.to_csv(f"{OUT_T}/entry_share_decomposition.csv", index=False)

# Save per-archetype contributions for the primary operationalization
all_per_arch["combined_augmented__domain"].merge(
    df_dom[["archetype", "archetype_name"]].drop_duplicates(), on="archetype"
).to_csv(f"{OUT_T}/entry_share_per_archetype_contrib.csv", index=False)

# ===========================================================================
# STEP 3: Domain-stratified scope inflation
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 3: Domain-stratified scope inflation")
print("=" * 70)

scope_metrics = [
    "requirement_breadth", "tech_count", "scope_count", "ai_mention",
    "credential_stack_depth", "text_len",
]

# Use df_dom_noagg (aggregator-excluded) for scope analyses
scope_rows = []
for (arch, name), sub in df_dom_noagg.groupby(["archetype", "archetype_name"]):
    for metric in scope_metrics:
        v24 = sub[sub["period2"] == "2024"][metric].dropna()
        v26 = sub[sub["period2"] == "2026"][metric].dropna()
        if len(v24) < 20 or len(v26) < 20:
            continue
        scope_rows.append({
            "archetype": arch,
            "archetype_name": name,
            "metric": metric,
            "n_2024": len(v24),
            "n_2026": len(v26),
            "mean_2024": v24.mean(),
            "mean_2026": v26.mean(),
            "median_2024": v24.median(),
            "median_2026": v26.median(),
            "delta_mean": v26.mean() - v24.mean(),
            "delta_pct": (v26.mean() - v24.mean()) / v24.mean() * 100 if v24.mean() != 0 else np.nan,
        })

scope_df = pd.DataFrame(scope_rows)
scope_df.to_csv(f"{OUT_T}/scope_inflation_by_archetype.csv", index=False)

# Pivot for readability: one table per metric
for metric in scope_metrics:
    sub = scope_df[scope_df["metric"] == metric].sort_values("delta_pct", ascending=False)
    print(f"\n--- {metric} ---")
    print(sub[["archetype_name", "n_2024", "n_2026", "mean_2024", "mean_2026", "delta_pct"]].round(2).to_string(index=False))

# Credential stack depth 7+ indicator: the T11 headline metric
df_dom_noagg["cred_stack_7plus"] = df_dom_noagg["credential_stack_depth"] >= 7
cred_rows = []
for (arch, name), sub in df_dom_noagg.groupby(["archetype", "archetype_name"]):
    v24 = sub[sub["period2"] == "2024"]["cred_stack_7plus"]
    v26 = sub[sub["period2"] == "2026"]["cred_stack_7plus"]
    if len(v24) < 20 or len(v26) < 20:
        continue
    cred_rows.append({
        "archetype": arch,
        "archetype_name": name,
        "n_2024": len(v24),
        "n_2026": len(v26),
        "pct_2024": v24.mean() * 100,
        "pct_2026": v26.mean() * 100,
        "ratio": (v26.mean() / v24.mean()) if v24.mean() > 0 else np.nan,
        "delta_pp": (v26.mean() - v24.mean()) * 100,
    })
cred_df = pd.DataFrame(cred_rows).sort_values("delta_pp", ascending=False)
cred_df.to_csv(f"{OUT_T}/credential_stack_7_by_archetype.csv", index=False)
print("\nCredential stack depth >=7 by archetype:")
print(cred_df.round(2).to_string(index=False))

# ===========================================================================
# STEP 4: Junior vs Senior content WITHIN each archetype
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 4: Junior vs Senior within each archetype (entry-proxy YOE vs senior-proxy YOE)")
print("=" * 70)

# Use YOE-based proxy because it's label-independent and has more data
df_dom_noagg["yoe_bucket"] = np.where(
    df_dom_noagg["yoe_extracted"].isna(), "unknown",
    np.where(df_dom_noagg["yoe_extracted"] <= 2, "entry_proxy",
             np.where(df_dom_noagg["yoe_extracted"] >= 5, "senior_proxy", "mid_proxy"))
)

within_metrics = ["requirement_breadth", "ai_mention", "scope_count", "mgmts_mentor"]
jr_sr_rows = []
for (arch, name), sub in df_dom_noagg.groupby(["archetype", "archetype_name"]):
    for period in ["2024", "2026"]:
        ssub = sub[sub["period2"] == period]
        entry = ssub[ssub["yoe_bucket"] == "entry_proxy"]
        senior = ssub[ssub["yoe_bucket"] == "senior_proxy"]
        if len(entry) < 15 or len(senior) < 15:
            continue
        row = {"archetype": arch, "archetype_name": name, "period": period,
               "n_entry": len(entry), "n_senior": len(senior)}
        for m in within_metrics:
            row[f"entry_{m}"] = entry[m].mean()
            row[f"senior_{m}"] = senior[m].mean()
            row[f"gap_{m}"] = senior[m].mean() - entry[m].mean()
        jr_sr_rows.append(row)

jr_sr_df = pd.DataFrame(jr_sr_rows)
jr_sr_df.to_csv(f"{OUT_T}/junior_senior_within_archetype.csv", index=False)

# For each archetype, show whether junior-senior gap is closing
gap_changes = []
for (arch, name), sub in jr_sr_df.groupby(["archetype", "archetype_name"]):
    if len(sub) != 2:
        continue
    r24 = sub[sub["period"] == "2024"].iloc[0]
    r26 = sub[sub["period"] == "2026"].iloc[0]
    row = {"archetype": arch, "archetype_name": name,
           "n_2024_entry": r24["n_entry"], "n_2024_senior": r24["n_senior"],
           "n_2026_entry": r26["n_entry"], "n_2026_senior": r26["n_senior"]}
    for m in within_metrics:
        row[f"gap_{m}_2024"] = r24[f"gap_{m}"]
        row[f"gap_{m}_2026"] = r26[f"gap_{m}"]
        row[f"gap_{m}_change"] = r26[f"gap_{m}"] - r24[f"gap_{m}"]
    gap_changes.append(row)
gap_df = pd.DataFrame(gap_changes)
gap_df.to_csv(f"{OUT_T}/junior_senior_gap_changes.csv", index=False)
print("\nJunior-senior gap changes (senior-entry, by metric, 2024 vs 2026):")
print(gap_df.round(2).to_string(index=False))

# ===========================================================================
# STEP 5: Senior archetype shift — mentoring vs people-management by domain
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 5: Senior-tier mentoring vs people-management shift by archetype")
print("=" * 70)

# Strict people-manager indicator: direct_reports, performance_review,
# people_manager, headcount. From T11's mgmts_* family.
strict_pm_cols = ["mgmts_direct_reports", "mgmts_performance_review",
                  "mgmts_people_manager", "mgmts_headcount"]
df_dom_noagg["strict_people_mgr"] = (df_dom_noagg[strict_pm_cols].sum(axis=1) > 0).astype(int)

senior_mask = (df_dom_noagg["yoe_extracted"] >= 5)
senior_df = df_dom_noagg[senior_mask]

sen_rows = []
for (arch, name), sub in senior_df.groupby(["archetype", "archetype_name"]):
    for period in ["2024", "2026"]:
        ssub = sub[sub["period2"] == period]
        if len(ssub) < 30:
            continue
        sen_rows.append({
            "archetype": arch, "archetype_name": name, "period": period,
            "n": len(ssub),
            "mentor_pct": ssub["mgmts_mentor"].mean() * 100,
            "people_mgr_pct": ssub["strict_people_mgr"].mean() * 100,
        })
sen_df = pd.DataFrame(sen_rows)
sen_pivot = sen_df.pivot_table(
    index=["archetype", "archetype_name"],
    columns="period",
    values=["mentor_pct", "people_mgr_pct", "n"],
    fill_value=np.nan,
)
sen_pivot.columns = [f"{a}_{b}" for a, b in sen_pivot.columns]
sen_pivot["mentor_delta_pp"] = sen_pivot["mentor_pct_2026"] - sen_pivot["mentor_pct_2024"]
sen_pivot["pmgr_delta_pp"] = sen_pivot["people_mgr_pct_2026"] - sen_pivot["people_mgr_pct_2024"]
sen_pivot = sen_pivot.sort_values("mentor_delta_pp", ascending=False)
sen_pivot.to_csv(f"{OUT_T}/senior_mentor_vs_pmgr_by_archetype.csv")
print(sen_pivot.round(2).to_string())

# ===========================================================================
# STEP 6: AI/ML archetype deep dive
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 6: AI/ML (archetype 1) deep dive")
print("=" * 70)

aiml = df_dom_noagg[df_dom_noagg["archetype"] == 1].copy()
print(f"AI/ML rows: {len(aiml):,} ({(aiml['period2'] == '2024').sum():,} 2024 / {(aiml['period2'] == '2026').sum():,} 2026)")

# Dedup by description_hash within company for employer counts
dedup = aiml.drop_duplicates(["company_name_canonical", "description_hash", "period2"])
top_emp = dedup.groupby(["period2", "company_name_canonical"]).size().reset_index(name="n")
top_emp_24 = top_emp[top_emp["period2"] == "2024"].nlargest(15, "n")
top_emp_26 = top_emp[top_emp["period2"] == "2026"].nlargest(15, "n")
print("\nTop AI/ML employers 2024 (dedup within company by desc_hash):")
print(top_emp_24.to_string(index=False))
print("\nTop AI/ML employers 2026:")
print(top_emp_26.to_string(index=False))
top_emp_24.to_csv(f"{OUT_T}/aiml_top_employers_2024.csv", index=False)
top_emp_26.to_csv(f"{OUT_T}/aiml_top_employers_2026.csv", index=False)

# AI/ML entry/mid/senior mix under combined and YOE proxy
aiml_ops = []
for op_name, op_col in [("combined_augmented", "seniority_best_available_aug"),
                        ("combined", "seniority_best_available")]:
    for period in ["2024", "2026"]:
        sub = aiml[aiml["period2"] == period]
        counts = sub[op_col].value_counts(dropna=False).to_dict()
        total = len(sub)
        row = {"op": op_name, "period": period, "n": total}
        for k, v in counts.items():
            row[f"{k}_n"] = v
            row[f"{k}_pct"] = v / total * 100
        aiml_ops.append(row)

# YOE proxy
for period in ["2024", "2026"]:
    sub = aiml[aiml["period2"] == period]
    row = {"op": "yoe_proxy", "period": period, "n": len(sub)}
    known = sub[sub["yoe_extracted"].notna()]
    row["yoe_known"] = len(known)
    row["yoe_le2_pct"] = (known["yoe_extracted"] <= 2).mean() * 100 if len(known) > 0 else np.nan
    row["yoe_3_4_pct"] = ((known["yoe_extracted"] >= 3) & (known["yoe_extracted"] <= 4)).mean() * 100 if len(known) > 0 else np.nan
    row["yoe_ge5_pct"] = (known["yoe_extracted"] >= 5).mean() * 100 if len(known) > 0 else np.nan
    aiml_ops.append(row)

aiml_ops_df = pd.DataFrame(aiml_ops)
aiml_ops_df.to_csv(f"{OUT_T}/aiml_seniority_mix.csv", index=False)
print("\nAI/ML seniority mix:")
print(aiml_ops_df.round(2).to_string(index=False))

# AI/ML vs rest-of-market entry-share comparison
print("\nAI/ML vs rest-of-market entry share (yoe<=2, among YOE-known):")
for period in ["2024", "2026"]:
    for subset_name, sub in [("aiml", aiml[aiml["period2"] == period]),
                              ("rest", df_dom_noagg[(df_dom_noagg["archetype"] != 1) & (df_dom_noagg["period2"] == period)])]:
        yk = sub[sub["yoe_extracted"].notna()]
        pct = (yk["yoe_extracted"] <= 2).mean() * 100 if len(yk) > 0 else np.nan
        print(f"  {period} {subset_name}: n_known={len(yk):,}  entry-proxy={pct:.2f}%")

# AI/ML scope metrics vs rest
print("\nAI/ML vs rest scope metrics (2026):")
for m in scope_metrics + ["mgmts_mentor"]:
    a_v = aiml[aiml["period2"] == "2026"][m].mean()
    r_v = df_dom_noagg[(df_dom_noagg["archetype"] != 1) & (df_dom_noagg["period2"] == "2026")][m].mean()
    print(f"  {m}: aiml={a_v:.3f} rest={r_v:.3f}")

# Continuing company concentration check
print(f"\nAI/ML top-20 company share of total AI/ML rows:")
emp_shares = aiml.groupby(["period2", "company_name_canonical"]).size().reset_index(name="n")
for period in ["2024", "2026"]:
    sub = emp_shares[emp_shares["period2"] == period].sort_values("n", ascending=False)
    total = sub["n"].sum()
    top20 = sub.head(20)["n"].sum()
    print(f"  {period}: top-20 = {top20}/{total} = {top20/total*100:.2f}%")

print("\nT28 step 2 complete.")
