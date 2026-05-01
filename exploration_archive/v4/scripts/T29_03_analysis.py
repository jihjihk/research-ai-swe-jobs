#!/usr/bin:env python3
"""
T29 step 3: Composite authorship score, spot-check validation, period
distributions, per-company profile, correlations with Wave 2 metrics,
and the headline low-LLM-subset re-test of Wave 2 findings.

Uses features from T29_01 (raw description, feature-rich) and T29_02
(LLM-cleaned subset, text-source-controlled).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

OUT_T = "exploration/tables/T29"
OUT_F = "exploration/figures/T29"
os.makedirs(OUT_T, exist_ok=True)
os.makedirs(OUT_F, exist_ok=True)

import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150})

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
raw = pd.read_parquet(f"{OUT_T}/authorship_scores.parquet")
ctrl = pd.read_parquet(f"{OUT_T}/authorship_scores_llmcleaned.parquet")
print(f"Raw: {len(raw):,} rows   Text-source-controlled: {len(ctrl):,} rows")

# Merge T11 features for correlations
t11 = pd.read_parquet("exploration/tables/T11/T11_features.parquet")
t11_cols = ["uid", "text_len", "tech_count", "scope_count", "ai_mention",
            "requirement_breadth", "credential_stack_depth", "tech_density",
            "mgmts_mentor"]
raw = raw.merge(t11[t11_cols], on="uid", how="left")
ctrl = ctrl.merge(t11[t11_cols], on="uid", how="left")

# ---------------------------------------------------------------------------
# Composite score — z-standardized sum of key signals
# Use RAW features for company-level + headline subset. Only use features
# that are reasonably robust to ingestion format (vocab density, TTR,
# hedge phrases, closer count, comma density). Bullet/emdash density are
# too ingestion-sensitive, but we keep them as SEPARATE signals.
# ---------------------------------------------------------------------------
robust_features = ["llm_vocab_density_1k", "hedge_phrase_count", "comma_density_1k"]
# Lower TTR -> more LLM-like (uniform vocab), so invert
# 1 - ttr
raw["ttr_inv"] = 1.0 - raw["ttr"]
ctrl["ttr_inv"] = 1.0 - ctrl["ttr"]
robust_features_inv = robust_features + ["ttr_inv"]


def zscore_then_sum(df, cols):
    z = pd.DataFrame()
    for c in cols:
        x = df[c].astype(float)
        mu, sd = x.mean(), x.std()
        if sd < 1e-9:
            z[c] = 0
        else:
            z[c] = (x - mu) / sd
    return z.sum(axis=1)


raw["llm_score_robust"] = zscore_then_sum(raw, robust_features_inv)
ctrl["llm_score_robust"] = zscore_then_sum(ctrl, robust_features_inv)

# ---------------------------------------------------------------------------
# Step (2): Spot-check 25 high and 25 low by llm_vocab_density_1k
# (use only the robust signals; emit sample text)
# ---------------------------------------------------------------------------
print("\n=== SPOT-CHECK (raw corpus, sorted by llm_vocab_density) ===")
import duckdb
con = duckdb.connect()
# fetch 25 high and 25 low + 5 randomly by llm_vocab_count
sample_uids_high = raw.nlargest(25, "llm_vocab_count")["uid"].tolist()
sample_uids_low = raw[raw["llm_vocab_count"] == 0].sample(25, random_state=1)["uid"].tolist()

uids_all = sample_uids_high + sample_uids_low
q = f"""
SELECT uid, source, period, company_name_canonical, substring(description, 1, 400) as snippet
FROM read_parquet('data/unified.parquet')
WHERE uid IN ({','.join([f"'{u}'" for u in uids_all])})
"""
sam = con.execute(q).df()
sam = sam.merge(raw[["uid", "llm_vocab_count", "llm_vocab_density_1k", "year"]], on="uid")

# Save to CSV for audit
sam["tier"] = np.where(sam["uid"].isin(sample_uids_high), "high", "low")
sam.to_csv(f"{OUT_T}/spotcheck_samples.csv", index=False)
print("High LLM-vocab snippets (first 5):")
for _, r in sam[sam["tier"] == "high"].head(5).iterrows():
    print(f"  [{r['period']} {r['company_name_canonical'][:20]} vocab={r['llm_vocab_count']}]: {r['snippet'][:200]}")
print("\nLow LLM-vocab snippets (first 5):")
for _, r in sam[sam["tier"] == "low"].head(5).iterrows():
    print(f"  [{r['period']} {r['company_name_canonical'][:20] if pd.notna(r['company_name_canonical']) else 'na'} vocab={r['llm_vocab_count']}]: {r['snippet'][:200]}")

# ---------------------------------------------------------------------------
# Step (3): Save the per-posting scores (already done in step 1/2 — re-save
# with the composite score).
# ---------------------------------------------------------------------------
raw.to_parquet(f"{OUT_T}/authorship_scores.parquet", index=False)
ctrl.to_parquet(f"{OUT_T}/authorship_scores_llmcleaned.parquet", index=False)

# ---------------------------------------------------------------------------
# Step (4): Distribution by period — raw corpus vs text-source-controlled
# ---------------------------------------------------------------------------
print("\n=== DISTRIBUTION BY PERIOD ===")

# Drop agg for fair comparison
raw_noagg = raw[~raw["is_aggregator"]].copy()
ctrl_noagg = ctrl[~ctrl["is_aggregator"]].copy()

feature_cols = ["llm_vocab_density_1k", "emdash_density_1k", "bullet_density_1k",
                "ttr", "mean_sent_len_chars", "std_sent_len_chars", "mean_para_len",
                "paragraph_count", "hedge_phrase_count", "closer_count",
                "comma_density_1k", "exclaim_density_1k", "n_chars",
                "llm_score_robust"]

period_dists = []
for label, d in [("raw_all", raw), ("raw_noagg", raw_noagg),
                 ("ctrl_all", ctrl), ("ctrl_noagg", ctrl_noagg)]:
    for period in ["2024", "2026"]:
        sub = d[d["year"] == period]
        row = {"subset": label, "year": period, "n": len(sub)}
        for c in feature_cols:
            if c not in sub.columns:
                continue
            row[f"{c}_mean"] = sub[c].mean()
            row[f"{c}_std"] = sub[c].std()
            row[f"{c}_median"] = sub[c].median()
        period_dists.append(row)

pd_df = pd.DataFrame(period_dists)
pd_df.to_csv(f"{OUT_T}/period_distribution.csv", index=False)

print("\nKey metrics 2024 vs 2026 by subset (means):")
key = ["llm_vocab_density_1k", "emdash_density_1k", "ttr",
       "hedge_phrase_count", "comma_density_1k", "llm_score_robust"]
for label in ["raw_all", "raw_noagg", "ctrl_all", "ctrl_noagg"]:
    print(f"\n-- {label} --")
    sub = pd_df[pd_df["subset"] == label]
    for k in key:
        c = f"{k}_mean"
        if c not in sub.columns:
            continue
        r24 = sub[sub["year"] == "2024"][c].iloc[0]
        r26 = sub[sub["year"] == "2026"][c].iloc[0]
        delta = r26 - r24
        ratio = r26 / r24 if r24 != 0 else float("inf")
        print(f"  {k}: 2024={r24:.4f}  2026={r26:.4f}  Δ={delta:+.4f}  ratio={ratio:.2f}")

# Cross-posting variance (if LLMs write more, cross-posting vocab should uniformize)
print("\n=== CROSS-POSTING VARIANCE (robust score std across postings) ===")
for label, d in [("raw", raw), ("ctrl", ctrl)]:
    for period in ["2024", "2026"]:
        sub = d[d["year"] == period]
        std = sub["llm_score_robust"].std()
        mad = (sub["llm_score_robust"] - sub["llm_score_robust"].median()).abs().median()
        print(f"  {label} {period}: n={len(sub):,}  std={std:.3f}  mad={mad:.3f}")

# ---------------------------------------------------------------------------
# Step (5): Per-company profile
# ---------------------------------------------------------------------------
print("\n=== PER-COMPANY AUTHORSHIP ===")
# Top companies with >=5 postings in both periods
co_cnt = raw_noagg.groupby(["company_name_canonical", "year"]).size().unstack(fill_value=0)
co_cnt = co_cnt[(co_cnt.get("2024", 0) >= 5) & (co_cnt.get("2026", 0) >= 5)]
print(f"Companies with >=5 postings in both periods: {len(co_cnt)}")

def co_score(sub):
    return sub["llm_score_robust"].mean()

co_scores = raw_noagg[raw_noagg["company_name_canonical"].isin(co_cnt.index)].groupby(
    ["company_name_canonical", "year"]
)["llm_score_robust"].mean().unstack()
co_scores["n_2024"] = co_cnt["2024"]
co_scores["n_2026"] = co_cnt["2026"]
co_scores["delta"] = co_scores["2026"] - co_scores["2024"]
co_scores = co_scores.sort_values("delta", ascending=False)
co_scores.to_csv(f"{OUT_T}/per_company_authorship.csv")
print("\nTop 15 companies with LARGEST increase in LLM-score 2024->2026:")
print(co_scores.head(15).round(3).to_string())
print("\nTop 15 companies with LARGEST decrease (still 'human' in 2026):")
print(co_scores.tail(15).round(3).to_string())

# Companies LLM-ish in 2024 already (top scoring in 2024):
top_2024 = co_scores.dropna(subset=["2024"]).sort_values("2024", ascending=False).head(15)
print("\n2024 early-adopter candidates (highest llm_score in 2024):")
print(top_2024.round(3).to_string())

# ---------------------------------------------------------------------------
# Step (6): Correlation with Wave 2 findings (per-posting, pooled)
# ---------------------------------------------------------------------------
print("\n=== CORRELATION WITH WAVE 2 METRICS ===")
corr_cols = {
    "description_length": "text_len",
    "tech_density": "tech_density",
    "ai_mention": "ai_mention",
    "credential_stack_depth": "credential_stack_depth",
    "requirement_breadth": "requirement_breadth",
    "mgmts_mentor": "mgmts_mentor",
}
corr_rows = []
for year in ["2024", "2026"]:
    sub = raw[raw["year"] == year]
    for label, col in corr_cols.items():
        if col not in sub.columns:
            continue
        try:
            rho = sub[["llm_score_robust", col]].corr(method="spearman").iloc[0, 1]
            corr_rows.append({"year": year, "metric": label, "spearman": rho})
        except Exception:
            pass
corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(f"{OUT_T}/wave2_correlations.csv", index=False)
print(corr_df.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# Step (7): HEADLINE — low-LLM-subset re-test of Wave 2 findings
# Split low-LLM = bottom 50% of llm_score_robust within year
# ---------------------------------------------------------------------------
print("\n=== HEADLINE: Low-LLM subset re-test of Wave 2 findings ===")
# Define the low-LLM mask within-year (to avoid period-shift dominating)
raw["llm_quartile"] = raw.groupby("year")["llm_score_robust"].transform(
    lambda x: pd.qcut(x, 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
)

def headline_metrics(d):
    return {
        "n": len(d),
        "text_len_mean": d["text_len"].mean(),
        "cred_stack_7plus_pct": (d["credential_stack_depth"] >= 7).mean() * 100,
        "ai_mention_pct": (d["ai_mention"] > 0).mean() * 100,
        "tech_density_mean": d["tech_density"].mean(),
        "requirement_breadth_mean": d["requirement_breadth"].mean(),
    }

headline_rows = []
for subset_label, d in [("all", raw), ("q1_low_llm", raw[raw["llm_quartile"] == "Q1_low"]),
                         ("q4_high_llm", raw[raw["llm_quartile"] == "Q4_high"])]:
    for year in ["2024", "2026"]:
        sub = d[d["year"] == year]
        m = headline_metrics(sub)
        m["subset"] = subset_label
        m["year"] = year
        headline_rows.append(m)

headline_df = pd.DataFrame(headline_rows)
headline_df.to_csv(f"{OUT_T}/headline_retest.csv", index=False)
print(headline_df.round(3).to_string(index=False))

# Compute deltas
print("\nDeltas (2026 - 2024) per subset:")
pv = headline_df.pivot_table(index="subset", columns="year",
                             values=["text_len_mean", "cred_stack_7plus_pct",
                                     "ai_mention_pct", "tech_density_mean",
                                     "requirement_breadth_mean"])
print(pv.round(3).to_string())

for metric in ["text_len_mean", "cred_stack_7plus_pct", "ai_mention_pct",
               "tech_density_mean", "requirement_breadth_mean"]:
    print(f"\n{metric}:")
    for s in ["all", "q1_low_llm", "q4_high_llm"]:
        v24 = headline_df[(headline_df["subset"] == s) & (headline_df["year"] == "2024")][metric].iloc[0]
        v26 = headline_df[(headline_df["subset"] == s) & (headline_df["year"] == "2026")][metric].iloc[0]
        delta = v26 - v24
        pct = (v26 - v24) / v24 * 100 if v24 else float("nan")
        print(f"  {s}: 2024={v24:.3f}  2026={v26:.3f}  Δ={delta:+.3f}  ({pct:+.1f}%)")

# ---------------------------------------------------------------------------
# Figure: score distribution by period
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (label, d) in zip(axes, [("raw (mixed text-source)", raw),
                                    ("LLM-cleaned subset (text-source-controlled)", ctrl)]):
    for period, color in [("2024", "#1f77b4"), ("2026", "#ff7f0e")]:
        sub = d[d["year"] == period]["llm_score_robust"]
        ax.hist(sub, bins=50, alpha=0.5, label=period, color=color, density=True)
    ax.set_title(label)
    ax.set_xlabel("LLM-authorship composite score (z-sum)")
    ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_F}/fig1_score_distribution.png")
plt.close()

print("\nT29 step 3 complete.")
