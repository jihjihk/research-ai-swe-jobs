#!/usr/bin/env python3
"""
T09 step 4: Method comparison between BERTopic and NMF; characterization of
BERTopic clusters (primary method); NMI vs seniority / period / tech; save
cluster-label artifact for downstream tasks.
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import duckdb
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

OUT_TABLES = "exploration/tables/T09"
OUT_ARTIFACT = "exploration/artifacts/shared"
os.makedirs(OUT_TABLES, exist_ok=True)
os.makedirs(OUT_ARTIFACT, exist_ok=True)

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
sample = pd.read_parquet(f"{OUT_TABLES}/sample.parquet")
bt = pd.read_parquet(f"{OUT_TABLES}/bertopic_topics.parquet")
nmf = pd.read_parquet(f"{OUT_TABLES}/nmf_assignments_wide.parquet")
terms_bt = pd.read_csv(f"{OUT_TABLES}/bertopic_topic_terms.csv")
terms_nmf = pd.read_csv(f"{OUT_TABLES}/nmf_topic_terms.csv")

df = sample.merge(bt, on="uid").merge(nmf, on="uid")
print(f"Merged sample: {len(df):,}")

# Tech matrix
tech = pd.read_parquet("exploration/artifacts/shared/swe_tech_matrix.parquet")
df = df.merge(tech, on="uid", how="left")
tech_cols = [c for c in tech.columns if c != "uid"]

# Add description length
df["desc_len"] = df["description_cleaned"].fillna("").str.len()
df["tech_count"] = df[tech_cols].sum(axis=1)
df["yoe"] = pd.to_numeric(df["yoe_extracted"], errors="coerce")

# ---------------------------------------------------------------------------
# 1. Topic alignment — BERTopic vs NMF via top-term Jaccard
# ---------------------------------------------------------------------------
def top_terms(rows_df, topic_col, topic_id, top_n=20):
    sub = rows_df[rows_df[topic_col] == topic_id].sort_values("rank").head(top_n)
    return set(sub["term"].tolist())


bt_topic_ids = sorted([t for t in df["bertopic_topic_mts30"].unique() if t >= 0])
nmf_k = 12  # primary NMF k for comparison (reasonable interpretability)
nmf_component_ids = sorted(df["nmf_k12"].unique())

bt_sets = {t: top_terms(terms_bt, "topic", t, top_n=20) for t in bt_topic_ids}
nmf_sets = {
    c: top_terms(terms_nmf[terms_nmf.k == nmf_k], "component", c, top_n=20)
    for c in nmf_component_ids
}

align_rows = []
for bt_id, bt_set in bt_sets.items():
    for nmf_id, nmf_set in nmf_sets.items():
        inter = len(bt_set & nmf_set)
        union = len(bt_set | nmf_set)
        jac = inter / union if union else 0.0
        align_rows.append(
            {
                "bertopic_topic": bt_id,
                "nmf_component": nmf_id,
                "top_term_jaccard": round(jac, 4),
                "intersection_size": int(inter),
            }
        )
align_df = pd.DataFrame(align_rows)
align_df.to_csv(f"{OUT_TABLES}/method_alignment_jaccard.csv", index=False)

# Best NMF match per BERTopic topic and vice versa
best_per_bt = align_df.loc[align_df.groupby("bertopic_topic")["top_term_jaccard"].idxmax()]
best_per_bt.to_csv(f"{OUT_TABLES}/method_best_nmf_per_bertopic.csv", index=False)
print("\nBERTopic -> best NMF (k=12) match:")
print(best_per_bt.to_string(index=False))

best_per_nmf = align_df.loc[align_df.groupby("nmf_component")["top_term_jaccard"].idxmax()]
best_per_nmf.to_csv(f"{OUT_TABLES}/method_best_bertopic_per_nmf.csv", index=False)
print("\nNMF -> best BERTopic match:")
print(best_per_nmf.to_string(index=False))

method_robust_threshold = 0.15
robust_topics = (best_per_bt[best_per_bt["top_term_jaccard"] >= method_robust_threshold]).copy()
method_specific = (best_per_bt[best_per_bt["top_term_jaccard"] < method_robust_threshold]).copy()
print(
    f"\nMethod-robust BERTopic topics (Jaccard>={method_robust_threshold}): "
    f"{len(robust_topics)}/{len(best_per_bt)}"
)

# ---------------------------------------------------------------------------
# 2. Methods comparison table
# ---------------------------------------------------------------------------
bertopic_sweep = pd.read_csv(f"{OUT_TABLES}/bertopic_min_topic_size_sweep.csv")
coherence = pd.read_csv(f"{OUT_TABLES}/bertopic_topic_coherence.csv")
ari = pd.read_csv(f"{OUT_TABLES}/bertopic_stability_ari.csv")

comparison_rows = [
    {
        "method": "BERTopic (mts=20)",
        "n_topics": int(bertopic_sweep.loc[bertopic_sweep.min_topic_size == 20, "n_topics"].iloc[0]),
        "pct_noise": float(bertopic_sweep.loc[bertopic_sweep.min_topic_size == 20, "pct_noise"].iloc[0]),
        "stability_ari_core": None,
        "mean_coherence": None,
        "interpretability": "Too fragmented: many duplicate industry-niche topics.",
    },
    {
        "method": "BERTopic (mts=30, primary)",
        "n_topics": int(bertopic_sweep.loc[bertopic_sweep.min_topic_size == 30, "n_topics"].iloc[0]),
        "pct_noise": float(bertopic_sweep.loc[bertopic_sweep.min_topic_size == 30, "pct_noise"].iloc[0]),
        "stability_ari_core": round(float(ari["ari_core_only"].mean()), 3),
        "mean_coherence": round(float(coherence["umass_coherence"].mean()), 3),
        "interpretability": "Clean domain clusters (AI/ML, cloud/devops, web, data eng, mobile, embedded, QA).",
    },
    {
        "method": "BERTopic (mts=50)",
        "n_topics": int(bertopic_sweep.loc[bertopic_sweep.min_topic_size == 50, "n_topics"].iloc[0]),
        "pct_noise": float(bertopic_sweep.loc[bertopic_sweep.min_topic_size == 50, "pct_noise"].iloc[0]),
        "stability_ari_core": None,
        "mean_coherence": None,
        "interpretability": "Over-merged: only 5 coarse clusters, loses domain structure.",
    },
    {
        "method": "NMF k=5",
        "n_topics": 5,
        "pct_noise": 0.0,
        "stability_ari_core": None,
        "mean_coherence": None,
        "interpretability": "Dominated by company-boilerplate artifacts (chatbots, enthusiastic).",
    },
    {
        "method": "NMF k=8",
        "n_topics": 8,
        "pct_noise": 0.0,
        "stability_ari_core": None,
        "mean_coherence": None,
        "interpretability": "Mixes real clusters (AI/ML, cloud) with boilerplate artifacts.",
    },
    {
        "method": "NMF k=12",
        "n_topics": 12,
        "pct_noise": 0.0,
        "stability_ari_core": None,
        "mean_coherence": None,
        "interpretability": "Best NMF config; surfaces AI, data eng, cloud, web, embedded, but 4/12 components are boilerplate signatures.",
    },
    {
        "method": "NMF k=15",
        "n_topics": 15,
        "pct_noise": 0.0,
        "stability_ari_core": None,
        "mean_coherence": None,
        "interpretability": "Adds clearance/embedded and agentic/govcloud micro-topics; boilerplate artifacts persist.",
    },
]
pd.DataFrame(comparison_rows).to_csv(f"{OUT_TABLES}/methods_comparison.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Characterization of BERTopic primary clusters
# ---------------------------------------------------------------------------
# Attach a human-readable name based on top terms
def name_from_top_terms(topic_id: int) -> str:
    sub = terms_bt[terms_bt.topic == topic_id].sort_values("rank").head(5)
    return " / ".join(sub["term"].tolist()[:3])


df["archetype"] = df["bertopic_topic_mts30"]
df["archetype_name"] = df["archetype"].apply(
    lambda t: "noise_outliers" if t == -1 else f"T{t:02d}_" + name_from_top_terms(int(t))
)

# Use combined best-available seniority (already in sample), plus YOE proxy
df["seniority_best_available"] = df["seniority_best_available"].fillna("unknown")
df["yoe_junior"] = (df["yoe"].fillna(999) <= 2).astype(int)
df["yoe_known"] = df["yoe"].notna().astype(int)

# Per cluster summary table
rows = []
for tid in sorted(df["archetype"].unique()):
    sub = df[df["archetype"] == tid]
    sen_counts = sub["seniority_best_available"].value_counts(normalize=False)
    sen_share = sub["seniority_best_available"].value_counts(normalize=True)
    known_mask = sub["seniority_best_available"].isin(["entry", "associate", "mid-senior", "director"])
    entry_known = (sub["seniority_best_available"] == "entry").sum() / max(known_mask.sum(), 1)
    # YOE-based junior share among rows with YOE known
    yoe_known_sub = sub[sub["yoe"].notna()]
    yoe_junior_share = (
        (yoe_known_sub["yoe"] <= 2).mean() if len(yoe_known_sub) else float("nan")
    )
    period_share = sub["period_bucket"].value_counts(normalize=True).to_dict()
    # Top 5 terms for archetype
    if tid == -1:
        top_terms_list = []
    else:
        top_terms_list = (
            terms_bt[terms_bt.topic == tid].sort_values("rank").head(8)["term"].tolist()
        )
    # Top 5 tech
    tech_sums = sub[tech_cols].sum(axis=0).sort_values(ascending=False)
    top_tech = [(c, int(v)) for c, v in tech_sums.head(5).items()]
    rows.append(
        {
            "archetype": int(tid),
            "archetype_name": sub["archetype_name"].iloc[0] if len(sub) else "",
            "n": int(len(sub)),
            "top_terms": ", ".join(top_terms_list[:8]),
            "top_tech": ", ".join(f"{c}({v})" for c, v in top_tech),
            "pct_2024": round(period_share.get("2024", 0) * 100, 2),
            "pct_2026_03": round(period_share.get("2026-03", 0) * 100, 2),
            "pct_2026_04": round(period_share.get("2026-04", 0) * 100, 2),
            "entry_share_combined": round(
                float((sub["seniority_best_available"] == "entry").sum()) / max(known_mask.sum(), 1),
                3,
            ),
            "entry_share_combined_n_known": int(known_mask.sum()),
            "yoe_junior_share_proxy": round(float(yoe_junior_share), 3)
            if not np.isnan(yoe_junior_share)
            else None,
            "yoe_n_known": int(len(yoe_known_sub)),
            "mean_desc_len": round(float(sub["desc_len"].mean()), 1),
            "median_yoe": float(sub["yoe"].median()) if len(yoe_known_sub) else None,
            "mean_tech_count": round(float(sub["tech_count"].mean()), 2),
        }
    )
arch_df = pd.DataFrame(rows).sort_values("n", ascending=False)
arch_df.to_csv(f"{OUT_TABLES}/archetype_characterization.csv", index=False)
print("\nArchetype characterization (head):")
print(arch_df.head(10)[["archetype", "archetype_name", "n", "pct_2024", "pct_2026_03", "pct_2026_04", "entry_share_combined"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Entry-level share by archetype x period (combined vs YOE proxy)
# ---------------------------------------------------------------------------
rows = []
for tid in sorted([t for t in df["archetype"].unique() if t != -1]):
    for period in ["2024", "2026-03", "2026-04"]:
        sub = df[(df.archetype == tid) & (df.period_bucket == period)]
        known = sub[sub["seniority_best_available"].isin(
            ["entry", "associate", "mid-senior", "director"]
        )]
        yoe_sub = sub[sub["yoe"].notna()]
        rows.append(
            {
                "archetype": int(tid),
                "period": period,
                "n": int(len(sub)),
                "n_known_combined": int(len(known)),
                "entry_share_combined": round(
                    float((known["seniority_best_available"] == "entry").mean()), 3
                )
                if len(known)
                else None,
                "n_known_yoe": int(len(yoe_sub)),
                "yoe_junior_share": round(float((yoe_sub["yoe"] <= 2).mean()), 3)
                if len(yoe_sub)
                else None,
            }
        )
pd.DataFrame(rows).to_csv(f"{OUT_TABLES}/entry_share_by_archetype_period.csv", index=False)

# ---------------------------------------------------------------------------
# 5. Temporal dynamics — archetype share by period
# ---------------------------------------------------------------------------
period_counts = df.groupby(["period_bucket", "archetype"]).size().rename("n").reset_index()
period_totals = df.groupby("period_bucket").size().rename("total").reset_index()
temporal = period_counts.merge(period_totals, on="period_bucket")
temporal["share"] = temporal["n"] / temporal["total"]
temporal_wide = temporal.pivot_table(
    index="archetype", columns="period_bucket", values="share", fill_value=0.0
).reset_index()
# Attach names
name_map = dict(zip(df["archetype"], df["archetype_name"]))
temporal_wide["archetype_name"] = temporal_wide["archetype"].map(name_map)
temporal_wide["delta_2024_to_2026"] = (
    temporal_wide.get("2026-03", 0) + temporal_wide.get("2026-04", 0)
) / 2 - temporal_wide.get("2024", 0)
temporal_wide = temporal_wide.sort_values("delta_2024_to_2026", ascending=False)
temporal_wide.to_csv(f"{OUT_TABLES}/archetype_temporal_dynamics.csv", index=False)
print("\nTemporal dynamics (top growers/shrinkers):")
print(temporal_wide.head(10)[["archetype", "archetype_name", "2024", "2026-03", "2026-04", "delta_2024_to_2026"]].to_string(index=False))
print("...")
print(temporal_wide.tail(5)[["archetype", "archetype_name", "2024", "2026-03", "2026-04", "delta_2024_to_2026"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 6. NMI — does cluster structure align with known labels?
# ---------------------------------------------------------------------------
labels_cluster = df["archetype"].values
nmi_rows = []

# seniority combined (drop unknown/NaN)
m = df["seniority_best_available"] != "unknown"
nmi_rows.append(
    {
        "target": "seniority_combined (known only)",
        "nmi": round(
            float(normalized_mutual_info_score(df.loc[m, "seniority_best_available"], labels_cluster[m])),
            4,
        ),
        "n": int(m.sum()),
    }
)

# yoe junior vs not (yoe known)
m = df["yoe"].notna()
nmi_rows.append(
    {
        "target": "yoe_junior (<=2 vs >2, yoe known)",
        "nmi": round(
            float(normalized_mutual_info_score(df.loc[m, "yoe_junior"], labels_cluster[m])),
            4,
        ),
        "n": int(m.sum()),
    }
)

nmi_rows.append(
    {
        "target": "period_bucket",
        "nmi": round(
            float(normalized_mutual_info_score(df["period_bucket"], labels_cluster)), 4
        ),
        "n": int(len(df)),
    }
)

# Tech domain proxy: which single dominant tech the posting mentions most
# (primary language: python/java/javascript/typescript/go/rust/csharp)
priority_langs = ["python", "java", "javascript", "typescript", "golang", "rust", "csharp"]
present = [c for c in priority_langs if c in tech_cols]
if present:
    def primary_lang(row):
        for c in present:
            if row[c]:
                return c
        return "none"

    df["primary_lang"] = df[present].apply(primary_lang, axis=1)
    nmi_rows.append(
        {
            "target": f"primary_language ({'/'.join(present)})",
            "nmi": round(
                float(normalized_mutual_info_score(df["primary_lang"], labels_cluster)),
                4,
            ),
            "n": int(len(df)),
        }
    )

# Text source (detect LLM text preference bias)
nmi_rows.append(
    {
        "target": "text_source (llm vs rule)",
        "nmi": round(
            float(normalized_mutual_info_score(df["text_source"], labels_cluster)), 4
        ),
        "n": int(len(df)),
    }
)

# Source (arshkon vs asaniczka vs scraped)
nmi_rows.append(
    {
        "target": "data source",
        "nmi": round(
            float(normalized_mutual_info_score(df["source"], labels_cluster)), 4
        ),
        "n": int(len(df)),
    }
)

# Aggregator flag
nmi_rows.append(
    {
        "target": "is_aggregator",
        "nmi": round(
            float(normalized_mutual_info_score(df["is_aggregator"].astype(str), labels_cluster)), 4
        ),
        "n": int(len(df)),
    }
)

nmi_df = pd.DataFrame(nmi_rows)
nmi_df.to_csv(f"{OUT_TABLES}/nmi_scores.csv", index=False)
print("\nNMI vs cluster assignment:")
print(nmi_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 7. BERTopic <-> NMF cross tab NMI (another method comparison metric)
# ---------------------------------------------------------------------------
cross_nmi = {
    "nmi_bertopic_vs_nmf_k5": round(
        float(normalized_mutual_info_score(df["bertopic_topic_mts30"], df["nmf_k5"])), 4
    ),
    "nmi_bertopic_vs_nmf_k8": round(
        float(normalized_mutual_info_score(df["bertopic_topic_mts30"], df["nmf_k8"])), 4
    ),
    "nmi_bertopic_vs_nmf_k12": round(
        float(normalized_mutual_info_score(df["bertopic_topic_mts30"], df["nmf_k12"])), 4
    ),
    "nmi_bertopic_vs_nmf_k15": round(
        float(normalized_mutual_info_score(df["bertopic_topic_mts30"], df["nmf_k15"])), 4
    ),
    "ari_bertopic_vs_nmf_k12": round(
        float(adjusted_rand_score(df["bertopic_topic_mts30"], df["nmf_k12"])), 4
    ),
}
with open(f"{OUT_TABLES}/method_cross_nmi.json", "w") as f:
    json.dump(cross_nmi, f, indent=2)
print("\nCross-method NMI:")
print(json.dumps(cross_nmi, indent=2))

# ---------------------------------------------------------------------------
# 8. Save cluster labels artifact for downstream tasks
# ---------------------------------------------------------------------------
out = df[["uid", "archetype", "archetype_name"]].copy()
out.to_parquet(f"{OUT_ARTIFACT}/swe_archetype_labels.parquet", index=False)
print(f"\nSaved {OUT_ARTIFACT}/swe_archetype_labels.parquet — {len(out)} rows")

# ---------------------------------------------------------------------------
# 9. Sensitivities: aggregator exclusion and text source ablation
# ---------------------------------------------------------------------------
sens_rows = []
# Baseline period share
base = (
    df.groupby(["archetype", "period_bucket"]).size().rename("n").reset_index()
)
base_totals = df.groupby("period_bucket").size().rename("total").reset_index()
base = base.merge(base_totals, on="period_bucket")
base["share"] = base["n"] / base["total"]
base["variant"] = "baseline"

# aggregator excluded
da = df[df["is_aggregator"] == False]
ag = da.groupby(["archetype", "period_bucket"]).size().rename("n").reset_index()
ag_t = da.groupby("period_bucket").size().rename("total").reset_index()
ag = ag.merge(ag_t, on="period_bucket")
ag["share"] = ag["n"] / ag["total"]
ag["variant"] = "no_aggregator"

# llm text only
dl = df[df["text_source"] == "llm"]
ll = dl.groupby(["archetype", "period_bucket"]).size().rename("n").reset_index()
ll_t = dl.groupby("period_bucket").size().rename("total").reset_index()
ll = ll.merge(ll_t, on="period_bucket")
ll["share"] = ll["n"] / ll["total"]
ll["variant"] = "llm_text_only"

sens = pd.concat([base, ag, ll], ignore_index=True)
sens.to_csv(f"{OUT_TABLES}/archetype_sensitivities.csv", index=False)

# Summary: max absolute delta per archetype from baseline across variants
piv = sens.pivot_table(
    index=["archetype", "period_bucket"], columns="variant", values="share", fill_value=0.0
).reset_index()
piv["abs_delta_no_agg"] = (piv["no_aggregator"] - piv["baseline"]).abs()
piv["abs_delta_llm"] = (piv["llm_text_only"] - piv["baseline"]).abs()
sens_summary = piv.groupby("archetype")[["abs_delta_no_agg", "abs_delta_llm"]].max().reset_index()
sens_summary.to_csv(f"{OUT_TABLES}/archetype_sensitivity_summary.csv", index=False)
print("\nMax absolute share-shift per archetype under sensitivities:")
print(sens_summary.sort_values("abs_delta_llm", ascending=False).head(10).to_string(index=False))

print("\nDONE characterization stage")
