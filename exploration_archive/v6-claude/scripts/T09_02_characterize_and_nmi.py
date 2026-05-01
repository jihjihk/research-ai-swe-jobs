"""T09 Step 6-9: Characterize archetypes, run NMI, produce visualizations.

Outputs:
  - exploration/tables/T09/archetype_characterization.csv
  - exploration/tables/T09/entry_share_by_period.csv
  - exploration/tables/T09/archetype_names.csv
  - exploration/tables/T09/nmi_verdict.csv
  - exploration/tables/T09/representative_postings.csv
  - exploration/figures/T09/*.png
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

ROOT = Path("/home/jihgaboot/gabor/job-research")
ART = ROOT / "exploration/artifacts"
SHARED = ART / "shared"
TABLES = ROOT / "exploration/tables/T09"
FIGS = ROOT / "exploration/figures/T09"
FIGS.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Load sample
log("Loading sample")
sample = pd.read_parquet(ART / "T09_sample.parquet")
X_2d = np.load(ART / "T09_umap2d.npy")
X_pca = np.load(ART / "T09_pca2d.npy")
X = np.load(ART / "T09_sample_embeddings.npy")
log(f"Sample rows: {len(sample):,}")

# Load tech matrix
log("Loading tech matrix")
con = duckdb.connect()
tech = con.execute(
    f"SELECT * FROM 'exploration/artifacts/shared/swe_tech_matrix.parquet' WHERE uid IN (SELECT uid FROM sample)",
    {"sample": sample[["uid"]]},
).df() if False else con.execute(
    "SELECT * FROM 'exploration/artifacts/shared/swe_tech_matrix.parquet'"
).df()
tech_in_sample = tech[tech["uid"].isin(set(sample["uid"]))].copy()
log(f"Tech rows matched: {len(tech_in_sample):,}")
sample_tech = sample.merge(tech_in_sample, on="uid", how="left")

# Tech-domain proxy: pick the strongest tech bucket per posting
TECH_DOMAINS = {
    "frontend": ["react", "angular", "vue", "nextjs", "svelte", "jquery", "html_css", "tailwind", "webpack", "typescript", "javascript"],
    "backend_java": ["java", "spring", "kotlin"],
    "backend_python": ["python", "django", "flask", "fastapi"],
    "backend_node": ["nodejs", "express"],
    "dotnet": ["dotnet", "c_sharp"],
    "mobile": ["swift", "objective_c", "kotlin", "dart"],  # kotlin appears twice but harmless
    "data_eng": ["spark", "hadoop", "airflow", "dbt", "snowflake", "databricks", "bigquery", "redshift", "kafka"],
    "ml_ai": ["tensorflow", "pytorch", "scikit_learn", "machine_learning", "deep_learning", "nlp", "computer_vision", "llm", "langchain", "langgraph", "rag", "vector_db", "huggingface", "openai_api", "claude_api", "prompt_engineering", "fine_tuning", "mcp", "agents_framework", "gpt", "transformer_arch", "embedding", "pinecone", "chromadb"],
    "devops_cloud": ["aws", "azure", "gcp", "kubernetes", "docker", "terraform", "ansible", "cicd", "jenkins", "github_actions", "argocd", "helm", "serverless", "lambda"],
    "embedded": ["c_lang", "c_plus_plus", "rust"],  # approximate
    "qa": ["selenium", "cypress", "playwright", "jest", "pytest", "junit"],
}


def pick_domain(row: pd.Series) -> str:
    scores: dict[str, int] = {}
    for dom, cols in TECH_DOMAINS.items():
        scores[dom] = sum(int(bool(row.get(c, False))) for c in cols if c in row)
    # Tiebreak: ml_ai wins over devops_cloud (devops_cloud catches everything)
    priority = ["ml_ai", "data_eng", "embedded", "mobile", "frontend", "backend_python", "backend_java", "backend_node", "dotnet", "qa", "devops_cloud"]
    best = max(scores.values())
    if best == 0:
        return "other"
    # Among tied winners, take first in priority
    for p in priority:
        if scores.get(p, 0) == best:
            return p
    return max(scores, key=scores.get)


log("Computing tech-domain proxy")
tech_cols_used = set()
for cols in TECH_DOMAINS.values():
    tech_cols_used.update(cols)
# Ensure all are present; skip any missing
tech_cols_present = [c for c in tech_cols_used if c in sample_tech.columns]
sample_tech["tech_domain"] = sample_tech.apply(pick_domain, axis=1)
log(f"Tech domain distribution: {sample_tech['tech_domain'].value_counts().to_dict()}")

# ---------------------------------------------------------------------------
# NMI verdict (Step 9)
# ---------------------------------------------------------------------------
log("Computing NMI verdict")
cluster = sample_tech["bertopic_primary"].values
# Restrict NMI to non-noise for a fair comparison (note also full)
nonnoise = cluster != -1

senio = sample_tech["seniority_3level"].fillna("unknown").values
period = sample_tech["period_group"].values
domain = sample_tech["tech_domain"].values

# Industry: arshkon + scraped only (asaniczka has no industry) — pull from unified
log("Loading industry from unified")
uids_needed = tuple(sample_tech["uid"].tolist())
ind_df = con.execute(
    f"""
    SELECT uid, company_industry
    FROM 'data/unified.parquet'
    WHERE uid IN {uids_needed}
    """
).df()
sample_tech = sample_tech.merge(ind_df, on="uid", how="left")
industry = sample_tech["company_industry"].fillna("_missing").values

nmi_rows = []
for label, vec in [
    ("seniority_3level", senio),
    ("period_group", period),
    ("tech_domain_proxy", domain),
    ("company_industry", industry),
]:
    full = normalized_mutual_info_score(cluster, vec)
    nn = normalized_mutual_info_score(cluster[nonnoise], vec[nonnoise])
    # Industry: restrict to rows with non-missing industry
    if label == "company_industry":
        ind_mask = industry != "_missing"
        nn_ind = normalized_mutual_info_score(cluster[nonnoise & ind_mask], vec[nonnoise & ind_mask])
        nmi_rows.append({"target": label, "nmi_full": full, "nmi_nonnoise": nn, "nmi_nonnoise_ind_only": nn_ind, "n_nonnoise": int(nonnoise.sum()), "n_nonnoise_labeled": int((nonnoise & ind_mask).sum())})
    else:
        nmi_rows.append({"target": label, "nmi_full": full, "nmi_nonnoise": nn, "nmi_nonnoise_ind_only": None, "n_nonnoise": int(nonnoise.sum()), "n_nonnoise_labeled": int(nonnoise.sum())})

nmi_df = pd.DataFrame(nmi_rows)
nmi_df.to_csv(TABLES / "nmi_verdict.csv", index=False)
print(nmi_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Archetype naming (manual based on top terms)
# ---------------------------------------------------------------------------
ARCHETYPE_NAMES = {
    -1: "Noise / unassigned",
    0: "LLM / GenAI / ML engineering",
    1: "Defense & cleared regulated work",
    2: "DevOps / SRE / platform",
    3: "JS frontend (React/TypeScript)",
    4: "Java enterprise (Spring/microservices)",
    5: "Data engineering (Spark/ETL)",
    6: "Embedded / firmware",
    7: ".NET / ASP.NET full-stack",
    8: "Agile/Scrum generalist",
    9: "iOS / Swift mobile",
    10: "AWS cloud SDE (Amazon template)",
    11: "QA / test automation",
    12: "Azure data platform",
    13: "Amazon program boilerplate",
    14: "US gov SCI-cleared SWE",
    15: "GPU / CUDA / systems perf",
    16: "Python web backend (Django/Flask)",
    17: "AWS cloud generic template",
    18: "Android / Kotlin mobile",
    19: "Generic JS/Python boilerplate",
}

name_rows = [{"archetype": k, "archetype_name": v} for k, v in ARCHETYPE_NAMES.items()]
pd.DataFrame(name_rows).to_csv(TABLES / "archetype_names.csv", index=False)

sample_tech["archetype_name"] = sample_tech["bertopic_primary"].map(ARCHETYPE_NAMES)

# ---------------------------------------------------------------------------
# Characterization (Step 6)
# ---------------------------------------------------------------------------
log("Characterization")
bt_topics = pd.read_csv(TABLES / "bertopic_topics.csv")
bt_top_terms = {int(r["topic_id"]): r["top_terms"] for _, r in bt_topics.iterrows()}

char_rows = []
for topic_id in sorted(sample_tech["bertopic_primary"].unique()):
    if topic_id == -1:
        continue
    grp = sample_tech[sample_tech["bertopic_primary"] == topic_id]
    n = len(grp)
    # Seniority share
    sen_dist = grp["seniority_3level"].fillna("unknown").value_counts(normalize=True).to_dict()
    # Period share
    per_dist = grp["period_group"].value_counts(normalize=True).to_dict()
    per_count = grp["period_group"].value_counts().to_dict()
    # Mean metrics
    desc_lens = grp["description_cleaned"].str.len()
    yoe = grp["yoe_extracted"]
    tech_count = grp[[c for c in tech_cols_present if c in grp.columns]].sum(axis=1)
    # Top 5 tech mentions
    tech_prev = grp[[c for c in tech_cols_present if c in grp.columns]].mean().sort_values(ascending=False).head(5)
    char_rows.append({
        "archetype": topic_id,
        "archetype_name": ARCHETYPE_NAMES[topic_id],
        "n": n,
        "top_terms": bt_top_terms.get(topic_id, ""),
        "share_junior": sen_dist.get("junior", 0.0),
        "share_mid": sen_dist.get("mid", 0.0),
        "share_senior": sen_dist.get("senior", 0.0),
        "share_unknown_sen": sen_dist.get("unknown", 0.0),
        "share_2024": per_dist.get("2024", 0.0),
        "share_2026_03": per_dist.get("2026-03", 0.0),
        "share_2026_04": per_dist.get("2026-04", 0.0),
        "n_2024": per_count.get("2024", 0),
        "n_2026_03": per_count.get("2026-03", 0),
        "n_2026_04": per_count.get("2026-04", 0),
        "desc_len_mean": desc_lens.mean(),
        "yoe_mean": yoe.mean(),
        "tech_count_mean": tech_count.mean(),
        "top5_tech": " | ".join(tech_prev.index.tolist()),
    })

char_df = pd.DataFrame(char_rows)
char_df.to_csv(TABLES / "archetype_characterization.csv", index=False)

# ---------------------------------------------------------------------------
# Entry share by archetype by period (Alternative B test)
# ---------------------------------------------------------------------------
log("Entry share by archetype x period")
# Use seniority_3level = 'junior' as entry, 'senior' as senior. Compute entry share
# of KNOWN seniority (junior + mid + senior).
# Also compute YOE <= 2 share as label-independent validator.
entry_rows = []
for topic_id in sorted(sample_tech["bertopic_primary"].unique()):
    if topic_id == -1:
        continue
    grp = sample_tech[sample_tech["bertopic_primary"] == topic_id]
    for pg, pgrp in grp.groupby("period_group"):
        known = pgrp[pgrp["seniority_3level"].isin(["junior", "mid", "senior"])]
        n_known = len(known)
        n_junior = (known["seniority_3level"] == "junior").sum()
        junior_share = n_junior / n_known if n_known > 0 else np.nan
        yoe_nn = pgrp[pgrp["yoe_extracted"].notna()]
        n_yoe = len(yoe_nn)
        yoe_le2 = (yoe_nn["yoe_extracted"] <= 2).sum()
        yoe_le2_share = yoe_le2 / n_yoe if n_yoe > 0 else np.nan
        entry_rows.append({
            "archetype": topic_id,
            "archetype_name": ARCHETYPE_NAMES[topic_id],
            "period_group": pg,
            "n_total": len(pgrp),
            "n_known_seniority": n_known,
            "junior_share_of_known": junior_share,
            "n_junior": n_junior,
            "n_yoe_known": n_yoe,
            "yoe_le2_share": yoe_le2_share,
        })

entry_df = pd.DataFrame(entry_rows)
entry_df.to_csv(TABLES / "entry_share_by_period.csv", index=False)

# ---------------------------------------------------------------------------
# Archetype proportion change (Step 7)
# ---------------------------------------------------------------------------
log("Archetype proportion by period")
prop_rows = []
per_totals = sample_tech.groupby("period_group").size().to_dict()
for topic_id in sorted(sample_tech["bertopic_primary"].unique()):
    grp = sample_tech[sample_tech["bertopic_primary"] == topic_id]
    row = {"archetype": topic_id, "archetype_name": ARCHETYPE_NAMES[topic_id]}
    for pg in ["2024", "2026-03", "2026-04"]:
        sub = grp[grp["period_group"] == pg]
        row[f"share_{pg}"] = len(sub) / per_totals[pg] if per_totals.get(pg, 0) > 0 else 0
        row[f"n_{pg}"] = len(sub)
    row["delta_2024_to_2026avg"] = (row["share_2026-03"] + row["share_2026-04"]) / 2 - row["share_2024"]
    prop_rows.append(row)
prop_df = pd.DataFrame(prop_rows).sort_values("delta_2024_to_2026avg", ascending=False)
prop_df.to_csv(TABLES / "archetype_proportion_by_period.csv", index=False)
print(prop_df[["archetype", "archetype_name", "share_2024", "share_2026-03", "share_2026-04", "delta_2024_to_2026avg"]].to_string(index=False))

# ---------------------------------------------------------------------------
# Representative postings
# ---------------------------------------------------------------------------
log("Representative postings")
# Pull first 200 chars + title from unified
title_df = con.execute(
    f"SELECT uid, title FROM 'data/unified.parquet' WHERE uid IN {uids_needed}"
).df()
sample_tech = sample_tech.merge(title_df, on="uid", how="left")

rep_rows = []
for topic_id in sorted(sample_tech["bertopic_primary"].unique()):
    if topic_id == -1:
        continue
    grp = sample_tech[sample_tech["bertopic_primary"] == topic_id]
    # Pick ~3-5 representative by distance to topic centroid in embedding space
    idx = grp.index.tolist()
    emb_rows = X[idx]
    centroid = emb_rows.mean(axis=0)
    dists = np.linalg.norm(emb_rows - centroid, axis=1)
    order = np.argsort(dists)[:5]
    for o in order:
        r = grp.iloc[o]
        rep_rows.append({
            "archetype": topic_id,
            "archetype_name": ARCHETYPE_NAMES[topic_id],
            "uid": r["uid"],
            "title": r.get("title"),
            "desc_first_200": (r.get("description_cleaned") or "")[:200],
        })
pd.DataFrame(rep_rows).to_csv(TABLES / "representative_postings.csv", index=False)

# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
log("Figures")
plt.rcParams["figure.dpi"] = 150

# Fig 1: UMAP colored by cluster
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
ax = axes[0]
# cluster colored
for cid in sorted(sample_tech["bertopic_primary"].unique()):
    m = sample_tech["bertopic_primary"] == cid
    if cid == -1:
        ax.scatter(X_2d[m, 0], X_2d[m, 1], s=3, c="lightgrey", alpha=0.3, label=None)
    else:
        ax.scatter(X_2d[m, 0], X_2d[m, 1], s=4, alpha=0.6, label=f"T{cid}")
ax.set_title(f"UMAP 2D, BERTopic cluster (n_topics={len([t for t in set(sample_tech['bertopic_primary'].unique()) if t != -1])})")
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")

# Period colored
ax = axes[1]
colors = {"2024": "tab:blue", "2026-03": "tab:orange", "2026-04": "tab:red"}
for pg, c in colors.items():
    m = sample_tech["period_group"] == pg
    ax.scatter(X_2d[m, 0], X_2d[m, 1], s=4, c=c, alpha=0.5, label=pg)
ax.set_title("UMAP 2D colored by period")
ax.legend()
ax.set_xlabel("UMAP-1")

# Seniority colored
ax = axes[2]
s_colors = {"junior": "tab:green", "mid": "tab:olive", "senior": "tab:purple", "unknown": "lightgrey"}
for sn, c in s_colors.items():
    m = sample_tech["seniority_3level"].fillna("unknown") == sn
    ax.scatter(X_2d[m, 0], X_2d[m, 1], s=4, c=c, alpha=0.5, label=sn)
ax.set_title("UMAP 2D colored by seniority_3level")
ax.legend()
ax.set_xlabel("UMAP-1")
plt.tight_layout()
plt.savefig(FIGS / "umap_colored.png", dpi=150, bbox_inches="tight")
plt.close()

# Fig 2: PCA counterpart (single pane)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
ax = axes[0]
for cid in sorted(sample_tech["bertopic_primary"].unique()):
    m = sample_tech["bertopic_primary"] == cid
    if cid == -1:
        ax.scatter(X_pca[m, 0], X_pca[m, 1], s=3, c="lightgrey", alpha=0.3)
    else:
        ax.scatter(X_pca[m, 0], X_pca[m, 1], s=4, alpha=0.6)
ax.set_title("PCA 2D, cluster (visual story — compare to UMAP)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

ax = axes[1]
for pg, c in colors.items():
    m = sample_tech["period_group"] == pg
    ax.scatter(X_pca[m, 0], X_pca[m, 1], s=4, c=c, alpha=0.5, label=pg)
ax.set_title("PCA 2D, period")
ax.legend()
ax.set_xlabel("PC1")

ax = axes[2]
for sn, c in s_colors.items():
    m = sample_tech["seniority_3level"].fillna("unknown") == sn
    ax.scatter(X_pca[m, 0], X_pca[m, 1], s=4, c=c, alpha=0.5, label=sn)
ax.set_title("PCA 2D, seniority")
ax.legend()
ax.set_xlabel("PC1")
plt.tight_layout()
plt.savefig(FIGS / "pca_colored.png", dpi=150, bbox_inches="tight")
plt.close()

# Fig 3: Archetype proportion stacked bar by period
log("Proportion stacked bar")
piv = prop_df.set_index("archetype_name")[["share_2024", "share_2026-03", "share_2026-04"]]
piv = piv.loc[piv.index != "Noise / unassigned"]
piv = piv.sort_values("share_2026-04", ascending=False)
fig, ax = plt.subplots(figsize=(11, 7))
# Plot as horizontal grouped bars
y = np.arange(len(piv))
h = 0.25
ax.barh(y - h, piv["share_2024"], height=h, label="2024")
ax.barh(y, piv["share_2026-03"], height=h, label="2026-03")
ax.barh(y + h, piv["share_2026-04"], height=h, label="2026-04")
ax.set_yticks(y)
ax.set_yticklabels(piv.index, fontsize=9)
ax.set_xlabel("Share of sample (within period)")
ax.set_title("Archetype share by period (BERTopic mts=30, sample n=7,730)")
ax.legend()
plt.tight_layout()
plt.savefig(FIGS / "archetype_proportion_by_period.png", dpi=150, bbox_inches="tight")
plt.close()

# Fig 4: Per-archetype entry share by period (junior share of known seniority)
log("Entry share trend plot")
# Keep only archetypes with enough known-seniority rows (>=15 in at least two periods)
ok_arch = []
for arch in entry_df["archetype"].unique():
    sub = entry_df[entry_df["archetype"] == arch]
    if (sub["n_known_seniority"] >= 15).sum() >= 2:
        ok_arch.append(arch)

fig, ax = plt.subplots(figsize=(12, 7))
for arch in ok_arch:
    sub = entry_df[entry_df["archetype"] == arch].sort_values("period_group")
    ax.plot(sub["period_group"], sub["junior_share_of_known"], marker="o", label=ARCHETYPE_NAMES[arch])
ax.set_ylabel("Junior share of known seniority")
ax.set_xlabel("Period")
ax.set_title("Per-archetype junior share over time (the Alt-B diagnostic)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(FIGS / "archetype_entry_share_trend.png", dpi=150, bbox_inches="tight")
plt.close()

# Fig 5: NMI bar chart
log("NMI bar chart")
fig, ax = plt.subplots(figsize=(8, 5))
nmi_plot = nmi_df.copy()
labels = nmi_plot["target"].tolist()
vals = nmi_plot["nmi_nonnoise"].tolist()
# Substitute industry-only for that row
ind_idx = nmi_plot["target"].tolist().index("company_industry")
vals[ind_idx] = nmi_plot.iloc[ind_idx]["nmi_nonnoise_ind_only"] or vals[ind_idx]
ax.bar(labels, vals, color=["tab:purple", "tab:blue", "tab:green", "tab:orange"])
ax.set_ylabel("NMI (cluster ↔ target)")
ax.set_title("Dominant-structure NMI: what does BERTopic cluster align with?")
ax.set_ylim(0, max(vals) * 1.15 + 0.02)
for i, v in enumerate(vals):
    ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig(FIGS / "nmi_verdict.png", dpi=150, bbox_inches="tight")
plt.close()

# Fig 6: Methods comparison — topic term overlap Jaccard heatmap
log("Method overlap heatmap")
overlap = pd.read_csv(TABLES / "method_overlap.csv")
# Build full pairwise jaccard
import itertools
bt = pd.read_csv(TABLES / "bertopic_topics.csv")
nmf_full = pd.read_csv(TABLES / "nmf_topics.csv")
nmf12 = nmf_full[nmf_full["k"] == 12]

bt_sets = {int(r["topic_id"]): set(r["top_terms"].split(" | ")) for _, r in bt.iterrows()}
nmf_sets = {int(r["component"]): set(r["top_terms"].split(" | ")) for _, r in nmf12.iterrows()}

M = np.zeros((len(bt_sets), len(nmf_sets)))
bt_ids = sorted(bt_sets)
nmf_ids = sorted(nmf_sets)
for i, b in enumerate(bt_ids):
    for j, n in enumerate(nmf_ids):
        a, c = bt_sets[b], nmf_sets[n]
        M[i, j] = len(a & c) / max(1, len(a | c))

fig, ax = plt.subplots(figsize=(9, 9))
im = ax.imshow(M, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(nmf_ids)))
ax.set_xticklabels([f"NMF-{n}" for n in nmf_ids], rotation=45)
ax.set_yticks(range(len(bt_ids)))
ax.set_yticklabels([f"T{b}: {ARCHETYPE_NAMES.get(b, '')[:30]}" for b in bt_ids], fontsize=8)
for i in range(len(bt_ids)):
    for j in range(len(nmf_ids)):
        if M[i, j] > 0.05:
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", color="white" if M[i, j] < 0.3 else "black", fontsize=7)
plt.colorbar(im, ax=ax, label="Jaccard overlap (top-20 terms)")
ax.set_title("BERTopic vs NMF-12 topic overlap")
plt.tight_layout()
plt.savefig(FIGS / "methods_overlap_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

log("Characterization + figures done.")
log(f"Primary NMI verdict:")
print(nmi_df.to_string(index=False))

# Save the characterized sample for the next script
sample_tech.to_parquet(ART / "T09_sample_characterized.parquet", index=False)
