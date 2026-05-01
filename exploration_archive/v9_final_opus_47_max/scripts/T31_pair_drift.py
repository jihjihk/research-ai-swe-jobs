"""T31 — Same-company × same-title longitudinal drift (H_M).

Tightens T16's within-employer rewriting signal to same-company × same-title pairs
across 2024 and 2026. Pair-level drift distinguishes posting-rewriting from
within-company title-composition noise.

Per dispatch:
  - Overlap panel: T16 arshkon_min3 / arshkon_min5 AND pooled_2024 ∩ scraped.
  - Pair identification via raw lowercased title (NOT title_normalized).
  - Thresholds: n>=3 per (company × title × period) primary; n>=2 secondary.
  - Drift metrics per pair (2024 -> 2026):
      * ai_strict_v1_rebuilt binary share (PRIMARY)
      * ai_broad (drop MCP from 2024 baseline)
      * requirement_breadth_resid (from T11_posting_features.parquet)
      * mgmt_strict_v1_rebuilt binary share (V1-validated)
      * requirements-section character share (T13_section_classifier)
      * mean yoe_min_years_llm (ablation: yoe_extracted)
      * median description_cleaned_length
  - Drift distribution: mean/median/p10/p90 per metric + 2D scatter.
  - Archetype stratification via nearest-centroid projection on T09 labels.
  - Top-20 exemplars: AI-mention Δ + breadth-resid Δ.
  - Consistency vs T16 (company-level).
  - Sensitivities: (a) aggregator exclusion, (b) cap 10 postings per cell.

Outputs:
  exploration/tables/T31/pair_drift.csv          per-pair deltas
  exploration/tables/T31/drift_distribution.csv  distribution summary
  exploration/tables/T31/top20_ai.csv
  exploration/tables/T31/top20_breadth.csv
  exploration/tables/T31/archetype_stratified.csv
  exploration/tables/T31/consistency_vs_t16.csv
  exploration/tables/T31/sensitivity_aggregator.csv
  exploration/tables/T31/sensitivity_cap10.csv
  exploration/figures/T31_drift_scatter.png (+ svg)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/jihgaboot/gabor/job-research")
DATA = REPO / "data" / "unified_core.parquet"
CLEANED = REPO / "exploration" / "artifacts" / "shared" / "swe_cleaned_text.parquet"
T11F = REPO / "exploration" / "artifacts" / "shared" / "T11_posting_features.parquet"
ARCH = REPO / "exploration" / "artifacts" / "shared" / "swe_archetype_labels.parquet"
EMB_NPY = REPO / "exploration" / "artifacts" / "shared" / "swe_embeddings.npy"
EMB_IDX = REPO / "exploration" / "artifacts" / "shared" / "swe_embedding_index.parquet"
PANEL = REPO / "exploration" / "tables" / "T16" / "overlap_panel.csv"
CHANGE_VEC = REPO / "exploration" / "tables" / "T16" / "company_change_vectors.csv"
VAL = REPO / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json"

OUT = REPO / "exploration" / "tables" / "T31"
OUT.mkdir(parents=True, exist_ok=True)
FIG = REPO / "exploration" / "figures"
FIG.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO / "exploration" / "scripts"))
from T13_section_classifier import classify_sections  # type: ignore  # noqa: E402

# --------------------------------------------------------------------------------------
# Patterns from validated_mgmt_patterns.json
# --------------------------------------------------------------------------------------
patterns = json.loads(VAL.read_text())
AI_STRICT_V1 = re.compile(patterns["v1_rebuilt_patterns"]["ai_strict_v1_rebuilt"]["pattern"], re.IGNORECASE)
MGMT_STRICT_V1 = re.compile(patterns["v1_rebuilt_patterns"]["mgmt_strict_v1_rebuilt"]["pattern"], re.IGNORECASE)

# ai_broad without MCP (per Gate 2 pre-commit #9: drop MCP from 2024 baseline)
AI_BROAD_NO_MCP = re.compile(
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|"
    r"llamaindex|langchain|prompt engineering|rag|vector databas(?:e|es)|"
    r"pinecone|huggingface|hugging face|agent|machine learning|ml|ai|llm|"
    r"artificial intelligence|"
    r"(?:fine[- ]tun(?:e|ed|ing))\s+(?:the\s+)?(?:model|llm|gpt|base model|foundation model|embeddings))\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------------------
# 1. Load overlap panel + postings joined with T11 features
# --------------------------------------------------------------------------------------

print("[T31] Loading overlap panel...")
panel_df = pd.read_csv(PANEL)
arshkon_companies = set(panel_df[panel_df["panel_type"] == "arshkon_min3"]["company_name_canonical"])
arshkon_min5_companies = set(panel_df[panel_df["panel_type"] == "arshkon_min5"]["company_name_canonical"])
pooled_companies = set(panel_df[panel_df["panel_type"] == "pooled_min5"]["company_name_canonical"])
print(f"  arshkon_min3: {len(arshkon_companies)}, arshkon_min5: {len(arshkon_min5_companies)}, pooled_min5: {len(pooled_companies)}")
all_panel_cos = arshkon_companies | arshkon_min5_companies | pooled_companies
print(f"  Union: {len(all_panel_cos)} companies")

aggregator_cos = set(panel_df[panel_df["is_aggregator"]]["company_name_canonical"])
print(f"  Aggregators in panel: {len(aggregator_cos)}")

con = duckdb.connect()

print("[T31] Loading SWE posting base...")
q = """
SELECT uid,
       source,
       period,
       CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period_year,
       company_name_canonical,
       is_aggregator,
       LOWER(title) AS title_lc,
       title,
       description,
       description_core_llm,
       description_length,
       llm_extraction_coverage,
       llm_classification_coverage,
       seniority_final,
       yoe_extracted,
       yoe_min_years_llm
FROM read_parquet('{path}')
WHERE source_platform='linkedin'
  AND is_english
  AND date_flag='ok'
  AND is_swe
""".format(path=str(DATA))
base = con.execute(q).fetchdf()
base = base[base["company_name_canonical"].isin(all_panel_cos)].copy()
print(f"  base rows (all panel cos): {len(base)}")
print(f"  unique companies: {base['company_name_canonical'].nunique()}, unique titles: {base['title_lc'].nunique()}")

# Join T11 features
print("[T31] Joining T11 features...")
t11 = con.execute(f"SELECT * FROM read_parquet('{T11F}')").fetchdf()
t11 = t11[["uid", "requirement_breadth_resid", "ai_binary", "description_cleaned_length", "mgmt_broad_density"]]
base = base.merge(t11, on="uid", how="left")
print(f"  after T11 join: {len(base)}; null breadth_resid rate: {base['requirement_breadth_resid'].isna().mean():.3f}")

# Archetype labels (T09 labeled 8k only; project via nearest centroid for rest)
print("[T31] Loading archetype labels...")
arch = con.execute(f"SELECT uid, archetype, archetype_name FROM read_parquet('{ARCH}') WHERE archetype != -1").fetchdf()
base = base.merge(arch, on="uid", how="left")
# For top-level drift-metric computation we don't need archetype; only for stratification.

# --------------------------------------------------------------------------------------
# 2. Compute per-posting text-based features we need that aren't in T11
# --------------------------------------------------------------------------------------

def pick_text(row):
    if isinstance(row.get("description_core_llm"), str) and row.get("llm_extraction_coverage") == "labeled":
        return row["description_core_llm"]
    return row.get("description") or ""

print("[T31] Computing per-posting binaries (ai_strict_v1, ai_broad_no_mcp, mgmt_strict_v1, reqs_share)...")
# Use raw description for binary presence (boilerplate-insensitive).
desc_raw = base["description"].fillna("").astype(str)
base["ai_strict_v1"] = desc_raw.str.contains(AI_STRICT_V1).astype(int)
base["ai_broad_nmcp"] = desc_raw.str.contains(AI_BROAD_NO_MCP).astype(int)
base["mgmt_strict_v1"] = desc_raw.str.contains(MGMT_STRICT_V1).astype(int)

# Requirements-section character share via T13 classifier.
print("[T31] Computing requirements-section char share via T13 classifier...")
req_shares = []
for text in desc_raw.values:
    sec = classify_sections(text)
    total = max(1, sum(sec.values()))
    req_shares.append(sec.get("requirements", 0) / total)
base["reqs_share_t13"] = req_shares

# length column mapping: description_cleaned_length (from T11). Fallback: description_length
base["desc_len_any"] = base["description_cleaned_length"].fillna(base["description_length"])

# YOE primary (LLM) and ablation (rule)
base["yoe_llm"] = base["yoe_min_years_llm"].where(base["llm_classification_coverage"] == "labeled")
base["yoe_rule"] = base["yoe_extracted"]

# --------------------------------------------------------------------------------------
# 3. Pair identification
# --------------------------------------------------------------------------------------

def build_pairs(df: pd.DataFrame, min_n: int = 3, companies: set = None, cap: int | None = None,
                exclude_aggregators: bool = False) -> pd.DataFrame:
    """Return per-pair metric table.

    Each row is a (company, title_lc) that appears with n>=min_n in BOTH periods.
    Metrics: mean/median over the cell.
    """
    sub = df.copy()
    if companies is not None:
        sub = sub[sub["company_name_canonical"].isin(companies)]
    if exclude_aggregators:
        sub = sub[~sub["is_aggregator"].fillna(False)]

    # Optional cap: random sample cap postings per (co × title × period).
    if cap is not None:
        sub = sub.groupby(["company_name_canonical", "title_lc", "period_year"], group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), cap), random_state=42)
        )

    # Counts per cell
    counts = sub.groupby(["company_name_canonical", "title_lc", "period_year"]).size().reset_index(name="n")
    wide = counts.pivot_table(index=["company_name_canonical", "title_lc"],
                              columns="period_year", values="n", aggfunc="sum").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"2024": "n_2024", "2026": "n_2026"})
    wide = wide.dropna(subset=["n_2024", "n_2026"])
    wide = wide[(wide["n_2024"] >= min_n) & (wide["n_2026"] >= min_n)]

    if len(wide) == 0:
        return pd.DataFrame()

    # Aggregate metrics per cell
    agg_dict = {
        "ai_strict_v1": "mean",
        "ai_broad_nmcp": "mean",
        "mgmt_strict_v1": "mean",
        "requirement_breadth_resid": "mean",
        "reqs_share_t13": "mean",
        "yoe_llm": "mean",
        "yoe_rule": "mean",
        "desc_len_any": "median",
    }
    per_cell = sub.groupby(["company_name_canonical", "title_lc", "period_year"]).agg(agg_dict).reset_index()
    # pivot on period_year
    out_cols = []
    for metric in agg_dict:
        pv = per_cell.pivot_table(index=["company_name_canonical", "title_lc"],
                                  columns="period_year", values=metric).reset_index()
        pv.columns.name = None
        pv = pv.rename(columns={"2024": f"{metric}_2024", "2026": f"{metric}_2026"})
        out_cols.append(pv)

    pair = out_cols[0]
    for df_ in out_cols[1:]:
        pair = pair.merge(df_, on=["company_name_canonical", "title_lc"], how="outer")
    pair = wide.merge(pair, on=["company_name_canonical", "title_lc"], how="left")

    # Deltas
    for metric in agg_dict:
        if metric != "desc_len_any":
            pair[f"{metric}_delta"] = pair[f"{metric}_2026"] - pair[f"{metric}_2024"]
        else:
            pair[f"{metric}_delta"] = pair[f"{metric}_2026"] - pair[f"{metric}_2024"]

    # attach is_aggregator for pair's company
    agg_map = sub[["company_name_canonical", "is_aggregator"]].drop_duplicates("company_name_canonical").set_index("company_name_canonical")
    pair["is_aggregator"] = pair["company_name_canonical"].map(agg_map["is_aggregator"]).fillna(False)
    return pair


print("[T31] Building pairs at n>=3 (primary)...")
pairs_arshkon3_n3 = build_pairs(base, min_n=3, companies=arshkon_companies)
pairs_pooled_n3 = build_pairs(base, min_n=3, companies=pooled_companies)
print(f"  arshkon_min3 pairs n>=3: {len(pairs_arshkon3_n3)}")
print(f"  pooled_min5  pairs n>=3: {len(pairs_pooled_n3)}")

print("[T31] Building pairs at n>=2 (secondary)...")
pairs_arshkon3_n2 = build_pairs(base, min_n=2, companies=arshkon_companies)
pairs_pooled_n2 = build_pairs(base, min_n=2, companies=pooled_companies)
print(f"  arshkon_min3 pairs n>=2: {len(pairs_arshkon3_n2)}")
print(f"  pooled_min5  pairs n>=2: {len(pairs_pooled_n2)}")

# Annotate which panel each pair belongs to, save primary (n>=3) combined table.
pairs_arshkon3_n3["panel"] = "arshkon_min3_n3"
pairs_pooled_n3["panel"] = "pooled_min5_n3"
pairs_arshkon3_n2["panel"] = "arshkon_min3_n2"
pairs_pooled_n2["panel"] = "pooled_min5_n2"

# We save all_pairs AFTER archetype_id is attached (see step 6 below).

# --------------------------------------------------------------------------------------
# 4. Drift distribution summary
# --------------------------------------------------------------------------------------
def distribution_summary(pairs: pd.DataFrame, tag: str) -> pd.DataFrame:
    if len(pairs) == 0:
        return pd.DataFrame()
    metrics = ["ai_strict_v1", "ai_broad_nmcp", "mgmt_strict_v1",
               "requirement_breadth_resid", "reqs_share_t13", "yoe_llm", "yoe_rule", "desc_len_any"]
    rows = []
    for m in metrics:
        col = f"{m}_delta"
        if col not in pairs.columns:
            continue
        v = pairs[col].dropna()
        if len(v) == 0:
            continue
        rows.append({
            "panel": tag,
            "metric": m,
            "n_pairs": len(v),
            "mean": float(v.mean()),
            "median": float(v.median()),
            "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)),
            "std": float(v.std()),
        })
    return pd.DataFrame(rows)


dist = pd.concat([
    distribution_summary(pairs_arshkon3_n3, "arshkon_min3_n3"),
    distribution_summary(pairs_pooled_n3, "pooled_min5_n3"),
    distribution_summary(pairs_arshkon3_n2, "arshkon_min3_n2"),
    distribution_summary(pairs_pooled_n2, "pooled_min5_n2"),
], ignore_index=True)
dist.to_csv(OUT / "drift_distribution.csv", index=False)
print(f"[T31] Saved drift_distribution.csv ({len(dist)} rows)")
print(dist.to_string())

# --------------------------------------------------------------------------------------
# 5. 2D scatter of (AI-mention Δ × breadth-resid Δ)
# --------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, pairs_, tag in zip(axes, [pairs_arshkon3_n3, pairs_pooled_n3], ["arshkon_min3 (n>=3)", "pooled_min5 (n>=3)"]):
    if len(pairs_) == 0:
        ax.text(0.5, 0.5, f"{tag}: no pairs", ha="center", va="center")
        ax.set_title(tag)
        continue
    x = pairs_["ai_strict_v1_delta"].astype(float)
    y = pairs_["requirement_breadth_resid_delta"].astype(float)
    mask = (~x.isna()) & (~y.isna())
    ax.scatter(x[mask], y[mask], alpha=0.35, s=14, c="tab:blue")
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.axvline(0, color="grey", lw=0.6, ls="--")
    ax.set_xlabel("Δ ai_strict_v1 (2026 − 2024)")
    ax.set_ylabel("Δ requirement_breadth_resid")
    ax.set_title(f"{tag}  n={mask.sum()}")
    mx, my = x[mask].mean(), y[mask].mean()
    ax.plot(mx, my, marker="x", color="red", markersize=12)
    ax.annotate(f"mean = ({mx:.3f}, {my:.2f})", xy=(mx, my), xytext=(5, 10),
                textcoords="offset points", fontsize=8, color="red")
fig.suptitle("T31: Same-company × same-title drift — AI prevalence Δ vs breadth_resid Δ", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "T31_drift_scatter.png", dpi=150, bbox_inches="tight")
fig.savefig(FIG / "T31_drift_scatter.svg", bbox_inches="tight")
plt.close(fig)
print("[T31] Saved T31_drift_scatter.png / .svg")

# --------------------------------------------------------------------------------------
# 6. Archetype stratification via nearest-centroid from swe_embeddings.npy
#    Approach: attach per-pair a 'representative posting' (most recent 2026 entry
#    in the cell), look up its archetype (if T09-labeled) or project by nearest
#    centroid of T09 archetype centroids.
# --------------------------------------------------------------------------------------
print("[T31] Archetype stratification (nearest-centroid over T09)...")
emb = np.load(EMB_NPY)
emb_idx = con.execute(f"SELECT row_idx, uid FROM read_parquet('{EMB_IDX}')").fetchdf()
emb_uid_to_row = dict(zip(emb_idx["uid"], emb_idx["row_idx"]))
arch_full = con.execute(f"SELECT uid, archetype FROM read_parquet('{ARCH}')").fetchdf()
arch_map = dict(zip(arch_full["uid"], arch_full["archetype"]))

# Compute centroids per archetype from labeled rows (non-noise)
labeled = [(arch_map[uid], emb_uid_to_row[uid])
           for uid in arch_full["uid"].values
           if uid in emb_uid_to_row and arch_map[uid] != -1]
if labeled:
    arch_ids = sorted(set(a for a, _ in labeled))
    centroids = {}
    for aid in arch_ids:
        rows = [r for a, r in labeled if a == aid]
        centroids[aid] = emb[rows].mean(axis=0)
    cent_mat = np.stack([centroids[a] for a in arch_ids])
    cent_norm = cent_mat / np.linalg.norm(cent_mat, axis=1, keepdims=True)
else:
    cent_norm = None
    arch_ids = []

def archetype_for_uid(uid: str) -> int | None:
    if uid in arch_map and arch_map[uid] != -1:
        return int(arch_map[uid])
    if cent_norm is None or uid not in emb_uid_to_row:
        return None
    v = emb[emb_uid_to_row[uid]]
    v = v / (np.linalg.norm(v) + 1e-9)
    sims = cent_norm @ v
    return int(arch_ids[int(np.argmax(sims))])

# Archetype-name map for output
arch_names = arch_full.drop_duplicates("archetype").set_index("archetype").to_dict() if "archetype_name" in arch_full.columns else {}

# For each pair, pick representative 2026 posting: highest desc_len_any (as a proxy)
def pair_archetype(pairs: pd.DataFrame, postings: pd.DataFrame) -> pd.DataFrame:
    if len(pairs) == 0:
        return pd.DataFrame()
    # Build map (co, title_lc) -> first uid (2026) with embedding
    post2026 = postings[postings["period_year"] == "2026"].copy()
    post2026["has_emb"] = post2026["uid"].isin(emb_uid_to_row)
    reps = (post2026[post2026["has_emb"]]
            .sort_values("description_length", ascending=False)
            .drop_duplicates(["company_name_canonical", "title_lc"], keep="first"))
    rep_map = reps.set_index(["company_name_canonical", "title_lc"])["uid"].to_dict()
    pairs["rep_uid_2026"] = pairs.apply(
        lambda r: rep_map.get((r["company_name_canonical"], r["title_lc"])), axis=1
    )
    pairs["archetype_id"] = pairs["rep_uid_2026"].apply(
        lambda u: archetype_for_uid(u) if isinstance(u, str) else None
    )
    return pairs


def archetype_drift_summary(pairs: pd.DataFrame, tag: str) -> pd.DataFrame:
    if len(pairs) == 0:
        return pd.DataFrame()
    metrics = ["ai_strict_v1_delta", "requirement_breadth_resid_delta",
               "mgmt_strict_v1_delta", "reqs_share_t13_delta", "desc_len_any_delta"]
    out = []
    for aid, g in pairs.groupby("archetype_id"):
        if len(g) < 3:
            continue
        # archetype name lookup
        name_rows = arch_full[arch_full["archetype"] == aid]
        # archetype_name isn't in arch_full by default — load separately
        name = None
        if aid is not None and aid in set(arch["archetype"]):
            name = arch[arch["archetype"] == aid]["archetype_name"].iloc[0]
        row = {
            "panel": tag,
            "archetype_id": int(aid) if pd.notna(aid) else -999,
            "archetype_name": name,
            "n_pairs": len(g),
        }
        for m in metrics:
            v = g[m].dropna()
            if len(v) > 0:
                row[f"mean_{m}"] = float(v.mean())
                row[f"median_{m}"] = float(v.median())
        out.append(row)
    return pd.DataFrame(out)


pairs_arshkon3_n3 = pair_archetype(pairs_arshkon3_n3, base)
pairs_pooled_n3 = pair_archetype(pairs_pooled_n3, base)
pairs_arshkon3_n2 = pair_archetype(pairs_arshkon3_n2, base)
pairs_pooled_n2 = pair_archetype(pairs_pooled_n2, base)

# Now that archetype_id is attached, save pair_drift.csv
all_pairs = pd.concat([pairs_arshkon3_n3, pairs_pooled_n3, pairs_arshkon3_n2, pairs_pooled_n2], ignore_index=True)
all_pairs.to_csv(OUT / "pair_drift.csv", index=False)
print(f"[T31] Saved pair_drift.csv ({len(all_pairs)} rows)")

arch_table = pd.concat([
    archetype_drift_summary(pairs_arshkon3_n3, "arshkon_min3_n3"),
    archetype_drift_summary(pairs_pooled_n3, "pooled_min5_n3"),
    archetype_drift_summary(pairs_arshkon3_n2, "arshkon_min3_n2"),
    archetype_drift_summary(pairs_pooled_n2, "pooled_min5_n2"),
], ignore_index=True)
# sort by n_pairs and mean_ai delta
arch_table = arch_table.sort_values(["panel", "mean_ai_strict_v1_delta"], ascending=[True, False])
arch_table.to_csv(OUT / "archetype_stratified.csv", index=False)
print(f"[T31] Saved archetype_stratified.csv ({len(arch_table)} rows)")

# --------------------------------------------------------------------------------------
# 7. Top-20 exemplars
# --------------------------------------------------------------------------------------
def top20(pairs: pd.DataFrame, by: str, tag: str) -> pd.DataFrame:
    if len(pairs) == 0:
        return pd.DataFrame()
    # order columns, dedupe (f"{by}_delta" could match ai_strict_v1_delta when
    # by==ai_strict_v1; use list(dict.fromkeys(..)))
    keep_raw = ["company_name_canonical", "title_lc", "n_2024", "n_2026",
            f"{by}_delta", f"{by}_2024", f"{by}_2026",
            "ai_strict_v1_delta", "requirement_breadth_resid_delta",
            "reqs_share_t13_delta", "desc_len_any_delta",
            "is_aggregator", "archetype_id"]
    keep = list(dict.fromkeys([c for c in keep_raw if c in pairs.columns]))
    t = pairs.sort_values(f"{by}_delta", ascending=False).head(20)[keep].copy()
    t["panel"] = tag
    return t

top20_ai = pd.concat([
    top20(pairs_arshkon3_n3, "ai_strict_v1", "arshkon_min3_n3"),
    top20(pairs_pooled_n3, "ai_strict_v1", "pooled_min5_n3"),
], ignore_index=True)
top20_ai.to_csv(OUT / "top20_ai.csv", index=False)

top20_breadth = pd.concat([
    top20(pairs_arshkon3_n3, "requirement_breadth_resid", "arshkon_min3_n3"),
    top20(pairs_pooled_n3, "requirement_breadth_resid", "pooled_min5_n3"),
], ignore_index=True)
top20_breadth.to_csv(OUT / "top20_breadth.csv", index=False)
print(f"[T31] Saved top20_ai.csv ({len(top20_ai)} rows) and top20_breadth.csv ({len(top20_breadth)} rows)")

# --------------------------------------------------------------------------------------
# 8. Consistency vs T16 (company-level)
# --------------------------------------------------------------------------------------
print("[T31] Consistency check vs T16 company-level deltas...")
cv = pd.read_csv(CHANGE_VEC)
# Only pooled_min5 rows in change_vectors - keep as-is
# Merge per-company aggregate of pair deltas (weighted by n_2024 + n_2026) and compare.

def company_level_from_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    if len(pairs) == 0:
        return pd.DataFrame()
    # weight: sqrt(n_2024 * n_2026) to keep both periods contributing
    pairs = pairs.copy()
    pairs["weight"] = np.sqrt(pairs["n_2024"].clip(lower=1) * pairs["n_2026"].clip(lower=1))
    g = pairs.groupby("company_name_canonical")
    def wmean(df, col):
        v = df[col].dropna()
        w = df.loc[v.index, "weight"]
        if w.sum() == 0:
            return np.nan
        return (v * w).sum() / w.sum()
    out = g.apply(lambda d: pd.Series({
        "pair_ai_strict_delta": wmean(d, "ai_strict_v1_delta"),
        "pair_breadth_resid_delta": wmean(d, "requirement_breadth_resid_delta"),
        "n_pairs": len(d),
    })).reset_index()
    return out


pool_pair_co = company_level_from_pairs(pairs_pooled_n3)
arsh_pair_co = company_level_from_pairs(pairs_arshkon3_n3)

# T16 company-level (ai_prevalence_delta_strict uses T11 ai_binary; breadth_resid_delta direct).
cv_pool = cv[cv["panel_type"] == "pooled_min5"][
    ["company_name_canonical", "ai_prevalence_delta_strict", "breadth_resid_delta"]
].rename(columns={"ai_prevalence_delta_strict": "t16_ai_delta", "breadth_resid_delta": "t16_breadth_delta"})

cons_pool = pool_pair_co.merge(cv_pool, on="company_name_canonical", how="inner")
# spearman + summary
from scipy.stats import spearmanr

def _summary(df, name):
    r_ai, _ = spearmanr(df["pair_ai_strict_delta"], df["t16_ai_delta"], nan_policy="omit")
    r_br, _ = spearmanr(df["pair_breadth_resid_delta"], df["t16_breadth_delta"], nan_policy="omit")
    ai_mean_pair = df["pair_ai_strict_delta"].mean()
    ai_mean_t16 = df["t16_ai_delta"].mean()
    br_mean_pair = df["pair_breadth_resid_delta"].mean()
    br_mean_t16 = df["t16_breadth_delta"].mean()
    return {
        "panel": name,
        "n_companies": len(df),
        "spearman_ai_pair_vs_t16": r_ai,
        "spearman_breadth_pair_vs_t16": r_br,
        "mean_ai_pair": ai_mean_pair,
        "mean_ai_t16": ai_mean_t16,
        "mean_breadth_pair": br_mean_pair,
        "mean_breadth_t16": br_mean_t16,
        "pair_less_than_t16_ai": ai_mean_pair < ai_mean_t16,
        "pair_less_than_t16_breadth": br_mean_pair < br_mean_t16,
    }

cons_rows = []
if len(cons_pool) > 0:
    cons_rows.append(_summary(cons_pool, "pooled_min5_n3"))

# Arshkon change_vectors entries (only pooled_min5 panel_type exists in T16 CSV)
if len(arsh_pair_co) > 0:
    arsh_merge = arsh_pair_co.merge(cv_pool, on="company_name_canonical", how="inner")
    if len(arsh_merge) > 0:
        cons_rows.append(_summary(arsh_merge, "arshkon_min3_n3_vs_t16_pooled"))

cons_df = pd.DataFrame(cons_rows)
cons_df.to_csv(OUT / "consistency_vs_t16.csv", index=False)
print(cons_df.to_string())

# --------------------------------------------------------------------------------------
# 9. Sensitivity (a) aggregator exclusion; (b) cap 10 per cell
# --------------------------------------------------------------------------------------
print("[T31] Sensitivities...")

pairs_pool_noagg = build_pairs(base, min_n=3, companies=pooled_companies, exclude_aggregators=True)
sens_agg_rows = []
if len(pairs_pool_noagg) > 0:
    # compare means to with-aggregators
    for m in ["ai_strict_v1", "requirement_breadth_resid", "mgmt_strict_v1", "reqs_share_t13", "desc_len_any"]:
        base_mean = pairs_pooled_n3[f"{m}_delta"].mean()
        sens_mean = pairs_pool_noagg[f"{m}_delta"].mean()
        sens_agg_rows.append({
            "panel": "pooled_min5_n3",
            "metric": m,
            "mean_with_agg": base_mean,
            "mean_without_agg": sens_mean,
            "n_with": int((~pairs_pooled_n3[f"{m}_delta"].isna()).sum()),
            "n_without": int((~pairs_pool_noagg[f"{m}_delta"].isna()).sum()),
            "delta": sens_mean - base_mean,
        })
pd.DataFrame(sens_agg_rows).to_csv(OUT / "sensitivity_aggregator.csv", index=False)
print(f"[T31] Saved sensitivity_aggregator.csv ({len(sens_agg_rows)} rows)")

pairs_pool_cap10 = build_pairs(base, min_n=3, companies=pooled_companies, cap=10)
sens_cap_rows = []
if len(pairs_pool_cap10) > 0:
    for m in ["ai_strict_v1", "requirement_breadth_resid", "mgmt_strict_v1", "reqs_share_t13", "desc_len_any"]:
        base_mean = pairs_pooled_n3[f"{m}_delta"].mean()
        sens_mean = pairs_pool_cap10[f"{m}_delta"].mean()
        sens_cap_rows.append({
            "panel": "pooled_min5_n3",
            "metric": m,
            "mean_uncapped": base_mean,
            "mean_cap10": sens_mean,
            "n_uncapped": int((~pairs_pooled_n3[f"{m}_delta"].isna()).sum()),
            "n_cap10": int((~pairs_pool_cap10[f"{m}_delta"].isna()).sum()),
            "delta": sens_mean - base_mean,
        })
pd.DataFrame(sens_cap_rows).to_csv(OUT / "sensitivity_cap10.csv", index=False)
print(f"[T31] Saved sensitivity_cap10.csv ({len(sens_cap_rows)} rows)")

print("[T31] DONE.")
