"""T20 — Seniority boundary clarity.

Builds per-posting feature vectors, trains logistic regression classifiers for
each adjacent seniority boundary (entry↔associate, associate↔mid-senior,
mid-senior↔director), separately for 2024 and 2026, and reports AUC and
top features.

Per Gate 2 corrections, scope_density uses only `end-to-end` and
`cross-functional` (bare `ownership` is 58% precision), and management density
uses the T21-rebuilt validated patterns. This script imports the pattern set
produced by T21's validation script; if that artifact is not yet present, it
falls back to a narrow mentor-only pattern (which survives validation per V1).

Inputs:
  - exploration/artifacts/shared/swe_cleaned_text.parquet
  - exploration/artifacts/shared/swe_tech_matrix.parquet
  - exploration/artifacts/shared/swe_archetype_labels.parquet
  - data/unified.parquet (for yoe_extracted and education level extraction)

Outputs:
  - exploration/tables/T20/auc_by_period_boundary.csv
  - exploration/tables/T20/feature_importance_by_period_boundary.csv
  - exploration/tables/T20/feature_heatmap_seniority_period.csv
  - exploration/tables/T20/auc_by_archetype.csv
  - exploration/tables/T20/auc_sensitivity.csv
  - exploration/figures/T20/auc_comparison.png
  - exploration/figures/T20/feature_heatmap.png
  - exploration/figures/T20/auc_by_archetype.png
  - exploration/figures/T20/feature_importance_change.png
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/jihgaboot/gabor/job-research")
SHARED = ROOT / "exploration/artifacts/shared"
T20_TABLES = ROOT / "exploration/tables/T20"
T20_FIGS = ROOT / "exploration/figures/T20"
T20_TABLES.mkdir(parents=True, exist_ok=True)
T20_FIGS.mkdir(parents=True, exist_ok=True)

# --- Inline assertions for pattern hygiene ---------------------------------

def count_re(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


# Scope pattern: end-to-end + cross-functional only (per Gate 2 correction 3)
SCOPE_RE = re.compile(r"\b(?:end[-\s]?to[-\s]?end|cross[-\s]?functional)\b", re.IGNORECASE)

assert SCOPE_RE.search("end-to-end ownership") is not None
assert SCOPE_RE.search("cross functional team") is not None
assert SCOPE_RE.search("end to end pipeline") is not None
assert SCOPE_RE.search("ownership mindset") is None  # ownership NOT in pattern
assert SCOPE_RE.search("employee-owned company") is None

# Narrow/validated mgmt pattern: attempt to load T21 validated patterns.
VALIDATED_PATH = SHARED / "validated_mgmt_patterns.json"
if VALIDATED_PATH.exists():
    with VALIDATED_PATH.open() as f:
        VAL = json.load(f)
    MGMT_PATTERN = re.compile(VAL["mgmt_combined_regex"], re.IGNORECASE)
    print(f"[T20] Loaded validated mgmt pattern from {VALIDATED_PATH}")
else:
    # Fallback: mentor-only (surviving component per V1 / Gate 2 corrections).
    MGMT_PATTERN = re.compile(r"\b(?:mentor|mentors|mentored|mentoring|mentorship)\b", re.IGNORECASE)
    print("[T20] validated_mgmt_patterns.json not found; using mentor-only fallback")

assert MGMT_PATTERN.search("mentor junior engineers") is not None


EDU_RE = {
    "phd": re.compile(r"\b(?:ph\.?d\.?|doctorate|doctoral)\b", re.IGNORECASE),
    "ms": re.compile(r"\b(?:m\.?s\.?|master'?s?|msc)\b", re.IGNORECASE),
    "bs": re.compile(r"\b(?:b\.?s\.?|bachelor'?s?|bsc|undergraduate degree)\b", re.IGNORECASE),
}

assert EDU_RE["phd"].search("PhD in computer science") is not None
assert EDU_RE["ms"].search("Master's degree required") is not None
assert EDU_RE["bs"].search("Bachelor's in engineering") is not None


def education_level(text: str) -> int:
    if not text:
        return 0
    if EDU_RE["phd"].search(text):
        return 3
    if EDU_RE["ms"].search(text):
        return 2
    if EDU_RE["bs"].search(text):
        return 1
    return 0


# --- Load data --------------------------------------------------------------

print("[T20] Loading shared artifacts")

con = duckdb.connect()

# AI keywords: use tech matrix ai columns (broad AI mention = any of these)
AI_COLS = [
    "llm", "langchain", "langgraph", "rag", "vector_db", "pinecone", "chromadb",
    "huggingface", "openai_api", "claude_api", "prompt_engineering", "fine_tuning",
    "mcp", "agents_framework", "gpt", "transformer_arch", "embedding", "copilot",
    "cursor_tool", "chatgpt", "claude_tool", "gemini_tool", "codex_tool",
    "machine_learning", "deep_learning", "nlp", "computer_vision",
]

# Tech count excludes agile/scrum/kanban/tdd (methodologies) per T11 convention
METHOD_COLS = {"agile", "scrum", "kanban", "tdd"}

# Load tech matrix -> per-row tech_count + ai_mention
tm = pd.read_parquet(SHARED / "swe_tech_matrix.parquet")
tech_col_names = [c for c in tm.columns if c != "uid" and c not in METHOD_COLS]
ai_col_names = [c for c in AI_COLS if c in tm.columns]
tm_tech = tm[tech_col_names].sum(axis=1)
tm_ai = tm[ai_col_names].any(axis=1).astype(int)
tech_df = pd.DataFrame({"uid": tm["uid"], "tech_count": tm_tech, "ai_mention": tm_ai})

# Load cleaned text (descriptions + metadata)
ct = pd.read_parquet(
    SHARED / "swe_cleaned_text.parquet",
    columns=[
        "uid", "description_cleaned", "text_source", "period", "seniority_final",
        "is_aggregator", "yoe_extracted", "swe_classification_tier",
    ],
)
ct = ct[ct["text_source"] == "llm"].copy()
print(f"[T20] LLM text frame: {len(ct):,}")

# Consolidate period
ct["period2"] = ct["period"].astype(str).str[:4]
ct = ct[ct["period2"].isin(["2024", "2026"])].copy()

# Join tech
df = ct.merge(tech_df, on="uid", how="left")
df["tech_count"] = df["tech_count"].fillna(0).astype(int)
df["ai_mention"] = df["ai_mention"].fillna(0).astype(int)

# Compute per-row density features
texts = df["description_cleaned"].fillna("").tolist()
desc_len_chars = np.array([len(t) for t in texts], dtype=float)
desc_len_k = np.maximum(desc_len_chars, 200.0) / 1000.0  # floor 200 chars

scope_counts = np.array([len(SCOPE_RE.findall(t)) for t in texts], dtype=float)
mgmt_counts = np.array([len(MGMT_PATTERN.findall(t)) for t in texts], dtype=float)
edu_levels = np.array([education_level(t) for t in texts], dtype=int)

df["desc_len_chars"] = desc_len_chars
df["scope_density"] = scope_counts / desc_len_k
df["mgmt_density"] = mgmt_counts / desc_len_k
df["education_level"] = edu_levels

# YOE: impute median per period
med_yoe_2024 = df.loc[df["period2"] == "2024", "yoe_extracted"].median()
med_yoe_2026 = df.loc[df["period2"] == "2026", "yoe_extracted"].median()
df["yoe_feat"] = df["yoe_extracted"].copy()
df.loc[(df["period2"] == "2024") & df["yoe_feat"].isna(), "yoe_feat"] = med_yoe_2024
df.loc[(df["period2"] == "2026") & df["yoe_feat"].isna(), "yoe_feat"] = med_yoe_2026
df["yoe_missing"] = df["yoe_extracted"].isna().astype(int)

# Log length for scale robustness
df["log_len"] = np.log1p(df["desc_len_chars"])

# Archetype
arch = pd.read_parquet(SHARED / "swe_archetype_labels.parquet")
df = df.merge(arch, on="uid", how="left")

print("[T20] Feature frame ready")

FEATURE_COLS = [
    "yoe_feat", "yoe_missing", "tech_count", "ai_mention", "scope_density",
    "mgmt_density", "log_len", "education_level",
]

# --- Boundary classifier ---------------------------------------------------


BOUNDARIES = [
    ("entry", "associate"),
    ("entry", "mid-senior"),  # the meaningful "junior vs mid" cut
    ("associate", "mid-senior"),
    ("mid-senior", "director"),
]


def boundary_auc(
    frame: pd.DataFrame, a: str, b: str, feature_cols: list[str], tag: str,
) -> dict:
    sub = frame[frame["seniority_final"].isin([a, b])].copy()
    sub = sub.dropna(subset=feature_cols)
    n_a = int((sub["seniority_final"] == a).sum())
    n_b = int((sub["seniority_final"] == b).sum())
    if n_a < 15 or n_b < 15:
        return {
            "boundary": f"{a}->{b}", "tag": tag, "n_a": n_a, "n_b": n_b,
            "auc_mean": np.nan, "auc_std": np.nan, "top_features": "",
            "coef": {}, "n_folds": 0,
        }
    X = sub[feature_cols].to_numpy(dtype=float)
    y = (sub["seniority_final"] == b).astype(int).to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    n_splits = 5
    if min(n_a, n_b) < n_splits:
        n_splits = max(2, min(n_a, n_b))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    coefs = np.zeros(len(feature_cols))
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        coefs += clf.coef_.ravel()
    coefs /= n_splits
    ordering = np.argsort(-np.abs(coefs))
    top = [(feature_cols[i], float(coefs[i])) for i in ordering[:5]]
    top_str = "; ".join(f"{f}:{c:+.2f}" for f, c in top)
    coef_map = {feature_cols[i]: float(coefs[i]) for i in range(len(feature_cols))}
    return {
        "boundary": f"{a}->{b}", "tag": tag, "n_a": n_a, "n_b": n_b,
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "top_features": top_str, "coef": coef_map, "n_folds": n_splits,
    }


def run_boundary_set(frame: pd.DataFrame, tag: str) -> pd.DataFrame:
    rows = []
    for a, b in BOUNDARIES:
        for period in ("2024", "2026"):
            sub = frame[frame["period2"] == period]
            r = boundary_auc(sub, a, b, FEATURE_COLS, f"{tag}|{period}")
            r["period"] = period
            rows.append(r)
    return pd.DataFrame(rows)


print("[T20] Running primary boundary classifier (seniority_final)")
primary = run_boundary_set(df, "primary_sen_final")

# YOE-based proxy: define "entry-like" as yoe<=2, "mid-like" as 3<=yoe<=6, "senior-like" as yoe>=7.
# We use it to reconstruct the entry vs mid-senior and mid-senior vs director boundaries.
df_yoe = df.copy()
df_yoe["yoe_bucket"] = pd.cut(
    df_yoe["yoe_extracted"],
    bins=[-0.01, 2.0, 6.0, 20.0, 100.0],
    labels=["yoe_le2", "yoe_3_6", "yoe_7_20", "yoe_ge21"],
)
yoe_map = {"yoe_le2": "entry", "yoe_3_6": "mid-senior", "yoe_7_20": "director"}
df_yoe["sen_yoe"] = df_yoe["yoe_bucket"].map(yoe_map)
df_yoe_valid = df_yoe[df_yoe["sen_yoe"].notna()].copy()
df_yoe_valid["seniority_final"] = df_yoe_valid["sen_yoe"]
yoe_boundaries = run_boundary_set(df_yoe_valid, "sensitivity_yoe_proxy")

# Sensitivity (a): aggregator exclusion
df_noagg = df[~df["is_aggregator"].astype(bool)]
noagg = run_boundary_set(df_noagg, "sens_a_no_agg")

# Sensitivity (g): exclude title_lookup_llm
df_notll = df[df["swe_classification_tier"] != "title_lookup_llm"]
notll = run_boundary_set(df_notll, "sens_g_no_title_llm")

all_auc = pd.concat([primary, yoe_boundaries, noagg, notll], ignore_index=True)

auc_out = all_auc.drop(columns=["coef"]).copy()
auc_out.to_csv(T20_TABLES / "auc_by_period_boundary.csv", index=False)
print(auc_out.to_string())

# Feature importance change (primary, mid-senior <-> director and entry <-> mid-senior)
imp_rows = []
for r in primary.to_dict("records"):
    for k, v in r["coef"].items():
        imp_rows.append({
            "boundary": r["boundary"],
            "period": r["period"],
            "feature": k,
            "coef": v,
        })
imp_df = pd.DataFrame(imp_rows)
imp_df.to_csv(T20_TABLES / "feature_importance_by_period_boundary.csv", index=False)

# --- Feature heatmap per seniority x period --------------------------------

heat = (
    df.groupby(["period2", "seniority_final"])[FEATURE_COLS]
    .mean()
    .reset_index()
)
heat.to_csv(T20_TABLES / "feature_heatmap_seniority_period.csv", index=False)

# --- Domain-stratified boundary analysis -----------------------------------

TOP_ARCHETYPES = [
    "LLM / GenAI / ML engineering",
    "JS frontend (React/TypeScript)",
    "Java enterprise (Spring/microservices)",
    "Data engineering (Spark/ETL)",
    "DevOps / SRE / platform",
    "Embedded / firmware",
]

arch_rows = []
for arch_name in TOP_ARCHETYPES:
    sub = df[df["archetype_name"] == arch_name]
    if len(sub) < 200:
        continue
    for a, b in BOUNDARIES:
        for period in ("2024", "2026"):
            s2 = sub[sub["period2"] == period]
            r = boundary_auc(s2, a, b, FEATURE_COLS, f"arch|{arch_name}")
            r["archetype_name"] = arch_name
            r["period"] = period
            arch_rows.append(r)
arch_df = pd.DataFrame(arch_rows)
if not arch_df.empty:
    arch_df.drop(columns=["coef"]).to_csv(T20_TABLES / "auc_by_archetype.csv", index=False)

# --- Missing middle analysis -----------------------------------------------

# Distance of associate centroid to entry and mid-senior centroid per period.
mm_rows = []
scaler_global = StandardScaler().fit(df[FEATURE_COLS])
Xall = scaler_global.transform(df[FEATURE_COLS])
df["_row"] = np.arange(len(df))
for period in ("2024", "2026"):
    sub_idx = df["period2"] == period
    cents = {}
    for lvl in ("entry", "associate", "mid-senior", "director"):
        m = sub_idx & (df["seniority_final"] == lvl)
        if m.sum() == 0:
            continue
        cents[lvl] = Xall[df.loc[m, "_row"].to_numpy()].mean(axis=0)
    if "associate" in cents and "entry" in cents and "mid-senior" in cents:
        d_e = float(np.linalg.norm(cents["associate"] - cents["entry"]))
        d_m = float(np.linalg.norm(cents["associate"] - cents["mid-senior"]))
        mm_rows.append({
            "period": period,
            "dist_associate_to_entry": d_e,
            "dist_associate_to_midsenior": d_m,
            "entry_share_of_distance": d_e / (d_e + d_m),
        })
pd.DataFrame(mm_rows).to_csv(T20_TABLES / "missing_middle_distances.csv", index=False)

# --- Figures ---------------------------------------------------------------

# AUC comparison bar chart
fig, ax = plt.subplots(figsize=(9, 5))
prim = primary.copy()
prim["label"] = prim["boundary"] + " (n=" + prim["n_a"].astype(str) + "/" + prim["n_b"].astype(str) + ")"
piv = prim.pivot(index="boundary", columns="period", values="auc_mean")
piv_err = prim.pivot(index="boundary", columns="period", values="auc_std")
piv.plot(kind="bar", yerr=piv_err, ax=ax, capsize=3, color=["#4C72B0", "#DD8452"])
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
ax.set_ylabel("AUC (5-fold CV)")
ax.set_title("T20 boundary discriminability: AUC by period")
ax.set_ylim(0.4, 1.0)
ax.set_xlabel("Seniority boundary")
ax.legend(title="Period")
plt.xticks(rotation=20, ha="right")
fig.tight_layout()
fig.savefig(T20_FIGS / "auc_comparison.png", dpi=150)
plt.close(fig)

# Feature importance change: entry -> mid-senior boundary only
fi = imp_df[imp_df["boundary"] == "entry->mid-senior"].copy()
if not fi.empty:
    fi_p = fi.pivot(index="feature", columns="period", values="coef")
    fi_p["delta"] = fi_p.get("2026", 0) - fi_p.get("2024", 0)
    fi_p = fi_p.sort_values("delta")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#DD8452" if v > 0 else "#4C72B0" for v in fi_p["delta"]]
    ax.barh(fi_p.index, fi_p["delta"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient change (2026 − 2024) — positive = feature gained importance")
    ax.set_title("T20 feature importance shift, entry vs mid-senior boundary")
    fig.tight_layout()
    fig.savefig(T20_FIGS / "feature_importance_change.png", dpi=150)
    plt.close(fig)

# Feature heatmap
if not heat.empty:
    heat_norm = heat.copy()
    for c in FEATURE_COLS:
        v = heat_norm[c]
        if v.std() > 0:
            heat_norm[c] = (v - v.mean()) / v.std()
    heat_norm = heat_norm.sort_values(["seniority_final", "period2"])
    labels = heat_norm["seniority_final"] + " | " + heat_norm["period2"]
    mat = heat_norm[FEATURE_COLS].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(np.arange(len(FEATURE_COLS)))
    ax.set_xticklabels(FEATURE_COLS, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, label="z-score")
    ax.set_title("T20 feature heatmap (z-scored across all cells)")
    fig.tight_layout()
    fig.savefig(T20_FIGS / "feature_heatmap.png", dpi=150)
    plt.close(fig)

# AUC by archetype
if not arch_df.empty:
    arch_sub = arch_df[arch_df["boundary"] == "entry->mid-senior"].copy()
    arch_piv = arch_sub.pivot(index="archetype_name", columns="period", values="auc_mean")
    fig, ax = plt.subplots(figsize=(9, 5))
    arch_piv.plot(kind="barh", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUC (entry vs mid-senior)")
    ax.set_title("T20 entry↔mid-senior boundary AUC by archetype")
    fig.tight_layout()
    fig.savefig(T20_FIGS / "auc_by_archetype.png", dpi=150)
    plt.close(fig)

print("[T20] Done")
