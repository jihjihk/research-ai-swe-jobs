"""T11. Requirements complexity & credential stacking.

Loads shared cleaned text + shared tech matrix. Builds per-row requirement
features, complexity metrics, and cross-period comparisons with an explicit
focus on:

  - `management_strict` vs `management_broad` (validation-first, per Gate 1).
  - Entry-level complexity under `seniority_final` AND yoe≤2 definitions.
  - Credential stacking: more simultaneous requirement TYPES vs more within tech.
  - Top 1% by requirement_breadth outlier analysis.
  - Essential sensitivities: aggregator exclusion, 50/company cap,
    seniority operationalization, within-2024 calibration.

Boilerplate-sensitive density metrics are computed on
`text_source = 'llm'` rows only. Binary presence uses all rows.
Step 7 (domain-stratified) is SKIPPED because
`swe_archetype_labels.parquet` is not available at build time.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/home/jihgaboot/gabor/job-research")
SHARED = ROOT / "exploration" / "artifacts" / "shared"
OUT_TABLES = ROOT / "exploration" / "tables" / "T11"
OUT_FIGS = ROOT / "exploration" / "figures" / "T11"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

# ---------- Load shared artifacts ----------
cleaned = pq.read_table(
    SHARED / "swe_cleaned_text.parquet",
    columns=[
        "uid", "description_cleaned", "text_source",
        "source", "period", "seniority_final", "seniority_3level",
        "is_aggregator", "company_name_canonical", "yoe_extracted",
    ],
).to_pandas()

cleaned["period_group"] = np.where(cleaned["period"].str.startswith("2024"), "2024", "2026")
print(f"cleaned rows: {len(cleaned)}")
print(cleaned["period_group"].value_counts())

tech_mat = pq.read_table(SHARED / "swe_tech_matrix.parquet").to_pandas()
print(f"tech matrix: {tech_mat.shape}")
tech_cols = [c for c in tech_mat.columns if c != "uid"]
tech_mat["tech_count"] = tech_mat[tech_cols].sum(axis=1).astype(int)

df = cleaned.merge(tech_mat[["uid", "tech_count"]], on="uid", how="left")
df["tech_count"] = df["tech_count"].fillna(0).astype(int)

# LLM-only subset for density metrics
df_llm = df[df["text_source"] == "llm"].copy()
print(f"llm subset: {len(df_llm)}")
df_llm["desc_len_chars"] = df_llm["description_cleaned"].fillna("").str.len()

# ---------- Keyword dictionaries ----------
# NOTE: patterns use (?:^|\W) ... (?:\W|$) to avoid \b fragility around '+', '#', '.'.
# Matching is case-insensitive on the cleaned text (already lowercased).
SOFT_SKILLS = {
    "communication": r"\bcommunication\b",
    "collaboration": r"\b(collaboration|collaborative|collaborate|collaborating)\b",
    "problem_solving": r"\bproblem[- ]solving\b|\bproblem solver\b",
    "teamwork": r"\bteamwork\b",
    "leadership_word": r"\bleadership\b",  # conceptually a soft-skill mention, separate from mgmt verbs
    "interpersonal": r"\binterpersonal\b",
    "adaptability": r"\badaptab(le|ility)\b|\bflexib(le|ility)\b",
    "time_management": r"\btime management\b|\borganizational skills\b",
    "analytical": r"\banalytical\b",
    "detail_oriented": r"\bdetail[- ]oriented\b|\battention to detail\b",
}
ORG_SCOPE = {
    "ownership": r"\b(ownership|own(s|ed|ing)?)\b",
    "end_to_end": r"\bend[- ]to[- ]end\b",
    "cross_functional": r"\bcross[- ]functional\b",
    "stakeholder": r"\bstakeholder(s)?\b",
    "autonomous": r"\bautonom(y|ous|ously)\b",
    "initiative": r"\binitiative(s)?\b",
    "strategic": r"\bstrateg(y|ic|ically)\b",
    "vision": r"\bvision(ary)?\b",
    "roadmap": r"\broadmap\b",
    "impact": r"\b(high[- ]impact|drive impact|business impact)\b",
}
# Management — two tiers for validation
MGMT_STRICT = {
    "manage": r"\bmanag(e|es|ed|ing)\b(?!ment\b)",  # exclude "management" as noun-only
    "mentor": r"\bmentor(s|ed|ing|ship)?\b",
    "coach": r"\bcoach(es|ed|ing)?\b",
    "hire_verb": r"\bhir(e|es|ed|ing)\b",
    "direct_reports": r"\bdirect reports\b",
    "performance_review": r"\bperformance review(s)?\b",
    "headcount": r"\bheadcount\b",
    "people_management": r"\bpeople management\b",
    "one_on_ones": r"\b1:1(s)?\b|\bone[- ]on[- ]one(s)?\b",
}
MGMT_BROAD_EXTRA = {
    "lead_word": r"\blead(s|ing)?\b",
    "team_word": r"\bteam(s)?\b",
    "stakeholder_word": r"\bstakeholder(s)?\b",
    "coordinate": r"\bcoordinat(e|es|ed|ing|ion)\b",
    "leadership_word": r"\bleadership\b",
}
EDUCATION = {
    "phd": r"\b(ph\.?d\.?|doctorate)\b",
    "masters": r"\b(m\.?s\.?|master(s|'s)?|m\.?eng\.?)\b",
    "bachelors": r"\b(b\.?s\.?|b\.?a\.?|bachelor(s|'s)?|b\.?eng\.?|undergraduate degree)\b",
}
AI_TERMS = {
    "ai_generic": r"\b(ai|a\.i\.)\b(?! 're)",
    "ml": r"\b(ml|machine learning)\b",
    "llm": r"\bllm(s)?\b",
    "agent": r"\bagent(s|ic)?\b",
    "copilot": r"\bcopilot\b",
    "cursor": r"\bcursor\b",
    "claude": r"\bclaude\b",
    "gpt": r"\bgpt\b",
    "rag": r"\brag\b",
    "generative": r"\bgenerative\b",
    "prompt": r"\bprompt engineering\b",
}

ALL_CATEGORIES = {
    "soft_skill": SOFT_SKILLS,
    "org_scope": ORG_SCOPE,
    "mgmt_strict": MGMT_STRICT,
    "mgmt_broad_extra": MGMT_BROAD_EXTRA,
    "education": EDUCATION,
    "ai_terms": AI_TERMS,
}

compiled = {
    cat: {name: re.compile(pat, flags=re.IGNORECASE) for name, pat in d.items()}
    for cat, d in ALL_CATEGORIES.items()
}


def count_matches(text: str, regex_map: dict[str, re.Pattern]) -> dict[str, int]:
    return {name: 1 if rx.search(text) else 0 for name, rx in regex_map.items()}


# ---------- Per-row feature extraction (llm subset) ----------
print("Extracting features on llm subset...")
feats = {cat: {name: np.zeros(len(df_llm), dtype=np.int8) for name in regs} for cat, regs in compiled.items()}

for i, text in enumerate(df_llm["description_cleaned"].fillna("").values):
    for cat, regs in compiled.items():
        for name, rx in regs.items():
            if rx.search(text):
                feats[cat][name][i] = 1
    if i % 5000 == 0 and i > 0:
        print(f"  {i}/{len(df_llm)}")

# Collapse to counts
def distinct_count(cat_feats: dict[str, np.ndarray]) -> np.ndarray:
    return np.sum(np.stack(list(cat_feats.values())), axis=0)

df_llm["soft_skill_count"] = distinct_count(feats["soft_skill"])
df_llm["org_scope_count"] = distinct_count(feats["org_scope"])
df_llm["mgmt_strict_count"] = distinct_count(feats["mgmt_strict"])
df_llm["mgmt_broad_count"] = distinct_count(feats["mgmt_strict"]) + distinct_count(feats["mgmt_broad_extra"])
df_llm["ai_term_count"] = distinct_count(feats["ai_terms"])

# Education: take highest tier
def highest_edu(i: int) -> int:
    if feats["education"]["phd"][i]:
        return 3
    if feats["education"]["masters"][i]:
        return 2
    if feats["education"]["bachelors"][i]:
        return 1
    return 0

df_llm["education_tier"] = np.array([highest_edu(i) for i in range(len(df_llm))], dtype=np.int8)

df_llm["has_soft_skill"] = (df_llm["soft_skill_count"] > 0).astype(int)
df_llm["has_org_scope"] = (df_llm["org_scope_count"] > 0).astype(int)
df_llm["has_mgmt_strict"] = (df_llm["mgmt_strict_count"] > 0).astype(int)
df_llm["has_mgmt_broad"] = (df_llm["mgmt_broad_count"] > 0).astype(int)
df_llm["has_ai_term"] = (df_llm["ai_term_count"] > 0).astype(int)
df_llm["has_education"] = (df_llm["education_tier"] > 0).astype(int)
df_llm["has_yoe"] = df_llm["yoe_extracted"].notna().astype(int)
df_llm["has_tech"] = (df_llm["tech_count"] > 0).astype(int)

# Complexity metrics
df_llm["requirement_breadth"] = (
    df_llm["tech_count"]
    + df_llm["soft_skill_count"]
    + df_llm["org_scope_count"]
    + df_llm["mgmt_strict_count"]  # use strict by default for breadth
    + df_llm["education_tier"].clip(upper=1)  # at most 1 per posting
    + df_llm["ai_term_count"]
    + df_llm["has_yoe"]  # YOE is binary contributor
)
df_llm["credential_stack_depth_strict"] = (
    df_llm["has_tech"]
    + df_llm["has_soft_skill"]
    + df_llm["has_org_scope"]
    + df_llm["has_mgmt_strict"]
    + df_llm["has_education"]
    + df_llm["has_yoe"]
    + df_llm["has_ai_term"]
)
df_llm["credential_stack_depth_broad"] = (
    df_llm["has_tech"]
    + df_llm["has_soft_skill"]
    + df_llm["has_org_scope"]
    + df_llm["has_mgmt_broad"]
    + df_llm["has_education"]
    + df_llm["has_yoe"]
    + df_llm["has_ai_term"]
)

df_llm["tech_density"] = df_llm["tech_count"] / (df_llm["desc_len_chars"] / 1000).clip(lower=0.1)
df_llm["scope_density"] = df_llm["org_scope_count"] / (df_llm["desc_len_chars"] / 1000).clip(lower=0.1)

FEATURE_COLS = [
    "tech_count", "soft_skill_count", "org_scope_count",
    "mgmt_strict_count", "mgmt_broad_count", "ai_term_count",
    "education_tier", "has_yoe",
    "requirement_breadth", "credential_stack_depth_strict", "credential_stack_depth_broad",
    "tech_density", "scope_density",
]

df_llm.to_parquet(OUT_TABLES / "_llm_features.parquet", index=False)

# ---------- 3. Distributions by period × seniority ----------
def summarize(grp: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n": len(grp),
        **{f"{c}_mean": grp[c].mean() for c in FEATURE_COLS},
        **{f"{c}_median": grp[c].median() for c in FEATURE_COLS},
    })

by_period = df_llm.groupby("period_group").apply(summarize)
by_period.to_csv(OUT_TABLES / "complexity_by_period.csv")
print("\nComplexity metrics by period:")
print(by_period.filter(regex="mean|^n$").to_string())

by_ps = df_llm.groupby(["period_group", "seniority_final"]).apply(summarize)
by_ps.to_csv(OUT_TABLES / "complexity_by_period_seniority.csv")

# Within-2024 calibration (arshkon vs asaniczka)
df_2024 = df_llm[df_llm["period_group"] == "2024"]
by_src_2024 = df_2024.groupby("source").apply(summarize)
by_src_2024.to_csv(OUT_TABLES / "complexity_within_2024_calibration.csv")

# Cohen's d computation for each metric
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((b.mean() - a.mean()) / pooled)


def snr_row(metric: str) -> dict:
    a = df_llm[df_llm["source"] == "kaggle_arshkon"][metric].values
    b = df_llm[df_llm["source"] == "kaggle_asaniczka"][metric].values
    c = df_llm[df_llm["period_group"] == "2026"][metric].values
    d_within = cohens_d(a, b)
    d_cross = cohens_d(df_llm[df_llm["period_group"] == "2024"][metric].values, c)
    snr = abs(d_cross) / abs(d_within) if abs(d_within) > 1e-6 else np.inf
    return {
        "metric": metric,
        "mean_2024": float(df_llm[df_llm["period_group"] == "2024"][metric].mean()),
        "mean_2026": float(c.mean()),
        "within_2024_d": d_within,
        "cross_period_d": d_cross,
        "snr": snr,
    }

snr_rows = [snr_row(m) for m in FEATURE_COLS]
snr_df = pd.DataFrame(snr_rows)
snr_df.to_csv(OUT_TABLES / "complexity_snr_calibration.csv", index=False)
print("\nComplexity SNR vs within-2024:")
print(snr_df.to_string(index=False))

# ---------- 4. Credential stacking question ----------
cs_pct = (
    df_llm.groupby("period_group")["credential_stack_depth_strict"]
    .apply(lambda s: s.value_counts(normalize=True).sort_index())
    .unstack(fill_value=0)
)
cs_pct.to_csv(OUT_TABLES / "credential_stack_distribution.csv")
print("\nCredential stack depth (strict) distribution:")
print(cs_pct.to_string())

# ---------- 5. Entry-level complexity ----------
# Under seniority_final
df_llm["is_entry_sen"] = df_llm["seniority_final"] == "entry"
df_llm["is_entry_yoe"] = df_llm["yoe_extracted"] <= 2
entry_sen = df_llm.groupby(["period_group", "is_entry_sen"]).apply(summarize)
entry_yoe = df_llm.groupby(["period_group", "is_entry_yoe"]).apply(summarize)
entry_sen.to_csv(OUT_TABLES / "entry_complexity_by_seniority_final.csv")
entry_yoe.to_csv(OUT_TABLES / "entry_complexity_by_yoe_le2.csv")

# Headline entry comparison table
def entry_headline(df_: pd.DataFrame, mask_col: str) -> pd.DataFrame:
    sub = df_[df_[mask_col] == True]
    out = sub.groupby("period_group")[FEATURE_COLS].agg(["mean", "median", "count"])
    return out

print("\nEntry (seniority_final) headline means:")
print(df_llm[df_llm["is_entry_sen"]].groupby("period_group")[FEATURE_COLS].mean().to_string())
print("\nEntry (yoe<=2) headline means:")
print(df_llm[df_llm["is_entry_yoe"] == True].groupby("period_group")[FEATURE_COLS].mean().to_string())

# ---------- 6. Management indicator deep dive ----------
# Top triggering terms for each pattern, stratified by period
def trigger_counts(text_series: pd.Series, patterns: dict[str, re.Pattern]) -> dict[str, int]:
    out = {}
    for name, rx in patterns.items():
        out[name] = int(text_series.str.contains(rx.pattern, case=False, regex=True).sum())
    return out

mgmt_rows = []
for period in ["2024", "2026"]:
    sub = df_llm[df_llm["period_group"] == period]
    n = len(sub)
    for name, rx in compiled["mgmt_strict"].items():
        hits = int(sub["description_cleaned"].fillna("").str.contains(rx.pattern, case=False, regex=True).sum())
        mgmt_rows.append({"tier": "strict", "period": period, "term": name, "n_hits": hits, "share": hits / n})
    for name, rx in compiled["mgmt_broad_extra"].items():
        hits = int(sub["description_cleaned"].fillna("").str.contains(rx.pattern, case=False, regex=True).sum())
        mgmt_rows.append({"tier": "broad_extra", "period": period, "term": name, "n_hits": hits, "share": hits / n})
mgmt_trig = pd.DataFrame(mgmt_rows)
mgmt_trig.to_csv(OUT_TABLES / "management_trigger_terms.csv", index=False)
print("\nManagement trigger term shares (strict + broad_extra):")
print(mgmt_trig.pivot(index="term", columns="period", values="share").round(4).to_string())

# Strict vs broad comparison
mgmt_rates = df_llm.groupby("period_group")[["has_mgmt_strict", "has_mgmt_broad"]].mean()
mgmt_rates["inflation_ratio"] = mgmt_rates["has_mgmt_broad"] / mgmt_rates["has_mgmt_strict"]
mgmt_rates.to_csv(OUT_TABLES / "management_strict_vs_broad.csv")
print("\nManagement strict vs broad rates:")
print(mgmt_rates)

# Precision sample: 50 matches stratified by period for each tier
def sample_matches(df_: pd.DataFrame, mask_col: str, n_per_period: int = 25) -> pd.DataFrame:
    samples = []
    for period in ["2024", "2026"]:
        sub = df_[(df_[mask_col] == 1) & (df_["period_group"] == period)]
        if len(sub) == 0:
            continue
        take = min(n_per_period, len(sub))
        idx = RNG.choice(len(sub), take, replace=False)
        samples.append(sub.iloc[idx][["uid", "period_group", "seniority_final", "description_cleaned"]])
    return pd.concat(samples, ignore_index=True)

strict_sample = sample_matches(df_llm, "has_mgmt_strict", 25)
broad_sample = sample_matches(df_llm, "has_mgmt_broad", 25)
strict_sample.to_csv(OUT_TABLES / "mgmt_strict_sample_50.csv", index=False)
broad_sample.to_csv(OUT_TABLES / "mgmt_broad_sample_50.csv", index=False)


# Automated precision heuristic: mark true-positive if any strict pattern actually matches the description.
# For strict sample, the strict condition is already satisfied, so it is by definition a match to a strict pattern;
# we use this as the lower-bound precision scaffold and then classify the broad sample by checking whether
# any strict pattern (manage/mentor/etc) also fires — if NOT, the broad hit was driven purely by lead/team/etc.
def auto_precision(df_: pd.DataFrame) -> tuple[float, int]:
    strict_rx_list = list(compiled["mgmt_strict"].values())
    hits = 0
    for text in df_["description_cleaned"].fillna(""):
        if any(rx.search(text) for rx in strict_rx_list):
            hits += 1
    return hits / max(len(df_), 1), hits

prec_strict, hits_strict = auto_precision(strict_sample)
prec_broad_vs_strict, hits_broad_vs_strict = auto_precision(broad_sample)
print(f"\nAuto heuristic strict sample: {prec_strict:.2f} ({hits_strict}/{len(strict_sample)}) contain a strict pattern (expected ~1.0)")
print(f"Auto heuristic broad sample: {prec_broad_vs_strict:.2f} ({hits_broad_vs_strict}/{len(broad_sample)}) also contain a strict pattern")
print("  -> Remaining fraction is broad-only (lead/team/stakeholder/coordinate-driven)")

# Mechanism: what pattern triggered the broad-only hits? Report which broad_extra term fired
broad_only = broad_sample.iloc[:0].copy()
broad_only_rows = []
strict_rx_list = list(compiled["mgmt_strict"].values())
for _, row in broad_sample.iterrows():
    text = row["description_cleaned"] or ""
    if any(rx.search(text) for rx in strict_rx_list):
        continue
    triggers = [name for name, rx in compiled["mgmt_broad_extra"].items() if rx.search(text)]
    broad_only_rows.append({
        "uid": row["uid"],
        "period": row["period_group"],
        "seniority": row["seniority_final"],
        "triggers": ",".join(triggers),
        "snippet": text[:400].replace("\n", " "),
    })
broad_only_df = pd.DataFrame(broad_only_rows)
broad_only_df.to_csv(OUT_TABLES / "mgmt_broad_only_triggers.csv", index=False)
print(f"\n{len(broad_only_df)} broad-only hits (no strict pattern). Trigger term distribution:")
if len(broad_only_df):
    print(broad_only_df["triggers"].value_counts().head(10).to_string())

# ---------- 7. Domain-stratified — SKIPPED ----------
archetype_path = SHARED / "swe_archetype_labels.parquet"
if archetype_path.exists():
    print("\n[T11 step 7] archetype labels present — running domain-stratified scope inflation.")
    arch = pq.read_table(archetype_path).to_pandas()
    merged = df_llm.merge(arch, on="uid", how="left")
    dom = merged.groupby(["period_group", "archetype_label"]).apply(summarize)
    dom.to_csv(OUT_TABLES / "entry_complexity_by_archetype.csv")
else:
    print("\n[T11 step 7] swe_archetype_labels.parquet NOT present — skipping domain-stratified scope inflation. Must be picked up by T28.")
    (OUT_TABLES / "step7_skipped.txt").write_text(
        "swe_archetype_labels.parquet was not available at T11 build time. "
        "Domain-stratified scope inflation (step 7) skipped; defer to T28."
    )

# ---------- 8. Outlier analysis: top 1% by requirement_breadth ----------
q99 = df_llm["requirement_breadth"].quantile(0.99)
outliers = df_llm[df_llm["requirement_breadth"] >= q99].copy()
print(f"\nTop 1% cutoff (requirement_breadth >= {q99}): n={len(outliers)}")
outlier_summary = outliers.groupby(["period_group", "source"]).agg(
    n=("uid", "count"),
    mean_breadth=("requirement_breadth", "mean"),
    mean_len=("desc_len_chars", "mean"),
    mean_tech=("tech_count", "mean"),
    top_company=("company_name_canonical", lambda s: s.value_counts().index[0] if len(s) else None),
)
outlier_summary.to_csv(OUT_TABLES / "outlier_top1pct_summary.csv")
outliers[["uid", "period_group", "source", "seniority_final", "company_name_canonical", "requirement_breadth", "desc_len_chars", "tech_count", "description_cleaned"]].head(50).to_csv(
    OUT_TABLES / "outlier_top1pct_sample.csv", index=False
)

# Top companies among outliers
outlier_top_co = (
    outliers["company_name_canonical"]
    .value_counts()
    .head(20)
    .reset_index()
    .rename(columns={"count": "n_outlier_postings"})
)
outlier_top_co.to_csv(OUT_TABLES / "outlier_top1pct_top_companies.csv", index=False)
print("\nTop companies in outlier set:")
print(outlier_top_co.head(10).to_string(index=False))

# ---------- Sensitivity runs ----------
# (a) Aggregator exclusion
df_llm_noagg = df_llm[~df_llm["is_aggregator"].fillna(False)].copy()
sens_a = df_llm_noagg.groupby("period_group")[FEATURE_COLS].mean()
sens_a.to_csv(OUT_TABLES / "sens_a_aggregator_excluded.csv")

# (b) Company cap 50
def cap_50(df_: pd.DataFrame) -> pd.DataFrame:
    return (
        df_.groupby("company_name_canonical", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), 50), random_state=42))
    )

df_llm_cap = cap_50(df_llm)
sens_b = df_llm_cap.groupby("period_group")[FEATURE_COLS].mean()
sens_b.to_csv(OUT_TABLES / "sens_b_company_cap50.csv")
print(f"\nCap 50/company: {len(df_llm_cap)} rows (from {len(df_llm)})")

# (c) seniority operationalization comparison — already computed via is_entry_sen vs is_entry_yoe

# Summary sensitivity table: mean requirement_breadth under each spec
sens_summary = pd.DataFrame({
    "primary": df_llm.groupby("period_group")["requirement_breadth"].mean(),
    "no_aggregator": df_llm_noagg.groupby("period_group")["requirement_breadth"].mean(),
    "cap_50_per_company": df_llm_cap.groupby("period_group")["requirement_breadth"].mean(),
})
sens_summary.to_csv(OUT_TABLES / "sensitivity_summary.csv")
print("\nSensitivity summary — mean requirement_breadth:")
print(sens_summary.to_string())

# ---------- Figures ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figure 1: distribution of requirement_breadth by period
fig, ax = plt.subplots(figsize=(7, 4))
for period, color in [("2024", "#4C72B0"), ("2026", "#DD8452")]:
    sub = df_llm[df_llm["period_group"] == period]["requirement_breadth"]
    ax.hist(sub.clip(upper=40), bins=40, alpha=0.55, label=f"{period} (n={len(sub)})", color=color)
ax.set_xlabel("requirement_breadth (total distinct requirement types)")
ax.set_ylabel("n postings")
ax.set_title("Requirement breadth distribution, 2024 vs 2026 (llm text only)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_FIGS / "requirement_breadth_hist.png", dpi=150)
plt.close()

# Figure 2: credential stack depth
fig, ax = plt.subplots(figsize=(7, 4))
cs_pct.T.plot(kind="bar", ax=ax)
ax.set_xlabel("credential_stack_depth_strict (# categories present)")
ax.set_ylabel("share of postings")
ax.set_title("Credential stack depth, strict (max 7)")
plt.tight_layout()
plt.savefig(OUT_FIGS / "credential_stack_depth.png", dpi=150)
plt.close()

# Figure 3: mgmt strict vs broad inflation
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(2)
w = 0.35
ax.bar(x - w / 2, mgmt_rates["has_mgmt_strict"], w, label="strict")
ax.bar(x + w / 2, mgmt_rates["has_mgmt_broad"], w, label="broad")
ax.set_xticks(x)
ax.set_xticklabels(mgmt_rates.index)
ax.set_ylabel("share of postings")
ax.set_title("Management indicator: strict vs broad")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_FIGS / "mgmt_strict_vs_broad.png", dpi=150)
plt.close()

# Figure 4: entry complexity bar chart
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
entry_sen_means = df_llm[df_llm["is_entry_sen"]].groupby("period_group")[
    ["tech_count", "requirement_breadth", "credential_stack_depth_strict"]
].mean()
entry_yoe_means = df_llm[df_llm["is_entry_yoe"] == True].groupby("period_group")[
    ["tech_count", "requirement_breadth", "credential_stack_depth_strict"]
].mean()
entry_sen_means.T.plot(kind="bar", ax=axes[0])
axes[0].set_title("Entry (seniority_final)")
axes[0].set_ylabel("mean")
axes[0].tick_params(axis="x", rotation=20)
entry_yoe_means.T.plot(kind="bar", ax=axes[1])
axes[1].set_title("Entry (yoe<=2)")
axes[1].tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig(OUT_FIGS / "entry_complexity_comparison.png", dpi=150)
plt.close()

print("\nT11 done.")
