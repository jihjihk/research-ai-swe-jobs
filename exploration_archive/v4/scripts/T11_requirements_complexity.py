"""T11 — Requirements complexity and credential stacking (SWE, LinkedIn-only).

Builds a structured feature extractor for each SWE posting and compares complexity
distributions across periods, seniority operationalizations, and (optionally) domain archetypes.

Key design
----------
* Text source: cleaned text from the shared artifact (`swe_cleaned_text.parquet`,
  `description_cleaned`), which prefers `description_core_llm` where available.
  For keyword presence we also fall back to raw `description` to improve recall.
* Tech counts: merged binary matrix from `exploration/artifacts/shared/swe_tech_matrix.parquet`.
* Density metrics: per 1,000 characters of cleaned description.
* Seniority operationalizations (reported in parallel):
    1. combined best-available (labeled -> seniority_llm; rule_sufficient -> seniority_final)
    2. augmented: best-available where defined, else seniority_final fallback
    3. YOE-based proxy (yoe<=2 vs yoe>2 vs unknown)
* Entry-level scope inflation comparison uses both operationalizations.
* Management indicator is split into two tiers with precision spot-checks.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNIFIED = ROOT / "data" / "unified.parquet"
SHARED = ROOT / "exploration" / "artifacts" / "shared"
TECH_MATRIX = SHARED / "swe_tech_matrix.parquet"
CLEANED_TEXT = SHARED / "swe_cleaned_text.parquet"
FIG_DIR = ROOT / "exploration" / "figures" / "T11"
TAB_DIR = ROOT / "exploration" / "tables" / "T11"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

BASE_FILTER = """
  source_platform = 'linkedin'
  AND is_english = true
  AND date_flag = 'ok'
  AND is_swe = true
"""

SENIORITY_CASE = """
  CASE
    WHEN llm_classification_coverage = 'labeled'         THEN seniority_llm
    WHEN llm_classification_coverage = 'rule_sufficient' THEN seniority_final
    ELSE NULL
  END AS seniority_best_available
"""

SENIORITY_CASE_AUG = """
  CASE
    WHEN llm_classification_coverage = 'labeled'         THEN seniority_llm
    WHEN llm_classification_coverage = 'rule_sufficient' THEN seniority_final
    WHEN seniority_final IS NOT NULL                     THEN seniority_final
    ELSE NULL
  END AS seniority_best_available_aug
"""

# --------------------------------------------------------------------------
# Requirement keyword categories
# --------------------------------------------------------------------------

SOFT_SKILLS = {
    "communication": re.compile(r"\b(communicat\w*|verbal|written)\b", re.I),
    "collaboration": re.compile(r"\b(collaborat\w*|team[- ]?work\w*|partner\w*|work\s+closely)\b", re.I),
    "problem_solving": re.compile(r"\b(problem[- ]?solv\w*|analytical|critical[- ]?think\w*|troubleshoot\w*)\b", re.I),
    "leadership": re.compile(r"\b(leadership|lead\w*)\b", re.I),
    "adaptability": re.compile(r"\b(adapt\w*|flexib\w*|learn quickly|fast[- ]?paced)\b", re.I),
    "organization": re.compile(r"\b(organization|time[- ]?management|prioriti[sz]\w*|detail[- ]?oriented|attention to detail)\b", re.I),
    "customer_focus": re.compile(r"\b(customer[- ]?focused|user[- ]?focused|user[- ]?centric|empath\w*)\b", re.I),
}

ORG_SCOPE = {
    "ownership": re.compile(r"\b(ownership|own the|owning|take ownership)\b", re.I),
    "end_to_end": re.compile(r"\b(end[- ]to[- ]end|e2e|full[- ]?cycle|full[- ]?lifecycle)\b", re.I),
    "cross_functional": re.compile(r"\bcross[- ]functional\w*\b", re.I),
    "stakeholder": re.compile(r"\bstakeholder\w*\b", re.I),
    "autonomous": re.compile(r"\b(autonom\w*|self[- ]?direct\w*|self[- ]?start\w*|independent\w*)\b", re.I),
    "initiative": re.compile(r"\b(initiative|drive results|proactive)\b", re.I),
    "impact": re.compile(r"\b(impact|business outcome|drive business)\b", re.I),
    "strategy": re.compile(r"\b(strategy|strategic|roadmap|vision)\b", re.I),
}

EDUCATION_PATTERNS = [
    ("phd", re.compile(r"\b(ph\.?d|doctorate|doctoral)\b", re.I), 4),
    ("ms", re.compile(r"\b(m\.?s\.?|m\.?sc|master'?s?|mba|m\.?eng|graduate degree)\b", re.I), 3),
    ("bs", re.compile(r"\b(b\.?s\.?|b\.?sc|bachelor'?s?|b\.?a\.?|b\.?e\.?|undergraduate degree)\b", re.I), 2),
    ("associate", re.compile(r"\bassociate[- ]?degree\b", re.I), 1),
]

# Management tiers
MANAGEMENT_STRONG = {
    "manage": re.compile(r"\bmanage(?!ment)\w*\b", re.I),
    "mentor": re.compile(r"\bmentor\w*\b", re.I),
    "coach": re.compile(r"\bcoach\w*\b", re.I),
    "hire": re.compile(r"\b(hire|hiring|recruit\w*)\b", re.I),
    "direct_reports": re.compile(r"\bdirect report\w*\b", re.I),
    "performance_review": re.compile(r"\bperformance review\w*\b", re.I),
    "headcount": re.compile(r"\bheadcount\b", re.I),
    "people_manager": re.compile(r"\bpeople manager\b|\bpeople management\b", re.I),
}
MANAGEMENT_BROAD = {
    **MANAGEMENT_STRONG,
    "lead": re.compile(r"\blead(?:ing)?\b", re.I),
    "team": re.compile(r"\bteam\b", re.I),
    "stakeholder": re.compile(r"\bstakeholder\w*\b", re.I),
    "coordinate": re.compile(r"\bcoordinat\w*\b", re.I),
}

# Broad AI mentions
AI_MENTIONS = re.compile(
    r"\b(ai|artificial intelligence|llm|llms|generative ai|genai|gen-ai|gpt|chatgpt|claude|copilot|"
    r"prompt engineering|rag|vector database|langchain|huggingface|hugging face|openai|anthropic|"
    r"machine learning|deep learning|transformer|neural net|mlops)\b",
    re.I,
)


def load_frame() -> pd.DataFrame:
    con = duckdb.connect()
    q = f"""
    SELECT
        uid,
        source,
        period,
        title,
        title_normalized,
        description,
        is_aggregator,
        company_name_canonical,
        {SENIORITY_CASE},
        {SENIORITY_CASE_AUG},
        seniority_final,
        seniority_native,
        yoe_extracted
    FROM '{UNIFIED}'
    WHERE {BASE_FILTER}
    """
    df = con.execute(q).fetchdf()
    df["year"] = df["period"].str.slice(0, 4)
    return df


def load_cleaned_text() -> pd.DataFrame:
    con = duckdb.connect()
    return con.execute(
        f"SELECT uid, description_cleaned, text_source FROM '{CLEANED_TEXT}'"
    ).fetchdf()


def load_tech_matrix() -> pd.DataFrame:
    con = duckdb.connect()
    return con.execute(f"SELECT * FROM '{TECH_MATRIX}'").fetchdf()


def count_keyword_hits(text: str, patterns: dict) -> dict:
    if not isinstance(text, str) or not text:
        return {k: 0 for k in patterns}
    return {k: 1 if pat.search(text) else 0 for k, pat in patterns.items()}


def extract_features(df: pd.DataFrame, cleaned: pd.DataFrame, tech: pd.DataFrame) -> pd.DataFrame:
    """Build requirement-level features per posting."""
    # Tech counts
    tech_cols = [c for c in tech.columns if c != "uid"]
    tech = tech.copy()
    tech["tech_count"] = tech[tech_cols].sum(axis=1)
    tech_small = tech[["uid", "tech_count"]]

    # Merge text
    base = df.merge(cleaned, on="uid", how="left")
    # For keyword presence we use cleaned where available, else raw description (recall boost)
    base["text_for_keywords"] = base["description_cleaned"].where(
        base["description_cleaned"].notna() & (base["description_cleaned"].str.len() > 50),
        base["description"],
    )
    base["text_len"] = base["text_for_keywords"].fillna("").str.len()

    # Keyword hits
    print("  Counting soft-skill hits…")
    soft_hits = base["text_for_keywords"].apply(lambda t: count_keyword_hits(t, SOFT_SKILLS))
    soft_df = pd.DataFrame(list(soft_hits))
    soft_df.columns = [f"softk_{c}" for c in soft_df.columns]
    soft_df["soft_count"] = soft_df.sum(axis=1)

    print("  Counting org-scope hits…")
    org_hits = base["text_for_keywords"].apply(lambda t: count_keyword_hits(t, ORG_SCOPE))
    org_df = pd.DataFrame(list(org_hits))
    org_df.columns = [f"scopek_{c}" for c in org_df.columns]
    org_df["scope_count"] = org_df.sum(axis=1)

    print("  Counting strong management hits…")
    mgs = base["text_for_keywords"].apply(lambda t: count_keyword_hits(t, MANAGEMENT_STRONG))
    mgs_df = pd.DataFrame(list(mgs))
    mgs_df.columns = [f"mgmts_{c}" for c in mgs_df.columns]
    mgs_df["mgmt_strong_any"] = (mgs_df.sum(axis=1) > 0).astype(int)

    print("  Counting broad management hits…")
    mgb = base["text_for_keywords"].apply(lambda t: count_keyword_hits(t, MANAGEMENT_BROAD))
    mgb_df = pd.DataFrame(list(mgb))
    mgb_df.columns = [f"mgmtb_{c}" for c in mgb_df.columns]
    mgb_df["mgmt_broad_any"] = (mgb_df.sum(axis=1) > 0).astype(int)

    print("  Determining education level…")

    def edu_level(text: str) -> int:
        if not isinstance(text, str) or not text:
            return 0
        for name, pat, level in EDUCATION_PATTERNS:
            if pat.search(text):
                return level
        return 0

    base["edu_level"] = base["text_for_keywords"].apply(edu_level)

    print("  Detecting AI mentions…")
    base["ai_mention"] = base["text_for_keywords"].fillna("").str.contains(AI_MENTIONS, na=False).astype(int)

    # Merge tech
    base = base.merge(tech_small, on="uid", how="left")
    base["tech_count"] = base["tech_count"].fillna(0).astype(int)

    # Concatenate features
    out = pd.concat(
        [base.reset_index(drop=True), soft_df.reset_index(drop=True), org_df.reset_index(drop=True), mgs_df.reset_index(drop=True), mgb_df.reset_index(drop=True)],
        axis=1,
    )

    # Complexity metrics
    out["requirement_breadth"] = out["tech_count"] + out["soft_count"] + out["scope_count"] + (out["edu_level"] > 0).astype(int) + (out["yoe_extracted"].notna()).astype(int) + out["mgmt_broad_any"] + out["ai_mention"]
    # Category count = number of categories with at least one mention
    out["credential_stack_depth"] = (
        (out["tech_count"] > 0).astype(int)
        + (out["soft_count"] > 0).astype(int)
        + (out["scope_count"] > 0).astype(int)
        + (out["edu_level"] > 0).astype(int)
        + out["yoe_extracted"].notna().astype(int)
        + out["mgmt_broad_any"]
        + out["ai_mention"]
    )
    # Densities per 1K chars (guard against zero-length)
    denom = out["text_len"].clip(lower=1)
    out["tech_density"] = 1000 * out["tech_count"] / denom
    out["scope_density"] = 1000 * out["scope_count"] / denom

    return out


# --------------------------------------------------------------------------
# Analysis helpers
# --------------------------------------------------------------------------

METRICS = ["tech_count", "soft_count", "scope_count", "edu_level", "mgmt_strong_any", "mgmt_broad_any", "ai_mention", "requirement_breadth", "credential_stack_depth", "tech_density", "scope_density"]


def summarize_metrics(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    agg = {m: ["mean", "median", "std"] for m in METRICS}
    out = df.groupby(by).agg(agg)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out = out.reset_index()
    out["n"] = df.groupby(by).size().values
    return out


def yoe_bucket(y) -> str:
    if pd.isna(y):
        return "unknown"
    if y <= 2:
        return "yoe<=2"
    return "yoe>2"


def compare_seniority_ablation(df: pd.DataFrame, tab_dir: Path) -> dict:
    results = {}
    df = df.copy()
    df["yoe_bucket"] = df["yoe_extracted"].apply(yoe_bucket)

    for col, tag in [
        ("seniority_best_available", "bestavail"),
        ("seniority_best_available_aug", "bestavail_aug"),
        ("seniority_final", "final"),
        ("yoe_bucket", "yoe"),
    ]:
        sub = df.copy()
        sub["seniority_op"] = sub[col].fillna("missing")
        tab = summarize_metrics(sub, ["year", "seniority_op"])
        tab.to_csv(tab_dir / f"complexity_by_{tag}.csv", index=False)
        results[tag] = tab
    return results


def entry_level_comparison(df: pd.DataFrame, tab_dir: Path) -> dict:
    """Focus on entry-level scope inflation under both operationalizations."""
    df = df.copy()
    df["yoe_bucket"] = df["yoe_extracted"].apply(yoe_bucket)

    rows = []
    for variant, filt in [
        ("combined", df["year"].isin(["2024", "2026"])),
        ("arshkon_only_2024", ((df["source"] == "kaggle_arshkon") & (df["year"] == "2024")) | (df["year"] == "2026")),
    ]:
        sub = df[filt]
        # Best-available (non-aug) entry
        for op_name, col, val in [
            ("seniority_best_available", "seniority_best_available", "entry"),
            ("seniority_best_available_aug", "seniority_best_available_aug", "entry"),
            ("seniority_final", "seniority_final", "entry"),
            ("yoe<=2", "yoe_bucket", "yoe<=2"),
        ]:
            for year in ["2024", "2026"]:
                mask = (sub["year"] == year) & (sub[col] == val)
                s = sub[mask]
                rows.append({
                    "variant": variant,
                    "operationalization": op_name,
                    "year": year,
                    "n": int(len(s)),
                    **{m: round(float(s[m].mean()), 4) if len(s) else np.nan for m in METRICS},
                })
    tab = pd.DataFrame(rows)
    tab.to_csv(tab_dir / "entry_level_scope_inflation.csv", index=False)

    # Compact delta table
    delta_rows = []
    for variant in tab["variant"].unique():
        for op in tab["operationalization"].unique():
            a = tab[(tab["variant"] == variant) & (tab["operationalization"] == op) & (tab["year"] == "2024")]
            b = tab[(tab["variant"] == variant) & (tab["operationalization"] == op) & (tab["year"] == "2026")]
            if len(a) == 0 or len(b) == 0:
                continue
            delta = {m: round(float(b[m].iloc[0] - a[m].iloc[0]), 4) for m in METRICS}
            delta_rows.append({
                "variant": variant,
                "operationalization": op,
                "n_2024": int(a["n"].iloc[0]),
                "n_2026": int(b["n"].iloc[0]),
                **delta,
            })
    delta_tab = pd.DataFrame(delta_rows)
    delta_tab.to_csv(tab_dir / "entry_level_deltas.csv", index=False)

    return {"wide": tab, "delta": delta_tab}


def management_term_breakdown(df: pd.DataFrame, tab_dir: Path) -> dict:
    """Top terms triggering management indicators per year, plus precision spot-checks."""
    strong_cols = [c for c in df.columns if c.startswith("mgmts_") and c != "mgmts_mgmt_strong_any"]
    broad_cols = [c for c in df.columns if c.startswith("mgmtb_") and c != "mgmtb_mgmt_broad_any"]

    rows = []
    for year in ["2024", "2026"]:
        sub = df[df["year"] == year]
        for col in strong_cols:
            rows.append({"year": year, "tier": "strong", "term": col.replace("mgmts_", ""), "n_postings": int(sub[col].sum()), "share": round(float(sub[col].mean()), 4)})
        for col in broad_cols:
            rows.append({"year": year, "tier": "broad", "term": col.replace("mgmtb_", ""), "n_postings": int(sub[col].sum()), "share": round(float(sub[col].mean()), 4)})
    tab = pd.DataFrame(rows).sort_values(["year", "tier", "n_postings"], ascending=[True, True, False])
    tab.to_csv(tab_dir / "management_term_breakdown.csv", index=False)

    # Precision spot-checks: sample 50 strong and 50 broad
    random.seed(42)
    samples_strong = df[df["mgmt_strong_any"] == 1][["uid", "year", "title", "text_for_keywords"]].sample(min(50, int((df["mgmt_strong_any"] == 1).sum())), random_state=42)
    samples_broad = df[(df["mgmt_broad_any"] == 1) & (df["mgmt_strong_any"] == 0)][["uid", "year", "title", "text_for_keywords"]].sample(min(50, int(((df["mgmt_broad_any"] == 1) & (df["mgmt_strong_any"] == 0)).sum())), random_state=42)
    samples_strong["snippet"] = samples_strong["text_for_keywords"].str.slice(0, 280)
    samples_broad["snippet"] = samples_broad["text_for_keywords"].str.slice(0, 280)
    samples_strong[["uid", "year", "title", "snippet"]].to_csv(tab_dir / "mgmt_strong_samples_50.csv", index=False)
    samples_broad[["uid", "year", "title", "snippet"]].to_csv(tab_dir / "mgmt_broad_only_samples_50.csv", index=False)
    return {"breakdown": tab, "n_strong_samples": len(samples_strong), "n_broad_samples": len(samples_broad)}


def credential_stacking_distribution(df: pd.DataFrame, tab_dir: Path, fig_dir: Path) -> pd.DataFrame:
    rows = []
    for year in ["2024", "2026"]:
        sub = df[df["year"] == year]
        vc = sub["credential_stack_depth"].value_counts().sort_index()
        for depth, n in vc.items():
            rows.append({"year": year, "credential_stack_depth": int(depth), "n": int(n), "share": round(n / len(sub), 4)})
    tab = pd.DataFrame(rows)
    tab.to_csv(tab_dir / "credential_stack_distribution.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    for year in ["2024", "2026"]:
        sub = tab[tab["year"] == year]
        ax.plot(sub["credential_stack_depth"], sub["share"] * 100, marker="o", label=year)
    ax.set_xlabel("Credential stack depth (# requirement categories)")
    ax.set_ylabel("Share of postings (%)")
    ax.set_title("Credential stacking distribution by year")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "credential_stack_distribution.png", dpi=150)
    plt.close(fig)
    return tab


def complexity_boxplot(df: pd.DataFrame, fig_dir: Path) -> None:
    metrics = ["tech_count", "soft_count", "scope_count", "requirement_breadth"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, m in zip(axes, metrics):
        data = [df[df["year"] == y][m].values for y in ["2024", "2026"]]
        ax.boxplot(data, labels=["2024", "2026"], showfliers=False)
        ax.set_title(m)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "complexity_distributions.png", dpi=150)
    plt.close(fig)


def outlier_analysis(df: pd.DataFrame, tab_dir: Path) -> pd.DataFrame:
    thresh = df["requirement_breadth"].quantile(0.99)
    top = df[df["requirement_breadth"] >= thresh].copy()
    top["snippet"] = top["text_for_keywords"].fillna("").str.slice(0, 400)
    out = top[["uid", "year", "source", "company_name_canonical", "title", "requirement_breadth", "tech_count", "soft_count", "scope_count", "text_len", "snippet"]].sort_values("requirement_breadth", ascending=False).head(40)
    out.to_csv(tab_dir / "top1pct_outliers.csv", index=False)
    return out


def aggregator_sensitivity(df: pd.DataFrame, tab_dir: Path) -> pd.DataFrame:
    """Essential sensitivity (a): exclude aggregators and recompute entry-level complexity deltas."""
    df = df[df["is_aggregator"] == False].copy()
    df["yoe_bucket"] = df["yoe_extracted"].apply(yoe_bucket)
    rows = []
    for op_name, col, val in [
        ("seniority_final=entry", "seniority_final", "entry"),
        ("yoe<=2", "yoe_bucket", "yoe<=2"),
    ]:
        for year in ["2024", "2026"]:
            s = df[(df["year"] == year) & (df[col] == val)]
            rows.append({
                "operationalization": op_name,
                "year": year,
                "n": len(s),
                **{m: round(float(s[m].mean()), 4) if len(s) else np.nan for m in METRICS},
            })
    tab = pd.DataFrame(rows)
    tab.to_csv(tab_dir / "sensitivity_noaggregators_entry.csv", index=False)
    return tab


def company_capping_sensitivity(df: pd.DataFrame, tab_dir: Path, cap: int = 20) -> pd.DataFrame:
    """Sensitivity (b): cap each company at N postings per year."""
    sub = (
        df.sort_values("uid")
        .groupby(["year", "company_name_canonical"], dropna=True)
        .head(cap)
        .copy()
    )
    sub["yoe_bucket"] = sub["yoe_extracted"].apply(yoe_bucket)
    rows = []
    for year in ["2024", "2026"]:
        s = sub[sub["year"] == year]
        rows.append({
            "year": year,
            "n": len(s),
            **{m: round(float(s[m].mean()), 4) for m in METRICS},
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(tab_dir / f"sensitivity_companycap{cap}_overall.csv", index=False)
    return tab


def main() -> None:
    print("Loading data…")
    df = load_frame()
    cleaned = load_cleaned_text()
    tech = load_tech_matrix()
    print(f"  SWE rows: {len(df):,}   cleaned: {len(cleaned):,}   tech matrix: {len(tech):,}")

    print("\nBuilding features…")
    feat = extract_features(df, cleaned, tech)
    print(f"  feature rows: {len(feat):,}")

    # Save per-row feature snapshot (small subset of columns) for downstream use
    keep_cols = ["uid", "source", "year", "title", "is_aggregator", "company_name_canonical", "seniority_best_available", "seniority_best_available_aug", "seniority_final", "yoe_extracted", "text_len"] + METRICS + [c for c in feat.columns if c.startswith(("softk_", "scopek_", "mgmts_", "mgmtb_"))]
    # De-dup any collisions just in case
    keep_cols = list(dict.fromkeys(keep_cols))
    feat[keep_cols].to_parquet(TAB_DIR / "T11_features.parquet", index=False)

    print("\n[2] Summary by year x seniority (full ablation)…")
    ablation = compare_seniority_ablation(feat, TAB_DIR)
    print(ablation["bestavail_aug"].to_string(index=False))

    print("\n[4-5] Entry-level scope inflation (two operationalizations x two variants)…")
    entry = entry_level_comparison(feat, TAB_DIR)
    print(entry["delta"].to_string(index=False))

    print("\n[6] Management term breakdown + samples…")
    mgmt = management_term_breakdown(feat, TAB_DIR)
    # Print top 10 strong + top 10 broad per year
    for year in ["2024", "2026"]:
        for tier in ["strong", "broad"]:
            sub = mgmt["breakdown"][(mgmt["breakdown"]["year"] == year) & (mgmt["breakdown"]["tier"] == tier)].head(10)
            print(f"  {year} {tier}:")
            print(sub[["term", "n_postings", "share"]].to_string(index=False))

    print("\n[4] Credential stacking distribution…")
    stack = credential_stacking_distribution(feat, TAB_DIR, FIG_DIR)
    print(stack.pivot(index="credential_stack_depth", columns="year", values="share").to_string())

    print("\n[2] Overall complexity distributions (plot)…")
    complexity_boxplot(feat, FIG_DIR)

    print("\n[8] Outliers (top 1% by requirement breadth)…")
    out = outlier_analysis(feat, TAB_DIR)
    print(out[["year", "source", "title", "requirement_breadth", "text_len"]].head(10).to_string(index=False))

    print("\n[Sensitivity a] Aggregator exclusion…")
    agg_sens = aggregator_sensitivity(feat, TAB_DIR)
    print(agg_sens.to_string(index=False))

    print("\n[Sensitivity b] Company capping (20/year)…")
    cap_sens = company_capping_sensitivity(feat, TAB_DIR, cap=20)
    print(cap_sens.to_string(index=False))

    print("\n[7] Domain archetype stratification…")
    archetype_path = SHARED / "swe_archetype_labels.parquet"
    if archetype_path.exists():
        con = duckdb.connect()
        arche = con.execute(f"SELECT * FROM '{archetype_path}'").fetchdf()
        feat2 = feat.merge(arche, on="uid", how="left")
        rows = []
        for year in ["2024", "2026"]:
            for arch in feat2["archetype"].dropna().unique():
                s = feat2[(feat2["year"] == year) & (feat2["archetype"] == arch) & (feat2["yoe_extracted"] <= 2)]
                rows.append({
                    "year": year,
                    "archetype": arch,
                    "n": len(s),
                    **{m: round(float(s[m].mean()), 4) if len(s) else np.nan for m in METRICS},
                })
        pd.DataFrame(rows).to_csv(TAB_DIR / "entry_by_archetype.csv", index=False)
    else:
        print("  T09 archetype labels not available; skipped.")

    # Summary JSON
    summary = {
        "n_rows": int(len(feat)),
        "entry_deltas": entry["delta"].to_dict(orient="records"),
        "mgmt_samples": {"strong": mgmt["n_strong_samples"], "broad_only": mgmt["n_broad_samples"]},
    }
    with open(TAB_DIR / "T11_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nDone. Outputs in:", TAB_DIR, "and", FIG_DIR)


if __name__ == "__main__":
    main()
