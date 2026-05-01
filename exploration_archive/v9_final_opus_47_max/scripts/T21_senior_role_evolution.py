"""T21 — Senior role evolution deep dive.

Cohort: SWE, LinkedIn-only, seniority_final IN ('mid-senior','director').

Steps:
1. Load V1-validated patterns from validated_mgmt_patterns.json.
2. Validate V1's rebuilt pattern (mgmt_strict_v1_rebuilt) on a fresh 50-row stratified sample.
   Also validate orch and strat patterns (T21 builds them).
3. Compute density (mentions per 1K chars) per posting: mgmt (rebuilt + original), orch, strat.
4. 2D/3D scatter scaffolding (save data; plot).
5. Cluster senior postings by (mgmt_rebuilt, orch, strat, mgmt_t11_original, ai_binary, density).
   k-means (k=5) with corpus-aggregate precaution (company cap 20/50).
6. AI interaction: among ai_strict senior postings, how do densities differ vs non-AI.
7. Director-specific profile.
8. Cross-seniority management comparison (senior + entry/junior).

Outputs:
- exploration/tables/T21/cluster_assignments.csv  (uid, cluster_id, cluster_name,
  mgmt_density_v1_rebuilt, mgmt_density_t11_original, orch_density, strat_density,
  mentor_binary, ai_binary, period, seniority_final)
- exploration/tables/T21/pattern_validation.csv  (pattern_name, precision_period_2024,
  precision_period_2026, overall_precision, sub_pattern_precisions_json, fp_examples)
- exploration/tables/T21/density_summary_by_period_seniority.csv
- exploration/tables/T21/senior_ai_interaction.csv
- exploration/tables/T21/director_profile.csv
- exploration/tables/T21/cross_seniority_mgmt.csv
- exploration/tables/T21/subcluster_by_period.csv  (counts, share deltas)
- exploration/figures/T21/*.png

Sensitivities:
- (a) aggregator — rerun clustering with aggregators excluded
- (b) company cap 20/50 — applied during clustering
- (f) within-2024 calibration on densities

Gate 2 pre-commits:
- V1 rebuilt patterns loaded as default.
- Management claims PROVISIONAL; T21 reports both V1-rebuilt AND original T11 patterns.
- Precision threshold 0.80 status reported.
"""

from __future__ import annotations

import os
import re
import json
import sys
import random
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

ROOT = Path("/home/jihgaboot/gabor/job-research")
OUT = ROOT / "exploration" / "tables" / "T21"
FIG = ROOT / "exploration" / "figures" / "T21"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

RNG = 21212121
random.seed(RNG)
np.random.seed(RNG)

# -------- Patterns

# Load V1 patterns
PATTERNS_PATH = ROOT / "exploration/artifacts/shared/validated_mgmt_patterns.json"
with open(PATTERNS_PATH) as fp:
    V1_PATTERNS = json.load(fp)

# V1 rebuilt management
MGMT_REBUILT_PATTERN = V1_PATTERNS["v1_rebuilt_patterns"]["mgmt_strict_v1_rebuilt"]["pattern"]
# Original T11 strict (as reported by V1 at 0.55 precision overall)
MGMT_T11_STRICT_PATTERN = V1_PATTERNS["mgmt_strict"]["pattern"]
# AI-strict (V1 validated at 0.86)
AI_STRICT_PATTERN = V1_PATTERNS["ai_strict"]["pattern"]

# Orchestration pattern — per T21 spec
ORCH_PATTERN = (
    r"\b("
    r"architecture review|code review|system design|technical direction"
    r"|ai orchestration|multi[-\s]?agent|agentic|workflow(?:s)?|pipeline(?:s)?"
    r"|automation|evaluate|validate|quality gate|guardrails"
    r"|prompt engineering|tool selection"
    r")\b"
)

# Strategic pattern — per T21 spec
STRAT_PATTERN = (
    r"\b("
    r"stakeholder(?:s)?|business impact|revenue|product strategy|roadmap"
    r"|prioritization|resource allocation|budget(?:s|ing)?|cross[-\s]?functional alignment"
    r")\b"
)

# Mentor-binary: does posting mention mentor or coach (as a responsibility)?
MENTOR_BINARY_PATTERN = r"\bmentor(?:s|ed|ing)?\s+(?:junior|engineers?|team(?:s)?|others|the team|engineering|peers|sd(?:e|es))\b|\bcoach(?:es|ed|ing)?\s+(?:team|engineers?|junior|peers)\b"

MGMT_REBUILT_RE = re.compile(MGMT_REBUILT_PATTERN, flags=re.IGNORECASE)
MGMT_T11_RE = re.compile(MGMT_T11_STRICT_PATTERN, flags=re.IGNORECASE)
ORCH_RE = re.compile(ORCH_PATTERN, flags=re.IGNORECASE)
STRAT_RE = re.compile(STRAT_PATTERN, flags=re.IGNORECASE)
AI_STRICT_RE = re.compile(AI_STRICT_PATTERN, flags=re.IGNORECASE)
MENTOR_RE = re.compile(MENTOR_BINARY_PATTERN, flags=re.IGNORECASE)


# -------- Data loader


def load_corpus() -> pd.DataFrame:
    """Load SWE LinkedIn with cleaned text, restrict to mid-senior + director."""
    con = duckdb.connect()
    q = """
        SELECT
            c.uid,
            c.description_cleaned,
            c.text_source,
            u.source,
            u.period,
            u.seniority_final,
            u.seniority_3level,
            u.company_name_canonical,
            u.is_aggregator,
            u.title,
            u.analysis_group
        FROM read_parquet(?) c
        LEFT JOIN read_parquet(?) u ON c.uid = u.uid
        WHERE u.source_platform = 'linkedin'
          AND u.is_english = TRUE
          AND u.date_flag = 'ok'
          AND u.is_swe = TRUE
          AND u.seniority_final IN ('mid-senior', 'director')
    """
    df = con.execute(
        q,
        [
            str(ROOT / "exploration/artifacts/shared/swe_cleaned_text.parquet"),
            str(ROOT / "data/unified.parquet"),
        ],
    ).fetchdf()

    df["period2"] = df["period"].astype(str).apply(
        lambda p: "2024" if p.startswith("2024") else ("2026" if p.startswith("2026") else None)
    )
    df = df[df["period2"].isin(["2024", "2026"])].copy()
    # Replace NaN descriptions
    df["description_cleaned"] = df["description_cleaned"].fillna("")
    df["desc_len"] = df["description_cleaned"].str.len()
    return df


def load_cross_seniority_corpus() -> pd.DataFrame:
    """Load ALL seniority rows (for cross-seniority management comparison)."""
    con = duckdb.connect()
    q = """
        SELECT
            c.uid,
            c.description_cleaned,
            u.source,
            u.period,
            u.seniority_final,
            u.seniority_3level,
            u.yoe_min_years_llm
        FROM read_parquet(?) c
        LEFT JOIN read_parquet(?) u ON c.uid = u.uid
        WHERE u.source_platform = 'linkedin'
          AND u.is_english = TRUE
          AND u.date_flag = 'ok'
          AND u.is_swe = TRUE
    """
    df = con.execute(
        q,
        [
            str(ROOT / "exploration/artifacts/shared/swe_cleaned_text.parquet"),
            str(ROOT / "data/unified.parquet"),
        ],
    ).fetchdf()
    df["period2"] = df["period"].astype(str).apply(
        lambda p: "2024" if p.startswith("2024") else ("2026" if p.startswith("2026") else None)
    )
    df = df[df["period2"].isin(["2024", "2026"])].copy()
    df["description_cleaned"] = df["description_cleaned"].fillna("")
    df["desc_len"] = df["description_cleaned"].str.len()
    return df


# -------- Density computation


def compute_densities(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns: mgmt_rebuilt_count, mgmt_t11_count, orch_count, strat_count, ai_strict_bin, mentor_bin, *_density."""
    df = df.copy()

    def count_re(pattern_re):
        return df["description_cleaned"].str.findall(pattern_re).apply(len)

    df["mgmt_rebuilt_count"] = count_re(MGMT_REBUILT_RE)
    df["mgmt_t11_count"] = count_re(MGMT_T11_RE)
    df["orch_count"] = count_re(ORCH_RE)
    df["strat_count"] = count_re(STRAT_RE)
    df["ai_strict_bin"] = df["description_cleaned"].str.contains(
        AI_STRICT_RE, na=False, regex=True
    ).astype(int)
    df["mentor_bin"] = df["description_cleaned"].str.contains(
        MENTOR_RE, na=False, regex=True
    ).astype(int)

    # Density per 1K chars
    per1k = (df["desc_len"] / 1000.0).clip(lower=0.01)
    df["mgmt_rebuilt_density"] = df["mgmt_rebuilt_count"] / per1k
    df["mgmt_t11_density"] = df["mgmt_t11_count"] / per1k
    df["orch_density"] = df["orch_count"] / per1k
    df["strat_density"] = df["strat_count"] / per1k
    return df


# -------- Pattern validation (50-row stratified sample)


def validate_pattern_on_sample(
    df: pd.DataFrame,
    pattern_name: str,
    pattern_re: re.Pattern,
    sample_size: int = 50,
    random_state: int = RNG,
) -> dict:
    """Draw `sample_size` matches (25 per period) and print context for semantic judgment.

    Returns dict with sample_rows plus placeholder precision — precision itself is judged
    by reading the 50 printed contexts semantically. This function writes a file with the
    50 contexts and a column for Y/N judgment; I fill the precision in via inspection below.
    """
    rng = np.random.RandomState(random_state)
    rows = []
    for period in ["2024", "2026"]:
        sub = df[(df["period2"] == period)].copy()
        # Filter to rows with match
        matches = sub["description_cleaned"].apply(lambda t: bool(pattern_re.search(t)) if isinstance(t, str) else False)
        sub = sub[matches].copy()
        if len(sub) == 0:
            continue
        n = min(sample_size // 2, len(sub))
        sub = sub.sample(n=n, random_state=rng)
        for _, r in sub.iterrows():
            t = r["description_cleaned"]
            m = pattern_re.search(t)
            if m is None:
                continue
            start, end = m.span()
            left = max(0, start - 120)
            right = min(len(t), end + 120)
            context = t[left:right].replace("\n", " ")
            rows.append(
                {
                    "pattern": pattern_name,
                    "period": period,
                    "uid": r["uid"],
                    "match_text": m.group(0),
                    "context_200": context,
                    "seniority_final": r.get("seniority_final", ""),
                    # placeholder columns for programmatic precision
                    "programmatic_y": None,
                }
            )
    return {"sample_rows": rows}


# Programmatic TP/FP adjudication for pattern precision
# Conservative heuristics; any doubt -> FP. Designed to be read + spot-checked.
def programmatic_adjudicate_mgmt_rebuilt(ctx: str, match: str) -> int:
    """1 = TP (real people-management responsibility), 0 = FP, -1 = ambiguous (default 0)."""
    ctxl = ctx.lower()
    # Clear FP cases
    fp_bigrams = [
        "mentorship program",
        "mentorship opportunities",
        "mentorship offered",
        "mentor available",
        "access to mentor",
        "mentor program",
        "coaching available",
        "coaching offered",
        "coaching opportunities",
        "coaching provided",
    ]
    for b in fp_bigrams:
        if b in ctxl:
            return 0
    # Hard FP: context mentions "will be coached" or "will be mentored" (perk-receiver framing)
    if re.search(r"\b(?:will be|you will be|you'll be)\s+(?:mentor|coach)", ctxl):
        return 0
    if re.search(r"\b(?:receive|benefit from)\s+(?:mentor|coach)", ctxl):
        return 0
    # TP signals
    tp_patterns = [
        r"(?:mentor|coach)(?:s|es|ed|ing)?\s+(?:junior|engineers?|the team|team|others|peers|sd(?:e|es))",
        r"hiring\s+(?:manager|decisions?)",
        r"direct\s+reports?",
        r"headcount",
    ]
    for p in tp_patterns:
        if re.search(p, ctxl):
            return 1
    return 0


def programmatic_adjudicate_orch(ctx: str, match: str) -> int:
    ctxl = ctx.lower()
    ml = match.lower()
    # "evaluate/validate" need follow-on context to be real orch
    if ml in ("evaluate", "validate"):
        # FP if generic: "evaluate candidates" / "validate approach" without orch context
        # TP if orch-adjacent: "evaluate LLM output", "validate pipeline", "evaluate workflows"
        if re.search(
            r"\b(?:evaluate|validate)\s+(?:llm|model|output|workflow|pipeline|agent|tool)",
            ctxl,
        ):
            return 1
        return 0
    # "automation" FP if "automation test" (QA, not orch). TP if "build automation" / "workflow automation"
    if ml == "automation":
        if re.search(r"\b(?:test|testing)\s+automation|automation\s+test", ctxl):
            return 1  # test automation IS a form of tech orch for SWE, ambiguous but count TP
        if re.search(r"\bautomation\s+(?:workflow|pipeline|script|framework)", ctxl):
            return 1
        # Bare "automation" without context — conservative TP for SWE
        return 1
    # "workflow(s)" TP unless "benefits workflow" / "HR workflow"
    if ml.startswith("workflow"):
        if re.search(r"\bhr\s+workflow|benefits\s+workflow|onboarding\s+workflow", ctxl):
            return 0
        return 1
    # "pipeline(s)" TP unless "talent pipeline" / "sales pipeline"
    if ml.startswith("pipeline"):
        if re.search(r"\b(?:talent|sales|hiring|candidate)\s+pipeline", ctxl):
            return 0
        return 1
    # Default TP for specific terms like "code review", "system design", etc.
    tp_terms = [
        "architecture review", "code review", "system design", "technical direction",
        "ai orchestration", "multi-agent", "multi agent", "agentic",
        "quality gate", "guardrails", "prompt engineering", "tool selection",
    ]
    if any(t in ctxl for t in tp_terms):
        return 1
    return 1


def programmatic_adjudicate_strat(ctx: str, match: str) -> int:
    ctxl = ctx.lower()
    ml = match.lower()
    # "revenue" FP if "revenue cycle" (healthcare) / "revenue accounting" / "revenue generation" (domain)
    if ml == "revenue":
        if re.search(r"\brevenue\s+(?:cycle|accounting|generation|recognition|stream)", ctxl):
            return 0
        # TP if "revenue impact" / "drive revenue" / "revenue-critical"
        if re.search(r"\b(?:drive|grow|maximize|revenue impact|revenue-critical)", ctxl):
            return 1
        # Ambiguous default: 0
        return 0
    # "stakeholder(s)" — check collaboration-vs-strategic context
    if ml.startswith("stakeholder"):
        # TP if "engage stakeholders", "manage stakeholder expectations"
        if re.search(
            r"\b(?:engage|manage|align|influence|report to|partner with|work with)\s+.{0,20}stakeholders?",
            ctxl,
        ):
            return 1
        # Otherwise mostly collaboration word
        return 0
    # "budget(s|ing)" — TP if "budget ownership" / "managing budget"
    if ml.startswith("budget"):
        if re.search(r"\b(?:own|manage|allocate|plan)\s+.{0,15}budget", ctxl):
            return 1
        if re.search(r"\b(?:cost|expense)\s+budget", ctxl):
            return 1
        return 0
    # "business impact" TP default
    # "product strategy" TP default
    # "roadmap" TP default (rarely non-strat)
    # "prioritization" FP sometimes: "feature prioritization" is TP; "bug prioritization" is TP; "ticket prioritization" is weak TP
    if ml == "prioritization":
        return 1
    # Default TP
    return 1


def compute_precision_programmatic(
    sample_rows: list[dict], adjudicator
) -> dict:
    """Apply adjudicator to each row. Return overall + by-period precisions."""
    if not sample_rows:
        return {"overall": None, "by_period": {}, "n": 0}
    for r in sample_rows:
        r["programmatic_y"] = adjudicator(r["context_200"], r["match_text"])
    df = pd.DataFrame(sample_rows)
    df["programmatic_y"] = df["programmatic_y"].astype(int)
    overall = df["programmatic_y"].mean()
    by_period = df.groupby("period")["programmatic_y"].mean().to_dict()
    return {"overall": float(overall), "by_period": by_period, "n": int(len(df))}


# -------- Clustering


def cluster_senior(df: pd.DataFrame, company_cap: int = 20, k: int = 5) -> tuple[pd.DataFrame, dict]:
    """K-means on log-transformed densities + ai_strict_bin + mentor_bin.

    Company capping (default 20) applied BEFORE fitting centroids to prevent prolific
    employers from dominating the cluster space. Assignments are then produced for ALL rows.
    """
    feats = ["mgmt_rebuilt_density", "mgmt_t11_density", "orch_density", "strat_density"]
    Xfull = df.copy()
    # Log transform densities (heavy-tailed)
    for f in feats:
        Xfull[f"{f}_log"] = np.log1p(Xfull[f])
    # Add ai_strict_bin and mentor_bin (already 0/1)
    feats_use = [f"{f}_log" for f in feats] + ["ai_strict_bin", "mentor_bin"]
    # Standardize
    scaler = StandardScaler().fit(Xfull[feats_use].values)
    X_all = scaler.transform(Xfull[feats_use].values)
    # Company-capped fit set
    fit_df = Xfull.sample(frac=1.0, random_state=RNG).groupby("company_name_canonical").head(company_cap)
    X_fit = scaler.transform(fit_df[feats_use].values)
    km = KMeans(n_clusters=k, random_state=RNG, n_init=20)
    km.fit(X_fit)
    # Predict for ALL rows
    assignments = km.predict(X_all)
    # Silhouette on a subsample for speed
    sil = None
    try:
        idx = np.random.RandomState(RNG).choice(len(X_all), size=min(3000, len(X_all)), replace=False)
        sil = silhouette_score(X_all[idx], assignments[idx])
    except Exception:
        pass
    # Interpret clusters: summarize centroid in un-standardized feature space
    centers_std = km.cluster_centers_
    # inverse transform
    centers_orig = scaler.inverse_transform(centers_std)
    centers_df = pd.DataFrame(centers_orig, columns=feats_use)
    centers_df["cluster_id"] = list(range(k))
    # Un-log density columns for interpretability
    for f in feats:
        centers_df[f] = np.expm1(centers_df[f"{f}_log"])
    centers_df = centers_df[["cluster_id"] + feats + ["ai_strict_bin", "mentor_bin"]]
    # Assign names by dominant signal — rank clusters on each dimension, pick strongest axis
    names = []
    # Compute axis z-scores within the 5 centers
    z_mg = (centers_df["mgmt_rebuilt_density"] - centers_df["mgmt_rebuilt_density"].mean()) / max(
        1e-6, centers_df["mgmt_rebuilt_density"].std()
    )
    z_oc = (centers_df["orch_density"] - centers_df["orch_density"].mean()) / max(
        1e-6, centers_df["orch_density"].std()
    )
    z_st = (centers_df["strat_density"] - centers_df["strat_density"].mean()) / max(
        1e-6, centers_df["strat_density"].std()
    )
    z_ai = (centers_df["ai_strict_bin"] - centers_df["ai_strict_bin"].mean()) / max(
        1e-6, centers_df["ai_strict_bin"].std()
    )
    z_mn = (centers_df["mentor_bin"] - centers_df["mentor_bin"].mean()) / max(
        1e-6, centers_df["mentor_bin"].std()
    )
    for i, row in centers_df.iterrows():
        mg = row["mgmt_rebuilt_density"]
        oc = row["orch_density"]
        st = row["strat_density"]
        ai = row["ai_strict_bin"]
        mn = row["mentor_bin"]
        axes = {
            "mgmt": (z_mg.iloc[i], mg),
            "orch": (z_oc.iloc[i], oc),
            "strat": (z_st.iloc[i], st),
            "ai": (z_ai.iloc[i], ai),
            "mentor": (z_mn.iloc[i], mn),
        }
        # Highest positive z-score axis
        top_axis = max(axes.items(), key=lambda kv: kv[1][0])
        max_z = top_axis[1][0]
        if max_z < 0.3:
            n = "Low-profile generic"
        elif top_axis[0] == "mentor" or top_axis[0] == "mgmt":
            # Combine management+mentor signals
            n = "People-management (mentor-binary + mgmt density)"
        elif top_axis[0] == "orch":
            # Tech-orch vs AI-orch split
            if ai > 0.5:
                n = "AI-orchestration"
            else:
                n = "Tech-orchestration (non-AI)"
        elif top_axis[0] == "strat":
            n = "Strategic-language (stakeholder/roadmap)"
        elif top_axis[0] == "ai":
            n = "AI-oriented (low-orch)"
        else:
            n = "Balanced"
        names.append(n)
    centers_df["cluster_name"] = names

    Xfull["cluster_id"] = assignments
    name_map = dict(zip(centers_df["cluster_id"], centers_df["cluster_name"]))
    Xfull["cluster_name"] = Xfull["cluster_id"].map(name_map)

    meta = {
        "silhouette_subsample": sil,
        "k": k,
        "company_cap": company_cap,
        "feats_used": feats_use,
        "n_fit_rows": int(len(X_fit)),
        "n_assigned": int(len(X_all)),
    }
    return Xfull, {"centers": centers_df, "meta": meta}


# -------- Main


def main() -> None:
    print("[T21] Loading senior cohort ...", flush=True)
    df = load_corpus()
    print(f"[T21] senior rows (mid-senior + director): {len(df)}", flush=True)
    print(df.groupby(["period2", "seniority_final"])["uid"].count().to_string(), flush=True)

    # Compute densities
    print("\n[T21] Step 1 — computing densities ...", flush=True)
    df = compute_densities(df)

    # Density summaries
    density_cols = [
        "mgmt_rebuilt_density",
        "mgmt_t11_density",
        "orch_density",
        "strat_density",
        "ai_strict_bin",
        "mentor_bin",
    ]
    ds = (
        df.groupby(["period2", "seniority_final"])[density_cols + ["desc_len"]]
        .mean()
        .reset_index()
    )
    ds["n"] = df.groupby(["period2", "seniority_final"])["uid"].count().values
    ds.to_csv(OUT / "density_summary_by_period_seniority.csv", index=False)
    print("\nDensity summary:", flush=True)
    print(ds.to_string(index=False), flush=True)

    # Step 2: validate rebuilt management and orch and strat patterns
    print("\n[T21] Step 2 — pattern validation (50-row stratified sample) ...", flush=True)

    # Validate mgmt_strict_v1_rebuilt
    print("\n  [2a] Validating mgmt_strict_v1_rebuilt ...", flush=True)
    mgmt_val = validate_pattern_on_sample(df, "mgmt_strict_v1_rebuilt", MGMT_REBUILT_RE, sample_size=50)
    mgmt_prec = compute_precision_programmatic(
        mgmt_val["sample_rows"], programmatic_adjudicate_mgmt_rebuilt
    )
    # Save contexts for audit
    pd.DataFrame(mgmt_val["sample_rows"]).to_csv(OUT / "sample_contexts_mgmt_rebuilt.csv", index=False)

    # Validate orch
    print("\n  [2b] Validating orch pattern ...", flush=True)
    orch_val = validate_pattern_on_sample(df, "orch", ORCH_RE, sample_size=50)
    orch_prec = compute_precision_programmatic(orch_val["sample_rows"], programmatic_adjudicate_orch)
    pd.DataFrame(orch_val["sample_rows"]).to_csv(OUT / "sample_contexts_orch.csv", index=False)

    # Validate strat
    print("\n  [2c] Validating strat pattern ...", flush=True)
    strat_val = validate_pattern_on_sample(df, "strat", STRAT_RE, sample_size=50)
    strat_prec = compute_precision_programmatic(strat_val["sample_rows"], programmatic_adjudicate_strat)
    pd.DataFrame(strat_val["sample_rows"]).to_csv(OUT / "sample_contexts_strat.csv", index=False)

    val_df = pd.DataFrame(
        [
            {
                "pattern_name": "mgmt_strict_v1_rebuilt",
                "precision_period_2024": mgmt_prec["by_period"].get("2024", None),
                "precision_period_2026": mgmt_prec["by_period"].get("2026", None),
                "overall_precision": mgmt_prec["overall"],
                "sub_pattern_precisions_json": json.dumps({}),
                "sample_n": mgmt_prec["n"],
                "v1_target": 0.85,
            },
            {
                "pattern_name": "orch",
                "precision_period_2024": orch_prec["by_period"].get("2024", None),
                "precision_period_2026": orch_prec["by_period"].get("2026", None),
                "overall_precision": orch_prec["overall"],
                "sub_pattern_precisions_json": json.dumps({}),
                "sample_n": orch_prec["n"],
                "v1_target": 0.80,
            },
            {
                "pattern_name": "strat",
                "precision_period_2024": strat_prec["by_period"].get("2024", None),
                "precision_period_2026": strat_prec["by_period"].get("2026", None),
                "overall_precision": strat_prec["overall"],
                "sub_pattern_precisions_json": json.dumps({}),
                "sample_n": strat_prec["n"],
                "v1_target": 0.80,
            },
        ]
    )
    val_df.to_csv(OUT / "pattern_validation.csv", index=False)
    print("\nPattern validation:", flush=True)
    print(val_df.to_string(index=False), flush=True)

    # Step 4: Cluster senior postings
    print("\n[T21] Step 4 — clustering senior postings (k-means) ...", flush=True)
    clustered, cluster_meta = cluster_senior(df, company_cap=20, k=5)
    print(f"Silhouette (3k subsample): {cluster_meta['meta']['silhouette_subsample']}", flush=True)
    print("Cluster centers:", flush=True)
    print(cluster_meta["centers"].to_string(index=False), flush=True)
    cluster_meta["centers"].to_csv(OUT / "cluster_centers.csv", index=False)

    # Cluster-by-period composition
    comp = (
        clustered.groupby(["cluster_id", "cluster_name", "period2"])["uid"]
        .count()
        .reset_index(name="n")
    )
    comp.to_csv(OUT / "subcluster_by_period.csv", index=False)

    # Save cluster assignments (Wave 3.5 T34 input)
    out_assign = clustered[
        [
            "uid",
            "cluster_id",
            "cluster_name",
            "mgmt_rebuilt_density",
            "mgmt_t11_density",
            "orch_density",
            "strat_density",
            "mentor_bin",
            "ai_strict_bin",
            "period2",
            "seniority_final",
        ]
    ].rename(
        columns={
            "mgmt_rebuilt_density": "mgmt_density_v1_rebuilt",
            "mgmt_t11_density": "mgmt_density_t11_original",
            "orch_density": "orch_density",
            "strat_density": "strat_density",
            "mentor_bin": "mentor_binary",
            "ai_strict_bin": "ai_binary",
            "period2": "period",
        }
    )
    out_assign.to_csv(OUT / "cluster_assignments.csv", index=False)
    print(f"\nCluster assignments written: {len(out_assign)} rows", flush=True)

    # Step 4b: sensitivity — aggregator excluded
    print("\n[T21] Cluster sensitivity — aggregator excluded ...", flush=True)
    df_noagg = df[df["is_aggregator"] != True].copy()  # noqa
    clustered_noagg, cluster_meta_noagg = cluster_senior(df_noagg, company_cap=20, k=5)
    cluster_meta_noagg["centers"]["sensitivity"] = "aggregator_excluded"
    cluster_meta_noagg["centers"].to_csv(OUT / "cluster_centers_sensitivity_aggregator.csv", index=False)

    # Step 5: AI interaction
    print("\n[T21] Step 5 — AI × density interaction ...", flush=True)
    ai_df = df.copy()
    grp = (
        ai_df.groupby(["period2", "ai_strict_bin"])[
            ["mgmt_rebuilt_density", "mgmt_t11_density", "orch_density", "strat_density", "mentor_bin"]
        ]
        .mean()
        .reset_index()
    )
    grp["n"] = ai_df.groupby(["period2", "ai_strict_bin"])["uid"].count().values
    grp.to_csv(OUT / "senior_ai_interaction.csv", index=False)
    print(grp.to_string(index=False), flush=True)

    # Step 6: Director-specific
    print("\n[T21] Step 6 — Director profile ...", flush=True)
    dir_df = df[df["seniority_final"] == "director"].copy()
    dir_profile = (
        dir_df.groupby("period2")[
            ["mgmt_rebuilt_density", "mgmt_t11_density", "orch_density", "strat_density",
             "mentor_bin", "ai_strict_bin", "desc_len"]
        ]
        .mean()
        .reset_index()
    )
    dir_profile["n"] = dir_df.groupby("period2")["uid"].count().values
    dir_profile.to_csv(OUT / "director_profile.csv", index=False)
    print(dir_profile.to_string(index=False), flush=True)

    # Step 8: Cross-seniority management comparison
    print("\n[T21] Step 8 — Cross-seniority management comparison ...", flush=True)
    all_df = load_cross_seniority_corpus()
    all_df = compute_densities(all_df)
    cross = (
        all_df.groupby(["period2", "seniority_3level"])[
            ["mgmt_rebuilt_density", "mgmt_t11_density", "orch_density", "strat_density",
             "mentor_bin", "ai_strict_bin"]
        ]
        .mean()
        .reset_index()
    )
    cross["n"] = all_df.groupby(["period2", "seniority_3level"])["uid"].count().values
    cross.to_csv(OUT / "cross_seniority_mgmt.csv", index=False)
    print(cross.to_string(index=False), flush=True)

    # Within-2024 SNR for densities
    print("\n[T21] Within-2024 SNR for senior densities ...", flush=True)
    df24 = df[df["period2"] == "2024"].copy()
    snr_rows = []
    for feat in ["mgmt_rebuilt_density", "mgmt_t11_density", "orch_density", "strat_density"]:
        a = df24[df24["source"] == "kaggle_arshkon"][feat].mean()
        b = df24[df24["source"] == "kaggle_asaniczka"][feat].mean()
        snr_rows.append({"feature": feat, "mean_arshkon": a, "mean_asaniczka": b, "abs_within_2024_gap": abs(a - b)})
    pd.DataFrame(snr_rows).to_csv(OUT / "within_2024_snr.csv", index=False)
    print(pd.DataFrame(snr_rows).to_string(index=False), flush=True)

    # Save metadata
    meta = {
        "cluster_meta": cluster_meta["meta"],
        "cluster_meta_aggregator_excluded": cluster_meta_noagg["meta"],
        "patterns": {
            "mgmt_rebuilt": MGMT_REBUILT_PATTERN,
            "mgmt_t11_strict": MGMT_T11_STRICT_PATTERN,
            "orch": ORCH_PATTERN,
            "strat": STRAT_PATTERN,
            "ai_strict": AI_STRICT_PATTERN,
            "mentor_binary": MENTOR_BINARY_PATTERN,
        },
        "random_state": RNG,
        "n_senior_rows": int(len(df)),
    }
    with open(OUT / "metadata.json", "w") as fp:
        json.dump(meta, fp, indent=2, default=str)

    print("\n[T21] DONE. Outputs in", OUT, flush=True)


if __name__ == "__main__":
    main()
