"""T28 — Domain-stratified scope changes by archetype.

Loads T09 archetype labels, joins to the SWE LinkedIn corpus, and runs:
  1. Archetype distribution by period.
  2. Domain x seniority decomposition for entry share (seniority_final AND YOE <=2).
     Kitagawa-style within/between/interaction decomposition.
  3. Domain-stratified scope/content changes (req_breadth, tech_count, scope_density,
     AI mention rate narrow+broad, credential stack depth).
  4. Junior vs senior content within each archetype.
  5. Senior archetype mentoring shift by domain.
  6. LLM/GenAI archetype deep dive.
  7. Sensitivities (aggregator exclusion, seniority operationalization).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNIFIED = ROOT / "data" / "unified.parquet"
SHARED = ROOT / "exploration" / "artifacts" / "shared"
OUT_TABLES = ROOT / "exploration" / "tables" / "T28"
OUT_FIGS = ROOT / "exploration" / "figures" / "T28"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

DEFAULT_FILTER = (
    "is_swe=true AND source_platform='linkedin' AND is_english=true AND date_flag='ok'"
)


# ---------------------------------------------------------------------------
# Pattern definitions (clean patterns only, per Gate 2 corrections)
# ---------------------------------------------------------------------------

# Scope patterns: end-to-end + cross-functional only (ownership is contaminated)
SCOPE_PATTERNS = {
    "end_to_end": re.compile(r"\bend[-\s]?to[-\s]?end\b", re.IGNORECASE),
    "cross_functional": re.compile(r"\bcross[-\s]?functional\b", re.IGNORECASE),
}

# Mentoring: validated clean pattern (mentor, mentoring, mentorship). No bare 'team'.
MENTOR_PATTERN = re.compile(r"\bmentor(?:ing|ship|s|ed)?\b", re.IGNORECASE)
# Broader management (still informative but flagged as noisier)
MANAGE_PATTERN = re.compile(r"\b(?:manage|managing|managed|manager)\b", re.IGNORECASE)

# AI mention patterns
AI_NARROW_PATTERN = re.compile(
    r"(?<![A-Za-z])(?:ai|artificial intelligence)(?![A-Za-z])", re.IGNORECASE
)
# Broad union list (T14 24-term union approximation using our tech_matrix cols)
AI_BROAD_TECH_COLS = [
    "llm",
    "langchain",
    "rag",
    "openai_api",
    "claude_api",
    "agents_framework",
    "gpt",
    "transformer_arch",
    "embedding",
    "copilot",
    "cursor_tool",
    "chatgpt",
    "claude_tool",
    "fine_tuning",
    "nlp",
    "pytorch",
    "tensorflow",
]

# Credential stack: years experience, degrees, certifications
CREDENTIAL_PATTERNS = {
    "years_exp": re.compile(r"\b\d+\+?\s*(?:\-\s*\d+\s*)?(?:years?|yrs)\b", re.IGNORECASE),
    "bachelor": re.compile(
        r"\b(?:bachelor|b\.?s\.?|b\.?a\.?|bsc|undergraduate degree)\b", re.IGNORECASE
    ),
    "master": re.compile(r"\b(?:master|m\.?s\.?|msc|graduate degree|m\.?eng\.?)\b", re.IGNORECASE),
    "phd": re.compile(r"\b(?:ph\.?d|doctorate|doctoral)\b", re.IGNORECASE),
    "certification": re.compile(
        r"\b(?:certif(?:ied|ication)|aws\s+certif|azure\s+certif|google\s+cloud\s+certif|pmp|scrum\s+master)\b",
        re.IGNORECASE,
    ),
}

# Requirement breadth categories: count distinct categories present
REQ_BREADTH_CATEGORIES = {
    "programming": re.compile(
        r"\b(?:python|java|javascript|typescript|golang|go|rust|c\+\+|c#|ruby|scala|kotlin|swift)\b",
        re.IGNORECASE,
    ),
    "cloud": re.compile(r"\b(?:aws|azure|gcp|google cloud|cloud)\b", re.IGNORECASE),
    "db": re.compile(
        r"\b(?:sql|postgres|mysql|mongodb|redis|cassandra|dynamodb|database)\b", re.IGNORECASE
    ),
    "frontend": re.compile(r"\b(?:react|angular|vue|html|css|frontend|front-end)\b", re.IGNORECASE),
    "backend": re.compile(
        r"\b(?:backend|back-end|api|rest|graphql|microservice)\b", re.IGNORECASE
    ),
    "devops": re.compile(
        r"\b(?:docker|kubernetes|k8s|ci/cd|cicd|jenkins|terraform|devops)\b", re.IGNORECASE
    ),
    "testing": re.compile(r"\b(?:test|testing|qa|unit test|integration test|tdd)\b", re.IGNORECASE),
    "agile": re.compile(r"\b(?:agile|scrum|kanban|sprint)\b", re.IGNORECASE),
    "security": re.compile(r"\b(?:security|authentication|authorization|oauth|encryption)\b", re.IGNORECASE),
    "data_ai": re.compile(
        r"\b(?:machine learning|ml|ai|deep learning|data science|analytics|etl)\b", re.IGNORECASE
    ),
    "communication": re.compile(
        r"\b(?:communication|stakeholder|collaborate|cross-functional)\b", re.IGNORECASE
    ),
    "leadership": re.compile(r"\b(?:lead|mentor|guide|coach)\b", re.IGNORECASE),
    "design": re.compile(r"\b(?:design|architect|ux|ui|user experience)\b", re.IGNORECASE),
    "degree": re.compile(r"\b(?:bachelor|master|degree|ph\.?d)\b", re.IGNORECASE),
}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_corpus() -> pd.DataFrame:
    """Load SWE LinkedIn corpus with archetype labels joined."""
    con = duckdb.connect()
    arche = pd.read_parquet(SHARED / "swe_archetype_labels.parquet")
    tech = pd.read_parquet(SHARED / "swe_tech_matrix.parquet")

    df = con.execute(
        f"""
        SELECT uid, period, source, is_aggregator, company_name_canonical,
               seniority_final, yoe_extracted,
               description, description_core_llm,
               llm_extraction_coverage, description_length
        FROM read_parquet('{UNIFIED}')
        WHERE {DEFAULT_FILTER}
        """
    ).fetchdf()
    con.close()

    df = df.merge(arche, on="uid", how="left")
    # Period collapse: 2024-01 + 2024-04 -> 2024; 2026-03 + 2026-04 -> 2026
    df["period2"] = df["period"].map(
        lambda p: "2024" if str(p).startswith("2024") else "2026"
    )
    # Seniority buckets
    df["is_entry_final"] = df["seniority_final"].eq("entry")
    df["is_senior_final"] = df["seniority_final"].isin(["senior", "staff", "principal"])
    df["is_mid_final"] = df["seniority_final"].isin(["mid", "mid-senior", "mid_senior"])
    df["seniority_3level"] = np.where(
        df["is_entry_final"],
        "entry",
        np.where(df["is_senior_final"] | df["is_mid_final"], "mid_senior", "unknown"),
    )
    df["yoe_le2"] = df["yoe_extracted"].le(2).fillna(False)
    df["yoe_known"] = df["yoe_extracted"].notna()

    # Select primary text column: description_core_llm when labeled, else raw
    df["text_clean"] = np.where(
        df["llm_extraction_coverage"].eq("labeled"),
        df["description_core_llm"].fillna(""),
        "",
    )
    df["text_raw"] = df["description"].fillna("")
    df["has_clean_text"] = df["text_clean"].str.len().gt(0)

    # Join tech matrix
    tech_cols = [c for c in tech.columns if c != "uid"]
    df = df.merge(tech, on="uid", how="left")
    for c in tech_cols:
        if c in df.columns:
            df[c] = df[c].fillna(False)

    return df, tech_cols


# ---------------------------------------------------------------------------
# Per-row metrics (computed only on has_clean_text rows for density metrics)
# ---------------------------------------------------------------------------

def compute_text_metrics(df: pd.DataFrame, tech_cols: list) -> pd.DataFrame:
    """Compute all per-row content metrics."""
    text = df["text_clean"]
    raw = df["text_raw"]
    n = len(df)
    out = pd.DataFrame(index=df.index)

    # Character length (cleaned text; 0 for raw-only rows)
    out["clean_len"] = text.str.len().fillna(0).astype(int)
    out["raw_len"] = raw.str.len().fillna(0).astype(int)

    # Scope density (clean patterns): mentions per 1K chars (clean text only)
    scope_hits = np.zeros(n, dtype=int)
    for pat in SCOPE_PATTERNS.values():
        scope_hits = scope_hits + text.str.count(pat.pattern).fillna(0).astype(int).to_numpy()
    out["scope_hits"] = scope_hits
    out["scope_density"] = np.where(
        out["clean_len"] > 0, (scope_hits * 1000.0) / out["clean_len"], np.nan
    )
    # Binary: any scope mention (in either clean or raw so we have coverage)
    any_scope_text = np.where(out["clean_len"] > 0, text, raw)
    out["any_scope"] = pd.Series(any_scope_text).str.contains(
        r"\bend[-\s]?to[-\s]?end\b|\bcross[-\s]?functional\b",
        case=False,
        regex=True,
        na=False,
    ).to_numpy()

    # Mentoring binary (any mention in clean, else raw for recall)
    out["any_mentor"] = pd.Series(any_scope_text).str.contains(
        MENTOR_PATTERN.pattern, case=False, regex=True, na=False
    ).to_numpy()
    out["any_manage"] = pd.Series(any_scope_text).str.contains(
        MANAGE_PATTERN.pattern, case=False, regex=True, na=False
    ).to_numpy()

    # AI narrow (binary on raw; this is boilerplate-insensitive enough)
    out["any_ai_narrow"] = raw.str.contains(AI_NARROW_PATTERN.pattern, case=False, regex=True, na=False).to_numpy()

    # AI broad: any of 17 tech-matrix AI terms
    broad_present = np.zeros(n, dtype=bool)
    for c in AI_BROAD_TECH_COLS:
        if c in df.columns:
            broad_present = broad_present | df[c].fillna(False).to_numpy().astype(bool)
    out["any_ai_broad"] = broad_present

    # Tech count (sum over all 123 tech columns)
    tech_count = np.zeros(n, dtype=int)
    for c in tech_cols:
        tech_count = tech_count + df[c].fillna(False).astype(int).to_numpy()
    out["tech_count"] = tech_count

    # Requirement breadth: count of distinct REQ_BREADTH categories present in text
    breadth_text = np.where(out["clean_len"] > 0, text, raw)
    breadth = np.zeros(n, dtype=int)
    for name, pat in REQ_BREADTH_CATEGORIES.items():
        hits = pd.Series(breadth_text).str.contains(pat.pattern, case=False, regex=True, na=False).to_numpy()
        breadth = breadth + hits.astype(int)
    out["requirement_breadth"] = breadth

    # Credential stack depth: count of distinct CREDENTIAL categories present
    cred = np.zeros(n, dtype=int)
    for name, pat in CREDENTIAL_PATTERNS.items():
        hits = pd.Series(breadth_text).str.contains(pat.pattern, case=False, regex=True, na=False).to_numpy()
        cred = cred + hits.astype(int)
    out["credential_stack_depth"] = cred

    return out


# ---------------------------------------------------------------------------
# Analysis: Step 1 — archetype distribution
# ---------------------------------------------------------------------------

def step1_archetype_distribution(df: pd.DataFrame) -> pd.DataFrame:
    by_period = (
        df.groupby(["period2", "archetype", "archetype_name"]).size().reset_index(name="n")
    )
    total = by_period.groupby("period2")["n"].transform("sum")
    by_period["share"] = by_period["n"] / total
    wide = by_period.pivot_table(
        index=["archetype", "archetype_name"],
        columns="period2",
        values=["n", "share"],
        fill_value=0,
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index().sort_values("archetype")
    wide["delta_share_pp"] = (wide.get("share_2026", 0) - wide.get("share_2024", 0)) * 100
    wide.to_csv(OUT_TABLES / "step1_archetype_distribution.csv", index=False)
    return wide


# ---------------------------------------------------------------------------
# Step 2 — Domain x seniority decomposition for entry share
# ---------------------------------------------------------------------------

def kitagawa_decompose(df: pd.DataFrame, entry_flag: str, label: str) -> pd.DataFrame:
    """Decompose aggregate 2024->2026 change in entry share into within/between/interaction.

    agg_2024 = sum_a w_{a,2024} * r_{a,2024}
    agg_2026 = sum_a w_{a,2026} * r_{a,2026}
    within  = sum_a w_{a,2024} * (r_{a,2026} - r_{a,2024})
    between = sum_a r_{a,2024} * (w_{a,2026} - w_{a,2024})
    interaction = residual
    """
    # Use ONLY labeled archetype rows (exclude archetype == -2)
    sub = df[df["archetype"] != -2].copy()
    # Aggregate rates by archetype x period
    grp = sub.groupby(["archetype", "archetype_name", "period2"]).agg(
        n=("uid", "count"), entry=(entry_flag, "sum")
    ).reset_index()
    grp["rate"] = grp["entry"] / grp["n"]

    # Pivot
    piv = grp.pivot_table(
        index=["archetype", "archetype_name"],
        columns="period2",
        values=["n", "rate"],
        fill_value=0,
    )
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()

    # Weights (share of labeled corpus in each period)
    total_2024 = piv["n_2024"].sum()
    total_2026 = piv["n_2026"].sum()
    piv["w_2024"] = piv["n_2024"] / total_2024 if total_2024 else 0
    piv["w_2026"] = piv["n_2026"] / total_2026 if total_2026 else 0
    piv["delta_rate"] = piv["rate_2026"] - piv["rate_2024"]
    piv["delta_w"] = piv["w_2026"] - piv["w_2024"]
    piv["within_contrib"] = piv["w_2024"] * piv["delta_rate"]
    piv["between_contrib"] = piv["rate_2024"] * piv["delta_w"]
    piv["interaction_contrib"] = piv["delta_w"] * piv["delta_rate"]

    within = piv["within_contrib"].sum()
    between = piv["between_contrib"].sum()
    interaction = piv["interaction_contrib"].sum()
    agg_2024 = (piv["w_2024"] * piv["rate_2024"]).sum()
    agg_2026 = (piv["w_2026"] * piv["rate_2026"]).sum()
    total_change = agg_2026 - agg_2024

    summary = pd.DataFrame(
        {
            "metric": [label],
            "agg_2024": [agg_2024],
            "agg_2026": [agg_2026],
            "total_change": [total_change],
            "within_domain_component": [within],
            "between_domain_component": [between],
            "interaction_component": [interaction],
            "within_pct": [within / total_change * 100 if total_change else np.nan],
            "between_pct": [between / total_change * 100 if total_change else np.nan],
        }
    )

    piv = piv.sort_values("archetype")
    piv.to_csv(
        OUT_TABLES / f"step2_decomp_per_archetype_{label}.csv", index=False
    )
    return summary, piv


def step2_entry_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    # 1. seniority_final
    s1, _ = kitagawa_decompose(df, "is_entry_final", "seniority_final")
    summaries.append(s1)
    # 2. YOE <= 2 (restricted to rows with YOE known)
    yoe_df = df[df["yoe_known"]].copy()
    s2, _ = kitagawa_decompose(yoe_df, "yoe_le2", "yoe_le2_of_known")
    summaries.append(s2)
    # 3. YOE <= 2 share of ALL (denominator = all rows in archetype)
    df["_yle2_all"] = df["yoe_le2"].astype(bool)
    s3, _ = kitagawa_decompose(df, "_yle2_all", "yoe_le2_of_all")
    summaries.append(s3)
    out = pd.concat(summaries, ignore_index=True)
    out.to_csv(OUT_TABLES / "step2_entry_decomposition_summary.csv", index=False)
    return out


def step2_uniform_rise_check(df: pd.DataFrame) -> pd.DataFrame:
    """Does the uniform within-archetype entry-share rise hold under YOE<=2?"""
    sub = df[df["archetype"] != -2].copy()
    rows = []
    for spec, flag, denom_filter in [
        ("seniority_final", "is_entry_final", None),
        ("yoe_le2_of_all", "yoe_le2", None),
        ("yoe_le2_of_known", "yoe_le2", "yoe_known"),
    ]:
        work = sub if denom_filter is None else sub[sub[denom_filter]]
        grp = work.groupby(["archetype", "archetype_name", "period2"]).agg(
            n=("uid", "count"), entry=(flag, "sum")
        ).reset_index()
        grp["rate"] = grp["entry"] / grp["n"]
        piv = grp.pivot_table(
            index=["archetype", "archetype_name"],
            columns="period2",
            values=["n", "rate"],
            fill_value=0,
        )
        piv.columns = [f"{a}_{b}" for a, b in piv.columns]
        piv = piv.reset_index()
        piv["spec"] = spec
        piv["delta_pp"] = (piv["rate_2026"] - piv["rate_2024"]) * 100
        piv["direction"] = np.sign(piv["delta_pp"]).map({1: "up", -1: "down", 0: "flat"})
        rows.append(piv)
    out = pd.concat(rows, ignore_index=True)
    out = out[
        [
            "spec",
            "archetype",
            "archetype_name",
            "n_2024",
            "n_2026",
            "rate_2024",
            "rate_2026",
            "delta_pp",
            "direction",
        ]
    ].sort_values(["spec", "archetype"])
    out.to_csv(OUT_TABLES / "step2_uniform_rise_check.csv", index=False)

    # Count how many large archetypes (n>=100 in each period) rose
    large = out[(out["n_2024"] >= 100) & (out["n_2026"] >= 100)]
    direction_count = (
        large.groupby(["spec", "direction"]).size().reset_index(name="n_archetypes")
    )
    direction_count.to_csv(OUT_TABLES / "step2_uniform_rise_counts.csv", index=False)
    return out, direction_count


# ---------------------------------------------------------------------------
# Step 3 — Domain-stratified scope / content changes
# ---------------------------------------------------------------------------

def step3_domain_scope(df: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = work.join(metrics)
    # Restrict to labeled text for density metrics; keep full row for binary metrics
    sub = work[work["archetype"] != -2].copy()
    rows = []
    metric_specs = [
        ("requirement_breadth", "has_clean_text", "mean"),
        ("tech_count", None, "mean"),
        ("scope_density", "has_clean_text", "mean"),
        ("any_scope", None, "mean"),
        ("any_ai_narrow", None, "mean"),
        ("any_ai_broad", None, "mean"),
        ("credential_stack_depth", "has_clean_text", "mean"),
        ("any_mentor", None, "mean"),
    ]
    for metric, restriction, agg in metric_specs:
        work_m = sub[sub[restriction]] if restriction else sub
        grp = work_m.groupby(["archetype", "archetype_name", "period2"]).agg(
            n=("uid", "count"), value=(metric, agg)
        ).reset_index()
        piv = grp.pivot_table(
            index=["archetype", "archetype_name"],
            columns="period2",
            values=["n", "value"],
            fill_value=np.nan,
        )
        piv.columns = [f"{a}_{b}" for a, b in piv.columns]
        piv = piv.reset_index()
        piv["delta"] = piv["value_2026"] - piv["value_2024"]
        piv["metric"] = metric
        piv["restriction"] = restriction or "all"
        rows.append(piv)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT_TABLES / "step3_domain_scope_changes.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Step 4 — Junior vs senior content within each archetype
# ---------------------------------------------------------------------------

def step4_junior_senior_within(df: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    work = df.join(metrics).copy()
    # Only rows with entry or mid/senior
    work = work[work["seniority_3level"].isin(["entry", "mid_senior"])]
    work = work[work["archetype"] != -2]

    metric_list = [
        ("requirement_breadth", "has_clean_text"),
        ("tech_count", None),
        ("any_ai_narrow", None),
        ("any_ai_broad", None),
        ("any_scope", None),
        ("scope_density", "has_clean_text"),
        ("any_mentor", None),
        ("credential_stack_depth", "has_clean_text"),
    ]
    rows = []
    for metric, restriction in metric_list:
        sub = work[work[restriction]] if restriction else work
        grp = (
            sub.groupby(["archetype", "archetype_name", "period2", "seniority_3level"])
            .agg(n=("uid", "count"), value=(metric, "mean"))
            .reset_index()
        )
        piv = grp.pivot_table(
            index=["archetype", "archetype_name", "period2"],
            columns="seniority_3level",
            values=["n", "value"],
            fill_value=np.nan,
        )
        piv.columns = [f"{a}_{b}" for a, b in piv.columns]
        piv = piv.reset_index()
        piv["gap"] = piv.get("value_mid_senior", np.nan) - piv.get("value_entry", np.nan)
        piv["metric"] = metric
        rows.append(piv)

    combined = pd.concat(rows, ignore_index=True)
    # Compute 2024->2026 gap change per archetype+metric
    gap_df = combined.pivot_table(
        index=["archetype", "archetype_name", "metric"],
        columns="period2",
        values="gap",
        fill_value=np.nan,
    )
    gap_df.columns = [f"gap_{c}" for c in gap_df.columns]
    gap_df = gap_df.reset_index()
    gap_df["gap_change"] = gap_df.get("gap_2026", np.nan) - gap_df.get("gap_2024", np.nan)
    gap_df.to_csv(OUT_TABLES / "step4_junior_senior_gap.csv", index=False)
    combined.to_csv(OUT_TABLES / "step4_junior_senior_within.csv", index=False)
    return combined, gap_df


# ---------------------------------------------------------------------------
# Step 5 — Senior mentoring shift by archetype x seniority
# ---------------------------------------------------------------------------

def step5_senior_mentoring(df: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    work = df.join(metrics).copy()
    work = work[work["archetype"] != -2]
    work = work[work["seniority_3level"].isin(["entry", "mid_senior"])]
    grp = (
        work.groupby(["archetype", "archetype_name", "period2", "seniority_3level"])
        .agg(n=("uid", "count"), mentor_rate=("any_mentor", "mean"))
        .reset_index()
    )
    piv = grp.pivot_table(
        index=["archetype", "archetype_name", "seniority_3level"],
        columns="period2",
        values=["n", "mentor_rate"],
        fill_value=np.nan,
    )
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    piv["delta_mentor_pp"] = (piv.get("mentor_rate_2026", np.nan) - piv.get("mentor_rate_2024", np.nan)) * 100
    piv.to_csv(OUT_TABLES / "step5_mentoring_by_domain.csv", index=False)
    return piv


# ---------------------------------------------------------------------------
# Step 6 — LLM/GenAI archetype deep dive
# ---------------------------------------------------------------------------

def step6_llm_genai_deepdive(df: pd.DataFrame, metrics: pd.DataFrame) -> dict:
    sub = df.join(metrics).copy()
    # archetype 0 = LLM/GenAI/ML engineering
    llm_df = sub[sub["archetype"] == 0].copy()
    out = {}

    # Top employers by period
    top_emp = (
        llm_df.groupby(["period2", "company_name_canonical"]).size().reset_index(name="n")
    )
    top_emp = top_emp.sort_values(["period2", "n"], ascending=[True, False])
    top_emp_each = top_emp.groupby("period2").head(20)
    top_emp_each.to_csv(OUT_TABLES / "step6_llm_genai_top_employers.csv", index=False)
    out["top_employers"] = top_emp_each

    # Period summary profile
    profile = (
        llm_df.groupby("period2")
        .agg(
            n=("uid", "count"),
            entry_final_rate=("is_entry_final", "mean"),
            yoe_le2_rate=("yoe_le2", "mean"),
            clean_len_mean=("clean_len", "mean"),
            tech_count_mean=("tech_count", "mean"),
            requirement_breadth_mean=("requirement_breadth", "mean"),
            scope_density_mean=("scope_density", "mean"),
            any_ai_narrow_rate=("any_ai_narrow", "mean"),
            any_ai_broad_rate=("any_ai_broad", "mean"),
            any_mentor_rate=("any_mentor", "mean"),
            credential_stack_mean=("credential_stack_depth", "mean"),
            aggregator_rate=("is_aggregator", "mean"),
        )
        .reset_index()
    )
    profile.to_csv(OUT_TABLES / "step6_llm_genai_profile.csv", index=False)
    out["profile"] = profile

    # Entrants vs existing — company set overlap
    c_2024 = set(llm_df[llm_df["period2"] == "2024"]["company_name_canonical"].dropna().unique())
    c_2026 = set(llm_df[llm_df["period2"] == "2026"]["company_name_canonical"].dropna().unique())
    shared = c_2024 & c_2026
    new_2026 = c_2026 - c_2024
    left_2024 = c_2024 - c_2026
    entrance = pd.DataFrame(
        {
            "metric": [
                "companies_2024",
                "companies_2026",
                "shared",
                "new_in_2026",
                "only_in_2024",
                "pct_volume_2026_from_new",
            ],
            "value": [
                len(c_2024),
                len(c_2026),
                len(shared),
                len(new_2026),
                len(left_2024),
                (
                    llm_df[
                        (llm_df["period2"] == "2026")
                        & (llm_df["company_name_canonical"].isin(new_2026))
                    ].shape[0]
                    / max(1, llm_df[llm_df["period2"] == "2026"].shape[0])
                ),
            ],
        }
    )
    entrance.to_csv(OUT_TABLES / "step6_llm_genai_entrants.csv", index=False)
    out["entrance"] = entrance

    # Dominant tech stack 2026
    tech_cols_set = [c for c in sub.columns if c not in df.columns and isinstance(sub[c].dtype, np.dtype)]
    return out


# ---------------------------------------------------------------------------
# Sensitivity: aggregator exclusion for step3 metrics
# ---------------------------------------------------------------------------

def sensitivity_aggregator(df: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    work = df.join(metrics).copy()
    work = work[work["archetype"] != -2]
    rows = []
    for include_agg in [True, False]:
        sub = work if include_agg else work[~work["is_aggregator"].fillna(False)]
        for metric in ["requirement_breadth", "tech_count", "any_ai_broad", "any_mentor", "credential_stack_depth"]:
            grp = sub.groupby("period2")[metric].mean().to_dict()
            rows.append(
                {
                    "include_aggregators": include_agg,
                    "metric": metric,
                    "value_2024": grp.get("2024"),
                    "value_2026": grp.get("2026"),
                    "delta": (grp.get("2026") or 0) - (grp.get("2024") or 0),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "sensitivity_aggregator.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[T28] Loading corpus with archetype labels...")
    df, tech_cols = load_corpus()
    print(f"[T28] Corpus rows: {len(df):,}")
    print(f"[T28] Labeled archetype rows: {(df['archetype'] != -2).sum():,}")
    print(f"[T28] Archetype coverage by period:")
    print(df.groupby("period2")["archetype"].apply(lambda s: (s != -2).mean()).to_string())

    print("\n[T28] Computing per-row text metrics...")
    metrics = compute_text_metrics(df, tech_cols)
    print(f"[T28] Metrics computed for {len(metrics):,} rows.")

    print("\n[T28] Step 1: archetype distribution by period")
    d1 = step1_archetype_distribution(df)
    print(d1[["archetype", "archetype_name", "n_2024", "n_2026", "delta_share_pp"]].to_string())

    print("\n[T28] Step 2: entry-share domain decomposition")
    summary = step2_entry_decomposition(df)
    print(summary.to_string())
    print("\n[T28] Step 2b: uniform-rise check across operationalizations")
    rise, rise_counts = step2_uniform_rise_check(df)
    print(rise_counts.to_string())

    print("\n[T28] Step 3: domain-stratified scope/content changes")
    d3 = step3_domain_scope(df, metrics)
    print(d3[d3["metric"].isin(["requirement_breadth", "any_ai_broad", "any_mentor"])].head(30).to_string())

    print("\n[T28] Step 4: junior vs senior content within archetypes")
    d4, gap_df = step4_junior_senior_within(df, metrics)
    print(gap_df.head(20).to_string())

    print("\n[T28] Step 5: mentoring shift by domain")
    d5 = step5_senior_mentoring(df, metrics)
    print(d5.head(30).to_string())

    print("\n[T28] Step 6: LLM/GenAI deep dive")
    d6 = step6_llm_genai_deepdive(df, metrics)
    print("Profile:")
    print(d6["profile"].to_string())
    print("\nEntrants:")
    print(d6["entrance"].to_string())

    print("\n[T28] Sensitivity: aggregator exclusion")
    sens = sensitivity_aggregator(df, metrics)
    print(sens.to_string())

    # Save metric-annotated subset for figure plotting in separate step
    out_full = df[["uid", "period2", "archetype", "archetype_name", "seniority_3level",
                   "is_entry_final", "yoe_le2", "has_clean_text", "is_aggregator",
                   "company_name_canonical"]].copy()
    for c in metrics.columns:
        out_full[c] = metrics[c].to_numpy()
    out_full.to_parquet(OUT_TABLES / "per_row_metrics.parquet", index=False)
    print(f"\n[T28] Saved per-row metrics parquet. Done.")


if __name__ == "__main__":
    main()
