"""T11 — Requirements complexity & credential stacking.

Builds per-posting feature table with:
  - tech_count from shared swe_tech_matrix.parquet (do NOT recompute)
  - soft_skill_count, org_scope_count, management_STRICT, management_BROAD, ai_count
  - education_level ordinal, yoe_numeric (impute median 3 flag)
  - tech_density, scope_density, credential_stack_depth, requirement_breadth

Primary slice: SWE, LinkedIn, LLM-cleaned text, default filter.
Management deep dive: top 10 per tier, precision sampling for strict + broad, per-term precision,
                     refine strict pattern by dropping <80% precision terms, recompute SNR.

Outputs:
  exploration/tables/T11/T11_metrics_by_slice.csv
  exploration/tables/T11/T11_complexity_distributions.csv
  exploration/tables/T11/T11_management_term_frequency.csv
  exploration/tables/T11/T11_management_precision_sample.csv
  exploration/tables/T11/T11_management_refined_snr.csv
  exploration/tables/T11/T11_top1pct_breadth_sample.csv
  exploration/artifacts/T11/T11_posting_features.parquet
  exploration/figures/T11/*.png
"""
from __future__ import annotations

import json
import re
import random
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNI = ROOT / "data" / "unified.parquet"
TEXT = ROOT / "exploration" / "artifacts" / "shared" / "swe_cleaned_text.parquet"
TECH = ROOT / "exploration" / "artifacts" / "shared" / "swe_tech_matrix.parquet"
SPEC = ROOT / "exploration" / "artifacts" / "shared" / "entry_specialist_employers.csv"
ARCH = ROOT / "exploration" / "artifacts" / "shared" / "swe_archetype_labels.parquet"

TABLES = ROOT / "exploration" / "tables" / "T11"
FIGS = ROOT / "exploration" / "figures" / "T11"
ART = ROOT / "exploration" / "artifacts" / "T11"

for d in [TABLES, FIGS, ART]:
    d.mkdir(parents=True, exist_ok=True)

SLICE_SQL = (
    "source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true"
)

# --- Pattern definitions ---
SOFT_SKILL_PATTERNS = {
    "collaboration": r"\bcollaborat(?:e|ion|ing|ive)\b",
    "communication": r"\bcommunicat(?:e|ion|ing|or|ors)\b",
    "problem_solving": r"\bproblem[- ]solv(?:ing|er|ers)\b",
    "leadership": r"\bleadership\b",
    "teamwork": r"\bteamwork\b|\bteam[- ]player\b",
    "interpersonal": r"\binterpersonal\b",
    "adaptable": r"\badaptab(?:le|ility)\b|\badaptive\b|\bflexible\b",
    "self_motivated": r"\bself[- ]motivat(?:e|ed|ion|ing)\b|\bself[- ]driven\b",
    "ownership": r"\bowner ?ship\b|\btakes? ownership\b",
    "autonomous": r"\bautonom(?:y|ous|ously)\b",
}

SCOPE_PATTERNS = {
    "ownership": r"\bowner ?ship\b|\btakes? ownership\b",
    "end_to_end": r"\bend[- ]to[- ]end\b",
    "cross_functional": r"\bcross[- ]functional\b",
    "stakeholder": r"\bstakeholder",
    "autonomous": r"\bautonom(?:y|ous|ously)\b",
    "initiative": r"\binitiative\b",
    "architect": r"\barchitect(?:ure|ing)?\b",
    "system_design": r"\bsystem design\b",
    "distributed_system": r"\bdistributed system",
    "scalability": r"\bscalab(?:le|ility)\b|\bat scale\b",
}

MANAGEMENT_STRICT = {
    "manage": r"\bmanag(?:e|er|ers|ing|ement)\b",
    "mentor": r"\bmentor(?:s|ing|ship)?\b",
    "coach": r"\bcoach(?:es|ing|ed)?\b",
    "hire": r"\bhir(?:e|es|ing|ed)\b|\brecruit(?:ing|ment|ed)?\b",
    "direct_reports": r"\bdirect reports?\b",
    "performance_review": r"\bperformance reviews?\b|\bperformance appraisals?\b",
    "headcount": r"\bheadcount\b",
    "people_management": r"\bpeople manage(?:ment|r)?\b",
    "team_building": r"\bteam building\b|\bbuild(?:ing)? (?:the|a|your) team\b|\bgrow the team\b",
}
MANAGEMENT_BROAD = {
    **MANAGEMENT_STRICT,
    "lead": r"\blead(?:s|ing|er|ers|ership)?\b",
    "team": r"\bteam(?:s|work)?\b",
    "stakeholder": r"\bstakeholder",
    "coordinate": r"\bcoordinat(?:e|es|ing|ion|or)\b",
}

EDU_PATTERNS = {
    1: r"\bb\.?s\.?\b|\bbachelor(?:'?s)?\b|\bb\.?sc\.?\b|\bundergraduate degree\b|\bba\b",
    2: r"\bm\.?s\.?\b|\bmaster(?:'?s)?\b|\bm\.?sc\.?\b|\bgraduate degree\b",
    3: r"\bph\.?d\b|\bdoctorate\b|\bdoctoral\b",
}

# AI-related tech columns (explicit enumeration). Must appear in swe_tech_matrix.parquet.
AI_TECH_COLS = [
    "langchain", "langgraph", "rag", "vector_database", "pinecone", "chromadb",
    "huggingface", "openai", "claude", "gemini", "mcp", "llamaindex", "anthropic", "ollama",
    "copilot", "cursor", "chatgpt", "codex", "llm_token", "prompt_engineering",
    "fine_tuning", "agent_framework",
    "pytorch", "tensorflow", "sklearn", "keras", "xgboost",
]


def compile_patterns(d: dict[str, str]) -> dict[str, re.Pattern]:
    return {k: re.compile(v, flags=re.IGNORECASE) for k, v in d.items()}


SOFT_CC = compile_patterns(SOFT_SKILL_PATTERNS)
SCOPE_CC = compile_patterns(SCOPE_PATTERNS)
MGMT_STRICT_CC = compile_patterns(MANAGEMENT_STRICT)
MGMT_BROAD_CC = compile_patterns(MANAGEMENT_BROAD)
EDU_CC = {lvl: re.compile(p, flags=re.IGNORECASE) for lvl, p in EDU_PATTERNS.items()}


def count_distinct_matches(text: str, compiled: dict[str, re.Pattern]) -> int:
    if not text:
        return 0
    return sum(1 for pat in compiled.values() if pat.search(text))


def term_hit_matrix(text_ser: pd.Series, compiled: dict[str, re.Pattern]) -> pd.DataFrame:
    """Boolean matrix rows=text, cols=term name."""
    data = {}
    for name, pat in compiled.items():
        data[name] = text_ser.str.contains(pat.pattern, regex=True, flags=re.IGNORECASE, na=False)
    return pd.DataFrame(data, index=text_ser.index)


def highest_education(text: str) -> int:
    if not text:
        return 0
    level = 0
    for lvl, pat in EDU_CC.items():
        if pat.search(text):
            level = max(level, lvl)
    return level


def build_feature_frame() -> pd.DataFrame:
    """Join posting metadata, cleaned text, tech matrix, compute features."""
    con = duckdb.connect()
    con.execute("SET memory_limit='22GB'")

    # Spec-list view
    con.execute(f"CREATE OR REPLACE TEMP VIEW spec AS SELECT lower(company) c FROM read_csv('{SPEC}')")

    # Load metadata
    q_meta = f"""
    SELECT uid, source, period,
           lower(title) AS title_lc,
           seniority_final, seniority_3level,
           is_aggregator,
           company_name_canonical,
           yoe_extracted,
           swe_classification_tier
    FROM read_parquet('{UNI}')
    WHERE {SLICE_SQL}
    """
    meta = con.execute(q_meta).df()
    meta["period_bucket"] = np.where(meta["source"] == "scraped", "2026", "2024")
    meta["source_bucket"] = meta["source"].map(
        {"kaggle_arshkon": "arshkon", "kaggle_asaniczka": "asaniczka", "scraped": "scraped"}
    )
    print(f"Meta rows: {len(meta):,}")

    # Load cleaned text (LLM only)
    q_text = f"""
    SELECT uid, description_cleaned, text_source
    FROM read_parquet('{TEXT}')
    WHERE text_source = 'llm'
    """
    text = con.execute(q_text).df()
    print(f"LLM text rows: {len(text):,}")

    # Load tech matrix
    tech = con.execute(f"SELECT * FROM read_parquet('{TECH}')").df()
    tech_cols = [c for c in tech.columns if c != "uid"]
    tech["tech_count"] = tech[tech_cols].sum(axis=1)
    for c in AI_TECH_COLS:
        if c not in tech.columns:
            raise ValueError(f"Missing AI tech column in matrix: {c}")
    tech["ai_count_tech"] = tech[AI_TECH_COLS].sum(axis=1)
    tech_slim = tech[["uid", "tech_count", "ai_count_tech"]].copy()

    # Merge
    df = meta.merge(text, on="uid", how="inner").merge(tech_slim, on="uid", how="left")
    # Fill missing tech counts (uid present in text but not tech — shouldn't happen, but guard)
    df["tech_count"] = df["tech_count"].fillna(0).astype(int)
    df["ai_count_tech"] = df["ai_count_tech"].fillna(0).astype(int)
    print(f"Joined rows (LLM text only): {len(df):,}")

    # Text-based features
    desc = df["description_cleaned"].fillna("")
    df["desc_len_chars"] = desc.str.len()

    print("Computing soft skill hits...")
    soft_mat = term_hit_matrix(desc, SOFT_CC)
    df["soft_skill_count"] = soft_mat.sum(axis=1)

    print("Computing scope hits...")
    scope_mat = term_hit_matrix(desc, SCOPE_CC)
    df["org_scope_count"] = scope_mat.sum(axis=1)

    print("Computing management hits (strict)...")
    mgmt_s_mat = term_hit_matrix(desc, MGMT_STRICT_CC)
    df["management_STRICT_count"] = mgmt_s_mat.sum(axis=1)
    df["management_STRICT_binary"] = df["management_STRICT_count"] > 0

    print("Computing management hits (broad)...")
    mgmt_b_mat = term_hit_matrix(desc, MGMT_BROAD_CC)
    df["management_BROAD_count"] = mgmt_b_mat.sum(axis=1)
    df["management_BROAD_binary"] = df["management_BROAD_count"] > 0

    print("Computing education level...")
    df["education_level"] = desc.map(highest_education).astype(int)

    # YOE with imputation flag
    df["yoe_imputed"] = df["yoe_extracted"].isna()
    df["yoe_numeric"] = df["yoe_extracted"].fillna(3.0)

    # AI count from tech matrix
    df["ai_count"] = df["ai_count_tech"]

    # Derived complexity
    df["tech_density"] = df["tech_count"] / (df["desc_len_chars"] / 1000.0).replace({0: np.nan})
    df["scope_density"] = df["org_scope_count"] / (df["desc_len_chars"] / 1000.0).replace({0: np.nan})

    # credential_stack_depth: how many CATEGORIES have ≥1 hit?
    df["has_tech"] = df["tech_count"] > 0
    df["has_scope"] = df["org_scope_count"] > 0
    df["has_soft"] = df["soft_skill_count"] > 0
    df["has_education"] = df["education_level"] > 0
    df["has_yoe"] = ~df["yoe_imputed"]
    df["has_mgmt_strict"] = df["management_STRICT_binary"]
    df["has_ai"] = df["ai_count"] > 0

    df["credential_stack_depth"] = (
        df["has_tech"].astype(int)
        + df["has_scope"].astype(int)
        + df["has_soft"].astype(int)
        + df["has_education"].astype(int)
        + df["has_yoe"].astype(int)
        + df["has_mgmt_strict"].astype(int)
        + df["has_ai"].astype(int)
    )

    df["requirement_breadth"] = (
        df["tech_count"]
        + df["org_scope_count"]
        + df["soft_skill_count"]
        + df["management_STRICT_count"]
        + df["ai_count"]
        + (df["education_level"] > 0).astype(int)
        + df["has_yoe"].astype(int)
    )

    # Specialist flag
    spec_set = set(
        con.execute(f"SELECT lower(company) AS c FROM read_csv('{SPEC}')").df()["c"].tolist()
    )
    df["is_specialist_company"] = df["company_name_canonical"].fillna("").str.lower().isin(spec_set)

    # Save management matrices for deep dive
    mgmt_s_mat["uid"] = df["uid"].values
    mgmt_s_mat["source_bucket"] = df["source_bucket"].values
    mgmt_s_mat["period_bucket"] = df["period_bucket"].values
    mgmt_s_mat.to_parquet(ART / "mgmt_strict_term_matrix.parquet")

    mgmt_b_mat["uid"] = df["uid"].values
    mgmt_b_mat["source_bucket"] = df["source_bucket"].values
    mgmt_b_mat["period_bucket"] = df["period_bucket"].values
    mgmt_b_mat.to_parquet(ART / "mgmt_broad_term_matrix.parquet")

    # Persist features
    df.drop(columns=["description_cleaned", "text_source"]).to_parquet(
        ART / "T11_posting_features.parquet", index=False
    )
    print(f"Features written ({len(df):,} rows)")

    return df


# --- Slice definitions ---

def assign_slices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["j1_entry"] = df["seniority_final"] == "entry"
    df["j2_entry_assoc"] = df["seniority_final"].isin(["entry", "associate"])
    df["j3_yoe_leq2"] = (df["yoe_extracted"] <= 2) & df["yoe_extracted"].notna()
    df["j4_j2_or_j3"] = df["j2_entry_assoc"] | df["j3_yoe_leq2"]
    df["s1_senior"] = df["seniority_final"].isin(["mid-senior", "director"])
    df["s4_yoe_geq5"] = (df["yoe_extracted"] >= 5) & df["yoe_extracted"].notna()
    return df


METRIC_COLS = [
    "tech_count",
    "soft_skill_count",
    "org_scope_count",
    "management_STRICT_count",
    "management_BROAD_count",
    "ai_count",
    "education_level",
    "yoe_numeric",
    "tech_density",
    "scope_density",
    "credential_stack_depth",
    "requirement_breadth",
    "desc_len_chars",
]

BINARY_COLS = [
    "management_STRICT_binary",
    "management_BROAD_binary",
    "has_ai",
    "has_scope",
    "has_mgmt_strict",
]


# --- Distribution tables ---

def dist_row(s: pd.Series) -> dict:
    if s.dropna().empty:
        return dict(n=int(s.notna().sum()), mean=None, median=None, p10=None, p90=None, std=None)
    return dict(
        n=int(s.notna().sum()),
        mean=float(s.mean()),
        median=float(s.median()),
        p10=float(np.nanpercentile(s, 10)),
        p90=float(np.nanpercentile(s, 90)),
        std=float(s.std()),
    )


def compile_distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    slices = [
        ("all", lambda d: d),
        ("no_specialist", lambda d: d[~d["is_specialist_company"]]),
        ("j2_entry_assoc", lambda d: d[d["j2_entry_assoc"] & ~d["is_specialist_company"]]),
        ("j3_yoe_leq2", lambda d: d[d["j3_yoe_leq2"] & ~d["is_specialist_company"]]),
        ("j4_j2_or_j3", lambda d: d[d["j4_j2_or_j3"] & ~d["is_specialist_company"]]),
        ("s1_senior", lambda d: d[d["s1_senior"] & ~d["is_specialist_company"]]),
        ("s4_yoe_geq5", lambda d: d[d["s4_yoe_geq5"] & ~d["is_specialist_company"]]),
    ]
    for slc_name, fn in slices:
        sub_total = fn(df)
        for src in ["arshkon", "asaniczka", "scraped"]:
            sub = sub_total[sub_total["source_bucket"] == src]
            if len(sub) == 0:
                continue
            for m in METRIC_COLS:
                r = dist_row(sub[m])
                r.update(dict(slice=slc_name, source_bucket=src, metric=m))
                rows.append(r)
    return pd.DataFrame(rows)


# --- SNR helper (within-2024 vs cross-period) ---

def compute_snr_row(df: pd.DataFrame, slc_mask: pd.Series, metric: str) -> dict:
    sub = df[slc_mask]
    arsh = sub[sub["source_bucket"] == "arshkon"][metric]
    asan = sub[sub["source_bucket"] == "asaniczka"][metric]
    scr = sub[sub["source_bucket"] == "scraped"][metric]
    if arsh.empty or asan.empty or scr.empty:
        return None
    within = arsh.mean() - asan.mean()
    pooled = pd.concat([arsh, asan]).mean()
    cross_a = scr.mean() - arsh.mean()
    cross_p = scr.mean() - pooled
    snr_a = abs(cross_a) / max(abs(within), 1e-9)
    snr_p = abs(cross_p) / max(abs(within), 1e-9)
    return dict(
        metric=metric,
        n_arsh=int(len(arsh)),
        n_asan=int(len(asan)),
        n_scr=int(len(scr)),
        arshkon_mean=float(arsh.mean()),
        asaniczka_mean=float(asan.mean()),
        scraped_mean=float(scr.mean()),
        within_2024=float(within),
        cross_arsh=float(cross_a),
        cross_pooled=float(cross_p),
        snr_arsh=float(snr_a),
        snr_pooled=float(snr_p),
    )


# --- Step 6: management deep dive ---

def management_term_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Per-term prevalence by period, strict + broad."""
    strict = pd.read_parquet(ART / "mgmt_strict_term_matrix.parquet")
    broad = pd.read_parquet(ART / "mgmt_broad_term_matrix.parquet")
    rows = []
    for label, mat, tier in [("strict", strict, "STRICT"), ("broad", broad, "BROAD")]:
        cols = [c for c in mat.columns if c not in ("uid", "source_bucket", "period_bucket")]
        for period in ["2024", "2026"]:
            sub = mat[mat["period_bucket"] == period]
            n = len(sub)
            for c in cols:
                rows.append(
                    dict(
                        tier=tier,
                        term=c,
                        period=period,
                        n_postings=n,
                        hits=int(sub[c].sum()),
                        share=float(sub[c].mean()) if n else 0.0,
                    )
                )
    return pd.DataFrame(rows)


def management_precision_sample(df: pd.DataFrame, n_per_period: int = 25, seed: int = 1337) -> pd.DataFrame:
    """Per-term precision sampling.

    For each term × period, sample up to 5 matches, pull surrounding sentence,
    aggregate into an auditable CSV. Precision judgement is rule-based:
    we classify each snippet as MANAGERIAL_PEOPLE, MANAGERIAL_PROJECT, NON_MANAGERIAL
    using a small keyword rubric. The rubric itself is documented and sampled
    matches are written so a human can override.
    """
    random.seed(seed)
    rng = random.Random(seed)
    con = duckdb.connect()
    text = con.execute(
        f"SELECT uid, description_cleaned FROM read_parquet('{TEXT}') WHERE text_source='llm'"
    ).df()
    text_by_uid = dict(zip(text["uid"], text["description_cleaned"].fillna("")))

    # Rubric keywords to classify precision bucket
    PEOPLE_CUES = [
        r"\bdirect reports?\b",
        r"\bperformance review",
        r"\b1[:-]?1s?\b",
        r"\bone[- ]on[- ]one",
        r"\bhir(e|ing|ed)\b",
        r"\bpeople manag",
        r"\bteam building\b",
        r"\bcareer development\b",
        r"\bmentor",
        r"\bcoach",
        r"\bhead of\b",
        r"\bmanager of\b",
        r"\bteam of \d+",
        r"\breports? to\b",
        r"\borganizational\b",
        r"\bheadcount",
        r"\brecruit",
        r"\bstaffing\b",
        r"\bterminat",
    ]
    PROJECT_CUES = [
        r"\blead (the|a|our|this)? ?(project|initiative|team|effort|charge|design|migration|development)",
        r"\bproject manage",
        r"\bprogram manage",
        r"\bproduct manage",
        r"\blead developer\b",
        r"\blead engineer\b",
        r"\blead role\b",
        r"\bleader\b",
        r"\bmanage(d|s)? the (project|initiative|backlog|product|system|platform|codebase|architecture)",
        r"\bmanage(d|s)? (deliverables|dependencies|priorities|scope|risk)",
        r"\btech ?lead\b",
    ]
    # NOISE_CUES: sentences where "lead", "team", "manage", etc. are used in non-managerial senses
    NOISE_CUES = [
        r"lead[- ]free",
        r"lead time",
        r"market[- ]leader",
        r"industry[- ]leader",
        r"leading (edge|provider|company|firm|brand|supplier)",
        r"product leader",
        r"tech leader",
        r"team member",
        r"team player",
        r"teams? (across|located|based)",
        r"teams? of (customers|clients|users|stakeholders)",
        r"manage(ment)? (stakeholders|expectations|timelines|conflicts|communications|consulting|of data|data|cloud|cost|budget|risk|information|configuration|security|compliance|change|incident|knowledge|content|access)",
        r"manager of (data|systems|applications|software)",
        r"automated management",
        r"configuration management",
        r"content management",
        r"knowledge management",
        r"data management",
        r"asset management",
        r"project management tool",
        r"api management",
        r"service management",
        r"user management",
        r"traffic management",
        r"order management",
        r"case management",
        r"data lead",
        r"mentor(ing)? (role|position)",
        r"stakeholder(s)? (expect|needs|requirements|buy[- ]in|communication)",
    ]
    import re as _re
    _cc = lambda cues: [_re.compile(c, flags=_re.IGNORECASE) for c in cues]
    PC = _cc(PEOPLE_CUES)
    ProC = _cc(PROJECT_CUES)
    NC = _cc(NOISE_CUES)

    SENTENCE_SPLIT = _re.compile(r"(?<=[.!?])\s+|[\n\r]+")

    def classify(snippet: str) -> str:
        snip_lc = snippet.lower()
        people = any(p.search(snip_lc) for p in PC)
        project = any(p.search(snip_lc) for p in ProC)
        noise = any(p.search(snip_lc) for p in NC)
        if people:
            return "MANAGERIAL_PEOPLE"
        if project and not noise:
            return "MANAGERIAL_PROJECT"
        if project and noise:
            return "AMBIGUOUS_PROJECT"
        if noise:
            return "NON_MANAGERIAL_NOISE"
        return "OTHER_AMBIGUOUS"

    def find_snippets(uid: str, term_pat: re.Pattern) -> list[str]:
        txt = text_by_uid.get(uid, "")
        if not txt:
            return []
        out = []
        for sent in SENTENCE_SPLIT.split(txt):
            if term_pat.search(sent):
                out.append(sent.strip())
                if len(out) >= 3:
                    break
        return out

    # Build per-term match lists
    strict_mat = pd.read_parquet(ART / "mgmt_strict_term_matrix.parquet")
    broad_mat = pd.read_parquet(ART / "mgmt_broad_term_matrix.parquet")

    sampled_rows = []
    for tier_label, mat, pats in [
        ("STRICT", strict_mat, MGMT_STRICT_CC),
        ("BROAD", broad_mat, MGMT_BROAD_CC),
    ]:
        for term, pat in pats.items():
            for period in ["2024", "2026"]:
                sub = mat[(mat["period_bucket"] == period) & (mat[term])]
                if len(sub) == 0:
                    continue
                pick = min(len(sub), n_per_period)
                uids = rng.sample(sub["uid"].tolist(), pick)
                for u in uids:
                    snippets = find_snippets(u, pat)
                    for snip in snippets[:1]:  # one snippet per posting (first match)
                        sampled_rows.append(
                            dict(
                                tier=tier_label,
                                term=term,
                                period=period,
                                uid=u,
                                snippet=snip,
                                classification=classify(snip),
                            )
                        )
    return pd.DataFrame(sampled_rows)


def compute_per_term_precision(sample: pd.DataFrame) -> pd.DataFrame:
    """Precision = share of samples classified as MANAGERIAL_PEOPLE (primary) OR
    MANAGERIAL_PROJECT (for strict it counts, for broad it counts)."""
    if sample.empty:
        return pd.DataFrame()
    s = sample.copy()
    s["is_managerial"] = s["classification"].isin(["MANAGERIAL_PEOPLE", "MANAGERIAL_PROJECT"])
    # More stringent: strict should primarily be MANAGERIAL_PEOPLE
    s["is_people_managerial"] = s["classification"] == "MANAGERIAL_PEOPLE"

    by_term = (
        s.groupby(["tier", "term", "period"])
        .agg(
            n=("uid", "size"),
            managerial_any=("is_managerial", "sum"),
            managerial_people=("is_people_managerial", "sum"),
        )
        .reset_index()
    )
    by_term["precision_any"] = by_term["managerial_any"] / by_term["n"]
    by_term["precision_people"] = by_term["managerial_people"] / by_term["n"]
    # Aggregate across periods (total sample)
    overall = (
        s.groupby(["tier", "term"])
        .agg(
            n=("uid", "size"),
            managerial_any=("is_managerial", "sum"),
            managerial_people=("is_people_managerial", "sum"),
        )
        .reset_index()
    )
    overall["precision_any"] = overall["managerial_any"] / overall["n"]
    overall["precision_people"] = overall["managerial_people"] / overall["n"]
    overall["period"] = "all"

    return pd.concat([by_term, overall], ignore_index=True)


def refine_strict_and_snr(df: pd.DataFrame, overall_prec: pd.DataFrame) -> pd.DataFrame:
    """Drop strict terms with precision_any < 0.80 (per spec). Recompute strict binary share,
    compute calibration-style SNR against original table."""
    strict_overall = overall_prec[(overall_prec["tier"] == "STRICT") & (overall_prec["period"] == "all")]
    kept_terms = strict_overall[strict_overall["precision_any"] >= 0.80]["term"].tolist()
    dropped_terms = strict_overall[strict_overall["precision_any"] < 0.80]["term"].tolist()

    strict_mat = pd.read_parquet(ART / "mgmt_strict_term_matrix.parquet")
    if not kept_terms:
        kept_terms = list(MGMT_STRICT_CC.keys())

    refined_hit = strict_mat[kept_terms].any(axis=1)
    strict_mat["refined_strict"] = refined_hit
    # share by source_bucket
    rows = []
    for src in ["arshkon", "asaniczka", "scraped"]:
        sub = strict_mat[strict_mat["source_bucket"] == src]
        rows.append(dict(source_bucket=src, n=len(sub), refined_strict_share=float(sub["refined_strict"].mean())))
    summary = pd.DataFrame(rows)
    # pooled
    pooled_2024 = strict_mat[strict_mat["source_bucket"].isin(["arshkon", "asaniczka"])]["refined_strict"].mean()
    # SNR: within = arshkon - asaniczka; cross = scraped - pooled
    arsh = float(summary[summary["source_bucket"] == "arshkon"]["refined_strict_share"].iloc[0])
    asan = float(summary[summary["source_bucket"] == "asaniczka"]["refined_strict_share"].iloc[0])
    scr = float(summary[summary["source_bucket"] == "scraped"]["refined_strict_share"].iloc[0])
    within = arsh - asan
    cross_p = scr - pooled_2024
    cross_a = scr - arsh
    snr_a = abs(cross_a) / max(abs(within), 1e-9)
    snr_p = abs(cross_p) / max(abs(within), 1e-9)

    summary.attrs["kept_terms"] = kept_terms
    summary.attrs["dropped_terms"] = dropped_terms
    summary_table = pd.DataFrame(
        [
            dict(
                metric="refined_management_STRICT_share",
                arshkon=arsh,
                asaniczka=asan,
                scraped=scr,
                pooled_2024=float(pooled_2024),
                within_2024=float(within),
                cross_arshkon=float(cross_a),
                cross_pooled=float(cross_p),
                snr_arshkon=float(snr_a),
                snr_pooled=float(snr_p),
                kept_terms=";".join(kept_terms),
                dropped_terms=";".join(dropped_terms),
            )
        ]
    )
    return summary_table


# --- Step 8: outlier analysis ---

def top_breadth_outliers(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    thresh = df["requirement_breadth"].quantile(0.99)
    out = df[df["requirement_breadth"] >= thresh].copy()
    con = duckdb.connect()
    text = con.execute(
        f"SELECT uid, description_cleaned[1:800] AS snippet FROM read_parquet('{TEXT}')"
    ).df()
    out = out.merge(text, on="uid", how="left")
    sample = out.sort_values("requirement_breadth", ascending=False).head(n)
    return sample[[
        "uid", "source_bucket", "period_bucket", "title_lc",
        "requirement_breadth", "tech_count", "soft_skill_count",
        "org_scope_count", "management_STRICT_count", "ai_count", "desc_len_chars",
        "snippet",
    ]]


def save_distribution_plots(df: pd.DataFrame):
    """Histograms of credential_stack_depth and requirement_breadth."""
    for col in ["credential_stack_depth", "requirement_breadth", "tech_count", "org_scope_count"]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for src, color in [("arshkon", "#1f77b4"), ("asaniczka", "#2ca02c"), ("scraped", "#d62728")]:
            vals = df[df["source_bucket"] == src][col].dropna()
            if vals.empty:
                continue
            ax.hist(vals, bins=40, density=True, alpha=0.45, color=color, label=f"{src} (n={len(vals):,})")
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("density")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / f"T11_dist_{col}.png", dpi=120)
        plt.close(fig)


def main():
    df = build_feature_frame()
    df = assign_slices(df)

    print("Computing distribution tables...")
    dist = compile_distribution_table(df)
    dist.to_csv(TABLES / "T11_complexity_distributions.csv", index=False)

    print("Computing SNR per metric × slice...")
    snr_rows = []
    slice_defs = [
        ("all_no_spec", ~df["is_specialist_company"]),
        ("j2_no_spec", df["j2_entry_assoc"] & ~df["is_specialist_company"]),
        ("j3_no_spec", df["j3_yoe_leq2"] & ~df["is_specialist_company"]),
        ("s1_no_spec", df["s1_senior"] & ~df["is_specialist_company"]),
        ("s4_no_spec", df["s4_yoe_geq5"] & ~df["is_specialist_company"]),
    ]
    for slc_name, mask in slice_defs:
        for m in METRIC_COLS:
            r = compute_snr_row(df, mask, m)
            if r:
                r["slice"] = slc_name
                snr_rows.append(r)
    snr_df = pd.DataFrame(snr_rows)
    snr_df.to_csv(TABLES / "T11_metrics_snr_by_slice.csv", index=False)

    # Binary SNR (for management share)
    bin_rows = []
    for slc_name, mask in slice_defs:
        for m in BINARY_COLS:
            r = compute_snr_row(df.assign(**{m: df[m].astype(float)}), mask, m)
            if r:
                r["slice"] = slc_name
                bin_rows.append(r)
    bin_df = pd.DataFrame(bin_rows)
    bin_df.to_csv(TABLES / "T11_binary_snr_by_slice.csv", index=False)

    print("Management term frequency...")
    term_freq = management_term_frequency(df)
    term_freq.to_csv(TABLES / "T11_management_term_frequency.csv", index=False)

    print("Management precision sampling...")
    sample = management_precision_sample(df, n_per_period=25)
    sample.to_csv(TABLES / "T11_management_precision_sample.csv", index=False)

    per_term = compute_per_term_precision(sample)
    per_term.to_csv(TABLES / "T11_management_per_term_precision.csv", index=False)

    print("Refining strict pattern & recomputing SNR...")
    refined = refine_strict_and_snr(df, per_term)
    refined.to_csv(TABLES / "T11_management_refined_snr.csv", index=False)

    print("Outlier top 1% breadth sample...")
    outliers = top_breadth_outliers(df, n=20)
    outliers.to_csv(TABLES / "T11_top1pct_breadth_sample.csv", index=False)

    print("Plots...")
    save_distribution_plots(df[~df["is_specialist_company"]])

    # Cross-period means headline table for J2/S1
    headline = []
    for slc_name, mask in [
        ("j2_no_spec", df["j2_entry_assoc"] & ~df["is_specialist_company"]),
        ("j3_no_spec", df["j3_yoe_leq2"] & ~df["is_specialist_company"]),
        ("s1_no_spec", df["s1_senior"] & ~df["is_specialist_company"]),
        ("s4_no_spec", df["s4_yoe_geq5"] & ~df["is_specialist_company"]),
        ("all_no_spec", ~df["is_specialist_company"]),
    ]:
        sub = df[mask]
        for src in ["arshkon", "asaniczka", "scraped"]:
            s = sub[sub["source_bucket"] == src]
            if s.empty:
                continue
            for m in ["credential_stack_depth", "requirement_breadth", "tech_count", "org_scope_count", "ai_count", "education_level", "yoe_numeric", "desc_len_chars"]:
                headline.append(
                    dict(
                        slice=slc_name,
                        source_bucket=src,
                        metric=m,
                        n=len(s),
                        mean=float(s[m].mean()),
                        median=float(s[m].median()),
                    )
                )
    pd.DataFrame(headline).to_csv(TABLES / "T11_headline_means.csv", index=False)

    # Archetype stratification (if available)
    arch_info = {"available": False}
    if ARCH.exists():
        con = duckdb.connect()
        arch = con.execute(f"SELECT * FROM read_parquet('{ARCH}')").df()
        arch_info["available"] = True
        df_arch = df.merge(arch, on="uid", how="left")
        # Expect a column named 'archetype' — guard it
        arch_col = None
        for candidate in ["archetype", "archetype_label", "domain", "domain_archetype"]:
            if candidate in df_arch.columns:
                arch_col = candidate
                break
        if arch_col:
            arch_info["column_used"] = arch_col
            agg = (
                df_arch[~df_arch["is_specialist_company"]]
                .groupby([arch_col, "source_bucket"])
                .agg(
                    n=("uid", "size"),
                    scope_density_mean=("scope_density", "mean"),
                    org_scope_count_mean=("org_scope_count", "mean"),
                    credential_stack_depth_mean=("credential_stack_depth", "mean"),
                    requirement_breadth_mean=("requirement_breadth", "mean"),
                )
                .reset_index()
            )
            agg.to_csv(TABLES / "T11_archetype_complexity.csv", index=False)
    else:
        arch_info["note"] = "archetype artifact not yet present — deferred to T28 Wave 3"

    with open(ART / "T11_summary.json", "w") as fp:
        json.dump({"arch": arch_info, "n_rows": len(df)}, fp, indent=2, default=str)
    print("T11 done.")


if __name__ == "__main__":
    main()
