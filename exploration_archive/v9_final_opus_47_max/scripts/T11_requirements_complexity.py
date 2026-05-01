"""T11. Requirements complexity & credential stacking.

Builds per-posting features:
  - tech_count (from swe_tech_matrix.parquet)
  - soft_skill_count, scope_count, mgmt_strong_count, mgmt_broad_count,
    ai_binary, education_level, yoe
  - composites: requirement_breadth, credential_stack_depth
  - densities: tech_density, scope_density, mgmt_*_density
  - residualized variants: requirement_breadth_resid, credential_stack_depth_resid

Outputs:
  - exploration/artifacts/shared/T11_posting_features.parquet  (Wave 2+ input)
  - exploration/tables/T11/* (distribution comparisons, top-10 mgmt terms,
    precision-validated patterns, outlier analysis)
  - exploration/reports/T11.md (written separately)

Scope: SWE LinkedIn English date_flag='ok'.
Text source: description_core_llm where labeled else raw description (reported).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
SHARED = ROOT / "exploration" / "artifacts" / "shared"
OUT_TAB = ROOT / "exploration" / "tables" / "T11"
OUT_TAB.mkdir(parents=True, exist_ok=True)
OUT_FIG = ROOT / "exploration" / "figures" / "T11"
OUT_FIG.mkdir(parents=True, exist_ok=True)

TECH_MATRIX = SHARED / "swe_tech_matrix.parquet"
SENIORITY_PANEL = SHARED / "seniority_definition_panel.csv"
ENTRY_SPECIALIST = SHARED / "entry_specialist_employers.csv"
COMPANY_STOPLIST = SHARED / "company_stoplist.txt"

FILTER = (
    "is_swe=TRUE AND source_platform='linkedin' "
    "AND is_english=TRUE AND date_flag='ok'"
)


# ---------------------------- pattern definitions -----------------------------


SOFT_SKILL_TERMS = [
    (
        "communication",
        r"\b(?:communicat(?:e|es|ed|ing|ion|ions)|verbal communication|written communication|presentation skills|public speaking)\b",
    ),
    (
        "collaboration",
        r"\b(?:collaborat(?:e|es|ed|ing|ion)|team[- ]?work|teamwork|work(?:ing)? with others|partner(?:s|ship)?)\b",
    ),
    (
        "problem_solving",
        r"\b(?:problem[- ]?solving|problem solver|analytical thinking|critical thinking|analytical skills|troubleshoot(?:ing)?)\b",
    ),
    (
        "leadership",
        r"\b(?:leadership|influence (?:skills|others)|influencing|lead(?:ing)? (?:teams|others))\b",
    ),
    (
        "adaptability",
        r"\b(?:adaptab(?:le|ility)|flexib(?:le|ility)|thrive(?:s)? in ambiguity|comfortable with change|thrive in change)\b",
    ),
    (
        "learning",
        r"\b(?:fast learner|quick learner|eager to learn|continuous learning|self[- ]?starter|self[- ]?motivated|growth mindset)\b",
    ),
    (
        "detail_oriented",
        r"\b(?:detail[- ]?oriented|attention to detail|meticulous|thorough)\b",
    ),
    (
        "time_management",
        r"\b(?:time management|prioriti[sz](?:e|ing|ation)|meet deadlines|manage(?:s|d|ing)? deadlines|multi[- ]?task)\b",
    ),
    (
        "curiosity",
        r"\b(?:curiosity|intellectual(?:ly)? curious|intellectual curiosity|passion for learning)\b",
    ),
    (
        "empathy",
        r"\b(?:empath(?:y|etic)|customer empathy|user empathy|customer focus(?:ed)?|customer[- ]?centric)\b",
    ),
]

SCOPE_TERMS = [
    ("ownership", r"\b(?:ownership|own(?:ing)? (?:the|a|your|features|outcomes|projects|systems|product)|take ownership)\b"),
    ("end_to_end", r"\b(?:end[- ]?to[- ]?end|end2end|full(?:[- ]?stack)? ownership|full lifecycle|soup to nuts)\b"),
    ("cross_functional", r"\b(?:cross[- ]?functional|x[- ]?functional|cross[- ]?team|cross[- ]?org|across teams)\b"),
    ("stakeholder", r"\b(?:stakeholder(?:s)?|business partner(?:s)?)\b"),
    ("autonomous", r"\b(?:autonom(?:y|ous(?:ly)?)|self[- ]?direct(?:ed|ion)|independent(?:ly)? (?:driv|work|deliver))\b"),
    ("initiative", r"\b(?:take(?:s)? initiative|proactive(?:ly)?|drive(?:s)? (?:change|initiatives|forward)|self[- ]?start(?:er|ing))\b"),
    ("scope", r"\b(?:define (?:scope|problems|solutions)|scope (?:work|projects)|shape (?:the )?roadmap|roadmap ownership)\b"),
    ("strategy", r"\b(?:strategic|technical strategy|strategy and execution|long[- ]?term vision|vision and strategy)\b"),
]

# Management indicators — strict and broad tiers.
MGMT_STRONG_TERMS = [
    ("manage_people", r"\b(?:manag(?:e|es|ed|ing) (?:people|engineers|a team|teams|team members|the team))\b"),
    ("direct_reports", r"\b(?:direct reports?|reports? (?:into|to) (?:you|me)|have (?:\d+|several|multiple) (?:direct )?reports?)\b"),
    ("mentor", r"\b(?:mentor(?:s|ed|ing|ship)?|coach(?:es|ed|ing)|grow(?:ing)? (?:engineers|talent|the team))\b"),
    ("hire", r"\b(?:hir(?:e|es|ed|ing) (?:engineers|people|the team|talent)|recruit(?:s|ed|ing)? (?:engineers|talent|for the team)|build(?:s|ing)? (?:the|a|your) team|headcount)\b"),
    ("performance_review", r"\b(?:performance review(?:s)?|performance management|performance evaluat(?:ion|e|ions)|career development|1[- ]?on[- ]?1s|1:1s|1on1s)\b"),
    ("eng_manager", r"\b(?:engineering manager|eng(?:ineering)? manag(?:er|ement)|em role|people manager|line manager)\b"),
]

MGMT_BROAD_EXTRA = [
    ("lead", r"\b(?:lead(?:s|ing)? (?:a|the|our|development|design|efforts|initiatives|projects|cross[- ]?functional))\b"),
    ("team", r"\b(?:collaborat(?:e|ive|ion) (?:within|across) (?:the |a )?team|work(?:ing)? within (?:a|the) team)\b"),
    ("stakeholder", r"\b(?:stakeholder(?:s)?|partner with (?:product|design|business)|business partner(?:s)?)\b"),
    ("coordinate", r"\b(?:coordinat(?:e|ed|ing|ion)|align(?:s|ment) (?:with|across))\b"),
]

# AI-specific requirement pattern — V1-style strict
AI_STRICT = re.compile(
    r"\b(?:copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|vector database|pinecone|huggingface|hugging face)\b",
    flags=re.IGNORECASE,
)

# Education level
EDU_LEVELS = [
    ("phd", 3, re.compile(r"\b(?:ph\.?d\.?|doctorate|doctoral)\b", re.I)),
    ("ms", 2, re.compile(r"\b(?:m\.?s\.?|m\.?sc\.?|master'?s?|ms degree|masters? degree)\b", re.I)),
    ("bs", 1, re.compile(
        r"\b(?:b\.?s\.?|b\.?sc\.?|bachelor'?s?|bs degree|undergraduate degree|4[- ]year degree|bachelor'?s? degree|b\.?a\.?)\b",
        re.I,
    )),
]


# ---------------------------- assertions --------------------------------------


def _assert_patterns():
    # Soft skills
    terms = dict(SOFT_SKILL_TERMS)
    assert re.search(terms["communication"], "strong verbal communication skills")
    assert re.search(terms["collaboration"], "ability to collaborate with peers")
    assert re.search(terms["problem_solving"], "analytical thinking and problem-solving")
    assert re.search(terms["leadership"], "leadership and influence")
    assert re.search(terms["learning"], "self-starter and quick learner")
    assert re.search(terms["detail_oriented"], "attention to detail")
    assert re.search(terms["time_management"], "ability to prioritize")
    assert re.search(terms["empathy"], "customer empathy")
    assert not re.search(terms["communication"], "python java cloud")
    # Scope
    scope = dict(SCOPE_TERMS)
    assert re.search(scope["end_to_end"], "end-to-end ownership")
    assert re.search(scope["cross_functional"], "cross-functional collaboration")
    assert re.search(scope["stakeholder"], "partner with stakeholders")
    assert re.search(scope["autonomous"], "work autonomously")
    assert re.search(scope["initiative"], "takes initiative")
    assert re.search(scope["ownership"], "take ownership of features")
    assert not re.search(scope["end_to_end"], "software engineer")
    # Management — strong
    mgmt = dict(MGMT_STRONG_TERMS)
    assert re.search(mgmt["manage_people"], "you will manage a team of engineers")
    assert re.search(mgmt["direct_reports"], "you will have 5 direct reports")
    assert re.search(mgmt["mentor"], "mentor junior engineers")
    assert re.search(mgmt["hire"], "hire engineers for the team")
    assert re.search(mgmt["performance_review"], "conduct 1-on-1s with reports")
    assert re.search(mgmt["eng_manager"], "engineering manager position")
    assert not re.search(mgmt["mentor"], "python java cloud")
    assert not re.search(mgmt["manage_people"], "manage dependencies")
    # Broad extras
    broad = dict(MGMT_BROAD_EXTRA)
    assert re.search(broad["lead"], "lead a team of engineers")
    assert re.search(broad["stakeholder"], "partner with product and design")
    assert re.search(broad["coordinate"], "coordinate with cross-functional teams")
    # AI strict
    assert AI_STRICT.search("experience with copilot")
    assert AI_STRICT.search("familiarity with langchain")
    assert AI_STRICT.search("building rag systems")
    assert AI_STRICT.search("using pinecone")
    assert AI_STRICT.search("fine-tuning llms")
    assert AI_STRICT.search("experience with gpt-4")
    assert not AI_STRICT.search("python and java experience")
    # Education
    for n, lvl, p in EDU_LEVELS:
        pass
    assert EDU_LEVELS[0][2].search("Ph.D. in computer science")
    assert EDU_LEVELS[0][2].search("PhD in CS")
    assert EDU_LEVELS[1][2].search("Master's degree in computer science")
    assert EDU_LEVELS[2][2].search("Bachelor's degree in CS")
    assert EDU_LEVELS[2][2].search("BS in computer science")


_assert_patterns()


# ---------------------------- feature extraction ------------------------------


def count_matches(text: str, pats: list[tuple[str, str]]) -> tuple[int, list[str]]:
    if text is None or not text:
        return 0, []
    hits = []
    for name, pat in pats:
        if re.search(pat, text, flags=re.IGNORECASE):
            hits.append(name)
    return len(hits), hits


def education_level(text: str) -> int:
    if text is None or not text:
        return 0
    for _, lvl, p in EDU_LEVELS:
        if p.search(text):
            return lvl
    return 0


def ai_binary_fn(text: str) -> bool:
    if text is None or not text:
        return False
    return AI_STRICT.search(text) is not None


def load_corpus() -> pd.DataFrame:
    con = duckdb.connect()
    df = con.sql(
        f"""
        SELECT uid, source, period, title, description_core_llm, description,
               description_length,
               company_name_canonical, is_aggregator,
               seniority_final, seniority_final_source, seniority_3level,
               yoe_min_years_llm, yoe_extracted,
               llm_extraction_coverage, llm_classification_coverage
        FROM '{DATA}'
        WHERE {FILTER}
        """
    ).df()
    return df


def build_features(df: pd.DataFrame, tech: pd.DataFrame) -> pd.DataFrame:
    # text source selection
    def pick_text(row):
        if row["llm_extraction_coverage"] == "labeled" and pd.notna(
            row["description_core_llm"]
        ):
            return row["description_core_llm"]
        return row["description"] or ""

    df = df.copy()
    df["text"] = df.apply(pick_text, axis=1)
    df["text_source"] = df["llm_extraction_coverage"].map(
        lambda v: "llm" if v == "labeled" else "raw"
    )
    df["description_cleaned_length"] = df["text"].str.len().fillna(0).astype("int64")

    # Tech count from tech matrix
    tech_cols = [c for c in tech.columns if c != "uid"]
    tech["tech_count"] = tech[tech_cols].sum(axis=1).astype("int64")
    df = df.merge(tech[["uid", "tech_count"]], on="uid", how="left")
    df["tech_count"] = df["tech_count"].fillna(0).astype("int64")

    # Soft skill, scope, mgmt, AI counts
    soft_counts, scope_counts, mgmt_strong_counts, mgmt_broad_counts = [], [], [], []
    edu_levels, ai_binaries = [], []
    soft_hits, scope_hits, mgmt_strong_hits, mgmt_broad_extra_hits = [], [], [], []
    for t in df["text"].tolist():
        sc, sh = count_matches(t, SOFT_SKILL_TERMS)
        soft_counts.append(sc)
        soft_hits.append(sh)
        scc, sch = count_matches(t, SCOPE_TERMS)
        scope_counts.append(scc)
        scope_hits.append(sch)
        msc, msh = count_matches(t, MGMT_STRONG_TERMS)
        mgmt_strong_counts.append(msc)
        mgmt_strong_hits.append(msh)
        mbc, mbh = count_matches(t, MGMT_BROAD_EXTRA)
        mgmt_broad_counts.append(msc + mbc)
        mgmt_broad_extra_hits.append(mbh)
        edu_levels.append(education_level(t))
        ai_binaries.append(ai_binary_fn(t))

    df["soft_skill_count"] = soft_counts
    df["scope_count"] = scope_counts
    df["mgmt_strong_count"] = mgmt_strong_counts
    df["mgmt_broad_count"] = mgmt_broad_counts
    df["soft_hits"] = soft_hits
    df["scope_hits"] = scope_hits
    df["mgmt_strong_hits"] = mgmt_strong_hits
    df["mgmt_broad_extra_hits"] = mgmt_broad_extra_hits
    df["education_level"] = edu_levels
    df["ai_binary"] = ai_binaries

    # Composites
    df["requirement_breadth"] = (
        df["tech_count"]
        + df["soft_skill_count"]
        + df["scope_count"]
        + df["mgmt_broad_count"]
        + df["ai_binary"].astype(int)
        + (df["education_level"] > 0).astype(int)
        + df["yoe_min_years_llm"].notna().astype(int)
    ).astype("int64")

    # credential stack depth: number of distinct requirement CATEGORIES with at least one mention
    # Categories (max 7): tech, education, YOE, soft, scope, mgmt, AI
    df["credential_stack_depth"] = (
        (df["tech_count"] > 0).astype(int)
        + (df["education_level"] > 0).astype(int)
        + df["yoe_min_years_llm"].notna().astype(int)
        + (df["soft_skill_count"] > 0).astype(int)
        + (df["scope_count"] > 0).astype(int)
        + (df["mgmt_broad_count"] > 0).astype(int)
        + df["ai_binary"].astype(int)
    ).astype("int64")

    # Per-1K-char densities
    safe_len = df["description_cleaned_length"].clip(lower=1)
    df["tech_density"] = df["tech_count"] / safe_len * 1000
    df["scope_density"] = df["scope_count"] / safe_len * 1000
    df["mgmt_strong_density"] = df["mgmt_strong_count"] / safe_len * 1000
    df["mgmt_broad_density"] = df["mgmt_broad_count"] / safe_len * 1000
    df["soft_density"] = df["soft_skill_count"] / safe_len * 1000
    return df


# ---------------------------- length residualization -------------------------


def residualize_length(df: pd.DataFrame, col: str, fit_mask: pd.Series = None) -> pd.Series:
    """Fit y = b0 + b1 * log(length) on fit_mask rows; return residual y - y_hat on all rows."""
    x = np.log(df["description_cleaned_length"].clip(lower=1).values)
    y = df[col].astype(float).values
    mask = fit_mask.values if fit_mask is not None else np.ones(len(df), dtype=bool)
    xm = x[mask]
    ym = y[mask]
    b1, b0 = np.polyfit(xm, ym, 1)
    yhat = b0 + b1 * x
    return pd.Series(y - yhat, index=df.index)


def residualize_all(df: pd.DataFrame) -> pd.DataFrame:
    # Fit on the whole SWE LinkedIn frame; single global regression for simplicity and reproducibility.
    for col in ["requirement_breadth", "credential_stack_depth", "scope_density", "tech_density", "mgmt_broad_density"]:
        df[f"{col}_resid"] = residualize_length(df, col)
    return df


# ---------------------------- period / seniority labels ----------------------


def period_bucket(row) -> str:
    if row["source"] == "kaggle_arshkon":
        return "arshkon_2024"
    if row["source"] == "kaggle_asaniczka":
        return "asaniczka_2024"
    if row["source"] == "scraped":
        return "scraped_2026"
    return "other"


def yoe_seniority(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lab = df["llm_extraction_coverage"] == "labeled"
    # J3 / S4 / J4 / S5 YOE labels
    j3 = lab & (df["yoe_min_years_llm"] <= 2)
    j4 = lab & (df["yoe_min_years_llm"] <= 3)
    s4 = lab & (df["yoe_min_years_llm"] >= 5)
    s5 = lab & (df["yoe_min_years_llm"] >= 7)
    df["J3"] = j3
    df["J4"] = j4
    df["S4"] = s4
    df["S5"] = s5
    # rule YOE (ablation for non-labeled)
    j3r = df["yoe_extracted"].notna() & (df["yoe_extracted"] <= 2)
    s4r = df["yoe_extracted"].notna() & (df["yoe_extracted"] >= 5)
    df["J3_rule"] = j3r
    df["S4_rule"] = s4r
    return df


# ---------------------------- steps -------------------------------------------


def step_distribution_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Complexity metrics by period × J3/S4 (primary) plus J1/J2/J4/S1/S3/S5 sensitivity."""
    df = df.copy()
    df["bucket"] = df.apply(period_bucket, axis=1)
    rows = []
    metrics = [
        "tech_count",
        "soft_skill_count",
        "scope_count",
        "mgmt_strong_count",
        "mgmt_broad_count",
        "ai_binary",
        "education_level",
        "requirement_breadth",
        "credential_stack_depth",
        "tech_density",
        "scope_density",
        "mgmt_strong_density",
        "mgmt_broad_density",
        "requirement_breadth_resid",
        "credential_stack_depth_resid",
        "scope_density_resid",
        "tech_density_resid",
        "mgmt_broad_density_resid",
        "description_cleaned_length",
    ]
    periods = {
        "arshkon_2024": df[df["bucket"] == "arshkon_2024"],
        "asaniczka_2024": df[df["bucket"] == "asaniczka_2024"],
        "pooled_2024": df[df["bucket"].isin(["arshkon_2024", "asaniczka_2024"])],
        "scraped_2026": df[df["bucket"] == "scraped_2026"],
    }
    seniority_cuts = {
        "all": lambda d: d,
        "J3": lambda d: d[d["J3"]],
        "J4": lambda d: d[d["J4"]],
        "S4": lambda d: d[d["S4"]],
        "S5": lambda d: d[d["S5"]],
        "J3_rule": lambda d: d[d["J3_rule"]],
        "S4_rule": lambda d: d[d["S4_rule"]],
        # Native labels for J1/J2, S1/S3 (from seniority_3level)
        "J1_label": lambda d: d[d["seniority_3level"] == "entry"],
        "S1_label": lambda d: d[d["seniority_3level"] == "senior"],
        # seniority_final senior (merged mid_senior) — not the strict J1
        "entry_final": lambda d: d[d["seniority_final"] == "entry"],
        "senior_final": lambda d: d[d["seniority_final"].isin(["senior", "mid_senior"])],
    }
    for pname, pd_df in periods.items():
        for sname, fn in seniority_cuts.items():
            sub = fn(pd_df)
            n = len(sub)
            if n == 0:
                continue
            rec = dict(period=pname, seniority_cut=sname, n=n)
            for m in metrics:
                if m not in sub.columns:
                    continue
                vals = sub[m].dropna()
                if vals.empty:
                    continue
                if m == "ai_binary":
                    rec[f"{m}_mean"] = round(float(vals.mean()), 4)
                else:
                    rec[f"{m}_mean"] = round(float(vals.mean()), 4)
                    rec[f"{m}_median"] = round(float(vals.median()), 4)
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TAB / "distribution_panel.csv", index=False)
    return out


def step_credential_stacking(df: pd.DataFrame) -> pd.DataFrame:
    """Share of postings with K+ requirement categories present, by period × seniority."""
    df = df.copy()
    df["bucket"] = df.apply(period_bucket, axis=1)
    rows = []
    periods = {
        "pooled_2024": df[df["bucket"].isin(["arshkon_2024", "asaniczka_2024"])],
        "arshkon_2024": df[df["bucket"] == "arshkon_2024"],
        "scraped_2026": df[df["bucket"] == "scraped_2026"],
    }
    seniority_cuts = {
        "all": lambda d: d,
        "J3": lambda d: d[d["J3"]],
        "J4": lambda d: d[d["J4"]],
        "S4": lambda d: d[d["S4"]],
        "S5": lambda d: d[d["S5"]],
    }
    for pname, pd_df in periods.items():
        for sname, fn in seniority_cuts.items():
            sub = fn(pd_df)
            n = len(sub)
            if n == 0:
                continue
            rec = dict(period=pname, seniority_cut=sname, n=n)
            for k in range(1, 8):
                rec[f"share_depth_ge_{k}"] = round(float((sub["credential_stack_depth"] >= k).mean()), 4)
            rec["mean_depth"] = round(float(sub["credential_stack_depth"].mean()), 4)
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TAB / "credential_stack_depth_panel.csv", index=False)
    return out


def step_entry_level_only(df: pd.DataFrame) -> pd.DataFrame:
    """Entry-level specific comparison 2024 vs 2026 under J3 primary + J1/J2/J4 sensitivities.

    Reports requirement_breadth + credential_stack_depth (raw + resid), tech_count, ai_binary.
    """
    df = df.copy()
    df["bucket"] = df.apply(period_bucket, axis=1)
    rows = []
    cuts = {
        "J3_pooled_2024": df[df["bucket"].isin(["arshkon_2024", "asaniczka_2024"]) & df["J3"]],
        "J3_arshkon": df[(df["bucket"] == "arshkon_2024") & df["J3"]],
        "J3_scraped_2026": df[(df["bucket"] == "scraped_2026") & df["J3"]],
        "J4_pooled_2024": df[df["bucket"].isin(["arshkon_2024", "asaniczka_2024"]) & df["J4"]],
        "J4_scraped_2026": df[(df["bucket"] == "scraped_2026") & df["J4"]],
        # Native entry labels
        "J1_label_pooled_2024": df[
            df["bucket"].isin(["arshkon_2024", "asaniczka_2024"])
            & (df["seniority_3level"] == "entry")
        ],
        "J1_label_scraped_2026": df[
            (df["bucket"] == "scraped_2026") & (df["seniority_3level"] == "entry")
        ],
    }
    metrics = [
        "requirement_breadth",
        "requirement_breadth_resid",
        "credential_stack_depth",
        "credential_stack_depth_resid",
        "tech_count",
        "tech_density",
        "tech_density_resid",
        "scope_count",
        "scope_density",
        "scope_density_resid",
        "soft_skill_count",
        "mgmt_strong_count",
        "mgmt_broad_count",
        "ai_binary",
        "education_level",
        "description_cleaned_length",
    ]
    for name, sub in cuts.items():
        if len(sub) == 0:
            continue
        rec = dict(cut=name, n=len(sub))
        for m in metrics:
            vals = sub[m].dropna()
            if vals.empty:
                continue
            rec[f"{m}_mean"] = round(float(vals.mean()), 4)
            rec[f"{m}_median"] = round(float(vals.median()), 4)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TAB / "entry_level_comparison.csv", index=False)
    return out


def step_correlation_check(df: pd.DataFrame) -> pd.DataFrame:
    """Report correlation of breadth + credential_stack_depth + densities with description_length."""
    df = df.copy()
    df["bucket"] = df.apply(period_bucket, axis=1)
    rows = []
    for pname, pd_df in [
        ("pooled_2024", df[df["bucket"].isin(["arshkon_2024", "asaniczka_2024"])]),
        ("scraped_2026", df[df["bucket"] == "scraped_2026"]),
        ("all", df),
    ]:
        for m in ["requirement_breadth", "credential_stack_depth", "tech_count",
                  "soft_skill_count", "scope_count", "mgmt_broad_count"]:
            x = pd_df["description_cleaned_length"].astype(float)
            y = pd_df[m].astype(float)
            r = float(x.corr(y))
            rows.append(dict(period=pname, metric=m, pearson_r=round(r, 4)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TAB / "length_correlation_check.csv", index=False)
    return out


def step_mgmt_term_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Top 10 specific terms triggering mgmt indicator per period, strict vs broad."""
    df = df.copy()
    df["bucket"] = df.apply(period_bucket, axis=1)
    rows = []
    for pname, pd_df in [
        ("arshkon_2024", df[df["bucket"] == "arshkon_2024"]),
        ("asaniczka_2024", df[df["bucket"] == "asaniczka_2024"]),
        ("pooled_2024", df[df["bucket"].isin(["arshkon_2024", "asaniczka_2024"])]),
        ("scraped_2026", df[df["bucket"] == "scraped_2026"]),
    ]:
        n = len(pd_df)
        if n == 0:
            continue
        # Count strict-tier term hits
        strict_counter = {}
        for hits in pd_df["mgmt_strong_hits"]:
            for h in hits:
                strict_counter[h] = strict_counter.get(h, 0) + 1
        for term, c in sorted(strict_counter.items(), key=lambda x: -x[1])[:10]:
            rows.append(
                dict(
                    period=pname,
                    tier="strong",
                    term=term,
                    n_postings=c,
                    share_of_postings=round(c / n, 4),
                )
            )
        broad_counter = {}
        for hits in pd_df["mgmt_broad_extra_hits"]:
            for h in hits:
                broad_counter[h] = broad_counter.get(h, 0) + 1
        for term, c in sorted(broad_counter.items(), key=lambda x: -x[1])[:10]:
            rows.append(
                dict(
                    period=pname,
                    tier="broad_extra",
                    term=term,
                    n_postings=c,
                    share_of_postings=round(c / n, 4),
                )
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TAB / "mgmt_term_top10.csv", index=False)
    return out


def step_mgmt_precision_sample(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Stratified semantic sample: 25 per period × tier, flag sentences around matches for manual-like heuristic precision.

    Since there is no human-in-the-loop in this pipeline, we apply a secondary
    "semantic context" heuristic: look for negation (e.g., "does not manage"),
    tool-context false positives (e.g., "manage dependencies"), and require
    the match to occur within a sentence that also contains people/team/reports tokens.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["bucket"] = df.apply(period_bucket, axis=1)
    rows = []

    neg_regex = re.compile(r"\b(?:not|no|don't|do not|neither|without)\b[\s\w]{0,20}", re.I)
    people_ctx = re.compile(r"\b(?:people|team|engineers?|staff|direct reports?|organi[sz]ation|headcount|reports?)\b", re.I)

    def context_validates(text: str, pattern: str) -> bool:
        # find the match; take a +-80-char window; check people/team tokens and no strong negation
        m = re.search(pattern, text, re.I)
        if m is None:
            return False
        start = max(0, m.start() - 80)
        end = min(len(text), m.end() + 80)
        window = text[start:end]
        if not people_ctx.search(window):
            # exception: "performance review" is assumed valid context
            if not re.search(r"\bperformance review\b|\b1[- ]?on[- ]?1\b|\b1:1\b|\bengineering manager\b", window, re.I):
                return False
        # negation right before the match, within a window of 30 chars
        pre = text[max(0, m.start() - 30) : m.start()]
        if neg_regex.search(pre):
            return False
        return True

    periods = {
        "pooled_2024": df[df["bucket"].isin(["arshkon_2024", "asaniczka_2024"])],
        "scraped_2026": df[df["bucket"] == "scraped_2026"],
    }
    # Stratify by tier × period, target 25 rows per (period × tier).
    samples = []
    for pname, pd_df in periods.items():
        for tier, pats in [("strong", MGMT_STRONG_TERMS), ("broad_extra", MGMT_BROAD_EXTRA)]:
            for term, pat in pats:
                # hits on this term only
                mask = pd_df["text"].str.contains(pat, regex=True, na=False, case=False)
                pool = pd_df[mask]
                if len(pool) == 0:
                    continue
                k = min(25, len(pool))
                idx = rng.choice(len(pool), size=k, replace=False)
                sub = pool.iloc[idx]
                for _, r in sub.iterrows():
                    ok = context_validates(r["text"] or "", pat)
                    rows.append(
                        dict(
                            uid=r["uid"],
                            period=pname,
                            tier=tier,
                            term=term,
                            precision_ok=ok,
                        )
                    )
    precision_df = pd.DataFrame(rows)
    precision_df.to_csv(OUT_TAB / "mgmt_precision_sample.csv", index=False)
    # summary by term
    summ = (
        precision_df.groupby(["period", "tier", "term"])
        .agg(n=("precision_ok", "size"), precision=("precision_ok", "mean"))
        .reset_index()
    )
    summ["precision"] = summ["precision"].round(4)
    summ.to_csv(OUT_TAB / "mgmt_precision_summary.csv", index=False)
    # Patterns below 80% precision
    below = summ[(summ["precision"] < 0.80) & (summ["n"] >= 10)].copy()
    below.to_csv(OUT_TAB / "mgmt_precision_below_80.csv", index=False)
    return summ


def step_outlier_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Top 1% by requirement_breadth — aggregate profile and sample titles/companies."""
    df = df.copy()
    df["bucket"] = df.apply(period_bucket, axis=1)
    rows = []
    for pname, pd_df in [
        ("pooled_2024", df[df["bucket"].isin(["arshkon_2024", "asaniczka_2024"])]),
        ("scraped_2026", df[df["bucket"] == "scraped_2026"]),
    ]:
        if len(pd_df) == 0:
            continue
        thr = pd_df["requirement_breadth"].quantile(0.99)
        outl = pd_df[pd_df["requirement_breadth"] >= thr]
        rest = pd_df[pd_df["requirement_breadth"] < thr]
        rec = dict(
            period=pname,
            n_outliers=len(outl),
            n_rest=len(rest),
            pct99_breadth_threshold=float(thr),
        )
        for m in [
            "requirement_breadth",
            "credential_stack_depth",
            "tech_count",
            "scope_count",
            "soft_skill_count",
            "mgmt_broad_count",
            "description_cleaned_length",
            "ai_binary",
        ]:
            rec[f"outl_{m}_mean"] = round(float(outl[m].mean()), 4)
            rec[f"rest_{m}_mean"] = round(float(rest[m].mean()), 4)
        # top employers among outliers
        top_emp = outl["company_name_canonical"].value_counts().head(10).to_dict()
        rec["top_employers"] = json.dumps(top_emp)
        # top titles among outliers
        top_titles = outl["title"].value_counts().head(10).to_dict()
        rec["top_titles"] = json.dumps(top_titles)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TAB / "outlier_top1pct_profile.csv", index=False)
    # Sample 20 outlier rows per period for inspection
    samples = []
    for pname, pd_df in [
        ("pooled_2024", df[df["bucket"].isin(["arshkon_2024", "asaniczka_2024"])]),
        ("scraped_2026", df[df["bucket"] == "scraped_2026"]),
    ]:
        thr = pd_df["requirement_breadth"].quantile(0.99)
        outl = pd_df[pd_df["requirement_breadth"] >= thr].sample(
            n=min(20, len(pd_df[pd_df["requirement_breadth"] >= thr])),
            random_state=42,
        )
        samples.append(
            outl.assign(period=pname)[
                [
                    "period",
                    "uid",
                    "title",
                    "company_name_canonical",
                    "requirement_breadth",
                    "credential_stack_depth",
                    "tech_count",
                    "description_cleaned_length",
                ]
            ]
        )
    sdf = pd.concat(samples, ignore_index=True)
    sdf.to_csv(OUT_TAB / "outlier_sample_20_per_period.csv", index=False)
    return out


def step_entry_specialist_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude entry-specialist employers and re-run J3 comparison."""
    es = pd.read_csv(ENTRY_SPECIALIST)
    es_names = set(es["company_name_canonical"].astype(str).str.lower())
    df = df.copy()
    df["bucket"] = df.apply(period_bucket, axis=1)
    df["co_low"] = df["company_name_canonical"].astype(str).str.lower()
    df_excl = df[~df["co_low"].isin(es_names)]
    rows = []
    cuts = {
        "J3_pooled_2024_full": df[df["bucket"].isin(["arshkon_2024", "asaniczka_2024"]) & df["J3"]],
        "J3_pooled_2024_ex_specialist": df_excl[
            df_excl["bucket"].isin(["arshkon_2024", "asaniczka_2024"]) & df_excl["J3"]
        ],
        "J3_scraped_2026_full": df[(df["bucket"] == "scraped_2026") & df["J3"]],
        "J3_scraped_2026_ex_specialist": df_excl[(df_excl["bucket"] == "scraped_2026") & df_excl["J3"]],
    }
    for name, sub in cuts.items():
        rec = dict(cut=name, n=len(sub))
        for m in [
            "requirement_breadth",
            "requirement_breadth_resid",
            "credential_stack_depth",
            "tech_count",
            "ai_binary",
            "description_cleaned_length",
        ]:
            vals = sub[m].dropna()
            if vals.empty:
                continue
            rec[f"{m}_mean"] = round(float(vals.mean()), 4)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TAB / "entry_specialist_sensitivity.csv", index=False)
    return out


def step_company_cap_sensitivity(df: pd.DataFrame, caps=(20, 50)) -> pd.DataFrame:
    df = df.copy()
    df["bucket"] = df.apply(period_bucket, axis=1)
    rows = []

    def cap_df(dd: pd.DataFrame, cap: int, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        parts = []
        for _, g in dd.groupby("company_name_canonical", dropna=False):
            if len(g) <= cap:
                parts.append(g)
            else:
                idx = rng.choice(len(g), size=cap, replace=False)
                parts.append(g.iloc[idx])
        return pd.concat(parts, ignore_index=True)

    for cap in caps:
        capped = cap_df(df, cap)
        for pname, pd_df in [
            ("pooled_2024", capped[capped["bucket"].isin(["arshkon_2024", "asaniczka_2024"])]),
            ("scraped_2026", capped[capped["bucket"] == "scraped_2026"]),
        ]:
            for sname, sub in [
                ("all", pd_df),
                ("J3", pd_df[pd_df["J3"]]),
                ("S4", pd_df[pd_df["S4"]]),
            ]:
                if len(sub) == 0:
                    continue
                rec = dict(cap=cap, period=pname, seniority_cut=sname, n=len(sub))
                for m in [
                    "requirement_breadth",
                    "requirement_breadth_resid",
                    "credential_stack_depth",
                    "tech_count",
                    "ai_binary",
                    "description_cleaned_length",
                ]:
                    vals = sub[m].dropna()
                    if vals.empty:
                        continue
                    rec[f"{m}_mean"] = round(float(vals.mean()), 4)
                rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TAB / "company_cap_sensitivity.csv", index=False)
    return out


def step_within_2024_snr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bucket"] = df.apply(period_bucket, axis=1)
    ars = df[df["bucket"] == "arshkon_2024"]
    asa = df[df["bucket"] == "asaniczka_2024"]
    scr = df[df["bucket"] == "scraped_2026"]
    pool = df[df["bucket"].isin(["arshkon_2024", "asaniczka_2024"])]
    rows = []
    for m in [
        "requirement_breadth",
        "requirement_breadth_resid",
        "credential_stack_depth",
        "tech_count",
        "ai_binary",
        "description_cleaned_length",
        "scope_density",
        "mgmt_strong_density",
        "mgmt_broad_density",
    ]:
        ars_mean = float(ars[m].mean())
        asa_mean = float(asa[m].mean())
        pool_mean = float(pool[m].mean())
        scr_mean = float(scr[m].mean())
        within24 = abs(ars_mean - asa_mean)
        cross = abs(pool_mean - scr_mean)
        snr = cross / max(within24, 1e-9)
        rows.append(
            dict(
                metric=m,
                arshkon_mean=round(ars_mean, 4),
                asaniczka_mean=round(asa_mean, 4),
                pooled_2024_mean=round(pool_mean, 4),
                scraped_2026_mean=round(scr_mean, 4),
                within_2024_abs=round(within24, 4),
                cross_period_abs=round(cross, 4),
                snr=round(snr, 3),
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TAB / "within_2024_snr.csv", index=False)
    return out


def step_text_source_split(df: pd.DataFrame) -> pd.DataFrame:
    """Labeled vs raw text source split — report all the metrics both ways for scraped rows."""
    df = df.copy()
    df["bucket"] = df.apply(period_bucket, axis=1)
    rows = []
    for pname, pd_df in [
        ("arshkon_2024", df[df["bucket"] == "arshkon_2024"]),
        ("asaniczka_2024", df[df["bucket"] == "asaniczka_2024"]),
        ("scraped_2026", df[df["bucket"] == "scraped_2026"]),
    ]:
        for src, sub in [
            ("llm", pd_df[pd_df["text_source"] == "llm"]),
            ("raw", pd_df[pd_df["text_source"] == "raw"]),
        ]:
            if len(sub) == 0:
                continue
            rec = dict(period=pname, text_source=src, n=len(sub))
            for m in [
                "requirement_breadth",
                "credential_stack_depth",
                "tech_count",
                "ai_binary",
                "description_cleaned_length",
            ]:
                vals = sub[m].dropna()
                if vals.empty:
                    continue
                rec[f"{m}_mean"] = round(float(vals.mean()), 4)
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TAB / "text_source_split.csv", index=False)
    return out


# ---------------------------- persist shared artifact ------------------------


def persist_features(df: pd.DataFrame):
    """Save the per-posting feature parquet consumed by T20 / T33 / T35."""
    cols = [
        "uid",
        "tech_count",
        "requirement_breadth",
        "requirement_breadth_resid",
        "credential_stack_depth",
        "credential_stack_depth_resid",
        "tech_density",
        "tech_density_resid",
        "scope_density",
        "scope_density_resid",
        "mgmt_strong_density",
        "mgmt_broad_density",
        "mgmt_broad_density_resid",
        "ai_binary",
        "education_level",
        "yoe_min_years_llm",
        "description_cleaned_length",
        "text_source",
        "source",
        "period",
    ]
    out = df[cols].copy()
    path = SHARED / "T11_posting_features.parquet"
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), path)
    print(f"[T11] wrote {path} ({len(out)} rows)")


# ---------------------------- main --------------------------------------------


def main():
    print("[T11] loading corpus ...")
    df = load_corpus()
    print(f"[T11] {len(df)} rows")
    print("[T11] loading tech matrix ...")
    tech = pq.read_table(TECH_MATRIX).to_pandas()
    print("[T11] building features ...")
    df = build_features(df, tech)
    print("[T11] residualizing ...")
    df = residualize_all(df)
    print("[T11] YOE seniority labels ...")
    df = yoe_seniority(df)

    print("[T11] step: distribution panel ...")
    step_distribution_panel(df)
    print("[T11] step: credential stacking ...")
    step_credential_stacking(df)
    print("[T11] step: entry-level comparison ...")
    step_entry_level_only(df)
    print("[T11] step: correlation check ...")
    step_correlation_check(df)
    print("[T11] step: mgmt term breakdown ...")
    step_mgmt_term_breakdown(df)
    print("[T11] step: mgmt precision sample ...")
    step_mgmt_precision_sample(df)
    print("[T11] step: outlier analysis ...")
    step_outlier_analysis(df)
    print("[T11] step: entry-specialist sensitivity ...")
    step_entry_specialist_sensitivity(df)
    print("[T11] step: company cap sensitivity ...")
    step_company_cap_sensitivity(df)
    print("[T11] step: within-2024 SNR ...")
    step_within_2024_snr(df)
    print("[T11] step: text-source split ...")
    step_text_source_split(df)

    print("[T11] persisting posting features ...")
    persist_features(df)
    print("[T11] done.")


if __name__ == "__main__":
    main()
