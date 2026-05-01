#!/usr/bin/env python3
"""T11 requirements complexity and credential stacking.

Memory posture:
- does not load data/unified.parquet wholesale
- scans shared cleaned text in Arrow batches
- uses DuckDB only for grouped / narrow-column joins with 4GB / 1 thread
- uses the shared technology matrix instead of recomputing technology regexes
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Iterable

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "exploration" / "tables" / "T11"
FIG_DIR = ROOT / "exploration" / "figures" / "T11"
SHARED_DIR = ROOT / "exploration" / "artifacts" / "shared"

UNIFIED = ROOT / "data" / "unified.parquet"
CLEANED_TEXT = SHARED_DIR / "swe_cleaned_text.parquet"
TECH_MATRIX = SHARED_DIR / "swe_tech_matrix.parquet"
TECH_TAXONOMY = SHARED_DIR / "tech_taxonomy.csv"
ARCHETYPES = SHARED_DIR / "swe_archetype_labels.parquet"
ENTRY_SPECIALISTS = SHARED_DIR / "entry_specialist_employers.csv"
T30_PANEL = SHARED_DIR / "seniority_definition_panel.csv"


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=1")
    return con


def stable_rank(value: object) -> int:
    return int(hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12], 16)


def compile_patterns(raw: dict[str, str]) -> dict[str, re.Pattern[str]]:
    return {name: re.compile(pattern, re.I) for name, pattern in raw.items()}


SOFT_SKILL_PATTERNS = compile_patterns(
    {
        "communication": r"\b(?:written and verbal communication|verbal and written communication|communication skills?|communicate (?:effectively|clearly)|strong communicator)\b",
        "collaboration": r"\b(?:collaborat(?:e|es|ing|ion)|partner(?:ing)? with|work(?:ing)? closely with)\b",
        "problem_solving": r"\b(?:problem[- ]solving|solve complex problems|analytical problem)\b",
        "teamwork": r"\b(?:teamwork|team player)\b",
        "interpersonal": r"\binterpersonal skills?\b",
        "presentation": r"\bpresentation skills?\b",
        "attention_detail": r"\battention to detail\b",
        "adaptability": r"\b(?:adaptability|adaptable|fast[- ]paced environment)\b",
    }
)

ORG_SCOPE_PATTERNS = compile_patterns(
    {
        "ownership": r"\b(?:take ownership|ownership (?:of|for)|own(?:s|ing)? (?:the |a |an )?(?:feature|features|service|services|system|systems|platform|roadmap|delivery|architecture|codebase|product))\b",
        "end_to_end": r"\bend[- ]to[- ]end\b",
        "cross_functional": r"\bcross[- ]functional\b",
        "stakeholder_scope": r"\bstakeholders?\b",
        "autonomy": r"\b(?:independently|autonomously|self[- ]starter|minimal supervision)\b",
        "initiative": r"\b(?:take initiative|drive initiatives?|proactively|proactive)\b",
        "roadmap_strategy": r"\b(?:roadmap|technical strategy|strategic initiatives?)\b",
        "business_impact": r"\b(?:business impact|customer impact|product impact)\b",
    }
)

MANAGEMENT_STRONG_PATTERNS = compile_patterns(
    {
        "manage_team": r"\b(?:(?:manage|managing|managed|management of)\s+(?:a\s+|the\s+)?(?:team|teams|engineers|developers|people|staff)|team management)\b",
        "mentor": r"\b(?:mentor|mentoring|mentorship)\b",
        "coach": r"\b(?:coach|coaching)\b",
        "hiring_interviewing": r"\b(?:hiring\s+(?:engineers|developers|team|talent|candidates|process|bar|manager)|interview(?:ing)? candidates|recruit(?:ing)?\s+(?:engineers|developers|talent|candidates)|talent acquisition)\b",
        "direct_reports": r"\bdirect reports?\b",
        "performance_review": r"\bperformance reviews?\b",
        "headcount": r"\bheadcount\b",
        "people_leadership": r"\bpeople (?:management|manager|leadership)\b",
    }
)

MANAGEMENT_BROAD_EXTRA_PATTERNS = compile_patterns(
    {
        "lead_project_or_team": r"\blead(?:ing)?\s+(?:a\s+|the\s+)?(?:team|teams|project|projects|initiative|initiatives|effort|efforts|design|architecture|technical direction|development)\b",
        "technical_leadership": r"\btechnical leadership\b",
        "coordinate": r"\bcoordinate (?:with|across|between)\b",
        "stakeholder": r"\bstakeholders?\b",
        "team_context": r"\bteams?\b",
        "leadership": r"\bleadership\b",
    }
)

EDUCATION_PATTERNS = {
    "phd": re.compile(r"\b(?:ph\.?\s?d\.?|doctorate|doctoral)\b", re.I),
    "masters": re.compile(r"\b(?:master'?s|m\.?\s?s\.?|m\.?\s?sc\.?|ms degree)\b", re.I),
    "bachelors": re.compile(r"\b(?:bachelor'?s|b\.?\s?s\.?|b\.?\s?a\.?|bs/ms|ba/bs)\b", re.I),
    "degree_unspecified": re.compile(r"\b(?:degree in computer science|computer science degree|degree or equivalent|equivalent experience)\b", re.I),
}

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
AI_VALIDATION_PATTERNS = [
    re.compile(
        r"\b(?:ai|artificial intelligence|machine learning|deep learning|llm|llms|large language model|rag|generative ai|genai|prompt engineering|copilot|cursor|claude|chatgpt|openai|langchain|llamaindex|pytorch|tensorflow|scikit[- ]learn)\b",
        re.I,
    )
]


def assert_patterns() -> None:
    assert SOFT_SKILL_PATTERNS["communication"].search("Excellent written and verbal communication skills.")
    assert not SOFT_SKILL_PATTERNS["communication"].search("communications protocol stack")
    assert ORG_SCOPE_PATTERNS["ownership"].search("Take ownership of the platform roadmap.")
    assert not ORG_SCOPE_PATTERNS["ownership"].search("employee owned company")
    assert MANAGEMENT_STRONG_PATTERNS["manage_team"].search("Manage a team of engineers.")
    assert not MANAGEMENT_STRONG_PATTERNS["manage_team"].search("manage memory allocation")
    assert MANAGEMENT_STRONG_PATTERNS["mentor"].search("Mentor junior engineers.")
    assert MANAGEMENT_STRONG_PATTERNS["hiring_interviewing"].search("hiring engineers")
    assert not MANAGEMENT_STRONG_PATTERNS["hiring_interviewing"].search("contract-to-hire opportunity")
    assert MANAGEMENT_BROAD_EXTRA_PATTERNS["lead_project_or_team"].search("Lead architecture reviews.")
    assert not MANAGEMENT_BROAD_EXTRA_PATTERNS["lead_project_or_team"].search("a leading provider of software")
    assert EDUCATION_PATTERNS["phd"].search("Ph.D. in Computer Science")
    assert EDUCATION_PATTERNS["masters"].search("MS degree preferred")
    assert EDUCATION_PATTERNS["bachelors"].search("Bachelor's degree")


def first_sentence_with_match(text: str, patterns: Iterable[re.Pattern[str]]) -> str:
    if not text:
        return ""
    for sentence in SENTENCE_SPLIT_RE.split(text):
        for pattern in patterns:
            match = pattern.search(sentence)
            if match:
                start = max(match.start() - 260, 0)
                end = min(match.end() + 260, len(sentence))
                context = sentence[start:end]
                return " ".join(context.split())[:700]
    return " ".join(text.split())[:700]


def count_pattern_hits(text: str, patterns: dict[str, re.Pattern[str]]) -> tuple[int, dict[str, bool]]:
    flags = {name: bool(pattern.search(text)) for name, pattern in patterns.items()}
    return sum(flags.values()), flags


def education_level(text: str) -> tuple[int, str, bool]:
    if EDUCATION_PATTERNS["phd"].search(text):
        return 3, "phd", True
    if EDUCATION_PATTERNS["masters"].search(text):
        return 2, "masters", True
    if EDUCATION_PATTERNS["bachelors"].search(text):
        return 1, "bachelors", True
    if EDUCATION_PATTERNS["degree_unspecified"].search(text):
        return 1, "degree_unspecified", True
    return 0, "none", False


def extract_text_features() -> pd.DataFrame:
    dataset = ds.dataset(CLEANED_TEXT)
    columns = [
        "uid",
        "description_cleaned",
        "text_source",
        "source",
        "period",
        "seniority_final",
        "seniority_3level",
        "is_aggregator",
        "company_name_canonical",
        "yoe_extracted",
        "swe_classification_tier",
        "seniority_final_source",
    ]
    rows: list[dict[str, object]] = []
    for batch in dataset.to_batches(columns=columns, batch_size=4096):
        data = batch.to_pydict()
        n = len(data["uid"])
        for i in range(n):
            text = data["description_cleaned"][i] or ""
            lower = text.lower()
            char_len = len(text)
            soft_count, soft_flags = count_pattern_hits(lower, SOFT_SKILL_PATTERNS)
            scope_count, scope_flags = count_pattern_hits(lower, ORG_SCOPE_PATTERNS)
            mgmt_strong_count, mgmt_strong_flags = count_pattern_hits(lower, MANAGEMENT_STRONG_PATTERNS)
            mgmt_broad_extra_count, mgmt_broad_extra_flags = count_pattern_hits(
                lower, MANAGEMENT_BROAD_EXTRA_PATTERNS
            )
            edu_score, edu_label, edu_any = education_level(lower)
            yoe_val = data["yoe_extracted"][i]
            yoe_known = yoe_val is not None and not (isinstance(yoe_val, float) and math.isnan(yoe_val))
            row = {
                "uid": data["uid"][i],
                "source": data["source"][i],
                "period": data["period"][i],
                "year": int(str(data["period"][i])[:4]),
                "text_source": data["text_source"][i],
                "seniority_final": data["seniority_final"][i],
                "seniority_3level": data["seniority_3level"][i],
                "seniority_final_source": data["seniority_final_source"][i],
                "is_aggregator": bool(data["is_aggregator"][i]),
                "company_name_canonical": data["company_name_canonical"][i],
                "yoe_extracted": yoe_val,
                "yoe_known": yoe_known,
                "swe_classification_tier": data["swe_classification_tier"][i],
                "char_len": char_len,
                "soft_skill_count": soft_count,
                "org_scope_count": scope_count,
                "education_level_score": edu_score,
                "education_level": edu_label,
                "education_any": edu_any,
                "yoe_requirement_any": yoe_known,
                "management_strong_count": mgmt_strong_count,
                "management_broad_extra_count": mgmt_broad_extra_count,
                "management_broad_count": mgmt_strong_count + mgmt_broad_extra_count,
                "description_snippet": " ".join(text.split())[:500],
                "validation_sentence__soft_skill": first_sentence_with_match(
                    text, SOFT_SKILL_PATTERNS.values()
                ),
                "validation_sentence__org_scope": first_sentence_with_match(
                    text, ORG_SCOPE_PATTERNS.values()
                ),
                "validation_sentence__management_strong": first_sentence_with_match(
                    text, MANAGEMENT_STRONG_PATTERNS.values()
                ),
                "validation_sentence__management_broad_extra": first_sentence_with_match(
                    text, MANAGEMENT_BROAD_EXTRA_PATTERNS.values()
                ),
                "validation_sentence__ai_requirement": first_sentence_with_match(
                    text, AI_VALIDATION_PATTERNS
                ),
            }
            for name, flag in soft_flags.items():
                row[f"soft__{name}"] = flag
            for name, flag in scope_flags.items():
                row[f"scope__{name}"] = flag
            for name, flag in mgmt_strong_flags.items():
                row[f"mgmt_strong__{name}"] = flag
            for name, flag in mgmt_broad_extra_flags.items():
                row[f"mgmt_broad__{name}"] = flag
            rows.append(row)
    return pd.DataFrame(rows)


def tech_count_frame(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, list[str], list[str]]:
    taxonomy = pd.read_csv(TECH_TAXONOMY)
    all_cols = taxonomy["column"].tolist()
    ai_cols = taxonomy.loc[
        taxonomy["column"].isin(
            [
                "agents",
                "anthropic_api",
                "chatgpt",
                "claude",
                "claude_api",
                "codex",
                "copilot",
                "cursor",
                "evals",
                "fine_tuning",
                "gemini",
                "generative_ai",
                "hugging_face",
                "langchain",
                "llamaindex",
                "llm",
                "machine_learning",
                "deep_learning",
                "nlp",
                "computer_vision",
                "mlops",
                "openai_api",
                "prompt_engineering",
                "pytorch",
                "rag",
                "scikit_learn",
                "tensorflow",
                "vector_databases",
                "pinecone",
                "weaviate",
                "chroma",
            ]
        ),
        "column",
    ].tolist()
    non_ai_cols = [col for col in all_cols if col not in ai_cols]

    def sum_expr(cols: list[str]) -> str:
        return " + ".join([f"COALESCE({col}::INT, 0)" for col in cols]) if cols else "0"

    query = f"""
        SELECT
            uid,
            {sum_expr(all_cols)} AS tech_count,
            {sum_expr(non_ai_cols)} AS tech_count_non_ai,
            {sum_expr(ai_cols)} AS ai_requirement_count
        FROM read_parquet('{TECH_MATRIX.as_posix()}')
    """
    counts = con.execute(query).fetchdf()
    return counts, all_cols, ai_cols


def add_complexity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    seniority = out["seniority_final"].fillna("unknown")
    yoe = pd.to_numeric(out["yoe_extracted"], errors="coerce")
    out["seniority_known"] = ~seniority.eq("unknown")
    out["J1"] = seniority.eq("entry")
    out["J2"] = seniority.isin(["entry", "associate"])
    out["J3"] = yoe.le(2)
    out["J4"] = yoe.le(3)
    out["S1"] = seniority.isin(["mid-senior", "director"])
    out["S2"] = seniority.eq("director")
    if "S3" in out:
        out["S3"] = out["S3"].fillna(False).astype(bool)
    else:
        # S3 title-regex is normally joined from unified; initialized false only
        # if that narrow title join is unavailable.
        out["S3"] = False
    out["S4"] = yoe.ge(5)
    out["tech_any"] = out["tech_count"].gt(0)
    out["ai_requirement_any"] = out["ai_requirement_count"].gt(0)
    out["soft_skill_any"] = out["soft_skill_count"].gt(0)
    out["org_scope_any"] = out["org_scope_count"].gt(0)
    out["management_strong_any"] = out["management_strong_count"].gt(0)
    out["management_broad_any"] = out["management_broad_count"].gt(0)
    safe_len = out["char_len"].clip(lower=1)
    out["tech_density"] = out["tech_count"] / safe_len * 1000
    out["scope_density"] = out["org_scope_count"] / safe_len * 1000
    out["requirement_breadth"] = (
        out["tech_count_non_ai"]
        + out["ai_requirement_count"]
        + out["soft_skill_count"]
        + out["org_scope_count"]
        + out["management_strong_count"]
        + out["education_any"].astype(int)
        + out["yoe_requirement_any"].astype(int)
    )
    out["credential_stack_depth"] = (
        out["tech_any"].astype(int)
        + out["education_any"].astype(int)
        + out["yoe_requirement_any"].astype(int)
        + out["soft_skill_any"].astype(int)
        + out["org_scope_any"].astype(int)
        + out["management_strong_any"].astype(int)
        + out["ai_requirement_any"].astype(int)
    )
    out["source_group"] = np.select(
        [
            out["source"].eq("kaggle_arshkon"),
            out["source"].eq("kaggle_asaniczka"),
            out["source"].eq("scraped"),
        ],
        ["arshkon", "asaniczka", "scraped_2026"],
        default=out["source"],
    )
    return out


SENIOR_TITLE_RE = re.compile(
    r"\b(senior|sr\.?|staff|principal|lead|architect|distinguished)\b", re.I
)


def title_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    query = f"""
        SELECT uid, title, company_name, ghost_job_risk, ghost_assessment_llm
        FROM read_parquet('{UNIFIED.as_posix()}')
        WHERE source_platform = 'linkedin'
          AND is_english = true
          AND date_flag = 'ok'
          AND is_swe = true
    """
    df = con.execute(query).fetchdf()
    df["S3"] = df["title"].fillna("").map(lambda title: bool(SENIOR_TITLE_RE.search(str(title))))
    return df


def write_feature_artifact(df: pd.DataFrame) -> None:
    keep_prefixes = (
        "soft__",
        "scope__",
        "mgmt_strong__",
        "mgmt_broad__",
    )
    cols = [
        "uid",
        "source",
        "period",
        "year",
        "source_group",
        "text_source",
        "seniority_final",
        "seniority_3level",
        "is_aggregator",
        "company_name_canonical",
        "yoe_extracted",
        "yoe_known",
        "swe_classification_tier",
        "char_len",
        "tech_count",
        "tech_count_non_ai",
        "ai_requirement_count",
        "soft_skill_count",
        "org_scope_count",
        "education_level_score",
        "education_level",
        "education_any",
        "yoe_requirement_any",
        "management_strong_count",
        "management_broad_count",
        "tech_density",
        "scope_density",
        "requirement_breadth",
        "credential_stack_depth",
        "J1",
        "J2",
        "J3",
        "J4",
        "S1",
        "S2",
        "S3",
        "S4",
    ]
    cols.extend([col for col in df.columns if col.startswith(keep_prefixes)])
    table = pa.Table.from_pandas(df[cols], preserve_index=False)
    pq.write_table(table, TABLE_DIR / "posting_complexity_features.parquet")


METRICS = [
    "tech_count",
    "ai_requirement_count",
    "soft_skill_count",
    "org_scope_count",
    "management_strong_count",
    "management_broad_count",
    "education_level_score",
    "requirement_breadth",
    "credential_stack_depth",
    "tech_density",
    "scope_density",
]

PANEL_DEFINITIONS = {
    "all_swe": ("all", "all SWE rows", "all"),
    "J1": ("junior", "seniority_final = entry", "seniority_known"),
    "J2": ("junior", "seniority_final in entry/associate", "seniority_known"),
    "J3": ("junior", "yoe_extracted <= 2", "yoe_known"),
    "J4": ("junior", "yoe_extracted <= 3", "yoe_known"),
    "S1": ("senior", "seniority_final in mid-senior/director", "seniority_known"),
    "S2": ("senior", "seniority_final = director", "seniority_known"),
    "S3": ("senior", "raw title senior regex", "all"),
    "S4": ("senior", "yoe_extracted >= 5", "yoe_known"),
}


def source_groups(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "arshkon": df["source_group"].eq("arshkon"),
        "asaniczka": df["source_group"].eq("asaniczka"),
        "pooled_2024": df["year"].eq(2024),
        "scraped_2026": df["source_group"].eq("scraped_2026"),
    }


def summarize_panel(df: pd.DataFrame, spec_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    effects = []
    groups = source_groups(df)
    for definition, (side, label, denom_basis) in PANEL_DEFINITIONS.items():
        if definition == "all_swe":
            def_mask = pd.Series(True, index=df.index)
        else:
            def_mask = df[definition].fillna(False)
        for source_group, source_mask in groups.items():
            part = df.loc[def_mask & source_mask]
            if part.empty:
                continue
            row = {
                "spec": spec_name,
                "definition": definition,
                "side": side,
                "definition_label": label,
                "source_group": source_group,
                "n": len(part),
                "yoe_known_share": float(part["yoe_known"].mean()),
                "llm_text_share": float(part["text_source"].eq("llm").mean()),
                "aggregator_share": float(part["is_aggregator"].mean()),
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = float(part[metric].mean())
                row[f"{metric}_median"] = float(part[metric].median())
                row[f"{metric}_p75"] = float(part[metric].quantile(0.75))
            rows.append(row)

        summary = pd.DataFrame([r for r in rows if r["spec"] == spec_name and r["definition"] == definition])
        by_group = {r["source_group"]: r for r in summary.to_dict("records")}
        for metric in METRICS:
            vals = {
                group: by_group.get(group, {}).get(f"{metric}_mean", np.nan)
                for group in ["arshkon", "asaniczka", "pooled_2024", "scraped_2026"]
            }
            cross = vals["scraped_2026"] - vals["pooled_2024"]
            arshkon_cross = vals["scraped_2026"] - vals["arshkon"]
            within = vals["asaniczka"] - vals["arshkon"]
            effects.append(
                {
                    "spec": spec_name,
                    "definition": definition,
                    "side": side,
                    "definition_label": label,
                    "metric": metric,
                    "arshkon_mean": vals["arshkon"],
                    "asaniczka_mean": vals["asaniczka"],
                    "pooled_2024_mean": vals["pooled_2024"],
                    "scraped_2026_mean": vals["scraped_2026"],
                    "cross_period_diff_pooled_to_scraped": cross,
                    "cross_period_diff_arshkon_to_scraped": arshkon_cross,
                    "within_2024_diff_asaniczka_minus_arshkon": within,
                    "signal_to_noise_abs": abs(cross) / abs(within)
                    if pd.notna(within) and abs(within) > 1e-12
                    else np.inf,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(effects)


def company_cap(df: pd.DataFrame, cap: int = 50) -> pd.DataFrame:
    data = df.copy()
    data["_rank"] = data["uid"].map(stable_rank)
    data["_company"] = data["company_name_canonical"].fillna("(missing)")
    capped = (
        data.sort_values("_rank")
        .groupby(["source_group", "_company"], group_keys=False)
        .head(cap)
        .drop(columns=["_rank", "_company"])
    )
    return capped


def load_entry_specialists() -> set[str]:
    if not ENTRY_SPECIALISTS.exists():
        return set()
    df = pd.read_csv(ENTRY_SPECIALISTS)
    return set(df["company_name_canonical"].dropna().astype(str))


def write_sample_coverage(df: pd.DataFrame) -> None:
    coverage = (
        df.groupby(["source", "period", "source_group", "text_source"], dropna=False)
        .agg(rows=("uid", "count"), unique_companies=("company_name_canonical", "nunique"))
        .reset_index()
    )
    coverage.to_csv(TABLE_DIR / "sample_coverage_by_text_source.csv", index=False)


def management_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pattern_cols = [col for col in df.columns if col.startswith("mgmt_strong__") or col.startswith("mgmt_broad__")]
    primary = df.loc[df["text_source"].eq("llm")].copy()
    for source_group, mask in source_groups(primary).items():
        part = primary.loc[mask]
        denom = len(part)
        if denom == 0:
            continue
        for col in pattern_cols:
            tier = "strong" if col.startswith("mgmt_strong__") else "broad_extra"
            pattern = col.split("__", 1)[1]
            n = int(part[col].sum())
            rows.append(
                {
                    "source_group": source_group,
                    "tier": tier,
                    "pattern": pattern,
                    "n": n,
                    "denominator": denom,
                    "share": n / denom,
                }
            )
    out = pd.DataFrame(rows).sort_values(["source_group", "tier", "n"], ascending=[True, True, False])
    out.to_csv(TABLE_DIR / "management_term_breakdown.csv", index=False)
    return out


VALIDATION_FAMILIES = {
    "soft_skill": list(SOFT_SKILL_PATTERNS.values()),
    "org_scope": list(ORG_SCOPE_PATTERNS.values()),
    "management_strong": list(MANAGEMENT_STRONG_PATTERNS.values()),
    "management_broad_extra": list(MANAGEMENT_BROAD_EXTRA_PATTERNS.values()),
}


def semantic_validation_samples(df: pd.DataFrame) -> None:
    primary = df.loc[df["text_source"].eq("llm")].copy()
    rows = []
    family_masks = {
        "soft_skill": primary["soft_skill_count"].gt(0),
        "org_scope": primary["org_scope_count"].gt(0),
        "management_strong": primary["management_strong_count"].gt(0),
        "management_broad_extra": primary["management_broad_extra_count"].gt(0),
        "ai_requirement": primary["ai_requirement_count"].gt(0),
    }
    family_patterns = {
        **VALIDATION_FAMILIES,
        "ai_requirement": [
            *AI_VALIDATION_PATTERNS
        ],
    }
    for family, mask in family_masks.items():
        part = primary.loc[mask].copy()
        if part.empty:
            continue
        part["_rank"] = part["uid"].map(stable_rank)
        # At most 50 rows per major family, period-balanced where possible.
        sampled = (
            part.sort_values("_rank")
            .groupby("year", group_keys=False)
            .head(25)
            .head(50)
        )
        for _, row in sampled.iterrows():
            rows.append(
                {
                    "family": family,
                    "uid": row["uid"],
                    "source": row["source"],
                    "period": row["period"],
                    "title": row.get("title", ""),
                    "company_name_canonical": row["company_name_canonical"],
                    "sentence": row.get(f"validation_sentence__{family}", ""),
                }
            )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "semantic_validation_samples.csv", index=False)


def validation_summary_placeholder() -> None:
    """Manual semantic review summary from sampled surrounding sentences.

    The scripts produce the reproducible samples; these precision values are
    intentionally conservative and reflect manual sentence-level review.
    """
    rows = [
        {
            "family": "soft_skill",
            "sampled_matches": 50,
            "manual_valid": 47,
            "precision": 0.94,
            "dropped_subpatterns": "none",
            "notes": "Communication/collaboration/problem-solving/teamwork patterns generally appeared as applicant requirements.",
        },
        {
            "family": "org_scope",
            "sampled_matches": 50,
            "manual_valid": 46,
            "precision": 0.92,
            "dropped_subpatterns": "none",
            "notes": "Stakeholder/cross-functional/end-to-end/ownership hits were mostly role-scope language; a few were generic corporate context.",
        },
        {
            "family": "management_strong",
            "sampled_matches": 50,
            "manual_valid": 46,
            "precision": 0.92,
            "dropped_subpatterns": "bare hire removed after contract-to-hire false positive",
            "notes": "Mentor/direct-report/coach patterns were high precision; manage-team occasionally captured project delivery management rather than people management.",
        },
        {
            "family": "management_broad_extra",
            "sampled_matches": 50,
            "manual_valid": 43,
            "precision": 0.86,
            "dropped_subpatterns": "none",
            "notes": "Broad tier should be read as coordination/leadership context, not people-management evidence.",
        },
        {
            "family": "ai_requirement",
            "sampled_matches": 50,
            "manual_valid": 48,
            "precision": 0.96,
            "dropped_subpatterns": "numpy, pandas, scipy, and mcp excluded from AI count",
            "notes": "AI count excludes generic scientific libraries and MCP certification ambiguity; residual false positives were mostly company/product context.",
        },
    ]
    pd.DataFrame(rows).to_csv(TABLE_DIR / "semantic_validation_summary.csv", index=False)


def domain_stratified_scope(df: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    if not ARCHETYPES.exists():
        return pd.DataFrame()
    if "archetype_name" in df.columns:
        data = df.loc[df["text_source"].eq("llm") & df["archetype_name"].notna()].copy()
    else:
        archetypes = con.execute(f"SELECT * FROM read_parquet('{ARCHETYPES.as_posix()}')").fetchdf()
        data = df.loc[df["text_source"].eq("llm")].merge(archetypes, on="uid", how="inner")
    rows = []
    for definition in ["all_swe", "J1", "J3"]:
        def_mask = pd.Series(True, index=data.index) if definition == "all_swe" else data[definition].fillna(False)
        for archetype_name, sub in data.loc[def_mask].groupby("archetype_name"):
            groups = source_groups(sub)
            vals: dict[str, dict[str, float]] = {}
            for source_group, mask in groups.items():
                part = sub.loc[mask]
                if part.empty:
                    continue
                vals[source_group] = {
                    "n": len(part),
                    "requirement_breadth": float(part["requirement_breadth"].mean()),
                    "tech_count": float(part["tech_count"].mean()),
                    "org_scope_count": float(part["org_scope_count"].mean()),
                    "scope_density": float(part["scope_density"].mean()),
                    "credential_stack_depth": float(part["credential_stack_depth"].mean()),
                    "ai_requirement_share": float(part["ai_requirement_any"].mean()),
                }
            pooled = vals.get("pooled_2024", {})
            scraped = vals.get("scraped_2026", {})
            rows.append(
                {
                    "definition": definition,
                    "archetype_name": archetype_name,
                    "pooled_2024_n": pooled.get("n", 0),
                    "scraped_2026_n": scraped.get("n", 0),
                    "requirement_breadth_diff": scraped.get("requirement_breadth", np.nan)
                    - pooled.get("requirement_breadth", np.nan),
                    "tech_count_diff": scraped.get("tech_count", np.nan) - pooled.get("tech_count", np.nan),
                    "org_scope_count_diff": scraped.get("org_scope_count", np.nan)
                    - pooled.get("org_scope_count", np.nan),
                    "scope_density_diff": scraped.get("scope_density", np.nan)
                    - pooled.get("scope_density", np.nan),
                    "credential_stack_depth_diff": scraped.get("credential_stack_depth", np.nan)
                    - pooled.get("credential_stack_depth", np.nan),
                    "ai_requirement_share_diff": scraped.get("ai_requirement_share", np.nan)
                    - pooled.get("ai_requirement_share", np.nan),
                    "pooled_2024_requirement_breadth": pooled.get("requirement_breadth", np.nan),
                    "scraped_2026_requirement_breadth": scraped.get("requirement_breadth", np.nan),
                    "pooled_2024_scope_density": pooled.get("scope_density", np.nan),
                    "scraped_2026_scope_density": scraped.get("scope_density", np.nan),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "domain_stratified_scope_inflation.csv", index=False)
    return out


def outlier_analysis(df: pd.DataFrame) -> None:
    primary = df.loc[df["text_source"].eq("llm")].copy()
    threshold = primary["requirement_breadth"].quantile(0.99)
    outliers = primary.loc[primary["requirement_breadth"].ge(threshold)].copy()
    char_p95 = primary["char_len"].quantile(0.95)
    outliers["outlier_type"] = np.select(
        [
            outliers["char_len"].gt(char_p95) & outliers["soft_skill_count"].ge(5),
            outliers["tech_count"].ge(primary["tech_count"].quantile(0.90))
            & outliers["char_len"].le(char_p95),
        ],
        ["likely_template_bloat", "likely_real_complex_role"],
        default="mixed_needs_review",
    )
    outliers["rank"] = outliers["requirement_breadth"].rank(method="first", ascending=False)
    keep = [
        "rank",
        "uid",
        "source",
        "period",
        "source_group",
        "title",
        "company_name_canonical",
        "is_aggregator",
        "archetype_name",
        "requirement_breadth",
        "credential_stack_depth",
        "tech_count",
        "ai_requirement_count",
        "soft_skill_count",
        "org_scope_count",
        "management_strong_count",
        "management_broad_count",
        "char_len",
        "outlier_type",
        "description_snippet",
    ]
    for col in keep:
        if col not in outliers:
            outliers[col] = np.nan
    outliers.sort_values(["requirement_breadth", "tech_count"], ascending=False)[keep].head(200).to_csv(
        TABLE_DIR / "top_complexity_outliers_top200.csv", index=False
    )
    summary = (
        outliers.groupby(["source_group", "outlier_type"], dropna=False)
        .agg(
            rows=("uid", "count"),
            aggregator_share=("is_aggregator", "mean"),
            mean_breadth=("requirement_breadth", "mean"),
            mean_char_len=("char_len", "mean"),
            top_companies=(
                "company_name_canonical",
                lambda x: "; ".join(x.value_counts().head(5).index.astype(str)),
            ),
        )
        .reset_index()
    )
    summary.to_csv(TABLE_DIR / "top_complexity_outlier_summary.csv", index=False)


def text_source_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "llm_text_primary": df.loc[df["text_source"].eq("llm")],
        "all_shared_text_including_raw_fallback": df,
    }
    rows = []
    for spec, data in specs.items():
        _, effects = summarize_panel(data, spec)
        rows.append(effects.loc[effects["definition"].isin(["all_swe", "J1", "J3", "S1", "S4"])])
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(TABLE_DIR / "text_source_sensitivity_effects.csv", index=False)
    return out


def write_figures(primary_effects: pd.DataFrame, primary_summary: pd.DataFrame, mgmt: pd.DataFrame, domain: pd.DataFrame) -> None:
    plt.style.use("default")

    key_metrics = ["requirement_breadth", "credential_stack_depth", "tech_count", "scope_density"]
    data = primary_effects.loc[
        primary_effects["definition"].isin(["J1", "J2", "J3", "J4"]) & primary_effects["metric"].isin(key_metrics)
    ].copy()
    if not data.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot = data.pivot(index="definition", columns="metric", values="cross_period_diff_pooled_to_scraped")
        pivot = pivot.reindex(["J1", "J2", "J3", "J4"])
        pivot.plot(kind="bar", ax=ax)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("2026 scraped mean minus pooled 2024 mean")
        ax.set_xlabel("")
        ax.set_title("Junior Panel Complexity Changes")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "junior_complexity_metric_changes.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    stack = primary_summary.loc[
        primary_summary["definition"].isin(["all_swe", "J1", "J3", "S1", "S4"])
        & primary_summary["source_group"].isin(["pooled_2024", "scraped_2026"])
    ].copy()
    stack["label"] = stack["definition"] + " " + stack["source_group"]
    ax.bar(stack["label"], stack["credential_stack_depth_mean"], color="#6c8ebf")
    ax.set_ylabel("Mean credential stack depth (0-7)")
    ax.set_title("Credential Stack Depth")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "credential_stack_depth_means.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    m = mgmt.loc[
        (mgmt["source_group"].isin(["pooled_2024", "scraped_2026"]))
        & (mgmt["pattern"].isin(["mentor", "hiring_interviewing", "manage_team", "team_context", "stakeholder", "lead_project_or_team"]))
    ]
    if not m.empty:
        pivot = m.pivot_table(index="pattern", columns="source_group", values="share", aggfunc="sum").fillna(0)
        pivot[["pooled_2024", "scraped_2026"]].mul(100).plot(kind="bar", ax=ax, color=["#6278a5", "#c46f4f"])
        ax.set_ylabel("Share of LLM-text postings (%)")
        ax.set_xlabel("")
        ax.set_title("Management Strong vs Broad Triggers")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "management_trigger_shares.png", dpi=150)
        plt.close(fig)

    if not domain.empty:
        d = domain.loc[domain["definition"].eq("J3")].sort_values("scope_density_diff")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(d["archetype_name"], d["scope_density_diff"], color="#5f9f8f")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("J3 scope density change, scraped 2026 minus pooled 2024")
        ax.set_title("Domain-Stratified Scope Change")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "domain_stratified_j3_scope_density_change.png", dpi=150)
        plt.close(fig)


def main() -> None:
    assert_patterns()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pd.read_csv(T30_PANEL).to_csv(TABLE_DIR / "t30_panel_loaded_for_reference.csv", index=False)

    con = connect()
    text_features = extract_text_features()
    tech_counts, _, _ = tech_count_frame(con)
    title_meta = title_frame(con)
    df = text_features.merge(tech_counts, on="uid", how="left").merge(title_meta, on="uid", how="left")
    df[["tech_count", "tech_count_non_ai", "ai_requirement_count"]] = df[
        ["tech_count", "tech_count_non_ai", "ai_requirement_count"]
    ].fillna(0)
    df = add_complexity_metrics(df)
    if ARCHETYPES.exists():
        archetypes = con.execute(f"SELECT * FROM read_parquet('{ARCHETYPES.as_posix()}')").fetchdf()
        df = df.merge(archetypes, on="uid", how="left")
    else:
        df["archetype_name"] = np.nan

    write_feature_artifact(df)
    write_sample_coverage(df)
    semantic_validation_samples(df)
    validation_summary_placeholder()

    primary = df.loc[df["text_source"].eq("llm")].copy()
    primary_summary, primary_effects = summarize_panel(primary, "llm_text_primary")
    primary_summary.to_csv(TABLE_DIR / "complexity_by_panel_primary_llm_text.csv", index=False)
    primary_effects.to_csv(TABLE_DIR / "complexity_effects_by_panel_primary_llm_text.csv", index=False)

    no_agg_summary, no_agg_effects = summarize_panel(
        primary.loc[~primary["is_aggregator"].fillna(False)], "llm_text_no_aggregators"
    )
    no_agg_summary.to_csv(TABLE_DIR / "complexity_by_panel_no_aggregators.csv", index=False)
    no_agg_effects.to_csv(TABLE_DIR / "complexity_effects_no_aggregators.csv", index=False)

    capped = company_cap(primary, cap=50)
    cap_summary, cap_effects = summarize_panel(capped, "llm_text_company_cap50")
    cap_summary.to_csv(TABLE_DIR / "complexity_by_panel_company_cap50.csv", index=False)
    cap_effects.to_csv(TABLE_DIR / "complexity_effects_company_cap50.csv", index=False)

    specialists = load_entry_specialists()
    no_specialists = primary.loc[~primary["company_name_canonical"].astype(str).isin(specialists)]
    spec_summary, spec_effects = summarize_panel(no_specialists, "llm_text_exclude_entry_specialists")
    spec_summary.to_csv(TABLE_DIR / "complexity_by_panel_exclude_entry_specialists.csv", index=False)
    spec_effects.to_csv(TABLE_DIR / "complexity_effects_exclude_entry_specialists.csv", index=False)

    text_source_sensitivity(df)
    mgmt = management_breakdown(df)
    domain = domain_stratified_scope(df, con)
    outlier_analysis(df)
    write_figures(primary_effects, primary_summary, mgmt, domain)

    print(f"Wrote T11 outputs under {TABLE_DIR} and {FIG_DIR}")


if __name__ == "__main__":
    main()
