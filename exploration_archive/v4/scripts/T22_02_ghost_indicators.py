"""T22 step 2 — Compute per-posting ghost indicators.

For every SWE LinkedIn row (default filter) compute:
  * hedge_count            -- number of validated hedge pattern hits (0..K)
  * firm_count             -- number of validated firm requirement hits (0..K)
  * hedge_any, firm_any    -- 0/1 markers
  * aspiration_ratio       -- hedge_count / max(firm_count, 1)
  * n_distinct_tech        -- number of techs (from swe_tech_matrix, with
                              locally-corrected C++/C# overrides on raw text)
  * n_org_scope            -- number of org-scope terms (T11-strict)
  * kitchen_sink_score     -- n_distinct_tech * n_org_scope
  * yoe_extracted          -- pass through
  * yoe_scope_mismatch     -- 1 if (yoe_extracted>=5) for combined-entry OR
                              yoe_extracted>=5 for yoe-proxy-entry OR
                              (senior_scope_terms>=3)
  * ai_hedge_cnt, ai_firm_cnt -- hedge/firm hits *within 60 chars* of ai terms
  * mgmt_strict_any        -- any strict mgmt pattern (mentor/hire/directreports)
  * credential_impossible  -- (yoe>=10 AND entry-level) OR (no degree + ms required)
  * is_aggregator, is_entry_combined, is_entry_yoe, is_entry_native, is_entry_final
  * period, seniority columns, company, description_hash

Also saves a uid-indexed parquet under exploration/artifacts/T22/ for
downstream cross-correlation (e.g., with T29's LLM-authorship score).
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

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNIFIED = ROOT / "data" / "unified.parquet"
TECH_MATRIX = ROOT / "exploration" / "artifacts" / "shared" / "swe_tech_matrix.parquet"
PATTERN_JSON = ROOT / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json"
OUT_DIR = ROOT / "exploration" / "artifacts" / "T22"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_POSTING_PARQUET = OUT_DIR / "ghost_indicators_per_posting.parquet"

BASE_FILTER = (
    "source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE"
)

SENIORITY_CASE = """
  CASE
    WHEN llm_classification_coverage = 'labeled'         THEN seniority_llm
    WHEN llm_classification_coverage = 'rule_sufficient' THEN seniority_final
    ELSE NULL
  END AS seniority_best_available
"""

HEDGE_PATTERNS = {
    "nice_to_have": r"nice[- ]to[- ]have",
    "preferred_hedge": r"(skills?|experience|knowledge|familiarity|qualification|background)[^.\n]{0,30}(preferred|a plus)",
    "preferred_alt": r"preferred[^.\n]{0,30}(skills?|experience|qualification|but not required)",
    "bonus_plus": r"\ba plus\b|\bis a plus\b|would be a plus",
    "ideally": r"\bideally\b",
    "bonus_points": r"bonus points|bonus if|as a bonus",
    "familiarity_with": r"familiarity with",
    "exposure_to": r"\bexposure to\b",
}
FIRM_PATTERNS = {
    "must_have": r"must[- ]have|must have",
    "required_req": r"(experience|knowledge|skills?|degree|qualification)[^.\n]{0,10}(is |are )?required|(required[^.\n]{0,10})(experience|knowledge|skills?|qualifications?)",
    "minimum_req": r"\bminimum[- ]of\b|\bat[- ]minimum\b|\bminimum qualification",
    "mandatory_req": r"\bmandatory\b",
}
MGMT_STRICT_PATTERNS = {
    "strict_mentor": r"mentor (engineers?|juniors?|team|developers?|interns?|the team)|coach engineers?|mentor(ing|ship) (engineers?|juniors?|team|developers?)",
    "strict_people_mgr": r"direct reports?|\bpeople manager\b|\bpeople management\b|performance reviews?",
    "strict_hire_mgmt": r"hire engineers?|hiring engineers?|grow the team|build (the |a |out )?team|lead hiring|own hiring|manage a team of|lead a team of",
    "strict_headcount": r"\bheadcount\b|budget responsibility",
}
SENIOR_SCOPE_PATTERNS = {
    "architecture_scope": r"system design|distributed systems?|architect(ing|ure) (of |decisions|review)",
    "ownership_scope": r"\bend[- ]to[- ]end\b|own the (delivery|product|system|project|feature)|take ownership",
    "ownership_word": r"\bownership\b",
    "system_design": r"system design",
}
AI_PATTERNS = {
    "ai_tool": r"\bcopilot\b|\bcursor\b|\bllm(s)?\b|prompt engineering|\blangchain\b|ai pair program|model context protocol",
    "ai_domain": r"machine learning|deep learning|\bnlp\b|natural language processing|computer vision|model training|\btransformers?\b|\bembeddings?\b|fine[- ]tun(e|ing)",
    "ai_general": r"\bai\b|artificial intelligence",
    "agentic": r"\bagentic\b",
    "ai_agent_phrase": r"ai agents?|multi[- ]agent|autonomous agents?|agentic (ai|workflow)",
    "rag_phrase": r"\brag\b|retrieval augmented|retrieval[- ]augmented",
}
# no-degree phrasing
NO_DEGREE_RE = re.compile(
    r"no (degree|bachelor'?s?) (required|needed)|degree not required|without a degree|"
    r"equivalent (work )?experience in lieu of a degree",
    re.I,
)
MS_REQUIRED_RE = re.compile(
    r"(master'?s?|m\.?s\.?|m\.?sc|phd|ph\.?d|doctoral|doctorate)\b[^.]{0,40}\b(required|mandatory)",
    re.I,
)


def compile_patterns(d: dict[str, str]) -> dict[str, re.Pattern]:
    return {k: re.compile(v, re.I) for k, v in d.items()}


PY_HEDGE = compile_patterns(HEDGE_PATTERNS)
PY_FIRM = compile_patterns(FIRM_PATTERNS)
PY_MGMT = compile_patterns(MGMT_STRICT_PATTERNS)
PY_SCOPE = compile_patterns(SENIOR_SCOPE_PATTERNS)
PY_AI = compile_patterns(AI_PATTERNS)


def count_hits(text: str, pats: dict[str, re.Pattern]) -> tuple[int, dict]:
    if not isinstance(text, str) or not text:
        return 0, {k: 0 for k in pats}
    hits = {k: (1 if p.search(text) else 0) for k, p in pats.items()}
    return sum(hits.values()), hits


def ai_proximity_hedge_firm(text: str) -> tuple[int, int, int]:
    """Count hedge and firm markers within 80 chars of an AI term.

    Returns (ai_hedge_hits, ai_firm_hits, ai_windows).
    """
    if not isinstance(text, str) or not text:
        return 0, 0, 0
    # Union AI regex
    ai_union = re.compile(
        r"\bcopilot\b|\bcursor\b|\bllm(s)?\b|prompt engineering|\blangchain\b|"
        r"machine learning|deep learning|\bnlp\b|natural language processing|"
        r"computer vision|model training|\btransformers?\b|fine[- ]tun(e|ing)|"
        r"artificial intelligence|\bagentic\b|\brag\b|retrieval augmented|"
        r"ai agents?|multi[- ]agent",
        re.I,
    )
    hedge_union = re.compile(
        "|".join(f"(?:{p})" for p in HEDGE_PATTERNS.values()), re.I
    )
    firm_union = re.compile(
        "|".join(f"(?:{p})" for p in FIRM_PATTERNS.values()), re.I
    )
    ai_hedge = 0
    ai_firm = 0
    ai_wins = 0
    for m in ai_union.finditer(text):
        start = max(0, m.start() - 80)
        end = min(len(text), m.end() + 80)
        window = text[start:end]
        ai_wins += 1
        if hedge_union.search(window):
            ai_hedge += 1
        if firm_union.search(window):
            ai_firm += 1
    return ai_hedge, ai_firm, ai_wins


def load_frame(con) -> pd.DataFrame:
    q = f"""
    SELECT
      uid,
      source,
      period,
      title,
      company_name_canonical,
      company_industry,
      is_aggregator,
      description_hash,
      description_core,
      description_core_llm,
      description,
      yoe_extracted,
      yoe_min_extracted,
      yoe_max_extracted,
      seniority_final,
      seniority_native,
      {SENIORITY_CASE}
    FROM read_parquet('{UNIFIED}')
    WHERE {BASE_FILTER}
    """
    df = con.execute(q).fetchdf()
    df["period2"] = np.where(
        df["source"].isin(["kaggle_arshkon", "kaggle_asaniczka"]), "2024", "2026"
    )
    df["text_best"] = (
        df["description_core_llm"]
        .where(df["description_core_llm"].notna() & (df["description_core_llm"].str.len() > 0))
        .fillna(df["description_core"])
        .fillna(df["description"])
    )
    df["text_best_lc"] = df["text_best"].fillna("").str.lower()
    return df


def load_tech(con) -> pd.DataFrame:
    return con.execute(f"SELECT * FROM read_parquet('{TECH_MATRIX}')").fetchdf()


def corrected_tech_count(df: pd.DataFrame, tech: pd.DataFrame) -> pd.Series:
    """Return n_distinct_tech per uid, applying C++/C# corrections from text."""
    tech_cols = [c for c in tech.columns if c != "uid"]
    tech = tech.copy()
    tech["_base_count"] = tech[tech_cols].sum(axis=1)
    # The matrix has broken c_cpp/csharp indicators. Override from text.
    merged = df[["uid", "text_best_lc"]].merge(tech[["uid", "_base_count", "c_cpp", "csharp"]] if "c_cpp" in tech.columns else tech[["uid", "_base_count"]], on="uid", how="left")
    merged["_base_count"] = merged["_base_count"].fillna(0).astype(int)
    # Remove the broken indicators from the base count (if present)
    for col in ("c_cpp", "csharp"):
        if col in merged.columns:
            merged["_base_count"] -= merged[col].fillna(0).astype(int)
    # Add corrected
    merged["cpp_fix"] = merged["text_best_lc"].str.contains(
        r"c\+\+|\bc plus plus\b|\bcpp\b", regex=True, na=False
    ).astype(int)
    merged["csharp_fix"] = merged["text_best_lc"].str.contains(
        r"\bc#|\bc-sharp\b|\bcsharp\b", regex=True, na=False
    ).astype(int)
    merged["n_distinct_tech"] = merged["_base_count"] + merged["cpp_fix"] + merged["csharp_fix"]
    return merged.set_index("uid")["n_distinct_tech"]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for uid, text in zip(df["uid"].values, df["text_best_lc"].values):
        hedge_cnt, hedge_h = count_hits(text, PY_HEDGE)
        firm_cnt, firm_h = count_hits(text, PY_FIRM)
        mgmt_cnt, mgmt_h = count_hits(text, PY_MGMT)
        scope_cnt, scope_h = count_hits(text, PY_SCOPE)
        ai_cnt, ai_h = count_hits(text, PY_AI)
        ai_hedge, ai_firm, ai_wins = ai_proximity_hedge_firm(text or "")
        # non-AI hedge/firm = totals - ai-proximal (approximation)
        rec = {
            "uid": uid,
            "hedge_count": hedge_cnt,
            "firm_count": firm_cnt,
            "hedge_any": int(hedge_cnt > 0),
            "firm_any": int(firm_cnt > 0),
            "mgmt_strict_count": mgmt_cnt,
            "mgmt_strict_any": int(mgmt_cnt > 0),
            "scope_count": scope_cnt,
            "ai_count": ai_cnt,
            "ai_tool": ai_h["ai_tool"],
            "ai_domain": ai_h["ai_domain"],
            "ai_general": ai_h["ai_general"],
            "agentic": ai_h["agentic"],
            "ai_agent_phrase": ai_h["ai_agent_phrase"],
            "rag_phrase": ai_h["rag_phrase"],
            "hedge_nice_to_have": hedge_h["nice_to_have"],
            "hedge_preferred": hedge_h["preferred_hedge"] | hedge_h["preferred_alt"],
            "hedge_bonus_plus": hedge_h["bonus_plus"] | hedge_h["bonus_points"],
            "hedge_ideally": hedge_h["ideally"],
            "hedge_familiarity": hedge_h["familiarity_with"] | hedge_h["exposure_to"],
            "firm_must_have": firm_h["must_have"],
            "firm_required": firm_h["required_req"],
            "firm_minimum": firm_h["minimum_req"],
            "firm_mandatory": firm_h["mandatory_req"],
            "mgmt_mentor": mgmt_h["strict_mentor"],
            "mgmt_people_mgr": mgmt_h["strict_people_mgr"],
            "mgmt_hire_mgmt": mgmt_h["strict_hire_mgmt"],
            "scope_architecture": scope_h["architecture_scope"],
            "scope_ownership": scope_h["ownership_scope"] | scope_h["ownership_word"],
            "scope_sys_design": scope_h["system_design"],
            "ai_prox_hedge": ai_hedge,
            "ai_prox_firm": ai_firm,
            "ai_windows": ai_wins,
            "no_degree_phrase": int(bool(NO_DEGREE_RE.search(text or ""))),
            "ms_required_phrase": int(bool(MS_REQUIRED_RE.search(text or ""))),
        }
        records.append(rec)
    feats = pd.DataFrame.from_records(records)
    return feats


def main():
    print("Connecting and loading frame...")
    con = duckdb.connect()
    df = load_frame(con)
    print(f"  Loaded {len(df):,} rows")
    print("Loading tech matrix...")
    tech = load_tech(con)
    print(f"  Tech matrix: {len(tech):,} rows, {len(tech.columns)} cols")

    print("Computing corrected tech counts...")
    n_distinct_tech = corrected_tech_count(df, tech)

    print("Extracting text features...")
    feats = extract_features(df)

    merged = df.merge(feats, on="uid", how="left")
    merged["n_distinct_tech"] = merged["uid"].map(n_distinct_tech).fillna(0).astype(int)

    # Derived scores
    merged["aspiration_ratio"] = merged["hedge_count"] / merged["firm_count"].clip(lower=1)
    merged["kitchen_sink_score"] = merged["n_distinct_tech"] * merged["scope_count"]

    # Seniority operationalizations
    merged["is_entry_combined"] = (merged["seniority_best_available"] == "entry").astype(int)
    merged["is_entry_final"] = (merged["seniority_final"] == "entry").astype(int)
    merged["is_entry_native"] = (merged["seniority_native"] == "entry").astype(int)
    merged["is_entry_yoe"] = (merged["yoe_extracted"].fillna(-1) <= 2).astype(int)
    merged["is_entry_yoe_strict"] = (
        (merged["yoe_extracted"].notna()) & (merged["yoe_extracted"] <= 2)
    ).astype(int)

    # YOE-scope mismatch: entry-level posting with senior YOE or >=3 senior scope terms
    senior_scope_n = (
        merged["scope_architecture"].fillna(0)
        + merged["scope_ownership"].fillna(0)
        + merged["scope_sys_design"].fillna(0)
    )
    merged["senior_scope_n"] = senior_scope_n
    merged["yoe_ge5"] = (merged["yoe_extracted"].fillna(0) >= 5).astype(int)
    merged["yoe_scope_mismatch_combined"] = (
        (merged["is_entry_combined"] == 1)
        & ((merged["yoe_ge5"] == 1) | (senior_scope_n >= 3))
    ).astype(int)
    merged["yoe_scope_mismatch_yoe"] = (
        (merged["is_entry_yoe"] == 1)
        & ((merged["yoe_ge5"] == 1) | (senior_scope_n >= 3))
    ).astype(int)

    # Credential impossibility: entry + yoe>=10 OR no-degree-phrase AND ms-required-phrase
    merged["credential_contradiction"] = (
        ((merged["is_entry_combined"] == 1) & (merged["yoe_extracted"].fillna(0) >= 10)).astype(int)
        | ((merged["no_degree_phrase"] == 1) & (merged["ms_required_phrase"] == 1)).astype(int)
    )

    # Save slim per-posting artifact (for T29 cross-corr)
    keep_cols = [
        "uid",
        "source",
        "period",
        "period2",
        "company_name_canonical",
        "company_industry",
        "is_aggregator",
        "description_hash",
        "seniority_best_available",
        "seniority_final",
        "seniority_native",
        "yoe_extracted",
        "is_entry_combined",
        "is_entry_final",
        "is_entry_native",
        "is_entry_yoe",
        "hedge_count",
        "firm_count",
        "hedge_any",
        "firm_any",
        "aspiration_ratio",
        "n_distinct_tech",
        "scope_count",
        "kitchen_sink_score",
        "mgmt_strict_count",
        "mgmt_strict_any",
        "mgmt_mentor",
        "mgmt_people_mgr",
        "mgmt_hire_mgmt",
        "ai_count",
        "ai_tool",
        "ai_domain",
        "ai_general",
        "agentic",
        "ai_agent_phrase",
        "rag_phrase",
        "ai_prox_hedge",
        "ai_prox_firm",
        "ai_windows",
        "senior_scope_n",
        "yoe_ge5",
        "yoe_scope_mismatch_combined",
        "yoe_scope_mismatch_yoe",
        "credential_contradiction",
        "hedge_nice_to_have",
        "hedge_preferred",
        "hedge_bonus_plus",
        "hedge_ideally",
        "hedge_familiarity",
        "firm_must_have",
        "firm_required",
        "firm_minimum",
        "firm_mandatory",
    ]
    out = merged[keep_cols].copy()
    table = pa.Table.from_pandas(out, preserve_index=False)
    pq.write_table(table, PER_POSTING_PARQUET, compression="snappy")
    print(f"Wrote {PER_POSTING_PARQUET} ({len(out):,} rows)")

    # Re-save the validated patterns artifact with the final, documented set
    artifact = {
        "schema_version": 2,
        "note": (
            "Precision-validated regex patterns from T22_01 sampling (50 per pattern). "
            "All patterns reviewed; precision estimated >=90% for all listed. "
            "MCP removed from ai_tool (Microsoft Certified Professional contamination); "
            "use 'model context protocol' as the firm signal instead."
        ),
        "hedge": HEDGE_PATTERNS,
        "firm": FIRM_PATTERNS,
        "mgmt_strict": MGMT_STRICT_PATTERNS,
        "senior_scope": SENIOR_SCOPE_PATTERNS,
        "ai": AI_PATTERNS,
        "precision_review": {
            "preferred_hedge": ">=95% (validated hedging in qualification context)",
            "preferred_alt": ">=95%",
            "bonus_plus": ">=95%",
            "nice_to_have": ">=98%",
            "ideally": ">=95%",
            "familiarity_with": ">=95% (hedged capability lang)",
            "must_have": ">=95%",
            "required_req": ">=95% (validated requirement lang)",
            "minimum_req": ">=98%",
            "mandatory_req": ">=90% (some 'mandatory access control' noise)",
            "strict_mentor": ">=95% (validated previously by T11/V1 sampling)",
            "strict_people_mgr": ">=95%",
            "strict_hire_mgmt": ">=90% ('lead a team of' captures genuine mgmt)",
            "agentic": "~95% (V1 confirmed)",
            "ai_general": "~90% (\\bai\\b is mostly AI context; artificial intelligence is 100%)",
            "ai_tool": ">=95% after removing bare 'mcp'",
            "ai_domain": ">=98%",
            "rag_phrase": ">=95% (2026); few FP from RAG as other acronym",
        },
    }
    with PATTERN_JSON.open("w") as f:
        json.dump(artifact, f, indent=2)
    print(f"Updated {PATTERN_JSON}")


if __name__ == "__main__":
    main()
