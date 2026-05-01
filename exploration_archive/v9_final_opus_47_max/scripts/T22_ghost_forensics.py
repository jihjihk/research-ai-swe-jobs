"""
T22. Ghost & aspirational requirements forensics.

Pipeline:
1) Ghost indicators per posting: kitchen-sink score, aspiration ratio,
   YOE-scope mismatch, template saturation, credential impossibility.
2) Prevalence by period x seniority (T30 panel).
3) AI ghostiness: aspiration ratio within AI vs non-AI postings.
4) 20 most ghost-like entry-level postings.
5) Aggregator vs direct (PRIMARY axis).
6) Industry patterns (within-period only).
7) Pattern validation pipeline -> updated validated_mgmt_patterns.json.

Inputs:
  data/unified_core.parquet
  exploration/artifacts/shared/swe_cleaned_text.parquet
  exploration/artifacts/shared/T11_posting_features.parquet
  exploration/artifacts/shared/swe_tech_matrix.parquet
  exploration/artifacts/shared/validated_mgmt_patterns.json (V1 seed)

Outputs:
  exploration/tables/T22/ghost_indicators.parquet (per-posting)
  exploration/tables/T22/prevalence_by_t30.csv
  exploration/tables/T22/aggregator_vs_direct.csv
  exploration/tables/T22/industry_patterns.csv
  exploration/tables/T22/top20_ghost_entry.csv
  exploration/tables/T22/ai_ghostiness.csv
  exploration/tables/T22/pattern_validation_samples.csv
  exploration/artifacts/shared/validated_mgmt_patterns.json (extended)

Memory: chunked via duckdb; peak RSS target ~4-6 GB.
"""
from __future__ import annotations
import duckdb
import json
import re
import random
import math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

REPO = Path("/home/jihgaboot/gabor/job-research")
DATA = REPO / "data" / "unified_core.parquet"
CLEANED = REPO / "exploration" / "artifacts" / "shared" / "swe_cleaned_text.parquet"
T11 = REPO / "exploration" / "artifacts" / "shared" / "T11_posting_features.parquet"
TECH = REPO / "exploration" / "artifacts" / "shared" / "swe_tech_matrix.parquet"
VAL_JSON = REPO / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json"
OUT = REPO / "exploration" / "tables" / "T22"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(22)
np.random.seed(22)

# --------------------------------------------------------------------------------------
# Patterns (V1-validated)
# --------------------------------------------------------------------------------------
# V1-validated
SCOPE_V1 = re.compile(
    r"\b(ownership|end[\s\-]to[\s\-]end|cross[\s\-]functional|initiative(?:s)?|stakeholder(?:s)?)\b",
    re.IGNORECASE,
)
AI_STRICT_V1 = re.compile(
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|"
    r"langchain|prompt engineering|rag|vector databas(?:e|es)|pinecone|huggingface|"
    r"hugging face|(?:fine[- ]tun(?:e|ed|ing))\s+(?:the\s+)?(?:model|llm|gpt|"
    r"base model|foundation model|embeddings))\b",
    re.IGNORECASE,
)

# T22 new: hedging / aspiration
HEDGING = re.compile(
    r"\b(ideally|nice to have|nice-to-have|preferred|bonus|a plus|would be (?:a )?plus|"
    r"added bonus|bonus points|would love|bonus skill|good to have|"
    r"pluses|pluses:|pluses\.|plus:|plus\.)\b",
    re.IGNORECASE,
)
# Firm requirement language
# V1 (T22) initial pattern precision 0.54; "requirements/requirement" was document-noun in most
# matches ("requirements gathering", "customer requirements"). Rebuilt pattern drops plain
# "requirement(s)" and keeps only unambiguous firm-qualification tokens. Precision expected ~0.90.
FIRM = re.compile(
    r"\b(must have|must-have|must possess|must demonstrate|must be able|"
    r"shall be able|need to have|need to demonstrate|we require|is required|are required|"
    r"required experience|required qualifications|required skills|required abilities|"
    r"required to|required for|required by|"
    r"minimum qualifications|minimum requirements|minimum experience|"
    r"mandatory|"
    r"at least \d+|must be proficient|must be familiar|must demonstrate|must hold)\b",
    re.IGNORECASE,
)

# Kitchen-sink scope terms (expanded beyond V1 scope)
SCOPE_KITCHEN_SINK = re.compile(
    r"\b(ownership|end[\s\-]to[\s\-]end|cross[\s\-]functional|initiative(?:s)?|"
    r"stakeholder(?:s)?|architect(?:ure|ing|ed)?|system design|distributed system(?:s)?|"
    r"scalability|scalable|high[- ]throughput|multi[- ]tenant|"
    r"greenfield|roadmap|strategy|strategic|vision|lead(?:ing)? (?:the|engineering|design)|"
    r"drive (?:the|engineering|adoption|design|initiative|strategy)|define (?:the|strategy|vision|roadmap)|"
    r"influence|shape|platform(?:s)?|enterprise[- ]scale|org[- ]wide|cross[- ]team|"
    r"end user|product vision|business outcome(?:s)?|stakeholder management|executive|c[- ]suite)\b",
    re.IGNORECASE,
)
# Senior scope term set (for YOE-scope mismatch) — expanded per dispatch
SENIOR_SCOPE_TERMS = [
    re.compile(r"\barchitect(?:ure|ing|ed|s)?\b", re.IGNORECASE),
    re.compile(r"\bownership\b", re.IGNORECASE),
    re.compile(r"\bsystem design\b", re.IGNORECASE),
    re.compile(r"\bdistributed system(?:s)?\b", re.IGNORECASE),
    re.compile(r"\b(?:scalab(?:le|ility))\b", re.IGNORECASE),
    re.compile(r"\bcross[\s\-]functional\b", re.IGNORECASE),
    re.compile(r"\bend[\s\-]to[\s\-]end\b", re.IGNORECASE),
    re.compile(r"\bmentor(?:s|ing|ed)?\b", re.IGNORECASE),
    re.compile(r"\blead(?:ing)? (?:the|engineering|design|team|initiative)\b", re.IGNORECASE),
    re.compile(r"\btechnical leadership\b", re.IGNORECASE),
    re.compile(r"\bstrategy\b", re.IGNORECASE),
    re.compile(r"\broadmap\b", re.IGNORECASE),
]

# Credential / degree patterns
DEG_MS = re.compile(r"\b(master'?s|m\.?s\.?\b|msc|mba|m\.eng|m\.s\.? (?:in|degree|required))\b", re.IGNORECASE)
DEG_BS = re.compile(r"\b(bachelor'?s|b\.?s\.?\b|bs degree|ba degree|undergraduate degree)\b", re.IGNORECASE)
DEG_NONE = re.compile(r"\b(no degree (?:required|necessary)|degree not required|without (?:a )?degree|"
                      r"self[- ]taught|equivalent experience)\b", re.IGNORECASE)
DEG_PHD = re.compile(r"\b(ph\.?d|doctorate|doctoral)\b", re.IGNORECASE)

# YOE regex ablation (rule-based; find any "X+ years" or "at least X years")
YOE_RULE = re.compile(r"\b(\d{1,2})\+?\s*(?:-\s*\d{1,2}\s*)?years?\b", re.IGNORECASE)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def count_regex(rx: re.Pattern, text: str) -> int:
    if not text:
        return 0
    return len(rx.findall(text))


def any_regex(rx: re.Pattern, text: str) -> bool:
    if not text:
        return False
    return rx.search(text) is not None


def senior_scope_count(text: str) -> int:
    if not text:
        return 0
    return sum(1 for rx in SENIOR_SCOPE_TERMS if rx.search(text) is not None)


# --------------------------------------------------------------------------------------
# Main build
# --------------------------------------------------------------------------------------
def main():
    con = duckdb.connect()

    # Build base cohort: SWE LinkedIn English ok.
    # Join unified_core -> cleaned text (cleaned text is the text used for patterns).
    # Also join T11 features (tech_count, requirement_breadth, ai_binary).
    q = f"""
    SELECT
      u.uid,
      u.source,
      u.period,
      u.title,
      u.company_name_canonical,
      u.is_aggregator,
      u.company_industry,
      u.seniority_final,
      u.seniority_3level,
      u.yoe_min_years_llm,
      u.yoe_extracted,
      u.ghost_assessment_llm,
      u.ghost_job_risk,
      u.description_core_llm,
      u.description AS description_raw,
      u.description_length,
      u.llm_extraction_coverage,
      u.llm_classification_coverage,
      ct.description_cleaned,
      ct.text_source,
      t11.tech_count,
      t11.requirement_breadth,
      t11.requirement_breadth_resid,
      t11.credential_stack_depth,
      t11.credential_stack_depth_resid,
      t11.ai_binary,
      t11.education_level
    FROM read_parquet('{DATA}') u
    LEFT JOIN read_parquet('{CLEANED}') ct ON u.uid = ct.uid
    LEFT JOIN read_parquet('{T11}') t11 ON u.uid = t11.uid
    WHERE u.source_platform='linkedin' AND u.is_english=true AND u.date_flag='ok' AND u.is_swe=true
    """
    print("Loading base cohort...")
    df = con.execute(q).fetchdf()
    print(f"Base cohort: {len(df):,} rows")
    # Choose text: cleaned > raw
    def pick_text(r):
        if isinstance(r.description_cleaned, str) and r.description_cleaned:
            return r.description_cleaned
        if isinstance(r.description_core_llm, str) and r.description_core_llm:
            return r.description_core_llm
        if isinstance(r.description_raw, str) and r.description_raw:
            return r.description_raw
        return ""
    df["text"] = df.apply(pick_text, axis=1)
    df["text_len"] = df["text"].str.len().fillna(0).astype(int)

    # --- Kitchen-sink score ---
    # scope term count * tech count
    print("Computing kitchen-sink scope counts...")
    df["scope_kitchen_count"] = df["text"].apply(lambda t: count_regex(SCOPE_KITCHEN_SINK, t))
    # Use T11 tech_count (already joined)
    df["kitchen_sink_score"] = df["tech_count"].fillna(0) * df["scope_kitchen_count"].fillna(0)

    # --- Aspiration ratio ---
    print("Computing hedging / firm counts (aspiration ratio)...")
    df["hedging_count"] = df["text"].apply(lambda t: count_regex(HEDGING, t))
    df["firm_count"] = df["text"].apply(lambda t: count_regex(FIRM, t))
    # avoid div-by-zero: add 1 to firm as smoothing, OR define as hedging/(hedging+firm)
    denom = df["hedging_count"] + df["firm_count"]
    df["aspiration_share"] = np.where(denom > 0, df["hedging_count"] / denom, np.nan)
    df["aspiration_ratio"] = np.where(df["firm_count"] > 0,
                                       df["hedging_count"] / df["firm_count"],
                                       np.nan)
    df["high_aspiration"] = (df["aspiration_share"] >= 0.4).astype("Int64")  # >=40% hedging
    # has_any_firm for defensibility
    df["has_hedge"] = (df["hedging_count"] > 0).astype(int)
    df["has_firm"] = (df["firm_count"] > 0).astype(int)

    # --- YOE-scope mismatch ---
    print("Computing YOE-scope mismatch indicators...")
    df["senior_scope_n"] = df["text"].apply(senior_scope_count)
    # J3 set: yoe_min_years_llm <= 2 (LLM path)
    df["is_j3_llm"] = (df["yoe_min_years_llm"] <= 2).fillna(False)
    # Rule ablation: yoe_extracted
    df["is_j_rule"] = (df["yoe_extracted"] <= 2).fillna(False)
    # Entry-label flag: seniority_final == 'entry'
    df["is_entry_label"] = (df["seniority_final"] == "entry")
    # For yoe_scope_mismatch: entry posting that requests senior scope (>=3 senior-scope terms)
    # Primary: LLM yoe<=2 OR entry label; with >=3 distinct senior-scope terms from 12-term list
    df["yoe_scope_mismatch_llm"] = ((df["is_j3_llm"] | df["is_entry_label"]) & (df["senior_scope_n"] >= 3)).astype(int)
    # Rule ablation: uses rule-based yoe<=2
    df["yoe_scope_mismatch_rule"] = ((df["is_j_rule"] | df["is_entry_label"]) & (df["senior_scope_n"] >= 3)).astype(int)
    # Stricter flag for interest
    df["yoe_scope_mismatch_strict"] = ((df["is_j3_llm"] | df["is_entry_label"]) & (df["senior_scope_n"] >= 5)).astype(int)

    # --- Credential impossibility (contradictory) ---
    # Note: YOE numbers are redacted in the cleaned text. Use raw description for YOE rule.
    print("Computing credential-impossibility flags (using raw description for YOE rule)...")
    def yoe_rule_max(t):
        if not isinstance(t, str) or not t:
            return 0
        nums = [int(m) for m in YOE_RULE.findall(t) if m.isdigit()]
        if not nums:
            return 0
        return max(nums)
    df["yoe_rule_max"] = df["description_raw"].apply(yoe_rule_max)
    # Contradictory: entry label and >=7 YOE in text
    df["cred_imp_entry_7y"] = ((df["is_entry_label"]) & (df["yoe_rule_max"] >= 7)).astype(int)
    df["cred_imp_entry_10y"] = ((df["is_entry_label"]) & (df["yoe_rule_max"] >= 10)).astype(int)
    # Also: J3 (LLM yoe<=2) with raw text containing "10+ years"
    df["cred_imp_j3_10y"] = ((df["is_j3_llm"]) & (df["yoe_rule_max"] >= 10)).astype(int)
    # Contradictory: "no degree" and "MS required"
    df["has_ms"] = df["text"].apply(lambda t: any_regex(DEG_MS, t)).astype(int)
    df["has_nodeg"] = df["text"].apply(lambda t: any_regex(DEG_NONE, t)).astype(int)
    df["cred_imp_nodeg_ms"] = ((df["has_nodeg"] == 1) & (df["has_ms"] == 1)).astype(int)

    # Ghost-likely via LLM classifier
    df["ghost_llm_inflated"] = (df["ghost_assessment_llm"] == "inflated").astype(int)
    df["ghost_llm_ghost_likely"] = (df["ghost_assessment_llm"] == "ghost_likely").astype(int)
    df["ghost_llm_any_not_realistic"] = (df["ghost_assessment_llm"].isin(["inflated", "ghost_likely"])).astype(int)
    df["ghost_rule_medhi"] = (df["ghost_job_risk"].isin(["medium", "high"])).astype(int)

    # Persist per-posting ghost indicators
    persist_cols = [
        "uid", "source", "period", "title", "company_name_canonical",
        "is_aggregator", "company_industry",
        "seniority_final", "seniority_3level",
        "yoe_min_years_llm", "yoe_extracted",
        "ghost_assessment_llm", "ghost_job_risk",
        "text_len", "tech_count", "requirement_breadth",
        "scope_kitchen_count", "kitchen_sink_score",
        "hedging_count", "firm_count", "aspiration_share", "aspiration_ratio",
        "high_aspiration", "senior_scope_n",
        "is_j3_llm", "is_j_rule", "is_entry_label",
        "yoe_scope_mismatch_llm", "yoe_scope_mismatch_rule", "yoe_scope_mismatch_strict",
        "yoe_rule_max",
        "has_ms", "has_nodeg",
        "cred_imp_entry_7y", "cred_imp_entry_10y", "cred_imp_j3_10y", "cred_imp_nodeg_ms",
        "ghost_llm_inflated", "ghost_llm_ghost_likely", "ghost_llm_any_not_realistic", "ghost_rule_medhi",
        "ai_binary", "education_level",
    ]
    gi = df[persist_cols].copy()
    gi.to_parquet(OUT / "ghost_indicators.parquet")
    print(f"Wrote {OUT/'ghost_indicators.parquet'}")

    # --- Prevalence by period x T30 seniority cells ---
    # T30 panel primary: J3 (yoe<=2) for junior, S4 (yoe>=5) for senior.
    # Define cohorts:
    # pooled-2024 = arshkon + asaniczka (2024-01 + 2024-04)
    # arshkon-only = arshkon
    # scraped = 2026-03 + 2026-04
    df["is_2024"] = df["source"].isin(["kaggle_arshkon", "kaggle_asaniczka"])
    df["is_arshkon"] = df["source"] == "kaggle_arshkon"
    df["is_scraped"] = df["source"] == "scraped"
    df["s4_llm"] = (df["yoe_min_years_llm"] >= 5).fillna(False)

    rows = []
    ghost_metrics = {
        "ghost_llm_inflated": gi["ghost_llm_inflated"],
        "ghost_llm_any_not_realistic": gi["ghost_llm_any_not_realistic"],
        "ghost_rule_medhi": gi["ghost_rule_medhi"],
        "yoe_scope_mismatch_llm": gi["yoe_scope_mismatch_llm"],
        "yoe_scope_mismatch_rule": gi["yoe_scope_mismatch_rule"],
        "cred_imp_entry_10y": gi["cred_imp_entry_10y"],
        "cred_imp_nodeg_ms": gi["cred_imp_nodeg_ms"],
        "high_aspiration": gi["high_aspiration"].fillna(0).astype(int),
    }
    cont_metrics = {
        "kitchen_sink_score": gi["kitchen_sink_score"],
        "aspiration_share": gi["aspiration_share"],
        "hedging_count": gi["hedging_count"],
        "firm_count": gi["firm_count"],
        "scope_kitchen_count": gi["scope_kitchen_count"],
    }

    cohorts = [
        ("2024_pooled", df[df["is_2024"]]),
        ("2024_arshkon_only", df[df["is_arshkon"]]),
        ("2026_scraped", df[df["is_scraped"]]),
    ]
    sen_cells = [
        ("all", None),
        ("J3_yoe_le2", "is_j3_llm"),
        ("S4_yoe_ge5", "s4_llm"),
        ("entry_label", "is_entry_label"),
    ]
    for cname, sub in cohorts:
        for sname, flag in sen_cells:
            if flag is None:
                s = sub
            else:
                s = sub[sub[flag] == True]
            if len(s) == 0:
                continue
            row = {"cohort": cname, "seniority_cell": sname, "n": len(s)}
            for m in ["ghost_llm_inflated", "ghost_llm_any_not_realistic",
                      "ghost_rule_medhi", "yoe_scope_mismatch_llm", "yoe_scope_mismatch_rule",
                      "yoe_scope_mismatch_strict",
                      "cred_imp_entry_7y", "cred_imp_entry_10y", "cred_imp_j3_10y",
                      "cred_imp_nodeg_ms", "high_aspiration"]:
                row[f"{m}_rate"] = float(s[m].fillna(0).mean())
            for m in ["kitchen_sink_score", "aspiration_share", "hedging_count",
                      "firm_count", "scope_kitchen_count"]:
                row[f"{m}_mean"] = float(s[m].astype(float).mean())
                row[f"{m}_median"] = float(s[m].astype(float).median())
            rows.append(row)
    prev = pd.DataFrame(rows)
    prev.to_csv(OUT / "prevalence_by_t30.csv", index=False)
    print(f"Wrote {OUT/'prevalence_by_t30.csv'}")

    # --- AI ghostiness: is AI requirement language more aspirational than non-AI? ---
    # Approach: within each posting, count hedging-near-AI-term windows vs firm-near-AI-term windows
    # Simpler: split postings into AI-mention (ai_binary=true) vs not; compute aspiration metrics on each.
    # Then: within AI postings, segment text around AI tokens (+/- 100 chars) vs rest-of-text.
    print("Computing AI-ghostiness windows...")
    def near_ai_aspiration(text: str):
        """Return (hedge_near_ai, firm_near_ai, hedge_far, firm_far)."""
        if not text:
            return (0, 0, 0, 0)
        # find AI matches
        ai_spans = [(m.start(), m.end()) for m in AI_STRICT_V1.finditer(text)]
        if not ai_spans:
            # All text is "far" (non-AI)
            h = count_regex(HEDGING, text)
            f = count_regex(FIRM, text)
            return (0, 0, h, f)
        # build union of windows +/- 100 chars
        win = 100
        intervals = []
        for s, e in ai_spans:
            intervals.append((max(0, s-win), min(len(text), e+win)))
        intervals.sort()
        merged = []
        for s, e in intervals:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        in_text = "".join(text[s:e] for s, e in merged)
        hedge_in = count_regex(HEDGING, in_text)
        firm_in = count_regex(FIRM, in_text)
        hedge_all = count_regex(HEDGING, text)
        firm_all = count_regex(FIRM, text)
        return (hedge_in, firm_in, max(0, hedge_all - hedge_in), max(0, firm_all - firm_in))

    near_cols = df["text"].apply(near_ai_aspiration)
    df["h_near_ai"] = near_cols.apply(lambda t: t[0])
    df["f_near_ai"] = near_cols.apply(lambda t: t[1])
    df["h_far_ai"] = near_cols.apply(lambda t: t[2])
    df["f_far_ai"] = near_cols.apply(lambda t: t[3])

    # Rebuild cohorts to include the newly-added near-AI columns
    cohorts_with_near = [
        ("2024_pooled", df[df["is_2024"]]),
        ("2024_arshkon_only", df[df["is_arshkon"]]),
        ("2026_scraped", df[df["is_scraped"]]),
    ]
    ai_rows = []
    for cname, sub in cohorts_with_near:
        sub_ai = sub[sub["ai_binary"].fillna(False) == True]
        if len(sub_ai) == 0:
            continue
        # aggregate totals
        row = {
            "cohort": cname,
            "n_ai_postings": int(len(sub_ai)),
            "hedge_near_ai_total": int(sub_ai["h_near_ai"].sum()),
            "firm_near_ai_total": int(sub_ai["f_near_ai"].sum()),
            "hedge_far_ai_total": int(sub_ai["h_far_ai"].sum()),
            "firm_far_ai_total": int(sub_ai["f_far_ai"].sum()),
        }
        # aspiration share within AI vs outside
        nh = row["hedge_near_ai_total"]; nf = row["firm_near_ai_total"]
        fh = row["hedge_far_ai_total"]; ff = row["firm_far_ai_total"]
        row["aspiration_share_near_ai"] = nh / max(1, (nh+nf))
        row["aspiration_share_far_ai"]  = fh / max(1, (fh+ff))
        row["diff_share_near_minus_far"] = row["aspiration_share_near_ai"] - row["aspiration_share_far_ai"]
        ai_rows.append(row)
    ai_prev = pd.DataFrame(ai_rows)
    ai_prev.to_csv(OUT / "ai_ghostiness.csv", index=False)
    print(f"Wrote {OUT/'ai_ghostiness.csv'}")

    # --- Aggregator vs direct (PRIMARY axis) ---
    print("Aggregator vs direct...")
    agg_rows = []
    for cname, sub in cohorts:
        for is_agg in [False, True]:
            s = sub[sub["is_aggregator"] == is_agg]
            if len(s) == 0:
                continue
            row = {
                "cohort": cname,
                "is_aggregator": bool(is_agg),
                "n": int(len(s)),
            }
            for m in ["ghost_llm_inflated", "ghost_llm_any_not_realistic",
                      "ghost_rule_medhi", "yoe_scope_mismatch_llm", "yoe_scope_mismatch_rule",
                      "yoe_scope_mismatch_strict",
                      "cred_imp_entry_7y", "cred_imp_entry_10y", "cred_imp_j3_10y",
                      "cred_imp_nodeg_ms", "high_aspiration"]:
                row[f"{m}_rate"] = float(s[m].fillna(0).mean())
            for m in ["kitchen_sink_score", "aspiration_share", "hedging_count",
                      "firm_count", "scope_kitchen_count", "tech_count"]:
                row[f"{m}_mean"] = float(s[m].astype(float).mean())
                row[f"{m}_median"] = float(s[m].astype(float).median())
            agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(OUT / "aggregator_vs_direct.csv", index=False)
    print(f"Wrote {OUT/'aggregator_vs_direct.csv'}")

    # --- Industry patterns (within-period only) ---
    # Keep only cohorts with industry (arshkon + scraped). 2024-arshkon and 2026-scraped as within-period cells.
    print("Industry patterns (within-period only)...")
    ind_rows = []
    for cname, sub in [("2024_arshkon_only", df[df["is_arshkon"] & df["company_industry"].notna()]),
                       ("2026_scraped", df[df["is_scraped"] & df["company_industry"].notna()])]:
        # top industries by n
        top_ind = sub["company_industry"].value_counts().head(12).index.tolist()
        for ind in top_ind:
            s = sub[sub["company_industry"] == ind]
            if len(s) < 20:
                continue
            row = {
                "cohort": cname,
                "industry": ind,
                "n": int(len(s)),
            }
            for m in ["ghost_llm_inflated", "ghost_llm_any_not_realistic",
                      "yoe_scope_mismatch_llm", "high_aspiration"]:
                row[f"{m}_rate"] = float(s[m].fillna(0).mean())
            row["kitchen_sink_score_mean"] = float(s["kitchen_sink_score"].mean())
            ind_rows.append(row)
    ind = pd.DataFrame(ind_rows)
    ind.to_csv(OUT / "industry_patterns.csv", index=False)
    print(f"Wrote {OUT/'industry_patterns.csv'}")

    # --- 20 most ghost-like entry-level postings ---
    # Entry = J3_yoe_le2 OR is_entry_label; rank by a composite score
    print("Top 20 ghost-like entry postings...")
    df["_entry_flag"] = (df["is_j3_llm"] | df["is_entry_label"])
    e = df[df["_entry_flag"] == True].copy()
    # Composite: z(kitchen_sink) + z(senior_scope_n) + high_aspiration + yoe_scope_mismatch_llm*2 + cred_imp_entry_10y*3 + ghost_llm_any_not_realistic*2
    def z(x):
        x = pd.Series(x).astype(float)
        m, s = x.mean(), x.std()
        if s == 0 or np.isnan(s):
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - m) / s
    e["ghost_score"] = (
        z(e["kitchen_sink_score"]).values
        + z(e["senior_scope_n"]).values
        + e["high_aspiration"].fillna(0).astype(float).values
        + 2 * e["yoe_scope_mismatch_llm"].astype(float).values
        + 3 * e["cred_imp_entry_7y"].astype(float).values
        + 3 * e["cred_imp_entry_10y"].astype(float).values
        + 2 * e["ghost_llm_any_not_realistic"].astype(float).values
    )
    top20 = e.sort_values("ghost_score", ascending=False).head(20)
    cols = ["uid", "source", "period", "title", "company_name_canonical", "is_aggregator",
            "company_industry", "seniority_final", "yoe_min_years_llm",
            "ghost_assessment_llm", "tech_count", "scope_kitchen_count", "senior_scope_n",
            "kitchen_sink_score", "hedging_count", "firm_count", "aspiration_share",
            "yoe_rule_max", "cred_imp_entry_10y", "cred_imp_nodeg_ms", "ghost_score"]
    top20[cols].to_csv(OUT / "top20_ghost_entry_meta.csv", index=False)
    # Full text for inspection
    top20_full = top20[cols + ["text"]].copy()
    # Truncate text to 6000 chars to keep file small
    top20_full["text_sample"] = top20_full["text"].fillna("").str.slice(0, 6000)
    top20_full.drop(columns=["text"]).to_csv(OUT / "top20_ghost_entry.csv", index=False)
    print(f"Wrote {OUT/'top20_ghost_entry.csv'}")

    # --- Template saturation ---
    # For companies with >=10 postings in a period, compute mean pairwise cosine similarity on
    # requirements sections. Flag companies with mean similarity > 0.8.
    # We use cleaned_text; for compute: TF-IDF with sklearn, then cosine mean over pairs per company.
    print("Template saturation (companies with >=10 postings per period)...")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Build per-company-per-period groups
        df["_period_bin"] = df["source"].map({"kaggle_arshkon": "2024", "kaggle_asaniczka": "2024", "scraped": "2026"})
        tmpl = df[df["text_len"] >= 200].copy()
        grp = tmpl.groupby(["_period_bin", "company_name_canonical"])
        sizes = grp.size().reset_index(name="n")
        big = sizes[sizes["n"] >= 10]
        print(f"  {len(big)} company-period cells with >=10 postings; computing sims...")
        results = []
        # Limit to top 400 by n to keep memory bounded
        big = big.sort_values("n", ascending=False).head(400)
        # Build corpus per cell
        for _, row in big.iterrows():
            pb = row["_period_bin"]
            cn = row["company_name_canonical"]
            sub = tmpl[(tmpl["_period_bin"] == pb) & (tmpl["company_name_canonical"] == cn)]
            texts = sub["text"].fillna("").tolist()
            if len(texts) < 2:
                continue
            vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,1))
            try:
                X = vec.fit_transform(texts)
                sim = cosine_similarity(X)
                # mean off-diagonal
                n = sim.shape[0]
                mask = np.ones_like(sim, dtype=bool)
                np.fill_diagonal(mask, False)
                mean_sim = float(sim[mask].mean())
                median_sim = float(np.median(sim[mask]))
                results.append({
                    "period": pb,
                    "company_name_canonical": cn,
                    "n_postings": int(n),
                    "mean_pairwise_cosine": mean_sim,
                    "median_pairwise_cosine": median_sim,
                    "saturation_flag": bool(mean_sim > 0.8),
                })
            except Exception as ex:
                continue
        tmpldf = pd.DataFrame(results).sort_values(["period","mean_pairwise_cosine"], ascending=[True, False])
        tmpldf.to_csv(OUT / "template_saturation.csv", index=False)
        # top flagged companies
        flagged = tmpldf[tmpldf["saturation_flag"] == True]
        flagged.to_csv(OUT / "template_saturation_flagged.csv", index=False)
        print(f"Wrote template_saturation ({len(tmpldf)} rows; flagged={len(flagged)})")
    except Exception as ex:
        print("template saturation skipped:", ex)

    # --- Pattern validation samples: pick stratified samples for V1 rebuilt + new patterns ---
    print("Sampling pattern-validation rows...")
    def match_idx(rx, limit_each=25):
        """Return uid,period_bin,match_context for up to 25 2024 + 25 2026."""
        out = []
        for pb, sub in df.groupby("_period_bin" if "_period_bin" in df.columns else df["source"].apply(lambda s: "2024" if s.startswith("kaggle") else "2026")):
            mask = sub["text"].fillna("").str.contains(rx.pattern, regex=True, case=False, na=False)
            matches = sub[mask]
            if len(matches) == 0:
                continue
            samp = matches.sample(n=min(limit_each, len(matches)), random_state=22)
            for _, r in samp.iterrows():
                t = r["text"] or ""
                m = rx.search(t)
                if not m:
                    continue
                start = max(0, m.start() - 120)
                end = min(len(t), m.end() + 120)
                ctx = t[start:end]
                out.append({
                    "period_bin": pb,
                    "uid": r["uid"],
                    "source": r["source"],
                    "match": m.group(0),
                    "context": ctx.replace("\n", " ")[:350],
                })
        return out

    # Make sure _period_bin is set
    if "_period_bin" not in df.columns:
        df["_period_bin"] = df["source"].map({"kaggle_arshkon": "2024", "kaggle_asaniczka": "2024", "scraped": "2026"})

    samples = {}
    samples["scope_kitchen_sink"] = match_idx(SCOPE_KITCHEN_SINK)
    samples["hedging"] = match_idx(HEDGING)
    samples["firm"] = match_idx(FIRM)
    # V1-rebuilt re-validation
    MGMT_V1_REB = re.compile(
        r"\b(?:mentor(?:s|ed|ing)? (?:junior|engineers?|team(?:s)?|others|the team|engineering|peers|sd(?:e|es))|"
        r"coach(?:es|ed|ing)? (?:team|engineers?|junior|peers)|direct reports?|headcount|hiring manager|hiring decisions?)\b",
        re.IGNORECASE)
    AI_V1_REB = AI_STRICT_V1  # already same
    SCOPE_V1_REB = SCOPE_V1
    samples["mgmt_strict_v1_rebuilt"] = match_idx(MGMT_V1_REB)
    samples["ai_strict_v1_rebuilt"] = match_idx(AI_V1_REB)
    samples["scope_v1_rebuilt"] = match_idx(SCOPE_V1_REB)
    # SENIOR_SCOPE_TERMS: we use SCOPE_V1 as it overlaps
    # YOE rule for yoe_scope_mismatch: validate by re-checking "10+ years" patterns
    samples["yoe_rule_10plus"] = match_idx(re.compile(r"\b(?:10|11|12|15|20)\+?\s*years?\b", re.IGNORECASE))

    # Save samples for inspection
    flat_rows = []
    for pname, rs in samples.items():
        for r in rs:
            r2 = dict(r); r2["pattern_name"] = pname
            flat_rows.append(r2)
    samp_df = pd.DataFrame(flat_rows)
    samp_df.to_csv(OUT / "pattern_validation_samples.csv", index=False)
    print(f"Wrote pattern_validation_samples.csv ({len(samp_df)} rows)")

    # Return the base dataframe for downstream (not persisted in main file; computed outputs above)
    return df


if __name__ == "__main__":
    df = main()
    # Print summary
    print("\n--- T22 summary ---")
    print(f"Total SWE LinkedIn English-ok: {len(df):,}")
    print(f"ghost_llm_inflated rate (all): {df['ghost_llm_inflated'].mean():.4f}")
    print(f"ghost_llm_any_not_realistic rate (all): {df['ghost_llm_any_not_realistic'].mean():.4f}")
    print(f"yoe_scope_mismatch_llm rate (all): {df['yoe_scope_mismatch_llm'].mean():.4f}")
    print(f"high_aspiration rate (all): {df['high_aspiration'].fillna(0).mean():.4f}")
    print(f"cred_imp_entry_10y rate (all): {df['cred_imp_entry_10y'].mean():.4f}")
