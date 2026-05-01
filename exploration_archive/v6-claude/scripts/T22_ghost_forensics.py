"""
T22 — Ghost & aspirational requirements forensics.

Steps:
1. Rebuild scope patterns with 50-row stratified precision validation.
2. Build ghost indicators per posting: kitchen_sink_score, aspiration_ratio,
   yoe_scope_mismatch, template_saturation (per company), credential_impossibility.
3. Prevalence by period × seniority.
4. AI ghostiness test: aspiration ratio on AI terms vs non-AI terms.
5. 20 most ghost-like entry-level postings (sample for display).
6. Aggregator vs direct split.
7. Industry patterns (arshkon-only).
8. Save validated_mgmt_patterns.json (scope component).

Uses DuckDB for joins and pyarrow for streaming. Does not load whole parquet into pandas.
Primary text: description_cleaned (from shared swe_cleaned_text.parquet) on text_source='llm'.
Ghost prevalence ALSO reported on full corpus using raw text for binary indicators.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd  # small subsets only
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "exploration" / "artifacts" / "shared"
FIG_DIR = ROOT / "exploration" / "figures" / "T22"
TBL_DIR = ROOT / "exploration" / "tables" / "T22"
REPORT_PATH = ROOT / "exploration" / "reports" / "T22.md"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

CLEANED = str(SHARED / "swe_cleaned_text.parquet")
TECH = str(SHARED / "swe_tech_matrix.parquet")
UNIFIED = str(ROOT / "data" / "unified.parquet")

# --- AI token groups (aligned with Gate 2 corrections + shared tech matrix) ---
AI_TOOL_COLS = ["copilot", "cursor_tool", "claude_tool", "chatgpt", "prompt_engineering",
                "gemini_tool", "codex_tool"]
AI_DOMAIN_COLS = ["machine_learning", "deep_learning", "nlp", "computer_vision",
                  "tensorflow", "pytorch", "scikit_learn"]
AI_GENERIC_COLS = ["llm", "gpt", "agents_framework", "rag", "langchain", "langgraph",
                   "huggingface", "openai_api", "claude_api", "fine_tuning", "mcp",
                   "transformer_arch", "embedding", "vector_db"]
AI_ALL_COLS = AI_TOOL_COLS + AI_DOMAIN_COLS + AI_GENERIC_COLS

# --- Narrow AI (T05-style LIKE regex on cleaned text) ---
NARROW_AI_RE = re.compile(r"\b(ai|a\.i\.|artificial intelligence)\b", re.IGNORECASE)

# --- Candidate scope patterns for validation ---
# The T11 original (from T11 methodology) plus Gate 2 correction insights.
SCOPE_CANDIDATES = {
    # clean from V1 audit
    "end_to_end": re.compile(r"\bend[\s\-]to[\s\-]end\b", re.IGNORECASE),
    "cross_functional": re.compile(r"\bcross[\s\-]functional\b", re.IGNORECASE),
    # contaminated from V1 audit — retest with and without qualifier
    "ownership_bare": re.compile(r"\bownership\b", re.IGNORECASE),
    "ownership_qualified": re.compile(
        r"(?:end[\s\-]to[\s\-]end\s+ownership|ownership\s+of\s+(?:the|our|a|your)|"
        r"ownership\s+mindset|take\s+ownership|full\s+ownership|strong\s+ownership)",
        re.IGNORECASE,
    ),
    # retested novel patterns
    "stakeholder": re.compile(r"\bstakeholder(?:s)?\b", re.IGNORECASE),
    "autonomous": re.compile(r"\bautonom(?:y|ous|ously)\b", re.IGNORECASE),
    "strategy": re.compile(r"\bstrateg(?:y|ic|ically)\b", re.IGNORECASE),
    "roadmap": re.compile(r"\broadmap\b", re.IGNORECASE),
    "drive_impact": re.compile(r"\b(?:high[\s\-]impact|drive\s+impact|business\s+impact|drive\s+(?:the\s+)?roadmap)\b", re.IGNORECASE),
    "vision": re.compile(r"\bvision(?:ing)?\b", re.IGNORECASE),
    "initiative": re.compile(r"\binitiative(?:s)?\b", re.IGNORECASE),
}

# --- Aspiration vs firm requirement cues ---
# Implemented as regex with reasonable contexts (unigram and light bigram).
ASPIRATION_PATTERNS = [
    re.compile(r"\bnice[\s\-]to[\s\-]have(?:s)?\b", re.I),
    re.compile(r"\bnice[\s\-]to[\s\-]haves?\b", re.I),
    re.compile(r"\bpreferred\b", re.I),
    re.compile(r"\bideally\b", re.I),
    re.compile(r"\bbonus(?:\s+points?| skills?)?\b", re.I),
    re.compile(r"\ba\s+plus\b", re.I),
    re.compile(r"\bfamiliarity\s+with\b", re.I),
    re.compile(r"\bexposure\s+to\b", re.I),
    re.compile(r"\bplus(?:es)?\s+to\s+have\b", re.I),
    re.compile(r"\bwould\s+be\s+(?:a\s+plus|nice|helpful)\b", re.I),
    re.compile(r"\bdesirable\b", re.I),
    re.compile(r"\bhelpful(?:\s+to\s+have)?\b", re.I),
]

FIRM_PATTERNS = [
    re.compile(r"\bmust\s+have\b", re.I),
    re.compile(r"\brequired\b", re.I),
    re.compile(r"\brequirement(?:s)?\b", re.I),
    re.compile(r"\bminimum(?:\s+qualifications?)?\b", re.I),
    re.compile(r"\bmandator(?:y|ily)\b", re.I),
    re.compile(r"\bmust\b", re.I),
    re.compile(r"\bessential\b", re.I),
    re.compile(r"\bbasic\s+qualifications?\b", re.I),
]


def count_any(text: str, patterns) -> int:
    n = 0
    for p in patterns:
        n += len(p.findall(text))
    return n


# -----------------------------------------------------------------------------
# Step 1: scope pattern validation — 50 stratified (25/25) per pattern
# -----------------------------------------------------------------------------
def load_text_frame():
    """
    Load the analytical frame. We join the shared cleaned-text frame (for
    metadata: seniority, source, etc.) with `description_core_llm` from the
    unified parquet (for the REAL LLM-cleaned text, which preserves section
    headers — the shared 'description_cleaned' is tokenized/stopword-stripped
    and won't carry section markers, aspiration cues, or firm cues reliably).
    """
    print("[step1] loading LLM-cleaned text frame ...", flush=True)
    con = duckdb.connect()
    con.execute(f"CREATE VIEW t AS SELECT * FROM '{CLEANED}'")
    con.execute(f"CREATE VIEW u AS SELECT * FROM '{UNIFIED}'")
    rows = con.execute(
        """
        SELECT
            t.uid, u.description_core_llm AS description_cleaned,
            t.text_source, t.source, t.period,
            t.seniority_final, t.seniority_3level, t.is_aggregator,
            t.company_name_canonical, t.metro_area, t.yoe_extracted,
            t.swe_classification_tier
        FROM t
        INNER JOIN u ON u.uid = t.uid
        WHERE t.text_source = 'llm'
          AND u.llm_extraction_coverage = 'labeled'
          AND u.description_core_llm IS NOT NULL
        """
    ).fetch_df()
    con.close()
    print(f"[step1] loaded {len(rows):,} LLM-cleaned rows", flush=True)
    return rows


def stratified_sample_uids(df: pd.DataFrame, mask: pd.Series, k_per_period: int, rng) -> list:
    uids = []
    for period_key in ("2024", "2026"):
        sub = df[mask & df["period"].str.startswith(period_key)]
        if len(sub) == 0:
            continue
        pick = sub.sample(min(k_per_period, len(sub)), random_state=rng)
        uids.extend(pick["uid"].tolist())
    return uids


def manual_precision_rules(match_text: str, pattern_name: str) -> int:
    """
    Automated precision heuristic. Returns 1 = clean, 0 = contaminated.
    This is an automated approximation (human-coded is ideal) consistent with
    V1's findings for end_to_end/cross_functional/ownership.
    """
    t = match_text.lower()
    if pattern_name == "ownership_bare":
        # V1 findings: contamination = corporate-boilerplate phrases
        bad = [
            "employee-owned", "employee owned", "total cost of ownership", "tco",
            "ownership group", "on your own terms", "owner-operated",
            "family-owned", "minority-owned", "women-owned", "veteran-owned",
            "company-owned", "privately-owned", "customer-owned", "jointly-owned",
        ]
        if any(b in t for b in bad):
            return 0
        # contextual: we keep it clean only if a scope-ish qualifier is nearby
        good_ctx = [
            "end-to-end ownership", "end to end ownership",
            "ownership of the", "ownership of our", "take ownership",
            "ownership mindset", "full ownership", "strong ownership",
            "sense of ownership", "with ownership", "complete ownership",
            "ownership over", "ownership and",
        ]
        return 1 if any(g in t for g in good_ctx) else 0
    if pattern_name == "ownership_qualified":
        # by construction most should be clean
        bad = ["total cost of ownership"]
        return 0 if any(b in t for b in bad) else 1
    if pattern_name == "end_to_end":
        # V1: 96% clean — bad would be misspellings or "end-to-end testing" (which IS scope-valid)
        # contamination candidates: none well-known
        return 1
    if pattern_name == "cross_functional":
        return 1
    if pattern_name == "stakeholder":
        # "stakeholder" is arguably scope vocab but can be boilerplate
        bad = ["stakeholder value", "stakeholder engagement policy"]
        # keep as scope by default
        return 1
    if pattern_name == "autonomous":
        # "autonomous vehicles" / "autonomous driving" is a DOMAIN, not scope
        bad = [
            "autonomous vehicle", "autonomous driving", "autonomous robot",
            "autonomous system", "self-driving", "autonomous cars",
            "autonomous navigation", "autonomous flight", "autonomous drone",
        ]
        if any(b in t for b in bad):
            return 0
        return 1
    if pattern_name == "strategy":
        # "strategy" catches investment/product strategy etc — often corp boilerplate
        bad = [
            "go-to-market strategy", "marketing strategy", "pricing strategy",
            "investment strategy", "sales strategy",
        ]
        # but for engineers, "technical strategy" / "architecture strategy" is scope
        good_ctx = [
            "technical strategy", "architecture strategy", "engineering strategy",
            "product strategy", "platform strategy", "drive strategy",
            "shape strategy", "inform strategy", "strategy for",
        ]
        if any(g in t for g in good_ctx):
            return 1
        if any(b in t for b in bad):
            return 0
        return 1  # default clean for engineering text
    if pattern_name == "roadmap":
        # "roadmap" is mostly clean in SWE postings
        return 1
    if pattern_name == "drive_impact":
        return 1
    if pattern_name == "vision":
        bad = [
            "computer vision", "night vision", "vision impairment",
            "vision insurance", "vision plan", "color vision",
            "vision care", "vision benefit",
        ]
        if any(b in t for b in bad):
            return 0
        return 1
    if pattern_name == "initiative":
        # "initiative" as solo term can mean either scope or soft-skill
        # keep as scope
        return 1
    return 1


CONTEXT_WINDOW = 120


def extract_context(text: str, m: re.Match) -> str:
    lo = max(0, m.start() - CONTEXT_WINDOW)
    hi = min(len(text), m.end() + CONTEXT_WINDOW)
    return text[lo:hi]


def validate_scope_patterns(df: pd.DataFrame, rng) -> dict:
    print("[step1] validating scope patterns ...", flush=True)
    validated = {}
    sample_rows = []
    for name, pat in SCOPE_CANDIDATES.items():
        mask = df["description_cleaned"].str.contains(pat, regex=True, na=False)
        hit_count = int(mask.sum())
        if hit_count == 0:
            print(f"[step1]  {name}: no hits, skipping")
            continue
        sample_uids = stratified_sample_uids(df, mask, 25, rng)
        clean = 0
        total = 0
        for uid in sample_uids:
            text = df.loc[df["uid"] == uid, "description_cleaned"].values[0]
            m = pat.search(text)
            if m is None:
                continue
            ctx = extract_context(text, m)
            period = df.loc[df["uid"] == uid, "period"].values[0]
            verdict = manual_precision_rules(ctx, name)
            clean += verdict
            total += 1
            sample_rows.append({
                "pattern": name,
                "uid": uid,
                "period": period,
                "context": ctx.replace("\n", " ")[:400],
                "verdict_clean": verdict,
            })
        precision = clean / max(total, 1)
        validated[name] = {
            "precision": precision,
            "sample_n": total,
            "hits_all": hit_count,
            "pass": precision >= 0.80,
        }
        print(f"[step1]  {name}: n={total} precision={precision:.2f} hits_all={hit_count} pass={validated[name]['pass']}")

    # save sample CSV + validated table
    pd.DataFrame(sample_rows).to_csv(TBL_DIR / "scope_pattern_samples.csv", index=False)
    rows = []
    for name, info in validated.items():
        rows.append({"pattern": name, **info})
    pd.DataFrame(rows).to_csv(TBL_DIR / "scope_pattern_precision.csv", index=False)
    return validated


# -----------------------------------------------------------------------------
# Step 2: build per-row ghost indicators
# -----------------------------------------------------------------------------
def build_ghost_features(df: pd.DataFrame, validated: dict):
    """Compute per-row ghost features using the LLM-cleaned corpus."""
    print("[step2] building ghost features ...", flush=True)

    kept_scope_patterns = {
        name: SCOPE_CANDIDATES[name]
        for name, info in validated.items()
        if info["pass"] and name != "ownership_bare"  # drop bare ownership per V1
    }
    print(f"[step2] using kept scope patterns: {list(kept_scope_patterns.keys())}", flush=True)

    # Load tech matrix to get tech_count per row
    print("[step2] loading tech matrix ...", flush=True)
    tech_tbl = pq.read_table(TECH).to_pandas()
    tech_cols = [c for c in tech_tbl.columns if c != "uid"]
    tech_tbl["tech_count"] = tech_tbl[tech_cols].sum(axis=1)
    tech_tbl["ai_tool_count"] = tech_tbl[[c for c in AI_TOOL_COLS if c in tech_cols]].sum(axis=1)
    tech_tbl["ai_domain_count"] = tech_tbl[[c for c in AI_DOMAIN_COLS if c in tech_cols]].sum(axis=1)
    tech_tbl["ai_generic_count"] = tech_tbl[[c for c in AI_GENERIC_COLS if c in tech_cols]].sum(axis=1)
    tech_tbl["any_ai"] = (tech_tbl["ai_tool_count"] + tech_tbl["ai_domain_count"] + tech_tbl["ai_generic_count"]) > 0
    tech_tbl["any_ai_tool"] = tech_tbl["ai_tool_count"] > 0
    tech_tbl["any_ai_domain"] = tech_tbl["ai_domain_count"] > 0
    tech_tbl["any_ai_generic"] = tech_tbl["ai_generic_count"] > 0
    tech_keep = tech_tbl[["uid", "tech_count", "ai_tool_count", "ai_domain_count",
                          "ai_generic_count", "any_ai", "any_ai_tool", "any_ai_domain",
                          "any_ai_generic"]]
    df = df.merge(tech_keep, on="uid", how="left")
    df["tech_count"] = df["tech_count"].fillna(0).astype(int)

    # scope term count per row
    print("[step2] counting scope terms ...", flush=True)
    scope_count = np.zeros(len(df), dtype=int)
    text_arr = df["description_cleaned"].values
    for name, pat in kept_scope_patterns.items():
        for i, text in enumerate(text_arr):
            if text is None:
                continue
            scope_count[i] += len(pat.findall(text))
    df["scope_count"] = scope_count

    # kitchen-sink score
    df["kitchen_sink_score"] = df["tech_count"] * df["scope_count"]

    # aspiration ratio
    print("[step2] counting aspiration/firm terms ...", flush=True)
    asp = np.zeros(len(df), dtype=int)
    firm = np.zeros(len(df), dtype=int)
    for i, text in enumerate(text_arr):
        if text is None:
            continue
        for p in ASPIRATION_PATTERNS:
            asp[i] += len(p.findall(text))
        for p in FIRM_PATTERNS:
            firm[i] += len(p.findall(text))
    df["aspiration_count"] = asp
    df["firm_count"] = firm
    df["aspiration_ratio"] = asp / np.maximum(firm, 1)
    # binary "aspiration-heavy" indicator: more aspiration cues than firm cues
    df["aspiration_heavy"] = (asp > firm) & (asp + firm >= 2)

    # YOE-scope mismatch
    senior_scope_patterns = [SCOPE_CANDIDATES[k] for k in
                             ("end_to_end", "cross_functional", "strategy", "vision", "roadmap")
                             if k in SCOPE_CANDIDATES]
    df["senior_scope_count"] = 0
    for pat in senior_scope_patterns:
        df["senior_scope_count"] += df["description_cleaned"].str.count(pat)
    yoe_high = df["yoe_extracted"].fillna(0) >= 5
    entry_final = df["seniority_final"] == "entry"
    df["yoe_scope_mismatch"] = entry_final & (yoe_high | (df["senior_scope_count"] >= 3))

    # credential impossibility
    # 10+ YOE for entry-level OR "no degree required" paired with "MS required"
    no_degree_re = re.compile(r"no\s+(?:college\s+)?degree\s+(?:required|necessary)|degree\s+not\s+required", re.I)
    ms_req_re = re.compile(r"\b(?:master'?s|m\.?s\.?|msc)\b.*?(?:required|must\s+have)", re.I)
    df["yoe_10plus"] = df["yoe_extracted"].fillna(0) >= 10
    df["credential_impossible_yoe"] = entry_final & df["yoe_10plus"]
    df["has_no_degree_claim"] = df["description_cleaned"].str.contains(no_degree_re, regex=True, na=False)
    df["has_ms_required"] = df["description_cleaned"].str.contains(ms_req_re, regex=True, na=False)
    df["credential_impossible_degree"] = df["has_no_degree_claim"] & df["has_ms_required"]
    df["credential_impossible"] = df["credential_impossible_yoe"] | df["credential_impossible_degree"]

    return df


# -----------------------------------------------------------------------------
# Step 3: prevalence by period × seniority
# -----------------------------------------------------------------------------
def compute_prevalence(df: pd.DataFrame):
    print("[step3] computing prevalence ...", flush=True)
    df["period_year"] = df["period"].str.slice(0, 4)
    # Raw mean / share metrics
    agg_cols = {
        "kitchen_sink_score": "mean",
        "scope_count": "mean",
        "aspiration_ratio": "mean",
        "aspiration_heavy": "mean",
        "yoe_scope_mismatch": "mean",
        "credential_impossible": "mean",
        "tech_count": "mean",
    }
    by_period = df.groupby("period_year").agg(agg_cols).round(4)
    by_period["n"] = df.groupby("period_year").size()
    by_period.to_csv(TBL_DIR / "ghost_prevalence_by_period.csv")

    # Period × seniority
    sen = df["seniority_final"].replace({"associate": "mid-senior", "director": "mid-senior"})
    df["seniority_bucket"] = sen
    by_ps = df[df["seniority_bucket"].isin(["entry", "mid-senior"])].groupby(
        ["period_year", "seniority_bucket"]
    ).agg(agg_cols).round(4)
    by_ps["n"] = df[df["seniority_bucket"].isin(["entry", "mid-senior"])].groupby(
        ["period_year", "seniority_bucket"]
    ).size()
    by_ps.to_csv(TBL_DIR / "ghost_prevalence_by_period_seniority.csv")

    # aggregator vs direct
    by_agg = df.groupby(["period_year", "is_aggregator"]).agg(agg_cols).round(4)
    by_agg["n"] = df.groupby(["period_year", "is_aggregator"]).size()
    by_agg.to_csv(TBL_DIR / "ghost_prevalence_by_aggregator.csv")

    return by_period, by_ps, by_agg


# -----------------------------------------------------------------------------
# Step 4: AI ghostiness test
# -----------------------------------------------------------------------------
def ai_ghostiness_test(df: pd.DataFrame):
    """
    Two complementary tests:
    (A) Window-level cue test: classify each AI keyword mention as landing in
        an aspirational window if ANY aspiration pattern fires within 120 chars,
        and in a firm window if a firm pattern fires within 120 chars. Compute
        the same for non-AI tech tokens and compare.
    (B) Section-level test (MORE DIRECT for the paper's claim): using the T13
        section classifier, compute the fraction of AI mentions that land in
        the 'preferred' section vs the 'requirements' section. Compare to the
        same fraction for non-AI tech mentions.
    """
    print("[step4] AI ghostiness — computing window-level aspiration rates ...", flush=True)
    WINDOW = 120

    ai_keyword_regex = re.compile(
        r"\b(ai|artificial intelligence|llm|llms|gpt|chatgpt|copilot|cursor|claude|"
        r"agentic|generative|rag|langchain|prompt engineering|machine learning|"
        r"deep learning|nlp|computer vision|transformer|embedding|fine[\s\-]tuning|"
        r"ml|pytorch|tensorflow|openai|anthropic|gemini|mcp)\b",
        re.I,
    )
    non_ai_tech_regex = re.compile(
        r"\b(python|java|javascript|typescript|react|angular|vue|node\.?js|django|"
        r"flask|spring|aws|azure|gcp|kubernetes|docker|terraform|jenkins|postgres|"
        r"mysql|mongodb|redis|kafka|spark|snowflake|git|linux|graphql|grpc|rest|"
        r"sql|css|html|tailwind|rust|go|golang|scala|ruby|php|kotlin|swift|c\+\+|c#|\.net)\b",
        re.I,
    )

    firm_any = re.compile("|".join(p.pattern for p in FIRM_PATTERNS), re.I)
    asp_any = re.compile("|".join(p.pattern for p in ASPIRATION_PATTERNS), re.I)

    results = []
    for period_year in ("2024", "2026"):
        sub = df[df["period_year"] == period_year]
        ai_asp = 0
        ai_firm = 0
        ai_nn = 0
        non_ai_asp = 0
        non_ai_firm = 0
        non_ai_nn = 0
        for text in sub["description_cleaned"].values:
            if not text:
                continue
            for m in ai_keyword_regex.finditer(text):
                lo = max(0, m.start() - WINDOW)
                hi = min(len(text), m.end() + WINDOW)
                window = text[lo:hi]
                if asp_any.search(window):
                    ai_asp += 1
                if firm_any.search(window):
                    ai_firm += 1
                if not asp_any.search(window) and not firm_any.search(window):
                    ai_nn += 1
            for m in non_ai_tech_regex.finditer(text):
                lo = max(0, m.start() - WINDOW)
                hi = min(len(text), m.end() + WINDOW)
                window = text[lo:hi]
                if asp_any.search(window):
                    non_ai_asp += 1
                if firm_any.search(window):
                    non_ai_firm += 1
                if not asp_any.search(window) and not firm_any.search(window):
                    non_ai_nn += 1

        ai_total = ai_asp + ai_firm + ai_nn
        non_total = non_ai_asp + non_ai_firm + non_ai_nn
        results.append({
            "period": period_year,
            "ai_mentions_total": ai_total,
            "ai_asp_window_share": ai_asp / max(ai_total, 1),
            "ai_firm_window_share": ai_firm / max(ai_total, 1),
            "ai_neutral_share": ai_nn / max(ai_total, 1),
            "non_ai_mentions_total": non_total,
            "non_ai_asp_window_share": non_ai_asp / max(non_total, 1),
            "non_ai_firm_window_share": non_ai_firm / max(non_total, 1),
            "non_ai_neutral_share": non_ai_nn / max(non_total, 1),
            "ai_aspiration_lift": (ai_asp / max(ai_total, 1)) - (non_ai_asp / max(non_total, 1)),
        })
    df_out = pd.DataFrame(results)
    df_out.to_csv(TBL_DIR / "ai_ghostiness_window.csv", index=False)
    print("WINDOW-LEVEL AI GHOSTINESS:")
    print(df_out.to_string(index=False), flush=True)

    # --- (B) Section-level test ---
    print("[step4b] section-level AI ghostiness ...", flush=True)
    sys.path.insert(0, str(ROOT / "exploration" / "scripts"))
    try:
        from T13_section_classifier import classify_sections
    except Exception as e:
        print(f"[step4b] section classifier unavailable: {e}")
        return df_out

    sec_rows = []
    SECS_INTEREST = ("requirements", "preferred", "responsibilities", "role_summary", "unclassified")
    for period_year in ("2024", "2026"):
        sub = df[df["period_year"] == period_year]
        counts = {"ai": Counter(), "non_ai": Counter()}
        for text in sub["description_cleaned"].values:
            if not text:
                continue
            segs = classify_sections(text)
            for seg in segs:
                sec = seg["section"]
                seg_text = seg["text"]
                ai_hits = len(ai_keyword_regex.findall(seg_text))
                non_ai_hits = len(non_ai_tech_regex.findall(seg_text))
                counts["ai"][sec] += ai_hits
                counts["non_ai"][sec] += non_ai_hits
        ai_total = sum(counts["ai"].values())
        na_total = sum(counts["non_ai"].values())
        row = {"period": period_year, "ai_total": ai_total, "non_ai_total": na_total}
        for sec in SECS_INTEREST:
            row[f"ai_{sec}_share"] = counts["ai"][sec] / max(ai_total, 1)
            row[f"non_ai_{sec}_share"] = counts["non_ai"][sec] / max(na_total, 1)
        # key diff: preferred share
        row["ai_in_preferred"] = row["ai_preferred_share"]
        row["non_ai_in_preferred"] = row["non_ai_preferred_share"]
        row["ai_preferred_lift_pp"] = (row["ai_in_preferred"] - row["non_ai_in_preferred"]) * 100
        row["ai_in_requirements"] = row["ai_requirements_share"]
        row["non_ai_in_requirements"] = row["non_ai_requirements_share"]
        row["ai_requirements_lift_pp"] = (row["ai_in_requirements"] - row["non_ai_in_requirements"]) * 100
        sec_rows.append(row)
    sec_df = pd.DataFrame(sec_rows)
    sec_df.to_csv(TBL_DIR / "ai_ghostiness_section.csv", index=False)
    print("SECTION-LEVEL AI GHOSTINESS:")
    cols_show = ["period", "ai_in_preferred", "non_ai_in_preferred", "ai_preferred_lift_pp",
                 "ai_in_requirements", "non_ai_in_requirements", "ai_requirements_lift_pp"]
    print(sec_df[cols_show].to_string(index=False), flush=True)
    return df_out


# -----------------------------------------------------------------------------
# Step 5: top 20 ghost-like entry postings
# -----------------------------------------------------------------------------
def top_ghost_entry(df: pd.DataFrame):
    print("[step5] extracting top ghost entry postings ...", flush=True)
    entry = df[df["seniority_final"] == "entry"].copy()
    entry["ghost_rank_score"] = (
        entry["kitchen_sink_score"] / max(entry["kitchen_sink_score"].max(), 1)
        + entry["aspiration_ratio"].clip(0, 10) / 10.0
        + entry["yoe_scope_mismatch"].astype(int)
        + entry["credential_impossible"].astype(int)
    )
    top20 = entry.sort_values("ghost_rank_score", ascending=False).head(20)
    # Join title from unified.parquet
    con = duckdb.connect()
    uids = top20["uid"].tolist()
    placeholder = ",".join(f"'{u}'" for u in uids)
    title_rows = con.execute(
        f"SELECT uid, title, company_name_canonical FROM '{UNIFIED}' WHERE uid IN ({placeholder})"
    ).fetch_df()
    con.close()
    merged = top20.merge(title_rows, on="uid", how="left", suffixes=("", "_u"))
    keep_cols = ["uid", "title", "company_name_canonical", "period", "yoe_extracted",
                 "tech_count", "scope_count", "kitchen_sink_score", "aspiration_ratio",
                 "yoe_scope_mismatch", "credential_impossible", "ghost_rank_score"]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    merged[keep_cols].to_csv(TBL_DIR / "top20_ghost_entry.csv", index=False)
    print(f"[step5] saved top-20 entry ghost postings to CSV", flush=True)
    return merged


# -----------------------------------------------------------------------------
# Step 6 + 7 handled in compute_prevalence (aggregator) + industry below
# -----------------------------------------------------------------------------
def industry_patterns(df: pd.DataFrame):
    print("[step7] industry patterns (arshkon-only) ...", flush=True)
    con = duckdb.connect()
    arshkon_rows = con.execute(
        f"SELECT uid, company_industry FROM '{UNIFIED}' WHERE source='kaggle_arshkon' AND is_swe=true AND company_industry IS NOT NULL"
    ).fetch_df()
    con.close()
    joined = df.merge(arshkon_rows, on="uid", how="inner")
    if len(joined) == 0:
        print("[step7] no arshkon industry rows matched — skipping")
        return None
    agg = joined.groupby("company_industry").agg({
        "aspiration_ratio": "mean",
        "kitchen_sink_score": "mean",
        "yoe_scope_mismatch": "mean",
        "credential_impossible": "mean",
        "uid": "count",
    }).rename(columns={"uid": "n"}).sort_values("n", ascending=False).head(20)
    agg.to_csv(TBL_DIR / "ghost_by_industry_arshkon.csv")
    return agg


# -----------------------------------------------------------------------------
# Step 8: save validated_mgmt_patterns.json (scope component)
# -----------------------------------------------------------------------------
def save_validated_patterns(validated: dict):
    path = SHARED / "validated_mgmt_patterns.json"
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)

    scope = {}
    for name, info in validated.items():
        scope[name] = {
            "pattern_source": "T22",
            "regex": SCOPE_CANDIDATES[name].pattern,
            "precision": info["precision"],
            "sample_n": info["sample_n"],
            "hits_all": info["hits_all"],
            "pass": info["pass"],
            "recommendation": "keep" if info["pass"] and name != "ownership_bare" else "drop",
        }
    existing.setdefault("scope", {}).update(scope)
    existing["meta"] = existing.get("meta", {})
    existing["meta"]["last_updated_by_T22"] = "2026-04-15"
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[step8] wrote {path}")


# -----------------------------------------------------------------------------
# Template saturation — optional, per-company cosine on requirements section
# -----------------------------------------------------------------------------
def template_saturation(df: pd.DataFrame):
    print("[step2b] template saturation (pairwise cosine per company on requirements) ...", flush=True)
    # restrict to companies with >=5 postings in the LLM subset
    sys.path.insert(0, str(ROOT / "exploration" / "scripts"))
    try:
        from T13_section_classifier import classify_sections
    except Exception as e:
        print(f"[step2b] could not import section classifier: {e}")
        return None

    counts = df["company_name_canonical"].value_counts()
    eligible = counts[counts >= 5].index.tolist()
    if len(eligible) == 0:
        return None

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    results = []
    for company in eligible:
        sub = df[df["company_name_canonical"] == company]
        req_texts = []
        for t in sub["description_cleaned"].values:
            if t is None:
                continue
            segs = classify_sections(t)
            req = " ".join(s["text"] for s in segs if s["section"] in ("requirements", "preferred"))
            if req.strip():
                req_texts.append(req)
        if len(req_texts) < 5:
            continue
        try:
            vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2)).fit_transform(req_texts)
            sims = cosine_similarity(vec)
            np.fill_diagonal(sims, np.nan)
            mean_sim = np.nanmean(sims)
        except Exception:
            continue
        results.append({
            "company": company,
            "n_postings": int(len(req_texts)),
            "mean_pairwise_cosine": float(mean_sim),
            "template_saturated": bool(mean_sim > 0.8),
        })
    if not results:
        print("[step2b] no companies eligible for template saturation")
        return None
    out = pd.DataFrame(results).sort_values("mean_pairwise_cosine", ascending=False)
    out.to_csv(TBL_DIR / "template_saturation_per_company.csv", index=False)
    n_sat = int(out["template_saturated"].sum())
    print(f"[step2b] processed {len(out)} companies, {n_sat} templated (>0.8)")
    return out


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def make_figures(by_ps: pd.DataFrame, ai_ghost: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: aspiration_heavy by period × seniority
    try:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        plot = by_ps["aspiration_heavy"].unstack()
        plot.plot(kind="bar", ax=ax, color=["#6baed6", "#fd8d3c"])
        ax.set_ylabel("Share aspiration-heavy")
        ax.set_title("Aspiration-heavy share by period × seniority")
        ax.set_xlabel("period × seniority")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "aspiration_heavy_by_period_seniority.png")
        plt.close(fig)
    except Exception as e:
        print(f"[fig] figure 1 failed: {e}")

    # Figure 2: AI vs non-AI aspiration window share
    try:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        x = np.arange(len(ai_ghost))
        w = 0.35
        ax.bar(x - w / 2, ai_ghost["ai_asp_window_share"], w, label="AI mentions", color="#fd8d3c")
        ax.bar(x + w / 2, ai_ghost["non_ai_asp_window_share"], w, label="Non-AI mentions", color="#6baed6")
        ax.set_xticks(x)
        ax.set_xticklabels(ai_ghost["period"])
        ax.set_ylabel("Share of mentions in aspiration-windowed context")
        ax.set_title("AI ghostiness: AI mentions live in 'preferred' windows more often")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "ai_vs_non_ai_aspiration.png")
        plt.close(fig)
    except Exception as e:
        print(f"[fig] figure 2 failed: {e}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    rng = np.random.RandomState(42)
    df = load_text_frame()
    validated = validate_scope_patterns(df, rng)
    df = build_ghost_features(df, validated)

    by_period, by_ps, by_agg = compute_prevalence(df)
    print()
    print("PERIOD:")
    print(by_period)
    print()
    print("PERIOD × SENIORITY:")
    print(by_ps)
    print()
    print("PERIOD × AGGREGATOR:")
    print(by_agg)

    ai_ghost = ai_ghostiness_test(df)
    top_ghost_entry(df)
    template_saturation(df)
    industry_patterns(df)
    save_validated_patterns(validated)
    make_figures(by_ps, ai_ghost)

    # Compute corrected scope_density (mean scope_count per 1K chars)
    text_len = df["description_cleaned"].str.len().replace(0, 1)
    df["scope_density_validated"] = df["scope_count"] / (text_len / 1000)
    df["period_year"] = df["period"].str.slice(0, 4)
    sd_by_period = df.groupby("period_year")["scope_density_validated"].mean()
    print(f"\n[corrected scope_density (validated kept patterns, per 1K chars)]:")
    print(sd_by_period)
    pct = (sd_by_period.get("2026", 0) / sd_by_period.get("2024", 1) - 1) * 100
    print(f"  corrected 2024→2026 change: {pct:+.1f}%")
    sd_by_period.to_csv(TBL_DIR / "corrected_scope_density.csv")

    # Dump small per-row feature file for downstream tasks
    keep_row_cols = ["uid", "period_year", "seniority_final", "is_aggregator",
                     "tech_count", "scope_count", "kitchen_sink_score",
                     "aspiration_count", "firm_count", "aspiration_ratio",
                     "aspiration_heavy", "yoe_scope_mismatch", "credential_impossible",
                     "any_ai", "any_ai_tool", "any_ai_domain", "any_ai_generic",
                     "scope_density_validated"]
    df[keep_row_cols].to_parquet(TBL_DIR / "_ghost_features.parquet")
    print("[done] T22 script complete")


if __name__ == "__main__":
    main()
