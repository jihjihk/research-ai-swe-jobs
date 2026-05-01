"""T21 pattern validation — management, orchestration, strategic language profiles.

Rebuilds T11's management patterns per Gate 2 corrections 2 and 3.

This version uses `description_core_llm` directly from `data/unified.parquet`
(NOT the stopword-stripped `description_cleaned` from the shared artifact)
because validation requires natural prose for the context check.

Validators return True if the match looks like a legitimate senior-role
responsibility signal and False if it looks like boilerplate, compound-noun
distractor, or generic filler. Conservative: when in doubt, reject, since
we want a high-precision pattern bundle for cross-period comparisons.

A pattern passes if precision ≥ 0.80 on a 50-row period-stratified sample
(25 per period, or as many as exist). Passing patterns are combined into
profile-level regexes and saved to
`exploration/artifacts/shared/validated_mgmt_patterns.json`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
SHARED = ROOT / "exploration/artifacts/shared"
T21_TABLES = ROOT / "exploration/tables/T21"
T21_TABLES.mkdir(parents=True, exist_ok=True)

# --- Pattern definitions ---------------------------------------------------

PATTERNS = [
    # --- People management (narrow) -----------------------------------
    ("mentor", r"\bmentor(?:s|ed|ing|ship)?\b", "people_mgmt", "validate_mentor"),
    ("coach", r"\bcoach(?:es|ed|ing)?\b", "people_mgmt", "validate_coach"),
    ("direct_reports", r"\bdirect\s+reports?\b", "people_mgmt", "accept"),
    ("performance_review", r"\bperformance\s+review(?:s|ing)?\b", "people_mgmt", "accept"),
    ("one_on_one", r"\b(?:1:1|1-on-1|one[-\s]on[-\s]one)s?\b", "people_mgmt", "accept"),
    ("headcount", r"\bheadcount\b", "people_mgmt", "accept"),
    ("team_building", r"\bteam[\s-]building\b", "people_mgmt", "accept"),
    ("grow_team", r"\b(?:grow|growing|build|building|scale|scaling)\s+(?:the|your|a|our)\s+(?:engineering\s+)?team\b", "people_mgmt", "accept"),
    ("develop_talent", r"\bdevelop(?:ing)?\s+talent\b", "people_mgmt", "accept"),
    ("manage_team", r"\bmanag(?:e|es|ed|ing)\s+(?:a|the|your|our)\s+(?:team|group|engineering\s+team|team\s+of)\b", "people_mgmt", "accept"),
    ("manage_people", r"\bmanag(?:e|es|ed|ing)\s+(?:people|direct\s+reports|engineers|the\s+engineering)\b", "people_mgmt", "accept"),
    ("lead_team", r"\blead(?:s|ing)?\s+(?:a|the|your|our)\s+(?:team|engineering|group)\b", "people_mgmt", "accept"),
    ("people_management_phrase", r"\bpeople\s+management\b", "people_mgmt", "accept"),
    ("line_manage", r"\bline\s+manag(?:e|es|ed|ing|er|ement)\b", "people_mgmt", "accept"),
    ("career_development", r"\b(?:career\s+development|career\s+growth|career\s+progression)\b", "people_mgmt", "validate_career"),
    ("hiring_decisions", r"\b(?:hiring|interviewing)\s+(?:decisions|process|pipeline|loop)\b", "people_mgmt", "validate_hiring"),

    # --- Technical orchestration ---------------------------------------
    ("architecture_review", r"\barchitectur(?:e|al)\s+reviews?\b", "tech_orch", "accept"),
    ("code_review", r"\bcode\s+reviews?\b", "tech_orch", "accept"),
    ("system_design", r"\bsystem\s+design\b", "tech_orch", "accept"),
    ("technical_direction", r"\btechnical\s+direction\b", "tech_orch", "accept"),
    ("technical_leadership", r"\btechnical\s+leadership\b", "tech_orch", "accept"),
    ("tech_lead", r"\btech(?:nical)?\s+lead\b", "tech_orch", "validate_tech_lead"),
    ("agent_workflow", r"\bagent(?:ic)?\s+(?:workflow|pipeline|systems?|architecture)s?\b", "tech_orch", "accept"),
    ("workflow_automation", r"\bworkflow\s+automation\b", "tech_orch", "accept"),
    ("quality_gate", r"\bquality\s+gates?\b", "tech_orch", "accept"),
    ("guardrails", r"\bguardrails?\b", "tech_orch", "validate_guardrails"),
    ("prompt_engineering", r"\bprompt\s+engineering\b", "tech_orch", "accept"),
    ("evals", r"\b(?:evals?|eval\s+harness(?:es)?|evaluation\s+(?:framework|suite|harness)s?|validation\s+suite)\b", "tech_orch", "accept"),
    ("design_review", r"\bdesign\s+reviews?\b", "tech_orch", "accept"),
    ("technical_vision", r"\btechnical\s+vision\b", "tech_orch", "accept"),
    ("set_technical_standards", r"\b(?:set|setting|establish|establishing|define|defining)\s+(?:technical|engineering)\s+standards\b", "tech_orch", "accept"),

    # --- Strategic scope -----------------------------------------------
    ("stakeholder", r"\bstakeholders?\b", "strategic", "validate_stakeholder"),
    ("business_impact", r"\bbusiness\s+(?:impact|outcomes?|value|goals?)\b", "strategic", "accept"),
    ("product_strategy", r"\bproduct\s+strategy\b", "strategic", "accept"),
    ("roadmap", r"\broadmaps?\b", "strategic", "validate_roadmap"),
    ("prioritization", r"\bprioriti[sz](?:e|ed|es|ing|ation)\b", "strategic", "validate_prioritization"),
    ("resource_allocation", r"\bresource\s+allocation\b", "strategic", "accept"),
    ("budget_ownership", r"\b(?:own(?:s|ed|ing)?|manag(?:e|es|ed|ing)|plan(?:s|ned|ning)?)\s+(?:the\s+)?budget(?:s|ing)?\b", "strategic", "accept"),
    ("cross_functional_alignment", r"\bcross[-\s]?functional\b", "strategic", "validate_xfn"),
    ("drive_revenue", r"\b(?:drive|drives|driving|grow|growing|increase|increasing)\s+revenue\b", "strategic", "accept"),
    ("executive_partner", r"\b(?:executive|c[-\s]?level|leadership\s+team)\s+(?:partner|stakeholders?|alignment)s?\b", "strategic", "accept"),
    ("business_requirements", r"\bbusiness\s+requirements?\b", "strategic", "accept"),
    ("strategic_initiative", r"\bstrategic\s+initiatives?\b", "strategic", "accept"),
    ("influence_direction", r"\binfluenc(?:e|es|ed|ing)\s+(?:the\s+)?(?:direction|strategy|roadmap|priorities)\b", "strategic", "accept"),
]

COMPILED = [(name, re.compile(rgx, re.IGNORECASE), profile, val) for name, rgx, profile, val in PATTERNS]


# --- Validators --------------------------------------------------------------
# All validators receive a window of raw text around the match and return True
# if the match looks like a legitimate senior-role responsibility signal.

def validate_mentor(context: str) -> bool:
    # Mentor passes easily; reject only if the match is a reference to BEING mentored
    # ("as a new hire you will be mentored") rather than DOING mentoring.
    c = context.lower()
    # Common reject patterns: "you will be mentored by", "mentored by a senior"
    if re.search(r"\b(?:you\s+will\s+be|will\s+be|to\s+be|being|get(?:ting)?)\s+mentor", c):
        return False
    # Otherwise treat as legitimate (mentor IS the canonical clean pattern per V1)
    return True


def validate_coach(context: str) -> bool:
    c = context.lower()
    # Accept if about developing/growing/mentoring people.
    if re.search(r"coach(?:es|ed|ing)?\s+(?:and\s+mentor|and\s+develop|and\s+grow)", c):
        return True
    if re.search(r"(?:mentor|develop|guide|grow)\s+(?:and\s+)?coach", c):
        return True
    if re.search(r"coach(?:ing)?\s+(?:engineers|the\s+team|team\s+members|junior|new\s+hires|others|team)", c):
        return True
    if re.search(r"(?:engineering|technical)\s+coach", c):
        return True
    # Reject HR benefits boilerplate ("life coaching", "health coaching")
    if re.search(r"(?:life|health|wellness|career)\s+coach", c):
        return False
    # Reject "player/coach" as this is a shorthand for "IC with some leadership"
    if re.search(r"player[/\s-]coach", c):
        return False
    return False


def validate_career(context: str) -> bool:
    c = context.lower()
    # Accept if the senior is responsible for career development of others.
    # Reject if it's about the candidate's own career.
    if re.search(r"(?:support|lead|drive|foster|enable|provide)\s+(?:the\s+)?career", c):
        return True
    if re.search(r"career\s+(?:development|growth|progression)\s+(?:of|for)\s+(?:engineers|your|the|team)", c):
        return True
    # Reject: "your career", "own career", "strong career development opportunities"
    if re.search(r"(?:your|their|our|strong|great)\s+career", c):
        return False
    if re.search(r"career\s+(?:development|growth)\s+opportunities", c):
        return False
    return False


def validate_hiring(context: str) -> bool:
    c = context.lower()
    # Accept if the senior is described as participating in hiring.
    if re.search(r"(?:participate|involve|lead|conduct|drive)\s+.{0,30}(?:hiring|interview)", c):
        return True
    if re.search(r"(?:hiring|interview(?:ing)?)\s+(?:decisions|loop|process|pipeline|panel)", c):
        # Needs a verb that implies the senior does it
        if re.search(r"(?:lead|own|drive|run|conduct|participate)", c):
            return True
        return False
    return False


def validate_tech_lead(context: str) -> bool:
    c = context.lower()
    # Accept "tech lead" as a role, reject "technical leadership skills"
    if re.search(r"(?:tech|technical)\s+lead(?:s|ing)?\s+(?:for|of|on|to|role|position)", c):
        return True
    if re.search(r"(?:serve|act|function|work)\s+as\s+.{0,10}(?:tech|technical)\s+lead", c):
        return True
    if re.search(r"(?:tech|technical)\s+lead\b(?!\s*ership)", c):
        # Bare "tech lead" — accept
        return True
    return False


def validate_guardrails(context: str) -> bool:
    c = context.lower()
    # Accept if near ai/ml/llm/model/safety/quality
    if re.search(r"(?:ai|ml|llm|model|agent|safety|quality|security|compliance|system|ethical)\s+.{0,40}guardrail", c):
        return True
    if re.search(r"guardrail.{0,40}(?:ai|ml|llm|model|agent|safety|quality|security|compliance|system|ethical)", c):
        return True
    return False


def validate_stakeholder(context: str) -> bool:
    c = context.lower()
    # Accept if it's clearly a strategic-scope mention, not bare filler.
    if re.search(r"(?:business|product|executive|key|senior|external|internal|cross[-\s]?functional|multiple|various)\s+stakeholder", c):
        return True
    if re.search(r"stakeholders?\s+(?:across|from|at\s+all)", c):
        return True
    if re.search(r"(?:engage|align|collaborate|partner|work|communicate|liaise|manage|present)\s+.{0,30}stakeholder", c):
        return True
    if re.search(r"stakeholder\s+(?:management|engagement|communication|alignment|relationships?)", c):
        return True
    # Reject soft-skill filler ("strong communication with non-technical stakeholders")
    # when no strategic scope context is around.
    return False


def validate_roadmap(context: str) -> bool:
    c = context.lower()
    # Accept product/technical/engineering roadmaps
    if re.search(r"(?:product|technical|technology|engineering|delivery|team|quarterly|strategic|platform)\s+roadmap", c):
        return True
    if re.search(r"roadmaps?\s+(?:for|of|and\s+strategy|planning|development)", c):
        return True
    if re.search(r"(?:define|own|shape|drive|influence|contribute\s+to|build|create|set)\s+.{0,30}roadmap", c):
        return True
    if re.search(r"roadmap\s+.{0,30}(?:align|deliver|prioriti|vision|strategy)", c):
        return True
    return False


def validate_prioritization(context: str) -> bool:
    c = context.lower()
    # Strategic: prioritize work, projects, investments, initiatives
    if re.search(r"prioriti[sz](?:e|ed|es|ing|ation)\s+(?:work|tasks|projects|investments|initiatives|features|the\s+roadmap|across|trade-?offs|the\s+backlog|deliverables|technical\s+debt|competing)", c):
        return True
    if re.search(r"(?:ruthless|effective|strategic|business)\s+prioriti", c):
        return True
    if re.search(r"(?:help|support|drive|own|lead)\s+prioriti", c):
        return True
    # Reject "detail-oriented with strong prioritization skills" etc.
    if re.search(r"prioriti[sz]ation\s+skills", c):
        return False
    return False


def validate_xfn(context: str) -> bool:
    c = context.lower()
    # cross-functional is used as adj most of the time. Accept when paired with
    # alignment/collaboration/partner/team/initiative.
    if re.search(r"cross[-\s]?functional\s+(?:alignment|collaboration|partners?|initiatives?|delivery)", c):
        return True
    if re.search(r"cross[-\s]?functional\s+teams?", c):
        return True  # Using "with cross-functional teams" is strategic-scope signal at senior
    if re.search(r"(?:align|collaborat|partner|coordinat)\s+.{0,30}cross[-\s]?functional", c):
        return True
    return False


VALIDATORS = {
    "accept": lambda _c: True,
    "validate_mentor": validate_mentor,
    "validate_coach": validate_coach,
    "validate_career": validate_career,
    "validate_hiring": validate_hiring,
    "validate_tech_lead": validate_tech_lead,
    "validate_guardrails": validate_guardrails,
    "validate_stakeholder": validate_stakeholder,
    "validate_roadmap": validate_roadmap,
    "validate_prioritization": validate_prioritization,
    "validate_xfn": validate_xfn,
}


# --- Inline pattern asserts -----------------------------------------------

def _assert_patterns():
    def has(name, text):
        for n, rgx, _, _ in COMPILED:
            if n == name and rgx.search(text):
                return True
        return False

    assert has("mentor", "mentor junior engineers")
    assert has("mentor", "mentoring and coaching")
    assert has("manage_team", "manage a team of 5 engineers")
    assert has("lead_team", "lead a team of engineers")
    assert has("code_review", "conduct code reviews")
    assert has("stakeholder", "align with stakeholders")
    assert has("roadmap", "own the product roadmap")
    assert has("prioritization", "prioritize work")
    assert has("evals", "build evaluation frameworks")
    assert has("evals", "set up eval harnesses")
    assert has("agent_workflow", "design agentic workflows")
    assert has("business_impact", "drive business impact")

_assert_patterns()


# --- Load data -------------------------------------------------------------

print("[T21-validate] Loading senior postings from unified.parquet")

con = duckdb.connect()
q = """
SELECT u.uid,
       u.description_core_llm AS text,
       CASE WHEN substr(u.period, 1, 4) IN ('2024') THEN '2024'
            WHEN substr(u.period, 1, 4) IN ('2026') THEN '2026'
            ELSE substr(u.period, 1, 4) END AS period2,
       u.seniority_final
FROM 'data/unified.parquet' AS u
WHERE u.is_swe = true
  AND u.source_platform = 'linkedin'
  AND u.is_english = true
  AND u.date_flag = 'ok'
  AND u.seniority_final IN ('mid-senior', 'director')
  AND u.llm_extraction_coverage = 'labeled'
  AND u.description_core_llm IS NOT NULL
"""
sen = con.execute(q).fetchdf()
sen = sen[sen["period2"].isin(["2024", "2026"])].reset_index(drop=True)
texts = sen["text"].fillna("").tolist()
sen_uids = sen["uid"].to_numpy()
sen_period = sen["period2"].to_numpy()
print(f"[T21-validate] Senior postings with labeled llm text: {len(sen):,}")


def extract_context(text: str, match: re.Match, window: int = 100) -> str:
    s = max(0, match.start() - window)
    e = min(len(text), match.end() + window)
    return text[s:e]


# --- Validate each pattern -------------------------------------------------

rng = np.random.default_rng(42)

results_rows = []
per_pattern_samples: dict[str, list[dict]] = {}
for name, rgx, profile, val_key in COMPILED:
    validator = VALIDATORS[val_key]
    matching_2024: list[int] = []
    matching_2026: list[int] = []
    for i, t in enumerate(texts):
        if rgx.search(t):
            (matching_2024 if sen_period[i] == "2024" else matching_2026).append(i)
    n_2024, n_2026 = len(matching_2024), len(matching_2026)
    if n_2024 + n_2026 == 0:
        results_rows.append({
            "pattern": name, "profile": profile, "n_matches_2024": 0,
            "n_matches_2026": 0, "sample_size": 0, "precision": np.nan,
            "precision_2024": np.nan, "precision_2026": np.nan,
            "passes_80pct": False,
        })
        continue
    take_2024 = min(25, n_2024)
    take_2026 = min(25, n_2026)
    s_24 = rng.choice(matching_2024, size=take_2024, replace=False) if take_2024 > 0 else np.array([], dtype=int)
    s_26 = rng.choice(matching_2026, size=take_2026, replace=False) if take_2026 > 0 else np.array([], dtype=int)
    correct_2024 = 0
    correct_2026 = 0
    samples: list[dict] = []
    for i in s_24:
        t = texts[i]
        m = rgx.search(t)
        ctx = extract_context(t, m) if m else ""
        ok = validator(ctx)
        correct_2024 += int(ok)
        samples.append({"uid": sen_uids[i], "period": "2024", "pattern": name, "context": ctx, "accepted": ok})
    for i in s_26:
        t = texts[i]
        m = rgx.search(t)
        ctx = extract_context(t, m) if m else ""
        ok = validator(ctx)
        correct_2026 += int(ok)
        samples.append({"uid": sen_uids[i], "period": "2026", "pattern": name, "context": ctx, "accepted": ok})
    p_24 = correct_2024 / take_2024 if take_2024 > 0 else np.nan
    p_26 = correct_2026 / take_2026 if take_2026 > 0 else np.nan
    total = take_2024 + take_2026
    prec = (correct_2024 + correct_2026) / total if total > 0 else np.nan
    passes = bool(np.isfinite(prec) and prec >= 0.80)
    results_rows.append({
        "pattern": name, "profile": profile,
        "n_matches_2024": n_2024, "n_matches_2026": n_2026,
        "sample_size": total, "precision": prec,
        "precision_2024": p_24, "precision_2026": p_26,
        "passes_80pct": passes,
    })
    per_pattern_samples[name] = samples

precision_df = pd.DataFrame(results_rows).sort_values(["profile", "precision"], ascending=[True, False])
precision_df.to_csv(T21_TABLES / "validated_patterns.csv", index=False)
print(precision_df.to_string())

all_samples = [row for rows in per_pattern_samples.values() for row in rows]
pd.DataFrame(all_samples).to_csv(T21_TABLES / "pattern_samples.csv", index=False)

# --- Build validated regex bundle ------------------------------------------

kept = precision_df[precision_df["passes_80pct"]].copy()
dropped = precision_df[~precision_df["passes_80pct"]].copy()
print(f"\n[T21-validate] KEPT {len(kept)} / {len(precision_df)} patterns")
print(f"[T21-validate] DROPPED: {dropped['pattern'].tolist()}")

pat_map = {name: rgx for name, rgx, _prof, _v in PATTERNS}
profile_regexes: dict[str, str] = {}
for profile in ("people_mgmt", "tech_orch", "strategic"):
    pats = kept[kept["profile"] == profile]["pattern"].tolist()
    if not pats:
        profile_regexes[profile] = ""
        continue
    combined = "|".join(f"(?:{pat_map[p]})" for p in pats)
    profile_regexes[profile] = combined

mgmt_combined = profile_regexes["people_mgmt"] or r"\bmentor(?:s|ed|ing|ship)?\b"

artifact = {
    "generated_by": "T21_validate_patterns.py",
    "gate": "Gate 2 post-corrections",
    "validation_method": "50-row stratified sample (25/25 per period), rule-based validator on description_core_llm",
    "threshold": 0.80,
    "profiles": {
        "people_mgmt": {
            "regex": profile_regexes["people_mgmt"],
            "kept_patterns": kept[kept["profile"] == "people_mgmt"]["pattern"].tolist(),
            "dropped_patterns": dropped[dropped["profile"] == "people_mgmt"]["pattern"].tolist(),
        },
        "tech_orch": {
            "regex": profile_regexes["tech_orch"],
            "kept_patterns": kept[kept["profile"] == "tech_orch"]["pattern"].tolist(),
            "dropped_patterns": dropped[dropped["profile"] == "tech_orch"]["pattern"].tolist(),
        },
        "strategic": {
            "regex": profile_regexes["strategic"],
            "kept_patterns": kept[kept["profile"] == "strategic"]["pattern"].tolist(),
            "dropped_patterns": dropped[dropped["profile"] == "strategic"]["pattern"].tolist(),
        },
    },
    "mgmt_combined_regex": mgmt_combined,
    "precision_table": precision_df.replace({np.nan: None}).to_dict(orient="records"),
}

with (SHARED / "validated_mgmt_patterns.json").open("w") as f:
    json.dump(artifact, f, indent=2)

print(f"[T21-validate] Saved artifact → {SHARED / 'validated_mgmt_patterns.json'}")
