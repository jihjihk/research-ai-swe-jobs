"""T22 step 1 — Pattern validation.

Builds candidate pattern sets for T22 (ghost forensics) and validates each
against samples drawn from the raw best-available text. For each pattern we
sample 50 matches (25 per period if possible) and write a human-reviewable
text file; the script also records each pattern's row hit count by period.

The goal: surface boilerplate contamination before we use patterns downstream.

Patterns covered
----------------
Hedging language ("preferred"):
  - nice_to_have         "nice to have" / "nice-to-have"
  - preferred_hedge      "preferred" when it follows / precedes qualification
                         context (NOT "preferred customer", "preferred vendor")
  - bonus_hedge          "is a plus" / "would be a plus" / "a plus"
  - ideally              "ideally"
  - bonus_word           "bonus" in "bonus points" / "bonus if"

Firm requirements ("must have"):
  - must_have            "must have"
  - required_req         "required" specifically in requirement context
                         (NOT "required field")
  - minimum_req          "minimum" / "at minimum"
  - mandatory_req        "mandatory"

Management (strict versions, build on T11 validated):
  - strict_mentor        "mentor engineers" / "mentor junior" / "mentor team"
                         / "coach engineers"
  - strict_people_mgr    "direct reports" / "people manager" / "people
                         management" / "performance review"
  - strict_hire_mgmt     "hire engineers" / "grow the team" / "build the
                         team" / "lead hiring" / "own hiring"

Scope terms (senior scope):
  - senior_architecture  "system design" / "architecture" (in design context)
  - ownership_senior     "own the" / "end-to-end ownership"
  - distributed_systems  "distributed systems"

AI categories (for T23):
  - ai_tool              copilot / cursor / llm / prompt engineering /
                         langchain / mcp
  - ai_domain            machine learning / deep learning / nlp /
                         transformer / embedding / fine-tuning
  - ai_general           artificial intelligence / AI (ambiguous)
  - agentic              agentic (~95% AI-precision per V1)
  - ai_agent_phrase      'ai agent' / 'multi-agent' / 'agentic AI'
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNIFIED = ROOT / "data" / "unified.parquet"
OUT_DIR = ROOT / "exploration" / "tables" / "T22"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_TXT = OUT_DIR / "pattern_validation_samples.txt"
COUNTS_CSV = OUT_DIR / "pattern_validation_counts.csv"
ARTIFACT_JSON = ROOT / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json"

BASE_FILTER = (
    "source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE"
)

# Patterns are stored as RE2/duckdb-compatible regex strings (case-insensitive
# applied at query time via lower(text)). Python-side re.compile mirrors with IGNORECASE.
PATTERNS: dict[str, dict[str, str]] = {
    # --- hedging (aspirational) ---
    "nice_to_have": {
        "re": r"nice[- ]to[- ]have",
        "cat": "hedge",
    },
    "preferred_hedge": {
        # "preferred" adjacent to qualification language. Must be near
        # skills/experience words. Reject "preferred customer/vendor/partner".
        "re": r"(skills?|experience|knowledge|familiarity|qualification|background)[^.\n]{0,30}(preferred|a plus)",
        "cat": "hedge",
    },
    "preferred_alt": {
        "re": r"preferred[^.\n]{0,30}(skills?|experience|qualification|but not required)",
        "cat": "hedge",
    },
    "bonus_plus": {
        "re": r"\ba plus\b|\bis a plus\b|would be a plus",
        "cat": "hedge",
    },
    "ideally": {
        "re": r"\bideally\b",
        "cat": "hedge",
    },
    "bonus_points": {
        "re": r"bonus points|bonus if|as a bonus",
        "cat": "hedge",
    },
    "nice_if": {
        "re": r"nice if",
        "cat": "hedge",
    },
    "familiarity_with": {
        # "familiarity with X" is hedged lang when paired with tools.
        "re": r"familiarity with",
        "cat": "hedge",
    },
    "exposure_to": {
        "re": r"\bexposure to\b",
        "cat": "hedge",
    },
    "nice_to_know": {
        "re": r"nice to know",
        "cat": "hedge",
    },
    # --- firm requirement language ---
    "must_have": {
        "re": r"must[- ]have|must have",
        "cat": "firm",
    },
    "required_req": {
        # "is required" / "required experience" / "experience required" but
        # NOT "required field" (form boilerplate) or "required credit".
        "re": r"(experience|knowledge|skills?|degree|qualification)[^.\n]{0,10}(is |are )?required|(required[^.\n]{0,10})(experience|knowledge|skills?|qualifications?)",
        "cat": "firm",
    },
    "minimum_req": {
        "re": r"\bminimum[- ]of\b|\bat[- ]minimum\b|\bminimum qualification",
        "cat": "firm",
    },
    "mandatory_req": {
        "re": r"\bmandatory\b",
        "cat": "firm",
    },
    # --- management strict (for saved artifact) ---
    "strict_mentor": {
        "re": r"mentor (engineers?|juniors?|team|developers?|interns?|the team)|coach engineers?|mentor(ing|ship) (engineers?|juniors?|team|developers?)",
        "cat": "mgmt",
    },
    "strict_people_mgr": {
        "re": r"direct reports?|\bpeople manager\b|\bpeople management\b|performance reviews?",
        "cat": "mgmt",
    },
    "strict_hire_mgmt": {
        "re": r"hire engineers?|hiring engineers?|grow the team|build (the |a |out )?team|lead hiring|own hiring|manage a team of|lead a team of",
        "cat": "mgmt",
    },
    "strict_headcount": {
        "re": r"\bheadcount\b|budget responsibility",
        "cat": "mgmt",
    },
    # --- senior scope terms ---
    "architecture_scope": {
        "re": r"system design|distributed systems?|architect(ing|ure) (of |decisions|review)",
        "cat": "scope",
    },
    "ownership_scope": {
        "re": r"\bend[- ]to[- ]end\b|own the (delivery|product|system|project|feature)|take ownership",
        "cat": "scope",
    },
    # --- AI indicators (for T22 AI ghostiness + T23) ---
    "ai_tool": {
        "re": r"\bcopilot\b|\bcursor\b|\bllm(s)?\b|prompt engineering|\blangchain\b|\bmcp\b|ai pair program",
        "cat": "ai_tool",
    },
    "ai_domain": {
        "re": r"machine learning|deep learning|\bnlp\b|natural language processing|computer vision|model training|\btransformers?\b|\bembeddings?\b|fine[- ]tun(e|ing)",
        "cat": "ai_domain",
    },
    "ai_general": {
        "re": r"\bai\b|artificial intelligence",
        "cat": "ai_general",
    },
    "agentic": {
        "re": r"\bagentic\b",
        "cat": "ai_tool",
    },
    "ai_agent_phrase": {
        "re": r"ai agents?|multi[- ]agent|autonomous agents?|agentic (ai|workflow)",
        "cat": "ai_tool",
    },
    "rag_phrase": {
        "re": r"\brag\b|retrieval augmented|retrieval[- ]augmented",
        "cat": "ai_tool",
    },
}


def build_text_view(con: duckdb.DuckDBPyConnection):
    con.execute(
        f"""
        CREATE OR REPLACE VIEW swe AS
        SELECT
            uid,
            title,
            CASE WHEN source IN ('kaggle_arshkon','kaggle_asaniczka') THEN '2024'
                 WHEN source='scraped' THEN '2026' END AS period2,
            lower(COALESCE(NULLIF(description_core_llm, ''), description_core, description)) AS text_best
        FROM read_parquet('{UNIFIED}')
        WHERE {BASE_FILTER}
        """
    )


def count_pattern(con, name, regex) -> dict:
    # DuckDB regexp_matches is case-sensitive; text_best is already lower.
    rows = con.execute(
        f"""
        SELECT period2, COUNT(*) AS matches
        FROM swe
        WHERE regexp_matches(text_best, '{regex.replace("'","''")}')
        GROUP BY period2
        ORDER BY period2
        """
    ).fetchall()
    return {p: n for p, n in rows}


def sample_matches(con, name, regex, n_per_period=25, seed=42) -> list[tuple]:
    con.execute(f"SELECT setseed({seed/100})")
    out = []
    py_re = re.compile(regex, re.IGNORECASE)
    for period in ("2024", "2026"):
        rows = con.execute(
            f"""
            SELECT uid, title, text_best
            FROM swe
            WHERE period2 = '{period}'
              AND regexp_matches(text_best, '{regex.replace("'","''")}')
            ORDER BY random()
            LIMIT {n_per_period}
            """
        ).fetchall()
        for uid, title, text in rows:
            m = py_re.search(text or "")
            if m:
                s = max(0, m.start() - 120)
                e = min(len(text), m.end() + 200)
                snippet = text[s:e].replace("\n", " ")
            else:
                snippet = (text or "")[:320].replace("\n", " ")
            out.append((period, uid, title, snippet))
    return out


def main() -> None:
    con = duckdb.connect()
    build_text_view(con)

    counts_rows: list[dict] = []
    with SAMPLE_TXT.open("w") as f:
        for name, info in PATTERNS.items():
            regex = info["re"]
            cat = info["cat"]
            cts = count_pattern(con, name, regex)
            row = {"pattern": name, "category": cat, **cts}
            counts_rows.append(row)

            f.write("=" * 100 + "\n")
            f.write(f"PATTERN: {name}  [{cat}]\n")
            f.write(f"REGEX:   {regex}\n")
            f.write(f"COUNTS:  {cts}\n")
            f.write("=" * 100 + "\n")
            samples = sample_matches(con, name, regex)
            for period, uid, title, snippet in samples:
                f.write(f"\n[{period}] {uid} | {title}\n")
                f.write(f"  ... {snippet} ...\n")
            f.write("\n\n")

    counts_df = pd.DataFrame(counts_rows).fillna(0)
    for col in ("2024", "2026"):
        if col not in counts_df.columns:
            counts_df[col] = 0
    counts_df.to_csv(COUNTS_CSV, index=False)

    # Save initial artifact. Will be re-written by 02 after precision review.
    initial_artifact = {
        "schema_version": 1,
        "note": "Initial (pre-precision-review) regex patterns used in T22. "
                "See exploration/reports/T22.md for precision estimates and the "
                "precision-reviewed final set written in T22_02.",
        "patterns": {k: v["re"] for k, v in PATTERNS.items()},
    }
    ARTIFACT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_JSON.open("w") as f:
        json.dump(initial_artifact, f, indent=2)

    print(f"Wrote {COUNTS_CSV}")
    print(f"Wrote {SAMPLE_TXT}")
    print(f"Wrote {ARTIFACT_JSON}")


if __name__ == "__main__":
    main()
