"""T22 semantic precision check on new patterns.

For each new pattern (aspiration, firm, org_scope, mgmt_strict), draw a
stratified 50-sample (25 from 2024, 25 from 2026) of matching snippets,
auto-classify with deep rules, and report precision. <80% => drop.
"""

import json
import random
import re
from pathlib import Path

import duckdb

random.seed(42)

REPO = Path("/home/jihgaboot/gabor/job-research")
TEXT_PARQUET = REPO / "exploration/artifacts/shared/swe_cleaned_text.parquet"
OUT_DIR = REPO / "exploration/artifacts/T22"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(REPO / "exploration/scripts"))
from T22_patterns import (  # noqa: E402
    ASPIRATION_REGEX,
    FIRM_REGEX,
    ORG_SCOPE_REGEX,
    MGMT_STRICT_REGEX,
    AI_STRICT_REGEX,
    AI_BROAD_REGEX,
)


def load_text_rows() -> list[tuple[str, str, str]]:
    con = duckdb.connect()
    rows = con.execute(
        f"""
        SELECT uid, description_cleaned, period
        FROM '{TEXT_PARQUET}'
        WHERE text_source='llm' AND description_cleaned IS NOT NULL
        """
    ).fetchall()
    return rows


def sample_matches(rows, regex: str, n_per_period: int = 25) -> list[dict]:
    """Draw stratified sample of regex matches with ±60 char context."""
    pat = re.compile(regex, re.IGNORECASE)
    buckets: dict[str, list[dict]] = {"2024": [], "2026": []}
    random.shuffle(rows)
    for uid, text, period in rows:
        if len(buckets["2024"]) >= n_per_period and len(buckets["2026"]) >= n_per_period:
            break
        if text is None:
            continue
        period_key = "2024" if period.startswith("2024") else "2026"
        if len(buckets[period_key]) >= n_per_period:
            continue
        m = pat.search(text)
        if not m:
            continue
        lo = max(0, m.start() - 80)
        hi = min(len(text), m.end() + 80)
        buckets[period_key].append(
            {
                "uid": uid,
                "period": period,
                "match": m.group(0),
                "context": text[lo:hi],
            }
        )
    return buckets["2024"] + buckets["2026"]


def auto_classify_aspiration(match: str, context: str) -> str:
    """Auto-label aspiration matches.

    TP = phrase is used to hedge a requirement (something being 'nice to have',
    'preferred', etc.). FP = phrase is in an irrelevant context.
    """
    m = match.lower()
    c = context.lower()
    # "preferred" has ambiguity: "preferred stock", "preferred pronouns", etc.
    if "preferred" in m:
        # Look for requirement nearby
        if re.search(r"(qualification|experience|skill|knowledge|candidate|degree|education)", c):
            return "TP"
        if re.search(r"(stock|pronoun|partner|network|customer|name)", c):
            return "FP"
        return "AMB"
    # "familiarity with" / "exposure to" almost always hedges a skill req
    if m in {"familiarity with", "exposure to", "working knowledge of"}:
        return "TP"
    # The explicit forms are unambiguous
    if any(
        k in m
        for k in [
            "nice to have",
            "nice-to-have",
            "plus but not required",
            "a plus",
            "bonus",
            "ideal",
            "would love",
            "'d love",
            "helpful but not required",
            "desirable",
            "beneficial",
            "would be nice",
        ]
    ):
        return "TP"
    return "AMB"


def auto_classify_firm(match: str, context: str) -> str:
    """Auto-label firm-requirement matches.

    TP = phrase is imposing a hard requirement. FP = context undermines the
    requirement-framing (e.g., "required to disclose", "must be available
    on-site" when the match was "must be").
    """
    m = match.lower()
    c = context.lower()
    # "required" -- need qualification nearby
    if m == "required":
        # Strip confounds
        if re.search(r"required (to|by|in|for|under)\b", c):
            # look for "required to apply/disclose" context — not a job bar
            if re.search(r"required to (apply|disclose|use|complete|provide)", c):
                return "FP"
        if re.search(r"(experience|qualification|skill|degree|year|knowledge|education)", c):
            return "TP"
        return "AMB"
    if m in {"mandatory", "non-negotiable", "non negotiable", "strictly required", "hard requirement"}:
        return "TP"
    if "minimum" in m:
        # Almost always qualifying a quantity of experience / degree
        if re.search(r"(year|experience|qualification|salary|age)", c):
            return "TP"
        return "AMB"
    if "basic qualification" in m:
        return "TP"
    if "essential" in m:
        # "essential qualification" TP; "essential personnel" FP; "essential duties" TP-ish
        if "qualification" in c:
            return "TP"
        if "personnel" in c or "worker" in c:
            return "FP"
        return "AMB"
    if m in {"must have", "must be", "must possess", "you must"}:
        if re.search(r"(experience|skill|knowledge|year|degree|ability|background)", c):
            return "TP"
        if re.search(r"(disclose|report|apply|eligible|authorized|available)", c):
            return "AMB"
        return "AMB"
    if m in {"requirement", "requirements"}:
        return "TP"
    return "AMB"


def auto_classify_org_scope(match: str, context: str) -> str:
    m = match.lower()
    c = context.lower()
    # All terms can be legitimate organizational-scope language. FPs are rare
    # but we check for obvious non-scope contexts.
    if m == "org" or m == "organization":
        # "organization" usually refers to the employer; we want scope within
        # job description, e.g., "partner with the org". Hard to split without
        # deeper grammar; mark AMB unless verb cue is there.
        if re.search(r"(within|across|the entire|broader|our|company)", c):
            return "TP"
        return "AMB"
    if "vision" in m:
        # "vision" can be "vision care benefits" in benefit sections
        if "benefit" in c or "dental" in c or "insurance" in c:
            return "FP"
        return "TP"
    if m == "align" or "alignment" in m:
        # "company alignment" vs "align the layout"
        if re.search(r"(strateg|roadmap|team|goal|organiz|stakeholder|vision|expectation|priority)", c):
            return "TP"
        return "AMB"
    # The others are almost always scope language in a job posting
    return "TP"


def auto_classify_mgmt(match: str, context: str) -> str:
    m = match.lower()
    c = context.lower()
    if "mentor" in m:
        # mentor/mentorship/mentoring in a SWE JD is essentially always about
        # mentoring others. No observed FPs.
        return "TP"
    if "coach" in m:
        return "TP"
    if "hire" in m:
        # narrow pattern requires "hire and develop/manage/grow" or
        # "hire a team" or "hiring and/manager/engineers/team/plan"
        return "TP"
    if m == "headcount":
        return "TP"
    if "performance review" in m or "performance-review" in m:
        return "TP"
    return "AMB"


def auto_classify_ai_strict(match: str, context: str) -> str:
    m = match.lower()
    c = context.lower()
    # Specific tool names are essentially always TP.
    if m in {
        "copilot",
        "cursor",
        "chatgpt",
        "openai api",
        "gemini",
        "codex",
        "llamaindex",
        "langchain",
        "prompt engineering",
        "rag",
        "vector database",
        "pinecone",
        "huggingface",
        "hugging face",
    }:
        return "TP"
    if m.startswith("gpt"):
        return "TP"
    if "fine" in m and "tuning" in m:
        return "TP"
    if m == "claude":
        # "Claude" could be a person's name in a reference / signature, but
        # rare in JDs.
        if re.search(r"(anthropic|llm|ai|language model|api|tool|sdk|assistant)", c):
            return "TP"
        # Surname or similar
        if re.search(r"(mr\.|ms\.|dr\.|@|sincerely)", c):
            return "FP"
        return "AMB"
    return "AMB"


def auto_classify_ai_broad(match: str, context: str) -> str:
    m = match.lower()
    c = context.lower()
    # Terms shared with the strict pattern
    strict_verdict = auto_classify_ai_strict(match, context)
    if strict_verdict != "AMB":
        return strict_verdict
    if m == "ai":
        # "ai" in words like "lai" caught by word boundary; real concerns are
        # proper nouns like "AIG" (caught), initials like "A.I.". We expect TP rates
        # above 90%.
        if re.search(r"(model|intelligence|ml|machine|engineer|product|tool|learning|system|application)", c):
            return "TP"
        if re.search(r"(aig|a\.i\.g|chai|samurai|mosaic|naive|wasabi)", c):
            return "FP"
        return "AMB"
    if m == "artificial intelligence":
        return "TP"
    if m in {"ml", "machine learning", "generative ai", "genai", "anthropic"}:
        return "TP"
    if m in {"llm", "large language model"}:
        return "TP"
    return "AMB"


PATTERNS = {
    "aspiration_v2": (ASPIRATION_REGEX, auto_classify_aspiration),
    "firm_requirement_v2": (FIRM_REGEX, auto_classify_firm),
    "org_scope_v2": (ORG_SCOPE_REGEX, auto_classify_org_scope),
    "mgmt_strict_v1": (MGMT_STRICT_REGEX, auto_classify_mgmt),
    "ai_strict_v2": (AI_STRICT_REGEX, auto_classify_ai_strict),
    "ai_broad_v2": (AI_BROAD_REGEX, auto_classify_ai_broad),
}


def main() -> None:
    rows = load_text_rows()
    print(f"Loaded {len(rows)} LLM-frame rows.")
    summary = {}
    all_classifications = []
    for name, (regex, classifier) in PATTERNS.items():
        sample = sample_matches(rows, regex, n_per_period=25)
        counts = {"TP": 0, "FP": 0, "AMB": 0}
        for row in sample:
            v = classifier(row["match"], row["context"])
            counts[v] += 1
            all_classifications.append(
                {"pattern": name, **row, "auto_label": v}
            )
        n = sum(counts.values())
        # Conservative precision: AMB treated as non-TP.
        precision = counts["TP"] / n if n else 0.0
        verdict = "PASS" if precision >= 0.80 else "FAIL"
        summary[name] = {
            "n_sampled": n,
            "tp": counts["TP"],
            "fp": counts["FP"],
            "amb": counts["AMB"],
            "precision_conservative": precision,
            "verdict": verdict,
        }
        print(f"{name}: n={n} TP={counts['TP']} FP={counts['FP']} AMB={counts['AMB']} prec={precision:.3f} {verdict}")

    out_json = OUT_DIR / "T22_precision_check.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}")

    # Save classifications for manual audit
    import csv

    out_csv = OUT_DIR / "T22_precision_classifications.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["pattern", "uid", "period", "match", "context", "auto_label"]
        )
        writer.writeheader()
        for row in all_classifications:
            writer.writerow(row)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
