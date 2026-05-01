"""V2 Part D: Validate T22 10x hedge ratio for AI requirements.

T22 claim: within 80-char windows around AI terms, hedge:firm ratio is ~10:1
in BOTH periods, vs global baseline ~1.5:1.
"""

import duckdb
import re

con = duckdb.connect()

HEDGE_TERMS = [
    r"nice to have", r"preferred", r"bonus", r"a plus", r"\bplus\b",
    r"ideally", r"experience with", r"familiarity with", r"exposure to",
    r"knowledge of", r"working knowledge",
]
FIRM_TERMS = [
    r"must have", r"required", r"requirement", r"minimum", r"mandatory",
    r"you need", r"you must", r"required to", r"must be able",
]

HEDGE_PAT = r"(?i)\b(" + "|".join(HEDGE_TERMS) + r")\b"
# Firm: require not followed by "field", "fields", "by law" etc. Use a negative lookahead.
FIRM_PAT = r"(?i)\b(" + "|".join(FIRM_TERMS) + r")\b(?!\s+field)"

# Validation
def count(pat, s): return len(re.findall(pat, s))
assert count(HEDGE_PAT, "nice to have: python") == 1
assert count(HEDGE_PAT, "experience with django is a plus") == 2  # 'experience with' + 'a plus' (also 'plus' bare)
assert count(FIRM_PAT, "this is a required field") == 0  # negative lookahead
assert count(FIRM_PAT, "must have 5 years required") == 2
print("hedge/firm patterns validated")

AI_PAT = (
    r"(?i)\b("
    r"agentic|multi[- ]agent|ai[- ]agent|ai agents?|"
    r"\bllm\b|\bllms\b|large language model|"
    r"retrieval[- ]augmented|"
    r"generative ai|gen[- ]ai\b|genai|"
    r"openai|chatgpt|anthropic|\bclaude\b|"
    r"\bcopilot\b|"
    r"langchain|langgraph|llamaindex|"
    r"pytorch|tensorflow|"
    r"prompt engineering|"
    r"ai[/-]?powered|ai[/-]?driven|"
    r"machine learning|deep learning"
    r")\b"
)

UNI = "data/unified.parquet"
BASE = "source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE"


# Pull samples per period
def load(period, n):
    src = "source='scraped'" if period == "2026" else "source IN ('kaggle_arshkon','kaggle_asaniczka')"
    q = f"""
    SELECT description FROM '{UNI}'
    WHERE {BASE} AND {src} AND description IS NOT NULL
    USING SAMPLE {n}
    """
    return [r[0] for r in con.execute(q).fetchall()]


for period in ["2024", "2026"]:
    # Load all (chunked)
    q = f"""
    SELECT count(*) FROM '{UNI}'
    WHERE {BASE} AND {'source=' + repr('scraped') if period == '2026' else "source IN ('kaggle_arshkon','kaggle_asaniczka')"}
    """
    total = con.execute(q).fetchone()[0]

    # Stream
    src = "source='scraped'" if period == "2026" else "source IN ('kaggle_arshkon','kaggle_asaniczka')"
    q = f"""
    SELECT description FROM '{UNI}'
    WHERE {BASE} AND {src} AND description IS NOT NULL
    """
    hedge_global = 0; firm_global = 0
    hedge_ai = 0; firm_ai = 0
    n_rows = 0
    n_with_ai = 0
    for (desc,) in con.execute(q).fetchall():
        n_rows += 1
        if desc is None: continue
        lo = desc.lower()
        # Global counts
        hedge_global += len(re.findall(HEDGE_PAT, lo))
        firm_global += len(re.findall(FIRM_PAT, lo))
        # AI windows: find every AI match, take ±80 char window, count hedges/firms inside
        ai_matches = list(re.finditer(AI_PAT, lo))
        if ai_matches:
            n_with_ai += 1
        for m in ai_matches:
            start = max(0, m.start() - 80)
            end = min(len(lo), m.end() + 80)
            window = lo[start:end]
            hedge_ai += len(re.findall(HEDGE_PAT, window))
            firm_ai += len(re.findall(FIRM_PAT, window))
    global_ratio = hedge_global / firm_global if firm_global else float("nan")
    ai_ratio = hedge_ai / firm_ai if firm_ai else float("nan")
    print(f"\n=== {period} ({n_rows} postings, {n_with_ai} with AI) ===")
    print(f"  Global: hedge={hedge_global}, firm={firm_global}, ratio={global_ratio:.2f}")
    print(f"  AI windows: hedge={hedge_ai}, firm={firm_ai}, ratio={ai_ratio:.2f}")
    print(f"  AI/Global ratio multiplier: {ai_ratio/global_ratio:.2f}")
