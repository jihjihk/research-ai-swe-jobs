"""V2 Part F: T29 LLM-authorship rejection re-test.

Simple 2-feature score:
  1. Em-dash density per 1k chars (U+2014 or '--')
  2. LLM vocabulary density: delve|leverage|robust|comprehensive|seamless|furthermore|moreover per 1k chars

Identify Q1 (bottom quartile). Re-compute three headline metrics on Q1 subset:
  1. Description length (median)
  2. Credential stack depth proxy (count categories with any mention)
  3. AI mention rate
Compare Q1 deltas to full-corpus deltas.
"""

import duckdb
import re
import random

random.seed(42)
con = duckdb.connect()
UNI = "data/unified.parquet"
BASE = "source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE"

AI_PAT_STR = r"(?i)\b(agentic|multi[- ]agent|ai[- ]agent|ai agents?|\bllm\b|large language model|generative ai|genai|openai|\bclaude\b|\bcopilot\b|langchain|pytorch|tensorflow|prompt engineering|machine learning|deep learning|\bai\b|artificial intelligence)\b"

# Load sample and compute features
EM_DASH_PAT = r"(—|--)"
LLM_VOCAB_PAT = r"(?i)\b(delve|leverage|robust|comprehensive|seamless|furthermore|moreover)\b"

# Category proxies for credential stack
TECH_CAT = r"(?i)\b(python|java|javascript|typescript|aws|azure|gcp|kubernetes|docker|sql|react|angular|node|spring|django|flask|pytorch|tensorflow|kafka|terraform|linux)\b"
SOFT_CAT = r"(?i)\b(communication|collaborat|teamwork|leadership|problem.solving|analytical)\b"
SCOPE_CAT = r"(?i)\b(cross.functional|stakeholders?|roadmap|strategic|company.wide|org.wide)\b"
EDU_CAT = r"(?i)\b(bachelor|master|phd|b\.?s\.?|m\.?s\.?|degree)\b"
YOE_CAT = r"(\d+\+?\s*years?|yoe)"
MGMT_CAT = r"(?i)\b(mentor|coach|lead engineers|lead a team|manage engineers)\b"
AI_CAT_PAT = AI_PAT_STR

CATS = {
    "tech": TECH_CAT, "soft": SOFT_CAT, "scope": SCOPE_CAT,
    "edu": EDU_CAT, "yoe": YOE_CAT, "mgmt": MGMT_CAT, "ai": AI_CAT_PAT,
}


def compute_score_and_metrics(desc):
    if desc is None or len(desc) < 50:
        return None
    n = len(desc)
    k = 1000.0 / n  # per 1k chars
    em_dash = len(re.findall(EM_DASH_PAT, desc)) * k
    llm_vocab = len(re.findall(LLM_VOCAB_PAT, desc)) * k
    score = em_dash + llm_vocab
    # Metrics
    categories = sum(1 for pat in CATS.values() if re.search(pat, desc))
    has_ai = 1 if re.search(AI_PAT_STR, desc) else 0
    return {
        "score": score, "n_chars": n, "categories": categories, "has_ai": has_ai,
    }


for period in ["2024", "2026"]:
    src = "source='scraped'" if period == "2026" else "source IN ('kaggle_arshkon','kaggle_asaniczka')"
    # Sample ~8000 rows via ORDER BY random()
    q = f"""
    SELECT description FROM '{UNI}'
    WHERE {BASE} AND {src} AND description IS NOT NULL AND length(description) >= 100
    ORDER BY random() LIMIT 8000
    """
    rows = [r[0] for r in con.execute(q).fetchall()]
    feats = [compute_score_and_metrics(d) for d in rows]
    feats = [f for f in feats if f is not None]
    scores = sorted(f["score"] for f in feats)
    q1_threshold = scores[len(scores) // 4]
    # Full and Q1
    full_len_med = sorted(f["n_chars"] for f in feats)[len(feats) // 2]
    full_cats_med = sorted(f["categories"] for f in feats)[len(feats) // 2]
    full_ai_rate = sum(f["has_ai"] for f in feats) / len(feats) * 100
    full_stack7 = sum(1 for f in feats if f["categories"] >= 7) / len(feats) * 100
    q1 = [f for f in feats if f["score"] <= q1_threshold]
    q1_len_med = sorted(f["n_chars"] for f in q1)[len(q1) // 2]
    q1_cats_med = sorted(f["categories"] for f in q1)[len(q1) // 2]
    q1_ai_rate = sum(f["has_ai"] for f in q1) / len(q1) * 100
    q1_stack7 = sum(1 for f in q1 if f["categories"] >= 7) / len(q1) * 100
    print(f"\n=== {period} (n={len(feats)}, q1_threshold={q1_threshold:.2f}) ===")
    print(f"  Full  : median_len={full_len_med}, median_cats={full_cats_med}, ai_rate={full_ai_rate:.1f}%, cat>=7 pct={full_stack7:.2f}%")
    print(f"  Q1    : median_len={q1_len_med}, median_cats={q1_cats_med}, ai_rate={q1_ai_rate:.1f}%, cat>=7 pct={q1_stack7:.2f}%")
