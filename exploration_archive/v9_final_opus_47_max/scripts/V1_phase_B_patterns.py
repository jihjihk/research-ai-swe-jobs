"""V1 Phase B — Pattern validation (semantic precision sampling).

Sample 50 matches per pattern, stratified 25 pre-2026 (arshkon+asaniczka) + 25 scraped.
Extract 200-char context around each match, save for semantic judgment.
"""

import re
import sys
import duckdb
import pandas as pd
import numpy as np
import json
from pathlib import Path

RNG = np.random.RandomState(171717)

print("[V1_B] Loading SWE LinkedIn text", flush=True)
con = duckdb.connect()
df = con.execute("""
    SELECT u.uid,
           u.description,
           u.period,
           u.source,
           u.title,
           c.description_cleaned,
           c.text_source
    FROM read_parquet('data/unified.parquet') u
    LEFT JOIN read_parquet('exploration/artifacts/shared/swe_cleaned_text.parquet') c
      ON u.uid = c.uid
    WHERE u.source_platform = 'linkedin'
      AND u.is_english = TRUE
      AND u.date_flag = 'ok'
      AND u.is_swe = TRUE
""").fetchdf()
df['period2'] = df['period'].astype(str).apply(
    lambda p: '2024' if p.startswith('2024') else ('2026' if p.startswith('2026') else 'other')
)
df = df[df['period2'].isin(['2024','2026'])].copy()
print(f"[V1_B] Loaded {len(df)} rows", flush=True)

# Primary text for matching: prefer cleaned (LLM) where available, else raw description.
df['text'] = df['description_cleaned'].fillna(df['description']).astype(str).str.lower()

# Pattern definitions to validate
PATTERNS = {
    # AI-mention strict: key tokens (tool/protocol/product names + tech techniques)
    "ai_strict": r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|langchain|prompt engineering|fine[- ]tun(?:e|ed|ing)|rag|vector databas(?:e|es)|pinecone|huggingface|hugging face)\b",
    # AI-mention broad: strict + generic tokens
    "ai_broad":  r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|langchain|prompt engineering|fine[- ]tun(?:e|ed|ing)|rag|vector databas(?:e|es)|pinecone|huggingface|hugging face|agent|machine learning|ml|ai|llm|artificial intelligence|mcp)\b",
    # Management strict: T11's strict tier
    "mgmt_strict": r"\b(mentor(?:s|ed|ing|ship)?|coach(?:es|ed|ing)?|hire(?:s|d|ing)?|headcount|performance review|direct reports?)\b",
    # Management broad: strict + broad tokens
    "mgmt_broad": r"\b(mentor(?:s|ed|ing|ship)?|coach(?:es|ed|ing)?|hire(?:s|d|ing)?|headcount|performance review|direct reports?|lead(?:s|ing)?|team(?:s)?|stakeholder(?:s)?|coordinate(?:s|d|ing)?|manage(?:s|d|r|ment|rs|ing)?)\b",
    # Scope terms
    "scope": r"\b(ownership|end[\s\-]to[\s\-]end|cross[\s\-]functional|autonomous|initiative(?:s)?|stakeholder(?:s)?)\b",
    # Soft skills
    "soft_skills": r"\b(collaborative|communication|teamwork|problem[\s\-]solving|interpersonal|leadership)\b",
}

# Assertion tests
for nm, pat in PATTERNS.items():
    re.compile(pat)  # just compile check
# semantic asserts
assert re.search(PATTERNS['ai_strict'], "we use rag for retrieval")
assert re.search(PATTERNS['ai_strict'], "prompt engineering skills")
assert re.search(PATTERNS['mgmt_strict'], "mentorship program")
assert re.search(PATTERNS['mgmt_strict'], "mentor junior engineers")
assert re.search(PATTERNS['mgmt_broad'], "lead a team")
assert re.search(PATTERNS['scope'], "end-to-end ownership")
assert re.search(PATTERNS['soft_skills'], "strong communication skills")
# Negative assertions: some false-positive guardrails
assert re.search(PATTERNS['ai_strict'], "dragged the system") is None  # "rag" should be bounded
assert re.search(PATTERNS['ai_strict'], "forage") is None

def sample_matches(pattern, n_per_period=25):
    """Sample n_per_period rows per period and return (row_idx, first_match_info)."""
    pat = re.compile(pattern)
    rows = []
    for per in ['2024', '2026']:
        sub = df[df['period2']==per]
        matched = []
        # Iterate rows in random order, keep first n with match
        idx_order = np.arange(len(sub))
        RNG.shuffle(idx_order)
        for ii in idx_order:
            row = sub.iloc[ii]
            m = pat.search(row['text'])
            if m is None:
                continue
            start = max(m.start() - 100, 0)
            end = min(m.end() + 100, len(row['text']))
            ctx = row['text'][start:end]
            matched.append({
                "period": per,
                "uid": row['uid'],
                "source": row['source'],
                "title": row['title'],
                "matched_sub": m.group(0),
                "context_200c": ctx,
            })
            if len(matched) >= n_per_period:
                break
        rows.extend(matched)
    return rows

print("[V1_B] Sampling 50 matches per pattern (25 per period) ...", flush=True)
all_samples = {}
for nm, pat in PATTERNS.items():
    samples = sample_matches(pat)
    all_samples[nm] = samples
    print(f"  {nm}: {len(samples)} samples", flush=True)

# Save raw samples for inspection
Path("exploration/tables/V1").mkdir(parents=True, exist_ok=True)
for nm, samples in all_samples.items():
    pd.DataFrame(samples).to_csv(f"exploration/tables/V1/pattern_samples_{nm}.csv", index=False)
    print(f"[V1_B] Wrote pattern_samples_{nm}.csv", flush=True)

# Also examine specific sub-patterns for AI-strict
print("\n[V1_B] === SUB-PATTERN sampling for AI-strict key tokens ===", flush=True)
AI_SUB = {
    "rag": r"\brag\b",
    "copilot": r"\bcopilot\b",
    "langchain": r"\blangchain\b",
    "prompt_engineering": r"\bprompt engineering\b",
    "huggingface": r"\bhuggingface|hugging face\b",
    "vector_databases": r"\bvector databas(e|es)\b",
    "fine_tuning": r"\bfine[- ]tun(?:e|ed|ing)\b",
    "gpt": r"\bgpt-?\d+\b",
    "gemini": r"\bgemini\b",
}

ai_sub_samples = {}
for nm, pat in AI_SUB.items():
    samples = sample_matches(pat, n_per_period=15)
    ai_sub_samples[nm] = samples
    print(f"  AI-strict sub / {nm}: {len(samples)} samples", flush=True)
    if samples:
        pd.DataFrame(samples).to_csv(f"exploration/tables/V1/pattern_sub_ai_{nm}.csv", index=False)

MGMT_SUB = {
    "mentor": r"\bmentor(?:s|ed|ing|ship)?\b",
    "hire": r"\bhire(?:s|d|ing)?\b",
    "performance_review": r"\bperformance review\b",
    "direct_reports": r"\bdirect reports?\b",
    "coach": r"\bcoach(?:es|ed|ing)?\b",
    "lead_broad": r"\blead(?:s|ing)?\b",
    "team_broad": r"\bteam(?:s)?\b",
    "stakeholder_broad": r"\bstakeholder(?:s)?\b",
    "coordinate_broad": r"\bcoordinate(?:s|d|ing)?\b",
    "manage_broad": r"\bmanage(?:s|d|r|ment|rs|ing)?\b",
}

mgmt_sub_samples = {}
for nm, pat in MGMT_SUB.items():
    samples = sample_matches(pat, n_per_period=15)
    mgmt_sub_samples[nm] = samples
    print(f"  Mgmt sub / {nm}: {len(samples)} samples", flush=True)
    if samples:
        pd.DataFrame(samples).to_csv(f"exploration/tables/V1/pattern_sub_mgmt_{nm}.csv", index=False)

print("\n[V1_B] Samples written. Next step: semantic judgment.", flush=True)
