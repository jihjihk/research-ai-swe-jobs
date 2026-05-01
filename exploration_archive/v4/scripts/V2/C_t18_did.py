"""V2 Part C: Validate T18 cross-occupation DiD.

T18 claims:
  SWE AI rate: +22.9pp (2024->2026)
  SWE-adjacent: +19.0pp (83% of SWE magnitude)
  Control: +1.2pp
  DiD SWE-control = +21.7pp
  DiD SWE-adjacent = +3.9pp
"""

import duckdb
import re

con = duckdb.connect()

# Build AI indicator with assertions
# Use validated patterns. 'mcp' is contaminated (Microsoft Certified Professional).
# Bare 'agent' is contaminated. Use 'agentic' or 'ai agent'.

AI_PAT = (
    r"(?i)\b("
    r"agentic|multi[- ]agent|ai[- ]agent|ai agents?|"
    r"\bllm\b|\bllms\b|large language model|"
    r"\brag\b[^s]|retrieval[- ]augmented|"
    r"generative ai|gen[- ]ai\b|genai|"
    r"\bgpt[- ]?\d|openai|chatgpt|anthropic|\bclaude\b|gemini(?:\s*(?:pro|api|flash))?|"
    r"\bcopilot\b|cursor(?:\s+(?:ide|editor|ai))|"
    r"langchain|langgraph|llamaindex|"
    r"pytorch|tensorflow|hugging\s*face|"
    r"prompt engineering|"
    r"ai[/-]?powered|ai[/-]?driven|ai[/-]?native|ai[/-]?first|"
    r"machine learning|\bml\b|deep learning"
    r")\b"
)

# Python-level validation with assertions
def has_ai(s):
    if s is None: return False
    return re.search(AI_PAT, s) is not None

# Positives
for s in [
    "we use llms daily", "agentic workflows", "ai-agent orchestration",
    "genai stack", "pytorch experience", "claude api",
    "machine learning engineer", "ml experience required",
    "prompt engineering skills", "langchain framework",
    "ai-powered product", "retrieval-augmented generation",
]:
    assert has_ai(s), f"FAIL positive: {s}"

# Negatives
for s in [
    "microsoft certified professional", "mcp required",  # mcp contaminated
    "insurance agent network", "change agent mindset",  # bare agent contaminated
    "ragged edge of the plate",  # rag word start
    "customer service representative", "hr recruiter",
    "project manager", "real estate agent",
]:
    assert not has_ai(s), f"FAIL negative: {s}"

print(f"AI pattern validated.")

BASE = "source_platform='linkedin' AND is_english=TRUE AND date_flag='ok'"
UNI = "data/unified.parquet"

# Compute AI rate by group and period
q = f"""
WITH f AS (
  SELECT uid, description, title_normalized,
    CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period,
    is_swe, is_swe_adjacent, is_control, swe_classification_tier,
    CASE WHEN regexp_matches(lower(description), $${AI_PAT}$$) THEN 1 ELSE 0 END AS has_ai
  FROM '{UNI}'
  WHERE {BASE}
),
g AS (
  SELECT period,
    CASE
      WHEN is_swe THEN 'SWE'
      WHEN is_swe_adjacent THEN 'SWE_adjacent'
      WHEN is_control THEN 'control'
      ELSE 'other'
    END AS grp,
    count(*) AS n,
    avg(has_ai)*100 AS ai_rate_pct
  FROM f
  GROUP BY 1,2
  ORDER BY 1,2
)
SELECT * FROM g WHERE grp != 'other' ORDER BY grp, period
"""
print("\n=== AI rate by group and period ===")
df = con.execute(q).fetchdf()
print(df.to_string())

# DiD
import pandas as pd
pv = df.pivot(index='grp', columns='period', values='ai_rate_pct')
pv['delta'] = pv['2026'] - pv['2024']
print("\n=== Delta (pp) ===")
print(pv.to_string())
print(f"\nDiD SWE - control:  {pv.loc['SWE','delta'] - pv.loc['control','delta']:.2f}pp")
print(f"DiD SWE - adjacent: {pv.loc['SWE','delta'] - pv.loc['SWE_adjacent','delta']:.2f}pp")

# T18 reported: SWE +22.9, adjacent +19.0, control +1.2; DiD SWE-ctrl +21.7, SWE-adj +3.9

# Sensitivity: restrict adjacent to embedding_adjacent only
q2 = f"""
WITH f AS (
  SELECT uid, description, swe_classification_tier,
    CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period,
    is_swe, is_swe_adjacent, is_control,
    CASE WHEN regexp_matches(lower(description), $${AI_PAT}$$) THEN 1 ELSE 0 END AS has_ai
  FROM '{UNI}'
  WHERE {BASE}
)
SELECT period, count(*) AS n, avg(has_ai)*100 AS ai_rate_pct
FROM f
WHERE is_swe_adjacent = TRUE AND swe_classification_tier = 'embedding_adjacent'
GROUP BY 1 ORDER BY 1
"""
print("\n=== SWE-adjacent (embedding_adjacent ONLY) AI rate ===")
print(con.execute(q2).fetchdf().to_string())

# AI Engineer title spot check
q3 = f"""
WITH f AS (
  SELECT title_normalized, description,
    CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period,
    CASE WHEN regexp_matches(lower(description), $${AI_PAT}$$) THEN 1 ELSE 0 END AS has_ai,
    CASE WHEN regexp_matches(lower(description), 'agentic') THEN 1 ELSE 0 END AS has_agentic,
    CASE WHEN regexp_matches(lower(description), 'pytorch') THEN 1 ELSE 0 END AS has_pytorch,
    CASE WHEN regexp_matches(lower(description), '\\bllm\\b') THEN 1 ELSE 0 END AS has_llm
  FROM '{UNI}'
  WHERE {BASE} AND is_swe=TRUE AND lower(title_normalized) = 'ai engineer'
)
SELECT period, count(*) AS n,
  avg(has_ai)*100 AS ai_pct,
  avg(has_agentic)*100 AS agentic_pct,
  avg(has_pytorch)*100 AS pytorch_pct,
  avg(has_llm)*100 AS llm_pct
FROM f GROUP BY 1 ORDER BY 1
"""
print("\n=== 'AI Engineer' title evolution ===")
print(con.execute(q3).fetchdf().to_string())
