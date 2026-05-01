"""
T18 Step 4 — Specific adjacent titles and AI Engineer profile.
"""
import os
import duckdb
import pandas as pd

PARQUET = "data/unified.parquet"
OUT_DIR = "exploration/tables/T18"
os.makedirs(OUT_DIR, exist_ok=True)

con = duckdb.connect()

BASE = """source_platform='linkedin' AND is_english=true AND date_flag='ok'"""

# 1. AI Engineer title profile
q_aie = f"""
SELECT period, COUNT(*) as n,
  AVG(LENGTH(coalesce(description_core_llm, description_core, description))::DOUBLE) as mean_len,
  AVG(CASE WHEN regexp_matches(lower(coalesce(description_core_llm, description_core, description)), 'agentic') THEN 1.0 ELSE 0.0 END) as agentic_rate,
  AVG(CASE WHEN regexp_matches(lower(coalesce(description_core_llm, description_core, description)), 'pytorch') THEN 1.0 ELSE 0.0 END) as pytorch_rate,
  AVG(CASE WHEN regexp_matches(lower(coalesce(description_core_llm, description_core, description)), '\\blangchain\\b|\\blangraph\\b|langgraph|\\bllm\\b|\\brag\\b') THEN 1.0 ELSE 0.0 END) as llm_stack_rate,
  AVG(CASE WHEN is_swe THEN 1.0 ELSE 0.0 END) as swe_frac,
  AVG(CASE WHEN is_swe_adjacent THEN 1.0 ELSE 0.0 END) as adj_frac
FROM '{PARQUET}'
WHERE {BASE} AND (lower(title_normalized) LIKE '%ai engineer%' OR lower(title_normalized) LIKE '%ai/ml engineer%')
GROUP BY period ORDER BY period
"""
aie = con.execute(q_aie).df()
print("=== AI Engineer title profile ===")
print(aie.to_string())
aie.to_csv(os.path.join(OUT_DIR, "ai_engineer_profile.csv"), index=False)

# 2. Top adjacent titles — profile change
q_adj_titles = f"""
SELECT title_normalized, COUNT(*) as n
FROM '{PARQUET}'
WHERE {BASE} AND is_swe_adjacent = true
GROUP BY title_normalized ORDER BY n DESC LIMIT 10
"""
top_adj = con.execute(q_adj_titles).df()
print("\n=== Top 10 adjacent titles (all periods) ===")
print(top_adj)

# For the top 5 adjacent titles, period-by-period
rows = []
for title in top_adj.title_normalized.head(5):
    q = f"""
    SELECT '{title.replace("'", "''")}' AS title, period, COUNT(*) as n,
      AVG(CASE WHEN regexp_matches(lower(coalesce(description_core_llm, description_core, description)), 'agentic|multi[- ]agent|\\bllm\\b|\\brag\\b|generative ai|\\bcopilot\\b') THEN 1.0 ELSE 0.0 END) as ai_rate,
      AVG(CASE WHEN regexp_matches(lower(coalesce(description_core_llm, description_core, description)), 'python') THEN 1.0 ELSE 0.0 END) as python_rate,
      median(LENGTH(coalesce(description_core_llm, description_core, description))) as median_len
    FROM '{PARQUET}'
    WHERE {BASE} AND title_normalized = '{title.replace("'", "''")}'
    GROUP BY period ORDER BY period
    """
    rows.append(con.execute(q).df())
adj_profile = pd.concat(rows, ignore_index=True)
print("\n=== Top adjacent titles profile by period ===")
print(adj_profile.to_string())
adj_profile.to_csv(os.path.join(OUT_DIR, "adjacent_titles_profile.csv"), index=False)

# 3. AI gradient plot data (SWE vs adjacent vs control, AI rate by period)
q_grad = f"""
WITH base AS (
  SELECT
    CASE WHEN is_swe THEN 'SWE' WHEN is_swe_adjacent THEN 'SWE_adjacent' WHEN is_control THEN 'control' END AS occ,
    period,
    CASE WHEN regexp_matches(lower(coalesce(description_core_llm, description_core, description)), 'agentic|multi[- ]agent|\\bai[- ]agent|\\bllm\\b|\\brag\\b|generative ai|\\bcopilot\\b|ai[- ]powered') THEN 1 ELSE 0 END AS ai_flag,
    CASE WHEN regexp_matches(lower(coalesce(description_core_llm, description_core, description)), 'agentic') THEN 1 ELSE 0 END AS agentic_flag
  FROM '{PARQUET}'
  WHERE {BASE} AND (is_swe OR is_swe_adjacent OR is_control)
)
SELECT occ, period, COUNT(*) as n, AVG(ai_flag::DOUBLE) as ai_rate, AVG(agentic_flag::DOUBLE) as agentic_rate
FROM base
WHERE occ IS NOT NULL
GROUP BY occ, period ORDER BY occ, period
"""
grad = con.execute(q_grad).df()
print("\n=== AI gradient by occupation group ===")
print(grad.to_string())
grad.to_csv(os.path.join(OUT_DIR, "ai_gradient_by_period.csv"), index=False)
