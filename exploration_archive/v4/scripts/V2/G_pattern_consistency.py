"""V2 Part G: Cross-task pattern consistency check.

Compute strict_mentor rate using:
  (a) T22 validated pattern (from JSON)
  (b) T11's pattern (from T11 script)
  (c) T21's pattern (from T21 script if available)
Compare on the same SWE 2024/2026 sample.
"""

import duckdb
import json

con = duckdb.connect()

with open("exploration/artifacts/shared/validated_mgmt_patterns.json") as f:
    P = json.load(f)

T22_MENTOR = P["mgmt_strict"]["strict_mentor"]
# T11's mentor pattern (from inspection of script — simpler)
T11_MENTOR = r"mentor\s+(junior|engineers?|developers?)|coach\s+engineers?"
# T21's: similar

UNI = "data/unified.parquet"
BASE = "source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE"

q = f"""
WITH f AS (
  SELECT
    CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period,
    CASE WHEN regexp_matches(lower(coalesce(description_core_llm, description_core, description)), $${T22_MENTOR}$$) THEN 1 ELSE 0 END AS t22_mentor,
    CASE WHEN regexp_matches(lower(coalesce(description_core_llm, description_core, description)), $${T11_MENTOR}$$) THEN 1 ELSE 0 END AS t11_mentor
  FROM '{UNI}'
  WHERE {BASE}
)
SELECT period, count(*) AS n,
       avg(t22_mentor)*100 AS pct_t22,
       avg(t11_mentor)*100 AS pct_t11,
       (avg(t22_mentor)-avg(t11_mentor))*100 AS diff_pp
FROM f GROUP BY 1 ORDER BY 1
"""
df = con.execute(q).fetchdf()
print("=== Strict mentor rate: T22 pattern vs T11 pattern ===")
print(df.to_string())

# Relative difference
r24 = df.iloc[0]; r26 = df.iloc[1]
print(f"\n2024: T22={r24['pct_t22']:.2f}%, T11={r24['pct_t11']:.2f}%, relative diff = {(r24['pct_t22']/r24['pct_t11']-1)*100:.1f}%")
print(f"2026: T22={r26['pct_t22']:.2f}%, T11={r26['pct_t11']:.2f}%, relative diff = {(r26['pct_t22']/r26['pct_t11']-1)*100:.1f}%")

# Also try ai_tool
T22_AI_TOOL = P["ai"]["ai_tool"]
# T11 ai_mention (from T11: just \bai\b|artificial intelligence)
T11_AI = r"\bai\b|artificial intelligence"
q2 = f"""
WITH f AS (
  SELECT
    CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period,
    CASE WHEN regexp_matches(lower(description), $${T22_AI_TOOL}$$) THEN 1 ELSE 0 END AS t22_ai_tool,
    CASE WHEN regexp_matches(lower(description), $${T11_AI}$$) THEN 1 ELSE 0 END AS t11_ai_gen
  FROM '{UNI}'
  WHERE {BASE}
)
SELECT period, count(*) AS n,
       avg(t22_ai_tool)*100 AS pct_t22_tool,
       avg(t11_ai_gen)*100 AS pct_t11_gen
FROM f GROUP BY 1 ORDER BY 1
"""
print("\n=== AI rate: T22 ai_tool vs T11 ai_general ===")
print(con.execute(q2).fetchdf().to_string())
