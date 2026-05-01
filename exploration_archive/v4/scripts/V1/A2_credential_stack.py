"""V1 Part A task 2 — Credential stack depth.

Independent re-derivation of T11's '3.8% -> 20.5% for depth 7' headline.
Defines 7 categories from scratch using simple regex lists:
  1 any technology mention (from a small reference list)
  2 any soft skill term
  3 any organizational scope term
  4 any explicit education mention
  5 yoe_extracted non-null
  6 any management/leadership term
  7 any AI mention
Shares are reported by period.
"""
import duckdb

DB = duckdb.connect()
DB.execute(
    "CREATE VIEW u AS SELECT * FROM read_parquet('/home/jihgaboot/gabor/job-research/data/unified.parquet')"
)
DB.execute(
    """
CREATE VIEW swe AS
SELECT *,
       CASE WHEN source IN ('kaggle_arshkon','kaggle_asaniczka') THEN '2024'
            WHEN source='scraped' THEN '2026' END AS period2,
       COALESCE(NULLIF(description_core_llm, ''), description_core, description) AS text_best
FROM u
WHERE source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE
"""
)

CATS = {
    "tech": r"\b(python|java|javascript|typescript|c\+\+|c\\\+\\\+|c\#|c sharp|go|rust|ruby|scala|kotlin|swift|sql|aws|azure|gcp|docker|kubernetes|react|node|django|spring|postgres|mysql)\b",
    "soft": r"\b(communicat\w*|collaborat\w*|teamwork|problem[- ]solving|interpersonal|leadership skills|organized|self[- ]motivated|analytical)\b",
    "scope": r"\b(cross[- ]functional|end[- ]to[- ]end|full[- ]stack|full stack|architect\w*|system design|scalable|distributed|large[- ]scale|enterprise|roadmap|strategy|stakeholder\w*|partner teams?)\b",
    "edu": r"\b(bs|b\.s\.|ms|m\.s\.|phd|ph\.d\.|bachelor\w*|master\w*|degree|doctorate|doctoral)\b",
    # yoe_extracted handled separately from numeric column
    "mgmt": r"\b(manage|manager|managing|lead|leading|mentor|mentoring|coach|coaching|hire|hiring|supervise|supervising|direct reports?)\b",
    "ai": r"\b(ai|machine[- ]learning|ml|llms?|deep[- ]learning|neural network|pytorch|tensorflow|nlp|generative ai|gen[- ]?ai|openai|copilot|claude|rag)\b",
}

parts = []
for name, pat in CATS.items():
    parts.append(
        f"(CASE WHEN regexp_matches(lower(text_best), '{pat}') THEN 1 ELSE 0 END) AS has_{name}"
    )
parts.append("(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS has_yoe")

q = f"""
WITH tagged AS (
  SELECT period2,
         {', '.join(parts)}
  FROM swe
)
SELECT period2,
       COUNT(*) AS n,
       AVG(has_tech) AS pct_tech,
       AVG(has_soft) AS pct_soft,
       AVG(has_scope) AS pct_scope,
       AVG(has_edu) AS pct_edu,
       AVG(has_yoe) AS pct_yoe,
       AVG(has_mgmt) AS pct_mgmt,
       AVG(has_ai) AS pct_ai,
       AVG(has_tech + has_soft + has_scope + has_edu + has_yoe + has_mgmt + has_ai) AS mean_depth,
       100.0 * AVG(CASE WHEN (has_tech + has_soft + has_scope + has_edu + has_yoe + has_mgmt + has_ai) = 7 THEN 1 ELSE 0 END) AS pct_depth7,
       100.0 * AVG(CASE WHEN (has_tech + has_soft + has_scope + has_edu + has_yoe + has_mgmt + has_ai) >= 6 THEN 1 ELSE 0 END) AS pct_depth_ge6,
       100.0 * AVG(CASE WHEN (has_tech + has_soft + has_scope + has_edu + has_yoe + has_mgmt + has_ai) >= 5 THEN 1 ELSE 0 END) AS pct_depth_ge5
FROM tagged GROUP BY 1 ORDER BY 1
"""
df = DB.execute(q).fetchdf()
import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
print(df.to_string(index=False))

print("\nRatio (2026 depth7 / 2024 depth7):")
p24 = float(df.loc[df['period2'] == '2024', 'pct_depth7'].iloc[0])
p26 = float(df.loc[df['period2'] == '2026', 'pct_depth7'].iloc[0])
print(f"  2024: {p24:.2f}%, 2026: {p26:.2f}%, ratio: {p26/p24:.2f}x, delta: {p26-p24:+.2f}pp")
