"""V1 Part D — company-capping sensitivity investigation.

Identifies the top 20 companies driving the 2026 entry-level pool under
  (a) combined best-available column
  (b) YOE<=2 proxy
Profiles each, and determines whether the 2026 entry-share rise is driven
by broad-based change or by a small set of high-volume companies.
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
       CASE
         WHEN llm_classification_coverage = 'labeled'         THEN seniority_llm
         WHEN llm_classification_coverage = 'rule_sufficient' THEN seniority_final
         ELSE NULL
       END AS seniority_best_available
FROM u
WHERE source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE
"""
)

print("\n=== D1: Top 20 scraped companies by ENTRY count (combined column) ===")
q_combined = """
SELECT company_name_canonical,
       COUNT(*) AS total_swe,
       COUNT(*) FILTER (WHERE seniority_best_available = 'entry') AS n_entry_combined,
       100.0 * COUNT(*) FILTER (WHERE seniority_best_available = 'entry') / COUNT(*) AS pct_entry_own,
       AVG(yoe_extracted) FILTER (WHERE seniority_best_available = 'entry') AS mean_yoe_entry,
       ANY_VALUE(company_industry) AS industry
FROM swe
WHERE period2 = '2026'
GROUP BY 1
HAVING COUNT(*) FILTER (WHERE seniority_best_available = 'entry') > 0
ORDER BY n_entry_combined DESC
LIMIT 20
"""
top_combined = DB.execute(q_combined).fetchdf()
import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_colwidth', 60)
print(top_combined.to_string(index=False))

print("\n=== D2: Top 20 scraped companies by YOE<=2 count ===")
q_yoe = """
SELECT company_name_canonical,
       COUNT(*) AS total_swe,
       COUNT(*) FILTER (WHERE yoe_extracted <= 2) AS n_entry_yoe,
       100.0 * COUNT(*) FILTER (WHERE yoe_extracted <= 2) / COUNT(*) AS pct_entry_own,
       AVG(yoe_extracted) FILTER (WHERE yoe_extracted <= 2) AS mean_yoe_entry,
       ANY_VALUE(company_industry) AS industry
FROM swe
WHERE period2 = '2026'
GROUP BY 1
HAVING COUNT(*) FILTER (WHERE yoe_extracted <= 2) > 0
ORDER BY n_entry_yoe DESC
LIMIT 20
"""
top_yoe = DB.execute(q_yoe).fetchdf()
print(top_yoe.to_string(index=False))

print("\n=== D3: Total entry pool and concentration (combined column) ===")
print(DB.execute(
    """
WITH combined_entry AS (
  SELECT company_name_canonical, COUNT(*) AS n
  FROM swe
  WHERE period2='2026' AND seniority_best_available='entry'
  GROUP BY 1
),
ranked AS (
  SELECT *, SUM(n) OVER () AS total_entry,
         ROW_NUMBER() OVER (ORDER BY n DESC) AS rk
  FROM combined_entry
)
SELECT
  SUM(n) AS total_entry_2026,
  SUM(CASE WHEN rk <= 10 THEN n ELSE 0 END) AS top10_sum,
  SUM(CASE WHEN rk <= 20 THEN n ELSE 0 END) AS top20_sum,
  SUM(CASE WHEN rk <= 50 THEN n ELSE 0 END) AS top50_sum,
  100.0 * SUM(CASE WHEN rk <= 10 THEN n ELSE 0 END) / SUM(n) AS pct_top10,
  100.0 * SUM(CASE WHEN rk <= 20 THEN n ELSE 0 END) / SUM(n) AS pct_top20,
  100.0 * SUM(CASE WHEN rk <= 50 THEN n ELSE 0 END) / SUM(n) AS pct_top50
FROM ranked
    """
).fetchdf().to_string(index=False))

print("\n=== D4: Same concentration under YOE<=2 ===")
print(DB.execute(
    """
WITH yoe_entry AS (
  SELECT company_name_canonical, COUNT(*) AS n
  FROM swe
  WHERE period2='2026' AND yoe_extracted <= 2
  GROUP BY 1
),
ranked AS (
  SELECT *, SUM(n) OVER () AS total_entry,
         ROW_NUMBER() OVER (ORDER BY n DESC) AS rk
  FROM yoe_entry
)
SELECT
  SUM(n) AS total_yoe_entry_2026,
  100.0 * SUM(CASE WHEN rk <= 10 THEN n ELSE 0 END) / SUM(n) AS pct_top10,
  100.0 * SUM(CASE WHEN rk <= 20 THEN n ELSE 0 END) / SUM(n) AS pct_top20,
  100.0 * SUM(CASE WHEN rk <= 50 THEN n ELSE 0 END) / SUM(n) AS pct_top50
FROM ranked
    """
).fetchdf().to_string(index=False))

print("\n=== D5: Same concentration comparisons for 2024 (to contrast) ===")
print(DB.execute(
    """
WITH combined_entry AS (
  SELECT company_name_canonical, COUNT(*) AS n
  FROM swe
  WHERE period2='2024' AND seniority_best_available='entry'
  GROUP BY 1
),
ranked AS (
  SELECT *, ROW_NUMBER() OVER (ORDER BY n DESC) AS rk FROM combined_entry
)
SELECT
  SUM(n) AS total_entry_2024,
  100.0 * SUM(CASE WHEN rk <= 10 THEN n ELSE 0 END) / SUM(n) AS pct_top10,
  100.0 * SUM(CASE WHEN rk <= 20 THEN n ELSE 0 END) / SUM(n) AS pct_top20
FROM ranked
    """
).fetchdf().to_string(index=False))
