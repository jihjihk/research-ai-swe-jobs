"""V1 Part D final — quantify duplicate templates and reach verdict."""
import duckdb
import pandas as pd
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_colwidth', 120)

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

print("=== Duplicate posting templates among top 20 combined entry contributors ===")
print("(How many entry postings share an identical description_hash per company?)")
top20 = ['TikTok', 'Affirm', 'Canonical', 'ByteDance', 'Cisco', 'Epic',
         'Jobs via Dice', 'SMX', 'WayUp', 'Google', 'Uber', 'Leidos',
         'General Motors', 'SkillStorm', 'Amazon', 'SynergisticIT',
         'Lockheed Martin', 'Emonics LLC', 'HP', 'Applied Materials']
names = "', '".join([c.replace("'", "''") for c in top20])
q = f"""
SELECT company_name_canonical,
       COUNT(*) AS n_entry,
       COUNT(DISTINCT description_hash) AS n_unique_desc,
       COUNT(DISTINCT title) AS n_unique_title,
       ROUND(100.0 * COUNT(*)::DOUBLE / COUNT(DISTINCT description_hash), 1) AS dup_ratio_pct
FROM swe
WHERE period2='2026' AND seniority_best_available='entry'
  AND company_name_canonical IN ('{names}')
GROUP BY 1 ORDER BY n_entry DESC
"""
print(DB.execute(q).fetchdf().to_string(index=False))

print("\n=== SynergisticIT full sample: are these bootcamp ads? ===")
print(DB.execute(
    """
SELECT title, substr(COALESCE(NULLIF(description_core_llm,''), description_core, description), 1, 300) AS snippet
FROM swe
WHERE period2='2026' AND company_name_canonical='SynergisticIT' AND seniority_best_available='entry'
LIMIT 5
    """
).fetchdf().to_string(index=False))

# Check what the combined entry share looks like after dedup on description_hash per period
print("\n=== Entry share after DEDUPING by description_hash ===")
print(DB.execute(
    """
WITH dedup AS (
  SELECT period2, description_hash, ANY_VALUE(seniority_best_available) AS seniority,
         ANY_VALUE(yoe_extracted) AS yoe
  FROM swe GROUP BY 1, 2
)
SELECT period2,
       COUNT(*) FILTER (WHERE seniority IS NOT NULL) AS n_known,
       100.0 * COUNT(*) FILTER (WHERE seniority='entry') / NULLIF(COUNT(*) FILTER (WHERE seniority IS NOT NULL), 0) AS pct_entry_dedup,
       100.0 * COUNT(*) FILTER (WHERE yoe <= 2) / NULLIF(COUNT(*) FILTER (WHERE yoe IS NOT NULL), 0) AS pct_yoe_le2_dedup,
       COUNT(*) FILTER (WHERE yoe IS NOT NULL) AS n_yoe
FROM dedup GROUP BY 1 ORDER BY 1
    """
).fetchdf().to_string(index=False))

# Entry share after company-cap-20
print("\n=== Entry share after company-cap-20 ===")
print(DB.execute(
    """
WITH capped AS (
  SELECT period2, seniority_best_available, yoe_extracted,
         ROW_NUMBER() OVER (PARTITION BY period2, company_name_canonical ORDER BY uid) AS rn
  FROM swe
)
SELECT period2,
       COUNT(*) AS n,
       COUNT(*) FILTER (WHERE seniority_best_available IS NOT NULL) AS n_known,
       100.0 * COUNT(*) FILTER (WHERE seniority_best_available='entry') / NULLIF(COUNT(*) FILTER (WHERE seniority_best_available IS NOT NULL), 0) AS pct_entry_capped,
       100.0 * COUNT(*) FILTER (WHERE yoe_extracted <= 2) / NULLIF(COUNT(*) FILTER (WHERE yoe_extracted IS NOT NULL), 0) AS pct_yoe_le2_capped
FROM capped WHERE rn <= 20 GROUP BY 1 ORDER BY 1
    """
).fetchdf().to_string(index=False))

# Broad-based check: among companies with >= 5 SWE and any entry, what is the
# distribution of within-company entry shares?
print("\n=== Within-company entry share distribution (scraped, >=5 SWE) ===")
print(DB.execute(
    """
WITH per_co AS (
  SELECT company_name_canonical,
         COUNT(*) AS n_swe,
         COUNT(*) FILTER (WHERE seniority_best_available='entry') AS n_entry,
         100.0 * COUNT(*) FILTER (WHERE seniority_best_available='entry') / COUNT(*) AS pct_entry
  FROM swe WHERE period2='2026' GROUP BY 1 HAVING COUNT(*) >= 5
)
SELECT COUNT(*) AS n_co,
       COUNT(*) FILTER (WHERE n_entry > 0) AS n_co_with_entry,
       AVG(pct_entry) AS mean_pct_entry,
       MEDIAN(pct_entry) AS median_pct_entry
FROM per_co
    """
).fetchdf().to_string(index=False))
