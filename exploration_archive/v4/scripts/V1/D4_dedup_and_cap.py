"""Combined dedup + cap sensitivities and entry-share rise verdict."""
import duckdb
import pandas as pd
pd.set_option('display.width', 220)

DB = duckdb.connect()
DB.execute("CREATE VIEW u AS SELECT * FROM read_parquet('/home/jihgaboot/gabor/job-research/data/unified.parquet')")
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

print("\n=== Sensitivity grid: entry share by cap type ===")
print("(Each row shows entry share under combined best-available and YOE<=2)")
# Baseline
rows = []

def record(name, df):
    for _, r in df.iterrows():
        rows.append({'variant': name, 'period': r['period2'], **{k: r[k] for k in df.columns if k != 'period2'}})

df = DB.execute("""
SELECT period2,
       100.0 * COUNT(*) FILTER (WHERE seniority_best_available='entry') / NULLIF(COUNT(*) FILTER (WHERE seniority_best_available IS NOT NULL), 0) AS pct_entry,
       100.0 * COUNT(*) FILTER (WHERE yoe_extracted<=2) / NULLIF(COUNT(*) FILTER (WHERE yoe_extracted IS NOT NULL), 0) AS pct_yoe_le2,
       COUNT(*) AS n
FROM swe GROUP BY 1 ORDER BY 1
""").fetchdf()
record('(a) baseline', df)

# Cap 20
df = DB.execute("""
WITH capped AS (
  SELECT period2, seniority_best_available, yoe_extracted,
         ROW_NUMBER() OVER (PARTITION BY period2, company_name_canonical ORDER BY uid) AS rn
  FROM swe
) SELECT period2,
       100.0 * COUNT(*) FILTER (WHERE seniority_best_available='entry') / NULLIF(COUNT(*) FILTER (WHERE seniority_best_available IS NOT NULL), 0) AS pct_entry,
       100.0 * COUNT(*) FILTER (WHERE yoe_extracted<=2) / NULLIF(COUNT(*) FILTER (WHERE yoe_extracted IS NOT NULL), 0) AS pct_yoe_le2,
       COUNT(*) AS n
FROM capped WHERE rn <= 20 GROUP BY 1 ORDER BY 1
""").fetchdf()
record('(b) cap20 per company', df)

# Dedup by description_hash within period
df = DB.execute("""
WITH dedup AS (
  SELECT period2, description_hash,
         ANY_VALUE(seniority_best_available) AS seniority_best_available,
         ANY_VALUE(yoe_extracted) AS yoe_extracted
  FROM swe WHERE description_hash IS NOT NULL GROUP BY 1,2
) SELECT period2,
       100.0 * COUNT(*) FILTER (WHERE seniority_best_available='entry') / NULLIF(COUNT(*) FILTER (WHERE seniority_best_available IS NOT NULL), 0) AS pct_entry,
       100.0 * COUNT(*) FILTER (WHERE yoe_extracted<=2) / NULLIF(COUNT(*) FILTER (WHERE yoe_extracted IS NOT NULL), 0) AS pct_yoe_le2,
       COUNT(*) AS n
FROM dedup GROUP BY 1 ORDER BY 1
""").fetchdf()
record('(c) dedup desc_hash', df)

# Dedup + cap 20
df = DB.execute("""
WITH dedup AS (
  SELECT period2, description_hash,
         ANY_VALUE(seniority_best_available) AS seniority_best_available,
         ANY_VALUE(yoe_extracted) AS yoe_extracted,
         ANY_VALUE(company_name_canonical) AS company_name_canonical
  FROM swe WHERE description_hash IS NOT NULL GROUP BY 1,2
), capped AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY period2, company_name_canonical ORDER BY description_hash) AS rn
  FROM dedup
) SELECT period2,
       100.0 * COUNT(*) FILTER (WHERE seniority_best_available='entry') / NULLIF(COUNT(*) FILTER (WHERE seniority_best_available IS NOT NULL), 0) AS pct_entry,
       100.0 * COUNT(*) FILTER (WHERE yoe_extracted<=2) / NULLIF(COUNT(*) FILTER (WHERE yoe_extracted IS NOT NULL), 0) AS pct_yoe_le2,
       COUNT(*) AS n
FROM capped WHERE rn <= 20 GROUP BY 1 ORDER BY 1
""").fetchdf()
record('(d) dedup + cap20', df)

# Exclude aggregators
df = DB.execute("""
SELECT period2,
       100.0 * COUNT(*) FILTER (WHERE seniority_best_available='entry') / NULLIF(COUNT(*) FILTER (WHERE seniority_best_available IS NOT NULL), 0) AS pct_entry,
       100.0 * COUNT(*) FILTER (WHERE yoe_extracted<=2) / NULLIF(COUNT(*) FILTER (WHERE yoe_extracted IS NOT NULL), 0) AS pct_yoe_le2,
       COUNT(*) AS n
FROM swe WHERE is_aggregator = FALSE GROUP BY 1 ORDER BY 1
""").fetchdf()
record('(e) exclude aggregators', df)

# Exclude top 20 combined contributors
top20 = ['TikTok', 'Affirm', 'Canonical', 'ByteDance', 'Cisco', 'Epic',
         'Jobs via Dice', 'SMX', 'WayUp', 'Google', 'Uber', 'Leidos',
         'General Motors', 'SkillStorm', 'Amazon', 'SynergisticIT',
         'Lockheed Martin', 'Emonics LLC', 'HP', 'Applied Materials']
names = "', '".join([c.replace("'", "''") for c in top20])
df = DB.execute(f"""
SELECT period2,
       100.0 * COUNT(*) FILTER (WHERE seniority_best_available='entry') / NULLIF(COUNT(*) FILTER (WHERE seniority_best_available IS NOT NULL), 0) AS pct_entry,
       100.0 * COUNT(*) FILTER (WHERE yoe_extracted<=2) / NULLIF(COUNT(*) FILTER (WHERE yoe_extracted IS NOT NULL), 0) AS pct_yoe_le2,
       COUNT(*) AS n
FROM swe WHERE company_name_canonical NOT IN ('{names}') GROUP BY 1 ORDER BY 1
""").fetchdf()
record('(f) excl. top20 contributors', df)

out = pd.DataFrame(rows)
print(out.to_string(index=False))
