"""V1 Part A task 4 — description length growth."""
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
            WHEN source='scraped' THEN '2026' END AS period2
FROM u
WHERE source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE
"""
)

print("\n=== Overall core_length and description_length by period ===")
print(DB.execute(
    """
SELECT period2,
       COUNT(*) AS n,
       MEDIAN(core_length) AS median_core,
       MEDIAN(description_length) AS median_desc,
       AVG(core_length) AS mean_core,
       AVG(description_length) AS mean_desc
FROM swe GROUP BY 1 ORDER BY 1
    """
).fetchdf().to_string(index=False))

print("\n=== Length of description_core_llm vs description_core by period ===")
print(DB.execute(
    """
SELECT period2,
       MEDIAN(LENGTH(description_core_llm)) FILTER (WHERE description_core_llm IS NOT NULL AND description_core_llm != '') AS median_llm,
       COUNT(*) FILTER (WHERE description_core_llm IS NOT NULL AND description_core_llm != '') AS n_llm,
       MEDIAN(LENGTH(description_core)) FILTER (WHERE description_core IS NOT NULL AND description_core != '') AS median_rule,
       COUNT(*) FILTER (WHERE description_core IS NOT NULL AND description_core != '') AS n_rule
FROM swe GROUP BY 1 ORDER BY 1
    """
).fetchdf().to_string(index=False))
