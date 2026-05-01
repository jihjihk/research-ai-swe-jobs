"""V1 Part A task 3 — Entry share under combined column, imputed, and YOE proxy."""
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

print("\n=== Combined best-available, entry share (over known rows) ===")
print(DB.execute(
    """
SELECT period2,
       COUNT(*) FILTER (WHERE seniority_best_available IS NOT NULL) AS n_known,
       COUNT(*) FILTER (WHERE seniority_best_available = 'entry') AS n_entry,
       100.0 * COUNT(*) FILTER (WHERE seniority_best_available = 'entry')
           / NULLIF(COUNT(*) FILTER (WHERE seniority_best_available IS NOT NULL), 0) AS pct_entry_known,
       100.0 * COUNT(*) FILTER (WHERE seniority_best_available = 'entry') / COUNT(*) AS pct_entry_overall
FROM swe
GROUP BY 1 ORDER BY 1
    """
).fetchdf().to_string(index=False))

print("\n=== seniority_imputed entry share (where != unknown) ===")
print(DB.execute(
    """
SELECT period2,
       COUNT(*) FILTER (WHERE seniority_imputed IS NOT NULL AND seniority_imputed != 'unknown') AS n_known,
       COUNT(*) FILTER (WHERE seniority_imputed = 'entry') AS n_entry,
       100.0 * COUNT(*) FILTER (WHERE seniority_imputed = 'entry')
           / NULLIF(COUNT(*) FILTER (WHERE seniority_imputed IS NOT NULL AND seniority_imputed != 'unknown'), 0) AS pct_entry_known
FROM swe
GROUP BY 1 ORDER BY 1
    """
).fetchdf().to_string(index=False))

print("\n=== YOE<=2 proxy (over rows with yoe_extracted non-null) ===")
print(DB.execute(
    """
SELECT period2,
       COUNT(*) FILTER (WHERE yoe_extracted IS NOT NULL) AS n_with_yoe,
       COUNT(*) FILTER (WHERE yoe_extracted <= 2) AS n_yoe_le2,
       100.0 * COUNT(*) FILTER (WHERE yoe_extracted <= 2)
           / NULLIF(COUNT(*) FILTER (WHERE yoe_extracted IS NOT NULL), 0) AS pct_yoe_le2_of_known,
       100.0 * COUNT(*) FILTER (WHERE yoe_extracted <= 2) / COUNT(*) AS pct_yoe_le2_overall
FROM swe
GROUP BY 1 ORDER BY 1
    """
).fetchdf().to_string(index=False))

print("\n=== seniority_native entry share (arshkon-only baseline) ===")
print(DB.execute(
    """
SELECT period2,
       COUNT(*) AS n,
       COUNT(*) FILTER (WHERE seniority_native IS NOT NULL) AS n_known,
       100.0 * COUNT(*) FILTER (WHERE seniority_native = 'entry')
           / NULLIF(COUNT(*) FILTER (WHERE seniority_native IS NOT NULL), 0) AS pct_entry_native_known
FROM swe
WHERE source != 'kaggle_asaniczka'
GROUP BY 1 ORDER BY 1
    """
).fetchdf().to_string(index=False))
