"""V1 Part C — re-derive C++ and C# mention rates with corrected patterns.

Handles markdown backslash escapes (`C\\+\\+`, `C\\#`) that appear in scraped
LinkedIn text. Reports share by period.
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

# C++ — needs (a) boundary on LEFT (not a preceding letter), (b) the two pluses
# with an optional backslash before each, (c) no trailing alphanumeric.
# Rule: match " c++" or " c\+\+" in lowercased text. Python regex via DuckDB
# regexp_matches uses RE2 but supports [^a-z]c\\+\\+ (we'll simulate via
# multiple alternatives).
# Simplest robust: LIKE '% c++%' OR LIKE '% c\\+\\+%'  (literal strings)
CPP_SQL = (
    "POSITION(' c++' IN lower(text_best)) > 0 "
    "OR POSITION(' c\\+\\+' IN lower(text_best)) > 0 "
    "OR POSITION('(c++' IN lower(text_best)) > 0 "
    "OR POSITION('/c++' IN lower(text_best)) > 0 "
    "OR POSITION(',c++' IN lower(text_best)) > 0 "
    "OR POSITION('(c\\+\\+' IN lower(text_best)) > 0 "
    "OR POSITION('/c\\+\\+' IN lower(text_best)) > 0 "
    "OR POSITION(',c\\+\\+' IN lower(text_best)) > 0"
)
CS_SQL = (
    "POSITION(' c#' IN lower(text_best)) > 0 "
    "OR POSITION(' c\\#' IN lower(text_best)) > 0 "
    "OR POSITION('(c#' IN lower(text_best)) > 0 "
    "OR POSITION('/c#' IN lower(text_best)) > 0 "
    "OR POSITION(',c#' IN lower(text_best)) > 0 "
    "OR POSITION('(c\\#' IN lower(text_best)) > 0 "
    "OR POSITION('/c\\#' IN lower(text_best)) > 0 "
    "OR POSITION(',c\\#' IN lower(text_best)) > 0"
)

print("\n=== C++ and C# mention rates (corrected) ===")
print(DB.execute(
    f"""
SELECT period2,
       COUNT(*) AS n,
       100.0 * AVG(CASE WHEN ({CPP_SQL}) THEN 1 ELSE 0 END) AS pct_cpp,
       100.0 * AVG(CASE WHEN ({CS_SQL}) THEN 1 ELSE 0 END) AS pct_csharp
FROM swe GROUP BY 1 ORDER BY 1
    """
).fetchdf().to_string(index=False))

print("\n=== Verify with existing shared tech matrix for comparison ===")
DB.execute(
    "CREATE VIEW tm AS SELECT * FROM read_parquet('/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_tech_matrix.parquet')"
)
print(DB.execute(
    """
SELECT s.period2,
       100.0 * AVG(CASE WHEN tm.c_cpp = TRUE THEN 1 ELSE 0 END) AS pct_cpp_matrix,
       100.0 * AVG(CASE WHEN tm.csharp = TRUE THEN 1 ELSE 0 END) AS pct_csharp_matrix,
       COUNT(*) AS n
FROM swe s JOIN tm ON s.uid = tm.uid
GROUP BY 1 ORDER BY 1
    """
).fetchdf().to_string(index=False))
