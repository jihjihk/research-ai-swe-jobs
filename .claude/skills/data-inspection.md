# Skill: Data inspection with DuckDB

Use this skill whenever you need to inspect parquet files or CSVs in this project. DuckDB is installed in the project venv and reads parquet/CSV files directly without loading them into memory.

## When to use

- Checking row counts, column types, null rates
- Grouped summaries (by source, seniority, period, etc.)
- Spot-checking random samples
- Comparing distributions across datasets or pipeline stages
- Any data question that can be answered with SQL

## How to run

```bash
.venv/bin/python -c "
import duckdb
print(duckdb.sql(\"YOUR SQL HERE\").df())
"
```

## Common queries for this project

### Row counts and schema
```sql
SELECT count(*) FROM 'data/unified.parquet'
DESCRIBE SELECT * FROM 'data/unified.parquet'
```

### Source breakdown
```sql
SELECT source, count(*) n, count(DISTINCT company_name) companies
FROM 'data/unified.parquet' GROUP BY source ORDER BY n DESC
```

### Seniority distribution by source
```sql
SELECT source, seniority_native, count(*) n
FROM 'data/unified.parquet'
WHERE is_swe GROUP BY source, seniority_native ORDER BY source, n DESC
```

### Null rates for key fields
```sql
SELECT
  count(*) total,
  sum(CASE WHEN description IS NULL OR description = '' THEN 1 ELSE 0 END) desc_null,
  sum(CASE WHEN seniority_native IS NULL OR seniority_native = '' THEN 1 ELSE 0 END) seniority_null,
  sum(CASE WHEN min_salary IS NULL THEN 1 ELSE 0 END) salary_null
FROM 'data/unified.parquet'
```

### Compare two pipeline stages
```sql
SELECT 'stage1' stage, count(*) FROM 'preprocessing/intermediate/stage1_unified.parquet'
UNION ALL
SELECT 'stage4', count(*) FROM 'preprocessing/intermediate/stage4_dedup.parquet'
```

### Spot-check random samples
```sql
SELECT title, company_name, seniority_native, description[:200] desc_preview
FROM 'data/unified.parquet' WHERE is_swe
USING SAMPLE 5
```

### Raw CSV inspection (scraped data)
```sql
SELECT site, query_tier, count(*) n
FROM 'data/scraped/2026-03-21_swe_jobs.csv' GROUP BY site, query_tier
```

### Asaniczka with description join
```sql
SELECT m.job_title, m.job_level, length(s.job_summary) desc_len
FROM 'data/kaggle-asaniczka-1.3m/linkedin_job_postings.csv' m
JOIN 'data/kaggle-asaniczka-1.3m/job_summary.csv' s ON m.job_link = s.job_link
WHERE m.search_country = 'United States'
LIMIT 5
```

## Key file paths

- `data/unified.parquet` — final analysis-ready output
- `data/unified_observations.parquet` — daily panel
- `preprocessing/intermediate/stage*.parquet` — pipeline stage outputs
- `data/scraped/2026-03-{20,21}_swe_jobs.csv` — new-format scraped data
- `data/kaggle-linkedin-jobs-2023-2024/postings.csv` — arshkon
- `data/kaggle-asaniczka-1.3m/linkedin_job_postings.csv` — asaniczka main
- `data/kaggle-asaniczka-1.3m/job_summary.csv` — asaniczka descriptions
- `data/kaggle-asaniczka-1.3m/job_skills.csv` — asaniczka skills

## When NOT to use DuckDB

- Writing pipeline transformation code (use pyarrow chunked I/O instead)
- Processing that needs to modify data in place
- Complex Python-specific logic (regex, NLP, embeddings)
