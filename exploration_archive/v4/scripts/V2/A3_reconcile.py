"""V2 Part A reconciliation: merge matrix with corrected c++/c#, compute corrected tech_count.

Also check: does T19's 'safe 15' list really show flat growth? Replicate it.
"""

import duckdb

con = duckdb.connect()

MATRIX = "exploration/artifacts/shared/swe_tech_matrix.parquet"
UNI = "data/unified.parquet"

# 1. Compute corrected matrix tech_count: matrix minus c_cpp/csharp, plus V2-independent c++/c#
mcols = con.execute(f"DESCRIBE SELECT * FROM '{MATRIX}' LIMIT 1").fetchdf()
TECHS = [c for c in mcols['column_name'].tolist() if c != 'uid']
FIXED_TECHS = [t for t in TECHS if t not in ('c_cpp', 'csharp')]

sum_expr_fixed = " + ".join([f"CAST(m.{t} AS INT)" for t in FIXED_TECHS])

q = f"""
WITH m AS (SELECT * FROM '{MATRIX}'),
     u AS (SELECT uid, description,
                  CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period
           FROM '{UNI}'
           WHERE source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE),
     j AS (
       SELECT u.uid, u.period,
              ({sum_expr_fixed}) AS tc_matrix_fixed,
              (CAST(m.c_cpp AS INT) + CAST(m.csharp AS INT)) AS tc_matrix_broken,
              CASE WHEN position('c++' in lower(u.description)) > 0
                        OR position('c\\+\\+' in lower(u.description)) > 0
                        OR position(' cpp ' in lower(u.description)) > 0 THEN 1 ELSE 0 END AS has_cpp,
              CASE WHEN position('c#' in lower(u.description)) > 0
                        OR position('c\\#' in lower(u.description)) > 0 THEN 1 ELSE 0 END AS has_csharp
       FROM u JOIN m USING (uid)
     ),
     k AS (
       SELECT period, uid,
              tc_matrix_fixed + tc_matrix_broken AS tc_matrix_raw,
              tc_matrix_fixed + has_cpp + has_csharp AS tc_matrix_corrected
       FROM j
     )
SELECT period, count(*) AS n,
       avg(tc_matrix_raw) AS mean_raw,
       median(tc_matrix_raw) AS median_raw,
       avg(tc_matrix_corrected) AS mean_corrected,
       median(tc_matrix_corrected) AS median_corrected
FROM k GROUP BY 1 ORDER BY 1
"""
print("=== Matrix tech_count: raw (broken c++/c#) vs corrected ===")
df = con.execute(q).fetchdf()
print(df.to_string())

# Deltas
r2024 = df.iloc[0]; r2026 = df.iloc[1]
mean_delta_raw = (r2026['mean_raw'] - r2024['mean_raw']) / r2024['mean_raw'] * 100
mean_delta_fix = (r2026['mean_corrected'] - r2024['mean_corrected']) / r2024['mean_corrected'] * 100
med_delta_raw  = (r2026['median_raw'] - r2024['median_raw']) / r2024['median_raw'] * 100
med_delta_fix  = (r2026['median_corrected'] - r2024['median_corrected']) / r2024['median_corrected'] * 100
print(f"\nRaw (broken) matrix:      mean Δ = {mean_delta_raw:.1f}%   median Δ = {med_delta_raw:.1f}%")
print(f"c++/c#-corrected matrix:  mean Δ = {mean_delta_fix:.1f}%   median Δ = {med_delta_fix:.1f}%")

# 2. T19 safe-15 list replication
SAFE_15 = ['python', 'java', 'javascript', 'typescript', 'sql', 'aws', 'azure',
           'gcp', 'kubernetes', 'docker', 'terraform', 'react', 'angular', 'node', 'git']
# Use the matrix versions except for 'git' which is github in the matrix
# Columns: python, java, javascript, typescript, sql, aws, azure, gcp, kubernetes,
#          docker, terraform, react, angular, nodejs, ...
# Matrix doesn't have a bare 'git' so we skip; safe 14.
MATRIX_SAFE = ['python', 'java', 'javascript', 'typescript', 'sql', 'aws', 'azure',
               'gcp', 'kubernetes', 'docker', 'terraform', 'react', 'angular', 'nodejs']
sum_safe = " + ".join([f"CAST({t} AS INT)" for t in MATRIX_SAFE])
q2 = f"""
WITH m AS (
  SELECT uid, ({sum_safe}) AS tc_safe
  FROM '{MATRIX}'
),
u AS (
  SELECT uid, CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period
  FROM '{UNI}'
  WHERE source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE
)
SELECT period, count(*) AS n, avg(tc_safe) AS mean_tc, median(tc_safe) AS median_tc
FROM u JOIN m USING (uid)
GROUP BY 1 ORDER BY 1
"""
print(f"\n=== Matrix safe-14 (T19 replication) ===")
df2 = con.execute(q2).fetchdf()
print(df2.to_string())

# 3. Split T19 by 3 snapshots: 2024, 2026-03, 2026-04
q3 = f"""
WITH m AS (
  SELECT uid, ({sum_safe}) AS tc_safe
  FROM '{MATRIX}'
),
u AS (
  SELECT uid,
    CASE
      WHEN source='scraped' AND period = '2026-03' THEN '2026_03'
      WHEN source='scraped' AND period = '2026-04' THEN '2026_04'
      WHEN source='scraped' THEN '2026_other'
      ELSE '2024'
    END AS snap
  FROM '{UNI}'
  WHERE source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE
)
SELECT snap, count(*) AS n, avg(tc_safe) AS mean_tc, median(tc_safe) AS median_tc
FROM u JOIN m USING (uid)
GROUP BY 1 ORDER BY 1
"""
print(f"\n=== Safe-14 by period snapshot ===")
print(con.execute(q3).fetchdf().to_string())
