-- Seniority workaround for exploration (pre-Stage 5 bugfix)
--
-- Problem: seniority_final ignores ~9,500 SWE native labels due to a
-- native_backfill bug. This CTE replicates what the fixed pipeline will produce.
--
-- Usage: include this CTE at the top of any query, then use seniority_patched
-- and seniority_3level_patched instead of seniority_final / seniority_3level.
--
-- Example:
--   WITH swe_patched AS ( <paste CTE contents> )
--   SELECT seniority_patched, count(*) FROM swe_patched GROUP BY 1;

-- Copy this block into your queries as a CTE:
swe_patched AS (
  SELECT *,
    CASE
      WHEN seniority_final_source IN ('title_keyword', 'title_manager')
        THEN seniority_final
      WHEN seniority_native IS NOT NULL AND seniority_native != ''
        THEN CASE seniority_native
               WHEN 'intern' THEN 'entry'
               WHEN 'executive' THEN 'director'
               ELSE seniority_native
             END
      WHEN seniority_final != 'unknown'
        THEN seniority_final
      ELSE 'unknown'
    END AS seniority_patched,
    CASE
      WHEN seniority_final_source IN ('title_keyword', 'title_manager')
        THEN CASE seniority_final
               WHEN 'entry' THEN 'junior'
               WHEN 'associate' THEN 'mid'
               WHEN 'mid-senior' THEN 'senior'
               WHEN 'director' THEN 'senior'
               ELSE 'unknown'
             END
      WHEN seniority_native IS NOT NULL AND seniority_native != ''
        THEN CASE seniority_native
               WHEN 'entry' THEN 'junior'
               WHEN 'intern' THEN 'junior'
               WHEN 'associate' THEN 'mid'
               WHEN 'mid-senior' THEN 'senior'
               WHEN 'director' THEN 'senior'
               WHEN 'executive' THEN 'senior'
               ELSE 'unknown'
             END
      WHEN seniority_final != 'unknown'
        THEN CASE seniority_final
               WHEN 'entry' THEN 'junior'
               WHEN 'associate' THEN 'mid'
               WHEN 'mid-senior' THEN 'senior'
               WHEN 'director' THEN 'senior'
               ELSE 'unknown'
             END
      ELSE 'unknown'
    END AS seniority_3level_patched,
    CASE
      WHEN seniority_final_source IN ('title_keyword', 'title_manager')
        THEN 'title_strong'
      WHEN seniority_native IS NOT NULL AND seniority_native != ''
        THEN 'native_backfill'
      WHEN seniority_final != 'unknown'
        THEN 'weak_signal'
      ELSE 'unknown'
    END AS seniority_patched_source
  FROM 'preprocessing/intermediate/stage8_final.parquet'
  WHERE source_platform = 'linkedin'
    AND is_english = true
    AND date_flag = 'ok'
)
