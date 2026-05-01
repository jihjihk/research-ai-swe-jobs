"""V1 Part C — verify C++ cooccurs with systems-community anchors.

If C++ had been correctly detected, it would anchor a systems community with
Linux/kernel/embedded/firmware/assembly. We check the Jaccard co-occurrence
between corrected C++ detection and a small set of systems tokens.
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

CPP = (
    "(POSITION(' c++' IN lower(text_best)) > 0 OR POSITION(' c\\+\\+' IN lower(text_best)) > 0)"
)
PARTNERS = {
    "python": r"\bpython\b",
    "java": r"\bjava\b",
    "linux": r"\blinux\b",
    "kernel": r"\bkernel\b",
    "embedded": r"\bembedded\b",
    "firmware": r"\bfirmware\b",
    "assembly": r"\bassembly\b",
    "rust": r"\brust\b",
    "rtos": r"\brtos\b",
    "pytorch": r"\bpytorch\b",
    "react": r"\breact\b",
    "aws": r"\baws\b",
}

for name, pat in PARTNERS.items():
    q = f"""
    WITH t AS (
      SELECT period2,
             ({CPP}) AS cpp,
             (regexp_matches(lower(text_best), '{pat}')) AS p
      FROM swe
    )
    SELECT period2,
           SUM(CASE WHEN cpp AND p THEN 1 ELSE 0 END)::DOUBLE /
             NULLIF(SUM(CASE WHEN cpp OR p THEN 1 ELSE 0 END), 0) AS jaccard,
           SUM(CASE WHEN cpp AND p THEN 1 ELSE 0 END) AS n_both,
           SUM(CASE WHEN cpp THEN 1 ELSE 0 END) AS n_cpp,
           SUM(CASE WHEN p THEN 1 ELSE 0 END) AS n_partner
    FROM t GROUP BY 1 ORDER BY 1
    """
    df = DB.execute(q).fetchdf()
    print(f"\n--- C++ x {name} ---")
    print(df.to_string(index=False))
