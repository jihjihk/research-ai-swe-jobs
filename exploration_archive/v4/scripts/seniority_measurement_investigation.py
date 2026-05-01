"""
Seniority Measurement Investigation
====================================
Comprehensive analysis of why entry-level SWE posting share trends flip
direction depending on the measurement method.

Produces all tables referenced in exploration/reports/seniority_measurement_investigation.md.

Usage:
    .venv/bin/python exploration/scripts/seniority_measurement_investigation.py

Outputs CSV tables to exploration/tables/seniority_measurement/.
"""

import os
import duckdb
import pandas as pd

OUT_DIR = "exploration/tables/seniority_measurement"
os.makedirs(OUT_DIR, exist_ok=True)

DATA = "data/unified.parquet"
BASE_FILTER = """
    source_platform = 'linkedin'
    AND is_english = true
    AND date_flag = 'ok'
    AND is_swe = true
"""
PERIOD_FILTER = "AND source IN ('kaggle_arshkon', 'scraped')"

con = duckdb.connect()


def run(sql, name=None):
    """Run SQL and optionally save to CSV."""
    df = con.sql(sql).df()
    if name:
        path = os.path.join(OUT_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  -> {path}")
    return df


# ============================================================
# Analysis 1: Detection method decomposition
# ============================================================
print("\n=== Analysis 1: Detection method decomposition ===")

run(f"""
SELECT
    source,
    seniority_final_source,
    count(*) AS n,
    round(100.0 * count(*) / sum(count(*)) OVER (PARTITION BY source), 2) AS pct,
    sum(CASE WHEN seniority_final = 'entry' THEN 1 ELSE 0 END) AS entry_n,
    round(100.0 * sum(CASE WHEN seniority_final = 'entry' THEN 1 ELSE 0 END) / count(*), 2) AS entry_within_pct
FROM '{DATA}'
WHERE {BASE_FILTER} {PERIOD_FILTER}
GROUP BY source, seniority_final_source
ORDER BY source, n DESC
""", "detection_method_decomposition")

run(f"""
SELECT
    source,
    count(*) AS total,
    sum(CASE WHEN seniority_final != 'unknown' THEN 1 ELSE 0 END) AS known_n,
    round(100.0 * sum(CASE WHEN seniority_final != 'unknown' THEN 1 ELSE 0 END) / count(*), 2) AS known_pct,
    sum(CASE WHEN seniority_final_source NOT IN ('native_backfill','unknown') THEN 1 ELSE 0 END) AS explicit_n,
    round(100.0 * sum(CASE WHEN seniority_final_source NOT IN ('native_backfill','unknown') THEN 1 ELSE 0 END) / count(*), 2) AS explicit_pct
FROM '{DATA}'
WHERE {BASE_FILTER} {PERIOD_FILTER}
GROUP BY source ORDER BY source
""", "known_vs_explicit_rates")


# ============================================================
# Analysis 2: Explicit vs implicit entry detection
# ============================================================
print("\n=== Analysis 2: Explicit vs implicit entry ===")

run(f"""
SELECT
    source,
    count(*) AS total_swe,
    sum(CASE WHEN seniority_final = 'entry' THEN 1 ELSE 0 END) AS total_entry,
    round(100.0 * sum(CASE WHEN seniority_final = 'entry' THEN 1 ELSE 0 END) / count(*), 2) AS total_entry_pct,
    sum(CASE WHEN seniority_final = 'entry' AND seniority_final_source IN ('title_keyword','description_explicit') THEN 1 ELSE 0 END) AS explicit_entry,
    round(100.0 * sum(CASE WHEN seniority_final = 'entry' AND seniority_final_source IN ('title_keyword','description_explicit') THEN 1 ELSE 0 END) / count(*), 2) AS explicit_entry_pct,
    sum(CASE WHEN seniority_final = 'entry' AND seniority_final_source = 'native_backfill' THEN 1 ELSE 0 END) AS implicit_entry,
    round(100.0 * sum(CASE WHEN seniority_final = 'entry' AND seniority_final_source = 'native_backfill' THEN 1 ELSE 0 END) / count(*), 2) AS implicit_entry_pct
FROM '{DATA}'
WHERE {BASE_FILTER} {PERIOD_FILTER}
GROUP BY source ORDER BY source
""", "explicit_vs_implicit_entry")


# ============================================================
# Analysis 3: YOE as label-independent proxy
# ============================================================
print("\n=== Analysis 3: YOE label-independent proxy ===")

run(f"""
SELECT
    source,
    count(*) AS n_with_yoe,
    round(avg(yoe_extracted), 2) AS mean_yoe,
    round(median(yoe_extracted), 2) AS median_yoe,
    round(percentile_cont(0.25) WITHIN GROUP (ORDER BY yoe_extracted), 2) AS p25,
    round(percentile_cont(0.75) WITHIN GROUP (ORDER BY yoe_extracted), 2) AS p75,
    sum(CASE WHEN yoe_extracted <= 1 THEN 1 ELSE 0 END) AS yoe_le1,
    round(100.0 * sum(CASE WHEN yoe_extracted <= 1 THEN 1 ELSE 0 END) / count(*), 2) AS pct_le1,
    sum(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END) AS yoe_le2,
    round(100.0 * sum(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END) / count(*), 2) AS pct_le2,
    sum(CASE WHEN yoe_extracted <= 3 THEN 1 ELSE 0 END) AS yoe_le3,
    round(100.0 * sum(CASE WHEN yoe_extracted <= 3 THEN 1 ELSE 0 END) / count(*), 2) AS pct_le3
FROM '{DATA}'
WHERE {BASE_FILTER} {PERIOD_FILTER} AND yoe_extracted IS NOT NULL
GROUP BY source ORDER BY source
""", "yoe_distribution_by_period")

run(f"""
SELECT
    source,
    CAST(yoe_extracted AS INTEGER) AS yoe_int,
    count(*) AS n,
    round(100.0 * count(*) / sum(count(*)) OVER (PARTITION BY source), 2) AS pct
FROM '{DATA}'
WHERE {BASE_FILTER} {PERIOD_FILTER} AND yoe_extracted IS NOT NULL AND yoe_extracted <= 10
GROUP BY source, yoe_int
ORDER BY source, yoe_int
""", "yoe_histogram")


# ============================================================
# Analysis 4: Unknown-pool analysis
# ============================================================
print("\n=== Analysis 4: Unknown-pool analysis ===")

run(f"""
SELECT
    source,
    CASE
        WHEN seniority_final = 'entry' THEN 'entry'
        WHEN seniority_final = 'associate' THEN 'associate'
        WHEN seniority_final IN ('mid-senior','mid_senior') THEN 'mid-senior'
        WHEN seniority_final = 'director' THEN 'director'
        WHEN seniority_final = 'unknown' THEN 'unknown'
        ELSE 'other'
    END AS seniority_group,
    count(*) AS n,
    sum(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS has_yoe,
    round(avg(yoe_extracted), 2) AS mean_yoe,
    round(median(yoe_extracted), 2) AS median_yoe,
    sum(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted <= 2 THEN 1 ELSE 0 END) AS yoe_le2,
    round(100.0 * sum(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted <= 2 THEN 1 ELSE 0 END)
          / NULLIF(sum(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END), 0), 2) AS pct_le2
FROM '{DATA}'
WHERE {BASE_FILTER} {PERIOD_FILTER}
GROUP BY source, seniority_group
ORDER BY source, seniority_group
""", "unknown_pool_yoe")

run(f"""
WITH base_counts AS (
    SELECT
        source,
        count(*) AS total,
        sum(CASE WHEN seniority_final = 'unknown' THEN 1 ELSE 0 END) AS unknown_n,
        sum(CASE WHEN seniority_final != 'unknown' THEN 1 ELSE 0 END) AS known_n,
        sum(CASE WHEN seniority_final = 'entry' THEN 1 ELSE 0 END) AS entry_n,
        round(100.0 * sum(CASE WHEN seniority_final = 'entry' THEN 1 ELSE 0 END)
              / NULLIF(sum(CASE WHEN seniority_final != 'unknown' THEN 1 ELSE 0 END), 0), 2) AS known_entry_rate
    FROM '{DATA}'
    WHERE {BASE_FILTER} {PERIOD_FILTER}
    GROUP BY source
)
SELECT
    source, total, unknown_n, known_n, entry_n, known_entry_rate,
    round(100.0 * entry_n / total, 2) AS entry_pct_unk_zero,
    round(100.0 * (entry_n + unknown_n * known_entry_rate / 100.0) / total, 2) AS entry_pct_unk_same,
    round(100.0 * (entry_n + unknown_n) / total, 2) AS entry_pct_unk_all
FROM base_counts ORDER BY source
""", "unknown_pool_bounds")


# ============================================================
# Analysis 5: Title-level explicit/implicit gap
# ============================================================
print("\n=== Analysis 5: Title-level analysis ===")

run(f"""
SELECT
    title_normalized,
    count(*) AS n,
    sum(CASE WHEN seniority_final_source = 'native_backfill' THEN 1 ELSE 0 END) AS native_only,
    sum(CASE WHEN seniority_final_source IN ('title_keyword','description_explicit') THEN 1 ELSE 0 END) AS explicit_signal,
    round(100.0 * sum(CASE WHEN seniority_final_source IN ('title_keyword','description_explicit') THEN 1 ELSE 0 END) / count(*), 1) AS explicit_pct
FROM '{DATA}'
WHERE {BASE_FILTER} AND source = 'kaggle_arshkon' AND seniority_native = 'entry'
GROUP BY title_normalized
ORDER BY n DESC
LIMIT 25
""", "arshkon_entry_top_titles")

run(f"""
WITH arshkon_entry AS (
    SELECT title_normalized, count(*) AS arshkon_n
    FROM '{DATA}'
    WHERE {BASE_FILTER} AND source = 'kaggle_arshkon' AND seniority_native = 'entry'
    GROUP BY title_normalized HAVING count(*) >= 5
),
scraped AS (
    SELECT
        title_normalized,
        count(*) AS scraped_total,
        sum(CASE WHEN seniority_native = 'entry' THEN 1 ELSE 0 END) AS scraped_entry,
        round(100.0 * sum(CASE WHEN seniority_native = 'entry' THEN 1 ELSE 0 END) / count(*), 1) AS scraped_entry_pct
    FROM '{DATA}'
    WHERE {BASE_FILTER} AND source = 'scraped'
    GROUP BY title_normalized
)
SELECT a.title_normalized, a.arshkon_n, s.scraped_total, s.scraped_entry, s.scraped_entry_pct
FROM arshkon_entry a LEFT JOIN scraped s ON a.title_normalized = s.title_normalized
ORDER BY a.arshkon_n DESC
""", "title_entry_rate_comparison")


# ============================================================
# Analysis 6: Labeling-explicitness hypothesis
# ============================================================
print("\n=== Analysis 6: Labeling explicitness ===")

run(f"""
SELECT
    source,
    count(*) AS total,
    sum(CASE WHEN seniority_imputed != 'unknown' THEN 1 ELSE 0 END) AS imputed_known,
    round(100.0 * sum(CASE WHEN seniority_imputed != 'unknown' THEN 1 ELSE 0 END) / count(*), 2) AS imputed_known_pct,
    sum(CASE WHEN seniority_final_source IN ('title_keyword','description_explicit') THEN 1 ELSE 0 END) AS explicit_signal,
    round(100.0 * sum(CASE WHEN seniority_final_source IN ('title_keyword','description_explicit') THEN 1 ELSE 0 END) / count(*), 2) AS explicit_signal_pct
FROM '{DATA}'
WHERE {BASE_FILTER} {PERIOD_FILTER}
GROUP BY source ORDER BY source
""", "labeling_explicitness")


# ============================================================
# Analysis 7: YOE validation of native labels
# ============================================================
print("\n=== Analysis 7: YOE validation of native labels ===")

run(f"""
SELECT
    source,
    CASE
        WHEN seniority_final = 'entry' AND seniority_final_source IN ('title_keyword','description_explicit') THEN 'explicit_entry'
        WHEN seniority_final = 'entry' AND seniority_final_source = 'native_backfill' THEN 'implicit_entry'
        WHEN seniority_final = 'entry' THEN 'other_entry'
        WHEN seniority_final IN ('mid-senior','mid_senior') THEN 'mid_senior'
        WHEN seniority_final = 'unknown' THEN 'unknown'
        ELSE 'other_known'
    END AS category,
    count(*) AS n,
    sum(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS has_yoe,
    round(avg(yoe_extracted), 2) AS mean_yoe,
    round(median(yoe_extracted), 2) AS median_yoe,
    sum(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted <= 2 THEN 1 ELSE 0 END) AS yoe_le2,
    round(100.0 * sum(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted <= 2 THEN 1 ELSE 0 END)
          / NULLIF(sum(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END), 0), 2) AS pct_le2
FROM '{DATA}'
WHERE {BASE_FILTER} {PERIOD_FILTER}
GROUP BY source, category ORDER BY source, category
""", "yoe_by_entry_type")


# ============================================================
# Summary: All entry-share estimates
# ============================================================
print("\n=== Summary table ===")

estimates = []
for method, sql in [
    ("seniority_final (all rows)", f"""
        SELECT source, round(100.0*sum(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END)/count(*),2) AS pct, count(*) AS n
        FROM '{DATA}' WHERE {BASE_FILTER} {PERIOD_FILTER} GROUP BY source ORDER BY source"""),
    ("seniority_final (known)", f"""
        SELECT source, round(100.0*sum(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END)/count(*),2) AS pct, count(*) AS n
        FROM '{DATA}' WHERE {BASE_FILTER} {PERIOD_FILTER} AND seniority_final!='unknown' GROUP BY source ORDER BY source"""),
    ("seniority_native (non-null)", f"""
        SELECT source, round(100.0*sum(CASE WHEN seniority_native='entry' THEN 1 ELSE 0 END)/count(*),2) AS pct, count(*) AS n
        FROM '{DATA}' WHERE {BASE_FILTER} {PERIOD_FILTER} AND seniority_native IS NOT NULL AND seniority_native!='' GROUP BY source ORDER BY source"""),
    ("seniority_imputed (known)", f"""
        SELECT source, round(100.0*sum(CASE WHEN seniority_imputed='entry' THEN 1 ELSE 0 END)/count(*),2) AS pct, count(*) AS n
        FROM '{DATA}' WHERE {BASE_FILTER} {PERIOD_FILTER} AND seniority_imputed!='unknown' GROUP BY source ORDER BY source"""),
    ("seniority_llm (non-unknown)", f"""
        SELECT source, round(100.0*sum(CASE WHEN seniority_llm='entry' THEN 1 ELSE 0 END)/count(*),2) AS pct, count(*) AS n
        FROM '{DATA}' WHERE {BASE_FILTER} {PERIOD_FILTER} AND seniority_llm IS NOT NULL AND seniority_llm!='unknown' AND seniority_llm!='' GROUP BY source ORDER BY source"""),
    ("YOE<=2 proxy", f"""
        SELECT source, round(100.0*sum(CASE WHEN yoe_extracted<=2 THEN 1 ELSE 0 END)/count(*),2) AS pct, count(*) AS n
        FROM '{DATA}' WHERE {BASE_FILTER} {PERIOD_FILTER} AND yoe_extracted IS NOT NULL GROUP BY source ORDER BY source"""),
    ("YOE<=3 proxy", f"""
        SELECT source, round(100.0*sum(CASE WHEN yoe_extracted<=3 THEN 1 ELSE 0 END)/count(*),2) AS pct, count(*) AS n
        FROM '{DATA}' WHERE {BASE_FILTER} {PERIOD_FILTER} AND yoe_extracted IS NOT NULL GROUP BY source ORDER BY source"""),
    ("explicit entry only", f"""
        SELECT source, round(100.0*sum(CASE WHEN seniority_final='entry' AND seniority_final_source IN ('title_keyword','description_explicit') THEN 1 ELSE 0 END)/count(*),2) AS pct, count(*) AS n
        FROM '{DATA}' WHERE {BASE_FILTER} {PERIOD_FILTER} GROUP BY source ORDER BY source"""),
]:
    df = con.sql(sql).df()
    for _, row in df.iterrows():
        estimates.append({"method": method, "source": row["source"], "entry_pct": row["pct"], "denominator": row["n"]})

summary = pd.DataFrame(estimates)
summary_wide = summary.pivot(index="method", columns="source", values="entry_pct").reset_index()
summary_wide.columns = ["method", "arshkon_2024", "scraped_2026"]
summary_wide["direction"] = summary_wide.apply(
    lambda r: "DECREASE" if r["arshkon_2024"] > r["scraped_2026"]
    else ("INCREASE" if r["arshkon_2024"] < r["scraped_2026"] else "FLAT"), axis=1)
summary_wide["delta_pp"] = round(summary_wide["scraped_2026"] - summary_wide["arshkon_2024"], 2)

# Reorder for readability
order = [
    "seniority_native (non-null)",
    "seniority_final (known)",
    "seniority_final (all rows)",
    "seniority_imputed (known)",
    "seniority_llm (non-unknown)",
    "explicit entry only",
    "YOE<=2 proxy",
    "YOE<=3 proxy",
]
summary_wide["sort"] = summary_wide["method"].map({m: i for i, m in enumerate(order)})
summary_wide = summary_wide.sort_values("sort").drop(columns="sort")

summary_wide.to_csv(os.path.join(OUT_DIR, "entry_share_summary.csv"), index=False)
print(f"  -> {OUT_DIR}/entry_share_summary.csv")
print(summary_wide.to_string(index=False))

print("\nDone. All tables saved to", OUT_DIR)
