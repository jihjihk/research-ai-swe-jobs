"""T04 — SWE classification audit.

Produces:
 - Tier breakdown by source x period for SWE rows
 - Dual-flag invariant check
 - Sample 50 borderline SWE postings (swe_confidence 0.3-0.7 or tier title_lookup_llm)
 - Sample 50 borderline non-SWE postings (title contains engineer/developer/software
   but is_swe=false)
 - Profile of is_swe_adjacent and is_control title distributions
 - Boundary case analysis for roles like ML Engineer, Data Engineer, DevOps

All outputs go to exploration/tables/T04/.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from collections import Counter

import duckdb

random.seed(42)

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data" / "unified.parquet"
OUT = ROOT / "exploration" / "tables" / "T04"
OUT.mkdir(parents=True, exist_ok=True)

BASE_FILTER = (
    "is_english = TRUE AND date_flag = 'ok' AND source_platform = 'linkedin'"
)


def write_csv(path, header, rows):
    with Path(path).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def clean_ascii(s):
    if s is None:
        return ""
    return "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in s)


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    # 1. SWE tier breakdown
    rows = con.execute(f"""
        SELECT source, period, swe_classification_tier, COUNT(*) AS n
        FROM '{DATA}'
        WHERE {BASE_FILTER} AND is_swe = TRUE
        GROUP BY 1,2,3 ORDER BY 1,2,3
    """).fetchall()
    write_csv(OUT / "01_swe_tier_breakdown.csv",
              ["source", "period", "swe_classification_tier", "n"], rows)

    # 1b. With percentages
    rows = con.execute(f"""
        WITH t AS (
          SELECT source, period, swe_classification_tier, COUNT(*) AS n
          FROM '{DATA}' WHERE {BASE_FILTER} AND is_swe = TRUE
          GROUP BY 1,2,3
        ),
        tot AS (SELECT source, period, SUM(n) AS total FROM t GROUP BY 1,2)
        SELECT t.source, t.period, t.swe_classification_tier, t.n,
               ROUND(100.0*t.n/tot.total, 2) AS pct
        FROM t JOIN tot USING (source, period)
        ORDER BY 1,2,3
    """).fetchall()
    write_csv(OUT / "01b_swe_tier_breakdown_pct.csv",
              ["source", "period", "swe_classification_tier", "n", "pct"], rows)

    # 2. Dual-flag invariant
    r = con.execute(f"""
        SELECT
          SUM(CASE WHEN (CAST(is_swe AS INT) + CAST(COALESCE(is_swe_adjacent,FALSE) AS INT) + CAST(COALESCE(is_control,FALSE) AS INT)) > 1 THEN 1 ELSE 0 END) AS n_dual,
          SUM(CASE WHEN (CAST(is_swe AS INT) + CAST(COALESCE(is_swe_adjacent,FALSE) AS INT) + CAST(COALESCE(is_control,FALSE) AS INT)) = 0 THEN 1 ELSE 0 END) AS n_none,
          SUM(CASE WHEN is_swe THEN 1 ELSE 0 END) AS n_swe,
          SUM(CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) AS n_adj,
          SUM(CASE WHEN is_control THEN 1 ELSE 0 END) AS n_ctrl,
          COUNT(*) AS n_total
        FROM '{DATA}' WHERE {BASE_FILTER}
    """).fetchall()
    write_csv(OUT / "02_dual_flag_check.csv",
              ["n_dual_flag", "n_none", "n_swe", "n_adjacent", "n_control", "n_total"],
              r)

    # 3. swe_confidence distribution by tier
    rows = con.execute(f"""
        SELECT swe_classification_tier,
               ROUND(AVG(swe_confidence),3) AS mean_conf,
               MIN(swe_confidence) AS min_conf,
               MAX(swe_confidence) AS max_conf,
               COUNT(*) AS n,
               SUM(CASE WHEN swe_confidence BETWEEN 0.3 AND 0.7 THEN 1 ELSE 0 END) AS n_in_03_07
        FROM '{DATA}' WHERE {BASE_FILTER} AND is_swe=TRUE
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    write_csv(OUT / "03_swe_conf_by_tier.csv",
              ["swe_classification_tier", "mean_conf", "min_conf", "max_conf",
               "n", "n_in_03_07"], rows)

    # 4. Sample 50 borderline SWE rows (confidence 0.3-0.7 OR tier=title_lookup_llm)
    candidates = con.execute(f"""
        SELECT source, period, title, title_normalized, description,
               swe_classification_tier, swe_confidence, swe_classification_llm
        FROM '{DATA}'
        WHERE {BASE_FILTER} AND is_swe=TRUE
          AND ((swe_confidence BETWEEN 0.3 AND 0.7) OR swe_classification_tier='title_lookup_llm')
    """).fetchall()
    random.shuffle(candidates)
    sample = candidates[:50]
    out_rows = []
    for src, per, title, title_norm, desc, tier, conf, llm_cls in sample:
        out_rows.append([
            src, per, clean_ascii(title), clean_ascii(title_norm),
            clean_ascii(desc)[:250].replace("\n"," ") if desc else "",
            tier, conf, llm_cls,
        ])
    write_csv(OUT / "04_borderline_swe_sample.csv",
              ["source", "period", "title", "title_normalized",
               "desc_prefix_250", "tier", "swe_confidence", "swe_classification_llm"],
              out_rows)

    # 5. Sample 50 borderline non-SWE rows with engineer/developer/software in title
    candidates = con.execute(f"""
        SELECT source, period, title, title_normalized, description,
               is_swe_adjacent, is_control, swe_confidence
        FROM '{DATA}'
        WHERE {BASE_FILTER} AND is_swe=FALSE
          AND (lower(title) LIKE '%engineer%' OR lower(title) LIKE '%developer%'
               OR lower(title) LIKE '%software%')
    """).fetchall()
    random.shuffle(candidates)
    sample = candidates[:50]
    out_rows = []
    for src, per, title, title_norm, desc, adj, ctrl, conf in sample:
        out_rows.append([
            src, per, clean_ascii(title), clean_ascii(title_norm),
            clean_ascii(desc)[:250].replace("\n"," ") if desc else "",
            adj, ctrl, conf,
        ])
    write_csv(OUT / "05_borderline_nonswe_sample.csv",
              ["source", "period", "title", "title_normalized", "desc_prefix_250",
               "is_swe_adjacent", "is_control", "swe_confidence"],
              out_rows)

    # 6. Profile is_swe_adjacent top titles
    rows = con.execute(f"""
        SELECT title_normalized, COUNT(*) AS n
        FROM '{DATA}'
        WHERE {BASE_FILTER} AND is_swe_adjacent=TRUE
          AND title_normalized IS NOT NULL
        GROUP BY 1 ORDER BY n DESC LIMIT 50
    """).fetchall()
    # Sanitize titles that may have bad bytes
    rows_clean = [(clean_ascii(r[0]), r[1]) for r in rows]
    write_csv(OUT / "06_adjacent_top_titles.csv",
              ["title_normalized", "n"], rows_clean)

    # 7. Profile is_control top titles
    rows = con.execute(f"""
        SELECT title_normalized, COUNT(*) AS n
        FROM '{DATA}'
        WHERE {BASE_FILTER} AND is_control=TRUE
          AND title_normalized IS NOT NULL
        GROUP BY 1 ORDER BY n DESC LIMIT 50
    """).fetchall()
    rows_clean = [(clean_ascii(r[0]), r[1]) for r in rows]
    write_csv(OUT / "07_control_top_titles.csv",
              ["title_normalized", "n"], rows_clean)

    # 8. Boundary cases: ML Engineer, Data Engineer, DevOps Engineer
    boundary_queries = [
        ("ml_engineer",
         "lower(title) LIKE '%machine learning engineer%' OR lower(title) LIKE '%ml engineer%'"),
        ("data_engineer",
         "lower(title) LIKE '%data engineer%' AND lower(title) NOT LIKE '%data engineering manager%'"),
        ("devops_engineer",
         "lower(title) LIKE '%devops engineer%' OR lower(title) LIKE '%devops%'"),
        ("platform_engineer",
         "lower(title) LIKE '%platform engineer%'"),
        ("security_engineer",
         "lower(title) LIKE '%security engineer%'"),
        ("qa_engineer",
         "lower(title) LIKE '%qa engineer%' OR lower(title) LIKE '%quality assurance%'"),
    ]
    rows_out = []
    for tag, pred in boundary_queries:
        r = con.execute(f"""
            SELECT '{tag}' AS category, source, period,
                   SUM(CASE WHEN is_swe THEN 1 ELSE 0 END) AS n_swe,
                   SUM(CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) AS n_adj,
                   SUM(CASE WHEN is_control THEN 1 ELSE 0 END) AS n_ctrl,
                   SUM(CASE WHEN NOT is_swe AND NOT COALESCE(is_swe_adjacent,FALSE)
                                 AND NOT COALESCE(is_control,FALSE) THEN 1 ELSE 0 END) AS n_other,
                   COUNT(*) AS n_total
            FROM '{DATA}'
            WHERE {BASE_FILTER} AND ({pred})
            GROUP BY 1,2,3
            ORDER BY 2,3
        """).fetchall()
        rows_out.extend(r)
    write_csv(OUT / "08_boundary_classification.csv",
              ["category", "source", "period", "n_swe", "n_adjacent",
               "n_control", "n_other", "n_total"], rows_out)

    # 9. swe_classification_llm distribution on routed rows (cross-check)
    rows = con.execute(f"""
        SELECT source, period, swe_classification_llm, COUNT(*) AS n
        FROM '{DATA}'
        WHERE {BASE_FILTER} AND is_swe=TRUE
          AND swe_classification_llm IS NOT NULL
        GROUP BY 1,2,3 ORDER BY 1,2,3
    """).fetchall()
    write_csv(OUT / "09_swe_classification_llm_on_is_swe.csv",
              ["source", "period", "swe_classification_llm", "n"], rows)

    # 10. How often does is_swe=TRUE but LLM says NOT_SWE? (potential FP indicator)
    rows = con.execute(f"""
        SELECT source, period,
               SUM(CASE WHEN is_swe AND swe_classification_llm='NOT_SWE' THEN 1 ELSE 0 END) AS is_swe_but_llm_notswe,
               SUM(CASE WHEN is_swe AND swe_classification_llm='SWE_ADJACENT' THEN 1 ELSE 0 END) AS is_swe_but_llm_adj,
               SUM(CASE WHEN is_swe AND swe_classification_llm='SWE' THEN 1 ELSE 0 END) AS is_swe_and_llm_swe,
               SUM(CASE WHEN is_swe AND swe_classification_llm IS NULL THEN 1 ELSE 0 END) AS is_swe_llm_null,
               SUM(CASE WHEN is_swe THEN 1 ELSE 0 END) AS n_is_swe
        FROM '{DATA}'
        WHERE {BASE_FILTER}
        GROUP BY 1,2 ORDER BY 1,2
    """).fetchall()
    write_csv(OUT / "10_is_swe_vs_llm.csv",
              ["source", "period",
               "is_swe_but_llm_notswe", "is_swe_but_llm_adj",
               "is_swe_and_llm_swe", "is_swe_llm_null", "n_is_swe"], rows)

    print("Wrote tables to", OUT)


if __name__ == "__main__":
    main()
