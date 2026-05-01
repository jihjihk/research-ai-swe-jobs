"""T04 — SWE classification audit.

  1. SWE rows by swe_classification_tier breakdown.
  2. Sample 50 borderline SWE (swe_confidence in [0.3,0.7] OR tier=title_lookup_llm):
     print title + 200 chars description.
  3. Sample 50 borderline non-SWE: titles with engineer/developer/software but is_swe = False.
  4. Profile is_swe_adjacent + is_control rows.
  5. Estimate false-positive / false-negative rates.
  6. Verify no dual-flag violations: (is_swe + is_swe_adjacent + is_control) > 1 should be 0.
  7. Boundary cases: ML/Data/DevOps Engineer classification consistency across periods.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data/unified.parquet"
TAB_DIR = ROOT / "exploration/tables/T04"
FIG_DIR = ROOT / "exploration/figures/T04"
TAB_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

FILTER = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'"


def q(sql: str) -> pd.DataFrame:
    return duckdb.sql(sql).df()


# ---- Step 1: tier breakdown ----

def step_1_tier_breakdown():
    print("\n=== STEP 1: SWE by swe_classification_tier ===")
    df = q(f"""
        SELECT source, swe_classification_tier, COUNT(*) n,
               AVG(swe_confidence) avg_confidence
        FROM '{DATA}'
        WHERE {FILTER} AND is_swe = true
        GROUP BY source, swe_classification_tier
        ORDER BY source, n DESC
    """)
    df.to_csv(TAB_DIR / "step1_tier_breakdown_by_source.csv", index=False)
    print(df.to_string())

    # By period
    df2 = q(f"""
        SELECT period, swe_classification_tier, COUNT(*) n,
               AVG(swe_confidence) avg_confidence
        FROM '{DATA}'
        WHERE {FILTER} AND is_swe = true
        GROUP BY period, swe_classification_tier
        ORDER BY period, n DESC
    """)
    df2.to_csv(TAB_DIR / "step1_tier_breakdown_by_period.csv", index=False)
    print("\nBy period:")
    print(df2.to_string())

    # All rows: distribution of is_swe vs is_swe_adjacent vs is_control vs none
    df3 = q(f"""
        SELECT source,
               SUM(CASE WHEN is_swe THEN 1 ELSE 0 END) n_swe,
               SUM(CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) n_adjacent,
               SUM(CASE WHEN is_control THEN 1 ELSE 0 END) n_control,
               SUM(CASE WHEN NOT is_swe AND NOT is_swe_adjacent AND NOT is_control THEN 1 ELSE 0 END) n_other,
               COUNT(*) n_total
        FROM '{DATA}'
        WHERE {FILTER}
        GROUP BY source
    """)
    df3.to_csv(TAB_DIR / "step1_occupation_breakdown_by_source.csv", index=False)
    print("\nOccupation breakdown:")
    print(df3.to_string())
    return df, df3


# ---- Step 2: borderline SWE sample ----

def step_2_borderline_swe():
    print("\n=== STEP 2: Borderline SWE sample ===")
    df = q(f"""
        WITH pool AS (
          SELECT uid, source, title, swe_confidence, swe_classification_tier,
                 seniority_final, swe_classification_llm,
                 substr(coalesce(description, ''), 1, 200) desc_preview
          FROM '{DATA}'
          WHERE {FILTER}
            AND is_swe = true
            AND (
              (swe_confidence >= 0.3 AND swe_confidence < 0.7)
              OR swe_classification_tier = 'title_lookup_llm'
            )
        )
        SELECT * FROM pool USING SAMPLE 50
    """)
    df.to_csv(TAB_DIR / "step2_borderline_swe_sample.csv", index=False)
    print(f"Sampled {len(df)} borderline SWE rows.")
    # Auto-assess: if the title contains classic non-SWE markers, flag as likely FP.
    non_swe_markers = re.compile(
        r"\b(sales|marketing|manager|analyst|account|business analyst|support|hr|"
        r"customer|project manager|program manager|designer|writer|recruiter|"
        r"finance|accountant|nurse|teacher|driver|operator|technician|mechanic)\b",
        re.IGNORECASE,
    )
    df["title_suggests_non_swe"] = df["title"].fillna("").str.contains(non_swe_markers, regex=True)
    fp_count = int(df["title_suggests_non_swe"].sum())
    print(f"Auto-flagged potential FP (title suggests non-SWE): {fp_count} / {len(df)}")
    return df


# ---- Step 3: borderline non-SWE sample ----

def step_3_borderline_non_swe():
    print("\n=== STEP 3: Borderline non-SWE sample (titles w/ engineer/developer/software but is_swe=false) ===")
    df = q(f"""
        WITH pool AS (
          SELECT uid, source, title, swe_confidence, swe_classification_tier,
                 is_swe, is_swe_adjacent, is_control,
                 substr(coalesce(description, ''), 1, 200) desc_preview
          FROM '{DATA}'
          WHERE {FILTER}
            AND is_swe = false
            AND regexp_matches(lower(title), '\\b(engineer|developer|software)\\b')
        )
        SELECT * FROM pool USING SAMPLE 50
    """)
    df.to_csv(TAB_DIR / "step3_borderline_non_swe_sample.csv", index=False)
    print(f"Sampled {len(df)} borderline non-SWE rows.")
    # Auto-assess: if title has strong SWE pattern, flag as likely FN
    swe_markers = re.compile(
        r"\b(software engineer|software developer|full[- ]?stack|backend|frontend|"
        r"front[- ]?end|back[- ]?end|devops engineer|sre|site reliability|"
        r"application developer|web developer)\b",
        re.IGNORECASE,
    )
    df["title_suggests_swe"] = df["title"].fillna("").str.contains(swe_markers, regex=True)
    fn_count = int(df["title_suggests_swe"].sum())
    print(f"Auto-flagged potential FN (title strongly suggests SWE): {fn_count} / {len(df)}")
    return df


# ---- Step 4: adjacent + control profile ----

def step_4_adjacent_control():
    print("\n=== STEP 4: is_swe_adjacent and is_control profile ===")
    # Top titles in adjacent
    adj_titles = q(f"""
        SELECT source, title, COUNT(*) n
        FROM '{DATA}'
        WHERE {FILTER} AND is_swe_adjacent = true
        GROUP BY source, title
        ORDER BY n DESC
        LIMIT 30
    """)
    adj_titles.to_csv(TAB_DIR / "step4_adjacent_top_titles.csv", index=False)
    print("Top is_swe_adjacent titles (source-split):")
    print(adj_titles.to_string())

    # Top titles in control
    ctrl_titles = q(f"""
        SELECT source, title, COUNT(*) n
        FROM '{DATA}'
        WHERE {FILTER} AND is_control = true
        GROUP BY source, title
        ORDER BY n DESC
        LIMIT 30
    """)
    ctrl_titles.to_csv(TAB_DIR / "step4_control_top_titles.csv", index=False)
    print("\nTop is_control titles:")
    print(ctrl_titles.to_string())

    # Adjacent tier distribution
    adj_tiers = q(f"""
        SELECT source, swe_classification_tier, COUNT(*) n
        FROM '{DATA}'
        WHERE {FILTER} AND is_swe_adjacent = true
        GROUP BY source, swe_classification_tier
        ORDER BY source, n DESC
    """)
    adj_tiers.to_csv(TAB_DIR / "step4_adjacent_tiers.csv", index=False)
    print("\nAdjacent tiers:")
    print(adj_tiers.to_string())

    # Control occupation sample
    ctrl_sample = q(f"""
        WITH pool AS (
          SELECT uid, source, title, swe_classification_tier,
                 substr(coalesce(description, ''), 1, 200) desc_preview
          FROM '{DATA}'
          WHERE {FILTER} AND is_control = true
        )
        SELECT * FROM pool USING SAMPLE 30
    """)
    ctrl_sample.to_csv(TAB_DIR / "step4_control_sample.csv", index=False)
    print(f"\nControl sample ({len(ctrl_sample)} rows):")
    print(ctrl_sample[["title"]].to_string())
    return adj_titles, ctrl_titles


# ---- Step 6: dual-flag violations ----

def step_6_dual_flag():
    print("\n=== STEP 6: Dual-flag violations ===")
    df = q(f"""
        SELECT source,
               SUM(CASE WHEN (CAST(is_swe AS INTEGER) + CAST(is_swe_adjacent AS INTEGER) + CAST(is_control AS INTEGER)) > 1 THEN 1 ELSE 0 END) n_violations,
               COUNT(*) n_total
        FROM '{DATA}'
        WHERE {FILTER}
        GROUP BY source
    """)
    df.to_csv(TAB_DIR / "step6_dual_flag.csv", index=False)
    print(df.to_string())
    return df


# ---- Step 7: boundary cases (ML Engineer, Data Engineer, DevOps Engineer) ----

def step_7_boundary_cases():
    print("\n=== STEP 7: Boundary cases ===")
    # Grab counts and is_swe share by source, for three boundary titles
    # DuckDB uses RE2 — no lookahead. Use character class approach for Data Engineer
    # to avoid matching 'Data Engineering'.
    patterns = [
        ("ML Engineer", r"(?i)\b(ml\s+engineer|machine\s+learning\s+engineer)\b"),
        # "data engineer" word-boundary at the end suffices because 'engineer'
        # before 'ing' would not have a word-boundary between 'r' and 'i' — wait
        # actually there IS always a word boundary at end of 'engineer' since
        # next char could be space or word. Use literal ' ' / end-of-string.
        ("Data Engineer", r"(?i)\bdata\s+engineer\b"),
        ("DevOps Engineer", r"(?i)\b(dev\s*ops|devops)\s+engineer\b"),
        ("AI Engineer", r"(?i)\bai\s+engineer\b"),
        ("Data Scientist", r"(?i)\bdata\s+scientist\b"),
        ("ML Scientist", r"(?i)\b(ml|machine\s+learning)\s+scientist\b"),
    ]
    rows = []
    for label, pat in patterns:
        sql = f"""
            SELECT source,
                   SUM(CASE WHEN regexp_matches(title, '{pat}') THEN 1 ELSE 0 END) n_match,
                   SUM(CASE WHEN regexp_matches(title, '{pat}') AND is_swe THEN 1 ELSE 0 END) n_swe,
                   SUM(CASE WHEN regexp_matches(title, '{pat}') AND is_swe_adjacent THEN 1 ELSE 0 END) n_adjacent,
                   SUM(CASE WHEN regexp_matches(title, '{pat}') AND is_control THEN 1 ELSE 0 END) n_control,
                   SUM(CASE WHEN regexp_matches(title, '{pat}') AND NOT is_swe AND NOT is_swe_adjacent AND NOT is_control THEN 1 ELSE 0 END) n_other
            FROM '{DATA}'
            WHERE {FILTER}
            GROUP BY source
        """
        df = duckdb.sql(sql).df()
        for _, r in df.iterrows():
            rows.append({
                "title_pattern": label,
                "source": r["source"],
                "n_match": int(r["n_match"] or 0),
                "n_swe": int(r["n_swe"] or 0),
                "n_adjacent": int(r["n_adjacent"] or 0),
                "n_control": int(r["n_control"] or 0),
                "n_other": int(r["n_other"] or 0),
                "swe_rate": (r["n_swe"] / r["n_match"]) if r["n_match"] else 0,
                "adjacent_rate": (r["n_adjacent"] / r["n_match"]) if r["n_match"] else 0,
            })
    out = pd.DataFrame(rows)
    out.to_csv(TAB_DIR / "step7_boundary_classification.csv", index=False)
    print(out.to_string())
    return out


def main():
    step_1_tier_breakdown()
    step_2_borderline_swe()
    step_3_borderline_non_swe()
    step_4_adjacent_control()
    step_6_dual_flag()
    step_7_boundary_cases()
    print("\nT04 artifacts under exploration/tables/T04/.")


if __name__ == "__main__":
    main()
