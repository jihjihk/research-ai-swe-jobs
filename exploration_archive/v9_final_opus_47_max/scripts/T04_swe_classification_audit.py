"""T04. SWE classification audit.

Audits is_swe classification quality after preprocessing fixes. Produces:
  1. SWE rows by swe_classification_tier breakdown
  2. Sample 50 borderline SWE postings (swe_confidence 0.3-0.7 or tier title_lookup_llm)
  3. Sample 50 borderline non-SWE (titles with 'engineer'/'developer'/'software' but is_swe=False)
  4. Profile is_swe_adjacent and is_control rows: titles, counts
  5. False-positive / false-negative estimates
  6. Dual-flag violation check
  7. Boundary cases: ML Engineer, Data Engineer, DevOps Engineer
  8. Cross-check is_swe vs swe_classification_llm
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

PARQUET = "data/unified.parquet"
OUT_TABLES = Path("exploration/tables/T04")
OUT_TABLES.mkdir(parents=True, exist_ok=True)

DEFAULT_FILTER = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'"


def q(sql: str) -> pd.DataFrame:
    return duckdb.sql(sql).df()


def save(df: pd.DataFrame, name: str) -> None:
    path = OUT_TABLES / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  -> {path}  ({len(df)} rows)")


def step1_tier_breakdown() -> None:
    print("\n[Step 1] SWE classification tier breakdown (default filter).")
    df = q(f"""
      SELECT swe_classification_tier,
             count(*) total,
             sum(CAST(is_swe AS INT)) swe,
             sum(CAST(is_swe_adjacent AS INT)) adjacent,
             sum(CAST(is_control AS INT)) control,
             avg(swe_confidence) avg_confidence
      FROM '{PARQUET}'
      WHERE {DEFAULT_FILTER}
      GROUP BY swe_classification_tier
      ORDER BY total DESC
    """)
    save(df, "tier_breakdown")

    # Confidence distribution histogram-like summary for is_swe
    df2 = q(f"""
      SELECT
        CASE
          WHEN swe_confidence IS NULL THEN 'null'
          WHEN swe_confidence < 0.3 THEN '<0.3'
          WHEN swe_confidence < 0.5 THEN '0.3-0.5'
          WHEN swe_confidence < 0.7 THEN '0.5-0.7'
          WHEN swe_confidence < 0.85 THEN '0.7-0.85'
          WHEN swe_confidence < 1.0 THEN '0.85-1.0'
          ELSE '1.0'
        END AS conf_bin,
        count(*) total,
        sum(CAST(is_swe AS INT)) swe,
        sum(CAST(is_swe_adjacent AS INT)) adjacent,
        sum(CAST(is_control AS INT)) control
      FROM '{PARQUET}'
      WHERE {DEFAULT_FILTER}
      GROUP BY conf_bin
      ORDER BY conf_bin
    """)
    save(df2, "swe_confidence_bins")

    # Tier × source breakdown
    df3 = q(f"""
      SELECT source, swe_classification_tier,
             count(*) total,
             sum(CAST(is_swe AS INT)) swe
      FROM '{PARQUET}'
      WHERE {DEFAULT_FILTER}
      GROUP BY source, swe_classification_tier
      ORDER BY source, total DESC
    """)
    save(df3, "tier_breakdown_by_source")


def step2_borderline_swe_sample() -> None:
    """Sample 50 borderline SWE postings: is_swe=True with swe_confidence 0.3-0.7 or tier=title_lookup_llm."""
    print("\n[Step 2] 50 borderline SWE postings sample.")
    df = q(rf"""
      SELECT * FROM (
        SELECT uid, source, title, swe_confidence, swe_classification_tier,
               swe_classification_llm,
               substr(description, 1, 300) desc_preview
        FROM '{PARQUET}'
        WHERE {DEFAULT_FILTER}
          AND is_swe = true
          AND (
            (swe_confidence >= 0.3 AND swe_confidence < 0.7)
            OR swe_classification_tier = 'title_lookup_llm'
          )
      ) USING SAMPLE 50
    """)
    save(df, "borderline_swe_sample")


def step3_borderline_non_swe_sample() -> None:
    """Sample 50 non-SWE rows whose titles contain 'engineer', 'developer', or 'software' keywords — potential false negatives."""
    print("\n[Step 3] 50 borderline non-SWE postings sample.")
    df = q(rf"""
      SELECT * FROM (
        SELECT uid, source, title, is_swe, is_swe_adjacent, is_control,
               swe_confidence, swe_classification_tier, swe_classification_llm,
               substr(description, 1, 300) desc_preview
        FROM '{PARQUET}'
        WHERE {DEFAULT_FILTER}
          AND is_swe = false
          AND regexp_matches(lower(title), 'engineer|developer|software|programmer')
      ) USING SAMPLE 50
    """)
    save(df, "borderline_non_swe_sample")


def step4_adjacent_and_control_profile() -> None:
    """Profile is_swe_adjacent and is_control: top titles, counts."""
    print("\n[Step 4] SWE_adjacent and control profiles.")
    # Top adjacent titles
    df_adj = q(f"""
      SELECT title_normalized, count(*) n
      FROM '{PARQUET}'
      WHERE {DEFAULT_FILTER} AND is_swe_adjacent = true
      GROUP BY title_normalized
      ORDER BY n DESC
      LIMIT 50
    """)
    save(df_adj, "adjacent_top_titles")

    # Top control titles
    df_ctrl = q(f"""
      SELECT title_normalized, count(*) n
      FROM '{PARQUET}'
      WHERE {DEFAULT_FILTER} AND is_control = true
      GROUP BY title_normalized
      ORDER BY n DESC
      LIMIT 50
    """)
    save(df_ctrl, "control_top_titles")

    # Adjacent by source
    df_adj_src = q(f"""
      SELECT source, count(*) n
      FROM '{PARQUET}'
      WHERE {DEFAULT_FILTER} AND is_swe_adjacent = true
      GROUP BY source
    """)
    save(df_adj_src, "adjacent_by_source")

    df_ctrl_src = q(f"""
      SELECT source, count(*) n
      FROM '{PARQUET}'
      WHERE {DEFAULT_FILTER} AND is_control = true
      GROUP BY source
    """)
    save(df_ctrl_src, "control_by_source")


def step5_fp_fn_estimate() -> None:
    """Estimate false-positive and false-negative rates using swe_classification_llm cross-check."""
    print("\n[Step 5] FP/FN estimate via LLM cross-check.")
    # Cross-tab is_swe x swe_classification_llm for LLM-labeled rows
    df = q(f"""
      SELECT is_swe, is_swe_adjacent, is_control, swe_classification_llm, count(*) n
      FROM '{PARQUET}'
      WHERE {DEFAULT_FILTER}
        AND llm_classification_coverage = 'labeled'
        AND swe_classification_llm IS NOT NULL
      GROUP BY is_swe, is_swe_adjacent, is_control, swe_classification_llm
      ORDER BY is_swe DESC, is_swe_adjacent DESC, is_control DESC, swe_classification_llm
    """)
    save(df, "fp_fn_crosstab_llm")

    # Summary: rule SWE vs LLM SWE
    df2 = q(f"""
      SELECT
        sum(CASE WHEN is_swe=true AND swe_classification_llm='SWE' THEN 1 ELSE 0 END) rule_swe_llm_swe,
        sum(CASE WHEN is_swe=true AND swe_classification_llm='SWE_ADJACENT' THEN 1 ELSE 0 END) rule_swe_llm_adj,
        sum(CASE WHEN is_swe=true AND swe_classification_llm='NOT_SWE' THEN 1 ELSE 0 END) rule_swe_llm_not,
        sum(CASE WHEN is_swe=false AND swe_classification_llm='SWE' THEN 1 ELSE 0 END) rule_notswe_llm_swe,
        sum(CASE WHEN is_swe=false AND is_swe_adjacent=true AND swe_classification_llm='SWE' THEN 1 ELSE 0 END) rule_adj_llm_swe,
        sum(CASE WHEN is_swe=false AND is_control=true AND swe_classification_llm='SWE' THEN 1 ELSE 0 END) rule_ctrl_llm_swe,
        sum(CASE WHEN is_swe=true THEN 1 ELSE 0 END) rule_swe_total,
        sum(CASE WHEN swe_classification_llm='SWE' THEN 1 ELSE 0 END) llm_swe_total
      FROM '{PARQUET}'
      WHERE {DEFAULT_FILTER}
        AND llm_classification_coverage = 'labeled'
        AND swe_classification_llm IS NOT NULL
    """)
    save(df2, "fp_fn_summary")


def step6_dual_flag_violations() -> None:
    print("\n[Step 6] Dual-flag violations.")
    df = q(f"""
      SELECT
        sum(CASE WHEN is_swe AND is_swe_adjacent THEN 1 ELSE 0 END) swe_and_adj,
        sum(CASE WHEN is_swe AND is_control THEN 1 ELSE 0 END) swe_and_ctrl,
        sum(CASE WHEN is_swe_adjacent AND is_control THEN 1 ELSE 0 END) adj_and_ctrl,
        sum(CASE WHEN (CAST(is_swe AS INT) + CAST(is_swe_adjacent AS INT) + CAST(is_control AS INT)) > 1 THEN 1 ELSE 0 END) any_dual_flag
      FROM '{PARQUET}'
    """)
    save(df, "dual_flag_violations")


def step7_boundary_cases() -> None:
    """ML Engineer, Data Engineer, DevOps Engineer — boundary cases that may be misclassified differently."""
    print("\n[Step 7] Boundary cases.")
    df = q(rf"""
      SELECT
        CASE
          WHEN regexp_matches(lower(title), '\bmachine learning\b|\bml engineer\b|\bml ops\b|\bmlops\b') THEN 'ML Engineer'
          WHEN regexp_matches(lower(title), '\bdata engineer\b|\bdata engineering\b') THEN 'Data Engineer'
          WHEN regexp_matches(lower(title), '\bdevops\b|\bdev ops\b|\bsite reliability\b|\bsre\b|\bplatform engineer\b') THEN 'DevOps/SRE/Platform'
          WHEN regexp_matches(lower(title), '\bai engineer\b|\bartificial intelligence engineer\b|\bgenai\b|\bllm engineer\b') THEN 'AI Engineer'
          WHEN regexp_matches(lower(title), '\bsecurity engineer\b|\bcyber security engineer\b') THEN 'Security Engineer'
          WHEN regexp_matches(lower(title), '\bqa engineer\b|\bquality engineer\b|\btest engineer\b|\bsdet\b') THEN 'QA/Test Engineer'
          WHEN regexp_matches(lower(title), '\bsolutions? engineer\b|\bsystems engineer\b|\binfrastructure engineer\b|\bcloud engineer\b') THEN 'Solutions/Systems/Infra'
          ELSE 'Other'
        END AS boundary_class,
        source, period,
        sum(CAST(is_swe AS INT)) swe,
        sum(CAST(is_swe_adjacent AS INT)) adjacent,
        sum(CAST(is_control AS INT)) control,
        count(*) total
      FROM '{PARQUET}'
      WHERE {DEFAULT_FILTER}
      GROUP BY boundary_class, source, period
      ORDER BY boundary_class, source, period
    """)
    save(df, "boundary_cases")

    # Within-title-group distribution: does ML Engineer get classified consistently across periods/sources?
    df2 = q(rf"""
      SELECT
        CASE
          WHEN regexp_matches(lower(title), '\bmachine learning\b|\bml engineer\b|\bml ops\b|\bmlops\b') THEN 'ML Engineer'
          WHEN regexp_matches(lower(title), '\bdata engineer\b|\bdata engineering\b') THEN 'Data Engineer'
          WHEN regexp_matches(lower(title), '\bdevops\b|\bdev ops\b|\bsite reliability\b|\bsre\b|\bplatform engineer\b') THEN 'DevOps/SRE/Platform'
          WHEN regexp_matches(lower(title), '\bai engineer\b|\bartificial intelligence engineer\b|\bgenai\b|\bllm engineer\b') THEN 'AI Engineer'
          ELSE 'Other'
        END AS boundary_class,
        source,
        count(*) total,
        avg(CAST(is_swe AS INT)) swe_share,
        avg(CAST(is_swe_adjacent AS INT)) adj_share,
        avg(CAST(is_control AS INT)) ctrl_share
      FROM '{PARQUET}'
      WHERE {DEFAULT_FILTER}
      GROUP BY boundary_class, source
      ORDER BY boundary_class, source
    """)
    save(df2, "boundary_cases_shares")


def step8_temporal_consistency() -> None:
    """Check if SWE classification shares shifted between periods due to preprocessing changes."""
    print("\n[Step 8] Temporal consistency of SWE classification.")
    df = q(f"""
      SELECT source, period,
             count(*) total,
             sum(CAST(is_swe AS INT)) swe,
             sum(CAST(is_swe_adjacent AS INT)) adj,
             sum(CAST(is_control AS INT)) ctrl,
             sum(CAST(is_swe AS INT)) * 1.0 / count(*) swe_share,
             sum(CAST(is_swe_adjacent AS INT)) * 1.0 / count(*) adj_share,
             sum(CAST(is_control AS INT)) * 1.0 / count(*) ctrl_share
      FROM '{PARQUET}'
      WHERE {DEFAULT_FILTER}
      GROUP BY source, period
      ORDER BY source, period
    """)
    save(df, "temporal_consistency")


def main():
    print("=" * 60)
    print("T04. SWE classification audit")
    print("=" * 60)

    step1_tier_breakdown()
    step2_borderline_swe_sample()
    step3_borderline_non_swe_sample()
    step4_adjacent_and_control_profile()
    step5_fp_fn_estimate()
    step6_dual_flag_violations()
    step7_boundary_cases()
    step8_temporal_consistency()

    print("\n[T04] complete.")


if __name__ == "__main__":
    main()
