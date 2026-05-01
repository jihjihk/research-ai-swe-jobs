"""
T18 - Cross-occupation boundary analysis
Step 1: Parallel trends + DiD for SWE, SWE-adjacent, Control groups.

Metrics per period x group:
  - Entry share (combined best-available) and YOE<=2 share
  - AI keyword prevalence (agentic, multi-agent, AI agent, ai-powered, llm, rag, copilot, claude)
  - Description length (raw + LLM-text-only subset)
  - Org scope language (cross-functional, stakeholders, roadmap, strategy, etc.)
  - Tech mention count (direct LIKE counting of top techs, avoiding broken c++/c# regex)

Filters: linkedin, english, date_flag=ok. NO is_swe filter — span all three groups.
Sensitivities reported: aggregator exclusion; SWE tier excluding title_lookup_llm.
"""
import os
import duckdb
import pandas as pd
import numpy as np

PARQUET = "data/unified.parquet"
OUT_DIR = "exploration/tables/T18"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Period buckets ---
# We collapse to two periods: "2024" (all 2024) and "2026" (all 2026).
# For some metrics we also break out the four native periods.

BASE_FILTER = """
source_platform = 'linkedin'
AND is_english = true
AND date_flag = 'ok'
"""

GROUP_EXPR = """
CASE
  WHEN is_swe THEN 'SWE'
  WHEN is_swe_adjacent THEN 'SWE_adjacent'
  WHEN is_control THEN 'control'
  ELSE NULL
END
"""

PERIOD2_EXPR = """
CASE
  WHEN period IN ('2024-01','2024-04') THEN '2024'
  WHEN period IN ('2026-03','2026-04') THEN '2026'
  ELSE NULL
END
"""

SENIORITY_BEST = """
CASE
  WHEN llm_classification_coverage = 'labeled'         THEN seniority_llm
  WHEN llm_classification_coverage = 'rule_sufficient' THEN seniority_final
  ELSE NULL
END
"""

# AI keywords: use agentic (~95% precision) plus other high-precision tokens
AI_CASE = """
CASE WHEN
  regexp_matches(lower(coalesce(description_core_llm, description_core, description)), 'agentic')
  OR regexp_matches(lower(coalesce(description_core_llm, description_core, description)), 'multi[- ]agent')
  OR regexp_matches(lower(coalesce(description_core_llm, description_core, description)), '\\bai[- ]agent')
  OR regexp_matches(lower(coalesce(description_core_llm, description_core, description)), '\\bllm\\b')
  OR regexp_matches(lower(coalesce(description_core_llm, description_core, description)), '\\brag\\b')
  OR regexp_matches(lower(coalesce(description_core_llm, description_core, description)), '\\bcopilot\\b')
  OR regexp_matches(lower(coalesce(description_core_llm, description_core, description)), '\\bclaude\\b')
  OR regexp_matches(lower(coalesce(description_core_llm, description_core, description)), 'generative ai')
  OR regexp_matches(lower(coalesce(description_core_llm, description_core, description)), 'large language model')
  OR regexp_matches(lower(coalesce(description_core_llm, description_core, description)), 'ai[- ]powered')
THEN 1 ELSE 0 END
"""

# Strict "agentic-only" variant for sensitivity
AGENTIC_CASE = """
CASE WHEN
  regexp_matches(lower(coalesce(description_core_llm, description_core, description)), 'agentic')
THEN 1 ELSE 0 END
"""

# Org scope language (validated-ish; cross-functional etc.)
SCOPE_CASE = """
CASE WHEN
  regexp_matches(lower(coalesce(description_core_llm, description_core, description)), 'cross[- ]functional')
  OR regexp_matches(lower(coalesce(description_core_llm, description_core, description)), 'stakeholders?')
  OR regexp_matches(lower(coalesce(description_core_llm, description_core, description)), '\\broadmap\\b')
  OR regexp_matches(lower(coalesce(description_core_llm, description_core, description)), 'org[- ]wide')
  OR regexp_matches(lower(coalesce(description_core_llm, description_core, description)), 'company[- ]wide')
  OR regexp_matches(lower(coalesce(description_core_llm, description_core, description)), 'strategic (direction|initiative|vision)')
THEN 1 ELSE 0 END
"""

# Tech mention count using safe LIKE-based detection of 15 common techs
# (avoids the broken c_cpp/csharp regex)
TECHS = [
    ("python", r"\bpython\b"),
    ("java", r"\bjava\b"),
    ("javascript", r"javascript|\bjs\b"),
    ("typescript", r"typescript|\bts\b"),
    ("golang", r"\bgolang\b|\bgo programming\b"),
    ("rust", r"\brust\b"),
    ("cpp", r"\bc\+\+|\bcpp\b"),
    ("csharp", r"\bc#|\bc\\#|\\bc sharp"),
    ("react", r"\breact\b"),
    ("kubernetes", r"kubernetes|\bk8s\b"),
    ("docker", r"\bdocker\b"),
    ("aws", r"\baws\b"),
    ("sql", r"\bsql\b"),
    ("tensorflow", r"tensorflow"),
    ("pytorch", r"pytorch"),
]

tech_exprs = " + ".join(
    [
        f"(CASE WHEN regexp_matches(lower(coalesce(description_core_llm, description_core, description)), '{pat}') THEN 1 ELSE 0 END)"
        for _, pat in TECHS
    ]
)


def query_trends(agg_exclude=False, exclude_title_lookup=False):
    extra = ""
    if agg_exclude:
        extra += " AND (is_aggregator IS FALSE OR is_aggregator IS NULL)"
    if exclude_title_lookup:
        # swe_classification_tier != 'title_lookup_llm' for all three groups
        extra += " AND (swe_classification_tier IS NULL OR swe_classification_tier != 'title_lookup_llm')"

    q = f"""
    WITH base AS (
      SELECT
        uid,
        {GROUP_EXPR} AS occ_group,
        {PERIOD2_EXPR} AS period2,
        period AS period_raw,
        {SENIORITY_BEST} AS seniority_best,
        yoe_extracted,
        COALESCE(LENGTH(description_core_llm), LENGTH(description_core), LENGTH(description)) AS desc_len,
        description_core_llm IS NOT NULL AND LENGTH(description_core_llm) > 0 AS has_llm_text,
        {AI_CASE} AS ai_any,
        {AGENTIC_CASE} AS agentic_flag,
        {SCOPE_CASE} AS scope_flag,
        {tech_exprs} AS tech_count
      FROM '{PARQUET}'
      WHERE {BASE_FILTER}
        AND (is_swe OR is_swe_adjacent OR is_control)
        {extra}
    )
    SELECT
      occ_group,
      period2,
      COUNT(*) AS n,
      AVG(CASE WHEN seniority_best = 'entry' THEN 1.0
               WHEN seniority_best IS NULL THEN NULL ELSE 0.0 END) AS entry_share_best,
      AVG(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted <= 2 THEN 1.0
               WHEN yoe_extracted IS NULL THEN NULL ELSE 0.0 END) AS yoe_le2_share,
      AVG(ai_any::DOUBLE) AS ai_rate,
      AVG(agentic_flag::DOUBLE) AS agentic_rate,
      AVG(scope_flag::DOUBLE) AS scope_rate,
      median(desc_len) AS median_len,
      AVG(desc_len::DOUBLE) AS mean_len,
      AVG(tech_count::DOUBLE) AS mean_tech_count,
      AVG(has_llm_text::INT::DOUBLE) AS llm_text_share,
      median(CASE WHEN has_llm_text THEN desc_len END) AS median_len_llm_text,
      AVG(CASE WHEN has_llm_text THEN ai_any::DOUBLE END) AS ai_rate_llm_text
    FROM base
    WHERE occ_group IS NOT NULL AND period2 IS NOT NULL
    GROUP BY occ_group, period2
    ORDER BY occ_group, period2
    """
    return con.execute(q).df()


con = duckdb.connect()

print("[T18.1] Computing parallel trends (baseline)...")
df = query_trends()
df.to_csv(os.path.join(OUT_DIR, "trends_baseline.csv"), index=False)
print(df.to_string())

print("\n[T18.1] Aggregator-excluded trends...")
df_na = query_trends(agg_exclude=True)
df_na.to_csv(os.path.join(OUT_DIR, "trends_no_agg.csv"), index=False)
print(df_na.to_string())

print("\n[T18.1] title_lookup_llm excluded trends...")
df_tt = query_trends(exclude_title_lookup=True)
df_tt.to_csv(os.path.join(OUT_DIR, "trends_no_title_lookup_llm.csv"), index=False)
print(df_tt.to_string())


# --- DiD table (baseline) ---
def compute_did(df):
    """Compute SWE 2024->2026 delta vs control delta vs adjacent delta for each metric."""
    metrics = ["entry_share_best", "yoe_le2_share", "ai_rate", "agentic_rate",
               "scope_rate", "median_len", "mean_tech_count", "median_len_llm_text",
               "ai_rate_llm_text"]
    out = []
    groups = ["SWE", "SWE_adjacent", "control"]
    for m in metrics:
        row = {"metric": m}
        for g in groups:
            try:
                v24 = df[(df.occ_group == g) & (df.period2 == "2024")][m].iloc[0]
                v26 = df[(df.occ_group == g) & (df.period2 == "2026")][m].iloc[0]
                row[f"{g}_2024"] = v24
                row[f"{g}_2026"] = v26
                row[f"{g}_delta"] = v26 - v24
                row[f"{g}_rel_delta"] = (v26 - v24) / v24 if v24 and not pd.isna(v24) and v24 != 0 else np.nan
            except (IndexError, TypeError):
                row[f"{g}_2024"] = row[f"{g}_2026"] = row[f"{g}_delta"] = row[f"{g}_rel_delta"] = np.nan
        # DiD: SWE delta - control delta
        row["DiD_SWE_minus_control"] = row["SWE_delta"] - row["control_delta"]
        row["DiD_SWE_minus_adjacent"] = row["SWE_delta"] - row["SWE_adjacent_delta"]
        out.append(row)
    return pd.DataFrame(out)


did = compute_did(df)
did.to_csv(os.path.join(OUT_DIR, "did_baseline.csv"), index=False)
print("\n[T18.1] DiD table (baseline):")
print(did[["metric","SWE_delta","SWE_adjacent_delta","control_delta","DiD_SWE_minus_control","DiD_SWE_minus_adjacent"]].to_string())

did_na = compute_did(df_na)
did_na.to_csv(os.path.join(OUT_DIR, "did_no_agg.csv"), index=False)

did_tt = compute_did(df_tt)
did_tt.to_csv(os.path.join(OUT_DIR, "did_no_title_lookup_llm.csv"), index=False)

print("\n[T18.1] DiD table (no title_lookup_llm):")
print(did_tt[["metric","SWE_delta","SWE_adjacent_delta","control_delta","DiD_SWE_minus_control","DiD_SWE_minus_adjacent"]].to_string())

print("\nDone. Outputs in", OUT_DIR)
