"""T18 — Cross-occupation boundary analysis.

Compare SWE vs SWE-adjacent vs control groups on key metrics and
compute difference-in-differences (2024 -> 2026) to test whether the
AI/tech explosion is SWE-specific or field-wide.

Approach:
- Use DuckDB directly over data/unified.parquet for metric computation.
- For text-dependent metrics, filter to llm_extraction_coverage='labeled'.
- For binary keyword presence (AI/tech mention) we use raw description
  (boilerplate-insensitive) so SWE, adjacent, and control are scored
  under identical conditions. This matches Gate 1 decision 4.
- AI broad set = 24-term union (same as shared tech matrix AI terms).
- For SWE, cross-check with the shared tech matrix to sanity-check.
- Confidence intervals: Wilson for proportions; normal approx for
  DiD differences bootstrap-based.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
PARQUET = ROOT / "data/unified.parquet"
TECH_MATRIX = ROOT / "exploration/artifacts/shared/swe_tech_matrix.parquet"
OUT_TABLES = ROOT / "exploration/tables/T18"
OUT_FIGS = ROOT / "exploration/figures/T18"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

# Default filter applied throughout
DEFAULT_FILTER = (
    "source_platform='linkedin' AND is_english=true AND date_flag='ok'"
)

# ----------------------------------------------------------------------
# Regex patterns — run against RAW description for group-uniform scoring.
# These mirror the shared tech matrix AI-adjacent terms. We use a
# case-insensitive search directly; we rely on word-ish boundaries via
# `\b` except where the token has special chars.
# ----------------------------------------------------------------------
# 24-term broad AI union (matches T14's construction plus common synonyms)
AI_BROAD_TERMS = {
    "ai": r"\bai\b",
    "artificial_intelligence": r"\bartificial intelligence\b",
    "machine_learning": r"\b(?:machine learning|\bml\b)",
    "deep_learning": r"\bdeep learning\b",
    "nlp": r"\bnlp\b",
    "llm": r"\bllms?\b",
    "generative_ai": r"\bgenerative ai\b",
    "gen_ai": r"\bgen\s?ai\b",
    "rag": r"\brag\b",
    "langchain": r"\blangchain\b",
    "langgraph": r"\blanggraph\b",
    "copilot": r"\bcopilot\b",
    "claude": r"\bclaude\b",
    "anthropic": r"\banthropic\b",
    "openai": r"\bopenai\b",
    "gpt": r"\bgpt(?:-?4|-?3)?\b",
    "chatgpt": r"\bchatgpt\b",
    "gemini": r"\bgemini\b",
    "agents": r"\bagents?\b",
    "agentic": r"\bagentic\b",
    "vector_db": r"\bvector (?:db|database)\b",
    "mcp": r"\bmcp\b",
    "fine_tuning": r"\bfine[- ]?tun",
    "prompt_engineering": r"\bprompt engineering\b",
}

# Narrow AI pattern from T05 calibration — single LIKE '%ai%'-style match
# implemented as the boundary-aware \bai\b pattern.
AI_NARROW_PATTERN = r"\bai\b"

# Per-tool headline breakouts
PER_TOOL = {
    "claude_tool": r"\bclaude\b",
    "copilot": r"\bcopilot\b",
    "langchain": r"\blangchain\b",
    "agents_framework": r"\b(?:agent|agents|agentic)\b",
}

# Scope patterns — Gate 2 corrections: ONLY end-to-end and cross-functional
SCOPE_PATTERNS = {
    "end_to_end": r"\bend[- ]?to[- ]?end\b",
    "cross_functional": r"\bcross[- ]?functional\b",
}

# 123-tech taxonomy: reuse the same pattern list from prep_04_tech_matrix
# Compile here to run against raw descriptions for adj/ctrl groups.
import importlib.util

spec = importlib.util.spec_from_file_location(
    "prep_04_tech_matrix",
    ROOT / "exploration/scripts/prep_04_tech_matrix.py",
)
prep_mod = importlib.util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(prep_mod)  # type: ignore
TECH_PATTERNS: list[tuple[str, str]] = prep_mod.TECH_PATTERNS


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def compute_group_metrics(con, group: str, period_label: str, period_filter: str) -> dict:
    """Compute per-group metrics for a period using DuckDB SQL."""
    group_filter = {
        "swe": "is_swe = true",
        "adj": "is_swe_adjacent = true",
        "ctrl": "is_control = true",
    }[group]

    base = f"""
    FROM read_parquet('{PARQUET}')
    WHERE {DEFAULT_FILTER}
      AND {group_filter}
      AND {period_filter}
    """
    # Total n
    n_total = con.execute(f"SELECT COUNT(*) {base}").fetchone()[0]
    if n_total == 0:
        return {
            "group": group,
            "period": period_label,
            "n_total": 0,
        }

    # seniority_final entry share (of known) and YOE<=2 share (of all)
    sen = con.execute(
        f"""
        SELECT
          COUNT(*) FILTER (WHERE seniority_final = 'entry') as entry,
          COUNT(*) FILTER (WHERE seniority_final != 'unknown' AND seniority_final IS NOT NULL) as known,
          COUNT(*) as total,
          COUNT(*) FILTER (WHERE yoe_extracted <= 2) as yoe_le2,
          COUNT(*) FILTER (WHERE yoe_extracted IS NOT NULL) as yoe_known
        {base}
        """
    ).fetchone()

    # Description length
    length = con.execute(
        f"SELECT AVG(description_length), MEDIAN(description_length) {base}"
    ).fetchone()

    # Binary keyword metrics against raw description — group-uniform
    # Build SQL regex_matches over raw description for AI broad, narrow, per-tool, scope
    binary_cols = []
    col_names: list[str] = []

    for k, pat in AI_BROAD_TERMS.items():
        binary_cols.append(
            f"SUM(CASE WHEN regexp_matches(lower(description), '{pat}') THEN 1 ELSE 0 END) AS ai_broad_{k}"
        )
        col_names.append(f"ai_broad_{k}")

    binary_cols.append(
        f"SUM(CASE WHEN regexp_matches(lower(description), '{AI_NARROW_PATTERN}') THEN 1 ELSE 0 END) AS ai_narrow"
    )
    col_names.append("ai_narrow")

    for k, pat in PER_TOOL.items():
        binary_cols.append(
            f"SUM(CASE WHEN regexp_matches(lower(description), '{pat}') THEN 1 ELSE 0 END) AS tool_{k}"
        )
        col_names.append(f"tool_{k}")

    for k, pat in SCOPE_PATTERNS.items():
        binary_cols.append(
            f"SUM(CASE WHEN regexp_matches(lower(description), '{pat}') THEN 1 ELSE 0 END) AS scope_{k}"
        )
        col_names.append(f"scope_{k}")

    sql = "SELECT " + ", ".join(binary_cols) + f" {base}"
    row = con.execute(sql).fetchone()
    binary_counts = dict(zip(col_names, row))

    # AI broad union = OR across broad terms — compute via a single SQL
    union_parts = " OR ".join(
        f"regexp_matches(lower(description), '{pat}')" for pat in AI_BROAD_TERMS.values()
    )
    ai_broad_any = con.execute(
        f"SELECT SUM(CASE WHEN ({union_parts}) THEN 1 ELSE 0 END) {base}"
    ).fetchone()[0]

    # Scope any (end-to-end OR cross-functional)
    scope_union = " OR ".join(
        f"regexp_matches(lower(description), '{pat}')" for pat in SCOPE_PATTERNS.values()
    )
    scope_any = con.execute(
        f"SELECT SUM(CASE WHEN ({scope_union}) THEN 1 ELSE 0 END) {base}"
    ).fetchone()[0]

    # Tech mention count — sum of binary hits for 123-tech taxonomy (raw desc,
    # approximated using DuckDB regex per tech). The prep_04 patterns expect
    # cleaned text with (?:^|\s) anchors; we convert to \b for raw text.
    # Use literal r"\b" so regex boundaries survive into the SQL.
    tech_per_row_parts = []
    for tech_name, pat in TECH_PATTERNS:
        raw_pat = pat.replace("(?:^|\\s)", r"\b").replace("(?:$|\\s)", r"\b")
        # Escape single quotes for SQL string literal
        raw_pat_sql = raw_pat.replace("'", "''")
        tech_per_row_parts.append(
            f"CAST(regexp_matches(lower(description), '{raw_pat_sql}') AS INTEGER)"
        )
    tech_sum_expr = " + ".join(tech_per_row_parts)
    tech_stats_sql = f"""
        SELECT AVG(tc) as mean_tc, MEDIAN(tc) as med_tc,
               AVG(CASE WHEN tc > 0 THEN 1.0 ELSE 0.0 END) as any_tech
        FROM (SELECT ({tech_sum_expr}) AS tc {base})
    """
    tech_stats = con.execute(tech_stats_sql).fetchone()

    return {
        "group": group,
        "period": period_label,
        "n_total": n_total,
        "n_entry_seniority_final": sen[0],
        "n_known_seniority_final": sen[1],
        "entry_share_of_known": (sen[0] / sen[1]) if sen[1] else None,
        "entry_share_of_all": (sen[0] / sen[2]) if sen[2] else None,
        "n_yoe_le2": sen[3],
        "n_yoe_known": sen[4],
        "yoe_le2_share_of_yoe_known": (sen[3] / sen[4]) if sen[4] else None,
        "yoe_le2_share_of_all": (sen[3] / sen[2]) if sen[2] else None,
        "desc_length_mean": float(length[0]) if length[0] else None,
        "desc_length_median": float(length[1]) if length[1] else None,
        "ai_broad_any_count": ai_broad_any or 0,
        "ai_broad_any_share": (ai_broad_any or 0) / n_total,
        "ai_narrow_count": binary_counts["ai_narrow"] or 0,
        "ai_narrow_share": (binary_counts["ai_narrow"] or 0) / n_total,
        "scope_any_count": scope_any or 0,
        "scope_any_share": (scope_any or 0) / n_total,
        "tool_claude_share": (binary_counts["tool_claude_tool"] or 0) / n_total,
        "tool_copilot_share": (binary_counts["tool_copilot"] or 0) / n_total,
        "tool_langchain_share": (binary_counts["tool_langchain"] or 0) / n_total,
        "tool_agents_share": (binary_counts["tool_agents_framework"] or 0) / n_total,
        "scope_end_to_end_share": (binary_counts["scope_end_to_end"] or 0) / n_total,
        "scope_cross_functional_share": (binary_counts["scope_cross_functional"] or 0) / n_total,
        "tech_count_mean": float(tech_stats[0]) if tech_stats[0] is not None else None,
        "tech_count_median": float(tech_stats[1]) if tech_stats[1] is not None else None,
        "any_tech_share": float(tech_stats[2]) if tech_stats[2] is not None else None,
    }


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=6")
    con.execute("PRAGMA memory_limit='12GB'")

    # Three periods: 2024 (pooled arshkon+asaniczka), 2026 (scraped). Also
    # provide arshkon-only 2024 for sensitivity.
    periods = {
        "2024_pooled": "(source IN ('kaggle_arshkon','kaggle_asaniczka'))",
        "2024_arshkon": "(source = 'kaggle_arshkon')",
        "2026_scraped": "(source = 'scraped')",
    }
    groups = ["swe", "adj", "ctrl"]

    print("Computing group × period metrics...")
    results = []
    for period_label, period_filter in periods.items():
        for group in groups:
            t0 = time.time()
            m = compute_group_metrics(con, group, period_label, period_filter)
            print(
                f"  {group:4s} × {period_label:13s} n={m.get('n_total', 0):8,} "
                f"ai_broad={m.get('ai_broad_any_share', 0):.4f} "
                f"elapsed={time.time()-t0:.1f}s"
            )
            results.append(m)

    df = pd.DataFrame(results)
    df.to_csv(OUT_TABLES / "group_period_metrics.csv", index=False)
    print(f"Wrote {OUT_TABLES / 'group_period_metrics.csv'}")

    # Sensitivity: exclude aggregators
    print("\nSensitivity (a) — aggregator exclusion...")
    results_no_agg = []
    for period_label, period_filter in periods.items():
        for group in groups:
            group_filter = {
                "swe": "is_swe = true",
                "adj": "is_swe_adjacent = true",
                "ctrl": "is_control = true",
            }[group]
            # Reuse logic but with additional filter — we only re-compute AI broad,
            # AI narrow, desc_length, and entry share since those are the headline
            # comparators.
            base = f"""
            FROM read_parquet('{PARQUET}')
            WHERE {DEFAULT_FILTER}
              AND {group_filter}
              AND {period_filter}
              AND (is_aggregator = false OR is_aggregator IS NULL)
            """
            n_total = con.execute(f"SELECT COUNT(*) {base}").fetchone()[0]
            if n_total == 0:
                continue
            union_parts = " OR ".join(
                f"regexp_matches(lower(description), '{pat}')"
                for pat in AI_BROAD_TERMS.values()
            )
            ai_broad_any = con.execute(
                f"SELECT SUM(CASE WHEN ({union_parts}) THEN 1 ELSE 0 END) {base}"
            ).fetchone()[0] or 0
            ai_narrow = con.execute(
                f"SELECT SUM(CASE WHEN regexp_matches(lower(description), '{AI_NARROW_PATTERN}') THEN 1 ELSE 0 END) {base}"
            ).fetchone()[0] or 0
            length = con.execute(
                f"SELECT AVG(description_length) {base}"
            ).fetchone()[0]
            results_no_agg.append(
                {
                    "group": group,
                    "period": period_label,
                    "n_total": n_total,
                    "ai_broad_share": ai_broad_any / n_total,
                    "ai_narrow_share": ai_narrow / n_total,
                    "desc_length_mean": float(length) if length else None,
                }
            )
    pd.DataFrame(results_no_agg).to_csv(
        OUT_TABLES / "group_period_metrics_no_agg.csv", index=False
    )

    # Sensitivity (g): SWE tier — exclude title_lookup_llm SWE tier
    print("Sensitivity (g) — SWE classification tier...")
    results_tier = []
    for period_label, period_filter in periods.items():
        base = f"""
        FROM read_parquet('{PARQUET}')
        WHERE {DEFAULT_FILTER}
          AND is_swe = true
          AND swe_classification_tier != 'title_lookup_llm'
          AND {period_filter}
        """
        n_total = con.execute(f"SELECT COUNT(*) {base}").fetchone()[0]
        if n_total == 0:
            continue
        union_parts = " OR ".join(
            f"regexp_matches(lower(description), '{pat}')"
            for pat in AI_BROAD_TERMS.values()
        )
        ai_broad_any = con.execute(
            f"SELECT SUM(CASE WHEN ({union_parts}) THEN 1 ELSE 0 END) {base}"
        ).fetchone()[0] or 0
        results_tier.append(
            {
                "group": "swe_strict_tier",
                "period": period_label,
                "n_total": n_total,
                "ai_broad_share": ai_broad_any / n_total,
            }
        )
    pd.DataFrame(results_tier).to_csv(
        OUT_TABLES / "swe_tier_sensitivity.csv", index=False
    )

    print("\nDone T18 metrics step.")


if __name__ == "__main__":
    main()
