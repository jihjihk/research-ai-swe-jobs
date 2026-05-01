"""
T07 Part A: Feasibility table for cross-period comparisons.

Computes group sizes, MDE for binary and continuous outcomes at 80% power, alpha=0.05,
for each (comparison × seniority definition) combination.

Output: exploration/tables/T07/feasibility_table.csv

Columns: analysis_type | comparison | seniority_def | n_group1 | n_group2 |
         MDE_binary | MDE_continuous | verdict
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data" / "unified.parquet"
TABLES = ROOT / "exploration" / "tables" / "T07"
TABLES.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Constants for MDE computation
# -----------------------------------------------------------------------------
# Two-sided test, alpha=0.05 (z = 1.96), power=0.80 (z = 0.84).
# For a two-sample comparison of proportions (balanced), MDE ~= (z_{alpha/2}+z_{beta}) * sqrt(2*p(1-p)/n_per_group)
# but we often have unbalanced groups. Use the general unequal-n formula:
#   MDE_binary = (z_a + z_b) * sqrt( p*(1-p) * (1/n1 + 1/n2) )
# where p is a baseline proportion (we report at p=0.5 for conservative upper bound,
# and at p=0.25 for a more realistic scenario like entry share).
#
# For a continuous outcome with standardized effect size (Cohen's d):
#   MDE_d = (z_a + z_b) * sqrt( 1/n1 + 1/n2 )
Z_ALPHA = 1.959963984540054  # two-sided 0.05
Z_BETA = 0.8416212335729143  # one-sided 0.80

# Standardized baseline proportions for binary MDE reporting.
# We report MDE_binary at p=0.5 (worst case) — reported in feasibility table.
# A supplemental p=0.25 will also be computed for sensitivity.
P_BASE_MAIN = 0.5
P_BASE_ALT = 0.25


def mde_binary(n1: int, n2: int, p_base: float = P_BASE_MAIN) -> float:
    """Minimum detectable effect (absolute) for two-sample binary at balanced-p."""
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    return (Z_ALPHA + Z_BETA) * math.sqrt(p_base * (1 - p_base) * (1 / n1 + 1 / n2))


def mde_continuous(n1: int, n2: int) -> float:
    """Minimum detectable standardized effect (Cohen's d)."""
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    return (Z_ALPHA + Z_BETA) * math.sqrt(1 / n1 + 1 / n2)


def verdict_from_mde(
    mde_bin: float, n1: int, n2: int, *, binary_thresh_strong: float = 0.03, binary_thresh_ok: float = 0.07
) -> str:
    """Classify feasibility verdict given MDE and sample sizes.

    The RQ1 decline hypothesis is typically a shift of >= 5 pp (e.g., entry share 10% -> 5%).
    - Strong: MDE_binary <= 3 pp (detects modest shifts easily)
    - OK: MDE_binary <= 7 pp (detects meaningful shifts)
    - Underpowered: MDE_binary > 7 pp or n_group < 30
    """
    if n1 < 30 or n2 < 30:
        return "underpowered_n<30"
    if math.isnan(mde_bin):
        return "underpowered_n<30"
    if mde_bin <= binary_thresh_strong:
        return "strong"
    if mde_bin <= binary_thresh_ok:
        return "ok"
    return "underpowered"


# -----------------------------------------------------------------------------
# SQL building blocks
# -----------------------------------------------------------------------------
BASE_WHERE = (
    "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe"
)

SENIOR_TITLE_REGEX = r"\b(senior|sr\.?|staff|principal|lead|architect|distinguished)\b"

# Definitions: each maps to a SQL predicate (applied IN ADDITION to BASE_WHERE + is_swe).
# J3, J4, S4 additionally require llm_classification_coverage='labeled'.
DEFINITIONS = {
    "J1": {
        "side": "junior",
        "pred": "seniority_final = 'entry'",
        "requires_llm_labeled": False,
        "note": "entry only",
    },
    "J2": {
        "side": "junior",
        "pred": "seniority_final IN ('entry','associate')",
        "requires_llm_labeled": False,
        "note": "entry + associate",
    },
    "J3": {
        "side": "junior",
        "pred": "llm_classification_coverage = 'labeled' AND yoe_min_years_llm <= 2",
        "requires_llm_labeled": True,
        "note": "LLM YOE <= 2 (primary)",
    },
    "J4": {
        "side": "junior",
        "pred": "llm_classification_coverage = 'labeled' AND yoe_min_years_llm <= 3",
        "requires_llm_labeled": True,
        "note": "LLM YOE <= 3",
    },
    "S1": {
        "side": "senior",
        "pred": "seniority_final IN ('mid-senior','director')",
        "requires_llm_labeled": False,
        "note": "mid-senior + director",
    },
    "S2": {
        "side": "senior",
        "pred": "seniority_final = 'director'",
        "requires_llm_labeled": False,
        "note": "director only",
    },
    "S3": {
        "side": "senior",
        # Use a raw-title regex via regexp_matches.
        "pred": f"regexp_matches(lower(title), '{SENIOR_TITLE_REGEX}')",
        "requires_llm_labeled": False,
        "note": "title keyword senior",
    },
    "S4": {
        "side": "senior",
        "pred": "llm_classification_coverage = 'labeled' AND yoe_min_years_llm >= 5",
        "requires_llm_labeled": True,
        "note": "LLM YOE >= 5 (primary)",
    },
    "ALL": {
        "side": "all",
        "pred": "1=1",
        "requires_llm_labeled": False,
        "note": "all SWE",
    },
}


def count_group(defn: str, source_filter: str, agg_excluded: bool = False) -> int:
    """Count rows for a definition under a source filter (e.g. source='kaggle_arshkon').

    source_filter can be something like:
      "source = 'kaggle_arshkon'"
      "source IN ('kaggle_arshkon','kaggle_asaniczka')"
      "source = 'scraped'"
    """
    pred = DEFINITIONS[defn]["pred"]
    agg_clause = " AND is_aggregator = false " if agg_excluded else ""
    sql = f"""
        SELECT count(*) FROM '{DATA}'
        WHERE {BASE_WHERE}
          AND {source_filter}
          AND ({pred})
          {agg_clause}
    """
    n = duckdb.sql(sql).fetchone()[0]
    return int(n)


# -----------------------------------------------------------------------------
# Comparisons we evaluate
# -----------------------------------------------------------------------------
COMPARISONS = [
    {
        "name": "arshkon_vs_scraped",
        "group1_filter": "source = 'kaggle_arshkon'",
        "group2_filter": "source = 'scraped'",
        "group1_label": "arshkon (2024)",
        "group2_label": "scraped (2026)",
    },
    {
        "name": "pooled_2024_vs_scraped",
        "group1_filter": "source IN ('kaggle_arshkon','kaggle_asaniczka')",
        "group2_filter": "source = 'scraped'",
        "group1_label": "arshkon+asaniczka (2024)",
        "group2_label": "scraped (2026)",
    },
    {
        "name": "arshkon_senior_vs_scraped_senior",
        "group1_filter": "source = 'kaggle_arshkon'",
        "group2_filter": "source = 'scraped'",
        "group1_label": "arshkon senior (2024)",
        "group2_label": "scraped senior (2026)",
        # senior-side only — restrict DEFINITIONS used
        "senior_only": True,
    },
]


def build_feasibility_rows(agg_excluded: bool = False) -> list[dict]:
    rows: list[dict] = []
    for comp in COMPARISONS:
        is_senior_only = comp.get("senior_only", False)
        # Which defs apply to this comparison?
        if is_senior_only:
            defs = [k for k, v in DEFINITIONS.items() if v["side"] == "senior"]
        else:
            defs = list(DEFINITIONS.keys())

        for defn in defs:
            side = DEFINITIONS[defn]["side"]
            # Decide analysis_type label
            if defn == "ALL":
                seniority_def = "N/A"
                analysis_type = "all_SWE"
            else:
                seniority_def = defn
                analysis_type = f"{side}_share" if side in ("junior", "senior") else side

            n1 = count_group(defn, comp["group1_filter"], agg_excluded=agg_excluded)
            n2 = count_group(defn, comp["group2_filter"], agg_excluded=agg_excluded)
            mde_b = mde_binary(n1, n2, P_BASE_MAIN)
            mde_b_alt = mde_binary(n1, n2, P_BASE_ALT)
            mde_c = mde_continuous(n1, n2)
            vdct = verdict_from_mde(mde_b, n1, n2)
            rows.append(
                {
                    "analysis_type": analysis_type,
                    "comparison": comp["name"],
                    "seniority_def": seniority_def,
                    "def_note": DEFINITIONS[defn]["note"],
                    "n_group1": n1,
                    "n_group2": n2,
                    "group1_label": comp["group1_label"],
                    "group2_label": comp["group2_label"],
                    "MDE_binary_p50": round(mde_b, 4),
                    "MDE_binary_p25": round(mde_b_alt, 4),
                    "MDE_continuous": round(mde_c, 4),
                    "verdict": vdct,
                    "spec": "agg_excluded" if agg_excluded else "agg_included",
                }
            )
    return rows


def main() -> None:
    all_rows: list[dict] = []
    all_rows.extend(build_feasibility_rows(agg_excluded=False))
    all_rows.extend(build_feasibility_rows(agg_excluded=True))

    df = pd.DataFrame(all_rows)
    out = TABLES / "feasibility_table.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")

    # Print a short summary grouped by (comparison, seniority_def) in agg_included spec
    print()
    print("=" * 120)
    print("Feasibility summary — aggregator INCLUDED")
    print("=" * 120)
    main_spec = df[df["spec"] == "agg_included"].copy()
    main_spec_print = main_spec[
        [
            "analysis_type",
            "comparison",
            "seniority_def",
            "n_group1",
            "n_group2",
            "MDE_binary_p50",
            "MDE_binary_p25",
            "MDE_continuous",
            "verdict",
        ]
    ]
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 200)
    print(main_spec_print.to_string(index=False))

    print()
    print("=" * 120)
    print("Feasibility summary — aggregator EXCLUDED (sensitivity)")
    print("=" * 120)
    excl_spec = df[df["spec"] == "agg_excluded"].copy()
    excl_spec_print = excl_spec[
        [
            "analysis_type",
            "comparison",
            "seniority_def",
            "n_group1",
            "n_group2",
            "MDE_binary_p50",
            "MDE_binary_p25",
            "MDE_continuous",
            "verdict",
        ]
    ]
    print(excl_spec_print.to_string(index=False))


if __name__ == "__main__":
    main()
