"""T07 — Feasibility / power analysis for cross-period SWE comparisons.

Primary output: exploration/tables/T07/feasibility.csv

Computes:
  - Group sizes (entry, mid-senior, all SWE, etc.) by source, filtered to
    LinkedIn + English + date_flag ok + is_swe=true.
  - Minimum detectable effect sizes (MDE) at 80% power, alpha=0.05 for binary
    (two-proportion z test) and continuous (two-sample t test) outcomes on the
    four key comparisons:
      * entry arshkon vs scraped
      * senior (mid-senior) arshkon vs scraped
      * all SWE arshkon vs scraped
      * pooled 2024 (arshkon+asaniczka) vs scraped
  - Metro-level feasibility (metros qualifying at >=50 and >=100 per period).
  - Company overlap panel (>=3 SWE in BOTH arshkon and scraped).

Assumptions:
  - Binary MDE: computed for a baseline proportion p1 = 0.5 (most conservative,
    largest required n) AND for p1 = 0.2 (closer to realistic entry-share /
    AI-prevalence magnitudes) so consumers can see both. Primary verdict uses
    the p1 = 0.5 MDE as a worst-case.
  - Continuous MDE: reported in standardized units (Cohen's d). Multiply by
    the empirical SD of a particular metric to get a raw MDE. The function
    `mde_continuous_d` returns d only.
  - Equal allocation is NOT assumed; we use the actual n1, n2 as-is (the
    harmonic mean dominates the variance term).
"""

from __future__ import annotations

import math
import os
import csv
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "unified.parquet"
OUT_DIR = REPO / "exploration" / "tables" / "T07"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Statistical helpers -------------------------------------------------

# z-scores for two-sided alpha=0.05 and power=0.80 (one-sided beta)
Z_ALPHA_2 = 1.959963984540054  # two-sided alpha=0.05
Z_BETA = 0.8416212335729143    # power=0.80

def _assert_helpers():
    # Sanity checks for helpers.
    # A two-sample t with equal n=5000 should be able to detect d ~ 0.056.
    d = mde_continuous_d(5000, 5000)
    assert 0.05 < d < 0.07, f"unexpected d: {d}"
    # Two-proportion MDE with p1=0.5 and n=n=5000 ~ 0.028 absolute difference.
    b = mde_binary_absolute(5000, 5000, p1=0.5)
    assert 0.02 < b < 0.04, f"unexpected binary MDE: {b}"
    # monotonic: larger n => smaller MDE
    assert mde_continuous_d(100, 100) > mde_continuous_d(1000, 1000)
    assert mde_binary_absolute(100, 100, p1=0.5) > mde_binary_absolute(1000, 1000, p1=0.5)


def mde_continuous_d(n1: int, n2: int) -> float:
    """Smallest standardized mean difference (Cohen's d) detectable at alpha=0.05,
    power=0.80, two-sided, pooled SD assumed equal.
    Uses z approximation (n large): d = (z_alpha/2 + z_beta) * sqrt(1/n1 + 1/n2).
    """
    if n1 <= 1 or n2 <= 1:
        return float("inf")
    return (Z_ALPHA_2 + Z_BETA) * math.sqrt(1.0 / n1 + 1.0 / n2)


def mde_binary_absolute(n1: int, n2: int, p1: float = 0.5) -> float:
    """Smallest detectable absolute difference p2 - p1 at alpha=0.05, power=0.80,
    two-sided, for a two-proportion z test. Uses a conservative variance estimate
    with the pooled p = p1 (worst case when p1=0.5).

    Reference: standard two-proportion sample size formula inverted.
    """
    if n1 <= 1 or n2 <= 1:
        return float("inf")
    # Conservative: use p(1-p) at p=p1; a tighter derivation would use (p1+p2)/2,
    # but we don't know p2 yet. p1=0.5 gives the hardest case.
    var = p1 * (1 - p1) * (1.0 / n1 + 1.0 / n2)
    return (Z_ALPHA_2 + Z_BETA) * math.sqrt(var)


def verdict(n1: int, n2: int, mde_binary_50: float) -> str:
    """Label the comparison. Thresholds chosen to reflect what's useful for RQ1.

    - well-powered: binary MDE @ p=0.5 <= 0.05 (5 pp) AND both cells >= 500
    - marginal: binary MDE <= 0.10 (10 pp) AND both cells >= 100
    - underpowered: otherwise
    """
    if n1 >= 500 and n2 >= 500 and mde_binary_50 <= 0.05:
        return "well-powered"
    if n1 >= 100 and n2 >= 100 and mde_binary_50 <= 0.10:
        return "marginal"
    return "underpowered"


# --- Data queries --------------------------------------------------------

FILT = (
    "source_platform='linkedin' AND is_english=true AND date_flag='ok' "
    "AND is_swe=true"
)


def group_sizes(con: duckdb.DuckDBPyConnection) -> dict:
    """Return a nested dict: {source: {seniority_final: n}} over filtered SWE rows.
    Also adds an 'ALL' key for total SWE by source.
    """
    rows = con.execute(f"""
        SELECT source, seniority_final, COUNT(*) AS n
        FROM '{DATA}'
        WHERE {FILT}
        GROUP BY source, seniority_final
    """).fetchall()
    out: dict = {}
    for s, sen, n in rows:
        out.setdefault(s, {})[sen or "null"] = n
    for s, d in out.items():
        d["ALL"] = sum(d.values())
    return out


def metro_feasibility(con: duckdb.DuckDBPyConnection) -> dict:
    """Metros with >=50 / >=100 SWE in BOTH arshkon and scraped (period proxies).

    Returns a dict with counts, qualifying metro lists, and an excluded-multi-location count.
    """
    rows = con.execute(f"""
        SELECT metro_area, source, COUNT(*) n
        FROM '{DATA}'
        WHERE {FILT}
          AND metro_area IS NOT NULL
          AND source IN ('kaggle_arshkon','scraped')
        GROUP BY metro_area, source
    """).fetchall()
    by_metro: dict = {}
    for m, s, n in rows:
        by_metro.setdefault(m, {})[s] = n

    def qual(th: int) -> list:
        return sorted(
            m for m, v in by_metro.items()
            if v.get("kaggle_arshkon", 0) >= th and v.get("scraped", 0) >= th
        )

    q50 = qual(50)
    q100 = qual(100)

    excluded = con.execute(f"""
        SELECT
          COUNT(*) FILTER (WHERE is_multi_location = true) AS multi_loc_rows,
          COUNT(*) FILTER (WHERE metro_area IS NULL) AS null_metro_rows,
          COUNT(*) FILTER (WHERE is_multi_location = true AND metro_area IS NULL) AS multi_loc_null,
          COUNT(*) FILTER (WHERE metro_area IS NULL AND is_multi_location = false) AS null_metro_not_multi
        FROM '{DATA}' WHERE {FILT}
    """).fetchone()

    return {
        "distinct_metros": len(by_metro),
        "qualifying_50": q50,
        "qualifying_100": q100,
        "excluded_multi_loc": excluded[0],
        "excluded_null_metro": excluded[1],
        "excluded_multi_loc_null_metro": excluded[2],
        "null_metro_not_multi_loc": excluded[3],
        "by_metro": by_metro,
    }


def company_overlap_panel(con: duckdb.DuckDBPyConnection) -> dict:
    """How many companies have >=k SWE postings in BOTH arshkon and scraped."""
    out = {}
    for k in (1, 2, 3, 5, 10):
        n = con.execute(f"""
            WITH swe AS (
              SELECT source, company_name_canonical FROM '{DATA}'
              WHERE {FILT}
                AND source IN ('kaggle_arshkon','scraped')
                AND company_name_canonical IS NOT NULL
            ),
            ark AS (
              SELECT company_name_canonical, COUNT(*) n
              FROM swe WHERE source='kaggle_arshkon'
              GROUP BY 1 HAVING COUNT(*) >= {k}
            ),
            scr AS (
              SELECT company_name_canonical, COUNT(*) n
              FROM swe WHERE source='scraped'
              GROUP BY 1 HAVING COUNT(*) >= {k}
            )
            SELECT COUNT(*) FROM ark JOIN scr USING(company_name_canonical)
        """).fetchone()[0]
        out[k] = n
    return out


# --- Main --------------------------------------------------------------

def main():
    _assert_helpers()
    con = duckdb.connect()

    sizes = group_sizes(con)
    metros = metro_feasibility(con)
    panel = company_overlap_panel(con)

    # Build the feasibility rows.
    rows = []

    def add(analysis_type: str, comparison: str, n1: int, n2: int, notes: str = ""):
        mde_b_50 = mde_binary_absolute(n1, n2, p1=0.5)
        mde_b_20 = mde_binary_absolute(n1, n2, p1=0.2)
        mde_d = mde_continuous_d(n1, n2)
        v = verdict(n1, n2, mde_b_50)
        rows.append({
            "analysis_type": analysis_type,
            "comparison": comparison,
            "n_group1": n1,
            "n_group2": n2,
            "MDE_binary_p50": round(mde_b_50, 4),
            "MDE_binary_p20": round(mde_b_20, 4),
            "MDE_continuous_d": round(mde_d, 4),
            "verdict": v,
            "notes": notes,
        })

    ark = sizes.get("kaggle_arshkon", {})
    asa = sizes.get("kaggle_asaniczka", {})
    scr = sizes.get("scraped", {})

    # -------- the four key comparisons --------
    add("seniority-stratified",
        "entry: arshkon vs scraped",
        ark.get("entry", 0), scr.get("entry", 0),
        "Uses seniority_final='entry'. Arshkon entry is rule-based; scraped entry mostly LLM.")

    add("seniority-stratified",
        "senior (mid-senior): arshkon vs scraped",
        ark.get("mid-senior", 0), scr.get("mid-senior", 0),
        "Uses seniority_final='mid-senior'.")

    add("all-SWE",
        "all SWE: arshkon vs scraped",
        ark.get("ALL", 0), scr.get("ALL", 0),
        "All is_swe rows regardless of seniority.")

    pooled_2024 = ark.get("ALL", 0) + asa.get("ALL", 0)
    add("all-SWE",
        "pooled 2024 vs scraped",
        pooled_2024, scr.get("ALL", 0),
        "kaggle_arshkon + kaggle_asaniczka vs scraped. Caveat: asaniczka is a different instrument (no native entry labels).")

    # -------- supporting rows --------
    add("seniority-stratified",
        "entry: pooled 2024 vs scraped",
        ark.get("entry", 0) + asa.get("entry", 0), scr.get("entry", 0),
        "Asaniczka entry signal comes from LLM only (no native entry labels).")

    add("seniority-stratified",
        "mid-senior: pooled 2024 vs scraped",
        ark.get("mid-senior", 0) + asa.get("mid-senior", 0), scr.get("mid-senior", 0),
        "")

    add("seniority-stratified",
        "unknown-seniority: arshkon vs scraped",
        ark.get("unknown", 0), scr.get("unknown", 0),
        "Residual category; high in scraped because many rows weren't routed to LLM.")

    # 3-level collapse: junior = entry + associate
    add("seniority-stratified (3-level)",
        "junior (entry+assoc): arshkon vs scraped",
        ark.get("entry", 0) + ark.get("associate", 0),
        scr.get("entry", 0) + scr.get("associate", 0),
        "seniority_3level='junior'.")

    # Metro-level analysis: use the smallest qualifying metro cell as the binding constraint
    q50 = metros["qualifying_50"]
    if q50:
        min_cell = min(
            min(metros["by_metro"][m]["kaggle_arshkon"], metros["by_metro"][m]["scraped"])
            for m in q50
        )
        add("metro-level",
            f"smallest qualifying metro (n={len(q50)} metros, >=50 threshold)",
            min_cell, min_cell,
            "Per-metro comparison; worst-case metro pair. Typical metro has much larger n.")
    q100 = metros["qualifying_100"]
    if q100:
        min_cell100 = min(
            min(metros["by_metro"][m]["kaggle_arshkon"], metros["by_metro"][m]["scraped"])
            for m in q100
        )
        add("metro-level",
            f"smallest qualifying metro (n={len(q100)} metros, >=100 threshold)",
            min_cell100, min_cell100,
            "Same as above but tighter threshold.")

    # Company overlap panel (>=3 SWE in both sources)
    add("company panel",
        f"company overlap panel (>=3 SWE both sources, N={panel[3]} companies)",
        panel[3], panel[3],
        "Unit of analysis is the company, not the posting. N here is the number of companies in the panel.")

    # Write the CSV.
    primary = OUT_DIR / "feasibility.csv"
    with primary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {primary}")

    # Also write supporting tables.
    # 1) raw group sizes
    gs_path = OUT_DIR / "group_sizes.csv"
    with gs_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "seniority_final", "n"])
        for src in sorted(sizes):
            for sen, n in sorted(sizes[src].items()):
                w.writerow([src, sen, n])
    print(f"Wrote {gs_path}")

    # 2) metro table with per-metro arshkon + scraped counts
    metro_path = OUT_DIR / "metro_feasibility.csv"
    with metro_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "metro_area", "n_arshkon", "n_scraped", "min_cell",
            "qualifies_50", "qualifies_100",
        ])
        for m, v in sorted(metros["by_metro"].items()):
            a = v.get("kaggle_arshkon", 0)
            s = v.get("scraped", 0)
            mn = min(a, s)
            w.writerow([m, a, s, mn, mn >= 50, mn >= 100])
    print(f"Wrote {metro_path}")

    # 3) company overlap panel thresholds
    panel_path = OUT_DIR / "company_panel.csv"
    with panel_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["min_postings_per_source", "n_companies_in_panel"])
        for k, n in sorted(panel.items()):
            w.writerow([k, n])
    print(f"Wrote {panel_path}")

    # Summary to stdout
    print()
    print("=== Feasibility summary ===")
    for r in rows:
        print(
            f"  [{r['verdict']:13s}] {r['comparison']:55s} "
            f"n1={r['n_group1']:>7d} n2={r['n_group2']:>7d} "
            f"MDE_bin50={r['MDE_binary_p50']:.3f} MDE_d={r['MDE_continuous_d']:.3f}"
        )
    print()
    print(f"Metros (>=50 both periods): {len(metros['qualifying_50'])}")
    print(f"Metros (>=100 both periods): {len(metros['qualifying_100'])}")
    print(f"Rows excluded because metro_area IS NULL: {metros['excluded_null_metro']}")
    print(f"  of which is_multi_location=true: {metros['excluded_multi_loc_null_metro']}")
    print(f"Company overlap panel (>=3 both sources): {panel[3]}")


if __name__ == "__main__":
    main()
