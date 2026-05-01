"""T07 Part A: Feasibility table for cross-period comparisons.

Computes group sizes under each T30 panel variant (J1-J4, S1-S4, all-SWE),
minimum detectable effect sizes (MDE) for binary and continuous outcomes
at 80% power / alpha=0.05, metro-level feasibility, and company overlap.

Outputs
- exploration/tables/T07/group_sizes.csv        # n_arshkon, n_asaniczka, n_scraped per definition
- exploration/tables/T07/feasibility_table.csv  # key cross-tab deliverable
- exploration/tables/T07/metro_feasibility.csv  # per-metro SWE counts by period
- exploration/tables/T07/company_overlap.csv    # T16 panel feasibility
"""
from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pandas as pd
from scipy import stats

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data" / "unified.parquet"
OUT = ROOT / "exploration" / "tables" / "T07"
OUT.mkdir(parents=True, exist_ok=True)

# Regex sanity asserts (TDD rule in preamble)
import re
J5_RE = re.compile(r"\b(junior|jr|entry[- ]level|graduate|new[- ]grad|intern)\b", re.IGNORECASE)
S3_RE = re.compile(r"\b(senior|sr\.?|staff|principal|lead|architect|distinguished)\b", re.IGNORECASE)

# J5 edge cases
assert J5_RE.search("Junior Software Engineer")
assert J5_RE.search("Jr. Developer")
assert J5_RE.search("Entry-Level Engineer")
assert J5_RE.search("Entry Level Engineer")
assert J5_RE.search("New Grad Software Engineer")
assert J5_RE.search("new-grad SWE")
assert J5_RE.search("Software Engineering Intern")
assert not J5_RE.search("Integrator")  # shouldn't match intern via prefix
assert not J5_RE.search("senior engineer")

# S3 edge cases
assert S3_RE.search("Senior Software Engineer")
assert S3_RE.search("Sr Developer")
assert S3_RE.search("Sr. Engineer")
assert S3_RE.search("Staff Engineer")
assert S3_RE.search("Principal Engineer")
assert S3_RE.search("Lead Developer")
assert S3_RE.search("Solutions Architect")
assert S3_RE.search("Distinguished Engineer")
assert not S3_RE.search("Junior Engineer")
# Known false-positive caveat: "senior" appearing in phrases like "senior stakeholders"
# is flagged via title-only regex (T30 precision sample audits this); reported here.
print("Regex asserts passed.")

# DuckDB-compatible regex (case-insensitive via (?i))
J5_SQL_RE = r"(?i)\b(junior|jr|entry[- ]level|graduate|new[- ]grad|intern)\b"
S3_SQL_RE = r"(?i)\b(senior|sr\.?|staff|principal|lead|architect|distinguished)\b"

DEFAULT_FILTER = """
  source_platform = 'linkedin'
  AND is_english = true
  AND date_flag = 'ok'
  AND is_swe = true
"""

DEF_EXPR = {
    # junior side
    "J1": "seniority_final = 'entry'",
    "J2": "seniority_final IN ('entry','associate')",
    "J3": "yoe_extracted <= 2",
    "J4": "yoe_extracted <= 3",
    # senior side
    "S1": "seniority_final IN ('mid-senior','director')",
    "S2": "seniority_final = 'director'",
    # S3 as-written in the spec (matches `title_normalized`). Note: `title_normalized`
    # has level indicators STRIPPED (per preprocessing-schema.md Section 2), so this
    # operationalization loses ~95% of legitimate senior-keyword rows across all sources.
    # T30 should flag this and use S3_raw instead.
    "S3": f"regexp_matches(title_normalized, '{S3_SQL_RE}')",
    # S3_raw: corrected version using raw title (lowercased). This is what the spec
    # intended — title-keyword senior.
    "S3_raw": f"regexp_matches(lower(title), '{S3_SQL_RE}')",
    "S4": "yoe_extracted >= 5",
    # all-SWE baseline
    "ALL": "TRUE",
}
DEF_SIDE = {
    "J1": "junior", "J2": "junior", "J3": "junior", "J4": "junior",
    "S1": "senior", "S2": "senior", "S3": "senior", "S3_raw": "senior", "S4": "senior",
    "ALL": "all",
}

# Denominators for "share of all" vs "share of known-seniority" vs "share of YOE-known"
DENOM_EXPR = {
    "J1": "TRUE",  # denominator = all SWE
    "J2": "TRUE",
    "J3": "yoe_extracted IS NOT NULL",  # YOE-known denominator
    "J4": "yoe_extracted IS NOT NULL",
    "S1": "TRUE",
    "S2": "TRUE",
    "S3": "TRUE",
    "S3_raw": "TRUE",
    "S4": "yoe_extracted IS NOT NULL",
    "ALL": "TRUE",
}


def compute_group_sizes(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """n (satisfying definition) and n_denom (rows in the applicable denominator) per source."""
    rows = []
    for dname, expr in DEF_EXPR.items():
        denom_expr = DENOM_EXPR[dname]
        q = f"""
        SELECT source,
               COUNT(*) FILTER (WHERE {denom_expr}) AS n_denom,
               COUNT(*) FILTER (WHERE {expr} AND {denom_expr}) AS n_sat,
               COUNT(*) FILTER (WHERE {expr} AND {denom_expr})::DOUBLE / NULLIF(COUNT(*) FILTER (WHERE {denom_expr}), 0) AS share
        FROM read_parquet('{DATA}')
        WHERE {DEFAULT_FILTER}
        GROUP BY source
        ORDER BY source
        """
        df = con.execute(q).df()
        df["definition"] = dname
        df["side"] = DEF_SIDE[dname]
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    return out[["definition", "side", "source", "n_denom", "n_sat", "share"]]


# Power analysis helpers ---------------------------------------------------
Z_ALPHA_TWOSIDED = stats.norm.ppf(0.975)  # 1.959964
Z_BETA = stats.norm.ppf(0.80)  # 0.841621


def mde_binary(n1: int, n2: int, p0: float) -> float:
    """MDE on a proportion difference at 80% power, alpha=0.05 two-sided.

    Uses normal approximation:
        delta = (z_alpha * sqrt(p_bar * (1 - p_bar) * (1/n1 + 1/n2))
                 + z_beta  * sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2))
    Simplification assumes p1 ~= p2 ~= p0 under null. Good enough for feasibility
    screening. Returns absolute proportion difference (pp, on 0-1 scale).

    Reference: standard two-sample proportion z-test power formula.
    """
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    if p0 is None or math.isnan(p0):
        p0 = 0.1  # default placeholder; MDE is ~flat for 0.05 < p < 0.5
    # pooled variance approximation (conservative)
    var = p0 * (1 - p0) * (1.0 / n1 + 1.0 / n2)
    mde = (Z_ALPHA_TWOSIDED + Z_BETA) * math.sqrt(var)
    return mde


def mde_cohen_d(n1: int, n2: int) -> float:
    """MDE on Cohen's d at 80% power, alpha=0.05 (two-sample t).

    d = (z_alpha + z_beta) * sqrt(1/n1 + 1/n2). Uses z approximation for large n.
    """
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    return (Z_ALPHA_TWOSIDED + Z_BETA) * math.sqrt(1.0 / n1 + 1.0 / n2)


def verdict(mde_b: float, mde_c: float) -> str:
    """well_powered | marginal | underpowered based on the more permissive of the two."""
    tags = []
    if not math.isnan(mde_b):
        if mde_b < 0.05:
            tags.append("well_powered")
        elif mde_b < 0.10:
            tags.append("marginal")
        else:
            tags.append("underpowered")
    if not math.isnan(mde_c):
        if mde_c < 0.20:
            tags.append("well_powered")
        elif mde_c < 0.40:
            tags.append("marginal")
        else:
            tags.append("underpowered")
    # take the worse (most conservative) of binary and continuous verdicts
    order = {"well_powered": 0, "marginal": 1, "underpowered": 2}
    if not tags:
        return "underpowered"
    return max(tags, key=lambda t: order[t])


def verdict_binary_only(mde_b: float) -> str:
    if math.isnan(mde_b):
        return "underpowered"
    if mde_b < 0.05:
        return "well_powered"
    if mde_b < 0.10:
        return "marginal"
    return "underpowered"


def verdict_continuous_only(mde_c: float) -> str:
    if math.isnan(mde_c):
        return "underpowered"
    if mde_c < 0.20:
        return "well_powered"
    if mde_c < 0.40:
        return "marginal"
    return "underpowered"


# Assumed prevalence (p0) for each analysis type. Used in the binary MDE formula.
# These come from the preprocessing schema and typical observed rates.
ANALYSIS_TYPES = {
    # analysis_type: (p0_assumed, needs_seniority_def)
    "junior_share":            (0.10, True),   # share of juniors among SWE
    "senior_share":            (0.60, True),   # mid-senior+director
    "ai_mention_rate":         (0.20, False),  # binary: any AI tool mention
    "mgmt_mention_rate":       (0.30, False),  # management/leadership language
    "description_length":      (None, False),  # continuous — only d MDE
    "yoe_mean":                (None, False),  # continuous
    "remote_share":            (0.40, False),  # share with remote/hybrid
    "aggregator_share":        (0.15, False),  # share from aggregator
    "metro_entry_share":       (0.10, True),   # metro-level entry share
}


def build_feasibility_table(sizes: pd.DataFrame) -> pd.DataFrame:
    """Build the cross-tab of (comparison × definition × analysis_type)."""
    # Extract n per (source, definition) as a dict
    # Use n_sat (rows satisfying the definition) as the active-group n.
    # For ratio/share outcomes the active n is what matters; for all-SWE the n is total.
    sz = {}
    for _, row in sizes.iterrows():
        sz[(row["source"], row["definition"])] = int(row["n_sat"])

    n_arshkon = {d: sz.get(("kaggle_arshkon", d), 0) for d in DEF_EXPR}
    n_asaniczka = {d: sz.get(("kaggle_asaniczka", d), 0) for d in DEF_EXPR}
    n_scraped = {d: sz.get(("scraped", d), 0) for d in DEF_EXPR}
    n_pooled_2024 = {d: n_arshkon[d] + n_asaniczka[d] for d in DEF_EXPR}

    rows = []
    # Comparisons with labels and group-n dicts
    comparisons = [
        ("arshkon_vs_scraped",        n_arshkon, n_scraped),
        ("pooled_2024_vs_scraped",    n_pooled_2024, n_scraped),
    ]
    # arshkon_only senior vs scraped senior is a subset of arshkon_vs_scraped
    # restricted to senior definitions. All-SWE is one row.
    # We'll produce one row per (analysis_type, comparison, seniority_def).

    for atype, (p0, needs_def) in ANALYSIS_TYPES.items():
        # pick definitions based on analysis type
        if atype == "junior_share":
            defs = ["J1", "J2", "J3", "J4"]
        elif atype == "senior_share":
            defs = ["S1", "S2", "S3", "S4"]
        elif atype == "metro_entry_share":
            # metro-level: use J1 as primary (other defs noted in report)
            defs = ["J1", "J2", "J3", "J4"]
        elif needs_def:
            defs = ["J1", "J2", "J3", "J4", "S1", "S2", "S3", "S4"]
        else:
            # content/outcome metrics: report per definition-strata AND for all-SWE
            defs = ["J1", "J2", "J3", "J4", "S1", "S2", "S3", "S4", "ALL"]

        for cname, n1d, n2d in comparisons:
            # arshkon-only senior vs scraped senior rule: captured by (arshkon_vs_scraped, S1..S4)
            for d in defs:
                n1 = n1d[d]
                n2 = n2d[d]
                # compute MDE
                if atype in ("description_length", "yoe_mean"):
                    # continuous only
                    mb = float("nan")
                    mc = mde_cohen_d(n1, n2)
                    v = verdict_continuous_only(mc)
                elif atype in ("junior_share", "senior_share", "metro_entry_share"):
                    # For share-of-definition at population level, the denominator n is
                    # total SWE (or YOE-known), NOT n_sat. Re-use totals:
                    pass
                    mb = mde_binary(n1, n2, p0)  # using n_sat as group size is the
                    # conservative analogue when the "group" is the defined slice itself.
                    # For share changes at the POPULATION level the relevant n is the
                    # total SWE-per-source (same denominator as the share). We'll
                    # generate a separate "population-denominator" MDE below.
                    mc = float("nan")
                    v = verdict_binary_only(mb)
                else:
                    # binary outcome rate within the defined slice
                    mb = mde_binary(n1, n2, p0)
                    mc = mde_cohen_d(n1, n2)
                    v = verdict(mb, mc)

                rows.append({
                    "analysis_type": atype,
                    "comparison": cname,
                    "seniority_def": d if d != "ALL" else "N/A",
                    "n_group1": n1,
                    "n_group2": n2,
                    "MDE_binary": round(mb, 4) if not math.isnan(mb) else None,
                    "MDE_continuous": round(mc, 4) if not math.isnan(mc) else None,
                    "verdict": v,
                })
    # For share-of-definition outcomes (junior_share, senior_share), override MDE
    # using the POPULATION-level denominator so the number reflects "pp change in
    # the share among all SWE" rather than within-slice variance.
    # Load total SWE per source:
    pass
    return pd.DataFrame(rows)


def build_feasibility_table_v2(sizes: pd.DataFrame) -> pd.DataFrame:
    """Clean version: uses proper denominators for each analysis type.

    For junior_share / senior_share: the group-n is the TOTAL rows in the
    comparator (all SWE for J1/J2/S1-S3; YOE-known for J3/J4/S4). The
    event rate (p0) is the observed share in that slice.

    For within-slice content metrics (ai_mention_rate, description_length, etc.):
    the group-n is n_sat (number of rows satisfying the definition). Sample
    size limits how well we can measure within-slice quantities.
    """
    # totals per source (all SWE)
    sz_all = {}
    sz_yoe_known = {}
    sz_definition = {}
    for _, row in sizes.iterrows():
        key = (row["source"], row["definition"])
        if row["definition"] == "ALL":
            sz_all[row["source"]] = int(row["n_sat"])
        sz_definition[key] = int(row["n_sat"])
        # n_denom is YOE-known for J3/J4/S4, else same as total
        if row["definition"] in ("J3", "J4", "S4"):
            sz_yoe_known[row["source"]] = int(row["n_denom"])

    def get_total(source: str, dname: str) -> int:
        if dname in ("J3", "J4", "S4"):
            return sz_yoe_known.get(source, 0)
        return sz_all.get(source, 0)

    # Observed share (used as p0 for share-outcome MDEs)
    # Use the scraped-source share as p0 (conservative: if 2024 share was lower,
    # the MDE at p0=scraped-share is slightly larger)
    share_lookup = {}
    for _, row in sizes.iterrows():
        share_lookup[(row["source"], row["definition"])] = (
            float(row["share"]) if row["share"] is not None else 0.0
        )

    sources = {
        "arshkon": "kaggle_arshkon",
        "asaniczka": "kaggle_asaniczka",
        "scraped": "scraped",
    }
    # Pre-compute "pooled 2024" values (arshkon + asaniczka)
    pooled_total = {d: get_total("kaggle_arshkon", d) + get_total("kaggle_asaniczka", d)
                    for d in DEF_EXPR}
    pooled_sat = {d: sz_definition.get(("kaggle_arshkon", d), 0)
                  + sz_definition.get(("kaggle_asaniczka", d), 0)
                  for d in DEF_EXPR}
    pooled_share = {d: (pooled_sat[d] / pooled_total[d]) if pooled_total[d] > 0 else 0.0
                    for d in DEF_EXPR}

    def comparison_totals(cname: str, d: str):
        if cname == "arshkon_vs_scraped":
            return (get_total("kaggle_arshkon", d), get_total("scraped", d))
        if cname == "pooled_2024_vs_scraped":
            return (pooled_total[d], get_total("scraped", d))
        raise ValueError(cname)

    def comparison_sat(cname: str, d: str):
        if cname == "arshkon_vs_scraped":
            return (sz_definition.get(("kaggle_arshkon", d), 0),
                    sz_definition.get(("scraped", d), 0))
        if cname == "pooled_2024_vs_scraped":
            return (pooled_sat[d], sz_definition.get(("scraped", d), 0))
        raise ValueError(cname)

    rows = []
    comparisons_list = ["arshkon_vs_scraped", "pooled_2024_vs_scraped"]

    # Analysis rows ------------------------------------------------------
    for atype, (p0, _) in ANALYSIS_TYPES.items():
        if atype == "junior_share":
            defs = ["J1", "J2", "J3", "J4"]
        elif atype == "senior_share":
            defs = ["S1", "S2", "S3", "S3_raw", "S4"]
        elif atype == "metro_entry_share":
            defs = ["J1", "J2", "J3", "J4"]
        else:
            # content/outcome metrics: stratified by seniority_def + also all-SWE
            defs = ["J1", "J2", "J3", "J4", "S1", "S2", "S3", "S3_raw", "S4", "ALL"]

        for cname in comparisons_list:
            for d in defs:
                if atype in ("junior_share", "senior_share"):
                    # share-of-population: n = total SWE in source
                    n1, n2 = comparison_totals(cname, d)
                    # use observed share under scraped as p0 (fallback to default)
                    p_obs = share_lookup.get(("scraped", d), p0 or 0.1)
                    if p_obs == 0 or p_obs is None:
                        p_obs = p0 or 0.1
                    mb = mde_binary(n1, n2, p_obs)
                    mc = float("nan")
                    v = verdict_binary_only(mb)
                elif atype == "metro_entry_share":
                    # per-metro feasibility handled separately below; for the table,
                    # we use a representative top-metro n of ~200 per period. This
                    # row is marker; see metro_feasibility.csv for per-metro detail.
                    # Skip in table (would be misleading); emit a summary row.
                    continue
                elif atype in ("description_length", "yoe_mean"):
                    # continuous outcome WITHIN the definition slice.
                    if d == "ALL":
                        n1, n2 = comparison_totals(cname, "ALL")
                    else:
                        n1, n2 = comparison_sat(cname, d)
                    mb = float("nan")
                    mc = mde_cohen_d(n1, n2)
                    v = verdict_continuous_only(mc)
                else:
                    # binary content metric WITHIN the definition slice
                    if d == "ALL":
                        n1, n2 = comparison_totals(cname, "ALL")
                    else:
                        n1, n2 = comparison_sat(cname, d)
                    mb = mde_binary(n1, n2, p0 or 0.1)
                    mc = mde_cohen_d(n1, n2)
                    v = verdict(mb, mc)

                rows.append({
                    "analysis_type": atype,
                    "comparison": cname,
                    "seniority_def": d if d != "ALL" else "N/A",
                    "n_group1": n1,
                    "n_group2": n2,
                    "MDE_binary": round(mb, 4) if not math.isnan(mb) else None,
                    "MDE_continuous": round(mc, 4) if not math.isnan(mc) else None,
                    "verdict": v,
                })
    return pd.DataFrame(rows)


def metro_feasibility(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per-metro SWE counts by period. Report multi-location separately."""
    q = f"""
    SELECT metro_area, source, period,
           COUNT(*) AS n
    FROM read_parquet('{DATA}')
    WHERE {DEFAULT_FILTER}
      AND metro_area IS NOT NULL
    GROUP BY metro_area, source, period
    ORDER BY metro_area, source, period
    """
    by_metro_period = con.execute(q).df()

    # Multi-location
    q_ml = f"""
    SELECT source, COUNT(*) AS n_multi_location
    FROM read_parquet('{DATA}')
    WHERE {DEFAULT_FILTER}
      AND is_multi_location = true
    GROUP BY source
    ORDER BY source
    """
    multi_loc = con.execute(q_ml).df()
    print("\nMulti-location (excluded from metro counts):")
    print(multi_loc.to_string())

    # Pivot to wide per-metro×period matrix with all three sources
    pivot = by_metro_period.pivot_table(
        index="metro_area",
        columns=["source", "period"],
        values="n",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()
    # flatten column names
    pivot.columns = [
        "__".join(str(x) for x in col if x != "") if isinstance(col, tuple) else col
        for col in pivot.columns
    ]
    return pivot, multi_loc


def company_overlap(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Number of companies with ≥3/5/10 SWE postings in BOTH arshkon and scraped."""
    q = f"""
    WITH arshkon_companies AS (
      SELECT company_name_canonical, COUNT(*) AS n_arshkon
      FROM read_parquet('{DATA}')
      WHERE {DEFAULT_FILTER}
        AND source = 'kaggle_arshkon'
        AND company_name_canonical IS NOT NULL
      GROUP BY company_name_canonical
    ),
    scraped_companies AS (
      SELECT company_name_canonical, COUNT(*) AS n_scraped
      FROM read_parquet('{DATA}')
      WHERE {DEFAULT_FILTER}
        AND source = 'scraped'
        AND company_name_canonical IS NOT NULL
      GROUP BY company_name_canonical
    ),
    asaniczka_companies AS (
      SELECT company_name_canonical, COUNT(*) AS n_asaniczka
      FROM read_parquet('{DATA}')
      WHERE {DEFAULT_FILTER}
        AND source = 'kaggle_asaniczka'
        AND company_name_canonical IS NOT NULL
      GROUP BY company_name_canonical
    )
    SELECT
      -- arshkon vs scraped
      COUNT(*) FILTER (WHERE a.n_arshkon >= 1 AND s.n_scraped >= 1) AS arshkon_scraped_any,
      COUNT(*) FILTER (WHERE a.n_arshkon >= 3 AND s.n_scraped >= 3) AS arshkon_scraped_3plus,
      COUNT(*) FILTER (WHERE a.n_arshkon >= 5 AND s.n_scraped >= 5) AS arshkon_scraped_5plus,
      COUNT(*) FILTER (WHERE a.n_arshkon >= 10 AND s.n_scraped >= 10) AS arshkon_scraped_10plus,
      -- row contributions
      SUM(CASE WHEN a.n_arshkon >= 3 AND s.n_scraped >= 3 THEN a.n_arshkon ELSE 0 END) AS sum_arshkon_in_3plus_panel,
      SUM(CASE WHEN a.n_arshkon >= 3 AND s.n_scraped >= 3 THEN s.n_scraped ELSE 0 END) AS sum_scraped_in_3plus_panel,
      SUM(CASE WHEN a.n_arshkon >= 5 AND s.n_scraped >= 5 THEN a.n_arshkon ELSE 0 END) AS sum_arshkon_in_5plus_panel,
      SUM(CASE WHEN a.n_arshkon >= 5 AND s.n_scraped >= 5 THEN s.n_scraped ELSE 0 END) AS sum_scraped_in_5plus_panel
    FROM arshkon_companies a
    INNER JOIN scraped_companies s ON a.company_name_canonical = s.company_name_canonical
    """
    arshkon_scraped = con.execute(q).df()

    # Pooled 2024 (arshkon + asaniczka) vs scraped
    q2 = f"""
    WITH pooled_2024_companies AS (
      SELECT company_name_canonical, COUNT(*) AS n_2024
      FROM read_parquet('{DATA}')
      WHERE {DEFAULT_FILTER}
        AND source IN ('kaggle_arshkon','kaggle_asaniczka')
        AND company_name_canonical IS NOT NULL
      GROUP BY company_name_canonical
    ),
    scraped_companies AS (
      SELECT company_name_canonical, COUNT(*) AS n_scraped
      FROM read_parquet('{DATA}')
      WHERE {DEFAULT_FILTER}
        AND source = 'scraped'
        AND company_name_canonical IS NOT NULL
      GROUP BY company_name_canonical
    )
    SELECT
      COUNT(*) FILTER (WHERE p.n_2024 >= 1 AND s.n_scraped >= 1) AS pooled_scraped_any,
      COUNT(*) FILTER (WHERE p.n_2024 >= 3 AND s.n_scraped >= 3) AS pooled_scraped_3plus,
      COUNT(*) FILTER (WHERE p.n_2024 >= 5 AND s.n_scraped >= 5) AS pooled_scraped_5plus,
      COUNT(*) FILTER (WHERE p.n_2024 >= 10 AND s.n_scraped >= 10) AS pooled_scraped_10plus
    FROM pooled_2024_companies p
    INNER JOIN scraped_companies s ON p.company_name_canonical = s.company_name_canonical
    """
    pooled_scraped = con.execute(q2).df()

    combined = pd.concat([arshkon_scraped, pooled_scraped], axis=1)
    return combined


def state_sample_counts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """State-level SWE counts by source (used for BLS correlation)."""
    q = f"""
    SELECT state_normalized, source, COUNT(*) AS n
    FROM read_parquet('{DATA}')
    WHERE {DEFAULT_FILTER}
      AND state_normalized IS NOT NULL
      AND is_multi_location = false
    GROUP BY state_normalized, source
    ORDER BY state_normalized, source
    """
    return con.execute(q).df()


def industry_distribution(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Industry distribution in SWE (arshkon + scraped — asaniczka has no industry)."""
    q = f"""
    SELECT source, company_industry, COUNT(*) AS n
    FROM read_parquet('{DATA}')
    WHERE {DEFAULT_FILTER}
      AND company_industry IS NOT NULL
      AND source IN ('kaggle_arshkon','scraped')
    GROUP BY source, company_industry
    ORDER BY source, n DESC
    """
    return con.execute(q).df()


def main():
    con = duckdb.connect()
    print(f"Reading {DATA}")
    print("\n=== Group sizes under T30 panel ===")
    sizes = compute_group_sizes(con)
    sizes.to_csv(OUT / "group_sizes.csv", index=False)
    # Pretty print as matrix
    piv = sizes.pivot_table(index=["definition", "side"],
                            columns="source",
                            values="n_sat",
                            aggfunc="sum",
                            fill_value=0).reset_index()
    print(piv.to_string())

    print("\n=== Feasibility table ===")
    feas = build_feasibility_table_v2(sizes)
    feas.to_csv(OUT / "feasibility_table.csv", index=False)
    # Print a summary of verdicts per (comparison, seniority_def) for junior/senior shares
    print("\nJunior share verdicts:")
    js = feas[feas["analysis_type"] == "junior_share"][
        ["comparison", "seniority_def", "n_group1", "n_group2", "MDE_binary", "verdict"]
    ]
    print(js.to_string(index=False))
    print("\nSenior share verdicts:")
    ss = feas[feas["analysis_type"] == "senior_share"][
        ["comparison", "seniority_def", "n_group1", "n_group2", "MDE_binary", "verdict"]
    ]
    print(ss.to_string(index=False))
    print("\nAll-SWE content metric verdicts:")
    allm = feas[feas["seniority_def"] == "N/A"][
        ["analysis_type", "comparison", "n_group1", "n_group2", "MDE_binary", "MDE_continuous", "verdict"]
    ]
    print(allm.to_string(index=False))

    print("\n=== Metro feasibility ===")
    metro_pivot, multi_loc = metro_feasibility(con)
    metro_pivot.to_csv(OUT / "metro_feasibility.csv", index=False)
    multi_loc.to_csv(OUT / "multi_location_excluded.csv", index=False)
    # Summary: metros with >=50 and >=100 SWE by source-period
    q_summary = f"""
    WITH per_metro AS (
      SELECT metro_area, source, period, COUNT(*) AS n
      FROM read_parquet('{DATA}')
      WHERE {DEFAULT_FILTER}
        AND metro_area IS NOT NULL
      GROUP BY metro_area, source, period
    )
    SELECT source, period,
           COUNT(*) FILTER (WHERE n >= 50) AS metros_ge_50,
           COUNT(*) FILTER (WHERE n >= 100) AS metros_ge_100,
           COUNT(*) AS total_metros,
           SUM(n) AS total_swe_with_metro
    FROM per_metro
    GROUP BY source, period
    ORDER BY source, period
    """
    metro_summary = con.execute(q_summary).df()
    metro_summary.to_csv(OUT / "metro_summary.csv", index=False)
    print(metro_summary.to_string())

    print("\n=== Company overlap ===")
    overlap = company_overlap(con)
    overlap.to_csv(OUT / "company_overlap.csv", index=False)
    print(overlap.to_string())

    print("\n=== State-level sample (for BLS correlation) ===")
    states = state_sample_counts(con)
    states.to_csv(OUT / "state_counts.csv", index=False)
    # Summary of state coverage
    print(states.groupby("source")["n"].agg(["count", "sum"]).to_string())

    print("\n=== Industry distribution ===")
    ind = industry_distribution(con)
    ind.to_csv(OUT / "industry_distribution.csv", index=False)
    print(f"Industries: {len(ind['company_industry'].unique())} unique")
    print(f"Rows: arshkon={len(ind[ind.source=='kaggle_arshkon'])}, "
          f"scraped={len(ind[ind.source=='scraped'])}")

    print("\nDone.")


if __name__ == "__main__":
    main()
