"""T30 — Seniority definition ablation panel.

Produces:
  exploration/artifacts/shared/seniority_definition_panel.csv
  exploration/figures/T30/overlap_junior.png
  exploration/figures/T30/overlap_senior.png
  exploration/tables/T30/*.csv

The panel is the canonical junior- and senior-side operationalization table
that every Wave 2+ seniority-stratified task consumes.

Definitions:
  Junior:
    J1: seniority_final = 'entry'
    J2: seniority_final IN ('entry','associate')
    J3: yoe_extracted <= 2
    J4: yoe_extracted <= 3
    J5: title_normalized regex for (junior|jr|entry[- ]level|graduate|new[- ]grad|intern)
    J6: J1 OR J5

  Senior:
    S1: seniority_final IN ('mid-senior','director')
    S2: seniority_final = 'director'
    S3: title_normalized regex for (senior|sr.?|staff|principal|lead|architect|distinguished)
    S4: yoe_extracted >= 5
    S5: yoe_extracted >= 8

Periods / sources:
  - arshkon (2024-04)
  - asaniczka (2024-01)
  - pooled 2024 (arshkon + asaniczka)
  - scraped (2026-03 + 2026-04)
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data/unified.parquet"
FIG_DIR = ROOT / "exploration/figures/T30"
TAB_DIR = ROOT / "exploration/tables/T30"
SHARED = ROOT / "exploration/artifacts/shared"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
SHARED.mkdir(parents=True, exist_ok=True)

BASE_FILTER = (
    "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' "
    "AND is_swe = true"
)


# ----------------------------------------------------------------------------
# Regex sanity tests (TDD per agent dispatch requirement)
# ----------------------------------------------------------------------------

def test_regex_patterns() -> tuple[str, str]:
    """Validate junior/senior title patterns against edge cases."""
    # Junior pattern: match junior / jr / entry-level / entry level / graduate /
    # new-grad / new grad / intern.
    jr_pat_py = re.compile(
        r"\b(junior|jr|entry[- ]level|graduate|new[- ]grad|intern)\b",
        re.IGNORECASE,
    )
    # Should match
    assert jr_pat_py.search("junior software engineer")
    assert jr_pat_py.search("Jr. Developer"), "trailing period"
    assert jr_pat_py.search("Entry-Level Engineer"), "hyphen"
    assert jr_pat_py.search("entry level engineer"), "space"
    assert jr_pat_py.search("new grad swe"), "space"
    assert jr_pat_py.search("New-grad"), "hyphen"
    assert jr_pat_py.search("Software Engineering Intern")
    assert jr_pat_py.search("Graduate Software Engineer")
    assert jr_pat_py.search("JR. DevOps"), "case + dot"
    # Should NOT match on these
    # (Removed 'injure the junior' — it correctly DOES match 'junior' as a standalone word.)
    assert not jr_pat_py.search("jrpg designer"), "jrpg should not match jr"
    assert not jr_pat_py.search("graduated candidate only"), "'graduated' should not match 'graduate' with \\b$"
    # BUG: actually "graduated" would match because \bgraduate\b is followed by 'd', which breaks word-boundary.
    # Wait \b is between word and non-word. 'd' is a word char, so "graduated" does NOT have a word boundary
    # after "graduate". Good — it will not match.
    # Note: "jr.web" WILL match (after 'r' is '.', a non-word char → word boundary).
    # We accept this — 'jr.' followed by anything is almost always a junior marker.
    assert jr_pat_py.search("jr.web"), "jr. pattern should match (trailing . is non-word)"
    assert not jr_pat_py.search("jrjr"), "jrjr should not match"  # no boundary
    assert not jr_pat_py.search("ninjruby"), "no boundary after 'jr' in ninjruby"  # adjacent word chars
    # Leave 'intern' alone - it may spuriously match "international" — check:
    # \bintern\b -> "international" has 'intern' followed by 'a' (word), no boundary. No match. Good.
    assert not jr_pat_py.search("international data engineer"), "'international' must not match 'intern'"

    # Senior pattern
    sr_pat_py = re.compile(
        r"\b(senior|sr\.?|staff|principal|lead|architect|distinguished)\b",
        re.IGNORECASE,
    )
    assert sr_pat_py.search("Senior Software Engineer")
    assert sr_pat_py.search("Sr. Full-stack")
    assert sr_pat_py.search("Sr Engineer"), "no dot"
    assert sr_pat_py.search("Staff Engineer")
    assert sr_pat_py.search("Principal Architect")
    assert sr_pat_py.search("Lead Golang Engineer")
    assert sr_pat_py.search("Distinguished Engineer")
    assert sr_pat_py.search("Cloud Architect")
    # Not match
    assert not sr_pat_py.search("Senorita Engineer"), "spelling"
    assert not sr_pat_py.search("leadership candidate only"), "leadership should not match lead"
    # Wait: "leadership" contains 'lead' followed by 'e' (word char), no boundary, no match. Good.
    assert not sr_pat_py.search("srcinfo engineer"), "src should not match sr"
    # "srcinfo" — 's','r' followed by 'c' (word), no boundary, no match. Good.
    assert not sr_pat_py.search("architecture design"), "architecture should not match architect"
    # "architecture" contains 'architect' followed by 'u' (word), no boundary, no match. Good.

    # DuckDB equivalents
    jr_pat_sql = r"\b(junior|jr|entry[- ]level|graduate|new[- ]grad|intern)\b"
    sr_pat_sql = r"\b(senior|sr\.?|staff|principal|lead|architect|distinguished)\b"
    return jr_pat_sql, sr_pat_sql


# ----------------------------------------------------------------------------
# Period scope helpers
# ----------------------------------------------------------------------------

SCOPES = {
    "arshkon": "source = 'kaggle_arshkon'",
    "asaniczka": "source = 'kaggle_asaniczka'",
    "pooled_2024": "source IN ('kaggle_arshkon','kaggle_asaniczka')",
    "scraped_2026": "source = 'scraped'",
}


def definition_sql(defn: str, jr_pat: str, sr_pat: str) -> str:
    """Return SQL predicate for a definition label."""
    if defn == "J1":
        return "seniority_final = 'entry'"
    if defn == "J2":
        return "seniority_final IN ('entry','associate')"
    if defn == "J3":
        return "yoe_extracted IS NOT NULL AND yoe_extracted <= 2"
    if defn == "J4":
        return "yoe_extracted IS NOT NULL AND yoe_extracted <= 3"
    if defn == "J5":
        # NOTE: Task spec says `title_normalized`, but that column has level
        # indicators stripped per schema ("senior"/"junior"/"jr" etc. are
        # pre-removed). We use lower(title) to preserve level markers.
        return f"regexp_matches(lower(coalesce(title, '')), '(?i){jr_pat}')"
    if defn == "J6":
        return (
            f"(seniority_final = 'entry' "
            f"OR regexp_matches(lower(coalesce(title, '')), '(?i){jr_pat}'))"
        )
    if defn == "S1":
        return "seniority_final IN ('mid-senior','director')"
    if defn == "S2":
        return "seniority_final = 'director'"
    if defn == "S3":
        # See J5 note: title_normalized would not work — using raw title.
        return f"regexp_matches(lower(coalesce(title, '')), '(?i){sr_pat}')"
    if defn == "S4":
        return "yoe_extracted IS NOT NULL AND yoe_extracted >= 5"
    if defn == "S5":
        return "yoe_extracted IS NOT NULL AND yoe_extracted >= 8"
    raise ValueError(defn)


JUNIOR_DEFS = ["J1", "J2", "J3", "J4", "J5", "J6"]
SENIOR_DEFS = ["S1", "S2", "S3", "S4", "S5"]
ALL_DEFS = JUNIOR_DEFS + SENIOR_DEFS


# ----------------------------------------------------------------------------
# Share computation (n_of_all, n_of_known, shares)
# ----------------------------------------------------------------------------

def known_filter(defn: str) -> str:
    """Known denominator filter appropriate for each definition."""
    if defn in ("J1", "J2", "J6", "S1", "S2"):
        # label-based: known = seniority_final != 'unknown'
        return "seniority_final != 'unknown'"
    if defn in ("J3", "J4", "S4", "S5"):
        # YOE-based: known = yoe_extracted IS NOT NULL
        return "yoe_extracted IS NOT NULL"
    if defn in ("J5", "S3"):
        # title-keyword: the set of rows where we can tell if the title matches
        # is all rows with a non-null title. Since nearly all rows have a title,
        # known ≈ all SWE. We'll use all SWE as the denominator.
        return "TRUE"
    raise ValueError(defn)


def compute_share(scope_sql: str, defn: str, jr_pat: str, sr_pat: str) -> dict:
    d_sql = definition_sql(defn, jr_pat, sr_pat)
    k_sql = known_filter(defn)
    sql = f"""
        SELECT
          SUM(CASE WHEN {d_sql} THEN 1 ELSE 0 END) AS n_match,
          SUM(CASE WHEN {k_sql} THEN 1 ELSE 0 END) AS n_known,
          COUNT(*) AS n_all
        FROM '{DATA}'
        WHERE {BASE_FILTER}
          AND ({scope_sql})
    """
    row = duckdb.sql(sql).fetchone()
    n_match, n_known, n_all = row
    n_match = int(n_match or 0)
    n_known = int(n_known or 0)
    n_all = int(n_all or 0)
    return {
        "n_match": n_match,
        "n_all": n_all,
        "n_known": n_known,
        "share_of_all": (n_match / n_all) if n_all else float("nan"),
        "share_of_known": (n_match / n_known) if n_known else float("nan"),
    }


# ----------------------------------------------------------------------------
# Power / MDE helpers (two-proportion test, 80% power, alpha = 0.05 two-sided)
# ----------------------------------------------------------------------------

def two_prop_mde(n1: int, n2: int, p_baseline: float,
                 power: float = 0.80, alpha: float = 0.05) -> float:
    """Compute the minimum detectable difference in proportions (absolute)."""
    if n1 <= 0 or n2 <= 0 or math.isnan(p_baseline):
        return float("nan")
    if p_baseline <= 0 or p_baseline >= 1:
        # degenerate baseline — use p=0.5 as worst-case
        p_baseline = 0.5
    # For two-proportion z-test, the standard approximation:
    #   delta = (z_{alpha/2} + z_{beta}) * sqrt(p_bar*(1-p_bar)*(1/n1 + 1/n2))
    # at the level of p_baseline. This is an iterative solution if we want
    # MDE exactly; use the non-iterative "average proportion" approx with
    # p_bar = p_baseline (conservative when the treatment p is close to baseline).
    from math import sqrt
    # Normal quantiles via scipy if available, else use hardcoded values
    z_alpha = 1.959963984540054  # alpha/2 = 0.025 two-sided
    z_beta = 0.8416212335729143  # 80% power
    p_bar = p_baseline
    se = sqrt(p_bar * (1 - p_bar) * (1 / n1 + 1 / n2))
    mde = (z_alpha + z_beta) * se
    return mde


# ----------------------------------------------------------------------------
# Panel build
# ----------------------------------------------------------------------------

def build_panel(jr_pat: str, sr_pat: str) -> pd.DataFrame:
    """Long panel: (definition, period, source, side, metrics...)."""
    rows = []
    # Per-scope shares
    scope_results: dict[tuple[str, str], dict] = {}
    for scope_name, scope_sql in SCOPES.items():
        for defn in ALL_DEFS:
            r = compute_share(scope_sql, defn, jr_pat, sr_pat)
            scope_results[(scope_name, defn)] = r

    # Cross-period effect (arshkon-only vs scraped; pooled_2024 vs scraped)
    for defn in ALL_DEFS:
        side = "junior" if defn.startswith("J") else "senior"
        for scope_name in SCOPES:
            r = scope_results[(scope_name, defn)]
            # Effect: arshkon->scraped_2026, pooled->scraped_2026, within_2024
            arsh = scope_results[("arshkon", defn)]
            asan = scope_results[("asaniczka", defn)]
            pool = scope_results[("pooled_2024", defn)]
            scrp = scope_results[("scraped_2026", defn)]

            arsh_share_all = arsh["share_of_all"]
            arsh_share_known = arsh["share_of_known"]
            asan_share_all = asan["share_of_all"]
            asan_share_known = asan["share_of_known"]
            pool_share_all = pool["share_of_all"]
            pool_share_known = pool["share_of_known"]
            scrp_share_all = scrp["share_of_all"]
            scrp_share_known = scrp["share_of_known"]

            within_2024_diff_all = arsh_share_all - asan_share_all
            within_2024_diff_known = arsh_share_known - asan_share_known
            cross_pooled_diff_all = scrp_share_all - pool_share_all
            cross_pooled_diff_known = scrp_share_known - pool_share_known
            cross_arsh_diff_all = scrp_share_all - arsh_share_all
            cross_arsh_diff_known = scrp_share_known - arsh_share_known

            # MDEs
            mde_arsh_vs_scrp_all = two_prop_mde(arsh["n_all"], scrp["n_all"], arsh_share_all)
            mde_arsh_vs_scrp_known = two_prop_mde(arsh["n_known"], scrp["n_known"], arsh_share_known)
            mde_pool_vs_scrp_all = two_prop_mde(pool["n_all"], scrp["n_all"], pool_share_all)
            mde_pool_vs_scrp_known = two_prop_mde(pool["n_known"], scrp["n_known"], pool_share_known)

            # Direction (cross-period pooled vs scraped)
            if math.isnan(cross_pooled_diff_all):
                direction = "n/a"
            elif abs(cross_pooled_diff_all) < 0.001:
                direction = "flat"
            else:
                direction = "up" if cross_pooled_diff_all > 0 else "down"

            rows.append({
                "definition": defn,
                "side": side,
                "scope": scope_name,
                "n_match": r["n_match"],
                "n_all": r["n_all"],
                "n_known": r["n_known"],
                "share_of_all": r["share_of_all"],
                "share_of_known": r["share_of_known"],
                "within_2024_diff_all": within_2024_diff_all,
                "within_2024_diff_known": within_2024_diff_known,
                "cross_pooled_diff_all": cross_pooled_diff_all,
                "cross_pooled_diff_known": cross_pooled_diff_known,
                "cross_arshkon_diff_all": cross_arsh_diff_all,
                "cross_arshkon_diff_known": cross_arsh_diff_known,
                "mde_arshkon_vs_scraped_all": mde_arsh_vs_scrp_all,
                "mde_arshkon_vs_scraped_known": mde_arsh_vs_scrp_known,
                "mde_pooled_vs_scraped_all": mde_pool_vs_scrp_all,
                "mde_pooled_vs_scraped_known": mde_pool_vs_scrp_known,
                "direction": direction,
            })

    panel = pd.DataFrame(rows)
    return panel


# ----------------------------------------------------------------------------
# Overlap matrices
# ----------------------------------------------------------------------------

def compute_overlap_matrix(defs: list[str], jr_pat: str, sr_pat: str,
                           scope_sql: str) -> np.ndarray:
    """|X ∩ Y| / |X| for each (row=X, col=Y). Scope: full SWE rows."""
    # Materialize a lightweight table of booleans for each definition to keep
    # memory bounded.
    selects = ", ".join(
        f"CAST(CASE WHEN {definition_sql(d, jr_pat, sr_pat)} THEN 1 ELSE 0 END AS INTEGER) AS {d}"
        for d in defs
    )
    sql = f"""
        SELECT {selects}
        FROM '{DATA}'
        WHERE {BASE_FILTER}
          AND ({scope_sql})
    """
    df = duckdb.sql(sql).df()
    mat = np.zeros((len(defs), len(defs)), dtype=float)
    for i, di in enumerate(defs):
        n_i = int(df[di].sum())
        for j, dj in enumerate(defs):
            if n_i == 0:
                mat[i, j] = float("nan")
            else:
                both = int((df[di] & df[dj]).sum())
                mat[i, j] = both / n_i
    return mat


def plot_overlap(mat: np.ndarray, defs: list[str], title: str, outfile: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(defs)))
    ax.set_yticks(range(len(defs)))
    ax.set_xticklabels(defs)
    ax.set_yticklabels(defs)
    ax.set_xlabel("Y")
    ax.set_ylabel("X")
    ax.set_title(f"{title}\n|X ∩ Y| / |X| (row-normalized)")
    for i in range(len(defs)):
        for j in range(len(defs)):
            v = mat[i, j]
            if np.isnan(v):
                s = "—"
            else:
                s = f"{v:.2f}"
            color = "white" if (not np.isnan(v) and v > 0.55) else "black"
            ax.text(j, i, s, ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Title-keyword content spot-check (step 6)
# ----------------------------------------------------------------------------

def spot_check_titles(jr_pat: str, sr_pat: str):
    print("\n=== Spot check: J5-only (matches J5 but not J1) ===")
    j1 = definition_sql("J1", jr_pat, sr_pat)
    j5 = definition_sql("J5", jr_pat, sr_pat)
    sql = f"""
        WITH pool AS (
          SELECT uid, source, title, seniority_final, yoe_extracted,
                 substr(coalesce(description, ''), 1, 200) desc_preview
          FROM '{DATA}'
          WHERE {BASE_FILTER}
            AND ({j5})
            AND NOT ({j1})
        )
        SELECT * FROM pool USING SAMPLE 50
    """
    df_j = duckdb.sql(sql).df()
    df_j.to_csv(TAB_DIR / "spot_j5_only.csv", index=False)
    print(f"Sampled {len(df_j)} rows matched ONLY by J5. Saved to {TAB_DIR / 'spot_j5_only.csv'}.")

    print("\n=== Spot check: S3-only (matches S3 but not S1) ===")
    s1 = definition_sql("S1", jr_pat, sr_pat)
    s3 = definition_sql("S3", jr_pat, sr_pat)
    sql = f"""
        WITH pool AS (
          SELECT uid, source, title, seniority_final, yoe_extracted,
                 substr(coalesce(description, ''), 1, 200) desc_preview
          FROM '{DATA}'
          WHERE {BASE_FILTER}
            AND ({s3})
            AND NOT ({s1})
        )
        SELECT * FROM pool USING SAMPLE 50
    """
    df_s = duckdb.sql(sql).df()
    df_s.to_csv(TAB_DIR / "spot_s3_only.csv", index=False)
    print(f"Sampled {len(df_s)} rows matched ONLY by S3. Saved to {TAB_DIR / 'spot_s3_only.csv'}.")

    # Lightweight auto-scoring: for J5-only, does the title contain 'senior'/'staff'/'principal'?
    # That would be a false positive on J5 (e.g., "Senior Architect, Principal Engineer" is S-side).
    def is_likely_fp_jr(row) -> bool:
        t = (row["title"] or "").lower()
        if re.search(r"\b(senior|sr\.?|staff|principal|lead|architect|distinguished)\b", t):
            return True
        # Ghost ping: if seniority_final is mid-senior or director, J5 is suspicious
        if row["seniority_final"] in ("mid-senior", "director"):
            return True
        return False

    def is_likely_fp_sr(row) -> bool:
        t = (row["title"] or "").lower()
        if re.search(r"\b(junior|jr|entry[- ]level|graduate|new[- ]grad|intern)\b", t):
            return True
        if row["seniority_final"] == "entry":
            return True
        return False

    # Guard against empty frames (pandas .apply on empty returns a DataFrame)
    df_j["auto_fp_flag"] = df_j.apply(is_likely_fp_jr, axis=1) if len(df_j) else False
    df_s["auto_fp_flag"] = df_s.apply(is_likely_fp_sr, axis=1) if len(df_s) else False
    fp_j = int(df_j["auto_fp_flag"].sum()) if len(df_j) else 0
    fp_s = int(df_s["auto_fp_flag"].sum()) if len(df_s) else 0
    print(f"\nJ5-only: auto-flagged false positives = {fp_j} / {len(df_j)}")
    print(f"S3-only: auto-flagged false positives = {fp_s} / {len(df_s)}")

    return df_j, df_s, fp_j, fp_s


# ----------------------------------------------------------------------------
# Sensitivities (aggregator exclusion)
# ----------------------------------------------------------------------------

def sensitivities(jr_pat: str, sr_pat: str) -> pd.DataFrame:
    rows = []
    for agg_filter in [("all", "TRUE"), ("no_aggregator", "is_aggregator = false")]:
        label, pred = agg_filter
        for scope_name, scope_sql in SCOPES.items():
            for defn in ALL_DEFS:
                d_sql = definition_sql(defn, jr_pat, sr_pat)
                k_sql = known_filter(defn)
                sql = f"""
                    SELECT
                      SUM(CASE WHEN {d_sql} THEN 1 ELSE 0 END) AS n_match,
                      SUM(CASE WHEN {k_sql} THEN 1 ELSE 0 END) AS n_known,
                      COUNT(*) AS n_all
                    FROM '{DATA}'
                    WHERE {BASE_FILTER}
                      AND ({scope_sql})
                      AND ({pred})
                """
                n_match, n_known, n_all = duckdb.sql(sql).fetchone()
                n_match = int(n_match or 0)
                n_known = int(n_known or 0)
                n_all = int(n_all or 0)
                rows.append({
                    "aggregator_treatment": label,
                    "scope": scope_name,
                    "definition": defn,
                    "side": "junior" if defn.startswith("J") else "senior",
                    "n_match": n_match,
                    "n_all": n_all,
                    "n_known": n_known,
                    "share_of_all": (n_match / n_all) if n_all else float("nan"),
                    "share_of_known": (n_match / n_known) if n_known else float("nan"),
                })
    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "sensitivity_aggregator.csv", index=False)
    return df


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    jr_pat, sr_pat = test_regex_patterns()
    print(f"Regex patterns OK. jr={jr_pat}\nsr={sr_pat}")

    # Build the long panel
    panel = build_panel(jr_pat, sr_pat)
    # Save the full long panel
    panel.to_csv(TAB_DIR / "panel_long.csv", index=False)

    # Also produce the compact required schema artifact with:
    # definition | side | n_of_all | n_of_known | share_of_all | share_of_known |
    #   mde_arshkon_vs_scraped | mde_pooled_vs_scraped | within_2024_effect |
    #   cross_period_effect | direction
    # with one row per (definition, scope). We use the "of_all" MDE as the
    # canonical value because the known-denominator varies in composition.
    compact = panel.rename(columns={
        "n_all": "n_of_all",
        "n_known": "n_of_known",
        "within_2024_diff_all": "within_2024_effect",
        "cross_pooled_diff_all": "cross_period_effect",
        "mde_arshkon_vs_scraped_all": "mde_arshkon_vs_scraped",
        "mde_pooled_vs_scraped_all": "mde_pooled_vs_scraped",
    })[[
        "definition", "side", "scope", "n_of_all", "n_of_known",
        "share_of_all", "share_of_known",
        "mde_arshkon_vs_scraped", "mde_pooled_vs_scraped",
        "within_2024_effect", "cross_period_effect", "direction",
        # include "known" versions as extra columns (not part of required schema)
        "within_2024_diff_known", "cross_pooled_diff_known",
        "mde_arshkon_vs_scraped_known", "mde_pooled_vs_scraped_known",
    ]]
    # rename scope -> period label for the panel CSV
    compact = compact.rename(columns={"scope": "period_source"})

    out_panel = SHARED / "seniority_definition_panel.csv"
    compact.to_csv(out_panel, index=False)
    print(f"\nWrote canonical panel: {out_panel}")

    # Overlap matrices on SWE pooled corpus (all sources)
    full_scope_sql = "TRUE"  # all SWE passing base filter
    jr_mat = compute_overlap_matrix(JUNIOR_DEFS, jr_pat, sr_pat, full_scope_sql)
    sr_mat = compute_overlap_matrix(SENIOR_DEFS, jr_pat, sr_pat, full_scope_sql)
    pd.DataFrame(jr_mat, index=JUNIOR_DEFS, columns=JUNIOR_DEFS).to_csv(TAB_DIR / "overlap_junior.csv")
    pd.DataFrame(sr_mat, index=SENIOR_DEFS, columns=SENIOR_DEFS).to_csv(TAB_DIR / "overlap_senior.csv")
    plot_overlap(jr_mat, JUNIOR_DEFS, "Junior definition overlap (all SWE)",
                 FIG_DIR / "overlap_junior.png")
    plot_overlap(sr_mat, SENIOR_DEFS, "Senior definition overlap (all SWE)",
                 FIG_DIR / "overlap_senior.png")
    # Combined overlap figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, defs, title in [
        (axs[0], jr_mat, JUNIOR_DEFS, "Junior"),
        (axs[1], sr_mat, SENIOR_DEFS, "Senior"),
    ]:
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(defs)))
        ax.set_yticks(range(len(defs)))
        ax.set_xticklabels(defs)
        ax.set_yticklabels(defs)
        ax.set_title(title)
        for i in range(len(defs)):
            for j in range(len(defs)):
                v = mat[i, j]
                s = "—" if np.isnan(v) else f"{v:.2f}"
                color = "white" if (not np.isnan(v) and v > 0.55) else "black"
                ax.text(j, i, s, ha="center", va="center", color=color, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Seniority definition overlap |X ∩ Y| / |X|")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "overlap_heatmap.png", dpi=150)
    plt.close(fig)

    # Spot check
    spot_check_titles(jr_pat, sr_pat)

    # Sensitivity: aggregator exclusion
    sensitivities(jr_pat, sr_pat)

    # Direction consistency summary (step 5)
    dir_summary = compact[compact["period_source"] == "scraped_2026"][
        ["definition", "side", "share_of_all", "cross_period_effect", "direction"]
    ].copy()
    dir_summary.to_csv(TAB_DIR / "direction_consistency.csv", index=False)
    print("\nDirection consistency (pooled 2024 -> scraped 2026, of_all denominator):")
    print(dir_summary.to_string())

    print("\nT30 artifacts written:")
    print(f"  {SHARED / 'seniority_definition_panel.csv'}")
    print(f"  {TAB_DIR}")
    print(f"  {FIG_DIR}")


if __name__ == "__main__":
    main()
