"""T30. Seniority definition ablation panel.

Builds the canonical junior- and senior-side operationalization panel that every
downstream seniority-stratified task consumes.

Produces:
  - exploration/artifacts/shared/seniority_definition_panel.csv   (CANONICAL DOWNSTREAM ARTIFACT)
  - exploration/tables/T30/*.csv  (supporting tables)

Definitions:
  Junior: J1 entry, J2 entry+associate, J3 YOE_llm<=2 [primary], J4 YOE_llm<=3,
          J5 title-keyword junior, J6 = J1 ∪ J5, J3_rule YOE_extracted<=2
  Senior: S1 mid-senior+director, S2 director, S3 title-keyword senior,
          S4 YOE_llm>=5 [primary], S5 YOE_llm>=8, S4_rule YOE_extracted>=5

Analysis groups (period x source):
  2024-01 kaggle_asaniczka
  2024-04 kaggle_arshkon
  pooled-2024 (asaniczka + arshkon)
  2026-03 scraped
  2026-04 scraped
  pooled-2026 (scraped 03+04)
"""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pandas as pd
import numpy as np

PARQUET = "data/unified.parquet"
OUT_TABLES = Path("exploration/tables/T30")
OUT_SHARED = Path("exploration/artifacts/shared")
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_SHARED.mkdir(parents=True, exist_ok=True)

DEFAULT_FILTER = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'"

# Regex patterns (RE2-compatible for DuckDB, (?i) for case-insensitive)
JUNIOR_RE = r"(?i)\b(junior|jr|entry[- ]level|graduate|new[- ]grad|intern)\b"
SENIOR_RE = r"(?i)\b(senior|sr\.?|staff|principal|lead|architect|distinguished)\b"


def q(sql: str) -> pd.DataFrame:
    return duckdb.sql(sql).df()


def save(df: pd.DataFrame, name: str, dir_: Path = OUT_TABLES) -> None:
    path = dir_ / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  -> {path}  ({len(df)} rows)")


# --------------------------------------------------------------------------- #
# 1. Unit tests for regexes (inline asserts before any analysis)              #
# --------------------------------------------------------------------------- #

def _test_regexes():
    """Regex correctness smoke tests. Fails loudly if any behavior drifts."""
    import re
    j = re.compile(r"\b(junior|jr|entry[- ]level|graduate|new[- ]grad|intern)\b", re.IGNORECASE)
    s = re.compile(r"\b(senior|sr\.?|staff|principal|lead|architect|distinguished)\b", re.IGNORECASE)
    # Junior positive
    for t in ["Junior SWE", "Jr. SWE", "Jr SWE", "Entry Level SWE", "Entry-Level SWE",
              "Graduate SWE", "New-Grad SWE", "New Grad SWE", "Software Engineering Intern"]:
        assert j.search(t), f"J regex missed: {t}"
    # Junior negative
    for t in ["Senior SWE", "Principal Engineer", "Interns Supervisor",
              "International Engineer", "Interim Engineer", "Software Engineer"]:
        assert not j.search(t), f"J regex false-positive: {t}"
    # Senior positive
    for t in ["Senior SWE", "Sr. SWE", "Sr SWE", "Staff SWE", "Principal Engineer",
              "Tech Lead", "Software Architect", "Distinguished Engineer", "Lead SWE"]:
        assert s.search(t), f"S regex missed: {t}"
    # Senior negative
    for t in ["Junior SWE", "Associate SWE", "Leadership Development",
              "Principle Engineer", "Leader Software Role", "Seniors Engineer",
              "Software Engineer"]:
        assert not s.search(t), f"S regex false-positive: {t}"

    # DuckDB regex check
    df = q(rf"""
      WITH titles(t) AS (VALUES
        ('Junior SWE'), ('Senior SWE'), ('Staff Engineer'),
        ('Principal Engineer'), ('Tech Lead'), ('Software Architect'),
        ('New Grad SWE'), ('Leadership Development'),
        ('International Engineer'))
      SELECT t,
        regexp_matches(t, '{JUNIOR_RE}') AS j,
        regexp_matches(t, '{SENIOR_RE}') AS s
      FROM titles
    """)
    expected = {
        "Junior SWE": (True, False),
        "Senior SWE": (False, True),
        "Staff Engineer": (False, True),
        "Principal Engineer": (False, True),
        "Tech Lead": (False, True),
        "Software Architect": (False, True),
        "New Grad SWE": (True, False),
        "Leadership Development": (False, False),
        "International Engineer": (False, False),
    }
    for _, r in df.iterrows():
        assert (bool(r["j"]), bool(r["s"])) == expected[r["t"]], \
            f"DuckDB regex mismatch on {r['t']}: got ({r['j']},{r['s']})"


# --------------------------------------------------------------------------- #
# 2. Definition SQL clauses                                                   #
# --------------------------------------------------------------------------- #

# Each returns a SQL boolean expression usable in WHERE or CASE.
DEFINITIONS = {
    # Junior
    "J1": ("junior", "label",       "seniority_final = 'entry'"),
    "J2": ("junior", "label",       "seniority_final IN ('entry','associate')"),
    "J3": ("junior", "yoe_llm",     "yoe_min_years_llm <= 2"),
    "J4": ("junior", "yoe_llm",     "yoe_min_years_llm <= 3"),
    "J5": ("junior", "title_keyword", f"regexp_matches(title, '{JUNIOR_RE}')"),
    "J6": ("junior", "label",       f"seniority_final = 'entry' OR regexp_matches(title, '{JUNIOR_RE}')"),
    "J3_rule": ("junior", "yoe_rule", "yoe_extracted <= 2"),
    # Senior
    "S1": ("senior", "label",       "seniority_final IN ('mid-senior','director')"),
    "S2": ("senior", "label",       "seniority_final = 'director'"),
    "S3": ("senior", "title_keyword", f"regexp_matches(title, '{SENIOR_RE}')"),
    "S4": ("senior", "yoe_llm",     "yoe_min_years_llm >= 5"),
    "S5": ("senior", "yoe_llm",     "yoe_min_years_llm >= 8"),
    "S4_rule": ("senior", "yoe_rule", "yoe_extracted >= 5"),
}

# Denominator for each family
DENOM_PREDICATE = {
    "label":         "TRUE",  # all SWE rows (known+unknown) — report both
    "title_keyword": "TRUE",
    "yoe_llm":       "yoe_min_years_llm IS NOT NULL AND llm_classification_coverage = 'labeled'",
    "yoe_rule":      "yoe_extracted IS NOT NULL",
}

# --------------------------------------------------------------------------- #
# 3. Analysis groups                                                          #
# --------------------------------------------------------------------------- #

GROUPS = {
    "2024-01_asaniczka": "source = 'kaggle_asaniczka' AND period = '2024-01'",
    "2024-04_arshkon":   "source = 'kaggle_arshkon' AND period = '2024-04'",
    "pooled-2024":       "(source = 'kaggle_asaniczka' OR source = 'kaggle_arshkon')",
    "2026-03_scraped":   "source = 'scraped' AND period = '2026-03'",
    "2026-04_scraped":   "source = 'scraped' AND period = '2026-04'",
    "pooled-2026":       "source = 'scraped' AND source_platform = 'linkedin'",
}


def base_where() -> str:
    return f"is_swe AND {DEFAULT_FILTER}"


# --------------------------------------------------------------------------- #
# 4. Count tables                                                             #
# --------------------------------------------------------------------------- #

def compute_counts_table() -> pd.DataFrame:
    """For each (definition, group): n_all, n_denom, n_in_def, share_of_all, share_of_denom."""
    rows = []
    for group, gpred in GROUPS.items():
        for defn, (side, family, pred) in DEFINITIONS.items():
            denom_pred = DENOM_PREDICATE[family]
            df = q(f"""
              SELECT
                count(*) AS n_all,
                sum(CASE WHEN ({denom_pred}) THEN 1 ELSE 0 END) AS n_denominator,
                sum(CASE WHEN ({pred}) THEN 1 ELSE 0 END) AS n_in_def_all,
                sum(CASE WHEN ({pred}) AND ({denom_pred}) THEN 1 ELSE 0 END) AS n_in_def_denom
              FROM '{PARQUET}'
              WHERE {base_where()} AND {gpred}
            """)
            r = df.iloc[0]
            n_all = int(r["n_all"])
            n_denom = int(r["n_denominator"])
            n_def_all = int(r["n_in_def_all"])
            n_def_denom = int(r["n_in_def_denom"])
            rows.append({
                "definition": defn,
                "side": side,
                "family": family,
                "group": group,
                "n_all": n_all,
                "n_denominator": n_denom,
                "n_in_def_all": n_def_all,
                "n_in_def_denom": n_def_denom,
                "share_of_all": n_def_all / n_all if n_all > 0 else float("nan"),
                "share_of_denom": n_def_denom / n_denom if n_denom > 0 else float("nan"),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. Overlap matrices                                                         #
# --------------------------------------------------------------------------- #

def compute_overlap_matrix(defs: list, group_pred: str) -> pd.DataFrame:
    """For a list of definitions, compute |X ∩ Y| / |X| pairwise on the given group."""
    # For each definition, compute set of uids (as ROWID proxy — but we use uid)
    n_rows = len(defs)
    mat = np.full((n_rows, n_rows), float("nan"))
    card = {}
    inter = {}
    for i, d in enumerate(defs):
        _, _, pred_i = DEFINITIONS[d]
        ni = q(f"""
          SELECT count(*) n FROM '{PARQUET}'
          WHERE {base_where()} AND {group_pred} AND ({pred_i})
        """).iloc[0]["n"]
        card[d] = int(ni)
    for i, d1 in enumerate(defs):
        _, _, p1 = DEFINITIONS[d1]
        for j, d2 in enumerate(defs):
            _, _, p2 = DEFINITIONS[d2]
            n_inter = q(f"""
              SELECT count(*) n FROM '{PARQUET}'
              WHERE {base_where()} AND {group_pred} AND ({p1}) AND ({p2})
            """).iloc[0]["n"]
            inter[(d1, d2)] = int(n_inter)
            if card[d1] > 0:
                mat[i, j] = n_inter / card[d1]
    df = pd.DataFrame(mat, index=defs, columns=defs)
    return df


# --------------------------------------------------------------------------- #
# 6. MDE calculator                                                           #
# --------------------------------------------------------------------------- #

def mde_binary(p: float, n1: int, n2: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Minimum detectable effect (absolute difference in proportions) for a two-sample
    test at given alpha/power, assuming baseline prevalence p in group 1."""
    if n1 <= 1 or n2 <= 1 or p is None or math.isnan(p):
        return float("nan")
    # Two-sided z-alpha/2, z-beta
    from scipy.stats import norm
    za = norm.ppf(1 - alpha / 2)
    zb = norm.ppf(power)
    # Approximation with pooled variance at baseline; iterate a couple steps for p2
    # This is the standard formula for MDE given sample sizes and baseline.
    # SE(p1 - p2) ≈ sqrt(p*(1-p)*(1/n1 + 1/n2)) under null; under alt use (p1(1-p1)/n1 + p2(1-p2)/n2).
    # For MDE, we solve for delta such that (za + zb) * SE_alt = delta with SE_alt using p and p+delta.
    # Simplest closed-form: delta ≈ (za + zb) * sqrt( p*(1-p) * (1/n1 + 1/n2) * 2 ) ... use iterative:
    delta = (za + zb) * math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    for _ in range(5):
        p2 = max(min(p + delta, 0.9999), 0.0001)
        se = math.sqrt(p * (1 - p) / n1 + p2 * (1 - p2) / n2)
        delta = (za + zb) * se
    return float(delta)


# --------------------------------------------------------------------------- #
# 7. Panel assembly                                                           #
# --------------------------------------------------------------------------- #

def compute_group_size(group_pred: str, family: str) -> int:
    """Denominator size for this group under this family (for MDE)."""
    denom = DENOM_PREDICATE[family]
    df = q(f"""
      SELECT count(*) n FROM '{PARQUET}'
      WHERE {base_where()} AND {group_pred} AND {denom}
    """)
    return int(df.iloc[0]["n"])


def assemble_panel(counts: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical panel per the schema in the prompt.

    One row per (definition × period × source).
    Columns: definition | side | family | period | source | n_of_all | n_of_denominator |
             share_of_all | share_of_denominator | mde_arshkon_vs_scraped | mde_pooled_vs_scraped |
             within_2024_effect | cross_period_effect | direction
    """
    # period × source decomposition
    # The 6 groups map to (period, source) pairs.
    group_meta = {
        "2024-01_asaniczka": ("2024-01", "kaggle_asaniczka"),
        "2024-04_arshkon":   ("2024-04", "kaggle_arshkon"),
        "pooled-2024":       ("pooled-2024", "kaggle_asaniczka+arshkon"),
        "2026-03_scraped":   ("2026-03", "scraped"),
        "2026-04_scraped":   ("2026-04", "scraped"),
        "pooled-2026":       ("pooled-2026", "scraped"),
    }

    # Build a dict: (defn, group) -> (share_of_all, share_of_denom, n_all, n_denom)
    by_key = {}
    for _, r in counts.iterrows():
        by_key[(r["definition"], r["group"])] = r

    # Also need n for arshkon and pooled-2024 and pooled-2026 for MDE per family
    family_group_n = {}
    for g, gpred in GROUPS.items():
        for fam in {"label", "title_keyword", "yoe_llm", "yoe_rule"}:
            family_group_n[(fam, g)] = compute_group_size(gpred, fam)

    panel_rows = []
    for defn, (side, family, _pred) in DEFINITIONS.items():
        # For cross-period direction calculation:
        #   use share_of_denom pooled-2024 vs pooled-2026 (for label/yoe variants)
        #   or share_of_all for title_keyword since denom = all
        def _get_share(group, want="share_of_denom"):
            k = (defn, group)
            if k not in by_key:
                return float("nan")
            return float(by_key[k][want])

        share_col = "share_of_denom"
        within_2024 = _get_share("2024-04_arshkon", share_col) - _get_share("2024-01_asaniczka", share_col)
        cross_period = _get_share("pooled-2026", share_col) - _get_share("pooled-2024", share_col)

        # MDE: arshkon vs pooled-2026; pooled-2024 vs pooled-2026
        n_arshkon = family_group_n[(family, "2024-04_arshkon")]
        n_pooled_2024 = family_group_n[(family, "pooled-2024")]
        n_pooled_2026 = family_group_n[(family, "pooled-2026")]
        p_pooled_2024 = _get_share("pooled-2024", share_col)
        p_arshkon = _get_share("2024-04_arshkon", share_col)
        mde_arsh = mde_binary(p_arshkon, n_arshkon, n_pooled_2026) if not math.isnan(p_arshkon) else float("nan")
        mde_pooled = mde_binary(p_pooled_2024, n_pooled_2024, n_pooled_2026) if not math.isnan(p_pooled_2024) else float("nan")

        direction = (
            "up" if cross_period > 0.005 else
            "down" if cross_period < -0.005 else
            "flat"
        ) if not math.isnan(cross_period) else None

        # Emit one row per (group) — each group gives a (period, source) tuple
        for g, (period, src) in group_meta.items():
            k = (defn, g)
            if k not in by_key:
                continue
            r = by_key[k]
            panel_rows.append({
                "definition": defn,
                "side": side,
                "family": family,
                "period": period,
                "source": src,
                "n_of_all": int(r["n_all"]),
                "n_of_denominator": int(r["n_denominator"]),
                "share_of_all": float(r["share_of_all"]),
                "share_of_denominator": float(r["share_of_denom"]),
                "mde_arshkon_vs_scraped": mde_arsh,
                "mde_pooled_vs_scraped": mde_pooled,
                "within_2024_effect": within_2024 if g not in {"2024-01_asaniczka", "2024-04_arshkon"} else float("nan"),
                "cross_period_effect": cross_period if g.startswith("pooled") else float("nan"),
                # direction applies only to cross-period rows per spec; null otherwise
                "direction": direction if g.startswith("pooled-2026") else None,
            })
    return pd.DataFrame(panel_rows)


# --------------------------------------------------------------------------- #
# 8. Verification audits                                                      #
# --------------------------------------------------------------------------- #

def audit_llm_vs_rule_yoe() -> None:
    """LLM-vs-rule YOE exact-agreement rate and MAE by source."""
    print("\n[Audit] LLM-vs-rule YOE agreement by source.")
    df = q(f"""
      SELECT source,
        count(*) AS n_both,
        sum(CASE WHEN yoe_min_years_llm = yoe_extracted THEN 1 ELSE 0 END) AS n_exact,
        avg(abs(yoe_min_years_llm - yoe_extracted)) AS mae,
        avg(CASE WHEN yoe_min_years_llm - yoe_extracted >= 3 OR yoe_min_years_llm - yoe_extracted <= -3 THEN 1.0 ELSE 0.0 END) AS pct_disagree3plus
      FROM '{PARQUET}'
      WHERE {base_where()}
        AND yoe_min_years_llm IS NOT NULL
        AND yoe_extracted IS NOT NULL
      GROUP BY source
      ORDER BY source
    """)
    df["exact_agreement_share"] = df["n_exact"] / df["n_both"]
    save(df, "audit_llm_vs_rule_yoe_by_source")

    # Sample 20 rows with |diff| >= 3
    samp = q(rf"""
      SELECT * FROM (
        SELECT uid, source, title, yoe_min_years_llm, yoe_extracted,
               substr(description, 1, 300) desc_preview
        FROM '{PARQUET}'
        WHERE {base_where()}
          AND yoe_min_years_llm IS NOT NULL
          AND yoe_extracted IS NOT NULL
          AND abs(yoe_min_years_llm - yoe_extracted) >= 3
      ) USING SAMPLE 20
    """)
    save(samp, "audit_llm_vs_rule_yoe_sample_diff3plus")


def audit_yoe_zero() -> None:
    """Stratified sample of 20-30 rows where yoe_min_years_llm=0."""
    print("\n[Audit] yoe_min_years_llm=0 audit (stratified across sources).")
    df = q(rf"""
      SELECT source, is_swe, count(*) n
      FROM '{PARQUET}'
      WHERE {base_where()}
        AND yoe_min_years_llm = 0
        AND llm_classification_coverage = 'labeled'
      GROUP BY source, is_swe
      ORDER BY source, is_swe
    """)
    save(df, "audit_yoe_zero_counts")

    # Sample ~8 per source (SWE only)
    samples = []
    for source in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]:
        s = q(rf"""
          SELECT * FROM (
            SELECT uid, source, title, seniority_final, seniority_final_source,
                   yoe_min_years_llm, yoe_extracted,
                   substr(description, 1, 400) desc_preview
            FROM '{PARQUET}'
            WHERE is_swe AND {DEFAULT_FILTER}
              AND source = '{source}'
              AND yoe_min_years_llm = 0
              AND llm_classification_coverage = 'labeled'
          ) USING SAMPLE 10
        """)
        samples.append(s)
    if samples:
        out = pd.concat(samples, ignore_index=True)
        save(out, "audit_yoe_zero_sample")


def audit_asaniczka_senior_asymmetry() -> None:
    """Compute S4 and S1 senior shares on asaniczka SWE (LLM-labeled, 2024) vs arshkon SWE (LLM-labeled, 2024)."""
    print("\n[Audit] Asaniczka senior-asymmetry test (S1 and S4).")
    rows = []
    for defn in ["S1", "S4", "S2", "S5", "S4_rule"]:
        _, family, pred = DEFINITIONS[defn]
        denom = DENOM_PREDICATE[family]
        for src in ["kaggle_arshkon", "kaggle_asaniczka"]:
            df = q(f"""
              SELECT count(*) n_denom,
                     sum(CASE WHEN ({pred}) THEN 1 ELSE 0 END) n_hit
              FROM '{PARQUET}'
              WHERE {base_where()} AND source = '{src}' AND {denom}
            """)
            r = df.iloc[0]
            n_hit = int(r["n_hit"]) if r["n_hit"] is not None else 0
            n_denom = int(r["n_denom"])
            share = n_hit / n_denom if n_denom > 0 else float("nan")
            rows.append({
                "definition": defn,
                "family": family,
                "source": src,
                "n_denom": n_denom,
                "n_hit": n_hit,
                "share": share,
            })
    out = pd.DataFrame(rows)
    # Add a pivot for readability
    save(out, "audit_asaniczka_senior_asymmetry")


def main():
    print("=" * 60)
    print("T30. Seniority definition ablation panel")
    print("=" * 60)

    print("\n[Step 0] Regex unit tests.")
    _test_regexes()
    print("  regex tests passed.")

    print("\n[Step 1] Counts per (definition × group).")
    counts = compute_counts_table()
    save(counts, "counts_raw")

    print("\n[Step 2] Assemble canonical panel.")
    panel = assemble_panel(counts)
    save(panel, "seniority_definition_panel_expanded")  # full expanded diagnostic
    # Also save the canonical artifact with exactly the schema specified in the prompt
    canonical_cols = [
        "definition", "side", "family", "period", "source",
        "n_of_all", "n_of_denominator", "share_of_all", "share_of_denominator",
        "mde_arshkon_vs_scraped", "mde_pooled_vs_scraped",
        "within_2024_effect", "cross_period_effect", "direction",
    ]
    canon = panel[canonical_cols].copy()
    # Normalize types: integers for n cols, floats for share/mde/effect, direction string-or-null
    canon["n_of_all"] = canon["n_of_all"].astype("int64")
    canon["n_of_denominator"] = canon["n_of_denominator"].astype("int64")
    for c in ["share_of_all", "share_of_denominator", "mde_arshkon_vs_scraped",
              "mde_pooled_vs_scraped", "within_2024_effect", "cross_period_effect"]:
        canon[c] = canon[c].astype("float64")
    canonical_path = OUT_SHARED / "seniority_definition_panel.csv"
    canon.to_csv(canonical_path, index=False)
    print(f"  -> CANONICAL artifact: {canonical_path}  ({len(canon)} rows)")

    print("\n[Step 3] Overlap matrices.")
    junior_defs = ["J1", "J2", "J3", "J4", "J5", "J6", "J3_rule"]
    senior_defs = ["S1", "S2", "S3", "S4", "S5", "S4_rule"]
    for g in ["pooled-2024", "pooled-2026"]:
        gp = GROUPS[g]
        m = compute_overlap_matrix(junior_defs, gp)
        m.to_csv(OUT_TABLES / f"overlap_junior_{g}.csv")
        print(f"  -> overlap_junior_{g}.csv")
        m = compute_overlap_matrix(senior_defs, gp)
        m.to_csv(OUT_TABLES / f"overlap_senior_{g}.csv")
        print(f"  -> overlap_senior_{g}.csv")

    print("\n[Step 4] Verification audits.")
    audit_llm_vs_rule_yoe()
    audit_yoe_zero()
    audit_asaniczka_senior_asymmetry()

    print("\n[T30] complete.")


if __name__ == "__main__":
    main()
