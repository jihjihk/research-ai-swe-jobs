"""T03 — Seniority label audit.

Audits `seniority_final` against available diagnostics:
  1) seniority_final_source profile by source & period (SWE rows).
  2) Rule-vs-LLM spot check: 100 LLM-labeled rows with weak-marker titles (I/II/III).
  3) seniority_final vs seniority_native cross-tab + Cohen's kappa (arshkon, scraped).
  4) Native-label YOE diagnostic.
  5) Defensibility verdict.

Outputs:
  exploration/tables/T03/*.csv
  exploration/figures/T03/*.png (if useful)
  exploration/reports/T03.md  (assembled separately)
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data/unified.parquet"
TAB_DIR = ROOT / "exploration/tables/T03"
FIG_DIR = ROOT / "exploration/figures/T03"
TAB_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

FILTER = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe = true"


def q(sql: str) -> pd.DataFrame:
    return duckdb.sql(sql).df()


# -----------------------------------------------------------------------------
# Step 1: seniority_final_source profile by source + by period
# -----------------------------------------------------------------------------

def step_1_source_profile() -> pd.DataFrame:
    print("\n=== STEP 1: seniority_final_source profile ===")
    by_source = q(f"""
        SELECT source,
               seniority_final_source,
               COUNT(*) n,
               100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY source) pct_of_source
        FROM '{DATA}'
        WHERE {FILTER}
        GROUP BY source, seniority_final_source
        ORDER BY source, n DESC
    """)
    by_source.to_csv(TAB_DIR / "source_profile_by_source.csv", index=False)
    print(by_source.to_string())

    by_period = q(f"""
        SELECT period,
               seniority_final_source,
               COUNT(*) n,
               100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY period) pct_of_period
        FROM '{DATA}'
        WHERE {FILTER}
        GROUP BY period, seniority_final_source
        ORDER BY period, n DESC
    """)
    by_period.to_csv(TAB_DIR / "source_profile_by_period.csv", index=False)
    print("\nBy period:")
    print(by_period.to_string())

    # By source x seniority_final (to see how much of each final label came from rule vs LLM)
    prov = q(f"""
        SELECT source,
               seniority_final,
               seniority_final_source,
               COUNT(*) n
        FROM '{DATA}'
        WHERE {FILTER}
        GROUP BY source, seniority_final, seniority_final_source
        ORDER BY source, seniority_final, n DESC
    """)
    prov.to_csv(TAB_DIR / "source_profile_by_source_x_final.csv", index=False)
    return by_source


# -----------------------------------------------------------------------------
# Step 2: rule-vs-LLM internal agreement (weak-marker titles)
# -----------------------------------------------------------------------------

def step_2_rule_vs_llm():
    print("\n=== STEP 2: LLM-on-weak-marker spot check ===")
    # Compile regex with assertions to catch edge cases before applying on the DB
    level_re = re.compile(r"\b(?:i{1,3}|iv|v)\b", re.IGNORECASE)
    # Test assertions for the level-code extractor (word-bounded Roman numerals)
    assert level_re.search("engineer ii"), "should match 'engineer ii'"
    assert level_re.search("Engineer III"), "should match 'Engineer III'"
    assert level_re.search("Software Engineer IV"), "should match 'IV'"
    assert not level_re.search("engineer iteration"), "should NOT match 'iteration'"
    assert not level_re.search("engineervi"), "should NOT match when no word boundary"
    # Avoid spurious single 'i' — we do a post-filter later
    # DuckDB regex is POSIX-like; use REGEXP_MATCHES with '(?i)' inline flag.
    # We search for II/III/IV/V with explicit word boundaries; skipping the 'I' alone
    # case because it has high noise.
    rows = q(f"""
        WITH matched AS (
          SELECT uid, source, period, title, seniority_final, seniority_final_source,
                 yoe_extracted, seniority_native
          FROM '{DATA}'
          WHERE {FILTER}
            AND seniority_final_source = 'llm'
            AND regexp_matches(title, '(?i)\\b(II|III|IV)\\b')
        )
        SELECT * FROM matched USING SAMPLE 100
    """)
    rows.to_csv(TAB_DIR / "step2_llm_weak_marker_sample.csv", index=False)
    print(f"Sampled {len(rows)} LLM-labeled rows whose title contains II/III/IV.")

    # Quick auto-check: for titles with explicit II/III, does the LLM return an
    # identifiable seniority that is not 'unknown'?
    def expected_from_roman(title: str) -> str:
        t = title.lower()
        if re.search(r"\b(iv|v)\b", t):
            return "senior-ish"
        if re.search(r"\biii\b", t):
            return "mid-senior"
        if re.search(r"\bii\b", t):
            return "mid"
        return "unknown"

    rows["expected_level"] = rows["title"].fillna("").map(expected_from_roman)
    # Compare LLM's seniority_final against a broad expectation:
    #   II  -> entry/associate plausible, not director
    #   III -> associate/mid-senior plausible, not entry
    #   IV/V -> mid-senior/director plausible, not entry
    def agreement(row) -> str:
        exp = row["expected_level"]
        f = row["seniority_final"]
        if f == "unknown":
            return "unknown"
        if exp == "senior-ish":
            return "agree" if f in ("mid-senior", "director") else "disagree"
        if exp == "mid-senior":
            return "agree" if f in ("associate", "mid-senior") else "disagree"
        if exp == "mid":
            return "agree" if f in ("entry", "associate", "mid-senior") else "disagree"
        return "na"

    rows["roman_vs_final"] = rows.apply(agreement, axis=1)
    summary = rows["roman_vs_final"].value_counts().to_frame("n")
    summary["pct"] = 100 * summary["n"] / summary["n"].sum()
    summary.to_csv(TAB_DIR / "step2_roman_vs_final_summary.csv")
    print("\nLLM answer vs roman-numeral expectation:")
    print(summary.to_string())

    # Also tally agreement on sr/jr/principal/staff weak-keyword subtle cases.
    # We sample 100 LLM-labeled rows where title already hints at seniority via
    # a strong keyword (e.g. 'senior', 'lead') — if the LLM says 'unknown' for
    # a 'senior' titled role, that's a routing-error signal because Stage 5
    # should have fired first.
    strong_rows = q(f"""
        WITH matched AS (
          SELECT uid, source, period, title, seniority_final, yoe_extracted
          FROM '{DATA}'
          WHERE {FILTER}
            AND seniority_final_source = 'llm'
            AND regexp_matches(lower(title), '\\b(senior|staff|principal|lead|junior)\\b')
        )
        SELECT * FROM matched USING SAMPLE 100
    """)
    strong_rows.to_csv(TAB_DIR / "step2_llm_strong_keyword_sample.csv", index=False)
    print(f"\nSampled {len(strong_rows)} LLM-labeled rows whose title HAS a strong seniority keyword.")
    print("If these reached the LLM, Stage 5 did not fire — inspect for routing bugs.")
    # Quick agreement: does LLM final match strong keyword?
    def strong_agreement(row) -> str:
        t = (row["title"] or "").lower()
        f = row["seniority_final"]
        if re.search(r"\bjunior\b", t):
            return "agree" if f in ("entry", "associate") else "disagree"
        if re.search(r"\b(senior|sr\.?)\b", t):
            return "agree" if f in ("mid-senior", "director") else "disagree"
        if re.search(r"\b(staff|principal|lead)\b", t):
            return "agree" if f in ("mid-senior", "director") else "disagree"
        return "na"

    strong_rows["strong_match"] = strong_rows.apply(strong_agreement, axis=1)
    summary2 = strong_rows["strong_match"].value_counts().to_frame("n")
    summary2["pct"] = 100 * summary2["n"] / summary2["n"].sum()
    summary2.to_csv(TAB_DIR / "step2_strong_keyword_agreement.csv")
    print("\nLLM on strong-keyword titles (should have been rule-handled):")
    print(summary2.to_string())

    return rows, strong_rows


# -----------------------------------------------------------------------------
# Step 3: seniority_final vs seniority_native cross-tab + Cohen's kappa
# -----------------------------------------------------------------------------

def cohens_kappa(cm: np.ndarray) -> float:
    n = cm.sum()
    if n == 0:
        return float("nan")
    po = np.trace(cm) / n
    row_totals = cm.sum(axis=1)
    col_totals = cm.sum(axis=0)
    pe = np.sum(row_totals * col_totals) / (n * n)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def cross_tab(source: str) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    print(f"\n--- Cross-tab for source={source} ---")
    df = q(f"""
        SELECT seniority_native, seniority_final, COUNT(*) n
        FROM '{DATA}'
        WHERE {FILTER}
          AND source = '{source}'
          AND seniority_native IS NOT NULL
          AND seniority_final IS NOT NULL
          AND seniority_native != 'intern'
          AND seniority_native != 'executive'
          AND seniority_final != 'unknown'
          AND seniority_native != 'unknown'
        GROUP BY seniority_native, seniority_final
    """)
    # Use 4-level alignment: entry, associate, mid-senior, director
    labels = ["entry", "associate", "mid-senior", "director"]
    mat = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for _, r in df.iterrows():
        nat = r["seniority_native"]
        fin = r["seniority_final"]
        if nat in labels and fin in labels:
            mat.loc[nat, fin] = int(r["n"])

    # Marginals + per-class accuracy
    totals_native = mat.sum(axis=1)
    diagonal = pd.Series({lab: mat.loc[lab, lab] for lab in labels})
    acc = (diagonal / totals_native.replace(0, np.nan)).fillna(0)
    per_class = pd.DataFrame({
        "native_label": labels,
        "n_native": totals_native.values,
        "diag": diagonal.values,
        "per_class_accuracy": acc.values,
    })
    per_class.to_csv(TAB_DIR / f"kappa_per_class_{source}.csv", index=False)
    mat.to_csv(TAB_DIR / f"kappa_crosstab_{source}.csv")
    kappa = cohens_kappa(mat.values)
    print(f"Cohen's kappa (native vs final, {source}): {kappa:.4f}")
    print("Crosstab (rows=native, cols=final):")
    print(mat.to_string())
    print("Per-class accuracy:")
    print(per_class.to_string())
    return mat, kappa, per_class


def step_3_kappa():
    print("\n=== STEP 3: kappa vs seniority_native ===")
    _, kappa_arshkon, _ = cross_tab("kaggle_arshkon")
    _, kappa_scraped, _ = cross_tab("scraped")
    # asaniczka native has only mid-senior + associate, still useful to check
    _, kappa_asaniczka, _ = cross_tab("kaggle_asaniczka")

    summary = pd.DataFrame({
        "source": ["kaggle_arshkon", "kaggle_asaniczka", "scraped"],
        "cohen_kappa_4level": [kappa_arshkon, kappa_asaniczka, kappa_scraped],
    })
    summary.to_csv(TAB_DIR / "kappa_summary.csv", index=False)
    print("\nKappa summary:")
    print(summary.to_string())
    return summary


# -----------------------------------------------------------------------------
# Step 4: Native-label YOE diagnostic (does the native label track YOE?)
# -----------------------------------------------------------------------------

def step_4_native_yoe():
    print("\n=== STEP 4: Native-label YOE diagnostic ===")
    yoe = q(f"""
        SELECT source, seniority_native,
               COUNT(*) n,
               AVG(yoe_extracted) avg_yoe,
               QUANTILE(yoe_extracted, 0.25) q25,
               QUANTILE(yoe_extracted, 0.5) q50,
               QUANTILE(yoe_extracted, 0.75) q75
        FROM '{DATA}'
        WHERE {FILTER}
          AND seniority_native IS NOT NULL
          AND yoe_extracted IS NOT NULL
        GROUP BY source, seniority_native
        ORDER BY source, seniority_native
    """)
    yoe.to_csv(TAB_DIR / "native_yoe_by_source.csv", index=False)
    print(yoe.to_string())

    yoe_final = q(f"""
        SELECT source, seniority_final,
               COUNT(*) n,
               AVG(yoe_extracted) avg_yoe,
               QUANTILE(yoe_extracted, 0.25) q25,
               QUANTILE(yoe_extracted, 0.5) q50,
               QUANTILE(yoe_extracted, 0.75) q75
        FROM '{DATA}'
        WHERE {FILTER}
          AND yoe_extracted IS NOT NULL
        GROUP BY source, seniority_final
        ORDER BY source, seniority_final
    """)
    yoe_final.to_csv(TAB_DIR / "final_yoe_by_source.csv", index=False)
    print("\nYOE by seniority_final:")
    print(yoe_final.to_string())
    return yoe, yoe_final


def main():
    step_1_source_profile()
    step_2_rule_vs_llm()
    step_3_kappa()
    step_4_native_yoe()
    print("\nDone. Artifacts under exploration/tables/T03/ and exploration/figures/T03/.")


if __name__ == "__main__":
    main()
