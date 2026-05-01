#!/usr/bin/env python3
"""T03 seniority label audit.

Reads data/unified.parquet with DuckDB and writes small aggregate/sample
artifacts for the Agent B Wave 1 seniority-quality report.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import duckdb
import pandas as pd


DATA = "data/unified.parquet"
OUT = Path("exploration/tables/T03")
OUT.mkdir(parents=True, exist_ok=True)

BASE_FILTER = """
  source_platform = 'linkedin'
  AND is_english = true
  AND date_flag = 'ok'
"""

LABELS = ["entry", "associate", "mid-senior", "director", "unknown"]


def cohen_kappa(counts: pd.DataFrame) -> float:
    total = counts["n"].sum()
    if total == 0:
        return math.nan
    observed = counts.loc[counts["native"] == counts["final"], "n"].sum() / total
    native_marg = counts.groupby("native")["n"].sum()
    final_marg = counts.groupby("final")["n"].sum()
    expected = 0.0
    for label in sorted(set(native_marg.index).union(final_marg.index)):
        expected += (native_marg.get(label, 0) / total) * (final_marg.get(label, 0) / total)
    if expected == 1:
        return math.nan
    return (observed - expected) / (1 - expected)


def infer_weak_expected(title: str | None) -> str:
    t = (title or "").lower()
    if re.search(r"\b(?:i|1|l1|level\s*1|grade\s*1)\b", t):
        return "entry"
    if re.search(r"\b(?:ii|2|l2|level\s*2|grade\s*2)\b", t):
        return "associate_or_mid"
    if re.search(r"\b(?:iii|iv|v|3|4|5|l3|l4|l5|level\s*[3-5]|grade\s*[3-5])\b", t):
        return "mid-senior"
    return "ambiguous"


def weak_consistent(expected: str, final: str | None) -> bool | None:
    if expected == "entry":
        return final in {"entry", "associate"}
    if expected == "associate_or_mid":
        return final in {"associate", "mid-senior"}
    if expected == "mid-senior":
        return final == "mid-senior"
    return None


def main() -> None:
    con = duckdb.connect()

    source_profile = con.execute(
        f"""
        WITH swe AS (
          SELECT source, period, seniority_final_source, count(*) AS n
          FROM read_parquet('{DATA}')
          WHERE {BASE_FILTER}
            AND is_swe = true
          GROUP BY source, period, seniority_final_source
        )
        SELECT
          source,
          period,
          seniority_final_source,
          n,
          n::DOUBLE / sum(n) OVER (PARTITION BY source, period) AS share
        FROM swe
        ORDER BY source, period, seniority_final_source
        """
    ).df()
    source_profile.to_csv(OUT / "seniority_final_source_by_source_period.csv", index=False)

    source_profile_overall = con.execute(
        f"""
        WITH swe AS (
          SELECT period, seniority_final_source, count(*) AS n
          FROM read_parquet('{DATA}')
          WHERE {BASE_FILTER}
            AND is_swe = true
          GROUP BY period, seniority_final_source
        )
        SELECT
          period,
          seniority_final_source,
          n,
          n::DOUBLE / sum(n) OVER (PARTITION BY period) AS share
        FROM swe
        ORDER BY period, seniority_final_source
        """
    ).df()
    source_profile_overall.to_csv(OUT / "seniority_final_source_by_period.csv", index=False)

    crosstab = con.execute(
        f"""
        WITH base AS (
          SELECT
            CASE
              WHEN source = 'kaggle_arshkon' THEN 'arshkon_2024'
              WHEN source = 'scraped' THEN 'scraped_2026'
            END AS source_group,
            seniority_native AS native,
            seniority_final AS final
          FROM read_parquet('{DATA}')
          WHERE {BASE_FILTER}
            AND is_swe = true
            AND source IN ('kaggle_arshkon', 'scraped')
            AND seniority_native IS NOT NULL
        )
        SELECT source_group, native, final, count(*) AS n
        FROM base
        WHERE source_group IS NOT NULL
        GROUP BY source_group, native, final
        ORDER BY source_group, native, final
        """
    ).df()
    crosstab.to_csv(OUT / "seniority_final_vs_native_crosstab_long.csv", index=False)

    rows = []
    for source_group, sub in crosstab.groupby("source_group"):
        total = int(sub["n"].sum())
        exact = int(sub.loc[sub["native"] == sub["final"], "n"].sum())
        rows.append(
            {
                "source_group": source_group,
                "n": total,
                "exact_match_n": exact,
                "exact_match_share": exact / total if total else math.nan,
                "cohen_kappa": cohen_kappa(sub),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "seniority_native_kappa.csv", index=False)

    mapped = crosstab.copy()
    mapped["native"] = mapped["native"].replace({"intern": "entry", "executive": "director"})
    mapped = mapped.groupby(["source_group", "native", "final"], as_index=False)["n"].sum()
    sensitivity_rows = []
    for comparison, frame in {
        "native_intern_entry_executive_director_all_final_values": mapped,
        "native_intern_entry_executive_director_final_known_only": mapped[mapped["final"] != "unknown"],
    }.items():
        for source_group, sub in frame.groupby("source_group"):
            total = int(sub["n"].sum())
            exact = int(sub.loc[sub["native"] == sub["final"], "n"].sum())
            sensitivity_rows.append(
                {
                    "comparison": comparison,
                    "source_group": source_group,
                    "n": total,
                    "exact_match_n": exact,
                    "exact_match_share": exact / total if total else math.nan,
                    "cohen_kappa": cohen_kappa(sub),
                }
            )
    pd.DataFrame(sensitivity_rows).to_csv(OUT / "seniority_native_kappa_sensitivity.csv", index=False)

    per_class = (
        crosstab.assign(match=lambda d: d["native"] == d["final"])
        .groupby(["source_group", "native"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "native_n": int(g["n"].sum()),
                    "exact_match_n": int(g.loc[g["match"], "n"].sum()),
                    "accuracy_using_native_reference": (
                        int(g.loc[g["match"], "n"].sum()) / int(g["n"].sum())
                        if int(g["n"].sum())
                        else math.nan
                    ),
                }
            )
        )
        .reset_index(drop=True)
    )
    per_class.to_csv(OUT / "seniority_native_per_class_accuracy.csv", index=False)

    yoe_diag = con.execute(
        f"""
        WITH base AS (
          SELECT
            CASE
              WHEN source = 'kaggle_arshkon' THEN 'arshkon_2024'
              WHEN source = 'scraped' THEN 'scraped_2026'
            END AS source_group,
            seniority_native,
            seniority_final,
            yoe_extracted
          FROM read_parquet('{DATA}')
          WHERE {BASE_FILTER}
            AND is_swe = true
            AND source IN ('kaggle_arshkon', 'scraped')
        ),
        labels AS (
          SELECT source_group, 'native' AS label_system, seniority_native AS label, yoe_extracted
          FROM base
          WHERE source_group IS NOT NULL AND seniority_native IS NOT NULL
          UNION ALL
          SELECT source_group, 'final' AS label_system, seniority_final AS label, yoe_extracted
          FROM base
          WHERE source_group IS NOT NULL
        )
        SELECT
          source_group,
          label_system,
          label,
          count(*) AS n,
          count(yoe_extracted) AS yoe_known_n,
          count(yoe_extracted)::DOUBLE / count(*) AS yoe_known_share,
          avg(yoe_extracted) AS avg_yoe,
          quantile_cont(yoe_extracted, 0.25) AS p25_yoe,
          quantile_cont(yoe_extracted, 0.50) AS median_yoe,
          quantile_cont(yoe_extracted, 0.75) AS p75_yoe,
          CASE WHEN count(yoe_extracted) > 0
            THEN sum(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END)::DOUBLE / count(yoe_extracted)
          END AS share_yoe_le_2_among_known,
          CASE WHEN count(yoe_extracted) > 0
            THEN sum(CASE WHEN yoe_extracted >= 5 THEN 1 ELSE 0 END)::DOUBLE / count(yoe_extracted)
          END AS share_yoe_ge_5_among_known
        FROM labels
        GROUP BY source_group, label_system, label
        ORDER BY source_group, label_system, label
        """
    ).df()
    yoe_diag.to_csv(OUT / "native_and_final_yoe_diagnostic.csv", index=False)

    weak_pattern = r"(?i)\b(i|ii|iii|iv|v|[1-5]|l[1-5]|level\s*[1-5]|grade\s*[1-5])\b"
    weak_population = con.execute(
        f"""
        SELECT count(*) AS n
        FROM read_parquet('{DATA}')
        WHERE {BASE_FILTER}
          AND is_swe = true
          AND seniority_final_source = 'llm'
          AND regexp_matches(coalesce(title_normalized, ''), ?)
        """,
        [weak_pattern],
    ).fetchone()[0]

    weak_sample = con.execute(
        f"""
        SELECT
          uid,
          source,
          period,
          title,
          title_normalized,
          company_name,
          seniority_final,
          seniority_native,
          yoe_extracted,
          left(coalesce(description, ''), 240) AS description_excerpt
        FROM read_parquet('{DATA}')
        WHERE {BASE_FILTER}
          AND is_swe = true
          AND seniority_final_source = 'llm'
          AND regexp_matches(coalesce(title_normalized, ''), ?)
        ORDER BY hash(uid)
        LIMIT 100
        """,
        [weak_pattern],
    ).df()
    weak_sample["weak_marker_expected"] = weak_sample["title_normalized"].map(infer_weak_expected)
    weak_sample["heuristic_consistent_with_title_marker"] = [
        weak_consistent(exp, final)
        for exp, final in zip(weak_sample["weak_marker_expected"], weak_sample["seniority_final"])
    ]
    weak_sample.to_csv(OUT / "llm_weak_marker_spotcheck_sample.csv", index=False)

    weak_summary = pd.DataFrame(
        [
            {
                "llm_weak_marker_population_n": int(weak_population),
                "sample_n": int(len(weak_sample)),
                "heuristic_consistent_n": int(
                    weak_sample["heuristic_consistent_with_title_marker"].fillna(False).sum()
                ),
                "heuristic_inconsistent_n": int(
                    (weak_sample["heuristic_consistent_with_title_marker"] == False).sum()
                ),
                "heuristic_inconsistent_share": (
                    float((weak_sample["heuristic_consistent_with_title_marker"] == False).mean())
                    if len(weak_sample)
                    else math.nan
                ),
            }
        ]
    )
    weak_summary.to_csv(OUT / "llm_weak_marker_spotcheck_summary.csv", index=False)

    print(f"Wrote T03 tables to {OUT}")


if __name__ == "__main__":
    main()
