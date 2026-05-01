"""T03. Seniority label audit.

Audits `seniority_final` against available diagnostics. Produces tables for:
  1. seniority_final_source profile by source/period/aggregator
  2. Rule-vs-LLM internal agreement (where both populated)
  3. seniority_final vs seniority_native (arshkon SWE + scraped SWE) — Cohen kappa + per-class accuracy
  4. Aggregator-stratified kappa
  5. Sample 100 LLM-labeled rows with weak seniority markers (I/II/III) for qualitative routing-error estimate

Output CSVs go to exploration/tables/T03/.
"""

from __future__ import annotations

import random
from pathlib import Path

import duckdb
import pandas as pd
import numpy as np

PARQUET = "data/unified.parquet"
OUT_TABLES = Path("exploration/tables/T03")
OUT_TABLES.mkdir(parents=True, exist_ok=True)

DEFAULT_FILTER = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'"


def q(sql: str) -> pd.DataFrame:
    return duckdb.sql(sql).df()


def save(df: pd.DataFrame, name: str) -> None:
    path = OUT_TABLES / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  -> {path}  ({len(df)} rows)")


def cohen_kappa(df_counts: pd.DataFrame, label_col_a: str, label_col_b: str, count_col: str = "n") -> dict:
    """Compute Cohen's kappa + per-class accuracy from a contingency-style dataframe.

    df_counts has columns label_col_a, label_col_b, count_col. Returns dict with
    kappa, observed_agreement, expected_agreement, overall_accuracy, per_class_accuracy_a.
    """
    if df_counts.empty:
        return {"kappa": float("nan"), "observed_agreement": float("nan"),
                "expected_agreement": float("nan"), "n": 0, "per_class": {}}
    # Labels
    labels = sorted(set(df_counts[label_col_a].dropna()) | set(df_counts[label_col_b].dropna()))
    # Contingency matrix
    mat = pd.pivot_table(df_counts, index=label_col_a, columns=label_col_b,
                         values=count_col, aggfunc="sum", fill_value=0)
    mat = mat.reindex(index=labels, columns=labels, fill_value=0)
    total = mat.values.sum()
    if total == 0:
        return {"kappa": float("nan"), "observed_agreement": float("nan"),
                "expected_agreement": float("nan"), "n": 0, "per_class": {}}
    observed = np.trace(mat.values) / total
    row_sums = mat.sum(axis=1).values / total
    col_sums = mat.sum(axis=0).values / total
    expected = np.sum(row_sums * col_sums)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else float("nan")
    # Per-class accuracy (along rows of label_col_a — treat A as reference)
    per_class = {}
    for label in labels:
        row_total = mat.loc[label].sum() if label in mat.index else 0
        diag = mat.loc[label, label] if (label in mat.index and label in mat.columns) else 0
        per_class[label] = {
            "row_total": int(row_total),
            "correct": int(diag),
            "accuracy": float(diag / row_total) if row_total > 0 else float("nan"),
        }
    return {
        "kappa": float(kappa),
        "observed_agreement": float(observed),
        "expected_agreement": float(expected),
        "n": int(total),
        "per_class": per_class,
    }


def step1_source_profile() -> None:
    print("\n[Step 1] seniority_final_source profile (SWE, default filter).")
    # By source
    df_by_source = q(f"""
      SELECT source, seniority_final_source, count(*) n
      FROM '{PARQUET}'
      WHERE is_swe AND {DEFAULT_FILTER}
      GROUP BY source, seniority_final_source
      ORDER BY source, seniority_final_source
    """)
    save(df_by_source, "source_profile_by_source")

    # By source x period
    df_by_src_period = q(f"""
      SELECT source, period, seniority_final_source, count(*) n
      FROM '{PARQUET}'
      WHERE is_swe AND {DEFAULT_FILTER}
      GROUP BY source, period, seniority_final_source
      ORDER BY source, period, seniority_final_source
    """)
    save(df_by_src_period, "source_profile_by_source_period")

    # By aggregator
    df_by_agg = q(f"""
      SELECT source, is_aggregator, seniority_final_source, count(*) n
      FROM '{PARQUET}'
      WHERE is_swe AND {DEFAULT_FILTER}
      GROUP BY source, is_aggregator, seniority_final_source
      ORDER BY source, is_aggregator, seniority_final_source
    """)
    save(df_by_agg, "source_profile_by_source_aggregator")


def step2_rule_vs_llm_agreement() -> None:
    print("\n[Step 2] Rule-vs-LLM agreement where both populated.")
    # Rows where seniority_rule != 'unknown' AND seniority_final_source == 'llm'
    # Means rule fired AND LLM fired; we can compare them.
    df = q(f"""
      SELECT source, seniority_rule, seniority_final, count(*) n
      FROM '{PARQUET}'
      WHERE is_swe AND {DEFAULT_FILTER}
        AND seniority_rule IS NOT NULL AND seniority_rule != 'unknown'
        AND seniority_final_source = 'llm'
      GROUP BY source, seniority_rule, seniority_final
      ORDER BY source, seniority_rule, seniority_final
    """)
    save(df, "rule_vs_llm_agreement_crosstab")

    # Kappa per source
    rows = []
    for source in sorted(df["source"].unique()):
        sub = df[df["source"] == source].rename(columns={"seniority_rule": "a", "seniority_final": "b"})[["a", "b", "n"]]
        k = cohen_kappa(sub, "a", "b")
        rows.append({
            "source": source,
            "n": k["n"],
            "kappa_rule_vs_llm": k["kappa"],
            "observed_agreement": k["observed_agreement"],
            "expected_agreement": k["expected_agreement"],
        })
    kappa_df = pd.DataFrame(rows)
    save(kappa_df, "rule_vs_llm_kappa_by_source")

    # Per-class accuracy table (rule as reference)
    per_class_rows = []
    for source in sorted(df["source"].unique()):
        sub = df[df["source"] == source].rename(columns={"seniority_rule": "a", "seniority_final": "b"})[["a", "b", "n"]]
        k = cohen_kappa(sub, "a", "b")
        for label, stats in k["per_class"].items():
            per_class_rows.append({
                "source": source,
                "seniority_rule": label,
                "n": stats["row_total"],
                "llm_matches_rule_n": stats["correct"],
                "llm_matches_rule_share": stats["accuracy"],
            })
    per_class_df = pd.DataFrame(per_class_rows)
    save(per_class_df, "rule_vs_llm_per_class_accuracy")


def _harmonize_native(label: str) -> str:
    """Map arshkon's 'intern' to 'entry' and 'executive' to 'director' so native and final share the canonical 5-level enum."""
    return {"intern": "entry", "executive": "director"}.get(label, label)


def step3_native_vs_final() -> None:
    print("\n[Step 3] seniority_native vs seniority_final (arshkon SWE + scraped SWE).")
    # Arshkon SWE (only source with reliable native labels across all 5 enum values)
    df_arsh = q(f"""
      SELECT seniority_native, seniority_final, count(*) n
      FROM '{PARQUET}'
      WHERE source = 'kaggle_arshkon' AND is_swe AND {DEFAULT_FILTER}
        AND seniority_native IS NOT NULL
      GROUP BY seniority_native, seniority_final
      ORDER BY seniority_native, seniority_final
    """)
    save(df_arsh, "native_vs_final_crosstab_arshkon")

    # Scraped LinkedIn SWE
    df_scraped = q(f"""
      SELECT seniority_native, seniority_final, count(*) n
      FROM '{PARQUET}'
      WHERE source = 'scraped' AND is_swe AND {DEFAULT_FILTER}
        AND seniority_native IS NOT NULL
      GROUP BY seniority_native, seniority_final
      ORDER BY seniority_native, seniority_final
    """)
    save(df_scraped, "native_vs_final_crosstab_scraped")

    # Kappa per source — three variants:
    #   (a) raw: native (with intern/executive) vs final
    #   (b) harmonized: intern->entry, executive->director
    #   (c) harmonized, excluding final='unknown' (conditional agreement)
    rows = []
    for source_tag, df_ in [("kaggle_arshkon", df_arsh), ("scraped", df_scraped)]:
        # Variant (a) raw
        sub = df_.rename(columns={"seniority_native": "a", "seniority_final": "b"})[["a", "b", "n"]]
        ka = cohen_kappa(sub, "a", "b")

        # Variant (b) harmonized
        df_h = df_.copy()
        df_h["a"] = df_h["seniority_native"].map(_harmonize_native)
        df_h["b"] = df_h["seniority_final"]
        df_h = df_h.groupby(["a", "b"], as_index=False)["n"].sum()
        kb = cohen_kappa(df_h, "a", "b")

        # Variant (c) harmonized, excl. final='unknown'
        df_c = df_h[df_h["b"] != "unknown"].copy()
        kc = cohen_kappa(df_c, "a", "b")

        rows.append({
            "source": source_tag,
            "n_raw": ka["n"], "kappa_raw": ka["kappa"], "observed_raw": ka["observed_agreement"],
            "n_harm": kb["n"], "kappa_harm": kb["kappa"], "observed_harm": kb["observed_agreement"],
            "n_harm_excl_unknown": kc["n"], "kappa_harm_excl_unknown": kc["kappa"],
            "observed_harm_excl_unknown": kc["observed_agreement"],
        })
    kappa_df = pd.DataFrame(rows)
    save(kappa_df, "native_vs_final_kappa")

    # Per-class accuracy (native as reference, raw labels)
    per_class_rows = []
    for source_tag, df_ in [("kaggle_arshkon", df_arsh), ("scraped", df_scraped)]:
        sub = df_.rename(columns={"seniority_native": "a", "seniority_final": "b"})[["a", "b", "n"]]
        k = cohen_kappa(sub, "a", "b")
        for label, stats in k["per_class"].items():
            per_class_rows.append({
                "source": source_tag,
                "seniority_native": label,
                "n_native": stats["row_total"],
                "final_matches_native_n": stats["correct"],
                "final_matches_native_share": stats["accuracy"],
            })
    per_class_df = pd.DataFrame(per_class_rows)
    save(per_class_df, "native_vs_final_per_class_accuracy")

    # Per-class accuracy excluding final='unknown'
    per_class_rows2 = []
    for source_tag, df_ in [("kaggle_arshkon", df_arsh), ("scraped", df_scraped)]:
        df_h = df_.copy()
        df_h["a"] = df_h["seniority_native"].map(_harmonize_native)
        df_h["b"] = df_h["seniority_final"]
        df_h = df_h.groupby(["a", "b"], as_index=False)["n"].sum()
        df_c = df_h[df_h["b"] != "unknown"].copy()
        k = cohen_kappa(df_c, "a", "b")
        for label, stats in k["per_class"].items():
            per_class_rows2.append({
                "source": source_tag,
                "seniority_native_harmonized": label,
                "n_native_labeled": stats["row_total"],
                "final_matches_native_n": stats["correct"],
                "final_matches_native_share": stats["accuracy"],
            })
    per_class_df2 = pd.DataFrame(per_class_rows2)
    save(per_class_df2, "native_vs_final_per_class_accuracy_excl_unknown")


def step4_aggregator_stratified() -> None:
    print("\n[Step 4] Aggregator-stratified diagnostics (kappa + per-class accuracy).")
    rows = []
    per_class_rows = []
    for source_tag in ["kaggle_arshkon", "scraped"]:
        for is_agg in [True, False]:
            df = q(f"""
              SELECT seniority_native, seniority_final, count(*) n
              FROM '{PARQUET}'
              WHERE source = '{source_tag}' AND is_swe AND {DEFAULT_FILTER}
                AND is_aggregator = {is_agg}
                AND seniority_native IS NOT NULL
              GROUP BY seniority_native, seniority_final
            """)

            # Raw
            sub = df.rename(columns={"seniority_native": "a", "seniority_final": "b"})[["a", "b", "n"]]
            k_raw = cohen_kappa(sub, "a", "b")

            # Harmonized + exclude unknown
            df_h = df.copy()
            df_h["a"] = df_h["seniority_native"].map(_harmonize_native)
            df_h["b"] = df_h["seniority_final"]
            df_h = df_h.groupby(["a", "b"], as_index=False)["n"].sum()
            df_c = df_h[df_h["b"] != "unknown"].copy()
            k_c = cohen_kappa(df_c, "a", "b")

            rows.append({
                "source": source_tag,
                "is_aggregator": is_agg,
                "n_raw": k_raw["n"],
                "kappa_raw": k_raw["kappa"],
                "observed_raw": k_raw["observed_agreement"],
                "n_harm_excl_unknown": k_c["n"],
                "kappa_harm_excl_unknown": k_c["kappa"],
                "observed_harm_excl_unknown": k_c["observed_agreement"],
            })
            # Per-class (harmonized, excl. unknown)
            for label, stats in k_c["per_class"].items():
                per_class_rows.append({
                    "source": source_tag,
                    "is_aggregator": is_agg,
                    "seniority_native": label,
                    "n_native": stats["row_total"],
                    "final_matches_native_n": stats["correct"],
                    "final_matches_native_share": stats["accuracy"],
                })
    kappa_df = pd.DataFrame(rows)
    save(kappa_df, "aggregator_stratified_kappa")
    pc_df = pd.DataFrame(per_class_rows)
    save(pc_df, "aggregator_stratified_per_class_accuracy")


def step5_weak_marker_sample() -> None:
    """Qualitative routing-error estimate: sample LLM-labeled rows whose titles have weak seniority markers (I/II/III)."""
    print("\n[Step 5] Sample 100 LLM-labeled rows with weak seniority markers (I/II/III).")
    # Rows where seniority_final_source='llm' AND title contains weak markers like 'II' or 'III' or roman levels
    # Use case-sensitive roman numerals (II, III, IV, V) at word boundaries, or 'level N' / 'lvl N' patterns.
    df = q(rf"""
      SELECT * FROM (
        SELECT uid, source, title, seniority_final, seniority_rule, yoe_min_years_llm,
               substr(description, 1, 200) desc_preview
        FROM '{PARQUET}'
        WHERE is_swe AND {DEFAULT_FILTER}
          AND seniority_final_source = 'llm'
          AND (regexp_matches(title, '\b(II|III|IV|V)\b')
               OR regexp_matches(lower(title), '\b(level|lvl)\s*[0-9]+\b'))
      ) USING SAMPLE 100
    """)
    save(df, "weak_marker_llm_sample")

    # Summary stats: how often does LLM assign 'unknown' on these weak-marker rows?
    df_stats = q(rf"""
      SELECT source, seniority_final, count(*) n
      FROM '{PARQUET}'
      WHERE is_swe AND {DEFAULT_FILTER}
        AND seniority_final_source = 'llm'
        AND (regexp_matches(title, '\b(II|III|IV|V)\b')
             OR regexp_matches(lower(title), '\b(level|lvl)\s*[0-9]+\b'))
      GROUP BY source, seniority_final
      ORDER BY source, seniority_final
    """)
    save(df_stats, "weak_marker_llm_summary")


def step6_llm_unknown_rate() -> None:
    """Extra diagnostic: how often does seniority_final_source=llm result in seniority_final='unknown'?"""
    print("\n[Step 6] LLM 'unknown' rate where source=llm.")
    df = q(f"""
      SELECT source, seniority_final_source, seniority_final, count(*) n
      FROM '{PARQUET}'
      WHERE is_swe AND {DEFAULT_FILTER}
        AND seniority_final_source = 'llm'
      GROUP BY source, seniority_final_source, seniority_final
      ORDER BY source, seniority_final
    """)
    save(df, "llm_labeled_final_distribution")

    # Share that's unknown vs not
    df2 = q(f"""
      SELECT source,
             count(*) total,
             sum(CASE WHEN seniority_final = 'unknown' THEN 1 ELSE 0 END) n_unknown,
             sum(CASE WHEN seniority_final = 'unknown' THEN 1 ELSE 0 END) * 1.0 / count(*) share_unknown
      FROM '{PARQUET}'
      WHERE is_swe AND {DEFAULT_FILTER}
        AND seniority_final_source = 'llm'
      GROUP BY source
      ORDER BY source
    """)
    save(df2, "llm_unknown_share_by_source")


def main():
    step1_source_profile()
    step2_rule_vs_llm_agreement()
    step3_native_vs_final()
    step4_aggregator_stratified()
    step5_weak_marker_sample()
    step6_llm_unknown_rate()
    print("\n[T03] complete.")


if __name__ == "__main__":
    main()
