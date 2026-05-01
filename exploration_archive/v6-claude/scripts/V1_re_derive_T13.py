"""V1 Target 3 — Independent re-derivation of T13 section-anatomy decomposition.

Spec: re-derive the per-section char-length means for 2024 vs 2026 on a
2,000-row stratified sample per period using the T13 section classifier
(the classifier is a reusable function and counts as tooling, not analysis).

T13 reported: responsibilities +196 (52%), role_summary +139 (37%),
preferred +111 (29%), requirements -2 (flat).

This script draws a fresh random sample, uses description_core_llm from
`data/unified.parquet` filtered to the standard SWE frame, and recomputes
the growth decomposition from scratch.
"""

from __future__ import annotations

import random
import sys

import duckdb
import pandas as pd

sys.path.insert(0, "/home/jihgaboot/gabor/job-research/exploration/scripts")
from T13_section_classifier import SECTIONS, section_char_proportions  # noqa: E402

UNI = "/home/jihgaboot/gabor/job-research/data/unified.parquet"
OUT = "/home/jihgaboot/gabor/job-research/exploration/tables/V1"

random.seed(2026_04_15)


def main() -> None:
    con = duckdb.connect()

    q = f"""
    SELECT uid, source, period, description_core_llm
    FROM '{UNI}'
    WHERE is_swe = true
      AND source_platform = 'linkedin'
      AND is_english = true
      AND date_flag = 'ok'
      AND llm_extraction_coverage = 'labeled'
      AND description_core_llm IS NOT NULL
    """
    df = con.execute(q).fetchdf()
    df["period_bucket"] = df["source"].apply(
        lambda s: "2026" if s == "scraped" else "2024"
    )
    print(f"population rows: {len(df):,}")
    print(df.groupby("period_bucket").size())

    # Stratified sample: 2,000 per period
    samp = (
        df.groupby("period_bucket", group_keys=False)
        .apply(lambda g: g.sample(n=min(2000, len(g)), random_state=42))
        .reset_index(drop=True)
    )
    print(f"sampled rows: {len(samp):,}")

    # Run classifier on each
    rows = []
    for _, r in samp.iterrows():
        text = r["description_core_llm"] or ""
        props = section_char_proportions(text)
        row = {"uid": r["uid"], "period_bucket": r["period_bucket"], "total_chars": len(text)}
        row.update(props)
        rows.append(row)
    out = pd.DataFrame(rows)

    # Aggregate by period
    agg = out.groupby("period_bucket")[SECTIONS + ["total_chars"]].mean().round(1)
    print()
    print("Per-period mean chars by section (from stratified 2k/period sample):")
    print(agg.T)

    # Build growth decomposition table
    deltas = agg.loc["2026"] - agg.loc["2024"]
    tot_growth = deltas["total_chars"]
    rows_out = []
    for s in SECTIONS:
        d = deltas[s]
        rows_out.append(
            {
                "section": s,
                "mean_2024": agg.loc["2024", s],
                "mean_2026": agg.loc["2026", s],
                "delta": d,
                "share_of_growth_pct": (d / tot_growth * 100) if tot_growth else 0,
            }
        )
    rows_out.append(
        {
            "section": "TOTAL",
            "mean_2024": agg.loc["2024", "total_chars"],
            "mean_2026": agg.loc["2026", "total_chars"],
            "delta": tot_growth,
            "share_of_growth_pct": 100.0,
        }
    )
    decomp = pd.DataFrame(rows_out)
    decomp.to_csv(f"{OUT}/V1_T13_section_decomposition.csv", index=False)
    print()
    print("Growth decomposition (V1 re-derived):")
    with pd.option_context("display.width", 160, "display.max_columns", 99):
        print(decomp.to_string(index=False))

    print()
    print("T13 reported:")
    print("  role_summary +139 (37.0%)")
    print("  responsibilities +196 (51.8%)")
    print("  requirements -2 (-0.5%)")
    print("  preferred +111 (29.4%)")
    print("  unclassified -71 (-18.7%)")
    print("  TOTAL +377")


if __name__ == "__main__":
    main()
