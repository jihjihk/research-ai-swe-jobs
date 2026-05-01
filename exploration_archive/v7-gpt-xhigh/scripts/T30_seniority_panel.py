#!/usr/bin/env python3
"""T30 seniority definition panel.

Builds the canonical seniority operationalization panel plus overlap matrices,
MDE tables, aggregator sensitivity tables, and title-keyword spot-check samples.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA = "data/unified.parquet"
TABLE_OUT = Path("exploration/tables/T30")
FIG_OUT = Path("exploration/figures/T30")
SHARED_OUT = Path("exploration/artifacts/shared")
for path in (TABLE_OUT, FIG_OUT, SHARED_OUT):
    path.mkdir(parents=True, exist_ok=True)

BASE_FILTER = """
  source_platform = 'linkedin'
  AND is_english = true
  AND date_flag = 'ok'
  AND is_swe = true
"""

JUNIOR_PATTERN = r"\b(junior|jr|entry[- ]level|graduate|new[- ]grad|intern)\b"
SENIOR_PATTERN = r"\b(senior|sr\.?|staff|principal|lead|architect|distinguished)\b"

DEFINITIONS = [
    {
        "definition": "J1",
        "side": "junior",
        "label": "seniority_final = entry",
        "condition": "seniority_final = 'entry'",
        "known_condition": "seniority_final IS NOT NULL AND seniority_final <> 'unknown'",
        "known_basis": "known_seniority_final",
    },
    {
        "definition": "J2",
        "side": "junior",
        "label": "seniority_final in entry/associate",
        "condition": "seniority_final IN ('entry', 'associate')",
        "known_condition": "seniority_final IS NOT NULL AND seniority_final <> 'unknown'",
        "known_basis": "known_seniority_final",
    },
    {
        "definition": "J3",
        "side": "junior",
        "label": "yoe_extracted <= 2",
        "condition": "yoe_extracted <= 2",
        "known_condition": "yoe_extracted IS NOT NULL",
        "known_basis": "yoe_known",
    },
    {
        "definition": "J4",
        "side": "junior",
        "label": "yoe_extracted <= 3",
        "condition": "yoe_extracted <= 3",
        "known_condition": "yoe_extracted IS NOT NULL",
        "known_basis": "yoe_known",
    },
    {
        "definition": "J5",
        "side": "junior",
        "label": f"title regex {JUNIOR_PATTERN}",
        "condition": f"regexp_matches(coalesce(title_normalized, ''), '{JUNIOR_PATTERN}')",
        "known_condition": "title_normalized IS NOT NULL AND title_normalized <> ''",
        "known_basis": "title_known",
    },
    {
        "definition": "J6",
        "side": "junior",
        "label": "J1 union J5",
        "condition": (
            f"(seniority_final = 'entry' OR regexp_matches(coalesce(title_normalized, ''), "
            f"'{JUNIOR_PATTERN}'))"
        ),
        "known_condition": (
            "(seniority_final IS NOT NULL AND seniority_final <> 'unknown') "
            "OR (title_normalized IS NOT NULL AND title_normalized <> '')"
        ),
        "known_basis": "known_seniority_or_title",
    },
    {
        "definition": "S1",
        "side": "senior",
        "label": "seniority_final in mid-senior/director",
        "condition": "seniority_final IN ('mid-senior', 'director')",
        "known_condition": "seniority_final IS NOT NULL AND seniority_final <> 'unknown'",
        "known_basis": "known_seniority_final",
    },
    {
        "definition": "S2",
        "side": "senior",
        "label": "seniority_final = director",
        "condition": "seniority_final = 'director'",
        "known_condition": "seniority_final IS NOT NULL AND seniority_final <> 'unknown'",
        "known_basis": "known_seniority_final",
    },
    {
        "definition": "S3",
        "side": "senior",
        "label": f"title regex {SENIOR_PATTERN}",
        "condition": f"regexp_matches(coalesce(title_normalized, ''), '{SENIOR_PATTERN}')",
        "known_condition": "title_normalized IS NOT NULL AND title_normalized <> ''",
        "known_basis": "title_known",
    },
    {
        "definition": "S4",
        "side": "senior",
        "label": "yoe_extracted >= 5",
        "condition": "yoe_extracted >= 5",
        "known_condition": "yoe_extracted IS NOT NULL",
        "known_basis": "yoe_known",
    },
    {
        "definition": "S5",
        "side": "senior",
        "label": "yoe_extracted >= 8",
        "condition": "yoe_extracted >= 8",
        "known_condition": "yoe_extracted IS NOT NULL",
        "known_basis": "yoe_known",
    },
]


def assert_regexes() -> None:
    junior_re = re.compile(JUNIOR_PATTERN, re.IGNORECASE)
    senior_re = re.compile(SENIOR_PATTERN, re.IGNORECASE)

    for text in [
        "junior software engineer",
        "jr. backend engineer",
        "entry-level software engineer",
        "entry level developer",
        "new-grad software engineer",
        "new grad developer",
        "graduate software engineer",
        "software engineer intern",
    ]:
        assert junior_re.search(text), text
    for text in ["internal tools engineer", "internship coordinator", "midlevel developer"]:
        assert not junior_re.search(text), text

    for text in [
        "senior software engineer",
        "sr. backend engineer",
        "staff platform engineer",
        "principal software engineer",
        "technical lead",
        "cloud architect",
        "distinguished engineer",
    ]:
        assert senior_re.search(text), text
    for text in ["leadership program manager", "a leading company", "architecture analyst"]:
        assert not senior_re.search(text), text


def cohen_h_mde(p1: float, n1: int, n2: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Approximate two-sample proportion MDE in percentage points.

    Uses Cohen's h normal approximation. The returned value is the positive
    p2 - p1 difference detectable at the requested power.
    """
    if n1 <= 0 or n2 <= 0 or p1 < 0 or p1 > 1 or math.isnan(p1):
        return math.nan
    z_alpha = 1.959963984540054
    z_power = 0.8416212335729143
    h = (z_alpha + z_power) * math.sqrt(1 / n1 + 1 / n2)
    angle = math.asin(math.sqrt(min(max(p1, 0.0), 1.0))) + h / 2
    p2 = math.sin(min(angle, math.pi / 2)) ** 2
    return max(0.0, p2 - p1)


def direction(effect: float) -> str:
    if pd.isna(effect):
        return "unknown"
    if effect > 0.0005:
        return "up"
    if effect < -0.0005:
        return "down"
    return "flat"


def make_group_view(con: duckdb.DuckDBPyConnection, exclude_aggregators: bool) -> None:
    extra = "AND coalesce(is_aggregator, false) = false" if exclude_aggregators else ""
    con.execute("DROP VIEW IF EXISTS base_t30")
    con.execute("DROP VIEW IF EXISTS grouped_t30")
    con.execute(
        f"""
        CREATE TEMP VIEW base_t30 AS
        SELECT
          uid,
          source,
          period,
          seniority_final,
          seniority_final_source,
          yoe_extracted,
          title_normalized,
          title,
          description,
          seniority_native,
          is_aggregator
        FROM read_parquet('{DATA}')
        WHERE {BASE_FILTER}
          {extra}
        """
    )
    con.execute(
        """
        CREATE TEMP VIEW grouped_t30 AS
        SELECT *, 'arshkon' AS source_group, '2024' AS panel_period
        FROM base_t30
        WHERE source = 'kaggle_arshkon'
        UNION ALL
        SELECT *, 'asaniczka' AS source_group, '2024' AS panel_period
        FROM base_t30
        WHERE source = 'kaggle_asaniczka'
        UNION ALL
        SELECT *, 'pooled_2024' AS source_group, '2024' AS panel_period
        FROM base_t30
        WHERE source IN ('kaggle_arshkon', 'kaggle_asaniczka')
        UNION ALL
        SELECT *, 'scraped_2026' AS source_group, '2026' AS panel_period
        FROM base_t30
        WHERE source = 'scraped'
        """
    )


def compute_panel(con: duckdb.DuckDBPyConnection, exclude_aggregators: bool) -> pd.DataFrame:
    make_group_view(con, exclude_aggregators)
    pieces = []
    for definition in DEFINITIONS:
        query = f"""
            SELECT
              '{definition["definition"]}' AS definition,
              '{definition["side"]}' AS side,
              '{definition["label"]}' AS definition_label,
              '{definition["known_basis"]}' AS known_denominator_basis,
              panel_period AS period,
              source_group AS source,
              count(*) AS all_swe_denominator,
              sum(CASE WHEN {definition["known_condition"]} THEN 1 ELSE 0 END) AS known_denominator,
              sum(CASE WHEN {definition["condition"]} THEN 1 ELSE 0 END) AS n_of_all,
              sum(CASE WHEN ({definition["condition"]}) AND ({definition["known_condition"]}) THEN 1 ELSE 0 END)
                AS n_of_known
            FROM grouped_t30
            GROUP BY panel_period, source_group
        """
        pieces.append(con.execute(query).df())

    panel = pd.concat(pieces, ignore_index=True)
    panel["share_of_all"] = panel["n_of_all"] / panel["all_swe_denominator"]
    panel["share_of_known"] = panel["n_of_known"] / panel["known_denominator"].replace({0: np.nan})

    effects = []
    for definition, sub in panel.groupby("definition"):
        by_source = sub.set_index("source")
        ar = by_source.loc["arshkon"]
        az = by_source.loc["asaniczka"]
        pooled = by_source.loc["pooled_2024"]
        scraped = by_source.loc["scraped_2026"]
        within_known = az["share_of_known"] - ar["share_of_known"]
        cross_known = scraped["share_of_known"] - pooled["share_of_known"]
        within_all = az["share_of_all"] - ar["share_of_all"]
        cross_all = scraped["share_of_all"] - pooled["share_of_all"]
        mde_ar = cohen_h_mde(ar["share_of_known"], int(ar["known_denominator"]), int(scraped["known_denominator"]))
        mde_pooled = cohen_h_mde(
            pooled["share_of_known"],
            int(pooled["known_denominator"]),
            int(scraped["known_denominator"]),
        )
        effects.append(
            {
                "definition": definition,
                "within_2024_effect": within_known,
                "cross_period_effect": cross_known,
                "within_2024_effect_all_denom": within_all,
                "cross_period_effect_all_denom": cross_all,
                "mde_arshkon_vs_scraped": mde_ar,
                "mde_pooled_vs_scraped": mde_pooled,
                "direction": direction(cross_known),
                "direction_all_denom": direction(cross_all),
                "signal_to_noise_known": (
                    abs(cross_known) / abs(within_known) if abs(within_known) > 0 else math.inf
                ),
            }
        )
    effects_df = pd.DataFrame(effects)
    return panel.merge(effects_df, on="definition", how="left")


def compute_overlap(con: duckdb.DuckDBPyConnection, side: str) -> pd.DataFrame:
    defs = [d for d in DEFINITIONS if d["side"] == side]
    rows = []
    for row_def in defs:
        denom = con.execute(
            f"""
            SELECT count(*)
            FROM base_t30
            WHERE {row_def["condition"]}
            """
        ).fetchone()[0]
        row = {"definition": row_def["definition"], "n": int(denom)}
        for col_def in defs:
            numerator = con.execute(
                f"""
                SELECT count(*)
                FROM base_t30
                WHERE ({row_def["condition"]}) AND ({col_def["condition"]})
                """
            ).fetchone()[0]
            row[col_def["definition"]] = numerator / denom if denom else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_overlap(df: pd.DataFrame, side: str) -> None:
    labels = df["definition"].tolist()
    matrix = df[labels].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(f"{side.title()} definition row overlap: |X and Y| / |X|")
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            text = "" if math.isnan(val) else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", color="white" if val < 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_OUT / f"{side}_overlap_heatmap.png", dpi=150)
    plt.close(fig)


def write_samples(con: duckdb.DuckDBPyConnection) -> None:
    j5 = next(d for d in DEFINITIONS if d["definition"] == "J5")
    j1 = next(d for d in DEFINITIONS if d["definition"] == "J1")
    s3 = next(d for d in DEFINITIONS if d["definition"] == "S3")
    s1 = next(d for d in DEFINITIONS if d["definition"] == "S1")

    for name, condition in {
        "J5_only_not_J1": f"({j5['condition']}) AND NOT ({j1['condition']})",
        "S3_only_not_S1": f"({s3['condition']}) AND NOT ({s1['condition']})",
    }.items():
        pop_n = con.execute(f"SELECT count(*) FROM base_t30 WHERE {condition}").fetchone()[0]
        sample = con.execute(
            f"""
            SELECT
              uid,
              source,
              period,
              title,
              seniority_final,
              seniority_final_source,
              seniority_native,
              yoe_extracted,
              left(coalesce(description, ''), 240) AS description_excerpt
            FROM base_t30
            WHERE {condition}
            ORDER BY hash(uid)
            LIMIT 50
            """
        ).df()
        sample.insert(0, "spotcheck_population_n", int(pop_n))
        sample.to_csv(TABLE_OUT / f"spotcheck_{name}.csv", index=False)


def write_raw_title_sensitivity(con: duckdb.DuckDBPyConnection) -> None:
    rows = []
    for name, side, pattern in [
        ("J5_raw_title", "junior", f"(?i){JUNIOR_PATTERN}"),
        ("S3_raw_title", "senior", f"(?i){SENIOR_PATTERN}"),
    ]:
        df = con.execute(
            """
            WITH grouped AS (
              SELECT *, 'arshkon' AS source_group, '2024' AS panel_period
              FROM base_t30
              WHERE source = 'kaggle_arshkon'
              UNION ALL
              SELECT *, 'asaniczka' AS source_group, '2024' AS panel_period
              FROM base_t30
              WHERE source = 'kaggle_asaniczka'
              UNION ALL
              SELECT *, 'pooled_2024' AS source_group, '2024' AS panel_period
              FROM base_t30
              WHERE source IN ('kaggle_arshkon', 'kaggle_asaniczka')
              UNION ALL
              SELECT *, 'scraped_2026' AS source_group, '2026' AS panel_period
              FROM base_t30
              WHERE source = 'scraped'
            )
            SELECT
              ? AS definition,
              ? AS side,
              panel_period AS period,
              source_group AS source,
              count(*) AS all_swe_denominator,
              sum(CASE WHEN regexp_matches(coalesce(title, ''), ?) THEN 1 ELSE 0 END) AS n,
              sum(CASE WHEN regexp_matches(coalesce(title, ''), ?) THEN 1 ELSE 0 END)::DOUBLE
                / count(*) AS share_of_all
            FROM grouped
            GROUP BY panel_period, source_group
            ORDER BY source_group
            """,
            [name, side, pattern, pattern],
        ).df()
        rows.append(df)
    pd.concat(rows, ignore_index=True).to_csv(TABLE_OUT / "title_keyword_raw_title_sensitivity.csv", index=False)


def main() -> None:
    assert_regexes()
    con = duckdb.connect()

    primary = compute_panel(con, exclude_aggregators=False)
    primary.to_csv(TABLE_OUT / "seniority_definition_panel_detailed.csv", index=False)

    canonical_cols = [
        "definition",
        "side",
        "period",
        "source",
        "n_of_all",
        "n_of_known",
        "share_of_all",
        "share_of_known",
        "mde_arshkon_vs_scraped",
        "mde_pooled_vs_scraped",
        "within_2024_effect",
        "cross_period_effect",
        "direction",
    ]
    primary[canonical_cols].to_csv(SHARED_OUT / "seniority_definition_panel.csv", index=False)

    effect_cols = [
        "definition",
        "side",
        "definition_label",
        "known_denominator_basis",
        "mde_arshkon_vs_scraped",
        "mde_pooled_vs_scraped",
        "within_2024_effect",
        "cross_period_effect",
        "within_2024_effect_all_denom",
        "cross_period_effect_all_denom",
        "direction",
        "direction_all_denom",
        "signal_to_noise_known",
    ]
    (
        primary[primary["source"] == "pooled_2024"][effect_cols]
        .sort_values("definition")
        .to_csv(TABLE_OUT / "definition_effects_and_mde.csv", index=False)
    )

    no_aggs = compute_panel(con, exclude_aggregators=True)
    no_aggs.to_csv(TABLE_OUT / "seniority_definition_panel_no_aggregators.csv", index=False)
    (
        no_aggs[no_aggs["source"] == "pooled_2024"][effect_cols]
        .sort_values("definition")
        .to_csv(TABLE_OUT / "definition_effects_and_mde_no_aggregators.csv", index=False)
    )

    make_group_view(con, exclude_aggregators=False)
    junior_overlap = compute_overlap(con, "junior")
    senior_overlap = compute_overlap(con, "senior")
    junior_overlap.to_csv(TABLE_OUT / "junior_overlap_matrix.csv", index=False)
    senior_overlap.to_csv(TABLE_OUT / "senior_overlap_matrix.csv", index=False)
    plot_overlap(junior_overlap, "junior")
    plot_overlap(senior_overlap, "senior")
    write_samples(con)
    write_raw_title_sensitivity(con)

    print(f"Wrote T30 tables to {TABLE_OUT}")
    print(f"Wrote T30 figures to {FIG_OUT}")
    print(f"Wrote canonical panel to {SHARED_OUT / 'seniority_definition_panel.csv'}")


if __name__ == "__main__":
    main()
