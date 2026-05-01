from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
TAB_DIR = ROOT / "exploration" / "tables" / "T02"

PRIMARY_FILTER = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'"

TARGET_LABELS = ["entry", "associate", "mid-senior"]
ARSHKON_SOURCE = "kaggle_arshkon"
ASANICZKA_SOURCE = "kaggle_asaniczka"


JUNIOR_RE = re.compile(
    r"(^|[^a-z0-9])(junior|jr\.?|entry(?:[ -]?level)?|new[ -]?grad(?:uate)?|intern(?:ship)?|apprentice|trainee)([^a-z0-9]|$)"
)
SENIOR_RE = re.compile(
    r"(^|[^a-z0-9])(senior|sr\.?|staff|principal|lead|manager|director|head|vp|vice president|architect|distinguished|fellow|expert)([^a-z0-9]|$)"
)
ASSOCIATE_RE = re.compile(r"(^|[^a-z0-9])associate([^a-z0-9]|$)")


def match(pattern: re.Pattern[str], text: str | None) -> bool:
    return bool(pattern.search((text or "").lower()))


def main() -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    # Inline assertions for the custom title-cue logic.
    assert match(JUNIOR_RE, "Junior Software Engineer")
    assert match(JUNIOR_RE, "Entry-Level Data Engineer")
    assert match(JUNIOR_RE, "New Grad Software Engineer")
    assert not match(JUNIOR_RE, "A leading company")
    assert match(SENIOR_RE, "Staff Software Engineer")
    assert match(SENIOR_RE, "Sr. Backend Engineer")
    assert not match(SENIOR_RE, "Assistant Engineer")
    assert match(ASSOCIATE_RE, "Associate Software Engineer")
    assert not match(ASSOCIATE_RE, "Associated Systems Engineer")

    con = duckdb.connect()
    base = f"""
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {PRIMARY_FILTER}
          AND is_swe = true
          AND title_normalized IS NOT NULL
          AND title_normalized <> ''
    """

    # Label inventory for the sources of interest.
    label_inventory = con.execute(
        f"""
        SELECT source, seniority_native, COUNT(*) AS n
        {base}
        GROUP BY 1, 2
        ORDER BY 1, 3 DESC, 2
        """
    ).fetchdf()
    label_inventory.to_csv(TAB_DIR / "T02_native_label_inventory.csv", index=False)

    final_inventory = con.execute(
        f"""
        SELECT source, seniority_native, seniority_final, COUNT(*) AS n
        {base}
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 4 DESC, 3
        """
    ).fetchdf()
    final_inventory.to_csv(TAB_DIR / "T02_final_by_native_distribution.csv", index=False)

    # Pull the relevant title sets and row-level diagnostics into pandas.
    title_rows = con.execute(
        f"""
        SELECT source,
               seniority_native,
               seniority_final,
               seniority_final_source,
               COALESCE(title_normalized, lower(title)) AS title_norm,
               yoe_extracted,
               metro_area,
               description_core_llm,
               llm_extraction_coverage
        {base}
        AND (
            (source = '{ASANICZKA_SOURCE}' AND seniority_native = 'associate')
            OR (source = '{ARSHKON_SOURCE}' AND seniority_native IN ('entry', 'associate', 'mid-senior'))
        )
        """
    ).fetchdf()

    title_rows["title_norm"] = title_rows["title_norm"].fillna("").str.strip().str.lower()
    title_rows["junior_cue"] = title_rows["title_norm"].map(lambda x: match(JUNIOR_RE, x))
    title_rows["senior_cue"] = title_rows["title_norm"].map(lambda x: match(SENIOR_RE, x))
    title_rows["associate_token"] = title_rows["title_norm"].map(lambda x: match(ASSOCIATE_RE, x))

    groups = {
        (ASANICZKA_SOURCE, "associate"): set(
            title_rows.loc[
                (title_rows["source"] == ASANICZKA_SOURCE) & (title_rows["seniority_native"] == "associate"),
                "title_norm",
            ].unique()
        ),
        (ARSHKON_SOURCE, "entry"): set(
            title_rows.loc[
                (title_rows["source"] == ARSHKON_SOURCE) & (title_rows["seniority_native"] == "entry"),
                "title_norm",
            ].unique()
        ),
        (ARSHKON_SOURCE, "associate"): set(
            title_rows.loc[
                (title_rows["source"] == ARSHKON_SOURCE) & (title_rows["seniority_native"] == "associate"),
                "title_norm",
            ].unique()
        ),
        (ARSHKON_SOURCE, "mid-senior"): set(
            title_rows.loc[
                (title_rows["source"] == ARSHKON_SOURCE) & (title_rows["seniority_native"] == "mid-senior"),
                "title_norm",
            ].unique()
        ),
    }

    overlap_rows = []
    base_titles = groups[(ASANICZKA_SOURCE, "associate")]
    for native in ["entry", "associate", "mid-senior"]:
        other = groups[(ARSHKON_SOURCE, native)]
        shared = base_titles & other
        overlap_rows.append(
            {
                "asaniczka_group": "associate",
                "arshkon_group": native,
                "asaniczka_unique_titles": len(base_titles),
                "arshkon_unique_titles": len(other),
                "shared_titles": len(shared),
                "asaniczka_shared_rate": None if not base_titles else len(shared) / len(base_titles),
                "arshkon_shared_rate": None if not other else len(shared) / len(other),
                "jaccard": None if not (base_titles or other) else len(shared) / len(base_titles | other),
            }
        )
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(TAB_DIR / "T02_exact_title_overlap.csv", index=False)

    cue_summary = (
        title_rows.assign(
            title_group=lambda d: d["source"].astype(str) + "::" + d["seniority_native"].astype(str)
        )
        .groupby(["source", "seniority_native"], dropna=False)
        .agg(
            n=("title_norm", "size"),
            unique_titles=("title_norm", "nunique"),
            junior_cue_rate=("junior_cue", "mean"),
            senior_cue_rate=("senior_cue", "mean"),
            associate_token_rate=("associate_token", "mean"),
            yoe_non_null=("yoe_extracted", lambda s: s.notna().sum()),
            yoe_le_2_rate=("yoe_extracted", lambda s: s.dropna().le(2).mean()),
            yoe_le_3_rate=("yoe_extracted", lambda s: s.dropna().le(3).mean()),
            yoe_median=("yoe_extracted", "median"),
            yoe_p25=("yoe_extracted", lambda s: s.quantile(0.25)),
            yoe_p75=("yoe_extracted", lambda s: s.quantile(0.75)),
        )
        .reset_index()
    )
    cue_summary.to_csv(TAB_DIR / "T02_title_cue_and_yoe_summary.csv", index=False)

    final_dist = (
        title_rows.groupby(["source", "seniority_native", "seniority_final"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    final_totals = final_dist.groupby(["source", "seniority_native"])["n"].transform("sum")
    final_dist["share"] = final_dist["n"] / final_totals
    final_dist.to_csv(TAB_DIR / "T02_final_distribution_conditional_on_native.csv", index=False)

    entry_rows = con.execute(
        f"""
        SELECT source,
               seniority_final_source,
               COUNT(*) AS entry_n,
               COUNT(*) FILTER (WHERE yoe_extracted IS NOT NULL) AS entry_with_yoe_n,
               COUNT(*) FILTER (WHERE metro_area IS NOT NULL AND metro_area <> '') AS entry_with_metro_n,
               COUNT(*) FILTER (WHERE llm_extraction_coverage = 'labeled') AS entry_with_llm_text_n,
               COUNT(*) FILTER (WHERE description_core_llm IS NOT NULL AND description_core_llm <> '') AS entry_with_core_text_n,
               AVG(CASE WHEN yoe_extracted <= 2 THEN 1.0 END) AS share_yoe_le_2_among_yoe_non_null,
               AVG(CASE WHEN yoe_extracted <= 3 THEN 1.0 END) AS share_yoe_le_3_among_yoe_non_null
        {base}
          AND seniority_final = 'entry'
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchdf()
    entry_rows.to_csv(TAB_DIR / "T02_entry_sample_sizes_by_source_and_final_source.csv", index=False)

    entry_source_totals = (
        entry_rows.groupby("source", as_index=False)[
            ["entry_n", "entry_with_yoe_n", "entry_with_metro_n", "entry_with_llm_text_n", "entry_with_core_text_n"]
        ]
        .sum()
    )
    entry_source_totals.to_csv(TAB_DIR / "T02_entry_sample_sizes_by_source_total.csv", index=False)


if __name__ == "__main__":
    main()
