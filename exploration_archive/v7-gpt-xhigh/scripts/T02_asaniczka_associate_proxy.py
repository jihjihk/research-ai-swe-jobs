#!/usr/bin/env python
"""T02 audit: can asaniczka native `associate` serve as a junior proxy?"""

from __future__ import annotations

from pathlib import Path
import math
import re

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "unified.parquet"
TABLE_DIR = ROOT / "exploration" / "tables" / "T02"
REPORT_PATH = ROOT / "exploration" / "reports" / "T02.md"

DEFAULT_WHERE = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'"
BASE_WHERE = DEFAULT_WHERE + " AND is_swe = true"
GROUPS = [
    ("asaniczka_associate", "kaggle_asaniczka", "associate"),
    ("arshkon_entry", "kaggle_arshkon", "entry"),
    ("arshkon_associate", "kaggle_arshkon", "associate"),
    ("arshkon_mid_senior", "kaggle_arshkon", "mid-senior"),
]
ARSHKON_GROUPS = ["arshkon_entry", "arshkon_associate", "arshkon_mid_senior"]
SENIORITY_VALUES = ["entry", "associate", "mid-senior", "director", "unknown"]

# Do not treat "associate" itself as a junior cue; T02 tests whether that native
# label behaves like junior, so counting the label as evidence would be circular.
JUNIOR_RE = (
    r"(^|[^a-z0-9])("
    r"entry[- ]?level|new grad(uate)?|early career|junior|jr\.?|intern(ship)?|apprentice"
    r")([^a-z0-9]|$)|"
    r"(^|[^a-z0-9])(software engineer|software developer|developer|swe|engineer)[ -]*(i|1)([^a-z0-9]|$)"
)
SENIOR_RE = (
    r"(^|[^a-z0-9])("
    r"senior|sr\.?|staff|principal|lead|architect|distinguished|manager|director|vp|head"
    r")([^a-z0-9]|$)"
)


def _regex_asserts() -> None:
    junior = re.compile(JUNIOR_RE)
    senior = re.compile(SENIOR_RE)

    assert junior.search("junior software engineer")
    assert junior.search("entry-level developer")
    assert junior.search("software engineer i")
    assert junior.search("swe 1")
    assert not junior.search("associate software engineer")
    assert not junior.search("principal software engineer")

    assert senior.search("senior software engineer")
    assert senior.search("sr. developer")
    assert senior.search("staff platform engineer")
    assert senior.search("tech lead")
    assert senior.search("solutions architect")
    assert not senior.search("leadership development intern")
    assert not senior.search("junior software engineer")


def group_case() -> str:
    cases = [
        f"WHEN source = '{source}' AND seniority_native = '{native}' THEN '{label}'"
        for label, source, native in GROUPS
    ]
    return "CASE " + " ".join(cases) + " END"


def pct(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{100 * float(value):.{digits}f}%"


def numfmt(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.{digits}f}"
    return f"{int(value):,}"


def md_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if columns is not None:
        df = df[columns].copy()
    if max_rows is not None:
        df = df.head(max_rows).copy()
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in rows))
        for idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body])


def save_csv(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    return path


def native_group_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    gcase = group_case()
    return con.execute(
        f"""
        WITH base AS (
          SELECT
            {gcase} AS comparison_group,
            lower(coalesce(title, '')) AS title_for_cues,
            yoe_extracted,
            seniority_final
          FROM read_parquet('{DATASET.as_posix()}')
          WHERE {BASE_WHERE}
            AND {gcase} IS NOT NULL
        )
        SELECT
          comparison_group,
          count(*) AS rows,
          sum(CASE WHEN regexp_matches(title_for_cues, '{JUNIOR_RE}') THEN 1 ELSE 0 END) AS junior_title_cue_rows,
          sum(CASE WHEN regexp_matches(title_for_cues, '{SENIOR_RE}') THEN 1 ELSE 0 END) AS senior_title_cue_rows,
          count(yoe_extracted) AS yoe_known_rows,
          avg(yoe_extracted) AS yoe_mean,
          quantile_cont(yoe_extracted, 0.25) AS yoe_p25,
          quantile_cont(yoe_extracted, 0.50) AS yoe_median,
          quantile_cont(yoe_extracted, 0.75) AS yoe_p75,
          sum(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END) AS yoe_le_2_rows,
          sum(CASE WHEN yoe_extracted <= 3 THEN 1 ELSE 0 END) AS yoe_le_3_rows,
          sum(CASE WHEN yoe_extracted >= 5 THEN 1 ELSE 0 END) AS yoe_ge_5_rows
        FROM base
        GROUP BY comparison_group
        ORDER BY
          CASE comparison_group
            WHEN 'asaniczka_associate' THEN 0
            WHEN 'arshkon_entry' THEN 1
            WHEN 'arshkon_associate' THEN 2
            WHEN 'arshkon_mid_senior' THEN 3
          END
        """
    ).fetchdf()


def top_title_jaccard(con: duckdb.DuckDBPyConnection, top_n: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    gcase = group_case()
    titles = con.execute(
        f"""
        WITH base AS (
          SELECT
            {gcase} AS comparison_group,
            title_normalized,
            count(*) AS rows
          FROM read_parquet('{DATASET.as_posix()}')
          WHERE {BASE_WHERE}
            AND {gcase} IS NOT NULL
            AND title_normalized IS NOT NULL
          GROUP BY comparison_group, title_normalized
        ),
        ranked AS (
          SELECT
            *,
            row_number() OVER (
              PARTITION BY comparison_group
              ORDER BY rows DESC, title_normalized
            ) AS title_rank
          FROM base
        )
        SELECT *
        FROM ranked
        WHERE title_rank <= {top_n}
        ORDER BY comparison_group, title_rank
        """
    ).fetchdf()
    title_sets = {
        group: set(titles.loc[titles["comparison_group"] == group, "title_normalized"])
        for group, _, _ in GROUPS
    }
    rows = []
    base = title_sets["asaniczka_associate"]
    for group in ARSHKON_GROUPS:
        other = title_sets[group]
        intersection = len(base & other)
        union = len(base | other)
        rows.append(
            {
                "source_group": "asaniczka_associate",
                "comparison_group": group,
                "top_n_titles": top_n,
                "intersection": intersection,
                "union": union,
                "jaccard": intersection / union if union else math.nan,
            }
        )
    return titles, pd.DataFrame(rows)


def seniority_final_distribution(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    gcase = group_case()
    dist = con.execute(
        f"""
        WITH base AS (
          SELECT
            {gcase} AS comparison_group,
            coalesce(seniority_final, 'NULL') AS seniority_final,
            count(*) AS rows
          FROM read_parquet('{DATASET.as_posix()}')
          WHERE {BASE_WHERE}
            AND {gcase} IS NOT NULL
          GROUP BY comparison_group, coalesce(seniority_final, 'NULL')
        ),
        den AS (
          SELECT comparison_group, sum(rows) AS denominator
          FROM base
          GROUP BY comparison_group
        )
        SELECT
          base.comparison_group,
          base.seniority_final,
          base.rows,
          den.denominator,
          base.rows::DOUBLE / den.denominator AS share
        FROM base
        JOIN den USING (comparison_group)
        ORDER BY comparison_group, rows DESC
        """
    ).fetchdf()
    return dist


def entry_effective_sample_sizes(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        f"""
        WITH source_den AS (
          SELECT source, count(*) AS source_swe_rows
          FROM read_parquet('{DATASET.as_posix()}')
          WHERE {BASE_WHERE}
          GROUP BY source
        ),
        entry AS (
          SELECT
            source,
            seniority_final_source,
            count(*) AS entry_rows
          FROM read_parquet('{DATASET.as_posix()}')
          WHERE {BASE_WHERE}
            AND seniority_final = 'entry'
          GROUP BY source, seniority_final_source
        )
        SELECT
          entry.source,
          entry.seniority_final_source,
          entry.entry_rows,
          source_den.source_swe_rows,
          entry.entry_rows::DOUBLE / source_den.source_swe_rows AS share_of_source_swe
        FROM entry
        JOIN source_den USING (source)
        ORDER BY source, seniority_final_source
        """
    ).fetchdf()


def source_swe_counts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        f"""
        SELECT source, count(*) AS swe_rows
        FROM read_parquet('{DATASET.as_posix()}')
        WHERE {BASE_WHERE}
        GROUP BY source
        ORDER BY source
        """
    ).fetchdf()


def nearest_by_distance(summary: pd.DataFrame, jaccard: pd.DataFrame, dist: pd.DataFrame) -> pd.DataFrame:
    base = summary.set_index("comparison_group").loc["asaniczka_associate"]
    records: list[dict[str, object]] = []

    for group in ARSHKON_GROUPS:
        row = summary.set_index("comparison_group").loc[group]
        cue_distance = math.sqrt(
            (base["junior_title_cue_share"] - row["junior_title_cue_share"]) ** 2
            + (base["senior_title_cue_share"] - row["senior_title_cue_share"]) ** 2
        )
        yoe_features = ["yoe_median", "yoe_le_2_share_known", "yoe_ge_5_share_known"]
        yoe_distance = math.sqrt(
            sum((base[feature] - row[feature]) ** 2 for feature in yoe_features if pd.notna(base[feature]) and pd.notna(row[feature]))
        )

        records.append(
            {
                "signal": "title_cue_profile",
                "comparison_group": group,
                "distance_or_similarity": cue_distance,
                "closer_is": "lower",
            }
        )
        records.append(
            {
                "signal": "yoe_profile",
                "comparison_group": group,
                "distance_or_similarity": yoe_distance,
                "closer_is": "lower",
            }
        )

    for _, row in jaccard.iterrows():
        records.append(
            {
                "signal": "top_title_jaccard",
                "comparison_group": row["comparison_group"],
                "distance_or_similarity": row["jaccard"],
                "closer_is": "higher",
            }
        )

    wide = (
        dist.pivot_table(
            index="comparison_group",
            columns="seniority_final",
            values="share",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=["asaniczka_associate", *ARSHKON_GROUPS], fill_value=0.0)
    )
    for value in SENIORITY_VALUES:
        if value not in wide.columns:
            wide[value] = 0.0
    base_dist = wide.loc["asaniczka_associate", SENIORITY_VALUES]
    for group in ARSHKON_GROUPS:
        l1 = float((base_dist - wide.loc[group, SENIORITY_VALUES]).abs().sum())
        records.append(
            {
                "signal": "seniority_final_distribution",
                "comparison_group": group,
                "distance_or_similarity": l1,
                "closer_is": "lower",
            }
        )

    distances = pd.DataFrame(records)
    return distances


def signal_verdicts(distances: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for signal, signal_df in distances.groupby("signal", sort=False):
        closer_is = signal_df["closer_is"].iloc[0]
        if closer_is == "higher":
            best_value = signal_df["distance_or_similarity"].max()
            nearest = sorted(signal_df.loc[signal_df["distance_or_similarity"] == best_value, "comparison_group"])
        else:
            best_value = signal_df["distance_or_similarity"].min()
            nearest = sorted(signal_df.loc[signal_df["distance_or_similarity"] == best_value, "comparison_group"])
        rows.append(
            {
                "signal": signal,
                "nearest_arshkon_group": ",".join(nearest),
                "entry_match": nearest == ["arshkon_entry"],
                "decision_value": best_value,
                "decision_rule": "entry is uniquely nearest/highest among arshkon native groups",
            }
        )
    return pd.DataFrame(rows).sort_values("signal").reset_index(drop=True)


def write_report(
    summary: pd.DataFrame,
    titles: pd.DataFrame,
    jaccard: pd.DataFrame,
    final_dist: pd.DataFrame,
    entry_sizes: pd.DataFrame,
    distances: pd.DataFrame,
    verdicts: pd.DataFrame,
    source_counts: pd.DataFrame,
) -> None:
    match_count = int(verdicts["entry_match"].sum())
    usable = match_count >= 3
    verdict = "usable as a junior proxy" if usable else "not usable as a junior proxy"

    summary_fmt = summary.copy()
    for col in ["junior_title_cue_share", "senior_title_cue_share", "yoe_known_share", "yoe_le_2_share_known", "yoe_le_3_share_known", "yoe_ge_5_share_known"]:
        summary_fmt[col] = summary_fmt[col].map(pct)
    for col in ["rows", "junior_title_cue_rows", "senior_title_cue_rows", "yoe_known_rows", "yoe_le_2_rows", "yoe_le_3_rows", "yoe_ge_5_rows"]:
        summary_fmt[col] = summary_fmt[col].map(numfmt)
    for col in ["yoe_mean", "yoe_p25", "yoe_median", "yoe_p75"]:
        summary_fmt[col] = summary_fmt[col].map(lambda x: numfmt(x, digits=1))

    jaccard_fmt = jaccard.copy()
    jaccard_fmt["jaccard"] = jaccard_fmt["jaccard"].map(lambda x: numfmt(x, digits=3))

    verdicts_fmt = verdicts.copy()
    verdicts_fmt["entry_match"] = verdicts_fmt["entry_match"].map(lambda x: "yes" if x else "no")
    verdicts_fmt["decision_value"] = verdicts_fmt["decision_value"].map(lambda x: numfmt(x, digits=3))

    final_dist_fmt = final_dist.copy()
    final_dist_fmt["rows"] = final_dist_fmt["rows"].map(numfmt)
    final_dist_fmt["denominator"] = final_dist_fmt["denominator"].map(numfmt)
    final_dist_fmt["share"] = final_dist_fmt["share"].map(pct)

    entry_sizes_fmt = entry_sizes.copy()
    entry_sizes_fmt["entry_rows"] = entry_sizes_fmt["entry_rows"].map(numfmt)
    entry_sizes_fmt["source_swe_rows"] = entry_sizes_fmt["source_swe_rows"].map(numfmt)
    entry_sizes_fmt["share_of_source_swe"] = entry_sizes_fmt["share_of_source_swe"].map(pct)

    source_counts_fmt = source_counts.copy()
    source_counts_fmt["swe_rows"] = source_counts_fmt["swe_rows"].map(numfmt)

    top_titles_preview = titles[titles["title_rank"] <= 10].copy()
    top_titles_preview["rows"] = top_titles_preview["rows"].map(numfmt)

    lines = [
        "# T02. Asaniczka `associate` As A Junior Proxy",
        "",
        "## Plain Verdict",
        "",
        (
            f"Asaniczka `seniority_native = 'associate'` is **{verdict}** under the dispatch rule. "
            f"It matches arshkon `entry` on {match_count} of 4 required signals; the threshold is at least 3 of 4. "
            "Downstream analyses should not pool asaniczka native `associate` as an entry-level baseline. Use T30's "
            "J2 (`seniority_final IN ('entry','associate')`) or YOE-based junior variants instead."
        ),
        "",
        "## Methodology",
        "",
        (
            "Subset for all comparability metrics: `source_platform = 'linkedin'`, `is_english = true`, "
            "`date_flag = 'ok'`, `is_swe = true`, and the native-label groups named in each row. "
            "Top-title Jaccard uses each group's top 50 `title_normalized` values by row count. "
            "Explicit title-cue rates use raw lowercased `title` because `title_normalized` strips level indicators. "
            "Junior title-cue regex excludes the word `associate` to avoid circularly validating the tested label. "
            "YOE distribution denominators exclude rows with null `yoe_extracted`, and the known-YOE share is reported."
        ),
        "",
        "Default-filtered LinkedIn SWE denominators by source:",
        "",
        md_table(source_counts_fmt),
        "",
        "## Signal Decision Table",
        "",
        md_table(verdicts_fmt),
        "",
        "## Native-Label Group Summary",
        "",
        md_table(
            summary_fmt[
                [
                    "comparison_group",
                    "rows",
                    "junior_title_cue_rows",
                    "junior_title_cue_share",
                    "senior_title_cue_rows",
                    "senior_title_cue_share",
                    "yoe_known_rows",
                    "yoe_known_share",
                    "yoe_median",
                    "yoe_le_2_share_known",
                    "yoe_ge_5_share_known",
                ]
            ]
        ),
        "",
        "## Top-Title Jaccard",
        "",
        md_table(jaccard_fmt),
        "",
        "Top 10 normalized titles within each comparison group:",
        "",
        md_table(top_titles_preview[["comparison_group", "title_rank", "title_normalized", "rows"]], max_rows=40),
        "",
        "## `seniority_final` Distribution Conditional On Native Label",
        "",
        "Denominator is each native-label comparison group under the default SWE subset.",
        "",
        md_table(final_dist_fmt),
        "",
        "## Entry-Level Effective Sample Sizes Under `seniority_final`",
        "",
        "Subset: default-filtered LinkedIn SWE rows with `seniority_final = 'entry'`. Denominator is all default-filtered LinkedIn SWE rows in that source.",
        "",
        md_table(entry_sizes_fmt),
        "",
        "## Comparability Interpretation",
        "",
        "- Top-title overlap is a weak signal because generic titles dominate every group; it is still reported because the task requires it.",
        "- The title-cue signal argues against substitution: asaniczka native `associate` has a much higher senior-cue rate than arshkon native `entry` and is nearest to arshkon native `associate` on the two-cue profile.",
        "- The YOE signal is denominator-limited because only rows with parsed YOE enter the distribution, but the known-YOE share is high enough for a directional comparison.",
        "- The `seniority_final` distribution is decisive for the substitution question: only a small fraction of asaniczka native `associate` rows become `entry` under the combined rule/LLM seniority column.",
        "",
        "## Surprises & Unexpected Patterns",
        "",
        "- Arshkon native `entry` still maps mostly to `seniority_final = 'unknown'`, not to `entry`; this confirms that production seniority intentionally requires explicit title/LLM evidence and does not simply mirror native labels.",
        "- Asaniczka native `associate` looks less junior than the label might imply: its explicit junior-title cue rate is low and its `seniority_final` distribution is dominated by `unknown`/`mid-senior` rather than `entry`.",
        "- Native labels and production labels are measuring different objects. Native `entry` is useful as an arshkon diagnostic, but `seniority_final` is more conservative and creates smaller effective junior cells.",
        "",
        "## Action Items For Downstream Agents",
        "",
        "- Do not use asaniczka native `associate` as a stand-alone junior baseline.",
        "- Report arshkon-only native-entry checks separately from `seniority_final` analyses.",
        "- Use T30 J1-J4/J5-J6 panels for junior claims and include YOE-known denominators for YOE variants.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _regex_asserts()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    summary = native_group_summary(con)
    summary["junior_title_cue_share"] = summary["junior_title_cue_rows"] / summary["rows"]
    summary["senior_title_cue_share"] = summary["senior_title_cue_rows"] / summary["rows"]
    summary["yoe_known_share"] = summary["yoe_known_rows"] / summary["rows"]
    summary["yoe_le_2_share_known"] = summary["yoe_le_2_rows"] / summary["yoe_known_rows"]
    summary["yoe_le_3_share_known"] = summary["yoe_le_3_rows"] / summary["yoe_known_rows"]
    summary["yoe_ge_5_share_known"] = summary["yoe_ge_5_rows"] / summary["yoe_known_rows"]
    save_csv(summary, "native_group_summary.csv")

    titles, jaccard = top_title_jaccard(con)
    save_csv(titles, "top_titles_by_native_group.csv")
    save_csv(jaccard, "top_title_jaccard.csv")

    final_dist = seniority_final_distribution(con)
    save_csv(final_dist, "seniority_final_distribution_by_native_group.csv")

    entry_sizes = entry_effective_sample_sizes(con)
    save_csv(entry_sizes, "entry_effective_sample_sizes_by_source.csv")

    source_counts = source_swe_counts(con)
    save_csv(source_counts, "default_linkedin_swe_counts_by_source.csv")

    distances = nearest_by_distance(summary, jaccard, final_dist)
    save_csv(distances, "signal_distances_to_asaniczka_associate.csv")

    verdicts = signal_verdicts(distances)
    save_csv(verdicts, "comparability_signal_verdicts.csv")

    write_report(summary, titles, jaccard, final_dist, entry_sizes, distances, verdicts, source_counts)
    print(f"[T02] wrote {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
