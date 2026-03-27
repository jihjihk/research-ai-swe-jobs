#!/usr/bin/env python3
"""
Temporary row-audit helper for a small-batch Stage 9 -> Stage 10 run.

Inputs:
  - sampled Stage 8 parquet
  - Stage 9 cleaned parquet
  - Stage 9 extraction-results parquet
  - Stage 10 integrated parquet

Outputs:
  - row-by-row audit parquet
  - concise markdown summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from io_utils import write_parquet_atomic, write_text_atomic


PROJECT_ROOT = Path(__file__).parent.parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
TMP_DIR = INTERMEDIATE_DIR / "tmp_llm_small_batch"

DEFAULT_STAGE8_PATH = TMP_DIR / "stage8_sample.parquet"
DEFAULT_STAGE9_CLEANED_PATH = TMP_DIR / "stage9_llm_cleaned_sample.parquet"
DEFAULT_STAGE9_RESULTS_PATH = TMP_DIR / "stage9_llm_extraction_results_sample.parquet"
DEFAULT_STAGE10_PATH = TMP_DIR / "stage10_llm_integrated_sample.parquet"
DEFAULT_AUDIT_OUTPUT = TMP_DIR / "stage9_stage10_row_audit.parquet"
DEFAULT_REPORT_OUTPUT = TMP_DIR / "stage9_stage10_audit_report.md"


def _row_audit_query(
    stage8_path: Path,
    stage9_cleaned_path: Path,
    stage9_results_path: Path,
    stage10_path: Path,
) -> str:
    return f"""
    WITH stage8 AS (
        SELECT
            uid,
            job_id,
            source,
            source_platform,
            period,
            title,
            company_name,
            seniority_final,
            swe_classification_tier,
            seniority_source,
            ghost_job_risk,
            is_english,
            is_swe,
            is_swe_adjacent,
            is_control,
            description,
            description_core
        FROM read_parquet('{stage8_path.as_posix()}')
    ),
    stage9_cleaned AS (
        SELECT
            uid,
            job_id,
            extraction_input_hash,
            eligible_for_extraction,
            selected_for_control_cohort,
            llm_text_skip_reason,
            llm_extraction_reason,
            description_core_llm
        FROM read_parquet('{stage9_cleaned_path.as_posix()}')
    ),
    stage9_results AS (
        SELECT
            extraction_input_hash,
            llm_model_extraction,
            llm_prompt_version_extraction,
            extraction_response_json
        FROM read_parquet('{stage9_results_path.as_posix()}')
    ),
    stage10 AS (
        SELECT
            uid,
            job_id,
            classification_input_hash,
            needs_llm_classification,
            llm_classification_reason,
            llm_model_classification,
            swe_classification_llm,
            seniority_llm,
            ghost_assessment_llm,
            yoe_min_years_llm
        FROM read_parquet('{stage10_path.as_posix()}')
    )
    SELECT
        stage8.uid,
        stage8.job_id,
        stage8.source,
        stage8.source_platform,
        stage8.period,
        stage8.title,
        stage8.company_name,
        stage8.seniority_final,
        stage8.swe_classification_tier,
        stage8.seniority_source,
        stage8.ghost_job_risk,
        coalesce(stage8.is_english, FALSE) AS is_english,
        coalesce(stage8.is_swe, FALSE) AS is_swe,
        coalesce(stage8.is_swe_adjacent, FALSE) AS is_swe_adjacent,
        coalesce(stage8.is_control, FALSE) AS is_control,
        stage8.description,
        stage8.description_core,
        length(coalesce(stage8.description, '')) AS raw_description_chars,
        array_length(string_split(trim(coalesce(stage8.description, '')), ' ')) AS raw_description_words,
        length(coalesce(stage8.description_core, '')) AS rule_core_chars,
        stage9_cleaned.uid IS NOT NULL AS present_in_stage9,
        stage10.uid IS NOT NULL AS present_in_stage10,
        stage9_cleaned.extraction_input_hash,
        coalesce(stage9_cleaned.eligible_for_extraction, FALSE) AS eligible_for_extraction,
        coalesce(stage9_cleaned.selected_for_control_cohort, FALSE) AS selected_for_control_cohort,
        stage9_cleaned.llm_text_skip_reason,
        stage9_cleaned.llm_extraction_reason,
        stage9_cleaned.description_core_llm,
        length(coalesce(stage9_cleaned.description_core_llm, '')) AS llm_core_chars,
        stage9_results.llm_model_extraction,
        stage9_results.llm_prompt_version_extraction,
        stage9_results.extraction_response_json,
        CASE
            WHEN trim(coalesce(stage9_cleaned.description_core_llm, '')) = '' AND stage9_cleaned.description_core_llm IS NULL THEN 'null'
            WHEN trim(coalesce(stage9_cleaned.description_core_llm, '')) = '' THEN 'empty'
            ELSE 'nonempty'
        END AS description_core_llm_state,
        CASE
            WHEN trim(coalesce(stage9_cleaned.description_core_llm, '')) <> '' THEN 'description_core_llm'
            WHEN trim(coalesce(stage8.description_core, '')) <> '' THEN 'description_core'
            WHEN trim(coalesce(stage8.description, '')) <> '' THEN 'description'
            ELSE 'empty'
        END AS classification_input_source_expected,
        stage10.classification_input_hash,
        coalesce(stage10.needs_llm_classification, FALSE) AS needs_llm_classification,
        stage10.llm_classification_reason,
        stage10.llm_model_classification,
        stage10.swe_classification_llm,
        stage10.seniority_llm,
        stage10.ghost_assessment_llm,
        stage10.yoe_min_years_llm
    FROM stage8
    LEFT JOIN stage9_cleaned USING (uid)
    LEFT JOIN stage9_results USING (extraction_input_hash)
    LEFT JOIN stage10 USING (uid)
    ORDER BY stage8.job_id, stage8.uid
    """


def _parse_extraction_payload(raw: str | None) -> tuple[str | None, str | None]:
    if raw is None or not str(raw).strip():
        return None, None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return "invalid_json", None
    status = payload.get("task_status")
    reason = payload.get("reason")
    return None if status is None else str(status), None if reason is None else str(reason)


def build_row_audit(
    stage8_path: Path,
    stage9_cleaned_path: Path,
    stage9_results_path: Path,
    stage10_path: Path,
) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        df = con.execute(
            _row_audit_query(stage8_path, stage9_cleaned_path, stage9_results_path, stage10_path)
        ).fetchdf()
    finally:
        con.close()

    extraction_status = []
    extraction_reason = []
    for raw in df["extraction_response_json"].tolist():
        status, reason = _parse_extraction_payload(raw)
        extraction_status.append(status)
        extraction_reason.append(reason)
    df["extraction_status"] = extraction_status
    df["extraction_status_reason"] = extraction_reason

    llm_core_nonempty = df["description_core_llm_state"].eq("nonempty")
    routed_in_scope = df["llm_extraction_reason"].fillna("").eq("routed")
    long_routed = routed_in_scope & df["raw_description_words"].fillna(0).ge(80)

    df["flag_missing_stage9_row"] = ~df["present_in_stage9"].fillna(False)
    df["flag_missing_stage10_row"] = ~df["present_in_stage10"].fillna(False)
    df["flag_long_routed_missing_cleaned_text"] = long_routed & ~llm_core_nonempty
    df["flag_extraction_missing_result_payload"] = routed_in_scope & df["extraction_response_json"].isna()
    df["flag_invalid_extraction_fallback"] = (
        routed_in_scope
        & df["extraction_status"].fillna("").isin(["cannot_complete", "invalid_json"])
        & llm_core_nonempty
    )
    df["flag_extraction_ok_but_cleaned_text_missing"] = (
        routed_in_scope
        & df["extraction_status"].fillna("").eq("ok")
        & ~llm_core_nonempty
    )
    df["flag_short_skip_without_empty_cleaned_text"] = (
        df["llm_text_skip_reason"].fillna("").eq("short_description_under_15_words")
        & df["description_core_llm_state"].ne("empty")
    )
    df["flag_classification_unexpectedly_missing"] = (
        df["needs_llm_classification"].fillna(False)
        & df["swe_classification_llm"].isna()
        & df["seniority_llm"].isna()
        & df["ghost_assessment_llm"].isna()
    )
    df["flag_unrouted_with_llm_outputs"] = (
        ~df["needs_llm_classification"].fillna(False)
        & (
            df["swe_classification_llm"].notna()
            | df["seniority_llm"].notna()
            | df["ghost_assessment_llm"].notna()
        )
    )
    df["flag_rule_skip_but_stage10_routed"] = (
        df["llm_text_skip_reason"].fillna("").eq("short_description_under_15_words")
        & df["needs_llm_classification"].fillna(False)
    )
    df["flag_llm_core_longer_than_raw"] = df["llm_core_chars"].fillna(0) > df["raw_description_chars"].fillna(0)

    issue_columns = [
        "flag_missing_stage9_row",
        "flag_missing_stage10_row",
        "flag_long_routed_missing_cleaned_text",
        "flag_extraction_missing_result_payload",
        "flag_invalid_extraction_fallback",
        "flag_extraction_ok_but_cleaned_text_missing",
        "flag_short_skip_without_empty_cleaned_text",
        "flag_classification_unexpectedly_missing",
        "flag_unrouted_with_llm_outputs",
        "flag_rule_skip_but_stage10_routed",
        "flag_llm_core_longer_than_raw",
    ]

    def collect_issue_tags(row: pd.Series) -> str:
        tags = [col.removeprefix("flag_") for col in issue_columns if bool(row[col])]
        return "|".join(tags)

    df["issue_tags"] = df.apply(collect_issue_tags, axis=1)
    df["issue_count"] = df[issue_columns].sum(axis=1).astype(int)
    return df


def _format_rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0/0 (0.0%)"
    return f"{numerator}/{denominator} ({(numerator / denominator) * 100:.1f}%)"


def build_report(df: pd.DataFrame, suspicious_limit: int) -> str:
    total_rows = len(df)
    extraction_routed = int(df["llm_extraction_reason"].fillna("").eq("routed").sum())
    short_skip = int(df["llm_text_skip_reason"].fillna("").eq("short_description_under_15_words").sum())
    control_rows = int(df["selected_for_control_cohort"].fillna(False).sum())
    extraction_ok = int(df["extraction_status"].fillna("").eq("ok").sum())
    extraction_cannot_complete = int(df["extraction_status"].fillna("").eq("cannot_complete").sum())
    classification_routed = int(df["needs_llm_classification"].fillna(False).sum())
    classification_with_outputs = int(
        (
            df["swe_classification_llm"].notna()
            | df["seniority_llm"].notna()
            | df["ghost_assessment_llm"].notna()
        ).sum()
    )
    suspicious = df.loc[df["issue_count"] > 0].copy()

    lines = [
        "# Stage 9/10 Small-Batch Audit",
        "",
        f"- Rows audited: `{total_rows}`",
        f"- Stage 9 routed extraction: `{_format_rate(extraction_routed, total_rows)}`",
        f"- Stage 9 short-description skips: `{_format_rate(short_skip, total_rows)}`",
        f"- Control-cohort selected rows: `{_format_rate(control_rows, total_rows)}`",
        f"- Extraction status ok: `{_format_rate(extraction_ok, total_rows)}`",
        f"- Extraction status cannot_complete: `{_format_rate(extraction_cannot_complete, total_rows)}`",
        f"- Stage 10 routed classification: `{_format_rate(classification_routed, total_rows)}`",
        f"- Rows with LLM classification outputs: `{_format_rate(classification_with_outputs, total_rows)}`",
        f"- Rows with any discrepancy flag: `{_format_rate(len(suspicious), total_rows)}`",
        "",
        "## Flag counts",
        "",
    ]

    for col in sorted([c for c in df.columns if c.startswith("flag_")]):
        lines.append(f"- `{col}`: `{int(df[col].fillna(False).sum())}`")

    lines.extend(["", "## Suspicious rows", ""])
    if suspicious.empty:
        lines.append("No discrepancy flags were raised.")
    else:
        suspicious = suspicious.sort_values(
            ["issue_count", "needs_llm_classification", "raw_description_words", "job_id"],
            ascending=[False, False, False, True],
        ).head(suspicious_limit)
        for row in suspicious.itertuples(index=False):
            lines.append(
                f"- `{row.job_id}` | {row.title} | {row.company_name} | tags=`{row.issue_tags}`"
            )
            lines.append(
                f"  Stage 9: extraction_reason=`{row.llm_extraction_reason}` skip=`{row.llm_text_skip_reason}` "
                f"control={bool(row.selected_for_control_cohort)} status=`{row.extraction_status}` "
                f"status_reason=`{row.extraction_status_reason}` llm_core_state=`{row.description_core_llm_state}`"
            )
            lines.append(
                f"  Stage 10: classify={bool(row.needs_llm_classification)} "
                f"reason=`{row.llm_classification_reason}` swe=`{row.swe_classification_llm}` "
                f"seniority=`{row.seniority_llm}` ghost=`{row.ghost_assessment_llm}` yoe=`{row.yoe_min_years_llm}`"
            )
            lines.append("")

    lines.extend(["## All rows", ""])
    for row in df.sort_values(["job_id", "uid"]).itertuples(index=False):
        lines.append(
            f"- `{row.job_id}` | {row.title} | {row.company_name} | "
            f"extract=`{row.llm_extraction_reason}` | skip=`{row.llm_text_skip_reason}` | "
            f"control={bool(row.selected_for_control_cohort)} | extraction_status=`{row.extraction_status}` | "
            f"llm_core=`{row.description_core_llm_state}` | class=`{row.llm_classification_reason}` | "
            f"swe=`{row.swe_classification_llm}` | seniority=`{row.seniority_llm}` | "
            f"ghost=`{row.ghost_assessment_llm}` | tags=`{row.issue_tags or 'ok'}`"
        )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a small-batch Stage 9 -> Stage 10 run")
    parser.add_argument("--stage8-input", type=Path, default=DEFAULT_STAGE8_PATH)
    parser.add_argument("--stage9-cleaned-input", type=Path, default=DEFAULT_STAGE9_CLEANED_PATH)
    parser.add_argument("--stage9-results-input", type=Path, default=DEFAULT_STAGE9_RESULTS_PATH)
    parser.add_argument("--stage10-input", type=Path, default=DEFAULT_STAGE10_PATH)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--suspicious-limit", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_df = build_row_audit(
        args.stage8_input,
        args.stage9_cleaned_input,
        args.stage9_results_input,
        args.stage10_input,
    )
    write_parquet_atomic(audit_df, args.audit_output)
    write_text_atomic(build_report(audit_df, suspicious_limit=args.suspicious_limit), args.report_output)


if __name__ == "__main__":
    main()
