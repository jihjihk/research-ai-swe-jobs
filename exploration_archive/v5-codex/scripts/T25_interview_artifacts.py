#!/usr/bin/env python3
"""Build interview elicitation artifacts for Wave 4 T25.

The outputs are discussion prompts, not proof. They are intentionally
truncated and selection-biased toward the strongest contradictions in the
exploration findings.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "T25"
DATA_PATH = ROOT / ".." / "data" / "unified.parquet"


def assert_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required if col not in frame.columns]
    assert not missing, f"{label} missing columns: {missing}"


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def excerpt(value: str | None, width: int = 460) -> str:
    normalized = normalize_text(value)
    if not normalized:
        return "[no text available]"
    return textwrap.shorten(normalized, width=width, placeholder="…")


def wrapped(value: str | None, width: int = 90) -> str:
    return "\n".join(textwrap.fill(line, width=width) for line in normalize_text(value).split("\n"))


def load_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    assert not frame.empty, f"{path.name} is empty"
    return frame


def select_junior_prompts(t22: pd.DataFrame) -> pd.DataFrame:
    assert_columns(
        t22,
        [
            "uid",
            "source",
            "period",
            "title",
            "company_name_canonical",
            "yoe_extracted",
            "kitchen_sink_product",
            "aspiration_ratio",
            "yoe_scope_mismatch",
            "degree_contra",
        ],
        "T22 top ghost postings",
    )
    title_mask = t22["title"].str.contains(r"(?i)\b(?:jr|junior|intern|entry|graduate)\b", regex=True, na=False)
    frame = (
        t22.loc[title_mask]
        .sort_values(["kitchen_sink_product", "aspiration_ratio", "yoe_scope_mismatch"], ascending=[False, False, False])
        .head(5)
        .copy()
    )
    assert len(frame) >= 3, "expected at least three junior-like examples from T22"
    return frame


def fetch_postings(con: duckdb.DuckDBPyConnection, uids: list[str] | None = None, company: str | None = None, period_prefix: str | None = None) -> pd.DataFrame:
    filters: list[str] = [
        "source_platform = 'linkedin'",
        "is_english = true",
        "date_flag = 'ok'",
    ]
    params: list[object] = []
    if uids is not None:
        filters.append(f"uid IN ({', '.join(['?'] * len(uids))})")
        params.extend(uids)
    if company is not None:
        filters.append("company_name_canonical = ?")
        params.append(company)
    if period_prefix is not None:
        filters.append("period LIKE ?")
        params.append(f"{period_prefix}%")

    query = f"""
        SELECT
            uid,
            company_name_canonical,
            period,
            title,
            seniority_final,
            seniority_final_source,
            yoe_extracted,
            description_length,
            llm_extraction_coverage,
            coalesce(nullif(description_core_llm, ''), description, description_raw) AS posting_text
        FROM read_parquet(?)
        WHERE {" AND ".join(filters)}
        ORDER BY description_length DESC, uid
    """
    params = [str(DATA_PATH)] + params
    return con.execute(query, params).fetchdf()


def pick_company_posting(con: duckdb.DuckDBPyConnection, company: str, period_prefix: str) -> pd.Series:
    frame = fetch_postings(con, company=company, period_prefix=period_prefix)
    assert not frame.empty, f"no rows found for {company} in {period_prefix}"
    return frame.iloc[0]


def render_text_card(ax, header: str, meta: str, body: str, subtitle: str | None = None, accent: str = "#1f4e79") -> None:
    ax.set_axis_off()
    ax.set_facecolor("#fbfbfc")
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#fbfbfc", edgecolor="#d0d7de", linewidth=1.0, zorder=0))
    ax.add_patch(Rectangle((0, 0), 0.016, 1, facecolor=accent, edgecolor=accent, zorder=1))
    ax.text(0.035, 0.94, header, fontsize=11.5, weight="bold", ha="left", va="top", transform=ax.transAxes)
    if subtitle:
        ax.text(0.035, 0.885, subtitle, fontsize=9.2, color="#555", ha="left", va="top", transform=ax.transAxes)
    ax.text(0.035, 0.82, meta, fontsize=8.6, color="#333", ha="left", va="top", transform=ax.transAxes)
    ax.text(
        0.035,
        0.72,
        wrapped(body, width=98),
        fontsize=8.3,
        color="#111",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )


def render_junior_prompt_figure(out_path: Path, examples: pd.DataFrame, postings: pd.DataFrame) -> None:
    merged = postings.set_index("uid")
    fig, axes = plt.subplots(len(examples), 1, figsize=(12.5, 3.15 * len(examples)))
    if len(examples) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, examples.iterrows()):
        uid = row["uid"]
        posting = merged.loc[uid]
        header = f'{row["title"]}  |  {row["company_name_canonical"]}  |  {row["period"]}'
        meta = (
            f"YOE {row['yoe_extracted']}  |  kitchen-sink {row['kitchen_sink_product']}  |  "
            f"aspiration ratio {row['aspiration_ratio']:.2f}  |  yoe/scope mismatch {int(row['yoe_scope_mismatch'])}"
        )
        subtitle = "prompt for reaction: entry label, YOE floor, scope bundling, and AI language do not line up cleanly"
        body = excerpt(posting["posting_text"], width=420)
        render_text_card(ax, header, meta, body, subtitle=subtitle, accent="#8b1e3f")
    fig.suptitle("T25A. Junior-like postings that look overpacked or aspirational", fontsize=15, weight="bold", y=0.995)
    fig.text(0.02, 0.012, "These are discussion prompts, not proof. Text is truncated to keep the examples readable.", fontsize=8.5, color="#555")
    fig.tight_layout(rect=[0, 0.025, 1, 0.985])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_company_pairs(out_path: Path, pairs: list[dict], con: duckdb.DuckDBPyConnection, change_profile: pd.DataFrame) -> None:
    change_profile = change_profile.set_index("company_name_canonical")
    fig, axes = plt.subplots(len(pairs), 2, figsize=(16, 4.15 * len(pairs)))
    if len(pairs) == 1:
        axes = [axes]  # type: ignore[assignment]

    for i, pair in enumerate(pairs):
        company = pair["company"]
        cluster = pair["cluster"]
        profile = change_profile.loc[company]
        row_label = (
            f"{company}  |  {cluster}  |  ΔAI {profile['delta_ai_any_share']:+.2f}  "
            f"Δscope {profile['delta_scope_any_share']:+.2f}  Δlen {profile['delta_clean_len_mean']:+.0f}"
        )
        left = axes[i][0]
        right = axes[i][1]
        left.text(0.0, 1.12, row_label, transform=left.transAxes, fontsize=11.2, weight="bold", ha="left", va="bottom")
        left_row = pick_company_posting(con, company, "2024-04")
        right_row = pick_company_posting(con, company, "2026-")
        left_header = f"2024-04  |  {left_row['title']}"
        right_header = f"{right_row['period']}  |  {right_row['title']}"
        left_meta = f"seniority {left_row['seniority_final']} ({left_row['seniority_final_source']})  |  YOE {left_row['yoe_extracted']}  |  length {int(left_row['description_length'])}"
        right_meta = f"seniority {right_row['seniority_final']} ({right_row['seniority_final_source']})  |  YOE {right_row['yoe_extracted']}  |  length {int(right_row['description_length'])}"
        subtitle = "prompt for reaction: same employer, different document shape, different requirement bundle"
        render_text_card(left, left_header, left_meta, excerpt(left_row["posting_text"], width=300), subtitle=subtitle, accent="#1f77b4")
        render_text_card(right, right_header, right_meta, excerpt(right_row["posting_text"], width=300), subtitle=subtitle, accent="#2ca02c")

    fig.suptitle("T25B. Same-company pairs that show posting strategy change over time", fontsize=15, weight="bold", y=0.996)
    fig.text(0.02, 0.008, "Each row uses the longest posting found in the company-period slice. These are prompts for discussion, not estimates.", fontsize=8.5, color="#555")
    fig.tight_layout(rect=[0, 0.02, 1, 0.985])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_junior_trend(out_path: Path, t08: pd.DataFrame) -> None:
    assert_columns(
        t08,
        ["period", "entry_share_all", "yoe_le2_all", "entry_share_known", "yoe_le2_known"],
        "T08 junior trends",
    )
    frame = t08.copy()
    frame["period_dt"] = pd.to_datetime(frame["period"] + "-01")
    release_points = [
        ("2024-03-01", "Claude 3"),
        ("2024-05-13", "GPT-4o"),
        ("2024-06-20", "Claude 3.5 Sonnet"),
        ("2024-09-12", "o1"),
        ("2024-12-26", "DeepSeek V3"),
        ("2025-02-14", "Claude 3.5 MAX"),
        ("2025-02-13", "GPT-4.5"),
        ("2025-04-15", "Claude 3.6 Sonnet"),
        ("2025-09-15", "Claude 4 Opus"),
        ("2025-10-15", "Claude 4.5 Haiku"),
        ("2026-03-25", "Gemini 2.5 Pro"),
    ]
    fig = plt.figure(figsize=(13.5, 8.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.25], hspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)

    ax.plot(frame["period_dt"], frame["entry_share_all"] * 100, marker="o", linewidth=2.5, color="#8b1e3f", label="explicit entry share")
    ax.plot(frame["period_dt"], frame["yoe_le2_all"] * 100, marker="o", linewidth=2.5, color="#1f77b4", label="YOE <= 2 share")
    for x, y in zip(frame["period_dt"], frame["entry_share_all"] * 100):
        ax.text(x, y + 0.35, f"{y:.1f}%", fontsize=8.5, ha="center", va="bottom", color="#8b1e3f")
    for x, y in zip(frame["period_dt"], frame["yoe_le2_all"] * 100):
        ax.text(x, y + 0.35, f"{y:.1f}%", fontsize=8.5, ha="center", va="bottom", color="#1f77b4")

    ax.set_ylabel("Share of SWE rows (%)")
    ax.set_ylim(0, max(frame["yoe_le2_all"].max(), frame["entry_share_all"].max()) * 100 + 4)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("T25C. Junior-share trend plot with model-release annotations", loc="left", fontsize=15, weight="bold", pad=12)
    ax.text(
        0.0,
        1.02,
        "The explicit label stays conservative while the YOE proxy remains much broader.",
        transform=ax.transAxes,
        fontsize=9.6,
        color="#444",
        ha="left",
    )

    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(False)
    for date_str, label in release_points:
        dt = pd.Timestamp(date_str)
        if dt < frame["period_dt"].min() - pd.Timedelta(days=20) or dt > frame["period_dt"].max() + pd.Timedelta(days=40):
            continue
        ax2.axvline(dt, color="#999", linewidth=0.8, alpha=0.45)
        ax2.text(dt, 0.18, label, rotation=90, ha="center", va="bottom", fontsize=7.3, color="#555")
    ax2.text(
        0.0,
        0.83,
        "Model-release windows from the task reference: GPT-4o, Claude 3.5 Sonnet, o1, DeepSeek V3, Claude 3.5 MAX, GPT-4.5, Claude 3.6 Sonnet, Claude 4 Opus, Claude 4.5 Haiku, Gemini 2.5 Pro.",
        transform=ax2.transAxes,
        fontsize=8.5,
        color="#555",
        ha="left",
        va="top",
        wrap=True,
    )
    ax2.set_xlabel("Period")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=0)
    fig.text(0.02, 0.02, "Prompt for reaction: does the apparent junior shift track model release timing, or merely posting form and measurement choice?", fontsize=8.6, color="#555")
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_senior_archetype(out_path: Path, t21: pd.DataFrame) -> None:
    assert_columns(
        t21,
        ["period_group", "seniority_final", "management_strict_density", "orchestration_strict_density", "strategic_strict_density"],
        "T21 senior archetype profiles",
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=True)
    order = ["2024", "2026"]
    measures = [
        ("management_strict_density", "strict management", "#7f7f7f"),
        ("orchestration_strict_density", "strict orchestration", "#1f77b4"),
        ("strategic_strict_density", "strict strategy", "#2ca02c"),
    ]
    for ax, seniority in zip(axes, ["mid-senior", "director"]):
        subset = t21[t21["seniority_final"] == seniority].copy()
        subset["period_group"] = subset["period_group"].astype(str)
        subset = subset.set_index("period_group")
        x = range(len(measures))
        width = 0.34
        for offset, period in enumerate(order):
            values = [subset.loc[period, key] for key, _, _ in measures]
            ax.bar([i + (offset - 0.5) * width for i in x], values, width=width, label=period)
        ax.set_xticks(list(x))
        ax.set_xticklabels([label for _, label, _ in measures], rotation=0)
        ax.set_title(seniority.replace("-", " ").title(), fontsize=12.5, weight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(0, 0.45 if seniority == "mid-senior" else 0.48)
    axes[0].set_ylabel("Density per posting")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("T25D. Senior-role profiles: orchestration grows, management holds", fontsize=15, weight="bold", y=0.99)
    fig.text(0.02, 0.015, "Uses strict management only; broad management was demoted to sensitivity in the exploration.", fontsize=8.6, color="#555")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_divergence_chart(out_path: Path, core: pd.DataFrame, raw: pd.DataFrame, bench: pd.DataFrame) -> None:
    assert_columns(core, ["text_source", "ai_tool_rate", "any_ai_rate"], "T23 core text-source summary")
    assert_columns(raw, ["text_source", "ai_tool_rate", "any_ai_rate"], "T23 raw text-source summary")
    assert_columns(bench, ["benchmark", "benchmark_rate"], "T23 benchmark sensitivity")

    rows = [
        ("LinkedIn core AI-tool", float(core.loc[core["text_source"] == "llm_core", "ai_tool_rate"].iloc[0])),
        ("LinkedIn raw AI-tool", float(raw.loc[raw["text_source"] == "raw_full", "ai_tool_rate"].iloc[0])),
    ]
    bench_rows = [
        ("StackOverflow 2024: AI-assisted tech at work", 0.324),
        ("StackOverflow 2025: daily AI-tool use", 0.51),
        ("StackOverflow 2025: use or plan to use", 0.84),
        ("GitHub US 2024: AI coding tools", 0.99),
    ]
    values = rows + bench_rows
    labels = [label for label, _ in values]
    rates = [rate * 100 for _, rate in values]
    colors = ["#8b1e3f", "#c43d3d", "#bbbbbb", "#bbbbbb", "#bbbbbb", "#bbbbbb"]

    fig, ax = plt.subplots(figsize=(12.8, 6.5))
    y = list(range(len(values)))
    ax.barh(y, rates, color=colors, edgecolor="#333", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlabel("Share mentioning AI tools / AI use (%)")
    ax.set_title("T25E. The divergence claim flips with the benchmark", loc="left", fontsize=15, weight="bold", pad=16)
    fig.text(
        0.02,
        0.905,
        "Core LinkedIn AI-tool rate is 30.3%; raw-text sensitivity is 40.7%. Whether that is above workers depends on the benchmark chosen.",
        fontsize=9.1,
        color="#444",
        ha="left",
    )
    for yi, rate in zip(y, rates):
        ax.text(rate + 1.0, yi, f"{rate:.1f}%", va="center", fontsize=8.6, color="#222")
    fig.text(0.02, 0.015, "Prompt for reaction: is this a labor-market gap, a benchmark problem, or a posting-language problem?", fontsize=8.6, color="#555")
    fig.tight_layout(rect=[0, 0.04, 1, 0.88])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_readme(out_path: Path) -> None:
    out_path.write_text(
        """# T25 Interview Elicitation Artifacts

These figures are prompts for discussion, not proof.

Selection rules:

- `T25A` uses the most scope- and aspiration-heavy junior-like postings surfaced by T22.
- `T25B` uses one representative pair from each of the four T16 company trajectories.
- `T25C` annotates the junior-share trend from T08 with model-release windows supplied by the task reference.
- `T25D` uses the strict management / orchestration / strategy profiles from T21.
- `T25E` shows how the employer-worker AI divergence claim changes when the benchmark changes.

Notes:

- Text excerpts are truncated to keep the panels readable and to reduce verbatim reproduction.
- The same-company pairs are chosen by longest posting length within each company-period slice.
- The senior-artifact uses strict management only; broad management was demoted during the exploration and stays out of the main visualization.
- The divergence chart is deliberately benchmark-sensitive because T23 showed that the sign depends on the comparison object.

Files:

- `inflated_junior_jds.png`
- `paired_company_jd_examples.png`
- `junior_share_trend_annotated.png`
- `senior_archetype_profiles.png`
- `posting_usage_divergence.png`
""",
        encoding="utf-8",
    )


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    t22 = load_table(ROOT / "tables" / "T22" / "T22_top_ghost_entry_postings.csv")
    t16_pairs = [
        {"company": "ServiceNow", "cluster": "AI-forward recomposition"},
        {"company": "Adobe", "cluster": "template inflation / text-heavy"},
        {"company": "Visa", "cluster": "stack expansion"},
        {"company": "Anduril Industries", "cluster": "entry-heavy compact"},
    ]
    change_profile = load_table(ROOT / "tables" / "T16" / "T16_company_change_profile_primary.csv")
    cluster_top = load_table(ROOT / "tables" / "T16" / "T16_cluster_top_companies_primary.csv")
    assert set(pair["company"] for pair in t16_pairs).issubset(set(cluster_top["company_name_canonical"])), "one or more paired companies missing from T16 top-company table"

    examples = select_junior_prompts(t22)
    junior_uids = examples["uid"].tolist()
    junior_postings = fetch_postings(con, uids=junior_uids)
    assert len(junior_postings) == len(junior_uids), "could not fetch every selected junior posting"

    render_junior_prompt_figure(ARTIFACT_DIR / "inflated_junior_jds.png", examples, junior_postings)
    render_company_pairs(ARTIFACT_DIR / "paired_company_jd_examples.png", t16_pairs, con, change_profile)

    t08 = load_table(ROOT / "tables" / "T08" / "T08_junior_share_trends.csv")
    render_junior_trend(ARTIFACT_DIR / "junior_share_trend_annotated.png", t08)

    t21 = load_table(ROOT / "tables" / "T21" / "T21_primary_profile_by_period_seniority.csv")
    render_senior_archetype(ARTIFACT_DIR / "senior_archetype_profiles.png", t21)

    t23_core = load_table(ROOT / "tables" / "T23" / "T23_text_source_sensitivity.csv")
    t23_bench = load_table(ROOT / "tables" / "T23" / "T23_benchmark_sensitivity.csv")
    t23_raw = load_table(ROOT / "tables" / "T23" / "T23_text_source_sensitivity.csv")
    render_divergence_chart(ARTIFACT_DIR / "posting_usage_divergence.png", t23_core, t23_raw, t23_bench)

    write_readme(ARTIFACT_DIR / "README.md")


if __name__ == "__main__":
    main()
