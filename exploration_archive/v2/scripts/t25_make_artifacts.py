from __future__ import annotations

import math
import os
import textwrap
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd


ROOT = Path("/home/jihgaboot/gabor/job-research")
ART_DIR = ROOT / "exploration" / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

STAGE8 = ROOT / "preprocessing" / "intermediate" / "stage8_final.parquet"


def query_df(sql: str) -> pd.DataFrame:
    con = duckdb.connect()
    return con.execute(sql).fetchdf()


def clean_text(value: str) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\r", "\n").split())


def wrap_block(text: str, width: int) -> str:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return ""
    return "\n".join(textwrap.fill(p, width=width) for p in paragraphs)


def draw_card(ax, title: str, meta: list[str], body: str, *, facecolor="#f8fafc", edgecolor="#cbd5e1"):
    ax.set_axis_off()
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.0,
            edgecolor=edgecolor,
            facecolor=facecolor,
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.text(
        0.03,
        0.95,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#0f172a",
        wrap=True,
    )
    y = 0.885
    for line in meta:
        ax.text(
            0.03,
            y,
            line,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#334155",
            wrap=True,
        )
        y -= 0.055
    ax.text(
        0.03,
        y - 0.015,
        body,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color="#111827",
        linespacing=1.25,
        wrap=True,
    )


def save_fig(fig, stem: str):
    png = ART_DIR / f"{stem}.png"
    pdf = ART_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def make_inflated_junior_cards() -> None:
    uids = [
        "arshkon_3895521599",
        "arshkon_3889476960",
        "arshkon_3885854922",
        "linkedin_li-4388198120",
        "linkedin_li-4387728717",
    ]
    uid_list = ",".join([f"'{u}'" for u in uids])
    df = query_df(
        f"""
        SELECT
            uid,
            period,
            company_name_effective,
            title,
            seniority_final,
            seniority_3level,
            source,
            description_length,
            core_length,
            COALESCE(NULLIF(description_core, ''), description) AS text
        FROM parquet_scan('{STAGE8.as_posix()}')
        WHERE uid IN ({uid_list})
        ORDER BY period, company_name_effective, uid
        """
    )

    fig = plt.figure(figsize=(15, 16))
    gs = fig.add_gridspec(len(df), 1, hspace=0.16)
    fig.suptitle(
        "T25 artifact 1: inflated junior job descriptions",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.02,
        0.972,
        "Entry-level postings where title and scope language pull toward mid-senior or senior responsibilities.",
        ha="left",
        va="top",
        fontsize=10,
        color="#475569",
    )

    for i, row in df.iterrows():
        ax = fig.add_subplot(gs[i, 0])
        snippet = clean_text(row["text"])
        snippet = snippet[:900]
        if len(snippet) == 900:
            snippet += "..."
        meta = [
            f"{row['uid']} | {row['period']} | {row['source']} | {row['seniority_final']} ({row['seniority_3level']})",
            f"{row['company_name_effective']} | {row['description_length']} chars | core {int(row['core_length']) if not pd.isna(row['core_length']) else 'NA'} chars",
        ]
        draw_card(
            ax,
            str(row["title"]),
            meta,
            wrap_block(snippet, width=145),
            facecolor="#fffaf2" if row["period"] == "2026-03" else "#f8fafc",
            edgecolor="#f59e0b" if row["period"] == "2026-03" else "#cbd5e1",
        )

    save_fig(fig, "T25_inflated_junior_jds")


def make_paired_jd_cards() -> None:
    pairs = [
        ("Microsoft", "software engineer"),
        ("Amazon", "software development engineer"),
        ("Uber", "software engineer"),
        ("Leidos", "software engineer"),
        ("Oracle", "software developer"),
    ]
    pair_rows = []
    con = duckdb.connect()
    for company, title in pairs:
        df = con.execute(
            f"""
            WITH f AS (
              SELECT
                uid, period, company_name_effective, title, title_normalized, seniority_final,
                seniority_3level, source, description_length, core_length,
                COALESCE(NULLIF(description_core, ''), description) AS text
              FROM parquet_scan('{STAGE8.as_posix()}')
              WHERE source_platform='linkedin'
                AND is_english=true
                AND date_flag='ok'
                AND is_swe=true
                AND company_name_canonical = '{company.replace("'", "''")}'
                AND title_normalized = '{title.replace("'", "''")}'
                AND period IN ('2024-04', '2026-03')
            ),
            ranked AS (
              SELECT *, ROW_NUMBER() OVER (PARTITION BY period ORDER BY description_length DESC, uid) AS rn
              FROM f
            )
            SELECT *
            FROM ranked
            WHERE rn = 1
            ORDER BY period
            """
        ).fetchdf()
        if len(df) != 2:
            raise RuntimeError(f"Expected two rows for {company} / {title}, found {len(df)}")
        pair_rows.append((company, title, df))

    fig = plt.figure(figsize=(16, 24))
    outer = fig.add_gridspec(len(pair_rows), 2, width_ratios=[1, 1], wspace=0.05, hspace=0.28)
    fig.suptitle(
        "T25 artifact 2: same-company paired JDs over time",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.02,
        0.972,
        "Matched company/title families showing how requirements changed between the 2024-04 baseline and 2026-03 scrape.",
        ha="left",
        va="top",
        fontsize=10,
        color="#475569",
    )

    for r, (company, title, df) in enumerate(pair_rows):
        left = fig.add_subplot(outer[r, 0])
        right = fig.add_subplot(outer[r, 1])
        left_row = df.iloc[0]
        right_row = df.iloc[1]
        fig.text(
            0.02,
            0.958 - r * (0.91 / len(pair_rows)),
            f"{company} | {title}",
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="#1e293b",
        )
        for ax, row, panel_color in [
            (left, left_row, "#f8fafc"),
            (right, right_row, "#fffaf2"),
        ]:
            snippet = clean_text(row["text"])[:850]
            if len(snippet) == 850:
                snippet += "..."
            meta = [
                f"{row['uid']} | {row['period']} | {row['source']} | {row['seniority_final']} ({row['seniority_3level']})",
                f"{row['company_name_effective']} | {row['description_length']} chars | core {int(row['core_length']) if not pd.isna(row['core_length']) else 'NA'} chars",
            ]
            draw_card(
                ax,
                str(row["title"]),
                meta,
                wrap_block(snippet, width=68),
                facecolor=panel_color,
                edgecolor="#f59e0b" if row["period"] == "2026-03" else "#cbd5e1",
            )

    save_fig(fig, "T25_paired_jds_over_time")


def make_junior_share_trend() -> None:
    df = pd.read_csv(ROOT / "exploration" / "tables" / "T09" / "seniority_variant_entry_share_by_period.csv")
    line = df[df["variant"] == "final_all_nonunknown"].copy()
    line["period_date"] = pd.to_datetime(line["period"] + "-01")
    line = line.sort_values("period_date")

    release_dates = [
        ("GPT-4", "2023-03-14"),
        ("Claude 3", "2024-03-04"),
        ("GPT-4o", "2024-05-13"),
        ("Claude 3.5 Sonnet", "2024-06-20"),
        ("o1", "2024-09-12"),
        ("DeepSeek V3", "2024-12-26"),
        ("Claude 3.5 MAX", "2025-02-24"),
        ("GPT-4.5", "2025-02-27"),
        ("Claude 3.6 Sonnet", "2025-04-08"),
        ("Claude 4 Opus", "2025-09-25"),
        ("Claude 4.5 Haiku", "2025-10-22"),
        ("Gemini 2.5 Pro", "2026-03-25"),
    ]
    rel = pd.DataFrame(release_dates, columns=["label", "date"])
    rel["date"] = pd.to_datetime(rel["date"])

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)

    ax.plot(line["period_date"], line["entry_share_known"] * 100, color="#0f766e", lw=2.5, marker="o", ms=7)
    for _, row in line.iterrows():
        ax.annotate(
            f"{row['period']}\n{row['entry_share_known']*100:.1f}%",
            (row["period_date"], row["entry_share_known"] * 100),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="#134e4a",
        )

    ax.set_ylim(0, max(line["entry_share_known"] * 100) * 1.25)
    ax.set_ylabel("Entry share among known seniority (%)")
    ax.set_title("T25 artifact 3: junior-share trend with AI release markers", loc="left", fontsize=16, fontweight="bold")
    ax.grid(axis="y", color="#e2e8f0", lw=0.8)

    ax2.axhline(0, color="#94a3b8", lw=1)
    colors = ["#64748b", "#94a3b8"]
    for i, (_, row) in enumerate(rel.iterrows()):
        y = 0.2 if i % 2 == 0 else -0.2
        ax2.scatter(row["date"], y, s=28, color=colors[i % 2], zorder=3)
        ax2.vlines(row["date"], 0, y, color=colors[i % 2], lw=0.6, alpha=0.6)
        ax2.text(
            row["date"],
            y + (0.12 if y > 0 else -0.12),
            row["label"],
            ha="center",
            va="bottom" if y > 0 else "top",
            fontsize=7.5,
            rotation=90,
            color="#334155",
        )
    ax2.set_ylim(-0.75, 0.75)
    ax2.set_yticks([])
    ax2.set_xlabel("Model release timeline")
    ax2.spines[["left", "right", "top"]].set_visible(False)
    ax2.spines["bottom"].set_color("#cbd5e1")
    ax2.grid(False)
    ax2.set_xlim(pd.Timestamp("2023-01-01"), pd.Timestamp("2026-04-30"))

    fig.text(
        0.02,
        0.02,
        "Primary series uses seniority_final; 2024-01 remains context only because asaniczka has no native entry labels.",
        fontsize=9,
        color="#475569",
    )

    save_fig(fig, "T25_junior_share_trend")


def make_senior_archetype_chart() -> None:
    summary = pd.read_csv(ROOT / "exploration" / "tables" / "T21" / "T21_summary.csv").sort_values("period")
    arch = pd.read_csv(ROOT / "exploration" / "tables" / "T21" / "T21_archetypes.csv")
    order = ["2024-01", "2024-04", "2026-03"]
    summary["period"] = pd.Categorical(summary["period"], order, ordered=True)
    summary = summary.sort_values("period")
    arch["period"] = pd.Categorical(arch["period"], order, ordered=True)
    arch = arch.sort_values(["period", "archetype"])

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.05], wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    x = range(len(summary))
    ax1.plot(x, summary["mean_mgmt_per_1k"], marker="o", lw=2.5, color="#b45309", label="Management")
    ax1.plot(x, summary["mean_orch_per_1k"], marker="o", lw=2.5, color="#0f766e", label="Orchestration")
    ax1.set_xticks(list(x), summary["period"].astype(str))
    ax1.set_ylabel("Mentions per 1K chars")
    ax1.set_title("Senior language profile", loc="left", fontsize=15, fontweight="bold")
    ax1.grid(axis="y", color="#e2e8f0", lw=0.8)
    ax1.legend(frameon=False, loc="upper left")

    for i, row in summary.iterrows():
        ax1.annotate(f"{row['mean_orch_per_1k']:.2f}", (list(x)[list(summary.index).index(i)], row["mean_orch_per_1k"]),
                     xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color="#0f766e")
        ax1.annotate(f"{row['mean_mgmt_per_1k']:.3f}", (list(x)[list(summary.index).index(i)], row["mean_mgmt_per_1k"]),
                     xytext=(0, -14), textcoords="offset points", ha="center", fontsize=7.5, color="#b45309")

    ax2 = fig.add_subplot(gs[0, 1])
    base = pd.DataFrame({"period": order})
    base["new_senior"] = base["period"].map(summary.set_index("period")["share_new_senior"])
    base["classic_senior"] = base["period"].map(summary.set_index("period")["share_classic_senior"])
    base["mixed"] = base["period"].map(summary.set_index("period")["share_mixed"])
    x2 = range(len(base))
    ax2.bar(x2, base["classic_senior"] * 100, color="#94a3b8", label="Classic senior")
    ax2.bar(x2, base["mixed"] * 100, bottom=base["classic_senior"] * 100, color="#cbd5e1", label="Mixed")
    ax2.bar(x2, base["new_senior"] * 100, bottom=(base["classic_senior"] + base["mixed"]) * 100, color="#0f766e", label="New senior")
    ax2.set_xticks(list(x2), base["period"])
    ax2.set_ylabel("Share of mid-senior/director SWE (%)")
    ax2.set_title("Archetype mix", loc="left", fontsize=15, fontweight="bold")
    ax2.grid(axis="y", color="#e2e8f0", lw=0.8)
    ax2.legend(frameon=False, loc="upper left")

    fig.suptitle(
        "T25 artifact 4: senior archetype chart",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.02,
        0.02,
        "Management language stays sparse while orchestration language rises, and the 'new senior' share roughly doubles from 2024-01 to 2026-03.",
        fontsize=9,
        color="#475569",
    )

    save_fig(fig, "T25_senior_archetype_chart")


def make_posting_usage_divergence_chart() -> None:
    post = pd.read_csv(ROOT / "exploration" / "tables" / "T14" / "posting_ai_rates.csv")
    bench = pd.read_csv(ROOT / "exploration" / "tables" / "T14" / "benchmarks.csv")
    sweep = post[(post["occ_group"] == "swe") & (post["seniority_final"] == "mid-senior") & (post["period"].isin(["2024-04", "2026-03"]))].copy()
    sweep = sweep.sort_values("period")
    bench = bench[bench["benchmark_source"].isin(["Anthropic Economic Index 2025", "Stack Overflow 2025"])].copy()

    items = [
        ("2024-04 SWE posting any AI", float(sweep.loc[sweep["period"] == "2024-04", "any_rate_pct"].iloc[0]), "#0f766e"),
        ("2024-04 SWE posting tool AI", float(sweep.loc[sweep["period"] == "2024-04", "tool_rate_pct"].iloc[0]), "#0f766e"),
        ("2026-03 SWE posting any AI", float(sweep.loc[sweep["period"] == "2026-03", "any_rate_pct"].iloc[0]), "#0f766e"),
        ("2026-03 SWE posting tool AI", float(sweep.loc[sweep["period"] == "2026-03", "tool_rate_pct"].iloc[0]), "#0f766e"),
        ("Anthropic SOC 15-0000", float(bench.loc[bench["benchmark_label"].str.contains("SOC 15-0000"), "value_pct"].iloc[0]), "#64748b"),
        ("Stack Overflow mid-career", float(bench.loc[bench["benchmark_label"].str.contains("Mid career"), "value_pct"].iloc[0]), "#64748b"),
        ("Stack Overflow experienced", float(bench.loc[bench["benchmark_label"].str.contains("Experienced"), "value_pct"].iloc[0]), "#64748b"),
    ]

    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    y = list(range(len(items)))[::-1]
    for yi, (label, value, color) in zip(y, items):
        ax.scatter(value, yi, s=110, color=color, zorder=3)
        ax.hlines(yi, 0, value, color=color, lw=4, alpha=0.4)
        ax.text(value + 1.2, yi, f"{value:.1f}%", va="center", ha="left", fontsize=9, color="#0f172a")
    ax.set_yticks(y, [x[0] for x in items])
    ax.set_xlim(0, 100)
    ax.set_xlabel("AI share / usage (%)")
    ax.set_title("T25 artifact 5: posting-usage divergence", loc="left", fontsize=16, fontweight="bold")
    ax.grid(axis="x", color="#e2e8f0", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(44.0, color="#94a3b8", lw=1, ls="--")
    ax.axvline(83.5, color="#cbd5e1", lw=1, ls=":")
    ax.axvline(86.8, color="#cbd5e1", lw=1, ls=":")
    ax.text(44.0, len(items) - 0.3, "Anthropic 44%", rotation=90, va="bottom", ha="right", fontsize=8, color="#475569")
    ax.text(83.5, len(items) - 0.3, "Stack Overflow 83.5%", rotation=90, va="bottom", ha="right", fontsize=8, color="#475569")
    ax.text(86.8, len(items) - 0.3, "Stack Overflow 86.8%", rotation=90, va="bottom", ha="right", fontsize=8, color="#475569")
    fig.text(
        0.02,
        0.02,
        "Posting requirements are above the Anthropic occupation-level benchmark by 2026, but still below the developer self-report benchmarks.",
        fontsize=9,
        color="#475569",
    )
    save_fig(fig, "T25_posting_usage_divergence")


def make_readme() -> None:
    readme = ART_DIR / "README.md"
    readme.write_text(
        """# T25 / T26 interview artifacts

Source basis:
- Stage 8 parquet: `preprocessing/intermediate/stage8_final.parquet`
- Seniority trend: `exploration/tables/T09/seniority_variant_entry_share_by_period.csv`
- Senior archetype chart: `exploration/tables/T21/T21_summary.csv`, `exploration/tables/T21/T21_archetypes.csv`
- Posting / usage divergence: `exploration/tables/T14/posting_ai_rates.csv`, `exploration/tables/T14/benchmarks.csv`

Text field rule:
- Stage 8 does not contain `description_core_llm`
- These excerpt cards use `description_core` as the primary field, with `description` as fallback only when needed

Files:
- `T25_inflated_junior_jds.png` / `.pdf`
- `T25_paired_jds_over_time.png` / `.pdf`
- `T25_junior_share_trend.png` / `.pdf`
- `T25_senior_archetype_chart.png` / `.pdf`
- `T25_posting_usage_divergence.png` / `.pdf`

Selection notes:
- Inflated junior cards are drawn from the strongest scope-inflation examples surfaced in T12/T19/T23
- Paired JD cards use same-company, same-title families in 2024-04 vs 2026-03
- The junior-share trend is the default `seniority_final` series with AI release markers added as timeline annotations
- The senior archetype chart combines management/orchestration intensity with the new/classic senior split
- The divergence chart juxtaposes posting AI shares against Anthropic and Stack Overflow benchmarks
""",
        encoding="utf-8",
    )


def main() -> None:
    make_inflated_junior_cards()
    make_paired_jd_cards()
    make_junior_share_trend()
    make_senior_archetype_chart()
    make_posting_usage_divergence_chart()
    make_readme()


if __name__ == "__main__":
    main()
