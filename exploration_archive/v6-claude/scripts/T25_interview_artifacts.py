"""T25 — Interview elicitation artifacts (Agent N, Wave 4).

Produces 5 PNG artifacts for the RQ4 qualitative-phase data-prompted interviews,
saved to exploration/artifacts/interview/. Each artifact is paired with a README
written separately; this script only produces the figures + a paired-JD CSV.

Artifacts:
  1. inverted_rq3_divergence.png  — posting vs worker AI adoption (RQ3 inversion)
  2. senior_orchestration_shift.png — T21 density by period × seniority
  3. archetype_pivot.png — T16 archetype pivot rate + examples
  4. authorship_style_shift.png — T29 score distribution shift
  5. paired_company_jds.csv — raw JD text pairs for interview stimulus

Inputs:
  - exploration/tables/T23/per_tool_divergence.csv
  - exploration/tables/T23/benchmark_sensitivity.csv
  - exploration/tables/T23/divergence_headline.csv
  - exploration/tables/T21/density_by_profile_period_seniority.csv
  - exploration/tables/T21/ai_senior_interaction.csv
  - exploration/tables/T16/archetype_dominant_pivot.csv
  - exploration/tables/T29/period_stats.csv
  - exploration/tables/T29/authorship_scores.csv
  - data/unified.parquet (for paired JD text)
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
OUT = ROOT / "exploration/artifacts/interview"
OUT.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "exploration/tables"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def fig1_inverted_rq3_divergence() -> None:
    """Posting-side AI tool naming vs worker-side AI usage — the RQ3 inversion."""
    per_tool = pd.read_csv(TAB / "T23" / "per_tool_divergence.csv")
    sensitivity = pd.read_csv(TAB / "T23" / "benchmark_sensitivity.csv")

    # Broad AI lead number — use the sensitivity file for the bands, and the
    # headline 28.6% vs 80.8% as the primary pair.
    posting_broad = 0.2861
    posting_narrow = 0.346
    posting_tool = 0.069
    posting_hard_req = 0.060  # AI × requirements section from T22/T23
    worker_primary = 0.808

    # Sensitivity band for worker rate (50% to 85%).
    worker_low, worker_high = 0.50, 0.85

    # --- figure ---
    fig, (ax_aggregate, ax_tool) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.1, 1]}
    )

    # Left panel — aggregate divergence with sensitivity band
    categories = [
        "Worker\nany-use\n(SO 2025)",
        "Posting\nbroad AI\n(28.6%)",
        "Posting\nnarrow AI\n(34.6%)",
        "Posting\nAI-as-tool\n(6.9%)",
        "Hard AI\nrequirement\n(AI × req, 6.0%)",
    ]
    values = [
        worker_primary * 100,
        posting_broad * 100,
        posting_narrow * 100,
        posting_tool * 100,
        posting_hard_req * 100,
    ]
    colors = ["#2E86AB", "#C73E1D", "#C73E1D", "#C73E1D", "#8B0000"]

    bars = ax_aggregate.bar(categories, values, color=colors, edgecolor="black", linewidth=0.8)

    # Worker sensitivity band (50-85%)
    ax_aggregate.axhspan(
        worker_low * 100,
        worker_high * 100,
        alpha=0.15,
        color="#2E86AB",
        label="Worker any-use sensitivity band 50-85%",
    )

    for bar, v in zip(bars, values):
        ax_aggregate.text(
            bar.get_x() + bar.get_width() / 2,
            v + 1.2,
            f"{v:.1f}%",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax_aggregate.set_ylabel("Share of developers / postings (%)")
    ax_aggregate.set_ylim(0, 100)
    ax_aggregate.set_title(
        "RQ3 inverted: workers outpace postings on AI adoption\n"
        "Worker AI any-use ≈ 80.8%, posting broad-AI rate 28.6%, hard AI requirement 6.0%",
        fontsize=11,
        pad=12,
    )
    ax_aggregate.legend(loc="upper right", fontsize=9)
    ax_aggregate.set_axisbelow(True)

    # Right panel — per-tool ratios (softened per Gate 3 narrowing 3)
    tools_df = per_tool.dropna(subset=["ratio"]).sort_values("ratio")
    tool_names = [
        {"copilot": "Copilot", "chatgpt": "ChatGPT", "claude_tool": "Claude Code"}.get(t, t)
        for t in tools_df["tool"]
    ]
    posting_rates = (tools_df["posting_rate_2026"].values * 100).tolist()
    worker_rates = (tools_df["worker_rate"].values * 100).tolist()

    x = np.arange(len(tool_names))
    width = 0.38
    ax_tool.bar(
        x - width / 2,
        posting_rates,
        width,
        label="Posting rate 2026",
        color="#C73E1D",
        edgecolor="black",
    )
    ax_tool.bar(
        x + width / 2,
        worker_rates,
        width,
        label="Worker rate (SO 2025 extrapolated)",
        color="#2E86AB",
        edgecolor="black",
    )

    for i, (p, w) in enumerate(zip(posting_rates, worker_rates)):
        ax_tool.text(i - width / 2, p + 1.5, f"{p:.1f}%", ha="center", fontsize=9)
        ax_tool.text(i + width / 2, w + 1.5, f"{w:.1f}%", ha="center", fontsize=9)

    ax_tool.set_xticks(x)
    ax_tool.set_xticklabels(tool_names)
    ax_tool.set_ylabel("Share (%)")
    ax_tool.set_ylim(0, 85)
    ax_tool.set_title(
        "Per-tool divergence: ~10-15× worker-to-posting ratio\n"
        "(Worker rates extrapolated from SO 2025 subset — approximate)",
        fontsize=11,
        pad=12,
    )
    ax_tool.legend(loc="upper left", fontsize=9)
    ax_tool.set_axisbelow(True)

    fig.suptitle(
        "Employers rarely name AI tools in postings — even though ~80% of developers use them daily",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(OUT / "inverted_rq3_divergence.png", bbox_inches="tight")
    plt.close()
    print("Saved: inverted_rq3_divergence.png")


def fig2_senior_orchestration_shift() -> None:
    """T21 senior orchestration rise by period × seniority, with AI × senior interaction."""
    density = pd.read_csv(TAB / "T21" / "density_by_profile_period_seniority.csv")
    ai_inter = pd.read_csv(TAB / "T21" / "ai_senior_interaction.csv")

    # Keep mid-senior and director; wide-reshape for easy plotting
    dens = density[density["seniority_final"].isin(["mid-senior", "director"])].copy()

    fig, (ax_profile, ax_ai) = plt.subplots(1, 2, figsize=(13, 6))

    # --- Left panel: Profile density by period × seniority
    profiles = [
        ("people_density_mean", "People management", "#4F6D7A"),
        ("orch_density_mean", "Technical orchestration", "#C73E1D"),
        ("strat_density_mean", "Strategic scope", "#E89F3E"),
    ]

    seniorities = ["mid-senior", "director"]
    periods = [2024, 2026]

    x = np.arange(len(seniorities) * len(periods))
    width = 0.25

    labels = []
    for sen in seniorities:
        for per in periods:
            labels.append(f"{sen}\n{int(per)}")

    for i, (col, name, color) in enumerate(profiles):
        values = []
        for sen in seniorities:
            for per in periods:
                row = dens[(dens["seniority_final"] == sen) & (dens["period2"] == per)]
                values.append(row[col].iloc[0])
        ax_profile.bar(x + (i - 1) * width, values, width, label=name, color=color, edgecolor="black")

    ax_profile.set_xticks(x)
    ax_profile.set_xticklabels(labels)
    ax_profile.set_ylabel("Density (matches per 1K chars)")
    ax_profile.set_title(
        "Senior profiles 2024 → 2026\n"
        "Mid-senior orch +98%, Director orch +156%, Director people-mgmt −21%",
        fontsize=11,
        pad=12,
    )
    ax_profile.legend(loc="upper left", fontsize=9)
    ax_profile.set_axisbelow(True)

    # Annotate mid-senior → 2026 orchestration bar with percent change
    ax_profile.annotate(
        "+98%",
        xy=(1, 0.332),
        xytext=(1.2, 0.42),
        fontsize=10,
        fontweight="bold",
        color="#C73E1D",
        arrowprops=dict(arrowstyle="->", color="#C73E1D"),
    )
    ax_profile.annotate(
        "+156%",
        xy=(3, 0.302),
        xytext=(3.3, 0.42),
        fontsize=10,
        fontweight="bold",
        color="#C73E1D",
        arrowprops=dict(arrowstyle="->", color="#C73E1D"),
    )

    # --- Right panel: AI × senior interaction 2026
    inter_2026 = ai_inter[ai_inter["period2"] == 2026]

    ai_groups = ["no_ai", "ai"]
    ai_labels = ["Non-AI\nsenior postings", "AI-mentioning\nsenior postings"]

    profile_metrics = [
        ("people_density", "People\nmanagement", "#4F6D7A"),
        ("orch_density", "Technical\norchestration", "#C73E1D"),
        ("strat_density", "Strategic\nscope", "#E89F3E"),
    ]

    xi = np.arange(len(ai_groups))
    for i, (col, name, color) in enumerate(profile_metrics):
        vals = []
        for grp in ai_groups:
            row = inter_2026[inter_2026["ai_mention"] == grp]
            vals.append(row[col].iloc[0] if len(row) else 0)
        ax_ai.bar(xi + (i - 1) * 0.25, vals, 0.25, label=name, color=color, edgecolor="black")

    ax_ai.set_xticks(xi)
    ax_ai.set_xticklabels(ai_labels)
    ax_ai.set_ylabel("Density (matches per 1K chars)")
    ax_ai.set_title(
        "AI × senior interaction, 2026\n"
        "Orchestration density nearly doubles when AI mentioned; people-mgmt identical",
        fontsize=11,
        pad=12,
    )
    ax_ai.legend(loc="upper left", fontsize=9)
    ax_ai.set_axisbelow(True)
    ax_ai.annotate(
        "+76%",
        xy=(1, 0.482),
        xytext=(1.2, 0.56),
        fontsize=10,
        fontweight="bold",
        color="#C73E1D",
        arrowprops=dict(arrowstyle="->", color="#C73E1D"),
    )

    fig.suptitle(
        "Senior SWE roles specialized toward hands-on technical orchestration\n"
        "The AI × senior interaction is entirely in orchestration, NOT in people management",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(OUT / "senior_orchestration_shift.png", bbox_inches="tight")
    plt.close()
    print("Saved: senior_orchestration_shift.png")


def fig3_archetype_pivot() -> None:
    """Archetype pivot rate (74.6%) with illustrative company examples."""
    pivot_df = pd.read_csv(TAB / "T16" / "archetype_dominant_pivot.csv")

    # High-interest examples to highlight (cluster-3 tool-stack adopters + notable)
    highlight = [
        "AT&T",
        "Adobe",
        "American Express",
        "Deloitte",
        "Aditi Consulting",
    ]
    hl_rows = pivot_df[pivot_df["company"].isin(highlight)].copy()

    fig, (ax_bar, ax_tbl) = plt.subplots(
        1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [0.9, 1.3]}
    )

    # --- Left: headline pivot rate
    n_total = len(pivot_df)
    n_pivoted = pivot_df["pivoted"].sum()
    pct_pivoted = n_pivoted / n_total * 100

    ax_bar.bar(
        ["Pivoted dominant\narchetype", "Same dominant\narchetype"],
        [n_pivoted, n_total - n_pivoted],
        color=["#C73E1D", "#4F6D7A"],
        edgecolor="black",
    )
    ax_bar.set_ylabel(f"Number of companies (panel n={n_total})")
    ax_bar.set_title(
        f"74.6% of overlap-panel companies\npivoted dominant archetype in 2 years\n"
        f"(n={n_pivoted}/{n_total}; 73.2% at ≥5 per period)",
        fontsize=11,
        pad=12,
    )
    for i, v in enumerate([n_pivoted, n_total - n_pivoted]):
        ax_bar.text(i, v + 2, f"{v}\n({v/n_total*100:.1f}%)", ha="center", fontsize=10, fontweight="bold")
    ax_bar.set_axisbelow(True)

    # --- Right: illustrative pivots (table + short labels)
    ax_tbl.axis("off")
    ax_tbl.set_title(
        "Illustrative company archetype pivots\n(tool-stack adopter cluster members)",
        fontsize=11,
        pad=12,
    )

    def shorten(name: str) -> str:
        mapping = {
            "Java enterprise (Spring/microservices)": "Java enterprise",
            "LLM / GenAI / ML engineering": "LLM / GenAI",
            "Data engineering (Spark/ETL)": "Data engineering",
            "DevOps / SRE / platform": "DevOps / SRE",
            "Defense & cleared regulated work": "Defense / cleared",
            "Agile/Scrum generalist": "Agile generalist",
            ".NET / ASP.NET full-stack": ".NET / ASP.NET",
            "Embedded / firmware": "Embedded",
            "JS frontend (React/TypeScript)": "JS frontend",
            "Python web backend": "Python backend",
            "Amazon program boilerplate": "Amazon template",
        }
        return mapping.get(name, name)

    rows = []
    for _, r in hl_rows.iterrows():
        rows.append(
            [
                r["company"],
                shorten(r["y2024"]),
                "→",
                shorten(r["y2026"]),
            ]
        )

    table = ax_tbl.table(
        cellText=rows,
        colLabels=["Company", "2024 dominant", "", "2026 dominant"],
        loc="center",
        cellLoc="left",
        colWidths=[0.25, 0.32, 0.05, 0.32],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    fig.suptitle(
        "Within-company restructuring in 2 years: archetype pivot rate 74.6%",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(OUT / "archetype_pivot.png", bbox_inches="tight")
    plt.close()
    print("Saved: archetype_pivot.png")


def fig4_authorship_style_shift() -> None:
    """T29 authorship score distribution 2024 → 2026."""
    scores = pd.read_csv(TAB / "T29" / "authorship_scores.csv")
    col = "authorship_score"
    s_2024 = scores[scores["period2"] == 2024][col].dropna()
    s_2026 = scores[scores["period2"] == 2026][col].dropna()

    fig, (ax_hist, ax_cum) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Left: overlapping histograms
    bins = np.linspace(-1.5, 2.0, 60)
    ax_hist.hist(s_2024, bins=bins, alpha=0.55, label="2024", color="#4F6D7A", edgecolor="black", linewidth=0.3)
    ax_hist.hist(s_2026, bins=bins, alpha=0.55, label="2026", color="#C73E1D", edgecolor="black", linewidth=0.3)

    med_2024 = s_2024.median()
    med_2026 = s_2026.median()
    ax_hist.axvline(med_2024, color="#4F6D7A", linestyle="--", linewidth=1.5)
    ax_hist.axvline(med_2026, color="#C73E1D", linestyle="--", linewidth=1.5)
    ax_hist.text(
        med_2024, ax_hist.get_ylim()[1] * 0.92, f"2024 median\n{med_2024:.2f}", fontsize=9, ha="right"
    )
    ax_hist.text(
        med_2026, ax_hist.get_ylim()[1] * 0.92, f"2026 median\n{med_2026:.2f}", fontsize=9, ha="left"
    )
    ax_hist.set_xlabel("Authorship score (higher = more LLM-style)")
    ax_hist.set_ylabel("Count of postings")
    ax_hist.set_title(
        f"Distribution shifted upward +{med_2026 - med_2024:.2f} std\n"
        f"88.7% of 2026 postings score above 2024 median; 3.9% fall below 2024 p25",
        fontsize=11,
        pad=12,
    )
    ax_hist.legend(loc="upper right")
    ax_hist.set_axisbelow(True)

    # --- Right: cumulative distribution
    sorted_2024 = np.sort(s_2024.values)
    sorted_2026 = np.sort(s_2026.values)
    cum_2024 = np.arange(1, len(sorted_2024) + 1) / len(sorted_2024)
    cum_2026 = np.arange(1, len(sorted_2026) + 1) / len(sorted_2026)

    ax_cum.plot(sorted_2024, cum_2024, label="2024", color="#4F6D7A", linewidth=2)
    ax_cum.plot(sorted_2026, cum_2026, label="2026", color="#C73E1D", linewidth=2)
    ax_cum.axvline(med_2024, color="#4F6D7A", linestyle="--", linewidth=1)
    ax_cum.axvline(med_2026, color="#C73E1D", linestyle="--", linewidth=1)
    ax_cum.set_xlabel("Authorship score")
    ax_cum.set_ylabel("Cumulative share of postings")
    ax_cum.set_title(
        "Cumulative distributions — translate upward\n"
        "Variance barely changes; whole corpus moved, didn't compress",
        fontsize=11,
        pad=12,
    )
    ax_cum.legend(loc="lower right")
    ax_cum.set_axisbelow(True)

    fig.suptitle(
        "LLM-style authorship score shifted +0.33 std between 2024 and 2026\n"
        "Candidate mechanism: recruiter-LLM drafting adoption. AI explosion is NOT style-driven (0-7% attenuation).",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(OUT / "authorship_style_shift.png", bbox_inches="tight")
    plt.close()
    print("Saved: authorship_style_shift.png")


def artifact5_paired_company_jds() -> None:
    """Pull 3-5 paired JD excerpts from cluster-3 tool-stack adopter companies."""
    con = duckdb.connect()

    # Pick companies with enough text on both sides: Adobe, Deloitte, AT&T
    companies = ["Adobe", "Deloitte", "AT&T", "American Express", "Aditi Consulting"]

    query = f"""
    WITH ranked AS (
      SELECT
        company_name_canonical AS company,
        source,
        CASE WHEN source IN ('kaggle_arshkon','kaggle_asaniczka') THEN '2024' ELSE '2026' END AS period,
        title,
        seniority_final,
        description_core_llm,
        LENGTH(description_core_llm) AS dl,
        ROW_NUMBER() OVER (
          PARTITION BY company_name_canonical,
                       CASE WHEN source IN ('kaggle_arshkon','kaggle_asaniczka') THEN '2024' ELSE '2026' END
          ORDER BY LENGTH(description_core_llm) DESC
        ) AS rn
      FROM parquet_scan('{ROOT}/data/unified.parquet')
      WHERE is_swe = TRUE AND source_platform = 'linkedin'
        AND is_english = TRUE AND date_flag = 'ok'
        AND description_core_llm IS NOT NULL
        AND LENGTH(description_core_llm) > 800
        AND company_name_canonical IN ({','.join(repr(c) for c in companies)})
    )
    SELECT company, period, title, seniority_final, description_core_llm, dl
    FROM ranked
    WHERE rn = 1
    ORDER BY company, period
    """

    df = con.execute(query).fetchdf()

    # Keep only companies with both periods
    keep = df["company"].value_counts()
    keep_companies = keep[keep >= 2].index.tolist()
    df = df[df["company"].isin(keep_companies)].copy()
    print(f"Paired JDs: {len(df)} rows across {len(keep_companies)} companies.")

    # Save full dataframe as CSV for interview stimulus
    df.to_csv(OUT / "paired_company_jds.csv", index=False)

    # Also save a human-readable Markdown version excerpt
    out_lines = [
        "# Paired company job descriptions (2024 vs 2026) — interview stimulus",
        "",
        "Source: `data/unified.parquet`, `description_core_llm` column.",
        "",
        "Selected from T16 tool-stack adopter cluster (k-means cluster 3) where both-period text was available.",
        "",
    ]
    for comp in keep_companies:
        out_lines.append(f"## {comp}")
        out_lines.append("")
        sub = df[df["company"] == comp].sort_values("period")
        for _, r in sub.iterrows():
            title = r["title"] if pd.notna(r["title"]) else "[no title]"
            out_lines.append(f"### {r['period']} — {title}")
            out_lines.append("")
            out_lines.append(f"**Source:** {r['source'] if 'source' in r else '—'}  ·  **seniority_final:** {r['seniority_final']}  ·  **desc_len:** {int(r['dl'])}")
            out_lines.append("")
            # Truncate to first ~1500 chars for readability
            text = str(r["description_core_llm"])[:2000]
            if len(r["description_core_llm"]) > 2000:
                text += "  [… truncated for display; full text in paired_company_jds.csv]"
            out_lines.append("> " + text.replace("\n", "\n> "))
            out_lines.append("")
        out_lines.append("")

    (OUT / "paired_company_jds.md").write_text("\n".join(out_lines))
    print("Saved: paired_company_jds.csv and paired_company_jds.md")


def main() -> None:
    fig1_inverted_rq3_divergence()
    fig2_senior_orchestration_shift()
    fig3_archetype_pivot()
    fig4_authorship_style_shift()
    artifact5_paired_company_jds()
    print("\nAll T25 interview artifacts written to:", OUT)


if __name__ == "__main__":
    main()
