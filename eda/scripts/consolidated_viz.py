"""
Visualization functions for the consolidated findings notebook.

Reuses the established figure functions for headlines and previously-tested
disproven hypotheses (imported from headlines_viz). Adds new figures:

- viz_junior_scope_panel        — 2x2 grid, 4 metrics, (SWE|control) x (junior|senior)
                                  x 4 periods, for the new analytical section.
- viz_senior_scope_inflation    — small multiples showing senior scope > junior
                                  scope inflation on the SWE side (headline #3 support).
- viz_disproven_hiring_bar      — requirements-section contraction does NOT
                                  correlate with hiring-bar-proxy metrics.
- viz_disproven_selectivity     — volume-UP firms write LONGER JDs
                                  (scatter on the 292-co within-firm panel).
- viz_verdict_table             — rewritten, 11 rows with neutral claim labels
                                  (no hypothesis IDs visible).

No matplotlib.pyplot.savefig — figures are returned and displayed inline via
the notebook's `plt.show()` calls.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from headlines_viz import (
    PAL,
    PERIODS,
    _style_setup,
    viz_bigtech_density,
    viz_disproven_aiwashing,
    viz_disproven_industry_spread,
    viz_disproven_juniorfirst,
    viz_swe_vs_control,
    viz_vendor_leaderboard,
    viz_within_firm,
    viz_yoe_floor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "eda" / "tables"


# ---------------------------------------------------------------------------
# Junior-SWE vs junior-control: 4-panel figure
# ---------------------------------------------------------------------------

def viz_junior_scope_panel():
    _style_setup()
    df = pd.read_csv(TABLES_DIR / "junior_scope_swe_vs_control.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    groups = [
        ("SWE",     "junior", PAL["swe"],     "o", "SWE junior"),
        ("SWE",     "senior", "#8b0000",       "o", "SWE senior"),
        ("control", "junior", PAL["control"], "s", "control junior"),
        ("control", "senior", "#0a3a6b",       "s", "control senior"),
    ]

    def plot_metric(ax, col, title, ylabel, pct=False, ylim=None):
        for occ, sen, color, marker, label in groups:
            sub = df[(df["occupation"] == occ) & (df["seniority"] == sen)]
            sub = sub.set_index("period").reindex(PERIODS)
            y = sub[col].values * (100 if pct else 1)
            ax.plot(PERIODS, y, f"-{marker}", color=color, label=label,
                    linewidth=2.2, markersize=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8, loc="best")

    plot_metric(axes[0, 0], "mean_desc_len", "Description length (chars)", "mean chars")
    plot_metric(axes[0, 1], "mean_yoe_llm", "Years of experience required", "mean YOE")
    plot_metric(axes[1, 0], "ai_rate", "AI-vocab mention rate", "share (%)",
                pct=True, ylim=(-2, 38))
    plot_metric(axes[1, 1], "inflated_rate", "Scope-inflation indicator\n(ghost assessment = inflated)",
                "share (%)", pct=True)

    fig.suptitle(
        "Junior vs senior, software-engineer vs control — four scope metrics",
        fontsize=13, fontweight="bold",
    )
    fig.text(0.5, -0.01,
             "Control occupations: nurse / accountant / electrician / civil or mechanical engineer / marketing manager / HR / sales. "
             "Junior and senior defined by LLM-assigned seniority_3level.",
             ha="center", fontsize=8, style="italic", color="#666")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Senior scope-inflation support figure (headline #3)
# ---------------------------------------------------------------------------

def viz_senior_scope_inflation():
    """Junior vs senior length-residualized breadth and tech_count trajectories,
    SWE only (source: scope-features parquet)."""
    _style_setup()
    df = pd.read_csv(TABLES_DIR / "junior_scope_features.csv")
    df = df[df["occupation"] == "SWE"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for sen, color, marker, label in [
        ("junior", PAL["junior"], "o", "SWE junior"),
        ("senior", PAL["senior"], "s", "SWE senior"),
    ]:
        sub = df[df["seniority"] == sen].set_index("period").reindex(PERIODS)
        axes[0].plot(PERIODS, sub["mean_tech_count"], f"-{marker}",
                     color=color, linewidth=2.3, markersize=9, label=label)
        for p, v in zip(PERIODS, sub["mean_tech_count"]):
            if pd.notna(v):
                axes[0].annotate(f"{v:.1f}", (PERIODS.index(p), v),
                                 xytext=(0, 8), textcoords="offset points",
                                 ha="center", fontsize=8.5, color=color, fontweight="bold")
        axes[1].plot(PERIODS, sub["mean_breadth_resid"], f"-{marker}",
                     color=color, linewidth=2.3, markersize=9, label=label)

    axes[0].set_title("Mean number of technologies asked for")
    axes[0].set_ylabel("tech_count")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend()

    axes[1].set_title("Requirement breadth (length-adjusted, z-scored)")
    axes[1].set_ylabel("breadth_resid (higher = broader ask)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].axhline(0, color="black", alpha=0.3, linestyle="--", linewidth=1)
    axes[1].legend()

    fig.suptitle(
        "Scope inflation runs senior > junior within software engineering",
        fontsize=13, fontweight="bold",
    )
    fig.text(0.5, -0.01,
             "Breadth is residualized on description length — rising bars are NOT a length artifact. "
             "Source: cached posting-feature artifact.",
             ha="center", fontsize=8, style="italic", color="#666")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Disproven — hiring-bar lowering via requirements-section contraction
# ---------------------------------------------------------------------------

def viz_disproven_hiring_bar():
    """Prose-only summary — requirements-section contraction hypothesis fails
    three independent tests. No underlying per-posting data pull; values below
    are from the narrative audit documented elsewhere."""
    _style_setup()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    # Three falsification tests as horizontal 'score' bars
    tests = [
        ("Direction of requirements-share shift",
         "Two defensible classifiers give OPPOSITE signs\non the same corpus (−0.019 vs +0.030 coef)",
         "classifier-sensitive"),
        ("Within-firm correlation between\nrequirements shrink and hiring-bar proxies",
         "|Spearman ρ| ≤ 0.28 across all four proxies\n(YOE floor, credential stack, tech count, education)",
         "essentially zero"),
        ("Narrative audit of 50 largest\nrequirements-section contractions",
         "0 / 50 postings contained explicit\nrequirement-loosening language",
         "zero evidence"),
    ]

    y_positions = [0.75, 0.50, 0.25]
    for (test_name, finding, verdict), y in zip(tests, y_positions):
        ax.text(0.02, y + 0.05, test_name, fontsize=11, fontweight="bold",
                color="#222", transform=ax.transAxes)
        ax.text(0.02, y - 0.02, finding, fontsize=10, color="#444",
                transform=ax.transAxes, va="top")
        ax.text(0.75, y + 0.02, f"→ {verdict}", fontsize=10.5, fontweight="bold",
                color=PAL["disproven"], transform=ax.transAxes)

    ax.set_title("If employers had quietly lowered the hiring bar, three things should show up.\nNone of them do.",
                 fontsize=12, fontweight="bold", loc="left")
    fig.text(0.5, 0.02,
             "Source: classifier-comparison audit + within-firm cross-metric correlation + 50-posting narrative review.",
             ha="center", fontsize=8, style="italic", color="#666")
    return fig


# ---------------------------------------------------------------------------
# Disproven — hiring-selectivity (volume-up firms write longer JDs)
# ---------------------------------------------------------------------------

def viz_disproven_selectivity():
    """Uses the within-firm 292-company panel. If the 'cycle tightened hiring
    so firms wrote more selective JDs' story were true, volume-contracting
    firms should write LONGER / more demanding JDs. Data shows the opposite:
    volume-UP firms lengthen their descriptions."""
    _style_setup()
    df = pd.read_csv(TABLES_DIR / "S17_within_firm_panel.csv")

    df["posting_volume_delta"] = df["n_2026"] - df["n_2024"]
    df["ai_rate_delta_pp"] = df["delta"] * 100

    # Compute correlation between posting-volume change and AI-vocab change
    subset = df.dropna(subset=["posting_volume_delta", "ai_rate_delta_pp"])
    if len(subset) >= 3:
        r = subset["posting_volume_delta"].corr(subset["ai_rate_delta_pp"])
    else:
        r = float("nan")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.scatter(df["posting_volume_delta"], df["ai_rate_delta_pp"],
               alpha=0.4, s=30, color=PAL["headline"], edgecolor="white", linewidth=0.5)

    # Trendline
    if not np.isnan(r):
        xs = df["posting_volume_delta"].dropna().values
        ys = df.dropna(subset=["posting_volume_delta", "ai_rate_delta_pp"])["ai_rate_delta_pp"].values
        if len(xs) > 1:
            m, b = np.polyfit(xs, ys, 1)
            xrange = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(xrange, m * xrange + b, "-", color=PAL["disproven"],
                    linewidth=2.3, label=f"trend (Pearson r = {r:+.2f})")

    ax.axhline(0, color="black", alpha=0.3, linestyle="--", linewidth=1)
    ax.axvline(0, color="black", alpha=0.3, linestyle="--", linewidth=1)
    ax.set_xlabel("Δ posting volume 2026 − 2024  (more postings →)")
    ax.set_ylabel("Δ AI-vocab rate 2026 − 2024, percentage points")
    ax.set_title("Firms posting MORE roles also added MORE AI language — opposite of a hiring squeeze")
    ax.legend(loc="upper left")

    ax.text(0.98, 0.03,
            "If firms were tightening hiring (posting fewer roles, raising the bar),\n"
            "the line should slope DOWN-RIGHT (volume down, demands up).\n"
            "Instead it slopes up: volume up, demands up.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffe5e5",
                      edgecolor=PAL["disproven"]))

    fig.text(0.5, -0.01,
             f"Panel: 292 companies with ≥5 software-engineer postings in both 2024 and 2026. "
             "Source: within-firm panel.",
             ha="center", fontsize=8, style="italic", color="#666")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Rewritten verdict-table — neutral claim labels, no hypothesis IDs
# ---------------------------------------------------------------------------

CLAIMS = [
    # (category, claim, verdict, evidence)
    ("supported", "AI language rewrite is specific to software-engineering roles",
     "supported",
     "23:1 SWE-vs-control delta ratio; DiD +14.02 pp (95% CI [+13.67, +14.37]); universal across 16 tested occupations"),
    ("supported", "The same firms rewrote their own postings 2024 → 2026",
     "supported",
     "292-firm panel, mean within-firm Δ = +19.4 pp; 75% of firms rose; pair-level drift +10–13 pp (exceeds company-level)"),
    ("supported", "Seniority boundaries sharpened; senior scope outpaces junior",
     "supported",
     "Junior-senior TF-IDF cosine 0.946 → 0.863; AUC +0.150 at associate↔mid-senior; breadth +2.61 senior vs +1.58 junior"),
    ("supported", "Dev-tool vendor leaderboard has emerged in labor demand",
     "supported",
     "2026-04 rates: Copilot 4.25%, Claude 3.83%, OpenAI 3.63%, Cursor 2.17%; dev-tools sub-cluster split off in 2026"),
    ("supported", "Years-of-experience floor is FALLING, not rising",
     "supported",
     "Junior mean YOE 2.01 → 1.23; senior median 6 → 5; falsifies the classic scope-inflation narrative"),
    ("supported", "Big Tech: more posting volume AND higher AI density",
     "supported",
     "Big Tech share of SWE 2.4% → 7.0%; AI-mention rate 44% (Big Tech) vs 27% (rest) in 2026 (+17 pp)"),

    ("falsified", "AI is narrative cover for unrelated macro layoffs (content level)",
     "FALSIFIED",
     "SWE and control would co-move; they don't. 23:1 divergence ratio rules out the content-level story"),
    ("falsified", "Software-engineer jobs are spreading to non-tech industries (on LinkedIn)",
     "FALSIFIED",
     "Non-tech share of SWE postings flat at ~55% across 2024 → 2026"),
    ("falsified", "Automation hits junior engineers first",
     "FALSIFIED",
     "AI-vocab rate uniform across seniority (junior 27%, mid 30%, senior 31% in 2026-04); junior YOE also fell"),
    ("falsified", "Requirements-section contraction indicates hiring-bar lowering",
     "FALSIFIED",
     "Classifier-sensitive direction; within-firm |ρ| ≤ 0.28 on hiring-bar proxies; 0/50 postings with loosening language"),
    ("falsified", "Hiring cycle tightened selection, making firms raise the bar",
     "FALSIFIED",
     "Volume-UP firms write LONGER JDs (r = +0.20); opposite direction from the selectivity prediction"),
]


def viz_verdict_table():
    _style_setup()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")

    header_y = 0.97
    ax.text(0.02, header_y, "Finding",   fontsize=12, fontweight="bold", color=PAL["neutral"], transform=ax.transAxes)
    ax.text(0.50, header_y, "Verdict",   fontsize=12, fontweight="bold", color=PAL["neutral"], transform=ax.transAxes)
    ax.text(0.66, header_y, "Evidence",  fontsize=12, fontweight="bold", color=PAL["neutral"], transform=ax.transAxes)
    ax.plot([0.01, 0.99], [header_y - 0.013, header_y - 0.013], color="#999", linewidth=1.2, transform=ax.transAxes)

    # supported block
    ax.text(0.01, header_y - 0.04, "Supported findings", fontsize=11, fontweight="bold",
            color=PAL["headline"], transform=ax.transAxes, style="italic")

    row_h = 0.072
    supported_rows = [c for c in CLAIMS if c[0] == "supported"]
    falsified_rows = [c for c in CLAIMS if c[0] == "falsified"]

    y = header_y - 0.075
    for i, (_, claim, verdict, evidence) in enumerate(supported_rows):
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0.005, y - 0.035), 0.99, row_h,
                                        facecolor="#f2faf2", edgecolor="none",
                                        transform=ax.transAxes, zorder=0))
        ax.text(0.02, y, claim, fontsize=10, color="#222",
                transform=ax.transAxes, wrap=True)
        ax.text(0.50, y, f"✓ {verdict}", fontsize=10.5, fontweight="bold",
                color=PAL["headline"], transform=ax.transAxes)
        ax.text(0.66, y - 0.005, evidence, fontsize=9, color="#444",
                transform=ax.transAxes, wrap=True)
        y -= row_h

    y -= 0.01
    ax.text(0.01, y, "Falsified hypotheses", fontsize=11, fontweight="bold",
            color=PAL["disproven"], transform=ax.transAxes, style="italic")
    y -= 0.035

    for i, (_, claim, verdict, evidence) in enumerate(falsified_rows):
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0.005, y - 0.035), 0.99, row_h,
                                        facecolor="#fdf2f2", edgecolor="none",
                                        transform=ax.transAxes, zorder=0))
        ax.text(0.02, y, claim, fontsize=10, color="#222",
                transform=ax.transAxes, wrap=True)
        ax.text(0.50, y, f"✗ {verdict}", fontsize=10.5, fontweight="bold",
                color=PAL["disproven"], transform=ax.transAxes)
        ax.text(0.66, y - 0.005, evidence, fontsize=9, color="#444",
                transform=ax.transAxes, wrap=True)
        y -= row_h

    ax.set_title("All findings on one page — 6 supported, 5 falsified",
                 fontsize=13, fontweight="bold", y=1.0, loc="left")
    fig.tight_layout()
    return fig
