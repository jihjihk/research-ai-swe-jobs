"""
Shared visualization helpers for the findings-consolidated notebook.

Each function reads from an existing CSV in eda/tables/ and returns a
matplotlib Figure (no save_fig; figures are embedded inline).

Imported by consolidated_viz, which composes these with its own
notebook-specific figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "eda" / "tables"

# Consistent palette across figures
PAL = {
    "swe":          "#d62728",  # red
    "swe_adjacent": "#ff7f0e",  # orange
    "control":      "#1f77b4",  # blue
    "junior":       "#d62728",
    "mid":          "#ff7f0e",
    "senior":       "#2ca02c",
    "unknown":      "#7f7f7f",
    "big_tech":     "#1f77b4",
    "rest":         "#bcbd22",
    "ai":           "#9467bd",
    "no_ai":        "#7f7f7f",
    "headline":     "#2ca02c",
    "disproven":    "#d62728",
    "neutral":      "#4f4f4f",
}

PERIODS = ["2024-01", "2024-04", "2026-03", "2026-04"]


def _style_setup():
    """Set consistent matplotlib styling for headline figures."""
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
    })


# ---------------------------------------------------------------------------
# HEADLINE 1: within-firm AI rewrite
# ---------------------------------------------------------------------------

def viz_within_firm():
    _style_setup()
    s17 = pd.read_csv(TABLES_DIR / "S17_within_firm_panel.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) scatter
    axes[0].scatter(s17["ai_rate_2024"] * 100, s17["ai_rate_2026"] * 100,
                    alpha=0.4, s=35, color=PAL["headline"], edgecolor="white", linewidth=0.5)
    lim = 100
    axes[0].plot([0, lim], [0, lim], "k--", alpha=0.4, label="no change (y = x)", linewidth=1)

    notable_risers = ["Microsoft", "Wells Fargo", "Amazon", "Walmart",
                      "Capital One", "Google", "JPMorgan Chase & Co.",
                      "Amazon Web Services (AWS)"]
    for name in notable_risers:
        row = s17[s17["company_name_canonical"] == name]
        if not row.empty:
            x = row["ai_rate_2024"].iloc[0] * 100
            y = row["ai_rate_2026"].iloc[0] * 100
            axes[0].annotate(name, (x, y), fontsize=8.5, alpha=0.85,
                             xytext=(6, 4), textcoords="offset points")
            axes[0].scatter([x], [y], s=60, color=PAL["headline"],
                            edgecolor="black", linewidth=1, zorder=3)

    # Mark a non-mover
    flat = s17[s17["company_name_canonical"] == "Raytheon"]
    if not flat.empty:
        x = flat["ai_rate_2024"].iloc[0] * 100
        y = flat["ai_rate_2026"].iloc[0] * 100
        axes[0].annotate("Raytheon (defense, flat)", (x, y), fontsize=8.5,
                         xytext=(8, -8), textcoords="offset points",
                         color=PAL["disproven"])
        axes[0].scatter([x], [y], s=60, color=PAL["disproven"],
                        edgecolor="black", linewidth=1, zorder=3)

    axes[0].set_xlabel("AI-vocab rate, 2024 (%)")
    axes[0].set_ylabel("AI-vocab rate, 2026 (%)")
    axes[0].set_title("Same firms rewrote their own postings 2024 → 2026")
    axes[0].set_xlim(-2, 102)
    axes[0].set_ylim(-2, 102)
    axes[0].legend(loc="lower right")

    # (b) histogram
    deltas = s17["delta"] * 100
    axes[1].hist(deltas, bins=40, color=PAL["headline"], edgecolor="white")
    axes[1].axvline(0, color="black", linestyle="--", alpha=0.5, label="no change")
    mean_d = deltas.mean()
    axes[1].axvline(mean_d, color=PAL["disproven"], linestyle="-",
                    linewidth=2.5, label=f"mean = +{mean_d:.1f}pp")
    axes[1].set_xlabel("within-firm Δ AI-vocab rate (2026 − 2024), percentage points")
    axes[1].set_ylabel("# companies")
    pct_up = (s17["delta"] > 0).mean() * 100
    pct_10 = (s17["delta"] > 0.10).mean() * 100
    axes[1].set_title(f"{pct_up:.0f}% of {len(s17)} companies rose; {pct_10:.0f}% rose >10pp")
    axes[1].legend()

    fig.suptitle("The same role at the same company reads differently in 2026 than in 2024")
    fig.text(0.5, -0.02,
             "Panel: 292 companies with ≥5 SWE postings in BOTH 2024 (Kaggle) and 2026 (scraped).  Source: eda/tables/S17_within_firm_panel.csv",
             ha="center", fontsize=8, style="italic", color="#666")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# HEADLINE 2: SWE-specific divergence
# ---------------------------------------------------------------------------

def viz_swe_vs_control():
    _style_setup()
    df = pd.read_csv(TABLES_DIR / "S11_core_swe_vs_control.csv")
    fig, ax = plt.subplots(figsize=(11, 6))

    for grp, color in [("swe", PAL["swe"]), ("swe_adjacent", PAL["swe_adjacent"]), ("control", PAL["control"])]:
        sub = df[df["analysis_group"] == grp].set_index("period").reindex(PERIODS)
        ax.plot(PERIODS, sub["ai_rate"] * 100, "-o", color=color, label=grp,
                linewidth=2.5, markersize=8)
        for p, v in zip(PERIODS, sub["ai_rate"]):
            if pd.notna(v):
                ax.annotate(f"{v*100:.1f}%", (PERIODS.index(p), v*100),
                            xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xlabel("period")
    ax.set_ylabel("AI-vocab mention rate (%)")
    ax.set_title("AI language rewrite is specific to software-engineer postings")
    ax.legend(title="analysis_group", loc="upper left")
    ax.set_ylim(-2, 33)

    # Big annotation
    swe_delta = df[(df["analysis_group"]=="swe") & (df["period"]=="2026-04")]["ai_rate"].iloc[0] - \
                df[(df["analysis_group"]=="swe") & (df["period"]=="2024-01")]["ai_rate"].iloc[0]
    ctrl_delta = df[(df["analysis_group"]=="control") & (df["period"]=="2026-04")]["ai_rate"].iloc[0] - \
                 df[(df["analysis_group"]=="control") & (df["period"]=="2024-01")]["ai_rate"].iloc[0]
    ratio = swe_delta / ctrl_delta if ctrl_delta else float("inf")
    ax.text(0.5, 0.96,
            f"SWE Δ = +{swe_delta*100:.1f}pp  ·  control Δ = +{ctrl_delta*100:.1f}pp  ·  ratio = {ratio:.0f}×",
            transform=ax.transAxes, ha="center", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8c4", edgecolor="#999"))

    fig.text(0.5, -0.01,
             "If AI talk were a generic economy-wide narrative, or if macro forces alone drove content change, SWE and control should co-move. They don't.  Source: eda/tables/S11_core_swe_vs_control.csv",
             ha="center", fontsize=8, style="italic", color="#666")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# HEADLINE 3: YOE floor falling
# ---------------------------------------------------------------------------

def viz_yoe_floor():
    _style_setup()
    df = pd.read_csv(TABLES_DIR / "S12_yoe_trajectory.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel (a) Mean YOE
    for lvl in ["junior", "mid", "senior"]:
        sub = df[df["seniority_3level"] == lvl].set_index("period").reindex(PERIODS)
        axes[0].plot(PERIODS, sub["mean_yoe"], "-o", color=PAL[lvl],
                     linewidth=2.5, markersize=9, label=lvl)
        for p, v in zip(PERIODS, sub["mean_yoe"]):
            if pd.notna(v):
                axes[0].annotate(f"{v:.2f}", (PERIODS.index(p), v),
                                 xytext=(0, 10), textcoords="offset points",
                                 ha="center", fontsize=9, color=PAL[lvl], fontweight="bold")
    axes[0].set_xlabel("period")
    axes[0].set_ylabel("mean yoe_min_years_llm (years)")
    axes[0].set_title("Mean years of experience required, by seniority")
    axes[0].legend(title="seniority_3level")
    axes[0].set_ylim(0, 8)

    # Panel (b) Median YOE
    for lvl in ["junior", "mid", "senior"]:
        sub = df[df["seniority_3level"] == lvl].set_index("period").reindex(PERIODS)
        axes[1].plot(PERIODS, sub["median_yoe"], "-s", color=PAL[lvl],
                     linewidth=2.5, markersize=9, label=lvl)
    axes[1].set_xlabel("period")
    axes[1].set_ylabel("median yoe_min_years_llm (years)")
    axes[1].set_title("Median years of experience required (junior: 2 → 1, senior: 6 → 5)")
    axes[1].legend(title="seniority_3level")
    axes[1].set_ylim(0, 8)

    fig.suptitle("Years of experience required is falling, not rising")
    fig.text(0.5, -0.01,
             "Classic 'scope inflation' predicts junior YOE rising. LLM-YOE shows the opposite across all levels.  Source: eda/tables/S12_yoe_trajectory.csv",
             ha="center", fontsize=8, style="italic", color="#666")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# HEADLINE 4: vendor leaderboard
# ---------------------------------------------------------------------------

def viz_vendor_leaderboard():
    _style_setup()
    rates = pd.read_csv(TABLES_DIR / "S13_vendor_mentions.csv")
    leaderboard = pd.read_csv(TABLES_DIR / "S13_vendor_leaderboard_2026_04.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # (a) horizontal leaderboard for 2026-04
    lb = leaderboard.sort_values("rate_2026_04")
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(lb)))
    bars = axes[0].barh(lb["vendor"], lb["rate_2026_04"] * 100, color=colors, edgecolor="white")
    for bar, rate in zip(bars, lb["rate_2026_04"]):
        axes[0].text(rate * 100 + 0.05, bar.get_y() + bar.get_height() / 2,
                     f"{rate*100:.2f}%", va="center", fontsize=9, fontweight="bold")
    axes[0].set_xlabel("share of SWE postings mentioning vendor (%)")
    axes[0].set_title("2026-04 leaderboard: Copilot leads, Cursor emerged from near-zero")
    axes[0].set_xlim(0, lb["rate_2026_04"].max() * 100 * 1.15)

    # (b) trajectory of top 5
    top5 = leaderboard.head(5)["vendor"].tolist()
    cmap = {v: plt.cm.tab10(i / 10) for i, v in enumerate(top5)}
    for v in top5:
        axes[1].plot(rates["period"], rates[f"{v}_rate"] * 100, "-o",
                     label=v, linewidth=2.5, markersize=8, color=cmap[v])
        # label end point
        last = rates[f"{v}_rate"].iloc[-1] * 100
        axes[1].annotate(f"{last:.2f}%", (3, last), xytext=(8, 0),
                         textcoords="offset points", va="center",
                         fontsize=9, color=cmap[v], fontweight="bold")
    axes[1].set_xlabel("period")
    axes[1].set_ylabel("mention rate (%)")
    axes[1].set_title("Top-5 vendor trajectory (Claude growth fastest)")
    axes[1].legend(loc="upper left")

    fig.suptitle("Dev-tool vendor leaderboard in software-engineer postings")
    fig.text(0.5, -0.01,
             "First published vendor-share table for dev tools from labor demand. ChatGPT brand plateauing; Claude/Cursor climbing fastest. Source: eda/tables/S13_*.csv",
             ha="center", fontsize=8, style="italic", color="#666")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# HEADLINE 5: Big Tech AI density gap
# ---------------------------------------------------------------------------

def viz_bigtech_density():
    _style_setup()
    df = pd.read_csv(TABLES_DIR / "S10_core_bigtech_vs_rest.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    bt = df[df["tier"] == "big_tech"].set_index("period").reindex(PERIODS)
    rest = df[df["tier"] == "rest"].set_index("period").reindex(PERIODS)

    # (a) volume share
    axes[0].bar(PERIODS, bt["volume_share"] * 100, color=PAL["big_tech"], edgecolor="white")
    for i, v in enumerate(bt["volume_share"]):
        axes[0].text(i, v * 100 + 0.15, f"{v*100:.2f}%", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Big Tech share of SWE postings (%)")
    axes[0].set_title("(a) BT volume share ROSE 2.4% → 7.0% (opposite of layoff prior)")
    axes[0].set_ylim(0, max(bt["volume_share"] * 100) * 1.4)

    # (b) AI rate by tier
    x = np.arange(len(PERIODS))
    width = 0.38
    axes[1].bar(x - width/2, bt["ai_rate"] * 100, width, color=PAL["big_tech"],
                label="Big Tech", edgecolor="white")
    axes[1].bar(x + width/2, rest["ai_rate"] * 100, width, color=PAL["rest"],
                label="rest", edgecolor="white")
    for i, (b, r) in enumerate(zip(bt["ai_rate"], rest["ai_rate"])):
        axes[1].text(i - width/2, b * 100 + 0.5, f"{b*100:.0f}%", ha="center",
                     fontsize=9, fontweight="bold")
        axes[1].text(i + width/2, r * 100 + 0.5, f"{r*100:.0f}%", ha="center",
                     fontsize=9, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(PERIODS)
    axes[1].set_ylabel("AI-vocab rate (%)")
    axes[1].set_title("(b) BT AI density 17pp HIGHER than rest in 2026")
    axes[1].legend()

    fig.suptitle("Big Tech: more posting volume AND more AI language")
    fig.text(0.5, -0.01,
             "Surprising direction: BT posting share rose, not fell. Pair with named-firm layoff timing in a follow-up. Source: eda/tables/S10_core_bigtech_vs_rest.csv",
             ha="center", fontsize=8, style="italic", color="#666")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# DISPROVEN: industry spread on LinkedIn
# ---------------------------------------------------------------------------

def viz_disproven_industry_spread():
    _style_setup()
    df = pd.read_csv(TABLES_DIR / "S8_nontech_industry_share.csv")
    fig, ax = plt.subplots(figsize=(11, 6))

    valid = df.dropna(subset=["nontech_share_of_labeled"])
    bars = ax.bar(valid["period"], valid["nontech_share_of_labeled"] * 100,
                  color=PAL["control"], edgecolor="white")
    for b, v in zip(bars, valid["nontech_share_of_labeled"]):
        ax.text(b.get_x() + b.get_width()/2, v * 100 + 0.7,
                f"{v*100:.1f}%", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("non-tech industry share of SWE postings (with labeled industry) (%)")
    ax.set_title("Software jobs are not visibly spreading to non-tech industries on LinkedIn")
    ax.set_ylim(0, 70)
    ax.text(0.5, 0.94,
            "The Economist predicted SWE jobs spreading to non-tech industries (retail +12%, property +75%, construction +100%).\n"
            "On LinkedIn posting share, the non-tech share is FLAT at ~55% across 2024 → 2026.",
            transform=ax.transAxes, ha="center", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffe5e5", edgecolor=PAL["disproven"]))
    fig.text(0.5, -0.01,
             "Caveat: Economist used BLS occupational data (employed people), not LinkedIn posting volumes. Both can be true. Source: eda/tables/S8_nontech_industry_share.csv",
             ha="center", fontsize=8, style="italic", color="#666")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# DISPROVEN 3: junior-first automation
# ---------------------------------------------------------------------------

def viz_disproven_juniorfirst():
    _style_setup()
    df = pd.read_csv(TABLES_DIR / "S1_core_ai_vocab.csv")
    sub = df[df["period"] == "2026-04"].copy()
    order = ["junior", "mid", "senior", "unknown"]
    sub = sub.set_index("seniority_3level").reindex(order).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(sub["seniority_3level"], sub["ai_rate"] * 100,
                  color=[PAL[l] for l in sub["seniority_3level"]],
                  edgecolor="white")
    for b, v in zip(bars, sub["ai_rate"]):
        ax.text(b.get_x() + b.get_width()/2, v * 100 + 0.7,
                f"{v*100:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("AI-vocab rate, 2026-04 (%)")
    ax.set_title("AI is not hitting junior engineers first")
    ax.set_ylim(0, max(sub["ai_rate"] * 100) * 1.35)
    ax.text(0.5, 0.94,
            "If automation hit juniors first (classic scope-inflation story), juniors would lead AI-vocab adoption.\n"
            "Observed: AI rate is FLAT across junior/mid/senior in 2026-04. Senior-restructuring (H5b) is the consistent reading.",
            transform=ax.transAxes, ha="center", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffe5e5", edgecolor=PAL["disproven"]))
    fig.tight_layout()
    return fig


