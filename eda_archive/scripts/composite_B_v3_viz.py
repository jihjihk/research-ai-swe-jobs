"""
Composite B v3 — figure code for the role-landscape evolution panel.

Four functions, each returning a matplotlib Figure. Called from
`findings_consolidated_2026-04-21.ipynb` as inline displays following the
existing headlines_viz / consolidated_viz pattern.

Inputs (produced by `composite_B_v3_evolution.py`):
  eda/artifacts/composite_B_v3llm_labels.parquet
  eda/tables/composite_B_v3llm_families_k30.csv
  eda/tables/composite_B_v3llm_families_k30_ai_split.csv
  eda/tables/composite_B_v3llm_emergence_k30.csv
  eda/tables/composite_B_v3llm_drift_k30.csv

See eda/research_memos/composite_B_v3_findings.md for the narrative.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from headlines_viz import PAL, _style_setup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "eda" / "tables"
ARTIFACTS_DIR = PROJECT_ROOT / "eda" / "artifacts"


# Human-readable archetype labels, derived from top c-TF-IDF terms in
# composite_B_v3llm_families_k30.csv. Family 0 is the data/AI mega-cluster
# that we split post-hoc via the AI-vocabulary regex.
FAMILY_LABELS: dict[int, str] = {
    0:  "Data + AI/ML",
    1:  "Cloud / DevOps",
    2:  "Embedded systems",
    3:  "Enterprise Java / .NET",
    4:  "Salesforce / frontend",
    5:  "Cybersecurity",
    6:  "QA / test automation",
    7:  "Mobile (iOS / Android)",
    8:  "ServiceNow consulting",
    9:  "Generic software dev",
    10: "SRE / reliability",
    11: "Robotics / CV",
    12: "Medical software",
    13: "Technical leadership",
    14: "PLC / industrial automation",
    15: "SAP / Oracle / mainframe",
    16: "Quant / trading",
    17: "Payments / fintech",
    18: "GPU / inference",
    19: "QA / compliance",
    20: "PHP / WordPress",
    21: "Defense contractor SWE",
    22: "Ad tech / streaming",
    23: "Network automation",
    24: "Automotive embedded",
    25: "Google fullstack",
    26: "Infosys / offshored IT",
    27: "Cleared / Northrop",
    28: "Rust backend",
}

# Families to foreground in the coloured UMAP (all others plotted grey).
# Picked to span: the AI/data split, the biggest shrinkers, the biggest
# growers, and stable-but-rewritten.
FOREGROUND_FAMILIES = [0, 1, 3, 5, 7, 10, 11, 18, 23, 28]

# Colour for each foreground family. Family 0 handled separately via the
# AI-split. The rest are chosen from matplotlib tab20 plus a few picks.
FAMILY_COLOURS: dict = {
    "0_AI":    "#d62728",  # red — the AI-coded surge
    "0_nonAI": "#ff7f0e",  # amber — data work without AI vocabulary
    1:  "#1f77b4",   # blue — cloud / DevOps
    3:  "#17365c",   # dark navy — shrinking enterprise Java
    5:  "#2ca02c",   # green — cybersecurity growth
    7:  "#8c564b",   # brown — mobile
    10: "#9edae5",   # pale cyan — SRE / reliability
    11: "#9467bd",   # purple — robotics
    18: "#e377c2",   # pink — GPU / inference
    23: "#bcbd22",   # olive — network automation
    28: "#17becf",   # teal — Rust backend
}
GREY = "#cccccc"


def _load_labels() -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS_DIR / "composite_B_v3llm_labels.parquet")


def _family_colour(family_split: str) -> str:
    """Map a family_split value ("0_AI", "0_nonAI", "1", "23", ..., "-1") to a colour."""
    if family_split in ("0_AI", "0_nonAI"):
        return FAMILY_COLOURS[family_split]
    try:
        fid = int(family_split)
    except ValueError:
        return GREY
    if fid == -1:
        return GREY
    return FAMILY_COLOURS.get(fid, GREY)


def _label_for_split(family_split: str) -> str:
    if family_split == "0_AI":
        return "AI-engineering (within data/AI cluster)"
    if family_split == "0_nonAI":
        return "Data work without AI vocabulary"
    try:
        fid = int(family_split)
    except ValueError:
        return family_split
    return FAMILY_LABELS.get(fid, f"Family {fid}")


# ---------------------------------------------------------------------------
# Panel A — the map, 2×2 small multiples
# ---------------------------------------------------------------------------

def viz_composite_b_map():
    """2×2 UMAP scatter: rows (SWE, SWE-adjacent) × cols (2024, 2026).

    Foreground families coloured distinctly; everything else grey. Family 0
    is split into its AI-coded and non-AI-coded sub-cohorts (same UMAP
    location, different story)."""
    _style_setup()
    df = _load_labels().copy()
    # Stable point order: grey background first, then coloured dots on top
    df["_colour"] = df["family_split_k30"].map(_family_colour)
    df["_is_foreground"] = df["_colour"] != GREY

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    cells = [
        ("SWE",          "2024", axes[0, 0]),
        ("SWE",          "2026", axes[0, 1]),
        ("SWE-adjacent", "2024", axes[1, 0]),
        ("SWE-adjacent", "2026", axes[1, 1]),
    ]
    for role, period, ax in cells:
        sub = df[(df["role_group"] == role) & (df["period_bucket"] == period)]
        bg = sub[~sub["_is_foreground"]]
        fg = sub[sub["_is_foreground"]]
        ax.scatter(bg["umap_x"], bg["umap_y"], s=3, alpha=0.25, c=GREY, linewidth=0)
        ax.scatter(fg["umap_x"], fg["umap_y"], s=5, alpha=0.55,
                   c=fg["_colour"].values, linewidth=0)
        ax.set_title(f"{role} · {period}  (n = {len(sub):,})",
                     fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)

    # Shared legend below the grid
    legend_keys = ["0_AI", "0_nonAI"] + FOREGROUND_FAMILIES[1:]  # skip family 0 raw
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=8,
                       markerfacecolor=_family_colour(str(k) if isinstance(k, int) else k),
                       markeredgecolor="white",
                       label=_label_for_split(str(k) if isinstance(k, int) else k))
               for k in legend_keys]
    handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=8,
                           markerfacecolor=GREY, markeredgecolor="white",
                           label="Other archetypes (tail) + uncategorised"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                bbox_to_anchor=(0.5, -0.08), fontsize=9)

    fig.suptitle("Composite B · The skill map reorganised between 2024 and 2026",
                  fontsize=13, fontweight="bold", y=0.995)
    fig.text(0.5, 0.955,
              "Same UMAP coordinates across all four cells. Density shifts toward the data/AI region "
              "(red + amber) on both SWE and SWE-adjacent sides; the red AI-coded portion is the "
              "specific surge underneath.",
              ha="center", fontsize=9, style="italic", color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    return fig


# ---------------------------------------------------------------------------
# Panel B — compositional delta, diverging bars
# ---------------------------------------------------------------------------

def _composition_deltas() -> pd.DataFrame:
    df = _load_labels()
    by = (df.groupby(["role_group", "period_bucket", "family_split_k30"])
            .size().unstack("family_split_k30", fill_value=0))
    pct = by.div(by.sum(axis=1), axis=0) * 100
    swe = pct.loc["SWE", "2026"] - pct.loc["SWE", "2024"]
    adj = pct.loc["SWE-adjacent", "2026"] - pct.loc["SWE-adjacent", "2024"]
    out = pd.DataFrame({"swe_delta_pp": swe, "adj_delta_pp": adj}).reset_index()
    out.columns = ["family_split", "swe_delta_pp", "adj_delta_pp"]
    out["label"] = out["family_split"].apply(_label_for_split)
    return out


def viz_composite_b_deltas():
    """Diverging horizontal bars: percentage-point change in corpus share
    per family-split, SWE and SWE-adjacent shown side-by-side."""
    _style_setup()
    d = _composition_deltas()
    # Drop noise (-1) and tiny long-tail families (< 0.2pp absolute move on
    # either side) for readability.
    d = d[d["family_split"] != "-1"]
    mask = (d["swe_delta_pp"].abs() >= 0.2) | (d["adj_delta_pp"].abs() >= 0.2)
    d = d[mask].copy()
    # Sort: biggest positive SWE delta at top
    d = d.sort_values("swe_delta_pp", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, 0.32 * len(d))))
    y = np.arange(len(d))
    h = 0.38
    swe_colour = ["#2ca02c" if v >= 0 else "#d62728" for v in d["swe_delta_pp"]]
    adj_colour = ["#1f7a3b" if v >= 0 else "#8b1a1a" for v in d["adj_delta_pp"]]
    ax.barh(y + h/2, d["swe_delta_pp"], height=h, color=swe_colour,
             edgecolor="white", label="SWE")
    ax.barh(y - h/2, d["adj_delta_pp"], height=h, color=adj_colour, alpha=0.55,
             edgecolor="white", label="SWE-adjacent")
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("2026 share minus 2024 share (percentage points)")
    ax.set_title("What grew, what shrank — by archetype, by role group",
                  fontsize=12, fontweight="bold")
    # Legend via proxy patches (colours in the bars encode sign, not group)
    handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=12,
                markerfacecolor="#2ca02c", markeredgecolor="white", label="SWE"),
        Line2D([0], [0], marker="s", linestyle="", markersize=12,
                markerfacecolor="#1f7a3b", markeredgecolor="white", alpha=0.55,
                label="SWE-adjacent"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.25)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.text(0.5, -0.01,
              "Archetype families from the k=30 joint BERTopic fit; the Data+AI mega-cluster (Family 0) is split "
              "post-hoc with the Article-A AI-vocabulary regex into AI-coded and non-AI sub-cohorts. "
              "Families with under 0.2pp movement on both sides omitted. Noise bucket omitted.",
              ha="center", fontsize=8.5, style="italic", color="#666")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel C — growth × emergence scatter
# ---------------------------------------------------------------------------

def viz_composite_b_growth_emergence():
    """Scatter of per-family 2026-growth ratio vs emergence share.

    Emergence share = fraction of a family's 2026 postings whose nearest
    2024 neighbour in embedding space sits further away than the 95th
    percentile of 2024→2024 within-period nearest-neighbour distances.
    Bubble size = 2026 posting count. Colour = foreground vs. background."""
    _style_setup()
    emerge = pd.read_csv(TABLES_DIR / "composite_B_v3llm_emergence_k30.csv")
    fam = pd.read_csv(TABLES_DIR / "composite_B_v3llm_families_k30.csv")
    d = emerge.merge(fam[["family_id", "growth_ratio", "n", "share_swe"]],
                       on="family_id", how="left")
    # Only families with >= 50 postings in 2026 for stability
    d = d[d["n_2026"] >= 50].copy()
    d["label"] = d["family_id"].map(FAMILY_LABELS).fillna(d["family_id"].astype(str))
    d["colour"] = d["family_id"].apply(
        lambda f: FAMILY_COLOURS.get(f, "#999999") if f in FOREGROUND_FAMILIES else GREY
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    sizes = np.sqrt(d["n_2026"].values) * 6
    ax.scatter(d["growth_ratio"], d["emergence_share"] * 100,
                s=sizes, c=d["colour"].values, alpha=0.7,
                edgecolor="white", linewidth=1.2)

    # Annotate: all foreground families + any non-foreground with extreme stats
    annotate_mask = (
        d["family_id"].isin(FOREGROUND_FAMILIES)
        | (d["growth_ratio"] <= 0.5)
        | (d["emergence_share"] >= 0.2)
    )
    for _, r in d[annotate_mask].iterrows():
        ax.annotate(r["label"],
                     xy=(r["growth_ratio"], r["emergence_share"] * 100),
                     xytext=(6, 4), textcoords="offset points",
                     fontsize=8.5,
                     color="#222" if r["family_id"] in FOREGROUND_FAMILIES else "#666")

    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.axhline(10.0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0])
    ax.set_xticklabels(["0.3×", "0.5×", "0.7×", "1×", "1.5×", "2×", "3×", "5×"])
    ax.set_xlabel("Growth ratio (2026 / 2024 posting count)")
    ax.set_ylabel("Emergence share — % of 2026 postings unlike anything in 2024")
    ax.set_title("Growing, and growing into new territory — are not the same thing",
                  fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2)

    # Quadrant labels
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.text(0.97, 0.97, "New frontiers\n(growing AND unlike 2024)",
             transform=ax.transAxes, ha="right", va="top", fontsize=9,
             color="#555", style="italic")
    ax.text(0.97, 0.03, "Mainstream growth\n(growing but similar to 2024)",
             transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
             color="#555", style="italic")
    ax.text(0.03, 0.97, "Quiet rewrites",
             transform=ax.transAxes, ha="left", va="top", fontsize=9,
             color="#555", style="italic")
    ax.text(0.03, 0.03, "Stable / shrinking legacy",
             transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
             color="#555", style="italic")

    # Annotate the Data+AI bubble with the split finding
    dai = d[d["family_id"] == 0]
    if not dai.empty:
        xb, yb = float(dai["growth_ratio"].iloc[0]), float(dai["emergence_share"].iloc[0]) * 100
        ax.annotate("Inside: AI-coded sub-cohort\ngrew 8.5× (520 → 4,447);\nnon-AI data work grew 1.35×",
                     xy=(xb, yb), xytext=(xb * 0.55, yb + 7),
                     fontsize=8.5, color="#222",
                     bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5f0",
                                edgecolor=FAMILY_COLOURS["0_AI"], linewidth=0.8),
                     arrowprops=dict(arrowstyle="->", color="#888", linewidth=0.7))

    fig.text(0.5, -0.02,
              "Bubble size ∝ √(2026 posting count). Emergence threshold calibrated on the 95th percentile of "
              "within-2024 nearest-neighbour cosine distances (0.314). Families with fewer than 50 2026 postings omitted.",
              ha="center", fontsize=8.5, style="italic", color="#666")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel D — within-family drift (words that entered / exited top-20)
# ---------------------------------------------------------------------------

def _wrap_words(words: list, per_line: int = 3) -> str:
    lines = [", ".join(words[i:i + per_line])
             for i in range(0, len(words), per_line)]
    return ",\n".join(lines)


def viz_composite_b_drift_words():
    """Four typographic panels showing vocabulary turnover inside four
    families. Picked for narrative: the AI/data cluster's internal rotation,
    enterprise Java modernising even as it shrinks, DevOps rebranding as
    platform/reliability, and QA embedding into development."""
    _style_setup()
    drift = pd.read_csv(TABLES_DIR / "composite_B_v3llm_drift_k30.csv")
    picks = [
        (0, "Data + AI/ML (Family 0): vocabulary rotated as it grew 2×",
             "From generic 'ai' as a buzzword to named practices."),
        (3, "Enterprise Java / .NET (Family 3): modernising as it halves",
             "Shrinking by half — surviving postings read cloud-native."),
        (1, "Cloud / DevOps (Family 1): 'DevOps' exits the vocabulary",
             "Reliability and monitoring fill the gap left by the old label."),
        (6, "QA / test automation (Family 6): 'QA' exits as testing embeds",
             "The QA label fades; testing becomes part of development."),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for (fid, title, subtitle), ax in zip(picks, axes.flat):
        row = drift[drift["family_id"] == fid]
        if row.empty:
            ax.axis("off")
            continue
        r = row.iloc[0]
        entered = [t.strip() for t in str(r["entered_top20"]).split(",") if t.strip()]
        exited  = [t.strip() for t in str(r["exited_top20"]).split(",")  if t.strip()]
        ax.axis("off")
        ax.text(0.02, 0.97, title, transform=ax.transAxes,
                fontsize=11.5, fontweight="bold", va="top")
        ax.text(0.02, 0.90, subtitle, transform=ax.transAxes,
                fontsize=9.5, style="italic", color="#555", va="top")

        # Two columns of words, each explicitly wrapped to fit.
        ax.text(0.02, 0.80, "Entered top-20 by 2026", transform=ax.transAxes,
                fontsize=10, fontweight="bold", color="#2ca02c", va="top")
        ax.text(0.02, 0.73, _wrap_words(entered, per_line=3),
                transform=ax.transAxes, fontsize=10, color="#1a1a1a",
                va="top", family="monospace", linespacing=1.4)
        ax.text(0.54, 0.80, "Exited top-20 from 2024", transform=ax.transAxes,
                fontsize=10, fontweight="bold", color="#d62728", va="top")
        ax.text(0.54, 0.73, _wrap_words(exited, per_line=3),
                transform=ax.transAxes, fontsize=10, color="#1a1a1a",
                va="top", family="monospace", linespacing=1.4)
        # n_2024 / n_2026 at the bottom
        ax.text(0.02, 0.04,
                f"n_2024 = {int(r['n_2024']):,}    n_2026 = {int(r['n_2026']):,}",
                transform=ax.transAxes, fontsize=8.5, color="#888")
        # Light frame around each panel for visual separation
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)

    fig.suptitle("Composite B · Same archetype, different words — four families rewriting themselves",
                  fontsize=12, fontweight="bold", y=0.995)
    fig.text(0.5, -0.005,
              "Top-20 c-TF-IDF terms within each archetype, compared between the family's 2024 and 2026 subsets. "
              "'Entered' = newly top-20 in 2026; 'exited' = no longer top-20.",
              ha="center", fontsize=8.5, style="italic", color="#666")
    fig.tight_layout(rect=(0, 0.01, 1, 0.96))
    return fig
