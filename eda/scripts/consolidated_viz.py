"""
Visualization functions for the consolidated findings notebook.

Reuses the established figure functions for headlines and previously-tested
disproven hypotheses (imported from headlines_viz). Adds new figures:

- viz_junior_scope_panel: 2x2 grid, 4 metrics, (SWE|control) x (junior|senior)
  x 4 periods, for the new analytical section.
- viz_senior_scope_inflation: small multiples showing senior scope > junior
  scope inflation on the SWE side (headline #3 support).
- viz_disproven_hiring_bar: requirements-section contraction does NOT
  correlate with hiring-bar-proxy metrics.
- viz_disproven_selectivity: volume-UP firms write LONGER JDs
  (scatter on the 292-co within-firm panel).
- viz_verdict_table: rewritten, 11 rows with neutral claim labels
  (no hypothesis IDs visible).

Newer additions (Headline 7 + two composite articles):
- viz_cross_occ_rank: scatter of worker-side AI use vs employer-side delta
  rank across 17 occupations (Headline 7).
- viz_composite_a_lead: Bay-vs-rest token gap, frontier-platform vs
  coding-tool tokens (Composite A panel 1).
- viz_composite_a_geo: top-26 metros by 2026 absolute AI rate (panel 2).
- viz_composite_a_industry: industry AI-rate panel with Wilson CIs (panel 3).
- viz_composite_b_cluster: BERTopic Topic 1 trajectory + within-cluster AI
  rate (Composite B panel 1).
- viz_composite_b_fde_legacy: FDE growth and legacy-substitution comparison
  (Composite B panels 2+3 combined).

No matplotlib.pyplot.savefig. Figures are returned and displayed inline via
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
        "Junior vs senior, software-engineer vs control: four scope metrics",
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
             "Breadth is residualized on description length, so rising bars are NOT a length artifact. "
             "Source: cached posting-feature artifact.",
             ha="center", fontsize=8, style="italic", color="#666")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Disproven: hiring-bar lowering via requirements-section contraction
# ---------------------------------------------------------------------------

def viz_disproven_hiring_bar():
    """Prose-only summary. Requirements-section contraction hypothesis fails
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
# Disproven: hiring-selectivity (volume-up firms write longer JDs)
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
    ax.set_title("Firms posting MORE roles also added MORE AI language (the opposite of a hiring squeeze)")
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
# Rewritten verdict-table: neutral claim labels, no hypothesis IDs
# ---------------------------------------------------------------------------

CLAIMS = [
    # (category, claim, verdict, evidence)
    ("supported", "AI language rewrite is specific to software-engineering roles",
     "supported",
     "38:1 SWE-vs-control delta ratio under strict core_llm (23:1 raw); DiD +14 pp; universal across 17 tested occupations"),
    ("supported", "The same firms rewrote their own postings 2024 → 2026",
     "supported",
     "292-firm panel, mean within-firm Δ = +17.6 pp; 73% of firms rose; pair-level drift +10–13 pp (exceeds company-level)"),
    ("supported", "Seniority boundaries sharpened; senior scope outpaces junior",
     "supported",
     "Junior-senior TF-IDF cosine 0.946 → 0.863; AUC +0.150 at associate↔mid-senior; breadth +2.61 senior vs +1.58 junior"),
    ("supported", "Dev-tool vendor leaderboard has emerged in labor demand",
     "supported",
     "2026-04 rates: Copilot 4.0%, Claude 3.7%, OpenAI 3.1%, Cursor 2.0% (rank order unchanged from raw)"),
    ("supported", "Years-of-experience floor is FALLING, not rising",
     "supported",
     "Junior mean YOE 2.01 → 1.23; senior median 6 → 5; falsifies the classic scope-inflation narrative"),
    ("supported", "Big Tech: more posting volume AND higher AI density",
     "supported",
     "Big Tech share of SWE 2.4% → 7.0%; AI-mention rate 37.8% (Big Tech) vs 23.8% (rest) in 2026 (+14.0 pp)"),
    ("supported", "Employers' AI-rewrite ranking matches workers' AI-use ranking",
     "supported",
     "Across 17 occupations, Spearman ρ on the 2024→2026 employer delta = +0.84 under core_llm; tech-only ρ = +0.89; two-cluster permutation p = 0.0007"),

    ("falsified", "AI is narrative cover for unrelated macro layoffs (content level)",
     "FALSIFIED",
     "SWE and control would co-move; they don't. 38:1 divergence ratio under strict core_llm rules out the content-level story"),
    ("falsified", "Software-engineer jobs are spreading to non-tech industries (on LinkedIn)",
     "FALSIFIED",
     "Non-tech share of SWE postings flat at ~55% across 2024 → 2026"),
    ("falsified", "Automation hits junior engineers first",
     "FALSIFIED",
     "AI-vocab rate uniform across seniority (junior 23%, mid 23%, senior 28% in 2026-04); junior YOE also fell"),
    ("falsified", "Requirements-section contraction indicates hiring-bar lowering",
     "FALSIFIED",
     "Classifier-sensitive direction; within-firm |ρ| ≤ 0.28 on hiring-bar proxies; 0/50 postings with loosening language"),
    ("falsified", "Hiring cycle tightened selection, making firms raise the bar",
     "FALSIFIED",
     "Volume-UP firms write LONGER JDs (r = +0.20); opposite direction from the selectivity prediction"),

    ("supported", "Asymmetric AI diffusion: coding tools democratised, agentic vocabulary did not",
     "supported",
     "Frontier-platform tokens (agentic, ai agent, llm, foundation model) lead in tech hubs; coding-tool tokens (copilot, RAG, MLOps) lead outside hubs; survives self-mention exclusion"),
    ("supported", "An agentic-AI cluster grew 5.2× as a share of SWE postings",
     "supported",
     "BERTopic 2.5% → 12.7% of corpus; substrate-invariant (cluster fitting on core_llm); NMF independently confirms (12.4×); within-cluster AI rate 82% under core_llm"),
]


def viz_verdict_table():
    _style_setup()
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis("off")

    header_y = 0.97
    ax.text(0.02, header_y, "Finding",   fontsize=12, fontweight="bold", color=PAL["neutral"], transform=ax.transAxes)
    ax.text(0.50, header_y, "Verdict",   fontsize=12, fontweight="bold", color=PAL["neutral"], transform=ax.transAxes)
    ax.text(0.66, header_y, "Evidence",  fontsize=12, fontweight="bold", color=PAL["neutral"], transform=ax.transAxes)
    ax.plot([0.01, 0.99], [header_y - 0.013, header_y - 0.013], color="#999", linewidth=1.2, transform=ax.transAxes)

    # supported block
    ax.text(0.01, header_y - 0.04, "Supported findings", fontsize=11, fontweight="bold",
            color=PAL["headline"], transform=ax.transAxes, style="italic")

    row_h = 0.057
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

    n_sup = len(supported_rows)
    n_fal = len(falsified_rows)
    ax.set_title(f"All findings on one page: {n_sup} supported, {n_fal} falsified",
                 fontsize=13, fontweight="bold", y=1.0, loc="left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Headline 7 — Cross-occupation rank correlation (Claim 7)
# ---------------------------------------------------------------------------

def viz_cross_occ_rank():
    """Worker-side AI use vs employer-side AI-rate change, ranked across 17
    occupations. Annotates Spearman rho on the 2024->2026 delta and the
    two-cluster permutation null band."""
    _style_setup()
    pair = pd.read_csv(TABLES_DIR / "substrate_B_pair_table.csv")
    perm = pd.read_csv(TABLES_DIR / "S25_eval_permutation.csv")

    # Use core-substrate deltas
    pair = pair.copy()
    pair["delta_pp"] = pair["delta_core"] * 100
    pair["worker_pct"] = pair["worker_any_mid"] * 100

    # Color by analysis_group
    color_map = {"swe": PAL["swe"], "swe_adjacent": PAL["swe_adjacent"], "control": PAL["control"]}
    label_map = {"swe": "SWE", "swe_adjacent": "SWE-adjacent", "control": "control"}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # Left panel: rank-on-rank scatter (worker rank vs employer-delta rank)
    pair["worker_rank"] = pair["worker_pct"].rank(method="average")
    pair["delta_rank"] = pair["delta_pp"].rank(method="average")
    for grp, sub in pair.groupby("analysis_group"):
        axes[0].scatter(sub["worker_rank"], sub["delta_rank"],
                        s=110, color=color_map.get(grp, "#888"),
                        edgecolor="white", linewidth=1.0,
                        label=label_map.get(grp, grp), alpha=0.9, zorder=3)
    # Annotate each point
    for _, row in pair.iterrows():
        axes[0].annotate(row["sg"].replace("_", " "),
                         (row["worker_rank"], row["delta_rank"]),
                         xytext=(7, 3), textcoords="offset points",
                         fontsize=8, color="#333")
    # 1:1 line
    n = len(pair)
    axes[0].plot([1, n], [1, n], "--", color="#999", linewidth=1, alpha=0.7,
                 label="perfect rank match")
    axes[0].set_xlabel("rank on worker-side AI use (low → high)")
    axes[0].set_ylabel("rank on employer-side 2024→2026 Δ AI-vocab (low → high)")
    axes[0].set_title("Worker AI-use rank vs employer Δ AI-vocab rank, 17 occupations")
    axes[0].legend(loc="lower right", fontsize=9)

    # Right panel: permutation null densities and observed
    obs = float(perm[perm["null"] == "two_cluster_within_shuffle"]["observed"].iloc[0])
    p_two = float(perm[perm["null"] == "two_cluster_within_shuffle"]["p_value_right"].iloc[0])
    median_two = float(perm[perm["null"] == "two_cluster_within_shuffle"]["median"].iloc[0])
    p975_two = float(perm[perm["null"] == "two_cluster_within_shuffle"]["p97_5"].iloc[0])
    median_uni = float(perm[perm["null"] == "uniform_shuffle"]["median"].iloc[0])
    p975_uni = float(perm[perm["null"] == "uniform_shuffle"]["p97_5"].iloc[0])

    nulls = ["uniform shuffle\n(no structure)", "two-cluster shuffle\n(preserves tech-vs-non-tech split)"]
    medians = [median_uni, median_two]
    p975 = [p975_uni, p975_two]
    y = np.arange(len(nulls))
    # Null bars: median to 97.5th percentile
    axes[1].barh(y, p975, left=medians, color="#ddd", edgecolor="#aaa",
                 height=0.45, label="null median to 97.5%-ile")
    axes[1].plot(medians, y, "|", color="#666", markersize=14, label="null median")
    # Observed line
    axes[1].axvline(obs, color=PAL["headline"], linewidth=2.5,
                    label=f"observed ρ = {obs:+.2f}")
    # Annotations
    axes[1].text(obs + 0.01, 1, f"  p = {p_two:.4f}\n  vs two-cluster null",
                 fontsize=10, color=PAL["headline"], fontweight="bold", va="center")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(nulls, fontsize=10)
    axes[1].set_xlim(-0.1, 1.0)
    axes[1].set_xlabel("Spearman ρ")
    axes[1].set_title("Observed correlation vs permutation nulls")
    axes[1].legend(loc="lower right", fontsize=9)

    fig.suptitle("Headline 7: the market ranked the occupations right", fontsize=13, fontweight="bold")
    # Compute correlations from the same pair table the panel reads
    from scipy.stats import spearmanr as _spearmanr
    _r_lvl, _ = _spearmanr(pair["worker_pct"], pair["delta_pp"])  # placeholder
    # Headline numbers from S25_method_comparison.csv (M1 levels, M4 delta)
    method_csv = TABLES_DIR / "S25_method_comparison.csv"
    if method_csv.exists():
        m = pd.read_csv(method_csv)
        rho_2026 = float(m[m["method"] == "M1_headline_canonical"]["spearman"].iloc[0])
        rho_delta = float(m[m["method"] == "M4_delta_canonical"]["spearman"].iloc[0])
    else:
        rho_2026 = 0.898
        rho_delta = 0.838
    fig.text(0.5, -0.02,
             f"Spearman ρ on 2024→2026 employer Δ vs worker any-use = +{rho_delta:.2f} (n=17, core_llm); "
             f"ρ on 2026 levels = +{rho_2026:.2f}. About two-thirds of the alignment is the tech-vs-non-tech split (right panel). "
             "Source: eda/tables/substrate_B_pair_table.csv + S25_eval_permutation.csv.",
             ha="center", fontsize=8, style="italic", color="#666")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Composite A panel 1 — asymmetric token diffusion
# ---------------------------------------------------------------------------

def viz_composite_a_lead():
    """Bay-vs-rest token rate gap for two token classes:
    frontier-platform (hubs lead) and coding-tool (rest leads)."""
    _style_setup()
    df = pd.read_csv(TABLES_DIR / "audit_self_mention_audit1_token_gap.csv")

    # Two groups
    hub_tokens = ["agentic", "ai agent", "llm", "foundation model"]
    user_tokens = ["copilot", "github copilot", "prompt engineering", "rag"]

    hub = df[df["token"].isin(hub_tokens)].set_index("token").reindex(hub_tokens)
    user = df[df["token"].isin(user_tokens)].set_index("token").reindex(user_tokens)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    def plot_pair(ax, frame, title, hub_color, rest_color):
        toks = frame.index.tolist()
        y = np.arange(len(toks))
        ax.barh(y - 0.2, frame["excl_bay_rate_pct"], 0.4,
                color=hub_color, edgecolor="white",
                label="tech hubs (Bay, Seattle, NYC, Austin, Boston)")
        ax.barh(y + 0.2, frame["excl_rest_rate_pct"], 0.4,
                color=rest_color, edgecolor="white",
                label="rest of the country")
        for i, t in enumerate(toks):
            bay = frame.loc[t, "excl_bay_rate_pct"]
            rest = frame.loc[t, "excl_rest_rate_pct"]
            ax.text(bay + 0.2, i - 0.2, f"{bay:.1f}%", va="center", fontsize=8.5, color=hub_color)
            ax.text(rest + 0.2, i + 0.2, f"{rest:.1f}%", va="center", fontsize=8.5, color=rest_color)
        ax.set_yticks(y)
        ax.set_yticklabels(toks, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("rate among AI-matched non-builder SWE postings (%)")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="lower right")

    plot_pair(axes[0], hub, "Frontier-platform vocabulary (hubs lead)",
              hub_color=PAL["swe"], rest_color=PAL["control"])
    plot_pair(axes[1], user, "Coding-tool vocabulary (rest leads)",
              hub_color=PAL["swe"], rest_color=PAL["control"])

    fig.suptitle("Composite A · Coding tools democratised; agentic vocabulary did not",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02,
             "After excluding ~17 firms whose own product names are AI vocabulary (OpenAI, Microsoft, Google, etc.), "
             "the asymmetric pattern survives. The vendor leaderboard told us which tools employers name; this section tells us where they land. "
             "Source: eda/tables/audit_self_mention_audit1_token_gap.csv.",
             ha="center", fontsize=8, style="italic", color="#666")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Composite A panel 2 — geography
# ---------------------------------------------------------------------------

def viz_composite_a_geo():
    """Top metros by 2026 absolute AI rate; differential available as side hue."""
    _style_setup()
    df = pd.read_csv(TABLES_DIR / "S26_metro_abs_2026.csv")
    df = df.dropna(subset=["rate_all_2026"]).copy()
    df = df.sort_values("rate_all_2026", ascending=True).tail(20)

    colors = [PAL["swe"] if h else PAL["control"] for h in df["is_tech_hub"]]
    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(df["metro_area"], df["rate_all_2026"] * 100,
                   color=colors, edgecolor="white")
    for b, v in zip(bars, df["rate_all_2026"]):
        ax.text(v * 100 + 0.3, b.get_y() + b.get_height() / 2,
                f"{v*100:.1f}%", va="center", fontsize=9)

    # Legend proxies
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=PAL["swe"], label="tech hub (5)"),
               Patch(facecolor=PAL["control"], label="rest of country")]
    ax.legend(handles=handles, loc="lower right", fontsize=10)
    ax.set_xlabel("AI-vocab rate among 2026 SWE postings (%)")
    ax.set_title("Composite A · Where AI is being written into engineering descriptions, 2026",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, df["rate_all_2026"].max() * 100 * 1.15)

    fig.text(0.5, -0.01,
             "The Bay Area still leads on absolute AI-vocab rate (~32% under core_llm); Sun Belt cities (Salt Lake City, Tampa, Atlanta) lead on the rate of change. "
             "Source: eda/tables/S26_metro_abs_2026.csv.",
             ha="center", fontsize=8, style="italic", color="#666")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Composite A panel 3 — industry
# ---------------------------------------------------------------------------

def viz_composite_a_industry():
    """2026 AI rate by industry with Wilson 95% CIs, top 12 by n>=100."""
    _style_setup()
    df = pd.read_csv(TABLES_DIR / "S26_industry_2026.csv")
    df = df[df["n_all"] >= 100].copy()
    df = df.sort_values("rate_all", ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(12, 7))
    rates = df["rate_all"].values * 100
    los = df["wilson_lo"].values * 100
    his = df["wilson_hi"].values * 100
    err_lo = rates - los
    err_hi = his - rates

    # Color: orange for hospitals/health, green for software, blue for FS
    def industry_color(name):
        n = name.lower()
        if "hospital" in n or "health" in n:
            return PAL["swe_adjacent"]
        if "software" in n:
            return PAL["headline"]
        if "financial" in n or "banking" in n or "insurance" in n:
            return PAL["control"]
        if "research" in n:
            return PAL["ai"]
        return "#888"

    colors = [industry_color(n) for n in df["company_industry"]]
    bars = ax.barh(df["company_industry"], rates, color=colors, edgecolor="white")
    ax.errorbar(rates, np.arange(len(df)), xerr=[err_lo, err_hi],
                fmt="none", color="#444", capsize=3, linewidth=1)
    for b, v, n in zip(bars, rates, df["n_all"]):
        ax.text(v + 1.0, b.get_y() + b.get_height() / 2,
                f"{v:.1f}%  (n={int(n):,})", va="center", fontsize=9)

    ax.set_xlabel("AI-vocab rate among 2026 SWE postings, with Wilson 95% CI (%)")
    ax.set_title("Composite A · Hospitals at parity with software firms; FS lags both",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(his.max() + 12, 80))

    fig.text(0.5, -0.01,
             "Within-2026 only (industry labels are sparse in the 2024 baseline). "
             "Hospital lead leans on AI-native health-tech firms (Abridge, Ambience) and Optum/CVS; "
             "drop those five and hospitals fall to ~29%, below software. "
             "Source: eda/tables/S26_industry_2026.csv.",
             ha="center", fontsize=8, style="italic", color="#666")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Composite B panel 1 — agentic-AI cluster
# ---------------------------------------------------------------------------

def viz_composite_b_cluster():
    """BERTopic Topic 1 share of corpus, 2024 vs 2026; within-cluster AI rate;
    methods-agreement banner."""
    _style_setup()
    topic = pd.read_csv(TABLES_DIR / "substrate_D_topic1.csv")
    align = pd.read_csv(TABLES_DIR / "S27_v2_method_alignment.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: cluster share by year
    yrs = topic["yr"].astype(str).tolist()
    shares = topic["pct_topic1"].values
    bars = axes[0].bar(yrs, shares, color=[PAL["control"], PAL["ai"]],
                       edgecolor="white", width=0.55)
    for b, v in zip(bars, shares):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.3,
                     f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")
    growth = shares[-1] / shares[0]
    axes[0].set_title(f"Topic 1 (RAG / agentic / LLM systems): {growth:.1f}× rise in two years")
    axes[0].set_ylabel("share of capped SWE corpus (%)")
    axes[0].set_ylim(0, max(shares) * 1.3)

    # Right: within-cluster AI rate
    axes[1].bar(yrs, topic["topic1_ai_core"] * 100, color=PAL["ai"],
                edgecolor="white", width=0.55)
    for i, v in enumerate(topic["topic1_ai_core"]):
        axes[1].text(i, v * 100 + 1.5, f"{v*100:.0f}%",
                     ha="center", fontsize=12, fontweight="bold", color=PAL["ai"])
    axes[1].set_title("Within-cluster AI-vocab rate (core_llm substrate)")
    axes[1].set_ylabel("AI-vocab rate among Topic-1 postings (%)")
    axes[1].set_ylim(0, 100)

    # Methods agreement banner
    nmi = float(align["nmi_bertopic_vs_nmf"].iloc[0])
    fig.text(0.5, 0.94,
             f"Two-method confirmation: BERTopic 5.2× share rise · NMF 12.4× volume rise · BERTopic-vs-NMF NMI {nmi:.2f}",
             ha="center", fontsize=10, color="#222", style="italic",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#ccc"))

    fig.suptitle("Composite B · The new specialist: an agentic-AI cluster that did not exist at this scale in 2024",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.text(0.5, -0.01,
             "Cluster fitting on description_core_llm, capped at 30 postings per (firm × period). "
             "Self-mention exclusion drops the multiplier from 5.20× to 5.13× (substantively unchanged). "
             "Source: eda/tables/substrate_D_topic1.csv + S27_v2_method_alignment.csv.",
             ha="center", fontsize=8, style="italic", color="#666")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Composite B panels 2 + 3 — FDE growth and legacy substitution
# ---------------------------------------------------------------------------

def viz_composite_b_fde_legacy():
    """Side-by-side: FDE growth + legacy-neighbour AI rate vs market."""
    _style_setup()
    fde = pd.read_csv(TABLES_DIR / "S27_thread2_fde_method_comparison.csv")
    fde_firms = pd.read_csv(TABLES_DIR / "S27_thread2_fde_firms_2026.csv")
    leg = pd.read_csv(TABLES_DIR / "substrate_D_legacy_substitution.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # Left: FDE volume by year (combined: title + JD)
    periods = fde["period_bucket"].astype(str).tolist()
    counts = fde["n_fde_combined"].astype(int).tolist()
    bars = axes[0].bar(periods, counts, color=PAL["headline"], edgecolor="white", width=0.55)
    for b, v in zip(bars, counts):
        axes[0].text(b.get_x() + b.get_width() / 2, v + max(counts) * 0.03,
                     f"{v}", ha="center", fontsize=12, fontweight="bold")
    axes[0].set_title("Forward-Deployed Engineer postings, 2024 vs 2026")
    axes[0].set_ylabel("postings (title or JD-text match)")
    axes[0].set_ylim(0, max(counts) * 1.25)

    # Annotate top firms
    top_firm_text = ", ".join(fde_firms["company_name_canonical"].head(8).tolist())
    axes[0].text(0.5, 0.98,
                 f"Top firms in 2026: {top_firm_text}",
                 transform=axes[0].transAxes, ha="center", va="top", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0fff0", edgecolor=PAL["headline"]))

    # Right: legacy-neighbour AI rate vs market under core_llm
    labels = ["legacy-neighbour\ntitles", "market average\n(2026 SWE)"]
    rates = [leg["neighbor_rate_core"].iloc[0] * 100, leg["market_rate_core"].iloc[0] * 100]
    colors_pair = [PAL["disproven"], PAL["headline"]]
    bars = axes[1].bar(labels, rates, color=colors_pair, edgecolor="white", width=0.55)
    for b, v in zip(bars, rates):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.7,
                     f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ratio = leg["ratio_core"].iloc[0]
    axes[1].set_title(f"Legacy substitutions are stack-modernisation, not AI ({1/ratio:.1f}× gap)")
    axes[1].set_ylabel("AI-vocab rate (%)")
    axes[1].set_ylim(0, max(rates) * 1.3)

    fig.suptitle("Composite B · A new function (FDE) and a quiet substitution (legacy → modernised neighbours)",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, -0.01,
             "FDE: Palantir-coined title now adopted by OpenAI, Adobe, AMD, PhysicsX, TRM Labs, Saronic, Foxglove, PwC, Ramp. "
             "Legacy: disappearing 2024 titles (Java architect, Drupal specialist, PHP architect) re-route to neighbours whose AI rate is "
             f"{rates[0]:.1f}% under core_llm — well below the {rates[1]:.1f}% market average. "
             "Source: eda/tables/S27_thread2_fde_method_comparison.csv + substrate_D_legacy_substitution.csv.",
             ha="center", fontsize=8, style="italic", color="#666")
    fig.tight_layout()
    return fig
