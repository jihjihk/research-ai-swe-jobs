"""
S26 — Composite A: Where AI is signaled most in software-engineering job requirements.

Three threads explored with multiple methodologies each:

1. Geographic distribution of AI requirements by metro × period
   - Sensitivity: aggregator exclusion, multi-location postings, volume cutoff,
     weighted vs unweighted metro averages, differential vs absolute rates
2. Industry distribution of AI requirements (within-2026 only — 2024 industry NULL)
   - Sensitivity: labeled-only, tech-firm classification, aggregator exclusion
3. Builder-vs-user role split
   - Method (a) v9 T28 archetype labels (cluster 1 + 25 = builder; AI-mentioning rest = user)
   - Method (b) Title regex proxy (Applied AI / ML / LLM / FDE titles = builder)

All AI-vocab uses canonical regex from eda/scripts/scans.py.

Run:
  ./.venv/bin/python eda/scripts/S26_composite_a.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Hook into existing AI vocab
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scans import AI_VOCAB_PATTERN, NEW_AI_TITLE_PATTERN, BIG_TECH_CANONICAL, text_col, text_filter  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE = PROJECT_ROOT / "data" / "unified_core.parquet"
T28_ARCHETYPE = (
    PROJECT_ROOT
    / "exploration-archive/v9_final_opus_47/tables/T28/T28_corpus_with_archetype.parquet"
)
TABLES = PROJECT_ROOT / "eda" / "tables"
FIGURES = PROJECT_ROOT / "eda" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

CORE_FILTER = "is_english = TRUE AND date_flag = 'ok' AND is_swe = TRUE"

# T17 26-metro panel: any metro with >=50 SWE postings in BOTH 2024 and 2026.
TECH_HUBS = {
    "San Francisco Bay Area",
    "Seattle Metro",
    "New York City Metro",
    "Austin Metro",
    "Boston Metro",
}

# Builder-title regex: more conservative than NEW_AI_TITLE_PATTERN; titles
# that strongly imply the worker BUILDS AI systems (not just uses them).
BUILDER_TITLE_PHRASES = [
    "applied ai",
    "applied ml",
    "applied scientist",
    "ai engineer",
    "ml engineer",
    "machine learning engineer",
    "llm engineer",
    "genai engineer",
    "gen ai engineer",
    "agent engineer",
    "ai/ml engineer",
    "forward deployed",
    "forward-deployed",
    "founding ai",
    "ai research",
    "ml research",
    "research engineer",  # often AI-flavored at frontier labs
    "mlops engineer",
    "ai architect",
    "machine learning scientist",
]
BUILDER_TITLE_PATTERN = (
    r"(?i)\b(" + "|".join(p.replace("/", r"\/") for p in BUILDER_TITLE_PHRASES) + r")\b"
)


# ---------------------------------------------------------------------------
# Thread 1 — Geography
# ---------------------------------------------------------------------------

def thread1_metro(con):
    """Metro × period AI rate, with sensitivities."""
    print("== Thread 1: Geography ==")

    # Base scan: per-metro AI rate by period_year (2024 vs 2026).
    base = con.execute(f"""
      WITH base AS (
        SELECT metro_area,
               CASE WHEN period LIKE '2024%' THEN '2024'
                    WHEN period LIKE '2026%' THEN '2026' END AS year,
               is_aggregator,
               is_multi_location,
               regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}') AS ai
        FROM '{CORE}'
        WHERE {CORE_FILTER}
          AND metro_area IS NOT NULL
          AND {text_filter()}
      )
      SELECT metro_area, year,
             COUNT(*)                                            AS n_all,
             SUM(ai::INTEGER)                                    AS n_ai_all,
             SUM(CASE WHEN NOT is_aggregator THEN 1 ELSE 0 END)  AS n_noagg,
             SUM(CASE WHEN NOT is_aggregator AND ai THEN 1 ELSE 0 END) AS n_ai_noagg,
             SUM(CASE WHEN NOT is_multi_location THEN 1 ELSE 0 END) AS n_single,
             SUM(CASE WHEN NOT is_multi_location AND ai THEN 1 ELSE 0 END) AS n_ai_single
      FROM base
      GROUP BY 1,2
    """).df()
    base["rate_all"] = base["n_ai_all"] / base["n_all"]
    base["rate_noagg"] = base["n_ai_noagg"] / base["n_noagg"]
    base["rate_single"] = base["n_ai_single"] / base["n_single"]

    # Pivot to wide (metro × {2024, 2026}) and compute deltas.
    wide = base.pivot(index="metro_area", columns="year")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    # Apply T17 inclusion threshold: ≥50 SWE postings in BOTH years.
    panel = wide[(wide["n_all_2024"] >= 50) & (wide["n_all_2026"] >= 50)].copy()
    panel["delta_all"] = panel["rate_all_2026"] - panel["rate_all_2024"]
    panel["delta_noagg"] = panel["rate_noagg_2026"] - panel["rate_noagg_2024"]
    panel["delta_single"] = panel["rate_single_2026"] - panel["rate_single_2024"]
    panel["is_tech_hub"] = panel["metro_area"].isin(TECH_HUBS)
    panel = panel.sort_values("delta_all", ascending=False)
    panel.to_csv(TABLES / "S26_metro_panel.csv", index=False)
    print(f"  metro panel: {len(panel)} metros pass ≥50/50 cut")

    # Headline: tech-hub premium under three filters.
    hub_premium_rows = []
    for col, label in [
        ("delta_all", "all postings"),
        ("delta_noagg", "no aggregators"),
        ("delta_single", "single-location only"),
    ]:
        hub = panel.loc[panel["is_tech_hub"], col].mean()
        rest = panel.loc[~panel["is_tech_hub"], col].mean()
        hub_premium_rows.append({
            "filter": label,
            "n_hub": int(panel["is_tech_hub"].sum()),
            "n_rest": int((~panel["is_tech_hub"]).sum()),
            "hub_mean_delta": hub,
            "rest_mean_delta": rest,
            "premium_pp": (hub - rest) * 100,
        })
    pd.DataFrame(hub_premium_rows).to_csv(TABLES / "S26_metro_hub_premium.csv", index=False)

    # Volume-cutoff sensitivity: try ≥30, ≥50, ≥100, ≥200.
    cutoff_rows = []
    for cutoff in [30, 50, 100, 200]:
        sub = wide[(wide["n_all_2024"] >= cutoff) & (wide["n_all_2026"] >= cutoff)].copy()
        sub["delta"] = sub["rate_all_2026"] - sub["rate_all_2024"]
        sub["is_hub"] = sub["metro_area"].isin(TECH_HUBS)
        if not sub["is_hub"].any() or not (~sub["is_hub"]).any():
            continue
        # Unweighted means.
        cutoff_rows.append({
            "cutoff": cutoff,
            "n_metros": len(sub),
            "n_hubs": int(sub["is_hub"].sum()),
            "delta_min_pp": sub["delta"].min() * 100,
            "delta_max_pp": sub["delta"].max() * 100,
            "delta_unweighted_mean_pp": sub["delta"].mean() * 100,
            "delta_volume_weighted_pp": (
                np.average(sub["delta"], weights=sub["n_all_2024"] + sub["n_all_2026"])
            ) * 100,
            "hub_mean_delta_pp": sub.loc[sub["is_hub"], "delta"].mean() * 100,
            "rest_mean_delta_pp": sub.loc[~sub["is_hub"], "delta"].mean() * 100,
            "premium_pp": (
                sub.loc[sub["is_hub"], "delta"].mean()
                - sub.loc[~sub["is_hub"], "delta"].mean()
            ) * 100,
        })
    pd.DataFrame(cutoff_rows).to_csv(TABLES / "S26_metro_cutoff_sensitivity.csv", index=False)

    # ABSOLUTE 2026 rate by metro (for "where is AI most signaled today").
    panel_abs = panel[["metro_area", "is_tech_hub", "n_all_2026", "rate_all_2026", "rate_noagg_2026"]].copy()
    panel_abs = panel_abs.sort_values("rate_all_2026", ascending=False)
    panel_abs.to_csv(TABLES / "S26_metro_abs_2026.csv", index=False)

    # Plot: metro deltas, ranked.
    fig, ax = plt.subplots(figsize=(10, 8))
    panel_plot = panel.sort_values("delta_all")
    colors = ["#d62728" if h else "#1f77b4" for h in panel_plot["is_tech_hub"]]
    ax.barh(panel_plot["metro_area"], panel_plot["delta_all"] * 100, color=colors)
    ax.set_xlabel("Δ AI-vocab rate, 2024 → 2026 (pp)")
    ax.set_title("S26.1 — Per-metro AI requirement growth, 2024→2026 (LinkedIn SWE)")
    ax.text(
        0.99, 0.02,
        "Red = tech-hub (SF, Seattle, NYC, Austin, Boston). "
        f"n=26 metros. Tech-hub premium = "
        f"{(panel.loc[panel['is_tech_hub'], 'delta_all'].mean() - panel.loc[~panel['is_tech_hub'], 'delta_all'].mean()) * 100:+.2f} pp.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, alpha=0.7
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "S26_metro_deltas.png", dpi=150)
    plt.close(fig)

    return panel, hub_premium_rows


# ---------------------------------------------------------------------------
# Thread 2 — Industry (within 2026 only — 2024 has no industry labels in core)
# ---------------------------------------------------------------------------

def thread2_industry(con):
    """Per-industry AI rate within 2026, with sensitivities."""
    print("== Thread 2: Industry (2026 only) ==")

    base = con.execute(f"""
      WITH ind AS (
        SELECT company_industry,
               is_aggregator,
               LOWER(company_name_canonical) IN ({", ".join("'" + b + "'" for b in BIG_TECH_CANONICAL)}) AS is_big_tech,
               regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}') AS ai
        FROM '{CORE}'
        WHERE {CORE_FILTER}
          AND period LIKE '2026%'
          AND company_industry IS NOT NULL
          AND {text_filter()}
      )
      SELECT company_industry,
             COUNT(*)                                          AS n_all,
             SUM(ai::INTEGER)                                  AS n_ai_all,
             SUM(CASE WHEN NOT is_aggregator THEN 1 ELSE 0 END)        AS n_noagg,
             SUM(CASE WHEN NOT is_aggregator AND ai THEN 1 ELSE 0 END) AS n_ai_noagg,
             SUM(CASE WHEN NOT is_big_tech THEN 1 ELSE 0 END)          AS n_nobig,
             SUM(CASE WHEN NOT is_big_tech AND ai THEN 1 ELSE 0 END)   AS n_ai_nobig
      FROM ind
      GROUP BY 1
    """).df()
    base["rate_all"] = base["n_ai_all"] / base["n_all"]
    base["rate_noagg"] = base["n_ai_noagg"] / base["n_noagg"]
    base["rate_nobig"] = base["n_ai_nobig"] / base["n_nobig"].replace(0, np.nan)

    # Wilson 95% CI on rate_all.
    z = 1.96
    n = base["n_all"]; p = base["rate_all"]
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    base["wilson_lo"] = (centre - half) / denom
    base["wilson_hi"] = (centre + half) / denom

    industries_n10 = base[base["n_all"] >= 100].sort_values("rate_all", ascending=False)
    industries_n10.to_csv(TABLES / "S26_industry_2026.csv", index=False)

    # Overall pooled rate.
    pooled = con.execute(f"""
      SELECT COUNT(*) AS n,
             SUM(CASE WHEN regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}') THEN 1 ELSE 0 END) AS n_ai
      FROM '{CORE}'
      WHERE {CORE_FILTER} AND period LIKE '2026%' AND company_industry IS NOT NULL AND {text_filter()}
    """).df().iloc[0]
    pooled_rate = pooled["n_ai"] / pooled["n"]
    print(f"  pooled 2026 labeled-industry AI rate: {pooled_rate*100:.2f}% (n={int(pooled['n']):,})")

    # Compare to UNlabeled rows.
    unlab = con.execute(f"""
      SELECT COUNT(*) AS n,
             SUM(CASE WHEN regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}') THEN 1 ELSE 0 END) AS n_ai
      FROM '{CORE}'
      WHERE {CORE_FILTER} AND period LIKE '2026%' AND company_industry IS NULL AND {text_filter()}
    """).df().iloc[0]
    unlab_rate = unlab["n_ai"] / unlab["n"] if unlab["n"] else float("nan")
    print(f"  pooled 2026 UNlabeled-industry AI rate: {unlab_rate*100:.2f}% (n={int(unlab['n']):,})")

    coverage_row = pd.DataFrame([{
        "labeled_n": int(pooled["n"]),
        "labeled_rate": pooled_rate,
        "unlabeled_n": int(unlab["n"]),
        "unlabeled_rate": unlab_rate,
    }])
    coverage_row.to_csv(TABLES / "S26_industry_coverage.csv", index=False)

    # Plot top industries.
    top = industries_n10.head(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(top))
    ax.barh(y, top["rate_all"] * 100, color="#1f77b4")
    ax.errorbar(
        top["rate_all"] * 100, y,
        xerr=[(top["rate_all"] - top["wilson_lo"]) * 100,
              (top["wilson_hi"] - top["rate_all"]) * 100],
        fmt="none", ecolor="black", capsize=2, alpha=0.6
    )
    ax.set_yticks(y); ax.set_yticklabels(top["company_industry"])
    ax.invert_yaxis()
    ax.axvline(pooled_rate * 100, linestyle="--", color="grey", label=f"Pooled = {pooled_rate*100:.1f}%")
    for i, r in enumerate(top.itertuples()):
        ax.text(r.rate_all * 100 + 0.3, i, f"{r.rate_all*100:.1f}% (n={int(r.n_all):,})",
                va="center", fontsize=8)
    ax.set_xlabel("AI-vocab rate, 2026 (LinkedIn SWE)")
    ax.set_title("S26.2 — AI-requirement rate by industry, 2026 (within-period only)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "S26_industry_2026.png", dpi=150)
    plt.close(fig)

    return industries_n10, pooled_rate


# ---------------------------------------------------------------------------
# Thread 3 — Builder vs user
# ---------------------------------------------------------------------------

def thread3_builder_user(con):
    """Two parallel methods for builder-vs-user split."""
    print("== Thread 3: Builder vs user ==")

    # Method (a) — v9 T28 archetype: cluster 1 (models/systems/llm) and
    # cluster 25 (systems/agent/workflows) tagged as builder.
    BUILDER_CLUSTERS = (1, 25)
    method_a = con.execute(f"""
      WITH joined AS (
        SELECT c.uid, c.metro_area, c.period, c.title,
               c.is_aggregator,
               regexp_matches(c.description, '{AI_VOCAB_PATTERN}') AS ai,
               t.archetype_primary,
               t.archetype_primary_name,
               CASE WHEN t.archetype_primary IN {BUILDER_CLUSTERS} THEN 'builder'
                    WHEN t.archetype_primary IS NULL THEN 'unlabeled'
                    ELSE 'general' END AS role_class
        FROM '{CORE}' c
        LEFT JOIN '{T28_ARCHETYPE}' t USING(uid)
        WHERE {CORE_FILTER} AND c.metro_area IS NOT NULL
      )
      SELECT period, role_class,
             COUNT(*) AS n,
             SUM(ai::INTEGER) AS n_ai,
             COUNT(DISTINCT metro_area) AS n_metros
      FROM joined
      GROUP BY 1,2 ORDER BY 1,2
    """).df()
    method_a["ai_rate"] = method_a["n_ai"] / method_a["n"]
    method_a.to_csv(TABLES / "S26_builder_user_archetype.csv", index=False)

    # Method (a) geography: builder (cluster 1+25) share by metro × year.
    method_a_metro = con.execute(f"""
      WITH joined AS (
        SELECT c.uid, c.metro_area,
               CASE WHEN c.period LIKE '2024%' THEN '2024' WHEN c.period LIKE '2026%' THEN '2026' END AS year,
               regexp_matches(c.description, '{AI_VOCAB_PATTERN}') AS ai,
               t.archetype_primary
        FROM '{CORE}' c
        LEFT JOIN '{T28_ARCHETYPE}' t USING(uid)
        WHERE {CORE_FILTER} AND c.metro_area IS NOT NULL
      )
      SELECT metro_area, year,
             COUNT(*) AS n,
             SUM(CASE WHEN archetype_primary IN {BUILDER_CLUSTERS} THEN 1 ELSE 0 END) AS n_builder,
             SUM(CASE WHEN archetype_primary IS NOT NULL AND archetype_primary NOT IN {BUILDER_CLUSTERS} AND ai THEN 1 ELSE 0 END) AS n_user_ai,
             SUM(CASE WHEN archetype_primary IS NOT NULL AND archetype_primary NOT IN {BUILDER_CLUSTERS} THEN 1 ELSE 0 END) AS n_general_labeled
      FROM joined
      GROUP BY 1,2
    """).df()
    method_a_metro["builder_share"] = method_a_metro["n_builder"] / method_a_metro["n"]
    method_a_metro["user_share_in_general"] = method_a_metro["n_user_ai"] / method_a_metro["n_general_labeled"]
    panel_a = method_a_metro.pivot(index="metro_area", columns="year")
    panel_a.columns = [f"{a}_{b}" for a, b in panel_a.columns]
    panel_a = panel_a.reset_index()
    if "n_2024" in panel_a.columns and "n_2026" in panel_a.columns:
        panel_a = panel_a[(panel_a["n_2024"] >= 50) & (panel_a["n_2026"] >= 50)]
    panel_a["builder_share_2026"] = panel_a.get("builder_share_2026")
    panel_a["user_share_2026"] = panel_a.get("user_share_in_general_2026")
    panel_a["builder_share_delta"] = panel_a.get("builder_share_2026", 0) - panel_a.get("builder_share_2024", 0)
    panel_a["user_share_delta"] = panel_a.get("user_share_in_general_2026", 0) - panel_a.get("user_share_in_general_2024", 0)
    panel_a["is_tech_hub"] = panel_a["metro_area"].isin(TECH_HUBS)
    panel_a.to_csv(TABLES / "S26_builder_user_metro_archetype.csv", index=False)

    # Method (b) — title regex.
    method_b = con.execute(f"""
      WITH base AS (
        SELECT title, metro_area,
               CASE WHEN period LIKE '2024%' THEN '2024' WHEN period LIKE '2026%' THEN '2026' END AS year,
               regexp_matches(LOWER(title), '{BUILDER_TITLE_PATTERN}') AS is_builder_title,
               regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}') AS ai
        FROM '{CORE}'
        WHERE {CORE_FILTER} AND metro_area IS NOT NULL AND {text_filter()}
      )
      SELECT metro_area, year,
             COUNT(*) AS n,
             SUM(CASE WHEN is_builder_title THEN 1 ELSE 0 END) AS n_builder,
             SUM(CASE WHEN NOT is_builder_title AND ai THEN 1 ELSE 0 END) AS n_user,
             SUM(CASE WHEN ai THEN 1 ELSE 0 END) AS n_ai_total
      FROM base
      GROUP BY 1,2
    """).df()
    method_b["builder_share"] = method_b["n_builder"] / method_b["n"]
    method_b["user_share_of_nonbuilder"] = method_b["n_user"] / (method_b["n"] - method_b["n_builder"])

    panel_b = method_b.pivot(index="metro_area", columns="year")
    panel_b.columns = [f"{a}_{b}" for a, b in panel_b.columns]
    panel_b = panel_b.reset_index()
    if "n_2024" in panel_b.columns and "n_2026" in panel_b.columns:
        panel_b = panel_b[(panel_b["n_2024"] >= 50) & (panel_b["n_2026"] >= 50)]
    panel_b["builder_share_delta"] = panel_b.get("builder_share_2026", 0) - panel_b.get("builder_share_2024", 0)
    panel_b["user_share_delta"] = panel_b.get("user_share_of_nonbuilder_2026", 0) - panel_b.get("user_share_of_nonbuilder_2024", 0)
    panel_b["is_tech_hub"] = panel_b["metro_area"].isin(TECH_HUBS)
    panel_b.to_csv(TABLES / "S26_builder_user_metro_title.csv", index=False)

    # Cross-method comparison: hub-vs-rest builder share, 2026.
    rows = []
    for label, panel in [("archetype", panel_a), ("title_regex", panel_b)]:
        if "builder_share_2026" not in panel.columns:
            continue
        hub_b = panel.loc[panel["is_tech_hub"], "builder_share_2026"].mean()
        rest_b = panel.loc[~panel["is_tech_hub"], "builder_share_2026"].mean()
        # The "user_share_2026" column means different things across methods;
        # in (a) it's AI-vocab rate within non-builder labeled rows; in (b) it's
        # AI-vocab rate within non-builder titles. Different denominators.
        col_user = "user_share_2026" if "user_share_2026" in panel.columns else "user_share_of_nonbuilder_2026"
        hub_u = panel.loc[panel["is_tech_hub"], col_user].mean()
        rest_u = panel.loc[~panel["is_tech_hub"], col_user].mean()
        rows.append({
            "method": label,
            "hub_builder_share_2026": hub_b,
            "rest_builder_share_2026": rest_b,
            "builder_hub_premium_pp": (hub_b - rest_b) * 100,
            "hub_user_share_2026": hub_u,
            "rest_user_share_2026": rest_u,
            "user_hub_premium_pp": (hub_u - rest_u) * 100,
        })
    cross = pd.DataFrame(rows)
    cross.to_csv(TABLES / "S26_builder_user_cross_method.csv", index=False)
    print(cross.to_string())

    # Plot: builder share by metro, 2026, both methods.
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharex=False)
    for ax, panel, label in [
        (axes[0], panel_a, "Method (a) v9 T28 archetype"),
        (axes[1], panel_b, "Method (b) title regex"),
    ]:
        if "builder_share_2026" not in panel.columns:
            continue
        sub = panel.sort_values("builder_share_2026")
        colors = ["#d62728" if h else "#1f77b4" for h in sub["is_tech_hub"]]
        ax.barh(sub["metro_area"], sub["builder_share_2026"] * 100, color=colors)
        ax.set_xlabel("Builder share of SWE postings (%), 2026")
        ax.set_title(label)
    fig.suptitle("S26.3 — Builder share by metro, 2026 (red = tech hub)")
    fig.tight_layout()
    fig.savefig(FIGURES / "S26_builder_share_metro.png", dpi=150)
    plt.close(fig)

    # Plot: user-share (AI-mention rate among non-builders) by metro, 2026.
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    for ax, panel, col, label in [
        (axes[0], panel_a, "user_share_2026", "Method (a) AI-mention rate among non-builder labeled"),
        (axes[1], panel_b, "user_share_of_nonbuilder_2026", "Method (b) AI-mention rate among non-builder titles"),
    ]:
        if col not in panel.columns:
            continue
        sub = panel.sort_values(col)
        colors = ["#d62728" if h else "#1f77b4" for h in sub["is_tech_hub"]]
        ax.barh(sub["metro_area"], sub[col] * 100, color=colors)
        ax.set_xlabel("AI-mention rate (%), non-builder rows, 2026")
        ax.set_title(label)
    fig.suptitle("S26.4 — User share (AI mention by general SWE) by metro, 2026")
    fig.tight_layout()
    fig.savefig(FIGURES / "S26_user_share_metro.png", dpi=150)
    plt.close(fig)

    return panel_a, panel_b, cross


# ---------------------------------------------------------------------------
# Composite 3-panel figure
# ---------------------------------------------------------------------------

def composite_panel(panel_metro, industries, panel_b):
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # Panel 1: top/bottom 5 metros by Δ.
    p = panel_metro.copy()
    p["color"] = ["#d62728" if h else "#1f77b4" for h in p["is_tech_hub"]]
    p = p.sort_values("delta_all")
    axes[0].barh(p["metro_area"], p["delta_all"] * 100, color=p["color"])
    axes[0].set_xlabel("Δ AI-vocab rate, 2024→2026 (pp)")
    axes[0].set_title("Where AI-requirement language is spreading")
    axes[0].axvline(0, color="black", linewidth=0.5)

    # Panel 2: top 12 industries 2026.
    top = industries.head(12).sort_values("rate_all")
    axes[1].barh(top["company_industry"], top["rate_all"] * 100, color="#2ca02c")
    axes[1].set_xlabel("AI-vocab rate, 2026 (%)")
    axes[1].set_title("Which industries write AI-dense JDs")

    # Panel 3: builder share metro, method (b).
    if "builder_share_2026" in panel_b.columns:
        sub = panel_b.sort_values("builder_share_2026")
        colors = ["#d62728" if h else "#1f77b4" for h in sub["is_tech_hub"]]
        axes[2].barh(sub["metro_area"], sub["builder_share_2026"] * 100, color=colors)
        axes[2].set_xlabel("Builder-title share (%), 2026")
        axes[2].set_title("Where AI-builder roles concentrate")

    fig.suptitle("S26 — Composite A: where AI is signaled most in SWE postings")
    fig.tight_layout()
    fig.savefig(FIGURES / "S26_composite_3panel.png", dpi=150)
    plt.close(fig)


def main():
    con = duckdb.connect()
    panel_metro, hub_premium = thread1_metro(con)
    industries, pooled_rate = thread2_industry(con)
    panel_a, panel_b, cross = thread3_builder_user(con)
    composite_panel(panel_metro, industries, panel_b)
    print("Done.")


if __name__ == "__main__":
    main()
