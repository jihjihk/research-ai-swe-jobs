"""
Substrate sensitivity audit: recompute every existing AI-vocab headline
under `description_core_llm` (LLM-stripped of boilerplate) and compare
against the headline number computed from raw `description`.

For each claim:
  1. Reproduce headline under raw description (sanity check).
  2. Recompute under description_core_llm.
  3. Verdict: SURVIVES / WEAKENS / STRENGTHENS / FLIPS.

All AI-vocab matches use the canonical AI_VOCAB_PATTERN from scans.py
(line 50-71). DO NOT redefine.

Outputs all CSVs to eda/tables/substrate_*.csv

Run:
  ./.venv/bin/python eda/scripts/substrate_sensitivity.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "eda" / "scripts"))
from scans import AI_VOCAB_PATTERN, BIG_TECH_CANONICAL, TECH_INDUSTRIES  # noqa: E402
from S26_composite_a import BUILDER_TITLE_PATTERN  # noqa: E402

CORE = PROJECT_ROOT / "data" / "unified_core.parquet"
UNIFIED = PROJECT_ROOT / "data" / "unified.parquet"
ARCH = PROJECT_ROOT / "eda" / "artifacts" / "composite_B_archetype_labels.parquet"
T28_ARCHETYPE = (
    PROJECT_ROOT
    / "exploration-archive/v9_final_opus_47/tables/T28/T28_corpus_with_archetype.parquet"
)
TABLES = PROJECT_ROOT / "eda" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

CORE_FILTER = "source_platform='linkedin' AND is_english=true AND date_flag='ok'"

# Substrate: build a substrate-aware match expression.
# raw  = AI matches on `description`
# core = AI matches on `description_core_llm` (boilerplate stripped)
RAW_MATCH = f"regexp_matches(COALESCE(description, ''), '{AI_VOCAB_PATTERN}')"
CORE_MATCH = f"regexp_matches(COALESCE(description_core_llm, ''), '{AI_VOCAB_PATTERN}')"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def verdict(raw: float, core: float, *, sig_tol_pp: float = 0.5) -> str:
    """Translate raw/core deltas into a verdict.

    sig_tol_pp: smaller absolute moves than this in percentage points are
    treated as 'unchanged' for verdict purposes (only relevant when both
    rates are sub-percent).
    """
    if pd.isna(raw) or pd.isna(core):
        return "UNDEFINED"
    delta_pp = (core - raw) * 100
    rel = (core - raw) / raw if raw else np.nan
    abs_delta_pp = abs(delta_pp)
    # FLIPS if sign changes (extremely rare for a single rate).
    # WEAKENS if core < raw by >2% relative AND > sig_tol_pp.
    # STRENGTHENS if core > raw by >2% relative AND > sig_tol_pp.
    # SURVIVES otherwise.
    if abs_delta_pp < sig_tol_pp and abs(rel) < 0.05:
        return "SURVIVES"
    if delta_pp < 0 and abs(rel) > 0.40:
        return "WEAKENS-STRONG"
    if delta_pp < 0:
        return "WEAKENS"
    if delta_pp > 0 and rel > 0.10:
        return "STRENGTHENS"
    return "SURVIVES"


def fmt_rate(x):
    return "" if pd.isna(x) else f"{x*100:.2f}%"


def fmt_pp(x):
    return "" if pd.isna(x) else f"{x*100:+.2f}pp"


# ---------------------------------------------------------------------------
# A1 — H1 SWE-vs-control 23:1
# ---------------------------------------------------------------------------

def a1_swe_vs_control(con):
    print("\n===== A1: H1 SWE-vs-control 23:1 ratio =====")
    sql = f"""
      SELECT period,
             CASE WHEN is_swe THEN 'swe'
                  WHEN is_swe_adjacent THEN 'swe_adjacent'
                  WHEN is_control THEN 'control' END AS analysis_group,
             COUNT(*) AS n,
             SUM(CASE WHEN {RAW_MATCH} THEN 1 ELSE 0 END) AS n_ai_raw,
             SUM(CASE WHEN {CORE_MATCH} THEN 1 ELSE 0 END) AS n_ai_core
      FROM '{CORE}'
      WHERE {CORE_FILTER}
        AND (is_swe OR is_swe_adjacent OR is_control)
      GROUP BY 1,2
      ORDER BY 1,2
    """
    df = con.execute(sql).df()
    df["rate_raw"] = df["n_ai_raw"] / df["n"]
    df["rate_core"] = df["n_ai_core"] / df["n"]
    df["delta_pp"] = (df["rate_core"] - df["rate_raw"]) * 100
    df.to_csv(TABLES / "substrate_A1_swe_vs_control_rates.csv", index=False)
    print(df.to_string())

    # Compute the 23:1 deltas
    def deltas(group):
        s = df[df["analysis_group"] == group].set_index("period")
        if "2024-01" not in s.index or "2026-04" not in s.index:
            return np.nan, np.nan
        return (s.loc["2026-04", "rate_raw"] - s.loc["2024-01", "rate_raw"],
                s.loc["2026-04", "rate_core"] - s.loc["2024-01", "rate_core"])

    swe_d_raw, swe_d_core = deltas("swe")
    ctrl_d_raw, ctrl_d_core = deltas("control")
    ratio_raw = swe_d_raw / ctrl_d_raw if ctrl_d_raw else np.nan
    ratio_core = swe_d_core / ctrl_d_core if ctrl_d_core else np.nan

    summary = pd.DataFrame([{
        "claim": "H1 SWE-vs-control delta ratio (2024-01 -> 2026-04)",
        "swe_delta_raw_pp": swe_d_raw * 100,
        "swe_delta_core_pp": swe_d_core * 100,
        "control_delta_raw_pp": ctrl_d_raw * 100,
        "control_delta_core_pp": ctrl_d_core * 100,
        "ratio_raw": ratio_raw,
        "ratio_core": ratio_core,
        "verdict": "SURVIVES" if ratio_core >= 0.7 * ratio_raw else "WEAKENS",
    }])
    summary.to_csv(TABLES / "substrate_A1_swe_control_ratio.csv", index=False)
    print(summary.to_string())
    return df, summary


# ---------------------------------------------------------------------------
# A2 — H2 within-firm rewrite +19.4 pp on 292-firm panel
# ---------------------------------------------------------------------------

def a2_within_firm(con):
    print("\n===== A2: H2 within-firm rewrite (292-firm panel) =====")
    # Reproduce the 292-firm panel (same SQL pattern as S17)
    sql = f"""
      WITH base AS (
        SELECT company_name_canonical,
               CASE WHEN period LIKE '2024%' THEN '2024' ELSE '2026' END AS yr,
               {RAW_MATCH} AS ai_raw,
               {CORE_MATCH} AS ai_core
        FROM '{CORE}'
        WHERE {CORE_FILTER} AND is_swe = true
          AND company_name_canonical IS NOT NULL
      ),
      g AS (
        SELECT company_name_canonical, yr,
               COUNT(*) AS n,
               SUM(ai_raw::INTEGER) AS n_ai_raw,
               SUM(ai_core::INTEGER) AS n_ai_core
        FROM base
        GROUP BY 1,2
      )
      SELECT * FROM g
    """
    g = con.execute(sql).df()
    pivot = g.pivot(index="company_name_canonical", columns="yr",
                    values=["n", "n_ai_raw", "n_ai_core"]).reset_index()
    pivot.columns = ["company"] + [f"{a}_{b}" for a, b in pivot.columns[1:]]
    panel = pivot.dropna(subset=["n_2024", "n_2026"])
    panel = panel[(panel["n_2024"] >= 5) & (panel["n_2026"] >= 5)].copy()
    for col in ["n_2024", "n_2026", "n_ai_raw_2024", "n_ai_raw_2026",
                "n_ai_core_2024", "n_ai_core_2026"]:
        panel[col] = panel[col].astype(int)
    panel["rate_raw_2024"] = panel["n_ai_raw_2024"] / panel["n_2024"]
    panel["rate_raw_2026"] = panel["n_ai_raw_2026"] / panel["n_2026"]
    panel["rate_core_2024"] = panel["n_ai_core_2024"] / panel["n_2024"]
    panel["rate_core_2026"] = panel["n_ai_core_2026"] / panel["n_2026"]
    panel["delta_raw"] = panel["rate_raw_2026"] - panel["rate_raw_2024"]
    panel["delta_core"] = panel["rate_core_2026"] - panel["rate_core_2024"]

    n_firms = len(panel)
    mean_raw = panel["delta_raw"].mean()
    mean_core = panel["delta_core"].mean()
    pct_up_raw = (panel["delta_raw"] > 0).mean() * 100
    pct_up_core = (panel["delta_core"] > 0).mean() * 100
    pct_up10_raw = (panel["delta_raw"] > 0.10).mean() * 100
    pct_up10_core = (panel["delta_core"] > 0.10).mean() * 100

    # Boilerplate-cancellation test: does each firm have similar boilerplate
    # share in 2024 vs 2026? If yes, within-firm delta should be insensitive.
    # Boilerplate share = 1 - mean(core_len)/mean(raw_len). Compute per
    # firm-period mean lengths to crosscheck.
    bp = con.execute(f"""
      SELECT company_name_canonical AS company,
             CASE WHEN period LIKE '2024%' THEN '2024' ELSE '2026' END AS yr,
             AVG(LENGTH(description)) AS raw_len,
             AVG(LENGTH(description_core_llm)) AS core_len
      FROM '{CORE}'
      WHERE {CORE_FILTER} AND is_swe = true
        AND company_name_canonical IS NOT NULL
      GROUP BY 1,2
    """).df()
    bp["bp_share"] = 1 - (bp["core_len"] / bp["raw_len"])
    bp_pivot = bp.pivot(index="company", columns="yr", values="bp_share").reset_index()
    bp_pivot.columns = ["company"] + [f"bp_{c}" for c in bp_pivot.columns[1:]]
    panel_bp = panel.merge(bp_pivot, on="company", how="left")
    panel_bp["bp_share_change"] = panel_bp.get("bp_2026", 0) - panel_bp.get("bp_2024", 0)
    bp_change_mean = panel_bp["bp_share_change"].mean()
    bp_change_sd = panel_bp["bp_share_change"].std()

    panel_bp.to_csv(TABLES / "substrate_A2_within_firm_panel.csv", index=False)

    summary = pd.DataFrame([{
        "claim": "H2 within-firm mean delta on 292-firm panel",
        "n_firms_panel": n_firms,
        "mean_delta_raw_pp": mean_raw * 100,
        "mean_delta_core_pp": mean_core * 100,
        "pct_firms_up_raw": pct_up_raw,
        "pct_firms_up_core": pct_up_core,
        "pct_firms_up10_raw": pct_up10_raw,
        "pct_firms_up10_core": pct_up10_core,
        "mean_bp_share_change_2026_2024": bp_change_mean,
        "sd_bp_share_change": bp_change_sd,
        "verdict": verdict(mean_raw, mean_core),
    }])
    summary.to_csv(TABLES / "substrate_A2_within_firm_summary.csv", index=False)
    print(summary.to_string())
    return panel_bp, summary


# ---------------------------------------------------------------------------
# A3 — H4 vendor leaderboard
# ---------------------------------------------------------------------------

def a3_vendor_leaderboard(con):
    print("\n===== A3: H4 vendor leaderboard 2026-04 =====")
    vendors = ["copilot", "claude", "openai", "cursor", "anthropic",
               "gemini", "gpt", "chatgpt", "github copilot"]
    rows = []
    for v in vendors:
        v_pat = r"\b" + v.replace(" ", r"\s") + r"\b"
        sql = f"""
          SELECT COUNT(*) AS n,
                 SUM(CASE WHEN regexp_matches(LOWER(COALESCE(description, '')), '{v_pat}') THEN 1 ELSE 0 END) AS n_raw,
                 SUM(CASE WHEN regexp_matches(LOWER(COALESCE(description_core_llm, '')), '{v_pat}') THEN 1 ELSE 0 END) AS n_core
          FROM '{CORE}'
          WHERE {CORE_FILTER} AND is_swe = true AND period = '2026-04'
        """
        r = con.execute(sql).df().iloc[0]
        rows.append({
            "vendor": v,
            "n": int(r["n"]),
            "rate_raw": r["n_raw"] / r["n"],
            "rate_core": r["n_core"] / r["n"],
        })
    df = pd.DataFrame(rows)
    df["delta_pp"] = (df["rate_core"] - df["rate_raw"]) * 100
    df["pct_change"] = (df["rate_core"] - df["rate_raw"]) / df["rate_raw"]
    df["verdict"] = [verdict(a, b) for a, b in zip(df["rate_raw"], df["rate_core"])]
    df.to_csv(TABLES / "substrate_A3_vendor_leaderboard.csv", index=False)
    print(df.to_string())
    return df


# ---------------------------------------------------------------------------
# A4 — H6 Big Tech +17 pp
# ---------------------------------------------------------------------------

def a4_bigtech(con):
    print("\n===== A4: H6 Big Tech vs rest =====")
    bt_list = ", ".join("'" + b + "'" for b in BIG_TECH_CANONICAL)
    sql = f"""
      SELECT period,
             CASE WHEN LOWER(company_name_canonical) IN ({bt_list}) THEN 'big_tech' ELSE 'rest' END AS tier,
             COUNT(*) AS n,
             SUM(CASE WHEN {RAW_MATCH} THEN 1 ELSE 0 END) AS n_ai_raw,
             SUM(CASE WHEN {CORE_MATCH} THEN 1 ELSE 0 END) AS n_ai_core
      FROM '{CORE}'
      WHERE {CORE_FILTER} AND is_swe = true
      GROUP BY 1,2 ORDER BY 1,2
    """
    df = con.execute(sql).df()
    df["rate_raw"] = df["n_ai_raw"] / df["n"]
    df["rate_core"] = df["n_ai_core"] / df["n"]
    df.to_csv(TABLES / "substrate_A4_bigtech.csv", index=False)
    print(df.to_string())

    # Headline: 2026-04 gap big_tech minus rest, raw vs core.
    sub = df[df["period"] == "2026-04"].set_index("tier")
    gap_raw = sub.loc["big_tech", "rate_raw"] - sub.loc["rest", "rate_raw"]
    gap_core = sub.loc["big_tech", "rate_core"] - sub.loc["rest", "rate_core"]
    summary = pd.DataFrame([{
        "claim": "H6 Big Tech minus Rest (2026-04)",
        "bt_rate_raw": sub.loc["big_tech", "rate_raw"],
        "bt_rate_core": sub.loc["big_tech", "rate_core"],
        "rest_rate_raw": sub.loc["rest", "rate_raw"],
        "rest_rate_core": sub.loc["rest", "rate_core"],
        "gap_raw_pp": gap_raw * 100,
        "gap_core_pp": gap_core * 100,
        "verdict": "SURVIVES" if (gap_core >= 0.7 * gap_raw and gap_core > 0.05) else "WEAKENS",
    }])
    summary.to_csv(TABLES / "substrate_A4_bigtech_summary.csv", index=False)
    print(summary.to_string())
    return df, summary


# ---------------------------------------------------------------------------
# A5 — F2 industry spread (non-tech share)
#       Uses title classification only (is_swe), no AI vocab.
#       Independent of substrate. Skip.
# A6 — F3 junior-first (AI rate by seniority)
# ---------------------------------------------------------------------------

def a6_seniority(con):
    print("\n===== A6: F3 AI rate by seniority (uniform) =====")
    sql = f"""
      SELECT period, seniority_3level,
             COUNT(*) AS n,
             SUM(CASE WHEN {RAW_MATCH} THEN 1 ELSE 0 END) AS n_ai_raw,
             SUM(CASE WHEN {CORE_MATCH} THEN 1 ELSE 0 END) AS n_ai_core
      FROM '{CORE}'
      WHERE {CORE_FILTER} AND is_swe = true
      GROUP BY 1,2 ORDER BY 1,2
    """
    df = con.execute(sql).df()
    df["rate_raw"] = df["n_ai_raw"] / df["n"]
    df["rate_core"] = df["n_ai_core"] / df["n"]
    df.to_csv(TABLES / "substrate_A6_seniority.csv", index=False)

    # Headline finding: rates are uniform across junior/mid/senior in 2026-04.
    # Test whether the gap junior-vs-senior survives in core.
    sub = df[df["period"] == "2026-04"]
    sub = sub[sub["seniority_3level"].isin(["junior", "mid", "senior"])]
    raw_gap = sub["rate_raw"].max() - sub["rate_raw"].min()
    core_gap = sub["rate_core"].max() - sub["rate_core"].min()
    summary = pd.DataFrame([{
        "claim": "F3 max-min AI rate spread across seniority (2026-04)",
        "raw_max_min_spread_pp": raw_gap * 100,
        "core_max_min_spread_pp": core_gap * 100,
        "verdict": "SURVIVES" if abs(core_gap - raw_gap) * 100 < 2 else "MOVED",
    }])
    summary.to_csv(TABLES / "substrate_A6_seniority_summary.csv", index=False)
    print(sub.to_string())
    print(summary.to_string())
    return df, summary


# ---------------------------------------------------------------------------
# A7 — Other observations: legacy roles & length growth
# ---------------------------------------------------------------------------

def a7_other(con):
    print("\n===== A7: misc — Copilot 0.10%, legacy disappearing 3.6%/14.4% =====")
    # Copilot rate among SWE 2024 (the "0.10% in 2024-Q1" claim).
    sql = f"""
      SELECT period,
             COUNT(*) AS n,
             SUM(CASE WHEN regexp_matches(LOWER(COALESCE(description, '')), '\\bcopilot\\b') THEN 1 ELSE 0 END) AS n_raw,
             SUM(CASE WHEN regexp_matches(LOWER(COALESCE(description_core_llm, '')), '\\bcopilot\\b') THEN 1 ELSE 0 END) AS n_core
      FROM '{CORE}'
      WHERE {CORE_FILTER} AND is_swe = true
      GROUP BY 1 ORDER BY 1
    """
    cop = con.execute(sql).df()
    cop["rate_raw"] = cop["n_raw"] / cop["n"]
    cop["rate_core"] = cop["n_core"] / cop["n"]
    cop.to_csv(TABLES / "substrate_A7_copilot_by_period.csv", index=False)
    print("Copilot rate by period:")
    print(cop.to_string())

    # Legacy disappearing titles (T36 source AI rate vs unified_core 2026 neighbor)
    # Use the 6 v9 disappearing titles.
    v9_disappearing = ["java architect", "drupal developer", "scala developer",
                       "java application developer", "database developer", "devops architect"]
    titles_sql = ",".join("'" + t + "'" for t in v9_disappearing)
    sql2 = f"""
      SELECT COUNT(*) AS n,
             SUM(CASE WHEN {RAW_MATCH} THEN 1 ELSE 0 END) AS n_raw,
             SUM(CASE WHEN {CORE_MATCH} THEN 1 ELSE 0 END) AS n_core
      FROM '{CORE}'
      WHERE {CORE_FILTER} AND is_swe = true AND period LIKE '2026%'
        AND LOWER(TRIM(title)) IN ({titles_sql})
    """
    leg = con.execute(sql2).df().iloc[0]
    rate_raw = leg["n_raw"] / leg["n"] if leg["n"] else np.nan
    rate_core = leg["n_core"] / leg["n"] if leg["n"] else np.nan

    # Market 2026 SWE
    sql3 = f"""
      SELECT COUNT(*) AS n,
             SUM(CASE WHEN {RAW_MATCH} THEN 1 ELSE 0 END) AS n_raw,
             SUM(CASE WHEN {CORE_MATCH} THEN 1 ELSE 0 END) AS n_core
      FROM '{CORE}'
      WHERE {CORE_FILTER} AND is_swe = true AND period LIKE '2026%'
    """
    mkt = con.execute(sql3).df().iloc[0]
    mkt_raw = mkt["n_raw"] / mkt["n"]
    mkt_core = mkt["n_core"] / mkt["n"]

    summary = pd.DataFrame([{
        "claim": "Legacy disappearing titles (6 v9 titles) AI rate 2026",
        "n_postings": int(leg["n"]),
        "rate_raw": rate_raw,
        "rate_core": rate_core,
        "market_rate_raw": mkt_raw,
        "market_rate_core": mkt_core,
        "ratio_raw": rate_raw / mkt_raw if mkt_raw else np.nan,
        "ratio_core": rate_core / mkt_core if mkt_core else np.nan,
        "verdict": verdict(rate_raw, rate_core),
    }])
    summary.to_csv(TABLES / "substrate_A7_legacy_titles.csv", index=False)
    print(summary.to_string())
    return cop, summary


# ---------------------------------------------------------------------------
# B — Claim 7: cross-occupation rank
# ---------------------------------------------------------------------------

# Subgroup classifiers ported from S25 (just the headline-occupation set)
SWE_RX = re.compile(r"\b(?:ml|machine learning|ai|mlops)\s*(?:engineer|developer|architect|scientist)\b", re.IGNORECASE)
SWE_ADJ_PATS = [
    ("ml_engineer", re.compile(r"\b(?:machine learning|ml|ai|mlops)\s*(?:engineer|developer|scientist|architect)\b", re.IGNORECASE)),
    ("data_scientist", re.compile(r"\b(?:data scientist|data science)\b", re.IGNORECASE)),
    ("data_engineer", re.compile(r"\b(?:data engineer|analytics engineer)\b", re.IGNORECASE)),
    ("data_analyst", re.compile(r"\b(?:data analyst|business analyst|analytics analyst)\b", re.IGNORECASE)),
    ("security_engineer", re.compile(r"\b(?:security|cyber|cybersecurity|information security|infosec)\s*(?:engineer|architect)\b", re.IGNORECASE)),
    ("network_engineer", re.compile(r"\bnetwork\s*(?:engineer|architect)\b", re.IGNORECASE)),
    ("qa_engineer", re.compile(r"\b(?:qa|quality)\s*(?:engineer|analyst)\b", re.IGNORECASE)),
    ("devops_engineer", re.compile(r"\b(?:devops|site reliability|sre|platform)\s*(?:engineer|architect)\b", re.IGNORECASE)),
    ("solutions_architect", re.compile(r"\b(?:solution|solutions|enterprise|cloud|technical)\s*architect\b", re.IGNORECASE)),
    ("systems_admin", re.compile(r"\b(?:systems?|database)\s*administrator\b", re.IGNORECASE)),
]
CTRL_PATS = [
    ("accountant", re.compile(r"\b(?:accountant|accounting|cpa|bookkeeper)\b", re.IGNORECASE)),
    ("financial_analyst", re.compile(r"\bfinancial\s*analyst\b", re.IGNORECASE)),
    ("nurse", re.compile(r"\b(?:registered nurse|rn|nurse practitioner|lpn|licensed practical nurse|cna|nursing)\b", re.IGNORECASE)),
    ("electrical_engineer", re.compile(r"\belectrical\s*engineer", re.IGNORECASE)),
    ("mechanical_engineer", re.compile(r"\bmechanical\s*engineer", re.IGNORECASE)),
    ("civil_engineer", re.compile(r"\bcivil\s*engineer", re.IGNORECASE)),
]

# Worker-side benchmarks (mid 'any-use' as in S25 headline pairing).
WORKER_ANY_MID = {
    "other_swe": (0.63 + 0.84 + 0.90) / 3,
    "ml_engineer": 0.85,
    "data_scientist": 0.75,
    "data_engineer": 0.75,
    "data_analyst": 0.60,
    "security_engineer": 0.40,
    "devops_engineer": 0.70,
    "solutions_architect": 0.65,
    "qa_engineer": 0.50,
    "network_engineer": 0.35,
    "systems_admin": 0.40,
    "accountant": 0.50,
    "financial_analyst": 0.30,
    "nurse": 0.15,
    "electrical_engineer": 0.30,
    "mechanical_engineer": 0.25,
    "civil_engineer": 0.22,
}

HEADLINE_OCCS = [
    ("other_swe", "swe"), ("ml_engineer", "swe_adjacent"),
    ("data_scientist", "swe_adjacent"), ("data_engineer", "swe_adjacent"),
    ("data_analyst", "swe_adjacent"), ("security_engineer", "swe_adjacent"),
    ("devops_engineer", "swe_adjacent"), ("solutions_architect", "swe_adjacent"),
    ("qa_engineer", "swe_adjacent"), ("network_engineer", "swe_adjacent"),
    ("systems_admin", "swe_adjacent"),
    ("accountant", "control"), ("financial_analyst", "control"),
    ("nurse", "control"), ("electrical_engineer", "control"),
    ("mechanical_engineer", "control"), ("civil_engineer", "control"),
]


def classify(t, g):
    if not isinstance(t, str):
        return None
    if g == "swe":
        return "ml_engineer" if SWE_RX.search(t) else "other_swe"
    if g == "swe_adjacent":
        for n, rx in SWE_ADJ_PATS:
            if rx.search(t):
                return n
        return "other_adjacent"
    if g == "control":
        for n, rx in CTRL_PATS:
            if rx.search(t):
                return n
        return "other_control"
    return None


def b_cross_occupation_rank(con):
    print("\n===== B: Claim 7 cross-occupation rank correlation =====")
    from scipy.stats import spearmanr

    # Load only what we need
    sql = f"""
      SELECT title, analysis_group,
             CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS yr,
             {RAW_MATCH} AS ai_raw,
             {CORE_MATCH} AS ai_core
      FROM '{CORE}'
      WHERE {CORE_FILTER}
        AND analysis_group IN ('swe','swe_adjacent','control')
    """
    df = con.execute(sql).df()
    df["sg"] = [classify(t, g) for t, g in zip(df["title"], df["analysis_group"])]
    df = df.dropna(subset=["sg"])
    grp = df.groupby(["analysis_group", "sg", "yr"]).agg(
        n=("ai_raw", "size"),
        rate_raw=("ai_raw", "mean"),
        rate_core=("ai_core", "mean"),
    ).reset_index()
    grp.to_csv(TABLES / "substrate_B_subgroup_rates.csv", index=False)

    # Pivot to wide (analysis_group, sg) x yr.
    wide = grp.pivot_table(index=["analysis_group", "sg"], columns="yr",
                           values=["rate_raw", "rate_core", "n"], aggfunc="first").reset_index()
    wide.columns = [a if not b else f"{a}_{b}" for a, b in wide.columns]
    wide["delta_raw"] = wide["rate_raw_2026"] - wide["rate_raw_2024"]
    wide["delta_core"] = wide["rate_core_2026"] - wide["rate_core_2024"]

    # Restrict to headline occupation set.
    headline = wide[wide.apply(lambda r: (r["sg"], r["analysis_group"]) in set(HEADLINE_OCCS), axis=1)].copy()
    headline["worker_any_mid"] = headline["sg"].map(WORKER_ANY_MID)
    headline = headline.dropna(subset=["worker_any_mid"])
    headline.to_csv(TABLES / "substrate_B_pair_table.csv", index=False)

    rows = []
    for label, col in [
        ("level_2026_raw", "rate_raw_2026"),
        ("level_2026_core", "rate_core_2026"),
        ("level_2024_raw", "rate_raw_2024"),
        ("level_2024_core", "rate_core_2024"),
        ("delta_raw", "delta_raw"),
        ("delta_core", "delta_core"),
    ]:
        d = headline.dropna(subset=["worker_any_mid", col])
        if len(d) >= 4:
            rho, p = spearmanr(d["worker_any_mid"], d[col])
        else:
            rho, p = (np.nan, np.nan)
        rows.append({"method": label, "n": len(d), "spearman": rho, "p": p})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "substrate_B_method_comparison.csv", index=False)
    print(out.to_string())

    # Compare delta-rho raw vs core for the verdict
    raw_delta_rho = out.loc[out["method"] == "delta_raw", "spearman"].iloc[0]
    core_delta_rho = out.loc[out["method"] == "delta_core", "spearman"].iloc[0]
    raw_2026_rho = out.loc[out["method"] == "level_2026_raw", "spearman"].iloc[0]
    core_2026_rho = out.loc[out["method"] == "level_2026_core", "spearman"].iloc[0]
    summary = pd.DataFrame([{
        "claim": "Claim 7 — cross-occupation Spearman rho (worker-side AI vs employer-side AI rate)",
        "rho_2026_raw": raw_2026_rho,
        "rho_2026_core": core_2026_rho,
        "rho_delta_raw": raw_delta_rho,
        "rho_delta_core": core_delta_rho,
        "verdict_2026_level": "SURVIVES" if abs(core_2026_rho - raw_2026_rho) < 0.10 else "MOVED",
        "verdict_delta": "SURVIVES" if abs(core_delta_rho - raw_delta_rho) < 0.10 else "MOVED",
    }])
    summary.to_csv(TABLES / "substrate_B_summary.csv", index=False)
    print(summary.to_string())
    return out, summary


# ---------------------------------------------------------------------------
# C — Composite A deepdives
# ---------------------------------------------------------------------------

TECH_HUBS = {
    "San Francisco Bay Area",
    "Seattle Metro",
    "New York City Metro",
    "Austin Metro",
    "Boston Metro",
}


def c_composite_a(con):
    print("\n===== C: Composite A deepdives =====")
    # DD1 — Hospitals & Health Care vs Software Development, 2026.
    sql = f"""
      SELECT company_industry,
             COUNT(*) AS n,
             SUM(CASE WHEN {RAW_MATCH} THEN 1 ELSE 0 END) AS n_ai_raw,
             SUM(CASE WHEN {CORE_MATCH} THEN 1 ELSE 0 END) AS n_ai_core
      FROM '{CORE}'
      WHERE {CORE_FILTER} AND is_swe = true AND period LIKE '2026%'
        AND company_industry IN ('Hospitals and Health Care', 'Software Development', 'Financial Services')
      GROUP BY 1
    """
    dd1 = con.execute(sql).df()
    dd1["rate_raw"] = dd1["n_ai_raw"] / dd1["n"]
    dd1["rate_core"] = dd1["n_ai_core"] / dd1["n"]
    dd1.to_csv(TABLES / "substrate_C_dd1_industries.csv", index=False)
    print("DD1 industry rates 2026:")
    print(dd1.to_string())

    # DD2 — FS vs SWE under three regex variants.
    REGEX_VARIANTS = {
        "canonical_broad": AI_VOCAB_PATTERN,
        "strict_v9like": (
            r"(?i)\b(large language model|generative ai|genai|gen ai|"
            r"foundation model|transformer model|chatgpt|openai|anthropic|"
            r"copilot|claude|github copilot|cursor ide|windsurf ide|llm|"
            r"prompt engineering|prompt engineer|ai agent|agentic|"
            r"retrieval augmented|vector database|mlops|llmops)\b"
        ),
        "tooling_only": (
            r"(?i)\b(chatgpt|claude|copilot|openai|anthropic|github copilot|"
            r"cursor ide|windsurf ide|llm|genai|gen ai|generative ai|"
            r"foundation model|ai agent|agentic|retrieval augmented|rag|"
            r"prompt engineering|prompt engineer|llmops)\b"
        ),
    }
    rows = []
    for name, pat in REGEX_VARIANTS.items():
        for ind in ["Financial Services", "Software Development"]:
            r = con.execute(f"""
              SELECT COUNT(*) AS n,
                     SUM(CASE WHEN regexp_matches(COALESCE(description, ''), '{pat}') THEN 1 ELSE 0 END) AS n_raw,
                     SUM(CASE WHEN regexp_matches(COALESCE(description_core_llm, ''), '{pat}') THEN 1 ELSE 0 END) AS n_core
              FROM '{CORE}'
              WHERE {CORE_FILTER} AND is_swe = true AND period LIKE '2026%'
                AND company_industry = '{ind}'
            """).df().iloc[0]
            rows.append({
                "regex": name, "industry": ind, "n": int(r["n"]),
                "rate_raw": r["n_raw"] / r["n"],
                "rate_core": r["n_core"] / r["n"],
            })
    dd2 = pd.DataFrame(rows)
    deltas = []
    for name in REGEX_VARIANTS:
        fs = dd2[(dd2.regex == name) & (dd2.industry == "Financial Services")].iloc[0]
        sw = dd2[(dd2.regex == name) & (dd2.industry == "Software Development")].iloc[0]
        deltas.append({
            "regex": name,
            "fs_rate_raw": fs["rate_raw"], "fs_rate_core": fs["rate_core"],
            "swe_rate_raw": sw["rate_raw"], "swe_rate_core": sw["rate_core"],
            "fs_minus_swe_raw_pp": (fs["rate_raw"] - sw["rate_raw"]) * 100,
            "fs_minus_swe_core_pp": (fs["rate_core"] - sw["rate_core"]) * 100,
        })
    dd2_summary = pd.DataFrame(deltas)
    dd2_summary.to_csv(TABLES / "substrate_C_dd2_fs_vs_swe.csv", index=False)
    print("\nDD2 FS vs SWE deltas across regexes:")
    print(dd2_summary.to_string())

    # DD3 — Bay vs rest builder/user token gaps.
    tokens = ["openai", "anthropic", "agentic", "ai agent", "llm", "foundation model",
              "copilot", "github copilot", "claude", "prompt engineering", "rag", "mlops"]
    rest_clause = "metro_area NOT IN (" + ",".join("'" + m + "'" for m in TECH_HUBS) + ")"
    rows = []
    for substrate, src_col in [("raw", "description"), ("core", "description_core_llm")]:
        for zone, clause in [("bay", "metro_area = 'San Francisco Bay Area'"),
                             ("rest", rest_clause)]:
            df = con.execute(f"""
              SELECT LOWER(COALESCE({src_col}, '')) AS d
              FROM '{CORE}'
              WHERE {CORE_FILTER} AND is_swe = true AND period LIKE '2026%'
                AND metro_area IS NOT NULL AND {clause}
                AND NOT regexp_matches(LOWER(COALESCE(title, '')), '{BUILDER_TITLE_PATTERN}')
                AND regexp_matches(COALESCE({src_col}, ''), '{AI_VOCAB_PATTERN}')
            """).df()
            n = len(df)
            for tok in tokens:
                hits = sum(1 for txt in df["d"] if re.search(r"\b" + re.escape(tok) + r"\b", txt or ""))
                rows.append({"substrate": substrate, "zone": zone, "token": tok,
                             "n_posts": n, "n_hit": hits, "rate": hits / n if n else 0.0})
    tok = pd.DataFrame(rows)
    out_rows = []
    for t in tokens:
        rb = tok[(tok.substrate == "raw") & (tok.zone == "bay") & (tok.token == t)].iloc[0]
        rr = tok[(tok.substrate == "raw") & (tok.zone == "rest") & (tok.token == t)].iloc[0]
        cb = tok[(tok.substrate == "core") & (tok.zone == "bay") & (tok.token == t)].iloc[0]
        cr = tok[(tok.substrate == "core") & (tok.zone == "rest") & (tok.token == t)].iloc[0]
        gap_raw = (rb["rate"] - rr["rate"]) * 100
        gap_core = (cb["rate"] - cr["rate"]) * 100
        out_rows.append({
            "token": t,
            "raw_bay_pct": rb["rate"] * 100,
            "raw_rest_pct": rr["rate"] * 100,
            "raw_gap_pp": gap_raw,
            "core_bay_pct": cb["rate"] * 100,
            "core_rest_pct": cr["rate"] * 100,
            "core_gap_pp": gap_core,
            "gap_change_pp": gap_core - gap_raw,
            "flips_sign": (gap_raw * gap_core) < 0,
        })
    dd3 = pd.DataFrame(out_rows).sort_values("raw_gap_pp", ascending=False)
    dd3.to_csv(TABLES / "substrate_C_dd3_token_gap.csv", index=False)
    print("\nDD3 token gap raw vs core:")
    print(dd3.to_string())
    return dd1, dd2_summary, dd3


# ---------------------------------------------------------------------------
# D — Composite B v2 (BERTopic cluster characterization)
# ---------------------------------------------------------------------------

FDE_TITLE_REGEX = r"(?i)forward[\s\-]?deployed"


def d_composite_b(con):
    print("\n===== D: Composite B v2 (BERTopic Topic 1 + FDE + legacy substitution) =====")
    # Cluster Topic 1 share + AI rate, raw vs core.
    sql = f"""
      WITH joined AS (
        SELECT u.uid, u.period,
               CASE WHEN u.period LIKE '2024%' THEN '2024' ELSE '2026' END AS yr,
               l.archetype_id,
               {RAW_MATCH.replace("description", "u.description")} AS ai_raw,
               {CORE_MATCH.replace("description_core_llm", "u.description_core_llm")} AS ai_core
        FROM '{CORE}' u
        JOIN '{ARCH}' l USING (uid)
      )
      SELECT yr,
             COUNT(*) AS n_total,
             SUM(CASE WHEN archetype_id=1 THEN 1 ELSE 0 END) AS n_topic1,
             SUM(CASE WHEN archetype_id=1 AND ai_raw THEN 1 ELSE 0 END) AS n_topic1_ai_raw,
             SUM(CASE WHEN archetype_id=1 AND ai_core THEN 1 ELSE 0 END) AS n_topic1_ai_core
      FROM joined
      GROUP BY 1 ORDER BY 1
    """
    cl = con.execute(sql).df()
    cl["pct_topic1"] = cl["n_topic1"] / cl["n_total"] * 100
    cl["topic1_ai_raw"] = cl["n_topic1_ai_raw"] / cl["n_topic1"]
    cl["topic1_ai_core"] = cl["n_topic1_ai_core"] / cl["n_topic1"]
    cl.to_csv(TABLES / "substrate_D_topic1.csv", index=False)
    print("Topic 1 share + AI rates (within cluster), raw vs core:")
    print(cl.to_string())

    # 5.2x growth check (cluster share 2024 -> 2026 — substrate-INVARIANT
    # because cluster membership doesn't depend on AI vocab; just record).
    p24 = cl[cl["yr"] == "2024"]["pct_topic1"].iloc[0]
    p26 = cl[cl["yr"] == "2026"]["pct_topic1"].iloc[0]
    ai24_raw = cl[cl["yr"] == "2024"]["topic1_ai_raw"].iloc[0]
    ai24_core = cl[cl["yr"] == "2024"]["topic1_ai_core"].iloc[0]
    ai26_raw = cl[cl["yr"] == "2026"]["topic1_ai_raw"].iloc[0]
    ai26_core = cl[cl["yr"] == "2026"]["topic1_ai_core"].iloc[0]
    summary = pd.DataFrame([{
        "claim": "Topic 1 RAG/AI cluster",
        "share_2024_pct": p24,
        "share_2026_pct": p26,
        "growth_multiple": p26 / p24,
        "topic1_ai_rate_2024_raw": ai24_raw,
        "topic1_ai_rate_2024_core": ai24_core,
        "topic1_ai_rate_2026_raw": ai26_raw,
        "topic1_ai_rate_2026_core": ai26_core,
        "verdict_share": "SURVIVES (substrate-invariant; cluster membership uses core)",
        "verdict_ai_rate_2026": verdict(ai26_raw, ai26_core),
    }])
    summary.to_csv(TABLES / "substrate_D_topic1_summary.csv", index=False)
    print(summary.to_string())

    # FDE thread — 1.95x AI density vs general SWE.
    sql_fde = f"""
      SELECT
        SUM(CASE WHEN regexp_matches(COALESCE(title, ''), '{FDE_TITLE_REGEX}') THEN 1 ELSE 0 END) AS n_fde,
        SUM(CASE WHEN regexp_matches(COALESCE(title, ''), '{FDE_TITLE_REGEX}') AND {RAW_MATCH} THEN 1 ELSE 0 END) AS n_fde_ai_raw,
        SUM(CASE WHEN regexp_matches(COALESCE(title, ''), '{FDE_TITLE_REGEX}') AND {CORE_MATCH} THEN 1 ELSE 0 END) AS n_fde_ai_core,
        COUNT(*) AS n_total,
        SUM(CASE WHEN {RAW_MATCH} THEN 1 ELSE 0 END) AS n_total_ai_raw,
        SUM(CASE WHEN {CORE_MATCH} THEN 1 ELSE 0 END) AS n_total_ai_core
      FROM '{CORE}'
      WHERE {CORE_FILTER} AND is_swe = true AND period LIKE '2026%'
    """
    fde = con.execute(sql_fde).df().iloc[0]
    fde_raw = fde["n_fde_ai_raw"] / fde["n_fde"]
    fde_core = fde["n_fde_ai_core"] / fde["n_fde"]
    gen_raw = fde["n_total_ai_raw"] / fde["n_total"]
    gen_core = fde["n_total_ai_core"] / fde["n_total"]
    fde_summary = pd.DataFrame([{
        "claim": "FDE 2026 AI density vs general SWE",
        "n_fde_postings": int(fde["n_fde"]),
        "fde_rate_raw": fde_raw,
        "fde_rate_core": fde_core,
        "gen_rate_raw": gen_raw,
        "gen_rate_core": gen_core,
        "ratio_raw": fde_raw / gen_raw,
        "ratio_core": fde_core / gen_core,
        "verdict": "SURVIVES" if fde_core / gen_core >= 0.7 * fde_raw / gen_raw else "WEAKENS",
    }])
    fde_summary.to_csv(TABLES / "substrate_D_fde.csv", index=False)
    print("\nFDE density check:")
    print(fde_summary.to_string())

    # Legacy substitution — neighbor titles AI rate vs market 2026.
    # Read the T36 neighbor titles list.
    t36_path = PROJECT_ROOT / "exploration-archive/v9_final_opus_47/tables/T36/substitution_table_top1.csv"
    t36 = pd.read_csv(t36_path)
    neigh = t36["top_2026_neighbor"].tolist()
    titles_sql = ",".join("'" + t.lower().replace("'", "''") + "'" for t in neigh)
    sql_neigh = f"""
      SELECT COUNT(*) AS n,
             SUM(CASE WHEN {RAW_MATCH} THEN 1 ELSE 0 END) AS n_raw,
             SUM(CASE WHEN {CORE_MATCH} THEN 1 ELSE 0 END) AS n_core
      FROM '{CORE}'
      WHERE {CORE_FILTER} AND is_swe = true AND period LIKE '2026%'
        AND LOWER(TRIM(title)) IN ({titles_sql})
    """
    neigh_r = con.execute(sql_neigh).df().iloc[0]
    n_rate_raw = neigh_r["n_raw"] / neigh_r["n"]
    n_rate_core = neigh_r["n_core"] / neigh_r["n"]
    leg_summary = pd.DataFrame([{
        "claim": "T36 legacy-neighbor 2026 AI rate vs market (substitution thinness)",
        "neighbor_n": int(neigh_r["n"]),
        "neighbor_rate_raw": n_rate_raw,
        "neighbor_rate_core": n_rate_core,
        "market_rate_raw": gen_raw,
        "market_rate_core": gen_core,
        "ratio_raw": n_rate_raw / gen_raw,
        "ratio_core": n_rate_core / gen_core,
        "verdict": verdict(n_rate_raw, n_rate_core),
    }])
    leg_summary.to_csv(TABLES / "substrate_D_legacy_substitution.csv", index=False)
    print("\nLegacy substitution check:")
    print(leg_summary.to_string())
    return summary, fde_summary, leg_summary


# ---------------------------------------------------------------------------
# E — Self-mention audit ("openai flips +2.2 -> -0.5pp")
# ---------------------------------------------------------------------------

SELF_MENTION_FIRMS = [
    "OpenAI", "Anthropic", "xAI", "Cohere", "Perplexity",
    "Inflection AI", "Character.AI",
    "Microsoft", "Microsoft AI", "GitHub", "Google", "Meta",
    "Amazon", "Amazon Web Services (AWS)", "NVIDIA", "Adobe", "Salesforce",
    "Databricks",
]


def e_self_mention(con):
    print("\n===== E: self-mention audit (openai flips) =====")
    excl_sql = ",".join("'" + n.replace("'", "''") + "'" for n in SELF_MENTION_FIRMS)
    rest_clause = "metro_area NOT IN (" + ",".join("'" + m + "'" for m in TECH_HUBS) + ")"

    rows = []
    for substrate, src_col in [("raw", "description"), ("core", "description_core_llm")]:
        for excl_label, excl_clause in [("original", ""),
                                        ("self_excluded", f"AND company_name_canonical NOT IN ({excl_sql})")]:
            for zone, clause in [("bay", "metro_area = 'San Francisco Bay Area'"),
                                 ("rest", rest_clause)]:
                df = con.execute(f"""
                  SELECT LOWER(COALESCE({src_col}, '')) AS d
                  FROM '{CORE}'
                  WHERE {CORE_FILTER} AND is_swe = true AND period LIKE '2026%'
                    AND metro_area IS NOT NULL AND {clause}
                    AND NOT regexp_matches(LOWER(COALESCE(title, '')), '{BUILDER_TITLE_PATTERN}')
                    AND regexp_matches(COALESCE({src_col}, ''), '{AI_VOCAB_PATTERN}')
                    {excl_clause}
                """).df()
                n = len(df)
                hits = sum(1 for t in df["d"] if re.search(r"\bopenai\b", t or ""))
                rows.append({
                    "substrate": substrate, "scenario": excl_label, "zone": zone,
                    "n_posts": n, "n_openai": hits,
                    "rate_pct": (hits / n * 100) if n else 0,
                })
    df = pd.DataFrame(rows)

    # Compute the gap per (substrate, scenario): bay - rest.
    out_rows = []
    for substrate in ["raw", "core"]:
        for scen in ["original", "self_excluded"]:
            b = df[(df.substrate == substrate) & (df.scenario == scen) & (df.zone == "bay")].iloc[0]
            r = df[(df.substrate == substrate) & (df.scenario == scen) & (df.zone == "rest")].iloc[0]
            out_rows.append({
                "substrate": substrate, "scenario": scen,
                "bay_n": b["n_posts"], "rest_n": r["n_posts"],
                "bay_pct": b["rate_pct"], "rest_pct": r["rate_pct"],
                "gap_pp": b["rate_pct"] - r["rate_pct"],
            })
    out = pd.DataFrame(out_rows)
    out.to_csv(TABLES / "substrate_E_openai_flip.csv", index=False)
    print(out.to_string())
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    con = duckdb.connect()
    a1_swe_vs_control(con)
    a2_within_firm(con)
    a3_vendor_leaderboard(con)
    a4_bigtech(con)
    a6_seniority(con)
    a7_other(con)
    b_cross_occupation_rank(con)
    c_composite_a(con)
    d_composite_b(con)
    e_self_mention(con)
    print("\nAll substrate sensitivity sections complete.")


if __name__ == "__main__":
    main()
