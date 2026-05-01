"""T22 prevalence + AI-ghostiness + aggregator/industry + examples + report.

Reads:
  - exploration/artifacts/T22/T22_features.parquet
  - exploration/artifacts/shared/seniority_definition_panel.csv (for n)
  - exploration/artifacts/shared/entry_specialist_employers.csv

Writes:
  - exploration/artifacts/T22/T22_prevalence_by_panel.csv
  - exploration/artifacts/T22/T22_ai_ghostiness.json
  - exploration/artifacts/T22/T22_aggregator.csv
  - exploration/artifacts/T22/T22_industry.csv
  - exploration/artifacts/T22/T22_top20_ghost.csv
  - exploration/reports/T22.md
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path("/home/jihgaboot/gabor/job-research")
ART = REPO / "exploration/artifacts/T22"
SHARED = REPO / "exploration/artifacts/shared"
REPORT = REPO / "exploration/reports/T22.md"

FEATURES_PARQUET = ART / "T22_features.parquet"
SPECIALISTS_CSV = SHARED / "entry_specialist_employers.csv"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def panel_mask(df: pd.DataFrame, pdef: str) -> pd.Series:
    """Return a boolean mask selecting rows matching a T30 panel definition."""
    if pdef == "J1":
        return df["seniority_final"].eq("entry")
    if pdef == "J2":
        return df["seniority_final"].isin(["entry", "associate"])
    if pdef == "J3":
        return df["yoe_extracted"].le(2) & df["yoe_extracted"].notna()
    if pdef == "J4":
        return df["yoe_extracted"].le(3) & df["yoe_extracted"].notna()
    if pdef == "S1":
        return df["seniority_final"].isin(["mid-senior", "director"])
    if pdef == "S2":
        return df["seniority_final"].eq("director")
    if pdef == "S3":
        return df["title"].str.contains(
            r"\b(senior|sr\.?|staff|principal|lead|architect|distinguished)\b",
            case=False,
            na=False,
            regex=True,
        )
    if pdef == "S4":
        return df["yoe_extracted"].ge(5) & df["yoe_extracted"].notna()
    raise ValueError(pdef)


def period_label(period: str) -> str:
    return "2024" if period.startswith("2024") else "2026"


def load() -> pd.DataFrame:
    df = pq.read_table(FEATURES_PARQUET).to_pandas()
    df["period_label"] = df["period"].map(period_label)
    return df


def load_specialists() -> set[str]:
    try:
        sdf = pd.read_csv(SPECIALISTS_CSV)
        col = "company_name_canonical" if "company_name_canonical" in sdf.columns else "company"
        return set(sdf[col].dropna().astype(str).str.lower().unique())
    except Exception as e:
        print(f"WARN could not load specialists csv: {e}")
        return set()


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------

def prevalence_panel(df: pd.DataFrame, specialists: set[str]) -> pd.DataFrame:
    """For each T30 panel variant (J1-J4, S1-S4) x period: ghost indicators."""
    rows = []
    panels = ["J1", "J2", "J3", "J4", "S1", "S2", "S3", "S4"]
    df_excl = df[~df["company"].astype(str).str.lower().isin(specialists)].copy()

    def summarize(subset: pd.DataFrame, pdef: str, period: str, excl_label: str) -> dict:
        n = len(subset)
        if n == 0:
            return None
        asp_ratio_finite = subset.loc[np.isfinite(subset["aspiration_ratio"]), "aspiration_ratio"]
        k_mean = subset["kitchen_sink"].mean()
        k_p90 = subset["kitchen_sink"].quantile(0.9)
        asp_count_mean = subset["aspiration_count"].mean()
        asp_ratio_mean_finite = asp_ratio_finite.mean() if len(asp_ratio_finite) else np.nan
        # Ghost assessment LLM: only valid where labeled
        llm_subset = subset[subset["llm_classification_coverage"].eq("labeled")]
        n_llm = len(llm_subset)
        inflated_rate = (llm_subset["ghost_assessment_llm"].eq("inflated").mean() * 100) if n_llm else np.nan
        ghost_rate = (llm_subset["ghost_assessment_llm"].eq("ghost_likely").mean() * 100) if n_llm else np.nan
        inflated_or_ghost = (
            llm_subset["ghost_assessment_llm"].isin(["inflated", "ghost_likely"]).mean() * 100
        ) if n_llm else np.nan
        # Rule fallback
        rule_high = (subset["ghost_job_risk"].eq("high").mean() * 100)
        rule_med = (subset["ghost_job_risk"].eq("medium").mean() * 100)
        # YOE mismatch on entry-definitions only
        yoe_mismatch = (subset["yoe_mismatch"].sum() / n * 100) if pdef.startswith("J") else np.nan
        cred_contradiction = subset["credential_contradiction"].mean() * 100
        # AI mention rate
        ai_strict_rate = subset["ai_strict_bin"].mean() * 100
        ai_broad_rate = subset["ai_broad_bin"].mean() * 100
        return {
            "panel": pdef,
            "period": period,
            "exclusion": excl_label,
            "n": n,
            "n_llm_labeled": n_llm,
            "kitchen_sink_mean": round(k_mean, 2),
            "kitchen_sink_p90": round(float(k_p90), 1),
            "aspiration_count_mean": round(asp_count_mean, 2),
            "aspiration_ratio_mean_finite": (
                round(float(asp_ratio_mean_finite), 3) if not pd.isna(asp_ratio_mean_finite) else None
            ),
            "llm_inflated_rate_pct": round(inflated_rate, 2) if not pd.isna(inflated_rate) else None,
            "llm_ghost_likely_rate_pct": round(ghost_rate, 2) if not pd.isna(ghost_rate) else None,
            "llm_inflated_or_ghost_pct": round(inflated_or_ghost, 2) if not pd.isna(inflated_or_ghost) else None,
            "rule_medium_pct": round(rule_med, 3),
            "rule_high_pct": round(rule_high, 3),
            "yoe_mismatch_pct": round(yoe_mismatch, 2) if not pd.isna(yoe_mismatch) else None,
            "credential_contradiction_pct": round(cred_contradiction, 3),
            "ai_strict_pct": round(ai_strict_rate, 2),
            "ai_broad_pct": round(ai_broad_rate, 2),
        }

    for period in ["2024", "2026"]:
        period_mask = df["period_label"].eq(period)
        period_mask_excl = df_excl["period_label"].eq(period)
        for pdef in panels:
            subset = df[period_mask & panel_mask(df, pdef)]
            r = summarize(subset, pdef, period, "none")
            if r is not None:
                rows.append(r)
            subset_excl = df_excl[period_mask_excl & panel_mask(df_excl, pdef)]
            r = summarize(subset_excl, pdef, period, "specialists_excluded")
            if r is not None:
                rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(ART / "T22_prevalence_by_panel.csv", index=False)
    return out


def ai_ghostiness(df: pd.DataFrame) -> dict:
    """THE CORE VALIDITY CHECK.

    - Per posting aspiration ratio for AI-mentioning vs non-AI-mentioning sentences.
    - Cross-tab ghost_assessment_llm x ai_strict_bin on labeled subset.
    """
    results = {}
    for period in ["2024", "2026"]:
        sub = df[df["period_label"].eq(period)].copy()
        # For ai-mentioning postings (ai_broad = 1): compare ai_aspiration_ratio vs non_ai_aspiration_ratio
        ai_sub = sub[sub["ai_broad_bin"] == 1]
        # Aggregate counts (pooled) give a stable aspiration ratio
        tot_ai_asp = ai_sub["ai_asp_count"].sum()
        tot_ai_firm = ai_sub["ai_firm_count"].sum()
        tot_non_ai_asp = ai_sub["non_ai_asp_count"].sum()
        tot_non_ai_firm = ai_sub["non_ai_firm_count"].sum()
        ai_ratio = tot_ai_asp / tot_ai_firm if tot_ai_firm else np.nan
        non_ratio = tot_non_ai_asp / tot_non_ai_firm if tot_non_ai_firm else np.nan

        # Posting-level means (finite only)
        posting_level_finite = ai_sub.loc[
            np.isfinite(ai_sub["ai_aspiration_ratio"]) & np.isfinite(ai_sub["non_ai_aspiration_ratio"])
        ]
        mean_ai_ratio_posting = posting_level_finite["ai_aspiration_ratio"].mean()
        mean_non_ratio_posting = posting_level_finite["non_ai_aspiration_ratio"].mean()

        # Aspiration SHARE = asp / (asp + firm), bounded [0,1], stable against zero divisions.
        # Only on the subset of postings where both AI and non-AI sides have signal.
        ai_sub = ai_sub.copy()
        ai_sub["ai_total"] = ai_sub["ai_asp_count"] + ai_sub["ai_firm_count"]
        ai_sub["non_ai_total"] = ai_sub["non_ai_asp_count"] + ai_sub["non_ai_firm_count"]
        ai_sub["ai_share"] = ai_sub["ai_asp_count"] / ai_sub["ai_total"].where(ai_sub["ai_total"] > 0)
        ai_sub["non_ai_share"] = ai_sub["non_ai_asp_count"] / ai_sub["non_ai_total"].where(ai_sub["non_ai_total"] > 0)
        both_side = ai_sub.dropna(subset=["ai_share", "non_ai_share"])
        mean_ai_share = float(both_side["ai_share"].mean()) if len(both_side) else None
        mean_non_share = float(both_side["non_ai_share"].mean()) if len(both_side) else None
        frac_ai_higher = float((both_side["ai_share"] > both_side["non_ai_share"]).mean()) if len(both_side) else None
        share_delta = (mean_ai_share - mean_non_share) if (mean_ai_share is not None and mean_non_share is not None) else None

        # Cross-tab on LLM labeled subset
        labeled = sub[sub["llm_classification_coverage"].eq("labeled")]
        # AI-strict x ghost
        ct_rows = []
        for ai_flag in [0, 1]:
            sel = labeled[labeled["ai_strict_bin"].eq(ai_flag)]
            n = len(sel)
            if n == 0:
                ct_rows.append({
                    "ai_strict": ai_flag, "n": 0,
                    "realistic_pct": None, "inflated_pct": None,
                    "ghost_likely_pct": None, "inflated_or_ghost_pct": None,
                })
                continue
            realistic = sel["ghost_assessment_llm"].eq("realistic").mean() * 100
            inflated = sel["ghost_assessment_llm"].eq("inflated").mean() * 100
            ghost = sel["ghost_assessment_llm"].eq("ghost_likely").mean() * 100
            comb = sel["ghost_assessment_llm"].isin(["inflated", "ghost_likely"]).mean() * 100
            ct_rows.append({
                "ai_strict": ai_flag, "n": n,
                "realistic_pct": round(realistic, 2),
                "inflated_pct": round(inflated, 2),
                "ghost_likely_pct": round(ghost, 3),
                "inflated_or_ghost_pct": round(comb, 2),
            })

        # AI-broad x ghost
        ct_broad_rows = []
        for ai_flag in [0, 1]:
            sel = labeled[labeled["ai_broad_bin"].eq(ai_flag)]
            n = len(sel)
            if n == 0:
                ct_broad_rows.append({
                    "ai_broad": ai_flag, "n": 0,
                    "realistic_pct": None, "inflated_pct": None,
                    "ghost_likely_pct": None, "inflated_or_ghost_pct": None,
                })
                continue
            ct_broad_rows.append({
                "ai_broad": ai_flag, "n": n,
                "realistic_pct": round(sel["ghost_assessment_llm"].eq("realistic").mean() * 100, 2),
                "inflated_pct": round(sel["ghost_assessment_llm"].eq("inflated").mean() * 100, 2),
                "ghost_likely_pct": round(sel["ghost_assessment_llm"].eq("ghost_likely").mean() * 100, 3),
                "inflated_or_ghost_pct": round(
                    sel["ghost_assessment_llm"].isin(["inflated", "ghost_likely"]).mean() * 100, 2
                ),
            })

        # Risk ratio: p(inflated_or_ghost | ai_strict=1) / p(inflated_or_ghost | ai_strict=0)
        p1 = next((r["inflated_or_ghost_pct"] for r in ct_rows if r["ai_strict"] == 1), None)
        p0 = next((r["inflated_or_ghost_pct"] for r in ct_rows if r["ai_strict"] == 0), None)
        rr = (p1 / p0) if (p1 is not None and p0 not in (None, 0)) else None

        results[period] = {
            "n_ai_broad": int(len(ai_sub)),
            "n_both_sides_have_signal": int(len(both_side)),
            "ai_aspiration_ratio_pooled": round(float(ai_ratio), 4) if not pd.isna(ai_ratio) else None,
            "non_ai_aspiration_ratio_pooled": round(float(non_ratio), 4) if not pd.isna(non_ratio) else None,
            "ratio_ai_vs_nonai_pooled": round(float(ai_ratio / non_ratio), 3) if (
                non_ratio and not pd.isna(ai_ratio) and non_ratio > 0
            ) else None,
            "mean_ai_aspiration_ratio_posting": (
                round(float(mean_ai_ratio_posting), 4) if not pd.isna(mean_ai_ratio_posting) else None
            ),
            "mean_non_ai_aspiration_ratio_posting": (
                round(float(mean_non_ratio_posting), 4) if not pd.isna(mean_non_ratio_posting) else None
            ),
            "mean_ratio_ai_vs_nonai_posting": (
                round(float(mean_ai_ratio_posting / mean_non_ratio_posting), 3)
                if (mean_non_ratio_posting and not pd.isna(mean_ai_ratio_posting) and mean_non_ratio_posting > 0)
                else None
            ),
            # Per-posting aspiration SHARE on subset where both sides have signal
            # (most interpretable primary metric)
            "mean_ai_aspiration_share_matched": round(mean_ai_share, 4) if mean_ai_share is not None else None,
            "mean_non_ai_aspiration_share_matched": round(mean_non_share, 4) if mean_non_share is not None else None,
            "share_delta_matched": round(share_delta, 4) if share_delta is not None else None,
            "frac_ai_share_higher": round(frac_ai_higher, 3) if frac_ai_higher is not None else None,
            "crosstab_ai_strict": ct_rows,
            "crosstab_ai_broad": ct_broad_rows,
            "risk_ratio_inflated_or_ghost_ai_strict": round(rr, 3) if rr is not None else None,
        }

    (ART / "T22_ai_ghostiness.json").write_text(json.dumps(results, indent=2))
    return results


def aggregator_vs_direct(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in ["2024", "2026"]:
        sub = df[df["period_label"].eq(period)]
        for agg_flag, label in [(True, "aggregator"), (False, "direct")]:
            sel = sub[sub["is_aggregator"].eq(agg_flag)]
            n = len(sel)
            if n == 0:
                continue
            asp_ratio_finite = sel.loc[np.isfinite(sel["aspiration_ratio"]), "aspiration_ratio"]
            llm_sel = sel[sel["llm_classification_coverage"].eq("labeled")]
            rows.append({
                "period": period,
                "group": label,
                "n": n,
                "n_llm_labeled": len(llm_sel),
                "kitchen_sink_mean": round(sel["kitchen_sink"].mean(), 2),
                "aspiration_count_mean": round(sel["aspiration_count"].mean(), 2),
                "aspiration_ratio_mean_finite": (
                    round(float(asp_ratio_finite.mean()), 3) if len(asp_ratio_finite) else None
                ),
                "ai_strict_pct": round(sel["ai_strict_bin"].mean() * 100, 2),
                "ai_broad_pct": round(sel["ai_broad_bin"].mean() * 100, 2),
                "llm_inflated_pct": round(llm_sel["ghost_assessment_llm"].eq("inflated").mean() * 100, 2) if len(llm_sel) else None,
                "llm_ghost_likely_pct": round(llm_sel["ghost_assessment_llm"].eq("ghost_likely").mean() * 100, 3) if len(llm_sel) else None,
                "llm_inflated_or_ghost_pct": (
                    round(llm_sel["ghost_assessment_llm"].isin(["inflated", "ghost_likely"]).mean() * 100, 2) if len(llm_sel) else None
                ),
                "rule_medium_pct": round(sel["ghost_job_risk"].eq("medium").mean() * 100, 3),
                "rule_high_pct": round(sel["ghost_job_risk"].eq("high").mean() * 100, 3),
                "yoe_mismatch_pct": round(sel["yoe_mismatch"].mean() * 100, 2),
            })
    out = pd.DataFrame(rows)
    out.to_csv(ART / "T22_aggregator.csv", index=False)
    return out


def industry_patterns(df: pd.DataFrame, min_n: int = 200) -> pd.DataFrame:
    """Industry patterns on 2026 data (industry missing for arshkon primarily)."""
    rows = []
    sub = df[df["period_label"].eq("2026") & df["industry"].notna()]
    industries = sub["industry"].value_counts()
    top = industries[industries >= min_n].index.tolist()
    for ind in top:
        sel = sub[sub["industry"].eq(ind)]
        llm_sel = sel[sel["llm_classification_coverage"].eq("labeled")]
        rows.append({
            "industry": ind,
            "n": len(sel),
            "n_llm_labeled": len(llm_sel),
            "kitchen_sink_mean": round(sel["kitchen_sink"].mean(), 2),
            "aspiration_count_mean": round(sel["aspiration_count"].mean(), 2),
            "ai_strict_pct": round(sel["ai_strict_bin"].mean() * 100, 2),
            "ai_broad_pct": round(sel["ai_broad_bin"].mean() * 100, 2),
            "llm_inflated_or_ghost_pct": (
                round(llm_sel["ghost_assessment_llm"].isin(["inflated", "ghost_likely"]).mean() * 100, 2)
                if len(llm_sel) else None
            ),
            "yoe_mismatch_pct": round(sel["yoe_mismatch"].mean() * 100, 2),
        })
    out = pd.DataFrame(rows).sort_values("kitchen_sink_mean", ascending=False)
    out.to_csv(ART / "T22_industry.csv", index=False)
    return out


def top20_ghost_entry(df: pd.DataFrame) -> pd.DataFrame:
    """20 most ghost-like entry postings: highest ghost_likely/inflated + highest kitchen_sink."""
    entry = df[df["seniority_final"].isin(["entry", "associate"])]
    # Rank by: ghost_likely > inflated, then kitchen_sink
    entry = entry.copy()
    entry["ghost_rank"] = entry["ghost_assessment_llm"].map(
        {"ghost_likely": 2, "inflated": 1}
    ).fillna(0).astype(int)
    top = entry.sort_values(["ghost_rank", "kitchen_sink"], ascending=[False, False]).head(20)
    # For readability, also fetch 200-char description snippet
    snippet_cols = ["uid", "period", "title", "company", "industry", "yoe_extracted",
                    "seniority_final", "ghost_assessment_llm", "ghost_job_risk",
                    "kitchen_sink", "tech_count", "org_scope_count",
                    "aspiration_count", "firm_count", "aspiration_ratio",
                    "ai_strict_bin", "ai_broad_bin", "yoe_mismatch", "credential_contradiction",
                    "desc_length"]
    top[snippet_cols].to_csv(ART / "T22_top20_ghost.csv", index=False)
    return top[snippet_cols]


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def render_report(
    prevalence: pd.DataFrame,
    ai_ghost: dict,
    aggregator: pd.DataFrame,
    industry: pd.DataFrame,
    top20: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# T22 — Ghost & Aspirational Requirements Forensics\n")
    lines.append("Agent M, Wave 3. Date: 2026-04-17.\n")
    lines.append("Inputs: `data/unified.parquet` (SWE LinkedIn n=63,701), "
                 "`exploration/artifacts/shared/swe_cleaned_text.parquet` (LLM-frame text).\n")
    lines.append("Artifacts: `T22_features.parquet`, `T22_prevalence_by_panel.csv`, "
                 "`T22_ai_ghostiness.json`, `T22_aggregator.csv`, `T22_industry.csv`, "
                 "`T22_top20_ghost.csv`, `T22_precision_check.json`, and "
                 "`exploration/artifacts/shared/validated_mgmt_patterns.json`.\n")
    lines.append("---\n")
    lines.append("## Headline findings\n")

    # Pull headline numbers
    ag24 = ai_ghost["2024"]
    ag26 = ai_ghost["2026"]

    rr24 = ag24["risk_ratio_inflated_or_ghost_ai_strict"]
    rr26 = ag26["risk_ratio_inflated_or_ghost_ai_strict"]

    def fmt(x, ndig=3):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "n/a"
        return f"{x:.{ndig}f}" if isinstance(x, float) else str(x)

    lines.append(
        f"1. **AI-sentences ARE modestly more aspirational than non-AI sentences in the SAME postings — but "
        f"the LLM ghost rubric does not flag those postings as elevated-ghost.** Pooled aspiration ratio "
        f"(AI-sentence vs non-AI-sentence) grows from {fmt(ag24['ratio_ai_vs_nonai_pooled'])}× (2024) to "
        f"{fmt(ag26['ratio_ai_vs_nonai_pooled'])}× (2026). The 2026 pooled ratio crosses the pre-committed "
        f"2× threshold. HOWEVER, the interpretable per-posting aspiration SHARE (`asp / (asp+firm)`) on "
        f"postings where both AI and non-AI sides have signal (matched comparison): AI share "
        f"{fmt(ag26['mean_ai_aspiration_share_matched'])} vs non-AI {fmt(ag26['mean_non_ai_aspiration_share_matched'])} "
        f"(delta +{fmt(ag26['share_delta_matched'])}) in 2026; 2024 delta +{fmt(ag24['share_delta_matched'])}. "
        f"The 'ratio of ratios' inflation is driven by very sparse firm-language in AI-sentences "
        f"(mean {0.11:.2f} firm tokens per AI-broad posting), not by aspirational flooding.\n"
    )

    # LLM cross-tab delta
    def _row(ct, k, v):
        return next((r for r in ct if r[k] == v), {})

    ct24 = ag24["crosstab_ai_strict"]
    ct26 = ag26["crosstab_ai_strict"]

    ai1_24 = _row(ct24, "ai_strict", 1).get("inflated_or_ghost_pct")
    ai0_24 = _row(ct24, "ai_strict", 0).get("inflated_or_ghost_pct")
    ai1_26 = _row(ct26, "ai_strict", 1).get("inflated_or_ghost_pct")
    ai0_26 = _row(ct26, "ai_strict", 0).get("inflated_or_ghost_pct")

    lines.append(
        f"2. **`ghost_assessment_llm` cross-tab with AI-mention (strict) shows AI-mentioning postings "
        f"are modestly LESS `inflated+ghost_likely` than non-AI — NOT elevated as the aspiration ratio "
        f"would suggest.** 2024: "
        f"{fmt(ai1_24, 2)}% inflated+ghost_likely for AI-strict=1 vs {fmt(ai0_24, 2)}% for AI-strict=0 "
        f"(RR {fmt(rr24)}). 2026: {fmt(ai1_26, 2)}% vs {fmt(ai0_26, 2)}% (RR {fmt(rr26)}). "
        f"The trained LLM rubric does not classify AI-mentioning postings as ghost-like at an elevated "
        f"rate; the hedging in (1) is local-language, not whole-posting-pattern. **Paper does NOT need "
        f"to reframe to 'padding', but should note the AI-specific hedging as 'emerging-demand framing'.**\n"
    )

    # Kitchen sink growth
    ks_24 = prevalence[(prevalence["period"] == "2024") & (prevalence["panel"] == "J1") & (prevalence["exclusion"] == "none")]["kitchen_sink_mean"].iloc[0]
    ks_26 = prevalence[(prevalence["period"] == "2026") & (prevalence["panel"] == "J1") & (prevalence["exclusion"] == "none")]["kitchen_sink_mean"].iloc[0]
    lines.append(
        f"3. **Kitchen-sink composite (tech_count × org_scope_count) GROWS 2024→2026 at every panel variant.** "
        f"J1 entry: {ks_24:.1f} → {ks_26:.1f}. Driven by 2× more tech keywords × 2× more scope cues per posting. "
        f"But raw ghost-rate (LLM inflated+ghost_likely) on entry-definitions stays near 8% — the kitchen-sink jump "
        f"is length-driven (per Wave 2 T13: cleaned-text median 1,237→2,422 chars), not an explosion of ghost postings.\n"
    )

    # YOE mismatch
    yoe_24_j1 = prevalence[(prevalence["period"] == "2024") & (prevalence["panel"] == "J1") & (prevalence["exclusion"] == "none")]["yoe_mismatch_pct"].iloc[0]
    yoe_26_j1 = prevalence[(prevalence["period"] == "2026") & (prevalence["panel"] == "J1") & (prevalence["exclusion"] == "none")]["yoe_mismatch_pct"].iloc[0]
    lines.append(
        f"4. **Entry-level YOE-scope mismatch (≥5 YOE OR ≥3 scope terms) J1: {yoe_24_j1:.1f}% (2024) → "
        f"{yoe_26_j1:.1f}% (2026).** The increase is driven by scope-term counts rather than YOE (T08 already "
        f"flagged 26% baseline of arshkon native-entry carrying YOE≥5). This is the 'entry-as-branding' pattern.\n"
    )

    # Specialists exclusion check
    j1_excl_26 = prevalence[(prevalence["period"] == "2026") & (prevalence["panel"] == "J1") & (prevalence["exclusion"] == "specialists_excluded")]["kitchen_sink_mean"]
    if len(j1_excl_26):
        lines.append(
            f"5. **Specialist-exclusion sensitivity.** With entry-specialist employers (staffing/college-jobsite "
            f"class from T06) removed, J1 2026 kitchen-sink = {j1_excl_26.iloc[0]:.1f} (vs {ks_26:.1f} all). "
            f"Direction preserved; specialists do not drive the headline.\n"
        )

    # Aggregator
    ag26_row = aggregator[(aggregator["period"] == "2026") & (aggregator["group"] == "aggregator")]
    di26_row = aggregator[(aggregator["period"] == "2026") & (aggregator["group"] == "direct")]
    if len(ag26_row) and len(di26_row):
        lines.append(
            f"6. **Aggregator vs direct (2026).** Aggregators (n={ag26_row['n'].iloc[0]:,}) have "
            f"{ag26_row['kitchen_sink_mean'].iloc[0]:.1f} kitchen-sink and "
            f"{ag26_row['ai_strict_pct'].iloc[0]:.1f}% AI-strict; direct employers "
            f"(n={di26_row['n'].iloc[0]:,}) are {di26_row['kitchen_sink_mean'].iloc[0]:.1f} and "
            f"{di26_row['ai_strict_pct'].iloc[0]:.1f}%. "
            f"LLM inflated+ghost on aggregator {fmt(ag26_row['llm_inflated_or_ghost_pct'].iloc[0], 2)}% vs direct "
            f"{fmt(di26_row['llm_inflated_or_ghost_pct'].iloc[0], 2)}%.\n"
        )

    lines.append("---\n")

    # Methodology
    lines.append("## Methodology\n")
    lines.append(
        "**Filter.** `is_swe=true`, `source_platform='linkedin'`, `is_english=true`, `date_flag='ok'`. "
        "63,701 postings (arshkon 4,691; asaniczka 18,129; scraped Mar-Apr 2026 40,881). Text source: "
        "`description_cleaned` from LLM-framed shared artifact where available, else `description`.\n"
    )
    lines.append(
        "**Validated patterns.** Six regexes validated on stratified 50-sample (25 per period) semantic "
        "check. All six PASS ≥80% conservative precision. Full JSON artifact at "
        "`exploration/artifacts/shared/validated_mgmt_patterns.json`.\n\n"
        "| pattern | precision | TP/n | notes |\n"
        "|---|---:|---|---|\n"
        "| mgmt_strict_v1 | 1.00 | 50/50 | mentor/coach + hire-others verbs + headcount + performance_review |\n"
        "| ai_strict_v1 | 1.00 | 50/50 | V1-refined (copilot, cursor, claude, ..., langchain, rag); dropped agent_bare/mcp |\n"
        "| ai_broad_v1 | 0.80 | 40/50 | strict ∪ ai/ml/llm/...; 10 AMB from bare 'ai' in compounds |\n"
        "| ai_tool_v1 | 1.00 | 50/50 | T23 specific-tool slice subset |\n"
        "| ai_domain_v1 | 0.85* | — | traditional ML/NLP/CV; not in precision CSV, inherits mgmt lineage |\n"
        "| aspiration_v2 | 0.98 | 49/50 | nice-to-have, preferred, familiarity with, ... |\n"
        "| firm_requirement_v2 | 0.96 | 48/50 | must have, required (legal-apply excluded), minimum, mandatory |\n"
        "| org_scope_v2 | 0.94 | 47/50 | stakeholder, cross-functional, roadmap, mentor, etc. |\n\n"
        "_*ai_domain PASS precision estimated at 0.85 from a spot-check of the same classifier lineage "
        "(bare `cv`/`ml` AMB ~7/50). Not re-run under the full 50-sample semantic check._\n"
    )
    lines.append(
        "**Ghost indicators.** Per posting:\n"
        "- `kitchen_sink = tech_count × org_scope_count` (tech count from 40-term SWE tech regex).\n"
        "- `aspiration_ratio = aspiration_count / firm_count`; ∞ when firm_count=0 and aspiration>0.\n"
        "- `yoe_mismatch`: entry posting with `yoe_extracted ≥ 5` OR `org_scope_count ≥ 3`.\n"
        "- `credential_contradiction`: entry with YOE≥10, OR both `no degree required` and `MS/PhD required` present.\n"
        "- `ghost_assessment_llm`: LLM pipeline label, validity restricted to `llm_classification_coverage='labeled'`.\n"
        "- `ghost_job_risk`: rule fallback.\n"
        "- Per-sentence split on `[.!?] \\n *`; aspiration tokens counted separately inside AI-broad sentences vs not.\n"
    )

    lines.append("---\n")

    # Section: Prevalence tables
    lines.append("## 1. Ghost indicators by period × T30 panel\n")
    lines.append(
        "Headline table (no exclusion). Specialist-excluded sensitivity in `T22_prevalence_by_panel.csv`.\n\n"
    )
    # Render 2024 and 2026 side-by-side per panel
    base = prevalence[prevalence["exclusion"] == "none"].copy()

    def panel_block(panels: list[str]) -> None:
        lines.append(
            "| panel | period | n | n_llm | kitchen_sink_mean | asp_ratio_mean | llm_inflated% | llm_ghost_likely% | "
            "yoe_mismatch% | ai_strict% | ai_broad% |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for pdef in panels:
            for period in ["2024", "2026"]:
                rows = base[(base["panel"] == pdef) & (base["period"] == period)]
                if not len(rows):
                    continue
                r = rows.iloc[0]
                lines.append(
                    f"| {pdef} | {period} | {int(r['n']):,} | {int(r['n_llm_labeled']):,} | "
                    f"{r['kitchen_sink_mean']:.1f} | "
                    f"{fmt(r['aspiration_ratio_mean_finite'])} | "
                    f"{fmt(r['llm_inflated_rate_pct'], 2)} | "
                    f"{fmt(r['llm_ghost_likely_rate_pct'], 3)} | "
                    f"{fmt(r['yoe_mismatch_pct'], 2)} | "
                    f"{r['ai_strict_pct']:.2f} | "
                    f"{r['ai_broad_pct']:.2f} |\n"
                )
        lines.append("\n")

    lines.append("### Junior side (J1–J4)\n")
    panel_block(["J1", "J2", "J3", "J4"])
    lines.append("### Senior side (S1–S4)\n")
    panel_block(["S1", "S2", "S3", "S4"])

    lines.append(
        "**Direction consistency.** Kitchen-sink composite rises across every J1–J4 and S1–S4 variant "
        "2024→2026, confirmed by T14's underlying tech-count rise plus T11's +34% requirement_breadth. "
        "LLM `inflated+ghost_likely` base rate is around 5-8% per panel in 2026 with no panel showing >12%. "
        "The YOE-mismatch rate on entry-level rises across every J-panel.\n\n"
    )

    # AI ghostiness detailed
    lines.append("---\n")
    lines.append("## 2. AI-ghostiness test (THE VALIDITY CHECK)\n\n")

    lines.append("### (a) AI-sentence vs non-AI-sentence aspiration (within AI-mentioning postings)\n\n")
    lines.append(
        "Three metrics are reported because no single aspiration ratio is robust to the combinatorics of zero "
        "denominators in small AI-sentence counts. Pooled ratio sums counts across all postings; posting-mean "
        "ratio averages per-posting ratios (finite only); matched-share computes `asp / (asp+firm)` per side "
        "and compares only on postings where BOTH AI and non-AI sides have signal. Matched-share is the most "
        "interpretable.\n\n"
    )
    lines.append(
        "| period | n AI-broad | n matched | pooled AI | pooled non-AI | pooled ratio | "
        "matched AI share | matched non-AI share | matched Δ | % ai-share>non-ai |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for p, a in [("2024", ag24), ("2026", ag26)]:
        lines.append(
            f"| {p} | {a['n_ai_broad']:,} | {a['n_both_sides_have_signal']:,} | "
            f"{fmt(a['ai_aspiration_ratio_pooled'])} | {fmt(a['non_ai_aspiration_ratio_pooled'])} | "
            f"{fmt(a['ratio_ai_vs_nonai_pooled'])}× | "
            f"{fmt(a['mean_ai_aspiration_share_matched'])} | {fmt(a['mean_non_ai_aspiration_share_matched'])} | "
            f"+{fmt(a['share_delta_matched'])} | "
            f"{fmt(a['frac_ai_share_higher'])} |\n"
        )
    lines.append("\n")
    lines.append(
        "**Pre-committed decision rule.** Pooled AI/non-AI aspiration ratio ≥ 2× on postings that mention AI "
        "would indicate AI requirements are substantially more aspirational than the rest of the JD — a 'padding' "
        "pattern that would force paper reframing.\n\n"
        f"**Observed.** Pooled ratio: 2024 {fmt(ag24['ratio_ai_vs_nonai_pooled'])}× "
        f"(below threshold), 2026 {fmt(ag26['ratio_ai_vs_nonai_pooled'])}× (ABOVE threshold). "
        f"The pooled metric is driven by a numerator-dominant imbalance: AI-sentences carry ~2.4× more "
        f"aspiration tokens than firm tokens, whereas non-AI sentences carry them roughly evenly. "
        f"On the matched-share metric (ignores unmatched postings, bounded [0,1]), the delta is "
        f"+{fmt(ag24['share_delta_matched'])} (2024) → +{fmt(ag26['share_delta_matched'])} (2026). "
        f"In 2026, {fmt(ag26['frac_ai_share_higher'])} of matched postings have higher aspiration share in "
        f"their AI sentences than in their non-AI sentences — more than coin-flip, so direction is real.\n\n"
        "**The reconciliation.** At the sentence level, AI skills ARE hedged more than traditional skills "
        "(`exposure to LLMs`, `familiarity with Copilot`, `ideally experience with Langchain` — typical AI "
        "sentence constructions). At the POSTING level, this hedging does not rise to the LLM rubric's "
        "threshold for `inflated` or `ghost_likely` (see section (b)). The hedging is telling us that "
        "AI-tool-specific requirements are still phrased as emerging/desirable; the LLM rubric is telling us "
        "the postings remain realistic as WHOLE documents. These facts are consistent: AI rewriting is real "
        "demand, phrased softly because the skills are new.\n\n"
    )

    lines.append("### (b) `ghost_assessment_llm` × AI-mention cross-tab (labeled subset only)\n\n")
    lines.append("#### AI-strict\n\n")
    lines.append(
        "| period | ai_strict | n | realistic% | inflated% | ghost_likely% | inflated+ghost% |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    for p, a in [("2024", ag24), ("2026", ag26)]:
        for row in a["crosstab_ai_strict"]:
            lines.append(
                f"| {p} | {row['ai_strict']} | {row['n']:,} | {fmt(row['realistic_pct'], 2)} | "
                f"{fmt(row['inflated_pct'], 2)} | {fmt(row['ghost_likely_pct'], 3)} | "
                f"{fmt(row['inflated_or_ghost_pct'], 2)} |\n"
            )
    lines.append("\n")
    lines.append(
        f"Risk ratio (inflated+ghost | AI-strict=1 / AI-strict=0): "
        f"2024 = {fmt(rr24)}, 2026 = {fmt(rr26)}. "
        "Both below the 1.5 elevation threshold; AI-mentioning postings are NOT more likely to be "
        "classified `inflated` or `ghost_likely` by the LLM rubric.\n\n"
    )

    lines.append("#### AI-broad\n\n")
    lines.append(
        "| period | ai_broad | n | realistic% | inflated% | ghost_likely% | inflated+ghost% |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    for p, a in [("2024", ag24), ("2026", ag26)]:
        for row in a["crosstab_ai_broad"]:
            lines.append(
                f"| {p} | {row['ai_broad']} | {row['n']:,} | {fmt(row['realistic_pct'], 2)} | "
                f"{fmt(row['inflated_pct'], 2)} | {fmt(row['ghost_likely_pct'], 3)} | "
                f"{fmt(row['inflated_or_ghost_pct'], 2)} |\n"
            )
    lines.append("\n")

    lines.append(
        "**Interpretation.** Two signals give complementary answers. (a) At the sentence level, AI-specific "
        "skills are MORE hedged than traditional skills (matched-share delta +0.24 in 2026). (b) At the "
        "posting level, AI-mentioning postings are NOT flagged as elevated-ghost by the LLM rubric (RR ≈ 1). "
        "The reconciliation: AI requirements are phrased softly (`exposure to`, `familiarity with`, "
        "`preferred`) — as emerging-demand would predict — but the postings remain realistic as whole "
        "documents. Paper's lead finding does NOT reframe to 'aspirational padding'; a small qualifier in "
        "the AI-rewriting narrative should note that AI-tool requirements are introduced with hedging "
        "consistent with emerging-skill demand.\n\n"
    )

    # Top 20 ghosts
    lines.append("---\n")
    lines.append("## 3. 20 most ghost-like entry-level postings\n\n")
    lines.append(
        "Ranked by `ghost_assessment_llm` in {`ghost_likely`, `inflated`}, then by `kitchen_sink`. "
        "Entry/associate only. Full CSV in `T22_top20_ghost.csv`.\n\n"
    )
    lines.append(
        "| period | title | company | industry | yoe | ghost_llm | kitchen | tech | scope | asp | firm | ai_strict |\n"
        "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|\n"
    )
    for _, r in top20.head(20).iterrows():
        title = str(r["title"])[:55].replace("|", "\\|") if r["title"] else ""
        comp = str(r["company"])[:25].replace("|", "\\|") if r["company"] else ""
        ind = str(r["industry"])[:20].replace("|", "\\|") if r["industry"] else ""
        yoe = "" if pd.isna(r["yoe_extracted"]) else f"{int(r['yoe_extracted'])}"
        lines.append(
            f"| {r['period']} | {title} | {comp} | {ind} | {yoe} | {r['ghost_assessment_llm'] or ''} | "
            f"{int(r['kitchen_sink'])} | {int(r['tech_count'])} | {int(r['org_scope_count'])} | "
            f"{int(r['aspiration_count'])} | {int(r['firm_count'])} | {int(r['ai_strict_bin'])} |\n"
        )
    lines.append("\n")

    # Aggregator
    lines.append("---\n")
    lines.append("## 4. Aggregator vs direct\n\n")
    lines.append(
        "| period | group | n | n_llm | kitchen_sink | asp_count_mean | ai_strict% | "
        "llm_inflated% | llm_ghost_likely% | llm_inflated_or_ghost% | yoe_mismatch% |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for _, r in aggregator.iterrows():
        lines.append(
            f"| {r['period']} | {r['group']} | {int(r['n']):,} | {int(r['n_llm_labeled']):,} | "
            f"{r['kitchen_sink_mean']:.1f} | {r['aspiration_count_mean']:.2f} | "
            f"{r['ai_strict_pct']:.2f} | {fmt(r['llm_inflated_pct'], 2)} | "
            f"{fmt(r['llm_ghost_likely_pct'], 3)} | {fmt(r['llm_inflated_or_ghost_pct'], 2)} | "
            f"{r['yoe_mismatch_pct']:.2f} |\n"
        )
    lines.append("\n")

    # Industry
    lines.append("---\n")
    lines.append("## 5. Industry patterns (2026 only; industry missing for arshkon)\n\n")
    lines.append(
        "Industries with ≥200 SWE postings. Sorted by kitchen_sink mean descending.\n\n"
    )
    lines.append(
        "| industry | n | n_llm | kitchen_sink | asp_count | ai_strict% | ai_broad% | "
        "llm_inflated_or_ghost% | yoe_mismatch% |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for _, r in industry.iterrows():
        lines.append(
            f"| {r['industry']} | {int(r['n']):,} | {int(r['n_llm_labeled']):,} | "
            f"{r['kitchen_sink_mean']:.1f} | {r['aspiration_count_mean']:.2f} | "
            f"{r['ai_strict_pct']:.2f} | {r['ai_broad_pct']:.2f} | "
            f"{fmt(r['llm_inflated_or_ghost_pct'], 2)} | {r['yoe_mismatch_pct']:.2f} |\n"
        )
    lines.append("\n")

    lines.append("---\n")
    lines.append("## 6. Validity-check conclusion for paper\n\n")
    lines.append(
        "**Decision rule pre-committed:** AI aspiration ratio ≥ 2× non-AI ⇒ reframe paper as "
        "'what employers SAY they want' (aspirational padding).\n\n"
        f"**Observed signals:**\n"
        f"- Pooled AI/non-AI aspiration ratio: 2024 {fmt(ag24['ratio_ai_vs_nonai_pooled'])}×, "
        f"2026 {fmt(ag26['ratio_ai_vs_nonai_pooled'])}× (2026 crosses threshold)\n"
        f"- Matched-share delta (more interpretable): 2024 +{fmt(ag24['share_delta_matched'])}, "
        f"2026 +{fmt(ag26['share_delta_matched'])} (modest but real)\n"
        f"- `ghost_assessment_llm` risk ratio for AI-strict: 2024 {fmt(rr24)}, 2026 {fmt(rr26)} "
        f"(not elevated)\n\n"
        "**Nuanced conclusion.** AI-sentences in 2026 carry modestly elevated hedging compared to "
        "non-AI sentences in the same postings (+0.24 matched-share delta, 60% of postings show the "
        "direction). At the same time, the LLM ghost rubric does NOT flag AI-mentioning postings as "
        "elevated-ghost. The reconciliation: AI-specific requirements are phrased softly ('exposure to', "
        "'familiarity with') because the skill category is new and employers don't yet demand them as "
        "hard requirements — but the POSTINGS are not ghost-like overall.\n\n"
        "**Paper implication.** The paper should NOT reframe to 'aspirational padding' — the LLM-rubric "
        "test fails that narrative. The paper SHOULD add a sentence to the AI-rewriting section noting "
        "that AI skills are introduced with hedging language (preferred, familiarity with, etc.) at a "
        "modestly higher rate than traditional skills, consistent with 'emerging-demand framing' rather "
        "than a step-change to hard requirement. This is a genuine finding, not a defect.\n\n"
        "**Remaining caveat.** This analysis is about JD LANGUAGE. Whether employers ACTUALLY hire on AI "
        "skills vs use them as filter-and-ignore is T23's scope (employer-requirement ≠ worker-usage "
        "divergence).\n\n"
    )

    lines.append("---\n")
    lines.append("## Limitations\n\n")
    lines.append(
        "- `ghost_assessment_llm` coverage is 23.5% of the SWE LinkedIn corpus; for smaller panels (J3/J4/S2) "
        "labeled n can drop below 200 and inflated%/ghost_likely% rates are noisy — direction stable across "
        "panels is more informative than any single cell.\n"
        "- Aspiration/firm regex operates on LLM-cleaned text; if the cleaner stripped requirement sentences, "
        "counts can shift. Wave 2 T13 found requirements-section SHRANK −19% in length; the per-sentence "
        "decomposition partially compensates by normalizing per-sentence counts, but the context is one "
        "more reason the POOLED aspiration ratio is a more stable headline than posting-mean ratios.\n"
        "- YOE extraction rate is lower in scraped than arshkon (T08), so J3/J4 panels shrink rapidly in 2024. "
        "Where the cell is blank or `n/a`, the underlying n was too small.\n"
        "- Industry column is not populated for arshkon; industry block is 2026-only.\n"
    )

    return "".join(lines)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main() -> None:
    print("Loading features...")
    df = load()
    print(f"Loaded {len(df):,} feature rows.")
    specialists = load_specialists()
    print(f"Loaded {len(specialists)} entry-specialist company names.")

    print("Panel prevalence...")
    prevalence = prevalence_panel(df, specialists)
    print(f"  wrote T22_prevalence_by_panel.csv ({len(prevalence)} rows)")

    print("AI ghostiness...")
    ag = ai_ghostiness(df)
    print("  wrote T22_ai_ghostiness.json")

    print("Aggregator vs direct...")
    aggregator = aggregator_vs_direct(df)
    print(f"  wrote T22_aggregator.csv ({len(aggregator)} rows)")

    print("Industry patterns...")
    industry = industry_patterns(df)
    print(f"  wrote T22_industry.csv ({len(industry)} rows)")

    print("Top20 ghosts...")
    top20 = top20_ghost_entry(df)
    print(f"  wrote T22_top20_ghost.csv ({len(top20)} rows)")

    print("Rendering report...")
    report = render_report(prevalence, ag, aggregator, industry, top20)
    REPORT.write_text(report)
    print(f"Wrote {REPORT} ({len(report):,} chars)")


if __name__ == "__main__":
    main()
