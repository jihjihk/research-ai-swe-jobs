"""Wave 1.5 Agent Prep - Step 6: Within-2024 calibration table.

For each metric:
 - arshkon_value, asaniczka_value, scraped_value (pooled over the SWE LinkedIn frame)
 - within_2024_effect: |arshkon - asaniczka| effect size (Cohen's d / proportion diff / mean diff)
 - cross_period_effect: |pooled_2024 - scraped| effect size
 - calibration_ratio = cross_period_effect / within_2024_effect
 - snr_flag = above_noise if ratio ≥ 2, near_noise if 1-2, below_noise if < 1
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

OUT_DIR = Path("exploration/artifacts/shared")
OUT_PATH = OUT_DIR / "calibration_table.csv"
TEXT_PATH = OUT_DIR / "swe_cleaned_text.parquet"
MATRIX_PATH = OUT_DIR / "swe_tech_matrix.parquet"

# Pre-defined AI / management / soft skill / scope / education patterns.
# We compile them with IGNORECASE. Apply the same escape fix on text.
ESCAPE_RE = re.compile(r"\\([+\-#.&_()\[\]\{\}!*])")

AI_STRICT_RE = re.compile(
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|vector database|pinecone|huggingface|hugging face)\b",
    re.IGNORECASE,
)
AI_BROAD_EXTRA_RE = re.compile(
    r"\b(agent|machine learning|ml|ai|llm|artificial intelligence|mcp)\b",
    re.IGNORECASE,
)

MGMT_STRICT_RE = re.compile(
    r"\b(mentor|coach|hire|headcount|performance review|direct reports?)\b",
    re.IGNORECASE,
)
MGMT_BROAD_EXTRA_RE = re.compile(
    r"\b(lead|team|stakeholder|coordinate|manage)\b",
    re.IGNORECASE,
)

SOFT_RE = re.compile(
    r"\b(collaborat(?:ive|ion)|communication|teamwork|problem[- ]solving|interpersonal|leadership)\b",
    re.IGNORECASE,
)

SCOPE_RE = re.compile(
    r"\b(ownership|end[- ]to[- ]end|cross[- ]functional|autonomous|initiative|stakeholder)\b",
    re.IGNORECASE,
)

EDU_PHD_RE = re.compile(
    r"\b(ph\.?d|doctorate|doctoral)\b",
    re.IGNORECASE,
)
EDU_MS_RE = re.compile(
    r"\b(master(?:'s)?(?:\s+degree)?|m\.?s\.|ms\s+degree|m\.?sc|masters?)\b",
    re.IGNORECASE,
)
EDU_BS_FLOOR_RE = re.compile(
    r"\b(bachelor(?:'s)?(?:\s+degree)?|b\.?s\.|bs\s+degree|b\.?sc|undergraduate\s+degree)\b",
    re.IGNORECASE,
)


def compute_text_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-row boolean text metrics on RAW description.
    Uses raw description from unified.parquet joined on uid.
    """
    # Fetch raw descriptions and lengths via DuckDB
    con = duckdb.connect()
    raw = con.execute(
        """
        SELECT uid, description
        FROM 'data/unified.parquet'
        WHERE is_swe AND source_platform='linkedin'
          AND is_english = true AND date_flag='ok'
        """
    ).df()
    merged = df.merge(raw, on="uid", how="left")

    # Apply escape fix
    print("[step6]  applying escape fix + running regex metrics")
    n = len(merged)
    ai_strict = np.zeros(n, dtype=bool)
    ai_broad = np.zeros(n, dtype=bool)
    mgmt_strict = np.zeros(n, dtype=bool)
    mgmt_broad = np.zeros(n, dtype=bool)
    soft_any = np.zeros(n, dtype=bool)
    soft_count = np.zeros(n, dtype=int)
    scope_any = np.zeros(n, dtype=bool)
    edu_phd = np.zeros(n, dtype=bool)
    edu_ms = np.zeros(n, dtype=bool)
    edu_bs = np.zeros(n, dtype=bool)
    descs = merged["description"].values
    for i in range(n):
        d = descs[i]
        if not isinstance(d, str):
            continue
        t = ESCAPE_RE.sub(r"\1", d)
        if AI_STRICT_RE.search(t):
            ai_strict[i] = True
            ai_broad[i] = True
        elif AI_BROAD_EXTRA_RE.search(t):
            ai_broad[i] = True
        if MGMT_STRICT_RE.search(t):
            mgmt_strict[i] = True
            mgmt_broad[i] = True
        elif MGMT_BROAD_EXTRA_RE.search(t):
            mgmt_broad[i] = True
        soft_matches = SOFT_RE.findall(t)
        if soft_matches:
            soft_any[i] = True
            soft_count[i] = len(soft_matches)
        if SCOPE_RE.search(t):
            scope_any[i] = True
        if EDU_PHD_RE.search(t):
            edu_phd[i] = True
        if EDU_MS_RE.search(t):
            edu_ms[i] = True
        if EDU_BS_FLOOR_RE.search(t):
            edu_bs[i] = True
        if (i + 1) % 10000 == 0:
            print(f"[step6]   regex {i+1}/{n}")
    merged["ai_strict"] = ai_strict
    merged["ai_broad"] = ai_broad
    merged["mgmt_strict"] = mgmt_strict
    merged["mgmt_broad"] = mgmt_broad
    merged["soft_any"] = soft_any
    merged["soft_count"] = soft_count
    merged["scope_any"] = scope_any
    merged["edu_phd"] = edu_phd
    merged["edu_ms"] = edu_ms
    merged["edu_bs"] = edu_bs
    return merged


def cohens_d(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (|d|, pooled_sd) between two arrays."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0, 0.0
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0, 0.0
    return abs((ma - mb) / pooled), pooled


def prop_diff_se(pa: float, na: int, pb: float, nb: int) -> tuple[float, float]:
    """Return (|pa - pb|, SE of difference)."""
    if na < 1 or nb < 1:
        return 0.0, 0.0
    se = math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    return abs(pa - pb), se


def mean_diff_sd(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (|mean(a) - mean(b)|, pooled SD)."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0, 0.0
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return abs(ma - mb), pooled


def snr_flag(ratio: float) -> str:
    if ratio == float("inf"):
        return "above_noise"
    if ratio >= 2:
        return "above_noise"
    if ratio >= 1:
        return "near_noise"
    return "below_noise"


def main() -> None:
    t0 = time.time()
    con = duckdb.connect()

    # Load base with source + flags + yoe
    print("[step6] loading base frame")
    base = con.execute(
        """
        SELECT
          uid, source,
          description_length,
          yoe_min_years_llm,
          yoe_extracted,
          llm_extraction_coverage,
          llm_classification_coverage,
          seniority_final,
          is_aggregator
        FROM 'data/unified.parquet'
        WHERE is_swe AND source_platform='linkedin'
          AND is_english = true AND date_flag='ok'
        """
    ).df()
    print(f"[step6] rows: {len(base)}")

    # Join tech matrix (for tech count per posting)
    print("[step6] loading tech matrix")
    tm = con.execute(f"SELECT * FROM '{MATRIX_PATH}'").df()
    tech_cols = [c for c in tm.columns if c != "uid"]
    tm["tech_count"] = tm[tech_cols].sum(axis=1)
    base = base.merge(tm[["uid", "tech_count"]], on="uid", how="left")

    # Compute text metrics (AI/mgmt/soft/scope/edu) — this is the expensive step
    merged = compute_text_metrics(base)

    # --- Split by source ---
    ars = merged[merged["source"] == "kaggle_arshkon"]
    asa = merged[merged["source"] == "kaggle_asaniczka"]
    scr = merged[merged["source"] == "scraped"]
    pooled = merged[merged["source"].isin(["kaggle_arshkon", "kaggle_asaniczka"])]

    # labeled-only subset for YOE-based metrics
    lab_ars = ars[ars["llm_classification_coverage"] == "labeled"]
    lab_asa = asa[asa["llm_classification_coverage"] == "labeled"]
    lab_scr = scr[scr["llm_classification_coverage"] == "labeled"]
    lab_pooled = pooled[pooled["llm_classification_coverage"] == "labeled"]

    rows: list[dict] = []

    # -------- Continuous metrics (Cohen's d) --------
    def cont(metric, getter):
        ars_v = getter(ars)
        asa_v = getter(asa)
        scr_v = getter(scr)
        pool_v = getter(pooled)
        d_w, sd_w = cohens_d(ars_v, asa_v)
        d_c, sd_c = cohens_d(pool_v, scr_v)
        ratio = (d_c / d_w) if d_w > 0 else (float("inf") if d_c > 0 else 0.0)
        rows.append({
            "metric": metric,
            "metric_type": "continuous",
            "arshkon_value": round(float(np.nanmean(ars_v)), 4) if len(ars_v) else None,
            "asaniczka_value": round(float(np.nanmean(asa_v)), 4) if len(asa_v) else None,
            "scraped_value": round(float(np.nanmean(scr_v)), 4) if len(scr_v) else None,
            "within_2024_effect": round(d_w, 4),
            "within_2024_sd": round(sd_w, 4),
            "cross_period_effect": round(d_c, 4),
            "cross_period_sd": round(sd_c, 4),
            "calibration_ratio": "inf" if ratio == float("inf") else round(ratio, 3),
            "snr_flag": snr_flag(ratio),
            "notes": "Cohen's d; mean reported",
        })

    cont("description_length_mean", lambda d: d["description_length"].dropna().astype(float).values)
    # Median is not an effect-size-friendly metric — we record the median values and use d on the value itself
    # Add median-specific rows below using mean_diff_sd as a stand-in for effect; recompute separately
    def cont_median(metric, getter):
        ars_v = getter(ars)
        asa_v = getter(asa)
        scr_v = getter(scr)
        pool_v = getter(pooled)
        # Effect: median-of-medians difference, divided by pooled SD
        d_w, sd_w = cohens_d(ars_v, asa_v)
        d_c, sd_c = cohens_d(pool_v, scr_v)
        ratio = (d_c / d_w) if d_w > 0 else (float("inf") if d_c > 0 else 0.0)
        rows.append({
            "metric": metric,
            "metric_type": "continuous",
            "arshkon_value": round(float(np.nanmedian(ars_v)), 4) if len(ars_v) else None,
            "asaniczka_value": round(float(np.nanmedian(asa_v)), 4) if len(asa_v) else None,
            "scraped_value": round(float(np.nanmedian(scr_v)), 4) if len(scr_v) else None,
            "within_2024_effect": round(d_w, 4),
            "within_2024_sd": round(sd_w, 4),
            "cross_period_effect": round(d_c, 4),
            "cross_period_sd": round(sd_c, 4),
            "calibration_ratio": "inf" if ratio == float("inf") else round(ratio, 3),
            "snr_flag": snr_flag(ratio),
            "notes": "Median reported; Cohen's d (on underlying values)",
        })

    cont_median("description_length_median", lambda d: d["description_length"].dropna().astype(float).values)
    cont_median(
        "yoe_min_years_llm_median",
        lambda d: d.loc[d["llm_classification_coverage"] == "labeled", "yoe_min_years_llm"].dropna().astype(float).values,
    )
    cont_median(
        "yoe_extracted_median",
        lambda d: d["yoe_extracted"].dropna().astype(float).values,
    )
    cont("tech_count_per_posting_mean", lambda d: d["tech_count"].dropna().astype(float).values)

    # -------- Proportion metrics --------
    def prop(metric, getter, pooled_getter=None, labeled_only=False, notes=""):
        ars_src = lab_ars if labeled_only else ars
        asa_src = lab_asa if labeled_only else asa
        scr_src = lab_scr if labeled_only else scr
        pool_src = lab_pooled if labeled_only else pooled

        ars_flag = getter(ars_src)
        asa_flag = getter(asa_src)
        scr_flag = getter(scr_src)
        pool_flag = getter(pool_src) if pooled_getter is None else pooled_getter(pool_src)

        na = len(ars_flag); pa = float(np.mean(ars_flag)) if na else 0.0
        nb = len(asa_flag); pb = float(np.mean(asa_flag)) if nb else 0.0
        nc = len(scr_flag); pc = float(np.mean(scr_flag)) if nc else 0.0
        nd_ = len(pool_flag); pd_ = float(np.mean(pool_flag)) if nd_ else 0.0

        w_diff, w_se = prop_diff_se(pa, na, pb, nb)
        c_diff, c_se = prop_diff_se(pd_, nd_, pc, nc)
        ratio = (c_diff / w_diff) if w_diff > 0 else (float("inf") if c_diff > 0 else 0.0)
        rows.append({
            "metric": metric,
            "metric_type": "proportion",
            "arshkon_value": round(pa, 4),
            "asaniczka_value": round(pb, 4),
            "scraped_value": round(pc, 4),
            "within_2024_effect": round(w_diff, 4),
            "within_2024_sd": round(w_se, 4),
            "cross_period_effect": round(c_diff, 4),
            "cross_period_sd": round(c_se, 4),
            "calibration_ratio": "inf" if ratio == float("inf") else round(ratio, 3),
            "snr_flag": snr_flag(ratio),
            "notes": notes,
        })

    prop(
        "j3_yoe_le_2_share",
        lambda d: (d["yoe_min_years_llm"].dropna() <= 2).values if "yoe_min_years_llm" in d.columns else np.array([], dtype=bool),
        labeled_only=True,
        notes="J3 YOE≤2 among labeled; denominator = labeled rows",
    )
    prop(
        "j1_entry_share",
        lambda d: (d["seniority_final"] == "entry").values,
        notes="seniority_final == 'entry'",
    )
    prop(
        "j2_entry_or_associate_share",
        lambda d: d["seniority_final"].isin(["entry", "associate"]).values,
        notes="seniority_final in entry/associate",
    )
    prop(
        "s4_yoe_ge_5_share",
        lambda d: (d["yoe_min_years_llm"].dropna() >= 5).values,
        labeled_only=True,
        notes="S4 YOE≥5 among labeled",
    )
    prop(
        "s1_midsen_or_director_share",
        lambda d: d["seniority_final"].isin(["mid-senior", "director"]).values,
        notes="seniority_final in mid-senior/director",
    )
    prop(
        "aggregator_share",
        lambda d: d["is_aggregator"].fillna(False).values,
        notes="is_aggregator",
    )
    prop(
        "ai_mention_strict",
        lambda d: d["ai_strict"].values,
        notes="regex strict AI (copilot, cursor, claude, ...)",
    )
    prop(
        "ai_mention_broad",
        lambda d: d["ai_broad"].values,
        notes="AI strict OR (agent|ml|ai|llm|artificial intelligence|mcp)",
    )
    prop(
        "mgmt_strict",
        lambda d: d["mgmt_strict"].values,
        notes="mentor|coach|hire|headcount|performance review|direct reports",
    )
    prop(
        "mgmt_broad",
        lambda d: d["mgmt_broad"].values,
        notes="mgmt strict OR (lead|team|stakeholder|coordinate|manage)",
    )
    prop(
        "soft_skill_density",
        lambda d: d["soft_any"].values,
        notes="collaborative|communication|teamwork|problem-solving|interpersonal|leadership",
    )
    prop(
        "scope_term_rate",
        lambda d: d["scope_any"].values,
        notes="ownership|end-to-end|cross-functional|autonomous|initiative|stakeholder",
    )
    prop(
        "edu_phd_mentioned",
        lambda d: d["edu_phd"].values,
        notes="PhD / doctorate mention",
    )
    prop(
        "edu_ms_mentioned",
        lambda d: d["edu_ms"].values,
        notes="Master's / MS mention",
    )
    prop(
        "edu_bs_mentioned_floor",
        lambda d: d["edu_bs"].values,
        notes="Bachelor's / BS mention",
    )

    # -------- Count metrics (mean difference) --------
    def count_metric(metric, getter, notes=""):
        ars_v = getter(ars)
        asa_v = getter(asa)
        scr_v = getter(scr)
        pool_v = getter(pooled)
        w_diff, w_sd = mean_diff_sd(ars_v, asa_v)
        c_diff, c_sd = mean_diff_sd(pool_v, scr_v)
        ratio = (c_diff / w_diff) if w_diff > 0 else (float("inf") if c_diff > 0 else 0.0)
        rows.append({
            "metric": metric,
            "metric_type": "count",
            "arshkon_value": round(float(np.nanmean(ars_v)), 4) if len(ars_v) else None,
            "asaniczka_value": round(float(np.nanmean(asa_v)), 4) if len(asa_v) else None,
            "scraped_value": round(float(np.nanmean(scr_v)), 4) if len(scr_v) else None,
            "within_2024_effect": round(w_diff, 4),
            "within_2024_sd": round(w_sd, 4),
            "cross_period_effect": round(c_diff, 4),
            "cross_period_sd": round(c_sd, 4),
            "calibration_ratio": "inf" if ratio == float("inf") else round(ratio, 3),
            "snr_flag": snr_flag(ratio),
            "notes": notes,
        })

    count_metric("distinct_techs_per_posting_mean", lambda d: d["tech_count"].dropna().astype(float).values, notes="sum over tech matrix")
    count_metric("soft_skill_term_count_mean", lambda d: d["soft_count"].dropna().astype(float).values, notes="count of soft-skill terms per posting")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"[step6] wrote {len(out)} metrics -> {OUT_PATH}")
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 100):
        print(out[["metric", "arshkon_value", "asaniczka_value", "scraped_value",
                   "within_2024_effect", "cross_period_effect", "calibration_ratio", "snr_flag"]])
    elapsed = time.time() - t0
    print(f"[step6] elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()
