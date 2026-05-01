from __future__ import annotations

import re
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from T22_T23_common import (
    AI_DOMAIN_RE,
    AI_GENERAL_RE,
    AI_TOOL_RE,
    DEGREE_PATTERNS,
    FIRM_PATTERNS,
    HEDGE_PATTERNS,
    MGMT_BROAD_PATTERNS,
    MGMT_STRONG_PATTERNS,
    REPORT_DIR,
    SENIOR_SCOPE_PATTERNS,
    SCOPE_PATTERNS,
    ensure_dirs,
    count_ai_general,
    count_hits,
    has_any,
    load_core_text_frame,
    load_full_text_frame,
    load_tech_counts,
    pattern_validation_payload,
    qdf,
    write_validated_patterns,
)


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "exploration" / "tables" / "T22"
FIG_DIR = ROOT / "exploration" / "figures" / "T22"


def save_csv(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    return path


def save_fig(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def build_feature_frame(con: duckdb.DuckDBPyConnection, text_source: str = "llm") -> pd.DataFrame:
    frame = load_core_text_frame(con, text_source=text_source)
    tech = load_tech_counts(con, frame["uid"].tolist())
    frame = frame.merge(tech, on="uid", how="left")
    frame["core_text"] = frame["core_text"].fillna("")
    frame["text_len"] = frame["core_text"].str.len()
    frame["scope_count"] = frame["core_text"].map(lambda s: count_hits(s, SCOPE_PATTERNS))
    frame["senior_scope_count"] = frame["core_text"].map(lambda s: count_hits(s, SENIOR_SCOPE_PATTERNS))
    frame["management_strong_count"] = frame["core_text"].map(lambda s: count_hits(s, MGMT_STRONG_PATTERNS))
    frame["management_broad_count"] = frame["core_text"].map(lambda s: count_hits(s, MGMT_BROAD_PATTERNS))
    frame["hedge_count"] = frame["core_text"].map(lambda s: count_hits(s, HEDGE_PATTERNS))
    frame["firm_count"] = frame["core_text"].map(lambda s: count_hits(s, FIRM_PATTERNS))
    frame["ai_tool_count_text"] = frame["core_text"].map(lambda s: len(AI_TOOL_RE.findall(s or "")))
    frame["ai_domain_count_text"] = frame["core_text"].map(lambda s: len(AI_DOMAIN_RE.findall(s or "")))
    frame["ai_general_count"] = frame["core_text"].map(count_ai_general)
    frame["any_ai_text"] = frame["core_text"].map(lambda s: has_any(s, AI_TOOL_RE) or has_any(s, AI_DOMAIN_RE) or has_any(s, AI_GENERAL_RE))
    frame["any_ai_tool_text"] = frame["core_text"].map(lambda s: has_any(s, AI_TOOL_RE))
    frame["any_ai_domain_text"] = frame["core_text"].map(lambda s: has_any(s, AI_DOMAIN_RE))
    frame["kitchen_sink_product"] = frame["tech_count"].fillna(0) * frame["scope_count"].fillna(0)
    frame["aspiration_ratio"] = frame.apply(lambda r: (r.hedge_count / r.firm_count) if r.firm_count else np.nan, axis=1)
    frame["yoe_scope_mismatch"] = (
        (frame["seniority_final"] == "entry")
        & ((frame["yoe_extracted"].fillna(-1) >= 5) | (frame["senior_scope_count"].fillna(0) >= 3))
    ).astype(int)
    degree = frame["core_text"].fillna("")
    no_degree = re.compile(DEGREE_PATTERNS["no_degree"], re.I)
    degree_any = re.compile(r"(?:bachelor|master|phd|m\.?s\.?)", re.I)
    exp_no = re.compile(r"no experience required|no experience", re.I)
    exp_many = re.compile(r"(?:[5-9]\+?\s*years?|10\+\s*years?)", re.I)
    frame["degree_contra"] = degree.map(lambda s: int(bool((no_degree.search(s) and degree_any.search(s)) or (exp_no.search(s) and exp_many.search(s)))))
    return frame


def compute_ghost_score(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    for col in ["kitchen_sink_product", "aspiration_ratio", "senior_scope_count", "management_strong_count", "management_broad_count", "tech_count", "scope_count"]:
        series = df[col].fillna(0).astype(float)
        denom = (series.rank(method="average", pct=True) - 0.5).fillna(0)
        df[f"{col}_pct"] = denom
    df["ghost_score"] = (
        df["kitchen_sink_product_pct"]
        + df["aspiration_ratio_pct"]
        + df["yoe_scope_mismatch"].astype(float)
        + df["degree_contra"].astype(float)
        + (df["management_broad_count"] > 0).astype(float) * 0.25
    )
    return df


def build_prevalence_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indicators = {
        "kitchen_sink": df["kitchen_sink_product"] > 0,
        "aspiration_heavy": df["aspiration_ratio"] >= 1.0,
        "yoe_scope_mismatch": df["yoe_scope_mismatch"] == 1,
        "degree_contra": df["degree_contra"] == 1,
        "management_broad": df["management_broad_count"] > 0,
        "management_strong": df["management_strong_count"] > 0,
    }
    for (source, period, seniority_final, is_aggregator), g in df.groupby(["source", "period", "seniority_final", "is_aggregator"], dropna=False):
        rec = {
            "source": source,
            "period": period,
            "seniority_final": seniority_final,
            "is_aggregator": bool(is_aggregator),
            "n": len(g),
            "mean_tech_count": g["tech_count"].mean(),
            "mean_scope_count": g["scope_count"].mean(),
            "mean_aspiration_ratio": g["aspiration_ratio"].replace([np.inf, -np.inf], np.nan).mean(),
            "median_aspiration_ratio": g["aspiration_ratio"].replace([np.inf, -np.inf], np.nan).median(),
            "mean_ghost_score": g["ghost_score"].mean(),
        }
        for name, mask in indicators.items():
            rec[f"{name}_share"] = float(mask.loc[g.index].mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def build_ai_aspiration_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, mask in [("AI", df["any_ai_text"]), ("non_AI", ~df["any_ai_text"])]:
        g = df.loc[mask].copy()
        rows.append(
            {
                "group": label,
                "n": len(g),
                "hedge_sum": int(g["hedge_count"].sum()),
                "firm_sum": int(g["firm_count"].sum()),
                "aggregate_ratio": float(g["hedge_count"].sum() / g["firm_count"].sum()) if g["firm_count"].sum() else np.nan,
                "mean_post_ratio": float(g["aspiration_ratio"].replace([np.inf, -np.inf], np.nan).mean()),
                "median_post_ratio": float(g["aspiration_ratio"].replace([np.inf, -np.inf], np.nan).median()),
                "ai_tool_rate": float(g["any_ai_tool_text"].mean()),
                "ai_domain_rate": float(g["any_ai_domain_text"].mean()),
            }
        )
    return pd.DataFrame(rows)


def template_saturation(frame: pd.DataFrame) -> pd.DataFrame:
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words=None,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
    )
    texts = frame["core_text"].fillna("").tolist()
    X = vec.fit_transform(texts)
    frame = frame.reset_index(drop=True).copy()
    frame["row_idx"] = np.arange(len(frame))
    rows = []
    for company, g in frame.groupby("company_name_canonical", dropna=False):
        if len(g) < 3:
            continue
        idx = g["row_idx"].to_numpy()
        if len(idx) > 50:
            idx = idx[:50]
        sub = X[idx]
        sim = cosine_similarity(sub)
        upper = sim[np.triu_indices_from(sim, k=1)]
        finite = upper[np.isfinite(upper)]
        rows.append(
            {
                "company_name_canonical": company,
                "n_postings": len(idx),
                "mean_pairwise_cosine": float(np.nanmean(finite)) if len(finite) else np.nan,
                "median_pairwise_cosine": float(np.nanmedian(finite)) if len(finite) else np.nan,
                "flag_gt_08": bool(np.nanmean(finite) > 0.8) if len(finite) else False,
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_pairwise_cosine", "n_postings"], ascending=[False, False])


def main() -> None:
    ensure_dirs(TABLE_DIR, FIG_DIR, REPORT_DIR)
    con = duckdb.connect()

    # Primary analysis: section-filtered core text on the LLM-cleaned frame.
    primary = build_feature_frame(con, text_source="llm")
    primary = compute_ghost_score(primary)

    # Sensitivity: same section-filtered frame across raw/llm text sources.
    all_core = build_feature_frame(con, text_source=None)
    all_core = compute_ghost_score(all_core)
    raw_full = load_full_text_frame(con, text_source="raw")
    raw_full = raw_full.merge(load_tech_counts(con, raw_full["uid"].tolist()), on="uid", how="left")
    raw_full["core_text"] = raw_full["core_text"].fillna("")
    raw_full["text_len"] = raw_full["core_text"].str.len()
    raw_full["scope_count"] = raw_full["core_text"].map(lambda s: count_hits(s, SCOPE_PATTERNS))
    raw_full["senior_scope_count"] = raw_full["core_text"].map(lambda s: count_hits(s, SENIOR_SCOPE_PATTERNS))
    raw_full["management_strong_count"] = raw_full["core_text"].map(lambda s: count_hits(s, MGMT_STRONG_PATTERNS))
    raw_full["management_broad_count"] = raw_full["core_text"].map(lambda s: count_hits(s, MGMT_BROAD_PATTERNS))
    raw_full["hedge_count"] = raw_full["core_text"].map(lambda s: count_hits(s, HEDGE_PATTERNS))
    raw_full["firm_count"] = raw_full["core_text"].map(lambda s: count_hits(s, FIRM_PATTERNS))
    raw_full["any_ai_text"] = raw_full["core_text"].map(lambda s: has_any(s, AI_TOOL_RE) or has_any(s, AI_DOMAIN_RE) or has_any(s, AI_GENERAL_RE))
    raw_full["any_ai_tool_text"] = raw_full["core_text"].map(lambda s: has_any(s, AI_TOOL_RE))
    raw_full["any_ai_domain_text"] = raw_full["core_text"].map(lambda s: has_any(s, AI_DOMAIN_RE))
    raw_full["kitchen_sink_product"] = raw_full["tech_count"].fillna(0) * raw_full["scope_count"].fillna(0)
    raw_full["aspiration_ratio"] = raw_full.apply(lambda r: (r.hedge_count / r.firm_count) if r.firm_count else np.nan, axis=1)
    raw_full["yoe_scope_mismatch"] = (
        (raw_full["seniority_final"] == "entry")
        & ((raw_full["yoe_extracted"].fillna(-1) >= 5) | (raw_full["senior_scope_count"].fillna(0) >= 3))
    ).astype(int)
    no_degree = re.compile(DEGREE_PATTERNS["no_degree"], re.I)
    degree_any = re.compile(r"(?:bachelor|master|phd|m\.?s\.?)", re.I)
    exp_no = re.compile(r"no experience required|no experience", re.I)
    exp_many = re.compile(r"(?:[5-9]\+?\s*years?|10\+\s*years?)", re.I)
    raw_full["degree_contra"] = raw_full["core_text"].fillna("").map(lambda s: int(bool((no_degree.search(s) and degree_any.search(s)) or (exp_no.search(s) and exp_many.search(s)))))
    raw_full = compute_ghost_score(raw_full)

    # Save indicator-level prevalence.
    prevalence = build_prevalence_table(primary)
    save_csv(prevalence.assign(text_source="llm_core"), "T22_ghost_prevalence_primary_llm.csv")
    save_csv(build_prevalence_table(raw_full).assign(text_source="raw_full"), "T22_ghost_prevalence_raw_full.csv")

    ai_asp = build_ai_aspiration_table(primary)
    save_csv(ai_asp.assign(text_source="llm_core"), "T22_ai_aspiration_primary_llm.csv")
    save_csv(build_ai_aspiration_table(raw_full).assign(text_source="raw_full"), "T22_ai_aspiration_raw_full.csv")

    # Top ghost-like entry postings.
    entry = primary[primary["seniority_final"] == "entry"].copy()
    entry["ghost_rank_score"] = (
        entry["kitchen_sink_product_pct"]
        + entry["aspiration_ratio_pct"]
        + entry["yoe_scope_mismatch"].astype(float)
        + entry["degree_contra"].astype(float)
        + (entry["management_broad_count"] > 0).astype(float) * 0.25
    )
    top20 = entry.sort_values(["ghost_rank_score", "aspiration_ratio", "kitchen_sink_product"], ascending=False).head(20)
    top20_cols = [
        "uid",
        "source",
        "period",
        "title",
        "company_name_canonical",
        "is_aggregator",
        "yoe_extracted",
        "kitchen_sink_product",
        "aspiration_ratio",
        "yoe_scope_mismatch",
        "degree_contra",
        "management_broad_count",
        "management_strong_count",
        "scope_count",
        "tech_count",
        "any_ai_text",
        "core_text",
    ]
    save_csv(top20[top20_cols], "T22_top_ghost_entry_postings.csv")

    # Template saturation.
    sat = template_saturation(primary)
    save_csv(sat, "T22_template_saturation_companies.csv")

    # Industry patterns.
    industry = (
        primary.groupby(["company_industry", "period"], dropna=False)
        .agg(
            n=("uid", "size"),
            ghost_mean=("ghost_score", "mean"),
            ai_any_share=("any_ai_text", "mean"),
            aspiration_ratio=("aspiration_ratio", "mean"),
            yoe_scope_mismatch_share=("yoe_scope_mismatch", "mean"),
            degree_contra_share=("degree_contra", "mean"),
        )
        .reset_index()
        .sort_values(["ghost_mean", "n"], ascending=[False, False])
    )
    save_csv(industry, "T22_industry_ghost_rates.csv")

    # Aggregator vs direct comparison.
    agg = (
        primary.groupby(["is_aggregator"], dropna=False)
        .agg(
            n=("uid", "size"),
            ghost_mean=("ghost_score", "mean"),
            kitchen_sink_share=("kitchen_sink_product", lambda s: float((s > 0).mean())),
            aspiration_ratio=("aspiration_ratio", "mean"),
            ai_any_share=("any_ai_text", "mean"),
            yoe_scope_mismatch_share=("yoe_scope_mismatch", "mean"),
        )
        .reset_index()
    )
    save_csv(agg, "T22_aggregator_direct_comparison.csv")

    # Validation payload for downstream reuse.
    write_validated_patterns()

    # Figures.
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_df = prevalence[~prevalence["is_aggregator"]].copy()
    plot_df["label"] = plot_df["seniority_final"] + " | " + plot_df["period"]
    plot_df = plot_df.sort_values(["period", "seniority_final"])
    ax.bar(plot_df["label"], plot_df["mean_ghost_score"], color="#2f6f8f")
    ax.set_ylabel("Mean ghost score")
    ax.set_title("Ghost score by period and seniority, direct employers only")
    ax.tick_params(axis="x", rotation=45)
    save_fig(fig, "T22_ghost_score_by_period_seniority.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(ai_asp["group"], ai_asp["aggregate_ratio"], color=["#2f6f8f", "#8a8a8a"])
    ax.set_ylabel("Hedge / firm count ratio")
    ax.set_title("AI postings are more aspirational than non-AI postings in core sections")
    save_fig(fig, "T22_ai_aspiration_ratio.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    top_sat = sat.head(15).copy()
    ax.barh(top_sat["company_name_canonical"].fillna("unknown")[::-1], top_sat["mean_pairwise_cosine"][::-1], color="#4c78a8")
    ax.axvline(0.8, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean pairwise cosine similarity")
    ax.set_title("Top template-saturated companies")
    save_fig(fig, "T22_template_saturation_top_companies.png")


if __name__ == "__main__":
    main()
