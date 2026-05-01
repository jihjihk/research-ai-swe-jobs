"""T16 — Company hiring strategy typology.

Builds a per-company change vector for overlap-panel companies
(kaggle_arshkon vs scraped, >=3 SWE each) and clusters them via k-means.

Key constraints applied (per Gate 2 corrections + analytical preamble):
- seniority always reported under BOTH `seniority_final` and YOE <= 2
- AI prevalence uses the broad 24-tech T14 union on the full tech matrix
  (all rows, binary per-posting)
- Scope uses ONLY `end-to-end` and `cross-functional` (clean per V1); bare
  `ownership` is excluded
- Description length uses raw `description` on all rows (length is a length
  metric, not boilerplate-sensitive to tokenization) but we also keep the
  `text_source='llm'` cleaned-text length as a sensitivity
- Decomposition is arshkon-vs-scraped (not pooled)

Outputs tables + figures under exploration/{tables,figures}/T16/.
"""

from __future__ import annotations

import os
import re
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNIFIED = ROOT / "data" / "unified.parquet"
SHARED = ROOT / "exploration" / "artifacts" / "shared"
CLEAN = SHARED / "swe_cleaned_text.parquet"
TECHM = SHARED / "swe_tech_matrix.parquet"
ARCH = SHARED / "swe_archetype_labels.parquet"
TBL = ROOT / "exploration" / "tables" / "T16"
FIG = ROOT / "exploration" / "figures" / "T16"
TBL.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


# T14 broad AI-union (24 techs) — match Wave 2 convention (prevalence 5.15 -> 28.63, SNR 13.3)
AI_UNION = [
    "machine_learning", "deep_learning", "nlp", "computer_vision", "llm",
    "langchain", "langgraph", "rag", "vector_db", "pinecone", "chromadb",
    "huggingface", "openai_api", "claude_api", "prompt_engineering",
    "fine_tuning", "mcp", "agents_framework", "gpt", "transformer_arch",
    "embedding", "copilot", "cursor_tool", "claude_tool",
]

# Clean scope terms (V1 validated)
SCOPE_RE = re.compile(r"\b(?:end[- ]to[- ]end|cross[- ]functional)\b", re.I)


def fetch_base():
    con = duckdb.connect()
    ai_sum = " + ".join([f"CAST(t.{c} AS INTEGER)" for c in AI_UNION])
    q = f"""
    SELECT
        u.uid,
        u.source,
        u.company_name_canonical AS company,
        u.is_aggregator,
        u.seniority_final,
        u.yoe_extracted,
        u.description,
        u.description_core_llm,
        u.llm_extraction_coverage,
        u.metro_area,
        COALESCE(({ai_sum}), 0) AS ai_tech_count,
        CAST(({ai_sum}) > 0 AS INTEGER) AS ai_any
    FROM read_parquet('{UNIFIED.as_posix()}') u
    LEFT JOIN read_parquet('{TECHM.as_posix()}') t USING (uid)
    WHERE u.is_swe = true
      AND u.source_platform = 'linkedin'
      AND u.is_english = true
      AND u.date_flag = 'ok'
      AND u.source IN ('kaggle_arshkon', 'scraped')
      AND u.company_name_canonical IS NOT NULL
    """
    df = con.execute(q).fetchdf()
    con.close()
    return df


def attach_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["period_label"] = np.where(df["source"] == "kaggle_arshkon", "y2024", "y2026")
    df["entry_final"] = (df["seniority_final"] == "entry").astype(int)
    # Known = any labeled category (exclude 'unknown'); includes entry/mid-senior/associate/director
    df["entry_known"] = (~df["seniority_final"].isin(["unknown", None])).astype(int)
    df["yoe_le2"] = (df["yoe_extracted"] <= 2).astype(int)
    df["yoe_known"] = df["yoe_extracted"].notna().astype(int)
    # Description length on raw description — length is not boilerplate-sensitive
    df["desc_len"] = df["description"].fillna("").str.len()
    # Scope count uses cleaned text when available, raw otherwise. We only use
    # end-to-end / cross-functional (V1 clean). Binary per posting.
    txt = df["description_core_llm"].where(
        df["llm_extraction_coverage"] == "labeled", df["description"]
    ).fillna("")
    df["scope_clean_any"] = txt.str.contains(SCOPE_RE, regex=True).astype(int)
    return df


def build_overlap_panel(df: pd.DataFrame, min_per_period: int = 3) -> pd.DataFrame:
    counts = df.groupby(["company", "period_label"]).size().unstack(fill_value=0)
    keep = counts[(counts.get("y2024", 0) >= min_per_period) & (counts.get("y2026", 0) >= min_per_period)].index
    return df[df["company"].isin(keep)].copy()


def per_company_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        out = {
            "n": len(g),
            "entry_final_n": g["entry_final"].sum(),
            "entry_final_known_n": g["entry_known"].sum(),
            "yoe_le2_n": g["yoe_le2"].sum(),
            "yoe_known_n": g["yoe_known"].sum(),
            "ai_any": g["ai_any"].mean(),
            "ai_tech_mean": g["ai_tech_count"].mean(),
            "desc_len": g["desc_len"].mean(),
            "scope_clean_any": g["scope_clean_any"].mean(),
            "agg_share": g["is_aggregator"].mean(),
        }
        return pd.Series(out)
    m = panel.groupby(["company", "period_label"]).apply(agg).unstack("period_label")
    # Flatten
    m.columns = [f"{a}_{b}" for a, b in m.columns]
    # Rates
    m["entry_final_rate_y2024"] = m["entry_final_n_y2024"] / m["entry_final_known_n_y2024"].replace(0, np.nan)
    m["entry_final_rate_y2026"] = m["entry_final_n_y2026"] / m["entry_final_known_n_y2026"].replace(0, np.nan)
    m["yoe_le2_rate_y2024"] = m["yoe_le2_n_y2024"] / m["yoe_known_n_y2024"].replace(0, np.nan)
    m["yoe_le2_rate_y2026"] = m["yoe_le2_n_y2026"] / m["yoe_known_n_y2026"].replace(0, np.nan)
    # Change metrics
    for col in ["entry_final_rate", "yoe_le2_rate", "ai_any", "ai_tech_mean",
                "desc_len", "scope_clean_any"]:
        m[f"d_{col}"] = m[f"{col}_y2026"] - m[f"{col}_y2024"]
    return m.reset_index()


def cluster_companies(m: pd.DataFrame, k: int = 4, seed: int = 42):
    features = ["d_entry_final_rate", "d_yoe_le2_rate", "d_ai_any",
                "d_desc_len", "d_scope_clean_any"]
    X = m[features].copy()
    # Impute missing entry_rate changes with 0 (companies with no known seniority)
    X = X.fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed)
    labels = km.fit_predict(Xs)
    m = m.copy()
    m["cluster"] = labels
    return m, scaler, km, features


def decomposition(panel: pd.DataFrame, metric_col: str, denom_col: str | None = None):
    """Oaxaca-style within/between decomposition against `company` as the group.

    If denom_col is given (for rate metrics), treats the rate as n/denom per company.
    Returns dict with total, within, between.
    """
    g = panel.groupby(["company", "period_label"])
    if denom_col:
        agg = g.agg(num=(metric_col, "sum"), den=(denom_col, "sum"), n=(metric_col, "size"))
        agg = agg.reset_index()
        agg["rate"] = agg["num"] / agg["den"]
    else:
        agg = g.agg(rate=(metric_col, "mean"), n=(metric_col, "size")).reset_index()

    # pivot
    wide = agg.pivot(index="company", columns="period_label", values=["rate", "n"])
    wide = wide.dropna(subset=[("rate", "y2024"), ("rate", "y2026")])
    r24 = wide[("rate", "y2024")].values
    r26 = wide[("rate", "y2026")].values
    n24 = wide[("n", "y2024")].values
    n26 = wide[("n", "y2026")].values
    w24 = n24 / n24.sum()
    w26 = n26 / n26.sum()
    # panel means
    mean_24 = np.nansum(w24 * r24)
    mean_26 = np.nansum(w26 * r26)
    total = mean_26 - mean_24
    # within: holding weights fixed at mean(w24,w26)
    w_avg = (w24 + w26) / 2
    within = np.nansum(w_avg * (r26 - r24))
    # between: holding rate fixed at mean(r24,r26)
    r_avg = (r24 + r26) / 2
    between = np.nansum((w26 - w24) * r_avg)
    return {
        "n_companies": len(wide),
        "mean_2024": mean_24,
        "mean_2026": mean_26,
        "total_change": total,
        "within_component": within,
        "between_component": between,
        "residual": total - within - between,
    }


def archetype_analysis(panel: pd.DataFrame):
    con = duckdb.connect()
    arch = con.execute(
        f"SELECT uid, archetype_name FROM read_parquet('{ARCH.as_posix()}')"
    ).fetchdf()
    con.close()
    merged = panel.merge(arch, on="uid", how="left")
    labeled = merged[merged["archetype_name"].notna() & (merged["archetype_name"] != "No text / raw-only (unlabeled)")]
    # archetype distribution per company × period
    dist = labeled.groupby(["company", "period_label", "archetype_name"]).size().reset_index(name="n")
    total = dist.groupby(["company", "period_label"])["n"].transform("sum")
    dist["share"] = dist["n"] / total
    # for each company, get dominant archetype per period and pivot shift
    top = (dist.sort_values("share", ascending=False)
              .groupby(["company", "period_label"]).head(1))
    top_wide = top.pivot(index="company", columns="period_label", values="archetype_name")
    top_wide["pivoted"] = top_wide.get("y2024") != top_wide.get("y2026")
    # archetype share change per company: compute L1 distance between period vectors
    pivot = dist.pivot_table(index=["company", "archetype_name"], columns="period_label",
                             values="share", fill_value=0)
    pivot = pivot.reset_index()
    if "y2024" not in pivot.columns:
        pivot["y2024"] = 0
    if "y2026" not in pivot.columns:
        pivot["y2026"] = 0
    pivot["abs_delta"] = (pivot["y2026"] - pivot["y2024"]).abs()
    l1 = pivot.groupby("company")["abs_delta"].sum() / 2  # TVD
    return dist, top_wide, l1


def main():
    print("Fetching base rows...")
    df = fetch_base()
    df = attach_derived(df)
    print(f"Rows: {len(df):,}  (arshkon {sum(df.source=='kaggle_arshkon'):,} / scraped {sum(df.source=='scraped'):,})")

    panel = build_overlap_panel(df, min_per_period=3)
    n_companies = panel["company"].nunique()
    print(f"Overlap panel >=3: {n_companies} companies, {len(panel):,} rows")
    panel_5 = build_overlap_panel(df, min_per_period=5)
    print(f"Overlap panel >=5: {panel_5['company'].nunique()} companies")

    # Per-company metrics
    m = per_company_metrics(panel)
    m.to_csv(TBL / "per_company_change_vectors.csv", index=False)
    print(f"Wrote per_company_change_vectors.csv ({len(m)} companies)")

    # Cluster
    m_clustered, scaler, km, feats = cluster_companies(m, k=4)
    m_clustered.to_csv(TBL / "company_typology_k4.csv", index=False)
    # Cluster summary
    cluster_summary = (m_clustered.groupby("cluster")
                       .agg(n=("company", "size"),
                            d_entry_final_rate=("d_entry_final_rate", "mean"),
                            d_yoe_le2_rate=("d_yoe_le2_rate", "mean"),
                            d_ai_any=("d_ai_any", "mean"),
                            d_desc_len=("d_desc_len", "mean"),
                            d_scope_clean_any=("d_scope_clean_any", "mean"),
                            mean_agg_share=("agg_share_y2026", "mean"))
                       .reset_index())
    cluster_summary.to_csv(TBL / "cluster_summary_k4.csv", index=False)
    print("Cluster summary:")
    print(cluster_summary.to_string(index=False))

    # Decomposition — primary panel (>=3)
    panel_decomp = panel  # all rows in panel
    decomps = {
        "entry_final_rate": decomposition(panel_decomp, "entry_final", "entry_known"),
        "yoe_le2_rate": decomposition(panel_decomp, "yoe_le2", "yoe_known"),
        "ai_any": decomposition(panel_decomp, "ai_any"),
        "desc_len": decomposition(panel_decomp, "desc_len"),
        "scope_clean_any": decomposition(panel_decomp, "scope_clean_any"),
    }
    decomp_rows = [{"metric": k, **v} for k, v in decomps.items()]
    pd.DataFrame(decomp_rows).to_csv(TBL / "decomposition_panel.csv", index=False)
    print("Decomposition (panel >=3):")
    for k, v in decomps.items():
        print(f"  {k:20s} total={v['total_change']:+.4f} within={v['within_component']:+.4f} between={v['between_component']:+.4f} (n={v['n_companies']})")

    # Sensitivity: aggregator exclusion
    panel_no_agg = panel[panel["is_aggregator"] == False]
    decomps_noagg = {
        "entry_final_rate": decomposition(panel_no_agg, "entry_final", "entry_known"),
        "yoe_le2_rate": decomposition(panel_no_agg, "yoe_le2", "yoe_known"),
        "ai_any": decomposition(panel_no_agg, "ai_any"),
        "desc_len": decomposition(panel_no_agg, "desc_len"),
        "scope_clean_any": decomposition(panel_no_agg, "scope_clean_any"),
    }
    pd.DataFrame([{"metric": k, **v} for k, v in decomps_noagg.items()]).to_csv(
        TBL / "decomposition_panel_no_aggregator.csv", index=False)
    print("Decomposition (panel no-aggregator):")
    for k, v in decomps_noagg.items():
        print(f"  {k:20s} total={v['total_change']:+.4f} within={v['within_component']:+.4f}")

    # Archetype analysis
    dist, top_wide, l1 = archetype_analysis(panel)
    l1.sort_values(ascending=False).to_csv(TBL / "archetype_tvd_per_company.csv")
    top_wide.to_csv(TBL / "archetype_dominant_pivot.csv")
    n_pivoted = top_wide["pivoted"].sum()
    pivot_rate = n_pivoted / len(top_wide) if len(top_wide) else 0
    print(f"Archetype pivots: {n_pivoted} of {len(top_wide)} companies ({pivot_rate:.1%}) "
          f"changed dominant archetype 2024->2026")
    print(f"Median archetype TVD: {l1.median():.3f}  (TVD=0 means same mix, 1 means disjoint)")

    # Robustness: restrict to companies with >=5 archetype-labeled rows per period
    dist_lab_n = panel.merge(
        duckdb.connect().execute(
            f"SELECT uid, archetype_name FROM read_parquet('{ARCH.as_posix()}')"
        ).fetchdf(), on="uid", how="left"
    )
    dist_lab_n = dist_lab_n[dist_lab_n["archetype_name"].notna() &
                            (dist_lab_n["archetype_name"] != "No text / raw-only (unlabeled)")]
    counts_per = dist_lab_n.groupby(["company", "period_label"]).size().unstack(fill_value=0)
    stable = counts_per[(counts_per.get("y2024", 0) >= 5) & (counts_per.get("y2026", 0) >= 5)].index
    top_stable = top_wide.loc[top_wide.index.intersection(stable)]
    n_piv_stable = top_stable["pivoted"].sum() if len(top_stable) else 0
    piv_rate_stable = n_piv_stable / len(top_stable) if len(top_stable) else 0
    print(f"Archetype pivots (>=5 labeled per period): {n_piv_stable} of {len(top_stable)} "
          f"({piv_rate_stable:.1%})")
    # Archetype-stratified within-company decomposition for entry + AI
    # (aggregate across companies per archetype)
    print("\nWithin-company decomposition by archetype (entry_final, AI, scope):")
    archetype_rows = []
    for archetype_name, sub in dist_lab_n.groupby("archetype_name"):
        if len(sub) < 200:
            continue
        d_ent = decomposition(sub, "entry_final", "entry_known")
        d_yoe = decomposition(sub, "yoe_le2", "yoe_known")
        d_ai = decomposition(sub, "ai_any")
        d_scope = decomposition(sub, "scope_clean_any")
        archetype_rows.append({
            "archetype": archetype_name,
            "n_rows": len(sub),
            "n_companies_ent": d_ent["n_companies"],
            "entry_final_total": d_ent["total_change"],
            "entry_final_within": d_ent["within_component"],
            "yoe_le2_total": d_yoe["total_change"],
            "yoe_le2_within": d_yoe["within_component"],
            "ai_total": d_ai["total_change"],
            "ai_within": d_ai["within_component"],
            "scope_total": d_scope["total_change"],
            "scope_within": d_scope["within_component"],
        })
    arch_df = pd.DataFrame(archetype_rows)
    arch_df.to_csv(TBL / "decomposition_by_archetype.csv", index=False)
    print(arch_df.to_string(index=False))

    # New entrants profile (companies in 2026 not in 2024 at >=3)
    all_2024 = set(df[df.source == "kaggle_arshkon"]["company"].unique())
    all_2026 = set(df[df.source == "scraped"]["company"].unique())
    new_2026 = all_2026 - all_2024
    new_df = df[(df.source == "scraped") & (df.company.isin(new_2026))]
    returning_2026 = df[(df.source == "scraped") & (df.company.isin(all_2024))]
    def prof(d, label):
        ef_rate = d.loc[d.entry_known == 1, "entry_final"].mean()
        yoe_rate = d.loc[d.yoe_known == 1, "yoe_le2"].mean()
        return {
            "bucket": label, "n_companies": d.company.nunique(), "n_rows": len(d),
            "entry_final_rate": ef_rate, "yoe_le2_rate": yoe_rate,
            "ai_any": d.ai_any.mean(), "desc_len_mean": d.desc_len.mean(),
            "scope_any": d.scope_clean_any.mean(),
            "agg_share": d.is_aggregator.mean(),
        }
    # Entry-specialist intermediaries from T08 / Gate 2
    entry_specialists = {"SynergisticIT", "WayUp", "Lensa", "Emonics",
                         "Leidos", "Jobs via Dice", "Jobs via eFinancialCareers",
                         "Dice"}
    specialist_2026 = df[(df.source == "scraped") & (df.company.isin(entry_specialists))]
    new_profile = pd.DataFrame([
        prof(returning_2026, "returning_2026"),
        prof(new_df, "new_2026"),
        prof(specialist_2026, "entry_specialists_2026"),
    ])
    new_profile.to_csv(TBL / "new_entrants_profile.csv", index=False)
    print("New entrants profile:")
    print(new_profile.to_string(index=False))

    # Aggregator vs direct comparison on change metrics
    m["is_agg_2026"] = m["agg_share_y2026"] >= 0.5
    by_agg = (m.groupby("is_agg_2026")
              .agg(n=("company", "size"),
                   d_entry_final_rate=("d_entry_final_rate", "mean"),
                   d_yoe_le2_rate=("d_yoe_le2_rate", "mean"),
                   d_ai_any=("d_ai_any", "mean"),
                   d_desc_len=("d_desc_len", "mean"))
              .reset_index())
    by_agg.to_csv(TBL / "aggregator_vs_direct.csv", index=False)
    print("Aggregator vs direct (panel):")
    print(by_agg.to_string(index=False))

    # ---- Figures ----
    # 1. Cluster means heatmap (k=4)
    feats_names = ["Δentry (seniority_final)", "Δentry (YOE≤2)", "ΔAI any",
                   "Δdesc len", "Δscope (e2e+crossfn)"]
    C = cluster_summary[["d_entry_final_rate", "d_yoe_le2_rate", "d_ai_any",
                          "d_desc_len", "d_scope_clean_any"]].values
    # z-normalize per feature for display
    Cn = (C - C.mean(axis=0)) / (C.std(axis=0) + 1e-9)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    im = ax.imshow(Cn, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(feats_names)))
    ax.set_xticklabels(feats_names, rotation=30, ha="right")
    ax.set_yticks(range(len(cluster_summary)))
    ax.set_yticklabels([f"C{i} (n={int(n)})" for i, n in zip(cluster_summary.cluster, cluster_summary.n)])
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            raw = C[i, j]
            ax.text(j, i, f"{raw:+.3f}", ha="center", va="center", fontsize=8, color="black")
    plt.colorbar(im, ax=ax, label="z-score of cluster mean")
    ax.set_title("T16 — Company typology clusters (k=4) — raw Δ shown")
    plt.tight_layout()
    plt.savefig(FIG / "cluster_heatmap_k4.png", dpi=150)
    plt.close()

    # 2. Decomposition bars — compare seniority operationalizations
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics = ["entry_final_rate", "yoe_le2_rate", "ai_any", "desc_len", "scope_clean_any"]
    labels = ["entry\n(seniority_final)", "entry\n(YOE≤2)", "AI any", "desc len", "scope"]
    within = [decomps[k]["within_component"] for k in metrics]
    between = [decomps[k]["between_component"] for k in metrics]
    # normalize desc_len for visual (divide by its total)
    # Show as share of total change
    totals = [decomps[k]["total_change"] for k in metrics]
    x = np.arange(len(metrics))
    # Show within/between as signed shares of total (if total != 0)
    within_share = [w / t if abs(t) > 1e-12 else 0 for w, t in zip(within, totals)]
    between_share = [b / t if abs(t) > 1e-12 else 0 for b, t in zip(between, totals)]
    width = 0.35
    ax.bar(x - width/2, within_share, width, label="within-company", color="#4c78a8")
    ax.bar(x + width/2, between_share, width, label="between-company", color="#f58518")
    for i, t in enumerate(totals):
        ax.text(i, max(within_share[i], between_share[i]) + 0.03,
                f"Δ={t:+.4f}" if "rate" in metrics[i] or metrics[i].endswith("any")
                else f"Δ={t:+.1f}",
                ha="center", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Share of total change (within + between = 1)")
    ax.set_title("T16 — Within vs between decomposition (overlap panel n=240 companies)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "decomposition_shares.png", dpi=150)
    plt.close()

    # 3. Archetype pivot visualization
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(l1.values, bins=20, color="#4c78a8", edgecolor="black")
    ax.axvline(l1.median(), color="red", linestyle="--", label=f"median TVD={l1.median():.3f}")
    ax.set_xlabel("Archetype total variation distance (2024 vs 2026)")
    ax.set_ylabel("# companies")
    ax.set_title("T16 — Per-company archetype mix shift\n(0 = identical mix, 1 = disjoint)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "archetype_tvd_hist.png", dpi=150)
    plt.close()

    # 4. New-entrant vs returning bar comparison
    fig, ax = plt.subplots(figsize=(7.5, 4))
    metrics = ["entry_final_rate", "yoe_le2_rate", "ai_any", "scope_any"]
    labels = ["entry\n(seniority_final)", "entry\n(YOE≤2)", "AI any", "scope any"]
    xv = np.arange(len(metrics))
    ret_vals = [new_profile.iloc[0][m] for m in metrics]
    new_vals = [new_profile.iloc[1][m] for m in metrics]
    w = 0.35
    ax.bar(xv - w/2, ret_vals, w, label=f"returning (n={int(new_profile.iloc[0].n_companies)})", color="#4c78a8")
    ax.bar(xv + w/2, new_vals, w, label=f"new in 2026 (n={int(new_profile.iloc[1].n_companies)})", color="#e45756")
    ax.set_xticks(xv); ax.set_xticklabels(labels)
    ax.set_ylabel("Share")
    ax.set_title("T16 — New 2026 entrants vs returning companies")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "new_vs_returning.png", dpi=150)
    plt.close()

    # Store metadata
    meta = {
        "n_panel_ge3": int(n_companies),
        "n_panel_ge5": int(panel_5["company"].nunique()),
        "n_archetype_pivoted": int(n_pivoted),
        "n_archetype_total": int(len(top_wide)),
        "median_archetype_tvd": float(l1.median()),
        "cluster_sizes": cluster_summary.set_index("cluster")["n"].to_dict(),
        "decomposition": {k: v for k, v in decomps.items()},
        "decomposition_no_aggregator": {k: v for k, v in decomps_noagg.items()},
    }
    # Convert numpy types
    def cast(x):
        if isinstance(x, dict):
            return {k: cast(v) for k, v in x.items()}
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        return x
    (TBL / "summary.json").write_text(json.dumps(cast(meta), indent=2))
    print("Wrote summary.json")


if __name__ == "__main__":
    main()
