"""T17: Geographic market structure — metro-level heterogeneity.

Uses `metro_area` (multi-location rows excluded by construction). Reports:
  - Per-metro entry / AI / org-scope / length / tech change metrics
  - Metro correlation: does AI surge correlate with entry decline?
  - Archetype geographic distribution (2024 vs 2026)
  - Heatmap of metros x metrics
  - Remote work dimension: infeasible (is_remote_inferred 0% everywhere).

Outputs go to exploration/tables/T17/, exploration/figures/T17/.
"""

import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
import pyarrow.parquet as pq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.stats import spearmanr, pearsonr

ROOT = Path("/home/jihgaboot/gabor/job-research")
OUT_TBL = ROOT / "exploration/tables/T17"
OUT_FIG = ROOT / "exploration/figures/T17"
OUT_TBL.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

UNIFIED = ROOT / "data/unified.parquet"
CLEANED = ROOT / "exploration/artifacts/shared/swe_cleaned_text.parquet"
TECH = ROOT / "exploration/artifacts/shared/swe_tech_matrix.parquet"
SPEC = ROOT / "exploration/artifacts/shared/entry_specialist_employers.csv"
ARCHE = ROOT / "exploration/artifacts/shared/swe_archetype_labels.parquet"

# V1-approved patterns
AI_STRICT = re.compile(
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|vector database|pinecone|huggingface|hugging face)\b",
    re.IGNORECASE,
)
AI_BROAD_EXTRA = re.compile(
    r"\b(ai|artificial intelligence|ml|machine learning|llm|large language model|generative ai|genai|anthropic)\b",
    re.IGNORECASE,
)
ORG_SCOPE = re.compile(
    r"\b(cross[- ]functional|end[- ]to[- ]end|stakeholder|ownership|own the|own this|drive the|drive this|strategic|roadmap|partner with|collaborate across)\b",
    re.IGNORECASE,
)


def open_con():
    con = duckdb.connect()
    con.execute("SET memory_limit='20GB'")
    con.execute(
        "CREATE VIEW u AS SELECT * FROM read_parquet('" + str(UNIFIED) + "')"
    )
    con.execute(
        "CREATE VIEW c AS SELECT * FROM read_parquet('" + str(CLEANED) + "')"
    )
    con.execute(
        "CREATE VIEW t AS SELECT * FROM read_parquet('" + str(TECH) + "')"
    )
    return con


def main():
    con = open_con()

    # Load row-level metro data
    print("[T17] Loading row-level data...")
    base = con.execute(
        """
        SELECT u.uid,
               u.company_name_canonical AS company,
               u.source,
               CASE WHEN u.source='scraped' THEN 'scraped'
                    WHEN u.source IN ('kaggle_arshkon','kaggle_asaniczka') THEN '2024'
                    ELSE 'other' END AS period,
               u.is_aggregator,
               u.is_multi_location,
               u.metro_area,
               u.state_normalized,
               u.seniority_final,
               u.yoe_extracted,
               u.description_length,
               u.description AS raw_desc
        FROM u
        WHERE u.source_platform='linkedin' AND u.is_english AND u.date_flag='ok' AND u.is_swe=true
        """
    ).df()
    print(f"  {len(base):,} rows loaded")

    # Tech counts
    tcols = [c for c in pq.read_table(str(TECH)).column_names if c != "uid"]
    cols_sum = " + ".join([f"CAST({c} AS INTEGER)" for c in tcols])
    tech = con.execute(f"SELECT uid, ({cols_sum}) AS tech_count FROM t").df()
    base = base.merge(tech, on="uid", how="left")

    # Cleaned text length
    ct = con.execute(
        "SELECT uid, description_cleaned AS desc_cleaned FROM c WHERE text_source='llm'"
    ).df()
    base = base.merge(ct, on="uid", how="left")
    base["desc_cleaned_len"] = base["desc_cleaned"].fillna("").str.len()

    # Row-level flags
    raw = base["raw_desc"].fillna("")
    base["ai_strict"] = raw.str.contains(AI_STRICT).astype(int)
    base["ai_broad_extra"] = raw.str.contains(AI_BROAD_EXTRA).astype(int)
    base["ai_broad"] = (base["ai_strict"] | base["ai_broad_extra"]).astype(int)
    base["org_scope_count"] = base["desc_cleaned"].fillna("").apply(
        lambda s: len(ORG_SCOPE.findall(s))
    )
    base["org_scope_per_1k"] = np.where(
        base["desc_cleaned_len"] > 0,
        base["org_scope_count"] / base["desc_cleaned_len"] * 1000,
        0.0,
    )
    base["j1"] = (base["seniority_final"] == "entry").astype(int)
    base["j2"] = base["seniority_final"].isin(["entry", "associate"]).astype(int)
    base["j3"] = (base["yoe_extracted"] <= 2).fillna(False).astype(int)

    # Archetype labels (T09)
    arche = pd.read_parquet(ARCHE)
    base = base.merge(arche[["uid", "archetype_name"]], on="uid", how="left")

    # Specialist flag
    specialists = set(pd.read_csv(SPEC)["company"].tolist())
    base["is_specialist"] = base["company"].isin(specialists).astype(int)

    # Report multi-location counts dropped from metro rollups
    print("[T17] Multi-location coverage:")
    ml_report = base.groupby("period").agg(
        n_total=("uid", "size"),
        n_multi=("is_multi_location", "sum"),
        n_null_metro=("metro_area", lambda x: x.isna().sum()),
    ).reset_index()
    print(ml_report.to_string(index=False))
    ml_report.to_csv(OUT_TBL / "00_multi_location_exclusions.csv", index=False)

    # Remote-work infeasible check
    rem = con.execute(
        "SELECT SUM(CASE WHEN is_remote_inferred THEN 1 ELSE 0 END) AS n_rem_true, COUNT(*) AS n FROM u WHERE source_platform='linkedin' AND is_english AND date_flag='ok' AND is_swe=true"
    ).df()
    print("  is_remote_inferred true:", rem["n_rem_true"].iloc[0], "of", rem["n"].iloc[0])

    # Restrict to rows with known metro
    ga = base[base["metro_area"].notna() & (base["metro_area"] != "multi-location")].copy()
    print(f"  after metro filter: {len(ga):,} rows")

    # --------------------------------------------------------------
    # STEP 1: Metro-level metrics for metros with >=50 per period
    # --------------------------------------------------------------
    print("[T17] Computing per-metro metrics (arshkon-only vs scraped)...")

    # Using pooled 2024 baseline per T07 feasibility recommendation
    def metro_profile(d, period_label):
        grp = d.groupby("metro_area")
        out = grp.agg(
            n=("uid", "size"),
            entry_j1=("j1", "mean"),
            entry_j2=("j2", "mean"),
            entry_j3=("j3", "mean"),
            ai_strict=("ai_strict", "mean"),
            ai_broad=("ai_broad", "mean"),
            org_scope_per_1k=("org_scope_per_1k", "mean"),
            median_desc_length=("description_length", "median"),
            median_tech_count=("tech_count", "median"),
            mean_tech_count=("tech_count", "mean"),
            agg_rate=("is_aggregator", "mean"),
            specialist_rate=("is_specialist", "mean"),
        ).reset_index()
        out["period"] = period_label
        return out

    ga_24 = ga[ga["period"] == "2024"]
    ga_sc = ga[ga["period"] == "scraped"]
    ga_arsh = ga[ga["source"] == "kaggle_arshkon"]

    m_pool = metro_profile(ga_24, "pooled_2024")
    m_arsh = metro_profile(ga_arsh, "arshkon")
    m_sc = metro_profile(ga_sc, "scraped_2026")
    m_all = pd.concat([m_pool, m_arsh, m_sc], ignore_index=True)
    m_all.to_csv(OUT_TBL / "01_metro_profiles_all_periods.csv", index=False)

    # Metros with >=50 in pooled 2024 AND >=50 in scraped
    metros_pool_ok = set(m_pool[m_pool["n"] >= 50]["metro_area"])
    metros_sc_ok = set(m_sc[m_sc["n"] >= 50]["metro_area"])
    metros_arsh_ok = set(m_arsh[m_arsh["n"] >= 50]["metro_area"])

    metros_primary = sorted(metros_pool_ok & metros_sc_ok)
    metros_sec = sorted(metros_arsh_ok & metros_sc_ok)
    print(f"  metros with >=50 pooled-2024 & >=50 scraped: {len(metros_primary)}")
    print(f"  metros with >=50 arshkon-only & >=50 scraped: {len(metros_sec)}")

    # Build delta table (primary: pooled_2024 vs scraped)
    def build_delta(mA, mB, metros):
        keepA = mA[mA["metro_area"].isin(metros)].copy()
        keepB = mB[mB["metro_area"].isin(metros)].copy()
        merged = keepA.merge(
            keepB, on="metro_area", suffixes=("_A", "_B"), how="inner"
        )
        for k in [
            "n", "entry_j1", "entry_j2", "entry_j3",
            "ai_strict", "ai_broad",
            "org_scope_per_1k", "median_desc_length", "mean_tech_count",
            "agg_rate", "specialist_rate",
        ]:
            merged[f"d_{k}"] = merged[f"{k}_B"] - merged[f"{k}_A"]
        return merged

    delta_primary = build_delta(m_pool, m_sc, metros_primary)
    delta_primary.to_csv(OUT_TBL / "02_metro_deltas_pooled_primary.csv", index=False)

    delta_arsh = build_delta(m_arsh, m_sc, metros_sec)
    delta_arsh.to_csv(OUT_TBL / "03_metro_deltas_arshkon.csv", index=False)

    # --------------------------------------------------------------
    # STEP 2: Rank metros by magnitude of change per metric
    # --------------------------------------------------------------
    print("[T17] Ranking metros by change magnitude...")

    def ranked_table(delta, metric, top_k=10):
        out = delta[["metro_area", f"{metric}_A", f"{metric}_B", f"d_{metric}", "n_A", "n_B"]].copy()
        out = out.sort_values(f"d_{metric}", ascending=False)
        return out

    rank_ai_strict = ranked_table(delta_primary, "ai_strict")
    rank_ai_strict.to_csv(OUT_TBL / "04_metro_rank_ai_strict.csv", index=False)
    print("Top 10 metros by AI-strict surge (pooled-2024 -> scraped):")
    print(rank_ai_strict.head(10).to_string(index=False))

    rank_entry_j2 = ranked_table(delta_primary, "entry_j2")
    rank_entry_j2.to_csv(OUT_TBL / "05_metro_rank_entry_j2.csv", index=False)
    print("\nTop 10 metros by J2 entry decline (ascending):")
    print(rank_entry_j2.sort_values("d_entry_j2").head(10).to_string(index=False))
    rank_entry_j3 = ranked_table(delta_primary, "entry_j3")
    rank_entry_j3.to_csv(OUT_TBL / "06_metro_rank_entry_j3.csv", index=False)

    rank_length = ranked_table(delta_primary, "median_desc_length")
    rank_length.to_csv(OUT_TBL / "07_metro_rank_length.csv", index=False)
    rank_tech = ranked_table(delta_primary, "mean_tech_count")
    rank_tech.to_csv(OUT_TBL / "08_metro_rank_tech.csv", index=False)
    rank_orgscope = ranked_table(delta_primary, "org_scope_per_1k")
    rank_orgscope.to_csv(OUT_TBL / "09_metro_rank_orgscope.csv", index=False)

    # --------------------------------------------------------------
    # STEP 3: Tech-hub vs non-hub concentration of change
    # --------------------------------------------------------------
    print("[T17] Tech-hub vs non-hub aggregate comparison...")
    HUBS = {
        "San Francisco Bay Area",
        "New York Metro",
        "Seattle Metro",
        "Austin Metro",
        "Los Angeles Metro",
        "Boston Metro",
    }
    # Use whatever metro-name canonicalization matched above
    print("  metros in 2024 distinct list:", m_pool["metro_area"].dropna().head(30).tolist())

    # Map common variations
    def is_hub(metro):
        if pd.isna(metro):
            return False
        s = str(metro).lower()
        hub_keys = ["san francisco", "bay area", "new york", "seattle", "austin", "los angeles", "boston"]
        return any(k in s for k in hub_keys)

    delta_primary["is_hub"] = delta_primary["metro_area"].apply(is_hub).astype(int)
    hub_split = delta_primary.groupby("is_hub").agg(
        n_metros=("metro_area", "size"),
        mean_d_ai_strict=("d_ai_strict", "mean"),
        mean_d_entry_j1=("d_entry_j1", "mean"),
        mean_d_entry_j2=("d_entry_j2", "mean"),
        mean_d_entry_j3=("d_entry_j3", "mean"),
        mean_d_tech=("d_mean_tech_count", "mean"),
        mean_d_length=("d_median_desc_length", "mean"),
    ).reset_index()
    hub_split.to_csv(OUT_TBL / "10_hub_vs_nonhub_aggregate.csv", index=False)
    print(hub_split.to_string(index=False))

    # --------------------------------------------------------------
    # STEP 4: Metro-level correlations (AI surge vs entry decline)
    # --------------------------------------------------------------
    print("[T17] Metro-level correlations across primary panel...")
    corrs = []
    for x, y in [
        ("d_ai_strict", "d_entry_j1"),
        ("d_ai_strict", "d_entry_j2"),
        ("d_ai_strict", "d_entry_j3"),
        ("d_ai_strict", "d_mean_tech_count"),
        ("d_ai_strict", "d_median_desc_length"),
        ("d_ai_broad", "d_entry_j1"),
        ("d_ai_broad", "d_entry_j2"),
        ("d_ai_broad", "d_entry_j3"),
        ("d_median_desc_length", "d_mean_tech_count"),
        ("d_ai_strict", "d_org_scope_per_1k"),
    ]:
        a = delta_primary[x].dropna()
        b = delta_primary[y].dropna()
        common = delta_primary[[x, y]].dropna()
        if len(common) < 5:
            corrs.append({"x": x, "y": y, "n": int(len(common)), "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan})
            continue
        pr, pp = pearsonr(common[x], common[y])
        sr, sp = spearmanr(common[x], common[y])
        corrs.append({"x": x, "y": y, "n": int(len(common)),
                      "pearson_r": float(pr), "pearson_p": float(pp),
                      "spearman_r": float(sr), "spearman_p": float(sp)})
    corrs_df = pd.DataFrame(corrs)
    corrs_df.to_csv(OUT_TBL / "11_metro_correlations.csv", index=False)
    print(corrs_df.to_string(index=False))

    # --------------------------------------------------------------
    # STEP 5: Archetype geographic distribution
    # --------------------------------------------------------------
    print("[T17] Archetype x metro geographic distribution...")
    arche_only = ga[ga["archetype_name"].notna()].copy()
    print(f"  archetype-labeled rows with metro: {len(arche_only):,}")

    # For primary metros (>=50 in both periods)
    arche_only = arche_only[arche_only["metro_area"].isin(metros_primary)]

    # period-specific shares per metro x archetype
    grp = arche_only.groupby(["metro_area", "period", "archetype_name"]).size().reset_index(name="n")
    totals = arche_only.groupby(["metro_area", "period"]).size().reset_index(name="n_metro_period")
    grp = grp.merge(totals, on=["metro_area", "period"])
    grp["share"] = grp["n"] / grp["n_metro_period"]
    grp.to_csv(OUT_TBL / "12_archetype_by_metro_period.csv", index=False)

    # Identify ML/AI-heavy metros in 2026: top metros by ai_ml_engineering share in scraped
    ml_share = grp[
        (grp["period"] == "scraped") & (grp["archetype_name"] == "ai_ml_engineering")
    ].copy().sort_values("share", ascending=False)
    ml_share.to_csv(OUT_TBL / "13_ml_archetype_share_by_metro_2026.csv", index=False)
    print("Top metros by ML/AI archetype share (scraped 2026):")
    print(ml_share.head(10).to_string(index=False))

    # Compare 2024 vs 2026 archetype share per metro
    pivot = grp.pivot_table(
        index=["metro_area", "archetype_name"],
        columns="period",
        values="share",
        aggfunc="sum",
    ).reset_index()
    if "2024" in pivot.columns and "scraped" in pivot.columns:
        pivot["d_share"] = pivot["scraped"] - pivot["2024"]
        pivot = pivot.sort_values("d_share", ascending=False)
        pivot.to_csv(OUT_TBL / "14_archetype_share_change_by_metro.csv", index=False)

    # --------------------------------------------------------------
    # STEP 6: Metro heatmap
    # --------------------------------------------------------------
    print("[T17] Drawing metro heatmap...")
    heat_cols = [
        "d_entry_j1", "d_entry_j2", "d_entry_j3",
        "d_ai_strict", "d_ai_broad",
        "d_median_desc_length", "d_mean_tech_count", "d_org_scope_per_1k"
    ]
    heat_df = delta_primary[["metro_area"] + heat_cols].set_index("metro_area")
    # Sort by d_ai_strict descending
    heat_df = heat_df.sort_values("d_ai_strict", ascending=False)

    # Normalize each column to [-1, 1] by its max absolute value
    heat_norm = heat_df.copy()
    for c in heat_cols:
        maxabs = heat_norm[c].abs().max()
        if maxabs > 0:
            heat_norm[c] = heat_norm[c] / maxabs

    fig, ax = plt.subplots(figsize=(10, max(5, 0.3 * len(heat_df))))
    im = ax.imshow(heat_norm.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_yticks(np.arange(len(heat_df)))
    ax.set_yticklabels(heat_df.index, fontsize=8)
    ax.set_xticks(np.arange(len(heat_cols)))
    ax.set_xticklabels(heat_cols, rotation=35, ha="right", fontsize=9)
    # annotate with raw values
    for i in range(len(heat_df)):
        for j, c in enumerate(heat_cols):
            v = heat_df.iloc[i, j]
            if pd.notna(v):
                label = f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=6.5,
                        color="white" if abs(heat_norm.iloc[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="normalized Δ (−1…1)")
    ax.set_title(f"Metro x metric change heatmap — pooled 2024 → scraped 2026 (n={len(heat_df)})")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "01_metro_heatmap.png", dpi=150)
    plt.close()

    # Fig 2: Scatter: d_ai_strict vs d_entry_j2
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(delta_primary["d_ai_strict"], delta_primary["d_entry_j2"],
               s=np.sqrt(delta_primary["n_B"]) * 2, alpha=0.7, color="steelblue")
    for i, row in delta_primary.iterrows():
        ax.annotate(row["metro_area"][:25], (row["d_ai_strict"], row["d_entry_j2"]),
                    fontsize=7, alpha=0.7, xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Δ AI-strict share")
    ax.set_ylabel("Δ J2 entry share")
    r = corrs_df[(corrs_df["x"] == "d_ai_strict") & (corrs_df["y"] == "d_entry_j2")].iloc[0]
    ax.set_title(f"Metro-level correlation: AI surge vs entry change (pearson r={r['pearson_r']:.3f}, p={r['pearson_p']:.3f}, n={int(r['n'])})")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "02_ai_vs_entry_scatter.png", dpi=150)
    plt.close()

    # Fig 3: Archetype share pivot heatmap (2026 only, top metros)
    if len(ml_share) > 0:
        # Pivot: metros x archetypes (scraped 2026 only)
        piv26 = grp[grp["period"] == "scraped"].pivot_table(
            index="metro_area", columns="archetype_name", values="share", aggfunc="sum",
        ).fillna(0)
        # Top 15 metros by total n in scraped + top 10 archetypes by total share
        top_metros = (
            arche_only[arche_only["period"] == "scraped"]["metro_area"].value_counts().head(20).index
        )
        top_archs = piv26.sum(axis=0).sort_values(ascending=False).head(12).index
        sub = piv26.loc[piv26.index.isin(top_metros), top_archs]
        # Sort metros by ml/ai share if present
        if "ai_ml_engineering" in sub.columns:
            sub = sub.sort_values("ai_ml_engineering", ascending=False)

        fig, ax = plt.subplots(figsize=(12, max(5, 0.4 * len(sub))))
        im = ax.imshow(sub.values, aspect="auto", cmap="viridis")
        ax.set_yticks(np.arange(len(sub)))
        ax.set_yticklabels(sub.index, fontsize=8)
        ax.set_xticks(np.arange(len(sub.columns)))
        ax.set_xticklabels(sub.columns, rotation=45, ha="right", fontsize=8)
        for i in range(len(sub)):
            for j in range(len(sub.columns)):
                v = sub.iloc[i, j]
                ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.15 else "black")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="share of metro 2026 postings")
        ax.set_title("Archetype x metro share — scraped 2026 (primary panel)")
        plt.tight_layout()
        plt.savefig(OUT_FIG / "03_archetype_metro_2026.png", dpi=150)
        plt.close()

    # --------------------------------------------------------------
    # STEP 7: Aggregator / specialist exclusion sensitivity on metro AI
    # --------------------------------------------------------------
    print("[T17] Sensitivity: metro AI surge without aggregators/specialists...")
    ga_noagg = ga[ga["is_aggregator"] == False].copy()
    m_pool_na = metro_profile(ga_noagg[ga_noagg["period"] == "2024"], "pooled_2024_noagg")
    m_sc_na = metro_profile(ga_noagg[ga_noagg["period"] == "scraped"], "scraped_2026_noagg")
    metros_ok_na = set(m_pool_na[m_pool_na["n"] >= 50]["metro_area"]) & set(m_sc_na[m_sc_na["n"] >= 50]["metro_area"])
    delta_na = build_delta(m_pool_na, m_sc_na, metros_ok_na)
    delta_na.to_csv(OUT_TBL / "15_metro_deltas_no_aggregator.csv", index=False)

    ga_nspec = ga[ga["is_specialist"] == 0].copy()
    m_pool_ns = metro_profile(ga_nspec[ga_nspec["period"] == "2024"], "pooled_2024_nospec")
    m_sc_ns = metro_profile(ga_nspec[ga_nspec["period"] == "scraped"], "scraped_2026_nospec")
    metros_ok_ns = set(m_pool_ns[m_pool_ns["n"] >= 50]["metro_area"]) & set(m_sc_ns[m_sc_ns["n"] >= 50]["metro_area"])
    delta_ns = build_delta(m_pool_ns, m_sc_ns, metros_ok_ns)
    delta_ns.to_csv(OUT_TBL / "16_metro_deltas_no_specialist.csv", index=False)

    # Summary
    print("[T17] Summary of primary AI / entry ranking:")
    top_ai = rank_ai_strict.head(3)[["metro_area", "d_ai_strict"]]
    top_entry_decline = rank_entry_j2.sort_values("d_entry_j2").head(3)[["metro_area", "d_entry_j2"]]
    print("  Top 3 AI surge:", top_ai.values.tolist())
    print("  Top 3 entry-J2 decline:", top_entry_decline.values.tolist())
    summary = {
        "metros_primary": metros_primary,
        "n_metros_primary": len(metros_primary),
        "top3_ai_surge": top_ai.to_dict("records"),
        "top3_entry_decline_j2": top_entry_decline.to_dict("records"),
        "top3_entry_decline_j3": rank_entry_j3.sort_values("d_entry_j3").head(3)[["metro_area", "d_entry_j3"]].to_dict("records"),
        "corrs": corrs,
    }
    with open(OUT_TBL / "17_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("[T17] Done.")


if __name__ == "__main__":
    main()
