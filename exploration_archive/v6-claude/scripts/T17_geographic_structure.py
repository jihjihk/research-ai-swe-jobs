"""T17 — Geographic market structure.

Per-metro SWE market metrics for metros with >=50 postings in BOTH periods
(arshkon 2024 vs scraped 2026). Tests the Alt C prediction: if the AI surge is
metro-uniform and uncorrelated with entry-share movement, that's evidence the
AI surge is an instrument-side restructuring rather than a junior-driven one.

Outputs under exploration/{tables,figures}/T17/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNIFIED = ROOT / "data" / "unified.parquet"
SHARED = ROOT / "exploration" / "artifacts" / "shared"
TECHM = SHARED / "swe_tech_matrix.parquet"
ARCH = SHARED / "swe_archetype_labels.parquet"
TBL = ROOT / "exploration" / "tables" / "T17"
FIG = ROOT / "exploration" / "figures" / "T17"
TBL.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

AI_UNION = [
    "machine_learning", "deep_learning", "nlp", "computer_vision", "llm",
    "langchain", "langgraph", "rag", "vector_db", "pinecone", "chromadb",
    "huggingface", "openai_api", "claude_api", "prompt_engineering",
    "fine_tuning", "mcp", "agents_framework", "gpt", "transformer_arch",
    "embedding", "copilot", "cursor_tool", "claude_tool",
]
AI_TOOLS_ONLY = ["copilot", "cursor_tool", "claude_tool", "openai_api", "claude_api", "mcp"]
TECH_COLS = None  # will load

SCOPE_RE = re.compile(r"\b(?:end[- ]to[- ]end|cross[- ]functional)\b", re.I)

QUALIFYING_MIN = 50  # per period


def fetch_base():
    con = duckdb.connect()
    ai_sum = " + ".join([f"CAST(t.{c} AS INTEGER)" for c in AI_UNION])
    ai_tools_sum = " + ".join([f"CAST(t.{c} AS INTEGER)" for c in AI_TOOLS_ONLY])
    q = f"""
    SELECT
        u.uid, u.source, u.company_name_canonical AS company,
        u.is_aggregator, u.is_remote, u.is_multi_location,
        u.seniority_final, u.yoe_extracted,
        u.description, u.description_core_llm, u.llm_extraction_coverage,
        u.metro_area,
        COALESCE(({ai_sum}), 0) AS ai_tech_count,
        CAST(({ai_sum}) > 0 AS INTEGER) AS ai_any,
        CAST(({ai_tools_sum}) > 0 AS INTEGER) AS ai_tools_any,
        t.* EXCLUDE (uid) AS _
    FROM read_parquet('{UNIFIED.as_posix()}') u
    LEFT JOIN read_parquet('{TECHM.as_posix()}') t USING (uid)
    WHERE u.is_swe = true
      AND u.source_platform = 'linkedin'
      AND u.is_english = true
      AND u.date_flag = 'ok'
      AND u.source IN ('kaggle_arshkon', 'scraped')
    """
    # The `t.* EXCLUDE (uid)` will flatten; but we really want to count tech mentions for tech_count.
    # Simpler approach: compute tech_count separately.
    q2 = f"""
    WITH base AS (
      SELECT
        u.uid, u.source, u.company_name_canonical AS company,
        u.is_aggregator, u.is_remote, u.is_multi_location,
        u.seniority_final, u.yoe_extracted,
        u.description, u.description_core_llm, u.llm_extraction_coverage,
        u.metro_area
      FROM read_parquet('{UNIFIED.as_posix()}') u
      WHERE u.is_swe = true
        AND u.source_platform = 'linkedin'
        AND u.is_english = true
        AND u.date_flag = 'ok'
        AND u.source IN ('kaggle_arshkon', 'scraped')
    ),
    t AS (SELECT * FROM read_parquet('{TECHM.as_posix()}'))
    SELECT b.*,
           COALESCE(({ai_sum}), 0) AS ai_tech_count,
           CAST(COALESCE(({ai_sum}), 0) > 0 AS INTEGER) AS ai_any,
           CAST(COALESCE(({ai_tools_sum}), 0) > 0 AS INTEGER) AS ai_tools_any
    FROM base b
    LEFT JOIN t USING (uid)
    """
    df = con.execute(q2).fetchdf()
    # Also fetch tech_count (count of all tech columns) per uid
    cols_q = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{TECHM.as_posix()}')").fetchall()
    tech_cols = [row[0] for row in cols_q if row[0] != "uid"]
    sumq = " + ".join([f'CAST("{c}" AS INTEGER)' for c in tech_cols])
    tc = con.execute(f"SELECT uid, ({sumq}) AS tech_count FROM read_parquet('{TECHM.as_posix()}')").fetchdf()
    df = df.merge(tc, on="uid", how="left")
    con.close()
    return df


def attach_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["period_label"] = np.where(df["source"] == "kaggle_arshkon", "y2024", "y2026")
    df["entry_final"] = (df["seniority_final"] == "entry").astype(int)
    df["entry_known"] = (~df["seniority_final"].isin(["unknown", None])).astype(int)
    df["yoe_le2"] = (df["yoe_extracted"] <= 2).astype(int)
    df["yoe_known"] = df["yoe_extracted"].notna().astype(int)
    df["desc_len"] = df["description"].fillna("").str.len()
    txt = df["description_core_llm"].where(
        df["llm_extraction_coverage"] == "labeled", df["description"]
    ).fillna("")
    df["scope_clean_any"] = txt.str.contains(SCOPE_RE, regex=True).astype(int)
    return df


def per_metro_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metro_df = df[df["metro_area"].notna() & (df["is_multi_location"] == False)]
    rows = []
    for metro, g in metro_df.groupby("metro_area"):
        by_p = g.groupby("period_label")
        n24 = (g["period_label"] == "y2024").sum()
        n26 = (g["period_label"] == "y2026").sum()
        if n24 < QUALIFYING_MIN or n26 < QUALIFYING_MIN:
            continue
        def pm(sub, col, filt=None):
            if filt is not None:
                sub = sub[sub[filt] == 1]
            return sub[col].mean() if len(sub) else np.nan
        d24 = g[g["period_label"] == "y2024"]
        d26 = g[g["period_label"] == "y2026"]
        rows.append({
            "metro": metro,
            "n_2024": int(n24),
            "n_2026": int(n26),
            "entry_final_2024": pm(d24[d24["entry_known"] == 1], "entry_final"),
            "entry_final_2026": pm(d26[d26["entry_known"] == 1], "entry_final"),
            "yoe_le2_2024": pm(d24[d24["yoe_known"] == 1], "yoe_le2"),
            "yoe_le2_2026": pm(d26[d26["yoe_known"] == 1], "yoe_le2"),
            "ai_any_2024": d24["ai_any"].mean(),
            "ai_any_2026": d26["ai_any"].mean(),
            "ai_tools_any_2024": d24["ai_tools_any"].mean(),
            "ai_tools_any_2026": d26["ai_tools_any"].mean(),
            "scope_2024": d24["scope_clean_any"].mean(),
            "scope_2026": d26["scope_clean_any"].mean(),
            "desc_len_2024": d24["desc_len"].median(),
            "desc_len_2026": d26["desc_len"].median(),
            "tech_count_2024": d24["tech_count"].median(),
            "tech_count_2026": d26["tech_count"].median(),
            "remote_share_2026": d26["is_remote"].mean(),
            "agg_share_2024": d24["is_aggregator"].mean(),
            "agg_share_2026": d26["is_aggregator"].mean(),
        })
    out = pd.DataFrame(rows)
    for base in ["entry_final", "yoe_le2", "ai_any", "ai_tools_any", "scope",
                 "desc_len", "tech_count"]:
        out[f"d_{base}"] = out[f"{base}_2026"] - out[f"{base}_2024"]
    return out.sort_values("n_2026", ascending=False).reset_index(drop=True)


def archetype_by_metro(df: pd.DataFrame, metros: list[str]):
    con = duckdb.connect()
    arch = con.execute(
        f"SELECT uid, archetype_name FROM read_parquet('{ARCH.as_posix()}')"
    ).fetchdf()
    con.close()
    merged = df.merge(arch, on="uid", how="left")
    merged = merged[merged["archetype_name"].notna() &
                    (merged["archetype_name"] != "No text / raw-only (unlabeled)") &
                    merged["metro_area"].isin(metros) &
                    (merged["is_multi_location"] == False)]
    tab = merged.groupby(["metro_area", "period_label", "archetype_name"]).size().reset_index(name="n")
    totals = tab.groupby(["metro_area", "period_label"])["n"].transform("sum")
    tab["share"] = tab["n"] / totals
    return tab


def main():
    print("Fetching base...")
    df = fetch_base()
    df = attach_derived(df)
    print(f"Rows: {len(df):,}")
    n_multi = df[(df.source == 'scraped') & (df.is_multi_location == True)].shape[0]
    print(f"SWE multi-location rows (2026 scraped, excluded from metro rollups): {n_multi:,}")

    # Primary — all rows
    metros_df = per_metro_metrics(df)
    metros_df.to_csv(TBL / "metro_metrics.csv", index=False)
    print(f"Qualifying metros (n>=50 per period): {len(metros_df)}")
    print(metros_df[["metro", "n_2024", "n_2026", "entry_final_2024", "entry_final_2026",
                      "ai_any_2024", "ai_any_2026", "scope_2024", "scope_2026",
                      "remote_share_2026"]].to_string(index=False))

    # Sensitivity — aggregator exclusion
    df_noagg = df[df["is_aggregator"] == False]
    metros_noagg = per_metro_metrics(df_noagg)
    metros_noagg.to_csv(TBL / "metro_metrics_no_aggregator.csv", index=False)

    # Sensitivity — company cap 50/company (for per-row rates aggregation)
    def cap(d, k=50):
        parts = []
        for (src, comp), g in d.groupby(["source", "company"]):
            if len(g) > k:
                g = g.sample(k, random_state=42)
            parts.append(g)
        return pd.concat(parts)
    df_capped = cap(df, 50)
    metros_capped = per_metro_metrics(df_capped)
    metros_capped.to_csv(TBL / "metro_metrics_cap50.csv", index=False)

    # Correlation analysis: d_ai_any vs d_entry_final (Pearson)
    def correlations(mdf, label):
        rows = []
        pairs = [("d_ai_any", "d_entry_final"),
                 ("d_ai_any", "d_yoe_le2"),
                 ("d_ai_any", "d_desc_len"),
                 ("d_ai_any", "d_scope"),
                 ("d_ai_tools_any", "d_entry_final"),
                 ("d_ai_tools_any", "d_yoe_le2"),
                 ("d_desc_len", "d_entry_final"),
                 ("d_scope", "d_entry_final")]
        for a, b in pairs:
            sub = mdf[[a, b]].dropna()
            if len(sub) < 3:
                continue
            pr, pp = pearsonr(sub[a], sub[b])
            sr, sp = spearmanr(sub[a], sub[b])
            rows.append({"spec": label, "x": a, "y": b, "n": len(sub),
                         "pearson_r": pr, "pearson_p": pp,
                         "spearman_r": sr, "spearman_p": sp})
        return rows
    corr_rows = []
    corr_rows += correlations(metros_df, "all_rows")
    corr_rows += correlations(metros_noagg, "no_aggregator")
    corr_rows += correlations(metros_capped, "cap50")
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(TBL / "metro_correlations.csv", index=False)
    print("\nMetro correlations:")
    print(corr_df.to_string(index=False))

    # Archetype by metro
    metros_list = metros_df["metro"].tolist()
    arch_tab = archetype_by_metro(df, metros_list)
    arch_tab.to_csv(TBL / "archetype_by_metro.csv", index=False)
    # Wide per-metro period distribution for heatmap
    arch_2026 = arch_tab[arch_tab.period_label == "y2026"].pivot(
        index="metro_area", columns="archetype_name", values="share").fillna(0)
    arch_2024 = arch_tab[arch_tab.period_label == "y2024"].pivot(
        index="metro_area", columns="archetype_name", values="share").fillna(0)
    arch_2026.to_csv(TBL / "archetype_shares_2026.csv")
    arch_2024.to_csv(TBL / "archetype_shares_2024.csv")

    # Concentration of archetypes across metros (over-index / under-index)
    # Compute each metro's archetype share relative to global 2026 distribution
    global_2026 = arch_tab[arch_tab.period_label == "y2026"].groupby("archetype_name")["n"].sum()
    global_2026 /= global_2026.sum()
    rel = arch_2026.div(global_2026, axis=1)
    rel.to_csv(TBL / "archetype_rel_index_2026.csv")

    # ---- Figures ----
    # 1. Metro heatmap: metros x delta metrics, z-normalized
    mm = metros_df.set_index("metro")[["d_entry_final", "d_yoe_le2", "d_ai_any",
                                         "d_ai_tools_any", "d_scope", "d_desc_len", "d_tech_count"]]
    mm = mm.dropna(how="all")
    Z = (mm - mm.mean()) / (mm.std() + 1e-9)
    fig, ax = plt.subplots(figsize=(8, max(5, 0.35 * len(mm))))
    im = ax.imshow(Z.values, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(mm.columns)))
    ax.set_xticklabels([c.replace("d_", "Δ") for c in mm.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(mm)))
    ax.set_yticklabels([m.replace(" Metro", "") for m in mm.index], fontsize=8)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            val = mm.values[i, j]
            if "desc_len" in mm.columns[j]:
                txt = f"{val:+.0f}"
            else:
                txt = f"{val:+.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5)
    plt.colorbar(im, ax=ax, label="z-score across metros")
    ax.set_title("T17 — Metro change-magnitude heatmap (18 qualifying metros)")
    plt.tight_layout()
    plt.savefig(FIG / "metro_heatmap.png", dpi=150)
    plt.close()

    # 2. Scatter: d_ai_any vs d_entry_final (primary correlation test)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    x = metros_df["d_ai_any"].values
    y = metros_df["d_entry_final"].values
    ok = ~(np.isnan(x) | np.isnan(y))
    ax.scatter(x[ok], y[ok], s=70, color="#4c78a8", edgecolor="black")
    for i, metro in enumerate(metros_df["metro"]):
        if ok[i]:
            short = metro.replace(" Metro", "").replace(" Bay Area", "-Bay")
            ax.annotate(short, (x[i], y[i]), fontsize=7, xytext=(4, 2), textcoords="offset points")
    pr, pp = pearsonr(x[ok], y[ok])
    # Fit line
    m_, b_ = np.polyfit(x[ok], y[ok], 1)
    xs = np.linspace(x[ok].min(), x[ok].max(), 10)
    ax.plot(xs, m_ * xs + b_, "r--", alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Δ AI prevalence (broad union) 2024→2026")
    ax.set_ylabel("Δ entry share (seniority_final) 2024→2026")
    ax.set_title(f"T17 — Metro-level: AI surge vs entry-share change\nPearson r={pr:+.3f}, p={pp:.3f} (n={ok.sum()})")
    plt.tight_layout()
    plt.savefig(FIG / "correlation_ai_vs_entry.png", dpi=150)
    plt.close()

    # 3. Archetype share by metro — grouped bar (top 6 archetypes)
    top_arch = arch_2026.sum().sort_values(ascending=False).head(6).index.tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.12
    xs = np.arange(len(arch_2026))
    for i, a in enumerate(top_arch):
        ax.bar(xs + i * width, arch_2026[a].values, width, label=a[:20])
    ax.set_xticks(xs + (len(top_arch) - 1) * width / 2)
    ax.set_xticklabels([m.replace(" Metro", "").replace(" Bay Area", "-Bay")
                        for m in arch_2026.index], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Share in 2026")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("T17 — Top-6 archetype shares by metro (2026)")
    plt.tight_layout()
    plt.savefig(FIG / "archetype_by_metro.png", dpi=150)
    plt.close()

    # 4. Remote pool vs metro-assigned pool comparison
    # Remote postings have metro_area=NULL in this schema; report the pool separately.
    scraped = df[df.source == "scraped"]
    rem = scraped[scraped["is_remote"] == True]
    loc = scraped[(scraped["is_remote"] == False) & scraped["metro_area"].notna() &
                   (scraped["is_multi_location"] == False)]
    remote_profile = pd.DataFrame([
        {
            "bucket": "metro_assigned",
            "n": len(loc),
            "entry_final": loc.loc[loc["entry_known"] == 1, "entry_final"].mean(),
            "yoe_le2": loc.loc[loc["yoe_known"] == 1, "yoe_le2"].mean(),
            "ai_any": loc["ai_any"].mean(),
            "ai_tools_any": loc["ai_tools_any"].mean(),
            "scope_clean_any": loc["scope_clean_any"].mean(),
            "desc_len_median": loc["desc_len"].median(),
            "tech_count_median": loc["tech_count"].median(),
        },
        {
            "bucket": "remote_only",
            "n": len(rem),
            "entry_final": rem.loc[rem["entry_known"] == 1, "entry_final"].mean(),
            "yoe_le2": rem.loc[rem["yoe_known"] == 1, "yoe_le2"].mean(),
            "ai_any": rem["ai_any"].mean(),
            "ai_tools_any": rem["ai_tools_any"].mean(),
            "scope_clean_any": rem["scope_clean_any"].mean(),
            "desc_len_median": rem["desc_len"].median(),
            "tech_count_median": rem["tech_count"].median(),
        },
    ])
    remote_profile.to_csv(TBL / "remote_vs_metro_profile.csv", index=False)
    print("\nRemote vs metro-assigned (2026 scraped):")
    print(remote_profile.to_string(index=False))
    fig, ax = plt.subplots(figsize=(7, 4))
    metrics = ["entry_final", "yoe_le2", "ai_any", "ai_tools_any", "scope_clean_any"]
    labels = ["entry\n(seniority_final)", "entry\n(YOE≤2)", "AI any", "AI tools", "scope"]
    xs = np.arange(len(metrics))
    w = 0.35
    ax.bar(xs - w/2, [remote_profile.iloc[0][m] for m in metrics], w,
           label=f"metro-assigned (n={int(remote_profile.iloc[0].n)})", color="#4c78a8")
    ax.bar(xs + w/2, [remote_profile.iloc[1][m] for m in metrics], w,
           label=f"remote-only (n={int(remote_profile.iloc[1].n)})", color="#e45756")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("share")
    ax.set_title("T17 — Metro-assigned vs remote-only pool (2026 scraped)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "remote_vs_metro_profile.png", dpi=150)
    plt.close()

    meta = {
        "qualifying_metros": len(metros_df),
        "n_multi_location_scraped": int(n_multi),
        "primary_correlations": corr_df[corr_df.spec == "all_rows"].to_dict(orient="records"),
    }
    def cast(x):
        if isinstance(x, dict):
            return {k: cast(v) for k, v in x.items()}
        if isinstance(x, list):
            return [cast(v) for v in x]
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
