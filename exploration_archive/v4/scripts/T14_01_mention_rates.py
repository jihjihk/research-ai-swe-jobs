"""T14 Step 1-2, 4-6: Technology mention rates by period x seniority, trajectory, stack diversity,
AI integration pattern. Produces heatmaps and tables.

Period convention:
  - 2024 = kaggle_arshkon (Apr) + kaggle_asaniczka (Jan) pooled
  - 2026 = scraped (Mar + Apr) pooled

Seniority operational column: seniority_3level (junior/mid/senior) — derived from seniority_final.
Also reports "best_available" when the combined column is non-null.

Outputs:
  tables/T14/tech_mention_rates.csv            Wide table of % rates
  tables/T14/tech_trajectory.csv               Rising/stable/declining classification
  tables/T14/stack_diversity.csv               Median/p25/p75 tech count by period x seniority
  tables/T14/ai_integration_cooccurrence.csv   Technologies co-occurring with AI among AI-mentioning postings
  tables/T14/ai_density_check.csv              Raw tech count and density for AI vs non-AI
  figures/T14/tech_shift_heatmap.png           Heatmap of rising/declining techs
  figures/T14/stack_diversity_hist.png         Hist of tech counts by period
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path("/home/jihgaboot/gabor/job-research")
SHARED = ROOT / "exploration/artifacts/shared"
TABLES = ROOT / "exploration/tables/T14"
FIGS = ROOT / "exploration/figures/T14"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


def load_base():
    # Join cleaned text metadata with combined best-available seniority from unified.parquet
    con = duckdb.connect()
    meta = con.execute(f"""
        SELECT c.uid, c.source, c.period, c.seniority_3level, c.seniority_final,
               c.is_aggregator, c.yoe_extracted, c.swe_classification_tier,
               CASE
                 WHEN u.llm_classification_coverage='labeled' THEN u.seniority_llm
                 WHEN u.llm_classification_coverage='rule_sufficient' THEN u.seniority_final
                 ELSE NULL
               END AS seniority_best_available,
               LENGTH(c.description_cleaned) AS text_len
        FROM read_parquet('{SHARED}/swe_cleaned_text.parquet') c
        LEFT JOIN read_parquet('{ROOT}/data/unified.parquet') u ON c.uid = u.uid
    """).df()
    # Pooled period
    meta["period2"] = meta["period"].map(
        {"2024-01": "2024", "2024-04": "2024", "2026-03": "2026", "2026-04": "2026"}
    )
    # Collapse combined seniority to 3-level for analysis
    sb = meta["seniority_best_available"]
    mapping = {
        "entry": "junior",
        "associate": "mid",
        "mid-senior": "senior",
        "director": "senior",
        "unknown": "unknown",
    }
    meta["best_available_3level"] = sb.map(mapping)
    return meta


def load_tech():
    return pq.read_table(SHARED / "swe_tech_matrix.parquet").to_pandas()


def main():
    print("Loading metadata and tech matrix...")
    meta = load_base()
    tech = load_tech()
    print(f"  meta rows: {len(meta)}  tech rows: {len(tech)}")

    # Aggregator exclusion (essential sensitivity (a))
    meta_noagg = meta[~meta["is_aggregator"].fillna(False)].copy()
    tech_noagg = tech[tech["uid"].isin(set(meta_noagg["uid"]))].copy()
    print(f"  after aggregator exclusion: {len(meta_noagg)}")

    # Merge
    df = meta_noagg.merge(tech_noagg, on="uid", how="inner")
    tech_cols = [c for c in tech.columns if c != "uid"]
    print(f"  merged: {len(df)}  tech cols: {len(tech_cols)}")

    # ---------- Step 2: Mention rates by period x seniority (3level) ----------
    # Use 3level as primary; exclude unknown from stratified view but keep overall.
    groups = [
        ("2024", "junior"),
        ("2024", "mid"),
        ("2024", "senior"),
        ("2024", "ALL"),
        ("2026", "junior"),
        ("2026", "mid"),
        ("2026", "senior"),
        ("2026", "ALL"),
    ]

    rows = []
    for period, sen in groups:
        if sen == "ALL":
            m = df[df["period2"] == period]
        else:
            m = df[(df["period2"] == period) & (df["seniority_3level"] == sen)]
        n = len(m)
        row = {"period": period, "seniority_3level": sen, "n": n}
        for t in tech_cols:
            row[t] = m[t].mean() if n > 0 else np.nan
        rows.append(row)
    mention_df = pd.DataFrame(rows)
    mention_df.to_csv(TABLES / "tech_mention_rates.csv", index=False)
    print(f"  wrote tech_mention_rates.csv ({len(mention_df)} rows)")

    # ---------- Step 4: Trajectory classification ----------
    # Rate in 2024 (ALL) vs 2026 (ALL); classify by absolute and relative change.
    r2024 = mention_df[(mention_df["period"] == "2024") & (mention_df["seniority_3level"] == "ALL")].iloc[0]
    r2026 = mention_df[(mention_df["period"] == "2026") & (mention_df["seniority_3level"] == "ALL")].iloc[0]

    traj = []
    for t in tech_cols:
        a = float(r2024[t])
        b = float(r2026[t])
        abs_delta = b - a
        rel_delta = (b - a) / a if a > 0 else np.nan
        traj.append({"tech": t, "rate_2024": a, "rate_2026": b,
                     "abs_delta_pp": abs_delta * 100, "rel_delta": rel_delta})
    traj_df = pd.DataFrame(traj)
    # Classify: rising if +>=2pp AND >=30% rel, declining if <=-2pp AND <=-25% rel, else stable.
    def classify(row):
        a, b = row["rate_2024"], row["rate_2026"]
        d = row["abs_delta_pp"]
        r = row["rel_delta"]
        if pd.isna(r):
            return "stable"
        if d >= 2 and r >= 0.30:
            return "rising"
        if d <= -2 and r <= -0.25:
            return "declining"
        return "stable"
    traj_df["trajectory"] = traj_df.apply(classify, axis=1)
    traj_df = traj_df.sort_values("abs_delta_pp", ascending=False).reset_index(drop=True)
    traj_df.to_csv(TABLES / "tech_trajectory.csv", index=False)
    print(f"  wrote tech_trajectory.csv  rising={int((traj_df.trajectory=='rising').sum())} "
          f"declining={int((traj_df.trajectory=='declining').sum())} "
          f"stable={int((traj_df.trajectory=='stable').sum())}")

    # ---------- Heatmap of top movers ----------
    # Combine top 20 rising and top 15 declining and top 15 stable with high baseline
    top_rise = traj_df[traj_df.trajectory == "rising"].nlargest(20, "abs_delta_pp")
    top_decl = traj_df[traj_df.trajectory == "declining"].nsmallest(15, "abs_delta_pp")
    high_stable = traj_df[(traj_df.trajectory == "stable") & (traj_df["rate_2024"] >= 0.10)].nlargest(15, "rate_2024")
    heatmap_techs = pd.concat([top_rise, high_stable, top_decl])["tech"].tolist()

    # Build heatmap matrix: rows = techs, cols = period x sen (3level only, skip unknown)
    hm_cols = [("2024", "junior"), ("2024", "mid"), ("2024", "senior"),
               ("2026", "junior"), ("2026", "mid"), ("2026", "senior")]
    mat = np.zeros((len(heatmap_techs), len(hm_cols)))
    for j, (p, s) in enumerate(hm_cols):
        row = mention_df[(mention_df["period"] == p) & (mention_df["seniority_3level"] == s)].iloc[0]
        for i, t in enumerate(heatmap_techs):
            mat[i, j] = row[t] * 100  # percent

    fig, ax = plt.subplots(figsize=(9, 14))
    sns.heatmap(mat, annot=True, fmt=".1f", cmap="RdYlGn",
                xticklabels=[f"{p}\n{s}" for p, s in hm_cols],
                yticklabels=heatmap_techs, cbar_kws={"label": "% postings"}, ax=ax)
    ax.set_title("Technology mention rates (%) by period x seniority\n(top 20 rising, 15 high-stable, 15 declining)")
    plt.tight_layout()
    plt.savefig(FIGS / "tech_shift_heatmap.png", dpi=130)
    plt.close()
    print("  wrote tech_shift_heatmap.png")

    # ---------- Step 5: Stack diversity (tech_count) ----------
    df["tech_count"] = df[tech_cols].sum(axis=1)
    div_rows = []
    for p in ["2024", "2026"]:
        for sen in ["junior", "mid", "senior", "ALL"]:
            if sen == "ALL":
                m = df[df["period2"] == p]
            else:
                m = df[(df["period2"] == p) & (df["seniority_3level"] == sen)]
            if len(m) == 0:
                continue
            div_rows.append({
                "period": p, "seniority_3level": sen, "n": len(m),
                "median": m["tech_count"].median(),
                "mean": m["tech_count"].mean(),
                "p25": m["tech_count"].quantile(0.25),
                "p75": m["tech_count"].quantile(0.75),
                "p90": m["tech_count"].quantile(0.90),
            })
    div_df = pd.DataFrame(div_rows)
    div_df.to_csv(TABLES / "stack_diversity.csv", index=False)
    print("  wrote stack_diversity.csv")
    print(div_df.to_string(index=False))

    # Length-normalized version: tech count per 1000 chars
    df["tech_density_1k"] = df["tech_count"] / (df["text_len"] / 1000).replace(0, np.nan)
    div_norm = []
    for p in ["2024", "2026"]:
        for sen in ["junior", "mid", "senior", "ALL"]:
            if sen == "ALL":
                m = df[df["period2"] == p]
            else:
                m = df[(df["period2"] == p) & (df["seniority_3level"] == sen)]
            if len(m) == 0:
                continue
            div_norm.append({
                "period": p, "seniority_3level": sen, "n": len(m),
                "median_density_per_1k": m["tech_density_1k"].median(),
                "mean_density_per_1k": m["tech_density_1k"].mean(),
                "median_text_len": m["text_len"].median(),
            })
    pd.DataFrame(div_norm).to_csv(TABLES / "stack_diversity_density.csv", index=False)
    print("  wrote stack_diversity_density.csv")

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    for p, color in [("2024", "steelblue"), ("2026", "tomato")]:
        vals = df[df["period2"] == p]["tech_count"]
        ax.hist(vals, bins=range(0, 30), alpha=0.55, label=f"{p} (median={vals.median():.0f})", color=color)
    ax.set_xlabel("Distinct technologies mentioned")
    ax.set_ylabel("# postings")
    ax.set_title("Stack diversity: tech count per posting, by period")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "stack_diversity_hist.png", dpi=130)
    plt.close()
    print("  wrote stack_diversity_hist.png")

    # ---------- Step 6: AI integration pattern ----------
    # Define "AI-mentioning": any of llm, openai_api, claude_api, rag, langchain, huggingface,
    # vector_databases, prompt_engineering, fine_tuning, mcp, ai_agents, generative_ai,
    # copilot, cursor_ide, chatgpt, claude_tool, gemini, codex
    ai_cols = ["llm", "openai_api", "claude_api", "rag", "langchain", "huggingface",
               "vector_databases", "prompt_engineering", "fine_tuning", "mcp", "ai_agents",
               "generative_ai", "copilot", "cursor_ide", "chatgpt", "claude_tool",
               "gemini", "codex"]
    ai_cols = [c for c in ai_cols if c in df.columns]
    df["mentions_ai"] = (df[ai_cols].sum(axis=1) > 0)

    ai_rows = []
    for p in ["2024", "2026"]:
        m_ai = df[(df["period2"] == p) & (df["mentions_ai"])]
        m_no = df[(df["period2"] == p) & (~df["mentions_ai"])]
        for t in tech_cols:
            if t in ai_cols:
                continue
            r_ai = m_ai[t].mean() if len(m_ai) > 0 else np.nan
            r_no = m_no[t].mean() if len(m_no) > 0 else np.nan
            if pd.isna(r_ai) or pd.isna(r_no):
                continue
            ai_rows.append({"period": p, "tech": t, "rate_in_ai_mentioning": r_ai,
                            "rate_in_non_ai": r_no, "excess": r_ai - r_no,
                            "n_ai": len(m_ai), "n_non_ai": len(m_no)})
    ai_df = pd.DataFrame(ai_rows)
    ai_df = ai_df.sort_values(["period", "excess"], ascending=[True, False])
    ai_df.to_csv(TABLES / "ai_integration_cooccurrence.csv", index=False)
    print("  wrote ai_integration_cooccurrence.csv")

    # Density confound check
    dens_rows = []
    for p in ["2024", "2026"]:
        for label, mask in [("ai_mentioning", df["mentions_ai"]),
                            ("non_ai", ~df["mentions_ai"])]:
            m = df[(df["period2"] == p) & mask]
            if len(m) == 0:
                continue
            dens_rows.append({
                "period": p, "group": label, "n": len(m),
                "mean_tech_count": m["tech_count"].mean(),
                "mean_text_len": m["text_len"].mean(),
                "mean_density_per_1k": m["tech_density_1k"].mean(),
                "median_tech_count": m["tech_count"].median(),
                "median_density_per_1k": m["tech_density_1k"].median(),
            })
    pd.DataFrame(dens_rows).to_csv(TABLES / "ai_density_check.csv", index=False)
    print("  wrote ai_density_check.csv")

    # ---------- Best-available seniority sensitivity (just for rising tech comparison) ----------
    # Compute mention rates using best_available_3level where non-null; report deltas
    ba_rows = []
    for p in ["2024", "2026"]:
        for sen in ["junior", "mid", "senior"]:
            m = df[(df["period2"] == p) & (df["best_available_3level"] == sen)]
            if len(m) == 0:
                continue
            row = {"period": p, "seniority_best_available_3level": sen, "n": len(m)}
            for t in tech_cols:
                row[t] = m[t].mean()
            ba_rows.append(row)
    pd.DataFrame(ba_rows).to_csv(TABLES / "tech_mention_rates_best_available.csv", index=False)
    print("  wrote tech_mention_rates_best_available.csv")

    # ---------- tech_count calibration ratio check (Agent Prep claimed 24.8x) ----------
    # Within-2024 (arshkon vs asaniczka) vs cross-period (arshkon vs scraped)
    d_arsh = df[df["source"] == "kaggle_arshkon"]["tech_count"]
    d_asan = df[df["source"] == "kaggle_asaniczka"]["tech_count"]
    d_scr = df[df["source"] == "scraped"]["tech_count"]
    def cohend(a, b):
        mean_diff = a.mean() - b.mean()
        pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        return mean_diff / pooled if pooled else np.nan
    within24 = cohend(d_arsh, d_asan)
    cross = cohend(d_arsh, d_scr)
    calib = {
        "metric": "tech_count",
        "arshkon_mean": d_arsh.mean(), "asaniczka_mean": d_asan.mean(), "scraped_mean": d_scr.mean(),
        "within_2024_cohen_d": within24, "cross_period_cohen_d": cross,
        "calibration_ratio": cross / within24 if within24 else np.nan,
    }
    pd.DataFrame([calib]).to_csv(TABLES / "tech_count_calibration.csv", index=False)
    print("  wrote tech_count_calibration.csv")
    print(f"  tech_count calibration ratio = {calib['calibration_ratio']:.2f}")

    print("Done T14 step 01.")


if __name__ == "__main__":
    main()
