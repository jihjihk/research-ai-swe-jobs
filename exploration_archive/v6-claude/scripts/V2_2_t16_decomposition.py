"""V2.2 — Independent T16 overlap panel decomposition.

Builds a 240-company overlap panel (≥3 SWE each in kaggle_arshkon 2024 and scraped 2026),
then computes within-company vs between-company AI-prevalence decomposition on the
broad 24-term union via the shared tech matrix (so boilerplate-insensitive).

Also runs alternative checks:
  (a) k-means k=4 with a different seed to test cluster 3 stability
  (b) Archetype pivot rate (74.6%) via archetype labels join
  (c) Specification-dependence: compute decomposition on companies with 0 AI mentions in 2024
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "exploration" / "tables" / "V2"
OUT.mkdir(parents=True, exist_ok=True)

UNIFIED = ROOT / "data" / "unified.parquet"
TECH = ROOT / "exploration" / "artifacts" / "shared" / "swe_tech_matrix.parquet"
ARCH = ROOT / "exploration" / "artifacts" / "shared" / "swe_archetype_labels.parquet"

BROAD_AI_COLS = [
    "llm", "rag", "agents_framework", "copilot", "claude_api", "claude_tool",
    "cursor_tool", "gemini_tool", "codex_tool", "chatgpt", "openai_api",
    "prompt_engineering", "fine_tuning", "mcp", "embedding", "transformer_arch",
    "machine_learning", "deep_learning", "pytorch", "tensorflow", "langchain",
    "langgraph", "nlp", "huggingface",
]


def main() -> None:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW u AS SELECT * FROM read_parquet('{UNIFIED}')")
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{TECH}')")

    broad_expr = " OR ".join([f"t.{c}" for c in BROAD_AI_COLS])

    # Step 1: assemble per-row data; restrict arshkon 2024 and scraped 2026 only.
    join_sql = f"""
    SELECT u.uid, u.company_name_canonical AS co, u.source,
           CASE WHEN u.source = 'kaggle_arshkon' THEN '2024'
                WHEN u.source = 'scraped' THEN '2026' END AS year,
           u.is_aggregator,
           (CASE WHEN {broad_expr} THEN 1 ELSE 0 END) AS any_ai
    FROM u JOIN t USING (uid)
    WHERE u.is_swe = TRUE AND u.source_platform = 'linkedin'
      AND u.is_english = TRUE AND u.date_flag = 'ok'
      AND u.source IN ('kaggle_arshkon', 'scraped')
      AND u.company_name_canonical IS NOT NULL
    """
    df = con.execute(join_sql).fetchdf()
    print(f"Rows: {len(df)}")
    print(df.groupby("year").size())

    # Build panel: companies with >=3 in both years
    counts = df.groupby(["co", "year"]).size().unstack(fill_value=0)
    counts.columns.name = None
    panel_cos = counts[(counts.get("2024", 0) >= 3) & (counts.get("2026", 0) >= 3)].index.tolist()
    print(f"Overlap panel companies (>=3 each): {len(panel_cos)}")

    panel_df = df[df["co"].isin(panel_cos)].copy()

    # Per-company-period AI rate
    pc = panel_df.groupby(["co", "year"]).agg(n=("uid", "count"), ai=("any_ai", "mean")).reset_index()
    pc_wide = pc.pivot(index="co", columns="year", values=["n", "ai"])
    pc_wide.columns = [f"{a}_{b}" for a, b in pc_wide.columns]
    pc_wide = pc_wide.dropna()
    print(f"Panel with both years non-null: {len(pc_wide)}")

    # Oaxaca-style (symmetric weights) decomposition for panel-mean AI rate
    # panel_mean_year = sum_i (w_i,year * r_i,year) where w_i = n_i/sum_n
    def panel_mean_rate(w, r):
        return (w * r).sum()

    w24 = pc_wide["n_2024"] / pc_wide["n_2024"].sum()
    w26 = pc_wide["n_2026"] / pc_wide["n_2026"].sum()
    r24 = pc_wide["ai_2024"]
    r26 = pc_wide["ai_2026"]

    mean_24 = panel_mean_rate(w24, r24)
    mean_26 = panel_mean_rate(w26, r26)

    # symmetric decomp: total = within + between + interaction
    w_avg = (w24 + w26) / 2
    r_avg = (r24 + r26) / 2
    within = (w_avg * (r26 - r24)).sum()  # same co, rate change
    between = (r_avg * (w26 - w24)).sum()  # same rate, weight change
    total = mean_26 - mean_24
    interaction = total - within - between

    print("\nT16 panel AI decomposition (broad 24-term):")
    print(f"  panel mean 2024: {mean_24:.4f}")
    print(f"  panel mean 2026: {mean_26:.4f}")
    print(f"  total Δ: {total:.4f}")
    print(f"  within: {within:.4f}")
    print(f"  between: {between:.4f}")
    print(f"  interaction: {interaction:.4f}")
    print(f"  within %: {100*within/total:.1f}%")

    # T16 reported: total +0.2291, within +0.2103 (92%)
    print("\nT16 reported: total +0.2291, within +0.2103 (92%)")

    results = pd.DataFrame([
        {
            "metric": "AI broad",
            "panel_n": len(pc_wide),
            "mean_2024": round(mean_24, 4),
            "mean_2026": round(mean_26, 4),
            "total": round(total, 4),
            "within": round(within, 4),
            "between": round(between, 4),
            "interaction": round(interaction, 4),
            "within_pct": round(100 * within / total, 1),
        }
    ])
    results.to_csv(OUT / "V2_2_decomposition.csv", index=False)

    # ------- Alt explanation: drop companies with 0 AI in 2024 -----
    zero_cos = pc_wide[pc_wide["ai_2024"] == 0].index
    not_zero = pc_wide[pc_wide["ai_2024"] > 0]
    print(f"\nCompanies with 0 AI in 2024: {len(zero_cos)} / {len(pc_wide)}")
    print(f"  Running decomp on companies with >0 2024 AI ({len(not_zero)}):")
    w24n = not_zero["n_2024"] / not_zero["n_2024"].sum()
    w26n = not_zero["n_2026"] / not_zero["n_2026"].sum()
    r24n = not_zero["ai_2024"]
    r26n = not_zero["ai_2026"]
    mean_24n = (w24n * r24n).sum()
    mean_26n = (w26n * r26n).sum()
    within_n = (((w24n + w26n) / 2) * (r26n - r24n)).sum()
    totn = mean_26n - mean_24n
    print(f"  mean 2024: {mean_24n:.4f} → 2026: {mean_26n:.4f}")
    print(f"  total: {totn:.4f}, within: {within_n:.4f} ({100*within_n/totn:.1f}%)")

    # And the zero-2024 companies only (pure new AI adoption)
    if len(zero_cos) > 0:
        zn = pc_wide.loc[zero_cos]
        w24z = zn["n_2024"] / zn["n_2024"].sum()
        w26z = zn["n_2026"] / zn["n_2026"].sum()
        within_z = (((w24z + w26z) / 2) * (zn["ai_2026"] - zn["ai_2024"])).sum()
        mean_24z = (w24z * zn["ai_2024"]).sum()
        mean_26z = (w26z * zn["ai_2026"]).sum()
        print(f"\n  Companies with 0 AI in 2024 only ({len(zn)}):")
        print(f"  mean 2024: {mean_24z:.4f} → 2026: {mean_26z:.4f}")
        print(f"  within: {within_z:.4f}")

    # ------- k-means k=4 with a different seed -----
    # Build per-company change vector: ΔAI, Δdesc_len, Δscope(e2e+cross-fn), Δentry, ΔYOE
    # We'll use a simpler vector: ΔAI, Δdesc_len (raw), Δscope from raw desc
    per_co = (
        panel_df
        .merge(pd.DataFrame(pc_wide.reset_index()["co"]), on="co")
        .groupby(["co", "year"])["any_ai"].mean()
        .unstack()
    )
    per_co["dAI"] = per_co["2026"] - per_co["2024"]

    # Add description length and scope via a separate SQL
    desc_sql = """
    SELECT u.company_name_canonical AS co,
           CASE WHEN u.source='kaggle_arshkon' THEN '2024' ELSE '2026' END AS year,
           AVG(length(u.description)) AS desc_len,
           AVG(CASE WHEN regexp_matches(lower(u.description), '\\b(end[- ]to[- ]end|cross[- ]functional)\\b') THEN 1 ELSE 0 END) AS scope
    FROM u
    WHERE u.is_swe=TRUE AND u.source_platform='linkedin' AND u.is_english=TRUE AND u.date_flag='ok'
      AND u.source IN ('kaggle_arshkon','scraped') AND u.company_name_canonical IS NOT NULL
    GROUP BY co, year
    """
    desc = con.execute(desc_sql).fetchdf()
    desc_p = desc.pivot(index="co", columns="year", values=["desc_len", "scope"])
    desc_p.columns = [f"{a}_{b}" for a, b in desc_p.columns]
    desc_p["ddesc_len"] = desc_p["desc_len_2026"] - desc_p["desc_len_2024"]
    desc_p["dscope"] = desc_p["scope_2026"] - desc_p["scope_2024"]

    vec = per_co.join(desc_p[["ddesc_len", "dscope"]], how="inner")
    vec = vec.loc[panel_cos].dropna()
    print(f"\nk-means panel n={len(vec)}")

    X = vec[["dAI", "ddesc_len", "dscope"]].values
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=4, random_state=7, n_init=20).fit(Xs)
    vec["cluster"] = km.labels_
    cluster_summary = vec.groupby("cluster").agg(
        n=("dAI", "size"),
        mean_dAI=("dAI", "mean"),
        mean_ddesc=("ddesc_len", "mean"),
        mean_dscope=("dscope", "mean"),
    )
    print("\nk-means k=4 cluster summary (seed=7, different from T16's):")
    print(cluster_summary)
    cluster_summary.to_csv(OUT / "V2_2_kmeans_k4_seed7.csv")

    # Identify the "tool-stack adopter" cluster = highest dAI
    top_cluster = cluster_summary["mean_dAI"].idxmax()
    print(f"Top-dAI cluster: {top_cluster}, n={cluster_summary.loc[top_cluster, 'n']}")

    # ------- Archetype pivot rate -----
    arch = con.execute(f"SELECT * FROM read_parquet('{ARCH}')").fetchdf()
    print(f"\nArchetype labels columns: {list(arch.columns)}")
    if "archetype_id" in arch.columns:
        ac = arch[["uid", "archetype_id"]]
    elif "archetype_label" in arch.columns:
        ac = arch[["uid", "archetype_label"]].rename(columns={"archetype_label": "archetype_id"})
    else:
        # Try common labels
        ac = arch[[c for c in arch.columns if c == "uid"] + [c for c in arch.columns if "arche" in c.lower()][:1]]
        ac.columns = ["uid", "archetype_id"]
    # Join
    arch_full = panel_df.merge(ac, on="uid", how="left")
    # For each company × year, compute dominant archetype
    arch_full = arch_full.dropna(subset=["archetype_id"])
    # Exclude the "no text" bucket (often -2)
    arch_full = arch_full[arch_full["archetype_id"] != -2]
    dominant = (
        arch_full.groupby(["co", "year", "archetype_id"]).size()
        .reset_index(name="n")
        .sort_values(["co", "year", "n"], ascending=[True, True, False])
        .drop_duplicates(subset=["co", "year"], keep="first")
    )
    dom_p = dominant.pivot(index="co", columns="year", values="archetype_id")
    dom_p = dom_p.dropna()
    pivot_rate = (dom_p["2024"] != dom_p["2026"]).mean()
    print(f"Archetype pivot rate (companies with both years labeled): {pivot_rate:.3f} (n={len(dom_p)})")
    # Restrict to panel companies only
    dom_pc = dom_p.loc[dom_p.index.isin(panel_cos)]
    print(f"  Restricted to overlap panel: {(dom_pc['2024'] != dom_pc['2026']).mean():.3f} (n={len(dom_pc)})")

    print("\nDone.")


if __name__ == "__main__":
    main()
