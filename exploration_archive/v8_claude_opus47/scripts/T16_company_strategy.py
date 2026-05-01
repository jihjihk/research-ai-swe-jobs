"""T16: Company hiring strategy typology.

Builds per-company change metrics (2024 -> 2026) for companies with >=3 SWE
postings in both periods, then:
  - k-means clusters the change profiles (k=3..6) and picks best by silhouette
  - decomposes aggregate changes into within/between components
  - examines scope inflation (requirement_breadth proxy) within-company
  - profiles new entrants vs returning companies
  - compares aggregator vs direct employer trajectories

Outputs go to exploration/tables/T16/, exploration/figures/T16/.
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

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/jihgaboot/gabor/job-research")
OUT_TBL = ROOT / "exploration/tables/T16"
OUT_FIG = ROOT / "exploration/figures/T16"
OUT_TBL.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

UNIFIED = ROOT / "data/unified.parquet"
CLEANED = ROOT / "exploration/artifacts/shared/swe_cleaned_text.parquet"
TECH = ROOT / "exploration/artifacts/shared/swe_tech_matrix.parquet"
SPEC = ROOT / "exploration/artifacts/shared/entry_specialist_employers.csv"
ARCHE = ROOT / "exploration/artifacts/shared/swe_archetype_labels.parquet"

# V1-approved AI patterns
AI_STRICT = re.compile(
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|vector database|pinecone|huggingface|hugging face)\b",
    re.IGNORECASE,
)
AI_BROAD_EXTRA = re.compile(
    r"\b(ai|artificial intelligence|ml|machine learning|llm|large language model|generative ai|genai|anthropic)\b",
    re.IGNORECASE,
)

# Org-scope (V1-approved per T11/V1 — stakeholder, cross-functional, etc.)
ORG_SCOPE = re.compile(
    r"\b(cross[- ]functional|end[- ]to[- ]end|stakeholder|ownership|own the|own this|drive the|drive this|strategic|roadmap|partner with|collaborate across)\b",
    re.IGNORECASE,
)


def run_assertions():
    """TDD asserts for the regex patterns."""
    assert AI_STRICT.search("We use copilot daily")
    assert AI_STRICT.search("Experience with LangChain and RAG pipelines")
    assert AI_STRICT.search("GPT-4 based tools")
    assert not AI_STRICT.search("random generic ai content")  # strict should NOT match bare ai
    assert AI_STRICT.search("fine-tuning LLMs")
    assert AI_BROAD_EXTRA.search("machine learning basics")
    assert AI_BROAD_EXTRA.search("LLM engineer")
    assert not AI_BROAD_EXTRA.search("realm of customers")  # ml substring guard
    assert ORG_SCOPE.search("cross-functional partner")
    assert ORG_SCOPE.search("end-to-end ownership")
    assert ORG_SCOPE.search("own the roadmap")
    assert not ORG_SCOPE.search("minor config change")


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


def build_row_level_df(con):
    """Load one row per SWE posting with all metrics we'll aggregate."""
    sql = """
    WITH base AS (
      SELECT u.uid,
             u.company_name_canonical AS company,
             u.source,
             CASE WHEN u.source='scraped' THEN 'scraped'
                  WHEN u.source IN ('kaggle_arshkon','kaggle_asaniczka') THEN '2024'
                  ELSE 'other' END AS period,
             u.source_platform,
             u.is_aggregator,
             u.seniority_final,
             u.yoe_extracted,
             u.description_length,
             u.description AS raw_desc,
             u.company_industry,
             u.metro_area
      FROM u
      WHERE u.source_platform='linkedin' AND u.is_english AND u.date_flag='ok' AND u.is_swe=true
        AND u.company_name_canonical IS NOT NULL AND u.company_name_canonical <> ''
    )
    SELECT * FROM base
    """
    df = con.execute(sql).df()
    return df


def add_tech_counts(df, con):
    # Sum tech column presence per uid
    tcols = [c for c in pq.read_table(str(TECH)).column_names if c != "uid"]
    cols_sum = " + ".join([f"CAST({c} AS INTEGER)" for c in tcols])
    sql = f"SELECT uid, ({cols_sum}) AS tech_count FROM t"
    tech = con.execute(sql).df()
    return df.merge(tech, on="uid", how="left")


def add_cleaned_text(df, con):
    # Join cleaned text for org-scope density & AI density
    sql = """
    SELECT uid, description_cleaned AS desc_cleaned
    FROM c
    WHERE text_source='llm'
    """
    ct = con.execute(sql).df()
    return df.merge(ct, on="uid", how="left")


def compute_post_level_features(df):
    """Per-posting binary flags, counts, lengths."""
    df = df.copy()
    raw = df["raw_desc"].fillna("").astype(str)

    df["ai_strict_any"] = raw.str.contains(AI_STRICT).astype(int)
    df["ai_broad_extra_any"] = raw.str.contains(AI_BROAD_EXTRA).astype(int)
    df["ai_broad_any"] = (df["ai_strict_any"] | df["ai_broad_extra_any"]).astype(int)

    # Org-scope: count per cleaned 1k chars (sensitivity using raw 1k chars)
    cleaned = df["desc_cleaned"].fillna("")
    df["desc_cleaned_len"] = cleaned.str.len()

    def count_pat(s, pat):
        if not s:
            return 0
        return len(pat.findall(s))

    df["org_scope_count_cleaned"] = cleaned.apply(lambda s: count_pat(s, ORG_SCOPE))
    df["org_scope_per_1k_cleaned"] = np.where(
        df["desc_cleaned_len"] > 0,
        df["org_scope_count_cleaned"] / df["desc_cleaned_len"] * 1000,
        0.0,
    )

    # Requirement breadth proxy (for within-company scope inflation)
    # Use cleaned text + tech_count + org_scope + credential keywords
    CRED = re.compile(
        r"\b(bachelor'?s?|master'?s?|phd|mba|degree|years of experience|yoe)\b",
        re.IGNORECASE,
    )
    SOFT = re.compile(
        r"\b(communication|collaboration|leadership|problem[- ]solving|teamwork|ownership|initiative|self[- ]starter|stakeholder)\b",
        re.IGNORECASE,
    )
    df["cred_count"] = cleaned.apply(lambda s: count_pat(s, CRED))
    df["soft_count"] = cleaned.apply(lambda s: count_pat(s, SOFT))

    # Simple composite: tech + soft + cred + org_scope counts
    df["breadth_raw"] = (
        df["tech_count"].fillna(0)
        + df["org_scope_count_cleaned"].fillna(0)
        + df["cred_count"].fillna(0)
        + df["soft_count"].fillna(0)
    )

    df["j1"] = (df["seniority_final"] == "entry").astype(int)
    df["j2"] = df["seniority_final"].isin(["entry", "associate"]).astype(int)
    df["j3"] = (df["yoe_extracted"] <= 2).fillna(False).astype(int)

    return df


def length_residualize(df, x_col, y_col):
    """OLS residualization: y_resid = y - (a + b*x)."""
    x = df[x_col].fillna(df[x_col].median()).values.astype(float)
    y = df[y_col].fillna(df[y_col].median()).values.astype(float)
    # Solve y = a + b*x
    A = np.vstack([np.ones_like(x), x]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    return y - (a + b * x), (a, b)


def per_company_aggregate(df_period):
    """Per-company aggregate metrics for a given period subset."""
    grp = df_period.groupby("company")
    out = pd.DataFrame(
        {
            "n": grp.size(),
            "entry_share_j1": grp["j1"].mean(),
            "entry_share_j2": grp["j2"].mean(),
            "entry_share_j3": grp["j3"].mean(),
            "ai_strict": grp["ai_strict_any"].mean(),
            "ai_broad": grp["ai_broad_any"].mean(),
            "mean_desc_length_raw": grp["description_length"].mean(),
            "mean_desc_length_cleaned": grp["desc_cleaned_len"].mean(),
            "mean_tech_count": grp["tech_count"].mean(),
            "mean_org_scope_per_1k": grp["org_scope_per_1k_cleaned"].mean(),
            "mean_breadth_raw": grp["breadth_raw"].mean(),
            "mean_breadth_resid": grp["breadth_resid"].mean()
            if "breadth_resid" in df_period.columns
            else np.nan,
            "agg_rate": grp["is_aggregator"].mean(),
        }
    )
    return out.reset_index()


def oaxaca_decomp(df_ar, df_sc, metric_col, weight_col="n"):
    """Decompose aggregate change into within-company and between-company.

    For overlap panel (same set of companies in both periods):
    - Aggregate(ar) = sum w_i^ar * y_i^ar / sum w_i^ar
    - Aggregate(sc) = sum w_i^sc * y_i^sc / sum w_i^sc
    - Within: sum w_i^avg * (y_i^sc - y_i^ar)
    - Between: driven by weight changes
    """
    merged = df_ar.merge(df_sc, on="company", suffixes=("_ar", "_sc"))
    w_ar = merged[weight_col + "_ar"].values.astype(float)
    w_sc = merged[weight_col + "_sc"].values.astype(float)
    y_ar = merged[metric_col + "_ar"].values.astype(float)
    y_sc = merged[metric_col + "_sc"].values.astype(float)

    # drop rows with nan in either y
    valid = ~(np.isnan(y_ar) | np.isnan(y_sc))
    w_ar, w_sc, y_ar, y_sc = w_ar[valid], w_sc[valid], y_ar[valid], y_sc[valid]

    sw_ar = w_ar.sum()
    sw_sc = w_sc.sum()
    if sw_ar == 0 or sw_sc == 0:
        return dict(aggregate=np.nan, within=np.nan, between=np.nan)

    agg_ar = (w_ar * y_ar).sum() / sw_ar
    agg_sc = (w_sc * y_sc).sum() / sw_sc
    aggregate_delta = agg_sc - agg_ar

    # Symmetric decomposition: use average weights
    s_ar = w_ar / sw_ar
    s_sc = w_sc / sw_sc
    s_avg = (s_ar + s_sc) / 2
    y_avg = (y_ar + y_sc) / 2

    within = (s_avg * (y_sc - y_ar)).sum()
    between = ((s_sc - s_ar) * y_avg).sum()

    return dict(
        aggregate=float(aggregate_delta),
        within=float(within),
        between=float(between),
        arshkon_agg=float(agg_ar),
        scraped_agg=float(agg_sc),
        n_companies=int(valid.sum()),
    )


def main():
    run_assertions()

    print("[T16] Opening connection and loading base dataframe...")
    con = open_con()
    df = build_row_level_df(con)
    print(f"  {len(df):,} SWE LinkedIn rows loaded")

    print("[T16] Adding tech counts & cleaned text...")
    df = add_tech_counts(df, con)
    df = add_cleaned_text(df, con)
    print("[T16] Computing post-level features...")
    df = compute_post_level_features(df)

    # Length-residualize breadth (per V1: breadth - (a + b*length))
    resid, (a, b) = length_residualize(df, "desc_cleaned_len", "breadth_raw")
    df["breadth_resid"] = resid
    print(f"  breadth_raw ~ {a:.3f} + {b:.5f}*len (residualized)")

    # Aggregator / specialist flags
    specialists = set(pd.read_csv(SPEC)["company"].tolist())
    df["is_specialist"] = df["company"].isin(specialists).astype(int)
    print(f"  specialists loaded: {len(specialists)}")

    # Build overlap panels
    comp_periods = (
        df.groupby(["company", "source"]).size().unstack("source", fill_value=0)
    )
    comp_periods["n_2024"] = (
        comp_periods.get("kaggle_arshkon", 0) + comp_periods.get("kaggle_asaniczka", 0)
    )
    comp_periods["n_arsh"] = comp_periods.get("kaggle_arshkon", 0)
    comp_periods["n_scraped"] = comp_periods.get("scraped", 0)

    overlap_arsh_3 = set(
        comp_periods[(comp_periods["n_arsh"] >= 3) & (comp_periods["n_scraped"] >= 3)].index
    )
    overlap_pool_3 = set(
        comp_periods[(comp_periods["n_2024"] >= 3) & (comp_periods["n_scraped"] >= 3)].index
    )
    print(
        f"  panels: arshkon>=3 & scraped>=3 = {len(overlap_arsh_3)}; "
        f"pooled>=3 & scraped>=3 = {len(overlap_pool_3)}"
    )

    # Cap per-company at 50 (sensitivity b)
    def cap_per_company(dfin, cap=50):
        def samp(g):
            if len(g) <= cap:
                return g
            return g.sample(n=cap, random_state=42)

        return dfin.groupby(["company", "source"], group_keys=False).apply(samp)

    # Period labels
    df["period"] = np.where(df["source"] == "scraped", "scraped", "2024")
    df["period_arsh"] = np.where(
        df["source"] == "scraped",
        "scraped",
        np.where(df["source"] == "kaggle_arshkon", "arshkon", "other"),
    )

    # --------------------------------------------------------------
    # STEP 2: Per-company change metrics on the PRIMARY panel (arshkon vs scraped, 240 co)
    # --------------------------------------------------------------
    print("[T16] Computing per-company aggregates on arshkon/scraped overlap panel...")

    df_panel = df[df["company"].isin(overlap_arsh_3)].copy()
    df_ar = df_panel[df_panel["source"] == "kaggle_arshkon"].copy()
    df_sc = df_panel[df_panel["source"] == "scraped"].copy()

    agg_ar = per_company_aggregate(df_ar)
    agg_sc = per_company_aggregate(df_sc)

    merged = agg_ar.merge(agg_sc, on="company", suffixes=("_ar", "_sc"))
    metrics = [
        "entry_share_j1",
        "entry_share_j2",
        "entry_share_j3",
        "ai_strict",
        "ai_broad",
        "mean_desc_length_raw",
        "mean_desc_length_cleaned",
        "mean_tech_count",
        "mean_org_scope_per_1k",
        "mean_breadth_raw",
        "mean_breadth_resid",
    ]
    for m in metrics:
        merged[f"d_{m}"] = merged[f"{m}_sc"] - merged[f"{m}_ar"]

    # Save the change profile table
    merged.to_csv(OUT_TBL / "01_company_change_profiles_arshkon_scraped.csv", index=False)
    print(f"  saved 01_company_change_profiles_arshkon_scraped.csv ({len(merged)} cos)")

    # --------------------------------------------------------------
    # STEP 3: Cluster companies by change vector
    # --------------------------------------------------------------
    print("[T16] k-means clustering on change vectors...")
    cluster_features = [
        "d_entry_share_j2",
        "d_ai_strict",
        "d_mean_desc_length_cleaned",
        "d_mean_tech_count",
        "d_mean_org_scope_per_1k",
        "d_mean_breadth_resid",
    ]
    X = merged[cluster_features].fillna(0).values
    Xs = StandardScaler().fit_transform(X)

    cluster_results = []
    for k in [3, 4, 5, 6]:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels) if len(set(labels)) > 1 else 0.0
        cluster_results.append({"k": k, "silhouette": float(sil), "inertia": float(km.inertia_)})
    cluster_metrics_df = pd.DataFrame(cluster_results)
    print(cluster_metrics_df.to_string(index=False))
    cluster_metrics_df.to_csv(OUT_TBL / "02_cluster_silhouette.csv", index=False)

    # Pick best k by silhouette (tiebreak: smaller k)
    best_k = int(cluster_metrics_df.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    print(f"  best k by silhouette = {best_k}")

    km = KMeans(n_clusters=best_k, n_init=50, random_state=42)
    merged["cluster"] = km.fit_predict(Xs)

    cluster_profile = (
        merged.groupby("cluster")[cluster_features + ["n_ar", "n_sc"]]
        .mean()
        .round(4)
        .reset_index()
    )
    cluster_profile["n_companies"] = merged.groupby("cluster").size().values
    cluster_profile.to_csv(OUT_TBL / "03_cluster_profile.csv", index=False)
    print("  cluster profile (means of change vector):")
    print(cluster_profile.to_string(index=False))

    # Save per-company cluster membership
    merged[
        ["company", "cluster", "n_ar", "n_sc", "agg_rate_ar", "agg_rate_sc"] + cluster_features
    ].to_csv(OUT_TBL / "04_company_cluster_membership.csv", index=False)

    # Name clusters by content (heuristic based on which dims are extreme)
    cluster_names = {}
    for c in range(best_k):
        row = cluster_profile[cluster_profile["cluster"] == c].iloc[0]
        tags = []
        if row["d_ai_strict"] > 0.15:
            tags.append("ai_forward")
        elif row["d_ai_strict"] > 0.08:
            tags.append("ai_moderate")
        if row["d_mean_desc_length_cleaned"] > 300:
            tags.append("desc_expanding")
        elif row["d_mean_desc_length_cleaned"] < -200:
            tags.append("desc_contracting")
        if row["d_mean_tech_count"] > 0.8:
            tags.append("tech_stacking")
        elif row["d_mean_tech_count"] < -0.5:
            tags.append("tech_consolidating")
        if row["d_entry_share_j2"] > 0.03:
            tags.append("entry_ramping")
        elif row["d_entry_share_j2"] < -0.03:
            tags.append("entry_dropping")
        if row["d_mean_breadth_resid"] > 0.8:
            tags.append("scope_expanding")
        cluster_names[c] = "_".join(tags) if tags else "stable"
    merged["cluster_name"] = merged["cluster"].map(cluster_names)

    # top example companies per cluster
    exemplars = (
        merged.sort_values("n_sc", ascending=False)
        .groupby("cluster")
        .head(10)[["cluster", "cluster_name", "company", "n_ar", "n_sc"] + cluster_features]
    )
    exemplars.to_csv(OUT_TBL / "05_cluster_exemplars.csv", index=False)
    print("  cluster names:", cluster_names)

    # --------------------------------------------------------------
    # STEP 4: Within-vs-between decomposition on overlap panel
    # --------------------------------------------------------------
    print("[T16] Within/between decomposition on arshkon<->scraped overlap panel...")
    decomp_metrics = [
        "entry_share_j1",
        "entry_share_j2",
        "entry_share_j3",
        "ai_strict",
        "ai_broad",
        "mean_desc_length_raw",
        "mean_desc_length_cleaned",
        "mean_tech_count",
        "mean_org_scope_per_1k",
        "mean_breadth_raw",
        "mean_breadth_resid",
    ]

    def run_decomp(dfA, dfB, label):
        A = per_company_aggregate(dfA)
        B = per_company_aggregate(dfB)
        out = []
        for m in decomp_metrics:
            r = oaxaca_decomp(A, B, m)
            r["metric"] = m
            r["spec"] = label
            out.append(r)
        return pd.DataFrame(out)

    decomp_primary = run_decomp(df_ar, df_sc, "arshkon_vs_scraped_all")
    decomp_primary.to_csv(OUT_TBL / "06_decomp_arsh_scraped.csv", index=False)
    print(decomp_primary[["metric", "aggregate", "within", "between", "n_companies"]].to_string(index=False))

    # Exclude specialists sensitivity
    print("[T16] Within/between decomposition EXCLUDING specialists...")
    df_ar_nsp = df_ar[df_ar["is_specialist"] == 0].copy()
    df_sc_nsp = df_sc[df_sc["is_specialist"] == 0].copy()
    decomp_nospec = run_decomp(df_ar_nsp, df_sc_nsp, "arshkon_vs_scraped_no_specialist")
    decomp_nospec.to_csv(OUT_TBL / "07_decomp_arsh_scraped_no_specialist.csv", index=False)

    # Exclude aggregators sensitivity
    df_ar_na = df_ar[df_ar["is_aggregator"] == False].copy()
    df_sc_na = df_sc[df_sc["is_aggregator"] == False].copy()
    decomp_noagg = run_decomp(df_ar_na, df_sc_na, "arshkon_vs_scraped_no_aggregator")
    decomp_noagg.to_csv(OUT_TBL / "08_decomp_arsh_scraped_no_aggregator.csv", index=False)

    # Pooled 2024 (asan + arsh) vs scraped on pooled>=3 panel
    print("[T16] Pooled-2024 -> scraped decomposition on 589-company panel...")
    df_panel_pool = df[df["company"].isin(overlap_pool_3)].copy()
    df_24 = df_panel_pool[df_panel_pool["period"] == "2024"].copy()
    df_sc2 = df_panel_pool[df_panel_pool["period"] == "scraped"].copy()
    decomp_pooled = run_decomp(df_24, df_sc2, "pooled2024_vs_scraped")
    decomp_pooled.to_csv(OUT_TBL / "09_decomp_pooled_scraped.csv", index=False)

    # Cap-at-50 sensitivity
    print("[T16] Cap-at-50 decomposition...")
    df_ar_cap = cap_per_company(df_ar, 50)
    df_sc_cap = cap_per_company(df_sc, 50)
    decomp_cap = run_decomp(df_ar_cap, df_sc_cap, "arshkon_vs_scraped_cap50")
    decomp_cap.to_csv(OUT_TBL / "10_decomp_arsh_scraped_cap50.csv", index=False)

    # --------------------------------------------------------------
    # STEP 5: Within-company scope inflation (length-residualized breadth)
    # --------------------------------------------------------------
    print("[T16] Within-company scope (breadth) change distribution...")
    breadth_change = merged[
        ["company", "n_ar", "n_sc", "d_mean_breadth_raw", "d_mean_breadth_resid", "d_mean_desc_length_cleaned"]
    ].copy()
    # share of companies with positive / negative breadth change
    summary = {
        "n_companies": int(len(breadth_change)),
        "mean_d_breadth_raw": float(breadth_change["d_mean_breadth_raw"].mean()),
        "mean_d_breadth_resid": float(breadth_change["d_mean_breadth_resid"].mean()),
        "median_d_breadth_resid": float(breadth_change["d_mean_breadth_resid"].median()),
        "share_positive_resid": float((breadth_change["d_mean_breadth_resid"] > 0).mean()),
        "share_large_positive_resid": float((breadth_change["d_mean_breadth_resid"] > 1.0).mean()),
    }
    with open(OUT_TBL / "11_breadth_within_company_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    breadth_change.to_csv(OUT_TBL / "12_breadth_change_per_company.csv", index=False)
    print(json.dumps(summary, indent=2))

    # --------------------------------------------------------------
    # STEP 6: New-market-entrants profile
    # --------------------------------------------------------------
    print("[T16] New market entrants profile...")
    companies_2024 = set(df[df["period"] == "2024"]["company"].unique())
    companies_sc = set(df[df["period"] == "scraped"]["company"].unique())
    new_entrants = companies_sc - companies_2024
    print(f"  new entrants = {len(new_entrants):,} cos of {len(companies_sc):,} scraped")

    df_sc_all = df[df["period"] == "scraped"].copy()
    df_sc_all["is_new_entrant"] = df_sc_all["company"].isin(new_entrants).astype(int)

    def ent_profile(d, label):
        return dict(
            cohort=label,
            n_rows=int(len(d)),
            n_cos=int(d["company"].nunique()),
            mean_desc_len_raw=float(d["description_length"].mean()),
            mean_desc_len_cleaned=float(d["desc_cleaned_len"].mean()),
            ai_strict=float(d["ai_strict_any"].mean()),
            ai_broad=float(d["ai_broad_any"].mean()),
            mean_tech=float(d["tech_count"].mean()),
            mean_breadth_resid=float(d["breadth_resid"].mean()),
            j1=float(d["j1"].mean()),
            j2=float(d["j2"].mean()),
            j3=float(d["j3"].mean()),
            mean_yoe=float(d["yoe_extracted"].dropna().mean() if d["yoe_extracted"].notna().sum() else np.nan),
            agg_rate=float(d["is_aggregator"].mean()),
            specialist_rate=float(d["is_specialist"].mean()),
        )

    ent_rows = [
        ent_profile(df_sc_all[df_sc_all["is_new_entrant"] == 1], "new"),
        ent_profile(df_sc_all[df_sc_all["is_new_entrant"] == 0], "returning"),
    ]
    # Top industries for new entrants
    if df_sc_all["company_industry"].notna().any():
        new_ind = (
            df_sc_all[df_sc_all["is_new_entrant"] == 1]["company_industry"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        new_ind.columns = ["industry", "n"]
        new_ind.to_csv(OUT_TBL / "14_new_entrant_top_industries.csv", index=False)
    ent_df = pd.DataFrame(ent_rows)
    ent_df.to_csv(OUT_TBL / "13_new_vs_returning_profile.csv", index=False)
    print(ent_df.to_string(index=False))

    # --------------------------------------------------------------
    # STEP 7: Aggregator vs direct employer trajectories
    # --------------------------------------------------------------
    print("[T16] Aggregator vs direct employer trajectory comparison...")
    # Within overlap panel: split by agg_rate >= 0.5 vs < 0.5
    merged["agg_heavy"] = (merged["agg_rate_sc"] >= 0.5).astype(int)
    agg_split = (
        merged.groupby("agg_heavy")[cluster_features + ["n_ar", "n_sc"]]
        .mean()
        .round(4)
        .reset_index()
    )
    agg_split["n_companies"] = merged.groupby("agg_heavy").size().values
    agg_split.to_csv(OUT_TBL / "15_aggregator_vs_direct_trajectories.csv", index=False)
    print(agg_split.to_string(index=False))

    # --------------------------------------------------------------
    # STEP 8: Archetype-within-company-within-seniority (T28 not ready; skip three-way)
    # --------------------------------------------------------------
    # We don't have T28 yet; we join T09 archetype labels to the overlap panel
    # and report a two-way within-company × within-archetype summary.
    print("[T16] Archetype × company decomposition (two-way)...")
    arche = pd.read_parquet(ARCHE)
    df_p = df_panel.merge(arche, on="uid", how="left")
    df_p["archetype_name"] = df_p["archetype_name"].fillna("unlabeled")
    # Compute aggregate AI-strict change within-company × within-archetype
    ai_by_ca = (
        df_p.groupby(["company", "archetype_name", "source"])["ai_strict_any"]
        .mean()
        .unstack("source", fill_value=np.nan)
        .reset_index()
    )
    if "kaggle_arshkon" in ai_by_ca.columns and "scraped" in ai_by_ca.columns:
        ai_by_ca["d_ai_strict"] = ai_by_ca["scraped"] - ai_by_ca["kaggle_arshkon"]
        # Pairs with both periods present
        ai_pairs = ai_by_ca.dropna(subset=["kaggle_arshkon", "scraped"])
        archetype_summary = (
            ai_pairs.groupby("archetype_name")
            .agg(n_pairs=("d_ai_strict", "size"),
                 mean_d_ai_strict=("d_ai_strict", "mean"),
                 median_d_ai_strict=("d_ai_strict", "median"))
            .reset_index()
            .sort_values("mean_d_ai_strict", ascending=False)
        )
        archetype_summary.to_csv(OUT_TBL / "16_ai_within_company_within_archetype.csv", index=False)
        print(archetype_summary.to_string(index=False))

    # --------------------------------------------------------------
    # FIGURES
    # --------------------------------------------------------------
    print("[T16] Drawing figures...")
    # Fig 1: cluster scatter (d_ai_strict vs d_entry_share_j2)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.arange(best_k))
    for c in range(best_k):
        sub = merged[merged["cluster"] == c]
        ax.scatter(
            sub["d_ai_strict"],
            sub["d_entry_share_j2"],
            s=np.sqrt(sub["n_sc"]) * 3,
            alpha=0.6,
            c=[colors[c]],
            label=f"{c}: {cluster_names[c]} (n={len(sub)})",
        )
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Δ AI-strict share (2024→2026)")
    ax.set_ylabel("Δ J2 entry share (2024→2026)")
    ax.set_title(f"Company change profiles — k={best_k} (overlap panel, n={len(merged)})")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "01_cluster_scatter_ai_vs_entry.png", dpi=150)
    plt.close()

    # Fig 2: decomposition bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    dshow = decomp_primary.set_index("metric").loc[
        ["entry_share_j1", "entry_share_j2", "entry_share_j3", "ai_strict", "ai_broad",
         "mean_tech_count", "mean_breadth_resid"]
    ]
    x = np.arange(len(dshow))
    ax.bar(x - 0.2, dshow["within"], width=0.4, label="within", alpha=0.8)
    ax.bar(x + 0.2, dshow["between"], width=0.4, label="between", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(dshow.index, rotation=35, ha="right")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Δ (scraped − arshkon)")
    ax.set_title(f"Within vs between decomposition — overlap panel (n={decomp_primary['n_companies'].iloc[0]})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG / "02_within_between_decomposition.png", dpi=150)
    plt.close()

    # Fig 3: breadth change distribution per company
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        breadth_change["d_mean_breadth_resid"].dropna(), bins=40, alpha=0.8, color="steelblue"
    )
    ax.axvline(0, color="k", lw=1)
    ax.axvline(breadth_change["d_mean_breadth_resid"].mean(), color="red", linestyle="--",
               label=f"mean = {breadth_change['d_mean_breadth_resid'].mean():.2f}")
    ax.set_xlabel("Δ length-residualized requirement breadth (per-company)")
    ax.set_ylabel("n companies")
    ax.set_title(f"Within-company scope-inflation distribution (overlap panel, n={len(breadth_change)})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG / "03_breadth_change_distribution.png", dpi=150)
    plt.close()

    # Fig 4: aggregator vs direct trajectories (means)
    if len(agg_split) == 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(cluster_features))
        w = 0.35
        ax.bar(x - w/2, agg_split[agg_split["agg_heavy"]==0][cluster_features].values[0],
               width=w, label="direct employers", alpha=0.8)
        ax.bar(x + w/2, agg_split[agg_split["agg_heavy"]==1][cluster_features].values[0],
               width=w, label="aggregator-heavy", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cluster_features, rotation=35, ha="right")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("mean Δ per-company")
        ax.set_title("Aggregator vs direct employer change trajectories (overlap panel)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT_FIG / "04_aggregator_vs_direct.png", dpi=150)
        plt.close()

    # Save full decomposition comparison (primary vs no_specialist vs no_agg vs pooled vs cap50)
    all_dec = pd.concat([decomp_primary, decomp_nospec, decomp_noagg, decomp_pooled, decomp_cap])
    all_dec.to_csv(OUT_TBL / "17_all_decompositions_combined.csv", index=False)

    # Save the length-residualization coefficients
    with open(OUT_TBL / "00_residualization_coefs.json", "w") as f:
        json.dump({"a": float(a), "b": float(b)}, f, indent=2)

    print("[T16] Done.")


if __name__ == "__main__":
    main()
