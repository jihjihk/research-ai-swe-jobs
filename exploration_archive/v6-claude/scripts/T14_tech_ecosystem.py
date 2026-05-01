"""T14: Technology ecosystem mapping.

Uses the shared tech matrix (exploration/artifacts/shared/swe_tech_matrix.parquet)
joined to the shared cleaned-text parquet (metadata only).

Outputs under exploration/figures/T14/ and exploration/tables/T14/.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from networkx.algorithms.community import greedy_modularity_communities
from scipy.stats import chi2_contingency, rankdata

REPO = Path("/home/jihgaboot/gabor/job-research")
SHARED = REPO / "exploration" / "artifacts" / "shared"
TBL = REPO / "exploration" / "tables" / "T14"
FIG = REPO / "exploration" / "figures" / "T14"
TBL.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

UNIFIED = REPO / "data" / "unified.parquet"


def load_joined() -> pd.DataFrame:
    print("[load] reading shared tech matrix + metadata")
    tm = pq.read_table(SHARED / "swe_tech_matrix.parquet").to_pandas()
    meta = pq.read_table(
        SHARED / "swe_cleaned_text.parquet",
        columns=[
            "uid",
            "text_source",
            "period",
            "seniority_final",
            "seniority_3level",
            "is_aggregator",
            "company_name_canonical",
            "source",
            "swe_classification_tier",
        ],
    ).to_pandas()
    df = meta.merge(tm, on="uid", how="inner")
    df["period_bucket"] = np.where(df["period"].str.startswith("2024"), "2024", "2026")
    return df


TECH_COLS_GLOBAL: list[str] = []


def step1_mention_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Binary mention rate per tech by period x seniority_3level."""
    print("[step1] mention rates by period x seniority")
    tech_cols = [c for c in df.columns if c not in _META_COLS]
    TECH_COLS_GLOBAL.extend(tech_cols)

    rows = []
    # overall per period
    for per in ["2024", "2026"]:
        sub = df[df["period_bucket"] == per]
        for tech in tech_cols:
            rows.append(
                {
                    "tech": tech,
                    "period": per,
                    "seniority_3level": "all",
                    "n": len(sub),
                    "mentions": int(sub[tech].sum()),
                    "rate": float(sub[tech].mean()),
                }
            )
    # per seniority_3level
    for per in ["2024", "2026"]:
        for sen in ["junior", "mid", "senior", "unknown"]:
            sub = df[(df["period_bucket"] == per) & (df["seniority_3level"] == sen)]
            if len(sub) == 0:
                continue
            for tech in tech_cols:
                rows.append(
                    {
                        "tech": tech,
                        "period": per,
                        "seniority_3level": sen,
                        "n": len(sub),
                        "mentions": int(sub[tech].sum()),
                        "rate": float(sub[tech].mean()),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(TBL / "tech_mention_rates_by_period_seniority.csv", index=False)
    return out


_META_COLS = {
    "uid",
    "text_source",
    "period",
    "seniority_final",
    "seniority_3level",
    "is_aggregator",
    "company_name_canonical",
    "source",
    "swe_classification_tier",
    "period_bucket",
}


def step2_ai_decomposition(df: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """Decompose the AI-mention SNR 925 signal: which specific AI tools dominate?"""
    print("[step2] AI mention decomposition")
    ai_techs = [
        "llm",
        "langchain",
        "langgraph",
        "rag",
        "vector_db",
        "pinecone",
        "chromadb",
        "huggingface",
        "openai_api",
        "claude_api",
        "prompt_engineering",
        "fine_tuning",
        "mcp",
        "agents_framework",
        "gpt",
        "transformer_arch",
        "embedding",
        "copilot",
        "cursor_tool",
        "chatgpt",
        "claude_tool",
        "gemini_tool",
        "codex_tool",
        "machine_learning",
        "deep_learning",
        "nlp",
        "computer_vision",
        "tensorflow",
        "pytorch",
        "scikit_learn",
        "keras",
        "xgboost",
    ]
    # per-tech SNR: cross-period effect / within-2024 effect
    df2024 = df[df["period_bucket"] == "2024"]
    arsh = df2024[df2024["source"] == "kaggle_arshkon"]
    asa = df2024[df2024["source"] == "kaggle_asaniczka"]
    scraped = df[df["source"] == "scraped"]

    rows = []
    for tech in ai_techs:
        if tech not in df.columns:
            continue
        rate_2024 = float(df2024[tech].mean())
        rate_2026 = float(scraped[tech].mean())
        within_24 = abs(float(arsh[tech].mean()) - float(asa[tech].mean()))
        cross = abs(rate_2026 - rate_2024)
        snr = (cross / within_24) if within_24 > 1e-9 else float("inf")
        rows.append(
            {
                "tech": tech,
                "rate_2024": rate_2024,
                "rate_2026": rate_2026,
                "abs_rise_pp": (rate_2026 - rate_2024) * 100,
                "multiplicative_rise": (rate_2026 / rate_2024) if rate_2024 > 1e-9 else float("inf"),
                "arshkon_rate": float(arsh[tech].mean()),
                "asaniczka_rate": float(asa[tech].mean()),
                "within_2024_effect": within_24,
                "cross_period_effect": cross,
                "snr": snr,
            }
        )
    out = pd.DataFrame(rows).sort_values("snr", ascending=False)
    out.to_csv(TBL / "ai_tech_per_tech_calibration.csv", index=False)

    # Absolute contribution to the AI-any rise
    # First define ai_any = union of ai_tool + ai_domain tech columns used in the calibration
    ai_any_cols = [t for t in ai_techs if t in df.columns]
    df["_ai_any"] = df[ai_any_cols].any(axis=1).astype(int)
    rate_ai_2024 = float(df[df["period_bucket"] == "2024"]["_ai_any"].mean())
    rate_ai_2026 = float(df[df["period_bucket"] == "2026"]["_ai_any"].mean())
    total_rise_pp = (rate_ai_2026 - rate_ai_2024) * 100

    with (TBL / "ai_decomposition_summary.txt").open("w") as fh:
        fh.write(f"ai_any_2024_rate={rate_ai_2024:.4f}\n")
        fh.write(f"ai_any_2026_rate={rate_ai_2026:.4f}\n")
        fh.write(f"ai_any_absolute_rise_pp={total_rise_pp:.2f}\n")
        fh.write(
            f"ai_any_multiplicative_rise={rate_ai_2026 / rate_ai_2024 if rate_ai_2024 > 0 else float('inf'):.2f}x\n"
        )
        top = out.nlargest(10, "abs_rise_pp")[
            ["tech", "rate_2024", "rate_2026", "abs_rise_pp", "snr"]
        ]
        fh.write("\nTop 10 techs by absolute rise (pp):\n")
        fh.write(top.to_string(index=False))

    df.drop(columns=["_ai_any"], inplace=True)
    return out


def step3_cooccurrence_network(df: pd.DataFrame) -> dict:
    """Phi-coefficient network, threshold phi>0.15, Louvain/greedy community detection.

    Cap 50/company per period before computing co-occurrences.
    """
    print("[step3] tech co-occurrence network (cap 50/company)")
    tech_cols = [c for c in df.columns if c not in _META_COLS]

    # Cap 50/company per period
    def _cap(d: pd.DataFrame, n: int = 50) -> pd.DataFrame:
        return d.groupby("company_name_canonical", group_keys=False).head(n)

    results = {}
    for per in ["2024", "2026"]:
        sub = df[df["period_bucket"] == per].copy()
        sub = _cap(sub, 50)
        mat = sub[tech_cols].astype(int).values
        # prevalence
        prev = mat.mean(axis=0)
        keep_mask = prev >= 0.01
        if keep_mask.sum() < 5:
            print(f"[step3] {per}: too few prevalent techs, skip")
            continue
        tech_keep = [t for t, k in zip(tech_cols, keep_mask) if k]
        mat_k = sub[tech_keep].astype(int).values
        n = mat_k.shape[0]

        # phi coefficient via 2x2 contingency.
        # phi(a,b) = (n11*n00 - n10*n01) / sqrt((n1.)(n0.)(n.1)(n.0))
        n1 = mat_k.sum(axis=0)
        n0 = n - n1
        # n11[i,j] = dot product
        n11 = mat_k.T @ mat_k  # (k,k)
        n10 = n1[:, None] - n11  # a=1, b=0
        n01 = n1[None, :] - n11  # a=0, b=1
        n00 = n - n11 - n10 - n01
        denom = np.sqrt(n1[:, None].astype(float) * n0[:, None] * n1[None, :] * n0[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            phi = (n11 * n00 - n10 * n01) / np.where(denom == 0, 1, denom)
        np.fill_diagonal(phi, 0.0)

        G = nx.Graph()
        for t in tech_keep:
            G.add_node(t)
        thresh = 0.15
        k = len(tech_keep)
        for i in range(k):
            for j in range(i + 1, k):
                if phi[i, j] >= thresh:
                    G.add_edge(tech_keep[i], tech_keep[j], weight=float(phi[i, j]))

        # community detection
        comms = list(greedy_modularity_communities(G, weight="weight"))
        comms_sorted = [sorted(c) for c in comms]
        comms_sorted.sort(key=lambda c: -len(c))

        modularity = nx.algorithms.community.quality.modularity(G, comms, weight="weight")

        results[per] = {
            "graph": G,
            "phi": phi,
            "tech_keep": tech_keep,
            "communities": comms_sorted,
            "modularity": modularity,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "n_rows_capped": int(n),
        }
        print(
            f"[step3] {per}: n_rows_capped={n}, nodes={G.number_of_nodes()},"
            f" edges={G.number_of_edges()}, n_communities={len(comms_sorted)},"
            f" modularity={modularity:.3f}"
        )

    # Write community tables
    for per, res in results.items():
        comm_rows = []
        for i, c in enumerate(res["communities"]):
            for t in c:
                comm_rows.append({"period": per, "community_id": i, "size": len(c), "tech": t})
        pd.DataFrame(comm_rows).to_csv(TBL / f"communities_{per}.csv", index=False)

    # Phi edgelist (above threshold) for each period
    for per, res in results.items():
        edges = [
            {"a": u, "b": v, "phi": d["weight"]} for u, v, d in res["graph"].edges(data=True)
        ]
        pd.DataFrame(edges).sort_values("phi", ascending=False).to_csv(
            TBL / f"phi_edges_{per}.csv", index=False
        )

    # Community assignment overlap (Jaccard-style) across periods
    if "2024" in results and "2026" in results:
        c24 = results["2024"]["communities"]
        c26 = results["2026"]["communities"]

        def _best_match(comm, other):
            best = 0.0
            best_idx = -1
            for j, oc in enumerate(other):
                inter = len(set(comm) & set(oc))
                union = len(set(comm) | set(oc)) or 1
                jac = inter / union
                if jac > best:
                    best = jac
                    best_idx = j
            return best_idx, best

        rows = []
        for i, c in enumerate(c24):
            j, jac = _best_match(c, c26)
            rows.append(
                {
                    "period": "2024->2026",
                    "community_2024": i,
                    "size_2024": len(c),
                    "best_match_2026": j,
                    "size_2026": len(c26[j]) if j >= 0 else 0,
                    "jaccard": jac,
                }
            )
        pd.DataFrame(rows).to_csv(TBL / "community_overlap_2024_to_2026.csv", index=False)

    return results


def step4_trajectories(rates: pd.DataFrame) -> pd.DataFrame:
    """Rising/stable/declining classification."""
    print("[step4] trajectory classification")
    r = rates[rates["seniority_3level"] == "all"].pivot(index="tech", columns="period", values="rate")
    r["delta_pp"] = (r.get("2026", 0) - r.get("2024", 0)) * 100
    r["mult"] = np.where(
        r.get("2024", 0) > 0, r.get("2026", 0) / r.get("2024", 0), np.inf
    )

    def classify(row):
        if row["delta_pp"] >= 2.0 and (row["mult"] >= 1.2 or row.get("2024", 0) < 0.01):
            return "rising"
        if row["delta_pp"] <= -2.0 or row["mult"] <= 0.8:
            return "declining"
        return "stable"

    r["trajectory"] = r.apply(classify, axis=1)
    r = r.sort_values("delta_pp", ascending=False).reset_index()
    r.to_csv(TBL / "tech_trajectories.csv", index=False)

    # heatmap of top 40 by absolute change
    top = r.reindex(r["delta_pp"].abs().sort_values(ascending=False).index).head(40)
    plot_df = top.set_index("tech")[["2024", "2026"]] * 100
    fig, ax = plt.subplots(figsize=(8, 12))
    sns.heatmap(
        plot_df, annot=True, fmt=".1f", cmap="RdYlBu_r", cbar_kws={"label": "mention rate (%)"}, ax=ax
    )
    ax.set_title("Tech mention rate: 2024 vs 2026 (top 40 by |Δpp|)")
    plt.tight_layout()
    plt.savefig(FIG / "tech_shift_heatmap.png", dpi=150)
    plt.close()
    return r


def step5_stack_diversity(df: pd.DataFrame) -> pd.DataFrame:
    print("[step5] stack diversity")
    tech_cols = [c for c in df.columns if c not in _META_COLS]
    df["_tech_count"] = df[tech_cols].sum(axis=1)
    rows = []
    for per in ["2024", "2026"]:
        for sen in ["all", "junior", "mid", "senior", "unknown"]:
            if sen == "all":
                sub = df[df["period_bucket"] == per]
            else:
                sub = df[(df["period_bucket"] == per) & (df["seniority_3level"] == sen)]
            if len(sub) == 0:
                continue
            rows.append(
                {
                    "period": per,
                    "seniority_3level": sen,
                    "n": len(sub),
                    "median": float(sub["_tech_count"].median()),
                    "mean": float(sub["_tech_count"].mean()),
                    "q25": float(sub["_tech_count"].quantile(0.25)),
                    "q75": float(sub["_tech_count"].quantile(0.75)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TBL / "stack_diversity.csv", index=False)
    df.drop(columns=["_tech_count"], inplace=True)
    return out


def step6_ai_integration_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Among AI-mentioning postings, what traditional tech co-occurs?
    Report both raw count and density per 1K chars.
    """
    print("[step6] AI integration pattern + length normalization")
    tech_cols = [c for c in df.columns if c not in _META_COLS]
    ai_cols = [
        c
        for c in [
            "llm",
            "rag",
            "copilot",
            "cursor_tool",
            "openai_api",
            "claude_api",
            "langchain",
            "langgraph",
            "prompt_engineering",
            "fine_tuning",
            "mcp",
            "agents_framework",
            "chatgpt",
            "claude_tool",
            "gemini_tool",
            "huggingface",
            "vector_db",
            "pinecone",
            "chromadb",
            "gpt",
            "embedding",
            "transformer_arch",
        ]
        if c in df.columns
    ]
    df["_ai_any"] = df[ai_cols].any(axis=1).astype(int)

    # Get description length from the unified data (single targeted DuckDB query)
    conn = duckdb.connect()
    uids = df["uid"].tolist()
    query = f"""
    SELECT uid, LENGTH(description) AS desc_len
    FROM read_parquet('{UNIFIED}')
    WHERE uid IN (SELECT unnest(?))
    """
    lens = conn.execute(query, [uids]).df()
    conn.close()
    df = df.merge(lens, on="uid", how="left")
    df["desc_len"] = df["desc_len"].fillna(0)

    df["_tech_count"] = df[tech_cols].sum(axis=1)
    # density: tech per 1K chars, guard against zero
    df["_tech_density"] = np.where(
        df["desc_len"] > 100, df["_tech_count"] * 1000.0 / df["desc_len"], 0.0
    )

    rows = []
    for per in ["2024", "2026"]:
        for ai in [0, 1]:
            sub = df[(df["period_bucket"] == per) & (df["_ai_any"] == ai)]
            if len(sub) == 0:
                continue
            rows.append(
                {
                    "period": per,
                    "ai_mentioning": bool(ai),
                    "n": len(sub),
                    "mean_tech_count": float(sub["_tech_count"].mean()),
                    "median_tech_count": float(sub["_tech_count"].median()),
                    "mean_tech_density_per_1k": float(sub["_tech_density"].mean()),
                    "median_tech_density_per_1k": float(sub["_tech_density"].median()),
                    "mean_desc_len": float(sub["desc_len"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TBL / "ai_integration_tech_density.csv", index=False)

    # co-occurring tech among AI postings (2026)
    sub26 = df[(df["period_bucket"] == "2026") & (df["_ai_any"] == 1)]
    non_ai = [c for c in tech_cols if c not in ai_cols]
    cooc = []
    for t in non_ai:
        cooc.append({"tech": t, "rate_among_ai_2026": float(sub26[t].mean())})
    cooc_df = pd.DataFrame(cooc).sort_values("rate_among_ai_2026", ascending=False)
    cooc_df.to_csv(TBL / "tech_cooccurring_with_ai_2026.csv", index=False)

    # plot raw vs density
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plot_df = out.copy()
    plot_df["group"] = plot_df["period"] + " / " + np.where(plot_df["ai_mentioning"], "AI", "non-AI")
    sns.barplot(data=plot_df, x="group", y="mean_tech_count", ax=axes[0], palette="Set2")
    axes[0].set_title("Mean tech count per posting")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=30)
    sns.barplot(
        data=plot_df, x="group", y="mean_tech_density_per_1k", ax=axes[1], palette="Set2"
    )
    axes[1].set_title("Mean tech density (per 1K chars)")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(FIG / "ai_density_vs_count.png", dpi=150)
    plt.close()

    df.drop(columns=["_ai_any", "_tech_count", "_tech_density", "desc_len"], inplace=True)
    return out


def step7_structured_skills_baseline() -> pd.DataFrame:
    print("[step7] structured skills baseline")
    s = pq.read_table(SHARED / "asaniczka_structured_skills.parquet").to_pandas()
    freq = s["skill"].value_counts().reset_index()
    freq.columns = ["skill", "count"]
    freq["share_of_uids"] = freq["count"] / s["uid"].nunique()
    freq.head(1000).to_csv(TBL / "structured_skills_top1000.csv", index=False)
    freq.head(100).to_csv(TBL / "structured_skills_top100.csv", index=False)
    return freq


def step8_structured_vs_extracted(
    df: pd.DataFrame, struct_freq: pd.DataFrame
) -> pd.DataFrame:
    """Rank correlation between parsed structured skills and regex tech prevalence (asaniczka SWE)."""
    print("[step8] structured vs extracted validation (asaniczka SWE)")
    asa = df[df["source"] == "kaggle_asaniczka"]
    # Map tech columns to canonical skill tokens
    mapping = {
        "python": ["python"],
        "java": ["java"],
        "javascript": ["javascript"],
        "typescript": ["typescript"],
        "go_lang": ["go", "golang"],
        "rust": ["rust"],
        "c_plus_plus": ["c++"],
        "c_sharp": ["c#"],
        "ruby": ["ruby"],
        "kotlin": ["kotlin"],
        "swift": ["swift"],
        "scala": ["scala"],
        "php": ["php"],
        "react": ["react", "react.js", "reactjs"],
        "angular": ["angular"],
        "vue": ["vue", "vue.js", "vuejs"],
        "nodejs": ["node.js", "nodejs", "node"],
        "django": ["django"],
        "flask": ["flask"],
        "spring": ["spring", "spring boot"],
        "dotnet": [".net", "dotnet"],
        "aws": ["aws", "amazon web services"],
        "azure": ["azure", "microsoft azure"],
        "gcp": ["gcp", "google cloud", "google cloud platform"],
        "kubernetes": ["kubernetes", "k8s"],
        "docker": ["docker"],
        "terraform": ["terraform"],
        "cicd": ["ci/cd", "cicd"],
        "jenkins": ["jenkins"],
        "git": ["git"],
        "sql_lang": ["sql"],
        "postgres": ["postgresql", "postgres"],
        "mysql": ["mysql"],
        "mongodb": ["mongodb", "mongo"],
        "redis": ["redis"],
        "kafka": ["kafka", "apache kafka"],
        "spark": ["spark", "apache spark"],
        "snowflake": ["snowflake"],
        "databricks": ["databricks"],
        "elasticsearch": ["elasticsearch"],
        "airflow": ["airflow", "apache airflow"],
        "bigquery": ["bigquery"],
        "tensorflow": ["tensorflow"],
        "pytorch": ["pytorch"],
        "scikit_learn": ["scikit-learn", "sklearn"],
        "pandas": ["pandas"],
        "numpy": ["numpy"],
        "machine_learning": ["machine learning"],
        "deep_learning": ["deep learning"],
        "nlp": ["nlp", "natural language processing"],
        "computer_vision": ["computer vision"],
        "llm": ["llm", "large language models", "llms"],
        "rest_api": ["rest", "rest api", "restful"],
        "graphql": ["graphql"],
        "microservices": ["microservices"],
        "agile": ["agile"],
        "scrum": ["scrum"],
        "linux": ["linux"],
    }

    struct_lookup = dict(zip(struct_freq["skill"].str.lower(), struct_freq["count"]))
    n_asa = asa.shape[0]
    n_uids = asa["uid"].nunique()

    rows = []
    for tech, alts in mapping.items():
        if tech not in asa.columns:
            continue
        extracted_rate = float(asa[tech].mean())
        struct_count = sum(struct_lookup.get(a, 0) for a in alts)
        struct_rate = struct_count / max(n_uids, 1)
        rows.append(
            {
                "tech": tech,
                "extracted_rate": extracted_rate,
                "structured_rate": struct_rate,
                "extracted_count": int(asa[tech].sum()),
                "structured_count": struct_count,
            }
        )
    out = pd.DataFrame(rows).sort_values("extracted_rate", ascending=False)

    from scipy.stats import spearmanr, pearsonr

    spr, sp = spearmanr(out["extracted_rate"], out["structured_rate"])
    pr, pp = pearsonr(out["extracted_rate"], out["structured_rate"])

    with (TBL / "structured_vs_extracted_summary.txt").open("w") as fh:
        fh.write(f"n_techs={len(out)}\n")
        fh.write(f"spearman_rho={spr:.4f} (p={sp:.2e})\n")
        fh.write(f"pearson_r={pr:.4f} (p={pp:.2e})\n")
        fh.write(f"n_asaniczka_rows={n_asa}, distinct_uids={n_uids}\n")
        out["diff_pp"] = (out["structured_rate"] - out["extracted_rate"]) * 100
        fh.write("\nTop 10 where structured > extracted (structured catches more):\n")
        fh.write(
            out.sort_values("diff_pp", ascending=False)
            .head(10)
            .to_string(index=False)
        )
        fh.write("\n\nTop 10 where extracted > structured (regex catches more):\n")
        fh.write(
            out.sort_values("diff_pp", ascending=True)
            .head(10)
            .to_string(index=False)
        )

    out.to_csv(TBL / "structured_vs_extracted.csv", index=False)
    return out


def step9_seniority_skills_chi2() -> pd.DataFrame:
    """Chi-squared per skill, FDR correction. Asaniczka parsed skills vs seniority_final."""
    print("[step9] seniority-level skill differences (chi2 + FDR)")
    s = pq.read_table(SHARED / "asaniczka_structured_skills.parquet").to_pandas()
    # Pull seniority_3level for asaniczka SWE uids from shared cleaned text parquet
    meta = pq.read_table(
        SHARED / "swe_cleaned_text.parquet",
        columns=["uid", "seniority_3level", "source"],
    ).to_pandas()
    meta = meta[meta["source"] == "kaggle_asaniczka"]
    s = s.merge(meta[["uid", "seniority_3level"]], on="uid", how="inner")

    # Define contrast: junior vs senior (drop mid + unknown)
    s = s[s["seniority_3level"].isin(["junior", "mid", "senior"])]
    # keep mid with senior? Many asaniczka rows have 'mid' or 'senior' via LLM; treat junior vs (mid+senior).
    s["class_ms"] = np.where(s["seniority_3level"] == "junior", "junior", "mid_senior")

    uid_level = s[["uid", "class_ms"]].drop_duplicates()
    n_junior = (uid_level["class_ms"] == "junior").sum()
    n_mid_senior = (uid_level["class_ms"] == "mid_senior").sum()

    # Per-skill 2x2: has_skill × class
    # Cap by skill frequency >= 50 to keep test meaningful
    top_skills = s["skill"].value_counts()
    top_skills = top_skills[top_skills >= 50].index.tolist()

    # Build presence: (uid, skill) boolean
    # For each skill: build a uid set that has it, then count per class
    uid_class = dict(zip(uid_level["uid"], uid_level["class_ms"]))

    rows = []
    for skill in top_skills:
        uids_with = s[s["skill"] == skill]["uid"].unique()
        # class counts for uids with skill
        j_with = sum(1 for u in uids_with if uid_class.get(u) == "junior")
        m_with = sum(1 for u in uids_with if uid_class.get(u) == "mid_senior")
        j_without = n_junior - j_with
        m_without = n_mid_senior - m_with
        if j_with + m_with < 20:
            continue
        table = np.array([[j_with, m_with], [j_without, m_without]])
        try:
            chi2, p, _, _ = chi2_contingency(table)
        except ValueError:
            continue
        rate_j = j_with / max(n_junior, 1)
        rate_m = m_with / max(n_mid_senior, 1)
        rows.append(
            {
                "skill": skill,
                "j_with": j_with,
                "m_with": m_with,
                "rate_junior": rate_j,
                "rate_mid_senior": rate_m,
                "diff_pp": (rate_j - rate_m) * 100,
                "chi2": chi2,
                "p_value": p,
            }
        )
    out = pd.DataFrame(rows)

    # Benjamini-Hochberg FDR
    out = out.sort_values("p_value").reset_index(drop=True)
    m = len(out)
    out["rank"] = np.arange(1, m + 1)
    out["p_fdr"] = out["p_value"] * m / out["rank"]
    out["p_fdr"] = out["p_fdr"].clip(upper=1.0)
    out["significant_fdr05"] = out["p_fdr"] < 0.05

    out.to_csv(TBL / "asaniczka_skill_seniority_chi2.csv", index=False)
    with (TBL / "asaniczka_skill_seniority_summary.txt").open("w") as fh:
        fh.write(f"n_junior_uids={n_junior}, n_mid_senior_uids={n_mid_senior}\n")
        fh.write(f"skills_tested={m}, significant_fdr05={int(out['significant_fdr05'].sum())}\n")
        fh.write("\nTop 15 skills OVER-represented in junior:\n")
        sig = out[out["significant_fdr05"]]
        fh.write(sig.sort_values("diff_pp", ascending=False).head(15).to_string(index=False))
        fh.write("\n\nTop 15 skills OVER-represented in mid+senior:\n")
        fh.write(sig.sort_values("diff_pp", ascending=True).head(15).to_string(index=False))
    return out


def step10_sensitivities(df: pd.DataFrame) -> dict:
    """(a) aggregator exclusion, (b) company capping, (f) within-2024 calibration."""
    print("[step10] sensitivities")
    tech_cols = [c for c in df.columns if c not in _META_COLS]

    results: dict = {}

    # (a) aggregator exclusion — recompute top-tech mention rate shifts
    def _rates(d):
        return {t: float(d[t].mean()) for t in tech_cols}

    base = _rates(df)
    excl_agg = _rates(df[~df["is_aggregator"].fillna(False)])
    rows = []
    for t in tech_cols:
        rows.append(
            {
                "tech": t,
                "rate_all": base[t],
                "rate_excl_aggregator": excl_agg[t],
                "diff_pp": (base[t] - excl_agg[t]) * 100,
            }
        )
    pd.DataFrame(rows).to_csv(TBL / "sensitivity_aggregator_exclusion.csv", index=False)

    # (b) company capping — compare uncapped vs 50-cap
    capped = df.groupby(["period_bucket", "company_name_canonical"], group_keys=False).head(50)
    rate_rows = []
    for per in ["2024", "2026"]:
        sub_full = df[df["period_bucket"] == per]
        sub_cap = capped[capped["period_bucket"] == per]
        for t in tech_cols:
            rate_rows.append(
                {
                    "tech": t,
                    "period": per,
                    "rate_uncapped": float(sub_full[t].mean()),
                    "rate_cap50": float(sub_cap[t].mean()),
                }
            )
    pd.DataFrame(rate_rows).to_csv(TBL / "sensitivity_company_capping.csv", index=False)

    # (f) within-2024 calibration per tech
    arsh = df[df["source"] == "kaggle_arshkon"]
    asa = df[df["source"] == "kaggle_asaniczka"]
    scraped = df[df["source"] == "scraped"]
    cal_rows = []
    for t in tech_cols:
        a = float(arsh[t].mean())
        z = float(asa[t].mean())
        sc = float(scraped[t].mean())
        pool24 = float(df[df["period_bucket"] == "2024"][t].mean())
        within = abs(a - z)
        cross = abs(sc - pool24)
        snr = cross / within if within > 1e-9 else float("inf")
        cal_rows.append(
            {
                "tech": t,
                "arshkon": a,
                "asaniczka": z,
                "scraped_2026": sc,
                "within_2024_effect": within,
                "cross_period_effect": cross,
                "snr": snr,
            }
        )
    cal_df = pd.DataFrame(cal_rows).sort_values("snr", ascending=False)
    cal_df.to_csv(TBL / "per_tech_calibration_extension.csv", index=False)
    results["calibration"] = cal_df
    return results


def visualize_network(results: dict) -> None:
    """Network visualization for 2024 vs 2026 with community coloring."""
    print("[vis] co-occurrence network")
    if "2026" not in results or "2024" not in results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    for ax, per in zip(axes, ["2024", "2026"]):
        res = results[per]
        G = res["graph"]
        # color by community
        comm_map = {}
        for i, c in enumerate(res["communities"]):
            for t in c:
                comm_map[t] = i
        colors = [comm_map.get(n, -1) for n in G.nodes()]
        pos = nx.spring_layout(G, k=0.5, seed=42, iterations=50)
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
        nx.draw_networkx_nodes(
            G, pos, node_color=colors, cmap="tab20", node_size=200, ax=ax
        )
        # label only top-degree nodes
        degs = dict(G.degree())
        top_nodes = sorted(degs, key=degs.get, reverse=True)[:20]
        nx.draw_networkx_labels(
            G, pos, labels={n: n for n in top_nodes}, font_size=7, ax=ax
        )
        ax.set_title(
            f"{per}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges,"
            f" {len(res['communities'])} communities, Q={res['modularity']:.2f}"
        )
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIG / "cooccurrence_network_2024_2026.png", dpi=150)
    plt.close()

    # community comparison bar chart: top-10 community sizes
    fig, ax = plt.subplots(figsize=(10, 6))
    data = []
    for per in ["2024", "2026"]:
        sizes = [len(c) for c in results[per]["communities"][:10]]
        for i, s in enumerate(sizes):
            data.append({"period": per, "community_rank": i + 1, "size": s})
    d = pd.DataFrame(data)
    sns.barplot(data=d, x="community_rank", y="size", hue="period", ax=ax)
    ax.set_title("Top-10 community sizes: 2024 vs 2026")
    plt.tight_layout()
    plt.savefig(FIG / "community_sizes_comparison.png", dpi=150)
    plt.close()


def main():
    df = load_joined()
    rates = step1_mention_rates(df)
    step2_ai_decomposition(df, rates)
    net_results = step3_cooccurrence_network(df)
    visualize_network(net_results)
    step4_trajectories(rates)
    step5_stack_diversity(df)
    step6_ai_integration_pattern(df)
    struct_freq = step7_structured_skills_baseline()
    step8_structured_vs_extracted(df, struct_freq)
    step9_seniority_skills_chi2()
    step10_sensitivities(df)
    print("[done] T14 complete")


if __name__ == "__main__":
    main()
