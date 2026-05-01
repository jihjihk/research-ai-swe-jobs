from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr
from statsmodels.stats.multitest import multipletests

from T14_T15_common import (
    SHARED_DIR,
    ensure_dir,
    load_cleaned_text,
    load_structured_skills,
    load_tech_matrix,
    linkedin_swe_full_meta,
    skill_to_tech_col,
    tech_columns,
    validate_skill_aliases,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ensure_dir(ROOT / "exploration" / "reports")
TABLE_DIR = ensure_dir(ROOT / "exploration" / "tables" / "T14")
FIG_DIR = ensure_dir(ROOT / "exploration" / "figures" / "T14")

CAP_PER_COMPANY = 30
TOP_HEATMAP_N = 25
TOP_SKILLS_N = 100


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def tech_category(name: str) -> str:
    if name in {
        "python",
        "java",
        "javascript",
        "typescript",
        "go",
        "rust",
        "c_plus_plus",
        "c_sharp",
        "ruby",
        "kotlin",
        "swift",
        "scala",
        "php",
        "perl",
        "dart",
        "lua",
        "haskell",
        "clojure",
        "elixir",
        "erlang",
        "julia",
        "matlab",
        "r_language",
    }:
        return "language"
    if name in {
        "react",
        "react_native",
        "angular",
        "vue",
        "nextjs",
        "svelte",
        "redux",
        "webpack",
        "vite",
        "storybook",
        "tailwind",
        "bootstrap",
        "material_ui",
    }:
        return "frontend"
    if name in {
        "nodejs",
        "express",
        "nestjs",
        "django",
        "django_rest_framework",
        "flask",
        "spring",
        "spring_boot",
        "dotnet",
        "aspnet",
        "rails",
        "laravel",
        "fastapi",
        "phoenix",
        "tornado",
        "bottle",
        "grpc",
        "graphql",
        "rest_api",
        "microservices",
        "event_driven",
        "oauth",
    }:
        return "backend"
    if name in {
        "sql",
        "postgresql",
        "mysql",
        "sqlite",
        "mongodb",
        "redis",
        "cassandra",
        "kafka",
        "spark",
        "hadoop",
        "hive",
        "presto",
        "trino",
        "snowflake",
        "databricks",
        "dbt",
        "elasticsearch",
        "airflow",
        "luigi",
        "airbyte",
        "delta_lake",
        "tableau",
        "powerbi",
        "looker",
        "superset",
        "metabase",
        "bigquery",
        "redshift",
    }:
        return "data"
    if name in {
        "aws",
        "azure",
        "gcp",
        "kubernetes",
        "docker",
        "terraform",
        "ansible",
        "helm",
        "jenkins",
        "github_actions",
        "gitlab_ci",
        "circleci",
        "argo_cd",
        "openshift",
        "nomad",
        "prometheus",
        "grafana",
        "datadog",
        "new_relic",
        "splunk",
        "opentelemetry",
        "linux",
        "bash",
        "git",
        "serverless",
        "cloudformation",
        "pulumi",
    }:
        return "cloud_devops"
    if name in {
        "junit",
        "pytest",
        "jest",
        "mocha",
        "chai",
        "selenium",
        "cypress",
        "playwright",
        "tdd",
        "agile",
        "scrum",
        "kanban",
        "ci_cd",
        "code_review",
        "pair_programming",
        "unit_testing",
        "integration_testing",
        "bdd",
        "qa",
    }:
        return "method_testing"
    if name in {
        "machine_learning",
        "deep_learning",
        "data_science",
        "statistics",
        "nlp",
        "computer_vision",
        "generative_ai",
        "tensorflow",
        "pytorch",
        "scikit_learn",
        "pandas",
        "numpy",
        "jupyter",
        "xgboost",
        "lightgbm",
        "catboost",
        "mlflow",
        "kubeflow",
        "ray",
        "hugging_face",
    }:
        return "ai_domain"
    if name in {
        "openai_api",
        "anthropic_api",
        "claude_api",
        "gemini_api",
        "langchain",
        "langgraph",
        "llamaindex",
        "rag",
        "vector_db",
        "pinecone",
        "weaviate",
        "chroma",
        "milvus",
        "faiss",
        "prompt_engineering",
        "fine_tuning",
        "mcp",
        "llm",
        "copilot",
        "cursor",
        "chatgpt",
        "claude",
        "gemini",
        "codex",
        "agent",
    }:
        return "ai_tool"
    return "other"


def build_frame() -> pd.DataFrame:
    meta = linkedin_swe_full_meta()
    text = load_cleaned_text(["uid", "text_source"])
    tech = load_tech_matrix()
    df = meta.merge(text, on="uid", how="inner").merge(tech, on="uid", how="inner")
    df["company_key"] = (
        df["company_name_canonical"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "unknown_company")
    )
    return df


def tech_rate_table(df: pd.DataFrame, tech_cols: list[str]) -> pd.DataFrame:
    rows = []
    group_cols = ["source", "period", "seniority_3level"]
    grouped = df.groupby(group_cols, dropna=False)
    for keys, sub in grouped:
        base = {
            "source": keys[0],
            "period": keys[1],
            "seniority_3level": keys[2],
            "n": int(len(sub)),
        }
        rates = sub[tech_cols].mean(axis=0)
        for tech, rate in rates.items():
            rows.append(
                {
                    **base,
                    "tech": tech,
                    "category": tech_category(tech),
                    "share": float(rate),
                }
            )
    return pd.DataFrame(rows)


def source_share_table(df: pd.DataFrame, tech_cols: list[str]) -> pd.DataFrame:
    rows = []
    for source, sub in df.groupby("source", dropna=False):
        rows.append(
            {
                "source": source,
                "n": int(len(sub)),
                **{tech: float(sub[tech].mean()) for tech in tech_cols},
            }
        )
    return pd.DataFrame(rows)


def company_capped_frame(df: pd.DataFrame, cap: int = CAP_PER_COMPANY) -> pd.DataFrame:
    out = df.sort_values(["source", "period", "company_key", "uid"]).copy()
    out["company_rank"] = out.groupby(["source", "period", "company_key"]).cumcount()
    return out[out["company_rank"] < cap].copy()


def cooccurrence_network(sub: pd.DataFrame, tech_cols: list[str], cap: int = CAP_PER_COMPANY):
    capped = company_capped_frame(sub, cap=cap)
    freq = capped[tech_cols].mean(axis=0)
    keep = freq[(freq > 0.01) & (freq < 0.99)].index.tolist()
    X = capped[keep].astype(np.int8).to_numpy()
    n = X.shape[0]
    means = X.mean(axis=0)
    Xc = X - means
    cov = (Xc.T @ Xc) / n
    std = np.sqrt(means * (1 - means))
    denom = np.outer(std, std)
    phi = np.divide(cov, denom, out=np.zeros_like(cov, dtype=float), where=denom != 0)
    np.fill_diagonal(phi, 1.0)

    G = nx.Graph()
    for i, tech in enumerate(keep):
        G.add_node(
            tech,
            freq=float(means[i]),
            category=tech_category(tech),
        )
    edge_rows = []
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            if phi[i, j] > 0.15:
                G.add_edge(keep[i], keep[j], weight=float(phi[i, j]))
                edge_rows.append(
                    {
                        "tech_a": keep[i],
                        "tech_b": keep[j],
                        "phi": float(phi[i, j]),
                        "freq_a": float(means[i]),
                        "freq_b": float(means[j]),
                    }
                )
    return G, pd.DataFrame(edge_rows), capped


def detect_communities(G: nx.Graph) -> tuple[list[set[str]], dict[str, int]]:
    if G.number_of_edges() == 0:
        comms = [set(G.nodes())]
    else:
        try:
            comms = list(
                nx.algorithms.community.louvain_communities(
                    G, weight="weight", resolution=1.0, seed=42
                )
            )
        except Exception:
            comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
    comms = sorted(comms, key=len, reverse=True)
    node_to_comm = {}
    for idx, comm in enumerate(comms, start=1):
        for node in comm:
            node_to_comm[node] = idx
    return comms, node_to_comm


def community_table(source: str, G: nx.Graph, comms: list[set[str]]) -> pd.DataFrame:
    rows = []
    for idx, comm in enumerate(comms, start=1):
        members = list(comm)
        top_members = sorted(members, key=lambda t: G.nodes[t]["freq"], reverse=True)[:10]
        cats = pd.Series([G.nodes[t]["category"] for t in members]).value_counts().to_dict()
        rows.append(
            {
                "source": source,
                "community_id": idx,
                "size": len(comm),
                "top_members": ", ".join(top_members),
                "dominant_category": max(cats, key=cats.get) if cats else "other",
                "n_categories": len(cats),
            }
        )
    return pd.DataFrame(rows)


def community_overlap_table(comms_a: list[set[str]], comms_b: list[set[str]], source_a: str, source_b: str) -> pd.DataFrame:
    rows = []
    for i, ca in enumerate(comms_a, start=1):
        best_j = None
        best_overlap = -1.0
        best_cb = None
        for j, cb in enumerate(comms_b, start=1):
            jac = len(ca & cb) / len(ca | cb)
            if jac > best_overlap:
                best_overlap = jac
                best_j = j
                best_cb = cb
        rows.append(
            {
                "source_a": source_a,
                "community_a": i,
                "size_a": len(ca),
                "best_source_b": source_b,
                "community_b": best_j,
                "size_b": len(best_cb) if best_cb is not None else 0,
                "jaccard": best_overlap,
                "status": (
                    "stable"
                    if best_overlap >= 0.5
                    else "fragmented"
                    if best_overlap >= 0.25
                    else "new"
                ),
                "members_a": ", ".join(sorted(list(ca), key=lambda x: x)),
            }
        )
    return pd.DataFrame(rows)


def plot_networks(networks: dict[str, tuple[nx.Graph, list[set[str]], dict[str, int]]]) -> plt.Figure:
    fig, axes = plt.subplots(1, len(networks), figsize=(20, 7), constrained_layout=True)
    if len(networks) == 1:
        axes = [axes]
    palette = sns.color_palette("tab20", 20)
    for ax, (source, (G, comms, node_to_comm)) in zip(axes, networks.items()):
        pos = nx.spring_layout(G, seed=42, weight="weight", k=0.7)
        sizes = [250 + 5000 * G.nodes[n]["freq"] for n in G.nodes()]
        colors = [palette[(node_to_comm.get(n, 1) - 1) % len(palette)] for n in G.nodes()]
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=[1.0 + 2.5 * d["weight"] for _, _, d in G.edges(data=True)])
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors, linewidths=0.5, edgecolors="white")
        top_nodes = sorted(G.nodes(), key=lambda n: G.nodes[n]["freq"], reverse=True)[:12]
        labels = {n: n.replace("_", "\n") if len(n) > 8 else n for n in top_nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)
        ax.set_title(source.replace("kaggle_", "").replace("_", " ").title())
        ax.axis("off")
    return fig


def plot_heatmap(df: pd.DataFrame, techs: list[str]) -> plt.Figure:
    pivot = df.pivot(index="tech", columns="source", values="share").reindex(techs)
    fig, ax = plt.subplots(figsize=(10, max(8, 0.28 * len(techs))))
    sns.heatmap(
        pivot,
        cmap="viridis",
        linewidths=0.15,
        linecolor="white",
        cbar_kws={"label": "Mention share"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Tech mention rates by source")
    return fig


def plot_validation_scatter(validation: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.scatterplot(
        data=validation,
        x="structured_log",
        y="extracted_log",
        hue="category",
        palette="Set2",
        ax=ax,
        s=55,
        edgecolor="white",
        linewidth=0.4,
    )
    maxv = max(validation["structured_log"].max(), validation["extracted_log"].max())
    ax.plot([0, maxv], [0, maxv], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("log1p(structured skill count)")
    ax.set_ylabel("log1p(description-extracted tech count)")
    ax.set_title("Structured vs extracted tech frequency overlap")
    ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    return fig


def main() -> None:
    validate_skill_aliases()
    df = build_frame()
    tech_cols = tech_columns()

    # Core counts and coverage diagnostics
    counts = (
        df.groupby(["source", "period"], dropna=False)
        .agg(
            n=("uid", "size"),
            text_labeled=("text_source", lambda s: (s == "llm").sum()),
            aggregators=("is_aggregator", "sum"),
        )
        .reset_index()
    )
    counts["text_share"] = counts["text_labeled"] / counts["n"]
    save_csv(counts, TABLE_DIR / "T14_counts_by_source_period.csv")

    # Rate tables
    tech_rates = tech_rate_table(df, tech_cols)
    save_csv(tech_rates, TABLE_DIR / "T14_tech_rates_by_source_period_seniority.csv")
    source_rates = source_share_table(df, tech_cols)
    save_csv(source_rates, TABLE_DIR / "T14_tech_rates_by_source.csv")

    # Category summary
    cat_summary = (
        tech_rates.groupby(["source", "period", "category"], dropna=False)
        .agg(
            n_rows=("share", "size"),
            mean_share=("share", "mean"),
            median_share=("share", "median"),
        )
        .reset_index()
    )
    save_csv(cat_summary, TABLE_DIR / "T14_category_summary.csv")

    # Overall tech change summary vs within-2024 calibration
    source_share = source_rates.set_index("source")
    change = pd.DataFrame({"tech": tech_cols})
    change["arshkon"] = change["tech"].map(source_share.loc["kaggle_arshkon"])
    change["asaniczka"] = change["tech"].map(source_share.loc["kaggle_asaniczka"])
    change["scraped"] = change["tech"].map(source_share.loc["scraped"])
    change["within_2024_abs"] = (change["arshkon"] - change["asaniczka"]).abs()
    change["cross_abs"] = (change["scraped"] - change["arshkon"]).abs()
    change["snr"] = change["cross_abs"] / change["within_2024_abs"].replace(0, np.nan)
    change["delta_scraped_minus_arshkon"] = change["scraped"] - change["arshkon"]
    change["delta_arshkon_minus_asaniczka"] = change["arshkon"] - change["asaniczka"]
    change["category"] = change["tech"].map(tech_category)
    save_csv(change.sort_values("cross_abs", ascending=False), TABLE_DIR / "T14_tech_change_summary.csv")

    # Plot tech shift heatmap
    top_heat = change.sort_values("cross_abs", ascending=False).head(TOP_HEATMAP_N)["tech"].tolist()
    source_long = source_rates.melt(id_vars=["source", "n"], var_name="tech", value_name="share")
    heat_df = source_long[source_long["tech"].isin(top_heat)].pivot(index="tech", columns="source", values="share").reindex(top_heat)
    fig, ax = plt.subplots(figsize=(10, max(7, 0.35 * len(top_heat))))
    sns.heatmap(
        heat_df,
        cmap="mako",
        linewidths=0.2,
        linecolor="white",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Share of postings"},
        ax=ax,
    )
    ax.set_title("Largest tech-shift terms by source")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_fig(fig, FIG_DIR / "T14_tech_shift_heatmap.png")

    # Co-occurrence networks with company capping
    networks = {}
    edge_tables = []
    community_tables = []
    for source in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]:
        sub = df[df["source"] == source].copy()
        G, edges, capped = cooccurrence_network(sub, tech_cols, cap=CAP_PER_COMPANY)
        comms, node_to_comm = detect_communities(G)
        networks[source] = (G, comms, node_to_comm)
        edges["source"] = source
        edges["n_rows_capped"] = len(capped)
        edge_tables.append(edges)
        community_tables.append(community_table(source, G, comms))
        save_csv(edges, TABLE_DIR / f"T14_cooccurrence_edges_{source}.csv")
    save_csv(pd.concat(community_tables, ignore_index=True), TABLE_DIR / "T14_community_table.csv")
    save_csv(
        community_overlap_table(networks["kaggle_arshkon"][1], networks["kaggle_asaniczka"][1], "kaggle_arshkon", "kaggle_asaniczka"),
        TABLE_DIR / "T14_community_overlap_arshkon_vs_asaniczka.csv",
    )
    save_csv(
        community_overlap_table(networks["kaggle_arshkon"][1], networks["scraped"][1], "kaggle_arshkon", "scraped"),
        TABLE_DIR / "T14_community_overlap_arshkon_vs_scraped.csv",
    )
    fig = plot_networks(networks)
    save_fig(fig, FIG_DIR / "T14_cooccurrence_network.png")

    # Structured skills baseline
    skills = load_structured_skills()
    asan_meta = df[df["source"] == "kaggle_asaniczka"][["uid", "seniority_final", "yoe_extracted"]].copy()
    skills = skills.merge(asan_meta, on="uid", how="inner")
    skill_counts = skills.groupby("skill").size().sort_values(ascending=False)
    top_skills = skill_counts.head(TOP_SKILLS_N).reset_index()
    top_skills.columns = ["skill", "count"]
    top_skills["mapped_tech"] = top_skills["skill"].map(lambda s: skill_to_tech_col(s, tech_cols))
    top_skills["category"] = top_skills["mapped_tech"].map(lambda t: tech_category(t) if pd.notna(t) else "other")
    save_csv(top_skills, TABLE_DIR / "T14_structured_skills_top100.csv")

    mapped = []
    for skill, count in skill_counts.items():
        tech = skill_to_tech_col(skill, tech_cols)
        if tech is not None:
            mapped.append((skill, tech, int(count)))
    mapped_df = pd.DataFrame(mapped, columns=["skill", "tech", "structured_count"])
    validation = (
        mapped_df.groupby("tech", as_index=False)
        .agg(structured_count=("structured_count", "sum"))
        .merge(
            pd.DataFrame({"tech": tech_cols, "extracted_count": df[df["source"] == "kaggle_asaniczka"][tech_cols].sum().astype(int).values}),
            on="tech",
            how="left",
        )
    )
    validation["category"] = validation["tech"].map(tech_category)
    validation["structured_share"] = validation["structured_count"] / len(asan_meta)
    validation["extracted_share"] = validation["extracted_count"] / len(asan_meta)
    validation["structured_log"] = np.log1p(validation["structured_count"])
    validation["extracted_log"] = np.log1p(validation["extracted_count"])
    validation["rank_structured"] = validation["structured_count"].rank(ascending=False, method="min")
    validation["rank_extracted"] = validation["extracted_count"].rank(ascending=False, method="min")
    validation["rank_diff"] = validation["rank_extracted"] - validation["rank_structured"]
    validation = validation.sort_values("structured_count", ascending=False).reset_index(drop=True)
    save_csv(validation, TABLE_DIR / "T14_structured_vs_extracted_validation.csv")
    spearman = spearmanr(validation["structured_count"], validation["extracted_count"])
    save_csv(
        pd.DataFrame(
            [
                {
                    "metric": "spearman_r",
                    "value": float(spearman.statistic),
                    "p_value": float(spearman.pvalue),
                    "n_mapped_techs": int(len(validation)),
                }
            ]
        ),
        TABLE_DIR / "T14_structured_vs_extracted_summary.csv",
    )
    fig = plot_validation_scatter(validation)
    save_fig(fig, FIG_DIR / "T14_structured_vs_extracted_scatter.png")

    # Seniority-level skill differences, label-based and YOE proxy
    skill_panel = skills[skills["skill"].map(lambda s: skill_to_tech_col(s, tech_cols) is not None)].copy()
    skill_panel["yoe_junior"] = skill_panel["yoe_extracted"].fillna(99).le(2)
    entry_total = int((asan_meta["seniority_final"] == "entry").sum())
    mid_total = int((asan_meta["seniority_final"] == "mid-senior").sum())
    junior_total = int(skill_panel["yoe_junior"].sum())  # wrong denominator, corrected below
    yoe_postings = asan_meta.assign(yoe_junior=asan_meta["yoe_extracted"].fillna(99).le(2))
    junior_total = int(yoe_postings["yoe_junior"].sum())
    senior_total = int((~yoe_postings["yoe_junior"]).sum())

    label_rows = []
    yoe_rows = []
    grouped = skill_panel.groupby("skill")
    for skill, sub in grouped:
        a = int((sub["seniority_final"] == "entry").sum())
        b = int((sub["seniority_final"] == "mid-senior").sum())
        c = entry_total - a
        d = mid_total - b
        if min(a, b, c, d) >= 5:
            _, p, _, _ = chi2_contingency([[a, c], [b, d]], correction=False)
            oratio = (a * d + 0.5) / ((b * c) + 0.5)
            label_rows.append(
                {
                    "skill": skill,
                    "entry_n": a,
                    "mid_n": b,
                    "total_n": int(len(sub)),
                    "p_value": float(p),
                    "odds_ratio_entry_vs_mid": float(oratio),
                }
            )

        a2 = int(sub["yoe_junior"].sum())
        b2 = int((~sub["yoe_junior"]).sum())
        c2 = junior_total - a2
        d2 = senior_total - b2
        if min(a2, b2, c2, d2) >= 5:
            _, p2, _, _ = chi2_contingency([[a2, c2], [b2, d2]], correction=False)
            oratio2 = (a2 * d2 + 0.5) / ((b2 * c2) + 0.5)
            yoe_rows.append(
                {
                    "skill": skill,
                    "junior_n": a2,
                    "senior_n": b2,
                    "total_n": int(len(sub)),
                    "p_value": float(p2),
                    "odds_ratio_junior_vs_senior": float(oratio2),
                }
            )

    label_df = pd.DataFrame(label_rows)
    if not label_df.empty:
        label_df["q_value"] = multipletests(label_df["p_value"], method="fdr_bh")[1]
        label_df["log2_or"] = np.log2(label_df["odds_ratio_entry_vs_mid"])
        save_csv(label_df.sort_values(["log2_or", "total_n"], ascending=[False, False]), TABLE_DIR / "T14_skill_entry_vs_mid_label_based.csv")
    yoe_df = pd.DataFrame(yoe_rows)
    if not yoe_df.empty:
        yoe_df["q_value"] = multipletests(yoe_df["p_value"], method="fdr_bh")[1]
        yoe_df["log2_or"] = np.log2(yoe_df["odds_ratio_junior_vs_senior"])
        save_csv(yoe_df.sort_values(["log2_or", "total_n"], ascending=[False, False]), TABLE_DIR / "T14_skill_yoe_junior_vs_senior.csv")
    summary = pd.DataFrame(
        [
            {
                "label_significant_q_lt_0.05": int((label_df["q_value"] < 0.05).sum()) if not label_df.empty else 0,
                "yoe_significant_q_lt_0.05": int((yoe_df["q_value"] < 0.05).sum()) if not yoe_df.empty else 0,
                "label_yoe_overlap": int(len(set(label_df.loc[label_df["q_value"] < 0.05, "skill"]) & set(yoe_df.loc[yoe_df["q_value"] < 0.05, "skill"]))) if not label_df.empty and not yoe_df.empty else 0,
            }
        ]
    )
    save_csv(summary, TABLE_DIR / "T14_skill_seniority_summary.csv")

    # AI integration pattern
    ai_cols = [
        "llm",
        "openai_api",
        "anthropic_api",
        "claude_api",
        "gemini_api",
        "prompt_engineering",
        "fine_tuning",
        "mcp",
        "agent",
        "copilot",
        "cursor",
        "chatgpt",
        "claude",
        "gemini",
        "codex",
        "langchain",
        "langgraph",
        "llamaindex",
        "rag",
        "vector_db",
        "pinecone",
        "weaviate",
        "chroma",
        "milvus",
        "faiss",
    ]
    ai_cols = [c for c in ai_cols if c in tech_cols]
    traditional_cols = [c for c in tech_cols if c not in ai_cols]
    df["ai_any"] = df[ai_cols].any(axis=1)
    df["tech_count"] = df[tech_cols].sum(axis=1)
    df["tech_density"] = df["tech_count"] / (df["description_length"].clip(lower=1) / 1000.0)
    ai_summary = (
        df.groupby(["source", "text_source", "ai_any"], dropna=False)
        .agg(
            n=("uid", "size"),
            mean_tech_count=("tech_count", "mean"),
            median_tech_count=("tech_count", "median"),
            mean_tech_density=("tech_density", "mean"),
            median_tech_density=("tech_density", "median"),
            mean_description_length=("description_length", "mean"),
        )
        .reset_index()
    )
    save_csv(ai_summary, TABLE_DIR / "T14_ai_integration_pattern.csv")
    # Co-occurring techs with AI postings
    ai_assoc = []
    for tech in tech_cols:
        if tech in ai_cols:
            continue
        a = int((df["ai_any"] & df[tech]).sum())
        b = int((~df["ai_any"] & df[tech]).sum())
        c = int((df["ai_any"] & ~df[tech]).sum())
        d = int((~df["ai_any"] & ~df[tech]).sum())
        if min(a, b, c, d) >= 25:
            _, p, _, _ = chi2_contingency([[a, c], [b, d]], correction=False)
            oratio = (a * d + 0.5) / ((b * c) + 0.5)
            ai_assoc.append(
                {
                    "tech": tech,
                    "category": tech_category(tech),
                    "ai_n": a,
                    "non_ai_n": b,
                    "odds_ratio_ai_vs_non_ai": float(oratio),
                    "p_value": float(p),
                }
            )
    ai_assoc_df = pd.DataFrame(ai_assoc)
    if not ai_assoc_df.empty:
        ai_assoc_df["q_value"] = multipletests(ai_assoc_df["p_value"], method="fdr_bh")[1]
        save_csv(ai_assoc_df.sort_values("odds_ratio_ai_vs_non_ai", ascending=False), TABLE_DIR / "T14_ai_cooccurrence_association.csv")

    # Core narrative table by source
    narrative = (
        change.sort_values("cross_abs", ascending=False)
        .head(25)[
            [
                "tech",
                "category",
                "arshkon",
                "asaniczka",
                "scraped",
                "within_2024_abs",
                "cross_abs",
                "snr",
            ]
        ]
        .copy()
    )
    save_csv(narrative, TABLE_DIR / "T14_top_tech_changes_top25.csv")

    # Key narrative text for the report
    summary_text = {
        "rows": int(len(df)),
        "swe_rows": int(len(df)),
        "text_llm_rows": int((df["text_source"] == "llm").sum()),
        "text_raw_rows": int((df["text_source"] == "raw").sum()),
        "ai_any_share_arshkon": float(df.loc[df["source"] == "kaggle_arshkon", "ai_any"].mean()),
        "ai_any_share_asaniczka": float(df.loc[df["source"] == "kaggle_asaniczka", "ai_any"].mean()),
        "ai_any_share_scraped": float(df.loc[df["source"] == "scraped", "ai_any"].mean()),
    }
    save_csv(pd.DataFrame([summary_text]), TABLE_DIR / "T14_summary_metrics.csv")


if __name__ == "__main__":
    main()
