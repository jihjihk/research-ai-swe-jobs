"""T09 Step 7 (CRITICAL): Compute Normalized Mutual Information (NMI) between
cluster assignments and {seniority_final, period, tech-domain}.

The tech-domain label is derived from the cluster's top c-TF-IDF terms by
mapping to a coarse domain taxonomy. We derive it here, NOT from the cluster
assignment itself, to avoid tautology: the domain label is a mapping of the
cluster's top-10 terms onto a fixed taxonomy (one label per cluster), then
we see how much cluster vs seniority vs period explains the assignment.

Actually, the instruction wants domain as an INDEPENDENT categorical label.
We derive per-doc domain via a separate rule-based matcher on the raw
text, independent of BERTopic. Then we compute NMI(clusters, domain)
alongside NMI(clusters, seniority), NMI(clusters, period).

Writes:
  exploration/tables/T09/nmi_cluster_axes.csv
"""
import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

OUTDIR = "exploration/artifacts/T09"
TABLES = "exploration/tables/T09"

# Coarse domain labels derived from tech matrix / raw text matching.
# Priority order: a doc that matches multiple domains is assigned the first
# one matched. Order reflects domain specificity (most-specific first).
DOMAIN_PATTERNS = [
    ("genai_llm",
        r"\b(llm|llms|gpt|gpt-?\d|chatgpt|gemini|claude|copilot|cursor|"
        r"rag|retrieval[- ]augmented|langchain|langgraph|llamaindex|"
        r"generative ai|gen[- ]ai|foundation model|prompt engineering|"
        r"agent(ic)? framework|agentic ai|vector database|pinecone|"
        r"huggingface|hugging face|openai|ollama|anthropic|mcp server|"
        r"fine[- ]tuning)\b"),
    ("ml_research",
        r"\b(pytorch|tensorflow|keras|scikit[- ]learn|sklearn|xgboost|"
        r"deep learning|reinforcement learning|computer vision|nlp|"
        r"natural language processing|transformer|transformers|neural network|"
        r"ml engineer|machine learning engineer|mlops|model training|"
        r"feature engineering|recommendation system|ranking model)\b"),
    ("data_eng",
        r"\b(etl|elt|snowflake|databricks|spark|kafka|airflow|redshift|"
        r"bigquery|dbt|data warehouse|data lake|data pipeline|data engineer|"
        r"data engineering|dimensional model|data modeling)\b"),
    ("devops_platform",
        r"\b(devops|kubernetes|k8s|docker|terraform|ansible|jenkins|"
        r"gitlab ci|github actions|argo ?cd|helm|ci/cd|cicd|sre|"
        r"site reliability|platform engineer|platform engineering|"
        r"infrastructure as code|iac)\b"),
    ("cloud",
        r"\b(aws|amazon web services|azure|gcp|google cloud|cloud platform|"
        r"lambda|ec2|s3|cloudformation|ecs|eks|fargate|route ?53|"
        r"cloud architecture|cloud engineer)\b"),
    ("frontend",
        r"\b(react|angular|vue|svelte|nextjs|next\.js|redux|tailwind|"
        r"front[- ]end|frontend|ui engineer|web developer|javascript|typescript|"
        r"html5?|css3?|jquery)\b"),
    ("mobile",
        r"\b(ios|android|swift(ui)?|kotlin|react native|flutter|xamarin|"
        r"mobile application|mobile developer|mobile engineer)\b"),
    ("backend_general",
        r"\b(backend|back[- ]end|api developer|rest api|graphql|microservice|"
        r"microservices|node\.?js|django|flask|spring boot|fastapi|rails|"
        r"laravel|express\.?js|go(lang)?|rust|scala|elixir)\b"),
    ("dotnet_java",
        r"\b(\.net|dot ?net|c#|csharp|asp\.net|entity framework|"
        r"java developer|spring framework|hibernate|jpa|j2ee|ejb)\b"),
    ("embedded_systems",
        r"\b(firmware|embedded|rtos|fpga|vhdl|verilog|microcontroller|"
        r"iot device|robotics|hardware[- ]software|linux kernel|driver development|"
        r"low[- ]level|bare metal)\b"),
    ("security",
        r"\b(cybersecurity|information security|infosec|security clearance|"
        r"oauth|penetration test|pen test|vulnerability|siem|soc analyst|"
        r"security engineer|application security|appsec)\b"),
    ("test_qa",
        r"\b(qa engineer|quality assurance|test engineer|test automation|"
        r"selenium|cypress|playwright|manual testing|automation tester|"
        r"sdet|software development engineer in test)\b"),
    ("game_dev",
        r"\b(game developer|game engine|unity3d?|unreal engine|gameplay|"
        r"game programmer|video game)\b"),
    ("network_systems",
        r"\b(network engineer|cisco|juniper|tcp/ip|routing|switching|"
        r"load balancer|bgp|ospf|vpn|firewall|network administration|"
        r"active directory|vmware|virtualization|windows server)\b"),
]


def derive_domain(text: str) -> str:
    t = text.lower() if isinstance(text, str) else ""
    for name, pat in DOMAIN_PATTERNS:
        if re.search(pat, t):
            return name
    return "generic_swe"


def main():
    sample = pd.read_parquet(f"{OUTDIR}/sample_docs.parquet")
    bt = pd.read_parquet(f"{OUTDIR}/bertopic_assignments.parquet")
    nmf = pd.read_parquet(f"{OUTDIR}/nmf_assignments.parquet")
    df = sample.merge(bt[["uid", "topic", "topic_reduced"]], on="uid").merge(
        nmf[["uid", "nmf_topic"]], on="uid")

    # Derive per-doc domain (independent of cluster ID)
    df["domain"] = df["description_cleaned"].apply(derive_domain)

    # Compact period buckets
    df["period_bucket"] = df["period"].map({
        "2024-04": "2024-04",
        "2024-01": "2024-01",
        "2026-03": "2026",
        "2026-04": "2026",
    })
    # Seniority with a single 'unknown' level
    df["seniority_cat"] = df["seniority_final"].fillna("unknown")

    # Compute NMI for each cluster source × each axis
    results = []
    for cluster_name, cluster_col in [
        ("bertopic_raw", "topic"),
        ("bertopic_reduced", "topic_reduced"),
        ("nmf", "nmf_topic"),
    ]:
        for axis_name, axis_col in [
            ("seniority_final", "seniority_cat"),
            ("period_4bucket", "period"),
            ("period_2bucket", "period_bucket"),
            ("derived_domain", "domain"),
            ("source", "source"),
            ("is_aggregator", "is_aggregator"),
        ]:
            nmi = normalized_mutual_info_score(df[cluster_col].astype(str),
                                                df[axis_col].astype(str))
            results.append({
                "cluster_method": cluster_name,
                "axis": axis_name,
                "nmi": round(nmi, 4),
            })
    out = pd.DataFrame(results)
    out.to_csv(f"{TABLES}/nmi_cluster_axes.csv", index=False)
    pivoted = out.pivot(index="cluster_method", columns="axis", values="nmi")
    print(pivoted)
    pivoted.to_csv(f"{TABLES}/nmi_cluster_axes_pivot.csv")

    # Also compute NMI between the derived_domain and the three coarse axes
    # (for sanity — domain should also be orthogonal to seniority/period)
    axis_rows = []
    for a in ["seniority_cat", "period", "period_bucket", "source", "is_aggregator"]:
        for b in ["domain"]:
            v = normalized_mutual_info_score(df[a].astype(str), df[b].astype(str))
            axis_rows.append({"a": a, "b": b, "nmi": round(v, 4)})
    pd.DataFrame(axis_rows).to_csv(f"{TABLES}/nmi_sanity_axes.csv", index=False)

    # Save merged file for downstream steps
    df.to_parquet(f"{OUTDIR}/sample_with_assignments.parquet", index=False)

    # Domain distribution
    print("\nDomain distribution:")
    print(df["domain"].value_counts().to_string())
    print("\nDomain × period (cross-tab):")
    print(pd.crosstab(df["domain"], df["period_bucket"], normalize="columns").round(3).to_string())


if __name__ == "__main__":
    main()
