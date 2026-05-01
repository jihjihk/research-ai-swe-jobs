"""T35 — Technology ecosystem crystallization (H_K).

Tests whether tech co-occurrence networks crystallized 2024 -> 2026.
Extends T14 co-occurrence via Louvain community detection across the 135-tech taxonomy.

Outputs under exploration/tables/T35/ and exploration/figures/T35/:
- per_period_network_summary.csv        (n_nodes, n_edges, n_communities, modularity primary + 10-seed stats)
- louvain_stability_10seeds.csv         (seed, n_communities, modularity per period)
- communities_2024.csv / communities_2026.csv (community_id, tech, size, period)
- community_classification.csv          (2026 community -> stable/coalesced/new/fragmented + Jaccard + named)
- llm_vendor_cluster.csv                (T14 expected members -> 2026 community + in_2024_cluster flag)
- archetype_crosstab.csv                (2026 community x domain/archetype share)
- phi_2024_edges.csv / phi_2026_edges.csv (thresholded edge list phi > 0.15)
- network_sidebyside.png                (two-panel fig)
- run_meta.json
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities, modularity

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNIFIED = ROOT / "data" / "unified.parquet"
TECH_MAT = ROOT / "exploration/artifacts/shared/swe_tech_matrix.parquet"
ARCHETYPE_PQ = ROOT / "exploration/tables/T28/T28_corpus_with_archetype.parquet"

OUT_TABLES = ROOT / "exploration/tables/T35"
OUT_FIGS = ROOT / "exploration/figures/T35"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

PHI_THRESHOLD = 0.15
COMPANY_CAP = 50  # Wave 3.5 per-firm cap
PRIMARY_SEED = 42
STABILITY_SEEDS = [11, 23, 29, 37, 42, 53, 61, 71, 83, 97]

# ----- helpers -----

def phi_from_counts(n_a: np.ndarray, n_b: np.ndarray, n_ab: np.ndarray, n_total: int) -> np.ndarray:
    """Phi coefficient (Pearson correlation of binaries) vectorized over pairs.
    phi = (n*n_ab - n_a*n_b) / sqrt(n_a*n_b*(n-n_a)*(n-n_b))
    Returns nan where denominator is zero.
    """
    n = n_total
    num = n * n_ab - n_a * n_b
    den = np.sqrt(n_a * n_b * (n - n_a) * (n - n_b))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def compute_phi_matrix(tech_bool: np.ndarray, tech_names: list[str]) -> tuple[pd.DataFrame, int]:
    """Given N x T boolean matrix (already company-capped, period-filtered),
    return long-format DataFrame (a, b, phi, n_a, n_b, n_ab) for upper-triangle pairs.
    """
    N = tech_bool.shape[0]
    x = tech_bool.astype(np.float32)
    n_a_vec = x.sum(axis=0)  # per-tech totals
    # pair co-occurrences: X^T X is T x T
    cooc = x.T @ x
    idx_i, idx_j = np.triu_indices(len(tech_names), k=1)
    n_a = n_a_vec[idx_i]
    n_b = n_a_vec[idx_j]
    n_ab = cooc[idx_i, idx_j]
    phi = phi_from_counts(n_a, n_b, n_ab, N)
    df = pd.DataFrame({
        "a": [tech_names[i] for i in idx_i],
        "b": [tech_names[j] for j in idx_j],
        "n_a": n_a.astype(int),
        "n_b": n_b.astype(int),
        "n_ab": n_ab.astype(int),
        "phi": phi,
    })
    return df, N


def build_graph(edges_df: pd.DataFrame, tech_names: list[str], threshold: float) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(tech_names)
    passing = edges_df[(edges_df["phi"] > threshold) & (~edges_df["phi"].isna())]
    for _, row in passing.iterrows():
        g.add_edge(row["a"], row["b"], weight=float(row["phi"]))
    return g


def louvain_run(g: nx.Graph, seed: int) -> tuple[list[set], float]:
    comms = louvain_communities(g, weight="weight", seed=seed, resolution=1.0)
    if len(comms) == 0:
        return [], math.nan
    q = modularity(g, comms, weight="weight")
    return [set(c) for c in comms], q


def communities_to_df(comms: list[set], period: str) -> pd.DataFrame:
    rows = []
    # sort by size desc for stable community_id
    ordered = sorted(comms, key=lambda s: (-len(s), sorted(s)[0] if s else ""))
    for i, c in enumerate(ordered):
        for tech in sorted(c):
            rows.append({"period": period, "community_id": i, "community_size": len(c), "tech": tech})
    return pd.DataFrame(rows)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


# ----- main -----

def main() -> None:
    t_start = time.time()
    meta: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}

    # Step 1 — load filter frame and tech matrix
    print("[step 1] loading unified filter frame + tech matrix ...")
    con = duckdb.connect()
    base = con.execute(f"""
        select uid, source, period, company_name_canonical, is_aggregator
        from read_parquet('{UNIFIED}')
        where is_swe=true and source_platform='linkedin'
          and is_english=true and date_flag='ok'
    """).df()
    print(f"  base rows: {len(base):,}")

    tech = pd.read_parquet(TECH_MAT)
    tech_cols = [c for c in tech.columns if c != "uid"]
    assert len(tech_cols) == 135, f"unexpected tech col count: {len(tech_cols)}"
    merged = base.merge(tech, on="uid", how="inner")
    assert len(merged) == len(base), "uid mismatch between unified frame and tech matrix"
    # split
    merged["period_era"] = np.where(merged["period"].str.startswith("2024"), "2024", "2026")
    print("  period breakdown:")
    print(merged.groupby(["period_era", "source"]).size())

    # Step 1b — aggregator exclusion sensitivity
    # Primary run INCLUDES aggregators for coverage; we run an aggregator-excluded sensitivity below.
    tech_names = tech_cols

    # Step 1c — company cap 50 per canonical company within each period era
    def cap_by_company(df: pd.DataFrame, cap: int) -> pd.DataFrame:
        # null companies: keep all (they're not grouped)
        has_co = df["company_name_canonical"].notna()
        capped = (
            df[has_co]
            .groupby(["period_era", "company_name_canonical"], group_keys=False)
            .head(cap)
        )
        return pd.concat([capped, df[~has_co]], ignore_index=True)

    capped = cap_by_company(merged, COMPANY_CAP)
    print(f"  after company cap {COMPANY_CAP}: {len(capped):,} rows (from {len(merged):,})")

    capped_nonagg = cap_by_company(merged[~merged["is_aggregator"].fillna(False)], COMPANY_CAP)
    print(f"  after company cap 50 + non-agg: {len(capped_nonagg):,} rows")

    # Step 2 — per-period phi
    def compute_period(df: pd.DataFrame, period: str) -> tuple[pd.DataFrame, int]:
        sub = df[df["period_era"] == period]
        mat = sub[tech_names].to_numpy(dtype=bool)
        edges, n_rows = compute_phi_matrix(mat, tech_names)
        return edges, n_rows

    print("[step 2] computing phi per period (primary: cap50 with aggregators) ...")
    edges_2024, n_2024 = compute_period(capped, "2024")
    edges_2026, n_2026 = compute_period(capped, "2026")
    edges_2024.assign(period="2024").to_csv(OUT_TABLES / "phi_2024_edges_full.csv", index=False)
    edges_2026.assign(period="2026").to_csv(OUT_TABLES / "phi_2026_edges_full.csv", index=False)
    # threshold-passing only (smaller files)
    edges_2024[(edges_2024["phi"] > PHI_THRESHOLD)].to_csv(OUT_TABLES / "phi_2024_edges.csv", index=False)
    edges_2026[(edges_2026["phi"] > PHI_THRESHOLD)].to_csv(OUT_TABLES / "phi_2026_edges.csv", index=False)

    meta["n_2024"] = int(n_2024)
    meta["n_2026"] = int(n_2026)
    meta["phi_threshold"] = PHI_THRESHOLD
    meta["company_cap"] = COMPANY_CAP
    print(f"  2024 rows {n_2024:,}, 2026 rows {n_2026:,}")

    # Step 3 — build graphs + Louvain
    print("[step 3] Louvain community detection ...")
    g24 = build_graph(edges_2024, tech_names, PHI_THRESHOLD)
    g26 = build_graph(edges_2026, tech_names, PHI_THRESHOLD)

    # isolate count (on active-node subgraphs = nodes with at least one neighbor)
    def active_subgraph(g: nx.Graph) -> nx.Graph:
        return g.subgraph([n for n in g.nodes if g.degree[n] > 0]).copy()

    g24_active = active_subgraph(g24)
    g26_active = active_subgraph(g26)

    comms_24, q_24 = louvain_run(g24_active, PRIMARY_SEED)
    comms_26, q_26 = louvain_run(g26_active, PRIMARY_SEED)

    # Stability: 10 seeds on same graphs
    stability_rows = []
    stab_24_q, stab_24_k = [], []
    stab_26_q, stab_26_k = [], []
    for s in STABILITY_SEEDS:
        c24, q24 = louvain_run(g24_active, s)
        c26, q26 = louvain_run(g26_active, s)
        stab_24_q.append(q24); stab_24_k.append(len(c24))
        stab_26_q.append(q26); stab_26_k.append(len(c26))
        stability_rows.append({
            "seed": s, "period": "2024",
            "n_communities": len(c24), "modularity": q24,
        })
        stability_rows.append({
            "seed": s, "period": "2026",
            "n_communities": len(c26), "modularity": q26,
        })
    pd.DataFrame(stability_rows).to_csv(OUT_TABLES / "louvain_stability_10seeds.csv", index=False)

    network_summary = pd.DataFrame([
        {"period": "2024", "n_nodes_total": g24.number_of_nodes(),
         "n_nodes_active": g24_active.number_of_nodes(),
         "n_edges": g24_active.number_of_edges(),
         "n_isolates": g24.number_of_nodes() - g24_active.number_of_nodes(),
         "n_communities_primary": len(comms_24),
         "modularity_primary": q_24,
         "modularity_mean10": float(np.mean(stab_24_q)),
         "modularity_sd10": float(np.std(stab_24_q, ddof=1)),
         "modularity_min10": float(np.min(stab_24_q)),
         "modularity_max10": float(np.max(stab_24_q)),
         "n_comm_mean10": float(np.mean(stab_24_k)),
         "n_comm_sd10": float(np.std(stab_24_k, ddof=1)),
         "n_comm_min10": int(np.min(stab_24_k)),
         "n_comm_max10": int(np.max(stab_24_k)),
         "mean_comm_size": float(np.mean([len(c) for c in comms_24])) if comms_24 else math.nan,
         "max_comm_size": int(max((len(c) for c in comms_24), default=0)),
         },
        {"period": "2026", "n_nodes_total": g26.number_of_nodes(),
         "n_nodes_active": g26_active.number_of_nodes(),
         "n_edges": g26_active.number_of_edges(),
         "n_isolates": g26.number_of_nodes() - g26_active.number_of_nodes(),
         "n_communities_primary": len(comms_26),
         "modularity_primary": q_26,
         "modularity_mean10": float(np.mean(stab_26_q)),
         "modularity_sd10": float(np.std(stab_26_q, ddof=1)),
         "modularity_min10": float(np.min(stab_26_q)),
         "modularity_max10": float(np.max(stab_26_q)),
         "n_comm_mean10": float(np.mean(stab_26_k)),
         "n_comm_sd10": float(np.std(stab_26_k, ddof=1)),
         "n_comm_min10": int(np.min(stab_26_k)),
         "n_comm_max10": int(np.max(stab_26_k)),
         "mean_comm_size": float(np.mean([len(c) for c in comms_26])) if comms_26 else math.nan,
         "max_comm_size": int(max((len(c) for c in comms_26), default=0)),
         },
    ])
    network_summary.to_csv(OUT_TABLES / "per_period_network_summary.csv", index=False)
    print(network_summary.to_string())

    # Persist community assignments
    comm_24_df = communities_to_df(comms_24, "2024")
    comm_26_df = communities_to_df(comms_26, "2026")
    comm_24_df.to_csv(OUT_TABLES / "communities_2024.csv", index=False)
    comm_26_df.to_csv(OUT_TABLES / "communities_2026.csv", index=False)

    # Step 4 — backward-stability classification
    # For each 2026 community, find Jaccard with each 2024 community; classify.
    print("[step 4] community classification ...")

    # Set of nodes that had at least one edge above phi threshold in 2024
    active_2024 = set(g24_active.nodes)
    active_2026 = set(g26_active.nodes)

    # Normalize: comms_24 ordered list of sets, same for 26
    comm24_sorted = sorted(comms_24, key=lambda s: (-len(s), sorted(s)[0] if s else ""))
    comm26_sorted = sorted(comms_26, key=lambda s: (-len(s), sorted(s)[0] if s else ""))

    classification_rows = []
    for i26, c26 in enumerate(comm26_sorted):
        # Jaccard vs each 2024 community
        jacs = []
        for i24, c24 in enumerate(comm24_sorted):
            j = jaccard(c26, c24)
            jacs.append((i24, len(c24), j))
        jacs.sort(key=lambda t: -t[2])
        top_i24, top_size24, top_j = jacs[0] if jacs else (None, 0, 0.0)

        # Fraction of 2026 members that were NOT active above threshold in 2024
        n_new_arrivals = len(c26 - active_2024)
        share_new_arrivals = n_new_arrivals / len(c26) if len(c26) else 0.0

        # Contribution of each 2024 community to this 2026 community
        contribs = []
        for i24, c24 in enumerate(comm24_sorted):
            overlap = len(c26 & c24)
            share = overlap / len(c26) if len(c26) else 0.0
            if share > 0:
                contribs.append((i24, overlap, share))
        # sort by share desc
        contribs.sort(key=lambda t: -t[2])
        top_share = contribs[0][2] if contribs else 0.0
        n_contributing_le_30 = sum(1 for _, _, s in contribs if s < 0.30)

        # 2024 communities that share ANY overlap and each contribute <30%
        n_contributing_2plus_each_lt_30 = sum(1 for _, _, s in contribs if s < 0.30 and s > 0.0)

        # Detect fragmentation: did any single 2024 community contribute to 2+ 2026 communities with share_of_2024 >= 30%?
        # We'll fill fragmentation in a second pass.

        # Initial classification rules:
        if share_new_arrivals >= 0.50:
            cls = "new"
        elif top_j >= 0.70:
            cls = "stable"
        elif top_share < 0.30 and n_contributing_2plus_each_lt_30 >= 2:
            cls = "coalesced"
        else:
            cls = "partial"  # intermediate label; may be updated to 'fragmented' after pass 2

        classification_rows.append({
            "comm_2026_id": i26,
            "comm_2026_size": len(c26),
            "share_new_arrivals": round(share_new_arrivals, 4),
            "top_2024_comm_id": top_i24,
            "top_2024_comm_size": top_size24,
            "top_jaccard": round(top_j, 4),
            "top_contribution_share": round(top_share, 4),
            "n_2024_contribs_lt_30": n_contributing_2plus_each_lt_30,
            "classification": cls,
            "top_techs": ",".join(sorted(c26)[:15]) + ("..." if len(c26) > 15 else ""),
        })

    # Second pass: fragmentation — a 2024 community that maps (share_of_2024 >= 30%) to 2+ 2026 communities
    frag_24 = set()
    for i24, c24 in enumerate(comm24_sorted):
        if len(c24) == 0:
            continue
        receiving_2026 = 0
        for i26, c26 in enumerate(comm26_sorted):
            overlap = len(c26 & c24)
            share_of_2024 = overlap / len(c24)
            if share_of_2024 >= 0.30:
                receiving_2026 += 1
        if receiving_2026 >= 2:
            frag_24.add(i24)

    # Upgrade classification to 'fragmented' for 2026 communities inheriting from a 2024 fragmented community
    for row in classification_rows:
        if row["classification"] == "partial" and row["top_2024_comm_id"] in frag_24:
            row["classification"] = "fragmented"
        elif row["classification"] == "partial":
            # if top Jaccard between 0.3 and 0.7 and not new/coalesced, treat as stable-partial
            if row["top_jaccard"] >= 0.30:
                row["classification"] = "stable_partial"
            else:
                row["classification"] = "weakly_mapped"

    # Also record which 2024 communities were fragmented (so both sides have record)
    frag_rows = []
    for i24 in sorted(frag_24):
        c24 = comm24_sorted[i24]
        frag_rows.append({
            "comm_2024_id": i24,
            "comm_2024_size": len(c24),
            "fragmented_into_n_2026": sum(
                1 for c26 in comm26_sorted
                if len(c26 & c24) / max(1, len(c24)) >= 0.30
            ),
            "members": ",".join(sorted(c24)[:15]) + ("..." if len(c24) > 15 else ""),
        })
    pd.DataFrame(frag_rows).to_csv(OUT_TABLES / "fragmented_2024_communities.csv", index=False)

    # Save community classification
    class_df = pd.DataFrame(classification_rows)

    # Step 5 — LLM-vendor cluster verification
    print("[step 5] LLM-vendor cluster verification ...")
    expected_llm = [
        "claude_tool", "claude_api", "copilot", "cursor_tool", "chatgpt", "gpt_model",
        "codex", "tabnine", "langchain", "llamaindex", "rag", "openai_api",
        "anthropic", "gemini", "llm", "ai_agent", "vector_database", "pinecone",
        "weaviate", "chroma", "hugging_face", "fine_tuning", "prompt_engineering",
        "mcp",
    ]
    # find which 2026 community contains the MOST expected members
    best_2026_i, best_overlap = None, -1
    for i26, c26 in enumerate(comm26_sorted):
        overlap = len(c26 & set(expected_llm))
        if overlap > best_overlap:
            best_overlap = overlap
            best_2026_i = i26
    best_comm_26 = comm26_sorted[best_2026_i] if best_2026_i is not None else set()

    # Which expected techs were NOT in any 2024 community at phi>0.15 ?
    # A tech was "in a community at phi>0.15 in 2024" iff it is in active_2024 (i.e. has at least one edge).
    llm_rows = []
    for t in expected_llm:
        in_2024_active = t in active_2024
        in_2026_active = t in active_2026
        in_best_2026 = t in best_comm_26
        # which 2024 community did it belong to?
        c2024_id = None
        for i24, c24 in enumerate(comm24_sorted):
            if t in c24:
                c2024_id = i24
                break
        # which 2026 community ?
        c2026_id = None
        for i26, c26 in enumerate(comm26_sorted):
            if t in c26:
                c2026_id = i26
                break
        llm_rows.append({
            "tech": t,
            "in_2024_active": in_2024_active,
            "comm_2024_id": c2024_id,
            "in_2026_active": in_2026_active,
            "comm_2026_id": c2026_id,
            "in_2026_llm_vendor_cluster": in_best_2026,
        })
    llm_vendor_df = pd.DataFrame(llm_rows)
    llm_vendor_df.to_csv(OUT_TABLES / "llm_vendor_cluster.csv", index=False)

    # Classification of the LLM-vendor cluster itself:
    llm_class_row = class_df[class_df["comm_2026_id"] == best_2026_i].iloc[0].to_dict() if best_2026_i is not None else {}
    meta["llm_vendor_cluster"] = {
        "comm_2026_id": best_2026_i,
        "size": len(best_comm_26),
        "expected_members_present": best_overlap,
        "expected_members_missing": [t for t in expected_llm if t not in best_comm_26],
        "classification": llm_class_row.get("classification"),
        "top_jaccard_vs_2024": llm_class_row.get("top_jaccard"),
        "share_new_arrivals": llm_class_row.get("share_new_arrivals"),
    }

    # Step 6 — name communities by members (focus on coalesced / new / stable_partial that matter)
    # Use a simple naming heuristic: if community intersects well-known anchors, assign that name.
    ANCHORS = {
        "python_ml_stack": {"python", "numpy", "pandas", "scikit_learn", "jupyter", "xgboost", "keras", "mlflow"},
        "traditional_ml_dl": {"tensorflow", "pytorch", "scikit_learn", "keras"},
        "llm_vendor_stack": {"langchain", "llamaindex", "rag", "pinecone", "vector_database", "claude_tool",
                             "copilot", "cursor_tool", "hugging_face", "llm", "ai_agent", "prompt_engineering"},
        "observability_stack": {"datadog", "new_relic", "pagerduty", "grafana", "prometheus", "splunk", "sentry"},
        "devops_platform_stack": {"terraform", "kubernetes", "docker", "helm", "argocd", "ansible"},
        "ci_cd_stack": {"jenkins", "github_actions", "circleci", "gitlab_ci", "buildkite", "travis_ci"},
        "data_engineering_stack": {"snowflake", "bigquery", "databricks", "dbt", "airflow", "kafka", "spark", "flink", "elasticsearch"},
        "js_frontend_stack": {"javascript", "typescript", "react", "angular", "vue", "nextjs", "nuxt", "svelte", "ember", "jquery", "nodejs", "express"},
        "jvm_backend_stack": {"java", "spring", "scala", "kotlin"},
        "dotnet_stack": {"c_sharp", "dot_net"},
        "mobile_stack": {"swift", "objective_c", "kotlin"},
        "cpp_systems_stack": {"c_plus_plus", "c_lang", "rust"},
        "ruby_rails_stack": {"ruby", "rails"},
        "php_laravel_stack": {"php", "laravel"},
        "legacy_config_mgmt": {"puppet", "chef"},
        "test_automation_stack": {"jest", "pytest", "selenium", "cypress", "playwright", "junit", "mocha"},
        "cloud_big3": {"aws", "azure", "gcp"},
        "agile_practices_stack": {"agile", "scrum", "tdd", "bdd", "ddd", "microservices", "ci_cd", "event_driven", "serverless"},
    }

    def name_community(members: set, anchors: dict) -> str:
        # rank anchors by Jaccard of overlap with community members
        best = ("unnamed", 0.0)
        for name, anchor_set in anchors.items():
            if not anchor_set:
                continue
            j = len(members & anchor_set) / max(1, len(members | anchor_set))
            # also weight by absolute overlap
            if j > best[1] and len(members & anchor_set) >= 2:
                best = (name, j)
        return best[0]

    class_df["name"] = [
        name_community(comm26_sorted[i], ANCHORS) for i in class_df["comm_2026_id"]
    ]
    class_df.to_csv(OUT_TABLES / "community_classification.csv", index=False)

    # Also name 2024 communities for reporting
    name_24_rows = []
    for i24, c24 in enumerate(comm24_sorted):
        name_24_rows.append({
            "comm_2024_id": i24,
            "comm_2024_size": len(c24),
            "name": name_community(c24, ANCHORS),
            "top_techs": ",".join(sorted(c24)[:15]) + ("..." if len(c24) > 15 else ""),
        })
    pd.DataFrame(name_24_rows).to_csv(OUT_TABLES / "communities_2024_named.csv", index=False)

    # Step 7 — Modularity delta: already in network_summary.

    # Step 8 — Visualization (2-panel)
    print("[step 8] side-by-side graph visualization ...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Build node -> 2026 community color
        palette = plt.colormaps.get_cmap("tab20").resampled(max(1, len(comm26_sorted)))
        node_color = {}
        for i26, c26 in enumerate(comm26_sorted):
            color = palette(i26 % 20)
            for t in c26:
                node_color[t] = color
        # isolates or active-24-only nodes -> grey
        GREY = (0.75, 0.75, 0.75, 1.0)
        for n in set(g24.nodes) | set(g26.nodes):
            node_color.setdefault(n, GREY)

        # Use union graph to compute shared spring layout (on active union subgraph)
        union_active = nx.compose(g24_active, g26_active)
        np.random.seed(42)
        pos = nx.spring_layout(union_active, seed=42, weight="weight", k=0.25, iterations=200)

        fig, axes = plt.subplots(1, 2, figsize=(22, 11))
        for ax, g, period in zip(axes, [g24_active, g26_active], ["2024", "2026"]):
            nodes = list(g.nodes)
            cols = [node_color.get(n, GREY) for n in nodes]
            # edge widths scaled by weight
            widths = [0.25 + 2.5 * g[u][v]["weight"] for u, v in g.edges]
            nx.draw_networkx_edges(g, pos, ax=ax, width=widths, alpha=0.25, edge_color="#555555")
            nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=nodes, node_color=cols, node_size=110, linewidths=0.3, edgecolors="black")
            # label high-degree nodes only
            deg = dict(g.degree)
            top_nodes = sorted(deg, key=lambda n: -deg[n])[:40]
            labels = {n: n for n in top_nodes}
            nx.draw_networkx_labels(g, pos, labels, font_size=7, ax=ax)
            ax.set_title(f"{period} · nodes={g.number_of_nodes()} · edges={g.number_of_edges()}", fontsize=13)
            ax.set_axis_off()
        plt.suptitle("T35 — Tech co-occurrence networks (phi > 0.15), colored by 2026 Louvain community", fontsize=14)
        plt.tight_layout()
        plt.savefig(OUT_FIGS / "network_sidebyside.png", dpi=140, bbox_inches="tight")
        plt.close()
        print("  saved network_sidebyside.png")
    except Exception as e:
        print(f"  viz skipped: {e}")
        meta["viz_error"] = str(e)

    # Step 9 — Domain / archetype cross-tab on 2026
    print("[step 9] archetype cross-tab ...")
    arche = pd.read_parquet(ARCHETYPE_PQ)[["uid", "archetype_primary_name", "domain"]]
    # merge into capped 2026 rows
    cap26 = capped[capped["period_era"] == "2026"].merge(arche, on="uid", how="left")

    # Each 2026 community: share of postings (in 2026) that contain at least one of its tech members.
    # Then stratify by domain.
    comm_membership = {i: set(members) for i, members in enumerate(comm26_sorted)}
    # posting-level membership: posting belongs to community C if it has >=2 techs in C
    # (reduce single-keyword-match noise). Store as boolean per (posting, community).
    tech_matrix_26 = cap26[tech_names].to_numpy(dtype=bool)
    tech_idx = {t: j for j, t in enumerate(tech_names)}

    post_comm = np.zeros((cap26.shape[0], len(comm_membership)), dtype=np.int8)
    for cid, members in comm_membership.items():
        idx = [tech_idx[t] for t in members if t in tech_idx]
        if not idx:
            continue
        counts = tech_matrix_26[:, idx].sum(axis=1)
        post_comm[:, cid] = (counts >= 2).astype(np.int8)

    # Aggregate: share of postings in each domain that hit each community
    cross_rows = []
    for domain, sub in cap26.groupby("domain"):
        n = len(sub)
        if n == 0:
            continue
        sub_idx = sub.index
        mat = post_comm[cap26.index.get_indexer(sub_idx)]
        # post_comm is aligned to cap26 reset_index? we need to be careful. Use positional indexing.
        pass
    # Redo correctly: reset index on cap26 to align post_comm with positional row index.
    cap26_r = cap26.reset_index(drop=True)
    # Rebuild post_comm on cap26_r
    tech_matrix_26r = cap26_r[tech_names].to_numpy(dtype=bool)
    post_comm_r = np.zeros((cap26_r.shape[0], len(comm_membership)), dtype=np.int8)
    for cid, members in comm_membership.items():
        idx = [tech_idx[t] for t in members if t in tech_idx]
        if not idx:
            continue
        counts = tech_matrix_26r[:, idx].sum(axis=1)
        post_comm_r[:, cid] = (counts >= 2).astype(np.int8)

    cross_rows = []
    domains = sorted(cap26_r["domain"].fillna("unknown").unique())
    archetypes = sorted(cap26_r["archetype_primary_name"].fillna("unassigned").unique())

    for cid in comm_membership:
        overall_share = post_comm_r[:, cid].mean()
        row = {
            "comm_2026_id": cid,
            "comm_name": class_df[class_df["comm_2026_id"] == cid]["name"].iloc[0],
            "comm_classification": class_df[class_df["comm_2026_id"] == cid]["classification"].iloc[0],
            "comm_size": len(comm_membership[cid]),
            "overall_share_postings": round(float(overall_share), 4),
        }
        for d in domains:
            mask = (cap26_r["domain"].fillna("unknown") == d).to_numpy()
            n = mask.sum()
            if n == 0:
                row[f"share__{d}"] = np.nan
            else:
                row[f"share__{d}"] = round(float(post_comm_r[mask, cid].mean()), 4)
        cross_rows.append(row)
    domain_cross = pd.DataFrame(cross_rows)
    domain_cross.to_csv(OUT_TABLES / "archetype_domain_crosstab.csv", index=False)

    # Archetype-level crosstab
    arch_rows = []
    for cid in comm_membership:
        row = {
            "comm_2026_id": cid,
            "comm_name": class_df[class_df["comm_2026_id"] == cid]["name"].iloc[0],
            "comm_classification": class_df[class_df["comm_2026_id"] == cid]["classification"].iloc[0],
        }
        for a in archetypes:
            mask = (cap26_r["archetype_primary_name"].fillna("unassigned") == a).to_numpy()
            n = mask.sum()
            if n == 0:
                row[f"share__{a}"] = np.nan
            else:
                row[f"share__{a}"] = round(float(post_comm_r[mask, cid].mean()), 4)
        arch_rows.append(row)
    pd.DataFrame(arch_rows).to_csv(OUT_TABLES / "archetype_crosstab.csv", index=False)

    # Step 2b/10 — Aggregator-exclusion sensitivity (on network structure, not on domain cross-tab)
    print("[sensitivity] aggregator-excluded rerun ...")

    def rerun_nonagg():
        edges24_ne, n24_ne = compute_period(capped_nonagg, "2024")
        edges26_ne, n26_ne = compute_period(capped_nonagg, "2026")
        g24_ne = build_graph(edges24_ne, tech_names, PHI_THRESHOLD)
        g26_ne = build_graph(edges26_ne, tech_names, PHI_THRESHOLD)
        g24_ne_a = active_subgraph(g24_ne)
        g26_ne_a = active_subgraph(g26_ne)
        c24_ne, q24_ne = louvain_run(g24_ne_a, PRIMARY_SEED)
        c26_ne, q26_ne = louvain_run(g26_ne_a, PRIMARY_SEED)
        return {
            "n_2024": int(n24_ne), "n_2026": int(n26_ne),
            "n_comm_2024": len(c24_ne), "n_comm_2026": len(c26_ne),
            "modularity_2024": q24_ne, "modularity_2026": q26_ne,
            "n_edges_2024": g24_ne_a.number_of_edges(),
            "n_edges_2026": g26_ne_a.number_of_edges(),
        }

    sens_nonagg = rerun_nonagg()
    pd.DataFrame([sens_nonagg]).to_csv(OUT_TABLES / "sensitivity_nonagg.csv", index=False)

    # Sensitivity at cap 20
    def rerun_cap20():
        cap20 = cap_by_company(merged, 20)
        edges24, n24 = compute_period(cap20, "2024")
        edges26, n26 = compute_period(cap20, "2026")
        g24c = build_graph(edges24, tech_names, PHI_THRESHOLD)
        g26c = build_graph(edges26, tech_names, PHI_THRESHOLD)
        g24c_a = active_subgraph(g24c)
        g26c_a = active_subgraph(g26c)
        c24c, q24c = louvain_run(g24c_a, PRIMARY_SEED)
        c26c, q26c = louvain_run(g26c_a, PRIMARY_SEED)
        return {
            "n_2024": int(n24), "n_2026": int(n26),
            "n_comm_2024": len(c24c), "n_comm_2026": len(c26c),
            "modularity_2024": q24c, "modularity_2026": q26c,
            "n_edges_2024": g24c_a.number_of_edges(),
            "n_edges_2026": g26c_a.number_of_edges(),
        }

    sens_cap20 = rerun_cap20()
    pd.DataFrame([sens_cap20]).to_csv(OUT_TABLES / "sensitivity_cap20.csv", index=False)

    meta["sensitivity_nonagg"] = sens_nonagg
    meta["sensitivity_cap20"] = sens_cap20
    meta["modularity_delta_primary"] = q_26 - q_24
    meta["n_communities_primary"] = {"2024": len(comms_24), "2026": len(comms_26)}
    meta["duration_sec"] = round(time.time() - t_start, 2)
    with open(OUT_TABLES / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[done] T35 in {meta['duration_sec']}s")
    print(f"  modularity Δ = {meta['modularity_delta_primary']:+.4f}")
    print(f"  n_communities 2024={len(comms_24)} 2026={len(comms_26)}")
    print(f"  LLM-vendor cluster → comm {meta['llm_vendor_cluster']['comm_2026_id']} "
          f"size={meta['llm_vendor_cluster']['size']} classification={meta['llm_vendor_cluster']['classification']}")


if __name__ == "__main__":
    main()
