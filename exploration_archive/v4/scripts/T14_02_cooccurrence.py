"""T14 Step 3: Technology co-occurrence network with community detection.

Computes phi coefficient between all technology pairs for 2024 and 2026 separately.
Thresholds at phi > 0.15, runs greedy modularity community detection, compares
communities across periods.

Outputs:
  tables/T14/cooccurrence_phi_2024.csv
  tables/T14/cooccurrence_phi_2026.csv
  tables/T14/community_membership.csv
  tables/T14/community_comparison.csv
  figures/T14/tech_network_2024.png
  figures/T14/tech_network_2026.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import duckdb
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.algorithms.community import greedy_modularity_communities

ROOT = Path("/home/jihgaboot/gabor/job-research")
SHARED = ROOT / "exploration/artifacts/shared"
TABLES = ROOT / "exploration/tables/T14"
FIGS = ROOT / "exploration/figures/T14"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

PHI_THRESH = 0.15
MIN_PREVALENCE = 0.01  # >1% in at least one period


def phi_matrix(X: np.ndarray) -> np.ndarray:
    """X: (n, k) binary matrix. Returns (k, k) phi coefficient matrix."""
    X = X.astype(np.float32)
    n = X.shape[0]
    col_sum = X.sum(axis=0)  # = n_1 for each col
    # Matrix product: (k,k) where entry = #rows with both = 1
    both = X.T @ X  # (k,k)
    p1 = col_sum / n  # marginal for each col
    # phi = (p11 - p1_a*p1_b) / sqrt(p1_a*(1-p1_a)*p1_b*(1-p1_b))
    p11 = both / n
    denom = np.sqrt(np.outer(p1 * (1 - p1), p1 * (1 - p1)))
    denom = np.where(denom == 0, 1, denom)
    phi = (p11 - np.outer(p1, p1)) / denom
    np.fill_diagonal(phi, 0.0)
    return phi


def main():
    # Load meta + tech
    con = duckdb.connect()
    meta = con.execute(f"""
        SELECT c.uid, c.source, c.period, c.seniority_3level, c.is_aggregator
        FROM read_parquet('{SHARED}/swe_cleaned_text.parquet') c
    """).df()
    meta["period2"] = meta["period"].map({"2024-01": "2024", "2024-04": "2024",
                                           "2026-03": "2026", "2026-04": "2026"})
    meta = meta[~meta["is_aggregator"].fillna(False)]

    tech = pq.read_table(SHARED / "swe_tech_matrix.parquet").to_pandas()
    tech_cols = [c for c in tech.columns if c != "uid"]
    df = meta.merge(tech, on="uid", how="inner")
    print(f"  merged: {len(df)}")

    # Compute prevalence per period; keep techs with >=1% in at least one period
    prev_2024 = df[df["period2"] == "2024"][tech_cols].mean()
    prev_2026 = df[df["period2"] == "2026"][tech_cols].mean()
    keep = prev_2024[(prev_2024 >= MIN_PREVALENCE) | (prev_2026 >= MIN_PREVALENCE)].index.tolist()
    # also keep if prev_2026 >= MIN
    keep2 = prev_2026[prev_2026 >= MIN_PREVALENCE].index.tolist()
    keep = sorted(set(keep) | set(keep2))
    print(f"  techs kept after prevalence filter: {len(keep)}")

    X_2024 = df[df["period2"] == "2024"][keep].to_numpy()
    X_2026 = df[df["period2"] == "2026"][keep].to_numpy()

    print("  computing phi for 2024...")
    phi_2024 = phi_matrix(X_2024)
    print("  computing phi for 2026...")
    phi_2026 = phi_matrix(X_2026)

    # Save phi matrices long-format (pairs above threshold)
    def long_pairs(phi, thresh=PHI_THRESH):
        rows = []
        for i in range(len(keep)):
            for j in range(i + 1, len(keep)):
                p = phi[i, j]
                if abs(p) >= thresh:
                    rows.append({"tech_a": keep[i], "tech_b": keep[j], "phi": p})
        return pd.DataFrame(rows).sort_values("phi", ascending=False)

    pairs_2024 = long_pairs(phi_2024)
    pairs_2026 = long_pairs(phi_2026)
    pairs_2024.to_csv(TABLES / "cooccurrence_phi_2024.csv", index=False)
    pairs_2026.to_csv(TABLES / "cooccurrence_phi_2026.csv", index=False)
    print(f"  pairs above phi>{PHI_THRESH}: 2024={len(pairs_2024)}  2026={len(pairs_2026)}")

    # Build networks
    def build_graph(pairs):
        G = nx.Graph()
        for _, r in pairs.iterrows():
            if r["phi"] > 0:  # positive associations only for communities
                G.add_edge(r["tech_a"], r["tech_b"], weight=r["phi"])
        return G

    G24 = build_graph(pairs_2024)
    G26 = build_graph(pairs_2026)
    print(f"  G24 nodes={G24.number_of_nodes()} edges={G24.number_of_edges()}")
    print(f"  G26 nodes={G26.number_of_nodes()} edges={G26.number_of_edges()}")

    # Communities via greedy modularity
    def communities(G):
        if G.number_of_nodes() == 0:
            return []
        return [sorted(c) for c in greedy_modularity_communities(G, weight="weight")]

    comms_2024 = communities(G24)
    comms_2026 = communities(G26)

    def summarize(comms):
        return [{"size": len(c), "members": "; ".join(c)} for c in comms]

    mem_rows = []
    for i, c in enumerate(comms_2024):
        for t in c:
            mem_rows.append({"period": "2024", "community": i, "tech": t, "size": len(c)})
    for i, c in enumerate(comms_2026):
        for t in c:
            mem_rows.append({"period": "2026", "community": i, "tech": t, "size": len(c)})
    mem_df = pd.DataFrame(mem_rows)
    mem_df.to_csv(TABLES / "community_membership.csv", index=False)

    # Compare: for each 2026 community, find best-matching 2024 community by Jaccard
    comp_rows = []
    for i, c26 in enumerate(comms_2026):
        s26 = set(c26)
        best_j = -1; best_jacc = 0; best_overlap = 0
        for j, c24 in enumerate(comms_2024):
            s24 = set(c24)
            if not (s26 | s24):
                continue
            jacc = len(s26 & s24) / len(s26 | s24)
            if jacc > best_jacc:
                best_jacc = jacc; best_j = j; best_overlap = len(s26 & s24)
        s24 = set(comms_2024[best_j]) if best_j >= 0 else set()
        comp_rows.append({
            "comm_2026_id": i,
            "comm_2026_size": len(c26),
            "best_match_2024_id": best_j,
            "best_match_2024_size": len(s24),
            "jaccard": best_jacc,
            "overlap_members": "; ".join(sorted(s26 & s24)),
            "new_members_2026": "; ".join(sorted(s26 - s24)),
            "lost_from_2024": "; ".join(sorted(s24 - s26)),
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(TABLES / "community_comparison.csv", index=False)
    print(f"  wrote community_comparison.csv  ({len(comp_df)} 2026 communities)")

    # Visualize
    def draw_network(G, title, comms, fpath):
        if G.number_of_nodes() == 0:
            return
        # Position with spring layout
        pos = nx.spring_layout(G, k=0.35, seed=42, iterations=60)
        plt.figure(figsize=(14, 11))
        # Color by community
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(comms), 1)))
        node_color = {}
        for i, c in enumerate(comms):
            for t in c:
                node_color[t] = colors[i % len(colors)]
        ncols = [node_color.get(n, (0.5, 0.5, 0.5, 1.0)) for n in G.nodes()]
        # Node size by degree
        deg = dict(G.degree(weight="weight"))
        sizes = [50 + 40 * deg.get(n, 0) for n in G.nodes()]
        # Edge width by weight
        weights = [G[u][v]["weight"] * 3 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, alpha=0.25, width=weights)
        nx.draw_networkx_nodes(G, pos, node_color=ncols, node_size=sizes, alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=7)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(fpath, dpi=150)
        plt.close()

    draw_network(G24, f"Technology co-occurrence network (2024) — phi>{PHI_THRESH}",
                 comms_2024, FIGS / "tech_network_2024.png")
    draw_network(G26, f"Technology co-occurrence network (2026) — phi>{PHI_THRESH}",
                 comms_2026, FIGS / "tech_network_2026.png")

    print(f"  wrote tech_network_2024.png and tech_network_2026.png")
    print("Done T14 step 02.")


if __name__ == "__main__":
    main()
