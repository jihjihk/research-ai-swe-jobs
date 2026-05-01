"""V1.3 Alt 2 — Tech network modularity under cap=20/company instead of 50.

Re-runs the T14 modularity calculation: build a co-occurrence graph from the
shared tech matrix with cap 20/company/period, phi threshold 0.15, and
compute Louvain modularity. Compare with Gate 2's reported 0.56 -> 0.66
(which was computed on cap-50).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

META = "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_cleaned_text.parquet"
TECH = "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_tech_matrix.parquet"
UNI = "/home/jihgaboot/gabor/job-research/data/unified.parquet"
OUT = "/home/jihgaboot/gabor/job-research/exploration/tables/V1"


def load(cap: int) -> pd.DataFrame:
    con = duckdb.connect()
    q = f"""
    SELECT m.uid, m.source, u.company_name_canonical, t.*
    FROM '{META}' m
    INNER JOIN '{TECH}' t USING (uid)
    INNER JOIN '{UNI}' u USING (uid)
    """
    df = con.execute(q).fetchdf()
    df["period_bucket"] = df["source"].apply(lambda s: "2026" if s == "scraped" else "2024")
    # Cap rows per (company, period)
    rng = np.random.default_rng(42)
    keep = []
    for (c, p), g in df.groupby(["company_name_canonical", "period_bucket"]):
        if len(g) <= cap:
            keep.append(g)
        else:
            keep.append(g.sample(n=cap, random_state=42))
    capped = pd.concat(keep, ignore_index=True)
    return capped


def phi(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    n11 = int((a & b).sum())
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    n00 = int((~a & ~b).sum())
    num = n11 * n00 - n10 * n01
    den = ((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)) ** 0.5
    if den == 0:
        return 0.0
    return num / den


def build_graph(period_df: pd.DataFrame, tech_cols: list[str], min_prev: float, phi_thresh: float) -> nx.Graph:
    # filter techs by min prevalence
    shares = period_df[tech_cols].mean()
    active = [c for c in tech_cols if shares[c] >= min_prev]
    G = nx.Graph()
    for c in active:
        G.add_node(c)
    mats = {c: period_df[c].to_numpy(dtype=bool) for c in active}
    for i, a in enumerate(active):
        for b in active[i + 1 :]:
            p = phi(mats[a], mats[b])
            if p >= phi_thresh:
                G.add_edge(a, b, weight=p)
    return G


def main(cap: int) -> None:
    print("=" * 72)
    print(f"Modularity recomputation under cap={cap} per company/period")
    print("=" * 72)
    capped = load(cap)

    # identify tech columns (boolean only)
    non_tech = {"uid", "source", "company_name_canonical", "period_bucket"}
    tech_cols = [
        c for c in capped.columns
        if c not in non_tech and capped[c].dtype == bool
    ]
    print(f"tech cols: {len(tech_cols)}")

    for p in ("2024", "2026"):
        sub = capped[capped["period_bucket"] == p]
        print(f"\n-- period {p}: rows {len(sub):,}")
        G = build_graph(sub, tech_cols, min_prev=0.01, phi_thresh=0.15)
        n_edges = G.number_of_edges()
        n_nodes = G.number_of_nodes()
        if n_nodes == 0:
            print("  empty graph")
            continue
        communities = list(greedy_modularity_communities(G, weight="weight"))
        # networkx modularity
        mod = nx.community.modularity(G, communities, weight="weight")
        print(f"  nodes {n_nodes}  edges {n_edges}  communities {len(communities)}  modularity {mod:.3f}")


if __name__ == "__main__":
    main(cap=20)
    print()
    main(cap=50)  # reproduce T14 as sanity check
