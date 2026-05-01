"""T14 — Technology ecosystem mapping (Agent I, Wave 2).

Loads shared artifacts (tech matrix, cleaned text, asaniczka structured skills) and
produces:
  - Mention rates per tech × period × T30 seniority slice (J2/J3/S1/S4/mid-senior/all)
  - Technology co-occurrence networks (phi coefficient, Louvain community)
  - Rising/stable/declining classification (arshkon-only + pooled-2024 baselines)
  - Stack diversity (distinct tech count per posting)
  - AI-integration co-occurrence pattern (AI-mentioning vs non-AI, raw count + density)
  - Structured-vs-extracted validation (asaniczka skills_raw rank correlation)
  - Seniority-level skill differences (chi-squared with BH-FDR) on asaniczka

Outputs go to exploration/tables/T14/ and exploration/figures/T14/.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = Path("/home/jihgaboot/gabor/job-research")
SHARED = BASE / "exploration" / "artifacts" / "shared"
FIG = BASE / "exploration" / "figures" / "T14"
TAB = BASE / "exploration" / "tables" / "T14"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# ---------- Load ----------
print("Loading shared artifacts...")
tech = pd.read_parquet(SHARED / "swe_tech_matrix.parquet")
meta = pd.read_parquet(
    SHARED / "swe_cleaned_text.parquet",
    columns=[
        "uid",
        "source",
        "period",
        "seniority_final",
        "seniority_3level",
        "text_source",
        "is_aggregator",
        "yoe_extracted",
        "company_name_canonical",
    ],
)
meta["clen"] = pd.read_parquet(
    SHARED / "swe_cleaned_text.parquet", columns=["description_cleaned"]
)["description_cleaned"].str.len()

# Join
df = tech.merge(meta, on="uid", how="inner")
assert len(df) == 63701, f"row count mismatch: {len(df)}"
TECHS = [c for c in tech.columns if c != "uid"]
assert len(TECHS) == 107

# Derive panel flags
df["J2"] = df.seniority_final.isin(["entry", "associate"])
df["J3"] = df.yoe_extracted <= 2
df["S1"] = df.seniority_final.isin(["mid-senior", "director"])
df["S4"] = df.yoe_extracted >= 5
df["MIDSENIOR_LABEL"] = df.seniority_final == "mid-senior"
df["period_bucket"] = df.period.map(
    {"2024-04": "2024", "2024-01": "2024", "2026-03": "2026", "2026-04": "2026"}
)

# Source buckets for calibration
df["arshkon"] = df.source == "kaggle_arshkon"
df["asaniczka"] = df.source == "kaggle_asaniczka"
df["scraped"] = df.source == "scraped"
df["pooled_2024"] = df.arshkon | df.asaniczka

print(f"Loaded {len(df):,} rows × {len(TECHS)} techs")
print(f"Source counts: arshkon={df.arshkon.sum():,} asaniczka={df.asaniczka.sum():,} scraped={df.scraped.sum():,}")

# AI flag (OR across AI/LLM techs)
AI_TECHS = [
    "copilot", "cursor", "claude", "chatgpt", "openai", "gemini", "langchain",
    "langgraph", "rag", "pytorch", "tensorflow", "huggingface", "prompt_engineering",
    "fine_tuning", "mcp", "llamaindex", "agent_framework", "anthropic", "ollama",
    "vector_database", "pinecone", "chromadb", "codex", "llm_token",
]
for t in AI_TECHS:
    assert t in TECHS, f"AI tech {t} not in matrix"
df["ai_mentioning"] = df[AI_TECHS].any(axis=1)
df["tech_count"] = df[TECHS].sum(axis=1)
print(f"ai_mentioning rate overall: {df.ai_mentioning.mean():.3f}")
print(f"tech_count median: {df.tech_count.median():.1f}")

# ---------- Step 2: Mention rates by period × seniority ----------
print("\n[Step 2] Mention rates by period × T30 slice...")

slices = {
    "ALL": np.ones(len(df), dtype=bool),
    "J2": df["J2"].values,
    "J3": df["J3"].values,
    "MID_SENIOR": df["MIDSENIOR_LABEL"].values,
    "S1": df["S1"].values,
    "S4": df["S4"].values,
}

rate_rows = []
for slice_name, slice_mask in slices.items():
    for src_name, src_mask in [
        ("arshkon", df.arshkon.values),
        ("asaniczka", df.asaniczka.values),
        ("pooled_2024", df.pooled_2024.values),
        ("scraped", df.scraped.values),
    ]:
        mask = slice_mask & src_mask
        n = mask.sum()
        if n < 10:
            continue
        for t in TECHS:
            rate = df.loc[mask, t].mean()
            rate_rows.append(
                {
                    "slice": slice_name,
                    "source_bucket": src_name,
                    "tech": t,
                    "n": int(n),
                    "rate": float(rate),
                }
            )
rates = pd.DataFrame(rate_rows)
rates.to_csv(TAB / "tech_rates_by_slice.csv", index=False)
print(f"  {len(rates):,} (slice × source × tech) rate rows")

# Trajectory classification — arshkon-only
# Also pooled_2024 → scraped
arshkon_all = rates[(rates.slice == "ALL") & (rates.source_bucket == "arshkon")].set_index("tech").rate
asaniczka_all = rates[(rates.slice == "ALL") & (rates.source_bucket == "asaniczka")].set_index("tech").rate
pooled_all = rates[(rates.slice == "ALL") & (rates.source_bucket == "pooled_2024")].set_index("tech").rate
scraped_all = rates[(rates.slice == "ALL") & (rates.source_bucket == "scraped")].set_index("tech").rate

# Two-proportion z-test for each tech (arshkon vs scraped, pooled vs scraped)
N_arsh = int(df.arshkon.sum())
N_asan = int(df.asaniczka.sum())
N_pool = int(df.pooled_2024.sum())
N_scrap = int(df.scraped.sum())


def two_prop_z(p1, n1, p2, n2):
    x1, x2 = p1 * n1, p2 * n2
    p_pool = (x1 + x2) / (n1 + n2) if (n1 + n2) > 0 else 0
    denom = math.sqrt(max(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2), 1e-12))
    return (p1 - p2) / denom, denom


traj_rows = []
for t in TECHS:
    p_arsh = arshkon_all.get(t, 0)
    p_asan = asaniczka_all.get(t, 0)
    p_pool = pooled_all.get(t, 0)
    p_scrap = scraped_all.get(t, 0)
    delta_arsh = p_scrap - p_arsh
    delta_pool = p_scrap - p_pool
    # within-2024: arshkon - asaniczka
    delta_within = p_arsh - p_asan
    # SNR arshkon, relative to within-2024 noise
    noise = abs(delta_within) if abs(delta_within) > 1e-6 else 1e-3
    snr_arsh = abs(delta_arsh) / noise
    snr_pool = abs(delta_pool) / noise
    # z-tests (arshkon vs scraped)
    z_arsh, _ = two_prop_z(p_scrap, N_scrap, p_arsh, N_arsh)
    z_pool, _ = two_prop_z(p_scrap, N_scrap, p_pool, N_pool)
    # label rising/declining/stable (arshkon-only primary)
    if max(p_arsh, p_scrap) < 0.005:
        label = "rare"
    elif abs(delta_arsh) < 0.005:
        label = "stable"
    elif delta_arsh > 0:
        label = "rising"
    else:
        label = "declining"
    traj_rows.append(
        {
            "tech": t,
            "arshkon_rate": p_arsh,
            "asaniczka_rate": p_asan,
            "pooled_2024_rate": p_pool,
            "scraped_rate": p_scrap,
            "delta_arshkon": delta_arsh,
            "delta_pooled": delta_pool,
            "delta_within_2024": delta_within,
            "snr_arshkon": snr_arsh,
            "snr_pooled": snr_pool,
            "z_arshkon": z_arsh,
            "z_pooled": z_pool,
            "trajectory_arshkon": label,
        }
    )
traj = pd.DataFrame(traj_rows).sort_values("delta_arshkon", ascending=False)
traj.to_csv(TAB / "tech_trajectory.csv", index=False)
print("  top 10 rising (arshkon→scraped):")
print(traj.head(10)[["tech", "arshkon_rate", "scraped_rate", "delta_arshkon", "snr_arshkon"]].to_string(index=False))
print("  top 10 declining (arshkon→scraped):")
print(traj.tail(10)[["tech", "arshkon_rate", "scraped_rate", "delta_arshkon", "snr_arshkon"]].to_string(index=False))

# ---------- Step 3: Co-occurrence network ----------
print("\n[Step 3] Co-occurrence network (phi coefficient)...")


def compute_phi_matrix(X: np.ndarray) -> np.ndarray:
    """Phi coefficient matrix for binary columns of X (n × k)."""
    X = X.astype(np.float32)
    n = X.shape[0]
    p = X.mean(axis=0)
    denom = np.sqrt(p * (1 - p))
    # guard
    denom[denom == 0] = 1e-9
    # phi_ij = (p_ij - p_i p_j) / (sqrt(p_i(1-p_i)) sqrt(p_j(1-p_j)))
    cov = (X.T @ X) / n - np.outer(p, p)
    phi = cov / np.outer(denom, denom)
    np.fill_diagonal(phi, 0)
    return phi


# Select techs with freq >1% in at least one source-period
freq_mask = (
    (arshkon_all > 0.01)
    | (asaniczka_all > 0.01)
    | (scraped_all > 0.01)
)
NET_TECHS = [t for t in TECHS if freq_mask.reindex([t]).iloc[0]]
print(f"  techs with >1% rate somewhere: {len(NET_TECHS)}")

# Phi on 2024 (arshkon ∪ asaniczka) vs 2026 scraped
X_2024 = df.loc[df.pooled_2024, NET_TECHS].values
X_2026 = df.loc[df.scraped, NET_TECHS].values
phi_2024 = compute_phi_matrix(X_2024)
phi_2026 = compute_phi_matrix(X_2026)

PHI_THRESHOLD = 0.15
# Save edges
edge_rows = []
for i, t1 in enumerate(NET_TECHS):
    for j, t2 in enumerate(NET_TECHS):
        if j <= i:
            continue
        if abs(phi_2024[i, j]) >= PHI_THRESHOLD or abs(phi_2026[i, j]) >= PHI_THRESHOLD:
            edge_rows.append(
                {
                    "tech1": t1,
                    "tech2": t2,
                    "phi_2024": float(phi_2024[i, j]),
                    "phi_2026": float(phi_2026[i, j]),
                    "phi_delta": float(phi_2026[i, j] - phi_2024[i, j]),
                }
            )
edges = pd.DataFrame(edge_rows).sort_values("phi_2026", ascending=False)
edges.to_csv(TAB / "cooccurrence_edges.csv", index=False)
print(f"  edges with |phi|≥{PHI_THRESHOLD} (either period): {len(edges)}")

# Community detection (Louvain via networkx.community)
import networkx as nx

def build_graph(techs, phi, thresh):
    G = nx.Graph()
    for t in techs:
        G.add_node(t)
    for i, a in enumerate(techs):
        for j in range(i + 1, len(techs)):
            b = techs[j]
            w = phi[i, j]
            if w >= thresh:
                G.add_edge(a, b, weight=float(w))
    return G


G_2024 = build_graph(NET_TECHS, phi_2024, PHI_THRESHOLD)
G_2026 = build_graph(NET_TECHS, phi_2026, PHI_THRESHOLD)


def louvain_communities(G, seed=42):
    try:
        from networkx.algorithms.community import louvain_communities

        return louvain_communities(G, seed=seed, weight="weight")
    except Exception:
        from networkx.algorithms.community import greedy_modularity_communities

        return list(greedy_modularity_communities(G, weight="weight"))


comm_2024 = louvain_communities(G_2024)
comm_2026 = louvain_communities(G_2026)

print(f"  2024 communities: {len(comm_2024)}  sizes: {sorted([len(c) for c in comm_2024], reverse=True)}")
print(f"  2026 communities: {len(comm_2026)}  sizes: {sorted([len(c) for c in comm_2026], reverse=True)}")

# Map tech→community
def comm_map(comms):
    d = {}
    for i, c in enumerate(comms):
        for t in c:
            d[t] = i
    return d


cm24 = comm_map(comm_2024)
cm26 = comm_map(comm_2026)

comm_rows = []
for t in NET_TECHS:
    comm_rows.append({"tech": t, "community_2024": cm24.get(t, -1), "community_2026": cm26.get(t, -1)})
pd.DataFrame(comm_rows).to_csv(TAB / "communities_by_tech.csv", index=False)

# Summarize community membership
comm_summary = []
# naming communities by their top 3 tech by 2026 rate
for cid, tech_set in enumerate(comm_2026):
    members = list(tech_set)
    rates_c = {t: scraped_all.get(t, 0) for t in members}
    top = sorted(members, key=lambda t: -rates_c[t])[:5]
    comm_summary.append(
        {
            "period": "2026",
            "community_id": cid,
            "size": len(members),
            "members_top5_by_rate": ", ".join(top),
            "members_all": ", ".join(sorted(members)),
        }
    )
for cid, tech_set in enumerate(comm_2024):
    members = list(tech_set)
    rates_c = {t: pooled_all.get(t, 0) for t in members}
    top = sorted(members, key=lambda t: -rates_c[t])[:5]
    comm_summary.append(
        {
            "period": "2024",
            "community_id": cid,
            "size": len(members),
            "members_top5_by_rate": ", ".join(top),
            "members_all": ", ".join(sorted(members)),
        }
    )
pd.DataFrame(comm_summary).to_csv(TAB / "communities_summary.csv", index=False)

# ---------- Step 5: Stack diversity ----------
print("\n[Step 5] Stack diversity...")
stack_rows = []
for slice_name, slice_mask in slices.items():
    for src_name, src_mask in [
        ("arshkon", df.arshkon.values),
        ("asaniczka", df.asaniczka.values),
        ("pooled_2024", df.pooled_2024.values),
        ("scraped", df.scraped.values),
    ]:
        mask = slice_mask & src_mask
        if mask.sum() < 10:
            continue
        tc = df.loc[mask, "tech_count"]
        cl = df.loc[mask, "clen"]
        density = (tc / cl.clip(lower=100)) * 1000
        stack_rows.append(
            {
                "slice": slice_name,
                "source_bucket": src_name,
                "n": int(mask.sum()),
                "tech_count_mean": float(tc.mean()),
                "tech_count_median": float(tc.median()),
                "tech_count_density_per_1k_mean": float(density.mean()),
                "tech_count_density_per_1k_median": float(density.median()),
                "char_len_mean": float(cl.mean()),
                "char_len_median": float(cl.median()),
            }
        )
stack = pd.DataFrame(stack_rows)
stack.to_csv(TAB / "stack_diversity.csv", index=False)
print("  overall diversity:")
print(stack[stack.slice == "ALL"].to_string(index=False))

# ---------- Step 6: AI integration pattern ----------
print("\n[Step 6] AI integration co-occurrence (+ density normalization)...")

# restrict to scraped (where AI signal is strong)
ai_rows = []
for src_name, src_mask in [
    ("arshkon", df.arshkon.values),
    ("asaniczka", df.asaniczka.values),
    ("pooled_2024", df.pooled_2024.values),
    ("scraped", df.scraped.values),
]:
    for ai_val in [True, False]:
        mask = src_mask & (df.ai_mentioning.values == ai_val)
        if mask.sum() < 10:
            continue
        tc = df.loc[mask, "tech_count"]
        cl = df.loc[mask, "clen"]
        density = (tc / cl.clip(lower=100)) * 1000
        ai_rows.append(
            {
                "source_bucket": src_name,
                "ai_mentioning": ai_val,
                "n": int(mask.sum()),
                "tech_count_mean": float(tc.mean()),
                "tech_count_median": float(tc.median()),
                "density_per_1k_mean": float(density.mean()),
                "density_per_1k_median": float(density.median()),
                "char_len_mean": float(cl.mean()),
                "char_len_median": float(cl.median()),
            }
        )
ai_summary = pd.DataFrame(ai_rows)
ai_summary.to_csv(TAB / "ai_vs_non_ai_density.csv", index=False)
print("  raw tech_count: AI vs non-AI (scraped):")
print(ai_summary[ai_summary.source_bucket == "scraped"].to_string(index=False))

# Tech co-occurrence with AI among AI-mentioning postings (scraped)
ai_mask = df.scraped & df.ai_mentioning
nonai_mask = df.scraped & ~df.ai_mentioning
ai_n = int(ai_mask.sum())
nonai_n = int(nonai_mask.sum())
ai_co_rows = []
for t in TECHS:
    if t in AI_TECHS:
        continue
    rate_ai = df.loc[ai_mask, t].mean() if ai_n > 0 else np.nan
    rate_non = df.loc[nonai_mask, t].mean() if nonai_n > 0 else np.nan
    lift = rate_ai / rate_non if rate_non > 1e-6 else np.nan
    ai_co_rows.append(
        {
            "tech": t,
            "rate_ai_mentioning": rate_ai,
            "rate_non_ai_mentioning": rate_non,
            "lift_ai_over_nonai": lift,
            "delta": rate_ai - rate_non,
        }
    )
ai_co = pd.DataFrame(ai_co_rows).sort_values("lift_ai_over_nonai", ascending=False)
ai_co.to_csv(TAB / "ai_cooccurrence_non_ai_techs.csv", index=False)
print("  Top 10 non-AI techs with highest lift in AI-mentioning postings (scraped):")
print(ai_co.head(10)[["tech", "rate_ai_mentioning", "rate_non_ai_mentioning", "lift_ai_over_nonai"]].to_string(index=False))

# ---------- Step 7+8: Structured skills baseline & validation ----------
print("\n[Step 7+8] Asaniczka structured skills and validation...")
sk = pd.read_parquet(SHARED / "asaniczka_structured_skills.parquet")
sf = pd.read_csv(SHARED / "asaniczka_skill_frequency.csv")
top100 = sf.sort_values("n_postings", ascending=False).head(100)
top100.to_csv(TAB / "asaniczka_top100_skills.csv", index=False)

# Build synonym mapping between TECHS and asaniczka skill names
# We construct candidate string matches: direct lowercased match or snake->display
synonyms = {
    "python": ["python"],
    "java": ["java", "java 8", "java 11"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "go": ["go", "golang"],
    "rust": ["rust"],
    "cpp": ["c++", "cpp"],
    "csharp": ["c#", "csharp", ".net c#"],
    "ruby": ["ruby", "ruby on rails"],
    "kotlin": ["kotlin"],
    "swift": ["swift"],
    "scala": ["scala"],
    "php": ["php"],
    "r_language": ["r", "r programming", "r language"],
    "perl": ["perl"],
    "bash": ["bash"],
    "shell": ["shell", "shell scripting"],
    "sql": ["sql", "t-sql", "pl/sql"],
    "react": ["react", "react.js", "reactjs", "react js"],
    "angular": ["angular", "angularjs", "angular.js"],
    "vue": ["vue", "vue.js", "vuejs"],
    "nextjs": ["next.js", "nextjs", "next js"],
    "svelte": ["svelte"],
    "jquery": ["jquery"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "redux": ["redux"],
    "tailwind": ["tailwind", "tailwind css", "tailwindcss"],
    "nodejs": ["node.js", "nodejs", "node js"],
    "django": ["django"],
    "flask": ["flask"],
    "spring": ["spring", "spring boot", "spring framework"],
    "dotnet": [".net", ".net core", "dot net", "dotnet"],
    "rails": ["rails", "ruby on rails"],
    "fastapi": ["fastapi"],
    "express": ["express", "express.js", "expressjs"],
    "laravel": ["laravel"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud platform", "google cloud"],
    "kubernetes": ["kubernetes", "k8s"],
    "docker": ["docker"],
    "terraform": ["terraform"],
    "ansible": ["ansible"],
    "cicd": ["ci/cd", "cicd", "continuous integration"],
    "jenkins": ["jenkins"],
    "github_actions": ["github actions"],
    "argocd": ["argocd", "argo cd"],
    "gitlab": ["gitlab"],
    "helm": ["helm"],
    "postgresql": ["postgresql", "postgres"],
    "mysql": ["mysql"],
    "mongodb": ["mongodb"],
    "redis": ["redis"],
    "kafka": ["kafka", "apache kafka"],
    "spark": ["spark", "apache spark"],
    "snowflake": ["snowflake"],
    "databricks": ["databricks"],
    "dbt": ["dbt"],
    "elasticsearch": ["elasticsearch"],
    "oracle": ["oracle"],
    "dynamodb": ["dynamodb"],
    "cassandra": ["cassandra"],
    "bigquery": ["bigquery", "google bigquery"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
    "sklearn": ["scikit-learn", "sklearn"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "jupyter": ["jupyter"],
    "keras": ["keras"],
    "xgboost": ["xgboost"],
    "langchain": ["langchain"],
    "langgraph": ["langgraph"],
    "rag": ["rag", "retrieval augmented generation"],
    "vector_database": ["vector database", "vector databases"],
    "pinecone": ["pinecone"],
    "chromadb": ["chroma", "chromadb"],
    "huggingface": ["hugging face", "huggingface"],
    "openai": ["openai"],
    "claude": ["claude"],
    "gemini": ["gemini"],
    "mcp": ["mcp"],
    "llamaindex": ["llamaindex", "llama index"],
    "anthropic": ["anthropic"],
    "ollama": ["ollama"],
    "copilot": ["github copilot", "copilot"],
    "cursor": ["cursor"],
    "chatgpt": ["chatgpt"],
    "codex": ["codex"],
    "llm_token": ["llm", "large language models"],
    "prompt_engineering": ["prompt engineering"],
    "fine_tuning": ["fine-tuning", "fine tuning"],
    "agent_framework": ["agent", "ai agents"],
    "jest": ["jest"],
    "pytest": ["pytest"],
    "selenium": ["selenium"],
    "cypress": ["cypress"],
    "junit": ["junit"],
    "mocha": ["mocha"],
    "playwright": ["playwright"],
    "agile": ["agile", "agile methodologies"],
    "scrum": ["scrum"],
    "tdd": ["tdd", "test driven development", "test-driven development"],
    "devops": ["devops", "dev ops"],
    "sre": ["sre", "site reliability engineering"],
    "microservices": ["microservices"],
}
# ensure every TECH has at least itself as synonym
for t in TECHS:
    synonyms.setdefault(t, [t])

# Compute asaniczka SWE uids + their structured skill sets
asan_uids = set(df.loc[df.asaniczka, "uid"].tolist())
sk_asan = sk[sk.uid.isin(asan_uids)].copy()
# lower once
sk_asan["skill_l"] = sk_asan["skill"].str.lower().str.strip()
struct_by_uid = sk_asan.groupby("uid")["skill_l"].apply(set).to_dict()

# For each TECH, compute asaniczka structured rate (share of uids whose set contains any synonym)
struct_rows = []
extract_asan_rates = arshkon_all  # placeholder; we want asaniczka specifically
asan_swe_n = len(asan_uids)
for t in TECHS:
    syns = set(s.lower() for s in synonyms.get(t, [t]))
    # count how many asaniczka uids have any synonym in their structured skills
    hit = 0
    for uid, sset in struct_by_uid.items():
        if sset & syns:
            hit += 1
    struct_rate = hit / asan_swe_n if asan_swe_n else 0
    # extraction rate: from the tech matrix on asaniczka rows
    ext_rate = df.loc[df.asaniczka, t].mean()
    struct_rows.append(
        {
            "tech": t,
            "asaniczka_extraction_rate": float(ext_rate),
            "asaniczka_structured_rate": float(struct_rate),
            "diff_structured_minus_extracted": float(struct_rate - ext_rate),
            "synonyms": "|".join(sorted(syns)),
        }
    )
struct_cmp = pd.DataFrame(struct_rows)
struct_cmp.to_csv(TAB / "structured_vs_extracted_asaniczka.csv", index=False)
# Spearman correlation on 107 techs
rho, pval = stats.spearmanr(struct_cmp.asaniczka_extraction_rate, struct_cmp.asaniczka_structured_rate)
print(f"  Spearman rho (extraction vs structured, n={len(struct_cmp)} techs) = {rho:.3f}  p={pval:.1e}")

# Largest divergences
struct_cmp_sorted = struct_cmp.copy()
struct_cmp_sorted["abs_diff"] = struct_cmp_sorted.diff_structured_minus_extracted.abs()
top_div = struct_cmp_sorted.sort_values("abs_diff", ascending=False).head(20)
top_div.to_csv(TAB / "structured_vs_extracted_top_divergence.csv", index=False)
print("  Top 10 divergences (structured - extracted):")
print(top_div.head(10)[["tech", "asaniczka_extraction_rate", "asaniczka_structured_rate", "diff_structured_minus_extracted"]].to_string(index=False))

# ---------- Step 9: Seniority-level skill differences from structured data ----------
print("\n[Step 9] Asaniczka seniority × skills (chi-squared + BH-FDR)...")
# Build asaniczka df: uid, skill_set, seniority
asan_df = df.loc[df.asaniczka, ["uid", "seniority_final"]].copy()
# map uid → structured skill set
asan_df["skill_set"] = asan_df.uid.map(lambda u: struct_by_uid.get(u, set()))
# Junior (entry or associate) vs mid-senior for structured data
asan_df["group"] = asan_df.seniority_final.map(
    lambda s: "junior_final" if s in ("entry", "associate") else ("mid_senior" if s == "mid-senior" else None)
)
asan_j = asan_df[asan_df.group == "junior_final"]
asan_m = asan_df[asan_df.group == "mid_senior"]
print(f"  asaniczka junior_final: n={len(asan_j)}, mid_senior: n={len(asan_m)}")
# top-500 skills from global frequency
top500 = sf.sort_values("n_postings", ascending=False).head(500).skill.tolist()

chi_rows = []
n_j = len(asan_j)
n_m = len(asan_m)
for skill in top500:
    in_j = sum(1 for s in asan_j.skill_set if skill in s)
    in_m = sum(1 for s in asan_m.skill_set if skill in s)
    if in_j + in_m < 10:
        continue
    out_j = n_j - in_j
    out_m = n_m - in_m
    table = np.array([[in_j, out_j], [in_m, out_m]])
    try:
        chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    except ValueError:
        continue
    rate_j = in_j / n_j if n_j else 0
    rate_m = in_m / n_m if n_m else 0
    # log-odds
    a, b, c, d2 = in_j + 0.5, out_j + 0.5, in_m + 0.5, out_m + 0.5
    log_or = math.log((a / b) / (c / d2))
    chi_rows.append(
        {
            "skill": skill,
            "in_junior": in_j,
            "in_midsenior": in_m,
            "rate_junior": rate_j,
            "rate_midsenior": rate_m,
            "delta": rate_j - rate_m,
            "chi2": chi2,
            "p_value": p,
            "log_odds_junior_over_midsenior": log_or,
        }
    )
chi_df = pd.DataFrame(chi_rows).sort_values("p_value")
# BH-FDR at q=0.05
m = len(chi_df)
chi_df["rank"] = np.arange(1, m + 1)
chi_df["bh_thresh"] = chi_df["rank"] / m * 0.05
chi_df["significant_fdr05"] = chi_df.p_value <= chi_df.bh_thresh
# Bonferroni
chi_df["bonferroni_thresh"] = 0.05 / m
chi_df["significant_bonferroni"] = chi_df.p_value <= chi_df.bonferroni_thresh
chi_df.to_csv(TAB / "asaniczka_seniority_skill_chi.csv", index=False)
print(f"  {chi_df.significant_bonferroni.sum()} skills significant at Bonferroni 0.05 (of {m} tested)")
print(f"  {chi_df.significant_fdr05.sum()} skills significant at BH-FDR 0.05")
print("  Top 10 skills associated with junior_final (positive delta, significant BH):")
print(
    chi_df[(chi_df.delta > 0) & (chi_df.significant_fdr05)]
    .head(10)[["skill", "rate_junior", "rate_midsenior", "delta", "p_value"]]
    .to_string(index=False)
)
print("  Top 10 skills associated with mid_senior (negative delta, significant BH):")
print(
    chi_df[(chi_df.delta < 0) & (chi_df.significant_fdr05)]
    .head(10)[["skill", "rate_junior", "rate_midsenior", "delta", "p_value"]]
    .to_string(index=False)
)

# ---------- Figures ----------
print("\n[Figures] Building T14 figures...")

# (1) Tech trajectory heatmap — top 30 rising + top 15 declining by delta_arshkon
fig_traj = traj.copy()
fig_traj = fig_traj[fig_traj.arshkon_rate + fig_traj.scraped_rate > 0.005]
# top 30 rising + top 15 declining
rising_top = fig_traj.sort_values("delta_arshkon", ascending=False).head(30)
declining_top = fig_traj.sort_values("delta_arshkon", ascending=True).head(15)
heat_techs = pd.concat([rising_top, declining_top]).drop_duplicates("tech")
heat_techs = heat_techs.sort_values("delta_arshkon", ascending=False)
heat = heat_techs[["tech", "arshkon_rate", "asaniczka_rate", "pooled_2024_rate", "scraped_rate"]].set_index("tech")

fig, ax = plt.subplots(figsize=(8, 12))
im = ax.imshow(heat.values * 100, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=60)
ax.set_yticks(range(len(heat)))
ax.set_yticklabels(heat.index, fontsize=7)
ax.set_xticks(range(4))
ax.set_xticklabels(["arshkon\n2024", "asaniczka\n2024", "pooled\n2024", "scraped\n2026"], fontsize=8)
ax.set_title("Technology mention rates (%) — top rising + declining by arshkon→scraped delta", fontsize=9)
# annotate
for i in range(len(heat)):
    for j in range(4):
        v = heat.values[i, j] * 100
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6, color="black" if v < 30 else "white")
cb = plt.colorbar(im, ax=ax, fraction=0.03)
cb.set_label("% of postings mentioning tech")
plt.tight_layout()
plt.savefig(FIG / "tech_trajectory_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# (2) Co-occurrence network visualization — 2024 vs 2026 side by side
print("  drawing co-occurrence network...")


def draw_network(G, ax, comms, title, size_scale, pos=None):
    node_to_comm = {}
    for i, c in enumerate(comms):
        for t in c:
            node_to_comm[t] = i
    comm_colors = plt.cm.tab20(np.linspace(0, 1, max(len(comms), 1)))
    if pos is None:
        pos = nx.spring_layout(G, seed=42, k=1.0 / math.sqrt(len(G.nodes())), iterations=100)
    # edges
    for (a, b, d) in G.edges(data=True):
        w = d.get("weight", 0)
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], "-", color="gray", alpha=0.25, linewidth=max(w * 5, 0.3))
    # nodes
    sizes = [max(size_scale.get(t, 0) * 2000, 20) for t in G.nodes()]
    colors = [comm_colors[node_to_comm.get(t, 0)] for t in G.nodes()]
    for t, (x, y), s, c in zip(G.nodes(), [pos[t] for t in G.nodes()], sizes, colors):
        ax.scatter(x, y, s=s, color=c, edgecolors="black", linewidths=0.4, alpha=0.9, zorder=2)
        ax.text(x, y, t, fontsize=6, ha="center", va="center", zorder=3)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return pos


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# shared layout: use 2026 layout for visual anchoring
pos_shared = nx.spring_layout(G_2026, seed=42, k=1.1 / math.sqrt(len(G_2026.nodes())), iterations=150)
draw_network(G_2024, ax1, comm_2024, f"2024 (pooled, n={N_pool:,}) — {len(comm_2024)} communities, |phi|≥{PHI_THRESHOLD}", pooled_all.to_dict(), pos=pos_shared)
draw_network(G_2026, ax2, comm_2026, f"2026 (scraped, n={N_scrap:,}) — {len(comm_2026)} communities, |phi|≥{PHI_THRESHOLD}", scraped_all.to_dict(), pos=pos_shared)
plt.tight_layout()
plt.savefig(FIG / "cooccurrence_network_2024_vs_2026.png", dpi=150, bbox_inches="tight")
plt.close()

# (3) Community comparison: heatmap of 2024 community × 2026 community (how membership shifted)
cm_table = pd.DataFrame(
    {
        "tech": NET_TECHS,
        "comm_2024": [cm24.get(t, -1) for t in NET_TECHS],
        "comm_2026": [cm26.get(t, -1) for t in NET_TECHS],
    }
)
conf = pd.crosstab(cm_table.comm_2024, cm_table.comm_2026)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(conf.values, aspect="auto", cmap="viridis")
ax.set_xticks(range(len(conf.columns)))
ax.set_xticklabels(conf.columns, rotation=45)
ax.set_yticks(range(len(conf.index)))
ax.set_yticklabels(conf.index)
ax.set_xlabel("2026 community id")
ax.set_ylabel("2024 community id")
ax.set_title("Tech community mapping: 2024 (pooled) → 2026 (scraped)\nLouvain on phi>0.15 edges")
for i in range(len(conf.index)):
    for j in range(len(conf.columns)):
        ax.text(j, i, int(conf.values[i, j]), ha="center", va="center", color="white" if conf.values[i, j] < conf.values.max() / 2 else "black", fontsize=8)
plt.colorbar(im, ax=ax, fraction=0.03)
plt.tight_layout()
plt.savefig(FIG / "community_shift_confusion.png", dpi=150, bbox_inches="tight")
plt.close()

# (4) Structured vs extracted scatter
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(
    struct_cmp.asaniczka_extraction_rate * 100,
    struct_cmp.asaniczka_structured_rate * 100,
    alpha=0.6,
    s=15,
    color="steelblue",
)
# label top 15 by combined rate
struct_cmp["combined"] = struct_cmp.asaniczka_extraction_rate + struct_cmp.asaniczka_structured_rate
for _, row in struct_cmp.nlargest(15, "combined").iterrows():
    ax.annotate(
        row["tech"],
        (row.asaniczka_extraction_rate * 100, row.asaniczka_structured_rate * 100),
        fontsize=7,
        alpha=0.8,
    )
# large divergences
for _, row in struct_cmp_sorted.nlargest(8, "abs_diff").iterrows():
    ax.annotate(
        row["tech"],
        (row.asaniczka_extraction_rate * 100, row.asaniczka_structured_rate * 100),
        fontsize=7,
        color="red",
        alpha=0.9,
    )
maxv = max(struct_cmp.asaniczka_extraction_rate.max(), struct_cmp.asaniczka_structured_rate.max()) * 100
ax.plot([0, maxv], [0, maxv], "--", color="gray", alpha=0.5, linewidth=1, label="y = x")
ax.set_xlabel("Asaniczka description extraction rate (%)")
ax.set_ylabel("Asaniczka structured skills_raw rate (%)")
ax.set_title(f"Structured vs extracted tech frequencies (asaniczka SWE, n_techs=107)\nSpearman ρ={rho:.3f}  p={pval:.1e}")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "structured_vs_extracted.png", dpi=150, bbox_inches="tight")
plt.close()

# Save summary JSON
summary = {
    "n_rows": int(len(df)),
    "n_techs": len(TECHS),
    "ai_rate_overall": float(df.ai_mentioning.mean()),
    "ai_rate_by_source": {
        "arshkon": float(df.loc[df.arshkon, "ai_mentioning"].mean()),
        "asaniczka": float(df.loc[df.asaniczka, "ai_mentioning"].mean()),
        "scraped": float(df.loc[df.scraped, "ai_mentioning"].mean()),
    },
    "tech_count_mean_by_source": {
        "arshkon": float(df.loc[df.arshkon, "tech_count"].mean()),
        "asaniczka": float(df.loc[df.asaniczka, "tech_count"].mean()),
        "scraped": float(df.loc[df.scraped, "tech_count"].mean()),
    },
    "phi_threshold": PHI_THRESHOLD,
    "n_edges": len(edges),
    "n_communities_2024": len(comm_2024),
    "n_communities_2026": len(comm_2026),
    "struct_extract_spearman_rho": float(rho),
    "struct_extract_spearman_p": float(pval),
    "asaniczka_junior_n": int(n_j),
    "asaniczka_midsenior_n": int(n_m),
    "n_chi_tests": int(m),
    "n_chi_bh05": int(chi_df.significant_fdr05.sum()),
    "n_chi_bonf05": int(chi_df.significant_bonferroni.sum()),
}
(TAB / "summary.json").write_text(json.dumps(summary, indent=2))
print("\nDone. Outputs:")
for p in sorted((TAB).glob("*")):
    print("  ", p.relative_to(BASE))
for p in sorted((FIG).glob("*")):
    print("  ", p.relative_to(BASE))
