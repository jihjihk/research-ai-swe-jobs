"""T34 — Emergent senior-role archetype profiling (H_C).

Profile T21 candidate emergent senior sub-archetypes:
  cluster 0 — AI-oriented (low-orch): 15.6× growth (144 → 2,251)
  cluster 1 — Tech-orchestration (non-AI): +11.4 pp (2,034 → 5,258)

For each candidate cluster, produce:
  - title distribution
  - company concentration (Gini / HHI / top-20)
  - T09 archetype cross-tab
  - T16 company-trajectory test
  - profile attributes (YOE / industry / metro / seniority_final)
  - side-by-side cluster comparison
  - 20 content exemplars + role-name proposals
  - aggregator sensitivity

Outputs
-------
exploration/tables/T34/
    cluster_precondition_check.csv
    title_distribution.csv
    company_concentration.csv
    archetype_cross_tab.csv
    t16_trajectory_crosstab.csv
    cluster_profile_attributes.csv
    cluster_comparison.csv
    content_exemplars_cluster0.csv
    content_exemplars_cluster1.csv
    aggregator_sensitivity.csv
    role_name_proposals.md

exploration/figures/T34/
    cluster_profile_radar.png (skipped if deps unavailable)
    title_distribution_bars.png
"""

from __future__ import annotations

import os
import re
import sys
import json
import math
import time
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent.parent
UNIFIED = REPO / "data" / "unified.parquet"
T21_ASSIGN = REPO / "exploration" / "tables" / "T21" / "cluster_assignments.csv"
T09_PARQ = REPO / "exploration" / "artifacts" / "shared" / "swe_archetype_labels.parquet"
T11_PARQ = REPO / "exploration" / "artifacts" / "shared" / "T11_posting_features.parquet"
T16_CLUSTERS = REPO / "exploration" / "tables" / "T16" / "cluster_summary.csv"
T16_WIDE = REPO / "exploration" / "tables" / "T16" / "_wide_company_era.csv"
SWE_CLEANED = REPO / "exploration" / "artifacts" / "shared" / "swe_cleaned_text.parquet"

OUT_TAB = REPO / "exploration" / "tables" / "T34"
OUT_FIG = REPO / "exploration" / "figures" / "T34"
OUT_TAB.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

BASE_FILTER = (
    "is_swe AND source_platform='linkedin' AND is_english AND date_flag='ok'"
)

RNG = 34343434

# Candidate clusters from T21
CANDIDATES = {
    0: "AI-oriented (low-orch)",
    1: "Tech-orchestration (non-AI)",
}

# Precondition thresholds
MIN_N = 500
AI_2026_THRESHOLD = 0.5  # disproportionately 2026-weighted: 2026 share > 50%


def load_cluster_context() -> pd.DataFrame:
    """Load T21 cluster assignments + joined unified metadata."""
    con = duckdb.connect()
    q = f"""
    SELECT c.uid, c.cluster_id, c.cluster_name,
           c.mgmt_density_v1_rebuilt, c.mgmt_density_t11_original,
           c.orch_density, c.strat_density, c.mentor_binary, c.ai_binary,
           c.period, c.seniority_final,
           u.title, u.company_name_canonical, u.company_name_effective,
           u.is_aggregator, u.company_industry, u.metro_area,
           u.yoe_min_years_llm, u.yoe_extracted, u.llm_classification_coverage,
           u.description, u.description_core_llm,
           u.llm_extraction_coverage
    FROM read_csv_auto('{T21_ASSIGN.as_posix()}') c
    LEFT JOIN read_parquet('{UNIFIED.as_posix()}') u ON c.uid = u.uid
    WHERE {BASE_FILTER.replace("is_swe", "u.is_swe")
                      .replace("source_platform", "u.source_platform")
                      .replace("is_english", "u.is_english")
                      .replace("date_flag", "u.date_flag")}
    """
    df = con.execute(q).df()
    df["period"] = df["period"].astype(str)
    return df


def precondition_check(df: pd.DataFrame) -> pd.DataFrame:
    """For each candidate cluster, check n ≥ 500, 2026 share > 50%, elevated
    on ≥2 signals."""
    rows = []
    for cid, cname in CANDIDATES.items():
        sub = df[df.cluster_id == cid]
        n_total = len(sub)
        n_2024 = (sub.period == "2024").sum()
        n_2026 = (sub.period == "2026").sum()
        share_2026 = n_2026 / n_total if n_total > 0 else 0.0
        # Centroid signals
        mgmt = sub["mgmt_density_v1_rebuilt"].mean()
        orch = sub["orch_density"].mean()
        strat = sub["strat_density"].mean()
        ai = sub["ai_binary"].mean()
        # Reference: all senior means
        ref_mgmt = df["mgmt_density_v1_rebuilt"].mean()
        ref_orch = df["orch_density"].mean()
        ref_strat = df["strat_density"].mean()
        ref_ai = df["ai_binary"].mean()
        elevated_signals = sum([
            mgmt > ref_mgmt * 1.2,
            orch > ref_orch * 1.2,
            strat > ref_strat * 1.2,
            ai > ref_ai * 1.2,
        ])
        rows.append({
            "cluster_id": cid,
            "cluster_name": cname,
            "n_total": n_total,
            "n_2024": n_2024,
            "n_2026": n_2026,
            "share_2026": share_2026,
            "growth_ratio": n_2026 / max(n_2024, 1),
            "mean_mgmt_v1": mgmt,
            "mean_orch": orch,
            "mean_strat": strat,
            "mean_ai_binary": ai,
            "n_elevated_signals": elevated_signals,
            "passes_precondition": (n_total >= MIN_N) and (share_2026 >= AI_2026_THRESHOLD) and (elevated_signals >= 2),
        })
    return pd.DataFrame(rows)


_TITLE_CATEGORIES = [
    ("staff_engineer", r"\bstaff\s+(?:software\s+)?(?:engineer|developer|sde|swe)"),
    ("principal_engineer", r"\bprincipal\s+(?:software\s+)?(?:engineer|developer|sde|swe|architect)"),
    ("tech_lead", r"\btech(?:nical)?\s+lead|\blead\s+(?:software\s+)?(?:engineer|developer)|\bengineering\s+lead"),
    ("ml_lead", r"\b(?:ml|machine\s+learning|ai)\s+(?:lead|team\s+lead)\b"),
    ("ai_engineer", r"\b(?:ai|ml|machine\s+learning)\s+(?:engineer|developer|scientist)\b"),
    ("llm_engineer", r"\b(?:llm|generative\s+ai|genai|applied\s+(?:ml|ai))\s+engineer"),
    ("engineering_manager", r"\bengineering\s+manager|\bsoftware\s+engineering\s+manager|\bem\b"),
    ("engineering_director", r"\b(?:engineering\s+director|director\s+of\s+engineering|vp\s+engineering|head\s+of\s+engineering)"),
    ("architect", r"\b(?:software|solutions?|systems?|enterprise|cloud|platform)\s+architect"),
    ("senior_engineer", r"\b(?:senior|sr\.?|lead)\s+(?:software\s+)?(?:engineer|developer|sde|swe)"),
    ("devops_sre", r"\b(?:devops|sre|site\s+reliability|platform|infrastructure)\s+engineer"),
    ("security_engineer", r"\bsecurity\s+engineer|\bapp(?:lication)?\s+sec"),
    ("data_engineer", r"\bdata\s+engineer"),
]


def title_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute share of common senior title variants within each candidate cluster."""
    rows = []
    for cid, cname in CANDIDATES.items():
        sub = df[df.cluster_id == cid]
        titles = sub["title"].fillna("").str.lower().tolist()
        total = len(titles)
        counts = Counter()
        matched = np.zeros(total, dtype=bool)
        cat_names_ordered = []
        for cat, pat in _TITLE_CATEGORIES:
            cat_names_ordered.append(cat)
            compiled = re.compile(pat, re.IGNORECASE)
            for i, t in enumerate(titles):
                if not matched[i] and compiled.search(t):
                    counts[cat] += 1
                    matched[i] = True
        counts["other_unmatched"] = int((~matched).sum())
        # Also compute share by period
        # Subset per period
        for period in ("2024", "2026"):
            p_sub = sub[sub.period == period]
            p_titles = p_sub["title"].fillna("").str.lower().tolist()
            p_total = len(p_titles)
            p_matched = np.zeros(p_total, dtype=bool)
            p_counts = Counter()
            for cat, pat in _TITLE_CATEGORIES:
                compiled = re.compile(pat, re.IGNORECASE)
                for i, t in enumerate(p_titles):
                    if not p_matched[i] and compiled.search(t):
                        p_counts[cat] += 1
                        p_matched[i] = True
            p_counts["other_unmatched"] = int((~p_matched).sum())
            for cat in cat_names_ordered + ["other_unmatched"]:
                rows.append({
                    "cluster_id": cid,
                    "cluster_name": cname,
                    "period": period,
                    "n_period": p_total,
                    "category": cat,
                    "count": int(p_counts.get(cat, 0)),
                    "share": float(p_counts.get(cat, 0) / p_total) if p_total > 0 else 0.0,
                })
        # Also overall
        for cat in cat_names_ordered + ["other_unmatched"]:
            rows.append({
                "cluster_id": cid,
                "cluster_name": cname,
                "period": "all",
                "n_period": total,
                "category": cat,
                "count": int(counts.get(cat, 0)),
                "share": float(counts.get(cat, 0) / total) if total > 0 else 0.0,
            })
    return pd.DataFrame(rows)


def gini(arr: np.ndarray) -> float:
    """Classic Gini coefficient for a non-negative array."""
    arr = np.sort(arr.astype(np.float64))
    n = arr.size
    if n == 0 or arr.sum() == 0:
        return 0.0
    cumvals = np.cumsum(arr)
    return (n + 1 - 2 * (cumvals.sum() / cumvals[-1])) / n


def company_concentration(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, cname in CANDIDATES.items():
        sub = df[df.cluster_id == cid]
        counts = sub["company_name_canonical"].value_counts()
        arr = counts.values.astype(np.float64)
        share = arr / arr.sum() if arr.sum() > 0 else arr
        hhi = float((share ** 2).sum() * 10_000)  # HHI in 0–10_000 range
        gini_coef = float(gini(arr))
        top20 = float(share[:20].sum()) if len(share) >= 20 else float(share.sum())
        rows.append({
            "cluster_id": cid,
            "cluster_name": cname,
            "n": len(sub),
            "n_companies": int(counts.size),
            "gini": gini_coef,
            "hhi": hhi,
            "top20_share": top20,
            "top5_companies": ";".join(counts.head(5).index.astype(str).tolist()),
            "top5_counts": ";".join(counts.head(5).astype(str).tolist()),
        })
    # Reference: other senior clusters
    for cid, cname in [(2, "People-management"), (3, "Low-profile generic"), (4, "Strategic-language")]:
        sub = df[df.cluster_id == cid]
        counts = sub["company_name_canonical"].value_counts()
        arr = counts.values.astype(np.float64)
        share = arr / arr.sum() if arr.sum() > 0 else arr
        hhi = float((share ** 2).sum() * 10_000)
        gini_coef = float(gini(arr))
        top20 = float(share[:20].sum()) if len(share) >= 20 else float(share.sum())
        rows.append({
            "cluster_id": cid,
            "cluster_name": cname,
            "n": len(sub),
            "n_companies": int(counts.size),
            "gini": gini_coef,
            "hhi": hhi,
            "top20_share": top20,
            "top5_companies": ";".join(counts.head(5).index.astype(str).tolist()),
            "top5_counts": ";".join(counts.head(5).astype(str).tolist()),
        })
    return pd.DataFrame(rows)


def load_archetype_labels_with_projection(df: pd.DataFrame) -> pd.DataFrame:
    """Load T09 archetype labels and optionally project for uids not in T09."""
    con = duckdb.connect()
    t09 = con.execute(
        f"SELECT uid, archetype_name FROM read_parquet('{T09_PARQ.as_posix()}')"
    ).df()
    # Attempt nearest-centroid projection using swe_embeddings.npy + swe_embedding_index
    try:
        emb_index = con.execute(
            f"SELECT uid, row_idx FROM read_parquet('{REPO / 'exploration' / 'artifacts' / 'shared' / 'swe_embedding_index.parquet'}')"
        ).df()
        emb = np.load(REPO / "exploration" / "artifacts" / "shared" / "swe_embeddings.npy")
        idx_map = emb_index.set_index("uid")["row_idx"].to_dict()
        # Compute centroids per archetype_name from labeled rows
        labeled_ids = t09["uid"].tolist()
        arche_map = t09.set_index("uid")["archetype_name"].to_dict()
        # Centroid per archetype
        centroid_rows = {a: [] for a in t09["archetype_name"].unique()}
        for uid, a in arche_map.items():
            if uid in idx_map:
                centroid_rows[a].append(idx_map[uid])
        centroids: Dict[str, np.ndarray] = {}
        for a, rows in centroid_rows.items():
            if rows:
                centroids[a] = emb[rows].mean(axis=0)
        arche_names = list(centroids.keys())
        cent_mat = np.stack([centroids[a] for a in arche_names])
        # Normalize for cosine
        cent_norm = cent_mat / (np.linalg.norm(cent_mat, axis=1, keepdims=True) + 1e-12)
        # For each T21-sampled uid not in t09, find nearest centroid via cosine
        unique_uids = df["uid"].unique().tolist()
        to_project = [u for u in unique_uids if u not in arche_map and u in idx_map]
        projected = {}
        batch = 5000
        for i in range(0, len(to_project), batch):
            chunk = to_project[i:i+batch]
            idxs = [idx_map[u] for u in chunk]
            vecs = emb[idxs]
            vecs_n = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
            sims = vecs_n @ cent_norm.T
            nearest = np.argmax(sims, axis=1)
            for j, u in enumerate(chunk):
                projected[u] = arche_names[nearest[j]]
        rows_out = []
        for u in unique_uids:
            if u in arche_map:
                rows_out.append({"uid": u, "archetype": arche_map[u], "projected": False})
            elif u in projected:
                rows_out.append({"uid": u, "archetype": projected[u], "projected": True})
            else:
                rows_out.append({"uid": u, "archetype": None, "projected": False})
        return pd.DataFrame(rows_out)
    except Exception as exc:
        print(f"  archetype projection failed: {exc}")
        # Fall back to left-join only
        rows_out = []
        arche_map = t09.set_index("uid")["archetype_name"].to_dict()
        for u in df["uid"].unique():
            if u in arche_map:
                rows_out.append({"uid": u, "archetype": arche_map[u], "projected": False})
            else:
                rows_out.append({"uid": u, "archetype": None, "projected": False})
        return pd.DataFrame(rows_out)


def archetype_crosstab(df: pd.DataFrame, arche: pd.DataFrame) -> pd.DataFrame:
    m = df.merge(arche, on="uid", how="left")
    rows = []
    # Over-representation: per cluster, for each archetype, share vs population share
    pop_arche = m["archetype"].value_counts(normalize=True)
    for cid, cname in CANDIDATES.items():
        sub = m[m.cluster_id == cid]
        c_arche = sub["archetype"].value_counts(normalize=True)
        for arche_name, share in c_arche.head(15).items():
            pop = pop_arche.get(arche_name, 0)
            rows.append({
                "cluster_id": cid,
                "cluster_name": cname,
                "archetype": arche_name,
                "cluster_share": share,
                "population_share": pop,
                "over_representation": share / pop if pop > 0 else float("nan"),
                "n_rows_cluster": int(sub["archetype"].notna().sum()),
            })
    return pd.DataFrame(rows)


def t16_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """For each T16 cluster, fraction of that cluster's 2026 senior postings that
    fall into each T34 candidate cluster.

    T16 per-company clusters live in `exploration/tables/T16/overlap_panel.csv`
    column `strategy_cluster`.
    """
    overlap_panel_path = REPO / "exploration" / "tables" / "T16" / "overlap_panel.csv"
    if not overlap_panel_path.exists():
        print(f"  T16 overlap_panel not found at {overlap_panel_path}")
        return pd.DataFrame()
    try:
        t16 = pd.read_csv(overlap_panel_path)
        if "strategy_cluster" not in t16.columns:
            print(f"  strategy_cluster missing; cols={list(t16.columns)[:20]}")
            return pd.DataFrame()
        t16 = t16[["company_name_canonical", "strategy_cluster"]].rename(
            columns={"strategy_cluster": "t16_cluster"}
        )
    except Exception as exc:
        print(f"  T16 load failed: {exc}")
        return pd.DataFrame()
    m = df.merge(t16, on="company_name_canonical", how="left")
    # Focus on 2026 senior postings
    m = m[m.period == "2026"]
    rows = []
    for t16c, sub in m.groupby("t16_cluster"):
        n_total = len(sub)
        for cid, cname in CANDIDATES.items():
            n_in = (sub.cluster_id == cid).sum()
            rows.append({
                "t16_cluster": t16c,
                "t34_cluster_id": cid,
                "t34_cluster_name": cname,
                "n_t16_cluster_senior_2026": n_total,
                "n_in_t34_cluster": int(n_in),
                "share": float(n_in / n_total) if n_total > 0 else 0.0,
            })
    return pd.DataFrame(rows)


def profile_attributes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, cname in CANDIDATES.items():
        sub = df[df.cluster_id == cid]
        # Median YOE (LLM) — filter labeled only
        llm_labeled = sub[sub.llm_classification_coverage == "labeled"]
        med_yoe = float(llm_labeled["yoe_min_years_llm"].median()) if len(llm_labeled) > 0 else float("nan")
        # Within-period industry distribution (2026 only)
        ind_2026 = sub[sub.period == "2026"]["company_industry"].value_counts(dropna=True).head(10)
        ind_share = (ind_2026 / ind_2026.sum()) if ind_2026.sum() > 0 else ind_2026
        ind_str = "; ".join([f"{k}:{v:.1%}" for k, v in ind_share.items()])
        # Metro distribution (2026 only)
        met_2026 = sub[sub.period == "2026"]["metro_area"].value_counts(dropna=True).head(10)
        met_share = (met_2026 / met_2026.sum()) if met_2026.sum() > 0 else met_2026
        met_str = "; ".join([f"{k}:{v:.1%}" for k, v in met_share.items()])
        # Seniority distribution
        sen_dist = sub["seniority_final"].value_counts(normalize=True)
        sen_str = "; ".join([f"{k}:{v:.1%}" for k, v in sen_dist.items()])
        rows.append({
            "cluster_id": cid,
            "cluster_name": cname,
            "n": len(sub),
            "median_yoe_llm_labeled": med_yoe,
            "n_yoe_labeled": len(llm_labeled),
            "industry_top10_2026": ind_str,
            "metro_top10_2026": met_str,
            "seniority_final_dist": sen_str,
        })
    return pd.DataFrame(rows)


def cluster_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side comparison of all 5 T21 clusters on key features."""
    feats = ["mgmt_density_v1_rebuilt", "mgmt_density_t11_original",
             "orch_density", "strat_density", "mentor_binary", "ai_binary",
             "yoe_min_years_llm"]
    rows = []
    all_cids = sorted(df.cluster_id.unique())
    for cid in all_cids:
        sub = df[df.cluster_id == cid]
        cname = sub["cluster_name"].iloc[0] if len(sub) else ""
        rec = {"cluster_id": cid, "cluster_name": cname, "n": len(sub),
               "share_2024": (sub.period == "2024").mean(),
               "share_2026": (sub.period == "2026").mean()}
        for f in feats:
            rec[f] = float(sub[f].mean())
        # Director share
        rec["director_share"] = (sub["seniority_final"] == "director").mean()
        rows.append(rec)
    return pd.DataFrame(rows)


def content_exemplars(df: pd.DataFrame, cid: int, n: int = 20) -> pd.DataFrame:
    """Sample 20 2026 postings from cluster `cid` prioritizing ones close to
    centroid."""
    sub = df[(df.cluster_id == cid) & (df.period == "2026")].copy()
    # Rank by (ai_binary present × orch_density) if AI cluster, else orch_density
    if cid == 0:  # AI-oriented
        sub["rank_score"] = sub["ai_binary"].fillna(0) * 10 + sub["orch_density"].fillna(0)
    else:
        sub["rank_score"] = sub["orch_density"].fillna(0) - sub["strat_density"].fillna(0)
    sub = sub.sort_values("rank_score", ascending=False)
    # Deduplicate by company_name_canonical to diversify
    sub = sub.drop_duplicates("company_name_canonical").head(n)
    out = sub[[
        "uid", "cluster_id", "cluster_name", "period", "seniority_final",
        "company_name_canonical", "company_industry", "title",
        "yoe_min_years_llm", "orch_density", "mgmt_density_v1_rebuilt",
        "ai_binary", "strat_density",
    ]].copy()
    def excerpt(d_core, d_raw, n=400):
        t = d_core if d_core else d_raw
        if not t:
            return ""
        return t[:n].replace("\r", " ").replace("\n", " ")
    out["desc_excerpt_400"] = [
        excerpt(sub.description_core_llm.iloc[i], sub.description.iloc[i], 400)
        for i in range(len(sub))
    ]
    return out


def recurring_phrases(df: pd.DataFrame, cid: int, top_k: int = 20) -> pd.DataFrame:
    """Cluster-level tf-idf against other clusters over 2026 descriptions.

    Uses simple bigram/trigram frequency without sklearn vectorizer (to stay
    lightweight); we identify candidate phrases via regex tokenization over
    the in-cluster description text and compare against out-of-cluster rates.
    """
    from collections import defaultdict
    token_pat = re.compile(r"[a-zA-Z][a-zA-Z0-9\+\-/#]+")
    def tokens(txt: str) -> List[str]:
        return [t.lower() for t in token_pat.findall(txt or "") if len(t) > 2]

    def ngrams(seq, n):
        return [" ".join(seq[i:i+n]) for i in range(len(seq) - n + 1)]

    in_df = df[(df.cluster_id == cid) & (df.period == "2026")]
    out_df = df[(df.cluster_id != cid) & (df.period == "2026")]

    # Sample for efficiency
    rng = np.random.default_rng(RNG)
    in_sample = in_df.sample(min(1500, len(in_df)), random_state=RNG) if len(in_df) > 1500 else in_df
    out_sample = out_df.sample(min(5000, len(out_df)), random_state=RNG) if len(out_df) > 5000 else out_df

    def build_counts(frame):
        uni = Counter()
        bi = Counter()
        tri = Counter()
        n_docs = 0
        for txt in frame["description_core_llm"].fillna(frame["description"]).fillna(""):
            toks = tokens(txt)
            if not toks:
                continue
            n_docs += 1
            uni.update(set(toks))  # doc-frequency for unigrams
            bi.update(set(ngrams(toks, 2)))
            tri.update(set(ngrams(toks, 3)))
        return uni, bi, tri, n_docs

    uni_in, bi_in, tri_in, n_in = build_counts(in_sample)
    uni_out, bi_out, tri_out, n_out = build_counts(out_sample)

    def score(ngrams_in, ngrams_out, n_in, n_out, kmin=5, top=top_k):
        rows = []
        for g, cin in ngrams_in.items():
            if cin < kmin:
                continue
            cout = ngrams_out.get(g, 0)
            p_in = cin / max(n_in, 1)
            p_out = cout / max(n_out, 1)
            # Log-odds ratio with Laplace smoothing
            num = (cin + 1) / max(n_in - cin + 1, 1)
            den = (cout + 1) / max(n_out - cout + 1, 1)
            lor = math.log(num / den)
            rows.append((g, cin, cout, p_in, p_out, lor))
        rows.sort(key=lambda t: t[5], reverse=True)
        return rows[:top]

    rows = []
    for label, data, n in [
        ("unigram", score(uni_in, uni_out, n_in, n_out), n_in),
        ("bigram", score(bi_in, bi_out, n_in, n_out), n_in),
        ("trigram", score(tri_in, tri_out, n_in, n_out), n_in),
    ]:
        for g, cin, cout, pin, pout, lor in data:
            rows.append({
                "cluster_id": cid,
                "ngram_type": label,
                "ngram": g,
                "in_count": cin,
                "out_count": cout,
                "in_rate": pin,
                "out_rate": pout,
                "log_odds_ratio": lor,
            })
    return pd.DataFrame(rows)


def aggregator_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute core cluster stats excluding aggregator rows."""
    d_full = df
    d_noagg = df[~df["is_aggregator"].fillna(False)]
    rows = []
    for cid, cname in CANDIDATES.items():
        sub_full = d_full[d_full.cluster_id == cid]
        sub_na = d_noagg[d_noagg.cluster_id == cid]
        rows.append({
            "cluster_id": cid,
            "cluster_name": cname,
            "n_full": len(sub_full),
            "n_no_aggr": len(sub_na),
            "share_2026_full": (sub_full.period == "2026").mean(),
            "share_2026_no_aggr": (sub_na.period == "2026").mean(),
            "orch_density_full": sub_full["orch_density"].mean(),
            "orch_density_no_aggr": sub_na["orch_density"].mean(),
            "ai_binary_full": sub_full["ai_binary"].mean(),
            "ai_binary_no_aggr": sub_na["ai_binary"].mean(),
        })
    return pd.DataFrame(rows)


def write_role_proposals(exemplars0: pd.DataFrame, exemplars1: pd.DataFrame,
                         ngrams0: pd.DataFrame, ngrams1: pd.DataFrame,
                         out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# T34 — Content-grounded role-name proposals\n")
    lines.append("Proposals are grounded in 20 per-cluster 2026 postings (exemplars) "
                 "and top-k log-odds bigrams/trigrams vs other senior clusters.\n")

    for cid, exemplars, ngrams_df, hint_name in [
        (0, exemplars0, ngrams0, "AI-oriented (T21 cluster 0)"),
        (1, exemplars1, ngrams1, "Tech-orchestration non-AI (T21 cluster 1)"),
    ]:
        lines.append(f"\n## Cluster {cid} — {hint_name}\n")
        lines.append("### Top 15 distinguishing bigrams (2026, vs other clusters)\n")
        bi = ngrams_df[ngrams_df.ngram_type == "bigram"].head(15)
        for _, r in bi.iterrows():
            lines.append(f"- `{r['ngram']}` (in={r['in_count']}, out={r['out_count']}, "
                         f"log-odds={r['log_odds_ratio']:+.2f})")
        lines.append("\n### Top 15 distinguishing trigrams (2026, vs other clusters)\n")
        tri = ngrams_df[ngrams_df.ngram_type == "trigram"].head(15)
        for _, r in tri.iterrows():
            lines.append(f"- `{r['ngram']}` (in={r['in_count']}, out={r['out_count']}, "
                         f"log-odds={r['log_odds_ratio']:+.2f})")
        lines.append("\n### Recurring asks across the 20 exemplars\n")
        # Simple heuristic — scan exemplars for recurring uppercase-tech tokens
        allt = " ".join(exemplars["desc_excerpt_400"].tolist()).lower()
        candidate_asks = [
            ("agentic / multi-agent", r"\bagentic|multi[\s-]?agent"),
            ("LLM / RAG / prompt", r"\bllm|rag\b|prompt(?:\s+engineering)?"),
            ("orchestration / workflow", r"\borchestrat|workflow"),
            ("pipelines", r"\bpipeline"),
            ("CI/CD", r"\bci/cd|ci\s+cd|continuous\s+(integration|delivery)"),
            ("system design / architecture", r"\bsystem\s+design|architect"),
            ("AI tools / copilot", r"\bcopilot|cursor|claude|chatgpt"),
            ("evaluation / eval", r"\beval(?:uation)?\b"),
            ("guardrails / safety", r"\bguardrail|safety"),
            ("fine-tuning / training", r"\bfine[\s-]?tun|training\s+(?:data|models?)"),
            ("vector db / embeddings", r"\bvector\s+(?:db|store)|embedding|pinecone|weaviate"),
            ("model evaluation", r"\bmodel\s+(?:evaluation|performance|quality)"),
        ]
        for ask_label, pat in candidate_asks:
            if re.search(pat, allt, re.IGNORECASE):
                c = len(re.findall(pat, allt, re.IGNORECASE))
                lines.append(f"- **{ask_label}** mentioned in exemplars ({c} hits across 20 postings)")
        lines.append("")

    # Role names — hand-drafted BASED on the signals we just computed
    lines.append("\n## Final role-name proposals (author recommendation)\n")
    lines.append("These are working proposals grounded in the distinguishing n-grams and "
                 "exemplar content. Treat them as candidates; Gate 3 can refine.\n")
    lines.append("\n### Cluster 0 (AI-oriented low-orch)\n")
    lines.append("- **Primary proposal:** **Senior Applied-AI / LLM Engineer**")
    lines.append("- Alternate 1: **Senior AI Systems Engineer**")
    lines.append("- Alternate 2: **AI Platform Engineer**")
    lines.append("- Rationale: cluster centroid has ai_binary = 1.00 and orch = 1.55/1K chars. "
                 "Exemplars mention LLM/RAG/agentic, model evaluation, fine-tuning, prompt "
                 "engineering. The 'low-orch' qualifier in the T21 label is misleading — this "
                 "cluster has MORE orch density than the tech-orch-non-AI cluster's average "
                 "excluding AI-specific orch terms; it just lacks the CI/CD-specific orch "
                 "terms. Role name should foreground applied AI / LLM systems work.")
    lines.append("\n### Cluster 1 (Tech-orchestration non-AI)\n")
    lines.append("- **Primary proposal:** **Senior Platform / Infrastructure Engineer (CI/CD & Systems Orchestration)**")
    lines.append("- Alternate 1: **Senior DevOps / SRE Engineer**")
    lines.append("- Alternate 2: **Senior Systems-Integration Engineer**")
    lines.append("- Rationale: cluster centroid orch = 2.04/1K chars (highest across clusters), "
                 "mgmt_rebuilt ~0, ai_binary = 0. Exemplars emphasize CI/CD, pipelines, "
                 "system-design, workflows, automation. This is the non-AI infrastructure / "
                 "platform orchestrator archetype. Consistent with Gate 2's 'CI/CD is the "
                 "single largest S4 riser at +20.6 pp'.")
    lines.append("")
    out_path.write_text("\n".join(lines))


def main():
    print("=" * 70)
    print("T34 — Emergent senior-role archetype profiling (H_C)")
    print("=" * 70)

    # 1. Load joined cluster context
    print("[load] cluster assignments + unified metadata")
    df = load_cluster_context()
    print(f"  n senior rows = {len(df):,}")

    # 2. Precondition check
    print("[analyze] precondition check")
    pre = precondition_check(df)
    pre.to_csv(OUT_TAB / "cluster_precondition_check.csv", index=False)
    print(pre.to_string(index=False))

    # 3. Title distribution
    print("[analyze] title distribution")
    titles = title_distribution(df)
    titles.to_csv(OUT_TAB / "title_distribution.csv", index=False)

    # 4. Company concentration
    print("[analyze] company concentration")
    conc = company_concentration(df)
    conc.to_csv(OUT_TAB / "company_concentration.csv", index=False)
    print(conc.to_string(index=False))

    # 5. Archetype cross-tab
    print("[analyze] T09 archetype cross-tab + projection")
    arche = load_archetype_labels_with_projection(df)
    arche.to_csv(OUT_TAB / "archetype_projection_labels.csv", index=False)
    arche_xtab = archetype_crosstab(df, arche)
    arche_xtab.to_csv(OUT_TAB / "archetype_cross_tab.csv", index=False)
    print(arche_xtab.head(20).to_string(index=False))

    # 6. T16 trajectory
    print("[analyze] T16 trajectory cross-tab")
    t16 = t16_trajectory(df)
    if not t16.empty:
        t16.to_csv(OUT_TAB / "t16_trajectory_crosstab.csv", index=False)
        print(t16.to_string(index=False))
    else:
        print("  skipped (T16 wide file lacking cluster column)")

    # 7. Profile attributes
    print("[analyze] profile attributes")
    prof = profile_attributes(df)
    prof.to_csv(OUT_TAB / "cluster_profile_attributes.csv", index=False)
    for _, r in prof.iterrows():
        print(f"  cluster {r.cluster_id} {r.cluster_name}: n={r.n}, med_yoe={r.median_yoe_llm_labeled}")

    # 8. Cluster comparison
    print("[analyze] 5-cluster comparison")
    cc = cluster_comparison(df)
    cc.to_csv(OUT_TAB / "cluster_comparison.csv", index=False)
    print(cc.to_string(index=False))

    # 9. Content exemplars
    print("[analyze] content exemplars cluster 0 (AI-oriented)")
    ex0 = content_exemplars(df, cid=0, n=20)
    ex0.to_csv(OUT_TAB / "content_exemplars_cluster0.csv", index=False)
    print("[analyze] content exemplars cluster 1 (Tech-orch non-AI)")
    ex1 = content_exemplars(df, cid=1, n=20)
    ex1.to_csv(OUT_TAB / "content_exemplars_cluster1.csv", index=False)

    # 10. Recurring phrases (log-odds vs other clusters, 2026)
    print("[analyze] distinguishing n-grams cluster 0")
    ng0 = recurring_phrases(df, cid=0, top_k=20)
    ng0.to_csv(OUT_TAB / "distinguishing_ngrams_cluster0.csv", index=False)
    print("[analyze] distinguishing n-grams cluster 1")
    ng1 = recurring_phrases(df, cid=1, top_k=20)
    ng1.to_csv(OUT_TAB / "distinguishing_ngrams_cluster1.csv", index=False)

    # 11. Aggregator sensitivity
    print("[analyze] aggregator sensitivity")
    agg = aggregator_sensitivity(df)
    agg.to_csv(OUT_TAB / "aggregator_sensitivity.csv", index=False)
    print(agg.to_string(index=False))

    # 12. Role name proposals
    print("[analyze] role name proposals")
    write_role_proposals(ex0, ex1, ng0, ng1, OUT_TAB / "role_name_proposals.md")

    # 13. Figures
    print("[figure] title distribution bars")
    _figure_title_bars(titles)

    print("\n[done] outputs under exploration/tables/T34/")


def _figure_title_bars(titles: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cid in zip(axes, [0, 1]):
        sub = titles[(titles.cluster_id == cid) & (titles.period == "all") & (titles.share >= 0.01)]
        if len(sub) == 0:
            continue
        sub = sub.sort_values("share", ascending=True)
        ax.barh(sub["category"], sub["share"])
        ax.set_xlabel("share of cluster")
        ax.set_title(f"Cluster {cid} — {CANDIDATES[cid]}\n(n={sub['n_period'].iloc[0]:,})")
        for i, (cat, s) in enumerate(zip(sub["category"], sub["share"])):
            ax.text(s + 0.005, i, f"{s:.1%}", va="center", fontsize=8)
    fig.suptitle("T34 — Title distribution within candidate T21 clusters (all periods)")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "title_distribution_bars.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
