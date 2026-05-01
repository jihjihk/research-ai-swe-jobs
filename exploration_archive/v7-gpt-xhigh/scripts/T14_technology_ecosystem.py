#!/usr/bin/env python3
"""T14 technology ecosystem mapping.

Memory posture:
- uses the Wave 1.5 shared technology matrix; does not rebuild regex scans
- reads only narrow metadata columns from data/unified.parquet
- DuckDB uses a 4GB memory limit and one thread
- co-occurrence networks use a company-capped in-memory boolean matrix
"""

from __future__ import annotations

import hashlib
import math
import re
import warnings
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm, spearmanr

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "exploration" / "artifacts" / "shared"
TABLE_DIR = ROOT / "exploration" / "tables" / "T14"
FIG_DIR = ROOT / "exploration" / "figures" / "T14"

UNIFIED = ROOT / "data" / "unified.parquet"
CLEANED = SHARED / "swe_cleaned_text.parquet"
TECH_MATRIX = SHARED / "swe_tech_matrix.parquet"
TAXONOMY = SHARED / "tech_taxonomy.csv"
SKILLS = SHARED / "asaniczka_structured_skills.parquet"
T30_PANEL = SHARED / "seniority_definition_panel.csv"

COMPANY_CAP = 50
MIN_TECH_FREQ = 0.01
MIN_EDGE_PHI = 0.15
MIN_EDGE_COMPANIES = 20

JUNIOR_DEFS = {
    "J1": "seniority_final = 'entry'",
    "J2": "seniority_final IN ('entry','associate')",
    "J3": "yoe_extracted <= 2",
    "J4": "yoe_extracted <= 3",
}
SENIOR_DEFS = {
    "S1": "seniority_final IN ('mid-senior','director')",
    "S2": "seniority_final = 'director'",
    "S3": r"title_normalized senior regex",
    "S4": "yoe_extracted >= 5",
}
SENIOR_TITLE_RE = re.compile(r"\b(senior|sr\.?|staff|principal|lead|architect|distinguished)\b", re.I)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=1")
    return con


def q(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def stable_rank(value: object) -> int:
    return int(hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12], 16)


def safe_period_group(period: object) -> str:
    year = str(period)[:4]
    return "2024" if year == "2024" else "2026" if year == "2026" else year


def load_base(taxonomy: pd.DataFrame) -> pd.DataFrame:
    tech_cols = taxonomy["column"].tolist()
    con = connect()
    select_tech = ", ".join([f"t.{col}" for col in tech_cols])
    df = con.execute(
        f"""
        SELECT
          c.uid,
          c.source,
          c.period,
          c.seniority_final,
          c.seniority_3level,
          c.is_aggregator,
          c.company_name_canonical,
          c.yoe_extracted,
          c.swe_classification_tier,
          c.text_source,
          length(c.description_cleaned)::INTEGER AS char_len,
          u.title_normalized,
          {select_tech}
        FROM read_parquet('{q(CLEANED)}') c
        JOIN read_parquet('{q(TECH_MATRIX)}') t USING (uid)
        LEFT JOIN (
          SELECT uid, title_normalized
          FROM read_parquet('{q(UNIFIED)}')
        ) u USING (uid)
        """
    ).fetchdf()
    con.close()

    df["period_group"] = df["period"].map(safe_period_group)
    df["source_group"] = df["source"].map(
        {
            "kaggle_arshkon": "arshkon",
            "kaggle_asaniczka": "asaniczka",
            "scraped": "scraped_2026",
        }
    )
    df["company_name_canonical"] = df["company_name_canonical"].fillna("unknown_company")
    df["is_aggregator"] = df["is_aggregator"].fillna(False).astype(bool)
    for col in tech_cols:
        df[col] = df[col].fillna(False).astype(bool)

    df["tech_count"] = df[tech_cols].sum(axis=1).astype(np.int16)
    df["tech_density_per_1k"] = np.where(
        df["char_len"] > 0, df["tech_count"] / df["char_len"] * 1000.0, np.nan
    )
    df["company_cap_rank"] = (
        df["uid"].map(stable_rank).groupby([df["period_group"], df["company_name_canonical"]]).rank(method="first")
    )
    add_panel_flags(df)
    return df


def add_panel_flags(df: pd.DataFrame) -> None:
    seniority = df["seniority_final"].fillna("unknown")
    yoe = df["yoe_extracted"]
    title = df["title_normalized"].fillna("")
    df["J1"] = seniority.eq("entry")
    df["J2"] = seniority.isin(["entry", "associate"])
    df["J3"] = yoe.le(2).fillna(False)
    df["J4"] = yoe.le(3).fillna(False)
    df["S1"] = seniority.isin(["mid-senior", "director"])
    df["S2"] = seniority.eq("director")
    df["S3"] = title.map(lambda s: bool(SENIOR_TITLE_RE.search(s)))
    df["S4"] = yoe.ge(5).fillna(False)


def subset_spec(df: pd.DataFrame, spec: str) -> pd.DataFrame:
    out = df
    if spec == "primary_all":
        return out
    if spec == "no_aggregators":
        return out.loc[~out["is_aggregator"]].copy()
    if spec == "company_cap50":
        return out.loc[out["company_cap_rank"] <= COMPANY_CAP].copy()
    if spec == "recommended_swe_tier":
        return out.loc[out["swe_classification_tier"] != "title_lookup_llm"].copy()
    raise ValueError(spec)


def corpus_filter(df: pd.DataFrame, corpus: str) -> pd.Series:
    if corpus == "arshkon":
        return df["source"].eq("kaggle_arshkon")
    if corpus == "asaniczka":
        return df["source"].eq("kaggle_asaniczka")
    if corpus == "pooled_2024":
        return df["period_group"].eq("2024")
    if corpus == "scraped_2026":
        return df["source"].eq("scraped")
    raise ValueError(corpus)


def rate_table(df: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    tech_cols = taxonomy["column"].tolist()
    meta = taxonomy.set_index("column")
    for keys, group in df.groupby(["period_group", "seniority_3level"], dropna=False):
        period_group, seniority = keys
        n = len(group)
        if n == 0:
            continue
        sums = group[tech_cols].sum()
        for tech, mentions in sums.items():
            rows.append(
                {
                    "period_group": period_group,
                    "seniority_3level": seniority,
                    "technology": tech,
                    "label": meta.loc[tech, "label"],
                    "category": meta.loc[tech, "category"],
                    "denominator": n,
                    "mentions": int(mentions),
                    "mention_rate": float(mentions / n),
                }
            )
    out = pd.DataFrame(rows)
    out.sort_values(["period_group", "seniority_3level", "category", "technology"], inplace=True)
    return out


def category_any_table(df: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category, sub in taxonomy.groupby("category"):
        cols = sub["column"].tolist()
        flag = df[cols].any(axis=1)
        tmp = df[["period_group", "seniority_3level"]].copy()
        tmp["flag"] = flag
        for keys, group in tmp.groupby(["period_group", "seniority_3level"], dropna=False):
            rows.append(
                {
                    "category": category,
                    "period_group": keys[0],
                    "seniority_3level": keys[1],
                    "denominator": len(group),
                    "mentions": int(group["flag"].sum()),
                    "mention_rate": float(group["flag"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["category", "period_group", "seniority_3level"])


def tech_shift(df: pd.DataFrame, taxonomy: pd.DataFrame, spec_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    tech_cols = taxonomy["column"].tolist()
    meta = taxonomy.set_index("column")
    corpora = ["arshkon", "asaniczka", "pooled_2024", "scraped_2026"]
    rates: dict[tuple[str, str], tuple[int, int, float]] = {}
    for corpus in corpora:
        sub = df.loc[corpus_filter(df, corpus)]
        n = len(sub)
        sums = sub[tech_cols].sum() if n else pd.Series(0, index=tech_cols)
        for tech in tech_cols:
            mentions = int(sums[tech])
            rates[(tech, corpus)] = (n, mentions, mentions / n if n else np.nan)

    for tech in tech_cols:
        ar = rates[(tech, "arshkon")][2]
        asa = rates[(tech, "asaniczka")][2]
        pooled = rates[(tech, "pooled_2024")][2]
        scraped = rates[(tech, "scraped_2026")][2]
        change = scraped - pooled
        ar_change = scraped - ar
        within = ar - asa
        within_abs = abs(within)
        snr = abs(ar_change) / within_abs if within_abs > 1e-9 else np.inf if abs(ar_change) > 0 else np.nan
        if change >= 0.02 and (within_abs < 0.005 or abs(change) >= 2 * within_abs):
            trajectory = "calibrated_rising"
        elif change <= -0.02 and (within_abs < 0.005 or abs(change) >= 2 * within_abs):
            trajectory = "calibrated_declining"
        elif change >= 0.02:
            trajectory = "rising_not_above_2024_noise"
        elif change <= -0.02:
            trajectory = "declining_not_above_2024_noise"
        else:
            trajectory = "stable_or_small_change"
        rows.append(
            {
                "spec": spec_name,
                "technology": tech,
                "label": meta.loc[tech, "label"],
                "category": meta.loc[tech, "category"],
                "arshkon_n": rates[(tech, "arshkon")][0],
                "asaniczka_n": rates[(tech, "asaniczka")][0],
                "pooled_2024_n": rates[(tech, "pooled_2024")][0],
                "scraped_2026_n": rates[(tech, "scraped_2026")][0],
                "arshkon_rate": ar,
                "asaniczka_rate": asa,
                "pooled_2024_rate": pooled,
                "scraped_2026_rate": scraped,
                "pooled_to_scraped_pp": change * 100,
                "arshkon_to_scraped_pp": ar_change * 100,
                "within_2024_arshkon_minus_asaniczka_pp": within * 100,
                "calibration_snr_arshkon_to_scraped": snr,
                "trajectory": trajectory,
            }
        )
    out = pd.DataFrame(rows)
    out.sort_values(["spec", "pooled_to_scraped_pp"], ascending=[True, False], inplace=True)
    return out


def breadth_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for keys, group in df.groupby(["period_group", "seniority_3level"], dropna=False):
        rows.append(
            {
                "period_group": keys[0],
                "seniority_3level": keys[1],
                "n": len(group),
                "mean_tech_count": group["tech_count"].mean(),
                "median_tech_count": group["tech_count"].median(),
                "mean_tech_density_per_1k": group["tech_density_per_1k"].mean(),
                "median_char_len": group["char_len"].median(),
            }
        )
    seniority = pd.DataFrame(rows).sort_values(["period_group", "seniority_3level"])

    panel_rows = []
    for definition in [*JUNIOR_DEFS, *SENIOR_DEFS]:
        for corpus in ["pooled_2024", "scraped_2026", "arshkon", "asaniczka"]:
            sub = df.loc[corpus_filter(df, corpus) & df[definition]]
            panel_rows.append(
                {
                    "definition": definition,
                    "side": "junior" if definition.startswith("J") else "senior",
                    "definition_label": {**JUNIOR_DEFS, **SENIOR_DEFS}[definition],
                    "corpus": corpus,
                    "n": len(sub),
                    "mean_tech_count": sub["tech_count"].mean() if len(sub) else np.nan,
                    "median_tech_count": sub["tech_count"].median() if len(sub) else np.nan,
                    "mean_tech_density_per_1k": sub["tech_density_per_1k"].mean() if len(sub) else np.nan,
                    "median_char_len": sub["char_len"].median() if len(sub) else np.nan,
                    "yoe_known_share": sub["yoe_extracted"].notna().mean() if len(sub) else np.nan,
                }
            )
    panel = pd.DataFrame(panel_rows)
    return seniority, panel


def ai_integration_tables(df: pd.DataFrame, taxonomy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ai_tool_cols = taxonomy.loc[taxonomy["category"].eq("ai_tool"), "column"].tolist()
    llm_stack_extra = [
        col
        for col in [
            "langchain",
            "llamaindex",
            "rag",
            "vector_databases",
            "pinecone",
            "weaviate",
            "chroma",
            "hugging_face",
            "generative_ai",
            "llm",
            "agents",
        ]
        if col in taxonomy["column"].tolist()
    ]
    ai_cols = sorted(set(ai_tool_cols + llm_stack_extra))
    traditional_cols = taxonomy.loc[~taxonomy["column"].isin(ai_cols), "column"].tolist()
    meta = taxonomy.set_index("column")
    work = df.copy()
    work["ai_tool_or_llm_mention"] = work[ai_cols].any(axis=1)
    work["non_ai_tech_count"] = work[traditional_cols].sum(axis=1)
    work["non_ai_tech_density_per_1k"] = np.where(
        work["char_len"] > 0, work["non_ai_tech_count"] / work["char_len"] * 1000, np.nan
    )

    summary_rows = []
    for keys, group in work.groupby(["period_group", "ai_tool_or_llm_mention"], dropna=False):
        summary_rows.append(
            {
                "period_group": keys[0],
                "ai_tool_or_llm_mention": bool(keys[1]),
                "n": len(group),
                "mean_tech_count": group["tech_count"].mean(),
                "median_tech_count": group["tech_count"].median(),
                "mean_non_ai_tech_count": group["non_ai_tech_count"].mean(),
                "median_non_ai_tech_count": group["non_ai_tech_count"].median(),
                "mean_tech_density_per_1k": group["tech_density_per_1k"].mean(),
                "mean_non_ai_tech_density_per_1k": group["non_ai_tech_density_per_1k"].mean(),
                "median_char_len": group["char_len"].median(),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["period_group", "ai_tool_or_llm_mention"])

    co_rows = []
    for period in ["2024", "2026"]:
        sub = work.loc[work["period_group"].eq(period)]
        ai = sub.loc[sub["ai_tool_or_llm_mention"]]
        non = sub.loc[~sub["ai_tool_or_llm_mention"]]
        for tech in traditional_cols:
            ai_rate = ai[tech].mean() if len(ai) else np.nan
            non_rate = non[tech].mean() if len(non) else np.nan
            co_rows.append(
                {
                    "period_group": period,
                    "technology": tech,
                    "label": meta.loc[tech, "label"],
                    "category": meta.loc[tech, "category"],
                    "ai_rows": len(ai),
                    "non_ai_rows": len(non),
                    "rate_among_ai_tool_or_llm": ai_rate,
                    "rate_among_non_ai": non_rate,
                    "lift_pp": (ai_rate - non_rate) * 100 if pd.notna(ai_rate) and pd.notna(non_rate) else np.nan,
                    "lift_ratio": ai_rate / non_rate if pd.notna(ai_rate) and non_rate > 0 else np.nan,
                }
            )
    co = pd.DataFrame(co_rows).sort_values(["period_group", "lift_pp"], ascending=[True, False])
    return summary, co


def phi_edges(df: pd.DataFrame, taxonomy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cap = df.loc[df["company_cap_rank"] <= COMPANY_CAP].copy()
    tech_cols = taxonomy["column"].tolist()
    meta = taxonomy.set_index("column")
    all_rates = cap.groupby("period_group")[tech_cols].mean()
    sufficient = all_rates.max(axis=0)
    selected_cols = sufficient.loc[sufficient >= MIN_TECH_FREQ].index.tolist()
    node_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []

    for period in ["2024", "2026"]:
        sub = cap.loc[cap["period_group"].eq(period)].reset_index(drop=True)
        if sub.empty:
            continue
        mat = sub[selected_cols].to_numpy(dtype=np.bool_)
        n = mat.shape[0]
        companies = sub["company_name_canonical"].to_numpy()
        sums = mat.sum(axis=0).astype(float)
        for idx, tech in enumerate(selected_cols):
            node_rows.append(
                {
                    "period_group": period,
                    "technology": tech,
                    "label": meta.loc[tech, "label"],
                    "category": meta.loc[tech, "category"],
                    "mention_rate_cap50": sums[idx] / n,
                    "mentions_cap50": int(sums[idx]),
                    "community": np.nan,
                }
            )
        for i in range(len(selected_cols)):
            xi = mat[:, i]
            a = sums[i]
            if a == 0 or a == n:
                continue
            for j in range(i + 1, len(selected_cols)):
                xj = mat[:, j]
                b = sums[j]
                if b == 0 or b == n:
                    continue
                both_mask = xi & xj
                c = float(both_mask.sum())
                if c == 0:
                    continue
                denom = math.sqrt(a * b * (n - a) * (n - b))
                if denom == 0:
                    continue
                phi = (n * c - a * b) / denom
                if phi < MIN_EDGE_PHI:
                    continue
                co_companies = int(pd.Series(companies[both_mask]).nunique())
                if co_companies < MIN_EDGE_COMPANIES:
                    continue
                edge_rows.append(
                    {
                        "period_group": period,
                        "source": selected_cols[i],
                        "target": selected_cols[j],
                        "source_label": meta.loc[selected_cols[i], "label"],
                        "target_label": meta.loc[selected_cols[j], "label"],
                        "phi": phi,
                        "co_mentions_cap50": int(c),
                        "co_mention_companies": co_companies,
                        "n_cap50": n,
                    }
                )

    edges = pd.DataFrame(edge_rows)
    nodes = pd.DataFrame(node_rows)
    if edges.empty:
        return nodes, edges

    community_rows = []
    for period in ["2024", "2026"]:
        sub_edges = edges.loc[edges["period_group"].eq(period)]
        graph = nx.Graph()
        for _, row in nodes.loc[nodes["period_group"].eq(period)].iterrows():
            graph.add_node(row["technology"])
        for _, row in sub_edges.iterrows():
            graph.add_edge(row["source"], row["target"], weight=row["phi"])
        graph.remove_nodes_from(list(nx.isolates(graph)))
        if graph.number_of_nodes() == 0:
            continue
        communities = nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
        for cid, community in enumerate(communities, start=1):
            for tech in sorted(community):
                community_rows.append({"period_group": period, "technology": tech, "community": cid})
    community = pd.DataFrame(community_rows)
    if not community.empty:
        nodes = nodes.drop(columns=["community"]).merge(community, on=["period_group", "technology"], how="left")
    return nodes, edges


def community_comparison(nodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if nodes.empty:
        return pd.DataFrame(), pd.DataFrame()
    pivot = nodes.pivot_table(
        index=["technology", "label", "category"],
        columns="period_group",
        values=["community", "mention_rate_cap50"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{period}" for metric, period in pivot.columns]
    pivot = pivot.reset_index()
    for col in ["community_2024", "community_2026", "mention_rate_cap50_2024", "mention_rate_cap50_2026"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["rate_change_pp_cap50"] = (
        pivot["mention_rate_cap50_2026"].fillna(0) - pivot["mention_rate_cap50_2024"].fillna(0)
    ) * 100
    pivot["community_status"] = np.select(
        [
            pivot["community_2024"].isna() & pivot["community_2026"].notna(),
            pivot["community_2024"].notna() & pivot["community_2026"].isna(),
            pivot["community_2024"].notna()
            & pivot["community_2026"].notna()
            & pivot["community_2024"].eq(pivot["community_2026"]),
            pivot["community_2024"].notna() & pivot["community_2026"].notna(),
        ],
        ["network_entry", "network_exit", "same_numeric_id", "reclustered"],
        default="outside_network",
    )
    pivot.sort_values("rate_change_pp_cap50", ascending=False, inplace=True)

    summary_rows = []
    for (period, community), group in nodes.dropna(subset=["community"]).groupby(["period_group", "community"]):
        group = group.sort_values("mention_rate_cap50", ascending=False)
        summary_rows.append(
            {
                "period_group": period,
                "community": int(community),
                "nodes": len(group),
                "top_labels": "; ".join(group["label"].head(8).tolist()),
                "top_categories": "; ".join(group["category"].value_counts().head(4).index.tolist()),
                "max_mention_rate_cap50": group["mention_rate_cap50"].max(),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["period_group", "nodes"], ascending=[True, False])
    return pivot, summary


def structured_skill_tables(df: pd.DataFrame, taxonomy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = connect()
    top_skills = con.execute(
        f"""
        SELECT skill_clean, min(skill_raw) AS example_skill_raw,
               count(*) AS skill_rows, count(DISTINCT uid) AS postings
        FROM read_parquet('{q(SKILLS)}')
        WHERE skill_clean IS NOT NULL AND trim(skill_clean) <> ''
        GROUP BY skill_clean
        ORDER BY postings DESC, skill_clean
        LIMIT 100
        """
    ).fetchdf()
    skills = con.execute(
        f"""
        SELECT DISTINCT uid, lower(skill_clean) AS skill_clean
        FROM read_parquet('{q(SKILLS)}')
        WHERE skill_clean IS NOT NULL AND trim(skill_clean) <> ''
        """
    ).fetchdf()
    con.close()

    asaniczka = df.loc[df["source"].eq("kaggle_asaniczka")].copy()
    asaniczka_uids = set(asaniczka["uid"])
    skills = skills.loc[skills["uid"].isin(asaniczka_uids)].copy()
    denom = asaniczka["uid"].nunique()

    validation_rows = []
    for _, row in taxonomy.iterrows():
        tech = row["column"]
        pattern = re.compile(row["regex"], re.I)
        mask = skills["skill_clean"].map(lambda s: bool(pattern.search(s)))
        structured_uids = skills.loc[mask, "uid"].nunique()
        extracted_mentions = int(asaniczka[tech].sum())
        validation_rows.append(
            {
                "technology": tech,
                "label": row["label"],
                "category": row["category"],
                "structured_postings": structured_uids,
                "structured_rate": structured_uids / denom if denom else np.nan,
                "extracted_postings": extracted_mentions,
                "extracted_rate": extracted_mentions / denom if denom else np.nan,
                "rate_diff_structured_minus_extracted_pp": (
                    structured_uids / denom - extracted_mentions / denom
                )
                * 100
                if denom
                else np.nan,
            }
        )
    validation = pd.DataFrame(validation_rows)
    validation["structured_rank"] = validation["structured_rate"].rank(ascending=False, method="average")
    validation["extracted_rank"] = validation["extracted_rate"].rank(ascending=False, method="average")
    rho, p = spearmanr(validation["structured_rate"], validation["extracted_rate"], nan_policy="omit")
    validation["spearman_rho_all_techs"] = rho
    validation["spearman_p_value"] = p
    validation.sort_values("rate_diff_structured_minus_extracted_pp", ascending=False, inplace=True)

    assoc = structured_skill_associations(asaniczka, skills)
    return top_skills, validation, assoc


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    ranked = pvals[order]
    n = len(pvals)
    adj = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * n / rank)
        adj[order[i]] = running
    return np.minimum(adj, 1.0)


def two_prop_p(count_a: int, n_a: int, count_b: int, n_b: int) -> float:
    if n_a == 0 or n_b == 0:
        return np.nan
    p_pool = (count_a + count_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 1.0
    z = (count_a / n_a - count_b / n_b) / se
    return float(2 * norm.sf(abs(z)))


def structured_skill_associations(asaniczka: pd.DataFrame, skills: pd.DataFrame) -> pd.DataFrame:
    uid_skills = skills.drop_duplicates(["uid", "skill_clean"])
    posting_counts = uid_skills.groupby("skill_clean")["uid"].nunique()
    keep = posting_counts.loc[posting_counts >= 50].index
    uid_skills = uid_skills.loc[uid_skills["skill_clean"].isin(keep)].copy()
    skill_to_uids = uid_skills.groupby("skill_clean")["uid"].apply(set).to_dict()

    comparisons = [
        ("J1_vs_S1", "J1", "S1"),
        ("J3_vs_S4", "J3", "S4"),
    ]
    rows = []
    for comparison, left_def, right_def in comparisons:
        left_uids = set(asaniczka.loc[asaniczka[left_def], "uid"])
        right_uids = set(asaniczka.loc[asaniczka[right_def], "uid"])
        n_left = len(left_uids)
        n_right = len(right_uids)
        pvals = []
        base_rows = []
        for skill, skill_uids in skill_to_uids.items():
            left_count = len(left_uids & skill_uids)
            right_count = len(right_uids & skill_uids)
            pval = two_prop_p(left_count, n_left, right_count, n_right)
            pvals.append(pval)
            left_rate = left_count / n_left if n_left else np.nan
            right_rate = right_count / n_right if n_right else np.nan
            base_rows.append(
                {
                    "comparison": comparison,
                    "left_definition": left_def,
                    "right_definition": right_def,
                    "skill_clean": skill,
                    "left_n": n_left,
                    "right_n": n_right,
                    "left_postings": left_count,
                    "right_postings": right_count,
                    "left_rate": left_rate,
                    "right_rate": right_rate,
                    "diff_left_minus_right_pp": (left_rate - right_rate) * 100
                    if pd.notna(left_rate) and pd.notna(right_rate)
                    else np.nan,
                    "p_value": pval,
                }
            )
        adj = bh_fdr(np.asarray(pvals))
        for row, fdr in zip(base_rows, adj, strict=True):
            row["fdr_bh"] = fdr
            row["association_direction"] = (
                "left_higher"
                if row["diff_left_minus_right_pp"] > 0
                else "right_higher"
                if row["diff_left_minus_right_pp"] < 0
                else "flat"
            )
            rows.append(row)
    out = pd.DataFrame(rows)
    out.sort_values(["comparison", "fdr_bh", "diff_left_minus_right_pp"], ascending=[True, True, False], inplace=True)
    return out


def plot_shift_heatmap(shift: pd.DataFrame) -> None:
    primary = shift.loc[shift["spec"].eq("primary_all")].copy()
    primary["abs_change"] = primary["pooled_to_scraped_pp"].abs()
    top = primary.sort_values("abs_change", ascending=False).head(28)
    mat = top.set_index("label")[["arshkon_rate", "asaniczka_rate", "scraped_2026_rate"]] * 100
    fig, ax = plt.subplots(figsize=(8.5, 9.2))
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(mat.index)), labels=mat.index, fontsize=8)
    ax.set_xticks(np.arange(3), labels=["arshkon 2024", "asaniczka 2024", "scraped 2026"], rotation=20, ha="right")
    ax.set_title("Technology mention rates for largest absolute movers")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mention rate (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tech_shift_heatmap.png", dpi=150)
    plt.close(fig)


def plot_networks(nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    for ax, period in zip(axes, ["2024", "2026"], strict=True):
        sub_edges = edges.loc[edges["period_group"].eq(period)].copy()
        sub_nodes = nodes.loc[nodes["period_group"].eq(period)].copy()
        graph = nx.Graph()
        for _, row in sub_nodes.iterrows():
            graph.add_node(
                row["technology"],
                label=row["label"],
                rate=row["mention_rate_cap50"],
                community=row["community"],
            )
        for _, row in sub_edges.iterrows():
            graph.add_edge(row["source"], row["target"], weight=row["phi"])
        graph.remove_nodes_from(list(nx.isolates(graph)))
        ax.set_title(f"{period} co-occurrence network")
        ax.axis("off")
        if graph.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "No edges after thresholds", ha="center", va="center")
            continue
        pos = nx.spring_layout(graph, seed=42, weight="weight", k=0.6)
        comm_values = [graph.nodes[n].get("community", 0) or 0 for n in graph.nodes]
        sizes = [80 + graph.nodes[n].get("rate", 0) * 900 for n in graph.nodes]
        widths = [0.4 + graph[u][v]["weight"] * 3 for u, v in graph.edges]
        nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.25, width=widths, edge_color="#555555")
        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_size=sizes,
            node_color=comm_values,
            cmap="tab20",
            alpha=0.9,
            linewidths=0.3,
            edgecolors="#222222",
        )
        labels = {n: graph.nodes[n]["label"] for n in graph.nodes if graph.nodes[n].get("rate", 0) >= 0.03}
        nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cooccurrence_networks.png", dpi=150)
    plt.close(fig)


def plot_breadth_panel(panel: pd.DataFrame) -> None:
    keep = panel.loc[panel["corpus"].isin(["pooled_2024", "scraped_2026"])].copy()
    keep = keep.loc[keep["definition"].isin(["J1", "J2", "J3", "J4", "S1", "S2", "S3", "S4"])]
    pivot = keep.pivot(index="definition", columns="corpus", values="mean_tech_count").loc[
        ["J1", "J2", "J3", "J4", "S1", "S2", "S3", "S4"]
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(pivot.index))
    width = 0.36
    ax.bar(x - width / 2, pivot["pooled_2024"], width, label="pooled 2024", color="#4C78A8")
    ax.bar(x + width / 2, pivot["scraped_2026"], width, label="scraped 2026", color="#F58518")
    ax.set_xticks(x, labels=pivot.index)
    ax.set_ylabel("Mean distinct technology indicators")
    ax.set_title("Technology breadth by T30 seniority definition")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tech_breadth_seniority_panel.png", dpi=150)
    plt.close(fig)


def plot_ai_integration(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, metric, title in [
        (axes[0], "mean_non_ai_tech_count", "Non-AI tech count"),
        (axes[1], "mean_non_ai_tech_density_per_1k", "Non-AI tech density"),
    ]:
        pivot = summary.pivot(index="period_group", columns="ai_tool_or_llm_mention", values=metric).loc[["2024", "2026"]]
        x = np.arange(len(pivot.index))
        ax.bar(x - 0.18, pivot[False], 0.36, label="No AI-tool/LLM", color="#54A24B")
        ax.bar(x + 0.18, pivot[True], 0.36, label="AI-tool/LLM mention", color="#E45756")
        ax.set_xticks(x, labels=pivot.index)
        ax.set_title(title)
    axes[0].set_ylabel("Mean per posting")
    axes[1].set_ylabel("Mean per 1K cleaned chars")
    axes[1].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ai_integration_stack.png", dpi=150)
    plt.close(fig)


def write_summary_tables(df: pd.DataFrame, taxonomy: pd.DataFrame) -> dict[str, float]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    pd.read_csv(T30_PANEL).to_csv(TABLE_DIR / "t30_panel_loaded_for_reference.csv", index=False)

    rates = rate_table(df, taxonomy)
    rates.to_csv(TABLE_DIR / "mention_rates_by_period_seniority.csv", index=False)

    category_any = category_any_table(df, taxonomy)
    category_any.to_csv(TABLE_DIR / "category_any_rates_by_period_seniority.csv", index=False)

    sensitivity_parts = []
    for spec in ["primary_all", "no_aggregators", "company_cap50", "recommended_swe_tier"]:
        sensitivity_parts.append(tech_shift(subset_spec(df, spec), taxonomy, spec))
    shift = pd.concat(sensitivity_parts, ignore_index=True)
    shift.to_csv(TABLE_DIR / "technology_shift_classification.csv", index=False)

    breadth, panel = breadth_tables(df)
    breadth.to_csv(TABLE_DIR / "stack_diversity_by_period_seniority.csv", index=False)
    panel.to_csv(TABLE_DIR / "tech_breadth_t30_panel.csv", index=False)

    ai_summary, ai_co = ai_integration_tables(df, taxonomy)
    ai_summary.to_csv(TABLE_DIR / "ai_integration_length_normalization.csv", index=False)
    ai_co.to_csv(TABLE_DIR / "ai_integration_traditional_cooccurrence.csv", index=False)

    nodes, edges = phi_edges(df, taxonomy)
    nodes.to_csv(TABLE_DIR / "cooccurrence_network_nodes.csv", index=False)
    edges.to_csv(TABLE_DIR / "cooccurrence_network_edges.csv", index=False)
    community = nodes.loc[nodes["community"].notna(), ["period_group", "technology", "label", "category", "community"]]
    community.to_csv(TABLE_DIR / "community_membership.csv", index=False)
    community_movement, community_summary = community_comparison(nodes)
    community_movement.to_csv(TABLE_DIR / "community_comparison.csv", index=False)
    community_summary.to_csv(TABLE_DIR / "community_summary.csv", index=False)
    if not edges.empty:
        plot_networks(nodes, edges)

    top_skills, validation, assoc = structured_skill_tables(df, taxonomy)
    top_skills.to_csv(TABLE_DIR / "asaniczka_structured_skills_top100.csv", index=False)
    validation.to_csv(TABLE_DIR / "structured_vs_extracted_validation.csv", index=False)
    assoc.to_csv(TABLE_DIR / "structured_skill_seniority_associations.csv", index=False)

    plot_shift_heatmap(shift)
    plot_breadth_panel(panel)
    plot_ai_integration(ai_summary)

    primary_shift = shift.loc[shift["spec"].eq("primary_all")]
    calibrated_risers = int(primary_shift["trajectory"].eq("calibrated_rising").sum())
    calibrated_decliners = int(primary_shift["trajectory"].eq("calibrated_declining").sum())
    rho = float(validation["spearman_rho_all_techs"].dropna().iloc[0])
    return {
        "calibrated_risers": calibrated_risers,
        "calibrated_decliners": calibrated_decliners,
        "structured_spearman_rho": rho,
        "network_edges_2024": int((edges["period_group"] == "2024").sum()) if not edges.empty else 0,
        "network_edges_2026": int((edges["period_group"] == "2026").sum()) if not edges.empty else 0,
    }


def main() -> None:
    taxonomy = pd.read_csv(TAXONOMY)
    df = load_base(taxonomy)
    summary = write_summary_tables(df, taxonomy)
    print("T14 complete")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
