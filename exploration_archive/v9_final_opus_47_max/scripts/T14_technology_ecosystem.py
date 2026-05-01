"""T14 — Technology ecosystem mapping.

Outputs:
  exploration/tables/T14/tech_rates_by_period_seniority.csv   (primary)
  exploration/tables/T14/phi_matrix_pooled.csv                (primary, 135x135)
  exploration/tables/T14/phi_matrix_pooled_long.csv           (long form for T35)
  exploration/tables/T14/top20_phi_pairs.csv
  exploration/tables/T14/tech_trajectory.csv                  (rising/stable/declining w/ calibration)
  exploration/tables/T14/stack_diversity.csv
  exploration/tables/T14/ai_co_occurrence.csv                 (raw rate + density)
  exploration/tables/T14/asaniczka_top100_structured.csv
  exploration/tables/T14/structured_vs_extracted.csv          (validation)
  exploration/tables/T14/seniority_skill_structured_chi2.csv
  exploration/tables/T14/phi_per_period_long.csv              (T35 prep)
  exploration/tables/T14/ml_tech_source_stratified.csv        (ML-engineer caveat)
  exploration/tables/T14/tech_rates_sensitivities.csv         (aggregator+capping+SWEtier)
  exploration/figures/T14_tech_trajectory.png
  exploration/figures/T14_stack_diversity.png
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/home/jihgaboot/gabor/job-research")
SHARED = ROOT / "exploration/artifacts/shared"
OUT_TBL = ROOT / "exploration/tables/T14"
OUT_FIG = ROOT / "exploration/figures"
OUT_TBL.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

UNIFIED = ROOT / "data/unified.parquet"
TECH_PATH = SHARED / "swe_tech_matrix.parquet"
SANITY_PATH = SHARED / "tech_matrix_sanity.csv"
ASANICZKA_SKILLS = SHARED / "asaniczka_structured_skills.parquet"

FILTER = "is_swe=TRUE AND source_platform='linkedin' AND is_english=TRUE AND date_flag='ok'"


def load_core_frame(con):
    """Load keys + seniority/period/company cols joined with tech matrix."""
    q = f"""
    WITH u AS (
      SELECT uid, source, period, company_name_canonical, is_aggregator,
             llm_classification_coverage,
             yoe_min_years_llm,
             seniority_final, seniority_3level,
             swe_classification_tier,
             title, title_normalized,
             description, description_length,
             description_core_llm, llm_extraction_coverage
      FROM '{UNIFIED}'
      WHERE {FILTER}
    ), t AS (
      SELECT * FROM '{TECH_PATH}'
    )
    SELECT u.*, t.* EXCLUDE (uid)
    FROM u JOIN t USING (uid)
    """
    df = con.execute(q).df()
    return df


TECH_COLS_CACHE: list[str] | None = None


def tech_cols(df):
    global TECH_COLS_CACHE
    if TECH_COLS_CACHE is None:
        non_tech = {
            "uid", "source", "period", "company_name_canonical", "is_aggregator",
            "llm_classification_coverage", "yoe_min_years_llm",
            "seniority_final", "seniority_3level", "swe_classification_tier",
            "title", "title_normalized",
            "description", "description_length",
            "description_core_llm", "llm_extraction_coverage",
        }
        TECH_COLS_CACHE = [c for c in df.columns if c not in non_tech]
    return TECH_COLS_CACHE


def add_strict_ai_flag(df):
    """Strict AI flag on raw description, as per T14 step 6."""
    import re
    pat = re.compile(
        r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|"
        r"llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|"
        r"vector database|pinecone|huggingface|hugging face)\b",
        re.IGNORECASE,
    )
    df = df.copy()
    s = df["description"].fillna("").astype(str)
    df["ai_strict"] = s.str.contains(pat)
    return df


def seniority_bin(df):
    """Add J3 / S4 / J3_rule-proxy labels. Primary = J3 / S4 (YOE via LLM)."""
    df = df.copy()
    df["is_j3"] = (df["yoe_min_years_llm"].notna() & (df["yoe_min_years_llm"] <= 2)) & (
        df["llm_classification_coverage"] == "labeled"
    )
    df["is_s4"] = (df["yoe_min_years_llm"].notna() & (df["yoe_min_years_llm"] >= 5)) & (
        df["llm_classification_coverage"] == "labeled"
    )
    # Label-based sensitivities. These are imprecise but align with T30:
    # J1 = seniority_final == 'entry'
    # S1 = seniority_final in ('senior','staff','principal','director')
    # S2 = seniority_final == 'director'
    df["is_j1"] = df["seniority_final"] == "entry"
    df["is_s1"] = df["seniority_final"].isin(["senior", "staff", "principal", "director"])
    df["is_s2"] = df["seniority_final"] == "director"

    # 2-period collapse for cross-period comparisons
    df["period2"] = df["period"].apply(
        lambda p: "2024" if isinstance(p, str) and p.startswith("2024")
        else ("2026" if isinstance(p, str) and p.startswith("2026") else None)
    )
    return df


def company_cap(df, cap=50, col="company_name_canonical"):
    """Stratified cap: within each (period2, source), keep up to `cap` per company."""
    if col not in df.columns:
        return df
    keep = (
        df.assign(_rn=df.groupby(["period2", "source", col]).cumcount())
          .query("_rn < @cap")
          .drop(columns=["_rn"])
    )
    return keep


# ---------------------------------------------------------------------------
# Step 2: tech mention rates by period x seniority (primary J3, S4)
# ---------------------------------------------------------------------------

def tech_rates_by_group(df, techs):
    rows = []
    for (p2, seniority), g in [
        (("2024", "J3"), df[(df.period2 == "2024") & df.is_j3]),
        (("2026", "J3"), df[(df.period2 == "2026") & df.is_j3]),
        (("2024", "S4"), df[(df.period2 == "2024") & df.is_s4]),
        (("2026", "S4"), df[(df.period2 == "2026") & df.is_s4]),
        (("2024", "all_swe"), df[df.period2 == "2024"]),
        (("2026", "all_swe"), df[df.period2 == "2026"]),
    ]:
        n = len(g)
        if n == 0:
            continue
        rates = g[techs].astype(bool).mean()
        counts = g[techs].astype(bool).sum()
        for tech in techs:
            rows.append({
                "period": p2, "seniority": seniority, "n": n,
                "technology": tech, "rate": float(rates[tech]),
                "mentions": int(counts[tech]),
            })
    return pd.DataFrame(rows)


def tech_rates_by_source_2024(df, techs):
    """2024-only per-source rates for within-2024 calibration."""
    rows = []
    sub = df[df.period2 == "2024"]
    for src in ["kaggle_arshkon", "kaggle_asaniczka"]:
        g = sub[sub.source == src]
        n = len(g)
        if n == 0:
            continue
        rates = g[techs].astype(bool).mean()
        for tech in techs:
            rows.append({"period": "2024", "source": src, "n": n,
                         "technology": tech, "rate": float(rates[tech])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3: phi coefficient co-occurrence, pooled
# ---------------------------------------------------------------------------

def phi_matrix(bool_matrix: np.ndarray) -> np.ndarray:
    """Compute phi (pearson correlation of binary variables) matrix.

    phi_AB = (n11 * n00 - n10 * n01) / sqrt((n11+n10)(n01+n00)(n11+n01)(n10+n00))

    Equivalent to np.corrcoef on a 0/1 float matrix. Handle zero-variance cols.
    """
    X = bool_matrix.astype(np.float32)
    var = X.var(axis=0)
    good = var > 0
    phi = np.full((X.shape[1], X.shape[1]), np.nan, dtype=np.float32)
    Xg = X[:, good]
    C = np.corrcoef(Xg.T)
    idx = np.where(good)[0]
    for i, gi in enumerate(idx):
        for j, gj in enumerate(idx):
            phi[gi, gj] = C[i, j]
    return phi


def write_phi(df, techs, out_pooled_csv, out_long_csv, top20_csv, label="pooled"):
    X = df[techs].astype(bool).values
    phi = phi_matrix(X)
    pdf = pd.DataFrame(phi, index=techs, columns=techs)
    pdf.to_csv(out_pooled_csv)

    # long form for top-N and downstream T35
    tri = np.triu_indices(len(techs), k=1)
    long_rows = []
    for i, j in zip(*tri):
        val = phi[i, j]
        if np.isnan(val):
            continue
        long_rows.append({
            "a": techs[i], "b": techs[j], "phi": float(val),
            "label": label,
        })
    ldf = pd.DataFrame(long_rows).sort_values("phi", ascending=False)
    ldf.to_csv(out_long_csv, index=False)

    # Top 20 pairs w/ supporting mention counts
    top = ldf.head(20).copy()
    N = len(df)
    for col in ["n_a", "n_b", "n_both"]:
        top[col] = 0
    for k, row in top.iterrows():
        a, b = row["a"], row["b"]
        top.loc[k, "n_a"] = int(df[a].astype(bool).sum())
        top.loc[k, "n_b"] = int(df[b].astype(bool).sum())
        top.loc[k, "n_both"] = int((df[a].astype(bool) & df[b].astype(bool)).sum())
    top["n_total"] = N
    top.to_csv(top20_csv, index=False)


# ---------------------------------------------------------------------------
# Step 4: trajectory classification (with within-2024 calibration)
# ---------------------------------------------------------------------------

def trajectory_classify(rates_all, rates_2024_src, min_rate=0.005, rise_factor=1.5):
    """Classify each tech as rising / stable / declining based on the gap between
    2024 -> 2026 cross-period effect and within-2024 arshkon-vs-asaniczka noise.
    """
    # pivot the all-SWE rates to wide
    wide = rates_all[rates_all.seniority == "all_swe"].pivot(
        index="technology", columns="period", values="rate"
    ).reset_index()
    wide.columns.name = None
    # pivot 2024 within
    wide24 = rates_2024_src.pivot(
        index="technology", columns="source", values="rate"
    ).reset_index()
    wide24.columns.name = None
    out = wide.merge(wide24, on="technology", how="left")
    out = out.rename(columns={"2024": "rate_2024_pooled", "2026": "rate_2026_scraped"})
    out["rate_2024_arshkon"] = out.get("kaggle_arshkon", pd.Series([np.nan] * len(out)))
    out["rate_2024_asaniczka"] = out.get("kaggle_asaniczka", pd.Series([np.nan] * len(out)))
    out["delta"] = out["rate_2026_scraped"] - out["rate_2024_pooled"]
    out["within_2024_gap"] = (out["rate_2024_arshkon"] - out["rate_2024_asaniczka"]).abs()
    out["snr"] = out["delta"].abs() / out["within_2024_gap"].replace(0, np.nan)
    out["snr"] = out["snr"].fillna(np.inf)

    def classify(r):
        d = r.get("delta", 0)
        sn = r.get("snr", 0)
        # absolute rise/decline thresholds: require >=0.003 absolute AND SNR>=2
        if abs(d) < 0.003 and max(r.get("rate_2024_pooled", 0), r.get("rate_2026_scraped", 0)) < min_rate:
            return "trace"
        if sn < 2:
            return "calibration_noise"
        if d > 0.005:
            return "rising"
        if d < -0.005:
            return "declining"
        return "stable"

    out["class"] = out.apply(classify, axis=1)
    out = out.sort_values("delta", ascending=False)
    return out[["technology", "rate_2024_arshkon", "rate_2024_asaniczka",
                "rate_2024_pooled", "rate_2026_scraped",
                "delta", "within_2024_gap", "snr", "class"]]


# ---------------------------------------------------------------------------
# Step 5: stack diversity
# ---------------------------------------------------------------------------

def stack_diversity(df, techs):
    """Count of distinct tech mentions per posting by period x seniority."""
    df = df.copy()
    df["n_techs"] = df[techs].astype(bool).sum(axis=1)
    rows = []
    for (p2, sen), g in [
        (("2024", "J3"), df[(df.period2 == "2024") & df.is_j3]),
        (("2026", "J3"), df[(df.period2 == "2026") & df.is_j3]),
        (("2024", "S4"), df[(df.period2 == "2024") & df.is_s4]),
        (("2026", "S4"), df[(df.period2 == "2026") & df.is_s4]),
        (("2024", "all_swe"), df[df.period2 == "2024"]),
        (("2026", "all_swe"), df[df.period2 == "2026"]),
    ]:
        if not len(g):
            continue
        s = g["n_techs"]
        rows.append({
            "period": p2, "seniority": sen, "n": len(g),
            "median": float(s.median()), "mean": float(s.mean()),
            "q25": float(s.quantile(0.25)), "q75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)), "pct_0": float((s == 0).mean()),
            "pct_ge10": float((s >= 10).mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 6: AI co-occurrence (strict)
# ---------------------------------------------------------------------------

def ai_co_occurrence(df, techs):
    """Among ai_strict postings, what techs co-occur? Report raw rate AND density."""
    dfai = df[df.ai_strict]
    dfno = df[~df.ai_strict]
    rows = []
    # 2026 scraped only where the action is; also do 2024 for comparison
    for (p2, name), sub in [
        (("2024", "ai_strict"), dfai[dfai.period2 == "2024"]),
        (("2024", "non_ai"), dfno[dfno.period2 == "2024"]),
        (("2026", "ai_strict"), dfai[dfai.period2 == "2026"]),
        (("2026", "non_ai"), dfno[dfno.period2 == "2026"]),
    ]:
        n = len(sub)
        if n == 0:
            continue
        rates = sub[techs].astype(bool).mean()
        mean_len = sub["description_length"].mean()
        mean_techs = sub[techs].astype(bool).sum(axis=1).mean()
        # techs per 1K chars (density)
        tech_density = (sub[techs].astype(bool).sum(axis=1) /
                        (sub["description_length"].replace(0, np.nan) / 1000.0))
        for tech in techs:
            rows.append({
                "period": p2, "group": name, "n": n,
                "technology": tech, "rate": float(rates[tech]),
                "mean_len_chars": float(mean_len),
                "mean_techs_raw": float(mean_techs),
                "mean_tech_density_per1k": float(tech_density.mean()),
                "median_tech_density_per1k": float(tech_density.median()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 7/8: structured skills baseline + structured-vs-extracted validation
# ---------------------------------------------------------------------------

def structured_skills_top100(con):
    q = f"""
    SELECT skill, COUNT(DISTINCT uid) AS n_postings
    FROM '{ASANICZKA_SKILLS}'
    GROUP BY skill
    ORDER BY n_postings DESC
    LIMIT 100
    """
    df = con.execute(q).df()
    # Add share of asaniczka SWE population (18,114 postings cited in README)
    q2 = f"""SELECT COUNT(DISTINCT uid) AS n FROM '{ASANICZKA_SKILLS}'"""
    total = con.execute(q2).df()["n"].iloc[0]
    df["share_of_asaniczka_swe"] = df["n_postings"] / total
    df["total_uids"] = total
    return df


def structured_vs_extracted_validation(df, techs, con):
    """Compare asaniczka-only: structured skill shares vs regex-extracted tech shares."""
    asan = df[df.source == "kaggle_asaniczka"].copy()
    # Only keep rows in structured skills set
    structured = con.execute(f"SELECT DISTINCT uid FROM '{ASANICZKA_SKILLS}'").df()
    common = asan[asan.uid.isin(structured.uid)]
    N = len(common)
    # Regex-extracted rates in that slice
    reg_rates = common[techs].astype(bool).mean()

    # Structured rates (pivot from long)
    struct = con.execute(
        f"""SELECT uid, skill FROM '{ASANICZKA_SKILLS}'
            WHERE uid IN (SELECT uid FROM '{ASANICZKA_SKILLS}')"""
    ).df()
    # Map a few names: regex taxonomy -> likely structured skill strings
    # Use a best-effort dictionary of skill aliases to compute comparable rates.
    alias_map = {
        "python": ["python"],
        "java": ["java"],
        "javascript": ["javascript"],
        "typescript": ["typescript"],
        "go_lang": ["go", "golang"],
        "rust": ["rust"],
        "c_plus_plus": ["c++"],
        "c_lang": ["c"],
        "c_sharp": ["c#"],
        "ruby": ["ruby"],
        "kotlin": ["kotlin"],
        "swift": ["swift"],
        "scala": ["scala"],
        "sql": ["sql"],
        "react": ["react", "reactjs", "react.js"],
        "angular": ["angular"],
        "vue": ["vue", "vuejs", "vue.js"],
        "nodejs": ["nodejs", "node.js", "node js", "node"],
        "spring": ["spring", "spring boot"],
        "django": ["django"],
        "flask": ["flask"],
        "fastapi": ["fastapi"],
        "dot_net": [".net", "asp.net", "dotnet"],
        "aws": ["aws", "amazon web services"],
        "azure": ["azure", "microsoft azure"],
        "gcp": ["gcp", "google cloud", "google cloud platform"],
        "kubernetes": ["kubernetes", "k8s"],
        "docker": ["docker"],
        "terraform": ["terraform"],
        "ansible": ["ansible"],
        "jenkins": ["jenkins"],
        "github_actions": ["github actions"],
        "postgresql": ["postgresql", "postgres"],
        "mysql": ["mysql"],
        "mongodb": ["mongodb"],
        "redis": ["redis"],
        "kafka": ["kafka", "apache kafka"],
        "spark": ["spark", "apache spark"],
        "airflow": ["airflow", "apache airflow"],
        "databricks": ["databricks"],
        "snowflake": ["snowflake"],
        "tensorflow": ["tensorflow"],
        "pytorch": ["pytorch"],
        "scikit_learn": ["scikit-learn", "scikitlearn", "sklearn"],
        "microservices": ["microservices"],
        "agile": ["agile"],
        "scrum": ["scrum"],
        "ci_cd": ["ci/cd", "cicd", "ci cd"],
        "graphql": ["graphql"],
        "rest_api": ["rest api", "rest", "restful api", "restful"],
        "git": ["git"],
        "linux": ["linux"],
        "langchain": ["langchain"],
        "rag": ["rag"],
        "llm": ["llm", "large language model", "large language models"],
    }
    # Build a per-tech boolean for structured skills
    rows = []
    struct_by_uid = struct.groupby("uid")["skill"].apply(set)
    for tech in techs:
        aliases = alias_map.get(tech, [])
        if not aliases:
            rows.append({"technology": tech,
                         "structured_rate": np.nan,
                         "extracted_rate": float(reg_rates.get(tech, np.nan)),
                         "diff_struct_minus_extracted": np.nan,
                         "aliases": ""})
            continue
        aliases_l = {a.lower() for a in aliases}
        has_struct = common["uid"].map(
            lambda u: bool(struct_by_uid.get(u, set()) & aliases_l)
        )
        sr = float(has_struct.mean())
        er = float(reg_rates.get(tech, np.nan))
        rows.append({"technology": tech, "structured_rate": sr,
                     "extracted_rate": er,
                     "diff_struct_minus_extracted": sr - er,
                     "aliases": "; ".join(aliases)})
    out = pd.DataFrame(rows)
    # Spearman rank correlation on the aligned subset
    from scipy.stats import spearmanr
    aligned = out.dropna(subset=["structured_rate", "extracted_rate"])
    if len(aligned) >= 5:
        r, p = spearmanr(aligned["structured_rate"], aligned["extracted_rate"])
    else:
        r, p = (np.nan, np.nan)
    out.attrs["spearman_r"] = float(r) if r is not None else np.nan
    out.attrs["spearman_p"] = float(p) if p is not None else np.nan
    out.attrs["n"] = len(aligned)
    out.attrs["N_postings"] = N
    return out


# ---------------------------------------------------------------------------
# Step 9: seniority-level structured-skill chi-squared (asaniczka only)
# ---------------------------------------------------------------------------

def structured_seniority_chi2(df, con):
    """Chi-squared per skill between entry-vs-senior for asaniczka, with FDR correction."""
    from scipy.stats import chi2_contingency
    from statsmodels.stats.multitest import multipletests

    asan = df[df.source == "kaggle_asaniczka"].copy()
    asan["is_entry_senior"] = None
    asan.loc[asan["is_j3"], "is_entry_senior"] = "entry"
    asan.loc[asan["is_s4"], "is_entry_senior"] = "senior"
    core = asan[asan["is_entry_senior"].isin(["entry", "senior"])][["uid", "is_entry_senior"]].copy()
    if len(core) < 200:
        return pd.DataFrame(), {"n_total": len(core)}
    core_uids = set(core["uid"].tolist())

    # Pull all skills for those uids
    placeholder = ",".join([f"'{u}'" for u in list(core_uids)[:30000]])  # safety cap
    # If too many, use pandas join instead
    all_skills = con.execute(f"SELECT uid, skill FROM '{ASANICZKA_SKILLS}'").df()
    sub = all_skills[all_skills.uid.isin(core_uids)]
    sub = sub.merge(core, on="uid", how="left")
    # top 150 skills by frequency
    top_skills = sub.skill.value_counts().head(150).index.tolist()

    n_entry = (core["is_entry_senior"] == "entry").sum()
    n_senior = (core["is_entry_senior"] == "senior").sum()

    rows = []
    for sk in top_skills:
        mask = sub.skill == sk
        e_has = sub[mask & (sub.is_entry_senior == "entry")].uid.nunique()
        s_has = sub[mask & (sub.is_entry_senior == "senior")].uid.nunique()
        e_no = n_entry - e_has
        s_no = n_senior - s_has
        table = np.array([[e_has, s_has], [e_no, s_no]])
        if table.min() < 5:
            chi2 = np.nan
            p = np.nan
        else:
            chi2, p, _, _ = chi2_contingency(table)
        rows.append({
            "skill": sk,
            "n_entry_has": int(e_has),
            "n_senior_has": int(s_has),
            "share_entry": float(e_has / max(n_entry, 1)),
            "share_senior": float(s_has / max(n_senior, 1)),
            "diff_senior_minus_entry": float(s_has / max(n_senior, 1) - e_has / max(n_entry, 1)),
            "chi2": float(chi2) if not np.isnan(chi2) else np.nan,
            "p_value": float(p) if not np.isnan(p) else np.nan,
        })
    df_out = pd.DataFrame(rows)
    valid = df_out["p_value"].notna()
    if valid.any():
        _, p_adj, _, _ = multipletests(df_out.loc[valid, "p_value"], alpha=0.05, method="fdr_bh")
        df_out.loc[valid, "p_fdr"] = p_adj
    df_out = df_out.sort_values("diff_senior_minus_entry", ascending=False)
    meta = {"n_entry": int(n_entry), "n_senior": int(n_senior), "n_skills": len(top_skills)}
    return df_out, meta


# ---------------------------------------------------------------------------
# ML-engineer source stratification (Gate 1 pre-commit)
# ---------------------------------------------------------------------------

def ml_engineer_caveat(df, techs):
    """For ML/AI-specific tech comparison on 2024, stratify by source.

    Flag the techs that are ML/AI-ecosystem-flavored.
    """
    ml_ai_techs = [t for t in techs if t in {
        "tensorflow", "pytorch", "scikit_learn", "pandas", "numpy", "jupyter",
        "mlflow", "xgboost", "keras", "langchain", "llamaindex", "rag",
        "vector_database", "pinecone", "weaviate", "chroma", "hugging_face",
        "openai_api", "claude_api", "anthropic", "gemini", "prompt_engineering",
        "fine_tuning", "mcp", "llm", "ai_agent", "copilot", "cursor_tool",
        "chatgpt", "claude_tool", "codex", "tabnine", "gpt_model",
    }]

    rows = []
    sub2024 = df[df.period2 == "2024"]
    for src in ["kaggle_arshkon", "kaggle_asaniczka"]:
        ss = sub2024[sub2024.source == src]
        n = len(ss)
        if n == 0:
            continue
        rates = ss[ml_ai_techs].astype(bool).mean()
        for tech in ml_ai_techs:
            rows.append({"source": src, "period": "2024", "n": n,
                         "technology": tech, "rate": float(rates[tech])})
    sub2026 = df[df.period2 == "2026"]
    n2026 = len(sub2026)
    if n2026:
        rates2026 = sub2026[ml_ai_techs].astype(bool).mean()
        for tech in ml_ai_techs:
            rows.append({"source": "scraped", "period": "2026", "n": n2026,
                         "technology": tech, "rate": float(rates2026[tech])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sensitivities: aggregator exclusion + company capping + SWE tier
# ---------------------------------------------------------------------------

def sensitivity_suite(df, techs):
    """Headline rates under (a) baseline, (b) excl aggregators, (c) company cap 50,
    (d) SWE tier 'rule'+'llm' only. Report for top-20 tech by scraped rate only."""
    top_by_scraped = df[df.period2 == "2026"][techs].astype(bool).mean().sort_values(
        ascending=False
    ).head(20).index.tolist()

    def rates(sub):
        rows = []
        for (p2, sen), g in [
            (("2024", "J3"), sub[(sub.period2 == "2024") & sub.is_j3]),
            (("2026", "J3"), sub[(sub.period2 == "2026") & sub.is_j3]),
            (("2024", "S4"), sub[(sub.period2 == "2024") & sub.is_s4]),
            (("2026", "S4"), sub[(sub.period2 == "2026") & sub.is_s4]),
        ]:
            n = len(g)
            if not n:
                continue
            r = g[top_by_scraped].astype(bool).mean()
            for t in top_by_scraped:
                rows.append({"period": p2, "seniority": sen, "n": n,
                             "technology": t, "rate": float(r[t])})
        return pd.DataFrame(rows)

    out = {}
    out["baseline"] = rates(df)
    out["excl_aggregator"] = rates(df[~df.is_aggregator.astype(bool)])
    capped = company_cap(df, cap=50)
    out["cap50"] = rates(capped)
    tier = df[df.swe_classification_tier.isin(["rule", "llm"])]
    out["swe_tier_rule_or_llm"] = rates(tier)

    rows = []
    for k, v in out.items():
        v = v.copy()
        v["sensitivity"] = k
        rows.append(v)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-period phi (for T35 prep)
# ---------------------------------------------------------------------------

def per_period_phi(df, techs, cap=50):
    """Produce phi-per-period long-form with company capping (50/company) for density."""
    rows = []
    for p2 in ["2024", "2026"]:
        sub = df[df.period2 == p2]
        sub = company_cap(sub, cap=cap)
        if len(sub) < 200:
            continue
        X = sub[techs].astype(bool).values
        phi = phi_matrix(X)
        tri = np.triu_indices(len(techs), k=1)
        for i, j in zip(*tri):
            v = phi[i, j]
            if np.isnan(v):
                continue
            rows.append({"period": p2, "a": techs[i], "b": techs[j],
                         "phi": float(v), "n": len(sub)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    con = duckdb.connect()
    print("[load] core frame with tech matrix...", flush=True)
    df = load_core_frame(con)
    print(f"  n = {len(df):,}, cols = {len(df.columns)}", flush=True)
    techs = tech_cols(df)
    print(f"  tech cols = {len(techs)}", flush=True)

    df = seniority_bin(df)
    df = add_strict_ai_flag(df)

    # 2. tech rates by period x seniority
    print("[step 2] tech rates by period x seniority...", flush=True)
    rates_all = tech_rates_by_group(df, techs)
    rates_all.to_csv(OUT_TBL / "tech_rates_by_period_seniority.csv", index=False)

    rates_2024_src = tech_rates_by_source_2024(df, techs)
    rates_2024_src.to_csv(OUT_TBL / "tech_rates_2024_by_source.csv", index=False)

    # 3. phi matrix (pooled)
    print("[step 3] phi coefficients (pooled)...", flush=True)
    # Apply company cap 50 for per-firm density per T14 spec
    capped = company_cap(df, cap=50)
    write_phi(
        capped, techs,
        OUT_TBL / "phi_matrix_pooled.csv",
        OUT_TBL / "phi_matrix_pooled_long.csv",
        OUT_TBL / "top20_phi_pairs.csv",
        label="pooled_cap50",
    )

    # 3b. per-period phi for T35 prep
    print("[step 3b] per-period phi for T35 prep...", flush=True)
    per_p = per_period_phi(df, techs, cap=50)
    per_p.to_csv(OUT_TBL / "phi_per_period_long.csv", index=False)

    # 4. trajectory classification
    print("[step 4] trajectory classification...", flush=True)
    traj = trajectory_classify(rates_all, rates_2024_src)
    traj.to_csv(OUT_TBL / "tech_trajectory.csv", index=False)

    # 5. stack diversity
    print("[step 5] stack diversity...", flush=True)
    sd = stack_diversity(df, techs)
    sd.to_csv(OUT_TBL / "stack_diversity.csv", index=False)

    # 6. AI co-occurrence
    print("[step 6] AI co-occurrence...", flush=True)
    ai_co = ai_co_occurrence(df, techs)
    ai_co.to_csv(OUT_TBL / "ai_co_occurrence.csv", index=False)

    # 7. structured skills top-100
    print("[step 7] structured skills top-100 (asaniczka)...", flush=True)
    top100 = structured_skills_top100(con)
    top100.to_csv(OUT_TBL / "asaniczka_top100_structured.csv", index=False)

    # 8. structured vs extracted
    print("[step 8] structured vs extracted validation...", flush=True)
    val = structured_vs_extracted_validation(df, techs, con)
    val.to_csv(OUT_TBL / "structured_vs_extracted.csv", index=False)
    with open(OUT_TBL / "structured_vs_extracted_meta.json", "w") as f:
        json.dump({"spearman_r": val.attrs.get("spearman_r"),
                   "spearman_p": val.attrs.get("spearman_p"),
                   "n_aligned": val.attrs.get("n"),
                   "n_postings": val.attrs.get("N_postings")}, f, indent=2)

    # 9. structured seniority chi2
    print("[step 9] structured seniority chi2...", flush=True)
    chi_df, chi_meta = structured_seniority_chi2(df, con)
    chi_df.to_csv(OUT_TBL / "seniority_skill_structured_chi2.csv", index=False)
    with open(OUT_TBL / "seniority_skill_structured_chi2_meta.json", "w") as f:
        json.dump(chi_meta, f, indent=2)

    # ML engineer caveat
    print("[ml caveat] ML/AI techs stratified by source 2024...", flush=True)
    ml = ml_engineer_caveat(df, techs)
    ml.to_csv(OUT_TBL / "ml_tech_source_stratified.csv", index=False)

    # Sensitivities
    print("[sensitivity] aggregator / cap / tier...", flush=True)
    sens = sensitivity_suite(df, techs)
    sens.to_csv(OUT_TBL / "tech_rates_sensitivities.csv", index=False)

    # ---- Figures ----
    print("[figures] tech trajectory + stack diversity...", flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig: tech trajectory heatmap (top 30 rising + 10 declining)
    traj_plot = traj.dropna(subset=["delta"]).copy()
    rising = traj_plot[traj_plot["class"] == "rising"].head(25)
    declining = traj_plot[traj_plot["class"] == "declining"].tail(10)
    noise = traj_plot[traj_plot["class"] == "calibration_noise"].head(10)
    combo = pd.concat([rising, declining, noise], ignore_index=True).drop_duplicates(
        subset=["technology"])
    combo = combo.sort_values("delta")
    fig, ax = plt.subplots(figsize=(9, max(6, 0.3 * len(combo))))
    colors = combo["class"].map({
        "rising": "#2a9d8f", "declining": "#e76f51",
        "calibration_noise": "#bbbbbb", "stable": "#f4a261",
        "trace": "#dddddd"
    })
    ax.barh(combo["technology"], combo["delta"] * 100.0, color=colors)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Δ rate (2026 - 2024), percentage points")
    ax.set_title("Technology trajectory (capped at 50/company).\n"
                 "Green=rising ≥0.5pp & SNR≥2; Red=declining; Grey=calibration noise")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "T14_tech_trajectory.png", dpi=130)
    plt.close()

    # Fig: stack diversity
    fig, ax = plt.subplots(figsize=(7, 5))
    pivot = sd.pivot_table(index="seniority", columns="period", values="median",
                           aggfunc="first")
    pivot.plot(kind="bar", ax=ax, color=["#8ecae6", "#219ebc"])
    ax.set_ylabel("Median distinct techs per posting")
    ax.set_title("Stack diversity by period × seniority")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "T14_stack_diversity.png", dpi=130)
    plt.close()

    print(f"[done] {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
