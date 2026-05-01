"""
T21. Senior role evolution deep dive

Tracks how senior SWE roles (mid-senior + director) shifted between 2024 and
2026 across four language profiles using strict-detector patterns from the
start: people-management, mentoring, technical orchestration, and strategic
scope. Also validates T11's IC+mentoring reframing, looks for emergent
sub-archetypes, and explores the AI-mentioning senior subset.

Outputs:
    exploration/tables/T21/*.csv
    exploration/figures/T21/*.png
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/home/jihgaboot/gabor/job-research")
OUT_TAB = ROOT / "exploration/tables/T21"
OUT_FIG = ROOT / "exploration/figures/T21"
OUT_TAB.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

T11_FEAT = ROOT / "exploration/tables/T11/T11_features.parquet"
CLEANED = ROOT / "exploration/artifacts/shared/swe_cleaned_text.parquet"
ARCHETYPE = ROOT / "exploration/artifacts/shared/swe_archetype_labels.parquet"
UNIFIED = ROOT / "data/unified.parquet"

COMPANY_CAP = 20  # cap primary sensitivity

# ---- Pattern definitions (four language profiles) ----

# People management (strict) — T11/V1 verified ~100% precision
PEOPLE_MGMT_PATTERNS = {
    "people_manager": r"\bpeople manager\b",
    "people_management": r"\bpeople management\b",
    "direct_reports": r"\bdirect reports?\b",
    "performance_review": r"\bperformance reviews?\b",
    "headcount": r"\bheadcount\b",
    "manage_team_of": r"\bmanag(?:e|ing)\s+(?:a\s+)?team of\b",
    "lead_team_of": r"\blead(?:ing)?\s+(?:a\s+)?team of\b",
    "interview_engineers": r"\binterview\s+(?:engineers?|developers?|candidates?)\b",
    "hire_engineers": r"\bhir(?:e|ing)\s+(?:engineers?|developers?)\b",
    "team_building": r"\bteam building\b",
}

# Mentoring (strict) — T11/V1 verified ~100% precision
MENTORING_PATTERNS = {
    "mentor_engineers": r"\bmentor(?:ing)?\s+(?:engineers?|junior|team|others|developers?)\b",
    "coach_engineers": r"\bcoach(?:ing)?\s+(?:engineers?|junior|developers?|team)\b",
    "mentor_and_develop": r"\bmentor\s+and\s+develop\b",
    "mentorship": r"\bmentorship\b",
}

# Technical orchestration
TECH_ORCH_PATTERNS = {
    "architecture_review": r"\barchitecture review\b",
    "code_review": r"\bcode reviews?\b",
    "system_design": r"\bsystem design\b",
    "technical_direction": r"\btechnical direction\b",
    "ai_orchestration": r"\bai orchestration\b",
    "agentic": r"\bagentic\b",
    "multi_agent": r"\bmulti[- ]agent\b",
    "automation": r"\bautomation\b",
    "evaluation": r"\bevaluation\b",
    "prompt_engineering": r"\bprompt engineer(?:ing)?\b",
    "tool_selection": r"\btool selection\b",
    "quality_gate": r"\bquality gates?\b",
    "guardrails": r"\bguardrails\b",
}

# Strategic scope
STRAT_PATTERNS = {
    "stakeholder": r"\bstakeholders?\b",
    "business_impact": r"\bbusiness impact\b",
    "revenue": r"\brevenue\b",
    "product_strategy": r"\bproduct strategy\b",
    "roadmap": r"\broadmap\b",
    "prioritization": r"\bprioritiz(?:e|ation)\b",
    "resource_allocation": r"\bresource allocation\b",
    "cross_functional": r"\bcross[- ]functional\b",
}

# Validated AI pattern (use for AI-mention filter, not bare `agent`)
AI_PATTERNS = {
    "agentic": r"\bagentic\b",
    "ai_agent": r"\bai\s+agent\b",
    "multi_agent": r"\bmulti[- ]agent\b",
    "llm": r"\bllms?\b",
    "gen_ai": r"\b(?:generative|gen[- ]?)ai\b",
    "rag": r"\brag\b",
    "retrieval_augmented": r"\bretrieval[- ]augmented\b",
    "copilot": r"\bcopilot\b",
    "chatgpt": r"\bchatgpt\b",
    "claude": r"\bclaude\b",
    "langchain": r"\blangchain\b",
    "langgraph": r"\blanggraph\b",
    "foundation_models": r"\bfoundation models?\b",
    "large_language_model": r"\blarge language models?\b",
    "prompt_engineering": r"\bprompt engineer(?:ing)?\b",
    "vector_db": r"\bvector (?:database|db|store)\b",
    "fine_tune": r"\bfine[- ]?tun(?:e|ing)\b",
}

# Credential-stripping vocabulary for T12 correlation
CREDENTIAL_PATTERNS = {
    "qualifications": r"\bqualifications\b",
    "required": r"\brequired\b",
    "requirements": r"\brequirements\b",
    "degree": r"\bdegree\b",
    "bachelor": r"\bbachelor(?:'s)?\b",
}


def _compile(group: dict[str, str]) -> dict[str, re.Pattern]:
    return {k: re.compile(v, re.IGNORECASE) for k, v in group.items()}


def _asserts():
    people = _compile(PEOPLE_MGMT_PATTERNS)
    mentor = _compile(MENTORING_PATTERNS)
    tech = _compile(TECH_ORCH_PATTERNS)
    strat = _compile(STRAT_PATTERNS)
    ai = _compile(AI_PATTERNS)

    # People management
    assert people["manage_team_of"].search("manage a team of 5 engineers")
    assert people["direct_reports"].search("you will have 5 direct reports")
    assert people["performance_review"].search("conduct performance reviews")
    assert people["hire_engineers"].search("hire engineers for the team")
    assert not people["people_manager"].search("software engineer")
    # mentor
    assert mentor["mentor_engineers"].search("mentor junior engineers")
    assert mentor["mentorship"].search("provide mentorship to others")
    assert not mentor["mentor_engineers"].search("we value respect")
    # tech orchestration
    assert tech["code_review"].search("conduct code reviews")
    assert tech["system_design"].search("lead system design")
    assert tech["agentic"].search("build agentic workflows")
    assert tech["multi_agent"].search("design multi-agent systems")
    # strategic
    assert strat["stakeholder"].search("align with stakeholders")
    assert strat["roadmap"].search("drive the roadmap forward")
    # AI
    assert ai["agentic"].search("agentic retrieval")
    assert ai["llm"].search("integrate LLMs")
    assert not ai["ai_agent"].search("insurance agent")
    print("[T21] regex asserts passed")


def count_all(text: str, patterns: dict[str, re.Pattern]) -> dict[str, int]:
    if not text:
        return {k: 0 for k in patterns}
    return {k: len(p.findall(text)) for k, p in patterns.items()}


def load_senior() -> pd.DataFrame:
    con = duckdb.connect()
    t11 = con.execute(f"SELECT * FROM read_parquet('{T11_FEAT}')").fetchdf()
    cleaned = con.execute(
        f"""SELECT uid, description_cleaned, text_source
            FROM read_parquet('{CLEANED}')"""
    ).fetchdf()
    t11 = t11.merge(cleaned, on="uid", how="left")

    raw = con.execute(
        f"""SELECT uid, description, seniority_llm, llm_classification_coverage
            FROM read_parquet('{UNIFIED}')
            WHERE source_platform='linkedin' AND is_english=true
              AND date_flag='ok' AND is_swe=true"""
    ).fetchdf()
    t11 = t11.merge(raw, on="uid", how="left")

    # Combined best-available seniority
    sba = np.where(
        t11["llm_classification_coverage"] == "labeled",
        t11["seniority_llm"],
        np.where(
            t11["llm_classification_coverage"] == "rule_sufficient",
            t11["seniority_final"],
            None,
        ),
    )
    t11["seniority_best_available_combined"] = sba

    # Text for regex: cleaned if >=50 char else raw
    text = t11["description_cleaned"].fillna("")
    fallback = t11["description"].fillna("")
    use_fallback = text.str.len() < 50
    t11["text_for_regex"] = np.where(use_fallback, fallback, text)
    t11["text_len_effective"] = t11["text_for_regex"].str.len().clip(lower=1)

    t11["year_bucket"] = t11["year"].astype(str)
    return t11


def add_profile_counts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    people = _compile(PEOPLE_MGMT_PATTERNS)
    mentor = _compile(MENTORING_PATTERNS)
    tech = _compile(TECH_ORCH_PATTERNS)
    strat = _compile(STRAT_PATTERNS)
    ai = _compile(AI_PATTERNS)
    cred = _compile(CREDENTIAL_PATTERNS)

    # Sums per profile
    def sum_profile(text, patt):
        return sum(len(p.findall(text)) for p in patt.values()) if text else 0

    df["people_mgmt_count"] = df["text_for_regex"].apply(lambda s: sum_profile(s, people))
    df["mentor_count"] = df["text_for_regex"].apply(lambda s: sum_profile(s, mentor))
    df["tech_orch_count"] = df["text_for_regex"].apply(lambda s: sum_profile(s, tech))
    df["strat_count"] = df["text_for_regex"].apply(lambda s: sum_profile(s, strat))
    df["ai_count"] = df["text_for_regex"].apply(lambda s: sum_profile(s, ai))
    df["cred_count"] = df["text_for_regex"].apply(lambda s: sum_profile(s, cred))

    per1k = 1000.0 / df["text_len_effective"]
    df["people_mgmt_density"] = df["people_mgmt_count"] * per1k
    df["mentor_density"] = df["mentor_count"] * per1k
    df["tech_orch_density"] = df["tech_orch_count"] * per1k
    df["strat_density"] = df["strat_count"] * per1k
    df["ai_density"] = df["ai_count"] * per1k
    df["cred_density"] = df["cred_count"] * per1k
    df["has_ai"] = (df["ai_count"] > 0).astype(int)
    return df


def profile_precision_sample(df: pd.DataFrame, patterns: dict[str, re.Pattern],
                             label: str, n_per_period: int = 25) -> pd.DataFrame:
    """Sample matches per period for manual/rule-based precision assessment."""
    rng = np.random.default_rng(7)
    rows = []
    for period in ["2024", "2026"]:
        sub = df[df["year_bucket"] == period]
        for pname, patt in patterns.items():
            mask = sub["text_for_regex"].apply(lambda s: bool(patt.search(s) if s else False))
            hits = sub[mask]
            if len(hits) == 0:
                continue
            take = min(n_per_period, len(hits))
            sample = hits.sample(take, random_state=int(rng.integers(0, 10000)))
            for _, row in sample.iterrows():
                text = row["text_for_regex"] or ""
                m = patt.search(text)
                if not m:
                    continue
                start = max(0, m.start() - 80)
                end = min(len(text), m.end() + 80)
                snippet = text[start:end].replace("\n", " ")
                rows.append({
                    "profile": label,
                    "pattern": pname,
                    "period": period,
                    "uid": row["uid"],
                    "seniority_final": row["seniority_final"],
                    "snippet": snippet,
                })
    return pd.DataFrame(rows)


def mean_density_by_period(senior: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Stratify by broad senior category
    for period in ["2024", "2026"]:
        for group, mask in [
            ("mid_senior", senior["seniority_final"] == "mid-senior"),
            ("director", senior["seniority_final"] == "director"),
            ("senior_all", senior["seniority_final"].isin(["mid-senior", "director"])),
        ]:
            sub = senior[mask & (senior["year_bucket"] == period)]
            if len(sub) == 0:
                continue
            rows.append({
                "period": period,
                "group": group,
                "n": len(sub),
                "people_mgmt_density": sub["people_mgmt_density"].mean(),
                "mentor_density": sub["mentor_density"].mean(),
                "tech_orch_density": sub["tech_orch_density"].mean(),
                "strat_density": sub["strat_density"].mean(),
                "ai_density": sub["ai_density"].mean(),
                "cred_density": sub["cred_density"].mean(),
                "people_mgmt_any_share": (sub["people_mgmt_count"] > 0).mean(),
                "mentor_any_share": (sub["mentor_count"] > 0).mean(),
                "tech_orch_any_share": (sub["tech_orch_count"] > 0).mean(),
                "strat_any_share": (sub["strat_count"] > 0).mean(),
                "ai_any_share": sub["has_ai"].mean(),
            })
    return pd.DataFrame(rows)


def company_capped(df: pd.DataFrame, cap: int = COMPANY_CAP) -> pd.DataFrame:
    """Sample up to cap postings per (company, period) to reduce volume-driven bias."""
    rng = np.random.default_rng(11)
    df = df.copy()
    df["__rng"] = rng.random(len(df))
    df["__rank"] = df.groupby(["company_name_canonical", "year_bucket"])["__rng"].rank("first")
    capped = df[df["__rank"] <= cap].drop(columns=["__rng", "__rank"])
    return capped


# ---- Corpus term-level comparison for each profile ----

def term_level_counts(df: pd.DataFrame, patterns: dict[str, str],
                       profile: str) -> pd.DataFrame:
    compiled = _compile(patterns)
    rows = []
    for period in ["2024", "2026"]:
        sub = df[df["year_bucket"] == period]
        n = len(sub)
        if n == 0:
            continue
        total_chars = sub["text_len_effective"].sum()
        for name, patt in compiled.items():
            match_count = sub["text_for_regex"].apply(
                lambda s: len(patt.findall(s)) if s else 0
            ).sum()
            post_with_any = sub["text_for_regex"].apply(
                lambda s: bool(patt.search(s)) if s else False
            ).sum()
            rows.append({
                "profile": profile,
                "term": name,
                "period": period,
                "n_postings": n,
                "postings_with_match": int(post_with_any),
                "postings_with_match_pct": post_with_any / n,
                "total_matches": int(match_count),
                "per_1k_chars": match_count / total_chars * 1000,
            })
    return pd.DataFrame(rows)


# ---- Main ----

def main():
    _asserts()
    print("[T21] loading data...")
    df_all = load_senior()
    df_all = add_profile_counts(df_all)

    # Senior pool: mid-senior + director under seniority_final (primary frame)
    senior = df_all[df_all["seniority_final"].isin(["mid-senior", "director"])].copy()
    print(f"[T21] senior n = {len(senior)}; by period: "
          f"{(senior['year_bucket']=='2024').sum()} | "
          f"{(senior['year_bucket']=='2026').sum()}")

    # 1. Mean densities by period (raw + capped) ------------------------------
    mean_raw = mean_density_by_period(senior)
    mean_raw.to_csv(OUT_TAB / "senior_density_by_period_raw.csv", index=False)

    senior_capped = company_capped(senior, COMPANY_CAP)
    mean_capped = mean_density_by_period(senior_capped)
    mean_capped.to_csv(OUT_TAB / "senior_density_by_period_capped20.csv", index=False)
    print("[T21] wrote density tables")

    # 2. Term-level breakdown per profile ------------------------------------
    all_terms = []
    for patt_dict, label in [
        (PEOPLE_MGMT_PATTERNS, "people_mgmt"),
        (MENTORING_PATTERNS, "mentoring"),
        (TECH_ORCH_PATTERNS, "tech_orch"),
        (STRAT_PATTERNS, "strategic"),
    ]:
        # Use company-capped pool as primary for corpus-level term frequency per instructions
        t = term_level_counts(senior_capped, patt_dict, label)
        all_terms.append(t)
    terms_df = pd.concat(all_terms, ignore_index=True)
    terms_df.to_csv(OUT_TAB / "senior_term_level_counts_capped20.csv", index=False)

    # Also raw
    all_terms_raw = []
    for patt_dict, label in [
        (PEOPLE_MGMT_PATTERNS, "people_mgmt"),
        (MENTORING_PATTERNS, "mentoring"),
        (TECH_ORCH_PATTERNS, "tech_orch"),
        (STRAT_PATTERNS, "strategic"),
    ]:
        all_terms_raw.append(term_level_counts(senior, patt_dict, label))
    pd.concat(all_terms_raw, ignore_index=True).to_csv(
        OUT_TAB / "senior_term_level_counts_raw.csv", index=False)
    print("[T21] wrote term-level counts")

    # 3. 2D + 3D scatter plots (people-mgmt vs mentor vs tech-orch) ----------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    for period, color, marker in [("2024", "#4C72B0", "o"), ("2026", "#C44E52", "x")]:
        sub = senior[senior["year_bucket"] == period]
        ax[0].scatter(sub["people_mgmt_density"], sub["mentor_density"],
                      alpha=0.25, s=10, color=color, label=period, marker=marker)
    ax[0].set_xlabel("People-management density (/1K chars)")
    ax[0].set_ylabel("Mentoring density (/1K chars)")
    ax[0].set_title("Senior: People-Management vs Mentoring")
    ax[0].set_xlim(0, 4)
    ax[0].set_ylim(0, 6)
    ax[0].legend()

    for period, color, marker in [("2024", "#4C72B0", "o"), ("2026", "#C44E52", "x")]:
        sub = senior[senior["year_bucket"] == period]
        ax[1].scatter(sub["mentor_density"], sub["tech_orch_density"],
                      alpha=0.25, s=10, color=color, label=period, marker=marker)
    ax[1].set_xlabel("Mentoring density (/1K chars)")
    ax[1].set_ylabel("Tech-orchestration density (/1K chars)")
    ax[1].set_title("Senior: Mentoring vs Tech-Orchestration")
    ax[1].set_xlim(0, 6)
    ax[1].set_ylim(0, 15)
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG / "senior_profile_scatter.png", dpi=120)
    plt.close()
    print("[T21] wrote senior_profile_scatter.png")

    # 4. Senior sub-archetypes via k-means ------------------------------------
    feat_cols = ["people_mgmt_density", "mentor_density", "tech_orch_density",
                 "strat_density"]
    X = senior[feat_cols].values
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    senior["cluster"] = km.fit_predict(Xs)

    cluster_profile = senior.groupby("cluster")[feat_cols].mean().reset_index()
    cluster_profile["n"] = senior.groupby("cluster").size().values
    cluster_profile.to_csv(OUT_TAB / "senior_kmeans_clusters.csv", index=False)

    cluster_shares = pd.crosstab(senior["cluster"], senior["year_bucket"],
                                  normalize="columns").reset_index()
    cluster_shares.to_csv(OUT_TAB / "senior_kmeans_shares_by_period.csv", index=False)

    # Label clusters by dominant profile
    def label_cluster(row):
        profs = {
            "PeopleMgr": row["people_mgmt_density"],
            "Mentor": row["mentor_density"],
            "TechOrch": row["tech_orch_density"],
            "Strategic": row["strat_density"],
        }
        if max(profs.values()) < 0.1:
            return "Generic"
        return max(profs, key=profs.get)
    cluster_profile["label"] = cluster_profile.apply(label_cluster, axis=1)
    cluster_profile.to_csv(OUT_TAB / "senior_kmeans_clusters.csv", index=False)

    # Bar plot of cluster shares by period
    shares = cluster_shares.copy()
    shares.columns = ["cluster", "2024", "2026"]
    shares = shares.merge(cluster_profile[["cluster", "label"]], on="cluster")
    shares["label_cluster"] = shares["label"] + " #" + shares["cluster"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(shares))
    w = 0.35
    ax.bar(x - w/2, shares["2024"], w, label="2024", color="#4C72B0")
    ax.bar(x + w/2, shares["2026"], w, label="2026", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(shares["label_cluster"], rotation=30, ha="right")
    ax.set_ylabel("Share of senior postings")
    ax.set_title("T21. Senior sub-archetype shares (k-means on density vector)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "senior_cluster_shares.png", dpi=120)
    plt.close()
    print("[T21] wrote senior_cluster_shares.png")

    # 5. AI interaction: AI-mentioning senior vs non-AI-mentioning -----------
    ai_rows = []
    for period in ["2024", "2026"]:
        for ai_flag, label in [(1, "ai_mentioning"), (0, "non_ai_mentioning")]:
            sub = senior[(senior["year_bucket"] == period) & (senior["has_ai"] == ai_flag)]
            if len(sub) == 0:
                continue
            ai_rows.append({
                "period": period, "subset": label, "n": len(sub),
                "people_mgmt_density": sub["people_mgmt_density"].mean(),
                "mentor_density": sub["mentor_density"].mean(),
                "tech_orch_density": sub["tech_orch_density"].mean(),
                "strat_density": sub["strat_density"].mean(),
            })
    pd.DataFrame(ai_rows).to_csv(OUT_TAB / "senior_ai_mentioning_comparison.csv", index=False)
    print("[T21] wrote ai mentioning comparison")

    # 6. Director specifically -----------------------------------------------
    director_rows = []
    for period in ["2024", "2026"]:
        for seniority in ["mid-senior", "director"]:
            sub = senior[(senior["year_bucket"] == period) &
                         (senior["seniority_final"] == seniority)]
            director_rows.append({
                "period": period, "seniority": seniority, "n": len(sub),
                "people_mgmt_density": sub["people_mgmt_density"].mean(),
                "mentor_density": sub["mentor_density"].mean(),
                "tech_orch_density": sub["tech_orch_density"].mean(),
                "strat_density": sub["strat_density"].mean(),
                "ai_density": sub["ai_density"].mean(),
            })
    pd.DataFrame(director_rows).to_csv(OUT_TAB / "director_vs_midsenior.csv", index=False)

    # 7. Cross-seniority management comparison (senior vs entry) -------------
    entry = df_all[df_all["seniority_final"] == "entry"].copy()
    cross_rows = []
    for label, sub in [("senior", senior), ("entry", entry)]:
        for period in ["2024", "2026"]:
            s = sub[sub["year_bucket"] == period]
            if len(s) == 0:
                continue
            cross_rows.append({
                "seniority": label,
                "period": period,
                "n": len(s),
                "people_mgmt_density": s["people_mgmt_density"].mean(),
                "mentor_density": s["mentor_density"].mean(),
                "tech_orch_density": s["tech_orch_density"].mean(),
                "ai_density": s["ai_density"].mean(),
            })
    pd.DataFrame(cross_rows).to_csv(OUT_TAB / "cross_seniority_comparison.csv", index=False)
    print("[T21] wrote cross seniority comparison")

    # 8. Credential vocabulary stripping correlation --------------------------
    senior_2026 = senior[senior["year_bucket"] == "2026"].copy()
    # Spearman-like rank correlation via pandas
    cred_corr = {}
    for col in ["mentor_density", "tech_orch_density", "people_mgmt_density", "ai_density"]:
        r = senior_2026[["cred_density", col]].corr(method="spearman").iloc[0, 1]
        cred_corr[col] = float(r)
    pd.DataFrame([cred_corr]).to_csv(OUT_TAB / "credential_stripping_correlation.csv",
                                      index=False)

    # Also for 2024 as comparison
    senior_2024 = senior[senior["year_bucket"] == "2024"]
    cred_corr_2024 = {}
    for col in ["mentor_density", "tech_orch_density", "people_mgmt_density", "ai_density"]:
        r = senior_2024[["cred_density", col]].corr(method="spearman").iloc[0, 1]
        cred_corr_2024[col] = float(r)
    pd.DataFrame([cred_corr_2024]).to_csv(
        OUT_TAB / "credential_stripping_correlation_2024.csv", index=False)
    print("[T21] wrote credential correlation")

    # 9. Domain stratification via archetype labels --------------------------
    try:
        arch = duckdb.connect().execute(
            f"SELECT uid, archetype, archetype_name FROM read_parquet('{ARCHETYPE}')"
        ).fetchdf()
    except Exception as e:
        print(f"[T21] archetype load failed: {e}")
        arch = None

    if arch is not None:
        senior_arch = senior.merge(arch, on="uid", how="inner")

        def bucket(name):
            name = (name or "").lower()
            if "ai" in name or "learning" in name:
                return "AI/ML"
            if "frontend" in name or "react" in name or "web" in name:
                return "Frontend/Web"
            if "embedded" in name or "firmware" in name or "radar" in name or "flight" in name:
                return "Embedded/Systems"
            if "data" in name or "analytics" in name:
                return "Data"
            if "cloud" in name or "devops" in name or "sre" in name or "reliability" in name:
                return "Cloud/Infra"
            if "spring" in name or "java" in name or ".net" in name or "net core" in name:
                return "Backend/Enterprise"
            return "Other"

        senior_arch["domain_bucket"] = senior_arch["archetype_name"].apply(bucket)
        dom_rows = []
        for domain in senior_arch["domain_bucket"].unique():
            for period in ["2024", "2026"]:
                sub = senior_arch[(senior_arch["domain_bucket"] == domain) &
                                   (senior_arch["year_bucket"] == period)]
                if len(sub) < 10:
                    continue
                dom_rows.append({
                    "domain": domain, "period": period, "n": len(sub),
                    "people_mgmt_density": sub["people_mgmt_density"].mean(),
                    "mentor_density": sub["mentor_density"].mean(),
                    "tech_orch_density": sub["tech_orch_density"].mean(),
                    "strat_density": sub["strat_density"].mean(),
                    "ai_density": sub["ai_density"].mean(),
                })
        pd.DataFrame(dom_rows).to_csv(OUT_TAB / "senior_by_domain.csv", index=False)
        print("[T21] wrote senior by domain")

    # 10. Sensitivity: aggregator exclusion ----------------------------------
    senior_noagg = senior[senior["is_aggregator"] == False]
    mean_noagg = mean_density_by_period(senior_noagg)
    mean_noagg.to_csv(OUT_TAB / "senior_density_no_aggregators.csv", index=False)

    # 11. Validation samples for each profile --------------------------------
    print("[T21] writing precision sample files...")
    for patt_dict, label, fname in [
        (PEOPLE_MGMT_PATTERNS, "people_mgmt", "samples_people_mgmt.csv"),
        (MENTORING_PATTERNS, "mentoring", "samples_mentoring.csv"),
        (TECH_ORCH_PATTERNS, "tech_orch", "samples_tech_orch.csv"),
        (STRAT_PATTERNS, "strategic", "samples_strategic.csv"),
    ]:
        s = profile_precision_sample(senior, _compile(patt_dict), label, n_per_period=25)
        s.to_csv(OUT_TAB / fname, index=False)

    # Summary JSON -----------------------------------------------------------
    summary = {
        "n_senior_total": int(len(senior)),
        "n_senior_2024": int((senior["year_bucket"] == "2024").sum()),
        "n_senior_2026": int((senior["year_bucket"] == "2026").sum()),
        "n_midsenior_2024": int(((senior["year_bucket"] == "2024") &
                                 (senior["seniority_final"] == "mid-senior")).sum()),
        "n_midsenior_2026": int(((senior["year_bucket"] == "2026") &
                                 (senior["seniority_final"] == "mid-senior")).sum()),
        "n_director_2024": int(((senior["year_bucket"] == "2024") &
                                (senior["seniority_final"] == "director")).sum()),
        "n_director_2026": int(((senior["year_bucket"] == "2026") &
                                (senior["seniority_final"] == "director")).sum()),
        "tables": sorted(p.name for p in OUT_TAB.glob("*.csv")),
        "figures": sorted(p.name for p in OUT_FIG.glob("*.png")),
    }
    with open(OUT_TAB / "T21_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[T21] done")


if __name__ == "__main__":
    main()
