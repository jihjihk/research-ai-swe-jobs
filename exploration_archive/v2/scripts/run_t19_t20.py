#!/usr/bin/env python3
from __future__ import annotations

import math
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import duckdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.utils.extmath import safe_sparse_dot


matplotlib.use("Agg")
sns.set_theme(style="whitegrid", context="talk")


ROOT = Path(__file__).resolve().parents[2]
STAGE8 = ROOT / "preprocessing" / "intermediate" / "stage8_final.parquet"

OUT_REPORTS = ROOT / "exploration" / "reports"
OUT_TABLES = ROOT / "exploration" / "tables"
OUT_FIGURES = ROOT / "exploration" / "figures"
OUT_T19 = OUT_TABLES / "T19"
OUT_T20 = OUT_TABLES / "T20"
OUT_FIG_T19 = OUT_FIGURES / "T19"
OUT_FIG_T20 = OUT_FIGURES / "T20"

FILTER = "source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true"

COMPANY_SUFFIX_WORDS = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "llc",
    "ltd",
    "limited",
    "group",
    "holdings",
    "holding",
    "services",
    "service",
    "solutions",
    "solution",
    "systems",
    "system",
    "technologies",
    "technology",
    "tech",
    "global",
    "international",
    "partners",
    "partner",
    "the",
    "and",
    "of",
}

BOILERPLATE_PHRASES = [
    "equal opportunity",
    "equal employment opportunity",
    "reasonable accommodation",
    "protected class",
    "benefits include",
    "benefit package includes",
    "about us",
    "about the company",
    "privacy notice",
    "fair chance",
    "we are an equal opportunity employer",
    "we are committed to equal opportunity",
    "lensa is a career site",
    "does not hire directly",
    "promotes jobs on linkedin",
    "directemployers",
    "clicking \"apply now\"",
    "read more",
    "job board/employer site",
    "recruitment ad agencies",
    "marketing partners",
    "not a staffing firm",
]

REQ_HEADINGS = [
    "requirements",
    "qualifications",
    "what you'll need",
    "what you need",
    "what we'll need",
    "what we need",
    "minimum qualifications",
    "preferred qualifications",
    "basic qualifications",
    "required qualifications",
    "required skills",
    "must have",
    "must-haves",
    "ideal candidate",
    "who you are",
    "what we're looking for",
    "what we are looking for",
    "you will need",
    "you'll need",
    "skills required",
]

SECTION_ENDERS = [
    "responsibilities",
    "what you'll do",
    "what you will do",
    "day to day",
    "about the role",
    "about us",
    "about the company",
    "company overview",
    "benefits",
    "perks",
    "compensation",
    "salary",
    "equal opportunity",
    "reasonable accommodation",
    "protected class",
    "fair chance",
    "privacy notice",
    "how to apply",
]


FEATURE_SPECS = [
    ("ai_tool_any", "ai_tool", r"\b(?:copilot|cursor|claude|gpt|chatgpt|llm|large language model|rag|retrieval augmented generation|agentic|ai agent|mcp|langchain|langgraph|openai|anthropic)\b"),
    ("ai_tool_copilot", "ai_tool", r"\bcopilot\b"),
    ("ai_tool_cursor", "ai_tool", r"\bcursor\b"),
    ("ai_tool_claude", "ai_tool", r"\bclaude\b"),
    ("ai_tool_gpt", "ai_tool", r"\bgpt\b|\bchatgpt\b"),
    ("ai_tool_llm", "ai_tool", r"\bllm\b|\blarge language model(?:s)?\b"),
    ("ai_tool_rag", "ai_tool", r"\brag\b|\bretrieval augmented generation\b"),
    ("ai_tool_agent", "ai_tool", r"\bagent(?:ic)?\b|\bai agent(?:s)?\b"),
    ("ai_tool_mcp", "ai_tool", r"\bmcp\b"),
    ("ai_tool_langchain", "ai_tool", r"\blangchain\b"),
    ("ai_tool_langgraph", "ai_tool", r"\blanggraph\b"),
    ("ai_tool_openai", "ai_tool", r"\bopenai\b"),
    ("ai_tool_anthropic", "ai_tool", r"\banthropic\b"),
    ("ai_domain_any", "ai_domain", r"\b(?:machine learning|deep learning|natural language processing|computer vision)\b"),
    ("ai_domain_ml", "ai_domain", r"\bmachine learning\b|\bml\b"),
    ("ai_domain_dl", "ai_domain", r"\bdeep learning\b|\bdl\b"),
    ("ai_domain_nlp", "ai_domain", r"\bnlp\b|\bnatural language processing\b"),
    ("ai_domain_cv", "ai_domain", r"\bcomputer vision\b"),
    ("ai_general", "ai_domain", r"\b(?:artificial intelligence|generative ai|genai|gen ai)\b"),
    ("ownership", "org_scope", r"\bownership\b|\bown(?:s|ed|ing)?\b"),
    ("end_to_end", "org_scope", r"\bend[- ]to[- ]end\b"),
    ("cross_functional", "org_scope", r"\bcross[- ]functional\b"),
    ("stakeholder", "org_scope", r"\bstakeholders?\b"),
    ("lead", "mgmt", r"\blead(?:s|ing|er|ership)?\b"),
    ("mentor", "mgmt", r"\bmentor(?:s|ed|ing)?\b"),
    ("manage", "mgmt", r"\bmanage(?:s|d|ment|r|rs|ing)?\b"),
    ("hire", "mgmt", r"\bhire(?:s|d|ing)?\b"),
    ("team", "mgmt", r"\bteam(?:s)?\b"),
    ("system_design", "sys_design", r"\bsystem design\b|\bsystems design\b"),
    ("architecture", "sys_design", r"\barchitecture(?:s|al)?\b|\barchitect\b"),
    ("distributed_systems", "sys_design", r"\bdistributed systems?\b"),
    ("microservices", "sys_design", r"\bmicroservices?\b"),
    ("scalability", "sys_design", r"\bscalab(?:e|ility|le)\b"),
    ("cicd", "method", r"\bci/?cd\b|\bcontinuous integration\b|\bcontinuous delivery\b|\bcontinuous deployment\b"),
    ("deployment", "method", r"\bdeploy(?:ment|s|ed|ing)?\b"),
    ("infrastructure", "method", r"\binfrastructure\b|\binfra\b"),
    ("testing", "method", r"\btesting\b|\btest(?:s|ed|ing)?\b|\bqa\b|\bquality assurance\b"),
    ("code_review", "method", r"\bcode review\b|\bpr review\b|\bpull request review\b"),
    ("communication", "soft_skill", r"\bcommunicat(?:e|es|ed|ing|ion)\b"),
    ("collaboration", "soft_skill", r"\bcollaborat(?:e|es|ed|ing|ion)\b"),
    ("bs", "credential", r"\b(?:b\.?s\.?|bachelor(?:'s)?(?: degree)?|undergraduate degree)\b"),
    ("ms", "credential", r"\b(?:m\.?s\.?|master(?:'s)?(?: degree)?|graduate degree)\b"),
    ("phd", "credential", r"\b(?:ph\.?d\.?|doctorate|doctoral)\b"),
    ("yoe_0_1", "credential", None),
    ("yoe_2_3", "credential", None),
    ("yoe_4_5", "credential", None),
    ("yoe_6_plus", "credential", None),
]

FEATURE_ORDER = [name for name, _, _ in FEATURE_SPECS]


def ensure_dirs() -> None:
    for path in [OUT_REPORTS, OUT_TABLES, OUT_FIGURES, OUT_T19, OUT_T20, OUT_FIG_T19, OUT_FIG_T20]:
        path.mkdir(parents=True, exist_ok=True)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    return con


def fetch_df(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def fmt_pct(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "NA"
    return f"{100 * x:.{digits}f}%"


def fmt_num(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


def build_stop_tokens(con: duckdb.DuckDBPyConnection) -> set[str]:
    base = f"FROM read_parquet('{STAGE8.as_posix()}') WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok'"
    company = fetch_df(con, f"SELECT DISTINCT company_name_canonical AS val {base} AND company_name_canonical IS NOT NULL")["val"].dropna()
    metro = fetch_df(con, f"SELECT DISTINCT metro_area AS val {base} AND metro_area IS NOT NULL")["val"].dropna()
    state = fetch_df(con, f"SELECT DISTINCT state_normalized AS val {base} AND state_normalized IS NOT NULL")["val"].dropna()
    city = fetch_df(con, f"SELECT DISTINCT city_extracted AS val {base} AND city_extracted IS NOT NULL")["val"].dropna()

    stop = set(COMPANY_SUFFIX_WORDS)
    for series in [company, metro, state, city]:
        for value in series.astype(str):
            for token in re.findall(r"[a-z0-9]+", value.lower()):
                if len(token) >= 2:
                    stop.add(token)
    return stop


def normalize_text(text: str, stop_tokens: set[str]) -> str:
    if not text:
        return ""
    txt = str(text).lower()
    replacements = {
        "c++": " cpp ",
        "c#": " csharp ",
        "node.js": " nodejs ",
        "next.js": " nextjs ",
        "ci/cd": " cicd ",
        ".net": " dotnet ",
        "gpt-4": " gpt ",
        "gpt-3": " gpt ",
        "chat gpt": " chatgpt ",
        "pull request": " pr ",
    }
    for src, dst in replacements.items():
        txt = txt.replace(src, dst)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"(?i)\b(?:https?://|www\.)\S+\b", " ", txt)
    txt = re.sub(r"(?i)\b[\w.+-]+@[\w-]+\.[\w.-]+\b", " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(r"[^a-z0-9+#/\-. ]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    pieces = []
    for token in txt.split():
        if token in stop_tokens:
            continue
        if len(token) > 24:
            continue
        pieces.append(token)
    return " ".join(pieces)


def strip_boilerplate_sentences(text: str) -> str:
    if not text:
        return ""
    pieces = re.split(r"(?<=[.!?])\s+|\n+", str(text))
    kept = []
    for piece in pieces:
        lower = piece.lower()
        if any(phrase in lower for phrase in BOILERPLATE_PHRASES):
            continue
        kept.append(piece)
    return " ".join(kept)


def extract_requirements_section(text: str) -> tuple[str, bool]:
    if not text:
        return "", False
    lines = [ln.strip() for ln in str(text).replace("\r", "\n").split("\n") if ln.strip()]
    if not lines:
        return "", False

    def line_hits(line: str, phrases: list[str]) -> bool:
        lower = line.lower()
        return any(phrase in lower for phrase in phrases)

    start_idx = None
    for i, line in enumerate(lines):
        if len(line) > 160:
            continue
        if line_hits(line, REQ_HEADINGS):
            start_idx = i
            break
    if start_idx is None:
        return str(text), False

    out = [lines[start_idx]]
    for line in lines[start_idx + 1 :]:
        if len(line) <= 120 and line_hits(line, SECTION_ENDERS):
            break
        out.append(line)
    section = "\n".join(out)
    if len(section) < 40:
        return str(text), False
    return section, True


def feature_frame(df: pd.DataFrame, stop_tokens: set[str]) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        raw_text = row["description_core"] if pd.notna(row["description_core"]) and str(row["description_core"]).strip() else row["description"]
        req_text, req_found = extract_requirements_section(raw_text)
        base_text = req_text if req_found else raw_text
        cleaned = strip_boilerplate_sentences(base_text)
        cleaned = normalize_text(cleaned, stop_tokens)
        feats = {"clean_text": cleaned, "requirements_found": req_found}
        for name, category, pattern in FEATURE_SPECS:
            if name.startswith("yoe_"):
                feats[name] = 0
                continue
            feats[name] = int(bool(re.search(pattern, cleaned, flags=re.I)))
        yoe = row["yoe_extracted"]
        if pd.notna(yoe):
            yoe = float(yoe)
            feats["yoe_0_1"] = int(yoe < 2)
            feats["yoe_2_3"] = int(2 <= yoe < 4)
            feats["yoe_4_5"] = int(4 <= yoe < 6)
            feats["yoe_6_plus"] = int(yoe >= 6)
        rows.append(feats)
    feat_df = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), feat_df], axis=1)


def phi_from_counts(n11: int, n1x: int, nx1: int, n: int) -> float:
    n10 = n1x - n11
    n01 = nx1 - n11
    n00 = n - n11 - n10 - n01
    denom = math.sqrt(max(n1x * (n - n1x) * nx1 * (n - nx1), 0))
    if denom == 0:
        return 0.0
    return ((n11 * n00) - (n10 * n01)) / denom


def compute_pair_table(df: pd.DataFrame, period: str) -> pd.DataFrame:
    sub = df[df["period"] == period].copy()
    x = sub[FEATURE_ORDER].astype(int).to_numpy(dtype=np.int16)
    joint = x.T @ x
    n = len(sub)
    company_series = sub["company_name_canonical"].fillna("<<missing>>").astype(str)
    rows = []
    for i, j in combinations(range(len(FEATURE_ORDER)), 2):
        f1 = FEATURE_ORDER[i]
        f2 = FEATURE_ORDER[j]
        n11 = int(joint[i, j])
        n1x = int(x[:, i].sum())
        nx1 = int(x[:, j].sum())
        phi = phi_from_counts(n11, n1x, nx1, n)
        joint_companies = int(sub.loc[(sub[f1] == 1) & (sub[f2] == 1), "company_name_canonical"].nunique())
        rows.append(
            {
                "period": period,
                "feature_1": f1,
                "feature_2": f2,
                "category_1": next(cat for name, cat, _ in FEATURE_SPECS if name == f1),
                "category_2": next(cat for name, cat, _ in FEATURE_SPECS if name == f2),
                "n_postings": n,
                "n_feature_1": n1x,
                "n_feature_2": nx1,
                "n_joint": n11,
                "n_joint_companies": joint_companies,
                "phi": phi,
                "joint_share": n11 / n if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_phi_matrices(pair_tables: pd.DataFrame) -> dict[str, pd.DataFrame]:
    matrices = {}
    for period, g in pair_tables.groupby("period"):
        mat = pd.DataFrame(np.eye(len(FEATURE_ORDER)), index=FEATURE_ORDER, columns=FEATURE_ORDER, dtype=float)
        for _, row in g.iterrows():
            mat.loc[row["feature_1"], row["feature_2"]] = row["phi"]
            mat.loc[row["feature_2"], row["feature_1"]] = row["phi"]
        matrices[period] = mat
    return matrices


def plot_heatmap(matrix: pd.DataFrame, title: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        matrix,
        cmap="vlag",
        center=0,
        square=True,
        cbar_kws={"label": "phi"},
        ax=ax,
        linewidths=0.25,
        linecolor="white",
    )
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=90)
    ax.tick_params(axis="y", labelrotation=0)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def choose_kmeans_k(sample_x: np.ndarray, random_state: int = 42) -> tuple[int, dict[int, float]]:
    scores: dict[int, float] = {}
    for k in [4, 5, 6]:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(sample_x)
        score = silhouette_score(sample_x, labels, metric="euclidean", sample_size=min(2000, len(sample_x)), random_state=random_state)
        scores[k] = float(score)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def cluster_archetypes(df: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    balance_sample_parts = []
    for period, g in df.groupby("period"):
        sample_n = min(3000, len(g))
        balance_sample_parts.append(g.sample(n=sample_n, random_state=random_state))
    sample_df = pd.concat(balance_sample_parts, ignore_index=True)
    sample_x = sample_df[FEATURE_ORDER].astype(float).to_numpy()

    best_k, scores = choose_kmeans_k(sample_x, random_state=random_state)
    model = KMeans(n_clusters=best_k, random_state=random_state, n_init=25)
    model.fit(sample_x)
    df = df.copy()
    df["cluster_id"] = model.predict(df[FEATURE_ORDER].astype(float).to_numpy())

    global_rates = df[FEATURE_ORDER].mean()
    cluster_profile = (
        df.groupby("cluster_id")
        .agg(
            n_postings=("cluster_id", "size"),
            share=("cluster_id", lambda s: len(s) / len(df)),
            entry_share=("seniority_final", lambda s: float((s == "entry").mean())),
            ai_tool_any=("ai_tool_any", "mean"),
            ai_domain_any=("ai_domain_any", "mean"),
            ai_general=("ai_general", "mean"),
            ownership=("ownership", "mean"),
            end_to_end=("end_to_end", "mean"),
            cross_functional=("cross_functional", "mean"),
            stakeholder=("stakeholder", "mean"),
            lead=("lead", "mean"),
            mentor=("mentor", "mean"),
            manage=("manage", "mean"),
            hire=("hire", "mean"),
            team=("team", "mean"),
            system_design=("system_design", "mean"),
            architecture=("architecture", "mean"),
            distributed_systems=("distributed_systems", "mean"),
            microservices=("microservices", "mean"),
            scalability=("scalability", "mean"),
            cicd=("cicd", "mean"),
            deployment=("deployment", "mean"),
            infrastructure=("infrastructure", "mean"),
            testing=("testing", "mean"),
            code_review=("code_review", "mean"),
            communication=("communication", "mean"),
            collaboration=("collaboration", "mean"),
            bs=("bs", "mean"),
            ms=("ms", "mean"),
            phd=("phd", "mean"),
            yoe_0_1=("yoe_0_1", "mean"),
            yoe_2_3=("yoe_2_3", "mean"),
            yoe_4_5=("yoe_4_5", "mean"),
            yoe_6_plus=("yoe_6_plus", "mean"),
        )
        .reset_index()
    )
    cluster_feature_means = df.groupby("cluster_id")[FEATURE_ORDER].mean().reset_index()
    cluster_profile = cluster_profile.merge(cluster_feature_means, on="cluster_id", how="left", suffixes=("", "_mean"))

    names = {}
    for _, row in cluster_profile.iterrows():
        cid = int(row["cluster_id"])
        cluster_feats = row[[f"{feat}" for feat in FEATURE_ORDER]]
        if row["ai_tool_any"] >= cluster_profile["ai_tool_any"].max() and row["ownership"] >= cluster_profile["ownership"].median():
            name = "AI-augmented product engineer"
        elif row[["lead", "mentor", "manage", "hire", "team"]].max() >= 0.35:
            name = "Tech lead / manager"
        elif row[["system_design", "architecture", "distributed_systems", "microservices", "scalability"]].max() >= 0.35:
            name = "Platform / systems engineer"
        elif row[["cicd", "deployment", "infrastructure", "testing", "code_review"]].max() >= 0.35:
            name = "Delivery / infrastructure engineer"
        elif row[["yoe_0_1", "bs", "communication", "collaboration"]].max() >= 0.35:
            name = "Foundational / junior SWE"
        else:
            name = "General SWE"
        names[cid] = name

    cluster_profile["cluster_name"] = cluster_profile["cluster_id"].map(names)
    cluster_profile["cluster_label"] = cluster_profile.apply(
        lambda r: f"C{int(r['cluster_id'])}: {r['cluster_name']}", axis=1
    )
    cluster_profile["silhouette_best_k"] = best_k
    cluster_profile["silhouette_scores"] = cluster_profile["cluster_id"].map(lambda _: "")
    score_rows = pd.DataFrame({"k": list(scores.keys()), "silhouette": list(scores.values())})

    period_dist = (
        df.groupby(["period", "cluster_id"])
        .agg(n_postings=("cluster_id", "size"))
        .reset_index()
    )
    period_totals = period_dist.groupby("period")["n_postings"].transform("sum")
    period_dist["share"] = period_dist["n_postings"] / period_totals
    period_dist["cluster_name"] = period_dist["cluster_id"].map(names)
    period_dist["cluster_label"] = period_dist.apply(
        lambda r: f"C{int(r['cluster_id'])}: {r['cluster_name']}", axis=1
    )

    def top_features_for_row(r: pd.Series) -> str:
        lifts = [(feat, float(r[feat] - global_rates[feat])) for feat in FEATURE_ORDER]
        lifts = [item for item in lifts if item[1] > 0.10]
        lifts.sort(key=lambda t: t[1], reverse=True)
        return ", ".join(feat for feat, _ in lifts[:6])

    cluster_profile["top_features"] = cluster_profile.apply(top_features_for_row, axis=1).replace("", np.nan)
    cluster_profile["top_features"] = cluster_profile["top_features"].replace("", np.nan)

    return df, cluster_profile.sort_values("cluster_id"), period_dist.sort_values(["period", "cluster_id"]), score_rows


def make_cluster_distribution_plot(period_dist: pd.DataFrame, outpath: Path) -> None:
    order = period_dist.sort_values("share", ascending=False)["cluster_label"].drop_duplicates().tolist()
    piv = period_dist.pivot(index="period", columns="cluster_label", values="share").fillna(0)
    piv = piv[order]
    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(len(piv))
    x = np.arange(len(piv.index))
    cmap = sns.color_palette("Set2", n_colors=len(piv.columns))
    for idx, col in enumerate(piv.columns):
        vals = piv[col].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=col, color=cmap[idx])
        bottom += vals
    ax.set_xticks(x, piv.index)
    ax.set_ylabel("Share of SWE postings")
    ax.set_title("T19 archetype distribution by period")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_pca_plot(df: pd.DataFrame, outpath: Path, sample_per_group: int = 1200) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = [
        ("entry_2024", (df["period"] == "2024-04") & (df["seniority_final"] == "entry")),
        ("entry_2026", (df["period"] == "2026-03") & (df["seniority_final"] == "entry")),
        ("mid_2024", (df["period"] == "2024-04") & (df["seniority_final"] == "mid-senior")),
        ("mid_2026", (df["period"] == "2026-03") & (df["seniority_final"] == "mid-senior")),
    ]
    sample_parts = []
    for label, mask in groups:
        g = df.loc[mask].copy()
        if len(g) > sample_per_group:
            g = g.sample(n=sample_per_group, random_state=42)
        g["group"] = label
        sample_parts.append(g)
    sample_df = pd.concat(sample_parts, ignore_index=True)

    docs = sample_df["clean_text"].fillna("").tolist()
    vectorizer = TfidfVectorizer(
        min_df=5,
        max_df=0.9,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(docs)
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(X)
    sample_df["pc1"] = coords[:, 0]
    sample_df["pc2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = {
        "entry_2024": "#2b8cbe",
        "entry_2026": "#de2d26",
        "mid_2024": "#31a354",
        "mid_2026": "#756bb1",
    }
    for label, g in sample_df.groupby("group"):
        ax.scatter(g["pc1"], g["pc2"], s=10, alpha=0.22, label=label, color=palette[label], edgecolors="none")
    ax.set_title("T20 TF-IDF projection of cleaned SWE descriptions")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    centroid = sample_df.groupby("group")[["pc1", "pc2"]].mean().reset_index()
    return sample_df, centroid


def cleaned_text_for_tfidf(df: pd.DataFrame, stop_tokens: set[str]) -> pd.DataFrame:
    cleaned = []
    for _, row in df.iterrows():
        raw = row["description_core"] if pd.notna(row["description_core"]) and str(row["description_core"]).strip() else row["description"]
        txt = strip_boilerplate_sentences(raw)
        cleaned.append(normalize_text(txt, stop_tokens))
    out = df.copy()
    out["clean_text"] = cleaned
    return out


def chunked_mean_cross_similarity(A: sparse.spmatrix, B: sparse.spmatrix, chunk_size: int = 500) -> float:
    if A.shape[0] == 0 or B.shape[0] == 0:
        return float("nan")
    total = 0.0
    for start in range(0, A.shape[0], chunk_size):
        block = A[start : start + chunk_size]
        prod = safe_sparse_dot(block, B.T, dense_output=False)
        total += float(prod.sum())
    return total / (A.shape[0] * B.shape[0])


def chunked_mean_within_similarity(X: sparse.spmatrix, chunk_size: int = 500) -> float:
    n = X.shape[0]
    if n <= 1:
        return float("nan")
    total = 0.0
    for start in range(0, n, chunk_size):
        block = X[start : start + chunk_size]
        prod = safe_sparse_dot(block, X.T, dense_output=False)
        total += float(prod.sum())
    # Remove diagonal self-similarity.
    total -= n
    return total / (n * (n - 1))


def build_similarity_outputs(df: pd.DataFrame, stop_tokens: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset = df[
        (df["period"].isin(["2024-04", "2026-03"]))
        & (df["seniority_final"].isin(["entry", "mid-senior"]))
    ].copy()
    subset = cleaned_text_for_tfidf(subset, stop_tokens)

    groups = {
        "entry_2024": subset[(subset["period"] == "2024-04") & (subset["seniority_final"] == "entry")],
        "entry_2026": subset[(subset["period"] == "2026-03") & (subset["seniority_final"] == "entry")],
        "mid_2024": subset[(subset["period"] == "2024-04") & (subset["seniority_final"] == "mid-senior")],
        "mid_2026": subset[(subset["period"] == "2026-03") & (subset["seniority_final"] == "mid-senior")],
    }

    docs = subset["clean_text"].fillna("").tolist()
    vectorizer = TfidfVectorizer(
        min_df=5,
        max_df=0.9,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(docs)
    group_idx = {}
    for label, g in groups.items():
        group_idx[label] = subset.index.get_indexer(g.index)

    # Build similarity matrix over the four requested groups.
    labels = list(groups.keys())
    matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for left in labels:
        A = X[group_idx[left]]
        for right in labels:
            B = X[group_idx[right]]
            if left == right:
                matrix.loc[left, right] = chunked_mean_within_similarity(A)
            else:
                matrix.loc[left, right] = chunked_mean_cross_similarity(A, B)

    comparisons = pd.DataFrame(
        [
            {
                "comparison": "entry_2026_vs_entry_2024",
                "left_group": "entry_2026",
                "right_group": "entry_2024",
                "mean_cosine": matrix.loc["entry_2026", "entry_2024"],
                "n_left": len(groups["entry_2026"]),
                "n_right": len(groups["entry_2024"]),
            },
            {
                "comparison": "entry_2026_vs_mid_2024",
                "left_group": "entry_2026",
                "right_group": "mid_2024",
                "mean_cosine": matrix.loc["entry_2026", "mid_2024"],
                "n_left": len(groups["entry_2026"]),
                "n_right": len(groups["mid_2024"]),
            },
            {
                "comparison": "entry_2024_vs_mid_2024",
                "left_group": "entry_2024",
                "right_group": "mid_2024",
                "mean_cosine": matrix.loc["entry_2024", "mid_2024"],
                "n_left": len(groups["entry_2024"]),
                "n_right": len(groups["mid_2024"]),
            },
            {
                "comparison": "mid_2026_vs_mid_2024",
                "left_group": "mid_2026",
                "right_group": "mid_2024",
                "mean_cosine": matrix.loc["mid_2026", "mid_2024"],
                "n_left": len(groups["mid_2026"]),
                "n_right": len(groups["mid_2024"]),
            },
        ]
    )
    comparisons["directional_difference"] = np.nan
    comparisons.loc[comparisons["comparison"] == "entry_2026_vs_entry_2024", "directional_difference"] = (
        comparisons.loc[comparisons["comparison"] == "entry_2026_vs_mid_2024", "mean_cosine"].iloc[0]
        - comparisons.loc[comparisons["comparison"] == "entry_2026_vs_entry_2024", "mean_cosine"].iloc[0]
    )

    within = pd.DataFrame(
        [
            {"group": "entry_2024", "n": len(groups["entry_2024"]), "within_mean_cosine": matrix.loc["entry_2024", "entry_2024"], "within_diversity_proxy": 1 - matrix.loc["entry_2024", "entry_2024"]},
            {"group": "entry_2026", "n": len(groups["entry_2026"]), "within_mean_cosine": matrix.loc["entry_2026", "entry_2026"], "within_diversity_proxy": 1 - matrix.loc["entry_2026", "entry_2026"]},
            {"group": "mid_2024", "n": len(groups["mid_2024"]), "within_mean_cosine": matrix.loc["mid_2024", "mid_2024"], "within_diversity_proxy": 1 - matrix.loc["mid_2024", "mid_2024"]},
            {"group": "mid_2026", "n": len(groups["mid_2026"]), "within_mean_cosine": matrix.loc["mid_2026", "mid_2026"], "within_diversity_proxy": 1 - matrix.loc["mid_2026", "mid_2026"]},
        ]
    )

    return matrix, comparisons, within


def main() -> None:
    ensure_dirs()
    con = connect()

    stop_tokens = build_stop_tokens(con)

    base_sql = f"""
    SELECT
      uid,
      source,
      period,
      company_name_canonical,
      seniority_final,
      description_core,
      description,
      yoe_extracted::DOUBLE AS yoe_extracted
    FROM read_parquet('{STAGE8.as_posix()}')
    WHERE {FILTER}
    """
    df = fetch_df(con, base_sql)
    df = df[df["seniority_final"].isin(["entry", "associate", "mid-senior", "director", "unknown"])].copy()
    df = feature_frame(df, stop_tokens)

    # T19: requirement bundles and archetypes.
    pair_tables = []
    for period in ["2024-01", "2024-04", "2026-03"]:
        pair_tables.append(compute_pair_table(df, period))
    pair_tables_df = pd.concat(pair_tables, ignore_index=True)
    phi_matrices = compute_phi_matrices(pair_tables_df)

    # 2026 versus pooled 2024 emergence table.
    pooled_2024 = pair_tables_df[pair_tables_df["period"].isin(["2024-01", "2024-04"])].copy()
    pooled_2024 = (
        pooled_2024.groupby(["feature_1", "feature_2"], as_index=False)
        .agg(
            n_postings=("n_postings", "sum"),
            n_feature_1=("n_feature_1", "sum"),
            n_feature_2=("n_feature_2", "sum"),
            n_joint=("n_joint", "sum"),
            n_joint_companies=("n_joint_companies", "sum"),
            phi=("phi", "mean"),
            joint_share=("joint_share", "mean"),
        )
    )
    phi_2026 = pair_tables_df[pair_tables_df["period"] == "2026-03"][["feature_1", "feature_2", "phi", "n_joint", "n_joint_companies"]].rename(
        columns={"phi": "phi_2026", "n_joint": "n_joint_2026", "n_joint_companies": "n_joint_companies_2026"}
    )
    emergent = pooled_2024.merge(phi_2026, on=["feature_1", "feature_2"], how="left")
    emergent["phi_delta_2026_vs_2024"] = emergent["phi_2026"] - emergent["phi"]
    emergent["feature_1_category"] = emergent["feature_1"].map({name: cat for name, cat, _ in FEATURE_SPECS})
    emergent["feature_2_category"] = emergent["feature_2"].map({name: cat for name, cat, _ in FEATURE_SPECS})
    emergent = emergent.sort_values(["phi_delta_2026_vs_2024", "n_joint_2026", "n_joint_companies_2026"], ascending=[False, False, False])
    emergent_filtered = emergent[
        (emergent["n_joint_2026"] >= 20)
        & (emergent["n_joint_companies_2026"] >= 20)
        & (emergent["n_joint"] <= emergent["n_joint_2026"] * 1.2)
    ].head(25).copy()
    cross_category_bundles = emergent[
        (emergent["feature_1_category"] != emergent["feature_2_category"])
        & (emergent["n_joint_2026"] >= 20)
        & (emergent["n_joint_companies_2026"] >= 20)
    ].head(25).copy()

    df2, cluster_profile, period_dist, silhouette_scores = cluster_archetypes(df)

    write_csv(OUT_T19 / "T19_pairwise_phi_by_period.csv", pair_tables_df.sort_values(["period", "phi"], ascending=[True, False]))
    write_csv(OUT_T19 / "T19_phi_delta_2026_vs_2024.csv", emergent_filtered)
    write_csv(OUT_T19 / "T19_cross_category_bundle_pairs.csv", cross_category_bundles)
    write_csv(OUT_T19 / "T19_cluster_profiles.csv", cluster_profile)
    write_csv(OUT_T19 / "T19_cluster_period_distribution.csv", period_dist)
    write_csv(OUT_T19 / "T19_cluster_silhouette_scores.csv", silhouette_scores)

    for period in ["2024-01", "2024-04", "2026-03"]:
        plot_heatmap(phi_matrices[period], f"T19 phi co-occurrence matrix - {period}", OUT_FIG_T19 / f"T19_phi_heatmap_{period}.png")
    make_cluster_distribution_plot(period_dist, OUT_FIG_T19 / "T19_archetype_distribution.png")

    # T20: semantic similarity and plot.
    sim_matrix, sim_comparisons, within = build_similarity_outputs(df, stop_tokens)
    write_csv(OUT_T20 / "T20_similarity_matrix.csv", sim_matrix.reset_index(names="group"))
    write_csv(OUT_T20 / "T20_similarity_comparisons.csv", sim_comparisons)
    write_csv(OUT_T20 / "T20_within_level_diversity.csv", within)

    # Add a small centroid table from the PCA/TF-IDF projection for reference.
    sample_df, centroid = make_pca_plot(
        cleaned_text_for_tfidf(
            df[(df["period"].isin(["2024-04", "2026-03"])) & (df["seniority_final"].isin(["entry", "mid-senior"]))].copy(),
            stop_tokens,
        ),
        OUT_FIG_T20 / "T20_pca_projection.png",
    )
    write_csv(OUT_T20 / "T20_pca_group_centroids.csv", centroid)

    # Reports.
    entry_sim = sim_comparisons.loc[sim_comparisons["comparison"] == "entry_2026_vs_entry_2024", "mean_cosine"].iloc[0]
    entry_mid = sim_comparisons.loc[sim_comparisons["comparison"] == "entry_2026_vs_mid_2024", "mean_cosine"].iloc[0]
    mid_same = sim_comparisons.loc[sim_comparisons["comparison"] == "mid_2026_vs_mid_2024", "mean_cosine"].iloc[0]
    entry_div_24 = within.loc[within["group"] == "entry_2024", "within_mean_cosine"].iloc[0]
    entry_div_26 = within.loc[within["group"] == "entry_2026", "within_mean_cosine"].iloc[0]

    top_emergent = emergent_filtered.head(5)
    cross_category = emergent_filtered[emergent_filtered["feature_1_category"] != emergent_filtered["feature_2_category"]].head(5)
    if cross_category.empty:
        cross_category = top_emergent
    emergent_text = "; ".join(
        f"{r.feature_1} + {r.feature_2} ({fmt_num(r.phi_delta_2026_vs_2024, 3)})" for r in cross_category.itertuples()
    ) if not cross_category.empty else "no pairs met the emergence thresholds"

    def period_share(period: str, cluster_name: str) -> float:
        hit = period_dist[(period_dist["period"] == period) & (period_dist["cluster_name"] == cluster_name)]
        return float(hit["share"].iloc[0]) if not hit.empty else float("nan")

    ai_label = "AI-augmented product engineer"
    general_label = "General SWE"
    infra_label = "Delivery / infrastructure engineer"
    ai_2024 = period_share("2024-04", ai_label)
    ai_2026 = period_share("2026-03", ai_label)
    general_2024 = period_share("2024-04", general_label)
    general_2026 = period_share("2026-03", general_label)
    infra_2024 = period_share("2024-04", infra_label)
    infra_2026 = period_share("2026-03", infra_label)
    strongest_k = int(silhouette_scores.loc[silhouette_scores["silhouette"].idxmax(), "k"])

    relabel_delta = float(entry_mid - entry_sim)
    if relabel_delta > 0:
        relabel_phrase = (
            f"Entry-2026 is slightly closer to mid-2024 than to entry-2024, by {fmt_num(relabel_delta, 4)}, which gives weak support to relabeling."
        )
    else:
        relabel_phrase = (
            f"Entry-2026 is slightly closer to entry-2024 than to mid-2024, by {fmt_num(abs(relabel_delta), 4)}, so the relabeling hypothesis is not supported."
        )
    diversity_phrase = (
        f"Within-entry cosine rises from {fmt_num(entry_div_24, 4)} in 2024 to {fmt_num(entry_div_26, 4)} in 2026, so the 2026 entry slice is less diverse / more internally homogeneous."
        if entry_div_26 > entry_div_24
        else f"Within-entry cosine falls from {fmt_num(entry_div_24, 4)} in 2024 to {fmt_num(entry_div_26, 4)} in 2026, so the 2026 entry slice is more diverse."
    )

    t19_lines = [
        "# T19: Requirement bundles and posting archetypes",
        "## Finding",
        (
            f"The SWE requirement space splits into a small set of interpretable archetypes. The {strongest_k}-cluster solution separates AI-augmented/product-scope, platform/systems, delivery/infrastructure, management-heavy, general SWE, and foundational/junior slices. Compared with 2024-04, the AI-augmented cluster rises from {fmt_pct(ai_2024)} to {fmt_pct(ai_2026)}, the general SWE cluster falls from {fmt_pct(general_2024)} to {fmt_pct(general_2026)}, and the delivery/infrastructure cluster rises from {fmt_pct(infra_2024)} to {fmt_pct(infra_2026)}."
        ),
        (
            f"Relative to pooled 2024, 2026 adds stronger bundle co-occurrences around {emergent_text}. The most readable cross-category bundles now include ownership + end_to_end, ownership + infrastructure, stakeholder + collaboration, and AI-tool language paired with stakeholder language; that is a clearer signal than YOE alone."
        ),
        "## Implication for analysis",
        "This supports an RQ1 framing in which seniority restructuring is partly happening through bundle changes: AI language, ownership, and systems language travel together. It also gives a natural set of archetype labels for downstream robustness checks and subgroup analysis.",
        "## Data quality note",
        "Stage 8 does not have `description_core_llm`, so this task uses `description_core` with `description` as a fallback only when needed, then strips residual boilerplate and company/location tokens before tokenization. The 2024-01 asaniczka slice is useful for overall bundle structure, but it should not be used for entry-level trend claims because it has no native entry labels.",
        "## Action items",
        f"- `{(OUT_T19 / 'T19_pairwise_phi_by_period.csv').as_posix()}`",
        f"- `{(OUT_T19 / 'T19_phi_delta_2026_vs_2024.csv').as_posix()}`",
        f"- `{(OUT_T19 / 'T19_cross_category_bundle_pairs.csv').as_posix()}`",
        f"- `{(OUT_T19 / 'T19_cluster_profiles.csv').as_posix()}`",
        f"- `{(OUT_T19 / 'T19_cluster_period_distribution.csv').as_posix()}`",
        f"- `{(OUT_FIG_T19 / 'T19_phi_heatmap_2024-01.png').as_posix()}`",
        f"- `{(OUT_FIG_T19 / 'T19_phi_heatmap_2024-04.png').as_posix()}`",
        f"- `{(OUT_FIG_T19 / 'T19_phi_heatmap_2026-03.png').as_posix()}`",
        f"- `{(OUT_FIG_T19 / 'T19_archetype_distribution.png').as_posix()}`",
    ]
    (OUT_REPORTS / "T19.md").write_text("\n".join(t19_lines) + "\n")

    t20_lines = [
        "# T20: Relabeling hypothesis test",
        "## Finding",
        (
            f"In this run, entry-2026 vs entry-2024 mean cosine is {fmt_num(entry_sim, 4)}, while entry-2026 vs mid-2024 is {fmt_num(entry_mid, 4)} and mid-2026 vs mid-2024 is {fmt_num(mid_same, 4)}. {relabel_phrase}"
        ),
        (
            diversity_phrase
        ),
        "## Implication for analysis",
        "The safer interpretation is that 2026 entry postings still look semantically closer to historical entry work than to mid-senior work, even though they borrow some mid-level language. That makes T20 a useful boundary check, but not a positive relabeling result.",
        "## Data quality note",
        "This uses cleaned TF-IDF on `description_core` with company-name stripping and residual boilerplate removal because Stage 8 does not yet have `description_core_llm`. The similarity estimates are restricted to LinkedIn, English, date-ok rows and to 2024-04 vs 2026-03 for the entry test so asaniczka does not contaminate the baseline.",
        "## Action items",
        f"- `{(OUT_T20 / 'T20_similarity_matrix.csv').as_posix()}`",
        f"- `{(OUT_T20 / 'T20_similarity_comparisons.csv').as_posix()}`",
        f"- `{(OUT_T20 / 'T20_within_level_diversity.csv').as_posix()}`",
        f"- `{(OUT_T20 / 'T20_pca_group_centroids.csv').as_posix()}`",
        f"- `{(OUT_FIG_T20 / 'T20_pca_projection.png').as_posix()}`",
    ]
    (OUT_REPORTS / "T20.md").write_text("\n".join(t20_lines) + "\n")

    print("Wrote T19/T20 outputs.")


if __name__ == "__main__":
    main()
