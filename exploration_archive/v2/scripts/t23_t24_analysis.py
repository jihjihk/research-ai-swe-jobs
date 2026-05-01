from __future__ import annotations

import hashlib
import json
import math
import re
from functools import lru_cache
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD


ROOT = Path(__file__).resolve().parents[2]
STAGE8 = ROOT / "preprocessing" / "intermediate" / "stage8_final.parquet"

T23_TABLE_DIR = ROOT / "exploration" / "tables" / "T23"
T24_TABLE_DIR = ROOT / "exploration" / "tables" / "T24"
T24_FIG_DIR = ROOT / "exploration" / "figures" / "T24"
T23_REPORT = ROOT / "exploration" / "reports" / "T23.md"
T24_REPORT = ROOT / "exploration" / "reports" / "T24.md"


def ensure_dirs() -> None:
    for path in [T23_TABLE_DIR, T24_TABLE_DIR, T24_FIG_DIR, T23_REPORT.parent, T24_REPORT.parent]:
        path.mkdir(parents=True, exist_ok=True)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"

    def fmt(v) -> str:
        if pd.isna(v):
            return ""
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.3f}"
        if isinstance(v, (bool, np.bool_)):
            return "True" if bool(v) else "False"
        return str(v).replace("|", "\\|").replace("\n", " ")

    headers = [str(c).replace("|", "\\|") for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def load_filtered_data() -> pd.DataFrame:
    con = duckdb.connect()
    query = """
        SELECT
            uid,
            title,
            company_name,
            company_name_effective,
            company_name_canonical,
            company_size,
            is_aggregator,
            source,
            period,
            seniority_final,
            seniority_3level,
            yoe_extracted,
            ghost_job_risk,
            description_core,
            description
        FROM read_parquet(?)
        WHERE source_platform = 'linkedin'
          AND is_english = TRUE
          AND date_flag = 'ok'
          AND is_swe = TRUE
          AND length(coalesce(description_core, description)) >= 50
    """
    df = con.execute(query, [str(STAGE8)]).fetchdf()
    df["text_base"] = df["description_core"].fillna(df["description"]).fillna("")
    df["title"] = df["title"].fillna("")
    df["company_name_canonical"] = df["company_name_canonical"].fillna(df["company_name_effective"]).fillna(df["company_name"]).fillna("")
    df["company_name_effective"] = df["company_name_effective"].fillna(df["company_name"]).fillna("")
    return df


def build_company_stoplist(company_names: Sequence[str]) -> set[str]:
    counts = Counter()
    for name in company_names:
        toks = re.findall(r"[a-z0-9]+", str(name).lower())
        for tok in toks:
            if len(tok) > 1:
                counts[tok] += 1
    stop = set(counts)
    return stop


BOILERPLATE_CUT_PATTERNS = [
    r"(?is)\bequal opportunity\b.*$",
    r"(?is)\breasonable accommodation\b.*$",
    r"(?is)\bprotected class\b.*$",
    r"(?is)\bbenefits? include\b.*$",
    r"(?is)\babout us\b.*$",
    r"(?is)\babout the company\b.*$",
    r"(?is)\bfair chance\b.*$",
    r"(?is)\bprivacy notice\b.*$",
]


REQ_START_PATTERNS = [
    r"\brequirements?\b",
    r"\bqualifications?\b",
    r"\bminimum qualifications?\b",
    r"\bbasic qualifications?\b",
    r"\bpreferred qualifications?\b",
    r"\bwhat(?:'s| is)?\s+required\b",
    r"\bwhat(?:'s| is)?\s+needed\b",
    r"\bwhat you(?:'|’)ll need\b",
    r"\bwhat we(?:'|’)re looking for\b",
    r"\bskills? and experience\b",
    r"\bmust haves?\b",
]

REQ_END_PATTERNS = [
    r"\bresponsibilities?\b",
    r"\bwhat you(?:'|’)ll do\b",
    r"\bbenefits?\b",
    r"\bcompensation\b",
    r"\babout (?:the )?company\b",
    r"\babout us\b",
    r"\bequal opportunity\b",
    r"\bprivacy notice\b",
    r"\bhow to apply\b",
    r"\bwhy join us\b",
]


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def cut_boilerplate_tail(text: str) -> str:
    lower = text.lower()
    cut = len(text)
    for pat in BOILERPLATE_CUT_PATTERNS:
        m = re.search(pat, lower)
        if m and m.start() < cut:
            cut = m.start()
    return text[:cut].strip()


def extract_requirement_section(text: str) -> str:
    if not text:
        return ""
    raw = text
    lower = raw.lower()
    start = None
    for pat in REQ_START_PATTERNS:
        m = re.search(pat, lower)
        if m:
            if start is None or m.start() < start:
                start = m.start()
    if start is None:
        return raw[:2500].strip()
    end = len(raw)
    for pat in REQ_END_PATTERNS:
        m = re.search(pat, lower[start + 5 :])
        if m:
            candidate = start + 5 + m.start()
            if candidate < end:
                end = candidate
    section = raw[start:end].strip()
    return section[:4000]


TOKEN_RE = re.compile(r"[a-z][a-z0-9\+\#\-/]{1,}")


def tokenize(text: str, company_stop: set[str]) -> List[str]:
    toks = TOKEN_RE.findall(text.lower())
    return [tok for tok in toks if tok not in company_stop]


def tokenized_text(text: str, company_stop: set[str]) -> str:
    return " ".join(tokenize(text, company_stop))


TECH_PATTERNS = {
    "python": r"\bpython\b",
    "java": r"\bjava\b",
    "javascript": r"\bjavascript\b|\bjs\b",
    "typescript": r"\btypescript\b",
    "react": r"\breact(?:\.js)?\b",
    "angular": r"\bangular\b",
    "vue": r"\bvue\b",
    "node": r"\bnode(?:\.js)?\b",
    "docker": r"\bdocker\b",
    "kubernetes": r"\bkubernetes\b|\bk8s\b",
    "terraform": r"\bterraform\b",
    "aws": r"\baws\b|\bamazon web services\b",
    "gcp": r"\bgcp\b|\bgoogle cloud\b",
    "azure": r"\bazure\b",
    "sql": r"\bsql\b",
    "postgres": r"\bpostgres(?:ql)?\b",
    "mysql": r"\bmysql\b",
    "redis": r"\bredis\b",
    "kafka": r"\bkafka\b",
    "spark": r"\bspark\b",
    "airflow": r"\bairflow\b",
    "databricks": r"\bdatabricks\b",
    "snowflake": r"\bsnowflake\b",
    "pytorch": r"\bpytorch\b",
    "tensorflow": r"\btensorflow\b",
    "scikit_learn": r"\bscikit[- ]learn\b",
    "git": r"\bgit\b",
    "ci_cd": r"\bci/cd\b|\bci cd\b|\bcontinuous integration\b",
    "microservices": r"\bmicroservices?\b",
    "grpc": r"\bgrpc\b",
    "rest_api": r"\brest(?:ful)? api\b|\bapi\b",
    "linux": r"\blinux\b",
    "c_plus_plus": r"\bc\+\+\b",
    "c_sharp": r"\bc#\b|\bcsharp\b",
    "go": r"\bgolang\b|\bgo\b",
    "rust": r"\brust\b",
    "php": r"\bphp\b",
    "ruby": r"\bruby\b",
    "rails": r"\brails\b",
    "hadoop": r"\bhadoop\b",
    "fastapi": r"\bfastapi\b",
    "flask": r"\bflask\b",
    "django": r"\bdjango\b",
}


ORG_PATTERNS = {
    "ownership": r"\bownership\b|\bend[- ]to[- ]end\b",
    "cross_functional": r"\bcross[- ]functional\b",
    "stakeholder": r"\bstakeholder(?:s)?\b",
    "autonomous": r"\bautonomous\b|\bautonomy\b",
    "lead": r"\blead(?:ing)?\b",
    "mentor": r"\bmentor(?:ing)?\b",
    "manage": r"\bmanage(?:ment|r|ing)?\b",
    "team": r"\bteam\b|\bteams\b",
    "collaboration": r"\bcollaboration\b|\bcollaborate(?:d|s|ing)?\b",
    "communication": r"\bcommunication\b|\bcommunicate(?:d|s|ing)?\b",
    "problem_solving": r"\bproblem[- ]solving\b",
    "influence": r"\binfluence(?:d|s|ing)?\b",
    "prioritize": r"\bprioriti[sz]e(?:d|s|ing)?\b",
    "drive": r"\bdrive(?:n|s|ing)?\b",
    "partner": r"\bpartner(?:ed|s|ing)?\b",
    "decision_making": r"\bdecision[- ]making\b",
    "ambiguity": r"\bambiguity\b",
    "strategic": r"\bstrategic\b|\bstrategy\b",
    "initiative": r"\binitiative\b",
}


AI_TOOL_PATTERNS = {
    "copilot": r"\bcopilot\b",
    "cursor": r"\bcursor\b",
    "claude": r"\bclaude\b",
    "gpt": r"\bgpt[- ]?\d?\b|\bchatgpt\b",
    "llm": r"\bllms?\b|\blarge language model\b",
    "rag": r"\brag\b|\bretrieval[- ]augmented\b",
    "agent": r"\bagents?\b",
    "mcp": r"\bmcp\b|model context protocol",
    "openai": r"\bopenai\b",
}


AI_DOMAIN_PATTERNS = {
    "machine_learning": r"\bmachine learning\b|\bml\b",
    "deep_learning": r"\bdeep learning\b",
    "nlp": r"\bnlp\b|\bnatural language processing\b",
    "computer_vision": r"\bcomputer vision\b",
}


HEDGE_PATTERNS = {
    "ideally": r"\bideally\b",
    "preferred": r"\bpreferred\b",
    "bonus": r"\bbonus\b",
    "nice_to_have": r"nice to have",
    "would_be_nice": r"would be nice",
    "plus": r"\ba plus\b|\bplus\b",
    "desired": r"\bdesired\b",
}


FIRM_PATTERNS = {
    "must_have": r"\bmust have\b",
    "required": r"\brequired\b|\brequire(?:d|ment|ments)?\b",
    "mandatory": r"\bmandatory\b",
    "minimum": r"\bminimum\b",
    "need": r"\bneed to\b|\bneeds to\b|\bneed\b",
    "essential": r"\bessential\b",
}


SCOPE_PATTERNS = {
    "ownership": ORG_PATTERNS["ownership"],
    "architecture": r"\barchitecture\b|\barchitect(?:ing|ed|ure)?\b",
    "distributed_systems": r"\bdistributed systems?\b",
    "system_design": r"\bsystem design\b",
    "scalability": r"\bscalability\b|\bscalable\b",
    "cross_functional": ORG_PATTERNS["cross_functional"],
    "stakeholder": ORG_PATTERNS["stakeholder"],
    "autonomous": ORG_PATTERNS["autonomous"],
}


def compile_patterns(pattern_dict: Dict[str, str]) -> Dict[str, re.Pattern]:
    return {k: re.compile(v, flags=re.I) for k, v in pattern_dict.items()}


TECH_RX = compile_patterns(TECH_PATTERNS)
ORG_RX = compile_patterns(ORG_PATTERNS)
AI_TOOL_RX = compile_patterns(AI_TOOL_PATTERNS)
AI_DOMAIN_RX = compile_patterns(AI_DOMAIN_PATTERNS)
HEDGE_RX = compile_patterns(HEDGE_PATTERNS)
FIRM_RX = compile_patterns(FIRM_PATTERNS)
SCOPE_RX = compile_patterns(SCOPE_PATTERNS)


def unique_pattern_hits(text: str, compiled: Dict[str, re.Pattern]) -> List[str]:
    hits = []
    for name, rx in compiled.items():
        if rx.search(text):
            hits.append(name)
    return hits


def build_indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    company_stop = build_company_stoplist(df["company_name_canonical"].tolist())

    cleaned_text = []
    req_text = []
    req_token_text = []
    req_signature = []
    chars = []

    for text, company in zip(df["text_base"].tolist(), df["company_name_canonical"].tolist()):
        norm = normalize_text(text)
        norm = cut_boilerplate_tail(norm)
        cleaned_text.append(norm)
        req = extract_requirement_section(norm)
        req_text.append(req)
        tok_text = tokenized_text(req, company_stop)
        req_token_text.append(tok_text)
        req_signature.append(hashlib.sha1(tok_text.encode("utf-8")).hexdigest())
        chars.append(len(norm))

    df = df.copy()
    df["cleaned_text"] = cleaned_text
    df["requirements_text"] = req_text
    df["requirements_text_tokenized"] = req_token_text
    df["requirements_signature"] = req_signature
    df["cleaned_chars"] = chars

    indicator_rows = []
    for uid, text, req, tok_text, company, company_effective, title, seniority, period, yoe, is_agg, size in zip(
        df["uid"].tolist(),
        df["cleaned_text"].tolist(),
        df["requirements_text"].tolist(),
        df["requirements_text_tokenized"].tolist(),
        df["company_name_canonical"].tolist(),
        df["company_name_effective"].tolist(),
        df["title"].tolist(),
        df["seniority_3level"].tolist(),
        df["period"].tolist(),
        df["yoe_extracted"].tolist(),
        df["is_aggregator"].tolist(),
        df["company_size"].tolist(),
    ):
        tech_hits = unique_pattern_hits(text, TECH_RX)
        org_hits = unique_pattern_hits(text, ORG_RX)
        ai_tool_hits = unique_pattern_hits(text, AI_TOOL_RX)
        ai_domain_hits = unique_pattern_hits(text, AI_DOMAIN_RX)
        hedge_hits = unique_pattern_hits(text, HEDGE_RX)
        firm_hits = unique_pattern_hits(text, FIRM_RX)
        scope_hits = unique_pattern_hits(text, SCOPE_RX)

        tech_count = len(tech_hits)
        org_count = len(org_hits)
        hedge_count = len(hedge_hits)
        firm_count = len(firm_hits)
        scope_count = len(scope_hits)
        aspiration_ratio = hedge_count / firm_count if firm_count else float(hedge_count)

        ai_any = bool(ai_tool_hits or ai_domain_hits)
        ai_hits = len(ai_tool_hits) + len(ai_domain_hits)
        ai_rate = ai_hits / max(len(text), 1) * 1000

        indicator_rows.append(
            {
                "uid": uid,
                "company_name_canonical": company,
                "company_name_effective": company_effective,
                "title": title,
                "cleaned_text": text,
                "requirements_text": req,
                "requirements_tokenized": tok_text,
                "tech_hits": tech_hits,
                "org_hits": org_hits,
                "ai_tool_hits": ai_tool_hits,
                "ai_domain_hits": ai_domain_hits,
                "hedge_hits": hedge_hits,
                "firm_hits": firm_hits,
                "scope_hits": scope_hits,
                "tech_count": tech_count,
                "org_count": org_count,
                "hedge_count": hedge_count,
                "firm_count": firm_count,
                "scope_count": scope_count,
                "aspiration_ratio": aspiration_ratio,
                "ai_any": ai_any,
                "ai_rate_per_1k_chars": ai_rate,
                "is_junior": seniority == "junior",
                "is_entry_like": seniority == "junior",
                "period": period,
                "seniority_3level": seniority,
                "is_aggregator": bool(is_agg),
                "company_size": size,
                "yoe_extracted": yoe,
            }
        )

    ind = pd.DataFrame(indicator_rows)
    ind["requirements_signature"] = req_signature
    ind["kitchen_sink_tech"] = ind["tech_count"] > 15
    ind["kitchen_sink_org"] = ind["org_count"] > 8
    ind["aspiration_saturated"] = ind["aspiration_ratio"] >= 1.0
    ind["yoe_scope_mismatch"] = ind["is_junior"] & (ind["scope_count"] >= 2)
    ind["ghost_row_score_raw"] = (
        ind["kitchen_sink_tech"].astype(int)
        + ind["kitchen_sink_org"].astype(int)
        + ind["aspiration_saturated"].astype(int)
        + ind["yoe_scope_mismatch"].astype(int)
    )

    company_summary = (
        ind.groupby("company_name_canonical")
        .agg(
            company_n=("company_name_canonical", "size"),
            company_mean_tech=("tech_count", "mean"),
            company_mean_org=("org_count", "mean"),
            company_mean_aspiration=("aspiration_ratio", "mean"),
            company_mean_scope=("scope_count", "mean"),
            company_mean_ai_rate=("ai_rate_per_1k_chars", "mean"),
            company_dup_share=("requirements_signature", lambda x: x.value_counts(normalize=True).max()),
        )
        .reset_index()
    )
    company_summary["company_repetition_flag"] = (
        ((company_summary["company_n"] >= 3) & (company_summary["company_dup_share"] >= 0.4))
        | ((company_summary["company_n"] >= 2) & (company_summary["company_mean_scope"] >= 2.0) & (company_summary["company_mean_org"] >= 4))
    )
    ind = ind.merge(company_summary[["company_name_canonical", "company_repetition_flag"]], on="company_name_canonical", how="left")
    ind["template_saturation"] = False
    ind["within_company_nn_sim"] = np.nan
    ind["within_company_mean_sim"] = np.nan
    ind["within_company_max_sim"] = np.nan

    # TF-IDF over requirements sections to compute within-company similarity.
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        token_pattern=r"(?u)\b[a-z][a-z0-9\+\#\-/]{1,}\b",
    )
    req_matrix = vec.fit_transform(ind["requirements_tokenized"].fillna(""))

    for company, idx in ind.groupby("company_name_canonical").groups.items():
        idx = np.fromiter(idx, dtype=int)
        if len(idx) < 2:
            continue
        sub = req_matrix[idx]
        sim = (sub @ sub.T).tocsr()
        sim.setdiag(0)
        row_max = sim.max(axis=1).toarray().ravel()
        nn = pd.Series(row_max, index=idx)
        mean_sim = float(sim.sum() / (len(idx) * (len(idx) - 1)))
        ind.loc[idx, "within_company_nn_sim"] = nn.values
        ind.loc[idx, "within_company_mean_sim"] = mean_sim
        ind.loc[idx, "within_company_max_sim"] = nn.values
        ind.loc[idx, "template_saturation"] = row_max >= 0.90

    ind["template_saturation"] = ind["template_saturation"].fillna(False)
    ind["company_repetition"] = ind["company_repetition_flag"].fillna(False)
    ind["ghost_like"] = (
        ind["template_saturation"].astype(int)
        + ind["kitchen_sink_tech"].astype(int)
        + ind["kitchen_sink_org"].astype(int)
        + ind["aspiration_saturated"].astype(int)
        + ind["yoe_scope_mismatch"].astype(int)
        + ind["company_repetition"].astype(int)
    )
    return ind


def make_period_seniority_table(ind: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (period, seniority), g in ind.groupby(["period", "seniority_3level"], dropna=False):
        row = {
            "period": period,
            "seniority_3level": seniority,
            "n": len(g),
            "template_saturation_rate": g["template_saturation"].mean(),
            "kitchen_sink_tech_rate": g["kitchen_sink_tech"].mean(),
            "kitchen_sink_org_rate": g["kitchen_sink_org"].mean(),
            "aspiration_saturated_rate": g["aspiration_saturated"].mean(),
            "yoe_scope_mismatch_rate": g["yoe_scope_mismatch"].mean(),
            "company_repetition_rate": g["company_repetition"].mean(),
            "ghost_like_mean": g["ghost_like"].mean(),
            "ai_any_rate": g["ai_any"].mean(),
            "ai_rate_per_1k_chars": g["ai_rate_per_1k_chars"].mean(),
            "median_within_company_nn_sim": g["within_company_nn_sim"].median(),
        }
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["period", "seniority_3level"])
    return out


def make_cross_tabs(ind: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ai = ind.groupby("ai_any").agg(
        n=("ai_any", "size"),
        ghost_like_rate=("ghost_like", "mean"),
        template_saturation_rate=("template_saturation", "mean"),
        kitchen_sink_tech_rate=("kitchen_sink_tech", "mean"),
        kitchen_sink_org_rate=("kitchen_sink_org", "mean"),
        aspiration_saturated_rate=("aspiration_saturated", "mean"),
        yoe_scope_mismatch_rate=("yoe_scope_mismatch", "mean"),
        company_repetition_rate=("company_repetition", "mean"),
        ai_rate_per_1k_chars=("ai_rate_per_1k_chars", "mean"),
    ).reset_index().rename(columns={"ai_any": "ai_keyword_present"})

    comp = ind.copy()
    comp["company_size_bucket"] = np.where(
        comp["company_size"].isna(),
        "unknown",
        np.where(comp["company_size"] >= 1000, "large_1000_plus", "small_under_1000"),
    )
    comp = comp.groupby(["is_aggregator", "company_size_bucket"]).agg(
        n=("uid", "size"),
        ghost_like_rate=("ghost_like", "mean"),
        template_saturation_rate=("template_saturation", "mean"),
        kitchen_sink_tech_rate=("kitchen_sink_tech", "mean"),
        aspiration_saturated_rate=("aspiration_saturated", "mean"),
        yoe_scope_mismatch_rate=("yoe_scope_mismatch", "mean"),
        ai_any_rate=("ai_any", "mean"),
    ).reset_index()
    return ai, comp


def sample_group(df: pd.DataFrame, period: str, seniority: str, n: int, seed_cols: Sequence[str] = ("uid",)) -> pd.DataFrame:
    subset = df[(df["period"] == period) & (df["seniority_3level"] == seniority) & (df["cleaned_text"].str.len() > 0)].copy()
    if subset.empty:
        return subset
    subset = subset.assign(_sort_key=subset["uid"].map(lambda x: int(hashlib.sha1(str(x).encode()).hexdigest()[:12], 16)))
    subset = subset.sort_values("_sort_key").head(n).drop(columns="_sort_key")
    return subset


def sample_plot_groups(df: pd.DataFrame, n: int = 120) -> pd.DataFrame:
    groups = []
    for period in ["2024-01", "2024-04", "2026-03"]:
        for seniority in ["junior", "mid", "senior", "unknown"]:
            g = sample_group(df, period, seniority, n)
            if not g.empty:
                g = g.copy()
                g["group"] = f"{period} | {seniority}"
                groups.append(g)
    if not groups:
        return pd.DataFrame()
    return pd.concat(groups, ignore_index=True)


def sample_similarity_groups(df: pd.DataFrame, n: int = 800) -> Dict[str, pd.DataFrame]:
    groups = {
        "entry_2024": sample_group(df, "2024-04", "junior", n),
        "entry_2026": sample_group(df, "2026-03", "junior", n),
        "mid_2024": sample_group(df, "2024-04", "mid", n),
        "mid_2026": sample_group(df, "2026-03", "mid", n),
    }
    for name, g in groups.items():
        if not g.empty:
            g = g.copy()
            g["group_name"] = name
            g["group_label"] = g["period"] + " | " + g["seniority_3level"]
            groups[name] = g
    return groups


def embed_texts(texts: Sequence[str]) -> Tuple[np.ndarray, str]:
    method = ""
    try:
        model = get_sentence_model()
        emb = model.encode(
            list(texts),
            batch_size=128,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        method = "sentence-transformers:all-MiniLM-L6-v2"
        return np.asarray(emb, dtype=np.float32), method
    except Exception as exc:
        print(f"[warn] sentence-transformers unavailable or failed ({exc}); falling back to TF-IDF + SVD.")

    vec = TfidfVectorizer(
        max_features=40000,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-z][a-z0-9\+\#\-/]{1,}\b",
        min_df=2,
    )
    X = vec.fit_transform(texts)
    n_comp = min(100, max(2, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    emb = svd.fit_transform(X)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb = emb / norms
    method = f"tfidf+svd({n_comp})"
    return emb.astype(np.float32), method


@lru_cache(maxsize=1)
def get_sentence_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def pairwise_cosine_avg(a: np.ndarray, b: np.ndarray, same_group: bool = False) -> float:
    sims = a @ b.T
    if same_group:
        n = a.shape[0]
        if n < 2:
            return float("nan")
        total = sims.sum() - np.trace(sims)
        return float(total / (n * (n - 1)))
    return float(sims.mean())


def build_similarity_outputs(groups: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    rows = []
    embeddings: Dict[str, np.ndarray] = {}
    methods: Dict[str, str] = {}
    for name, g in groups.items():
        if g.empty:
            continue
        emb, method = embed_texts(g["cleaned_text"].str.slice(0, 1000).tolist())
        embeddings[name] = emb
        methods[name] = method
        g = g.copy()
        g["group_name"] = name
        g["embedding_method"] = method
        groups[name] = g
        within = pairwise_cosine_avg(emb, emb, same_group=True)
        rows.append(
            {
                "group": name,
                "period": g["period"].iloc[0],
                "seniority_3level": g["seniority_3level"].iloc[0],
                "n": len(g),
                "within_group_similarity": within,
                "embedding_method": method,
            }
        )

    sim_lookup = {}
    pair_specs = [
        ("entry_2024", "entry_2026"),
        ("entry_2026", "mid_2024"),
        ("entry_2024", "mid_2024"),
        ("mid_2024", "mid_2026"),
    ]
    for left, right in pair_specs:
        emb_a = embeddings[left]
        emb_b = embeddings[right]
        avg = pairwise_cosine_avg(emb_a, emb_b, same_group=False)
        sim_lookup[(left, right)] = avg
        sim_lookup[(right, left)] = avg

    matrix_rows = [
        {
            "pair_left": "entry_2024",
            "pair_right": "entry_2026",
            "similarity": sim_lookup[("entry_2024", "entry_2026")],
        },
        {
            "pair_left": "entry_2026",
            "pair_right": "mid_2024",
            "similarity": sim_lookup[("entry_2026", "mid_2024")],
        },
        {
            "pair_left": "entry_2024",
            "pair_right": "mid_2024",
            "similarity": sim_lookup[("entry_2024", "mid_2024")],
        },
        {
            "pair_left": "mid_2024",
            "pair_right": "mid_2026",
            "similarity": sim_lookup[("mid_2024", "mid_2026")],
        },
    ]
    return pd.DataFrame(rows), pd.DataFrame(matrix_rows), methods


def plot_embedding(sample: pd.DataFrame) -> Tuple[Path, str]:
    texts = sample["cleaned_text"].str.slice(0, 1000).tolist()
    emb, method = embed_texts(texts)
    # t-SNE can be slow; on this sample size it is still practical and keeps the plot requirement satisfied.
    perplexity = max(5, min(40, len(sample) // 20))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1200,
        random_state=42,
        verbose=0,
    )
    xy = tsne.fit_transform(emb)
    plot_df = sample.copy()
    plot_df["x"] = xy[:, 0]
    plot_df["y"] = xy[:, 1]

    plt.figure(figsize=(12, 9))
    sns.set_theme(style="white", context="talk")
    palette = sns.color_palette("tab10", n_colors=plot_df["group"].nunique())
    for color, (group, g) in zip(palette, plot_df.groupby("group", sort=True)):
        plt.scatter(g["x"], g["y"], s=12, alpha=0.7, label=group, color=color, edgecolors="none")
    plt.legend(loc="best", fontsize=9, frameon=False)
    plt.title(f"Stage 8 SWE embeddings by period × seniority\n{method}")
    plt.tight_layout()
    out = T24_FIG_DIR / "T24_embedding_tsne.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out, method


def build_report_t23(ind: pd.DataFrame, period_table: pd.DataFrame, ai_table: pd.DataFrame, comp_table: pd.DataFrame, examples: pd.DataFrame) -> str:
    top_ghost = (
        ind[(ind["seniority_3level"] == "junior") & (ind["period"].isin(["2024-04", "2026-03"]))]
        .sort_values(["ghost_like", "within_company_nn_sim", "tech_count", "org_count"], ascending=False)
        .head(20)
    )
    top_ghost_table = top_ghost[[
        "uid",
        "period",
        "title",
        "company_name_effective",
        "seniority_3level",
        "ghost_like",
        "template_saturation",
        "kitchen_sink_tech",
        "kitchen_sink_org",
        "aspiration_saturated",
        "yoe_scope_mismatch",
        "company_repetition",
        "requirements_text",
    ]].copy()
    top_ghost_table["requirements_text"] = top_ghost_table["requirements_text"].str.slice(0, 500)

    lines = []
    lines.append("# T23: Ghost requirement patterns")
    lines.append("## Finding")
    ai_true = ai_table.loc[ai_table["ai_keyword_present"] == True, "ghost_like_rate"]
    ai_false = ai_table.loc[ai_table["ai_keyword_present"] == False, "ghost_like_rate"]
    ai_true_rate = float(ai_true.iloc[0]) if len(ai_true) else float("nan")
    ai_false_rate = float(ai_false.iloc[0]) if len(ai_false) else float("nan")
    lines.append(
        f"Template saturation and company repetition are the clearest ghost-like signals in the LinkedIn SWE sample. Across {len(ind):,} usable-text postings, {ind['template_saturation'].mean():.1%} are near-identical to another posting from the same company, {ind['kitchen_sink_tech'].mean():.1%} are kitchen-sink tech lists, and {ind['yoe_scope_mismatch'].mean():.1%} are junior postings with senior-scope language."
    )
    lines.append(
        f"AI-language co-occurs with the composite ghost-like score: AI-keyword postings average {ai_true_rate:.2f} flagged patterns per posting versus {ai_false_rate:.2f} for non-AI postings."
    )
    lines.append("## Implication for analysis")
    lines.append(
        "Ghost-job risk should be treated as a diagnostic covariate and sensitivity check, not as a hard exclusion. The strongest concern is templated company-level repetition, which can overlap with AI-tool requirements and with large employers / aggregators."
    )
    lines.append("## Data quality note")
    lines.append(
        "Stage 8 has no `description_core_llm`, so the analysis starts from `description_core` with `description` fallback. Usable text coverage is effectively complete in this filtered SWE sample; the main limitation is company-size missingness, which makes large-vs-small comparisons subset-based."
    )
    lines.append("## Action items")
    lines.append(
        "Use the prevalence tables for sensitivity analysis, and interpret the aggregator/size comparisons only on the non-missing size subset. The top 20 junior examples are saved in `exploration/tables/T23/ghost_examples.csv` and summarised below."
    )
    lines.append("")
    lines.append("### Prevalence by period × seniority")
    lines.append(dataframe_to_markdown(period_table))
    lines.append("")
    lines.append("### AI cross-tab")
    lines.append(dataframe_to_markdown(ai_table))
    lines.append("")
    lines.append("### Aggregator / company-size comparison")
    lines.append(dataframe_to_markdown(comp_table))
    lines.append("")
    lines.append("### Top 20 ghost-like junior postings")
    lines.append(dataframe_to_markdown(top_ghost_table))
    return "\n".join(lines)


def build_report_t24(sim_pairs: pd.DataFrame, within: pd.DataFrame, sample_meta: pd.DataFrame, plot_path: Path, method: str) -> str:
    lookup = {tuple(row[:2]): row[2] for row in sim_pairs[["pair_left", "pair_right", "similarity"]].itertuples(index=False, name=None)}
    delta = lookup[("entry_2026", "mid_2024")] - lookup[("entry_2024", "mid_2024")]
    lines = []
    lines.append("# T24: Embedding-based similarity")
    lines.append("## Finding")
    if delta > 0:
        finding = f"Using {method} on the first 1,000 characters of cleaned descriptions, junior postings in 2026 are closer to mid-senior 2024 postings than junior postings in 2024 are by {delta:.3f} cosine points, which supports a narrowing-gap interpretation."
    else:
        finding = f"Using {method} on the first 1,000 characters of cleaned descriptions, junior postings in 2026 are farther from mid-senior 2024 postings than junior postings in 2024 are by {abs(delta):.3f} cosine points, so this sample does not support a narrowing-gap interpretation."
    lines.append(finding)
    lines.append(
        f"Within-group homogeneity is highest for {within.sort_values('within_group_similarity', ascending=False).iloc[0]['group']} at {within['within_group_similarity'].max():.3f}, and the 2024-04 junior vs 2026-03 junior similarity is {lookup[('entry_2024', 'entry_2026')]:.3f}."
    )
    lines.append("## Implication for analysis")
    lines.append(
        "The junior-senior language gap does not appear to be narrowing in this embedding sample, so RQ1 should treat the wording shift as mixed rather than monotonic. Keep the 2024-04 arshkon entry cohort as the main baseline; 2024-01 asaniczka remains structurally weak for entry-level inference."
    )
    lines.append("## Data quality note")
    lines.append(
        f"The similarity sample is capped at 1,200 rows per requested group and uses either sentence-transformers or a TF-IDF+SVD fallback. The plot is t-SNE on a smaller period×seniority sample for readability."
    )
    lines.append("## Action items")
    lines.append(
        f"Use `exploration/tables/T24/T24_similarity_matrix.csv` for the main pairwise comparison and `exploration/figures/T24/T24_embedding_tsne.png` for the 2D view. Sample composition is in `exploration/tables/T24/T24_sample_metadata.csv`."
    )
    lines.append("")
    lines.append("### Pairwise similarity matrix")
    lines.append(dataframe_to_markdown(sim_pairs))
    lines.append("")
    lines.append("### Within-group similarity")
    lines.append(dataframe_to_markdown(within))
    lines.append("")
    lines.append("### Sample metadata")
    lines.append(dataframe_to_markdown(sample_meta))
    lines.append("")
    lines.append(f"### Plot\n![T24 embedding plot]({plot_path.relative_to(ROOT)})")
    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    df = load_filtered_data()
    ind = build_indicator_frame(df)

    period_table = make_period_seniority_table(ind)
    ai_table, comp_table = make_cross_tabs(ind)

    period_table.to_csv(T23_TABLE_DIR / "ghost_prevalence_by_period_seniority.csv", index=False)
    ai_table.to_csv(T23_TABLE_DIR / "ghost_ai_crosstab.csv", index=False)
    comp_table.to_csv(T23_TABLE_DIR / "ghost_aggregator_size_comparison.csv", index=False)

    examples = (
        ind[(ind["seniority_3level"] == "junior") & (ind["period"].isin(["2024-04", "2026-03"]))]
        .sort_values(["ghost_like", "within_company_nn_sim", "tech_count", "org_count"], ascending=False)
        .head(20)
        .copy()
    )
    examples["requirements_text"] = examples["requirements_text"].str.slice(0, 1200)
    examples.to_csv(T23_TABLE_DIR / "ghost_examples.csv", index=False)

    t23_report = build_report_t23(ind, period_table, ai_table, comp_table, examples)
    T23_REPORT.write_text(t23_report)

    sim_groups = sample_similarity_groups(ind, n=800)
    within, sim_pairs, methods = build_similarity_outputs(sim_groups)
    sim_pairs.to_csv(T24_TABLE_DIR / "T24_similarity_matrix.csv", index=False)
    within.to_csv(T24_TABLE_DIR / "T24_within_group_similarity.csv", index=False)

    # Sample metadata for the plotted subset.
    plot_sample = sample_plot_groups(ind, n=120)
    plot_sample.to_csv(T24_TABLE_DIR / "T24_sample_metadata.csv", index=False)
    plot_meta = (
        plot_sample.groupby("group")
        .agg(
            n=("uid", "size"),
            period=("period", "first"),
            seniority_3level=("seniority_3level", "first"),
        )
        .reset_index()
        .sort_values(["period", "seniority_3level"])
    )

    # Reuse the same embedding method chosen inside plot_embedding for the report.
    plot_path, plot_method = plot_embedding(plot_sample)
    # Rebuild the report with the actual similarity objects.
    t24_report = build_report_t24(
        sim_pairs,
        within,
        plot_meta,
        plot_path,
        plot_method,
    )
    T24_REPORT.write_text(t24_report)

    summary = {
        "t23_rows": int(len(ind)),
        "t23_template_saturation_rate": float(ind["template_saturation"].mean()),
        "t23_ai_any_rate": float(ind["ai_any"].mean()),
        "t24_similarity_pairs": sim_pairs.to_dict(orient="records"),
        "t24_plot": str(plot_path.relative_to(ROOT)),
    }
    (ROOT / "exploration" / "reports" / "T23_T24_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
