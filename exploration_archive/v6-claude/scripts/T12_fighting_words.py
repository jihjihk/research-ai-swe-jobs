"""T12 — Open-ended text evolution.

Uses log-odds-ratio with informative Dirichlet prior (Monroe, Colaresi,
Quinn 2008 — "Fightin' Words") to find the terms most distinguishing
2024 arshkon vs 2026 scraped SWE postings.

Runs:
  1. Primary: arshkon 2024 vs scraped 2026 (full llm text, cap 50/company).
  2. Section-filtered: only requirements + responsibilities sections
     (using T13_section_classifier).
  3. Secondary: entry 2024 vs entry 2026, mid-senior 2024 vs mid-senior 2026,
     entry 2026 vs mid-senior 2024 (relabeling diagnostic), arshkon mid-senior
     vs asaniczka mid-senior (within-2024 calibration).
  4. Emerging / accelerating / disappearing term lists.
  5. Bigram analysis alongside unigrams.
  6. Aggregator-exclusion sensitivity and raw-text sanity check.

Outputs: exploration/tables/T12/*.csv, exploration/figures/T12/*.png.
"""
from __future__ import annotations

import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from T13_section_classifier import classify_sections  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "exploration" / "artifacts" / "shared"
TABLES_DIR = ROOT / "exploration" / "tables" / "T12"
FIG_DIR = ROOT / "exploration" / "figures" / "T12"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# ---------------------------------------------------------------------------
# Tokenization (matches Prep stoplist-aware cleaning)
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#./\-]*")

# Markdown escape stripper. Scraped text (from markdown -> plain) sometimes
# contains backslash-escaped tech tokens like `c\+\+`, `c\#`, `\.net`. These
# break tokenization and make tech tokens disappear from the scraped corpus,
# which then shows up as a false 2024-heavy signal. We strip the escapes
# before tokenizing. We also collapse curly quotes to straight.
_MD_ESCAPE_RE = re.compile(r"\\([+\-#.&_()\[\]\{\}!*])")
def strip_md_escapes(text: str) -> str:
    if not text:
        return ""
    t = _MD_ESCAPE_RE.sub(r"\1", text)
    # Collapse curly quotes to straight
    t = t.replace("\u2018", "'").replace("\u2019", "'").replace("\u201C", '"').replace("\u201D", '"')
    return t
import nltk
try:
    ENGLISH_STOPS = set(nltk.corpus.stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    ENGLISH_STOPS = set(nltk.corpus.stopwords.words("english"))

TECH_PROTECT = frozenset({
    "python","java","javascript","typescript","go","golang","rust","c++","c#",
    "ruby","kotlin","swift","scala","php","perl","bash","shell","sql","matlab",
    "dart","haskell","elixir","solidity","r","react","angular","vue","nextjs",
    "next.js","svelte","jquery","html","css","tailwind","webpack","node","nodejs",
    "django","flask","spring","springboot",".net","net","dotnet","rails","fastapi",
    "express","graphql","rest","restful","grpc","aws","azure","gcp","kubernetes",
    "k8s","docker","terraform","ansible","ci/cd","cicd","jenkins","argocd","linux",
    "git","helm","serverless","lambda","postgres","postgresql","mysql","mongodb",
    "redis","kafka","spark","hadoop","snowflake","databricks","dbt","elasticsearch",
    "airflow","bigquery","redshift","cassandra","dynamodb","tensorflow","pytorch",
    "sklearn","scikit-learn","pandas","numpy","jupyter","keras","xgboost","ml","nlp",
    "llm","llms","langchain","langgraph","rag","pinecone","chroma","chromadb",
    "huggingface","openai","anthropic","mcp","agent","agents","agentic","gpt","gpt-4",
    "transformer","embedding","embeddings","copilot","cursor","chatgpt","claude",
    "gemini","codex","jest","pytest","selenium","cypress","junit","playwright",
    "agile","scrum","tdd","kanban","microservice","microservices","backend",
    "frontend","fullstack","devops","sre","api",
})

STOPLIST_PATH = SHARED / "company_stoplist.txt"

def load_stoplist():
    toks = set()
    with STOPLIST_PATH.open() as f:
        for line in f:
            t = line.strip().lower()
            if t:
                toks.add(t)
    toks -= TECH_PROTECT
    return toks

def tokenize(text, stoplist, min_len=2):
    if not text:
        return []
    out = []
    text = strip_md_escapes(text)
    for t in TOKEN_RE.findall(text.lower()):
        if len(t) < min_len:
            continue
        if t in stoplist:
            continue
        if t in ENGLISH_STOPS:
            continue
        out.append(t)
    return out

def bigrams(tokens):
    return [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]

# ---------------------------------------------------------------------------
# Fightin' Words / log-odds with Dirichlet prior
# ---------------------------------------------------------------------------
def fightin_words(counts_a, counts_b, prior_alpha=0.01, min_count=5):
    """Log-odds with informative Dirichlet prior, variance-adjusted z-score.

    Monroe, Colaresi, Quinn (2008). Uses the union vocabulary as the prior's
    informative base (prior_alpha * (freq in pooled corpus / total pooled)).

    Returns DataFrame with columns [term, count_a, count_b, z_score,
    log_odds_ratio] sorted by z_score desc (positive = 2026-heavy / b-heavy).
    """
    all_terms = set(counts_a) | set(counts_b)
    na = sum(counts_a.values())
    nb = sum(counts_b.values())
    pooled = Counter(counts_a) + Counter(counts_b)
    np_total = na + nb
    # Informative Dirichlet prior: alpha_w = prior_alpha * (pooled[w] / np_total) * vocab_size
    # Actually the standard implementation uses: alpha_w = prior_alpha * pooled[w]
    rows = []
    for w in all_terms:
        a = counts_a.get(w, 0)
        b = counts_b.get(w, 0)
        if (a + b) < min_count:
            continue
        a0 = prior_alpha * pooled.get(w, 0)
        # posterior log odds for A: log((a + a0) / (na + alpha_total - a - a0)) approx
        # but the Fightin' Words formula is:
        # delta_w = log((a + a0) / (na + alpha0 - a - a0)) - log((b + a0) / (nb + alpha0 - b - a0))
        alpha0 = prior_alpha * np_total  # total prior mass
        num_a = a + a0
        den_a = (na + alpha0) - num_a
        num_b = b + a0
        den_b = (nb + alpha0) - num_b
        if den_a <= 0 or den_b <= 0 or num_a <= 0 or num_b <= 0:
            continue
        lor = np.log(num_a / den_a) - np.log(num_b / den_b)
        var = 1.0 / num_a + 1.0 / num_b  # first-order variance approximation
        z = lor / np.sqrt(var)
        rows.append((w, a, b, z, lor))
    df = pd.DataFrame(rows, columns=["term", "count_a", "count_b", "z_score", "log_odds_ratio"])
    return df

def top_terms_per_direction(df, n=100, min_companies=20, company_map=None):
    """Return top-n 2026-heavy (negative z, since a=2024, b=2026 -> b-heavy = negative lor).

    Convention in this module: A = 2024, B = 2026, so positive z is 2024-heavy,
    negative z is 2026-heavy.
    """
    filt = df.copy()
    if company_map is not None:
        filt["company_count"] = filt["term"].map(company_map).fillna(0)
        filt = filt[filt["company_count"] >= min_companies]
    top_2024 = filt.sort_values("z_score", ascending=False).head(n)
    top_2026 = filt.sort_values("z_score", ascending=True).head(n)
    return top_2024, top_2026

# ---------------------------------------------------------------------------
# Semantic category tagging
# ---------------------------------------------------------------------------
CATEGORY_PATTERNS = [
    # Ordered — first match wins
    ("boilerplate", re.compile(
        r"^(benefit|benefits|compensation|salary|pay|equity|bonus|dental|401k|pto|"
        r"vacation|holidays|insurance|medical|vision|stipend|perks|culture|mission|"
        r"values|diversity|inclusion|inclusive|equal|sponsorship|visa|employees|"
        r"people|rewards|offer|opportunity|policy|hiring|employment|applicants|"
        r"accommodations|eligibility|family|caring|wellness|retirement|pension)$")),
    ("ai_tool", re.compile(
        r"^(copilot|cursor|claude|gpt|gpt-4|gpt-3|chatgpt|gemini|codex|llm|llms|"
        r"rag|agent|agents|agentic|mcp|anthropic|openai|huggingface|langchain|"
        r"langgraph|pinecone|chroma|chromadb|bedrock|vertex|prompt|multi-agent|"
        r"llm-based|llm-powered|llm-driven|guardrails|retrieval-augmented|"
        r"orchestration\.?|agents\.?)$")),
    ("ai_domain", re.compile(
        r"^(ml|nlp|ai|generative|genai|transformer|embedding|embeddings|fine-tuning|"
        r"finetuning|neural|deep|vision|computer-vision|ai/ml|ai-assisted|"
        r"ai-powered|ai-driven|ai-enabled|ai-generated|multimodal|"
        r"ml-ops|mlops)$")),
    ("tech_stack", re.compile(
        r"^(python|java|javascript|typescript|go|golang|rust|c\+\+|c#|ruby|kotlin|"
        r"swift|scala|php|perl|bash|sql|react|angular|vue|nextjs|next\.js|svelte|"
        r"node|nodejs|django|flask|spring|springboot|\.net|dotnet|rails|fastapi|"
        r"graphql|rest|restful|grpc|aws|azure|gcp|kubernetes|k8s|docker|terraform|"
        r"ansible|ci/cd|cicd|jenkins|linux|git|postgres|postgresql|mysql|mongodb|"
        r"redis|kafka|spark|hadoop|snowflake|databricks|dbt|elasticsearch|airflow|"
        r"bigquery|redshift|cassandra|dynamodb|tensorflow|pytorch|sklearn|scikit-learn|"
        r"pandas|numpy|jupyter|keras|xgboost|jest|pytest|selenium|cypress|junit|"
        r"playwright|microservice|microservices|serverless|api|helm|argocd)$")),
    ("org_scope", re.compile(
        r"^(ownership|own|end-to-end|cross-functional|stakeholder|stakeholders|"
        r"influence|partner|partnership|collaborate|collaboration|driving|strategic|"
        r"roadmap|scope|initiative|initiatives|impact|autonomy|autonomous)$")),
    ("mgmt", re.compile(
        r"^(lead|leads|leader|leadership|mentor|mentoring|manage|management|hire|"
        r"hiring|coach|coaching|guide|direct|oversee|supervise|principal|staff|"
        r"senior)$")),
    ("sys_design", re.compile(
        r"^(distributed|scalable|scalability|architecture|architect|architectural|"
        r"design|system|systems|performance|reliability|resilient|observability|"
        r"monitoring|throughput|latency|availability|fault-tolerant|high-availability)$")),
    ("method", re.compile(
        r"^(agile|scrum|kanban|ci/cd|cicd|tdd|devops|sre|code-review|waterfall|lean|"
        r"gitops|iterative|retrospective|sprint|sprints)$")),
    ("credential", re.compile(
        r"^(bachelor|bachelors|ba|bs|master|masters|ms|phd|degree|diploma|certification|"
        r"certified|certificate|years|year|experience|equivalent|minimum|required|"
        r"preferred|qualification|qualifications|requirement|requirements|education)$")),
    ("soft_skill", re.compile(
        r"^(communication|communicate|collaboration|collaborative|problem-solving|"
        r"problem|teamwork|adaptable|curious|independent|passionate|ownership|"
        r"interpersonal|written|verbal|presentation)$")),
]

def categorize(term):
    t = term.lower()
    for cat, pat in CATEGORY_PATTERNS:
        if pat.match(t):
            return cat
    return "noise"

# ---------------------------------------------------------------------------
# Data loading and corpus building
# ---------------------------------------------------------------------------
def load_corpus():
    """Load SWE LinkedIn llm-labeled rows. Returns a DataFrame with uid, source,
    period, period_label, seniority, is_aggregator, company, description_core_llm."""
    con = duckdb.connect()
    q = """
    SELECT
        uid, source, period, seniority_final, seniority_3level,
        is_aggregator, company_name_canonical,
        description_core_llm, description
    FROM 'data/unified.parquet'
    WHERE is_swe = true
      AND source_platform = 'linkedin'
      AND is_english = true
      AND date_flag = 'ok'
      AND llm_extraction_coverage = 'labeled'
      AND description_core_llm IS NOT NULL
    """
    df = con.execute(q).df()
    df["period_label"] = np.where(df["source"] == "scraped", "2026", "2024")
    print(f"Loaded {len(df)} llm-labeled SWE linkedin rows")
    print(df.groupby(["source", "period_label"]).size())
    return df

def cap_per_company(df, cap=50, seed=SEED):
    """Randomly cap postings at `cap` per company_name_canonical."""
    out = (
        df.groupby("company_name_canonical", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), cap), random_state=seed), include_groups=True)
    )
    return out.reset_index(drop=True)

def extract_section_text(text, keep=("requirements", "responsibilities")):
    segs = classify_sections(text)
    parts = []
    for s in segs:
        if s["section"] in keep:
            parts.append(s["text"])
    return "\n".join(parts)

def count_terms(texts, stoplist, use_bigrams=False):
    """Return (term_counts, doc_counts_by_company)."""
    counts = Counter()
    company_terms: defaultdict = defaultdict(set)  # term -> set of (company,)
    for text in texts:
        tokens = tokenize(text, stoplist)
        counts.update(tokens)
        if use_bigrams:
            counts.update(bigrams(tokens))
    return counts

def build_corpus_counts(df, text_col="description_core_llm", stoplist=None,
                       include_bigrams=True):
    """Return (unigram_counts, unigram_company_map) where company_map maps term
    to number of distinct companies mentioning it."""
    uni_counts = Counter()
    bi_counts = Counter()
    uni_comp = defaultdict(set)
    bi_comp = defaultdict(set)
    for _, row in df.iterrows():
        text = row[text_col]
        if not text:
            continue
        company = row["company_name_canonical"]
        toks = tokenize(text, stoplist)
        uni_counts.update(toks)
        for w in set(toks):
            uni_comp[w].add(company)
        if include_bigrams:
            bg = bigrams(toks)
            bi_counts.update(bg)
            for w in set(bg):
                bi_comp[w].add(company)
    uni_comp_n = {k: len(v) for k, v in uni_comp.items()}
    bi_comp_n = {k: len(v) for k, v in bi_comp.items()}
    return uni_counts, uni_comp_n, bi_counts, bi_comp_n

# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
def run_comparison(df_a, df_b, label, stoplist, text_col="description_core_llm",
                   n_top=100, min_companies=20):
    print(f"\n=== Comparison: {label}  n_a={len(df_a)} n_b={len(df_b)} ===")
    uni_a, uc_a, bi_a, bc_a = build_corpus_counts(df_a, text_col, stoplist)
    uni_b, uc_b, bi_b, bc_b = build_corpus_counts(df_b, text_col, stoplist)
    # Merged company map
    merged_uc = {}
    for t in set(uc_a) | set(uc_b):
        merged_uc[t] = uc_a.get(t, 0) + uc_b.get(t, 0)
    merged_bc = {}
    for t in set(bc_a) | set(bc_b):
        merged_bc[t] = bc_a.get(t, 0) + bc_b.get(t, 0)

    # Unigram Fightin' Words
    df_fw = fightin_words(uni_a, uni_b, prior_alpha=0.1, min_count=20)
    top_a, top_b = top_terms_per_direction(df_fw, n=n_top, min_companies=min_companies, company_map=merged_uc)
    top_a["direction"] = "A_heavy"
    top_b["direction"] = "B_heavy"
    top_a["category"] = top_a["term"].apply(categorize)
    top_b["category"] = top_b["term"].apply(categorize)
    combined = pd.concat([top_a, top_b], ignore_index=True)
    combined["comparison"] = label
    combined["ngram"] = "unigram"

    # Bigram Fightin' Words
    df_fb = fightin_words(bi_a, bi_b, prior_alpha=0.1, min_count=15)
    top_ba, top_bb = top_terms_per_direction(df_fb, n=50, min_companies=min_companies, company_map=merged_bc)
    top_ba["direction"] = "A_heavy"
    top_bb["direction"] = "B_heavy"
    top_ba["category"] = top_ba["term"].apply(categorize)
    top_bb["category"] = top_bb["term"].apply(categorize)
    combined_b = pd.concat([top_ba, top_bb], ignore_index=True)
    combined_b["comparison"] = label
    combined_b["ngram"] = "bigram"

    return pd.concat([combined, combined_b], ignore_index=True), {
        "uni_a": uni_a, "uni_b": uni_b, "uc": merged_uc,
    }

# ---------------------------------------------------------------------------
# Emerging / accelerating / disappearing
# ---------------------------------------------------------------------------
def term_presence_by_period(df_a, df_b, stoplist, min_companies=20):
    """For each term, compute per-posting presence rates in A and B."""
    # Per-posting binary presence
    def doc_presence(df):
        total = len(df)
        counts = Counter()
        comps = defaultdict(set)
        for _, r in df.iterrows():
            text = r["description_core_llm"]
            if not text:
                continue
            toks = set(tokenize(text, stoplist))
            for w in toks:
                counts[w] += 1
                comps[w].add(r["company_name_canonical"])
        return counts, comps, total
    cnt_a, comp_a, n_a = doc_presence(df_a)
    cnt_b, comp_b, n_b = doc_presence(df_b)
    rows = []
    for w in set(cnt_a) | set(cnt_b):
        a = cnt_a.get(w, 0)
        b = cnt_b.get(w, 0)
        ca = len(comp_a.get(w, set()))
        cb = len(comp_b.get(w, set()))
        n_comp_total = ca + cb
        if n_comp_total < min_companies:
            continue
        p_a = a / n_a if n_a else 0.0
        p_b = b / n_b if n_b else 0.0
        rows.append({"term": w, "p_a": p_a, "p_b": p_b,
                     "count_a": a, "count_b": b, "n_a": n_a, "n_b": n_b,
                     "companies_a": ca, "companies_b": cb})
    return pd.DataFrame(rows)

def classify_term_movement(pres_df):
    emerging = pres_df[(pres_df["p_b"] > 0.01) & (pres_df["p_a"] < 0.001)].copy()
    disappearing = pres_df[(pres_df["p_a"] > 0.01) & (pres_df["p_b"] < 0.001)].copy()
    accelerating = pres_df[(pres_df["p_a"] >= 0.001) & (pres_df["p_b"] >= 0.01)].copy()
    accelerating["ratio"] = accelerating["p_b"] / accelerating["p_a"]
    accelerating = accelerating[accelerating["ratio"] >= 3.0]
    emerging["category"] = emerging["term"].apply(categorize)
    disappearing["category"] = disappearing["term"].apply(categorize)
    accelerating["category"] = accelerating["term"].apply(categorize)
    return emerging.sort_values("p_b", ascending=False), \
           disappearing.sort_values("p_a", ascending=False), \
           accelerating.sort_values("ratio", ascending=False)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    stoplist = load_stoplist()
    print(f"Stoplist size: {len(stoplist)}")

    df = load_corpus()
    # Section-filtered column
    print("\nBuilding section-filtered text for llm rows (requirements + responsibilities)…")
    df["section_text"] = df["description_core_llm"].apply(extract_section_text)
    # How much text survives section filter?
    df["section_text_len"] = df["section_text"].str.len()
    df["full_text_len"] = df["description_core_llm"].str.len()
    print(f"Mean full len: {df['full_text_len'].mean():.0f}, "
          f"mean section-filtered len: {df['section_text_len'].mean():.0f} "
          f"({df['section_text_len'].mean() / df['full_text_len'].mean():.1%})")

    # Primary: arshkon 2024 vs scraped 2026, all SWE, cap 50/company
    arshkon = df[df["source"] == "kaggle_arshkon"]
    scraped = df[df["source"] == "scraped"]
    print(f"\narshkon rows: {len(arshkon)} | scraped rows: {len(scraped)}")
    arshkon_c = cap_per_company(arshkon, cap=50)
    scraped_c = cap_per_company(scraped, cap=50)
    print(f"cap50: arshkon {len(arshkon_c)} | scraped {len(scraped_c)}")

    all_tables = []

    # PRIMARY: full-text
    primary_tbl, _ = run_comparison(arshkon_c, scraped_c, "primary_arshkon_vs_scraped_fulltext", stoplist)
    primary_tbl.to_csv(TABLES_DIR / "primary_fulltext_top_terms.csv", index=False)
    all_tables.append(primary_tbl)

    # PRIMARY: section-filtered
    section_tbl, _ = run_comparison(arshkon_c, scraped_c, "primary_arshkon_vs_scraped_sections", stoplist,
                                    text_col="section_text")
    section_tbl.to_csv(TABLES_DIR / "primary_section_filtered_top_terms.csv", index=False)
    all_tables.append(section_tbl)

    # Compare full-text vs section-filtered: which 2026-heavy terms only in fulltext?
    ft_2026 = set(primary_tbl[(primary_tbl["ngram"] == "unigram") & (primary_tbl["direction"] == "B_heavy")]["term"])
    sf_2026 = set(section_tbl[(section_tbl["ngram"] == "unigram") & (section_tbl["direction"] == "B_heavy")]["term"])
    only_ft = ft_2026 - sf_2026
    both = ft_2026 & sf_2026
    only_sf = sf_2026 - ft_2026
    pd.DataFrame({
        "cohort": ["only_fulltext_2026_heavy", "both_2026_heavy", "only_section_2026_heavy"],
        "n": [len(only_ft), len(both), len(only_sf)],
        "terms": [",".join(sorted(only_ft)), ",".join(sorted(both)), ",".join(sorted(only_sf))],
    }).to_csv(TABLES_DIR / "fulltext_vs_section_top100_overlap.csv", index=False)
    print(f"2026-heavy: only fulltext={len(only_ft)} both={len(both)} only section={len(only_sf)}")

    # SECONDARY: seniority-stratified comparisons
    ar_entry = df[(df["source"] == "kaggle_arshkon") & (df["seniority_3level"] == "junior")]
    sc_entry = df[(df["source"] == "scraped") & (df["seniority_3level"] == "junior")]
    ar_pool_entry = df[(df["period_label"] == "2024") & (df["seniority_3level"] == "junior")]
    ar_mid = df[(df["period_label"] == "2024") & (df["seniority_3level"] == "mid")]
    ar_sen = df[(df["period_label"] == "2024") & (df["seniority_3level"] == "senior")]
    sc_mid = df[(df["period_label"] == "2026") & (df["seniority_3level"] == "mid")]
    sc_sen = df[(df["period_label"] == "2026") & (df["seniority_3level"] == "senior")]
    print(f"\nentry 2024 (arshkon)={len(ar_entry)}, entry 2024 (pooled)={len(ar_pool_entry)}, entry 2026={len(sc_entry)}")
    print(f"mid 2024={len(ar_mid)}, mid 2026={len(sc_mid)}")
    print(f"senior 2024={len(ar_sen)}, senior 2026={len(sc_sen)}")

    # Entry pooled 2024 vs 2026
    if len(ar_pool_entry) >= 100 and len(sc_entry) >= 100:
        t, _ = run_comparison(cap_per_company(ar_pool_entry), cap_per_company(sc_entry),
                              "entry_2024pooled_vs_2026", stoplist)
        all_tables.append(t)
    # mid 2024 vs 2026
    if len(ar_mid) >= 100 and len(sc_mid) >= 100:
        t, _ = run_comparison(cap_per_company(ar_mid), cap_per_company(sc_mid),
                              "mid_2024_vs_2026", stoplist)
        all_tables.append(t)
    # senior 2024 vs 2026
    if len(ar_sen) >= 100 and len(sc_sen) >= 100:
        t, _ = run_comparison(cap_per_company(ar_sen), cap_per_company(sc_sen),
                              "senior_2024_vs_2026", stoplist)
        all_tables.append(t)
    # Relabeling diagnostic: entry 2026 vs mid-senior 2024
    mid_senior_2024 = df[(df["period_label"] == "2024") & (df["seniority_3level"].isin(["mid", "senior"]))]
    if len(sc_entry) >= 100 and len(mid_senior_2024) >= 100:
        t, _ = run_comparison(cap_per_company(mid_senior_2024), cap_per_company(sc_entry),
                              "relabeling_midsenior2024_vs_entry2026", stoplist)
        all_tables.append(t)
    # Within-2024: arshkon mid-senior vs asaniczka mid-senior (calibration)
    ar_ms = df[(df["source"] == "kaggle_arshkon") & (df["seniority_3level"].isin(["mid", "senior"]))]
    as_ms = df[(df["source"] == "kaggle_asaniczka") & (df["seniority_3level"].isin(["mid", "senior"]))]
    if len(ar_ms) >= 100 and len(as_ms) >= 100:
        t, _ = run_comparison(cap_per_company(ar_ms), cap_per_company(as_ms),
                              "within2024_arshkon_vs_asaniczka_midsenior", stoplist)
        all_tables.append(t)

    # AGGREGATOR EXCLUSION SENSITIVITY (primary only)
    arshkon_na = arshkon[~arshkon["is_aggregator"].astype(bool)]
    scraped_na = scraped[~scraped["is_aggregator"].astype(bool)]
    arshkon_na_c = cap_per_company(arshkon_na)
    scraped_na_c = cap_per_company(scraped_na)
    print(f"\naggregator-excluded: arshkon {len(arshkon_na_c)}, scraped {len(scraped_na_c)}")
    t_noaggr, _ = run_comparison(arshkon_na_c, scraped_na_c, "primary_noaggregators", stoplist)
    all_tables.append(t_noaggr)

    # RAW TEXT SANITY CHECK (sensitivity d): use raw description
    # Load raw text directly
    con = duckdb.connect()
    raw_df = con.execute("""
    SELECT uid, source, company_name_canonical, description
    FROM 'data/unified.parquet'
    WHERE is_swe = true AND source_platform = 'linkedin' AND is_english = true
      AND date_flag = 'ok' AND description IS NOT NULL
      AND source IN ('kaggle_arshkon', 'scraped')
    """).df()
    raw_df["period_label"] = np.where(raw_df["source"] == "scraped", "2026", "2024")
    raw_arsh = raw_df[raw_df["source"] == "kaggle_arshkon"]
    raw_scrp = raw_df[raw_df["source"] == "scraped"]
    raw_arsh_c = cap_per_company(raw_arsh)
    raw_scrp_c = cap_per_company(raw_scrp)
    # Rename description column to description_core_llm so run_comparison works
    raw_arsh_c = raw_arsh_c.rename(columns={"description": "description_core_llm"})
    raw_scrp_c = raw_scrp_c.rename(columns={"description": "description_core_llm"})
    raw_arsh_c["description_core_llm"] = raw_arsh_c["description_core_llm"].astype(str)
    raw_scrp_c["description_core_llm"] = raw_scrp_c["description_core_llm"].astype(str)
    t_raw, _ = run_comparison(raw_arsh_c, raw_scrp_c, "primary_rawtext_sanity", stoplist)
    all_tables.append(t_raw)

    # Emerging / accelerating / disappearing (llm text, cap 50/company, arshkon vs scraped)
    print("\n=== Emerging / accelerating / disappearing ===")
    pres = term_presence_by_period(arshkon_c, scraped_c, stoplist, min_companies=20)
    emerging, disappearing, accelerating = classify_term_movement(pres)
    emerging.to_csv(TABLES_DIR / "emerging_terms.csv", index=False)
    disappearing.to_csv(TABLES_DIR / "disappearing_terms.csv", index=False)
    accelerating.to_csv(TABLES_DIR / "accelerating_terms.csv", index=False)
    print(f"emerging={len(emerging)}, disappearing={len(disappearing)}, accelerating={len(accelerating)}")
    print("Top 10 emerging:", emerging.head(10)["term"].tolist())
    print("Top 10 accelerating:", accelerating.head(10)["term"].tolist())
    print("Top 10 disappearing:", disappearing.head(10)["term"].tolist())

    # Save master combined table
    combined_all = pd.concat(all_tables, ignore_index=True)
    combined_all.to_csv(TABLES_DIR / "all_fightin_words_top_terms.csv", index=False)

    # Category summary figure — primary comparison, unigrams
    pri_uni = primary_tbl[primary_tbl["ngram"] == "unigram"]
    sf_uni = section_tbl[section_tbl["ngram"] == "unigram"]
    cat_summary_rows = []
    for tbl, name in [(pri_uni, "full_text"), (sf_uni, "section_filtered")]:
        for direction in ["A_heavy", "B_heavy"]:
            sub = tbl[tbl["direction"] == direction]
            counts = sub["category"].value_counts()
            for cat, c in counts.items():
                cat_summary_rows.append({"source": name, "direction": direction,
                                         "category": cat, "n": c})
    cat_summary = pd.DataFrame(cat_summary_rows)
    cat_summary.to_csv(TABLES_DIR / "category_summary.csv", index=False)

    # Figure: category distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for i, (name, title) in enumerate([("full_text", "Full text"), ("section_filtered", "Section-filtered")]):
        ax = axes[i]
        sub = cat_summary[cat_summary["source"] == name]
        pivot = sub.pivot(index="category", columns="direction", values="n").fillna(0)
        pivot.columns = ["2024-heavy" if c == "A_heavy" else "2026-heavy" for c in pivot.columns]
        pivot.plot(kind="barh", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("n terms (top-100 each direction)")
    fig.suptitle("Semantic category of distinguishing terms, arshkon vs scraped")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "category_summary.png", dpi=150)
    plt.close(fig)

    # Figure: top 20 2024-heavy and top 20 2026-heavy (primary unigram full-text)
    top20_a = pri_uni[pri_uni["direction"] == "A_heavy"].head(20)
    top20_b = pri_uni[pri_uni["direction"] == "B_heavy"].head(20)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    axes[0].barh(top20_a["term"][::-1], top20_a["z_score"][::-1], color="#1f77b4")
    axes[0].set_title("Top 20 2024-heavy (arshkon)")
    axes[0].set_xlabel("z-score (log-odds)")
    axes[1].barh(top20_b["term"][::-1], -top20_b["z_score"][::-1], color="#d62728")
    axes[1].set_title("Top 20 2026-heavy (scraped)")
    axes[1].set_xlabel("|z-score|")
    fig.suptitle("Most distinguishing unigrams (full text, cap-50)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_unigrams.png", dpi=150)
    plt.close(fig)

    print(f"\nDone in {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
