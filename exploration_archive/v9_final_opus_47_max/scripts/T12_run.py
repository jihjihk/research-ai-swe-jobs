"""T12 — Open-ended text evolution.

Depends on T13's section classifier. Runs Fightin' Words (log-odds ratio with
informative Dirichlet prior) and related discovery passes on the SWE LinkedIn
corpus to surface terms and bigrams that differ most between periods.

Comparisons
-----------

1. PRIMARY: arshkon (2024) vs scraped (2026) — full SWE cleaned text.
2. SECTION-FILTERED: same, but only requirements + responsibilities sections.
3. EMERGING / ACCELERATING / DISAPPEARING terms.
4. LABEL-BASED:
     (a) Entry 2024 vs Entry 2026 (J3 primary + J1 sensitivity).
     (b) Mid-senior 2024 vs Mid-senior 2026 (S4 primary + S1 sensitivity).
     (c) Entry 2026 vs Mid-senior 2024 (relabeling diagnostic).
5. YOE-BAND:
     (a) YOE ≤ 2 in 2024 vs 2026.
     (b) YOE ≥ 5 in 2024 vs 2026.
     (c) YOE ≤ 2 in 2026 vs YOE ≥ 5 in 2024 (YOE-based relabeling diagnostic).
6. BIGRAMS.
7. Within-2024 calibration: arshkon mid-senior vs asaniczka mid-senior.

Outputs
-------

tables/T12/
    primary_arshkon_vs_scraped_unigrams.csv
    primary_arshkon_vs_scraped_bigrams.csv
    section_filtered_arshkon_vs_scraped.csv
    emerging_terms.csv
    accelerating_terms.csv
    disappearing_terms.csv
    entry_label_2024_vs_2026.csv
    midsenior_label_2024_vs_2026.csv
    entry_2026_vs_midsenior_2024.csv
    yoe_le2_2024_vs_2026.csv
    yoe_ge5_2024_vs_2026.csv
    yoe_le2_2026_vs_yoe_ge5_2024.csv
    within_2024_arshkon_vs_asaniczka_midsenior.csv
    category_summary.csv
    comparison_summary.csv

figures/T12/
    category_stacked_bar.png
    relabeling_diagnostic.png
"""

from __future__ import annotations

import re
import sys
import time
import collections
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import duckdb

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent.parent
sys.path.insert(0, str(THIS_DIR))

from T13_section_classifier import classify_sections, SECTION_LABELS  # type: ignore  # noqa: E402

UNIFIED = REPO / "data" / "unified.parquet"
SWE_TEXT = REPO / "exploration" / "artifacts" / "shared" / "swe_cleaned_text.parquet"
COMPANY_STOPLIST = REPO / "exploration" / "artifacts" / "shared" / "company_stoplist.txt"
T13_METRICS = REPO / "exploration" / "artifacts" / "shared" / "T13_readability_metrics.parquet"
OUT_TABLES = REPO / "exploration" / "tables" / "T12"
OUT_FIG = REPO / "exploration" / "figures" / "T12"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Fightin' Words (Monroe et al. 2008)
# ---------------------------------------------------------------------------

def fighting_words(
    counts_a: Counter,
    counts_b: Counter,
    prior_alpha: float = 0.01,
    min_count: int = 5,
    top_k: int = 100,
) -> pd.DataFrame:
    """Compute log-odds ratio with informative Dirichlet prior.

    Returns a DataFrame sorted by z-score with columns:
        term, count_a, count_b, total_a, total_b, log_odds, z, rate_a, rate_b, ratio
    """
    vocab = set(counts_a) | set(counts_b)
    # Filter to terms with min_count in at least one corpus.
    vocab = {t for t in vocab if counts_a.get(t, 0) + counts_b.get(t, 0) >= min_count}
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())
    # Background prior = a + b
    prior = Counter()
    for t in vocab:
        prior[t] = prior_alpha * (counts_a.get(t, 0) + counts_b.get(t, 0))

    prior_total = sum(prior.values()) or 1.0

    rows = []
    for t in vocab:
        ca = counts_a.get(t, 0)
        cb = counts_b.get(t, 0)
        pa = prior[t]
        # Smoothed log-odds
        numer_a = ca + pa
        denom_a = total_a + prior_total - numer_a
        numer_b = cb + pa
        denom_b = total_b + prior_total - numer_b
        if denom_a <= 0 or denom_b <= 0 or numer_a <= 0 or numer_b <= 0:
            continue
        log_odds = np.log(numer_a / denom_a) - np.log(numer_b / denom_b)
        # Variance
        var = 1.0 / numer_a + 1.0 / numer_b
        z = log_odds / np.sqrt(var)
        rate_a = ca / total_a if total_a else 0
        rate_b = cb / total_b if total_b else 0
        ratio = (rate_a / rate_b) if rate_b > 0 else float("inf")
        rows.append({
            "term": t,
            "count_a": ca,
            "count_b": cb,
            "rate_a": rate_a,
            "rate_b": rate_b,
            "ratio_a_over_b": ratio,
            "log_odds": float(log_odds),
            "z": float(z),
        })
    df = pd.DataFrame(rows).sort_values("z", ascending=False)
    return df


# ---------------------------------------------------------------------------
# Tokenization and counting
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z][a-z0-9+#\-]*[a-z0-9+#]|[a-z]")
# Explicit: start with a letter, then letters/digits/+#- allowed, must end
# with a letter/digit/#/+ (no trailing dots or slashes).

# Preserve common tech patterns (c++, c#, .net) but treat '.' elsewhere as word boundary.
_PRESERVE = {
    r"c\+\+": "c++",
    r"c#": "c#",
    r"\.net": "dotnet",
    r"node\.js": "nodejs",
    r"next\.js": "nextjs",
    r"vue\.js": "vuejs",
}
_PRESERVE_RE = [(re.compile(p, re.IGNORECASE), sub) for p, sub in _PRESERVE.items()]


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    t = text.lower()
    # Preserve compound tech tokens before breaking on '.' and '/'.
    for pat, sub in _PRESERVE_RE:
        t = pat.sub(sub, t)
    # Replace everything that isn't a letter/digit/+/#/- with a space.
    t = re.sub(r"[^a-z0-9+#\- ]+", " ", t)
    return [tok for tok in _TOKEN_RE.findall(t) if len(tok) >= 2 or tok in {"c", "r", "go"}]


def bigrams(tokens: list[str]) -> list[str]:
    return [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def load_company_stoplist() -> set[str]:
    """Load company stoplist but whitelist important AI/tech tool names that are
    also company names (copilot, langchain, etc.). Otherwise the T12 primary
    comparison loses most of the LLM-era tool evidence.
    """
    whitelist = {
        # AI tools that are also company / product names
        "copilot", "langchain", "langgraph", "llamaindex", "pinecone", "weaviate",
        "anthropic", "openai", "huggingface", "mistral", "claude", "cursor",
        "codex", "cody", "gemini", "chatgpt", "llama", "tabnine", "devin",
        "replit", "grok", "deepseek",
        # Dev tools that are also companies
        "github", "gitlab", "docker", "kubernetes", "terraform", "jenkins",
        "snowflake", "databricks", "airflow", "mlflow",
    }
    with open(COMPANY_STOPLIST) as f:
        raw = {line.strip() for line in f if line.strip()}
    return raw - whitelist


def strip_boilerplate_sections(text: str) -> str:
    """Return only requirements + responsibilities characters of the raw text.

    Uses the T13 section classifier in two-pass mode: classify sections on the
    raw (or cleaned) description, then reconstruct only the keep-sections.
    """
    if not text:
        return ""
    # The classifier returns character counts, not spans. To extract spans we
    # re-run its internal logic. Simpler: split lines, re-classify line-level
    # walking, keep only lines attributed to requirements / responsibilities.
    from T13_section_classifier import (
        _normalize_for_header_detection,
        _match_header,
        _INLINE_HEADER_PATTERNS,
    )
    keep = {"requirements", "responsibilities"}
    # Inline rescue via pattern insertion of markers
    working = text
    for label, pat in _INLINE_HEADER_PATTERNS:
        working = pat.sub(lambda m: "\n" + m.group(1) + "\n", working)
    current = "unclassified"
    out_parts = []
    for raw_line in working.splitlines(keepends=True):
        content = raw_line.strip()
        if content:
            norm = _normalize_for_header_detection(raw_line)
            hit = _match_header(norm)
            if hit is not None:
                current = hit
                if current in keep:
                    # Include the header line too so that body of the section
                    # is continguous.
                    out_parts.append(raw_line)
                continue
        if current in keep:
            out_parts.append(raw_line)
    return "".join(out_parts)


# ---------------------------------------------------------------------------
# Semantic category tagging
# ---------------------------------------------------------------------------

CATEGORIES: dict[str, list[str]] = {
    "ai_tool": [
        r"copilot", r"chatgpt", r"claude", r"gemini", r"cursor", r"codex",
        r"llama", r"mistral", r"\banthropic\b", r"openai", r"huggingface|hugging",
        r"gpt-\d", r"gpt\d", r"gpt4", r"deepseek", r"grok", r"devin",
        r"cody", r"continue-ai", r"tabnine", r"replit", r"v0",
    ],
    "ai_concept": [
        r"\bllm\b|\bllms\b", r"prompt engineering", r"prompt\b", r"rag\b",
        r"\bagent\b|agentic|ai agent|\bagents\b", r"fine[- ]tuning", r"fine[- ]tune",
        r"vector database|vector db|embedding|embeddings", r"langchain|llama[- ]index|langgraph",
        r"multimodal", r"generative", r"transformer", r"reinforcement learning",
        r"reasoning model|chain of thought", r"generative ai|gen[\- ]ai",
    ],
    "ai_generic": [
        r"\bai\b|artificial intelligence", r"machine learning|\bml\b|\bmlops\b",
        r"deep learning", r"nlp\b|natural language",
        r"computer vision", r"neural network",
    ],
    "org_scope": [
        r"cross[- ]functional", r"stakeholder", r"mentor|mentoring|mentorship",
        r"lead|leadership|leader", r"ownership", r"drive|drives|drove",
        r"strategy|strategic", r"vision", r"influence|influencing",
    ],
    "tech_stack": [
        r"python|java|javascript|typescript|rust|\bgo\b|golang|scala|kotlin",
        r"react|angular|vue|svelte|next\.?js|nuxt",
        r"aws|azure|gcp|kubernetes|docker|terraform", r"sql|postgres|mongo",
        r"fastapi|flask|django|spring|express",
    ],
    "boilerplate": [
        r"equal opportunity|eeo", r"benefits|perks|401k|pto",
        r"reasonable accommodation", r"pay transparency", r"compensation range",
        r"background check", r"visa|sponsorship", r"hybrid|remote|onsite",
        r"our mission|we believe|our team|join us",
    ],
    "credential": [
        r"bachelor|b\.?s\.?|b\.a\.?|master|m\.?s\.?|phd|doctorate",
        r"degree|gpa|certification|certificate",
    ],
    "yoe": [r"\d+\+? years?\b", r"\d+ yoe\b", r"years of experience"],
    "responsibilities_verbs": [
        r"design|designing|implement|build|deliver|ship|ownership|architect",
    ],
    "metrics_perf": [
        r"uptime|sla|latency|throughput|scalab\w+|performance|reliab\w+",
    ],
    "ml_data": [
        r"data pipeline|etl|streaming|kafka|spark|airflow|dbt|snowflake|databricks",
    ],
}

COMPILED_CATEGORIES = {cat: [re.compile(p, re.IGNORECASE) for p in patterns] for cat, patterns in CATEGORIES.items()}


def categorize_term(term: str) -> str:
    for cat, pats in COMPILED_CATEGORIES.items():
        for p in pats:
            if p.search(term):
                return cat
    return "other"


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------

def load_main_frame() -> pd.DataFrame:
    """Load SWE LinkedIn rows with raw description, cleaned-text tokens, and
    seniority panel info."""
    con = duckdb.connect()
    q = f"""
    SELECT
      c.uid,
      c.source,
      CASE WHEN c.source LIKE 'kaggle_%' THEN '2024' ELSE '2026' END AS period_year,
      c.seniority_final,
      c.seniority_3level,
      c.yoe_min_years_llm,
      c.is_aggregator,
      c.llm_classification_coverage,
      c.text_source,
      c.description_cleaned,
      u.description AS description_raw,
      u.description_core_llm
    FROM read_parquet('{SWE_TEXT.as_posix()}') AS c
    LEFT JOIN read_parquet('{UNIFIED.as_posix()}') AS u
      ON u.uid = c.uid
    """
    df = con.execute(q).df()
    return df


def sum_counts(rows: pd.DataFrame, token_col: str = "tokens", bigram: bool = False) -> Counter:
    """Sum Counters across rows of a dataframe with a ``tokens`` list column."""
    c = Counter()
    if bigram:
        for toks in rows[token_col]:
            c.update(bigrams(toks))
    else:
        for toks in rows[token_col]:
            c.update(toks)
    return c


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def run_fw_labeled(tokens_by_uid: dict, df_a: pd.DataFrame, df_b: pd.DataFrame, bigram: bool = False, top_k: int = 100) -> pd.DataFrame:
    """Helper: build Counter for each subset by uid lookup, then FW."""
    ca: Counter = Counter()
    cb: Counter = Counter()
    for uid in df_a["uid"]:
        toks = tokens_by_uid.get(uid, [])
        if bigram:
            ca.update(bigrams(toks))
        else:
            ca.update(toks)
    for uid in df_b["uid"]:
        toks = tokens_by_uid.get(uid, [])
        if bigram:
            cb.update(bigrams(toks))
        else:
            cb.update(toks)
    res = fighting_words(ca, cb, top_k=top_k)
    res["category"] = res["term"].apply(categorize_term)
    return res


def tag_and_summarize(fw_df: pd.DataFrame, direction_tag: str = "") -> pd.DataFrame:
    fw_df = fw_df.copy()
    fw_df["direction"] = direction_tag
    return fw_df


def save_top(fw_df: pd.DataFrame, path: Path, top_k: int = 100, label_a: str = "A", label_b: str = "B"):
    """Save top-k in each direction with category tags."""
    top_a = fw_df.nlargest(top_k, "z").assign(direction=f"up_in_{label_a}")
    top_b = fw_df.nsmallest(top_k, "z").assign(direction=f"up_in_{label_b}")
    pd.concat([top_a, top_b]).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    t_start = time.time()

    print("Loading main frame...")
    df = load_main_frame()
    print(f"  n = {len(df):,}")

    # Attach J3/S4 flags
    yoe = df["yoe_min_years_llm"]
    labeled = df["llm_classification_coverage"] == "labeled"
    df["is_J3"] = (yoe <= 2) & yoe.notna() & labeled
    df["is_S4"] = (yoe >= 5) & yoe.notna() & labeled
    df["is_J1"] = df["seniority_final"].eq("entry")
    df["is_S1"] = df["seniority_final"].eq("mid-senior")

    # Restrict to llm-cleaned-text rows for primary comparisons (per spec)
    # NOTE: bigram/relabeling passes use same gate
    df_llm = df[df["text_source"] == "llm"].copy()
    print(f"  llm-text rows: {len(df_llm):,}")

    # Tokenize cleaned text
    print("Tokenizing cleaned text...")
    stoplist = load_company_stoplist()
    def tok_clean(s: str) -> list[str]:
        if not s:
            return []
        raw = tokenize(s)
        return [t for t in raw if t not in stoplist and len(t) > 1]

    df_llm["tokens"] = df_llm["description_cleaned"].apply(tok_clean)
    tokens_by_uid = dict(zip(df_llm["uid"], df_llm["tokens"]))

    # -----------------
    # Report text_source distribution
    # -----------------
    print("Text source distribution:")
    dist = df.groupby(["source", "text_source"]).size().reset_index(name="n")
    print(dist.to_string())

    # -----------------
    # 1. PRIMARY: arshkon 2024 vs scraped 2026 (ALL SWE, llm text)
    # -----------------
    print("[1] Primary: arshkon vs scraped — unigrams + bigrams")
    df_arshkon = df_llm[df_llm["source"] == "kaggle_arshkon"]
    df_scraped = df_llm[df_llm["source"] == "scraped"]
    print(f"  arshkon n={len(df_arshkon):,}, scraped n={len(df_scraped):,}")

    # Per spec: "arshkon vs scraped" is direction A=scraped, B=arshkon
    fw_primary_uni = run_fw_labeled(tokens_by_uid, df_scraped, df_arshkon, bigram=False)
    save_top(fw_primary_uni, OUT_TABLES / "primary_arshkon_vs_scraped_unigrams.csv", top_k=100,
             label_a="2026_scraped", label_b="2024_arshkon")

    fw_primary_bi = run_fw_labeled(tokens_by_uid, df_scraped, df_arshkon, bigram=True)
    save_top(fw_primary_bi, OUT_TABLES / "primary_arshkon_vs_scraped_bigrams.csv", top_k=100,
             label_a="2026_scraped", label_b="2024_arshkon")

    # -----------------
    # 2. SECTION-FILTERED: strip boilerplate using T13 classifier
    # -----------------
    print("[2] Section-filtered: requirements + responsibilities only")
    # Use raw description (which has structural markers); cleaned text has stopwords
    # removed but lost section headers because it's a bag-of-tokens. So we filter on
    # raw, then re-tokenize and stopword-strip.
    ENGLISH_STOP = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're",
        "you've", "you'll", "you'd", "your", "yours", "yourself", "yourselves", "he",
        "him", "his", "himself", "she", "she's", "her", "hers", "herself", "it",
        "it's", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "that'll", "these", "those",
        "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and",
        "but", "if", "or", "because", "as", "until", "while", "of", "at", "by",
        "for", "with", "about", "against", "between", "into", "through", "during",
        "before", "after", "above", "below", "to", "from", "up", "down", "in",
        "out", "on", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
        "just", "don", "don't", "should", "should've", "now", "d", "ll", "m", "o",
        "re", "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't", "didn",
        "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven",
        "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't",
        "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn",
        "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't",
    }

    def tok_for_sections(text: str) -> list[str]:
        if not text:
            return []
        toks = tokenize(text)
        return [t for t in toks if t not in stoplist and t not in ENGLISH_STOP and len(t) > 1]

    def section_tokens_for_row(r) -> list[str]:
        raw = r.description_raw
        if not raw:
            return []
        filt = strip_boilerplate_sections(raw)
        return tok_for_sections(filt)

    # Compute section-filtered tokens on arshkon + scraped llm-labeled rows only
    print("  Building section-filtered counters...")
    arshkon_sec_tok: Counter = Counter()
    scraped_sec_tok: Counter = Counter()
    n_arsh_sec = 0
    n_scr_sec = 0

    # We already have df; use df_llm subsets but pull raw desc from the loaded frame
    raw_by_uid = dict(zip(df["uid"], df["description_raw"]))
    for _, r in df_arshkon.iterrows():
        raw = raw_by_uid.get(r["uid"])
        if raw is None:
            continue
        filt = strip_boilerplate_sections(raw)
        toks = tok_for_sections(filt)
        if toks:
            arshkon_sec_tok.update(toks)
            n_arsh_sec += 1
    for _, r in df_scraped.iterrows():
        raw = raw_by_uid.get(r["uid"])
        if raw is None:
            continue
        filt = strip_boilerplate_sections(raw)
        toks = tok_for_sections(filt)
        if toks:
            scraped_sec_tok.update(toks)
            n_scr_sec += 1

    print(f"  arshkon rows w/ sections: {n_arsh_sec}, scraped: {n_scr_sec}")
    fw_sec = fighting_words(scraped_sec_tok, arshkon_sec_tok, top_k=100)
    fw_sec["category"] = fw_sec["term"].apply(categorize_term)
    save_top(fw_sec, OUT_TABLES / "section_filtered_arshkon_vs_scraped.csv",
             top_k=100, label_a="2026_scraped_reqresp", label_b="2024_arshkon_reqresp")

    # -----------------
    # 3. EMERGING / ACCELERATING / DISAPPEARING
    # -----------------
    print("[3] Emerging / accelerating / disappearing terms")
    total_2024 = sum(sum_counts(df_llm[df_llm["period_year"] == "2024"]).values())
    total_2026 = sum(sum_counts(df_llm[df_llm["period_year"] == "2026"]).values())
    counts_2024 = sum_counts(df_llm[df_llm["period_year"] == "2024"])
    counts_2026 = sum_counts(df_llm[df_llm["period_year"] == "2026"])
    rate_2024 = {t: c / total_2024 for t, c in counts_2024.items()}
    rate_2026 = {t: c / total_2026 for t, c in counts_2026.items()}

    EMPTY_EMERGE = pd.DataFrame(columns=[
        "term", "rate_2024", "rate_2026", "count_2024", "count_2026", "category"
    ])
    EMPTY_ACCEL = pd.DataFrame(columns=[
        "term", "rate_2024", "rate_2026", "ratio", "count_2024", "count_2026", "category"
    ])

    emerging_rows = []
    for t, r26 in rate_2026.items():
        r24 = rate_2024.get(t, 0)
        if r26 > 0.001 and r24 < 0.0001 and counts_2026[t] >= 20:
            emerging_rows.append({"term": t, "rate_2024": r24, "rate_2026": r26,
                                  "count_2024": counts_2024.get(t, 0), "count_2026": counts_2026[t],
                                  "category": categorize_term(t)})
    emerge_df = pd.DataFrame(emerging_rows) if emerging_rows else EMPTY_EMERGE.copy()
    if not emerge_df.empty:
        emerge_df = emerge_df.sort_values("rate_2026", ascending=False)
    emerge_df.to_csv(OUT_TABLES / "emerging_terms.csv", index=False)
    print(f"  emerging n={len(emerging_rows)}")

    accel_rows = []
    for t, r26 in rate_2026.items():
        r24 = rate_2024.get(t, 0)
        if r24 > 0 and r26 / max(r24, 1e-12) >= 3 and counts_2024.get(t, 0) >= 10 and counts_2026[t] >= 50:
            accel_rows.append({"term": t, "rate_2024": r24, "rate_2026": r26, "ratio": r26 / r24,
                               "count_2024": counts_2024.get(t, 0), "count_2026": counts_2026[t],
                               "category": categorize_term(t)})
    accel_df = pd.DataFrame(accel_rows) if accel_rows else EMPTY_ACCEL.copy()
    if not accel_df.empty:
        accel_df = accel_df.sort_values("ratio", ascending=False)
    accel_df.to_csv(OUT_TABLES / "accelerating_terms.csv", index=False)
    print(f"  accelerating n={len(accel_rows)}")

    disap_rows = []
    for t, r24 in rate_2024.items():
        r26 = rate_2026.get(t, 0)
        if r24 > 0.001 and r26 < 0.0001 and counts_2024[t] >= 20:
            disap_rows.append({"term": t, "rate_2024": r24, "rate_2026": r26,
                               "count_2024": counts_2024[t], "count_2026": counts_2026.get(t, 0),
                               "category": categorize_term(t)})
    disap_df = pd.DataFrame(disap_rows) if disap_rows else EMPTY_EMERGE.copy()
    if not disap_df.empty:
        disap_df = disap_df.sort_values("rate_2024", ascending=False)
    disap_df.to_csv(OUT_TABLES / "disappearing_terms.csv", index=False)
    print(f"  disappearing n={len(disap_rows)}")

    # -----------------
    # 4. Label-based comparisons
    # -----------------
    print("[4] Label-based")
    df_entry_2024 = df_llm[(df_llm["period_year"] == "2024") & df_llm["is_J3"]]
    df_entry_2026 = df_llm[(df_llm["period_year"] == "2026") & df_llm["is_J3"]]
    df_entry_2024_J1 = df_llm[(df_llm["period_year"] == "2024") & df_llm["is_J1"]]
    df_entry_2026_J1 = df_llm[(df_llm["period_year"] == "2026") & df_llm["is_J1"]]

    df_mid_2024 = df_llm[(df_llm["period_year"] == "2024") & df_llm["is_S4"]]
    df_mid_2026 = df_llm[(df_llm["period_year"] == "2026") & df_llm["is_S4"]]
    df_mid_2024_S1 = df_llm[(df_llm["period_year"] == "2024") & df_llm["is_S1"]]
    df_mid_2026_S1 = df_llm[(df_llm["period_year"] == "2026") & df_llm["is_S1"]]

    print(f"  J3 2024 n={len(df_entry_2024)}, J3 2026 n={len(df_entry_2026)}")
    print(f"  J1 2024 n={len(df_entry_2024_J1)}, J1 2026 n={len(df_entry_2026_J1)}")
    print(f"  S4 2024 n={len(df_mid_2024)}, S4 2026 n={len(df_mid_2026)}")
    print(f"  S1 2024 n={len(df_mid_2024_S1)}, S1 2026 n={len(df_mid_2026_S1)}")

    fw_entry_J3 = run_fw_labeled(tokens_by_uid, df_entry_2026, df_entry_2024)
    save_top(fw_entry_J3, OUT_TABLES / "entry_label_2024_vs_2026.csv", top_k=100,
             label_a="entry_J3_2026", label_b="entry_J3_2024")

    fw_mid_S4 = run_fw_labeled(tokens_by_uid, df_mid_2026, df_mid_2024)
    save_top(fw_mid_S4, OUT_TABLES / "midsenior_label_2024_vs_2026.csv", top_k=100,
             label_a="midsenior_S4_2026", label_b="midsenior_S4_2024")

    # Relabeling diagnostic: entry 2026 vs midsenior 2024
    fw_relabel = run_fw_labeled(tokens_by_uid, df_entry_2026, df_mid_2024)
    save_top(fw_relabel, OUT_TABLES / "entry_2026_vs_midsenior_2024.csv", top_k=100,
             label_a="entry_2026", label_b="midsenior_2024")

    # -----------------
    # 5. YOE-band comparisons
    # -----------------
    print("[5] YOE-band")
    df_yoe_le2_2024 = df_llm[(df_llm["period_year"] == "2024") & df_llm["is_J3"]]
    df_yoe_le2_2026 = df_llm[(df_llm["period_year"] == "2026") & df_llm["is_J3"]]
    df_yoe_ge5_2024 = df_llm[(df_llm["period_year"] == "2024") & df_llm["is_S4"]]
    df_yoe_ge5_2026 = df_llm[(df_llm["period_year"] == "2026") & df_llm["is_S4"]]
    # These are identical to J3/S4 label-based in our setup since J3/S4 ARE YOE-based
    # (they are YOE-based primaries — label-independent). Save them anyway so the
    # disagreement diagnostic against J1/S1 is clear in downstream reading.
    fw_yoe_le2_period = run_fw_labeled(tokens_by_uid, df_yoe_le2_2026, df_yoe_le2_2024)
    save_top(fw_yoe_le2_period, OUT_TABLES / "yoe_le2_2024_vs_2026.csv", top_k=100,
             label_a="yoe_le2_2026", label_b="yoe_le2_2024")

    fw_yoe_ge5_period = run_fw_labeled(tokens_by_uid, df_yoe_ge5_2026, df_yoe_ge5_2024)
    save_top(fw_yoe_ge5_period, OUT_TABLES / "yoe_ge5_2024_vs_2026.csv", top_k=100,
             label_a="yoe_ge5_2026", label_b="yoe_ge5_2024")

    fw_yoe_diag = run_fw_labeled(tokens_by_uid, df_yoe_le2_2026, df_yoe_ge5_2024)
    save_top(fw_yoe_diag, OUT_TABLES / "yoe_le2_2026_vs_yoe_ge5_2024.csv", top_k=100,
             label_a="yoe_le2_2026", label_b="yoe_ge5_2024")

    # -----------------
    # 6. Within-2024 calibration: arshkon S1 vs asaniczka S1
    # -----------------
    print("[6] Within-2024 calibration: arshkon midsenior vs asaniczka midsenior")
    df_arsh_mid = df_llm[(df_llm["source"] == "kaggle_arshkon") & df_llm["is_S1"]]
    df_asan_mid = df_llm[(df_llm["source"] == "kaggle_asaniczka") & df_llm["is_S1"]]
    print(f"  arshkon S1 n={len(df_arsh_mid)}, asaniczka S1 n={len(df_asan_mid)}")
    if len(df_arsh_mid) >= 100 and len(df_asan_mid) >= 100:
        fw_within = run_fw_labeled(tokens_by_uid, df_arsh_mid, df_asan_mid)
        save_top(fw_within, OUT_TABLES / "within_2024_arshkon_vs_asaniczka_midsenior.csv",
                 top_k=100, label_a="arshkon_mid", label_b="asaniczka_mid")

    # -----------------
    # 7. Category summary
    # -----------------
    print("[7] Category summaries")
    def cat_summary_for(fw_df: pd.DataFrame, top_k: int = 100):
        top = fw_df.nlargest(top_k, "z").assign(direction="up")
        bot = fw_df.nsmallest(top_k, "z").assign(direction="down")
        both = pd.concat([top, bot])
        summary = both.groupby(["direction", "category"]).size().unstack(fill_value=0)
        return summary

    summaries = {
        "primary_unigrams": cat_summary_for(fw_primary_uni),
        "primary_bigrams": cat_summary_for(fw_primary_bi),
        "section_filtered": cat_summary_for(fw_sec),
        "entry_J3": cat_summary_for(fw_entry_J3),
        "midsenior_S4": cat_summary_for(fw_mid_S4),
        "relabel_entry2026_vs_mid2024": cat_summary_for(fw_relabel),
        "yoe_le2_diagnostic": cat_summary_for(fw_yoe_diag),
    }
    # Flatten for CSV
    summary_rows = []
    for name, tbl in summaries.items():
        for direction in tbl.index:
            for cat in tbl.columns:
                summary_rows.append({"comparison": name, "direction": direction,
                                     "category": cat, "n_terms_in_top100": int(tbl.loc[direction, cat])})
    pd.DataFrame(summary_rows).to_csv(OUT_TABLES / "category_summary.csv", index=False)

    # -----------------
    # Comparison summary (n per corpus)
    # -----------------
    print("[8] Comparison summary (n per corpus)")
    cs_rows = [
        {"comparison": "primary: 2026_scraped vs 2024_arshkon", "n_a": len(df_scraped), "n_b": len(df_arshkon)},
        {"comparison": "section_filtered: same (reqresp only)", "n_a": n_scr_sec, "n_b": n_arsh_sec},
        {"comparison": "entry J3: 2026 vs 2024", "n_a": len(df_entry_2026), "n_b": len(df_entry_2024)},
        {"comparison": "entry J1: 2026 vs 2024", "n_a": len(df_entry_2026_J1), "n_b": len(df_entry_2024_J1)},
        {"comparison": "midsenior S4: 2026 vs 2024", "n_a": len(df_mid_2026), "n_b": len(df_mid_2024)},
        {"comparison": "midsenior S1: 2026 vs 2024", "n_a": len(df_mid_2026_S1), "n_b": len(df_mid_2024_S1)},
        {"comparison": "relabel: entry 2026 vs mid 2024 (J3/S4)", "n_a": len(df_entry_2026), "n_b": len(df_mid_2024)},
        {"comparison": "yoe-based relabel: yoe_le2 2026 vs yoe_ge5 2024", "n_a": len(df_yoe_le2_2026), "n_b": len(df_yoe_ge5_2024)},
        {"comparison": "within-2024: arshkon S1 vs asaniczka S1", "n_a": len(df_arsh_mid), "n_b": len(df_asan_mid)},
    ]
    pd.DataFrame(cs_rows).to_csv(OUT_TABLES / "comparison_summary.csv", index=False)

    # -----------------
    # 8. BERTopic cross-validation (lightweight, on sampled subset)
    # -----------------
    print("[8b] BERTopic cross-validation (sampled, 3-min time cap)")
    try:
        t_bert_start = time.time()
        import numpy as np_local
        import pyarrow.parquet as pq_local
        EMB = REPO / "exploration" / "artifacts" / "shared" / "swe_embeddings.npy"
        IDX = REPO / "exploration" / "artifacts" / "shared" / "swe_embedding_index.parquet"
        if EMB.exists() and IDX.exists():
            emb = np_local.load(EMB)
            idx = pq_local.read_table(IDX).to_pandas()
            # Sample balanced by period: 2500 per period
            idx_df = idx.merge(
                df_llm[["uid", "period_year", "description_cleaned"]], on="uid", how="inner"
            )
            sampled = (
                idx_df.groupby("period_year", group_keys=False)
                .apply(lambda g: g.sample(n=min(2500, len(g)), random_state=42))
                .reset_index(drop=True)
            )
            sampled_emb = emb[sampled["row_idx"].values]
            docs = sampled["description_cleaned"].tolist()
            classes = sampled["period_year"].tolist()
            print(f"  BERTopic sample: {len(docs)} docs (2024 + 2026)")
            from bertopic import BERTopic
            from umap import UMAP
            from hdbscan import HDBSCAN
            from sklearn.feature_extraction.text import CountVectorizer
            topic_model = BERTopic(
                umap_model=UMAP(n_neighbors=15, n_components=5, metric="cosine", random_state=42),
                hdbscan_model=HDBSCAN(min_cluster_size=30, metric="euclidean", prediction_data=True),
                vectorizer_model=CountVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2)),
                calculate_probabilities=False,
                verbose=False,
            )
            topics, _ = topic_model.fit_transform(docs, embeddings=sampled_emb)
            # Topics per class
            topics_per_class = topic_model.topics_per_class(docs, classes=classes)
            # Save top-20 period-specific topics
            tpc = topics_per_class.copy()
            # For each topic, compute frequency difference 2026 - 2024
            pivot = tpc.pivot_table(index="Topic", columns="Class", values="Frequency", aggfunc="sum", fill_value=0)
            if "2024" in pivot.columns and "2026" in pivot.columns:
                pivot["diff_2026_minus_2024"] = pivot["2026"] - pivot["2024"]
                # Attach topic words
                topic_words = []
                for t in pivot.index:
                    if t == -1:
                        topic_words.append("NOISE")
                        continue
                    words = topic_model.get_topic(t)
                    topic_words.append(", ".join(w for w, _ in (words or [])[:8]))
                pivot["top_words"] = topic_words
                pivot = pivot.sort_values("diff_2026_minus_2024", ascending=False)
                pivot.to_csv(OUT_TABLES / "bertopic_topics_per_class.csv")
                print(f"  BERTopic saved ({len(pivot)} topics) in {time.time()-t_bert_start:.0f}s")
            else:
                print("  Warning: BERTopic pivot missing expected classes, skipping")
        else:
            print(f"  embeddings not found at {EMB}")
    except Exception as e:
        print(f"  BERTopic skipped due to: {type(e).__name__}: {e}")

    # -----------------
    # Figures
    # -----------------
    print("[9] Figures")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Category stacked bar: primary unigrams up/down + section-filtered up/down
    def stacked_bar(ax, fw_df: pd.DataFrame, title: str, top_k: int = 100):
        top = fw_df.nlargest(top_k, "z").assign(dir="up_in_2026")
        bot = fw_df.nsmallest(top_k, "z").assign(dir="up_in_2024")
        both = pd.concat([top, bot])
        tbl = both.groupby(["dir", "category"]).size().unstack(fill_value=0)
        tbl = tbl.reindex(columns=sorted(tbl.columns), fill_value=0)
        bottoms = np.zeros(len(tbl))
        colors = plt.cm.tab20.colors
        for i, c in enumerate(tbl.columns):
            ax.barh(tbl.index, tbl[c].values, left=bottoms, label=c, color=colors[i % len(colors)])
            bottoms += tbl[c].values
        ax.set_title(title)
        ax.set_xlabel("Count of top-100 distinguishing terms")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    stacked_bar(axes[0], fw_primary_uni, "Full-text primary unigrams (top 100 each direction)")
    stacked_bar(axes[1], fw_sec, "Section-filtered (requirements + responsibilities) unigrams")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "category_stacked_bar.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Relabeling diagnostic figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    stacked_bar(axes[0], fw_relabel,
                "Entry 2026 vs Mid-senior 2024 (J3 vs S4)\nLabel-based relabeling diagnostic")
    stacked_bar(axes[1], fw_yoe_diag,
                "YOE ≤ 2 (2026) vs YOE ≥ 5 (2024)\nYOE-based relabeling diagnostic")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "relabeling_diagnostic.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s. Tables in {OUT_TABLES}, figures in {OUT_FIG}")


if __name__ == "__main__":
    main()
