"""T12 — Open-ended text evolution.

Fightin' Words (log-odds with informative Dirichlet prior, Monroe et al. 2008)
comparing:
  (primary) arshkon 2024 vs scraped 2026 — all SWE
  (filtered) same comparison on requirements + responsibilities sections only
  (secondary) entry 2024 vs entry 2026 (J2)
  (secondary) mid-senior 2024 vs mid-senior 2026
  (relabeling diagnostic) entry 2026 vs mid-senior 2024
  (within-2024 calibration) arshkon mid-senior vs asaniczka mid-senior

Also reports:
  - emerging terms (>1% 2026, <0.1% 2024)
  - accelerating (existing, >3× growth)
  - disappearing (>1% 2024, <0.1% 2026)
  - bigram analysis

Uses T13_section_classifier for section-filtered text.
"""
from __future__ import annotations

import os
import re
import sys
import json
from collections import Counter

import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, "/home/jihgaboot/gabor/job-research/exploration/scripts")
from T13_section_classifier import classify_description, filter_sections, SECTION_TYPES

ROOT = "/home/jihgaboot/gabor/job-research"
CLEANED = f"{ROOT}/exploration/artifacts/shared/swe_cleaned_text.parquet"
UNIFIED = f"{ROOT}/data/unified.parquet"
SPECIALISTS = f"{ROOT}/exploration/artifacts/shared/entry_specialist_employers.csv"
TABLES = f"{ROOT}/exploration/tables/T12"
FIGS = f"{ROOT}/exploration/figures/T12"
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

# English stopwords to strip — we don't want "the", "and", etc. dominating.
STOPWORDS = set("""
a about above across after afterwards again against all almost alone along
already also although always am among amongst amount an and another any
anyhow anyone anything anyway anywhere are around as at back be became
because become becomes becoming been before beforehand behind being below
beside besides between beyond both but by can cannot can't could couldn't
did didn't do does doesn't doing don't done down due during each either else
elsewhere empty enough even ever every everyone everything everywhere except
few first for former formerly from further get give go had has hasn't have
haven't having he hence her here hereafter hereby herein hereupon hers
herself him himself his how however hundred i if in indeed into is isn't it
its itself keep last latter latterly least less made make many may me
meanwhile might mine more moreover most mostly move much must my myself
name namely neither never nevertheless next nine no nobody none noone nor
not nothing now nowhere of off often on once one only onto or other others
otherwise our ours ourselves out over own part per perhaps please put
quite rather re really regarding same say see seem seemed seeming seems
serious several she should shouldn't show side since six so some somehow
someone something sometime sometimes somewhere still such take ten than
that the their theirs them themselves then thence there thereafter thereby
therefore therein thereupon these they thing things this those though
three through throughout thru thus to together too top toward towards two
under until up upon us used using various very via was wasn't we well went
were weren't what whatever when whence whenever where whereafter whereas
whereby wherein whereupon wherever whether which while whither who whoever
whole whom whose why will with within without would wouldn't yet you your
yours yourself yourselves
""".split())
# Add SWE-common noise words that drown signal
STOPWORDS.update("""
job role work working team company please click apply provide provided
providing description experience experienced based using use used uses utilizing
skills skill years year required preferred ability able etc ensure ensuring
will well new people strong develop developing developed development
responsibilities responsibility requirements qualifications
needed need needs including include includes
member members roles candidate candidates applicant applicants employee employees
environment environments eligible
also per any other others etc etc.
""".split())

# Category dictionary — used to tag each term with a semantic category.
# Each key is a category, each value is a set of regex strings.
CATEGORY_PATTERNS = {
    "ai_tool": [
        r"^copilot$", r"^cursor$", r"^codex$", r"^chatgpt$", r"^claude$",
        r"^openai$", r"^anthropic$", r"^gemini$", r"^bard$", r"^ollama$",
        r"^langchain$", r"^langgraph$", r"^llamaindex$", r"^huggingface$",
        r"^hugging$", r"^mcp$", r"agent_framework", r"^pinecone$", r"^chromadb$",
        r"^assistant", r"^bot$", r"^bots$", r"autogen", r"crewai",
    ],
    "ai_domain": [
        r"^ai$", r"^llm$", r"^llms$", r"^genai$", r"^gen_ai$", r"generative",
        r"embedding", r"transformer", r"diffusion", r"prompt", r"prompting",
        r"fine_tuning", r"finetun", r"rlhf", r"rag$", r"retrieval",
        r"vector", r"agentic", r"agent$", r"agents$", r"multimodal",
        r"foundation_model", r"inference", r"^ml$", r"machine_learning",
    ],
    "tech_stack": [
        r"^python$", r"^java$", r"^javascript$", r"^typescript$", r"^go$", r"^golang$",
        r"^rust$", r"^cpp$", r"^csharp$", r"^c\+\+$", r"^c#$", r"^ruby$",
        r"^kotlin$", r"^swift$", r"^scala$", r"^php$", r"^react$", r"^angular$",
        r"^vue$", r"^nextjs$", r"^django$", r"^flask$", r"^fastapi$", r"^spring$",
        r"^nodejs$", r"^node$", r"^aws$", r"^azure$", r"^gcp$", r"kubernetes",
        r"^docker$", r"terraform", r"^ansible$", r"^jenkins$", r"^mysql$",
        r"postgres", r"^mongodb$", r"^redis$", r"^kafka$", r"^spark$",
        r"^snowflake$", r"^databricks$", r"^dbt$", r"^elasticsearch$", r"^html$",
        r"^css$", r"^linux$", r"^windows$", r"^sql$", r"^graphql$", r"^rest$",
        r"^api$", r"^apis$", r"microservice", r"^ci$", r"^cd$", r"^devops$",
        r"tensorflow", r"^pytorch$", r"^pandas$", r"^numpy$", r"jupyter",
        r"xgboost", r"^cicd$", r"pipelines?$", r"serverless", r"lambda",
        r"^cloud$", r"^database$", r"databases",
    ],
    "org_scope": [
        r"^impact$", r"^scale$", r"^scalable$", r"cross_functional", r"stakeholders?",
        r"partners?", r"strategic", r"vision$", r"mission", r"global",
        r"organization", r"enterprise", r"companywide", r"business$",
        r"objectives?", r"outcomes?", r"initiative", r"leadership",
    ],
    "mgmt": [
        r"^lead$", r"^leader$", r"^leaders$", r"manage$", r"manages$", r"manager",
        r"management", r"mentor", r"mentoring", r"mentors", r"coach",
        r"coaching", r"hire", r"hiring", r"oversee", r"supervise",
        r"direct_report", r"reports?$", r"performance_review",
    ],
    "sys_design": [
        r"architecture", r"architect", r"distributed$", r"system_design",
        r"scalability", r"performance", r"optimize", r"optimization",
        r"reliability", r"availability", r"fault_tolerance", r"resilience",
        r"high_throughput", r"low_latency", r"concurrency", r"parallel",
    ],
    "method": [
        r"agile", r"^scrum$", r"kanban", r"^sprint$", r"sprints?", r"standup",
        r"retrospective", r"tdd", r"bdd", r"pair_programming",
        r"code_review", r"reviews$", r"estimation", r"planning",
        r"iteration", r"iterative", r"roadmap",
    ],
    "credential": [
        r"^bachelor", r"^master", r"^phd$", r"^doctorate$", r"degree$",
        r"^bs$", r"^ms$", r"^ba$", r"^mba$", r"^certified$", r"certification",
        r"clearance", r"accredited",
    ],
    "soft_skill": [
        r"communicat", r"collaborat", r"present", r"interpersonal", r"verbal",
        r"written", r"^problem$", r"problem_solv", r"analytical", r"critical",
        r"creative", r"adaptable", r"adaptability", r"proactive", r"motivated",
        r"self_start", r"self_direct", r"team_player", r"curious", r"curiosity",
        r"ownership", r"accountable", r"accountability", r"passion",
    ],
    "boilerplate": [
        r"^benefit", r"^perk", r"^pto$", r"^401k$", r"^insurance$", r"health",
        r"dental", r"vision$", r"^equity$", r"^bonus$", r"^salary$",
        r"compensation", r"^range$", r"^rate$", r"^pay$", r"^remote$",
        r"^hybrid$", r"^office$", r"^location$", r"^eeo$", r"^equal$",
        r"opportunity", r"affirmative", r"veteran", r"disability", r"accommodat",
        r"^legal$", r"statement", r"discrimin", r"background_check", r"drug",
        r"policy", r"^policies$", r"^ada$", r"without_regard",
    ],
    "noise": [
        r"^\d+$", r"^[a-z]$", r"^\d+k$",
    ],
}


def categorize(term: str) -> str:
    """Return the category name whose pattern list first matches `term`."""
    t = term.lower()
    for cat, patterns in CATEGORY_PATTERNS.items():
        for p in patterns:
            if re.search(p, t):
                return cat
    return "other"


# ---------------------------------------------------------------------------
# Log-odds ratio with informative Dirichlet prior (Monroe et al.)
# ---------------------------------------------------------------------------
def fightin_words(counts_i: np.ndarray, counts_j: np.ndarray,
                  prior: np.ndarray) -> np.ndarray:
    """Return the Z-score array for each term (positive = more characteristic
    of corpus i, negative = more characteristic of corpus j).

    Formula (Monroe 2008):
      n_i = sum(counts_i), n_j = sum(counts_j), alpha_0 = sum(prior)
      log_odds_i = log((y_iw + prior_w) / (n_i + alpha_0 - y_iw - prior_w))
      log_odds_j = log((y_jw + prior_w) / (n_j + alpha_0 - y_jw - prior_w))
      delta = log_odds_i - log_odds_j
      var(delta) ≈ 1/(y_iw + prior_w) + 1/(y_jw + prior_w)
      z = delta / sqrt(var)
    """
    n_i = counts_i.sum()
    n_j = counts_j.sum()
    alpha0 = prior.sum()
    num_i = counts_i + prior
    num_j = counts_j + prior
    # guard against negatives in log denominator
    denom_i = np.maximum(n_i + alpha0 - num_i, 1e-6)
    denom_j = np.maximum(n_j + alpha0 - num_j, 1e-6)
    lo_i = np.log(num_i) - np.log(denom_i)
    lo_j = np.log(num_j) - np.log(denom_j)
    delta = lo_i - lo_j
    var = 1.0 / num_i + 1.0 / num_j
    z = delta / np.sqrt(var)
    return z


def _build_corpus(df: pd.DataFrame, text_col: str = "description_cleaned") -> list[str]:
    """Extract non-empty text rows from a filtered dataframe."""
    texts = df[text_col].dropna().astype(str).tolist()
    return [t for t in texts if len(t) > 50]


def _tokenize_and_count(corpus_i: list[str], corpus_j: list[str],
                        ngram_range=(1, 1), min_df: int = 10,
                        max_features: int = 20000):
    """Vectorize both corpora and return term counts + vocabulary."""
    vec = CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        token_pattern=r"\b[a-zA-Z][a-zA-Z0-9+#.\-]{1,}\b",
        lowercase=True,
        stop_words=list(STOPWORDS),
    )
    all_texts = corpus_i + corpus_j
    X = vec.fit_transform(all_texts)
    vocab = np.array(vec.get_feature_names_out())
    counts_i = np.asarray(X[: len(corpus_i)].sum(axis=0)).ravel()
    counts_j = np.asarray(X[len(corpus_i):].sum(axis=0)).ravel()
    # drop tokens matching "noise" category (pure numbers, single chars)
    keep = np.array([not re.match(r"^(\d+|\w)$", v) for v in vocab])
    return vocab[keep], counts_i[keep], counts_j[keep]


def run_compare(corpus_a: list[str], corpus_b: list[str], label_a: str, label_b: str,
                ngram_range=(1, 1), min_df: int = 10, max_features: int = 20000,
                top_n: int = 100) -> pd.DataFrame:
    """Run fightin words comparison; return a DataFrame with top-N terms per
    side, ranked by Z-score, with counts and relative shares."""
    vocab, cn_a, cn_b = _tokenize_and_count(
        corpus_a, corpus_b, ngram_range=ngram_range, min_df=min_df,
        max_features=max_features)
    if vocab.size == 0:
        return pd.DataFrame()
    # informative prior = total corpus counts (Monroe recommends use of the
    # background frequency as the prior — we use the sum of both corpora).
    prior = (cn_a + cn_b).astype(float)
    prior = prior * (1.0 / max(prior.sum(), 1) * 1000.0)  # scale to small prior
    # We use prior[w] = pseudocounts scaled so total alpha0 ≈ 1000.
    z = fightin_words(cn_a.astype(float), cn_b.astype(float), prior)
    n_a = max(cn_a.sum(), 1)
    n_b = max(cn_b.sum(), 1)
    share_a = cn_a / n_a
    share_b = cn_b / n_b
    df = pd.DataFrame({
        "term": vocab,
        "z_score": z,
        f"count_{label_a}": cn_a,
        f"count_{label_b}": cn_b,
        f"share_{label_a}": share_a,
        f"share_{label_b}": share_b,
        "ratio_a_over_b": np.where(share_b > 0, share_a / np.maximum(share_b, 1e-10), np.inf),
    })
    df["category"] = [categorize(t) for t in vocab]
    # Top-N each direction
    top_a = df.sort_values("z_score", ascending=False).head(top_n)
    top_b = df.sort_values("z_score", ascending=True).head(top_n).iloc[::-1]
    top_a["direction"] = label_a
    top_b["direction"] = label_b
    out = pd.concat([top_a, top_b], ignore_index=True)
    return out


def category_summary(top_df: pd.DataFrame, label_a: str, label_b: str) -> pd.DataFrame:
    rows = []
    for d in [label_a, label_b]:
        sub = top_df[top_df["direction"] == d]
        total = len(sub)
        counts = sub["category"].value_counts()
        for cat, n in counts.items():
            rows.append({"direction": d, "category": cat, "n": int(n),
                         "share": n / total})
    return pd.DataFrame(rows).sort_values(["direction", "n"], ascending=[True, False])


def category_bar(cat_summary: pd.DataFrame, path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    piv = cat_summary.pivot_table(index="category", columns="direction",
                                   values="share", fill_value=0).sort_values(
        cat_summary["direction"].iloc[0], ascending=False
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    piv.plot(kind="bar", ax=ax)
    ax.set_ylabel("Share of top-100 distinguishing terms")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Emerging / accelerating / disappearing
# ---------------------------------------------------------------------------
def posting_presence_share(corpus: list[str], vocab: list[str]) -> dict:
    """For each term, share of postings containing it (binary 0/1 per posting).
    Uses simple regex on each posting."""
    out = {}
    # Precompile token regexes; this is O(V*N) — 20k terms × 40k postings.
    # We keep V small. For speed, we build one big combined regex of the vocab
    # and count unique matches per posting.
    vocab_set = set(vocab)
    N = len(corpus)
    presence = Counter()
    for post in corpus:
        # Simple tokenization (same pattern as CountVectorizer)
        toks = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.\-]{1,}\b", post.lower()))
        for t in toks:
            if t in vocab_set:
                presence[t] += 1
    for v in vocab:
        out[v] = presence[v] / N
    return out


def emerging_accelerating_disappearing(
    corpus_2024: list[str], corpus_2026: list[str],
    vocab: np.ndarray, cn_2024: np.ndarray, cn_2026: np.ndarray,
    companies_2024: list[set[str]], companies_2026: list[set[str]],
    threshold_emerging_2026: float = 0.01,
    threshold_emerging_2024_max: float = 0.001,
    threshold_accel_ratio: float = 3.0,
    threshold_disappearing_2024: float = 0.01,
    threshold_disappearing_2026_max: float = 0.001,
    min_companies: int = 20,
):
    """Compute posting-presence shares in both eras and return 3 DataFrames."""
    # posting presence = # postings containing term / total postings per era
    # We tokenize both corpora once each.
    def _presence(corp: list[str]):
        N = len(corp)
        presence = Counter()
        per_posting_companies = []  # not used per-term; we compute term-company separately
        vocab_set = set(vocab.tolist())
        for post in corp:
            toks = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.\-]{1,}\b", post.lower()))
            for t in toks:
                if t in vocab_set:
                    presence[t] += 1
        return {v: presence[v] / N for v in vocab}, presence

    p24, raw24 = _presence(corpus_2024)
    p26, raw26 = _presence(corpus_2026)

    # distinct-company count per term per era
    def _term_company_counts(corp: list[str], companies: list[set[str]]):
        vocab_set = set(vocab.tolist())
        term_companies = {v: set() for v in vocab}
        for post, cn in zip(corp, companies):
            toks = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.\-]{1,}\b", post.lower()))
            for t in toks:
                if t in vocab_set:
                    term_companies[t].add(cn)
        return {v: len(term_companies[v]) for v in vocab}

    tc24 = _term_company_counts(corpus_2024, companies_2024)
    tc26 = _term_company_counts(corpus_2026, companies_2026)

    rows_emerging, rows_accel, rows_disappear = [], [], []
    for v in vocab:
        a = p24.get(v, 0.0)
        b = p26.get(v, 0.0)
        n_c_26 = tc26.get(v, 0)
        n_c_24 = tc24.get(v, 0)
        ratio = b / max(a, 1e-10)
        # emerging
        if b >= threshold_emerging_2026 and a <= threshold_emerging_2024_max \
           and n_c_26 >= min_companies:
            rows_emerging.append({
                "term": v, "share_2024": a, "share_2026": b,
                "n_companies_2026": n_c_26, "category": categorize(v),
            })
        # accelerating
        if a > 0 and ratio >= threshold_accel_ratio and n_c_26 >= min_companies \
           and b >= 0.005:
            rows_accel.append({
                "term": v, "share_2024": a, "share_2026": b, "ratio_26_over_24": ratio,
                "n_companies_2026": n_c_26, "category": categorize(v),
            })
        # disappearing
        if a >= threshold_disappearing_2024 and b <= threshold_disappearing_2026_max \
           and n_c_24 >= min_companies:
            rows_disappear.append({
                "term": v, "share_2024": a, "share_2026": b,
                "n_companies_2024": n_c_24, "category": categorize(v),
            })
    em = pd.DataFrame(rows_emerging)
    ac = pd.DataFrame(rows_accel)
    di = pd.DataFrame(rows_disappear)
    if len(em):
        em = em.sort_values("share_2026", ascending=False)
    if len(ac):
        ac = ac.sort_values("ratio_26_over_24", ascending=False)
    if len(di):
        di = di.sort_values("share_2024", ascending=False)
    return em, ac, di


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("[T12] loading cleaned text")
    df = pd.read_parquet(CLEANED)
    specialists = set(pd.read_csv(SPECIALISTS)["company"].unique())
    df = df[~df["company_name_canonical"].isin(specialists)]
    df = df[df["is_aggregator"].fillna(False) == False]
    # Restrict to llm-cleaned text
    df_llm = df[df["text_source"] == "llm"].copy()
    print(f"[T12] total llm-cleaned rows: {len(df_llm)}")
    print(df_llm.groupby("source").size())

    # --- Primary comparison: arshkon vs scraped ---
    corpus_arshkon = _build_corpus(df_llm[df_llm["source"] == "kaggle_arshkon"])
    corpus_scraped = _build_corpus(df_llm[df_llm["source"] == "scraped"])
    print(f"[T12] primary: arshkon n={len(corpus_arshkon)}, scraped n={len(corpus_scraped)}")

    primary = run_compare(corpus_arshkon, corpus_scraped, "arshkon_2024", "scraped_2026",
                          ngram_range=(1, 1), min_df=20, top_n=100)
    primary.to_csv(f"{TABLES}/primary_arshkon_vs_scraped_top100.csv", index=False)
    print(f"[T12] wrote primary_arshkon_vs_scraped_top100.csv ({len(primary)} rows)")

    primary_cat = category_summary(primary, "arshkon_2024", "scraped_2026")
    primary_cat.to_csv(f"{TABLES}/primary_category_summary.csv", index=False)
    category_bar(primary_cat, f"{FIGS}/primary_category_share.png",
                 "Top-100 distinguishing terms by category (arshkon 2024 vs scraped 2026)")

    # --- Section-filtered comparison: requirements + responsibilities only ---
    print("[T12] building section-filtered texts")
    # For each posting, extract requirements + responsibilities text from raw
    # description using the classifier. We use raw description because section
    # headers live in raw text; results are later joined with the text_source
    # filter.
    c = duckdb.connect()
    meta = c.execute(
        "SELECT uid, description FROM read_parquet('"
        + UNIFIED +
        "') WHERE source_platform='linkedin' AND is_english=true "
        "AND date_flag='ok' AND is_swe=true"
    ).fetchdf()
    meta["req_resp_text"] = meta["description"].apply(
        lambda d: filter_sections(d or "", ["requirements", "responsibilities"])
    )
    merged = df_llm[["uid", "source", "period", "company_name_canonical",
                     "seniority_final"]].merge(
        meta[["uid", "req_resp_text"]], on="uid", how="left")
    # use only non-empty req_resp
    merged["req_resp_text"] = merged["req_resp_text"].fillna("").astype(str)
    merged = merged[merged["req_resp_text"].str.len() > 50]
    print(f"[T12] section-filtered rows (req+resp nonempty): {len(merged)}")
    print(merged.groupby("source").size())

    corpus_arshkon_sec = merged[merged["source"] == "kaggle_arshkon"]["req_resp_text"].tolist()
    corpus_scraped_sec = merged[merged["source"] == "scraped"]["req_resp_text"].tolist()
    filtered = run_compare(corpus_arshkon_sec, corpus_scraped_sec,
                           "arshkon_2024", "scraped_2026",
                           ngram_range=(1, 1), min_df=20, top_n=100)
    filtered.to_csv(f"{TABLES}/section_filtered_arshkon_vs_scraped_top100.csv", index=False)
    filtered_cat = category_summary(filtered, "arshkon_2024", "scraped_2026")
    filtered_cat.to_csv(f"{TABLES}/section_filtered_category_summary.csv", index=False)
    category_bar(filtered_cat, f"{FIGS}/section_filtered_category_share.png",
                 "Top-100 distinguishing terms (req+resp only)")

    # --- Compare: full-text vs section-filtered top terms ---
    primary_terms_2026 = set(primary[primary["direction"] == "scraped_2026"]["term"])
    filtered_terms_2026 = set(filtered[filtered["direction"] == "scraped_2026"]["term"])
    only_in_full = primary_terms_2026 - filtered_terms_2026
    only_in_filtered = filtered_terms_2026 - primary_terms_2026
    in_both = primary_terms_2026 & filtered_terms_2026
    print(f"[T12] 2026-characteristic: both={len(in_both)}, full_only={len(only_in_full)}, filtered_only={len(only_in_filtered)}")

    boilerplate_vs_genuine = pd.DataFrame([
        {"bucket": "in_both_genuine", "terms": ",".join(sorted(in_both))},
        {"bucket": "only_full_boilerplate_driven", "terms": ",".join(sorted(only_in_full))},
        {"bucket": "only_filtered", "terms": ",".join(sorted(only_in_filtered))},
    ])
    boilerplate_vs_genuine.to_csv(f"{TABLES}/boilerplate_vs_genuine_buckets.csv", index=False)

    # --- Emerging / accelerating / disappearing on ALL SWE corpora ---
    print("[T12] computing emerging/accelerating/disappearing")
    # Pooled 2024 = arshkon + asaniczka; 2026 = scraped
    df_2024 = df_llm[df_llm["source"].isin(["kaggle_arshkon", "kaggle_asaniczka"])]
    df_2026 = df_llm[df_llm["source"] == "scraped"]
    corpus_2024 = _build_corpus(df_2024)
    corpus_2026 = _build_corpus(df_2026)
    companies_2024 = df_2024["company_name_canonical"].fillna("").tolist()
    companies_2026 = df_2026["company_name_canonical"].fillna("").tolist()
    # Need vocab / counts
    vocab, cn_2024, cn_2026 = _tokenize_and_count(
        corpus_2024, corpus_2026, ngram_range=(1, 1), min_df=20, max_features=50000)
    em, ac, di = emerging_accelerating_disappearing(
        corpus_2024, corpus_2026, vocab, cn_2024, cn_2026,
        companies_2024, companies_2026,
    )
    em.head(200).to_csv(f"{TABLES}/emerging_terms.csv", index=False)
    ac.head(200).to_csv(f"{TABLES}/accelerating_terms.csv", index=False)
    di.head(200).to_csv(f"{TABLES}/disappearing_terms.csv", index=False)
    print(f"[T12] emerging={len(em)}, accelerating={len(ac)}, disappearing={len(di)}")

    # --- Secondary comparisons ---
    # helper
    def sub_by(src, senior_set):
        m = (df_llm["source"] == src) & df_llm["seniority_final"].isin(senior_set)
        return _build_corpus(df_llm[m])
    def sub_by_multi(srcs, senior_set):
        m = df_llm["source"].isin(srcs) & df_llm["seniority_final"].isin(senior_set)
        return _build_corpus(df_llm[m])

    # Entry 2024 pooled vs Entry 2026 (J2)
    entry24 = sub_by_multi(["kaggle_arshkon", "kaggle_asaniczka"], ["entry", "associate"])
    entry26 = sub_by("scraped", ["entry", "associate"])
    print(f"[T12] entry_24={len(entry24)} entry_26={len(entry26)}")
    if len(entry24) >= 100 and len(entry26) >= 100:
        df_e = run_compare(entry24, entry26, "entry_2024", "entry_2026", min_df=10, top_n=100)
        df_e.to_csv(f"{TABLES}/secondary_entry_2024_vs_entry_2026.csv", index=False)
        category_summary(df_e, "entry_2024", "entry_2026").to_csv(
            f"{TABLES}/secondary_entry_category.csv", index=False)

    # Mid-senior 2024 pooled vs Mid-senior 2026
    midsr24 = sub_by_multi(["kaggle_arshkon", "kaggle_asaniczka"], ["mid-senior"])
    midsr26 = sub_by("scraped", ["mid-senior"])
    print(f"[T12] midsenior_24={len(midsr24)} midsenior_26={len(midsr26)}")
    if len(midsr24) >= 100 and len(midsr26) >= 100:
        df_m = run_compare(midsr24, midsr26, "midsr_2024", "midsr_2026", min_df=20, top_n=100)
        df_m.to_csv(f"{TABLES}/secondary_midsr_2024_vs_midsr_2026.csv", index=False)
        category_summary(df_m, "midsr_2024", "midsr_2026").to_csv(
            f"{TABLES}/secondary_midsr_category.csv", index=False)

    # Relabeling diagnostic: entry 2026 vs mid-senior 2024
    if len(entry26) >= 100 and len(midsr24) >= 100:
        df_r = run_compare(midsr24, entry26, "midsr_2024", "entry_2026",
                           min_df=10, top_n=100)
        df_r.to_csv(f"{TABLES}/secondary_entry26_vs_midsr24_relabeling.csv", index=False)
        category_summary(df_r, "midsr_2024", "entry_2026").to_csv(
            f"{TABLES}/secondary_relabeling_category.csv", index=False)
        # Distance metric: how similar is entry26 to midsr24 vs entry24?
        # Use a simple cosine similarity on top tokens.

    # Within-2024 calibration: arshkon mid-senior vs asaniczka mid-senior
    arm = sub_by("kaggle_arshkon", ["mid-senior"])
    asm = sub_by("kaggle_asaniczka", ["mid-senior"])
    print(f"[T12] arshkon_msr={len(arm)} asaniczka_msr={len(asm)}")
    if len(arm) >= 100 and len(asm) >= 100:
        df_w = run_compare(arm, asm, "arshkon_msr_2024", "asaniczka_msr_2024",
                           min_df=10, top_n=100)
        df_w.to_csv(f"{TABLES}/secondary_within_2024_midsr_calibration.csv", index=False)

    # --- Bigram analysis (primary pair) ---
    print("[T12] bigrams primary")
    df_bg = run_compare(corpus_arshkon, corpus_scraped, "arshkon_2024", "scraped_2026",
                        ngram_range=(2, 2), min_df=20, top_n=100)
    df_bg.to_csv(f"{TABLES}/primary_bigrams_arshkon_vs_scraped_top100.csv", index=False)

    # --- Relabeling diagnostic metric: cosine similarity of term-share vectors ---
    def share_vec(corp: list[str], vocab: list[str]):
        vocab_set = set(vocab)
        cnt = Counter()
        total = 0
        for post in corp:
            toks = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.\-]{1,}\b", post.lower())
            for t in toks:
                if t in vocab_set:
                    cnt[t] += 1
                    total += 1
        if total == 0:
            return np.zeros(len(vocab))
        return np.array([cnt[v] / total for v in vocab])

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb + 1e-10))

    # Build a shared vocabulary across the three cells for the comparison
    # Use vocab from the primary comparison for stability.
    shared_vocab = list(vocab[:10000])  # keep it manageable
    v_entry24 = share_vec(entry24, shared_vocab)
    v_entry26 = share_vec(entry26, shared_vocab)
    v_midsr24 = share_vec(midsr24, shared_vocab)
    v_midsr26 = share_vec(midsr26, shared_vocab)
    v_all24 = share_vec(corpus_2024, shared_vocab)
    v_all26 = share_vec(corpus_2026, shared_vocab)

    relabel = pd.DataFrame([
        {"pair": "entry24_vs_entry26", "cosine": cos(v_entry24, v_entry26)},
        {"pair": "entry26_vs_midsr24", "cosine": cos(v_entry26, v_midsr24)},
        {"pair": "entry26_vs_midsr26", "cosine": cos(v_entry26, v_midsr26)},
        {"pair": "entry24_vs_midsr24", "cosine": cos(v_entry24, v_midsr24)},
        {"pair": "midsr24_vs_midsr26", "cosine": cos(v_midsr24, v_midsr26)},
        {"pair": "all24_vs_all26", "cosine": cos(v_all24, v_all26)},
    ])
    relabel.to_csv(f"{TABLES}/relabeling_diagnostic_cosine.csv", index=False)
    print("[T12] relabeling cosines:")
    print(relabel.round(4).to_string(index=False))

    # Summary of n per cell
    sizes = {
        "arshkon_2024 (primary A)": len(corpus_arshkon),
        "scraped_2026 (primary B)": len(corpus_scraped),
        "arshkon_section_filtered": len(corpus_arshkon_sec),
        "scraped_section_filtered": len(corpus_scraped_sec),
        "entry_2024_pooled": len(entry24),
        "entry_2026_scraped": len(entry26),
        "midsenior_2024_pooled": len(midsr24),
        "midsenior_2026_scraped": len(midsr26),
        "arshkon_midsenior_2024": len(arm),
        "asaniczka_midsenior_2024": len(asm),
    }
    with open(f"{TABLES}/cell_sizes.json", "w") as f:
        json.dump(sizes, f, indent=2)
    print("[T12] cell sizes:")
    print(json.dumps(sizes, indent=2))

    print("[T12] done.")


if __name__ == "__main__":
    main()
