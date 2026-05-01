"""T12 — Open-ended text evolution (Fightin' Words + BERTopic).

Compares SWE LinkedIn posting text between 2024 (arshkon + asaniczka) and
2026 (scraped), using:
    1. Log-odds ratio with informative Dirichlet prior (Monroe et al., 2008)
       on full text vs core-sections-only (responsibilities + requirements +
       preferred + summary), to separate genuine content evolution from
       boilerplate expansion.
    2. Emerging / accelerating / disappearing term lists.
    3. Bigram analysis.
    4. BERTopic with period as a class variable.
    5. Secondary comparisons: entry vs mid-senior, entry 2026 vs mid 2024,
       within-2024 calibration (arshkon vs asaniczka).

Inputs:
    - exploration/artifacts/shared/swe_cleaned_text.parquet (metadata)
    - data/unified.parquet (raw text via COALESCE(description_core_llm, ...))
    - exploration/tables/T13/section_chars_per_posting.parquet (from T13)
    - exploration/artifacts/shared/company_stoplist.txt
    - exploration/artifacts/shared/section_classifier.py

Outputs:
    exploration/tables/T12/ and exploration/figures/T12/
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/home/jihgaboot/gabor/job-research")
SHARED = ROOT / "exploration" / "artifacts" / "shared"
OUT_T = ROOT / "exploration" / "tables" / "T12"
OUT_F = ROOT / "exploration" / "figures" / "T12"
OUT_T.mkdir(parents=True, exist_ok=True)
OUT_F.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SHARED))
from section_classifier import extract_sections  # noqa: E402

# ---- Config ----
SEED = 42
TOP_K = 100
MIN_DOC_FREQ = 20          # unigram must appear in at least 20 postings overall
MIN_DOC_FREQ_BG = 15       # bigram threshold
COMPANY_CAP_STRIP = True   # strip company tokens

# ---- Semantic category taxonomy ----
CATEGORIES: Dict[str, List[str]] = {
    "ai_tool": [
        "copilot", "github copilot", "cursor", "windsurf", "codeium", "tabnine",
        "claude", "chatgpt", "gpt4", "gpt-4", "gpt 4", "gemini", "llama",
        "codewhisperer", "sourcegraph", "cody", "aider", "devin", "codex",
        "openai api", "anthropic api",
    ],
    "ai_domain": [
        "llm", "llms", "large language", "generative", "genai", "gen ai",
        "prompt engineering", "prompt", "rag", "retrieval augmented",
        "embeddings", "vector database", "vector db", "fine tuning",
        "fine-tuning", "transformer", "transformers", "agent", "agents",
        "agentic", "ai agent", "multi agent", "llmops", "mlops", "model",
        "models", "hallucination", "chain of thought", "few shot", "zero shot",
        "foundation model", "inference", "token", "tokens", "context window",
    ],
    "tech_stack": [
        "python", "java", "javascript", "typescript", "go", "golang", "rust",
        "c++", "cpp", "scala", "ruby", "php", "kotlin", "swift", "react",
        "angular", "vue", "node", "nodejs", "django", "flask", "fastapi",
        "spring", "rails", "express", "nextjs", "nuxt", "aws", "azure", "gcp",
        "kubernetes", "docker", "terraform", "jenkins", "git", "github",
        "gitlab", "postgres", "postgresql", "mysql", "mongodb", "redis",
        "kafka", "spark", "hadoop", "snowflake", "databricks", "airflow",
        "tensorflow", "pytorch", "scikit", "sklearn", "pandas", "numpy",
        "jupyter", "rest", "graphql", "grpc", "microservices", "serverless",
        "lambda", "s3", "ec2", "rds", "dynamodb", "linux", "unix", "bash",
        "shell", "sql", "nosql", "ci", "cd", "cicd",
    ],
    "org_scope": [
        "lead", "leading", "leadership", "mentor", "mentoring", "mentorship",
        "cross functional", "stakeholder", "stakeholders", "strategy",
        "strategic", "roadmap", "ownership", "own", "drive", "influence",
        "architecture", "architect", "designing", "technical leadership",
        "staff", "principal", "senior", "director", "vp", "head of",
    ],
    "mgmt": [
        "manage", "management", "managing", "direct report", "direct reports",
        "people manager", "engineering manager", "team lead", "hiring",
        "performance review", "one on one", "1:1", "reports to", "team size",
    ],
    "sys_design": [
        "system design", "distributed systems", "distributed", "scalability",
        "scalable", "high availability", "throughput", "latency", "sla",
        "load balancing", "caching", "sharding", "replication", "consistency",
        "eventual consistency", "cap theorem", "database design", "api design",
        "schema", "event driven", "pub sub", "message queue", "streaming",
        "observability", "monitoring", "logging", "tracing", "metrics",
        "reliability", "resilience", "fault tolerance",
    ],
    "method": [
        "agile", "scrum", "kanban", "waterfall", "tdd", "bdd", "xp",
        "pair programming", "code review", "pull request", "merge request",
        "sprint", "standup", "retrospective", "planning poker",
        "continuous integration", "continuous deployment", "devops", "sre",
        "incident response", "on call", "oncall", "postmortem",
    ],
    "credential": [
        "bachelor", "bachelors", "master", "masters", "phd", "degree",
        "computer science", "cs degree", "equivalent experience", "certified",
        "certification", "certifications", "clearance", "security clearance",
        "ts sci", "secret clearance", "gpa", "graduate", "undergraduate",
    ],
    "soft_skill": [
        "communication", "collaborative", "collaboration", "team player",
        "passionate", "self starter", "self motivated", "problem solving",
        "problem solver", "analytical", "detail oriented", "customer focused",
        "results driven", "proactive", "adaptable", "curious", "eager",
        "fast paced", "growth mindset", "willing to learn", "quick learner",
    ],
    "boilerplate": [
        "benefits", "benefit", "salary", "compensation", "bonus", "equity",
        "stock", "rsus", "401k", "401 k", "ira", "dental", "vision",
        "medical", "health", "insurance", "pto", "paid time off", "vacation",
        "holiday", "holidays", "parental leave", "maternity", "paternity",
        "wellness", "gym", "retirement", "commuter", "sponsorship", "visa",
        "eoe", "eeo", "equal opportunity", "affirmative action", "diversity",
        "inclusion", "inclusive", "culture", "mission", "values", "vision",
        "our story", "our team", "headquartered", "founded", "fortune 500",
        "fortune", "great place to work", "glassdoor", "remote", "hybrid",
        "flexible", "onsite", "work from home", "wfh", "accommodation",
        "disability", "protected", "veterans", "veteran",
    ],
    "noise": [
        "please", "note", "notes", "opportunity", "opportunities", "join",
        "joining", "work", "working", "role", "roles", "position",
        "positions", "candidate", "candidates", "apply", "application",
        "looking", "hire", "hires", "hiring",
    ],
}


def categorize_term(term: str) -> str:
    t = term.lower()
    for cat, vocab in CATEGORIES.items():
        if t in vocab:
            return cat
    # Suffix/prefix heuristics
    if any(k in t for k in ["llm", "gpt", "ai ", "genai", "agent", "prompt", "rag"]):
        return "ai_domain"
    return "other"


# ---- Data loading ----

def load_raw_text() -> pd.DataFrame:
    print("[load] metadata from swe_cleaned_text.parquet ...")
    meta = pq.read_table(SHARED / "swe_cleaned_text.parquet").to_pandas()
    meta = meta.drop(columns=["description_cleaned"])
    meta["period_bucket"] = meta["period"].map(
        lambda p: "2024" if str(p).startswith("2024") else "2026"
    )

    print("[load] raw text + seniority from unified.parquet ...")
    con = duckdb.connect()
    q = """
        SELECT
            uid,
            seniority_llm,
            llm_classification_coverage,
            CASE
                WHEN llm_classification_coverage = 'labeled'         THEN seniority_llm
                WHEN llm_classification_coverage = 'rule_sufficient' THEN seniority_final
                ELSE NULL
            END AS seniority_best_available,
            COALESCE(description_core_llm, description_core, description) AS raw_text
        FROM read_parquet(?)
        WHERE source_platform = 'linkedin' AND is_english = true
          AND date_flag = 'ok' AND is_swe = true
    """
    meta2 = con.execute(q, [str(ROOT / "data" / "unified.parquet")]).fetchdf()
    df = meta.merge(meta2, on="uid", how="left")
    con.close()
    df["yoe_entry_proxy"] = (df["yoe_extracted"].fillna(-1) <= 2) & (df["yoe_extracted"].fillna(-1) >= 0)
    return df


def load_stoplist() -> set:
    with open(SHARED / "company_stoplist.txt") as f:
        company = {line.strip().lower() for line in f if line.strip()}
    # Base English stopwords
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        eng = set(ENGLISH_STOP_WORDS)
    except Exception:
        eng = set()
    extra = {
        "job", "role", "work", "team", "company", "employee", "employees",
        "position", "candidate", "office", "year", "years", "experience",
        "skills", "skill", "ability", "able", "new", "will", "must", "should",
        "may", "also", "using", "used", "use", "one", "two", "three", "include",
        "including", "etc", "e.g", "e.g.", "i.e", "i.e.", "us", "well",
        "across", "within", "ensure", "provide", "provides", "working", "join",
        "looking", "want", "wanted", "opportunity", "opportunities", "apply",
    }
    return company | eng | extra


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#\.\-]{1,30}")

# Scraped text uses markdown escapes: C\+\+, F\#, .NET\-Core, list bullets as \*.
# These backslashes break tokenization (C++ -> "c" + "+" + "+"). Pre-clean
# the text by removing backslash escapes before tokenizing.
_MD_ESCAPE_RE = re.compile(r"\\([*+#.\-!()\[\]_`])")


def tokenize(text: str, stoplist: set) -> List[str]:
    if not text:
        return []
    # Unescape markdown so "C\+\+" -> "C++" before tokenization.
    text = _MD_ESCAPE_RE.sub(r"\1", text)
    toks = _TOKEN_RE.findall(text.lower())
    out = []
    for t in toks:
        if len(t) < 3:
            continue
        if t in stoplist:
            continue
        if t.isdigit():
            continue
        out.append(t)
    return out


def make_bigrams(tokens: List[str]) -> List[str]:
    return [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


# ---- Fightin' Words (log-odds with informative Dirichlet prior) ----

def fightin_words(
    counts_a: Counter,
    counts_b: Counter,
    prior: Counter,
    min_freq: int = MIN_DOC_FREQ,
) -> pd.DataFrame:
    """Compute log-odds ratio z-scores per Monroe et al. 2008.

    A positive z favors corpus B (e.g. 2026), negative favors A (2024).
    """
    vocab = set(counts_a) | set(counts_b)
    # Apply min-frequency filter on combined counts.
    vocab = {t for t in vocab if counts_a[t] + counts_b[t] >= min_freq}
    n_a = sum(counts_a.values())
    n_b = sum(counts_b.values())
    a0 = sum(prior.values())
    rows = []
    for w in vocab:
        f_aw = counts_a[w]
        f_bw = counts_b[w]
        a_aw = prior.get(w, 0)
        # Dirichlet-smoothed log odds for each corpus
        num_a = f_aw + a_aw
        den_a = n_a + a0 - num_a
        num_b = f_bw + a_aw
        den_b = n_b + a0 - num_b
        if num_a <= 0 or den_a <= 0 or num_b <= 0 or den_b <= 0:
            continue
        log_odds_a = np.log(num_a / den_a)
        log_odds_b = np.log(num_b / den_b)
        delta = log_odds_b - log_odds_a  # positive -> favors B (2026)
        var = (1.0 / num_a) + (1.0 / num_b)
        z = delta / np.sqrt(var)
        rows.append({
            "term": w,
            "count_a": f_aw,
            "count_b": f_bw,
            "rate_a_per_1k": f_aw / max(n_a, 1) * 1000,
            "rate_b_per_1k": f_bw / max(n_b, 1) * 1000,
            "log_odds_delta": delta,
            "z": z,
        })
    df = pd.DataFrame(rows)
    df["category"] = df["term"].map(categorize_term)
    df["abs_z"] = df["z"].abs()
    return df.sort_values("z", ascending=False).reset_index(drop=True)


def build_corpus_counts(
    df: pd.DataFrame,
    text_col: str,
    stoplist: set,
    use_bigrams: bool = False,
) -> Counter:
    """Tokenize a subcorpus and return term counts (document frequency)."""
    counts = Counter()
    for t in df[text_col].fillna(""):
        toks = tokenize(t, stoplist)
        if use_bigrams:
            terms = set(make_bigrams(toks))
        else:
            terms = set(toks)
        counts.update(terms)
    return counts


def build_prior(*counters: Counter) -> Counter:
    """Pool counters as Dirichlet prior (uniform smoothing in effect)."""
    prior = Counter()
    for c in counters:
        prior.update(c)
    # Soften: halve the prior (Monroe default uses combined corpus as prior).
    for k in list(prior.keys()):
        prior[k] = max(1, prior[k] // 2)
    return prior


# ---- Main analysis routines ----

def primary_comparison(
    df_a: pd.DataFrame, df_b: pd.DataFrame, text_col: str, label: str,
    stoplist: set, bigrams: bool = False,
) -> pd.DataFrame:
    print(f"[fw] {label}: tokenizing corpus A (n={len(df_a)}), B (n={len(df_b)}) ...")
    ca = build_corpus_counts(df_a, text_col, stoplist, use_bigrams=bigrams)
    cb = build_corpus_counts(df_b, text_col, stoplist, use_bigrams=bigrams)
    prior = build_prior(ca, cb)
    res = fightin_words(
        ca, cb, prior,
        min_freq=MIN_DOC_FREQ_BG if bigrams else MIN_DOC_FREQ,
    )
    return res


def emerging_accelerating_disappearing(
    df_a: pd.DataFrame, df_b: pd.DataFrame, text_col: str, stoplist: set,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Terms emerging/accelerating/disappearing between 2024 and 2026."""
    print("[ead] computing document-frequency rates ...")
    n_a = len(df_a)
    n_b = len(df_b)
    ca = Counter()
    cb = Counter()
    for t in df_a[text_col].fillna(""):
        ca.update(set(tokenize(t, stoplist)))
    for t in df_b[text_col].fillna(""):
        cb.update(set(tokenize(t, stoplist)))

    vocab = set(ca) | set(cb)
    rows = []
    for w in vocab:
        if len(w) < 3:
            continue
        a_freq = ca[w] / n_a
        b_freq = cb[w] / n_b
        if ca[w] + cb[w] < MIN_DOC_FREQ:
            continue
        rows.append({
            "term": w,
            "df_2024_pct": a_freq * 100,
            "df_2026_pct": b_freq * 100,
            "ratio": (b_freq + 1e-6) / (a_freq + 1e-6),
            "category": categorize_term(w),
        })
    df = pd.DataFrame(rows)
    emerging = df[(df["df_2024_pct"] < 0.1) & (df["df_2026_pct"] >= 1.0)].sort_values("df_2026_pct", ascending=False)
    accelerating = df[(df["df_2024_pct"] >= 0.1) & (df["ratio"] >= 3.0) & (df["df_2026_pct"] >= 1.0)].sort_values("ratio", ascending=False)
    disappearing = df[(df["df_2024_pct"] >= 1.0) & (df["df_2026_pct"] < 0.1)].sort_values("df_2024_pct", ascending=False)
    return emerging, accelerating, disappearing


def category_summary(fw_df: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
    """% of top-K distinguishing terms in each direction by semantic category."""
    top_b = fw_df.head(top_k)
    top_a = fw_df.tail(top_k).iloc[::-1]  # reverse for readability
    rows = []
    for direction, subset in [("2026_favored", top_b), ("2024_favored", top_a)]:
        counts = subset["category"].value_counts()
        for cat, n in counts.items():
            rows.append({"direction": direction, "category": cat, "n": int(n), "pct": float(n / top_k * 100)})
    return pd.DataFrame(rows)


def plot_category_summary(cat_full: pd.DataFrame, cat_core: pd.DataFrame, out_path: Path):
    """Grouped bar chart of category shares: full text vs core-filtered, two directions."""
    cats = sorted(set(cat_full["category"]) | set(cat_core["category"]))
    directions = ["2024_favored", "2026_favored"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, direction in zip(axes, directions):
        full_pct = {c: 0.0 for c in cats}
        core_pct = {c: 0.0 for c in cats}
        for _, r in cat_full[cat_full["direction"] == direction].iterrows():
            full_pct[r["category"]] = r["pct"]
        for _, r in cat_core[cat_core["direction"] == direction].iterrows():
            core_pct[r["category"]] = r["pct"]
        x = np.arange(len(cats))
        ax.bar(x - 0.2, [full_pct[c] for c in cats], width=0.4, label="Full text")
        ax.bar(x + 0.2, [core_pct[c] for c in cats], width=0.4, label="Core sections only")
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"Top-100 {direction}")
        ax.set_ylabel("% of top terms")
        ax.legend(fontsize=8)
    fig.suptitle("T12 — Semantic category share of top distinguishing terms")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[fig] wrote {out_path}")


def run_bertopic(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Fit BERTopic on a sample; compute per-period topic distribution.

    Uses precomputed shared embeddings when possible to match the rest of
    the pipeline. Tunes HDBSCAN/UMAP for a finer-grained topic model than
    the BERTopic defaults (which produced one mega-topic on SWE postings).
    """
    try:
        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
    except Exception as e:
        print(f"[bertopic] skipping: {e}")
        return pd.DataFrame()
    # Load precomputed embeddings to save compute.
    try:
        all_emb = np.load(SHARED / "swe_embeddings.npy")
        emb_idx = pq.read_table(SHARED / "swe_embedding_index.parquet").to_pandas()
        uid_to_row = {u: i for i, u in enumerate(emb_idx["uid"].tolist())}
    except Exception as e:
        print(f"[bertopic] cannot load shared embeddings: {e}")
        return pd.DataFrame()

    rng = np.random.default_rng(SEED)
    target = 8000
    frames = []
    for p, grp in df.groupby("period_bucket"):
        k = min(len(grp), target // 2)
        idx = rng.choice(grp.index.values, size=k, replace=False)
        frames.append(df.loc[idx])
    sub = pd.concat(frames, ignore_index=True)
    sub = sub[sub["uid"].isin(uid_to_row)]
    docs = sub[text_col].fillna("").tolist()
    docs = [d[:3000] for d in docs]
    rows = [uid_to_row[u] for u in sub["uid"].tolist()]
    emb = all_emb[rows]
    print(f"[bertopic] fitting on {len(docs)} docs with shared embeddings ...")
    try:
        umap_model = UMAP(
            n_neighbors=15, n_components=5, min_dist=0.0,
            metric="cosine", random_state=SEED,
        )
        hdb = HDBSCAN(
            min_cluster_size=50, min_samples=10,
            metric="euclidean", cluster_selection_method="eom",
        )
        cv = CountVectorizer(
            ngram_range=(1, 2), stop_words="english", min_df=5,
        )
        bt = BERTopic(
            embedding_model=None,
            umap_model=umap_model,
            hdbscan_model=hdb,
            vectorizer_model=cv,
            min_topic_size=50,
            verbose=False,
            calculate_probabilities=False,
        )
        topics, _ = bt.fit_transform(docs, embeddings=emb)
    except Exception as e:
        print(f"[bertopic] failed: {e}")
        return pd.DataFrame()
    sub = sub.copy()
    sub["topic"] = topics
    # Cross-tab topic x period
    xt = pd.crosstab(sub["topic"], sub["period_bucket"], normalize="columns") * 100
    xt = xt.reset_index()
    # Attach topic labels
    info = bt.get_topic_info()[["Topic", "Name", "Count"]]
    xt = xt.merge(info, left_on="topic", right_on="Topic", how="left")
    xt["delta"] = xt["2026"] - xt["2024"]
    xt = xt.sort_values("delta", ascending=False)
    return xt


def main():
    df = load_raw_text()
    stoplist = load_stoplist()
    print(f"[main] loaded {len(df):,} rows")

    # Load section counts and re-derive core text per posting.
    print("[core] extracting core sections per posting ...")
    core_texts = []
    for i, t in enumerate(df["raw_text"].fillna("")):
        if i % 10000 == 0:
            print(f"  [core] {i}/{len(df)}")
        secs = extract_sections(t)
        core = " ".join([secs["summary"], secs["responsibilities"],
                         secs["requirements"], secs["preferred"]]).strip()
        core_texts.append(core)
    df["core_text"] = core_texts

    df_24 = df[df["period_bucket"] == "2024"]
    df_26 = df[df["period_bucket"] == "2026"]
    print(f"  n 2024={len(df_24):,}, n 2026={len(df_26):,}")

    # ---- Primary: full text, arshkon+asaniczka 2024 vs scraped 2026 ----
    print("\n=== PRIMARY full-text unigrams ===")
    fw_full = primary_comparison(df_24, df_26, "raw_text", "primary-full", stoplist)
    fw_full.head(TOP_K).to_csv(OUT_T / "fw_primary_full_top100_2026.csv", index=False)
    fw_full.tail(TOP_K).iloc[::-1].to_csv(OUT_T / "fw_primary_full_top100_2024.csv", index=False)
    fw_full.to_csv(OUT_T / "fw_primary_full_all.csv", index=False)

    # ---- Core-filtered ----
    print("\n=== PRIMARY core-filtered unigrams ===")
    fw_core = primary_comparison(df_24, df_26, "core_text", "primary-core", stoplist)
    fw_core.head(TOP_K).to_csv(OUT_T / "fw_primary_core_top100_2026.csv", index=False)
    fw_core.tail(TOP_K).iloc[::-1].to_csv(OUT_T / "fw_primary_core_top100_2024.csv", index=False)
    fw_core.to_csv(OUT_T / "fw_primary_core_all.csv", index=False)

    # Category summaries
    cat_full = category_summary(fw_full)
    cat_core = category_summary(fw_core)
    cat_full.to_csv(OUT_T / "category_summary_full.csv", index=False)
    cat_core.to_csv(OUT_T / "category_summary_core.csv", index=False)
    plot_category_summary(cat_full, cat_core, OUT_F / "category_summary.png")

    # Full-vs-core delta: which terms appear only in full-text top 100?
    top_full_2026 = set(fw_full.head(TOP_K)["term"])
    top_core_2026 = set(fw_core.head(TOP_K)["term"])
    only_full = top_full_2026 - top_core_2026
    only_core = top_core_2026 - top_full_2026
    shared = top_full_2026 & top_core_2026
    max_len = max(len(only_full), len(only_core), len(shared))
    def pad(lst):
        lst = sorted(lst)
        return lst + [""] * (max_len - len(lst))
    deltas = pd.DataFrame({
        "full_only_2026": pad(only_full),
        "core_only_2026": pad(only_core),
        "shared_2026": pad(shared),
    })
    deltas.to_csv(OUT_T / "fw_full_vs_core_top100_overlap_2026.csv", index=False)
    print(f"[delta] 2026 top-100 overlap: {len(shared)} shared, {len(only_full)} full-only, {len(only_core)} core-only")

    # ---- Emerging / Accelerating / Disappearing ----
    print("\n=== Emerging / Accelerating / Disappearing (full text) ===")
    emerging, accelerating, disappearing = emerging_accelerating_disappearing(
        df_24, df_26, "raw_text", stoplist
    )
    emerging.to_csv(OUT_T / "emerging_terms.csv", index=False)
    accelerating.to_csv(OUT_T / "accelerating_terms.csv", index=False)
    disappearing.to_csv(OUT_T / "disappearing_terms.csv", index=False)
    print(f"[ead] emerging={len(emerging)}, accelerating={len(accelerating)}, disappearing={len(disappearing)}")

    # ---- Bigrams ----
    print("\n=== Bigrams (full text) ===")
    fw_bigrams = primary_comparison(df_24, df_26, "raw_text", "bigrams", stoplist, bigrams=True)
    fw_bigrams.head(TOP_K).to_csv(OUT_T / "fw_bigrams_top100_2026.csv", index=False)
    fw_bigrams.tail(TOP_K).iloc[::-1].to_csv(OUT_T / "fw_bigrams_top100_2024.csv", index=False)

    print("\n=== Bigrams (core-filtered) ===")
    fw_bigrams_core = primary_comparison(df_24, df_26, "core_text", "bigrams-core", stoplist, bigrams=True)
    fw_bigrams_core.head(TOP_K).to_csv(OUT_T / "fw_bigrams_core_top100_2026.csv", index=False)
    fw_bigrams_core.tail(TOP_K).iloc[::-1].to_csv(OUT_T / "fw_bigrams_core_top100_2024.csv", index=False)

    # ---- Secondary: entry vs mid, entry 2026 vs mid 2024 ----
    print("\n=== SECONDARY comparisons ===")
    entry_mask_best_24 = (df_24["seniority_best_available"].isin(["entry", "associate"]))
    entry_mask_best_26 = (df_26["seniority_best_available"].isin(["entry", "associate"]))
    mid_mask_best_24 = (df_24["seniority_best_available"] == "mid-senior")
    mid_mask_best_26 = (df_26["seniority_best_available"] == "mid-senior")

    # Report n per corpus
    ns = {
        "entry_2024_combined": int(entry_mask_best_24.sum()),
        "entry_2026_combined": int(entry_mask_best_26.sum()),
        "mid_2024_combined": int(mid_mask_best_24.sum()),
        "mid_2026_combined": int(mid_mask_best_26.sum()),
        "entry_yoe_2024": int(df_24["yoe_entry_proxy"].sum()),
        "entry_yoe_2026": int(df_26["yoe_entry_proxy"].sum()),
    }
    print(f"[n] {ns}")

    if ns["entry_2024_combined"] >= 100 and ns["entry_2026_combined"] >= 100:
        fw_entry = primary_comparison(
            df_24[entry_mask_best_24], df_26[entry_mask_best_26],
            "core_text", "entry_vs_entry_core", stoplist,
        )
        fw_entry.head(50).to_csv(OUT_T / "fw_entry_2024_vs_2026_core_top50_2026.csv", index=False)
        fw_entry.tail(50).iloc[::-1].to_csv(OUT_T / "fw_entry_2024_vs_2026_core_top50_2024.csv", index=False)

    if ns["mid_2024_combined"] >= 100 and ns["mid_2026_combined"] >= 100:
        fw_mid = primary_comparison(
            df_24[mid_mask_best_24], df_26[mid_mask_best_26],
            "core_text", "mid_vs_mid_core", stoplist,
        )
        fw_mid.head(50).to_csv(OUT_T / "fw_mid_2024_vs_2026_core_top50_2026.csv", index=False)
        fw_mid.tail(50).iloc[::-1].to_csv(OUT_T / "fw_mid_2024_vs_2026_core_top50_2024.csv", index=False)

    # Relabeling diagnostic: entry 2026 vs mid-senior 2024
    if ns["entry_2026_combined"] >= 100 and ns["mid_2024_combined"] >= 100:
        fw_relabel = primary_comparison(
            df_24[mid_mask_best_24], df_26[entry_mask_best_26],
            "core_text", "entry26_vs_mid24_core", stoplist,
        )
        fw_relabel.head(50).to_csv(OUT_T / "fw_entry26_vs_mid24_top50_entry26.csv", index=False)
        fw_relabel.tail(50).iloc[::-1].to_csv(OUT_T / "fw_entry26_vs_mid24_top50_mid24.csv", index=False)

    # Within-2024 calibration: arshkon vs asaniczka (mid-senior only)
    print("\n=== WITHIN-2024 calibration (arshkon mid vs asaniczka mid) ===")
    mid_ars = df_24[(df_24["source"] == "kaggle_arshkon") & (df_24["seniority_final"] == "mid-senior")]
    mid_asn = df_24[(df_24["source"] == "kaggle_asaniczka") & (df_24["seniority_final"] == "mid-senior")]
    if len(mid_ars) >= 100 and len(mid_asn) >= 100:
        fw_w24 = primary_comparison(mid_ars, mid_asn, "core_text", "within_2024_mid", stoplist)
        fw_w24.head(50).to_csv(OUT_T / "fw_within2024_top50_asaniczka.csv", index=False)
        fw_w24.tail(50).iloc[::-1].to_csv(OUT_T / "fw_within2024_top50_arshkon.csv", index=False)

    # Save n table
    pd.Series(ns).to_csv(OUT_T / "comparison_sample_sizes.csv")

    # ---- BERTopic ----
    print("\n=== BERTopic ===")
    bt = run_bertopic(df, "core_text")
    if not bt.empty:
        bt.to_csv(OUT_T / "bertopic_topic_period_share.csv", index=False)
        top5 = bt.head(20)
        print("Top 20 topics by 2026 - 2024 delta:")
        print(top5[["topic", "Name", "2024", "2026", "delta"]].to_string(index=False))

    # ---- Headline summary ----
    print("\n========= T12 HEADLINE =========")
    print(f"Full-text top-10 2026-favored: {fw_full.head(10)['term'].tolist()}")
    print(f"Full-text top-10 2024-favored: {fw_full.tail(10)['term'].tolist()}")
    print(f"Core-filtered top-10 2026-favored: {fw_core.head(10)['term'].tolist()}")
    print(f"Core-filtered top-10 2024-favored: {fw_core.tail(10)['term'].tolist()}")

    # Write headline md
    with open(OUT_T / "headline_numbers.md", "w") as f:
        f.write("# T12 headline\n\n")
        f.write(f"Full-text top-25 2026-favored:\n{fw_full.head(25)[['term','z','count_a','count_b','category']].to_string(index=False)}\n\n")
        f.write(f"Full-text top-25 2024-favored:\n{fw_full.tail(25).iloc[::-1][['term','z','count_a','count_b','category']].to_string(index=False)}\n\n")
        f.write(f"Core-filtered top-25 2026-favored:\n{fw_core.head(25)[['term','z','count_a','count_b','category']].to_string(index=False)}\n\n")
        f.write(f"Core-filtered top-25 2024-favored:\n{fw_core.tail(25).iloc[::-1][['term','z','count_a','count_b','category']].to_string(index=False)}\n\n")
        f.write(f"Top emerging terms: {emerging.head(20)['term'].tolist()}\n\n")
        f.write(f"Top accelerating terms: {accelerating.head(20)['term'].tolist()}\n\n")
        f.write(f"Top disappearing terms: {disappearing.head(20)['term'].tolist()}\n\n")
    print(f"[out] wrote headline")
    print("\n[done] T12 complete")


if __name__ == "__main__":
    main()
