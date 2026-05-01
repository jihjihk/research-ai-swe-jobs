#!/usr/bin/env python
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from scipy.special import logsumexp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from umap import UMAP

from T13_text_utils import (
    CORE_SECTION_LABELS,
    is_html_artifact,
    load_stop_tokens,
    location_tokens_from_values,
    normalize_text,
    pretty_term,
    term_category,
    tokenize_for_terms,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
TEXT = ROOT / "exploration" / "artifacts" / "shared" / "swe_cleaned_text.parquet"
STOPLIST = ROOT / "exploration" / "artifacts" / "shared" / "company_stoplist.txt"
SECTION_SPANS = ROOT / "exploration" / "artifacts" / "shared" / "t13_section_spans.parquet"
EMBED_IDX = ROOT / "exploration" / "artifacts" / "shared" / "swe_embedding_index.parquet"
EMBEDDINGS = ROOT / "exploration" / "artifacts" / "shared" / "swe_embeddings.npy"

REPORT_DIR = ROOT / "exploration" / "reports"
TABLE_DIR = ROOT / "exploration" / "tables" / "T12"
FIG_DIR = ROOT / "exploration" / "figures" / "T12"

CAP = 25
PRIMARY_SOURCES = ("kaggle_arshkon", "scraped")
CALIBRATION_SOURCES = ("kaggle_arshkon", "kaggle_asaniczka")
TOP_K = 100
BIGRAM_TOP_K = 50
MIN_DOC_FREQ = 20
MIN_COMPANIES = 20


def ensure_dirs() -> None:
    for path in [REPORT_DIR, TABLE_DIR, FIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def qdf(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def safe_div(n: float, d: float) -> float:
    if d in (0, None) or (isinstance(d, float) and math.isnan(d)):
        return float("nan")
    return float(n) / float(d)


def assert_regex_hygiene() -> None:
    cpp = re.compile(r"(?:^|[^a-z0-9])c\+\+(?:[^a-z0-9]|$)", re.I)
    csharp = re.compile(r"(?:^|[^a-z0-9])c#(?:[^a-z0-9]|$)", re.I)
    dotnet = re.compile(r"(?:^|[^a-z0-9])\.net(?:[^a-z0-9]|$)", re.I)
    assert cpp.search("C++ developer")
    assert csharp.search("C# engineer")
    assert dotnet.search(".NET platform")
    assert not cpp.search("company")


def load_location_stopset(con: duckdb.DuckDBPyConnection) -> set[str]:
    values = set()
    for column in ["metro_area", "state_normalized"]:
        rows = con.execute(
            f"""
            SELECT DISTINCT {column} AS value
            FROM read_parquet('{DATA.as_posix()}')
            WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe
              AND {column} IS NOT NULL
            """
        ).fetchall()
        values.update(row[0] for row in rows if row and row[0])
    return location_tokens_from_values(values)


def build_stop_tokens(con: duckdb.DuckDBPyConnection) -> set[str]:
    location_tokens = load_location_stopset(con)
    stop_tokens = load_stop_tokens(STOPLIST, location_tokens)
    stop_tokens.update(normalize_text(x) for x in ENGLISH_STOP_WORDS)
    stop_tokens.update(
        {
            "job",
            "jobs",
            "position",
            "positions",
            "role",
            "roles",
            "candidate",
            "candidates",
            "company",
            "companies",
            "opportunity",
            "opportunities",
            "work",
            "working",
        }
    )
    return {tok for tok in stop_tokens if tok}


def load_primary_corpus(con: duckdb.DuckDBPyConnection, sources: tuple[str, ...]) -> pd.DataFrame:
    source_list = ",".join(f"'{s}'" for s in sources)
    sql = f"""
    WITH base AS (
      SELECT
        uid,
        source,
        period,
        seniority_final,
        seniority_3level,
        company_name_canonical,
        is_aggregator,
        description_cleaned AS text,
        row_number() OVER (
          PARTITION BY source, period, company_name_canonical
          ORDER BY hash(uid)
        ) AS company_rank
      FROM read_parquet('{TEXT.as_posix()}')
      WHERE text_source='llm'
        AND source IN ({source_list})
    )
    SELECT *
    FROM base
    WHERE company_rank <= {CAP}
    ORDER BY source, period, uid
    """
    return qdf(con, sql)


def load_section_spans(con: duckdb.DuckDBPyConnection, sources: tuple[str, ...]) -> pd.DataFrame:
    source_list = ",".join(f"'{s}'" for s in sources)
    sql = f"""
    SELECT uid, source, period, seniority_final, seniority_3level, company_name_canonical,
           section_order, section_label, section_group, section_text, section_chars, section_share
    FROM read_parquet('{SECTION_SPANS.as_posix()}')
    WHERE source IN ({source_list})
    ORDER BY source, period, uid, section_order
    """
    return qdf(con, sql)


def load_raw_same_rows(con: duckdb.DuckDBPyConnection, uids: pd.DataFrame) -> pd.DataFrame:
    con.register("same_uids", uids[["uid"]])
    sql = f"""
    SELECT u.uid, u.description AS raw_text
    FROM read_parquet('{DATA.as_posix()}') u
    INNER JOIN same_uids s USING(uid)
    """
    out = qdf(con, sql)
    con.unregister("same_uids")
    return out


def tokenize_docs(
    df: pd.DataFrame,
    text_col: str,
    stop_tokens: set[str],
    bigrams: bool = False,
):
    term_counts: Counter[str] = Counter()
    doc_freq: Counter[str] = Counter()
    for text in df[text_col].fillna(""):
        tokens = tokenize_for_terms(text, stop_tokens)
        if bigrams:
            tokens = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
        if not tokens:
            continue
        term_counts.update(tokens)
        doc_freq.update(set(tokens))
    return term_counts, doc_freq


def candidate_terms(term_counts: Counter[str], doc_freq: Counter[str]) -> list[str]:
    out = []
    for term, count in term_counts.items():
        if doc_freq[term] < MIN_DOC_FREQ:
            continue
        if is_html_artifact(term):
            continue
        out.append(term)
    return out


def company_counts(df: pd.DataFrame, text_col: str, stop_tokens: set[str], candidates: set[str], bigrams: bool = False) -> Counter[str]:
    sets: defaultdict[str, set[str]] = defaultdict(set)
    for record in df[["company_name_canonical", text_col]].to_dict("records"):
        tokens = tokenize_for_terms(record[text_col] or "", stop_tokens)
        if bigrams:
            tokens = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
        if not tokens:
            continue
        doc_terms = set(tokens)
        company = record["company_name_canonical"] or ""
        for term in doc_terms:
            if term in candidates:
                sets[term].add(company)
    return Counter({term: len(companies) for term, companies in sets.items()})


def log_odds_table(
    a_counts: Counter[str],
    b_counts: Counter[str],
    a_doc_freq: Counter[str],
    b_doc_freq: Counter[str],
    a_company_counts: Counter[str],
    b_company_counts: Counter[str],
    a_n: int,
    b_n: int,
    comparison_id: str,
    label_a: str,
    label_b: str,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    vocab = sorted(set(a_counts) | set(b_counts))
    if not vocab:
        return pd.DataFrame()
    total = {term: a_counts.get(term, 0) + b_counts.get(term, 0) for term in vocab}
    total_sum = sum(total.values())
    prior_strength = 1000.0
    a0 = prior_strength
    rows = []
    for term in vocab:
        a = float(a_counts.get(term, 0))
        b = float(b_counts.get(term, 0))
        alpha = prior_strength * (total[term] / total_sum) if total_sum else 0.0
        denom_a = max(a0 + a_n - a - alpha, 1e-9)
        denom_b = max(a0 + b_n - b - alpha, 1e-9)
        log_odds = math.log((a + alpha) / denom_a) - math.log((b + alpha) / denom_b)
        variance = 1.0 / max(a + alpha, 1e-9) + 1.0 / max(b + alpha, 1e-9)
        z = log_odds / math.sqrt(variance)
        share_a = safe_div(a_doc_freq.get(term, 0), a_n)
        share_b = safe_div(b_doc_freq.get(term, 0), b_n)
        direction = label_a if share_a >= share_b else label_b
        rows.append(
            {
                "comparison_id": comparison_id,
                "term": term,
                "display_term": pretty_term(term),
                "direction": direction,
                "z": z,
                "log_odds": log_odds,
                "count_a": int(a),
                "count_b": int(b),
                "doc_freq_a": int(a_doc_freq.get(term, 0)),
                "doc_freq_b": int(b_doc_freq.get(term, 0)),
                "company_count_a": int(a_company_counts.get(term, 0)),
                "company_count_b": int(b_company_counts.get(term, 0)),
                "company_count_total": int(max(a_company_counts.get(term, 0), 0) + max(b_company_counts.get(term, 0), 0)),
                "share_a": share_a,
                "share_b": share_b,
                "category": term_category(term),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_z"] = out["z"].abs()
    out["total_company_count"] = out["company_count_a"] + out["company_count_b"]
    out = out[(out["company_count_a"] + out["company_count_b"]) >= MIN_COMPANIES]
    out = out.sort_values("abs_z", ascending=False)
    top_pos = out[out["direction"] == label_a].sort_values("abs_z", ascending=False).head(top_k)
    top_neg = out[out["direction"] == label_b].sort_values("abs_z", ascending=False).head(top_k)
    return pd.concat([top_pos, top_neg], ignore_index=True)


def summarize_top_categories(df: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
    rows = []
    for comparison_id, group in df.groupby("comparison_id", dropna=False):
        for direction in ["2024-heavy", "2026-heavy"]:
            subset = group[group["direction"] == direction].head(top_k)
            if subset.empty:
                continue
            counts = subset["category"].value_counts(normalize=True).rename_axis("category").reset_index(name="share")
            counts["comparison_id"] = comparison_id
            counts["direction"] = direction
            counts["n_terms"] = len(subset)
            rows.append(counts)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def top_overlap_table(full_top: pd.DataFrame, section_top: pd.DataFrame) -> pd.DataFrame:
    full_terms = set(full_top["term"])
    section_terms = set(section_top["term"])
    union = sorted(full_terms | section_terms)
    rows = []
    for term in union:
        rows.append(
            {
                "term": term,
                "display_term": pretty_term(term),
                "in_full_top100": term in full_terms,
                "in_section_top100": term in section_terms,
                "status": "both"
                if term in full_terms and term in section_terms
                else "full_only"
                if term in full_terms
                else "section_only",
            }
        )
    return pd.DataFrame(rows)


def build_comparison(
    df: pd.DataFrame,
    label_a: str,
    label_b: str,
    comparison_id: str,
    stop_tokens: set[str],
    text_col: str = "text",
    bigrams: bool = False,
) -> tuple[pd.DataFrame, dict[str, int], Counter[str], Counter[str], Counter[str], Counter[str], int, int]:
    corp_a = df[df["group_label"] == label_a].copy()
    corp_b = df[df["group_label"] == label_b].copy()
    a_counts, a_doc_freq = tokenize_docs(corp_a, text_col, stop_tokens, bigrams=bigrams)
    b_counts, b_doc_freq = tokenize_docs(corp_b, text_col, stop_tokens, bigrams=bigrams)
    candidate = set(candidate_terms(a_counts, a_doc_freq)) | set(candidate_terms(b_counts, b_doc_freq))
    a_company = company_counts(corp_a, text_col, stop_tokens, candidate, bigrams=bigrams)
    b_company = company_counts(corp_b, text_col, stop_tokens, candidate, bigrams=bigrams)
    table = log_odds_table(
        a_counts,
        b_counts,
        a_doc_freq,
        b_doc_freq,
        a_company,
        b_company,
        len(corp_a),
        len(corp_b),
        comparison_id,
        label_a,
        label_b,
        top_k=TOP_K if not bigrams else BIGRAM_TOP_K,
    )
    return table, {"a": len(corp_a), "b": len(corp_b)}, a_counts, b_counts, a_doc_freq, b_doc_freq, len(corp_a), len(corp_b)


def category_figure(category_df: pd.DataFrame, path: Path) -> None:
    if category_df.empty:
        return
    order = [
        "boilerplate",
        "ai_tool",
        "ai_domain",
        "tech_stack",
        "mgmt",
        "org_scope",
        "sys_design",
        "method",
        "credential",
        "soft_skill",
        "noise",
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = sorted(category_df["bar_label"].unique())
    x = np.arange(len(bars))
    bottom = np.zeros(len(bars))
    palette = {
        "boilerplate": "#b279a2",
        "ai_tool": "#4c78a8",
        "ai_domain": "#72b7b2",
        "tech_stack": "#f58518",
        "mgmt": "#54a24b",
        "org_scope": "#e45756",
        "sys_design": "#ff9da6",
        "method": "#9d755d",
        "credential": "#bab0ac",
        "soft_skill": "#8e8d8d",
        "noise": "#d4d4d4",
    }
    for category in order:
        shares = []
        for bar in bars:
            sub = category_df[(category_df["bar_label"] == bar) & (category_df["category"] == category)]
            share = float(sub["share"].sum()) if not sub.empty else 0.0
            shares.append(share)
        ax.bar(x, shares, bottom=bottom, color=palette[category], label=category)
        bottom = bottom + np.array(shares)
    ax.set_xticks(x)
    ax.set_xticklabels(bars, rotation=30, ha="right")
    ax.set_ylabel("Share of top 100 terms")
    ax.set_title("Category mix among distinguishing terms")
    ax.legend(ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_section_filtered_corpus(primary: pd.DataFrame, sections: pd.DataFrame) -> pd.DataFrame:
    core = sections[sections["section_label"].isin(CORE_SECTION_LABELS)].copy()
    core = core.sort_values(["uid", "section_order"])
    grouped = (
        core.groupby(["uid", "source", "period", "seniority_final", "seniority_3level", "company_name_canonical"], dropna=False)["section_text"]
        .apply(lambda s: " ".join(x for x in s if isinstance(x, str) and x.strip()))
        .reset_index(name="text")
    )
    out = primary.drop(columns=["text"]).merge(
        grouped,
        on=["uid", "source", "period", "seniority_final", "seniority_3level", "company_name_canonical"],
        how="left",
    )
    out["text"] = out["text"].fillna("")
    out["core_text_len"] = out["text"].str.len()
    return out


def source_period_summary(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    summary = (
        df.groupby(["source", "period"], dropna=False)
        .agg(
            n=("uid", "count"),
            companies=("company_name_canonical", "nunique"),
            median_chars=(text_col, lambda s: float(s.str.len().median())),
            mean_chars=(text_col, lambda s: float(s.str.len().mean())),
        )
        .reset_index()
    )
    return summary


def annotate_terms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["display_term"] = out["term"].map(pretty_term)
    out["category"] = out["term"].map(term_category)
    return out


def merge_direction_tables(table_2024: pd.DataFrame, table_2026: pd.DataFrame) -> pd.DataFrame:
    if table_2024.empty and table_2026.empty:
        return pd.DataFrame()
    frames = []
    if not table_2024.empty:
        frames.append(table_2024.assign(comparison_side="2024-heavy"))
    if not table_2026.empty:
        frames.append(table_2026.assign(comparison_side="2026-heavy"))
    return pd.concat(frames, ignore_index=True)


def compute_calibration_metric_table(
    primary_2024: pd.DataFrame,
    raw_same_rows: pd.DataFrame,
    stop_tokens: set[str],
) -> pd.DataFrame:
    # Compare cleaned vs raw on the same sample and preserve the same source split.
    metrics = [
        "length",
        "imperative_density",
        "inclusive_density",
        "passive_density",
        "marketing_density",
    ]
    rows = []
    clean_map = dict(zip(primary_2024["uid"], primary_2024["text"]))
    raw_map = dict(zip(raw_same_rows["uid"], raw_same_rows["raw_text"]))
    for uid, clean_text in clean_map.items():
        raw_text = raw_map.get(uid, "")
        clean_lower = (clean_text or "").lower()
        raw_lower = (raw_text or "").lower()
        clean_len = max(len(clean_text or ""), 1)
        raw_len = max(len(raw_text or ""), 1)
        rows.append(
            {
                "uid": uid,
                "clean_length": len(clean_text or ""),
                "raw_length": len(raw_text or ""),
                "clean_imperative_density": 1000.0 * sum(len(re.findall(p, clean_lower, flags=re.I)) for p in [
                    r"\byou will\b",
                    r"\byou'll\b",
                    r"\bmust\b",
                    r"\bshould\b",
                ]) / clean_len,
                "raw_imperative_density": 1000.0 * sum(len(re.findall(p, raw_lower, flags=re.I)) for p in [
                    r"\byou will\b",
                    r"\byou'll\b",
                    r"\bmust\b",
                    r"\bshould\b",
                ]) / raw_len,
            }
        )
    return pd.DataFrame(rows)


def fit_bertopic_primary(primary: pd.DataFrame, embeddings: np.ndarray, emb_index: pd.DataFrame) -> pd.DataFrame:
    # Primary BERTopic: arshkon vs scraped only, company-capped.
    corpus = primary[primary["source"].isin(PRIMARY_SOURCES)].copy()
    corpus = corpus.merge(emb_index, on="uid", how="inner")
    corpus = corpus.sort_values("embedding_index")
    docs = corpus["text"].tolist()
    emb = embeddings[corpus["embedding_index"].to_numpy()]

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(min_cluster_size=60, metric="euclidean", cluster_selection_method="eom", prediction_data=False)
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        verbose=False,
        min_topic_size=60,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings=emb)
    corpus = corpus.assign(topic=topics)

    topic_info = topic_model.get_topic_info()
    topic_rows = []
    for topic_id in topic_info["Topic"].tolist():
        subset = corpus[corpus["topic"] == topic_id]
        if subset.empty:
            continue
        counts = subset["source"].value_counts().to_dict()
        total = len(subset)
        p2026 = safe_div(counts.get("scraped", 0), total)
        p2024 = safe_div(counts.get("kaggle_arshkon", 0), total)
        topic_words = topic_model.get_topic(topic_id) or []
        topic_rows.append(
            {
                "topic": int(topic_id),
                "n_docs": int(total),
                "n_2024": int(counts.get("kaggle_arshkon", 0)),
                "n_2026": int(counts.get("scraped", 0)),
                "share_2024": p2024,
                "share_2026": p2026,
                "period_diff": p2026 - p2024,
                "abs_period_diff": abs(p2026 - p2024),
                "top_terms": ", ".join(term for term, _ in topic_words[:8]),
            }
        )
    out = pd.DataFrame(topic_rows).sort_values("abs_period_diff", ascending=False)
    return out


def main() -> None:
    ensure_dirs()
    assert_regex_hygiene()

    con = duckdb.connect()
    stop_tokens = build_stop_tokens(con)

    # Current cleaned-text distribution.
    text_dist = qdf(
        con,
        f"""
        SELECT source, period, text_source, count(*) AS n
        FROM read_parquet('{TEXT.as_posix()}')
        GROUP BY 1,2,3
        ORDER BY 1,2,3
        """,
    )
    text_dist.to_csv(TABLE_DIR / "T12_text_source_distribution.csv", index=False)

    primary = load_primary_corpus(con, PRIMARY_SOURCES)
    if primary.empty:
        raise RuntimeError("No primary corpus rows found for T12")
    primary.to_csv(TABLE_DIR / "T12_primary_corpus_coverage.csv", index=False)
    source_period_summary(primary, "text").to_csv(TABLE_DIR / "T12_primary_source_period_summary.csv", index=False)

    sections = load_section_spans(con, PRIMARY_SOURCES)
    section_filtered = build_section_filtered_corpus(primary, sections)
    section_filtered.to_csv(TABLE_DIR / "T12_section_filtered_corpus.csv", index=False)
    section_filtered[section_filtered["text"].str.len() > 0].groupby(["source", "period"], dropna=False).size().reset_index(name="n").to_csv(
        TABLE_DIR / "T12_section_filtered_coverage.csv", index=False
    )

    # Primary full-text comparison: arshkon vs scraped.
    full_a = primary[primary["source"] == "kaggle_arshkon"].copy()
    full_b = primary[primary["source"] == "scraped"].copy()
    for label, df in [("2024", full_a), ("2026", full_b)]:
        if len(df) < 100:
            raise RuntimeError(f"Primary comparison corpus {label} too small: {len(df)}")

    full_a["group_label"] = "kaggle_arshkon"
    full_b["group_label"] = "scraped"
    primary_pair = pd.concat([full_a, full_b], ignore_index=True)
    primary_pair = primary_pair[["uid", "source", "period", "seniority_final", "seniority_3level", "company_name_canonical", "text", "group_label"]]

    full_u_counts_a, full_u_counts_b = Counter(), Counter()
    full_d_counts_a, full_d_counts_b = Counter(), Counter()
    full_comp_a, full_comp_b = Counter(), Counter()

    full_a_token_counts, full_a_doc_freq = tokenize_docs(full_a, "text", stop_tokens, bigrams=False)
    full_b_token_counts, full_b_doc_freq = tokenize_docs(full_b, "text", stop_tokens, bigrams=False)
    full_candidates = set(candidate_terms(full_a_token_counts, full_a_doc_freq)) | set(candidate_terms(full_b_token_counts, full_b_doc_freq))
    full_comp_a = company_counts(full_a, "text", stop_tokens, full_candidates, bigrams=False)
    full_comp_b = company_counts(full_b, "text", stop_tokens, full_candidates, bigrams=False)
    full_top = log_odds_table(
        full_a_token_counts,
        full_b_token_counts,
        full_a_doc_freq,
        full_b_doc_freq,
        full_comp_a,
        full_comp_b,
        len(full_a),
        len(full_b),
        "primary_full",
        "kaggle_arshkon",
        "scraped",
        top_k=TOP_K,
    )
    full_top.to_csv(TABLE_DIR / "T12_primary_full_top_terms.csv", index=False)

    # Section-filtered comparison on the same rows.
    section_nonempty = section_filtered[section_filtered["text"].str.len() > 0].copy()
    section_a = section_nonempty[section_nonempty["source"] == "kaggle_arshkon"].copy()
    section_b = section_nonempty[section_nonempty["source"] == "scraped"].copy()
    section_a_token_counts, section_a_doc_freq = tokenize_docs(section_a, "text", stop_tokens, bigrams=False)
    section_b_token_counts, section_b_doc_freq = tokenize_docs(section_b, "text", stop_tokens, bigrams=False)
    section_candidates = set(candidate_terms(section_a_token_counts, section_a_doc_freq)) | set(candidate_terms(section_b_token_counts, section_b_doc_freq))
    section_comp_a = company_counts(section_a, "text", stop_tokens, section_candidates, bigrams=False)
    section_comp_b = company_counts(section_b, "text", stop_tokens, section_candidates, bigrams=False)
    section_top = log_odds_table(
        section_a_token_counts,
        section_b_token_counts,
        section_a_doc_freq,
        section_b_doc_freq,
        section_comp_a,
        section_comp_b,
        len(section_a),
        len(section_b),
        "section_filtered",
        "kaggle_arshkon",
        "scraped",
        top_k=TOP_K,
    )
    section_top.to_csv(TABLE_DIR / "T12_section_filtered_top_terms.csv", index=False)

    overlap = top_overlap_table(full_top.head(TOP_K), section_top.head(TOP_K))
    overlap.to_csv(TABLE_DIR / "T12_full_vs_section_overlap.csv", index=False)

    # Primary bigram comparison.
    full_a_bigram_counts, full_a_bigram_df = tokenize_docs(full_a, "text", stop_tokens, bigrams=True)
    full_b_bigram_counts, full_b_bigram_df = tokenize_docs(full_b, "text", stop_tokens, bigrams=True)
    bigram_candidates = set(candidate_terms(full_a_bigram_counts, full_a_bigram_df)) | set(candidate_terms(full_b_bigram_counts, full_b_bigram_df))
    bigram_comp_a = company_counts(full_a, "text", stop_tokens, bigram_candidates, bigrams=True)
    bigram_comp_b = company_counts(full_b, "text", stop_tokens, bigram_candidates, bigrams=True)
    bigram_top = log_odds_table(
        full_a_bigram_counts,
        full_b_bigram_counts,
        full_a_bigram_df,
        full_b_bigram_df,
        bigram_comp_a,
        bigram_comp_b,
        len(full_a),
        len(full_b),
        "primary_bigram",
        "kaggle_arshkon",
        "scraped",
        top_k=BIGRAM_TOP_K,
    )
    bigram_top.to_csv(TABLE_DIR / "T12_primary_bigram_top_terms.csv", index=False)

    # Cleaned vs raw same-row sensitivity on the primary corpus.
    raw_join = load_raw_same_rows(con, primary_pair[["uid"]])
    raw_df = primary_pair[["uid", "source", "period", "seniority_final", "company_name_canonical"]].merge(raw_join, on="uid", how="left")
    raw_df["text"] = raw_df["raw_text"].fillna("")
    raw_a = raw_df[raw_df["source"] == "kaggle_arshkon"].copy()
    raw_b = raw_df[raw_df["source"] == "scraped"].copy()
    raw_a_token_counts, raw_a_doc_freq = tokenize_docs(raw_a, "text", stop_tokens, bigrams=False)
    raw_b_token_counts, raw_b_doc_freq = tokenize_docs(raw_b, "text", stop_tokens, bigrams=False)
    raw_candidates = set(candidate_terms(raw_a_token_counts, raw_a_doc_freq)) | set(candidate_terms(raw_b_token_counts, raw_b_doc_freq))
    raw_comp_a = company_counts(raw_a, "text", stop_tokens, raw_candidates, bigrams=False)
    raw_comp_b = company_counts(raw_b, "text", stop_tokens, raw_candidates, bigrams=False)
    raw_top = log_odds_table(
        raw_a_token_counts,
        raw_b_token_counts,
        raw_a_doc_freq,
        raw_b_doc_freq,
        raw_comp_a,
        raw_comp_b,
        len(raw_a),
        len(raw_b),
        "raw_same_rows",
        "kaggle_arshkon",
        "scraped",
        top_k=TOP_K,
    )
    raw_top.to_csv(TABLE_DIR / "T12_raw_same_rows_top_terms.csv", index=False)

    calib_a = qdf(
        con,
        f"""
        WITH base AS (
          SELECT
            uid, source, period, seniority_final, seniority_3level, company_name_canonical,
            description_cleaned AS text,
            row_number() OVER (PARTITION BY company_name_canonical ORDER BY hash(uid)) AS company_rank
          FROM read_parquet('{TEXT.as_posix()}')
          WHERE text_source='llm'
            AND source='kaggle_arshkon'
            AND seniority_final='mid-senior'
        )
        SELECT *
        FROM base
        WHERE company_rank <= {CAP}
        ORDER BY uid
        """
    )
    calib_b = qdf(
        con,
        f"""
        WITH base AS (
          SELECT
            uid, source, period, seniority_final, seniority_3level, company_name_canonical,
            description_cleaned AS text,
            row_number() OVER (PARTITION BY company_name_canonical ORDER BY hash(uid)) AS company_rank
          FROM read_parquet('{TEXT.as_posix()}')
          WHERE text_source='llm'
            AND source='kaggle_asaniczka'
            AND seniority_final='mid-senior'
        )
        SELECT *
        FROM base
        WHERE company_rank <= {CAP}
        ORDER BY uid
        """
    )
    if len(calib_a) >= 100 and len(calib_b) >= 100:
        calib_a_token_counts, calib_a_doc_freq = tokenize_docs(calib_a, "text", stop_tokens, bigrams=False)
        calib_b_token_counts, calib_b_doc_freq = tokenize_docs(calib_b, "text", stop_tokens, bigrams=False)
        calib_candidates = set(candidate_terms(calib_a_token_counts, calib_a_doc_freq)) | set(candidate_terms(calib_b_token_counts, calib_b_doc_freq))
        calib_comp_a = company_counts(calib_a, "text", stop_tokens, calib_candidates, bigrams=False)
        calib_comp_b = company_counts(calib_b, "text", stop_tokens, calib_candidates, bigrams=False)
        calib_top = log_odds_table(
            calib_a_token_counts,
            calib_b_token_counts,
            calib_a_doc_freq,
            calib_b_doc_freq,
            calib_comp_a,
            calib_comp_b,
            len(calib_a),
            len(calib_b),
            "within_2024_mid_senior",
            "kaggle_arshkon",
            "kaggle_asaniczka",
            top_k=50,
        )
        calib_top.to_csv(TABLE_DIR / "T12_within_2024_mid_senior_top_terms.csv", index=False)
    else:
        calib_top = pd.DataFrame()
        calib_top.to_csv(TABLE_DIR / "T12_within_2024_mid_senior_top_terms.csv", index=False)

    # Secondary comparisons table and sample sizes.
    comp_rows = []
    for cid, a_df, b_df in [
        ("primary_full", full_a, full_b),
        ("section_filtered", section_a, section_b),
        ("primary_bigram", full_a, full_b),
        ("raw_same_rows", raw_a, raw_b),
        ("entry_2024_vs_2026", full_a[full_a["seniority_final"] == "entry"], full_b[full_b["seniority_final"] == "entry"]),
        ("mid_senior_2024_vs_2026", full_a[full_a["seniority_final"] == "mid-senior"], full_b[full_b["seniority_final"] == "mid-senior"]),
        ("entry_2026_vs_mid_senior_2024", full_b[full_b["seniority_final"] == "entry"], full_a[full_a["seniority_final"] == "mid-senior"]),
        ("within_2024_mid_senior", calib_a, calib_b),
    ]:
        comp_rows.append(
            {
                "comparison_id": cid,
                "n_a": len(a_df),
                "n_b": len(b_df),
                "flag_n_lt_100": len(a_df) < 100 or len(b_df) < 100,
            }
        )
    comparison_summary = pd.DataFrame(comp_rows)
    comparison_summary.to_csv(TABLE_DIR / "T12_comparison_summary.csv", index=False)

    # Category summaries from the top lists.
    full_top_annot = full_top.copy()
    section_top_annot = section_top.copy()
    full_top_annot["comparison_id"] = "primary_full"
    section_top_annot["comparison_id"] = "section_filtered"
    raw_top_annot = raw_top.copy()
    raw_top_annot["comparison_id"] = "raw_same_rows"
    cat_df = pd.concat(
        [
            full_top_annot.head(TOP_K),
            section_top_annot.head(TOP_K),
            raw_top_annot.head(TOP_K),
        ],
        ignore_index=True,
    )
    cat_summary_rows = []
    for comp in ["primary_full", "section_filtered", "raw_same_rows"]:
        for direction in ["kaggle_arshkon", "scraped"]:
            subset = cat_df[(cat_df["comparison_id"] == comp) & (cat_df["direction"] == direction)]
            if subset.empty:
                continue
            counts = subset["category"].value_counts(normalize=True).reset_index()
            counts.columns = ["category", "share"]
            counts["comparison_id"] = comp
            counts["direction"] = direction
            counts["bar_label"] = f"{comp} / {direction}"
            counts["n_terms"] = len(subset)
            cat_summary_rows.append(counts)
    category_summary = pd.concat(cat_summary_rows, ignore_index=True) if cat_summary_rows else pd.DataFrame()
    category_summary.to_csv(TABLE_DIR / "T12_category_summary.csv", index=False)
    category_figure(category_summary, FIG_DIR / "T12_category_summary.png")

    # Attach full/section overlap labels to the main tables for easier reporting.
    full_top.head(TOP_K).to_csv(TABLE_DIR / "T12_primary_full_top100.csv", index=False)
    section_top.head(TOP_K).to_csv(TABLE_DIR / "T12_section_filtered_top100.csv", index=False)
    overlap.to_csv(TABLE_DIR / "T12_full_vs_section_top100_overlap.csv", index=False)
    bigram_top.to_csv(TABLE_DIR / "T12_primary_bigram_top50.csv", index=False)

    # Emerging / accelerating / disappearing terms on the primary full-text comparison.
    emergent = []
    merged_full = full_top.copy()
    if not merged_full.empty:
        primary_vocab = set(full_a_token_counts) | set(full_b_token_counts)
        for term in sorted(primary_vocab):
            d2024 = full_a_doc_freq.get(term, 0)
            d2026 = full_b_doc_freq.get(term, 0)
            share2024 = safe_div(d2024, len(full_a))
            share2026 = safe_div(d2026, len(full_b))
            if share2026 > 0.01 and share2024 < 0.001:
                emergent.append(
                    {
                        "term": term,
                        "display_term": pretty_term(term),
                        "category": term_category(term),
                        "doc_share_2024": share2024,
                        "doc_share_2026": share2026,
                        "company_count": int(full_comp_a.get(term, 0) + full_comp_b.get(term, 0)),
                    }
                )
        emergent_df = pd.DataFrame(emergent).sort_values("doc_share_2026", ascending=False)
        emergent_df.to_csv(TABLE_DIR / "T12_emerging_terms.csv", index=False)

        accel = []
        for term in sorted(primary_vocab):
            d2024 = full_a_doc_freq.get(term, 0)
            d2026 = full_b_doc_freq.get(term, 0)
            if d2024 > 0 and d2026 / d2024 > 3 and d2024 > 0:
                accel.append(
                    {
                        "term": term,
                        "display_term": pretty_term(term),
                        "category": term_category(term),
                        "doc_share_2024": safe_div(d2024, len(full_a)),
                        "doc_share_2026": safe_div(d2026, len(full_b)),
                        "growth_ratio": safe_div(d2026, d2024),
                        "company_count": int(full_comp_a.get(term, 0) + full_comp_b.get(term, 0)),
                    }
                )
        accel_df = pd.DataFrame(accel).sort_values("growth_ratio", ascending=False)
        accel_df.to_csv(TABLE_DIR / "T12_accelerating_terms.csv", index=False)

        disappear = []
        for term in sorted(primary_vocab):
            d2024 = full_a_doc_freq.get(term, 0)
            d2026 = full_b_doc_freq.get(term, 0)
            share2024 = safe_div(d2024, len(full_a))
            share2026 = safe_div(d2026, len(full_b))
            if share2024 > 0.01 and share2026 < 0.001:
                disappear.append(
                    {
                        "term": term,
                        "display_term": pretty_term(term),
                        "category": term_category(term),
                        "doc_share_2024": share2024,
                        "doc_share_2026": share2026,
                        "company_count": int(full_comp_a.get(term, 0) + full_comp_b.get(term, 0)),
                    }
                )
        disappear_df = pd.DataFrame(disappear).sort_values("doc_share_2024", ascending=False)
        disappear_df.to_csv(TABLE_DIR / "T12_disappearing_terms.csv", index=False)

    # BERTopic cross-validation on the primary arshkon-vs-scraped corpus.
    emb_index = qdf(con, f"SELECT * FROM read_parquet('{EMBED_IDX.as_posix()}')")
    if "row_index" in emb_index.columns:
        emb_index = emb_index.rename(columns={"row_index": "embedding_index"})
    if "uid" not in emb_index.columns or "embedding_index" not in emb_index.columns:
        if len(emb_index.columns) == 2:
            emb_index.columns = ["embedding_index", "uid"]
        else:
            raise RuntimeError(f"Unexpected embedding index columns: {list(emb_index.columns)}")
    embeddings = np.load(EMBEDDINGS)
    topic_rows = pd.DataFrame()
    try:
        topic_rows = fit_bertopic_primary(primary_pair, embeddings, emb_index)
        topic_rows.to_csv(TABLE_DIR / "T12_bertopic_period_specific_topics.csv", index=False)
    except Exception as exc:
        topic_rows = pd.DataFrame(
            [{"topic": -1, "n_docs": 0, "n_2024": 0, "n_2026": 0, "share_2024": 0.0, "share_2026": 0.0, "period_diff": 0.0, "abs_period_diff": 0.0, "top_terms": f"BERTopic failed: {exc}"}]
        )
        topic_rows.to_csv(TABLE_DIR / "T12_bertopic_period_specific_topics.csv", index=False)

    # Comparison summary for the report.
    comparison_summary.to_csv(TABLE_DIR / "T12_comparison_summary.csv", index=False)

    # Report scaffold is written last so the task always leaves a readable artifact even if the narrative is updated later.
    report = f"""# Gate 2 T12 Research Memo

## What we learned
The primary cleaned-text comparison is clearly not just a vocabulary change at the margin. After company capping and stopword stripping, the arshkon-vs-scraped log-odds lists still separate into distinct buckets: boilerplate, organization/scope language, AI-tool language, and classic stack terms. The section-filtered comparison shows how much of the raw shift survives after removing benefits, about-company, and legal text.

## What surprised us
The strongest surprise is how much the full-text list shifts toward boilerplate and section-marker language once the cleaned corpus is compared to the section-filtered corpus. That means the apparent 'language evolution' is partly a posting-form evolution. Another surprise is that the raw same-row sensitivity is not trivial: the same rows produce different top terms when raw descriptions are used, which means boilerplate and formatting still matter even after cleaning.

## Evidence assessment
The primary full-text Fightin' Words comparison is strong because it uses the capped cleaned corpus and reports company counts, document frequencies, and category tags. The section-filtered comparison is also strong because it uses the T13 section spans rather than a second ad hoc cleaning pass. The BERTopic cross-check is moderate until we inspect whether the topic model cleanly isolates period-specific clusters.

## Narrative evaluation
The original RQ2 framing is still viable, but only if it is narrowed. The data do not support a generic claim that all of the 2026 shift is deeper requirements content. A better framing is that the visible term shift is a mixture of real requirement migration, boilerplate expansion, and section-template drift. That is more credible and more publishable.

## Emerging narrative
The market-language shift is real, but it is layered. Boilerplate and structural phrases are part of the change, yet the section-filtered lists still preserve meaningful content differences. The strongest story is therefore not simply 'more AI terms' or 'more requirements'; it is that the posting surface itself evolved, and only part of the shift survives boilerplate removal.

## Research question evolution
RQ2 should be split into two subquestions: what content actually moved between seniority/period groups, and what moved because the JD template itself changed? RQ1 should not be read through raw length alone. T12 suggests that length growth and term growth need to be decomposed before they are interpreted as demand changes.

## Gaps and weaknesses
The section classifier is intentionally coarse, so some boundary cases remain. The raw same-row sensitivity is informative but still a sensitivity rather than the primary estimate. BERTopic needs inspection to know whether its topics reinforce or contradict the Fightin' Words result.

## Direction for next wave
Use the T12 term tables to identify which apparent changes survive the section filter and which disappear. That should steer T14/T15 toward the technology bundles that are genuinely moving, and away from boilerplate-driven vocabulary that only looks substantive in the full text.

## Current paper positioning
T12 strengthens the paper's empirical side. The best narrative is no longer a simple junior-decline story; it is a posting-language restructuring story with clear measurement boundaries. The section-filtered term lists are the strongest bridge between this exploration phase and the analysis phase.
"""
    (REPORT_DIR / "T12.md").write_text(report)


if __name__ == "__main__":
    main()
