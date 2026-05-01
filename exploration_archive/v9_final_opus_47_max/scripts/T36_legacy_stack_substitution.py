"""T36 — Legacy-stack substitution map (H_L).

For each 2024 SWE title that disappeared in 2026 (per T10), find the nearest 2026
descriptive neighbor via TF-IDF cosine. Produces a role substitution map.

Scope caveat (per dispatch): T10 strict disappearing-titles list = n=2;
wider arshkon>=5 list = n=11. Results are exploratory — be candid in the report.

Outputs under exploration/tables/T36/:
- substitution_table.csv                 (disappearing -> top5 2026 with cosine + seniority + class)
- content_drift_per_pair.csv             (fightin-words top terms per pair)
- ai_vocab_comparison.csv                (ai_strict share disappearing vs 2026 neighbor)
- manual_inspection.csv                  (10 postings per title / 10 per neighbor)
- run_meta.json
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNIFIED = ROOT / "data" / "unified.parquet"
CLEANED_TEXT = ROOT / "exploration/artifacts/shared/swe_cleaned_text.parquet"
DISAPPEARING_WIDER = ROOT / "exploration/tables/T10/disappearing_titles_arshkon_ge5.csv"
DISAPPEARING_STRICT = ROOT / "exploration/tables/T10/disappearing_titles.csv"
VALIDATED_PATTERNS = ROOT / "exploration/artifacts/shared/validated_mgmt_patterns.json"
STOPLIST = ROOT / "exploration/artifacts/shared/company_stoplist.txt"

OUT_TABLES = ROOT / "exploration/tables/T36"
OUT_TABLES.mkdir(parents=True, exist_ok=True)

COMPANY_CAP = 50  # Wave 3.5 requirement
N_TITLE_UNIVERSE_MIN = 10  # 2026 titles >= 10
TOP_K_NEIGHBORS = 5
FW_TOP = 20

# Semantic category tags for Fightin' Words
LEGACY_TECH = {
    "cobol", "fortran", "pascal", "perl", "vb", "vba", "visualbasic", "delphi", "mainframe",
    "dotnet", ".net", "vbnet", "aspnet", "asp.net", "mvc", "sql server", "sqlserver", "ssis", "ssrs",
    "php", "laravel", "drupal", "wordpress", "jquery", "coldfusion",
    "java", "ejb", "j2ee", "jsp", "servlet", "struts",
    "ruby", "rails", "scala", "akka",
    "oracle", "oracle db",
}
MODERN_TECH = {
    "kubernetes", "docker", "terraform", "helm", "argocd", "aws", "azure", "gcp",
    "typescript", "react", "nextjs", "nodejs", "fastapi", "graphql",
    "go", "golang", "rust", "kotlin",
    "snowflake", "dbt", "airflow", "spark", "databricks", "kafka", "redis",
    "postgresql", "mongodb", "elasticsearch",
    "prometheus", "grafana", "datadog",
    "ci/cd", "cicd", "github", "gitlab", "jenkins",
    "microservices", "serverless", "event", "streaming",
}
AI_TOOL = {
    "llm", "rag", "copilot", "cursor", "claude", "chatgpt", "gpt", "openai", "anthropic",
    "langchain", "llamaindex", "huggingface", "agent", "agents", "ai", "ml", "ml/ai",
    "embedding", "embeddings", "pinecone", "weaviate", "vector", "finetune", "finetuning",
    "prompt", "mcp",
}
SCOPE = {
    "architect", "design", "designing", "architecture", "system", "systems", "scalability",
    "performance", "roadmap", "strategy", "ownership", "cross-functional", "leadership",
    "mentor", "mentorship", "coach", "lead", "leading", "leads",
    "senior", "staff", "principal",
}
MGMT = {
    "manage", "management", "managing", "director", "hiring", "headcount",
    "stakeholder", "stakeholders", "executive", "organization", "organisational",
}
METHODOLOGY = {
    "agile", "scrum", "kanban", "tdd", "bdd", "ddd", "waterfall", "sprint",
    "retrospective", "standup",
}


def tag_term(term: str) -> str:
    t = term.lower()
    if t in LEGACY_TECH:
        return "legacy_tech"
    if t in MODERN_TECH:
        return "modern_tech"
    if t in AI_TOOL:
        return "ai_tool"
    if t in SCOPE:
        return "scope"
    if t in MGMT:
        return "mgmt"
    if t in METHODOLOGY:
        return "methodology"
    return "other"


# ----- data loading -----

def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    base = con.execute(f"""
        select uid, source, period, company_name_canonical, is_aggregator,
               title_normalized,
               seniority_final, yoe_min_years_llm
        from read_parquet('{UNIFIED}')
        where is_swe=true and source_platform='linkedin'
          and is_english=true and date_flag='ok'
    """).df()
    text = pd.read_parquet(CLEANED_TEXT)[["uid", "description_cleaned", "text_source"]]
    df = base.merge(text, on="uid", how="inner")
    df["period_era"] = np.where(df["period"].str.startswith("2024"), "2024", "2026")
    return df


def cap_per_company(df: pd.DataFrame, cap: int, keep_col: str) -> pd.DataFrame:
    has_co = df["company_name_canonical"].notna()
    capped = (
        df[has_co]
        .groupby([keep_col, "company_name_canonical"], group_keys=False)
        .head(cap)
    )
    return pd.concat([capped, df[~has_co]], ignore_index=True)


# ----- TF-IDF centroid per title -----

def build_vectorizer(all_texts: list[str]) -> TfidfVectorizer:
    with open(STOPLIST) as f:
        company_stop = {w.strip() for w in f if w.strip()}
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop = list(ENGLISH_STOP_WORDS | company_stop)
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words=stop,
        min_df=5,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=40000,
    )
    vec.fit(all_texts)
    return vec


def centroid_per_title(df: pd.DataFrame, title_col: str, text_col: str, vec: TfidfVectorizer) -> tuple[pd.DataFrame, dict[str, "np.ndarray"]]:
    rows = []
    centroids = {}
    for title, sub in df.groupby(title_col):
        texts = sub[text_col].dropna().tolist()
        if not texts:
            continue
        mat = vec.transform(texts)
        c = np.asarray(mat.mean(axis=0)).ravel()
        centroids[title] = c
        rows.append({
            "title_normalized": title,
            "n_postings": len(texts),
            "mean_seniority_yoe": float(sub["yoe_min_years_llm"].mean()) if sub["yoe_min_years_llm"].notna().any() else math.nan,
            "mean_seniority_ordinal": float(sub["seniority_final"].map({
                "entry": 1, "associate": 2, "mid-senior": 3, "director": 4, "executive": 5,
            }).mean()),
        })
    return pd.DataFrame(rows), centroids


def cosine_topk(source_vec: "np.ndarray", candidate_matrix: "np.ndarray", candidate_labels: list[str], k: int) -> list[tuple[str, float]]:
    if source_vec is None or candidate_matrix.shape[0] == 0:
        return []
    sims = cosine_similarity(source_vec.reshape(1, -1), candidate_matrix).ravel()
    idx = np.argsort(-sims)[:k]
    return [(candidate_labels[i], float(sims[i])) for i in idx]


# ----- Fightin' Words (log-odds with informative Dirichlet prior) -----

def fighting_words(corpus_a: list[str], corpus_b: list[str], vec: TfidfVectorizer, top_n: int = 20, alpha: float = 0.01) -> pd.DataFrame:
    """Use Monroe/Colaresi/Quinn log-odds-ratio with informative Dirichlet prior.
    alpha sets prior strength; we use a background of corpus_a+corpus_b.
    Returns: DataFrame(word, z, favors, tag).
    """
    # We reuse the vectorizer's vocabulary.
    V = vec.transform(corpus_a + corpus_b)
    vocab = np.array(vec.get_feature_names_out())
    # But TF-IDF transforms don't give us counts; we need term counts.
    # Use CountVectorizer with same vocab.
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(vocabulary=vec.vocabulary_, lowercase=True, ngram_range=vec.ngram_range)
    A = cv.transform(corpus_a)
    B = cv.transform(corpus_b)
    a = np.asarray(A.sum(axis=0)).ravel().astype(np.float64)
    b = np.asarray(B.sum(axis=0)).ravel().astype(np.float64)
    n_a = a.sum()
    n_b = b.sum()
    if n_a == 0 or n_b == 0:
        return pd.DataFrame(columns=["word", "z", "favors", "tag"])
    # informative prior from combined counts
    alpha_vec = alpha * (a + b) + 1e-6
    alpha0 = alpha_vec.sum()
    # log-odds
    num_a = a + alpha_vec
    num_b = b + alpha_vec
    den_a = n_a + alpha0 - num_a
    den_b = n_b + alpha0 - num_b
    log_odds_a = np.log(num_a / den_a)
    log_odds_b = np.log(num_b / den_b)
    delta = log_odds_a - log_odds_b
    var = 1 / num_a + 1 / num_b
    z = delta / np.sqrt(var)
    order = np.argsort(-np.abs(z))
    rows = []
    taken_a = 0
    taken_b = 0
    for i in order:
        if z[i] > 0 and taken_a < top_n:
            rows.append({"word": vocab[i], "z": float(z[i]), "favors": "disappearing", "tag": tag_term(vocab[i])})
            taken_a += 1
        elif z[i] < 0 and taken_b < top_n:
            rows.append({"word": vocab[i], "z": float(z[i]), "favors": "neighbor_2026", "tag": tag_term(vocab[i])})
            taken_b += 1
        if taken_a >= top_n and taken_b >= top_n:
            break
    return pd.DataFrame(rows)


# ----- AI strict binary rate -----

def load_ai_pattern() -> re.Pattern:
    with open(VALIDATED_PATTERNS) as f:
        patterns = json.load(f)
    pat = patterns["ai_strict"]["pattern"]
    return re.compile(pat, flags=re.IGNORECASE)


def ai_strict_rate(texts: list[str], pat: re.Pattern) -> float:
    if not texts:
        return math.nan
    hits = sum(1 for t in texts if pat.search(t or ""))
    return hits / len(texts)


# ----- main -----

def main() -> None:
    t_start = time.time()
    meta: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}

    print("[step 1] loading unified + cleaned text ...")
    df = load_frames()
    print(f"  rows with text: {len(df):,}")
    print(f"  2024 rows: {(df['period_era']=='2024').sum():,}")
    print(f"  2026 rows: {(df['period_era']=='2026').sum():,}")

    # Only use llm-coverage text to reduce boilerplate noise per shared README.
    df = df[df["text_source"] == "llm"].copy()
    print(f"  rows after text_source='llm': {len(df):,}")

    print("[step 2] loading disappearing-titles lists ...")
    wider = pd.read_csv(DISAPPEARING_WIDER)
    strict = pd.read_csv(DISAPPEARING_STRICT)
    print(f"  strict n={len(strict)}  wider n={len(wider)}")

    disappearing_titles = wider["title_normalized"].tolist()
    meta["n_disappearing_strict"] = len(strict)
    meta["n_disappearing_wider"] = len(wider)

    # Step 3 — 2026 title universe (>=10 postings)
    print("[step 3] 2026 title universe ...")
    df_2026 = df[df["period_era"] == "2026"].copy()
    title_counts_2026 = df_2026["title_normalized"].value_counts()
    universe_titles = title_counts_2026[title_counts_2026 >= N_TITLE_UNIVERSE_MIN].index.tolist()
    print(f"  universe size: {len(universe_titles)}")

    # Step 3b — company-cap within 2026 universe for centroid computation
    df_2026_uni = df_2026[df_2026["title_normalized"].isin(universe_titles)].copy()
    df_2026_capped = cap_per_company(df_2026_uni, COMPANY_CAP, "title_normalized")
    print(f"  2026 universe rows pre/post cap{COMPANY_CAP}: {len(df_2026_uni):,} -> {len(df_2026_capped):,}")

    # Disappearing-title corpora on 2024 (arshkon primary per T10 spec — but pool with asaniczka if available)
    df_2024 = df[df["period_era"] == "2024"].copy()
    df_disappearing = df_2024[df_2024["title_normalized"].isin(disappearing_titles)].copy()
    df_disappearing = cap_per_company(df_disappearing, COMPANY_CAP, "title_normalized")
    print(f"  disappearing corpora total rows after cap: {len(df_disappearing):,}")

    # Step 4 — TF-IDF vectorizer trained on combined texts
    print("[step 4] training TF-IDF vectorizer ...")
    corpus_all = pd.concat([df_disappearing["description_cleaned"], df_2026_capped["description_cleaned"]]).dropna()
    vec = build_vectorizer(corpus_all.tolist())
    print(f"  vocab size: {len(vec.vocabulary_):,}")

    # Step 5 — centroids
    print("[step 5] computing centroids ...")
    disappearing_stats, disappearing_centroids = centroid_per_title(df_disappearing, "title_normalized", "description_cleaned", vec)
    neighbor_stats, neighbor_centroids = centroid_per_title(df_2026_capped, "title_normalized", "description_cleaned", vec)
    print(f"  disappearing titles with text: {len(disappearing_stats)}")
    print(f"  2026 neighbor titles: {len(neighbor_stats)}")

    neighbor_mat = np.vstack([neighbor_centroids[t] for t in neighbor_stats["title_normalized"]])
    neighbor_labels = neighbor_stats["title_normalized"].tolist()

    # Step 6 — top-K neighbors + classify
    print("[step 6] top-K neighbor search ...")
    ai_pat = load_ai_pattern()
    sub_rows = []
    content_drift_rows = []
    ai_vocab_rows = []
    manual_rows = []

    for _, r in disappearing_stats.iterrows():
        title = r["title_normalized"]
        arshkon_n = wider[wider["title_normalized"] == title]["arshkon_n"].iloc[0] if title in wider["title_normalized"].values else math.nan
        asaniczka_n = wider[wider["title_normalized"] == title]["asaniczka_n"].iloc[0] if title in wider["title_normalized"].values else math.nan
        src_vec = disappearing_centroids[title]
        topk = cosine_topk(src_vec, neighbor_mat, neighbor_labels, TOP_K_NEIGHBORS)

        source_yoe = r["mean_seniority_yoe"]
        source_sen = r["mean_seniority_ordinal"]
        source_texts = df_disappearing[df_disappearing["title_normalized"] == title]["description_cleaned"].tolist()
        source_ai_rate = ai_strict_rate(source_texts, ai_pat)

        for rank, (neigh, cos) in enumerate(topk, 1):
            nr = neighbor_stats[neighbor_stats["title_normalized"] == neigh].iloc[0]
            neigh_yoe = nr["mean_seniority_yoe"]
            neigh_sen = nr["mean_seniority_ordinal"]
            neigh_n = int(nr["n_postings"])

            # Classify seniority shift
            if math.isnan(source_sen) or math.isnan(neigh_sen):
                cls_shift = "unknown"
            else:
                dy = neigh_sen - source_sen
                if abs(dy) < 0.3:
                    cls_shift = "same_level"
                elif dy > 0:
                    cls_shift = "upward"
                else:
                    cls_shift = "downward"

            sub_rows.append({
                "disappearing_title": title,
                "arshkon_n": arshkon_n,
                "asaniczka_n": asaniczka_n,
                "n_2024_text": len(source_texts),
                "rank": rank,
                "top_2026_neighbor": neigh,
                "cosine": round(cos, 4),
                "neighbor_n_postings": neigh_n,
                "source_yoe_mean": round(float(source_yoe), 2) if not math.isnan(source_yoe) else math.nan,
                "neighbor_yoe_mean": round(float(neigh_yoe), 2) if not math.isnan(neigh_yoe) else math.nan,
                "source_seniority_ord": round(float(source_sen), 2) if not math.isnan(source_sen) else math.nan,
                "neighbor_seniority_ord": round(float(neigh_sen), 2) if not math.isnan(neigh_sen) else math.nan,
                "seniority_shift": cls_shift,
                "source_ai_strict_rate": round(float(source_ai_rate), 4) if not math.isnan(source_ai_rate) else math.nan,
            })

        # Step 7 — content drift per pair (top-1 only)
        if topk:
            top1, top1_cos = topk[0]
            neigh_texts = df_2026_capped[df_2026_capped["title_normalized"] == top1]["description_cleaned"].tolist()
            if len(source_texts) >= 3 and len(neigh_texts) >= 3:
                fw = fighting_words(source_texts, neigh_texts, vec, top_n=FW_TOP)
                for _, fr in fw.iterrows():
                    content_drift_rows.append({
                        "disappearing_title": title,
                        "top1_2026_neighbor": top1,
                        "cosine": round(top1_cos, 4),
                        "word": fr["word"],
                        "z": round(fr["z"], 3),
                        "favors": fr["favors"],
                        "tag": fr["tag"],
                    })
                # Step 8 — AI-vocab comparison on top-1 pair
                neigh_ai_rate = ai_strict_rate(neigh_texts, ai_pat)
                ai_vocab_rows.append({
                    "disappearing_title": title,
                    "top1_2026_neighbor": top1,
                    "cosine": round(top1_cos, 4),
                    "source_n": len(source_texts),
                    "neighbor_n": len(neigh_texts),
                    "source_ai_strict_rate": round(float(source_ai_rate), 4),
                    "neighbor_ai_strict_rate": round(float(neigh_ai_rate), 4),
                    "ai_rate_delta": round(float(neigh_ai_rate - source_ai_rate), 4),
                })

                # Step 9 — manual inspection sample (10 postings from each side)
                n_show = min(10, len(source_texts), len(neigh_texts))
                src_sample = df_disappearing[df_disappearing["title_normalized"] == title].head(n_show)
                nbr_sample = df_2026_capped[df_2026_capped["title_normalized"] == top1].head(n_show)
                for _, ss in src_sample.iterrows():
                    txt = (ss["description_cleaned"] or "")[:600]
                    manual_rows.append({
                        "disappearing_title": title,
                        "top1_2026_neighbor": top1,
                        "side": "disappearing",
                        "uid": ss["uid"],
                        "company": ss.get("company_name_canonical"),
                        "text_snippet": txt,
                    })
                for _, ns in nbr_sample.iterrows():
                    txt = (ns["description_cleaned"] or "")[:600]
                    manual_rows.append({
                        "disappearing_title": title,
                        "top1_2026_neighbor": top1,
                        "side": "neighbor_2026",
                        "uid": ns["uid"],
                        "company": ns.get("company_name_canonical"),
                        "text_snippet": txt,
                    })

    sub_df = pd.DataFrame(sub_rows)
    sub_df.to_csv(OUT_TABLES / "substitution_table.csv", index=False)
    pd.DataFrame(content_drift_rows).to_csv(OUT_TABLES / "content_drift_per_pair.csv", index=False)
    pd.DataFrame(ai_vocab_rows).to_csv(OUT_TABLES / "ai_vocab_comparison.csv", index=False)
    pd.DataFrame(manual_rows).to_csv(OUT_TABLES / "manual_inspection.csv", index=False)

    # Aggregate class for the strict-list subset
    strict_titles_set = set(strict["title_normalized"].tolist())
    sub_df["in_strict"] = sub_df["disappearing_title"].isin(strict_titles_set)

    # Summary table: top1 per disappearing title
    top1 = sub_df[sub_df["rank"] == 1].copy()
    top1.to_csv(OUT_TABLES / "substitution_table_top1.csv", index=False)

    # sensitivity: non-aggregator rerun on top1
    print("[sensitivity] non-aggregator rerun on top1 ...")
    df_nonagg = df[~df["is_aggregator"].fillna(False)].copy()
    df_2024_na = df_nonagg[df_nonagg["period_era"] == "2024"].copy()
    df_2026_na = df_nonagg[df_nonagg["period_era"] == "2026"].copy()
    # reuse same 2026 universe
    df_2026_na_uni = df_2026_na[df_2026_na["title_normalized"].isin(universe_titles)].copy()
    df_2026_na_cap = cap_per_company(df_2026_na_uni, COMPANY_CAP, "title_normalized")
    df_disap_na = df_2024_na[df_2024_na["title_normalized"].isin(disappearing_titles)].copy()
    df_disap_na = cap_per_company(df_disap_na, COMPANY_CAP, "title_normalized")

    disap_stats_na, disap_cent_na = centroid_per_title(df_disap_na, "title_normalized", "description_cleaned", vec)
    nbr_stats_na, nbr_cent_na = centroid_per_title(df_2026_na_cap, "title_normalized", "description_cleaned", vec)
    nbr_mat_na = np.vstack([nbr_cent_na[t] for t in nbr_stats_na["title_normalized"]])
    nbr_labels_na = nbr_stats_na["title_normalized"].tolist()

    sens_rows = []
    for _, r in disap_stats_na.iterrows():
        title = r["title_normalized"]
        topk = cosine_topk(disap_cent_na[title], nbr_mat_na, nbr_labels_na, 1)
        if topk:
            sens_rows.append({
                "disappearing_title": title,
                "n_2024_text_nonagg": int(r["n_postings"]),
                "top1_2026_neighbor_nonagg": topk[0][0],
                "cosine_nonagg": round(topk[0][1], 4),
            })
    pd.DataFrame(sens_rows).to_csv(OUT_TABLES / "sensitivity_nonagg_top1.csv", index=False)

    meta["universe_titles"] = len(universe_titles)
    meta["disappearing_with_text"] = len(disappearing_stats)
    meta["duration_sec"] = round(time.time() - t_start, 2)
    with open(OUT_TABLES / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[done] T36 in {meta['duration_sec']}s")
    print(f"  substitution table rows: {len(sub_df)}")


if __name__ == "__main__":
    main()
