"""T10 — Title taxonomy evolution (2024 → 2026).

Steps:
 1. Title vocabulary comparison (new / disappeared / changed).
 2. Title concentration & fragmentation.
 3. Compound/hybrid AI-era titles.
 4. Title-to-content alignment via TF-IDF cosine on descriptions for shared titles.
 5. Title inflation/deflation signals (senior / junior markers).
 6. Emerging role categories (light theme grouping on new titles).

Primary slice: SWE, LinkedIn, default filter.
Use raw `title` lowercased. T30 finding says `title_normalized` strips level markers.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNI = ROOT / "data" / "unified.parquet"
TEXT = ROOT / "exploration" / "artifacts" / "shared" / "swe_cleaned_text.parquet"
TABLES = ROOT / "exploration" / "tables" / "T10"
FIGS = ROOT / "exploration" / "figures" / "T10"
ART = ROOT / "exploration" / "artifacts" / "T10"
SPEC_CSV = ROOT / "exploration" / "artifacts" / "shared" / "entry_specialist_employers.csv"

SLICE_SQL = (
    "source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true"
)

# Seniority markers for step 5
SENIORITY_MARKERS = [
    ("senior", r"\bsenior\b|\bsnr\b|\bsr\.?\b"),
    ("lead", r"\blead\b"),
    ("principal", r"\bprincipal\b|\bprinciple\b"),
    ("staff", r"\bstaff\b"),
    ("junior", r"\bjunior\b|\bjr\.?\b"),
    ("associate", r"\bassociate\b"),
    ("intern", r"\bintern\b|\binternship\b"),
    ("entry", r"\bentry[- ]level\b|\bgraduate\b|\bgrad\b|\bnew[- ]grad\b"),
]

AI_TITLE_TOKENS = [
    ("ai", r"\bai\b|\bartificial intelligence\b"),
    ("ml", r"\bml\b|\bmachine learning\b"),
    ("llm", r"\bllm\b|\blarge language\b"),
    ("genai", r"\bgen ?ai\b|\bgenerative ai\b"),
    ("agent", r"\bagent(s|ic)?\b"),
    ("data", r"\bdata\b"),
    ("intelligence", r"\bintelligence\b"),
    ("mlops", r"\bmlops\b"),
    ("nlp", r"\bnlp\b|\bnatural language\b"),
]


def load_title_frame(con: duckdb.DuckDBPyConnection, *, specialist_excl: bool = True) -> pd.DataFrame:
    spec_filter = ""
    if specialist_excl:
        con.execute(
            f"CREATE OR REPLACE TEMP VIEW spec AS SELECT lower(company) c FROM read_csv('{SPEC_CSV}')"
        )
        spec_filter = "AND lower(company_name_canonical) NOT IN (SELECT c FROM spec)"
    q = f"""
    SELECT
        uid, source, period,
        lower(title) AS title_lc,
        seniority_final, seniority_3level,
        company_name_canonical,
        is_aggregator,
        swe_classification_tier,
        yoe_extracted
    FROM read_parquet('{UNI}')
    WHERE {SLICE_SQL}
      AND (is_aggregator IS NULL OR is_aggregator = false)
      {spec_filter}
    """
    df = con.execute(q).df()
    df["period_bucket"] = np.where(df["source"] == "scraped", "2026", "2024")
    df["source_bucket"] = df["source"].map(
        {"kaggle_arshkon": "arshkon", "kaggle_asaniczka": "asaniczka", "scraped": "scraped"}
    )
    # Normalize whitespace in lowered title
    df["title_lc"] = df["title_lc"].fillna("").str.strip()
    df["title_lc"] = df["title_lc"].str.replace(r"\s+", " ", regex=True)
    return df


def step1_vocabulary(df: pd.DataFrame) -> dict:
    """New / disappeared / shared titles."""
    arsh = df[df["source_bucket"] == "arshkon"]
    scr = df[df["source_bucket"] == "scraped"]

    arsh_counts = arsh["title_lc"].value_counts()
    scr_counts = scr["title_lc"].value_counts()
    asaniczka = df[df["source_bucket"] == "asaniczka"]
    asaniczka_counts = asaniczka["title_lc"].value_counts()

    arsh_set = set(arsh_counts.index)
    scr_set = set(scr_counts.index)
    pooled_2024 = arsh_set | set(asaniczka_counts.index)

    # NEW titles: in scraped but not arshkon, with rank by scraped volume
    new_titles = scr_counts[~scr_counts.index.isin(arsh_set)].sort_values(ascending=False)
    # Stricter "truly new": not in arshkon AND not in asaniczka
    truly_new = scr_counts[~scr_counts.index.isin(pooled_2024)].sort_values(ascending=False)
    disappeared = arsh_counts[~arsh_counts.index.isin(scr_set)].sort_values(ascending=False)

    # Shared high-frequency titles
    shared = sorted(arsh_set & scr_set)
    shared_panel = pd.DataFrame(
        {
            "title_lc": shared,
            "arshkon_n": [int(arsh_counts.get(t, 0)) for t in shared],
            "scraped_n": [int(scr_counts.get(t, 0)) for t in shared],
            "asaniczka_n": [int(asaniczka_counts.get(t, 0)) for t in shared],
        }
    ).sort_values("scraped_n", ascending=False)

    # Seniority mix change for top 30 shared titles
    top_shared = shared_panel.head(30)["title_lc"].tolist()
    mix_rows = []
    for t in top_shared:
        sub = df[df["title_lc"] == t]
        for sb in ["arshkon", "asaniczka", "scraped"]:
            s = sub[sub["source_bucket"] == sb]
            if len(s) == 0:
                continue
            j2 = (s["seniority_final"].isin(["entry", "associate"])).mean()
            mid_senior = (s["seniority_final"] == "mid-senior").mean()
            s1 = (s["seniority_final"].isin(["mid-senior", "director"])).mean()
            mix_rows.append(
                dict(
                    title_lc=t,
                    source_bucket=sb,
                    n=len(s),
                    j2_share=j2,
                    mid_senior_share=mid_senior,
                    s1_share=s1,
                    senior_unknown=float(s["seniority_final"].isin(["unknown"]).mean()),
                )
            )
    mix_df = pd.DataFrame(mix_rows)

    # Write
    truly_new.head(500).rename("n").reset_index().rename(columns={"title_lc": "title_lc"}).to_csv(
        TABLES / "T10_new_titles_scraped_only.csv", index=False
    )
    new_titles.head(500).rename("n").reset_index().to_csv(
        TABLES / "T10_new_titles_vs_arshkon.csv", index=False
    )
    disappeared.head(500).rename("n").reset_index().to_csv(
        TABLES / "T10_disappeared_titles.csv", index=False
    )
    shared_panel.head(200).to_csv(TABLES / "T10_shared_titles_top200.csv", index=False)
    mix_df.to_csv(TABLES / "T10_shared_titles_seniority_mix.csv", index=False)

    return dict(
        new_top=new_titles.head(30).to_dict(),
        truly_new_top=truly_new.head(30).to_dict(),
        disappeared_top=disappeared.head(30).to_dict(),
        n_new=int(len(new_titles)),
        n_truly_new=int(len(truly_new)),
        n_disappeared=int(len(disappeared)),
        n_shared=int(len(shared)),
    )


def step2_concentration(df: pd.DataFrame) -> dict:
    """Unique titles per 1K SWE postings; HHI of title shares."""
    rows = []
    for sb in ["arshkon", "asaniczka", "scraped"]:
        sub = df[df["source_bucket"] == sb]
        if len(sub) == 0:
            continue
        n = len(sub)
        u = sub["title_lc"].nunique()
        shares = sub["title_lc"].value_counts(normalize=True)
        hhi = float((shares ** 2).sum())
        top1 = float(shares.iloc[0]) if len(shares) else 0.0
        top10 = float(shares.head(10).sum()) if len(shares) >= 10 else float(shares.sum())
        singleton = int((sub["title_lc"].value_counts() == 1).sum())
        rows.append(
            dict(
                source_bucket=sb,
                n_postings=n,
                unique_titles=u,
                unique_per_1k=u / n * 1000.0,
                hhi=hhi,
                top1_share=top1,
                top10_share=top10,
                singleton_titles=singleton,
                singleton_share=singleton / u if u else 0.0,
            )
        )
    cdf = pd.DataFrame(rows)
    cdf.to_csv(TABLES / "T10_title_concentration.csv", index=False)
    return dict(concentration=cdf.to_dict("records"))


def step3_ai_titles(df: pd.DataFrame) -> dict:
    """AI-related title share."""
    rows = []
    for sb in ["arshkon", "asaniczka", "scraped"]:
        sub = df[df["source_bucket"] == sb]
        if len(sub) == 0:
            continue
        row = {"source_bucket": sb, "n": len(sub)}
        combined = sub["title_lc"]
        any_ai = np.zeros(len(sub), dtype=bool)
        for name, pat in AI_TITLE_TOKENS:
            m = combined.str.contains(pat, regex=True, na=False)
            row[f"{name}_share"] = float(m.mean())
            row[f"{name}_n"] = int(m.sum())
            if name in {"ai", "ml", "llm", "genai", "agent", "mlops", "nlp"}:
                any_ai |= m.values
        row["ai_related_title_share"] = float(any_ai.mean())
        row["ai_related_title_n"] = int(any_ai.sum())
        rows.append(row)
    ddf = pd.DataFrame(rows)
    ddf.to_csv(TABLES / "T10_ai_title_share.csv", index=False)

    # Rank-order AI titles themselves
    scr = df[df["source_bucket"] == "scraped"]
    scr_ai = scr[scr["title_lc"].str.contains(r"\b(ai|ml|llm|agent|machine learning)\b", regex=True, na=False)]
    top = scr_ai["title_lc"].value_counts().head(40).rename("n").reset_index()
    top.to_csv(TABLES / "T10_top_ai_titles_scraped.csv", index=False)
    return dict(ai_table=ddf.to_dict("records"))


def step4_tfidf_alignment(df: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> dict:
    """For top shared titles, compute TF-IDF cosine on concatenated cleaned text per title."""
    # Take top 10 titles present in both arshkon and scraped with ≥50 rows each
    arsh = df[df["source_bucket"] == "arshkon"]
    scr = df[df["source_bucket"] == "scraped"]
    a_cnt = arsh["title_lc"].value_counts()
    s_cnt = scr["title_lc"].value_counts()
    candidates = []
    for t, n in s_cnt.items():
        if n >= 50 and a_cnt.get(t, 0) >= 20:
            candidates.append((t, a_cnt.get(t, 0), n))
        if len(candidates) >= 20:
            break
    top = candidates[:10]

    # Load cleaned LLM text only, join to uid/title
    uids = df[df["title_lc"].isin([t for t, _, _ in top])][["uid", "title_lc", "source_bucket"]]
    # Register uids list and join in DuckDB
    con.register("title_uids", uids[["uid", "title_lc", "source_bucket"]])
    text = con.execute(
        f"""
        SELECT u.uid, u.title_lc, u.source_bucket, t.description_cleaned
        FROM title_uids u
        JOIN read_parquet('{TEXT}') t USING(uid)
        WHERE t.text_source = 'llm'
        """
    ).df()
    joined = text.copy()

    rows = []
    for t, _, _ in top:
        a_docs = joined[(joined["title_lc"] == t) & (joined["source_bucket"] == "arshkon")]["description_cleaned"].dropna().tolist()
        s_docs = joined[(joined["title_lc"] == t) & (joined["source_bucket"] == "scraped")]["description_cleaned"].dropna().tolist()
        if len(a_docs) < 5 or len(s_docs) < 5:
            rows.append(dict(title_lc=t, n_arsh=len(a_docs), n_scr=len(s_docs), cosine=None))
            continue
        corpus = a_docs + s_docs
        try:
            vec = TfidfVectorizer(max_features=5000, min_df=2, ngram_range=(1, 2))
            tf = vec.fit_transform(corpus)
            a_mean = np.asarray(tf[: len(a_docs)].mean(axis=0))
            s_mean = np.asarray(tf[len(a_docs) :].mean(axis=0))
            cos = float(cosine_similarity(a_mean, s_mean)[0, 0])
        except Exception as e:
            cos = None
        rows.append(dict(title_lc=t, n_arsh=len(a_docs), n_scr=len(s_docs), cosine=cos))

    adf = pd.DataFrame(rows)
    adf.to_csv(TABLES / "T10_title_content_alignment.csv", index=False)
    return dict(alignment=adf.to_dict("records"))


def step5_seniority_markers(df: pd.DataFrame) -> dict:
    rows = []
    for sb in ["arshkon", "asaniczka", "scraped"]:
        sub = df[df["source_bucket"] == sb]
        if len(sub) == 0:
            continue
        row = {"source_bucket": sb, "n": len(sub)}
        for name, pat in SENIORITY_MARKERS:
            m = sub["title_lc"].str.contains(pat, regex=True, na=False)
            row[f"{name}_share"] = float(m.mean())
            row[f"{name}_n"] = int(m.sum())
        rows.append(row)
    sdf = pd.DataFrame(rows)
    sdf.to_csv(TABLES / "T10_seniority_marker_shares.csv", index=False)

    # Cross with T30 seniority_final (J2 primary)
    mix_rows = []
    for sb in ["arshkon", "asaniczka", "scraped"]:
        sub = df[df["source_bucket"] == sb]
        if len(sub) == 0:
            continue
        is_j2 = sub["seniority_final"].isin(["entry", "associate"])
        is_s1 = sub["seniority_final"].isin(["mid-senior", "director"])
        for name, pat in SENIORITY_MARKERS:
            m = sub["title_lc"].str.contains(pat, regex=True, na=False)
            mix_rows.append(
                dict(
                    source_bucket=sb,
                    marker=name,
                    share_all=float(m.mean()),
                    share_j2=float(m[is_j2].mean()) if is_j2.sum() else None,
                    share_s1=float(m[is_s1].mean()) if is_s1.sum() else None,
                    n_all=len(sub),
                    n_j2=int(is_j2.sum()),
                    n_s1=int(is_s1.sum()),
                )
            )
    mdf = pd.DataFrame(mix_rows)
    mdf.to_csv(TABLES / "T10_marker_by_seniority.csv", index=False)
    return dict(markers=sdf.to_dict("records"))


def step6_emerging_themes(df: pd.DataFrame) -> dict:
    """Theme-group new scraped titles by keyword set."""
    arsh_set = set(df[df["source_bucket"] == "arshkon"]["title_lc"].unique())
    asan_set = set(df[df["source_bucket"] == "asaniczka"]["title_lc"].unique())
    pooled = arsh_set | asan_set
    scr = df[df["source_bucket"] == "scraped"]
    scr_counts = scr["title_lc"].value_counts()
    truly_new_tc = scr_counts[~scr_counts.index.isin(pooled)]

    themes = {
        "ai_ml_engineering": r"\b(ai|ml|machine learning|llm|agent|gen ?ai|mlops|nlp|deep learning|computer vision|generative)\b",
        "platform_infra": r"\b(platform|infrastructure|infra|devops|cloud|kubernetes|reliability|site reliability|sre|observability)\b",
        "data_analytics": r"\b(data|analytics|analyst|bi\b|etl|warehouse|lakehouse|pipeline)\b",
        "security": r"\b(security|appsec|secops|cyber|iam)\b",
        "frontend_fullstack": r"\b(front ?end|full ?stack|ui|ux|react|web)\b",
        "embedded_firmware": r"\b(embedded|firmware|hardware|fpga|iot|robotics)\b",
        "mobile": r"\b(ios|android|mobile|react native)\b",
        "staff_principal_leadership": r"\b(staff|principal|architect|director|head of|vp|lead engineer)\b",
        "qa_testing": r"\b(qa\b|test engineer|quality engineer|sdet)\b",
    }

    rows = []
    for theme, pat in themes.items():
        m = truly_new_tc.index.to_series().str.contains(pat, regex=True, na=False)
        subset = truly_new_tc[m.values]
        rows.append(
            dict(
                theme=theme,
                n_unique_new_titles=int(len(subset)),
                n_postings=int(subset.sum()),
                share_of_new_title_postings=float(subset.sum() / truly_new_tc.sum()) if truly_new_tc.sum() else 0.0,
                top_examples=subset.head(10).to_dict(),
            )
        )
    tdf = pd.DataFrame(rows)
    tdf.to_csv(TABLES / "T10_emerging_themes.csv", index=False)

    # Unclassified (fallback)
    assigned = np.zeros(len(truly_new_tc), dtype=bool)
    for _, pat in themes.items():
        assigned |= truly_new_tc.index.to_series().str.contains(pat, regex=True, na=False).values
    unclassified = truly_new_tc[~assigned]
    unclassified.head(100).rename("n").to_csv(TABLES / "T10_unclassified_new_titles.csv")
    return dict(themes=rows, n_unclassified=int(len(unclassified)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-specialist-excl", action="store_true")
    args = parser.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET memory_limit='24GB'")

    df = load_title_frame(con, specialist_excl=not args.no_specialist_excl)
    print(f"Loaded {len(df):,} rows (specialist_excl={not args.no_specialist_excl})")
    print(df.groupby("source_bucket").size())

    out = {}
    print("Step 1: vocabulary comparison")
    out["step1"] = step1_vocabulary(df)
    print("Step 2: concentration")
    out["step2"] = step2_concentration(df)
    print("Step 3: AI titles")
    out["step3"] = step3_ai_titles(df)
    print("Step 4: TF-IDF alignment")
    out["step4"] = step4_tfidf_alignment(df, con)
    print("Step 5: seniority markers")
    out["step5"] = step5_seniority_markers(df)
    print("Step 6: emerging themes")
    out["step6"] = step6_emerging_themes(df)

    # Robustness: include aggregators
    print("Robustness: including aggregators")
    df_all = load_title_frame(con, specialist_excl=not args.no_specialist_excl)
    # (same df — already aggregator excluded; robustness is different: include aggregators)
    q = f"""
    SELECT lower(title) AS title_lc, source, period, is_aggregator
    FROM read_parquet('{UNI}')
    WHERE {SLICE_SQL}
    """
    raw = con.execute(q).df()
    raw["source_bucket"] = raw["source"].map(
        {"kaggle_arshkon": "arshkon", "kaggle_asaniczka": "asaniczka", "scraped": "scraped"}
    )
    rob = (
        raw.groupby("source_bucket")
        .agg(
            n=("title_lc", "size"),
            unique=("title_lc", "nunique"),
        )
        .assign(unique_per_1k=lambda x: x["unique"] / x["n"] * 1000)
        .reset_index()
    )
    rob.to_csv(TABLES / "T10_concentration_with_aggregators.csv", index=False)

    with open(ART / "T10_summary.json", "w") as fp:
        json.dump(out, fp, indent=2, default=str)
    print("Done.")


if __name__ == "__main__":
    main()
