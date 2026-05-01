"""T10 — Title taxonomy evolution (SWE, LinkedIn-only).

Compares title vocabularies between 2024 (kaggle sources) and 2026 (scraped),
with arshkon-only sensitivity. Produces tables, figures, and diagnostic outputs
consumed by exploration/reports/T10.md.

Key design choices
------------------
* Two title columns analyzed:
  - `title_normalized` (seniority-stripped, useful for concentration/vocabulary)
  - `title_raw_clean` (lowercased raw title, preserves seniority markers)
* Two 2024 baselines:
  - "combined" — arshkon + asaniczka
  - "arshkon_only" — aggregator-sensitivity variant (asaniczka is an aggregator)
* Seniority distribution uses the combined best-available column:
      labeled -> seniority_llm else rule_sufficient -> seniority_final
* Default filters: source_platform='linkedin', is_english=true, date_flag='ok', is_swe=true.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNIFIED = ROOT / "data" / "unified.parquet"
FIG_DIR = ROOT / "exploration" / "figures" / "T10"
TAB_DIR = ROOT / "exploration" / "tables" / "T10"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

BASE_FILTER = """
  source_platform = 'linkedin'
  AND is_english = true
  AND date_flag = 'ok'
  AND is_swe = true
"""

SENIORITY_CASE = """
  CASE
    WHEN llm_classification_coverage = 'labeled'         THEN seniority_llm
    WHEN llm_classification_coverage = 'rule_sufficient' THEN seniority_final
    ELSE NULL
  END
"""

# Augmented fallback: use the intended combined column when in frame, else seniority_final.
# Only used where we need a seniority value on every row (e.g., top-N title distributions).
SENIORITY_CASE_AUG = """
  CASE
    WHEN llm_classification_coverage = 'labeled'         THEN seniority_llm
    WHEN llm_classification_coverage = 'rule_sufficient' THEN seniority_final
    WHEN seniority_final IS NOT NULL                     THEN seniority_final
    ELSE NULL
  END
"""


def _clean_raw_title(t: str) -> str:
    if t is None:
        return ""
    s = str(t).lower().strip()
    # Strip salary suffixes (common pattern "- up to $200k"), parenthetical tech stacks
    s = re.sub(r"\$[0-9][0-9k.,+\- ]*", "", s)
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)
    s = re.sub(r"[/\-_,|]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_frame() -> pd.DataFrame:
    con = duckdb.connect()
    q = f"""
    SELECT
        uid,
        source,
        period,
        title,
        title_normalized,
        is_aggregator,
        company_name_effective,
        {SENIORITY_CASE} AS seniority_best_available,
        {SENIORITY_CASE_AUG} AS seniority_best_available_aug,
        seniority_final,
        seniority_native,
        yoe_extracted
    FROM '{UNIFIED}'
    WHERE {BASE_FILTER}
    """
    df = con.execute(q).fetchdf()
    df["title_clean"] = df["title"].map(_clean_raw_title)
    df["title_norm_clean"] = df["title_normalized"].fillna("").str.lower().map(
        lambda s: re.sub(r"\s+", " ", re.sub(r"[/\-_,|]", " ", s)).strip()
    )
    # Period bucket: 2024 (both kaggle) vs 2026 (scraped)
    df["year"] = df["period"].str.slice(0, 4)
    return df


def title_vocabulary_comparison(df: pd.DataFrame, tab_dir: Path) -> dict:
    """Build new/disappeared title lists under two sensitivity variants."""
    out = {}
    for variant, filt in [
        ("combined", df["year"].isin(["2024", "2026"])),
        ("arshkon_only", ((df["source"] == "kaggle_arshkon") & (df["year"] == "2024")) | (df["year"] == "2026")),
    ]:
        sub = df[filt]
        titles_2024 = sub[sub["year"] == "2024"]
        titles_2026 = sub[sub["year"] == "2026"]

        vocab_2024 = Counter(titles_2024["title_norm_clean"])
        vocab_2026 = Counter(titles_2026["title_norm_clean"])

        # Remove empty
        vocab_2024.pop("", None)
        vocab_2026.pop("", None)

        # Use frequency threshold to filter noise (>= 3 in the period)
        kept_2024 = {t for t, c in vocab_2024.items() if c >= 3}
        kept_2026 = {t for t, c in vocab_2026.items() if c >= 3}

        only_2026 = kept_2026 - kept_2024  # new titles
        only_2024 = kept_2024 - kept_2026  # disappeared
        shared = kept_2024 & kept_2026

        # Record top-50 by 2026 frequency for new titles, top-50 by 2024 frequency for disappeared
        new_df = pd.DataFrame(
            [(t, vocab_2026[t], vocab_2024.get(t, 0)) for t in only_2026],
            columns=["title", "n_2026", "n_2024"],
        ).sort_values("n_2026", ascending=False).head(100)
        disappeared_df = pd.DataFrame(
            [(t, vocab_2024[t], vocab_2026.get(t, 0)) for t in only_2024],
            columns=["title", "n_2024", "n_2026"],
        ).sort_values("n_2024", ascending=False).head(100)

        new_df.to_csv(tab_dir / f"new_titles_{variant}.csv", index=False)
        disappeared_df.to_csv(tab_dir / f"disappeared_titles_{variant}.csv", index=False)

        out[variant] = {
            "n_2024_rows": int(len(titles_2024)),
            "n_2026_rows": int(len(titles_2026)),
            "unique_titles_2024": int(len(vocab_2024)),
            "unique_titles_2026": int(len(vocab_2026)),
            "unique_freq3_2024": int(len(kept_2024)),
            "unique_freq3_2026": int(len(kept_2026)),
            "new_titles_n": int(len(only_2026)),
            "disappeared_titles_n": int(len(only_2024)),
            "shared_titles_n": int(len(shared)),
            "top_new_titles": new_df.head(25).values.tolist(),
            "top_disappeared": disappeared_df.head(25).values.tolist(),
        }
    return out


def title_concentration(df: pd.DataFrame, tab_dir: Path) -> pd.DataFrame:
    """Unique titles per 1,000 postings + concentration (top-10, top-50 share, HHI)."""
    rows = []
    groups = [
        ("2024_combined", df[df["year"] == "2024"]),
        ("2024_arshkon", df[(df["year"] == "2024") & (df["source"] == "kaggle_arshkon")]),
        ("2024_asaniczka", df[(df["year"] == "2024") & (df["source"] == "kaggle_asaniczka")]),
        ("2026_scraped", df[df["year"] == "2026"]),
    ]
    for name, sub in groups:
        vocab = Counter(sub["title_norm_clean"])
        vocab.pop("", None)
        n = sum(vocab.values())
        unique = len(vocab)
        if n == 0:
            continue
        freqs = np.array(sorted(vocab.values(), reverse=True))
        shares = freqs / n
        top10 = shares[:10].sum()
        top50 = shares[:50].sum()
        hhi = float((shares ** 2).sum())
        rows.append(
            {
                "group": name,
                "n_postings": int(n),
                "unique_titles": int(unique),
                "unique_per_1k_postings": round(1000 * unique / n, 2),
                "top10_share": round(float(top10), 4),
                "top50_share": round(float(top50), 4),
                "hhi": round(hhi, 6),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(tab_dir / "title_concentration.csv", index=False)
    return out


# Compound/hybrid AI-themed terms
AI_TERMS = [
    r"\bai\b", r"\bml\b", r"machine learning", r"\bllm\b", r"\ballms\b",
    r"\bgenai\b", r"generative ai", r"\bnlp\b", r"deep learning",
    r"\bdata\b", r"\bagent\b", r"\bagents\b", r"prompt", r"chatgpt",
    r"neural", r"computer vision", r"\bcv\b",
]
AI_RE = re.compile("|".join(AI_TERMS), flags=re.I)


def compound_hybrid_titles(df: pd.DataFrame, tab_dir: Path) -> pd.DataFrame:
    df = df.copy()
    df["title_has_ai_term"] = df["title_clean"].str.contains(AI_RE, na=False)
    rows = []
    for variant, filt in [
        ("combined", df["year"].isin(["2024", "2026"])),
        ("arshkon_only", ((df["source"] == "kaggle_arshkon") & (df["year"] == "2024")) | (df["year"] == "2026")),
    ]:
        sub = df[filt]
        g = sub.groupby("year")["title_has_ai_term"].agg(["sum", "count"]).reset_index()
        g["share"] = (g["sum"] / g["count"]).round(4)
        g["variant"] = variant
        rows.append(g)
    tab = pd.concat(rows, ignore_index=True)
    tab.to_csv(tab_dir / "ai_terms_in_title.csv", index=False)
    return tab


# Seniority markers in raw title
JUNIOR_RE = re.compile(r"\b(junior|jr\.?|entry[ -]?level|intern|new ?grad|graduate|trainee|associate i\b|level i\b|i\b)", re.I)
# Note: "associate" alone too noisy; we require suffix or explicit entry cue.
JUNIOR_STRICT = re.compile(r"\b(junior|jr\.?|entry[ -]?level|intern|new ?grad|graduate|trainee)\b", re.I)
SENIOR_RE = re.compile(r"\b(senior|sr\.?|staff|principal|lead|distinguished|fellow)\b", re.I)


def seniority_markers_in_title(df: pd.DataFrame, tab_dir: Path, fig_dir: Path) -> pd.DataFrame:
    df = df.copy()
    df["title_is_junior"] = df["title_clean"].str.contains(JUNIOR_STRICT, na=False)
    df["title_is_senior"] = df["title_clean"].str.contains(SENIOR_RE, na=False)
    rows = []
    for variant, filt in [
        ("combined", df["year"].isin(["2024", "2026"])),
        ("arshkon_only", ((df["source"] == "kaggle_arshkon") & (df["year"] == "2024")) | (df["year"] == "2026")),
    ]:
        sub = df[filt]
        g = sub.groupby("year").agg(
            n=("title_clean", "count"),
            junior_n=("title_is_junior", "sum"),
            senior_n=("title_is_senior", "sum"),
        ).reset_index()
        g["junior_share"] = (g["junior_n"] / g["n"]).round(4)
        g["senior_share"] = (g["senior_n"] / g["n"]).round(4)
        g["variant"] = variant
        rows.append(g)
    tab = pd.concat(rows, ignore_index=True)
    tab.to_csv(tab_dir / "seniority_markers_in_title.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in zip(axes, ["junior_share", "senior_share"], ["Junior/entry markers", "Senior/lead markers"]):
        for variant in ["combined", "arshkon_only"]:
            sub = tab[tab["variant"] == variant]
            ax.plot(sub["year"], sub[col] * 100, marker="o", label=variant)
        ax.set_title(title)
        ax.set_ylabel("Share of titles (%)")
        ax.set_xlabel("Year")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "seniority_markers_in_title.png", dpi=150)
    plt.close(fig)
    return tab


def seniority_distribution_shifts(df: pd.DataFrame, tab_dir: Path) -> pd.DataFrame:
    """For the top-20 shared (normalized) titles, seniority distribution in 2024 vs 2026."""
    # Shared titles by frequency
    vocab_2024 = Counter(df[df["year"] == "2024"]["title_norm_clean"])
    vocab_2026 = Counter(df[df["year"] == "2026"]["title_norm_clean"])
    shared = [
        t for t in vocab_2024
        if t and t in vocab_2026 and vocab_2024[t] >= 20 and vocab_2026[t] >= 20
    ]
    shared_sorted = sorted(shared, key=lambda t: vocab_2024[t] + vocab_2026[t], reverse=True)[:20]

    rows = []
    for t in shared_sorted:
        for year in ["2024", "2026"]:
            sub = df[(df["year"] == year) & (df["title_norm_clean"] == t)]
            sen = sub["seniority_best_available_aug"].fillna("missing").value_counts(normalize=True)
            row = {"title": t, "year": year, "n": int(len(sub))}
            for level in ["entry", "associate", "mid-senior", "director", "unknown", "missing"]:
                row[level] = round(float(sen.get(level, 0)), 4)
            rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(tab_dir / "top20_shared_seniority_shift.csv", index=False)
    return tab


def title_content_alignment(df: pd.DataFrame, tab_dir: Path) -> pd.DataFrame:
    """For the 10 most common shared titles, compute cosine similarity between 2024 and 2026
    description_core_llm / description_core samples (using cleaned text from shared artifact)."""
    # Load cleaned text
    cleaned_path = ROOT / "exploration" / "artifacts" / "shared" / "swe_cleaned_text.parquet"
    con = duckdb.connect()
    text_df = con.execute(f"SELECT uid, description_cleaned FROM '{cleaned_path}'").fetchdf()
    merged = df.merge(text_df, on="uid", how="inner")
    merged = merged[merged["description_cleaned"].str.len() > 50]

    vocab_2024 = Counter(merged[merged["year"] == "2024"]["title_norm_clean"])
    vocab_2026 = Counter(merged[merged["year"] == "2026"]["title_norm_clean"])
    shared = [
        t for t in vocab_2024
        if t and t in vocab_2026 and vocab_2024[t] >= 30 and vocab_2026[t] >= 30
    ]
    shared_sorted = sorted(shared, key=lambda t: vocab_2024[t] + vocab_2026[t], reverse=True)[:10]

    rows = []
    for t in shared_sorted:
        docs_2024 = merged[(merged["year"] == "2024") & (merged["title_norm_clean"] == t)]["description_cleaned"].tolist()
        docs_2026 = merged[(merged["year"] == "2026") & (merged["title_norm_clean"] == t)]["description_cleaned"].tolist()
        # Concatenate per period
        corpus = [" ".join(docs_2024), " ".join(docs_2026)]
        try:
            vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 1), min_df=2)
            X = vec.fit_transform(corpus)
            sim = float(cosine_similarity(X[0], X[1])[0, 0])
        except ValueError:
            sim = np.nan
        rows.append({
            "title": t,
            "n_2024": len(docs_2024),
            "n_2026": len(docs_2026),
            "tfidf_cosine_sim": round(sim, 4),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(tab_dir / "title_content_alignment.csv", index=False)
    return tab


def emerging_role_categories(df: pd.DataFrame, tab_dir: Path) -> dict:
    """Group new 2026 titles by theme (AI/ML, platform, reliability, data, security, full-stack, etc.)."""
    vocab_2024 = set(df[df["year"] == "2024"]["title_norm_clean"])
    vocab_2026 = Counter(df[df["year"] == "2026"]["title_norm_clean"])
    only_2026 = {t: c for t, c in vocab_2026.items() if t and t not in vocab_2024 and c >= 3}

    themes = {
        "ai_ml": re.compile(r"\b(ai|ml|llm|machine learning|deep learning|gen ?ai|agentic|agent|mlops|nlp|computer vision|cv|neural|transformer)\b", re.I),
        "platform": re.compile(r"\b(platform|infrastructure|infra|cloud|kubernetes)\b", re.I),
        "reliability": re.compile(r"\b(sre|reliability|observability|dev ?ops)\b", re.I),
        "data": re.compile(r"\b(data|analytics|bi|warehouse|etl|pipeline)\b", re.I),
        "security": re.compile(r"\b(security|sec ?ops|appsec|application security|cyber)\b", re.I),
        "full_stack": re.compile(r"\b(full ?stack|full-stack)\b", re.I),
        "frontend": re.compile(r"\b(front ?end|react|ui)\b", re.I),
        "backend": re.compile(r"\b(back ?end|backend|api)\b", re.I),
        "mobile": re.compile(r"\b(mobile|ios|android)\b", re.I),
        "embedded": re.compile(r"\b(embedded|firmware|hardware)\b", re.I),
    }

    theme_counts = {k: 0 for k in themes}
    theme_volume = {k: 0 for k in themes}
    uncategorized = []
    for title, count in only_2026.items():
        matched = False
        for name, pat in themes.items():
            if pat.search(title):
                theme_counts[name] += 1
                theme_volume[name] += count
                matched = True
        if not matched:
            uncategorized.append((title, count))

    tab = pd.DataFrame([
        {"theme": k, "n_unique_new_titles": theme_counts[k], "n_postings": theme_volume[k]}
        for k in themes
    ]).sort_values("n_postings", ascending=False)
    tab.to_csv(tab_dir / "emerging_themes.csv", index=False)

    unc = pd.DataFrame(sorted(uncategorized, key=lambda x: -x[1])[:50], columns=["title", "n"])
    unc.to_csv(tab_dir / "emerging_uncategorized_top50.csv", index=False)

    return {"themes": tab, "uncategorized_top25": unc.head(25).values.tolist()}


def main() -> None:
    print("Loading unified frame…")
    df = load_frame()
    print(f"Loaded {len(df):,} SWE LinkedIn rows.")

    print("\n[1] Title vocabulary comparison…")
    vocab = title_vocabulary_comparison(df, TAB_DIR)
    for k, v in vocab.items():
        print(f"  {k}: new={v['new_titles_n']} disappeared={v['disappeared_titles_n']} shared={v['shared_titles_n']}")

    print("\n[2] Title concentration…")
    conc = title_concentration(df, TAB_DIR)
    print(conc.to_string(index=False))

    print("\n[3] Compound/AI titles…")
    ai_tab = compound_hybrid_titles(df, TAB_DIR)
    print(ai_tab.to_string(index=False))

    print("\n[5] Seniority markers (junior/senior in title)…")
    marker_tab = seniority_markers_in_title(df, TAB_DIR, FIG_DIR)
    print(marker_tab.to_string(index=False))

    print("\n[1b] Seniority distribution shifts for top-20 shared titles…")
    shift = seniority_distribution_shifts(df, TAB_DIR)
    print(shift.head(12).to_string(index=False))

    print("\n[4] Title-content alignment (top 10 shared)…")
    align = title_content_alignment(df, TAB_DIR)
    print(align.to_string(index=False))

    print("\n[6] Emerging role categories (new 2026 titles by theme)…")
    themes = emerging_role_categories(df, TAB_DIR)
    print(themes["themes"].to_string(index=False))

    # Save summary JSON
    summary = {
        "n_rows_total": int(len(df)),
        "vocabulary": {k: {kk: v[kk] for kk in ["n_2024_rows", "n_2026_rows", "unique_titles_2024", "unique_titles_2026", "unique_freq3_2024", "unique_freq3_2026", "new_titles_n", "disappeared_titles_n", "shared_titles_n"]} for k, v in vocab.items()},
    }
    with open(TAB_DIR / "T10_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nDone. Outputs written to:", TAB_DIR, "and", FIG_DIR)


if __name__ == "__main__":
    main()
