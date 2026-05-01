"""T18 Step 4 — TF-IDF cosine between SWE and SWE-adjacent per period.

Sample 200 SWE + 200 adjacent per period (2024-01 / 2024-04 / 2026-03 / 2026-04).
Use description_core_llm where labeled; fall back to raw description. Strip the
shared company-name stoplist from `exploration/artifacts/shared/company_stoplist.txt`.
Compute the mean TF-IDF vector for each (group, period) and cosine-sim between
SWE and adjacent within the same period. Question: is the SWE-adjacent group
becoming MORE similar to SWE over time (boundary blur)?
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path("/home/jihgaboot/gabor/job-research")
TAB = ROOT / "exploration" / "tables" / "T18"
FIG = ROOT / "exploration" / "figures" / "T18"
SHARED = ROOT / "exploration" / "artifacts" / "shared"

SEED = 20260417
PER_GROUP_N = 200
PERIODS = ["2024-01", "2024-04", "2026-03", "2026-04"]


def load_stoplist() -> set[str]:
    path = SHARED / "company_stoplist.txt"
    if not path.exists():
        return set()
    words = set()
    with open(path) as f:
        for line in f:
            t = line.strip().lower()
            if t and not t.startswith("#"):
                words.add(t)
    return words


def strip_companies(text: str, stops: set[str]) -> str:
    if not text:
        return ""
    # Simple word-level strip
    out_tokens = []
    for tok in re.findall(r"[A-Za-z][A-Za-z+\-#\.]{1,}", text):
        if tok.lower() in stops:
            continue
        out_tokens.append(tok.lower())
    return " ".join(out_tokens)


def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")

    sql = """
    SELECT uid, source, period, is_swe, is_swe_adjacent,
      COALESCE(NULLIF(description_core_llm, ''), description) AS text,
      description_core_llm IS NOT NULL AND description_core_llm != '' AS used_cleaned,
      llm_extraction_coverage
    FROM read_parquet('/home/jihgaboot/gabor/job-research/data/unified.parquet')
    WHERE source_platform='linkedin'
      AND is_english=true
      AND date_flag='ok'
      AND (is_swe=true OR is_swe_adjacent=true)
    """
    df = con.execute(sql).df()
    print("Loaded:", len(df))

    # Group label
    df["group"] = np.where(df["is_swe"], "SWE", np.where(df["is_swe_adjacent"], "adjacent", None))
    df = df[df["group"].notna()].copy()

    stops = load_stoplist()
    print(f"Loaded {len(stops)} stoplist tokens")

    # Sample
    rng = np.random.default_rng(SEED)
    samples = []
    for period in PERIODS:
        for group in ["SWE", "adjacent"]:
            sub = df[(df["period"] == period) & (df["group"] == group)]
            if len(sub) == 0:
                continue
            n = min(PER_GROUP_N, len(sub))
            idx = rng.choice(len(sub), size=n, replace=False)
            samp = sub.iloc[idx].copy()
            samples.append(samp)
    samp_df = pd.concat(samples, ignore_index=True)
    print("Sample size:", len(samp_df))
    print(samp_df.groupby(["period", "group"]).size())

    # Strip company names + tokenize
    samp_df["text_clean"] = samp_df["text"].fillna("").apply(
        lambda t: strip_companies(t, stops)
    )

    # Fit TF-IDF on the combined sample so vocab is shared
    vec = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
        stop_words="english",
    )
    X = vec.fit_transform(samp_df["text_clean"])

    # Mean vector per (period, group)
    records = []
    for period in PERIODS:
        for group in ["SWE", "adjacent"]:
            mask = (samp_df["period"] == period) & (samp_df["group"] == group)
            if mask.sum() == 0:
                continue
            avg = np.asarray(X[mask.values].mean(axis=0)).reshape(1, -1)
            records.append({"period": period, "group": group, "vec": avg, "n": mask.sum()})

    # Cosine within period
    rows = []
    for period in PERIODS:
        swe_rec = [r for r in records if r["period"] == period and r["group"] == "SWE"]
        adj_rec = [r for r in records if r["period"] == period and r["group"] == "adjacent"]
        if not swe_rec or not adj_rec:
            continue
        cos = cosine_similarity(swe_rec[0]["vec"], adj_rec[0]["vec"])[0, 0]
        rows.append({"period": period, "cosine_swe_adj": float(cos),
                     "n_swe": swe_rec[0]["n"], "n_adj": adj_rec[0]["n"]})

    # Also cross-period SWE vs SWE (anchor in 2024-01)
    anchor_swe = [r for r in records if r["period"] == "2024-01" and r["group"] == "SWE"]
    if anchor_swe:
        anchor = anchor_swe[0]["vec"]
        for r in records:
            c = cosine_similarity(anchor, r["vec"])[0, 0]
            r["cos_to_2024_01_SWE"] = float(c)

    cross = pd.DataFrame([{
        "period": r["period"], "group": r["group"],
        "n": r["n"], "cos_to_2024_01_SWE": r.get("cos_to_2024_01_SWE")
    } for r in records])

    out = pd.DataFrame(rows)
    print("\n=== TF-IDF cosine SWE vs adjacent per period ===")
    print(out.to_string())
    out.to_csv(TAB / "T18_tfidf_cosine_swe_vs_adj.csv", index=False)

    print("\n=== Cosine to 2024-01 SWE anchor ===")
    print(cross.to_string())
    cross.to_csv(TAB / "T18_tfidf_cosine_drift.csv", index=False)


if __name__ == "__main__":
    main()
