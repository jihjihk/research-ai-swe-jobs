"""
S26 DEEP-DIVE — validation of three composite-A findings.

DD1: Hospitals & Health Care leads Software Development in AI-vocab rate.
DD2: FS=SWE parity is regex-dependent.
DD3: Builder-vs-user geographic split, in particular the +8pp user-intensity
     premium in the Bay Area.

Outputs: eda/tables/S26_dd_*.csv  (one per sub-finding).
Stdout : hand-sampled posting excerpts the memo needs to cite.

Run:
  ./.venv/bin/python eda/scripts/S26_deepdive.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scans import AI_VOCAB_PATTERN, BIG_TECH_CANONICAL, text_col, text_filter  # noqa: E402
from S26_composite_a import BUILDER_TITLE_PATTERN  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE = PROJECT_ROOT / "data" / "unified_core.parquet"
TABLES = PROJECT_ROOT / "eda" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

CORE_FILTER = "is_english = TRUE AND date_flag = 'ok' AND is_swe = TRUE"

# Three regex variants for DD2.
# 1) CANONICAL (broad) — matches scans.py AI_VOCAB_PATTERN
REGEX_CANONICAL = AI_VOCAB_PATTERN

# 2) STRICT v9-style: drop single-letter/ambiguous tokens (gpt, rag, llm alone),
# keep only multi-word, unambiguous AI phrases + tool brands with context.
STRICT_PHRASES = [
    "large language model", "generative ai", "genai", "gen ai",
    "foundation model", "transformer model",
    "chatgpt", "openai", "anthropic", "copilot", "claude",
    "github copilot", "cursor ide", "windsurf ide",
    "llm", "prompt engineering", "prompt engineer", "ai agent", "agentic",
    "retrieval augmented", "vector database",
    "mlops", "llmops",
]
REGEX_STRICT = r"(?i)\b(" + "|".join(re.escape(p) for p in STRICT_PHRASES) + r")\b"

# 3) TOOLING-ONLY: only concrete AI-tool / LLM platform mentions.
# Ignores "machine learning" / "AI" alone / "prompt engineering" — tests what
# fraction of matches are about *coding-agent* AI rather than classical ML.
TOOLING_PHRASES = [
    "chatgpt", "claude", "copilot", "openai", "anthropic",
    "github copilot", "cursor ide", "windsurf ide",
    "llm", "genai", "gen ai", "generative ai", "foundation model",
    "ai agent", "agentic", "retrieval augmented", "rag",
    "prompt engineering", "prompt engineer",
    "llmops",
]
REGEX_TOOLING = r"(?i)\b(" + "|".join(re.escape(p) for p in TOOLING_PHRASES) + r")\b"

REGEX_VARIANTS = {
    "canonical_broad": REGEX_CANONICAL,
    "strict_v9like": REGEX_STRICT,
    "tooling_only": REGEX_TOOLING,
}

TECH_HUBS = {
    "San Francisco Bay Area",
    "Seattle Metro",
    "New York City Metro",
    "Austin Metro",
    "Boston Metro",
}

random.seed(7)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / denom, (c + h) / denom


# ---------------------------------------------------------------------------
# DD1 — Hospitals & Health Care lead?
# ---------------------------------------------------------------------------

def deepdive_1(con):
    print("\n==== DD1 — Hospitals & Health Care audit ====")

    # 1. Volume by period — hospitals label coverage.
    vol = con.execute(f"""
      SELECT period,
             COUNT(*) FILTER (WHERE company_industry = 'Hospitals and Health Care') AS n_hospitals,
             COUNT(*) FILTER (WHERE company_industry IS NOT NULL) AS n_labeled,
             COUNT(*) AS n_total
      FROM '{CORE}'
      WHERE {CORE_FILTER}
      GROUP BY 1 ORDER BY 1
    """).df()
    vol.to_csv(TABLES / "S26_dd1_volume_by_period.csv", index=False)
    print("Hospitals volume by period:")
    print(vol.to_string())

    # 2a. Title distribution within Hospitals, 2026.
    titles = con.execute(f"""
      SELECT LOWER(title) AS title_lc, COUNT(*) AS n,
             SUM(CASE WHEN regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}') THEN 1 ELSE 0 END) AS n_ai
      FROM '{CORE}'
      WHERE {CORE_FILTER}
        AND period LIKE '2026%'
        AND company_industry = 'Hospitals and Health Care'
        AND {text_filter()}
      GROUP BY 1
      ORDER BY n DESC
      LIMIT 40
    """).df()
    titles.to_csv(TABLES / "S26_dd1_hospital_top_titles.csv", index=False)
    print("\nTop 20 hospital-industry SWE titles, 2026:")
    print(titles.head(20).to_string())

    # 2b. Companies within Hospitals — is this really nurse postings or is this
    # health-tech firms misclassified into Hospitals?
    companies = con.execute(f"""
      SELECT company_name_canonical AS company,
             COUNT(*) AS n_swe,
             SUM(CASE WHEN regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}') THEN 1 ELSE 0 END) AS n_ai,
             AVG(CASE WHEN regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}') THEN 1.0 ELSE 0.0 END) AS ai_rate
      FROM '{CORE}'
      WHERE {CORE_FILTER}
        AND period LIKE '2026%'
        AND company_industry = 'Hospitals and Health Care'
        AND {text_filter()}
      GROUP BY 1
      ORDER BY n_swe DESC
      LIMIT 30
    """).df()
    companies.to_csv(TABLES / "S26_dd1_hospital_top_companies.csv", index=False)
    print("\nTop 20 hospital-industry SWE-posting companies, 2026:")
    print(companies.head(20).to_string())

    # 3. Spot-check 30 random hospital SWE postings that match the AI-vocab pattern.
    sample = con.execute(f"""
      SELECT uid, title, company_name_canonical AS company, description
      FROM '{CORE}'
      WHERE {CORE_FILTER}
        AND period LIKE '2026%'
        AND company_industry = 'Hospitals and Health Care'
        AND {text_filter()}
        AND regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}')
      ORDER BY hash(uid)
      LIMIT 30
    """).df()

    # For each match, find ~200 chars around the first AI-vocab hit.
    def excerpt(desc):
        m = re.search(AI_VOCAB_PATTERN, desc or "")
        if not m:
            return ""
        s = max(0, m.start() - 120)
        e = min(len(desc), m.end() + 120)
        return re.sub(r"\s+", " ", desc[s:e]).strip()

    sample["match_excerpt"] = sample["description"].apply(excerpt)
    sample["first_match_word"] = sample["description"].apply(
        lambda d: (re.search(AI_VOCAB_PATTERN, d or "") or re.match("", "")).group(0) if re.search(AI_VOCAB_PATTERN, d or "") else None
    )
    out = sample[["uid", "company", "title", "first_match_word", "match_excerpt"]].copy()
    out.to_csv(TABLES / "S26_dd1_hospital_spotcheck30.csv", index=False)
    print("\n--- Hospital spot-check (30 random AI-matched postings) ---")
    for _, r in out.iterrows():
        print(f"[{r.first_match_word}] {r.company} -- {r.title}")
        print(f"    {r.match_excerpt[:260]}")
        print()

    # 4. Is the result driven by a few health-tech firms? Recompute hospitals
    # rate after excluding the top-5 AI-heavy hospital companies.
    top5_ai_firms = companies.sort_values("n_ai", ascending=False).head(5)["company"].tolist()
    print(f"\nTop-5 AI-posting hospital firms: {top5_ai_firms}")
    names_sql = ",".join("'" + str(c).replace("'", "''") + "'" for c in top5_ai_firms if c is not None)
    if names_sql:
        reduced = con.execute(f"""
          SELECT COUNT(*) AS n,
                 SUM(CASE WHEN regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}') THEN 1 ELSE 0 END) AS n_ai
          FROM '{CORE}'
          WHERE {CORE_FILTER}
            AND period LIKE '2026%'
            AND company_industry = 'Hospitals and Health Care'
            AND company_name_canonical NOT IN ({names_sql})
            AND {text_filter()}
        """).df().iloc[0]
        rate_reduced = reduced["n_ai"] / reduced["n"] if reduced["n"] else float("nan")
        print(f"After dropping top-5 AI-heavy hospital firms:")
        print(f"  n = {int(reduced['n']):,}, AI rate = {rate_reduced*100:.2f}%")
    else:
        rate_reduced = float("nan")

    # For comparison: Software Development 2026 rate.
    swe_ind = con.execute(f"""
      SELECT COUNT(*) AS n,
             SUM(CASE WHEN regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}') THEN 1 ELSE 0 END) AS n_ai
      FROM '{CORE}'
      WHERE {CORE_FILTER} AND period LIKE '2026%' AND company_industry = 'Software Development' AND {text_filter()}
    """).df().iloc[0]
    swe_rate = swe_ind["n_ai"] / swe_ind["n"]
    swe_lo, swe_hi = wilson_ci(swe_ind["n_ai"], swe_ind["n"])
    print(f"Software Development 2026: n={int(swe_ind['n']):,} AI={swe_rate*100:.2f}% [{swe_lo*100:.1f}-{swe_hi*100:.1f}]")

    return {
        "hospitals_reduced_rate_noop5": float(rate_reduced),
        "top5_hospital_firms": top5_ai_firms,
    }


# ---------------------------------------------------------------------------
# DD2 — Financial Services vs SWE under three regexes
# ---------------------------------------------------------------------------

def deepdive_2(con):
    print("\n==== DD2 — FS vs SWE under three regexes ====")
    rows = []
    for name, pat in REGEX_VARIANTS.items():
        for ind_label, ind_name in [("FS", "Financial Services"),
                                    ("SWE", "Software Development")]:
            res = con.execute(f"""
              SELECT COUNT(*) AS n,
                     SUM(CASE WHEN regexp_matches(description, '{pat}') THEN 1 ELSE 0 END) AS n_ai
              FROM '{CORE}'
              WHERE {CORE_FILTER}
                AND period LIKE '2026%'
                AND company_industry = '{ind_name}'
            """).df().iloc[0]
            p = res["n_ai"] / res["n"] if res["n"] else float("nan")
            lo, hi = wilson_ci(res["n_ai"], res["n"])
            rows.append({
                "regex": name,
                "industry": ind_label,
                "n": int(res["n"]),
                "n_ai": int(res["n_ai"]),
                "rate": p,
                "ci_lo": lo,
                "ci_hi": hi,
            })
    summary = pd.DataFrame(rows)
    # Compute deltas.
    deltas = []
    for name in REGEX_VARIANTS:
        fs = summary[(summary.regex == name) & (summary.industry == "FS")].iloc[0]
        swe = summary[(summary.regex == name) & (summary.industry == "SWE")].iloc[0]
        delta_pp = (fs["rate"] - swe["rate"]) * 100
        overlap = not (fs["ci_hi"] < swe["ci_lo"] or swe["ci_hi"] < fs["ci_lo"])
        deltas.append({
            "regex": name,
            "fs_rate_pct": fs["rate"] * 100,
            "fs_ci": f"{fs['ci_lo']*100:.1f}-{fs['ci_hi']*100:.1f}",
            "swe_rate_pct": swe["rate"] * 100,
            "swe_ci": f"{swe['ci_lo']*100:.1f}-{swe['ci_hi']*100:.1f}",
            "fs_minus_swe_pp": delta_pp,
            "ci_overlap": overlap,
        })
    dtab = pd.DataFrame(deltas)
    dtab.to_csv(TABLES / "S26_dd2_regex_comparison.csv", index=False)
    summary.to_csv(TABLES / "S26_dd2_regex_breakdown.csv", index=False)
    print(dtab.to_string())

    # Precision audit: sample 20 FS postings matched by canonical, 20 matched by strict,
    # 20 matched by tooling-only (but NOT matched by stricter patterns — to see the
    # precision gap contributed by the broad-only tokens).
    # For canonical, we show what the broad pattern catches that tooling doesn't.
    spot_rows = []
    for regex_label, pat in REGEX_VARIANTS.items():
        sample = con.execute(f"""
          SELECT uid, title, company_name_canonical AS company, description
          FROM '{CORE}'
          WHERE {CORE_FILTER}
            AND period LIKE '2026%'
            AND company_industry = 'Financial Services'
            AND regexp_matches(description, '{pat}')
          ORDER BY hash(uid || '{regex_label}')
          LIMIT 20
        """).df()
        for _, r in sample.iterrows():
            m = re.search(pat, r["description"] or "")
            if not m:
                continue
            s = max(0, m.start() - 120)
            e = min(len(r["description"]), m.end() + 120)
            excerpt = re.sub(r"\s+", " ", r["description"][s:e]).strip()
            spot_rows.append({
                "regex": regex_label,
                "uid": r["uid"],
                "company": r["company"],
                "title": r["title"],
                "first_match": m.group(0),
                "excerpt": excerpt,
            })
    spot = pd.DataFrame(spot_rows)
    spot.to_csv(TABLES / "S26_dd2_fs_spotcheck.csv", index=False)
    print(f"\nSpot-check rows saved: {len(spot)} (20 per regex). First 6 canonical excerpts:")
    if len(spot) == 0:
        print("  (no rows)")
        return dtab
    for _, r in spot[spot["regex"] == "canonical_broad"].head(6).iterrows():
        print(f"  [{r.first_match}] {r.company} -- {r.title}")
        print(f"    {r.excerpt[:200]}")

    # 3. FS vs SWE top n-grams — only in AI-matched rows, under canonical regex.
    def top_words(industry, limit=60):
        df = con.execute(f"""
          SELECT LOWER(description) AS d
          FROM '{CORE}'
          WHERE {CORE_FILTER}
            AND period LIKE '2026%'
            AND company_industry = '{industry}'
            AND {text_filter()}
            AND regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}')
        """).df()
        from collections import Counter
        c = Counter()
        STOP = set("""a an and the of in for to is are be or at as our we your you with by from on this that
                     will have has had not but if it its was were they them their he she his her can
                     able more most well us job role team work working experience required preferred
                     please apply using use support tools tool new all may any etc via one two skills""".split())
        for txt in df["d"]:
            words = re.findall(r"[a-z][a-z0-9+.#-]{2,}", txt or "")
            c.update(w for w in words if w not in STOP)
        return c.most_common(limit)

    fs_top = top_words("Financial Services", 80)
    swe_top = top_words("Software Development", 80)
    # Rank-delta: rank in FS minus rank in SWE.
    swe_rank = {w: i for i, (w, _) in enumerate(swe_top)}
    fs_rank = {w: i for i, (w, _) in enumerate(fs_top)}
    all_words = set(swe_rank) | set(fs_rank)
    rank_delta = []
    for w in all_words:
        sr = swe_rank.get(w, 500)
        fr = fs_rank.get(w, 500)
        rank_delta.append({
            "word": w,
            "fs_rank": fr,
            "swe_rank": sr,
            "fs_more_than_swe": sr - fr,  # positive = more FS-typical
        })
    rdf = pd.DataFrame(rank_delta).sort_values("fs_more_than_swe", ascending=False)
    rdf.to_csv(TABLES / "S26_dd2_fs_vs_swe_words.csv", index=False)
    print("\nTop 20 FS-typical words (vs SWE) in AI-matched postings:")
    print(rdf.head(20).to_string())
    print("\nTop 20 SWE-typical words (vs FS) in AI-matched postings:")
    print(rdf.tail(20).to_string())

    return dtab


# ---------------------------------------------------------------------------
# DD3 — Bay Area user-intensity premium
# ---------------------------------------------------------------------------

def deepdive_3(con):
    print("\n==== DD3 — Bay Area user-intensity +8pp premium ====")

    # 1. Absolute volumes behind the 8pp gap.
    # Non-builder-title SWE 2026 rows, hub vs rest.
    query = f"""
      WITH rows AS (
        SELECT metro_area,
               CASE WHEN metro_area IN ({", ".join("'" + m + "'" for m in TECH_HUBS)})
                    THEN 'hub' ELSE 'rest' END AS zone,
               regexp_matches(LOWER(title), '{BUILDER_TITLE_PATTERN}') AS is_builder,
               regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}') AS ai
        FROM '{CORE}'
        WHERE {CORE_FILTER} AND period LIKE '2026%' AND metro_area IS NOT NULL AND {text_filter()}
      )
      SELECT zone,
             COUNT(*) FILTER (WHERE NOT is_builder) AS n_general,
             SUM(CASE WHEN NOT is_builder AND ai THEN 1 ELSE 0 END) AS n_general_ai
      FROM rows
      GROUP BY 1
    """
    vol = con.execute(query).df()
    vol["ai_rate"] = vol["n_general_ai"] / vol["n_general"]
    vol.to_csv(TABLES / "S26_dd3_volume.csv", index=False)
    print(vol.to_string())

    # 2. Sample 10 Bay-Area and 10 non-Bay ordinary-SWE (non-builder) postings
    # matched by AI vocab.
    def sample(metro_clause, n, seed):
        return con.execute(f"""
          SELECT uid, title, company_name_canonical AS company, company_industry, description, metro_area
          FROM '{CORE}'
          WHERE {CORE_FILTER}
            AND period LIKE '2026%'
            AND metro_area IS NOT NULL
            AND {metro_clause}
            AND NOT regexp_matches(LOWER(title), '{BUILDER_TITLE_PATTERN}')
            AND {text_filter()}
            AND regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}')
          ORDER BY hash(uid || '{seed}')
          LIMIT {n}
        """).df()

    bay = sample("metro_area = 'San Francisco Bay Area'", 10, 101)
    rest = sample(
        "metro_area NOT IN (" + ",".join("'" + m + "'" for m in TECH_HUBS) + ")",
        10, 102
    )

    def excerpt(desc, pat=AI_VOCAB_PATTERN):
        m = re.search(pat, desc or "")
        if not m:
            return ""
        s = max(0, m.start() - 120)
        e = min(len(desc), m.end() + 140)
        return re.sub(r"\s+", " ", desc[s:e]).strip()

    for label, df in [("BAY", bay), ("REST", rest)]:
        df["excerpt"] = df["description"].apply(excerpt)
        df["label"] = label
        df["first_match"] = df["description"].apply(
            lambda d: (re.search(AI_VOCAB_PATTERN, d or "") or re.match("", "")).group(0)
            if re.search(AI_VOCAB_PATTERN, d or "") else None
        )
    combined = pd.concat([bay, rest])[
        ["label", "uid", "metro_area", "company", "company_industry",
         "title", "first_match", "excerpt"]
    ]
    combined.to_csv(TABLES / "S26_dd3_bay_vs_rest_samples.csv", index=False)
    print("\n--- 10 Bay-Area ordinary-SWE postings with AI vocab ---")
    for _, r in combined[combined.label == "BAY"].iterrows():
        print(f"[{r.first_match}] {r.company} ({r.company_industry}) -- {r.title}")
        print(f"    {r.excerpt[:260]}")
        print()
    print("\n--- 10 Non-hub ordinary-SWE postings with AI vocab ---")
    for _, r in combined[combined.label == "REST"].iterrows():
        print(f"[{r.first_match}] {r.company} ({r.company_industry}) -- {r.title}")
        print(f"    {r.excerpt[:260]}")
        print()

    # 3. Which AI-vocab tokens drive the Bay premium vs rest?
    def mentions(metro_clause):
        return con.execute(f"""
          SELECT LOWER(description) AS d
          FROM '{CORE}'
          WHERE {CORE_FILTER}
            AND period LIKE '2026%'
            AND metro_area IS NOT NULL
            AND {metro_clause}
            AND NOT regexp_matches(LOWER(title), '{BUILDER_TITLE_PATTERN}')
            AND {text_filter()}
            AND regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}')
        """).df()

    from collections import Counter
    tokens = ["llm", "gpt", "chatgpt", "claude", "copilot", "openai", "anthropic",
              "gemini", "bard", "mistral", "llama", "large language model",
              "generative ai", "genai", "gen ai", "foundation model",
              "ai agent", "agentic", "ai-powered", "ai tooling",
              "rag", "retrieval augmented", "vector database",
              "prompt engineering", "mlops", "llmops", "cursor ide", "github copilot"]
    counts_rows = []
    for zone, clause in [
        ("bay", "metro_area = 'San Francisco Bay Area'"),
        ("rest", "metro_area NOT IN (" + ",".join("'" + m + "'" for m in TECH_HUBS) + ")"),
    ]:
        df = mentions(clause)
        n = len(df)
        c = Counter()
        for txt in df["d"]:
            for tok in tokens:
                if re.search(r"\b" + re.escape(tok) + r"\b", txt or ""):
                    c[tok] += 1
        for tok in tokens:
            counts_rows.append({"zone": zone, "token": tok, "n_posts": n, "n_hit": c[tok], "rate": c[tok] / n if n else 0})
    tok_df = pd.DataFrame(counts_rows)
    wide = tok_df.pivot(index="token", columns="zone", values="rate").reset_index()
    wide["delta_pp"] = (wide["bay"] - wide["rest"]) * 100
    wide = wide.sort_values("delta_pp", ascending=False)
    wide.to_csv(TABLES / "S26_dd3_token_gap.csv", index=False)
    print("\nToken-level Bay vs Rest gap (rate per non-builder AI-matched posting):")
    print(wide.to_string())


def main():
    con = duckdb.connect()
    dd1 = deepdive_1(con)
    dd2 = deepdive_2(con)
    dd3 = deepdive_3(con)
    print("\nDone.")


if __name__ == "__main__":
    main()
