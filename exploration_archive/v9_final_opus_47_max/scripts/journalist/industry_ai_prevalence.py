"""AI-strict prevalence by industry within 2026 scraped LinkedIn SWE corpus.

Inputs:
  - data/unified_core.parquet
  - exploration/artifacts/shared/validated_mgmt_patterns.json (uses ai_strict_v1_rebuilt)

Outputs:
  - exploration/tables/journalist/industry_ai_prevalence.csv
  - exploration/tables/journalist/industry_ai_top_companies.csv

Filter for this task (2026 scraped SWE only):
    source_platform = 'linkedin'
    AND is_swe
    AND is_english
    AND date_flag = 'ok'
    AND source = 'scraped'
    AND llm_extraction_coverage = 'labeled'      (text-sensitive: needs description_core_llm)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb

ROOT = Path("/home/jihgaboot/gabor/job-research")
PARQUET = ROOT / "data" / "unified_core.parquet"
PATTERNS_JSON = ROOT / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json"
OUT_DIR = ROOT / "exploration" / "tables" / "journalist"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PREVALENCE = OUT_DIR / "industry_ai_prevalence.csv"
OUT_COMPANIES = OUT_DIR / "industry_ai_top_companies.csv"


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_ai_strict_pattern() -> tuple[str, str]:
    """Return (pattern_name, regex_pattern). Prefer v1_rebuilt (0.96 precision)."""
    try:
        obj = json.loads(PATTERNS_JSON.read_text())
        v1 = obj.get("v1_rebuilt_patterns", {}).get("ai_strict_v1_rebuilt")
        if v1 and "pattern" in v1:
            return "ai_strict_v1_rebuilt", v1["pattern"]
        top = obj.get("ai_strict")
        if top and "pattern" in top:
            return "ai_strict", top["pattern"]
    except FileNotFoundError:
        pass
    # Hard fallback (same top-level ai_strict pattern, precision 0.86)
    return (
        "ai_strict_fallback",
        r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|"
        r"llamaindex|langchain|prompt engineering|fine[- ]tun(?:e|ed|ing)|rag|"
        r"vector databas(?:e|es)|pinecone|huggingface|hugging face)\b",
    )


def main() -> None:
    pattern_name, pattern = load_ai_strict_pattern()
    print(f"AI-strict pattern: {pattern_name}")
    print(f"Pattern regex: {pattern}")
    print()

    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE VIEW u AS SELECT * FROM read_parquet('{PARQUET.as_posix()}')")

    # --- Base filter: 2026 scraped LinkedIn SWE rows with labeled LLM text.
    base_where = """
        source_platform = 'linkedin'
        AND is_swe
        AND is_english
        AND date_flag = 'ok'
        AND source = 'scraped'
    """

    # 1) Industry coverage in 2026 scraped SWE (before requiring labeled).
    coverage_sql = f"""
        SELECT
          COUNT(*) AS total_rows,
          SUM(CASE WHEN company_industry IS NOT NULL AND company_industry <> '' THEN 1 ELSE 0 END)
            AS with_industry,
          ROUND(100.0 * SUM(CASE WHEN company_industry IS NOT NULL AND company_industry <> '' THEN 1 ELSE 0 END)
                / NULLIF(COUNT(*), 0), 2) AS pct_with_industry
        FROM u
        WHERE {base_where}
    """
    cov_df = con.execute(coverage_sql).df()
    print("=== Industry coverage (2026 scraped LinkedIn SWE, all labeling states) ===")
    print(cov_df.to_string(index=False))
    print()

    # Also coverage within labeled (the denominator we will use for all downstream).
    coverage_labeled_sql = f"""
        SELECT
          COUNT(*) AS total_labeled_rows,
          SUM(CASE WHEN company_industry IS NOT NULL AND company_industry <> '' THEN 1 ELSE 0 END)
            AS with_industry,
          ROUND(100.0 * SUM(CASE WHEN company_industry IS NOT NULL AND company_industry <> '' THEN 1 ELSE 0 END)
                / NULLIF(COUNT(*), 0), 2) AS pct_with_industry
        FROM u
        WHERE {base_where} AND llm_extraction_coverage = 'labeled'
    """
    cov_labeled_df = con.execute(coverage_labeled_sql).df()
    print("=== Industry coverage (2026 scraped LinkedIn SWE, labeled rows only) ===")
    print(cov_labeled_df.to_string(index=False))
    print()

    # 2) Overall AI-strict prevalence in 2026 scraped SWE (labeled rows).
    overall_sql = f"""
        SELECT
          COUNT(*) AS n_labeled,
          SUM(CASE WHEN regexp_matches(lower(description_core_llm), '{pattern}') THEN 1 ELSE 0 END) AS n_ai_strict
        FROM u
        WHERE {base_where}
          AND llm_extraction_coverage = 'labeled'
          AND description_core_llm IS NOT NULL
    """
    overall_df = con.execute(overall_sql).df()
    n_all = int(overall_df.iloc[0]["n_labeled"])
    k_all = int(overall_df.iloc[0]["n_ai_strict"])
    p_all = k_all / n_all if n_all else float("nan")
    lo_all, hi_all = wilson_ci(k_all, n_all)
    print("=== Overall AI-strict prevalence (2026 scraped LinkedIn SWE, labeled) ===")
    print(f"n_postings = {n_all}, n_ai_strict = {k_all}, pct = {100*p_all:.2f}%, "
          f"95% Wilson CI = [{100*lo_all:.2f}%, {100*hi_all:.2f}%]")
    print()

    # 3) AI-strict prevalence by industry (labeled rows, non-null industry).
    by_industry_sql = f"""
        SELECT
          company_industry,
          COUNT(*) AS n_postings,
          SUM(CASE WHEN regexp_matches(lower(description_core_llm), '{pattern}') THEN 1 ELSE 0 END) AS n_ai_strict
        FROM u
        WHERE {base_where}
          AND llm_extraction_coverage = 'labeled'
          AND description_core_llm IS NOT NULL
          AND company_industry IS NOT NULL
          AND company_industry <> ''
        GROUP BY company_industry
    """
    ind_df = con.execute(by_industry_sql).df()
    ind_df["pct_ai_strict"] = ind_df["n_ai_strict"] / ind_df["n_postings"]
    cis = [wilson_ci(int(k), int(n)) for k, n in zip(ind_df["n_ai_strict"], ind_df["n_postings"])]
    ind_df["ci95_low"] = [c[0] for c in cis]
    ind_df["ci95_high"] = [c[1] for c in cis]
    ind_df["meets_min_n_100"] = ind_df["n_postings"] >= 100
    ind_df = ind_df.sort_values(
        by=["meets_min_n_100", "pct_ai_strict"], ascending=[False, False]
    ).reset_index(drop=True)

    # Persist full table (all industries, with flag), sorted so the top-n>=100 set comes first.
    ind_out = ind_df[
        [
            "company_industry",
            "n_postings",
            "n_ai_strict",
            "pct_ai_strict",
            "ci95_low",
            "ci95_high",
            "meets_min_n_100",
        ]
    ].copy()
    ind_out["pct_ai_strict"] = (100 * ind_out["pct_ai_strict"]).round(2)
    ind_out["ci95_low"] = (100 * ind_out["ci95_low"]).round(2)
    ind_out["ci95_high"] = (100 * ind_out["ci95_high"]).round(2)
    ind_out.to_csv(OUT_PREVALENCE, index=False)

    # Log the top-10 of the n>=100 partition.
    top10 = ind_out[ind_out["meets_min_n_100"]].head(10)
    print("=== Top-10 industries by AI-strict prevalence (n_postings >= 100) ===")
    print(top10.to_string(index=False))
    print()
    print(f"Full industry table (with meets_min_n_100 flag) -> {OUT_PREVALENCE}")
    small = ind_out[~ind_out["meets_min_n_100"]]
    print(f"Small-n industries (n<100) in output with flag=False: {len(small)}")
    print()

    # Where does Financial Services sit vs. overall?
    fs_mask = ind_out["company_industry"].str.lower().str.contains("financial services", na=False)
    if fs_mask.any():
        print("=== Financial Services row(s) ===")
        print(ind_out[fs_mask].to_string(index=False))
        print()

    # 4) Top-10 companies (pct_ai_strict >= 30%, n>=10) inside three target industries.
    target_industries = [
        "Financial Services",
        "Software Development",
        "IT Services and IT Consulting",
    ]
    # Discover actual industry labels so we don't miss spelling drift.
    present_labels = set(ind_out["company_industry"].astype(str).tolist())
    used_labels = []
    for label in target_industries:
        if label in present_labels:
            used_labels.append(label)
        else:
            # Try contains-match.
            matches = [p for p in present_labels if label.lower() in p.lower()]
            used_labels.extend(matches)
    used_labels = sorted(set(used_labels))
    print(f"=== Target industries matched in data: {used_labels} ===")

    in_list = ", ".join(f"'{s.replace(chr(39), chr(39)+chr(39))}'" for s in used_labels)
    company_sql = f"""
        SELECT
          company_industry,
          company_name_canonical,
          COUNT(*) AS n_postings,
          SUM(CASE WHEN regexp_matches(lower(description_core_llm), '{pattern}') THEN 1 ELSE 0 END) AS n_ai_strict
        FROM u
        WHERE {base_where}
          AND llm_extraction_coverage = 'labeled'
          AND description_core_llm IS NOT NULL
          AND company_industry IN ({in_list})
          AND company_name_canonical IS NOT NULL
          AND company_name_canonical <> ''
        GROUP BY company_industry, company_name_canonical
    """
    comp_df = con.execute(company_sql).df()
    comp_df["pct_ai_strict"] = comp_df["n_ai_strict"] / comp_df["n_postings"].clip(lower=1)
    # Keep companies with rate >= 30% and at least 10 postings (so the rate is meaningful).
    filt = (comp_df["pct_ai_strict"] >= 0.30) & (comp_df["n_postings"] >= 10)
    comp_filt = comp_df[filt].copy()
    cis_c = [
        wilson_ci(int(k), int(n)) for k, n in zip(comp_filt["n_ai_strict"], comp_filt["n_postings"])
    ]
    comp_filt["ci95_low"] = [c[0] for c in cis_c]
    comp_filt["ci95_high"] = [c[1] for c in cis_c]
    comp_filt["pct_ai_strict"] = (100 * comp_filt["pct_ai_strict"]).round(2)
    comp_filt["ci95_low"] = (100 * comp_filt["ci95_low"]).round(2)
    comp_filt["ci95_high"] = (100 * comp_filt["ci95_high"]).round(2)
    comp_filt = comp_filt.sort_values(
        by=["company_industry", "pct_ai_strict", "n_postings"],
        ascending=[True, False, False],
    )
    # Top-10 per industry
    out_rows = []
    for ind, grp in comp_filt.groupby("company_industry", sort=False):
        out_rows.append(grp.head(10))
    import pandas as pd

    if out_rows:
        comp_out = pd.concat(out_rows, ignore_index=True)
    else:
        comp_out = comp_filt.head(0)
    comp_out = comp_out[
        [
            "company_industry",
            "company_name_canonical",
            "n_postings",
            "n_ai_strict",
            "pct_ai_strict",
            "ci95_low",
            "ci95_high",
        ]
    ]
    comp_out.to_csv(OUT_COMPANIES, index=False)
    print()
    print("=== Top-10 companies per target industry (pct_ai_strict >= 30%, n>=10) ===")
    print(comp_out.to_string(index=False))
    print()
    print(f"Companies CSV -> {OUT_COMPANIES}")

    # Preview first 10 lines of each CSV.
    print()
    print("=== Preview: industry_ai_prevalence.csv (first 10 lines) ===")
    with OUT_PREVALENCE.open() as f:
        for _ in range(10):
            line = f.readline()
            if not line:
                break
            print(line.rstrip())
    print()
    print("=== Preview: industry_ai_top_companies.csv (first 10 lines) ===")
    with OUT_COMPANIES.open() as f:
        for _ in range(10):
            line = f.readline()
            if not line:
                break
            print(line.rstrip())


if __name__ == "__main__":
    main()
