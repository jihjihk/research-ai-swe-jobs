"""Finance-industry SWE LinkedIn exemplars with high AI-requirement density.

Inputs:
  - data/unified_core.parquet (strict analysis-ready subset)
  - exploration/artifacts/shared/validated_mgmt_patterns.json (for ai_strict pattern)

Output:
  - exploration/tables/journalist/finance_exemplars.csv

Filters:
  - source_platform == 'linkedin'
  - is_swe == True
  - is_english == True
  - date_flag == 'ok'
  - source == 'scraped'
  - llm_extraction_coverage == 'labeled' (so description_core_llm is populated)

Ranking:
  - Primary: AI-strict mention density per 1K chars of description_core_llm
  - Secondary tiebreak: raw mention count
  - One exemplar per company; de-duplicate by description_core_llm prefix to
    avoid near-identical reposts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
UNIFIED_CORE = ROOT / "data" / "unified_core.parquet"
VALIDATED = ROOT / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json"
OUT_CSV = ROOT / "exploration" / "tables" / "journalist" / "finance_exemplars.csv"

# Finance target cohort. Matched with LIKE on LOWER(company_name_canonical).
# Each tuple: (display_name, [lowercase substrings]).
FINANCE_COHORT = [
    ("JPMorgan Chase", ["jpmorgan", "jp morgan", "chase & co"]),
    ("Citigroup", ["citigroup", "citibank", "citi inc"]),
    ("Wells Fargo", ["wells fargo"]),
    ("Capital One", ["capital one"]),
    ("Visa", ["visa"]),
    ("Mastercard", ["mastercard"]),
    ("American Express", ["american express", "amex"]),
    ("Bank of America", ["bank of america", "bofa"]),
    ("Goldman Sachs", ["goldman"]),
    ("Morgan Stanley", ["morgan stanley"]),
    ("BlackRock", ["blackrock"]),
    ("GEICO", ["geico"]),
    ("Solera", ["solera"]),
    ("Fidelity", ["fidelity investments"]),
    ("Vanguard", ["vanguard"]),
    ("USAA", ["usaa"]),
    ("PNC", ["pnc"]),
    ("U.S. Bank", ["u.s. bank", "us bank"]),
    ("Charles Schwab", ["schwab"]),
    ("Progressive", ["progressive"]),
]


def _load_ai_strict_pattern() -> str:
    """Return the validated ai_strict regex (0.86 overall precision).

    Falls back to the task-supplied simpler regex if the artifact is missing.
    """
    if VALIDATED.exists():
        with VALIDATED.open() as f:
            p = json.load(f)
        return p["ai_strict"]["pattern"]
    # Fallback pattern from the task spec.
    return (
        r"\b(copilot|cursor|claude|chatgpt|gpt-?\d+|llm|rag|langchain|"
        r"llamaindex|openai|pinecone|vector databas(?:e|es)|fine[- ]tun(?:e|ed|ing)|"
        r"prompt engineering)\b"
    )


def _build_finance_filter() -> tuple[str, list[str]]:
    """Build a SQL fragment like `(LOWER(c) LIKE ? OR LOWER(c) LIKE ? ...)`."""
    like_params: list[str] = []
    clauses: list[str] = []
    for _, subs in FINANCE_COHORT:
        for sub in subs:
            clauses.append("LOWER(company_name_canonical) LIKE ?")
            like_params.append(f"%{sub}%")
    return "(" + " OR ".join(clauses) + ")", like_params


def _canonical_display(name: str | None) -> str | None:
    """Map raw canonical name to a cleaner display label from FINANCE_COHORT."""
    if not name:
        return name
    low = name.lower()
    for display, subs in FINANCE_COHORT:
        if any(sub in low for sub in subs):
            return display
    return name


def load_candidates() -> pd.DataFrame:
    finance_clause, params = _build_finance_filter()
    query = f"""
        SELECT
            uid,
            company_name_canonical,
            title,
            date_posted,
            scrape_date,
            description_core_llm,
            length(description_core_llm) AS desc_len
        FROM read_parquet(?)
        WHERE source_platform = 'linkedin'
          AND is_swe = TRUE
          AND is_english = TRUE
          AND date_flag = 'ok'
          AND source = 'scraped'
          AND llm_extraction_coverage = 'labeled'
          AND description_core_llm IS NOT NULL
          AND length(description_core_llm) >= 400
          AND {finance_clause}
    """
    con = duckdb.connect()
    df = con.execute(query, [str(UNIFIED_CORE), *params]).fetchdf()
    return df


def score_ai_density(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    rx = re.compile(pattern, flags=re.IGNORECASE)
    texts = df["description_core_llm"].fillna("").to_list()
    mention_counts: list[int] = []
    hits_sample: list[str] = []
    for t in texts:
        hits = rx.findall(t)
        # findall on the validated pattern returns strings (single capture group).
        flat = [h.lower() if isinstance(h, str) else h[0].lower() for h in hits]
        mention_counts.append(len(flat))
        # Preserve order, drop duplicates.
        seen: set[str] = set()
        uniq: list[str] = []
        for x in flat:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        hits_sample.append(", ".join(uniq[:8]))
    out = df.copy()
    out["ai_strict_count"] = mention_counts
    out["ai_strict_hits"] = hits_sample
    # Density per 1K chars; guard against zero-length.
    out["ai_strict_density_per_1k"] = (
        out["ai_strict_count"] / out["desc_len"].clip(lower=1) * 1000.0
    )
    return out


def pick_exemplars(scored: pd.DataFrame, n: int = 7) -> pd.DataFrame:
    # Require at least 3 distinct AI-strict mentions so excerpts are substantive.
    cand = scored.query("ai_strict_count >= 3").copy()
    cand["company_display"] = cand["company_name_canonical"].map(_canonical_display)

    # Within-company: keep the posting with the highest density, tiebreak on count.
    cand = cand.sort_values(
        ["company_display", "ai_strict_density_per_1k", "ai_strict_count"],
        ascending=[True, False, False],
    )
    per_company = cand.groupby("company_display", as_index=False).head(1).copy()

    # De-duplicate near-identical reposts (same 200-char prefix).
    per_company["desc_prefix"] = per_company["description_core_llm"].str.slice(0, 200)
    per_company = per_company.drop_duplicates(subset=["desc_prefix"])

    # Rank across companies by density, pick top-n.
    per_company = per_company.sort_values(
        ["ai_strict_density_per_1k", "ai_strict_count"], ascending=False
    ).head(n)

    # Assemble the excerpt.
    per_company["excerpt_300c"] = per_company["description_core_llm"].apply(
        lambda t: (t[:300].rstrip() + "...") if isinstance(t, str) and len(t) > 300 else t
    )
    return per_company


def main() -> None:
    pattern = _load_ai_strict_pattern()
    print(f"Using AI-strict pattern (len={len(pattern)} chars):")
    print(f"  {pattern[:140]}{'...' if len(pattern) > 140 else ''}")

    df = load_candidates()
    print(f"Candidate finance SWE postings (labeled): {len(df):,}")
    print(f"Distinct companies in candidate pool: {df['company_name_canonical'].nunique()}")

    scored = score_ai_density(df, pattern)
    hits = scored.query("ai_strict_count >= 1")
    print(
        f"Postings with >=1 AI-strict hit: {len(hits):,} "
        f"({100.0 * len(hits) / max(len(scored), 1):.1f}%)"
    )

    exemplars = pick_exemplars(scored, n=7)
    print(f"Exemplars selected: {len(exemplars)}")

    cols = [
        "uid",
        "company_name_canonical",
        "company_display",
        "title",
        "date_posted",
        "scrape_date",
        "desc_len",
        "ai_strict_count",
        "ai_strict_density_per_1k",
        "ai_strict_hits",
        "excerpt_300c",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    exemplars[cols].to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

    # Preview top 3.
    pd.set_option("display.max_colwidth", 320)
    print("\n=== Top 3 exemplars (preview) ===")
    for _, r in exemplars.head(3).iterrows():
        print(
            f"\n[{r['company_display']}] {r['title']}  "
            f"uid={r['uid']}  "
            f"count={int(r['ai_strict_count'])}  "
            f"density/1k={r['ai_strict_density_per_1k']:.3f}  "
            f"hits={r['ai_strict_hits']}"
        )
        print(f"  date_posted={r['date_posted']}  scrape_date={r['scrape_date']}")
        print(f"  excerpt: {r['excerpt_300c']}")


if __name__ == "__main__":
    main()
