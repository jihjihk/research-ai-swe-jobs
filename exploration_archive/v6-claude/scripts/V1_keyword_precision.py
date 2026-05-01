"""V1.2 — Keyword precision sampling for scope_density, management strict,
agents_framework, and 'agentic'.

Samples rows that match each pattern, prints snippets with ~200 chars of
context around each match, and tallies precision. Human review happens in
the output (the script does not auto-classify precision).
"""

from __future__ import annotations

import random
import re

import duckdb
import pandas as pd

UNI = "/home/jihgaboot/gabor/job-research/data/unified.parquet"
META = "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_cleaned_text.parquet"
OUT = "/home/jihgaboot/gabor/job-research/exploration/tables/V1"

random.seed(20260415)


def window(text: str, start: int, end: int, ctx: int = 120) -> str:
    a = max(0, start - ctx)
    b = min(len(text), end + ctx)
    s = text[a:b].replace("\n", " ").replace("  ", " ")
    return f"...{s}..."


def sample_matches(df: pd.DataFrame, pattern: re.Pattern, n: int, period_label: str) -> list[dict]:
    """Return up to n rows that contain a match for pattern; include the
    matched-term context window."""
    hits = []
    for _, r in df.sample(frac=1, random_state=42).iterrows():
        txt = r["description"] or ""
        m = pattern.search(txt.lower())
        if m:
            hits.append(
                {
                    "uid": r["uid"],
                    "period": period_label,
                    "match": m.group(0),
                    "context": window(txt, m.start(), m.end()),
                }
            )
        if len(hits) >= n:
            break
    return hits


def load_pool():
    con = duckdb.connect()
    q = f"""
    SELECT uid, source, period, description
    FROM '{UNI}'
    WHERE is_swe = true
      AND source_platform = 'linkedin'
      AND is_english = true
      AND date_flag = 'ok'
      AND description IS NOT NULL
    """
    df = con.execute(q).fetchdf()
    df["period_bucket"] = df["source"].apply(lambda s: "2026" if s == "scraped" else "2024")
    return df


def main() -> None:
    df = load_pool()
    print(f"pool rows: {len(df):,}")
    d24 = df[df["period_bucket"] == "2024"]
    d26 = df[df["period_bucket"] == "2026"]

    # -------------------------------------------------------------------
    # 1) agents_framework: shared regex (\s|^)(agent|agents|agentic)(\s|$)
    #    sample 30 from each period
    # -------------------------------------------------------------------
    af_pat = re.compile(r"(?:^|\s)(agent|agents|agentic)(?:$|\s)", re.IGNORECASE)
    af_2024 = sample_matches(d24, af_pat, 30, "2024")
    af_2026 = sample_matches(d26, af_pat, 30, "2026")

    # -------------------------------------------------------------------
    # 2) Strict management pattern
    #    manage/mentor/coach/hire/direct reports/performance review/headcount
    # -------------------------------------------------------------------
    mgmt_pat = re.compile(
        r"\b(manag(?:e|es|ed|ing)|mentor(?:s|ed|ing)?|coach(?:es|ed|ing)?|"
        r"hir(?:e|es|ed|ing)|direct reports?|performance reviews?|headcount)\b",
        re.IGNORECASE,
    )
    mg_2024 = sample_matches(d24, mgmt_pat, 25, "2024")
    mg_2026 = sample_matches(d26, mgmt_pat, 25, "2026")

    # -------------------------------------------------------------------
    # 3) scope_density top-3: ownership, end-to-end, cross-functional
    # -------------------------------------------------------------------
    scope_pats = {
        "ownership": re.compile(r"\b(ownership|own(?:s|ed|ing)?)\b", re.IGNORECASE),
        "end_to_end": re.compile(r"\bend[\s\-]*to[\s\-]*end\b", re.IGNORECASE),
        "cross_functional": re.compile(r"\bcross[\s\-]*functional\b", re.IGNORECASE),
    }
    scope_samples = {}
    for name, pat in scope_pats.items():
        samples = sample_matches(d24, pat, 25, "2024") + sample_matches(d26, pat, 25, "2026")
        scope_samples[name] = samples

    # -------------------------------------------------------------------
    # 4) "agentic" term growth — count distinct companies + spot-check
    # -------------------------------------------------------------------
    agentic_pat = re.compile(r"\bagentic\b", re.IGNORECASE)
    # Count matches per period and distinct companies in 2026
    con = duckdb.connect()
    q = f"""
    SELECT period,
           COUNT(*) FILTER (WHERE regexp_matches(lower(description), 'agentic')) n_match,
           COUNT(DISTINCT company_name_canonical) FILTER (
             WHERE regexp_matches(lower(description), 'agentic')) n_distinct_companies,
           COUNT(*) n_total
    FROM '{UNI}'
    WHERE is_swe = true
      AND source_platform = 'linkedin'
      AND is_english = true
      AND date_flag = 'ok'
    GROUP BY period
    ORDER BY period
    """
    agentic_counts = con.execute(q).fetchdf()
    print()
    print("--- 'agentic' term counts per period ---")
    print(agentic_counts)

    agentic_2026 = sample_matches(d26, agentic_pat, 20, "2026")

    # -------------------------------------------------------------------
    # Dump all samples to files for manual review
    # -------------------------------------------------------------------
    pd.DataFrame(af_2024 + af_2026).to_csv(
        f"{OUT}/V1_agents_framework_samples.csv", index=False
    )
    pd.DataFrame(mg_2024 + mg_2026).to_csv(
        f"{OUT}/V1_mgmt_strict_samples.csv", index=False
    )
    for name, samples in scope_samples.items():
        pd.DataFrame(samples).to_csv(
            f"{OUT}/V1_scope_{name}_samples.csv", index=False
        )
    pd.DataFrame(agentic_2026).to_csv(f"{OUT}/V1_agentic_samples.csv", index=False)

    # Print snippet dumps
    def dump(label: str, samples: list[dict]) -> None:
        print(f"\n===== {label} ({len(samples)}) =====")
        for i, s in enumerate(samples, 1):
            print(f"[{i}] {s['period']} match='{s['match'].strip()}': {s['context'][:300]}")

    dump("agents_framework 2024", af_2024)
    dump("agents_framework 2026", af_2026)
    dump("mgmt_strict 2024", mg_2024)
    dump("mgmt_strict 2026", mg_2026)
    for name, samples in scope_samples.items():
        dump(f"scope_{name}", samples)
    dump("agentic 2026", agentic_2026)


if __name__ == "__main__":
    main()
