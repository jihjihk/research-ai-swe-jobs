"""Forward-Deployed Engineer (FDE) archetype prevalence check.

Third-party trackers (Flex.ai, Indeed, bloomberry, LangChain state-of-agent-
engineering) reported FDE postings grew ~800% YoY in 2025, concentrated at
Anthropic, OpenAI, Palantir, Cohere, Salesforce.

This script tests whether that archetype is visible in our 2024->2026 SWE
posting corpus, using DuckDB over data/unified_core.parquet. Filter:
  source_platform='linkedin' AND is_swe AND is_english AND date_flag='ok'

Outputs:
  exploration/tables/journalist/fde_counts.csv
  exploration/tables/journalist/fde_companies.csv
  exploration/tables/journalist/fde_exemplars.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

REPO = Path("/home/jihgaboot/gabor/job-research")
CORE = REPO / "data" / "unified_core.parquet"
OUT_DIR = REPO / "exploration" / "tables" / "journalist"
PATTERNS = REPO / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json"

BASE_FILTER = (
    "source_platform = 'linkedin' AND is_swe AND is_english AND date_flag = 'ok'"
)

# Title-level FDE matcher. Case-insensitive. "fde" uses word boundaries to
# avoid false matches. The forward-deployed engineer / forward deployed AI
# engineer phrases are strict substrings of the base "forward deployed" hits,
# so the single OR below is sufficient.
FDE_TITLE = (
    "(lower(title) LIKE '%forward deployed%' "
    "OR lower(title) LIKE '%forward-deployed%' "
    "OR regexp_matches(lower(title), '\\bfde\\b'))"
)

# 2024 baseline = arshkon + asaniczka (both LinkedIn, historical snapshots).
# 2026 current window = scraped (LinkedIn).
PERIOD_2024 = "source IN ('kaggle_arshkon', 'kaggle_asaniczka')"
PERIOD_2026 = "source = 'scraped'"


def load_ai_strict_pattern() -> str:
    with PATTERNS.open() as f:
        data = json.load(f)
    # v1_rebuilt ai_strict (validated 0.96 precision).
    return data["v1_rebuilt_patterns"]["ai_strict_v1_rebuilt"]["pattern"]


def fetch_counts(con: duckdb.DuckDBPyConnection) -> list[tuple]:
    rows: list[tuple] = []
    for label, period_filter in (("2024", PERIOD_2024), ("2026", PERIOD_2026)):
        total = con.execute(
            f"SELECT COUNT(*) FROM '{CORE}' WHERE {BASE_FILTER} AND {period_filter}"
        ).fetchone()[0]
        fde = con.execute(
            f"SELECT COUNT(*) FROM '{CORE}' "
            f"WHERE {BASE_FILTER} AND {period_filter} AND {FDE_TITLE}"
        ).fetchone()[0]
        rows.append((label, total, fde, fde / total if total else 0.0))
    return rows


def fetch_top_companies(con: duckdb.DuckDBPyConnection) -> list[tuple]:
    q = f"""
      SELECT company_name_canonical, COUNT(*) AS n
      FROM '{CORE}'
      WHERE {BASE_FILTER} AND {PERIOD_2026} AND {FDE_TITLE}
      GROUP BY 1 ORDER BY n DESC, company_name_canonical
      LIMIT 10
    """
    return con.execute(q).fetchall()


def fetch_yoe_stats(con: duckdb.DuckDBPyConnection) -> dict:
    # Overall 2026 SWE LinkedIn median (labeled rows only).
    overall_q = f"""
      SELECT median(yoe_min_years_llm) AS med, COUNT(*) AS n
      FROM '{CORE}'
      WHERE {BASE_FILTER} AND {PERIOD_2026}
        AND llm_classification_coverage='labeled'
        AND yoe_min_years_llm IS NOT NULL
    """
    ov_med, ov_n = con.execute(overall_q).fetchone()
    # FDE-titled 2026 median.
    fde_q = f"""
      SELECT median(yoe_min_years_llm) AS med, COUNT(*) AS n
      FROM '{CORE}'
      WHERE {BASE_FILTER} AND {PERIOD_2026} AND {FDE_TITLE}
        AND llm_classification_coverage='labeled'
        AND yoe_min_years_llm IS NOT NULL
    """
    fde_med, fde_n = con.execute(fde_q).fetchone()
    return {
        "fde_median": fde_med,
        "fde_n_labeled": fde_n,
        "overall_median": ov_med,
        "overall_n_labeled": ov_n,
    }


def fetch_industries(con: duckdb.DuckDBPyConnection) -> list[tuple]:
    q = f"""
      SELECT COALESCE(company_industry, '(null)') AS industry, COUNT(*) AS n
      FROM '{CORE}'
      WHERE {BASE_FILTER} AND {PERIOD_2026} AND {FDE_TITLE}
      GROUP BY 1 ORDER BY n DESC
      LIMIT 5
    """
    return con.execute(q).fetchall()


def fetch_ai_density(con: duckdb.DuckDBPyConnection, ai_pattern: str) -> dict:
    # For binary presence we use raw description (boilerplate-insensitive per
    # schema guidance). The regex is passed as a bound parameter so DuckDB
    # sees the literal \b / \d / \s escape sequences rather than re-parsing
    # them through a SQL string literal.

    # AI-strict rate: overall 2026 SWE LinkedIn.
    overall_q = f"""
      SELECT
        AVG(CASE WHEN regexp_matches(lower(description), ?) THEN 1.0 ELSE 0.0 END) AS rate,
        COUNT(*) AS n
      FROM '{CORE}'
      WHERE {BASE_FILTER} AND {PERIOD_2026}
    """
    ov_rate, ov_n = con.execute(overall_q, [ai_pattern]).fetchone()

    # AI-strict rate in FDE-titled 2026.
    fde_q = f"""
      SELECT
        AVG(CASE WHEN regexp_matches(lower(description), ?) THEN 1.0 ELSE 0.0 END) AS rate,
        COUNT(*) AS n
      FROM '{CORE}'
      WHERE {BASE_FILTER} AND {PERIOD_2026} AND {FDE_TITLE}
    """
    fde_rate, fde_n = con.execute(fde_q, [ai_pattern]).fetchone()

    # Average distinct AI-tool mentions per posting. Each distinct sub-term
    # contributes at most once per posting. Uses description_core_llm where
    # labeled for cleaner denominators; falls back to description otherwise.
    ai_terms = [
        "copilot", "cursor", "claude", "chatgpt", "openai", "gpt",
        "gemini", "codex", "llamaindex", "langchain", "prompt engineering",
        "rag", "vector database", "pinecone", "huggingface", "hugging face",
        "mcp", "agentic", "llm", "fine-tun", "fine tun", "anthropic",
    ]
    def term_cnt_expr(col: str) -> str:
        parts = [
            f"(CASE WHEN lower({col}) LIKE '%{t}%' THEN 1 ELSE 0 END)"
            for t in ai_terms
        ]
        return " + ".join(parts)

    fde_mean_q = f"""
      SELECT AVG(term_cnt) AS mean_cnt, COUNT(*) AS n
      FROM (
        SELECT {term_cnt_expr('description')} AS term_cnt
        FROM '{CORE}'
        WHERE {BASE_FILTER} AND {PERIOD_2026} AND {FDE_TITLE}
      )
    """
    fde_mean, fde_mean_n = con.execute(fde_mean_q).fetchone()

    ov_mean_q = f"""
      SELECT AVG(term_cnt) AS mean_cnt, COUNT(*) AS n
      FROM (
        SELECT {term_cnt_expr('description')} AS term_cnt
        FROM '{CORE}'
        WHERE {BASE_FILTER} AND {PERIOD_2026}
      )
    """
    ov_mean, ov_mean_n = con.execute(ov_mean_q).fetchone()

    return {
        "fde_ai_strict_rate": fde_rate,
        "fde_ai_strict_n": fde_n,
        "overall_ai_strict_rate": ov_rate,
        "overall_ai_strict_n": ov_n,
        "fde_mean_ai_tool_mentions": fde_mean,
        "fde_mean_ai_n": fde_mean_n,
        "overall_mean_ai_tool_mentions": ov_mean,
        "overall_mean_ai_n": ov_mean_n,
    }


def fetch_exemplars(con: duckdb.DuckDBPyConnection, k: int = 5) -> list[tuple]:
    # Diverse across companies — pick one posting per distinct canonical
    # company among the top firms. Prefer postings with labeled description.
    q = f"""
      WITH fde AS (
        SELECT
          uid, company_name_canonical, title, scrape_date,
          COALESCE(description_core_llm, description) AS text_for_preview,
          ROW_NUMBER() OVER (
            PARTITION BY company_name_canonical
            ORDER BY CASE WHEN llm_extraction_coverage='labeled' THEN 0 ELSE 1 END,
                     scrape_date
          ) AS rn
        FROM '{CORE}'
        WHERE {BASE_FILTER} AND {PERIOD_2026} AND {FDE_TITLE}
      )
      SELECT uid, company_name_canonical, title, scrape_date,
             SUBSTRING(text_for_preview, 1, 300) AS preview
      FROM fde
      WHERE rn = 1
      ORDER BY company_name_canonical
      LIMIT {k}
    """
    return con.execute(q).fetchall()


def write_counts_csv(rows: list[tuple]) -> None:
    path = OUT_DIR / "fde_counts.csv"
    path.write_text(
        "period,n_postings_swe_linkedin,n_fde_title,fde_share\n"
        + "\n".join(f"{p},{n},{f},{s:.6f}" for p, n, f, s in rows)
        + "\n"
    )
    print(f"[write] {path}")


def write_companies_csv(rows: list[tuple]) -> None:
    path = OUT_DIR / "fde_companies.csv"
    path.write_text(
        "company_name_canonical,n_fde_2026\n"
        + "\n".join(f"\"{c}\",{n}" for c, n in rows)
        + "\n"
    )
    print(f"[write] {path}")


def _csv_escape(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\r", " ").replace("\n", " ").replace('"', '""')
    return f'"{s}"'


def write_exemplars_csv(rows: list[tuple]) -> None:
    path = OUT_DIR / "fde_exemplars.csv"
    header = "uid,company_name_canonical,title,scrape_date,preview_300c"
    lines = [header]
    for uid, comp, title, scrape, preview in rows:
        lines.append(
            ",".join(
                [
                    _csv_escape(uid),
                    _csv_escape(comp),
                    _csv_escape(title),
                    _csv_escape(str(scrape) if scrape else ""),
                    _csv_escape(preview or ""),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"[write] {path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    # Q1: title-match counts per period.
    counts = fetch_counts(con)
    write_counts_csv(counts)
    c2024 = next(r for r in counts if r[0] == "2024")
    c2026 = next(r for r in counts if r[0] == "2026")
    growth = (c2026[2] / c2024[2]) if c2024[2] else float("inf")
    share_2024 = c2024[3]
    share_2026 = c2026[3]
    share_ratio = (share_2026 / share_2024) if share_2024 else float("inf")

    print("\n=== Q1 FDE title-match counts ===")
    for period, n_total, n_fde, share in counts:
        print(f"  {period}: n_fde={n_fde} / n_swe_linkedin={n_total} "
              f"(share={share*100:.4f}%)")
    print(f"  Absolute growth 2024 -> 2026: x{growth:.1f}  "
          f"(share growth: x{share_ratio:.1f})")

    # Q2: top companies hiring FDE in 2026.
    companies = fetch_top_companies(con)
    write_companies_csv(companies)
    print("\n=== Q2 Top FDE companies (2026, scraped LinkedIn) ===")
    for comp, n in companies:
        print(f"  {comp}: {n}")

    # Q3: YOE profile.
    yoe = fetch_yoe_stats(con)
    print("\n=== Q3 YOE profile ===")
    print(f"  FDE 2026 median yoe_min_years_llm: {yoe['fde_median']} "
          f"(n={yoe['fde_n_labeled']} labeled)")
    print(f"  Overall 2026 SWE LinkedIn median: {yoe['overall_median']} "
          f"(n={yoe['overall_n_labeled']} labeled)")

    # Q4: industry profile.
    industries = fetch_industries(con)
    print("\n=== Q4 Top industries (2026 FDE) ===")
    for ind, n in industries:
        print(f"  {ind}: {n}")

    # Q5: AI density.
    ai_pattern = load_ai_strict_pattern()
    ai = fetch_ai_density(con, ai_pattern)
    print("\n=== Q5 AI density (v1_rebuilt ai_strict) ===")
    print(f"  FDE 2026 ai_strict rate:     "
          f"{ai['fde_ai_strict_rate']*100:.1f}%  (n={ai['fde_ai_strict_n']})")
    print(f"  Overall 2026 ai_strict rate: "
          f"{ai['overall_ai_strict_rate']*100:.1f}%  (n={ai['overall_ai_strict_n']})")
    print(f"  FDE 2026 mean AI-tool mentions/posting:     "
          f"{ai['fde_mean_ai_tool_mentions']:.2f}")
    print(f"  Overall 2026 mean AI-tool mentions/posting: "
          f"{ai['overall_mean_ai_tool_mentions']:.2f}")

    # Q6: exemplars.
    exemplars = fetch_exemplars(con)
    write_exemplars_csv(exemplars)
    print("\n=== Q6 Exemplars ===")
    for uid, comp, title, scrape, preview in exemplars:
        print(f"\n  uid={uid}  company={comp}  scrape={scrape}")
        print(f"  title={title}")
        print(f"  preview={(preview or '')[:200]}...")

    # Summary JSON for quick downstream consumption.
    summary = {
        "counts": {
            "2024": {
                "n_swe_linkedin": c2024[1],
                "n_fde_title": c2024[2],
                "fde_share": c2024[3],
            },
            "2026": {
                "n_swe_linkedin": c2026[1],
                "n_fde_title": c2026[2],
                "fde_share": c2026[3],
            },
            "absolute_growth_2024_to_2026": growth,
            "share_growth_2024_to_2026": share_ratio,
        },
        "yoe": yoe,
        "ai_density": ai,
        "top_companies_2026": [{"company": c, "n": n} for c, n in companies],
        "top_industries_2026": [{"industry": i, "n": n} for i, n in industries],
    }
    json_path = OUT_DIR / "fde_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[write] {json_path}")


if __name__ == "__main__":
    main()
