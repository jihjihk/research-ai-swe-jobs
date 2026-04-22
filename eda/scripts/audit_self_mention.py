"""
Self-mention contamination audit.

Two audits:
  AUDIT 1: Recompute Bay-vs-Rest token gaps (Composite A DD3) under exclusion of
           postings authored by frontier-AI labs / hyperscalers / coding-agent
           vendors whose own product names match the AI vocabulary.
  AUDIT 2: Recompute the BERTopic 'rag/ai_solutions/ai_systems' cluster
           (Composite B v2) share of 2024 vs 2026 SWE under the same exclusion,
           and the 5.2x multiplier.

Outputs:
  eda/tables/audit_self_mention_audit1_token_gap.csv
  eda/tables/audit_self_mention_audit1_excluded_volume.csv
  eda/tables/audit_self_mention_audit1_spotcheck30.csv
  eda/tables/audit_self_mention_audit2_cluster_firms.csv
  eda/tables/audit_self_mention_audit2_share.csv
  eda/tables/audit_self_mention_audit2_spotcheck30.csv

Run:
  ./.venv/bin/python eda/scripts/audit_self_mention.py
"""

from __future__ import annotations
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "eda" / "scripts"))
from scans import AI_VOCAB_PATTERN, text_col, text_filter  # noqa: E402
from S26_composite_a import BUILDER_TITLE_PATTERN  # noqa: E402

CORE = PROJECT_ROOT / "data" / "unified_core.parquet"
ARCH = PROJECT_ROOT / "eda" / "artifacts" / "composite_B_archetype_labels.parquet"
TABLES = PROJECT_ROOT / "eda" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

CORE_FILTER = "is_english = TRUE AND date_flag = 'ok' AND is_swe = TRUE"

# ----- self-mention exclusion list -----
# Firms whose presence in a JD will mechanically pull self-product names
# (openai, anthropic, gemini, copilot, claude, llm, agentic, etc.) into the
# AI-vocabulary count irrespective of any underlying market trend.
SELF_MENTION_FIRMS = [
    # Frontier model labs
    "OpenAI", "Anthropic", "xAI", "Cohere", "Perplexity",
    "Inflection AI", "Character.AI",
    # Hyperscalers / firms whose AI products use these tokens
    "Microsoft", "Microsoft AI", "GitHub",
    "Google",
    "Meta",
    "Amazon", "Amazon Web Services (AWS)",
    "NVIDIA",
    "Adobe",
    "Salesforce",
    # AI infra providers whose model surface mentions these vocab tokens
    "Databricks",
]

TECH_HUBS = {
    "San Francisco Bay Area",
    "Seattle Metro",
    "New York City Metro",
    "Austin Metro",
    "Boston Metro",
}

# Tokens to evaluate (Composite A DD3 — both Bay-leading and Rest-leading)
HUB_TOKENS = ["openai", "anthropic", "agentic", "ai agent", "llm", "foundation model"]
USER_TOKENS = ["copilot", "github copilot", "claude", "prompt engineering", "rag", "mlops"]
ALL_TOKENS = HUB_TOKENS + USER_TOKENS


def excl_clause():
    names_sql = ",".join("'" + n.replace("'", "''") + "'" for n in SELF_MENTION_FIRMS)
    return f"company_name_canonical NOT IN ({names_sql})"


def hub_clause():
    return "metro_area IN (" + ",".join("'" + m + "'" for m in TECH_HUBS) + ")"


def rest_clause():
    return "metro_area IS NOT NULL AND metro_area NOT IN (" + ",".join("'" + m + "'" for m in TECH_HUBS) + ")"


# ============================================================================
# AUDIT 1 — Bay-vs-Rest token gap under self-mention exclusion
# ============================================================================

def audit1(con):
    print("\n==== AUDIT 1: Bay-vs-Rest token gap under self-mention exclusion ====\n")

    # First: how many postings are excluded, by zone?
    excluded_vol = con.execute(f"""
      SELECT
        CASE WHEN {hub_clause()} THEN 'bay_hub' ELSE 'rest' END AS zone,
        COUNT(*) AS n_total,
        SUM(CASE WHEN company_name_canonical IN ({",".join("'" + n.replace("'", "''") + "'" for n in SELF_MENTION_FIRMS)}) THEN 1 ELSE 0 END) AS n_excluded
      FROM '{CORE}'
      WHERE {CORE_FILTER}
        AND period LIKE '2026%'
        AND metro_area IS NOT NULL
        AND NOT regexp_matches(LOWER(title), '{BUILDER_TITLE_PATTERN}')
      GROUP BY 1 ORDER BY 1
    """).df()
    print("Excluded posting volume (non-builder 2026, by zone):")
    print(excluded_vol.to_string())
    excluded_vol.to_csv(TABLES / "audit_self_mention_audit1_excluded_volume.csv", index=False)

    # Per-firm exclusion counts in non-builder 2026 corpus
    per_firm = con.execute(f"""
      SELECT company_name_canonical AS firm,
             CASE WHEN {hub_clause()} THEN 'bay_hub' ELSE 'rest' END AS zone,
             COUNT(*) AS n
      FROM '{CORE}'
      WHERE {CORE_FILTER}
        AND period LIKE '2026%'
        AND metro_area IS NOT NULL
        AND NOT regexp_matches(LOWER(title), '{BUILDER_TITLE_PATTERN}')
        AND company_name_canonical IN ({",".join("'" + n.replace("'", "''") + "'" for n in SELF_MENTION_FIRMS)})
      GROUP BY 1,2 ORDER BY n DESC
    """).df()
    print("\nPer-firm excluded postings (non-builder 2026):")
    print(per_firm.to_string())

    # Compute token gap original (replicates DD3) and excluded
    # Original DD3 used denominator: non-builder AI-matched postings.
    # Here we replicate that exact framing (rate among AI-matched non-builder posts).
    rows = []
    for label, exclude_self in [("original", False), ("self_excluded", True)]:
        excl = f" AND {excl_clause()}" if exclude_self else ""
        for zone, clause in [("bay_hub", hub_clause()), ("rest", rest_clause())]:
            df = con.execute(f"""
              SELECT LOWER(description) AS d
              FROM '{CORE}'
              WHERE {CORE_FILTER}
                AND period LIKE '2026%'
                AND {clause}
                AND NOT regexp_matches(LOWER(title), '{BUILDER_TITLE_PATTERN}')
                AND {text_filter()}
                AND regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}')
                {excl}
            """).df()
            n = len(df)
            for tok in ALL_TOKENS:
                hits = sum(1 for txt in df["d"] if re.search(r"\b" + re.escape(tok) + r"\b", txt or ""))
                rows.append({
                    "scenario": label,
                    "zone": zone,
                    "token": tok,
                    "n_posts": n,
                    "n_hit": hits,
                    "rate": hits / n if n else 0.0,
                })
    tok_long = pd.DataFrame(rows)

    # Pivot to wide: token | original_bay | original_rest | original_gap_pp | excl_bay | excl_rest | excl_gap_pp | abs_change
    out_rows = []
    for tok in ALL_TOKENS:
        orig_bay = tok_long[(tok_long.scenario == "original") & (tok_long.zone == "bay_hub") & (tok_long.token == tok)].iloc[0]
        orig_rest = tok_long[(tok_long.scenario == "original") & (tok_long.zone == "rest") & (tok_long.token == tok)].iloc[0]
        excl_bay = tok_long[(tok_long.scenario == "self_excluded") & (tok_long.zone == "bay_hub") & (tok_long.token == tok)].iloc[0]
        excl_rest = tok_long[(tok_long.scenario == "self_excluded") & (tok_long.zone == "rest") & (tok_long.token == tok)].iloc[0]
        orig_gap = (orig_bay["rate"] - orig_rest["rate"]) * 100
        excl_gap = (excl_bay["rate"] - excl_rest["rate"]) * 100
        out_rows.append({
            "token": tok,
            "category": "hub_leading" if tok in HUB_TOKENS else "user_leading",
            "orig_bay_n": int(orig_bay["n_posts"]),
            "orig_rest_n": int(orig_rest["n_posts"]),
            "orig_bay_rate_pct": orig_bay["rate"] * 100,
            "orig_rest_rate_pct": orig_rest["rate"] * 100,
            "orig_gap_pp": orig_gap,
            "excl_bay_n": int(excl_bay["n_posts"]),
            "excl_rest_n": int(excl_rest["n_posts"]),
            "excl_bay_rate_pct": excl_bay["rate"] * 100,
            "excl_rest_rate_pct": excl_rest["rate"] * 100,
            "excl_gap_pp": excl_gap,
            "abs_change_pp": excl_gap - orig_gap,
        })
    out = pd.DataFrame(out_rows)
    out.to_csv(TABLES / "audit_self_mention_audit1_token_gap.csv", index=False)
    print("\nToken gap original vs excluded:")
    print(out.to_string())

    # Spot-check 30 Bay 'openai' postings AT NON-EXCLUDED FIRMS
    spot = con.execute(f"""
      SELECT uid, title, company_name_canonical AS company, company_industry, description
      FROM '{CORE}'
      WHERE {CORE_FILTER}
        AND period LIKE '2026%'
        AND metro_area = 'San Francisco Bay Area'
        AND NOT regexp_matches(LOWER(title), '{BUILDER_TITLE_PATTERN}')
        AND {text_filter()}
        AND regexp_matches({text_col()}, '{AI_VOCAB_PATTERN}')
        AND regexp_matches(LOWER(description), '\\bopenai\\b')
        AND {excl_clause()}
      ORDER BY hash(uid || 'spot1')
      LIMIT 30
    """).df()

    def excerpt(d):
        m = re.search(r"\bopenai\b", (d or "").lower())
        if not m:
            return ""
        s = max(0, m.start() - 140)
        e = min(len(d), m.end() + 200)
        return re.sub(r"\s+", " ", d[s:e]).strip()

    spot["excerpt"] = spot["description"].apply(excerpt)
    spot_out = spot[["uid", "company", "company_industry", "title", "excerpt"]]
    spot_out.to_csv(TABLES / "audit_self_mention_audit1_spotcheck30.csv", index=False)
    print(f"\nSpot-check 30 Bay-Area 'openai'-mentioning non-builder postings, post-exclusion:")
    for _, r in spot_out.iterrows():
        print(f"- {r.company} ({r.company_industry}) -- {r.title}")
        print(f"  {r.excerpt[:280]}")
    print()


# ============================================================================
# AUDIT 2 — BERTopic cluster share under self-mention exclusion
# ============================================================================

def audit2(con):
    print("\n==== AUDIT 2: BERTopic cluster (Topic 1 RAG/AI) under self-mention exclusion ====\n")

    # Top firms in Topic 1 (cluster), by period
    top_firms = con.execute(f"""
      WITH j AS (
        SELECT u.uid, u.period, u.company_name_canonical AS firm
        FROM '{CORE}' u
        JOIN '{ARCH}' l ON u.uid = l.uid
        WHERE l.archetype_id = 1
      )
      SELECT firm,
             SUM(CASE WHEN period LIKE '2024%' THEN 1 ELSE 0 END) AS n_2024,
             SUM(CASE WHEN period LIKE '2026%' THEN 1 ELSE 0 END) AS n_2026,
             COUNT(*) AS n_total
      FROM j
      GROUP BY firm
      ORDER BY n_total DESC
      LIMIT 20
    """).df()
    top_firms.to_csv(TABLES / "audit_self_mention_audit2_cluster_firms.csv", index=False)
    print("Top 20 firms in BERTopic Topic 1 (RAG/AI cluster):")
    print(top_firms.to_string())

    # Recompute cluster share — original and excluded
    # Note: BERTopic was fit on a CAPPED sample (30 per firm-period). The 12.7% / 2.5%
    # numbers come from that capped sample. We use the same join here.
    excl_names = ",".join("'" + n.replace("'", "''") + "'" for n in SELF_MENTION_FIRMS)
    share = con.execute(f"""
      WITH joined AS (
        SELECT u.uid, u.period, u.company_name_canonical AS firm, l.archetype_id
        FROM '{CORE}' u
        JOIN '{ARCH}' l ON u.uid=l.uid
      )
      SELECT
        CASE WHEN period LIKE '2024%' THEN '2024' ELSE '2026' END AS yr,
        COUNT(*) AS n_total_orig,
        SUM(CASE WHEN archetype_id=1 THEN 1 ELSE 0 END) AS n_topic1_orig,
        SUM(CASE WHEN firm NOT IN ({excl_names}) THEN 1 ELSE 0 END) AS n_total_excl,
        SUM(CASE WHEN firm NOT IN ({excl_names}) AND archetype_id=1 THEN 1 ELSE 0 END) AS n_topic1_excl
      FROM joined
      GROUP BY 1 ORDER BY 1
    """).df()
    share["pct_topic1_orig"] = 100.0 * share["n_topic1_orig"] / share["n_total_orig"]
    share["pct_topic1_excl"] = 100.0 * share["n_topic1_excl"] / share["n_total_excl"]
    share.to_csv(TABLES / "audit_self_mention_audit2_share.csv", index=False)
    print("\nCluster-share original vs excluded:")
    print(share.to_string())

    # Multiplier
    p2024_orig = share[share.yr == "2024"]["pct_topic1_orig"].iloc[0]
    p2026_orig = share[share.yr == "2026"]["pct_topic1_orig"].iloc[0]
    p2024_excl = share[share.yr == "2024"]["pct_topic1_excl"].iloc[0]
    p2026_excl = share[share.yr == "2026"]["pct_topic1_excl"].iloc[0]
    print(f"\nMultiplier original: {p2024_orig:.2f}% -> {p2026_orig:.2f}% = {p2026_orig/p2024_orig:.2f}x")
    print(f"Multiplier excluded: {p2024_excl:.2f}% -> {p2026_excl:.2f}% = {p2026_excl/p2024_excl:.2f}x")

    # Spot-check 30 random 2026 cluster postings at non-frontier-AI firms
    excl_for_spot = ["OpenAI", "Anthropic", "Microsoft", "Microsoft AI", "Google", "Meta", "GitHub"]
    excl_spot_sql = ",".join("'" + n + "'" for n in excl_for_spot)
    spot = con.execute(f"""
      SELECT u.uid, u.title, u.company_name_canonical AS firm, u.company_industry, u.description
      FROM '{CORE}' u
      JOIN '{ARCH}' l ON u.uid=l.uid
      WHERE l.archetype_id = 1
        AND u.period LIKE '2026%'
        AND u.company_name_canonical NOT IN ({excl_spot_sql})
        AND u.company_name_canonical IS NOT NULL
      ORDER BY hash(u.uid || 'spot2')
      LIMIT 30
    """).df()

    AGENTIC_PAT = r"(?i)\b(rag|retrieval[- ]augmented|agentic|ai agent|llm|generative ai|genai|gen ai|mlops|llmops|foundation model|prompt engineering|vector database|openai|anthropic|claude|copilot|chatgpt|cursor)\b"

    def excerpt(d):
        m = re.search(AGENTIC_PAT, d or "")
        if not m:
            return (re.sub(r"\s+", " ", (d or ""))[:240]).strip()
        s = max(0, m.start() - 140)
        e = min(len(d), m.end() + 220)
        return re.sub(r"\s+", " ", d[s:e]).strip()

    spot["excerpt"] = spot["description"].apply(excerpt)
    spot["first_match"] = spot["description"].apply(
        lambda d: (re.search(AGENTIC_PAT, d or "").group(0) if re.search(AGENTIC_PAT, d or "") else "(no agentic token)")
    )
    spot_out = spot[["uid", "firm", "company_industry", "title", "first_match", "excerpt"]]
    spot_out.to_csv(TABLES / "audit_self_mention_audit2_spotcheck30.csv", index=False)
    print(f"\nSpot-check 30 random 2026 Topic 1 postings at non-frontier-AI firms:")
    for _, r in spot_out.iterrows():
        print(f"- [{r.first_match}] {r.firm} ({r.company_industry}) -- {r.title}")
        print(f"  {r.excerpt[:280]}")
    print()


def main():
    con = duckdb.connect()
    audit1(con)
    audit2(con)
    print("\nDone.")


if __name__ == "__main__":
    main()
