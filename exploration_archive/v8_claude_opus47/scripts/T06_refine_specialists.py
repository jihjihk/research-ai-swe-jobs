"""T06 step 6 refinement.

- Reclassify employer categories using a more defensible rule.
- Compute aggregator rate (not OR) per company.
- Build a manual-review bundle for the top 20 flagged companies.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data" / "unified.parquet"
TAB = ROOT / "exploration" / "tables" / "T06"
SHARED = ROOT / "exploration" / "artifacts" / "shared"

CORE = """
  source_platform = 'linkedin'
  AND is_english = TRUE
  AND date_flag = 'ok'
  AND is_swe = TRUE
"""


def con():
    c = duckdb.connect()
    c.execute("SET memory_limit='12GB'")
    c.execute("SET threads=6")
    return c


def main():
    c = con()

    # Per-company pooled counts + aggregator rate + total aggregator sources
    df = c.execute(f"""
        SELECT company_name_canonical AS company,
               COUNT(*) AS n,
               SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END) AS j1,
               SUM(CASE WHEN seniority_final IN ('entry','associate') THEN 1 ELSE 0 END) AS j2,
               SUM(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted<=2 THEN 1 ELSE 0 END) AS j3,
               SUM(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted<=3 THEN 1 ELSE 0 END) AS j4,
               SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS n_yoe,
               SUM(CASE WHEN is_aggregator THEN 1 ELSE 0 END) AS agg_rows,
               MODE(company_industry) AS mode_industry,
               ARRAY_AGG(DISTINCT source) AS sources_present,
               ARRAY_AGG(DISTINCT title_normalized ORDER BY title_normalized)[1:3] AS sample_titles
        FROM read_parquet('{DATA}')
        WHERE {CORE} AND company_name_canonical IS NOT NULL
        GROUP BY 1
    """).df()
    df["agg_rate"] = df["agg_rows"] / df["n"]
    df["j1_share"] = df["j1"] / df["n"]
    df["j2_share"] = df["j2"] / df["n"]
    df["j3_share"] = df["j3"] / df["n_yoe"].replace(0, np.nan)
    df["j4_share"] = df["j4"] / df["n_yoe"].replace(0, np.nan)
    df["n_sources"] = df["sources_present"].apply(lambda v: len(v) if v is not None else 0)
    df["sources_present_str"] = df["sources_present"].apply(
        lambda v: ",".join(sorted(v)) if v is not None else "")

    big = df[df["n"] >= 5].copy()
    flag = (
        (big["j1_share"] > 0.6) |
        (big["j2_share"] > 0.6) |
        (big["j3_share"] > 0.6) |
        (big["j4_share"] > 0.6)
    )
    flagged = big[flag].copy()

    # Category heuristic v2: based on majority-aggregator, name tokens, and industry.
    TECH_GIANTS = ("amazon", "google", "microsoft", "meta", "apple", "nvidia", "oracle",
                   "ibm", "salesforce", "netflix", "uber", "linkedin", "intel", "adobe",
                   "cisco", "snap", "pinterest", "airbnb", "stripe", "tiktok", "bytedance",
                   "bloomberg", "citadel", "jane street", "jpmorgan", "morgan stanley",
                   "goldman", "walmart")
    CONSULTING = ("tata consult", "infosys", "wipro", "hcl", "cognizant", "accenture",
                  "capgemini", "ltimindtree", "genpact", "mphasis", "tech mahindra",
                  "deloitte", "pwc", "ey ", "kpmg", "booz allen", "sap")
    STAFFING_TOKENS = ("staffing", "recruit", "resource", "talent", "dice", "lensa",
                        "allegis", "randstad", "kforce", "insight global", "cybercoders",
                        "robert half", "apex systems", "motion recruitment", "teksystems",
                        "actalent", "compunnel", "akkodis", "experis", "erias ventures",
                        "turing", "hackajob", "haystack", "clearancejobs", "clickjobs",
                        "jobs via dice", "jobs for humanity", "bowman williams",
                        "goliath partners", "intellectt inc", "bright vision")
    COLLEGE_TOKENS = ("college", "university", "student", "graduate", "intern network",
                       "handshake", "wayup", "joinhandshake")

    def cat(row):
        nm = str(row["company"]).lower()
        ind = str(row.get("mode_industry") or "").lower()
        if row["agg_rate"] >= 0.5:
            return "a_staffing_aggregator"
        if any(t in nm for t in STAFFING_TOKENS):
            return "a_staffing"
        if any(t in nm for t in CONSULTING):
            return "d_bulk_consulting"
        if any(t in nm for t in COLLEGE_TOKENS):
            return "b_college_intermediary"
        if any(t in nm for t in TECH_GIANTS):
            return "c_tech_giant"
        # secondary signals
        if "staffing" in ind or "recruit" in ind:
            return "a_staffing"
        return "e_direct_employer"

    flagged["employer_category"] = flagged.apply(cat, axis=1)

    # Identify which variant(s) trigger the flag
    def trigger_variants(row):
        v = []
        for key in ("j1_share", "j2_share", "j3_share", "j4_share"):
            if row[key] is not None and row[key] > 0.6:
                v.append(key.split("_")[0])
        return "|".join(v) or "none"

    flagged["triggered_by"] = flagged.apply(trigger_variants, axis=1)

    # Save detailed CSV — overwrite the shared artifact with the refined version.
    keep_cols = ["company", "n", "n_yoe", "j1", "j2", "j3", "j4",
                  "j1_share", "j2_share", "j3_share", "j4_share",
                  "agg_rows", "agg_rate", "mode_industry",
                  "sources_present_str", "n_sources",
                  "employer_category", "triggered_by", "sample_titles"]
    flagged_out = flagged[keep_cols].sort_values("n", ascending=False).reset_index(drop=True)
    flagged_out = flagged_out.rename(columns={"sources_present_str": "sources_present"})
    flagged_out.to_csv(SHARED / "entry_specialist_employers.csv", index=False)
    flagged_out.to_csv(TAB / "06_entry_specialist_employers.csv", index=False)

    print(f"total companies n>=5: {len(big)}")
    print(f"flagged (any J1..J4 > 60%): {len(flagged_out)}")
    print("breakdown by category:")
    print(flagged_out["employer_category"].value_counts())
    print("\nbreakdown by triggered variant:")
    print(flagged_out["triggered_by"].value_counts())

    # Cross-tab category x agg
    xtab = flagged_out.groupby(["employer_category"]).agg(
        n_companies=("company", "count"),
        n_postings=("n", "sum"),
        n_agg_rate_ge50=("agg_rate", lambda s: int((s >= 0.5).sum())),
    )
    xtab.to_csv(TAB / "06_category_summary.csv")
    print("\ncategory summary:")
    print(xtab.to_string())

    # Top-20 for manual review
    top20 = flagged_out.head(20).copy()
    top20.to_csv(TAB / "06_top20_specialists_for_review.csv", index=False)
    print("\ntop-20 flagged (manual review bundle):")
    cols = ["company", "n", "j1_share", "j2_share", "j3_share", "j4_share",
            "agg_rate", "employer_category", "triggered_by", "sources_present"]
    print(top20[cols].round(3).to_string(index=False))

    # Summary JSON (JSON-safe)
    summary = {
        "n_companies_n_ge5": int(len(big)),
        "n_flagged": int(len(flagged_out)),
        "category_breakdown": flagged_out["employer_category"].value_counts().to_dict(),
        "trigger_breakdown": flagged_out["triggered_by"].value_counts().to_dict(),
        "top20_total_postings": int(top20["n"].sum()),
        "all_flagged_total_postings": int(flagged_out["n"].sum()),
    }
    (TAB / "06_specialist_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nsummary json:", summary)

    # Aggregated impact — how many SWE rows flow through flagged employers?
    spec_names = set(flagged_out["company"].tolist())
    impact = c.execute(f"""
        WITH swe AS (
          SELECT * FROM read_parquet('{DATA}')
          WHERE {CORE} AND company_name_canonical IS NOT NULL
        )
        SELECT source,
               COUNT(*) AS n_total,
               COUNT(*) FILTER (WHERE company_name_canonical IN ({",".join([f"'" + s.replace("'", "''") + "'" for s in spec_names])})) AS n_specialist
        FROM swe
        GROUP BY source ORDER BY source
    """).df()
    impact["specialist_share"] = impact["n_specialist"] / impact["n_total"]
    impact.to_csv(TAB / "06_specialist_impact.csv", index=False)
    print("\nimpact — SWE rows flowing through specialist employers:")
    print(impact.to_string(index=False))


if __name__ == "__main__":
    main()
