"""T06 — Company concentration deep investigation.

Produces:
  - exploration/reports/T06.md inputs (tables under exploration/tables/T06)
  - exploration/artifacts/shared/entry_specialist_employers.csv

All queries through DuckDB. Heavy aggregation happens server-side.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data" / "unified.parquet"
FIG = ROOT / "exploration" / "figures" / "T06"
TAB = ROOT / "exploration" / "tables" / "T06"
SHARED = ROOT / "exploration" / "artifacts" / "shared"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
SHARED.mkdir(parents=True, exist_ok=True)

CORE = """
  source_platform = 'linkedin'
  AND is_english = TRUE
  AND date_flag = 'ok'
  AND is_swe = TRUE
"""

SOURCES = ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]


def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute("SET memory_limit='12GB'")
    c.execute("SET threads=6")
    return c


def gini_from_counts(counts: np.ndarray) -> float:
    """Gini on a 1-D non-negative array."""
    x = np.asarray(counts, dtype=float)
    x = x[x > 0]
    if x.size == 0:
        return float("nan")
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return (2 * np.sum((np.arange(1, n + 1)) * x) - (n + 1) * cum[-1]) / (n * cum[-1])


def hhi_from_shares(shares: np.ndarray) -> float:
    """HHI on normalized shares (fractions summing to 1). Scaled 0-10000."""
    return float(np.sum((shares * 100.0) ** 2))


def concentration_metrics(df: pd.DataFrame, source: str, scope: str) -> dict:
    """df columns: c (company), n (postings)."""
    df = df.sort_values("n", ascending=False).reset_index(drop=True)
    total = df["n"].sum()
    if total == 0:
        return {"source": source, "scope": scope, "n_total": 0, "n_companies": 0}
    shares = df["n"].to_numpy() / total
    row = {
        "source": source,
        "scope": scope,
        "n_total_postings": int(total),
        "n_companies": int(len(df)),
        "hhi": hhi_from_shares(shares),
        "gini": gini_from_counts(df["n"].to_numpy()),
    }
    for k in (1, 5, 10, 20, 50):
        row[f"top{k}_share"] = float(shares[:k].sum())
    return row


# ---------------------------------------------------------------------------
# Step 1. Concentration metrics per source (all and aggregator-excluded)
# ---------------------------------------------------------------------------

def step1_concentration(c):
    print("--- Step 1: concentration metrics ---")
    rows = []
    for src in SOURCES:
        for scope, agg_filter in [("all", ""), ("no_aggregator", "AND (is_aggregator IS NULL OR is_aggregator=FALSE)")]:
            df = c.execute(f"""
                SELECT company_name_canonical AS c, COUNT(*) AS n
                FROM read_parquet('{DATA}')
                WHERE {CORE} AND source='{src}' {agg_filter}
                  AND company_name_canonical IS NOT NULL
                GROUP BY 1
            """).df()
            rows.append(concentration_metrics(df, src, scope))
    tbl = pd.DataFrame(rows)
    tbl.to_csv(TAB / "01_concentration.csv", index=False)
    print(tbl.to_string(index=False))
    return tbl


# ---------------------------------------------------------------------------
# Step 2. Top-20 employer profile per source
# ---------------------------------------------------------------------------

def step2_top20(c):
    print("--- Step 2: top-20 employer profile per source ---")
    out_all = []
    for src in SOURCES:
        df = c.execute(f"""
            WITH swe AS (
              SELECT * FROM read_parquet('{DATA}')
              WHERE {CORE} AND source='{src}' AND company_name_canonical IS NOT NULL
            ),
            src_total AS (SELECT COUNT(*) AS n FROM swe),
            top AS (
              SELECT company_name_canonical AS company, COUNT(*) AS n
              FROM swe GROUP BY 1 ORDER BY n DESC LIMIT 20
            )
            SELECT t.company AS company, t.n AS posting_count,
                   t.n::DOUBLE / (SELECT n FROM src_total) AS share_of_source,
                   AVG(swe.yoe_extracted) AS mean_yoe,
                   AVG(swe.description_length) AS mean_desc_len,
                   SUM(CASE WHEN swe.seniority_final='entry' THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS entry_share_senfinal,
                   SUM(CASE WHEN swe.yoe_extracted IS NOT NULL AND swe.yoe_extracted<=2 THEN 1 ELSE 0 END)::DOUBLE
                    / NULLIF(SUM(CASE WHEN swe.yoe_extracted IS NOT NULL THEN 1 ELSE 0 END), 0) AS entry_share_yoe,
                   SUM(CASE WHEN swe.is_aggregator THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS aggregator_share,
                   MODE(swe.company_industry) AS mode_industry
            FROM top t JOIN swe ON swe.company_name_canonical = t.company
            GROUP BY t.company, t.n
            ORDER BY t.n DESC
        """).df()
        df["source"] = src
        df.to_csv(TAB / f"02_top20_{src}.csv", index=False)
        print(f"  top20 — {src}:")
        print(df[["company", "posting_count", "share_of_source", "mean_yoe",
                  "mean_desc_len", "entry_share_senfinal", "entry_share_yoe",
                  "aggregator_share", "mode_industry"]].round(3).to_string(index=False))
        out_all.append(df)
    all_top20 = pd.concat(out_all, ignore_index=True)
    all_top20.to_csv(TAB / "02_top20_all.csv", index=False)
    return all_top20


# ---------------------------------------------------------------------------
# Step 3. Duplicate-template audit (per-source and cross-company collision)
# ---------------------------------------------------------------------------

def step3_duplicate_template(c):
    print("--- Step 3: duplicate-template audit ---")
    per_source_rows = []
    for src in SOURCES:
        df = c.execute(f"""
            WITH swe AS (
              SELECT * FROM read_parquet('{DATA}')
              WHERE {CORE} AND source='{src}' AND company_name_canonical IS NOT NULL
            )
            SELECT company_name_canonical AS company,
                   COUNT(*) AS posting_count,
                   COUNT(DISTINCT description_hash) AS distinct_hash,
                   MAX(cnt) AS max_rows_per_hash
            FROM (
              SELECT company_name_canonical, description_hash,
                     COUNT(*) AS cnt
              FROM swe
              GROUP BY 1, 2
            ) g
            GROUP BY 1
            HAVING COUNT(*) >= 5
            ORDER BY (COUNT(*)::DOUBLE / COUNT(DISTINCT description_hash)) DESC, COUNT(*) DESC
            LIMIT 10
        """).df()
        df["max_dup_ratio"] = df["posting_count"] / df["distinct_hash"].replace(0, np.nan)
        df["source"] = src
        df.to_csv(TAB / f"03_dup_templates_{src}.csv", index=False)
        print(f"  top dup-template — {src}:")
        print(df.to_string(index=False))
        per_source_rows.append(df)
    all_dup = pd.concat(per_source_rows, ignore_index=True)
    all_dup.to_csv(TAB / "03_dup_templates_all.csv", index=False)

    # Cross-company collisions: the same description_hash used by >1 company
    collisions = c.execute(f"""
        SELECT description_hash,
               COUNT(DISTINCT company_name_canonical) AS n_companies,
               COUNT(*) AS n_rows,
               ARRAY_AGG(DISTINCT company_name_canonical)[1:5] AS sample_companies,
               ARRAY_AGG(DISTINCT source)[1:3] AS sources
        FROM read_parquet('{DATA}')
        WHERE {CORE} AND company_name_canonical IS NOT NULL AND description_hash IS NOT NULL
        GROUP BY 1
        HAVING COUNT(DISTINCT company_name_canonical) >= 2
        ORDER BY n_rows DESC
        LIMIT 20
    """).df()
    collisions.to_csv(TAB / "03_cross_company_collisions.csv", index=False)
    print("cross-company description hash collisions (top 20):")
    print(collisions.to_string(index=False))

    return all_dup, collisions


# ---------------------------------------------------------------------------
# Step 4. Entry-level posting concentration
# ---------------------------------------------------------------------------

def step4_entry_concentration(c):
    print("--- Step 4: entry-level posting concentration ---")
    per_source = []
    for src in SOURCES:
        # For each company: total postings, entry (J1), J3
        comp = c.execute(f"""
            SELECT company_name_canonical AS company,
                   COUNT(*) AS n,
                   SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END) AS j1,
                   SUM(CASE WHEN seniority_final IN ('entry','associate') THEN 1 ELSE 0 END) AS j2,
                   SUM(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted<=2 THEN 1 ELSE 0 END) AS j3,
                   SUM(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted<=3 THEN 1 ELSE 0 END) AS j4,
                   SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS n_yoe,
                   SUM(CASE WHEN is_aggregator THEN 1 ELSE 0 END) AS agg_n
            FROM read_parquet('{DATA}')
            WHERE {CORE} AND source='{src}' AND company_name_canonical IS NOT NULL
            GROUP BY 1
        """).df()

        def shape_stats(mask_col, note):
            d = comp.copy()
            d["has_entry"] = d[mask_col] > 0
            d_big = d[d["n"] >= 5]
            any_entry = int((d[mask_col] > 0).sum())
            total_cos = len(d)
            big_zero = int(((d_big[mask_col] == 0)).sum())
            big_total = len(d_big)
            row = {
                "source": src,
                "definition": note,
                "n_companies": total_cos,
                "n_companies_ge5": big_total,
                "n_companies_any_entry": any_entry,
                "share_companies_any_entry": any_entry / total_cos if total_cos else float("nan"),
                "n_ge5_zero_entry": big_zero,
                "share_ge5_zero_entry": big_zero / big_total if big_total else float("nan"),
            }
            # Distribution of entry-share among companies that DO post entry
            d_has = d_big[d_big[mask_col] > 0].copy()
            if not d_has.empty:
                denom = "n" if note in ("J1", "J2") else "n_yoe"
                d_has["entry_share"] = d_has[mask_col] / d_has[denom]
                for q in (0.25, 0.5, 0.75, 0.9, 0.95):
                    row[f"p{int(q*100)}_own_entry_share"] = float(d_has["entry_share"].quantile(q))
                row["mean_own_entry_share"] = float(d_has["entry_share"].mean())
                row["n_ge5_any_entry"] = int(len(d_has))
            return row

        per_source.append(shape_stats("j1", "J1"))
        per_source.append(shape_stats("j3", "J3"))
        per_source.append(shape_stats("j2", "J2"))
        per_source.append(shape_stats("j4", "J4"))

        # Save per-company table for later use
        comp["source"] = src
        comp.to_csv(TAB / f"04_company_entry_{src}.csv", index=False)
    out = pd.DataFrame(per_source)
    out.to_csv(TAB / "04_entry_concentration_shape.csv", index=False)
    print(out.to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# Step 5. Within-company vs between-company decomposition (arshkon vs scraped)
# ---------------------------------------------------------------------------

def step5_decomposition(c):
    print("--- Step 5: within/between decomposition (arshkon vs scraped) ---")
    # Companies with >=5 SWE postings in BOTH arshkon and scraped
    panel = c.execute(f"""
        WITH by_co AS (
          SELECT company_name_canonical AS company, source, COUNT(*) AS n,
                 SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END) AS j1,
                 SUM(CASE WHEN seniority_final IN ('entry','associate') THEN 1 ELSE 0 END) AS j2,
                 SUM(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted<=2 THEN 1 ELSE 0 END) AS j3,
                 SUM(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted<=3 THEN 1 ELSE 0 END) AS j4,
                 SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS n_yoe,
                 AVG(description_length) AS mean_desc_len,
                 AVG(yoe_extracted) AS mean_yoe,
                 SUM(CASE WHEN regexp_matches(lower(description), '\\bai\\b|copilot|cursor|llm|gpt|claude|gemini|chatgpt|agentic|generative') THEN 1 ELSE 0 END) AS ai_mentions
          FROM read_parquet('{DATA}')
          WHERE {CORE} AND source IN ('kaggle_arshkon','scraped')
            AND company_name_canonical IS NOT NULL
          GROUP BY 1,2
        ),
        ar AS (SELECT * FROM by_co WHERE source='kaggle_arshkon'),
        sc AS (SELECT * FROM by_co WHERE source='scraped')
        SELECT ar.company,
               ar.n AS n_arsh, sc.n AS n_scraped,
               ar.j1 AS j1_arsh, sc.j1 AS j1_scraped,
               ar.j2 AS j2_arsh, sc.j2 AS j2_scraped,
               ar.j3 AS j3_arsh, sc.j3 AS j3_scraped,
               ar.j4 AS j4_arsh, sc.j4 AS j4_scraped,
               ar.n_yoe AS nyoe_arsh, sc.n_yoe AS nyoe_scraped,
               ar.mean_desc_len AS len_arsh, sc.mean_desc_len AS len_scraped,
               ar.ai_mentions AS ai_arsh, sc.ai_mentions AS ai_scraped
        FROM ar INNER JOIN sc ON ar.company = sc.company
        WHERE ar.n >= 5 AND sc.n >= 5
    """).df()
    panel.to_csv(TAB / "05_overlap_panel.csv", index=False)
    print(f"  overlap panel size: {len(panel)} companies")

    # Oaxaca-style decomposition for a share metric
    def decompose_share(panel_df, num_arsh, den_arsh, num_sc, den_sc):
        """Return dict with aggregate, within, between components."""
        N_a = panel_df[den_arsh].sum()
        N_s = panel_df[den_sc].sum()
        X_a = panel_df[num_arsh].sum()
        X_s = panel_df[num_sc].sum()
        if N_a == 0 or N_s == 0:
            return {"aggregate": float("nan"), "within": float("nan"), "between": float("nan"),
                    "share_arsh": float("nan"), "share_sc": float("nan")}
        p_a = X_a / N_a
        p_s = X_s / N_s
        agg = p_s - p_a
        # Shepard-style decomposition: use average weights
        w_a = panel_df[den_arsh] / N_a
        w_s = panel_df[den_sc] / N_s
        with np.errstate(divide="ignore", invalid="ignore"):
            p_co_a = np.where(panel_df[den_arsh] > 0, panel_df[num_arsh] / panel_df[den_arsh], 0)
            p_co_s = np.where(panel_df[den_sc] > 0, panel_df[num_sc] / panel_df[den_sc], 0)
        avg_w = (w_a + w_s) / 2
        avg_p = (p_co_a + p_co_s) / 2
        within = float(np.sum(avg_w * (p_co_s - p_co_a)))
        between = float(np.sum((w_s - w_a) * avg_p))
        return {"aggregate": float(agg), "within": within, "between": between,
                "share_arsh": float(p_a), "share_sc": float(p_s)}

    rows = []
    rows.append({"metric": "entry_share_J1",
                 **decompose_share(panel, "j1_arsh", "n_arsh", "j1_scraped", "n_scraped")})
    rows.append({"metric": "entry_share_J2",
                 **decompose_share(panel, "j2_arsh", "n_arsh", "j2_scraped", "n_scraped")})
    rows.append({"metric": "entry_share_J3",
                 **decompose_share(panel, "j3_arsh", "nyoe_arsh", "j3_scraped", "nyoe_scraped")})
    rows.append({"metric": "entry_share_J4",
                 **decompose_share(panel, "j4_arsh", "nyoe_arsh", "j4_scraped", "nyoe_scraped")})
    rows.append({"metric": "ai_mention_share",
                 **decompose_share(panel, "ai_arsh", "n_arsh", "ai_scraped", "n_scraped")})

    # For continuous mean metrics, weighted averages decomposition
    def decompose_mean(panel_df, num_arsh, den_arsh, num_sc, den_sc,
                        mean_arsh_col, mean_sc_col):
        """If per-company means already computed. Use weights = den."""
        N_a = panel_df[den_arsh].sum()
        N_s = panel_df[den_sc].sum()
        if N_a == 0 or N_s == 0:
            return {"aggregate": float("nan"), "within": float("nan"), "between": float("nan")}
        w_a = panel_df[den_arsh] / N_a
        w_s = panel_df[den_sc] / N_s
        m_a = panel_df[mean_arsh_col].fillna(0)
        m_s = panel_df[mean_sc_col].fillna(0)
        agg_a = float(np.sum(w_a * m_a))
        agg_s = float(np.sum(w_s * m_s))
        avg_w = (w_a + w_s) / 2
        avg_m = (m_a + m_s) / 2
        within = float(np.sum(avg_w * (m_s - m_a)))
        between = float(np.sum((w_s - w_a) * avg_m))
        return {"aggregate": agg_s - agg_a, "within": within, "between": between,
                "mean_arsh": agg_a, "mean_sc": agg_s}

    rows.append({"metric": "mean_desc_len",
                 **decompose_mean(panel, "len_arsh", "n_arsh", "len_scraped", "n_scraped",
                                    "len_arsh", "len_scraped")})

    dec = pd.DataFrame(rows)
    dec.to_csv(TAB / "05_decomposition.csv", index=False)
    print(dec.to_string(index=False))
    return dec, panel


# ---------------------------------------------------------------------------
# Step 6. Entry-specialist employer identification
# ---------------------------------------------------------------------------

def step6_specialists(c):
    print("--- Step 6: entry-specialist employers ---")
    # Compute junior share under J1..J4 per company (>=5 postings) across ALL sources.
    # Keep per-source output + a pooled row per company so downstream can pick.
    df = c.execute(f"""
        SELECT company_name_canonical AS company,
               source,
               COUNT(*) AS n,
               SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END) AS j1,
               SUM(CASE WHEN seniority_final IN ('entry','associate') THEN 1 ELSE 0 END) AS j2,
               SUM(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted<=2 THEN 1 ELSE 0 END) AS j3,
               SUM(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted<=3 THEN 1 ELSE 0 END) AS j4,
               SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS n_yoe,
               BOOL_OR(is_aggregator) AS is_aggregator,
               MODE(company_industry) AS mode_industry
        FROM read_parquet('{DATA}')
        WHERE {CORE} AND company_name_canonical IS NOT NULL
        GROUP BY 1,2
    """).df()

    # pooled company (across sources)
    pooled = df.groupby("company", as_index=False).agg(
        n=("n", "sum"), j1=("j1", "sum"), j2=("j2", "sum"),
        j3=("j3", "sum"), j4=("j4", "sum"), n_yoe=("n_yoe", "sum"),
        is_aggregator=("is_aggregator", "max"),
        mode_industry=("mode_industry", lambda s: s.mode().iat[0] if not s.mode().empty else None),
        sources_present=("source", lambda s: sorted(set(s))),
    )
    pooled["n_sources"] = pooled["sources_present"].apply(len)
    pooled["j1_share"] = pooled["j1"] / pooled["n"]
    pooled["j2_share"] = pooled["j2"] / pooled["n"]
    pooled["j3_share"] = pooled["j3"] / pooled["n_yoe"].replace(0, np.nan)
    pooled["j4_share"] = pooled["j4"] / pooled["n_yoe"].replace(0, np.nan)

    # Flag companies where ANY variant exceeds 60%
    big = pooled[pooled["n"] >= 5].copy()
    big["flag"] = (
        (big["j1_share"] > 0.6) |
        (big["j2_share"] > 0.6) |
        (big["j3_share"] > 0.6) |
        (big["j4_share"] > 0.6)
    )
    flagged = big[big["flag"]].sort_values("n", ascending=False)
    print(f"  total companies n>=5 (pooled across sources): {len(big)}")
    print(f"  flagged (junior >60% under any variant): {len(flagged)}")

    # Save per-source detail for reference
    df.to_csv(TAB / "06_company_junior_shares_bysource.csv", index=False)
    big.to_csv(TAB / "06_company_junior_shares_pooled.csv", index=False)

    # For the flagged set, build the shared artifact
    flagged_out = flagged.copy()
    flagged_out["sources_present"] = flagged_out["sources_present"].apply(lambda s: ",".join(s))

    # Categorization heuristics — apply to top 20 for manual label; apply heuristic to all.
    def categorize(row):
        nm = str(row["company"]).lower()
        ind = str(row.get("mode_industry") or "").lower()
        if row["is_aggregator"]:
            return "a_staffing"
        # staffing/college/recruiting heuristics
        staffing_tokens = ("staffing", "recruit", "resource", "talent", "dice", "allegis",
                           "randstad", "kforce", "insight global", "cybercoders", "robert half",
                           "tata consult", "infosys", "wipro", "hcl", "cognizant", "accenture",
                           "capgemini", "ltimindtree", "genpact", "mphasis", "tech mahindra",
                           "deloitte", "pwc", "ey ", "kpmg")
        if any(t in nm for t in staffing_tokens):
            return "d_bulk_consulting_or_staffing"
        college_tokens = ("college", "university", "student", "graduate", "intern network",
                          "handshake", "wayup", "joinhandshake")
        if any(t in nm for t in college_tokens):
            return "b_college_intermediary"
        tech_giants = ("amazon", "google", "microsoft", "meta", "apple", "nvidia", "oracle",
                        "ibm", "salesforce", "netflix", "uber", "linkedin", "intel",
                        "adobe", "cisco", "snap", "pinterest", "airbnb", "stripe", "tiktok",
                        "bytedance", "bloomberg", "citadel", "jane street", "jpmorgan",
                        "morgan stanley", "goldman")
        if any(t in nm for t in tech_giants):
            return "c_tech_giant"
        return "e_direct_employer"

    flagged_out["employer_category_heuristic"] = flagged_out.apply(categorize, axis=1)

    # Apply nicely-formatted ordering and save
    keep_cols = ["company", "n", "n_yoe", "j1", "j2", "j3", "j4",
                  "j1_share", "j2_share", "j3_share", "j4_share",
                  "is_aggregator", "mode_industry", "sources_present",
                  "n_sources", "employer_category_heuristic"]
    flagged_out = flagged_out[keep_cols].reset_index(drop=True)
    flagged_out.to_csv(SHARED / "entry_specialist_employers.csv", index=False)
    flagged_out.to_csv(TAB / "06_entry_specialist_employers.csv", index=False)
    print(flagged_out.head(20).to_string(index=False))

    # Cross-tab of flagged-but-not-aggregator (the invisible intermediary class)
    xtab = flagged_out.groupby(["employer_category_heuristic", "is_aggregator"]).size().unstack(fill_value=0)
    xtab.to_csv(TAB / "06_category_vs_aggregator.csv")
    print("category × aggregator cross-tab:"); print(xtab.to_string())

    # Top-20 table with fields useful for manual categorization
    top20 = flagged_out.head(20).copy()
    top20.to_csv(TAB / "06_top20_specialists_for_review.csv", index=False)

    return flagged_out, pooled


# ---------------------------------------------------------------------------
# Step 7. Aggregator profile
# ---------------------------------------------------------------------------

def step7_aggregator(c):
    print("--- Step 7: aggregator profile ---")
    df = c.execute(f"""
        SELECT source,
               COUNT(*) FILTER (WHERE is_aggregator) AS n_agg,
               COUNT(*) AS n_total,
               AVG(CASE WHEN is_aggregator THEN description_length END) AS agg_mean_len,
               AVG(CASE WHEN NOT is_aggregator OR is_aggregator IS NULL THEN description_length END) AS nonagg_mean_len,
               AVG(CASE WHEN is_aggregator THEN yoe_extracted END) AS agg_mean_yoe,
               AVG(CASE WHEN NOT is_aggregator OR is_aggregator IS NULL THEN yoe_extracted END) AS nonagg_mean_yoe,
               SUM(CASE WHEN is_aggregator AND seniority_final='entry' THEN 1 ELSE 0 END)::DOUBLE
                 / NULLIF(SUM(CASE WHEN is_aggregator THEN 1 ELSE 0 END), 0) AS agg_entry_share,
               SUM(CASE WHEN (NOT is_aggregator OR is_aggregator IS NULL) AND seniority_final='entry' THEN 1 ELSE 0 END)::DOUBLE
                 / NULLIF(SUM(CASE WHEN NOT is_aggregator OR is_aggregator IS NULL THEN 1 ELSE 0 END), 0) AS nonagg_entry_share
        FROM read_parquet('{DATA}')
        WHERE {CORE}
        GROUP BY source
        ORDER BY source
    """).df()
    df["agg_share_of_source"] = df["n_agg"] / df["n_total"]
    df.to_csv(TAB / "07_aggregator_profile.csv", index=False)
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Step 8. New entrants in 2026
# ---------------------------------------------------------------------------

def step8_new_entrants(c):
    print("--- Step 8: new entrants in 2026 ---")
    # companies in scraped NOT in any 2024 source
    entrants = c.execute(f"""
        WITH scraped AS (
          SELECT DISTINCT company_name_canonical AS company FROM read_parquet('{DATA}')
          WHERE {CORE} AND source='scraped' AND company_name_canonical IS NOT NULL
        ),
        hist AS (
          SELECT DISTINCT company_name_canonical AS company FROM read_parquet('{DATA}')
          WHERE {CORE} AND source IN ('kaggle_arshkon','kaggle_asaniczka')
            AND company_name_canonical IS NOT NULL
        )
        SELECT
          (SELECT COUNT(*) FROM scraped) AS n_scraped_co,
          (SELECT COUNT(*) FROM hist) AS n_hist_co,
          (SELECT COUNT(*) FROM scraped WHERE company NOT IN (SELECT company FROM hist)) AS n_new,
          (SELECT COUNT(*) FROM scraped WHERE company IN (SELECT company FROM hist)) AS n_returning
    """).df()
    entrants.to_csv(TAB / "08_new_entrants_summary.csv", index=False)
    print(entrants.to_string(index=False))

    profile = c.execute(f"""
        WITH scraped AS (
          SELECT * FROM read_parquet('{DATA}')
          WHERE {CORE} AND source='scraped' AND company_name_canonical IS NOT NULL
        ),
        hist AS (
          SELECT DISTINCT company_name_canonical AS c FROM read_parquet('{DATA}')
          WHERE {CORE} AND source IN ('kaggle_arshkon','kaggle_asaniczka')
            AND company_name_canonical IS NOT NULL
        )
        SELECT CASE WHEN company_name_canonical IN (SELECT c FROM hist) THEN 'returning' ELSE 'new' END AS cohort,
               COUNT(*) AS n,
               AVG(description_length) AS mean_len,
               AVG(yoe_extracted) AS mean_yoe,
               SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS entry_share,
               SUM(CASE WHEN seniority_final IN ('entry','associate') THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS j2_share,
               SUM(CASE WHEN is_aggregator THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS aggregator_share,
               AVG(CASE WHEN seniority_final='mid-senior' THEN 1.0 ELSE 0.0 END) AS midsenior_share,
               AVG(CASE WHEN seniority_final='director' THEN 1.0 ELSE 0.0 END) AS director_share
        FROM scraped
        GROUP BY 1
        ORDER BY 1
    """).df()
    profile.to_csv(TAB / "08_new_vs_returning.csv", index=False)
    print(profile.to_string(index=False))

    return entrants, profile


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    c = con()
    s1 = step1_concentration(c)
    s2 = step2_top20(c)
    s3_dup, s3_coll = step3_duplicate_template(c)
    s4 = step4_entry_concentration(c)
    s5_dec, s5_panel = step5_decomposition(c)
    s6_flag, s6_pool = step6_specialists(c)
    s7 = step7_aggregator(c)
    s8_entrants, s8_profile = step8_new_entrants(c)

    summary = {
        "concentration": s1.to_dict(orient="records"),
        "dup_top": s3_dup.to_dict(orient="records"),
        "cross_co_collisions_top": s3_coll.to_dict(orient="records"),
        "entry_concentration_shape": s4.to_dict(orient="records"),
        "decomposition": s5_dec.to_dict(orient="records"),
        "aggregator": s7.to_dict(orient="records"),
        "new_entrants": s8_entrants.to_dict(orient="records"),
        "new_vs_returning": s8_profile.to_dict(orient="records"),
        "n_specialists_flagged": int(len(s6_flag)),
    }
    (TAB / "T06_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("--- wrote summary ---")


if __name__ == "__main__":
    main()
