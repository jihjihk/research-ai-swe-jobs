"""V1.9 - Seniority-panel specification-dependence check.

Check whether Wave 2 seniority-stratified findings are unanimous or split
across the T30 panel (J1-J4 and S1-S4).

Specifically:
- T08 junior-share rise (arshkon vs scraped) under J1-J4 panel
- T11 requirement_breadth rise under J1-J4 and S1-S4
- AI-mention rise under J1-J4 and S1-S4
"""
import duckdb
import pandas as pd
import numpy as np
import json
import math
from pathlib import Path

OUT_DIR = Path("/home/jihgaboot/gabor/job-research/exploration/artifacts/V1")
T11_PARQ = "/home/jihgaboot/gabor/job-research/exploration/artifacts/T11/T11_posting_features.parquet"

BASE = """source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe = true"""

PANEL_JUNIOR = {
    "J1": "seniority_final IN ('entry')",
    "J2": "seniority_final IN ('entry', 'associate')",
    "J3": "yoe_extracted IS NOT NULL AND yoe_extracted <= 2",
    "J4": "yoe_extracted IS NOT NULL AND yoe_extracted <= 3",
}

PANEL_SENIOR = {
    "S1": "seniority_final IN ('mid-senior', 'director')",
    "S2": "seniority_final = 'director'",
    "S3": "(lower(title_normalized) LIKE '%senior%' OR lower(title_normalized) LIKE '%sr.%' OR lower(title_normalized) LIKE '%lead%' OR lower(title_normalized) LIKE '%staff%' OR lower(title_normalized) LIKE '%principal%')",
    "S4": "yoe_extracted IS NOT NULL AND yoe_extracted >= 5",
}


def ai_mention_pattern_sql():
    # Broad pattern (SQL-friendly OR of LIKEs) — approximate
    # Simpler: use REGEXP_MATCHES
    terms = ["artificial intelligence", "machine learning", "llm", "large language",
             "generative ai", "genai", "copilot", "cursor", "claude", "chatgpt",
             "openai", "anthropic", "gemini", "gpt", "prompt engineering", "rag",
             "retrieval augmented", "langchain"]
    # Use ILIKE OR chain for simplicity. Also include \bai\b and \bml\b via regex.
    like_clauses = [f"lower(description) LIKE '%{t}%'" for t in terms]
    regex_clauses = [
        "regexp_matches(lower(description), '\\\\bai\\\\b')",
        "regexp_matches(lower(description), '\\\\bml\\\\b')",
    ]
    return "(" + " OR ".join(like_clauses + regex_clauses) + ")"


def main():
    con = duckdb.connect()

    # ==== Junior share panel (share of corpus meeting junior criterion per period+source) ====
    print("=== AI-mention rise under junior panel ===")
    ai_pat = ai_mention_pattern_sql()

    # Junior panel: AI-mention share among junior postings
    rows = []
    for pname, pclause in PANEL_JUNIOR.items():
        # Include all SWE in-scope (not just junior) to compute AI-mention share within panel-defined sub
        q = f"""
        SELECT
            source,
            AVG(CASE WHEN {ai_pat} THEN 1.0 ELSE 0.0 END) AS ai_share,
            COUNT(*) AS n
        FROM '/home/jihgaboot/gabor/job-research/data/unified.parquet'
        WHERE {BASE} AND {pclause}
        GROUP BY 1 ORDER BY 1
        """
        df = con.execute(q).df()
        arsh = df[df.source == "kaggle_arshkon"]["ai_share"].iloc[0] if len(df[df.source == "kaggle_arshkon"]) > 0 else np.nan
        asan = df[df.source == "kaggle_asaniczka"]["ai_share"].iloc[0] if len(df[df.source == "kaggle_asaniczka"]) > 0 else np.nan
        scrap_n = df[df.source == "scraped"]["n"].sum()
        # Weighted scraped mean
        if scrap_n > 0:
            scrap = (df[df.source == "scraped"]["ai_share"] * df[df.source == "scraped"]["n"]).sum() / scrap_n
        else:
            scrap = np.nan
        pooled24 = ((arsh * df[df.source == "kaggle_arshkon"]["n"].sum() if not np.isnan(arsh) else 0) +
                    (asan * df[df.source == "kaggle_asaniczka"]["n"].sum() if not np.isnan(asan) else 0))
        pooled24_n = (df[df.source == "kaggle_arshkon"]["n"].sum() + df[df.source == "kaggle_asaniczka"]["n"].sum())
        pooled24_share = pooled24 / pooled24_n if pooled24_n > 0 else np.nan

        d_arsh = scrap - arsh
        d_pool = scrap - pooled24_share
        rows.append({
            "panel": pname,
            "arshkon": float(arsh) if not np.isnan(arsh) else None,
            "pooled24": float(pooled24_share) if not np.isnan(pooled24_share) else None,
            "scraped": float(scrap) if not np.isnan(scrap) else None,
            "delta_arsh_pp": float(d_arsh * 100) if not np.isnan(d_arsh) else None,
            "delta_pool_pp": float(d_pool * 100) if not np.isnan(d_pool) else None,
            "n_arsh": int(df[df.source == "kaggle_arshkon"]["n"].sum()),
            "n_pool": int(pooled24_n),
            "n_scrap": int(scrap_n),
        })

    junior_df = pd.DataFrame(rows)
    print(junior_df.to_string())
    # Unanimity check
    junior_delta_pool_signs = junior_df["delta_pool_pp"].dropna().apply(lambda x: "+" if x > 0 else "-")
    junior_delta_arsh_signs = junior_df["delta_arsh_pp"].dropna().apply(lambda x: "+" if x > 0 else "-")
    print(f"\nJunior AI-mention, arshkon baseline direction signs: {list(junior_delta_arsh_signs)}")
    print(f"Junior AI-mention, pooled baseline direction signs: {list(junior_delta_pool_signs)}")

    # ==== Senior panel (AI-mention rise under S1-S4) ====
    print("\n=== AI-mention rise under senior panel ===")
    rows_s = []
    for pname, pclause in PANEL_SENIOR.items():
        q = f"""
        SELECT
            source,
            AVG(CASE WHEN {ai_pat} THEN 1.0 ELSE 0.0 END) AS ai_share,
            COUNT(*) AS n
        FROM '/home/jihgaboot/gabor/job-research/data/unified.parquet'
        WHERE {BASE} AND {pclause}
        GROUP BY 1 ORDER BY 1
        """
        df = con.execute(q).df()
        arsh = df[df.source == "kaggle_arshkon"]["ai_share"].iloc[0] if len(df[df.source == "kaggle_arshkon"]) > 0 else np.nan
        asan = df[df.source == "kaggle_asaniczka"]["ai_share"].iloc[0] if len(df[df.source == "kaggle_asaniczka"]) > 0 else np.nan
        scrap_n = df[df.source == "scraped"]["n"].sum()
        if scrap_n > 0:
            scrap = (df[df.source == "scraped"]["ai_share"] * df[df.source == "scraped"]["n"]).sum() / scrap_n
        else:
            scrap = np.nan
        pooled24 = ((arsh * df[df.source == "kaggle_arshkon"]["n"].sum() if not np.isnan(arsh) else 0) +
                    (asan * df[df.source == "kaggle_asaniczka"]["n"].sum() if not np.isnan(asan) else 0))
        pooled24_n = (df[df.source == "kaggle_arshkon"]["n"].sum() + df[df.source == "kaggle_asaniczka"]["n"].sum())
        pooled24_share = pooled24 / pooled24_n if pooled24_n > 0 else np.nan

        d_arsh = scrap - arsh
        d_pool = scrap - pooled24_share
        rows_s.append({
            "panel": pname,
            "arshkon": float(arsh) if not np.isnan(arsh) else None,
            "pooled24": float(pooled24_share) if not np.isnan(pooled24_share) else None,
            "scraped": float(scrap) if not np.isnan(scrap) else None,
            "delta_arsh_pp": float(d_arsh * 100) if not np.isnan(d_arsh) else None,
            "delta_pool_pp": float(d_pool * 100) if not np.isnan(d_pool) else None,
            "n_arsh": int(df[df.source == "kaggle_arshkon"]["n"].sum()),
            "n_pool": int(pooled24_n),
            "n_scrap": int(scrap_n),
        })

    senior_df = pd.DataFrame(rows_s)
    print(senior_df.to_string())

    # ==== Requirement_breadth panel ====
    print("\n=== requirement_breadth rise under J1-J4 and S1-S4 ===")
    t11 = con.execute(f"SELECT * FROM '{T11_PARQ}'").df()
    # T11 doesn't have raw seniority_native so use seniority_final + yoe columns
    t11["period_group"] = t11["period"].apply(lambda p: "2024" if str(p).startswith("2024") else "2026")
    # Source column in T11
    t11["arsh"] = t11["source"] == "kaggle_arshkon"
    t11["pool24"] = t11["period_group"] == "2024"
    t11["scrap"] = t11["source"] == "scraped"

    panel_both = {**{f"J1": t11["seniority_final"].isin(["entry"]),
                     "J2": t11["seniority_final"].isin(["entry", "associate"]),
                     "J3": (t11["yoe_numeric"] <= 2) & t11["yoe_numeric"].notna(),
                     "J4": (t11["yoe_numeric"] <= 3) & t11["yoe_numeric"].notna(),
                     "S1": t11["seniority_final"].isin(["mid-senior", "director"]),
                     "S2": t11["seniority_final"] == "director",
                     "S4": (t11["yoe_numeric"] >= 5) & t11["yoe_numeric"].notna()}}
    # S3 needs title_lc
    panel_both["S3"] = t11["title_lc"].fillna("").str.contains(r"senior|\bsr\.|\blead|\bstaff|\bprincipal", regex=True)

    rows_b = []
    for pname, panel_mask in panel_both.items():
        sub = t11[panel_mask]
        arsh_b = sub[sub.arsh]["requirement_breadth"].mean()
        pool_b = sub[sub.pool24]["requirement_breadth"].mean()
        scrap_b = sub[sub.scrap]["requirement_breadth"].mean()
        rows_b.append({
            "panel": pname,
            "n": int(len(sub)),
            "arshkon_breadth": float(arsh_b) if not np.isnan(arsh_b) else None,
            "pooled24_breadth": float(pool_b) if not np.isnan(pool_b) else None,
            "scraped_breadth": float(scrap_b) if not np.isnan(scrap_b) else None,
            "delta_arsh": float(scrap_b - arsh_b) if not (np.isnan(scrap_b) or np.isnan(arsh_b)) else None,
            "delta_pool": float(scrap_b - pool_b) if not (np.isnan(scrap_b) or np.isnan(pool_b)) else None,
        })
    breadth_df = pd.DataFrame(rows_b)
    print(breadth_df.to_string())

    # Verdict
    junior_ai_all_up_pool = all(r["delta_pool_pp"] > 0 for r in rows if r["delta_pool_pp"] is not None)
    junior_ai_all_up_arsh = all(r["delta_arsh_pp"] > 0 for r in rows if r["delta_arsh_pp"] is not None)
    senior_ai_all_up_pool = all(r["delta_pool_pp"] > 0 for r in rows_s if r["delta_pool_pp"] is not None)
    senior_ai_all_up_arsh = all(r["delta_arsh_pp"] > 0 for r in rows_s if r["delta_arsh_pp"] is not None)
    breadth_all_up_pool = all(r["delta_pool"] > 0 for r in rows_b if r["delta_pool"] is not None)
    breadth_all_up_arsh = all(r["delta_arsh"] > 0 for r in rows_b if r["delta_arsh"] is not None)

    print(f"\n=== Panel unanimity verdicts ===")
    print(f"AI-mention junior (4-of-4 up, pooled baseline): {junior_ai_all_up_pool}")
    print(f"AI-mention junior (4-of-4 up, arshkon baseline): {junior_ai_all_up_arsh}")
    print(f"AI-mention senior (4-of-4 up, pooled baseline): {senior_ai_all_up_pool}")
    print(f"AI-mention senior (4-of-4 up, arshkon baseline): {senior_ai_all_up_arsh}")
    print(f"requirement_breadth (7-of-7 up, pooled baseline): {breadth_all_up_pool}")
    print(f"requirement_breadth (7-of-7 up, arshkon baseline): {breadth_all_up_arsh}")

    summary = {
        "junior_ai_panel": rows,
        "senior_ai_panel": rows_s,
        "breadth_panel": rows_b,
        "verdicts": {
            "junior_ai_unanimous_pooled": bool(junior_ai_all_up_pool),
            "junior_ai_unanimous_arshkon": bool(junior_ai_all_up_arsh),
            "senior_ai_unanimous_pooled": bool(senior_ai_all_up_pool),
            "senior_ai_unanimous_arshkon": bool(senior_ai_all_up_arsh),
            "breadth_unanimous_pooled": bool(breadth_all_up_pool),
            "breadth_unanimous_arshkon": bool(breadth_all_up_arsh),
        }
    }
    with open(OUT_DIR / "V1_9_panel_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'V1_9_panel_summary.json'}")


if __name__ == "__main__":
    main()
