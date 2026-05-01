"""V2 Phase A — H_w3_2: Independent re-derivation of T18 cross-occupation DiD.

Claim (T18): SWE − CONTROL DiD on ai_strict_prev = +14.02 pp, 95% CI [+13.67, +14.37]
control drift +0.17 pp; SWE rises 0.74% → 14.93% (+14.19 pp).

V2 independent approach: pull all LinkedIn English rows with is_swe|is_swe_adjacent|is_control;
apply V1-validated ai_strict on raw description (same input as T18 "raw-description" primary).

Phase E extension: robust to alt control definitions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNIFIED = str(ROOT / "data" / "unified.parquet")
VALIDATED = ROOT / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json"
OUT = ROOT / "exploration" / "tables" / "V2"
OUT.mkdir(parents=True, exist_ok=True)

with open(VALIDATED) as f:
    patj = json.load(f)
AI_V1 = patj["v1_rebuilt_patterns"]["ai_strict_v1_rebuilt"]["pattern"]
rx = re.compile(AI_V1, flags=re.IGNORECASE)


def fetch() -> pd.DataFrame:
    q = f"""
    SELECT uid, title, title_normalized, company_name_canonical,
      is_swe, is_swe_adjacent, is_control, source, period,
      CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS era,
      LOWER(COALESCE(description, '')) AS txt
    FROM '{UNIFIED}'
    WHERE source_platform='linkedin' AND is_english=TRUE AND date_flag='ok'
      AND (is_swe=TRUE OR is_swe_adjacent=TRUE OR is_control=TRUE)
    """
    con = duckdb.connect()
    df = con.execute(q).df()
    con.close()
    print(f"Loaded {len(df):,} rows")
    return df


def apply_ai(df):
    df = df.copy()
    tx = df["txt"].fillna("").to_numpy()
    df["ai"] = np.fromiter((1 if rx.search(t) else 0 for t in tx), dtype=np.int8, count=len(tx))
    return df


def grp(df):
    """Occupation group."""
    g = np.select(
        [df["is_swe"].fillna(False), df["is_swe_adjacent"].fillna(False), df["is_control"].fillna(False)],
        ["SWE", "SWE_ADJACENT", "CONTROL"],
        default="OTHER",
    )
    return g


def prev_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["grp"] = grp(df)
    out = df.groupby(["grp", "era"]).agg(n=("uid", "size"), ai=("ai", "mean")).reset_index()
    out["ai_pct"] = out["ai"] * 100
    return out


def did(pt: pd.DataFrame, g1: str, g2: str) -> dict:
    c = pt.set_index(["grp", "era"])
    d1 = c.loc[(g1, "2026"), "ai"] - c.loc[(g1, "2024"), "ai"]
    d2 = c.loc[(g2, "2026"), "ai"] - c.loc[(g2, "2024"), "ai"]
    did_val = d1 - d2
    # Proportion-SE pooled variance
    n1_24 = c.loc[(g1, "2024"), "n"]
    n1_26 = c.loc[(g1, "2026"), "n"]
    n2_24 = c.loc[(g2, "2024"), "n"]
    n2_26 = c.loc[(g2, "2026"), "n"]
    p1_24 = c.loc[(g1, "2024"), "ai"]
    p1_26 = c.loc[(g1, "2026"), "ai"]
    p2_24 = c.loc[(g2, "2024"), "ai"]
    p2_26 = c.loc[(g2, "2026"), "ai"]
    var = (
        p1_24 * (1 - p1_24) / n1_24
        + p1_26 * (1 - p1_26) / n1_26
        + p2_24 * (1 - p2_24) / n2_24
        + p2_26 * (1 - p2_26) / n2_26
    )
    se = np.sqrt(var)
    return {
        "group_A": g1,
        "group_B": g2,
        "deltaA_pp": float(d1 * 100),
        "deltaB_pp": float(d2 * 100),
        "DiD_pp": float(did_val * 100),
        "SE_pp": float(se * 100),
        "CI_low_pp": float((did_val - 1.96 * se) * 100),
        "CI_high_pp": float((did_val + 1.96 * se) * 100),
    }


def main():
    df = apply_ai(fetch())

    # Primary: all SWE vs CONTROL
    pt = prev_table(df)
    print("\nBasic prevalence table:")
    print(pt.to_string(index=False))

    did_main = did(pt, "SWE", "CONTROL")
    print("\nPrimary DiD SWE − CONTROL (V1-validated ai_strict):")
    print(json.dumps(did_main, indent=2))

    did_adj = did(pt, "SWE", "SWE_ADJACENT")
    print("\nDiD SWE − SWE_ADJACENT:")
    print(json.dumps(did_adj, indent=2))

    # Phase E — alt control definitions
    # Need title-level inspection of control titles
    # The scripts use title_normalized heuristics; pull a sample
    print("\nTop control titles (2024 and 2026):")
    ctl = df[df["is_control"].fillna(False)].copy()
    top = ctl.groupby("title_normalized").size().sort_values(ascending=False).head(40)
    print(top)

    # Alt 1: Drop data_analyst + financial_analyst
    # Detect via substring match on title_normalized
    def _cat(t):
        t = str(t).lower()
        if any(k in t for k in ["data analyst", "business analyst", "bi analyst", "business intelligence analyst"]):
            return "data_analyst"
        if any(k in t for k in ["financial analyst", "finance analyst", "investment analyst"]):
            return "financial_analyst"
        if any(k in t for k in ["nurse", "rn ", " rn,", "registered nurse", "nursing"]):
            return "nurse"
        if any(k in t for k in ["accountant", "accounting"]):
            return "accountant"
        if "civil engineer" in t:
            return "civil_engineer"
        if "mechanical engineer" in t:
            return "mechanical_engineer"
        if "electrical engineer" in t:
            return "electrical_engineer"
        if "industrial engineer" in t:
            return "industrial_engineer"
        return "other"

    ctl["sub"] = ctl["title_normalized"].apply(_cat)
    print("\nControl sub breakdown:", ctl["sub"].value_counts().to_dict())

    # Alt a: drop data_analyst + financial_analyst
    ctl_alt = ctl[~ctl["sub"].isin(["data_analyst", "financial_analyst"])].copy()
    pt_alt = pd.concat([
        df[df["is_swe"].fillna(False)].assign(grp="SWE"),
        ctl_alt.assign(grp="CONTROL")
    ])
    pt_alt_agg = pt_alt.groupby(["grp", "era"]).agg(n=("uid", "size"), ai=("ai", "mean")).reset_index()
    print("\nAlt A (drop data_analyst + financial_analyst):")
    print(pt_alt_agg.to_string(index=False))
    did_altA = did(pt_alt_agg, "SWE", "CONTROL")
    print(json.dumps(did_altA, indent=2))

    # Alt b: drop healthcare/nurse
    ctl_b = ctl[~ctl["sub"].isin(["nurse"])].copy()
    pt_b = pd.concat([
        df[df["is_swe"].fillna(False)].assign(grp="SWE"),
        ctl_b.assign(grp="CONTROL")
    ])
    pt_b_agg = pt_b.groupby(["grp", "era"]).agg(n=("uid", "size"), ai=("ai", "mean")).reset_index()
    did_altB = did(pt_b_agg, "SWE", "CONTROL")
    print("\nAlt B (drop nurse from control):")
    print(pt_b_agg.to_string(index=False))
    print(json.dumps(did_altB, indent=2))

    # Alt c: Restrict control to manual-work occupations (civil/mech/elect/industrial engineer + nurse + accountant)
    ctl_c = ctl[ctl["sub"].isin(["nurse", "accountant", "civil_engineer", "mechanical_engineer",
                                  "electrical_engineer", "industrial_engineer"])].copy()
    pt_c = pd.concat([
        df[df["is_swe"].fillna(False)].assign(grp="SWE"),
        ctl_c.assign(grp="CONTROL")
    ])
    pt_c_agg = pt_c.groupby(["grp", "era"]).agg(n=("uid", "size"), ai=("ai", "mean")).reset_index()
    did_altC = did(pt_c_agg, "SWE", "CONTROL")
    print("\nAlt C (control = manual/physical-work occupations only):")
    print(pt_c_agg.to_string(index=False))
    print(json.dumps(did_altC, indent=2))

    # Alt d: Drop title_lookup_llm tier (Phase F alternative explanation)
    df_notierllm = df.copy()
    q2 = f"""
    SELECT uid, swe_classification_tier FROM '{UNIFIED}'
    WHERE source_platform='linkedin' AND is_english=TRUE AND date_flag='ok'
    """
    con = duckdb.connect()
    tier = con.execute(q2).df()
    con.close()
    df_notierllm = df_notierllm.merge(tier, on="uid", how="left")
    # For SWE rows, drop those where swe_classification_tier = 'title_lookup_llm'
    sw_drop = df_notierllm[df_notierllm["is_swe"].fillna(False)]
    print("\nSWE tier breakdown:", sw_drop["swe_classification_tier"].value_counts().to_dict())
    df_no_tierllm = df_notierllm[
        ~((df_notierllm["is_swe"].fillna(False)) & (df_notierllm["swe_classification_tier"] == "title_lookup_llm"))
    ].copy()
    df_no_tierllm["grp"] = grp(df_no_tierllm)
    pt_d = df_no_tierllm.groupby(["grp", "era"]).agg(n=("uid", "size"), ai=("ai", "mean")).reset_index()
    did_altD = did(pt_d, "SWE", "CONTROL")
    print("\nAlt D (drop title_lookup_llm tier from SWE):")
    print(pt_d.to_string(index=False))
    print(json.dumps(did_altD, indent=2))

    # Save all
    all_did = [did_main, did_adj, did_altA, did_altB, did_altC, did_altD]
    names = ["primary_SWE-CTL", "SWE-ADJ", "altA_no_data_fin_analyst", "altB_no_nurse",
             "altC_manual_only", "altD_no_title_lookup_llm"]
    df_did = pd.DataFrame(all_did)
    df_did["variant"] = names
    df_did.to_csv(OUT / "H_w3_2_did_robustness.csv", index=False)
    print("\nSaved:", OUT / "H_w3_2_did_robustness.csv")


if __name__ == "__main__":
    main()
