"""T16 sensitivity: re-run decomposition with LLM-labeled-only subset.

V1 confirmed J2 flips between full-corpus (down) and LLM-labeled (up) — any
within-company finding must be reported under both. This script adds the
labeled-only variant to the primary decomposition.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
import pyarrow.parquet as pq

ROOT = Path("/home/jihgaboot/gabor/job-research")
OUT_TBL = ROOT / "exploration/tables/T16"

UNIFIED = ROOT / "data/unified.parquet"
CLEANED = ROOT / "exploration/artifacts/shared/swe_cleaned_text.parquet"
TECH = ROOT / "exploration/artifacts/shared/swe_tech_matrix.parquet"

AI_STRICT = re.compile(
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|vector database|pinecone|huggingface|hugging face)\b",
    re.IGNORECASE,
)
AI_BROAD_EXTRA = re.compile(
    r"\b(ai|artificial intelligence|ml|machine learning|llm|large language model|generative ai|genai|anthropic)\b",
    re.IGNORECASE,
)
ORG_SCOPE = re.compile(
    r"\b(cross[- ]functional|end[- ]to[- ]end|stakeholder|ownership|own the|own this|drive the|drive this|strategic|roadmap|partner with|collaborate across)\b",
    re.IGNORECASE,
)


def open_con():
    con = duckdb.connect()
    con.execute("SET memory_limit='20GB'")
    con.execute("CREATE VIEW u AS SELECT * FROM read_parquet('" + str(UNIFIED) + "')")
    con.execute("CREATE VIEW c AS SELECT * FROM read_parquet('" + str(CLEANED) + "')")
    con.execute("CREATE VIEW t AS SELECT * FROM read_parquet('" + str(TECH) + "')")
    return con


def per_company_aggregate(df):
    grp = df.groupby("company")
    out = pd.DataFrame({
        "n": grp.size(),
        "entry_share_j1": grp["j1"].mean(),
        "entry_share_j2": grp["j2"].mean(),
        "entry_share_j3": grp["j3"].mean(),
        "ai_strict": grp["ai_strict_any"].mean(),
        "ai_broad": grp["ai_broad_any"].mean(),
        "mean_desc_length_raw": grp["description_length"].mean(),
        "mean_desc_length_cleaned": grp["desc_cleaned_len"].mean(),
        "mean_tech_count": grp["tech_count"].mean(),
        "mean_org_scope_per_1k": grp["org_scope_per_1k"].mean(),
        "mean_breadth_raw": grp["breadth_raw"].mean(),
        "mean_breadth_resid": grp["breadth_resid"].mean(),
        "agg_rate": grp["is_aggregator"].mean(),
    })
    return out.reset_index()


def oaxaca_decomp(A, B, metric_col, wcol="n"):
    m = A.merge(B, on="company", suffixes=("_A", "_B"))
    w_a = m[wcol + "_A"].values.astype(float)
    w_b = m[wcol + "_B"].values.astype(float)
    y_a = m[metric_col + "_A"].values.astype(float)
    y_b = m[metric_col + "_B"].values.astype(float)
    valid = ~(np.isnan(y_a) | np.isnan(y_b))
    w_a, w_b, y_a, y_b = w_a[valid], w_b[valid], y_a[valid], y_b[valid]
    sa, sb = w_a.sum(), w_b.sum()
    if sa == 0 or sb == 0:
        return dict(aggregate=np.nan, within=np.nan, between=np.nan)
    aa = (w_a * y_a).sum() / sa
    ab = (w_b * y_b).sum() / sb
    s_a = w_a / sa
    s_b = w_b / sb
    s_avg = (s_a + s_b) / 2
    y_avg = (y_a + y_b) / 2
    within = (s_avg * (y_b - y_a)).sum()
    between = ((s_b - s_a) * y_avg).sum()
    return dict(aggregate=float(ab - aa), within=float(within), between=float(between),
                A_agg=float(aa), B_agg=float(ab), n_companies=int(valid.sum()))


def main():
    con = open_con()
    base = con.execute("""
        SELECT u.uid, u.company_name_canonical AS company, u.source,
               u.is_aggregator, u.seniority_final, u.yoe_extracted,
               u.description_length, u.description AS raw_desc,
               u.llm_extraction_coverage
        FROM u WHERE u.source_platform='linkedin' AND u.is_english
          AND u.date_flag='ok' AND u.is_swe=true
          AND u.company_name_canonical IS NOT NULL AND u.company_name_canonical <> ''
    """).df()

    tcols = [c for c in pq.read_table(str(TECH)).column_names if c != "uid"]
    cols_sum = " + ".join([f"CAST({c} AS INTEGER)" for c in tcols])
    tech = con.execute(f"SELECT uid, ({cols_sum}) AS tech_count FROM t").df()
    base = base.merge(tech, on="uid", how="left")
    ct = con.execute("SELECT uid, description_cleaned AS desc_cleaned FROM c WHERE text_source='llm'").df()
    base = base.merge(ct, on="uid", how="left")
    base["desc_cleaned_len"] = base["desc_cleaned"].fillna("").str.len()

    raw = base["raw_desc"].fillna("")
    base["ai_strict_any"] = raw.str.contains(AI_STRICT).astype(int)
    base["ai_broad_extra"] = raw.str.contains(AI_BROAD_EXTRA).astype(int)
    base["ai_broad_any"] = (base["ai_strict_any"] | base["ai_broad_extra"]).astype(int)
    base["org_scope_count"] = base["desc_cleaned"].fillna("").apply(lambda s: len(ORG_SCOPE.findall(s)))
    base["org_scope_per_1k"] = np.where(base["desc_cleaned_len"] > 0,
                                         base["org_scope_count"] / base["desc_cleaned_len"] * 1000, 0.0)
    CRED = re.compile(r"\b(bachelor'?s?|master'?s?|phd|mba|degree|years of experience|yoe)\b", re.IGNORECASE)
    SOFT = re.compile(r"\b(communication|collaboration|leadership|problem[- ]solving|teamwork|ownership|initiative|self[- ]starter|stakeholder)\b", re.IGNORECASE)
    base["cred_count"] = base["desc_cleaned"].fillna("").apply(lambda s: len(CRED.findall(s)))
    base["soft_count"] = base["desc_cleaned"].fillna("").apply(lambda s: len(SOFT.findall(s)))
    base["breadth_raw"] = (base["tech_count"].fillna(0) + base["org_scope_count"].fillna(0)
                            + base["cred_count"].fillna(0) + base["soft_count"].fillna(0))

    # residualize breadth by cleaned len using overall fit
    x = base["desc_cleaned_len"].fillna(base["desc_cleaned_len"].median()).values.astype(float)
    y = base["breadth_raw"].fillna(base["breadth_raw"].median()).values.astype(float)
    A_mat = np.vstack([np.ones_like(x), x]).T
    coef, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
    base["breadth_resid"] = y - (coef[0] + coef[1] * x)

    base["j1"] = (base["seniority_final"] == "entry").astype(int)
    base["j2"] = base["seniority_final"].isin(["entry", "associate"]).astype(int)
    base["j3"] = (base["yoe_extracted"] <= 2).fillna(False).astype(int)

    # Build overlap panel
    pvt = base.groupby(["company", "source"]).size().unstack("source", fill_value=0)
    pvt["n_arsh"] = pvt.get("kaggle_arshkon", 0)
    pvt["n_scraped"] = pvt.get("scraped", 0)
    overlap = set(pvt[(pvt["n_arsh"] >= 3) & (pvt["n_scraped"] >= 3)].index)
    panel = base[base["company"].isin(overlap)].copy()

    # Labeled-only subset
    lab = panel[panel["llm_extraction_coverage"] == "labeled"].copy()

    df_ar = panel[panel["source"] == "kaggle_arshkon"]
    df_sc = panel[panel["source"] == "scraped"]
    df_ar_lab = lab[lab["source"] == "kaggle_arshkon"]
    df_sc_lab = lab[lab["source"] == "scraped"]

    print(f"full panel: ar={len(df_ar)} sc={len(df_sc)}")
    print(f"labeled-only: ar={len(df_ar_lab)} sc={len(df_sc_lab)}")

    # Need companies with at least one row in both periods under labeled only
    lab_pvt = lab.groupby(["company", "source"]).size().unstack("source", fill_value=0)
    lab_pvt["n_arsh"] = lab_pvt.get("kaggle_arshkon", 0)
    lab_pvt["n_scraped"] = lab_pvt.get("scraped", 0)
    lab_overlap = set(lab_pvt[(lab_pvt["n_arsh"] >= 2) & (lab_pvt["n_scraped"] >= 2)].index)
    print(f"labeled overlap n_cos (>=2/>=2): {len(lab_overlap)}")

    lab_ar = df_ar_lab[df_ar_lab["company"].isin(lab_overlap)]
    lab_sc = df_sc_lab[df_sc_lab["company"].isin(lab_overlap)]

    metrics = [
        "entry_share_j1", "entry_share_j2", "entry_share_j3",
        "ai_strict", "ai_broad",
        "mean_desc_length_raw", "mean_desc_length_cleaned",
        "mean_tech_count", "mean_org_scope_per_1k",
        "mean_breadth_raw", "mean_breadth_resid",
    ]

    A = per_company_aggregate(df_ar)
    B = per_company_aggregate(df_sc)
    full_rows = []
    for m in metrics:
        r = oaxaca_decomp(A, B, m); r["metric"] = m; r["spec"] = "full_panel"
        full_rows.append(r)

    Al = per_company_aggregate(lab_ar)
    Bl = per_company_aggregate(lab_sc)
    lab_rows = []
    for m in metrics:
        r = oaxaca_decomp(Al, Bl, m); r["metric"] = m; r["spec"] = "labeled_only"
        lab_rows.append(r)

    combined = pd.DataFrame(full_rows + lab_rows)
    combined.to_csv(OUT_TBL / "18_decomp_labeled_sensitivity.csv", index=False)

    print("\nFull-panel decomposition:")
    print(pd.DataFrame(full_rows)[["metric", "aggregate", "within", "between", "n_companies"]].to_string(index=False))
    print("\nLabeled-only decomposition:")
    print(pd.DataFrame(lab_rows)[["metric", "aggregate", "within", "between", "n_companies"]].to_string(index=False))


if __name__ == "__main__":
    main()
