"""V2.6 — T16 within-company vs between-company Oaxaca decomposition.

Build 240-co arshkon∩scraped overlap panel. For each metric (AI-strict,
breadth-resid, J2 entry share, desc cleaned length), compute aggregate,
within-company, between-company components. Verify T16's "102% within" claim
on AI-strict.

Uses symmetric Oaxaca:
  aggregate = Σw_sc·y_sc/Σw_sc − Σw_ar·y_ar/Σw_ar
  within = Σ s_avg·(y_sc − y_ar)
  between = Σ (s_sc − s_ar)·y_avg
"""
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path("/home/jihgaboot/gabor/job-research")
OUT_DIR = REPO / "exploration/artifacts/V2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AI_STRICT = r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|vector database|pinecone|huggingface|hugging face)\b"


def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    # Features per SWE LinkedIn row using cleaned text when available
    q = f"""
    SELECT
      uid, source, period, company_name_canonical, seniority_final,
      CAST(regexp_matches(lower(COALESCE(description_core_llm, description, '')), '{AI_STRICT}') AS INTEGER) AS ai_strict,
      LENGTH(COALESCE(description_core_llm, description, '')) AS desc_len_cleaned,
      LENGTH(description) AS desc_len_raw
    FROM 'data/unified.parquet'
    WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true
      AND company_name_canonical != ''
    """
    print("[V2.6] loading SWE rows...")
    df = con.execute(q).df()
    # Arshkon and scraped
    ar = df[df["source"] == "kaggle_arshkon"].copy()
    sc = df[df["source"] == "scraped"].copy()
    print(f"[V2.6] arshkon rows: {len(ar)}, scraped rows: {len(sc)}")

    # Panel: cos with >=3 arshkon AND >=3 scraped
    ar_cnt = ar.groupby("company_name_canonical").size()
    sc_cnt = sc.groupby("company_name_canonical").size()
    eligible = ar_cnt[ar_cnt >= 3].index.intersection(sc_cnt[sc_cnt >= 3].index)
    print(f"[V2.6] overlap panel size: {len(eligible)}")

    ar_p = ar[ar["company_name_canonical"].isin(eligible)]
    sc_p = sc[sc["company_name_canonical"].isin(eligible)]

    # Metrics per company per period
    def agg_metrics(d):
        return d.groupby("company_name_canonical").agg(
            n=("uid", "count"),
            ai_strict=("ai_strict", "mean"),
            desc_len=("desc_len_cleaned", "mean"),
            j2_share=("seniority_final", lambda s: s.isin(["entry", "associate"]).mean()),
        )

    ar_m = agg_metrics(ar_p)
    sc_m = agg_metrics(sc_p)
    merged = ar_m.join(sc_m, lsuffix="_ar", rsuffix="_sc", how="inner")
    merged = merged.dropna(how="any")
    print(f"[V2.6] usable panel: {len(merged)}")

    # Oaxaca decomposition
    def oaxaca(df, col):
        y_ar = df[f"{col}_ar"].values
        y_sc = df[f"{col}_sc"].values
        w_ar = df["n_ar"].values
        w_sc = df["n_sc"].values
        s_ar = w_ar / w_ar.sum()
        s_sc = w_sc / w_sc.sum()
        agg = (y_sc * s_sc).sum() - (y_ar * s_ar).sum()
        s_avg = (s_ar + s_sc) / 2
        y_avg = (y_ar + y_sc) / 2
        within = (s_avg * (y_sc - y_ar)).sum()
        between = ((s_sc - s_ar) * y_avg).sum()
        return agg, within, between

    rows = []
    for col in ["ai_strict", "desc_len", "j2_share"]:
        agg, within, between = oaxaca(merged, col)
        pct_w = within / agg * 100 if abs(agg) > 1e-9 else float("nan")
        rows.append({
            "metric": col,
            "aggregate": agg, "within": within, "between": between,
            "pct_within": pct_w,
        })
    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT_DIR / "V2_6_oaxaca.csv", index=False)
    print("\n[V2.6] Oaxaca (arshkon vs scraped, 240-co panel):")
    print(rdf.to_string(index=False))
    print(f"\n  T16 claims: AI-strict within 12.36pp of 12.09pp agg = 102%; breadth_resid within 102%; J2 direction small")


if __name__ == "__main__":
    main()
