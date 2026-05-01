"""V1.5 - Within-LLM-frame J2 flip anomaly re-derivation.

Under arshkon-only-vs-scraped, test whether J2 direction FLIPS between full corpus
and llm_extraction_coverage='labeled' subset.
"""
import duckdb
import json
from pathlib import Path

OUT_DIR = Path("/home/jihgaboot/gabor/job-research/exploration/artifacts/V1")


def main():
    con = duckdb.connect()

    # Default SQL filter
    BASE = """source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe = true"""

    # 1. Full corpus, arshkon vs scraped, J2 share
    print("=== Full corpus: arshkon vs scraped, J2 share ===")
    q_full = f"""
        SELECT
            source,
            period,
            COUNT(*) AS n,
            SUM(CASE WHEN seniority_final IN ('entry', 'associate') THEN 1 ELSE 0 END) AS j2,
            AVG(CASE WHEN seniority_final IN ('entry', 'associate') THEN 1.0 ELSE 0.0 END) AS j2_share
        FROM '/home/jihgaboot/gabor/job-research/data/unified.parquet'
        WHERE {BASE}
        GROUP BY 1, 2 ORDER BY 1, 2
    """
    full_df = con.execute(q_full).df()
    print(full_df.to_string())

    # Compute arshkon -> scraped J2 delta (full corpus)
    arsh_j2_full = full_df[full_df.source == "kaggle_arshkon"]["j2_share"].iloc[0]
    scrap_j2_full = full_df[full_df.source == "scraped"]["j2_share"].mean()  # weighted? use both scraped periods pooled
    scrap_n_full = full_df[full_df.source == "scraped"]["n"].sum()
    scrap_j2_sum_full = full_df[full_df.source == "scraped"]["j2"].sum()
    scrap_j2_full = scrap_j2_sum_full / scrap_n_full
    delta_full = scrap_j2_full - arsh_j2_full
    print(f"\nFull corpus:")
    print(f"  arshkon J2 share: {arsh_j2_full:.4f} ({arsh_j2_full*100:.2f}%)")
    print(f"  scraped J2 share: {scrap_j2_full:.4f} ({scrap_j2_full*100:.2f}%)")
    print(f"  Δ: {delta_full*100:+.2f}pp")

    # 2. llm_extraction_coverage = 'labeled' subset
    print("\n=== llm_extraction_coverage='labeled' subset ===")
    q_llm = f"""
        SELECT
            source,
            period,
            COUNT(*) AS n,
            SUM(CASE WHEN seniority_final IN ('entry', 'associate') THEN 1 ELSE 0 END) AS j2,
            AVG(CASE WHEN seniority_final IN ('entry', 'associate') THEN 1.0 ELSE 0.0 END) AS j2_share
        FROM '/home/jihgaboot/gabor/job-research/data/unified.parquet'
        WHERE {BASE} AND llm_extraction_coverage = 'labeled'
        GROUP BY 1, 2 ORDER BY 1, 2
    """
    llm_df = con.execute(q_llm).df()
    print(llm_df.to_string())

    arsh_j2_llm = llm_df[llm_df.source == "kaggle_arshkon"]["j2_share"].iloc[0] if len(llm_df[llm_df.source == "kaggle_arshkon"]) > 0 else float("nan")
    scrap_n_llm = llm_df[llm_df.source == "scraped"]["n"].sum()
    scrap_j2_sum_llm = llm_df[llm_df.source == "scraped"]["j2"].sum()
    scrap_j2_llm = scrap_j2_sum_llm / scrap_n_llm if scrap_n_llm > 0 else float("nan")
    delta_llm = scrap_j2_llm - arsh_j2_llm
    print(f"\nLLM-labeled subset:")
    print(f"  arshkon J2 share: {arsh_j2_llm:.4f} ({arsh_j2_llm*100:.2f}%)")
    print(f"  scraped J2 share: {scrap_j2_llm:.4f} ({scrap_j2_llm*100:.2f}%)")
    print(f"  Δ: {delta_llm*100:+.2f}pp")

    # Direction comparison
    print("\n=== Direction flip check ===")
    print(f"  Full corpus direction: {'UP' if delta_full > 0 else 'DOWN' if delta_full < 0 else 'FLAT'}")
    print(f"  LLM-labeled direction: {'UP' if delta_llm > 0 else 'DOWN' if delta_llm < 0 else 'FLAT'}")
    flipped = (delta_full < 0 and delta_llm > 0) or (delta_full > 0 and delta_llm < 0)
    print(f"  FLIPPED: {flipped}")

    # Mechanism hypothesis: are LLM-labeled postings systematically different
    # in length, AI-mention rate, etc.?
    print("\n=== Mechanism: LLM-labeled vs non-labeled posting characteristics ===")
    q_mech = f"""
        SELECT
            source,
            period,
            llm_extraction_coverage,
            COUNT(*) AS n,
            AVG(description_length) AS avg_len,
            AVG(CASE WHEN description LIKE '%artificial intelligence%' OR description LIKE '%machine learning%'
                      OR description LIKE '%AI %' OR description LIKE '%ML %' THEN 1.0 ELSE 0.0 END) AS ai_share,
            AVG(CASE WHEN seniority_final IN ('entry', 'associate') THEN 1.0 ELSE 0.0 END) AS j2_share
        FROM '/home/jihgaboot/gabor/job-research/data/unified.parquet'
        WHERE {BASE}
        GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
    """
    mech_df = con.execute(q_mech).df()
    print(mech_df.to_string())

    summary = {
        "full_corpus": {
            "arshkon_j2_share": float(arsh_j2_full),
            "scraped_j2_share": float(scrap_j2_full),
            "delta_pp": float(delta_full * 100),
        },
        "llm_labeled_subset": {
            "arshkon_j2_share": float(arsh_j2_llm),
            "scraped_j2_share": float(scrap_j2_llm),
            "delta_pp": float(delta_llm * 100),
        },
        "direction_flipped": bool(flipped),
        "mechanism_table": mech_df.to_dict("records"),
    }
    with open(OUT_DIR / "V1_5_llm_flip_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'V1_5_llm_flip_summary.json'}")


if __name__ == "__main__":
    main()
