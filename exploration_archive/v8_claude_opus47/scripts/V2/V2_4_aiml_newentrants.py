"""V2.4 — Re-derive AI/ML new-entrant decomposition.

For the ai_ml_engineering archetype:
  - Count employers present only-2026, only-2024, both.
  - Share of 2026 AI/ML posting volume from new vs both-period.

Verify T28's "81% new-entrant-driven" claim.
"""
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

REPO = Path("/home/jihgaboot/gabor/job-research")
OUT_DIR = REPO / "exploration/artifacts/V2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("[V2.4] loading projection...")
    # Use my V2 projection first; fallback to T28's
    v2_proj_path = OUT_DIR / "V2_3_projection.parquet"
    if v2_proj_path.exists():
        proj = pd.read_parquet(v2_proj_path)
        col = "v2_archetype"
    else:
        proj = pd.read_parquet(REPO / "exploration/artifacts/T28/projected_archetypes.parquet")
        col = [c for c in proj.columns if c != "uid"][0]
    print(f"[V2.4] projection rows: {len(proj)}, col: {col}")

    con = duckdb.connect()
    meta = con.execute("""
        SELECT uid, company_name_canonical,
               CASE WHEN period LIKE '2024%' THEN '2024'
                    WHEN period LIKE '2026%' THEN '2026' ELSE 'other' END AS period_bucket
        FROM 'data/unified.parquet'
        WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true
    """).df()
    merged = proj.merge(meta, on="uid", how="inner")
    merged = merged[merged["period_bucket"].isin(["2024", "2026"])]

    # AI/ML archetype filter
    aiml = merged[merged[col] == "ai_ml_engineering"]
    print(f"[V2.4] ai_ml rows: total={len(aiml)}, 2024={(aiml['period_bucket']=='2024').sum()}, 2026={(aiml['period_bucket']=='2026').sum()}")

    # Per-company period set
    comps_24 = set(aiml.loc[(aiml["period_bucket"] == "2024") & (aiml["company_name_canonical"] != ""), "company_name_canonical"])
    comps_26 = set(aiml.loc[(aiml["period_bucket"] == "2026") & (aiml["company_name_canonical"] != ""), "company_name_canonical"])
    only_24 = comps_24 - comps_26
    only_26 = comps_26 - comps_24
    both = comps_24 & comps_26
    print(f"[V2.4] employers only-2024: {len(only_24)}")
    print(f"[V2.4] employers only-2026: {len(only_26)}")
    print(f"[V2.4] employers both:      {len(both)}")

    # Volume share per period
    aiml_26 = aiml[aiml["period_bucket"] == "2026"]
    vol_new = aiml_26[aiml_26["company_name_canonical"].isin(only_26)].shape[0]
    vol_both = aiml_26[aiml_26["company_name_canonical"].isin(both)].shape[0]
    vol_blank = aiml_26[~aiml_26["company_name_canonical"].astype(bool)].shape[0]
    total = len(aiml_26)
    print(f"\n[V2.4] 2026 AI/ML volume decomposition:")
    print(f"  new-only-2026: {vol_new} ({vol_new/total*100:.1f}%)")
    print(f"  both-periods:  {vol_both} ({vol_both/total*100:.1f}%)")
    print(f"  missing company: {vol_blank} ({vol_blank/total*100:.1f}%)")
    print(f"  total 2026 AI/ML: {total}")
    print(f"  T28 claim: 81% new-entrant-driven (935 only-2026, 83 both)")

    # Save
    out = {
        "only_2024_employers": len(only_24),
        "only_2026_employers": len(only_26),
        "both_period_employers": len(both),
        "total_2026_volume": total,
        "volume_new_only": vol_new,
        "volume_both": vol_both,
        "volume_missing_name": vol_blank,
        "pct_new_only_volume": vol_new / total * 100,
        "pct_both_volume": vol_both / total * 100,
    }
    pd.DataFrame([out]).to_csv(OUT_DIR / "V2_4_aiml_newentrant.csv", index=False)
    print("\n[V2.4] saved V2_4_aiml_newentrant.csv")


if __name__ == "__main__":
    main()
