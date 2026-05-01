"""T08 within-between for J1/J2/J3/J4 panel on returning-companies cohort.

Reproduces the Gate 1 finding under all four junior definitions.
"""

from pathlib import Path
import duckdb
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNIFIED = str(ROOT / "data" / "unified.parquet")
TAB_DIR = ROOT / "exploration" / "tables" / "T08"
BASE = "is_swe AND source_platform='linkedin' AND is_english AND date_flag='ok'"


def run() -> None:
    c = duckdb.connect()
    c.execute(f"CREATE VIEW u AS SELECT * FROM '{UNIFIED}' WHERE {BASE}")
    ret_path = str(ROOT / "exploration/artifacts/shared/returning_companies_cohort.csv")

    # Aggregate panel
    q = f"""
    WITH ret AS (SELECT company_name_canonical FROM read_csv('{ret_path}')),
         scope AS (
           SELECT u.*, CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS pc
           FROM u JOIN ret ON u.company_name_canonical = ret.company_name_canonical
         )
    SELECT pc,
           COUNT(*) AS n,
           COUNT(yoe_min_years_llm) AS n_lab,
           SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END)::DOUBLE AS n_j1,
           SUM(CASE WHEN seniority_final IN ('entry','associate') THEN 1 ELSE 0 END)::DOUBLE AS n_j2,
           SUM(CASE WHEN yoe_min_years_llm <= 2 THEN 1 ELSE 0 END)::DOUBLE AS n_j3,
           SUM(CASE WHEN yoe_min_years_llm <= 3 THEN 1 ELSE 0 END)::DOUBLE AS n_j4
    FROM scope GROUP BY 1 ORDER BY 1
    """
    agg = c.execute(q).df()
    agg["j1"] = agg["n_j1"] / agg["n"]
    agg["j2"] = agg["n_j2"] / agg["n"]
    agg["j3"] = agg["n_j3"] / agg["n_lab"]
    agg["j4"] = agg["n_j4"] / agg["n_lab"]
    agg.to_csv(TAB_DIR / "returning_cohort_jpanel_agg.csv", index=False)
    print("Aggregate on returning cohort:")
    print(agg.to_string())

    # Per-company panel
    q = f"""
    WITH ret AS (SELECT company_name_canonical FROM read_csv('{ret_path}')),
         scope AS (
           SELECT u.*, CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS pc
           FROM u JOIN ret ON u.company_name_canonical = ret.company_name_canonical
         )
    SELECT company_name_canonical, pc,
           COUNT(*) AS n,
           COUNT(yoe_min_years_llm) AS n_lab,
           SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END)::DOUBLE AS n_j1,
           SUM(CASE WHEN seniority_final IN ('entry','associate') THEN 1 ELSE 0 END)::DOUBLE AS n_j2,
           SUM(CASE WHEN yoe_min_years_llm <= 2 THEN 1 ELSE 0 END)::DOUBLE AS n_j3,
           SUM(CASE WHEN yoe_min_years_llm <= 3 THEN 1 ELSE 0 END)::DOUBLE AS n_j4
    FROM scope GROUP BY 1,2
    """
    per = c.execute(q).df()
    per["j1"] = per["n_j1"] / per["n"]
    per["j2"] = per["n_j2"] / per["n"]
    per["j3"] = per["n_j3"] / per["n_lab"].where(per["n_lab"] > 0)
    per["j4"] = per["n_j4"] / per["n_lab"].where(per["n_lab"] > 0)

    # Decomposition: for each company with both periods, Δ
    pivs = {}
    for m in ["j1", "j2", "j3", "j4"]:
        p = per.pivot_table(index="company_name_canonical", columns="pc", values=m, aggfunc="first")
        p = p.dropna(how="all")
        pivs[m] = p
    rows = []
    for m in ["j1", "j2", "j3", "j4"]:
        p = pivs[m]
        both = p.dropna(subset=["2024", "2026"])
        within_mean = (both["2026"] - both["2024"]).mean()
        # Weighted by 2024 postings
        w = per.pivot_table(index="company_name_canonical", columns="pc", values="n", aggfunc="first")
        w_both = w.loc[both.index]
        ww = w_both["2024"].fillna(0)
        within_w = ((both["2026"] - both["2024"]) * ww).sum() / ww.sum() if ww.sum() else None
        aggregate_delta = (agg[agg["pc"] == "2026"][m].iloc[0] - agg[agg["pc"] == "2024"][m].iloc[0])
        rows.append({
            "metric": m,
            "aggregate_delta_pp": aggregate_delta * 100,
            "within_co_mean_delta_pp": within_mean * 100,
            "within_co_weighted_delta_pp": within_w * 100 if within_w is not None else None,
            "between_co_residual_pp": (aggregate_delta - (within_w if within_w is not None else 0)) * 100,
            "n_cos_with_both_periods": int(both.shape[0]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "returning_cohort_jpanel_within_between.csv", index=False)
    print("\nWithin-between decomposition on returning cohort (J-panel):")
    print(df.to_string())


if __name__ == "__main__":
    run()
