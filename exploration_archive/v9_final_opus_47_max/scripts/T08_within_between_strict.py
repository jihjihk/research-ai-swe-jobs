"""T08 strict within-between — replicate T06's n=125 arshkon-only panel and the
full-pooled ≥5 panel. The finding is that J3 within-co signal depends on the
sample size filter.
"""

from pathlib import Path
import duckdb
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNIFIED = str(ROOT / "data" / "unified.parquet")
TAB_DIR = ROOT / "exploration" / "tables" / "T08"
BASE = "is_swe AND source_platform='linkedin' AND is_english AND date_flag='ok'"


def compute(c: duckdb.DuckDBPyConnection, min_n: int, sources_2024: list[str] | None, tag: str) -> pd.DataFrame:
    src_filter = ""
    if sources_2024:
        s = ",".join([f"'{x}'" for x in sources_2024])
        src_filter = f" AND (source='scraped' OR source IN ({s}))"
    q = f"""
    WITH scope AS (
      SELECT u.*, CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS pc
      FROM u WHERE 1=1 {src_filter}
    ),
    per AS (
      SELECT company_name_canonical, pc, COUNT(*) AS n
      FROM scope GROUP BY 1,2
    ),
    overlap AS (
      SELECT company_name_canonical FROM per WHERE pc='2024' AND n >= {min_n}
      INTERSECT
      SELECT company_name_canonical FROM per WHERE pc='2026' AND n >= {min_n}
    ),
    in_panel AS (
      SELECT s.* FROM scope s JOIN overlap USING (company_name_canonical)
    )
    SELECT pc, COUNT(*) AS n_total, COUNT(yoe_min_years_llm) AS n_lab,
           SUM(CASE WHEN yoe_min_years_llm<=2 THEN 1 ELSE 0 END)::DOUBLE AS n_j3,
           SUM(CASE WHEN yoe_min_years_llm>=5 THEN 1 ELSE 0 END)::DOUBLE AS n_s4,
           SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END)::DOUBLE AS n_j1,
           SUM(CASE WHEN seniority_final IN ('entry','associate') THEN 1 ELSE 0 END)::DOUBLE AS n_j2,
           (SELECT COUNT(*) FROM overlap) AS n_cos
    FROM in_panel GROUP BY 1 ORDER BY 1
    """
    agg = c.execute(q).df()
    agg["tag"] = tag
    agg["min_n"] = min_n
    return agg


def decompose(c: duckdb.DuckDBPyConnection, min_n: int, sources_2024: list[str] | None, tag: str) -> pd.DataFrame:
    src_filter = ""
    if sources_2024:
        s = ",".join([f"'{x}'" for x in sources_2024])
        src_filter = f" AND (source='scraped' OR source IN ({s}))"
    q = f"""
    WITH scope AS (
      SELECT u.*, CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS pc
      FROM u WHERE 1=1 {src_filter}
    ),
    per AS (
      SELECT company_name_canonical, pc, COUNT(*) AS n
      FROM scope GROUP BY 1,2
    ),
    overlap AS (
      SELECT company_name_canonical FROM per WHERE pc='2024' AND n >= {min_n}
      INTERSECT
      SELECT company_name_canonical FROM per WHERE pc='2026' AND n >= {min_n}
    ),
    in_panel AS (
      SELECT s.* FROM scope s JOIN overlap USING (company_name_canonical)
    ),
    perco AS (
      SELECT company_name_canonical, pc,
             COUNT(*) AS n, COUNT(yoe_min_years_llm) AS n_lab,
             SUM(CASE WHEN yoe_min_years_llm<=2 THEN 1 ELSE 0 END)::DOUBLE AS n_j3,
             SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END)::DOUBLE AS n_j1,
             SUM(CASE WHEN seniority_final IN ('entry','associate') THEN 1 ELSE 0 END)::DOUBLE AS n_j2,
             SUM(CASE WHEN yoe_min_years_llm<=3 THEN 1 ELSE 0 END)::DOUBLE AS n_j4
      FROM in_panel GROUP BY 1,2
    )
    SELECT * FROM perco
    """
    per = c.execute(q).df()
    rows = []
    for m, denom in [("n_j1", "n"), ("n_j2", "n"), ("n_j3", "n_lab"), ("n_j4", "n_lab")]:
        piv = per.pivot_table(index="company_name_canonical", columns="pc", values=[m, denom], aggfunc="first")
        piv = piv.dropna(how="all")
        rate24 = piv[m]["2024"] / piv[denom]["2024"].where(piv[denom]["2024"] > 0)
        rate26 = piv[m]["2026"] / piv[denom]["2026"].where(piv[denom]["2026"] > 0)
        both = pd.DataFrame({"rate_24": rate24, "rate_26": rate26, "n_24": piv[denom]["2024"], "n_26": piv[denom]["2026"]}).dropna()
        if both.empty:
            continue
        within_mean = (both["rate_26"] - both["rate_24"]).mean()
        w = both["n_24"].fillna(0)
        within_w = ((both["rate_26"] - both["rate_24"]) * w).sum() / w.sum() if w.sum() else None
        # aggregate
        agg24 = piv[m]["2024"].sum() / piv[denom]["2024"].sum()
        agg26 = piv[m]["2026"].sum() / piv[denom]["2026"].sum()
        rows.append({
            "tag": tag,
            "metric": m.replace("n_", ""),
            "min_n": min_n,
            "n_cos": len(both),
            "aggregate_24": agg24,
            "aggregate_26": agg26,
            "aggregate_delta_pp": (agg26 - agg24) * 100,
            "within_co_mean_pp": within_mean * 100,
            "within_co_weighted_pp": within_w * 100 if within_w is not None else None,
            "between_co_residual_pp": (agg26 - agg24 - (within_w if within_w is not None else 0)) * 100,
        })
    return pd.DataFrame(rows)


def main() -> None:
    c = duckdb.connect()
    c.execute(f"CREATE VIEW u AS SELECT * FROM '{UNIFIED}' WHERE {BASE}")

    configs = [
        (1, None, "pooled_min1"),
        (3, None, "pooled_min3"),
        (5, None, "pooled_min5"),
        (5, ["kaggle_arshkon"], "arshkon_min5"),
        (5, ["kaggle_asaniczka"], "asaniczka_min5"),
        (10, None, "pooled_min10"),
    ]
    all_panels = []
    all_decomps = []
    for min_n, sources_2024, tag in configs:
        a = compute(c, min_n, sources_2024, tag)
        d = decompose(c, min_n, sources_2024, tag)
        all_panels.append(a)
        all_decomps.append(d)

    panels = pd.concat(all_panels)
    decomps = pd.concat(all_decomps)
    panels.to_csv(TAB_DIR / "overlap_panel_aggregates.csv", index=False)
    decomps.to_csv(TAB_DIR / "overlap_panel_decomposition.csv", index=False)
    print("Panels:")
    print(panels.to_string())
    print("\nDecomposition:")
    print(decomps.to_string())


if __name__ == "__main__":
    main()
