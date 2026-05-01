"""T07: metro-level entry-share feasibility per (metro, seniority_def).

For each top metro appearing in both arshkon/pooled_2024 and scraped, compute
the MDE for the entry-share comparison. Adds rows to feasibility_table.csv.
"""
from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pandas as pd
from scipy import stats

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data" / "unified.parquet"
OUT = ROOT / "exploration" / "tables" / "T07"

Z_ALPHA = stats.norm.ppf(0.975)
Z_BETA = stats.norm.ppf(0.80)


def mde_binary(n1, n2, p0):
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    var = p0 * (1 - p0) * (1.0 / n1 + 1.0 / n2)
    return (Z_ALPHA + Z_BETA) * math.sqrt(var)


def verdict_binary(m):
    if math.isnan(m):
        return "underpowered"
    if m < 0.05:
        return "well_powered"
    if m < 0.10:
        return "marginal"
    return "underpowered"


def main():
    con = duckdb.connect()
    # Per-metro, per-source counts of total SWE + entry (J1) + combined junior (J2)
    # + YOE-known + J3/J4.
    q = """
    SELECT metro_area, source,
           COUNT(*) AS n_total,
           SUM(CASE WHEN seniority_final = 'entry' THEN 1 ELSE 0 END) AS n_j1,
           SUM(CASE WHEN seniority_final IN ('entry','associate') THEN 1 ELSE 0 END) AS n_j2,
           SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS n_yoe_known,
           SUM(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted <= 2 THEN 1 ELSE 0 END) AS n_j3,
           SUM(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted <= 3 THEN 1 ELSE 0 END) AS n_j4
    FROM read_parquet(?)
    WHERE source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'
      AND is_swe = true
      AND metro_area IS NOT NULL
    GROUP BY metro_area, source
    """
    df = con.execute(q, [str(DATA)]).df()

    # Pivot
    piv = df.pivot_table(index="metro_area", columns="source",
                         values=["n_total", "n_j1", "n_j2", "n_yoe_known", "n_j3", "n_j4"],
                         aggfunc="sum", fill_value=0).reset_index()
    # Flatten
    piv.columns = ["_".join(str(x) for x in c if x != "").strip("_")
                   if isinstance(c, tuple) else c for c in piv.columns]

    # For each definition, compute MDE at metro level
    # Base p0 = overall junior share in scraped
    # scraped J1 share ≈ 1275/40881 ≈ 0.031
    # J2 ≈ similar; J3 ≈ 4742/27581 ≈ 0.17; J4 ≈ 9307/27581 ≈ 0.34
    p0_map = {"J1": 0.03, "J2": 0.04, "J3": 0.17, "J4": 0.34}
    def_n_col_map = {
        "J1": ("n_total", "n_j1"),
        "J2": ("n_total", "n_j2"),
        "J3": ("n_yoe_known", "n_j3"),
        "J4": ("n_yoe_known", "n_j4"),
    }

    rows = []
    for _, metro in piv.iterrows():
        m = metro["metro_area"]
        for d, (denom_col, _) in def_n_col_map.items():
            # arshkon vs scraped
            n1 = int(metro.get(f"{denom_col}_kaggle_arshkon", 0))
            n2 = int(metro.get(f"{denom_col}_scraped", 0))
            mb = mde_binary(n1, n2, p0_map[d])
            rows.append({
                "analysis_type": "metro_entry_share",
                "comparison": "arshkon_vs_scraped",
                "seniority_def": d,
                "metro_area": m,
                "n_group1": n1,
                "n_group2": n2,
                "MDE_binary": round(mb, 4) if not math.isnan(mb) else None,
                "verdict": verdict_binary(mb),
            })
            # pooled
            npool = n1 + int(metro.get(f"{denom_col}_kaggle_asaniczka", 0))
            mbp = mde_binary(npool, n2, p0_map[d])
            rows.append({
                "analysis_type": "metro_entry_share",
                "comparison": "pooled_2024_vs_scraped",
                "seniority_def": d,
                "metro_area": m,
                "n_group1": npool,
                "n_group2": n2,
                "MDE_binary": round(mbp, 4) if not math.isnan(mbp) else None,
                "verdict": verdict_binary(mbp),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "metro_power.csv", index=False)

    # Summary: count of metros passing each verdict tier per (comparison, def)
    summary = out.groupby(["comparison", "seniority_def", "verdict"]).size().unstack(fill_value=0).reset_index()
    # Add count of metros with n>0 in both groups
    out["both_nonzero"] = (out["n_group1"] > 0) & (out["n_group2"] > 0)
    both = out.groupby(["comparison", "seniority_def"])["both_nonzero"].sum().reset_index(name="metros_both_nonzero")
    summary = summary.merge(both, on=["comparison", "seniority_def"], how="left")
    summary.to_csv(OUT / "metro_power_summary.csv", index=False)
    print("Metro-level power summary:")
    print(summary.to_string(index=False))
    print(f"\nPer-metro detail written to: {OUT / 'metro_power.csv'}")
    print(f"Total metro-level rows: {len(out)}")


if __name__ == "__main__":
    main()
