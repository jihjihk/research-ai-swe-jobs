"""Combine main feasibility table with metro-level summary into a single canonical output.

Also builds the (comparison × seniority_def) verdict cross-tab that the orchestrator reads at Gate 1.
"""
from pathlib import Path

import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
OUT = ROOT / "exploration" / "tables" / "T07"

main_feas = pd.read_csv(OUT / "feasibility_table.csv")
metro = pd.read_csv(OUT / "metro_power.csv")

# For metro rows, aggregate to a single row per (comparison, seniority_def) showing
# how many of the 26 scraped metros are well-powered. This is the row that appears
# in the main feasibility table. Individual metro detail remains in metro_power.csv.
metro_summary = (metro.groupby(["comparison", "seniority_def", "verdict"])
                 .size().unstack(fill_value=0).reset_index())
# Count metros with both_nonzero
metro["both_nonzero"] = (metro["n_group1"] > 0) & (metro["n_group2"] > 0)
bn = metro.groupby(["comparison", "seniority_def"])["both_nonzero"].sum().reset_index(name="n_metros_both_nonzero")
metro_summary = metro_summary.merge(bn, on=["comparison", "seniority_def"], how="left")

# Restrict to the 26-metro scraped study frame: only metros where scraped has any SWE
scraped_metros = set(metro[metro["n_group2"] > 0]["metro_area"])
# Also restrict the metro_summary rows to these
metro_inframe = metro[metro["metro_area"].isin(scraped_metros)]

# Recompute verdict summary on the 26-metro frame
metro_summary_inframe = (metro_inframe.groupby(["comparison", "seniority_def", "verdict"])
                         .size().unstack(fill_value=0).reset_index())

# Add a metro-level summary row to main table
for _, r in metro_summary_inframe.iterrows():
    sub = metro_inframe[(metro_inframe["comparison"] == r["comparison"]) &
                        (metro_inframe["seniority_def"] == r["seniority_def"]) &
                        (metro_inframe["n_group1"] > 0) &
                        (metro_inframe["n_group2"] > 0)]
    n1 = int(sub["n_group1"].sum())
    n2 = int(sub["n_group2"].sum())
    wp = int(r.get("well_powered", 0))
    marg = int(r.get("marginal", 0))
    under = int(r.get("underpowered", 0))
    total = wp + marg + under
    # Overall verdict across the 26-metro frame
    if total == 0:
        overall = "underpowered"
    elif wp / total >= 0.5:
        overall = "well_powered"
    elif (wp + marg) / total >= 0.5:
        overall = "marginal"
    else:
        overall = "underpowered"
    extra = f"well_powered_metros={wp}/{total}_scraped_frame"
    main_feas.loc[len(main_feas)] = {
        "analysis_type": "metro_entry_share",
        "comparison": r["comparison"],
        "seniority_def": r["seniority_def"],
        "n_group1": n1,
        "n_group2": n2,
        "MDE_binary": None,
        "MDE_continuous": None,
        "verdict": f"{overall} ({extra})",
    }

main_feas.to_csv(OUT / "feasibility_table.csv", index=False)
print(f"Feasibility table rows: {len(main_feas)}")

# Cross-tab of (comparison × seniority_def) verdicts, for the share analyses
verdict_only = main_feas.copy()
verdict_only["verdict_simple"] = verdict_only["verdict"].astype(str).str.split(" ").str[0]

# junior + senior share cross-tab
js = main_feas[main_feas["analysis_type"].isin(["junior_share", "senior_share"])]
crosstab = js.pivot_table(index="seniority_def", columns="comparison",
                          values="verdict", aggfunc="first")
crosstab.to_csv(OUT / "verdict_crosstab.csv")
print("\nJunior/senior share verdict cross-tab:")
print(crosstab.to_string())

# Key-cell summary: for the orchestrator's Gate 1 decision
# - primary junior definition per comparison
# - primary senior definition per comparison
print("\nAll unique verdicts observed in table:")
print(verdict_only["verdict_simple"].value_counts())
