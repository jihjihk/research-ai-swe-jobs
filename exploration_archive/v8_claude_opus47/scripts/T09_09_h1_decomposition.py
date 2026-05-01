"""T09 Step 9: Decomposition of the aggregate J2 entry share change.

Classic between/within decomposition:
  aggregate_2024 = sum_i s_i,2024 * p_i,2024
  aggregate_2026 = sum_i s_i,2026 * p_i,2026

Delta = sum_i (s_i,2026 - s_i,2024) * p_i,2024      (within: shifts in entry share within each archetype, holding share fixed)
      + sum_i (p_i,2026 - p_i,2024) * s_i,2024      (between: shifts in archetype share, holding entry share fixed)
      + sum_i (s_i,2026 - s_i,2024) * (p_i,2026 - p_i,2024)  (interaction)

Where:
  s_i = J2 entry+associate share of KNOWN seniority within archetype i
  p_i = archetype i's share-of-SWE in period

Pooled-2024 baseline = average of 2024-01 and 2024-04.

Writes:
  exploration/tables/T09/h1_decomposition_j2.csv
"""
import numpy as np
import pandas as pd
import duckdb

OUTDIR = "exploration/artifacts/T09"
TABLES = "exploration/tables/T09"


def main():
    df = pd.read_parquet(f"{OUTDIR}/sample_with_assignments.parquet")
    names = pd.read_csv(f"{OUTDIR}/archetype_names.csv")
    name_map = dict(zip(names.archetype_id, names.archetype_name))
    df["archetype_name"] = df["topic_reduced"].map(name_map).fillna("unknown")
    df["period_bucket"] = df["period"].map({
        "2024-04": "2024",
        "2024-01": "2024",
        "2026-03": "2026",
        "2026-04": "2026",
    })

    # Archetype share p_i per period (based on known-seniority pool? or full sample?)
    # Use full sample for p_i (archetype share of SWE within each period)
    p_share = pd.crosstab(df.archetype_name, df.period_bucket, normalize="columns")

    # Entry share s_i per archetype per period (denominator = known-seniority)
    known = df[df.seniority_final != "unknown"]
    known["is_j2"] = known.seniority_final.isin(["entry", "associate"]).astype(int)
    s_share = known.groupby(["archetype_name", "period_bucket"])["is_j2"].mean().unstack()

    # Restrict to archetypes present in both periods
    common = sorted(set(p_share.index) & set(s_share.index))
    p = p_share.loc[common].reindex(columns=["2024", "2026"]).fillna(0.0)
    s = s_share.loc[common].reindex(columns=["2024", "2026"]).fillna(0.0)

    # Aggregate entry share in the SAMPLE (not overall unified, just our 8k)
    # using known-only denominator per period
    agg_2024 = known[known.period_bucket == "2024"]["is_j2"].mean()
    agg_2026 = known[known.period_bucket == "2026"]["is_j2"].mean()
    delta_agg = agg_2026 - agg_2024
    print(f"Aggregate J2 share 2024 (sample): {agg_2024:.3f}")
    print(f"Aggregate J2 share 2026 (sample): {agg_2026:.3f}")
    print(f"Aggregate delta: {delta_agg:+.3f}")

    # Decomposition with known-seniority weighted shares
    p_known = (known.groupby(["archetype_name", "period_bucket"]).size()
                   .unstack().fillna(0))
    p_known = p_known.div(p_known.sum(axis=0), axis=1)
    p_k = p_known.loc[common].reindex(columns=["2024", "2026"]).fillna(0.0)

    # Rebuild s using known-only
    rows = []
    s1 = s["2024"]
    s2 = s["2026"]
    p1 = p_k["2024"]
    p2 = p_k["2026"]
    within = ((s2 - s1) * p1)
    between = ((p2 - p1) * s1)
    interaction = ((s2 - s1) * (p2 - p1))
    for arch in common:
        rows.append({
            "archetype": arch,
            "p_2024_share_of_known": round(p1[arch], 3),
            "p_2026_share_of_known": round(p2[arch], 3),
            "p_delta": round(p2[arch] - p1[arch], 3),
            "s_2024_j2_share": round(s1[arch], 3),
            "s_2026_j2_share": round(s2[arch], 3),
            "s_delta": round(s2[arch] - s1[arch], 3),
            "within_contribution": round(within[arch], 4),
            "between_contribution": round(between[arch], 4),
            "interaction_contribution": round(interaction[arch], 4),
            "total_contribution": round(within[arch] + between[arch] + interaction[arch], 4),
        })
    dec = pd.DataFrame(rows).sort_values("total_contribution", ascending=False)
    dec.to_csv(f"{TABLES}/h1_decomposition_j2.csv", index=False)

    total_within = within.sum()
    total_between = between.sum()
    total_interaction = interaction.sum()
    implied = total_within + total_between + total_interaction
    implied_agg = (s1 * p1).sum()
    implied_2026 = (s2 * p2).sum()
    print(f"\nDecomposition of J2 share change (known-seniority):")
    print(f"  Implied 2024 (sum s*p): {implied_agg:.4f}")
    print(f"  Implied 2026 (sum s*p): {implied_2026:.4f}")
    print(f"  Sum of contributions: {implied:.4f}")
    print(f"  Total within  (s shift): {total_within:+.4f}")
    print(f"  Total between (p shift): {total_between:+.4f}")
    print(f"  Total interaction      : {total_interaction:+.4f}")
    print(f"\n  Within pct of total:  {100 * total_within / implied:+.1f}%")
    print(f"  Between pct of total: {100 * total_between / implied:+.1f}%")

    # Summary row
    summary = pd.DataFrame([{
        "period_a": "2024 (pooled)",
        "period_b": "2026 (pooled)",
        "agg_j2_share_a": round(agg_2024, 4),
        "agg_j2_share_b": round(agg_2026, 4),
        "delta_agg": round(delta_agg, 4),
        "within_total": round(total_within, 4),
        "between_total": round(total_between, 4),
        "interaction_total": round(total_interaction, 4),
        "within_pct_of_total": round(100 * total_within / implied, 1),
        "between_pct_of_total": round(100 * total_between / implied, 1),
        "interaction_pct_of_total": round(100 * total_interaction / implied, 1),
    }])
    summary.to_csv(f"{TABLES}/h1_decomposition_summary.csv", index=False)
    print(summary.to_string(index=False))

    print("\nTop archetypes by decomposition contribution:")
    print(dec.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
