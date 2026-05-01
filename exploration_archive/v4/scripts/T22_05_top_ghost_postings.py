"""T22 step 5 — Top ghost-like postings and AI ghostiness drill-down.

1. For entry-level (under combined column AND YOE proxy), pick the 20 most
   ghost-like postings by a composite score:
     composite = 0.4*kitchen_sink_normalized + 0.3*aspiration_ratio_normalized
                 + 0.3*yoe_scope_mismatch_binary
   Report side-by-side: combined-operationalization top20 vs YOE-operationalization top20.
   Count the overlap.

2. AI ghostiness: compute global mean hedge_count and firm_count for AI terms
   vs non-AI rows. Also compute a matched comparison: within the same posting,
   does the AI term appear near more hedges per occurrence than the avg
   keyword in the same posting?

Outputs:
  tables/T22/top20_ghost_entry_combined.csv
  tables/T22/top20_ghost_entry_yoe.csv
  tables/T22/top20_ghost_overlap.txt
  tables/T22/ai_ghostiness_summary.csv
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
FEATS = ROOT / "exploration" / "artifacts" / "T22" / "ghost_indicators_per_posting.parquet"
UNIFIED = ROOT / "data" / "unified.parquet"
TAB = ROOT / "exploration" / "tables" / "T22"


def normalize(s: pd.Series) -> pd.Series:
    if s.std() == 0 or s.isna().all():
        return s * 0
    return (s - s.mean()) / s.std()


def composite_score(df: pd.DataFrame) -> pd.Series:
    ks = normalize(df["kitchen_sink_score"].astype(float))
    ar = normalize(df["aspiration_ratio"].clip(upper=10).astype(float))
    mis = df["yoe_scope_mismatch_combined"].astype(float)
    return 0.4 * ks + 0.3 * ar + 0.3 * mis


def main() -> None:
    df = pd.read_parquet(FEATS)
    print(f"Loaded {len(df):,} rows")

    # 2026 only for the entry ghost showcase (the interesting period)
    df26 = df[df["period2"] == "2026"].copy()
    df26["composite"] = composite_score(df26)

    entry_combined = df26[df26["is_entry_combined"] == 1].nlargest(20, "composite").copy()
    entry_yoe = df26[df26["is_entry_yoe"] == 1].nlargest(20, "composite").copy()

    # Pull title, requirements snippet from unified
    con = duckdb.connect()
    uids = list(set(entry_combined["uid"]) | set(entry_yoe["uid"]))
    uid_list = ",".join(f"'{u}'" for u in uids)
    extras = con.execute(f"""
        SELECT uid, title, company_name_canonical AS company, yoe_extracted,
               COALESCE(NULLIF(description_core_llm,''),description_core,description) AS text_best
        FROM read_parquet('{UNIFIED}')
        WHERE uid IN ({uid_list})
    """).fetchdf()
    extras["snippet"] = extras["text_best"].str.slice(0, 700).str.replace("\n", " ", regex=False)
    extras = extras.drop(columns=["text_best"])

    entry_combined = entry_combined.merge(extras, on="uid", how="left")
    entry_yoe = entry_yoe.merge(extras, on="uid", how="left")

    show_cols = [
        "uid", "title", "company", "yoe_extracted_y",
        "kitchen_sink_score", "aspiration_ratio",
        "hedge_count", "firm_count", "scope_count", "n_distinct_tech",
        "ai_count", "yoe_scope_mismatch_combined",
        "snippet",
    ]
    # normalize yoe columns
    for frame in (entry_combined, entry_yoe):
        if "yoe_extracted_y" not in frame.columns:
            frame["yoe_extracted_y"] = frame.get("yoe_extracted", np.nan)

    entry_combined[show_cols].to_csv(TAB / "top20_ghost_entry_combined.csv", index=False)
    entry_yoe[show_cols].to_csv(TAB / "top20_ghost_entry_yoe.csv", index=False)

    overlap = set(entry_combined["uid"]) & set(entry_yoe["uid"])
    with open(TAB / "top20_ghost_overlap.txt", "w") as f:
        f.write(f"combined-entry top20 vs yoe-proxy-entry top20 overlap: {len(overlap)} uids\n")
        for u in overlap:
            f.write(f"  {u}\n")

    print(f"combined-entry ↔ yoe-proxy-entry top20 overlap: {len(overlap)}")

    # --- AI ghostiness summary ---
    # Global mean hedge/firm counts
    summary = df.groupby("period2").agg(
        n=("uid", "size"),
        mean_hedge=("hedge_count", "mean"),
        mean_firm=("firm_count", "mean"),
        mean_ai_count=("ai_count", "mean"),
        mean_ai_prox_hedge=("ai_prox_hedge", "mean"),
        mean_ai_prox_firm=("ai_prox_firm", "mean"),
        ai_present_rate=("ai_count", lambda s: (s >= 1).mean()),
    ).reset_index()
    # Proxy AI-ghostiness ratio: (ai_hedge / ai_windows) / global hedge rate
    # Within posts that mention AI:
    ai = df[df["ai_windows"] > 0].copy()
    ai["ai_hedge_rate"] = ai["ai_prox_hedge"] / ai["ai_windows"]
    ai["ai_firm_rate"] = ai["ai_prox_firm"] / ai["ai_windows"]
    ai_sum = ai.groupby("period2").agg(
        n_ai=("uid", "size"),
        ai_hedge_rate=("ai_hedge_rate", "mean"),
        ai_firm_rate=("ai_firm_rate", "mean"),
    ).reset_index()
    ai_sum["ai_window_aspiration_ratio"] = ai_sum["ai_hedge_rate"] / ai_sum["ai_firm_rate"].clip(lower=1e-6)

    merged = summary.merge(ai_sum, on="period2")
    merged.to_csv(TAB / "ai_ghostiness_summary.csv", index=False)
    print(merged.to_string())


if __name__ == "__main__":
    main()
