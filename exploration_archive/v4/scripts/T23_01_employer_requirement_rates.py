"""T23 step 1 — AI requirement rates in SWE postings.

Computes:
  - AI-as-tool requirement rate (copilot/cursor/llm/prompt-engineering/langchain/mcp)
  - AI-as-domain requirement rate (ML/DL/NLP/CV/model training/transformer/
    embedding/fine-tuning)
  - AI-general requirement rate (\\bai\\b|artificial intelligence)
  - Agentic-specific rate (high precision)
  - ai_agent_phrase rate
  - rag_phrase rate

By period × seniority (combined best-available AND YOE proxy), with
aggregator-exclusion as a primary sensitivity.

Output:
  tables/T23/ai_requirement_rates.csv
  tables/T23/ai_requirement_rates_direct_only.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
FEATS = ROOT / "exploration" / "artifacts" / "T22" / "ghost_indicators_per_posting.parquet"
TAB = ROOT / "exploration" / "tables" / "T23"
TAB.mkdir(parents=True, exist_ok=True)


def compute(df: pd.DataFrame, group_cols: list[str], label: str) -> pd.DataFrame:
    cols = [
        "ai_tool",
        "ai_domain",
        "ai_general",
        "agentic",
        "ai_agent_phrase",
        "rag_phrase",
    ]
    g = df.groupby(group_cols, dropna=False).agg(
        n=("uid", "size"),
        **{f"{c}_rate": (c, "mean") for c in cols},
    ).reset_index()
    # Combined "any AI" = any of the categorical AI patterns
    df2 = df.copy()
    df2["ai_any"] = (
        df2[cols].sum(axis=1) > 0
    ).astype(int)
    g2 = df2.groupby(group_cols, dropna=False).agg(
        any_ai_rate=("ai_any", "mean"),
    ).reset_index()
    out = g.merge(g2, on=group_cols)
    out["scope"] = label
    return out


def main():
    df = pd.read_parquet(FEATS)
    print(f"Loaded {len(df):,} rows")

    # Seniority buckets
    def bucket(col):
        s = df[col].fillna("unknown").str.lower()
        return np.where(
            s == "entry", "entry",
            np.where(s.isin(["mid-senior", "senior", "staff", "principal"]), "mid-senior",
                     np.where(s == "associate", "associate", "other")),
        )
    df["sen_combined"] = bucket("seniority_best_available")
    df["sen_final"] = bucket("seniority_final")
    df["sen_yoe"] = np.where(
        df["yoe_extracted"].notna() & (df["yoe_extracted"] <= 2), "entry",
        np.where(df["yoe_extracted"] >= 5, "mid-senior",
                 np.where(df["yoe_extracted"].notna(), "associate", "unknown"))
    )

    all_rows = pd.concat([
        compute(df, ["period2", "sen_combined"], "combined"),
        compute(df, ["period2", "sen_yoe"], "yoe").rename(columns={"sen_yoe": "sen_combined"}),
        compute(df, ["period2", "sen_final"], "final").rename(columns={"sen_final": "sen_combined"}),
    ])
    all_rows.to_csv(TAB / "ai_requirement_rates.csv", index=False)

    # Direct only
    direct = df[df["is_aggregator"] == False].copy()
    direct["sen_combined"] = bucket("seniority_best_available")[direct.index]
    direct_out = compute(direct, ["period2", "sen_combined"], "combined_direct")
    direct_out.to_csv(TAB / "ai_requirement_rates_direct_only.csv", index=False)

    # Period-only (for headline)
    period = compute(df, ["period2"], "overall")
    period["sen_combined"] = "all"
    period.to_csv(TAB / "ai_requirement_rates_period_only.csv", index=False)

    print(period.round(3).to_string())


if __name__ == "__main__":
    main()
