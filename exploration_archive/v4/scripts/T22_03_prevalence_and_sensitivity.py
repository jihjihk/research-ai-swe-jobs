"""T22 step 3 — Ghost indicator prevalence and sensitivity.

Outputs:
  tables/T22/prevalence_by_period_seniority.csv
  tables/T22/prevalence_by_period_seniority_yoe.csv
  tables/T22/prevalence_aggregator_vs_direct.csv
  tables/T22/ai_vs_nonai_aspiration.csv
  tables/T22/aspiration_by_ai_category.csv
  tables/T22/industry_ghostiness.csv
  tables/T22/yoe_scope_mismatch_by_operationalization.csv
  tables/T22/credential_impossibility_by_period.csv
  figures/T22/aspiration_ratio_by_period.png
  figures/T22/kitchen_sink_by_period.png
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
FEATS = ROOT / "exploration" / "artifacts" / "T22" / "ghost_indicators_per_posting.parquet"
TAB = ROOT / "exploration" / "tables" / "T22"
FIG = ROOT / "exploration" / "figures" / "T22"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def load() -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{FEATS}')").fetchdf()
    return df


def bucket(df: pd.DataFrame, col: str) -> pd.Series:
    """Map seniority_best_available / seniority_final into entry/mid-senior/other."""
    s = df[col].fillna("unknown").str.lower()
    out = np.where(
        s == "entry", "entry",
        np.where(s.isin(["mid-senior", "senior", "staff", "principal"]), "mid-senior",
                 np.where(s == "associate", "associate", "other")),
    )
    return pd.Series(out, index=df.index)


def prevalence(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Compute prevalence of each indicator per group."""
    indicators = {
        "hedge_any": "hedge_any",
        "firm_any": "firm_any",
        "kitchen_sink_high_score_ge12": ("kitchen_sink_score", lambda s: (s >= 12).astype(int)),
        "kitchen_sink_high_score_ge20": ("kitchen_sink_score", lambda s: (s >= 20).astype(int)),
        "aspiration_ratio_gt1": ("aspiration_ratio", lambda s: (s > 1.0).astype(int)),
        "aspiration_ratio_gt2": ("aspiration_ratio", lambda s: (s > 2.0).astype(int)),
        "yoe_scope_mismatch_combined": "yoe_scope_mismatch_combined",
        "yoe_scope_mismatch_yoe": "yoe_scope_mismatch_yoe",
        "credential_contradiction": "credential_contradiction",
        "ai_count_ge1": ("ai_count", lambda s: (s >= 1).astype(int)),
        "ai_count_ge2": ("ai_count", lambda s: (s >= 2).astype(int)),
        "ai_tool": "ai_tool",
        "ai_domain": "ai_domain",
        "ai_general": "ai_general",
        "agentic": "agentic",
        "ai_agent_phrase": "ai_agent_phrase",
        "rag_phrase": "rag_phrase",
    }

    pieces = []
    for name, spec in indicators.items():
        if isinstance(spec, tuple):
            col, fn = spec
            tmp = df[group_cols].copy()
            tmp["_v"] = fn(df[col])
        else:
            tmp = df[group_cols].copy()
            tmp["_v"] = df[spec].astype(int)
        agg = tmp.groupby(group_cols, dropna=False).agg(
            n=("_v", "size"), rate=("_v", "mean"), hits=("_v", "sum")
        ).reset_index()
        agg["indicator"] = name
        pieces.append(agg)
    return pd.concat(pieces, ignore_index=True)


def continuous_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Mean/median of continuous ghost scores per group."""
    cols = [
        "hedge_count",
        "firm_count",
        "aspiration_ratio",
        "n_distinct_tech",
        "scope_count",
        "kitchen_sink_score",
        "ai_count",
        "senior_scope_n",
    ]
    g = df.groupby(group_cols, dropna=False)[cols].agg(["mean", "median"])
    g.columns = ["__".join(c) for c in g.columns]
    g = g.reset_index()
    n = df.groupby(group_cols, dropna=False).size().rename("n").reset_index()
    return g.merge(n, on=group_cols)


def main() -> None:
    df = load()
    print(f"Loaded {len(df):,} rows")

    # Seniority buckets
    df["sen_combined"] = bucket(df, "seniority_best_available")
    df["sen_final"] = bucket(df, "seniority_final")
    df["sen_native"] = bucket(df, "seniority_native")

    # --- primary: prevalence by period × combined seniority ---
    prev_combined = prevalence(df, ["period2", "sen_combined"])
    prev_combined.to_csv(TAB / "prevalence_by_period_seniority.csv", index=False)

    # --- YOE proxy ---
    df["sen_yoe"] = np.where(
        df["yoe_extracted"].notna() & (df["yoe_extracted"] <= 2), "entry",
        np.where(df["yoe_extracted"] >= 5, "mid-senior",
                 np.where(df["yoe_extracted"].notna(), "associate", "unknown"))
    )
    prev_yoe = prevalence(df, ["period2", "sen_yoe"])
    prev_yoe.to_csv(TAB / "prevalence_by_period_seniority_yoe.csv", index=False)

    # --- aggregator vs direct (CORE) ---
    prev_agg = prevalence(df, ["period2", "is_aggregator", "sen_combined"])
    prev_agg.to_csv(TAB / "prevalence_aggregator_vs_direct.csv", index=False)

    # --- continuous summary by period × combined seniority × aggregator ---
    cont = continuous_summary(df, ["period2", "sen_combined", "is_aggregator"])
    cont.to_csv(TAB / "continuous_summary_by_period_seniority_aggregator.csv", index=False)

    # --- AI ghostiness: hedge/firm rates in proximity windows ---
    ai_rows = df[df["ai_windows"] > 0].copy()
    ai_rows["ai_prox_hedge_rate"] = ai_rows["ai_prox_hedge"] / ai_rows["ai_windows"]
    ai_rows["ai_prox_firm_rate"] = ai_rows["ai_prox_firm"] / ai_rows["ai_windows"]

    ai_asp = ai_rows.groupby(["period2", "sen_combined"]).agg(
        n=("uid", "size"),
        ai_windows_mean=("ai_windows", "mean"),
        ai_prox_hedge_rate=("ai_prox_hedge_rate", "mean"),
        ai_prox_firm_rate=("ai_prox_firm_rate", "mean"),
    ).reset_index()
    ai_asp["ai_prox_aspiration_ratio"] = ai_asp["ai_prox_hedge_rate"] / ai_asp["ai_prox_firm_rate"].clip(lower=1e-6)
    ai_asp.to_csv(TAB / "ai_proximity_aspiration.csv", index=False)

    # --- Non-AI baseline for comparison ---
    # Mean hedge/firm rate per posting overall (proxy for non-AI-specific aspiration)
    overall = df.groupby(["period2", "sen_combined"]).agg(
        n=("uid", "size"),
        hedge_mean=("hedge_count", "mean"),
        firm_mean=("firm_count", "mean"),
        hedge_any_rate=("hedge_any", "mean"),
        firm_any_rate=("firm_any", "mean"),
    ).reset_index()
    overall["global_aspiration_ratio"] = overall["hedge_mean"] / overall["firm_mean"].clip(lower=1e-6)
    overall.to_csv(TAB / "global_aspiration_by_period_seniority.csv", index=False)

    # Comparison AI vs non-AI (use ai_rows vs rows with ai_windows==0)
    nonai_rows = df[df["ai_windows"] == 0].copy()
    nonai_summary = nonai_rows.groupby(["period2", "sen_combined"]).agg(
        n_nonai=("uid", "size"),
        hedge_any_nonai=("hedge_any", "mean"),
        firm_any_nonai=("firm_any", "mean"),
    ).reset_index()
    nonai_summary["aspiration_ratio_nonai"] = (
        nonai_summary["hedge_any_nonai"] / nonai_summary["firm_any_nonai"].clip(lower=1e-6)
    )

    ai_vs = ai_asp.merge(nonai_summary, on=["period2", "sen_combined"], how="outer")
    ai_vs.to_csv(TAB / "ai_vs_nonai_aspiration.csv", index=False)

    # --- YOE-scope mismatch by operationalization ---
    mism_rows = []
    for op, col in [
        ("combined", "is_entry_combined"),
        ("final", "is_entry_final"),
        ("native", "is_entry_native"),
        ("yoe_le2", "is_entry_yoe"),
    ]:
        sub = df[df[col] == 1].copy()
        if len(sub) == 0:
            continue
        grp = sub.groupby("period2").agg(
            n=("uid", "size"),
            yoe_ge5_rate=("yoe_ge5", "mean"),
            senior_scope_ge3_rate=("senior_scope_n", lambda s: (s >= 3).mean()),
            yoe_scope_mismatch=("yoe_scope_mismatch_combined", "mean"),
        ).reset_index()
        grp["operationalization"] = op
        mism_rows.append(grp)
    mism = pd.concat(mism_rows, ignore_index=True)
    mism.to_csv(TAB / "yoe_scope_mismatch_by_operationalization.csv", index=False)

    # --- Credential impossibility by period ---
    cred = df.groupby("period2").agg(
        n=("uid", "size"),
        credential_contradiction_rate=("credential_contradiction", "mean"),
        no_degree_phrase_rate=("uid", lambda s: df.loc[s.index, "hedge_any"].size),  # placeholder
    ).reset_index()
    # simpler:
    cred2 = df.groupby(["period2", "sen_combined"]).agg(
        n=("uid", "size"),
        credential_contradiction_rate=("credential_contradiction", "mean"),
    ).reset_index()
    cred2.to_csv(TAB / "credential_impossibility_by_period.csv", index=False)

    # --- Industry patterns (where available, aggregator=False only) ---
    direct = df[df["is_aggregator"] == False].copy()
    direct["industry_norm"] = direct["company_industry"].fillna("__unknown__").str.strip().str.lower()
    top_industries = (
        direct["industry_norm"].value_counts().head(20).index.tolist()
    )
    ind_df = direct[direct["industry_norm"].isin(top_industries)]
    ind_summary = ind_df.groupby(["industry_norm", "period2"]).agg(
        n=("uid", "size"),
        hedge_any_rate=("hedge_any", "mean"),
        firm_any_rate=("firm_any", "mean"),
        kitchen_sink_mean=("kitchen_sink_score", "mean"),
        ai_any_rate=("ai_count", lambda s: (s >= 1).mean()),
        yoe_scope_mismatch=("yoe_scope_mismatch_combined", "mean"),
    ).reset_index()
    ind_summary.to_csv(TAB / "industry_ghostiness.csv", index=False)

    # --- Figures ---
    # Aspiration ratio and kitchen sink by period × combined seniority
    cont_primary = cont[cont["is_aggregator"] == False].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = cont_primary.pivot_table(
        index="sen_combined", columns="period2", values="aspiration_ratio__mean"
    )
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Aspiration ratio (hedge_count / firm_count) — direct employers")
    ax.set_ylabel("Mean aspiration ratio per posting")
    ax.set_xlabel("Seniority (combined best-available)")
    plt.tight_layout()
    plt.savefig(FIG / "aspiration_ratio_by_period.png", dpi=110)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = cont_primary.pivot_table(
        index="sen_combined", columns="period2", values="kitchen_sink_score__mean"
    )
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Kitchen-sink score (tech × org_scope) — direct employers")
    ax.set_ylabel("Mean kitchen-sink score")
    ax.set_xlabel("Seniority (combined best-available)")
    plt.tight_layout()
    plt.savefig(FIG / "kitchen_sink_by_period.png", dpi=110)
    plt.close()

    # AI-proximity vs global aspiration
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ai_asp))
    ax.bar(x - 0.2, ai_asp["ai_prox_hedge_rate"], width=0.4, label="AI-proximity hedge rate")
    ax.bar(x + 0.2, ai_asp["ai_prox_firm_rate"], width=0.4, label="AI-proximity firm rate")
    labels = [f"{r.period2}\n{r.sen_combined}" for _, r in ai_asp.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel("Rate")
    ax.set_title("AI-proximity hedge vs firm markers")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "ai_proximity_hedge_firm.png", dpi=110)
    plt.close()

    print("Saved prevalence/sensitivity tables and figures.")


if __name__ == "__main__":
    main()
