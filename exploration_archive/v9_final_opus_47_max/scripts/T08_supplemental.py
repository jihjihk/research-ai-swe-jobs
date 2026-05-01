"""T08 supplemental diagnostics.

Add:
- labeled-vs-not split by period and source (dim h)
- within-vs-between decomposition for AI-mention binary (returning-cohort)
- entry-specialist composition profile — what share of 2026 entry postings
  comes from entry-specialist companies, and what is the arshkon-only
  magnitude of J3 change after entry-specialist exclusion
- J3 dynamics on the returning cohort under J1/J2/J4 as well
- source-composition profiling of SWE tier by period
- description_length percentile-matched AI rate (counterfactual: if 2026
  length distribution matched 2024, how much of AI rise remains?)
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNIFIED = str(ROOT / "data" / "unified.parquet")
TAB_DIR = ROOT / "exploration" / "tables" / "T08"
TAB_DIR.mkdir(parents=True, exist_ok=True)
BASE = "is_swe AND source_platform='linkedin' AND is_english AND date_flag='ok'"


def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute(f"CREATE VIEW u AS SELECT * FROM '{UNIFIED}' WHERE {BASE}")
    return c


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TAB_DIR / f"{name}.csv", index=False)


def labeled_split() -> pd.DataFrame:
    c = con()
    df = c.execute(
        """
        SELECT source, period,
               COUNT(*) AS n,
               SUM(CASE WHEN llm_extraction_coverage='labeled' THEN 1 ELSE 0 END) AS n_ext_labeled,
               SUM(CASE WHEN llm_classification_coverage='labeled' THEN 1 ELSE 0 END) AS n_cls_labeled,
               SUM(CASE WHEN llm_classification_coverage='not_selected' THEN 1 ELSE 0 END) AS n_cls_notsel,
               SUM(CASE WHEN llm_classification_coverage='deferred' THEN 1 ELSE 0 END) AS n_cls_deferred,
               SUM(CASE WHEN llm_classification_coverage='skipped_short' THEN 1 ELSE 0 END) AS n_cls_short
        FROM u
        GROUP BY 1,2 ORDER BY 1,2
        """
    ).df()
    save_table(df, "labeled_split_by_source")
    return df


def ai_rate_by_coverage() -> pd.DataFrame:
    c = con()
    ai_strict = r"(copilot|cursor|claude|gpt-4|gpt4|codex|devin|windsurf|anthropic|chatgpt|openai|llm\b|rag\b|prompt engineering|mcp\b|langchain|llamaindex)"
    df = c.execute(
        f"""
        WITH t AS (
          SELECT CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period_coarse,
                 llm_extraction_coverage,
                 lower(description) AS d
          FROM u
        )
        SELECT period_coarse, llm_extraction_coverage, COUNT(*) AS n,
               SUM(CASE WHEN regexp_matches(d,'{ai_strict}') THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS ai_strict_share
        FROM t GROUP BY 1,2 ORDER BY 1,2
        """
    ).df()
    save_table(df, "ai_strict_by_coverage")
    return df


def ai_within_between() -> dict:
    c = con()
    ai_strict = r"(copilot|cursor|claude|gpt-4|gpt4|codex|devin|windsurf|anthropic|chatgpt|openai|llm\b|rag\b|prompt engineering|mcp\b|langchain|llamaindex)"
    # Returning cohort
    ret_path = str(ROOT / "exploration/artifacts/shared/returning_companies_cohort.csv")
    agg = c.execute(
        f"""
        WITH ret AS (SELECT company_name_canonical FROM read_csv('{ret_path}')),
             scope AS (
               SELECT u.*, CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS pc
               FROM u JOIN ret ON u.company_name_canonical = ret.company_name_canonical
             )
        SELECT pc, COUNT(*) AS n,
               SUM(CASE WHEN regexp_matches(lower(description),'{ai_strict}') THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS ai_strict_share
        FROM scope GROUP BY 1 ORDER BY 1
        """
    ).df()
    save_table(agg, "returning_cohort_ai_strict_pooled")

    # Per-co delta, averaged (equal-co weighting approx of within-co)
    per_co = c.execute(
        f"""
        WITH ret AS (SELECT company_name_canonical FROM read_csv('{ret_path}')),
             scope AS (
               SELECT u.*, CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS pc
               FROM u JOIN ret ON u.company_name_canonical = ret.company_name_canonical
             )
        SELECT company_name_canonical, pc, COUNT(*) AS n,
               SUM(CASE WHEN regexp_matches(lower(description),'{ai_strict}') THEN 1 ELSE 0 END)::DOUBLE AS n_ai
        FROM scope GROUP BY 1,2
        """
    ).df()
    per_co["ai_rate"] = per_co["n_ai"] / per_co["n"]
    pivot = per_co.pivot_table(index="company_name_canonical", columns="pc", values="ai_rate", aggfunc="first")
    pivot = pivot.dropna(how="all")
    pivot["delta"] = pivot.get("2026", 0) - pivot.get("2024", 0)
    within = {
        "n_cos_with_both": int(pivot.dropna(subset=["2024", "2026"]).shape[0]),
        "n_cos_total": int(pivot.shape[0]),
        "mean_2024": float(pivot["2024"].mean()),
        "mean_2026": float(pivot["2026"].mean()),
        "mean_delta_within": float(pivot.dropna(subset=["2024", "2026"])["delta"].mean()),
    }
    save_table(pd.DataFrame([within]), "returning_cohort_ai_strict_within")
    return within


def entry_specialist_composition() -> dict:
    c = con()
    es_path = str(ROOT / "exploration/artifacts/shared/entry_specialist_employers.csv")
    df = c.execute(
        f"""
        WITH es AS (SELECT company_name_canonical FROM read_csv('{es_path}'))
        SELECT CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS pc,
               CASE WHEN company_name_canonical IN (SELECT company_name_canonical FROM es) THEN 'entry_specialist' ELSE 'other' END AS bucket,
               COUNT(*) AS n_total,
               COUNT(yoe_min_years_llm) AS n_labeled,
               SUM(CASE WHEN yoe_min_years_llm <= 2 THEN 1 ELSE 0 END)::DOUBLE AS n_j3,
               SUM(CASE WHEN seniority_final = 'entry' THEN 1 ELSE 0 END)::DOUBLE AS n_j1
        FROM u GROUP BY 1,2 ORDER BY 1,2
        """
    ).df()
    df["j3_share_labeled"] = df["n_j3"] / df["n_labeled"]
    df["j1_share_all"] = df["n_j1"] / df["n_total"]
    save_table(df, "entry_specialist_composition")

    # what share of 2026 entry (J1) comes from entry_specialist?
    w = {}
    for pc in ["2024", "2026"]:
        sub = df[df["pc"] == pc]
        total_j1 = sub["n_j1"].sum()
        total_j3 = sub["n_j3"].sum()
        w[pc] = {
            "n_total": int(sub["n_total"].sum()),
            "n_j1": int(total_j1),
            "n_j3": int(total_j3),
            "entry_specialist_share_of_total": float(sub[sub["bucket"] == "entry_specialist"]["n_total"].iloc[0] / sub["n_total"].sum()),
            "entry_specialist_share_of_j1": float(sub[sub["bucket"] == "entry_specialist"]["n_j1"].iloc[0] / total_j1) if total_j1 > 0 else None,
            "entry_specialist_share_of_j3": float(sub[sub["bucket"] == "entry_specialist"]["n_j3"].iloc[0] / total_j3) if total_j3 > 0 else None,
        }
    save_table(pd.DataFrame(w).T.reset_index().rename(columns={"index": "period"}),
               "entry_specialist_share_summary")
    return w


def length_matched_ai() -> pd.DataFrame:
    """For each decile of 2024 description_length, compute AI-strict share in 2024 vs 2026
    rows with description_length in that decile. Then report overall rate when 2026 is
    weighted by 2024 length distribution → counterfactual.
    """
    c = con()
    ai_strict = r"(copilot|cursor|claude|gpt-4|gpt4|codex|devin|windsurf|anthropic|chatgpt|openai|llm\b|rag\b|prompt engineering|mcp\b|langchain|llamaindex)"
    # 2024 deciles — compute on 2024 rows
    q = """
    WITH t AS (
      SELECT CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS pc,
             description_length,
             lower(description) AS d
      FROM u
    )
    SELECT pc, description_length, d
    FROM t
    """
    rows = c.execute(q).df()
    deciles_2024 = np.quantile(rows[rows["pc"] == "2024"]["description_length"], np.arange(0, 1.01, 0.1))
    # Slightly inflate last bin
    deciles_2024[-1] = deciles_2024[-1] + 1
    bins = deciles_2024
    rows["bin"] = np.digitize(rows["description_length"], bins, right=False)
    # Binary AI strict
    import re
    pat = re.compile(ai_strict)
    rows["ai"] = rows["d"].str.contains(pat, regex=True, na=False)
    tbl = rows.groupby(["pc", "bin"]).agg(n=("ai", "size"), ai=("ai", "sum")).reset_index()
    tbl["rate"] = tbl["ai"] / tbl["n"]
    save_table(tbl, "ai_by_length_decile")

    # Counterfactual: 2026 rates, but weighted by 2024 bin distribution
    bin_totals_2024 = tbl[tbl["pc"] == "2024"].set_index("bin")["n"]
    bin_rates_2026 = tbl[tbl["pc"] == "2026"].set_index("bin")["rate"]
    cf_rate = float(((bin_totals_2024 / bin_totals_2024.sum()) * bin_rates_2026.reindex(bin_totals_2024.index).fillna(0)).sum())
    actual_2024 = rows[rows["pc"] == "2024"]["ai"].mean()
    actual_2026 = rows[rows["pc"] == "2026"]["ai"].mean()
    summary = pd.DataFrame([{
        "actual_2024_ai_strict": float(actual_2024),
        "actual_2026_ai_strict": float(actual_2026),
        "counterfactual_2026_at_2024_length": cf_rate,
        "raw_delta_pp": (actual_2026 - actual_2024) * 100,
        "length_matched_delta_pp": (cf_rate - actual_2024) * 100,
        "share_of_delta_length_driven": 1 - (cf_rate - actual_2024) / (actual_2026 - actual_2024) if (actual_2026 - actual_2024) else None,
    }])
    save_table(summary, "length_matched_ai_summary")
    return summary


def j_s_per_title_family() -> pd.DataFrame:
    """Light title-family stratification: Software Engineer / Senior Software Engineer /
    Data Engineer / ML Engineer / SRE / DevOps / Full-Stack / Frontend / Backend.
    Check J3 direction holds within title family."""
    c = con()
    fam_cases = {
        "swe_generic": r"software (developer|engineer|architect)|developer",
        "senior_swe_title": r"(sr\.?\s|senior\s)software",
        "data_engineer": r"data engineer",
        "ml_engineer": r"(machine learning|ml)\s*engineer|ai engineer",
        "sre_devops": r"(site reliability|sre\b|devops|platform engineer)",
        "fullstack": r"full.?stack",
        "frontend": r"front.?end|ui engineer",
        "backend": r"back.?end|api engineer",
    }
    dfs = []
    for fam, pat in fam_cases.items():
        q = f"""
        SELECT CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS pc,
               COUNT(*) AS n,
               COUNT(yoe_min_years_llm) AS n_lab,
               SUM(CASE WHEN yoe_min_years_llm<=2 THEN 1 ELSE 0 END)::DOUBLE AS n_j3,
               SUM(CASE WHEN yoe_min_years_llm>=5 THEN 1 ELSE 0 END)::DOUBLE AS n_s4
        FROM u WHERE regexp_matches(lower(title), '{pat}')
        GROUP BY 1 ORDER BY 1
        """
        d = c.execute(q).df()
        d["j3_share_labeled"] = d["n_j3"] / d["n_lab"]
        d["s4_share_labeled"] = d["n_s4"] / d["n_lab"]
        d["family"] = fam
        dfs.append(d)
    df = pd.concat(dfs)[["family", "pc", "n", "n_lab", "j3_share_labeled", "s4_share_labeled"]]
    save_table(df, "title_family_j3_s4")
    return df


def source_specialist_confirm() -> pd.DataFrame:
    """How much does the J3 rise (pooled 2024 vs 2026) collapse when we exclude
    entry specialists? (already computed in junior_senior_sensitivities.csv, but
    frame it as within-co vs between-co)."""
    c = con()
    es_path = str(ROOT / "exploration/artifacts/shared/entry_specialist_employers.csv")
    df = c.execute(
        f"""
        WITH es AS (SELECT company_name_canonical FROM read_csv('{es_path}'))
        SELECT CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS pc,
               COUNT(*) AS n,
               COUNT(yoe_min_years_llm) AS n_lab,
               SUM(CASE WHEN yoe_min_years_llm<=2 THEN 1 ELSE 0 END)::DOUBLE AS n_j3
        FROM u WHERE company_name_canonical NOT IN (SELECT company_name_canonical FROM es)
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    df["j3_share_labeled"] = df["n_j3"] / df["n_lab"]
    save_table(df, "j3_no_entry_specialist_pooled")
    return df


def description_section_bimodality() -> pd.DataFrame:
    """Check length distributions for bimodality / tail behavior."""
    c = con()
    df = c.execute(
        """
        SELECT CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS pc,
               COUNT(*) AS n,
               SUM(CASE WHEN description_length <= 500 THEN 1 ELSE 0 END) AS short,
               SUM(CASE WHEN description_length BETWEEN 501 AND 2000 THEN 1 ELSE 0 END) AS short_med,
               SUM(CASE WHEN description_length BETWEEN 2001 AND 4000 THEN 1 ELSE 0 END) AS medium,
               SUM(CASE WHEN description_length BETWEEN 4001 AND 7000 THEN 1 ELSE 0 END) AS long,
               SUM(CASE WHEN description_length BETWEEN 7001 AND 15000 THEN 1 ELSE 0 END) AS very_long,
               SUM(CASE WHEN description_length > 15000 THEN 1 ELSE 0 END) AS extreme
        FROM u GROUP BY 1 ORDER BY 1
        """
    ).df()
    for col in ["short", "short_med", "medium", "long", "very_long", "extreme"]:
        df[col + "_share"] = df[col] / df["n"]
    save_table(df, "length_bucket_shares")
    return df


def main() -> None:
    print("Labeled split")
    print(labeled_split().to_string())

    print("\nAI rate by coverage state")
    print(ai_rate_by_coverage().to_string())

    print("\nAI within-between (returning cohort)")
    print(ai_within_between())

    print("\nEntry-specialist composition")
    print(entry_specialist_composition())

    print("\nLength-matched AI")
    print(length_matched_ai().to_string())

    print("\nTitle-family J3/S4")
    print(j_s_per_title_family().to_string())

    print("\nJ3 no entry specialist")
    print(source_specialist_confirm().to_string())

    print("\nLength bucket shares")
    print(description_section_bimodality().to_string())


if __name__ == "__main__":
    main()
