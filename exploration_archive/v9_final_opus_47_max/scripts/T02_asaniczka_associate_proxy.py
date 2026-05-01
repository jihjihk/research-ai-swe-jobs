"""
T02 — Asaniczka `associate` as a junior proxy.

Decision rule: asaniczka `associate` behaves directionally like arshkon `entry` on at
least three of these signals:
  (a) top-title Jaccard (asaniczka-associate ↔ each arshkon group)
  (b) explicit junior/senior title-cue rates
  (c) `yoe_min_years_llm` distribution (primary); rule `yoe_extracted` as ablation
  (d) `seniority_final` distribution conditional on `seniority_native`

Verdict: recorded at end of report.
Also: effective entry-level sample sizes by source under `seniority_final`, broken
down by `seniority_final_source`.

Outputs:
- exploration/tables/T02/top_titles_by_group.csv
- exploration/tables/T02/top_titles_jaccard.csv
- exploration/tables/T02/title_cue_rates.csv
- exploration/tables/T02/yoe_llm_distribution.csv   (LLM primary)
- exploration/tables/T02/yoe_rule_distribution.csv  (rule ablation)
- exploration/tables/T02/seniority_final_vs_native.csv
- exploration/tables/T02/entry_effective_sample_size.csv
- exploration/tables/T02/comparability_summary.csv
- exploration/figures/T02/top_titles_jaccard_heatmap.png
- exploration/figures/T02/title_cue_rates.png
- exploration/figures/T02/yoe_llm_cdf.png
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
DATA = str(ROOT / "data" / "unified.parquet")
TABLES = ROOT / "exploration" / "tables" / "T02"
FIGS = ROOT / "exploration" / "figures" / "T02"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

DEFAULT_FILTER = "is_swe AND source_platform='linkedin' AND is_english AND date_flag='ok'"

# --------------- Regex patterns for junior/senior title cues --------------- #
# Verified against test cases in the test run of this script.
JUNIOR_PAT = r"(^|[^a-z])(junior|jr\b|jr\.|entry.level|entry.lvl|intern\b|graduate program|new.grad|early.career|trainee|apprentice|early career)"
SENIOR_PAT = r"(^|[^a-z])(senior|sr\b|sr\.|principal|staff|\blead\b|architect|director|vp |chief|head of|manager|\bii\b|\biii\b|\biv\b)"


def _run_pattern_tests() -> None:
    """Regex validation — run these at script start to catch regressions."""
    con = duckdb.connect()
    tests_junior = [
        ("junior software engineer", True),
        ("entry-level engineer", True),
        ("software engineer intern", True),
        ("software engineer i", False),
        ("data engineer", False),
        ("associate engineer", False),
        ("senior engineer", False),
    ]
    tests_senior = [
        ("senior software engineer", True),
        ("principal engineer", True),
        ("engineering manager", True),
        ("software engineer ii", True),
        ("junior engineer", False),
        ("developer", False),
        ("software engineer", False),
    ]
    for text, expected in tests_junior:
        r = con.sql(f"SELECT regexp_matches('{text}', '{JUNIOR_PAT}')").fetchone()[0]
        assert r == expected, f"junior pattern FAIL for {text!r}: got {r}, expected {expected}"
    for text, expected in tests_senior:
        r = con.sql(f"SELECT regexp_matches('{text}', '{SENIOR_PAT}')").fetchone()[0]
        assert r == expected, f"senior pattern FAIL for {text!r}: got {r}, expected {expected}"
    print("[pattern tests] all passed")


# --------------- 1. Top titles per group --------------- #

GROUPS = [
    ("kaggle_arshkon", "entry"),
    ("kaggle_arshkon", "associate"),
    ("kaggle_arshkon", "mid-senior"),
    ("kaggle_asaniczka", "associate"),
    ("kaggle_asaniczka", "mid-senior"),
]


def top_titles(top_k: int = 50) -> dict[tuple[str, str], set[str]]:
    con = duckdb.connect()
    out = []
    title_sets: dict[tuple[str, str], set[str]] = {}
    for src, sen in GROUPS:
        df = con.sql(
            f"""
            SELECT title_normalized, count(*) n
            FROM '{DATA}'
            WHERE {DEFAULT_FILTER} AND source='{src}' AND seniority_native='{sen}'
            GROUP BY title_normalized ORDER BY n DESC LIMIT {top_k}
            """
        ).df()
        df["source"] = src
        df["seniority_native"] = sen
        df["rank"] = range(1, len(df) + 1)
        out.append(df[["source", "seniority_native", "rank", "title_normalized", "n"]])
        title_sets[(src, sen)] = set(df["title_normalized"].tolist())
    combined = pd.concat(out, ignore_index=True)
    combined.to_csv(TABLES / "top_titles_by_group.csv", index=False)
    print("\nTop-50 titles per group written")
    return title_sets


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def jaccard_table(title_sets: dict[tuple[str, str], set[str]]) -> pd.DataFrame:
    labels = [f"{s}:{n}" for (s, n) in GROUPS]
    mat = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for i, g1 in enumerate(GROUPS):
        for j, g2 in enumerate(GROUPS):
            mat.iloc[i, j] = jaccard(title_sets[g1], title_sets[g2])
    mat.to_csv(TABLES / "top_titles_jaccard.csv")
    print("\nTop-title Jaccard matrix:")
    print(mat.round(3).to_string())

    # Heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat.values.astype(float), cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")
    ax.set_title("Jaccard of top-50 titles by (source, seniority_native)\nSWE × LinkedIn default filters", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Jaccard")
    plt.tight_layout()
    plt.savefig(FIGS / "top_titles_jaccard_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return mat


# --------------- 2. Explicit title cue rates --------------- #

def title_cue_rates() -> pd.DataFrame:
    con = duckdb.connect()
    df = con.sql(
        f"""
        SELECT source, seniority_native,
            count(*) n,
            sum(CASE WHEN regexp_matches(title_normalized, '{JUNIOR_PAT}') THEN 1 ELSE 0 END) n_junior_cue,
            sum(CASE WHEN regexp_matches(title_normalized, '{SENIOR_PAT}') THEN 1 ELSE 0 END) n_senior_cue
        FROM '{DATA}'
        WHERE {DEFAULT_FILTER} AND source IN ('kaggle_arshkon','kaggle_asaniczka')
          AND seniority_native IN ('entry','associate','mid-senior')
        GROUP BY source, seniority_native ORDER BY source, seniority_native
        """
    ).df()
    df["junior_rate"] = (df["n_junior_cue"] / df["n"]).round(4)
    df["senior_rate"] = (df["n_senior_cue"] / df["n"]).round(4)
    df.to_csv(TABLES / "title_cue_rates.csv", index=False)
    print("\nTitle-cue rates:")
    print(df.to_string())

    # Figure: bar chart of junior + senior rates per group
    labels = [f"{r['source'].replace('kaggle_', '')}\n{r['seniority_native']}" for _, r in df.iterrows()]
    x = np.arange(len(df))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, df["junior_rate"] * 100, w, label="junior-cue %", color="#3a86ff")
    ax.bar(x + w / 2, df["senior_rate"] * 100, w, label="senior-cue %", color="#fb5607")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("% of titles matching cue")
    ax.set_title("Explicit junior/senior title-cue rates\nby (source, seniority_native) — SWE × LinkedIn default")
    ax.legend()
    for i, row in df.iterrows():
        ax.text(i - w / 2, row["junior_rate"] * 100 + 0.3, f"{row['junior_rate']*100:.1f}", ha="center", fontsize=7)
        ax.text(i + w / 2, row["senior_rate"] * 100 + 0.3, f"{row['senior_rate']*100:.1f}", ha="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGS / "title_cue_rates.png", dpi=150)
    plt.close(fig)
    return df


# --------------- 3. YOE distribution (LLM primary + rule ablation) --------------- #

def yoe_distribution() -> None:
    """Distribution of yoe_min_years_llm and yoe_extracted by source × seniority_native."""
    con = duckdb.connect()

    # LLM primary: yoe_min_years_llm, within LLM-labeled frame
    df_llm = con.sql(
        f"""
        SELECT source, seniority_native,
            count(*) AS n_in_frame,
            sum(CASE WHEN yoe_min_years_llm IS NOT NULL THEN 1 ELSE 0 END) AS n_with_yoe,
            sum(CASE WHEN yoe_min_years_llm = 0 THEN 1 ELSE 0 END) AS n_yoe_0,
            sum(CASE WHEN yoe_min_years_llm <= 2 THEN 1 ELSE 0 END) AS n_yoe_le_2,
            sum(CASE WHEN yoe_min_years_llm <= 3 THEN 1 ELSE 0 END) AS n_yoe_le_3,
            sum(CASE WHEN yoe_min_years_llm >= 5 THEN 1 ELSE 0 END) AS n_yoe_ge_5,
            sum(CASE WHEN yoe_min_years_llm >= 7 THEN 1 ELSE 0 END) AS n_yoe_ge_7,
            avg(yoe_min_years_llm) AS mean_yoe,
            approx_quantile(yoe_min_years_llm, 0.25) AS p25,
            approx_quantile(yoe_min_years_llm, 0.5)  AS p50,
            approx_quantile(yoe_min_years_llm, 0.75) AS p75
        FROM '{DATA}'
        WHERE {DEFAULT_FILTER} AND source IN ('kaggle_arshkon','kaggle_asaniczka')
          AND seniority_native IN ('entry','associate','mid-senior')
          AND llm_classification_coverage='labeled'
        GROUP BY source, seniority_native
        ORDER BY source, seniority_native
        """
    ).df()
    # Rates on yoe-populated denominator
    df_llm["frac_yoe_le_2_of_with"] = (df_llm["n_yoe_le_2"] / df_llm["n_with_yoe"]).round(4)
    df_llm["frac_yoe_le_3_of_with"] = (df_llm["n_yoe_le_3"] / df_llm["n_with_yoe"]).round(4)
    df_llm["frac_yoe_ge_5_of_with"] = (df_llm["n_yoe_ge_5"] / df_llm["n_with_yoe"]).round(4)
    df_llm["frac_yoe_ge_7_of_with"] = (df_llm["n_yoe_ge_7"] / df_llm["n_with_yoe"]).round(4)
    df_llm.to_csv(TABLES / "yoe_llm_distribution.csv", index=False)
    print("\nLLM YOE distribution (primary):")
    print(df_llm.to_string())

    # Rule ablation: yoe_extracted, on all rows (not just LLM-frame)
    df_rule = con.sql(
        f"""
        SELECT source, seniority_native,
            count(*) AS n,
            sum(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS n_with_yoe,
            sum(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END) AS n_yoe_le_2,
            sum(CASE WHEN yoe_extracted <= 3 THEN 1 ELSE 0 END) AS n_yoe_le_3,
            sum(CASE WHEN yoe_extracted >= 5 THEN 1 ELSE 0 END) AS n_yoe_ge_5,
            avg(yoe_extracted) AS mean_yoe,
            approx_quantile(yoe_extracted, 0.5) AS p50
        FROM '{DATA}'
        WHERE {DEFAULT_FILTER} AND source IN ('kaggle_arshkon','kaggle_asaniczka')
          AND seniority_native IN ('entry','associate','mid-senior')
        GROUP BY source, seniority_native ORDER BY source, seniority_native
        """
    ).df()
    df_rule["frac_yoe_le_2_of_with"] = (df_rule["n_yoe_le_2"] / df_rule["n_with_yoe"]).round(4)
    df_rule["frac_yoe_le_3_of_with"] = (df_rule["n_yoe_le_3"] / df_rule["n_with_yoe"]).round(4)
    df_rule["frac_yoe_ge_5_of_with"] = (df_rule["n_yoe_ge_5"] / df_rule["n_with_yoe"]).round(4)
    df_rule.to_csv(TABLES / "yoe_rule_distribution.csv", index=False)
    print("\nRule YOE distribution (ablation):")
    print(df_rule.to_string())

    # Build CDF figure on LLM YOE
    # Fetch raw values for plotting
    yoe_raw = con.sql(
        f"""
        SELECT source, seniority_native, yoe_min_years_llm
        FROM '{DATA}'
        WHERE {DEFAULT_FILTER} AND source IN ('kaggle_arshkon','kaggle_asaniczka')
          AND seniority_native IN ('entry','associate','mid-senior')
          AND llm_classification_coverage='labeled'
          AND yoe_min_years_llm IS NOT NULL
        """
    ).df()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (src, sen), sub in yoe_raw.groupby(["source", "seniority_native"]):
        vals = np.sort(sub["yoe_min_years_llm"].values)
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        label = f"{src.replace('kaggle_','')} / {sen} (n={len(vals):,})"
        ax.plot(vals, cdf, label=label, linewidth=1.8)
    ax.set_xlabel("yoe_min_years_llm (years)")
    ax.set_ylabel("CDF")
    ax.set_title("YOE distribution by (source, seniority_native) — LLM primary\nSWE × LinkedIn default × LLM-labeled")
    ax.set_xlim(0, 15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.axvline(x=2, color="grey", linestyle=":", alpha=0.6)
    ax.axvline(x=5, color="grey", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGS / "yoe_llm_cdf.png", dpi=150)
    plt.close(fig)


# --------------- 4. seniority_final conditional on seniority_native --------------- #

def seniority_final_vs_native() -> pd.DataFrame:
    con = duckdb.connect()
    df = con.sql(
        f"""
        SELECT source, seniority_native, seniority_final, count(*) n
        FROM '{DATA}'
        WHERE {DEFAULT_FILTER} AND source IN ('kaggle_arshkon','kaggle_asaniczka')
          AND seniority_native IN ('entry','associate','mid-senior')
        GROUP BY source, seniority_native, seniority_final
        ORDER BY source, seniority_native, n DESC
        """
    ).df()
    # Row totals to compute conditional shares
    totals = df.groupby(["source", "seniority_native"])["n"].sum().reset_index().rename(columns={"n": "row_total"})
    df = df.merge(totals, on=["source", "seniority_native"])
    df["share"] = (df["n"] / df["row_total"]).round(4)
    df.to_csv(TABLES / "seniority_final_vs_native.csv", index=False)
    print("\nseniority_final conditional on seniority_native:")
    print(df.to_string())
    return df


# --------------- 5. Entry-level effective sample size --------------- #

def entry_effective_sample_size() -> None:
    con = duckdb.connect()
    df = con.sql(
        f"""
        SELECT source, seniority_final, seniority_final_source, count(*) n
        FROM '{DATA}'
        WHERE {DEFAULT_FILTER} AND seniority_final IN ('entry', 'associate')
        GROUP BY source, seniority_final, seniority_final_source
        ORDER BY source, seniority_final, n DESC
        """
    ).df()
    df.to_csv(TABLES / "entry_effective_sample_size.csv", index=False)
    print("\nEntry-level effective sample size by seniority_final x seniority_final_source:")
    print(df.to_string())


# --------------- Main --------------- #

def main() -> None:
    _run_pattern_tests()
    title_sets = top_titles()
    jaccard_table(title_sets)
    title_cue_rates()
    yoe_distribution()
    seniority_final_vs_native()
    entry_effective_sample_size()
    print("\nT02 artifacts written.")


if __name__ == "__main__":
    main()
