"""
Composite C — "Who says 'junior,' and what does it mean when they do?"

Exploratory analysis of how "junior" labelling behaves in SWE postings:
  1. Multiple operationalizations of "labelled junior" + precision proxy.
  2. Lower-bound vs lower-experience hypothesis (per-op compare to unlabelled
     entry-level).
  3. Labelling-rate trajectory among true-entry (low-YOE) postings.
  4. Who labels "junior" — industry, metro, source, company-volume.
  5. Substitute-term n-gram extraction among unlabelled true-entry postings.

Outputs:
  eda/tables/S28_*.csv
  eda/figures/S28_*.png

Run:
  ./.venv/bin/python eda/scripts/composite_C_junior_label.py
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

random.seed(0)
np.random.seed(0)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = PROJECT_ROOT / "data" / "unified_core.parquet"
T11_FEATURES_PATH = (
    PROJECT_ROOT / "exploration-archive" / "v9_final_opus_47" / "artifacts"
    / "shared" / "T11_posting_features.parquet"
)
TABLES_DIR = PROJECT_ROOT / "eda" / "tables"
FIGURES_DIR = PROJECT_ROOT / "eda" / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Operationalizations of "labelled junior"
# ---------------------------------------------------------------------------
#
# Each entry maps a short id -> a SQL-side BOOLEAN expression evaluated against
# `title_lower` and `desc_lower` (LOWER() applied once in CTE).
#
# Notes on regex choices:
# - We use \b word boundaries throughout; ".jr" without boundary would over-match.
# - "I" / "II" suffix patterns require explicit role context to avoid matching
#   pronouns ("I will lead..."). We anchor on common SWE role words.
# - Level codes (L3, L4, IC2, SDE-2, etc.) likewise are anchored.

OPS = {
    # (a) explicit "junior" only
    "OP_A_junior_title": (
        r"regexp_matches(title_lower, '\bjunior\b')"
    ),
    # (a') explicit "junior" or "jr." in title
    "OP_A2_junior_jr_title": (
        r"regexp_matches(title_lower, '\b(junior|jr\.?)\b')"
    ),
    # (b) explicit "junior" / "entry-level" / "new grad" / "graduate" / "associate"
    #     — broader title family
    "OP_B_explicit_entry_family": (
        r"regexp_matches(title_lower, "
        r"'\b(junior|jr\.?|entry[ -]?level|new[ -]?grad|graduate|associate|early[ -]?career)\b')"
    ),
    # (c) "junior" in description (looser)
    "OP_C_junior_in_desc": (
        r"regexp_matches(desc_lower, '\b(junior|entry[ -]?level|new[ -]?grad|early[ -]?career)\b')"
    ),
    # (d) Level codes / I-II suffix in title (associate/level codes)
    #     anchor on engineer/developer/programmer to reduce noise
    "OP_E_levelcode_title": (
        r"regexp_matches(title_lower, "
        r"'\b(software|swe|sde|developer|engineer|programmer|backend|frontend|full[ -]?stack)\s*"
        r"(i|1|ii|2)\b') OR "
        r"regexp_matches(title_lower, '\b(l[1-4]|ic[1-3]|sde[- ]?[12]|swe[- ]?[12]|t1|t2)\b')"
    ),
    # (e) any explicit early-career marker in TITLE (a', b, levelcode unioned)
    "OP_F_any_explicit_title": (
        r"(regexp_matches(title_lower, "
        r"'\b(junior|jr\.?|entry[ -]?level|new[ -]?grad|graduate|associate|early[ -]?career)\b') "
        r"OR regexp_matches(title_lower, "
        r"'\b(software|swe|sde|developer|engineer|programmer|backend|frontend|full[ -]?stack)\s*"
        r"(i|1|ii|2)\b') "
        r"OR regexp_matches(title_lower, '\b(l[1-4]|ic[1-3]|sde[- ]?[12]|swe[- ]?[12])\b'))"
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def base_cte() -> str:
    """SQL CTE that produces the analysis frame: SWE labeled rows, with op flags
    and lowered title/desc."""
    op_select = ",\n    ".join(
        f"({expr}) AS {name}" for name, expr in OPS.items()
    )
    return f"""
    WITH base AS (
      SELECT
        uid, source, period, title, description, description_core_llm,
        company_name_canonical, company_name_effective, company_industry,
        metro_area, is_aggregator,
        seniority_final, seniority_3level, seniority_native,
        yoe_min_years_llm, yoe_extracted, description_length,
        LOWER(COALESCE(title, '')) AS title_lower,
        LOWER(COALESCE(description, '')) AS desc_lower
      FROM '{CORE_PATH}'
      WHERE is_swe = true
        AND llm_classification_coverage = 'labeled'
    ),
    flagged AS (
      SELECT *,
        {op_select}
      FROM base
    )
    """


def precision_sample(con, op_name: str, n_per_period: int = 30, periods=("2024-01", "2026-04")) -> pd.DataFrame:
    """Pull a hand-sample (in-script: titles + first 200 chars of description) per op.
    Used for an offline precision proxy."""
    rows = []
    for p in periods:
        df = con.execute(f"""
          {base_cte()}
          , candidates AS (
            SELECT title, SUBSTR(description, 1, 250) AS desc_snippet,
                   yoe_min_years_llm, seniority_final
            FROM flagged
            WHERE {op_name} = TRUE AND period = '{p}'
          )
          SELECT * FROM candidates
          USING SAMPLE reservoir({n_per_period} ROWS) REPEATABLE (42)
        """).df()
        df["op"] = op_name
        df["period"] = p
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Analysis 1: per-op coverage by period
# ---------------------------------------------------------------------------

def op_coverage_by_period(con) -> pd.DataFrame:
    op_aggs = ",\n    ".join(
        f"SUM(CASE WHEN {n} THEN 1 ELSE 0 END) AS {n}_n,"
        f" 1.0 * SUM(CASE WHEN {n} THEN 1 ELSE 0 END) / COUNT(*) AS {n}_rate"
        for n in OPS
    )
    df = con.execute(f"""
      {base_cte()}
      SELECT period, COUNT(*) AS n_total,
        {op_aggs}
      FROM flagged
      GROUP BY 1 ORDER BY 1
    """).df()
    return df


# ---------------------------------------------------------------------------
# Analysis 2: "junior is a lower bound" — per-op contrast vs unlabelled true entry
# ---------------------------------------------------------------------------
#
# "Unlabelled true entry" = yoe_min_years_llm <= 2 AND no junior-family token.
# We compare each labelled-junior op to that pool.

def lower_bound_contrast(con) -> pd.DataFrame:
    """For each operationalization, compute YOE/desc-length stats and contrast
    against unlabelled-true-entry (yoe<=2 AND OP_F = false)."""
    out_rows = []

    # First: stats on the unlabelled-true-entry baseline per period.
    base = con.execute(f"""
      {base_cte()}
      SELECT period, COUNT(*) AS n,
             AVG(yoe_min_years_llm) AS mean_yoe,
             MEDIAN(yoe_min_years_llm) AS median_yoe,
             quantile_cont(yoe_min_years_llm, 0.75) AS p75_yoe,
             AVG(description_length) AS mean_desc_len
      FROM flagged
      WHERE yoe_min_years_llm IS NOT NULL
        AND yoe_min_years_llm <= 2
        AND NOT OP_F_any_explicit_title
      GROUP BY 1 ORDER BY 1
    """).df()
    base["op"] = "BASELINE_unlabelled_yoe_le2"
    out_rows.append(base)

    for op in OPS:
        df = con.execute(f"""
          {base_cte()}
          SELECT period, COUNT(*) AS n,
                 AVG(yoe_min_years_llm) AS mean_yoe,
                 MEDIAN(yoe_min_years_llm) AS median_yoe,
                 quantile_cont(yoe_min_years_llm, 0.75) AS p75_yoe,
                 AVG(description_length) AS mean_desc_len
          FROM flagged
          WHERE {op} = TRUE
          GROUP BY 1 ORDER BY 1
        """).df()
        df["op"] = op
        out_rows.append(df)

    return pd.concat(out_rows, ignore_index=True)[
        ["op", "period", "n", "mean_yoe", "median_yoe", "p75_yoe", "mean_desc_len"]
    ]


def lower_bound_contrast_within_entry(con) -> pd.DataFrame:
    """Restrict to seniority_final='entry' and compare junior-labelled vs unlabelled.

    The hypothesis is: within true entry-level (LLM seniority = entry), do
    junior-titled postings demand MORE YOE than the rest?
    """
    out_rows = []
    base = con.execute(f"""
      {base_cte()}
      SELECT period, COUNT(*) AS n,
             AVG(yoe_min_years_llm) AS mean_yoe,
             MEDIAN(yoe_min_years_llm) AS median_yoe,
             quantile_cont(yoe_min_years_llm, 0.75) AS p75_yoe,
             AVG(description_length) AS mean_desc_len
      FROM flagged
      WHERE seniority_final = 'entry'
        AND NOT OP_A2_junior_jr_title
      GROUP BY 1 ORDER BY 1
    """).df()
    base["op"] = "BASELINE_entry_no_junior_in_title"
    out_rows.append(base)

    for op in ["OP_A_junior_title", "OP_A2_junior_jr_title", "OP_B_explicit_entry_family",
               "OP_E_levelcode_title"]:
        df = con.execute(f"""
          {base_cte()}
          SELECT period, COUNT(*) AS n,
                 AVG(yoe_min_years_llm) AS mean_yoe,
                 MEDIAN(yoe_min_years_llm) AS median_yoe,
                 quantile_cont(yoe_min_years_llm, 0.75) AS p75_yoe,
                 AVG(description_length) AS mean_desc_len
          FROM flagged
          WHERE seniority_final = 'entry' AND {op} = TRUE
          GROUP BY 1 ORDER BY 1
        """).df()
        df["op"] = op + "__within_entry"
        out_rows.append(df)
    return pd.concat(out_rows, ignore_index=True)[
        ["op", "period", "n", "mean_yoe", "median_yoe", "p75_yoe", "mean_desc_len"]
    ]


# ---------------------------------------------------------------------------
# Analysis 2b: scope features (T11) — junior label vs not
# ---------------------------------------------------------------------------

def scope_features_for_juniors(con) -> pd.DataFrame:
    if not T11_FEATURES_PATH.exists():
        return pd.DataFrame()

    df = con.execute(f"""
      {base_cte()}
      , feats AS (
        SELECT uid, tech_count, requirement_breadth_resid,
               scope_density, credential_stack_depth, ai_binary
        FROM '{T11_FEATURES_PATH}'
      )
      SELECT
        f.period,
        CASE
          WHEN f.OP_A2_junior_jr_title THEN 'a_junior_titled'
          WHEN f.OP_F_any_explicit_title AND NOT f.OP_A2_junior_jr_title THEN 'b_other_explicit_marker'
          WHEN f.yoe_min_years_llm IS NOT NULL AND f.yoe_min_years_llm <= 2 THEN 'c_unlabelled_yoe_le2'
          ELSE 'd_other'
        END AS bucket,
        COUNT(*) AS n,
        AVG(t.tech_count) AS mean_tech_count,
        AVG(t.requirement_breadth_resid) AS mean_breadth_resid,
        AVG(t.scope_density) AS mean_scope_density,
        AVG(t.credential_stack_depth) AS mean_cred_stack,
        AVG(t.ai_binary::DOUBLE) AS ai_rate
      FROM flagged f INNER JOIN feats t USING (uid)
      GROUP BY 1,2 ORDER BY 1,2
    """).df()
    return df


# ---------------------------------------------------------------------------
# Analysis 3: labelling-rate trajectory among true-entry
# ---------------------------------------------------------------------------

def labelling_trajectory(con) -> pd.DataFrame:
    """Among postings with yoe_min_years_llm <= 2, what fraction explicitly
    say junior / entry / new grad / associate? Run for several thresholds."""
    rows = []
    for threshold in (0, 1, 2, 3):
        for include_seniority in (False, True):
            extra_filter = ""
            tag = f"yoe_le{threshold}"
            if include_seniority:
                extra_filter = " OR seniority_final = 'entry'"
                tag = f"yoe_le{threshold}_OR_seniority_entry"
            df = con.execute(f"""
              {base_cte()}
              SELECT period, COUNT(*) AS n,
                SUM(CASE WHEN OP_A_junior_title THEN 1 ELSE 0 END)::DOUBLE/COUNT(*) AS junior_rate,
                SUM(CASE WHEN OP_A2_junior_jr_title THEN 1 ELSE 0 END)::DOUBLE/COUNT(*) AS junior_jr_rate,
                SUM(CASE WHEN OP_B_explicit_entry_family THEN 1 ELSE 0 END)::DOUBLE/COUNT(*) AS broad_entry_rate,
                SUM(CASE WHEN OP_F_any_explicit_title THEN 1 ELSE 0 END)::DOUBLE/COUNT(*) AS any_explicit_rate
              FROM flagged
              WHERE (yoe_min_years_llm IS NOT NULL AND yoe_min_years_llm <= {threshold}){extra_filter}
              GROUP BY 1 ORDER BY 1
            """).df()
            df["definition"] = tag
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Analysis 4: who labels "junior"?
# ---------------------------------------------------------------------------

def who_labels_junior(con) -> dict[str, pd.DataFrame]:
    out = {}
    # By industry
    out["industry"] = con.execute(f"""
      {base_cte()}
      SELECT period, COALESCE(company_industry, 'unknown') AS industry,
             COUNT(*) AS n,
             SUM(CASE WHEN OP_A2_junior_jr_title THEN 1 ELSE 0 END) AS n_junior,
             1.0 * SUM(CASE WHEN OP_A2_junior_jr_title THEN 1 ELSE 0 END) / COUNT(*) AS junior_rate
      FROM flagged
      GROUP BY 1,2 HAVING COUNT(*) >= 30
      ORDER BY 1, junior_rate DESC
    """).df()

    # By metro
    out["metro"] = con.execute(f"""
      {base_cte()}
      SELECT period, COALESCE(metro_area, 'unknown') AS metro,
             COUNT(*) AS n,
             SUM(CASE WHEN OP_A2_junior_jr_title THEN 1 ELSE 0 END) AS n_junior,
             1.0 * SUM(CASE WHEN OP_A2_junior_jr_title THEN 1 ELSE 0 END) / COUNT(*) AS junior_rate
      FROM flagged
      GROUP BY 1,2 HAVING COUNT(*) >= 30
      ORDER BY 1, junior_rate DESC
    """).df()

    # By source (kaggle vs scraped)
    out["source"] = con.execute(f"""
      {base_cte()}
      SELECT period, source, COUNT(*) AS n,
             SUM(CASE WHEN OP_A2_junior_jr_title THEN 1 ELSE 0 END) AS n_junior,
             1.0 * SUM(CASE WHEN OP_A2_junior_jr_title THEN 1 ELSE 0 END) / COUNT(*) AS junior_rate
      FROM flagged GROUP BY 1,2 ORDER BY 1,2
    """).df()

    # By aggregator status
    out["aggregator"] = con.execute(f"""
      {base_cte()}
      SELECT period, is_aggregator, COUNT(*) AS n,
             SUM(CASE WHEN OP_A2_junior_jr_title THEN 1 ELSE 0 END) AS n_junior,
             1.0 * SUM(CASE WHEN OP_A2_junior_jr_title THEN 1 ELSE 0 END) / COUNT(*) AS junior_rate
      FROM flagged GROUP BY 1,2 ORDER BY 1,2
    """).df()

    # By company-volume bucket (within period, bucket companies by their posting count)
    out["company_volume"] = con.execute(f"""
      {base_cte()}
      , co AS (
        SELECT period, company_name_canonical, COUNT(*) AS posts
        FROM flagged
        WHERE company_name_canonical IS NOT NULL
        GROUP BY 1,2
      ),
      bucketed AS (
        SELECT period, company_name_canonical,
               CASE
                 WHEN posts = 1 THEN 'a_1'
                 WHEN posts BETWEEN 2 AND 5 THEN 'b_2_5'
                 WHEN posts BETWEEN 6 AND 25 THEN 'c_6_25'
                 WHEN posts BETWEEN 26 AND 100 THEN 'd_26_100'
                 ELSE 'e_100plus'
               END AS vol_bucket
        FROM co
      )
      SELECT b.period, b.vol_bucket, COUNT(*) AS n,
             SUM(CASE WHEN f.OP_A2_junior_jr_title THEN 1 ELSE 0 END) AS n_junior,
             1.0 * SUM(CASE WHEN f.OP_A2_junior_jr_title THEN 1 ELSE 0 END) / COUNT(*) AS junior_rate
      FROM flagged f INNER JOIN bucketed b
        ON f.period = b.period AND f.company_name_canonical = b.company_name_canonical
      GROUP BY 1,2 ORDER BY 1,2
    """).df()

    # Top companies that say "junior" most often (latest period) — among >= 5 posts
    out["top_companies_2026_04"] = con.execute(f"""
      {base_cte()}
      SELECT company_name_canonical, COUNT(*) AS n,
             SUM(CASE WHEN OP_A2_junior_jr_title THEN 1 ELSE 0 END) AS n_junior,
             1.0 * SUM(CASE WHEN OP_A2_junior_jr_title THEN 1 ELSE 0 END) / COUNT(*) AS junior_rate
      FROM flagged
      WHERE period = '2026-04' AND company_name_canonical IS NOT NULL
      GROUP BY 1 HAVING COUNT(*) >= 10
      ORDER BY junior_rate DESC, n DESC
      LIMIT 30
    """).df()

    return out


# ---------------------------------------------------------------------------
# Analysis 5: substitute-term n-gram extraction
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[/&\-][a-z0-9]+)*")

def title_ngrams(titles: list[str], n: int = 2) -> Counter:
    c = Counter()
    for t in titles:
        toks = TOKEN_RE.findall(t.lower())
        for i in range(len(toks) - n + 1):
            c[" ".join(toks[i:i+n])] += 1
    return c


def substitute_terms_among_unlabelled_entry(con) -> dict[str, pd.DataFrame]:
    """Among yoe<=2 postings that do NOT carry an explicit early-career marker,
    what title patterns are most common? Compare 2024-01 vs 2026-04."""

    sql = f"""
      {base_cte()}
      SELECT period, title
      FROM flagged
      WHERE yoe_min_years_llm IS NOT NULL AND yoe_min_years_llm <= 2
        AND NOT OP_F_any_explicit_title
    """
    df = con.execute(sql).df()
    out = {}
    for n in (1, 2, 3):
        rows = []
        for p in sorted(df["period"].unique()):
            titles = df[df["period"] == p]["title"].dropna().tolist()
            counter = title_ngrams(titles, n=n)
            top = counter.most_common(50)
            for token, count in top:
                rows.append({"period": p, "n": n, "token": token,
                             "count": count, "rate": count / max(len(titles), 1),
                             "n_titles_in_period": len(titles)})
        out[f"ngram_n{n}"] = pd.DataFrame(rows)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_op_coverage(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    periods = df["period"].tolist()
    ops = list(OPS.keys())
    x = np.arange(len(periods))
    width = 0.13
    for i, op in enumerate(ops):
        rates = df[f"{op}_rate"].values * 100
        ax.bar(x + (i - len(ops)/2) * width, rates, width, label=op.replace("OP_", "").replace("_title", ""))
    ax.set_xticks(x); ax.set_xticklabels(periods)
    ax.set_ylabel("% of SWE postings flagged")
    ax.set_title("Operationalizations of 'labelled junior' — share of all SWE postings")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_labelling_trajectory(df: pd.DataFrame, out_path: Path) -> None:
    """Show, among yoe<=2 postings, the explicit-junior rate over time."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, defn in zip(axes, ["yoe_le2", "yoe_le2_OR_seniority_entry"]):
        sub = df[df["definition"] == defn].sort_values("period")
        ax.plot(sub["period"], sub["junior_jr_rate"] * 100, "o-", label="junior/jr")
        ax.plot(sub["period"], sub["broad_entry_rate"] * 100, "s-", label="junior+entry+grad+associate")
        ax.plot(sub["period"], sub["any_explicit_rate"] * 100, "^-", label="any explicit (incl. levelcode)")
        ax.set_title(f"True-entry def: {defn}")
        ax.set_ylabel("% of true-entry postings explicitly labelled (title)")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Has the explicit 'junior' label rate fallen among true-entry postings?")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_lower_bound_yoe(df_contrast: pd.DataFrame, out_path: Path) -> None:
    """Mean YOE per op per period."""
    pivot = df_contrast.pivot(index="period", columns="op", values="mean_yoe")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(ax=ax, marker="o")
    ax.set_ylabel("Mean LLM-extracted YOE")
    ax.set_title("Mean YOE by operationalization — testing 'junior is a lower bound'")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    con = duckdb.connect()

    print("== Analysis 1: per-op coverage by period ==")
    cov = op_coverage_by_period(con)
    cov.to_csv(TABLES_DIR / "S28_01_op_coverage_by_period.csv", index=False)
    print(cov.to_string()); print()

    print("== Precision-proxy hand-samples (saved for offline review) ==")
    samples = []
    for op in OPS:
        s = precision_sample(con, op, n_per_period=15, periods=("2024-01", "2026-04"))
        samples.append(s)
    pd.concat(samples, ignore_index=True).to_csv(
        TABLES_DIR / "S28_02_precision_handsamples.csv", index=False
    )
    print(f"  wrote {TABLES_DIR / 'S28_02_precision_handsamples.csv'} (~{15 * 2 * len(OPS)} rows)\n")

    print("== Analysis 2: lower-bound contrast (op vs unlabelled-true-entry) ==")
    contrast = lower_bound_contrast(con)
    contrast.to_csv(TABLES_DIR / "S28_03_lower_bound_contrast.csv", index=False)
    print(contrast.to_string()); print()

    print("== Analysis 2b: within-entry contrast ==")
    within = lower_bound_contrast_within_entry(con)
    within.to_csv(TABLES_DIR / "S28_04_within_entry_contrast.csv", index=False)
    print(within.to_string()); print()

    print("== Analysis 2c: scope features by junior bucket ==")
    scope = scope_features_for_juniors(con)
    if not scope.empty:
        scope.to_csv(TABLES_DIR / "S28_05_scope_features_by_bucket.csv", index=False)
        print(scope.to_string()); print()
    else:
        print("  (T11 features parquet not found — skipped)\n")

    print("== Analysis 3: labelling-rate trajectory ==")
    traj = labelling_trajectory(con)
    traj.to_csv(TABLES_DIR / "S28_06_labelling_trajectory.csv", index=False)
    print(traj.to_string()); print()

    print("== Analysis 4: who labels 'junior'? ==")
    who = who_labels_junior(con)
    for k, v in who.items():
        path = TABLES_DIR / f"S28_07_who_labels_{k}.csv"
        v.to_csv(path, index=False)
        print(f"  wrote {path}  ({len(v)} rows)")

    print("\n== Analysis 5: substitute-term n-grams ==")
    subs = substitute_terms_among_unlabelled_entry(con)
    for k, v in subs.items():
        path = TABLES_DIR / f"S28_08_substitute_{k}.csv"
        v.to_csv(path, index=False)
        print(f"  wrote {path}  ({len(v)} rows)")

    # Figures
    plot_op_coverage(cov, FIGURES_DIR / "S28_01_op_coverage_by_period.png")
    plot_labelling_trajectory(traj, FIGURES_DIR / "S28_06_labelling_trajectory.png")
    plot_lower_bound_yoe(contrast, FIGURES_DIR / "S28_03_lower_bound_yoe.png")
    print("\nFigures written.")


if __name__ == "__main__":
    main()
