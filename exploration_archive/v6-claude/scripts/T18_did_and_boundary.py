"""T18 part 2 — DiD computations + TF-IDF boundary blurring.

Reads the group_period_metrics.csv from T18_cross_occupation.py and
produces:
  1. DiD table (SWE vs control) with 95% CIs via normal approximation for
     proportions, plus signal-to-noise ratio using within-2024 variability
     when available.
  2. TF-IDF cosine similarity between SWE and SWE-adjacent corpora per
     period (balanced samples: 200 each).
  3. Per-title deep-dives for key adjacent roles.
  4. AI adoption gradient plot.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
PARQUET = ROOT / "data/unified.parquet"
TABLES = ROOT / "exploration/tables/T18"
FIGS = ROOT / "exploration/figures/T18"

# ----------------------------------------------------------------------
# 1. DiD table
# ----------------------------------------------------------------------
def proportion_diff_ci(p1: float, n1: int, p2: float, n2: int) -> tuple[float, float]:
    """95% CI for p1-p2 via normal approximation."""
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    return (d - 1.96 * se, d + 1.96 * se)


def did_ci(
    p_swe_26: float,
    n_swe_26: int,
    p_swe_24: float,
    n_swe_24: int,
    p_ctrl_26: float,
    n_ctrl_26: int,
    p_ctrl_24: float,
    n_ctrl_24: int,
) -> tuple[float, float, float]:
    """DiD = (p_swe_26 - p_swe_24) - (p_ctrl_26 - p_ctrl_24).

    Variance = sum of four binomial variances (assumed independent).
    Returns (DiD, lo, hi).
    """
    did = (p_swe_26 - p_swe_24) - (p_ctrl_26 - p_ctrl_24)
    var = (
        p_swe_26 * (1 - p_swe_26) / n_swe_26
        + p_swe_24 * (1 - p_swe_24) / n_swe_24
        + p_ctrl_26 * (1 - p_ctrl_26) / n_ctrl_26
        + p_ctrl_24 * (1 - p_ctrl_24) / n_ctrl_24
    )
    se = np.sqrt(var)
    return did, did - 1.96 * se, did + 1.96 * se


def build_did_table() -> pd.DataFrame:
    df = pd.read_csv(TABLES / "group_period_metrics.csv")
    # Use pooled 2024 as the 2024 baseline (per preamble sensitivity e — arshkon
    # vs scraped is primary elsewhere but for cross-occupation DiD we want the
    # larger asaniczka control sample).
    metrics = [
        ("ai_broad_any_share", "AI broad (24-term union)"),
        ("ai_narrow_share", "AI narrow (bare 'ai')"),
        ("tool_claude_share", "Claude tool mention"),
        ("tool_copilot_share", "Copilot mention"),
        ("tool_langchain_share", "LangChain mention"),
        ("tool_agents_share", "Agents framework mention"),
        ("scope_any_share", "Scope (end-to-end / cross-func)"),
        ("scope_end_to_end_share", "End-to-end"),
        ("scope_cross_functional_share", "Cross-functional"),
        ("desc_length_mean", "Description length (mean chars)"),
        ("tech_count_mean", "Tech count mean"),
        ("any_tech_share", "Any tech mention"),
        ("entry_share_of_known", "Entry share of known"),
        ("yoe_le2_share_of_yoe_known", "YOE ≤ 2 share"),
    ]

    rows: list[dict] = []
    for baseline in ["2024_pooled", "2024_arshkon"]:
        swe_24 = df[(df.group == "swe") & (df.period == baseline)].iloc[0]
        adj_24 = df[(df.group == "adj") & (df.period == baseline)].iloc[0]
        ctrl_24 = df[(df.group == "ctrl") & (df.period == baseline)].iloc[0]
        swe_26 = df[(df.group == "swe") & (df.period == "2026_scraped")].iloc[0]
        adj_26 = df[(df.group == "adj") & (df.period == "2026_scraped")].iloc[0]
        ctrl_26 = df[(df.group == "ctrl") & (df.period == "2026_scraped")].iloc[0]

        for metric_col, label in metrics:
            p_swe_24 = swe_24[metric_col]
            p_swe_26 = swe_26[metric_col]
            p_adj_24 = adj_24[metric_col]
            p_adj_26 = adj_26[metric_col]
            p_ctrl_24 = ctrl_24[metric_col]
            p_ctrl_26 = ctrl_26[metric_col]

            swe_delta = p_swe_26 - p_swe_24
            adj_delta = p_adj_26 - p_adj_24
            ctrl_delta = p_ctrl_26 - p_ctrl_24
            did_swe_vs_ctrl = swe_delta - ctrl_delta
            did_adj_vs_ctrl = adj_delta - ctrl_delta
            did_swe_vs_adj = swe_delta - adj_delta

            row = {
                "baseline": baseline,
                "metric": metric_col,
                "label": label,
                "swe_2024": p_swe_24,
                "swe_2026": p_swe_26,
                "swe_delta": swe_delta,
                "adj_2024": p_adj_24,
                "adj_2026": p_adj_26,
                "adj_delta": adj_delta,
                "ctrl_2024": p_ctrl_24,
                "ctrl_2026": p_ctrl_26,
                "ctrl_delta": ctrl_delta,
                "did_swe_vs_ctrl": did_swe_vs_ctrl,
                "did_adj_vs_ctrl": did_adj_vs_ctrl,
                "did_swe_vs_adj": did_swe_vs_adj,
                "n_swe_24": int(swe_24["n_total"]),
                "n_swe_26": int(swe_26["n_total"]),
                "n_adj_24": int(adj_24["n_total"]),
                "n_adj_26": int(adj_26["n_total"]),
                "n_ctrl_24": int(ctrl_24["n_total"]),
                "n_ctrl_26": int(ctrl_26["n_total"]),
            }

            # For binary metrics compute CI on DiD
            if metric_col not in ("desc_length_mean", "tech_count_mean"):
                did, lo, hi = did_ci(
                    p_swe_26,
                    int(swe_26["n_total"]),
                    p_swe_24,
                    int(swe_24["n_total"]),
                    p_ctrl_26,
                    int(ctrl_26["n_total"]),
                    p_ctrl_24,
                    int(ctrl_24["n_total"]),
                )
                row["did_swe_vs_ctrl_ci_lo"] = lo
                row["did_swe_vs_ctrl_ci_hi"] = hi
                did_a, lo_a, hi_a = did_ci(
                    p_adj_26,
                    int(adj_26["n_total"]),
                    p_adj_24,
                    int(adj_24["n_total"]),
                    p_ctrl_26,
                    int(ctrl_26["n_total"]),
                    p_ctrl_24,
                    int(ctrl_24["n_total"]),
                )
                row["did_adj_vs_ctrl_ci_lo"] = lo_a
                row["did_adj_vs_ctrl_ci_hi"] = hi_a
            else:
                row["did_swe_vs_ctrl_ci_lo"] = None
                row["did_swe_vs_ctrl_ci_hi"] = None
                row["did_adj_vs_ctrl_ci_lo"] = None
                row["did_adj_vs_ctrl_ci_hi"] = None
            rows.append(row)
    dfd = pd.DataFrame(rows)
    dfd.to_csv(TABLES / "did_table.csv", index=False)
    print(f"Wrote {TABLES / 'did_table.csv'}")
    return dfd


# ----------------------------------------------------------------------
# 2. TF-IDF cosine between SWE ↔ SWE-adjacent corpora per period
# ----------------------------------------------------------------------
def boundary_similarity() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    con = duckdb.connect()
    # Sample 200 rows per group × period. Use description_core_llm when
    # available (labeled), fall back to raw description; note split.
    rng = np.random.default_rng(42)

    def fetch_sample(group_filter: str, period_filter: str, n: int = 200) -> list[str]:
        sql = f"""
        SELECT COALESCE(description_core_llm, description) AS txt
        FROM (
          SELECT description, description_core_llm
          FROM read_parquet('{PARQUET}')
          WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok'
            AND {group_filter}
            AND {period_filter}
            AND description IS NOT NULL
        )
        ORDER BY random()
        LIMIT {n}
        """
        df = con.execute(sql).fetch_df()
        return df["txt"].fillna("").tolist()

    periods = {
        "2024_pooled": "(source IN ('kaggle_arshkon','kaggle_asaniczka'))",
        "2024_arshkon": "(source = 'kaggle_arshkon')",
        "2026_scraped": "(source = 'scraped')",
    }

    results: list[dict] = []
    for period_label, period_filter in periods.items():
        swe_texts = fetch_sample("is_swe=true", period_filter)
        adj_texts = fetch_sample("is_swe_adjacent=true", period_filter)
        ctrl_texts = fetch_sample("is_control=true", period_filter)

        # Fit TF-IDF on union of all three corpora to share vocab
        all_texts = swe_texts + adj_texts + ctrl_texts
        vec = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 1),
            min_df=2,
            max_df=0.95,
        )
        X = vec.fit_transform(all_texts)
        n_swe = len(swe_texts)
        n_adj = len(adj_texts)
        n_ctrl = len(ctrl_texts)

        swe_centroid = np.asarray(X[:n_swe].mean(axis=0))
        adj_centroid = np.asarray(X[n_swe : n_swe + n_adj].mean(axis=0))
        ctrl_centroid = np.asarray(X[n_swe + n_adj :].mean(axis=0))

        cos_swe_adj = cosine_similarity(swe_centroid, adj_centroid)[0, 0]
        cos_swe_ctrl = cosine_similarity(swe_centroid, ctrl_centroid)[0, 0]
        cos_adj_ctrl = cosine_similarity(adj_centroid, ctrl_centroid)[0, 0]

        results.append(
            {
                "period": period_label,
                "n_swe": n_swe,
                "n_adj": n_adj,
                "n_ctrl": n_ctrl,
                "cos_swe_adj": cos_swe_adj,
                "cos_swe_ctrl": cos_swe_ctrl,
                "cos_adj_ctrl": cos_adj_ctrl,
            }
        )

    df_bound = pd.DataFrame(results)
    df_bound.to_csv(TABLES / "boundary_cosine.csv", index=False)
    print(f"Wrote {TABLES / 'boundary_cosine.csv'}")
    print(df_bound.to_string(index=False))

    # Top migrating terms: words 2026-heavier in adjacent relative to their
    # 2024 weight in adjacent. We compare term-freq centroids.
    # Use 2024_pooled vs 2026_scraped.
    def centroid_sample(group_filter, period_filter, n=400):
        return fetch_sample(group_filter, period_filter, n)

    # Fit a combined vocab on SWE-adjacent 2024 + 2026
    adj_24 = centroid_sample("is_swe_adjacent=true", periods["2024_pooled"], 400)
    adj_26 = centroid_sample("is_swe_adjacent=true", periods["2026_scraped"], 400)
    swe_24 = centroid_sample("is_swe=true", periods["2024_pooled"], 400)
    swe_26 = centroid_sample("is_swe=true", periods["2026_scraped"], 400)

    corpus = adj_24 + adj_26 + swe_24 + swe_26
    vec = TfidfVectorizer(max_features=3000, stop_words="english", min_df=3)
    X = vec.fit_transform(corpus)
    vocab = np.array(vec.get_feature_names_out())
    n = 400
    adj_24_cent = np.asarray(X[:n].mean(axis=0)).ravel()
    adj_26_cent = np.asarray(X[n : 2 * n].mean(axis=0)).ravel()
    swe_24_cent = np.asarray(X[2 * n : 3 * n].mean(axis=0)).ravel()
    swe_26_cent = np.asarray(X[3 * n :].mean(axis=0)).ravel()

    # Terms that are in SWE 2024 -> SWE 2026 heavy AND migrated to adjacent
    delta_adj = adj_26_cent - adj_24_cent
    delta_swe = swe_26_cent - swe_24_cent
    # Migration = term rose in adj by (≥) fraction of swe rise
    migration = delta_adj / (delta_swe + 1e-9)
    swe_up = delta_swe > np.percentile(delta_swe, 95)
    adj_up = delta_adj > np.percentile(delta_adj, 95)
    migrated = np.where(swe_up & adj_up)[0]
    migration_scores = pd.DataFrame(
        {
            "term": vocab[migrated],
            "adj_2024": adj_24_cent[migrated],
            "adj_2026": adj_26_cent[migrated],
            "adj_delta": delta_adj[migrated],
            "swe_delta": delta_swe[migrated],
        }
    ).sort_values("adj_delta", ascending=False).head(40)
    migration_scores.to_csv(TABLES / "migrating_terms_swe_to_adj.csv", index=False)
    print(f"Wrote {TABLES / 'migrating_terms_swe_to_adj.csv'}")


# ----------------------------------------------------------------------
# 3. Per-title deep dives for adjacent roles
# ----------------------------------------------------------------------
def per_title_deep_dive() -> None:
    con = duckdb.connect()
    targets = [
        ("data_engineer", r"data engineer|data engineering"),
        ("data_scientist", r"data scientist"),
        ("ml_engineer", r"(machine learning|ml) engineer|ml engineer"),
        ("network_engineer", r"network engineer"),
        ("qa_engineer", r"(quality|qa) engineer|quality assurance"),
        ("security_engineer", r"security engineer|cybersecurity"),
        ("solutions_architect", r"(solutions? architect|solution architect)"),
        ("data_analyst", r"data analyst"),
        ("database_administrator", r"database administrator|dba"),
    ]
    # AI broad union pattern (OR of regexes)
    ai_patterns = [
        r"\bai\b",
        r"\bartificial intelligence\b",
        r"\bmachine learning\b",
        r"\bml\b",
        r"\bllms?\b",
        r"\bgpt\b",
        r"\bcopilot\b",
        r"\bclaude\b",
        r"\bopenai\b",
        r"\banthropic\b",
        r"\bagentic\b",
        r"\blangchain\b",
    ]
    union_parts = " OR ".join(
        f"regexp_matches(lower(description), '{pat}')" for pat in ai_patterns
    )

    rows = []
    for key, pattern in targets:
        for period_label, period_filter in [
            ("2024_pooled", "(source IN ('kaggle_arshkon','kaggle_asaniczka'))"),
            ("2026_scraped", "(source='scraped')"),
        ]:
            sql = f"""
            SELECT
              COUNT(*) as n,
              AVG(CASE WHEN ({union_parts}) THEN 1.0 ELSE 0.0 END) as ai_broad,
              AVG(description_length) as len,
              -- SWE similarity proxy: share classified is_swe rather than is_swe_adjacent
              AVG(CASE WHEN is_swe THEN 1.0 ELSE 0.0 END) as swe_share
            FROM read_parquet('{PARQUET}')
            WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok'
              AND (is_swe OR is_swe_adjacent)
              AND regexp_matches(lower(title_normalized), '{pattern}')
              AND {period_filter}
            """
            r = con.execute(sql).fetchone()
            rows.append(
                {
                    "title_group": key,
                    "period": period_label,
                    "n": r[0],
                    "ai_broad_share": r[1],
                    "desc_length_mean": r[2],
                    "swe_classified_share": r[3],
                }
            )

    pd.DataFrame(rows).to_csv(TABLES / "per_adjacent_title.csv", index=False)
    print(f"Wrote {TABLES / 'per_adjacent_title.csv'}")


# ----------------------------------------------------------------------
# 4. AI adoption gradient figure
# ----------------------------------------------------------------------
def figure_adoption_gradient() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(TABLES / "group_period_metrics.csv")
    order = ["swe", "adj", "ctrl"]
    labels = {"swe": "SWE", "adj": "SWE-adjacent", "ctrl": "Control"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Plot 1: AI broad share across groups, three periods
    periods_plot = ["2024_pooled", "2026_scraped"]
    colors = {"swe": "#1f77b4", "adj": "#ff7f0e", "ctrl": "#2ca02c"}
    for g in order:
        sub = df[(df.group == g) & (df.period.isin(periods_plot))]
        sub = sub.set_index("period").loc[periods_plot]
        axes[0].plot(
            range(len(periods_plot)),
            sub["ai_broad_any_share"].values,
            "-o",
            color=colors[g],
            label=labels[g],
            linewidth=2,
            markersize=8,
        )
    axes[0].set_xticks(range(len(periods_plot)))
    axes[0].set_xticklabels(["2024 pooled", "2026 scraped"])
    axes[0].set_ylabel("AI broad prevalence (share)")
    axes[0].set_title("AI prevalence gradient: SWE > adjacent > control (broad 24-term)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 0.6)

    # Plot 2: gap between SWE and control over time (widening/narrowing)
    pivot = df[df.period.isin(periods_plot)].pivot_table(
        index="period", columns="group", values="ai_broad_any_share"
    ).loc[periods_plot]
    pivot["swe_minus_ctrl"] = pivot["swe"] - pivot["ctrl"]
    pivot["adj_minus_ctrl"] = pivot["adj"] - pivot["ctrl"]
    pivot["swe_minus_adj"] = pivot["swe"] - pivot["adj"]
    axes[1].plot(
        range(len(periods_plot)),
        pivot["swe_minus_ctrl"].values,
        "-o",
        label="SWE − control",
        linewidth=2,
        markersize=8,
    )
    axes[1].plot(
        range(len(periods_plot)),
        pivot["adj_minus_ctrl"].values,
        "-o",
        label="Adjacent − control",
        linewidth=2,
        markersize=8,
    )
    axes[1].plot(
        range(len(periods_plot)),
        pivot["swe_minus_adj"].values,
        "-o",
        label="SWE − adjacent",
        linewidth=2,
        markersize=8,
    )
    axes[1].set_xticks(range(len(periods_plot)))
    axes[1].set_xticklabels(["2024 pooled", "2026 scraped"])
    axes[1].set_ylabel("Prevalence gap")
    axes[1].set_title("Cross-group AI gap: widening or narrowing?")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGS / "ai_adoption_gradient.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {FIGS / 'ai_adoption_gradient.png'}")


def figure_parallel_trends() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(TABLES / "group_period_metrics.csv")
    periods_plot = ["2024_pooled", "2026_scraped"]
    df = df[df.period.isin(periods_plot)]

    metrics = [
        ("ai_broad_any_share", "AI broad prevalence"),
        ("desc_length_mean", "Description length (chars)"),
        ("tech_count_mean", "Tech count mean"),
        ("scope_any_share", "Scope (end-to-end / cross-func)"),
        ("tool_claude_share", "Claude tool mention"),
        ("tool_copilot_share", "Copilot mention"),
    ]
    colors = {"swe": "#1f77b4", "adj": "#ff7f0e", "ctrl": "#2ca02c"}
    labels = {"swe": "SWE", "adj": "SWE-adjacent", "ctrl": "Control"}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=150)
    axes = axes.flatten()
    for i, (col, title) in enumerate(metrics):
        ax = axes[i]
        for g in ["swe", "adj", "ctrl"]:
            sub = df[df.group == g].set_index("period").loc[periods_plot]
            ax.plot(
                range(len(periods_plot)),
                sub[col].values,
                "-o",
                label=labels[g],
                color=colors[g],
                linewidth=2,
                markersize=7,
            )
        ax.set_xticks(range(len(periods_plot)))
        ax.set_xticklabels(["2024", "2026"])
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    plt.suptitle("T18 parallel trends: SWE, SWE-adjacent, control", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGS / "parallel_trends.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {FIGS / 'parallel_trends.png'}")


def figure_boundary() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(TABLES / "boundary_cosine.csv")
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    periods_plot = ["2024_pooled", "2026_scraped"]
    df = df[df.period.isin(periods_plot)]
    df = df.set_index("period").loc[periods_plot].reset_index()
    x = range(len(df))
    ax.plot(x, df["cos_swe_adj"], "-o", label="SWE ↔ SWE-adjacent", linewidth=2, markersize=8)
    ax.plot(x, df["cos_swe_ctrl"], "-o", label="SWE ↔ control", linewidth=2, markersize=8)
    ax.plot(x, df["cos_adj_ctrl"], "-o", label="SWE-adjacent ↔ control", linewidth=2, markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["2024", "2026"])
    ax.set_ylabel("TF-IDF cosine similarity (centroid)")
    ax.set_title("Cross-occupation corpus similarity over time")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "boundary_cosine.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {FIGS / 'boundary_cosine.png'}")


def main() -> None:
    t0 = time.time()
    build_did_table()
    boundary_similarity()
    per_title_deep_dive()
    figure_adoption_gradient()
    figure_parallel_trends()
    figure_boundary()
    print(f"Total elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
