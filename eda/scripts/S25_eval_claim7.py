"""S25_eval_claim7 — Adversarial evaluation of Claim 7 (cross-occupation rank
correlation between worker-side AI usage and employer-side AI requirement
language).

Reuses the pair table produced by S25_cross_occupation_rank.py. Adds:
  - Permutation / null-model test at n=17 (random shuffle of occupation pairings)
  - Two-cluster null: tech-vs-non-tech split only; within-cluster ranks uniform
  - Jackknife leave-one-out (full set, tech-only, control-only)
  - Time-stability: 2024 levels, 2026 levels, 2024->2026 delta
  - Additional benchmark sources (Pew 2025, Bick/Blandin/Deming 2024,
    Anthropic Economic Index occupation category shares, ADP 2025,
    Indeed Hiring Lab AI share by function)
  - Replication with blended worker benchmark that incorporates the extra
    sources

Outputs (all under eda/):
  tables/S25_eval_permutation.csv       — permutation null summary
  tables/S25_eval_jackknife.csv         — per-occupation leave-one-out
  tables/S25_eval_timestability.csv     — 2024 vs 2026 vs delta
  tables/S25_eval_extra_benchmarks.csv  — newly curated sources
  tables/S25_eval_blended.csv           — per-method correlations under blended benchmark
  figures/S25_eval_permutation.png      — null distribution vs observed
  figures/S25_eval_jackknife.png        — per-occupation leverage
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, pearsonr

REPO = Path("/home/jihgaboot/gabor/job-research")
TABLES = REPO / "eda" / "tables"
FIGS = REPO / "eda" / "figures"


def fisher_z_ci(rho, n, alpha=0.05):
    if n < 4 or abs(rho) >= 0.999999:
        return (np.nan, np.nan)
    from scipy.stats import norm
    z = 0.5 * np.log((1 + rho) / (1 - rho))
    se = 1.0 / np.sqrt(n - 3)
    crit = norm.ppf(1 - alpha / 2)
    return (np.tanh(z - crit * se), np.tanh(z + crit * se))


def load_pair():
    pair = pd.read_csv(TABLES / "S25_pair_table.csv")
    headline = pair[pair["is_headline"]].copy()
    headline = headline.dropna(subset=["worker_any_mid", "ai_canonical_rate_2026"])
    return headline


# ---------------------------------------------------------------------------
# Q1: Permutation / null-model test
# ---------------------------------------------------------------------------

def permutation_null(x, y, n_perm=10_000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    y = np.asarray(y)
    rhos = np.empty(n_perm)
    for i in range(n_perm):
        rhos[i] = spearmanr(x, rng.permutation(y))[0]
    return rhos


def two_cluster_null(n_tech, n_ctrl, worker_tech, worker_ctrl,
                     employer_tech, employer_ctrl, n_perm=10_000, seed=1):
    """Shuffle WITHIN cluster only. Preserves the tech>=non-tech level gap,
    randomizes within-cluster rank. If the observed rho is well above this
    null, the signal is not just 'tech vs non-tech'."""
    rng = np.random.default_rng(seed)
    w = np.concatenate([worker_tech, worker_ctrl])
    rhos = np.empty(n_perm)
    for i in range(n_perm):
        e_t = rng.permutation(employer_tech)
        e_c = rng.permutation(employer_ctrl)
        e = np.concatenate([e_t, e_c])
        rhos[i] = spearmanr(w, e)[0]
    return rhos


# ---------------------------------------------------------------------------
# Q2: Jackknife leave-one-out
# ---------------------------------------------------------------------------

def jackknife(df, x_col, y_col, label):
    rows = []
    full_rho, _ = spearmanr(df[x_col], df[y_col])
    for i in range(len(df)):
        d = df.drop(df.index[i])
        rho, _ = spearmanr(d[x_col], d[y_col])
        rows.append(dict(
            scope=label,
            dropped=df.iloc[i]["subgroup"],
            analysis_group=df.iloc[i]["analysis_group"],
            spearman_loo=rho,
            delta_from_full=rho - full_rho,
            n_remaining=len(d),
        ))
    return pd.DataFrame(rows), full_rho


# ---------------------------------------------------------------------------
# Q4: Time-stability
# ---------------------------------------------------------------------------

def time_stability(headline):
    out = []
    for label, ycol in [
        ("2024_levels_canonical", "ai_canonical_rate_2024"),
        ("2026_levels_canonical", "ai_canonical_rate_2026"),
        ("delta_2024_2026_canonical", "delta_2024_2026_canonical"),
        ("2024_levels_strict_v1", "ai_strict_v1_rate_2024"),
        ("2026_levels_strict_v1", "ai_strict_v1_rate_2026"),
        ("delta_2024_2026_strict_v1", "delta_2024_2026_strict_v1"),
    ]:
        d = headline.dropna(subset=["worker_any_mid", ycol])
        rho, p = spearmanr(d["worker_any_mid"], d[ycol])
        tau, _ = kendalltau(d["worker_any_mid"], d[ycol])
        r, _ = pearsonr(d["worker_any_mid"], d[ycol])
        flo, fhi = fisher_z_ci(rho, len(d))
        out.append(dict(
            scope="full", y=label, n=len(d),
            spearman=rho, spearman_p=p, kendall=tau, pearson=r,
            fisher_lo=flo, fisher_hi=fhi,
        ))
        # Restricted to tech-only and control-only
        for name, mask in [
            ("tech_only", d["analysis_group"].isin(["swe", "swe_adjacent"])),
            ("control_only", d["analysis_group"] == "control"),
        ]:
            dd = d[mask]
            if len(dd) < 4:
                continue
            rho, p = spearmanr(dd["worker_any_mid"], dd[ycol])
            tau, _ = kendalltau(dd["worker_any_mid"], dd[ycol])
            flo, fhi = fisher_z_ci(rho, len(dd))
            out.append(dict(
                scope=name, y=label, n=len(dd),
                spearman=rho, spearman_p=p, kendall=tau, pearson=np.nan,
                fisher_lo=flo, fisher_hi=fhi,
            ))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Q3: Additional benchmark sources (layer onto the existing table)
# ---------------------------------------------------------------------------
# Values are "any work-related use" rates where possible, chosen to be
# directly comparable to the existing S25 worker-any-mid aggregations.
# All access_date = 2026-04-21.

EXTRA_BENCHMARKS = pd.DataFrame([
    # Pew Research (Sept 2025) — workers using AI "at least some" of work time.
    # 28% of workers with bachelor's+; 16% with some college or less. Pew doesn't
    # publish occupation-level cuts directly but the education split is the
    # closest proxy for tech vs non-tech (SWE and data roles are bachelor's+-heavy).
    dict(subgroup="other_swe", group="swe",
         source="Pew Research Sep 2025 (college+ workers)",
         rate_any=0.28, rate_daily=0.02,
         url="https://www.pewresearch.org/short-reads/2025/10/06/about-1-in-5-us-workers-now-use-ai-in-their-job-up-since-last-year/",
         notes="Upper bound on any-use for college-educated workers."),
    # Bick/Blandin/Deming NBER 2024 — computer/math occupations 49.6% workplace
    # any-use. Treat as a corroboration for other_swe (already near 0.63-0.90 range).
    dict(subgroup="other_swe", group="swe",
         source="Bick Blandin Deming NBER 2024 (w32966)",
         rate_any=0.496, rate_daily=0.09,
         url="https://www.nber.org/papers/w32966",
         notes="Computer/mathematical occupations, work any-use 49.6%; overall 23%."),
    # Anthropic Economic Index (Mar 2026) — ~35% of Claude.ai conversations are
    # computer/mathematical tasks; treat as a RELATIVE share that corroborates
    # the tech-heavy concentration but NOT a worker-use rate per se.
    # (not directly comparable; we DO NOT add to the blended mean.)
    dict(subgroup="data_analyst", group="swe_adjacent",
         source="Anthropic Economic Index Mar 2026 (Claude.ai task mix)",
         rate_any=None, rate_daily=None,
         url="https://www.anthropic.com/research/economic-index-march-2026-report",
         notes="Share of Claude.ai conversations, not a worker-use rate. Context only."),
    # ADP Research Institute People at Work 2025 — 43% frequently use AI,
    # 20% essentially daily. Cross-occupation macro average. Not subgroup-specific
    # but establishes a macro ceiling (~43%).
    dict(subgroup="other_control", group="control",
         source="ADP Research People at Work 2025",
         rate_any=0.43, rate_daily=0.20,
         url="https://www.adpresearch.com/wp-content/uploads/2025/02/PAW2025_AI-Final.pdf",
         notes="Cross-occupation macro, treat as floor for control-cluster average."),
    # Indeed Hiring Lab AI at Work Sep 2025 — 45% of data & analytics postings
    # contain AI-related terms, vs 15% marketing, 9% HR. This is EMPLOYER-side,
    # so useful as convergent evidence on the ORDERING not the level.
    dict(subgroup="data_analyst", group="swe_adjacent",
         source="Indeed Hiring Lab Sep 2025 (data & analytics postings)",
         rate_any=None, rate_daily=None,
         url="https://www.hiringlab.org/2025/09/23/ai-at-work-report-2025-how-genai-is-rewiring-the-dna-of-jobs/",
         notes="Employer-side AI share, not worker use. Context only."),
])


def blended_benchmark_analysis(headline):
    """Merge extra benchmarks where we have comparable any-use rates and
    recompute M1/M7/M8 rank correlations."""
    extra_usable = EXTRA_BENCHMARKS.dropna(subset=["rate_any"]).copy()
    # Keep only subgroups that exist in headline
    mask = extra_usable["subgroup"].isin(headline["subgroup"])
    extra_usable = extra_usable[mask]
    # Build blended = mean of S25 benchmarks + extras per subgroup
    # We need S25_worker_benchmarks.csv for the originals.
    orig = pd.read_csv(TABLES / "S25_worker_benchmarks.csv")
    combo = pd.concat([orig[["subgroup", "rate_any"]],
                       extra_usable[["subgroup", "rate_any"]]], ignore_index=True)
    blended = combo.groupby("subgroup")["rate_any"].mean().rename("worker_any_blended")
    h = headline.merge(blended, on="subgroup", how="left")

    rows = []

    def _add(name, sub):
        sub = sub.dropna(subset=["worker_any_blended", "ai_canonical_rate_2026"])
        if len(sub) < 4:
            return
        rho, p = spearmanr(sub["worker_any_blended"], sub["ai_canonical_rate_2026"])
        flo, fhi = fisher_z_ci(rho, len(sub))
        rows.append(dict(scope=name, n=len(sub), spearman=rho, p=p,
                         fisher_lo=flo, fisher_hi=fhi))

    _add("full_n17", h)
    _add("tech_only", h[h["analysis_group"].isin(["swe", "swe_adjacent"])])
    _add("control_only", h[h["analysis_group"] == "control"])
    return pd.DataFrame(rows), h


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    headline = load_pair()
    print(f"[eval] n headline = {len(headline)}")

    # Q1a: full permutation
    x = headline["worker_any_mid"].values
    y = headline["ai_canonical_rate_2026"].values
    observed_rho = spearmanr(x, y)[0]
    perm = permutation_null(x, y, n_perm=10_000, seed=0)
    p_perm = float((perm >= observed_rho).mean())
    perm_q = np.percentile(perm, [50, 95, 97.5, 99, 99.5])
    print(f"[eval] observed rho = {observed_rho:.4f}, perm p = {p_perm:.4f}")

    # Q1b: two-cluster null (preserve tech vs non-tech level gap)
    tech_mask = headline["analysis_group"].isin(["swe", "swe_adjacent"])
    ctrl_mask = headline["analysis_group"] == "control"
    two_null = two_cluster_null(
        tech_mask.sum(), ctrl_mask.sum(),
        headline.loc[tech_mask, "worker_any_mid"].values,
        headline.loc[ctrl_mask, "worker_any_mid"].values,
        headline.loc[tech_mask, "ai_canonical_rate_2026"].values,
        headline.loc[ctrl_mask, "ai_canonical_rate_2026"].values,
        n_perm=10_000, seed=1,
    )
    p_two = float((two_null >= observed_rho).mean())
    two_q = np.percentile(two_null, [50, 95, 97.5, 99, 99.5])
    print(f"[eval] two-cluster null median = {two_q[0]:.4f}, 97.5% = {two_q[2]:.4f}, p = {p_two:.4f}")

    perm_summary = pd.DataFrame([
        dict(null="uniform_shuffle", n_perm=10_000, observed=observed_rho,
             median=perm_q[0], p95=perm_q[1], p97_5=perm_q[2],
             p99=perm_q[3], p99_5=perm_q[4], p_value_right=p_perm),
        dict(null="two_cluster_within_shuffle", n_perm=10_000, observed=observed_rho,
             median=two_q[0], p95=two_q[1], p97_5=two_q[2],
             p99=two_q[3], p99_5=two_q[4], p_value_right=p_two),
    ])
    perm_summary.to_csv(TABLES / "S25_eval_permutation.csv", index=False)

    # Permutation figure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(perm, bins=50, alpha=0.5, label=f"uniform shuffle (median {perm_q[0]:+.2f})", color="#888")
    ax.hist(two_null, bins=50, alpha=0.5, label=f"two-cluster null (median {two_q[0]:+.2f})", color="#e67e22")
    ax.axvline(observed_rho, color="#c0392b", lw=2, label=f"observed rho = {observed_rho:+.3f}")
    ax.set_xlabel("Spearman rho")
    ax.set_ylabel("count")
    ax.set_title("S25 eval Q1: permutation null vs two-cluster null (n=17)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "S25_eval_permutation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Q2: jackknife
    jk_full, full_rho_full = jackknife(headline, "worker_any_mid", "ai_canonical_rate_2026", "full_n17")
    tech = headline[tech_mask]
    ctrl = headline[ctrl_mask]
    jk_tech, full_rho_tech = jackknife(tech, "worker_any_mid", "ai_canonical_rate_2026", "tech_only_n11")
    jk_ctrl, full_rho_ctrl = jackknife(ctrl, "worker_any_mid", "ai_canonical_rate_2026", "control_only_n6")
    jk = pd.concat([jk_full, jk_tech, jk_ctrl], ignore_index=True)
    jk.to_csv(TABLES / "S25_eval_jackknife.csv", index=False)
    print(f"[eval] jackknife written ({len(jk)} rows)")
    print(f"  full_rho = {full_rho_full:.4f}, tech = {full_rho_tech:.4f}, ctrl = {full_rho_ctrl:.4f}")

    # Jackknife figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, df, title, full in zip(
        axes, [jk_full, jk_tech, jk_ctrl],
        [f"full (n=17, full rho={full_rho_full:+.3f})",
         f"tech only (n=11, full rho={full_rho_tech:+.3f})",
         f"control only (n=6, full rho={full_rho_ctrl:+.3f})"],
        [full_rho_full, full_rho_tech, full_rho_ctrl],
    ):
        d = df.sort_values("spearman_loo")
        ax.barh(d["dropped"], d["spearman_loo"], color="#3070b3", edgecolor="black")
        ax.axvline(full, color="#c0392b", lw=1.5, linestyle="--", label="full")
        ax.set_xlabel("leave-one-out Spearman rho")
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("S25 eval Q2: jackknife leverage (each bar = drop that occupation)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "S25_eval_jackknife.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Q4: time-stability
    ts = time_stability(headline)
    ts.to_csv(TABLES / "S25_eval_timestability.csv", index=False)
    print("\n[eval] time stability:")
    print(ts.to_string(index=False))

    # Q3: extra benchmarks + blended recomputation
    EXTRA_BENCHMARKS.to_csv(TABLES / "S25_eval_extra_benchmarks.csv", index=False)
    blended_res, _ = blended_benchmark_analysis(headline)
    blended_res.to_csv(TABLES / "S25_eval_blended.csv", index=False)
    print("\n[eval] blended worker benchmark (S25 + extras):")
    print(blended_res.to_string(index=False))

    print("\n[eval] DONE.")


if __name__ == "__main__":
    main()
