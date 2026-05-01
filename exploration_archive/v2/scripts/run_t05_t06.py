from pathlib import Path

import duckdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp


ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "preprocessing/intermediate/stage8_final.parquet"

OUT_FIG_T05 = ROOT / "exploration/figures/T05"
OUT_TAB_T05 = ROOT / "exploration/tables/T05"
OUT_REP_T05 = ROOT / "exploration/reports/T05.md"
OUT_FIG_T06 = ROOT / "exploration/figures/T06"
OUT_TAB_T06 = ROOT / "exploration/tables/T06"
OUT_REP_T06 = ROOT / "exploration/reports/T06.md"


SOURCE_ORDER = ["kaggle_asaniczka", "kaggle_arshkon", "scraped"]
SOURCE_SHORT = {
    "kaggle_asaniczka": "asaniczka",
    "kaggle_arshkon": "arshkon",
    "scraped": "scraped",
}
SOURCE_COLORS = {
    "kaggle_asaniczka": "#1f77b4",
    "kaggle_arshkon": "#ff7f0e",
    "scraped": "#2ca02c",
}


def gini(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[arr >= 0]
    if arr.size == 0:
        return np.nan
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    cumx = np.cumsum(arr)
    return float((n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n)


def hhi(counts: np.ndarray) -> float:
    arr = np.asarray(counts, dtype=float)
    total = arr.sum()
    if total == 0:
        return np.nan
    shares = arr / total
    return float(np.sum(shares ** 2) * 10000)


def share_top_n(counts: np.ndarray, n: int) -> float:
    arr = np.sort(np.asarray(counts, dtype=float))[::-1]
    total = arr.sum()
    if total == 0:
        return np.nan
    return float(arr[:n].sum() / total)


def safe_jaccard(a, b) -> float:
    a = set(a)
    b = set(b)
    if not a and not b:
        return np.nan
    return float(len(a & b) / len(a | b))


def pairwise_chi2(table: pd.DataFrame, source_a: str, source_b: str) -> dict:
    sub = table[[source_a, source_b]].copy()
    sub = sub[(sub[source_a] + sub[source_b]) > 0]
    sub = sub.loc[:, (sub.sum(axis=0) > 0)]
    chi2, p, dof, _ = chi2_contingency(sub.to_numpy(), correction=False)
    return {"source_a": source_a, "source_b": source_b, "chi2": chi2, "dof": dof, "p_value": p}


def pairwise_ks(df: pd.DataFrame, metric: str, source_a: str, source_b: str) -> dict:
    a = df.loc[df["source"] == source_a, metric].dropna().to_numpy()
    b = df.loc[df["source"] == source_b, metric].dropna().to_numpy()
    stat = ks_2samp(a, b, alternative="two-sided", method="auto")
    return {
        "metric": metric,
        "source_a": source_a,
        "source_b": source_b,
        "n_a": int(a.size),
        "n_b": int(b.size),
        "ks_d": float(stat.statistic),
        "p_value": float(stat.pvalue),
    }


def fmt_pp(x: float) -> str:
    return f"{x * 100:.1f} pp"


def write_md(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines))


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None_"
    cols = list(df.columns)
    rows = [cols]
    for _, row in df.iterrows():
        rows.append(["" if pd.isna(v) else str(v) for v in row.tolist()])
    widths = [max(len(r[i]) for r in rows) for i in range(len(cols))]
    header = "| " + " | ".join(cols[i].ljust(widths[i]) for i in range(len(cols))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = ["| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(cols))) + " |" for r in rows[1:]]
    return "\n".join([header, sep, *body])


def main() -> None:
    for p in [OUT_FIG_T05, OUT_TAB_T05, OUT_FIG_T06, OUT_TAB_T06, OUT_REP_T05.parent, OUT_REP_T06.parent]:
        p.mkdir(parents=True, exist_ok=True)

    matplotlib.use("Agg")

    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT source,
               company_name_canonical,
               title_normalized,
               state_normalized,
               seniority_final,
               company_industry,
               description_length::DOUBLE AS description_length,
               core_length::DOUBLE AS core_length
        FROM read_parquet('{DATA}')
        WHERE source_platform = 'linkedin'
          AND is_english = true
          AND date_flag = 'ok'
          AND is_swe = true
        """
    ).fetchdf()

    # ---------- T05 tables ----------
    length_rows = []
    for metric in ["description_length", "core_length"]:
        for src in SOURCE_ORDER:
            s = df.loc[df["source"] == src, metric].dropna()
            length_rows.append(
                {
                    "metric": metric,
                    "source": src,
                    "n": int(s.size),
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "p10": float(s.quantile(0.10)),
                    "p90": float(s.quantile(0.90)),
                    "p95": float(s.quantile(0.95)),
                    "p99": float(s.quantile(0.99)),
                }
            )
    length_summary = pd.DataFrame(length_rows)
    length_summary.to_csv(OUT_TAB_T05 / "T05_length_summary.csv", index=False)

    ks_df = pd.DataFrame(
        [pairwise_ks(df, metric, a, b) for metric in ["description_length", "core_length"] for i, a in enumerate(SOURCE_ORDER) for b in SOURCE_ORDER[i + 1 :]]
    )
    ks_df.to_csv(OUT_TAB_T05 / "T05_length_ks_tests.csv", index=False)

    company_sets = {src: set(df.loc[df["source"] == src, "company_name_canonical"].dropna().astype(str)) for src in SOURCE_ORDER}
    company_counts = {
        src: df.loc[df["source"] == src]
        .dropna(subset=["company_name_canonical"])
        .groupby("company_name_canonical")
        .size()
        .sort_values(ascending=False)
        for src in SOURCE_ORDER
    }
    company_overlap_rows = []
    for i, a in enumerate(SOURCE_ORDER):
        for b in SOURCE_ORDER[i + 1 :]:
            top_a = set(company_counts[a].head(50).index)
            top_b = set(company_counts[b].head(50).index)
            company_overlap_rows.append(
                {
                    "source_a": a,
                    "source_b": b,
                    "company_jaccard": safe_jaccard(company_sets[a], company_sets[b]),
                    "top50_overlap_count": int(len(top_a & top_b)),
                    "top50_jaccard": safe_jaccard(top_a, top_b),
                }
            )
    company_overlap_df = pd.DataFrame(company_overlap_rows)
    company_overlap_df.to_csv(OUT_TAB_T05 / "T05_company_overlap.csv", index=False)

    state_df = df.assign(state=df["state_normalized"].fillna("Unknown")).groupby(["source", "state"]).size().reset_index(name="n")
    state_df.to_csv(OUT_TAB_T05 / "T05_state_counts.csv", index=False)
    state_pivot = state_df.pivot(index="state", columns="source", values="n").fillna(0).astype(int)
    state_chi_df = pd.DataFrame([pairwise_chi2(state_pivot, a, b) for i, a in enumerate(SOURCE_ORDER) for b in SOURCE_ORDER[i + 1 :]])
    state_chi_df.to_csv(OUT_TAB_T05 / "T05_state_chi2.csv", index=False)

    sen_df = df.loc[df["seniority_final"] != "unknown", ["source", "seniority_final"]].copy()
    sen_counts = sen_df.groupby(["source", "seniority_final"]).size().reset_index(name="n")
    sen_pivot = sen_counts.pivot(index="seniority_final", columns="source", values="n").fillna(0).astype(int)
    sen_pivot.to_csv(OUT_TAB_T05 / "T05_seniority_counts_known.csv")
    sen_chi_df = pd.DataFrame([pairwise_chi2(sen_pivot, a, b) for i, a in enumerate(SOURCE_ORDER) for b in SOURCE_ORDER[i + 1 :]])
    sen_chi_df.to_csv(OUT_TAB_T05 / "T05_seniority_chi2.csv", index=False)

    title_sets = {src: set(df.loc[df["source"] == src, "title_normalized"].dropna().astype(str)) for src in SOURCE_ORDER}
    title_counts = {
        src: df.loc[df["source"] == src]
        .dropna(subset=["title_normalized"])
        .groupby("title_normalized")
        .size()
        .sort_values(ascending=False)
        for src in SOURCE_ORDER
    }
    title_overlap_rows = []
    for i, a in enumerate(SOURCE_ORDER):
        for b in SOURCE_ORDER[i + 1 :]:
            title_overlap_rows.append(
                {
                    "source_a": a,
                    "source_b": b,
                    "title_jaccard": safe_jaccard(title_sets[a], title_sets[b]),
                    "top50_overlap_count": int(len(set(title_counts[a].head(50).index) & set(title_counts[b].head(50).index))),
                    "top50_jaccard": safe_jaccard(set(title_counts[a].head(50).index), set(title_counts[b].head(50).index)),
                }
            )
    title_overlap_df = pd.DataFrame(title_overlap_rows)
    title_overlap_df.to_csv(OUT_TAB_T05 / "T05_title_overlap.csv", index=False)

    title_source_map = (
        df.dropna(subset=["title_normalized"])
        .groupby("title_normalized")["source"]
        .agg(lambda s: tuple(sorted(set(s))))
    )
    title_freq = df.dropna(subset=["title_normalized"]).groupby("title_normalized").size()
    unique_title_rows = []
    for title, sources in title_source_map.items():
        if len(sources) != 1:
            continue
        n = int(title_freq.loc[title])
        if n < 3:
            continue
        unique_title_rows.append({"title_normalized": title, "source": sources[0], "n": n})
    unique_title_df = pd.DataFrame(unique_title_rows).sort_values(["source", "n", "title_normalized"], ascending=[True, False, True])
    unique_title_df.to_csv(OUT_TAB_T05 / "T05_unique_titles_ge3.csv", index=False)

    ind_df = df.loc[df["source"].isin(["kaggle_arshkon", "scraped"])].copy()
    ind_df["company_industry"] = ind_df["company_industry"].fillna("Unknown")
    ind_counts = ind_df.groupby(["source", "company_industry"]).size().reset_index(name="n")
    ind_counts.to_csv(OUT_TAB_T05 / "T05_industry_counts_raw.csv", index=False)
    overall_ind = ind_counts.loc[ind_counts["company_industry"] != "Unknown"].groupby("company_industry")["n"].sum().sort_values(ascending=False)
    keep_inds = list(overall_ind.head(15).index)
    ind_df["industry_bucket"] = np.where(ind_df["company_industry"].isin(keep_inds), ind_df["company_industry"], "Other/Unknown")
    ind_pivot = ind_df.groupby(["source", "industry_bucket"]).size().reset_index(name="n").pivot(index="industry_bucket", columns="source", values="n").fillna(0).astype(int)
    ind_pivot.to_csv(OUT_TAB_T05 / "T05_industry_buckets.csv")
    ind_chi_df = pd.DataFrame([pairwise_chi2(ind_pivot, "kaggle_arshkon", "scraped")])
    ind_chi_df.to_csv(OUT_TAB_T05 / "T05_industry_chi2.csv", index=False)

    # ---------- T05 figures ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=150)
    for metric, ax, title in [("description_length", axes[0], "Description length"), ("core_length", axes[1], "Core length")]:
        all_vals = df[metric].dropna().to_numpy()
        xmax = np.quantile(all_vals, 0.995)
        bins = np.linspace(0, xmax, 55)
        for src in SOURCE_ORDER:
            vals = df.loc[df["source"] == src, metric].dropna().to_numpy()
            vals = vals[vals <= xmax]
            ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.6, color=SOURCE_COLORS[src], label=SOURCE_SHORT[src])
        ax.set_title(title)
        ax.set_xlabel("Characters")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=8)
        ax.set_xlim(0, xmax)
    fig.suptitle("T05: SWE LinkedIn length distributions (trimmed at 99.5th percentile)")
    fig.tight_layout()
    fig.savefig(OUT_FIG_T05 / "T05_length_histograms.png", bbox_inches="tight")
    plt.close(fig)

    state_total = state_df.groupby("state")["n"].sum().sort_values(ascending=False)
    top_states = list(state_total.head(12).index)
    if "Unknown" in state_total.index and "Unknown" not in top_states:
        top_states.append("Unknown")
    plot_state = state_df[state_df["state"].isin(top_states)].pivot(index="state", columns="source", values="n").fillna(0).reindex(top_states)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    plot_state.plot(kind="bar", ax=ax, color=[SOURCE_COLORS[s] for s in plot_state.columns])
    ax.set_ylabel("SWE postings")
    ax.set_xlabel("State")
    ax.set_title("T05: top state counts by source")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG_T05 / "T05_state_counts_top_states.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- T06 tables ----------
    conc_rows = []
    company_detail = []
    for src in SOURCE_ORDER:
        c = (
            df.loc[df["source"] == src]
            .dropna(subset=["company_name_canonical"])
            .groupby("company_name_canonical")
            .size()
            .sort_values(ascending=False)
        )
        total = int(c.sum())
        conc_rows.append(
            {
                "source": src,
                "postings": total,
                "companies": int(c.size),
                "hhi_10k": hhi(c.values),
                "gini": gini(c.values),
                "top1_share": share_top_n(c.values, 1),
                "top5_share": share_top_n(c.values, 5),
                "top10_share": share_top_n(c.values, 10),
                "top20_share": share_top_n(c.values, 20),
            }
        )
        tmp = c.reset_index()
        tmp.columns = ["company_name_canonical", "n"]
        tmp["source"] = src
        tmp["share"] = tmp["n"] / total
        company_detail.append(tmp)
    conc_df = pd.DataFrame(conc_rows)
    conc_df.to_csv(OUT_TAB_T06 / "T06_concentration_metrics.csv", index=False)
    company_detail_df = pd.concat(company_detail, ignore_index=True)
    company_detail_df.to_csv(OUT_TAB_T06 / "T06_company_counts_by_source.csv", index=False)

    company_wide = company_detail_df.pivot_table(index="company_name_canonical", columns="source", values=["n", "share"], fill_value=0)
    company_wide.columns = [f"{a}_{b}" for a, b in company_wide.columns]
    company_wide = company_wide.reset_index()
    company_wide["max_share"] = company_wide[[c for c in company_wide.columns if c.startswith("share_")]].max(axis=1)
    company_wide = company_wide[company_wide["max_share"] >= 0.03].sort_values("max_share", ascending=False)
    company_wide.to_csv(OUT_TAB_T06 / "T06_companies_ge_3pct_any_source.csv", index=False)

    sen_known = df[df["seniority_final"] != "unknown"].copy()
    sen_known["is_junior"] = (sen_known["seniority_final"] == "entry").astype(int)
    cap_rows = []
    for src in SOURCE_ORDER:
        tmp = sen_known[sen_known["source"] == src]
        by_company = tmp.groupby("company_name_canonical").agg(total_known=("is_junior", "size"), junior=("is_junior", "sum")).reset_index()
        by_company = by_company[by_company["total_known"] > 0].copy()
        by_company["cap_weight"] = by_company["total_known"].clip(upper=10)
        raw_share = by_company["junior"].sum() / by_company["total_known"].sum()
        capped_share = (by_company["junior"] * by_company["cap_weight"] / by_company["total_known"]).sum() / by_company["cap_weight"].sum()
        cap_rows.append(
            {
                "source": src,
                "companies_with_known_seniority": int(by_company.shape[0]),
                "known_postings": int(by_company["total_known"].sum()),
                "junior_postings": int(by_company["junior"].sum()),
                "raw_junior_share": raw_share,
                "capped_junior_share_at_10": capped_share,
                "difference_pp": (capped_share - raw_share) * 100,
            }
        )
    cap_df = pd.DataFrame(cap_rows)
    cap_df.to_csv(OUT_TAB_T06 / "T06_junior_share_cap_sensitivity.csv", index=False)

    overlap = set(df.loc[df["source"] == "kaggle_arshkon", "company_name_canonical"].dropna().astype(str)) & set(
        df.loc[df["source"] == "scraped", "company_name_canonical"].dropna().astype(str)
    )
    ov_rows = []
    for src in ["kaggle_arshkon", "scraped"]:
        tmp = sen_known[sen_known["source"] == src]
        g = tmp.groupby("company_name_canonical").agg(total_known=("is_junior", "size"), junior=("is_junior", "sum")).reset_index()
        g["junior_share"] = g["junior"] / g["total_known"]
        g["source"] = src
        ov_rows.append(g)
    ov_df = pd.concat(ov_rows, ignore_index=True)
    ov_df = ov_df[ov_df["company_name_canonical"].isin(overlap)]
    ov_df = ov_df[ov_df["total_known"] >= 5]
    stable = ov_df.groupby("company_name_canonical").filter(lambda x: x.shape[0] == 2 and (x["total_known"] >= 5).all())
    stable_pivot = stable.pivot(index="company_name_canonical", columns="source", values=["total_known", "junior", "junior_share"])
    stable_pivot.columns = [f"{a}_{b}" for a, b in stable_pivot.columns]
    stable_pivot = stable_pivot.reset_index()
    stable_pivot["delta_junior_share_pp"] = (stable_pivot["junior_share_scraped"] - stable_pivot["junior_share_kaggle_arshkon"]) * 100
    stable_pivot["abs_delta_pp"] = stable_pivot["delta_junior_share_pp"].abs()
    stable_pivot = stable_pivot.sort_values("abs_delta_pp", ascending=False)
    stable_pivot.to_csv(OUT_TAB_T06 / "T06_overlap_company_seniority_junior_share.csv", index=False)

    # ---------- T06 figures ----------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    plot_conc = conc_df.set_index("source")[["top1_share", "top5_share", "top10_share", "top20_share"]].rename(index=SOURCE_SHORT)
    plot_conc.plot(kind="bar", ax=ax)
    ax.set_ylabel("Share of SWE postings")
    ax.set_xlabel("Source")
    ax.set_title("T06: concentration by source")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG_T06 / "T06_concentration_shares.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    plot_cap = cap_df.set_index("source")[["raw_junior_share", "capped_junior_share_at_10"]].rename(index=SOURCE_SHORT)
    plot_cap.plot(kind="bar", ax=ax, color=["#7f7f7f", "#d62728"])
    ax.set_ylabel("Junior share among known seniority labels")
    ax.set_xlabel("Source")
    ax.set_title("T06: junior share raw vs company-capped")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG_T06 / "T06_junior_share_raw_vs_capped.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- T05 report ----------
    lines = []
    lines.append("# T05: Cross-dataset comparability")
    lines.append("## Finding")
    lines.append(
        "The three LinkedIn SWE slices are similar enough to compare, but not interchangeable: scraped has the longest descriptions, arshkon is shortest, and the pairwise KS tests reject equality for both `description_length` and `core_length` in every comparison. Company overlap is moderate at the raw-name level but much weaker in the top-50 employer lists, and geography/seniority/title mixes all differ materially across periods."
    )
    lines.append("## Implication for analysis")
    lines.append(
        "Use source-fixed comparisons rather than pooling the three periods as if they were a single sample. Length-normalized keyword rates and within-source trend estimates are safer than raw count comparisons, and any RQ1/RQ2 statement should be checked for sensitivity to the source mix."
    )
    lines.append("## Data quality note")
    lines.append(
        "Stage 8 does not contain `description_core_llm`, so this task used the stage-8 rule-based text fields only for length diagnostics and did not rely on boilerplate-stripped text for vocabulary work. Asaniczka also has no entry-level native labels, so its seniority distribution is structurally less comparable at the junior end."
    )
    lines.append("## Action items")
    lines.append(
        "Treat scraped vs 2024 Kaggle as a cross-period comparison with measurable composition drift. For downstream analysis, keep state and title controls explicit, and avoid interpreting asaniczka entry-level patterns as a market baseline."
    )
    lines.append("")
    lines.append("### Length tests")
    lines.append(df_to_md(length_summary))
    lines.append("")
    lines.append(df_to_md(ks_df))
    lines.append("")
    lines.append("### Company overlap")
    lines.append(df_to_md(company_overlap_df))
    lines.append("")
    lines.append("### Geography")
    state_top = state_df.pivot(index="state", columns="source", values="n").fillna(0).astype(int)
    state_top["total"] = state_top.sum(axis=1)
    state_top = state_top.sort_values("total", ascending=False).head(15).reset_index()
    lines.append(df_to_md(state_top))
    lines.append("")
    lines.append(df_to_md(state_chi_df))
    lines.append("")
    lines.append("### Seniority")
    lines.append(df_to_md(sen_pivot.reset_index()))
    lines.append("")
    lines.append(df_to_md(sen_chi_df))
    lines.append("")
    lines.append("### Titles")
    lines.append(df_to_md(title_overlap_df))
    lines.append("")
    lines.append("### Unique titles to one source (count >= 3)")
    if unique_title_df.empty:
        lines.append("_None_")
    else:
        lines.append(df_to_md(unique_title_df.groupby("source").head(10).reset_index(drop=True)))
    lines.append("")
    lines.append("### Industry")
    lines.append(df_to_md(ind_pivot.reset_index()))
    lines.append("")
    lines.append(df_to_md(ind_chi_df))
    write_md(OUT_REP_T05, lines)

    # ---------- T06 report ----------
    lines = []
    lines.append("# T06: Company concentration")
    lines.append("## Finding")
    lines.append(
        "Company concentration is meaningful and source-specific: asaniczka is the most concentrated on HHI/top-share metrics, scraped is intermediate, and arshkon is the least concentrated. The seniority mix also shifts after capping employers at 10 postings, which means a handful of prolific firms do influence the junior-share estimate."
    )
    lines.append("## Implication for analysis")
    lines.append(
        "Company-level weighting matters for RQ1 and any title/seniority composition analysis. The capped sensitivity should be treated as the preferred robustness check when a few employers contribute a large fraction of postings."
    )
    lines.append("## Data quality note")
    lines.append(
        "The overlap-company comparison is limited to firms present in both arshkon and scraped with at least 5 known seniority labels in each period, so the within-company comparison is informative but not exhaustive. Asaniczka has no native entry labels, so its junior-share interpretation is inherently different from the other two sources."
    )
    lines.append("## Action items")
    lines.append(
        "Use the concentration table as a default robustness appendix. For any headline result, rerun the statistic with company capping and report whether the conclusion changes."
    )
    lines.append("")
    lines.append("### Concentration metrics")
    lines.append(df_to_md(conc_df))
    lines.append("")
    lines.append("### Companies above 3% share in any source")
    lines.append(df_to_md(company_wide))
    lines.append("")
    lines.append("### Seniority capping sensitivity")
    lines.append(df_to_md(cap_df))
    lines.append("")
    lines.append("### Overlap-company junior share comparison")
    lines.append(df_to_md(stable_pivot.head(20).reset_index(drop=True)))
    write_md(OUT_REP_T06, lines)

    print("WROTE:")
    print(OUT_REP_T05)
    print(OUT_REP_T06)
    print(OUT_TAB_T05)
    print(OUT_TAB_T06)
    print(OUT_FIG_T05)
    print(OUT_FIG_T06)


if __name__ == "__main__":
    main()
