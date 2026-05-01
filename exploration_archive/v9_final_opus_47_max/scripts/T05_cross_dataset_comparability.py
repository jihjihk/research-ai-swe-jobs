"""T05. Cross-dataset comparability

Test whether the three datasets are measuring the same thing by running
pairwise comparisons across description length, company overlap, geographic
/seniority/title distributions; plus 2024 within-baseline calibration and
LinkedIn-vs-Indeed cross-validation.

SWE, LinkedIn as primary; Indeed used only in the platform stability cross-
validation (step 9).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data" / "unified.parquet"
TBL = ROOT / "exploration" / "tables" / "T05"
FIG = ROOT / "exploration" / "figures" / "T05"
TBL.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")
con.execute("SET threads=8")

# Filters — use DEFAULT filter set described in the dispatch
BASE_WHERE = "source_platform='linkedin' AND is_english AND date_flag='ok' AND is_swe"
SOURCES = ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]


def savecsv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TBL / name, index=False)
    print(f"  wrote {TBL / name} ({len(df)} rows)")


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Step 1. Description length KS + histograms
# ---------------------------------------------------------------------------
print("=" * 78)
print("STEP 1. Description length (KS + histograms)")
print("=" * 78)


def fetch_lengths(source: str, include_aggregators: bool) -> np.ndarray:
    where = f"{BASE_WHERE} AND source='{source}'"
    if not include_aggregators:
        where += " AND NOT is_aggregator"
    q = f"SELECT description_length FROM '{DATA}' WHERE {where} AND description_length IS NOT NULL"
    return con.execute(q).fetchdf()["description_length"].values


rows = []
for agg_mode, agg_flag in [("all", True), ("non_aggregator", False)]:
    lens = {src: fetch_lengths(src, agg_flag) for src in SOURCES}
    for i in range(len(SOURCES)):
        for j in range(i + 1, len(SOURCES)):
            a, b = SOURCES[i], SOURCES[j]
            ks = stats.ks_2samp(lens[a], lens[b])
            rows.append(
                {
                    "aggregator_mode": agg_mode,
                    "source_a": a,
                    "source_b": b,
                    "n_a": int(len(lens[a])),
                    "n_b": int(len(lens[b])),
                    "mean_a": float(np.mean(lens[a])),
                    "mean_b": float(np.mean(lens[b])),
                    "median_a": float(np.median(lens[a])),
                    "median_b": float(np.median(lens[b])),
                    "ks_statistic": float(ks.statistic),
                    "ks_pvalue": float(ks.pvalue),
                }
            )
length_ks_df = pd.DataFrame(rows)
savecsv(length_ks_df, "description_length_ks.csv")
print(length_ks_df.to_string())

# Overlapping histogram (LinkedIn SWE, non-aggregator)
plt.figure(figsize=(9, 5))
bins = np.linspace(0, 15000, 80)
for src in SOURCES:
    arr = fetch_lengths(src, include_aggregators=False)
    plt.hist(arr, bins=bins, alpha=0.45, density=True, label=f"{src} (n={len(arr):,})")
plt.xlabel("description_length (chars)")
plt.ylabel("density")
plt.title("SWE description length, LinkedIn, non-aggregator")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "description_length_hist.png", dpi=120)
plt.close()
print(f"  wrote {FIG / 'description_length_hist.png'}")


# ---------------------------------------------------------------------------
# Step 2. Company overlap Jaccard (pairwise; top-50 overlap)
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("STEP 2. Company overlap (Jaccard + top-50)")
print("=" * 78)


def fetch_companies(source: str, include_aggregators: bool) -> pd.DataFrame:
    where = f"{BASE_WHERE} AND source='{source}' AND company_name_canonical IS NOT NULL"
    if not include_aggregators:
        where += " AND NOT is_aggregator"
    q = f"""SELECT company_name_canonical, COUNT(*) as n
            FROM '{DATA}' WHERE {where}
            GROUP BY company_name_canonical"""
    return con.execute(q).fetchdf()


jac_rows = []
top50_rows = []
for agg_mode, agg_flag in [("all", True), ("non_aggregator", False)]:
    comp = {src: fetch_companies(src, agg_flag) for src in SOURCES}
    for i in range(len(SOURCES)):
        for j in range(i + 1, len(SOURCES)):
            a, b = SOURCES[i], SOURCES[j]
            set_a = set(comp[a]["company_name_canonical"])
            set_b = set(comp[b]["company_name_canonical"])
            top_a = set(comp[a].nlargest(50, "n")["company_name_canonical"])
            top_b = set(comp[b].nlargest(50, "n")["company_name_canonical"])
            jac_rows.append(
                {
                    "aggregator_mode": agg_mode,
                    "source_a": a,
                    "source_b": b,
                    "n_companies_a": len(set_a),
                    "n_companies_b": len(set_b),
                    "jaccard": jaccard(set_a, set_b),
                    "intersection": len(set_a & set_b),
                    "union": len(set_a | set_b),
                    "top50_jaccard": jaccard(top_a, top_b),
                    "top50_intersection": len(top_a & top_b),
                }
            )
            if agg_mode == "non_aggregator":
                # Save top-50 overlap table
                merged = comp[a].merge(
                    comp[b], on="company_name_canonical", how="outer", suffixes=("_a", "_b")
                ).fillna(0)
                merged["min_n"] = merged[["n_a", "n_b"]].min(axis=1)
                merged = merged.sort_values("min_n", ascending=False).head(50)
                merged["source_a"] = a
                merged["source_b"] = b
                top50_rows.append(merged)

savecsv(pd.DataFrame(jac_rows), "company_jaccard.csv")
if top50_rows:
    savecsv(pd.concat(top50_rows, ignore_index=True), "company_top50_overlap.csv")
print(pd.DataFrame(jac_rows).to_string())


# ---------------------------------------------------------------------------
# Step 3. Geographic (state-level) chi-squared
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("STEP 3. State-level chi-squared")
print("=" * 78)


def fetch_states(source: str, include_aggregators: bool) -> pd.DataFrame:
    where = (
        f"{BASE_WHERE} AND source='{source}' AND state_normalized IS NOT NULL "
        "AND NOT is_multi_location"
    )
    if not include_aggregators:
        where += " AND NOT is_aggregator"
    q = f"""SELECT state_normalized, COUNT(*) as n
            FROM '{DATA}' WHERE {where}
            GROUP BY state_normalized"""
    return con.execute(q).fetchdf()


state_rows = []
state_share_tables = []
for agg_mode, agg_flag in [("all", True), ("non_aggregator", False)]:
    states = {src: fetch_states(src, agg_flag) for src in SOURCES}
    # shared state set = union of all
    all_states = sorted(set().union(*[set(s["state_normalized"]) for s in states.values()]))
    mat = pd.DataFrame(
        {src: states[src].set_index("state_normalized").reindex(all_states, fill_value=0)["n"]
         for src in SOURCES}
    )
    # remove all-zero rows (shouldn't exist, but safe)
    mat = mat[mat.sum(axis=1) > 0]
    if agg_mode == "non_aggregator":
        share = mat.div(mat.sum(axis=0), axis=1).reset_index().rename(
            columns={"index": "state_normalized"}
        )
        share.to_csv(TBL / "state_shares.csv", index=False)
    for i in range(len(SOURCES)):
        for j in range(i + 1, len(SOURCES)):
            a, b = SOURCES[i], SOURCES[j]
            sub = mat[[a, b]]
            sub = sub[sub.sum(axis=1) >= 5]  # drop tiny rows
            if len(sub) < 2:
                continue
            chi2, p, dof, _ = stats.chi2_contingency(sub.T.values)
            state_rows.append(
                {
                    "aggregator_mode": agg_mode,
                    "source_a": a,
                    "source_b": b,
                    "n_states": int(len(sub)),
                    "total_a": int(sub[a].sum()),
                    "total_b": int(sub[b].sum()),
                    "chi2": float(chi2),
                    "dof": int(dof),
                    "p_value": float(p),
                    # Cramer's V
                    "cramers_v": float(
                        np.sqrt(chi2 / (sub.values.sum() * min(sub.shape[0] - 1, 1)))
                    ),
                }
            )
savecsv(pd.DataFrame(state_rows), "state_chi2.csv")
print(pd.DataFrame(state_rows).to_string())


# ---------------------------------------------------------------------------
# Step 4. Seniority distributions chi-squared
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("STEP 4. Seniority (seniority_final) chi-squared")
print("=" * 78)


def fetch_seniority(source: str, include_aggregators: bool) -> pd.DataFrame:
    where = f"{BASE_WHERE} AND source='{source}' AND seniority_final IS NOT NULL AND seniority_final != 'unknown'"
    if not include_aggregators:
        where += " AND NOT is_aggregator"
    q = f"""SELECT seniority_final, COUNT(*) as n
            FROM '{DATA}' WHERE {where}
            GROUP BY seniority_final"""
    return con.execute(q).fetchdf()


sen_rows = []
for agg_mode, agg_flag in [("all", True), ("non_aggregator", False)]:
    sens = {src: fetch_seniority(src, agg_flag) for src in SOURCES}
    all_lvls = ["entry", "associate", "mid-senior", "director"]
    mat = pd.DataFrame(
        {src: sens[src].set_index("seniority_final").reindex(all_lvls, fill_value=0)["n"]
         for src in SOURCES}
    )
    if agg_mode == "non_aggregator":
        share = mat.div(mat.sum(axis=0), axis=1).reset_index().rename(
            columns={"index": "seniority_final"}
        )
        share.to_csv(TBL / "seniority_shares.csv", index=False)
    for i in range(len(SOURCES)):
        for j in range(i + 1, len(SOURCES)):
            a, b = SOURCES[i], SOURCES[j]
            sub = mat[[a, b]]
            sub = sub[sub.sum(axis=1) > 0]
            chi2, p, dof, _ = stats.chi2_contingency(sub.T.values)
            sen_rows.append(
                {
                    "aggregator_mode": agg_mode,
                    "source_a": a,
                    "source_b": b,
                    "total_a": int(sub[a].sum()),
                    "total_b": int(sub[b].sum()),
                    "chi2": float(chi2),
                    "dof": int(dof),
                    "p_value": float(p),
                    "cramers_v": float(
                        np.sqrt(chi2 / (sub.values.sum() * min(sub.shape[0] - 1, 1)))
                    ),
                }
            )
savecsv(pd.DataFrame(sen_rows), "seniority_chi2.csv")
print(pd.DataFrame(sen_rows).to_string())


# ---------------------------------------------------------------------------
# Step 5. Title vocabulary Jaccard
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("STEP 5. Title vocabulary Jaccard")
print("=" * 78)


def fetch_titles(source: str, include_aggregators: bool, min_n: int = 1) -> pd.DataFrame:
    where = f"{BASE_WHERE} AND source='{source}' AND title_normalized IS NOT NULL"
    if not include_aggregators:
        where += " AND NOT is_aggregator"
    q = f"""SELECT title_normalized, COUNT(*) as n
            FROM '{DATA}' WHERE {where}
            GROUP BY title_normalized HAVING COUNT(*) >= {min_n}"""
    return con.execute(q).fetchdf()


title_rows = []
unique_samples = []
for agg_mode, agg_flag in [("all", True), ("non_aggregator", False)]:
    for min_n in [1, 3]:
        t = {src: fetch_titles(src, agg_flag, min_n=min_n) for src in SOURCES}
        for i in range(len(SOURCES)):
            for j in range(i + 1, len(SOURCES)):
                a, b = SOURCES[i], SOURCES[j]
                sa, sb = set(t[a]["title_normalized"]), set(t[b]["title_normalized"])
                title_rows.append(
                    {
                        "aggregator_mode": agg_mode,
                        "min_title_count": min_n,
                        "source_a": a,
                        "source_b": b,
                        "n_titles_a": len(sa),
                        "n_titles_b": len(sb),
                        "jaccard": jaccard(sa, sb),
                        "intersection": len(sa & sb),
                        "union": len(sa | sb),
                    }
                )
        # Collect titles unique to one source (non-aggregator, min_n=3)
        if agg_mode == "non_aggregator" and min_n == 3:
            for src in SOURCES:
                others = set().union(
                    *[set(t[o]["title_normalized"]) for o in SOURCES if o != src]
                )
                own = set(t[src]["title_normalized"])
                unique = own - others
                top_unique = t[src][t[src]["title_normalized"].isin(unique)].nlargest(30, "n")
                top_unique = top_unique.assign(unique_to=src)
                unique_samples.append(top_unique)
savecsv(pd.DataFrame(title_rows), "title_jaccard.csv")
if unique_samples:
    savecsv(pd.concat(unique_samples, ignore_index=True), "titles_unique_per_source.csv")
print(pd.DataFrame(title_rows).to_string())


# ---------------------------------------------------------------------------
# Step 6. Industry chi-squared (arshkon vs scraped; asaniczka has no industry)
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("STEP 6. Industry chi-squared (arshkon vs scraped)")
print("=" * 78)


def fetch_industry(source: str, include_aggregators: bool) -> pd.DataFrame:
    where = f"{BASE_WHERE} AND source='{source}' AND company_industry IS NOT NULL"
    if not include_aggregators:
        where += " AND NOT is_aggregator"
    q = f"""SELECT company_industry, COUNT(*) as n
            FROM '{DATA}' WHERE {where}
            GROUP BY company_industry"""
    return con.execute(q).fetchdf()


ind_rows = []
for agg_mode, agg_flag in [("all", True), ("non_aggregator", False)]:
    ar = fetch_industry("kaggle_arshkon", agg_flag)
    sc = fetch_industry("scraped", agg_flag)
    if len(ar) == 0 or len(sc) == 0:
        continue
    all_ind = sorted(set(ar["company_industry"]) | set(sc["company_industry"]))
    mat = pd.DataFrame(
        {
            "kaggle_arshkon": ar.set_index("company_industry").reindex(all_ind, fill_value=0)["n"],
            "scraped": sc.set_index("company_industry").reindex(all_ind, fill_value=0)["n"],
        }
    )
    mat = mat[mat.sum(axis=1) >= 10]
    if agg_mode == "non_aggregator":
        share = mat.div(mat.sum(axis=0), axis=1).reset_index().rename(
            columns={"index": "company_industry"}
        )
        share.to_csv(TBL / "industry_shares.csv", index=False)
    chi2, p, dof, _ = stats.chi2_contingency(mat.T.values)
    ind_rows.append(
        {
            "aggregator_mode": agg_mode,
            "n_industries": int(len(mat)),
            "total_arshkon": int(mat["kaggle_arshkon"].sum()),
            "total_scraped": int(mat["scraped"].sum()),
            "chi2": float(chi2),
            "dof": int(dof),
            "p_value": float(p),
            "cramers_v": float(
                np.sqrt(chi2 / (mat.values.sum() * min(mat.shape[0] - 1, 1)))
            ),
        }
    )
savecsv(pd.DataFrame(ind_rows), "industry_chi2.csv")
print(pd.DataFrame(ind_rows).to_string())


# ---------------------------------------------------------------------------
# Step 8. Within-2024 calibration (arshkon vs asaniczka only, same underlying period)
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("STEP 8. Within-2024 calibration (arshkon vs asaniczka)")
print("=" * 78)
# We already have arshkon vs asaniczka in all pairwise tables; write a
# consolidated view as "baseline" for later interpretation.
cal = {
    "description_length_ks": length_ks_df[
        (length_ks_df.source_a == "kaggle_arshkon") & (length_ks_df.source_b == "kaggle_asaniczka")
    ],
    "company_jaccard": pd.DataFrame(jac_rows)[
        (pd.DataFrame(jac_rows).source_a == "kaggle_arshkon")
        & (pd.DataFrame(jac_rows).source_b == "kaggle_asaniczka")
    ],
    "seniority_chi2": pd.DataFrame(sen_rows)[
        (pd.DataFrame(sen_rows).source_a == "kaggle_arshkon")
        & (pd.DataFrame(sen_rows).source_b == "kaggle_asaniczka")
    ],
    "title_jaccard": pd.DataFrame(title_rows)[
        (pd.DataFrame(title_rows).source_a == "kaggle_arshkon")
        & (pd.DataFrame(title_rows).source_b == "kaggle_asaniczka")
    ],
    "state_chi2": pd.DataFrame(state_rows)[
        (pd.DataFrame(state_rows).source_a == "kaggle_arshkon")
        & (pd.DataFrame(state_rows).source_b == "kaggle_asaniczka")
    ],
}
with open(TBL / "calibration_2024.json", "w") as fh:
    json.dump({k: v.to_dict(orient="records") for k, v in cal.items()}, fh, indent=2, default=str)
print("  wrote calibration_2024.json")


# ---------------------------------------------------------------------------
# Step 9. Platform labeling stability — top 20 SWE titles in arshkon ∩ scraped
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("STEP 9. Title-level seniority_native + YOE stability (arshkon vs scraped)")
print("=" * 78)

# Compute top 20 SWE titles shared between arshkon and scraped (LinkedIn, non-agg, is_swe)
shared_titles_df = con.execute(
    f"""
WITH titles AS (
  SELECT source, title_normalized, COUNT(*) as n
  FROM '{DATA}'
  WHERE {BASE_WHERE} AND NOT is_aggregator AND title_normalized IS NOT NULL
    AND source IN ('kaggle_arshkon', 'scraped')
  GROUP BY source, title_normalized
),
wide_titles AS (
  SELECT title_normalized,
         SUM(CASE WHEN source='kaggle_arshkon' THEN n ELSE 0 END) as n_ar,
         SUM(CASE WHEN source='scraped' THEN n ELSE 0 END) as n_sc
  FROM titles
  GROUP BY title_normalized
)
SELECT *, LEAST(n_ar, n_sc) as min_n FROM wide_titles
WHERE n_ar > 0 AND n_sc > 0
ORDER BY min_n DESC LIMIT 20
"""
).fetchdf()
savecsv(shared_titles_df, "shared_titles_top20.csv")
print(shared_titles_df.to_string())

# Per-title: seniority_native distribution + yoe distribution
rows = []
for title in shared_titles_df["title_normalized"]:
    # Build a safe param via parameterization
    q = f"""
    SELECT source,
           seniority_native,
           COUNT(*) as n
    FROM '{DATA}'
    WHERE {BASE_WHERE} AND NOT is_aggregator AND title_normalized = ?
      AND source IN ('kaggle_arshkon', 'scraped')
    GROUP BY source, seniority_native
    """
    d = con.execute(q, [title]).fetchdf()
    for _, r in d.iterrows():
        rows.append(
            {
                "title_normalized": title,
                "source": r["source"],
                "seniority_native": r["seniority_native"],
                "n": int(r["n"]),
            }
        )
native_df = pd.DataFrame(rows)
savecsv(native_df, "shared_titles_seniority_native.csv")

# YOE distribution per title/source (LLM primary, rule fallback)
yoe_rows = []
for title in shared_titles_df["title_normalized"]:
    q = f"""
    SELECT source,
           AVG(yoe_min_years_llm) as mean_llm,
           MEDIAN(yoe_min_years_llm) as median_llm,
           COUNT(*) FILTER (WHERE yoe_min_years_llm IS NOT NULL) as n_llm,
           AVG(yoe_extracted) as mean_rule,
           MEDIAN(yoe_extracted) as median_rule,
           COUNT(*) FILTER (WHERE yoe_extracted IS NOT NULL) as n_rule,
           COUNT(*) as n
    FROM '{DATA}'
    WHERE {BASE_WHERE} AND NOT is_aggregator AND title_normalized = ?
      AND source IN ('kaggle_arshkon','scraped')
    GROUP BY source
    """
    d = con.execute(q, [title]).fetchdf()
    for _, r in d.iterrows():
        yoe_rows.append({"title_normalized": title, **r.to_dict()})
yoe_df = pd.DataFrame(yoe_rows)
savecsv(yoe_df, "shared_titles_yoe.csv")
print(yoe_df.to_string())

# Indeed cross-validation for the same top-20 titles: rule-based yoe_extracted <= 2 as primary entry indicator
indeed_rows = []
for title in shared_titles_df["title_normalized"]:
    q = f"""
    SELECT source_platform, source,
           COUNT(*) as n,
           COUNT(*) FILTER (WHERE yoe_extracted IS NOT NULL AND yoe_extracted <= 2) as n_rule_le2,
           COUNT(*) FILTER (WHERE yoe_extracted IS NOT NULL) as n_with_rule,
           COUNT(*) FILTER (WHERE seniority_final='entry') as n_sen_entry,
           COUNT(*) FILTER (WHERE seniority_final IS NOT NULL AND seniority_final != 'unknown') as n_sen_known
    FROM '{DATA}'
    WHERE is_swe AND is_english AND date_flag='ok' AND NOT is_aggregator
      AND title_normalized = ?
    GROUP BY source_platform, source
    """
    d = con.execute(q, [title]).fetchdf()
    for _, r in d.iterrows():
        indeed_rows.append({"title_normalized": title, **r.to_dict()})
platform_df = pd.DataFrame(indeed_rows)
savecsv(platform_df, "shared_titles_platform_cross.csv")
print("  head of platform cross-validation:")
print(platform_df.head(20).to_string())


# ---------------------------------------------------------------------------
# Aggregated summary: effect-sizes table
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("SUMMARY effect-size table (non-aggregator only)")
print("=" * 78)
summary = []
for i in range(len(SOURCES)):
    for j in range(i + 1, len(SOURCES)):
        a, b = SOURCES[i], SOURCES[j]
        length = length_ks_df[(length_ks_df.aggregator_mode == "non_aggregator") &
                              (length_ks_df.source_a == a) &
                              (length_ks_df.source_b == b)]
        jac = pd.DataFrame(jac_rows)
        jac = jac[(jac.aggregator_mode == "non_aggregator") & (jac.source_a == a) & (jac.source_b == b)]
        sen = pd.DataFrame(sen_rows)
        sen = sen[(sen.aggregator_mode == "non_aggregator") & (sen.source_a == a) & (sen.source_b == b)]
        st = pd.DataFrame(state_rows)
        st = st[(st.aggregator_mode == "non_aggregator") & (st.source_a == a) & (st.source_b == b)]
        tit = pd.DataFrame(title_rows)
        tit = tit[(tit.aggregator_mode == "non_aggregator") & (tit.min_title_count == 1) & (tit.source_a == a) & (tit.source_b == b)]
        summary.append(
            {
                "source_a": a,
                "source_b": b,
                "desc_len_ks_D": float(length["ks_statistic"].iloc[0]) if len(length) else np.nan,
                "desc_len_ks_p": float(length["ks_pvalue"].iloc[0]) if len(length) else np.nan,
                "company_jaccard": float(jac["jaccard"].iloc[0]) if len(jac) else np.nan,
                "top50_company_jaccard": float(jac["top50_jaccard"].iloc[0]) if len(jac) else np.nan,
                "seniority_chi2_V": float(sen["cramers_v"].iloc[0]) if len(sen) else np.nan,
                "state_chi2_V": float(st["cramers_v"].iloc[0]) if len(st) else np.nan,
                "title_jaccard": float(tit["jaccard"].iloc[0]) if len(tit) else np.nan,
            }
        )
summary_df = pd.DataFrame(summary)
savecsv(summary_df, "effect_size_summary.csv")
print(summary_df.to_string())

print()
print("Done.")
