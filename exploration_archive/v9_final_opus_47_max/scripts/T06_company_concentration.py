"""T06. Company concentration deep investigation.

Produces:
- concentration metrics per source (HHI, Gini, top-k share, with/without aggregators)
- top-20 employer profile per source
- duplicate-template audit (post-Stage-4 dedup verification)
- entry-level posting concentration under the T30 panel (J1/J2/J3/J4)
- within-company vs between-company decomposition for entry share / AI mentions /
  description length / tech count (arshkon vs scraped, also asaniczka vs scraped)
- entry-specialist employer identification (>60% junior share under ANY variant)
- aggregator profile
- new entrants (2026 companies with no 2024 match)
- per-finding concentration prediction table

Also saves two shared artifacts consumed by Wave 3 / Wave 3.5:
- entry_specialist_employers.csv
- returning_companies_cohort.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data" / "unified.parquet"
TBL = ROOT / "exploration" / "tables" / "T06"
FIG = ROOT / "exploration" / "figures" / "T06"
SHARED = ROOT / "exploration" / "artifacts" / "shared"
TBL.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SHARED.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
con.execute("SET memory_limit='10GB'")
con.execute("SET threads=8")

SOURCES = ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]
BASE_WHERE = "source_platform='linkedin' AND is_english AND date_flag='ok' AND is_swe"


def savecsv(df: pd.DataFrame, name: str, dir: Path = TBL) -> None:
    df.to_csv(dir / name, index=False)
    print(f"  wrote {dir / name} ({len(df)} rows)")


# ---------------------------------------------------------------------------
# T30 JUNIOR VARIANTS — compute locally per dispatch
# J1  seniority_final='entry'
# J2  seniority_final IN ('entry','associate')
# J3  yoe_min_years_llm <= 2 (LLM frame only; labeled rows; primary)
# J4  yoe_min_years_llm <= 3 (LLM frame only; labeled rows)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# STEP 1. Concentration metrics per source.
# ---------------------------------------------------------------------------
print("=" * 78)
print("STEP 1. Concentration metrics per source (HHI, Gini, top-k share)")
print("=" * 78)


def hhi(shares: np.ndarray) -> float:
    """Herfindahl-Hirschman Index on shares (sum to 1). Returns sum of squared shares."""
    return float(np.sum(shares * shares))


def gini(vals: np.ndarray) -> float:
    """Standard Gini coefficient on posting-count distribution."""
    v = np.sort(np.asarray(vals, dtype=float))
    n = len(v)
    if n == 0:
        return float("nan")
    if v.sum() == 0:
        return 0.0
    cum = np.cumsum(v)
    # (2*sum_{i=1..n} i * v_i) / (n * sum(v)) - (n+1)/n
    return float((2 * np.sum((np.arange(1, n + 1)) * v)) / (n * v.sum()) - (n + 1) / n)


def concentration_metrics(source: str, include_aggregators: bool) -> dict:
    where = f"{BASE_WHERE} AND source='{source}'"
    if not include_aggregators:
        where += " AND NOT is_aggregator"
    df = con.execute(
        f"""SELECT company_name_canonical, COUNT(*) as n
           FROM '{DATA}' WHERE {where}
           GROUP BY company_name_canonical"""
    ).fetchdf()
    if df.empty:
        return {}
    vals = df["n"].values
    total = vals.sum()
    shares = vals / total
    shares_sorted = np.sort(shares)[::-1]
    cum = np.cumsum(shares_sorted)

    def top_k_share(k):
        return float(cum[min(k, len(cum)) - 1]) if len(cum) else float("nan")

    return {
        "source": source,
        "aggregator_mode": "all" if include_aggregators else "non_aggregator",
        "n_companies": int(len(df)),
        "n_postings": int(total),
        "hhi": hhi(shares),
        "hhi_scaled": hhi(shares) * 10000.0,
        "gini": gini(vals),
        "top1_share": top_k_share(1),
        "top5_share": top_k_share(5),
        "top10_share": top_k_share(10),
        "top20_share": top_k_share(20),
        "top50_share": top_k_share(50),
        "max_company_postings": int(vals.max()),
    }


rows = []
for src in SOURCES:
    for agg in [True, False]:
        rows.append(concentration_metrics(src, agg))
conc = pd.DataFrame(rows)
savecsv(conc, "concentration_metrics.csv")
print(conc.to_string())


# ---------------------------------------------------------------------------
# STEP 2. Top-20 employer profile per source.
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("STEP 2. Top-20 employer profile per source")
print("=" * 78)


def top20_profile(source: str) -> pd.DataFrame:
    where = f"{BASE_WHERE} AND source='{source}' AND company_name_canonical IS NOT NULL"
    # Get top 20 companies
    top = con.execute(
        f"""SELECT company_name_canonical, COUNT(*) as n
           FROM '{DATA}' WHERE {where}
           GROUP BY company_name_canonical
           ORDER BY n DESC LIMIT 20"""
    ).fetchdf()
    total = con.execute(f"SELECT COUNT(*) FROM '{DATA}' WHERE {where}").fetchone()[0]
    top["source"] = source
    top["share_of_source"] = top["n"] / total

    # Enrich with is_aggregator, industry, mean YOE, mean desc length, entry shares
    names = [n for n in top["company_name_canonical"].tolist() if n is not None]
    name_list = ",".join([f"'{n.replace(chr(39), chr(39)*2)}'" for n in names])

    enr = con.execute(
        f"""
        SELECT company_name_canonical,
               ANY_VALUE(is_aggregator) AS is_aggregator_any,
               -- industry: just take the mode
               MODE(company_industry) AS company_industry_mode,
               AVG(CAST(description_length AS DOUBLE)) AS mean_desc_length,
               AVG(CASE WHEN llm_classification_coverage='labeled'
                        THEN CAST(yoe_min_years_llm AS DOUBLE) END) AS mean_yoe_llm,
               COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                                AND yoe_min_years_llm IS NOT NULL) AS n_yoe_llm,
               AVG(CAST(yoe_extracted AS DOUBLE)) AS mean_yoe_rule,
               COUNT(*) FILTER (WHERE yoe_extracted IS NOT NULL) AS n_yoe_rule,
               COUNT(*) FILTER (WHERE seniority_final='entry') AS n_j1,
               COUNT(*) FILTER (WHERE seniority_final IN ('entry','associate')) AS n_j2,
               COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                                AND yoe_min_years_llm IS NOT NULL
                                AND yoe_min_years_llm <= 2) AS n_j3,
               COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                                AND yoe_min_years_llm IS NOT NULL) AS n_j3_denom,
               COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                                AND yoe_min_years_llm IS NOT NULL
                                AND yoe_min_years_llm <= 3) AS n_j4
        FROM '{DATA}' WHERE {where} AND company_name_canonical IN ({name_list})
        GROUP BY company_name_canonical
        """
    ).fetchdf()
    out = top.merge(enr, on="company_name_canonical", how="left")
    out["entry_share_j1"] = out["n_j1"] / out["n"]
    out["entry_share_j2"] = out["n_j2"] / out["n"]
    out["entry_share_j3"] = np.where(out["n_j3_denom"] > 0, out["n_j3"] / out["n_j3_denom"], np.nan)
    out["entry_share_j4"] = np.where(out["n_j3_denom"] > 0, out["n_j4"] / out["n_j3_denom"], np.nan)
    return out


t20_dfs = []
for src in SOURCES:
    d = top20_profile(src)
    t20_dfs.append(d)
top20 = pd.concat(t20_dfs, ignore_index=True)
savecsv(top20, "top20_employer_profile.csv")
print(top20[
    [
        "source",
        "company_name_canonical",
        "n",
        "share_of_source",
        "is_aggregator_any",
        "company_industry_mode",
        "mean_yoe_llm",
        "mean_desc_length",
        "entry_share_j1",
        "entry_share_j3",
    ]
].to_string())


# ---------------------------------------------------------------------------
# STEP 3. Duplicate-template audit
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("STEP 3. Duplicate-template audit")
print("=" * 78)


def dup_audit(source: str) -> pd.DataFrame:
    where = f"{BASE_WHERE} AND source='{source}'"
    df = con.execute(
        f"""SELECT company_name_canonical,
                   COUNT(*) as n_posts,
                   COUNT(DISTINCT description_hash) as n_distinct,
                   COUNT(*) FILTER (WHERE is_multi_location) as n_multi_loc
           FROM '{DATA}' WHERE {where}
           GROUP BY company_name_canonical"""
    ).fetchdf()
    if df.empty:
        return df
    df["max_dup_ratio"] = df["n_posts"] / df["n_distinct"]
    df["source"] = source
    return df


dup_dfs = []
top_dups = []
for src in SOURCES:
    d = dup_audit(src)
    dup_dfs.append(d)
    # Companies with >=5 postings and max_dup_ratio >= 2 (i.e. at least half are duplicates)
    flagged = d[(d["n_posts"] >= 5) & (d["max_dup_ratio"] >= 2.0)].copy()
    flagged = flagged.sort_values("max_dup_ratio", ascending=False).head(10)
    top_dups.append(flagged)
    print(f"{src}: total companies={len(d)},"
          f" with dup>=2 and n>=5 = {(d[(d['n_posts']>=5) & (d['max_dup_ratio']>=2)]).shape[0]},"
          f" mean ratio among >=5 posts = {d[d['n_posts']>=5]['max_dup_ratio'].mean():.2f}")

savecsv(pd.concat(dup_dfs, ignore_index=True)[
    ["source", "company_name_canonical", "n_posts", "n_distinct", "max_dup_ratio", "n_multi_loc"]
], "dup_audit_all_companies.csv")
savecsv(pd.concat(top_dups, ignore_index=True), "dup_audit_top10_per_source.csv")


# ---------------------------------------------------------------------------
# STEP 4. Entry-level posting concentration under T30 panel
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("STEP 4. Entry-level posting concentration (T30 panel J1/J2/J3/J4)")
print("=" * 78)


def entry_concentration(source: str) -> dict:
    where = f"{BASE_WHERE} AND source='{source}'"
    # per-company posting-level aggregation
    q = f"""
    SELECT company_name_canonical,
           COUNT(*) AS n_swe,
           COUNT(*) FILTER (WHERE seniority_final='entry') AS n_j1,
           COUNT(*) FILTER (WHERE seniority_final IN ('entry','associate')) AS n_j2,
           COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                            AND yoe_min_years_llm IS NOT NULL) AS n_llm_labeled,
           COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                            AND yoe_min_years_llm IS NOT NULL
                            AND yoe_min_years_llm <= 2) AS n_j3,
           COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                            AND yoe_min_years_llm IS NOT NULL
                            AND yoe_min_years_llm <= 3) AS n_j4
    FROM '{DATA}'
    WHERE {where}
    GROUP BY company_name_canonical
    """
    df = con.execute(q).fetchdf()
    out = {"source": source, "total_companies": len(df)}
    # How many companies post ANY entry-labeled under each variant
    for var, col in [("J1", "n_j1"), ("J2", "n_j2")]:
        out[f"n_companies_any_{var}"] = int((df[col] > 0).sum())
    # J3/J4: denominators restricted to LLM frame
    df_llm = df[df["n_llm_labeled"] > 0]
    out["n_companies_with_llm_row"] = int(len(df_llm))
    for var, col in [("J3", "n_j3"), ("J4", "n_j4")]:
        out[f"n_companies_any_{var}"] = int((df_llm[col] > 0).sum())

    # Companies with >=5 SWE postings: what share have ZERO entry-labeled rows?
    big = df[df["n_swe"] >= 5]
    out["n_companies_ge5"] = int(len(big))
    for var, col in [("J1", "n_j1"), ("J2", "n_j2")]:
        if len(big):
            out[f"pct_zero_{var}_among_ge5"] = float((big[col] == 0).mean())
    big_llm = big[big["n_llm_labeled"] > 0]
    out["n_companies_ge5_with_llm"] = int(len(big_llm))
    for var, col in [("J3", "n_j3"), ("J4", "n_j4")]:
        if len(big_llm):
            out[f"pct_zero_{var}_among_ge5_llm"] = float((big_llm[col] == 0).mean())
    return out


ec_rows = []
for src in SOURCES:
    ec_rows.append(entry_concentration(src))
ec_df = pd.DataFrame(ec_rows)
savecsv(ec_df, "entry_concentration_summary.csv")
print(ec_df.to_string())

# Distribution of entry-share-of-own-postings among companies that DO post entry roles
# (J3 primary; save histograms)
for variant, col, denom_col in [
    ("J1", "n_j1", "n_swe"),
    ("J2", "n_j2", "n_swe"),
    ("J3", "n_j3", "n_llm_labeled"),
    ("J4", "n_j4", "n_llm_labeled"),
]:
    all_rows = []
    for src in SOURCES:
        where = f"{BASE_WHERE} AND source='{src}'"
        q = f"""
        SELECT company_name_canonical,
               COUNT(*) AS n_swe,
               COUNT(*) FILTER (WHERE seniority_final='entry') AS n_j1,
               COUNT(*) FILTER (WHERE seniority_final IN ('entry','associate')) AS n_j2,
               COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                                AND yoe_min_years_llm IS NOT NULL) AS n_llm_labeled,
               COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                                AND yoe_min_years_llm IS NOT NULL
                                AND yoe_min_years_llm <= 2) AS n_j3,
               COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                                AND yoe_min_years_llm IS NOT NULL
                                AND yoe_min_years_llm <= 3) AS n_j4
        FROM '{DATA}'
        WHERE {where}
        GROUP BY company_name_canonical
        HAVING n_swe >= 5
        """
        d = con.execute(q).fetchdf()
        if variant in ("J3", "J4"):
            d = d[d["n_llm_labeled"] > 0]
            shares = d[col] / d[denom_col]
        else:
            shares = d[col] / d["n_swe"]
        has_any = shares[shares > 0]
        all_rows.append({"source": src, "variant": variant,
                         "n_companies_any_entry": int((shares > 0).sum()),
                         "mean_entry_share_among_any": float(has_any.mean()) if len(has_any) else np.nan,
                         "median_entry_share_among_any": float(has_any.median()) if len(has_any) else np.nan,
                         "q25": float(has_any.quantile(0.25)) if len(has_any) else np.nan,
                         "q75": float(has_any.quantile(0.75)) if len(has_any) else np.nan,
                         "max": float(has_any.max()) if len(has_any) else np.nan})
    pd.DataFrame(all_rows).to_csv(TBL / f"entry_share_distribution_{variant}.csv", index=False)
    print(f"  wrote entry_share_distribution_{variant}.csv")


# ---------------------------------------------------------------------------
# STEP 5. Within-company vs between-company decomposition
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("STEP 5. Within- vs between-company decomposition (arshkon vs scraped)")
print("=" * 78)

# Identify companies with >=5 SWE postings in BOTH arshkon and scraped
q = f"""
WITH a AS (
  SELECT company_name_canonical, COUNT(*) AS n_ar,
         SUM(CAST(description_length AS DOUBLE)) AS sum_len_ar,
         AVG(CAST(yoe_min_years_llm AS DOUBLE)) FILTER (WHERE llm_classification_coverage='labeled') AS mean_yoe_ar,
         COUNT(*) FILTER (WHERE seniority_final='entry') AS n_j1_ar,
         COUNT(*) FILTER (WHERE seniority_final IN ('entry','associate')) AS n_j2_ar,
         COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                          AND yoe_min_years_llm IS NOT NULL) AS n_llm_ar,
         COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                          AND yoe_min_years_llm IS NOT NULL
                          AND yoe_min_years_llm <= 2) AS n_j3_ar,
         COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                          AND yoe_min_years_llm IS NOT NULL
                          AND yoe_min_years_llm <= 3) AS n_j4_ar
  FROM '{DATA}'
  WHERE {BASE_WHERE} AND source='kaggle_arshkon'
  GROUP BY company_name_canonical
  HAVING COUNT(*) >= 5
),
s AS (
  SELECT company_name_canonical, COUNT(*) AS n_sc,
         SUM(CAST(description_length AS DOUBLE)) AS sum_len_sc,
         AVG(CAST(yoe_min_years_llm AS DOUBLE)) FILTER (WHERE llm_classification_coverage='labeled') AS mean_yoe_sc,
         COUNT(*) FILTER (WHERE seniority_final='entry') AS n_j1_sc,
         COUNT(*) FILTER (WHERE seniority_final IN ('entry','associate')) AS n_j2_sc,
         COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                          AND yoe_min_years_llm IS NOT NULL) AS n_llm_sc,
         COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                          AND yoe_min_years_llm IS NOT NULL
                          AND yoe_min_years_llm <= 2) AS n_j3_sc,
         COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                          AND yoe_min_years_llm IS NOT NULL
                          AND yoe_min_years_llm <= 3) AS n_j4_sc
  FROM '{DATA}'
  WHERE {BASE_WHERE} AND source='scraped'
  GROUP BY company_name_canonical
  HAVING COUNT(*) >= 5
)
SELECT a.company_name_canonical, a.n_ar, s.n_sc,
       a.sum_len_ar, s.sum_len_sc,
       a.mean_yoe_ar, s.mean_yoe_sc,
       a.n_j1_ar, s.n_j1_sc, a.n_j2_ar, s.n_j2_sc,
       a.n_llm_ar, s.n_llm_sc, a.n_j3_ar, s.n_j3_sc, a.n_j4_ar, s.n_j4_sc
FROM a JOIN s ON a.company_name_canonical = s.company_name_canonical
"""
panel = con.execute(q).fetchdf()
print(f"  overlap panel size (arshkon ∩ scraped, >=5 each): {len(panel)}")
savecsv(panel, "overlap_panel_arshkon_scraped.csv")


# AI-mention prevalence in overlap panel: use raw description (boilerplate-insensitive binary pattern)
AI_REGEX = r"\b(ai|a\.i\.|artificial intelligence|gen(erative)?[- ]?ai|copilot|chatgpt|gpt-?[0-9]|anthropic|claude|llm|large language model|machine learning|ml engineer|mlops|genai)\b"

# Compute AI mention share and tech-count proxy per company
# Tech count proxy: count of unique tokens among a curated list of tech keywords in description
TECH_REGEX = (
    r"\b(python|java|javascript|typescript|react|vue|angular|node|aws|gcp|azure|docker|"
    r"kubernetes|k8s|golang|rust|scala|kotlin|swift|c\+\+|sql|postgres|mysql|nosql|mongo|"
    r"redis|kafka|spark|hadoop|terraform|ansible|git|jenkins|ci/cd|graphql|rest|jwt|oauth)\b"
)


def ai_tech_for_panel(ar_co_list, sc_co_list):
    # Compute per-row AI mention and tech count, aggregate per company per source
    def run(source, co_list):
        co_list = [n for n in co_list if n is not None]
        if not co_list:
            return pd.DataFrame(columns=[
                "company_name_canonical", "n", "ai_mentions", "tech_tokens_sum",
                "desc_len_sum"
            ])
        names = ",".join([f"'{n.replace(chr(39), chr(39)*2)}'" for n in co_list])
        q2 = f"""
        SELECT company_name_canonical,
               COUNT(*) AS n,
               SUM(CASE WHEN regexp_matches(lower(description), '{AI_REGEX}') THEN 1 ELSE 0 END)::BIGINT AS ai_mentions,
               SUM(len(regexp_extract_all(lower(description), '{TECH_REGEX}')))::BIGINT AS tech_tokens_sum,
               SUM(CAST(description_length AS DOUBLE)) AS desc_len_sum
        FROM '{DATA}'
        WHERE {BASE_WHERE} AND source='{source}'
          AND company_name_canonical IN ({names})
        GROUP BY company_name_canonical
        """
        return con.execute(q2).fetchdf()

    return run("kaggle_arshkon", ar_co_list), run("scraped", sc_co_list)


co_list = panel["company_name_canonical"].tolist()
ai_ar, ai_sc = ai_tech_for_panel(co_list, co_list)
# Only add the ai/tech columns to avoid n_ar/n_sc clashes with panel
ai_ar = ai_ar.rename(columns={"n": "n_ar_ai", "ai_mentions": "ai_ar", "tech_tokens_sum": "tech_ar", "desc_len_sum": "len_ar"})
ai_sc = ai_sc.rename(columns={"n": "n_sc_ai", "ai_mentions": "ai_sc", "tech_tokens_sum": "tech_sc", "desc_len_sum": "len_sc"})

pm = panel.merge(ai_ar, on="company_name_canonical", how="left").merge(
    ai_sc, on="company_name_canonical", how="left"
)
# Compute per-posting metrics using panel n_ar / n_sc (source of truth)
pm["desc_len_ar"] = pm["sum_len_ar"] / pm["n_ar"]
pm["desc_len_sc"] = pm["sum_len_sc"] / pm["n_sc"]
pm["ai_share_ar"] = pm["ai_ar"] / pm["n_ar"]
pm["ai_share_sc"] = pm["ai_sc"] / pm["n_sc"]
pm["tech_per_post_ar"] = pm["tech_ar"] / pm["n_ar"]
pm["tech_per_post_sc"] = pm["tech_sc"] / pm["n_sc"]


def decompose(panel_df, metric_ar_col, metric_sc_col, weight_ar_col="n_ar", weight_sc_col="n_sc"):
    """Shift-share decomposition for aggregate change.

    Aggregate means:
    - A = sum(metric_ar * n_ar) / sum(n_ar)
    - S = sum(metric_sc * n_sc) / sum(n_sc)
    - Total change: S - A
    Within component: mean_sc weights held at ar:
      W = sum((metric_sc - metric_ar) * n_ar) / sum(n_ar)
    Between component: metric_ar held constant, weights shift to sc:
      B = sum(metric_ar * n_sc) / sum(n_sc) - sum(metric_ar * n_ar) / sum(n_ar)
    Interaction: T - W - B
    """
    d = panel_df.dropna(subset=[metric_ar_col, metric_sc_col, weight_ar_col, weight_sc_col]).copy()
    if d.empty:
        return None
    A = (d[metric_ar_col] * d[weight_ar_col]).sum() / d[weight_ar_col].sum()
    S = (d[metric_sc_col] * d[weight_sc_col]).sum() / d[weight_sc_col].sum()
    total = S - A
    within = ((d[metric_sc_col] - d[metric_ar_col]) * d[weight_ar_col]).sum() / d[weight_ar_col].sum()
    between = (d[metric_ar_col] * d[weight_sc_col]).sum() / d[weight_sc_col].sum() - A
    interaction = total - within - between
    return {
        "mean_ar": float(A),
        "mean_sc": float(S),
        "total_change": float(total),
        "within_component": float(within),
        "between_component": float(between),
        "interaction": float(interaction),
        "n_panel_rows": int(len(d)),
    }


# Compute entry shares per company under each J variant
for v in ["j1", "j2"]:
    pm[f"{v}_ar_share"] = pm[f"n_{v}_ar"] / pm["n_ar"]
    pm[f"{v}_sc_share"] = pm[f"n_{v}_sc"] / pm["n_sc"]

for v in ["j3", "j4"]:
    pm[f"{v}_ar_share"] = np.where(pm["n_llm_ar"] > 0, pm[f"n_{v}_ar"] / pm["n_llm_ar"], np.nan)
    pm[f"{v}_sc_share"] = np.where(pm["n_llm_sc"] > 0, pm[f"n_{v}_sc"] / pm["n_llm_sc"], np.nan)

decomp_rows = []
for metric, ar_col, sc_col, w_ar, w_sc, note in [
    ("entry_share_J1", "j1_ar_share", "j1_sc_share", "n_ar", "n_sc", "seniority_final='entry'"),
    ("entry_share_J2", "j2_ar_share", "j2_sc_share", "n_ar", "n_sc", "seniority_final in entry,associate"),
    ("entry_share_J3", "j3_ar_share", "j3_sc_share", "n_llm_ar", "n_llm_sc", "yoe_llm<=2 (LLM frame)"),
    ("entry_share_J4", "j4_ar_share", "j4_sc_share", "n_llm_ar", "n_llm_sc", "yoe_llm<=3 (LLM frame)"),
    ("ai_mention_share", "ai_share_ar", "ai_share_sc", "n_ar", "n_sc", "raw AI keyword regex"),
    ("desc_length_mean", "desc_len_ar", "desc_len_sc", "n_ar", "n_sc", "description_length chars"),
    ("tech_per_post_mean", "tech_per_post_ar", "tech_per_post_sc", "n_ar", "n_sc", "curated tech regex token count"),
]:
    res = decompose(pm, ar_col, sc_col, w_ar, w_sc)
    if res:
        decomp_rows.append({"metric": metric, "definition": note, **res})

savecsv(pd.DataFrame(decomp_rows), "within_between_decomposition.csv")
print(pd.DataFrame(decomp_rows).to_string())


# ---------------------------------------------------------------------------
# STEP 6. Entry-specialist employer identification (any variant > 60%)
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("STEP 6. Entry-specialist employer identification")
print("=" * 78)


def all_source_entry(source_list):
    """For every company with >=5 SWE postings across these sources, compute J1..J4 shares."""
    src_clause = ",".join([f"'{s}'" for s in source_list])
    q = f"""
    SELECT company_name_canonical,
           ANY_VALUE(is_aggregator) AS is_aggregator_any,
           COUNT(*) AS n_swe,
           COUNT(*) FILTER (WHERE seniority_final='entry') AS n_j1,
           COUNT(*) FILTER (WHERE seniority_final IN ('entry','associate')) AS n_j2,
           COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                            AND yoe_min_years_llm IS NOT NULL) AS n_llm,
           COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                            AND yoe_min_years_llm IS NOT NULL
                            AND yoe_min_years_llm <= 2) AS n_j3,
           COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                            AND yoe_min_years_llm IS NOT NULL
                            AND yoe_min_years_llm <= 3) AS n_j4
    FROM '{DATA}'
    WHERE {BASE_WHERE} AND source IN ({src_clause})
    GROUP BY company_name_canonical
    HAVING COUNT(*) >= 5
    """
    return con.execute(q).fetchdf()


full = all_source_entry(SOURCES)
full["j1_share"] = full["n_j1"] / full["n_swe"]
full["j2_share"] = full["n_j2"] / full["n_swe"]
full["j3_share"] = np.where(full["n_llm"] > 0, full["n_j3"] / full["n_llm"], np.nan)
full["j4_share"] = np.where(full["n_llm"] > 0, full["n_j4"] / full["n_llm"], np.nan)

full["flag_any_over_60"] = (
    (full["j1_share"] > 0.60)
    | (full["j2_share"] > 0.60)
    | (full["j3_share"].fillna(0) > 0.60)
    | (full["j4_share"].fillna(0) > 0.60)
)
flagged = full[full["flag_any_over_60"]].copy()
print(f"  flagged companies (any variant >60%): {len(flagged)}")
print(f"  flagged non-aggregator: {(~flagged['is_aggregator_any']).sum()}")
print(f"  flagged aggregator: {flagged['is_aggregator_any'].sum()}")


# Manually categorize the top 20 flagged by n_swe
top_flag = flagged.nlargest(40, "n_swe")
# Heuristic rules for categorization — do a best-effort first pass
STAFFING_KEYWORDS = [
    "staffing", "recruit", "consult", "outsourc", "technology partner", "solutions",
    "tek", "talent", "dice", "lensa", "cybercod", "robert half", "aerotek",
    "insight global", "kforce", "infinity", "kelly", "adecco", "randstad",
    "modis", "capgemini", "tcs", "tata", "cognizant", "accenture", "infosys",
    "wipro", "hcl", "deloitte", "technology corp", "softw", "motion", "byteforge", "revature", "smartit", "amdocs",
    "spiceorb", "hirewith", "turing",
]
COLLEGE_KEYWORDS = [
    "univer", "college", "intern ", "handshake", "career", "campus",
    "postgrad", "student",
]
TECH_GIANT_KEYWORDS = [
    "amazon", "google", "microsoft", "meta", "facebook", "apple", "netflix",
    "tesla", "oracle", "ibm", "intel", "nvidia", "salesforce", "adobe", "qualcomm",
    "cisco", "vmware",
]


def categorize(name: str, is_agg: bool) -> str:
    if not isinstance(name, str):
        return "unknown"
    n = name.lower()
    if is_agg:
        return "aggregator"
    if any(k in n for k in TECH_GIANT_KEYWORDS):
        return "tech-giant-intern"
    if any(k in n for k in STAFFING_KEYWORDS):
        return "staffing"
    if any(k in n for k in COLLEGE_KEYWORDS):
        return "college-jobsite"
    return "direct-employer"


flagged["specialist_category"] = flagged.apply(
    lambda r: categorize(r["company_name_canonical"], bool(r["is_aggregator_any"])), axis=1
)

# Save specialist file with required schema
out_spec = pd.DataFrame(
    {
        "company_name_canonical": flagged["company_name_canonical"],
        "n_swe_postings": flagged["n_swe"],
        "junior_share_j3": flagged["j3_share"].fillna(-1).round(4),
        "junior_share_j1_j2": np.maximum(flagged["j1_share"], flagged["j2_share"]).round(4),
        "is_aggregator": flagged["is_aggregator_any"].astype(bool),
        "specialist_category": flagged["specialist_category"],
    }
).sort_values("n_swe_postings", ascending=False)
out_spec.to_csv(SHARED / "entry_specialist_employers.csv", index=False)
print(f"  wrote {SHARED / 'entry_specialist_employers.csv'} ({len(out_spec)} rows)")

# Also a detailed companion table in tables/T06/
flagged_out = flagged[
    [
        "company_name_canonical", "n_swe", "n_j1", "n_j2", "n_llm", "n_j3", "n_j4",
        "j1_share", "j2_share", "j3_share", "j4_share",
        "is_aggregator_any", "specialist_category",
    ]
].sort_values("n_swe", ascending=False)
savecsv(flagged_out, "entry_specialist_details.csv")


# ---------------------------------------------------------------------------
# STEP 7. Aggregator profile per source
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("STEP 7. Aggregator profile")
print("=" * 78)

q = f"""
SELECT source, is_aggregator,
       COUNT(*) AS n_postings,
       COUNT(DISTINCT company_name_canonical) AS n_companies,
       AVG(CAST(description_length AS DOUBLE)) AS mean_desc_length,
       COUNT(*) FILTER (WHERE seniority_final='entry') / CAST(COUNT(*) AS DOUBLE) AS entry_share_j1,
       COUNT(*) FILTER (WHERE seniority_final IN ('entry','associate')) / CAST(COUNT(*) AS DOUBLE) AS entry_share_j2,
       AVG(CASE WHEN llm_classification_coverage='labeled' THEN CAST(yoe_min_years_llm AS DOUBLE) END) AS mean_yoe_llm,
       COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                        AND yoe_min_years_llm IS NOT NULL
                        AND yoe_min_years_llm <= 2) /
       NULLIF(CAST(COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                                    AND yoe_min_years_llm IS NOT NULL) AS DOUBLE), 0) AS entry_share_j3
FROM '{DATA}'
WHERE {BASE_WHERE}
GROUP BY source, is_aggregator
ORDER BY source, is_aggregator
"""
agg_df = con.execute(q).fetchdf()
savecsv(agg_df, "aggregator_profile.csv")
print(agg_df.to_string())


# ---------------------------------------------------------------------------
# STEP 8. New entrants (2026 companies with no 2024 match) + returning cohort
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("STEP 8. New entrants vs returning companies")
print("=" * 78)

# Use arshkon ∪ asaniczka as 2024; scraped as 2026 LinkedIn.
# Exclude NULL canonical names (otherwise NOT IN is contaminated by SQL NULL semantics).
q = f"""
WITH c24 AS (
  SELECT company_name_canonical, ANY_VALUE(is_aggregator) AS is_aggregator_any,
         COUNT(*) AS n_swe
  FROM '{DATA}'
  WHERE {BASE_WHERE} AND source IN ('kaggle_arshkon', 'kaggle_asaniczka')
    AND company_name_canonical IS NOT NULL
  GROUP BY company_name_canonical
),
c26 AS (
  SELECT company_name_canonical, ANY_VALUE(is_aggregator) AS is_aggregator_any,
         COUNT(*) AS n_swe
  FROM '{DATA}'
  WHERE {BASE_WHERE} AND source='scraped'
    AND company_name_canonical IS NOT NULL
  GROUP BY company_name_canonical
)
SELECT
  (SELECT COUNT(*) FROM c24) AS n_companies_2024,
  (SELECT COUNT(*) FROM c26) AS n_companies_2026,
  (SELECT COUNT(*) FROM c26 WHERE company_name_canonical NOT IN (SELECT company_name_canonical FROM c24)) AS n_new_entrants_2026,
  (SELECT COUNT(*) FROM c26 JOIN c24 USING (company_name_canonical)) AS n_returning_2026
"""
summary_ne = con.execute(q).fetchdf()
print(summary_ne.to_string())
summary_ne.to_csv(TBL / "new_vs_returning_counts.csv", index=False)

# Content profile of new vs returning
q = f"""
WITH c24 AS (
  SELECT DISTINCT company_name_canonical
  FROM '{DATA}'
  WHERE {BASE_WHERE} AND source IN ('kaggle_arshkon', 'kaggle_asaniczka')
    AND company_name_canonical IS NOT NULL
),
flagged AS (
  SELECT u.*,
         CASE WHEN c24.company_name_canonical IS NOT NULL THEN 'returning' ELSE 'new_entrant' END AS cohort
  FROM '{DATA}' u
  LEFT JOIN c24 USING (company_name_canonical)
  WHERE {BASE_WHERE.replace("is_swe", "u.is_swe")
                   .replace("source_platform", "u.source_platform")
                   .replace("is_english", "u.is_english")
                   .replace("date_flag", "u.date_flag")}
    AND u.source='scraped'
)
SELECT cohort,
       COUNT(*) AS n,
       COUNT(DISTINCT company_name_canonical) AS n_companies,
       AVG(CAST(description_length AS DOUBLE)) AS mean_desc_length,
       COUNT(*) FILTER (WHERE seniority_final='entry') / CAST(COUNT(*) AS DOUBLE) AS entry_share_j1,
       COUNT(*) FILTER (WHERE seniority_final IN ('entry','associate')) / CAST(COUNT(*) AS DOUBLE) AS entry_share_j2,
       AVG(CASE WHEN llm_classification_coverage='labeled' THEN CAST(yoe_min_years_llm AS DOUBLE) END) AS mean_yoe_llm,
       COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                        AND yoe_min_years_llm IS NOT NULL
                        AND yoe_min_years_llm <= 2) /
       NULLIF(CAST(COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                                    AND yoe_min_years_llm IS NOT NULL) AS DOUBLE), 0) AS entry_share_j3
FROM flagged
GROUP BY cohort
ORDER BY cohort
"""
ne = con.execute(q).fetchdf()
savecsv(ne, "new_vs_returning_profile.csv")
print(ne.to_string())

# Save returning-companies cohort artifact (arshkon ∪ asaniczka) ∩ scraped
q = f"""
WITH c24 AS (
  SELECT company_name_canonical,
         ANY_VALUE(is_aggregator) AS is_aggregator_any,
         COUNT(*) AS n_swe_2024
  FROM '{DATA}'
  WHERE {BASE_WHERE} AND source IN ('kaggle_arshkon', 'kaggle_asaniczka')
    AND company_name_canonical IS NOT NULL
  GROUP BY company_name_canonical
),
c26 AS (
  SELECT company_name_canonical,
         COUNT(*) AS n_swe_2026
  FROM '{DATA}'
  WHERE {BASE_WHERE} AND source='scraped'
    AND company_name_canonical IS NOT NULL
  GROUP BY company_name_canonical
)
SELECT c24.company_name_canonical,
       c24.n_swe_2024,
       c26.n_swe_2026,
       c24.is_aggregator_any AS is_aggregator
FROM c24 JOIN c26 USING (company_name_canonical)
ORDER BY c26.n_swe_2026 + c24.n_swe_2024 DESC
"""
returning = con.execute(q).fetchdf()
returning.to_csv(SHARED / "returning_companies_cohort.csv", index=False)
print(f"  wrote {SHARED / 'returning_companies_cohort.csv'} ({len(returning)} rows)")

print()
print("Done.")
