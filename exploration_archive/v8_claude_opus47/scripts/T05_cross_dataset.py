"""T05 — Cross-dataset comparability.

Tests whether the three sources measure the same labor market or different
instruments. Produces the artefacts for exploration/reports/T05.md.

All queries go through DuckDB (the parquet is 6.6 GB and must not enter
pandas in full). Small aggregate frames are fine in pandas.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data" / "unified.parquet"
FIG = ROOT / "exploration" / "figures" / "T05"
TAB = ROOT / "exploration" / "tables" / "T05"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

CORE = """
  source_platform = 'linkedin'
  AND is_english = TRUE
  AND date_flag = 'ok'
  AND is_swe = TRUE
"""

SOURCES = ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]

# Inline regex sanity asserts (TDD for title parsing) ------------------------
JUNIOR_TITLE_RE = re.compile(r"\b(junior|jr|entry[- ]level|graduate|new[- ]grad|intern)\b", re.I)
SENIOR_TITLE_RE = re.compile(r"\b(senior|sr\.?|staff|principal|lead|architect|distinguished)\b", re.I)

assert JUNIOR_TITLE_RE.search("Junior Software Engineer")
assert JUNIOR_TITLE_RE.search("jr. developer")
assert JUNIOR_TITLE_RE.search("entry-level engineer")
assert not JUNIOR_TITLE_RE.search("Senior Software Engineer")
assert SENIOR_TITLE_RE.search("Sr. Engineer")
assert SENIOR_TITLE_RE.search("Staff Software Engineer")
assert SENIOR_TITLE_RE.search("Principal Architect")
assert not SENIOR_TITLE_RE.search("Junior Engineer")


def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    # Keep memory safe — duckdb auto-spills
    c.execute("SET memory_limit='12GB'")
    c.execute("SET threads=6")
    return c


def cohen_h(p1: float, p2: float) -> float:
    """Effect size for two proportions."""
    if any(v is None or np.isnan(v) for v in (p1, p2)):
        return float("nan")
    p1 = min(max(p1, 1e-12), 1 - 1e-12)
    p2 = min(max(p2, 1e-12), 1 - 1e-12)
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def prop_diff(x1, n1, x2, n2):
    """Two-proportion z-test + effect size h."""
    if n1 == 0 or n2 == 0:
        return {"p1": float("nan"), "p2": float("nan"), "z": float("nan"),
                "p_value": float("nan"), "h": float("nan"), "diff": float("nan")}
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    if p in (0, 1):
        z = 0.0
        pv = 1.0
    else:
        se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
        z = (p1 - p2) / se if se > 0 else 0.0
        pv = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"p1": p1, "p2": p2, "z": z, "p_value": pv,
            "h": cohen_h(p1, p2), "diff": p1 - p2}


# ---------------------------------------------------------------------------
# Step 1. Description length KS + histograms
# ---------------------------------------------------------------------------

def step1_length(c):
    print("--- Step 1: description length ---")
    summary = c.execute(f"""
        SELECT source,
               COUNT(*) AS n,
               AVG(description_length) AS mean,
               MEDIAN(description_length) AS median,
               STDDEV_POP(description_length) AS sd,
               QUANTILE_CONT(description_length, 0.1) AS p10,
               QUANTILE_CONT(description_length, 0.9) AS p90
        FROM read_parquet('{DATA}')
        WHERE {CORE}
        GROUP BY source
        ORDER BY source
    """).df()
    summary.to_csv(TAB / "01_length_summary.csv", index=False)
    print(summary.to_string(index=False))

    # Histograms — downsample to keep memory low
    samples = {}
    for src in SOURCES:
        df = c.execute(f"""
            SELECT description_length AS x
            FROM read_parquet('{DATA}')
            WHERE {CORE} AND source = '{src}' AND description_length IS NOT NULL
            USING SAMPLE 20000 ROWS
        """).df()
        samples[src] = df["x"].to_numpy()
        print(f"  sample({src}) n={len(df)}")

    # KS pairwise
    rows = []
    from itertools import combinations
    for a, b in combinations(SOURCES, 2):
        s, p = stats.ks_2samp(samples[a], samples[b])
        # Cliff's delta-lite via Mann-Whitney
        u, pu = stats.mannwhitneyu(samples[a], samples[b], alternative="two-sided")
        rows.append({"a": a, "b": b, "ks_stat": s, "ks_p": p,
                     "mw_u": u, "mw_p": pu,
                     "median_a": float(np.median(samples[a])),
                     "median_b": float(np.median(samples[b]))})
    ks = pd.DataFrame(rows)
    ks.to_csv(TAB / "01_length_ks.csv", index=False)
    print(ks.to_string(index=False))

    # Overlap histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 10000, 60)
    colors = {"kaggle_arshkon": "tab:blue", "kaggle_asaniczka": "tab:orange", "scraped": "tab:green"}
    for src in SOURCES:
        ax.hist(np.clip(samples[src], 0, 10000), bins=bins, alpha=0.45,
                label=f"{src} (med {np.median(samples[src]):.0f})",
                color=colors[src], density=True)
    ax.set_xlabel("description_length (clipped at 10,000)")
    ax.set_ylabel("density")
    ax.set_title("Raw description length across sources (SWE, LinkedIn)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "01_length_hist.png", dpi=150)
    plt.close(fig)

    return {"summary": summary, "ks": ks}


# ---------------------------------------------------------------------------
# Step 2. Company overlap (Jaccard on company_name_canonical)
# ---------------------------------------------------------------------------

def step2_company(c):
    print("--- Step 2: company overlap ---")
    sets = {}
    for src in SOURCES:
        names = c.execute(f"""
            SELECT DISTINCT company_name_canonical
            FROM read_parquet('{DATA}')
            WHERE {CORE} AND source = '{src}'
              AND company_name_canonical IS NOT NULL
        """).df()["company_name_canonical"].tolist()
        sets[src] = set(names)
        print(f"  {src}: {len(sets[src])} distinct companies")

    from itertools import combinations
    rows = []
    for a, b in combinations(SOURCES, 2):
        inter = sets[a] & sets[b]
        union = sets[a] | sets[b]
        rows.append({"a": a, "b": b, "na": len(sets[a]), "nb": len(sets[b]),
                     "intersection": len(inter), "union": len(union),
                     "jaccard": len(inter) / len(union) if union else float("nan")})
    ov = pd.DataFrame(rows)
    ov.to_csv(TAB / "02_company_overlap.csv", index=False)
    print(ov.to_string(index=False))

    # Top 50 by volume per source + cross-source membership
    top50s = {}
    for src in SOURCES:
        t50 = c.execute(f"""
            SELECT company_name_canonical AS c, COUNT(*) AS n
            FROM read_parquet('{DATA}')
            WHERE {CORE} AND source = '{src}'
              AND company_name_canonical IS NOT NULL
            GROUP BY 1
            ORDER BY n DESC
            LIMIT 50
        """).df()
        top50s[src] = t50
        t50.to_csv(TAB / f"02_top50_{src}.csv", index=False)
        print(f"  top50 {src}: head rows")
        print(t50.head(10).to_string(index=False))

    rows = []
    for src, t in top50s.items():
        for other in SOURCES:
            if other == src:
                continue
            hits = t["c"].isin(sets[other]).sum()
            rows.append({"top50_of": src, "overlaps_with": other,
                         "top50_in_other": int(hits), "frac": hits / 50})
    t50x = pd.DataFrame(rows)
    t50x.to_csv(TAB / "02_top50_overlap.csv", index=False)
    print(t50x.to_string(index=False))

    return {"overlap": ov, "top50s": top50s, "top50x": t50x, "sets": sets}


# ---------------------------------------------------------------------------
# Step 3. Geographic state-level distribution + chi-squared
# ---------------------------------------------------------------------------

def step3_geo(c):
    print("--- Step 3: geographic ---")
    ml = c.execute(f"""
        SELECT source,
               COUNT(*) FILTER (WHERE is_multi_location) AS n_multi_loc,
               COUNT(*) FILTER (WHERE state_normalized IS NULL AND NOT is_multi_location) AS n_null_state,
               COUNT(*) AS total
        FROM read_parquet('{DATA}')
        WHERE {CORE}
        GROUP BY source ORDER BY source
    """).df()
    ml.to_csv(TAB / "03_multi_location_and_null.csv", index=False)
    print(ml.to_string(index=False))

    state_counts = c.execute(f"""
        SELECT source, state_normalized, COUNT(*) AS n
        FROM read_parquet('{DATA}')
        WHERE {CORE} AND state_normalized IS NOT NULL
        GROUP BY source, state_normalized
    """).df()
    pivot = state_counts.pivot(index="state_normalized", columns="source", values="n").fillna(0)
    pivot = pivot[SOURCES]  # enforce column order
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False)
    pivot.to_csv(TAB / "03_state_counts.csv")
    print(pivot.head(20).to_string())

    # Share & chi2 pairwise
    rows = []
    from itertools import combinations
    for a, b in combinations(SOURCES, 2):
        sub = pivot[[a, b]].copy()
        # Drop tiny states (expected count < 5) to avoid chi2 warnings
        sub = sub[(sub[a] + sub[b]) >= 20]
        contingency = sub[[a, b]].to_numpy()
        chi2, p, dof, _ = stats.chi2_contingency(contingency)
        # Cramer's V
        n = contingency.sum()
        v = math.sqrt(chi2 / (n * (min(contingency.shape) - 1))) if n > 0 else float("nan")
        rows.append({"a": a, "b": b, "n_states_compared": sub.shape[0],
                     "chi2": chi2, "dof": dof, "p": p, "cramer_v": v})
    chi = pd.DataFrame(rows)
    chi.to_csv(TAB / "03_state_chi2.csv", index=False)
    print(chi.to_string(index=False))

    return {"ml": ml, "pivot": pivot, "chi": chi}


# ---------------------------------------------------------------------------
# Step 4. Seniority distributions + chi-squared (exclude unknown)
# ---------------------------------------------------------------------------

def step4_seniority(c):
    print("--- Step 4: seniority (exclude unknown) ---")
    sen = c.execute(f"""
        SELECT source, seniority_final, COUNT(*) AS n
        FROM read_parquet('{DATA}')
        WHERE {CORE} AND seniority_final != 'unknown'
        GROUP BY source, seniority_final
        ORDER BY source, seniority_final
    """).df()
    pivot = sen.pivot(index="seniority_final", columns="source", values="n").fillna(0)
    pivot = pivot[SOURCES]
    pivot.to_csv(TAB / "04_seniority_counts.csv")
    print(pivot.to_string())

    # Also report shares
    shares = pivot.div(pivot.sum(axis=0), axis=1)
    shares.to_csv(TAB / "04_seniority_shares.csv")
    print("shares:"); print(shares.round(3).to_string())

    rows = []
    from itertools import combinations
    for a, b in combinations(SOURCES, 2):
        ct = pivot[[a, b]].to_numpy()
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        n = ct.sum()
        v = math.sqrt(chi2 / (n * (min(ct.shape) - 1))) if n > 0 else float("nan")
        rows.append({"a": a, "b": b, "chi2": chi2, "dof": dof, "p": p, "cramer_v": v})
    chi = pd.DataFrame(rows)
    chi.to_csv(TAB / "04_seniority_chi2.csv", index=False)
    print(chi.to_string(index=False))

    # Unknown rate
    unk = c.execute(f"""
        SELECT source,
               SUM(CASE WHEN seniority_final='unknown' THEN 1 ELSE 0 END) AS n_unknown,
               COUNT(*) AS total
        FROM read_parquet('{DATA}')
        WHERE {CORE}
        GROUP BY source
        ORDER BY source
    """).df()
    unk["unknown_share"] = unk["n_unknown"] / unk["total"]
    unk.to_csv(TAB / "04_unknown_rate.csv", index=False)
    print(unk.to_string(index=False))

    return {"pivot": pivot, "shares": shares, "chi": chi, "unknown": unk}


# ---------------------------------------------------------------------------
# Step 5. Title vocabulary overlap (Jaccard on title_normalized)
# ---------------------------------------------------------------------------

def step5_titles(c):
    print("--- Step 5: title vocabulary ---")
    sets = {}
    for src in SOURCES:
        t = c.execute(f"""
            SELECT DISTINCT title_normalized
            FROM read_parquet('{DATA}')
            WHERE {CORE} AND source = '{src}' AND title_normalized IS NOT NULL
        """).df()["title_normalized"].tolist()
        sets[src] = set(t)
        print(f"  {src}: {len(sets[src])} distinct titles")

    from itertools import combinations
    rows = []
    for a, b in combinations(SOURCES, 2):
        inter = sets[a] & sets[b]
        union = sets[a] | sets[b]
        rows.append({"a": a, "b": b, "na": len(sets[a]), "nb": len(sets[b]),
                     "intersection": len(inter),
                     "jaccard": len(inter) / len(union) if union else float("nan")})
    ov = pd.DataFrame(rows)
    ov.to_csv(TAB / "05_title_overlap.csv", index=False)
    print(ov.to_string(index=False))

    # Titles unique to each source (by frequency)
    for src in SOURCES:
        others = set().union(*(sets[o] for o in SOURCES if o != src))
        unique_titles = c.execute(f"""
            SELECT title_normalized AS t, COUNT(*) AS n
            FROM read_parquet('{DATA}')
            WHERE {CORE} AND source = '{src}' AND title_normalized IS NOT NULL
            GROUP BY 1
            ORDER BY n DESC
            LIMIT 2000
        """).df()
        unique_titles = unique_titles[~unique_titles["t"].isin(others)]
        unique_titles.head(50).to_csv(TAB / f"05_titles_unique_{src}.csv", index=False)

    return ov


# ---------------------------------------------------------------------------
# Step 6. Industry (arshkon vs scraped)
# ---------------------------------------------------------------------------

def step6_industry(c):
    print("--- Step 6: company_industry (arshkon vs scraped LinkedIn) ---")
    ind = c.execute(f"""
        SELECT source, company_industry, COUNT(*) AS n
        FROM read_parquet('{DATA}')
        WHERE {CORE}
          AND source IN ('kaggle_arshkon', 'scraped')
          AND company_industry IS NOT NULL
        GROUP BY source, company_industry
    """).df()
    pivot = ind.pivot(index="company_industry", columns="source", values="n").fillna(0)
    # Ensure cols
    for src in ("kaggle_arshkon", "scraped"):
        if src not in pivot.columns:
            pivot[src] = 0
    pivot = pivot[["kaggle_arshkon", "scraped"]]
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False)
    pivot.head(40).to_csv(TAB / "06_industry_top40.csv")
    print(pivot.head(20).to_string())

    # Shares on top-20 industries
    top20 = pivot.head(20)
    shares = top20[["kaggle_arshkon", "scraped"]].div(
        top20[["kaggle_arshkon", "scraped"]].sum(axis=0), axis=1)
    shares.to_csv(TAB / "06_industry_shares_top20.csv")
    print("shares:"); print(shares.round(3).to_string())

    # Chi2 on pivot (drop tiny rows)
    sub = pivot[(pivot["kaggle_arshkon"] + pivot["scraped"]) >= 20]
    ct = sub[["kaggle_arshkon", "scraped"]].to_numpy()
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.sum()
    v = math.sqrt(chi2 / (n * (min(ct.shape) - 1))) if n > 0 else float("nan")
    chi_row = {"chi2": chi2, "dof": dof, "p": p, "cramer_v": v,
               "n_industries_compared": int(sub.shape[0])}
    pd.DataFrame([chi_row]).to_csv(TAB / "06_industry_chi2.csv", index=False)
    print(chi_row)

    # Label semantics: how many arshkon industries are single-token vs compound?
    sem = c.execute(f"""
        SELECT source,
               SUM(CASE WHEN company_industry LIKE '%,%' OR company_industry LIKE '%and %' THEN 1 ELSE 0 END) AS compound,
               COUNT(*) AS n
        FROM read_parquet('{DATA}')
        WHERE {CORE}
          AND source IN ('kaggle_arshkon', 'scraped')
          AND company_industry IS NOT NULL
        GROUP BY source
    """).df()
    sem["compound_share"] = sem["compound"] / sem["n"]
    sem.to_csv(TAB / "06_industry_semantics.csv", index=False)
    print(sem.to_string(index=False))

    return {"pivot": pivot, "chi": chi_row, "sem": sem}


# ---------------------------------------------------------------------------
# Step 8. Within-2024 calibration — arshkon vs asaniczka mirror
# ---------------------------------------------------------------------------

def step8_within2024(c, step1, step4):
    print("--- Step 8: within-2024 calibration ---")
    # Mirror a subset of the same metrics
    w = c.execute(f"""
        SELECT source, seniority_final,
               AVG(description_length) AS mean_len,
               MEDIAN(description_length) AS median_len,
               AVG(yoe_extracted) AS mean_yoe,
               MEDIAN(yoe_extracted) AS median_yoe,
               COUNT(*) AS n
        FROM read_parquet('{DATA}')
        WHERE {CORE} AND source IN ('kaggle_arshkon', 'kaggle_asaniczka')
        GROUP BY source, seniority_final
        ORDER BY source, seniority_final
    """).df()
    w.to_csv(TAB / "08_within2024_by_seniority.csv", index=False)
    print(w.to_string(index=False))

    # Non-stratified pooled comparison
    pool = c.execute(f"""
        SELECT source,
               AVG(description_length) AS mean_len,
               MEDIAN(description_length) AS median_len,
               AVG(yoe_extracted) AS mean_yoe,
               COUNT(*) AS n
        FROM read_parquet('{DATA}')
        WHERE {CORE}
        GROUP BY source
        ORDER BY source
    """).df()
    pool.to_csv(TAB / "08_pooled_by_source.csv", index=False)
    print(pool.to_string(index=False))

    # Cross-period vs within-2024 effect sizes on length (median ratio) and
    # entry-share under J1.
    # Entry share under J1 by source
    entry = c.execute(f"""
        SELECT source,
               SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END) AS entry,
               SUM(CASE WHEN seniority_final IN ('entry','associate') THEN 1 ELSE 0 END) AS entry_or_assoc,
               SUM(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted<=2 THEN 1 ELSE 0 END) AS yoe_le2,
               SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS n_yoe,
               COUNT(*) AS n
        FROM read_parquet('{DATA}')
        WHERE {CORE}
        GROUP BY source
        ORDER BY source
    """).df()
    entry.to_csv(TAB / "08_entry_shares.csv", index=False)
    entry["j1_share"] = entry["entry"] / entry["n"]
    entry["j2_share"] = entry["entry_or_assoc"] / entry["n"]
    entry["j3_share"] = entry["yoe_le2"] / entry["n_yoe"]
    print(entry.to_string(index=False))

    # Comparisons: arsh-vs-asa (within-2024) AND arsh-vs-scraped (cross-period)
    def pull(src, col):
        return int(entry.set_index("source").loc[src, col])

    comparisons = [
        ("J1", "entry", "n"),
        ("J2", "entry_or_assoc", "n"),
        ("J3", "yoe_le2", "n_yoe"),
    ]
    rows = []
    for name, num, den in comparisons:
        within = prop_diff(pull("kaggle_arshkon", num), pull("kaggle_arshkon", den),
                            pull("kaggle_asaniczka", num), pull("kaggle_asaniczka", den))
        cross = prop_diff(pull("kaggle_arshkon", num), pull("kaggle_arshkon", den),
                           pull("scraped", num), pull("scraped", den))
        pooled_2024_num = pull("kaggle_arshkon", num) + pull("kaggle_asaniczka", num)
        pooled_2024_den = pull("kaggle_arshkon", den) + pull("kaggle_asaniczka", den)
        cross_pool = prop_diff(pooled_2024_num, pooled_2024_den,
                                pull("scraped", num), pull("scraped", den))
        rows.append({
            "definition": name,
            "within_2024_h": within["h"], "within_2024_diff": within["diff"],
            "within_2024_p": within["p_value"],
            "arsh_vs_scraped_h": cross["h"], "arsh_vs_scraped_diff": cross["diff"],
            "arsh_vs_scraped_p": cross["p_value"],
            "pool2024_vs_scraped_h": cross_pool["h"], "pool2024_vs_scraped_diff": cross_pool["diff"],
            "pool2024_vs_scraped_p": cross_pool["p_value"],
            "snr": abs(cross["h"]) / abs(within["h"]) if abs(within["h"]) > 1e-9 else float("inf"),
        })
    calib = pd.DataFrame(rows)
    calib.to_csv(TAB / "08_calibration_entry_share.csv", index=False)
    print(calib.to_string(index=False))

    return {"entry": entry, "within_by_seniority": w, "pool": pool, "calib": calib}


# ---------------------------------------------------------------------------
# Step 9. Platform labeling stability + Indeed cross-validation
# ---------------------------------------------------------------------------

def step9_platform(c):
    print("--- Step 9: platform labeling stability ---")

    # Top 20 titles present in both arshkon and scraped (SWE LinkedIn)
    overlap_titles = c.execute(f"""
        WITH a AS (
          SELECT title_normalized AS t, COUNT(*) AS n
          FROM read_parquet('{DATA}')
          WHERE {CORE} AND source='kaggle_arshkon' AND title_normalized IS NOT NULL
          GROUP BY 1
        ),
        s AS (
          SELECT title_normalized AS t, COUNT(*) AS n
          FROM read_parquet('{DATA}')
          WHERE {CORE} AND source='scraped' AND title_normalized IS NOT NULL
          GROUP BY 1
        )
        SELECT a.t, a.n AS n_arsh, s.n AS n_scraped,
               LEAST(a.n, s.n) AS min_n
        FROM a JOIN s USING (t)
        WHERE a.n >= 20 AND s.n >= 20
        ORDER BY (a.n + s.n) DESC
        LIMIT 20
    """).df()
    overlap_titles.to_csv(TAB / "09_top20_shared_titles.csv", index=False)
    print(overlap_titles.to_string(index=False))

    titles = overlap_titles["t"].tolist()
    if not titles:
        print("  (no overlap titles — bail on step 9)")
        return {"overlap_titles": overlap_titles}

    title_list = ",".join([f"'" + t.replace("'", "''") + "'" for t in titles])

    # seniority_native distribution per shared title × source
    nat = c.execute(f"""
        SELECT title_normalized AS t, source, seniority_native, COUNT(*) AS n
        FROM read_parquet('{DATA}')
        WHERE {CORE}
          AND source IN ('kaggle_arshkon','scraped')
          AND title_normalized IN ({title_list})
          AND seniority_native IS NOT NULL
        GROUP BY 1,2,3
    """).df()
    nat.to_csv(TAB / "09_shared_title_native_seniority.csv", index=False)

    # For each shared title, fraction of 'entry' in native labels per source
    nat_entry = (nat[nat["seniority_native"] == "entry"]
                   .groupby(["t", "source"], as_index=False)["n"].sum()
                   .rename(columns={"n": "entry_n"}))
    nat_total = nat.groupby(["t", "source"], as_index=False)["n"].sum().rename(columns={"n": "total_n"})
    nat_share = nat_total.merge(nat_entry, on=["t", "source"], how="left").fillna(0)
    nat_share["entry_share_native"] = nat_share["entry_n"] / nat_share["total_n"]

    nat_pivot = nat_share.pivot(index="t", columns="source", values="entry_share_native")
    for src in ("kaggle_arshkon", "scraped"):
        if src not in nat_pivot.columns:
            nat_pivot[src] = 0
    nat_pivot = nat_pivot[["kaggle_arshkon", "scraped"]].fillna(0)
    nat_pivot["delta"] = nat_pivot["scraped"] - nat_pivot["kaggle_arshkon"]
    nat_pivot.to_csv(TAB / "09_native_entry_share_shift.csv")
    print("native entry share shift (shared titles):"); print(nat_pivot.round(3).to_string())

    # YOE distribution per title × source on shared titles
    yoe = c.execute(f"""
        SELECT title_normalized AS t, source,
               AVG(yoe_extracted) AS mean_yoe,
               MEDIAN(yoe_extracted) AS median_yoe,
               COUNT(yoe_extracted) AS n_yoe
        FROM read_parquet('{DATA}')
        WHERE {CORE}
          AND source IN ('kaggle_arshkon','scraped')
          AND title_normalized IN ({title_list})
        GROUP BY 1,2
    """).df()
    yoe.to_csv(TAB / "09_shared_title_yoe.csv", index=False)
    yoe_pivot = yoe.pivot(index="t", columns="source", values="mean_yoe")
    for src in ("kaggle_arshkon", "scraped"):
        if src not in yoe_pivot.columns:
            yoe_pivot[src] = np.nan
    yoe_pivot = yoe_pivot[["kaggle_arshkon", "scraped"]]
    yoe_pivot["delta"] = yoe_pivot["scraped"] - yoe_pivot["kaggle_arshkon"]
    yoe_pivot.to_csv(TAB / "09_mean_yoe_shift.csv")
    print("mean YOE by source (shared titles):"); print(yoe_pivot.round(2).to_string())

    # Indeed cross-validation — Stage 5 strong-rule only (seniority_final_source title_keyword/title_manager)
    # Indeed is LinkedIn-free and excluded from LLM frame so seniority_final comes from Stage 5 strong rules or 'unknown'
    indeed = c.execute(f"""
        SELECT source, source_platform, seniority_final, seniority_final_source, COUNT(*) AS n
        FROM read_parquet('{DATA}')
        WHERE is_swe AND is_english AND date_flag='ok'
          AND source='scraped' AND source_platform='indeed'
        GROUP BY 1,2,3,4
        ORDER BY 3,4
    """).df()
    indeed.to_csv(TAB / "09_indeed_seniority.csv", index=False)
    print(indeed.to_string(index=False))

    indeed_entry = c.execute(f"""
        SELECT
          SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END) AS entry,
          SUM(CASE WHEN seniority_final IN ('entry','associate') THEN 1 ELSE 0 END) AS entry_or_assoc,
          SUM(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted<=2 THEN 1 ELSE 0 END) AS yoe_le2,
          SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS n_yoe,
          COUNT(*) AS n
        FROM read_parquet('{DATA}')
        WHERE is_swe AND is_english AND date_flag='ok'
          AND source='scraped' AND source_platform='indeed'
    """).df()
    indeed_entry["j1_share"] = indeed_entry["entry"] / indeed_entry["n"]
    indeed_entry["j2_share"] = indeed_entry["entry_or_assoc"] / indeed_entry["n"]
    indeed_entry["j3_share"] = indeed_entry["yoe_le2"] / indeed_entry["n_yoe"]
    indeed_entry.to_csv(TAB / "09_indeed_entry_share.csv", index=False)
    print("Indeed scraped entry shares (J1,J2,J3):"); print(indeed_entry.to_string(index=False))

    return {"nat_pivot": nat_pivot, "yoe_pivot": yoe_pivot, "indeed_entry": indeed_entry,
            "indeed_seniority": indeed}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    c = con()
    s1 = step1_length(c)
    s2 = step2_company(c)
    s3 = step3_geo(c)
    s4 = step4_seniority(c)
    s5 = step5_titles(c)
    s6 = step6_industry(c)
    s8 = step8_within2024(c, s1, s4)
    s9 = step9_platform(c)

    # Small summary JSON for the report
    summary = {
        "length": s1["ks"].to_dict(orient="records"),
        "company_jaccard": s2["overlap"].to_dict(orient="records"),
        "top50_cross_source": s2["top50x"].to_dict(orient="records"),
        "state_chi2": s3["chi"].to_dict(orient="records"),
        "multi_location": s3["ml"].to_dict(orient="records"),
        "seniority_chi2": s4["chi"].to_dict(orient="records"),
        "unknown_share": s4["unknown"].to_dict(orient="records"),
        "industry_chi2": s6["chi"],
        "calibration": s8["calib"].to_dict(orient="records"),
        "indeed_entry": s9["indeed_entry"].to_dict(orient="records"),
    }
    (TAB / "T05_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("--- wrote summary ---")


if __name__ == "__main__":
    main()
