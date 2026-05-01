"""Additional figures for T06."""
from __future__ import annotations
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data" / "unified.parquet"
FIG = ROOT / "exploration" / "figures" / "T06"
TBL = ROOT / "exploration" / "tables" / "T06"

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")

BASE_WHERE = "source_platform='linkedin' AND is_english AND date_flag='ok' AND is_swe"
SOURCES = ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]


# ---------------------------------------------------------------------------
# Lorenz / cumulative share
# ---------------------------------------------------------------------------
plt.figure(figsize=(9, 6))
for src in SOURCES:
    for agg_flag, label_suffix, ls in [(True, " (all)", "--"), (False, " (non-agg)", "-")]:
        where = f"{BASE_WHERE} AND source='{src}'"
        if not agg_flag:
            where += " AND NOT is_aggregator"
        df = con.execute(
            f"""SELECT company_name_canonical, COUNT(*) as n
               FROM '{DATA}' WHERE {where}
               GROUP BY company_name_canonical"""
        ).fetchdf()
        if df.empty:
            continue
        n = df["n"].values
        n = np.sort(n)[::-1]  # descending
        cum = np.cumsum(n) / n.sum()
        xs = np.arange(1, len(cum) + 1) / len(cum)
        plt.plot(xs, cum, linestyle=ls, label=f"{src}{label_suffix}")
plt.xlabel("company rank (fraction)")
plt.ylabel("cumulative share of postings")
plt.title("Cumulative posting share by company (SWE, LinkedIn)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "cumulative_posting_share.png", dpi=120)
plt.close()
print(f"wrote {FIG / 'cumulative_posting_share.png'}")


# ---------------------------------------------------------------------------
# Entry-share distribution histogram per source (J3 primary)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for i, src in enumerate(SOURCES):
    where = f"{BASE_WHERE} AND source='{src}'"
    q = f"""
    SELECT company_name_canonical,
           COUNT(*) AS n_swe,
           COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                            AND yoe_min_years_llm IS NOT NULL) AS n_llm,
           COUNT(*) FILTER (WHERE llm_classification_coverage='labeled'
                            AND yoe_min_years_llm IS NOT NULL
                            AND yoe_min_years_llm <= 2) AS n_j3
    FROM '{DATA}'
    WHERE {where}
    GROUP BY company_name_canonical
    HAVING n_llm >= 5
    """
    df = con.execute(q).fetchdf()
    shares = df["n_j3"] / df["n_llm"]
    axes[i].hist(shares, bins=np.linspace(0, 1, 21), edgecolor="black")
    axes[i].axvline(0.60, color="red", linestyle="--", label="60% threshold")
    axes[i].axvline(shares.median(), color="green", linestyle="--", label=f"median={shares.median():.3f}")
    axes[i].set_title(f"{src}\n(n companies with >=5 LLM-labeled = {len(df):,})")
    axes[i].set_xlabel("J3 share (yoe<=2 within LLM frame)")
    axes[i].legend()
axes[0].set_ylabel("n companies")
plt.suptitle("J3 entry-share distribution per company (SWE, LinkedIn, >=5 LLM-labeled)")
plt.tight_layout()
plt.savefig(FIG / "j3_entry_share_distribution.png", dpi=120)
plt.close()
print(f"wrote {FIG / 'j3_entry_share_distribution.png'}")


# ---------------------------------------------------------------------------
# J1..J4 variants within a single source (scraped)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
variants = [
    ("J1 (seniority='entry')", "n_j1", "n_swe"),
    ("J2 (entry or associate)", "n_j2", "n_swe"),
    ("J3 (yoe_llm<=2)", "n_j3", "n_llm"),
    ("J4 (yoe_llm<=3)", "n_j4", "n_llm"),
]
where = f"{BASE_WHERE} AND source='scraped'"
q = f"""
SELECT company_name_canonical,
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
WHERE {where}
GROUP BY company_name_canonical
HAVING n_swe >= 5
"""
df = con.execute(q).fetchdf()
for i, (label, col, denom) in enumerate(variants):
    sub = df[df[denom] > 0].copy()
    shares = sub[col] / sub[denom]
    axes[i].hist(shares, bins=np.linspace(0, 1, 21), edgecolor="black")
    axes[i].axvline(0.60, color="red", linestyle="--", label="60%")
    axes[i].axvline(shares.median(), color="green", linestyle="--")
    axes[i].set_title(f"{label}\n(n={len(sub):,}, med={shares.median():.3f})")
    axes[i].set_xlabel("entry-share per company")
axes[0].set_ylabel("n companies")
plt.suptitle("Entry-share distribution per company (scraped SWE, LinkedIn, >=5 postings)")
plt.tight_layout()
plt.savefig(FIG / "entry_share_all_variants_scraped.png", dpi=120)
plt.close()
print(f"wrote {FIG / 'entry_share_all_variants_scraped.png'}")


# ---------------------------------------------------------------------------
# Top-20 employer volume bar chart per source
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 14))
for i, src in enumerate(SOURCES):
    df = pd.read_csv(TBL / "top20_employer_profile.csv")
    df = df[df["source"] == src].sort_values("n", ascending=True)
    axes[i].barh(df["company_name_canonical"], df["n"])
    axes[i].set_title(f"{src} top-20 SWE employers")
    axes[i].set_xlabel("n postings")
plt.tight_layout()
plt.savefig(FIG / "top20_volume_bar.png", dpi=120)
plt.close()
print(f"wrote {FIG / 'top20_volume_bar.png'}")
