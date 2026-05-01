"""Produce the headline figures for T05 and T06.

Figures (max 4 each):
T05/
  01_length_hist.png          (already produced by T05_cross_dataset.py)
  02_entry_share_calibration.png
  03_shared_title_native_vs_yoe.png
T06/
  01_lorenz_concentration.png
  02_entry_concentration_shape.png
  03_specialist_impact.png
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data" / "unified.parquet"
FIG_T05 = ROOT / "exploration" / "figures" / "T05"
FIG_T06 = ROOT / "exploration" / "figures" / "T06"
TAB_T05 = ROOT / "exploration" / "tables" / "T05"
TAB_T06 = ROOT / "exploration" / "tables" / "T06"


def con():
    c = duckdb.connect()
    c.execute("SET memory_limit='12GB'")
    c.execute("SET threads=6")
    return c


# -------- T05.2 — entry share calibration bars (J1,J2,J3 × source) --------

def t05_fig_calibration():
    entry = pd.read_csv(TAB_T05 / "08_entry_shares.csv")
    entry["j1_share"] = entry["entry"] / entry["n"]
    entry["j2_share"] = entry["entry_or_assoc"] / entry["n"]
    entry["j3_share"] = entry["yoe_le2"] / entry["n_yoe"]
    entry = entry.set_index("source")

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=False)
    defs = [("J1 — seniority_final=entry", "j1_share"),
            ("J2 — entry ∪ associate", "j2_share"),
            ("J3 — yoe_extracted ≤ 2", "j3_share")]
    cmap = {"kaggle_arshkon": "#1f77b4", "kaggle_asaniczka": "#ff7f0e", "scraped": "#2ca02c"}
    for ax, (title, col) in zip(axes, defs):
        vals = entry[col]
        ax.bar(vals.index, vals.values, color=[cmap[s] for s in vals.index])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("share")
        ax.tick_params(axis="x", rotation=25)
        for i, v in enumerate(vals.values):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Entry-share under J1/J2/J3 across sources (LinkedIn SWE)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_T05 / "02_entry_share_calibration.png", dpi=150)
    plt.close(fig)


# -------- T05.3 — native entry-share vs YOE for shared titles --------

def t05_fig_titles():
    nat = pd.read_csv(TAB_T05 / "09_native_entry_share_shift.csv")
    yoe = pd.read_csv(TAB_T05 / "09_mean_yoe_shift.csv")
    m = nat.merge(yoe, on="t", suffixes=("_nat", "_yoe"))
    # Drop rows with NaN
    m = m.dropna()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.axvline(0, color="gray", linestyle="--", alpha=0.6)
    ax.scatter(m["delta_nat"], range(len(m)), color="tab:blue", s=55)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(m["t"], fontsize=7)
    ax.set_xlabel("Δ native entry share (scraped − arshkon)")
    ax.set_title("Same title, native entry share shift")

    ax = axes[1]
    ax.axvline(0, color="gray", linestyle="--", alpha=0.6)
    ax.scatter(m["delta_yoe"], range(len(m)), color="tab:orange", s=55)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels([""] * len(m))
    ax.set_xlabel("Δ mean YOE (scraped − arshkon)")
    ax.set_title("Same title, mean YOE shift")
    fig.suptitle("Platform stability test: shared LinkedIn SWE titles")
    fig.tight_layout()
    fig.savefig(FIG_T05 / "03_shared_title_native_vs_yoe.png", dpi=150)
    plt.close(fig)


# -------- T06.1 — Lorenz curves --------

def t06_fig_lorenz():
    c = con()
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = {"kaggle_arshkon": "#1f77b4", "kaggle_asaniczka": "#ff7f0e", "scraped": "#2ca02c"}
    for src in ("kaggle_arshkon", "kaggle_asaniczka", "scraped"):
        df = c.execute(f"""
            SELECT company_name_canonical AS c, COUNT(*) AS n
            FROM read_parquet('{DATA}')
            WHERE source_platform='linkedin' AND is_english AND date_flag='ok' AND is_swe
              AND source='{src}' AND company_name_canonical IS NOT NULL
            GROUP BY 1
        """).df()
        counts = np.sort(df["n"].to_numpy())
        total = counts.sum()
        cum_co = np.arange(1, len(counts) + 1) / len(counts)
        cum_post = np.cumsum(counts) / total
        ax.plot(cum_co, cum_post, label=src, color=cmap[src], lw=1.8)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="equality")
    ax.set_xlabel("cumulative fraction of companies (sorted by postings ascending)")
    ax.set_ylabel("cumulative fraction of postings")
    ax.set_title("Lorenz curves of SWE posting volume (LinkedIn)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_T06 / "01_lorenz_concentration.png", dpi=150)
    plt.close(fig)


# -------- T06.2 — Entry concentration shape (share of companies with 0 entry) --------

def t06_fig_entry_shape():
    shape = pd.read_csv(TAB_T06 / "04_entry_concentration_shape.csv")
    labels = ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]
    defs = ["J1", "J2", "J3", "J4"]
    data = np.zeros((len(labels), len(defs)))
    for i, src in enumerate(labels):
        for j, d in enumerate(defs):
            row = shape[(shape["source"] == src) & (shape["definition"] == d)]
            if not row.empty:
                data[i, j] = float(row["share_ge5_zero_entry"].iloc[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(defs))
    w = 0.25
    for i, src in enumerate(labels):
        ax.bar(x + (i - 1) * w, data[i], w, label=src)
    ax.set_xticks(x)
    ax.set_xticklabels(defs)
    ax.set_ylabel("share of companies (n≥5) with ZERO entry rows")
    ax.set_title("Most SWE employers post zero entry-level roles under any variant")
    ax.set_ylim(0, 1)
    ax.legend()
    for i, src in enumerate(labels):
        for j, v in enumerate(data[i]):
            ax.text(x[j] + (i - 1) * w, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_T06 / "02_entry_concentration_shape.png", dpi=150)
    plt.close(fig)


# -------- T06.3 — specialist impact (entry share with / without specialists) --------

def t06_fig_specialist_impact():
    # Recompute on-the-fly to stay self-contained (uses artifact CSV)
    c = con()
    res = c.execute(f"""
        WITH spec AS (
          SELECT company FROM read_csv('{ROOT}/exploration/artifacts/shared/entry_specialist_employers.csv')
        ),
        swe AS (
          SELECT * FROM read_parquet('{DATA}')
          WHERE source_platform='linkedin' AND is_english AND date_flag='ok' AND is_swe
        )
        SELECT source,
               SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS j1_full,
               SUM(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END) FILTER (WHERE company_name_canonical NOT IN (SELECT company FROM spec))::DOUBLE
                 / NULLIF(COUNT(*) FILTER (WHERE company_name_canonical NOT IN (SELECT company FROM spec)), 0) AS j1_excl,
               SUM(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted<=2 THEN 1 ELSE 0 END)::DOUBLE
                 / NULLIF(SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END), 0) AS j3_full,
               SUM(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted<=2 THEN 1 ELSE 0 END) FILTER (WHERE company_name_canonical NOT IN (SELECT company FROM spec))::DOUBLE
                 / NULLIF(SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) FILTER (WHERE company_name_canonical NOT IN (SELECT company FROM spec)), 0) AS j3_excl
        FROM swe
        GROUP BY source ORDER BY source
    """).df()
    res.to_csv(TAB_T06 / "06_specialist_impact_detailed.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(res))
    w = 0.35
    axes[0].bar(x - w/2, res["j1_full"], w, label="all", color="#1f77b4")
    axes[0].bar(x + w/2, res["j1_excl"], w, label="exclude specialists", color="#aec7e8")
    axes[0].set_xticks(x); axes[0].set_xticklabels(res["source"], rotation=20)
    axes[0].set_ylabel("J1 share")
    axes[0].set_title("J1 — seniority_final = entry")
    axes[0].legend()
    for i, (a, b) in enumerate(zip(res["j1_full"], res["j1_excl"])):
        axes[0].text(i - w/2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        axes[0].text(i + w/2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(x - w/2, res["j3_full"], w, label="all", color="#2ca02c")
    axes[1].bar(x + w/2, res["j3_excl"], w, label="exclude specialists", color="#98df8a")
    axes[1].set_xticks(x); axes[1].set_xticklabels(res["source"], rotation=20)
    axes[1].set_ylabel("J3 share")
    axes[1].set_title("J3 — yoe_extracted ≤ 2")
    axes[1].legend()
    for i, (a, b) in enumerate(zip(res["j3_full"], res["j3_excl"])):
        axes[1].text(i - w/2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        axes[1].text(i + w/2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Entry-specialist exclusion flips the J3 cross-period direction")
    fig.tight_layout()
    fig.savefig(FIG_T06 / "03_specialist_impact.png", dpi=150)
    plt.close(fig)


def main():
    t05_fig_calibration()
    t05_fig_titles()
    t06_fig_lorenz()
    t06_fig_entry_shape()
    t06_fig_specialist_impact()
    print("wrote figures")


if __name__ == "__main__":
    main()
