"""
T07 Part B: BLS OES state-level correlation.

Compares our LinkedIn SWE posting counts by state to BLS OES May 2024 state-level
employment for software developers (SOC 15-1252) and QA analysts/testers (SOC
15-1253). Compute Pearson r on total and per-capita bases.

Note: Dispatch referenced SOC 15-1256 for QA; actual OES code is 15-1253. Use 15-1253.

Output:
  exploration/tables/T07/bls_oes_state_correlation.csv (state-level join)
  exploration/tables/T07/bls_oes_correlation_summary.csv (Pearson r values)
  exploration/figures/T07/bls_oes_scatter.png
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
DATA = ROOT / "data" / "unified.parquet"
TABLES = ROOT / "exploration" / "tables" / "T07"
FIGS = ROOT / "exploration" / "figures" / "T07"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

BLS_FILE = "/tmp/oes_st/oesm24st/state_M2024_dl.xlsx"

BASE_WHERE = (
    "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe"
)


STATE_NAME_TO_CODE = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
    "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "District of Columbia": "DC", "Puerto Rico": "PR",
}


def load_bls() -> pd.DataFrame:
    df = pd.read_excel(BLS_FILE)
    df = df[df["AREA_TYPE"] == 2]  # state-level
    df = df[df["OCC_CODE"].isin(["15-1252", "15-1253"])]
    df["TOT_EMP"] = pd.to_numeric(df["TOT_EMP"], errors="coerce")
    pivot = df.pivot_table(
        index="AREA_TITLE", columns="OCC_CODE", values="TOT_EMP", aggfunc="sum"
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"15-1252": "bls_sd_emp", "15-1253": "bls_qa_emp"})
    pivot["bls_sd_emp"] = pivot["bls_sd_emp"].fillna(0)
    pivot["bls_qa_emp"] = pivot["bls_qa_emp"].fillna(0)
    pivot["bls_total_emp"] = pivot["bls_sd_emp"] + pivot["bls_qa_emp"]
    pivot["state_code"] = pivot["AREA_TITLE"].map(STATE_NAME_TO_CODE)
    return pivot[pivot["state_code"].notna()]


def load_our_counts(source_filter: str) -> pd.DataFrame:
    sql = f"""
        SELECT state_normalized AS state_code, count(*) AS n
        FROM '{DATA}'
        WHERE {BASE_WHERE}
          AND {source_filter}
          AND state_normalized IS NOT NULL
        GROUP BY state_normalized
    """
    return duckdb.sql(sql).df()


def pearson(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    """Pearson r with N used."""
    mask = a.notna() & b.notna()
    a2 = a[mask]
    b2 = b[mask]
    if len(a2) < 3:
        return (float("nan"), int(mask.sum()))
    # Use numpy for robustness
    r = float(np.corrcoef(a2.values, b2.values)[0, 1])
    return (r, int(mask.sum()))


def main() -> None:
    bls = load_bls()
    print("BLS OES state-level rows:", len(bls))
    print(bls.head())

    sources = [
        ("arshkon", "source = 'kaggle_arshkon'"),
        ("asaniczka", "source = 'kaggle_asaniczka'"),
        ("pooled_2024", "source IN ('kaggle_arshkon','kaggle_asaniczka')"),
        ("scraped", "source = 'scraped'"),
    ]

    results_rows = []
    # Merge to a wide frame for sharing
    wide = bls.copy()
    for label, filt in sources:
        counts = load_our_counts(filt)
        counts = counts.rename(columns={"n": f"our_n_{label}"})
        wide = wide.merge(counts, on="state_code", how="left")

    # Fill missing with 0
    for label, _ in sources:
        wide[f"our_n_{label}"] = wide[f"our_n_{label}"].fillna(0)

    wide.to_csv(TABLES / "bls_oes_state_counts_merged.csv", index=False)

    # Correlations against total BLS (SD+QA) and against SD alone, log-log and linear
    for bls_metric in ["bls_sd_emp", "bls_total_emp"]:
        for label, _ in sources:
            our_col = f"our_n_{label}"
            r_lin, n_lin = pearson(wide[bls_metric], wide[our_col])
            # log-log (add 1 to handle zeros)
            r_log, n_log = pearson(
                np.log1p(wide[bls_metric]), np.log1p(wide[our_col])
            )
            results_rows.append(
                {
                    "our_source": label,
                    "bls_metric": bls_metric,
                    "r_linear": round(r_lin, 4),
                    "n_linear": n_lin,
                    "r_log_log": round(r_log, 4),
                    "n_log_log": n_log,
                }
            )

    res = pd.DataFrame(results_rows)
    res.to_csv(TABLES / "bls_oes_correlation_summary.csv", index=False)
    print()
    print("Correlation summary:")
    print(res.to_string(index=False))

    # Also compute per-capita version (our / BLS) is meaningless; instead look
    # at share-of-total correlation.
    # Share correlation: state_share_ours vs state_share_bls
    wide2 = wide.copy()
    for label, _ in sources:
        col = f"our_n_{label}"
        tot = wide2[col].sum()
        if tot > 0:
            wide2[f"share_{label}"] = wide2[col] / tot
    wide2["share_bls_sd"] = wide2["bls_sd_emp"] / wide2["bls_sd_emp"].sum()
    wide2["share_bls_tot"] = wide2["bls_total_emp"] / wide2["bls_total_emp"].sum()

    share_rows = []
    for label, _ in sources:
        if f"share_{label}" in wide2.columns:
            r_sd, n_sd = pearson(wide2["share_bls_sd"], wide2[f"share_{label}"])
            r_tot, n_tot = pearson(wide2["share_bls_tot"], wide2[f"share_{label}"])
            share_rows.append(
                {
                    "our_source": label,
                    "share_vs_bls_sd": round(r_sd, 4),
                    "share_vs_bls_tot": round(r_tot, 4),
                    "n": n_sd,
                }
            )
    share_df = pd.DataFrame(share_rows)
    share_df.to_csv(TABLES / "bls_oes_share_correlation.csv", index=False)
    print()
    print("State-share correlation summary:")
    print(share_df.to_string(index=False))

    # Scatter plots for pooled_2024 and scraped vs bls_total_emp (log-log)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, label in zip(axes, ["pooled_2024", "scraped"]):
        x = np.log1p(wide["bls_total_emp"])
        y = np.log1p(wide[f"our_n_{label}"])
        ax.scatter(x, y, alpha=0.6)
        # Label top 5 by our count
        top = wide.sort_values(f"our_n_{label}", ascending=False).head(6)
        for _, r in top.iterrows():
            ax.annotate(
                r["state_code"],
                (np.log1p(r["bls_total_emp"]), np.log1p(r[f"our_n_{label}"])),
                fontsize=9,
            )
        r_log, _ = pearson(x, y)
        ax.set_title(f"Our {label} vs BLS OES (15-1252+15-1253), state-level\nlog1p scale, r={r_log:.3f}")
        ax.set_xlabel("log(1+BLS employment)")
        ax.set_ylabel(f"log(1+our {label} postings)")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "bls_oes_scatter.png", dpi=120)
    plt.close(fig)
    print(f"\nScatter figure: {FIGS / 'bls_oes_scatter.png'}")


if __name__ == "__main__":
    pd.set_option("display.max_rows", 60)
    pd.set_option("display.width", 200)
    main()
