"""
T07 Part B: JOLTS contextualization.

Uses the FRED JOLTS Information sector Job Openings series (JTU5100JOL, NSA) and
total nonfarm (JTSJOL, SA) to contextualize our 2024 baseline and 2026 scraped
window within the hiring cycle.

Output:
  exploration/tables/T07/jolts_context.csv (annual averages + key ratios)
  exploration/figures/T07/jolts_information_series.png
  exploration/figures/T07/jolts_total_nonfarm.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
TABLES = ROOT / "exploration" / "tables" / "T07"
FIGS = ROOT / "exploration" / "figures" / "T07"

INFO_FILE = "/tmp/jtu5100_csv.txt"
NF_FILE = "/tmp/jtsjol.txt"


def main() -> None:
    info = pd.read_csv(INFO_FILE, parse_dates=["observation_date"])
    info = info.rename(columns={"JTU5100JOL": "info_openings_k"})
    info["year"] = info["observation_date"].dt.year

    nf = pd.read_csv(NF_FILE, parse_dates=["observation_date"])
    nf = nf.rename(columns={"JTSJOL": "total_openings_k"})
    nf["year"] = nf["observation_date"].dt.year

    # Annual averages for Info sector
    info_annual = info.groupby("year")["info_openings_k"].mean().reset_index()
    nf_annual = nf.groupby("year")["total_openings_k"].mean().reset_index()

    combined = info_annual.merge(nf_annual, on="year", how="outer")
    combined["info_share_pct"] = 100 * combined["info_openings_k"] / combined["total_openings_k"]

    # Ratios of interest
    info_2023 = info_annual.loc[info_annual["year"] == 2023, "info_openings_k"].mean()
    info_2024 = info_annual.loc[info_annual["year"] == 2024, "info_openings_k"].mean()
    info_2025 = info_annual.loc[info_annual["year"] == 2025, "info_openings_k"].mean()
    # 2026 partial average
    info_2026 = info_annual.loc[info_annual["year"] == 2026, "info_openings_k"].mean()
    info_latest = info.iloc[-1]
    nf_latest = nf.iloc[-1]

    summary_rows = [
        {"metric": "2023 avg info openings (thousands, NSA)", "value": round(info_2023, 1)},
        {"metric": "2024 avg info openings (thousands, NSA)", "value": round(info_2024, 1)},
        {"metric": "2025 avg info openings (thousands, NSA)", "value": round(info_2025, 1)},
        {"metric": "2026 YTD avg info openings (thousands, NSA)", "value": round(info_2026, 1)},
        {"metric": "latest month info openings (thousands)", "value": info_latest["info_openings_k"]},
        {"metric": "latest month date", "value": str(info_latest["observation_date"].date())},
        {"metric": "ratio 2026_YTD / 2023_avg", "value": round(info_2026 / info_2023, 3)},
        {"metric": "ratio 2024_avg / 2023_avg", "value": round(info_2024 / info_2023, 3)},
        {"metric": "ratio 2025_avg / 2023_avg", "value": round(info_2025 / info_2023, 3)},
        {"metric": "total nonfarm 2023 avg", "value": round(nf_annual.loc[nf_annual['year']==2023,'total_openings_k'].mean(), 1)},
        {"metric": "total nonfarm 2024 avg", "value": round(nf_annual.loc[nf_annual['year']==2024,'total_openings_k'].mean(), 1)},
        {"metric": "total nonfarm 2025 avg", "value": round(nf_annual.loc[nf_annual['year']==2025,'total_openings_k'].mean(), 1)},
        {"metric": "total nonfarm 2026 YTD avg", "value": round(nf_annual.loc[nf_annual['year']==2026,'total_openings_k'].mean(), 1)},
        {"metric": "total nonfarm latest", "value": nf_latest["total_openings_k"]},
        {"metric": "total nonfarm latest date", "value": str(nf_latest["observation_date"].date())},
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES / "jolts_context_summary.csv", index=False)
    print(summary.to_string(index=False))
    print()

    combined.to_csv(TABLES / "jolts_annual.csv", index=False)
    print("Annual:")
    print(combined.tail(12).to_string(index=False))

    # Plots
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(info["observation_date"], info["info_openings_k"], label="Info sector job openings (NSA, thousands)", color="steelblue")
    # 12-month rolling mean
    info["r12"] = info["info_openings_k"].rolling(12, min_periods=12).mean()
    ax.plot(info["observation_date"], info["r12"], label="12-month rolling mean", color="darkorange", linewidth=2)

    # Mark our comparison windows
    for start, end, label, color in [
        ("2024-01-01", "2024-05-31", "2024 baseline", "lightgreen"),
        ("2026-03-01", "2026-04-30", "2026 scraped", "lightcoral"),
    ]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.3, color=color, label=label)

    ax.set_title("JOLTS Information Sector Job Openings (JTU5100JOL)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Thousands of openings (NSA)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "jolts_information_series.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(nf["observation_date"], nf["total_openings_k"], color="navy", label="Total nonfarm openings (SA)")
    for start, end, label, color in [
        ("2024-01-01", "2024-05-31", "2024 baseline", "lightgreen"),
        ("2026-03-01", "2026-04-30", "2026 scraped", "lightcoral"),
    ]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.3, color=color, label=label)
    ax.set_title("JOLTS Total Nonfarm Job Openings (JTSJOL, SA)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Thousands of openings")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "jolts_total_nonfarm.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    pd.set_option("display.max_rows", 40)
    pd.set_option("display.width", 200)
    main()
