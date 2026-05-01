"""T07 Part B: BLS OES industry breakdown for Software Developers (15-1252)
and crude comparison against our company_industry distribution.

Our company_industry comes from LinkedIn's industry taxonomy, which differs
from NAICS. So a direct quantitative correlation is not meaningful. Instead:
- Report BLS OEWS share-by-NAICS-industry for SWE employment
- Report our top industries in arshkon and scraped
- Note the qualitative alignment (tech services dominates both)

Also analyzes JOLTS Information sector for cycle-positioning.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
BENCH = ROOT / "exploration" / "artifacts" / "T07_benchmarks"
OUT_TABLES = ROOT / "exploration" / "tables" / "T07"
OUT_TABLES.mkdir(parents=True, exist_ok=True)

# Industries where Software Developers are concentrated (NAICS 4-digit)
NAICS_CANDIDATES = {
    "5415": "Computer Systems Design and Related Services",
    "5182": "Computing Infrastructure Providers and Data Processing",
    "5112": "Software Publishers",  # (may not have data post-NAICS 2022 revision)
    "5413": "Architectural, Engineering, and Related Services",
    "5416": "Management, Scientific, and Technical Consulting Services",
    "5417": "Scientific Research and Development Services",
    "3364": "Aerospace Product and Parts Manufacturing",
    "5221": "Depository Credit Intermediation",
    "5220": "Credit Intermediation and Related Activities (NAICS 522)",
    "5611": "Office Administrative Services",
    "5611": "Office Administrative Services",
    "5191": "Other Information Services",
    "5173": "Wired & Wireless Telecommunications",
    "5241": "Insurance Carriers and Related Activities",
    "5621": "Waste Collection",
    "5617": "Services to Buildings and Dwellings",
    "5231": "Securities, Commodity Contracts, and Other Financial Investments",
    "5239": "Other Financial Investment Activities",
    "6221": "General Medical and Surgical Hospitals",
    "2211": "Electric Power Generation, Transmission and Distribution",
    "4541": "Electronic Shopping and Mail-Order Houses",
    "3344": "Semiconductor and Other Electronic Component Manufacturing",
    "3345": "Navigational, Measuring, Electromedical, and Control Instruments Manufacturing",
    "5191": "Other Information Services",
    "5323": "General Rental Centers",
    "6214": "Outpatient Care Centers",
    "9200": "Government",
    "6111": "Elementary and Secondary Schools",
    "6113": "Colleges and Universities",
    "4812": "Nonscheduled Air Transportation",
    "5411": "Legal Services",
    "5179": "Other Telecommunications",
    "3363": "Motor Vehicle Parts Manufacturing",
    "5415": "Computer Systems Design Services",
}


def bls_api_call(series_ids: list[str], year: str = "2024") -> dict:
    payload = {"seriesid": series_ids, "startyear": year, "endyear": year}
    result = subprocess.run(
        ["curl", "-s", "-L", "--http1.1", "-A", "Mozilla/5.0", "-X", "POST",
         "-H", "Content-Type: application/json",
         "-d", json.dumps(payload),
         "--max-time", "60",
         "https://api.bls.gov/publicAPI/v2/timeseries/data/"],
        capture_output=True, text=True, timeout=90,
    )
    return json.loads(result.stdout)


def fetch_industry_swe_employment() -> dict[str, int | None]:
    """BLS OEWS 2024 national SWE employment by NAICS 4-digit industry."""
    series_to_naics = {}
    for naics, name in NAICS_CANDIDATES.items():
        # 6-digit industry code padded with 00 at the end
        ind_code = f"{naics}00"  # 6 chars
        sid = f"OEUN0000000{ind_code}15125201"
        assert len(sid) == 25, f"{sid} is {len(sid)} chars"
        series_to_naics[sid] = (naics, name)

    out = {}
    ids = list(series_to_naics.keys())
    # Batch of 25
    for i in range(0, len(ids), 25):
        batch = ids[i:i + 25]
        resp = bls_api_call(batch, "2024")
        for s in resp.get("Results", {}).get("series", []):
            sid = s["seriesID"]
            naics, name = series_to_naics[sid]
            if not s["data"]:
                continue
            annuals = [d for d in s["data"] if d["period"] == "A01"]
            if not annuals:
                continue
            v = annuals[0]["value"]
            try:
                out[naics] = {"name": name, "value": int(v)}
            except (ValueError, TypeError):
                pass

    return out


def load_jolts() -> pd.DataFrame:
    with open(BENCH / "jolts_info_510_openings.json") as f:
        d = json.load(f)
    rows = []
    for obs in d["data"]:
        # Value is in thousands, convert to integer job openings
        rows.append({
            "year": int(obs["year"]),
            "period": obs["period"],
            "month": int(obs["period"][1:]),  # M01 -> 1
            "value_thousands": int(obs["value"]),
        })
    df = pd.DataFrame(rows).sort_values(["year", "month"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    return df


def main():
    print("=== Fetching BLS OEWS industry-level SWE employment (2024) ===")
    ind_data = fetch_industry_swe_employment()
    rows = []
    for naics, rec in ind_data.items():
        rows.append({"naics": naics, "industry": rec["name"], "swe_employment_2024": rec["value"]})
    ind_df = pd.DataFrame(rows).sort_values("swe_employment_2024", ascending=False).reset_index(drop=True)
    total_known = ind_df["swe_employment_2024"].sum()
    ind_df["share_of_queried"] = (ind_df["swe_employment_2024"] / total_known * 100).round(1)
    ind_df.to_csv(OUT_TABLES / "bls_industry_swe.csv", index=False)
    print(ind_df.to_string(index=False))
    print(f"\nCoverage (sum of queried industries / national 1,654,440): "
          f"{total_known / 1654440:.1%}")

    print("\n=== JOLTS Information sector job openings ===")
    jolts = load_jolts()
    print(jolts.to_string())

    # 3-month averages around our observation windows
    print("\nContextual averages:")
    arshkon_period = jolts[(jolts["year"] == 2024) & (jolts["month"].between(3, 5))]
    asaniczka_period = jolts[(jolts["year"] == 2024) & (jolts["month"].between(1, 3))]
    scraped_period_mar = jolts[(jolts["year"] == 2026) & (jolts["month"].between(1, 2))]  # JOLTS through Feb 2026
    peak_2022 = jolts[(jolts["year"] == 2024) & (jolts["month"] == 3)]
    print(f"  Arshkon (2024-Apr): {arshkon_period['value_thousands'].mean():.0f}K avg openings")
    print(f"  Asaniczka (2024-Jan): {asaniczka_period['value_thousands'].mean():.0f}K avg openings")
    print(f"  Scraped (2026-Mar/Apr, using Jan-Feb 2026 as proxy since JOLTS Apr 2026 not yet published): "
          f"{scraped_period_mar['value_thousands'].mean():.0f}K avg openings")

    # Ratio: scraped window vs arshkon window
    arshkon_avg = arshkon_period["value_thousands"].mean()
    scraped_avg = scraped_period_mar["value_thousands"].mean()
    ratio = scraped_avg / arshkon_avg
    print(f"\n  JOLTS Info-sector openings: 2026-Q1 / 2024-Apr = {ratio:.2f}")
    print(f"  (Values <1 indicate lower aggregate hiring intensity in scraped window.)")

    jolts.to_csv(OUT_TABLES / "jolts_info_monthly.csv", index=False)


if __name__ == "__main__":
    main()
