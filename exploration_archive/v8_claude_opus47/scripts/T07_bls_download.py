"""T07 Part B: Download BLS OEWS state-level employment for SOC 15-1252 and 15-1256.

Uses BLS public API (no key required for basic access; 500 queries/day).
Queries annual 2024 data. SWE SOC 15-1252 works; 15-1256 is a rarer series
that may or may not exist at state level. Falls back to national only where
state series aren't published.

Also downloads JOLTS Information sector job openings (JTS510000000000000JOL).
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/jihgaboot/gabor/job-research")
OUT = ROOT / "exploration" / "artifacts" / "T07_benchmarks"
OUT.mkdir(parents=True, exist_ok=True)

API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# US state FIPS codes (50 states + DC)
STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56",
}


def state_series_id(state_fips: str, occ: str, datatype: str = "01") -> str:
    """OEWS series: OEU + S (state) + area(7) + industry(6) + occupation(6) + datatype(2) = 25 chars.
    Area = state_fips(2) + 00000 (5 zeros) = 7 chars total.
    Industry = 000000 (cross-industry).
    """
    area = f"{state_fips}00000"
    industry = "000000"
    return f"OEUS{area}{industry}{occ}{datatype}"


def national_series_id(occ: str, datatype: str = "01") -> str:
    return f"OEUN0000000000000{occ}{datatype}"


def bls_api_call(series_ids: list[str], start_year: str, end_year: str) -> dict:
    """Call BLS public API with a batch of series IDs (max 25 per call)."""
    payload = {
        "seriesid": series_ids,
        "startyear": start_year,
        "endyear": end_year,
    }
    result = subprocess.run(
        [
            "curl", "-s", "-L", "--http1.1",
            "-A", "Mozilla/5.0",
            "-X", "POST",
            "-H", "Content-Type: application/json",
            "-d", json.dumps(payload),
            "--max-time", "60",
            API_URL,
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr}")
    return json.loads(result.stdout)


def extract_annual(data: dict) -> dict[str, int | None]:
    """Extract (latest) annual value for each seriesID."""
    out = {}
    for s in data.get("Results", {}).get("series", []):
        sid = s["seriesID"]
        if not s.get("data"):
            out[sid] = None
            continue
        # Find most recent annual record
        annuals = [d for d in s["data"] if d["period"] == "A01"]
        if not annuals:
            out[sid] = None
            continue
        latest = max(annuals, key=lambda d: int(d["year"]))
        try:
            out[sid] = int(latest["value"])
        except (ValueError, TypeError):
            out[sid] = None
    return out


def fetch_oes_state_data(occ_code: str, occ_name: str) -> dict[str, int | None]:
    """Fetch OEWS state-level employment for all states."""
    states = list(STATE_FIPS.items())
    result: dict[str, int | None] = {}

    # Batch into groups of 25
    for i in range(0, len(states), 25):
        batch = states[i:i + 25]
        series_map = {state_series_id(fips, occ_code): abbr for abbr, fips in batch}
        series_ids = list(series_map.keys())
        try:
            resp = bls_api_call(series_ids, "2024", "2024")
        except Exception as e:
            print(f"  Batch {i} failed: {e}")
            continue

        vals = extract_annual(resp)
        for sid, val in vals.items():
            abbr = series_map[sid]
            result[abbr] = val
        time.sleep(0.5)

    return result


def fetch_jolts_info() -> list[dict]:
    """Download JOLTS Information sector job openings."""
    # Information (NAICS 51): series JTS510000000000000JOL
    series_id = "JTS510000000000000JOL"
    resp = bls_api_call([series_id], "2023", "2026")
    series = resp.get("Results", {}).get("series", [])
    if not series or not series[0].get("data"):
        return []
    data = series[0]["data"]
    return [{"year": d["year"], "period": d["period"],
             "periodName": d["periodName"], "value": d["value"]}
            for d in data]


def main():
    # --- OEWS state-level data ---
    print("Fetching OEWS state-level data for SOC 15-1252 (Software Developers)...")
    swe_by_state = fetch_oes_state_data("151252", "Software Developers")
    print(f"  Got data for {sum(1 for v in swe_by_state.values() if v is not None)} / {len(swe_by_state)} states")

    print("Fetching OEWS state-level data for SOC 15-1256 (QA Analysts & Testers)...")
    qa_by_state = fetch_oes_state_data("151256", "QA Analysts and Testers")
    print(f"  Got data for {sum(1 for v in qa_by_state.values() if v is not None)} / {len(qa_by_state)} states")

    # National totals (context)
    print("Fetching national totals...")
    nat_resp = bls_api_call(
        [national_series_id("151252"), national_series_id("151256")],
        "2024", "2024",
    )
    nat = extract_annual(nat_resp)
    print(f"  National 15-1252: {nat.get(national_series_id('151252'))}")
    print(f"  National 15-1256: {nat.get(national_series_id('151256'))}")

    # Save
    with open(OUT / "oes_state_2024.json", "w") as f:
        json.dump({
            "swe_15_1252": swe_by_state,
            "qa_15_1256": qa_by_state,
            "national_15_1252": nat.get(national_series_id("151252")),
            "national_15_1256": nat.get(national_series_id("151256")),
            "source": "BLS OEWS public API",
            "year": 2024,
        }, f, indent=2)
    print(f"  Saved: {OUT / 'oes_state_2024.json'}")

    # --- JOLTS Information sector ---
    print("\nFetching JOLTS Information sector job openings...")
    jolts = fetch_jolts_info()
    print(f"  Got {len(jolts)} monthly observations")
    if jolts:
        most_recent = max(jolts, key=lambda d: (int(d["year"]), d["period"]))
        print(f"  Most recent: {most_recent['year']}-{most_recent['period']} = {most_recent['value']}")
    with open(OUT / "jolts_info_510_openings.json", "w") as f:
        json.dump({
            "series_id": "JTS510000000000000JOL",
            "data": jolts,
            "source": "BLS JOLTS public API",
        }, f, indent=2)
    print(f"  Saved: {OUT / 'jolts_info_510_openings.json'}")


if __name__ == "__main__":
    main()
