"""Fetch BLS OES state-level employment for SOC 15-1252 and 15-1256 via the
BLS public API and compute the geographic correlation against our state-level
SWE counts.

Outputs:
  - exploration/tables/T07/bls_state_oes.csv (state, soc, employment)
  - exploration/tables/T07/bls_vs_ours_state.csv (state, bls, arshkon_n, scraped_n, pooled_n)
  - exploration/figures/T07/bls_vs_ours_state.png (scatter)
  - prints Pearson / Spearman correlations

Notes:
  - BLS public API v1 is rate limited (25 queries / day, up to 25 series per
    query, max 10 years per query). 51 states x 2 SOCs = 102 series; that fits
    in 5 calls of <=25 series. We also limit to one year window (2024) to
    avoid the 10-year cap.
  - This script is safe to rerun. If a state is missing it just drops it.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "exploration" / "tables" / "T07"
FIG = REPO / "exploration" / "figures" / "T07"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

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

SOC_CODES = {
    "software_developers_15_1252": "151252",
    "software_qa_15_1256": "151256",
}


def series_id(fips: str, soc: str) -> str:
    # OEUS + areacode(7) + industry(6) + occupation(6) + datatype(2)
    return f"OEUS{fips}00000000000{soc}01"


def query_bls(series_ids: list[str]) -> dict:
    payload = json.dumps({"seriesid": series_ids})
    # Use curl with http1.1 because http/2 is flaky in this environment.
    proc = subprocess.run(
        [
            "curl", "--http1.1", "-sS", "-X", "POST",
            "https://api.bls.gov/publicAPI/v1/timeseries/data/",
            "-H", "Content-Type: application/json",
            "-d", payload,
        ],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout)


def fetch_all():
    # Build all series (102).
    all_ids = []
    id_to_meta = {}
    for soc_label, soc in SOC_CODES.items():
        for state, fips in STATE_FIPS.items():
            sid = series_id(fips, soc)
            all_ids.append(sid)
            id_to_meta[sid] = (state, soc_label)

    # Chunk at 25 per call.
    results = {}
    for i in range(0, len(all_ids), 25):
        chunk = all_ids[i:i+25]
        resp = query_bls(chunk)
        if resp.get("status") != "REQUEST_SUCCEEDED":
            print(f"chunk {i}: failed: {resp.get('status')} {resp.get('message')}")
            continue
        for s in resp.get("Results", {}).get("series", []):
            sid = s["seriesID"]
            state, soc_label = id_to_meta[sid]
            for d in s.get("data", []):
                if d.get("period") == "A01":
                    try:
                        v = int(d["value"])
                    except Exception:
                        v = None
                    results[(state, soc_label)] = {
                        "year": d["year"], "value": v,
                    }
    return results


def main():
    # 1) Fetch from BLS API.
    r = fetch_all()
    print(f"Fetched {len(r)} (state, soc) observations")

    # Write per-soc per-state employment table.
    bls_path = OUT / "bls_state_oes.csv"
    with bls_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "soc_label", "year", "employment"])
        for (state, soc_label), d in sorted(r.items()):
            w.writerow([state, soc_label, d["year"], d["value"]])
    print(f"Wrote {bls_path}")

    # 2) Our state-level SWE counts, via duckdb.
    import duckdb
    con = duckdb.connect()
    rows_ark = con.execute(
        """
        SELECT state_normalized, COUNT(*) n FROM 'data/unified.parquet'
        WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true
          AND source='kaggle_arshkon' AND state_normalized IS NOT NULL
        GROUP BY 1
        """
    ).fetchall()
    rows_scr = con.execute(
        """
        SELECT state_normalized, COUNT(*) n FROM 'data/unified.parquet'
        WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true
          AND source='scraped' AND state_normalized IS NOT NULL
        GROUP BY 1
        """
    ).fetchall()
    ark = {s: n for s, n in rows_ark}
    scr = {s: n for s, n in rows_scr}

    # Sum 15-1252 + 15-1256 to approximate "software developers + QA" total.
    bls_total: dict[str, int] = {}
    for (state, soc_label), d in r.items():
        if d["value"] is None:
            continue
        bls_total[state] = bls_total.get(state, 0) + d["value"]
    # Also devs-only (15-1252) for a cleaner comparison.
    bls_devs: dict[str, int] = {}
    for (state, soc_label), d in r.items():
        if soc_label == "software_developers_15_1252" and d["value"] is not None:
            bls_devs[state] = d["value"]

    compare_path = OUT / "bls_vs_ours_state.csv"
    with compare_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "state", "bls_employment_devs", "bls_employment_devs_plus_qa",
            "ours_arshkon", "ours_scraped", "ours_pooled",
        ])
        states = sorted(set(bls_devs) | set(ark) | set(scr))
        for s in states:
            a = ark.get(s, 0)
            sc = scr.get(s, 0)
            w.writerow([
                s, bls_devs.get(s, ""), bls_total.get(s, ""), a, sc, a + sc,
            ])
    print(f"Wrote {compare_path}")

    # 3) Correlations. Use numpy.
    import numpy as np
    def corr(xs, ys):
        x = np.array(xs, dtype=float)
        y = np.array(ys, dtype=float)
        mask = (~np.isnan(x)) & (~np.isnan(y))
        x = x[mask]; y = y[mask]
        if len(x) < 3:
            return float("nan"), float("nan"), len(x)
        pr = float(np.corrcoef(x, y)[0, 1])
        # Spearman via ranks
        def rank(v):
            order = v.argsort()
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(v))
            return ranks
        sr = float(np.corrcoef(rank(x), rank(y))[0, 1])
        return pr, sr, len(x)

    shared = sorted(set(bls_devs) & (set(ark) | set(scr)))
    print(f"\nStates with BLS devs data: {len(bls_devs)}")
    print(f"States with any of ours:    {len(set(ark) | set(scr))}")
    print(f"Overlap:                    {len(shared)}")

    for name, ours in (("arshkon", ark), ("scraped", scr),
                       ("arshkon+scraped", {s: ark.get(s,0)+scr.get(s,0) for s in set(ark)|set(scr)})):
        xs, ys = [], []
        for s in shared:
            xs.append(bls_devs.get(s, np.nan))
            ys.append(ours.get(s, np.nan))
        pr, sr, n = corr(xs, ys)
        print(f"  BLS 15-1252 vs ours ({name}): Pearson r={pr:.4f} Spearman rho={sr:.4f} n={n}")
        # log-log too, since counts are right-skewed
        xs_log = [np.log1p(x) for x in xs]
        ys_log = [np.log1p(y) for y in ys]
        pr2, sr2, _ = corr(xs_log, ys_log)
        print(f"    log-log:               Pearson r={pr2:.4f} Spearman rho={sr2:.4f}")

    # 4) Scatter plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Linear scatter
        xs = [bls_devs.get(s, 0) for s in shared]
        ys_ark = [ark.get(s, 0) for s in shared]
        ys_scr = [scr.get(s, 0) for s in shared]
        ax = axes[0]
        ax.scatter(xs, ys_ark, s=20, label="arshkon", alpha=0.7)
        ax.scatter(xs, ys_scr, s=20, label="scraped", alpha=0.7, marker="x")
        ax.set_xlabel("BLS OES 15-1252 employment")
        ax.set_ylabel("Our SWE posting count")
        ax.set_title("BLS state employment vs our state posting counts")
        ax.legend()
        # Log-log
        ax2 = axes[1]
        ax2.scatter(xs, ys_ark, s=20, label="arshkon", alpha=0.7)
        ax2.scatter(xs, ys_scr, s=20, label="scraped", alpha=0.7, marker="x")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("BLS OES 15-1252 employment (log)")
        ax2.set_ylabel("Our SWE posting count (log)")
        ax2.set_title("Log-log")
        ax2.legend()
        plt.tight_layout()
        out_fig = FIG / "bls_vs_ours_state.png"
        plt.savefig(out_fig, dpi=150)
        print(f"Wrote {out_fig}")
    except Exception as e:
        print(f"plotting skipped: {e}")


if __name__ == "__main__":
    main()
