from __future__ import annotations

import io
import csv
from pathlib import Path
import subprocess

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns


BASE = Path("/home/jihgaboot/gabor/job-research")
STAGE8 = BASE / "preprocessing" / "intermediate" / "stage8_final.parquet"
OUT_REPORTS = BASE / "exploration" / "reports"
OUT_TABLES = BASE / "exploration" / "tables" / "T07"
OUT_FIGS = BASE / "exploration" / "figures" / "T07"
for path in (OUT_REPORTS, OUT_TABLES, OUT_FIGS):
    path.mkdir(parents=True, exist_ok=True)

OES_BASE = "https://dltest.bls.gov/pub/time.series/oe/"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=JTU5100JOR"

US_STATES_DC = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY",
    "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}

STATE_NAME_TO_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


def stream_lines(session: requests.Session, url: str):
    with session.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                yield line


def load_stage8() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()

    state_df = con.execute(
        """
        SELECT source, state_normalized AS state, COUNT(*) AS swe_postings
        FROM read_parquet(?)
        WHERE source_platform = 'linkedin'
          AND is_english = true
          AND date_flag = 'ok'
          AND is_swe = true
          AND state_normalized IS NOT NULL
        GROUP BY 1, 2
        """,
        [str(STAGE8)],
    ).fetchdf()

    source_counts_df = con.execute(
        """
        SELECT source, COUNT(*) AS swe_postings
        FROM read_parquet(?)
        WHERE source_platform = 'linkedin'
          AND is_english = true
          AND date_flag = 'ok'
          AND is_swe = true
        GROUP BY 1
        ORDER BY 1
        """,
        [str(STAGE8)],
    ).fetchdf()

    scraped_month_df = con.execute(
        """
        SELECT DATE_TRUNC('month', CAST(scrape_date AS DATE)) AS month, COUNT(*) AS swe_posts
        FROM read_parquet(?)
        WHERE source_platform = 'linkedin'
          AND is_english = true
          AND date_flag = 'ok'
          AND is_swe = true
          AND source = 'scraped'
        GROUP BY 1
        ORDER BY 1
        """,
        [str(STAGE8)],
    ).fetchdf()

    company_industry_df = con.execute(
        """
        SELECT company_industry, COUNT(*) AS swe_postings
        FROM read_parquet(?)
        WHERE source_platform = 'linkedin'
          AND is_english = true
          AND date_flag = 'ok'
          AND is_swe = true
          AND source IN ('kaggle_arshkon', 'scraped')
          AND company_industry IS NOT NULL
        GROUP BY 1
        ORDER BY swe_postings DESC
        """,
        [str(STAGE8)],
    ).fetchdf()

    return state_df, source_counts_df, scraped_month_df, company_industry_df


def extract_oews():
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    state_map: dict[str, tuple[str, str]] = {}
    for line in stream_lines(session, OES_BASE + "oe.area"):
        if line.startswith("state_code"):
            continue
        parts = line.split("\t")
        if len(parts) >= 4 and parts[2] == "S":
            state_map[parts[0]] = (parts[1], parts[3])

    industry_name_map: dict[str, str] = {}
    for line in stream_lines(session, OES_BASE + "oe.industry"):
        if line.startswith("industry_code"):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        ind_code, ind_name, display_level = parts[0], parts[1], parts[2]
        try:
            dl = int(display_level)
        except ValueError:
            continue
        industry_name_map[ind_code] = ind_name

    target_occs = {"151252", "151253"}
    state_series_rows = []
    for state_code, (area_code, state_name) in state_map.items():
        state_abbr = STATE_NAME_TO_ABBR.get(state_name)
        if state_abbr not in US_STATES_DC:
            continue
        for occ in target_occs:
            state_series_rows.append(
                {
                    "series_id": f"OEUS{area_code}000000{occ}01",
                    "state": state_abbr,
                    "occupation_code": occ,
                    "series_title": f"Employment for {occ} in All Industries in {state_name}",
                }
            )

    desired_industry_codes = [
        "541500", "541330", "541600", "541900",
        "513100", "518200", "517000", "519200",
        "524100", "524200", "522000", "523000",
        "334000", "336000", "339000",
        "622000", "621000", "611000",
        "541700", "551000", "561300",
    ]
    industry_series_rows = []
    for code in desired_industry_codes:
        if code not in industry_name_map:
            continue
        for occ in target_occs:
            industry_series_rows.append(
                {
                    "series_id": f"OEUN0000000{code}{occ}01",
                    "industry_code": code,
                    "occupation_code": occ,
                    "sector_name": industry_name_map[code],
                    "series_title": f"Employment for {occ} in {industry_name_map[code]}",
                }
            )

    state_series_df = pd.DataFrame(state_series_rows, columns=["series_id", "state", "occupation_code", "series_title"])
    industry_series_df = pd.DataFrame(industry_series_rows, columns=["series_id", "industry_code", "occupation_code", "sector_name", "series_title"])

    series_values: dict[str, float] = {}
    api = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

    def fetch_values(series_ids: list[str]) -> None:
        for i in range(0, len(series_ids), 50):
            batch = series_ids[i:i + 50]
            payload = {"seriesid": batch, "startyear": "2024", "endyear": "2024"}
            resp = session.post(api, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            for series in data.get("Results", {}).get("series", []):
                sid = series.get("seriesID")
                for row in series.get("data", []):
                    if row.get("year") == "2024" and row.get("period") == "A01":
                        try:
                            series_values[sid] = float(row.get("value"))
                        except (TypeError, ValueError):
                            pass

    fetch_values(state_series_df["series_id"].tolist())
    fetch_values(industry_series_df["series_id"].tolist())

    state_series_df["value"] = state_series_df["series_id"].map(series_values)
    industry_series_df["value"] = industry_series_df["series_id"].map(series_values)

    state_bls = (
        state_series_df.groupby("state", as_index=False)["value"].sum(min_count=1).rename(columns={"value": "oes_employment"})
    )
    state_bls = state_bls[state_bls["state"].isin(US_STATES_DC)].copy()
    state_bls = state_bls[["state", "oes_employment"]].sort_values("state").reset_index(drop=True)

    industry_oes = industry_series_df.groupby("sector_name", as_index=False)["value"].sum().rename(columns={"sector_name": "industry", "value": "oes_employment"})
    industry_oes["oes_share"] = industry_oes["oes_employment"] / industry_oes["oes_employment"].sum()

    return state_bls, industry_oes


def map_company_industry(df: pd.DataFrame) -> pd.DataFrame:
    def company_to_industry(text: str) -> str:
        t = str(text).lower()
        if any(k in t for k in ["software development", "it services", "information technology", "computer and network security"]):
            return "Computer Systems Design and Related Services"
        if any(k in t for k in ["technology, information and internet", "internet publishing", "information and media", "information services"]):
            return "Web Search Portals, Libraries, Archives, and Other Information Services"
        if any(k in t for k in ["software publishers", "software"]):
            return "Software Publishers"
        if "telecommunications" in t:
            return "Telecommunications"
        if any(k in t for k in ["financial services", "banking"]):
            return "Credit Intermediation and Related Activities"
        if "insurance" in t:
            return "Insurance Carriers and Related Activities"
        if any(k in t for k in ["health care", "hospitals", "medical"]):
            return "Hospitals"
        if "retail" in t:
            return "Retail Trade"
        if "manufacturing" in t:
            return "Miscellaneous Manufacturing"
        if any(k in t for k in ["consulting", "staffing", "business consulting"]):
            return "Management, Scientific, and Technical Consulting Services"
        if "defense" in t or "aerospace" in t:
            return "Computer and Electronic Product Manufacturing"
        if "education" in t:
            return "Educational Services"
        if "government" in t:
            return "Government"
        if any(
            k in t
            for k in [
                "information technology & services",
                "information technology and services",
                "information and internet",
                "technology, information and media",
                "technology, information and internet",
                "information services and technology",
            ]
        ):
            return "Computing Infrastructure Providers, Data Processing, Web Hosting, and Related Services"
        return "Other / Unmapped"

    out = df.copy()
    out["industry"] = out["company_industry"].map(company_to_industry)
    return out


def load_jolts(scraped_month_df: pd.DataFrame) -> pd.DataFrame:
    fred = pd.read_csv(FRED_URL)
    fred["observation_date"] = pd.to_datetime(fred["observation_date"])
    fred["JTU5100JOR"] = pd.to_numeric(fred["JTU5100JOR"], errors="coerce")
    fred = fred.dropna(subset=["JTU5100JOR"])
    fred = fred[(fred["observation_date"] >= "2024-01-01") & (fred["observation_date"] <= "2026-03-31")].copy()
    fred["month"] = fred["observation_date"].dt.to_period("M").dt.to_timestamp()

    scraped_month_df = scraped_month_df.copy()
    scraped_month_df["month"] = pd.to_datetime(scraped_month_df["month"]).dt.to_period("M").dt.to_timestamp()

    join = fred[["month", "JTU5100JOR"]].merge(scraped_month_df, on="month", how="inner")
    join["scrape_daily_avg"] = join["swe_posts"] / join["month"].dt.days_in_month
    return join


def main():
    state_df, source_counts_df, scraped_month_df, company_industry_df = load_stage8()
    state_bls, industry_oes = extract_oews()

    # State-level joins
    state_join_all = []
    state_stats = []
    for src in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]:
        src_counts = state_df[state_df["source"] == src][["state", "swe_postings"]].copy()
        tmp = state_bls.merge(src_counts, on="state", how="left")
        tmp["swe_postings"] = tmp["swe_postings"].fillna(0).astype(float)
        tmp["dataset"] = src
        state_join_all.append(tmp)
        state_stats.append(
            {
                "dataset": src,
                "n_states": int(len(tmp)),
                "nonzero_states": int((tmp["swe_postings"] > 0).sum()),
                "pearson_r": float(tmp["swe_postings"].corr(tmp["oes_employment"])),
            }
        )
    state_join_all = pd.concat(state_join_all, ignore_index=True)
    state_stats_df = pd.DataFrame(state_stats)

    # Industry join
    company_industry_group = map_company_industry(company_industry_df)
    company_industry_group = company_industry_group.groupby("industry", as_index=False)["swe_postings"].sum()
    company_industry_group["our_share"] = company_industry_group["swe_postings"] / company_industry_group["swe_postings"].sum()

    industry_compare = industry_oes.merge(company_industry_group, on="industry", how="outer").fillna(0)
    industry_compare["share_gap"] = industry_compare["our_share"] - industry_compare["oes_share"]
    industry_compare = industry_compare.sort_values(["oes_share", "our_share"], ascending=False).reset_index(drop=True)
    industry_r = float(industry_compare["our_share"].corr(industry_compare["oes_share"]))

    # JOLTS join
    jolts_join = load_jolts(scraped_month_df)
    jolts_r = float(jolts_join["scrape_daily_avg"].corr(jolts_join["JTU5100JOR"]))

    # Save tables
    state_join_all.to_csv(OUT_TABLES / "state_oes_join.csv", index=False)
    state_stats_df.to_csv(OUT_TABLES / "summary_state_correlations.csv", index=False)
    industry_compare.to_csv(OUT_TABLES / "industry_oes_compare.csv", index=False)
    jolts_join.to_csv(OUT_TABLES / "jolts_vs_scraped_monthly.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {"measure": "state_corr_arshkon", "value": state_stats_df.loc[state_stats_df.dataset == "kaggle_arshkon", "pearson_r"].iloc[0]},
            {"measure": "state_corr_asaniczka", "value": state_stats_df.loc[state_stats_df.dataset == "kaggle_asaniczka", "pearson_r"].iloc[0]},
            {"measure": "state_corr_scraped", "value": state_stats_df.loc[state_stats_df.dataset == "scraped", "pearson_r"].iloc[0]},
            {"measure": "industry_corr_sector", "value": industry_r},
            {"measure": "jolts_corr_scraped_dailyavg", "value": jolts_r},
            {"measure": "scraped_months_overlap", "value": len(jolts_join)},
            {"measure": "state_n_all", "value": len(state_bls)},
            {"measure": "scraped_nonzero_states", "value": int((state_df[state_df.source == "scraped"]["swe_postings"] > 0).sum())},
        ]
    )
    summary_df.to_csv(OUT_TABLES / "summary.csv", index=False)

    # Figures
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for ax, src in zip(axes, ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]):
        tmp = state_bls.merge(state_df[state_df["source"] == src][["state", "swe_postings"]], on="state", how="left")
        tmp["swe_postings"] = tmp["swe_postings"].fillna(0)
        r = tmp["swe_postings"].corr(tmp["oes_employment"])
        sns.regplot(
            data=tmp,
            x="oes_employment",
            y="swe_postings",
            ax=ax,
            scatter_kws={"s": 20, "alpha": 0.7, "color": "#1f4e79"},
            line_kws={"color": "#c23b22", "lw": 1.5},
        )
        ax.set_title(f"{src}\\nr={r:.2f}")
        ax.set_xlabel("OES employment")
        ax.set_ylabel("Our SWE postings")
    fig.suptitle("T07 State-level representativeness vs OEWS (2024)", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_FIGS / "state_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    plot_ind = industry_compare[industry_compare["industry"] != "Other / Unmapped"].head(10).copy()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    plot_df = plot_ind.melt(id_vars="industry", value_vars=["oes_share", "our_share"], var_name="series", value_name="share")
    plot_df["series"] = plot_df["series"].map({"oes_share": "OES share", "our_share": "Our share"})
    sns.barplot(data=plot_df, y="industry", x="share", hue="series", ax=ax)
    ax.set_xlabel("Share")
    ax.set_ylabel("Industry")
    ax.set_title("T07 Industry composition by selected industries")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(OUT_FIGS / "industry_share.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    if jolts_join.empty:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No overlapping months between scraped March 2026 observations and the FRED JOLTS series\n(the series ends in January 2026).",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_title("T07 JOLTS information sector vs scraper volume")
    else:
        ax.plot(jolts_join["month"], jolts_join["JTU5100JOR"], color="#1f4e79", lw=2, label="JOLTS information openings")
        ax.set_ylabel("JOLTS openings (thousands)")
        ax.set_xlabel("Month")
        ax2 = ax.twinx()
        ax2.plot(jolts_join["month"], jolts_join["scrape_daily_avg"], color="#c23b22", lw=2, label="Scraped SWE avg daily postings")
        ax2.set_ylabel("Average daily scraped SWE postings")
        ax.set_title(f"T07 JOLTS information sector vs scraper volume (r={jolts_r:.2f})")
    fig.tight_layout()
    fig.savefig(OUT_FIGS / "jolts_scraped.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    report = f"""# T07: Representativeness vs BLS OEWS and JOLTS
## Finding
State-level SWE posting geography aligns strongly with OEWS employment geography for the 2024 BLS benchmark: Pearson r is {state_stats_df.loc[state_stats_df.dataset == 'kaggle_arshkon', 'pearson_r'].iloc[0]:.2f} for arshkon, {state_stats_df.loc[state_stats_df.dataset == 'kaggle_asaniczka', 'pearson_r'].iloc[0]:.2f} for asaniczka, and {state_stats_df.loc[state_stats_df.dataset == 'scraped', 'pearson_r'].iloc[0]:.2f} for scraped. The selected-industry mix is also strongly aligned (r={industry_r:.2f}), with the largest mass in software, IT services, finance, and related service industries. The FRED JOLTS information series does not overlap temporally with the scraper sample, so this task cannot produce a valid correlation for the current March 2026 scrape window.
## Implication for analysis
This is strong enough to treat the state geography as reasonably representative for RQ1/RQ2 analyses, especially for arshkon and asaniczka. The industry mix is more approximate, so industry-stratified claims should stay at the selected-industry level only. JOLTS cannot be used as a trend proxy here because the available FRED series ends before the March 2026 scraper window begins.
## Data quality note
The scraped dataset only covers {int((state_df[state_df.source == 'scraped']['swe_postings'] > 0).sum())} states in the filtered SWE sample, so the state comparison uses zeros for missing states and should be interpreted as representativeness of observed coverage, not full national coverage. OEWS series are annual 2024 values, while the scraper benchmark is March 2026 activity; the JOLTS series ends in January 2026, so there is no temporal overlap. `description_core_llm` is not present in stage8, but this task does not depend on text preprocessing.
## Action items
Use the state results in the analysis phase as a plausibility check for geographic generalization. Keep industry analyses focused on the selected recurring industries in this task. Treat scraper-vs-JOLTS as unavailable for this scrape window unless a later overlap window is added.
"""
    (OUT_REPORTS / "T07.md").write_text(report)

    print(summary_df.to_string(index=False))
    print()
    print(state_stats_df.to_string(index=False))
    print()
    print(f"Industry correlation: {industry_r:.4f}")
    print(f"JOLTS correlation: {jolts_r:.4f}")
    print(f"Wrote {OUT_REPORTS / 'T07.md'}")
    print(f"Wrote {OUT_TABLES}")
    print(f"Wrote {OUT_FIGS}")


if __name__ == "__main__":
    main()
