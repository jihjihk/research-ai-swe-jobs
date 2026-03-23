#!/usr/bin/env python3
"""
Build a dataset-specific city/state -> metro reference lookup.

This script:
1. Parses the official 2023 Census CBSA delineation workbook.
2. Builds a county/state -> CBSA reference table.
3. Geocodes unique unresolved US city/state pairs with Nominatim.
4. Joins geocoded county/state results to the official CBSA table.
5. Writes a parquet lookup that Stage 6 can use offline.

The main pipeline stays network-free. Public geocoding is only used here
to build or expand the cached reference table.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests


PROJECT_ROOT = Path(__file__).parent.parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
REFERENCE_DIR = PROJECT_ROOT / "preprocessing" / "reference"
RAW_DIR = REFERENCE_DIR / "raw"

INPUT_PATH = INTERMEDIATE_DIR / "stage8_final.parquet"
CBSA_XLSX_PATH = RAW_DIR / "list1_2023.xlsx"
COUNTY_REFERENCE_PATH = REFERENCE_DIR / "metro_cbsa_county_reference.parquet"
LOOKUP_OUTPUT_PATH = REFERENCE_DIR / "metro_city_state_lookup.parquet"
OVERRIDES_PATH = REFERENCE_DIR / "metro_cbsa_label_overrides.json"
CACHE_PATH = INTERMEDIATE_DIR / "metro_city_state_reference_cache.json"

CENSUS_XLSX_URL = (
    "https://www2.census.gov/programs-surveys/metro-micro/geographies/"
    "reference-files/2023/delineation-files/list1_2023.xlsx"
)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "job-research-metro-reference/1.0"

STATE_NAME_TO_ABBR = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "district of columbia": "DC", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN",
    "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
    "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}


def normalize_key(value: str | None) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    text = text.replace("&", " and ")
    for token in [
        " city and borough of ", " municipality of ", " census area",
        " city and county of ", " borough", " county", " parish", " municipio",
    ]:
        text = text.replace(token, " ")
    text = re.sub(r"\bst[.]?\b", "saint", text)
    text = text.replace(".", " ")
    text = text.replace("'", "")
    text = text.replace("-", " ")
    return " ".join(text.split())


def ensure_census_workbook() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if CBSA_XLSX_PATH.exists():
        return
    response = requests.get(CENSUS_XLSX_URL, timeout=60)
    response.raise_for_status()
    CBSA_XLSX_PATH.write_bytes(response.content)


def iter_xlsx_rows(path: Path) -> list[dict[str, str]]:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        shared = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        shared_strings = [
            "".join(t.text or "" for t in si.iterfind(".//m:t", ns))
            for si in shared.findall("m:si", ns)
        ]
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    rows: list[list[str]] = []
    for row in sheet.findall(".//m:sheetData/m:row", ns):
        values: list[str] = []
        for cell in row.findall("m:c", ns):
            cell_type = cell.attrib.get("t")
            raw = cell.find("m:v", ns)
            if raw is None:
                values.append("")
            elif cell_type == "s":
                values.append(shared_strings[int(raw.text)])
            else:
                values.append(raw.text or "")
        rows.append(values)

    header = rows[2]
    records: list[dict[str, str]] = []
    for values in rows[3:]:
        padded = values + [""] * (len(header) - len(values))
        records.append(dict(zip(header, padded)))
    return records


def load_cbsa_label_overrides() -> dict[str, str]:
    if not OVERRIDES_PATH.exists():
        return {}
    raw = json.loads(OVERRIDES_PATH.read_text())
    return {normalize_key(k): v for k, v in raw.items()}


def build_metro_area_label(cbsa_title: str, overrides: dict[str, str]) -> str:
    override = overrides.get(normalize_key(cbsa_title))
    if override is not None:
        return override

    city_part = cbsa_title.split(",", 1)[0].strip()
    if city_part.endswith(" Metro"):
        return city_part
    return f"{city_part} Metro"


def build_county_reference() -> pd.DataFrame:
    ensure_census_workbook()
    records = iter_xlsx_rows(CBSA_XLSX_PATH)
    overrides = load_cbsa_label_overrides()

    rows: list[dict[str, str]] = []
    for record in records:
        if record["Metropolitan/Micropolitan Statistical Area"] != "Metropolitan Statistical Area":
            continue

        state_name = record["State Name"].strip()
        state_abbr = STATE_NAME_TO_ABBR.get(state_name.lower())
        if state_abbr is None:
            continue

        county_name = record["County/County Equivalent"].strip()
        rows.append(
            {
                "cbsa_code": str(record["CBSA Code"]).strip(),
                "cbsa_title": record["CBSA Title"].strip(),
                "metro_area": build_metro_area_label(record["CBSA Title"].strip(), overrides),
                "state_name": state_name,
                "state_abbr": state_abbr,
                "county_name": county_name,
                "county_key": normalize_key(county_name),
            }
        )

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["cbsa_code", "state_abbr", "county_key"]
    ).reset_index(drop=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, COUNTY_REFERENCE_PATH)
    return df


def unique_cbsa_reference(county_reference: pd.DataFrame) -> pd.DataFrame:
    ref = county_reference[
        ["cbsa_code", "cbsa_title", "metro_area"]
    ].drop_duplicates().reset_index(drop=True)
    ref["city_part_key"] = ref["cbsa_title"].str.split(",", n=1).str[0].map(normalize_key)
    ref["state_codes"] = ref["cbsa_title"].map(extract_title_states)
    return ref


def extract_title_states(cbsa_title: str) -> list[str]:
    if "," not in cbsa_title:
        return []
    state_part = cbsa_title.rsplit(",", 1)[1].strip()
    return [piece.strip().upper() for piece in state_part.split("-") if piece.strip()]


def title_city_match(query_city: str, query_state: str, cbsa_reference: pd.DataFrame) -> dict[str, str] | None:
    city_key = normalize_key(query_city)
    if not city_key:
        return None

    matches: list[dict[str, str]] = []
    needle = f" {city_key} "
    for row in cbsa_reference.itertuples(index=False):
        if query_state not in row.state_codes:
            continue
        haystack = f" {row.city_part_key} "
        if needle in haystack:
            matches.append(
                {
                    "cbsa_code": row.cbsa_code,
                    "cbsa_title": row.cbsa_title,
                    "metro_area": row.metro_area,
                }
            )

    if len(matches) == 1:
        return matches[0]
    return None


def load_cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text())


def save_cache(cache: dict[str, dict]) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def geocode_location(query: str, cache: dict[str, dict], sleep_seconds: float) -> dict:
    if query in cache:
        return cache[query]

    params = {
        "q": f"{query}, USA",
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 1,
        "countrycodes": "us",
    }
    payload = None
    for attempt in range(4):
        response = requests.get(
            NOMINATIM_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=60,
        )
        if response.status_code == 429:
            time.sleep(max(sleep_seconds, 1.0) * (attempt + 1))
            continue
        response.raise_for_status()
        payload = response.json()
        break
    if payload is None:
        raise RuntimeError(f"Nominatim rate-limited query after retries: {query}")

    if not payload:
        result = {"status": "no_match"}
    else:
        hit = payload[0]
        address = hit.get("address", {})
        result = {
            "status": "ok",
            "display_name": hit.get("display_name"),
            "lat": hit.get("lat"),
            "lon": hit.get("lon"),
            "type": hit.get("type"),
            "addresstype": hit.get("addresstype"),
            "state_name": address.get("state"),
            "county_name": address.get("county"),
        }

    cache[query] = result
    save_cache(cache)
    time.sleep(sleep_seconds)
    return result


def existing_lookup_keys(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    table = pq.read_table(output_path, columns=["metro_lookup_key"])
    df = table.to_pandas()
    return {
        str(value)
        for value in df["metro_lookup_key"].dropna().tolist()
        if str(value).strip()
    }


def unresolved_city_state_candidates(
    input_path: Path,
    limit: int | None,
    offset: int,
    excluded_keys: set[str],
) -> pd.DataFrame:
    con = duckdb.connect()
    limit_sql = f"limit {limit}" if limit is not None else ""
    excluded_sql = ""
    if excluded_keys:
        escaped = "', '".join(key.replace("'", "''") for key in sorted(excluded_keys))
        excluded_sql = (
            "and trim(regexp_replace(lower(city_extracted), '[^a-zA-Z0-9]+', ' ', 'g')) "
            f"|| '|' || upper(state_normalized) not in ('{escaped}')"
        )
    query = f"""
        select
            city_extracted,
            state_normalized,
            count(*) as row_count
        from read_parquet(?)
        where metro_source = 'unresolved'
          and coalesce(is_remote, false) = false
          and coalesce(is_remote_inferred, false) = false
          and country_extracted = 'US'
          and city_extracted is not null
          and state_normalized is not null
          {excluded_sql}
        group by 1, 2
        order by row_count desc, city_extracted, state_normalized
        offset {offset}
        {limit_sql}
    """
    return con.execute(query, [str(input_path)]).fetchdf()


def build_city_state_reference(
    input_path: Path,
    county_reference: pd.DataFrame,
    limit: int | None,
    offset: int,
    sleep_seconds: float,
    existing_keys: set[str],
) -> pd.DataFrame:
    cache = load_cache()
    candidates = unresolved_city_state_candidates(input_path, limit, offset, existing_keys)
    cbsa_reference = unique_cbsa_reference(county_reference)

    rows: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        query = f"{row.city_extracted}, {row.state_normalized}"
        direct_match = title_city_match(row.city_extracted, row.state_normalized, cbsa_reference)
        if direct_match is not None:
            rows.append(
                {
                    "city_extracted": row.city_extracted,
                    "state_normalized": row.state_normalized,
                    "row_count": int(row.row_count),
                    "query": query,
                    "geocode_status": "skipped_title_match",
                    "geocode_display_name": None,
                    "geocode_type": None,
                    "geocode_addresstype": None,
                    "geocode_state_name": None,
                    "geocode_state_abbr": row.state_normalized,
                    "geocode_county_name": None,
                    "county_key": "",
                    "cbsa_code": direct_match["cbsa_code"],
                    "cbsa_title": direct_match["cbsa_title"],
                    "metro_area": direct_match["metro_area"],
                    "state_name": None,
                    "state_abbr": row.state_normalized,
                    "county_name": None,
                    "match_method": "title_fallback",
                }
            )
            continue

        geocoded = geocode_location(query, cache, sleep_seconds=sleep_seconds)
        county_key = normalize_key(geocoded.get("county_name"))
        state_name = geocoded.get("state_name")
        state_abbr = STATE_NAME_TO_ABBR.get(str(state_name).lower()) if state_name else None

        rows.append(
            {
                "city_extracted": row.city_extracted,
                "state_normalized": row.state_normalized,
                "row_count": int(row.row_count),
                "query": query,
                "geocode_status": geocoded.get("status"),
                "geocode_display_name": geocoded.get("display_name"),
                "geocode_type": geocoded.get("type"),
                "geocode_addresstype": geocoded.get("addresstype"),
                "geocode_state_name": state_name,
                "geocode_state_abbr": state_abbr,
                "geocode_county_name": geocoded.get("county_name"),
                "county_key": county_key,
                "match_method": None,
            }
        )

    geocoded_df = pd.DataFrame(rows)
    geocoded_mask = geocoded_df["cbsa_code"].isna()
    geocoded_rows = geocoded_df.loc[geocoded_mask].drop(
        columns=["cbsa_code", "cbsa_title", "metro_area"],
        errors="ignore",
    )
    merged = geocoded_rows.merge(
        county_reference,
        how="left",
        left_on=["geocode_state_abbr", "county_key"],
        right_on=["state_abbr", "county_key"],
    )
    merged["match_method"] = merged["cbsa_code"].notna().map({True: "county_join", False: None})
    unmatched_mask = merged["cbsa_code"].isna()
    if unmatched_mask.any():
        fallback_rows = merged.loc[unmatched_mask, ["city_extracted", "state_normalized"]]
        fallback_matches = []
        for row in fallback_rows.itertuples(index=False):
            fallback = title_city_match(row.city_extracted, row.state_normalized, cbsa_reference)
            fallback_matches.append(fallback or {})
        fallback_df = pd.DataFrame(fallback_matches, index=merged.index[unmatched_mask])
        for column in ["cbsa_code", "cbsa_title", "metro_area"]:
            if column in fallback_df.columns:
                merged[column] = merged[column].where(merged[column].notna(), fallback_df[column])
        merged.loc[unmatched_mask & merged["cbsa_code"].notna(), "match_method"] = "title_fallback"

    merged["match_confidence"] = "low"
    merged.loc[merged["match_method"] == "county_join", "match_confidence"] = "high"
    merged.loc[merged["match_method"] == "title_fallback", "match_confidence"] = "medium"
    direct = geocoded_df.loc[~geocoded_mask].copy()
    if not direct.empty:
        direct["match_confidence"] = "medium"
    combined = pd.concat([merged, direct], ignore_index=True, sort=False)
    combined["metro_lookup_key"] = (
        combined["city_extracted"].str.strip().str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True)
        + "|"
        + combined["state_normalized"].str.upper()
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=LOOKUP_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    args = parser.parse_args()

    county_reference = build_county_reference()
    existing_keys = existing_lookup_keys(args.output_path)
    lookup = build_city_state_reference(
        input_path=args.input_path,
        county_reference=county_reference,
        limit=None if args.limit <= 0 else args.limit,
        offset=max(args.offset, 0),
        sleep_seconds=args.sleep_seconds,
        existing_keys=existing_keys,
    )

    if args.output_path.exists():
        existing = pq.read_table(args.output_path).to_pandas()
        existing = existing.loc[existing["metro_lookup_key"].notna()].copy()
        lookup = pd.concat([existing, lookup], ignore_index=True)
        lookup = lookup.drop_duplicates(subset=["metro_lookup_key"], keep="first")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(lookup, preserve_index=False), args.output_path)

    matched = int(lookup["cbsa_code"].notna().sum())
    total = len(lookup)
    print(f"Wrote county reference: {COUNTY_REFERENCE_PATH}")
    print(f"Wrote city/state lookup: {args.output_path}")
    print(f"Matched {matched}/{total} candidate city/state pairs")
    if total:
        print(
            lookup[
                [
                    "city_extracted",
                    "state_normalized",
                    "row_count",
                    "metro_area",
                    "cbsa_title",
                    "geocode_county_name",
                    "geocode_display_name",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
