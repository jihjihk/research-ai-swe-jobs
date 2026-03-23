#!/usr/bin/env python3
"""
Audit unresolved metro locations with small-sample public geocoding calls.

This script is intentionally separate from Stage 6. The main preprocessing
pipeline stays deterministic and offline. Public API calls here are capped,
cached, and used only to inspect unresolved location patterns so we can improve
the local metro lookup over time.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

from stage678_normalize_temporal_flags import (
    CHUNK_SIZE,
    INPUT_PATH,
    INTERMEDIATE_DIR,
    build_scraped_city_state_metro_lookup,
    infer_metro,
    load_metro_aliases,
    normalize_location,
)

CACHE_PATH = INTERMEDIATE_DIR / "metro_geocode_cache.json"
OUTPUT_PATH = INTERMEDIATE_DIR / "metro_fallthrough_audit.parquet"
USER_AGENT = "job-research-metro-audit/1.0"


def load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text())


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def fetch_json(url: str, headers: dict[str, str] | None = None) -> dict | list | None:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload)


def geocode_census(location: str) -> dict:
    encoded = urllib.parse.quote(location)
    url = (
        "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
        f"?address={encoded}&benchmark=Public_AR_Current&format=json"
    )
    try:
        data = fetch_json(url)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    matches = data.get("result", {}).get("addressMatches", []) if isinstance(data, dict) else []
    if not matches:
        return {"status": "no_match"}

    top = matches[0]
    coords = top.get("coordinates") or {}
    return {
        "status": "ok",
        "matched_address": top.get("matchedAddress"),
        "match_type": top.get("tigerLine", {}).get("side"),
        "lat": coords.get("y"),
        "lon": coords.get("x"),
    }


def geocode_nominatim(location: str) -> dict:
    encoded = urllib.parse.quote(location)
    url = (
        "https://nominatim.openstreetmap.org/search"
        f"?q={encoded}&format=jsonv2&limit=1&countrycodes=us"
    )
    headers = {"User-Agent": USER_AGENT}
    try:
        data = fetch_json(url, headers=headers)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    if not isinstance(data, list) or not data:
        return {"status": "no_match"}

    top = data[0]
    return {
        "status": "ok",
        "display_name": top.get("display_name"),
        "lat": top.get("lat"),
        "lon": top.get("lon"),
        "type": top.get("type"),
        "class": top.get("class"),
        "importance": top.get("importance"),
    }


def collect_unresolved_locations(limit: int) -> list[dict]:
    import pyarrow.parquet as pq

    metro_aliases = load_metro_aliases()
    city_state_lookup, _ = build_scraped_city_state_metro_lookup(INPUT_PATH, metro_aliases)
    pf = pq.ParquetFile(INPUT_PATH)

    counts: Counter = Counter()
    examples: dict[str, dict] = {}

    for batch in pf.iter_batches(
        batch_size=CHUNK_SIZE,
        columns=["source", "location", "search_metro_name", "is_remote"],
    ):
        chunk = batch.to_pandas()
        for row in chunk.itertuples(index=False):
            loc = normalize_location(row.location)
            is_remote_inferred = bool(loc["is_remote_location"])
            metro = infer_metro(
                row.source,
                row.location,
                row.search_metro_name,
                row.is_remote,
                is_remote_inferred,
                loc["city_extracted"],
                loc["state_normalized"],
                metro_aliases,
                city_state_lookup,
            )
            if metro["metro_area"] is not None:
                continue
            if bool(row.is_remote) or is_remote_inferred:
                continue
            if loc["country_extracted"] not in {"US", None}:
                continue
            if row.location is None or str(row.location).strip() == "":
                continue

            key = str(row.location).strip()
            counts[key] += 1
            examples.setdefault(
                key,
                {
                    "source": row.source,
                    "location": key,
                    "city_extracted": loc["city_extracted"],
                    "state_normalized": loc["state_normalized"],
                    "country_extracted": loc["country_extracted"],
                },
            )

    rows = []
    for location, n in counts.most_common(limit):
        row = examples[location].copy()
        row["rows"] = n
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit unresolved metro locations.")
    parser.add_argument("--limit", type=int, default=25, help="Number of unresolved unique locations to geocode")
    parser.add_argument("--sleep-seconds", type=float, default=1.1, help="Delay between Nominatim requests")
    args = parser.parse_args()

    unresolved = collect_unresolved_locations(args.limit)
    cache = load_cache()
    output_rows = []

    for idx, row in enumerate(unresolved, start=1):
        location = row["location"]
        if location not in cache:
            cache[location] = {
                "census": geocode_census(location),
            }
            time.sleep(args.sleep_seconds)
            cache[location]["nominatim"] = geocode_nominatim(location)
            save_cache(cache)

        cached = cache[location]
        out = row.copy()
        out["audit_rank"] = idx
        out["census_status"] = cached.get("census", {}).get("status")
        out["census_matched_address"] = cached.get("census", {}).get("matched_address")
        out["census_lat"] = cached.get("census", {}).get("lat")
        out["census_lon"] = cached.get("census", {}).get("lon")
        out["nominatim_status"] = cached.get("nominatim", {}).get("status")
        out["nominatim_display_name"] = cached.get("nominatim", {}).get("display_name")
        out["nominatim_type"] = cached.get("nominatim", {}).get("type")
        out["nominatim_lat"] = cached.get("nominatim", {}).get("lat")
        out["nominatim_lon"] = cached.get("nominatim", {}).get("lon")
        output_rows.append(out)

    df = pd.DataFrame(output_rows)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df):,} audit rows to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
