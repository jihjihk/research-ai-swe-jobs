"""T03 — Sample 100 LLM-routed SWE rows with weak seniority markers in title.

Uses a simpler query (no window function) to avoid the unicode bug in
DuckDB's window pipeline. Samples are balanced across source x marker_kind.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import duckdb

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data" / "unified.parquet"
OUT = ROOT / "exploration" / "tables" / "T03"

random.seed(42)

BASE_FILTER = (
    "is_english = TRUE AND date_flag = 'ok' AND source_platform = 'linkedin' "
    "AND is_swe = TRUE AND seniority_final_source = 'llm'"
)


def tag_marker(title_lower: str) -> str | None:
    t = title_lower
    if (
        "junior" in t or "entry level" in t or "entry-level" in t
        or " intern" in t or "intern " in t or "interns" in t
        or "jr." in t or " jr " in t or "new grad" in t
        or "new-grad" in t or "early career" in t or "early-career" in t
    ):
        return "junior_marker"
    if (
        "senior" in t or "staff " in t or " staff" in t
        or "principal" in t or "distinguished" in t
        or " lead " in t or t.startswith("lead ") or t.endswith(" lead")
        or "sr." in t or " sr " in t or "director" in t
    ):
        return "senior_marker"
    # Roman numeral markers as whole token
    tokens = t.split()
    romans = {"i", "ii", "iii", "iv"}
    if any(tok in romans for tok in tokens):
        return "roman_marker"
    return None


def clean_ascii(s: str | None) -> str:
    if s is None:
        return ""
    return "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in s)


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    # Pull just the columns we need; filter by marker presence via LIKE in SQL
    # so python doesn't have to iterate every row.
    sql = f"""
    SELECT source, period, title, title_normalized,
           seniority_final, seniority_native, yoe_extracted
    FROM '{DATA}'
    WHERE {BASE_FILTER}
      AND (
            lower(title) LIKE '%junior%'
         OR lower(title) LIKE '%jr.%'
         OR lower(title) LIKE '% jr %'
         OR lower(title) LIKE '%intern%'
         OR lower(title) LIKE '%entry level%'
         OR lower(title) LIKE '%entry-level%'
         OR lower(title) LIKE '%new grad%'
         OR lower(title) LIKE '%new-grad%'
         OR lower(title) LIKE '%early career%'
         OR lower(title) LIKE '%early-career%'
         OR lower(title) LIKE '%senior%'
         OR lower(title) LIKE '%staff%'
         OR lower(title) LIKE '%principal%'
         OR lower(title) LIKE '%distinguished%'
         OR lower(title) LIKE '% lead %'
         OR lower(title) LIKE 'lead %'
         OR lower(title) LIKE '% lead'
         OR lower(title) LIKE '%sr.%'
         OR lower(title) LIKE '% sr %'
         OR lower(title) LIKE '%director%'
         OR lower(title) LIKE '% i '
         OR lower(title) LIKE '% ii %'
         OR lower(title) LIKE '% ii'
         OR lower(title) LIKE '% iii %'
         OR lower(title) LIKE '% iii'
         OR lower(title) LIKE '% iv %'
         OR lower(title) LIKE '% iv'
      )
    """
    results = con.execute(sql).fetchall()
    print(f"candidate rows: {len(results)}")

    # Tag marker kind in python and bucket
    buckets: dict[tuple, list] = {}
    for row in results:
        source, period, title, title_norm, sfin, snat, yoe = row
        if title is None:
            continue
        title_clean = clean_ascii(title)
        t = title_clean.lower()
        marker = tag_marker(t)
        if marker is None:
            continue
        key = (source, marker)
        buckets.setdefault(key, []).append(
            (source, period, marker, title_clean, title_norm, sfin, snat, yoe)
        )

    print("bucket sizes:")
    for k, v in sorted(buckets.items()):
        print(f"  {k}: {len(v)}")

    # Sample up to ~15 per (source, marker) bucket for ~100 total
    SAMPLE_PER_BUCKET = 15
    sampled = []
    for key, items in sorted(buckets.items()):
        random.shuffle(items)
        sampled.extend(items[:SAMPLE_PER_BUCKET])

    print(f"total sampled: {len(sampled)}")

    out = OUT / "05_llm_routed_weak_marker_sample_v2.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "source", "period", "marker_kind", "title_clean", "title_normalized",
            "seniority_final", "seniority_native", "yoe_extracted",
        ])
        for row in sampled:
            w.writerow(row)
    print(f"wrote {out}")

    # Write a global per-bucket summary
    summary_rows = []
    for (source, marker), items in sorted(buckets.items()):
        from collections import Counter
        cnt = Counter(it[5] for it in items)  # seniority_final
        total = len(items)
        summary_rows.append({
            "source": source,
            "marker_kind": marker,
            "n": total,
            **{f"n_final_{k}": v for k, v in cnt.most_common()},
        })
    all_keys = set()
    for r in summary_rows:
        all_keys.update(r.keys())
    keys = ["source", "marker_kind", "n"] + sorted(k for k in all_keys if k.startswith("n_final_"))
    out2 = OUT / "05_llm_routed_marker_summary.csv"
    with out2.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
