"""Smoke test section classifier on a few real postings from each source."""
import sys
sys.path.insert(0, 'exploration/scripts')
import duckdb
from T13_section_classifier import classify_sections, section_char_proportions, SECTIONS

con = duckdb.connect()
rows = con.execute("""
SELECT uid, source, description_core_llm
FROM 'data/unified.parquet'
WHERE is_swe = true AND source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'
  AND llm_extraction_coverage = 'labeled'
  AND description_core_llm IS NOT NULL
  AND LENGTH(description_core_llm) BETWEEN 800 AND 4000
ORDER BY hash(uid) % 997
LIMIT 10
""").df()

for _, r in rows.iterrows():
    print(f"=== {r['source']}  uid={r['uid']}  len={len(r['description_core_llm'])} ===")
    segs = classify_sections(r['description_core_llm'])
    for s in segs:
        head = s['text'][:80].replace('\n', ' / ')
        print(f"  [{s['section']:18s}] {s['char_len']:5d}c  |  {head}")
    total = len(r['description_core_llm'])
    prop = section_char_proportions(r['description_core_llm'])
    print(f"  proportions: " + ", ".join(f"{k}={v/total:.0%}" for k, v in prop.items() if v > 0))
    print()
