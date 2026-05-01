"""V1.3 Alt 3 — Audit T13 'responsibilities' classifications.

Spec: sample 30 responsibilities segments and check whether they are
actually responsibilities content vs misclassified requirements content.

If a large fraction of 'responsibilities' segments are actually
requirements content, the T13 "requirements flat, responsibilities +196"
finding would weaken.
"""

from __future__ import annotations

import random
import sys

import duckdb
import pandas as pd

sys.path.insert(0, "/home/jihgaboot/gabor/job-research/exploration/scripts")
from T13_section_classifier import classify_sections  # noqa: E402

UNI = "/home/jihgaboot/gabor/job-research/data/unified.parquet"
OUT = "/home/jihgaboot/gabor/job-research/exploration/tables/V1"

random.seed(777)


def main() -> None:
    con = duckdb.connect()
    q = f"""
    SELECT uid, source, period, description_core_llm
    FROM '{UNI}'
    WHERE is_swe = true
      AND source_platform = 'linkedin'
      AND is_english = true
      AND date_flag = 'ok'
      AND llm_extraction_coverage = 'labeled'
      AND description_core_llm IS NOT NULL
      AND source = 'scraped'
      AND LENGTH(description_core_llm) > 1000
    """
    df = con.execute(q).fetchdf().sample(n=1000, random_state=42)

    samples = []
    for _, r in df.iterrows():
        text = r["description_core_llm"] or ""
        segs = classify_sections(text)
        for s in segs:
            if s["section"] == "responsibilities" and s["char_len"] > 100:
                samples.append(
                    {
                        "uid": r["uid"],
                        "snippet": s["text"][:400].replace("\n", " "),
                    }
                )
                break
        if len(samples) >= 30:
            break

    out = pd.DataFrame(samples)
    out.to_csv(f"{OUT}/V1_t13_responsibilities_audit.csv", index=False)
    for i, r in enumerate(samples, 1):
        print(f"\n[{i}] {r['uid']}")
        print(f"    {r['snippet']}")


if __name__ == "__main__":
    main()
