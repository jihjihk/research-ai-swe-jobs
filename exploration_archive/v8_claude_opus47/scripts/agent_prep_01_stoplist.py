"""Agent Prep step 1: build company name stoplist.

Extract all unique tokens from company_name_canonical across the full parquet,
tokenize on whitespace + common punctuation, lowercase, dedupe, write one token
per line to exploration/artifacts/shared/company_stoplist.txt.
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb

OUT_PATH = Path("/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/company_stoplist.txt")
SPLIT_RE = re.compile(r"[\s,.\-_&()\[\]/'\"|]+")


def main() -> None:
    con = duckdb.connect()
    # Full parquet — all rows, not just SWE — because any posting may reference
    # any company name.
    rows = con.execute(
        """
        SELECT DISTINCT company_name_canonical
        FROM read_parquet('/home/jihgaboot/gabor/job-research/data/unified.parquet')
        WHERE company_name_canonical IS NOT NULL
        """
    ).fetchall()
    print(f"Distinct canonical companies: {len(rows):,}")

    tokens: set[str] = set()
    for (name,) in rows:
        if not name:
            continue
        for tok in SPLIT_RE.split(name.lower()):
            tok = tok.strip()
            if tok:
                tokens.add(tok)

    # Drop a handful of 1-char tokens that are obviously noise but keep single
    # letters that might appear as company words (e.g. 'a', 'i'). Keep
    # everything for simplicity; Wave 2 tokenizers can apply their own min-len.
    sorted_tokens = sorted(tokens)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for tok in sorted_tokens:
            f.write(tok + "\n")
    print(f"Wrote {len(sorted_tokens):,} tokens to {OUT_PATH}")


if __name__ == "__main__":
    main()
