"""Wave 1.5 Agent Prep — Step 4: company-name stoplist.

Build a stoplist of all tokens from `company_name_canonical` across ALL rows
(not just SWE) so downstream cross-source analyses share one stoplist.

Output: exploration/artifacts/shared/company_stoplist.txt
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import duckdb

PARQUET = "/home/jihgaboot/gabor/job-research/data/unified.parquet"
OUT = Path("/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/company_stoplist.txt")

# Tokenize on whitespace and common punctuation.
TOKEN_SPLIT = re.compile(r"[\s,./\-_&'()\[\]{}|+:;!?\"*<>]+")


def main() -> None:
    t0 = time.time()
    con = duckdb.connect()
    rows = con.execute(
        f"""
        SELECT DISTINCT company_name_canonical
        FROM '{PARQUET}'
        WHERE company_name_canonical IS NOT NULL
          AND company_name_canonical <> ''
        """
    ).fetchall()

    print(f"Distinct canonical companies: {len(rows):,}")

    tokens: set[str] = set()
    for (name,) in rows:
        if not name:
            continue
        for tok in TOKEN_SPLIT.split(name.lower()):
            tok = tok.strip()
            if not tok:
                continue
            # Keep only meaningful tokens (>=2 chars, not pure digits)
            if len(tok) < 2:
                continue
            if tok.isdigit():
                continue
            tokens.add(tok)

    # Inline assertions
    test = "Google LLC & Co., Inc."
    test_toks = [t for t in TOKEN_SPLIT.split(test.lower()) if t]
    assert "google" in test_toks, test_toks
    assert "llc" in test_toks, test_toks
    assert "inc" in test_toks, test_toks

    sorted_tokens = sorted(tokens)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(sorted_tokens) + "\n")
    print(f"Wrote {len(sorted_tokens):,} tokens to {OUT} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
