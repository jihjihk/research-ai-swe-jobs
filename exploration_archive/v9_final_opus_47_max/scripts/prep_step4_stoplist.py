"""Wave 1.5 Agent Prep - Step 4: Company stoplist

Tokenize all unique company_name_canonical values, lowercase, dedupe.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import duckdb

OUT_DIR = Path("exploration/artifacts/shared")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "company_stoplist.txt"

# Generic tokens we filter out — including them strips too much from job text.
GENERIC_TOKENS = {
    "technology", "technologies", "tech", "inc", "llc", "corp", "corporation",
    "ltd", "limited", "solutions", "services", "systems", "company", "companies",
    "co", "group", "the", "and", "of", "a", "an", "for", "on", "in",
    "software", "digital", "global", "international", "consulting",
    "management", "holdings", "labs", "plc", "pte",
    "usa", "us", "america", "american",
}

# Super-short/numeric tokens to skip
def skip_token(tok: str) -> bool:
    if not tok:
        return True
    if len(tok) < 2:
        return True
    if tok in GENERIC_TOKENS:
        return True
    # numeric-only
    if tok.isdigit():
        return True
    return False


def main() -> None:
    t0 = time.time()
    con = duckdb.connect()
    df = con.execute(
        """
        SELECT DISTINCT company_name_canonical
        FROM 'data/unified.parquet'
        WHERE is_swe AND source_platform='linkedin'
          AND is_english = true AND date_flag='ok'
          AND company_name_canonical IS NOT NULL
        """
    ).df()

    print(f"[step4] distinct canonical companies: {len(df)}")

    # Split on whitespace and common punctuation
    split_re = re.compile(r"[\s\.\&\-\(\)/,'\"\+:;\|]+")
    tokens: set[str] = set()
    for name in df["company_name_canonical"].tolist():
        if not isinstance(name, str):
            continue
        lowered = name.lower()
        for tok in split_re.split(lowered):
            tok = tok.strip()
            if skip_token(tok):
                continue
            tokens.add(tok)

    # Write one token per line, sorted
    tokens_sorted = sorted(tokens)
    OUT_PATH.write_text("\n".join(tokens_sorted) + "\n")
    elapsed = time.time() - t0
    print(f"[step4] wrote {len(tokens_sorted)} tokens -> {OUT_PATH} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
