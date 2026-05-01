"""Wave 1.5 Agent Prep - Step 1: Cleaned text column.

For all SWE LinkedIn rows (filtered by default SQL):
 - primary text = description_core_llm where llm_extraction_coverage = 'labeled'
 - fallback = raw description (no rule-based description_core — retired)
 - Strip company-name tokens (from step 4 stoplist) and English stopwords.
Output: `swe_cleaned_text.parquet` with metadata columns.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import duckdb
import nltk
import pyarrow as pa
import pyarrow.parquet as pq

# Ensure stopwords resource is available
try:
    from nltk.corpus import stopwords
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords  # noqa: F811

OUT_DIR = Path("exploration/artifacts/shared")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "swe_cleaned_text.parquet"
STOPLIST_PATH = OUT_DIR / "company_stoplist.txt"

# Pre-build combined stopword set (English stopwords + company tokens)
ENGLISH_STOP = set(stopwords.words("english"))
COMPANY_STOP = {
    line.strip() for line in STOPLIST_PATH.read_text().splitlines() if line.strip()
}
STOP = ENGLISH_STOP | COMPANY_STOP

# Token pattern: keep alphanumerics and a few tech-relevant joiners, but for text
# cleaning we want to collapse everything that's not alphanumeric. We keep '+' and
# '#' and '.' for downstream uses (e.g., c++, c#, .net) — no, those are handled by
# the tech matrix step which uses raw text. For cleaned text we lowercase and
# tokenize on non-alphanumeric.
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\+\#\-\./]+|[A-Za-z]")


def clean_text(text: str | None) -> str:
    if not isinstance(text, str) or not text:
        return ""
    lowered = text.lower()
    tokens = TOKEN_RE.findall(lowered)
    out = []
    for tok in tokens:
        # Strip trailing punctuation for filtering purposes
        core = tok.strip(".-/")
        if not core or len(core) < 2:
            continue
        if core in STOP:
            continue
        # Keep the core form; original casing has been lowered
        out.append(core)
    return " ".join(out)


def main() -> None:
    t0 = time.time()
    con = duckdb.connect()

    # Column selection keeps the result set small
    print("[step1] loading SWE LinkedIn filtered rows (metadata only, text streamed)")
    df = con.execute(
        """
        SELECT
          uid,
          source,
          period,
          description,
          description_core_llm,
          llm_extraction_coverage,
          llm_classification_coverage,
          seniority_final,
          seniority_3level,
          seniority_final_source,
          is_aggregator,
          company_name_canonical,
          metro_area,
          yoe_min_years_llm,
          yoe_extracted,
          swe_classification_tier
        FROM 'data/unified.parquet'
        WHERE is_swe AND source_platform = 'linkedin'
          AND is_english = true AND date_flag = 'ok'
        """
    ).df()
    print(f"[step1] rows: {len(df)}")

    # Resolve text source per row
    coverage = df["llm_extraction_coverage"].values
    desc_llm = df["description_core_llm"].values
    desc_raw = df["description"].values

    text_source_list: list[str] = []
    cleaned_list: list[str] = []
    n = len(df)
    report_every = 5000
    for i in range(n):
        cov = coverage[i]
        llm_t = desc_llm[i]
        use_llm = (
            cov == "labeled"
            and isinstance(llm_t, str)
            and llm_t.strip() != ""
        )
        if use_llm:
            text_source_list.append("llm")
            cleaned_list.append(clean_text(llm_t))
        else:
            text_source_list.append("raw")
            cleaned_list.append(clean_text(desc_raw[i]))
        if (i + 1) % report_every == 0:
            print(f"[step1]  processed {i+1}/{n}")

    # Free the big text columns we no longer need
    df = df.drop(columns=["description", "description_core_llm"])

    out_df = df.copy()
    out_df["description_cleaned"] = cleaned_list
    out_df["text_source"] = text_source_list

    # Reorder + select final columns
    col_order = [
        "uid",
        "description_cleaned",
        "text_source",
        "source",
        "period",
        "seniority_final",
        "seniority_3level",
        "is_aggregator",
        "company_name_canonical",
        "metro_area",
        "yoe_min_years_llm",
        "yoe_extracted",
        "llm_classification_coverage",
        "swe_classification_tier",
        "seniority_final_source",
    ]
    out_df = out_df[col_order]

    # Convert to pyarrow and write
    table = pa.Table.from_pandas(out_df, preserve_index=False)
    pq.write_table(table, OUT_PATH, compression="zstd")
    print(f"[step1] wrote {len(out_df)} rows -> {OUT_PATH}")

    # Text-source distribution by source
    dist = (
        out_df.groupby(["source", "text_source"])
        .size()
        .rename("n")
        .reset_index()
    )
    print("[step1] text_source distribution:")
    print(dist)

    elapsed = time.time() - t0
    print(f"[step1] elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()
