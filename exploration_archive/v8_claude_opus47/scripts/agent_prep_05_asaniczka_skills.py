"""Agent Prep step 5: parse asaniczka SWE skills_raw into structured artifacts.

- long-form uid | skill parquet
- skill frequency CSV
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/home/jihgaboot/gabor/job-research")
OUT_LONG = ROOT / "exploration/artifacts/shared/asaniczka_structured_skills.parquet"
OUT_FREQ = ROOT / "exploration/artifacts/shared/asaniczka_skill_frequency.csv"


def main() -> None:
    con = duckdb.connect()
    df = con.execute(
        """
        SELECT uid, skills_raw
        FROM read_parquet('/home/jihgaboot/gabor/job-research/data/unified.parquet')
        WHERE source='kaggle_asaniczka'
          AND is_swe=true
          AND is_english=true
          AND date_flag='ok'
          AND source_platform='linkedin'
        """
    ).df()
    print(f"Loaded {len(df):,} asaniczka SWE rows ({df['skills_raw'].notna().sum():,} have skills)")

    rows = []
    counter: Counter[str] = Counter()
    for uid, raw in zip(df["uid"].values, df["skills_raw"].values):
        if not raw:
            continue
        # comma-separated, lowercase, trim, dedupe within the row
        seen = set()
        for piece in raw.split(","):
            skill = piece.strip().lower()
            if not skill or skill in seen:
                continue
            seen.add(skill)
            rows.append((uid, skill))
        counter.update(seen)

    long_df = pd.DataFrame(rows, columns=["uid", "skill"])
    table = pa.Table.from_pandas(long_df, preserve_index=False)
    pq.write_table(table, OUT_LONG, compression="snappy")
    size_kb = OUT_LONG.stat().st_size / 1024
    print(f"Wrote long-form uid|skill to {OUT_LONG} ({len(long_df):,} rows, {size_kb:.1f} KB)")

    n_postings = len(df)
    freq = (
        pd.DataFrame(counter.items(), columns=["skill", "n_postings"])
        .sort_values("n_postings", ascending=False)
    )
    freq["share_of_asaniczka_swe"] = freq["n_postings"] / n_postings
    freq.to_csv(OUT_FREQ, index=False)
    print(f"Wrote frequency CSV to {OUT_FREQ} ({len(freq):,} unique skills)")
    print("Top 20 skills:")
    print(freq.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
