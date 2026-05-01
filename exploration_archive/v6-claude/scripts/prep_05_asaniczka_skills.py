"""Wave 1.5 Agent Prep — Step 5: asaniczka structured skills (long format).

Parse `skills_raw` for asaniczka SWE rows (comma-separated). Save per-skill:
  uid | skill

Output: exploration/artifacts/shared/asaniczka_structured_skills.parquet
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET = "/home/jihgaboot/gabor/job-research/data/unified.parquet"
OUT = Path(
    "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/asaniczka_structured_skills.parquet"
)


def main() -> None:
    t0 = time.time()
    con = duckdb.connect()
    q = f"""
    SELECT uid, skills_raw
    FROM '{PARQUET}'
    WHERE is_swe
      AND source_platform = 'linkedin'
      AND is_english = true
      AND date_flag = 'ok'
      AND source = 'kaggle_asaniczka'
      AND skills_raw IS NOT NULL
      AND skills_raw <> ''
    """
    rows = con.execute(q).fetchall()
    print(f"Asaniczka SWE rows with skills_raw: {len(rows):,}")

    long_uids: list[str] = []
    long_skills: list[str] = []
    for uid, raw in rows:
        if not raw:
            continue
        for s in raw.split(","):
            s = s.strip().lower()
            if not s:
                continue
            long_uids.append(uid)
            long_skills.append(s)

    print(f"Total skill rows (long format): {len(long_skills):,}")
    print(f"Distinct skills: {len(set(long_skills)):,}")

    tbl = pa.table(
        {
            "uid": pa.array(long_uids, type=pa.string()),
            "skill": pa.array(long_skills, type=pa.string()),
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, OUT, compression="zstd")
    print(f"Wrote {OUT} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
