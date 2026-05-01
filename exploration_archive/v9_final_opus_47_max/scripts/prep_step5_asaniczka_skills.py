"""Wave 1.5 Agent Prep - Step 5: Structured skills extraction (asaniczka only).

Parse `skills_raw` from asaniczka SWE rows, lowercase, strip, explode to long.
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

OUT_DIR = Path("exploration/artifacts/shared")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "asaniczka_structured_skills.parquet"


def main() -> None:
    t0 = time.time()
    con = duckdb.connect()
    # Get asaniczka SWE LinkedIn rows with skills_raw populated
    df = con.execute(
        """
        SELECT uid, skills_raw
        FROM 'data/unified.parquet'
        WHERE source = 'kaggle_asaniczka'
          AND is_swe AND source_platform='linkedin'
          AND is_english = true AND date_flag = 'ok'
          AND skills_raw IS NOT NULL AND skills_raw != ''
        """
    ).df()
    print(f"[step5] asaniczka SWE rows with skills_raw: {len(df)}")

    uids: list[str] = []
    skills_out: list[str] = []
    for uid, raw in zip(df["uid"].tolist(), df["skills_raw"].tolist(), strict=True):
        if not isinstance(raw, str):
            continue
        # Comma-separated, strip whitespace, lowercase
        for part in raw.split(","):
            skill = part.strip().lower()
            if not skill:
                continue
            uids.append(uid)
            skills_out.append(skill)

    print(f"[step5] total (uid, skill) rows: {len(uids)}")
    table = pa.table({"uid": uids, "skill": skills_out})
    pq.write_table(table, OUT_PATH)
    elapsed = time.time() - t0
    print(f"[step5] wrote -> {OUT_PATH} in {elapsed:.1f}s")
    # Quick summary: top 20 skills
    tally = (
        duckdb.sql(
            f"SELECT skill, count(*) n FROM '{OUT_PATH}' GROUP BY skill ORDER BY n DESC LIMIT 20"
        )
        .df()
    )
    print(tally)


if __name__ == "__main__":
    main()
