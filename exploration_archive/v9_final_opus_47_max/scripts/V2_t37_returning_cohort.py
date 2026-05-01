"""V2 — H_T37: Verify returning cohort AI-strict robustness.

Claim (T37): AI-strict +9.72 pp full → +8.36 pp returning, ratio 0.86 (robust).
V2: independent calculation using V1-validated pattern on raw description.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNIFIED = str(ROOT / "data" / "unified.parquet")
COHORT = ROOT / "exploration" / "artifacts" / "shared" / "returning_companies_cohort.csv"
VALIDATED = ROOT / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json"
OUT = ROOT / "exploration" / "tables" / "V2"

with open(VALIDATED) as f:
    p = json.load(f)
AI_V1 = p["v1_rebuilt_patterns"]["ai_strict_v1_rebuilt"]["pattern"]
rx = re.compile(AI_V1, flags=re.IGNORECASE)

cohort = pd.read_csv(COHORT)
returning = set(cohort["company_name_canonical"])
print(f"Returning cohort: {len(returning)} companies")

q = f"""
SELECT uid, source, period, company_name_canonical, LOWER(description) AS txt
FROM '{UNIFIED}'
WHERE is_swe=TRUE AND source_platform='linkedin' AND is_english=TRUE AND date_flag='ok'
  AND company_name_canonical IS NOT NULL
"""
con = duckdb.connect()
df = con.execute(q).df()
con.close()
df["era"] = np.where(df["source"] == "scraped", "2026", "2024")

tx = df["txt"].fillna("").to_numpy()
df["ai"] = np.fromiter((1 if rx.search(t) else 0 for t in tx), dtype=np.int8, count=len(tx))

full_24 = df[df["era"] == "2024"]["ai"].mean()
full_26 = df[df["era"] == "2026"]["ai"].mean()
ret = df[df["company_name_canonical"].isin(returning)]
ret_24 = ret[ret["era"] == "2024"]["ai"].mean()
ret_26 = ret[ret["era"] == "2026"]["ai"].mean()

print(f"Full corpus (V1 rebuilt): 2024={full_24*100:.3f}%, 2026={full_26*100:.3f}%, Δ={full_26-full_24:+.4f} = {(full_26-full_24)*100:+.2f} pp")
print(f"Returning cohort (V1 rebuilt): 2024={ret_24*100:.3f}%, 2026={ret_26*100:.3f}%, Δ={(ret_26-ret_24)*100:+.2f} pp")
print(f"Retention ratio: {(ret_26-ret_24)/(full_26-full_24):.2f}")
