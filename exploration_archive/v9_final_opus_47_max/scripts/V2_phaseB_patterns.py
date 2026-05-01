"""V2 Phase B — Pattern validation spot-checks.

Targets per spec:
- T22 patterns (2 of 7 primaries): spot-check firm_requirement + hedging on fresh 30-row samples.
- T21 cluster 0 composition: sample 20 rows, confirm AI/LLM engineering content.
"""

from __future__ import annotations

import json
import re
import random
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNIFIED = str(ROOT / "data" / "unified.parquet")
VALIDATED = ROOT / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json"
OUT = ROOT / "exploration" / "tables" / "V2"

with open(VALIDATED) as f:
    pats = json.load(f)

HEDGE = pats["t22_patterns"]["aspiration_hedging"]["pattern"]
FIRM = pats["t22_patterns"]["firm_requirement"]["pattern"]

rx_hedge = re.compile(HEDGE, flags=re.IGNORECASE)
rx_firm = re.compile(FIRM, flags=re.IGNORECASE)

# Sample matches for each pattern, stratified 15 pre-2026 + 15 scraped
random.seed(20202020)

def sample_matches(rx, n_per_period=15):
    q = f"""
    SELECT uid, source, period, company_name_canonical, title, description
    FROM '{UNIFIED}'
    WHERE is_swe=TRUE AND source_platform='linkedin' AND is_english=TRUE AND date_flag='ok'
    """
    con = duckdb.connect()
    df = con.execute(q).df()
    con.close()
    df["era"] = df["source"].map(lambda s: "2026" if s == "scraped" else "2024")
    df["desc_low"] = df["description"].fillna("").str.lower()
    df["match"] = df["desc_low"].apply(lambda t: bool(rx.search(t)))
    samples = []
    for era in ["2024", "2026"]:
        sub = df[(df["era"] == era) & df["match"]]
        idx = random.sample(range(len(sub)), min(n_per_period, len(sub)))
        for i in idx:
            row = sub.iloc[i]
            # Extract 200-char context around first match
            text = row["desc_low"]
            m = rx.search(text)
            if m is None:
                continue
            start = max(0, m.start() - 100)
            end = min(len(text), m.end() + 100)
            ctx = text[start:end]
            samples.append({
                "uid": row["uid"],
                "era": era,
                "match_text": m.group(0),
                "context": ctx.replace("\n", " ")[:400],
            })
    return pd.DataFrame(samples)


# Sample T22 patterns
for tag, rx in [("hedging", rx_hedge), ("firm_requirement", rx_firm)]:
    s = sample_matches(rx, n_per_period=15)
    print(f"\n=== {tag} samples (n={len(s)}) ===")
    for _, row in s.iterrows():
        print(f"  [{row['era']}] match='{row['match_text']}': ... {row['context']} ...")
        print()
    s.to_csv(OUT / f"phaseB_samples_{tag}.csv", index=False)


# Cluster 0 semantic check (T21 / T34 Applied-AI archetype)
print("\n=== T21 Cluster 0 sample (20 rows, 2026) ===")
cluster_df = pd.read_csv(ROOT / "exploration" / "tables" / "T21" / "cluster_assignments.csv")
c0_2026 = cluster_df[(cluster_df["cluster_id"] == 0) & (cluster_df["period"] == 2026)]
print(f"Cluster 0 2026: {len(c0_2026)} rows")
sample = c0_2026.sample(n=20, random_state=20202020)

con = duckdb.connect()
enriched = con.execute(f"""
SELECT ca.uid, u.title, LEFT(u.description, 400) AS desc_head
FROM read_csv_auto('/home/jihgaboot/gabor/job-research/exploration/tables/T21/cluster_assignments.csv') ca
JOIN '{UNIFIED}' u ON ca.uid = u.uid
WHERE ca.cluster_id = 0 AND ca.period = 2026
""").df()
enriched = enriched.sample(n=20, random_state=20202020)
print("\nTitles in cluster 0 (20 sampled 2026 rows):")
for _, r in enriched.iterrows():
    print(f"  - {r['title']}")

enriched.to_csv(OUT / "phaseB_cluster0_sample.csv", index=False)
print("\nSaved sample CSVs.")
