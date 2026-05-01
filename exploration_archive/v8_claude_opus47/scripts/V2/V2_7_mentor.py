"""V2.7 — T21 cross-seniority mentor rate re-derivation.

Compute mentor-binary rate per seniority bucket per period using V1-refined
strict pattern. Verify T21's claim: 1.73× at mid-senior vs 1.07× at entry.

Also sample 50 matches per senior period to semantically check precision.
"""
import re
import random
import pandas as pd
import duckdb
from pathlib import Path

REPO = Path("/home/jihgaboot/gabor/job-research")
OUT_DIR = REPO / "exploration/artifacts/V2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MGMT_STRICT = re.compile(
    r"\b(mentor(?:s|ed|ing|ship)?|coach(?:es|ed|ing)?|(?:hire\s+and\s+(?:develop|manage|grow))|(?:hire(?:\s+a)?\s+team)|(?:hiring\s+(?:and|manager|engineers|team|plan))|headcount|performance[- ]review)\b",
    re.IGNORECASE,
)
# T21 primary: bare mentor pattern
MENTOR_ONLY = re.compile(r"\bmentor\w*", re.IGNORECASE)


def main():
    con = duckdb.connect()
    # LLM-text-only primary (T21 spec)
    q = """
    SELECT uid, description, description_core_llm, seniority_final,
           CASE WHEN period LIKE '2024%' THEN '2024'
                WHEN period LIKE '2026%' THEN '2026' ELSE 'other' END AS period_bucket,
           CASE WHEN description_core_llm IS NOT NULL AND description_core_llm != ''
                THEN 'llm' ELSE 'raw' END AS text_source
    FROM 'data/unified.parquet'
    WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true
      AND seniority_final IS NOT NULL
    """
    print("[V2.7] loading SWE rows with seniority (LLM-only)...")
    df = con.execute(q).df()
    df = df[df["text_source"] == "llm"]  # LLM-only primary (T21 spec)
    df["text"] = df["description_core_llm"].fillna("")
    df = df[df["period_bucket"].isin(["2024", "2026"])]
    df["mgmt_strict"] = df["text"].apply(lambda t: bool(MGMT_STRICT.search(t)) if t else False).astype(int)
    df["mentor_only"] = df["text"].apply(lambda t: bool(MENTOR_ONLY.search(t)) if t else False).astype(int)
    print(f"[V2.7] rows: {len(df)}")

    # Per seniority × period — mentor_only (T21 primary)
    pivots = df.groupby(["seniority_final", "period_bucket"]).agg(
        n=("uid", "count"),
        mentor_rate=("mentor_only", "mean"),
        mgmt_strict_rate=("mgmt_strict", "mean"),
    ).reset_index()
    piv = pivots.pivot(index="seniority_final", columns="period_bucket", values="mentor_rate").rename(
        columns={"2024": "r_2024", "2026": "r_2026"}
    )
    n_piv = pivots.pivot(index="seniority_final", columns="period_bucket", values="n").rename(
        columns={"2024": "n_2024", "2026": "n_2026"}
    )
    out = piv.join(n_piv)
    out["ratio_26_24"] = out["r_2026"] / out["r_2024"]
    out.to_csv(OUT_DIR / "V2_7_mentor_by_seniority.csv")
    print("\n[V2.7] Mentor rate by seniority × period:")
    print(out.round(4).to_string())
    print("\n  T21 claim: entry 1.07x, associate 0.61x, mid-senior 1.73x, director 1.30x")

    # Precision check: sample 50 senior 2024 + 50 senior 2026 matches
    sr = df[df["seniority_final"].isin(["mid-senior", "director"]) & (df["mgmt_strict"] == 1)]
    sr24 = sr[sr["period_bucket"] == "2024"]
    sr26 = sr[sr["period_bucket"] == "2026"]
    random.seed(42)
    s24 = sr24.sample(min(50, len(sr24)), random_state=42)
    s26 = sr26.sample(min(50, len(sr26)), random_state=42)
    # Extract ±100 char window
    def window(text):
        m = MGMT_STRICT.search(text)
        if not m:
            return ""
        s = max(0, m.start() - 100)
        e = min(len(text), m.end() + 100)
        return text[s:e]
    s24 = s24.copy()
    s26 = s26.copy()
    s24["window"] = s24["text"].apply(window)
    s26["window"] = s26["text"].apply(window)
    s24[["uid", "seniority_final", "window"]].to_csv(OUT_DIR / "V2_7_mentor_precision_sample_2024.csv", index=False)
    s26[["uid", "seniority_final", "window"]].to_csv(OUT_DIR / "V2_7_mentor_precision_sample_2026.csv", index=False)

    # Print 10 windows from each for rapid semantic check
    print("\n[V2.7] Sample 2024 senior mentor matches (first 10):")
    for i, r in s24.head(10).iterrows():
        print(f"  {r['uid']}: ...{r['window']}...")
    print("\n[V2.7] Sample 2026 senior mentor matches (first 10):")
    for i, r in s26.head(10).iterrows():
        print(f"  {r['uid']}: ...{r['window']}...")


if __name__ == "__main__":
    main()
