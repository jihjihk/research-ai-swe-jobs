"""V2.9b — T13 section classifier on non-SWE postings.

Apply T13's section classifier to 500 control postings sampled across periods
and compare requirements-section rate to SWE. If classifier breaks down on
non-SWE, the T18 requirements-section DiD may be artifactual.
"""
import sys
import random
import pandas as pd
import duckdb
from pathlib import Path

REPO = Path("/home/jihgaboot/gabor/job-research")
sys.path.insert(0, str(REPO / "exploration/scripts"))
from T13_section_classifier import classify_description, SECTION_TYPES

OUT_DIR = REPO / "exploration/artifacts/V2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    con = duckdb.connect()
    q = """
    SELECT uid, description, is_swe, is_swe_adjacent, is_control,
           CASE WHEN period LIKE '2024%' THEN '2024'
                WHEN period LIKE '2026%' THEN '2026' ELSE 'other' END AS period_bucket
    FROM 'data/unified.parquet'
    WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok'
      AND description IS NOT NULL AND description != ''
      AND (is_swe OR is_control)
    """
    df = con.execute(q).df()
    df = df[df["period_bucket"].isin(["2024", "2026"])]

    # Sample 500 control, 500 SWE across periods (250 each)
    random.seed(42)
    samples = []
    for grp in ["swe", "control"]:
        if grp == "swe":
            sub = df[df["is_swe"]]
        else:
            sub = df[df["is_control"]]
        for per in ["2024", "2026"]:
            ss = sub[sub["period_bucket"] == per]
            s = ss.sample(min(250, len(ss)), random_state=42)
            s["group"] = grp
            samples.append(s)
    sample = pd.concat(samples)
    print(f"[V2.9b] sampled {len(sample)} rows (250 per grp × per)")

    # Classify each
    rows = []
    for _, r in sample.iterrows():
        sec = classify_description(r["description"])
        total = sum(s["chars"] for s in sec.values()) or 1
        req_share = sec["requirements"]["chars"] / total
        resp_share = sec["responsibilities"]["chars"] / total
        benefits_share = sec["benefits"]["chars"] / total
        unclassified_share = sec["unclassified"]["chars"] / total
        has_req = sec["requirements"]["chars"] > 0
        rows.append({
            "uid": r["uid"], "group": r["group"], "period": r["period_bucket"],
            "req_share": req_share, "resp_share": resp_share,
            "benefits_share": benefits_share, "unclassified_share": unclassified_share,
            "has_req": has_req,
        })
    cls = pd.DataFrame(rows)
    cls.to_csv(OUT_DIR / "V2_9b_t13_nonswe_sample.csv", index=False)

    # Aggregates
    agg = cls.groupby(["group", "period"]).agg(
        n=("uid", "count"),
        req_share_mean=("req_share", "mean"),
        resp_share_mean=("resp_share", "mean"),
        benefits_share_mean=("benefits_share", "mean"),
        unclassified_share_mean=("unclassified_share", "mean"),
        has_req_rate=("has_req", "mean"),
    ).round(4)
    print("\n[V2.9b] T13 section shares by group × period:")
    print(agg)
    agg.to_csv(OUT_DIR / "V2_9b_t13_nonswe_agg.csv")

    # Compute Δ 2024→2026 per group
    print("\n[V2.9b] Δ req share 2024→2026:")
    for grp in ["swe", "control"]:
        r24 = cls[(cls["group"] == grp) & (cls["period"] == "2024")]["req_share"].mean()
        r26 = cls[(cls["group"] == grp) & (cls["period"] == "2026")]["req_share"].mean()
        print(f"  {grp}: 2024={r24:.4f}, 2026={r26:.4f}, Δ={r26-r24:+.4f}")
    print("\n  T18 claim: SWE Δ=-10.7pp, control Δ=+0.9pp")


if __name__ == "__main__":
    main()
