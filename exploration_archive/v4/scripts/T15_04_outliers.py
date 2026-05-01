"""T15 Step 8: Outlier identification — postings most unlike their seniority peers.

For each period x seniority group, identifies postings with the lowest cosine
similarity to the group trimmed-centroid. Dumps sample titles and the most
distinctive TF-IDF terms per outlier for qualitative inspection.

Outputs:
  tables/T15/outliers_per_group.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

ROOT = Path("/home/jihgaboot/gabor/job-research")
ART = ROOT / "exploration/artifacts/T15"
TABLES = ROOT / "exploration/tables/T15"
SHARED = ROOT / "exploration/artifacts/shared"
TABLES.mkdir(parents=True, exist_ok=True)

TRIM_FRAC = 0.10
N_OUTLIERS = 15


def main():
    idx = pd.read_parquet(ART / "sample_index.parquet")
    emb = np.load(ART / "sample_embeddings.npy")
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    idx["group"] = idx["period2"] + "_" + idx["seniority_3level"]
    groups = ["2024_junior", "2024_mid", "2024_senior",
              "2026_junior", "2026_mid", "2026_senior"]

    # Get titles from unified parquet (not in cleaned-text artifact)
    con = duckdb.connect()
    titles = con.execute(f"""
        SELECT uid, title, company_name_canonical
        FROM read_parquet('{ROOT}/data/unified.parquet')
        WHERE uid IN ({','.join([repr(u) for u in idx['uid'].tolist()])})
    """).df() if False else None
    # That would be a huge literal. Use temp parquet join.
    tmp = ART / "_tmp_sample_uids.parquet"
    idx[["uid"]].to_parquet(tmp, index=False)
    titles = con.execute(f"""
        SELECT u.uid, u.title, u.company_name_canonical
        FROM read_parquet('{ROOT}/data/unified.parquet') u
        INNER JOIN read_parquet('{tmp}') s USING (uid)
    """).df()
    tmp.unlink()
    idx = idx.merge(titles, on="uid", how="left")

    all_rows = []
    for g in groups:
        m = idx[idx["group"] == g]
        if len(m) < 20:
            continue
        rows = m["sample_row"].values
        Xg = emb[rows]
        # Trimmed centroid
        c = Xg.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        sims = Xg @ c
        cutoff = np.quantile(sims, TRIM_FRAC)
        mask = sims >= cutoff
        c2 = Xg[mask].mean(axis=0)
        c2 = c2 / (np.linalg.norm(c2) + 1e-12)
        sims2 = Xg @ c2
        # Lowest 15
        order = np.argsort(sims2)
        outliers = order[:N_OUTLIERS]
        for k in outliers:
            all_rows.append({
                "group": g,
                "uid": m["uid"].values[k],
                "title": m["title"].values[k],
                "company": m["company_name_canonical"].values[k],
                "source": m["source"].values[k],
                "yoe_extracted": m["yoe_extracted"].values[k],
                "cosine_to_centroid": float(sims2[k]),
            })
    df = pd.DataFrame(all_rows)
    df.to_csv(TABLES / "outliers_per_group.csv", index=False)
    print(f"  wrote outliers_per_group.csv ({len(df)} rows)")
    for g in groups:
        sub = df[df["group"] == g].head(5)
        print(f"\n  {g} — top 5 outliers:")
        for _, r in sub.iterrows():
            title = (r["title"] or "")[:80]
            print(f"    {r['cosine_to_centroid']:.3f}  {title}")


if __name__ == "__main__":
    main()
