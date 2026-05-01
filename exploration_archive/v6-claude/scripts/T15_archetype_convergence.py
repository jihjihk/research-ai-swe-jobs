"""T15 follow-up: per-archetype convergence analysis.

Runs after T15 main script once archetype labels are available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path("/home/jihgaboot/gabor/job-research")
SHARED = REPO / "exploration" / "artifacts" / "shared"
TBL = REPO / "exploration" / "tables" / "T15"


def trimmed_centroid(X: np.ndarray, trim: float = 0.1) -> np.ndarray:
    if len(X) == 0:
        return np.zeros(X.shape[1])
    mu = X.mean(axis=0)
    mu /= max(np.linalg.norm(mu), 1e-9)
    sims = X @ mu
    k = max(1, int(len(X) * (1 - trim)))
    keep = np.argsort(sims)[-k:]
    mu2 = X[keep].mean(axis=0)
    mu2 /= max(np.linalg.norm(mu2), 1e-9)
    return mu2


def main():
    print("[load]")
    idx = pq.read_table(SHARED / "swe_embedding_index.parquet").to_pandas()
    emb = np.load(SHARED / "swe_embeddings.npy")
    meta = pq.read_table(
        SHARED / "swe_cleaned_text.parquet",
        columns=["uid", "text_source", "period", "seniority_3level", "source"],
    ).to_pandas()
    arch = pq.read_table(SHARED / "swe_archetype_labels.parquet").to_pandas()
    idx = idx.merge(meta, on="uid", how="left").merge(arch, on="uid", how="left")
    idx = idx[idx["text_source"] == "llm"].reset_index(drop=True)
    idx["period_bucket"] = np.where(idx["period"].str.startswith("2024"), "2024", "2026")
    emb_aligned = emb[idx["row_idx"].values]

    # Drop the unlabeled archetype
    idx = idx[idx["archetype_name"] != "No text / raw-only (unlabeled)"].reset_index(drop=True)
    # Re-align embeddings to the filtered idx by computing a mapping
    # Easier: keep full emb_aligned and reference by new index
    # Rebuild: filter idx against original row alignment
    print("loaded archetype-labeled rows:", len(idx))

    # Reload from scratch and filter
    idx2 = pq.read_table(SHARED / "swe_embedding_index.parquet").to_pandas()
    emb_full = np.load(SHARED / "swe_embeddings.npy")
    idx2 = idx2.merge(meta, on="uid", how="left").merge(arch, on="uid", how="left")
    idx2 = idx2[idx2["text_source"] == "llm"]
    idx2 = idx2[idx2["archetype_name"] != "No text / raw-only (unlabeled)"].reset_index(drop=True)
    idx2["period_bucket"] = np.where(idx2["period"].str.startswith("2024"), "2024", "2026")
    emb_a = emb_full[idx2["row_idx"].values]

    # For each archetype: compute junior-senior cosine per period, and within-2024 source split
    rows = []
    archetypes = sorted(idx2["archetype_name"].dropna().unique())
    for a in archetypes:
        sub = idx2[idx2["archetype_name"] == a]
        if len(sub) < 100:
            continue
        # 2024 pooled junior vs senior
        def _js(mask):
            d = sub[mask]
            j = d[d["seniority_3level"] == "junior"].index.values
            s = d[d["seniority_3level"] == "senior"].index.values
            if len(j) < 15 or len(s) < 15:
                return float("nan"), len(j), len(s)
            cj = trimmed_centroid(emb_a[j])
            cs = trimmed_centroid(emb_a[s])
            return float(np.dot(cj, cs)), len(j), len(s)

        js_2024, nj24, ns24 = _js(sub["period_bucket"] == "2024")
        js_2026, nj26, ns26 = _js(sub["period_bucket"] == "2026")
        # within-2024 cross-source
        js_ars, _, _ = _js(sub["source"] == "kaggle_arshkon")
        js_asa, _, _ = _js(sub["source"] == "kaggle_asaniczka")
        within_24 = (
            abs(js_ars - js_asa) if not (np.isnan(js_ars) or np.isnan(js_asa)) else float("nan")
        )
        cross = (
            abs(js_2026 - js_2024) if not (np.isnan(js_2026) or np.isnan(js_2024)) else float("nan")
        )
        snr = cross / within_24 if within_24 and within_24 > 1e-9 else float("nan")

        rows.append(
            {
                "archetype": a,
                "n": len(sub),
                "js_2024": js_2024,
                "js_2026": js_2026,
                "shift_2024_to_2026": js_2026 - js_2024 if not np.isnan(js_2024) and not np.isnan(js_2026) else float("nan"),
                "arshkon_js": js_ars,
                "asaniczka_js": js_asa,
                "within_2024_noise": within_24,
                "cross_period_shift": cross,
                "snr": snr,
                "nj_2024": nj24,
                "ns_2024": ns24,
                "nj_2026": nj26,
                "ns_2026": ns26,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(TBL / "per_archetype_convergence.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
