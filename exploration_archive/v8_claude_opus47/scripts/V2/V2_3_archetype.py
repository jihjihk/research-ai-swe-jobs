"""V2.3 — Independent archetype projection via nearest-centroid on embeddings.

Project 8,000 labeled archetypes onto 34,102 LLM-labeled SWE corpus via
nearest-centroid in the MiniLM 384-dim embedding space. Compute AI-STRICT Δ
per top-10 archetype. Verify T28's claims: 20/20 archetypes positive,
≥+10pp in 15/20, systems_engineering at ~+0.16pp.
"""
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path("/home/jihgaboot/gabor/job-research")
OUT_DIR = REPO / "exploration/artifacts/V2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AI_STRICT = r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|vector database|pinecone|huggingface|hugging face)\b"
AI_BROAD = r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|vector database|pinecone|huggingface|hugging face|ai|artificial intelligence|ml|machine learning|llm|large language model|generative ai|genai|anthropic)\b"


def main():
    print("[V2.3] loading embeddings + labels...")
    emb = np.load(REPO / "exploration/artifacts/shared/swe_embeddings.npy").astype(np.float32)
    idx = pd.read_parquet(REPO / "exploration/artifacts/shared/swe_embedding_index.parquet")
    labs = pd.read_parquet(REPO / "exploration/artifacts/shared/swe_archetype_labels.parquet")

    # Merge labels to idx (8k of 34k)
    idx_l = idx.merge(labs, on="uid", how="left")
    print(f"[V2.3] idx={len(idx)}, labeled={idx_l['archetype_name'].notna().sum()}")

    # Compute centroids
    labeled_mask = idx_l["archetype_name"].notna()
    centroids = {}
    for name in idx_l.loc[labeled_mask, "archetype_name"].unique():
        m = idx_l["archetype_name"] == name
        rows = idx_l.loc[m, "row_idx"].values
        if len(rows) < 3:
            continue
        centroids[name] = emb[rows].mean(axis=0)
    # Normalize centroids for cosine
    archs = list(centroids.keys())
    cent_arr = np.stack([centroids[a] for a in archs])
    cent_arr = cent_arr / np.linalg.norm(cent_arr, axis=1, keepdims=True)
    emb_n = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    # For each row, nearest centroid
    print("[V2.3] assigning nearest centroid...")
    sims = emb_n @ cent_arr.T  # (N, K)
    best = sims.argmax(axis=1)
    projected = [archs[b] for b in best]
    idx_l["v2_archetype"] = projected

    # Agreement with T28's projection?
    t28 = pd.read_parquet(REPO / "exploration/artifacts/T28/projected_archetypes.parquet")
    print("T28 projected columns:", t28.columns.tolist())
    comp = idx_l.merge(t28, on="uid", how="inner")
    print("merged count:", len(comp))
    # Find T28 archetype column
    t28_col = [c for c in t28.columns if c != "uid"][0]
    agreement = (comp[t28_col] == comp["v2_archetype"]).mean()
    print(f"[V2.3] V2 vs T28 projection agreement: {agreement*100:.2f}%")

    # Pull period + compute AI-strict per archetype per period
    con = duckdb.connect()
    per = con.execute(f"""
        SELECT uid,
               CASE WHEN period LIKE '2024%' THEN '2024'
                    WHEN period LIKE '2026%' THEN '2026' ELSE 'other' END AS period_bucket,
               CAST(regexp_matches(lower(COALESCE(description_core_llm, description, '')), '{AI_STRICT}') AS INTEGER) AS ai_strict,
               CAST(regexp_matches(lower(COALESCE(description_core_llm, description, '')), '{AI_BROAD}') AS INTEGER) AS ai_broad
        FROM 'data/unified.parquet'
        WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true
    """).df()
    merged = idx_l.merge(per, on="uid", how="inner")
    merged = merged[merged["period_bucket"].isin(["2024", "2026"])]
    print(f"[V2.3] merged {len(merged)} rows")

    # Per archetype Δ (both strict and broad)
    rows = []
    for a in archs:
        sub = merged[merged["v2_archetype"] == a]
        n24 = (sub["period_bucket"] == "2024").sum()
        n26 = (sub["period_bucket"] == "2026").sum()
        if n24 < 50 or n26 < 50:
            continue
        rows.append({
            "archetype": a, "n_2024": n24, "n_2026": n26,
            "ai_strict_2024": sub.loc[sub["period_bucket"] == "2024", "ai_strict"].mean(),
            "ai_strict_2026": sub.loc[sub["period_bucket"] == "2026", "ai_strict"].mean(),
            "delta_strict_pp": (sub.loc[sub["period_bucket"] == "2026", "ai_strict"].mean() - sub.loc[sub["period_bucket"] == "2024", "ai_strict"].mean()) * 100,
            "ai_broad_2024": sub.loc[sub["period_bucket"] == "2024", "ai_broad"].mean(),
            "ai_broad_2026": sub.loc[sub["period_bucket"] == "2026", "ai_broad"].mean(),
            "delta_broad_pp": (sub.loc[sub["period_bucket"] == "2026", "ai_broad"].mean() - sub.loc[sub["period_bucket"] == "2024", "ai_broad"].mean()) * 100,
        })
    dr = pd.DataFrame(rows).sort_values("n_2024", ascending=False)
    dr.to_csv(OUT_DIR / "V2_3_ai_by_archetype.csv", index=False)
    print("\n[V2.3] AI-strict + AI-broad by archetype (V2 projection):")
    print(dr.to_string(index=False))

    # Verify claims
    top10 = dr.head(10)
    print("\n[V2.3] Top-10 by n_2024:")
    print(top10.to_string(index=False))
    print(f"\n  STRICT: Archetypes >=+5pp: {(dr['delta_strict_pp'] >= 5).sum()} / {len(dr)}")
    print(f"  STRICT: Archetypes >=+10pp: {(dr['delta_strict_pp'] >= 10).sum()} / {len(dr)}")
    print(f"  STRICT: Archetypes positive: {(dr['delta_strict_pp'] > 0).sum()} / {len(dr)}")
    print(f"  BROAD: Archetypes >=+5pp: {(dr['delta_broad_pp'] >= 5).sum()} / {len(dr)}")
    print(f"  BROAD: Archetypes >=+10pp: {(dr['delta_broad_pp'] >= 10).sum()} / {len(dr)}")
    print(f"  BROAD: Archetypes positive: {(dr['delta_broad_pp'] > 0).sum()} / {len(dr)}")
    if "systems_engineering" in dr["archetype"].values:
        s = dr[dr["archetype"] == "systems_engineering"].iloc[0]
        print(f"  systems_engineering Δ_strict = +{s['delta_strict_pp']:.2f}pp  (T28 claim +0.16pp)")
        print(f"  systems_engineering Δ_broad  = +{s['delta_broad_pp']:.2f}pp")
    # Save projection for V2.4
    idx_l[["uid", "v2_archetype"]].to_parquet(OUT_DIR / "V2_3_projection.parquet", index=False)


if __name__ == "__main__":
    main()
