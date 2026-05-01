"""V1.6 - T12 relabeling diagnostic: independent TF-IDF cosine computation.

Compare:
(a) 2026 entry corpus vs 2024 entry corpus
(b) 2026 entry corpus vs 2024 mid-senior corpus

If (a) cosine > (b) cosine: "period-effect dominant" (entry26 is closer to entry24 than to midsr24).
If inverted: relabeling hypothesis (entry26 looks more like 2024 mid-senior).
"""
import duckdb
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OUT_DIR = Path("/home/jihgaboot/gabor/job-research/exploration/artifacts/V1")
CLEAN_PARQ = "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_cleaned_text.parquet"


def main():
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT uid, description_cleaned, text_source, source, period, seniority_final
        FROM '{CLEAN_PARQ}'
        WHERE text_source = 'llm'
          AND description_cleaned IS NOT NULL AND description_cleaned != ''
    """).df()
    print(f"Loaded {len(df)} LLM-cleaned rows")

    df["period_group"] = df["period"].apply(lambda p: "2024" if str(p).startswith("2024") else "2026")

    # Pooled 2024: arshkon + asaniczka
    # Entry slice = J2 (entry + associate)
    # Mid-senior = "mid-senior"
    entry_24 = df[(df.period_group == "2024") & (df.seniority_final.isin(["entry", "associate"]))]
    entry_26 = df[(df.period_group == "2026") & (df.seniority_final.isin(["entry", "associate"]))]
    midsr_24 = df[(df.period_group == "2024") & (df.seniority_final == "mid-senior")]
    midsr_26 = df[(df.period_group == "2026") & (df.seniority_final == "mid-senior")]

    print(f"Entry 2024: {len(entry_24)}")
    print(f"Entry 2026: {len(entry_26)}")
    print(f"Mid-senior 2024: {len(midsr_24)}")
    print(f"Mid-senior 2026: {len(midsr_26)}")

    # Build combined corpus text per group (as single mega-document per group)
    def combine_corpus(grp, cap=None):
        # Concatenate descriptions; cap total chars if specified
        if cap and len(grp) > cap:
            grp = grp.sample(n=cap, random_state=42)
        return " ".join(grp["description_cleaned"].fillna("").astype(str).tolist())

    g_entry_24 = combine_corpus(entry_24)
    g_entry_26 = combine_corpus(entry_26)
    g_midsr_24 = combine_corpus(midsr_24)
    g_midsr_26 = combine_corpus(midsr_26)

    # Fit TF-IDF on all 4 concatenated docs
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, stop_words="english")
    X = vec.fit_transform([g_entry_24, g_entry_26, g_midsr_24, g_midsr_26])
    # 4x20000 matrix
    labels = ["entry24", "entry26", "midsr24", "midsr26"]
    print(f"TF-IDF matrix: {X.shape}")

    # Compute pairwise cosines
    sim = cosine_similarity(X)
    cos_df = pd.DataFrame(sim, index=labels, columns=labels)
    print("\n=== Pairwise cosines ===")
    print(cos_df.round(4).to_string())

    # Key comparisons
    cos_entry26_entry24 = sim[1, 0]  # entry26 vs entry24
    cos_entry26_midsr24 = sim[1, 2]  # entry26 vs midsr24
    cos_entry26_midsr26 = sim[1, 3]  # entry26 vs midsr26
    cos_entry24_midsr24 = sim[0, 2]  # entry24 vs midsr24

    print(f"\n=== Key comparisons ===")
    print(f"  cos(entry26, entry24) = {cos_entry26_entry24:.4f}")
    print(f"  cos(entry26, midsr24) = {cos_entry26_midsr24:.4f}")
    print(f"  cos(entry26, midsr26) = {cos_entry26_midsr26:.4f}")
    print(f"  cos(entry24, midsr24) = {cos_entry24_midsr24:.4f}")

    period_effect_dominant = cos_entry26_entry24 > cos_entry26_midsr24
    print(f"\nPeriod-effect dominant (entry26 closer to entry24 than to midsr24): {period_effect_dominant}")
    print(f"Difference: {(cos_entry26_entry24 - cos_entry26_midsr24):+.4f}")

    # Also check sen26-sen24 cosine (T12's "seniors changed more" finding; sen changes more means lower cosine)
    cos_sen26_sen24 = sim[3, 2]
    print(f"\ncos(midsr26, midsr24) = {cos_sen26_sen24:.4f}  (T12 reported ~0.942)")
    print(f"cos(entry26, entry24) = {cos_entry26_entry24:.4f}  (T12 reported ~0.953)")
    print(f"Change magnitude: senior (1-cos) = {1-cos_sen26_sen24:.4f}; entry (1-cos) = {1-cos_entry26_entry24:.4f}")
    sen_changed_more = (1 - cos_sen26_sen24) > (1 - cos_entry26_entry24)
    print(f"Seniors changed more than juniors: {sen_changed_more}")

    summary = {
        "cosines": {
            "entry26_entry24": float(cos_entry26_entry24),
            "entry26_midsr24": float(cos_entry26_midsr24),
            "entry26_midsr26": float(cos_entry26_midsr26),
            "entry24_midsr24": float(cos_entry24_midsr24),
            "midsr26_midsr24": float(cos_sen26_sen24),
        },
        "period_effect_dominant": bool(period_effect_dominant),
        "seniors_changed_more": bool(sen_changed_more),
        "n_rows": {
            "entry_24": int(len(entry_24)),
            "entry_26": int(len(entry_26)),
            "midsr_24": int(len(midsr_24)),
            "midsr_26": int(len(midsr_26)),
        }
    }
    with open(OUT_DIR / "V1_6_relabeling_cosines.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'V1_6_relabeling_cosines.json'}")


if __name__ == "__main__":
    main()
