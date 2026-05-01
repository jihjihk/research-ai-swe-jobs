#!/usr/bin/env python3
"""
T29 step 2: Compute authorship features on description_core_llm
(Stage 9 LLM-cleaned core body) as the text-source-controlled subset.

The description_core_llm text is produced by the SAME Stage 9 cleaner
across periods. Rows with llm_extraction_coverage='labeled' in both 2024
and 2026 provide a fair comparison surface: any remaining stylistic
differences are authoring-style differences, not ingestion differences.

For this subset we use only features that survive normalization:
  - Vocabulary density (classic LLM tells)
  - Em-dash density
  - Type-Token Ratio
  - Hedging phrases
  - Sentence length mean/std
  - Comma/exclaim density
(We SKIP bullet density and paragraph length because Stage 9 may drop
 formatting.)
"""
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import duckdb
from tqdm import tqdm

OUT_T = "exploration/tables/T29"
os.makedirs(OUT_T, exist_ok=True)

# Reuse feature functions from step 1
import importlib.util
spec = importlib.util.spec_from_file_location("t29_feat", "exploration/scripts/T29_01_features.py")
# Do not re-run tests — just extract functions by importing module
# (importlib will execute top-level, which includes tests — they're fine.)
t29 = importlib.util.module_from_spec(spec)
# Avoid circular issue by patching argv
import sys
_orig_argv = sys.argv
sys.argv = ["T29_01_features.py"]
try:
    # Running the module triggers feature extraction on the whole corpus — NO.
    # Instead, import by executing ONLY the function definitions. Use exec with
    # a filtered version of the file.
    with open("exploration/scripts/T29_01_features.py") as f:
        src = f.read()
    # Cut off the "# Load SWE LinkedIn rows" section to avoid re-running
    cutoff = src.index("# Load SWE LinkedIn rows")
    hdr_src = src[:cutoff]
    ns = {"__name__": "t29_feat"}
    exec(hdr_src, ns)
finally:
    sys.argv = _orig_argv

compute_features = ns["compute_features"]

# ---------------------------------------------------------------------------
# Load description_core_llm (Stage 9 cleaned) — restrict to labeled rows
# ---------------------------------------------------------------------------
print("Loading LLM-cleaned core bodies...")
con = duckdb.connect()
df = con.execute("""
    SELECT uid, source, period,
           description_core_llm,
           CASE WHEN description_core_llm IS NOT NULL THEN 'llm'
                WHEN description_core IS NOT NULL THEN 'rule'
                ELSE 'raw'
           END AS text_source_core,
           is_aggregator,
           company_name_canonical,
           seniority_final,
           yoe_extracted
    FROM read_parquet('data/unified.parquet')
    WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true
      AND llm_extraction_coverage='labeled'
""").df()
print(f"LLM-labeled subset: {len(df):,} rows")

df["year"] = np.where(df["period"].astype(str).str.startswith("2024"), "2024", "2026")
print(df.groupby(["year", "source"]).size())

print("\nExtracting features on description_core_llm...")
tqdm.pandas()
feats = df["description_core_llm"].progress_apply(compute_features)
feat_df = pd.DataFrame(list(feats))
out = pd.concat([df.drop(columns=["description_core_llm"]).reset_index(drop=True),
                 feat_df.reset_index(drop=True)], axis=1)

print(f"\nMean features by year (LLM-cleaned subset):")
cols = ["n_chars", "llm_vocab_density_1k", "emdash_density_1k", "ttr",
        "mean_sent_len_chars", "std_sent_len_chars", "hedge_phrase_count",
        "comma_density_1k", "exclaim_density_1k"]
print(out.groupby("year")[cols].mean().round(3).T)

out.to_parquet(f"{OUT_T}/authorship_scores_llmcleaned.parquet", index=False)
print(f"\nSaved: {OUT_T}/authorship_scores_llmcleaned.parquet ({len(out):,} rows)")
