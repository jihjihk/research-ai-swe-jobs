"""T17 calibration — within-2024 metro noise for J3 and AI-strict, remote category summary."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "exploration" / "tables" / "T17"
PATTERNS = json.loads((ROOT / "exploration" / "artifacts" / "shared" / "validated_mgmt_patterns.json").read_text())
AI_STRICT = PATTERNS["ai_strict"]["pattern"]
SCOPE = PATTERNS["scope"]["pattern"]

con = duckdb.connect()

con.execute(f"""
CREATE OR REPLACE TABLE tf AS
SELECT uid,
  CASE WHEN regexp_matches(lower(description_cleaned), ?) THEN 1 ELSE 0 END AS ai_strict,
  CAST(list_reduce(list_transform(regexp_extract_all(lower(description_cleaned), ?), x -> 1), (acc, v) -> acc + v, 0) AS BIGINT) AS scope_term_count
FROM 'exploration/artifacts/shared/swe_cleaned_text.parquet'
""", [AI_STRICT, SCOPE])

con.execute("""
CREATE OR REPLACE TABLE base AS
SELECT
  u.uid, u.metro_area, u.source,
  CASE WHEN u.period IN ('2024-01','2024-04') THEN '2024' ELSE '2026' END AS era,
  u.yoe_min_years_llm, u.llm_classification_coverage,
  u.is_remote, u.is_multi_location,
  COALESCE(tf.ai_strict,0) AS ai_strict,
  COALESCE(tf.scope_term_count,0) AS scope_term_count,
  CASE WHEN u.llm_classification_coverage='labeled' AND u.yoe_min_years_llm IS NOT NULL AND u.yoe_min_years_llm<=2 THEN 1 ELSE 0 END AS j3,
  CASE WHEN u.llm_classification_coverage='labeled' AND u.yoe_min_years_llm IS NOT NULL THEN 1 ELSE 0 END AS lab
FROM 'data/unified.parquet' u
LEFT JOIN tf USING (uid)
WHERE u.is_swe=true AND u.source_platform='linkedin' AND u.is_english=true AND u.date_flag='ok'
""")


# 1. Within-2024 noise (arshkon vs asaniczka) for J3, AI-strict, scope per metro (≥50 per 2024 source)
w2024 = con.execute("""
SELECT metro_area,
  CASE WHEN source='kaggle_arshkon' THEN 'arshkon' ELSE 'asaniczka' END AS s,
  SUM(j3)::DOUBLE / NULLIF(SUM(lab),0) AS j3_share,
  AVG(ai_strict) AS ai_strict,
  AVG(scope_term_count) AS scope,
  COUNT(*) AS n
FROM base
WHERE era='2024' AND source IN ('kaggle_arshkon','kaggle_asaniczka') AND metro_area IS NOT NULL
GROUP BY 1, 2
HAVING COUNT(*) >= 50
""").df()

wid = w2024.pivot(index="metro_area", columns="s", values=["j3_share", "ai_strict", "scope", "n"]).dropna()
wid.columns = [f"{a}_{b}" for a, b in wid.columns]
wid = wid.reset_index()
wid["j3_abs_diff_w24"] = (wid["j3_share_arshkon"] - wid["j3_share_asaniczka"]).abs()
wid["ai_abs_diff_w24"] = (wid["ai_strict_arshkon"] - wid["ai_strict_asaniczka"]).abs()
wid["scope_abs_diff_w24"] = (wid["scope_arshkon"] - wid["scope_asaniczka"]).abs()
wid.to_csv(OUT_DIR / "within_2024_metro_calibration.csv", index=False)
print("[within-2024 metro noise (metros ≥50 per source)]")
print(wid[["metro_area", "j3_abs_diff_w24", "ai_abs_diff_w24", "scope_abs_diff_w24", "n_arshkon", "n_asaniczka"]].round(4).to_string(index=False))
print(f"\nn qualifying metros: {len(wid)}")
print(f"J3 median abs diff w24: {wid['j3_abs_diff_w24'].median():.4f}")
print(f"J3 mean abs diff w24: {wid['j3_abs_diff_w24'].mean():.4f}")
print(f"AI-strict median abs diff w24: {wid['ai_abs_diff_w24'].median():.4f}")
print(f"AI-strict mean abs diff w24: {wid['ai_abs_diff_w24'].mean():.4f}")
print(f"Scope median abs diff w24: {wid['scope_abs_diff_w24'].median():.4f}")

# 2. Metro-level SNR: cross-period |Δ| / within-2024 |diff|
# Load metro_metrics
metro_m = pd.read_csv(OUT_DIR / "metro_metrics.csv")
snr_cols = ["metro_area"]
merged = metro_m[["metro_area", "j3_share_labeled_delta", "ai_strict_delta", "scope_term_count_delta"]].merge(
    wid[["metro_area", "j3_abs_diff_w24", "ai_abs_diff_w24", "scope_abs_diff_w24"]], on="metro_area", how="left"
)
merged["j3_snr"] = merged["j3_share_labeled_delta"].abs() / merged["j3_abs_diff_w24"]
merged["ai_snr"] = merged["ai_strict_delta"].abs() / merged["ai_abs_diff_w24"]
merged["scope_snr"] = merged["scope_term_count_delta"].abs() / merged["scope_abs_diff_w24"]
merged.to_csv(OUT_DIR / "metro_snr.csv", index=False)
print("\n[metro SNR — metros with within-2024 calibration]")
print(merged[["metro_area", "j3_share_labeled_delta", "j3_snr", "ai_strict_delta", "ai_snr", "scope_term_count_delta", "scope_snr"]].round(3).to_string(index=False))

# 3. Remote vs metro content comparison (2026 only)
remote_vs_metro = con.execute("""
SELECT
  CASE WHEN is_remote THEN 'remote'
       WHEN is_multi_location THEN 'multi_location'
       WHEN metro_area IS NOT NULL THEN 'metro'
       ELSE 'other_null' END AS loc_cat,
  AVG(ai_strict) AS ai_strict,
  SUM(j3)::DOUBLE / NULLIF(SUM(lab),0) AS j3_share,
  AVG(scope_term_count) AS scope_terms,
  COUNT(*) AS n
FROM base
WHERE era='2026'
GROUP BY 1
""").df()
remote_vs_metro.to_csv(OUT_DIR / "remote_vs_metro_content.csv", index=False)
print("\n[2026 location-category content profile]")
print(remote_vs_metro.round(3).to_string(index=False))
