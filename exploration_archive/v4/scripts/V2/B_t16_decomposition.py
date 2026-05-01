"""V2 Part B: Re-derive T16's 87% between-company entry decomposition.

Independent implementation:
1. Define overlap panel: companies with >=3 SWE postings in BOTH 2024 and 2026 (after description_hash dedup within company).
2. Compute YOE-proxy entry share (yoe_extracted<=2) per company per period.
3. Shift-share: Δ_total = Δ_within + Δ_between + Δ_interaction.

Standard formulas (sectoral shift-share):
  w_i^t = share of panel posts at company i in period t
  p_i^t = entry rate at company i in period t
  Aggregate rate P^t = sum_i w_i^t * p_i^t
  ΔP = sum_i (Δw_i * p_i^2024)       # between (composition shift at 2024 prevalence)
     + sum_i (w_i^2024 * Δp_i)       # within (entry-rate change at 2024 weights)
     + sum_i (Δw_i * Δp_i)           # interaction

Also computed with t=2026 base and midpoint, reported as sensitivity.
"""

import duckdb
import pandas as pd

con = duckdb.connect()
UNI = "data/unified.parquet"
BASE = "source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE"


def build_panel(min_posts=3, scope="arshkon"):
    """Build overlap panel.
    scope: 'arshkon' = arshkon∩scraped (T16's convention)
           'pooled' = (arshkon+asaniczka)∩scraped
    """
    if scope == "arshkon":
        src_filter = "(source='kaggle_arshkon' OR source='scraped')"
    else:
        src_filter = "(source IN ('kaggle_arshkon','kaggle_asaniczka','scraped'))"
    q = f"""
    WITH f AS (
      SELECT company_name_canonical AS company,
             CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period,
             description_hash,
             uid, yoe_extracted
      FROM '{UNI}'
      WHERE {BASE} AND company_name_canonical IS NOT NULL AND {src_filter}
    ),
    dedup AS (
      SELECT company, period, description_hash,
             any_value(uid) AS uid,
             any_value(yoe_extracted) AS yoe_extracted
      FROM f
      GROUP BY 1,2,3
    ),
    counts AS (
      SELECT company, period,
             count(*) AS n_swe,
             sum(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS n_yoe_known,
             sum(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted <= 2 THEN 1 ELSE 0 END) AS n_entry_yoe
      FROM dedup GROUP BY 1,2
    ),
    pv AS (
      SELECT company,
             sum(CASE WHEN period='2024' THEN n_swe ELSE 0 END) AS n_2024,
             sum(CASE WHEN period='2026' THEN n_swe ELSE 0 END) AS n_2026,
             sum(CASE WHEN period='2024' THEN n_yoe_known ELSE 0 END) AS k_2024,
             sum(CASE WHEN period='2026' THEN n_yoe_known ELSE 0 END) AS k_2026,
             sum(CASE WHEN period='2024' THEN n_entry_yoe ELSE 0 END) AS e_2024,
             sum(CASE WHEN period='2026' THEN n_entry_yoe ELSE 0 END) AS e_2026
      FROM counts GROUP BY 1
    )
    SELECT * FROM pv
    WHERE n_2024 >= {min_posts} AND n_2026 >= {min_posts}
    """
    df = con.execute(q).fetchdf()
    return df


def decompose(panel, use_yoe_known=True):
    """Two decomposition variants:
    - V2 standard (base=2024 with interaction)
    - T16 midpoint (avg shares x dp for within, dw x avg p for between, no interaction)

    use_yoe_known: if True (T16 style), metric is e/k_yoe_known. Else metric is e/n_swe.
    Weights: always n_swe (T16 uses n, not n_yoe_known).
    """
    panel = panel.copy()
    # drop rows where yoe_known == 0 in either period (no valid metric)
    panel = panel[(panel["k_2024"] > 0) & (panel["k_2026"] > 0)].copy()
    panel["w24"] = panel["n_2024"] / panel["n_2024"].sum()
    panel["w26"] = panel["n_2026"] / panel["n_2026"].sum()
    if use_yoe_known:
        panel["p24"] = panel["e_2024"] / panel["k_2024"]
        panel["p26"] = panel["e_2026"] / panel["k_2026"]
    else:
        panel["p24"] = panel["e_2024"] / panel["n_2024"]
        panel["p26"] = panel["e_2026"] / panel["n_2026"]
    panel["dw"] = panel["w26"] - panel["w24"]
    panel["dp"] = panel["p26"] - panel["p24"]
    panel["w_avg"] = (panel["w24"] + panel["w26"]) / 2
    panel["p_avg"] = (panel["p24"] + panel["p26"]) / 2

    P24 = (panel["w24"] * panel["p24"]).sum()
    P26 = (panel["w26"] * panel["p26"]).sum()
    delta_total = P26 - P24

    # V2 (standard Layard-style base 2024)
    between_std = (panel["dw"] * panel["p24"]).sum()
    within_std = (panel["w24"] * panel["dp"]).sum()
    inter = (panel["dw"] * panel["dp"]).sum()

    # T16 (midpoint, no interaction)
    within_t16 = (panel["w_avg"] * panel["dp"]).sum()
    between_t16 = (panel["dw"] * panel["p_avg"]).sum()

    return {
        "n_companies": len(panel),
        "n_posts_2024": int(panel["n_2024"].sum()),
        "n_posts_2026": int(panel["n_2026"].sum()),
        "P24_pct": P24 * 100, "P26_pct": P26 * 100,
        "total_delta_pp": delta_total * 100,
        # V2 standard
        "v2_within_pp": within_std * 100,
        "v2_between_pp": between_std * 100,
        "v2_interaction_pp": inter * 100,
        "v2_within_share": within_std / delta_total if delta_total != 0 else 0,
        # T16 midpoint
        "t16_within_pp": within_t16 * 100,
        "t16_between_pp": between_t16 * 100,
        "t16_within_share": within_t16 / delta_total if delta_total != 0 else 0,
        "t16_between_share": between_t16 / delta_total if delta_total != 0 else 0,
    }


print("\n\n##### T16 CONVENTION (arshkon∩scraped, metric=entry/yoe_known, weight=n_swe) #####")
for min_posts in [3, 5, 10]:
    panel = build_panel(min_posts, scope="arshkon")
    r = decompose(panel, use_yoe_known=True)
    print(f"\n=== arshkon∩scraped, >={min_posts} SWE both periods ===")
    for k, v in r.items():
        print(f"  {k}: {v}")

print("\n\n##### POOLED 2024 (arshkon+asaniczka)∩scraped, metric=entry/yoe_known #####")
for min_posts in [3, 5]:
    panel = build_panel(min_posts, scope="pooled")
    r = decompose(panel, use_yoe_known=True)
    print(f"\n=== pooled∩scraped, >={min_posts} SWE both periods ===")
    for k, v in r.items():
        print(f"  {k}: {v}")

# Without dedup (raw), arshkon∩scraped
print("\n\n##### NO DEDUP (arshkon∩scraped) — does dedup matter? #####")
q = f"""
WITH f AS (
  SELECT company_name_canonical AS company,
         CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period,
         uid, yoe_extracted
  FROM '{UNI}'
  WHERE {BASE} AND company_name_canonical IS NOT NULL
    AND (source='kaggle_arshkon' OR source='scraped')
),
counts AS (
  SELECT company, period, count(*) AS n_swe,
         sum(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS n_yoe_known,
         sum(CASE WHEN yoe_extracted IS NOT NULL AND yoe_extracted <= 2 THEN 1 ELSE 0 END) AS n_entry_yoe
  FROM f GROUP BY 1,2
),
pv AS (
  SELECT company,
         sum(CASE WHEN period='2024' THEN n_swe ELSE 0 END) AS n_2024,
         sum(CASE WHEN period='2026' THEN n_swe ELSE 0 END) AS n_2026,
         sum(CASE WHEN period='2024' THEN n_yoe_known ELSE 0 END) AS k_2024,
         sum(CASE WHEN period='2026' THEN n_yoe_known ELSE 0 END) AS k_2026,
         sum(CASE WHEN period='2024' THEN n_entry_yoe ELSE 0 END) AS e_2024,
         sum(CASE WHEN period='2026' THEN n_entry_yoe ELSE 0 END) AS e_2026
  FROM counts GROUP BY 1
)
SELECT * FROM pv WHERE n_2024 >= 3 AND n_2026 >= 3
"""
panel_raw = con.execute(q).fetchdf()
r = decompose(panel_raw, use_yoe_known=True)
for k, v in r.items():
    print(f"  {k}: {v}")

# Cross-check: combined column entry definition
# T16 reported -0.27pp within under combined column
print("\n\n##### COMBINED-COLUMN ENTRY (arshkon∩scraped, dedup, metric=entry/n_best_known) #####")
q_cb = f"""
WITH f AS (
  SELECT company_name_canonical AS company,
         CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period,
         description_hash, uid,
         CASE
           WHEN llm_classification_coverage='labeled' THEN seniority_llm
           WHEN llm_classification_coverage='rule_sufficient' THEN seniority_final
           ELSE NULL
         END AS sen_best
  FROM '{UNI}'
  WHERE {BASE} AND company_name_canonical IS NOT NULL
    AND (source='kaggle_arshkon' OR source='scraped')
),
dedup AS (
  SELECT company, period, description_hash,
         any_value(uid) AS uid,
         any_value(sen_best) AS sen_best
  FROM f GROUP BY 1,2,3
),
counts AS (
  SELECT company, period,
         count(*) AS n_swe,
         sum(CASE WHEN sen_best IS NOT NULL THEN 1 ELSE 0 END) AS n_best_known,
         sum(CASE WHEN sen_best = 'entry' THEN 1 ELSE 0 END) AS n_entry
  FROM dedup GROUP BY 1,2
),
pv AS (
  SELECT company,
         sum(CASE WHEN period='2024' THEN n_swe ELSE 0 END) AS n_2024,
         sum(CASE WHEN period='2026' THEN n_swe ELSE 0 END) AS n_2026,
         sum(CASE WHEN period='2024' THEN n_best_known ELSE 0 END) AS k_2024,
         sum(CASE WHEN period='2026' THEN n_best_known ELSE 0 END) AS k_2026,
         sum(CASE WHEN period='2024' THEN n_entry ELSE 0 END) AS e_2024,
         sum(CASE WHEN period='2026' THEN n_entry ELSE 0 END) AS e_2026
  FROM counts GROUP BY 1
)
SELECT * FROM pv WHERE n_2024 >= 3 AND n_2026 >= 3
"""
panel_cb = con.execute(q_cb).fetchdf()
r_cb = decompose(panel_cb, use_yoe_known=True)
for k, v in r_cb.items():
    print(f"  {k}: {v}")
