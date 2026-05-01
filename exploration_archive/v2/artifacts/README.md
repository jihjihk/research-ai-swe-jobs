# T25 / T26 interview artifacts

Source basis:
- Stage 8 parquet: `preprocessing/intermediate/stage8_final.parquet`
- Seniority trend: `exploration/tables/T09/seniority_variant_entry_share_by_period.csv`
- Senior archetype chart: `exploration/tables/T21/T21_summary.csv`, `exploration/tables/T21/T21_archetypes.csv`
- Posting / usage divergence: `exploration/tables/T14/posting_ai_rates.csv`, `exploration/tables/T14/benchmarks.csv`

Text field rule:
- Stage 8 does not contain `description_core_llm`
- These excerpt cards use `description_core` as the primary field, with `description` as fallback only when needed

Files:
- `T25_inflated_junior_jds.png` / `.pdf`
- `T25_paired_jds_over_time.png` / `.pdf`
- `T25_junior_share_trend.png` / `.pdf`
- `T25_senior_archetype_chart.png` / `.pdf`
- `T25_posting_usage_divergence.png` / `.pdf`

Selection notes:
- Inflated junior cards are drawn from the strongest scope-inflation examples surfaced in T12/T19/T23
- Paired JD cards use same-company, same-title families in 2024-04 vs 2026-03
- The junior-share trend is the default `seniority_final` series with AI release markers added as timeline annotations
- The senior archetype chart combines management/orchestration intensity with the new/classic senior split
- The divergence chart juxtaposes posting AI shares against Anthropic and Stack Overflow benchmarks
