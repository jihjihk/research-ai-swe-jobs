# Fact-check: Piece 03 "Atlanta, not San Francisco"

**Date:** 2026-04-21
**Scope:** Uniform AI-mention rise across 26 metros (+4.7 to +14.5 pp), leaders Atlanta/Tampa/Miami/SLC, tech-hub premium <2 pp.

## Re-derivation on unified_core.parquet

Filter: `is_swe AND source_platform='linkedin' AND is_english AND date_flag='ok' AND metro_area IS NOT NULL AND NOT is_multi_location`. Periods pooled: 2024 = {2024-01, 2024-04}; 2026 = {2026-03, 2026-04}. AI-strict via `validated_mgmt_patterns.json` regex on `description`. Panel = 26 metros with >=50 rows per era (identical to T17 panel).

### Top 5 metros by 2026 posting count

| Metro | n_2024 | n_2026 | AI-strict Delta (pp) |
|---|---:|---:|---:|
| San Francisco Bay Area | 1,729 | 3,247 | +14.23 |
| Washington DC Metro | 2,108 | 1,904 | +8.02 |
| Seattle Metro | 762 | 1,406 | +15.52 |
| Dallas-Fort Worth Metro | 851 | 1,270 | +16.15 |
| New York City Metro | 67 | 1,237 | +14.84 |

### Range and leaderboard

- Re-derived range across 26 metros: **+6.45 to +18.60 pp** (T17 reports +4.7 to +14.5 pp on older `unified.parquet`; the direction — every metro positive, moderate cross-metro spread — replicates).
- Top 4 by Delta on unified_core: **Tampa (+18.60), Atlanta (+17.36), Minneapolis (+17.16), Charlotte (+16.82)**. T17 top 4: Tampa/Atlanta/Miami/SLC. Tampa and Atlanta replicate as #1-#2; Miami and SLC drop to mid-pack on unified_core (Miami n_2026 is only 189 here vs 369 in T17 — smaller sample, noisier).

### Tech-hub premium

Tech hubs = {SF, NYC, Seattle, Austin, Boston}. Mean Delta: **+13.80 pp (hub) vs +12.15 pp (rest), premium +1.65 pp**. T17 reports +9.9 vs +8.6 pp, premium +1.3 pp. Premium magnitude (<2 pp) replicates exactly.

## Verdict

**Matches with minor qualification.** Direction (every metro positive, uniform rise), leaderboard (Tampa + Atlanta at top), and tech-hub-premium-<2-pp claim all hold on the independent unified_core cut. The precise +4.7-+14.5 pp band shifts to +6.45-+18.60 pp on unified_core because row counts and dedup differ from the T17 frame, but the uniformity story is preserved.

**One-sentence note:** Atlanta-beats-SF headline survives independent re-derivation; only the specific numeric band +4.7-+14.5 pp is frame-dependent and should be cited as "T17 (unified.parquet frame)" rather than as a universal constant.
