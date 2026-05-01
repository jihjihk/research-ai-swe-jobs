# Factcheck 04: "What the compliance lobby can't see"

Headline claim: Financial Services AI-strict 15.74% vs Software Development 15.25% in 2026 scraped SWE LinkedIn.

## Method

- Source: `/home/jihgaboot/gabor/job-research/data/unified_core.parquet` (via DuckDB).
- Filter: `source_platform='linkedin' AND is_swe AND is_english AND date_flag='ok' AND source='scraped' AND llm_extraction_coverage='labeled'`, restricted to rows with non-null `description_core_llm` (matches prior investigator's universe; n=25,547 — all 2026-03/2026-04).
- Pattern: `ai_strict_v1_rebuilt` from `exploration/artifacts/shared/validated_mgmt_patterns.json` (precision 0.96; 2026 sub-period precision 0.92).
- Text field matched: `description_core_llm` (case-insensitive regex).
- Wilson 95% CIs (z=1.96).

## Independently computed

| Slice | n | k | Rate | Wilson 95% CI |
|---|---|---|---|---|
| Financial Services | 1,785 | 281 | 15.74% | [14.13, 17.51] |
| Software Development | 6,870 | 1,048 | 15.25% | [14.42, 16.12] |
| Overall | 25,547 | 3,382 | 13.24% | [12.83, 13.66] |

## Comparison to prior CSV

| Slice | Prior rate | My rate | Δ (pp) |
|---|---|---|---|
| Financial Services | 15.74% | 15.74% | 0.00 |
| Software Development | 15.25% | 15.25% | 0.00 |
| Overall | 13.24% | 13.24% | 0.00 |

All three match exactly (well within the 0.5 pp tolerance). n, k, and CI bounds are identical.

## CI overlap

Financial Services CI [14.13, 17.51] and Software Development CI [14.42, 16.12] **overlap substantially** — the 0.49 pp gap is not statistically distinguishable at 95%. Both are, however, clearly above the overall 13.24% rate (non-overlapping with [12.83, 13.66]).

## Verdict

**MATCHES with a qualification.** Headline numbers replicate exactly. The rhetorical framing "Financial Services edges Software Development" is numerically accurate but the CIs overlap; the defensible story is "Financial Services matches Software Development on AI-strict adoption," both above the overall SWE baseline.

Note: the pattern's 2026 precision is 0.92, so ~8% of flagged rows may be false positives; this applies equally to both industries and does not change the relative comparison.
