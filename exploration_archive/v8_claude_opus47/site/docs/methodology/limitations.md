# Limitations

The following are known constraints that affect which claims the paper can make and how it must frame them.

## Known pipeline bugs

| Bug | Status | Impact |
|---|---|---|
| `is_remote_inferred = 100% False` | Confirmed T17 | Blocks every remote/hybrid analysis. Preprocessing fix required before any hybrid claim. |
| "Graduate" / "New College Graduate" under-detected (Stage 5) | Confirmed T03 | ~1,000 scraped SWE rows with clear junior titles are `seniority_final = 'unknown'`. J1 under-counted by ~2.4 pp on scraped. |
| `title_normalized` strips level indicators | Confirmed T30 | J5/S3 title-keyword matches must operate on raw `title`, not `title_normalized`. |
| `posting_age_days` degenerate on scraped | Confirmed T19 | Most scraped LinkedIn rows have 2.8% `date_posted` populated; `posting_age_days` is effectively null. Use `scrape_week` for within-scraped temporal analysis. |

## Macro context

**2026 is a JOLTS Info-sector hiring low.** Information-sector openings Feb 2026 = 91K = 0.66× the 2023 average, 0.74× the 2024 average. Every "employers are hiring more X" framing is disallowed. All claims are share-of-SWE.

See [T07 — benchmarks & power analysis](../raw/wave-1/T07.md) for the full JOLTS time series.

## Instrumentation differences

Three sources are three different instruments:

| Source | Format | Instrument feature | Comparability risk |
|---|---|---|---|
| arshkon | HTML-stripped text, companion tables | Curated industry + company size joins | High internal quality, small SWE count |
| asaniczka | HTML-stripped, description join | Largest volume | Zero native entry labels |
| scraped | Markdown-preserving, search metadata | Fresh; daily panel | Taxonomy drift (bare "developer" lost 61pp entry share) |

"Comparability" is dimension-dependent and task-specific. Every cross-period comparison must flag:

- **Platform taxonomy drift** — LinkedIn's "IT Services and IT Consulting" → "Software Development" +17pp relabeling.
- **Native-label semantic drift** — bare "developer" lost 61pp native-entry share.
- **Aggregator share nearly doubled** (9.2% → 16.6%) — aggregator-excluded is a standard sensitivity.

## Recruiter-LLM mediation

T29 measured an authorship-score shift of +1.14 SDs 2024 → 2026. Recruiter-LLM drafting tools have measurably entered the posting pipeline.

| Metric | Low-40% authorship subset retention | Verdict |
|---|---:|---|
| AI-strict | 75-77 % | Robust (real content change) |
| AI-broad | 86 % | Robust |
| Mentor-binary | 72 % (T29) / 105 % (V2 3-feature) | Method-sensitive |
| Breadth-residualized | 71 % | Method-sensitive |

Paper cites AI-strict as "15-30% recruiter-LLM mediated" with robust attribution; mentor/breadth as "0-30% mediated with method uncertainty."

## Asaniczka native-label caveat

Asaniczka has **zero** native entry labels. Its `seniority_native` distribution contains only `mid-senior` and `associate`. Consequences:

- Any pooled-2024 comparison of native-entry shares is structurally broken (asaniczka drags entry to zero).
- Senior-side findings must lead with arshkon-only baseline; pooled as sensitivity only.
- Any YOE-based junior proxy (J3, J4) is label-independent and can safely pool.

## Worker-benchmark caveats

T23 benchmarks are all self-reported, platform-biased:

- Stack Overflow Developer Survey 2024 — response bias toward engaged developers.
- Octoverse 2024 — OSS sample, not representative of closed-source work.
- Anthropic 2025 — task-exposure, not usage; may over-state.

The paper reports **direction** (employers trail workers) with a **range** (−15 to −30 pp), not a point estimate.

## Sampling-frame artifact

74.5% of scraped companies are new entrants (no 2024 match). The 2026 LinkedIn SWE corpus samples a substantively different firm population than 2024:

- More startups, more AI-native firms, more tech-giant coverage, more aggregators.
- Within-company decompositions (240-co panel, T16) correct for this at the company level.
- Corpus-level metrics have an implicit "which companies are posting" confound.

Every longitudinal corpus-level claim needs the **returning-cos-only sensitivity** (see H_H in T24). The paper should address this head-on in limitations.

## AI-broad pattern precision

The `ai_broad` pattern has 0.80 semantic precision (V1). V1-refined strict pattern is primary (SNR 38.7, precision 98%); broad is sensitivity.

## LLM-frame selection artifact (V1.5)

Scraped 2026 rows with `llm_extraction_coverage = 'labeled'` have J2 share 4.3-6.2% while `not_selected` rows have 2.1-2.6%. The LLM frame preferentially selects junior postings.

Consequence: any junior-claim restricted to the labeled frame systematically overstates junior direction by +1-2 pp. Every junior claim reports **both** labeled and full-corpus directions.

## Links to raw

- [T03 — seniority audit](../raw/wave-1/T03.md)
- [T07 — JOLTS benchmarks](../raw/wave-1/T07.md)
- [T17 — geography (is_remote_inferred bug)](../raw/wave-3/T17.md)
- [T29 — authorship mediation](../raw/wave-3/T29.md)
- [V1 verification §5 — LLM-frame J2 flip](../raw/verification/V1_verification.md)
- [V2 verification](../raw/verification/V2_verification.md)
</content>
</invoke>