# Priors — pre-data contact

Written before touching `data/unified.parquet` and before opening v8 archive outputs. A reconciliation section is appended at the end after profiling + v8 read.

Grounded in: `docs/1-research-design.md` (RQ1–RQ4), `docs/preprocessing-schema.md` (fields available), user's gut sense (H1 AI-washing, H2 new job types), and The Economist "Code red" (Apr 13, 2026).

## Per-hypothesis priors

**H1 — AI-washing.** *Expect:* weakly supported. 2026 postings will show more AI vocabulary than 2024, but AI-requirement intensity (actual job content) will lag adoption rhetoric. The Economist's adoption data (~25% SF firms, BoE survey "essentially zero" employment impact) backs this. *Falsifier:* AI-requirement density scales tightly with layoff depth across periods. *Bet:* 65% directionally for H1.

**H2 — AI creates new job types.** *Expect:* supported direction. AI/LLM/agent/applied-AI/prompt-engineer titles will show measurable emergence in 2026, but share will be small (< 10% of 2026 SWE) — the new types are niche, not replacements. *Falsifier:* none of these titles shows meaningful 2026 share; or conversely, they dominate (> 25%). *Bet:* 80% for emergence direction, 30% for "substantial share."

**H3 — Non-AI macro.** *Expect:* moderately supported. Remote-posting share up modestly; aggregator share up more noticeably (offshoring intermediaries). Covid-binge correction + rate hikes are the real drivers per Economist. *Falsifier:* remote and aggregator shares flat or falling 2024→2026. *Bet:* 60% for H3.

**H4 — Industry spread to non-tech.** *Expect:* supported. Non-tech SWE share will rise; Big Tech's share of SWE postings will fall. *Falsifier:* industry distribution of SWE postings is stable across 2024→2026. *Bet:* 70% for H4.

**H5 — Junior vs senior divergence.** *Expect:* contested. Sub-prediction (a) junior-first (automation story) versus (b) senior-restructuring (content change concentrated in senior postings) are both plausible. On priors alone, 50/50. *Falsifier:* flat content change across all seniority buckets. *Bet:* hold until data.

**H6 — Big Tech vs rest.** *Expect:* supported. Public layoff record (Oracle, Block, Amazon, Meta) should manifest as Big Tech SWE-share decline + elevated AI-mention density. *Falsifier:* Big Tech share rises; or AI density is equal across tiers. *Bet:* 75% for Big Tech share drop; 70% for AI-density differential.

**H7 — SWE vs control.** *Expect:* mixed. If H1+H3 hold, SWE and control should co-move (macro story). If AI is the real driver, control occupations should be flat while SWE moves. Asymmetric-evidence limitation: no 2024 control baseline. *Bet:* 55% that SWE and control move in the same direction (supporting H1+H3).

## Pre-committed reference lists

**Big Tech canonical-name list (H6 proxy)** — matched against `company_name_canonical`:
`alphabet, google, meta, facebook, amazon, amazon web services, aws, apple, microsoft, oracle, netflix, block, square, uber, airbnb, salesforce, anthropic, openai, nvidia, tesla, adobe, ibm, linkedin`

**Tech-industry list (H4 "non-tech" = anything not in this list)** — matched against `company_industry`:
`Computer Software, Information Technology and Services, Internet, Computer Games, Computer Hardware, Computer Networking, Computer & Network Security, Semiconductors, Telecommunications, Software Development, IT Services and IT Consulting, Technology, Information and Internet`

Everything outside this set counts as non-tech for H4/S8 scans.

## Reconciliation with v8 archive

*Appended after Phase A profile run and v8 INDEX.md / gate_3 read.*

Read after Phase A profile: `exploration-archive/v8_claude_opus47/reports/INDEX.md`, `memos/gate_3.md`. What v8 actually answered on these seven hypotheses, and how my priors hold up:

| H | My prior bet | What v8 found (headline source) | Updated post-v8 stance |
|---|---|---|---|
| H1 | 65% AI-washing | **Against.** T18 DiD: AI-vocab rewriting is 99% SWE-specific vs control; tech-count 95%, breadth 72%. Req-section shrink is SWE −10.7pp / adj −10.9pp / control +0.9pp (opposite direction on control). If H1 were true (AI narrative layer over macro cuts), SWE and control should co-move in content; they don't. | Drop to **20%** for content-level H1. Layoff-narrative H1 (volumes, not content) still untested. |
| H2 | 80% direction | **Supported with twist.** T10: 15,021 new titles. T28: ML/AI archetype grew 2.6%→16.2%, **81% new-entrant-driven** (new firms entering, not pivots). | Keep **80%** but reframe: emergence happens via new-company entry. S3/S4 scans still worth running for title-level granularity v8 didn't publish. |
| H3 | 60% macro | **Partially against.** T17: AI surge 26/26 metros (CV 0.29, uncorrelated with junior decline, r=−0.11). Not concentrated in layoff hubs. Gate 1: JOLTS Info-sector hiring low at 2026-03/04 (0.66× of 2023 avg) — macro backdrop real for volumes, but content change is SWE-specific. | Drop to **35%** for H3 as causal story of CONTENT; macro still likely explains volume. |
| H4 | 70% industry spread | **Partially untested.** v8's T28 did cross-archetype but not cross-industry dispersion. T16 focused on within-company. | Keep **70%**; S7/S8 scans add genuine value — v8 did not directly publish `company_industry` dispersion. |
| H5 | 50/50 | **Strongly against (a), for (b).** T21: mentor at mid-senior 1.46–1.73× 2024→2026 vs 1.07× at entry (senior-specific). T20: seniority boundaries SHARPENED, not blurred (mid-senior/associate +0.084 AUC, J3/S4 +0.14). T12: seniors changed MORE than juniors in content. | Sub-prediction (b) senior-restructuring: **85%**. (a) junior-first: **10%**. |
| H6 | 75% Big Tech drop | **Mostly untested.** T16's 240-company overlap panel shows 76% of companies broadened — not polarized by tier. But Big Tech tier stratification is absent. | Keep **65%**; S10 adds value v8 didn't publish. |
| H7 | 55% co-movement | **Against co-movement.** T18 DiD explicitly tests this and finds SWE moved, control did not (except for description length and soft-skills). Strongest direct evidence. | Drop to **15%** for co-movement; the v8 answer is already strong but S11 re-derives on the current artifact as a sanity check. |

### Net value-add of this EDA over v8

1. **Replicate v8's T18-style divergence (Sv, S11)** — independent re-derivation on `unified.parquet` that someone can audit in a single notebook.
2. **Big Tech stratification (S10, H6)** — new lens v8 did not publish. Tests whether the cross-seniority rewriting concentrates at Big Tech or is uniform across tiers.
3. **Industry dispersion (S7/S8, H4)** — v8 stopped at archetype level; we add `company_industry` distributional analysis, which is the direct test of Economist H4.
4. **New-AI-title emergence (S3)** — v8's T10 noted 15,021 new titles but no systematic share-of-corpus by AI-keyword title. S3 closes that.
5. **Layoff-narrative vs content framing of H1** — v8 tested content-level H1 only. We surface the volume/share trajectory per Sv so the interview-phase (RQ4) can pursue the narrative-layer question.

The EDA is NOT a replacement for v8; it is a narrower audit + three extensions (H4, H6, and the narrative framing of H1). Triangulation with v8 tables is a verification step, not a novelty claim.
