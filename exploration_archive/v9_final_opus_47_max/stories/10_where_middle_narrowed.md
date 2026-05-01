---
title: "Where the middle narrowed"
slug: 10_where_middle_narrowed
word_count_target: 500-650
status: revised_after_factcheck_critic
---

# Where the middle narrowed

*The received wisdom blamed the entry rung. A recent revisionist view blames the mid-career rung. Both are looking in the wrong place — though how strongly to claim that depends on which 2024 baseline one trusts.*

The first story about AI and software hiring went after the junior rung. Stanford's Digital Economy Lab, Anthropic's Dario Amodei, SignalFire's State of Tech Talent — the case that the AI dividend comes at the expense of the zero-to-two-year engineer has been made prominently since early 2025. The first piece in this collection contested it. A competing, and more recent, revisionist story is that the casualty is the middle: the three-to-seven-year engineer, with enough experience to cost real money but not enough to guarantee an architectural contribution that a junior and a copilot could not produce together. Simon Willison's late-December 2025 Substack essay introduced the framing for a broad technical audience; Amir Hosseini and Alex Lichtinger's November 2025 SSRN paper, based on 62 million résumés, provided an adjacent empirical case — though its unit of analysis is education tier rather than years of experience.

Neither thesis survives the posting data as written.

Split the 22,811 labelled 2024 SWE postings and the 25,815 labelled 2026 SWE postings into five years-of-experience buckets: 0-2, 3-4, 5-7, 8-10 and 11+. The results are baseline-sensitive in a way that matters for interpretation. Under the pooled-2024 baseline: 0-2 rises 3.97 percentage points, 3-4 rises 1.94 points, 5-7 falls 1.88 points, 8-10 falls 3.32 points, 11+ falls 1.13 points. Under the more conservative arshkon-only 2024 baseline (the dataset's within-2024 calibration subset, without the asaniczka source's known senior-share over-report): 0-2 rises 1.70 points, 3-4 rises 1.78 points, 5-7 rises 2.70 points, 8-10 falls 1.00 points, 11+ rises 0.58 points.

The only bucket that falls under both baselines is 8-to-10 years of experience.

That is not the entry rung. Nor is it the three-to-seven middle. It is the rung above the middle — where engineers sit after a decade but before they have been promoted to principal or architect. On the returning-firms cohort of 2,109 employers that posted in both 2024 and 2026, the 8-10 rung falls 3.69 percentage points under the pooled baseline and 1.49 under arshkon. The pooled and arshkon magnitudes differ by a factor of more than three, and the pooled number owes some of its size to the asaniczka source's mechanical senior over-representation in 2024. The cleaner reading is: the 8-10 rung fell, across baselines, at a magnitude of one to three percentage points on a base of roughly 17.

What happens to the other rungs depends on which baseline one believes. On the pooled reading the 5-7 middle also falls, a little; on arshkon it rises, a little. The 3-4 rung rises on both. The junior 0-2 rung rises on both. Willison's "mid-career squeeze" does not survive either reading; Hosseini and Lichtinger's résumé-level analysis cuts on education tier rather than years of experience, so it maps only loosely onto our buckets, but the posting data do not produce the U-shape they identify.

The most defensible conclusion is narrower and more modest than the language of "squeeze" allows. Software-engineering job advertisements between 2024 and 2026 show growth in the junior and 3-to-4-year tiers, near-stability or modest rise in the 5-to-7 tier, and consistent contraction in the 8-to-10 tier. The 11-plus rung's behaviour is noisy across baselines. This is a posting-content result. Hiring volume is falling across the sector — JOLTS information-sector openings are well below the 2023 average — but within the postings that do appear, the shape is this.

Two cautions apply. The first is that payroll data and posting data describe different things, a distinction this collection has emphasised elsewhere. Firms can advertise fewer 8-to-10-year roles while still *employing* them at previous headcounts. The second is that the LLM-extracted years-of-experience field is integer-rounded, and postings at bucket edges ("7-9 years", "8+", "10+") can land in adjacent buckets by construction. A small fraction of the 8-to-10 contraction may be boundary leakage.

What the data nonetheless do say, across two baselines and a within-firm panel, is that if any rung contracted in the AI-era rewriting of software-engineering job descriptions, it was the one just above the middle. The layoff discourse, on two successive iterations — entry-level, then mid-career — has been looking at rungs one to two notches lower than where the posting data register the contraction.

---

## Evidence block (not for publication)

1. **Five YOE buckets (pooled-2024 share, 2026-scraped share, pp change):** A 0-2 (7.31 → 11.28, +3.97); B 3-4 (16.04 → 17.98, +1.94); C 5-7 (35.15 → 33.28, −1.88); D 8-10 (17.36 → 14.04, **−3.32**); E 11+ (4.02 → 2.88, −1.13). `exploration/tables/journalist/yoe_bucket_shares.csv`; fact-check 10 independent re-derivation within 0.03 pp.
2. **Arshkon-only robustness: A +1.70, B +1.78, C +2.70, D −1.00, E +0.58.** D is the only bucket negative on both baselines. Fact-check 10 confirmed.
3. **Returning-cohort (n=2,109) intensification: D pooled −3.69 pp, arshkon −1.49 pp.** Data-investigator return; fact-check 10 confirmed.
4. **Absolute postings: 2026 labelled base = 25,815; D bucket fell by 3.32 pp × base ≈ 850 postings (pooled) or 1.00 pp × base ≈ 260 postings (arshkon).**
5. **T30 primary J3/S4 numbers reproduce.**
6. **YOE-LLM extraction accuracy: integer-rounded; boundary misclassification possible at 7/8 and 10/11.** Fact-check 10 flagged.

**Conventional-wisdom opponents:**
- Amodei/Brynjolfsson/SignalFire "entry-level vanishes" narrative (contested in Piece 01).
- Simon Willison, "The Seniority Rollercoaster" (December 2025) — broader labour-market argument for mid-career vulnerability, not a specific YOE-bucket forecast.
- Hosseini & Lichtinger, SSRN 5425555 — 62-million-résumé U-shape across *education tiers*, mapped here to YOE buckets only by analogy.

**Sensitivity verdict:** D (8-10) is the unique bucket falling on both baselines; its magnitude differs by a factor of more than three between pooled (−3.32 pp) and arshkon (−1.00 pp). The pooled baseline's larger magnitude is partly the asaniczka senior-share over-representation documented by T30. The more defensible magnitude is arshkon's. Posting-vs-payroll caveat is explicit. YOE integer-rounding caveat is explicit.

**Revision note:** Per critic feedback: made arshkon-vs-pooled asymmetry explicit (factor of 3+ in D magnitude), surfaced the baseline caveat in prose rather than burying it, reframed "5-7 flat on weaker baseline" to "near-stability or modest rise" with honest acknowledgment of the pooled decline. Removed two speculative mechanisms (scope inflation and 2019-2020-graduate-sweet-spot) as unsupported extrapolation. Disambiguated Willison (broader labour argument) from Hosseini-Lichtinger (education-tier resume study, different unit of analysis). Added absolute-count context (≈850 postings pooled, ≈260 arshkon). Retained the kicker but honoured the posting-vs-payroll distinction the collection emphasises elsewhere.
