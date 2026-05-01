# What did not change

The easier version of this exploration would be to declare every observed shift a finding. Discipline required rejecting several pre-exploration hypotheses and flagging headline numbers that cannot be cited unqualified.

## Junior share is baseline-contingent

Under arshkon-only baseline (the fairer comparison for native entry labels): **J1/J2 −0.6 pp, J3/J4 +2.3-3.4 pp**.

Under pooled-2024 baseline: **J1/J2 +1.6 pp, J3/J4 +3.1-4.0 pp**.

T05 found SNR < 1 on every junior metric within-2024 — the noise floor (arshkon vs asaniczka within-2024) exceeds the cross-period effect. Asaniczka has **zero** native entry labels, which drags pooled-2024 entry share toward zero.

The junior-share direction flips between J1 (label-based) and J3 (YOE-based) under the same baseline:

| Def | arshkon → scraped | Interpretation |
|---|---|---|
| J1 = `seniority_native = entry` | +3.73 % → +3.12 % | flat-down |
| J2 = `seniority_final ∈ {entry, associate}` | −0.65 pp within-co | flat |
| J3 = `yoe ≤ 2` | +7.2 pp overall | 95% between-company |
| J4 = `yoe ≤ 3` | +3.4 pp | between-co dominated |

T06 shows 95% of the J3 rise is between-company — tech-giant intern pipelines (Google J3=52%, Walmart J3=74%, Microsoft J3=39%) that are largely absent from Kaggle 2024 snapshots. Per V1.5, LLM-frame coverage preferentially selects junior postings, inflating any labeled-subset junior direction by 1-2 pp.

**The defensible statement:** junior-share direction cannot be cited as a headline. It is a measurement-regime story (platform taxonomy + LLM-frame selection + tech-giant composition).

## Title concentration is stable

Unique titles per 1K postings: **554 → 507 → 533** across arshkon / asaniczka / scraped. The title space did not fragment. What changed:

- **Staff doubled** (2.6% → 6.3%) — senior-tier internal redistribution.
- **`senior` share flat.**
- **Disappearing titles are legacy-stack**: `java architect`, `drupal developer`, `devops architect`, `senior php developer`, `sr. .net developer`.
- **Emerging titles** bundle AI tooling (`AI engineer`, `ML platform engineer`, `applied ML engineer`).

## Junior-senior convergence is rejected

The "AI blurs the junior/senior line" hypothesis was cleanly rejected — see [boundary sharpening](boundary-sharpening.md). All three adjacent boundaries gained AUC.

## Anticipatory over-specification is rejected

The "employers over-specify AI because they anticipate adoption" hypothesis was inverted — see [RQ3 inversion](rq3-inversion.md). Employers trail workers by 15-30 pp.

## Period ~180× seniority in embedding space is wrong

T15 initially reported period is ~180× more discriminating than seniority in embedding space. V1 re-derived on the full corpus and found **~1.2× centroid-pairwise**. The defensible ratio is **NMI 1.9× period/seniority** and **NMI 8.6× domain/period**. The corpus organizes by domain, second by period, third by seniority — but not by wildly different ratios.

## Geographic AI displacement is rejected

T17 found r(Δ_ai_strict, Δ_entry_j2) = −0.11 across 26 metros (p = 0.60). The spatial "AI exposure → junior displacement" narrative is cleanly rejected. Top AI surges are Sunbelt finance/healthcare (Tampa Bay, Atlanta, Charlotte), not tech hubs.

## Posting-level aspirational padding is rejected

T22 found posting-level LLM ghost risk-ratio 0.98 for ai_strict vs non-AI postings — **not elevated**. AI sentences are modestly more hedged (matched-share +0.24) but the posting as a whole is not more aspirational. Reframe as "emerging-demand framing," not "aspirational padding."

## Seniors changed more than juniors

This is a **positive** finding that surprised the pre-exploration framing. T12 cosine shift: entry 2026-vs-2024 = 0.953, senior = 0.942. Seniors moved more in content than juniors. Every pre-exploration framing assumed junior was the dynamic cell; the data says senior was.

Combined with boundary sharpening (+0.134 AUC yoe-panel), the story is: **juniors stayed recognizably junior; seniors transformed.**

## Links to raw

- [T05 — cross-dataset comparability](../raw/wave-1/T05.md)
- [T06 — company concentration](../raw/wave-1/T06.md)
- [T10 — title taxonomy evolution](../raw/wave-2/T10.md)
- [T12 — open-ended text evolution](../raw/wave-2/T12.md)
- [T15 — semantic similarity](../raw/wave-2/T15.md)
- [T17 — geographic structure](../raw/wave-3/T17.md)
- [T22 — ghost forensics](../raw/wave-3/T22.md)
- [T30 — seniority definition panel](../raw/wave-1/T30.md)
- [V1 verification](../raw/verification/V1_verification.md)
</content>
</invoke>