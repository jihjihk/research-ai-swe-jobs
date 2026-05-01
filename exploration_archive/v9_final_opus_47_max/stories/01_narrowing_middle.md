---
title: "The narrowing middle that wasn't"
slug: 01_narrowing_middle
word_count_target: 500-650
status: revised_after_factcheck_critic
---

# The narrowing middle that wasn't

*Two years of AI have reshaped software hiring. The junior rung is not where it thinned.*

The prevailing story in Silicon Valley is that the junior software engineer is on the way out. In May 2025 Dario Amodei of Anthropic warned that half of all entry-level white-collar jobs could vanish within five years. Stanford's most-cited recent labour paper, published that August, dubs young software engineers "canaries in the coal mine" and reports a 13% employment decline, since late 2022, among 22-to-25-year-olds in AI-exposed occupations. A widely shared report from SignalFire puts Big Tech hiring of new graduates 25% below the prior year. The received wisdom is that artificial intelligence is eating the first rung of the ladder.

The posting data tells a different story about composition.

Among 68,000 software-engineering advertisements scraped from LinkedIn across 2024 and 2026, the junior share — the share of labelled postings calling for two or fewer years of experience — rose from 9.2% to 14.2%. That is five percentage points of increase on a base of nine points: a relative rise of roughly 55%. Under the more conservative arshkon-only baseline (the dataset's within-2024 calibration floor) the junior rise is smaller, 1.2 points, closer to the noise floor; under the 2,109-firm subset that posted in both 2024 and 2026 it intensifies to 6.2 points. Seven different junior-definition panels — label-based, years-of-experience-based, title-keyword-based — all move in the same direction. Five of six senior definitions move the opposite way. The inversion is not a quirk of one operationalisation.

Nor is the gap between junior and senior job descriptions narrowing, as an "AI collapses the skill premium" reading would predict. It is widening. The text similarity between junior and senior postings fell from 0.95 to 0.86 by one measure; a supervised model's ability to distinguish associate from mid-senior roles improved by 0.15 points of area-under-curve between the two periods. Whatever 2026 junior postings look like, they look less like 2026 senior ones. A strong relabelling hypothesis — that juniors and seniors are being flattened into a single "AI engineer" blob — is not supported.

Employment surveys and posting surveys measure different things. SignalFire and Brynjolfsson's Stanford team count hires from payroll records; this study counts the shape of live demand. Both can be true at once: hiring volume can be low while the ladder depicted in what hiring there is remains ordered. That is what the posting data shows. Within the same firms, senior roles gained more breadth of requirements than junior roles — a length-residualised breadth score rose by 1.97 items for five-or-more-years roles and by 1.43 for two-or-fewer, measured on the 724 firms present in both snapshots. Across nine of thirteen technical domains, the senior-junior breadth gap widened rather than narrowing.

Two implications follow. The first is that the contemporary layoff story has an implicit model of who the AI dividend accrues to: the senior IC is safe, the junior is replaceable. The data complicate this: the senior IC's job description now bundles orchestration, applied AI, platform engineering and vendor tooling under a tighter credential bar. The median years-of-experience asked by the newly emergent Applied-AI senior archetype is six, a year above the median of its senior neighbours. The role most identified with AI asks for more experience, not less.

The second is that the junior rung, the subject of most mainstream coverage, is not the rung the posting data most clearly reshapes. The clearer signal lies a tier above: the senior postings that absorb AI language are also the postings that widened in scope. If AI is redefining software-engineering work, it is doing so in the senior tier. The junior rung, on posting evidence, is holding its share.

---

## Evidence block (not for publication)

1. **J3 (YOE≤2) share 9.15% (2024 pooled) → 14.19% (2026 scraped), +5.04 pp, ~55% relative rise.** Fact-check 01 independent re-derivation from `unified_core.parquet`; matches T30 panel exactly.
2. **Arshkon-only baseline J3 +1.19 pp (near-noise-floor under arshkon-vs-asaniczka SNR 1.06).** Same fact-check.
3. **Returning-cohort J3 +6.17 pp (n=2,109 firms in both 2024 and 2026).** T37.
4. **7/7 junior definitions UP, 5/6 senior definitions DOWN.** T30 panel.
5. **TF-IDF junior↔senior cosine 0.946 → 0.863.** T15, V1 verified.
6. **Supervised AUC associate↔mid-senior +0.150.** T20; V2 replication +0.146.
7. **Within-firm breadth residualised S4 +1.97 vs J3 +1.43, n=724 returning firms.** T11, V1 decomposition.
8. **Applied-AI senior archetype median YOE 6.0 vs 5.0 neighbour clusters.** T34; fact-check 07 verified.

**Conventional-wisdom opponent:** Amodei interview (Axios, May 2025); Brynjolfsson et al., Stanford Digital Economy Lab, "Canaries in the Coal Mine?" (August 2025) — the piece credits Brynjolfsson for explicitly bracketing postings vs payroll; SignalFire State of Tech Talent (May 2025); Derek Thompson, Substack (June/August 2025). All four cite payroll/employment counts; the piece contrasts them with live-requisition composition.

**Sensitivity verdict:** Direction survives T30 12/13 definitions, T37 returning-cohort (amplifies), T29 low-LLM-authorship subset (80-130% content preservation), aggregator exclusion (<20% movement). Junior-share rise is near the within-2024 noise floor (SNR 1.06) on the pooled panel; it is more clearly separated on the returning-cohort subsample (+6.2 pp, CI [+3.21, +9.86]). Senior-share decline is above noise on all panels. Seniority-boundary sharpening has SNR >5 and is methods-convergent (TF-IDF + supervised classifier + continuous YOE interaction p<1e-44).

**Revision note:** Added in-text denominators ("from 9.2% to 14.2%", "55% relative rise"). Surfaced within-2024 noise-floor caveat in body. Softened "directly contradicted" to "is not supported." Named Brynjolfsson's own posting-vs-payroll bracketing. Moved the senior-tier observation to the body rather than leaving it in the closing kicker — it is the piece's substantive finding, not its ornament. Recalibrated kicker to "the junior rung is holding its share" rather than "the senior one to watch."
