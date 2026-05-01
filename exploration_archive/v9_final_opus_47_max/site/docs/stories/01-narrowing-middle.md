# The narrowing middle that wasn't

### *Two years of AI have reshaped software hiring. The junior rung is not where it thinned.*

The prevailing story in Silicon Valley is that the junior software engineer is on the way out. In May 2025 Anthropic's CEO Dario Amodei warned, in an *Axios* interview, that half of all entry-level white-collar jobs could vanish within five years. Stanford's most-cited recent labour paper, by Erik Brynjolfsson and colleagues and published that August, dubs young software engineers "canaries in the coal mine" and reports a 13% employment decline, since late 2022, among 22-to-25-year-olds in AI-exposed occupations. A widely shared May 2025 report from the venture-capital firm SignalFire puts Big Tech hiring of new graduates 25% below the prior year. The received wisdom is that artificial intelligence is eating the first rung of the ladder.

The posting data tells a different story about composition.

The study draws on roughly 68,000 software-engineering LinkedIn postings from 2024 and 2026. Among postings that explicitly named a seniority or a years-of-experience floor, the junior share (postings asking for two or fewer years of experience, the study's primary junior definition) rose from 9.2% to 14.2%. Five points on a base of nine is a 55% proportional rise, a number that deserves a caveat. The 2024 pool blends two Kaggle datasets; within-2024 noise, measured by comparing them against each other, is almost as large as the 2024-to-2026 move, so the pooled rise reads as directional, not precisely calibrated. On the 2,109 firms that posted in both periods, the junior share rose by 6.2 points, clearly above the noise. Seven different junior-definition panels (label-based, years-of-experience-based, title-keyword-based) all move in the same direction. Five of six senior definitions move the opposite way. The inversion is not a quirk of one operationalisation.

Nor is the gap between junior and senior job descriptions narrowing, as an "AI collapses the skill premium" reading would predict. It is widening. The text similarity between junior and senior postings fell from 0.95 to 0.86 by one measure (a cosine similarity score: 0 to 1, closer to 1 means more shared vocabulary). A supervised model's ability to distinguish associate from mid-senior roles improved by 0.15 points of area-under-curve between the two periods (AUC: a classifier-separation score where 0.5 is chance and 1.0 is perfect). Whatever 2026 junior postings look like, they look less like 2026 senior ones. A strong relabelling hypothesis, that juniors and seniors are being flattened into a single "AI engineer" blob, is not supported.

Employment surveys and posting surveys measure different things. SignalFire and Brynjolfsson's Stanford team count hires from payroll records; this study counts the shape of live demand. Both can be true at once: hiring volume can be low while the ladder depicted in what hiring there is remains ordered. That is what the posting data shows. Within the same firms, senior roles gained more breadth of requirements than junior roles. A length-residualised breadth score (the number of distinct skill or tool requirements per posting, adjusted for posting length) rose by 1.97 items for roles asking five or more years of experience and by 1.43 for two or fewer, measured on the 724 firms present in both snapshots. Across nine of thirteen technical domains, the senior-junior breadth gap widened rather than narrowing.

The contemporary layoff story has an implicit model of who the AI dividend accrues to: the senior IC is safe, the junior is replaceable. The data complicate this. The senior IC's job description now bundles orchestration, applied AI, platform engineering and vendor tooling under a tighter credential bar. The median years-of-experience asked by the newly emergent Applied-AI senior archetype is six, a year above the median of its senior neighbours. The role most identified with AI asks for more experience, not less.

So the junior rung, the subject of most mainstream coverage, is not the rung the posting data most clearly reshapes. The clearer signal lies a tier above: the senior postings that absorb AI language are also the postings that widened in scope. If AI is redefining software-engineering work, it is doing so in the senior tier. The junior rung, on posting evidence, is holding its share.

---

??? note "Evidence and sources"

    **Headline numbers**

    - J3 (YOE ≤ 2) share: 9.15% (2024 pooled) → 14.19% (2026 scraped), +5.04 pp, ~55% relative rise. Fact-check re-derived from `unified_core.parquet`; matches T30 seniority-definition panel exactly.
    - Arshkon-only baseline: J3 +1.19 pp (near-noise-floor under arshkon-vs-asaniczka SNR 1.06).
    - Returning-cohort (n=2,109 firms): J3 +6.17 pp.
    - 7 of 7 junior definitions up, 5 of 6 senior definitions down on T30 panel.
    - TF-IDF junior↔senior cosine 0.946 → 0.863 (T15); supervised AUC associate↔mid-senior +0.150 (T20); V2 replication +0.146.
    - Within-firm breadth residualised S4 +1.97 vs J3 +1.43, n=724 returning firms (T11 / V1).
    - Applied-AI senior archetype median YOE 6.0 vs 5.0 neighbour clusters (T34; fact-check 07).

    **Conventional-wisdom opponent**

    Dario Amodei interview (*Axios*, May 2025); Erik Brynjolfsson et al., "Canaries in the Coal Mine?", Stanford Digital Economy Lab (August 2025); SignalFire *State of Tech Talent* (May 2025); Derek Thompson Substack (June / August 2025). All four cite payroll or employment counts; the piece contrasts them with live-requisition composition. Brynjolfsson explicitly brackets posting-versus-payroll measurement in the paper, and is credited in the body.

    **Sensitivity verdict**

    Direction survives T30 (12/13 definitions), T37 returning-cohort (amplifies), T29 low-LLM-authorship subset (80-130% content preservation), aggregator exclusion (<20% movement). The junior-share rise is near the within-2024 noise floor (SNR 1.06) on the pooled panel; it is more clearly separated on the returning-cohort subsample (+6.2 pp, CI [+3.21, +9.86]). Senior-share decline is above noise on all panels. Seniority-boundary sharpening has SNR > 5 and is methods-convergent (TF-IDF + supervised classifier + continuous YOE interaction, p < 1e-44).

---

## Related in Findings

- [Junior and senior job descriptions moved apart between 2024 and 2026, not together](../findings/a3-seniority-sharpening.md) — the academic version of the junior-senior divergence.
- [Every tier asks for broader skills than in 2024; senior roles gained more breadth than junior](../findings/a4-scope-inflation.md) — the within-firm breadth decomposition cited here.
