# A decline that wasn't

### *A plausible-looking labour-market finding collapsed under independent scrutiny. Many plausible-looking findings in the same genre may be waiting for the same treatment, though nobody has yet performed it.*

The first-pass version of this study reported a tidy fact: the density of managerial language in senior software-engineering job descriptions fell between 2024 and 2026. It was the sort of finding a labour economist might happily write up. The shift from "management" to "orchestration" would fit an AI-era story; the senior engineer, on this reading, was becoming an architect of systems rather than of people.

The finding did not survive.

An adversarial check, of a kind the published labour literature rarely performs, tested the regular-expression patterns used to measure "management" against randomly sampled postings, to see whether matches were actually about management. The result was blunt. The broader pattern (a disjunction of keywords including "lead", "team", "stakeholder", "coordinate", "manage") hit 28% precision on a fifty-posting stratified sample. All five broad extensions failed the 80%-precision bar: "lead" at 12%, "team" at 8%, "stakeholder" at 18%, "coordinate" at 28%, "manage" at 22%. At 28% overall, barely more than one in four matches describes actual managerial responsibility. The rest describe software components ("manage state"), generic professional filler ("lead contributor"), or stakeholder-communication language in individual-contributor roles.

The stricter pattern scored 55% precision. Its worst sub-patterns were "hire" (7% precision; overwhelmingly matching "contract-to-hire" boilerplate) and "performance review" (25%, mostly "code review" or "peer review" in quality-assurance contexts). Under a rebuilt pattern that insists on co-occurrence with "engineers", "junior", "team" or "peer" (a pattern validated at 98% precision on a fifty-posting sample and 100% precision on an independent thirty-posting re-sample), the 2024-to-2026 trend in senior managerial density is flat: from 0.039 per thousand characters to 0.038, a signal-to-noise ratio of 0.1 (the 2024-to-2026 move is one-tenth the size of the within-2024 variation, which is to say, invisible).

The finding was not that management was in decline. It was that a particular set of keywords, some of which have drifted in their software-industry meaning faster than others, was slightly less common in 2026 text.

The question this raises is not whether other recent posting-content findings are wrong. It is whether a reader can tell.

The most cited paper in the genre, David Deming and Lisa Kahn's 2018 study of skills in 45 million job advertisements, relies on keyword lists for "cognitive", "social", "character", "people-management" and other latent categories. The paper does not report a precision check. Nor does Acemoglu, Autor, Hazell and Restrepo's widely influential 2022 AI-vacancy paper, whose flagship claim (that AI-exposed establishments reorganise their hiring) uses an AI-related keyword list that is not validated against sampled postings. Lightcast's 2024 and 2025 "Global AI Skills Outlook" reports, which anchor Brookings' mainstream AI-labour coverage, publish headline counts of tens of thousands of AI postings but, in their public methodology, do not disclose an accuracy test of the pattern that generates those counts. The reader has no way of knowing whether those patterns are at 98% precision or 28%.

The exception, and it is a chastening one, is Stephen Hansen, Peter Lambert, Nicholas Bloom, Stephen Davis, Raffaella Sadun and Bledi Taska's work on remote-work measurement. That team explicitly validated their remote-work dictionary against thirty thousand human-coded postings, found dictionary precision was lower than the authors had assumed (because "this position allows remote work" and "this position involves working on remote sites" match the same pattern), and replaced the dictionary with a language-model classifier. The failure mode they documented for "remote" is the same failure mode this study documented for "hire" and "performance review": a word whose software-industry meaning does not match its dictionary meaning. How many others are there? Nobody has checked.

That, not the "management declined" correction, is the story. The discipline's current standard (cite the keyword list in the appendix, and write as though its precision were one) generates findings that cannot be falsified by a reader. Researchers are not being sloppy; the tooling for validation has only recently become cheap. A modern language model will score a fifty-posting precision sample in under an hour, for a few dollars. It should be a reviewer's demand, not an author's option.

Until then, readers of the labour press should treat every new headline about "rising soft skills" or "falling management" or "emergent AI demand" the way Hansen's group came to treat their "remote work" dictionary. The finding is what the pattern catches. The test, still often un-run, is whether what it catches is what it says.

---

??? note "Evidence and sources"

    **Headline numbers**

    - mgmt_broad pattern precision = 0.28. Five of five broad extensions below 0.80: lead 0.12, team 0.08, stakeholder 0.18, coordinate 0.28, manage 0.22. V1_verification.md §2; fact-check 05 verified all sub-patterns.
    - mgmt_strict pattern precision = 0.55. Worst sub-patterns: "hire" 0.07, "performance review" 0.25.
    - Rebuilt mgmt_strict_v1_rebuilt precision = 0.98 on T21 50-row sample; 1.00 on T22 independent 50-row re-sample; 1.00 on V2 30-row re-sample. `exploration/artifacts/shared/validated_mgmt_patterns.json`.
    - Senior management density under v1_rebuilt: mid-senior 0.039 → 0.038 (flat, SNR 0.1). T21; fact-check 05 verified.

    **Prior-art genre sources (all with no public precision validation)**

    - David Deming & Lisa Kahn, "Skill Requirements Across Firms and Labor Markets" — *Journal of Labor Economics*, 2018. 45-million-posting study.
    - Daron Acemoglu, David Autor, Jonathon Hazell, Pascual Restrepo, "AI and Jobs: Evidence from Online Vacancies" — *JOLE*, 2022.
    - Lightcast, "Global AI Skills Outlook" — 2024 and 2025 reports.

    **Counter-example (validation done correctly)**

    - Stephen Hansen, Peter Lambert, Nicholas Bloom, Stephen Davis, Raffaella Sadun, Bledi Taska, NBER working paper 31007 (2023 / 2025) — validated remote-work dictionary against 30,000 human-coded postings; replaced with LLM classifier.

    **Sensitivity verdict**

    The 28% precision is a point estimate from a single 50-posting stratified sample; true precision could be ± 5 pp on repeated sampling. The rebuilt pattern has been validated at 0.98 – 1.00 on three independent samples. The "management flat" finding is robust across the T30 seniority panel and to aggregator exclusion. The methods-as-news claim — that published posting-analysis literature often omits precision validation — is an empirical claim about four named sources, not a universal indictment. The piece explicitly does not claim the prior-art findings are wrong; it claims the reader cannot tell.

---

## Related in Findings

- [Measurement corrections](../findings/corrections.md) — the academic note on the same pattern-validation correction.
- [Pattern validation](../methodology/pattern-validation.md) — the validated-patterns artifact referenced here.
