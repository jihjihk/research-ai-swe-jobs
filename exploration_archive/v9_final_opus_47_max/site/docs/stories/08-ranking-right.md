# The market got the pattern right

### *Employers are said to be panicking about artificial intelligence, or oblivious to it, or both. The data suggest they are doing something harder: falling behind on writing it down.*

The contemporary framing of the employer-worker AI divide is a framing of a gap. McKinsey's "Superagency" report, published in January 2025, headlines that the C-suite thinks 4% of its workforce uses generative AI while the actual figure is 13%. BCG's 2025 "AI at Work" survey reports that worker adoption is outpacing employer strategy. The World Economic Forum, in a January 2026 perception-gap piece, finds that leaders consistently underestimate their workers' AI use. MIT, writing with *Fortune* in August 2025, describes a "shadow AI economy" in which 78 – 90% of workers use unapproved tools their organisations do not see. The verdict, across sources, is familiar: employers are confused, panicky, or oblivious.

The posting data tells a less dismissive story, and a more specific one. It distinguishes between two quantities: the *level* at which a firm codifies AI in its requirements, and the *rank* of that codification relative to other occupations. The two are commonly treated as one thing in the commentary. They are not.

The software-engineering corpus was extended to sixteen occupation subgroups (covering software engineers, machine-learning engineers, data scientists, devops engineers, security engineers, accountants, civil engineers, nurses, financial analysts, and others) and two rankings were computed. The first ranked these occupations by the share of 2026 postings containing AI-requirement language. The second ranked them by worker-side AI usage from eighteen external surveys. The Spearman rank correlation between the two orderings is +0.92 (a Spearman rank-correlation: +1 means two measures order items identically; 0 means no relationship), with a Fisher-z 95% confidence interval of [+0.79, +0.97]. That means: employers' ordering (software engineers above devops engineers above data scientists above security engineers above accountants above nurses) is essentially identical to workers' own ordering of where they actually use AI. The level gap is enormous, but the pattern is not confused.

How large are the level gaps? They vary by occupation, and they vary by which worker-side benchmark is used. Accountants, on the most permissive worker benchmark (Thomson Reuters' 2024 "have ever tried AI" figure of 50%), register a 72-fold gap against the 0.7% of 2,910 accountant postings that codify AI. Using a stricter "daily use" benchmark, that gap compresses but remains order-of-magnitude. Nurses, across 6,801 postings, register a literal zero under the AI-strict pattern (the Wilson upper 95% confidence bound is 0.06%). Among software engineers, worker AI use runs at 63 to 90% depending on survey, against an employer codification rate that has risen from 1.0% to 10.6% between 2024 and 2026. Software engineering is the occupation where the level gap is smallest, around a seven-fold gap on the central benchmark, because it is the occupation where employers are codifying most quickly. The software sector is an outlier not in how much AI its workers use, but in how fast its employers are catching up on the formal record.

This distinction has not appeared in the mainstream coverage. The level-gap headlines (McKinsey, BCG, WEF, MIT/Fortune) imply, with remarkable consistency, that employers are *wrong* about AI: wrong in their perception, wrong in their strategy, wrong in their workforce planning. A rank correlation of +0.92 is hard to reconcile with that framing. Employers are not wrong about which jobs have been changed by AI. They are slow, by their own standards, to update the contractual document that describes each job, and slow for institutional reasons. The job description is legal, HR, and risk-approved text. It does not get rewritten when a tool gains traction. It gets rewritten when somebody has to post for a new hire.

The distinction matters because it implies different policy. If employers were wrong about AI exposure, the useful intervention would be education: surveys, dashboards, C-suite training of the sort that the management-consulting industry has sold since 2023. If employers are right about exposure but slow to codify it, the useful intervention is procedural: shorten the path from "workers use this tool" to "the job description reflects that they do". The first is a perception problem. The second is a latency problem. The data are more consistent with the second.

The tidy conclusion would be that employers are not panicking. The honest one is that the data do not show whether employers are panicking or not. What the data do show is that employers have the pattern right, and are writing the level down slowly. The next two years will establish whether the writing catches up.

---

??? note "Evidence and sources"

    **Headline numbers**

    - Spearman(worker-mid, employer-2026) = +0.9233 across 16 occupation subgroups; Fisher-z 95% CI [+0.79, +0.97]. T32 + fact-check 08 independent re-derivation; V2 replication +0.923.
    - Spearman(worker-mid, gap-2026) = +0.7094, p = 0.00208.
    - 16/16 subgroups: employer rate < worker rate in BOTH 2024 and 2026.
    - Accountant: 0.69% employer (n=2,910), 50% worker benchmark (Thomson Reuters "ever tried"), 72.75× gap. Under daily-use worker benchmarks, gap compresses.
    - Nurse: 0.00% employer rate on 6,801 postings (Wilson upper 95% CI = 0.06%).
    - SWE employer AI-strict: 1.03% (2024) → 10.61% (2026), ~7× factor gap vs worker 63 – 90%. Direction robust under top-level (0.86 precision) and V1-rebuilt (0.96 precision) patterns.
    - SWE DiD vs control = +14.02 pp, CI [+13.67, +14.37]; control drift +0.17 pp.
    - 18 worker-benchmark surveys from Stack Overflow, NBER, Anthropic, DORA, Thomson Reuters, CFA, ISC2.

    **Conventional-wisdom opponent**

    McKinsey "Superagency in the Workplace" (January 2025) — the "4% vs 13%" framing; BCG "AI at Work" (October 2025); WEF "AI Perception Gap" (January 2026); MIT / *Fortune* "Shadow AI Economy" (August 2025); Lightcast "Beyond the Buzz" (July 2025). All five treat employer-worker divergence as a level gap implying employer error.

    **Sensitivity verdict**

    Spearman +0.92 at n=16 has CI [+0.79, +0.97] — still strong. The 72× accountant gap is sensitive to benchmark choice (Thomson Reuters 50% "ever tried" is upward-biased; stricter benchmarks compress the gap to order-of-magnitude). Nurse 0.00% is literal on 6,801 postings but the Wilson upper CI is 0.06% — essentially zero. SWE DiD survives 5 alternative control-group definitions (V2 Phase E). Pattern-precision caveat on T23 / T16 (top-level 0.86 vs V1-rebuilt 0.96) flagged — direction robust under both.

---

## Related in Findings

- [Employers describe AI work at a fraction of the rate workers report using it](../findings/a1-cross-occupation-divergence.md) — the academic version of the same employer-worker divergence.
- [The software-engineering vs. control gap doesn't depend on which controls we picked](../findings/a6-did-robustness.md) — the +14 pp SWE-versus-control robustness test cited here.
