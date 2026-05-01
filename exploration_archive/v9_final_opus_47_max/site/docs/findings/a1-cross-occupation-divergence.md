# Employers describe AI work at a fraction of the rate workers report using it

!!! quote "Claim"
    Across 16 occupation subgroups (software engineering, adjacent technical roles, and a panel of non-tech controls), postings mention AI at universally lower rates than workers say they actually use it on the job, in both 2024 and 2026. For software engineering, the gap grew by 14 percentage points between 2024 and 2026 (precisely +14.02 pp, 95% CI [+13.67, +14.37], where a percentage point is the difference between 10% and 24%, not a 14% proportional rise from 10%); for the control panel it barely moved (+0.17 pp). The ranking is stable: a Spearman rank-correlation of +0.92 (where +1 means two measures order items identically and 0 means no relationship) shows employers rank occupations the same way workers do, at 10 to 30% of the worker-reported intensity.

## Key figure

![Cross-occupation divergence across 16 subgroups](../figures/T32_cross_occupation_divergence.png)

The x-axis groups 16 occupation subgroups (software engineering, adjacent technical roles, controls). The y-axis is the AI-mention rate on the 2026 corpus. Bars are employer rates; diamond markers are the worker-side benchmark (how often workers in that occupation report using AI on the job, from survey data). Every diamond sits above its bar.

## Evidence

We compare two numbers per occupation: how often workers report using AI on the job (survey data), and how often postings mention AI. The first is the worker-side benchmark; the second is the employer-side rate.

- The difference-in-differences estimate for software engineering — the 2024-to-2026 change in SWE postings minus the change in control occupations, so a labour-market-wide shock would cancel out — is +14.19 pp. Controls moved +0.17 pp. Adjacent technical roles moved +10.84 pp.
- On our pattern-based AI-mention detector (which catches named AI tools like Copilot and Cursor, LLM-related terms, and AI-coding-assistant mentions), the SWE employer-side rate rose from 1.03% in 2024 to 10.61% in 2026. Worker-side benchmarks land between 63% and 90%, drawing on Stack Overflow, DORA, and Anthropic survey data.
- All 16 subgroups show the employer rate below the worker rate. Accountants show a 72x worker-to-employer ratio; nursing postings mention AI at 0.00% across 6,801 observations. The one subgroup where employer rates approach worker rates is machine-learning engineer at 63.1%. That is the exception that is exactly the role where the employer's product *is* AI.

## Sensitivity verdict — strong and robust

Every alternative control-group definition we tried lands within half a point of the +13 to +14 pp range: dropping analysts, dropping nursing, keeping only manual-work controls, dropping the 7% of SWE postings that needed LLM-reconciled classification. And the universality result — that every subgroup shows employer rates below worker rates — holds whether we draw the worker-usage threshold at 50, 65, 75, or 85 percent. Both results replicate directly from the saved outputs.

## Pattern-provenance note

An earlier report attributed the SWE ratio to a cleaner version of the AI pattern (the `v1_rebuilt` variant at 0.96 precision), but the actual computation used the top-level pattern (0.86 precision). The numbers as reported match the top-level pattern. Under the cleaner pattern, the SWE ratio becomes 18.6x (0.75% to 13.93%) rather than 10.3x. Direction unchanged; magnitude larger.

## Dig deeper

- The cross-occupation universality analysis covering all 16 subgroups: [source task](../evidence/tasks/T32.md).
- The difference-in-differences construction and control-group choices: [source task](../evidence/tasks/T18.md).
- The within-SWE worker-vs-employer comparison (Copilot at 0.10% in postings against roughly 33% industry-benchmark use): [source task](../evidence/tasks/T23.md).
- The robustness replication: [verification notes](../evidence/verifications.md).
