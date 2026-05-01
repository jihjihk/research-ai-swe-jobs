# What the paper can claim

This page is a plain inventory: for each of the four research questions the study set out to answer, which claims the evidence supports, which need caveats, and which the data cannot support. It is organized by research question.

## Research question 1: is the entry rung narrowing, and are senior roles being redefined?

**Supported by the evidence:**

- The primary junior and senior definitions (J3, postings asking for two or fewer years of experience; S4, postings asking for five or more) move consistently across a thirteen-definition robustness panel. Twelve of thirteen definitions move in the same direction between 2024 and 2026.
- Senior claims use a pooled-2024 baseline with an arshkon-only co-primary, because the 2024 asaniczka dataset's lack of native entry-level labels inflates the pooled senior magnitude.
- Junior and senior job descriptions sharpened apart (became more distinguishable) rather than converged.
- Within the same firm, senior postings gained more skill breadth than junior postings.

**Needs a caveat:**

- The senior share change has to be reported as both pooled (-7.6 pp) and arshkon-only (-1.8 pp) magnitudes. The pooled magnitude is inflated because asaniczka has no native entry labels and therefore a higher senior baseline; the arshkon-only magnitude is the conservative read.

**Not supported:**

- Claims based on LinkedIn's native seniority labels. Between 2024 and 2026, LinkedIn appears to have re-tagged the same titles with 15 to 40 percentage points different labels even when the years-of-experience ask was flat. Platform relabeling drift makes the native labels unusable for cross-period work.
- The original "management language fell" claim. The regex pattern it rested on has only 28% precision, meaning most of its matches are false positives.

## Research question 2: how did tasks and requirements migrate across seniority levels?

**Supported by the evidence:**

- AI-tool mentions are universal across seniority levels. Copilot mentions at junior (4.6%) are almost identical to senior (4.1%); a three-way test of AI × seniority × period produces a null interaction (p = 0.66).
- Orchestration language concentrates at senior level specifically: 0.67 more mentions per 1,000 characters at mid-senior than elsewhere, well above the within-2024 noise floor.
- CI/CD-related asks at senior level rose by +20.6 percentage points between periods.

**Needs a caveat:**

- The "requirements-section share" direction is classifier-sensitive. Two different sentence classifiers give opposite aggregate signs on whether the explicit-requirements section of postings shrank. The site cites this as "narrative expansion dominating over requirements contraction" (a positive correlation of +0.35 between requirements characters and description length) rather than as a directional claim.

**Not supported:**

- "Requirements migrated downward" (from senior to junior). Contradicted by the null three-way interaction test.
- "Management language migrated downward". Contradicted by the rebuilt management pattern showing flat levels at all tiers.

## Research question 3: are employer descriptions of AI work diverging from worker-reported AI use?

**Supported by the evidence:**

- The difference-in-differences comparison (the 2024-to-2026 change in software-engineering postings minus the change in control occupations, so a labour-market-wide shock would cancel out) gives +14.02 pp AI-mention growth for software engineering, robust across four alternative control-occupation definitions.
- Employer-side AI mentions rose from 1.03% to 10.61%; worker-side reported AI use across triangulated surveys sits at 63 to 90%.
- Sixteen of sixteen tested occupation subgroups show a worker-employer gap, and the Spearman rank-correlation between worker and employer rankings is +0.92. Employers order occupations identically to workers, at 10 to 30% of the worker-reported intensity.

**Needs a caveat:**

- The worker-side benchmark triangulates three surveys (Stack Overflow, DORA, Anthropic) that each define "AI use at work" differently. The site reports across four definitional bands. The direction is positive under all four; the magnitude varies.
- The report-text pattern-provenance issue: the headline numbers are computed against the top-level AI pattern (0.86 precision), while the reports text sometimes refers to the audited v1_rebuilt version. Under v1_rebuilt the employer multiplier is 18.6x (0.75% → 13.93%) rather than 10.3x. Direction is unchanged; magnitude is larger under the audited pattern.

## Research question 4: what are the mechanisms behind the shift?

No single mechanism is supported. But the study ruled out three alternative explanations at the quantitative level:

**Alternative explanation 1: postings look different because recruiters started using LLMs to write them.** Ruled out. Content effects persist at 80 to 130% of their full-sample magnitude on the quartile of postings least likely to be LLM-authored. Length growth is roughly 52% LLM-mediated, but the content deltas are not.

**Alternative explanation 2: firms are lowering their hiring bars, not changing the work.** Ruled out. Across all hiring-bar proxies the correlation is small (|ρ| ≤ 0.28), and a manual check of 50 randomly-sampled 2026 postings found zero with explicit "loosening" language.

**Alternative explanation 3: firms are becoming more selective, not less.** Ruled out. Correlations between posting-volume change and content metrics are all small (|r| < 0.11). The one significant correlation (firms posting more postings write longer descriptions, r = +0.20) runs in the opposite direction of the selectivity prediction.

The remaining mechanism questions need interview evidence, which is a later research phase. The four open questions are:

1. What fraction of entry-level AI asks are genuine filters versus wish-list signalling?
2. Why did the requirements section shrink without explicit hiring-bar loosening?
3. Is "applied AI" a genuinely new role, a rebranding of ML engineer, or a skill-stack addition to existing senior roles?
4. Within senior-level postings specifically, how much of the 2024-to-2026 content shift is mediated by recruiter-side LLM authorship?

## Ten caveats a follow-on analysis should inherit

From the final consolidation memo:

1. **Pattern-provenance mismatch on the within-firm AI finding.** Re-derive the headline against the audited v1_rebuilt pattern.
2. **Same-title pair count is panel-dependent.** Report the range +10 to +13 pp; pre-register a specific panel construction.
3. **Requirements-section direction is classifier-sensitive.** Use two classifiers and report both.
4. **The disappearing-titles list is thin** (n = 2 to 11). Cite only as a qualified negative.
5. **Control-occupation definition sensitivity.** Pre-register a narrower or wider set than the study used.
6. **The asaniczka senior-baseline asymmetry is +7.1 pp.** Always cite both pooled and arshkon-only senior magnitudes.
7. **LinkedIn industry taxonomy drift.** No cross-period claim stratified by raw industry label is valid.
8. **Posting-age coverage is 0.9%.** Lifecycle analysis is not feasible.
9. **Benchmark sensitivity on the worker-employer gap.** Report the range, not a point estimate.
10. **ML Engineer cross-source classification gap.** Source-stratification is required and persists as a caveat.

## Six hypotheses the exploration deferred

These are neither rejected nor supported by the current data. Each needs something the exploration did not have:

| Hypothesis | Priority | What would be needed |
|---|---|---|
| Senior individual contributors as team multipliers | **high** | External firm hiring-panel data |
| Same-firm junior-entry drop combined with junior-YOE rise as a regime shift | low | A longer panel (2025 to 2027 data) |
| Sunbelt AI-adoption catchup | low | A formal event study on geographic uniformity |
| Staff-title redistribution | medium | Formalization of supporting descriptive findings |
| AI mentions as a coordination signal rather than a work description | medium | A mechanism test built around interviews |
| Recruiter-LLM bias specifically at senior level | medium | A senior-subset re-test of the LLM-authorship diagnostic |

Three additional questions surfaced late in the exploration:

- Does within-firm AI rewriting correlate with a 2024 digital-maturity index? (external data needed)
- Is the applied-AI concentration in financial services driven by compliance AI adoption? (17% of the main applied-AI cluster is FS; needs regulatory-context interviews)
- Are junior postings specifically hedging their AI asks? (data supports a qualitative read; formal test is deferred)
