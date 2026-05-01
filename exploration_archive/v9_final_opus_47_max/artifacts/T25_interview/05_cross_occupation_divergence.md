# Artifact 5 — Cross-occupation employer-worker AI divergence

**Source:** T32 cross-occupation divergence analysis.
**Figure:** `exploration/figures/T32_cross_occupation_divergence.png` (+ SVG). PAPER-LEAD CANDIDATE.

This artifact communicates the RQ3 finding: employers across 16 occupations under-codify AI skills relative to worker adoption, with direction universal in both 2024 and 2026 and magnitude SWE-concentrated.

---

## Headline numbers to present to interviewees

| Group | Subgroup | Employer ai_strict 2026 | Worker any-AI benchmark (2024-2025) | Gap |
|---|---|---|---|---|
| SWE | ml_engineer | **47.5%** | 85% | +37.5 pp |
| SWE | other_swe | **11.2%** | 84% | +72.8 pp |
| SWE-adjacent | data_scientist | 17.4% | 75% | +57.6 pp |
| SWE-adjacent | solutions_architect | 15.6% | 65% | +49.4 pp |
| SWE-adjacent | security_engineer | 5.1% | 40% | +34.9 pp |
| SWE-adjacent | network_engineer | 0.6% | 35% | +34.4 pp |
| Control | accountant | 0.7% | 50% | **+49.3 pp (72× ratio gap)** |
| Control | financial_analyst | 1.4% | 30% | +28.6 pp |
| Control | electrical_engineer | 0.09% | 30% | +30 pp |
| Control | nurse | **0.00%** | 15% | +15 pp |

## Key universality statistics

- **16 of 16 subgroups: employer rate < worker rate in BOTH 2024 and 2026.** Direction is universal.
- **Spearman(worker_mid, employer_2026) = +0.92.** Employers rank occupations identically to workers; the gap is in LEVEL, not rank.
- **Spearman(worker_mid, gap_2026) = +0.71.** Occupations with higher worker adoption have LARGER absolute pp gaps.
- **ML Engineer (adjacent, n=187 thin) is the ONLY case** where employer rate (63.1%) approaches worker rate (85%).

## Counter-prior finding: Nurse ai_strict = 0.00% on 6,801 postings

Zero strict AI mentions in 6,801 nursing job postings in 2026, despite:
- 31.5% of US hospitals report using GenAI (2024 hospital survey)
- 8-15% of nurses self-report using AI tools

The nursing occupation decouples formal job-posting AI-language from organizational AI-deployment entirely.

## Counter-prior finding: Accountant 72× ratio gap

Accountant worker any-mid = 0.50 (Thomson Reuters 2024: 44% daily AI use among firms using AI); employer ai_strict = 0.69%. The ratio gap is 72× — the largest ratio gap in the table, larger than SWE's 7× gap despite SWE having a larger absolute pp gap.

---

## Why this is a paper-lead candidate figure

The T32 finding (16/16 direction universality + Spearman +0.92) is the strongest single novelty signal in the exploration. It reframes the RQ3 divergence from "SWE-specific phenomenon" to "occupation-universal phenomenon with SWE-concentrated magnitude". The figure communicates both the universality and the magnitude-compression in one chart.

**Visual elements of T32 figure:**
- Bars: employer 2024 (light) + 2026 (dark) ai_strict rate per subgroup.
- Diamond markers: worker any-AI midpoint.
- X markers: worker daily-AI midpoint.
- Color coding by group (SWE red, adjacent blue, control grey).
- Sorted within each group by worker benchmark descending.

The visual is: **diamonds uniformly ABOVE bars across all 16 subgroups**. The universality pattern is visually immediate.

---

## Interview questions

### To labor-market informants across occupations

1. **(Show figure)** "This chart shows that across 16 occupations — from software engineering to nursing — employers list AI skills in job postings at rates FAR below what workers actually use AI. Why might this be? What's the function of a job posting if it doesn't match what the job actually requires?"

2. "The gap is UNIVERSAL: it holds for SWE (11% employer vs 84% worker), accountants (0.7% vs 50%), and nurses (0.00% vs 15%). If this is a universal phenomenon, what's the mechanism — inertia, cautious-signaling, legal/compliance, or something else?"

### To SWE informants

3. "Worker surveys say 84% of developers use AI daily or weekly. Job postings mention AI in 11% of SWE roles. Where does the gap come from? (a) Employers assume everyone uses AI and don't list it, (b) Employers don't want to commit to specific AI tools in a JD, (c) Hiring is lagging the actual workforce, (d) Something else?"

4. "GitHub Copilot has ~33% regular-use rate among developers. It appears in **0.10%** of SWE postings. Why is the most-adopted AI tool the LEAST-formalized requirement?"

### To accounting / finance informants (72× ratio gap)

5. "Thomson Reuters reported that 44% of accountants use AI daily or multiple times daily in firms that adopted AI. Your employers mention AI in under 1% of accountant postings. Is this a lag, a deliberate under-codification, or a compliance-caution pattern?"

### To ML Engineer / data scientist informants (the one near-convergent case)

6. "For ML Engineer roles specifically, employer AI-mention rate (63%) approaches worker AI-adoption (85%) — the only occupation where the gap is below 25 pp. What makes ML Engineer different? Does the role title itself carry the AI-signal, such that explicit AI language is redundant?"

### Cross-occupational

7. "If the employer-worker gap has the same DIRECTION across 16 occupations but the MAGNITUDE scales with how AI-native the occupation is perceived, this suggests employers know which occupations are AI-heavy and order them correctly — they just don't codify at those levels. What does this imply about job-posting function in a labor market? Is a JD a hiring filter, a candidate-attraction tool, or a signal to existing employees?"

## Mechanistic readings to probe

The exploration found three candidate mechanisms T32 cannot adjudicate quantitatively but RQ4 interviews can:

- **(a) Under-specification as employer caution.** Employers avoid committing to specific tool requirements that may constrain candidate pool or expose to legal risk (e.g., disability-discrimination if a tool is required).
- **(b) Coordination signal, not requirement.** AI-mentions are signals to candidates about employer sophistication, not specifications of daily work. Supports H_I hypothesis from T24.
- **(c) Generational lag.** Job postings are rewritten when HR has capacity; actual work shifts faster. Consistent with T37 "returning-cohort amplifies content-shifts" finding.

Interview protocol should reserve unstructured prompts for informants to raise mechanisms not anticipated above.
