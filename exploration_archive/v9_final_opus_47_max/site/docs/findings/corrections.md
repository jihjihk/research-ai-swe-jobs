# Measurement corrections

Two findings from an earlier pass were corrected under adversarial verification. Publishing the corrections openly is part of the methods argument: longitudinal studies of posting content need semantic validation on their patterns, or they will mistake instrument drift for real content change.

## Correction 1 — Management language is flat, not falling

### What the earlier analysis reported

Senior management density — how often managerial responsibilities are described in senior postings — looked like it was falling between 2024 and 2026. The earlier write-up interpreted this as a "senior shift away from management."

### What verification found

The regex pattern the earlier analysis was using for "management language" had a semantic precision of just 0.28 on a stratified 50-row hand-check. Precision of 0.28 means that on a random posting the pattern flagged, it was correctly identifying a management reference only 28% of the time. Every individual token broke too: `lead` at 0.12 precision, `team` at 0.08, `stakeholder` at 0.18, `coordinate` at 0.28, `manage` at 0.22. A stricter variant did slightly better at 0.55, but that version was still catching HR metadata like "contract-to-hire" or "how-we-hire/accommodations" under its `hire` sub-pattern (precision 0.07) and flagging "code review" or "peer review" as management under `performance_review` (precision 0.25).

A 0.28-precision pattern will confabulate a content shift. Whatever its false positives are most correlated with across periods determines the direction of the spurious signal.

### The corrected measurement

Verification published a rebuilt pattern:

```regex
\b(?:mentor(?:s|ed|ing)? (?:junior|engineers?|team(?:s)?|others|the team|engineering|peers|sd(?:e|es))|coach(?:es|ed|ing)? (?:team|engineers?|junior|peers)|direct reports?|headcount|hiring manager|hiring decisions?)\b
```

Revalidated on a fresh stratified 50-row sample: precision 0.98 to 1.00.

### Results under the corrected pattern

- Mid-senior management density: 0.039 to 0.038. Flat. Signal-to-noise ratio of 0.1, meaning the 2024-to-2026 move is roughly one-tenth the size of within-2024 variation (pure noise).
- Director-level density: 0.031 to 0.026. Essentially flat.
- Junior-side density: 0.000 to 0.004. Near-zero in both periods.
- An independent follow-up replication on a fresh calculation: 0.034 to 0.034.

### The corrected narrative

The senior shift is *toward orchestration* (working across systems and pipelines), *not away from management*. Orchestration density at the mid-senior tier rose by 0.67 mentions per 1,000 characters, with a signal-to-noise ratio of 5.6. The move is several times larger than within-period variation. So the specific claim that senior roles stopped describing management responsibilities collapses; the claim that senior content changed shape in a different direction survives.

This is the basis for a separate tier-B finding: "Management flat, not falling — a measurement correction."

## Correction 2 — Junior requirements-section shrink is classifier-sensitive (demoted)

### What the earlier analysis reported

Junior postings looked like their requirements-section character count had shrunk by around 5%, while responsibilities and benefits sections had grown. The earlier narrative was "length growth is boilerplate and responsibilities, not requirements."

### What verification found

Under the original classifier, J3 (postings asking for two or fewer years of experience — the study's primary junior definition) requirements went from 1,057 to 1,000 characters, a 5.4% drop. That reproduced exactly. But under an independently implemented simpler-regex classifier on the same J3 subset, the direction *flipped*: requirements grew by 88%. The measured direction depends on which classifier you use.

### What follow-up work added

A classifier-controlled analysis of posting authorship found that the requirements-share decline near-disappears when you restrict to postings that were probably not LLM-authored: the shrink goes from -0.025 to -0.002. Much of what looked like content change was authorship artifact: recruiters pasting LLM-generated filler into other sections, not the requirements section actually shrinking.

Classifier sensitivity extends from J3 to the aggregate tier as well. Under the original classifier, the aggregate period coefficient is -0.019 (shrink); under the simpler-regex classifier, it is +0.030 (growth). Both are statistically clean (p < 1e-13 with HC3-robust standard errors). They point in opposite directions.

### The corrected claim

"Junior descriptions densified on AI and tech-stack tokens; the net direction of the requirements section as a share of the full description is classifier-uncertain."

What survives direction-robust:

- Length growth is led by boilerplate: benefits +89%, legal +80%, responsibilities +49%.
- Requirements *content* changed — AI tokens, tech stack, orchestration all rose — even if the *section share* is classifier-dependent.

This is the basis for the demoted tier-D finding: "Junior requirements shrank — flagged, direction classifier-dependent."

## What these corrections teach, methodologically

Three takeaways worth carrying to any longitudinal posting-content study.

**Semantic precision validation is not optional.** A 0.28-precision pattern will invent a content shift in whatever direction its false positives happen to drift across periods. The fix is straightforward — stratified hand-audits per pattern, with a published precision threshold — but it does cost time.

**Classifier sensitivity is a headline quantity, not a footnote.** Any claim that rests on section-segmented text needs dual-classifier reporting, or ideally LLM-adjudicated spot-checks, before it graduates from exploratory to confirmed.

**Keep the validated artifacts.** The primary patterns used here — seven of them, each at 0.92 precision or higher — are saved with their validation results under `exploration/artifacts/shared/validated_mgmt_patterns.json`, so a future reanalysis can trace exactly what was measured and on what sample.

## Dig deeper

- The full adversarial audit: [verification notes](../evidence/verifications.md).
- The pattern re-validation on fresh samples: [source task](../evidence/tasks/T22.md).
- The corrected senior-cluster analysis: [source task](../evidence/tasks/T21.md).
- The pattern artifact: `exploration/artifacts/shared/validated_mgmt_patterns.json`.
