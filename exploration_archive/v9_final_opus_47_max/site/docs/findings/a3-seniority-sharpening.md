# Junior and senior job descriptions moved apart between 2024 and 2026, not together

!!! quote "Claim"
    Junior and senior SWE postings became *more* distinct between 2024 and 2026, under two independent methods. On a cosine similarity score (a 0-to-1 measure of shared vocabulary between two documents; closer to 1 means more overlap) using TF-IDF weighting (which emphasises distinctive words and downweights common filler), the junior-senior similarity dropped from 0.946 to 0.863. A supervised classifier's ability to separate the two tiers — measured by area under the curve, where 0.5 is chance and 1.0 is perfect — rose by 0.150 at the cleanest boundary. Two relabeling diagnostics confirm: 2026 entry postings do not look like 2024 senior postings. This falsifies a prior hypothesis that employers were quietly relabelling junior work as senior.

## Key figures

![AUC separating junior from senior, four boundaries, 2024 vs 2026](../figures/T20/auc_by_boundary.png)

Four pairs of bars, one for each seniority boundary (entry-to-associate, associate-to-mid-senior, mid-senior-to-director, and the aggregate), showing how well a supervised classifier separates the two tiers in 2024 vs 2026. The associate-to-mid-senior bar — the cleanest signal — rises from 0.743 to 0.893.

![Cross-period cosine similarity heatmap](../figures/T15_similarity_heatmap_tfidf.png)

A heatmap of cosine similarities between each seniority tier in each period. Cells comparing junior to senior across periods are darker than within-period cells, meaning junior-2026 and senior-2026 share less vocabulary than junior-2024 and senior-2024 did.

## Evidence

- Junior-senior cosine similarity on TF-IDF centroids fell from 0.946 to 0.863. A follow-up replication landed within 0.008 of that figure (0.950 to 0.871).
- A supervised logistic regression separating adjacent seniority tiers rose on two of three boundaries. The entry-to-associate AUC rose by 0.093. The associate-to-mid-senior AUC, the cleanest signal, rose by 0.150. The mid-senior-to-director AUC fell by 0.022. That softening is likely explained by the director-share shift into the new "applied AI" senior archetype (see [A5](a5-archetypes.md)).
- Treating years of experience as a continuous signal, the interaction with period is +0.273 items per YOE-year on requirement breadth (p < 1e-44). More experience in 2026 buys disproportionately more listed requirements than in 2024.
- Relabelling diagnostics using both the native LinkedIn seniority labels and the YOE-based definitions agree: 2026 entry postings are not sitting where 2024 senior postings used to sit.

## Sensitivity verdict — strong and methods-convergent

Two independent methods agree: unsupervised similarity says postings diverged; supervised classification says they got easier to tell apart. A follow-up replication reproduced the associate-to-mid-senior AUC lift at +0.146 (against the original +0.150). Removing YOE as a classifier feature entirely — to rule out the possibility that this is just a YOE story — actually strengthens the entry-to-associate lift to +0.194 and holds associate-to-mid-senior at +0.147. The sharpening is about more than YOE numbers.

One caveat. The entry-to-associate and mid-senior-to-director boundaries were flagged as sample-thin; treat those as direction-only, not precisely calibrated.

## What this falsifies

- The hypothesis that junior roles had been quietly relabelled as senior between 2024 and 2026: rejected.
- The hypothesis that seniority boundaries were blurring as AI compressed skill differences: rejected.

## Dig deeper

- The TF-IDF similarity analysis: [source task](../evidence/tasks/T15.md).
- The supervised AUC analysis: [source task](../evidence/tasks/T20.md).
- The relabelling diagnostics: [source task](../evidence/tasks/T12.md).
- The robustness replication: [verification notes](../evidence/verifications.md).
