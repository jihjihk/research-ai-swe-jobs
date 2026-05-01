# Job descriptions got longer and more sectioned, but AI-authorship is only a hypothesis.

## Claim
Length growth is real, but it is partly a template shift: 2026 ads are far more explicitly sectioned than 2024 ads.

T13 is a validity result as much as a substantive one. Cleaned text length rises from roughly 1,969 chars in arshkon to 2,644 in scraped, but the bigger change is that core sections are nearly absent in 2024 and common in 2026. T12 then shows that cleaned-word shifts are messy unless the section structure is respected, which is why AI-authorship should stay a hypothesis rather than a conclusion.

## Evidence
- Only about 1.6% of arshkon docs and 4.5% of asaniczka docs have detected core sections, versus about 61-62% in scraped 2026.
- Requirements, responsibilities, preferred, and other structured sections account for a much larger share of the scraped corpus.
- The cleaned corpus is denser than the raw corpus; it is not a simpler one.
- The raw-vs-cleaned comparison in T12 is too imbalanced to stand alone, which is exactly why the section artifact matters.

## Figures
![T13_section_composition.png](../assets/figures/T13/T13_section_composition.png)

![T12_category_summary.png](../assets/figures/T12/T12_category_summary.png)

![description_length_overlap.png](../assets/figures/T05/description_length_overlap.png)

## Sensitivity and caveats
- The section parser misses unmarked historical structure, so 'unclassified' is a real residual category.
- Raw text remains acceptable only for recall-sensitive checks, not as the primary text surface.
- The historical-vs-scraped gap is larger than the within-2024 difference, so template drift is a first-order confound.
- Longer and more sectioned postings are compatible with AI-assisted drafting, but the exploration does not identify who wrote the posting.

## Raw trail
- [T13 report](../audit/raw/reports/T13.md)
- [T12 report](../audit/raw/reports/T12.md)
- [T05 report](../audit/raw/reports/T05.md)

## What this means
- The paper should treat posting structure as part of the measurement problem, not a nuisance detail.
- Any later text claim has to condition on sectioning or explicitly say it does not.
- AI-authorship belongs in the interview and mechanism discussion, not the headline result.
