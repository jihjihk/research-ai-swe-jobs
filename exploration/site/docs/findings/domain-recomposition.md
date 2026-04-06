# The SWE Domain Landscape Recomposed: ML/AI Grew from 4% to 27%

## Headline

Unsupervised clustering of SWE job postings reveals that technology domain -- not seniority level -- is the market's dominant structural axis (NMI: domain 0.175 vs seniority 0.018, a **10x difference**). The largest temporal shift: ML/AI Engineering grew from 4% to 27% of SWE postings (+22pp), while Frontend/Web contracted from 41% to 24% (-17pp).

## Key numbers

| Archetype | 2024 share | 2026 share | Change |
|-----------|-----------|-----------|--------|
| Frontend/Web | 41% | 24% | -17pp |
| ML/AI Engineering | 4% | 27% | **+22pp** |
| Embedded/Systems | 26% | 19% | -7pp |
| Data/Analytics | 22% | 21% | -1pp (stable) |

- **Domain NMI:** 0.175 (strongest structural signal)
- **Seniority NMI:** 0.018 (weakest signal)
- A senior data engineer posting is more similar to a junior data engineer posting than to a senior frontend posting.

## Evidence

### 1. Method-robust clustering (T09)

BERTopic identified 92 fine-grained topics, reduced to 14 archetypes, with four macro-archetypes accounting for 81% of non-noise postings. The clustering is method-robust: Adjusted Rand Index >= 0.996 across alternative algorithms and parameter choices. This is not an artifact of a particular clustering method.

![Temporal change in archetype shares](../assets/figures/T09/temporal_change.png)

### 2. Domain dominates seniority in posting organization (T09)

The normalized mutual information analysis quantifies how much structure each dimension contributes:

- **Domain NMI = 0.175** -- postings within the same domain cluster tightly regardless of seniority
- **Seniority NMI = 0.018** -- knowing seniority tells you almost nothing about posting content

This has a critical implication: **all seniority-stratified analyses should also be domain-stratified.** Pooled junior-vs-senior comparisons mask domain-specific dynamics.

![Dimension alignment analysis](../assets/figures/T09/dimension_alignment.png)

### 3. ML/AI archetype character

The ML/AI Engineering archetype is characterized by: AI/ML model building, Python-centric stacks, cloud infrastructure, and emphasis on production ML systems. Its character is distinct from Data/Analytics (which emphasizes data pipelines and cloud services) and from traditional software engineering archetypes.

![PCA comparison of archetypes](../assets/figures/T09/pca_comparison.png)

### 4. Implication for the junior decline (T24, H1)

ML/AI engineering has a lower entry-level share than frontend/web engineering. The compositional shift from frontend (high entry share) to ML/AI (low entry share) could mechanically account for much of the aggregate junior decline. **H1 -- "domain recomposition drives junior decline" -- is the most testable new hypothesis** and should be the first analysis the formal phase runs.

## Text source sensitivity

The archetype distributions show 0.88 cosine similarity between LLM-cleaned text and rule-based text analyses. While high, this is not perfect. The ML/AI archetype share may shift modestly when LLM-cleaned 2026 text becomes available (currently 0% coverage for scraped data).

## Sensitivity

- Method-robust: ARI >= 0.996 across BERTopic parameter variations
- Inherently SWE-specific (domain archetypes are SWE constructs)
- Text source sensitivity: cosine similarity 0.88 (LLM vs rule-based)
- Sample sensitivity: robust to random subsampling

## Full analysis

- [T09: Posting Archetypes](../reports/T09.md) -- clustering methodology and results
- [T24: New Hypotheses](../reports/T24.md) -- H1 (domain drives junior decline)
- [T12: Text Evolution](../reports/T12.md) -- supporting text signal analysis
