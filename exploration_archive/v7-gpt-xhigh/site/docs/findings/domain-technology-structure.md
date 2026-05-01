# Domain and technology structure dominate seniority structure.

<p class="lead">The posting space is organized more by domain and technology than by junior/senior labels. This is why the paper should treat seniority as a measurement problem inside a broader technology shift.</p>

## Archetypes are mostly domain/technology archetypes

T09's chosen NMF-8 taxonomy aligns far more with technology/domain than seniority.

| Factor | NMI with NMF archetype labels |
|---|---:|
| Tech/domain signal | 0.205 |
| Top-30 company group | 0.098 |
| Source | 0.047 |
| Period | 0.043 |
| Seniority 3-level | 0.0069 |

Source: [T09](../raw/reports/T09.md).

## Representation variance tells the same story

T15 finds that archetype explains about 15-18x as much representation variance as seniority in embeddings and TF-IDF/SVD.

| Representation | Archetype eta-squared | Seniority eta-squared |
|---|---:|---:|
| Embedding | 0.111 | 0.007 |
| TF-IDF/SVD | 0.132 | 0.007 |

Source: [T15](../raw/reports/T15.md). V1 verifies the same direction with an independent technology-family proxy.

<div class="figure-frame">
  <img src="../assets/figures/T15/pca_structural_map.png" alt="PCA structural map from T15">
  <div class="figure-caption">The map is a communication aid. The tables drive the claim that domain/archetype structure is stronger than seniority structure.</div>
</div>

## Temporal movement

In T09's LLM-labeled subset, AI LLM Platforms rises from 3.1% of labeled 2024 rows to 17.4% of labeled 2026 rows. Product Backend Engineering and Cloud DevOps Infrastructure also grow. Enterprise Application Support, Java Dotnet Web Services, and Defense Clearance Systems shrink as corpus shares.

## Boundaries

Archetype and semantic findings describe LLM-cleaned rows with enough text. Scraped 2026 archetype coverage is about 30%, so these are not full-corpus scraped shares. NMF labels are descriptive market structure labels, not causal treatment categories.
