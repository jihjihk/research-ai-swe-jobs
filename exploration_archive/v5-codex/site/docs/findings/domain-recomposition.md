# The market's latent structure is domain-first, and AI/LLM became a coherent skill ecosystem.

## Claim
The clearest structural result is that posting bundles line up with tech domains far more than with seniority or period, and the AI/LLM layer shows up as an organized stack rather than scattered keyword growth.

T09 is the best latent-structure result in the exploration. NMF with `k=15` is the first representation that cleanly separates backend, data, infra, embedded, frontend, mobile, AI/LLM workflows, and requirements/compliance. T14 then shows the AI/LLM layer is a coherent ecosystem embedded in the stack, not just a larger count of isolated keywords.

## Evidence
- T09 finds `tech_domain` NMI around 0.115-0.123, while seniority NMI is about 0.003.
- The AI/LLM workflows archetype is 91.5% 2026 in the full-corpus labels.
- T14 shows the underlying technology network now has a stable cloud/frontend backbone with a much denser AI/LLM layer.
- T15 rejects a clean junior-senior convergence story, which makes a seniority-first interpretation even weaker.

## Figures
![T09_embedding_maps.png](../assets/figures/T09/T09_embedding_maps.png)

![T14_cooccurrence_network.png](../assets/figures/T14/T14_cooccurrence_network.png)

![T15_similarity_heatmaps.png](../assets/figures/T15/T15_similarity_heatmaps.png)

## Sensitivity and caveats
- BERTopic is useful as a coarse comparator, but it collapses the structure too aggressively for downstream claims.
- The exact tech-domain decimal shifts a little under independent rebuilding, but the ordering does not.
- The raw-text sensitivity preserves the domain map, but it makes boilerplate more visible and should not replace the cleaned-text analysis.

## Raw trail
- [T09 report](../audit/raw/reports/T09.md)
- [T14 report](../audit/raw/reports/T14.md)
- [T15 report](../audit/raw/reports/T15.md)
- [Synthesis](../audit/raw/reports/SYNTHESIS.md)

## What this means
- Lead the paper with domain recomposition, not a broad seniority axis.
- Use NMF `k=15` as the downstream archetype map and keep BERTopic as a coarse check only.
- Treat AI/LLM as an ecosystem-level shift, not a keyword trend.
