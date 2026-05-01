# Evidence and sensitivity.

The safest way to read the exploration is to treat every finding as a specification plus a sensitivity envelope. A result is only robust if it survives the checks that matter for that unit of analysis.

## Sensitivity framework

| Dimension | Primary reading | Why it matters |
|---|---|---|
| Aggregator exclusion | Include all rows unless the claim is about direct employers only. | Staffing intermediaries have different text and seniority patterns. |
| Company capping | Cap prolific employers for corpus-level text work. | Prevents a handful of employers from dominating frequencies and topics. |
| Seniority operationalization | Use `seniority_final`, then check the YOE proxy. | Explicit entry is conservative and junior claims are otherwise fragile. |
| Description text source | Use `description_core_llm` for text-sensitive work. | Raw `description` is recall-oriented and can be boilerplate-driven. |
| Source restriction | Compare arshkon vs scraped, and use pooled 2024 only as a sensitivity. | The historical snapshots are different instruments. |
| Within-2024 calibration | Compare the 2024 source gap to the 2024-to-2026 change. | If the temporal change is smaller than instrument noise, flag it. |
| SWE classification tier | Exclude `title_lookup_llm` when boundary sensitivity matters. | That tier has a higher false-positive risk. |
| LLM coverage | Report labeled rows and total eligible rows separately. | Coverage is thin in scraped text, so the sample is not the population. |
| Indeed cross-platform check | Use it as a sensitivity, not as the primary platform. | It is useful context, but LinkedIn remains the core analysis surface. |

## How to read the paper

- Strong findings usually survive the main sensitivities and the within-2024 calibration check.
- Spec-dependent findings should be labeled as such and should not be treated as headline proof.
- If a result changes by more than about 30 percent across the core sensitivity dimensions, the mechanism matters as much as the direction.
- Whenever explicit seniority and the YOE proxy disagree, the disagreement itself should be reported.
- The data can identify template drift and AI-like language, but it cannot prove who authored a posting. AI-authorship should stay in the hypothesis bucket unless interviews or other evidence resolve it.

## Limitations and open questions

- The cleaned-text coverage on scraped SWE is still thin.
- The section parser exposes a real template shift, which means later text claims must be careful about form versus content.
- AI-authorship is plausible but unproven; the exploration does not identify authors.
- Company composition matters, so pooled market averages can hide within-firm change.
- The benchmark for employer-worker divergence is not settled.
- The interview phase should focus on the contradictions surfaced by T22 and T23.
