# Wave 3 Retrospective: Changes Implemented

Date: 2026-04-06

## Changes made

### Core preamble (affects all agents)
1. **Asaniczka entry-level exclusion rule** — bolded warning about pooling asaniczka into entry-level metrics. Arshkon-only is the required primary 2024 baseline unless `seniority_llm` is available.
2. **Seniority ablation framework** — all entry-level analyses must report results under `seniority_llm` (if available), `seniority_native`, `seniority_final`, and `seniority_imputed`. Agreement strengthens; disagreement is a critical flag.
3. **LLM seniority anticipation** — preamble now checks for `seniority_llm` availability and notes it as the preferred uniform method.

### Analytical preamble (Wave 2+)
4. **Keyword indicator validation** (item 6 in text analysis hygiene) — sample 50 matches, flag <80% precision patterns. Prevents the +31pp → +4-10pp type of measurement error.

### Sensitivity framework
5. **Sensitivity dimension (c) updated** — references the full seniority ablation framework instead of a single primary/alt.

### Preprocessing schema
6. **Binary vs density text source guidance** — added to schema doc. Binary presence can use full text; density metrics should use cleaned text.

### Task specs
7. **T16** — added source-restriction sensitivity (pooled vs arshkon-only decomposition), within-company scope inflation with validated indicators, seniority ablation for entry metrics.
8. **T17** — added domain archetype geographic distribution step.
9. **T19** — added explicit asaniczka entry-level warning, note about re-computing with `seniority_llm` when available.
10. **T20** — added domain-stratified boundary analysis, seniority ablation.
11. **T21** — added pattern validation requirement, cross-seniority management comparison step.
12. **T22** — added validated patterns shared artifact output.
13. **T23** — added benchmark sensitivity range (50%/65%/75%/85% usage assumptions).

### Agent dispatch blocks
14. **Agent L** — added pattern validation warning and T09 archetype reference.
15. **Agent M** — added validated patterns artifact note.

## Key methodological decisions for future runs

- **LLM seniority (`seniority_llm`)** is the anticipated fix for the asaniczka problem. Once available, it should be the primary seniority variable and enables asaniczka to contribute a genuine 2024-01 entry-level baseline.
- **The seniority ablation (4 operationalizations)** replaces the simpler "primary + alt" approach. Entry-level trend direction DEPENDS on operationalization choice — this must be transparent.
- **Keyword precision validation** is now a standard practice, not a post-hoc correction. Every indicator should be validated at construction time.
