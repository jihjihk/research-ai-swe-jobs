# Gate 2 Verification

## Headline checks

| Claim | Verdict | Independent check |
|---|---|---|
| T09 domain-first structure | Partially verified | `tech_domain` NMI rebuild = `0.1151` vs memo `~0.123`; `seniority_3level` NMI = `0.00306` matches, and `AI / LLM workflows` is `91.48%` 2026. |
| T11 credential stacking | Partially verified | `credential_stack_depth` SNR = `22.28` vs memo `~22.5`; `scope_count` SNR = `14.00`; `scope_density` is extremely strong in the same direction, but the broad AI count is noisier than the memo implies. |
| T13 template/sectioning shift | Verified | Section artifact reproduces `1.59%` arshkon, `4.69%` asaniczka, `60.34%` 2026-03, and `62.43%` 2026-04 docs with core sections. `description_core_llm` means are `1,982.8`, `2,294.5`, and `2,659.9` chars, still well above the 2024 baseline gap. |
| T14 AI/LLM technology signal | Verified | AI-tool / LLM share = `2.046%` arshkon vs `17.194%` scraped; no-aggregator sensitivity = `2.136%` vs `17.611%`. |
| T15 no junior-senior convergence | Verified | Embedding junior-senior similarity = `0.96493` arshkon vs `0.96183` scraped; TF-IDF = `0.91791` vs `0.86988`. The embedding change is inside the 2024 source gap. |

## Seniority cross-check

The seniority disagreement is real and should stay visible in Wave 3. On the current LinkedIn SWE frame, `seniority_final = entry` is `3.73%` in arshkon 2024-04 and `2.18%` in scraped 2026-04. The label-independent YOE proxy is much broader: `yoe_extracted <= 2` is `9.29%` in arshkon and `11.54%` in scraped 2026-04 overall. Arshkon native `seniority_native = entry` rows average `4.18` YOE, while scraped native entry rows average `2.34` YOE in 2026-03 and `2.39` in 2026-04. That is not a stable baseline.

## Keyword precision

The narrow AI-tool/LLM indicator is clean: the 50-match sample was 50/50 on-target, with terms like `llm`, `fine_tuning`, `claude`, `cursor`, `mcp`, and `langgraph`. Scope is also clean in sample: 50/50, dominated by `cross_functional`, `end_to_end`, `autonomous`, and `roadmap`. Strong management is usable: 50/50, with clear hits on `manage`, `mentor`, `supervise`, and `hire`.

Broad management is the bad one. Its 50-match sample is dominated by `team`, `collaborate`, `coordinate`, and `stakeholder`, which are generic cross-functional words, not stable management evidence. I would flag broad management as below 80% precision and keep it as a sensitivity only. The broader T11 AI count is also not the clean headline; use the narrower AI-tool signal for the paper.

## Alternative explanations

T09 can still be affected by how `tech_domain` is reconstructed from the tech matrix, but the ordering is stable even when the exact NMI shifts a bit. T11 can be inflated by longer, more sectioned postings and company concentration, but the company-capped and no-aggregator checks do not flip the sign. T13 is the strongest artifact check in the set: the document form changed enough that later text claims have to condition on sectioning. T14 could still be aspirational posting language rather than actual tool usage, and that should be stated plainly. T15 looks like standardization plus stronger period structure, not a true junior-senior collapse.

## What to correct before Wave 3

1. Soften the T09 tech-domain decimal if we cite the current artifact rebuild. The conclusion is intact, but `0.123` is a little sharper than the independent rebuild (`0.1151`).
2. Lead T11 with requirement breadth, stack depth, and scope. Treat broad management as noisy and treat the broad AI count as less important than the narrower AI/LLM tool signal.
3. Use the T13 section artifact as the text-structure baseline for downstream work. The exact section shares are stable enough to trust; the raw text surface is not.

## Outputs

- [Verifier script](/home/jihgaboot/gabor/job-research/exploration/scripts/V1_verification.py)
- [Key metrics](/home/jihgaboot/gabor/job-research/exploration/tables/V1/V1_summary_metrics.csv)
- [T13 section summary](/home/jihgaboot/gabor/job-research/exploration/tables/V1/V1_t13_section_summary.csv)
- [Keyword samples](/home/jihgaboot/gabor/job-research/exploration/tables/V1/V1_keyword_precision_samples_with_titles.csv)
- [AI-tool sample](/home/jihgaboot/gabor/job-research/exploration/tables/V1/V1_ai_tool_precision_samples.csv)
