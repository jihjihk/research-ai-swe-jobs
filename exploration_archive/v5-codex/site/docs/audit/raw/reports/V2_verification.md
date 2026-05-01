# Gate 3 Verification

I stopped the long-running recomputation jobs at the user's request and finished this memo with the checks already completed. I rederived the T16 company-overlap/decomposition pieces independently, partially checked the pooled-2024 entry reversal, and left the slower sampling-heavy checks explicitly incomplete.

## Verdict Table

| Headline | Wave 3 claim | Independent check | Verdict |
|---|---|---|---|
| T16 company decomposition and typology | 237 overlap companies at `>=3`, 122 at `>=5`; four cluster sizes `100 / 52 / 49 / 36`; AI/scope/length mostly within-company; entry trend instrument-dependent | Overlap counts match exactly. The within/between decomposition also matches in direction and magnitude. The pooled-2024 entry sign flip is verified. The exact 4-cluster split did not reproduce in my independent reruns. | Partial |
| T18 cross-occupation specificity | SWE AI-tool `1.98% -> 20.38%`, adjacent `2.45% -> 18.14%`, controls `1.25% -> 1.41%`; length/scope rise in all groups; boundary similarity `0.80-0.83` | The recomputation was stopped before the labeled-core aggregation and TF-IDF boundary sample finished. | Incomplete |
| T21 senior role evolution | strict management stable/slightly up; strict orchestration rises; cluster proportions `97.6 / 1.2 / 1.2 -> 78.5 / 15.3 / 6.3` | The run that would have rederived this was stopped before the aggregate and clustering output finished. The shared validation artifact still confirms broad management is noisy (`precision = 0.68`, `keep = false`). | Incomplete |
| T22 ghost / aspiration | AI hedge/firm ratio `0.73 vs 0.52` in section-filtered LLM core; `1.00 vs 0.80` raw sensitivity; broad management demoted | The shared validation artifact confirms broad management is demoted (`precision = 0.68`, `keep = false`). The AI aspiration ratio was not independently rederived before the recomputation stop. | Incomplete |
| T23 divergence | primary section-filtered AI-tool about `30.3%`, raw about `40.7%`, benchmark choice flips the claim | The benchmark-sensitivity recomputation was not completed before the stop. | Incomplete |

## What I verified

### T16 overlap counts and within-company decomposition

I independently rederived the overlap panel from `data/unified.parquet` and the shared cleaned-text/tech artifacts.

- Overlap companies with at least 3 postings in both periods: `237`
- Robust subset with at least 5 postings in both periods: `122`
- No-aggregator overlap panel: `205` at `>=3`, `99` at `>=5`

The main decomposition results also reproduced:

- `entry_final_share`: `0.0349 -> 0.0266` (`-0.0083`)
- `entry_yoe_share`: `0.0952 -> 0.1523` (`+0.0571`)
- `ai_any_share`: `0.0186 -> 0.1595` (`+0.1409`)
- `scope_any_share`: `0.1966 -> 0.3757` (`+0.1790`)
- `clean_len_mean`: `618.9 -> 1421.1` chars (`+802.2`)
- `tech_count_mean`: `2.94 -> 3.37` (`+0.43`)
- `requirement_breadth_mean`: `4.33 -> 5.46` (`+1.13`)
- `stack_depth_mean`: `2.11 -> 2.66` (`+0.55`)

The within-company components dominate the headline content changes:

- AI share within-company: `+0.1365` of `+0.1409`
- Scope share within-company: `+0.1609` of `+0.1790`
- Length within-company: `+592.8` chars of `+802.2`

That supports the core interpretation: the AI/scope/length shift is mostly a within-firm posting change, not just a company-composition artifact.

### T16 pooled-2024 entry reversal

I separately rederived the pooled historical baseline that includes `kaggle_asaniczka` in 2024.

- `entry_final_share`: `0.0140 -> 0.0283` (`+0.0144`)
- `entry_yoe_share`: `0.0719 -> 0.1479` (`+0.0760`)
- pooled overlap panel size: `572` companies at `>=3` in both years

That confirms the sign flip on explicit entry that the wave report flagged. It also confirms that the YOE proxy stays positive under pooling.

## What did not verify cleanly

### T16 cluster typology

This is the main place where my independent reruns diverged from the wave report.

The report claims four cluster sizes of `100 / 52 / 49 / 36`. My independent runs did not reproduce that split. Depending on the exact feature set I got materially different collapses, including a three-label merge pattern rather than the reported four-way split.

Interpretation:

- The company-overlap panel and the within/between decomposition are solid.
- The exact typology labels and cluster sizes are not yet robust enough to treat as a headline without a final spec lock.

My recommendation is to treat the typology as provisional until the feature set and clustering rule are frozen and rechecked.

### T18 boundary and cross-occupation DiD

I did not finish the labeled-core aggregation plus the TF-IDF boundary similarity sample before the recomputation was stopped.

The wave report claims:

- SWE AI-tool: `1.98% -> 20.38%`
- Adjacent AI-tool: `2.45% -> 18.14%`
- Control AI-tool: `1.25% -> 1.41%`
- Boundary cosine similarity: roughly `0.80-0.83`

I am not marking those as verified here because I did not independently complete the recomputation.

### T21 senior-role evolution

The strict-vs-broad management distinction is supported by the shared validation artifact:

- `management_broad` precision: `0.68`
- `management_broad` keep flag: `false`

That is enough to keep broad management in sensitivity only.

What I did not finish was the aggregate rederivation of the strict management / strict orchestration trend and the 3-cluster proportion output. Those remain incomplete in this verification pass.

### T22 ghost / aspiration

The shared validation artifact again supports the methodological call:

- broad management is noisy and should stay demoted

But I did not finish the independent recomputation of the AI hedge/firm ratios or the direct-vs-aggregator ghost-score comparison. Those remain incomplete.

### T23 divergence

I did not finish the benchmark-sensitivity recomputation. I therefore cannot independently confirm the `30.3%` vs `40.7%` posting-side rates or the benchmark-flip claim here.

## Pattern precision

Verified from the shared artifact:

- `management_broad` precision: `0.68`
- `management_broad` keep: `false`

This is the most important precision result for Wave 3, because it means any management narrative built on broad `lead/team/strategic/collaborate` language is too noisy to lead the paper.

## Alternative explanations to keep on the table

- Text-source coverage: the 2026 window relies heavily on `description_core_llm` coverage and raw fallback patterns differ from the 2024 snapshots.
- Section/template drift: section filtering and boilerplate stripping can shift apparent AI/scope rates materially.
- Company composition: pooled 2024 versus arshkon-only 2024 changes the explicit entry story.
- Aggregator effects: aggregators behave differently on aspiration and ghost-style scoring.
- Domain recomposition: the movement may be toward AI / infra / embedded mix shifts rather than a pure seniority story.
- Benchmark mismatch: posting-side AI language and worker-usage benchmarks are not directly interchangeable.
- Seniority-instrument disagreement: `seniority_final` and the YOE proxy do not always tell the same story on entry-level trends.

## Corrections needed before Wave 4

1. Freeze the T16 clustering spec before treating the typology as a result. The overlap panel and within-company decomposition are solid; the four-cluster split is not yet.
2. Finish T18 on the labeled core, then check the boundary cosine and the `title_lookup_llm` / aggregator sensitivities. Do not frame SWE-specific densification as final until that is done.
3. Keep `management_broad` out of the main story. The shared validation artifact already says why.
4. Treat T22 and T23 as calibration / validity sections until the independent recomputation is finished.
5. Keep the entry-level narrative anchored in the instrument split: explicit labels and YOE proxy do not agree cleanly enough to overstate junior collapse or junior growth.

## Bottom line

The most defensible Wave 3 result right now is still the company-level recomposition story: returning employers increased AI/scope/length within firms, and the pooled 2024 baseline confirms that the explicit-entry trend is instrument-dependent. The stronger occupation- and benchmark-level claims are not yet independently secured in this verification pass and should stay provisional until the next round of recomputation.
