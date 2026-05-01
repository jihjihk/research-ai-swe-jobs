# Exploration Synthesis

The exploration now supports a different paper than the one that launched it. The strongest story is **differential densification under template drift**: SWE postings became more AI/LLM-centered, more multi-constraint, and more standardized in form, but the change is heterogeneous across companies and only partly SWE-specific. Junior findings remain instrument-dependent, management decline is not supported, and the employer-worker divergence claim is benchmark-sensitive rather than robust. The text evidence supports template drift and denser requirement language, not AI-authorship attribution.

## Headline Claims, Stated Precisely

- Job descriptions got longer and more explicitly sectioned across SWE, adjacent technical roles, and controls. That is evidence for template drift and denser requirement language, not proof that AI authored the postings.
- Requirement stacking is strongest in SWE, especially AI/tool language and scope or credential bundling, but the length/scope/template shift is broader than SWE.
- Senior postings did not move from management to orchestration. Strict management stays stable or slightly up, while AI/tool orchestration grows around it.
- Explicit `entry` labels are conservative lower bounds. Low-YOE postings remain and move differently, so an entry collapse is not established.
- The latent market structure is domain or technology first, not seniority first.
- AI/LLM is now a coherent ecosystem in the posting language, not just scattered keywords.
- Returning employers changed their own postings. Within-company AI/scope/length growth is stronger than the exact provisional company typology.
- The simple dramatic stories do not survive: no clean junior collapse, no clean management decline, and no robust employer-worker divergence.
- Caveat: T16 exact cluster labels are provisional; T18, T21, T22, and T23 headline numbers were not all independently rederived in V2.

## Data Quality Verdict per RQ

| RQ | Safe to use now | Main caveats | Analysis-phase posture |
|---|---|---|---|
| RQ1 employer-side restructuring | Yes, but only with `seniority_final` + YOE proxy + company decomposition | Explicit entry is conservative; company composition and posting format matter; `seniority_native` is diagnostic only | Lead with company strategy, explicit-entry vs YOE divergence, domain-first recomposition, and within-company change |
| RQ2 task and requirement migration | Yes, especially for breadth, scope, AI/LLM, and section-level change | Cleaned text coverage is thin in scraped 2026; raw text is recall-oriented only; template drift is a real confound; AI-authorship is not established | Reframe as requirement stacking and posting-form change, not simple downward migration |
| RQ3 employer-requirement / worker-usage divergence | Only as calibration | Benchmark choice flips the sign; the posting and worker objects are not interchangeable | Keep as benchmark-sensitive validity analysis, not headline proof |
| RQ4 mechanisms | Yes, but as interview-led interpretation | Quantitative evidence alone cannot separate real work, template drift, aspirational language, and AI-assisted drafting | Use interviews to adjudicate the contradictions surfaced by T22/T23/T21/T03 |

## Recommended Analytical Samples

| Analysis type | Recommended sample | Key columns | Notes |
|---|---|---|---|
| Seniority / junior share | LinkedIn only, `source_platform='linkedin'`, `is_english=true`, `date_flag='ok'` | `seniority_final`, `yoe_extracted`, `seniority_final_source`, `period`, `company_name_canonical` | Always show explicit-entry and YOE proxy side by side; use arshkon native only as a diagnostic |
| Text / requirement change | LinkedIn SWE rows with `llm_extraction_coverage='labeled'` | `description_core_llm`, `period`, `company_name_canonical`, `llm_extraction_coverage` | Raw `description` only for binary presence or recall-sensitive sensitivity checks |
| Section-level text change | LLM-cleaned core sections from T13 | section spans + cleaned text | Use as a document-form control, not as a full 2024-vs-2026 balance test |
| Archetype / domain | SWE rows with `swe_archetype_labels.parquet` | archetype label, `period`, `company_name_canonical` | NMF `k=15` is the downstream labeler; BERTopic is only a coarse comparator |
| Company trajectories | Returning-company panel with `>=3` postings both periods; `>=5` as robustness cut | company-level deltas | The overlap decomposition is verified; the exact cluster typology is still provisional |
| Cross-occupation boundary | SWE, SWE-adjacent, control, company-capped | `is_swe`, `is_swe_adjacent`, `is_control`, `period` | Exclude `title_lookup_llm` for the strictest boundary-sensitive sensitivity |
| Ghost / aspiration | Section-filtered LLM core with raw-text sensitivity | hedge/firm ratios, template saturation, `ghost_assessment_llm`-derived scores | Treat as a ranking and validation device, not a latent truth variable |
| Benchmark comparison | T23 core vs raw + public benchmark bands | posting-side AI rates, benchmark rates | Never reduce this to one benchmark number |

## Seniority Validation Summary

The seniority story is the paper’s main measurement constraint.

- `seniority_final` is a strict lower-bound junior label. When it says `entry`, the row is usually genuinely junior by YOE.
- But `seniority_final` misses many junior-like rows. The YOE proxy is materially broader in every source and period.
- The direction of the entry-level trend depends on the instrument:
  - explicit `seniority_final = entry` is tiny and often flat or declining
  - `yoe_extracted <= 2` is much larger and often rises
  - pooled 2024 changes the sign of the explicit-entry comparison in T16
- `seniority_native` is diagnostic only. Asaniczka has zero native entry labels, and arshkon native `entry` rows are not a stable junior baseline across snapshots.

Bottom line: any junior claim must carry `seniority_final` and the YOE proxy together. If they disagree, that disagreement is itself a finding.

## Known Confounders

| Confounder | Severity | Why it matters |
|---|---|---|
| Description length growth | High | It inflates raw text metrics and changes density denominators |
| Template drift / sectioning change | High | `T13` shows the posting surface itself changed materially |
| Asaniczka label gap | High | Native `entry` is unavailable, so pooled seniority comparisons can mislead |
| Company composition shift | High | `T06` and `T16` show company mix explains part of the change, especially for junior YOE |
| Aggregator contamination | Moderate | It changes magnitudes, especially for AI and template metrics, but usually not the sign |
| Field-wide vs SWE-specific trend | High | `T18` shows AI-tool language is SWE-specific, but length/scope also rise in controls |
| Benchmark mismatch | High | `T23` shows the divergence claim depends on what worker benchmark you pick |
| Broad management noise | High | The broad pattern is too generic; keep strict management only |

## Discovery Findings

### Confirmed Hypotheses

- SWE postings became more AI/LLM-centered and more multi-constraint. `T11`, `T14`, `T16`, and `T18` all point the same way.
- The market’s latent structure is domain-first, not seniority-first. `T09` is the clearest structural result.
- Posting form changed materially. `T13` is a first-order validity result, not a side note.
- Company heterogeneity matters. Returning firms do not move together; within-company change is real and large. `T16` verified the overlap counts and decomposition.
- AI/LLM has become a coherent ecosystem in the posting language, not just a larger keyword bucket.

### Contradicted Hypotheses

- Junior collapse as the lead story. The junior signal is instrument-dependent, and the explicit label is conservative.
- Management decline as the senior-role story. Strict management is stable or slightly up; orchestration is the growth signal.
- Clean employer-worker divergence. `T23` is benchmark-sensitive, so the claim cannot be stated unqualified.
- A sharp convergence of junior and senior language. `T15` rejects that story.

### New Discoveries

- **Differential densification under template drift.** SWE densifies around AI, scope, and requirement stacking, but the document surface also becomes more structured.
- **AI/tool orchestration is the senior growth type.** Senior roles split toward orchestration and strategy rather than away from management entirely.
- **Template saturation and aspiration are real.** `T22` shows that some of the AI signal is more hedged, repeated, and template-like than a pure screening-bar story would imply.
- **The geography is broad, not hub-only.** `T17` shows AI/scope growth across many metros, while remote is unusable here.
- **The company strategy space is heterogeneous.** `T16` suggests at least four provisional trajectories, though the exact typology still needs spec lock.

### Unresolved Tensions

- How much of the measured text change is actual requirement change versus document-form drift?
- How much of the AI language is screening versus aspirational signaling?
- Why does the junior story flip depending on explicit label versus YOE proxy?
- Is the company typology stable enough to use as a formal result, or only as an exploratory lens?
- How much of the benchmark divergence is measurement mismatch rather than a genuine labor-market gap?

## Posting Archetype Summary

`T09` says the natural structure of the market is domain-first. NMF `k=15` is the downstream archetype map; BERTopic is too coarse to carry the paper. The clearest growth bundle is `AI / LLM workflows`, and seniority explains very little of the latent structure. That should become the domain stratifier in analysis, but only after the feature set is frozen.

## Technology Evolution Summary

`T14` is the cleanest technology-side result. The base stack stays recognizable: cloud/devops, frontend/web, testing, and backend families remain central. What changes is the density of the AI/LLM layer, which grows from a small historical signal into a coherent community rather than scattered keywords. The result survives aggregator exclusion and company capping. It is posting language, not actual use or AI-authorship, but it is still one of the strongest signals in the project.

## Geographic Heterogeneity Summary

`T17` shows geography matters as a modifier, not a lead story. AI/LLM and scope language rise broadly across metros, and the explicit-entry story depends on whether you use `seniority_final` or YOE. Remote is unusable in the selected metro frame. The geography evidence is useful for robustness and for telling the story that the change is broad, but it does not support a geography-first paper.

## Senior Archetype Characterization

`T21` revises the senior story. Strict management does not fall; strict orchestration rises sharply; strategic language rises modestly. The senior space is better described as a stable people-manager base plus a growing AI/tool-orchestrator cluster and a smaller AI-domain strategist cluster, not a management-to-orchestration substitution. Broad management is noisy and should stay out of the lead.

## Ghost / Aspirational Prevalence

`T22` says the signal is not literal fake jobs. It is overloaded postings, template repetition, and aspirational AI language. AI-posting language is more hedge-heavy than non-AI language, and direct employers are not immune. That makes ghost/aspiration a valid mechanism question, but not a literal-fake-job headline.

## Ranked New Hypotheses

1. **Differential densification under template drift.** Highest priority. This is the cleanest synthesis of T11, T13, T14, T16, and T18.
2. **Template drift mediates measured requirement growth.** High priority. It is the main validity issue for any language claim.
3. **AI/tool orchestration grows around stable management, rather than replacing it.** High priority. This is the best seniority-side discovery.
4. **Domain-first recomposition dominates seniority-first structure.** High priority. It is the structural claim that can anchor the paper.
5. **Explicit seniority is a lower bound; YOE is the broader junior pool.** Medium-high priority. This is essential for every junior claim.
6. **AI/LLM has become a coherent ecosystem in SWE postings.** Medium-high priority. Important for the technology section and the paper’s headline.
7. **AI language is aspirational and template-sensitive.** Medium-high priority. Important for interpretation and qualitative interviews.
8. **SWE is densifying more than adjacent/control roles on AI-tool language, but length/scope drift is field-wide.** Medium priority. Good for the cross-occupation section.
9. **Benchmark-sensitive divergence is a calibration problem, not a single estimate.** Medium priority. Important, but only as a validity section.

## Method Recommendations

- Use company-capped panels by default for any corpus-level text metric.
- Keep `seniority_final` and `yoe_extracted <= 2` together in junior tables.
- Treat `seniority_native` as arshkon-only diagnostic evidence.
- Use `description_core_llm` for the primary text surface; raw `description` only as a sensitivity or recall check.
- Freeze the archetype specification before turning company clusters into a claim.
- Use strict management only; broad management is too noisy.
- Report company-weighted and within-company versions whenever aggregate text change is discussed.
- For divergence claims, present a benchmark band, not a binary yes/no.

## Sensitivity Requirements

The following findings need explicit robustness checks in the analysis phase:

- T16 cluster labels and cluster sizes, because V2 did not reproduce the exact four-way split.
- T18 cross-occupation DiD and boundary similarity, because V2 did not independently finish those checks.
- T21 strict management / orchestration trends, because V2 only confirmed the broad-management demotion.
- T22 hedge/firm ratios and template saturation, because V2 did not rederive the headline AI aspiration ratios.
- T23 benchmark divergence, because the claim is benchmark-sensitive by design.
- All entry-level comparisons, because explicit labels and YOE proxy disagree materially.
- All text claims that mix cleaned and raw surfaces, because `T13` shows the document form changed.

## Interview Priorities

The qualitative work should directly target the contradictions surfaced by the quantitative work:

1. **AI requirement vs aspiration.** When is AI wording a hard bar, and when is it signaling or templating?
2. **Template drift vs real work.** Who edits the JDs, how often are they copied, and what part of the text actually changes the screening bar?
3. **Explicit seniority vs YOE.** Why do titles and years-of-experience tell different stories about juniors?
4. **Senior orchestration without management decline.** What changed in senior work if people-management did not disappear?
5. **Benchmark sensitivity.** What does it mean for postings to “outpace” worker usage if the benchmark itself is unstable?

## Bottom Line

The paper should now be framed as an empirical restructuring study with a methods-validity backbone. The best current claim is not “AI killed junior jobs” and not “employers outrun workers.” It is that AI-era SWE postings became more AI/LLM-centered, more multi-constraint, and more standardized/template-shaped, with heterogeneous company strategies and aspirational signaling mediating how those changes should be read.
