# V1 Gate 2 Verification

## Findings First

| Headline | V1 status | V1 result | Action before Wave 3 |
|---|---|---:|---|
| T11 all-SWE requirement breadth | **Verified with minor construct drift** | V1: 7.46 -> 9.67 (+2.20). Reported: 7.23 -> 9.39 (+2.16). Levels are within 3.2%; direction/effect verified. | Use as a lead finding, but call it regex-derived requirement breadth in the LLM-cleaned subset. |
| T11 J1-J4 junior breadth | **Verified direction; J1/J2 revised slightly upward in V1** | V1: J1 +1.83, J2 +1.94, J3 +1.82, J4 +2.11. Reported: +1.74, +1.84, +1.75, +2.08. | Keep unanimous direction. Treat exact J1/J2 magnitudes as definition-sensitive because independent regexes are slightly broader. |
| T08/T14 broad AI prevalence | **Verified** | 3.751% -> 23.696% (+19.94 pp). | Split broad AI into AI-tool, AI-domain, and ambiguous terms in Wave 3. |
| T14 technology movers | **Verified** | Python 32.34% -> 49.37%; CI/CD 15.40% -> 33.59%; API design 13.04% -> 27.40%; observability 1.87% -> 13.88%; LLM 0.99% -> 12.96%. | Safe as binary mention screens; do not imply density growth. |
| T09 AI LLM Platforms share | **Verified on labeled subset** | 3.11% -> 17.45% of archetype-labeled rows. | Cite only as LLM-labeled subset evidence, not full scraped corpus share. |
| T09/T15 domain over seniority structure | **Verified** | V1 NMI proxy: tech-family 0.152 vs seniority 0.0063. T15 variance: archetype eta2 0.111/0.131 vs seniority 0.007/0.0068. | Keep as organizing-frame evidence. |
| T13 cleaned length and core share | **Partly verified; classifier needs follow-up** | Length: 945.99 -> 1,094.22 chars. Core req/resp/preferred mean row share: 42.02% -> 41.46%. | Length verified. Core share is table-arithmetic verified from T13 artifact; section parser needs manual validation. |
| T15 generic junior-senior convergence rejection | **Verified** | Embedding shift +0.00496, below within-2024 +0.02151. V1 TF-IDF/SVD rerun is negative (-0.0476). | Do not dispatch Wave 3 around generic semantic convergence. Focus on label-vs-YOE and targeted requirement migration. |

Primary V1 tables:

- `exploration/tables/V1/headline_T11_requirement_breadth.csv`
- `exploration/tables/V1/headline_T08_T14_broad_ai_tech.csv`
- `exploration/tables/V1/headline_T14_top_movers.csv`
- `exploration/tables/V1/headline_T09_ai_llm_share.csv`
- `exploration/tables/V1/headline_T09_domain_vs_seniority_nmi.csv`
- `exploration/tables/V1/headline_T13_cleaned_length_core_share.csv`
- `exploration/tables/V1/headline_T15_centroid_convergence.csv`
- `exploration/tables/V1/semantic_precision_summary.csv`
- `exploration/tables/V1/citation_audit.csv`

Script: `exploration/scripts/V1_verify_wave2.py`.

## Method

V1 used `./.venv/bin/python`, DuckDB with `PRAGMA memory_limit='4GB'; PRAGMA threads=1;`, and shared artifacts rather than rerunning preprocessing, embeddings, the tech matrix, or any LLM stages.

The verification script does not import prior task scripts. It reads prior reports/tables only to identify target numbers and, for T15/T13, to run explicit table-consistency audits where independent recomputation is not fully meaningful without reusing a prior artifact boundary.

## Headline Details

### Requirement Breadth

V1 reimplemented requirement breadth from shared cleaned text and tech matrix:

`non_ai_tech_count + ai_requirement_count + soft_skill_count + org_scope_count + management_strong_count + education_any + yoe_extracted nonnull`.

The all-SWE result is close to T11: V1 gets 7.46 -> 9.67, while T11 reported 7.23 -> 9.39. The difference is traceable to regex-family implementation. V1's independent soft-skill and org-scope patterns are not byte-identical to T11 and remain slightly broader. The more important effect size, +2.20 signals, matches T11's +2.16 closely.

Junior definitions all move up. J1/J2 are just outside the 5% threshold relative to reported changes (+5.2% and +5.5%) because V1's independent detector assigns slightly more 2026 entry/associate rows to org-scope/soft-skill categories. This is not a directional failure, but exact junior magnitudes should be treated as feature-specification dependent.

### AI And Technology Expansion

The broad AI prevalence claim matches exactly from the shared tech matrix: 3.751% in pooled 2024 and 23.696% in scraped 2026. The same audit verifies the cited top movers and denominators.

Citation caveat: T08 broad AI includes AI-domain terms and ambiguous `mcp`; T11's AI requirement count excludes `mcp`, `numpy`, `pandas`, and `scipy`. These should not be mixed in one prevalence/SNR cell.

### Archetypes And Structure

The AI LLM Platforms archetype share is verified from `swe_archetype_labels.parquet`: 664 / 21,361 labeled 2024 rows and 2,160 / 12,380 labeled 2026 rows.

V1 also ran an independent NMI proxy using dominant tech-family labels from the shared tech taxonomy. The result is directionally consistent with T09: technology/domain structure is much stronger than seniority structure. It is not numerically identical to T09's `tech_domain` NMI because T09 used its own 8,000-row modeling-sample factor.

T15 independently reinforces the same conclusion: archetype explains about 16-19x as much representation variance as seniority in embeddings and TF-IDF/SVD.

### Cleaned Text And Sections

Cleaned length is directly verified from `swe_cleaned_text.parquet`, filtered to `text_source='llm'` and nonempty text: 945.99 -> 1,094.22 chars.

The core-section share is verified as table arithmetic from T13's `section_text_by_uid.parquet`: responsibilities + requirements + preferred mean row share is 42.02% -> 41.46%. This is not the char-weighted aggregate share. Paper text should say "mean row share" if it cites 42.0% -> 41.5%.

V1 did not rebuild the full T13 section parser. Before this becomes paper-grade "not boilerplate" evidence, Wave 3 should manually validate the section classifier on stratified Kaggle/scraped examples.

### Semantic Convergence

Using T15's bounded sample index and the shared embeddings, V1 recomputed trimmed junior/senior centroids. The embedding values exactly reproduce T15: arshkon 0.95433, asaniczka 0.97584, scraped 0.95929; the favorable arshkon-to-scraped shift is only +0.00496 and is smaller than the within-2024 gap of +0.02151.

V1 also reran a separate TF-IDF/SVD representation on the same sample. It gives arshkon 0.89793 and scraped 0.85036, a negative shift of -0.0476. This agrees with T15's rejection even though the exact TF-IDF settings are independent.

## Semantic Precision

V1 sampled 50 contexts per family, stratified 25/25 by period, and reviewed surrounding text. Final precision table:

| Family | Valid / n | Precision | Verdict |
|---|---:|---:|---|
| AI/tool/LLM | 50 / 50 | 100% | Pass |
| Workflow/pipeline/platform | 50 / 50 | 100% | Pass |
| Org-scope/ownership | 50 / 50 | 100% | Pass |
| Management/mentorship | 50 / 50 | 100% | Pass |

Important adjustment: during review, V1 found `environment(s)` too ambiguous for the workflow/pipeline/platform family because it often means a general work setting. It is excluded from the final validation family. If Wave 3 or the paper cites `environment(s)` as a requirement-surface metric, validate it separately.

No final family falls below 80% precision.

## Citation Audit

The main transparency risks are:

1. **Broad vs narrow AI.** T08 broad AI, T14 AI-tool category, T11 AI requirement count, and T12 open AI terms are different constructs.
2. **Subset leakage.** T09 archetype shares and T15 semantic results are LLM-cleaned/labeled subset evidence; scraped coverage is limited.
3. **Core-section denominator.** T13's 42.0% -> 41.5% is average row share, not a char-weighted corpus share.
4. **Density vs prevalence.** T14 binary tech prevalence rises strongly, but tech density does not generally rise after length normalization.
5. **Ambiguous tokens.** `mcp` and `environment(s)` require separate handling before citation as substantive AI/platform requirements.

Full audit: `exploration/tables/V1/citation_audit.csv`.

## Composite Matching

V1 found no Wave 2 headline discussed here that depends on a matched-delta or composite-score control. The search found ordinary deltas and token text, not composite matching designs. Therefore the per-component outcome-correlation check is not applicable for Wave 2.

Table: `exploration/tables/V1/composite_matching_audit.csv`.

## Alternative Explanations

Requirement breadth:

- Source/instrument composition: Kaggle vs scraped formatting and platform behavior can change observed requirement breadth.
- LLM-cleaned subset coverage: scraped LLM text covers only about one-third of scraped LinkedIn SWE rows.
- Longer descriptions: some breadth growth could be because longer descriptions list more categories.
- Company concentration: prolific employers and entry-specialist employers can move YOE and junior metrics.

AI/technology expansion:

- Real AI-domain hiring vs generic AI signaling are mixed in broad prevalence.
- Recruiter/LLM-authored text may add contemporary AI/tooling language without changing screening.
- Tech matrix is regex-based; it measures mentions, not required proficiency or screening intensity.
- External labor-market conditions in 2026 may favor AI/platform employers independently of coding agents.

Archetype shift:

- NMF labels are descriptive, not ground-truth occupational categories.
- Scraped archetype coverage is LLM-labeled subset evidence.
- Company/source composition contributes to archetype structure; V1 top-company NMI is nontrivial.

Cleaned length / non-boilerplate:

- Section classifier is heuristic and not yet independently hand-validated.
- LLM cleaning may remove boilerplate unevenly across Kaggle and scraped formats.
- Raw descriptions show substantial benefits/about/legal material, so text-source discipline remains essential.

Convergence rejection:

- Bounded sample and LLM coverage limit full-corpus generalization.
- S2/S3 cells remain thin; do not overread title-keyword senior results.
- The null corpus-level result could hide targeted within-domain requirement migration.

## Specification Dependencies

Seniority-stratified results must keep the T30 panel. Requirement breadth rises under J1-J4 and S1/S4 in V1, but exact J1/J2 magnitudes are feature-specification dependent. For any Wave 3 seniority claim, report all applicable T30 variants and state whether the result is about employer labels or low-YOE rows.

Other dependencies to flag in downstream tasks:

- text source: `description_core_llm` / `text_source='llm'` vs raw fallback
- source restriction: arshkon-only vs pooled 2024
- company cap and aggregator exclusion
- SWE-tier exclusion
- LLM coverage for text/archetype/semantic claims
- broad vs narrow AI pattern definitions

## Wave 3 Recommendation

Modify Wave 3 before dispatch in these ways:

1. T16/T28 should decompose requirement breadth, broad AI, AI-tool-only, and tech breadth by company and archetype, not only junior share.
2. T18 should be a gatekeeper for SWE specificity. If controls show the same AI/platform/workflow expansion, the narrative must become broader than SWE restructuring.
3. T20 should focus on the label-vs-YOE split, not generic junior-senior semantic convergence.
4. T21 should separate mentorship/coordination from direct-report management.
5. T22 should test whether AI/platform/workflow breadth is screened, preferred, aspirational, or template/ghost language.
6. Validate T13's section classifier and any use of `environment(s)` before treating "not boilerplate" and workflow/platform prevalence as paper-grade.

Bottom line: Wave 2's strongest reframed story survives V1: SWE postings show broader AI/platform/workflow and technology surfaces across seniority definitions. The paper should avoid overclaiming junior elimination, full-corpus archetype shares, generic semantic convergence, or unvalidated boilerplate mechanisms.
