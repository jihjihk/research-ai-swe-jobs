# Gate 2 Research Memo

## What We Learned

1. **The strongest Wave 2 finding is skill-surface expansion, not junior elimination.** Requirement breadth, credential stack depth, technology breadth, AI/tool mentions, platform/workflow language, and title-level AI/data/agent hybridization all rise from 2024 to 2026. The best-supported version is not "postings got longer," but "postings ask for a broader AI/platform/workflow stack."

2. **The market's natural structure is domain/technology, not seniority.** T09 and T15 converge on this. T09's NMF archetypes align far more with tech/domain than seniority (`NMI = 0.205` vs `0.0069`). T15 finds archetype explains much more representation variance than seniority in both embeddings and TF-IDF/SVD.

3. **Generic junior-senior convergence is rejected.** T15 finds junior-senior centroid similarity does not survive within-2024 calibration. Label-based J1 entry rows remain junior-neighbor enriched, while low-YOE J3/J4 rows are more senior-neighbor-heavy. This sharpens the Gate 1 puzzle: "entry" and "low-YOE" are different labor-market signals.

4. **Scope growth is not mostly boilerplate.** T13 shows LLM-cleaned descriptions are longer, but benefits/about/legal contribute almost none of the cleaned-text growth. T12's workflow/platform/AI term signal survives section filtering, aggregator inclusion/exclusion, company capping, raw-description sensitivity, source restriction, and SWE-tier sensitivity.

5. **Senior role evolution is not a simple move away from management.** T11 finds senior complexity rises under S1-S4, and strong management/mentorship indicators rise rather than fall. Direct-report management remains rare. The senior story is closer to "more complex senior roles with mentorship/coordination and technical breadth" than "management language disappeared."

## What Surprised Us

- **AI/LLM growth is not mechanically anti-junior.** T09's AI LLM Platforms archetype rises sharply but is not especially entry-poor in the LLM-labeled subset. Cloud DevOps is less junior-heavy than AI/LLM.
- **Low-YOE rows are mostly not entry-labeled.** T08 shows that in scraped 2026, 72.4% of `yoe_extracted <= 2` rows are `unknown`, 21.5% are `mid-senior`, and only 5.5% are J1 entry. T15 adds that these low-YOE rows are more senior-neighbor-heavy than J1 entries.
- **The tech expansion is broad and networked.** T14 finds 29 calibrated tech risers and a new AI/Python/platform cluster where Python reclusters with LLM, generative AI, RAG, PyTorch, TensorFlow, Claude, and MLOps.
- **Requirement breadth rises almost everywhere.** T11 finds breadth growth across all junior definitions, all senior definitions, and 7 of 8 T09 archetypes. Domain recomposition alone is unlikely to explain the whole change.
- **Testing does not rise with the rest of the stack.** T14 finds testing-category prevalence roughly flat while AI, cloud/devops, API/platform, and operational practice language rise.

## Evidence Assessment

| Finding | Strength | Sample / n | Calibration / sensitivity | Main confounds | Interpretation |
|---|---|---:|---|---|---|
| Requirement breadth rose | Strong | 34,258 LLM-cleaned SWE rows; scraped LLM n = 12,534 | Survives within-2024 calibration, aggregator exclusion, company cap, entry-specialist exclusion, raw fallback sensitivity | LLM-cleaned subset coverage; regex feature limits | Lead candidate finding |
| Junior requirement breadth rose | Strong | J1/J2 thin but J3/J4 powered; all four rise | J1 +1.74, J2 +1.84, J3 +1.75, J4 +2.08 | J1 label vs J3/J4 construct mismatch | Strong scope-expansion finding |
| Tech breadth rose | Strong | 63,701 shared tech-matrix rows | J1-J4 and S1-S4 all rise; major movers survive aggregator/company/SWE-tier checks | Binary regex mentions; density falls | Strong, but phrase as breadth in longer postings |
| AI/platform/workflow text shift | Strong-moderate | T12 primary n = 4,221 arshkon vs 9,531 scraped | Sensitivity overlap 0.82-0.98; within-2024 overlap 0.12 | Open-token discovery | Strong discovery signal; needs V1 semantic validation |
| Domain/technology dominates seniority | Strong | T09 8,000 sample; T15 9,054 sample | T09 and T15 agree | LLM-cleaned subset coverage | Reframes paper around domain/technology expansion |
| AI/LLM archetype grew | Moderate-strong | T09 labels cover 33,741 LLM-cleaned rows; scraped coverage ~30% | Survives aggregator and SWE-tier sensitivity | LLM-labeled subset, NMF labels | Strong within labeled text; not full scraped corpus |
| Generic junior-senior convergence | Rejected | T15 bounded sample n = 9,054 | Embedding shift +0.005 < within-2024 +0.022; TF-IDF negative | LLM subset; tiny S2/S3 cells | Do not use as lead claim |
| Senior management declined | Contradicted/needs reframing | T11 S1/S4 powered; S2 sparse | Strong-management rises under S1-S4; direct-report terms rare | Mentorship vs management semantics | Reframe as mentorship/coordination/complexity |
| Scope growth not boilerplate | Moderate-strong | T13 LLM text n = 34,101 nonempty rows | Cleaned text/section-filtered signal persists; raw shows boilerplate risk | Heuristic section classifier | Good validity support; section classifier should be verified |
| Hybrid AI/data titles rose | Moderate-strong | 63,701 SWE title rows | Aggregator exclusion stable; source-sensitive title lists handled | Titles are signals, not occupations | Useful supporting finding |

## Seniority Panel

### Junior Requirement Breadth

This is the strongest junior-stratified Wave 2 finding. It can be cited as a lead claim because all four definitions agree directionally.

| Variant | Definition | Effect | Direction | n basis | Agreement verdict |
|---|---|---:|---|---:|---|
| J1 | `seniority_final = 'entry'` | +1.74 breadth signals | Up | label-based, thin | Unanimous |
| J2 | entry/associate | +1.84 | Up | label-based, thin/moderate | Unanimous |
| J3 | `yoe_extracted <= 2` | +1.75 | Up | powered low-YOE | Unanimous |
| J4 | `yoe_extracted <= 3` | +2.08 | Up | powered broad low-YOE | Unanimous |

Interpretation: junior/low-YOE postings ask for broader requirement bundles in 2026. This supports scope expansion, not junior elimination.

### Junior Scope Density

Scope-density growth is less uniform.

| Variant | Definition | Effect | Direction | n basis | Agreement verdict |
|---|---|---:|---|---:|---|
| J1 | entry label | positive but weak vs calibration | Up/weak | thin | Split in strength |
| J2 | entry/associate | positive but weak vs calibration | Up/weak | thin/moderate | Split in strength |
| J3 | YOE <= 2 | +0.17 per 1K chars | Up | powered | Stronger for YOE |
| J4 | YOE <= 3 | +0.19 per 1K chars | Up | powered | Stronger for YOE |

Interpretation: low-YOE roles show clearer scope-density growth than explicitly entry-labeled roles. This reinforces the label-vs-YOE distinction.

### Junior Share Direction

Wave 2 confirms Gate 1's warning.

| Variant | Definition | Effect | Direction | n basis | Agreement verdict |
|---|---|---:|---|---:|---|
| J1 | entry label | pooled up, arshkon-only slightly down | Mixed | 379/1,275 pooled | Source-sensitive |
| J2 | entry/associate | pooled up, arshkon-only slightly down | Mixed | 418/1,327 pooled | Source-sensitive |
| J3 | YOE <= 2 | up under pooled and arshkon-only | Up | 1,766/4,742 pooled | Split vs J1/J2 source-restricted |
| J4 | YOE <= 3 | up under pooled and arshkon-only | Up | 4,035/9,307 pooled | Split vs J1/J2 source-restricted |

Interpretation: do not lead with junior share. Use it as a mechanism puzzle: explicit entry labels and low-YOE postings diverge.

### Senior Complexity

Senior requirement breadth rises across S1-S4.

| Variant | Definition | Effect | Direction | n basis | Agreement verdict |
|---|---|---:|---|---:|---|
| S1 | mid-senior/director | +2.40 breadth signals | Up | powered | Unanimous |
| S2 | director | +6.16 | Up | sparse/descriptive | Unanimous direction, weak power |
| S3 | raw senior title regex | +2.32 | Up | diagnostic | Unanimous |
| S4 | YOE >= 5 | +2.61 | Up | powered | Unanimous |

Interpretation: senior roles are also expanding in breadth. This weakens a "requirements migrated downward only" framing and favors field-wide or domain-wide skill-surface expansion.

### Semantic Convergence

T15 rejects generic junior-senior convergence under the panel. The only apparent embedding passes involve S3, which is artifact-prone and thin.

| Variant Pair | Result | Direction | n basis | Agreement verdict |
|---|---|---|---:|---|
| J1/S1, J2/S1 | Embedding tiny positive, TF-IDF negative | Reject | bounded LLM sample | Contradictory methods |
| J1/S4, J2/S4 | Flat/negative | Reject | bounded LLM sample | Reject |
| J3/J4 vs S1/S4 | Small embedding positives below calibration, TF-IDF flat/negative | Reject | bounded LLM sample | Reject |
| J3/J4 vs S3 | Some embedding positives | Descriptive only | S3 thin/artifact-prone | Not admissible lead |

Interpretation: do not claim junior postings became semantically senior-like at the corpus level. Use targeted requirement migration instead.

## Narrative Evaluation

- **RQ1: Employer-side restructuring. Needs major reframing.** Junior share decline is weakened/contradicted. Junior scope expansion is now supported. Senior role redefinition is supported only if reframed away from "management decline" toward broader senior complexity, mentorship/coordination, and top-ladder/directed responsibility. RQ1 should not lead with junior volume; it should lead with changing requirement breadth and label/YOE boundary ambiguity.

- **RQ2: Task and requirement migration. Partially supported, but not as simple downward migration.** AI, platform, workflow, observability, API, CI/CD, and exposure-style requirement language rise. But because breadth rises for senior roles too and generic junior-senior convergence is rejected, the better framing is **skill-surface expansion across roles**, not one-way migration from senior to junior.

- **RQ3: Employer-requirement / worker-usage divergence. Not tested yet.** Wave 2 gives strong employer-side AI requirement/title/tech evidence. T23 must compare AI-tool-specific, AI-domain, and generic AI mentions against external worker-usage benchmarks.

- **RQ4: Mechanisms. Strongly motivated.** Interviews should focus on whether broadened requirement stacks and AI/workflow/platform language are screened, aspirational, recruiter-template-driven, or anticipatory.

Original narrative status:

- Junior share decline: **weakened/contradicted**.
- Junior scope inflation: **supported**, especially requirement breadth; scope density is stronger for low-YOE than entry-labeled roles.
- Senior archetype shift toward AI orchestration: **needs reframing**. Evidence supports senior complexity/coordination/mentorship; management decline is not supported.
- Employer-usage divergence: **pending**.

## Emerging Narrative

The data is pointing toward **AI-era skill-surface expansion and domain restructuring**, not a clean collapse of junior hiring. SWE postings in 2026 ask for broader stacks: AI/LLM tools, Python/platform ecosystems, CI/CD, observability, API design, ownership, workflows, and exposure-style credentials. This expansion appears across junior and senior definitions and across most domains.

The most interesting seniority result is boundary ambiguity. Explicit entry labels, low-YOE postings, and semantic junior-neighbor structure do not identify the same rows. Low-YOE roles look more senior-like than entry-labeled roles, suggesting either relabeling, ambiguous ladders, or postings that ask for senior-style scope while preserving low YOE.

Domain/technology is the primary organizing axis. This means the paper should not frame the market primarily as "junior vs senior changed." It should frame seniority as one boundary affected by a broader technology/domain shift.

## Research Question Evolution

Recommended current RQs:

- **RQ1 revised:** How did SWE postings' requirement breadth and technology/workflow surface change from 2024 to 2026, and how much of that change survives source, company, aggregator, and text-quality controls?
- **RQ2 revised:** How did seniority boundaries change across employer labels, YOE floors, and semantic content, especially for low-YOE postings that are not labeled entry?
- **RQ3 revised:** Are AI/tool/platform requirements concentrated in specific domains and companies, or do they represent broad within-domain expansion?
- **RQ4 retained:** Do employer AI requirements outpace worker usage benchmarks, especially for specific AI coding tools and agentic workflows?
- **RQ5 retained:** How do hiring-side actors explain broadened requirements: real screening, aspirational/ghost requirements, template inflation, or anticipatory restructuring?

This set follows the evidence better than the original junior-decline-centered RQs.

## Gaps And Weaknesses

- The strongest text, archetype, and semantic findings describe the LLM-cleaned subset. Scraped coverage is about 31%, so analysis-phase claims need either coverage expansion or careful labeled-subset language.
- Open-token findings need verification. T12 is strong discovery evidence, but V1 must semantically validate top keyword patterns and prevent prevalence/pattern mismatches.
- Requirement breadth is regex-derived. It is validated enough for exploration, but ghost/aspirational status remains unknown until T22 and interviews.
- Company and domain decomposition is not finished. T06/T09/T11 suggest within-domain breadth growth, but T16/T28 must separate within-company from market-entry/domain recomposition.
- Cross-occupation specificity is unknown. If control occupations also show broad requirement inflation and AI/workflow vocabulary, the SWE-specific restructuring story weakens.
- AI requirement/usage divergence is untested. Wave 2 only establishes the posting-side numerator.

## Direction For Next Wave

Wave 3 should be modified to pursue the strongest Wave 2 story:

- **T16:** Decompose requirement breadth, AI-tool prevalence, tech breadth, and low-YOE/entry divergence within common companies. Do not focus only on entry share.
- **T17:** Geographic analysis is useful but secondary. Prioritize whether AI/platform expansion is geographically concentrated and whether metro patterns are domain-composition effects.
- **T18:** Make cross-occupation specificity a gatekeeper. If AI/platform/workflow expansion appears similarly in controls, the paper becomes a broader posting-language/technology-change study, not SWE-specific restructuring.
- **T19:** Temporal stability should characterize scraped-window consistency for AI/tool/requirement breadth, not just entry share.
- **T20:** Pivot from generic convergence to label-vs-YOE boundary structure. Compare J1 entry vs J3/J4 low-YOE unknown/mid-senior rows explicitly.
- **T21:** Test senior complexity/mentorship/coordination, not "management decline." Separate direct reports/hiring/headcount from mentoring, technical coordination, architecture/review, and strategic scope.
- **T22:** Highest priority validity task. Determine whether AI/platform/workflow breadth is ghost/aspirational or likely screened. Compare firm requirements vs preferred/nice-to-have language.
- **T23:** Split divergence by AI-tool, AI-domain, and generic AI. Specific tool/agent mentions are more publishable than broad AI prevalence.
- **Add/keep T28:** Domain-stratified scope changes are now important. T28 should be run after J-M or alongside if memory permits, with T09/T11/T14 as core inputs. T29 is lower priority but useful if description-authorship artifacts remain a concern.

Memory rule remains: run Wave 3 staged, not all heavy agents simultaneously.

## Current Paper Positioning

If stopped here, the best paper is now an **empirical posting-content restructuring paper with a strong measurement contribution**. The provisional one-sentence abstract would be:

> Between 2024 and 2026, SWE postings did not simply eliminate junior roles; they broadened the required skill surface around AI, platform, workflow, and systems responsibilities across seniority levels, while employer labels, YOE floors, and semantic role content diverged.

Most promising lead findings:

1. Requirement breadth rose strongly across all junior and senior definitions, surviving major sensitivities.
2. AI/platform/workflow technology expansion is broad, calibrated, and networked, not just generic AI keyword inflation.
3. Domain/technology structure dominates seniority structure in unsupervised and semantic analyses.
4. Generic junior-senior convergence is rejected; the real seniority puzzle is low-YOE unlabeled/mid-senior roles.
5. Cleaned-text and section-filtered analyses argue that the signal is not mostly boilerplate.

The next wave must decide whether these findings are SWE-specific, within-company, and real hiring-bar changes rather than ghost/template requirements.

