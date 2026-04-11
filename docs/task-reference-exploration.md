# Exploration Task Reference

Date: 2026-04-05
Input: `data/unified.parquet`
Schema reference: `docs/preprocessing-schema.md`

---

## 1. Shared preambles

There are two preamble sections: a **core preamble** (for all waves) and an **analytical preamble** (for Wave 2+ only). Prepend the core preamble to every sub-agent prompt. Prepend both the core and analytical preambles to Wave 2+ agent prompts.

### 1a. Core preamble (all waves)

```
## Exploration task context

You are a sub-agent executing exploration tasks for a SWE labor market research project studying how AI coding agents are restructuring software engineering roles.

**Your orientation is DISCOVERY, not confirmation.** The project has initial research questions (RQ1-RQ4 in docs/1-research-design.md) about junior scope inflation, senior archetype shifts, and employer-requirement/worker-usage divergence. These are hypotheses, not conclusions. Your job is to report WHAT THE DATA SHOWS. Findings that surprise, contradict, or go beyond existing RQs are especially valuable. Every task report should include a "Surprises & unexpected patterns" section.

**Input data:** `data/unified.parquet`
Read `docs/preprocessing-schema.md` for column definitions and recommended usage.

**Critical facts:**
- Three sources: kaggle_arshkon (April 2024), kaggle_asaniczka (Jan 2024), scraped (March 2026). The scraper runs daily, so row counts grow. Query the data to get current counts — do not rely on documented numbers.
- Asaniczka has ZERO native entry-level seniority labels. `seniority_native` cannot detect entry-level postings in asaniczka by construction — use arshkon-only for any sanity check that depends on `seniority_native`.
- Remote work flags are 0% in 2024 sources (data artifact). Do not interpret as a real change.
- LLM classification columns: `swe_classification_llm`, `ghost_assessment_llm`, `yoe_min_years_llm`. Check `llm_classification_coverage` for coverage. Use `ghost_assessment_llm` as the primary ghost indicator with `ghost_job_risk` as fallback. (LLM seniority writes back to `seniority_final`; there is no separate `seniority_llm` column.)
- **Seniority — use `seniority_final`.** It is the combined column. Stage 5 fills it from high-confidence title keywords (`title_keyword`, `title_manager`); Stage 10 overwrites it with the LLM result for rows the router sent to the LLM. `seniority_final_source` records which path produced each value (`title_keyword`, `title_manager`, `llm`, or `unknown`). `seniority_final` is `'unknown'` only when no signal fired. See `docs/preprocessing-schema.md` Section 4 for the full schema.
- `selected_for_llm_frame` marks the sticky balanced core only. `selection_target` is the minimum core size, not the full usable LLM set.
- Stage 9 and Stage 10 use separate caches, so row coverage can differ between them. A row may have usable Stage 9 text without Stage 10 classification, or vice versa.
- `llm_extraction_sample_tier` and `llm_classification_sample_tier` take `core`, `supplemental_cache`, or `none`. Balanced-sample claims apply only to `selected_for_llm_frame = true`.
- For raw Stage 9/Stage 10 LLM columns (text-based: `description_core_llm`, `swe_classification_llm`, `ghost_assessment_llm`, `yoe_min_years_llm`), filter to `llm_*_coverage == 'labeled'`. **Seniority is the exception:** `seniority_final` is the combined column and should be used directly without coverage filtering.
- 31GB RAM limit — use DuckDB or pyarrow for queries, never load full parquet into pandas.

**Seniority validation discipline (CRITICAL — the direction of any entry-level finding depends on this):**

`seniority_final` is the production seniority column. For any seniority-stratified finding — especially entry-level shares, counts, or trends — validate it against the following label-independent and arshkon-only checks:

1. **YOE-based proxy (label-independent, primary validator).** Compute the share of postings with `yoe_extracted <= 2` (and `<= 3`) by period. This proxy does not depend on any seniority classifier. If `seniority_final` and the YOE-based proxy disagree on direction or magnitude, do NOT pick a side without investigating WHY. Differential native-label quality across data snapshots is one possible explanation; real market change is another; shifts in employer labeling explicitness are a third. Report the disagreement honestly and investigate the mechanism.

2. **`seniority_native` arshkon-only baseline (diagnostic).** For any sanity check that uses `seniority_native`, restrict to arshkon. Asaniczka has zero native entry-level labels, so pooling it under `seniority_native` would dilute the 2024 entry rate to near-zero and can flip trend direction. `seniority_final` and the YOE-based proxy CAN include asaniczka because they do not depend on native labels (asaniczka entry signal in `seniority_final` comes via the Stage 10 LLM).

3. **Profile native-label quality before trusting it.** For `seniority_native = 'entry'` rows in each source, compute the YOE distribution. If entry-labeled rows have systematically different YOE across snapshots, the native classifier has differential accuracy across snapshots and any cross-period comparison using `seniority_native` inherits that bias. This diagnostic is most valuable for arshkon (the only 2024 source with native entry labels).

**Reporting:** When you report a seniority-stratified result, present `seniority_final` as the primary, then list the YOE-based proxy and (where applicable) the arshkon-only `seniority_native` baseline as label-independence checks. Agreement across the three strengthens the finding; disagreement is itself an important finding to investigate.

**Description text quality — critical for text-based analyses:**
`description_core_llm` (LLM-based boilerplate removal) is the **only** cleaned-text column. The former rule-based `description_core` was retired on 2026-04-10 because ~44% accuracy was misleading downstream analysis. Check `llm_extraction_coverage` to confirm coverage by source; filter to `labeled` whenever you use `description_core_llm`.

Text column rules:
- **Text-dependent analysis** (embeddings, topic models, requirement extraction, corpus comparison, density metrics): Use `description_core_llm` with `llm_extraction_coverage = 'labeled'`. Do not backfill missing rows with raw `description` for boilerplate-sensitive work — restrict the sample instead and report coverage.
- **Binary keyword presence** (does the posting mention X anywhere?): Raw `description` is acceptable for recall when the presence signal is insensitive to boilerplate phrasing. Density metrics (mentions per 1K chars) must still use `description_core_llm`.
- **Non-text analyses** (seniority counts, company analysis, geographic patterns): Use all rows regardless of text column.
- **Sensitivity check:** When a finding depends on text, the meaningful sensitivity is `description_core_llm` vs raw `description` (no rule-based alternative exists). If the direction flips under raw text, the finding is boilerplate-driven and must be flagged.
- For Stage 10 LLM-only diagnostics on `swe_classification_llm` or `ghost_assessment_llm` in isolation, filter to `labeled` and treat `rule_sufficient` separately (since for those columns the LLM was deliberately skipped). This is distinct from seniority, where `seniority_final` is the combined column and should be used directly.

**Default SQL filters (apply unless task says otherwise):**
```sql
WHERE source_platform = 'linkedin'
  AND is_english = true
  AND date_flag = 'ok'
```

**Available Python packages (confirmed installed):**
- **Topic modeling:** bertopic (0.17), sklearn (LDA, NMF)
- **Clustering:** hdbscan, sklearn (k-means, DBSCAN, spectral, agglomerative, GMM)
- **Dimensionality reduction:** umap-learn, sklearn (PCA, t-SNE, SVD)
- **Embeddings:** sentence_transformers (all-MiniLM-L6-v2), gensim (Doc2Vec, Word2Vec)
- **NLP:** spacy, nltk, textstat (readability metrics)
- **Core:** sklearn, scipy, numpy, pandas (small subsets only), matplotlib, seaborn
- **Graph:** networkx
- **Data:** duckdb, pyarrow

**Test-driven development for non-trivial logic.** Before relying on a regex, parser, or custom extractor, write a handful of `assert` statements covering edge cases (special characters, word boundaries, escape sequences, empty/short/long inputs). A few asserts inline at the top of your script is enough — do not build a test framework or create separate test files. Patterns with `\b` near non-word characters (`\bc\+\+\b`, `\b\.net\b`) and patterns over text containing markdown escapes (`C\+\+`, `C\#`) fail silently and are particularly worth testing. Validate your output against an independent reference (direct LIKE query, structured field, manual sample) when one exists.

**Output conventions:**
- Figures -> `exploration/figures/TASK_ID/` (PNG, 150dpi, max 4 per task)
- Tables -> `exploration/tables/TASK_ID/` (CSV)
- Scripts -> `exploration/scripts/TASK_ID_descriptive_name.py` (reusable analysis scripts)
- Report -> `exploration/reports/TASK_ID.md`. Use a structure appropriate to the task. Include at minimum: headline finding, methodology, surprises/unexpected patterns, data quality caveats, and action items for downstream tasks. Wave 2+ tasks should also include a sensitivity checks section.

Create exploration/ directories if they don't exist.
Use DuckDB CLI or pyarrow for all data queries. Do NOT load the full parquet into pandas.
Use the project virtualenv: `./.venv/bin/python` for running Python scripts.
```

### 1b. Analytical preamble (Wave 2+ only)

Prepend this AFTER the core preamble for Wave 2+ agents.

```
## Text analysis hygiene — apply to ALL text-based tasks

1. **Company-name stripping.** Before any corpus comparison or term-frequency analysis, build a stoplist from all `company_name_canonical` values, tokenize them into words, and strip them during tokenization. Company names dominate results otherwise.

2. **Length normalization.** Description length grew substantially from 2024 to 2026. For keyword analyses:
   - Primary: binary indicator (any mention per posting)
   - Secondary: rate per 1,000 characters
   - Always report both. Never report raw counts without normalization.

3. **Artifact filtering.** For term lists:
   - Require terms to appear in >=20 distinct companies
   - Exclude HTML concatenation artifacts (tokens >12 chars with no spaces that aren't real words)
   - Exclude city/location names (check against `metro_area` and `state_normalized` values)

4. **Semantic categorization.** Tag reported terms with categories:
   - `ai_tool`: AI coding tools, LLMs, specific models (copilot, cursor, claude, gpt, llm, rag, agent, mcp)
   - `ai_domain`: ML/AI as a domain (machine learning, deep learning, NLP, computer vision)
   - `tech_stack`: Specific technologies, frameworks, languages
   - `org_scope`: Organizational/scope language (ownership, end-to-end, cross-functional, stakeholder)
   - `mgmt`: Management/leadership (lead, mentor, manage, team, hire, coach)
   - `sys_design`: Systems/architecture (distributed systems, scalability, architecture)
   - `method`: Development methodology (agile, scrum, ci/cd, tdd)
   - `credential`: Formal requirements (years experience, BS/MS/PhD, certification)
   - `soft_skill`: Interpersonal (collaboration, communication, problem-solving)
   - `boilerplate`: Benefits/compensation/company-culture language (salary, benefits, compensation, pay, equity, bonus, dental, 401k, pto, culture, mission, values, diversity, inclusion, sponsorship, visa, employees, people)
   - `noise`: Residual after filtering (target <10%)

5. **Within-2024 calibration.** When comparing 2024 vs 2026, also compare arshkon (2024-04) vs asaniczka (2024-01) on the same metric where possible. This establishes baseline cross-source variability. If a 2024-to-2026 change is smaller than within-2024 cross-source variation, flag it as potentially artifactual.

6. **Keyword indicator validation.** For any keyword indicator you construct (management, AI, scope, etc.), sample 50 matches and assess whether they actually represent the intended concept. If a pattern matches generic language (e.g., `\bleading\b` matching "a leading company" instead of "leading a team"), flag it and report results with and without that pattern. Measurement artifacts in keyword indicators can inflate findings by 3-5x.

7. **Sampling protocol.** For any analysis based on a sample rather than the full dataset:
   - Document sample size, stratification method, and what was excluded
   - Report what fraction of each source/period/seniority group is represented
   - Prefer balanced period representation over proportional-to-population (avoids asaniczka domination)
   - For keyword pattern validation, stratify samples by period — pattern behavior may differ between 2024 and 2026
   - If a finding could change with a different sample, test with at least one alternative sample

8. **Source composition profiling.** Before using a single source as a baseline for cross-period comparisons (e.g., arshkon as the 2024 baseline), profile its top-20 employers and any obvious composition skews (industry where available, company size, geographic concentration). Differential composition across sources can produce false "temporal" signals — what looks like change between periods may just be different employer mixes. The Wave 1 company-concentration task surfaces this for the standard sources; check its findings before drawing cross-period conclusions.

## Sensitivity framework — apply to all analytical tasks

Every analytical task must report results under its primary specification AND under essential sensitivity checks. A finding is only robust if it survives its essential sensitivities.

Eight dimensions (referenced by letter in each task spec):

(a) **Aggregator exclusion.** Primary: include all rows. Alt: exclude `is_aggregator = true`. Rationale: aggregators have systematically different descriptions, seniority patterns, and template-driven requirements.

(b) **Company capping.** A few prolific employers dominate any analysis that aggregates over a corpus (term frequencies, topic models, co-occurrence networks, embedding centroids). For these, cap at 20-50 postings per `company_name_canonical` as the primary specification. For per-row metrics (rates, distributions) or company-level analyses, capping is not appropriate — it's a sensitivity check or N/A. Choose based on your unit of analysis.

(c) **Seniority operationalization.** Present results under `seniority_final` as the primary, plus the YOE-based proxy (`yoe_extracted <= 2` share by period) as the label-independent validator. Where the finding depends on `seniority_native` (e.g., as a sanity check), use arshkon-only — asaniczka has zero native entry labels. Material disagreement between `seniority_final` and the YOE-based proxy is itself a finding to report and investigate.

(d) **Description text source.** Primary: `description_core_llm` (filtered to `llm_extraction_coverage = 'labeled'`). Alt: raw `description`. No rule-based cleaned-text alternative exists — the former `description_core` was retired on 2026-04-10. Rationale: findings that only appear under raw `description` are boilerplate-driven and should be flagged as such.

(e) **Source restriction.** Primary: arshkon (2024-04) vs scraped (2026-03). Alt: arshkon + asaniczka pooled as 2024 baseline. Rationale: asaniczka is a different instrument; pooling increases power but introduces noise.

(f) **Within-2024 calibration (signal-to-noise).** Mandatory diagnostic. For every metric compared 2024-to-2026, also compute the arshkon-vs-asaniczka difference on the same metric. Signal-to-noise ratio: (cross-period effect size) / (within-2024 effect size). If ratio < 2, flag as "not clearly above instrument noise."

(g) **SWE classification tier.** Primary: all `is_swe = true`. Alt: exclude `swe_classification_tier = 'title_lookup_llm'` (retaining regex + embedding_high only). Rationale: title_lookup_llm has elevated false-positive rate.

(h) **LLM text coverage.** For text analyses: primary restricts to rows with `llm_extraction_coverage = 'labeled'` and uses `description_core_llm`. Report labeled-row coverage by period and source; thin cells must be flagged. This dimension is about sample restriction, not an alternative text column.

(i) **Indeed cross-platform validation.** For key findings (entry share, AI prevalence, description length), compute the same metric on Indeed scraped data. Indeed has no native seniority and is excluded from the Stage 9 LLM frame, so its `seniority_final` only carries Stage 5 strong-rule labels — the unknown rate will be high. Where Indeed coverage is too thin to support a seniority-stratified comparison, fall back to non-seniority-stratified metrics. If Indeed patterns match LinkedIn, findings are more robust. If they diverge, the finding may be LinkedIn-specific.

**Materiality threshold:** A finding is **materially sensitive** to a dimension if the alternative specification changes the main effect size by >30% or flips the direction.

**Sensitivity disagreement is itself a finding to investigate, not just to flag.** When a finding is materially sensitive, drill in: identify the rows/companies/terms driving the difference, characterize what they have in common, state the most likely mechanism, and recommend a follow-up that would resolve the question. A bare "this finding is materially sensitive to dimension X" without drilling in is insufficient — the mechanism behind the disagreement is often more interesting than either of the two estimates.

**Text source discipline (CRITICAL):** Use `description_core_llm` as the primary text column for all text-dependent analyses. The only alternative is raw `description`, and only for analyses that are demonstrably insensitive to boilerplate phrasing (binary keyword presence, rough length checks). NEVER mix rows that used cleaned text with rows that fell back to raw text without explicitly reporting the split and testing whether findings differ between the two subsets. The former rule-based `description_core` column is no longer available; any legacy shared artifact that still mixes rule-based text with LLM text is stale and should be rebuilt from the current pipeline before use.

**Instrument comparison is a first-class concern.** The 2024 Kaggle sources and the 2026 scraped source are different instruments: Kaggle text is unformatted (HTML-stripped), scraped text preserves markdown formatting (including **bold** headers and bullet points). Section classifiers must handle both formats. The within-2024 calibration (dimension f) is the primary mitigation.
```

---

## 2. Agent dispatch blocks

These are prepended (after the preamble) to each agent's prompt, before the task specs.

### Agent A — Wave 1: Data profile & seniority comparability (T01 + T02)

Profile the dataset (actual row counts, column coverage, semantic differences across sources) and produce the coverage heatmap that downstream tasks depend on. Then run the seniority comparability audit: can asaniczka `associate` serve as a junior proxy? Execute tasks T01 and T02.

### Agent B — Wave 1: Classifier quality (T03 + T04)

Evaluate seniority label quality and SWE classification accuracy. Cross-tabulate all seniority variants, compute agreement metrics, and test whether the RQ1 junior-share metric changes depending on which seniority column is used. Assess SWE classification via manual sampling of borderline cases. Execute tasks T03 and T04.

### Agent C — Wave 1: Dataset comparability & concentration (T05 + T06)

Test whether the three datasets are measuring the same thing by running pairwise comparisons (description length, company overlap, geographic/seniority/title distributions). Also assess company concentration and whether a few employers dominate findings. Execute tasks T05 and T06.

### Agent D — Wave 1: External benchmarks & power analysis (T07)

Compare our data against BLS OES occupation/state data and JOLTS information sector trends. Additionally, conduct a power and feasibility analysis for all planned cross-period comparisons — compute minimum detectable effect sizes and identify which analyses are well-powered vs underpowered. This requires web access to download benchmark data from FRED and BLS. Execute task T07.

### Agent Prep — Wave 1.5: Shared preprocessing

Build shared analytical artifacts that Wave 2+ agents will load instead of recomputing independently. Compute cleaned text, sentence-transformer embeddings, technology mention matrix, and company name stoplist. Save all to `exploration/artifacts/shared/`. Execute the shared preprocessing spec.

### Agent E — Wave 2: Distribution profiling (T08)

Compute baseline distributions for ALL available variables by period and seniority, with emphasis on anomaly detection and unexpected patterns. This task carries a heavy sensitivity framework — run essential sensitivity checks on all core findings. Read `exploration/reports/INDEX.md` for Wave 1 guidance. Load shared artifacts from `exploration/artifacts/shared/`. Execute task T08.

### Agent F — Wave 2: Posting archetype discovery (T09)

Discover natural posting archetypes through unsupervised methods. This is the primary methods comparison task — run BERTopic (primary) and NMF (comparison) on the same data and compare what each surfaces. Load shared embeddings and cleaned text from `exploration/artifacts/shared/`. Execute task T09.

### Agent G — Wave 2: Title evolution & requirements complexity (T10 + T11)

Map how the SWE title taxonomy has evolved between 2024 and 2026 — what titles emerged, disappeared, or changed meaning. Then quantify the structural complexity of job requirements (credential stacking, technology density, scope breadth). Load shared technology matrix from `exploration/artifacts/shared/` for T11 tech counting. Execute tasks T10 and T11.

### Agent H — Wave 2: Linguistic evolution & text discovery (T13 then T12)

**Run T13 first** (section anatomy, readability, tone), **then T12** (corpus comparison using T13's section classifier to strip boilerplate). T12 depends on T13's section classifier output to isolate genuine content changes from boilerplate expansion. Load shared artifacts from `exploration/artifacts/shared/`. Execute tasks T13 and T12 in that order.

### Agent V1 — Gate 2 Verification

Adversarial quality assurance after Wave 2. Re-derive the top 3-5 headline numbers from Wave 2 from scratch (independent code). Validate keyword patterns by sampling 50 matches stratified by period. Propose alternative explanations for each headline finding. Flag specification-dependent findings. Write `exploration/reports/V1_verification.md`.

### Agent I — Wave 2: Technology ecosystems & semantic landscape (T14 + T15)

Map the technology ecosystem — not just individual mentions but co-occurrence networks and natural skill bundles. Also validate using asaniczka's structured skills field. Then compute the full semantic similarity landscape across all period x seniority groups. Load shared technology matrix and embeddings from `exploration/artifacts/shared/`. Execute tasks T14 and T15.

### Agent J — Wave 3: Company strategies & geographic structure (T16 + T17)

Among companies appearing in both periods, cluster them by HOW their postings changed. Then analyze geographic market segmentation. Execute tasks T16 and T17.

### Agent V2 — Gate 3 Verification

Adversarial quality assurance after Wave 3. Re-derive top 3-5 Wave 3 headline numbers independently. Verify the cross-occupation DiD (T18) under alternative control definitions. Verify the decomposition (T16) under arshkon-only vs pooled 2024. Validate T22's corrected management patterns. Write `exploration/reports/V2_verification.md`.

### Agent K — Wave 3: Cross-occupation boundaries & temporal patterns (T18 + T19)

Compare SWE, SWE-adjacent, and control occupations to determine which changes are SWE-specific vs field-wide. Then estimate rates of change and characterize the temporal structure of our data. Execute tasks T18 and T19.

### Agent L — Wave 3: Seniority boundaries & senior role evolution (T20 + T21)

Measure how sharp the seniority boundaries are and whether they blurred or shifted between periods. Then conduct a deep dive into how senior SWE roles evolved. For T21's management language analysis, validate your keyword patterns by sampling matches — prior waves found that broad patterns can inflate management indicators by 3-5x. If T09 archetype labels exist in shared artifacts, use them for domain stratification. Execute tasks T20 and T21.

### Agent M — Wave 3: Ghost forensics & employer-usage divergence (T22 + T23)

Identify ghost-like and aspirational requirement patterns, with emphasis on whether AI requirements are more aspirational than traditional ones. Save validated management/scope patterns as a shared artifact for downstream use. Then compute the employer-requirement vs worker-usage divergence. Execute tasks T22 and T23.

### Agent O — Wave 3: Domain-stratified scope changes & LLM authorship detection (T28 + T29)

Two complementary analyses that depend on T09's archetype labels and the cleaned text artifact. T28 (priority): re-decompose scope and content changes by domain archetype now that T09's clusters are available — does scope inflation differ across Frontend, Embedded, Data, ML/AI? T29 (lower priority, exploratory): test the hypothesis that part of the apparent content change is downstream of recruiters using LLMs to draft job descriptions. Execute tasks T28 and T29.

### Agent N — Wave 4: Hypothesis generation, artifacts & synthesis (T24 + T25 + T26)

Read ALL reports. First, generate new hypotheses from the findings. Then produce interview elicitation artifacts. Finally, write the synthesis document. Execute tasks T24, T25, and T26.

### Agent P — Wave 5: Presentation (T27)

Read `exploration/reports/SYNTHESIS.md`, gate memos, and INDEX.md. Also read `docs/preprocessing-guide.md` and extract a minimal description of the preprocessing pipeline (stages, rationale, data structure, LLM prompts) to integrate into the site's methodology layer. Produce a ~20-25 slide MARP presentation for the research advisor and stakeholders. Follow the presentation principles in the orchestrator prompt (complete-sentence slide titles, one idea per slide, tell what you learned not what you did, frame corrections as rigor). Reference existing figures from `exploration/figures/`. Export to HTML and PDF via `npx @marp-team/marp-cli`. Execute task T27.

---

## 3. Task specs

### Wave 1 — Data Foundation

---

### T01. Data profile & column coverage `[Agent A]`

**Goal:** Profile the dataset and determine which columns are usable for cross-period analysis.

**Steps:**
1. Query the data to determine actual row counts by source, platform, and SWE/adjacent/control status. Report these as the authoritative counts — do not rely on documented numbers.
2. For every column, compute by source (arshkon, asaniczka, scraped) and by `is_swe` subset: non-null rate and distinct count.
3. Produce a coverage heatmap (columns x sources, colored by non-null rate).
4. Flag columns >50% null for any source used in cross-period comparisons.
5. Check `description_core_llm` coverage: report the `llm_extraction_coverage` distribution by source for SWE rows. This determines which rows can use high-quality text.
6. Note columns with DIFFERENT semantics across sources (e.g., `company_industry` has compound labels in scraped but single labels in arshkon).
7. **Key constraint mapping:** For each planned Wave 2-3 analysis category (text, seniority, geography, company, requirements), state the binding data constraint and severity.

**Output:** `exploration/reports/T01.md` + coverage heatmap PNG + CSV

### T02. Seniority comparability & label quality `[Agent A]`

**Goal:** Test whether asaniczka `associate` can serve as a junior proxy, and characterize how `seniority_final` compares against `seniority_native` on the source where both are available (arshkon).

This task focuses on seniority label quality and the asaniczka comparability question. Coverage and missingness are handled by T01.

**Steps:**
1. Document which seniority labels each source provides. Note the critical gap: asaniczka has zero native entry-level labels (so `seniority_native` cannot detect entry in asaniczka). All asaniczka entry signal in `seniority_final` comes from the Stage 10 LLM.
2. SWE-only native-label comparability audit:
   - Compare asaniczka `associate` (`seniority_native`) against arshkon `entry`, `associate`, and `mid-senior` (`seniority_native`)
   - Use exact `title_normalized` overlap, explicit junior/senior title-cue rates, `yoe_extracted`, and `seniority_final` distributions conditional on native label
   - State whether asaniczka `associate` behaves more like junior, lower-mid, mixed, or indeterminate
3. Decision rule: `usable as junior proxy` only if evidence is directionally close to arshkon `entry` on multiple signals.
4. Entry-level effective sample sizes per source under `seniority_final`: how many entry-level rows have YOE, metro_area, description_core_llm, etc.? Break down the entry rows by `seniority_final_source` so the reader sees which fraction came from strong rule vs the LLM.

**Output:** `exploration/reports/T02.md` with comparability audit tables and a clear verdict

### T03. Seniority label audit `[Agent B]`

**Goal:** Audit `seniority_final` against the available diagnostics. This is the one task in the exploration that interrogates the seniority labels themselves; everything downstream uses `seniority_final` directly without re-running the comparison.

**Steps:**
1. **`seniority_final_source` profile.** For SWE rows, report the distribution of `seniority_final_source` (`title_keyword`, `title_manager`, `llm`, `unknown`) by source and by period. This shows how the rule and LLM halves of `seniority_final` are composed.
2. **Rule-vs-LLM internal agreement (where both could fire).** Restrict to rows where `seniority_final_source = 'llm'` and inspect whether the LLM's answer is consistent with what a strong title rule WOULD have produced if one existed (e.g., spot-check by sampling 100 LLM-labeled rows whose titles contain weak seniority markers like "I/II/III"). Estimate routing-error rate qualitatively.
3. **`seniority_final` vs `seniority_native` (arshkon SWE only).** Cross-tabulate. Compute Cohen's kappa and per-class accuracy using native as the comparison reference. Repeat on scraped LinkedIn SWE. If accuracy differs between arshkon and scraped, that suggests temporal instability of the native classifier (the same LinkedIn label may mean different things across the 2024→2026 window).
4. **Junior share under three operationalizations.** Compute RQ1 junior share by period under:
   a. `seniority_final` (the production column)
   b. `seniority_native` arshkon-only (sanity check)
   c. The label-independent YOE-based proxy (`yoe_extracted <= 2` share)
   Do all three agree on direction? Magnitude? If they disagree, drill into the mechanism — is it differential native quality across snapshots, real market change, or LLM bias?
5. **Recommendation.** State plainly whether `seniority_final` looks defensible as the primary seniority column for the rest of the exploration. If it is, the rest of Wave 2/3 uses it without re-litigating. If it isn't (e.g., the rule and LLM halves disagree systematically with each other and with the YOE proxy), document the failure mode and propose a remediation.

**Output:** `exploration/reports/T03.md` with the source profile, the cross-tab + kappa table, the three-way junior-share comparison, and the recommendation. This report is the canonical seniority-quality reference for downstream agents.

### T04. SWE classification audit `[Agent B]`

**Goal:** Assess SWE classification quality after preprocessing fixes.

**Steps:**
1. SWE rows by `swe_classification_tier` breakdown
2. Sample 50 borderline SWE postings (`swe_confidence` 0.3-0.7 or tier `title_lookup_llm`): print title + 200 chars description, assess quality
3. Sample 50 borderline non-SWE (titles with "engineer"/"developer"/"software" but `is_swe = False`): same
4. Profile `is_swe_adjacent` and `is_control` rows: what titles/occupations? How distinct are they from SWE?
5. Estimated false-positive and false-negative rates
6. Verify no dual-flag violations: `(is_swe + is_swe_adjacent + is_control) > 1` should be 0
7. **Boundary cases:** Are there roles that straddle SWE/adjacent that might be misclassified differently across periods? (e.g., "ML Engineer", "Data Engineer", "DevOps Engineer")

**Output:** `exploration/reports/T04.md`

### T05. Cross-dataset comparability `[Agent C]`

**Goal:** Test whether dataset differences reflect real labor market changes vs artifacts.

**Steps (SWE, LinkedIn-only):**
1. Description length: KS test + overlapping histograms for `description_length` across 3 sources (the legacy `core_length` column no longer exists)
2. Company overlap: Jaccard similarity of `company_name_canonical` pairwise. Top-50 overlap.
3. Geographic: state-level SWE counts, chi-squared on state shares (multi-location postings — `is_multi_location = true` — have `location = "multi-location"` and no state; they are naturally excluded from state rollups)
4. Seniority: `seniority_final` distributions (exclude unknown), chi-squared pairwise
5. Title vocabulary: Jaccard of `title_normalized` sets. Titles unique to one period.
6. Industry: `company_industry` for arshkon vs scraped (asaniczka has no industry data)
7. **Artifact diagnostic:** For each metric with significant cross-dataset difference, can the difference be attributed to data collection method vs real change? Which comparisons are most trustworthy?
8. **Within-2024 calibration:** Run same comparisons between arshkon and asaniczka (both 2024) to establish baseline cross-source variability
9. **Platform labeling stability test.** For the top 20 SWE titles appearing in both arshkon and scraped:
   - Compare `seniority_native` distributions per title. If the same title has systematically different native labels across periods, that suggests platform relabeling rather than market change.
   - For title×seniority cells existing in both periods, compare YOE distributions. If YOE didn't change but frequency shifted, that's composition. If YOE changed too, it's content change.
   - Cross-validate with Indeed data: compute entry-level share on Indeed scraped rows using `seniority_final` (Stage 5 strong-rule labels only — Indeed is excluded from the LLM frame, so coverage is narrow). Where coverage is too thin, fall back to the YOE-based proxy. If Indeed shows similar patterns to LinkedIn, the LinkedIn platform artifact hypothesis weakens.

**Output:** `exploration/reports/T05.md` with test results, artifact assessment, calibration table, and platform stability assessment

### T06. Company concentration deep investigation `[Agent C]`

**Goal:** Understand the company-level shape of each source. The downstream wave 2/3 tasks aggregate over postings constantly — corpus-level term frequencies, entry-share computations, length distributions, AI-mention rates — and a few prolific employers can drive almost any of those aggregates. This task surfaces concentration patterns *before* substantive analysis runs, so downstream tasks know what to expect and which findings are concentration-driven.

**Why this matters (read first):** Prior runs of this exploration discovered, only after Wave 2 had been written and partially verified, that ~23% of the 2026 entry-labeled pool came from six companies posting the *same exact description* between 4 and 25 times each, that the entry-share rise survives or reverses depending on which de-concentration variant you apply, and that 91% of companies with substantial scraped presence have zero entry-labeled postings at all. None of these were surfaced by the original (lighter) version of this task. Treat the company axis as a first-class feature of the data, not an afterthought.

**Steps (SWE, all sources):**

1. **Concentration metrics per source.** HHI, Gini, top-1/5/10/20/50 share of `company_name_canonical` posting volume. Same metrics excluding aggregators (`is_aggregator = true`). Report as a single comparison table across sources.

2. **Top-20 employer profile per source.** For each source separately, list the top 20 companies by SWE posting volume. For each: posting count, share of source, industry where available, mean YOE, mean description length, and the within-company entry share (under `seniority_final` AND the YOE-based proxy). This is the foundational profiling that downstream tasks will reference when they need to know "is this finding driven by Amazon" or similar.

3. **Duplicate-template audit.** For each source, identify companies whose SWE postings collapse onto a small number of distinct `description_hash` values. Report the top-10 "duplicate-template" employers per source with: posting count, distinct description count, max-dup-ratio (postings / distinct descriptions). A company posting the same description 25 times is structurally different from a company posting 25 distinct descriptions, even if both contribute "25 rows" to corpus-level aggregates. **Stage 4 now collapses multi-location groups (same company + title + description across 2+ locations) into a single representative row flagged with `is_multi_location = True`; see the `is_multi_location` entry in `docs/preprocessing-schema.md`. Expect this audit to find ~0 duplicate-template groups within a single source after the fix — its role is now a verification, not a discovery. Surface any residual dup-templates (cross-company collisions, near-duplicate descriptions that differ only in boilerplate, etc.) as anomalies worth investigating.**

4. **Entry-level posting concentration (CRITICAL).** Entry-level posting is a specialized activity, not a market-wide one. For each source, compute:
   - How many companies post any entry-labeled SWE roles at all? (Under both `seniority_final` and the YOE-based proxy.)
   - What share of companies with >=5 SWE postings have ZERO entry-labeled rows?
   - For the companies that DO post entry roles, what is their distribution of entry-share-of-own-postings?
   - Which companies are entry-poster specialists (>50% of own SWE postings are entry)? Profile them.
   - Any cross-source comparisons of entry posting volume must be read against this concentration backdrop — entry-share trends are about a small subset of employers, not a market-wide shift.

5. **Within-company vs between-company decomposition.** Identify companies with >=5 SWE postings in BOTH arshkon and scraped. For this overlap panel, decompose the aggregate 2024-to-2026 change in entry share, AI mention prevalence, mean description length, and mean tech count into:
   - **Within-company component:** change holding company composition constant
   - **Between-company component:** change driven by different companies entering/exiting the panel
   - Run the entry-share decomposition under both `seniority_final` and the YOE-based proxy. If they disagree on direction, report both — the disagreement is itself a finding (see sensitivity-disagreement principle in the analytical preamble).

6. **Aggregator profile.** What fraction of SWE postings are from aggregators per source? Do aggregator postings differ systematically in seniority/length/requirements? Aggregator share shifts across sources are themselves a confound for any seniority-stratified comparison.

7. **New entrants.** How many 2026 companies have no 2024 match? What is their seniority and content profile vs returning companies?

8. **Per-finding concentration prediction.** For each major analysis category planned in Wave 2/3 (entry share, AI mention rate, description length, term frequencies, topic models, co-occurrence networks), predict whether it would be concentration-driven if computed naively over the full corpus. Recommend a default: cap, dedup, weight, or use as-is. This prediction table is the most important output of the task — it tells downstream agents what to do before they hit the same surprises.

**Essential sensitivities:** (a) aggregator exclusion. (b) Capping is the analysis subject of this task, so do not also apply it as a sensitivity.

**Output:** `exploration/reports/T06.md` with the concentration table, top-20 employer profile per source, duplicate-template audit, entry-poster concentration finding, decomposition table under multiple operationalizations, and the per-finding concentration prediction table. The prediction table is the deliverable that downstream wave 2/3 tasks should consult first.

### T07. External benchmarks & power analysis `[Agent D]`

**Goal:** Compare our data against BLS/JOLTS benchmarks. Assess statistical power and feasibility for all planned cross-period comparisons.

**Steps:**

*Part A — Feasibility table (primary output, drives all downstream decisions):*
1. Query the data for actual group sizes: entry-level, mid-senior, all SWE by source. Use these for power calculations.
2. Power analysis for cross-period comparisons: compute minimum detectable effect sizes (MDE) for binary and continuous outcomes at 80% power, alpha=0.05, for each key comparison (entry arshkon vs scraped, senior arshkon vs scraped, all SWE, pooled 2024 vs scraped).
3. Metro-level feasibility: How many metros have >=50 SWE per period? >=100? Which qualify for metro-level analysis? (Multi-location postings — `is_multi_location = true` — have `metro_area = NULL` and are excluded from per-metro counts; report the excluded count separately.)
4. Company overlap panel feasibility: How many companies have >=3 SWE postings in both arshkon and scraped? This determines T16's panel size.
5. Produce a feasibility summary table: `analysis_type | comparison | n_group1 | n_group2 | MDE_binary | MDE_continuous | verdict (well-powered / marginal / underpowered)`.

*Part B — External benchmarks (useful context, not blocking):*
6. Download BLS OES for SOC 15-1252 and 15-1256: state-level employment. Pearson r vs our state-level SWE counts.
7. Industry distribution: our SWE vs OES SWE industry (arshkon + scraped).
8. Download JOLTS information sector from FRED. Contextualize our data periods within the hiring cycle.
9. **Frame the data:** What population does our sample represent? What can and can't we generalize to?

**Output:** `exploration/reports/T07.md` + feasibility table CSV (the most important output). Target: r > 0.80 geographic.

---

### Wave 1.5 — Shared Preprocessing

---

### Shared preprocessing spec `[Agent Prep]`

**Goal:** Build shared analytical artifacts that multiple Wave 2+ agents need, preventing duplicate computation and ensuring consistency across agents.

**Steps:**
1. **Cleaned text column.** For all SWE LinkedIn rows (filtered by default SQL): use `description_core_llm` where `llm_extraction_coverage = 'labeled'`, otherwise fall back to raw `description` (the former rule-based `description_core` was retired on 2026-04-10 and must not be used). Strip company names using stoplist from all `company_name_canonical` values, remove standard English stopwords. Save as `exploration/artifacts/shared/swe_cleaned_text.parquet` with columns: `uid`, `description_cleaned`, `text_source` (which column was used: 'llm' or 'raw'), `source`, `period`, `seniority_final`, `seniority_3level`, `is_aggregator`, `company_name_canonical`, `metro_area`, `yoe_extracted`, `swe_classification_tier`, `seniority_final_source`. Downstream tasks that are sensitive to boilerplate must filter to `text_source = 'llm'`; tasks that only need recall (binary keyword presence) can use both.
2. **Sentence-transformer embeddings.** Using `all-MiniLM-L6-v2`, compute embeddings on first 512 tokens of `description_cleaned` for all rows in the cleaned text artifact. Process in batches of 256 to respect RAM limits. Save as `exploration/artifacts/shared/swe_embeddings.npy` (float32) with a companion `exploration/artifacts/shared/swe_embedding_index.parquet` mapping row index to `uid`.
3. **Technology mention binary matrix.** Using the ~100-120 technology taxonomy (define regex patterns for: Python, Java, JavaScript/TypeScript, Go, Rust, C/C++, C#, Ruby, Kotlin, Swift, Scala, PHP, React, Angular, Vue, Next.js, Node.js, Django, Flask, Spring, .NET, Rails, FastAPI, AWS, Azure, GCP, Kubernetes, Docker, Terraform, CI/CD, Jenkins, GitHub Actions, SQL, PostgreSQL, MongoDB, Redis, Kafka, Spark, Snowflake, Databricks, dbt, Elasticsearch, TensorFlow, PyTorch, scikit-learn, Pandas, NumPy, LangChain, RAG, vector databases, Pinecone, Hugging Face, OpenAI API, Claude API, prompt engineering, fine-tuning, MCP, LLM, Copilot, Cursor, ChatGPT, Claude, Gemini, Codex, Jest, Pytest, Selenium, Cypress, Agile, Scrum, TDD — expand to ~100+ with regex variations). Scan `description_cleaned` for each. Save as `exploration/artifacts/shared/swe_tech_matrix.parquet` (columns: `uid` + one boolean column per technology).
4. **Company name stoplist.** Extract all unique tokens from `company_name_canonical` values (tokenize on whitespace and common punctuation, lowercase, deduplicate). Save as `exploration/artifacts/shared/company_stoplist.txt`, one token per line.
5. **Structured skills extraction (asaniczka only).** Parse `skills_raw` from asaniczka SWE rows (comma-separated). Save parsed skills with uid as `exploration/artifacts/shared/asaniczka_structured_skills.parquet`.

6. **Within-2024 calibration table.** For ~30 common metrics (description_length, yoe_extracted median, tech_count mean, AI keyword prevalence, management indicator rate, scope term rate, soft skill rate, etc.), compute:
   - Arshkon value, asaniczka value, within-2024 effect size (Cohen's d or proportion difference)
   - Arshkon value, scraped value, cross-period effect size
   - Calibration ratio: cross-period / within-2024
   Save as `exploration/artifacts/shared/calibration_table.csv`. Wave 2+ agents load this instead of recomputing calibration independently.

**Output:** `exploration/artifacts/shared/` directory with all artifacts + a `README.md` documenting contents, row counts, `text_source` distribution, and build time.

**Fallback:** If embedding computation fails (OOM), save partial results and document which rows are covered. Wave 2 agents should check coverage and compute missing embeddings locally if needed.

**Note on LLM budget:** If Stage 9 LLM budget has been allocated for scraped data since the last run, the cleaned text artifact will have higher `text_source = 'llm'` coverage. Re-run this step after any LLM budget allocation to update the shared artifacts.

---

### Wave 2 — Open Structural Discovery

The goal of Wave 2 is to DISCOVER patterns in the data without imposing the RQ1-RQ4 framework. What natural structures exist? What changed? What's unexpected?

---

### T08. Distribution profiling & anomaly detection `[Agent E]`

**Goal:** Establish comprehensive baseline distributions and identify anomalies, surprises, and unexpected patterns across ALL available variables.

**Steps (SWE, LinkedIn-only):**
1. **Univariate profiling:** For every meaningful numeric and categorical column, compute distributions by period and by seniority (use `seniority_final` as primary). Produce side-by-side histograms/bar charts for at minimum: `description_length`, `yoe_extracted`, `seniority_final`, `seniority_3level`, `is_aggregator`, `metro_area` (top 15), `company_industry` (top 15 where available).
2. **Anomaly detection:** Flag any distribution that is bimodal, heavily skewed, or shows an unexpected pattern. Are there subpopulations hiding in the data?
3. **Native-label quality diagnostic (arshkon-only).** For `seniority_native = 'entry'` rows in arshkon, compute the YOE distribution (mean, median, share with YOE>=5, share with YOE<=2). If arshkon entry-labeled rows have an unexpected YOE profile, the native classifier may have temporal stability issues that affect any `seniority_native`-based sanity check on later snapshots. Asaniczka has zero native entry labels so cannot be profiled here. This is an important data-quality check; the answer determines whether `seniority_native` can be used as a sanity-check baseline at all.
4. **Within-2024 baseline calibration (arshkon vs asaniczka, mid-senior SWE only):**
   - Compare description length, AI keyword prevalence, organizational language, and top-20 tech stack
   - Compute Cohen's d or equivalent effect sizes
   - Produce calibration table: metric, within-2024 difference, 2024-to-2026 difference, ratio
5. **Junior share trends:** Entry share by period using `seniority_final` as the primary measure, validated against the label-independent YOE proxy (share with `yoe_extracted <= 2`). Where comparing against `seniority_native` would be informative, restrict to arshkon. Compute share both of all rows and of known-seniority rows only. If `seniority_final` and the YOE proxy disagree on direction, investigate WHY rather than picking one.
6. **What variables show the LARGEST changes between periods?** Rank all available metrics by effect size. This identifies where to look deeper.
7. **Domain × seniority decomposition (tests H1).** If T09 archetype labels are available (from `exploration/artifacts/shared/swe_archetype_labels.parquet`), compute entry share by domain archetype by period. Decompose the aggregate entry share change into:
   - **Within-domain component:** entry share change holding archetype composition constant
   - **Between-domain component:** change driven by the market shifting between domain archetypes (e.g., from frontend to ML/AI)
   If the between-domain component accounts for a substantial portion of the aggregate decline, the junior decline is partly a domain recomposition effect, not purely within-domain elimination.
8. **Company size stratification (where data allows).** `company_size` is available for arshkon (99%). Within arshkon, stratify entry share, AI prevalence, and tech count by company size quartile. Do large companies show different patterns? For cross-period analysis where `company_size` is unavailable, use posting volume per company as a rough proxy.

**Essential sensitivities:** (a) aggregator exclusion, (b) company capping, (c) seniority operationalization (`seniority_final` + YOE proxy), (e) source restriction, (f) within-2024 calibration
**Recommended sensitivities:** (g) SWE classification tier, (i) Indeed cross-platform

**Output:** `exploration/reports/T08.md` with plots, summary stats, anomaly flags, calibration table, ranked change list, and domain decomposition

### T09. Posting archetype discovery — methods laboratory `[Agent F]`

**Goal:** Discover natural posting archetypes through unsupervised methods, WITHOUT imposing pre-defined categories. This is also the primary **methods comparison task** — run BERTopic (primary) and NMF (comparison) on the same data and compare what each surfaces. Method agreement strengthens findings; disagreements reveal data structure.

**Steps:**
1. **Sample:** Up to 8,000 SWE LinkedIn postings with **balanced period representation** (~2,700 per period). Within each period, stratify by seniority. For the 2024 allocation, prefer arshkon rows over asaniczka (arshkon has entry-level labels and better text quality). Prefer rows with `text_source = 'llm'` from the shared cleaned text artifact. Record exact sample composition including text_source distribution. Use the SAME sample for all methods.
2. **Load shared artifacts** from `exploration/artifacts/shared/` — cleaned text and embeddings. Build TF-IDF from cleaned text. If shared artifacts unavailable, compute locally.

3. **Method A — BERTopic (primary):**
   - Run BERTopic with sentence-transformer embeddings, UMAP reduction, HDBSCAN clustering
   - Use `min_topic_size=30` to avoid micro-topics; experiment with 20 and 50 as well
   - Extract c-TF-IDF topic representations
   - Record: number of topics found, topic coherence, outlier/noise percentage
   - Use BERTopic's `topics_over_time` or manual period stratification to track topic evolution
   - Use BERTopic's `visualize_topics()`, `visualize_hierarchy()`, and `visualize_barchart()` — save as static images

4. **Method B — NMF (comparison):**
   - Run NMF on TF-IDF matrix with k=5,8,12,15 components
   - NMF often produces more interpretable topics than LDA on medium-length documents
   - Extract top 20 terms per component

5. **Method comparison:**
   - **Topic alignment:** Between BERTopic and NMF (by top-term overlap). Which topics are **method-robust** (found by both methods)? Which are method-specific?
   - **Cluster stability:** 3 runs with different seeds each. How stable are the assignments? (Adjusted Rand Index between runs.)
   - **Interpretability ranking:** Subjectively rank which method produces the most interpretable and useful topics for this specific data.
   - **Noise handling:** BERTopic/HDBSCAN identifies outlier documents. What % are outliers? What do they look like?
   - Produce a **methods comparison table:** method, # topics, coherence/stability, noise %, interpretability notes.

6. **Characterization (using best method's clusters):** For each archetype:
   - Top 20 terms (by c-TF-IDF or centroid weight)
   - Seniority distribution (what % of each seniority level falls in this cluster?)
   - **Entry-level share** of known seniority within each archetype, by period. This is critical for H1: if ML/AI has structurally lower entry share AND grew from 4% to 27%, the aggregate entry decline may be driven by domain composition.
   - Period distribution (what % of each period falls in this cluster?)
   - Average description length, YOE, tech count
   - Give each cluster a descriptive name based on its content, not based on RQ1-RQ4

7. **Temporal dynamics:** How did archetype proportions change from 2024 to 2026? Which grew? Which shrank? Are there archetypes that only exist in one period?

8. **Visualization:** UMAP (2D) of the embeddings, colored by: (a) best-method clusters, (b) period, (c) seniority. Three separate plots. Also produce the same with PCA for comparison — does the visual story change?

9. **Key discovery question:** Do the clusters align with seniority levels (entry/mid/senior map to different clusters)? Or do they align with something else entirely (industry, role type, tech stack, company size)? The answer reveals the dominant structure of the market. Compute Normalized Mutual Information (NMI) between cluster assignments and seniority, period, and tech domain to quantify this.

10. **Save cluster labels for downstream use.** Save the best method's cluster assignments as `exploration/artifacts/shared/swe_archetype_labels.parquet` (columns: `uid`, `archetype`, `archetype_name`). This allows T11, T16, T20 and other downstream tasks to stratify by domain archetype.

**Essential sensitivities:** (a) aggregator exclusion, (d) description text source
**Recommended sensitivities:** (g) SWE classification tier

**Output:** `exploration/reports/T09.md` + methods comparison table + cluster plots + cluster characterization CSV + archetype labels artifact + method recommendation for downstream use

### T10. Title taxonomy evolution `[Agent G]`

**Goal:** Map how the SWE title landscape has evolved between 2024 and 2026.

**Steps (SWE, LinkedIn-only):**
1. **Title vocabulary comparison:**
   - Titles appearing in scraped 2026 but NOT in arshkon 2024 (truly new titles)
   - Titles appearing in arshkon 2024 but NOT in scraped 2026 (disappeared titles)
   - For high-frequency titles appearing in both: how did their seniority distribution change?
2. **Title concentration:** Are there fewer or more unique titles per 1,000 SWE postings in 2026 vs 2024? Is the title space becoming more standardized or more fragmented?
3. **Compound/hybrid titles:** Count titles containing AI-related terms (AI, ML, machine learning, data, LLM, agent). How did their share change?
4. **Title-to-content alignment:** For the 10 most common titles appearing in BOTH periods, compute TF-IDF cosine similarity between 2024 and 2026 descriptions with that title. Are the same titles being used for different roles?
5. **Title inflation/deflation signals:** For titles with clear seniority markers (e.g., "senior", "lead", "principal", "staff", "junior", "associate"), track their share over time.
6. **Emerging role categories:** Group new 2026 titles by theme. Are there coherent new role types (e.g., AI engineering, platform engineering, reliability engineering)?

**Essential sensitivities:** (a) aggregator exclusion
**Recommended sensitivities:** (c) seniority operationalization

**Output:** `exploration/reports/T10.md` + title evolution tables + new/disappeared title lists

### T11. Requirements complexity & credential stacking `[Agent G]`

**Goal:** Quantify the structural complexity of what employers are asking for, beyond simple "scope inflation."

**Steps (SWE, LinkedIn-only):**
1. **Build a requirements feature extractor.** For each posting's description, count:
   - Distinct technology mentions (from a ~80-100 tech taxonomy — languages, frameworks, cloud, tools)
   - Distinct soft skill terms (communication, collaboration, problem-solving, leadership, teamwork, etc.)
   - Distinct organizational scope terms (ownership, end-to-end, cross-functional, stakeholder, autonomous, initiative)
   - Education level mentioned (none, BS, MS, PhD — take highest)
   - YOE from `yoe_extracted`
   - Management/leadership indicators (manage, lead, mentor, coach, hire, team)
   - AI-specific requirements (any AI tool or domain mention)
2. **Complexity metrics per posting:**
   - `tech_count`: number of distinct technologies
   - `requirement_breadth`: total distinct requirement types across all categories
   - `credential_stack_depth`: number of distinct requirement CATEGORIES with at least one mention (tech + education + YOE + soft skills + scope + management + AI = max 7)
   - `tech_density`: tech_count per 1K chars (length-normalized)
   - `scope_density`: org_scope terms per 1K chars
3. **Compare distributions** of all complexity metrics by period x seniority.
4. **The credential stacking question:** Are 2026 postings asking for MORE types of things simultaneously? (Not just more tech, but tech + scope + soft skills + AI + YOE all in one posting.)
5. **Entry-level complexity specifically:** How did entry-level requirement complexity change? Compare entry 2024 vs entry 2026 on all metrics.
6. **Management indicator deep dive:** Report the top 10 specific terms triggering the management indicator, separately for 2024 and 2026. Define two tiers: `management_strong` (manage, mentor, coach, hire, direct reports, performance review, headcount) and `management_broad` (includes lead, team, stakeholder, coordinate). Report both tiers separately so reviewers can assess whether the indicator captures genuine management language vs generic collaboration language.
7. **Domain-stratified scope inflation.** If T09 archetype labels are available (from `exploration/artifacts/shared/swe_archetype_labels.parquet`), stratify the entry-level scope inflation analysis by domain archetype. Report whether scope inflation differs across Frontend/Web, Embedded/Systems, Data/Analytics, and ML/AI clusters. This tests whether the redefinition is AI-domain-driven or market-wide.
8. **Outlier analysis:** What do the most complex postings look like? (Top 1% by requirement_breadth.) Are they real or template-bloated?

**Essential sensitivities:** (a) aggregator exclusion, (b) company capping, (c) seniority operationalization, (f) within-2024 calibration

**Output:** `exploration/reports/T11.md` + complexity distribution plots + per-seniority comparison tables + management term breakdown + domain-stratified scope inflation

### T12. Open-ended text evolution `[Agent H]`

**Dependency:** Run T13 first (or at least its section anatomy portion). T12 uses T13's section classifier to separate core content from boilerplate sections.

**Goal:** Discover the terms and phrases that changed MOST between periods, using open-ended comparison rather than pre-defined pairs.

**Steps (SWE, LinkedIn-only):**
1. **Text preparation (critical):**
   - Load shared cleaned text from `exploration/artifacts/shared/swe_cleaned_text.parquet`. Use `description_cleaned` column. Report `text_source` distribution.
   - Load company stoplist from shared artifacts.
2. **Primary comparison: arshkon (2024) vs scraped (2026), ALL SWE.**
   - Fightin' Words (Monroe et al.) or log-odds ratio with informative Dirichlet prior
   - Top 100 distinguishing terms in EACH direction (2024-heavy and 2026-heavy)
   - Tag every term with semantic category from preamble taxonomy (including `boilerplate` category)
   - Produce category-level summary: what % of distinguishing terms are ai_tool, org_scope, tech_stack, boilerplate, etc.?
3. **Section-filtered comparison (if T13 section classifier available).** Re-run the Fightin' Words comparison on ONLY the requirements/responsibilities sections (stripping benefits, about-company, and legal sections). This isolates genuine content evolution from boilerplate expansion. Compare the top-100 lists from full-text vs section-filtered — which terms appear only in the full-text version (boilerplate-driven) vs both (genuine)?
4. **Emerging terms** (>1% in 2026, <0.1% in 2024): artifact-filtered, categorized
5. **Accelerating terms** (existed in 2024 but grew >3x)
6. **Disappearing terms** (>1% in 2024, <0.1% in 2026)
7. **Secondary comparisons (if sample sizes allow):**
   - Entry 2024 vs Entry 2026
   - Mid-senior 2024 vs Mid-senior 2026
   - **Entry 2026 vs Mid-senior 2024 (relabeling diagnostic):** This tests whether 2026 entry postings resemble relabeled 2024 senior postings (seniority-content dominant) or whether the changes represent a temporal shift applied to entry-level roles (period-effect dominant). If period-effect dominates, scope inflation is adding NEW dimensions rather than importing existing senior requirements downward. Interpret the result in this frame.
   - Within-2024: arshkon mid-senior vs asaniczka mid-senior (calibration — how much change is instrument noise?)
8. **Bigram analysis:** Phrase-level changes (prompt engineering, AI agent, code review) are often more informative than unigrams.
9. **BERTopic cross-validation:** Fit BERTopic on the combined corpus with period as class variable. Which topics are most period-specific? Compare with Fightin' Words.
10. **Report n per corpus for every comparison.** Flag any with n < 100.

**Essential sensitivities:** (a) aggregator exclusion, (b) company capping, (d) description text source, (e) source restriction, (f) within-2024 calibration
**Recommended sensitivities:** (c) seniority operationalization, (g) SWE classification tier

**Output:** `exploration/reports/T12.md` + categorized term tables (CSV) + category summary figure

### T13. Linguistic & structural evolution `[Agent H]`

**Goal:** Analyze how the FORM (not just content) of job postings evolved — readability, structure, tone, and anatomy.

**Steps (SWE, LinkedIn-only):**
1. **Readability metrics** (use the `textstat` package):
   - Flesch-Kincaid Grade Level (`textstat.flesch_kincaid_grade`)
   - Flesch Reading Ease (`textstat.flesch_reading_ease`)
   - Gunning Fog Index (`textstat.gunning_fog`)
   - Average sentence length (`textstat.avg_sentence_length`)
   - Vocabulary richness: type-token ratio on first 1,000 chars (to normalize for length)
   - Lexicon count and syllable count per posting
   - Compare all metrics by period x seniority
   - **Note:** Run on a sample (e.g., 2,000 per period) rather than the full corpus if performance is an issue
2. **Description section anatomy.** Define a section classifier using regex for common JD sections:
   - Role summary / About the role
   - Responsibilities / What you'll do
   - Requirements / Qualifications / What you'll need
   - Preferred / Nice-to-have
   - Benefits / Perks / Compensation
   - About the company
   - Legal / EEO / Equal opportunity
   - Unclassified
   - Estimate character count per section for each posting. Compute median section proportions by period x seniority.
3. **What's driving the 56% length growth?** Stacked bar chart: description composition by period. Did requirements grow? Or benefits/boilerplate/about-company? This is critical for interpreting whether "more requirements" is a real signal or just longer postings.
4. **Tone markers:**
   - Imperative density: "you will", "you'll", "must", "should" per 1K chars
   - Inclusive language: "we", "our team", "you'll join" per 1K chars
   - Formal vs informal: passive constructions vs active/direct address
   - Marketing language: "exciting", "innovative", "cutting-edge", "world-class" per 1K chars
5. **Entry-level specifically:** How did the structure and tone of entry-level JDs change compared to mid-senior?

**Essential sensitivities:** (a) aggregator exclusion, (d) description text source

**Output:** `exploration/reports/T13.md` + readability comparison table + stacked section chart + tone metrics

### T14. Technology ecosystem mapping `[Agent I]`

**Goal:** Map not just individual technology mentions, but how technologies co-occur and form natural skill bundles — and how those bundles changed.

**Steps (SWE, LinkedIn-only):**
1. **Technology taxonomy (~100-120 technologies).** Define regex patterns for:
   - **Languages:** Python, Java, JavaScript/TypeScript, Go, Rust, C/C++, C#, Ruby, Kotlin, Swift, Scala, PHP
   - **Frontend:** React, Angular, Vue, Next.js, Svelte
   - **Backend:** Node.js, Django, Flask, Spring, .NET, Rails, FastAPI
   - **Cloud/DevOps:** AWS, Azure, GCP, Kubernetes, Docker, Terraform, CI/CD, Jenkins, GitHub Actions, ArgoCD
   - **Data:** SQL, PostgreSQL, MongoDB, Redis, Kafka, Spark, Snowflake, Databricks, dbt, Elasticsearch
   - **AI/ML traditional:** TensorFlow, PyTorch, scikit-learn, Pandas, NumPy, Jupyter
   - **AI/LLM new:** LangChain, LangGraph, RAG, vector databases, Pinecone, ChromaDB, Hugging Face, OpenAI API, Claude API, prompt engineering, fine-tuning, MCP, agent frameworks, LLM
   - **AI tools:** Copilot, Cursor, ChatGPT, Claude, Gemini, Codex
   - **Testing:** Jest, Pytest, Selenium, Cypress, JUnit
   - **Practices:** Agile, Scrum, TDD, CI/CD
2. **Mention rates:** For each technology, compute % of postings mentioning it, by period x seniority. Use binary (any mention) as primary metric.
3. **Technology co-occurrence network:**
   - Compute phi coefficient (binary co-occurrence) for all technology pairs with sufficient frequency (>1% in at least one period)
   - Build adjacency matrix, threshold at phi > 0.15
   - Use networkx community detection (Louvain or greedy modularity) to find natural technology clusters
   - Compare 2024 vs 2026 community structure: which clusters gained members? Which fragmented? Which are new?
4. **Rising, stable, declining:** Classify each technology by its trajectory. Produce a "technology shift" heatmap.
5. **Stack diversity:** How many distinct technologies does the median posting mention? By period x seniority. Is tech breadth increasing or specializing?
6. **AI integration pattern:** Among postings mentioning AI tools/LLM, what traditional technologies co-occur? (Is AI adding to existing stacks, or replacing components?) **Length-normalization check:** The finding that AI-mentioning postings have more technologies could partly be an artifact of AI postings being longer. Compute tech density (techs per 1K chars) for AI-mentioning vs non-AI postings. Report both raw count and density to quantify the length confound.
7. **Structured skills baseline (asaniczka only).** Load parsed skills from `exploration/artifacts/shared/asaniczka_structured_skills.parquet`. Compute frequency table of all distinct skills across 23K SWE rows. Produce top-100 skills list.
8. **Structured vs extracted validation.** Compare technology frequencies from structured `skills_raw` (step 7) against description-extracted tech frequencies (step 2) for asaniczka SWE. Compute rank correlation. Where they diverge: is the structured field capturing things the regex misses, or vice versa?
9. **Seniority-level skill differences from structured data.** Using asaniczka's parsed skills and seniority labels: which skills are significantly more associated with entry-level vs mid-senior? (Chi-squared per skill, with Bonferroni or FDR correction.) This provides a structured-data baseline for the seniority boundary question.

**Essential sensitivities:** (a) aggregator exclusion, (b) company capping, (f) within-2024 calibration
**Recommended sensitivities:** (g) SWE classification tier

**Output:** `exploration/reports/T14.md` + tech heatmap + co-occurrence network visualization + community comparison + structured skills baseline CSV + structured-vs-extracted validation

### T15. Semantic similarity landscape & convergence analysis `[Agent I]`

**Goal:** Map the full semantic structure of the SWE posting space, visualize how it changed between periods, and test whether seniority levels are converging. Compare text representations and dimensionality reduction methods.

**Steps:**
1. **Sample and embed:** Stratified sample of up to 2,000 SWE postings per period x seniority_3level group (up to ~12,000 total). Load shared embeddings and cleaned text from `exploration/artifacts/shared/`. Build TF-IDF from cleaned text, reduce via SVD to 100 components. Report `text_source` distribution.

2. **Structural map.** Define groups as period x seniority_3level (and optionally period x archetype if T09 labels available). The central question: what is the dominant source of variation — period, seniority, or domain? Compute centroid similarity matrix across all groups under both representations.

3. **Convergence analysis (improved methodology).**
   - Use **trimmed centroids** (remove 10% most distant embeddings from each group centroid) to reduce outlier sensitivity.
   - Compute centroid similarity between seniority levels within each period.
   - **Within-2024 calibration is mandatory:** if the asaniczka→arshkon similarity shift exceeds the arshkon→scraped shift, the convergence signal doesn't survive calibration. State this explicitly.
   - If T09 archetype labels are available, compute convergence WITHIN each domain archetype separately. Are ML/AI entry-senior roles converging faster than Frontend entry-senior?

4. **Within-group dispersion.** Average pairwise cosine within each group. More/less homogeneous over time?

5. **Visualization — the most important output for communicating the structure.**
   - UMAP (2D) with **density contours** per group, colored by period x seniority. Show how clusters shift and overlap between periods.
   - Also produce PCA and t-SNE for comparison.
   - If T09 archetype labels available, produce a UMAP colored by archetype x period showing how domain clusters evolved.
   - Annotate with movement arrows between period centroids for each seniority level.
   - These visualizations should be publication-quality — they may be key paper figures.

6. **Nearest-neighbor analysis.** For each 2026 entry posting, find 5 nearest 2024 neighbors. What seniority are they? Repeat with TF-IDF. Report excess over base rate (not just raw percentage).

7. **Representation robustness table.** For each finding: does it hold under both embedding and TF-IDF?

8. **Outlier identification.** Most unlike their seniority peers — what makes them different?

**Essential sensitivities:** (a) aggregator exclusion, (c) seniority operationalization, (d) description text source, (e) source restriction, (f) within-2024 calibration
**Recommended sensitivities:** (g) SWE classification tier

**Output:** `exploration/reports/T15.md` + density-contour UMAP plots + similarity heatmaps + robustness table + nearest-neighbor analysis

---

### Wave 3 — Market Dynamics & Cross-cutting Patterns

Wave 3 builds on Wave 2's discoveries to examine market structure, actors, and boundaries.

---

### T16. Company hiring strategy typology `[Agent J]`

**Goal:** Among companies appearing in both periods, discover different hiring strategy trajectories — how are different companies changing?

**Steps (SWE, LinkedIn-only):**
1. **Overlap panel:** Identify companies with >=3 SWE postings in BOTH arshkon and scraped.
2. **Per-company change metrics:** For each overlap company, compute 2024-to-2026 change in:
   - Entry share — compute under `seniority_final` AND under the YOE-based proxy (`yoe_extracted <= 2` share). If the two disagree at the company level, treat it as a per-company measurement signal worth examining.
   - AI keyword prevalence (binary per posting)
   - Mean description length
   - Mean tech count
   - Mean org_scope term count
3. **Cluster companies by their change profile.** k-means on the change vectors. Are there distinct strategies? Name them (e.g., "AI-forward", "traditional hold", "scope inflator", "downsizer").
4. **Within-company vs between-company decomposition:** For entry share, AI prevalence, and description length, how much of the aggregate 2024-to-2026 change is driven by within-company change vs different companies entering/exiting the sample? Run the entry-share decomposition under both `seniority_final` (the primary) and the YOE-based proxy (label-independent validator). If results disagree in direction, report both and discuss the mechanism — this is a critical methodological finding, not a problem to bury. If T09 archetype labels are available, add a domain dimension: decompose the entry share change into within-domain, between-domain, and between-company components.
5. **Within-company scope inflation (if validated patterns available).** If T22's validated management/scope patterns are available (from shared artifacts), compute within-company change in entry-level scope indicators for the overlap panel. This is the cleanest test of scope inflation: same companies across periods.
6. **New market entrants:** Profile companies in 2026 with no 2024 match. What industries? How do their postings compare?
7. **Aggregator vs direct employer:** Compare change patterns. Are aggregators showing different trends?

**Essential sensitivities:** (a) aggregator exclusion, (c) seniority operationalization (`seniority_final` + YOE proxy for entry metrics)
**Recommended sensitivities:** (b) company capping

**Output:** `exploration/reports/T16.md` + company cluster characterization + decomposition results (both pooled and arshkon-only)

### T17. Geographic market structure `[Agent J]`

**Goal:** Map geographic heterogeneity in SWE market changes.

**Note on multi-location postings:** Stage 4 collapses multi-location posting groups (one role syndicated to multiple metros, e.g., a remote LinkedIn listing fanned out across 27 metro pages) into a single representative row with `is_multi_location = True`, `location = "multi-location"`, and `metro_area = NULL`. These rows are naturally excluded from any `metro_area`-keyed rollup below — this is correct because the posting does not belong to a single metro. Report the count of `is_multi_location = True` SWE postings alongside your metro-level analysis so readers understand the pool that was excluded from metro rollups. Do **not** attempt to expand multi-location rows back out to individual metros.

**Steps (SWE, LinkedIn-only, using `metro_area`):**
1. **Metro-level metrics** for each metro with >=50 SWE postings per period:
   - Entry share (`seniority_final`)
   - AI keyword prevalence (broad + AI-tool-specific)
   - Org scope language composite (ownership + cross-functional + end-to-end)
   - Median description length
   - Tech diversity (median distinct tech mentions)
2. **Rank metros** by magnitude of change on each metric.
3. **Geographic patterns:** Is the entry decline concentrated in tech hubs (SF, NYC, Seattle, Austin) or uniform? Is the AI surge concentrated or uniform?
4. **Metro-level correlation:** Do metros with larger AI surges show larger entry declines? Compute correlations between metro-level changes.
5. **Remote work dimension:** For scraped 2026 data, what share of postings are remote by metro? How does remote share relate to other metrics?
6. **Domain archetype geographic distribution.** If T09 archetype labels are available (from `exploration/artifacts/shared/swe_archetype_labels.parquet`), compute archetype distribution by metro. Which metros are disproportionately ML/AI vs Frontend vs Embedded? Did the geographic concentration of domain archetypes change between periods?
7. **Metro heatmap:** metros x metrics, colored by change magnitude.

**Essential sensitivities:** (a) aggregator exclusion, (b) company capping

**Output:** `exploration/reports/T17.md` + metro heatmap + correlation analysis + domain archetype geographic distribution

### T18. Cross-occupation boundary analysis `[Agent K]`

**Goal:** Determine which observed changes are SWE-specific vs field-wide, and whether occupation boundaries are shifting.

**Steps:**
1. **Parallel trends:** For SWE, SWE-adjacent, and control groups, compute:
   - Seniority distribution by period
   - AI keyword prevalence by period
   - Description length by period
   - Org scope language by period
   - Tech mention count by period
2. **SWE-specificity test:** If control shows the same patterns, this suggests confounding by macro trends. Compute the difference-in-differences: (SWE change) - (control change) for each metric.
3. **Boundary shift analysis:** Focus on the SWE <-> SWE-adjacent boundary.
   - Sample 200 SWE and 200 SWE-adjacent descriptions from each period
   - Compute TF-IDF cosine similarity between the two groups, by period
   - Is the SWE-adjacent group becoming MORE similar to SWE over time? (boundary blurring)
   - What terms are migrating from SWE to adjacent roles?
4. **Specific adjacent roles:** For the top SWE-adjacent titles (data engineer, network engineer, data scientist, etc.), how did their descriptions change? Are any of them becoming indistinguishable from SWE?
5. **AI adoption gradient:** Plot AI keyword prevalence across all three groups. Is there a clear gradient (SWE > adjacent > control), and is it widening or narrowing?

**Essential sensitivities:** (a) aggregator exclusion, (g) SWE classification tier

**Output:** `exploration/reports/T18.md` + parallel trends plots + boundary similarity analysis

### T19. Temporal patterns & rate-of-change estimation `[Agent K]`

**Goal:** Characterize the temporal structure of our data and estimate rates of change in the SWE market. Our data consists of discrete snapshots (Jan 2024, Apr 2024, Mar 2026), not a continuous time series — this task works within that constraint honestly.

**Data context:** The scraped data covers approximately 8 days (March 20-27, 2026). Arshkon spans April 5-20, 2024 (~3 weekly bins). Asaniczka spans January 12-17, 2024 (6 days). We have discrete snapshots, not a continuous time series.

**Entry-level rates involving asaniczka depend on the operationalization.** Asaniczka has zero native entry labels, so `seniority_native` cannot detect asaniczka entry — only arshkon supports a `seniority_native`-based 2024 baseline. For rate-of-change estimation:
- Under `seniority_final`: all three snapshots can be included. Asaniczka entry signal in `seniority_final` comes via the Stage 10 LLM (the rule half is unknown for asaniczka where titles lack strong keywords). Report the three-snapshot rate.
- Under `seniority_native`: arshkon-only sanity check. Report the arshkon vs scraped rate.
- Under the YOE-based proxy (label-independent): all three snapshots can be included.
Report the rate-of-change under each operationalization and discuss any disagreement.

**Steps:**
1. **Rate-of-change estimation.** For key metrics (entry share of known seniority, AI keyword prevalence, median description length, median tech count, org scope density):
   - Compute value at each snapshot: asaniczka (Jan 2024), arshkon (Apr 2024), scraped (Mar 2026)
   - Within-2024 annualized rate: (arshkon value - asaniczka value) / 3 months * 12
   - Cross-period annualized rate: (scraped value - arshkon value) / 23 months * 12
   - Acceleration ratio: cross-period annualized rate / within-2024 annualized rate
   - If acceleration ratio >> 1 for a metric, something changed faster after 2024 than during 2024
   - Produce a rate-of-change comparison table

2. **Within-arshkon stability.** Arshkon spans ~2 weeks with 3 weekly bins (April 1-7: ~397 SWE, April 8-14: ~912, April 15-20: ~3,715). Using `date_posted`, check whether key metrics (AI keyword prevalence, description length, tech count) vary significantly across weeks. If they do, our "April 2024" snapshot has internal heterogeneity.

3. **Scraper yield characterization.** Examine daily SWE posting counts across scrape dates. Is the first day an accumulated backlog while subsequent days capture new flow? Compare content, seniority distribution, and AI mention rates across scrape dates. This helps calibrate whether our scraped snapshot represents stock (accumulated postings) or flow (new postings).

4. **Posting age analysis.** Examine `posting_age_days` for scraped rows where available (limited coverage — only ~49 SWE rows have this field). Characterize what we can. If posting ages cluster at specific values, this reveals the market's posting lifecycle.

5. **Within-March stability and day-of-week analysis.** With ~8 days of scraped data, check: are key metrics (AI mention rate, seniority distribution, description length) stable across days? Are there day-of-week effects? This tells us whether our scraped snapshot is internally consistent.

6. **Timeline contextualization.** Place our three snapshots on a timeline and annotate with major AI tool releases between them: GPT-4 (Mar 2023), Claude 3 (Mar 2024), GPT-4o (May 2024), Claude 3.5 Sonnet (Jun 2024), o1 (Sep 2024), DeepSeek V3 (Dec 2024), GPT-4.5 (Feb 2025), Claude 3.6 Sonnet (Apr 2025), Claude 4 Opus (Sep 2025), Gemini 2.5 Pro (Mar 2026). This provides qualitative temporal context even with only 3 data points.

**Essential sensitivities:** (none strictly essential for temporal analysis)
**Recommended sensitivities:** (e) source restriction, (f) within-2024 calibration (conceptually backbone of the rate-of-change estimation)

**Output:** `exploration/reports/T19.md` + rate-of-change comparison table + within-arshkon stability check + within-March stability analysis

### T20. Seniority boundary clarity `[Agent L]`

**Goal:** Measure how sharp the boundaries between seniority levels are and whether they blurred between periods. This goes BEYOND the "relabeling hypothesis" to map the full boundary structure.

**Steps (SWE, LinkedIn-only, seniority_final != unknown):**
1. **Feature extraction:** For each SWE posting, build a feature vector from:
   - `yoe_extracted` (numeric, impute median where null)
   - Tech count (distinct technologies mentioned)
   - AI mention (binary)
   - Org scope density (scope terms per 1K chars)
   - Management language density (mgmt terms per 1K chars)
   - Description length (normalized)
   - Education level (ordinal: none=0, BS=1, MS=2, PhD=3)
2. **Boundary discriminability:** For each pair of adjacent seniority levels (entry<->associate, associate<->mid-senior, mid-senior<->director):
   - Train a logistic regression classifier (L2 regularized, sklearn)
   - Compute AUC on held-out data (stratified 5-fold cross-validation)
   - Record top 5 discriminating features for each boundary
   - Repeat for 2024 data and 2026 data SEPARATELY
3. **Boundary change:** Compare AUC between periods for each boundary. If AUC decreased, the boundary blurred. If AUC increased, the boundary sharpened.
4. **What drives separation?** For each boundary, what features matter most? And did the FEATURES that separate levels change between periods?
5. **The "missing middle" question:** Is the associate level becoming more like entry, more like mid-senior, or disappearing? Compute distances.
6. **Domain-stratified boundary analysis.** If T09 archetype labels are available (from `exploration/artifacts/shared/swe_archetype_labels.parquet`), run the boundary analysis within each domain archetype separately. Does boundary blur/sharpening differ across ML/AI vs Frontend vs Embedded vs Data domains?
7. **Full similarity matrix** using the structured features (not text): compute average feature profiles per seniority x period and present as a heatmap.

**Essential sensitivities:** (a) aggregator exclusion, (c) seniority operationalization (`seniority_final` + YOE proxy)
**Recommended sensitivities:** (g) SWE classification tier

**Output:** `exploration/reports/T20.md` + AUC comparison + feature importance analysis + boundary heatmap + domain-stratified results

### T21. Senior role evolution deep dive `[Agent L]`

**Goal:** Go deep on how senior SWE roles specifically are evolving — not just management-to-orchestration, but the full picture.

**Steps (SWE, LinkedIn-only, seniority_final IN ('mid-senior', 'director')):**
1. **Language profiles.** Define three profiles (not just two):
   - **People management:** manage, mentor, coach, hire, interview, grow, develop talent, performance review, career development, 1:1, headcount, people management, team building, direct reports
   - **Technical orchestration:** architecture review, code review, system design, technical direction, AI orchestration, agent, workflow, pipeline, automation, evaluate, validate, quality gate, guardrails, prompt engineering, tool selection
   - **Strategic scope:** stakeholder, business impact, revenue, product strategy, roadmap, prioritization, resource allocation, budgeting, cross-functional alignment
   **Validate your patterns:** Sample 50 matches for each profile's key patterns and check precision. Generic terms like "leading", "leadership", "strategic" used as adjectives rather than role verbs have been shown to inflate management indicators by 3-5x in prior waves. Remove low-precision patterns and report results with strict and broad sets.
2. **Per posting:** Compute density (mentions per 1K chars) for each profile.
3. **2D and 3D scatter:** Management vs Orchestration vs Strategic, colored by period. How did the distribution shift?
4. **Senior sub-archetypes:** Cluster senior postings by their language profiles. Are there distinct types (people-manager, tech-lead, architect, strategist)? How did their proportions change?
5. **AI interaction:** Among senior postings mentioning AI, how does the management/orchestration/strategic balance differ from non-AI-mentioning senior postings?
6. **Director specifically:** Directors are a small but important group. What do their postings look like? How do they differ from mid-senior?
7. **The "new senior" question:** Is there an emergent senior archetype that didn't exist in 2024? What does it look like?
8. **Cross-seniority management comparison.** How did management language in SENIOR postings change compared to the entry-level change? If management language expanded at all levels (not just entry), that's evidence against downward migration and for a field-wide template shift.

**Essential sensitivities:** (a) aggregator exclusion

**Output:** `exploration/reports/T21.md` + management-orchestration-strategic charts + senior sub-archetype analysis + cross-seniority management comparison

### T22. Ghost & aspirational requirements forensics `[Agent M]`

**Goal:** Identify ghost-like and aspirational requirement patterns through systematic text analysis.

**Steps (SWE, LinkedIn-only):**
1. **Ghost indicators per posting:**
   - **Kitchen-sink score:** Number of distinct technologies x number of organizational scope terms (high product = everything-and-the-kitchen-sink)
   - **Aspiration ratio:** Count of hedging language ("ideally", "nice to have", "preferred", "bonus", "a plus") / count of firm requirement language ("must have", "required", "minimum", "mandatory"). Higher ratio = more aspirational.
   - **YOE-scope mismatch:** Entry-level postings (`seniority_final = 'entry'`) with `yoe_extracted >= 5` OR with >=3 senior scope terms (architecture, ownership, system design, distributed systems)
   - **Template saturation:** Within each company, compute pairwise cosine similarity of requirement sections across their postings. Flag companies with mean similarity > 0.8 (copy-paste templates).
   - **Credential impossibility:** Postings requiring contradictory credentials (e.g., 10+ YOE for entry-level, or both "no degree required" and "MS required")
2. **Prevalence by period x seniority:** How common is each ghost indicator? Did ghost-like patterns increase or decrease?
3. **AI ghostiness test:** Are AI requirements MORE aspirational than traditional requirements? Compute aspiration ratio separately for AI terms vs non-AI terms within the same postings.
4. **The 20 most ghost-like entry-level postings:** Display their title, company, and requirements section. Are they real roles or artifacts?
5. **Aggregator vs direct:** Compare ghost indicators. Are aggregators more ghost-like?
6. **Industry patterns:** Where company_industry is available, do certain industries have more ghost-like postings?

**Essential sensitivities:** (aggregator comparison IS core to this task — run all ghost indicators separately for aggregator vs direct employer)
**Recommended sensitivities:** (d) description text source

**Output:** `exploration/reports/T22.md` + ghost prevalence tables + examples + `exploration/artifacts/shared/validated_mgmt_patterns.json` (validated patterns with precision scores for downstream use)

### T23. Employer-requirement / worker-usage divergence `[Agent M]`

**Goal:** Compare posting-side AI requirements against worker-side AI usage benchmarks (RQ3).

**Steps:**
1. **AI requirement rate in SWE postings** by period and seniority. Separate:
   - "AI-as-tool" (copilot, cursor, LLM, prompt engineering, AI pair programming)
   - "AI-as-domain" (ML, DL, NLP, computer vision, model training)
   - "AI-general" (artificial intelligence, AI, machine learning — ambiguous)
2. **External benchmarks.** Try to access:
   - Anthropic occupation-level AI usage data (https://www.anthropic.com/research/labor-market-impacts)
   - StackOverflow Developer Survey AI usage rates
   - GitHub Copilot adoption statistics
   - Any other public benchmark data
3. **Divergence computation:** AI requirement rate (our data) vs AI usage rate (benchmarks), by seniority where possible. Note: these are fundamentally different measurements — the contribution is the divergence PATTERN, not the exact gap.
4. **Temporal divergence:** How fast are requirements growing vs how fast usage seems to be growing?
5. **Divergence by specificity:** Are employers asking for specific AI tools (Copilot, Cursor) at rates that exceed likely adoption? Or is the divergence mainly in generic AI mentions?
6. **Benchmark sensitivity.** Compute the divergence under multiple benchmark assumptions (developer AI usage at 50%, 65%, 75%, 85%) to account for survey self-selection bias. Report the range and note which qualitative findings are robust across all assumptions.
7. **Produce divergence chart** showing requirement rate vs usage benchmarks, with appropriate uncertainty bands.

**Essential sensitivities:** (a) aggregator exclusion
**Recommended sensitivities:** (c) seniority operationalization

**Output:** `exploration/reports/T23.md` + divergence chart + benchmark sensitivity table

---

### T28. Domain-stratified scope changes by archetype `[Agent O]`

**Goal:** T09 found that the SWE posting market is organized primarily by tech domain (NMI of cluster × language is 5-7x larger than cluster × seniority or × period), and that AI/ML is the only large-growing archetype while being structurally less junior-heavy than other domains. This suggests that aggregate scope-and-content changes may be partly within-domain and partly between-domain composition shifts. T08 step 7 and T11 step 7 were both spec'd to do this decomposition but were deferred because T09 had not yet produced the archetype labels. This task picks them up and extends them.

**Dependency:** Requires `exploration/artifacts/shared/swe_archetype_labels.parquet` from T09.

**Steps:**

1. **Load T09 archetype labels** and join to the SWE corpus. Report archetype distribution by period.

2. **Domain × seniority decomposition for the entry-share trend.** Using `seniority_final` AND the YOE-based proxy, compute entry share by archetype × period. Decompose the aggregate change into:
   - Within-domain component (entry share change holding archetype composition constant)
   - Between-domain component (change driven by the market shifting between archetypes)
   - Interaction
   - Report under both seniority operationalizations. If they disagree at the archetype level, drill in.

3. **Domain-stratified scope inflation.** Within each archetype, compute the change in `requirement_breadth`, `tech_count`, `scope_count`, AI mention rate, and credential stack depth (as defined in T11) between periods. Is scope inflation a within-domain phenomenon, a between-domain composition effect, or both? Which archetypes are growing the most in scope?

4. **Junior vs senior content within each archetype.** For each archetype, compare entry vs mid-senior postings on requirement breadth, AI mention rate, scope language, and management/mentorship language. Is the junior/senior gap closing within some archetypes and not others? This is the more nuanced version of the convergence question that T15 ran at the corpus level (and rejected). The corpus-level null may hide within-domain convergence in some archetypes.

5. **Senior archetype shift by domain.** T11 found the senior tier is shifting toward IC+mentoring rather than people-management. Does this hold across all domains, or is it concentrated in some (e.g., AI/ML, where the work itself is changing)? Use the strict mentoring detector from T11.

6. **Cross-validate the AI/ML expansion.** T09 reported AI/ML +11pp. Within the AI/ML archetype, profile: who are the top employers, what is the entry vs senior mix, what tech stack dominates, what is the description length and credential stack profile. Is the AI/ML growth coming from new entrants or from existing employers shifting their mix?

**Essential sensitivities:** (a) aggregator exclusion, (b) company capping per the unit-of-analysis rule (capping is appropriate for the within-archetype term/scope analyses, not appropriate for the entry-share decomposition), (c) seniority operationalization (`seniority_final` primary, YOE proxy as co-equal validator)

**Output:** `exploration/reports/T28.md` with the domain × seniority decomposition table, per-archetype scope changes table, per-archetype junior/senior comparison, and the AI/ML deep dive.

---

### T29. LLM-authored description detection `[Agent O]`

**Goal:** Test the hypothesis that part of what we measure as "employer requirements changing" between 2024 and 2026 is actually downstream of recruiters adopting LLMs to draft job descriptions during the same window. This is exploratory and may yield no signal — which is itself informative. If the hypothesis is supported, it would unify several Wave 2 findings (length growth, tech-density decrease, AI mention explosion, credential vocabulary stripping, mid-senior tone shift) into a single mechanism, AND it would be a methodological warning of broad applicability for any longitudinal posting study.

**Steps:**

1. **Define LLM-authorship signals.** Build a per-posting authorship score from observable text features. Candidate features (you should add or remove based on your judgment and any quick research on current LLM stylistic tells):
   - **Signature vocabulary density:** classic LLM tells like `delve`, `tapestry`, `leverage`, `robust`, `unleash`, `embark on`, `navigate`, `cutting-edge`, `in the realm of`, `comprehensive`, `seamless`, `furthermore`, `moreover`, `it's worth noting`, `notably`, `align with`, `at the forefront`, `pivotal`, `harness`, `dynamic`, `vibrant`. Compute density per 1K chars.
   - **Em-dash density:** LLMs use em-dashes (`—` and `--`) noticeably more than human writers. Per 1K chars.
   - **Sentence length distribution:** mean and standard deviation. LLMs produce longer, more uniform sentences than humans.
   - **Vocabulary diversity:** type-token ratio, both within-posting and across-posting. If LLMs are writing the descriptions, postings may become more uniform in vocabulary across the corpus.
   - **Bullet structure:** number and depth of bulleted lists per 1K chars.
   - **Paragraph structure:** average paragraph length and uniformity.

2. **Validate the signals on a known reference if possible.** If you can find a public sample of pre-2023 (clearly human-written) job descriptions and post-2024 (likely-LLM-authored) ones, check whether your signals discriminate. If no reference is available, document the limitation and proceed with the corpus-internal analysis only.

3. **Compute the authorship score for the SWE LinkedIn corpus.** Per posting. Save as `exploration/tables/T29/authorship_scores.csv` with `uid` and the per-feature columns.

4. **Distribution by period.** What is the distribution of the authorship score by period? Does the median shift between 2024 and 2026? Does the variance change? Are 2026 postings more uniform (low cross-posting variance) than 2024 postings?

5. **Distribution by company.** Are some companies using LLM-style writing in 2024 already? Are some companies still writing human-style in 2026? Identify the companies most/least likely to be LLM-authoring.

6. **Correlation with Wave 2 findings.** At the per-posting level, correlate the authorship score with: description length, tech density, AI mention rate, credential stack depth, and the change in any of these metrics by company (if the company appears in both periods).

7. **The unifying-mechanism test.** Subset the corpus to "low-LLM-score" postings only. Re-compute the headline Wave 2 findings (length growth, AI mention growth, credential stack jump, mid-senior credential vocabulary stripping) on the low-LLM-score subset. If the findings substantially weaken when LLM-style postings are excluded, much of the apparent change is mediated by recruiter tooling. If the findings persist, the changes are real labor-market signal independent of authorship style.

8. **Verdict.** Report whether the hypothesis is supported, partially supported, or rejected. Be explicit about the uncertainty — this is a noisy test and individual signal features can be confounded (e.g., AI-domain postings genuinely use AI-related vocabulary; that's not LLM authorship).

**Essential sensitivities:** (a) aggregator exclusion, (d) description text source (text-source composition is itself a major confound for several of these features — control by reading raw `description` only or restricting to LLM-cleaned text only)

**Output:** `exploration/reports/T29.md` with the authorship score distribution, per-company profile, correlation table, the low-LLM-subset re-test of headline findings, and a verdict on the hypothesis.

---

### Wave 4 — Integration & Hypothesis Generation

---

### T24. Hypothesis generation from findings `[Agent N]`

**Goal:** The most important task in the exploration. Read ALL prior reports and generate NEW research hypotheses that go beyond RQ1-RQ4.

**Steps:**
1. Read all `exploration/reports/T*.md`
2. **Confirmation inventory:** Which existing RQ1-RQ4 hypotheses are supported by the data? Which are contradicted? Which are ambiguous?
3. **Surprise inventory:** Across all tasks, what findings were unexpected or contradicted prior assumptions? List each with its evidence strength.
4. **New hypothesis generation:** Based on the full body of findings, propose 5-10 new testable hypotheses that are NOT already in the research design. For each:
   - State the hypothesis precisely
   - What evidence from exploration supports it?
   - What additional analysis would test it?
   - How novel/publishable is it?
5. **Method suitability assessment:** For each RQ (existing and new), what statistical/analytical methods are best suited given our data's specific characteristics (sample sizes, confounds, measurement quality)?
6. **Key tensions:** List the 5 most important tensions or puzzles that the analysis phase needs to resolve.
7. **Data gaps:** What data would we need to answer the most interesting questions we can't currently answer?

**Output:** `exploration/reports/T24.md` — this is the intellectual payoff of the exploration

### T25. Interview elicitation artifacts `[Agent N]`

**Goal:** Produce 5 artifacts for RQ4 data-prompted interviews, drawing on the full body of exploration findings.

**Steps (reads all prior reports):**
1. **Inflated junior JDs:** From T11/T22, select 3-5 entry-level postings with the most extreme scope-inflated or ghost-like requirements. Query parquet for actual text.
2. **Paired JDs over time:** From T16, select 3-5 same-company pairs (2024 vs 2026). Format side-by-side.
3. **Junior-share trend plot:** From T08, annotated with AI model release dates (GPT-4: Mar 2023, Claude 3: Mar 2024, GPT-4o: May 2024, Claude 3.5 Sonnet: Jun 2024, o1: Sep 2024, DeepSeek V3: Dec 2024, Claude 3.5 MAX: Feb 2025, GPT-4.5: Feb 2025, Claude 3.6 Sonnet: Apr 2025, Claude 4 Opus: Sep 2025, Claude 4.5 Haiku: Oct 2025, Gemini 2.5 Pro: Mar 2026).
4. **Senior archetype chart:** From T21, management vs orchestration vs strategic language profiles (2024 vs 2026).
5. **Posting-usage divergence chart:** From T23.
6. **Bonus: Any particularly striking discovery from T24** that would be valuable to present to interviewees for reaction.

**Output:** `exploration/artifacts/` with each artifact as PNG + a README

### T26. Exploration synthesis `[Agent N]`

**Goal:** Consolidate everything into a single handoff for the analysis phase.

**Steps (reads all reports):**
1. Read all `exploration/reports/T*.md`
2. Write `exploration/reports/SYNTHESIS.md` covering:
   - **Data quality verdict per RQ** — what analyses are safe, what need caveats?
   - **Recommended analytical samples** (rows, columns, filters) for each type of analysis
   - **Seniority validation summary** — does `seniority_final` agree with the YOE-based proxy on the directional findings? Where does it diverge, and why?
   - **Known confounders** with severity assessment (description length growth, asaniczka label gap, aggregator contamination, company composition shift, field-wide vs SWE-specific trends)
   - **Discovery findings** organized by:
     - Confirmed hypotheses (with confidence level)
     - Contradicted hypotheses (with evidence)
     - New discoveries (with novelty assessment)
     - Unresolved tensions
   - **Posting archetype summary** from T09 — the natural structure of the market
   - **Technology evolution summary** from T14
   - **Geographic heterogeneity summary** from T17
   - **Senior archetype characterization** from T21
   - **Ghost/aspirational prevalence** from T22
   - **New hypotheses** from T24 — ranked by priority for analysis phase
   - **Method recommendations** for the analysis phase
   - **Sensitivity requirements** — which findings need robustness checks?
   - **Interview priorities** — what should qualitative work focus on?

**Output:** `exploration/reports/SYNTHESIS.md` — the one document the analysis agent reads first.

---

### Wave 5 — Presentation

---

### T27. Exploration findings package `[Agent P]`

**Goal:** Package the exploration's findings into a navigable, presentable artifact that works at multiple levels of depth — from a 10-minute slide presentation down to the raw task reports and gate memos. Host it and share the link.

**Inputs:** Read `exploration/reports/SYNTHESIS.md` (primary — the consolidated findings), gate memos (`exploration/memos/gate_*.md`), and `exploration/reports/INDEX.md` for the full task inventory. Reference existing figures from `exploration/figures/` and reports from `exploration/reports/`. Also read `docs/preprocessing-guide.md` (and skim `docs/preprocessing-schema.md` for the output schema) to extract a concise description of the preprocessing pipeline for the methodology layer — see "Preprocessing description" below. Do not regenerate analysis — package what exists.

---

#### The three depth layers

The deliverable has three layers. A reader should be able to enter at any layer and navigate to the others.

**Layer 1 — The presentation (the story).**
A MARP markdown slide deck (~20-25 slides) that tells the story of what the exploration found. This is the entry point. It should be embeddable in the site as an interactive carousel.

Read `docs/fatahalian-clear-talks.pdf` — Kayvon Fatahalian's "Tips for Giving Clear Talks" (67 slides, 12 tips). This is the definitive guide for how the slides should be designed. The full PDF is the reference; the distillation below highlights what matters most for this presentation:

**Core philosophy:** The goal is to convey what you *learned*, not what you *did*. Tell the audience the most important things they should know but probably don't. Put smart people in the best position to help you.

**Slide design principles:**
- **Every sentence matters.** If it doesn't make the point, remove it. If the audience won't understand it, remove it. If you can't justify how it helps the listener, remove it.
- **One point per slide, and the point IS the title.** Slide titles should be complete-sentence claims, not topic labels. Reading only the titles in sequence should give a great summary of the entire talk. Not "Entry-Level Analysis" but "Entry-level SWE share declined while control occupations increased (DiD = -25pp)."
- **Show, don't tell.** Communicate with figures and images rather than text wherever possible. When you show a figure, explain every visual element — never make the audience decode it themselves. Describe the axes, then describe the one point the figure makes.
- **The audience prefers not to think** (about things you can just tell them). Lead them through the story. Tell them what to look for in a graph. Tell them your assumptions. They want to spend their mental energy on whether your approach is sound and how it connects to their work.
- **No surprises in narrative.** Always say where you're going and why before saying what you did. The audience should be able to anticipate what comes next. If you can't remember what comes next when practicing, the narrative structure is wrong.
- **The intro frames the problem, not the pipeline.** The intro should tell the listener: "here is the way I want you to think about the problem I'm studying." An excellent strategy is to make them aware of something they didn't know they didn't know.
- **Establish goals and constraints early.** What are the inputs, outputs, and constraints of the research?
- **Use section slides** to stage transitions — they help the audience compartmentalize and re-engage if lost.
- **End on a positive note.** Don't list limitations — contribute intellectual thought about what the work means for the broader picture.

Each finding slide should link to a deeper page where the reader can explore the evidence.

**Layer 2 — Findings, methodology, and claims (the evidence).**
Curated pages that let a reader dig into any specific claim. This is where the site goes beyond a slide deck and functions more like a research paper's results and methods sections, but with more detail since the exploration was broad. Include:
- **Thematic findings pages** — one per major finding, synthesizing evidence from multiple task reports. Each should state the claim, present the supporting evidence with specific citations (e.g., "T18, Section 2"), show key figures, summarize sensitivity checks, and link to the raw reports.
- **Methodology and data** — data sources, the preprocessing pipeline (see below), the sensitivity framework, limitations and open questions. A skeptical reader should be able to assess whether to trust the findings.
- **Narrative pages** — what the paper can and cannot claim, how the story evolved across gates, and the full synthesis.

#### Preprocessing description (required, integrated into Layer 2)

The Layer 2 methodology section MUST include a concise description of the preprocessing pipeline. Do not invent or re-derive this content — extract and condense it from `docs/preprocessing-guide.md` (and `docs/preprocessing-schema.md` for the output schema). The goal is for a reader to understand, without leaving the site, how raw scraped/Kaggle postings become the analysis-ready dataset the findings are built on.

Cover, at minimum:
- **Why preprocessing exists.** A short framing paragraph: the raw inputs are heterogeneous (two Kaggle snapshots + a daily scrape), and the pipeline normalizes, deduplicates, classifies, and enriches them into a single comparable corpus.
- **Stage-by-stage summary.** A compact table or list walking through each pipeline stage in order (ingest, dedup, classification, normalization/temporal/flags, LLM extraction, LLM classification, integration, final output). For each stage: one sentence on *what* it does and one sentence on *why* it matters for the analysis. Pull this directly from the "Stage Reference" section of the preprocessing guide — keep it minimal, do not copy the whole guide.
- **Output data structure.** A short description of the unified row schema that downstream analysis consumes — what one row represents, the key column families (identity, text, seniority, classification, temporal, LLM-derived), and the coverage columns (`llm_extraction_coverage`, `llm_classification_coverage`, `selected_for_llm_frame`) that gate which rows are usable for which analyses.
- **LLM stages and prompts.** For the LLM stages (extraction, classification, integration), summarize what each LLM call is asked to do and include the actual prompt text or a faithful condensed version of it, as documented in the preprocessing guide. A reader should be able to see, for each LLM-derived column, the prompt that produced it. If full prompts are long, put a short summary inline and link to or embed the full prompt in a collapsible block.
- **Budgeting and coverage caveats.** A brief note that LLM stages run under an explicit budget, so not every row has LLM-derived columns, and that findings built on LLM columns are reported against the labeled subset (the sticky core frame).

Integrate this naturally into the methodology pages — it should not feel like a bolted-on appendix. A reader arriving from a finding that depends on, e.g., `description_core_llm` or `seniority_final` should be able to click through to the relevant preprocessing stage and see how that column was produced. Cross-link from findings pages to the preprocessing section where appropriate.

Keep the total preprocessing section tight — aim for a single navigable page (or a small cluster of pages) rather than reproducing the full guide. The principle is "minimal description that lets a skeptical reader trust the upstream pipeline," not "complete reference documentation."

**Layer 3 — Raw evidence (the trail).**
All 26 task reports organized by wave, all gate memos, and retrospectives. This is the audit trail — a reader who wants to see exactly what analysis was done can find it here.

---

#### Site design

Use **mkdocs-material** with `navigation.tabs` to separate the depth layers into top-level tabs. The exact tab names and organization are up to you — the key principle is that each tab answers a different reader question (e.g., "give me the overview" vs. "tell me more about claim X" vs. "should I trust this?" vs. "show me the raw work"). Design the hierarchy so a reader isn't overwhelmed by all layers at once.

The slide deck should be embedded on the landing/presentation page as an interactive iframe so stakeholders can navigate slides without opening a separate file.

Copy exploration artifacts (reports, memos, figures) into the site's `docs/` directory. Fix image paths as needed.

---

#### Build ordering and hosting

There is one critical build constraint: **mkdocs build clears the `site/` directory**, so the MARP HTML export must happen AFTER the mkdocs build. The sequence is:
1. `mkdocs build`
2. MARP CLI export into `site/`
3. Start a Python HTTP server on port 8080

Also note the **iframe path gotcha**: mkdocs renders `slides.md` to `site/slides/index.html` (a subdirectory), so the iframe `src` to the MARP HTML at `site/presentation.html` needs to go up one directory (`../presentation.html`).

After hosting, print the Tailscale URL clearly so the user can copy-paste and share.

**Output:** `exploration/site/site/` — self-contained static folder, hosted on port 8080 on the tailnet.

---

## 4. Deferred to analysis plan

These items require formal statistical framework, LLM-stage outputs, or analysis-phase infrastructure:

- Robustness pre-registration / specification curve
- Placebo and falsification tests
- Oaxaca-Blinder decomposition
- Selection bias reweighting / IPSW
- Full power analysis refinement (effect-size-specific, beyond T07 feasibility estimates)
- Seniority boundary classifier (analysis-phase, needs embeddings + LLM labels)
- Company fixed-effects regression (uses T16's overlap panel)
- Formal break detection / event-study plots
- Causal timing analysis with model release windows
- Full embedding-based document classification

---

## 5. Bias threat summary

| Bias | Direction | Mitigation task | Residual risk |
|---|---|---|---|
| Platform selection | Favors SWE | T07 | Low for SWE |
| Scraper query design | Misses long-tail | T05 | Moderate |
| Aggregator contamination | Inflates some companies | T06, T16 | Low after flagging |
| Description length inflation | Biases raw keyword counts | Length-normalization everywhere | Low after normalization |
| Company composition shift | Could drive aggregate seniority shift | T06, T16 | Low after decomposition |
| Temporal selection (volatility) | Oversamples long-lived postings | T19 | Moderate |
| Kaggle provenance unknown | Unknown | T05 | High (irreducible) |
| asaniczka missing entry-level | Thin baseline | T02, T03 | Moderate |
| Boilerplate in raw text | Noisy text analysis | Use `description_core_llm` (LLM-cleaned); raw `description` only for boilerplate-insensitive checks | Low after LLM coverage |
| Company-name contamination | Pollutes corpus comparisons | Company-name stripping in preamble | Low after stripping |
| Remote flag incomparability | 0% in 2024 (data gap, not real) | Do not interpret as real change | Low if noted |
| SWE classification temporal instability | Could change SWE sample composition | T04 | Moderate |
| Within-2024 cross-source variation | Could inflate 2024-to-2026 effect sizes | Within-2024 calibration in T05, T08 | Low after calibration |
| Instrument difference (Kaggle unformatted vs scraped markdown) | Inflates text-based 2024-to-2026 differences | Within-2024 calibration (sensitivity dim f) | Moderate |
| SWE classification tier uncertainty | 9-10% from elevated-FP tier could shift sample composition | Sensitivity dim g in T09, T14, T15, T18 | Low after check |

---

## 6. Discovery principles

This exploration is designed around discovery, not confirmation. Key principles:

1. **Let the data speak first.** Unsupervised methods (clustering, dimensionality reduction, co-occurrence networks) before hypothesis-testing methods.
2. **Report surprises.** Every task report must include unexpected findings, even if they don't map to RQ1-RQ4.
3. **Calibrate everything.** Within-2024 (arshkon vs asaniczka) comparisons establish how much variation is "normal." 2024-to-2026 changes are only interesting if they exceed this baseline.
4. **Normalize for length.** The 56% description length growth is the single biggest confound. Every text metric needs length normalization.
5. **Decompose aggregate changes.** Company composition, seniority composition, geographic composition can all drive aggregate changes. Always check within-stratum patterns.
6. **Map boundaries, not just levels.** The boundaries between seniority levels, between SWE and adjacent roles, between AI-mentioning and non-AI-mentioning postings may be more informative than the levels themselves.
7. **Generate, don't just test.** The exploration succeeds if it produces NEW research questions, not just if it confirms old ones.
8. **Follow the evidence, not the hypothesis.** The initial RQ1-RQ4 framing may turn out to be wrong. If scope inflation is an artifact of aggregator composition, if the senior archetype shift is stronger than the junior story, if the most interesting pattern is something we never anticipated — follow it. The paper's narrative should emerge from the data, not be imposed on it.
