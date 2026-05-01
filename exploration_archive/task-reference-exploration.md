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
- Three sources: kaggle_arshkon, kaggle_asaniczka, and scraped. The scraped source keeps growing. Query the data to get current date ranges and counts — do not rely on documented numbers.
- Asaniczka has ZERO native entry-level seniority labels. `seniority_native` cannot detect entry-level postings in asaniczka by construction — use arshkon-only for any sanity check that depends on `seniority_native`.
- Remote work flags are 0% in 2024 sources (data artifact). Do not interpret as a real change.
- LLM classification columns: `swe_classification_llm`, `ghost_assessment_llm`, `yoe_min_years_llm`. Check `llm_classification_coverage` for coverage. Use `ghost_assessment_llm` as the primary ghost indicator with `ghost_job_risk` as fallback. (LLM seniority writes back to `seniority_final`; there is no separate `seniority_llm` column.)
- **Seniority — use `seniority_final`.** It is the combined column. Stage 5 fills it from high-confidence title keywords (`title_keyword`, `title_manager`); Stage 10 overwrites it with the LLM result for rows the router sent to the LLM. `seniority_final_source` records which path produced each value (`title_keyword`, `title_manager`, `llm`, or `unknown`). `seniority_final` is `'unknown'` only when no signal fired. See `docs/preprocessing-schema.md` Section 4 for the full schema.
- `selected_for_llm_frame` marks the sticky balanced core only. `selection_target` is the minimum core size, not the full usable LLM set.
- Stage 9 and Stage 10 use separate caches, so row coverage can differ between them. A row may have usable Stage 9 text without Stage 10 classification, or vice versa.
- `llm_extraction_sample_tier` and `llm_classification_sample_tier` take `core`, `supplemental_cache`, or `none`. Balanced-sample claims apply only to `selected_for_llm_frame = true`.
- For raw Stage 9/Stage 10 LLM columns (text-based: `description_core_llm`, `swe_classification_llm`, `ghost_assessment_llm`, `yoe_min_years_llm`), filter to `llm_*_coverage == 'labeled'`. **Seniority is the exception:** `seniority_final` is the combined column and should be used directly without coverage filtering.
- 31GB RAM limit — use DuckDB or pyarrow for queries, never load full parquet into pandas.

**Seniority — use the T30 ablation panel, not a single operationalization.**

`seniority_final` is the combined rule+LLM label column. The T30 panel pairs a YOE-based primary (`yoe_min_years_llm`, restricted to the LLM frame) with label-based sensitivities, because label variants carry known instrument artifacts (asaniczka has no native entry labels, LLM-frame selection skews junior share, platform taxonomy drifts across years) while YOE extracted from description text is instrument-invariant. Every seniority-stratified finding must be reported under the canonical panel built by T30 and saved at `exploration/artifacts/shared/seniority_definition_panel.csv`:

- **Junior side:** J1 = `seniority_final = 'entry'` (label sensitivity), J2 = `seniority_final IN ('entry','associate')` (label sensitivity), J3 = `yoe_min_years_llm <= 2` (**primary**), J4 = `yoe_min_years_llm <= 3`.
- **Senior side:** S1 = `seniority_final IN ('mid-senior','director')` (label sensitivity), S2 = `seniority_final = 'director'` (label sensitivity), S3 = title-keyword senior (`\b(senior|sr\.?|staff|principal|lead|architect|distinguished)\b`), S4 = `yoe_min_years_llm >= 5` (**primary**).
- T30 additionally reports J5 title-keyword junior, J6 = J1 ∪ J5, S5 = `yoe_min_years_llm >= 8`, and `yoe_extracted`-based J3_rule / S4_rule as an extractor-ablation companion.

**Reporting rule.** For a junior claim, report J1, J2, J3, J4 as a 4-row table. For a senior claim, report S1–S4. Primary is J3 / S4 (YOE-based); J1/J2/S1/S2 are sensitivities testing robustness against label artifacts. Directional agreement across all four = robust; YOE-vs-label disagreement is itself a finding — label variants carry the asaniczka-entry-gap and LLM-frame-selection artifacts noted above. Load `exploration/artifacts/shared/seniority_definition_panel.csv` (produced by T30) rather than recomputing.

**Practical caveats.** (i) `seniority_native = 'entry'` is arshkon-only as a diagnostic; asaniczka has no native entry labels. (ii) YOE-based variants filter to `llm_classification_coverage = 'labeled' AND yoe_min_years_llm IS NOT NULL` (posting stated a YOE floor); report the labeled-and-YOE-stated share alongside. Outside the LLM frame (Indeed, unlabeled rows), fall back to rule-based `yoe_extracted`. (iii) `yoe_min_years_llm = 0` is included in the `<= 2` bucket by default — T30 runs a 0-value audit to confirm that most are valid entry signals (literal "0 years", entry-framing, internships) rather than LLM errors, and the 4-row panel's agreement check absorbs residual error.

**Description text quality — critical for text-based analyses:**
`description_core_llm` (LLM-based boilerplate removal) is the **only** cleaned-text column. The former rule-based `description_core` was retired on 2026-04-10 because ~44% accuracy was misleading downstream analysis. Check `llm_extraction_coverage` to confirm coverage by source; filter to `labeled` whenever you use `description_core_llm`.

Text column rules:
- **Text-dependent analysis** (embeddings, topic models, requirement extraction, corpus comparison, density metrics): Use `description_core_llm` with `llm_extraction_coverage = 'labeled'`. Do not backfill missing rows with raw `description` for boilerplate-sensitive work — restrict the sample instead and report coverage.
- **Binary keyword presence** (does the posting mention X anywhere?): Raw `description` is acceptable for recall when the presence signal is insensitive to boilerplate phrasing. Density metrics (mentions per 1K chars) must still use `description_core_llm`.
- **Non-text analyses** (seniority counts, company analysis, geographic patterns): Use all rows regardless of text column.
- **Sensitivity check:** When a finding depends on text, the meaningful sensitivity is `description_core_llm` vs raw `description` (no rule-based alternative exists). If the direction flips under raw text, the finding is boilerplate-driven and must be flagged.
- For Stage 10 LLM-only diagnostics on `swe_classification_llm` or `ghost_assessment_llm` in isolation, filter to `labeled` (the only coverage value where those columns are populated). For rule-vs-LLM seniority audits, compare `seniority_rule` (Stage 5 rule-based snapshot, always populated) against `seniority_final` (LLM-overridden where available).

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

**Prevalence citation transparency.** Every prevalence, density, share, or effect-size number must be cited with (a) the exact pattern or column definition, (b) the subset filter (sample, cap, aggregator exclusion, LLM-coverage restriction), and (c) the denominator ("of all" vs "of known-seniority" vs "of LLM-labeled"). Cross-task citations that combine numbers from different patterns or subsets into one cell are prohibited — if task X cites task Y, use Y's exact pattern and subset or explicitly re-derive. This prevents the "broad-union rate cited with narrow-pattern SNR" class of error.

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

5. **Within-historical calibration.** When comparing the historical baseline vs the current scraped window, also compare arshkon vs asaniczka on the same metric where possible. This establishes baseline cross-source variability. If the historical-to-current change is smaller than within-historical cross-source variation, flag it as potentially artifactual.

6. **Semantic precision before density reporting.** For any keyword pattern used in a prevalence, density, or rate metric, sample 50 matches stratified by period (25/25) **before** reporting the metric. Judge each match *semantically* — read the surrounding sentence and decide whether it represents the intended concept. `\bleading\b` matching "a leading company" does not count toward "leading a team." Report precision per sub-pattern (not per compound) and drop any sub-pattern below 80%. Rebuild the compound and re-run. **A precision check that only verifies "the regex matches its own regex" is tautological and not valid** — if an upstream task's precision check was tautological, V1 must rebuild the pattern before findings are cited. A finding may not cite ≥80% precision unless the precision was measured semantically on a stratified sample.

7. **Sampling protocol.** For any analysis based on a sample rather than the full dataset:
   - Document sample size, stratification method, and what was excluded
   - Report what fraction of each source/period/seniority group is represented
   - Prefer balanced period representation over proportional-to-population (avoids asaniczka domination)
   - For keyword pattern validation, stratify samples by period — pattern behavior may differ between 2024 and 2026
   - If a finding could change with a different sample, test with at least one alternative sample

8. **Source composition profiling.** Before using a single source as a baseline for cross-period comparisons (e.g., arshkon as the 2024 baseline), profile its top-20 employers and any obvious composition skews (industry where available, company size, geographic concentration). Differential composition across sources can produce false "temporal" signals — what looks like change between periods may just be different employer mixes. The Wave 1 company-concentration task surfaces this for the standard sources; check its findings before drawing cross-period conclusions.

## Sensitivity framework — apply to all analytical tasks

Every analytical task must report results under its primary specification AND under essential sensitivity checks. A finding is only robust if it survives its essential sensitivities.

Nine dimensions (referenced by letter in each task spec):

(a) **Aggregator exclusion.** Primary: include all rows. Alt: exclude `is_aggregator = true`. Rationale: aggregators have systematically different descriptions, seniority patterns, and template-driven requirements. **Exception:** T22 (ghost forensics) makes aggregator-vs-direct comparison its primary axis — downstream tasks citing T22 should load T22's stratified results and not re-pool.

(b) **Company capping.** A few prolific employers dominate any analysis that aggregates over a corpus (term frequencies, topic models, co-occurrence networks, embedding centroids). For these, cap postings per `company_name_canonical` as the primary specification. **Cap-size guidance:** 20 as the default; 50 when the analysis requires per-firm data density (co-occurrence phi stability, network community detection — see T35); 10 when the unit of analysis is per-company × per-title × per-period (cell-stability, see T31); not applicable at all for per-row metrics (rates, distributions) or company-level analyses. Document the cap chosen in the task's methods section.

(c) **Seniority operationalization — T30 panel required.** Every seniority-stratified result must be reported under all four applicable panel variants (J1–J4 for a junior claim; S1–S4 for a senior claim). Primary is J3 / S4 (YOE-based from `yoe_min_years_llm`); J1/J2 and S1/S2 are label-based sensitivities. Panel defined in Section 1a; load the canonical CSV per §1a. Robust iff 3 of 4 agree on direction AND the effect-size spread across the four is within 30%. Directional disagreement — especially YOE-vs-label (the label variants carry asaniczka and LLM-frame-selection artifacts) — is itself a finding; investigate and report the mechanism.

(d) **Description text source.** Primary: `description_core_llm` (filtered to `llm_extraction_coverage = 'labeled'`). Alt: raw `description`. No rule-based cleaned-text alternative exists — the former `description_core` was retired on 2026-04-10. Rationale: findings that only appear under raw `description` are boilerplate-driven and should be flagged as such.

(e) **Source restriction.** Primary: arshkon vs scraped, with the current scraped window explicitly queried and reported. Alt: arshkon + asaniczka pooled as the historical baseline. Rationale: asaniczka is a different instrument; pooling increases power but introduces noise.

(f) **Within-2024 calibration (signal-to-noise).** Mandatory diagnostic. For every metric compared 2024-to-2026, also compute the arshkon-vs-asaniczka difference on the same metric. Signal-to-noise ratio: (cross-period effect size) / (within-2024 effect size). If ratio < 2, flag as "not clearly above instrument noise."

(g) **SWE classification tier.** Primary: all `is_swe = true`. Alt: exclude `swe_classification_tier = 'title_lookup_llm'` (retaining regex + embedding_high only). Rationale: title_lookup_llm has elevated false-positive rate.

(h) **LLM text coverage.** For text analyses: primary restricts to rows with `llm_extraction_coverage = 'labeled'` and uses `description_core_llm`. Report labeled-row coverage by period and source; thin cells must be flagged. This dimension is about sample restriction, not an alternative text column.

(i) **Indeed cross-platform validation.** For key findings (entry share, AI prevalence, description length), compute the same metric on Indeed scraped data. Indeed has no native seniority and is excluded from the Stage 9 LLM frame, so its `seniority_final` only carries Stage 5 strong-rule labels — the unknown rate will be high. Where Indeed coverage is too thin to support a seniority-stratified comparison, fall back to non-seniority-stratified metrics. If Indeed patterns match LinkedIn, findings are more robust. If they diverge, the finding may be LinkedIn-specific.

**Materiality threshold:** A finding is **materially sensitive** to a dimension if the alternative specification changes the main effect size by >30% or flips the direction.

**Sensitivity disagreement is itself a finding to investigate, not just to flag.** When a finding is materially sensitive, drill in: identify the rows/companies/terms driving the difference, characterize what they have in common, state the most likely mechanism, and recommend a follow-up that would resolve the question. A bare "this finding is materially sensitive to dimension X" without drilling in is insufficient — the mechanism behind the disagreement is often more interesting than either of the two estimates.

**Text source discipline (CRITICAL):** Use `description_core_llm` as the primary text column for all text-dependent analyses. The only alternative is raw `description`, and only for analyses that are demonstrably insensitive to boilerplate phrasing (binary keyword presence, rough length checks). NEVER mix rows that used cleaned text with rows that fell back to raw text without explicitly reporting the split and testing whether findings differ between the two subsets. The former rule-based `description_core` column is no longer available; any legacy shared artifact that still mixes rule-based text with LLM text is stale and should be rebuilt from the current pipeline before use.

**Composite-score correlation check (required for matched deltas).** When a composite score (authorship-drift, ghost-index, length-composite, etc.) is used to match or control for a confound, report the pairwise correlation between each score component and the outcome metric *before* interpreting any matched delta. If any component correlates r > 0.3 with the outcome, the matching is confounded on that dimension — either drop the component or report the matched result under ablated score versions. Matching on a length-correlated score to assess length change is tautological and not an attribution. "X attenuates under matching" is not a defensible control claim without this check.

**Length residualization (canonical formula).** When a composite score has components correlated with description length (e.g., `requirement_breadth` aggregates counts of distinct terms, which scales with length), report a length-residualized version as primary. The residualization: fit `composite_score ~ b0 + b1 * log(description_cleaned_length)` via OLS on the full corpus (or the relevant subset); then report `composite_resid = composite_score - (b0 + b1 * log(length))` as the primary metric, with the raw score as a sensitivity. For breadth specifically, the method is: compute raw `requirement_breadth`, then regress on `log(desc_cleaned_length)` and retain residuals. Downstream tasks use `_resid` metrics by default and cite this section for the formula.

**Instrument comparison is a first-class concern.** The 2024 Kaggle sources and the 2026 scraped source are different instruments: Kaggle text is unformatted (HTML-stripped), scraped text preserves markdown formatting (including **bold** headers and bullet points). Section classifiers must handle both formats. The within-2024 calibration (dimension f) is the primary mitigation.
```

---

## 2. Agent dispatch blocks

These are prepended (after the preamble) to each agent's prompt, before the task specs. Dispatch order follows pipeline chronology: Wave 1 → Wave 1.5 → Wave 2 → V1 → Wave 3 → Wave 3.5 → V2 → Wave 4 → Wave 5. Each agent's block is written for the prompt that will dispatch them, so forward-references to later waves (e.g., "your output feeds Wave 3.5 T31") help the agent understand what artifacts must be persisted cleanly.

### Agent A — Wave 1: Data profile & seniority comparability (T01 + T02)

Profile the dataset (actual row counts, column coverage, semantic differences across sources) and produce the coverage heatmap that downstream tasks depend on. Then run the seniority comparability audit: can asaniczka `associate` serve as a junior proxy? Execute tasks T01 and T02.

### Agent B — Wave 1: Seniority audit + panel + SWE classification (T03 + T30 + T04)

Audit `seniority_final` label quality (T03), build the canonical seniority definition ablation panel that every downstream task consumes (T30), and assess SWE classification accuracy (T04). Execute in that order: T03 establishes whether the production seniority column is trustworthy; T30 builds the multi-operationalization panel and recommends which slice to use primary; T04 audits the SWE sample independently. T30's panel is consumed by every Wave 2, Wave 3, and Wave 3.5 seniority-stratified task — save it at the exact path `exploration/artifacts/shared/seniority_definition_panel.csv` with the schema documented in the T30 spec.

### Agent C — Wave 1: Dataset comparability & concentration (T05 + T06)

Test whether the three datasets are measuring the same thing by running pairwise comparisons (description length, company overlap, geographic/seniority/title distributions). Also assess company concentration and whether a few employers dominate findings. T06's entry-specialist list (`entry_specialist_employers.csv`) and the returning-companies cohort are consumed by Wave 3 T16 and by Wave 3.5 T37. Execute tasks T05 and T06.

### Agent D — Wave 1: External benchmarks & power analysis (T07)

Compare our data against BLS OES occupation/state data and JOLTS information sector trends. Additionally, conduct a power and feasibility analysis for all planned cross-period comparisons — compute minimum detectable effect sizes and identify which analyses are well-powered vs underpowered. This requires web access to download benchmark data from FRED and BLS. Execute task T07.

### Agent Prep — Wave 1.5: Shared preprocessing

Build shared analytical artifacts that Wave 2+ agents will load instead of recomputing independently. Compute cleaned text, sentence-transformer embeddings, technology mention matrix, company name stoplist, and the within-2024-vs-cross-period calibration table. Save all to `exploration/artifacts/shared/`. These artifacts are consumed throughout Waves 2, 3, and 3.5 — stable paths and schemas matter. Execute the shared preprocessing spec.

### Agent E — Wave 2: Distribution profiling (T08)

Compute baseline distributions for ALL available variables by period and seniority, with emphasis on anomaly detection and unexpected patterns. This task carries a heavy sensitivity framework — run essential sensitivity checks on all core findings. Read `exploration/reports/INDEX.md` for Wave 1 guidance. Load shared artifacts from `exploration/artifacts/shared/`. Execute task T08.

### Agent F — Wave 2: Posting archetype discovery (T09)

Discover natural posting archetypes through unsupervised methods. This is the primary methods comparison task — run BERTopic (primary) and NMF (comparison) on the same data and compare what each surfaces. Load shared embeddings and cleaned text from `exploration/artifacts/shared/`. Save archetype labels at `exploration/artifacts/shared/swe_archetype_labels.parquet` — consumed by Wave 2 T15/T17/T20, Wave 3 T16/T21/T28, and Wave 3.5 T31/T34/T35. Execute task T09.

**Timing within Wave 2:** Wave 2 tasks that depend on the archetype labels (T15 archetype-stratified convergence, T17 geographic × archetype, T20 domain-stratified boundary analysis) should execute their non-archetype steps first and defer archetype-stratified sub-steps until T09's artifact is published. Wave 3 and Wave 3.5 always execute after Wave 2 completes and can rely on the artifact being present.

### Agent G — Wave 2: Title evolution & requirements complexity (T10 + T11)

Map how the SWE title taxonomy has evolved between 2024 and 2026 — what titles emerged, disappeared, or changed meaning. Then quantify the structural complexity of job requirements (credential stacking, technology density, scope breadth). Load shared technology matrix from `exploration/artifacts/shared/` for T11 tech counting. T10's disappearing-title list and T11's per-posting feature parquet (`T11_posting_features.parquet`) are direct Wave 3.5 inputs (T36 and T33 respectively). Execute tasks T10 and T11.

### Agent H — Wave 2: Linguistic evolution & text discovery (T13 then T12)

**Run T13 first** (section anatomy, readability, tone), **then T12** (corpus comparison using T13's section classifier to strip boilerplate). T12 depends on T13's section classifier output to isolate genuine content changes from boilerplate expansion. Save T13's section classifier as a reusable module at `exploration/scripts/T13_section_classifier.py` — Wave 3 T18 and Wave 3.5 T33 both import it. Load shared artifacts from `exploration/artifacts/shared/`. Execute tasks T13 and T12 in that order.

### Agent I — Wave 2: Technology ecosystems & semantic landscape (T14 + T15)

Map the technology ecosystem — not just individual mentions but co-occurrence networks and natural skill bundles. Also validate using asaniczka's structured skills field. Then compute the full semantic similarity landscape across all period x seniority groups. Load shared technology matrix and embeddings from `exploration/artifacts/shared/`. T14's co-occurrence analysis is extended by Wave 3.5 T35 into a period-split crystallization test — structure T14 so its phi matrices can be re-computed per period cleanly. Execute tasks T14 and T15.

### Agent V1 — Gate 2 Verification

Adversarial quality assurance after Wave 2. Re-derive the top 3-5 headline numbers from Wave 2 from scratch (independent code). Validate keyword patterns by sampling 50 matches stratified by period. Propose alternative explanations for each headline finding. Flag specification-dependent findings. The refined keyword patterns V1 produces (AI-mention strict/broad, management strict, etc.) are consumed by Wave 3 AND Wave 3.5 — V1's precision findings are pipeline-global, not Wave-2-local, so write them precisely. Write `exploration/reports/V1_verification.md`.

### Agent J — Wave 3: Company strategies & geographic structure (T16 + T17)

Among companies appearing in both periods, cluster them by HOW their postings changed. Then analyze geographic market segmentation. T16's arshkon∩scraped overlap panel (≥3 SWE postings in both periods) is consumed by Wave 3.5 T31 (same-company × same-title drift), T37 (sampling-frame robustness), and T38 (hiring-selectivity correlation). Persist the overlap-panel company list and the per-company change vectors to `exploration/tables/T16/` so downstream tasks can reload without re-deriving. Execute tasks T16 and T17.

### Agent K — Wave 3: Cross-occupation boundaries & temporal patterns (T18 + T19)

Compare SWE, SWE-adjacent, and control occupations to determine which changes are SWE-specific vs field-wide. Then estimate rates of change and characterize the temporal structure of our data. T18's DiD findings anchor the paper's lead SWE-specificity claim; Wave 3.5 T32 extends T18's framework to a cross-occupation benchmark-informed divergence test. Execute tasks T18 and T19.

### Agent L — Wave 3: Seniority boundaries & senior role evolution (T20 + T21)

Measure how sharp the seniority boundaries are and whether they blurred or shifted between periods. Then conduct a deep dive into how senior SWE roles evolved. For T21's management language analysis, validate your keyword patterns by sampling matches — broad management patterns are a known risk for inflating indicators. If T09 archetype labels exist in shared artifacts, use them for domain stratification. Save T21's k-means cluster assignments to `exploration/tables/T21/cluster_assignments.csv` — Wave 3.5 T34 profiles any emergent senior-role candidate from those cluster memberships. Execute tasks T20 and T21.

### Agent M — Wave 3: Ghost forensics & employer-usage divergence (T22 + T23)

Identify ghost-like and aspirational requirement patterns, with emphasis on whether AI requirements are more aspirational than traditional ones. Save validated management/scope/AI patterns at `exploration/artifacts/shared/validated_mgmt_patterns.json` with measured precision per sub-pattern — Wave 3.5 agents MUST consume these V1-validated + T22-validated patterns rather than re-deriving their own. Then compute the employer-requirement vs worker-usage divergence. T23's SWE-only divergence is generalized by Wave 3.5 T32 to SWE-adjacent and control occupations. Execute tasks T22 and T23.

### Agent O — Wave 3: Domain-stratified scope changes & LLM authorship detection (T28 + T29)

Two complementary analyses that depend on T09's archetype labels and the cleaned text artifact. T28 (priority): re-decompose scope and content changes by domain archetype now that T09's clusters are available — does scope inflation differ across Frontend, Embedded, Data, ML/AI? T29 (lower priority, exploratory): test the hypothesis that part of the apparent content change is downstream of recruiters using LLMs to draft job descriptions. Execute tasks T28 and T29.

### Agent Q — Wave 3.5: Same-company longitudinal drift + cross-occupation inversion (T31 + T32)

T31 tightens the within-company rewriting test to the finest possible unit — same company × same title × different period — by building on T16's arshkon∩scraped overlap panel. T32 extends T23's SWE employer-vs-worker AI-adoption comparison to SWE-adjacent and control occupations via T18's DiD framework with benchmark-informed divergence tests (testing whether the SWE-side pattern, whatever T23 finds, is universal or SWE-specific). Both tasks require Wave 3 outputs; dispatch only after Agent J (T16), Agent K (T18 DiD framework), and Agent M (T22, T23) have persisted their shared artifacts. Load `seniority_definition_panel.csv`, `entry_specialist_employers.csv`, `validated_mgmt_patterns.json`, archetype labels, and T16's overlap panel from `exploration/artifacts/shared/` and `exploration/tables/T16/`.

### Agent R — Wave 3.5: Hiring-bar signal + emergent senior-role profiling (T33 + T34)

T33 tests whether the 2024→2026 change in requirements-section size (as measured by T13 and T18, whatever direction they find) correlates with implicit hiring-bar change (YOE, credential, education asks). T34 profiles any emergent senior-role candidate that T21 identifies, with content-driven naming. Both tasks connect substantive Wave 2-3 findings to paper-relevant interpretations that SYNTHESIS.md needs. Load T13's section classifier (`exploration/scripts/T13_section_classifier.py`), T21's cluster assignments (`exploration/tables/T21/cluster_assignments.csv`), T11's per-posting feature parquet, and T09's archetype labels.

### Agent S — Wave 3.5: Technology ecosystem crystallization + legacy substitution (T35 + T36)

T35 extends T14's co-occurrence analysis to detect technology ecosystems that crystallized between 2024 and 2026 (applying Louvain community detection at each period separately and comparing modularity). T36 builds the legacy-stack substitution map for the disappearing 2024 titles surfaced in T10. Both are descriptive extensions that strengthen the paper's technology-evolution section and provide concrete before/after examples the presentation (Agent P) can cite. Load `swe_tech_matrix.parquet`, `swe_cleaned_text.parquet`, and T10's disappearing-title list.

### Agent T — Wave 3.5: Sampling-frame and hiring-selectivity robustness (T37 + T38)

Two paper-defensibility tasks. T37 re-runs the top Gate-3 headlines (selected from the orchestrator's post-Wave-3 ranking) on the returning-companies-only subset (firms with presence in both the 2024 sources and scraped 2026) to quantify the sampling-frame artifact per T06's new-entrant profile — results feed directly into SYNTHESIS.md's robustness appendix. T38 tests whether any scope-change direction observed at the company level correlates with posting-volume contraction (a test of the JOLTS 2026 hiring-low interaction). Requires the arshkon∩scraped overlap panel (from T16) and the returning-companies cohort (from T06).

### Agent V2 — Gate 3 Verification

Adversarial quality assurance after Wave 3 AND Wave 3.5. Re-derive the top 3-5 headline numbers from Wave 3 independently (loaded from the orchestrator's draft Gate 3 ranking, whatever they are). Also re-derive a headline from each Wave 3.5 task that produced one, classifying by verification type (pair-level drift, cross-group pattern extension, mechanism regression, sampling-frame retention, emergent-role profiling, descriptive network, hiring-selectivity correlation). Validate new or rebuilt keyword patterns semantically on 50-row stratified samples; flag any tautological precision claim and re-run it. Audit prevalence citation transparency across both Wave 3 and Wave 3.5 reports — flag any cross-task citation that combines different patterns or subsets. Audit composite-score matching for any matched-delta finding. Test whether cross-occupation DiD findings are robust to alternative control group definitions and whether decomposition results hold across T30 panel variants. Write `exploration/reports/V2_verification.md`.

### Agent N — Wave 4: Hypothesis consolidation, artifacts & synthesis (T24 + T25 + T26)

Read ALL reports from Waves 1 through 3.5, plus both verification reports (V1, V2) and all gate memos (0-3). T24 consolidates Wave 3.5 test verdicts (directly tested hypotheses H_A/H_B/H_C/H_H/H_M and new-in-Wave-3.5 H_K/H_L/H_N) alongside any additional hypotheses emerging from the full body of evidence and the deferred inventory (H_D/H_E/H_F/H_G/H_I/H_J). T25 produces interview elicitation artifacts drawing on the strongest Wave 2-3.5 cases (pair-level exemplars from T31, emergent-role exemplars from T34 if T34's precondition was met, ghost-pattern examples from T22). T26 writes SYNTHESIS.md with Wave 3.5 findings integrated into the ranked findings list and robustness appendix — NOT as a separate wave-6 section. Execute tasks T24, T25, and T26.

### Agent P — Wave 5: Presentation (T27)

Read `exploration/reports/SYNTHESIS.md`, gate memos, and INDEX.md. Also read `docs/preprocessing-guide.md` and extract a minimal description of the preprocessing pipeline (stages, rationale, data structure, LLM prompts) to integrate into the site's methodology layer. Produce a ~20-25 slide MARP presentation for the research advisor and stakeholders. Follow the presentation principles in the orchestrator prompt (complete-sentence slide titles, one idea per slide, tell what you learned not what you did, frame corrections as rigor). Reference existing figures from `exploration/figures/`, including Wave 3.5 figures where they support lead claims. Export to HTML and PDF via `npx @marp-team/marp-cli`. Execute task T27.

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

**Essential sensitivities:** (a) aggregator exclusion — report coverage + key counts separately for `is_aggregator = true` vs `false`, since aggregator composition varies across sources and affects downstream interpretation.

**Output:** `exploration/reports/T01.md` + coverage heatmap PNG + CSV

### T02. Asaniczka `associate` as a junior proxy `[Agent A]`

**Goal:** Answer the narrow substitution question: given that asaniczka has zero native entry-level labels, can its `associate`-labeled rows serve as a junior proxy in analyses that need a 2024 baseline? The broader operationalization panel is built by T30.

**Steps:**
1. Compare asaniczka `associate` (`seniority_native`) against arshkon `entry`, `associate`, and `mid-senior` on: top-title Jaccard, explicit junior/senior title-cue rates, `yoe_min_years_llm` distribution (rule `yoe_extracted` as ablation), and `seniority_final` distribution conditional on native label.
2. State whether asaniczka `associate` behaves directionally like arshkon `entry` on at least three of those signals. This is the decision rule.
3. Entry-level effective sample sizes per source under `seniority_final`, broken down by `seniority_final_source` (so the reader sees which fraction came from a strong rule vs the LLM).

**Output:** `exploration/reports/T02.md` with the comparability table and a plain verdict. If the verdict is "not usable," downstream tasks must use T30's J2 (combined `entry`+`associate` under `seniority_final`) or one of the YOE variants instead of pooling asaniczka `associate` rows.

### T03. Seniority label audit `[Agent B]`

**Goal:** Audit `seniority_final` against the available diagnostics. This is the one task in the exploration that interrogates the seniority labels themselves; everything downstream uses `seniority_final` directly without re-running the comparison.

**Steps:**
1. **`seniority_final_source` profile.** For SWE rows, report the distribution of `seniority_final_source` (`title_keyword`, `title_manager`, `llm`, `unknown`) by source and by period. This shows how the rule and LLM halves of `seniority_final` are composed.
2. **Rule-vs-LLM internal agreement (where both could fire).** Restrict to rows where `seniority_final_source = 'llm'` and inspect whether the LLM's answer is consistent with what a strong title rule WOULD have produced if one existed (e.g., spot-check by sampling 100 LLM-labeled rows whose titles contain weak seniority markers like "I/II/III"). Estimate routing-error rate qualitatively.
3. **`seniority_final` vs `seniority_native` (arshkon SWE only).** Cross-tabulate. Compute Cohen's kappa and per-class accuracy using native as the comparison reference. Repeat on scraped LinkedIn SWE. If accuracy differs between arshkon and scraped, that suggests temporal instability of the native classifier (the same LinkedIn label may mean different things across the 2024→2026 window).
4. **Defensibility recommendation.** State plainly whether `seniority_final` looks defensible as the production seniority column, citing the kappa + routing-error + native-label-YOE evidence. T03 answers "is the column trustworthy?"; T30 answers "which slice of it should we use?" — do not re-derive the multi-operationalization junior-share comparison here. If `seniority_final` is not defensible (e.g., the rule and LLM halves disagree systematically), document the failure mode and propose a remediation; otherwise T30 proceeds on top of it.

**Essential sensitivities:** (a) aggregator exclusion — label quality may differ for aggregator vs direct-employer postings; report the kappa and per-class accuracy for each stratum.

**Output:** `exploration/reports/T03.md` with the source profile, the kappa table, the native-label YOE diagnostic, and the defensibility recommendation. This is the canonical seniority-quality reference for downstream agents.

### T30. Seniority definition ablation panel `[Agent B]`

**Goal:** Build the canonical junior- and senior-side operationalization panel that every downstream seniority-stratified task consumes. The panel pairs a YOE-based primary (from `yoe_min_years_llm` within the LLM frame) with label-based sensitivities, and tests whether the 2024→2026 direction is stable across operationalizations.

**Relationship to T03:** T03 audits whether the seniority label column itself is trustworthy (kappa, routing, native-vs-final comparison). T30 builds the operational panel downstream tasks will use, informed by T03's quality verdict. T03 evaluates; T30 prescribes.

**Definitions:**

Junior side:
- **J1** `seniority_final = 'entry'` (label sensitivity)
- **J2** `seniority_final IN ('entry','associate')` (label sensitivity, larger n)
- **J3** `yoe_min_years_llm <= 2` (**YOE-based primary**; filter to `llm_classification_coverage = 'labeled'`)
- **J4** `yoe_min_years_llm <= 3` (YOE-based primary, generous band)
- **J5** title-keyword junior — raw `title` matches `\b(junior|jr|entry[- ]level|graduate|new[- ]grad|intern)\b`
- **J6** J1 ∪ J5 (any junior label signal)
- **J3_rule** `yoe_extracted <= 2` (rule-extractor ablation)

Senior side:
- **S1** `seniority_final IN ('mid-senior','director')` (label sensitivity)
- **S2** `seniority_final = 'director'` (label sensitivity, top of ladder)
- **S3** title-keyword senior — raw `title` matches `\b(senior|sr\.?|staff|principal|lead|architect|distinguished)\b`
- **S4** `yoe_min_years_llm >= 5` (**YOE-based primary**)
- **S5** `yoe_min_years_llm >= 8` (stricter YOE-based primary)
- **S4_rule** `yoe_extracted >= 5` (rule-extractor ablation)

Use raw `title` for J5/S3 — `title_normalized` strips level indicators. Build every regex with inline `assert` edge-case tests before applying.

**Steps:**
1. For each definition, compute `n` in arshkon, asaniczka, pooled 2024, and scraped 2026. Report counts against the appropriate denominator: "of all SWE rows", "of known-seniority rows" (label variants), or "of LLM-YOE-stated rows" (YOE variants).
2. For each definition, compute share 2024 → 2026 under each denominator, within-2024 (arshkon vs asaniczka) calibration, and cross-period effect size.
3. Pairwise row-overlap matrices per side. Entries = |X ∩ Y| / |X|. Shows which definitions largely agree and which are disjoint.
4. Per-definition MDE (binary, 80% power, α = 0.05) for arshkon-only-vs-scraped and pooled-2024-vs-scraped.
5. Direction consistency across J1/J2/J3/J4 and S1/S2/S3/S4; flag disagreement — especially YOE-vs-label — and propose the most likely mechanism. Where J5/S3 title-keyword matches drive the disagreement, sample 20 rows to check regex precision.
6. **Verification audits.**
   - **LLM-vs-rule YOE agreement per source.** On rows where both `yoe_min_years_llm` and `yoe_extracted` are populated: exact-agreement rate, mean absolute difference. Flag any source below 80% exact agreement. Sample 20 rows where they disagree by ≥ 3 years and classify which extractor was correct.
   - **`yoe_min_years_llm = 0` audit.** Stratified sample of 20-30 rows across sources and analysis groups. Classify into: (a) literal "0 years" on a qualification path, (b) entry/new-grad framing, (c) internship/residency, (d) LLM extraction error, (e) other. Report counts. The preamble includes `0` in the `<= 2` bucket by default — confirm or override based on the audit.
   - **Asaniczka senior-asymmetry test.** Compute S4 and S1 senior shares on asaniczka SWE (LLM-labeled, 2024) vs arshkon SWE (LLM-labeled, 2024). If S4 shares match within the within-2024 calibration noise floor, asaniczka can be pooled into the 2024 senior baseline for all downstream senior-side analyses. If the asymmetry persists under the LLM-YOE variant, treat arshkon-only as a senior-side sensitivity (not a mandated primary). State the verdict plainly — downstream tasks read this decision.
7. **Primary recommendation.** Default: J3 for junior claims and S4 for senior claims, both YOE-based from `yoe_min_years_llm`. Override to J4 (generous) only if step 4's MDE shows J3 underpowered for a specific comparison; override to a label-based variant only with a 2-sentence MDE-grounded justification.
8. Save the panel as `exploration/artifacts/shared/seniority_definition_panel.csv`. One row per (definition × period × source). Columns: `definition | side | family | period | source | n_of_all | n_of_denominator | share_of_all | share_of_denominator | mde_arshkon_vs_scraped | mde_pooled_vs_scraped | within_2024_effect | cross_period_effect | direction`. `definition` ∈ `{J1, J2, J3, J4, J5, J6, J3_rule, S1, S2, S3, S4, S5, S4_rule}`. `side` ∈ `{junior, senior}`. `family` ∈ `{yoe_llm, yoe_rule, label, title_keyword}`. `direction` ∈ `{up, down, flat}` for cross-period; null for within-source rows. Integer columns are integers; share / effect / mde columns are floats in [0, 1]. Every Wave 2+ agent loads this artifact rather than recomputing.

**Essential sensitivities:** (a) aggregator exclusion; (f) within-2024 calibration per definition. Dimension (c) is not applicable — T30 *is* dimension (c).

**Output:** `exploration/reports/T30.md` with the definitions panel, overlap matrices, per-definition MDE, direction consistency verdict, verification audits (LLM-rule agreement, 0-value, asaniczka senior-asymmetry), primary recommendation + `exploration/artifacts/shared/seniority_definition_panel.csv`.

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

**Essential sensitivities:** (a) aggregator exclusion — aggregator postings may have systematically different title/description patterns; report SWE-classification quality per stratum.

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
   - Cross-validate with Indeed data: Indeed is excluded from the LLM frame, so `yoe_min_years_llm` is null and `seniority_final` carries only Stage 5 strong-rule labels. Use rule-based `yoe_extracted <= 2` as the primary entry indicator for Indeed; `seniority_final = 'entry'` as a thin sensitivity. If Indeed shows similar patterns to LinkedIn under either indicator, the LinkedIn platform artifact hypothesis weakens.

**Essential sensitivities:** (a) aggregator exclusion — cross-source differences partly reflect aggregator-composition differences; run the key comparability tests with aggregators included and excluded.

**Output:** `exploration/reports/T05.md` with test results, artifact assessment, calibration table, and platform stability assessment

### T06. Company concentration deep investigation `[Agent C]`

**Goal:** Understand the company-level shape of each source. The downstream wave 2/3 tasks aggregate over postings constantly — corpus-level term frequencies, entry-share computations, length distributions, AI-mention rates — and a few prolific employers can drive almost any of those aggregates. This task surfaces concentration patterns *before* substantive analysis runs, so downstream tasks know what to expect and which findings are concentration-driven.

**Why this matters (read first):** Prior exploration runs have repeatedly discovered, only after Wave 2 was partially verified, that entry-labeled pools can be driven by a small number of companies posting the *same exact description* many times each, that entry-share trends can survive or reverse depending on which de-concentration variant is applied, and that a large fraction of companies with substantial scraped presence have zero entry-labeled postings at all. None of these are surfaced by a lighter concentration audit. Treat the company axis as a first-class feature of the data, not an afterthought.

**Steps (SWE, all sources):**

1. **Concentration metrics per source.** HHI, Gini, top-1/5/10/20/50 share of `company_name_canonical` posting volume. Same metrics excluding aggregators (`is_aggregator = true`). Report as a single comparison table across sources.

2. **Top-20 employer profile per source.** For each source separately, list the top 20 companies by SWE posting volume. For each: posting count, share of source, industry where available, mean YOE (primary `yoe_min_years_llm` within the LLM frame; rule `yoe_extracted` as fallback), mean description length, and the within-company entry share under the T30 panel (primary J3; J1/J2 as label sensitivities). This is the foundational profiling that downstream tasks will reference when they need to know "is this finding driven by Amazon" or similar.

3. **Duplicate-template audit.** For each source, identify companies whose SWE postings collapse onto a small number of distinct `description_hash` values. Report the top-10 "duplicate-template" employers per source with: posting count, distinct description count, max-dup-ratio (postings / distinct descriptions). A company posting the same description 25 times is structurally different from a company posting 25 distinct descriptions, even if both contribute "25 rows" to corpus-level aggregates. **Stage 4 now collapses multi-location groups (same company + title + description across 2+ locations) into a single representative row flagged with `is_multi_location = True`; see the `is_multi_location` entry in `docs/preprocessing-schema.md`. Expect this audit to find ~0 duplicate-template groups within a single source after the fix — its role is now a verification, not a discovery. Surface any residual dup-templates (cross-company collisions, near-duplicate descriptions that differ only in boilerplate, etc.) as anomalies worth investigating.**

4. **Entry-level posting concentration (CRITICAL).** Entry-level posting is a specialized activity, not a market-wide one. For each source, compute:
   - How many companies post any entry-labeled SWE roles at all? (Under the T30 panel: primary J3, sensitivities J1/J2.)
   - What share of companies with >=5 SWE postings have ZERO entry-labeled rows?
   - For the companies that DO post entry roles, what is their distribution of entry-share-of-own-postings?
   - Step 6 follows up with per-company specialist identification and categorization; this step measures the shape of the concentration.
   - Any cross-source comparisons of entry posting volume must be read against this concentration backdrop — entry-share trends are about a small subset of employers, not a market-wide shift.

5. **Within-company vs between-company decomposition.** Identify companies with >=5 SWE postings in BOTH arshkon and scraped. For this overlap panel, decompose the aggregate 2024-to-2026 change in entry share, AI mention prevalence, mean description length, and mean tech count into:
   - **Within-company component:** change holding company composition constant
   - **Between-company component:** change driven by different companies entering/exiting the panel
   - Run the entry-share decomposition under each junior variant in the T30 panel (J1–J4). Direction flips between variants are themselves findings (see sensitivity-disagreement principle in the analytical preamble).

6. **Entry-specialist employer identification.** Load the T30 panel. For every company with >=5 SWE postings, compute junior share under J1, J2, J3, and J4. Flag companies where ANY variant gives a junior share >60%. Manually categorize the top 20 flagged companies by employer type: (a) staffing firm, (b) college-jobsite intermediary, (c) tech-giant intern pipeline, (d) bulk-posting consulting, (e) direct employer. Cross-tab against `is_aggregator` — the flagged companies NOT in `is_aggregator` are the invisible intermediary class. Save as `exploration/artifacts/shared/entry_specialist_employers.csv`. Downstream aggregate entry findings must report with and without excluding this set.

7. **Aggregator profile.** What fraction of SWE postings are from aggregators per source? Do aggregator postings differ systematically in seniority/length/requirements? Aggregator share shifts across sources are themselves a confound for any seniority-stratified comparison.

8. **New entrants.** How many 2026 companies have no 2024 match? What is their seniority and content profile vs returning companies?

9. **Per-finding concentration prediction.** For each major analysis category planned in Wave 2/3 (entry share, AI mention rate, description length, term frequencies, topic models, co-occurrence networks), predict whether it would be concentration-driven if computed naively over the full corpus. Recommend a default: cap, dedup, weight, or use as-is. This prediction table is the most important output of the task — it tells downstream agents what to do before they hit the same surprises.

**Essential sensitivities:** (a) aggregator exclusion. (b) Capping is the analysis subject of this task, so do not also apply it as a sensitivity.

**Output:** `exploration/reports/T06.md` with the concentration table, top-20 employer profile per source, duplicate-template audit, entry-poster concentration, within-vs-between decomposition under the T30 panel, aggregator profile, new-entrants profile, and the per-finding concentration prediction table. The prediction table is the deliverable that downstream wave 2/3 tasks should consult first. Also save two shared artifacts:
- `exploration/artifacts/shared/entry_specialist_employers.csv` — companies flagged as entry specialists per step 6. Columns: `company_name_canonical`, `n_swe_postings`, `junior_share_j3`, `junior_share_j1_j2`, `is_aggregator`, `specialist_category` (staffing / college-jobsite / tech-giant-intern / bulk-consulting / direct-employer).
- `exploration/artifacts/shared/returning_companies_cohort.csv` — companies with SWE presence in both 2024 sources (arshkon ∪ asaniczka) and scraped 2026. Columns: `company_name_canonical`, `n_swe_2024`, `n_swe_2026`, `is_aggregator`. Consumed by T37 (sampling-frame retention) and T38 (hiring-selectivity correlation).

### T07. External benchmarks & power analysis `[Agent D]`

**Goal:** Compare our data against BLS/JOLTS benchmarks. Assess statistical power and feasibility for all planned cross-period comparisons.

**Steps:**

*Part A — Feasibility table (primary output, drives all downstream decisions):*
1. Query the data for actual group sizes by source under every T30 panel variant (J1–J4 on the junior side, S1–S4 on the senior side, plus all-SWE). If T30 has not yet published the panel CSV when you run, compute the definitions locally from the same specs in Section 1a.
2. Power analysis for cross-period comparisons: compute minimum detectable effect sizes (MDE) for binary and continuous outcomes at 80% power, α = 0.05, for each key comparison × each applicable seniority definition. Key comparisons: arshkon vs scraped, pooled 2024 vs scraped, arshkon-only senior vs scraped senior, all SWE.
3. Metro-level feasibility: How many metros have ≥50 SWE per period? ≥100? Which qualify for metro-level analysis? (Multi-location postings — `is_multi_location = true` — have `metro_area = NULL` and are excluded from per-metro counts; report the excluded count separately.)
4. Company overlap panel feasibility: How many companies have ≥3 SWE postings in both arshkon and scraped? This determines T16's panel size.
5. Produce a feasibility summary table with columns `analysis_type | comparison | seniority_def | n_group1 | n_group2 | MDE_binary | MDE_continuous | verdict`. For junior-specific rows, `seniority_def` ∈ {J1, J2, J3, J4}; for senior-specific rows, {S1, S2, S3, S4}; for all-SWE rows, `N/A`. The **cross-tab of (comparison × definition)** is the primary deliverable — if J1 is underpowered but J2 is well-powered, Wave 2 should center J2 with J1 as a sensitivity. The orchestrator uses this table at Gate 1 to pick the Wave 2 primary definition.

*Part B — External benchmarks (useful context, not blocking):*
6. Download BLS OES for SOC 15-1252 and 15-1256: state-level employment. Pearson r vs our state-level SWE counts.
7. Industry distribution: our SWE vs OES SWE industry (arshkon + scraped).
8. Download JOLTS information sector from FRED. Contextualize our data periods within the hiring cycle.
9. **Frame the data:** What population does our sample represent? What can and can't we generalize to?

**Essential sensitivities:** (a) aggregator exclusion — report MDEs and feasibility verdicts under both included-all and aggregator-excluded specifications, since aggregator composition affects sample sizes for cross-period comparisons.

**Output:** `exploration/reports/T07.md` + feasibility table CSV (the most important output). Target: r > 0.80 geographic.

---

### Wave 1.5 — Shared Preprocessing

---

### Shared preprocessing spec `[Agent Prep]`

**Goal:** Build shared analytical artifacts that multiple Wave 2+ agents need, preventing duplicate computation and ensuring consistency across agents.

**Steps:**
1. **Cleaned text column.** For all SWE LinkedIn rows (filtered by default SQL): use `description_core_llm` where `llm_extraction_coverage = 'labeled'`, otherwise fall back to raw `description` (the former rule-based `description_core` was retired on 2026-04-10 and must not be used). Strip company names using stoplist from all `company_name_canonical` values, remove standard English stopwords. Save as `exploration/artifacts/shared/swe_cleaned_text.parquet` with columns: `uid`, `description_cleaned`, `text_source` (which column was used: 'llm' or 'raw'), `source`, `period`, `seniority_final`, `seniority_3level`, `is_aggregator`, `company_name_canonical`, `metro_area`, `yoe_min_years_llm`, `yoe_extracted`, `llm_classification_coverage`, `swe_classification_tier`, `seniority_final_source`. Downstream tasks that are sensitive to boilerplate must filter to `text_source = 'llm'`; tasks that only need recall (binary keyword presence) can use both.
2. **Sentence-transformer embeddings.** Using `all-MiniLM-L6-v2`, compute embeddings on first 512 tokens of `description_cleaned` for rows where `text_source = 'llm'`. Process in batches of 256 to respect RAM limits. Save as `exploration/artifacts/shared/swe_embeddings.npy` (float32) with a companion `exploration/artifacts/shared/swe_embedding_index.parquet` mapping row index to `uid`.
3. **Technology mention binary matrix.** Using the ~100-120 technology taxonomy (define regex patterns for: Python, Java, JavaScript/TypeScript, Go, Rust, C/C++, C#, Ruby, Kotlin, Swift, Scala, PHP, React, Angular, Vue, Next.js, Node.js, Django, Flask, Spring, .NET, Rails, FastAPI, AWS, Azure, GCP, Kubernetes, Docker, Terraform, CI/CD, Jenkins, GitHub Actions, SQL, PostgreSQL, MongoDB, Redis, Kafka, Spark, Snowflake, Databricks, dbt, Elasticsearch, TensorFlow, PyTorch, scikit-learn, Pandas, NumPy, LangChain, RAG, vector databases, Pinecone, Hugging Face, OpenAI API, Claude API, prompt engineering, fine-tuning, MCP, LLM, Copilot, Cursor, ChatGPT, Claude, Gemini, Codex, Jest, Pytest, Selenium, Cypress, Agile, Scrum, TDD — expand to ~100+ with regex variations). Scan `description_cleaned` for each. Save as `exploration/artifacts/shared/swe_tech_matrix.parquet` (columns: `uid` + one boolean column per technology).
4. **Company name stoplist.** Extract all unique tokens from `company_name_canonical` values (tokenize on whitespace and common punctuation, lowercase, deduplicate). Save as `exploration/artifacts/shared/company_stoplist.txt`, one token per line.
5. **Structured skills extraction (asaniczka only).** Parse `skills_raw` from asaniczka SWE rows (comma-separated). Save parsed skills with uid as `exploration/artifacts/shared/asaniczka_structured_skills.parquet`.

6. **Within-2024 calibration table.** For ~30 common metrics (description_length, `yoe_min_years_llm` median, tech_count mean, AI keyword prevalence, management indicator rate, scope term rate, soft skill rate, etc.), compute:
   - Arshkon value, asaniczka value, within-2024 effect size (Cohen's d or proportion difference)
   - Arshkon value, scraped value, cross-period effect size
   - Calibration ratio: cross-period / within-2024
   
   Save as `exploration/artifacts/shared/calibration_table.csv`. One row per metric. Columns: `metric`, `metric_type` (`continuous` | `proportion` | `count`), `arshkon_value`, `asaniczka_value`, `scraped_value`, `within_2024_effect`, `within_2024_sd`, `cross_period_effect`, `cross_period_sd`, `calibration_ratio`, `snr_flag` (`above_noise` if ratio ≥ 2, `near_noise` if 1–2, `below_noise` if < 1), `notes`. Wave 2+ agents load this instead of recomputing calibration independently.

7. **Tech-matrix sanity check (mandatory before Wave 2 dispatch).** After building the tech matrix in step 3, compute per-technology mention rate in arshkon SWE, asaniczka SWE, and scraped SWE separately. Flag any technology where the arshkon-to-scraped ratio is >3× or <0.33× as a likely tokenization / text-format issue. Known prior failure: `c\+\+`, `c\#`, `\.net` were backslash-escaped in scraped markdown and under-detected by naive regex (fix: `re.sub(r"\\([+\-#.&_()\[\]\{\}!*])", r"\1", text)` before tokenization). Investigate each flag — either patch the preprocessing and rebuild, or document the residual in the README. Save the sanity table as `exploration/artifacts/shared/tech_matrix_sanity.csv`.

**Output:** `exploration/artifacts/shared/` directory with all artifacts + a `README.md` documenting contents, row counts, `text_source` distribution, sanity-check results, and build time.

**Fallback:** If embedding computation fails (OOM), save partial results and document which rows are covered. Wave 2 agents should check coverage and compute missing embeddings locally if needed.

**Note on LLM budget:** If Stage 9 LLM budget has been allocated for scraped data since the last run, the cleaned text artifact will have higher `text_source = 'llm'` coverage. Re-run this step after any LLM budget allocation to update the shared artifacts.

---

### Wave 2 — Open Structural Discovery

The goal of Wave 2 is to DISCOVER patterns in the data without imposing the RQ1-RQ4 framework. What natural structures exist? What changed? What's unexpected?

---

### T08. Distribution profiling & anomaly detection `[Agent E]`

**Goal:** Establish comprehensive baseline distributions and identify anomalies, surprises, and unexpected patterns across ALL available variables.

**Steps (SWE, LinkedIn-only):**
1. **Univariate profiling:** For every meaningful numeric and categorical column, compute distributions by period and by seniority (T30 panel primary J3/S4; `seniority_final` as the label-sensitivity axis). Produce side-by-side histograms/bar charts for at minimum: `description_length`, `yoe_min_years_llm`, `seniority_final`, `seniority_3level`, `is_aggregator`, `metro_area` (top 15), `company_industry` (top 15 where available).
2. **Anomaly detection:** Flag any distribution that is bimodal, heavily skewed, or shows an unexpected pattern. Are there subpopulations hiding in the data?
3. **Native-label quality diagnostic (arshkon-only).** For `seniority_native = 'entry'` rows in arshkon, compute the YOE distribution (mean, median, share with YOE>=5, share with YOE<=2). If arshkon entry-labeled rows have an unexpected YOE profile, the native classifier may have temporal stability issues that affect any `seniority_native`-based sanity check on later snapshots. Asaniczka has zero native entry labels so cannot be profiled here. This is an important data-quality check; the answer determines whether `seniority_native` can be used as a sanity-check baseline at all.
4. **Within-2024 baseline calibration (arshkon vs asaniczka, mid-senior SWE only):**
   - Compare description length, AI keyword prevalence, organizational language, and top-20 tech stack
   - Compute Cohen's d or equivalent effect sizes
   - Produce calibration table: metric, within-2024 difference, 2024-to-2026 difference, ratio
5. **Junior share trends:** Entry share by period using `yoe_min_years_llm <= 2` as the primary measure (LLM frame), with `seniority_final` label buckets (J1, J2) as sensitivities and `yoe_extracted <= 2` as the extractor ablation. Where comparing against `seniority_native` would be informative, restrict to arshkon. Compute share of all rows and of the appropriate denominator. If primary and sensitivity disagree on direction, investigate WHY rather than picking one.
6. **What variables show the LARGEST changes between periods?** Rank all available metrics by effect size. This identifies where to look deeper.
7. **Domain × seniority decomposition (tests H1).** If T09 archetype labels are available (from `exploration/artifacts/shared/swe_archetype_labels.parquet`), compute entry share by domain archetype by period. Decompose the aggregate entry share change into:
   - **Within-domain component:** entry share change holding archetype composition constant
   - **Between-domain component:** change driven by the market shifting between domain archetypes (e.g., from frontend to ML/AI)
   If the between-domain component accounts for a substantial portion of the aggregate decline, the junior decline is partly a domain recomposition effect, not purely within-domain elimination.
8. **Company size stratification (where data allows).** `company_size` is available for arshkon (99%). Within arshkon, stratify entry share, AI prevalence, and tech count by company size quartile. Do large companies show different patterns? For cross-period analysis where `company_size` is unavailable, use posting volume per company as a rough proxy.

**Essential sensitivities:** (a) aggregator exclusion, (b) company capping, (c) seniority operationalization (T30 panel), (e) source restriction, (f) within-2024 calibration
**Recommended sensitivities:** (g) SWE classification tier, (i) Indeed cross-platform

**Output:** `exploration/reports/T08.md` with plots, summary stats, anomaly flags, calibration table, ranked change list, and domain decomposition

### T09. Posting archetype discovery — methods laboratory `[Agent F]`

**Goal:** Discover natural posting archetypes through unsupervised methods, WITHOUT imposing pre-defined categories. This is also the primary **methods comparison task** — run BERTopic (primary) and NMF (comparison) on the same data and compare what each surfaces. Method agreement strengthens findings; disagreements reveal data structure.

**Steps:**
1. **Sample:** Up to 8,000 SWE LinkedIn postings with **balanced period representation** (~2,700 per period). Within each period, stratify by seniority. For the 2024 allocation, prefer arshkon rows over asaniczka (arshkon has entry-level labels and better text quality). Prefer rows with `text_source = 'llm'` from the shared cleaned text artifact. Record exact sample composition including text_source distribution. Use the SAME sample for all methods.
2. **Load shared artifacts** from `exploration/artifacts/shared/` — cleaned text and embeddings. Build TF-IDF from cleaned text. If shared artifacts unavailable, compute locally.

3. **Method A — BERTopic (primary):**
   - Run BERTopic with sentence-transformer embeddings, UMAP reduction, HDBSCAN clustering
   - Set `min_topic_size` in the 20–50 range to avoid micro-topics; document the chosen value and, if results are hyperparameter-sensitive, report results across the range
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
   - **Entry-level share** of known seniority within each archetype, by period. This is critical for H1: if ML/AI has structurally lower entry share AND grew substantially as a share of the corpus, the aggregate entry decline may be driven by domain composition.
   - Period distribution (what % of each period falls in this cluster?)
   - Average description length, YOE, tech count
   - Give each cluster a descriptive name based on its content, not based on RQ1-RQ4

7. **Temporal dynamics:** How did archetype proportions change from 2024 to 2026? Which grew? Which shrank? Are there archetypes that only exist in one period?

8. **Visualization:** UMAP (2D) of the embeddings, colored by: (a) best-method clusters, (b) period, (c) seniority. Three separate plots. Also produce the same with PCA for comparison — does the visual story change?

9. **Key discovery question:** Do the clusters align with seniority levels (entry/mid/senior map to different clusters)? Or do they align with something else entirely (industry, role type, tech stack, company size)? The answer reveals the dominant structure of the market. Compute Normalized Mutual Information (NMI) between cluster assignments and seniority, period, and tech domain to quantify this.

10. **Save cluster labels for downstream use.** Save the best method's cluster assignments as `exploration/artifacts/shared/swe_archetype_labels.parquet` (columns: `uid`, `archetype`, `archetype_name`). This allows T11, T16, T17, T20 and other downstream tasks to stratify by domain archetype.

**Essential sensitivities:** (a) aggregator exclusion, (b) company capping (cap at 20-50 per `company_name_canonical` before embedding / clustering; a few prolific employers can dominate corpus-level archetypes), (d) description text source
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

**Essential sensitivities:** (a) aggregator exclusion, (b) company capping (title-frequency is a corpus-aggregate metric; cap at 20-50 per `company_name_canonical` to prevent prolific employers from dominating the emerging-title list)
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
   - YOE from `yoe_min_years_llm` (primary); `yoe_extracted` as rule ablation for out-of-frame rows
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
7. **Outlier analysis:** What do the most complex postings look like? (Top 1% by requirement_breadth.) Are they real or template-bloated?

**Note on domain-stratified scope inflation:** this analysis is performed in T28 (Wave 3), which consumes T11's per-posting feature parquet (see Output) once T09 archetype labels are available. T11 does not run it directly.

**Essential sensitivities:** (a) aggregator exclusion, (b) company capping, (c) seniority operationalization, (f) within-2024 calibration

**Output:** `exploration/reports/T11.md` + complexity distribution plots + per-seniority comparison tables + management term breakdown + domain-stratified scope inflation. Also save the per-posting feature frame as `exploration/artifacts/shared/T11_posting_features.parquet` with columns: `uid`, `tech_count`, `requirement_breadth`, `credential_stack_depth`, `tech_density`, `scope_density`, `mgmt_strong_density`, `mgmt_broad_density`, `ai_binary`, `education_level`, `yoe_min_years_llm`. Consumed by T20 (boundary feature vector), T33 (hiring-bar regression), T35 (ecosystem features).

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
7. **Secondary comparisons — label-based (`seniority_final`), if sample sizes allow:**
   - Entry 2024 vs Entry 2026
   - Mid-senior 2024 vs Mid-senior 2026
   - **Entry 2026 vs Mid-senior 2024 (relabeling diagnostic):** This tests whether 2026 entry postings resemble relabeled 2024 senior postings (seniority-content dominant) or whether the changes represent a temporal shift applied to entry-level roles (period-effect dominant). If period-effect dominates, scope inflation is adding NEW dimensions rather than importing existing senior requirements downward. Interpret the result in this frame.
   - Within-2024: arshkon mid-senior vs asaniczka mid-senior (calibration — how much change is instrument noise?)
8. **Secondary comparisons — YOE-band (`yoe_min_years_llm`), strips out label-quality confound:**
   - YOE ≤ 2 in 2024 vs YOE ≤ 2 in 2026 (junior-band content evolution at a fixed YOE threshold)
   - YOE ≥ 5 in 2024 vs YOE ≥ 5 in 2026 (senior-band equivalent)
   - **YOE ≤ 2 in 2026 vs YOE ≥ 5 in 2024 (YOE-based relabeling diagnostic):** the parallel to step 7's label-based relabeling test, but with experience-band operationalization. If the two versions disagree on direction, the difference points to label drift (asaniczka entry gap, LLM-frame selection, platform taxonomy drift).
9. **Bigram analysis:** Phrase-level changes (prompt engineering, AI agent, code review) are often more informative than unigrams.
10. **BERTopic cross-validation:** Fit BERTopic on the combined corpus with period as class variable. Which topics are most period-specific? Compare with Fightin' Words.
11. **Report n per corpus for every comparison.** Flag any with n < 100.

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
3. **What's driving description length growth?** Stacked bar chart: description composition by period. Did requirements grow? Or benefits/boilerplate/about-company? This is critical for interpreting whether "more requirements" is a real signal or just longer postings.
4. **Tone markers:**
   - Imperative density: "you will", "you'll", "must", "should" per 1K chars
   - Inclusive language: "we", "our team", "you'll join" per 1K chars
   - Formal vs informal: passive constructions vs active/direct address
   - Marketing language: "exciting", "innovative", "cutting-edge", "world-class" per 1K chars
5. **Entry-level specifically:** How did the structure and tone of entry-level JDs change compared to mid-senior?

**Essential sensitivities:** (a) aggregator exclusion, (d) description text source, (f) within-2024 calibration (readability and section-anatomy metrics are cross-period comparisons; compute the arshkon-vs-asaniczka baseline on the same metrics)

**Output:** `exploration/reports/T13.md` + readability comparison table + stacked section chart + tone metrics. Also save per-posting readability/structure metrics as `exploration/artifacts/shared/T13_readability_metrics.parquet` with columns: `uid`, `flesch_kincaid_grade`, `flesch_reading_ease`, `gunning_fog`, `avg_sentence_length`, `sentence_length_sd`, `type_token_ratio`, `syllable_count`, `lexicon_count`, `imperative_density`, `inclusive_density`, `marketing_density`. Consumed by T29 (LLM-authorship detection reuses sentence-length stats). Save the section classifier script as `exploration/scripts/T13_section_classifier.py` — T12 and Wave 3.5 T33 import it.

### T14. Technology ecosystem mapping `[Agent I]`

**Goal:** Map not just individual technology mentions, but how technologies co-occur and form natural skill bundles — and how those bundles changed.

**Steps (SWE, LinkedIn-only):**
1. **Technology taxonomy.** Use the canonical taxonomy defined in the Wave 1.5 Shared preprocessing spec (step 3), which enumerates ~100–120 technologies across languages, frontend/backend frameworks, cloud/DevOps, data, AI/ML traditional and LLM-era, AI tools, testing, and practices. The shared tech matrix (`exploration/artifacts/shared/swe_tech_matrix.parquet`) is built from that taxonomy. Load it rather than redefining patterns here.
2. **Mention rates:** For each technology, compute % of postings mentioning it, by period x seniority. Use binary (any mention) as primary metric.
3. **Technology co-occurrence network (deferred to T35).** Full per-period co-occurrence network construction, community detection, and cross-period community comparison are executed by Wave 3.5 task T35 (ecosystem crystallization), which consumes this task's tech matrix. Do not re-run the network analysis here. T14 focuses on per-technology rates, diversity, and structured-skills validation.
4. **Rising, stable, declining:** Classify each technology by its trajectory. Produce a "technology shift" heatmap.
5. **Stack diversity:** How many distinct technologies does the median posting mention? By period x seniority. Is tech breadth increasing or specializing?
6. **AI integration pattern:** Among postings mentioning AI tools/LLM, what traditional technologies co-occur? (Is AI adding to existing stacks, or replacing components?) **Length-normalization check:** The finding that AI-mentioning postings have more technologies could partly be an artifact of AI postings being longer. Compute tech density (techs per 1K chars) for AI-mentioning vs non-AI postings. Report both raw count and density to quantify the length confound.
7. **Structured skills baseline (asaniczka only).** Load parsed skills from `exploration/artifacts/shared/asaniczka_structured_skills.parquet`. Compute frequency table of all distinct skills across the asaniczka SWE rows. Produce top-100 skills list.
8. **Structured vs extracted validation.** Compare technology frequencies from structured `skills_raw` (step 7) against description-extracted tech frequencies (step 2) for asaniczka SWE. Compute rank correlation. Where they diverge: is the structured field capturing things the regex misses, or vice versa?
9. **Seniority-level skill differences from structured data.** Using asaniczka's parsed skills and seniority labels: which skills are significantly more associated with entry-level vs mid-senior? (Chi-squared per skill, with Bonferroni or FDR correction.) This provides a structured-data baseline for the seniority boundary question.

**Essential sensitivities:** (a) aggregator exclusion, (b) company capping, (f) within-2024 calibration
**Recommended sensitivities:** (g) SWE classification tier

**Output:** `exploration/reports/T14.md` + tech heatmap + co-occurrence network visualization + community comparison + structured skills baseline CSV + structured-vs-extracted validation

### T15. Semantic similarity landscape & convergence analysis `[Agent I]`

**Goal:** Map the full semantic structure of the SWE posting space, visualize how it changed between periods, and test whether seniority levels are converging. Compare text representations and dimensionality reduction methods.

**Method complement:** T20 measures boundary discriminability using structured-feature classifiers (logistic regression AUC on tech count, YOE, mgmt density, etc.). T15 complements this with unsupervised embedding and TF-IDF centroid similarity. Cross-method agreement strengthens the seniority-boundary finding; disagreement is itself informative about mechanism (text-semantic blurring vs structured-feature blurring).

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

**Essential sensitivities:** (a) aggregator exclusion, (b) company capping (centroid computation is corpus-aggregate; cap at 20-50 per `company_name_canonical` before embedding / TF-IDF), (c) seniority operationalization, (d) description text source, (e) source restriction, (f) within-2024 calibration
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
   - Entry share — compute under `yoe_min_years_llm <= 2` (primary) and under `seniority_final IN ('entry','associate')` (label sensitivity). If the two disagree at the company level, treat it as a per-company measurement signal worth examining.
   - AI keyword prevalence (binary per posting)
   - Mean description length
   - Mean tech count
   - Mean org_scope term count
3. **Cluster companies by their change profile.** k-means on the change vectors. Are there distinct strategies? Name them (e.g., "AI-forward", "traditional hold", "scope inflator", "downsizer").
4. **Within-company vs between-company decomposition:** For entry share, AI prevalence, and description length, how much of the aggregate 2024-to-2026 change is driven by within-company change vs different companies entering/exiting the sample? Run the entry-share decomposition under `yoe_min_years_llm <= 2` (primary) and under `seniority_final` (label-based sensitivity). If results disagree in direction, report both and discuss the mechanism — this is a critical methodological finding, not a problem to bury. If T09 archetype labels are available, add a domain dimension: decompose the entry share change into within-domain, between-domain, and between-company components.
5. **Within-company scope inflation (if validated patterns available).** If T22's validated management/scope patterns are available (from shared artifacts), compute within-company change in entry-level scope indicators for the overlap panel. This is the cleanest test of scope inflation: same companies across periods.
6. **New market entrants:** Profile companies in 2026 with no 2024 match. What industries? How do their postings compare?
7. **Aggregator vs direct employer:** Compare change patterns. Are aggregators showing different trends?

**Essential sensitivities:** (a) aggregator exclusion, (c) seniority operationalization (T30 panel)
**Recommended sensitivities:** (b) company capping

**Output:** `exploration/reports/T16.md` + company cluster characterization + decomposition results. Also save two shared artifacts consumed by Wave 3.5:
- `exploration/tables/T16/overlap_panel.csv` — the arshkon∩scraped overlap panel (companies with ≥3 SWE postings in both periods). Columns: `company_name_canonical`, `n_2024`, `n_2026`, `is_aggregator`, `is_entry_specialist` (from T06), `strategy_cluster` (from step 3).
- `exploration/tables/T16/company_change_vectors.csv` — per-company 2024→2026 delta vectors. Columns: `company_name_canonical`, `entry_share_delta_j3`, `entry_share_delta_j1_j2` (label sensitivity), `ai_prevalence_delta_strict`, `ai_prevalence_delta_broad`, `desc_length_delta`, `tech_count_delta`, `org_scope_delta`, `breadth_resid_delta`. These feed T31 (pair-level drift), T37 (sampling-frame retention), T38 (hiring-selectivity correlation).

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

**Goal:** Characterize the temporal structure of our data and estimate rates of change in the SWE market. Our data consists of discrete historical snapshots plus a growing scraped window, not a continuous time series — this task works within that constraint honestly.

**Data context:** Query the current `date_posted`, `scrape_date`, and `period` ranges for each source before estimating rates. Treat arshkon and asaniczka as historical snapshots and scraped as the current growing window. We have snapshots/windows, not a continuous time series.

**Entry-level rates involving asaniczka depend on the operationalization.** Asaniczka has zero native entry labels, so `seniority_native` cannot detect asaniczka entry. For rate-of-change estimation:
- Under `yoe_min_years_llm <= 2` (**primary**, label-independent, all three snapshots within the LLM frame).
- Under `seniority_final`: label-based sensitivity. Asaniczka entry signal here comes via the Stage 10 LLM; the rule half is unknown where titles lack strong keywords. Report the three-snapshot rate.
- Under `seniority_native`: arshkon-only sanity check. Report the arshkon vs scraped rate.
- Under `yoe_extracted <= 2`: rule-extractor ablation for the primary.
Report the rate-of-change under each operationalization and discuss any disagreement.

**Steps:**
1. **Rate-of-change estimation.** For key metrics (entry share of known seniority, AI keyword prevalence, median description length, median tech count, org scope density):
   - Compute value at each source window: asaniczka, arshkon, scraped
   - Historical-snapshot annualized rate: compute elapsed time from the observed source date ranges, not from hardcoded month gaps
   - Cross-period annualized rate: compute elapsed time from the observed arshkon and scraped windows, not from hardcoded month gaps
   - Acceleration ratio: cross-period annualized rate / within-2024 annualized rate
   - If acceleration ratio >> 1 for a metric, something changed faster after 2024 than during 2024
   - Produce a rate-of-change comparison table

2. **Within-arshkon stability.** Using `date_posted`, bin arshkon by its observed internal date range and check whether key metrics (AI keyword prevalence, description length, tech count) vary significantly across bins. If they do, the arshkon snapshot has internal heterogeneity.

3. **Scraper yield characterization.** Examine daily SWE posting counts across scrape dates. Is the first day an accumulated backlog while subsequent days capture new flow? Compare content, seniority distribution, and AI mention rates across scrape dates. This helps calibrate whether our scraped snapshot represents stock (accumulated postings) or flow (new postings).

4. **Posting age analysis.** Examine `posting_age_days` for scraped rows where available. Characterize current coverage before interpreting the distribution. If posting ages cluster at specific values, this reveals the market's posting lifecycle.

5. **Within-scraped-window stability and day-of-week analysis.** Across the currently observed scraped dates, check: are key metrics (AI mention rate, seniority distribution, description length) stable across days? Are there day-of-week effects? This tells us whether our scraped window is internally consistent.

6. **Timeline contextualization.** Place our three snapshots on a timeline and annotate with major AI tool releases between them: GPT-4 (Mar 2023), Claude 3 (Mar 2024), GPT-4o (May 2024), Claude 3.5 Sonnet (Jun 2024), o1 (Sep 2024), DeepSeek V3 (Dec 2024), GPT-4.5 (Feb 2025), Claude 3.6 Sonnet (Apr 2025), Claude 4 Opus (Sep 2025), Gemini 2.5 Pro (Mar 2026). This provides qualitative temporal context even with only 3 data points.

**Essential sensitivities:** (f) within-2024 calibration — backbone of the rate-of-change estimation; temporal comparisons without arshkon-vs-asaniczka baseline are methodologically weak.
**Recommended sensitivities:** (e) source restriction

**Output:** `exploration/reports/T19.md` + rate-of-change comparison table + within-arshkon stability check + within-scraped-window stability analysis

### T20. Seniority boundary clarity `[Agent L]`

**Goal:** Measure how sharp the boundaries between seniority levels are and whether they blurred between periods. This goes BEYOND the "relabeling hypothesis" to map the full boundary structure.

**Method complement:** T15 tests seniority convergence using embedding and TF-IDF centroid similarity (unsupervised, text-based). T20 complements with supervised-learning boundary discriminability via logistic-regression AUC on structured features. Cross-method agreement strengthens the boundary finding.

**Steps (SWE, LinkedIn-only, seniority_final != unknown):**
1. **Feature extraction.** Load precomputed features from `exploration/artifacts/shared/T11_posting_features.parquet` (produced by T11). These include the feature vector needed for boundary analysis:
   - `yoe_min_years_llm` (numeric; fall back to `yoe_extracted` for non-LLM-frame rows; impute median where both null)
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
5. **Gap evolution per metric (Δgap + attribution).** For each feature in step 1, compute `gap_2024 = mean(M | senior, 2024) − mean(M | junior, 2024)`, the same for 2026, and `Δgap = gap_2026 − gap_2024`. Alongside report `ΔM_senior` and `ΔM_junior` (same-level change 2024→2026) and `attribution_senior = |ΔM_senior| / (|ΔM_senior| + |ΔM_junior|)` — the share of gap movement driven by the senior side. Use the T30 panel primary (S4 vs J3) with S1/S2 and J1/J2 as label sensitivities. This complements step 3's classifier AUC by decomposing which specific metrics drove the boundary shift and which side moved.
6. **Continuous YOE × period interaction.** Using `yoe_min_years_llm`, fit `M ~ yoe + period + yoe × period` (OLS with heteroskedasticity-robust SE; control for `log(description_length)`) for each continuous feature in step 1 and for the key outcome metrics (AI-strict binary, requirement_breadth, mgmt density, orch density). The `yoe × period` coefficient tests whether YOE sorts roles more strongly in 2026 — a continuous-variable analogue of boundary sharpening that avoids threshold choices. Report the coefficient, 95% CI, and direction per metric. Flag metrics where the interaction is significant: that's where the YOE-to-content relationship structurally changed.
7. **The "missing middle" question:** Is the associate level becoming more like entry, more like mid-senior, or disappearing? Compute distances.
8. **Domain-stratified boundary analysis.** If T09 archetype labels are available (from `exploration/artifacts/shared/swe_archetype_labels.parquet`), run the boundary analysis within each domain archetype separately. Does boundary blur/sharpening differ across ML/AI vs Frontend vs Embedded vs Data domains?
9. **Full similarity matrix** using the structured features (not text): compute average feature profiles per seniority x period and present as a heatmap.

**Essential sensitivities:** (a) aggregator exclusion, (c) seniority operationalization (T30 panel)
**Recommended sensitivities:** (g) SWE classification tier

**Output:** `exploration/reports/T20.md` + AUC comparison + feature importance analysis + gap-evolution table + continuous YOE × period coefficients + boundary heatmap + domain-stratified results

### T21. Senior role evolution deep dive `[Agent L]`

**Goal:** Go deep on how senior SWE roles specifically are evolving — not just management-to-orchestration, but the full picture.

**Steps (SWE, LinkedIn-only, seniority_final IN ('mid-senior', 'director')):**
1. **Language profiles.** Define three profiles (not just two):
   - **People management:** manage, mentor, coach, hire, interview, grow, develop talent, performance review, career development, 1:1, headcount, people management, team building, direct reports
   - **Technical orchestration:** architecture review, code review, system design, technical direction, AI orchestration, agent, workflow, pipeline, automation, evaluate, validate, quality gate, guardrails, prompt engineering, tool selection
   - **Strategic scope:** stakeholder, business impact, revenue, product strategy, roadmap, prioritization, resource allocation, budgeting, cross-functional alignment
   **Validate your patterns:** Sample 50 matches for each profile's key patterns and check precision. Generic terms like "leading", "leadership", "strategic" used as adjectives rather than role verbs are known to inflate management indicators substantially. Remove low-precision patterns and report results with strict and broad sets.
2. **Per posting:** Compute density (mentions per 1K chars) for each profile.
3. **2D and 3D scatter:** Management vs Orchestration vs Strategic, colored by period. How did the distribution shift?
4. **Senior sub-archetypes:** Cluster senior postings by their language profiles. Are there distinct types (people-manager, tech-lead, architect, strategist)? How did their proportions change?
5. **AI interaction:** Among senior postings mentioning AI, how does the management/orchestration/strategic balance differ from non-AI-mentioning senior postings?
6. **Director specifically:** Directors are a small but important group. What do their postings look like? How do they differ from mid-senior?
7. **The "new senior" question:** Is there an emergent senior archetype that didn't exist in 2024? What does it look like?
8. **Cross-seniority management comparison.** How did management language in SENIOR postings change compared to the entry-level change? If management language expanded at all levels (not just entry), that's evidence against downward migration and for a field-wide template shift.

**Essential sensitivities:** (a) aggregator exclusion, (b) company capping (clustering on management/orchestration/strategic density is corpus-aggregate; cap at 20-50 per `company_name_canonical` to prevent prolific employers from dominating cluster centroids), (f) within-2024 calibration (senior-role language densities should be calibrated against the arshkon-vs-asaniczka baseline per the preamble rule)

**Output:** `exploration/reports/T21.md` + management-orchestration-strategic charts + senior sub-archetype analysis + cross-seniority management comparison. Also save senior-cluster assignments as `exploration/tables/T21/cluster_assignments.csv` with columns: `uid`, `cluster_id`, `cluster_name` (descriptive, content-driven — see step 7), `mgmt_density`, `orch_density`, `strat_density`, `mentor_binary`, `ai_binary`, `period`, `seniority_final`. Consumed by T34 (cluster profiling) and the senior-archetype section of T26 SYNTHESIS.

### T22. Ghost & aspirational requirements forensics `[Agent M]`

**Goal:** Identify ghost-like and aspirational requirement patterns through systematic text analysis.

**Scope complement:** T22 characterizes the prevalence and features of ghost / aspirational postings (descriptive). T33 (hidden hiring-bar) tests whether requirements-section contraction correlates with lowered YOE / credential asks (correlational / mechanism). The two are complementary: T22 describes, T33 tests.

**Steps (SWE, LinkedIn-only):**
1. **Ghost indicators per posting:**
   - **Kitchen-sink score:** Number of distinct technologies x number of organizational scope terms (high product = everything-and-the-kitchen-sink)
   - **Aspiration ratio:** Count of hedging language ("ideally", "nice to have", "preferred", "bonus", "a plus") / count of firm requirement language ("must have", "required", "minimum", "mandatory"). Higher ratio = more aspirational.
   - **YOE-scope mismatch:** Entry-labeled postings (J1 label OR J3 LLM-YOE bucket) where `yoe_min_years_llm >= 5` (primary) or `yoe_extracted >= 5` (rule ablation), OR with ≥3 senior scope terms (architecture, ownership, system design, distributed systems). Note: `yoe_min_years_llm = 0` is an entry-signal confirmation, not a contradiction.
   - **Template saturation:** Within each company, compute pairwise cosine similarity of requirement sections across their postings. Flag companies with mean similarity > 0.8 (copy-paste templates).
   - **Credential impossibility:** Postings requiring contradictory credentials (e.g., 10+ YOE for entry-level, or both "no degree required" and "MS required")
2. **Prevalence by period x seniority:** How common is each ghost indicator? Did ghost-like patterns increase or decrease?
3. **AI ghostiness test:** Are AI requirements MORE aspirational than traditional requirements? Compute aspiration ratio separately for AI terms vs non-AI terms within the same postings.
4. **The 20 most ghost-like entry-level postings:** Display their title, company, and requirements section. Are they real roles or artifacts?
5. **Aggregator vs direct:** Compare ghost indicators. Are aggregators more ghost-like?
6. **Industry patterns:** Where company_industry is available, do certain industries have more ghost-like postings?

**Essential sensitivities:** (aggregator comparison IS core to this task — run all ghost indicators separately for aggregator vs direct employer)
**Recommended sensitivities:** (d) description text source

**Output:** `exploration/reports/T22.md` + ghost prevalence tables + examples + `exploration/artifacts/shared/validated_mgmt_patterns.json` (validated patterns with precision scores for downstream use). JSON schema: one object per pattern family, e.g. `{"management_strict": {"pattern": "mentor|coach|hire|…", "precision": 0.XX, "sample_n": 50, "semantic_precision_measured": true, "sub_pattern_precisions": {"mentor": 0.XX, …}}, "management_broad": {…}, "ai_strict": {…}, …}`. Wave 3.5 agents load this file and MUST NOT re-derive patterns.

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

**Goal:** T09 characterizes how the SWE posting market is organized (by tech domain, seniority, period, etc.) and which archetypes are growing. If archetype composition shifts between periods, aggregate scope-and-content changes may be partly within-domain and partly between-domain composition shifts. T08 step 7 and T11 step 7 were both spec'd to do this decomposition but were deferred because T09 had not yet produced the archetype labels. This task picks them up and extends them.

**Dependency:** Requires `exploration/artifacts/shared/swe_archetype_labels.parquet` from T09.

**Steps:**

1. **Load T09 archetype labels** and join to the SWE corpus. Report archetype distribution by period.

2. **Domain × seniority decomposition for the entry-share trend.** Using `yoe_min_years_llm <= 2` (primary) and `seniority_final` (label-based sensitivity), compute entry share by archetype × period. Decompose the aggregate change into:
   - Within-domain component (entry share change holding archetype composition constant)
   - Between-domain component (change driven by the market shifting between archetypes)
   - Interaction
   - Report under both seniority operationalizations. If they disagree at the archetype level, drill in.

3. **Domain-stratified scope inflation (supersedes T11 step 7).** Load T11's per-posting feature parquet (`exploration/artifacts/shared/T11_posting_features.parquet`). Within each archetype, compute the change in `requirement_breadth`, `tech_count`, `scope_density`, AI mention rate, and credential stack depth between periods. Is scope inflation a within-domain phenomenon, a between-domain composition effect, or both? Which archetypes are growing the most in scope?

4. **Junior vs senior content within each archetype.** For each archetype, compare entry vs mid-senior postings on requirement breadth, AI mention rate, scope language, and management/mentorship language. Is the junior/senior gap closing within some archetypes and not others? This is the more nuanced version of the convergence question that T15 ran at the corpus level (and rejected). The corpus-level null may hide within-domain convergence in some archetypes.

5. **Senior archetype shift by domain.** T11 found the senior tier is shifting toward IC+mentoring rather than people-management. Does this hold across all domains, or is it concentrated in some (e.g., AI/ML, where the work itself is changing)? Use the strict mentoring detector from T11.

6. **Cross-validate the AI/ML expansion.** If T09 reports the AI/ML archetype as growing, profile it: who are the top employers, what is the entry vs senior mix, what tech stack dominates, what is the description length and credential stack profile. Is any AI/ML growth coming from new entrants or from existing employers shifting their mix?

**Essential sensitivities:** (a) aggregator exclusion, (b) company capping per the unit-of-analysis rule (capping is appropriate for the within-archetype term/scope analyses, not appropriate for the entry-share decomposition), (c) seniority operationalization (T30 panel)

**Output:** `exploration/reports/T28.md` with the domain × seniority decomposition table, per-archetype scope changes table, per-archetype junior/senior comparison, and the AI/ML deep dive.

---

### T29. LLM-authored description detection `[Agent O]`

**Goal:** Test the hypothesis that part of what we measure as "employer requirements changing" between 2024 and 2026 is actually downstream of recruiters adopting LLMs to draft job descriptions during the same window. This is exploratory and may yield no signal — which is itself informative. If the hypothesis is supported, it would unify several Wave 2 findings (length growth, tech-density decrease, AI mention explosion, credential vocabulary stripping, mid-senior tone shift) into a single mechanism, AND it would be a methodological warning of broad applicability for any longitudinal posting study.

**Steps:**

1. **Define LLM-authorship signals.** Build a per-posting authorship score from observable text features. For sentence-length and vocabulary-diversity metrics, load the shared readability parquet (`exploration/artifacts/shared/T13_readability_metrics.parquet`) produced by T13 rather than recomputing. Candidate features (you should add or remove based on your judgment and any quick research on current LLM stylistic tells):
   - **Signature vocabulary density:** classic LLM tells like `delve`, `tapestry`, `leverage`, `robust`, `unleash`, `embark on`, `navigate`, `cutting-edge`, `in the realm of`, `comprehensive`, `seamless`, `furthermore`, `moreover`, `it's worth noting`, `notably`, `align with`, `at the forefront`, `pivotal`, `harness`, `dynamic`, `vibrant`. Compute density per 1K chars.
   - **Em-dash density:** LLMs use em-dashes (`—` and `--`) noticeably more than human writers. Per 1K chars.
   - **Sentence length distribution:** use `avg_sentence_length` and `sentence_length_sd` from T13's parquet. LLMs produce longer, more uniform sentences than humans.
   - **Vocabulary diversity:** use `type_token_ratio` from T13's parquet (within-posting). Additionally compute across-posting uniformity (corpus-level type-token variance). If LLMs are writing the descriptions, postings may become more uniform in vocabulary across the corpus.
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

### Wave 3.5 — Induced Hypothesis Tests

Wave 3.5 is a dependent computational phase between Wave 3 and V2 Gate 3 verification. Each task operationalizes a high-value hypothesis that Wave 1-3 findings induced — the tasks are designed so their outputs flow directly into T26 SYNTHESIS.md's ranked findings and robustness appendix, not as an optional appendix or a post-synthesis extension. Wave 3.5 is part of the main flow of the pipeline.

**Why Wave 3.5 exists as a separate phase (and not as Wave 3 extensions).** Every Wave 3.5 task depends on specific artifacts that only exist after Wave 3 completes: T16's arshkon∩scraped overlap panel, T21's senior cluster assignments, T09's archetype labels, T13's section classifier, T22's validated management/AI patterns, T10's disappearing-title list, T06's returning-companies cohort. Folding these tasks into Wave 3 would create circular dependencies. Separating them into Wave 3.5 keeps each wave coherent and lets V2 Gate 3 verification cover both phases in one adversarial pass.

**Dependency chain (dispatch only after Wave 3 completes and its artifacts are persisted):**
- T31 (same-co × same-title drift) requires T16's overlap panel persisted to `exploration/tables/T16/`.
- T32 (cross-occupation extension of T23's employer-vs-worker AI-adoption gap) requires T18's DiD framework and T23's SWE divergence estimate.
- T33 (hidden hiring-bar) requires T13's section classifier module (`exploration/scripts/T13_section_classifier.py`) and T11's per-posting feature parquet.
- T34 (emergent senior-role profiling) requires T21's k-means cluster assignments and T09's archetype labels.
- T35 (ecosystem crystallization) requires T14's co-occurrence tooling and the full tech matrix.
- T36 (legacy substitution map) requires T10's disappearing-title list and the cleaned text artifact.
- T37 (sampling-frame robustness) requires T06's returning-companies cohort and T16's overlap panel.
- T38 (hiring-selectivity × scope correlation) requires T16's per-company change vectors.

**Hypothesis coverage.** Wave 3.5 directly tests 8 hypotheses:
- Four from T24's planned H_A-H_J list: **H_A** cross-occupation employer/worker AI divergence (T32), **H_B** hidden hiring-bar signal (T33), **H_C** emergent senior-role archetype (T34), **H_H** sampling-frame artifact (T37).
- Four introduced by Wave 3.5 itself: **H_K** ecosystem crystallization (T35), **H_L** legacy substitution (T36), **H_M** same-co × same-title drift (T31), **H_N** hiring-selectivity × scope (T38).

The remaining T24 hypotheses (H_D, H_E, H_F, H_G, H_I, H_J) are deferred to the analysis phase — T24 inventories them with explicit priority.

**Agents.** Q (T31, T32), R (T33, T34), S (T35, T36), T (T37, T38). All four dispatch in parallel once Wave 3 outputs are in place.

**Integration with Gate 3 and synthesis.** V2 verification runs AFTER Wave 3.5 completes and covers BOTH Wave 3 and Wave 3.5 headline numbers. One unified Gate 3 memo documents the full pre-synthesis state. Agent N (Wave 4) reads Wave 3 + Wave 3.5 reports together; T24 consolidates hypothesis verdicts and T26 integrates Wave 3.5 findings into the ranked findings list of SYNTHESIS.md as first-class paper claims. Agent P (Wave 5) consumes the unified SYNTHESIS.md, so Wave 3.5 findings naturally surface in the presentation without special handling.

**All Wave 3.5 agents MUST:**

- Use the V1-validated AI-mention patterns (strict primary: `\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|vector database|pinecone|huggingface|hugging face)\b`; broad drops low-precision patterns such as bare `agent` and `mcp`) and the V1-validated strict management pattern (`mentor|coach|hire|headcount|performance_review`). Load `exploration/artifacts/shared/validated_mgmt_patterns.json` rather than re-deriving.
- Respect the Gate 0 pre-committed ablation dimensions (T30 panel for every seniority-stratified headline, aggregator exclusion, length residualization for composites, semantic precision ≥80% for any new pattern introduced).
- Flag the LLM-frame J2 selection artifact when restricting text-sensitive analyses to labeled rows.
- Report findings under pooled-2024 as the default baseline. Arshkon-only is a sensitivity where label-based variants (J1/J2, S1/S2) are cited, or where T30's asaniczka senior-asymmetry verdict flagged a material senior-side skew.
- **Close each report with a "Headline claims for SYNTHESIS" section** listing the 1-3 specific claims the task contributes to the paper, in the form "[claim sentence] — evidence: [specific table/figure], sensitivity verdict: [robust / partial / flagged]." This is the contract Agent N (T26) relies on when integrating Wave 3.5 into SYNTHESIS.md.

---

### T31. Same-company × same-title longitudinal drift `[Agent Q]`

**Goal:** Quantify the tightest-possible within-employer rewriting signal by comparing same-company × same-title pairs across 2024 and 2026. Produces the per-pair drift distribution that the paper's "employer-side rewriting" claim can cite at the pair level — cleaner than T16's company-level decomposition because it holds title composition constant.

**Hypothesis under test:** H_M — introduced by Wave 3.5. If T16 finds that AI rewriting is within-company, that finding should hold or strengthen when we restrict to same-title pairs, because title composition within a company can shift between periods and inject noise.

**Steps:**
1. **Overlap panel.** Start from the arshkon∩scraped companies with ≥3 SWE postings in both periods (from T16). Optionally extend to the pooled-2024∩scraped panel for a secondary analysis.
2. **Pair identification.** Within each company, identify title pairs present in both periods. Use the raw `title` field lowercased (not `title_normalized` — that strips level indicators; see T30 fix). Primary threshold: require ≥3 postings per (company × title × period) cell to reduce per-pair drift noise. Also produce a ≥2-postings sensitivity cut (looser, higher n pairs) and compare the two drift distributions — if the ≥2 cut's mean / median drift differs materially from the ≥3 cut, small-cell noise is driving the looser estimate. Report the actual qualifying-pair counts under each threshold.
3. **Per-pair feature drift.** For each pair compute the 2024 → 2026 delta on:
   - AI-mention strict binary share (V1-validated pattern)
   - AI-mention broad binary share
   - `requirement_breadth` residualized on description cleaned length (per the canonical length-residualization formula in §1b)
   - Mentor-binary rate (V1-validated strict)
   - Requirements-section character share (T13 classifier)
   - Mean `yoe_min_years_llm` (within the pair); `yoe_extracted` as ablation
   - Median description cleaned length
4. **Drift distribution.** Mean, median, p10, p90 per metric. Produce 2D scatter (AI-mention Δ × breadth-resid Δ) at the pair level. Identify outlier pairs.
5. **Archetype stratification.** Project each pair's representative posting onto T09 archetype labels via nearest-centroid from `swe_embeddings.npy`. Is drift uniform across archetypes, or concentrated in ML/AI / cloud_devops / frontend?
6. **Top-20 drift pair inspection.** Manual read of the 20 pairs with the largest AI-mention Δ and the 20 with the largest breadth-resid Δ. What changed in the content? Is there a common narrative (e.g., "Senior Backend Engineer at Bank X" gained LangChain/RAG in 2026 while keeping the same title)?
7. **Consistency check vs T16.** Pair-level drift should be ≤ company-level drift (less noise from within-company title composition shifts). If pair-level drift EXCEEDS company-level drift, investigate — may indicate that within-company, same-title postings change MORE than the company-level aggregate (because composition shifts partially mask within-pair change).

**Essential sensitivities:** (a) aggregator exclusion, (b) cap at 10 postings per (company × title × period) cell.

**Output:** `exploration/reports/T31.md` + pair-level drift distribution CSV + top-20 pair examples + per-archetype drift table + scatter plots (AI × breadth).

---

### T32. Cross-occupation employer/worker AI divergence `[Agent Q]`

**Goal:** Test whether the employer-side AI-requirement rate vs worker-side AI-usage rate relationship that T23 measured for SWE holds, weakens, or inverts when extended to SWE-adjacent and control occupation groups. Produces a cross-occupation divergence figure regardless of the direction T23 reports.

**Precondition:** T23 has produced a SWE-side employer-vs-worker AI-adoption estimate (any direction — over-specification, under-specification, or parity). If T23 could not produce a usable estimate (e.g., worker-usage benchmark unavailable), mark this task deferred and recommend analysis-phase follow-up.

**Hypothesis under test:** H_A (T24, priority 1). If the gap direction observed in T23 for SWE is universal across AI-exposed occupations, the divergence becomes a cross-occupation labor-market finding rather than a SWE specialization. If the gap direction or magnitude diverges across occupations, that heterogeneity is itself a finding.

**Steps:**
1. **Occupation group definition.** Use the existing flags in `data/unified.parquet`: `is_swe`, `is_swe_adjacent`, `is_control`. Within adjacent, sub-stratify by title regex into: data_scientist, ml_engineer, data_engineer, security_engineer, devops_engineer (if distinguishable from SWE). Within control, sub-stratify by title regex into: accountant, nurse, civil_engineer, mechanical_engineer, electrical_engineer, financial_analyst.
2. **Employer AI-requirement rate.** For each subgroup × period, compute ai_strict and ai_broad binary share using V1-validated patterns. Report n_group × period. Flag subgroups where n < 500 per period as thin (report but do not cite as lead).
3. **External worker-AI-usage benchmarks.** Fetch where accessible:
   - SWE / data / ML: Stack Overflow Developer Survey 2024 (62% currently using; data professionals ~68%).
   - General professional: Anthropic 2025 Economic Index (occupation-level task-coverage percentages).
   - Accountants: Deloitte/PwC/KPMG 2024 finance surveys (~22-30% using AI at work).
   - Nurses: AMA or Kaiser 2024 surveys (~8-15%).
   - Civil/mechanical/electrical engineers: Engineering.com or ASME 2024 surveys (~15-25%).
   - Financial analysts: CFA Institute or Bloomberg 2024 surveys.
   Document each benchmark source + date + methodology. If a benchmark is unavailable, report the subgroup as "employer-only" and exclude from the cross-occupation comparison.
4. **Per-subgroup divergence.** Compute employer_rate − worker_benchmark_midpoint for each subgroup with a benchmark. Report under 50% / 65% / 75% / 85% worker-usage assumption bands (the T23 sensitivity protocol).
5. **Direction universality test.** Is the gap direction (employer < worker) universal across subgroups? Does gap magnitude correlate with occupation AI-exposure score? If exposure scores from Anthropic's Economic Index are accessible, compute Spearman correlation(AI-exposure-score, gap-magnitude).
6. **Alternative framing check.** Benchmarks like "tried AI ever" vs "daily workflow use" may be definitionally different from "employer requires AI in JD" (see T23's sensitivity protocol). Re-run with "daily use" benchmarks where available. Report direction under both framings.
7. **Cross-occupation divergence chart.** Single-page figure: x-axis subgroup, y-axis rate, two series (employer requirement, worker benchmark), error bars for benchmark uncertainty. This is a potential paper figure.

**Essential sensitivities:** (a) aggregator exclusion, (f) within-2024 calibration (cross-occupation rate comparisons need the within-2024 baseline per occupation group to distinguish signal from instrument noise), (g) SWE classification tier.

**Output:** `exploration/reports/T32.md` + cross-occupation divergence chart + per-subgroup gap table + benchmark-source table + direction-universality verdict.

---

### T33. Requirement-section change as hiring-bar signal `[Agent R]`

**Goal:** Test whether the 2024→2026 change in requirements-section size (whatever direction T13 and T18 find — contraction, expansion, or stability) correlates with implicit hiring-bar shifts (YOE ask, credential stack depth, tech count, education level). If contraction correlates with lower credential asks, that supports a hidden-hiring-bar-lowering interpretation; if expansion correlates with higher asks, that supports hiring-bar raising; if the correlations are null, section size is a narrative-reallocation artifact independent of credential requirements.

**Precondition:** T13 has produced a section classifier and T18 has produced a cross-occupation DiD comparing SWE vs control on section-share change. If either is missing, document the gap and proceed with whatever is available.

**Hypothesis under test:** H_B (T24, priority 2). Connects the section-restructuring substantive finding to a policy-relevant interpretation (hiring-bar change). Counter-interpretation: any shrinkage is proportional not absolute and reflects narrative expansion rather than real requirement relaxation.

**Steps:**
1. **Full-corpus section classification.** Load T13's section classifier (`exploration/scripts/T13_section_classifier.py`) and apply to the full SWE corpus (not the Wave 2 T13 sample). For each posting, record `req_section_chars`, `req_section_share`, and chars per other section (responsibilities, role_summary, preferred, benefits, about_company, legal, unclassified).
2. **Period-effect regression.** Fit: `req_section_share ~ period + seniority_final + archetype + is_aggregator + log(desc_length) + period × seniority_final + period × archetype`. Report period-effect coefficient, confidence interval, and marginal effect at mean of covariates. Compare to T13's raw pooled→scraped Δ (should attenuate if composition-driven; persist if real content change).
3. **Hiring-bar proxy correlations (the key test).** Within the 2026 scraped SWE corpus, compute correlation of `req_section_share` with:
   - `yoe_min_years_llm` (within J3 and within S4 as primary; J2/S1 as label sensitivities per T30) — does shrinkage correlate with lower YOE asks?
   - `credential_stack_depth` (T11) — fewer credential types?
   - `tech_count` — fewer technologies listed?
   - `education_level` (ordinal) — fewer degree asks?
   
   If req_section shrinkage correlates with lower YOE, lower stack depth, lower tech count, AND lower education level, the hiring-bar-lowering hypothesis is strongly supported. If the correlations are near zero, the shrinkage is purely narrative reallocation.
4. **Within-company cross-metric test.** On the arshkon∩scraped overlap panel: do companies with the largest req-section contraction also have the largest J3 (yoe≤2) rise within their own postings? Compute per-company correlation of Δ(req_section_share) with Δ(J3_share). Negative correlation supports hiring-bar lowering at the company level.
5. **Narrative-content semantic test.** Sample 50 postings from the 2026 scraped SWE pool with the LARGEST req-section contraction (relative to the company-title's 2024 mean). Read the narrative sections (responsibilities + role_summary + about_company). Classify each sampled posting into: (a) genuine technical-requirement migration into responsibilities, (b) pure culture/benefits expansion with no added requirements, (c) substantive requirement loosening ("no degree required," "self-taught OK"), (d) something else. Report fractions.
5a. **Alternative-explanation check.** Compute correlation of Δ(req_section_share) with Δ(desc_cleaned_length) at the posting level. If strongly positive, the shrinkage is proportional/relative, not absolute, and the "hiring bar lowering" framing weakens to "narrative expansion dominating."
6. **Verdict.** Hypothesis supported / partially supported / rejected. Be explicit about uncertainty.

**Essential sensitivities:** (a) aggregator exclusion, (c) T30 panel, (d) description text source — run the regression on both `text_source='llm'` subset and the full corpus with raw `description`, (f) within-2024 calibration — the period effect in the regression should be contextualized by the arshkon-vs-asaniczka baseline on `req_section_share`.

**Output:** `exploration/reports/T33.md` + regression output + hiring-bar correlation table + within-company scatter + narrative-content classification of 50 samples + verdict.

---

### T34. Emergent senior-role archetype profiling `[Agent R]`

**Goal:** Profile any senior sub-archetype that T21 identifies as a candidate emergent role (disproportionately 2026-weighted, elevated on two or more of {mgmt density, orch density, strat density, AI mention}, distinct from other senior clusters in title distribution or company concentration). Test whether it is a genuine emergent role with coherent title / company / content structure, or a clustering artifact. Let the content determine the role's name; do not presuppose a specific label.

**Precondition:** T21 has produced senior sub-clusters with assignments and per-cluster feature profiles. At least one cluster meets the candidate-emergent-role criteria (n ≥ 500 combined 2024+2026; disproportionately 2026-weighted; elevated on ≥ 2 of the feature dimensions above). If no cluster meets these criteria, document in the report and recommend analysis-phase follow-up (alternative clustering hyperparameters, alternative feature spaces); do not execute steps 2-8.

**Hypothesis under test:** H_C (T24, priority 3). If the candidate cluster is a real emergent role, its title distribution, company concentration, and content profile should differ meaningfully from other senior clusters.

**Steps (execute only if precondition is met):**
1. **Load T21 cluster assignments.** From `exploration/tables/T21/cluster_assignments.csv`. Identify the candidate cluster (call it cluster_k); confirm n_2024, n_2026, and per-cluster feature means.
2. **Title distribution within cluster_k.** Tokenize titles via regex. Compute share of the cluster's titles across common senior variants: `staff engineer`, `tech lead`, `principal engineer`, `senior engineer`, `ML lead`, `AI engineer`, `engineering manager`, `engineering director`, `architect`, `other`. Rank titles by frequency.
3. **Company concentration within cluster_k.** Gini coefficient, HHI, top-20 share of `company_name_canonical`. Compare to company concentration in the other senior clusters T21 produced (use T21's cluster names, not pre-specified labels).
4. **Archetype cross-tab.** For each cluster_k posting, look up its T09 archetype label (project via nearest-centroid from `swe_embeddings.npy` where not in the T09 sample). Which archetypes disproportionately contribute to cluster_k?
5. **Company-trajectory test.** Cross-reference with T16 company clusters: for each T16 cluster, what fraction of its 2026 senior postings fall in cluster_k? Is cluster_k concentrated in one or two T16 clusters (diagnostic for whether the role is organic to a subset of employers)?
6. **Profile attributes.** Within cluster_k, median `yoe_min_years_llm`, `company_industry` distribution (top 10 where available), metro distribution (top 10 from `metro_area`), `seniority_final` distribution.
7. **Comparative profile.** Side-by-side table: cluster_k vs the other T21 senior clusters. Which features most distinguish cluster_k?
8. **Content exemplars + name proposal.** Sample 20 cluster_k postings. Read titles + first 400 chars of cleaned description. Identify 2-3 recurring phrases / asks that define the role. Based on the content, propose a descriptive role name grounded in what the postings actually say (not a pre-committed label). Alternative names if warranted.

**Essential sensitivities:** (a) aggregator exclusion.

**Output:** `exploration/reports/T34.md` + title distribution + company concentration + archetype cross-tab + cluster comparison table + 20 content exemplars + recommended role name.

---

### T35. Technology ecosystem crystallization `[Agent S]`

**Goal:** Test whether technology co-occurrence networks crystallized between 2024 and 2026. Extends T14's LLM-vendor-cluster analysis (the phi>0.15 LLM/AI co-occurrence cluster that T14 characterizes) by measuring ecosystem formation across the full 107-tech taxonomy.

**Hypothesis under test:** H_K. The LLM-vendor cluster may be one example of a broader "AI-era ecosystem crystallization" pattern; other stacks (observability, data engineering, DevOps) may also be forming more coherent co-occurrence neighborhoods.

**Steps:**
1. **Period-split tech matrix.** Load `swe_tech_matrix.parquet` (full SWE corpus × 107 technology taxonomy). Split into pooled-2024 and scraped-2026 panels.
2. **Per-period co-occurrence networks.** Compute phi coefficient for all tech pairs separately in 2024 and 2026. Threshold at phi > 0.15 (matching T14). Build two networkx graphs.
3. **Louvain community detection.** Apply `python-louvain` or `networkx.algorithms.community.louvain_communities` separately to each graph. **Louvain is randomized — fix `random_state=42` as the primary run AND report modularity stability across 10 re-runs with different seeds (mean, SD, range).** If modularity SD exceeds 0.05 or community-count range exceeds 2, the assignment is unstable and conclusions about "coalesced / new / fragmented" must be downweighted. Record per period: community count, modularity score (primary seed + 10-run mean/SD), mean community size, max community size, isolate count (nodes in own community).
4. **Backward-stability classification.** For each 2026 community: compute Jaccard of membership with each 2024 community. Classify:
   - **Stable** (Jaccard ≥ 0.70 with a 2024 counterpart) — community persisted.
   - **Coalesced** (2026 members were scattered across 2+ 2024 communities, each contributing <30% of 2026 members) — ecosystem formed.
   - **New** (≥50% of 2026 members didn't exceed the phi threshold in 2024) — new tech arrivals clustered.
   - **Fragmented** (a 2024 community split into 2+ 2026 communities) — ecosystem broke apart.
5. **LLM-vendor cluster verification.** Confirm T14's LLM-vendor cluster (expected to include items like claude, copilot, cursor, chatgpt, gpt, langchain, rag, openai, llm, vector_database, huggingface, fine_tuning, prompt_engineering). Is it "coalesced" or "new"? Which technologies were NOT in a community at phi>0.15 in 2024 but ARE in this community in 2026?
6. **Other crystallizations.** Identify all "coalesced" and "new" communities. Name each by inspecting its member technologies:
   - Expected candidates: observability stack (Datadog + New Relic + PagerDuty + Grafana); data-engineering stack (Snowflake + dbt + Airflow + Kafka); DevOps/platform (Terraform + Kubernetes + Helm + ArgoCD + CircleCI).
   - For each, report member techs, community size, modularity contribution, and Jaccard with nearest 2024 counterpart.
7. **Modularity Δ.** Compare 2024 modularity to 2026 modularity. If modularity ROSE, the network is more cleanly clustered in 2026 (ecosystem crystallization). If FELL, more blurred.
8. **Visualization.** Two-panel graph layout: 2024 on left, 2026 on right, same layout algorithm (spring layout or Fruchterman-Reingold), nodes colored by 2026 community (same colors across panels so the reader can see which 2024 nodes ended up in which 2026 community).
9. **Domain cross-check.** Project each 2026 community onto T09 archetype concentrations: which archetypes disproportionately use the LLM-vendor community vs the data-engineering community? Does ecosystem membership align with archetype membership?

**Essential sensitivities:** (a) aggregator exclusion, (b) cap at 50 postings per company before computing phi (co-occurrence is sensitive to prolific posters).

**Output:** `exploration/reports/T35.md` + modularity Δ table + community classification table (stable / coalesced / new / fragmented) + named ecosystem list + side-by-side network visualization + archetype-community cross-tab.

---

### T36. Legacy-stack substitution map `[Agent S]`

**Goal:** For each 2024 SWE title that disappeared in 2026 (per T10), find the nearest 2026 descriptive neighbor. Produces a "role substitution map" that strengthens the paper's legacy-stack-consolidation narrative.

**Hypothesis under test:** H_L. T10 listed disappearing titles (Java architect, Drupal, PHP, .NET, DevOps architect) but didn't map what replaced them. The substitution pattern tells us whether legacy roles are being rebranded (same content, new name), replaced by cross-stack generalists, or absorbed into AI-enabled roles.

**Steps:**
1. **Disappearing-title list.** Load T10's top 20 disappearing titles (by arshkon volume, confirmed absent from scraped 2026 with threshold ≥10 arshkon postings and <3 scraped postings). Include the Wave 2 finding's headline list: `java architect`, `drupal developer`, `devops architect`, `senior php developer`, `sr. .net developer`, and expand to the full top-20.
2. **Per-title centroid.** For each disappearing title, collect all arshkon SWE postings with that title (where `text_source='llm'`). Compute TF-IDF centroid on cleaned descriptions using shared vocab from `swe_cleaned_text.parquet`.
3. **2026 title universe.** Enumerate 2026 titles with ≥10 scraped postings (top ~500-1000 titles). For each, compute its cleaned-description TF-IDF centroid.
4. **Substitution search.** For each disappearing title, compute cosine similarity to every 2026 title centroid. Record the top-5 nearest 2026 neighbors per disappearing title with cosine, posting volume, and mean `seniority_final`.
5. **Substitution table.** Single table: disappearing_title | arshkon_n | top1_2026_neighbor | cosine | 2026_n | neighbor_seniority_shift. For each row, note whether the substitution is (a) same seniority level (e.g., senior PHP developer → senior backend engineer), (b) upward (entry DevOps → senior platform engineer), (c) downward, (d) title consolidation (legacy specialist → cross-stack generalist).
6. **Content drift per pair.** For each disappearing title and its top-1 2026 neighbor, run Fightin' Words (log-odds with informative Dirichlet prior) between the two corpora. Report top-20 terms favoring the disappearing corpus (what's leaving) and top-20 favoring the 2026 neighbor (what's arriving). Tag with semantic categories (legacy_tech / modern_tech / ai_tool / scope / mgmt / methodology).
7. **AI-vocabulary comparison.** For each substitution pair, compute: ai_strict binary share in disappearing corpus vs in 2026 neighbor corpus. If 2026 neighbors have systematically higher AI-mention rates, the "legacy consolidation → AI-enabled roles" interpretation strengthens.
8. **Manual inspection.** Read 10 disappearing-title postings and 10 top-1-neighbor postings side-by-side. Is the substitution credible? Does the content genuinely map, or is it a loose cosine match?

**Essential sensitivities:** (a) aggregator exclusion, (b) company capping (TF-IDF centroid computation is corpus-aggregate; cap at 20-50 per `company_name_canonical` before computing title centroids to prevent prolific employers from dominating).

**Output:** `exploration/reports/T36.md` + substitution table (20 disappearing × top-5 neighbors) + content-drift summary per pair + AI-vocabulary comparison + manual-inspection notes.

---

### T37. Sampling-frame returning-companies sensitivity `[Agent T]`

**Goal:** Quantify how much of each anticipated Gate-3 headline is a sampling-frame artifact vs a genuine longitudinal signal. T06 typically finds that most scraped companies are new entrants to the panel, and T28 may find that archetype growth is disproportionately new-entrant-driven. T37 restricts analysis to the returning-companies cohort and re-runs each headline on that restricted sample.

**Hypothesis under test:** H_H (T24, priority 6). Paper-defensibility: without this restriction-test, every longitudinal claim carries an implicit "which companies are posting" confound.

**Steps:**
1. **Returning-company cohort.** Load the returning-companies cohort from `exploration/artifacts/shared/returning_companies_cohort.csv` (produced by T06 — companies with SWE presence in both 2024 sources and scraped 2026). Recompute from `data/unified.parquet` joining on `company_name_canonical` only if T06's artifact is missing.
2. **Headline list.** Load the orchestrator's post-Wave-3 draft ranking (in the Gate 2 memo for Wave 2 headlines; in the Gate 3 draft for Wave 3 headlines) and select the top 5-10 headline metrics the paper is most likely to cite. These will span prevalence metrics (e.g., keyword binary shares), composite scores (e.g., length-residualized breadth), section-share metrics, seniority-stratified rates under the T30 panel, and archetype-stratified metrics. The specific list depends on what Wave 2-3 actually found; do not hardcode it from this task spec. Report the exact headline set you chose and cite the orchestrator memo that sourced each.
3. **Restrict-and-recompute.** For each headline, recompute the 2024→2026 Δ on the returning-cohort-only subset. For seniority-stratified headlines, apply the T30 panel primary (J3 for junior, S4 for senior) plus the label-based sensitivities.
4. **Retention ratio.** For each headline, compute: returning-cohort Δ / full-corpus Δ. If > 0.80, the headline survives sampling-frame restriction. If 0.50-0.80, partially robust. If < 0.50, sampling-frame-driven. **Materiality threshold:** the ratio is only interpretable when the full-corpus |Δ| is large enough that small-sample noise in the returning cohort does not dominate. Require |full-corpus Δ| ≥ 1pp (for share metrics) / ≥ 0.2 SD (for composite scores) before citing a ratio. When the full-corpus Δ falls below this threshold, mark the ratio "undefined — full-corpus Δ too small" and report the absolute returning-cohort Δ and its 95% CI side-by-side instead.
5. **Verdict per headline.** Classify as: robust / partially-robust / sampling-frame-driven.
6. **Cross-check with T16.** If T16 reports a within-company AI-strict rewriting estimate on the arshkon∩scraped overlap panel, T37 extends that estimate to the returning cohort. Compare the two within-company estimates; agreement within the metric's within-2024 calibration noise floor (from the Prep calibration table) confirms consistency. If they diverge by more than that, investigate.
7. **Sensitivity table.** Produce a one-page table for the paper's robustness appendix: metric | full-corpus Δ | returning-cohort Δ | retention ratio | verdict. Bold the lead-paper headlines.
8. **Implication statement.** Write 2-3 paragraphs interpreting the results: which claims are defensible even under the strictest sampling-frame restriction, and which need to be qualified in the paper text.

**Essential sensitivities:** (c) T30 panel — verify the returning-cohort sample is not too thin for panel-variant MDEs.

**Output:** `exploration/reports/T37.md` + sampling-frame sensitivity table + per-headline retention ratio + verdict + paper-appendix-ready text block.

---

### T38. Hiring-selectivity × scope-broadening correlation `[Agent T]`

**Goal:** Test whether the 2024-to-2026 scope-broadening pattern is partly a selectivity response to the JOLTS hiring slowdown (2026 Info-sector openings 0.66× of 2023 average). If companies with the largest posting-volume contraction are also writing the broadest JDs, "scope broadening" is partly "filter-raising under hiring constraint."

**Hypothesis under test:** H_N. Novel — not in T24. Explains a possible macro-mediated mechanism for the within-company scope rise that T16 documented.

**Steps:**
1. **Panel.** Start from the arshkon∩scraped overlap panel (T16). Compute per company:
   - `posting_volume_2024` = arshkon SWE posting count / (arshkon window days)
   - `posting_volume_2026` = scraped SWE posting count / (scraped window days as of cutoff)
   - `posting_volume_log_ratio` = log(posting_volume_2026) − log(posting_volume_2024). This is the per-company daily-rate change.
2. **Content Δ metrics per company.** For each company, compute 2024 → 2026 change in:
   - `breadth_resid_delta` (length-residualized `requirement_breadth`)
   - `ai_strict_delta` (V1-validated pattern)
   - `mentor_rate_delta` (V1-validated strict mentor pattern, on S1 postings only)
   - `desc_len_delta` (median cleaned description length)
   - `yoe_min_years_llm_delta` (median YOE within the company's SWE postings, LLM frame; `yoe_extracted_delta` as rule ablation)
3. **Correlation matrix.** Pearson AND Spearman of `posting_volume_log_ratio` with each content Δ. Report with 95% CI. Direction predictions:
   - If hypothesis holds: negative correlation (volume ↓ + breadth ↑, volume ↓ + yoe ↑).
   - If null: no correlation.
   - If positive: volume-up companies are scope-expanding (reverse of selectivity prediction).
4. **Stratification by company size.** Using `company_size` where available (arshkon companion data). Re-run correlation within large (≥10K employees), mid (1K-10K), and small companies. Does the selectivity response differ by firm size?
5. **Stratification by archetype.** Using T09 archetype labels (project via nearest-centroid). Are specific archetypes (legacy / consulting / tech-giant / ML-AI) carrying the signal?
6. **Robustness — exclude tech giants.** Tech giants (Google, Amazon, Microsoft, AWS, Apple, Meta — mapped from whichever T16 cluster captures "consolidating giants") have idiosyncratic volume patterns. Re-run correlations on the mid-market subset (full panel minus the tech-giant rows). Does the correlation direction hold?
7. **Robustness — exclude aggregators.** Aggregator volume is driven by ad spend, not hiring. Re-run on non-aggregator subset.
8. **Interpretation.** If correlations are negative and significant after robustness checks, the hiring-selectivity hypothesis is supported and the paper should acknowledge macro-mediation of the scope-broadening finding. If null or positive, scope-broadening is NOT a selectivity response and the paper can claim it as a demand-side content shift independent of volume.

**Essential sensitivities:** (a) aggregator exclusion (central to this task), (c) T30 panel for seniority-stratified sub-analysis, (f) within-2024 calibration — the volume-vs-content correlation magnitude should be contextualized by the within-2024 baseline (does the same correlation hold arshkon-vs-asaniczka?).

**Output:** `exploration/reports/T38.md` + correlation table (Pearson + Spearman with CI) + per-size-class stratification + per-archetype stratification + robustness re-runs + interpretation paragraph.

---

### Wave 4 — Integration & Hypothesis Generation

---

### T24. Hypothesis consolidation and analysis-phase priorities `[Agent N]`

**Goal:** Consolidate the exploration's hypothesis landscape after all computational waves (1, 2, 3, 3.5) and both verification gates (V1, V2). For hypotheses that Wave 3.5 directly tested, report verdicts; for hypotheses that emerge from the full body of evidence but were not tested, write new specifications; for hypotheses that remain analysis-phase deferred, assign priority. T24's output is the input to T26 SYNTHESIS.md and to the analysis plan's hypothesis-test pre-registration.

The Wave 3.5 layer changes T24 from "generate new hypotheses from scratch" to "consolidate tested + untested hypotheses into a priority-ranked analysis-phase roadmap." Most high-priority induced hypotheses (H_A, H_B, H_C, H_H, H_M, plus new H_K, H_L, H_N) will have already been tested in Wave 3.5 — T24 reports those verdicts and reasons about what remains.

**Steps:**
1. Read all `exploration/reports/T*.md` (T01-T38), both verification reports (V1_verification.md, V2_verification.md), and all gate memos (gate_0_pre_exploration.md, gate_1.md, gate_2.md, gate_3.md).
2. **Confirmation inventory across the full pipeline.** For each RQ1-RQ4 claim and each sub-claim, report:
   - Verdict (supported / contradicted / ambiguous / inverted / decomposed)
   - Evidence cites (specific task numbers, including Wave 3.5 where applicable)
   - Whether V1 or V2 verification confirmed or corrected the magnitude
3. **Wave 3.5 verdict table.** For each task T31-T38, a row with:
   - Hypothesis tested (H_M, H_A, H_B, H_C, H_K, H_L, H_H, H_N for T31-T38 respectively)
   - One-sentence verdict (supported / partially supported / rejected / null)
   - Headline numeric result (effect size, CI or robustness range, p-value where applicable)
   - How this Wave 3.5 verdict changes or strengthens the SYNTHESIS.md ranked-findings list
4. **Surprise inventory.** Across all tasks, which findings were unexpected or contradicted prior assumptions? Separately flag surprises from Wave 3.5 (often these are results on tests that Gate 2/3 predicted but that turned out differently).
5. **New-hypothesis generation (post-Wave-3.5 gap).** Based on the full body of evidence including Wave 3.5 results, propose any NEW testable hypotheses that are NOT already covered by: RQ1-RQ4, Wave 3.5's 8 tasks, or the T24 H_A-H_J planned list. Fewer new hypotheses are expected at this stage because Wave 3.5 pre-tested the highest-priority induced hypotheses. For each new hypothesis: precise statement, supporting evidence, proposed analysis-phase test, novelty/publishability.
6. **Deferred-hypothesis inventory.** Hypotheses from the T24 H_A-H_J planned list that were NOT directly tested in Wave 3.5 (specifically H_D senior IC-as-team-multiplier, H_E same-co J1 drop + J3 rise regime shift, H_F Sunbelt AI surge catchup, H_G staff-title redistribution, H_I AI as coordination signal, H_J recruiter-LLM senior bias) should be listed here with:
   - Why Wave 3.5 did not cover them (scope, data needs, lower priority)
   - Analysis-phase priority ranking
   - What data or method would be required to test them
7. **Method-suitability assessment.** For each confirmed or partially-confirmed claim, what analytical methods are best suited for the analysis phase (accounting for sample sizes, confounds, measurement quality surfaced in Waves 1 through 3.5)?
8. **Key tensions.** List the 5 most important tensions or puzzles that the analysis phase must resolve. The post-Wave-3.5 tension list differs from the Gate 3 tension list — Wave 3.5 resolves some (e.g., the T16 within-company vs T06 between-company tension is clarified by T31's pair-level drift) and introduces others. List the current set.
9. **Data gaps.** What data would we need to answer the most interesting questions we still cannot answer? Distinguish "gaps that Wave 3.5 surfaced" from "pre-existing gaps."

**Output:** `exploration/reports/T24.md` — the hypothesis consolidation document. Structured so T26 SYNTHESIS.md can pull section by section without reworking.

### T25. Interview elicitation artifacts `[Agent N]`

**Goal:** Produce 5-7 artifacts for RQ4 data-prompted interviews, drawing on the full body of exploration findings including Wave 3.5.

**Steps (reads all prior reports including Wave 3.5):**
1. **Inflated junior JDs:** From T11/T22, select 3-5 entry-level postings with the most extreme scope-inflated or ghost-like requirements. Query parquet for actual text.
2. **Paired JDs over time:** From T16/T31, select 3-5 same-company (and where possible same-title) pairs (2024 vs 2026). If T31 produced a "top-20 drift pairs" list, use those. Format side-by-side.
3. **Junior-share trend plot:** From T08, annotated with AI model release dates (GPT-4: Mar 2023, Claude 3: Mar 2024, GPT-4o: May 2024, Claude 3.5 Sonnet: Jun 2024, o1: Sep 2024, DeepSeek V3: Dec 2024, Claude 3.5 MAX: Feb 2025, GPT-4.5: Feb 2025, Claude 3.6 Sonnet: Apr 2025, Claude 4 Opus: Sep 2025, Claude 4.5 Haiku: Oct 2025, Gemini 2.5 Pro: Mar 2026).
4. **Senior archetype chart:** From T21 and T34 (emergent senior-role profile, whatever T34 named it), management vs orchestration vs strategic language profiles (2024 vs 2026). Include exemplar postings from the emergent cluster if T34's precondition was met.
5. **Posting-usage divergence chart:** From T23 (SWE) and T32 (cross-occupation extension). If T32 confirmed the inversion cross-occupation, present as a multi-occupation chart.
6. **Hidden hiring-bar exemplars:** From T33, select 3-5 postings with the largest requirements-section contraction paired with lowered YOE / credential asks. Format as before/after company snapshots.
7. **Bonus — striking discoveries from T24's surprise inventory** that would be valuable to present to interviewees for reaction.

**Output:** `exploration/artifacts/T25_interview/` with each artifact as PNG or markdown + a README that maps each artifact to the interview question it probes.

### T26. Exploration synthesis `[Agent N]`

**Goal:** Consolidate everything from Waves 1 through 3.5 (plus both verification reports and all gate memos) into a single handoff document for the analysis phase. SYNTHESIS.md is the one document the analysis agent reads first and the one document Agent P (presentation) reads to tell the story; Wave 3.5 findings must appear as first-class claims in the ranked findings and robustness material, not as an optional appendix section.

**Steps (reads all prior reports including Wave 3.5):**
1. Read all `exploration/reports/T*.md` (T01-T38), verification reports (V1, V2), gate memos (gate_0 through gate_3), and T24's hypothesis consolidation.
2. Write `exploration/reports/SYNTHESIS.md` covering:
   - **Executive summary (≤400 words):** paper's lead sentence + 3-4 supporting claims + RQ evolution summary + method recommendations + key caveats. Wave 3.5 findings that materially sharpen the lead (for instance, a pair-level within-company drift estimate tightening a company-level rewriting claim, or a cross-occupation extension elevating an RQ3-style divergence from a SWE-specific story to a general labor-market finding) must be integrated here, not deferred to an appendix.
   - **Data quality verdict per RQ** — what analyses are safe, what need caveats. Integrate V1 + V2 findings including Wave 3.5 verifications.
   - **Recommended analytical samples** (rows, columns, filters) per analysis type. Cite the arshkon∩scraped overlap panel, the returning-companies cohort (from T06 / T37), the LLM-labeled subset, and any Wave 3.5-specific sample frames.
   - **Seniority validation summary.** Integrates T03, T30, V1's LLM-frame audit, and any Wave 3.5 pair-level results under the T30 panel.
   - **Known confounders with severity + mitigation.** Include (as applicable given what the fresh run finds): description length growth, asaniczka label gap, aggregator contamination, company composition shift, field-wide vs SWE-specific trends, LLM-frame selection artifact, recruiter-LLM mediation, macro hiring-context confounds, platform taxonomy drift. For each confounder, cite the specific Wave 2/3/3.5 task(s) that characterized it. T37's sampling-frame sensitivity table is the primary defense against composition-shift confounds; T38's selectivity correlation handles macro-hiring-context.
   - **Ranked findings** organized by evidence strength × novelty × narrative value. Wave 3.5 findings sit in the same ranked list alongside Wave 2-3 findings, not a separate section. Build the ranking empirically: start from the orchestrator's Gate 3 memo ranking, apply the evidence × novelty × narrative-value filter, and produce the final ordered list. Include for each finding: the claim sentence, evidence strength (strong / moderate / weak), sensitivity verdict (robust / partial / flagged), and the tasks that produced it. Do not pre-commit the ordering; let the evidence determine it.
   - **Discovery findings organized:** confirmed, contradicted, new discoveries, unresolved tensions. Each with evidence cite to specific task + wave.
   - **Posting archetype summary (T09 + T28 + T34).**
   - **Technology evolution summary (T14 + T35 + T36).**
   - **Geographic heterogeneity summary (T17).**
   - **Senior archetype characterization (T21 + T34).**
   - **Ghost/aspirational prevalence (T22).**
   - **Robustness appendix.** Include: the sampling-frame sensitivity table from T37 (per-headline retention ratio under full-corpus vs returning-cohort restriction); hiring-selectivity correlation from T38; aggregator-exclusion sensitivities from T06/T16; within-2024 calibration SNRs from the Prep calibration table; V1/V2 corrections summary.
   - **Hypothesis status table (from T24).** Every hypothesis (RQ1-RQ4 original + T24 H_A-H_J + Wave 3.5 H_K/H_L/H_M/H_N + any new post-3.5 hypotheses): verdict, evidence, analysis-phase action (test / revisit / archive).
   - **Method recommendations** for the analysis phase (draws from V1, V2, T24, T07).
   - **Sensitivity requirements** — which findings still need robustness checks the exploration didn't run?
   - **Interview priorities** — what should qualitative RQ4 work focus on? Integrate the mechanism-relevant Wave 3.5 and Wave 2-3 findings (ghost / aspirational patterns from T22, hiring-bar mechanism from T33, emergent senior role profiles from T34, recruiter-LLM mediation from T29) as mechanism hypotheses interviews should adjudicate.
   - **Recommended paper positioning** — hybrid dataset/methods × substantive labor paper. Lead-finding candidate and alternative positionings should be derived from the ranked findings above; do not pre-commit to a specific lead before the ranking is complete.
   - **Paper figures candidate list.** Top 5-7 figures that should make the paper, sourced from `exploration/figures/`. The figure set should span the ranked findings; specific candidates depend on what Wave 2/3/3.5 surfaces.

**Output:** `exploration/reports/SYNTHESIS.md` — the one document the analysis agent reads first and Agent P consumes for the presentation.

---

### Wave 5 — Presentation

---

### T27. Exploration findings package `[Agent P]`

**Goal:** Package the exploration's findings into a navigable, presentable artifact that works at multiple levels of depth — from a 10-minute slide presentation down to the raw task reports and gate memos. Host it and share the link.

**Inputs:** Read `exploration/reports/SYNTHESIS.md` (primary — the consolidated findings), gate memos (`exploration/memos/gate_*.md`), and `exploration/reports/INDEX.md` for the full task inventory. Reference existing figures from `exploration/figures/` and reports from `exploration/reports/`. Also read `docs/preprocessing-guide.md` (and skim `docs/preprocessing-schema.md` for the output schema) to extract a concise description of the preprocessing pipeline for the methodology layer — see "Preprocessing description" below. Do not regenerate analysis — package what exists.

**You MUST** Think critically about the findings, identify interesting facts, conclusions and present them in a simple to read way. You must review the website and all the text to identify unnecessary complexity/verbosity in the language and make it more concise. 

---

#### The three depth layers

The deliverable has three layers. A reader should be able to enter at any layer and navigate to the others.

**Layer 1 — The presentation (the story).**
A MARP markdown slide deck (~20-25 slides) that tells the story of what the exploration found. This is the entry point. It should be embeddable in the site as an interactive carousel.

Read `docs/fatahalian-clear-talks.pdf` — Kayvon Fatahalian's "Tips for Giving Clear Talks" (67 slides, 12 tips). This is the definitive guide for how the slides should be designed. The full PDF is the reference; the distillation below highlights what matters most for this presentation:

**Core philosophy:** The goal is to convey what you *learned*, not what you *did*. Tell the audience the most important things they should know but probably don't. Put smart people in the best position to help you.

**Slide design principles:**
- **Every sentence matters.** If it doesn't make the point, remove it. If the audience won't understand it, remove it. If you can't justify how it helps the listener, remove it.
- **One point per slide, and the point IS the title.** Slide titles should be complete-sentence claims, not topic labels. Reading only the titles in sequence should give a great summary of the entire talk. Not "Entry-Level Analysis" but a complete sentence that states the actual finding and its magnitude.
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
All task reports organized by wave, all gate memos, and retrospectives. This is the audit trail — a reader who wants to see exactly what analysis was done can find it here.

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

Items below still require formal statistical framework, LLM-stage outputs, or analysis-phase infrastructure beyond what the exploration waves (including Wave 3.5) can produce. Where Wave 3.5 partially addresses an item descriptively, the item is retained here because the formal/causal version still needs to be done.

- **Robustness pre-registration / specification curve** — exploration enumerated sensitivity dimensions; formal curve requires pre-registration.
- **Placebo and falsification tests** — Wave 3.5 T37 is a sampling-frame sensitivity test but not a pre-registered placebo. Analysis phase pre-registers placebos.
- **Oaxaca-Blinder decomposition** — T16 and T31 run within/between decompositions descriptively; formal O-B with standard errors remains analysis-phase.
- **Selection bias reweighting / IPSW** — T37 quantifies sampling-frame sensitivity but does not reweight. IPSW remains analysis-phase.
- **Full power analysis refinement** — T07 gave feasibility estimates; effect-size-specific refinement after Wave 3/3.5 results is analysis-phase.
- **Seniority boundary classifier** — T20 runs logistic-regression boundary classifiers; embedding-based learned classifier with proper CV is analysis-phase.
- **Company fixed-effects regression** — T16 and T31 use within/between decomposition and pair-level deltas descriptively; formal FE with clustered SEs is analysis-phase.
- **Formal break detection / event-study plots** — we have three snapshots, not a continuous series; break detection needs more dense sampling.
- **Causal timing analysis with model release windows** — T19 annotates the timeline; causal timing needs pre-registered event-study design.
- **Full embedding-based document classification** — Wave 2/3 embed for similarity/clustering; supervised classification is analysis-phase.
- **Remaining T24 hypotheses** (H_D senior IC-as-team-multiplier, H_E same-co J1/J3 regime shift, H_F Sunbelt catchup, H_G staff-title redistribution, H_I AI as coordination signal, H_J recruiter-LLM senior bias) — T24 inventories with priority. Wave 3.5 covered H_A, H_B, H_C, H_H + introduced and tested H_K, H_L, H_M, H_N.

---

## 5. Bias threat summary

Mitigations cite the tasks that characterized or reduced the bias. Wave 3.5 entries (T31, T37, T38) handle composition-shift, sampling-frame, and hiring-context threats at the pre-synthesis stage.

| Bias | Direction | Mitigation task | Residual risk |
|---|---|---|---|
| Platform selection | Favors SWE | T07 | Low for SWE |
| Scraper query design | Misses long-tail | T05 | Moderate |
| Aggregator contamination | Inflates some companies | T06, T16 | Low after flagging |
| Description length inflation | Biases raw keyword counts | Length-normalization everywhere; residualize composites (T11, T16, V1 correction) | Low after normalization |
| Company composition shift | Could drive aggregate seniority shift | T06, T16, T31 (same-co × same-title pair test), T37 (returning-cohort restriction) | Low after Wave 3/3.5 decomposition |
| Temporal selection (volatility) | Oversamples long-lived postings | T19 | Moderate |
| Kaggle provenance unknown | Unknown | T05 | High (irreducible) |
| asaniczka missing entry-level | Thin baseline | T02, T03, T30 panel | Moderate |
| Boilerplate in raw text | Noisy text analysis | Use `description_core_llm` (LLM-cleaned); raw `description` only for boilerplate-insensitive checks | Low after LLM coverage |
| Company-name contamination | Pollutes corpus comparisons | Company-name stripping in preamble | Low after stripping |
| Remote flag incomparability | 0% in 2024 (data gap, not real) | Do not interpret as real change | Low if noted |
| SWE classification temporal instability | Could change SWE sample composition | T04 | Moderate |
| Within-2024 cross-source variation | Could inflate 2024-to-2026 effect sizes | Within-2024 calibration in T05, T08, Prep calibration table | Low after calibration |
| Instrument difference (Kaggle unformatted vs scraped markdown) | Inflates text-based 2024-to-2026 differences | Within-2024 calibration (sensitivity dim f); Wave 3.5 T35 re-tests co-occurrence structure per period separately | Moderate |
| SWE classification tier uncertainty | 9-10% from elevated-FP tier could shift sample composition | Sensitivity dim g in T09, T14, T15, T18 | Low after check |
| Sampling-frame artifact (most scraped cos are new entrants) | Longitudinal claims conflate firm population shift with content change | T06, T16 (within-co decomp); T37 (returning-cohort sensitivity on anticipated Gate-3 headlines); T31 (same-co × same-title pair drift) | Low after Wave 3.5 sensitivity table |
| Keyword pattern precision | Broad patterns inflate effect sizes (e.g., `manage`, `agent`) | V1 semantic precision refinement; T22 validated_mgmt_patterns.json; Wave 3.5 agents load validated patterns | Low after refinement |
| LLM-frame selection artifact | Scraped `llm_extraction_coverage='labeled'` subset is non-random w.r.t. junior signal | V1 flagged; Wave 3.5 MUST requires agents to flag the artifact when restricting text-sensitive analyses to labeled rows, and to report labeled-row coverage by period and source per sensitivity dim (h) | Moderate (analysis-phase IPSW) |
| Recruiter-LLM mediation of JD content | Apparent content shifts partly driven by recruiter tooling, not employer demand | T29 authorship score + low-score subset test | Moderate (analysis-phase needs labeled-authorship calibration set) |
| 2026 macro hiring slowdown | JOLTS Info-sector 0.66× of 2023 avg confounds volume-based claims | T07 JOLTS contextualization; Wave 3.5 T38 tests selectivity-scope interaction | Low for share-based metrics, moderate for volume claims |

---

## 6. Discovery principles

This exploration is designed around discovery, not confirmation. Key principles:

1. **Let the data speak first.** Unsupervised methods (clustering, dimensionality reduction, co-occurrence networks) before hypothesis-testing methods.
2. **Report surprises.** Every task report must include unexpected findings, even if they don't map to RQ1-RQ4.
3. **Calibrate everything.** Within-2024 (arshkon vs asaniczka) comparisons establish how much variation is "normal." 2024-to-2026 changes are only interesting if they exceed this baseline.
4. **Normalize for length.** Description length grew substantially between periods and is the single biggest confound. Every text metric needs length normalization.
5. **Decompose aggregate changes.** Company composition, seniority composition, geographic composition can all drive aggregate changes. Always check within-stratum patterns.
6. **Map boundaries, not just levels.** The boundaries between seniority levels, between SWE and adjacent roles, between AI-mentioning and non-AI-mentioning postings may be more informative than the levels themselves.
7. **Generate, don't just test.** The exploration succeeds if it produces NEW research questions, not just if it confirms old ones.
8. **Follow the evidence, not the hypothesis.** The initial RQ1-RQ4 framing may turn out to be wrong. If scope inflation is an artifact of aggregator composition, if the senior archetype shift is stronger than the junior story, if the most interesting pattern is something we never anticipated — follow it. The paper's narrative should emerge from the data, not be imposed on it.
