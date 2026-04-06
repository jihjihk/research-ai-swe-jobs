# Exploration Task Reference

Date: 2026-04-05
Input: `data/unified.parquet`
Schema reference: `docs/preprocessing-schema.md` (canonical), `docs/schema-stage8-and-stage12.md` (historical)

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
- Asaniczka has ZERO native entry-level seniority labels. Its `seniority_final` entry rate (~0.6%) is an imputation artifact.
- Remote work flags are 0% in 2024 sources (data artifact). Do not interpret as a real change.
- LLM classification columns: `seniority_llm` (uniform explicit-signal-only seniority), `ghost_assessment_llm` (richer ghost detection than rule-based), `swe_classification_llm`, `yoe_min_years_llm`. Check `llm_classification_coverage` for coverage. Use `seniority_llm` as the primary seniority variable with `seniority_native`/`seniority_final`/`seniority_imputed` as ablation variants. Use `ghost_assessment_llm` as primary ghost indicator with `ghost_job_risk` as fallback.
- 31GB RAM limit — use DuckDB or pyarrow for queries, never load full parquet into pandas.

**Entry-level metric rule (CRITICAL — the direction of the entry-share trend depends on this):**
For ANY analysis involving entry-level share, counts, or entry-level-specific comparisons: pooling asaniczka into "2024" with columns that lack entry-level labels (like `seniority_final` or `seniority_3level`) makes the 2024 entry rate artificially near-zero and CAN FLIP the direction of the entry-share trend. You MUST:
1. Report arshkon-only as the primary 2024 baseline for entry-level metrics.
2. If you pool asaniczka, report the arshkon-only result alongside and flag any directional discrepancy.
3. If `seniority_llm` is available, it resolves this problem by applying consistent labels across all sources — use it.

**Seniority ablation framework:** When reporting entry-level trends, present results under multiple seniority operationalizations:
- `seniority_llm` — uniform LLM method, **primary**. Applies the same explicit-signal-only classification across all sources including asaniczka.
- `seniority_native` — platform-provided labels, high quality where available, but excludes asaniczka entry
- `seniority_final` — combined rule + native, best coverage but mixes methods across sources
- `seniority_imputed` (where != unknown) — rule-based only, uniform method but thin coverage and low entry recall
Agreement across operationalizations strengthens findings; disagreement is a critical flag.

**Description text quality — critical for text-based analyses:**
`description_core_llm` (LLM-based boilerplate removal) is the primary text column. `description_core` (rule-based, ~44% accuracy) retains substantial boilerplate garbage and should only be used as a sensitivity check. Check `llm_extraction_coverage` to confirm coverage by source — it should be available for all sources after LLM budget allocation.

Text column rules:
- **Text-dependent analysis** (embeddings, topic models, requirement extraction, corpus comparison, density metrics): Use `description_core_llm`. Do NOT use `description_core` as primary for any boilerplate-sensitive analysis. Report `text_source` distribution.
- **Binary keyword presence** (does the posting mention X anywhere?): Raw `description` is acceptable for recall. But density metrics (mentions per 1K chars) must use cleaned text.
- **Non-text analyses** (seniority counts, company analysis, geographic patterns): Use all rows regardless of text column.
- **Sensitivity check:** Run text-dependent findings on both `description_core_llm` and `description_core` to quantify the boilerplate effect.

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

## Sensitivity framework — apply to all analytical tasks

Every analytical task must report results under its primary specification AND under essential sensitivity checks. A finding is only robust if it survives its essential sensitivities.

Eight dimensions (referenced by letter in each task spec):

(a) **Aggregator exclusion.** Primary: include all rows. Alt: exclude `is_aggregator = true`. Rationale: aggregators have systematically different descriptions, seniority patterns, and template-driven requirements.

(b) **Company capping.** Primary: uncapped. Alt: max 20 postings per `company_name_canonical`. Rationale: a few prolific posters could dominate uncapped results.

(c) **Seniority operationalization.** Present results under the seniority ablation framework defined in the core preamble: `seniority_llm` (if available), `seniority_native`, `seniority_final`, `seniority_imputed` (where != unknown). For entry-level analyses, arshkon-only is the primary 2024 baseline unless `seniority_llm` is available.

(d) **Description text source.** Primary: `description_core_llm` (where available). Alt: `description_core` (rule-based, all sources). Further alt: raw `description`. Rationale: `description_core` retains substantial boilerplate; findings may differ with cleaner text.

(e) **Source restriction.** Primary: arshkon (2024-04) vs scraped (2026-03). Alt: arshkon + asaniczka pooled as 2024 baseline. Rationale: asaniczka is a different instrument; pooling increases power but introduces noise.

(f) **Within-2024 calibration (signal-to-noise).** Mandatory diagnostic. For every metric compared 2024-to-2026, also compute the arshkon-vs-asaniczka difference on the same metric. Signal-to-noise ratio: (cross-period effect size) / (within-2024 effect size). If ratio < 2, flag as "not clearly above instrument noise."

(g) **SWE classification tier.** Primary: all `is_swe = true`. Alt: exclude `swe_classification_tier = 'title_lookup_llm'` (retaining regex + embedding_high only). Rationale: title_lookup_llm has elevated false-positive rate.

(h) **LLM text coverage.** For text analyses: primary uses `description_core_llm`. Alt: uses `description_core` (lower text quality). Report both and note differences.

(i) **Indeed cross-platform validation.** For key findings (entry share, AI prevalence, description length), compute the same metric on Indeed scraped data. Indeed has no native seniority but has `seniority_imputed`. If Indeed patterns match LinkedIn, findings are more robust. If they diverge, the finding may be LinkedIn-specific.

**Materiality threshold:** A finding is **materially sensitive** to a dimension if the alternative specification changes the main effect size by >30% or flips the direction. **When a finding IS materially sensitive, investigate WHY.** Report what the alternative specification reveals about the mechanism. A sensitivity that flips a finding is itself an important discovery.

**Text source discipline (CRITICAL):** Use `description_core_llm` as the primary text column for all text-dependent analyses. Use `description_core` only as a sensitivity check. NEVER mix text sources (some rows LLM, some rule-based) without explicitly reporting the `text_source` distribution and testing whether findings differ between the LLM-cleaned and rule-based subsets. When loading shared cleaned text artifacts, report the `text_source` distribution for your analysis sample.

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

### Agent N — Wave 4: Hypothesis generation, artifacts & synthesis (T24 + T25 + T26)

Read ALL reports. First, generate new hypotheses from the findings. Then produce interview elicitation artifacts. Finally, write the synthesis document. Execute tasks T24, T25, and T26.

### Agent P — Wave 5: Presentation (T27)

Read `exploration/reports/SYNTHESIS.md`, gate memos, and INDEX.md. Produce a ~20-25 slide MARP presentation for the research advisor and stakeholders. Follow the presentation principles in the orchestrator prompt (complete-sentence slide titles, one idea per slide, tell what you learned not what you did, frame corrections as rigor). Reference existing figures from `exploration/figures/`. Export to HTML and PDF via `npx @marp-team/marp-cli`. Execute task T27.

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

**Goal:** Test whether asaniczka `associate` can serve as a junior proxy, and document which seniority labels each source provides.

This task focuses on seniority label quality and the asaniczka comparability question. Coverage and missingness are handled by T01.

**Steps:**
1. Document which seniority labels each source provides natively. Note the critical gap: asaniczka has zero native entry-level labels.
2. SWE-only native-label comparability audit:
   - Compare asaniczka `associate` against arshkon `entry`, `associate`, and `mid-senior`
   - Use exact `title_normalized` overlap, explicit junior/senior title-cue rates, `yoe_extracted`, and downstream `seniority_final` distributions conditional on native label
   - State whether asaniczka `associate` behaves more like junior, lower-mid, mixed, or indeterminate
3. Decision rule: `usable as junior proxy` only if evidence is directionally close to arshkon `entry` on multiple signals
4. Entry-level effective sample sizes per source: how many entry-level rows have YOE, metro_area, description_core_llm, etc.?

**Output:** `exploration/reports/T02.md` with comparability audit tables and a clear verdict

### T03. Seniority label comparison `[Agent B]`

**Goal:** Compare all seniority variants — where do they agree, and does the choice change RQ1 results?

**Steps:**
1. SWE rows: cross-tabulate `seniority_native` vs `seniority_final`, `seniority_imputed` vs `seniority_final`, `seniority_native` vs `seniority_imputed` (where both non-null)
2. Compute agreement rate and Cohen's kappa for each pair
3. Arshkon SWE (native exists): per-class accuracy of rule-based classifier using native as ground truth
4. Scraped LinkedIn SWE (native exists): same. If accuracy differs, this means classifier temporal instability.
5. Compute RQ1 junior share using 4 variants:
   a. `seniority_final` (all non-unknown)
   b. `seniority_native` only
   c. High-confidence: `seniority_final_source IN ('title_keyword', 'native_backfill')`
   d. Including weak signals
6. Do variants agree on direction and magnitude of change?
7. **Which seniority operationalization is most defensible for each type of analysis?** (e.g., trend estimation vs cross-sectional comparison vs classification training)

**Output:** `exploration/reports/T03.md` with cross-tabs, kappa, per-class accuracy, junior-share comparison, recommendation

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
1. Description length: KS test + overlapping histograms for `description_length` and `core_length` across 3 sources
2. Company overlap: Jaccard similarity of `company_name_canonical` pairwise. Top-50 overlap.
3. Geographic: state-level SWE counts, chi-squared on state shares
4. Seniority: `seniority_final` distributions (exclude unknown), chi-squared pairwise
5. Title vocabulary: Jaccard of `title_normalized` sets. Titles unique to one period.
6. Industry: `company_industry` for arshkon vs scraped (asaniczka has no industry data)
7. **Artifact diagnostic:** For each metric with significant cross-dataset difference, can the difference be attributed to data collection method vs real change? Which comparisons are most trustworthy?
8. **Within-2024 calibration:** Run same comparisons between arshkon and asaniczka (both 2024) to establish baseline cross-source variability
9. **Platform labeling stability test.** For the top 20 SWE titles appearing in both arshkon and scraped:
   - Compare native seniority label distributions per title. If the same title has systematically different native labels across periods, that suggests platform relabeling rather than market change.
   - For title×seniority cells existing in both periods, compare YOE distributions. If YOE didn't change but frequency shifted, that's composition. If YOE changed too, it's content change.
   - Cross-validate with Indeed data: compute entry-level share using `seniority_imputed` on Indeed scraped rows. If Indeed shows similar patterns to LinkedIn, the LinkedIn platform artifact hypothesis weakens.

**Output:** `exploration/reports/T05.md` with test results, artifact assessment, calibration table, and platform stability assessment

### T06. Company concentration & within-company decomposition `[Agent C]`

**Goal:** Check if a few employers dominate and bias findings, and decompose aggregate seniority changes into within-company vs between-company components.

**Steps (SWE):**
1. Per dataset: HHI, top-1/5/10/20 share, Gini of posting counts
2. Companies with >3% SWE postings in any dataset
3. **Within-company vs aggregate decomposition (core analytical output).** Identify companies with >=5 SWE postings in BOTH arshkon and scraped. For this overlap set:
   - Compute aggregate entry share in each period (within overlap companies only)
   - Compare to the full-sample aggregate entry share
   - If the within-company decline exceeds the aggregate decline, composition effects are dampening the signal (new entrants are more junior-friendly). If it's smaller, composition effects are inflating the signal. State this clearly.
4. Company-capped sensitivity: junior share after capping at 10 postings/company
5. **Aggregator analysis:** What fraction of SWE postings are from aggregators per source? Do aggregator postings differ systematically in seniority/length/requirements?
6. **New entrants:** How many companies in 2026 scraped have NO match in 2024 arshkon? What is their seniority profile vs returning companies?

**Output:** `exploration/reports/T06.md` with concentration table, **within-company decomposition** (most important output), aggregator profile

### T07. External benchmarks & power analysis `[Agent D]`

**Goal:** Compare our data against BLS/JOLTS benchmarks. Assess statistical power and feasibility for all planned cross-period comparisons.

**Steps:**

*Part A — Feasibility table (primary output, drives all downstream decisions):*
1. Query the data for actual group sizes: entry-level, mid-senior, all SWE by source. Use these for power calculations.
2. Power analysis for cross-period comparisons: compute minimum detectable effect sizes (MDE) for binary and continuous outcomes at 80% power, alpha=0.05, for each key comparison (entry arshkon vs scraped, senior arshkon vs scraped, all SWE, pooled 2024 vs scraped).
3. Metro-level feasibility: How many metros have >=50 SWE per period? >=100? Which qualify for metro-level analysis?
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
1. **Cleaned text column.** For all SWE LinkedIn rows (filtered by default SQL): use `description_core_llm` where `llm_extraction_coverage = 'labeled'`, otherwise fall back to `description_core`, otherwise `description`. Strip company names using stoplist from all `company_name_canonical` values, remove standard English stopwords. Save as `exploration/artifacts/shared/swe_cleaned_text.parquet` with columns: `uid`, `description_cleaned`, `text_source` (which column was used: 'llm', 'rule', 'raw'), `source`, `period`, `seniority_final`, `seniority_3level`, `is_aggregator`, `company_name_canonical`, `metro_area`, `yoe_extracted`, `swe_classification_tier`, `seniority_final_source`.
2. **Sentence-transformer embeddings.** Using `all-MiniLM-L6-v2`, compute embeddings on first 512 tokens of `description_cleaned` for all rows in the cleaned text artifact. Process in batches of 256 to respect RAM limits. Save as `exploration/artifacts/shared/swe_embeddings.npy` (float32) with a companion `exploration/artifacts/shared/swe_embedding_index.parquet` mapping row index to `uid`.
3. **Technology mention binary matrix.** Using the ~100-120 technology taxonomy (define regex patterns for: Python, Java, JavaScript/TypeScript, Go, Rust, C/C++, C#, Ruby, Kotlin, Swift, Scala, PHP, React, Angular, Vue, Next.js, Node.js, Django, Flask, Spring, .NET, Rails, FastAPI, AWS, Azure, GCP, Kubernetes, Docker, Terraform, CI/CD, Jenkins, GitHub Actions, SQL, PostgreSQL, MongoDB, Redis, Kafka, Spark, Snowflake, Databricks, dbt, Elasticsearch, TensorFlow, PyTorch, scikit-learn, Pandas, NumPy, LangChain, RAG, vector databases, Pinecone, Hugging Face, OpenAI API, Claude API, prompt engineering, fine-tuning, MCP, LLM, Copilot, Cursor, ChatGPT, Claude, Gemini, Codex, Jest, Pytest, Selenium, Cypress, Agile, Scrum, TDD — expand to ~100+ with regex variations). Scan `description_cleaned` for each. Save as `exploration/artifacts/shared/swe_tech_matrix.parquet` (columns: `uid` + one boolean column per technology).
4. **Company name stoplist.** Extract all unique tokens from `company_name_canonical` values (tokenize on whitespace and common punctuation, lowercase, deduplicate). Save as `exploration/artifacts/shared/company_stoplist.txt`, one token per line.
5. **Structured skills extraction (asaniczka only).** Parse `skills_raw` from asaniczka SWE rows (comma-separated). Save parsed skills with uid as `exploration/artifacts/shared/asaniczka_structured_skills.parquet`.

6. **Within-2024 calibration table.** For ~30 common metrics (description_length, core_length, yoe_extracted median, tech_count mean, AI keyword prevalence, management indicator rate, scope term rate, soft skill rate, etc.), compute:
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

**Steps (SWE, LinkedIn-only, seniority_final != unknown where applicable):**
1. **Univariate profiling:** For every meaningful numeric and categorical column, compute distributions by period and by seniority. Produce side-by-side histograms/bar charts for at minimum: `description_length`, `core_length`, `yoe_extracted`, `seniority_final`, `seniority_3level`, `is_aggregator`, `metro_area` (top 15), `company_industry` (top 15 where available).
2. **Anomaly detection:** Flag any distribution that is bimodal, heavily skewed, or shows an unexpected pattern. Are there subpopulations hiding in the data?
3. **The YOE paradox:** Entry-level YOE appears to DECREASE from 2024 to 2026 (arshkon entry median 3.0, scraped entry median 2.0). Investigate this thoroughly: is it real, or an artifact of different entry-level composition? Break down by title, company type, source.
4. **Within-2024 baseline calibration (arshkon vs asaniczka, mid-senior SWE only):**
   - Compare description length, AI keyword prevalence, organizational language, and top-20 tech stack
   - Compute Cohen's d or equivalent effect sizes
   - Produce calibration table: metric, within-2024 difference, 2024-to-2026 difference, ratio
5. **Junior share trends:** Entry share by period using the seniority ablation framework (seniority_llm primary, then native, final, imputed). Also compute as share of known-seniority rows only.
6. **What variables show the LARGEST changes between periods?** Rank all available metrics by effect size. This identifies where to look deeper.
7. **Domain × seniority decomposition (tests H1).** If T09 archetype labels are available (from `exploration/artifacts/shared/swe_archetype_labels.parquet`), compute entry share by domain archetype by period. Decompose the aggregate entry share change into:
   - **Within-domain component:** entry share change holding archetype composition constant
   - **Between-domain component:** change driven by the market shifting between domain archetypes (e.g., from frontend to ML/AI)
   If the between-domain component accounts for a substantial portion of the aggregate decline, the junior decline is partly a domain recomposition effect, not purely within-domain elimination.
8. **Company size stratification (where data allows).** `company_size` is available for arshkon (99%). Within arshkon, stratify entry share, AI prevalence, and tech count by company size quartile. Do large companies show different patterns? For cross-period analysis where `company_size` is unavailable, use posting volume per company as a rough proxy.

**Essential sensitivities:** (a) aggregator exclusion, (b) company capping, (c) seniority operationalization (full ablation), (e) source restriction, (f) within-2024 calibration
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
   - Entry share (of known seniority) — apply the seniority ablation framework
   - AI keyword prevalence (binary per posting)
   - Mean description length
   - Mean tech count
   - Mean org_scope term count
3. **Cluster companies by their change profile.** k-means on the change vectors. Are there distinct strategies? Name them (e.g., "AI-forward", "traditional hold", "scope inflator", "downsizer").
4. **Within-company vs between-company decomposition:** For entry share, AI prevalence, and description length: how much of the aggregate 2024-to-2026 change is driven by within-company change vs different companies entering/exiting the sample? **Run the decomposition under two specifications: (1) pooled 2024 (arshkon + asaniczka) and (2) arshkon-only as 2024 baseline. If entry-share results disagree in direction, report and discuss both — this is a critical methodological finding.** If T09 archetype labels are available, add a domain dimension: decompose the entry share change into within-domain, between-domain, and between-company components.
5. **Within-company scope inflation (if validated patterns available).** If T22's validated management/scope patterns are available (from shared artifacts), compute within-company change in entry-level scope indicators for the overlap panel. This is the cleanest test of scope inflation: same companies across periods.
6. **New market entrants:** Profile companies in 2026 with no 2024 match. What industries? How do their postings compare?
7. **Aggregator vs direct employer:** Compare change patterns. Are aggregators showing different trends?

**Essential sensitivities:** (a) aggregator exclusion, (c) seniority operationalization (full ablation for entry metrics)
**Recommended sensitivities:** (b) company capping

**Output:** `exploration/reports/T16.md` + company cluster characterization + decomposition results (both pooled and arshkon-only)

### T17. Geographic market structure `[Agent J]`

**Goal:** Map geographic heterogeneity in SWE market changes.

**Steps (SWE, LinkedIn-only, using `metro_area`):**
1. **Metro-level metrics** for each metro with >=50 SWE postings per period:
   - Entry share (seniority_final)
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

**Entry-level rates involving asaniczka are unreliable** due to its zero native entry labels. For entry-level rate-of-change, use arshkon (Apr 2024) vs scraped (Mar 2026) only. Do not compute within-2024 entry-level rates unless `seniority_llm` is available (which resolves the asaniczka gap). If `seniority_llm` is available, include asaniczka as a genuine 2024-01 data point and re-compute the rate-of-change with all three snapshots.

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

**Essential sensitivities:** (a) aggregator exclusion, (c) seniority operationalization (full ablation)
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
   - **Seniority column recommendation** for different analytical purposes
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

**Inputs:** Read `exploration/reports/SYNTHESIS.md` (primary — the consolidated findings), gate memos (`exploration/memos/gate_*.md`), and `exploration/reports/INDEX.md` for the full task inventory. Reference existing figures from `exploration/figures/` and reports from `exploration/reports/`. Do not regenerate analysis — package what exists.

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
- **Methodology and data** — data sources, the sensitivity framework, limitations and open questions. A skeptical reader should be able to assess whether to trust the findings.
- **Narrative pages** — what the paper can and cannot claim, how the story evolved across gates, and the full synthesis.

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
| Boilerplate removal noise | Noisy text analysis | Use `description_core` + `description` | Low-Moderate |
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
