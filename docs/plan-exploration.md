# Exploration & Validation Plan

Date: 2026-03-23
Status: Executable (v5 — enhanced)
Input: `preprocessing/intermediate/stage8_final.parquet` (~76 cols, ~1.22M rows)
Schema reference: `docs/schema-stage8-and-stage12.md`

---

## 1. Orchestrator instructions

You are the orchestrator. Your job is to dispatch sub-agents, gate between waves, and track progress. You do NOT execute exploration tasks yourself.

### Dispatch pattern

For each wave, launch up to 4 sub-agents in parallel using the Agent tool. Each agent's prompt is:

```
PREAMBLE (Section 2 below)
+ agent-specific dispatch block (from Section 3)
+ task specs for assigned tasks (from Section 4)
```

### Workflow

```
Wave 1 → dispatch agents A, B, C, D (parallel)
       → wait for all to complete
       → Gate 1: read reports, check for blockers
       → update exploration/reports/INDEX.md

Wave 2 → dispatch agents E, F, G, H (parallel)
       → wait for all to complete
       → Gate 2: read reports, check for blockers
       → update INDEX.md

Wave 3 → dispatch agents I, J, K, L (parallel)
       → wait for all to complete
       → Gate 3: read reports, check for blockers
       → update INDEX.md

Wave 4 → dispatch agent M
       → wait for completion
       → final INDEX.md update
```

### Gate logic

At each gate, read the `exploration/reports/T*.md` files from the just-completed wave. Check:

1. **Blocking issues** — Did any task report data that makes downstream work invalid? For example:
   - Column coverage too low for a required analysis
   - SWE classification fundamentally broken (>20% estimated false positive/negative)
   - Seniority labels contradictory across sources in a way that invalidates cross-period comparison
   - Entry-level SWE sample so small it's meaningless (<30 rows)
2. **Warnings** — Issues to note but not block on:
   - Thin coverage for specific columns (flag which analyses are affected)
   - Moderate disagreement between seniority sources (record which is recommended)
   - Cross-dataset differences that need interpretation

If a blocker is found: stop, report to user, discuss whether to fix preprocessing first.
If only warnings: record them in INDEX.md and proceed to next wave.

### Progress tracking

Maintain `exploration/reports/INDEX.md` with:
```
| Task | Agent | Wave | Status | One-line finding |
```

Update after each gate.

---

## 2. Shared preamble

Prepend this to every sub-agent prompt.

```
## Exploration task context

You are a sub-agent executing exploration tasks for a SWE labor market research project.

**Input data:** `preprocessing/intermediate/stage8_final.parquet` (~76 columns, ~1.22M rows)
Read `docs/schema-stage8-and-stage12.md` for column definitions and recommended usage.

**Key data context:**
- asaniczka has zero entry-level native labels (only mid-senior and associate). Default: exclude it from entry-level trend analysis. T03 now explicitly tests whether `associate` can support a limited junior-proxy sensitivity; do not assume `Associate == entry` unless that audit recommends it.
- 31GB RAM limit — use DuckDB or pyarrow for queries, never load full parquet into pandas.

**Default SQL filters (apply unless task says otherwise):**
```sql
WHERE source_platform = 'linkedin'
  AND is_english = true
  AND date_flag = 'ok'
```

**Three periods:** 2024-01 (asaniczka), 2024-04 (arshkon), 2026-03 (scraped)
**Three sources:** kaggle_arshkon (118K rows), kaggle_asaniczka (1.06M rows), scraped (40K rows and growing)

**Text analysis hygiene suggestions — apply to text-based tasks (T10+):**

1. **Company-name stripping.** Before any corpus comparison or term-frequency analysis, build a stoplist from all `company_name_canonical` values, tokenize them into words, and strip them during tokenization. This prevents company names from dominating results ("Capital One" was the top distinguishing term in multiple v1 comparisons).

3. **Artifact filtering.** For emerging/disappearing term lists:
   - Require terms to appear in >=20 distinct companies (prevents single-company artifacts like "dataannotation", "amazonians")
   - Exclude HTML concatenation artifacts (tokens >12 chars with no spaces that aren't real words, like "skillsa", "positionyou", "chatbotwrite")
   - Exclude city/location names (check against `metro_area` and `state_normalized` values)

4. **Semantic categorization.** Tag every reported term with a category from this taxonomy:
   - `ai_tool`: AI coding tools, LLMs, specific models (copilot, cursor, claude, gpt, llm, rag, agent, mcp)
   - `ai_domain`: ML/AI as a domain (machine learning, deep learning, NLP, computer vision)
   - `tech_stack`: Specific technologies, frameworks, languages (react, kubernetes, terraform, python, docker)
   - `org_scope`: Organizational/scope language (ownership, end-to-end, cross-functional, stakeholder, autonomous)
   - `mgmt`: Management/leadership (lead, mentor, manage, team, hire, coach, 1:1)
   - `sys_design`: Systems/architecture (distributed systems, scalability, architecture, microservices)
   - `method`: Development methodology (agile, scrum, ci/cd, tdd)
   - `credential`: Formal requirements (years experience, BS/MS/PhD, certification)
   - `soft_skill`: Interpersonal (collaboration, communication, problem-solving)
   - `noise`: Residual after filtering (target <10% of reported terms)

5. **Length normalization.** Description length grew ~50-60% from 2024 to 2026. For keyword analyses:
   - Primary: binary indicator (any mention per posting)
   - Secondary: rate per 1,000 characters
   - Always report both. Never report raw counts without normalization.

**Output conventions:**
- Figures → exploration/figures/TASK_ID/ (PNG, 150dpi, max 4 per task)
- Tables → exploration/tables/TASK_ID/ (CSV)
- Report → exploration/reports/TASK_ID.md using this template:

  # TASK_ID: [title]
  ## Finding
  [1-3 sentence headline result]
  ## Implication for analysis
  [What this means for RQ1-RQ4]
  ## Methodology
  [How was the conclusion derived from the data]
  ## Data quality note
  [Caveats, issues, thin samples]
  ## Action items
  [What downstream agents or the analysis phase needs to know]

Create exploration/ directories if they don't exist.
Use DuckDB CLI or pyarrow for all data queries. Do NOT load the full parquet into pandas.
```

---

## 3. Wave definitions

### Wave 1 — Data audit & validation

Launch 4 agents in parallel. These establish what we have and whether it's usable.

#### Agent A: Data coverage (T01 + T03)

**Dispatch:** Audit column coverage and missing data patterns across all sources and the SWE subset. Produce the coverage heatmap and missing data tables that all downstream tasks depend on understanding. Also run the native-label comparability audit for whether asaniczka `associate` behaves enough like arshkon `entry` to justify a limited sensitivity use. What are some alternative labels that we can use? Execute tasks T01 and T03.

#### Agent B: Classifier quality (T02 + T04)

**Dispatch:** Evaluate seniority label quality and SWE classification accuracy. Cross-tabulate all seniority variants, compute agreement metrics, and test whether the RQ1 junior-share metric changes depending on which seniority column is used. Assess SWE classification via manual sampling of borderline cases. Execute tasks T02 and T04.

#### Agent C: Dataset comparability (T05 + T06)

**Dispatch:** Test whether the three datasets are measuring the same thing by running pairwise comparisons (description length, company overlap, geographic/seniority/title distributions). Also assess company concentration and whether a few employers dominate findings. Execute tasks T05 and T06.

#### Agent D: External benchmarks (T07)

**Dispatch:** Compare our data against BLS OES occupation/state data and JOLTS information sector trends. This requires web access to download benchmark data from FRED and BLS. Execute task T07.

#### Gate 1 checklist

After all Wave 1 agents complete, read these reports and check:

- [ ] `T01.md` — Are the columns needed for RQ1-RQ3 adequately covered? Which are unusable?
- [ ] `T02.md` — Do seniority variants agree on junior-share direction? What's the recommended seniority column?
- [ ] `T03.md` — What's the effective sample size for entry-level SWE? Is it >30? Does the asaniczka `associate` audit support any limited junior-proxy use, or should it remain excluded?
- [ ] `T04.md` — Is SWE classification adequate (<10% estimated error)?
- [ ] `T05.md` — Are cross-dataset differences explainable? Any artifact red flags?
- [ ] `T06.md` — Does any single company dominate >10% of SWE postings?
- [ ] `T07.md` — Is geographic correlation with OES >0.80?

**Pass to Wave 2:** Record seniority recommendation from T02, any column exclusions from T01, and the T03 verdict on whether asaniczka `associate` can be used in any appendix-only junior sensitivity. Wave 2 agents will read INDEX.md for this guidance.

---

### Wave 2 — Core exploratory analysis

Launch 4 agents in parallel. These generate the substantive findings.

#### Agent E: Distributions, sensitivity & baseline (T08 + T09)

**Dispatch:** Compute baseline distributions for all key variables by period and seniority. Run the seniority source sensitivity analysis. Also establish within-2024 baseline variability by comparing arshkon vs asaniczka on the same metrics, to calibrate how surprising the 2024-to-2026 changes are. Read `exploration/reports/INDEX.md` for the seniority recommendation from Wave 1, including the T03 verdict on whether any appendix-only `asaniczka associate` vs `arshkon entry` sensitivity is allowed. Execute tasks T08 and T09.

#### Agent F: Text analysis (T10 + T11)

**Dispatch:** Run Fightin' Words corpus comparisons (6 pairs: junior/senior x period, plus cross-occupation) to identify statistically distinguishing terms. Then measure temporal drift via Jensen-Shannon divergence, keyword emergence/disappearance, and AI term prevalence. Apply text hygiene rules from the preamble and proactively identify new hygenie rules — this is critical for quality. Execute tasks T10 and T11.

#### Agent G: Requirements & companies (T12 + T13)

**Dispatch:** Parse job description requirements sections to extract structured data (education, tech count, soft skills) and use data from existing columns (i.e. YOE). Then analyze company-level patterns: within-company vs between-company changes for companies appearing in both periods. Execute tasks T12 and T13.

#### Agent H: RQ3 + quality + controls (T14 + T15 + T16)

**Dispatch:** Three tasks. Compute the employer-requirement vs worker-usage divergence for RQ3. Profile ghost jobs and anomalies. Run the full cross-occupation comparison (SWE vs SWE-adjacent vs control) to test whether findings are SWE-specific. Execute tasks T14, T15, and T16.

#### Gate 2 checklist

After all Wave 2 agents complete:

- [ ] `T08.md` — Do distribution profiles show expected patterns? Any surprises? Is within-2024 baseline variability small relative to 2024-2026 change?
- [ ] `T09.md` — Do all seniority variants agree on direction of junior-share change?
- [ ] `T10.md` — Are the top distinguishing terms clean (not company names/artifacts)? Do they align with RQ1-RQ2 hypotheses? Is the category breakdown informative?
- [ ] `T11.md` — How large is temporal drift? Is JSD for junior > JSD for senior (supporting RQ1)? Are emerging terms properly cleaned?
- [ ] `T12.md` — Does scope inflation show up in structured requirements?
- [ ] `T13.md` — Does within-company analysis confirm or weaken the cross-period finding?
- [ ] `T14.md` — Is there a posting-usage divergence? (RQ3)
- [ ] `T15.md` — Any systematic data quality issues to exclude?
- [ ] `T16.md` — Are SWE patterns distinct from control occupations?

**Pass to Wave 3:** Note any findings that are contradictory or surprising — Wave 3 needs to dig deeper.

---

### Wave 3 — Deep analysis

Launch 4 agents in parallel. These tasks go deeper into the research constructs.

#### Agent I: Technology stacks & description anatomy (T17 + T18)

**Dispatch:** Track specific technology mentions across periods using a ~100-150 technology taxonomy. Then decompose the description length growth to understand what sections are getting longer. Execute tasks T17 and T18.

#### Agent J: Requirement bundles & relabeling test (T19 + T20)

**Dispatch:** Analyze requirement co-occurrence patterns to find latent posting archetypes (e.g., "traditional SWE" vs "AI-augmented SWE"). Then test the relabeling hypothesis: are 2026 entry-level postings semantically more similar to 2024 mid-senior postings than to 2024 entry-level postings? Execute tasks T19 and T20.

#### Agent K: Senior archetype & metro heterogeneity (T21 + T22)

**Dispatch:** Systematically measure the senior SWE archetype shift (management vs orchestration language profiles) — this is a core construct in the research design. Then analyze whether the headline findings vary by metro area. Execute tasks T21 and T22.

#### Agent L: Ghost patterns & embedding similarity (T23 + T24)

**Dispatch:** Identify ghost-like requirement patterns through text analysis (template saturation, aspirational language, kitchen-sink postings, company repetition). Then compute semantic similarity between junior and senior postings over time using embeddings or TF-IDF to test the convergence hypothesis. Execute tasks T23 and T24.

#### Gate 3 checklist

After all Wave 3 agents complete:

- [x] `T17.md` — What technology stacks are rising/declining beyond just "AI"?
- [x] `T18.md` — What's driving description length growth? Requirements vs boilerplate?
- [ ] `T19.md` — Do posting archetypes emerge? Does an "AI-augmented SWE" archetype grow?
- [ ] `T20.md` — Does entry-2026 converge toward mid-senior-2024 (relabeling confirmed)?
- [ ] `T21.md` — Is the management→orchestration shift real and measurable?
- [ ] `T22.md` — Is the entry decline uniform across metros or concentrated?
- [ ] `T23.md` — Are ghost requirements prevalent enough to matter?
- [ ] `T24.md` — Does embedding similarity confirm the convergence hypothesis?

**Pass to Wave 4:** Compile key tensions and surprising findings for synthesis.

---

### Wave 4 — Synthesis

Launch 1 agent after Waves 1-3 are complete.

#### Agent M: Artifacts & synthesis (T25 + T26)

**Dispatch:** Read ALL reports in `exploration/reports/`. Produce the 5 interview elicitation artifacts for RQ4. Then write the synthesis document that consolidates all findings into a single handoff for the analysis agent. Execute tasks T25 and T26.

---

## 4. Task reference

Each task is assigned to one agent. The agent receives the task spec below as part of its prompt.

### T01. Column coverage audit `[Agent A]`

**Goal:** Which columns have enough non-null coverage to be analysis-worthy, by source and SWE subset?

**Steps:**
1. For every column, compute by source (arshkon, asaniczka, scraped) AND by `is_swe` subset: non-null rate, distinct count, top 5 values
2. Produce a coverage heatmap (columns x sources, colored by non-null rate)
3. Flag columns >50% null for any source used in cross-period comparisons
4. Produce a "usable columns per RQ" table
5. Check whether `description_core_llm` exists and what its coverage is

**Output:** `exploration/reports/T01.md` + coverage heatmap PNG + CSV

### T02. Seniority label comparison `[Agent B]`

**Goal:** Compare all seniority variants — where do they agree, and does the choice change RQ1 results?

**Steps:**
1. SWE rows: cross-tabulate `seniority_native` vs `seniority_final`, `seniority_imputed` vs `seniority_final`, `seniority_native` vs `seniority_imputed` (where both non-null)
2. Compute agreement rate and Cohen's kappa for each pair
3. Arshkon SWE (native exists): per-class accuracy of rule-based classifier using native as ground truth
4. Scraped LinkedIn SWE (native exists): same. If accuracy differs → classifier temporal instability.
5. Compute RQ1 junior share using 4 variants:
   a. `seniority_final` (all non-unknown)
   b. `seniority_native` only
   c. High-confidence: `seniority_final_source IN ('title_keyword', 'native_backfill')`
   d. Including weak signals
6. Do variants agree on direction and magnitude of change?

**Output:** `exploration/reports/T02.md` with cross-tabs, kappa, per-class accuracy, junior-share comparison, recommendation

### T03. Missing data audit + native-label comparability `[Agent A]`

**Goal:** Document missingness patterns constraining cross-period comparisons, and test whether asaniczka `associate` can be treated as a limited junior proxy.

**Steps:**
1. Field x source x platform missing data table (% non-null), all rows and SWE subset
2. Document which seniority labels each source provides
3. SWE-only native-label comparability audit:
   - Compare asaniczka `associate` against arshkon `entry`, `associate`, and `mid-senior`
   - Use exact `title_normalized` overlap, explicit junior/senior title-cue rates, `yoe_extracted`, and downstream `seniority_final` distributions conditional on native label
   - State whether asaniczka `associate` behaves more like junior, lower-mid, mixed, or indeterminate
4. Decision rule:
   - `usable as junior proxy` only if the evidence is directionally close to arshkon `entry` on multiple signals
   - otherwise keep asaniczka excluded from junior-native baselines and treat `associate` as a distinct or mixed bucket
5. Effective sample size per source after excluding nulls, for each cross-period analysis field

**Output:** `exploration/reports/T03.md` with tables and a clear verdict on whether any appendix-only `asaniczka associate` junior sensitivity is justified

### T04. SWE classification audit `[Agent B]`

**Goal:** Assess SWE classification quality after preprocessing fixes.

**Steps:**
1. SWE rows by `swe_classification_tier` breakdown
2. Sample 50 borderline SWE postings (`swe_confidence` 0.3-0.7 or tier `title_lookup_llm`): print title + 200 chars description, assess quality
3. Sample 50 borderline non-SWE (titles with "engineer"/"developer"/"software" but `is_swe = False`): same
4. Profile `is_swe_adjacent` and `is_control` rows: what titles/occupations?
5. Estimated false-positive and false-negative rates
6. Verify no dual-flag violations: `(is_swe + is_swe_adjacent + is_control) > 1` should be 0

**Output:** `exploration/reports/T04.md`

### T05. Cross-dataset comparability `[Agent C]`

**Goal:** Test whether dataset differences reflect real labor market changes vs artifacts.

**Steps (SWE, LinkedIn-only):**
1. Description length: KS test + histogram for `description_length` and `core_length` across 3 sources
2. Company overlap: Jaccard similarity of `company_name_canonical` pairwise. Top-50 overlap.
3. Geographic: state-level SWE counts, chi-squared on state shares
4. Seniority: `seniority_final` distributions (exclude unknown), chi-squared pairwise
5. Title vocabulary: Jaccard of `title_normalized` sets. Titles unique to one period.
6. Industry: `company_industry` for arshkon vs scraped

**Output:** `exploration/reports/T05.md` with test results and interpretation

### T06. Company concentration `[Agent C]`

**Goal:** Check if a few employers dominate and bias findings.

**Steps (SWE):**
1. Per dataset: HHI, top-1/5/10/20 share, Gini of posting counts
2. Companies with >3% SWE postings in any dataset
3. Overlap set (arshkon ∩ scraped): compare seniority distributions within-company across periods
4. Company-capped sensitivity: junior share after capping at 10 postings/company

**Output:** `exploration/reports/T06.md` with concentration table and within-company comparison

### T07. Representativeness `[Agent D]`

**Goal:** Compare our data against BLS/JOLTS benchmarks.

**Steps:**
1. Download BLS OES for SOC 15-1252 and 15-1256: state-level employment
2. Pearson r: our state-level SWE counts vs OES, per dataset
3. Industry distribution: our SWE vs OES SWE industry (arshkon + scraped)
4. Download JOLTS information sector from FRED. Compare against our scraper daily counts.
5. Representativeness summary table

**Output:** `exploration/reports/T07.md`. Target: r > 0.80 geographic.

### T08. Distribution profiles & within-2024 baseline `[Agent E]`

**Goal:** Baseline distributions before hypothesis testing, plus within-2024 variability calibration.

**Steps (SWE, LinkedIn-only, seniority_final != unknown where applicable):**
1. Side-by-side histograms by period for: `description_length`, `core_length`, `yoe_extracted`, `seniority_final`, `seniority_3level`
2. Junior/mid/senior share by period (entry / total known seniority)
3. Arshkon-specific: entry-level share of SWE postings
4. Remote work rate and aggregator rate by period
5. **Within-2024 baseline comparison (arshkon vs asaniczka, mid-senior SWE only):**
   - Compare description length, AI keyword prevalence, organizational language (cross-functional, ownership, end-to-end, collaboration), and top-20 tech stack on the same metrics used for 2024→2026 comparisons
   - Compute within-2024 effect sizes (Cohen's d or equivalent)
   - Produce a calibration table: metric, within-2024 difference, 2024→2026 difference, ratio
   - This establishes how much variation is "normal" within 2024
6. Optional appendix-only sensitivity:
   - If and only if T03 explicitly recommends it, compare asaniczka `associate` vs arshkon `entry` on the same baseline metrics
   - Report this as a label-semantic sensitivity, not as a replacement for the main junior baseline

**Output:** `exploration/reports/T08.md` with plots, summary stats, and baseline calibration table

### T09. Seniority source sensitivity `[Agent E]`

**Goal:** Is the junior-share finding robust to seniority method choice?

**Steps:**
1. Junior share by period using 4 seniority variants (same as T02 step 5)
2. Arshkon-to-scraped change for each variant (absolute pp and relative %)
3. Sensitivity comparison chart (all 4 estimates)
4. Agreement assessment: same direction? Magnitude within 2x?

**Output:** `exploration/reports/T09.md` with chart and verdict

### T10. Fightin' Words corpus comparison `[Agent F]`

**Goal:** Statistically identify the words that distinguish groups, with clean results.

**Steps:**
1. **Text cleaning first (critical):**
   - Build company-name stoplist from all `company_name_canonical` values
   - Strip EEO/legal/benefits/about-company sections via regex before tokenization
   - Use `description_core_llm` if available, else `description`
2. Run 6 comparisons (SWE, LinkedIn-only, seniority_final != unknown):

   | # | Corpus A | Corpus B | RQ |
   |---|---|---|---|
   | 1 | Junior 2024 (arshkon entry) | Junior 2026 (scraped entry) | RQ1 |
   | 2 | Senior 2024 | Senior 2026 | RQ1 senior redefinition |
   | 3 | Junior 2024 | Senior 2024 | RQ1/RQ2 baseline gap |
   | 4 | Junior 2026 | Senior 2026 | RQ1/RQ2 current gap |
   | 5 | Junior 2026 | Senior 2024 | RQ1 redefinition hypothesis |
   | 6 | SWE 2026 | Control 2026 | Supporting |

3. Top 50 distinguishing terms (|z-score| > 3.0) for unigrams and bigrams per comparison. **Every term must be tagged** with a semantic category from the preamble taxonomy. Target <10% `noise` category.
4. Produce a **category-level summary**: what fraction of distinguishing terms are ai_tool, org_scope, mgmt, sys_design, etc.? How does this change across comparisons? This is the high-level finding.
5. Produce Scattertext HTML for comparisons 1 and 2 (if Scattertext is available)

**Note:** Report n per corpus for every comparison. Flag any with n < 50.

**Output:** `exploration/reports/T10.md` + categorized CSV tables + category summary figure

### T11. Temporal drift `[Agent F]`

**Goal:** Quantify how much posting content changed between periods, with clean term lists.

**Steps (SWE, LinkedIn-only):**
1. **Apply same text cleaning as T10** before all frequency computations.
2. JSD on unigram frequencies: overall and per seniority level. Also compute JSD on cleaned text and compare to uncleaned JSD to quantify noise contribution.
3. **Emerging terms** (>1% in 2026, <0.1% in 2024): apply artifact filters — require >=20 distinct companies, exclude proper nouns/locations, exclude HTML artifacts. Categorize every surviving term.
4. **Accelerating terms** (existed in 2024 but grew >3x): these may be more informative than the binary emerging/disappearing threshold. Report top-30 accelerating terms with categories.
5. **Disappearing terms** (>1% in 2024, <0.1% in 2026): same filtering. Separate true vocabulary change from source artifacts.
6. YOE inflation: `yoe_extracted` for entry-level SWE, arshkon vs scraped
7. AI keyword prevalence: share mentioning AI/LLM/agent/copilot/GPT/Claude, by period × seniority. Separate "AI-as-tool" (copilot, cursor, AI pair programming, prompt engineering) from "AI-as-domain" (machine learning, deep learning, NLP).

**Output:** `exploration/reports/T11.md` with cleaned JSD values, categorized term lists, AI prevalence chart

### T12. Requirements parsing `[Agent G]`

**Goal:** Extract structured data from requirements/qualifications sections.

**Steps:**
1. Parse `description` (or `description_core_llm` if available) to extract requirements sections (regex for "Requirements", "Qualifications", "What you'll need", etc.)
2. Extract: YOE patterns, education (BS/MS/PhD), distinct tech count, soft skill / management language
3. Per seniority × period: median YOE, % MS/PhD, median tech count, % "cross-functional"/"stakeholder"/"ownership"/"end-to-end"
4. Junior 2024 vs junior 2026 comparison: are junior roles requiring more?
5. **Length-normalized tech density** (techs per 1K chars) alongside raw counts — the 2024→2026 description length growth inflates raw counts

**Output:** `exploration/reports/T12.md` — produces scope inflation evidence for RQ1

### T13. Company-level patterns `[Agent G]`

**Goal:** Separate within-company change from between-company composition effects.

**Steps:**
1. Companies in both arshkon and scraped (`company_name_canonical`)
2. Overlapping companies: seniority distributions, description lengths, AI keyword prevalence across periods
3. Non-overlapping: what companies are new in 2026? What disappeared?
4. Size-band split where available
5. Formal within-company vs composition decomposition for: entry share, AI prevalence, description length

**Output:** `exploration/reports/T13.md`

### T14. RQ3 divergence `[Agent H]`

**Goal:** Compare posting-side AI requirements against worker-side AI usage benchmarks.

**Steps:**
1. AI requirement rate in SWE postings, by period and seniority. Separate "AI-as-tool" (copilot, cursor, LLM, prompt engineering) from "AI-as-domain" (ML, DL, NLP).
2. Pull Anthropic occupation-level AI usage data (https://www.anthropic.com/research/labor-market-impacts, https://www.anthropic.com/economic-index). Map to comparable SOC codes. Also use StackOverflow Developer Survey benchmarks.
3. Divergence: requirement rate vs usage rate, by seniority. Make sure to do fair analysis given the data sources are completely different. What comparisons can we make?
4. Produce divergence chart

**Output:** `exploration/reports/T14.md` + divergence chart (becomes interview artifact)

### T15. Ghost job & anomaly profiling `[Agent H]`

**Goal:** Characterize ghost jobs and outliers.

**Steps:**
1. Profile `ghost_job_risk` non-low rows: companies, seniority, geography
2. `yoe_seniority_contradiction = True`: how many, patterns
3. `description_quality_flag != 'ok'`: fraction per source
4. Extreme `description_length` outliers (>15K or <100 chars)

**Output:** `exploration/reports/T15.md`

### T16. Cross-occupation comparison `[Agent H]`

**Goal:** Are observed changes SWE-specific or a broader trend?

**Steps:**
1. `is_control` rows: seniority distribution by period
2. `is_swe_adjacent` rows: same
3. Junior share trends: SWE vs adjacent vs control
4. AI keyword prevalence across all three groups
5. If control shows same patterns → confounded by macro trends

**Output:** `exploration/reports/T16.md`

### T17. Technology stack tracking `[Agent I]`

**Goal:** Track specific technology mentions across periods to understand how the SWE toolkit is evolving.

**Steps:**
1. Define a taxonomy of ~100-150 technologies by category:
   - **Languages:** Python, Java, JavaScript/TypeScript, Go, Rust, C/C++, C#, Ruby, Kotlin, Swift, Scala, Elixir, PHP
   - **Frontend:** React, Angular, Vue, Next.js, Svelte
   - **Backend/Infra:** Node.js, Django, Flask, Spring, .NET, Rails
   - **Cloud/DevOps:** AWS, Azure, GCP, Kubernetes, Docker, Terraform, CI/CD, Jenkins, GitHub Actions
   - **Data:** SQL, PostgreSQL, MongoDB, Redis, Kafka, Spark, Snowflake, Databricks, dbt
   - **AI/ML traditional:** TensorFlow, PyTorch, scikit-learn, Pandas, NumPy
   - **AI/LLM new:** LangChain, LangGraph, RAG, vector databases, Pinecone, ChromaDB, Hugging Face, OpenAI API, Claude API, prompt engineering, fine-tuning, MCP, agent frameworks
   - **AI tools:** Copilot, Cursor, ChatGPT, Claude, Gemini, Codex
   - **Testing/Practices:** Jest, Pytest, Selenium, Cypress, Agile, Scrum, TDD
2. For each technology, compute mention rate (% of postings mentioning it) by period × seniority. Use regex patterns that account for common variations.
3. Produce a "technology shift" heatmap: rows = technologies, columns = period × seniority, cell = mention rate. Highlight technologies with >3x change.
4. Identify "rising stacks" (new AI/LLM tools) vs "stable stacks" (languages, cloud) vs "declining stacks".
5. Length-normalize: rate per 1K chars alongside raw rates.

**Output:** `exploration/reports/T17.md` + tech heatmap + CSVs

### T18. Description anatomy `[Agent I]`

**Goal:** Decompose the description length growth to understand what's getting longer.

**Steps:**
1. Define a section classifier using regex for common JD sections:
   - Role summary, Requirements/Qualifications, Preferred/Nice-to-have, Responsibilities, Benefits/Perks, About the company, Legal/EEO, Unclassified
2. For each SWE posting, estimate character count per section.
3. Compute median section length by period × seniority. Which sections grew the most?
4. Stacked bar chart: description composition by period.
5. Test: did the "requirements" section grow disproportionately, or is the growth mainly in benefits/legal/about?
6. Entry-level specific: what changed in the structure of entry-level JDs?

**Output:** `exploration/reports/T18.md` + stacked bar chart + section-length tables

### T19. Requirement bundle analysis `[Agent J]`

**Goal:** Identify how requirements co-occur and whether there are distinct posting archetypes.

**Steps:**
1. For each SWE posting, create a binary feature vector from ~30-40 requirement indicators:
   - AI-tool, AI-domain, AI-general
   - Ownership, end-to-end, cross-functional, stakeholder
   - Leadership, mentoring, hiring, team management
   - System design, architecture, distributed systems, scalability
   - CI/CD, deployment, infrastructure
   - Testing, code review
   - Communication, collaboration
   - YOE buckets (0-1, 2-3, 4-5, 6+), Education (BS, MS, PhD)
2. Compute pairwise co-occurrence matrix (phi coefficient) by period.
3. Identify new co-occurrence pairs in 2026 (e.g., "AI + ownership" becoming a bundle).
4. Cluster postings into 4-6 archetypes (k-means or hierarchical on binary features).
5. Name and characterize each archetype. Compare archetype distributions across periods.
6. Test: does an "AI-augmented SWE" archetype grow at the expense of "traditional SWE"?

**Output:** `exploration/reports/T19.md` + co-occurrence heatmaps + archetype distributions

### T20. Relabeling hypothesis test `[Agent J]`

**Goal:** Test whether 2026 entry-level postings are semantically more similar to 2024 mid-senior postings than to 2024 entry-level postings.

**Steps:**
1. Compute TF-IDF vectors on cleaned descriptions (company names and boilerplate removed).
2. Compute average cosine similarity between:
   - Entry 2026 ↔ Entry 2024 (same-level, cross-period)
   - Entry 2026 ↔ Mid-senior 2024 (cross-level, cross-period — the relabeling test)
   - Entry 2024 ↔ Mid-senior 2024 (cross-level baseline)
   - Mid-senior 2026 ↔ Mid-senior 2024 (same-level, cross-period)
3. If entry-2026 is closer to mid-senior-2024 than to entry-2024, that supports relabeling.
4. Dimensionality reduction (PCA or UMAP on TF-IDF) to visualize cluster positions.
5. Within-level similarity: is entry becoming more or less diverse over time?

**Output:** `exploration/reports/T20.md` + similarity matrix + UMAP/PCA plot

### T21. Senior archetype analysis `[Agent K]`

**Goal:** Systematically measure the senior SWE archetype shift from people-management toward AI-enabled orchestration.

**Steps:**
1. Define two language profiles:
   - **Management:** manage, mentor, coach, hire, interview, grow, develop talent, performance review, career development, 1:1, headcount, people management, team building, direct reports
   - **Orchestration:** architecture review, code review, system design, technical direction, AI orchestration, agent, workflow, pipeline, automation, evaluate, validate, quality gate, guardrails, prompt engineering, tool selection
2. For each mid-senior/director SWE posting, compute management score and orchestration score (mentions per 1K chars).
3. Compare distributions across periods. Compute management-to-orchestration ratio by period.
4. Identify "new senior" (high orchestration, low management) vs "classic senior" (high management, low orchestration). How did their proportions change?
5. Cross-tabulate with AI keyword presence: among AI-mentioning senior postings, is orchestration stronger?
6. 2D scatter (management vs orchestration) colored by period.

**Output:** `exploration/reports/T21.md` + management-orchestration charts

### T22. Metro heterogeneity `[Agent K]`

**Goal:** Test whether the headline findings vary by metro area.

**Steps:**
1. Using `metro_area` (stage8-only, ~66-75% coverage), compute for each metro with >=50 SWE postings per period:
   - Entry share (seniority_final)
   - AI keyword prevalence (broad and AI-tool-specific)
   - Organizational language composite (ownership + cross-functional + end-to-end)
   - Description length
2. Rank metros by entry-share decline (arshkon→scraped).
3. Is the entry decline concentrated in certain metros (SF, NYC, Seattle) or uniform?
4. Is the AI surge concentrated in tech hubs or uniform?
5. Metro-level heatmap: metros × metrics, colored by change magnitude.
6. Correlation test: do metros with larger AI surges show larger entry declines?

**Output:** `exploration/reports/T22.md` + metro heatmap + correlation chart

### T23. Ghost requirement patterns `[Agent L]`

**Goal:** Identify ghost-like requirement patterns through text analysis.

**Steps:**
1. Define ghost-like indicators:
   - **Template saturation:** Postings where the requirements section is near-identical to other postings from the same company (compute within-company requirements similarity)
   - **Kitchen-sink postings:** Postings listing >15 distinct technologies or >8 organizational competencies
   - **Aspiration markers:** Ratio of hedging language ("ideally", "nice to have", "preferred", "bonus") relative to firm requirements ("must have", "required", "minimum")
   - **YOE-scope mismatch:** Entry-level postings with senior scope language (ownership, architecture, distributed systems)
   - **Company repetition:** Companies posting near-identical requirements across roles
2. Compute prevalence of each indicator by period × seniority.
3. Identify the 20 most "ghost-like" entry-level postings. Display their requirements sections.
4. Cross-tabulate ghost indicators with AI keyword presence: are AI requirements more likely to be ghost-like?
5. Compare aggregators vs direct employers, large vs small companies.

**Output:** `exploration/reports/T23.md` + ghost prevalence tables + examples

### T24. Embedding-based similarity `[Agent L]`

**Goal:** Compute semantic similarity between junior and senior postings over time.

**Steps:**
1. Check if `sentence-transformers` is installed. If yes: use `all-MiniLM-L6-v2`. If no: fall back to TF-IDF + SVD (100 components).
2. Sample up to 2,000 SWE postings per period × seniority group (to fit in RAM). Use cleaned description text (first 1000 chars to normalize length).
3. Compute average pairwise cosine similarity between all group pairs:
   - Entry 2024 ↔ Entry 2026
   - Entry 2026 ↔ Mid-senior 2024 (convergence test)
   - Entry 2024 ↔ Mid-senior 2024 (baseline gap)
   - Mid-senior 2024 ↔ Mid-senior 2026
4. If entry-2026 is closer to mid-senior-2024 than entry-2024 is, the junior-senior gap is narrowing.
5. UMAP or t-SNE visualization of embeddings, colored by period × seniority.
6. Within-group similarity (how homogeneous each group is).

**Output:** `exploration/reports/T24.md` + similarity matrix + dimensionality reduction plot

### T25. Interview elicitation artifacts `[Agent M]`

**Goal:** Produce 5 artifacts for RQ4 data-prompted interviews.

**Steps (reads all prior reports):**
1. **Inflated junior JDs:** From T12/T19, select 3-5 entry-level postings with scope-inflated requirements. Query parquet for actual text.
2. **Paired JDs over time:** From T13, select 3-5 same-company pairs (2024 vs 2026). Format side-by-side.
3. **Junior-share trend plot:** From T08/T09, annotated with AI model release dates (GPT-4: Mar 2023, Claude 3: Mar 2024, GPT-4o: May 2024, Claude 3.5 Sonnet: Jun 2024, o1: Sep 2024, DeepSeek V3: Dec 2024, Claude 3.5 MAX: Feb 2025, GPT-4.5: Feb 2025, Claude 3.6 Sonnet: Apr 2025, Claude 4 Opus: Sep 2025, Claude 4.5 Haiku: Oct 2025, Gemini 2.5 Pro: Mar 2026).
4. **Senior archetype chart:** From T21, management vs orchestration language profiles (2024 vs 2026).
5. **Posting-usage divergence chart:** From T14.

**Output:** `exploration/artifacts/` with each artifact as PNG/PDF + a README

### T26. Exploration synthesis `[Agent M]`

**Goal:** Consolidate everything into a single handoff for the analysis agent.

**Steps (reads all reports):**
1. Read all `exploration/reports/T*.md`
2. Write `exploration/reports/SYNTHESIS.md` covering:
   - Data quality verdict per RQ
   - Recommended analytical samples (rows, columns, filters)
   - Seniority column recommendation
   - Known confounders (description length growth, asaniczka label gap, aggregator contamination, field-wide scope inflation)
   - Preliminary findings for RQ1-RQ4 (direction, magnitude, confidence)
   - Key tensions to resolve (YOE decrease vs scope increase, field-wide vs SWE-specific, within-company vs composition, relabeling evidence, ghost requirement prevalence)
   - Technology evolution summary
   - Metro heterogeneity summary
   - Senior archetype characterization
   - Sensitivity requirements
   - Pipeline issues remaining (if any)

**Output:** `exploration/reports/SYNTHESIS.md` — the one document the analysis agent reads first.

---

## 5. Deferred to analysis plan

These items are valuable but premature without LLM-stage outputs or formal statistical framework:

- Robustness pre-registration / specification curve
- Placebo and falsification tests
- Oaxaca-Blinder decomposition
- Selection bias reweighting / IPSW
- Power analysis for specific effect sizes
- Seniority boundary classifier (analysis-phase, needs embeddings)
- Company fixed-effects regression (uses T13's overlap panel)
- Formal break detection / event-study plots

---

## 6. Bias threat summary

| Bias | Direction | Mitigation task | Residual risk |
|---|---|---|---|
| Platform selection | Favors SWE | T07 | Low for SWE |
| Scraper query design | Misses long-tail | T05 | Moderate |
| Aggregator contamination | Inflates some companies | T06 | Low after flagging |
| Temporal selection (volatility) | Oversamples long-lived | T15 | Moderate |
| Kaggle provenance unknown | Unknown | T05 | High (irreducible) |
| asaniczka missing entry-level | Thin baseline | T02, T03 | Moderate |
| Company composition shift | Could drive seniority shift | T06, T13 | Low after decomposition |
| Boilerplate removal noise | Noisy text analysis | Use `description_core_llm` or `description` | Low-Moderate |
| Description length inflation | Biases raw keyword counts | Length-normalization in all text tasks | Low after normalization |
| Company-name contamination | Pollutes corpus comparisons | Company-name stripping in preamble | Low after stripping |
