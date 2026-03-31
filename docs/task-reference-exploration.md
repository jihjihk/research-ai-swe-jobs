# Exploration Task Reference

Date: 2026-03-31
Input: `preprocessing/intermediate/stage8_final.parquet` (~76 cols, ~1.22M rows)
Schema reference: `docs/preprocessing-schema.md` (canonical), `docs/schema-stage8-and-stage12.md` (historical)

---

## 1. Shared preamble

Prepend this verbatim to every sub-agent prompt.

```
## Exploration task context

You are a sub-agent executing exploration tasks for a SWE labor market research project studying how AI coding agents are restructuring software engineering roles. This is the FIRST systematic analysis of the data — preprocessing is complete but no substantive analysis has been performed yet.

**Your orientation is DISCOVERY, not confirmation.** The project has research questions (RQ1-RQ4 in docs/1-research-design.md) about junior scope inflation, senior archetype shifts, and employer-requirement/worker-usage divergence. These are useful context, but your job is to report WHAT THE DATA SHOWS, not to confirm or deny pre-existing hypotheses. Findings that surprise, contradict, or go beyond existing RQs are especially valuable. Every task report should include a "Surprises & unexpected patterns" section.

**Input data:** `preprocessing/intermediate/stage8_final.parquet` (~76 columns, ~1.22M rows)
Read `docs/preprocessing-schema.md` for column definitions and recommended usage. If that file doesn't exist, fall back to `docs/schema-stage8-and-stage12.md`.

**Key data facts you need to know:**
- Three sources: kaggle_arshkon (118K rows, April 2024), kaggle_asaniczka (1.06M rows, Jan 2024), scraped (27K LinkedIn + 14K Indeed, March 2026)
- SWE sample: ~33K rows total (5K arshkon, 23K asaniczka, 4.5K scraped LinkedIn)
- Entry-level is THIN: 830 arshkon entry, 574 scraped entry, only 130 asaniczka "entry" (asaniczka has zero native entry labels; its entries come from rule-based imputation)
- Seniority is 71.8% unknown across all rows. For SWE specifically: arshkon 81% known, asaniczka 100% known (but almost all mid-senior/associate), scraped 94% known
- Description length grew ~56% from 2024 to 2026 (median 3,316 to 5,181 for SWE). THIS IS A MASSIVE CONFOUND — normalize everything.
- Remote work flags are 0% in 2024 sources (data artifact, not real), ~24% in scraped. Do not interpret as a real change.
- Company overlap between arshkon and scraped is only ~18-25% (406 of 2,300/1,640 unique companies). Composition effects could be large.
- Entry-level YOE paradox: arshkon entry median 3.0 YOE, scraped entry median 2.0 YOE. This CONTRADICTS the scope inflation narrative and needs investigation.
- 31GB RAM limit — use DuckDB or pyarrow for queries, never load full parquet into pandas.

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

**Method comparison philosophy:** Different analytical methods surface different patterns. When a task involves topic discovery, clustering, or similarity analysis, run MULTIPLE methods and compare. Method agreement strengthens findings; disagreements are themselves informative and should be reported. See T09 for the primary methods laboratory.

**Text analysis hygiene — apply to ALL text-based tasks:**

1. **Description text selection.** Use `description_core` as primary text source despite ~44% boilerplate removal accuracy. Fall back to `description` when `description_core` is null or for analyses where boilerplate matters less (e.g., keyword presence). Note which you used.

2. **Company-name stripping.** Before any corpus comparison or term-frequency analysis, build a stoplist from all `company_name_canonical` values, tokenize them into words, and strip them during tokenization. Company names dominate results otherwise.

3. **Length normalization.** Description length grew ~56% from 2024 to 2026. For keyword analyses:
   - Primary: binary indicator (any mention per posting)
   - Secondary: rate per 1,000 characters
   - Always report both. Never report raw counts without normalization.

4. **Artifact filtering.** For term lists:
   - Require terms to appear in >=20 distinct companies
   - Exclude HTML concatenation artifacts (tokens >12 chars with no spaces that aren't real words)
   - Exclude city/location names (check against `metro_area` and `state_normalized` values)

5. **Semantic categorization.** Tag reported terms with categories:
   - `ai_tool`: AI coding tools, LLMs, specific models (copilot, cursor, claude, gpt, llm, rag, agent, mcp)
   - `ai_domain`: ML/AI as a domain (machine learning, deep learning, NLP, computer vision)
   - `tech_stack`: Specific technologies, frameworks, languages
   - `org_scope`: Organizational/scope language (ownership, end-to-end, cross-functional, stakeholder)
   - `mgmt`: Management/leadership (lead, mentor, manage, team, hire, coach)
   - `sys_design`: Systems/architecture (distributed systems, scalability, architecture)
   - `method`: Development methodology (agile, scrum, ci/cd, tdd)
   - `credential`: Formal requirements (years experience, BS/MS/PhD, certification)
   - `soft_skill`: Interpersonal (collaboration, communication, problem-solving)
   - `noise`: Residual after filtering (target <10%)

6. **Within-2024 calibration.** When comparing 2024 vs 2026, also compare arshkon (2024-04) vs asaniczka (2024-01) on the same metric where possible. This establishes baseline cross-source variability. If a 2024-to-2026 change is smaller than within-2024 cross-source variation, flag it as potentially artifactual.

**Output conventions:**
- Figures -> exploration/figures/TASK_ID/ (PNG, 150dpi, max 4 per task)
- Tables -> exploration/tables/TASK_ID/ (CSV)
- Report -> exploration/reports/TASK_ID.md using this template:

  # TASK_ID: [title]
  ## Finding
  [1-3 sentence headline result]
  ## Surprises & unexpected patterns
  [What did you NOT expect? What contradicts prior assumptions? What's new?]
  ## Implication for research
  [What this means for RQ1-RQ4, AND what new questions it raises]
  ## Methodology
  [How was the conclusion derived from the data]
  ## Data quality note
  [Caveats, issues, thin samples, confounds]
  ## Action items
  [What downstream agents or the analysis phase needs to know]

Create exploration/ directories if they don't exist.
Use DuckDB CLI or pyarrow for all data queries. Do NOT load the full parquet into pandas.
```

---

## 2. Agent dispatch blocks

These are prepended (after the preamble) to each agent's prompt, before the task specs.

### Agent A — Wave 1: Data coverage & missingness (T01 + T02)

Audit column coverage and missing data patterns across all sources and the SWE subset. Produce the coverage heatmap and missing data tables that all downstream tasks depend on. Run the native-label comparability audit for whether asaniczka `associate` behaves enough like arshkon `entry` to justify any limited sensitivity use. Execute tasks T01 and T02.

### Agent B — Wave 1: Classifier quality (T03 + T04)

Evaluate seniority label quality and SWE classification accuracy. Cross-tabulate all seniority variants, compute agreement metrics, and test whether the RQ1 junior-share metric changes depending on which seniority column is used. Assess SWE classification via manual sampling of borderline cases. Execute tasks T03 and T04.

### Agent C — Wave 1: Dataset comparability & concentration (T05 + T06)

Test whether the three datasets are measuring the same thing by running pairwise comparisons (description length, company overlap, geographic/seniority/title distributions). Also assess company concentration and whether a few employers dominate findings. Execute tasks T05 and T06.

### Agent D — Wave 1: External benchmarks (T07)

Compare our data against BLS OES occupation/state data and JOLTS information sector trends. This requires web access to download benchmark data from FRED and BLS. Execute task T07.

### Agent E — Wave 2: Distribution profiling & archetype discovery (T08 + T09)

Compute baseline distributions for ALL available variables by period and seniority, with emphasis on anomaly detection and unexpected patterns. Then run unsupervised clustering on SWE posting descriptions to discover natural posting archetypes WITHOUT imposing a priori categories. Read `exploration/reports/INDEX.md` for the seniority recommendation from Wave 1. Execute tasks T08 and T09.

### Agent F — Wave 2: Title evolution & requirements complexity (T10 + T11)

Map how the SWE title taxonomy has evolved between 2024 and 2026 — what titles emerged, disappeared, or changed meaning. Then quantify the structural complexity of job requirements (credential stacking, technology density, scope breadth) and how it changed. Execute tasks T10 and T11.

### Agent G — Wave 2: Open-ended text discovery & linguistic evolution (T12 + T13)

Run open-ended corpus comparisons to discover the terms and phrases that changed MOST between periods, without limiting to pre-defined comparison pairs. Then analyze how the linguistic structure, readability, and tone of job postings evolved. Execute tasks T12 and T13.

### Agent H — Wave 2: Technology ecosystems & semantic landscape (T14 + T15)

Map the technology ecosystem in SWE postings — not just individual tech mentions, but how technologies co-occur and form natural skill bundles. Then compute the full semantic similarity landscape across all period x seniority groups to map how the posting space has restructured. Execute tasks T14 and T15.

### Agent I — Wave 3: Company strategies & geographic structure (T16 + T17)

Among companies appearing in both periods, cluster them by HOW their postings changed (early AI adopters vs traditionalists vs scope inflators). Then analyze geographic market segmentation — which metros lead, which lag, and whether the findings are uniform or concentrated. Execute tasks T16 and T17.

### Agent J — Wave 3: Cross-occupation boundaries & temporal dynamics (T18 + T19)

Compare SWE, SWE-adjacent, and control occupations to determine which observed changes are SWE-specific vs field-wide. Pay special attention to whether SWE-adjacent roles are absorbing SWE-like requirements. Then analyze temporal dynamics within the March 2026 scraped data. Execute tasks T18 and T19.

### Agent K — Wave 3: Seniority boundaries & senior role evolution (T20 + T21)

Measure how sharp the seniority boundaries are and whether they blurred or shifted between periods — this goes beyond testing "relabeling" to mapping the full boundary structure. Then conduct a deep dive into how senior SWE roles specifically evolved. Execute tasks T20 and T21.

### Agent L — Wave 3: Ghost forensics & employer-usage divergence (T22 + T23)

Identify ghost-like and aspirational requirement patterns through text analysis, with emphasis on whether AI requirements are more aspirational than traditional ones. Then compute the employer-requirement vs worker-usage divergence for RQ3. Execute tasks T22 and T23.

### Agent M — Wave 4: Hypothesis generation, artifacts & synthesis (T24 + T25 + T26)

Read ALL reports in `exploration/reports/`. First, generate new hypotheses from the findings — what research questions should we ADD or MODIFY? Then produce the 5 interview elicitation artifacts for RQ4. Finally, write the synthesis document that consolidates all findings into a single handoff for the analysis phase. Execute tasks T24, T25, and T26.

---

## 3. Task specs

### Wave 1 — Data Foundation

---

### T01. Column coverage audit `[Agent A]`

**Goal:** Which columns have enough non-null coverage to be analysis-worthy, by source and SWE subset?

**Steps:**
1. For every column, compute by source (arshkon, asaniczka, scraped) AND by `is_swe` subset: non-null rate, distinct count, top 5 values
2. Produce a coverage heatmap (columns x sources, colored by non-null rate)
3. Flag columns >50% null for any source used in cross-period comparisons
4. Produce a "usable columns per RQ" table — which analyses are constrained by missing data?
5. Check whether `description_core_llm` exists and what its coverage is
6. Note any columns with DIFFERENT semantics across sources (e.g., `company_industry` has compound labels in scraped but single labels in arshkon)

**Output:** `exploration/reports/T01.md` + coverage heatmap PNG + CSV

### T02. Missing data & native-label comparability `[Agent A]`

**Goal:** Document missingness patterns constraining cross-period comparisons, and test whether asaniczka `associate` can be treated as a limited junior proxy.

**Steps:**
1. Field x source x platform missing data table (% non-null), all rows and SWE subset
2. Document which seniority labels each source provides natively
3. SWE-only native-label comparability audit:
   - Compare asaniczka `associate` against arshkon `entry`, `associate`, and `mid-senior`
   - Use exact `title_normalized` overlap, explicit junior/senior title-cue rates, `yoe_extracted`, and downstream `seniority_final` distributions conditional on native label
   - State whether asaniczka `associate` behaves more like junior, lower-mid, mixed, or indeterminate
4. Decision rule: `usable as junior proxy` only if evidence is directionally close to arshkon `entry` on multiple signals
5. Effective sample size per source after excluding nulls, for each cross-period analysis field
6. **Key constraint mapping:** For each planned Wave 2-3 analysis category (text, seniority, geography, company, requirements), what's the binding data constraint?

**Output:** `exploration/reports/T02.md` with tables and a clear verdict

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

**Output:** `exploration/reports/T05.md` with test results, artifact assessment, and calibration table

### T06. Company concentration `[Agent C]`

**Goal:** Check if a few employers dominate and bias findings.

**Steps (SWE):**
1. Per dataset: HHI, top-1/5/10/20 share, Gini of posting counts
2. Companies with >3% SWE postings in any dataset
3. Overlap set (arshkon intersection scraped): compare seniority distributions within-company across periods
4. Company-capped sensitivity: junior share after capping at 10 postings/company
5. **Aggregator analysis:** What fraction of SWE postings are from aggregators per source? Do aggregator postings differ systematically in seniority/length/requirements?
6. **New entrants:** How many companies in 2026 scraped have NO match in 2024 arshkon? What industries/sizes are they?

**Output:** `exploration/reports/T06.md` with concentration table, within-company comparison, aggregator profile

### T07. External benchmarks `[Agent D]`

**Goal:** Compare our data against BLS/JOLTS benchmarks.

**Steps:**
1. Download BLS OES for SOC 15-1252 and 15-1256: state-level employment
2. Pearson r: our state-level SWE counts vs OES, per dataset
3. Industry distribution: our SWE vs OES SWE industry (arshkon + scraped)
4. Download JOLTS information sector from FRED. Compare against our scraper daily counts if date_posted coverage allows.
5. Representativeness summary table
6. **Frame the data:** What population does our sample represent? What can and can't we generalize to?

**Output:** `exploration/reports/T07.md`. Target: r > 0.80 geographic.

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
5. **Junior share trends:** Entry share by period using `seniority_final`. Also compute as share of known-seniority rows only.
6. **What variables show the LARGEST changes between periods?** Rank all available metrics by effect size. This identifies where to look deeper.

**Output:** `exploration/reports/T08.md` with plots, summary stats, anomaly flags, calibration table, and ranked change list

### T09. Posting archetype discovery — methods laboratory `[Agent E]`

**Goal:** Discover natural posting archetypes through unsupervised methods, WITHOUT imposing pre-defined categories. This is also the primary **methods comparison task** — run multiple topic modeling and clustering approaches on the same data and compare what each surfaces. Method agreement strengthens findings; disagreements reveal data structure.

**Steps:**
1. **Sample:** Stratified sample of up to 8,000 SWE LinkedIn postings (balanced across period x seniority where possible). Record exact sample composition. Use the SAME sample for all methods to ensure comparability.
2. **Text preparation:** Clean description text (company names stripped, standard English stopwords removed). Prepare two representations:
   - **Sparse:** TF-IDF matrix (1-2 grams, min_df=5, max_df=0.7, 3,000-5,000 features)
   - **Dense:** sentence-transformers `all-MiniLM-L6-v2` embeddings (first 512 tokens per posting)

3. **Method A — BERTopic (primary):**
   - Run BERTopic with sentence-transformer embeddings, UMAP reduction, HDBSCAN clustering
   - Use `min_topic_size=30` to avoid micro-topics; experiment with 20 and 50 as well
   - Extract c-TF-IDF topic representations
   - Record: number of topics found, topic coherence, outlier/noise percentage
   - Use BERTopic's `topics_over_time` or manual period stratification to track topic evolution
   - Use BERTopic's `visualize_topics()`, `visualize_hierarchy()`, and `visualize_barchart()` — save as static images

4. **Method B — LDA (sklearn or gensim):**
   - Run LDA on the TF-IDF/count-vectorized corpus with k=5,8,12,15 topics
   - Compute coherence scores (gensim `CoherenceModel` with 'c_v' metric if feasible, else use perplexity)
   - Extract top 20 terms per topic

5. **Method C — NMF:**
   - Run NMF on the TF-IDF matrix with k=5,8,12,15 components
   - NMF often produces more interpretable topics than LDA on short-to-medium documents
   - Extract top 20 terms per component

6. **Method D — k-means on embeddings:**
   - Run k-means with k=5,6,7,8 on the sentence-transformer embeddings (UMAP-reduced to 20 dims first)
   - Compute silhouette scores

7. **Method comparison:**
   - **Topic alignment:** For each BERTopic topic, find its closest match in LDA, NMF, and k-means (by top-term overlap or centroid similarity). Which topics are **method-robust** (found by all methods)? Which are method-specific?
   - **Cluster stability:** Run each method 3 times with different random seeds. How stable are the assignments? (Adjusted Rand Index between runs.)
   - **Interpretability ranking:** Subjectively rank which method produces the most interpretable and useful topics for this specific data.
   - **Noise handling:** BERTopic/HDBSCAN identifies outlier documents. What % are outliers? What do they look like? k-means forces everything into a cluster — does this help or hurt?
   - Produce a **methods comparison table:** method, # topics, coherence, stability, noise %, interpretability notes.

8. **Characterization (using best method's clusters):** For each archetype:
   - Top 20 terms (by c-TF-IDF or centroid weight)
   - Seniority distribution (what % of each seniority level falls in this cluster?)
   - Period distribution (what % of each period falls in this cluster?)
   - Average description length, YOE, tech count
   - Give each cluster a descriptive name based on its content, not based on RQ1-RQ4

9. **Temporal dynamics:** How did archetype proportions change from 2024 to 2026? Which grew? Which shrank? Are there archetypes that only exist in one period?

10. **Visualization:** UMAP (2D) of the embeddings, colored by: (a) best-method clusters, (b) period, (c) seniority. Three separate plots. Also produce the same with PCA for comparison — does the visual story change?

11. **Key discovery question:** Do the clusters align with seniority levels (entry/mid/senior map to different clusters)? Or do they align with something else entirely (industry, role type, tech stack, company size)? The answer reveals the dominant structure of the market.

**Output:** `exploration/reports/T09.md` + methods comparison table + cluster plots + cluster characterization CSV + method recommendation for downstream use

### T10. Title taxonomy evolution `[Agent F]`

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

**Output:** `exploration/reports/T10.md` + title evolution tables + new/disappeared title lists

### T11. Requirements complexity & credential stacking `[Agent F]`

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
6. **Outlier analysis:** What do the most complex postings look like? (Top 1% by requirement_breadth.) Are they real or template-bloated?

**Output:** `exploration/reports/T11.md` + complexity distribution plots + per-seniority comparison tables

### T12. Open-ended text evolution `[Agent G]`

**Goal:** Discover the terms and phrases that changed MOST between periods, using open-ended comparison rather than pre-defined pairs.

**Steps (SWE, LinkedIn-only):**
1. **Text cleaning first (critical):**
   - Build company-name stoplist from all `company_name_canonical` values
   - Use `description_core` if available, else `description`
   - Standard English stopwords + company stopwords
2. **Primary comparison: 2024 (arshkon) vs 2026 (scraped), ALL SWE combined.**
   - Fightin' Words (Monroe et al.) or log-odds ratio with informative Dirichlet prior
   - Top 100 distinguishing terms in EACH direction (2024-heavy and 2026-heavy)
   - Tag every term with semantic category from preamble taxonomy
   - Produce category-level summary: what % of distinguishing terms are ai_tool, org_scope, tech_stack, etc.?
3. **Emerging terms** (>1% in 2026, <0.1% in 2024): artifact-filtered, categorized
4. **Accelerating terms** (existed in 2024 but grew >3x): these may be more informative than binary emerging/disappearing
5. **Disappearing terms** (>1% in 2024, <0.1% in 2026): separate true vocabulary change from source artifacts
6. **Secondary comparisons (if sample sizes allow):**
   - Entry 2024 vs Entry 2026 (n: 830 vs 574)
   - Mid-senior 2024 vs Mid-senior 2026 (n: 3,003 vs 3,459)
   - Entry 2026 vs Mid-senior 2024 (the relabeling diagnostic)
   - Within-2024: arshkon mid-senior vs asaniczka mid-senior (the calibration comparison — how much change is "normal"?)
7. **Bigram analysis:** Repeat top findings for bigrams. Phrase-level changes (e.g., "prompt engineering", "AI agent", "code review") are often more informative than unigrams.
8. **BERTopic cross-validation:** If T09's BERTopic model is available (saved to disk), load it and examine which topics are most period-specific. Alternatively, fit a fresh BERTopic on the combined corpus and use `topics_per_class` with period as the class variable. This provides a complementary view to Fightin' Words: BERTopic surfaces thematic shifts, while Fightin' Words surfaces individual term shifts. Report where they agree and disagree.
9. **Report n per corpus for every comparison.** Flag any with n < 100.

**Output:** `exploration/reports/T12.md` + categorized term tables (CSV) + category summary figure

### T13. Linguistic & structural evolution `[Agent G]`

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

**Output:** `exploration/reports/T13.md` + readability comparison table + stacked section chart + tone metrics

### T14. Technology ecosystem mapping `[Agent H]`

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
6. **AI integration pattern:** Among postings mentioning AI tools/LLM, what traditional technologies co-occur? (Is AI adding to existing stacks, or replacing components?)

**Output:** `exploration/reports/T14.md` + tech heatmap + co-occurrence network visualization + community comparison

### T15. Semantic similarity landscape — multi-representation comparison `[Agent H]`

**Goal:** Map the full semantic structure of the SWE posting space and how it changed between periods. Also compare whether different text representations (embeddings vs TF-IDF) and dimensionality reduction methods (UMAP vs PCA vs t-SNE) tell the same story.

**Steps:**
1. **Sample and embed:** Stratified sample of up to 2,000 SWE postings per period x seniority_3level group (up to ~12,000 total). Use first 512 tokens of cleaned description text. Build TWO representations:
   - **Dense:** sentence-transformers `all-MiniLM-L6-v2` embeddings
   - **Sparse:** TF-IDF vectors (same parameters as T09), reduced via SVD to 100 components

2. **Full group similarity matrix (both representations).** Define groups as period x seniority_3level. Compute average cosine similarity between all group centroids. Present as two side-by-side heatmaps (embedding-based vs TF-IDF-based). **Do they agree on which groups are most/least similar?**

3. **Within-group dispersion.** For each group, compute average pairwise cosine similarity within the group. Are postings within each group becoming more or less homogeneous over time? Compare dispersion under both representations.

4. **The convergence question (but broader):** Is the junior-senior gap narrowing? Is the junior-mid gap narrowing? Is senior becoming more homogeneous or more diverse? What's happening to the "mid" level?

5. **Visualization — dimensionality reduction comparison.** On the embedding representation, produce 2D projections using THREE methods:
   - **UMAP** (n_neighbors=15, min_dist=0.1) — preserves local cluster structure
   - **PCA** — preserves global variance structure
   - **t-SNE** (perplexity=30) — preserves local neighborhoods
   - All three colored by period x seniority. **Does the visual separation story change across methods?** UMAP typically shows tighter clusters; PCA shows global relationships; t-SNE is intermediate. Report which method reveals the most useful structure for this data.

6. **Nearest-neighbor analysis.** For each 2026 entry posting, find its 5 nearest neighbors in 2024 (using embeddings). What seniority are they? If 2026 entry postings are nearest to 2024 mid-senior, that's strong convergence evidence. Repeat with TF-IDF representation — does the finding hold?

7. **Representation comparison summary:** Produce a table: for each key finding (convergence, dispersion change, cluster separation), does it hold under both embedding and TF-IDF representations? Findings that are representation-robust are more trustworthy.

8. **Outlier identification.** Which postings are most UNLIKE their seniority peers? What makes them different?

**Output:** `exploration/reports/T15.md` + dual similarity heatmaps + UMAP/PCA/t-SNE comparison plots + representation robustness table + nearest-neighbor analysis

---

### Wave 3 — Market Dynamics & Cross-cutting Patterns

Wave 3 builds on Wave 2's discoveries to examine market structure, actors, and boundaries.

---

### T16. Company hiring strategy typology `[Agent I]`

**Goal:** Among companies appearing in both periods, discover different hiring strategy trajectories — how are different companies changing?

**Steps (SWE, LinkedIn-only):**
1. **Overlap panel:** Identify companies with >=3 SWE postings in BOTH arshkon and scraped (~406 overlap companies, filter to those with enough postings).
2. **Per-company change metrics:** For each overlap company, compute 2024-to-2026 change in:
   - Entry share (of known seniority)
   - AI keyword prevalence (binary per posting)
   - Mean description length
   - Mean tech count
   - Mean org_scope term count
3. **Cluster companies by their change profile.** k-means on the change vectors. Are there distinct strategies? Name them (e.g., "AI-forward", "traditional hold", "scope inflator", "downsizer").
4. **Within-company vs between-company decomposition:** For entry share, AI prevalence, and description length: how much of the aggregate 2024-to-2026 change is driven by within-company change vs different companies entering/exiting the sample?
5. **New market entrants:** Profile companies in 2026 with no 2024 match. What industries? What do their postings look like compared to established companies?
6. **Aggregator vs direct employer:** Compare change patterns. Are aggregators showing different trends?

**Output:** `exploration/reports/T16.md` + company cluster characterization + decomposition results

### T17. Geographic market structure `[Agent I]`

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
6. **Metro heatmap:** metros x metrics, colored by change magnitude.

**Output:** `exploration/reports/T17.md` + metro heatmap + correlation analysis

### T18. Cross-occupation boundary analysis `[Agent J]`

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

**Output:** `exploration/reports/T18.md` + parallel trends plots + boundary similarity analysis

### T19. Temporal dynamics & market velocity `[Agent J]`

**Goal:** Analyze within-period temporal patterns, especially within the March 2026 scraped data.

**Steps:**
1. **Daily volume patterns** in scraped data: posting counts by `scrape_date` (or `date_posted` where available). Day-of-week effects?
2. **Posting age analysis:** Distribution of `posting_age_days` for scraped data. How old are the postings we're observing? Are there clusters (fresh postings vs long-lived)?
3. **Content stability within March 2026:** Split scraped data into early-March vs late-March. Are the text patterns stable, or is there within-month drift?
4. **Within-2024 temporal variation:** If arshkon has `date_posted` variability, is there within-source temporal variation?
5. **Repost detection:** Using `preprocessing/intermediate/stage1_observations.parquet` (if accessible), how many postings appear on multiple dates? What's the repost rate?
6. **Velocity indicators:** Are certain types of postings (entry vs senior, AI-mentioning vs not) posted/removed faster? Use posting age as a proxy for demand.

**Output:** `exploration/reports/T19.md` + temporal plots

### T20. Seniority boundary clarity `[Agent K]`

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
6. **Full similarity matrix** using the structured features (not text): compute average feature profiles per seniority x period and present as a heatmap.

**Output:** `exploration/reports/T20.md` + AUC comparison + feature importance analysis + boundary heatmap

### T21. Senior role evolution deep dive `[Agent K]`

**Goal:** Go deep on how senior SWE roles specifically are evolving — not just management-to-orchestration, but the full picture.

**Steps (SWE, LinkedIn-only, seniority_final IN ('mid-senior', 'director')):**
1. **Language profiles.** Define three profiles (not just two):
   - **People management:** manage, mentor, coach, hire, interview, grow, develop talent, performance review, career development, 1:1, headcount, people management, team building, direct reports
   - **Technical orchestration:** architecture review, code review, system design, technical direction, AI orchestration, agent, workflow, pipeline, automation, evaluate, validate, quality gate, guardrails, prompt engineering, tool selection
   - **Strategic scope:** stakeholder, business impact, revenue, product strategy, roadmap, prioritization, resource allocation, budgeting, cross-functional alignment
2. **Per posting:** Compute density (mentions per 1K chars) for each profile.
3. **2D and 3D scatter:** Management vs Orchestration vs Strategic, colored by period. How did the distribution shift?
4. **Senior sub-archetypes:** Cluster senior postings by their language profiles. Are there distinct types (people-manager, tech-lead, architect, strategist)? How did their proportions change?
5. **AI interaction:** Among senior postings mentioning AI, how does the management/orchestration/strategic balance differ from non-AI-mentioning senior postings?
6. **Director specifically:** Directors are a small but important group (22 arshkon, 55 scraped). What do their postings look like? How do they differ from mid-senior?
7. **The "new senior" question:** Is there an emergent senior archetype that didn't exist in 2024? What does it look like?

**Output:** `exploration/reports/T21.md` + management-orchestration-strategic charts + senior sub-archetype analysis

### T22. Ghost & aspirational requirements forensics `[Agent L]`

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

**Output:** `exploration/reports/T22.md` + ghost prevalence tables + examples

### T23. Employer-requirement / worker-usage divergence `[Agent L]`

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
6. **Produce divergence chart** showing requirement rate vs usage benchmarks, with appropriate uncertainty bands.

**Output:** `exploration/reports/T23.md` + divergence chart

---

### Wave 4 — Integration & Hypothesis Generation

---

### T24. Hypothesis generation from findings `[Agent M]`

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

### T25. Interview elicitation artifacts `[Agent M]`

**Goal:** Produce 5 artifacts for RQ4 data-prompted interviews, drawing on the full body of exploration findings.

**Steps (reads all prior reports):**
1. **Inflated junior JDs:** From T11/T22, select 3-5 entry-level postings with the most extreme scope-inflated or ghost-like requirements. Query parquet for actual text.
2. **Paired JDs over time:** From T16, select 3-5 same-company pairs (2024 vs 2026). Format side-by-side.
3. **Junior-share trend plot:** From T08, annotated with AI model release dates (GPT-4: Mar 2023, Claude 3: Mar 2024, GPT-4o: May 2024, Claude 3.5 Sonnet: Jun 2024, o1: Sep 2024, DeepSeek V3: Dec 2024, Claude 3.5 MAX: Feb 2025, GPT-4.5: Feb 2025, Claude 3.6 Sonnet: Apr 2025, Claude 4 Opus: Sep 2025, Claude 4.5 Haiku: Oct 2025, Gemini 2.5 Pro: Mar 2026).
4. **Senior archetype chart:** From T21, management vs orchestration vs strategic language profiles (2024 vs 2026).
5. **Posting-usage divergence chart:** From T23.
6. **Bonus: Any particularly striking discovery from T24** that would be valuable to present to interviewees for reaction.

**Output:** `exploration/artifacts/` with each artifact as PNG + a README

### T26. Exploration synthesis `[Agent M]`

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

## 4. Deferred to analysis plan

These items require formal statistical framework, LLM-stage outputs, or analysis-phase infrastructure:

- Robustness pre-registration / specification curve
- Placebo and falsification tests
- Oaxaca-Blinder decomposition
- Selection bias reweighting / IPSW
- Power analysis for specific effect sizes
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
